//! 行列演算と要素ごとの演算カーネル（REQ-41・REQ-32・TASK-41.7-2・#416）。
//!
//! # 役割
//!
//! `docs/design/qwen2-scoring-onnx-ops.md` 3 節で担当 #416 とされた 8 演算
//! （MatMul・Gemm・Add・Sub・Mul・Sqrt・Reciprocal・Cast）だけを実装する。
//! 共有型とブロードキャストは親モジュールのものを使い、再実装しない。
//!
//! # 契約
//!
//! - 実行器へは未接続。許可リスト照合（演算×属性×dtype×rank）は別の担当（#581・#420）。
//!   opset 13 より意図して狭めた: Cast は `to=FLOAT` のみ（入力は BFLOAT16 と INT64）、
//!   MatMul は rank 2〜[`MAX_RANK`]（1 次元の昇格なし）、Gemm は transA=0・transB=1・alpha=1・beta=1 固定
//! - 形状・要素数は checked 演算で求め、`MAX_ELEMENTS` と照合してから確保する（REQ-39）。panic しない
//! - 縮約（MatMul・Gemm）の加算順は固定で、スレッド数・分配順に依存しない（REQ-28）。MatMul は k 昇順、
//!   Gemm は lane 別（k mod 4）に k 昇順で累積してから lane 順に足す（[`gemm_tile`]）。並列化は出力の
//!   行・ブロック単位で、1 つの出力要素の加算を複数スレッドに分けない（#629）
//! - Sqrt・Reciprocal は IEEE どおり（負の Sqrt は NaN、0 の Reciprocal は inf）。拒否せず伝播させ、
//!   確率を作る Softmax 系が [`OpError::NonFinite`] で fail-closed にする
//! - 許容差: f32 の演算に 1e-9 は適用しない（1e-9 は f64 計算の場合）。テストは 2 進で正確な値の
//!   厳密一致、または IEEE の単一演算とのビット一致で比べる

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

use super::{
    MAX_RANK, OpError, Tensor, advance_index, broadcast_map2, broadcast_shape, broadcast_strides,
    element_count,
};

/// BFLOAT16（u16 のビット列）を f32 へ。上位 16 bit へ置くだけで正確（丸めなし）。
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

/// Add（FLOAT・多方向ブロードキャスト）。
pub fn add(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, OpError> {
    broadcast_map2(a, b, |x, y| Ok(x + y))
}

/// Sub（FLOAT・多方向ブロードキャスト）。
pub fn sub(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, OpError> {
    broadcast_map2(a, b, |x, y| Ok(x - y))
}

/// Mul（FLOAT・多方向ブロードキャスト）。
pub fn mul(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, OpError> {
    broadcast_map2(a, b, |x, y| Ok(x * y))
}

/// Add（INT64。位置計算用）。あふれは [`OpError::IntegerOverflow`]。
pub fn add_i64(a: &Tensor<i64>, b: &Tensor<i64>) -> Result<Tensor<i64>, OpError> {
    broadcast_map2(a, b, |x, y| {
        x.checked_add(y).ok_or(OpError::IntegerOverflow)
    })
}

/// 要素ごとの単項演算。
pub(crate) fn map1(x: &Tensor<f32>, f: impl Fn(f32) -> f32) -> Result<Tensor<f32>, OpError> {
    Tensor::new(x.shape().to_vec(), x.data().iter().map(|&v| f(v)).collect())
}

/// Sqrt（FLOAT）。負は NaN。
pub fn sqrt(x: &Tensor<f32>) -> Result<Tensor<f32>, OpError> {
    map1(x, f32::sqrt)
}

/// Reciprocal（FLOAT）。0 は inf。
pub fn reciprocal(x: &Tensor<f32>) -> Result<Tensor<f32>, OpError> {
    map1(x, |v| 1.0 / v)
}

/// Cast（`to=FLOAT`）。入力は BFLOAT16（u16 のビット列）。
pub fn cast_bf16_to_f32(x: &Tensor<u16>) -> Result<Tensor<f32>, OpError> {
    Tensor::new(
        x.shape().to_vec(),
        x.data().iter().map(|&b| bf16_to_f32(b)).collect(),
    )
}

/// Cast（`to=FLOAT`）。入力は INT64。2^24 を超える値は最近接偶数へ丸める（`as f32`）。
pub fn cast_i64_to_f32(x: &Tensor<i64>) -> Result<Tensor<f32>, OpError> {
    Tensor::new(
        x.shape().to_vec(),
        x.data().iter().map(|&v| v as f32).collect(),
    )
}

/// 末尾 2 次元 `[a, b]` を取り出す。
fn last_two(s: &[usize]) -> Result<(usize, usize), OpError> {
    match s {
        [.., a, b] => Ok((*a, *b)),
        _ => Err(OpError::ShapeMismatch),
    }
}

/// これ未満の積和回数（M×N×K）ならスレッドを起こさない（起動コストが計算を上回るため）。
const PARALLEL_MIN_MACS: usize = 1 << 22;

/// 使うスレッド数。`available_parallelism` を上限にし、仕事の単位数と積和回数で絞る（無制限に生成しない。REQ-39）。
/// 呼び出し側がこの演算を並行に走らせると、スレッド数は並行数との掛け算になる。
fn worker_count(macs: usize, items: usize) -> usize {
    if macs < PARALLEL_MIN_MACS {
        return 1;
    }
    std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1)
        .min(items)
        .max(1)
}

/// `worker` を `threads` 本で走らせ、最初のエラーを返す。各 worker は `stop` を見て早期に抜ける。
/// スレッドの panic（本モジュールは panic しない設計だが念のため）は計算不能として
/// [`OpError::LimitExceeded`] に写し、結果を返さない（fail-closed）。
fn run_workers(
    threads: usize,
    worker: impl Fn(&AtomicBool) -> Result<(), OpError> + Sync,
) -> Result<(), OpError> {
    let stop = AtomicBool::new(false);
    let first_err: Mutex<Option<OpError>> = Mutex::new(None);
    let run = || {
        if let Err(e) = worker(&stop) {
            stop.store(true, Ordering::Relaxed);
            if let Ok(mut slot) = first_err.lock()
                && slot.is_none()
            {
                *slot = Some(e);
            }
        }
    };
    let joined = std::thread::scope(|sc| {
        // 生成に失敗したら生成済みの分（最悪は呼び出しスレッドのみ）で続ける。仕事は動的に配るので結果は同じ
        let handles: Vec<_> = (1..threads)
            .map_while(|_| std::thread::Builder::new().spawn_scoped(sc, run).ok())
            .collect();
        run();
        // 短絡せず全件 join する（panic したスレッドがあっても残りを待つ）
        handles
            .into_iter()
            .fold(true, |ok, h| h.join().is_ok() && ok)
    });
    if let Some(e) = first_err.into_inner().ok().flatten() {
        return Err(e);
    }
    if !joined {
        return Err(OpError::LimitExceeded);
    }
    Ok(())
}

/// MatMul（FLOAT）。両入力 rank 2〜[`MAX_RANK`]。先頭（バッチ）次元は numpy 方式でブロードキャストする。
///
/// 各出力要素は k の昇順の f32 加算（i-k-j のループ順でも要素ごとの加算順は変わらない）。
/// 出力の行（バッチ × m）を連続した範囲でスレッドに静的に割り当てる。1 つの出力要素の加算を
/// 複数スレッドに分けないので、結果はスレッド数に依存しない（REQ-28）。期限検査は出力行ごと、
/// k × n が大きいときは行の中でも k の 64 ステップごと（REQ-39）。
pub fn matmul(
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    until: Option<Instant>,
) -> Result<Tensor<f32>, OpError> {
    matmul_with(None, a, b, until)
}

/// [`matmul`] の本体。`threads` が `Some` ならそのスレッド数（行数で頭打ち）、`None` なら [`worker_count`]。
/// 結果がスレッド数に依存しないことをテストで照合するために分けてある（REQ-28）。
fn matmul_with(
    threads: Option<usize>,
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    until: Option<Instant>,
) -> Result<Tensor<f32>, OpError> {
    let (sa, sb) = (a.shape(), b.shape());
    if !(2..=MAX_RANK).contains(&sa.len()) || !(2..=MAX_RANK).contains(&sb.len()) {
        return Err(OpError::ShapeMismatch);
    }
    let (m, k) = last_two(sa)?;
    let (k2, n) = last_two(sb)?;
    if k != k2 {
        return Err(OpError::ShapeMismatch);
    }
    let ba = sa.get(..sa.len() - 2).ok_or(OpError::ShapeMismatch)?;
    let bb = sb.get(..sb.len() - 2).ok_or(OpError::ShapeMismatch)?;
    let batch = broadcast_shape(ba, bb)?;
    let mut out_shape = batch.clone();
    out_shape.extend([m, n]);
    if out_shape.len() > MAX_RANK {
        return Err(OpError::ShapeMismatch);
    }
    let total = element_count(&out_shape)?;
    // 空出力は計算不要。0 次元の隣に巨大次元があると m * k 等があふれるため、ここで返す（REQ-39）
    if total == 0 {
        return Tensor::new(out_shape, Vec::new());
    }
    let nbatch = element_count(&batch)?;
    // 各入力は Tensor::new で上限検証済み、出力も上限内なので積はあふれない
    let (a_mat, b_mat) = (m * k, k * n);
    let rank = batch.len();
    let ra: Vec<usize> = broadcast_strides(ba, rank)
        .iter()
        .map(|s| s * a_mat)
        .collect();
    let rb: Vec<usize> = broadcast_strides(bb, rank)
        .iter()
        .map(|s| s * b_mat)
        .collect();
    // バッチごとの A・B の先頭オフセット（出力行 → バッチの対応をスレッドから引くため先に求める）
    let mut offsets = Vec::with_capacity(nbatch);
    let mut idx = vec![0usize; rank];
    for _ in 0..nbatch {
        let (mut oa, mut ob) = (0usize, 0usize);
        for (d, &i) in idx.iter().enumerate() {
            oa += i * ra.get(d).copied().unwrap_or(0);
            ob += i * rb.get(d).copied().unwrap_or(0);
        }
        offsets.push((oa, ob));
        advance_index(&mut idx, &batch);
    }
    let rows = nbatch * m;
    let threads = threads.map_or_else(
        || worker_count(total.saturating_mul(k), rows),
        |t| t.min(rows).max(1),
    );
    let rows_per_thread = rows.div_ceil(threads);
    let mut out = vec![0.0f32; total];
    let ranges: Mutex<Vec<(usize, &mut [f32])>> = Mutex::new(
        out.chunks_mut(rows_per_thread * n)
            .enumerate()
            .map(|(t, c)| (t * rows_per_thread, c))
            .collect(),
    );
    run_workers(threads, |stop| {
        loop {
            let Some((row0, chunk)) = ranges.lock().ok().and_then(|mut v| v.pop()) else {
                return Ok(());
            };
            for (r, orow) in chunk.chunks_mut(n).enumerate() {
                if stop.load(Ordering::Relaxed) {
                    return Ok(());
                }
                check_deadline(until)?;
                let (bi, mi) = ((row0 + r) / m, (row0 + r) % m);
                let (oa, ob) = offsets.get(bi).copied().ok_or(OpError::ShapeMismatch)?;
                let arow = a
                    .data()
                    .get(oa + mi * k..oa + (mi + 1) * k)
                    .ok_or(OpError::ShapeMismatch)?;
                let bm = b.data().get(ob..ob + b_mat).ok_or(OpError::ShapeMismatch)?;
                for (kk, (&av, brow)) in arow.iter().zip(bm.chunks_exact(n)).enumerate() {
                    if kk % 64 == 0 && b_mat >= MATMUL_ROW_CHECK_MIN {
                        check_deadline(until)?;
                    }
                    for (o, &bv) in orow.iter_mut().zip(brow) {
                        *o += av * bv;
                    }
                }
            }
        }
    })?;
    Tensor::new(out_shape, out)
}

/// k × n がこれ以上なら MatMul は行の中でも期限を検査する（1 行が長くなる形状向け）。
const MATMUL_ROW_CHECK_MIN: usize = 1 << 16;

/// 期限を過ぎていれば [`OpError::DeadlineExceeded`]。積和ループが行またはブロックごとに呼ぶ。
fn check_deadline(until: Option<Instant>) -> Result<(), OpError> {
    match until {
        Some(t) if Instant::now() > t => Err(OpError::DeadlineExceeded),
        _ => Ok(()),
    }
}

/// Gemm のタイル: A の行数。
const GEMM_MR: usize = 4;
/// Gemm のタイル: W の行数。
const GEMM_NR: usize = 4;
/// Gemm の lane 数（k を `GEMM_LANES` ごとに分けて独立に累積する。NEON 128 bit = f32 × 4）。
const GEMM_LANES: usize = 4;
/// 1 つの仕事の単位が受け持つ W の行数（[`GEMM_NR`] の倍数。スレッドへ動的に配る粒度）。
const GEMM_NB: usize = 16;
/// W のブロックを使い回す A の行数（この行数分の A を L2 に留めたまま W のブロックを再走査する）。
const GEMM_M_CHUNK: usize = 32;
// [`gemm_tile`] の zip は A・W の 4 行ずつを明示的に並べているので、タイル幅を変えるならそこも直す
const _: () = assert!(GEMM_MR == 4 && GEMM_NR == 4);

/// `s`（長さ < GEMM_LANES。k の端数）を配列の先頭へ。残りの lane は 0。
///
/// 不変条件: [`gemm_tile`] が全行の長さ == k を検証したうえで `as_chunks` の余りを渡すので、
/// 長さは常に `k % GEMM_LANES`。可変長のコピーは memcpy 呼び出しになり積和の自動ベクトル化を
/// 妨げるので、本体ループとは分けて端数の 1 ステップだけで使う。
#[inline(always)]
fn gemm_load_tail(s: &[f32]) -> [f32; GEMM_LANES] {
    let mut v = [0.0; GEMM_LANES];
    for (d, &x) in v.iter_mut().zip(s) {
        *d = x;
    }
    v
}

/// タイルの 1 ステップ（lane 幅ぶんの k）: ロード済みの A・W の lane 配列を `acc` に積む。
///
/// 添字は定数範囲（`GEMM_MR`・`GEMM_NR`・`GEMM_LANES`）で範囲外にならない。イテレータの zip で書くと
/// `acc` がレジスタに乗らず（要素への参照を経由するため）約 4 倍遅くなったので、添字で書く。
#[inline(always)]
#[allow(clippy::needless_range_loop)]
fn gemm_fma(
    acc: &mut [[[f32; GEMM_LANES]; GEMM_NR]; GEMM_MR],
    av: &[[f32; GEMM_LANES]; GEMM_MR],
    wv: &[[f32; GEMM_LANES]; GEMM_NR],
) {
    for i in 0..GEMM_MR {
        for j in 0..GEMM_NR {
            for l in 0..GEMM_LANES {
                acc[i][j][l] += av[i][l] * wv[j][l];
            }
        }
    }
}

/// [`GEMM_MR`] 行 × [`GEMM_NR`] 行のタイルの内積（各行の長さは `k`）。
///
/// 加算順（固定・スレッド数に依存しない。REQ-28）: 出力要素ごとに k を `k mod GEMM_LANES` で
/// lane に分け、lane 内は k 昇順で f32 累積し、最後に lane 0 から順に足す。
#[inline(always)]
fn gemm_tile(
    a: [&[f32]; GEMM_MR],
    w: [&[f32]; GEMM_NR],
    k: usize,
) -> Result<[[f32; GEMM_NR]; GEMM_MR], OpError> {
    // 以降のロードの不変条件（全行の長さ == k）をここで 1 度だけ検証する
    if a.iter().chain(&w).any(|r| r.len() != k) {
        return Err(OpError::ShapeMismatch);
    }
    let mut acc = [[[0f32; GEMM_LANES]; GEMM_NR]; GEMM_MR];
    // 各行を lane 幅の固定長チャンクと端数に分け、8 本を zip で同時に進める。長さは上で全行 == k と
    // 検証済みなので zip の打ち切りは起きない。ロードに Option・境界検査を挟むと（#629 のレビュー後の
    // 版で）積和が約 2.4 倍遅くなったため、本体ループは分岐を持たない形にする
    let (ac, at): ([&[[f32; GEMM_LANES]]; GEMM_MR], [&[f32]; GEMM_MR]) = {
        let c = a.map(|r| r.as_chunks::<GEMM_LANES>());
        (c.map(|(h, _)| h), c.map(|(_, t)| t))
    };
    let (wc, wt): ([&[[f32; GEMM_LANES]]; GEMM_NR], [&[f32]; GEMM_NR]) = {
        let c = w.map(|r| r.as_chunks::<GEMM_LANES>());
        (c.map(|(h, _)| h), c.map(|(_, t)| t))
    };
    let steps = ac[0]
        .iter()
        .zip(ac[1])
        .zip(ac[2])
        .zip(ac[3])
        .zip(wc[0])
        .zip(wc[1])
        .zip(wc[2])
        .zip(wc[3]);
    for (((((((a0, a1), a2), a3), w0), w1), w2), w3) in steps {
        gemm_fma(&mut acc, &[*a0, *a1, *a2, *a3], &[*w0, *w1, *w2, *w3]);
    }
    if !at[0].is_empty() {
        let av = at.map(gemm_load_tail);
        let wv = wt.map(gemm_load_tail);
        gemm_fma(&mut acc, &av, &wv);
    }
    let mut out = [[0f32; GEMM_NR]; GEMM_MR];
    for (oi, acc_i) in out.iter_mut().zip(&acc) {
        for (o, lanes) in oi.iter_mut().zip(acc_i) {
            *o = lanes.iter().fold(0.0, |s, &v| s + v);
        }
    }
    Ok(out)
}

/// Gemm の仕事の単位: W の行 `n0..n1`（高々 [`GEMM_NB`] 行）× A の全行。結果は `[m][n1-n0]` の行優先。
///
/// W のブロック（[`GEMM_NR`] 行）を `scratch` に f32 で 1 度だけ展開し、[`GEMM_M_CHUNK`] 行分の A と
/// 突き合わせる（bf16→f32 の展開を A の行で共用し、W の再走査を m のチャンク単位にする）。
#[allow(clippy::too_many_arguments)]
fn gemm_item<W: Copy>(
    a: &[f32],
    m: usize,
    k: usize,
    w: &[W],
    c: Option<&[f32]>,
    n0: usize,
    n1: usize,
    to_f32: &impl Fn(W) -> f32,
    scratch: &mut Vec<f32>,
    until: Option<Instant>,
) -> Result<Vec<f32>, OpError> {
    let len = n1 - n0;
    let mut res = vec![0.0f32; m * len];
    scratch.clear();
    scratch.resize(GEMM_NR * k, 0.0);
    for mc in (0..m).step_by(GEMM_M_CHUNK) {
        check_deadline(until)?;
        let mend = (mc + GEMM_M_CHUNK).min(m);
        for nb in (n0..n1).step_by(GEMM_NR) {
            let nend = (nb + GEMM_NR).min(n1);
            // タイルの端で W の行が足りない分は最後の行を重ねて計算し、書き戻さない（分岐を増やさない）
            for j in 0..GEMM_NR {
                let ni = (nb + j).min(nend - 1);
                let src = w.get(ni * k..(ni + 1) * k).ok_or(OpError::ShapeMismatch)?;
                let dst = scratch
                    .get_mut(j * k..(j + 1) * k)
                    .ok_or(OpError::ShapeMismatch)?;
                for (d, &s) in dst.iter_mut().zip(src) {
                    *d = to_f32(s);
                }
            }
            let mut wr: [&[f32]; GEMM_NR] = [&[]; GEMM_NR];
            for (j, r) in wr.iter_mut().enumerate() {
                *r = scratch
                    .get(j * k..(j + 1) * k)
                    .ok_or(OpError::ShapeMismatch)?;
            }
            for mb in (mc..mend).step_by(GEMM_MR) {
                let mut ar: [&[f32]; GEMM_MR] = [&[]; GEMM_MR];
                for (i, r) in ar.iter_mut().enumerate() {
                    let mi = (mb + i).min(mend - 1);
                    *r = a.get(mi * k..(mi + 1) * k).ok_or(OpError::ShapeMismatch)?;
                }
                let t = gemm_tile(ar, wr, k)?;
                for (i, ti) in t.iter().enumerate().take(mend - mb) {
                    for (j, &v) in ti.iter().enumerate().take(nend - nb) {
                        // bias 無しのときは何も足さず、タイルの値をそのまま書く
                        let v = match c {
                            Some(c) => v + c.get(nb + j).copied().ok_or(OpError::ShapeMismatch)?,
                            None => v,
                        };
                        *res.get_mut((mb + i) * len + (nb - n0) + j)
                            .ok_or(OpError::ShapeMismatch)? = v;
                    }
                }
            }
        }
    }
    Ok(res)
}

/// Gemm（transA=0・transB=1・alpha=1・beta=1 固定）。`A[M,K] × W[N,K]ᵀ (+ C[N])`。
///
/// 重みは元の型 `W` のまま受け、`to_f32` でブロックごとに展開する（全体を f32 へ複製しない。
/// bf16 なら [`bf16_to_f32`]）。`out[m][n] = Σ_k a[m,k]*w[n,k]` を求めてから `c[n]` を足す。
///
/// 加算順は [`gemm_tile`] の固定順（lane 別に k 昇順 → lane 順）で、スレッド数・分配順に依存しない
/// （REQ-28。k 昇順の逐次加算とは丸めが異なりうるが決定的）。W の行を [`GEMM_NB`] 行ずつの単位に
/// 切ってスレッドへ動的に配り、1 つの出力要素は 1 つのスレッドだけが計算する。
/// 期限検査は単位ごと、および単位の中の A の [`GEMM_M_CHUNK`] 行ごと（REQ-39。1 検査あたり高々
/// 16 × 32 × K 積和。Qwen2 1.5B の実形状〔K ≤ 8960〕では ms 程度で、K が極端に大きい形状では伸びる）。
pub fn gemm_trans_b<W: Copy + Sync>(
    a: &Tensor<f32>,
    w: &Tensor<W>,
    c: Option<&Tensor<f32>>,
    to_f32: impl Fn(W) -> f32 + Sync,
    until: Option<Instant>,
) -> Result<Tensor<f32>, OpError> {
    gemm_trans_b_with(None, a, w, c, to_f32, until)
}

/// [`gemm_trans_b`] の本体。`threads` が `Some` ならそのスレッド数（仕事の単位数で頭打ち）、`None` なら
/// [`worker_count`]。結果がスレッド数に依存しないことをテストで照合するために分けてある（REQ-28）。
fn gemm_trans_b_with<W: Copy + Sync>(
    threads: Option<usize>,
    a: &Tensor<f32>,
    w: &Tensor<W>,
    c: Option<&Tensor<f32>>,
    to_f32: impl Fn(W) -> f32 + Sync,
    until: Option<Instant>,
) -> Result<Tensor<f32>, OpError> {
    let (m, k) = match a.shape() {
        [m, k] => (*m, *k),
        _ => return Err(OpError::ShapeMismatch),
    };
    let (n, k2) = match w.shape() {
        [n, k2] => (*n, *k2),
        _ => return Err(OpError::ShapeMismatch),
    };
    if k != k2 {
        return Err(OpError::ShapeMismatch);
    }
    if let Some(c) = c
        && c.shape() != [n]
    {
        return Err(OpError::ShapeMismatch);
    }
    let total = element_count(&[m, n])?;
    // 空出力は行ループを回さない（m が巨大でも n=0 なら無限に近い反復になるため。REQ-39）
    if total == 0 {
        return Tensor::new(vec![m, n], Vec::new());
    }
    let nitems = n.div_ceil(GEMM_NB);
    let threads = threads.map_or_else(
        || worker_count(total.saturating_mul(k), nitems),
        |t| t.min(nitems).max(1),
    );
    let next = AtomicUsize::new(0);
    let slots: Mutex<Vec<Vec<f32>>> = Mutex::new((0..nitems).map(|_| Vec::new()).collect());
    let c = c.map(Tensor::data);
    run_workers(threads, |stop| {
        let mut scratch = Vec::new();
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= nitems || stop.load(Ordering::Relaxed) {
                return Ok(());
            }
            check_deadline(until)?;
            let n0 = i * GEMM_NB;
            let n1 = (n0 + GEMM_NB).min(n);
            let r = gemm_item(
                a.data(),
                m,
                k,
                w.data(),
                c,
                n0,
                n1,
                &to_f32,
                &mut scratch,
                until,
            )?;
            let mut s = slots.lock().map_err(|_| OpError::LimitExceeded)?;
            *s.get_mut(i).ok_or(OpError::ShapeMismatch)? = r;
        }
    })?;
    let slots = slots.into_inner().map_err(|_| OpError::LimitExceeded)?;
    // ゼロ初期化の確保は未書き込みのページを実メモリに載せないので、単位の結果を 1 件ずつ
    // 書き戻して drop すれば、出力と全単位の結果が同時に常駐する 2 倍ピークにならない
    let mut out = vec![0.0f32; total];
    for (i, r) in slots.into_iter().enumerate() {
        let n0 = i * GEMM_NB;
        let len = (n0 + GEMM_NB).min(n) - n0;
        if r.len() != m * len {
            // 未計算の単位（全単位が計算済みなら起きない）を 0 のまま成功として返さない
            return Err(OpError::ShapeMismatch);
        }
        for (mi, src) in r.chunks_exact(len).enumerate() {
            out.get_mut(mi * n + n0..mi * n + n0 + len)
                .ok_or(OpError::ShapeMismatch)?
                .copy_from_slice(src);
        }
    }
    Tensor::new(vec![m, n], out)
}

#[cfg(test)]
mod tests {
    use super::super::norm_act::{reduce_mean_last_axis, sigmoid};
    use super::*;

    fn t(shape: &[usize], data: Vec<f32>) -> Tensor<f32> {
        Tensor::new(shape.to_vec(), data).unwrap()
    }

    #[test]
    fn req39_matmul_gemm_stop_at_deadline() {
        // 過去の期限を渡すと、積和ループの中で DeadlineExceeded になる
        let past = Instant::now().checked_sub(std::time::Duration::from_secs(1));
        let a = t(&[2, 3], vec![1.0; 6]);
        let b = t(&[3, 2], vec![1.0; 6]);
        assert_eq!(matmul(&a, &b, past), Err(OpError::DeadlineExceeded));
        let w = t(&[2, 3], vec![1.0; 6]);
        assert_eq!(
            gemm_trans_b(&a, &w, None, |v: f32| v, past),
            Err(OpError::DeadlineExceeded)
        );
    }

    #[test]
    fn req41_add_sub_mul_broadcast() {
        let a = t(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = t(&[3], vec![10.0, 20.0, 30.0]);
        assert_eq!(
            add(&a, &b).unwrap().data(),
            &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]
        );
        assert_eq!(
            sub(&a, &b).unwrap().data(),
            &[-9.0, -18.0, -27.0, -6.0, -15.0, -24.0]
        );
        assert_eq!(
            mul(&a, &b).unwrap().data(),
            &[10.0, 40.0, 90.0, 40.0, 100.0, 180.0]
        );
        let x = t(&[2, 1, 4], vec![1.0; 8]);
        let y = t(&[3, 1], vec![1.0, 2.0, 3.0]);
        let r = mul(&x, &y).unwrap();
        assert_eq!(r.shape(), &[2, 3, 4]);
        assert_eq!(r.data().get(4..8).unwrap(), &[2.0; 4]);
        let s = t(&[], vec![0.5]);
        assert_eq!(add(&s, &b).unwrap().data(), &[10.5, 20.5, 30.5]);
        assert_eq!(
            add(&a, &t(&[2], vec![1.0, 2.0])),
            Err(OpError::ShapeMismatch)
        );
    }

    #[test]
    fn req41_add_i64_checked() {
        let p = Tensor::new(vec![3], vec![0i64, 1, 2]).unwrap();
        let o = Tensor::new(vec![1], vec![5i64]).unwrap();
        assert_eq!(add_i64(&p, &o).unwrap().data(), &[5, 6, 7]);
        let m = Tensor::new(vec![1], vec![i64::MAX]).unwrap();
        let one = Tensor::new(vec![1], vec![1i64]).unwrap();
        assert_eq!(add_i64(&m, &one), Err(OpError::IntegerOverflow));
    }

    #[test]
    fn req41_sqrt_reciprocal() {
        let r = sqrt(&t(&[4], vec![4.0, 9.0, 0.25, 0.0])).unwrap();
        assert_eq!(r.data(), &[2.0, 3.0, 0.5, 0.0]);
        assert!(sqrt(&t(&[1], vec![-1.0])).unwrap().data()[0].is_nan());
        assert_eq!(
            reciprocal(&t(&[3], vec![2.0, 4.0, -0.5])).unwrap().data(),
            &[0.5, 0.25, -2.0]
        );
        let third = reciprocal(&t(&[1], vec![3.0])).unwrap().data()[0];
        assert_eq!(third.to_bits(), (1.0f32 / 3.0).to_bits());
        assert_eq!(
            reciprocal(&t(&[1], vec![0.0])).unwrap().data()[0],
            f32::INFINITY
        );
    }

    #[test]
    fn req41_cast_bf16_and_i64() {
        let b = Tensor::new(vec![4], vec![0x3F80u16, 0xC000, 0x3F81, 0x7F80]).unwrap();
        assert_eq!(
            cast_bf16_to_f32(&b).unwrap().data(),
            &[1.0, -2.0, 1.0078125, f32::INFINITY]
        );
        let i = Tensor::new(vec![4], vec![0i64, 1, 32767, -5]).unwrap();
        assert_eq!(
            cast_i64_to_f32(&i).unwrap().data(),
            &[0.0, 1.0, 32767.0, -5.0]
        );
        // 2^24+1 は f32 で表せず最近接偶数（2^24）へ丸まる
        let big = Tensor::new(vec![1], vec![16_777_217i64]).unwrap();
        assert_eq!(cast_i64_to_f32(&big).unwrap().data(), &[16_777_216.0]);
    }

    #[test]
    fn req41_matmul_2d_and_batch_broadcast() {
        let a = t(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = t(&[3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let r = matmul(&a, &b, None).unwrap();
        assert_eq!(r.shape(), &[2, 2]);
        assert_eq!(r.data(), &[58.0, 64.0, 139.0, 154.0]);

        // [2,1,2,2] @ [3,2,2] -> [2,3,2,2]。b のバッチ 0,1,2 = I, 2I, 3I
        let a4 = t(&[2, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let mut bd = Vec::new();
        for s in [1.0f32, 2.0, 3.0] {
            bd.extend([s, 0.0, 0.0, s]);
        }
        let r = matmul(&a4, &t(&[3, 2, 2], bd), None).unwrap();
        assert_eq!(r.shape(), &[2, 3, 2, 2]);
        assert_eq!(r.data().get(4..8).unwrap(), &[2.0, 4.0, 6.0, 8.0]);
        assert_eq!(r.data().get(20..24).unwrap(), &[15.0, 18.0, 21.0, 24.0]);

        // GQA 形 [1,2,2,T,hd] @ [1,2,1,hd,T]（T=2, hd=1）
        let q = t(
            &[1, 2, 2, 2, 1],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        );
        let k = t(&[1, 2, 1, 1, 2], vec![1.0, 10.0, 2.0, 20.0]);
        let r = matmul(&q, &k, None).unwrap();
        assert_eq!(r.shape(), &[1, 2, 2, 2, 2]);
        assert_eq!(r.data().get(0..4).unwrap(), &[1.0, 10.0, 2.0, 20.0]);
        assert_eq!(r.data().get(12..16).unwrap(), &[14.0, 140.0, 16.0, 160.0]);

        // K=0 は全 0
        let z = matmul(&t(&[2, 0], vec![]), &t(&[0, 3], vec![]), None).unwrap();
        assert_eq!(z.data(), &[0.0; 6]);
    }

    #[test]
    fn req41_matmul_rejects() {
        let a = t(&[2, 3], vec![0.0; 6]);
        assert_eq!(
            matmul(&a, &t(&[2, 3], vec![0.0; 6]), None),
            Err(OpError::ShapeMismatch)
        );
        assert_eq!(
            matmul(&t(&[3], vec![0.0; 3]), &t(&[3, 1], vec![0.0; 3]), None),
            Err(OpError::ShapeMismatch)
        );
        // rank 6 は Tensor 自体が作れない
        assert_eq!(
            Tensor::new(vec![1, 1, 1, 1, 1, 1], vec![0.0f32]),
            Err(OpError::ShapeMismatch)
        );
        // バッチ不整合
        assert_eq!(
            matmul(
                &t(&[2, 1, 1], vec![0.0; 2]),
                &t(&[3, 1, 1], vec![0.0; 3]),
                None
            ),
            Err(OpError::ShapeMismatch)
        );
        // 巨大出力は確保前に拒否（入力は小さい）
        let x = t(&[1 << 15, 1], vec![0.0; 1 << 15]);
        let y = t(&[1, 1 << 15], vec![0.0; 1 << 15]);
        assert_eq!(matmul(&x, &y, None), Err(OpError::LimitExceeded));
    }

    #[test]
    fn req41_matmul_empty_output_with_huge_dim_does_not_overflow() {
        // REQ-39: A=[0, MAX, 2]・B=[0, 2, 1] は m * k があふれる
        let a = Tensor::new(vec![0, usize::MAX, 2], Vec::<f32>::new()).unwrap();
        let b = Tensor::new(vec![0, 2, 1], Vec::<f32>::new()).unwrap();
        let r = matmul(&a, &b, None).unwrap();
        assert_eq!(r.shape(), &[0, usize::MAX, 1]);
        assert!(r.data().is_empty());
    }

    #[test]
    fn req41_gemm_empty_output_returns_immediately() {
        // REQ-39: A=[MAX, 0]・W=[0, 0] は 0..m を回さない
        let a = Tensor::new(vec![usize::MAX, 0], Vec::<f32>::new()).unwrap();
        let w = Tensor::new(vec![0, 0], Vec::<f32>::new()).unwrap();
        let r = gemm_trans_b(&a, &w, None, |v: f32| v, None).unwrap();
        assert_eq!(r.shape(), &[usize::MAX, 0]);
        assert!(r.data().is_empty());
    }

    #[test]
    fn req41_matmul_sum_order_is_fixed() {
        // k 昇順なら ((1e8 + 1) - 1e8) = 0。順序が変わると 1 になる（REQ-28）
        let a = t(&[1, 3], vec![1e8, 1.0, -1e8]);
        let b = t(&[3, 1], vec![1.0, 1.0, 1.0]);
        assert_eq!(matmul(&a, &b, None).unwrap().data(), &[0.0]);
    }

    #[test]
    fn req28_gemm_sum_order_is_lane_fixed() {
        // lane 別（k mod 4）に k 昇順で累積 → lane 順に加算。a=[1e8,1,0,0,-1e8]・w=1 なら
        // lane0 = 1e8 + (-1e8) = 0、lane1 = 1 で合計 1.0（k 昇順の逐次なら (1e8+1)-1e8 = 0）。
        // 順序が変わると値が変わるので、加算順の固定を機械照合する（REQ-28・#629）
        let a = t(&[1, 5], vec![1e8, 1.0, 0.0, 0.0, -1e8]);
        let w = t(&[1, 5], vec![1.0; 5]);
        assert_eq!(
            gemm_trans_b(&a, &w, None, |v: f32| v, None).unwrap().data(),
            &[1.0]
        );
    }

    /// 小さな整数の積和（和が 2^24 未満）は加算順によらず正確なので、並列経路の配置の正しさを
    /// 素朴な定義と比べられる。
    fn small_int(i: usize, j: usize, a: usize, b: usize) -> f32 {
        ((i * a + j) % b) as f32 - (b / 2) as f32
    }

    #[test]
    fn req28_gemm_parallel_matches_naive_and_is_deterministic() {
        // m*n*k >= PARALLEL_MIN_MACS でスレッドを使う。n は GEMM_NB・GEMM_NR の端数、k は lane の端数を含む
        let (m, n, k) = (9usize, 101usize, 4621usize);
        assert!(m * n * k >= PARALLEL_MIN_MACS);
        let a = t(
            &[m, k],
            (0..m * k).map(|x| small_int(x / k, x % k, 7, 5)).collect(),
        );
        let w = t(
            &[n, k],
            (0..n * k).map(|x| small_int(x / k, x % k, 3, 7)).collect(),
        );
        let bias = t(&[n], (0..n).map(|x| x as f32 * 0.5).collect());
        let r1 = gemm_trans_b(&a, &w, Some(&bias), |v: f32| v, None).unwrap();
        let r2 = gemm_trans_b(&a, &w, Some(&bias), |v: f32| v, None).unwrap();
        assert_eq!(r1, r2);
        assert_eq!(r1.shape(), &[m, n]);
        for mi in 0..m {
            for ni in 0..n {
                let mut s = 0.0f32;
                for kk in 0..k {
                    s += a.data()[mi * k + kk] * w.data()[ni * k + kk];
                }
                s += bias.data()[ni];
                assert_eq!(r1.data()[mi * n + ni], s, "m={mi} n={ni}");
            }
        }
    }

    #[test]
    fn req28_matmul_parallel_matches_naive() {
        // バッチ 2×5（ブロードキャスト）× m=104 行をスレッドへ分ける
        let (bat, m, k, n) = (5usize, 104usize, 64usize, 64usize);
        assert!(2 * bat * m * k * n >= PARALLEL_MIN_MACS);
        let a = t(
            &[2, 1, m, k],
            (0..2 * m * k)
                .map(|x| small_int(x / k, x % k, 5, 5))
                .collect(),
        );
        let b = t(
            &[bat, k, n],
            (0..bat * k * n)
                .map(|x| small_int(x / n, x % n, 11, 7))
                .collect(),
        );
        let r = matmul(&a, &b, None).unwrap();
        assert_eq!(r.shape(), &[2, bat, m, n]);
        for bi in 0..2 {
            for bj in 0..bat {
                for mi in 0..m {
                    for ni in 0..n {
                        let mut s = 0.0f32;
                        for kk in 0..k {
                            s += a.data()[bi * m * k + mi * k + kk]
                                * b.data()[bj * k * n + kk * n + ni];
                        }
                        assert_eq!(r.data()[((bi * bat + bj) * m + mi) * n + ni], s);
                    }
                }
            }
        }
    }

    /// 固定 seed の疑似乱数（xorshift64）。大小混在の非整数 f32 を作り、丸めに敏感なデータにする。
    fn rand_f32(state: &mut u64, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *state ^= *state << 13;
                *state ^= *state >> 7;
                *state ^= *state << 17;
                let u = (*state >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                u * [1e-3, 1.0, 1e3, 1e5][(*state % 4) as usize]
            })
            .collect()
    }

    fn bits(t: &Tensor<f32>) -> Vec<u32> {
        t.data().iter().map(|v| v.to_bits()).collect()
    }

    #[test]
    fn req28_gemm_result_is_independent_of_thread_count() {
        // m=70 は GEMM_M_CHUNK(32) を跨ぐ 3 チャンク、n=37 は 3 単位・端数あり。
        // k は < 4・k%4=1/2/3・lane 割り切れを含む。f32 と bf16 の両方（REQ-28・#629）
        let (m, n) = (70usize, 37usize);
        assert!(m > 2 * GEMM_M_CHUNK);
        let mut st = 0x9E37_79B9_7F4A_7C15u64;
        for k in [1usize, 2, 3, 4, 5, 130, 131, 132] {
            let a = t(&[m, k], rand_f32(&mut st, m * k));
            let bias = t(&[n], rand_f32(&mut st, n));
            let wf = t(&[n, k], rand_f32(&mut st, n * k));
            let wb = Tensor::new(
                vec![n, k],
                wf.data()
                    .iter()
                    .map(|v| (v.to_bits() >> 16) as u16)
                    .collect::<Vec<u16>>(),
            )
            .unwrap();
            let base = gemm_trans_b_with(Some(1), &a, &wf, Some(&bias), |v| v, None).unwrap();
            let baseb =
                gemm_trans_b_with(Some(1), &a, &wb, Some(&bias), bf16_to_f32, None).unwrap();
            for th in [3usize, 10] {
                let r = gemm_trans_b_with(Some(th), &a, &wf, Some(&bias), |v| v, None).unwrap();
                assert_eq!(bits(&r), bits(&base), "f32 k={k} threads={th}");
                let r =
                    gemm_trans_b_with(Some(th), &a, &wb, Some(&bias), bf16_to_f32, None).unwrap();
                assert_eq!(bits(&r), bits(&baseb), "bf16 k={k} threads={th}");
            }
            // lane 別（k mod 4）に k 昇順 → lane 順 → bias の定義と一致する
            for mi in 0..m {
                for ni in 0..n {
                    let mut lanes = [0.0f32; GEMM_LANES];
                    for kk in 0..k {
                        lanes[kk % GEMM_LANES] += a.data()[mi * k + kk] * wf.data()[ni * k + kk];
                    }
                    let s = lanes.iter().fold(0.0, |s, &v| s + v) + bias.data()[ni];
                    assert_eq!(base.data()[mi * n + ni], s, "k={k} m={mi} n={ni}");
                }
            }
        }
    }

    #[test]
    fn req28_matmul_result_is_independent_of_thread_count() {
        let mut st = 0xD1B5_4A32_D192_ED03u64;
        let (m, k, n) = (13usize, 37usize, 9usize);
        let a = t(&[2, 1, m, k], rand_f32(&mut st, 2 * m * k));
        let b = t(&[5, k, n], rand_f32(&mut st, 5 * k * n));
        let base = matmul_with(Some(1), &a, &b, None).unwrap();
        for th in [3usize, 10] {
            let r = matmul_with(Some(th), &a, &b, None).unwrap();
            assert_eq!(bits(&r), bits(&base), "threads={th}");
        }
    }

    #[test]
    fn req39_parallel_paths_stop_at_deadline() {
        let past = Instant::now().checked_sub(std::time::Duration::from_secs(1));
        let a = t(&[70, 8], vec![1.0; 70 * 8]);
        let w = t(&[37, 8], vec![1.0; 37 * 8]);
        assert_eq!(
            gemm_trans_b_with(Some(4), &a, &w, None, |v| v, past),
            Err(OpError::DeadlineExceeded)
        );
        let b = t(&[8, 37], vec![1.0; 8 * 37]);
        assert_eq!(
            matmul_with(Some(4), &a, &b, past),
            Err(OpError::DeadlineExceeded)
        );
    }

    #[test]
    fn req41_gemm_trans_b_bf16_and_bias() {
        let a = t(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        // W = [[1,0,-1],[2,2,2]] を bf16 ビット列で
        let w = Tensor::new(
            vec![2, 3],
            vec![0x3F80u16, 0x0000, 0xBF80, 0x4000, 0x4000, 0x4000],
        )
        .unwrap();
        let bias = t(&[2], vec![0.5, -1.0]);
        let r = gemm_trans_b(&a, &w, Some(&bias), bf16_to_f32, None).unwrap();
        assert_eq!(r.shape(), &[2, 2]);
        assert_eq!(r.data(), &[-1.5, 11.0, -1.5, 29.0]);
        let r = gemm_trans_b(&a, &w, None, bf16_to_f32, None).unwrap();
        assert_eq!(r.data(), &[-2.0, 12.0, -2.0, 30.0]);
        let wf = t(&[2, 3], vec![1.0, 0.0, -1.0, 2.0, 2.0, 2.0]);
        let r = gemm_trans_b(&a, &wf, None, |v| v, None).unwrap();
        assert_eq!(r.data(), &[-2.0, 12.0, -2.0, 30.0]);
    }

    #[test]
    fn req41_gemm_rejects() {
        let a = t(&[2, 3], vec![0.0; 6]);
        let w = t(&[2, 3], vec![0.0; 6]);
        let id = |v: f32| v;
        assert_eq!(
            gemm_trans_b(&a, &t(&[2, 2], vec![0.0; 4]), None, id, None),
            Err(OpError::ShapeMismatch)
        );
        assert_eq!(
            gemm_trans_b(&t(&[6], vec![0.0; 6]), &w, None, id, None),
            Err(OpError::ShapeMismatch)
        );
        assert_eq!(
            gemm_trans_b(&a, &t(&[6], vec![0.0; 6]), None, id, None),
            Err(OpError::ShapeMismatch)
        );
        let bad = t(&[3], vec![0.0; 3]);
        assert_eq!(
            gemm_trans_b(&a, &w, Some(&bad), id, None),
            Err(OpError::ShapeMismatch)
        );
    }

    #[test]
    fn req41_rmsnorm_chain_and_silu() {
        // x=2 の行: mean(x^2)=4 -> sqrt=2 -> recip=0.5 -> x*0.5 = 1（eps=0）
        let x = t(&[2, 2], vec![2.0; 4]);
        let var = reduce_mean_last_axis(&mul(&x, &x).unwrap()).unwrap();
        let eps = t(&[1], vec![0.0]);
        let inv = reciprocal(&sqrt(&add(&var, &eps).unwrap()).unwrap()).unwrap();
        assert_eq!(mul(&x, &inv).unwrap().data(), &[1.0; 4]);
        // SiLU: x * sigmoid(x)。x=0 -> 0
        let z = t(&[1], vec![0.0]);
        assert_eq!(mul(&z, &sigmoid(&z).unwrap()).unwrap().data(), &[0.0]);
    }
}
