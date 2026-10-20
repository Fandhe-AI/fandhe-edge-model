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
    MAX_RANK, OpError, Tensor, broadcast_map2, broadcast_shape, broadcast_strides, element_count,
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
        let joined: Vec<bool> = handles.into_iter().map(|h| h.join().is_ok()).collect();
        joined.iter().all(|&ok| ok)
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
/// 複数スレッドに分けないので、結果はスレッド数に依存しない（REQ-28）。期限と停止フラグの検査は
/// 出力行ごと、および行の中でも高々 [`MATMUL_CHECK_MACS`] 積和ごと（REQ-39）。
/// バッチごとのオフセットは行から都度求め、バッチ数に比例する表を確保しない（REQ-39）。
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
                let (oa, ob) = batch_offsets(bi, &batch, &ra, &rb);
                let arow = a
                    .data()
                    .get(oa + mi * k..oa + (mi + 1) * k)
                    .ok_or(OpError::ShapeMismatch)?;
                let bm = b.data().get(ob..ob + b_mat).ok_or(OpError::ShapeMismatch)?;
                // 列を MATMUL_COL_CHUNK ごとに分け、k 方向は列幅に応じた歩幅で検査する（検査 1 回あたりの
                // 積和を n にも k にも依らず有界にする）。要素ごとの加算順（k 昇順）は変わらない
                for (ci, ocols) in orow.chunks_mut(MATMUL_COL_CHUNK).enumerate() {
                    // 列チャンクの開始ごとにも検査する（K が小さく n が大きいと k ループの検査が走らない。REQ-39）
                    if stop.load(Ordering::Relaxed) {
                        return Ok(());
                    }
                    check_deadline(until)?;
                    let col0 = ci * MATMUL_COL_CHUNK;
                    let kstep = matmul_check_step(ocols.len());
                    let mut left = kstep;
                    for (&av, brow) in arow.iter().zip(bm.chunks_exact(n)) {
                        if left == 0 {
                            if stop.load(Ordering::Relaxed) {
                                return Ok(());
                            }
                            check_deadline(until)?;
                            left = kstep;
                        }
                        left -= 1;
                        let bcols = brow
                            .get(col0..col0 + ocols.len())
                            .ok_or(OpError::ShapeMismatch)?;
                        for (o, &bv) in ocols.iter_mut().zip(bcols) {
                            *o = mul_acc(av, bv, *o);
                        }
                    }
                }
            }
        }
    })?;
    check_deadline(until)?;
    Tensor::new(out_shape, out)
}

/// バッチの線形添字 `bi` に対応する A・B の先頭オフセット。表を持たず行ごとに求める（REQ-39）。
/// `batch` の各次元は呼び出し側で 1 以上（出力が空でない）と分かっているが、0 でも割らない。
fn batch_offsets(mut bi: usize, batch: &[usize], ra: &[usize], rb: &[usize]) -> (usize, usize) {
    let (mut oa, mut ob) = (0usize, 0usize);
    for ((&d, &sa), &sb) in batch.iter().zip(ra).zip(rb).rev() {
        let d = d.max(1);
        let i = bi % d;
        bi /= d;
        oa += i * sa;
        ob += i * sb;
    }
    (oa, ob)
}

/// MatMul の期限・停止フラグの検査 1 回あたりの積和回数の上限（REQ-39）。
const MATMUL_CHECK_MACS: usize = 1 << 18;
/// MatMul が出力行を列方向に分ける幅。1 行が長くても検査間隔を [`MATMUL_CHECK_MACS`] に収める。
const MATMUL_COL_CHUNK: usize = 1 << 16;

/// 列幅 `cols`（1〜[`MATMUL_COL_CHUNK`]）に対する k 方向の検査歩幅。`cols × 歩幅 ≤ MATMUL_CHECK_MACS`。
fn matmul_check_step(cols: usize) -> usize {
    (MATMUL_CHECK_MACS / cols.max(1)).max(1)
}

/// 期限を過ぎていれば [`OpError::DeadlineExceeded`]。積和ループが行またはブロックごとに呼ぶ。
fn check_deadline(until: Option<Instant>) -> Result<(), OpError> {
    // テスト専用: 検査の呼び出し回数を数え、指定回数を超えたら期限超過にする（実時間に依らず検査位置を観測する）
    #[cfg(test)]
    if let Some(left) = TEST_CHECKS_LEFT.get() {
        if left == 0 {
            return Err(OpError::DeadlineExceeded);
        }
        TEST_CHECKS_LEFT.set(Some(left - 1));
    }
    match until {
        Some(t) if Instant::now() > t => Err(OpError::DeadlineExceeded),
        _ => Ok(()),
    }
}

#[cfg(test)]
thread_local! {
    /// テスト専用: `Some(n)` なら [`check_deadline`] の n 回の呼び出しの後で期限超過にする。
    static TEST_CHECKS_LEFT: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// Gemm のタイル: A の行数。
const GEMM_MR: usize = 6;
/// Gemm のタイル: W の行数。
const GEMM_NR: usize = 4;
/// Gemm の lane 数（k を `GEMM_LANES` ごとに分けて独立に累積する。NEON 128 bit = f32 × 4）。
const GEMM_LANES: usize = 4;
/// 1 つの仕事の単位が受け持つ W の行数（[`GEMM_NR`] の倍数。スレッドへ動的に配る粒度）。
const GEMM_NB: usize = 16;
/// W のブロックを使い回す A の行数（この行数分の A を L2 に留めたまま W のブロックを再走査する）。
const GEMM_M_CHUNK: usize = 48;
/// k 方向のブロック幅（[`GEMM_LANES`] の倍数）。W の展開領域を `GEMM_NR × GEMM_KB` に有界にし、
/// 期限・停止フラグの検査間隔を高々 `GEMM_M_CHUNK × GEMM_NR × GEMM_KB` 積和（約 13 万）にする（REQ-39）。
const GEMM_KB: usize = 512;
// [`gemm_tile`] の zip は A の 6 行・W の 4 行を明示的に並べているので、タイル幅を変えるならそこも直す
const _: () = assert!(GEMM_MR == 6 && GEMM_NR == 4);
const _: () = assert!(GEMM_KB.is_multiple_of(GEMM_LANES) && GEMM_NB.is_multiple_of(GEMM_NR));
const _: () = assert!(GEMM_M_CHUNK.is_multiple_of(GEMM_MR));

/// タイル 1 枚の累積器（出力要素ごとに lane 別の部分和）。k のブロックをまたいで持ち越す。
type GemmAcc = [[[f32; GEMM_LANES]; GEMM_NR]; GEMM_MR];

/// Gemm がカーネル内で確保する一時領域の上界（バイト）。worker ごとの W の展開領域（heap）と
/// 累積器・書き戻し用バッファ（stack）× 最大スレッド数。形状に依らない定数で、scoring の
/// 確保前検査が出力の見積もりに足す（REQ-39）。
pub fn gemm_temp_bytes() -> usize {
    let per_worker = (GEMM_NR * GEMM_KB
        + GEMM_M_CHUNK * GEMM_NB
        + (GEMM_M_CHUNK / GEMM_MR) * GEMM_MR * GEMM_NR * GEMM_LANES)
        * std::mem::size_of::<f32>();
    std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1)
        .saturating_mul(per_worker)
}

/// `s`（長さ < GEMM_LANES。k の端数）を配列の先頭へ。残りの lane は 0。
///
/// 不変条件: [`gemm_tile`] が全行の長さ == kl を検証したうえで `as_chunks` の余りを渡すので、
/// 長さは常に `kl % GEMM_LANES`。可変長のコピーは memcpy 呼び出しになり積和の自動ベクトル化を
/// 妨げるので、本体ループとは分けて端数の 1 ステップだけで使う。
#[inline(always)]
fn gemm_load_tail(s: &[f32]) -> [f32; GEMM_LANES] {
    let mut v = [0.0; GEMM_LANES];
    for (d, &x) in v.iter_mut().zip(s) {
        *d = x;
    }
    v
}

/// 積和 `a * b + c`。FMA 命令がある target（aarch64・`fma` 付き）だけ融合積和にする。
/// 無い target の `mul_add` は libm のソフトウェア実装で極端に遅いため従来式のまま（REQ-31・#636）。
#[inline(always)]
fn mul_acc(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(any(target_arch = "aarch64", target_feature = "fma"))]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(any(target_arch = "aarch64", target_feature = "fma")))]
    {
        a * b + c
    }
}

/// タイルの 1 ステップ（lane 幅ぶんの k）: ロード済みの A・W の lane 配列を `acc` に積む。
///
/// 添字は定数範囲（`GEMM_MR`・`GEMM_NR`・`GEMM_LANES`）で範囲外にならない。イテレータの zip で書くと
/// `acc` がレジスタに乗らず（要素への参照を経由するため）約 4 倍遅くなったので、添字で書く。
#[inline(always)]
#[allow(clippy::needless_range_loop)]
fn gemm_fma(
    acc: &mut GemmAcc,
    av: &[[f32; GEMM_LANES]; GEMM_MR],
    wv: &[[f32; GEMM_LANES]; GEMM_NR],
) {
    for i in 0..GEMM_MR {
        for j in 0..GEMM_NR {
            for l in 0..GEMM_LANES {
                acc[i][j][l] = mul_acc(av[i][l], wv[j][l], acc[i][j][l]);
            }
        }
    }
}

/// [`GEMM_MR`] 行 × [`GEMM_NR`] 行のタイルの内積を k のブロック（各行の長さ `kl`）ぶん `acc` に積む。
///
/// 加算順（固定・スレッド数に依存しない。REQ-28）: 出力要素ごとに k を `k mod GEMM_LANES` で
/// lane に分け、lane 内は k 昇順で f32 累積する（ブロックの先頭は lane の倍数なので、ブロックを
/// またいでも lane の割り当ては変わらない）。lane の合算は全ブロックの後に [`gemm_fold`] が行う。
#[inline(always)]
fn gemm_tile(
    acc: &mut GemmAcc,
    a: [&[f32]; GEMM_MR],
    w: [&[f32]; GEMM_NR],
    kl: usize,
) -> Result<(), OpError> {
    // 以降のロードの不変条件（全行の長さ == kl）をここで 1 度だけ検証する
    if a.iter().chain(&w).any(|r| r.len() != kl) {
        return Err(OpError::ShapeMismatch);
    }
    // 累積器はブロックの間だけローカルに写してレジスタに乗せる（参照経由だと遅い。gemm_fma の注記）
    let mut local = *acc;
    // 各行を lane 幅の固定長チャンクと端数に分け、8 本を zip で同時に進める。長さは上で全行 == kl と
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
        .zip(ac[4])
        .zip(ac[5])
        .zip(wc[0])
        .zip(wc[1])
        .zip(wc[2])
        .zip(wc[3]);
    for (((((((((a0, a1), a2), a3), a4), a5), w0), w1), w2), w3) in steps {
        gemm_fma(
            &mut local,
            &[*a0, *a1, *a2, *a3, *a4, *a5],
            &[*w0, *w1, *w2, *w3],
        );
    }
    if !at[0].is_empty() {
        let av = at.map(gemm_load_tail);
        let wv = wt.map(gemm_load_tail);
        gemm_fma(&mut local, &av, &wv);
    }
    *acc = local;
    Ok(())
}

/// 累積器の lane を lane 0 から順に足して出力要素にする（[`gemm_tile`] の加算順の最終段）。
fn gemm_fold(acc: &GemmAcc) -> [[f32; GEMM_NR]; GEMM_MR] {
    let mut out = [[0f32; GEMM_NR]; GEMM_MR];
    for (oi, acc_i) in out.iter_mut().zip(acc) {
        for (o, lanes) in oi.iter_mut().zip(acc_i) {
            *o = lanes.iter().fold(0.0, |s, &v| s + v);
        }
    }
    out
}

/// Gemm の仕事の単位: W の行 `n0..n1`（高々 [`GEMM_NB`] 行）× A の全行を `out`（`[m][n]` 行優先）へ書く。
///
/// A の [`GEMM_M_CHUNK`] 行 × W の [`GEMM_NR`] 行ごとに、k を [`GEMM_KB`] のブロックで進める:
/// ブロックの W を `scratch` に f32 で展開し（bf16→f32 の展開を A の行で共用）、各タイルの累積器に
/// 積む。ブロックごとに期限と `stop` を検査する。結果は A のチャンクごとに固定長のバッファへ畳み、
/// `out` のロックの下で書き戻す（出力以外に m×n の領域を持たない。REQ-39）。
#[allow(clippy::too_many_arguments)]
fn gemm_item<W: Copy>(
    a: &[f32],
    m: usize,
    k: usize,
    w: &[W],
    c: Option<&[f32]>,
    n: usize,
    n0: usize,
    n1: usize,
    to_f32: &impl Fn(W) -> f32,
    scratch: &mut [f32],
    out: &Mutex<Vec<f32>>,
    until: Option<Instant>,
    stop: &AtomicBool,
) -> Result<(), OpError> {
    let len = n1 - n0;
    let mut res = [0.0f32; GEMM_M_CHUNK * GEMM_NB];
    let mut accs: [GemmAcc; GEMM_M_CHUNK / GEMM_MR];
    for mc in (0..m).step_by(GEMM_M_CHUNK) {
        // k = 0 だと k ブロックの検査が走らないので、m チャンクの開始ごとにも検査する（REQ-39）
        if stop.load(Ordering::Relaxed) {
            return Ok(());
        }
        check_deadline(until)?;
        let mend = (mc + GEMM_M_CHUNK).min(m);
        for nb in (n0..n1).step_by(GEMM_NR) {
            let nend = (nb + GEMM_NR).min(n1);
            accs = [[[[0.0f32; GEMM_LANES]; GEMM_NR]; GEMM_MR]; GEMM_M_CHUNK / GEMM_MR];
            for kb in (0..k).step_by(GEMM_KB) {
                if stop.load(Ordering::Relaxed) {
                    return Ok(());
                }
                check_deadline(until)?;
                let kend = (kb + GEMM_KB).min(k);
                let kl = kend - kb;
                // タイルの端で W の行が足りない分は最後の行を重ねて計算し、書き戻さない（分岐を増やさない）
                for j in 0..GEMM_NR {
                    let ni = (nb + j).min(nend - 1);
                    let src = w
                        .get(ni * k + kb..ni * k + kend)
                        .ok_or(OpError::ShapeMismatch)?;
                    let dst = scratch
                        .get_mut(j * kl..(j + 1) * kl)
                        .ok_or(OpError::ShapeMismatch)?;
                    for (d, &s) in dst.iter_mut().zip(src) {
                        *d = to_f32(s);
                    }
                }
                let mut wr: [&[f32]; GEMM_NR] = [&[]; GEMM_NR];
                for (j, r) in wr.iter_mut().enumerate() {
                    *r = scratch
                        .get(j * kl..(j + 1) * kl)
                        .ok_or(OpError::ShapeMismatch)?;
                }
                for (acc, mb) in accs.iter_mut().zip((mc..mend).step_by(GEMM_MR)) {
                    let mut ar: [&[f32]; GEMM_MR] = [&[]; GEMM_MR];
                    for (i, r) in ar.iter_mut().enumerate() {
                        let mi = (mb + i).min(mend - 1);
                        *r = a
                            .get(mi * k + kb..mi * k + kend)
                            .ok_or(OpError::ShapeMismatch)?;
                    }
                    gemm_tile(acc, ar, wr, kl)?;
                }
            }
            for (acc, mb) in accs.iter().zip((mc..mend).step_by(GEMM_MR)) {
                let t = gemm_fold(acc);
                for (i, ti) in t.iter().enumerate().take(mend - mb) {
                    for (j, &v) in ti.iter().enumerate().take(nend - nb) {
                        // bias 無しのときは何も足さず、タイルの値をそのまま書く
                        let v = match c {
                            Some(c) => v + c.get(nb + j).copied().ok_or(OpError::ShapeMismatch)?,
                            None => v,
                        };
                        *res.get_mut((mb - mc + i) * len + (nb - n0) + j)
                            .ok_or(OpError::ShapeMismatch)? = v;
                    }
                }
            }
        }
        let mut o = out.lock().map_err(|_| OpError::LimitExceeded)?;
        for mi in mc..mend {
            o.get_mut(mi * n + n0..mi * n + n1)
                .ok_or(OpError::ShapeMismatch)?
                .copy_from_slice(
                    res.get((mi - mc) * len..(mi - mc + 1) * len)
                        .ok_or(OpError::ShapeMismatch)?,
                );
        }
    }
    Ok(())
}

/// Gemm（transA=0・transB=1・alpha=1・beta=1 固定）。`A[M,K] × W[N,K]ᵀ (+ C[N])`。
///
/// 重みは元の型 `W` のまま受け、`to_f32` でブロックごとに展開する（全体を f32 へ複製しない。
/// bf16 なら [`bf16_to_f32`]）。`out[m][n] = Σ_k a[m,k]*w[n,k]` を求めてから `c[n]` を足す。
///
/// 加算順は [`gemm_tile`] の固定順（lane 別に k 昇順 → lane 順）で、スレッド数・分配順・k の
/// ブロック幅に依存しない（REQ-28。k 昇順の逐次加算とは丸めが異なりうるが決定的）。W の行を
/// [`GEMM_NB`] 行ずつの単位に切ってスレッドへ動的に配り、1 つの出力要素は 1 つのスレッドだけが計算する。
/// 期限と停止フラグ（他の worker のエラー）の検査は k の [`GEMM_KB`] ブロックごとで、1 検査あたり
/// 高々 `GEMM_M_CHUNK × GEMM_NR × GEMM_KB` 積和（形状に依らない。REQ-39）。一時領域は worker ごとの
/// 定数（[`gemm_temp_bytes`]）で、出力のほかに m×n の領域を持たない。
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
/// 統合テストのベンチ（`tests/gemm_bench.rs`。#643）がスレッド数を指定するための入口で、製品の API ではない。
#[doc(hidden)]
pub fn gemm_trans_b_with<W: Copy + Sync>(
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
    let done = AtomicUsize::new(0);
    // 出力は 1 つだけ確保し、各単位が自分の列範囲をロックの下で書く（単位の結果を別に溜めない）
    let out: Mutex<Vec<f32>> = Mutex::new(vec![0.0f32; total]);
    let c = c.map(Tensor::data);
    run_workers(threads, |stop| {
        let mut scratch = vec![0.0f32; GEMM_NR * GEMM_KB];
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= nitems || stop.load(Ordering::Relaxed) {
                return Ok(());
            }
            let n0 = i * GEMM_NB;
            let n1 = (n0 + GEMM_NB).min(n);
            gemm_item(
                a.data(),
                m,
                k,
                w.data(),
                c,
                n,
                n0,
                n1,
                &to_f32,
                &mut scratch,
                &out,
                until,
                stop,
            )?;
            done.fetch_add(1, Ordering::Relaxed);
        }
    })?;
    check_deadline(until)?;
    // 未計算の単位（全 worker が正常終了したなら起きない）を 0 のまま成功として返さない
    if done.load(Ordering::Relaxed) != nitems {
        return Err(OpError::ShapeMismatch);
    }
    let out = out.into_inner().map_err(|_| OpError::LimitExceeded)?;
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

    /// K=0 の Gemm でも m チャンクごとに期限を検査する（REQ-39）。m チャンク 3 つで 1 回目だけ通る設定にし、
    /// 2 つ目のチャンクの開始で止まること（検査が無いと最後の検査まで全出力を書き戻す）。
    #[test]
    fn req39_gemm_k0_checks_deadline_per_m_chunk() {
        let m = 3 * GEMM_M_CHUNK;
        let a = t(&[m, 0], Vec::new());
        let w = t(&[4, 0], Vec::new());
        TEST_CHECKS_LEFT.set(Some(1));
        let r = gemm_trans_b_with(Some(1), &a, &w, None, |v: f32| v, None);
        TEST_CHECKS_LEFT.set(None);
        assert_eq!(r, Err(OpError::DeadlineExceeded));
    }

    /// K が小さく n が大きい MatMul でも列チャンクごとに期限を検査する（REQ-39）。行の検査と
    /// 1 つ目の列チャンクだけ通る設定にし、2 つ目の列チャンクの開始で止まること。
    #[test]
    fn req39_matmul_small_k_checks_deadline_per_col_chunk() {
        let n = 3 * MATMUL_COL_CHUNK;
        let a = t(&[1, 4], vec![1.0; 4]);
        let b = t(&[4, n], vec![1.0; 4 * n]);
        TEST_CHECKS_LEFT.set(Some(2));
        let r = matmul_with(Some(1), &a, &b, None);
        TEST_CHECKS_LEFT.set(None);
        assert_eq!(r, Err(OpError::DeadlineExceeded));
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
        // m=110 は GEMM_M_CHUNK を跨ぐ 3 チャンク、m=1/5/7/47/49 はタイル端数。k=515・1030 は GEMM_KB を跨ぐ、n=37 は 3 単位・端数あり。
        // k は < 4・k%4=1/2/3・lane 割り切れを含む。f32 と bf16 の両方（REQ-28・#629）
        let n = 37usize;
        const _: () = assert!(110 > 2 * GEMM_M_CHUNK);
        let mut st = 0x9E37_79B9_7F4A_7C15u64;
        for (m, k) in [1usize, 5, 6, 7, 47, 48, 49, 110]
            .into_iter()
            .flat_map(|m| {
                [1usize, 2, 3, 4, 5, 130, 131, 132, 515, 1030]
                    .into_iter()
                    .map(move |k| (m, k))
            })
        {
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
            // lane 別（k mod 4）に k 昇順（積和は [`mul_acc`]）→ lane 順 → bias の定義と一致する
            for mi in 0..m {
                for ni in 0..n {
                    let mut lanes = [0.0f32; GEMM_LANES];
                    for kk in 0..k {
                        let l = kk % GEMM_LANES;
                        lanes[l] = mul_acc(a.data()[mi * k + kk], wf.data()[ni * k + kk], lanes[l]);
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

    /// 1 つの worker のエラーで停止フラグが立ち、他の worker が仕事を残したまま早期に抜ける（REQ-39・#630）。
    #[test]
    fn req39_run_workers_stop_flag_halts_other_workers() {
        let claimed = AtomicUsize::new(0);
        let r = run_workers(3, |stop| {
            if claimed.fetch_add(1, Ordering::Relaxed) == 0 {
                return Err(OpError::DeadlineExceeded);
            }
            // 停止フラグが立つまで待つ。立たなければ（伝播が壊れていれば）10 秒で別のエラーにして失敗させる
            let t0 = Instant::now();
            while !stop.load(Ordering::Relaxed) {
                if t0.elapsed() > std::time::Duration::from_secs(10) {
                    return Err(OpError::LimitExceeded);
                }
                std::thread::yield_now();
            }
            Ok(())
        });
        assert_eq!(r, Err(OpError::DeadlineExceeded));
    }

    /// Gemm は k のブロックごとに期限を検査する。W の展開（`to_f32`）で最初のブロックの間に期限を
    /// 跨がせると、次のブロックの前で止まり、展開の回数は 1 ブロック分（GEMM_NR × GEMM_KB）に留まる。
    /// 検査が A のチャンクごと（k に比例）だと 4 ブロック分を展開し切ってしまう（REQ-39・#630）。
    #[test]
    fn req39_gemm_checks_deadline_per_k_block() {
        let k = 4 * GEMM_KB;
        let a = t(&[4, k], vec![1.0; 4 * k]);
        let w = t(&[4, k], vec![1.0; 4 * k]);
        let calls = AtomicUsize::new(0);
        let to_f32 = |v: f32| {
            if calls.fetch_add(1, Ordering::Relaxed) == 0 {
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            v
        };
        let until = Instant::now().checked_add(std::time::Duration::from_millis(5));
        assert_eq!(
            gemm_trans_b_with(Some(1), &a, &w, None, to_f32, until),
            Err(OpError::DeadlineExceeded)
        );
        assert_eq!(calls.load(Ordering::Relaxed), GEMM_NR * GEMM_KB);
    }

    /// 最後の計算区間で期限を跨いでも、結果を返す直前の検査で期限超過にする（REQ-39・#630）。
    #[test]
    fn req39_gemm_checks_deadline_before_returning() {
        let a = t(&[1, 1], vec![1.0]);
        let w = t(&[1, 1], vec![1.0]);
        let to_f32 = |v: f32| {
            std::thread::sleep(std::time::Duration::from_millis(50));
            v
        };
        let until = Instant::now().checked_add(std::time::Duration::from_millis(5));
        assert_eq!(
            gemm_trans_b(&a, &w, None, to_f32, until),
            Err(OpError::DeadlineExceeded)
        );
    }

    /// MatMul の行内の検査歩幅は列幅に依らず 1 検査あたり MATMUL_CHECK_MACS 積和以下（REQ-39・#630）。
    #[test]
    fn req39_matmul_check_step_is_bounded() {
        for cols in [1usize, 2, 3, 64, 1000, 1 << 15, MATMUL_COL_CHUNK] {
            let step = matmul_check_step(cols);
            assert!(step >= 1, "cols={cols}");
            assert!(step * cols <= MATMUL_CHECK_MACS, "cols={cols} step={step}");
        }
        assert_eq!(matmul_check_step(1), MATMUL_CHECK_MACS);
        assert_eq!(matmul_check_step(MATMUL_COL_CHUNK), 4);
    }

    /// MatMul はバッチ数に比例する表を確保しない。バッチ 2^22（旧実装なら 64 MiB のオフセット表）でも
    /// 最初の行の検査で止まる（REQ-39・#630）。出力は calloc で未使用ページを実メモリに載せない。
    #[test]
    fn req39_matmul_huge_batch_has_no_offset_table() {
        let a = Tensor::new(vec![1 << 22, 1, 0], Vec::<f32>::new()).unwrap();
        let b = Tensor::new(vec![0, 1], Vec::<f32>::new()).unwrap();
        let past = Instant::now().checked_sub(std::time::Duration::from_secs(1));
        assert_eq!(
            matmul_with(Some(1), &a, &b, past),
            Err(OpError::DeadlineExceeded)
        );
        // 都度計算のオフセットがブロードキャストを正しく辿ることは
        // req41_matmul_2d_and_batch_broadcast・req28_matmul_parallel_matches_naive で照合する
        assert_eq!(batch_offsets(5, &[2, 3], &[0, 4], &[12, 0]), (2 * 4, 12));
    }

    /// Gemm の一時領域は形状（特に K）に依らない定数で、2 GiB の保持上限に対して無視できる大きさ（REQ-39・#630）。
    #[test]
    fn req39_gemm_temp_bytes_is_shape_independent_constant() {
        let threads = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1);
        let per = (GEMM_NR * GEMM_KB
            + GEMM_M_CHUNK * GEMM_NB
            + (GEMM_M_CHUNK / GEMM_MR) * GEMM_MR * GEMM_NR * GEMM_LANES)
            * 4;
        assert_eq!(gemm_temp_bytes(), threads * per);
        assert!(per < 1 << 16);
        assert_eq!(gemm_temp_bytes(), gemm_temp_bytes());
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
