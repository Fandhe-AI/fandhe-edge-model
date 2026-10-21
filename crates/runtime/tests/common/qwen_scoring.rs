//! 極小 Qwen2 の KV 使い回し採点の参照順伝播（テスト専用。REQ-28・REQ-41・TASK-41.7-7・#421）。
//!
//! 役割: `docs/design/qwen2-scoring-onnx-ops.md` 2 節の順に、#416〜#419 のカーネル
//! （`fandhe_edge_runtime::onnx::ops::*`）だけを組み合わせて採点する。ONNX グラフの実行器ではない
//! （実行器は未実装で、実装済みを装わない）。#118 の全件一致ハーネス（`full_match.rs`）へ
//! `ScoringBackend` として差し込み、Qwen 系の採点でも 1 件・バッチ・評価器経路が一致することを確かめる。
//!
//! 証拠種別: テストハーネス（合成重み・極小構成）。実重み・実データは使わない。
//! 数値: 計算は f32、出力 softmax のみ f64。f32 の計算に 1e-9 の許容差は適用しない
//! （1e-9 は同じ入力からの同じ計算の決定性比較にのみ使い、KV 使い回しと素朴な全系列採点の比較は別の許容差）。
//! テスト専用のため `unwrap`・添字を使ってよい。

use fandhe_edge_runtime::onnx::ops::arith::{
    add, bf16_to_f32, cast_i64_to_f32, gemm_trans_b, matmul, mul, reciprocal, sqrt, sub,
};
use fandhe_edge_runtime::onnx::ops::mask_rope::{cos, greater_i64, range_i64, sin, where_f32};
use fandhe_edge_runtime::onnx::ops::norm_act::{
    log_softmax_last_axis, reduce_mean_last_axis, reduce_sum, sigmoid, softmax_last_axis,
    softmax_last_axis_f64,
};
use fandhe_edge_runtime::onnx::ops::shape::{
    concat, expand, gather_axis0, gather_elements_last_axis, reshape, slice, split, transpose,
    unsqueeze,
};
use fandhe_edge_runtime::onnx::ops::{OpError, Tensor};
use fandhe_edge_runtime::pipeline::{BackendError, ScoringBackend, TokenIds};
use std::sync::OnceLock;

type R<T> = Result<T, OpError>;
type Kv = (Tensor<f32>, Tensor<f32>);

pub const LAYERS: usize = 2;
pub const HIDDEN: usize = 16;
pub const HEADS: usize = 4;
pub const KV_HEADS: usize = 2;
pub const HEAD_DIM: usize = 4;
pub const INTER: usize = 32;
pub const VOCAB: usize = 64;
/// 選択肢の右詰めに使う pad の id（語彙の最後。`endoftext` 相当）。
pub const PAD: i64 = (VOCAB - 1) as i64;
pub const K: usize = 3;
pub const M: usize = 3;
const THETA: f32 = 10000.0;
const EPS: f32 = 1e-6;
/// 計算量を有界にする系列長の上限（REQ-39 の作法）。
pub const MAX_T: usize = 8;

/// 選択肢 token 列（長さが違うものは右を pad で埋める）。
const CAND: [[i64; M]; K] = [[5, 9, 13], [22, 3, PAD], [31, 40, PAD]];
/// 有効位置の重み（長さ正規化なしの合計が契約。有効位置は 1、pad の位置は 0）。
const WEIGHT: [[f32; M]; K] = [[1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.0]];

fn lcg(st: &mut u64) -> f32 {
    *st = st
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*st >> 40) as f32 / (1u64 << 24) as f32) - 0.5
}

fn bf16_weights(st: &mut u64, shape: &[usize], scale: f32) -> Tensor<u16> {
    let n: usize = shape.iter().product();
    let d = (0..n)
        .map(|_| ((lcg(st) * scale).to_bits() >> 16) as u16)
        .collect();
    Tensor::new(shape.to_vec(), d).unwrap()
}

struct Layer {
    wq: Tensor<u16>,
    bq: Tensor<f32>,
    wk: Tensor<u16>,
    bk: Tensor<f32>,
    wv: Tensor<u16>,
    bv: Tensor<f32>,
    wo: Tensor<u16>,
    n1: Tensor<f32>,
    n2: Tensor<f32>,
    gate: Tensor<u16>,
    up: Tensor<u16>,
    down: Tensor<u16>,
}

struct Weights {
    embed: Tensor<u16>,
    layers: Vec<Layer>,
    fnorm: Tensor<f32>,
    inv_freq: Tensor<f32>,
}

fn f32t(shape: &[usize], d: Vec<f32>) -> Tensor<f32> {
    Tensor::new(shape.to_vec(), d).unwrap()
}
fn i64t(shape: &[usize], d: Vec<i64>) -> Tensor<i64> {
    Tensor::new(shape.to_vec(), d).unwrap()
}
fn dims(d: &[usize]) -> Tensor<i64> {
    i64t(&[d.len()], d.iter().map(|&x| x as i64).collect())
}
fn scalar(v: f32) -> Tensor<f32> {
    f32t(&[], vec![v])
}
fn iscalar(v: i64) -> Tensor<i64> {
    i64t(&[], vec![v])
}
fn ivec(v: &[i64]) -> Tensor<i64> {
    i64t(&[v.len()], v.to_vec())
}
fn small(st: &mut u64, n: usize, base: f32, scale: f32) -> Tensor<f32> {
    f32t(&[n], (0..n).map(|_| base + lcg(st) * scale).collect())
}

fn weights() -> &'static Weights {
    static W: OnceLock<Weights> = OnceLock::new();
    W.get_or_init(|| {
        let mut st = 0x4_2026_1017_u64;
        let (qd, kd) = (HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM);
        let embed = bf16_weights(&mut st, &[VOCAB, HIDDEN], 3.0);
        let layers = (0..LAYERS)
            .map(|_| Layer {
                wq: bf16_weights(&mut st, &[qd, HIDDEN], 1.2),
                bq: small(&mut st, qd, 0.0, 0.2),
                wk: bf16_weights(&mut st, &[kd, HIDDEN], 1.2),
                bk: small(&mut st, kd, 0.0, 0.2),
                wv: bf16_weights(&mut st, &[kd, HIDDEN], 1.2),
                bv: small(&mut st, kd, 0.0, 0.2),
                wo: bf16_weights(&mut st, &[HIDDEN, qd], 1.2),
                n1: small(&mut st, HIDDEN, 1.0, 1.0),
                n2: small(&mut st, HIDDEN, 1.0, 1.0),
                gate: bf16_weights(&mut st, &[INTER, HIDDEN], 1.2),
                up: bf16_weights(&mut st, &[INTER, HIDDEN], 1.2),
                down: bf16_weights(&mut st, &[HIDDEN, INTER], 1.2),
            })
            .collect();
        let half = HEAD_DIM / 2;
        let inv = (0..half)
            .map(|i| THETA.powf(-(2.0 * i as f32) / HEAD_DIM as f32))
            .collect();
        Weights {
            embed,
            layers,
            fnorm: small(&mut st, HIDDEN, 1.0, 1.0),
            inv_freq: f32t(&[1, half], inv),
        }
    })
}

fn lin(x: &Tensor<f32>, w: &Tensor<u16>, b: Option<&Tensor<f32>>) -> R<Tensor<f32>> {
    gemm_trans_b(x, w, b, bf16_to_f32, None)
}

/// RMSNorm: Mul → ReduceMean → Add(eps) → Sqrt → Reciprocal → Mul → Mul(w)。
fn rmsnorm(x: &Tensor<f32>, w: &Tensor<f32>) -> R<Tensor<f32>> {
    let v = reduce_mean_last_axis(&mul(x, x)?)?;
    let r = reciprocal(&sqrt(&add(&v, &scalar(EPS))?)?)?;
    mul(&mul(x, &r)?, w)
}

/// `[B*R, n*hd]` → `[B, n, R, hd]`。
fn heads(x: Tensor<f32>, b: usize, r: usize, n: usize) -> R<Tensor<f32>> {
    transpose(&reshape(x, &dims(&[b, r, n, HEAD_DIM]))?, &[0, 2, 1, 3])
}

/// RoPE（Split 等分 2 つ → Mul・Sub・Add → Concat）。cos・sin は `[R, hd/2]`。
fn rope(x: &Tensor<f32>, c: &Tensor<f32>, s: &Tensor<f32>) -> R<Tensor<f32>> {
    let half = HEAD_DIM / 2;
    let p = split(x, -1, &dims(&[half, half]))?;
    let (x1, x2) = (&p[0], &p[1]);
    let a = sub(&mul(x1, c)?, &mul(x2, s)?)?;
    let b = add(&mul(x2, c)?, &mul(x1, s)?)?;
    concat(&[&a, &b], -1)
}

/// RoPE の表 `[R, hd/2]`（位置は `start..start+rows`）。
fn rope_tables(start: i64, rows: usize) -> R<Kv> {
    let pos = range_i64(&iscalar(start), &iscalar(start + rows as i64), &iscalar(1))?;
    let ang = mul(
        &unsqueeze(cast_i64_to_f32(&pos)?, &ivec(&[1]))?,
        &weights().inv_freq,
    )?;
    Ok((cos(&ang)?, sin(&ang)?))
}

/// 1 層。x は `[B*R, H]`、prefix は `[1,nkv,P,hd]` の K・V（無ければ P=0）。
/// 戻りは (出力, この層で作った K, V = `[B,nkv,R,hd]`)。
fn layer(
    l: &Layer,
    x: &Tensor<f32>,
    (b, r): (usize, usize),
    cs: &Kv,
    prefix: Option<&Kv>,
) -> R<(Tensor<f32>, Tensor<f32>, Tensor<f32>)> {
    let p = prefix.map_or(0, |(k, _)| k.shape()[2]);
    let s = p + r;
    let h = rmsnorm(x, &l.n1)?;
    let q = rope(
        &heads(lin(&h, &l.wq, Some(&l.bq))?, b, r, HEADS)?,
        &cs.0,
        &cs.1,
    )?;
    let k = rope(
        &heads(lin(&h, &l.wk, Some(&l.bk))?, b, r, KV_HEADS)?,
        &cs.0,
        &cs.1,
    )?;
    let v = heads(lin(&h, &l.wv, Some(&l.bv))?, b, r, KV_HEADS)?;
    let (kf, vf) = match prefix {
        Some((pk, pv)) => {
            let sh = dims(&[b, KV_HEADS, p, HEAD_DIM]);
            (
                concat(&[&expand(pk, &sh)?, &k], 2)?,
                concat(&[&expand(pv, &sh)?, &v], 2)?,
            )
        }
        None => (k.clone(), v.clone()),
    };
    // GQA: q を [B,nkv,grp,R,hd]、K・V を [B,nkv,1,S,hd] にして MatMul のバッチ方向で広げる。
    let q5 = reshape(q, &dims(&[b, KV_HEADS, HEADS / KV_HEADS, r, HEAD_DIM]))?;
    let kt = transpose(&unsqueeze(kf, &ivec(&[2]))?, &[0, 1, 2, 4, 3])?;
    let v5 = unsqueeze(vf, &ivec(&[2]))?;
    let sc = mul(
        &matmul(&q5, &kt, None)?,
        &scalar(1.0 / (HEAD_DIM as f32).sqrt()),
    )?;
    // 因果マスク: key 位置 > query 位置（prefix は全て見える）。
    let qpos = range_i64(&iscalar(p as i64), &iscalar(s as i64), &iscalar(1))?;
    let kpos = range_i64(&iscalar(0), &iscalar(s as i64), &iscalar(1))?;
    let future = greater_i64(
        &reshape(kpos, &dims(&[1, s]))?,
        &reshape(qpos, &dims(&[r, 1]))?,
    )?;
    let at = softmax_last_axis(&where_f32(&future, &scalar(-1e30), &sc)?)?;
    let o = matmul(&at, &v5, None)?;
    let o = transpose(&reshape(o, &dims(&[b, HEADS, r, HEAD_DIM]))?, &[0, 2, 1, 3])?;
    let o = lin(&reshape(o, &dims(&[b * r, HEADS * HEAD_DIM]))?, &l.wo, None)?;
    let x = add(x, &o)?;
    let h = rmsnorm(&x, &l.n2)?;
    let g = lin(&h, &l.gate, None)?;
    let act = mul(&g, &sigmoid(&g)?)?;
    let m = lin(&mul(&act, &lin(&h, &l.up, None)?)?, &l.down, None)?;
    Ok((add(&x, &m)?, k, v))
}

/// 全層を通す。ids は `[B*R]` の平坦列。戻りは (隠れ状態 `[B*R,H]`, 層ごとの K・V)。
fn stack(
    ids: &[i64],
    br: (usize, usize),
    start: i64,
    prefix: Option<&[Kv]>,
) -> R<(Tensor<f32>, Vec<Kv>)> {
    let w = weights();
    let mut x = gather_axis0(&w.embed, &ivec(ids), bf16_to_f32)?;
    let cs = rope_tables(start, br.1)?;
    let mut kvs = Vec::new();
    for (i, l) in w.layers.iter().enumerate() {
        let (nx, k, v) = layer(l, &x, br, &cs, prefix.and_then(|p| p.get(i)))?;
        x = nx;
        kvs.push((k, v));
    }
    Ok((x, kvs))
}

/// 隠れ状態 `[K*M,H]`（選択肢 k の位置 j は cand[k][j] を予測する行）から選択肢間の確率 `[K]`。
fn finish(h: &Tensor<f32>) -> R<Vec<f64>> {
    let w = weights();
    let logits = gemm_trans_b(&rmsnorm(h, &w.fnorm)?, &w.embed, None, bf16_to_f32, None)?;
    let lp = log_softmax_last_axis(&reshape(logits, &dims(&[K, M, VOCAB]))?)?;
    let cand = i64t(&[K, M, 1], CAND.iter().flatten().copied().collect());
    let picked = gather_elements_last_axis(&lp, &cand)?;
    let wt = f32t(&[K, M, 1], WEIGHT.iter().flatten().copied().collect());
    let tot = reduce_sum(&mul(&picked, &wt)?, &ivec(&[1, 2]))?;
    Ok(softmax_last_axis_f64(&tot)?.into_data())
}

/// `axis` 方向の `[from, to)` を切り出す。
fn rows<T: Copy + Default + Send + Sync>(
    x: &Tensor<T>,
    axis: i64,
    from: usize,
    to: usize,
) -> R<Tensor<T>> {
    slice(
        x,
        &ivec(&[from as i64]),
        &ivec(&[to as i64]),
        Some(&ivec(&[axis])),
        None,
    )
}

fn bounded(ids: &[i64]) -> R<&[i64]> {
    // 空・語彙外は拒否する（前処理が守る前提だが、直接呼ばれても panic させない）。
    if ids.is_empty() || ids.iter().any(|&i| !(0..VOCAB as i64).contains(&i)) {
        return Err(OpError::InvalidAttribute);
    }
    Ok(ids)
}

/// KV 使い回し採点: prefix を 1 回流して K・V を取り、K 件の選択肢ブロックが共有する。
pub fn kv_reuse_scores(ids: &[i64]) -> R<Vec<f64>> {
    let t = bounded(ids)?.len();
    let (px, kvs) = stack(ids, (1, t), 0, None)?;
    let flat: Vec<i64> = CAND.iter().flatten().copied().collect();
    let (bx, _) = stack(&flat, (K, M), t as i64, Some(&kvs))?;
    // prefix の最終行を K 件に広げ、ブロックの先頭 M-1 行と連結する。
    let last = reshape(rows(&px, 0, t - 1, t)?, &dims(&[1, 1, HIDDEN]))?;
    let last = expand(&last, &dims(&[K, 1, HIDDEN]))?;
    let head = rows(&reshape(bx, &dims(&[K, M, HIDDEN]))?, 1, 0, M - 1)?;
    let h = reshape(concat(&[&last, &head], 1)?, &dims(&[K * M, HIDDEN]))?;
    finish(&h)
}

/// 比較用の素朴な採点: 選択肢ごとに prefix+cand の全系列を 1 回流す（KV を使い回さない）。
pub fn naive_full_scores(ids: &[i64]) -> R<Vec<f64>> {
    let t = bounded(ids)?.len();
    let mut picked: Vec<Tensor<f32>> = Vec::new();
    for c in &CAND {
        let seq: Vec<i64> = ids.iter().chain(c.iter()).copied().collect();
        let (x, _) = stack(&seq, (1, t + M), 0, None)?;
        // cand[j] を予測する行は位置 t-1+j。
        picked.push(rows(&x, 0, t - 1, t + M - 1)?);
    }
    let refs: Vec<&Tensor<f32>> = picked.iter().collect();
    finish(&concat(&refs, 0)?)
}

/// 入力のバイトを `% (VOCAB-1)` で id にし、先頭 [`MAX_T`] に切り詰める。空なら `[0]`。
pub fn bytes_to_ids(input: &str) -> Vec<i64> {
    if input.is_empty() {
        return vec![0];
    }
    input
        .bytes()
        .take(MAX_T)
        .map(|b| i64::from(b) % (VOCAB as i64 - 1))
        .collect()
}

/// 参照順伝播のバックエンド。`&self` だけの純粋計算で、呼び出し間の状態を持たない（REQ-28）。
pub struct QwenRefBackend;

impl ScoringBackend for QwenRefBackend {
    fn scores(&self, ids: &TokenIds) -> Result<Vec<f64>, BackendError> {
        kv_reuse_scores(ids.as_slice()).map_err(|_| BackendError::Failed)
    }
    /// 計算は小さく有界（`MAX_T`）で打ち切り対象の反復を持たないため、`scores` と同じ結果を返す。
    fn scores_limited(
        &self,
        ids: &TokenIds,
        _limit: std::time::Duration,
    ) -> Result<Vec<f64>, BackendError> {
        self.scores(ids)
    }
}
