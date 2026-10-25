//! Qwen2 採点グラフ用の汎用演算カーネル（REQ-41・REQ-32・TASK-41.7-2・#416）。
//!
//! # 役割
//!
//! 共有型（[`Tensor`]・[`OpError`]・ブロードキャスト）と形状操作（[`shape`]。#417）を提供する。
//! `docs/design/qwen2-scoring-onnx-ops.md` の演算一覧のうち、行列演算と要素ごとの演算
//! （MatMul・Gemm・Add・Sub・Mul・Sqrt・Reciprocal・Cast）は [`arith`]（#416）。正規化と活性化は [`norm_act`]（#418）、マスク・Where・RoPE の表の構成演算（Range・Greater・Where・Sin・Cos）は [`mask_rope`]（#419）。
//!
//! # 契約
//!
//! - 実行器へは未接続（グラフの読み込み・演算×属性×dtype×rank の許可リスト照合は別の担当。
//!   未実装を実装済みとして扱わない）。カーネルは属性が検証済みである前提で、設計書に固定された
//!   意味だけを実装する
//! - 形状・要素数は `checked_mul` で計算し、[`MAX_RANK`]・[`MAX_ELEMENTS`] と照合してから確保する
//!   （REQ-39）。あふれ・上限超過は panic せず [`OpError`] で返す。エラーに値は含めない
//! - 縮約は単一スレッドで添字の昇順に f32 で加算し、スレッド数に依存させない（REQ-28）
//! - 許容差: f32 の演算に 1e-9 は適用しない（1e-9 は f64 計算の場合）。テストは 2 進で正確な値の
//!   厳密一致で比べる

pub mod arith;
pub mod mask_rope;
pub mod norm_act;
pub mod shape;

use std::fmt;

/// テンソルの rank の上限。GQA の `[1,nkv,grp,T,hd]` までを許す（設計書 4 節）。
pub const MAX_RANK: usize = 5;

/// 1 つのテンソルの要素数の上限。確保の前に必ず照合する（REQ-39）。
///
/// `1 << 30`（f32 換算で 4 GiB 相当）。7B の埋め込み（152064 × 3584 ≒ 5.45e8）が収まる値
/// （オーナー判断 2026-10-25・#654・REQ-39・REQ-41）。確保前の照合上限であり、bf16 の重みは
/// `u16` のまま保持してカーネル内で展開するため全量の f32 化はしない（#581）。
/// 本モジュールのカーネルは `qwen2_lora` の採点実行器（`scoring.rs`）と外部データの検査
/// （`proto.rs`。外部データを拒否する C1・C3 には効かない）からだけ使われ、C1・C3 の上限は緩めない。
pub const MAX_ELEMENTS: usize = 1 << 30;

/// カーネルのエラー。終了コードへの写像は CLI 側の責務。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum OpError {
    /// 形状の不整合（ブロードキャスト不可・内側の次元の不一致・rank 範囲外・data 長の不一致）。
    ShapeMismatch,
    /// 要素数が上限を超える、または形状の積があふれる。
    LimitExceeded,
    /// INT64 の加算があふれた。
    IntegerOverflow,
    /// Gather 系の index が範囲外（負値を含む）。
    IndexOutOfRange,
    /// 属性・補助入力（axes・perm・shape 等）が不正、または本実装が扱わない値。
    InvalidAttribute,
    /// Softmax 系の入力に NaN・±inf がある（fail-closed。不正な重みを確率として出さない）。
    NonFinite,
    /// 呼び出し側が渡した期限を超えた（積和ループ内の打ち切り。REQ-39）。
    DeadlineExceeded,
}

impl OpError {
    /// 機械可読な英語コード。
    pub fn code(&self) -> &'static str {
        match self {
            Self::ShapeMismatch => "op_shape_mismatch",
            Self::LimitExceeded => "limit_exceeded",
            Self::IntegerOverflow => "op_integer_overflow",
            Self::IndexOutOfRange => "op_index_out_of_range",
            Self::InvalidAttribute => "op_invalid_attribute",
            Self::NonFinite => "op_non_finite",
            Self::DeadlineExceeded => "op_deadline_exceeded",
        }
    }
}

impl fmt::Display for OpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}

impl std::error::Error for OpError {}

/// 形状の積を checked で求め、[`MAX_ELEMENTS`] と照合する。
pub fn element_count(shape: &[usize]) -> Result<usize, OpError> {
    let mut n: usize = 1;
    for &d in shape {
        n = n.checked_mul(d).ok_or(OpError::LimitExceeded)?;
    }
    if n > MAX_ELEMENTS {
        return Err(OpError::LimitExceeded);
    }
    Ok(n)
}

/// 行優先のテンソル（形状と平坦なデータ）。rank 0 は要素 1 のスカラー。
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor<T> {
    shape: Vec<usize>,
    data: Vec<T>,
}

impl<T> Tensor<T> {
    /// 形状とデータから作る。rank ≤ [`MAX_RANK`]・要素数が data 長と一致することを確認する。
    pub fn new(shape: Vec<usize>, data: Vec<T>) -> Result<Self, OpError> {
        if shape.len() > MAX_RANK {
            return Err(OpError::ShapeMismatch);
        }
        if element_count(&shape)? != data.len() {
            return Err(OpError::ShapeMismatch);
        }
        Ok(Self { shape, data })
    }

    /// 形状。
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// 行優先のデータ。
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// データを取り出す。
    pub fn into_data(self) -> Vec<T> {
        self.data
    }
}

/// 右揃えで `i` 番目（0 起点・出力側）の次元を取り出す。範囲外は 1。
fn dim_from_right(s: &[usize], rank: usize, i: usize) -> usize {
    let pad = rank - s.len();
    if i < pad {
        1
    } else {
        s.get(i - pad).copied().unwrap_or(1)
    }
}

/// numpy 方式のブロードキャスト後の形状。右から揃え、各次元は等しいか片方が 1。
/// 出力の rank は [`MAX_RANK`] 以下、要素数は [`MAX_ELEMENTS`] 以下（確保はしない）。
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, OpError> {
    let rank = a.len().max(b.len());
    if rank > MAX_RANK {
        return Err(OpError::ShapeMismatch);
    }
    let mut out = Vec::with_capacity(rank);
    for i in 0..rank {
        let da = dim_from_right(a, rank, i);
        let db = dim_from_right(b, rank, i);
        out.push(match (da, db) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            _ => return Err(OpError::ShapeMismatch),
        });
    }
    element_count(&out)?;
    Ok(out)
}

/// 入力形状を出力 rank へ右揃えし、ブロードキャストする次元の stride を 0 にした stride 列。
/// `shape` は出力へブロードキャスト可能であること。
pub(crate) fn broadcast_strides(shape: &[usize], rank: usize) -> Vec<usize> {
    let mut strides = vec![0usize; rank];
    let mut acc = 1usize;
    for (i, &d) in shape.iter().enumerate().rev() {
        let pos = rank - shape.len() + i;
        if let Some(s) = strides.get_mut(pos) {
            *s = if d == 1 { 0 } else { acc };
        }
        acc = acc.saturating_mul(d);
    }
    strides
}

/// 多重添字 `idx` を行優先で 1 進める。
pub(crate) fn advance_index(idx: &mut [usize], shape: &[usize]) {
    for k in (0..idx.len()).rev() {
        let lim = shape.get(k).copied().unwrap_or(1);
        if let Some(v) = idx.get_mut(k) {
            *v += 1;
            if *v < lim {
                return;
            }
            *v = 0;
        }
    }
}

/// 入力が最内ブロック内でどう並ぶか。`Contig` はブロック内で連続、`Const` は全要素が同一位置。
#[derive(Clone, Copy, PartialEq)]
enum InnerMode {
    Contig,
    Const,
}

/// 出力の末尾次元から、入力が連続または定数であり続ける範囲を伸ばした境界 `k` と、
/// 各入力の方式を返す。`out_shape[k..]` が最内ブロック（REQ-28・#637）。
fn inner_block(
    out_shape: &[usize],
    sa: &[usize],
    sb: &[usize],
) -> (usize, Option<InnerMode>, Option<InnerMode>) {
    let (mut ma, mut mb) = (None, None);
    let mut acc = 1usize;
    let mut k = out_shape.len();
    // 入力 1 本が次元 `j` を現在の方式のまま含められるか。含められなければ None。
    let fit =
        |m: Option<InnerMode>, d: usize, st: usize, acc: usize| -> Option<Option<InnerMode>> {
            if d == 1 {
                Some(m)
            } else if st == 0 && m != Some(InnerMode::Contig) {
                Some(Some(InnerMode::Const))
            } else if st == acc && m != Some(InnerMode::Const) {
                Some(Some(InnerMode::Contig))
            } else {
                None
            }
        };
    while k > 0 {
        let j = k - 1;
        let d = out_shape.get(j).copied().unwrap_or(1);
        let st_a = sa.get(j).copied().unwrap_or(0);
        let st_b = sb.get(j).copied().unwrap_or(0);
        match (fit(ma, d, st_a, acc), fit(mb, d, st_b, acc)) {
            (Some(na), Some(nb)) => {
                ma = na;
                mb = nb;
                acc = acc.saturating_mul(d);
                k = j;
            }
            _ => break,
        }
    }
    (k, ma, mb)
}

/// 2 入力の要素ごとの演算を多方向ブロードキャストで適用する。出力型 `U` は入力と異なってよい。
/// 同形状・スカラー・末尾ブロードキャストなど、最内ブロックが連続または定数になる形は
/// ブロック単位のループで処理する。要素の適用順と各要素の計算は汎用経路と同一（REQ-28・#637）。
/// 出力要素数が [`arith::PARALLEL_MIN_WORK`] 以上なら、出力を連続した要素範囲に分けて並列に計算する。
/// 各要素は独立に計算されるので結果はスレッド数に依存しない（REQ-28・#648）。
pub fn broadcast_map2<A: Copy + Sync, B: Copy + Sync, U: Copy + Default + Send>(
    a: &Tensor<A>,
    b: &Tensor<B>,
    f: impl Fn(A, B) -> Result<U, OpError> + Sync,
) -> Result<Tensor<U>, OpError> {
    let out_shape = broadcast_shape(a.shape(), b.shape())?;
    let n = element_count(&out_shape)?;
    let rank = out_shape.len();
    let sa = broadcast_strides(a.shape(), rank);
    let sb = broadcast_strides(b.shape(), rank);
    if n == 0 {
        return Tensor::new(out_shape, Vec::new());
    }
    let (k, ma, mb) = inner_block(&out_shape, &sa, &sb);
    let len = element_count(out_shape.get(k..).unwrap_or(&[]))?;
    let outer = out_shape.get(..k).unwrap_or(&[]);
    let out = arith::par_collect(n, 1, 1, |range, out| {
        let mut slots = out.iter_mut();
        let mut put = |v: U| -> Result<(), OpError> {
            *slots.next().ok_or(OpError::ShapeMismatch)? = v;
            Ok(())
        };
        // 範囲の先頭が属するブロックと、そのブロックの出力先頭の多重添字
        let mut idx = vec![0usize; k];
        let mut rem = range.start / len;
        for (i, &d) in idx.iter_mut().zip(outer).rev() {
            *i = rem % d;
            rem /= d;
        }
        let mut pos = range.start - range.start % len;
        while pos < range.end {
            let (lo, hi) = (range.start.max(pos) - pos, range.end.min(pos + len) - pos);
            let (mut pa, mut pb) = (0usize, 0usize);
            for (j, &i) in idx.iter().enumerate() {
                pa += i * sa.get(j).copied().unwrap_or(0);
                pb += i * sb.get(j).copied().unwrap_or(0);
            }
            if ma == Some(InnerMode::Contig) {
                let xs = a
                    .data()
                    .get(pa + lo..pa + hi)
                    .ok_or(OpError::ShapeMismatch)?;
                if mb == Some(InnerMode::Contig) {
                    let ys = b
                        .data()
                        .get(pb + lo..pb + hi)
                        .ok_or(OpError::ShapeMismatch)?;
                    for (&x, &y) in xs.iter().zip(ys) {
                        put(f(x, y)?)?;
                    }
                } else {
                    let y = *b.data().get(pb).ok_or(OpError::ShapeMismatch)?;
                    for &x in xs {
                        put(f(x, y)?)?;
                    }
                }
            } else {
                let x = *a.data().get(pa).ok_or(OpError::ShapeMismatch)?;
                if mb == Some(InnerMode::Contig) {
                    let ys = b
                        .data()
                        .get(pb + lo..pb + hi)
                        .ok_or(OpError::ShapeMismatch)?;
                    for &y in ys {
                        put(f(x, y)?)?;
                    }
                } else {
                    let y = *b.data().get(pb).ok_or(OpError::ShapeMismatch)?;
                    for _ in lo..hi {
                        put(f(x, y)?)?;
                    }
                }
            }
            pos += len;
            advance_index(&mut idx, outer);
        }
        Ok(())
    })?;
    Tensor::new(out_shape, out)
}

/// 参照実装（要素ごとに多重添字から位置を計算する汎用経路）。高速経路との一致照合専用。
#[cfg(test)]
fn broadcast_map2_reference<A: Copy, B: Copy, U>(
    a: &Tensor<A>,
    b: &Tensor<B>,
    f: impl Fn(A, B) -> Result<U, OpError>,
) -> Result<Tensor<U>, OpError> {
    let out_shape = broadcast_shape(a.shape(), b.shape())?;
    let n = element_count(&out_shape)?;
    let rank = out_shape.len();
    let sa = broadcast_strides(a.shape(), rank);
    let sb = broadcast_strides(b.shape(), rank);
    let mut out = Vec::with_capacity(n);
    let mut idx = vec![0usize; rank];
    for _ in 0..n {
        let (mut pa, mut pb) = (0usize, 0usize);
        for (k, &i) in idx.iter().enumerate() {
            pa += i * sa.get(k).copied().unwrap_or(0);
            pb += i * sb.get(k).copied().unwrap_or(0);
        }
        let x = *a.data().get(pa).ok_or(OpError::ShapeMismatch)?;
        let y = *b.data().get(pb).ok_or(OpError::ShapeMismatch)?;
        out.push(f(x, y)?);
        advance_index(&mut idx, &out_shape);
    }
    Tensor::new(out_shape, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-39・REQ-41・#654: 7B の埋め込み形状は通り、`1 << 30` 超は確保前に拒否される。
    #[test]
    fn req39_element_count_boundary_for_7b() {
        assert_eq!(element_count(&[152064, 3584]), Ok(152064 * 3584));
        assert_eq!(element_count(&[1 << 30]), Ok(1 << 30));
        assert_eq!(element_count(&[(1 << 30) + 1]), Err(OpError::LimitExceeded));
        assert_eq!(
            element_count(&[1 << 16, 1 << 15]),
            Err(OpError::LimitExceeded)
        );
    }

    #[test]
    fn req41_broadcast_shape_cases() {
        assert_eq!(broadcast_shape(&[2, 3], &[3]).unwrap(), vec![2, 3]);
        assert_eq!(broadcast_shape(&[2, 1, 4], &[3, 1]).unwrap(), vec![2, 3, 4]);
        assert_eq!(broadcast_shape(&[], &[2]).unwrap(), vec![2]);
        assert_eq!(broadcast_shape(&[2, 3], &[4]), Err(OpError::ShapeMismatch));
        assert_eq!(
            broadcast_shape(&[1, 1, 1, 1, 1, 1], &[1]),
            Err(OpError::ShapeMismatch)
        );
        // 確保せずに上限超過を判定する
        assert_eq!(
            broadcast_shape(&[1 << 16, 1], &[1, 1 << 15]),
            Err(OpError::LimitExceeded)
        );
    }

    /// 高速経路が汎用経路とビット一致する（REQ-28・#637）。
    #[test]
    fn req28_broadcast_map2_fast_paths_match_reference_637() {
        let mk = |shape: &[usize]| {
            let n: usize = shape.iter().product();
            let d: Vec<f32> = (0..n)
                .map(|i| (i as f32 * 0.37 - 5.0).sin() * 1.7)
                .collect();
            Tensor::new(shape.to_vec(), d).unwrap()
        };
        let cases: &[(&[usize], &[usize])] = &[
            (&[4, 6], &[4, 6]),
            (&[4, 6], &[]),
            (&[], &[4, 6]),
            (&[4, 6], &[1]),
            (&[4, 6], &[6]),
            (&[6], &[4, 6]),
            (&[4, 6], &[4, 1]),
            (&[4, 1], &[4, 6]),
            (&[2, 3, 4], &[3, 1]),
            (&[2, 3, 4], &[2, 1, 4]),
            (&[2, 1, 4], &[3, 1]),
            (&[2, 3, 1], &[1, 1, 4]),
            (&[1, 4], &[3, 1]),
            (&[0, 3], &[3]),
        ];
        for (sa, sb) in cases {
            let (a, b) = (mk(sa), mk(sb));
            let fast = broadcast_map2(&a, &b, |x, y| Ok(x * y + x)).unwrap();
            let slow = broadcast_map2_reference(&a, &b, |x, y| Ok(x * y + x)).unwrap();
            assert_eq!(fast.shape(), slow.shape(), "{sa:?} {sb:?}");
            let fb: Vec<u32> = fast.data().iter().map(|v| v.to_bits()).collect();
            let sb_: Vec<u32> = slow.data().iter().map(|v| v.to_bits()).collect();
            assert_eq!(fb, sb_, "{sa:?} {sb:?}");
        }
    }

    #[test]
    fn req41_tensor_new_validates() {
        assert_eq!(
            Tensor::new(vec![2, 2], vec![1.0f32; 3]),
            Err(OpError::ShapeMismatch)
        );
        assert_eq!(
            Tensor::<f32>::new(vec![usize::MAX, 2], vec![]),
            Err(OpError::LimitExceeded)
        );
        assert_eq!(Tensor::new(vec![], vec![7i64]).unwrap().data(), &[7]);
        assert_eq!(OpError::LimitExceeded.code(), "limit_exceeded");
        assert_eq!(OpError::NonFinite.code(), "op_non_finite");
    }

    /// 並列版（範囲分割）が汎用経路とビット一致する。範囲がブロックの途中で切れる形を含む（REQ-28・#648）。
    #[test]
    fn req28_broadcast_map2_parallel_matches_reference_648() {
        let mk = |shape: &[usize]| {
            let n: usize = shape.iter().product();
            let d: Vec<f32> = (0..n)
                .map(|i| (i as f32 * 0.11 - 3.0).cos() * 2.3)
                .collect();
            Tensor::new(shape.to_vec(), d).unwrap()
        };
        let cases: &[(&[usize], &[usize])] = &[
            (&[156, 896], &[156, 896]),
            (&[156, 896], &[896]),
            (&[156, 896], &[156, 1]),
            (&[156, 896], &[]),
            (&[2, 3, 7, 11, 13], &[7, 11, 13]),
            (&[2, 3, 7, 11, 13], &[3, 1, 1, 13]),
            (&[2, 3, 1, 1], &[1, 1, 7, 5]),
        ];
        for (sa, sb) in cases {
            let (a, b) = (mk(sa), mk(sb));
            let slow = broadcast_map2_reference(&a, &b, |x, y| Ok(x * y + x)).unwrap();
            let sbits: Vec<u32> = slow.data().iter().map(|v| v.to_bits()).collect();
            for t in [1, 3, 7] {
                let fast = arith::with_threads(t, || broadcast_map2(&a, &b, |x, y| Ok(x * y + x)))
                    .unwrap();
                let fb: Vec<u32> = fast.data().iter().map(|v| v.to_bits()).collect();
                assert_eq!(fb, sbits, "{sa:?} {sb:?} threads={t}");
            }
            // しきい値を超える実スレッド数（既定）でも一致する
            let fast = broadcast_map2(&a, &b, |x, y| Ok(x * y + x)).unwrap();
            let fb: Vec<u32> = fast.data().iter().map(|v| v.to_bits()).collect();
            assert_eq!(fb, sbits, "{sa:?} {sb:?}");
        }
    }
}
