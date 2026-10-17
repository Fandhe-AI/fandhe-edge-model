//! Qwen2 採点グラフ用の汎用演算カーネル（REQ-41・REQ-32・TASK-41.7-2・#416）。
//!
//! # 役割
//!
//! 共有型（[`Tensor`]・[`OpError`]・ブロードキャスト）と形状操作（[`shape`]。#417）を提供する。
//! `docs/design/qwen2-scoring-onnx-ops.md` の演算一覧のうち、行列演算と要素ごとの演算
//! （MatMul・Gemm・Add・Sub・Mul・Sqrt・Reciprocal・Cast）は #416 の担当で、本差分では未実装
//! （後続実装）。正規化と活性化は [`norm_act`]（#418）、マスクと RoPE は #419 の担当。
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

pub mod norm_act;
pub mod shape;

use std::fmt;

/// テンソルの rank の上限。GQA の `[1,nkv,grp,T,hd]` までを許す（設計書 4 節）。
pub const MAX_RANK: usize = 5;

// ponytail: 暫定値（f32 で 1 GiB）。種類別の上限は #581 で確定する。
/// 1 つのテンソルの要素数の上限。確保の前に必ず照合する（REQ-39）。
pub const MAX_ELEMENTS: usize = 1 << 28;

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

/// 2 入力の要素ごとの演算を多方向ブロードキャストで適用する。出力型 `U` は入力と異なってよい。
pub fn broadcast_map2<A: Copy, B: Copy, U>(
    a: &Tensor<A>,
    b: &Tensor<B>,
    mut f: impl FnMut(A, B) -> Result<U, OpError>,
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
            broadcast_shape(&[1 << 15, 1], &[1, 1 << 15]),
            Err(OpError::LimitExceeded)
        );
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
}
