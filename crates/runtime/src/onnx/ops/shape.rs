//! 形状操作の演算カーネル（REQ-41・REQ-32・TASK-41.7-3・#417）。
//!
//! # 役割
//!
//! `docs/design/qwen2-scoring-onnx-ops.md` 3 節の「担当 #417」の 10 演算（opset 13）
//! Shape・Gather・GatherElements・Reshape・Transpose・Unsqueeze・Concat・Slice・Split・Expand を
//! 実装する。共有型（[`Tensor`]・[`OpError`]・ブロードキャスト）は親モジュールのものを使う。
//!
//! # 契約
//!
//! - 実行器へは未接続。グラフの読み込みと演算×属性×dtype×rank の許可リスト照合は別の担当で、
//!   未実装を実装済みとして扱わない。ここは属性が設計書の範囲に収まる前提で意味だけを実装する
//! - opset 13 より意図して狭めた点: Gather は axis=0 のみ・負の index を拒否（語彙外 id の
//!   読み替えを避ける fail-closed）、GatherElements は axis=-1 のみ、Slice は steps が 1 のみ、
//!   Transpose は perm 必須、Shape は start・end 属性なし
//! - index・形状は信頼できないモデルや入力 id から来るため、checked 演算と `try_into` で扱い、
//!   出力形状を [`element_count`] で上限照合してから確保する（REQ-39）。panic しない。
//!   エラーに値は含めない
//! - 値を並べ替えるだけで算術をしないため、テストは厳密一致で比べる（1e-9 は適用しない）

use super::arith::par_collect;
use super::{
    MAX_RANK, OpError, Tensor, advance_index, broadcast_shape, broadcast_strides, element_count,
};

// 値の根拠: Qwen2 の採点グラフの Concat は入力数が数個、Split は q・k・v の 3 個程度（設計書）で、
// 64 は十分な余裕を持つ上限。1.5B 向けに据え置く（#581）。
/// Concat の入力数の上限。
pub const MAX_CONCAT_INPUTS: usize = 64;
/// Split の出力数の上限。
pub const MAX_SPLIT_OUTPUTS: usize = 64;

/// 負値を rank 基準で正規化する。範囲外は [`OpError::InvalidAttribute`]。
pub(crate) fn normalize_axis(axis: i64, rank: usize) -> Result<usize, OpError> {
    let r = i64::try_from(rank).map_err(|_| OpError::InvalidAttribute)?;
    let a = if axis < 0 {
        axis.checked_add(r).ok_or(OpError::InvalidAttribute)?
    } else {
        axis
    };
    if a < 0 || a >= r {
        return Err(OpError::InvalidAttribute);
    }
    usize::try_from(a).map_err(|_| OpError::InvalidAttribute)
}

fn row_strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![0usize; shape.len()];
    let mut acc = 1usize;
    for (k, &d) in shape.iter().enumerate().rev() {
        if let Some(v) = s.get_mut(k) {
            *v = acc;
        }
        acc = acc.saturating_mul(d);
    }
    s
}

/// 多重添字ごとに `base + Σ idx[k] * strides[k]` の位置を読んで集める共通ループ。
/// 最内次元は行単位で写す（stride 1 は連続コピー、0 は同一要素の複製）。外側の位置計算は
/// 行ごとに 1 回で、要素の並びは要素ごとの汎用経路と同一（REQ-28・#637）。
/// 出力の行は [`par_collect`] で連続範囲に分けて並列に写す（要素は独立で結果は不変。REQ-28・#648）。
fn gather_by_strides<T: Copy + Default + Send + Sync>(
    data: &[T],
    out_shape: Vec<usize>,
    strides: &[usize],
    base: usize,
) -> Result<Tensor<T>, OpError> {
    let n = element_count(&out_shape)?;
    if n == 0 {
        return Tensor::new(out_shape, Vec::new());
    }
    let Some((&len, outer)) = out_shape.split_last() else {
        let v = *data.get(base).ok_or(OpError::ShapeMismatch)?;
        return Tensor::new(out_shape, vec![v]);
    };
    let inner = strides.get(outer.len()).copied().unwrap_or(0);
    let out = par_collect(n / len, len, 1, |rows, out| {
        // 範囲の先頭の行の多重添字
        let mut idx = vec![0usize; outer.len()];
        let mut rem = rows.start;
        for (i, &d) in idx.iter_mut().zip(outer).rev() {
            *i = rem % d;
            rem /= d;
        }
        for orow in out.chunks_mut(len) {
            let mut p = base;
            for (k, &i) in idx.iter().enumerate() {
                p += i * strides.get(k).copied().unwrap_or(0);
            }
            match inner {
                1 => orow.copy_from_slice(data.get(p..p + len).ok_or(OpError::ShapeMismatch)?),
                0 => orow.fill(*data.get(p).ok_or(OpError::ShapeMismatch)?),
                s => {
                    for (j, o) in orow.iter_mut().enumerate() {
                        *o = *data.get(p + j * s).ok_or(OpError::ShapeMismatch)?;
                    }
                }
            }
            advance_index(&mut idx, outer);
        }
        Ok(())
    })?;
    Tensor::new(out_shape, out)
}

/// Shape: 形状を 1 次元 INT64 で返す。
pub fn shape<T>(x: &Tensor<T>) -> Result<Tensor<i64>, OpError> {
    let mut v = Vec::with_capacity(x.shape().len());
    for &d in x.shape() {
        v.push(i64::try_from(d).map_err(|_| OpError::LimitExceeded)?);
    }
    Tensor::new(vec![v.len()], v)
}

/// Gather（axis=0 固定）。出力形状は `indices.shape ++ data.shape[1..]`。
/// `f` は取り出した要素への変換で、埋め込みでは bf16 → f32 を渡して参照行だけを展開する
/// （設計書 2 節の融合実行）。恒等でよければ `|v| v`。index は 0 以上 先頭次元未満のみ。
pub fn gather_axis0<T: Copy, U>(
    data: &Tensor<T>,
    indices: &Tensor<i64>,
    f: impl Fn(T) -> U,
) -> Result<Tensor<U>, OpError> {
    let (dim0, rest) = data.shape().split_first().ok_or(OpError::ShapeMismatch)?;
    let row = element_count(rest)?;
    let mut out_shape = indices.shape().to_vec();
    out_shape.extend_from_slice(rest);
    // 出力 rank の上限は確保・走査より前に検証する（REQ-39）。
    if out_shape.len() > MAX_RANK {
        return Err(OpError::ShapeMismatch);
    }
    let n = element_count(&out_shape)?;
    let mut out = Vec::with_capacity(n);
    for &i in indices.data() {
        let i = usize::try_from(i).map_err(|_| OpError::IndexOutOfRange)?;
        if i >= *dim0 {
            return Err(OpError::IndexOutOfRange);
        }
        let start = i * row;
        let src = data
            .data()
            .get(start..start + row)
            .ok_or(OpError::ShapeMismatch)?;
        out.extend(src.iter().map(|&v| f(v)));
    }
    Tensor::new(out_shape, out)
}

/// GatherElements（axis=-1 固定）。rank は同じで、最後以外の次元は一致を要求する。
pub fn gather_elements_last_axis<T: Copy>(
    data: &Tensor<T>,
    indices: &Tensor<i64>,
) -> Result<Tensor<T>, OpError> {
    let (dl, dlead) = data.shape().split_last().ok_or(OpError::ShapeMismatch)?;
    let (il, ilead) = indices.shape().split_last().ok_or(OpError::ShapeMismatch)?;
    if dlead != ilead {
        return Err(OpError::ShapeMismatch);
    }
    let n = element_count(indices.shape())?;
    let mut out = Vec::with_capacity(n);
    if *il > 0 {
        for (o, chunk) in indices.data().chunks(*il).enumerate() {
            for &i in chunk {
                let i = usize::try_from(i).map_err(|_| OpError::IndexOutOfRange)?;
                if i >= *dl {
                    return Err(OpError::IndexOutOfRange);
                }
                let p = o * *dl + i;
                out.push(*data.data().get(p).ok_or(OpError::ShapeMismatch)?);
            }
        }
    }
    Tensor::new(indices.shape().to_vec(), out)
}

/// Reshape。所有権ごと受け取りデータはコピーしない。0 は入力の同位置の次元、-1 は最大 1 つ。
pub fn reshape<T>(x: Tensor<T>, shape: &Tensor<i64>) -> Result<Tensor<T>, OpError> {
    if shape.data().len() > MAX_RANK {
        return Err(OpError::ShapeMismatch);
    }
    let total = x.data().len();
    let mut dims: Vec<usize> = Vec::with_capacity(shape.data().len());
    let mut infer: Option<usize> = None;
    let mut known: usize = 1;
    for (i, &v) in shape.data().iter().enumerate() {
        let d = match v {
            -1 => {
                if infer.replace(i).is_some() {
                    return Err(OpError::InvalidAttribute);
                }
                1
            }
            0 => *x.shape().get(i).ok_or(OpError::InvalidAttribute)?,
            v if v > 0 => usize::try_from(v).map_err(|_| OpError::InvalidAttribute)?,
            _ => return Err(OpError::InvalidAttribute),
        };
        known = known.checked_mul(d).ok_or(OpError::LimitExceeded)?;
        dims.push(d);
    }
    if let Some(i) = infer {
        if known == 0 {
            return Err(OpError::InvalidAttribute);
        }
        if !total.is_multiple_of(known) {
            return Err(OpError::ShapeMismatch);
        }
        if let Some(d) = dims.get_mut(i) {
            *d = total / known;
        }
    }
    Tensor::new(dims, x.into_data())
}

/// Transpose。`perm` は `0..rank` の順列であること。
pub fn transpose<T: Copy + Default + Send + Sync>(
    x: &Tensor<T>,
    perm: &[i64],
) -> Result<Tensor<T>, OpError> {
    let rank = x.shape().len();
    if perm.len() != rank {
        return Err(OpError::InvalidAttribute);
    }
    let mut seen = vec![false; rank];
    let mut p = Vec::with_capacity(rank);
    for &a in perm {
        let a = usize::try_from(a).map_err(|_| OpError::InvalidAttribute)?;
        let s = seen.get_mut(a).ok_or(OpError::InvalidAttribute)?;
        if std::mem::replace(s, true) {
            return Err(OpError::InvalidAttribute);
        }
        p.push(a);
    }
    let in_strides = row_strides(x.shape());
    let mut out_shape = Vec::with_capacity(rank);
    let mut strides = Vec::with_capacity(rank);
    for &a in &p {
        out_shape.push(x.shape().get(a).copied().ok_or(OpError::InvalidAttribute)?);
        strides.push(
            in_strides
                .get(a)
                .copied()
                .ok_or(OpError::InvalidAttribute)?,
        );
    }
    gather_by_strides(x.data(), out_shape, &strides, 0)
}

/// Unsqueeze。`axes` は出力 rank 基準（負値可）で、重複と範囲外は拒否する。
pub fn unsqueeze<T>(x: Tensor<T>, axes: &Tensor<i64>) -> Result<Tensor<T>, OpError> {
    let out_rank = x.shape().len() + axes.data().len();
    if out_rank > MAX_RANK {
        return Err(OpError::ShapeMismatch);
    }
    let mut ins = vec![false; out_rank];
    for &a in axes.data() {
        let a = normalize_axis(a, out_rank)?;
        let s = ins.get_mut(a).ok_or(OpError::InvalidAttribute)?;
        if std::mem::replace(s, true) {
            return Err(OpError::InvalidAttribute);
        }
    }
    let mut src = x.shape().iter().copied();
    let mut shape = Vec::with_capacity(out_rank);
    for &is_new in &ins {
        shape.push(if is_new {
            1
        } else {
            src.next().ok_or(OpError::ShapeMismatch)?
        });
    }
    Tensor::new(shape, x.into_data())
}

/// Concat。rank は全入力で同じで、axis 以外の次元は一致すること。
pub fn concat<T: Copy>(inputs: &[&Tensor<T>], axis: i64) -> Result<Tensor<T>, OpError> {
    if inputs.len() > MAX_CONCAT_INPUTS {
        return Err(OpError::LimitExceeded);
    }
    let first = inputs.first().ok_or(OpError::ShapeMismatch)?;
    let rank = first.shape().len();
    let ax = normalize_axis(axis, rank).map_err(|_| OpError::ShapeMismatch)?;
    let mut out_shape = first.shape().to_vec();
    let mut sum: usize = 0;
    for t in inputs {
        if t.shape().len() != rank {
            return Err(OpError::ShapeMismatch);
        }
        for (k, (&a, &b)) in t.shape().iter().zip(first.shape()).enumerate() {
            if k != ax && a != b {
                return Err(OpError::ShapeMismatch);
            }
        }
        sum = sum
            .checked_add(t.shape().get(ax).copied().ok_or(OpError::ShapeMismatch)?)
            .ok_or(OpError::LimitExceeded)?;
    }
    if let Some(d) = out_shape.get_mut(ax) {
        *d = sum;
    }
    let n = element_count(&out_shape)?;
    let outer = element_count(out_shape.get(..ax).ok_or(OpError::ShapeMismatch)?)?;
    let inner = element_count(out_shape.get(ax + 1..).ok_or(OpError::ShapeMismatch)?)?;
    let mut out = Vec::with_capacity(n);
    for o in 0..outer {
        for t in inputs {
            let chunk = t.shape().get(ax).copied().ok_or(OpError::ShapeMismatch)? * inner;
            let src = t
                .data()
                .get(o * chunk..(o + 1) * chunk)
                .ok_or(OpError::ShapeMismatch)?;
            out.extend_from_slice(src);
        }
    }
    Tensor::new(out_shape, out)
}

/// 各軸の `(start, len)` で切り出す共通処理（軸数は x の rank と同じ）。
fn slice_ranges<T: Copy + Default + Send + Sync>(
    x: &Tensor<T>,
    ranges: &[(usize, usize)],
) -> Result<Tensor<T>, OpError> {
    let strides = row_strides(x.shape());
    let mut base = 0usize;
    let mut out_shape = Vec::with_capacity(ranges.len());
    for (k, &(s, l)) in ranges.iter().enumerate() {
        let st = strides.get(k).copied().ok_or(OpError::ShapeMismatch)?;
        // 空の出力ではオフセットを使わないため、あふれる積は検証せず空テンソルで返す。
        base = s
            .checked_mul(st)
            .and_then(|v| v.checked_add(base))
            .unwrap_or(usize::MAX);
        out_shape.push(l);
    }
    if element_count(&out_shape)? == 0 {
        return Tensor::new(out_shape, Vec::new());
    }
    if base == usize::MAX {
        return Err(OpError::LimitExceeded);
    }
    gather_by_strides(x.data(), out_shape, &strides, base)
}

/// Slice。steps は省略か全要素 1 のみ。starts・ends は負なら dim を足し `[0, dim]` に clamp する。
pub fn slice<T: Copy + Default + Send + Sync>(
    x: &Tensor<T>,
    starts: &Tensor<i64>,
    ends: &Tensor<i64>,
    axes: Option<&Tensor<i64>>,
    steps: Option<&Tensor<i64>>,
) -> Result<Tensor<T>, OpError> {
    let k = starts.data().len();
    if ends.data().len() != k
        || axes.is_some_and(|a| a.data().len() != k)
        || steps.is_some_and(|s| s.data().len() != k || s.data().iter().any(|&v| v != 1))
    {
        return Err(OpError::InvalidAttribute);
    }
    let rank = x.shape().len();
    let mut ranges: Vec<(usize, usize)> = x.shape().iter().map(|&d| (0, d)).collect();
    let mut seen = vec![false; rank];
    for i in 0..k {
        let ax = match axes {
            Some(a) => a.data().get(i).copied().ok_or(OpError::InvalidAttribute)?,
            None => i64::try_from(i).map_err(|_| OpError::InvalidAttribute)?,
        };
        let ax = normalize_axis(ax, rank)?;
        if std::mem::replace(seen.get_mut(ax).ok_or(OpError::InvalidAttribute)?, true) {
            return Err(OpError::InvalidAttribute);
        }
        let dim = x
            .shape()
            .get(ax)
            .copied()
            .ok_or(OpError::InvalidAttribute)?;
        let d = i64::try_from(dim).map_err(|_| OpError::LimitExceeded)?;
        let clamp = |v: i64| -> usize {
            let v = if v < 0 { v.saturating_add(d) } else { v };
            usize::try_from(v.clamp(0, d)).unwrap_or(0)
        };
        let s = clamp(
            starts
                .data()
                .get(i)
                .copied()
                .ok_or(OpError::InvalidAttribute)?,
        );
        let e = clamp(
            ends.data()
                .get(i)
                .copied()
                .ok_or(OpError::InvalidAttribute)?,
        );
        if let Some(r) = ranges.get_mut(ax) {
            *r = (s, e.saturating_sub(s));
        }
    }
    slice_ranges(x, &ranges)
}

/// Split。`split` の各要素は 0 以上で和は軸の次元と一致すること。
pub fn split<T: Copy + Default + Send + Sync>(
    x: &Tensor<T>,
    axis: i64,
    split: &Tensor<i64>,
) -> Result<Vec<Tensor<T>>, OpError> {
    if split.data().len() > MAX_SPLIT_OUTPUTS {
        return Err(OpError::LimitExceeded);
    }
    let ax = normalize_axis(axis, x.shape().len())?;
    let dim = x
        .shape()
        .get(ax)
        .copied()
        .ok_or(OpError::InvalidAttribute)?;
    let mut ranges: Vec<(usize, usize)> = x.shape().iter().map(|&d| (0, d)).collect();
    let mut outs = Vec::with_capacity(split.data().len());
    let mut pos = 0usize;
    for &l in split.data() {
        let l = usize::try_from(l).map_err(|_| OpError::InvalidAttribute)?;
        if let Some(r) = ranges.get_mut(ax) {
            *r = (pos, l);
        }
        pos = pos.checked_add(l).ok_or(OpError::LimitExceeded)?;
        if pos > dim {
            return Err(OpError::ShapeMismatch);
        }
        outs.push(slice_ranges(x, &ranges)?);
    }
    if pos != dim {
        return Err(OpError::ShapeMismatch);
    }
    Ok(outs)
}

/// Expand（双方向ブロードキャスト）。`shape` の負の次元は拒否する。
pub fn expand<T: Copy + Default + Send + Sync>(
    x: &Tensor<T>,
    shape: &Tensor<i64>,
) -> Result<Tensor<T>, OpError> {
    // rank 上限は target の確保・コピーより前に検証する（REQ-39）。
    if shape.data().len() > MAX_RANK {
        return Err(OpError::ShapeMismatch);
    }
    let mut target = Vec::with_capacity(shape.data().len());
    for &v in shape.data() {
        target.push(usize::try_from(v).map_err(|_| OpError::InvalidAttribute)?);
    }
    let out_shape = broadcast_shape(x.shape(), &target)?;
    let strides = broadcast_strides(x.shape(), out_shape.len());
    gather_by_strides(x.data(), out_shape, &strides, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t<T>(shape: &[usize], data: Vec<T>) -> Tensor<T> {
        Tensor::new(shape.to_vec(), data).unwrap()
    }
    fn v(d: &[i64]) -> Tensor<i64> {
        t(&[d.len()], d.to_vec())
    }
    fn iota(shape: &[usize]) -> Tensor<f32> {
        let n: usize = shape.iter().product();
        t(shape, (0..n).map(|i| i as f32).collect())
    }

    /// 高速経路（最内の連続コピー・複製）が要素ごとの汎用経路と一致する（REQ-28・#637）。
    #[test]
    fn req28_gather_by_strides_matches_reference_637() {
        let data: Vec<f32> = (0..64).map(|i| i as f32 * 0.5 - 3.0).collect();
        // (out_shape, strides, base): 連続・最内ブロードキャスト・外側ブロードキャスト・非連続・rank 0
        let cases: &[(&[usize], &[usize], usize)] = &[
            (&[2, 3, 4], &[12, 4, 1], 0),
            (&[2, 3, 4], &[12, 4, 0], 1),
            (&[2, 3, 4], &[0, 4, 1], 0),
            (&[2, 3, 4], &[12, 1, 3], 0),
            (&[3, 4], &[1, 4], 2),
            (&[5], &[1], 3),
            (&[], &[], 7),
            (&[2, 0, 4], &[12, 4, 1], 0),
        ];
        for &(shape, strides, base) in cases {
            let fast = gather_by_strides(&data, shape.to_vec(), strides, base).unwrap();
            let n: usize = shape.iter().product();
            let mut want = Vec::new();
            let mut idx = vec![0usize; shape.len()];
            for _ in 0..n {
                let p: usize = base + idx.iter().zip(strides).map(|(i, s)| i * s).sum::<usize>();
                want.push(data[p]);
                advance_index(&mut idx, shape);
            }
            let fb: Vec<u32> = fast.data().iter().map(|v| v.to_bits()).collect();
            let wb: Vec<u32> = want.iter().map(|v| v.to_bits()).collect();
            assert_eq!(fb, wb, "{shape:?} {strides:?}");
        }
        // 範囲外は panic せず Err
        assert!(gather_by_strides(&data, vec![2, 40], &[0, 1], 30).is_err());
    }

    #[test]
    fn req41_shape_op() {
        assert_eq!(shape(&iota(&[2, 3, 4])).unwrap().data(), &[2, 3, 4]);
        let s = shape(&t(&[], vec![1.0f32])).unwrap();
        assert_eq!(s.shape(), &[0]);
    }

    #[test]
    fn req41_gather_rows_scalar_and_fused_convert() {
        let d = iota(&[3, 2]);
        let g = gather_axis0(&d, &t(&[2], vec![2, 0]), |x| x).unwrap();
        assert_eq!(g.shape(), &[2, 2]);
        assert_eq!(g.data(), &[4.0, 5.0, 0.0, 1.0]);
        // rank 0 の index でスカラーを取り出す
        let g = gather_axis0(&v(&[10, 20, 30]), &t(&[], vec![1]), |x| x).unwrap();
        assert_eq!(g.shape(), &[] as &[usize]);
        assert_eq!(g.data(), &[20]);
        // bf16 → f32 を融合
        let e = t(&[2, 1], vec![0x3F80u16, 0x4000]);
        let g = gather_axis0(&e, &v(&[1]), |b| f32::from_bits((b as u32) << 16)).unwrap();
        assert_eq!(g.data(), &[2.0]);
    }

    #[test]
    fn req41_gather_rejects_out_of_range_and_negative() {
        let d = iota(&[3, 2]);
        assert_eq!(
            gather_axis0(&d, &v(&[3]), |x| x),
            Err(OpError::IndexOutOfRange)
        );
        assert_eq!(
            gather_axis0(&d, &v(&[-1]), |x| x),
            Err(OpError::IndexOutOfRange)
        );
    }

    #[test]
    fn req41_gather_elements() {
        let d = iota(&[2, 2, 3]);
        let i = t(&[2, 2, 1], vec![2, 0, 1, 1]);
        let g = gather_elements_last_axis(&d, &i).unwrap();
        assert_eq!(g.shape(), &[2, 2, 1]);
        assert_eq!(g.data(), &[2.0, 3.0, 7.0, 10.0]);
        assert_eq!(
            gather_elements_last_axis(&d, &t(&[2, 2, 1], vec![3, 0, 0, 0])),
            Err(OpError::IndexOutOfRange)
        );
        assert_eq!(
            gather_elements_last_axis(&d, &t(&[2, 1, 1], vec![0, 0])),
            Err(OpError::ShapeMismatch)
        );
    }

    #[test]
    fn req41_reshape_zero_and_infer() {
        // 0 は入力の同位置の次元。位置が入力 rank 以上なら拒否
        assert_eq!(
            reshape(iota(&[2, 6]), &v(&[0, 3, 0])),
            Err(OpError::InvalidAttribute)
        );
        let r = reshape(iota(&[2, 6]), &v(&[0, -1, 2])).unwrap();
        assert_eq!(r.shape(), &[2, 3, 2]);
        assert_eq!(r.data().len(), 12);
        assert_eq!(
            reshape(iota(&[2, 6]), &v(&[-1, -1])),
            Err(OpError::InvalidAttribute)
        );
        assert_eq!(
            reshape(iota(&[2, 6]), &v(&[5, -1])),
            Err(OpError::ShapeMismatch)
        );
        assert_eq!(
            reshape(iota(&[2, 6]), &v(&[5, 2])),
            Err(OpError::ShapeMismatch)
        );
        assert_eq!(
            reshape(iota(&[2, 6]), &v(&[-2, 6])),
            Err(OpError::InvalidAttribute)
        );
        assert_eq!(
            reshape(iota(&[0, 6]), &v(&[0, 0, -1])),
            Err(OpError::InvalidAttribute)
        );
    }

    #[test]
    fn req41_transpose() {
        let r = transpose(&iota(&[2, 3]), &[1, 0]).unwrap();
        assert_eq!(r.shape(), &[3, 2]);
        assert_eq!(r.data(), &[0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
        let r = transpose(&iota(&[1, 2, 3, 2]), &[0, 2, 1, 3]).unwrap();
        assert_eq!(r.shape(), &[1, 3, 2, 2]);
        assert_eq!(&r.data()[..4], &[0.0, 1.0, 6.0, 7.0]);
        assert_eq!(
            transpose(&iota(&[2, 3]), &[0, 0]),
            Err(OpError::InvalidAttribute)
        );
        assert_eq!(
            transpose(&iota(&[2, 3]), &[0]),
            Err(OpError::InvalidAttribute)
        );
        assert_eq!(
            transpose(&iota(&[2, 3]), &[0, 2]),
            Err(OpError::InvalidAttribute)
        );
    }

    #[test]
    fn req41_unsqueeze() {
        let r = unsqueeze(iota(&[2, 3]), &v(&[-1, 0])).unwrap();
        assert_eq!(r.shape(), &[1, 2, 3, 1]);
        assert_eq!(
            unsqueeze(iota(&[2]), &v(&[0, 0])),
            Err(OpError::InvalidAttribute)
        );
        assert_eq!(
            unsqueeze(iota(&[2]), &v(&[2])),
            Err(OpError::InvalidAttribute)
        );
        assert_eq!(
            unsqueeze(iota(&[1, 1, 1, 1, 1]), &v(&[0])),
            Err(OpError::ShapeMismatch)
        );
    }

    #[test]
    fn req41_concat() {
        let a = iota(&[2, 1]);
        let b = t(&[2, 2], vec![10.0, 11.0, 12.0, 13.0]);
        let r = concat(&[&a, &b], -1).unwrap();
        assert_eq!(r.shape(), &[2, 3]);
        assert_eq!(r.data(), &[0.0, 10.0, 11.0, 1.0, 12.0, 13.0]);
        let r = concat(&[&iota(&[1, 2]), &iota(&[2, 2])], 0).unwrap();
        assert_eq!(r.shape(), &[3, 2]);
        let empty = t::<f32>(&[0, 2], vec![]);
        let r = concat(&[&empty, &iota(&[1, 2])], 0).unwrap();
        assert_eq!(r.data(), &[0.0, 1.0]);
        assert_eq!(
            concat(&[&iota(&[2, 2]), &iota(&[3, 2])], 1),
            Err(OpError::ShapeMismatch)
        );
        let many = vec![&a; MAX_CONCAT_INPUTS + 1];
        assert_eq!(concat(&many, 0), Err(OpError::LimitExceeded));
    }

    #[test]
    fn req41_slice() {
        let x = iota(&[2, 4]);
        let r = slice(&x, &v(&[1]), &v(&[3]), Some(&v(&[1])), None).unwrap();
        assert_eq!(r.shape(), &[2, 2]);
        assert_eq!(r.data(), &[1.0, 2.0, 5.0, 6.0]);
        // 負の start と i64::MAX の end
        let r = slice(
            &x,
            &v(&[-1]),
            &v(&[i64::MAX]),
            Some(&v(&[-1])),
            Some(&v(&[1])),
        )
        .unwrap();
        assert_eq!(r.data(), &[3.0, 7.0]);
        // 既定の axes
        let r = slice(&x, &v(&[1]), &v(&[2]), None, None).unwrap();
        assert_eq!(r.shape(), &[1, 4]);
        // 長さ 0
        let r = slice(&x, &v(&[3]), &v(&[1]), Some(&v(&[1])), None).unwrap();
        assert_eq!(r.shape(), &[2, 0]);
        assert_eq!(
            slice(&x, &v(&[0]), &v(&[1]), None, Some(&v(&[2]))),
            Err(OpError::InvalidAttribute)
        );
        assert_eq!(
            slice(&x, &v(&[0, 0]), &v(&[1, 1]), Some(&v(&[0, 0])), None),
            Err(OpError::InvalidAttribute)
        );
    }

    #[test]
    fn req41_split() {
        let x = iota(&[2, 4]);
        let r = split(&x, -1, &v(&[2, 2])).unwrap();
        assert_eq!(r[0].data(), &[0.0, 1.0, 4.0, 5.0]);
        assert_eq!(r[1].data(), &[2.0, 3.0, 6.0, 7.0]);
        assert_eq!(split(&x, -1, &v(&[2, 1])), Err(OpError::ShapeMismatch));
        assert_eq!(split(&x, -1, &v(&[3, 3])), Err(OpError::ShapeMismatch));
        assert_eq!(split(&x, 1, &v(&[-1, 5])), Err(OpError::InvalidAttribute));
        assert_eq!(
            split(&x, 1, &t(&[65], vec![0; 65])),
            Err(OpError::LimitExceeded)
        );
    }

    #[test]
    fn req41_slice_empty_tensor_huge_stride_no_panic() {
        let x = t::<f32>(&[0, 3, i64::MAX as usize], vec![]);
        let o = slice(&x, &v(&[3]), &v(&[3]), Some(&v(&[1])), None).unwrap();
        assert_eq!(o.shape(), &[0, 0, i64::MAX as usize]);
    }

    #[test]
    fn req41_expand_gather_reject_rank_before_alloc() {
        let x = t(&[1], vec![1.0f32]);
        let big = Tensor::new(vec![MAX_RANK + 1], vec![1i64; MAX_RANK + 1]).unwrap();
        assert_eq!(expand(&x, &big).unwrap_err(), OpError::ShapeMismatch);
        let d = t(&[2, 1, 1, 1, 1], vec![0.0f32; 2]);
        let i = t(&[1, 1], vec![0i64]);
        assert_eq!(
            gather_axis0(&d, &i, |v| v).unwrap_err(),
            OpError::ShapeMismatch
        );
    }

    #[test]
    fn req41_expand() {
        let x = t(&[3, 1], vec![1.0f32, 2.0, 3.0]);
        let r = expand(&x, &v(&[2, 3, 4])).unwrap();
        assert_eq!(r.shape(), &[2, 3, 4]);
        assert_eq!(&r.data()[..8], &[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]);
        // shape 側が 1 なら入力の次元を保つ
        let r = expand(&x, &v(&[1, 1])).unwrap();
        assert_eq!(r.shape(), &[3, 1]);
        assert_eq!(expand(&x, &v(&[-1, 2])), Err(OpError::InvalidAttribute));
        assert_eq!(expand(&x, &v(&[2, 2])), Err(OpError::ShapeMismatch));
        // 確保前に上限で失敗する
        assert_eq!(
            expand(&t(&[1, 1], vec![0.0f32]), &v(&[1 << 20, 1 << 20])),
            Err(OpError::LimitExceeded)
        );
    }

    #[test]
    fn req41_gqa_reshape_transpose_and_rope_split_concat() {
        let x = iota(&[3, 8]);
        let r = reshape(x.clone(), &v(&[1, 3, 2, 4])).unwrap();
        let r = transpose(&r, &[0, 2, 1, 3]).unwrap();
        assert_eq!(r.shape(), &[1, 2, 3, 4]);
        let halves = split(&x, -1, &v(&[4, 4])).unwrap();
        let back = concat(&[&halves[0], &halves[1]], -1).unwrap();
        assert_eq!(back, x);
    }

    #[test]
    fn req41_shape_then_gather_gives_scalar() {
        let s = shape(&iota(&[2, 5, 3])).unwrap();
        let g = gather_axis0(&s, &t(&[], vec![1]), |x| x).unwrap();
        assert_eq!(g.shape(), &[] as &[usize]);
        assert_eq!(g.data(), &[5]);
    }

    /// 並列版の gather（Transpose・Expand）が要素ごとの汎用経路とビット一致する（REQ-28・#648）。
    #[test]
    fn req28_gather_by_strides_parallel_matches_reference_648() {
        let data: Vec<f32> = (0..2 * 195 * 128).map(|i| i as f32 * 0.25 - 9.0).collect();
        let cases: &[(&[usize], &[usize], usize)] = &[
            (&[2, 195, 128], &[195 * 128, 128, 1], 0),
            (&[128, 195, 2], &[1, 128, 195 * 128], 0),
            (&[2, 3, 195, 128], &[195 * 128, 0, 128, 1], 0),
            (&[2, 195, 128], &[195 * 128, 128, 0], 5),
        ];
        for &(shape, strides, base) in cases {
            let mut want = Vec::new();
            let mut idx = vec![0usize; shape.len()];
            for _ in 0..shape.iter().product::<usize>() {
                let p: usize = base + idx.iter().zip(strides).map(|(i, s)| i * s).sum::<usize>();
                want.push(data[p].to_bits());
                advance_index(&mut idx, shape);
            }
            for t in [1, 3, 7] {
                let got = crate::onnx::ops::arith::with_threads(t, || {
                    gather_by_strides(&data, shape.to_vec(), strides, base)
                })
                .unwrap();
                let gb: Vec<u32> = got.data().iter().map(|v| v.to_bits()).collect();
                assert_eq!(gb, want, "{shape:?} {strides:?} threads={t}");
            }
            let got = gather_by_strides(&data, shape.to_vec(), strides, base).unwrap();
            let gb: Vec<u32> = got.data().iter().map(|v| v.to_bits()).collect();
            assert_eq!(gb, want, "{shape:?} {strides:?}");
        }
    }
}
