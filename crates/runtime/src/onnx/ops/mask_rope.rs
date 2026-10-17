//! 注意機構まわりの演算（Range・Greater・Where・Sin・Cos。REQ-41・REQ-32・TASK-41.7-5・#419）。
//!
//! # 役割
//!
//! Qwen2 採点グラフ（#405 案 A）で、位置の列（Range）・因果マスク（Greater → Where）・
//! RoPE の表（Sin・Cos）を作る演算だけを実装する。演算一覧の正は
//! `docs/design/qwen2-scoring-onnx-ops.md`。RoPE の残り（Split・Concat・Mul 等）は
//! [`super::arith`]（#416）・[`super::shape`]（#417）の担当。
//!
//! # 契約
//!
//! - 実行器へは未接続。opset 13 より狭め、Range は INT64 のスカラーのみ、Greater は INT64 のみ、
//!   Where の値は FLOAT のみとする
//! - Range の要素数は i128 で求め、[`MAX_RANGE_ELEMENTS`] と照合してから確保する（REQ-39）。
//!   あふれ・上限超過は panic せず [`OpError`] で返す
//! - Sin・Cos は platform の libm に依存するが、同一マシン上では決定的（REQ-28）。
//!   学習側との数値照合は #421・#431 で行う（本 issue では未検証）
//! - 因果マスクは有限の -1e30 を使うため、Softmax で exp が 0 にアンダーフローし
//!   [`OpError::NonFinite`] にはならない

use super::arith::map1;
use super::{OpError, Tensor, broadcast_map2, broadcast_shape};

// ponytail: 暫定値。種類別の上限は #581 で確定する。
/// Range が生成する要素数の上限（`max_position_embeddings` 以上を想定）。
pub const MAX_RANGE_ELEMENTS: usize = 1 << 17;

/// Range（INT64 のスカラー 3 つ）。位置の列 `[start, start+delta, ...)` を作る。
/// 要素数は `max(ceil((limit-start)/delta), 0)`。`delta == 0` は拒否する。
pub fn range_i64(
    start: &Tensor<i64>,
    limit: &Tensor<i64>,
    delta: &Tensor<i64>,
) -> Result<Tensor<i64>, OpError> {
    let scalar = |t: &Tensor<i64>| -> Result<i64, OpError> {
        if !t.shape().is_empty() {
            return Err(OpError::ShapeMismatch);
        }
        t.data().first().copied().ok_or(OpError::ShapeMismatch)
    };
    let (s, l, d) = (scalar(start)?, scalar(limit)?, scalar(delta)?);
    if d == 0 {
        return Err(OpError::InvalidAttribute);
    }
    let (diff, step) = (i128::from(l) - i128::from(s), i128::from(d));
    // 切り上げ除算。正負どちらの delta でも diff と step の符号が揃うときだけ正になる。
    let n = if (diff > 0) == (step > 0) && diff != 0 {
        (diff.abs() + step.abs() - 1) / step.abs()
    } else {
        0
    };
    let n = usize::try_from(n).map_err(|_| OpError::LimitExceeded)?;
    if n > MAX_RANGE_ELEMENTS {
        return Err(OpError::LimitExceeded);
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let i = i64::try_from(i).map_err(|_| OpError::IntegerOverflow)?;
        let v = d
            .checked_mul(i)
            .and_then(|m| s.checked_add(m))
            .ok_or(OpError::IntegerOverflow)?;
        out.push(v);
    }
    Tensor::new(vec![n], out)
}

/// Greater（INT64 → BOOL、多方向ブロードキャスト）。因果マスク `kpos > qpos` を作る。
pub fn greater_i64(a: &Tensor<i64>, b: &Tensor<i64>) -> Result<Tensor<bool>, OpError> {
    broadcast_map2(a, b, |x, y| Ok(x > y))
}

/// Where（cond は BOOL、値は FLOAT、3 入力の多方向ブロードキャスト）。
/// `Where(future, -1e30, s)` でマスクを適用する。
pub fn where_f32(
    cond: &Tensor<bool>,
    x: &Tensor<f32>,
    y: &Tensor<f32>,
) -> Result<Tensor<f32>, OpError> {
    // 確保の前に出力形状を照合する（REQ-39）。
    broadcast_shape(&broadcast_shape(cond.shape(), x.shape())?, y.shape())?;
    // ponytail: 中間に (cond, x) の列を 1 つ確保する。遅ければ 3 入力のストライドループにする（#421 で計測）。
    let pairs = broadcast_map2(cond, x, |c, v| Ok((c, v)))?;
    broadcast_map2(&pairs, y, |(c, v), w| Ok(if c { v } else { w }))
}

/// Sin（FLOAT）。RoPE の表用。±inf は NaN になり、そのまま伝播する。
pub fn sin(x: &Tensor<f32>) -> Result<Tensor<f32>, OpError> {
    map1(x, f32::sin)
}

/// Cos（FLOAT）。RoPE の表用。±inf は NaN になり、そのまま伝播する。
pub fn cos(x: &Tensor<f32>) -> Result<Tensor<f32>, OpError> {
    map1(x, f32::cos)
}

#[cfg(test)]
mod tests {
    use super::super::arith::{cast_i64_to_f32, mul};
    use super::super::norm_act::softmax_last_axis;
    use super::super::shape::unsqueeze;
    use super::*;

    fn sc(v: i64) -> Tensor<i64> {
        Tensor::new(vec![], vec![v]).unwrap()
    }
    fn f(shape: &[usize], data: Vec<f32>) -> Tensor<f32> {
        Tensor::new(shape.to_vec(), data).unwrap()
    }
    fn ax(v: i64) -> Tensor<i64> {
        Tensor::new(vec![1], vec![v]).unwrap()
    }

    #[test]
    fn req41_range_basic() {
        let r = range_i64(&sc(0), &sc(5), &sc(1)).unwrap();
        assert_eq!((r.shape(), r.data()), (&[5][..], &[0, 1, 2, 3, 4][..]));
        assert_eq!(
            range_i64(&sc(0), &sc(10), &sc(3)).unwrap().data(),
            &[0, 3, 6, 9]
        );
        assert_eq!(
            range_i64(&sc(5), &sc(0), &sc(-2)).unwrap().data(),
            &[5, 3, 1]
        );
        for (a, b) in [(3, 3), (5, 0)] {
            let r = range_i64(&sc(a), &sc(b), &sc(1)).unwrap();
            assert_eq!((r.shape(), r.data().len()), (&[0][..], 0));
        }
    }

    #[test]
    fn req41_range_rejects() {
        assert_eq!(
            range_i64(&sc(0), &sc(1), &sc(0)),
            Err(OpError::InvalidAttribute)
        );
        assert_eq!(
            range_i64(&ax(0), &sc(1), &sc(1)),
            Err(OpError::ShapeMismatch)
        );
        assert_eq!(
            range_i64(&sc(0), &sc(i64::MAX), &sc(1)),
            Err(OpError::LimitExceeded)
        );
        assert_eq!(
            range_i64(&sc(i64::MIN), &sc(i64::MAX), &sc(1)),
            Err(OpError::LimitExceeded)
        );
        let m = MAX_RANGE_ELEMENTS as i64;
        assert_eq!(
            range_i64(&sc(0), &sc(m), &sc(1)).unwrap().data().len(),
            MAX_RANGE_ELEMENTS
        );
        assert_eq!(
            range_i64(&sc(0), &sc(m + 1), &sc(1)),
            Err(OpError::LimitExceeded)
        );
    }

    #[test]
    fn req41_greater_broadcast() {
        let a = Tensor::new(vec![3, 1], vec![0i64, 1, 2]).unwrap();
        let b = Tensor::new(vec![1, 3], vec![0i64, 1, 2]).unwrap();
        let g = greater_i64(&b, &a).unwrap();
        assert_eq!(g.shape(), &[3, 3]);
        assert_eq!(
            g.data(),
            &[false, true, true, false, false, true, false, false, false]
        );
        let c = Tensor::new(vec![2], vec![0i64, 1]).unwrap();
        assert_eq!(greater_i64(&c, &b), Err(OpError::ShapeMismatch));
    }

    #[test]
    fn req41_where_broadcast() {
        let cond = Tensor::new(vec![2, 2], vec![true, false, false, true]).unwrap();
        let x = f(&[], vec![-1e30]);
        let y = f(&[1, 2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let o = where_f32(&cond, &x, &y).unwrap();
        assert_eq!(o.shape(), &[1, 2, 2, 2]);
        assert_eq!(o.data(), &[-1e30, 2.0, 3.0, -1e30, -1e30, 6.0, 7.0, -1e30]);
        let bad = f(&[3], vec![0.0; 3]);
        assert_eq!(where_f32(&cond, &x, &bad), Err(OpError::ShapeMismatch));
    }

    #[test]
    fn req41_sin_cos() {
        let x = f(&[5], vec![0.0, -0.0, 1.0, 0.5, -2.5]);
        let (s, c) = (sin(&x).unwrap(), cos(&x).unwrap());
        assert_eq!(s.data()[0].to_bits(), 0.0f32.to_bits());
        assert_eq!(s.data()[1].to_bits(), (-0.0f32).to_bits());
        assert_eq!(c.data()[0], 1.0);
        for (i, &v) in x.data().iter().enumerate() {
            assert_eq!(s.data()[i].to_bits(), v.sin().to_bits());
            assert_eq!(c.data()[i].to_bits(), v.cos().to_bits());
        }
        let inf = f(&[2], vec![f32::INFINITY, f32::NEG_INFINITY]);
        assert!(sin(&inf).unwrap().data().iter().all(|v| v.is_nan()));
        assert!(cos(&inf).unwrap().data().iter().all(|v| v.is_nan()));
    }

    #[test]
    fn req41_rope_table_chain() {
        let pos = range_i64(&sc(0), &sc(3), &sc(1)).unwrap();
        let pos = cast_i64_to_f32(&unsqueeze(pos, &ax(1)).unwrap()).unwrap();
        let freqs = mul(&pos, &f(&[2], vec![1.0, 0.5])).unwrap();
        let (c, s) = (cos(&freqs).unwrap(), sin(&freqs).unwrap());
        assert_eq!((c.shape(), s.shape()), (&[3, 2][..], &[3, 2][..]));
        assert_eq!(&c.data()[..2], &[1.0, 1.0]);
        assert_eq!(&s.data()[..2], &[0.0, 0.0]);
        assert_eq!(c.data()[2], 1.0f32.cos());
        assert_eq!(s.data()[3], 0.5f32.sin());
    }

    #[test]
    fn req41_causal_mask_chain() {
        let t = range_i64(&sc(0), &sc(3), &sc(1)).unwrap();
        let q = unsqueeze(t.clone(), &ax(1)).unwrap();
        let k = unsqueeze(t, &ax(0)).unwrap();
        let future = greater_i64(&k, &q).unwrap();
        let masked = where_f32(&future, &f(&[], vec![-1e30]), &f(&[3, 3], vec![0.0; 9])).unwrap();
        let p = softmax_last_axis(&masked).unwrap();
        assert_eq!(
            p.data(),
            &[
                1.0,
                0.0,
                0.0,
                0.5,
                0.5,
                0.0,
                p.data()[6],
                p.data()[7],
                p.data()[8]
            ]
        );
        let third = 1.0f32 / 3.0;
        assert_eq!(&p.data()[6..], &[third, third, third]);
    }
}
