//! 正規化と活性化の演算カーネル（REQ-41・REQ-32・TASK-41.7-4・#418）。
//!
//! # 役割
//!
//! `docs/design/qwen2-scoring-onnx-ops.md` 3 節で担当 #418 とされた 5 演算
//! （ReduceMean・ReduceSum・Sigmoid・Softmax・LogSoftmax）だけを実装する。
//! RMSNorm の残り（Mul・Add・Sqrt・Reciprocal）と SiLU の Mul は #416 の担当。
//! GELU・LayerNorm・Erf は一覧に無い（Qwen2 は RMSNorm と silu のみ）ため実装しない。
//!
//! # 契約
//!
//! - 実行器へは未接続。許可リスト照合（演算×属性×dtype×rank）は別の担当（#581・#420）。
//!   opset 13 より意図して狭めた: ReduceMean・Softmax・LogSoftmax は axis=-1 固定、
//!   ReduceMean は keepdims=1 固定、ReduceSum は keepdims=0 固定で空 axes は全縮約
//! - 計算は f32。グラフ出力を作る Softmax だけ f64（[`softmax_last_axis_f64`]。C1・C3 と同じ実装を再利用）
//! - 数値安定化: Softmax 系は行ごとに max を引いてから exp する（`exp` のあふれ防止）。
//!   Sigmoid は符号で式を分け `exp` の引数を常に非正にする。
//!   Softmax 系の行に NaN・±inf があれば [`OpError::NonFinite`]（因果マスクは有限の -1e30 を使うため当たらない）
//! - 縮約・合計は単一スレッドで添字の昇順に加算し、スレッド数に依存させない（REQ-28）
//! - 形状は [`element_count`]・[`Tensor::new`] で上限照合してから確保する（REQ-39）。panic しない

use super::arith::{map1, par_collect};
use super::shape::normalize_axis;
use super::{OpError, Tensor, advance_index, element_count};

/// 最後の軸の長さ。rank 0 と長さ 0（0 除算・空 softmax）は拒否する。
fn last_dim(shape: &[usize]) -> Result<usize, OpError> {
    match shape.last() {
        Some(&d) if d > 0 => Ok(d),
        _ => Err(OpError::ShapeMismatch),
    }
}

/// ReduceMean（axes=[-1]・keepdims=1）。RMSNorm の分散項に使う。
pub fn reduce_mean_last_axis(x: &Tensor<f32>) -> Result<Tensor<f32>, OpError> {
    let d = last_dim(x.shape())?;
    let mut shape = x.shape().to_vec();
    if let Some(l) = shape.last_mut() {
        *l = 1;
    }
    let n = d as f32;
    let out: Vec<f32> = x
        .data()
        .chunks(d)
        .map(|row| row.iter().fold(0.0f32, |a, &v| a + v) / n)
        .collect();
    Tensor::new(shape, out)
}

/// ReduceSum（axes は入力・keepdims=0・noop_with_empty_axes=0）。空 axes は全軸を縮約する。
pub fn reduce_sum(x: &Tensor<f32>, axes: &Tensor<i64>) -> Result<Tensor<f32>, OpError> {
    let rank = x.shape().len();
    if axes.shape().len() != 1 {
        return Err(OpError::InvalidAttribute);
    }
    let mut reduced = vec![false; rank];
    for &a in axes.data() {
        let a = normalize_axis(a, rank)?;
        let r = reduced.get_mut(a).ok_or(OpError::InvalidAttribute)?;
        if *r {
            return Err(OpError::InvalidAttribute); // 重複
        }
        *r = true;
    }
    if axes.data().is_empty() {
        reduced.iter_mut().for_each(|r| *r = true);
    }
    let out_shape: Vec<usize> = x
        .shape()
        .iter()
        .zip(&reduced)
        .filter(|(_, r)| !**r)
        .map(|(&d, _)| d)
        .collect();
    let out_n = element_count(&out_shape)?;
    // 入力の各軸が出力で持つ stride（縮約軸は 0）
    let mut strides = vec![0usize; rank];
    let mut acc = 1usize;
    for k in (0..rank).rev() {
        if !reduced.get(k).copied().unwrap_or(true) {
            if let Some(s) = strides.get_mut(k) {
                *s = acc;
            }
            acc = acc.saturating_mul(x.shape().get(k).copied().unwrap_or(1));
        }
    }
    let mut out = vec![0.0f32; out_n];
    let mut idx = vec![0usize; rank];
    for &v in x.data() {
        let pos: usize = idx.iter().zip(&strides).map(|(i, s)| i * s).sum();
        *out.get_mut(pos).ok_or(OpError::ShapeMismatch)? += v;
        advance_index(&mut idx, x.shape());
    }
    Tensor::new(out_shape, out)
}

/// Sigmoid（FLOAT）。SiLU は #416 の Mul と組み合わせる。
/// `exp` の引数を常に非正にしてあふれを避ける。±inf は 1・0、NaN は伝播する。
pub fn sigmoid(x: &Tensor<f32>) -> Result<Tensor<f32>, OpError> {
    // `exp` を含むので重み 4（#648）
    map1(x, 4, |v| {
        if v >= 0.0 {
            1.0 / (1.0 + (-v).exp())
        } else {
            let e = v.exp();
            e / (1.0 + e)
        }
    })
}

/// 行ごとに非有限を拒否し max を求め、`f(row, max, out)` で行を変換して連結する。
/// 行は [`par_collect`] で連続範囲に分けて並列に処理する（1 行は 1 ワーカーで、行内の順序は不変。
/// REQ-28・#648）。`weight` は 1 要素あたりの重み。
fn map_rows<T: Send>(
    x: &Tensor<f32>,
    weight: usize,
    f: impl Fn(&[f32], f32, &mut Vec<T>) -> Result<(), OpError> + Sync,
) -> Result<(Vec<usize>, Vec<T>), OpError> {
    let d = last_dim(x.shape())?;
    let rows = x.data().len() / d;
    let out = par_collect(rows, weight.saturating_mul(d), |r| {
        let part = x
            .data()
            .get(r.start * d..r.end * d)
            .ok_or(OpError::ShapeMismatch)?;
        let mut out = Vec::with_capacity(part.len());
        for row in part.chunks(d) {
            if row.iter().any(|v| !v.is_finite()) {
                return Err(OpError::NonFinite);
            }
            let max = row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
            f(row, max, &mut out)?;
        }
        Ok(out)
    })?;
    Ok((x.shape().to_vec(), out))
}

/// Softmax（axis=-1）。max を引いて f32 で計算する。注意の重みに使う。
pub fn softmax_last_axis(x: &Tensor<f32>) -> Result<Tensor<f32>, OpError> {
    let (shape, out) = map_rows(x, 4, |row, max, out| {
        let start = out.len();
        let mut sum = 0.0f32;
        for &v in row {
            let e = (v - max).exp();
            sum += e;
            out.push(e);
        }
        out.iter_mut().skip(start).for_each(|e| *e /= sum);
        Ok(())
    })?;
    Tensor::new(shape, out)
}

/// Softmax（axis=-1）の f64 出力。グラフ出力 `probs` 用で、C1・C3 と同じ `softmax_f64` を行ごとに使う。
pub fn softmax_last_axis_f64(x: &Tensor<f32>) -> Result<Tensor<f64>, OpError> {
    let (shape, out) = map_rows(x, 4, |row, _, out| {
        out.extend(super::super::softmax_f64(row).map_err(|_| OpError::NonFinite)?);
        Ok(())
    })?;
    Tensor::new(shape, out)
}

/// LogSoftmax（axis=-1）。`x - max - ln(Σ exp(x - max))` で桁あふれ・桁落ちを避ける。採点に使う。
pub fn log_softmax_last_axis(x: &Tensor<f32>) -> Result<Tensor<f32>, OpError> {
    let (shape, out) = map_rows(x, 4, |row, max, out| {
        let sum: f32 = row.iter().fold(0.0f32, |a, &v| a + (v - max).exp());
        let lse = sum.ln();
        out.extend(row.iter().map(|&v| (v - max) - lse));
        Ok(())
    })?;
    Tensor::new(shape, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(shape: &[usize], d: &[f32]) -> Tensor<f32> {
        Tensor::new(shape.to_vec(), d.to_vec()).unwrap()
    }
    fn ax(a: &[i64]) -> Tensor<i64> {
        Tensor::new(vec![a.len()], a.to_vec()).unwrap()
    }
    const TOL: f32 = 1e-6;

    #[test]
    fn req41_reduce_mean_last_axis() {
        let r = reduce_mean_last_axis(&t(&[2, 4], &[1., 2., 3., 4., 0., 0., 0., 8.])).unwrap();
        assert_eq!(r.shape(), &[2, 1]);
        assert_eq!(r.data(), &[2.5, 2.0]);
        // RMSNorm の分散項: mean([3,4]^2) = 12.5
        let r = reduce_mean_last_axis(&t(&[1, 2], &[9., 16.])).unwrap();
        assert_eq!(r.data(), &[12.5]);
        assert_eq!(
            reduce_mean_last_axis(&t(&[2, 0], &[])),
            Err(OpError::ShapeMismatch)
        );
        assert_eq!(
            reduce_mean_last_axis(&t(&[], &[1.])),
            Err(OpError::ShapeMismatch)
        );
    }

    #[test]
    fn req41_reduce_sum_axes() {
        let x = t(&[2, 2, 1], &[1., 2., 3., 4.]);
        let r = reduce_sum(&x, &ax(&[1, 2])).unwrap();
        assert_eq!((r.shape(), r.data()), (&[2][..], &[3.0, 7.0][..]));
        let r = reduce_sum(&x, &ax(&[0])).unwrap();
        assert_eq!((r.shape(), r.data()), (&[2, 1][..], &[4.0, 6.0][..]));
        let r = reduce_sum(&x, &ax(&[-1])).unwrap();
        assert_eq!((r.shape(), r.data()), (&[2, 2][..], &[1., 2., 3., 4.][..]));
        let r = reduce_sum(&x, &ax(&[])).unwrap();
        assert_eq!((r.shape(), r.data()), (&[][..], &[10.0][..]));
        assert_eq!(reduce_sum(&x, &ax(&[1, 1])), Err(OpError::InvalidAttribute));
        assert_eq!(reduce_sum(&x, &ax(&[3])), Err(OpError::InvalidAttribute));
    }

    #[test]
    fn req41_sigmoid_stable() {
        let inf = f32::INFINITY;
        let r = sigmoid(&t(
            &[7],
            &[0., 1., 1000., -1000., inf, f32::NEG_INFINITY, f32::NAN],
        ))
        .unwrap();
        let d = r.data();
        assert_eq!(d[0], 0.5);
        assert!((d[1] - 0.731_058_6).abs() < TOL);
        assert_eq!((d[2], d[3], d[4], d[5]), (1.0, 0.0, 1.0, 0.0));
        assert!(d[6].is_nan());
    }

    #[test]
    fn req41_softmax_f32() {
        assert_eq!(
            softmax_last_axis(&t(&[4], &[1.; 4])).unwrap().data(),
            &[0.25; 4]
        );
        // max を引かないと inf/inf で NaN になる回帰
        assert_eq!(
            softmax_last_axis(&t(&[2], &[1000., 1000.])).unwrap().data(),
            &[0.5, 0.5]
        );
        assert_eq!(
            softmax_last_axis(&t(&[2], &[0., -1e30])).unwrap().data(),
            &[1.0, 0.0]
        );
        let r = softmax_last_axis(&t(&[2, 2], &[0., 0., 5., 5.])).unwrap();
        assert_eq!(r.data(), &[0.5; 4]);
        assert_eq!(
            softmax_last_axis(&t(&[2], &[f32::NAN, 0.])),
            Err(OpError::NonFinite)
        );
        assert_eq!(
            softmax_last_axis(&t(&[2], &[f32::INFINITY, 0.])),
            Err(OpError::NonFinite)
        );
        assert_eq!(
            softmax_last_axis(&t(&[0], &[])),
            Err(OpError::ShapeMismatch)
        );
    }

    #[test]
    fn req41_softmax_f64_matches_c1_c3() {
        let row = [0.5f32, -1.0, 2.0];
        let r = softmax_last_axis_f64(&t(&[2, 3], &[0.5, -1.0, 2.0, 0.5, -1.0, 2.0])).unwrap();
        let want = super::super::super::softmax_f64(&row).unwrap();
        assert_eq!(&r.data()[..3], &want[..]);
        assert_eq!(&r.data()[3..], &want[..]);
        assert!((r.data()[..3].iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert_eq!(
            softmax_last_axis_f64(&t(&[1], &[f32::NAN])),
            Err(OpError::NonFinite)
        );
    }

    #[test]
    fn req41_log_softmax() {
        let ln2 = std::f32::consts::LN_2;
        for x in [[0., 0.], [1000., 1000.]] {
            let r = log_softmax_last_axis(&t(&[2], &x)).unwrap();
            assert!(r.data().iter().all(|v| (v + ln2).abs() < TOL));
        }
        assert_eq!(
            log_softmax_last_axis(&t(&[2], &[0., -1e30]))
                .unwrap()
                .data()[0],
            0.0
        );
        let r = log_softmax_last_axis(&t(&[3], &[1., 2., 3.])).unwrap();
        assert!((r.data().iter().map(|v| v.exp()).sum::<f32>() - 1.0).abs() < TOL);
        assert_eq!(
            log_softmax_last_axis(&t(&[1], &[f32::INFINITY])),
            Err(OpError::NonFinite)
        );
    }

    /// Sigmoid・Softmax・LogSoftmax の並列版が 1 スレッド版とビット一致する（REQ-28・#648）。
    #[test]
    fn req28_sigmoid_softmax_parallel_match_serial_648() {
        use super::super::arith::with_threads;
        let (rows, d) = (157usize, 901usize); // 行数が分割数で割り切れない・しきい値超え
        let data: Vec<f32> = (0..rows * d)
            .map(|i| ((i as f32) * 0.013).sin() * 9.0 - 1.0)
            .collect();
        let x = t(&[rows, d], &data);
        let bits = |r: Tensor<f32>| r.data().iter().map(|v| v.to_bits()).collect::<Vec<u32>>();
        type Op = fn(&Tensor<f32>) -> Result<Tensor<f32>, OpError>;
        for op in [sigmoid as Op, softmax_last_axis, log_softmax_last_axis] {
            let want = bits(with_threads(1, || op(&x)).unwrap());
            for th in [2, 3, 7] {
                assert_eq!(
                    bits(with_threads(th, || op(&x)).unwrap()),
                    want,
                    "threads={th}"
                );
            }
            assert_eq!(bits(op(&x).unwrap()), want);
        }
        // 非有限は並列でも拒否する
        let mut bad = data.clone();
        bad[5 * d + 3] = f32::NAN;
        let r = with_threads(4, || softmax_last_axis(&t(&[rows, d], &bad)));
        assert_eq!(r, Err(OpError::NonFinite));
    }
}
