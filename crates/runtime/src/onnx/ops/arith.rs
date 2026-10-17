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
//! - 縮約（MatMul・Gemm）は単一スレッドで k の昇順に f32 で加算し、スレッド数に依存させない（REQ-28）
//! - Sqrt・Reciprocal は IEEE どおり（負の Sqrt は NaN、0 の Reciprocal は inf）。拒否せず伝播させ、
//!   確率を作る Softmax 系が [`OpError::NonFinite`] で fail-closed にする
//! - 許容差: f32 の演算に 1e-9 は適用しない（1e-9 は f64 計算の場合）。テストは 2 進で正確な値の
//!   厳密一致、または IEEE の単一演算とのビット一致で比べる

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

/// MatMul（FLOAT）。両入力 rank 2〜[`MAX_RANK`]。先頭（バッチ）次元は numpy 方式でブロードキャストする。
///
/// 各出力要素は k の昇順の f32 加算（i-k-j のループ順でも要素ごとの加算順は変わらない）。
// ponytail: 単一スレッドの素朴な積和。LM head（K*M 行 × 151936 × 1536）が律速になりうる。
// 速度は #421 で実測し、並列化する場合も出力要素ごとの k 昇順の加算順を固定すること。
pub fn matmul(
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
    let (a_mat, b_mat, o_mat) = (m * k, k * n, m * n);
    let rank = batch.len();
    let ra: Vec<usize> = broadcast_strides(ba, rank)
        .iter()
        .map(|s| s * a_mat)
        .collect();
    let rb: Vec<usize> = broadcast_strides(bb, rank)
        .iter()
        .map(|s| s * b_mat)
        .collect();
    let mut out = vec![0.0f32; total];
    let mut idx = vec![0usize; rank];
    for bi in 0..nbatch {
        let (mut oa, mut ob) = (0usize, 0usize);
        for (d, &i) in idx.iter().enumerate() {
            oa += i * ra.get(d).copied().unwrap_or(0);
            ob += i * rb.get(d).copied().unwrap_or(0);
        }
        let am = a.data().get(oa..oa + a_mat).ok_or(OpError::ShapeMismatch)?;
        let bm = b.data().get(ob..ob + b_mat).ok_or(OpError::ShapeMismatch)?;
        let om = out
            .get_mut(bi * o_mat..(bi + 1) * o_mat)
            .ok_or(OpError::ShapeMismatch)?;
        for mi in 0..m {
            let arow = am.get(mi * k..(mi + 1) * k).ok_or(OpError::ShapeMismatch)?;
            let orow = om
                .get_mut(mi * n..(mi + 1) * n)
                .ok_or(OpError::ShapeMismatch)?;
            for (kk, &av) in arow.iter().enumerate() {
                check_deadline(until)?;
                let brow = bm.get(kk * n..(kk + 1) * n).ok_or(OpError::ShapeMismatch)?;
                for (o, &bv) in orow.iter_mut().zip(brow) {
                    *o += av * bv;
                }
            }
        }
        advance_index(&mut idx, &batch);
    }
    check_deadline(until)?;
    Tensor::new(out_shape, out)
}

/// 期限を過ぎていれば [`OpError::DeadlineExceeded`]。積和ループが行（k または n）ごとに呼ぶ。
fn check_deadline(until: Option<Instant>) -> Result<(), OpError> {
    match until {
        Some(t) if Instant::now() > t => Err(OpError::DeadlineExceeded),
        _ => Ok(()),
    }
}

/// Gemm（transA=0・transB=1・alpha=1・beta=1 固定）。`A[M,K] × W[N,K]ᵀ (+ C[N])`。
///
/// 重みは元の型 `W` のまま受け、`to_f32` で積和の中で展開する（全体を f32 へ複製しない。
/// bf16 なら [`bf16_to_f32`]）。`out[m][n] = Σ_k a[m,k]*w[n,k]`（k 昇順）を求めてから `c[n]` を足す。
pub fn gemm_trans_b<W: Copy>(
    a: &Tensor<f32>,
    w: &Tensor<W>,
    c: Option<&Tensor<f32>>,
    to_f32: impl Fn(W) -> f32,
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
    let mut out = Vec::with_capacity(total);
    for mi in 0..m {
        let arow = a
            .data()
            .get(mi * k..(mi + 1) * k)
            .ok_or(OpError::ShapeMismatch)?;
        for ni in 0..n {
            check_deadline(until)?;
            let wrow = w
                .data()
                .get(ni * k..(ni + 1) * k)
                .ok_or(OpError::ShapeMismatch)?;
            let mut s = 0.0f32;
            for (&x, &y) in arow.iter().zip(wrow) {
                s += x * to_f32(y);
            }
            if let Some(c) = c {
                s += c.data().get(ni).copied().ok_or(OpError::ShapeMismatch)?;
            }
            out.push(s);
        }
    }
    check_deadline(until)?;
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
