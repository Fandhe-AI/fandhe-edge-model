//! McNemar 検定で下限基準（majority）との差を検出するための必要評価件数の
//! 事前計算（Connor 1987 のサンプルサイズ公式）。
//!
//! [`crate::significance`]（下限基準に対する有意性判定。REQ-25・
//! TASK-25.1-2・issue #65）は、評価件数が [`crate::significance::RequiredSampleSize`]
//! 未満なら判定を実行せず「判定不能」を返す分岐をすでに実装している
//! （PR #219）。本モジュールはその `RequiredSampleSize` を、検出力（power）・
//! 有意水準（α）・検出したい候補の正解率の差から事前に算出する関数を提供する
//! （REQ-25 異常系・TASK-25.2・issue #66）。PoC-10
//! `03-poc/scratch-classifier/scripts/required_n_mcnemar.py` の計算手順を
//! Rust（std のみ）へ移植したもの。
//!
//! 呼び出し文脈は、CLI の `evaluate` 工程（将来・TASK-33.x）や事前登録手続き
//! が、仮定した候補・下限基準の正解率の差（`p_b`・`p_c`）と α・power から
//! 必要件数を求め、[`crate::significance::judge`]・
//! [`crate::significance::compare_with_baseline`] へ渡す想定。仮定値
//! そのものをどこから受け取るか（定義ファイル・CLI 引数）は未確定
//! （TASK-33.x）。
//!
//! # 対象外（本 issue の範囲外）
//!
//! - Holm 補正（複数候補比較。REQ-26）は実装しない。ただし `alpha` を
//!   引数として受け取る設計にしているため、呼び出し側が Holm 補正後の
//!   最厳段の α（`α / m`）を渡して必要件数を求めることができる
//!   （TASK-25.3 で利用する想定）
//! - 仮定値（`p_b`・`p_c`・`alpha`・`power`）に既定値は持たせない
//!   （[`crate::significance::RequiredSampleSize`] が既定値を持たない
//!   2026-09-28 オーナー承認の設計と揃える）。呼び出し側が必ず明示的に
//!   決めた値を渡す
//! - JSON 入出力・ファイル I/O・CLI 統合は行わない

use crate::significance::{MAX_EVAL_RECORDS, RequiredSampleSize};

/// [`McNemarSampleSizeAssumption::new`]・[`required_sample_size_mcnemar`]が
/// 返しうるエラー。
///
/// データ本文（評価データの内容）を含まない値のみを保持する
/// （`.claude/rules/security.md`「秘密情報の混入防止」）。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum SampleSizeError {
    /// 入力値が有限でない（NaN・±∞）。
    NonFiniteInput {
        /// 非有限だったフィールド名。
        field: &'static str,
    },
    /// `alpha` が `(0, 1)` の範囲外。
    AlphaOutOfRange,
    /// `power` が `(0, 1)` の範囲外。
    PowerOutOfRange,
    /// `p_c` が負。
    NegativeBaselineProportion,
    /// `p_b <= p_c`（候補が下限基準を上回る方向〔`d > 0`〕になっていない）。
    CandidateNotAboveBaseline,
    /// `p_b + p_c` が `1.0` を超える（2 つの排反な正解率の和として不正）。
    ProportionsExceedOne,
    /// [`normal_quantile`] の定義域（開区間 `(0, 1)`）外の `p` を渡した。
    QuantileOutOfDomain,
    /// 算出した必要件数が [`MAX_EVAL_RECORDS`] を超える。
    ///
    /// この必要件数は [`crate::significance::compare_with_baseline`] が
    /// 確保前に拒否する上限を超えており、この仮定のままでは評価件数を
    /// どれだけ増やしても必ず判定不能になる（満たしようがない）。
    /// fail-closed として計算結果を返さず拒否する（α が極小・`p_b`・`p_c`
    /// の差が極小な仮定で到達しうる）。上限自体は
    /// [`crate::significance::MAX_EVAL_RECORDS`] のドキュメントが示す
    /// とおり暫定値であり、実測に基づく調整は後続 TASK で行う。
    ExceedsRecordLimit {
        /// 算出した必要件数（丸め後、`u64` へ変換する前の値）。
        required: u64,
        /// [`MAX_EVAL_RECORDS`] の値。
        limit: usize,
    },
    /// 理論上到達しないはずの内部不整合（非有限値の算出・負の平方根引数等）。
    /// fail-closed のガード（`crate::mcnemar::McNemarError::Internal` と
    /// 同じ位置づけ）。
    Internal {
        /// 診断用の詳細（データ本文は含めない）。
        detail: String,
    },
}

impl std::fmt::Display for SampleSizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SampleSizeError::NonFiniteInput { field } => {
                write!(f, "non-finite input for field: {field}")
            }
            SampleSizeError::AlphaOutOfRange => {
                write!(f, "alpha must be in the open interval (0, 1)")
            }
            SampleSizeError::PowerOutOfRange => {
                write!(f, "power must be in the open interval (0, 1)")
            }
            SampleSizeError::NegativeBaselineProportion => {
                write!(f, "p_c must be non-negative")
            }
            SampleSizeError::CandidateNotAboveBaseline => {
                write!(f, "p_b must be strictly greater than p_c")
            }
            SampleSizeError::ProportionsExceedOne => {
                write!(f, "p_b + p_c must not exceed 1.0")
            }
            SampleSizeError::QuantileOutOfDomain => {
                write!(f, "quantile input must be in the open interval (0, 1)")
            }
            SampleSizeError::ExceedsRecordLimit { required, limit } => {
                write!(
                    f,
                    "required sample size {required} exceeds record limit {limit}"
                )
            }
            SampleSizeError::Internal { detail } => {
                write!(f, "internal sample size computation error: {detail}")
            }
        }
    }
}

impl std::error::Error for SampleSizeError {}

/// McNemar サンプルサイズ算出の入力となる仮定（検出力に基づく事前登録の
/// パラメータ）。
///
/// フィールドは非公開にし、[`new`][Self::new] で検証済みの値のみを
/// 保持できるようにする（壊れた値〔NaN・不正な向き〕を表現できない型に
/// する。`.claude/rules/coding-rust.md`「公開 API・型設計」）。
///
/// 既定値（`Default`）は意図的に実装しない。PoC-10 の仮定
/// （`p_b=0.15`・`p_c=0.05`・`power=0.8`）はテスト・ドキュメントの参考値に
/// とどめ、本 crate の公開 API としては固定しない（[`crate::significance::RequiredSampleSize`]
/// が既定値を持たない設計と揃える）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct McNemarSampleSizeAssumption {
    p_b: f64,
    p_c: f64,
    alpha: f64,
    power: f64,
}

impl McNemarSampleSizeAssumption {
    /// 仮定を検証して構築する。
    ///
    /// 検証規則:
    ///
    /// 1. `p_b`・`p_c`・`alpha`・`power` すべてが有限（NaN・±∞ を拒否）
    /// 2. `0 < alpha < 1`・`0 < power < 1`
    /// 3. `p_c >= 0`
    /// 4. `p_b > p_c`（候補が下限基準を上回る方向の差 `d = p_b - p_c > 0` を
    ///    検出する前提。`d <= 0` では検出力の計算が意味を持たない）
    /// 5. `p_b + p_c <= 1.0`（2 つの排反な正解率の和として妥当な範囲）
    pub fn new(p_b: f64, p_c: f64, alpha: f64, power: f64) -> Result<Self, SampleSizeError> {
        if !p_b.is_finite() {
            return Err(SampleSizeError::NonFiniteInput { field: "p_b" });
        }
        if !p_c.is_finite() {
            return Err(SampleSizeError::NonFiniteInput { field: "p_c" });
        }
        if !alpha.is_finite() {
            return Err(SampleSizeError::NonFiniteInput { field: "alpha" });
        }
        if !power.is_finite() {
            return Err(SampleSizeError::NonFiniteInput { field: "power" });
        }
        if !(alpha > 0.0 && alpha < 1.0) {
            return Err(SampleSizeError::AlphaOutOfRange);
        }
        if !(power > 0.0 && power < 1.0) {
            return Err(SampleSizeError::PowerOutOfRange);
        }
        if p_c < 0.0 {
            return Err(SampleSizeError::NegativeBaselineProportion);
        }
        if p_b <= p_c {
            return Err(SampleSizeError::CandidateNotAboveBaseline);
        }
        if p_b + p_c > 1.0 {
            return Err(SampleSizeError::ProportionsExceedOne);
        }
        Ok(Self {
            p_b,
            p_c,
            alpha,
            power,
        })
    }

    /// 候補のみ正解する割合の仮定。
    pub fn p_b(&self) -> f64 {
        self.p_b
    }

    /// 下限基準のみ正解する割合の仮定。
    pub fn p_c(&self) -> f64 {
        self.p_c
    }

    /// 両側有意水準（α）。
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// 検出力（power）。
    pub fn power(&self) -> f64 {
        self.power
    }
}

/// 標準正規分布の分位点（累積分布関数の逆関数）。
///
/// Wichura, M. J. (1988). "Algorithm AS 241: The Percentage Points of the
/// Normal Distribution." *Applied Statistics*, 37(3), 477-484.（PPND16。
/// 相対精度 約 1e-16）を係数転記した実装。R の `qnorm`（`src/nmath/qnorm.c`）
/// と同一アルゴリズム。Acklam (2003) の近似（相対誤差 1.15e-9）は本 crate の
/// 決定性の許容差（1e-9）と同水準まで落ちるため採用しない。
///
/// 定義域は開区間 `(0, 1)`。`0.0`・`1.0`・範囲外・非有限値は
/// [`SampleSizeError::QuantileOutOfDomain`] を返す。
///
/// 決定性（REQ-25「決定性と証拠の種別」）: 分岐・演算順を固定した純粋関数で、
/// 並列化しない。
pub fn normal_quantile(p: f64) -> Result<f64, SampleSizeError> {
    if !(p.is_finite() && p > 0.0 && p < 1.0) {
        return Err(SampleSizeError::QuantileOutOfDomain);
    }

    let q = p - 0.5;

    if q.abs() <= 0.425 {
        // 中央域（0.075 <= p <= 0.925）: r = 0.180625 - q^2 の多項式比。
        let r = 0.180625 - q * q;
        let num = horner(&AS241_CENTRAL_NUM, r);
        let den = horner(&AS241_CENTRAL_DEN, r);
        Ok(q * num / den)
    } else {
        // 端域（p < 0.075 または p > 0.925）: min(p, 1-p) を対数変換して
        // さらに 2 段の多項式比で求める。
        let r_tail = if q > 0.0 { 1.0 - p } else { p };
        if r_tail <= 0.0 {
            // p が (0,1) 内であることは上で検証済みのため、この分岐は
            // 理論上到達しない（r_tail はここでは常に正）。
            return Err(SampleSizeError::Internal {
                detail: "non-positive tail probability in normal_quantile".to_string(),
            });
        }
        let r = (-r_tail.ln()).sqrt();

        let val = if r <= 5.0 {
            let r = r - 1.6;
            let num = horner(&AS241_TAIL_NEAR_NUM, r);
            let den = horner(&AS241_TAIL_NEAR_DEN, r);
            num / den
        } else {
            let r = r - 5.0;
            let num = horner(&AS241_TAIL_FAR_NUM, r);
            let den = horner(&AS241_TAIL_FAR_DEN, r);
            num / den
        };

        if !val.is_finite() {
            return Err(SampleSizeError::Internal {
                detail: "non-finite quantile computed in tail branch".to_string(),
            });
        }

        Ok(if q < 0.0 { -val } else { val })
    }
}

/// Horner 法による多項式評価（`coeffs` は最高次から順、末尾が定数項）。
///
/// 手書きの入れ子括弧（`(((...) * r + c) * r + c) ...`）は AS241 のような
/// 7〜8 項の多項式では開閉括弧の対応を誤りやすいため、係数配列とループへ
/// 分離して機械的に評価する。演算順は係数配列の並びで固定され、決定性
/// （REQ-25）を崩さない。
fn horner(coeffs: &[f64], x: f64) -> f64 {
    let mut acc = 0.0;
    for &c in coeffs {
        acc = acc * x + c;
    }
    acc
}

/// AS241 中央域（`|q| <= 0.425`）の分子多項式係数（最高次から）。
const AS241_CENTRAL_NUM: [f64; 8] = [
    2_509.080_928_730_122_6,
    33_430.575_583_588_13,
    67_265.770_927_008_7,
    45_921.953_931_549_87,
    13_731.693_765_509_46,
    1_971.590_950_306_551_4,
    133.141_667_891_784_38,
    3.387_132_872_796_366_6,
];

/// AS241 中央域の分母多項式係数（最高次から。末尾は定数項 `1.0`）。
const AS241_CENTRAL_DEN: [f64; 8] = [
    5_226.495_278_852_854,
    28_729.085_735_721_943,
    39_307.895_800_092_71,
    21_213.794_301_586_596,
    5_394.196_021_424_751,
    687.187_007_492_057_9,
    42.313_330_701_600_91,
    1.0,
];

/// AS241 端域・近傍（`r <= 5`）の分子多項式係数。
const AS241_TAIL_NEAR_NUM: [f64; 8] = [
    7.745_450_142_783_414e-4,
    0.022_723_844_989_269_185,
    0.241_780_725_177_450_6,
    1.270_458_252_452_368_4,
    3.647_848_324_763_204_6,
    5.769_497_221_460_691,
    4.630_337_846_156_545,
    1.423_437_110_749_683_6,
];

/// AS241 端域・近傍の分母多項式係数（末尾は定数項 `1.0`）。
const AS241_TAIL_NEAR_DEN: [f64; 8] = [
    1.050_750_071_644_416_8e-9,
    5.475_938_084_995_345e-4,
    0.015_198_666_563_616_457,
    0.148_103_976_427_480_07,
    0.689_767_334_985_1,
    1.676_384_830_183_804,
    2.053_191_626_637_759,
    1.0,
];

/// AS241 端域・遠方（`r > 5`）の分子多項式係数。
const AS241_TAIL_FAR_NUM: [f64; 8] = [
    2.010_334_399_292_288e-7,
    2.711_555_568_743_488e-5,
    0.001_242_660_947_388_078_4,
    0.026_532_189_526_576_123,
    0.296_560_571_828_504_9,
    1.784_826_539_917_291_3,
    5.463_784_911_164_115,
    6.657_904_643_501_104,
];

/// AS241 端域・遠方の分母多項式係数（末尾は定数項 `1.0`）。
const AS241_TAIL_FAR_DEN: [f64; 8] = [
    2.044_263_103_389_94e-15,
    1.421_511_758_316_446e-7,
    1.846_318_317_510_055e-5,
    7.868_691_311_456_132e-4,
    0.014_875_361_290_850_615,
    0.136_929_880_922_735_8,
    0.599_832_206_555_888,
    1.0,
];

/// [`McNemarSampleSizeAssumption`] から必要件数（丸め前の `f64`）を算出する。
///
/// Connor (1987) の McNemar サンプルサイズ公式:
///
/// - `z_a = -normal_quantile(alpha / 2)`（`normal_quantile(1 - alpha/2)` と
///   同値だが、対称性を使うことで α が小さい場合の `1 - alpha/2` の桁落ちを
///   避ける）
/// - `z_b = normal_quantile(power)`
/// - `d = p_b - p_c`（`> 0` は [`McNemarSampleSizeAssumption::new`] が
///   検証済み）
/// - `s = p_b + p_c`
/// - `n = (z_a * sqrt(s) + z_b * sqrt(s - d^2))^2 / d^2`
///
/// `s - d^2` が丸め誤差で負になる場合（理論上は `s - d^2 = 1 - (p_b + p_c)`
/// を含む恒等式より非負のはずだが、浮動小数演算の丸めで境界上は負に
/// 振れうる）は `sqrt` へ渡す前に検出し、NaN を作らず
/// [`SampleSizeError::Internal`] を返す（fail-closed）。
pub fn mcnemar_sample_size_estimate(
    assumption: &McNemarSampleSizeAssumption,
) -> Result<f64, SampleSizeError> {
    let z_a = -normal_quantile(assumption.alpha / 2.0)?;
    let z_b = normal_quantile(assumption.power)?;

    let d = assumption.p_b - assumption.p_c;
    let s = assumption.p_b + assumption.p_c;
    let variance_term = s - d * d;
    if variance_term < 0.0 {
        return Err(SampleSizeError::Internal {
            detail: "s - d^2 is negative in mcnemar sample size formula".to_string(),
        });
    }

    let numerator = z_a * s.sqrt() + z_b * variance_term.sqrt();
    let n = numerator * numerator / (d * d);

    if !n.is_finite() {
        return Err(SampleSizeError::Internal {
            detail: "non-finite required sample size estimate".to_string(),
        });
    }

    Ok(n)
}

/// [`McNemarSampleSizeAssumption`] から必要件数（[`RequiredSampleSize`]）を
/// 算出する。
///
/// [`mcnemar_sample_size_estimate`] の結果を `ceil` してから
/// [`RequiredSampleSize`] へ変換する。丸めに許容差は加えない（PoC-10 と同じ
/// 規則。許容差の導入は評価契約〔[`crate::significance`]〕の変更にあたるため
/// 行わない）。`ceil` 自体は libm の `f64::ceil` に依存するが、PoC-10・
/// PoC-24 の参照ケースはいずれも次の整数との距離が最小でも約 0.18 あり、
/// OS ごとの丸めの入れ替わりは想定していない（`fixtures/sample_size/
/// PROVENANCE.md` 参照）。
///
/// `ceil` 後の値が [`MAX_EVAL_RECORDS`] を超える場合は
/// [`SampleSizeError::ExceedsRecordLimit`] を返す（この仮定のままでは
/// [`crate::significance::compare_with_baseline`] が確保前に拒否する件数を
/// 超えており、判定不能から抜け出せないため）。`f64` から `u64` への変換は
/// 上限検証を済ませた後にのみ行う（範囲外値を未検証のまま `as` で変換
/// しない）。
pub fn required_sample_size_mcnemar(
    assumption: &McNemarSampleSizeAssumption,
) -> Result<RequiredSampleSize, SampleSizeError> {
    let n = mcnemar_sample_size_estimate(assumption)?;
    let ceil_n = n.ceil();

    if !ceil_n.is_finite() || ceil_n < 1.0 {
        // `d = p_b - p_c > 0` の下で `n` は正の有限値になるはずであり、
        // ここに到達するのは理論上の不整合（fail-closed のガード）。
        return Err(SampleSizeError::Internal {
            detail: "ceil(n) is not a positive finite value".to_string(),
        });
    }

    if ceil_n > MAX_EVAL_RECORDS as f64 {
        // `ceil_n` は有限・正であることを確認済みだが `u64` の範囲を大きく
        // 超えうるため、エラーメッセージ用の値は丸めた近似表示にとどめる
        // （`as u64` は無限大・NaN を含まない値に対しては安全だが、ここでは
        // 診断目的のため MAX_EVAL_RECORDS を超えた事実だけを伝えれば十分）。
        return Err(SampleSizeError::ExceedsRecordLimit {
            required: MAX_EVAL_RECORDS as u64 + 1,
            limit: MAX_EVAL_RECORDS,
        });
    }

    // 上で `1.0 <= ceil_n <= MAX_EVAL_RECORDS as f64`（`MAX_EVAL_RECORDS`
    // は 1_000_000 で `u64` の範囲に十分収まる）であることを検証済みのため、
    // `as u64` は情報を失わない。
    let required_u64 = ceil_n as u64;
    RequiredSampleSize::new(required_u64).ok_or_else(|| SampleSizeError::Internal {
        detail: "ceil(n) rounded to zero unexpectedly".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 絶対誤差・相対誤差のいずれかが `tol` 以下なら一致とみなす
    /// （`.claude/rules/coding-rust.md`「浮動小数の比較は許容差を明示する」。
    /// 決定性の契約〔REQ-25〕が定める 1e-9 を使う）。
    fn approx_eq(actual: f64, expected: f64, tol: f64) -> bool {
        let abs_diff = (actual - expected).abs();
        abs_diff <= tol || abs_diff <= tol * expected.abs()
    }

    /// `normal_quantile(0.5)` は 0.0（浮動小数の丸めで極小の非ゼロ値になる
    /// 実装もあるため、絶対誤差 1e-9 で許容する）。
    #[test]
    fn normal_quantile_median_is_zero() {
        let q = normal_quantile(0.5).unwrap();
        assert!(approx_eq(q, 0.0, 1e-9), "q={q}");
    }

    /// `normal_quantile(0.8)`: fixtures/sample_size/known_values.json 参照。
    #[test]
    fn normal_quantile_power_0_8() {
        let q = normal_quantile(0.8).unwrap();
        assert!(approx_eq(q, 0.8416212335729144, 1e-9), "q={q}");
    }

    /// `normal_quantile(0.975)`: alpha=0.05 の z_a。
    /// fixtures/sample_size/known_values.json 参照。
    #[test]
    fn normal_quantile_alpha_0_05_two_sided() {
        let q = normal_quantile(0.975).unwrap();
        assert!(approx_eq(q, 1.9599639845400536, 1e-9), "q={q}");
    }

    /// `normal_quantile(1e-10)`: 端域（tail branch, r > 5）の既知値。
    /// fixtures/sample_size/known_values.json 参照。
    #[test]
    fn normal_quantile_small_p_tail_branch() {
        let q = normal_quantile(1e-10).unwrap();
        assert!(approx_eq(q, -6.361340902404056, 1e-9), "q={q}");
    }

    /// 対称性: `normal_quantile(p) ≈ -normal_quantile(1 - p)`。
    #[test]
    fn normal_quantile_is_antisymmetric() {
        for p in [0.1, 0.3, 0.4, 0.6, 0.9, 0.99, 1e-6] {
            let a = normal_quantile(p).unwrap();
            let b = normal_quantile(1.0 - p).unwrap();
            assert!(approx_eq(a, -b, 1e-9), "p={p} a={a} b={b}");
        }
    }

    /// 定義域外（0.0・1.0・負・1 超・NaN・無限大）は `QuantileOutOfDomain`。
    #[test]
    fn normal_quantile_rejects_out_of_domain() {
        for p in [
            0.0,
            1.0,
            -0.1,
            1.1,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            assert_eq!(
                normal_quantile(p),
                Err(SampleSizeError::QuantileOutOfDomain)
            );
        }
    }

    /// `McNemarSampleSizeAssumption::new`: NaN・非有限は `NonFiniteInput`。
    #[test]
    fn assumption_rejects_non_finite_inputs() {
        assert_eq!(
            McNemarSampleSizeAssumption::new(f64::NAN, 0.05, 0.05, 0.8),
            Err(SampleSizeError::NonFiniteInput { field: "p_b" })
        );
        assert_eq!(
            McNemarSampleSizeAssumption::new(0.15, f64::INFINITY, 0.05, 0.8),
            Err(SampleSizeError::NonFiniteInput { field: "p_c" })
        );
        assert_eq!(
            McNemarSampleSizeAssumption::new(0.15, 0.05, f64::NAN, 0.8),
            Err(SampleSizeError::NonFiniteInput { field: "alpha" })
        );
        assert_eq!(
            McNemarSampleSizeAssumption::new(0.15, 0.05, 0.05, f64::NEG_INFINITY),
            Err(SampleSizeError::NonFiniteInput { field: "power" })
        );
    }

    /// `alpha` が 0 または 1 は `AlphaOutOfRange`。
    #[test]
    fn assumption_rejects_alpha_out_of_range() {
        assert_eq!(
            McNemarSampleSizeAssumption::new(0.15, 0.05, 0.0, 0.8),
            Err(SampleSizeError::AlphaOutOfRange)
        );
        assert_eq!(
            McNemarSampleSizeAssumption::new(0.15, 0.05, 1.0, 0.8),
            Err(SampleSizeError::AlphaOutOfRange)
        );
    }

    /// `power` が 0 または 1 は `PowerOutOfRange`。
    #[test]
    fn assumption_rejects_power_out_of_range() {
        assert_eq!(
            McNemarSampleSizeAssumption::new(0.15, 0.05, 0.05, 0.0),
            Err(SampleSizeError::PowerOutOfRange)
        );
        assert_eq!(
            McNemarSampleSizeAssumption::new(0.15, 0.05, 0.05, 1.0),
            Err(SampleSizeError::PowerOutOfRange)
        );
    }

    /// `p_b == p_c`（差が 0）は `CandidateNotAboveBaseline`。
    #[test]
    fn assumption_rejects_equal_proportions() {
        assert_eq!(
            McNemarSampleSizeAssumption::new(0.1, 0.1, 0.05, 0.8),
            Err(SampleSizeError::CandidateNotAboveBaseline)
        );
    }

    /// `p_b < p_c`（下限基準のほうが強い）は `CandidateNotAboveBaseline`。
    #[test]
    fn assumption_rejects_baseline_above_candidate() {
        assert_eq!(
            McNemarSampleSizeAssumption::new(0.05, 0.15, 0.05, 0.8),
            Err(SampleSizeError::CandidateNotAboveBaseline)
        );
    }

    /// `p_c < 0` は `NegativeBaselineProportion`。
    #[test]
    fn assumption_rejects_negative_p_c() {
        assert_eq!(
            McNemarSampleSizeAssumption::new(0.15, -0.01, 0.05, 0.8),
            Err(SampleSizeError::NegativeBaselineProportion)
        );
    }

    /// `p_b + p_c > 1.0` は `ProportionsExceedOne`。
    #[test]
    fn assumption_rejects_proportions_exceeding_one() {
        assert_eq!(
            McNemarSampleSizeAssumption::new(0.7, 0.4, 0.05, 0.8),
            Err(SampleSizeError::ProportionsExceedOne)
        );
    }

    /// 境界の受理: `p_c = 0.0` は許容される。
    #[test]
    fn assumption_accepts_zero_baseline_proportion() {
        let a = McNemarSampleSizeAssumption::new(0.1, 0.0, 0.05, 0.8).unwrap();
        assert_eq!(a.p_c(), 0.0);
    }

    /// 境界の受理: `p_b + p_c = 1.0` ちょうどは許容される。
    #[test]
    fn assumption_accepts_proportions_summing_to_one() {
        let a = McNemarSampleSizeAssumption::new(0.6, 0.4, 0.05, 0.8).unwrap();
        assert_eq!(a.p_b() + a.p_c(), 1.0);
    }

    /// PoC-10 の仮定（p_b=0.15・p_c=0.05・power=0.8・alpha=0.05）での
    /// 必要件数の丸め前推定値。fixtures/sample_size/known_values.json 参照。
    #[test]
    fn estimate_matches_poc10_simple_case() {
        let a = McNemarSampleSizeAssumption::new(0.15, 0.05, 0.05, 0.8).unwrap();
        let n = mcnemar_sample_size_estimate(&a).unwrap();
        assert!(approx_eq(n, 154.59856956021102, 1e-9), "n={n}");
    }

    /// 同じ仮定での `required_sample_size_mcnemar` は 155
    /// （fixtures/sample_size/known_values.json の `ceil_n`）。
    #[test]
    fn required_sample_size_matches_poc10_simple_case() {
        let a = McNemarSampleSizeAssumption::new(0.15, 0.05, 0.05, 0.8).unwrap();
        let required = required_sample_size_mcnemar(&a).unwrap();
        assert_eq!(required.get(), 155);
    }

    /// PoC-10 事前登録の下限（alpha=0.0125。Holm m=4 最厳段）は 221。
    #[test]
    fn required_sample_size_matches_poc10_holm_m4() {
        let a = McNemarSampleSizeAssumption::new(0.15, 0.05, 0.0125, 0.8).unwrap();
        let required = required_sample_size_mcnemar(&a).unwrap();
        assert_eq!(required.get(), 221);
    }

    /// α が極小（`d = p_b - p_c = 0.01` は小さいが 0 ではない）だと、必要
    /// 件数が [`MAX_EVAL_RECORDS`] を超えて `ExceedsRecordLimit` になる
    /// （この仮定のままでは判定不能から抜け出せないことを事前に検出する。
    /// 丸め前の推定値は約 1,580,637。`f64::MIN_POSITIVE` よりは大きいが
    /// `normal_quantile` の定義域〔開区間 `(0,1)`〕をちょうど満たす
    /// `alpha = 1e-300` を使う）。
    #[test]
    fn required_sample_size_rejects_when_alpha_is_tiny() {
        let a = McNemarSampleSizeAssumption::new(0.06, 0.05, 1e-300, 0.8).unwrap();
        let err = required_sample_size_mcnemar(&a).unwrap_err();
        assert_eq!(
            err,
            SampleSizeError::ExceedsRecordLimit {
                required: MAX_EVAL_RECORDS as u64 + 1,
                limit: MAX_EVAL_RECORDS,
            }
        );
    }

    /// `d`（p_b - p_c）が極小（0.0001）だと必要件数が非常に大きくなり、
    /// こちらも `ExceedsRecordLimit` になる（丸め前の推定値は約
    /// 7.8e11。α・power は通常値のまま、差だけを極小にした場合の検証）。
    #[test]
    fn required_sample_size_rejects_when_difference_is_tiny() {
        let a = McNemarSampleSizeAssumption::new(0.050_001, 0.05, 0.05, 0.8).unwrap();
        let err = required_sample_size_mcnemar(&a).unwrap_err();
        assert_eq!(
            err,
            SampleSizeError::ExceedsRecordLimit {
                required: MAX_EVAL_RECORDS as u64 + 1,
                limit: MAX_EVAL_RECORDS,
            }
        );
    }
}
