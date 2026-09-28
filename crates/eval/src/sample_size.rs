//! McNemar 検定で下限基準（majority）との差を検出するための必要評価件数の
//! 事前計算（Connor 1987 のサンプルサイズ公式）。
//!
//! [`crate::significance`]（下限基準に対する有意性判定。REQ-25・
//! TASK-25.1-2・issue #65）は、評価件数が [`crate::significance::RequiredSampleSize`]
//! 未満なら判定を実行せず「判定不能」を返す分岐をすでに実装している
//! （PR #219）。本モジュールはその `RequiredSampleSize` を、検出力（power）・
//! 有意水準（α）・検出したい候補の正解率の差から事前に算出する関数を提供する
//! （REQ-25 異常系・TASK-25.2・issue #66）。[`mcnemar_sample_size_estimate`]
//! は PoC-10 `03-poc/scratch-classifier/scripts/required_n_mcnemar.py` の
//! 計算手順（Connor 1987 の正規近似）を Rust（std のみ）へ移植したもの。
//! ただし [`required_sample_size_mcnemar`] は、この正規近似の `ceil(n)` を
//! そのまま採用するのではなく、評価データ総件数を仮定した両側正確検定の
//! 実際の検出力（[`power_given_total_n`]）で引き上げた値を返す（PR #230
//! レビュー指摘・P0: 正規近似だけでは実際に使う正確検定の検出力を
//! 保証できなかったため）。
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
    /// `alpha` は `(0, 1)` の範囲内だが、非正規数（subnormal）等の極小値で
    /// `alpha / 2.0` が丸めで `0.0` になり、[`normal_quantile`] の定義域
    /// （開区間 `(0, 1)`）を満たせない。構築時点でこの矛盾を検出し、
    /// [`mcnemar_sample_size_estimate`] の呼び出し時になって初めて
    /// `QuantileOutOfDomain` が返る（公開 API の入力契約と計算結果が
    /// 食い違う）事態を防ぐ（P1・PR #230 レビュー指摘）。
    AlphaTooSmallForQuantile,
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
    /// 正規近似の `ceil(n)` が [`EXACT_POWER_SEARCH_MAX_N`] を超えるため、
    /// 正確検定に基づく検出力探索（[`required_sample_size_mcnemar`]）を
    /// 行わずに拒否した。または、探索を行ったが上限内に目標検出力を
    /// 満たす総件数が見つからなかった。
    ///
    /// いずれも fail-closed の計算量上限であり、`ExceedsRecordLimit` とは
    /// 異なる（こちらは [`MAX_EVAL_RECORDS`] よりずっと小さい、正確検定の
    /// 検出力探索固有の暫定上限）。
    ExceedsExactSearchLimit {
        /// 正規近似の `ceil(n)`（探索の起点。上限超過で拒否した場合に限り、
        /// この値が [`EXACT_POWER_SEARCH_MAX_N`] を超えている）。
        ceil_n: u64,
        /// [`EXACT_POWER_SEARCH_MAX_N`] の値。
        limit: u64,
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
            SampleSizeError::AlphaTooSmallForQuantile => {
                write!(
                    f,
                    "alpha is too small: alpha / 2.0 underflows to 0.0, which is outside the domain of normal_quantile"
                )
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
            SampleSizeError::ExceedsExactSearchLimit { ceil_n, limit } => {
                write!(
                    f,
                    "normal-approximation ceil(n) {ceil_n} exceeds exact power search limit {limit}"
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
    /// 3. `alpha / 2.0` が丸めで `0.0` にならない（非正規数等の極小 `alpha`
    ///    は `(0, 1)` の範囲内でも [`normal_quantile`] の定義域を満たせず、
    ///    構築後の計算が必ず失敗するため、構築時点で拒否する）
    /// 4. `p_c >= 0`
    /// 5. `p_b > p_c`（候補が下限基準を上回る方向の差 `d = p_b - p_c > 0` を
    ///    検出する前提。`d <= 0` では検出力の計算が意味を持たない）
    /// 6. `p_b + p_c <= 1.0`（2 つの排反な正解率の和として妥当な範囲）
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
        if alpha / 2.0 <= 0.0 {
            // `alpha` は上の検査で `(0, 1)` の範囲内と確認済みだが、非正規数
            // 等の極小値では `alpha / 2.0` が丸めで `0.0` になりうる。
            // `normal_quantile` の定義域（開区間 `(0, 1)`）を満たせないため、
            // 計算に進む前にここで拒否する。
            return Err(SampleSizeError::AlphaTooSmallForQuantile);
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

/// `exponent * log_base` を計算するが、`exponent == 0.0` のときは
/// `log_base` の値（`-∞` を含む）に関わらず常に `0.0` を返す。
///
/// `x^0 = 1`（`ln(x^0) = 0`）は `x = 0` であっても成り立つ恒等式だが、
/// 素朴に `exponent * log_base` を計算すると `0.0 * f64::NEG_INFINITY` が
/// `NaN` になってしまう。`p_c = 0`（`theta = p_b / (p_b + p_c) = 1.0` と
/// なり `ln(1 - theta) = -∞`）等、確率が厳密に 0 になる境界的な仮定で
/// 対数尤度の項を計算する際に必要になる（PR #230 レビュー指摘・P0の
/// 修正で判明した境界ケース。`p_b=0.95, p_c=0`・`p_b=1.0, p_c=0` の
/// いずれも該当する）。
fn ln_pow(exponent: f64, log_base: f64) -> f64 {
    if exponent == 0.0 {
        0.0
    } else {
        exponent * log_base
    }
}

/// 不一致ペア数 `n` が与えられたときの、両側正確検定の条件付き検出力
/// （power）を計算する。
///
/// 対立仮説の下では、各不一致ペアが独立に確率
/// `theta = p_b / (p_b + p_c)` で候補のみ正解（McNemar の `b` 側）になると
/// 仮定する。つまり `n` 件の不一致ペアのうち候補favor件数
/// `B ~ Binomial(n, theta)`、`C = n - B`。[`crate::significance::judge`]
/// が「候補が有意に優れる」と判定するのは `b > c` かつ両側 p 値が `alpha`
/// 未満のときだけであり、`b < c`（下限基準が有意に優れる）は判定不能でも
/// 合格でもない別の結果のため、本関数の検出力には数えない。
///
/// 棄却域（`b > c` 方向）は [`crate::mcnemar::mcnemar_exact_two_sided`] と
/// 同じ規則（帰無分布 `Binomial(n, 0.5)` の下で両側 p 値が `alpha` 未満）
/// で決まる。`p_two_sided(b, n-b) = min(1, 2 * F(k; n, 0.5))`
/// （`k = min(b, n-b)`、`F` は帰無分布の CDF）という関係を使い、
/// `k = 0, 1, ...` と増やしながら `F(k)` を対数空間で逐次計算する
/// （二項係数の対数を差分更新する手法は [`crate::mcnemar`] の内部関数と
/// 同じ）。`F(k) < alpha / 2` を満たす間だけ、`b = n - k > c = k` 側
/// （`k == n - k` のとき `b == c` となり `b > c` を満たさないため除外する）
/// の `theta` 下の二項確率を足し込み、条件を外れた時点で打ち切る（`p` は
/// `k` について単調非減少なので、それ以降は棄却域に入らない）。
///
/// `F(0) = 2^-n` は `n` が大きいとアンダーフローして `0.0` になりうるが、
/// これは意図した挙動である。`k` が棄却域の境界（`alpha` に応じた
/// 有意水準の閾値。おおむね `n/2` から `O(sqrt(n))` 離れた位置）に近づく
/// までの小さい `k` での過小評価は、その `k` の真の寄与が実際に無視できる
/// 大きさであることに対応するため、最終的な検出力の値を歪めない
/// （[`crate::mcnemar`] の「アンダーフロー」節と同じ考え方）。
///
/// この検出力は「不一致ペアが厳密に `n` 件」という条件の下でのものであり
/// （`n` に依存しない）、評価データ総件数 `N` を仮定したときの検出力は
/// [`power_given_total_n`] が本関数を `N` に依存する形で加重平均する。
///
/// 計算量は棄却域の大きさ（`k <= n/2` 件）に比例する `O(n)` で、`Vec` 等の
/// 確保はしない。
fn exact_test_power(n: u64, theta: f64, alpha: f64) -> Result<f64, SampleSizeError> {
    if n == 0 {
        // `crate::mcnemar` の規約により n=0 では p_two_sided は常に 1.0
        // であり、`alpha < 1.0`（構築時に検証済み）の下では決して有意に
        // ならない。
        return Ok(0.0);
    }

    let ln_theta = theta.ln();
    let ln_one_minus_theta = (1.0 - theta).ln();
    let ln_2 = std::f64::consts::LN_2;
    let n_f = n as f64;
    let half_threshold = alpha / 2.0;
    let half_n = n / 2;

    let mut ln_choose = 0.0_f64; // ln C(n, 0) = 0
    let mut cdf_null = (-n_f * ln_2).exp(); // F(0) = C(n,0) * 2^-n
    let mut power = 0.0_f64;

    let mut k: u64 = 0;
    loop {
        if k > 0 {
            let k_f = k as f64;
            let n_minus_k_plus_1 = (n - k + 1) as f64;
            ln_choose += n_minus_k_plus_1.ln() - k_f.ln();
            let pmf_k = (ln_choose - n_f * ln_2).exp();
            if !pmf_k.is_finite() {
                return Err(SampleSizeError::Internal {
                    detail: "non-finite null pmf in exact_test_power".to_string(),
                });
            }
            cdf_null += pmf_k;
            if !cdf_null.is_finite() {
                return Err(SampleSizeError::Internal {
                    detail: "non-finite null cdf in exact_test_power".to_string(),
                });
            }
        }

        if cdf_null >= half_threshold {
            break;
        }

        let k_f = k as f64;
        if k != n - k {
            // b = n - k > c = k 側のみを数える（`judge` の `b > c` 方向。
            // モジュールコメント参照）。
            let ln_pmf_upper = ln_pow(n_f - k_f, ln_theta) + ln_pow(k_f, ln_one_minus_theta);
            power += (ln_choose + ln_pmf_upper).exp();
        }

        if !power.is_finite() {
            return Err(SampleSizeError::Internal {
                detail: "non-finite power in exact_test_power".to_string(),
            });
        }

        if k >= half_n {
            break;
        }
        k += 1;
    }

    Ok(power.min(1.0))
}

/// 評価データ総件数 `total_n` を仮定したときの、両側正確検定の実際の
/// 検出力（power）を計算する。
///
/// [`RequiredSampleSize`]・[`crate::significance::judge`] が扱う「件数」は
/// 不一致ペア数ではなく評価データの**総件数**（`both_correct` 行も含む）
/// である（`crates/eval/tests/required_sample_size.rs`
/// `undeterminable_when_evaluated_count_is_one_below_required` 等が
/// `total` として組み立てる件数と同じ）。したがって
/// [`mcnemar_sample_size_estimate`]（Connor 式）が返す `n` も総件数を
/// 指しており、[`exact_test_power`]（不一致ペア数が固定で `n` 件という
/// 条件付き検出力）をそのまま総件数の検出力として使うのは誤り
/// （PR #230 レビュー指摘・P0 の修正過程で判明。修正前の実装が
/// `exact_test_power` を総件数にそのまま適用していたところ、
/// `p_b=0.15, p_c=0.05` のような通常のケースで検出力が常に 1 に近い
/// 誤った値になっていた）。
///
/// 正しくは、評価データ総件数 `total_n` 件のうち不一致ペア数
/// `D ~ Binomial(total_n, p_b + p_c)`（各件が独立に確率 `p_b + p_c` で
/// 不一致になると仮定）であり、`D = d` が与えられたときの候補favor件数の
/// 条件付き分布は多項分布の性質により `Binomial(d, theta)`
/// （`theta = p_b / (p_b + p_c)`）になる。両側正確検定は `D` の実現値
/// だけから決まる（[`crate::mcnemar::mcnemar_exact_two_sided`] は `b`・`c`
/// しか見ない）ため、総件数 `total_n` での検出力は `D` の周辺分布で
/// [`exact_test_power`] を加重平均した値になる:
///
/// `power(total_n) = Σ_{d=0}^{total_n} P(D=d; total_n, p_b+p_c) * exact_test_power(d, theta, alpha)`
///
/// `cond_power` は呼び出し側が [`exact_test_power`] で事前計算した、
/// 添字 `d` が条件付き検出力 `exact_test_power(d, theta, alpha)` に対応する
/// キャッシュ（長さ `total_n + 1` 以上。複数の `total_n` にまたがって
/// 再利用でき、二重計算を避ける。[`required_sample_size_mcnemar`] 参照）。
///
/// `D` の pmf（`Binomial(total_n, q)`。`q = p_b + p_c`）は対数空間で
/// 逐次計算する。`q == 1.0`（`p_b + p_c == 1.0` ちょうど。
/// [`McNemarSampleSizeAssumption::new`] が許容する境界値）では
/// `ln(1 - q) = -∞` になるため、[`ln_pow`] で `exponent == 0.0` の場合を
/// 個別に扱う。
///
/// 計算量は `O(total_n)`（`cond_power` の参照は `O(1)`）で、追加の `Vec`
/// 確保はしない。
fn power_given_total_n(total_n: u64, q: f64, cond_power: &[f64]) -> Result<f64, SampleSizeError> {
    if total_n == 0 {
        // 不一致ペアが発生しえず（D=0 のみ）、cond_power[0] は
        // exact_test_power(0, ..) = 0.0 のはず。
        return Ok(0.0);
    }

    let ln_q = q.ln();
    let ln_one_minus_q = (1.0 - q).ln();
    let n_f = total_n as f64;

    let mut ln_choose = 0.0_f64; // ln C(total_n, 0) = 0
    let mut total_power = 0.0_f64;

    for d in 0..=total_n {
        if d > 0 {
            let d_f = d as f64;
            let n_minus_d_plus_1 = (total_n - d + 1) as f64;
            ln_choose += n_minus_d_plus_1.ln() - d_f.ln();
        }

        let d_f = d as f64;
        let ln_pmf_d = ln_choose + ln_pow(d_f, ln_q) + ln_pow(n_f - d_f, ln_one_minus_q);
        let pmf_d = ln_pmf_d.exp();
        if !pmf_d.is_finite() {
            return Err(SampleSizeError::Internal {
                detail: "non-finite discordant-count pmf in power_given_total_n".to_string(),
            });
        }

        let cond = *cond_power
            .get(d as usize)
            .ok_or_else(|| SampleSizeError::Internal {
                detail: "cond_power cache is shorter than total_n + 1".to_string(),
            })?;

        total_power += pmf_d * cond;
        if !total_power.is_finite() {
            return Err(SampleSizeError::Internal {
                detail: "non-finite total power in power_given_total_n".to_string(),
            });
        }
    }

    Ok(total_power.min(1.0))
}

/// 正確検定に基づく検出力探索を行う際の評価データ総件数の上限（暫定値。
/// REQ-39）。
///
/// [`exact_test_power`]・[`power_given_total_n`] は総件数に対して概ね
/// `O(N)` だが、[`required_sample_size_mcnemar`] は候補となる総件数を
/// 1 件ずつ増やしながらこれらを繰り返し呼ぶため、正規近似の `ceil(n)` が
/// 大きい場合（効果量が小さい・`alpha` が小さい仮定）は計算コストが
/// 増大する。[`MAX_EVAL_RECORDS`]（1,000,000）よりずっと小さい値に抑え、
/// 正規近似の `ceil(n)` がこの上限を超える場合は正確検定の検出力探索を
/// 行わずに拒否する（fail-closed。上限値自体は暫定であり、実測に基づく
/// 調整は後続 TASK で行う）。
pub const EXACT_POWER_SEARCH_MAX_N: u64 = 20_000;

/// [`McNemarSampleSizeAssumption`] から必要件数（[`RequiredSampleSize`]）を
/// 算出する。
///
/// [`mcnemar_sample_size_estimate`]（Connor 式の正規近似）が返す `ceil(n)`
/// を探索の起点とし、そこから評価データ総件数を 1 件ずつ増やしながら
/// [`power_given_total_n`]（実際に使う両側正確検定の真の検出力）が
/// 仮定した `power` 以上になる最小の総件数を探す（PR #230 レビュー
/// 指摘・P0: 正規近似の `ceil(n)` と「正確検定で有意になりうる最小件数」の
/// `max` を採用するだけの旧実装では、目標検出力を満たす保証がなかった。
/// 例えば `p_b=0.95, p_c=0, alpha=0.05, power=0.8` では旧実装が 6 を
/// 返すが、6 件全て候補のみ正解というケースで正確検定が有意になる確率は
/// `0.95^6 ≈ 0.735 < 0.8` だった。正しい必要件数は 7 件
/// `required_sample_size_matches_reviewed_p0_case` 参照）。
///
/// 正規近似の `ceil(n)` を探索の起点にできる根拠: [`mcnemar_sample_size_estimate`]
/// は「これ未満では正規近似上も目標検出力に届かない」という下限を与える
/// ため、真の必要件数がこれを下回ることは想定しない。起点から `1` ずつ
/// 増やして最初に条件を満たした総件数を採用する（正確検定の検出力は
/// 総件数について厳密に単調増加するとは限らない〔離散性による小さな
/// 上下動がありうる〕が、[`power_given_total_n`] の `D` に関する加重平均が
/// この上下動を平滑化するため、最初に条件を満たした点を採用する方法は
/// 標準的な実務にならう）。
///
/// `cond_power` キャッシュ（[`exact_test_power`] の結果。総件数に依存
/// しない）は候補の総件数をまたいで使い回し、総件数を 1 増やすごとに
/// 新しい添字 1 件分だけ追加で計算する。これにより探索全体の計算量は
/// キャッシュの構築が `O(N_final)`、[`power_given_total_n`] の呼び出しが
/// 候補数 × `O(N_final)` に抑えられる（`N_final` は最終的に採用する
/// 総件数）。
///
/// 丸めに許容差は加えない（PoC-10 と同じ規則。許容差の導入は評価契約
/// 〔[`crate::significance`]〕の変更にあたるため行わない）。
///
/// 正規近似の `ceil(n)` が [`MAX_EVAL_RECORDS`] を超える場合は
/// [`SampleSizeError::ExceedsRecordLimit`] を、[`EXACT_POWER_SEARCH_MAX_N`]
/// を超える場合は [`SampleSizeError::ExceedsExactSearchLimit`] を返す
/// （前者は [`crate::significance::compare_with_baseline`] が確保前に
/// 拒否する件数を超えている場合、後者は正確検定の検出力探索の計算量上限。
/// いずれも fail-closed）。`f64` から `u64` への変換は上限検証を済ませた
/// 後にのみ行う（範囲外値を未検証のまま `as` で変換しない）。
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
    let ceil_n_u64 = ceil_n as u64;

    if ceil_n_u64 > EXACT_POWER_SEARCH_MAX_N {
        return Err(SampleSizeError::ExceedsExactSearchLimit {
            ceil_n: ceil_n_u64,
            limit: EXACT_POWER_SEARCH_MAX_N,
        });
    }

    let q = assumption.p_b + assumption.p_c;
    let theta = assumption.p_b / q;

    // `cond_power[d]` は `exact_test_power(d, theta, alpha)`（総件数に
    // 依存しない）。候補の総件数を増やすたびに末尾へ追加する。
    let mut cond_power: Vec<f64> = Vec::new();
    let mut required_u64: Option<u64> = None;

    for candidate_n in ceil_n_u64..=EXACT_POWER_SEARCH_MAX_N {
        while (cond_power.len() as u64) <= candidate_n {
            let d = cond_power.len() as u64;
            cond_power.push(exact_test_power(d, theta, assumption.alpha)?);
        }

        let power = power_given_total_n(candidate_n, q, &cond_power)?;
        if power >= assumption.power {
            required_u64 = Some(candidate_n);
            break;
        }
    }

    let required_u64 = required_u64.ok_or(SampleSizeError::ExceedsExactSearchLimit {
        ceil_n: ceil_n_u64,
        limit: EXACT_POWER_SEARCH_MAX_N,
    })?;

    RequiredSampleSize::new(required_u64).ok_or_else(|| SampleSizeError::Internal {
        detail: "required sample size rounded to zero unexpectedly".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcnemar::mcnemar_exact_two_sided;

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

    /// `alpha` が `(0, 1)` の範囲内でも、非正規数（subnormal）等の極小値で
    /// `alpha / 2.0` が丸めで `0.0` になる場合は `AlphaTooSmallForQuantile`
    /// を返す（P1・PR #230 レビュー指摘。`f64::from_bits(1)` は最小の正の
    /// 非正規数で、`/ 2.0` は `0.0` に丸まる）。
    #[test]
    fn assumption_rejects_alpha_too_small_for_quantile() {
        let tiny_subnormal = f64::from_bits(1);
        assert_eq!(
            tiny_subnormal / 2.0,
            0.0,
            "前提: この値は / 2.0 で 0.0 に丸まる"
        );
        assert_eq!(
            McNemarSampleSizeAssumption::new(0.15, 0.05, tiny_subnormal, 0.8),
            Err(SampleSizeError::AlphaTooSmallForQuantile)
        );
    }

    /// 境界の受理: `alpha = 1e-300`（既存テスト
    /// `required_sample_size_rejects_when_alpha_is_tiny` が使う値）は
    /// 正規数であり `alpha / 2.0` がアンダーフローしないため、構築時点では
    /// 引き続き受理される。
    #[test]
    fn assumption_accepts_alpha_1e_minus_300() {
        let alpha: f64 = 1e-300;
        assert!(alpha / 2.0 > 0.0, "前提: 1e-300 は正規数");
        assert!(McNemarSampleSizeAssumption::new(0.06, 0.05, alpha, 0.8).is_ok());
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

    /// `required_sample_size_mcnemar`（評価データ総件数を仮定した正確検定の
    /// 検出力探索）は 168。正規近似の `ceil(n)`（155。
    /// fixtures/sample_size/known_values.json の `ceil_n`）とは一致しない
    /// （正確検定は正規近似より保守的なため、真の必要件数は正規近似を
    /// 上回る。PR #230 レビュー指摘・P0 の修正）。
    #[test]
    fn required_sample_size_matches_poc10_simple_case() {
        let a = McNemarSampleSizeAssumption::new(0.15, 0.05, 0.05, 0.8).unwrap();
        let required = required_sample_size_mcnemar(&a).unwrap();
        assert_eq!(required.get(), 168);
    }

    /// PoC-10 事前登録の下限（alpha=0.0125。Holm m=4 最厳段）の正規近似
    /// `ceil(n)` は 221 だが、正確検定の検出力探索では 229 になる
    /// （上記と同じ理由）。
    #[test]
    fn required_sample_size_matches_poc10_holm_m4() {
        let a = McNemarSampleSizeAssumption::new(0.15, 0.05, 0.0125, 0.8).unwrap();
        let required = required_sample_size_mcnemar(&a).unwrap();
        assert_eq!(required.get(), 229);
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

    /// 正確検定の境界値: `n=5` 件すべて候補のみ正解でも
    /// `p=2^-4=0.0625 >= 0.05` で有意にならない。
    #[test]
    fn exact_two_sided_all_favor_candidate_n5_not_significant_at_alpha_0_05() {
        let exact = mcnemar_exact_two_sided(5, 0).unwrap();
        assert!(approx_eq(exact.p_two_sided().value(), 0.0625, 1e-9));
        assert!(exact.p_two_sided().value() >= 0.05);
    }

    /// 正確検定の境界値: `n=6` 件すべて候補のみ正解なら
    /// `p=2^-5=0.03125 < 0.05` で有意になる。
    #[test]
    fn exact_two_sided_all_favor_candidate_n6_significant_at_alpha_0_05() {
        let exact = mcnemar_exact_two_sided(6, 0).unwrap();
        assert!(approx_eq(exact.p_two_sided().value(), 0.03125, 1e-9));
        assert!(exact.p_two_sided().value() < 0.05);
    }

    /// PR #230 レビュー指摘（P0）の再現ケース: `p_b=1.0, p_c=0.0, alpha=0.05,
    /// power=0.8` では、正規近似（Connor 式）だけだと `ceil(n)=4` を返すが、
    /// 4 件すべて候補のみ正解でも正確検定は `p=0.125 >= 0.05` で有意に
    /// ならない（検出力 0）。修正後は正確検定で到達可能な下限 6 を返す
    /// （`exact_two_sided_all_favor_candidate_n6_significant_at_alpha_0_05`
    /// で `n=6` が正確検定で有意になる最小値であることを確認済み）。
    #[test]
    fn required_sample_size_uses_exact_test_floor_when_normal_approximation_is_too_small() {
        let a = McNemarSampleSizeAssumption::new(1.0, 0.0, 0.05, 0.8).unwrap();

        // 正規近似だけの丸め前推定値は約 3.84（ceil で 4）で、正確検定の
        // 下限 6 を下回ることを確認する（この乖離が P0 指摘の原因）。
        let n_normal_approx = mcnemar_sample_size_estimate(&a).unwrap();
        assert!(approx_eq(n_normal_approx, 3.841_458_820_694_124, 1e-9));
        assert!(n_normal_approx.ceil() < 6.0);

        let required = required_sample_size_mcnemar(&a).unwrap();
        assert_eq!(
            required.get(),
            6,
            "正規近似の ceil(n)=4 ではなく、正確検定で有意になれる最小件数 6 を返すこと"
        );
    }

    /// 同じ再現ケースを Holm 補正後の厳しい α（0.0125）で確認する。
    /// 正確検定の下限は `2^(1-n) < 0.0125` を満たす最小の `n=8`
    /// （`n=7`: `p=2^-6=0.015625 >= 0.0125`。`n=8`: `p=2^-7=0.0078125 <
    /// 0.0125`）。
    #[test]
    fn required_sample_size_uses_exact_test_floor_with_stricter_alpha() {
        let exact_7 = mcnemar_exact_two_sided(7, 0).unwrap();
        assert!(exact_7.p_two_sided().value() >= 0.0125);
        let exact_8 = mcnemar_exact_two_sided(8, 0).unwrap();
        assert!(exact_8.p_two_sided().value() < 0.0125);

        let a = McNemarSampleSizeAssumption::new(1.0, 0.0, 0.0125, 0.8).unwrap();
        let required = required_sample_size_mcnemar(&a).unwrap();
        assert_eq!(required.get(), 8);
    }

    /// PoC-10・PoC-24 の 4 通りの仮定（差が緩やかで正規近似の `ceil(n)` が
    /// もともと大きい）でも、正確検定の検出力探索は正規近似の `ceil(n)`
    /// （155・188・221・272）を一貫して上回る（168・196・229・278。正確
    /// 検定は正規近似より保守的なため。PR #230 レビュー指摘・P0 の修正で
    /// 判明。修正前はこれらの値が正規近似と一致すると誤って固定していた）。
    #[test]
    fn required_sample_size_poc_anchors_exceed_normal_approximation_ceil() {
        let cases = [
            (0.05, 168u64),
            (0.025, 196u64),
            (0.0125, 229u64),
            (0.05 / 12.0, 278u64),
        ];
        for (alpha, expected) in cases {
            let a = McNemarSampleSizeAssumption::new(0.15, 0.05, alpha, 0.8).unwrap();
            let required = required_sample_size_mcnemar(&a).unwrap();
            assert_eq!(required.get(), expected, "alpha={alpha}");
        }
    }

    /// PR #230 レビュー指摘（P0）そのものの再現ケース: `p_b=0.95, p_c=0,
    /// alpha=0.05, power=0.8`。旧実装は 6 を返していたが、6 件全て候補のみ
    /// 正解というケースで正確検定が有意になる確率は `0.95^6 ≈ 0.735 <
    /// 0.8` で目標検出力を満たさない。正しい必要件数は 7
    /// （`P[D>=6] = C(7,6)*0.95^6*0.05 + 0.95^7 ≈ 0.9556 >= 0.8`。
    /// `D ~ Binomial(N, p_b+p_c=0.95)`、`theta=1.0` なので
    /// `cond_power(d)` は `d>=6` で 1、`d<6` で 0。
    /// [`power_given_total_n`] のドキュメント参照）。
    #[test]
    fn required_sample_size_matches_reviewed_p0_case() {
        let a = McNemarSampleSizeAssumption::new(0.95, 0.0, 0.05, 0.8).unwrap();
        let required = required_sample_size_mcnemar(&a).unwrap();
        assert_eq!(required.get(), 7);

        // N=6 では目標検出力に届かないことも固定する（旧実装の誤りの直接
        // 再現）。
        let theta = 1.0;
        let cond_power_0_to_6: Vec<f64> = (0..=6)
            .map(|d| exact_test_power(d, theta, a.alpha()).unwrap())
            .collect();
        let power_at_6 = power_given_total_n(6, a.p_b() + a.p_c(), &cond_power_0_to_6).unwrap();
        assert!(
            approx_eq(power_at_6, 0.95_f64.powi(6), 1e-9),
            "power_at_6={power_at_6}"
        );
        assert!(power_at_6 < 0.8, "power_at_6={power_at_6}");
    }
}
