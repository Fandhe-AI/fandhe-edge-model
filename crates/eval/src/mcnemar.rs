//! McNemar の正確検定（両側）の統計計算コア。
//!
//! CLI の `evaluate` 工程（REQ-33）が、下限基準（majority 等）とモデル候補を
//! 同一の評価データ上で比較する際に、本モジュールの関数を使って対応のある
//! 正誤から統計量（不一致ペア数 b・c）と p 値を求める想定（REQ-25「下限基準
//! に対する有意性の判定」）。PoC-10
//! `03-poc/scratch-classifier/scripts/stats_mcnemar.py` の計算手順を
//! Rust（std のみ）へ移植したもの。TASK-25.1-1・issue #64。
//!
//! # 範囲（本モジュールが担わないこと）
//!
//! - α（0.05）との比較・「有意に上回る」（`b > c` かつ `p < α`）の判定は
//!   [`mcnemar_exact_two_sided`] の戻り値を使って呼び出し側が行う
//!   （TASK-25.1-2・issue #65）
//! - Holm 補正（複数候補の比較。REQ-26）は未実装（TASK-25.3）
//! - 件数不足（評価件数 N が事前登録の必要件数未満）の「判定不能」の判定は
//!   本モジュールの範囲外で、[`crate::significance::judge`]
//!   （呼び出し側が渡す必要件数と比較する）が担う。必要件数自体を算出する
//!   関数（Connor 式）は [`crate::sample_size::required_sample_size_mcnemar`]
//!   として実装済み（REQ-25・TASK-25.2・issue #66）。本モジュールはどんな
//!   `b`・`c` の組でも計算を試み、資源上限を超える場合のみ
//!   [`McNemarError::TooManyDiscordantPairs`] を返す
//! - [`paired_counts`] は、渡された 2 本の真偽値スライス（各件が正解なら
//!   `true`）から対応のある正誤の件数を数えるだけで、[`EvalRecord`][crate::metrics::EvalRecord]
//!   から「正解」をどう判定するか（予測欠落・`status` 異常・gold 側の欠陥
//!   行の除外等。PoC-10 の規則）は呼び出し側（issue #65）が決める
//! - JSON 入出力・ファイル I/O・CLI 統合は行わない（`Cargo.toml` の
//!   `[dependencies]` は空のまま。依存を追加していない）
//!
//! # 評価契約との関係
//!
//! - 入力（正誤のスライス）は `&` 参照でのみ受け取り、書き換えない
//!   （REQ-27: 評価の前後で評価データのハッシュが一致すること）
//! - p 値は `[0, 1]` の有限値しか保持できない [`PValue`] 型で表し、壊れた値
//!   （NaN・負・1 超）を表現できないようにする
//!   （`.claude/rules/coding-rust.md`「公開 API・型設計」）
//!
//! # 資源上限（REQ-39）
//!
//! 不一致ペア数 `n = b + c` に対する計算量は O(n) で、`Vec` 等の確保は
//! 行わない（逐次和のみ）。ただし極端に大きい `n` はループ回数が
//! 際限なく増えるため、[`MAX_DISCORDANT_PAIRS`] を超える場合は計算せず
//! [`McNemarError::TooManyDiscordantPairs`] を返す。上限の下では
//! `u64 → f64` の変換が `2^53` 未満に収まり、件数の表現に丸め誤差が
//! 生じない。ただし、この変換精度は「誤差の見積もり」節で述べる
//! `ln_choose_n_k` の桁落ち（大きな値どうしの差を取ることによる相対誤差の
//! 蓄積）を抑える根拠にはならない。上限値自体は主に計算量（ループ回数）を
//! 抑えるために設定したものであり、その下で契約の許容差 1e-9
//! （[`evaluation-contract`](../../../.claude/rules/evaluation-contract.md)）
//! を満たすことを検証済みとは主張しない（下記「誤差の見積もり」参照）。
//!
//! # 数値精度・決定性
//!
//! - `ln`・`exp` は `f64::ln`・`f64::exp`（プラットフォームの libm）を使う。
//!   3 OS の CI matrix では結果が `==` で一致するとは限らないため、
//!   呼び出し側・テストは許容差（絶対誤差 1e-9、小さい p 値は相対誤差 1e-9
//!   も併用）で比較する（`.claude/rules/evaluation-contract.md`「決定性と
//!   証拠の種別」。厳しくする方向の運用であり、契約の許容差を緩めてはいない）
//! - ループの向きは固定し、並列化しない（決定性。同一 seed・同一マシン・
//!   同一依存版で結果が一致することを保証するため）
//! - **アンダーフロー**: 真の p 値が `f64` の最小正規化数を大きく下回る
//!   場合（例: `b=1100, c=0` の真値は `2^-1099`）、計算結果は `0.0` になる。
//!   これはエラーとせずそのまま返す（`p < α` の判定は正しく保たれるため）。
//!   `ln_p`（対数スケールの p 値）を公開するかどうかは、Holm 補正
//!   （TASK-25.3）で必要になった時点で改めて検討する（本 issue の範囲外）。
//! - **誤差の見積もり（証拠の種別: 推定・未検証）**: 相対誤差は `n`
//!   （不一致ペア数）とともに増える傾向がある。計画時の試算（有理数の厳密値
//!   との比較）では `n = 650` で相対誤差約 5e-14、`n = 9,700` で約 1e-11、
//!   `n = 39,000` で約 2e-11 だった。実測しているのはこの範囲（最大
//!   `n = 39,000`）までであり、[`MAX_DISCORDANT_PAIRS`]（1000 万）付近の
//!   `n` での誤差は、この範囲からの外挿にすぎず実測していない。外挿は
//!   `ln_choose_n_k`（`ln_2 + ln_choose_n_k - n * ln_2` の桁落ち）の傾向が
//!   `n` の増加とともに単調である前提に基づくため、契約の許容差 1e-9 を
//!   `n` の上限付近まで満たし続けることを保証しない（`.claude/rules/
//!   coding-rust.md`「未実装・簡易実装の箇所は実装済みを装わない」）。
//!   テスト（`tests/mcnemar_known_answer.rs`）は `n <= 9,700` の範囲で
//!   照合している。`MAX_DISCORDANT_PAIRS` を許容差の実測範囲まで下げるか
//!   どうかは、上限を狭める評価契約上の判断（`.claude/rules/
//!   evaluation-contract.md`）にあたるため main の設計判断とし、本 issue の
//!   範囲外とする。

use std::fmt;

/// 1 回の計算で扱う不一致ペア数（`b + c`）の上限（REQ-39）。
///
/// この上限の下では `n as f64` が整数として厳密に表現できる
/// （`2^53` 未満）。上限を超える入力はループ回数の際限ない増加を防ぐため
/// 計算せずに拒否する。
pub const MAX_DISCORDANT_PAIRS: u64 = 10_000_000;

/// `[0, 1]` の有限値しか持てない p 値。
///
/// フィールドは非公開で、生成はモジュール内の関数（[`mcnemar_exact_two_sided`]）
/// に限る。NaN・負・1 超の値を表現できない型にすることで、壊れた p 値が
/// 上位層（CLI の `evaluate` 工程・Holm 補正）へ伝播しないようにする。
///
/// `PartialEq` は意図的に derive しない（`.claude/rules/coding-rust.md`
/// 「浮動小数の比較は許容差を明示し `==` で比較しない」）。呼び出し側
/// （TASK-25.1-2 等）が `==` で比較する経路を型で塞ぎ、[`value`][Self::value]
/// を取り出したうえで許容差付きの比較関数を使わせる。
#[derive(Debug, Clone, Copy)]
pub struct PValue(f64);

impl PValue {
    /// `[0, 1]` の有限値なら `Some` を返す。それ以外（NaN・負・1 超・無限大）は `None`。
    fn new(value: f64) -> Option<Self> {
        if value.is_finite() && (0.0..=1.0).contains(&value) {
            Some(Self(value))
        } else {
            None
        }
    }

    /// 中身の `f64` 値を取り出す。
    pub fn value(&self) -> f64 {
        self.0
    }
}

impl fmt::Display for PValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// McNemar の正確検定（両側）の結果。
///
/// [`PValue`] を含むため `PartialEq` は derive しない（上記 [`PValue`] の
/// ドキュメント参照）。件数（`b`・`c`・`n_discordant`）だけを比較したい
/// 場合はそれぞれのアクセサ経由で比較する。
#[derive(Debug, Clone, Copy)]
pub struct McNemarExact {
    b: u64,
    c: u64,
    p_two_sided: PValue,
}

impl McNemarExact {
    /// 候補のみ正解した不一致ペア数。
    pub fn b(&self) -> u64 {
        self.b
    }

    /// 下限基準のみ正解した不一致ペア数。
    pub fn c(&self) -> u64 {
        self.c
    }

    /// 不一致ペア数の合計（`b + c`）。
    pub fn n_discordant(&self) -> u64 {
        // 生成時に checked_add 済みであり、オーバーフローしない。
        self.b + self.c
    }

    /// 両側の正確 p 値。
    pub fn p_two_sided(&self) -> PValue {
        self.p_two_sided
    }
}

/// 対応のある正誤の集計（候補・下限基準）。
///
/// `paired_counts` が返す。件数はすべて非負であることが保証される
/// （`u64` のため型で表現）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PairedCounts {
    /// 評価対象の総件数。
    pub n: u64,
    /// 候補・下限基準ともに正解した件数。
    pub both_correct: u64,
    /// 候補のみ正解した件数（McNemar の `b`）。
    pub b_candidate_only: u64,
    /// 下限基準のみ正解した件数（McNemar の `c`）。
    pub c_baseline_only: u64,
    /// 候補・下限基準ともに不正解だった件数。
    pub both_wrong: u64,
}

/// [`mcnemar_exact_two_sided`]・[`paired_counts`] が返しうるエラー。
///
/// [`crate::metrics::EvalError`] とは意図的に分けている。`EvalError` は
/// ラベル集合・混同行列の文脈（`EmptyLabels`・`UnknownGoldLabel` 等）に
/// 特化しており、本モジュールの統計計算の失敗とは性質が異なるため
/// （呼び出し側が終了コード（REQ-21）へ写す際にも判別しやすくする狙い。
/// 写像そのものは本モジュールの範囲外）。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum McNemarError {
    /// `b + c`、または `paired_counts` の内訳の合計が `u64` の範囲を超える。
    CountOverflow,
    /// 不一致ペア数（`b + c`）が [`MAX_DISCORDANT_PAIRS`] を超える（REQ-39）。
    TooManyDiscordantPairs {
        /// 実際の不一致ペア数。
        n_discordant: u64,
        /// 上限値。
        limit: u64,
    },
    /// `paired_counts` に渡した 2 本のスライスの長さが一致しない。
    LengthMismatch {
        /// 候補側の長さ。
        candidate: usize,
        /// 下限基準側の長さ。
        baseline: usize,
    },
    /// 理論上到達しないはずの内部不整合（非有限値の算出等）。fail-closed のガード。
    Internal {
        /// 診断用の詳細（データ本文は含めない）。
        detail: String,
    },
}

impl fmt::Display for McNemarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            McNemarError::CountOverflow => write!(f, "count overflow while computing b + c"),
            McNemarError::TooManyDiscordantPairs {
                n_discordant,
                limit,
            } => {
                write!(
                    f,
                    "too many discordant pairs: {n_discordant} exceeds limit {limit}"
                )
            }
            McNemarError::LengthMismatch {
                candidate,
                baseline,
            } => {
                write!(
                    f,
                    "candidate/baseline length mismatch: {candidate} vs {baseline}"
                )
            }
            McNemarError::Internal { detail } => {
                write!(f, "internal mcnemar computation error: {detail}")
            }
        }
    }
}

impl std::error::Error for McNemarError {}

/// 対応のある正誤 2 本（候補・下限基準。各件が正解なら `true`）から、
/// [`PairedCounts`] を数える。
///
/// 「正解」の判定規則（予測欠落・`status` 異常・gold 側の欠陥行の除外等。
/// PoC-10 の規則）は呼び出し側（issue #65）が決め、本関数へは判定済みの
/// 真偽値だけを渡す。入力スライスは書き換えない（REQ-27）。
///
/// 長さが一致しない場合は [`McNemarError::LengthMismatch`] を返す。
/// 空スライス（`n = 0`）はエラーにせず、全件 0 の [`PairedCounts`] を返す
/// （必要件数との比較による「判定不能」の判定は
/// [`crate::significance::judge`] が担う）。
pub fn paired_counts(
    candidate_correct: &[bool],
    baseline_correct: &[bool],
) -> Result<PairedCounts, McNemarError> {
    if candidate_correct.len() != baseline_correct.len() {
        return Err(McNemarError::LengthMismatch {
            candidate: candidate_correct.len(),
            baseline: baseline_correct.len(),
        });
    }

    let mut both_correct: u64 = 0;
    let mut b_candidate_only: u64 = 0;
    let mut c_baseline_only: u64 = 0;
    let mut both_wrong: u64 = 0;

    for (&cand, &base) in candidate_correct.iter().zip(baseline_correct.iter()) {
        match (cand, base) {
            (true, true) => {
                both_correct = both_correct
                    .checked_add(1)
                    .ok_or(McNemarError::CountOverflow)?;
            }
            (true, false) => {
                b_candidate_only = b_candidate_only
                    .checked_add(1)
                    .ok_or(McNemarError::CountOverflow)?;
            }
            (false, true) => {
                c_baseline_only = c_baseline_only
                    .checked_add(1)
                    .ok_or(McNemarError::CountOverflow)?;
            }
            (false, false) => {
                both_wrong = both_wrong
                    .checked_add(1)
                    .ok_or(McNemarError::CountOverflow)?;
            }
        }
    }

    let n = both_correct
        .checked_add(b_candidate_only)
        .and_then(|v| v.checked_add(c_baseline_only))
        .and_then(|v| v.checked_add(both_wrong))
        .ok_or(McNemarError::CountOverflow)?;

    Ok(PairedCounts {
        n,
        both_correct,
        b_candidate_only,
        c_baseline_only,
        both_wrong,
    })
}

/// McNemar の正確検定（両側）を計算する。
///
/// `n = b + c`、`k = min(b, c)` として、`X ~ Binomial(n, 0.5)` の下で
/// `p = min(1, 2 * P[X <= k])` を求める（PoC-10 `stats_mcnemar.py` と同じ
/// 定義）。`n == 0` の場合は `p = 1.0` を返す（エラーにしない）。
///
/// `n` が [`MAX_DISCORDANT_PAIRS`] を超える場合は計算せず
/// [`McNemarError::TooManyDiscordantPairs`] を返す（REQ-39）。
pub fn mcnemar_exact_two_sided(b: u64, c: u64) -> Result<McNemarExact, McNemarError> {
    let n = b.checked_add(c).ok_or(McNemarError::CountOverflow)?;

    if n > MAX_DISCORDANT_PAIRS {
        return Err(McNemarError::TooManyDiscordantPairs {
            n_discordant: n,
            limit: MAX_DISCORDANT_PAIRS,
        });
    }

    let p = if n == 0 {
        1.0
    } else {
        two_sided_p_value(n, b.min(c))?
    };

    let p_two_sided = PValue::new(p).ok_or_else(|| McNemarError::Internal {
        detail: format!("computed p value is not a finite value in [0, 1]: {p}"),
    })?;

    Ok(McNemarExact { b, c, p_two_sided })
}

/// `n`・`k = min(b, c)`（`k <= n / 2`）から両側 p 値を計算する内部関数。
///
/// `Vec` 等の入力サイズに比例する確保をせず、逐次和（線形空間）で
/// 尾部確率を求める。`ln_gamma` は Rust std に無い（`unstable` 機能のみ）
/// ため使わず、対数階乗の差分の和で二項係数の対数を求める。
fn two_sided_p_value(n: u64, k: u64) -> Result<f64, McNemarError> {
    // ln C(n, k) を漸化式 Σ_{i=1..k} (ln(n - i + 1) - ln(i)) で求める。
    // k <= n/2 なので n - i + 1 > 0 が保証される。
    let mut ln_choose_n_k = 0.0_f64;
    for i in 1..=k {
        // i <= k <= n なので (n - i + 1) は 1 以上。
        let n_minus_i_plus_1 = (n - i + 1) as f64;
        let i_f = i as f64;
        ln_choose_n_k += n_minus_i_plus_1.ln() - i_f.ln();
    }

    // 尾部の和 Σ_{i=0..=k} C(n, i) を、最大項 C(n, k) で割った線形空間の
    // 相対値 s として逐次計算する。比 C(n, i) / C(n, i + 1) = (i + 1) / (n - i)。
    // ループの向きは固定し（i = k-1, ..., 0）、並列化しない（決定性）。
    let mut s = 1.0_f64;
    let mut t = 1.0_f64;
    let mut i = k;
    while i > 0 {
        // t は C(n, i-1) / C(n, k) に対応する項を漸化的に求める。
        // 比: C(n, i-1) / C(n, i) = i / (n - i + 1)。
        let i_f = i as f64;
        let n_minus_i_plus_1 = (n - i + 1) as f64;
        t *= i_f / n_minus_i_plus_1;
        s += t;
        i -= 1;
    }

    // ln p = ln 2 + ln C(n,k) - n * ln 2 + ln s
    //
    // s は 1.0 から始めて正の項だけを加算するため常に有限かつ 1 以上、
    // ln_choose_n_k は n <= MAX_DISCORDANT_PAIRS の下で有界な有限和、
    // n_f * ln_2 も有限のため、ln_p は理論上必ず有限になる（無限大や
    // NaN にはならない）。exp(ln_p) がアンダーフローして 0.0 になることは
    // 起こりうる（意図した挙動。モジュール冒頭の「アンダーフロー」節を参照）。
    let ln_2 = std::f64::consts::LN_2;
    let n_f = n as f64;
    let ln_p = ln_2 + ln_choose_n_k - n_f * ln_2 + s.ln();

    let p = ln_p.exp();
    if !p.is_finite() {
        // 上記の理由により理論上到達しないはずの経路。万一到達した場合に
        // 壊れた p 値を返さないための fail-closed ガード。
        return Err(McNemarError::Internal {
            detail: format!("exp(ln_p) is not finite: ln_p={ln_p}"),
        });
    }

    Ok(p.min(1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-9;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    /// REQ-39: `b + c` が `u64` を溢れる場合は計算せず `CountOverflow` を返す。
    #[test]
    fn count_overflow_on_add() {
        let err = mcnemar_exact_two_sided(u64::MAX, 1).unwrap_err();
        assert_eq!(err, McNemarError::CountOverflow);
    }

    /// REQ-39: 不一致ペア数が上限を 1 件超えると計算せず拒否する。
    #[test]
    fn too_many_discordant_pairs_rejected() {
        let n = MAX_DISCORDANT_PAIRS + 1;
        let err = mcnemar_exact_two_sided(n, 0).unwrap_err();
        assert_eq!(
            err,
            McNemarError::TooManyDiscordantPairs {
                n_discordant: n,
                limit: MAX_DISCORDANT_PAIRS,
            }
        );
    }

    /// REQ-39: 上限ちょうどの不一致ペア数は受け付ける（境界値）。
    #[test]
    fn max_discordant_pairs_boundary_is_accepted() {
        // b = c = MAX/2 として n = MAX_DISCORDANT_PAIRS ちょうどにする。
        let half = MAX_DISCORDANT_PAIRS / 2;
        let result = mcnemar_exact_two_sided(half, half);
        assert!(result.is_ok());
    }

    /// `paired_counts`: 長さ不一致は `LengthMismatch` を返す。
    #[test]
    fn paired_counts_length_mismatch() {
        let candidate = [true, false, true];
        let baseline = [true, false];
        let err = paired_counts(&candidate, &baseline).unwrap_err();
        assert_eq!(
            err,
            McNemarError::LengthMismatch {
                candidate: 3,
                baseline: 2,
            }
        );
    }

    /// 空入力は件数がすべて 0 の `PairedCounts` を返す（エラーにしない）。
    #[test]
    fn paired_counts_empty_input() {
        let counts = paired_counts(&[], &[]).unwrap();
        assert_eq!(
            counts,
            PairedCounts {
                n: 0,
                both_correct: 0,
                b_candidate_only: 0,
                c_baseline_only: 0,
                both_wrong: 0,
            }
        );
    }

    /// `b + c == 0` のときは p = 1.0（PoC と同じ規約）。
    #[test]
    fn zero_discordant_pairs_gives_p_one() {
        let result = mcnemar_exact_two_sided(0, 0).unwrap();
        assert_eq!(result.p_two_sided().value(), 1.0);
        assert_eq!(result.n_discordant(), 0);
    }

    /// PoC-10 の selftest から移植した合成例（gold 側の欠陥行を除いた 5 行）。
    #[test]
    fn paired_counts_matches_poc_selftest_example() {
        let candidate = [true, true, true, false, false];
        let baseline = [true, true, false, false, true];
        let counts = paired_counts(&candidate, &baseline).unwrap();
        assert_eq!(
            counts,
            PairedCounts {
                n: 5,
                both_correct: 2,
                b_candidate_only: 1,
                c_baseline_only: 1,
                both_wrong: 1,
            }
        );
        let result =
            mcnemar_exact_two_sided(counts.b_candidate_only, counts.c_baseline_only).unwrap();
        assert!(approx_eq(result.p_two_sided().value(), 1.0));
    }

    /// `b`・`c` の判定方向の取り違えを検出する非対称ケース。
    ///
    /// `paired_counts_matches_poc_selftest_example` は `b == c == 1` の
    /// 対称ケースのみのため、`(true, false)` を `b`（候補のみ正解）に、
    /// `(false, true)` を `c`（下限基準のみ正解）に数える向きを取り違えても
    /// 検出できない。本テストは `b != c` にすることで、取り違えが起きれば
    /// `b_candidate_only`・`c_baseline_only` の値が入れ替わって失敗する
    /// ようにする。
    #[test]
    fn paired_counts_direction_is_not_swapped() {
        // 候補: 正解, 正解, 不正解 / 下限基準: 不正解, 不正解, 不正解
        // → 候補のみ正解した件（b）が 2、下限基準のみ正解した件（c）は 0。
        let candidate = [true, true, false];
        let baseline = [false, false, false];
        let counts = paired_counts(&candidate, &baseline).unwrap();
        assert_eq!(
            counts,
            PairedCounts {
                n: 3,
                both_correct: 0,
                b_candidate_only: 2,
                c_baseline_only: 0,
                both_wrong: 1,
            }
        );
    }

    /// 既知値（有理数の厳密計算との照合）: b=3, c=1 → p = 5/8。
    #[test]
    fn known_value_b3_c1() {
        let result = mcnemar_exact_two_sided(3, 1).unwrap();
        assert!(approx_eq(result.p_two_sided().value(), 0.625));
    }

    /// 既知値: b=6, c=0 → p = 1/32。
    #[test]
    fn known_value_b6_c0() {
        let result = mcnemar_exact_two_sided(6, 0).unwrap();
        assert!(approx_eq(result.p_two_sided().value(), 0.03125));
    }

    /// 対称性: p(b, c) == p(c, b)。
    #[test]
    fn symmetric_in_b_and_c() {
        let forward = mcnemar_exact_two_sided(9, 3).unwrap();
        let backward = mcnemar_exact_two_sided(3, 9).unwrap();
        assert!(approx_eq(
            forward.p_two_sided().value(),
            backward.p_two_sided().value()
        ));
    }
}
