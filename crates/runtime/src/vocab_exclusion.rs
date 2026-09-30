//! 語彙ファイル超過構成の除外記録（REQ-30 境界値・TASK-30.3・#125）。
//!
//! # 役割と呼び出し文脈
//!
//! 成果物・推論 SDK 層の一部。語彙ファイルを持つ構成（PoC-13 の Qwen2.5 既存語彙の流用。実測合計
//! 46,365,993 バイト。証拠種別: 実機）が容量の目安 40MB を超える場合に、「超過」として記録し、
//! 既定候補（自動選定の対象）から外す。CLI の `select`・既定候補の解決（TASK-19.2）・`package`
//! 工程（TASK-33.x）が、候補ごとに [`crate::capacity::measure_package`] で得た内訳を
//! [`screen_vocab_candidates`] へ渡す。CLI の `select` 工程が接続済み（現行の既定候補 c1・c3 は
//! 語彙を ONNX グラフ内に持ち語彙ファイルが無いため、実際に除外が起きるのは語彙ファイルを持つ
//! 構成が加わったとき）。`package` 工程・既定候補の解決への接続は未着手。JSON 直列化は CLI 側。
//!
//! # 判定規則
//!
//! 比較の対象は語彙成分単体ではなく**パッケージ合計**である（PoC-13 の語彙ファイルは 7.0MB で、
//! 46.4MB は埋め込みを含む合計の値）。除外の条件は次の両方を満たすこと。
//!
//! - 語彙ファイルを持つ（呼び出し側が `has_vocab_file` で明示する。0 バイトのファイルも持つ扱い）
//! - 合計が 40,000,000 バイト（10 進）を超える（ちょうど 40,000,000 は目安内）
//!
//! 成分 [`PackageComponent::VocabOrFeatureTransform`] は語彙ファイルと特徴量変換の定義を同じ成分に
//! 集計するため、その件数からは語彙ファイルの有無を判別できない。特徴量変換だけを持つ超過構成を
//! 誤って除外しないよう、語彙ファイルの有無は内訳とは独立に呼び出し側から受け取る。
//!
//! 語彙を持たない構成が超過した場合は比較結果（[`GuidelineComparison::Exceeded`]）を記録するが
//! 除外はしない。spec の対象は語彙ファイルを持つ構成に限られ、利用者設定の上限による停止は
//! TASK-30.2 の経路にあるためである（設計判断。見直す場合はユーザー確認が要る）。
//!
//! # 上限の経路との分離
//!
//! 40MB は目安であり、正式な合格ラインではない。本モジュールは既定候補からの除外と記録に限り、
//! 合否・終了コード（`limit_exceeded`=20。[`crate::package_outcome`]）は変えない。
//!
//! # 呼び出し側への契約
//!
//! 計測エラー（[`crate::capacity::CapacityError`]）は本モジュールの入力の手前で起き、「除外」に
//! せず呼び出し側が処理全体を止める（fail-closed。REQ-39）。

use crate::capacity::{CapacityBreakdown, PackageComponent};
use crate::capacity_limit::REFERENCE_CAPACITY_BYTES;
use std::fmt;

/// 容量の目安（バイト。10 進の 40MB。PoC-13 の `fits_40mb`・REQ-30）。
/// 値は [`REFERENCE_CAPACITY_BYTES`] に集約済みで、ここでは別名として参照するだけ。
pub const VOCAB_GUIDELINE_BYTES: u64 = REFERENCE_CAPACITY_BYTES;
/// 語彙ファイルを持つ成果物が成果物ディレクトリに置くファイル名（ある場合のみ）。
/// `select` 工程はこのファイルの有無で `has_vocab_file` を決め、存在すれば容量計測の対象
/// （[`PackageComponent::VocabOrFeatureTransform`]）に含める（REQ-30・TASK-30.3・#125）。
pub const VOCAB_FILE_NAME: &str = "vocab.json";
/// 1 回の選別で受け付ける候補数の上限（REQ-39）。
pub const MAX_VOCAB_SCREENING_CANDIDATES: usize = 64;
/// 根拠の出典。
pub const VOCAB_GUIDELINE_SOURCE: &str = "PoC-13";

/// 目安との比較結果（合否ではない）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum GuidelineComparison {
    /// 目安以内。
    Within {
        /// パッケージ合計。
        total_bytes: u64,
        /// 目安。
        guideline_bytes: u64,
    },
    /// 目安超過。
    Exceeded {
        /// パッケージ合計。
        total_bytes: u64,
        /// 目安。
        guideline_bytes: u64,
        /// 超過分。
        excess_bytes: u64,
    },
}

/// 除外の理由。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum VocabExclusionReason {
    /// 語彙ファイルを持つ構成の合計が目安を超えた。
    VocabPackageOverGuideline,
}

impl VocabExclusionReason {
    /// 機械可読な識別子。
    pub fn code(self) -> &'static str {
        match self {
            VocabExclusionReason::VocabPackageOverGuideline => "vocab_package_over_guideline",
        }
    }
}

/// 1 候補の判定。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum VocabDecision {
    /// 既定候補に残す。
    Eligible,
    /// 既定候補から外す。
    Excluded(VocabExclusionReason),
}

/// 1 候補の内訳から目安との比較と判定を行う純粋関数（上限の経路とは別。終了コードを持たない）。
///
/// `has_vocab_file` は語彙ファイルを持つ構成か（特徴量変換の定義だけの構成では `false`）。
pub fn assess_vocab_guideline(
    breakdown: &CapacityBreakdown,
    has_vocab_file: bool,
) -> (GuidelineComparison, VocabDecision) {
    let total_bytes = breakdown.total_bytes();
    let has_vocab = has_vocab_file;
    let comparison = match total_bytes.checked_sub(VOCAB_GUIDELINE_BYTES) {
        Some(excess_bytes) if excess_bytes > 0 => GuidelineComparison::Exceeded {
            total_bytes,
            guideline_bytes: VOCAB_GUIDELINE_BYTES,
            excess_bytes,
        },
        _ => GuidelineComparison::Within {
            total_bytes,
            guideline_bytes: VOCAB_GUIDELINE_BYTES,
        },
    };
    let decision = match comparison {
        GuidelineComparison::Exceeded { .. } if has_vocab => {
            VocabDecision::Excluded(VocabExclusionReason::VocabPackageOverGuideline)
        }
        _ => VocabDecision::Eligible,
    };
    (comparison, decision)
}

/// 候補ごとの記録。入力本文・パスは持たない。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VocabGuidelineRecord<K> {
    /// 呼び出し側の候補 ID（runtime は train に依存しないため汎用型）。
    pub candidate: K,
    /// 語彙ファイルを持つ構成か（呼び出し側の申告。特徴量変換だけの構成は `false`）。
    pub has_vocab_file: bool,
    /// 語彙・特徴量変換の成分のバイト数（両者の合算）。
    pub vocab_bytes: u64,
    /// 語彙・特徴量変換の成分のファイル数（両者の合算）。
    pub vocab_file_count: u32,
    /// パッケージ合計。
    pub total_bytes: u64,
    /// 目安との比較。
    pub comparison: GuidelineComparison,
    /// 判定。
    pub decision: VocabDecision,
}

impl<K> VocabGuidelineRecord<K> {
    /// 語彙ファイルを持つ構成か。
    pub fn has_vocab(&self) -> bool {
        self.has_vocab_file
    }

    /// 除外されたか。
    pub fn is_excluded(&self) -> bool {
        matches!(self.decision, VocabDecision::Excluded(_))
    }

    /// 除外の理由コード。除外でなければ `None`。
    pub fn code(&self) -> Option<&'static str> {
        match self.decision {
            VocabDecision::Excluded(r) => Some(r.code()),
            VocabDecision::Eligible => None,
        }
    }

    /// 根拠の出典。
    pub fn source(&self) -> &'static str {
        VOCAB_GUIDELINE_SOURCE
    }

    /// 英語の固定文（数値のみ。候補名・パスを含めない）。
    pub fn public_message(&self) -> String {
        match self.comparison {
            GuidelineComparison::Exceeded {
                total_bytes,
                guideline_bytes,
                excess_bytes,
            } if self.is_excluded() => format!(
                "vocab-bearing package exceeds the {guideline_bytes}-byte guideline by {excess_bytes} bytes (total {total_bytes})"
            ),
            GuidelineComparison::Exceeded {
                total_bytes,
                guideline_bytes,
                excess_bytes,
            } => format!(
                "package without vocab files exceeds the {guideline_bytes}-byte guideline by {excess_bytes} bytes (total {total_bytes}); not excluded"
            ),
            GuidelineComparison::Within {
                total_bytes,
                guideline_bytes,
            } => format!(
                "package is within the {guideline_bytes}-byte guideline (total {total_bytes})"
            ),
        }
    }
}

/// 選別結果。`records` は入力順（決定性）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VocabScreening<K> {
    /// 候補ごとの記録。
    pub records: Vec<VocabGuidelineRecord<K>>,
}

impl<K: Clone> VocabScreening<K> {
    /// 既定候補に残す候補 ID（入力順）。
    pub fn eligible(&self) -> Vec<K> {
        self.records
            .iter()
            .filter(|r| !r.is_excluded())
            .map(|r| r.candidate.clone())
            .collect()
    }

    /// 除外した候補の記録（入力順）。
    pub fn excluded(&self) -> Vec<&VocabGuidelineRecord<K>> {
        self.records.iter().filter(|r| r.is_excluded()).collect()
    }
}

/// 選別のエラー。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum VocabScreeningError {
    /// 候補数が [`MAX_VOCAB_SCREENING_CANDIDATES`] を超えた。
    TooManyCandidates,
    /// 同じ候補 ID が複数あった。
    DuplicateCandidate,
}

impl VocabScreeningError {
    /// 機械可読な識別子。
    pub fn code(self) -> &'static str {
        match self {
            VocabScreeningError::TooManyCandidates => "limit_exceeded",
            VocabScreeningError::DuplicateCandidate => "duplicate_candidate",
        }
    }
}

impl fmt::Display for VocabScreeningError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VocabScreeningError::TooManyCandidates => {
                write!(
                    f,
                    "too many candidates (max {MAX_VOCAB_SCREENING_CANDIDATES})"
                )
            }
            VocabScreeningError::DuplicateCandidate => write!(f, "duplicate candidate"),
        }
    }
}

impl std::error::Error for VocabScreeningError {}

/// 候補ごとの内訳を目安と比較し、語彙超過構成を既定候補から外す（REQ-30 境界値・TASK-30.3）。
///
/// 入力は `(候補 ID, 内訳, 語彙ファイルを持つか)`。件数上限と重複を確認してから記録を確保する
/// （REQ-39）。
pub fn screen_vocab_candidates<K: Clone + PartialEq>(
    candidates: &[(K, CapacityBreakdown, bool)],
) -> Result<VocabScreening<K>, VocabScreeningError> {
    if candidates.len() > MAX_VOCAB_SCREENING_CANDIDATES {
        return Err(VocabScreeningError::TooManyCandidates);
    }
    let mut records: Vec<VocabGuidelineRecord<K>> = Vec::with_capacity(candidates.len());
    for (key, breakdown, has_vocab_file) in candidates {
        if records.iter().any(|r| &r.candidate == key) {
            return Err(VocabScreeningError::DuplicateCandidate);
        }
        let (comparison, decision) = assess_vocab_guideline(breakdown, *has_vocab_file);
        let vocab = breakdown.component(PackageComponent::VocabOrFeatureTransform);
        records.push(VocabGuidelineRecord {
            candidate: key.clone(),
            has_vocab_file: *has_vocab_file,
            vocab_bytes: vocab.bytes,
            vocab_file_count: vocab.file_count,
            total_bytes: breakdown.total_bytes(),
            comparison,
            decision,
        });
    }
    Ok(VocabScreening { records })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bd(sizes: &[(PackageComponent, u64)]) -> CapacityBreakdown {
        CapacityBreakdown::from_sizes(sizes.iter().copied()).expect("breakdown")
    }

    fn qwen() -> CapacityBreakdown {
        bd(&[
            (PackageComponent::Weights, 39_333_412),
            (PackageComponent::VocabOrFeatureTransform, 7_031_673),
            (PackageComponent::LabelTable, 544),
            (PackageComponent::Metadata, 364),
        ])
    }

    #[test]
    fn req30_guideline_is_40_000_000() {
        assert_eq!(VOCAB_GUIDELINE_BYTES, 40_000_000);
    }

    #[test]
    fn req30_boundary_qwen_vocab_reuse_is_excluded() {
        let (cmp, dec) = assess_vocab_guideline(&qwen(), true);
        assert_eq!(
            cmp,
            GuidelineComparison::Exceeded {
                total_bytes: 46_365_993,
                guideline_bytes: 40_000_000,
                excess_bytes: 6_365_993
            }
        );
        assert_eq!(
            dec,
            VocabDecision::Excluded(VocabExclusionReason::VocabPackageOverGuideline)
        );
        let s = screen_vocab_candidates(&[("qwen", qwen(), true)]).expect("screen");
        let r = &s.records[0];
        assert_eq!(r.code(), Some("vocab_package_over_guideline"));
        assert_eq!(r.source(), "PoC-13");
        assert_eq!(r.vocab_bytes, 7_031_673);
        assert_eq!(
            r.public_message(),
            "vocab-bearing package exceeds the 40000000-byte guideline by 6365993 bytes (total 46365993)"
        );
    }

    #[test]
    fn req30_sw2k_within_guideline_is_kept() {
        let b = bd(&[
            (PackageComponent::Weights, 1_018_916),
            (PackageComponent::VocabOrFeatureTransform, 149_305),
            (PackageComponent::LabelTable, 544),
            (PackageComponent::Metadata, 362),
        ]);
        let (cmp, dec) = assess_vocab_guideline(&b, true);
        assert_eq!(
            cmp,
            GuidelineComparison::Within {
                total_bytes: 1_169_127,
                guideline_bytes: 40_000_000
            }
        );
        assert_eq!(dec, VocabDecision::Eligible);
    }

    #[test]
    fn req30_boundary_exact_and_one_over() {
        let at = bd(&[
            (PackageComponent::Weights, 39_000_000),
            (PackageComponent::VocabOrFeatureTransform, 1_000_000),
        ]);
        assert_eq!(assess_vocab_guideline(&at, true).1, VocabDecision::Eligible);
        let over = bd(&[
            (PackageComponent::Weights, 39_000_001),
            (PackageComponent::VocabOrFeatureTransform, 1_000_000),
        ]);
        let (cmp, dec) = assess_vocab_guideline(&over, true);
        assert!(matches!(
            cmp,
            GuidelineComparison::Exceeded {
                excess_bytes: 1,
                ..
            }
        ));
        assert!(matches!(dec, VocabDecision::Excluded(_)));
    }

    #[test]
    fn req30_no_vocab_over_guideline_is_recorded_not_excluded() {
        let b = bd(&[(PackageComponent::Weights, 50_000_000)]);
        let s = screen_vocab_candidates(&[("big", b, false)]).expect("screen");
        let r = &s.records[0];
        assert!(!r.has_vocab());
        assert!(matches!(
            r.comparison,
            GuidelineComparison::Exceeded {
                excess_bytes: 10_000_000,
                ..
            }
        ));
        assert_eq!(r.decision, VocabDecision::Eligible);
        assert_eq!(r.code(), None);
        assert_eq!(
            r.public_message(),
            "package without vocab files exceeds the 40000000-byte guideline by 10000000 bytes (total 50000000); not excluded"
        );
    }

    #[test]
    fn req30_feature_transform_only_over_guideline_is_not_excluded() {
        // 特徴量変換の定義だけを持つ超過構成は、同じ成分に集計されても除外しない。
        let b = bd(&[
            (PackageComponent::Weights, 50_000_000),
            (PackageComponent::VocabOrFeatureTransform, 1_000),
        ]);
        let s = screen_vocab_candidates(&[("ft", b, false)]).expect("screen");
        let r = &s.records[0];
        assert_eq!(r.vocab_file_count, 1);
        assert!(!r.has_vocab());
        assert_eq!(r.decision, VocabDecision::Eligible);
        assert_eq!(s.eligible(), vec!["ft"]);
    }

    #[test]
    fn req30_zero_byte_vocab_file_counts_as_vocab() {
        let b = bd(&[
            (PackageComponent::Weights, 50_000_000),
            (PackageComponent::VocabOrFeatureTransform, 0),
        ]);
        let s = screen_vocab_candidates(&[(1u8, b, true)]).expect("screen");
        assert!(s.records[0].has_vocab());
        assert!(s.records[0].is_excluded());
    }

    #[test]
    fn screening_keeps_input_order_and_splits() {
        let small = bd(&[(PackageComponent::Weights, 10)]);
        let s = screen_vocab_candidates(&[
            ("a", small.clone(), false),
            ("q", qwen(), true),
            ("c", small, false),
        ])
        .expect("screen");
        assert_eq!(s.eligible(), vec!["a", "c"]);
        let ex = s.excluded();
        assert_eq!(ex.len(), 1);
        assert_eq!(ex[0].candidate, "q");
        assert_eq!(
            s.records.iter().map(|r| r.candidate).collect::<Vec<_>>(),
            vec!["a", "q", "c"]
        );
    }

    #[test]
    fn screening_empty_input_is_empty() {
        let s = screen_vocab_candidates::<u8>(&[]).expect("screen");
        assert!(s.records.is_empty());
        assert!(s.eligible().is_empty());
    }

    #[test]
    fn screening_rejects_too_many_and_duplicates() {
        let b = bd(&[(PackageComponent::Weights, 10)]);
        let many: Vec<(usize, CapacityBreakdown, bool)> = (0..=MAX_VOCAB_SCREENING_CANDIDATES)
            .map(|i| (i, b.clone(), false))
            .collect();
        let e = screen_vocab_candidates(&many).expect_err("too many");
        assert_eq!(e, VocabScreeningError::TooManyCandidates);
        assert_eq!(e.code(), "limit_exceeded");
        let e = screen_vocab_candidates(&[(1, b.clone(), false), (1, b, false)]).expect_err("dup");
        assert_eq!(e, VocabScreeningError::DuplicateCandidate);
        assert_eq!(e.code(), "duplicate_candidate");
    }
}
