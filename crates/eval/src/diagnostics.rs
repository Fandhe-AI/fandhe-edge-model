//! 診断レポートの基礎統計（行数・ユニーク数・ラベル別件数）。
//!
//! CLI の `evaluate` 工程（配線は issue #140 で未実装）が、学習データ・評価データの
//! それぞれについて 1 回ずつ [`basic_stats`] を呼び、精度の変化要因を読み解く材料を
//! 得る想定（REQ-29 正常系・TASK-29.1-1・issue #107）。定義は PoC-11 の
//! `build_series.py::stat_of`（rows / unique_inputs / unique_outputs / label_counts）
//! に対応する。期待値の出典ではなく統計の定義の出典であり、テストは手組みの合成
//! データ（証拠種別: テストハーネス）で検証する。
//!
//! - **診断専用**: 結果は合否判定に使わない（TASK-29.1・REQ-29「精度の目安は検討中」）。
//! - **数え方の規則と正規化を一体で受け取る**: ユニーク数の数え方は [`InputKey`] で
//!   指定する。[`InputKey::ByteExact`] は `input` のバイト一致、[`InputKey::Normalized`]
//!   は [`InputNormalizer`]（規則 ID と処理を 1 実装に束ねた trait）で、本モジュールが
//!   各 `input` へ適用してから数える。結果の `input_key_rule` は実際に使った実装から
//!   導出するため、規則 ID と処理を別々に指定して食い違わせることはできない（issue #107
//!   codex/review 指摘）。層の境界のため data 層へは依存せず、実装は呼び出し側が渡す。
//! - **資源の上限**: 行数（[`MAX_EVAL_RECORDS`]）に加え、1 行の入力長・入力の総バイト数・
//!   保持するユニークキーの総バイト数を、正規化・保持の前後で検証する（REQ-39）。
//! - **本文を保持しない**: 結果・エラーは件数とラベル ID のみを持ち、入力本文を
//!   複製・転記しない（`.claude/rules/security.md`）。
//! - 引数は共有参照のみで書き換えない（REQ-27 の評価前後ハッシュ不変と両立）。
//!
//! 未実装: 混同しやすいラベルの組・レポート統合（TASK-29.1-2・#108）、診断限界の
//! 明記（TASK-29.2・#109）、データ量水準別報告（TASK-29.3・#110）、group 数、
//! JSON 直列化（CLI 層）。

use std::borrow::Cow;
use std::collections::BTreeSet;
use std::fmt;

use crate::baseline::validate_label_order;
use crate::metrics::EvalError;
use crate::significance::MAX_EVAL_RECORDS;

/// 集計対象の 1 行（入力とラベル ID。いずれも借用）。
#[derive(Debug, Clone, Copy)]
pub struct StatsRow<'a> {
    /// 入力（数え方は [`InputKey`] に従う）。
    pub input: &'a str,
    /// ラベル ID。
    pub label: &'a str,
}

/// ラベルごとの件数。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabelCount {
    /// ラベル ID。
    pub label: String,
    /// 出現件数（0 もありうる）。
    pub count: u64,
}

/// 1 データセットの基礎統計（診断専用。合否判定には使わない）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BasicStats {
    /// 行数。
    pub n_rows: u64,
    /// ユニークな入力数（`input_key_rule` の規則で数える）。
    pub unique_inputs: u64,
    /// 1 件以上出現したラベルの数（PoC-11 の `unique_outputs`）。
    pub unique_labels: u64,
    /// 宣言順のラベル別件数（0 件のラベルも含む）。
    pub label_counts: Vec<LabelCount>,
    /// ラベル別件数の最小値。
    pub min_label_count: u64,
    /// 最小件数に並ぶラベル（宣言順）。
    pub min_labels: Vec<String>,
    /// ユニーク数の数え方の規則 ID（実際に使った [`InputKey`] から導出。
    /// [`InputKey::ByteExact`] なら [`BYTE_EXACT_RULE`]）。
    pub input_key_rule: &'static str,
}

/// [`InputKey::ByteExact`] の規則 ID。
pub const BYTE_EXACT_RULE: &str = "byte_exact";

/// 1 行の `input`（正規化前・正規化後とも）の最大バイト数（REQ-39 資源の上限）。
///
/// 値はデータ契約層の重複・リーク検査（`fandhe_edge_data::leak::MAX_LEAK_CHECK_INPUT_BYTES`）
/// と同じ暫定値。層の境界のため data 層へは依存せず本層で持つ。
pub const MAX_STATS_INPUT_BYTES: usize = 4096;

/// 走査する `input` の総バイト数と、ユニーク入力として保持するキーの総バイト数の上限
/// （REQ-39。走査・保持の前に検証する。`MAX_EVAL_RECORDS` は行数のみの上限のため別に必要）。
pub const MAX_STATS_TOTAL_INPUT_BYTES: usize = 64 * 1024 * 1024;

/// 正規化規則（規則 ID と、その規則を適用する処理を 1 つの実装に束ねる）。
///
/// 規則 ID と処理を呼び出し側が別々に渡せないようにするための trait。実装（データ契約層の
/// `NfkcWhitespaceNormalizer` に対する薄いアダプター等）が両方を所有する。処理は決定的で
/// あること。
pub trait InputNormalizer {
    /// 規則 ID（英語。結果の `input_key_rule` へ記録される）。
    fn rule_id(&self) -> &'static str;
    /// `input` を正規化し、結果を `out` へ書き込む。
    ///
    /// 出力は上限付きの [`BoundedString`] へ追記する。上限超過は追記の時点で
    /// [`OutputLimitExceeded`] として検出されるため、短い入力から巨大な文字列を
    /// 生成する規則でも、確保が上限を超える前に打ち切れる（REQ-39）。実装は
    /// `push_str` / `push` の `Err` を `?` で呼び出し元へ返すこと。
    fn normalize(&self, input: &str, out: &mut BoundedString) -> Result<(), OutputLimitExceeded>;
}

/// 正規化の出力が上限を超えた（[`BoundedString`] への追記時に検出）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputLimitExceeded;

/// 追記のたびに上限を検査する文字列バッファ（正規化の出力用。REQ-39）。
///
/// 上限を超える追記は、バッファへ反映する前に拒否する。
#[derive(Debug)]
pub struct BoundedString {
    buf: String,
    limit: usize,
}

impl BoundedString {
    fn new(limit: usize) -> Self {
        Self {
            buf: String::new(),
            limit,
        }
    }

    /// 文字列を追記する。追記後の長さが上限を超えるなら何も追記せず `Err`。
    pub fn push_str(&mut self, s: &str) -> Result<(), OutputLimitExceeded> {
        self.buf
            .len()
            .checked_add(s.len())
            .filter(|&n| n <= self.limit)
            .ok_or(OutputLimitExceeded)?;
        self.buf.push_str(s);
        Ok(())
    }

    /// 1 文字を追記する（上限の扱いは [`BoundedString::push_str`] と同じ）。
    pub fn push(&mut self, c: char) -> Result<(), OutputLimitExceeded> {
        let mut tmp = [0u8; 4];
        self.push_str(c.encode_utf8(&mut tmp))
    }

    /// 現在の内容。
    pub fn as_str(&self) -> &str {
        &self.buf
    }
}

/// ユニーク入力数の数え方。
#[derive(Clone, Copy)]
pub enum InputKey<'n> {
    /// `input` のバイト一致で数える（正規化しない）。
    ByteExact,
    /// 各 `input` へ正規化を適用した結果で数える。規則 ID は正規化実装自身から導出する。
    Normalized(&'n dyn InputNormalizer),
}

impl InputKey<'_> {
    fn rule(&self) -> &'static str {
        match self {
            InputKey::ByteExact => BYTE_EXACT_RULE,
            InputKey::Normalized(n) => n.rule_id(),
        }
    }

    /// 数え方のキーを得る。正規化の出力が [`MAX_STATS_INPUT_BYTES`] を超えたら `None`。
    fn key<'a>(&self, input: &'a str) -> Option<Cow<'a, str>> {
        match self {
            InputKey::ByteExact => Some(Cow::Borrowed(input)),
            InputKey::Normalized(n) => {
                let mut out = BoundedString::new(MAX_STATS_INPUT_BYTES);
                n.normalize(input, &mut out).ok()?;
                Some(Cow::Owned(out.buf))
            }
        }
    }
}

/// [`basic_stats`] のエラー。メッセージは英語で、入力本文・ラベル値を含めない。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DiagnosticsError {
    /// ラベル集合の検証エラー（重複は値を含めない [`DiagnosticsError::DuplicateLabel`] へ変換済み）。
    Labels(EvalError),
    /// ラベル集合に重複がある（2 回目の出現位置のみ。ラベル値は含めない）。
    DuplicateLabel {
        /// 重複した側（2 回目の出現）のラベル位置。
        index: usize,
    },
    /// 行が 0 件（集計済みを装わない）。
    EmptyRows,
    /// 行数が [`MAX_EVAL_RECORDS`] を超える（REQ-39。走査前に拒否）。
    TooManyRows {
        /// 渡された行数。
        n_rows: usize,
        /// 上限。
        limit: usize,
    },
    /// `input` が [`MAX_STATS_INPUT_BYTES`] を超える（正規化前後のいずれか。REQ-39）。
    InputTooLong {
        /// 行位置。
        index: usize,
        /// 上限。
        limit: usize,
    },
    /// `input` の総バイト数、または保持するユニーク入力キーの総バイト数が
    /// [`MAX_STATS_TOTAL_INPUT_BYTES`] を超える（REQ-39）。
    TotalInputTooLarge {
        /// 上限。
        limit: usize,
    },
    /// 行のラベルがラベル集合に無い。
    UnknownLabel {
        /// 行位置。
        index: usize,
    },
    /// 桁あふれ・添字不整合（理論上到達しない）。
    Internal {
        /// 詳細（英語）。
        detail: String,
    },
}

impl fmt::Display for DiagnosticsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiagnosticsError::Labels(err) => write!(f, "{err}"),
            DiagnosticsError::DuplicateLabel { index } => {
                write!(f, "duplicate label at label index {index}")
            }
            DiagnosticsError::EmptyRows => write!(f, "rows must not be empty"),
            DiagnosticsError::TooManyRows { n_rows, limit } => {
                write!(f, "too many rows: {n_rows} (limit: {limit})")
            }
            DiagnosticsError::InputTooLong { index, limit } => {
                write!(
                    f,
                    "input too long at row index {index} (limit: {limit} bytes)"
                )
            }
            DiagnosticsError::TotalInputTooLarge { limit } => {
                write!(f, "total input size exceeds limit ({limit} bytes)")
            }
            DiagnosticsError::UnknownLabel { index } => {
                write!(f, "unknown label at row index {index}")
            }
            DiagnosticsError::Internal { detail } => {
                write!(f, "internal diagnostics error: {detail}")
            }
        }
    }
}

impl std::error::Error for DiagnosticsError {}

fn internal(detail: &str) -> DiagnosticsError {
    DiagnosticsError::Internal {
        detail: detail.to_string(),
    }
}

/// 基礎統計を集計する（REQ-29 正常系・TASK-29.1-1）。
///
/// `labels` は宣言順のラベル ID。学習データ・評価データそれぞれで 1 回ずつ呼ぶ。
/// 決定的（`BTreeSet` のみ・浮動小数なし）で panic しない。
pub fn basic_stats(
    labels: &[&str],
    rows: &[StatsRow<'_>],
    input_key: InputKey<'_>,
) -> Result<BasicStats, DiagnosticsError> {
    if rows.len() > MAX_EVAL_RECORDS {
        return Err(DiagnosticsError::TooManyRows {
            n_rows: rows.len(),
            limit: MAX_EVAL_RECORDS,
        });
    }
    let index = validate_label_order(labels).map_err(|err| match err {
        // 重複ラベルの値はエラーへ載せず、位置のみ返す（値を含めない契約）。
        EvalError::DuplicateLabel { .. } => {
            let mut seen: BTreeSet<&str> = BTreeSet::new();
            let dup = labels.iter().position(|l| !seen.insert(l)).unwrap_or(0);
            DiagnosticsError::DuplicateLabel { index: dup }
        }
        other => DiagnosticsError::Labels(other),
    })?;
    if rows.is_empty() {
        return Err(DiagnosticsError::EmptyRows);
    }

    // 正規化・保持の前に、生の入力長と総バイト数を検証する（REQ-39）。
    let mut raw_total: usize = 0;
    for (i, row) in rows.iter().enumerate() {
        if row.input.len() > MAX_STATS_INPUT_BYTES {
            return Err(DiagnosticsError::InputTooLong {
                index: i,
                limit: MAX_STATS_INPUT_BYTES,
            });
        }
        raw_total = raw_total
            .checked_add(row.input.len())
            .filter(|&t| t <= MAX_STATS_TOTAL_INPUT_BYTES)
            .ok_or(DiagnosticsError::TotalInputTooLarge {
                limit: MAX_STATS_TOTAL_INPUT_BYTES,
            })?;
    }

    let mut counts: Vec<u64> = vec![0; labels.len()];
    let mut inputs: BTreeSet<Cow<'_, str>> = BTreeSet::new();
    let mut retained_bytes: usize = 0;
    for (i, row) in rows.iter().enumerate() {
        let &pos = index
            .get(row.label)
            .ok_or(DiagnosticsError::UnknownLabel { index: i })?;
        let slot = counts
            .get_mut(pos)
            .ok_or_else(|| internal("label position out of bounds of counts table"))?;
        *slot = slot
            .checked_add(1)
            .ok_or_else(|| internal("count overflow while tallying labels"))?;
        // 正規化の出力は上限付きバッファへ書かれ、超過は確保の途中で検出される（REQ-39）。
        let key = input_key
            .key(row.input)
            .ok_or(DiagnosticsError::InputTooLong {
                index: i,
                limit: MAX_STATS_INPUT_BYTES,
            })?;
        let key_len = key.len();
        if inputs.insert(key) {
            retained_bytes = retained_bytes
                .checked_add(key_len)
                .filter(|&t| t <= MAX_STATS_TOTAL_INPUT_BYTES)
                .ok_or(DiagnosticsError::TotalInputTooLarge {
                    limit: MAX_STATS_TOTAL_INPUT_BYTES,
                })?;
        }
    }

    let min_label_count = *counts
        .iter()
        .min()
        .ok_or_else(|| internal("empty counts table"))?;
    let unique_labels = counts.iter().filter(|&&c| c > 0).count();
    let mut label_counts = Vec::with_capacity(labels.len());
    let mut min_labels = Vec::new();
    for (&label, &count) in labels.iter().zip(counts.iter()) {
        label_counts.push(LabelCount {
            label: label.to_string(),
            count,
        });
        if count == min_label_count {
            min_labels.push(label.to_string());
        }
    }

    Ok(BasicStats {
        n_rows: u64::try_from(rows.len()).map_err(|_| internal("row count conversion"))?,
        unique_inputs: u64::try_from(inputs.len())
            .map_err(|_| internal("unique input count conversion"))?,
        unique_labels: u64::try_from(unique_labels)
            .map_err(|_| internal("unique label count conversion"))?,
        label_counts,
        min_label_count,
        min_labels,
        input_key_rule: input_key.rule(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-29・TASK-29.1-1: 1 行・1 ラベルの境界。
    #[test]
    fn single_row_single_label() {
        let rows = [StatsRow {
            input: "x",
            label: "A",
        }];
        let s = basic_stats(&["A"], &rows, InputKey::ByteExact).unwrap();
        assert_eq!(s.n_rows, 1);
        assert_eq!(s.unique_inputs, 1);
        assert_eq!(s.unique_labels, 1);
        assert_eq!(s.min_label_count, 1);
        assert_eq!(s.min_labels, vec!["A".to_string()]);
        assert_eq!(s.input_key_rule, "byte_exact");
    }

    /// REQ-29: 正規化規則は関数と一体で適用され、表記違いが同一入力として数えられる。
    #[test]
    fn normalized_key_applies_rule_and_reports_it() {
        fn squash(s: &str) -> String {
            s.split_whitespace().collect::<Vec<_>>().join(" ")
        }
        let rows = [
            StatsRow {
                input: "a  b",
                label: "A",
            },
            StatsRow {
                input: "a b",
                label: "A",
            },
        ];
        let exact = basic_stats(&["A"], &rows, InputKey::ByteExact).unwrap();
        assert_eq!(exact.unique_inputs, 2);
        struct Squash;
        impl InputNormalizer for Squash {
            fn rule_id(&self) -> &'static str {
                "nfkc_whitespace"
            }
            fn normalize(
                &self,
                input: &str,
                out: &mut BoundedString,
            ) -> Result<(), OutputLimitExceeded> {
                out.push_str(&squash(input))
            }
        }
        let s = basic_stats(&["A"], &rows, InputKey::Normalized(&Squash)).unwrap();
        assert_eq!(s.unique_inputs, 1);
        assert_eq!(s.input_key_rule, "nfkc_whitespace");
    }

    /// REQ-29: エラーメッセージに入力本文・ラベル値を含めない。
    #[test]
    fn unknown_label_message_hides_values() {
        let rows = [StatsRow {
            input: "secret-body",
            label: "secret-label",
        }];
        let err = basic_stats(&["A"], &rows, InputKey::ByteExact).unwrap_err();
        assert_eq!(err, DiagnosticsError::UnknownLabel { index: 0 });
        let msg = err.to_string();
        assert_eq!(msg, "unknown label at row index 0");
        assert!(!msg.contains("secret"));
    }

    /// REQ-39: 短い入力から巨大な出力を作る正規化は、上限を超える追記の時点で拒否される
    /// （出力全体を確保し終えてから検査するのではない）。
    #[test]
    fn expanding_normalizer_is_stopped_at_limit() {
        use std::cell::Cell;
        struct Expand<'c>(&'c Cell<usize>);
        impl InputNormalizer for Expand<'_> {
            fn rule_id(&self) -> &'static str {
                "expand"
            }
            fn normalize(
                &self,
                _input: &str,
                out: &mut BoundedString,
            ) -> Result<(), OutputLimitExceeded> {
                // 上限の数千倍を生成しようとするが、超過した最初の追記で打ち切られる。
                for _ in 0..(MAX_STATS_INPUT_BYTES * 1000) {
                    out.push_str("xxxxxxxx")?;
                    self.0.set(self.0.get() + 1);
                }
                Ok(())
            }
        }
        let calls = Cell::new(0);
        let rows = [StatsRow {
            input: "a",
            label: "A",
        }];
        let err = basic_stats(&["A"], &rows, InputKey::Normalized(&Expand(&calls))).unwrap_err();
        assert_eq!(
            err,
            DiagnosticsError::InputTooLong {
                index: 0,
                limit: MAX_STATS_INPUT_BYTES
            }
        );
        assert_eq!(calls.get(), MAX_STATS_INPUT_BYTES / 8);
    }

    /// REQ-39: BoundedString は上限ちょうどまで受け付け、超過する追記は反映しない。
    #[test]
    fn bounded_string_rejects_overflow_without_appending() {
        let mut b = BoundedString::new(4);
        assert_eq!(b.push_str("abc"), Ok(()));
        assert_eq!(b.push_str("de"), Err(OutputLimitExceeded));
        assert_eq!(b.as_str(), "abc");
        assert_eq!(b.push('d'), Ok(()));
        assert_eq!(b.push('e'), Err(OutputLimitExceeded));
        assert_eq!(b.as_str(), "abcd");
    }

    /// REQ-29: 重複ラベルのエラーは位置のみでラベル値を含めない。
    #[test]
    fn duplicate_label_error_hides_value() {
        let rows = [StatsRow {
            input: "x",
            label: "secret-label",
        }];
        let err = basic_stats(
            &["secret-label", "B", "secret-label"],
            &rows,
            InputKey::ByteExact,
        )
        .unwrap_err();
        assert_eq!(err, DiagnosticsError::DuplicateLabel { index: 2 });
        let msg = err.to_string();
        assert_eq!(msg, "duplicate label at label index 2");
        assert!(!msg.contains("secret"));
    }
}
