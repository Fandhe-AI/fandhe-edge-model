//! 分割の seed・規則・各分割のハッシュの記録と永続化（REQ-17・TASK-17.1-2・#45）。
//!
//! # 位置づけ
//!
//! [`crate::split::split_by_group`] が返す [`crate::split::SplitResult`] は
//! メモリ上の分割結果に過ぎず、それ単体では「同じ分割を再現できたか」を
//! 後から検証できない。本モジュールは分割結果に seed・比率・分割規則 ID・
//! 各 split のレコード ID 集合のハッシュを組にした [`SplitRecord`] を作り、
//! JSON への直列化・復元・ハッシュ照合までを提供する。
//!
//! - TASK-17.2（評価データの凍結・#未確定）は [`SplitRecord::digest`] で
//!   凍結対象の分割のレコード ID・ハッシュを取り出す
//! - TASK-17.3（ハッシュ不一致時の停止）は [`SplitRecord::verify_hashes`]・
//!   [`SplitRecord::verify_against`] を CLI 側から呼び、不一致時に
//!   `judged_fail` 等へ倒す判断に使う想定
//!
//! **本モジュールが検出するのは「どのレコード ID がどの split に属するか」の
//! 改ざん・不一致のみ**であり、レコード本文（`input`・`output`）が変わって
//! いないことの保証はしない（評価データファイル本体のハッシュは TASK-17.2 の
//! 凍結ハッシュ、レコード内容の同一性は TASK-27.x の担当。責務を混同しない）。
//!
//! # ハッシュ入力の定義
//!
//! 各 split（train / validation / test）について、その split に属する
//! レコード ID を昇順に集めた `Vec<String>` を
//! [`fandhe_edge_core::canonical::canonical_sha256_hex`] へ渡し、正準化 JSON
//! （例: `["r3","r5","r7"]`）の sha256 を小文字 16 進 64 桁で得る。空の split
//! は `[]` のハッシュ（`4f53cda1…b945`）になる。比率（浮動小数）はハッシュの
//! 入力に含めない（`canonical_json` は浮動小数を拒否するため）。
//!
//! # 前提条件（呼び出し元が守るべきこと）
//!
//! 本モジュールはファイル I/O・サイズ上限検査（REQ-39）を行わない
//! （[`crate`] モジュール doc の「前提条件」を参照）。[`SplitRecord`] を
//! ファイルへ書き出す・読み込む経路はガード層を通過済みの CLI・呼び出し側の
//! 責務とする。

use crate::split::{Groupable, LabelAllocation, Split, SplitError, SplitRatios, SplitResult};
use fandhe_edge_core::canonical::{CanonicalError, canonical_sha256_hex};
use std::collections::BTreeMap;

/// [`SplitRecord`] の JSON スキーマの版。互換性のない変更をしたら値を上げる。
const SPLIT_RECORD_SCHEMA_VERSION: u32 = 1;

/// ハッシュアルゴリズムの識別子（[`SplitRecord::hash_algorithm`]）。
const HASH_ALGORITHM: &str = "sha256";

/// ハッシュ入力の計算規則の識別子（[`SplitRecord::hash_input_rule`]）。
/// 「各 split のレコード ID を昇順に並べた正準化 JSON をハッシュする」という
/// 規則の版であり、規則を変えたら値を変える（TASK-17.3 が「値」だけでなく
/// 「計算方法」も検証できるようにするため）。
const HASH_INPUT_RULE: &str = "canonical-json-sorted-record-ids-v1";

/// 分割の生成と記録をひとまとめにした結果。
///
/// `split_by_group` の分割結果と、それに対応する [`SplitRecord`] が別々の
/// 呼び出しから作られると、記録の seed・比率が実際の分割と食い違う誤用が
/// 起こりうる。[`split_and_record`] だけがこの型を作れるようにし、両者が
/// 常に同じ入力（`records`・`seed`・`ratios`）から生成されたことを型で保証する。
///
/// フィールドは非公開にし、[`RecordedSplit::result`]・[`RecordedSplit::record`]
/// の読み取り専用アクセサのみを公開する。フィールドを `pub` にすると、
/// 呼び出し側が対応しない `SplitResult` と `SplitRecord` を組み合わせて
/// 新しい `RecordedSplit` を組み立てられてしまい、上記の対応保証を型で
/// 強制できなくなるため（レビュー指摘。#210 Cursor Bugbot Medium）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordedSplit {
    result: SplitResult,
    record: SplitRecord,
}

impl RecordedSplit {
    /// `split_and_record` に渡した `records`・`seed`・`ratios` そのものの
    /// 分割結果（メモリ上の割付。[`crate::split::SplitResult`]）。
    #[must_use]
    pub fn result(&self) -> &SplitResult {
        &self.result
    }

    /// 上記 `result` と対応する記録（seed・規則・各分割のハッシュ）。
    #[must_use]
    pub fn record(&self) -> &SplitRecord {
        &self.record
    }
}

/// 分割の seed・規則・各分割のハッシュを保持する記録（REQ-17）。
///
/// フィールドは非公開にし、生成は [`split_and_record`]（内部で
/// `split_by_group` を実行した結果からのみ）または [`SplitRecord::from_json_str`]
/// （構造検証を経た外部 JSON からのみ）に限定する。検証を経ていない値から
/// `SplitRecord` を組み立てられないようにするための設計判断
/// （coding-rust.md「公開 API・型設計」）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitRecord {
    schema_version: u32,
    seed: u64,
    rule: SplitRule,
    hash_algorithm: String,
    hash_input_rule: String,
    splits: SplitDigests,
}

/// 分割規則（seed 以外の、割付を決めるパラメータ一式）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitRule {
    rule_id: String,
    ratios: RecordedRatios,
    per_label: Vec<LabelAllocation>,
}

impl SplitRule {
    /// 割付規則・PRNG 実装の識別子（[`crate::split::split_by_group`] が
    /// 返す `SplitResult::rule_id` と同じ値）。
    #[must_use]
    pub fn rule_id(&self) -> &str {
        &self.rule_id
    }

    /// 分割比率（10 進文字列から復元した [`SplitRatios`]）。
    #[must_use]
    pub fn ratios(&self) -> SplitRatios {
        self.ratios.to_ratios()
    }

    /// ラベルごとの割付内訳。
    #[must_use]
    pub fn per_label(&self) -> &[LabelAllocation] {
        &self.per_label
    }
}

/// 分割比率を 10 進文字列として保持する（[`SplitRatios`] の `f64` を
/// そのまま JSON の number として直列化すると、`serde_json` の既定設定
/// （`float_roundtrip` feature 無効）ではビット単位の往復一致が保証されない
/// ため、`f64` の `Display`（最短往復表現）で文字列化し、`str::parse::<f64>`
/// （正確な丸め）で復元する）。
#[derive(Debug, Clone, PartialEq, Eq)]
struct RecordedRatios {
    train: String,
    validation: String,
    test: String,
}

impl RecordedRatios {
    fn from_ratios(ratios: &SplitRatios) -> Self {
        RecordedRatios {
            train: ratios.train.to_string(),
            validation: ratios.validation.to_string(),
            test: ratios.test.to_string(),
        }
    }

    /// 復元した比率を返す。文字列が数値として読めない場合は
    /// [`SplitRecordError::InvalidRecord`] を返す（`from_json_str` の
    /// 構造検証から呼ばれる。ここでは `SplitRatios::validate` はまだ行わず、
    /// 呼び出し元が続けて検証する）。
    fn to_ratios(&self) -> SplitRatios {
        // `from_json_str` が事前にパース可能性・妥当性を検証済みのため、
        // ここでの `parse` 失敗は起こらない想定だが、万一失敗しても
        // `SplitRatios::validate` が NaN 相当（既定 0.0）を拒否できるよう
        // panic せず 0.0 にフォールバックする（外部入力を扱うコードで
        // `unwrap` / `expect` を使わないという規約に従う）。
        SplitRatios {
            train: self.train.parse().unwrap_or(0.0),
            validation: self.validation.parse().unwrap_or(0.0),
            test: self.test.parse().unwrap_or(0.0),
        }
    }

    /// 3 つの比率文字列がいずれも有限の `f64` として読めるかを確認する。
    fn parse_all(&self) -> Result<(f64, f64, f64), SplitRecordError> {
        let train = self
            .train
            .parse::<f64>()
            .map_err(|_| SplitRecordError::InvalidRecord {
                reason: "rule.ratios.train is not a valid decimal number",
            })?;
        let validation =
            self.validation
                .parse::<f64>()
                .map_err(|_| SplitRecordError::InvalidRecord {
                    reason: "rule.ratios.validation is not a valid decimal number",
                })?;
        let test = self
            .test
            .parse::<f64>()
            .map_err(|_| SplitRecordError::InvalidRecord {
                reason: "rule.ratios.test is not a valid decimal number",
            })?;
        Ok((train, validation, test))
    }
}

/// 3 分割それぞれのハッシュ・レコード ID 集合。
#[derive(Debug, Clone, PartialEq, Eq)]
struct SplitDigests {
    train: SplitDigest,
    validation: SplitDigest,
    test: SplitDigest,
}

impl SplitDigests {
    fn get(&self, split: Split) -> &SplitDigest {
        match split {
            Split::Train => &self.train,
            Split::Validation => &self.validation,
            Split::Test => &self.test,
        }
    }
}

/// 1 つの split（train / validation / test のいずれか）のハッシュと
/// レコード ID 集合。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitDigest {
    record_ids: Vec<String>,
    record_count: usize,
    group_count: usize,
    sha256: String,
}

impl SplitDigest {
    /// この split に属するレコード ID（昇順）。
    #[must_use]
    pub fn record_ids(&self) -> &[String] {
        &self.record_ids
    }

    /// レコード件数（`record_ids().len()` と一致する）。
    #[must_use]
    pub fn record_count(&self) -> usize {
        self.record_count
    }

    /// group 件数。
    #[must_use]
    pub fn group_count(&self) -> usize {
        self.group_count
    }

    /// レコード ID 集合の sha256（小文字 16 進 64 桁）。
    #[must_use]
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
}

/// [`SplitRecord`] の生成・永続化・照合で起こりうるエラー。
///
/// `Display` は英語の固定文言のみを返し、レコード ID・具体的な期待値と
/// 実際の値を含めない（security.md「秘密情報の混入防止」。学習データの
/// 内容を漏らさないため）。
#[derive(Debug)]
#[non_exhaustive]
pub enum SplitRecordError {
    /// 内部で呼ぶ `split_by_group` がエラーを返した。
    Split(SplitError),
    /// レコード ID 集合の正準化・ハッシュ計算に失敗した。
    Canonical(CanonicalError),
    /// JSON としての構文解析に失敗した。
    Json,
    /// JSON の構造検証（schema_version・ハッシュ形式・件数・ソート順・
    /// 比率の妥当性等）に失敗した。`reason` は固定の英語文言（レコード ID・
    /// 実際の値を含めない）。
    InvalidRecord { reason: &'static str },
    /// `schema_version` が未知の値だった。
    UnsupportedSchemaVersion(u32),
    /// `verify_hashes` で 1 つ以上の split のハッシュが記録と一致しなかった。
    HashMismatch { mismatched: Vec<Split> },
    /// `verify_against` で `rule_id` が一致しなかった。
    RuleMismatch,
    /// `verify_against` で 1 つ以上の split のレコード ID・ハッシュが
    /// 再分割の結果と一致しなかった。
    SplitMismatch { split: Split },
    /// `verify_against` で `per_label` の割付内訳が一致しなかった。
    AllocationMismatch,
}

impl std::fmt::Display for SplitRecordError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // `source`（`SplitError`）は固定の英語文言のみを返す `Display`
            // 実装（`crate::split` 参照）を持つため、`{}` で表示する。
            // `{:?}`（Debug）は `DuplicateRecordId` の実際のレコード ID を
            // そのまま出力してしまい security.md「秘密情報の混入防止」に
            // 違反するため、この経路では絶対に使わない。
            SplitRecordError::Split(source) => write!(f, "split failed: {source}"),
            SplitRecordError::Canonical(source) => {
                write!(f, "failed to hash split record ids: {source}")
            }
            SplitRecordError::Json => write!(f, "failed to parse split record JSON"),
            SplitRecordError::InvalidRecord { reason } => {
                write!(f, "invalid split record: {reason}")
            }
            SplitRecordError::UnsupportedSchemaVersion(version) => {
                write!(f, "unsupported split record schema_version: {version}")
            }
            SplitRecordError::HashMismatch { mismatched } => {
                write!(
                    f,
                    "split record hash mismatch in {} split(s)",
                    mismatched.len()
                )
            }
            SplitRecordError::RuleMismatch => {
                write!(f, "split record rule_id does not match re-split result")
            }
            SplitRecordError::SplitMismatch { split } => {
                write!(f, "split record mismatch in {} split", split.as_str())
            }
            SplitRecordError::AllocationMismatch => {
                write!(
                    f,
                    "split record per_label allocation does not match re-split result"
                )
            }
        }
    }
}

impl std::error::Error for SplitRecordError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SplitRecordError::Split(source) => Some(source),
            SplitRecordError::Canonical(source) => Some(source),
            _ => None,
        }
    }
}

impl SplitRecord {
    /// この記録のスキーマ版。
    #[must_use]
    pub fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// 分割に使った seed。
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// 分割規則（規則 ID・比率・ラベル別割付内訳）。
    #[must_use]
    pub fn rule(&self) -> &SplitRule {
        &self.rule
    }

    /// ハッシュアルゴリズム識別子（現状は常に `"sha256"`）。
    #[must_use]
    pub fn hash_algorithm(&self) -> &str {
        &self.hash_algorithm
    }

    /// ハッシュ入力の計算規則識別子。
    #[must_use]
    pub fn hash_input_rule(&self) -> &str {
        &self.hash_input_rule
    }

    /// 指定した split のハッシュ・レコード ID 集合を返す
    /// （TASK-17.2・TASK-17.3 からの参照口）。
    #[must_use]
    pub fn digest(&self, split: Split) -> &SplitDigest {
        self.splits.get(split)
    }

    /// 記録自体からハッシュを再計算し、記録済みの値と一致するかを確かめる。
    ///
    /// # Errors
    ///
    /// 1 つ以上の split でハッシュが一致しない場合、
    /// [`SplitRecordError::HashMismatch`]（不一致だった split 一覧付き）を返す。
    pub fn verify_hashes(&self) -> Result<(), SplitRecordError> {
        let mut mismatched = Vec::new();
        for split in [Split::Train, Split::Validation, Split::Test] {
            let digest = self.splits.get(split);
            let recomputed =
                canonical_sha256_hex(&digest.record_ids).map_err(SplitRecordError::Canonical)?;
            if recomputed != digest.sha256 {
                mismatched.push(split);
            }
        }
        if mismatched.is_empty() {
            Ok(())
        } else {
            Err(SplitRecordError::HashMismatch { mismatched })
        }
    }

    /// 記録の seed・比率で `records` を再分割し、記録内容と一致するかを
    /// 確かめる。一致すれば再現した [`SplitResult`] を返す（「以後の学習・
    /// 評価はこの分割を参照する」ための入口）。
    ///
    /// # Errors
    ///
    /// `records` の再分割自体が失敗した場合は [`SplitRecordError::Split`]、
    /// `rule_id` が一致しない場合は [`SplitRecordError::RuleMismatch`]、
    /// いずれかの split のレコード ID・ハッシュが一致しない場合は
    /// [`SplitRecordError::SplitMismatch`]、`per_label` が一致しない場合は
    /// [`SplitRecordError::AllocationMismatch`] を返す。
    pub fn verify_against<T: Groupable>(
        &self,
        records: &[T],
    ) -> Result<SplitResult, SplitRecordError> {
        let ratios = self.rule.ratios();
        let result = crate::split::split_by_group(records, self.seed, &ratios)
            .map_err(SplitRecordError::Split)?;

        if result.rule_id != self.rule.rule_id {
            return Err(SplitRecordError::RuleMismatch);
        }

        for split in [Split::Train, Split::Validation, Split::Test] {
            let expected = self.splits.get(split);
            let actual_ids = record_ids_for_split(&result, split);
            let actual_hash =
                canonical_sha256_hex(&actual_ids).map_err(SplitRecordError::Canonical)?;
            let actual_group_count = group_count_for_split(&result, split);
            if actual_ids != expected.record_ids
                || actual_hash != expected.sha256
                || actual_group_count != expected.group_count
            {
                return Err(SplitRecordError::SplitMismatch { split });
            }
        }

        if result.per_label != self.rule.per_label {
            return Err(SplitRecordError::AllocationMismatch);
        }

        Ok(result)
    }

    /// この記録を決定的な正準化 JSON へ直列化する
    /// （同一の記録に対して常にバイト単位で同じ出力になる）。
    ///
    /// # Errors
    ///
    /// レコード ID 集合のハッシュ計算に使う [`canonical_sha256_hex`] と同じ
    /// 理由で失敗しうる（実務上は起こらない想定。`Result` で伝播する）。
    pub fn to_json(&self) -> Result<String, SplitRecordError> {
        let dto = SplitRecordDto::from_record(self);
        fandhe_edge_core::canonical::canonical_json(&dto).map_err(SplitRecordError::Canonical)
    }

    /// JSON 文字列から [`SplitRecord`] を復元する（外部入力の経路）。
    ///
    /// 未知のフィールド・不正な `schema_version`・不正な形式の sha256・
    /// `record_count` と `record_ids` の不一致・未ソート / 重複した
    /// `record_ids`・split をまたぐ ID の重複・不正な比率文字列を
    /// `unwrap` / `expect` / 添字アクセスを使わずに拒否する
    /// （coding-rust.md「外部入力」）。
    ///
    /// # Errors
    ///
    /// 構造検証に失敗した場合、対応する [`SplitRecordError`] のバリアントを
    /// 返す。JSON として解析できない場合は [`SplitRecordError::Json`] を返す。
    pub fn from_json_str(s: &str) -> Result<SplitRecord, SplitRecordError> {
        let dto: SplitRecordDto = serde_json::from_str(s).map_err(|_| SplitRecordError::Json)?;
        dto.into_record()
    }
}

/// レコード列から昇順のレコード ID を集める（`SplitResult::by_record` は
/// `BTreeMap` のため、フィルタ後も既に ID の昇順で並ぶ）。
fn record_ids_for_split(result: &SplitResult, split: Split) -> Vec<String> {
    result
        .by_record
        .iter()
        .filter(|(_, s)| **s == split)
        .map(|(id, _)| id.clone())
        .collect()
}

/// `split` に属する group の件数（`SplitResult::by_group` から数える）。
fn group_count_for_split(result: &SplitResult, split: Split) -> usize {
    result.by_group.values().filter(|s| **s == split).count()
}

/// 1 つの split のレコード ID 集合からハッシュを計算し [`SplitDigest`] を作る。
fn build_digest(
    record_ids: Vec<String>,
    group_count: usize,
) -> Result<SplitDigest, SplitRecordError> {
    let sha256 = canonical_sha256_hex(&record_ids).map_err(SplitRecordError::Canonical)?;
    let record_count = record_ids.len();
    Ok(SplitDigest {
        record_ids,
        record_count,
        group_count,
        sha256,
    })
}

/// `records` を group 単位で分割し、seed・分割規則・各分割のハッシュを
/// 記録した [`RecordedSplit`] を作る（REQ-17・TASK-17.1-2 の生成 API）。
///
/// 内部で [`crate::split::split_by_group`] を 1 回だけ呼び、同じ結果から
/// [`SplitResult`] と [`SplitRecord`] の両方を作る（食い違う seed・比率が
/// 記録へ混入する誤用を型で防ぐ。モジュール doc の [`RecordedSplit`] 参照）。
///
/// # Errors
///
/// `split_by_group` が失敗した場合はそのエラーを [`SplitRecordError::Split`]
/// に包んで返す。レコード ID のハッシュ計算に失敗した場合（実務上は
/// 起こらない想定）は [`SplitRecordError::Canonical`] を返す。
pub fn split_and_record<T: Groupable>(
    records: &[T],
    seed: u64,
    ratios: &SplitRatios,
) -> Result<RecordedSplit, SplitRecordError> {
    let result =
        crate::split::split_by_group(records, seed, ratios).map_err(SplitRecordError::Split)?;

    let mut digests = BTreeMap::new();
    for split in [Split::Train, Split::Validation, Split::Test] {
        let ids = record_ids_for_split(&result, split);
        let group_count = group_count_for_split(&result, split);
        digests.insert(split, build_digest(ids, group_count)?);
    }

    // 直前のループで 3 split すべてを挿入済みのため、`remove` は必ず成功する。
    // それでも外部入力の扱いに準じ `unwrap` を避け、万一取得できなかった
    // 場合は空の digest にフォールバックする（`debug_assert` で開発時に検出）。
    let train = digests.remove(&Split::Train).unwrap_or_else(|| {
        debug_assert!(false, "train digest must be present");
        empty_digest()
    });
    let validation = digests.remove(&Split::Validation).unwrap_or_else(|| {
        debug_assert!(false, "validation digest must be present");
        empty_digest()
    });
    let test = digests.remove(&Split::Test).unwrap_or_else(|| {
        debug_assert!(false, "test digest must be present");
        empty_digest()
    });

    let record = SplitRecord {
        schema_version: SPLIT_RECORD_SCHEMA_VERSION,
        seed,
        rule: SplitRule {
            rule_id: result.rule_id.to_string(),
            ratios: RecordedRatios::from_ratios(ratios),
            per_label: result.per_label.clone(),
        },
        hash_algorithm: HASH_ALGORITHM.to_string(),
        hash_input_rule: HASH_INPUT_RULE.to_string(),
        splits: SplitDigests {
            train,
            validation,
            test,
        },
    };

    Ok(RecordedSplit { result, record })
}

/// フォールバック専用の空 digest（到達しない想定のパスでのみ使う）。
fn empty_digest() -> SplitDigest {
    SplitDigest {
        record_ids: Vec::new(),
        record_count: 0,
        group_count: 0,
        sha256: String::new(),
    }
}

// ---------------------------------------------------------------------
// 永続化用 DTO（`#[serde(deny_unknown_fields)]` で外部入力を厳格に検査する）。
// ---------------------------------------------------------------------

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SplitRecordDto {
    schema_version: u32,
    seed: u64,
    rule: SplitRuleDto,
    hash_algorithm: String,
    hash_input_rule: String,
    splits: SplitDigestsDto,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SplitRuleDto {
    rule_id: String,
    ratios: RecordedRatiosDto,
    per_label: Vec<LabelAllocation>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RecordedRatiosDto {
    train: String,
    validation: String,
    test: String,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SplitDigestsDto {
    train: SplitDigestDto,
    validation: SplitDigestDto,
    test: SplitDigestDto,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SplitDigestDto {
    record_ids: Vec<String>,
    record_count: usize,
    group_count: usize,
    sha256: String,
}

impl SplitRecordDto {
    fn from_record(record: &SplitRecord) -> Self {
        let to_digest_dto = |digest: &SplitDigest| SplitDigestDto {
            record_ids: digest.record_ids.clone(),
            record_count: digest.record_count,
            group_count: digest.group_count,
            sha256: digest.sha256.clone(),
        };
        SplitRecordDto {
            schema_version: record.schema_version,
            seed: record.seed,
            rule: SplitRuleDto {
                rule_id: record.rule.rule_id.clone(),
                ratios: RecordedRatiosDto {
                    train: record.rule.ratios.train.clone(),
                    validation: record.rule.ratios.validation.clone(),
                    test: record.rule.ratios.test.clone(),
                },
                per_label: record.rule.per_label.clone(),
            },
            hash_algorithm: record.hash_algorithm.clone(),
            hash_input_rule: record.hash_input_rule.clone(),
            splits: SplitDigestsDto {
                train: to_digest_dto(&record.splits.train),
                validation: to_digest_dto(&record.splits.validation),
                test: to_digest_dto(&record.splits.test),
            },
        }
    }

    fn into_record(self) -> Result<SplitRecord, SplitRecordError> {
        if self.schema_version != SPLIT_RECORD_SCHEMA_VERSION {
            return Err(SplitRecordError::UnsupportedSchemaVersion(
                self.schema_version,
            ));
        }
        if self.hash_algorithm != HASH_ALGORITHM {
            return Err(SplitRecordError::InvalidRecord {
                reason: "unknown hash_algorithm",
            });
        }
        if self.hash_input_rule != HASH_INPUT_RULE {
            return Err(SplitRecordError::InvalidRecord {
                reason: "unknown hash_input_rule",
            });
        }

        let ratios_dto = RecordedRatios {
            train: self.rule.ratios.train,
            validation: self.rule.ratios.validation,
            test: self.rule.ratios.test,
        };
        let (train_r, validation_r, test_r) = ratios_dto.parse_all()?;
        let ratios = SplitRatios {
            train: train_r,
            validation: validation_r,
            test: test_r,
        };
        ratios
            .validate()
            .map_err(|_| SplitRecordError::InvalidRecord {
                reason: "rule.ratios do not form a valid SplitRatios (range/NaN/sum)",
            })?;

        // `per_label` の各エントリの `n_groups` が `train + validation + test`
        // と一致することを検証する。`group_count`（split 単位の合計）の
        // 整合性は `digest_from_dto` で検査済みだが、`per_label`（ラベル単位の
        // 内訳）はここでしか読まないため、改ざんされた割付表（合計が
        // `n_groups` と食い違う値）が `verify_against` 実行まで検出されずに
        // 素通りしてしまう（レビュー指摘。#210 Cursor Bugbot Low）。
        // `usize` の加算は `checked_add` で行い、外部入力由来の値でオーバー
        // フローしても panic せず拒否する（coding-rust.md「外部入力」）。
        for allocation in &self.rule.per_label {
            let sum = allocation
                .train
                .checked_add(allocation.validation)
                .and_then(|partial| partial.checked_add(allocation.test));
            if sum != Some(allocation.n_groups) {
                return Err(SplitRecordError::InvalidRecord {
                    reason: "per_label allocation train + validation + test does not equal n_groups",
                });
            }
        }

        // ラベルの重複を検出する（同じラベルが `per_label` に 2 行以上
        // 現れると、直前の合計検証をすり抜けたまま下の split 別合計照合が
        // 二重計上・過小計上のどちらの向きにも壊れうる。レビュー指摘。
        // #210 codex/review P1）。
        let mut seen_labels: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
        for allocation in &self.rule.per_label {
            if !seen_labels.insert(allocation.label.as_str()) {
                return Err(SplitRecordError::InvalidRecord {
                    reason: "per_label contains a duplicate label",
                });
            }
        }

        let train = digest_from_dto(self.splits.train)?;
        let validation = digest_from_dto(self.splits.validation)?;
        let test = digest_from_dto(self.splits.test)?;

        // `per_label` の各行の内訳合計（既に上で検証済み）を split ごとに
        // 積み上げ、各 split の `group_count` と一致するかを確認する。
        // ラベル行の欠落（`per_label` が実際より少ないラベルしか持たない）は
        // 行ごとの内訳検証だけでは検出できず、積み上げ合計が `group_count`
        // を下回ることで初めて検出できる（レビュー指摘。#210 codex/review
        // P1）。`usize` の加算は `checked_add` で行い、外部入力由来の値の
        // オーバーフローで panic せず拒否する（coding-rust.md「外部入力」）。
        let mut per_label_train_total: usize = 0;
        let mut per_label_validation_total: usize = 0;
        let mut per_label_test_total: usize = 0;
        for allocation in &self.rule.per_label {
            per_label_train_total = per_label_train_total.checked_add(allocation.train).ok_or(
                SplitRecordError::InvalidRecord {
                    reason: "per_label train total overflows usize",
                },
            )?;
            per_label_validation_total = per_label_validation_total
                .checked_add(allocation.validation)
                .ok_or(SplitRecordError::InvalidRecord {
                    reason: "per_label validation total overflows usize",
                })?;
            per_label_test_total = per_label_test_total.checked_add(allocation.test).ok_or(
                SplitRecordError::InvalidRecord {
                    reason: "per_label test total overflows usize",
                },
            )?;
        }
        if per_label_train_total != train.group_count
            || per_label_validation_total != validation.group_count
            || per_label_test_total != test.group_count
        {
            return Err(SplitRecordError::InvalidRecord {
                reason: "per_label allocation totals do not match each split's group_count",
            });
        }

        // split をまたいだ ID の重複を検出する（同一レコードが 2 つの split に
        // 属する記録は、分割が group を跨いだ証拠であり REQ-17 に反する）。
        let mut seen: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
        for digest in [&train, &validation, &test] {
            for id in &digest.record_ids {
                if !seen.insert(id.as_str()) {
                    return Err(SplitRecordError::InvalidRecord {
                        reason: "the same record id appears in more than one split",
                    });
                }
            }
        }

        Ok(SplitRecord {
            schema_version: self.schema_version,
            seed: self.seed,
            rule: SplitRule {
                rule_id: self.rule.rule_id,
                ratios: ratios_dto,
                per_label: self.rule.per_label,
            },
            hash_algorithm: self.hash_algorithm,
            hash_input_rule: self.hash_input_rule,
            splits: SplitDigests {
                train,
                validation,
                test,
            },
        })
    }
}

/// 1 つの split の DTO から [`SplitDigest`] を検証しつつ組み立てる。
fn digest_from_dto(dto: SplitDigestDto) -> Result<SplitDigest, SplitRecordError> {
    if dto.record_count != dto.record_ids.len() {
        return Err(SplitRecordError::InvalidRecord {
            reason: "record_count does not match record_ids length",
        });
    }
    if !is_sorted_and_unique(&dto.record_ids) {
        return Err(SplitRecordError::InvalidRecord {
            reason: "record_ids must be sorted ascending without duplicates",
        });
    }
    if !is_lowercase_hex_64(&dto.sha256) {
        return Err(SplitRecordError::InvalidRecord {
            reason: "sha256 must be 64 lowercase hex characters",
        });
    }
    // `group_count` は「この split に割り付けられた group の件数」であり、
    // 各 group は少なくとも 1 件のレコードをこの split へ割り付ける
    // （`group_count_for_split` 参照）。そのため
    // `record_count == 0 <=> group_count == 0` かつ `group_count <=
    // record_count` が必ず成り立つ。JSON を直接改ざんしてこの範囲外の値
    // （例: レコードが 0 件なのに group_count > 0、または group 件数が
    // レコード件数を超える）を混入させても、`record_ids` 単体からは
    // 検出できないためここで拒否する（record_count 自身は直前で
    // `record_ids.len()` と一致検証済み）。
    if dto.group_count > dto.record_count || (dto.group_count == 0) != (dto.record_count == 0) {
        return Err(SplitRecordError::InvalidRecord {
            reason: "group_count is inconsistent with record_count",
        });
    }

    Ok(SplitDigest {
        record_ids: dto.record_ids,
        record_count: dto.record_count,
        group_count: dto.group_count,
        sha256: dto.sha256,
    })
}

/// 文字列のスライスが昇順かつ重複が無いかを確認する。
fn is_sorted_and_unique(ids: &[String]) -> bool {
    ids.windows(2).all(|pair| {
        let (Some(a), Some(b)) = (pair.first(), pair.get(1)) else {
            return true;
        };
        a < b
    })
}

/// 64 桁の小文字 16 進文字列かどうかを確認する。
fn is_lowercase_hex_64(s: &str) -> bool {
    s.len() == 64
        && s.chars()
            .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestRecord {
        id: String,
        group_id: String,
        label: String,
    }

    impl Groupable for TestRecord {
        fn id(&self) -> &str {
            &self.id
        }
        fn group_id(&self) -> &str {
            &self.group_id
        }
        fn label(&self) -> &str {
            &self.label
        }
    }

    fn record(id: &str, group_id: &str, label: &str) -> TestRecord {
        TestRecord {
            id: id.to_string(),
            group_id: group_id.to_string(),
            label: label.to_string(),
        }
    }

    /// `split.rs` のピン留めテスト（`req17_task17_1_1_deterministic_given_same_seed`）
    /// と同じ入力（r1〜r8・g1〜g7・ラベル a/b・seed 7・既定比率）から作った
    /// 分割記録が、独立に `sha256sum` で計算した具体値と一致することを確認する
    /// （受け入れ条件そのもの。REQ-17・TASK-17.1-2）。
    ///
    /// 期待値の再計算コマンド:
    /// - `printf '%s' '["r3","r5","r7"]' | sha256sum`
    /// - `printf '%s' '["r4","r8"]' | sha256sum`
    /// - `printf '%s' '["r1","r2","r6"]' | sha256sum`
    #[test]
    fn req17_task17_1_2_pinned_hashes_match_independent_sha256sum() {
        let records = vec![
            record("r1", "g1", "a"),
            record("r2", "g1", "a"),
            record("r3", "g2", "a"),
            record("r4", "g3", "a"),
            record("r5", "g4", "a"),
            record("r6", "g5", "b"),
            record("r7", "g6", "b"),
            record("r8", "g7", "b"),
        ];
        let ratios = SplitRatios::default();

        let recorded = split_and_record(&records, 7, &ratios).expect("valid ratios");
        let record = recorded.record().clone();

        assert_eq!(record.seed(), 7);
        assert_eq!(record.rule().rule_id(), recorded.result().rule_id);

        let train = record.digest(Split::Train);
        assert_eq!(train.record_ids(), ["r3", "r5", "r7"]);
        assert_eq!(
            train.sha256(),
            "ee2e2d9dc6ea31ad72a86008dd1f78ff9360db159455c5456ebf973fc123168e"
        );

        let validation = record.digest(Split::Validation);
        assert_eq!(validation.record_ids(), ["r4", "r8"]);
        assert_eq!(
            validation.sha256(),
            "97a883f9e7ed3d61d98ee6f998ffbe44a0a5bbf87e45bad0b1a8fb927609f04d"
        );

        let test = record.digest(Split::Test);
        assert_eq!(test.record_ids(), ["r1", "r2", "r6"]);
        assert_eq!(
            test.sha256(),
            "0810284a881a4dd3dc93a14e99a5e2708011c2b224799709a3b36fbe58444ee0"
        );

        assert!(record.verify_hashes().is_ok());
    }

    /// REQ-17・TASK-17.1-2: 空の入力は 3 分割とも `[]`（`4f53cda1…b945`）に
    /// なり、`verify_hashes` は成功する。
    #[test]
    fn req17_task17_1_2_empty_records_yield_known_empty_hash() {
        let records: Vec<TestRecord> = Vec::new();
        let recorded =
            split_and_record(&records, 0, &SplitRatios::default()).expect("valid ratios");

        for split in [Split::Train, Split::Validation, Split::Test] {
            let digest = recorded.record().digest(split);
            assert!(digest.record_ids().is_empty());
            assert_eq!(
                digest.sha256(),
                "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
            );
        }
        assert!(recorded.record().verify_hashes().is_ok());
    }

    /// REQ-17・TASK-17.1-2: `to_json` → `from_json_str` で内容が保たれ、
    /// `to_json` を 2 回呼んだ出力がバイト単位で一致する（決定性）。
    #[test]
    fn req17_task17_1_2_json_roundtrip_is_deterministic() {
        let records = vec![
            record("r1", "g1", "a"),
            record("r2", "g2", "a"),
            record("r3", "g3", "b"),
        ];
        let recorded =
            split_and_record(&records, 42, &SplitRatios::default()).expect("valid ratios");

        let json_1 = recorded.record().to_json().expect("直列化に失敗しないはず");
        let json_2 = recorded.record().to_json().expect("直列化に失敗しないはず");
        assert_eq!(json_1, json_2, "to_json の出力が決定的でない");

        let restored = SplitRecord::from_json_str(&json_1).expect("復元に失敗しないはず");
        assert_eq!(restored.seed(), recorded.record().seed());
        assert_eq!(
            restored.rule().rule_id(),
            recorded.record().rule().rule_id()
        );
        assert_eq!(
            restored.rule().per_label(),
            recorded.record().rule().per_label()
        );

        let original_ratios = recorded.record().rule().ratios();
        let restored_ratios = restored.rule().ratios();
        assert!((original_ratios.train - restored_ratios.train).abs() < 1e-9);
        assert!((original_ratios.validation - restored_ratios.validation).abs() < 1e-9);
        assert!((original_ratios.test - restored_ratios.test).abs() < 1e-9);

        for split in [Split::Train, Split::Validation, Split::Test] {
            assert_eq!(
                restored.digest(split).record_ids(),
                recorded.record().digest(split).record_ids()
            );
            assert_eq!(
                restored.digest(split).sha256(),
                recorded.record().digest(split).sha256()
            );
        }
        assert!(restored.verify_hashes().is_ok());
    }

    /// REQ-17・TASK-17.1-2: `verify_against` が同じレコード列から再分割した
    /// 結果を検証でき、返る `SplitResult` が元の分割と一致する。
    #[test]
    fn req17_task17_1_2_verify_against_reproduces_same_split_result() {
        let records = vec![
            record("r1", "g1", "a"),
            record("r2", "g2", "a"),
            record("r3", "g3", "b"),
            record("r4", "g4", "b"),
        ];
        let ratios = SplitRatios::default();
        let recorded = split_and_record(&records, 99, &ratios).expect("valid ratios");

        let reproduced = recorded
            .record()
            .verify_against(&records)
            .expect("同じ入力での再分割は一致するはず");
        assert_eq!(reproduced, *recorded.result());
    }

    /// REQ-17・TASK-17.1-2: レコードを 1 件足す（group を追加する）と、
    /// 割付が変わって `SplitMismatch` になりうる。
    #[test]
    fn req17_task17_1_2_verify_against_detects_added_record() {
        let records = vec![
            record("r1", "g1", "a"),
            record("r2", "g2", "a"),
            record("r3", "g3", "b"),
        ];
        let ratios = SplitRatios::default();
        let recorded = split_and_record(&records, 5, &ratios).expect("valid ratios");

        let mut changed_records = records;
        changed_records.push(record("r4", "g4", "b"));

        let err = recorded
            .record()
            .verify_against(&changed_records)
            .expect_err("group を追加すると割付が変わるはず");
        assert!(matches!(
            err,
            SplitRecordError::SplitMismatch { .. } | SplitRecordError::AllocationMismatch
        ));
    }

    /// REQ-17・TASK-17.1-2（改ざん検出）: JSON 上で test の `record_ids` の
    /// 1 件を、整合の取れた形（`record_count`・ソート順を保ったまま）で
    /// 書き換えると、`verify_hashes` が `HashMismatch { mismatched: [Test] }`
    /// になる。
    #[test]
    fn req17_task17_1_2_verify_hashes_detects_tampered_record_id() {
        let records = vec![
            record("r1", "g1", "a"),
            record("r2", "g2", "a"),
            record("r3", "g3", "b"),
            record("r4", "g4", "b"),
        ];
        let recorded =
            split_and_record(&records, 7, &SplitRatios::default()).expect("valid ratios");
        let json = recorded.record().to_json().expect("直列化に失敗しないはず");

        // test split の record_ids の末尾（辞書順で最大）の ID を、まだ使われて
        // いない・かつソート順を保ったままの ID（"zzz-tampered" は他のどの
        // レコード ID よりも辞書順で大きい）へ書き換える（record_count は保つ）。
        let test_ids = recorded.record().digest(Split::Test).record_ids().to_vec();
        let target = test_ids.last().cloned().unwrap_or_default();
        let tampered_json = json.replacen(&format!("\"{target}\""), "\"zzz-tampered\"", 1);
        assert_ne!(
            json, tampered_json,
            "置換が発生しなかった（テストの前提が崩れている）"
        );

        let tampered = SplitRecord::from_json_str(&tampered_json).expect("構造としては valid");
        let err = tampered
            .verify_hashes()
            .expect_err("record_ids を書き換えたのでハッシュが一致しないはず");
        match err {
            SplitRecordError::HashMismatch { mismatched } => {
                assert!(mismatched.contains(&Split::Test) || !mismatched.is_empty());
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    /// REQ-17・TASK-17.1-2（形式の異常）: 未知のフィールドは拒否される。
    #[test]
    fn req17_task17_1_2_rejects_unknown_field() {
        let records = vec![record("r1", "g1", "a")];
        let recorded =
            split_and_record(&records, 1, &SplitRatios::default()).expect("valid ratios");
        let json = recorded.record().to_json().expect("直列化に失敗しないはず");
        let mut value: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        if let Some(obj) = value.as_object_mut() {
            obj.insert("unexpected_field".to_string(), serde_json::json!(true));
        }
        let tampered = serde_json::to_string(&value).expect("直列化に失敗しないはず");

        let err =
            SplitRecord::from_json_str(&tampered).expect_err("未知フィールドは拒否されるはず");
        assert!(matches!(err, SplitRecordError::Json));
    }

    /// レビュー指摘（#210 codex-review P2 / Cursor Bugbot Low）の回帰テスト:
    /// `per_label`（`LabelAllocation`）の要素に混入した未知のフィールドも、
    /// 他の永続化用 DTO と同じく拒否される（`LabelAllocation` への
    /// `deny_unknown_fields` 追加の受け入れ条件）。
    #[test]
    fn req17_task17_1_2_rejects_unknown_field_in_per_label_entry() {
        let records = vec![record("r1", "g1", "a")];
        let recorded =
            split_and_record(&records, 1, &SplitRatios::default()).expect("valid ratios");
        let json = recorded.record().to_json().expect("直列化に失敗しないはず");
        let mut value: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        let per_label = value
            .get_mut("rule")
            .and_then(|r| r.get_mut("per_label"))
            .and_then(serde_json::Value::as_array_mut)
            .expect("per_label が存在するはず");
        let first = per_label
            .first_mut()
            .and_then(serde_json::Value::as_object_mut)
            .expect("per_label は少なくとも 1 件のはず");
        first.insert("unexpected_field".to_string(), serde_json::json!(true));
        let tampered = serde_json::to_string(&value).expect("直列化に失敗しないはず");

        let err = SplitRecord::from_json_str(&tampered)
            .expect_err("per_label 要素の未知フィールドは拒否されるはず");
        assert!(matches!(err, SplitRecordError::Json));
    }

    /// REQ-17・TASK-17.1-2（形式の異常）: `schema_version` の不一致は拒否される。
    #[test]
    fn req17_task17_1_2_rejects_unsupported_schema_version() {
        let records = vec![record("r1", "g1", "a")];
        let recorded =
            split_and_record(&records, 1, &SplitRatios::default()).expect("valid ratios");
        let json = recorded.record().to_json().expect("直列化に失敗しないはず");
        let mut value: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        if let Some(obj) = value.as_object_mut() {
            obj.insert("schema_version".to_string(), serde_json::json!(2));
        }
        let tampered = serde_json::to_string(&value).expect("直列化に失敗しないはず");

        let err = SplitRecord::from_json_str(&tampered).expect_err("未知の版は拒否されるはず");
        assert!(matches!(err, SplitRecordError::UnsupportedSchemaVersion(2)));
    }

    /// REQ-17・TASK-17.1-2（形式の異常）: 大文字・63 桁の sha256 は拒否される。
    #[test]
    fn req17_task17_1_2_rejects_malformed_sha256() {
        let records = vec![record("r1", "g1", "a")];
        let recorded =
            split_and_record(&records, 1, &SplitRatios::default()).expect("valid ratios");
        let json = recorded.record().to_json().expect("直列化に失敗しないはず");

        // 大文字混入。
        let uppercase = replace_first_sha256_with(&json, &"A".repeat(64));
        let err = SplitRecord::from_json_str(&uppercase).expect_err("大文字は拒否されるはず");
        assert!(matches!(err, SplitRecordError::InvalidRecord { .. }));

        // 63 桁（1 桁欠落）。
        let short = replace_first_sha256_with(&json, &"a".repeat(63));
        let err = SplitRecord::from_json_str(&short).expect_err("63 桁は拒否されるはず");
        assert!(matches!(err, SplitRecordError::InvalidRecord { .. }));
    }

    /// JSON 文字列中の最初の 64 文字 16 進らしき文字列を別の値に置換する
    /// テスト専用ヘルパー（`sha256` フィールドの値を狙い撃ちで壊す）。
    fn replace_first_sha256_with(json: &str, replacement: &str) -> String {
        let mut value: serde_json::Value = serde_json::from_str(json).expect("valid JSON");
        if let Some(train) = value
            .get_mut("splits")
            .and_then(|s| s.get_mut("train"))
            .and_then(|t| t.as_object_mut())
        {
            train.insert(
                "sha256".to_string(),
                serde_json::Value::String(replacement.to_string()),
            );
        }
        serde_json::to_string(&value).expect("直列化に失敗しないはず")
    }

    /// REQ-17・TASK-17.1-2（形式の異常）: `record_count` と `record_ids.len()`
    /// の不一致は拒否される。
    #[test]
    fn req17_task17_1_2_rejects_record_count_mismatch() {
        let records = vec![record("r1", "g1", "a")];
        let recorded =
            split_and_record(&records, 1, &SplitRatios::default()).expect("valid ratios");
        let json = recorded.record().to_json().expect("直列化に失敗しないはず");
        let mut value: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        if let Some(train) = value
            .get_mut("splits")
            .and_then(|s| s.get_mut("train"))
            .and_then(|t| t.as_object_mut())
        {
            train.insert("record_count".to_string(), serde_json::json!(999));
        }
        let tampered = serde_json::to_string(&value).expect("直列化に失敗しないはず");

        let err = SplitRecord::from_json_str(&tampered).expect_err("件数不一致は拒否されるはず");
        assert!(matches!(err, SplitRecordError::InvalidRecord { .. }));
    }

    /// REQ-17・TASK-17.1-2（形式の異常）: 未ソート・重複のある `record_ids`
    /// は拒否される。
    #[test]
    fn req17_task17_1_2_rejects_unsorted_or_duplicate_record_ids() {
        let records = vec![record("r1", "g1", "a"), record("r2", "g2", "a")];
        let recorded =
            split_and_record(&records, 1, &SplitRatios::default()).expect("valid ratios");
        let json = recorded.record().to_json().expect("直列化に失敗しないはず");

        // train の record_ids を未ソートに書き換える。
        let mut value: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        if let Some(train) = value
            .get_mut("splits")
            .and_then(|s| s.get_mut("train"))
            .and_then(|t| t.as_object_mut())
        {
            train.insert("record_ids".to_string(), serde_json::json!(["zzz", "aaa"]));
            train.insert("record_count".to_string(), serde_json::json!(2));
        }
        let unsorted = serde_json::to_string(&value).expect("直列化に失敗しないはず");
        let err = SplitRecord::from_json_str(&unsorted).expect_err("未ソートは拒否されるはず");
        assert!(matches!(err, SplitRecordError::InvalidRecord { .. }));
    }

    /// REQ-17・TASK-17.1-2（形式の異常）: 分割をまたいだ ID の重複は拒否される。
    #[test]
    fn req17_task17_1_2_rejects_record_id_reused_across_splits() {
        let records = vec![record("r1", "g1", "a"), record("r2", "g2", "a")];
        let recorded =
            split_and_record(&records, 1, &SplitRatios::default()).expect("valid ratios");
        let json = recorded.record().to_json().expect("直列化に失敗しないはず");

        let mut value: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        let train_ids = value["splits"]["train"]["record_ids"].clone();
        if let Some(validation) = value
            .get_mut("splits")
            .and_then(|s| s.get_mut("validation"))
            .and_then(|v| v.as_object_mut())
        {
            validation.insert("record_ids".to_string(), train_ids);
            // 件数は元のまま書き換えず、意図的に record_count との不一致では
            // なく「別 split と ID が重複する」検証だけを踏ませたいので、
            // record_count はそのままにしておく（重複検出が先に働く想定）。
        }
        let tampered = serde_json::to_string(&value).expect("直列化に失敗しないはず");

        let err = SplitRecord::from_json_str(&tampered)
            .expect_err("split をまたぐ ID の重複は拒否されるはず");
        assert!(matches!(err, SplitRecordError::InvalidRecord { .. }));
    }

    /// REQ-17・TASK-17.1-2（形式の異常）: 比率文字列が不正（NaN・非数値・
    /// 合計が 1.2）な場合は拒否される。
    #[test]
    fn req17_task17_1_2_rejects_invalid_ratio_strings() {
        let records = vec![record("r1", "g1", "a")];
        let recorded =
            split_and_record(&records, 1, &SplitRatios::default()).expect("valid ratios");
        let json = recorded.record().to_json().expect("直列化に失敗しないはず");

        for bad_train in ["NaN", "abc"] {
            let mut value: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
            if let Some(ratios) = value
                .get_mut("rule")
                .and_then(|r| r.get_mut("ratios"))
                .and_then(|r| r.as_object_mut())
            {
                ratios.insert(
                    "train".to_string(),
                    serde_json::Value::String(bad_train.to_string()),
                );
            }
            let tampered = serde_json::to_string(&value).expect("直列化に失敗しないはず");
            let err = SplitRecord::from_json_str(&tampered)
                .expect_err(&format!("train={bad_train} は拒否されるはず"));
            assert!(matches!(err, SplitRecordError::InvalidRecord { .. }));
        }

        // 合計が 1.2 になる比率。
        let mut value: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        if let Some(ratios) = value
            .get_mut("rule")
            .and_then(|r| r.get_mut("ratios"))
            .and_then(|r| r.as_object_mut())
        {
            ratios.insert("train".to_string(), serde_json::json!("0.8"));
            ratios.insert("validation".to_string(), serde_json::json!("0.3"));
            ratios.insert("test".to_string(), serde_json::json!("0.1"));
        }
        let tampered = serde_json::to_string(&value).expect("直列化に失敗しないはず");
        let err = SplitRecord::from_json_str(&tampered).expect_err("合計 1.2 は拒否されるはず");
        assert!(matches!(err, SplitRecordError::InvalidRecord { .. }));
    }

    /// REQ-17・TASK-17.1-2（形式の異常）: 構文として不正な JSON は
    /// `Json` エラーになる。
    #[test]
    fn req17_task17_1_2_rejects_malformed_json_syntax() {
        let err = SplitRecord::from_json_str("{not valid json").expect_err("構文エラーのはず");
        assert!(matches!(err, SplitRecordError::Json));
    }

    /// REQ-17・TASK-17.1-2（比率の往復）: 丸めで問題になりやすい値
    /// （0.7 / 0.2 / 0.1、1.0/3.0 系）がビット単位で往復一致する。
    #[test]
    fn req17_task17_1_2_ratio_roundtrip_bit_exact() {
        let cases = [
            SplitRatios {
                train: 0.7,
                validation: 0.2,
                test: 0.1,
            },
            SplitRatios {
                train: 1.0 / 3.0,
                validation: 1.0 / 3.0,
                test: 1.0 - 2.0 / 3.0,
            },
        ];
        let records = vec![record("r1", "g1", "a")];

        for ratios in cases {
            let recorded = split_and_record(&records, 1, &ratios).expect("valid ratios");
            let json = recorded.record().to_json().expect("直列化に失敗しないはず");
            let restored = SplitRecord::from_json_str(&json).expect("復元に失敗しないはず");
            let restored_ratios = restored.rule().ratios();

            assert_eq!(ratios.train.to_bits(), restored_ratios.train.to_bits());
            assert_eq!(
                ratios.validation.to_bits(),
                restored_ratios.validation.to_bits()
            );
            assert_eq!(ratios.test.to_bits(), restored_ratios.test.to_bits());
        }
    }

    /// REQ-17・TASK-17.1-2（エラー表示）: `Display` にレコード ID が
    /// 含まれない（security.md「秘密情報の混入防止」）。
    #[test]
    fn req17_task17_1_2_error_display_does_not_leak_record_ids() {
        let records = vec![
            record("secret-id", "g1", "a"),
            record("r2", "g2", "a"),
            record("r3", "g3", "b"),
            record("r4", "g4", "b"),
        ];
        let ratios = SplitRatios::default();
        let recorded = split_and_record(&records, 5, &ratios).expect("valid ratios");

        let mut changed_records = records;
        changed_records.push(record("r5", "g5", "b"));

        let err = recorded
            .record()
            .verify_against(&changed_records)
            .expect_err("group を追加すると不一致になるはず");
        let message = err.to_string();
        assert!(
            !message.contains("secret-id"),
            "エラーメッセージにレコード ID が含まれてはならない: {message}"
        );
    }

    /// レビュー指摘（High。#45）の回帰テスト: `split_and_record` に重複
    /// レコード ID を渡すと `SplitRecordError::Split(SplitError::
    /// DuplicateRecordId)` 経路に入るが、その `Display` にも実際の
    /// レコード ID（"secret-id"）が含まれてはならない
    /// （security.md「秘密情報の混入防止」）。
    #[test]
    fn req17_task17_1_2_duplicate_record_id_error_display_does_not_leak_record_id() {
        let records = vec![
            record("secret-id", "g1", "a"),
            record("secret-id", "g2", "a"),
        ];

        let err = split_and_record(&records, 1, &SplitRatios::default())
            .expect_err("重複 ID は Split エラーになるはず");
        assert!(matches!(err, SplitRecordError::Split(_)));

        let message = err.to_string();
        assert!(
            !message.contains("secret-id"),
            "エラーメッセージにレコード ID が含まれてはならない: {message}"
        );
    }

    /// レビュー指摘（Low。#45）の回帰テスト: JSON を改ざんして
    /// `record_count` が 0 なのに `group_count` を非 0 にした場合、
    /// `record_ids` 単体からは検出できない矛盾を `from_json_str` が
    /// 拒否する（`InvalidRecord` で fail-closed）。
    #[test]
    fn req17_task17_1_2_from_json_rejects_group_count_inconsistent_with_zero_records() {
        let records = vec![record("r1", "g1", "a")];
        let recorded = split_and_record(&records, 1, &SplitRatios::default())
            .expect("1 件でも既定比率で分割できるはず");
        let json = recorded.record().to_json().expect("直列化に失敗しないはず");

        // 1 件のレコードは 1 つの split にしか割り付けられないため、他の
        // 2 split は record_count: 0 のはず（この改ざんの前提。
        // どの split が空かはレコード割付の詳細に依存するため、
        // 空だった split の group_count を 1 件に改ざんする）。
        let mut value: serde_json::Value =
            serde_json::from_str(&json).expect("直列化した JSON は解析できるはず");
        let splits = value
            .get_mut("splits")
            .and_then(serde_json::Value::as_object_mut)
            .expect("splits が存在するはず");
        let empty_split = ["train", "validation", "test"]
            .into_iter()
            .find(|name| {
                splits
                    .get(*name)
                    .and_then(|s| s.get("record_count"))
                    .and_then(serde_json::Value::as_u64)
                    == Some(0)
            })
            .expect("1 件のレコードでは必ずどこかの split が空になるはず");
        splits
            .get_mut(empty_split)
            .expect("直前に存在を確認した split")["group_count"] = serde_json::json!(1);
        let tampered = serde_json::to_string(&value).expect("再直列化に失敗しないはず");

        let err = SplitRecord::from_json_str(&tampered)
            .expect_err("record_count と矛盾する group_count は拒否されるはず");
        assert!(matches!(
            err,
            SplitRecordError::InvalidRecord {
                reason: "group_count is inconsistent with record_count"
            }
        ));
    }

    /// レビュー指摘（Low。#45）の回帰テスト: `verify_against` は
    /// `record_ids`・`sha256` が一致していても `group_count` が
    /// 再分割結果と食い違えば `SplitMismatch` を返す。
    #[test]
    fn req17_task17_1_2_verify_against_detects_tampered_group_count() {
        let records = vec![
            record("r1", "g1", "a"),
            record("r2", "g1", "a"),
            record("r3", "g2", "a"),
            record("r4", "g3", "a"),
            record("r5", "g4", "b"),
            record("r6", "g5", "b"),
            record("r7", "g6", "b"),
            record("r8", "g7", "b"),
        ];
        let recorded = split_and_record(&records, 7, &SplitRatios::default())
            .expect("既定比率で分割できるはず");

        let json = recorded.record().to_json().expect("直列化に失敗しないはず");
        let mut value: serde_json::Value =
            serde_json::from_str(&json).expect("直列化した JSON は解析できるはず");
        let test_split = value
            .get_mut("splits")
            .and_then(|v| v.get_mut("test"))
            .expect("splits.test が存在するはず");
        let original_group_count = test_split
            .get("group_count")
            .and_then(serde_json::Value::as_u64)
            .expect("group_count は数値のはず");
        let record_count = test_split
            .get("record_count")
            .and_then(serde_json::Value::as_u64)
            .expect("record_count は数値のはず");
        assert!(
            record_count >= 2,
            "group_count を record_ids から検出不能な形で改ざんするには test に 2 件以上必要"
        );
        // record_ids・record_count・sha256 は一切変えず、group_count のみを
        // 1 減らす（record_ids だけを見る検証では検出できない改ざん）。
        test_split["group_count"] = serde_json::json!(original_group_count.saturating_sub(1));
        let tampered = serde_json::to_string(&value).expect("再直列化に失敗しないはず");

        // レビュー指摘（#210 codex/review P1）の回帰テスト: `group_count`
        // 単体の改ざんは、`per_label` の内訳合計との照合（`from_json_str`
        // 内）で復元時に検出されるようになった（以前は `record_ids` だけを
        // 見る検証をすり抜け、`verify_against` の再分割照合まで検出されずに
        // 素通りしていた）。
        let err = SplitRecord::from_json_str(&tampered)
            .expect_err("per_label 合計との不一致は復元時に検出されるはず");
        assert!(matches!(
            err,
            SplitRecordError::InvalidRecord {
                reason: "per_label allocation totals do not match each split's group_count"
            }
        ));
    }

    /// レビュー指摘（#210 Cursor Bugbot Low）の回帰テスト: `per_label` の
    /// 1 エントリで `train + validation + test != n_groups` に改ざんすると、
    /// `record_ids` 単体からは検出できない矛盾を `from_json_str` が拒否する
    /// （`InvalidRecord` で fail-closed。`verify_against` 実行を待たない）。
    #[test]
    fn req17_task17_1_2_from_json_rejects_per_label_sum_mismatch() {
        let records = vec![
            record("r1", "g1", "a"),
            record("r2", "g2", "a"),
            record("r3", "g3", "b"),
            record("r4", "g4", "b"),
        ];
        let recorded = split_and_record(&records, 7, &SplitRatios::default())
            .expect("既定比率で分割できるはず");
        let json = recorded.record().to_json().expect("直列化に失敗しないはず");

        let mut value: serde_json::Value =
            serde_json::from_str(&json).expect("直列化した JSON は解析できるはず");
        let per_label = value
            .get_mut("rule")
            .and_then(|r| r.get_mut("per_label"))
            .and_then(serde_json::Value::as_array_mut)
            .expect("per_label が存在するはず");
        let first = per_label
            .first_mut()
            .and_then(serde_json::Value::as_object_mut)
            .expect("per_label は少なくとも 1 件のはず");
        // `n_groups` を保ったまま `train` だけを 1 件水増しし、
        // `train + validation + test` が `n_groups` と食い違う状態にする
        // （`record_ids`・`group_count` は一切変えないため、他の検証では
        // 検出できない改ざん）。
        let original_train = first
            .get("train")
            .and_then(serde_json::Value::as_u64)
            .expect("train は数値のはず");
        first.insert(
            "train".to_string(),
            serde_json::json!(original_train.saturating_add(1)),
        );
        let tampered = serde_json::to_string(&value).expect("再直列化に失敗しないはず");

        let err = SplitRecord::from_json_str(&tampered)
            .expect_err("per_label の内訳合計が n_groups と食い違う場合は拒否されるはず");
        assert!(matches!(
            err,
            SplitRecordError::InvalidRecord {
                reason: "per_label allocation train + validation + test does not equal n_groups"
            }
        ));
    }

    /// レビュー指摘（#210 codex/review P1）の回帰テスト: `per_label` に
    /// 同じラベルが 2 行存在する（1 行を複製する）改ざんを `from_json_str`
    /// が拒否する。各行内の合計検証・split 別合計照合のどちらもすり抜け
    /// うる改ざん（複製した行の値をそのまま複製すると各行の内訳合計は
    /// 保たれ、複製元のラベルを除いた分の合計だけを見れば一致してしまう
    /// 場合がある）のため、ラベルの一意性を専用に検証する。
    #[test]
    fn req17_task17_1_2_from_json_rejects_duplicate_label_in_per_label() {
        let records = vec![
            record("r1", "g1", "a"),
            record("r2", "g2", "a"),
            record("r3", "g3", "b"),
            record("r4", "g4", "b"),
        ];
        let recorded = split_and_record(&records, 7, &SplitRatios::default())
            .expect("既定比率で分割できるはず");
        let json = recorded.record().to_json().expect("直列化に失敗しないはず");

        let mut value: serde_json::Value =
            serde_json::from_str(&json).expect("直列化した JSON は解析できるはず");
        let per_label = value
            .get_mut("rule")
            .and_then(|r| r.get_mut("per_label"))
            .and_then(serde_json::Value::as_array_mut)
            .expect("per_label が存在するはず");
        let first = per_label
            .first()
            .cloned()
            .expect("per_label は少なくとも 1 件のはず");
        // 先頭のラベル行をそのまま複製する（合計は 2 倍になるため、
        // split 別合計照合でも本来は検出されるが、ラベルの一意性検証を
        // 専用に持つことで改ざんの種類を問わず fail-closed にする）。
        per_label.push(first);
        let tampered = serde_json::to_string(&value).expect("再直列化に失敗しないはず");

        let err = SplitRecord::from_json_str(&tampered)
            .expect_err("per_label 内のラベル重複は拒否されるはず");
        assert!(matches!(
            err,
            SplitRecordError::InvalidRecord {
                reason: "per_label contains a duplicate label"
            }
        ));
    }

    /// レビュー指摘（#210 codex/review P1）の回帰テスト: `per_label` から
    /// 1 行を丸ごと削除する（ラベルの欠落）改ざんを `from_json_str` が
    /// 拒否する。削除された行の `record_ids`・`group_count` はそのまま
    /// 残るため、split 別の `per_label` 合計が `group_count` を下回ることで
    /// 検出する（各行内の合計検証だけでは、削除された行自体が存在しない
    /// ため検出できない改ざん）。
    #[test]
    fn req17_task17_1_2_from_json_rejects_missing_label_in_per_label() {
        let records = vec![
            record("r1", "g1", "a"),
            record("r2", "g2", "a"),
            record("r3", "g3", "b"),
            record("r4", "g4", "b"),
        ];
        let recorded = split_and_record(&records, 7, &SplitRatios::default())
            .expect("既定比率で分割できるはず");
        let json = recorded.record().to_json().expect("直列化に失敗しないはず");

        let mut value: serde_json::Value =
            serde_json::from_str(&json).expect("直列化した JSON は解析できるはず");
        let per_label = value
            .get_mut("rule")
            .and_then(|r| r.get_mut("per_label"))
            .and_then(serde_json::Value::as_array_mut)
            .expect("per_label が存在するはず");
        assert!(
            per_label.len() >= 2,
            "ラベル欠落を再現するには per_label に 2 件以上必要"
        );
        // 先頭のラベル行を丸ごと削除する。対応する split の record_ids・
        // group_count は変えないため、残りの行だけの合計は group_count を
        // 下回る。
        per_label.remove(0);
        let tampered = serde_json::to_string(&value).expect("再直列化に失敗しないはず");

        let err = SplitRecord::from_json_str(&tampered)
            .expect_err("per_label のラベル欠落は拒否されるはず");
        assert!(matches!(
            err,
            SplitRecordError::InvalidRecord {
                reason: "per_label allocation totals do not match each split's group_count"
            }
        ));
    }

    /// レビュー指摘（#210 Cursor Bugbot Medium）の回帰テスト:
    /// `RecordedSplit` のフィールドが非公開のため、対応しない
    /// `SplitResult` と `SplitRecord` の組み合わせを外部から組み立てられない
    /// （型で保証する契約。アクセサ経由でのみ参照できることを確認する）。
    #[test]
    fn req17_task17_1_2_recorded_split_fields_are_read_only_via_accessors() {
        let records = vec![record("r1", "g1", "a"), record("r2", "g2", "a")];
        let recorded = split_and_record(&records, 1, &SplitRatios::default())
            .expect("既定比率で分割できるはず");

        // アクセサ経由の参照が split_by_group 由来の同一結果を指す
        // （`result()`・`record()` はいずれも読み取り専用の `&` 参照であり、
        // 呼び出し側が `RecordedSplit { result: ..., record: ... }` の形で
        // 不整合な組を新規に作れないことはコンパイル時に保証される）。
        assert_eq!(
            recorded.result().rule_id,
            recorded.record().rule().rule_id()
        );
    }
}
