//! データ検査（[`crate::inspect::inspect_records`]）と来歴の取り込み
//! （[`crate::provenance::ingest::parse_provenance_json`]）の接続点
//! （REQ-40・TASK-40.1-2・issue #75）。
//!
//! 呼び出し文脈: 将来は CLI の `inspect` 工程（TASK-33.x・パス未確定）
//! から、ガード層を通過済みの JSONL 本文と（任意の）来歴 JSON 文字列を
//! 受け取って呼ばれる想定。`crates/cli` は本 crate に未依存（TASK-33.1
//! 未着手）のため、本モジュールは接続点となるライブラリ関数を提供する
//! に留まる。
//!
//! [`crate::inspect::InspectOutcome`]・[`crate::inspect::inspect_records`]
//! のシグネチャ・構造は変更しない（既存テスト
//! `crates/data/tests/inspect_contract.rs`・`inspect_report*.rs` が
//! 依存するため）。本モジュールは別関数としてラップして接続する。

use std::collections::BTreeSet;

use crate::inspect::{self, EmptyLabelSet, InspectOutcome};
use crate::provenance::ingest::{ProvenanceIngestError, parse_provenance_json};
use crate::provenance::{DisallowedSourceError, ProvenanceRecord, check_default_training_source};

/// [`ingest_records`] の結果。
///
/// `inspect` フィールドの中身は [`inspect::inspect_records`] を単体で
/// 呼んだ場合の結果と完全に一致する（本モジュールは検査ロジック自体を
/// 変更しない）。`provenance` は取り込み時に渡された来歴 JSON
/// （[`ingest_records`] の `provenance_json` 引数）を検証した結果で、
/// `provenance_json` が `None`（利用者自身が用意したデータ等、来歴が
/// 無いデータ）の場合は `None` になる。来歴が無いデータは許可する
/// （TASK-40.2 で判断済み。来歴が **ある** 場合のみ、生成元が Jev 出力なら
/// 拒否する。主経路は利用者自身が用意したデータ〔来歴なし〕のため）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IngestOutcome {
    pub inspect: InspectOutcome,
    pub provenance: Option<ProvenanceRecord>,
}

/// [`ingest_records`] のエラー。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum IngestError {
    /// 有効なラベル ID の集合が空だった（[`inspect::inspect_records`] の
    /// [`EmptyLabelSet`] をそのまま伝播する）。
    EmptyLabelSet(EmptyLabelSet),
    /// 来歴 JSON の検証に失敗した。
    Provenance(ProvenanceIngestError),
    /// 来歴は解析できたが、学習データの既定生成元として許可されない
    /// 生成元だった（Jev 出力。REQ-40 受け入れ基準 2・TASK-40.2・
    /// issue #76）。上書き手段は無く、常に拒否する。
    DisallowedSource(DisallowedSourceError),
}

/// データ検査と来歴の取り込みを 1 回の呼び出しで行う（fail-closed）。
///
/// # 挙動
///
/// 1. `provenance_json` が `Some` の場合、先にそれを解析する。解析に
///    失敗したら **データ検査を行わずに** [`IngestError::Provenance`] を
///    返す（来歴不正を黙って無視して検査だけ進めることはしない）。
/// 2. 来歴が解析できた場合、[`check_default_training_source`] で生成元の
///    採用可否を判定する。Jev 出力なら **データ検査を行わずに**
///    [`IngestError::DisallowedSource`] を返す（役割〔学習／評価〕を問わず
///    無条件に拒否する。本関数は役割を知らないため）。
/// 3. [`inspect::inspect_records`] を呼び、[`EmptyLabelSet`] はそのまま
///    [`IngestError::EmptyLabelSet`] として返す。
/// 4. すべて成功したら [`IngestOutcome`] を返す。
///
/// ファイル読み込み・サイズ上限（REQ-39）は呼び出し側（ガード層）の責務
/// （[`crate`] クレート doc「前提条件」節と同じ方針）。CLI 接続時の終了
/// コード写像（`invalid_input`=64 か `out_of_scope`=11 か）は TASK-33.x の
/// 範囲で未実装。
pub fn ingest_records(
    content: &str,
    valid_label_ids: &BTreeSet<String>,
    provenance_json: Option<&str>,
) -> Result<IngestOutcome, IngestError> {
    let provenance = match provenance_json {
        None => None,
        Some(json) => {
            let record = parse_provenance_json(json).map_err(IngestError::Provenance)?;
            check_default_training_source(&record).map_err(IngestError::DisallowedSource)?;
            Some(record)
        }
    };

    let outcome =
        inspect::inspect_records(content, valid_label_ids).map_err(IngestError::EmptyLabelSet)?;

    Ok(IngestOutcome {
        inspect: outcome,
        provenance,
    })
}
