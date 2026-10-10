//! 版管理台帳ファイル `<project>/version_ledger.json` の直列化形式（REQ-39・TASK-39.3・TASK-39.6・#491）。
//!
//! CLI の `package` 工程だけが書き（公開する `package/` の直前に新規作成し、読み取り専用にする）、
//! `package --previous-project-dir` と `infer --version-ledger` が読む。形式は 1 行 JSON＋改行で、キー順は
//! 構造体の宣言順（`schema_version`・`entries`、各件は `kind`・`id`・`sha256`・`created_at_unix`）に固定し、
//! 未知キーは拒否する。
//!
//! 本モジュールは形（型・キー・16 進 64 桁の sha256）だけを検査する。版 ID の文字種・作成時刻の範囲・
//! `(kind, id)` の重複・件数の上限は、ガード層の `VersionLedger::from_file`（`fandhe-edge-guard`）が
//! 既存の検証（`VersionId::new`・`CreatedAt::from_unix_seconds`・`record`）で読み戻すときに行う
//! （検証規則を 1 箇所に保つため。ガード層に serde を入れない）。
//!
//! 台帳ファイル自体の改変（読み取り専用を外して全体を書き直すこと）の検出は範囲外（#491 の契約）。

use serde::{Deserialize, Serialize};

use crate::hash::Sha256Digest;

/// プロジェクト直下の台帳ファイル名。
pub const VERSION_LEDGER_FILE: &str = "version_ledger.json";

/// 台帳ファイルの読み込み上限（2 MiB。読み込み前に確認する。REQ-39）。
pub const MAX_VERSION_LEDGER_BYTES: u64 = 2 * 1024 * 1024;

/// 本形式の版。
pub const VERSION_LEDGER_SCHEMA_VERSION: u32 = 1;

/// 台帳ファイルの中身。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VersionLedgerRecord {
    /// 形式の版（[`VERSION_LEDGER_SCHEMA_VERSION`] のみ受け付ける）。
    pub schema_version: u32,
    /// 記録順の全件。
    pub entries: Vec<VersionLedgerEntryRecord>,
}

/// 台帳の 1 件。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VersionLedgerEntryRecord {
    /// 対象種別。
    pub kind: VersionLedgerKind,
    /// 版 ID（`v<n>`。文字種の検証はガード層）。
    pub id: String,
    /// 対象のバイト列の sha256（小文字 16 進 64 桁）。
    pub sha256: Sha256Digest,
    /// 作成時刻（UTC・UNIX 秒。範囲の検証はガード層）。
    pub created_at_unix: u64,
}

/// 版管理の対象種別（model ＝ 配布用 `package/artifact.json`、data ＝ `data/train.jsonl`、
/// experiment ＝ `selection_record.json`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VersionLedgerKind {
    /// モデル。
    Model,
    /// データ。
    Data,
    /// 実験。
    Experiment,
}

impl VersionLedgerRecord {
    /// 1 行 JSON＋改行へ直列化する。
    ///
    /// # Errors
    /// 直列化に失敗した場合。
    pub fn to_json_line(&self) -> Result<Vec<u8>, serde_json::Error> {
        let mut bytes = serde_json::to_vec(self)?;
        bytes.push(b'\n');
        Ok(bytes)
    }

    /// 読み込む（未知キー・型違い・欠落・不正な sha256 は拒否）。`schema_version` の照合は呼び出し側。
    /// 呼び出し側は [`MAX_VERSION_LEDGER_BYTES`] 以下であることを読み込み前に確認すること。
    ///
    /// # Errors
    /// JSON として不正、または形が合わない場合。
    pub fn from_json_slice(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-39・#491: キー順は宣言順で 1 行＋改行。読み戻すと同じ値。未知キー・不正な hex は拒否する。
    #[test]
    fn req39_issue491_ledger_file_has_fixed_key_order_and_rejects_unknown_keys() {
        let sha = Sha256Digest::of_bytes(b"a");
        let record = VersionLedgerRecord {
            schema_version: 1,
            entries: vec![VersionLedgerEntryRecord {
                kind: VersionLedgerKind::Experiment,
                id: "v1".to_string(),
                sha256: sha,
                created_at_unix: 7,
            }],
        };
        let line = record.to_json_line().unwrap();
        let expected = format!(
            "{{\"schema_version\":1,\"entries\":[{{\"kind\":\"experiment\",\"id\":\"v1\",\"sha256\":\"{}\",\"created_at_unix\":7}}]}}\n",
            sha.to_hex()
        );
        assert_eq!(String::from_utf8(line.clone()).unwrap(), expected);
        assert_eq!(VersionLedgerRecord::from_json_slice(&line).unwrap(), record);
        for bad in [
            expected.replace("]}", "],\"x\":1}"),
            expected.replace(",\"created_at_unix\"", ",\"x\":1,\"created_at_unix\""),
            expected.replace(&sha.to_hex(), &"A".repeat(64)),
            expected.replace("\"experiment\"", "\"weights\""),
        ] {
            assert!(VersionLedgerRecord::from_json_slice(bad.as_bytes()).is_err());
        }
    }
}
