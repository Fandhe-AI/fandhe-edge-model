//! 版管理台帳（REQ-39・TASK-39.3-1・#167。PoC-20 ケース 7 の `manifest_add_version` 相当）。
//!
//! モデル・データ・実験の版ごとに、版 ID・sha256 ハッシュ・作成時刻を記録して取り出す。
//! ガード層の「完全性と版」（親 TASK-39.3・#166）のうち、本モジュールは記録と取得だけを担う。
//!
//! # 責務境界
//!
//! - 前版へのロールバックと、復元後に再計算したハッシュと台帳記録との一致検証は
//!   TASK-39.3-2（#168）の範囲。#168 が `(種別, 版 ID)` での取得・記録順の列挙・
//!   ダイジェストの読み取りだけで書けるよう API を揃えている
//! - 台帳はメモリ上のみで、永続化（JSON への保存・読み戻し）は未実装。ガード層への
//!   `serde` 系の配置が dependency-policy で未承認のため（永続化時は台帳ファイルの改ざん検証も課題）
//! - [`VersionLedger::record_file`] は渡されたパスをそのまま開く。ルート外参照の拒否は
//!   呼び出し側が [`crate::path`] を通す責務（CLI への組み込みは TASK-39.4-2・#159）
//! - ファイルサイズ上限の値は TASK-39.5 が決め、本モジュールは渡された値を強制する
//! - 追記のみ。同じ `(種別, 版 ID)` の再記録は拒否し、記録済みハッシュの差し替えを防ぐ（完全性）
//!
//! 時刻は共通コア・データ契約の型を流用せず本モジュールの [`CreatedAt`] で持つ
//! （ガード層は共通コアのみに依存する方針のため。REQ-32）。

use fandhe_edge_core::exitcode::ExitCode;
use fandhe_edge_core::fs::{FsError, sha256_file_bounded};
use fandhe_edge_core::hash::Sha256Digest;
use std::fmt;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// 版 ID の最大バイト数。
pub const VERSION_ID_MAX_BYTES: usize = 64;

/// 作成時刻の上限（9999-12-31T23:59:59Z の UNIX 秒）。RFC 3339 の年を常に 4 桁に保つ。
pub const CREATED_AT_MAX_UNIX_SECONDS: u64 = 253_402_300_799;

/// 台帳の最大件数。無制限のアロケーションを作らないための仮置き値（証拠種別: 仮置き。
/// 実運用の値は TASK-39.5 の資源上限で見直す）。
pub const LEDGER_MAX_ENTRIES: usize = 10_000;

/// 版管理の対象種別（REQ-39）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum ArtifactKind {
    /// モデル。
    Model,
    /// データ。
    Data,
    /// 実験。
    Experiment,
}

impl ArtifactKind {
    /// 機械可読な種別名（英語）。
    pub const fn name(self) -> &'static str {
        match self {
            ArtifactKind::Model => "model",
            ArtifactKind::Data => "data",
            ArtifactKind::Experiment => "experiment",
        }
    }
}

/// 版 ID が不正な理由。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum VersionIdProblem {
    /// 空文字。
    Empty,
    /// [`VERSION_ID_MAX_BYTES`] 超（`len` は入力のバイト数）。
    TooLong {
        /// 入力のバイト数。
        len: usize,
    },
    /// 許可文字（ASCII の英数字・`.`・`_`・`-`）以外を含む。
    InvalidChar,
    /// `.` または `..` そのもの（経路成分になりうるため）。
    DotSegment,
}

/// 検証済みの版 ID。CLI 引数などの非信頼入力から [`VersionId::new`] で作る。
///
/// 許可文字を ASCII の `[A-Za-z0-9._-]` に限るため、将来ファイル名・JSON キー・ログへ流れても
/// 経路成分や制御文字を含まない。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VersionId(String);

impl VersionId {
    /// 版 ID を検証して作る。長さ → 文字種 → ドット成分の順に検査する。
    pub fn new(raw: &str) -> Result<Self, LedgerError> {
        let problem = if raw.is_empty() {
            Some(VersionIdProblem::Empty)
        } else if raw.len() > VERSION_ID_MAX_BYTES {
            Some(VersionIdProblem::TooLong { len: raw.len() })
        } else if !raw
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
        {
            Some(VersionIdProblem::InvalidChar)
        } else if raw == "." || raw == ".." {
            Some(VersionIdProblem::DotSegment)
        } else {
            None
        };
        match problem {
            Some(reason) => Err(LedgerError::InvalidVersionId { reason }),
            None => Ok(VersionId(raw.to_owned())),
        }
    }

    /// 版 ID の文字列表現。
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for VersionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// 作成時刻（UTC・UNIX 秒）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CreatedAt(u64);

impl CreatedAt {
    /// UNIX 秒から作る。[`CREATED_AT_MAX_UNIX_SECONDS`] 超は拒否する。
    pub fn from_unix_seconds(secs: u64) -> Result<Self, LedgerError> {
        if secs > CREATED_AT_MAX_UNIX_SECONDS {
            return Err(LedgerError::CreatedAtOutOfRange { unix_seconds: secs });
        }
        Ok(CreatedAt(secs))
    }

    /// OS の時計から作る。時計が UNIX epoch より前などの異常は panic させず
    /// [`LedgerError::ClockUnavailable`] を返す（通信なし。REQ-38）。
    pub fn now() -> Result<Self, LedgerError> {
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| LedgerError::ClockUnavailable)?
            .as_secs();
        Self::from_unix_seconds(secs)
    }

    /// UNIX 秒。
    pub const fn unix_seconds(self) -> u64 {
        self.0
    }

    /// `YYYY-MM-DDTHH:MM:SSZ` 形式（RFC 3339・UTC）。
    pub fn to_rfc3339_utc(self) -> String {
        let days = self.0 / 86_400;
        let rem = self.0 % 86_400;
        // Howard Hinnant の civil_from_days。入力は 9999 年までに制限済みで桁あふれしない。
        let z = days + 719_468;
        let era = z / 146_097;
        let doe = z % 146_097;
        let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let day = doy - (153 * mp + 2) / 5 + 1;
        let month = if mp < 10 { mp + 3 } else { mp - 9 };
        let year = yoe + era * 400 + u64::from(month <= 2);
        format!(
            "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
            rem / 3_600,
            rem % 3_600 / 60,
            rem % 60
        )
    }
}

/// 台帳の 1 件（種別・版 ID・sha256・作成時刻）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VersionEntry {
    kind: ArtifactKind,
    id: VersionId,
    sha256: Sha256Digest,
    created_at: CreatedAt,
}

impl VersionEntry {
    /// 対象種別。
    pub fn kind(&self) -> ArtifactKind {
        self.kind
    }
    /// 版 ID。
    pub fn id(&self) -> &VersionId {
        &self.id
    }
    /// 記録したハッシュ。
    pub fn sha256(&self) -> &Sha256Digest {
        &self.sha256
    }
    /// 作成時刻。
    pub fn created_at(&self) -> CreatedAt {
        self.created_at
    }
}

/// 版管理台帳（追記のみ・メモリ上）。
#[derive(Debug, Clone, Default)]
pub struct VersionLedger {
    entries: Vec<VersionEntry>,
}

impl VersionLedger {
    /// 空の台帳。
    pub fn new() -> Self {
        Self::default()
    }

    /// 重複と容量を検査する（記録・ファイル読み込みの前に呼ぶ）。
    fn check_room(&self, kind: ArtifactKind, id: &VersionId) -> Result<(), LedgerError> {
        if self.get(kind, id).is_some() {
            return Err(LedgerError::DuplicateVersion {
                kind,
                id: id.clone(),
            });
        }
        if self.entries.len() >= LEDGER_MAX_ENTRIES {
            return Err(LedgerError::CapacityExceeded {
                limit: LEDGER_MAX_ENTRIES,
            });
        }
        Ok(())
    }

    /// 版を記録する。作成時刻は呼び出し側が渡す（テストで決定的にするため）。
    /// 同じ `(種別, 版 ID)` は上書きせず [`LedgerError::DuplicateVersion`] とする。
    pub fn record(
        &mut self,
        kind: ArtifactKind,
        id: VersionId,
        sha256: Sha256Digest,
        created_at: CreatedAt,
    ) -> Result<&VersionEntry, LedgerError> {
        self.check_room(kind, &id)?;
        self.entries.push(VersionEntry {
            kind,
            id,
            sha256,
            created_at,
        });
        // 外部入力経路の規約に従い添字アクセスを使わない（直前の push により必ず存在する）。
        self.entries.last().ok_or(LedgerError::Internal)
    }

    /// ファイルの sha256 をサイズ上限付きのストリームで計算して記録する。
    /// 重複・容量を先に検査し、ハッシュ計算に失敗したときは台帳を変更しない。
    /// パスの閉じ込めは呼び出し側の責務（モジュール doc 参照）。
    pub fn record_file(
        &mut self,
        kind: ArtifactKind,
        id: VersionId,
        path: &Path,
        max_bytes: u64,
        created_at: CreatedAt,
    ) -> Result<&VersionEntry, LedgerError> {
        self.check_room(kind, &id)?;
        let digest = sha256_file_bounded(path, max_bytes).map_err(LedgerError::Io)?;
        self.record(kind, id, digest, created_at)
    }

    /// `(種別, 版 ID)` で 1 件を引く。
    pub fn get(&self, kind: ArtifactKind, id: &VersionId) -> Option<&VersionEntry> {
        self.entries.iter().find(|e| e.kind == kind && &e.id == id)
    }

    /// 全件を記録順で返す。
    pub fn entries(&self) -> &[VersionEntry] {
        &self.entries
    }

    /// 指定種別の版を記録順で返す。
    pub fn versions_of(&self, kind: ArtifactKind) -> impl Iterator<Item = &VersionEntry> {
        self.entries.iter().filter(move |e| e.kind == kind)
    }

    /// 記録済みの件数。
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// 台帳が空か。
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// 台帳操作の失敗理由。
#[derive(Debug)]
#[non_exhaustive]
pub enum LedgerError {
    /// 版 ID が不正。
    InvalidVersionId {
        /// 不正な理由。
        reason: VersionIdProblem,
    },
    /// 同じ `(種別, 版 ID)` が記録済み。
    DuplicateVersion {
        /// 対象種別。
        kind: ArtifactKind,
        /// 検証済みの版 ID。
        id: VersionId,
    },
    /// 台帳の件数上限に達した。
    CapacityExceeded {
        /// 件数上限。
        limit: usize,
    },
    /// 作成時刻が上限を超えた。
    CreatedAtOutOfRange {
        /// 入力された UNIX 秒。
        unix_seconds: u64,
    },
    /// OS の時計を読めない。
    ClockUnavailable,
    /// 内部不整合（push 直後に末尾要素が無い場合など。到達しない想定）。
    Internal,
    /// ファイルの読み込み・ハッシュ計算の失敗。
    Io(FsError),
}

impl LedgerError {
    /// 終了コード（REQ-21）。入力起因 → `InvalidInput`、上限超過 → `LimitExceeded`、
    /// 時計・その他の I/O 失敗 → `RuntimeError`（`format.rs` の写像と同じ）。
    pub fn exit_code(&self) -> ExitCode {
        match self {
            LedgerError::InvalidVersionId { .. }
            | LedgerError::DuplicateVersion { .. }
            | LedgerError::CreatedAtOutOfRange { .. } => ExitCode::InvalidInput,
            LedgerError::CapacityExceeded { .. } => ExitCode::LimitExceeded,
            LedgerError::ClockUnavailable | LedgerError::Internal => ExitCode::RuntimeError,
            LedgerError::Io(FsError::NotRegularFile { .. }) => ExitCode::InvalidInput,
            LedgerError::Io(FsError::TooLarge { .. }) => ExitCode::LimitExceeded,
            LedgerError::Io(FsError::Read { source, .. })
                if source.kind() == std::io::ErrorKind::NotFound =>
            {
                ExitCode::InvalidInput
            }
            LedgerError::Io(_) => ExitCode::RuntimeError,
        }
    }
}

impl fmt::Display for LedgerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LedgerError::InvalidVersionId { reason } => match reason {
                VersionIdProblem::Empty => f.write_str("version id is empty"),
                VersionIdProblem::TooLong { len } => write!(
                    f,
                    "version id is too long ({len} > {VERSION_ID_MAX_BYTES} bytes)"
                ),
                VersionIdProblem::InvalidChar => {
                    f.write_str("version id contains characters outside [A-Za-z0-9._-]")
                }
                VersionIdProblem::DotSegment => f.write_str("version id must not be . or .."),
            },
            LedgerError::DuplicateVersion { kind, id } => {
                write!(
                    f,
                    "version {id} of kind {} is already recorded",
                    kind.name()
                )
            }
            LedgerError::CapacityExceeded { limit } => {
                write!(f, "version ledger is full (limit {limit} entries)")
            }
            LedgerError::CreatedAtOutOfRange { unix_seconds } => write!(
                f,
                "created-at {unix_seconds} exceeds {CREATED_AT_MAX_UNIX_SECONDS}"
            ),
            LedgerError::ClockUnavailable => f.write_str("system clock is unavailable"),
            LedgerError::Internal => f.write_str("internal ledger inconsistency"),
            LedgerError::Io(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for LedgerError {}

#[cfg(test)]
mod tests {
    use super::*;

    const V1_HEX: &str = "1a1f4502024df8a68d12e64bb2364ad6308d04ed0a7d5e8300a676ec70867140";

    fn id(s: &str) -> VersionId {
        VersionId::new(s).expect("valid id")
    }
    fn at(s: u64) -> CreatedAt {
        CreatedAt::from_unix_seconds(s).expect("valid time")
    }

    /// REQ-39・TASK-39.3-1: 記録した版のハッシュと作成時刻を取り出せる。
    #[test]
    fn req39_record_then_get_returns_hash_and_time() {
        let mut l = VersionLedger::new();
        let d = Sha256Digest::of_bytes(b"model-v1");
        l.record(ArtifactKind::Model, id("v1"), d, at(1_790_000_000))
            .unwrap();
        let e = l.get(ArtifactKind::Model, &id("v1")).unwrap();
        assert_eq!(e.sha256().to_hex(), V1_HEX);
        assert_eq!(e.created_at().unix_seconds(), 1_790_000_000);
        assert_eq!(e.created_at().to_rfc3339_utc(), "2026-09-21T14:13:20Z");
        assert!(l.get(ArtifactKind::Data, &id("v1")).is_none());
    }

    /// REQ-39: 種別ごとに版 ID は一意で、記録順を保つ。
    #[test]
    fn req39_kinds_are_separate_and_order_is_kept() {
        let mut l = VersionLedger::new();
        let d = Sha256Digest::of_bytes(b"x");
        for (k, v) in [
            (ArtifactKind::Model, "v1"),
            (ArtifactKind::Data, "v1"),
            (ArtifactKind::Model, "v2"),
            (ArtifactKind::Experiment, "e1"),
        ] {
            l.record(k, id(v), d, at(1)).unwrap();
        }
        let ids: Vec<&str> = l
            .versions_of(ArtifactKind::Model)
            .map(|e| e.id().as_str())
            .collect();
        assert_eq!(ids, ["v1", "v2"]);
        let all: Vec<(&str, &str)> = l
            .entries()
            .iter()
            .map(|e| (e.kind().name(), e.id().as_str()))
            .collect();
        assert_eq!(
            all,
            [
                ("model", "v1"),
                ("data", "v1"),
                ("model", "v2"),
                ("experiment", "e1")
            ]
        );
    }

    /// REQ-39: 再記録は拒否され、元のハッシュが残る。
    #[test]
    fn req39_duplicate_is_rejected_and_original_kept() {
        let mut l = VersionLedger::new();
        l.record(
            ArtifactKind::Model,
            id("v1"),
            Sha256Digest::of_bytes(b"model-v1"),
            at(1_790_000_000),
        )
        .unwrap();
        let err = l
            .record(
                ArtifactKind::Model,
                id("v1"),
                Sha256Digest::of_bytes(b"tampered"),
                at(2),
            )
            .unwrap_err();
        assert!(matches!(err, LedgerError::DuplicateVersion { .. }));
        assert_eq!(err.exit_code(), ExitCode::InvalidInput);
        assert_eq!(l.len(), 1);
        let e = l.get(ArtifactKind::Model, &id("v1")).unwrap();
        assert_eq!(e.sha256().to_hex(), V1_HEX);
    }

    /// REQ-39: 版 ID の境界値と拒否理由。
    #[test]
    fn req39_version_id_validation() {
        assert!(VersionId::new(&"a".repeat(64)).is_ok());
        let reason = |s: &str| match VersionId::new(s).unwrap_err() {
            LedgerError::InvalidVersionId { reason } => reason,
            other => panic!("unexpected {other:?}"),
        };
        assert_eq!(
            reason(&"a".repeat(65)),
            VersionIdProblem::TooLong { len: 65 }
        );
        assert_eq!(reason(""), VersionIdProblem::Empty);
        for bad in ["../x", "a/b", "a b", "a\\b", "vé", "a\nb"] {
            assert_eq!(reason(bad), VersionIdProblem::InvalidChar, "{bad:?}");
        }
        assert_eq!(reason("."), VersionIdProblem::DotSegment);
        assert_eq!(reason(".."), VersionIdProblem::DotSegment);
        let e = VersionId::new("").unwrap_err();
        assert_eq!(e.exit_code(), ExitCode::InvalidInput);
        assert_eq!(e.to_string(), "version id is empty");
    }

    /// REQ-39: 作成時刻の境界値と RFC 3339 整形（うるう日・上限）。
    #[test]
    fn req39_created_at_bounds_and_format() {
        assert_eq!(at(0).to_rfc3339_utc(), "1970-01-01T00:00:00Z");
        assert_eq!(at(951_782_400).to_rfc3339_utc(), "2000-02-29T00:00:00Z");
        assert_eq!(at(253_402_300_799).to_rfc3339_utc(), "9999-12-31T23:59:59Z");
        let err = CreatedAt::from_unix_seconds(253_402_300_800).unwrap_err();
        assert!(matches!(
            err,
            LedgerError::CreatedAtOutOfRange {
                unix_seconds: 253_402_300_800
            }
        ));
        assert_eq!(err.exit_code(), ExitCode::InvalidInput);
        assert!(CreatedAt::now().unwrap().unix_seconds() >= 1_790_000_000);
    }

    /// REQ-39: 容量上限で `limit_exceeded`。
    #[test]
    fn req39_capacity_limit() {
        let mut l = VersionLedger::new();
        let d = Sha256Digest::of_bytes(b"x");
        for i in 0..LEDGER_MAX_ENTRIES {
            l.record(ArtifactKind::Data, id(&format!("v{i}")), d, at(1))
                .unwrap();
        }
        let err = l
            .record(ArtifactKind::Data, id("over"), d, at(1))
            .unwrap_err();
        assert!(matches!(
            err,
            LedgerError::CapacityExceeded { limit: 10_000 }
        ));
        assert_eq!(err.exit_code(), ExitCode::LimitExceeded);
        assert_eq!(l.len(), 10_000);
    }
}
