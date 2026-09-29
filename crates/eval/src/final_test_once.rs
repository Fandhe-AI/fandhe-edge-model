//! 凍結した最終 test への 1 回限り適用の強制（REQ-27 境界値・TASK-27.3・issue #72）。
//!
//! 評価契約（`.claude/rules/evaluation-contract.md`「評価の独立性」）は「凍結した
//! 最終 test への適用は 1 回限り。最終 test の結果を見て候補・しきい値を選び直さない」
//! と定める。本モジュールはこれを、台帳ディレクトリ内のロックファイルを
//! `O_CREAT|O_EXCL`（`create_new`）で原子的に作る方式で機械的に強制する
//! （PoC-10 の `APPLIED.json` 事前登録・PoC-25 のモデル × 評価データ sha256 ごとの
//! ロックファイルの手順を移植。`docs/spec` は読まない）。
//!
//! # 呼び出し文脈
//!
//! CLI の `evaluate` 工程（issue #140 で配線予定・未配線）が、データ契約層の
//! 凍結記録から組み立てた [`FrozenEvalData`] を渡して [`apply_once`] を呼ぶ想定。
//!
//! # 唯一の公開経路（ロックと凍結記録の結び付け）
//!
//! ロックのキー（`FinalTestKey`）は非公開型で、コンストラクタも非公開。
//! 公開の入口は [`apply_once`] だけで、内部で
//! [`evaluate_with_eval_data_invariance`] を呼び、**実データから計算した sha256 が
//! 凍結記録と一致した後にのみ**、その照合済みダイジェストからキーを作る。
//! 呼び出し側が任意のダイジェストでキーを作って別ロックを取得し、2 回目の
//! 予測を通すことはできない（レビュー指摘 P0 への対応）。
//!
//! # 順序の不変条件
//!
//! 1. 凍結記録との照合（`evaluate_with_eval_data_invariance`）が通ってから
//! 2. ロックを取得し、永続化（ファイル・ディレクトリの `sync_all`）まで確認し
//! 3. 予測を当てる
//!
//! 永続化の確認に失敗したら予測は呼ばずエラーを返す（fail-closed）。ディレクトリの
//! fsync ができない環境（非 Unix 等）では永続性を保証できないため、
//! [`AcquireError::DurabilityUnsupported`] で拒否する（サポート外を成功扱いにしない）。
//!
//! 凍結ハッシュ不一致では `eval` クロージャが呼ばれないためロックは作られず、
//! 1 回の適用を消費しない。ロック取得後に予測が失敗しても、ロックは残す
//! （fail-closed。やり直しは拒否する）。
//!
//! # ロックの種類
//!
//! - 代表構成ロック `config-<hex>.lock`: 評価データ × 代表構成 ID。
//! - 重みロック `weights-<hex>.lock`: 評価データ × 重みの sha256。同じモデルを
//!   別の構成名で、あるいはしきい値・校正だけ変えて当て直す迂回を拒否する。
//!
//! 代表構成ロック作成後に重みロックが既存だった場合、作成済みの代表構成ロックは
//! 削除しない（最終 test への適用を試みた事実として消費扱い。ロールバックなし）。
//! ファイル名はハッシュ由来の固定長 hex のみで、呼び出し側の文字列は
//! パスに入らない。記録にはダイジェストと検証済み ID のみを書き、評価データ本文・
//! 入力・正解ラベルは書かない。
//!
//! # 範囲外（実装済みを装わない）
//!
//! - CLI `evaluate` への配線（issue #140）・終了コードへの写像（TASK-33.3）は未実装。
//! - 台帳ディレクトリの配置・ルート配下への閉じ込めはガード層（REQ-39）と呼び出し側の
//!   責務。本モジュールは symlink でない実ディレクトリであることのみ検査する。
//! - 既存予測の再採点（予測を当てない操作）はロック不要のため対象外。
//! - seed は独自フィールドにせず、呼び出し側が事前登録（PoC-10: 代表構成を seed ごとに
//!   1 回）に従って構成 ID へ畳み込む（例 `c1:seed0`）。

use crate::eval_data_invariance::{
    EvalDataInvarianceError, FrozenEvalData, evaluate_with_eval_data_invariance,
};
use crate::invariance::{ModelComponent, ModelPackageSnapshot};
use fandhe_edge_core::hash::Sha256Digest;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};

/// 代表構成 ID の最大バイト数（確保・検証の上限。REQ-39）。
pub const MAX_CONFIG_ID_BYTES: usize = 128;

const CONFIG_LOCK_DOMAIN: &[u8] = b"fandhe-edge/final-test-lock/config/v1\0";
const WEIGHTS_LOCK_DOMAIN: &[u8] = b"fandhe-edge/final-test-lock/weights/v1\0";

/// 検証済みの代表構成 ID（1〜128 バイトの ASCII `[A-Za-z0-9._:-]`）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepresentativeConfigId(String);

impl RepresentativeConfigId {
    /// 文字列を検証して ID にする。違反時は実値を含めずに拒否する。
    pub fn parse(raw: &str) -> Result<Self, AcquireError> {
        if raw.is_empty() {
            return Err(AcquireError::InvalidConfigId { reason: "empty" });
        }
        if raw.len() > MAX_CONFIG_ID_BYTES {
            return Err(AcquireError::InvalidConfigId {
                reason: "longer than 128 bytes",
            });
        }
        let allowed = |b: u8| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b':' | b'-');
        if !raw.bytes().all(allowed) {
            return Err(AcquireError::InvalidConfigId {
                reason: "contains a character outside [A-Za-z0-9._:-]",
            });
        }
        Ok(RepresentativeConfigId(raw.to_string()))
    }

    /// 検証済みの文字列表現。
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// 最終 test 適用の識別キー（評価データ × 代表構成 × 重み）。
///
/// 非公開型。評価データのダイジェストは [`apply_once`] が凍結記録と照合した値だけが
/// 入る（外部から任意ダイジェストで作れない）。
#[derive(Debug, Clone, PartialEq, Eq)]
struct FinalTestKey {
    eval_data_sha256: Sha256Digest,
    config_id: RepresentativeConfigId,
    weights_sha256: Sha256Digest,
}

impl FinalTestKey {
    /// 凍結記録との照合を通過した評価データの sha256・代表構成 ID・モデル
    /// スナップショットから作る。重みのダイジェストが取れなければ fail-closed。
    fn from_verified(
        frozen_eval_sha256: Sha256Digest,
        config_id: RepresentativeConfigId,
        model: &ModelPackageSnapshot,
    ) -> Result<Self, AcquireError> {
        let weights_sha256 = *model
            .digest(ModelComponent::Weights)
            .ok_or(AcquireError::MissingWeightsDigest)?;
        Ok(FinalTestKey {
            eval_data_sha256: frozen_eval_sha256,
            config_id,
            weights_sha256,
        })
    }

    fn config_lock_name(&self) -> String {
        let id = self.config_id.as_str().as_bytes();
        let mut buf = Vec::with_capacity(CONFIG_LOCK_DOMAIN.len() + 32 + 8 + id.len());
        buf.extend_from_slice(CONFIG_LOCK_DOMAIN);
        buf.extend_from_slice(self.eval_data_sha256.as_bytes());
        buf.extend_from_slice(&(id.len() as u64).to_be_bytes());
        buf.extend_from_slice(id);
        format!("config-{}.lock", Sha256Digest::of_bytes(&buf).to_hex())
    }

    fn weights_lock_name(&self) -> String {
        let mut buf = Vec::with_capacity(WEIGHTS_LOCK_DOMAIN.len() + 64);
        buf.extend_from_slice(WEIGHTS_LOCK_DOMAIN);
        buf.extend_from_slice(self.eval_data_sha256.as_bytes());
        buf.extend_from_slice(self.weights_sha256.as_bytes());
        format!("weights-{}.lock", Sha256Digest::of_bytes(&buf).to_hex())
    }

    fn record(&self, kind: &str) -> String {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        format!(
            "fandhe-edge-final-test-application v1\nlock={kind}\neval_data_sha256={}\nrepresentative_config_id={}\nweights_sha256={}\napplied_unix_secs={secs}\n",
            self.eval_data_sha256.to_hex(),
            self.config_id.as_str(),
            self.weights_sha256.to_hex(),
        )
    }
}

/// どのロックが既存だったか。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AppliedBy {
    /// 同じ評価データ × 同じ代表構成。
    RepresentativeConfig,
    /// 同じ評価データ × 同じ重み（構成名やしきい値を変えた当て直し）。
    ModelWeights,
}

/// [`FinalTestLedger`] の操作が返しうるエラー。メッセージは英語。
#[derive(Debug)]
#[non_exhaustive]
pub enum AcquireError {
    /// 既に適用済み。予測は呼ばれない。
    AlreadyApplied {
        /// 既存だったロックの種類。
        by: AppliedBy,
        /// 既存ロックのパス。
        lock_path: PathBuf,
    },
    /// 代表構成 ID が不正（実値は載せない）。
    InvalidConfigId {
        /// 違反の種別。
        reason: &'static str,
    },
    /// モデルスナップショットに重みのダイジェストが無い。
    MissingWeightsDigest,
    /// 台帳ディレクトリが存在しない・ディレクトリでない・symlink。
    LedgerDirInvalid {
        /// 台帳ディレクトリのパス。
        path: PathBuf,
    },
    /// ロックの永続化（ロックファイルまたは台帳ディレクトリの `sync_all`）に失敗した。
    /// 予測は呼ばれない。ロックは残る（適用を試みた事実として消費扱い）。
    DurabilityFailed {
        /// 対象パス。
        path: PathBuf,
        /// 原因。
        source: std::io::Error,
    },
    /// 台帳ディレクトリの fsync ができない環境（非 Unix）で、ロックの永続性を
    /// 保証できない。予測は呼ばれない。
    DurabilityUnsupported,
    /// ロックの作成に失敗した。
    Io {
        /// 対象パス。
        path: PathBuf,
        /// 原因。
        source: std::io::Error,
    },
    /// ロックは作成済み（適用 1 回を消費済み）だが記録の書き込みに失敗した。
    /// fail-closed でロックは残す。
    RecordWriteFailed {
        /// 対象パス。
        path: PathBuf,
        /// 原因。
        source: std::io::Error,
    },
}

impl fmt::Display for AcquireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AcquireError::AlreadyApplied { by, lock_path } => write!(
                f,
                "final test already applied (matched by {}): {}",
                match by {
                    AppliedBy::RepresentativeConfig => "representative config",
                    AppliedBy::ModelWeights => "model weights",
                },
                lock_path.display()
            ),
            AcquireError::InvalidConfigId { reason } => {
                write!(f, "invalid representative config id: {reason}")
            }
            AcquireError::MissingWeightsDigest => {
                write!(f, "model snapshot has no weights digest")
            }
            AcquireError::LedgerDirInvalid { path } => {
                write!(f, "ledger path is not a real directory: {}", path.display())
            }
            AcquireError::DurabilityFailed { path, source } => write!(
                f,
                "failed to persist lock (application consumed) {}: {source}",
                path.display()
            ),
            AcquireError::DurabilityUnsupported => write!(
                f,
                "cannot guarantee lock durability on this platform (directory fsync unsupported)"
            ),
            AcquireError::Io { path, source } => {
                write!(f, "failed to create lock {}: {source}", path.display())
            }
            AcquireError::RecordWriteFailed { path, source } => write!(
                f,
                "lock created but record write failed (application consumed) {}: {source}",
                path.display()
            ),
        }
    }
}

impl std::error::Error for AcquireError {}

/// [`apply_once`] のエラー。予測クロージャ自身のエラー型 `E` を包む。
#[derive(Debug)]
#[non_exhaustive]
pub enum ApplyOnceError<E> {
    /// ロック取得に失敗（予測は呼ばれていない）。
    Acquire(AcquireError),
    /// 予測クロージャが失敗した。ロックは残るため再試行は拒否される。
    Prediction(E),
}

impl<E: fmt::Display> fmt::Display for ApplyOnceError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ApplyOnceError::Acquire(e) => write!(f, "{e}"),
            ApplyOnceError::Prediction(e) => write!(f, "prediction failed: {e}"),
        }
    }
}

impl<E: fmt::Debug + fmt::Display> std::error::Error for ApplyOnceError<E> {}

/// 適用権（消費される値）。取得できた呼び出し側だけが予測を当ててよい。
#[derive(Debug)]
pub struct ApplicationTicket {
    config_lock: PathBuf,
}

impl ApplicationTicket {
    /// 代表構成ロックのパス。
    #[must_use]
    pub fn config_lock_path(&self) -> &Path {
        &self.config_lock
    }
}

/// 呼び出し側が用意した台帳ディレクトリを包む。
#[derive(Debug, Clone)]
pub struct FinalTestLedger {
    dir: PathBuf,
}

impl FinalTestLedger {
    /// symlink でない実ディレクトリであることを検証して開く（作成はしない）。
    pub fn open(dir: &Path) -> Result<Self, AcquireError> {
        match fs::symlink_metadata(dir) {
            Ok(meta) if meta.is_dir() => Ok(FinalTestLedger {
                dir: dir.to_path_buf(),
            }),
            _ => Err(AcquireError::LedgerDirInvalid {
                path: dir.to_path_buf(),
            }),
        }
    }

    fn create_lock(&self, name: &str, by: AppliedBy) -> Result<(File, PathBuf), AcquireError> {
        let path = self.dir.join(name);
        let mut opts = OpenOptions::new();
        opts.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            opts.mode(0o600);
        }
        match opts.open(&path) {
            Ok(file) => Ok((file, path)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                Err(AcquireError::AlreadyApplied {
                    by,
                    lock_path: path,
                })
            }
            Err(source) => Err(AcquireError::Io { path, source }),
        }
    }

    fn write_record(mut file: File, path: &Path, body: &str) -> Result<(), AcquireError> {
        file.write_all(body.as_bytes())
            .and_then(|()| file.sync_all())
            .map_err(|source| AcquireError::RecordWriteFailed {
                path: path.to_path_buf(),
                source,
            })
    }

    /// 台帳ディレクトリのエントリを永続化する。失敗を握りつぶさない（予測後の
    /// クラッシュでロックのエントリが失われると再適用できてしまうため）。
    #[cfg(unix)]
    fn sync_dir(&self) -> Result<(), AcquireError> {
        File::open(&self.dir)
            .and_then(|dir| dir.sync_all())
            .map_err(|source| AcquireError::DurabilityFailed {
                path: self.dir.clone(),
                source,
            })
    }

    /// 非 Unix ではディレクトリの fsync 手段が無く、永続性を保証できないため拒否する。
    #[cfg(not(unix))]
    fn sync_dir(&self) -> Result<(), AcquireError> {
        Err(AcquireError::DurabilityUnsupported)
    }

    /// 適用権を取得する（代表構成ロック → 重みロックの順）。
    fn acquire(&self, key: &FinalTestKey) -> Result<ApplicationTicket, AcquireError> {
        let (cfg_file, cfg_path) =
            self.create_lock(&key.config_lock_name(), AppliedBy::RepresentativeConfig)?;
        // 以降、失敗してもロールバックしない（適用を試みた事実として消費扱い）。
        let (w_file, w_path) =
            self.create_lock(&key.weights_lock_name(), AppliedBy::ModelWeights)?;
        Self::write_record(cfg_file, &cfg_path, &key.record("config"))?;
        Self::write_record(w_file, &w_path, &key.record("weights"))?;
        self.sync_dir()?;
        Ok(ApplicationTicket {
            config_lock: cfg_path,
        })
    }
}

/// 凍結記録との照合 → ロック取得・永続化 → 予測の順で、`predict` を高々 1 回呼ぶ。
///
/// 唯一の公開経路。`frozen` の sha256・バイト長と実データが一致した場合のみ、
/// その照合済み sha256 でロックのキーを作る。`predict` には適用権と、照合済みの
/// 評価データ本体を渡す（評価データ本体は評価器側の経路。推論関数へ渡す情報の
/// 絞り込みは TASK-27.2 の責務）。
pub fn apply_once<T, E>(
    ledger: &FinalTestLedger,
    frozen: &FrozenEvalData<'_>,
    config_id: RepresentativeConfigId,
    model: &ModelPackageSnapshot,
    predict: impl FnOnce(ApplicationTicket, &[u8]) -> Result<T, E>,
) -> Result<T, EvalDataInvarianceError<ApplyOnceError<E>>> {
    evaluate_with_eval_data_invariance(frozen, |bytes| {
        // ここに来た時点で bytes の sha256 == frozen.sha256（照合済み）。
        let key = FinalTestKey::from_verified(frozen.sha256, config_id, model)
            .map_err(ApplyOnceError::Acquire)?;
        let ticket = ledger.acquire(&key).map_err(ApplyOnceError::Acquire)?;
        predict(ticket, bytes).map_err(ApplyOnceError::Prediction)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(b: u8) -> Sha256Digest {
        Sha256Digest::of_bytes(&[b])
    }

    #[test]
    fn req27_lock_names_are_deterministic_hex() {
        let id = RepresentativeConfigId::parse("c1:seed0").unwrap();
        let key = FinalTestKey {
            eval_data_sha256: d(1),
            config_id: id,
            weights_sha256: d(2),
        };
        let a = key.config_lock_name();
        assert_eq!(a, key.config_lock_name());
        assert!(a.starts_with("config-") && a.ends_with(".lock"));
        assert_eq!(a.len(), "config-".len() + 64 + ".lock".len());
        assert_ne!(a[7..], key.weights_lock_name()[8..]);
    }

    #[test]
    fn req27_config_id_boundaries() {
        assert!(RepresentativeConfigId::parse(&"a".repeat(128)).is_ok());
        for bad in ["", "../x", "a/b", "a b", "é"] {
            assert!(matches!(
                RepresentativeConfigId::parse(bad),
                Err(AcquireError::InvalidConfigId { .. })
            ));
        }
        assert!(RepresentativeConfigId::parse(&"a".repeat(129)).is_err());
    }
}
