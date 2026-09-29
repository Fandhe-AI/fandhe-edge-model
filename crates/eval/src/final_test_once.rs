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
//! # 唯一の公開経路（ロック・凍結記録・実モデルの結び付け）
//!
//! ロックのキー（`FinalTestKey`）は非公開型で、コンストラクタも非公開。
//! 公開の入口は [`apply_once`] だけで、次の 3 つを内部で結び付ける。
//!
//! - **評価データ**: [`evaluate_with_eval_data_invariance`] で実データの sha256 が
//!   凍結記録と一致した後にのみ、その照合済みダイジェストからキーを作る。
//! - **モデル**: 呼び出し側が申告するダイジェストは受け取らない。[`ModelPackagePaths`]
//!   を受け取り、[`evaluate_with_invariance`] の内側で重みファイルを自らハッシュして
//!   キーを作り、`predict` へは**同じ [`ModelPackagePaths`]** を渡す。予測に使う
//!   モデルとロックキーが同一のパスから導出され、評価前後のモデル不変性検証
//!   （REQ-27）と一体で動く。
//! - **事前登録**: 候補・seed の代表構成 ID と、その構成で当てるモデルパッケージ全構成要素
//!   （重み・語彙・校正・しきい値。無い要素は「無い」）の sha256 の組
//!   ([`RegisteredConfig`]) を、評価前に [`FinalTestLedger::register_configs`] で
//!   評価データごとに 1 回だけ台帳へ凍結する。[`apply_once`] は登録済み集合に含まれない
//!   ID、および登録したダイジェストと異なる構成要素での適用を拒否する（未使用 ID に別重みを当てて
//!   再適用する迂回を拒否する。適用対象は評価前に確定し、初回適用後に新たな候補を
//!   追加できない。PoC-10 の `APPLIED.json` 事前登録に相当）。登録後は集合を変更できない
//!   （`create_new`）。
//!
//! # 順序の不変条件
//!
//! 1. 凍結記録との照合（`evaluate_with_eval_data_invariance`）が通ってから
//! 2. モデルの評価前スナップショットを取り、事前登録の照合（ロックは作らない）
//! 3. ロックを取得し、永続化（ファイルの `sync_all`、Unix ではディレクトリも）まで確認し
//! 4. 予測を当て、評価後にモデル・評価データの不変性を検証する
//!
//! 事前登録・ID・重みの読み込みに失敗した場合はロックを作らず、適用を消費しない。
//! 永続化の確認に失敗したら予測は呼ばずエラーを返す（fail-closed）。
//!
//! 台帳ディレクトリの fsync は Unix のみ行う。Windows 等ではディレクトリハンドルの
//! `sync_all` が使えないため、ロックファイル自体の `sync_all` までで成功扱いとする
//! （ディレクトリエントリの永続化は OS 任せ。この限界は既知で、ロックのクラッシュ
//! 耐性は Unix より弱い）。
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
//! - 事前登録集合の内容（どの候補・seed を登録するか）の決定は呼び出し側（選定工程）の
//!   責務。本モジュールは登録の凍結と照合のみを行う。
//! - 台帳ディレクトリの配置・ルート配下への閉じ込めはガード層（REQ-39）と呼び出し側の
//!   責務。本モジュールは symlink でない実ディレクトリであることのみ検査する。
//! - 既存予測の再採点（予測を当てない操作）はロック不要のため対象外。
//! - seed は独自フィールドにせず、呼び出し側が事前登録（PoC-10: 代表構成を seed ごとに
//!   1 回）に従って構成 ID へ畳み込む（例 `c1:seed0`）。

use crate::eval_data_invariance::{
    EvalDataInvarianceError, FrozenEvalData, evaluate_with_eval_data_invariance,
};
use crate::invariance::{
    EvaluationInvarianceError, MAX_MODEL_COMPONENT_BYTES, ModelPackagePaths,
    evaluate_with_invariance,
};
use fandhe_edge_core::fs::FsError;
use fandhe_edge_core::hash::Sha256Digest;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};

/// 代表構成 ID の最大バイト数（確保・検証の上限。REQ-39）。
pub const MAX_CONFIG_ID_BYTES: usize = 128;

const CONFIG_LOCK_DOMAIN: &[u8] = b"fandhe-edge/final-test-lock/config/v1\0";
const WEIGHTS_LOCK_DOMAIN: &[u8] = b"fandhe-edge/final-test-lock/weights/v1\0";
const REGISTRY_DOMAIN: &[u8] = b"fandhe-edge/final-test-lock/registry/v1\0";
const REGISTRY_HEADER: &str = "fandhe-edge-final-test-registry v1\n";

/// 事前登録できる代表構成 ID の最大件数（確保・検証の上限。REQ-39）。
pub const MAX_REGISTERED_CONFIGS: usize = 1024;

/// 事前登録ファイルの最大バイト数（読み込み前のサイズ上限。REQ-39）。
/// 128 バイトの ID と sha256（hex 64 桁）4 個の行を最大件数並べても収まる値。
const MAX_REGISTRY_BYTES: u64 = (MAX_CONFIG_ID_BYTES as u64 + 4 * (1 + 64) + 1)
    * MAX_REGISTERED_CONFIGS as u64
    + REGISTRY_HEADER.len() as u64;

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

/// 事前登録する 1 件（代表構成 ID と、その構成で最終 test に当てる重みの sha256）。
///
/// 重み・語彙・校正・しきい値のダイジェストを評価前に ID へ結び付けて凍結することで、登録済みだが未使用の
/// ID に別の重みを当てて最終 test を再適用する迂回を拒否する（REQ-27）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisteredConfig {
    id: RepresentativeConfigId,
    weights_sha256: Sha256Digest,
    vocab_sha256: Option<Sha256Digest>,
    calibration_sha256: Option<Sha256Digest>,
    thresholds_sha256: Option<Sha256Digest>,
}

impl RegisteredConfig {
    /// 代表構成 ID と、評価前に確定した重みファイルの sha256 から作る。
    #[must_use]
    pub fn new(id: RepresentativeConfigId, weights_sha256: Sha256Digest) -> Self {
        RegisteredConfig {
            id,
            weights_sha256,
            vocab_sha256: None,
            calibration_sha256: None,
            thresholds_sha256: None,
        }
    }

    /// 語彙ファイルの sha256 も結び付ける（呼ばなければ「語彙なし」を登録する）。
    #[must_use]
    pub fn with_vocab(mut self, sha256: Sha256Digest) -> Self {
        self.vocab_sha256 = Some(sha256);
        self
    }

    /// 校正パラメータファイルの sha256 も結び付ける。
    #[must_use]
    pub fn with_calibration(mut self, sha256: Sha256Digest) -> Self {
        self.calibration_sha256 = Some(sha256);
        self
    }

    /// しきい値ファイルの sha256 も結び付ける。
    #[must_use]
    pub fn with_thresholds(mut self, sha256: Sha256Digest) -> Self {
        self.thresholds_sha256 = Some(sha256);
        self
    }

    /// モデルパッケージの全構成要素（重み・語彙・校正・しきい値）を自らハッシュして
    /// 登録項目を作る。存在しない構成要素（`None`）は「無い」ことを登録する。
    ///
    /// # Errors
    ///
    /// いずれかのファイルのダイジェストを計算できない場合（[`AcquireError::WeightsDigest`]）。
    pub fn from_package(
        id: RepresentativeConfigId,
        package: &ModelPackagePaths<'_>,
    ) -> Result<Self, AcquireError> {
        Ok(RegisteredConfig {
            id,
            weights_sha256: digest_component(package.weights)?,
            vocab_sha256: digest_optional(package.vocab)?,
            calibration_sha256: digest_optional(package.calibration)?,
            thresholds_sha256: digest_optional(package.thresholds)?,
        })
    }
}

fn digest_component(path: &Path) -> Result<Sha256Digest, AcquireError> {
    fandhe_edge_core::fs::sha256_file_bounded(path, MAX_MODEL_COMPONENT_BYTES)
        .map_err(|source| AcquireError::WeightsDigest { source })
}

fn digest_optional(path: Option<&Path>) -> Result<Option<Sha256Digest>, AcquireError> {
    path.map(digest_component).transpose()
}

fn opt_hex(d: &Option<Sha256Digest>) -> String {
    d.as_ref().map_or_else(|| "-".to_string(), |d| d.to_hex())
}

fn parse_opt_digest(raw: &str) -> Result<Option<Sha256Digest>, AcquireError> {
    if raw == "-" {
        return Ok(None);
    }
    raw.parse::<Sha256Digest>()
        .map(Some)
        .map_err(|_| AcquireError::RegistryInvalid {
            reason: "malformed registry digest",
        })
}

/// 評価データ × 代表構成のロックファイル名（ハッシュ由来の固定長 hex のみ）。
fn config_lock_name(eval_data_sha256: &Sha256Digest, config_id: &RepresentativeConfigId) -> String {
    let id = config_id.as_str().as_bytes();
    let mut buf = Vec::with_capacity(CONFIG_LOCK_DOMAIN.len() + 32 + 8 + id.len());
    buf.extend_from_slice(CONFIG_LOCK_DOMAIN);
    buf.extend_from_slice(eval_data_sha256.as_bytes());
    buf.extend_from_slice(&(id.len() as u64).to_be_bytes());
    buf.extend_from_slice(id);
    format!("config-{}.lock", Sha256Digest::of_bytes(&buf).to_hex())
}

/// 評価データごとの事前登録ファイル名。
fn registry_name(eval_data_sha256: &Sha256Digest) -> String {
    let mut buf = Vec::with_capacity(REGISTRY_DOMAIN.len() + 32);
    buf.extend_from_slice(REGISTRY_DOMAIN);
    buf.extend_from_slice(eval_data_sha256.as_bytes());
    format!("registry-{}.lock", Sha256Digest::of_bytes(&buf).to_hex())
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
    /// 凍結記録との照合を通過した評価データの sha256・代表構成 ID・予測に使う
    /// 重みファイルから [`apply_once`] が自ら計算したダイジェストから作る。
    fn from_verified(
        frozen_eval_sha256: Sha256Digest,
        config_id: RepresentativeConfigId,
        weights_sha256: Sha256Digest,
    ) -> Self {
        FinalTestKey {
            eval_data_sha256: frozen_eval_sha256,
            config_id,
            weights_sha256,
        }
    }

    fn config_lock_name(&self) -> String {
        config_lock_name(&self.eval_data_sha256, &self.config_id)
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
    /// 予測に使う重みファイルのダイジェストを計算できなかった（読み込み失敗・
    /// サイズ上限超過・通常ファイルでない）。ロックは作られない。
    WeightsDigest {
        /// 原因。
        source: FsError,
    },
    /// この評価データに代表構成 ID の事前登録が無い。[`FinalTestLedger::register_configs`]
    /// を評価前に呼ぶ必要がある。ロックは作られない。
    NotRegistered,
    /// 代表構成 ID が事前登録集合に含まれない。ロックは作られない。
    UnregisteredConfig,
    /// 予測に使う重みが、事前登録でその ID に結び付けた重みと一致しない。
    /// ロックは作られない。
    WeightsNotRegistered,
    /// 予測に使う語彙・校正・しきい値のいずれかが、事前登録でその ID に結び付けた
    /// ダイジェスト（または「無い」）と一致しない。ロックは作られない。
    ComponentNotRegistered {
        /// 不一致の構成要素名（`vocab` / `calibration` / `thresholds`）。
        component: &'static str,
    },
    /// この評価データの事前登録は既にある（登録の変更・上書きは拒否する）、または
    /// 登録しようとした ID の適用が既に行われている。
    AlreadyRegistered {
        /// 既存の登録ファイルまたは既存ロックのパス。
        path: PathBuf,
    },
    /// 事前登録の内容が不正（空・件数超過・形式違反）。
    RegistryInvalid {
        /// 違反の種別（ID の実値は載せない）。
        reason: &'static str,
    },
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
            AcquireError::WeightsDigest { source } => {
                write!(f, "failed to compute weights digest: {source}")
            }
            AcquireError::NotRegistered => write!(
                f,
                "no representative config registration for this eval data"
            ),
            AcquireError::UnregisteredConfig => {
                write!(f, "representative config id is not registered")
            }
            AcquireError::WeightsNotRegistered => {
                write!(f, "model weights do not match the registered weights")
            }
            AcquireError::ComponentNotRegistered { component } => {
                write!(f, "model {component} does not match the registered digest")
            }
            AcquireError::AlreadyRegistered { path } => {
                write!(
                    f,
                    "registration conflicts with existing state: {}",
                    path.display()
                )
            }
            AcquireError::RegistryInvalid { reason } => {
                write!(f, "invalid registration: {reason}")
            }
            AcquireError::LedgerDirInvalid { path } => {
                write!(f, "ledger path is not a real directory: {}", path.display())
            }
            AcquireError::DurabilityFailed { path, source } => write!(
                f,
                "failed to persist lock (application consumed) {}: {source}",
                path.display()
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

    /// 非 Unix（Windows 等）ではディレクトリハンドルの `sync_all` が使えない。
    /// ロックファイル自体は `write_record` で `sync_all` 済みのため成功扱いとする
    /// （ディレクトリエントリの永続化は OS 任せ。モジュール docs の限界を参照）。
    #[cfg(not(unix))]
    fn sync_dir(&self) -> Result<(), AcquireError> {
        let _ = &self.dir;
        Ok(())
    }

    /// 評価データの事前登録を読む。無ければ [`AcquireError::NotRegistered`]。
    fn load_registry(
        &self,
        eval_data_sha256: &Sha256Digest,
    ) -> Result<Vec<RegisteredConfig>, AcquireError> {
        let path = self.dir.join(registry_name(eval_data_sha256));
        // 通常ファイル判定・`O_NONBLOCK` 付きオープン・サイズ上限付き読み込みは共通コアに
        // 集約されている。FIFO 等では open 前に拒否され、無期限に待たない（REQ-39）。
        let bytes = match fandhe_edge_core::fs::read_bounded(&path, MAX_REGISTRY_BYTES) {
            Ok(b) => b,
            Err(FsError::Read { source, .. }) if source.kind() == std::io::ErrorKind::NotFound => {
                return Err(AcquireError::NotRegistered);
            }
            Err(FsError::Read { source, .. }) => return Err(AcquireError::Io { path, source }),
            Err(FsError::TooLarge { .. }) => {
                return Err(AcquireError::RegistryInvalid {
                    reason: "registry file too large",
                });
            }
            Err(_) => {
                return Err(AcquireError::RegistryInvalid {
                    reason: "registry is not a regular file",
                });
            }
        };
        let body = String::from_utf8(bytes).map_err(|_| AcquireError::RegistryInvalid {
            reason: "registry is not valid utf-8",
        })?;
        parse_registry(&body)
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

/// [`apply_once`] の戻り値。外側が評価データ、内側がモデルの不変性検証。
pub type ApplyOnceResult<T, E> =
    Result<T, EvalDataInvarianceError<EvaluationInvarianceError<ApplyOnceError<E>>>>;

/// 事前登録集合の正準化本文（ID 順にソート済みの `<ID> <重み> <語彙> <校正> <しきい値>`（sha256 hex。無い要素は `-`）を 1 行ずつ）。
fn registry_body(entries: &[RegisteredConfig]) -> String {
    let mut out = String::from(REGISTRY_HEADER);
    for e in entries {
        out.push_str(e.id.as_str());
        for d in [
            Some(e.weights_sha256),
            e.vocab_sha256,
            e.calibration_sha256,
            e.thresholds_sha256,
        ] {
            out.push(' ');
            out.push_str(&opt_hex(&d));
        }
        out.push('\n');
    }
    out
}

fn parse_registry(body: &str) -> Result<Vec<RegisteredConfig>, AcquireError> {
    let rest = body
        .strip_prefix(REGISTRY_HEADER)
        .ok_or(AcquireError::RegistryInvalid {
            reason: "bad registry header",
        })?;
    let mut entries = Vec::new();
    for line in rest.lines() {
        if entries.len() >= MAX_REGISTERED_CONFIGS {
            return Err(AcquireError::RegistryInvalid {
                reason: "too many registered configs",
            });
        }
        let fields: Vec<&str> = line.split(' ').collect();
        let [id, w, v, c, t] = fields[..] else {
            return Err(AcquireError::RegistryInvalid {
                reason: "malformed registry line",
            });
        };
        let Some(weights_sha256) = parse_opt_digest(w)? else {
            return Err(AcquireError::RegistryInvalid {
                reason: "malformed registry digest",
            });
        };
        entries.push(RegisteredConfig {
            id: RepresentativeConfigId::parse(id)?,
            weights_sha256,
            vocab_sha256: parse_opt_digest(v)?,
            calibration_sha256: parse_opt_digest(c)?,
            thresholds_sha256: parse_opt_digest(t)?,
        });
    }
    if entries.is_empty() {
        return Err(AcquireError::RegistryInvalid {
            reason: "empty registry",
        });
    }
    Ok(entries)
}

impl FinalTestLedger {
    /// 評価データ（凍結記録の sha256）に対し、最終 test に適用してよい代表構成 ID の
    /// 集合を評価前に 1 回だけ凍結する（REQ-27。PoC-10 の事前登録）。
    ///
    /// - 集合はソート・重複除去して `create_new` で保存し、以後変更できない
    ///   （同じ評価データへの再登録は [`AcquireError::AlreadyRegistered`]）。
    /// - 登録しようとする ID のいずれかが既に適用済み（代表構成ロックが存在）の場合も
    ///   拒否する（適用後の事後登録で履歴を正当化させない）。
    /// - 各 ID には重みの sha256 を結び付けて凍結する。同じ ID に異なる重みを登録する
    ///   ことはできない（完全に同一の重複は除去する）。
    /// - 空集合・[`MAX_REGISTERED_CONFIGS`] 超過は [`AcquireError::RegistryInvalid`]。
    ///   件数は複製・ソートの前に検査する（REQ-39）。
    pub fn register_configs(
        &self,
        eval_data_sha256: &Sha256Digest,
        entries: &[RegisteredConfig],
    ) -> Result<(), AcquireError> {
        if entries.len() > MAX_REGISTERED_CONFIGS {
            return Err(AcquireError::RegistryInvalid {
                reason: "too many registered configs",
            });
        }
        let mut sorted: Vec<RegisteredConfig> = entries.to_vec();
        sorted.sort_by(|a, b| {
            a.id.as_str().cmp(b.id.as_str()).then_with(|| {
                registry_body(std::slice::from_ref(a)).cmp(&registry_body(std::slice::from_ref(b)))
            })
        });
        sorted.dedup();
        if sorted.is_empty() {
            return Err(AcquireError::RegistryInvalid {
                reason: "empty registry",
            });
        }
        if sorted.windows(2).any(|w| w[0].id == w[1].id) {
            return Err(AcquireError::RegistryInvalid {
                reason: "same config id with different weights",
            });
        }
        for entry in &sorted {
            let path = self.dir.join(config_lock_name(eval_data_sha256, &entry.id));
            if fs::symlink_metadata(&path).is_ok() {
                return Err(AcquireError::AlreadyRegistered { path });
            }
        }
        let (file, path) = match self.create_lock(
            &registry_name(eval_data_sha256),
            AppliedBy::RepresentativeConfig,
        ) {
            Ok(v) => v,
            Err(AcquireError::AlreadyApplied { lock_path, .. }) => {
                return Err(AcquireError::AlreadyRegistered { path: lock_path });
            }
            Err(e) => return Err(e),
        };
        Self::write_record(file, &path, &registry_body(&sorted))?;
        self.sync_dir()
    }
}

/// 凍結記録との照合 → 事前登録の照合 → ロック取得・永続化 → 予測の順で、
/// `predict` を高々 1 回呼ぶ。唯一の公開経路。
///
/// - `frozen` の sha256・バイト長と実データが一致した場合のみ、その照合済み sha256 で
///   ロックのキーを作る。
/// - `model` の重みファイルは本関数が自らハッシュしてキーにし、`predict` へは同じ
///   `model` を渡す。評価前後のモデル不変性も本関数が検証する（REQ-27）。
/// - `config_id` は [`FinalTestLedger::register_configs`] で事前登録済みの ID に限る。
/// - `predict` には適用権・照合済み評価データ本体・`model` を渡す（推論関数へ渡す
///   情報の絞り込みは TASK-27.2 の責務）。`predict` は渡された `model` のパスから
///   モデルを読むこと。ロックは `model` の重みに対して消費される。
pub fn apply_once<T, E>(
    ledger: &FinalTestLedger,
    frozen: &FrozenEvalData<'_>,
    config_id: RepresentativeConfigId,
    model: &ModelPackagePaths<'_>,
    predict: impl FnOnce(ApplicationTicket, &[u8], &ModelPackagePaths<'_>) -> Result<T, E>,
) -> ApplyOnceResult<T, E> {
    evaluate_with_eval_data_invariance(frozen, |bytes| {
        // ここに来た時点で bytes の sha256 == frozen.sha256（照合済み）。
        evaluate_with_invariance(model, |paths| {
            let registered = ledger
                .load_registry(&frozen.sha256)
                .map_err(ApplyOnceError::Acquire)?;
            let Some(entry) = registered.iter().find(|e| e.id == config_id) else {
                return Err(ApplyOnceError::Acquire(AcquireError::UnregisteredConfig));
            };
            let weights_sha256 =
                digest_component(paths.weights).map_err(ApplyOnceError::Acquire)?;
            if weights_sha256 != entry.weights_sha256 {
                return Err(ApplyOnceError::Acquire(AcquireError::WeightsNotRegistered));
            }
            // 語彙・校正・しきい値も登録時のダイジェスト（または「無い」）と照合する。
            // 重みだけ一致させて未使用構成の他要素を調整する迂回を拒否する（REQ-27）。
            for (component, path, registered) in [
                ("vocab", paths.vocab, &entry.vocab_sha256),
                ("calibration", paths.calibration, &entry.calibration_sha256),
                ("thresholds", paths.thresholds, &entry.thresholds_sha256),
            ] {
                let actual = digest_optional(path).map_err(ApplyOnceError::Acquire)?;
                if &actual != registered {
                    return Err(ApplyOnceError::Acquire(
                        AcquireError::ComponentNotRegistered { component },
                    ));
                }
            }
            let key = FinalTestKey::from_verified(frozen.sha256, config_id, weights_sha256);
            let ticket = ledger.acquire(&key).map_err(ApplyOnceError::Acquire)?;
            predict(ticket, bytes, paths).map_err(ApplyOnceError::Prediction)
        })
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
