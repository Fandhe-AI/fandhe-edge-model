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
//! - **事前登録の改変検出**: 登録ファイルは `create_new` の後に読み取り専用（Unix では
//!   0400）へ落とし、登録本文の sha256（共通コアの [`Sha256Digest`]。正準化は
//!   `registry_body` の 1 箇所）を別の封印ファイルへ 1 回だけ書いて同じく読み取り専用にする。
//!   [`apply_once`] は適用前に、(1) 本文と封印の一致、(2) 登録・封印がともに読み取り専用、
//!   (3) 登録済み ID のうち適用済みのロックが記録した登録ダイジェストとの一致、を確認し、
//!   違反なら [`AcquireError::RegistryTampered`] で拒否する（ロックは作らない。fail-closed）。
//!   (3) により、権限を戻して登録と封印の両方を差し替えても、適用済みの構成が 1 つでも
//!   あれば検出できる。限界: 台帳ディレクトリを自由に書ける主体が、適用前に登録と封印を
//!   まとめて差し替える、または適用済みロックまで含めて全て作り直す場合は、台帳内の
//!   記録だけでは検出できない（台帳ディレクトリの保護は REQ-39 のガード層の責務）。
//!   非 Unix でも `Permissions::readonly` で同じ検査をする。
//! - **台帳の配置**: 登録・封印・適用ロックは評価データごとのサブディレクトリ
//!   `<台帳>/<scope>/`（scope は評価データ sha256 のドメイン分離付き sha256 の hex 64 桁。
//!   `create_dir`・Unix では 0700・symlink は拒否）に置く。既存ロックの照合はそのサブ
//!   ディレクトリだけを列挙し、エントリ総数（許可パターン外を含む）に上限を置く。他の
//!   評価データのファイルは列挙されず、走査量にも適用の可否にも影響しない（REQ-39）。
//! - **推論への入力**: 照合済みの評価データ本体は評価器側の `decode` で `input` と
//!   正解に分け、`predict` へは `input` の列だけを渡す（REQ-27）。正解・評価データ
//!   本体は予測側から見えず、正解は [`AppliedOnce::golds`] として評価器側へ返す。
//!
//! # 順序の不変条件
//!
//! 1. 凍結記録との照合（`evaluate_with_eval_data_invariance`）が通ってから
//! 2. モデルの評価前スナップショットを取り、事前登録の照合（ロックは作らない）
//! 3. ロックを取得し、永続化（ファイルの `sync_all`、Unix ではディレクトリも）まで確認し
//! 4. 予測を当て、評価後にモデル・評価データの不変性を検証する
//!
//! 評価データ本文は 3 のロック取得後にのみ `decode` へ渡す（本文を見てから `Err` で
//! 抜けて呼び直す迂回を塞ぐ）。分解失敗も適用を消費する。
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
//! # 適用権の消費と評価の成功は別の状態（REQ-27）
//!
//! 予測と評価後の不変性検証まで成功した場合だけ、適用ロックとは別ファイル `done-<hex>.lock`（完了記録。
//! 読み取り専用）を書く。失敗した適用はロックのみが残り、[`FinalTestLedger::is_applied`] は偽を返す
//! （`package` が評価完了の根拠にするのは完了記録）。
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
const COMPLETION_DOMAIN: &[u8] = b"fandhe-edge/final-test-lock/completion/v1\0";
const REGISTRY_DOMAIN: &[u8] = b"fandhe-edge/final-test-lock/registry/v1\0";
const REGISTRY_SEAL_DOMAIN: &[u8] = b"fandhe-edge/final-test-lock/registry-seal/v1\0";
const REGISTRY_CONTENT_DOMAIN: &[u8] = b"fandhe-edge/final-test-lock/registry-content/v1\0";
const SEAL_HEADER: &str = "fandhe-edge-final-test-registry-seal v1\n";
const REGISTRY_HEADER: &str = "fandhe-edge-final-test-registry v1\n";

/// 封印ファイル・適用ロック記録の最大バイト数（読み込み前のサイズ上限。REQ-39）。
const MAX_SEAL_BYTES: u64 = 256;
const MAX_LOCK_RECORD_BYTES: u64 = 1024;

/// 1 つの評価データの台帳サブディレクトリで列挙してよいエントリの最大数（許可パターンに
/// 合わないファイルも数える。走査時間・I/O の上限。REQ-39）。他の評価データの
/// サブディレクトリは列挙しない。
const MAX_LOCK_SCAN_ENTRIES: usize = 65_536;

const EVAL_SCOPE_DOMAIN: &[u8] = b"fandhe-edge/final-test-lock/eval-scope/v1\0";

/// 評価データごとの台帳サブディレクトリ名（ドメイン分離付き sha256 の hex 64 桁）。
/// 登録・封印・適用ロックはすべてこの中に置き、走査を他の評価データから切り離す。
/// コードが生成した hex のみで、呼び出し側の文字列は入らない。
fn eval_scope(eval_data_sha256: &Sha256Digest) -> String {
    let mut buf = Vec::with_capacity(EVAL_SCOPE_DOMAIN.len() + 32);
    buf.extend_from_slice(EVAL_SCOPE_DOMAIN);
    buf.extend_from_slice(eval_data_sha256.as_bytes());
    Sha256Digest::of_bytes(&buf).to_hex()
}

/// 適用ロックのファイル名（`config-` または `weights-` + 小文字 hex 64 桁 + `.lock`）か。
fn is_lock_file_name(name: &str) -> bool {
    ["config-", "weights-"].iter().any(|prefix| {
        name.strip_prefix(prefix)
            .and_then(|r| r.strip_suffix(".lock"))
            .is_some_and(|h| {
                h.len() == 64 && h.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
            })
    })
}

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

/// 事前登録の封印ファイル名（登録本文の sha256 を 1 回だけ書き込む別ファイル）。
fn seal_name(eval_data_sha256: &Sha256Digest) -> String {
    let mut buf = Vec::with_capacity(REGISTRY_SEAL_DOMAIN.len() + 32);
    buf.extend_from_slice(REGISTRY_SEAL_DOMAIN);
    buf.extend_from_slice(eval_data_sha256.as_bytes());
    format!("seal-{}.lock", Sha256Digest::of_bytes(&buf).to_hex())
}

/// 事前登録の正準化本文（[`registry_body`]）のドメイン分離付き sha256。
/// 正準化は [`registry_body`] の 1 箇所に集約し、ハッシュは共通コアの
/// [`Sha256Digest`] で計算する（新しい正準化規則は持たない）。
fn registry_digest(body: &[u8]) -> Sha256Digest {
    let mut buf = Vec::with_capacity(REGISTRY_CONTENT_DOMAIN.len() + body.len());
    buf.extend_from_slice(REGISTRY_CONTENT_DOMAIN);
    buf.extend_from_slice(body);
    Sha256Digest::of_bytes(&buf)
}

fn is_read_only(path: &Path) -> bool {
    fs::symlink_metadata(path)
        .map(|m| m.file_type().is_file() && m.permissions().readonly())
        .unwrap_or(false)
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

    /// 評価成功（予測・評価後の不変性検証まで完了）の記録ファイル名。適用権の消費（ロック）とは
    /// 別の状態で、[`FinalTestLedger::is_applied`] はこのファイルがある場合だけ完了とみなす。
    fn completion_name(&self) -> String {
        let mut buf = Vec::with_capacity(COMPLETION_DOMAIN.len() + 96);
        buf.extend_from_slice(COMPLETION_DOMAIN);
        buf.extend_from_slice(self.eval_data_sha256.as_bytes());
        buf.extend_from_slice(self.weights_sha256.as_bytes());
        buf.extend_from_slice(&(self.config_id.as_str().len() as u64).to_be_bytes());
        buf.extend_from_slice(self.config_id.as_str().as_bytes());
        format!("done-{}.lock", Sha256Digest::of_bytes(&buf).to_hex())
    }

    fn record(&self, kind: &str, registry_sha256: &Sha256Digest) -> String {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        format!(
            "fandhe-edge-final-test-application v1\nlock={kind}\neval_data_sha256={}\nrepresentative_config_id={}\nweights_sha256={}\nregistry_sha256={}\napplied_unix_secs={secs}\n",
            self.eval_data_sha256.to_hex(),
            self.config_id.as_str(),
            self.weights_sha256.to_hex(),
            registry_sha256.to_hex(),
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
    /// 事前登録が改変された、または改変を検出できない状態にある（封印ファイルの欠落・
    /// 内容不一致、登録・封印が読み取り専用でない、適用済みロックの記録との不一致）。
    /// ロックは作られず、適用は拒否される（fail-closed。REQ-27）。
    RegistryTampered {
        /// 違反の種別（登録内容の実値は載せない）。
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
            AcquireError::RegistryTampered { reason } => {
                write!(f, "registration integrity check failed: {reason}")
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
#[non_exhaustive]
pub enum ApplyOnceError<E> {
    /// ロック取得に失敗（予測は呼ばれていない）。
    Acquire(AcquireError),
    /// 評価データの分解に失敗した（予測は呼ばれていない）。ロックは取得済みで
    /// 適用は消費されており、再試行は拒否される（REQ-27）。理由は持たない
    /// （分解側が返す文字列に評価データの本文が混ざる経路を型で塞ぐ。REQ-39）。
    Decode,
    /// 予測クロージャが失敗した。ロックは残るため再試行は拒否される。
    ///
    /// `E` は呼び出し側の値としてそのまま返すが、本型の `Display`・`Debug` には
    /// 中身を出さない（`E` が入力本文を抱えていても公開エラー・ログへ流さない。REQ-39）。
    Prediction(E),
}

impl<E> fmt::Debug for ApplyOnceError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ApplyOnceError::Acquire(e) => f.debug_tuple("Acquire").field(e).finish(),
            ApplyOnceError::Decode => f.write_str("Decode"),
            ApplyOnceError::Prediction(_) => f.write_str("Prediction(..)"),
        }
    }
}

impl<E> fmt::Display for ApplyOnceError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ApplyOnceError::Acquire(e) => write!(f, "{e}"),
            ApplyOnceError::Decode => write!(f, "failed to decode eval data"),
            ApplyOnceError::Prediction(_) => write!(f, "prediction failed"),
        }
    }
}

impl<E> std::error::Error for ApplyOnceError<E> {}

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

    fn create_lock(dir: &Path, name: &str, by: AppliedBy) -> Result<(File, PathBuf), AcquireError> {
        let path = dir.join(name);
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

    /// 書き込み権を外す（登録・封印を通常のファイル操作で書き換えられなくする。
    /// 権限を戻せる主体には効かないため、改変の検出は封印・ロック記録との照合が担う）。
    fn make_read_only(path: &Path) -> Result<(), AcquireError> {
        let io = |source| AcquireError::Io {
            path: path.to_path_buf(),
            source,
        };
        let mut perms = fs::metadata(path).map_err(io)?.permissions();
        perms.set_readonly(true);
        fs::set_permissions(path, perms).map_err(io)
    }

    /// 台帳ディレクトリのエントリを永続化する。失敗を握りつぶさない（予測後の
    /// クラッシュでロックのエントリが失われると再適用できてしまうため）。
    #[cfg(unix)]
    fn sync_dir(dir: &Path) -> Result<(), AcquireError> {
        File::open(dir)
            .and_then(|d| d.sync_all())
            .map_err(|source| AcquireError::DurabilityFailed {
                path: dir.to_path_buf(),
                source,
            })
    }

    /// 非 Unix（Windows 等）ではディレクトリハンドルの `sync_all` が使えない。
    /// ロックファイル自体は `write_record` で `sync_all` 済みのため成功扱いとする
    /// （ディレクトリエントリの永続化は OS 任せ。モジュール docs の限界を参照）。
    #[cfg(not(unix))]
    fn sync_dir(dir: &Path) -> Result<(), AcquireError> {
        let _ = dir;
        Ok(())
    }

    /// 評価データごとのサブディレクトリのパス（名前はコードが生成した hex のみ）。
    fn scope_dir(&self, eval_data_sha256: &Sha256Digest) -> PathBuf {
        self.dir.join(eval_scope(eval_data_sha256))
    }

    /// 評価データのサブディレクトリを `create_dir`（Unix では 0700）で作る。既存なら
    /// symlink でない実ディレクトリであることを検証する（`symlink_metadata`）。
    fn ensure_scope_dir(&self, eval_data_sha256: &Sha256Digest) -> Result<PathBuf, AcquireError> {
        let path = self.scope_dir(eval_data_sha256);
        #[cfg_attr(not(unix), allow(unused_mut))]
        let mut builder = fs::DirBuilder::new();
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt as _;
            builder.mode(0o700);
        }
        match builder.create(&path) {
            Ok(()) => Self::sync_dir(&self.dir)?,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(source) => return Err(AcquireError::Io { path, source }),
        }
        match fs::symlink_metadata(&path) {
            Ok(m) if m.is_dir() => Ok(path),
            _ => Err(AcquireError::LedgerDirInvalid { path }),
        }
    }

    /// 評価データの事前登録を読む。無ければ [`AcquireError::NotRegistered`]。
    ///
    /// 読み込んだ本文は、登録時に別ファイル（封印）へ 1 回だけ書いた sha256 と一致し、
    /// 登録・封印がともに読み取り専用のままで、かつ適用済みのロックが記録した登録
    /// ダイジェストとも一致する場合にだけ返す。いずれかに違反したら
    /// [`AcquireError::RegistryTampered`]（REQ-27。適用後に登録へ未使用 ID を足して
    /// 適用対象を選び直す迂回を拒否する）。
    fn load_registry(
        &self,
        eval_data_sha256: &Sha256Digest,
    ) -> Result<(Vec<RegisteredConfig>, Sha256Digest), AcquireError> {
        let sdir = self.scope_dir(eval_data_sha256);
        match fs::symlink_metadata(&sdir) {
            Ok(m) if m.is_dir() => {}
            Ok(_) => {
                return Err(AcquireError::RegistryTampered {
                    reason: "scope directory is not a real directory",
                });
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(AcquireError::NotRegistered);
            }
            Err(source) => return Err(AcquireError::Io { path: sdir, source }),
        }
        let path = sdir.join(registry_name(eval_data_sha256));
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
        let actual = registry_digest(&bytes);
        let seal_path = sdir.join(seal_name(eval_data_sha256));
        let seal_bytes = match fandhe_edge_core::fs::read_bounded(&seal_path, MAX_SEAL_BYTES) {
            Ok(b) => b,
            Err(_) => {
                return Err(AcquireError::RegistryTampered {
                    reason: "registry seal is missing or unreadable",
                });
            }
        };
        let sealed = String::from_utf8(seal_bytes)
            .ok()
            .and_then(|t| {
                t.strip_prefix(SEAL_HEADER)
                    .and_then(|r| r.strip_prefix("registry_sha256="))
                    .and_then(|r| r.strip_suffix('\n'))
                    .and_then(|r| r.parse::<Sha256Digest>().ok())
            })
            .ok_or(AcquireError::RegistryTampered {
                reason: "registry seal is malformed",
            })?;
        if sealed != actual {
            return Err(AcquireError::RegistryTampered {
                reason: "registry content does not match its seal",
            });
        }
        if !is_read_only(&path) || !is_read_only(&seal_path) {
            return Err(AcquireError::RegistryTampered {
                reason: "registry or seal is writable",
            });
        }
        let body = String::from_utf8(bytes).map_err(|_| AcquireError::RegistryInvalid {
            reason: "registry is not valid utf-8",
        })?;
        let entries = parse_registry(&body)?;
        // 適用済みのロックは、適用時点の登録ダイジェストを記録している。登録と封印の
        // 両方を差し替えても、既存ロックの記録と食い違えば検出できる。現在の登録に含まれる
        // ID に限らず、この評価データの既存ロックをすべて列挙して照合する（登録から
        // 適用済み ID を除いた集合への差し替えを検出するため）。
        Self::check_existing_locks(&sdir, eval_data_sha256, &sealed, MAX_LOCK_SCAN_ENTRIES)?;
        Ok((entries, sealed))
    }

    /// この評価データのサブディレクトリ内の適用ロック（`config-<hex>.lock`・
    /// `weights-<hex>.lock`）をすべて照合し、記録された登録ダイジェストが `sealed` と
    /// 一致することを確認する。
    ///
    /// 列挙するのは `sdir`（この評価データのサブディレクトリ）だけで、他の評価データの
    /// ファイルは読まず・数えず、適用の可否にも走査量にも影響しない。列挙したエントリ
    /// 総数（許可パターン外を含む）に `limit`（通常は [`MAX_LOCK_SCAN_ENTRIES`]）を置き、
    /// 1 件のサイズにも [`MAX_LOCK_RECORD_BYTES`] の上限を置く。読めない・壊れた・通常
    /// ファイルでないロックは fail-closed で拒否する。空のロックは、代表構成ロック作成後に
    /// 重みロックの衝突で記録を書かず失敗した消費済みの残骸で、記録が無いので照合しない。
    fn check_existing_locks(
        sdir: &Path,
        eval_data_sha256: &Sha256Digest,
        sealed: &Sha256Digest,
        limit: usize,
    ) -> Result<(), AcquireError> {
        let tampered = |reason| AcquireError::RegistryTampered { reason };
        let entries = fs::read_dir(sdir).map_err(|_| tampered("ledger is unreadable"))?;
        let want_eval = eval_data_sha256.to_hex();
        let want_registry = sealed.to_hex();
        let mut scanned = 0usize;
        for entry in entries {
            let entry = entry.map_err(|_| tampered("ledger is unreadable"))?;
            let name = entry.file_name();
            // 許可パターンに合わないエントリも数える（走査量の上限。REQ-39）。
            scanned += 1;
            if scanned > limit {
                return Err(tampered("too many ledger entries"));
            }
            let Some(name) = name.to_str() else { continue };
            if !is_lock_file_name(name) {
                continue;
            }
            let bytes = fandhe_edge_core::fs::read_bounded(&entry.path(), MAX_LOCK_RECORD_BYTES)
                .map_err(|_| tampered("application lock is unreadable"))?;
            if bytes.is_empty() {
                continue;
            }
            let text =
                String::from_utf8(bytes).map_err(|_| tampered("application lock is malformed"))?;
            let field = |key: &str| {
                text.lines()
                    .find_map(|l| l.strip_prefix(key).and_then(|r| r.strip_prefix('=')))
            };
            let Some(eval) = field("eval_data_sha256") else {
                return Err(tampered("application lock is malformed"));
            };
            if eval != want_eval {
                return Err(tampered("application lock is malformed"));
            }
            if field("registry_sha256") != Some(want_registry.as_str()) {
                return Err(tampered(
                    "registry does not match the digest recorded at application",
                ));
            }
        }
        Ok(())
    }

    /// 適用権を取得する（代表構成ロック → 重みロックの順）。
    fn acquire(
        &self,
        key: &FinalTestKey,
        registry_sha256: &Sha256Digest,
    ) -> Result<ApplicationTicket, AcquireError> {
        let sdir = self.scope_dir(&key.eval_data_sha256);
        let (cfg_file, cfg_path) = Self::create_lock(
            &sdir,
            &key.config_lock_name(),
            AppliedBy::RepresentativeConfig,
        )?;
        // 以降、失敗してもロールバックしない（適用を試みた事実として消費扱い）。
        let (w_file, w_path) =
            Self::create_lock(&sdir, &key.weights_lock_name(), AppliedBy::ModelWeights)?;
        Self::write_record(cfg_file, &cfg_path, &key.record("config", registry_sha256))?;
        Self::write_record(w_file, &w_path, &key.record("weights", registry_sha256))?;
        Self::sync_dir(&sdir)?;
        Ok(ApplicationTicket {
            config_lock: cfg_path,
        })
    }

    /// 評価の成功（予測と評価後の不変性検証まで完了）を、適用ロックとは別ファイルへ記録する
    /// （`create_new`・読み取り専用。REQ-27）。[`apply_once`] だけが成功後に呼ぶ。
    fn record_completion(
        &self,
        key: &FinalTestKey,
        registry_sha256: &Sha256Digest,
    ) -> Result<(), AcquireError> {
        let sdir = self.scope_dir(&key.eval_data_sha256);
        let (file, path) = Self::create_lock(
            &sdir,
            &key.completion_name(),
            AppliedBy::RepresentativeConfig,
        )?;
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let body = format!(
            "fandhe-edge-final-test-completion v1\nlock=completed\neval_data_sha256={}\nrepresentative_config_id={}\nweights_sha256={}\nregistry_sha256={}\ncompleted_unix_secs={secs}\n",
            key.eval_data_sha256.to_hex(),
            key.config_id.as_str(),
            key.weights_sha256.to_hex(),
            registry_sha256.to_hex(),
        );
        Self::write_record(file, &path, &body)?;
        Self::make_read_only(&path)?;
        Self::sync_dir(&sdir)
    }

    /// 代表構成 `config_id`・重み `weights_sha256` の最終 test 適用が、この台帳で完了しているかを返す
    /// （REQ-27。`package` が評価完了の根拠にする読み取り専用の照会）。
    ///
    /// 事前登録（封印・読み取り専用・既存ロックの登録ダイジェスト）を [`apply_once`] と同じ検査で
    /// 読み、登録された重みが `weights_sha256` と一致し、代表構成ロックと重みロックの両方が
    /// 評価データ・代表構成 ID・重み・登録ダイジェストを正しく記録しているときだけ `Ok(true)`。
    /// 未登録・ロックが無い（または記録が空の消費済み残骸）場合は `Ok(false)`。
    /// 記録が壊れている・食い違う場合は [`AcquireError::RegistryTampered`]（fail-closed）。
    ///
    /// 台帳のファイルへ書き込める主体による丸ごとの偽造は防げない（外部台帳は #168・TASK-39.3-2）。
    ///
    /// # Errors
    /// 台帳の読み取り失敗・改変の検出（[`AcquireError`]）。
    pub fn is_applied(
        &self,
        eval_data_sha256: &Sha256Digest,
        config_id: &RepresentativeConfigId,
        weights_sha256: &Sha256Digest,
    ) -> Result<bool, AcquireError> {
        let (registered, sealed) = match self.load_registry(eval_data_sha256) {
            Ok(v) => v,
            Err(AcquireError::NotRegistered) => return Ok(false),
            Err(e) => return Err(e),
        };
        let Some(entry) = registered.iter().find(|e| &e.id == config_id) else {
            return Ok(false);
        };
        if &entry.weights_sha256 != weights_sha256 {
            return Ok(false);
        }
        let key = FinalTestKey {
            eval_data_sha256: *eval_data_sha256,
            config_id: config_id.clone(),
            weights_sha256: *weights_sha256,
        };
        let sdir = self.scope_dir(eval_data_sha256);
        let tampered = |reason| AcquireError::RegistryTampered { reason };
        for (name, kind) in [
            (key.config_lock_name(), "config"),
            (key.weights_lock_name(), "weights"),
        ] {
            let bytes =
                match fandhe_edge_core::fs::read_bounded(&sdir.join(name), MAX_LOCK_RECORD_BYTES) {
                    Ok(b) => b,
                    Err(FsError::Read { source, .. })
                        if source.kind() == std::io::ErrorKind::NotFound =>
                    {
                        return Ok(false);
                    }
                    Err(_) => return Err(tampered("application lock is unreadable")),
                };
            if bytes.is_empty() {
                return Ok(false);
            }
            let text =
                String::from_utf8(bytes).map_err(|_| tampered("application lock is malformed"))?;
            let field = |k: &str| {
                text.lines()
                    .find_map(|l| l.strip_prefix(k).and_then(|r| r.strip_prefix('=')))
            };
            let matches = field("lock") == Some(kind)
                && field("eval_data_sha256") == Some(eval_data_sha256.to_hex().as_str())
                && field("representative_config_id") == Some(config_id.as_str())
                && field("weights_sha256") == Some(weights_sha256.to_hex().as_str())
                && field("registry_sha256") == Some(sealed.to_hex().as_str());
            if !matches {
                return Err(tampered("application lock does not match the application"));
            }
        }
        // 適用権の消費（ロック）と評価の成功は別の状態。成功記録が無ければ、予測の失敗などで
        // 消費だけが済んだ適用であり、完了とはみなさない（REQ-27）。
        let bytes = match fandhe_edge_core::fs::read_bounded(
            &sdir.join(key.completion_name()),
            MAX_LOCK_RECORD_BYTES,
        ) {
            Ok(b) => b,
            Err(FsError::Read { source, .. }) if source.kind() == std::io::ErrorKind::NotFound => {
                return Ok(false);
            }
            Err(_) => return Err(tampered("completion record is unreadable")),
        };
        if bytes.is_empty() {
            return Ok(false);
        }
        let text =
            String::from_utf8(bytes).map_err(|_| tampered("completion record is malformed"))?;
        let field = |k: &str| {
            text.lines()
                .find_map(|l| l.strip_prefix(k).and_then(|r| r.strip_prefix('=')))
        };
        let matches = field("lock") == Some("completed")
            && field("eval_data_sha256") == Some(eval_data_sha256.to_hex().as_str())
            && field("representative_config_id") == Some(config_id.as_str())
            && field("weights_sha256") == Some(weights_sha256.to_hex().as_str())
            && field("registry_sha256") == Some(sealed.to_hex().as_str());
        if !matches {
            return Err(tampered("completion record does not match the application"));
        }
        Ok(true)
    }
}

/// 評価データの分解失敗を表す理由なしの標識（REQ-27・REQ-39）。
///
/// 分解側のエラー文字列に評価データの本文（問題の行・値）が混ざって公開エラーや
/// ログへ流れないよう、理由を運べない型にしている。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeFailed;

/// 評価データ 1 件を評価器側で `input` と正解に分けた所有値（REQ-27）。
///
/// 分解は呼び出し側の `decode`（CLI の `evaluate` 配線。issue #140）が行い、
/// `gold` は評価器側にのみ残して予測関数へは渡さない。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabeledInput {
    /// 推論関数へ渡す入力。
    pub input: String,
    /// 正解ラベル（評価器側でのみ保持し、推論側へは渡さない）。
    pub gold: String,
}

/// [`apply_once`] の成功値。予測結果と、評価器側に残した正解ラベル（レコード順）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppliedOnce<T> {
    /// 予測クロージャの戻り値（`inputs` と同順であること）。
    pub output: T,
    /// 各レコードの正解ラベル（レコード順。指標計算は評価器側で行う）。
    pub golds: Vec<String>,
}

/// [`apply_once`] の戻り値。外側が評価データ、内側がモデルの不変性検証。
pub type ApplyOnceResult<T, E> =
    Result<AppliedOnce<T>, EvalDataInvarianceError<EvaluationInvarianceError<ApplyOnceError<E>>>>;

/// [`apply_once_then`] の戻り値。成功値は [`AppliedOnce`] と `finish` の戻り値の組。
pub type ApplyOnceThenResult<T, E, R> = Result<
    (AppliedOnce<T>, R),
    EvalDataInvarianceError<EvaluationInvarianceError<ApplyOnceError<E>>>,
>;

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
        let sdir = self.ensure_scope_dir(eval_data_sha256)?;
        for entry in &sorted {
            let path = sdir.join(config_lock_name(eval_data_sha256, &entry.id));
            if fs::symlink_metadata(&path).is_ok() {
                return Err(AcquireError::AlreadyRegistered { path });
            }
        }
        let (file, path) = match Self::create_lock(
            &sdir,
            &registry_name(eval_data_sha256),
            AppliedBy::RepresentativeConfig,
        ) {
            Ok(v) => v,
            Err(AcquireError::AlreadyApplied { lock_path, .. }) => {
                return Err(AcquireError::AlreadyRegistered { path: lock_path });
            }
            Err(e) => return Err(e),
        };
        let body = registry_body(&sorted);
        Self::write_record(file, &path, &body)?;
        Self::make_read_only(&path)?;
        // 登録本文の sha256 を別ファイルへ 1 回だけ封印する。登録ファイルだけを
        // 書き換えても、封印との不一致で適用が拒否される。
        let (seal_file, seal_path) = match Self::create_lock(
            &sdir,
            &seal_name(eval_data_sha256),
            AppliedBy::RepresentativeConfig,
        ) {
            Ok(v) => v,
            Err(AcquireError::AlreadyApplied { lock_path, .. }) => {
                return Err(AcquireError::AlreadyRegistered { path: lock_path });
            }
            Err(e) => return Err(e),
        };
        let seal = format!(
            "{SEAL_HEADER}registry_sha256={}\n",
            registry_digest(body.as_bytes()).to_hex()
        );
        Self::write_record(seal_file, &seal_path, &seal)?;
        Self::make_read_only(&seal_path)?;
        Self::sync_dir(&sdir)
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
/// - `decode` は照合済みの評価データ本体を [`LabeledInput`] の列へ分ける（評価器側の責務）。
///   ロック取得後に呼ばれ、失敗しても適用は消費済み（理由は運ばない。[`DecodeFailed`]）。
/// - `predict` には適用権・各レコードの `input` のみ・`model` を渡す。正解ラベルと評価データ
///   本体は渡さない（REQ-27「推論関数には `input` だけを渡す」）。`predict` は渡された
///   `model` のパスからモデルを読むこと。ロックは `model` の重みに対して消費される。
///   正解ラベルは戻り値の [`AppliedOnce::golds`] として評価器側へ返す。
pub fn apply_once<T, E>(
    ledger: &FinalTestLedger,
    frozen: &FrozenEvalData<'_>,
    config_id: RepresentativeConfigId,
    model: &ModelPackagePaths<'_>,
    decode: impl FnOnce(&[u8]) -> Result<Vec<LabeledInput>, DecodeFailed>,
    predict: impl FnOnce(ApplicationTicket, &[&str], &ModelPackagePaths<'_>) -> Result<T, E>,
) -> ApplyOnceResult<T, E> {
    apply_once_then(
        ledger,
        frozen,
        config_id,
        model,
        decode,
        predict,
        |_| Ok(()),
    )
    .map(|(applied, ())| applied)
}

/// [`apply_once`] に、完了記録の直前に走る `finish` を加えた版（REQ-27）。
///
/// `finish` は予測・不変性検証がすべて成功した後、完了記録の **前** に 1 回だけ呼ばれる
/// （指標の算出・結果の検証・評価記録の書き込みなど、成功の一部である後処理用）。
/// `finish` が失敗すると完了は記録されず（適用権のロックのみが残り、`is_applied` は偽）、
/// 「台帳だけが完了状態で結果が無い」不整合を作らない。失敗は [`ApplyOnceError::Prediction`] として返る。
/// `finish` へ渡すのは予測結果と正解ラベル（評価器側の値）で、推論側には渡らない。
pub fn apply_once_then<T, E, R>(
    ledger: &FinalTestLedger,
    frozen: &FrozenEvalData<'_>,
    config_id: RepresentativeConfigId,
    model: &ModelPackagePaths<'_>,
    decode: impl FnOnce(&[u8]) -> Result<Vec<LabeledInput>, DecodeFailed>,
    predict: impl FnOnce(ApplicationTicket, &[&str], &ModelPackagePaths<'_>) -> Result<T, E>,
    finish: impl FnOnce(&AppliedOnce<T>) -> Result<R, E>,
) -> ApplyOnceThenResult<T, E, R> {
    // 評価が最後まで成功したときに記録する完了記録の材料（適用権の消費とは別の状態。REQ-27）。
    let mut pending_completion: Option<(FinalTestKey, Sha256Digest)> = None;
    let result = evaluate_with_eval_data_invariance(frozen, |bytes| {
        // ここに来た時点で bytes の sha256 == frozen.sha256（照合済み）。
        evaluate_with_invariance(model, |paths| {
            let (registered, registry_sha256) = ledger
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
            // 適用権は評価データ本文を `decode` へ渡す前に消費する。本文を見てから
            // `Err` で抜けて何度でも呼び直す迂回を塞ぐ（REQ-27）。
            let key = FinalTestKey::from_verified(frozen.sha256, config_id, weights_sha256);
            let completion_key = key.clone();
            let ticket = ledger
                .acquire(&key, &registry_sha256)
                .map_err(ApplyOnceError::Acquire)?;
            let records = decode(bytes).map_err(|DecodeFailed| ApplyOnceError::Decode)?;
            let (inputs, golds): (Vec<String>, Vec<String>) =
                records.into_iter().map(|r| (r.input, r.gold)).unzip();
            let input_refs: Vec<&str> = inputs.iter().map(String::as_str).collect();
            let output = predict(ticket, &input_refs, paths).map_err(ApplyOnceError::Prediction)?;
            pending_completion = Some((completion_key, registry_sha256));
            Ok(AppliedOnce { output, golds })
        })
    });
    // 予測・評価後のモデル / 評価データの不変性検証まで成功した場合だけ完了を記録する。
    // 失敗した適用はロックのみが残り、`is_applied` は false を返す。
    let applied = result?;
    let wrap = |e: ApplyOnceError<E>| {
        EvalDataInvarianceError::Evaluation(EvaluationInvarianceError::Evaluation(e))
    };
    let finished = finish(&applied).map_err(|e| wrap(ApplyOnceError::Prediction(e)))?;
    if let Some((key, registry_sha256)) = pending_completion {
        ledger
            .record_completion(&key, &registry_sha256)
            .map_err(|e| wrap(ApplyOnceError::Acquire(e)))?;
    }
    Ok((applied, finished))
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
        let w = key.weights_lock_name();
        assert!(w.starts_with("weights-") && w.ends_with(".lock"));
        assert_ne!(a["config-".len()..], w["weights-".len()..]);
        assert_eq!(a.len(), "config-".len() + 64 + ".lock".len());
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

    /// 列挙上限は対象の評価データのサブディレクトリ内のエントリ総数（許可パターン外を含む）
    /// に掛かり、他の評価データのサブディレクトリは数えない（REQ-27・REQ-39）。
    #[test]
    fn req39_scan_limit_counts_only_this_scope_dir_entries() {
        let dir = std::env::temp_dir().join(format!(
            "fandhe-edge-eval-scan-limit-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir(&dir).unwrap();
        let ledger = FinalTestLedger::open(&dir).unwrap();
        let (mine, other) = (d(1), d(2));
        let sealed = d(9);
        let my_dir = ledger.ensure_scope_dir(&mine).unwrap();
        let other_dir = ledger.ensure_scope_dir(&other).unwrap();
        for i in 0..50 {
            fs::write(other_dir.join(format!("junk-{i}")), "garbage").unwrap();
        }
        for i in 0..2 {
            fs::write(my_dir.join(format!("junk-{i}")), "x").unwrap();
        }
        let check = |limit| FinalTestLedger::check_existing_locks(&my_dir, &mine, &sealed, limit);
        assert!(check(2).is_ok());
        assert!(matches!(
            check(1),
            Err(AcquireError::RegistryTampered {
                reason: "too many ledger entries"
            })
        ));
        let _ = fs::remove_dir_all(&dir);
    }
}
