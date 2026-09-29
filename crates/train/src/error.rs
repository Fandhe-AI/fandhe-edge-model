//! 学習リクエスト・結果 JSON の検証エラー（REQ-21・REQ-39）。
//!
//! [`crate::request::TrainRequest`]・[`crate::result::TrainOutcome`] の組み
//! 立て・解析経路で発生するエラーを表す。外部入力（学習リクエスト・ワーカー
//! 出力）の経路のため `panic`・`unwrap` はせず、すべて `Result` で返す
//! （`.claude/rules/coding-rust.md`）。メッセージは英語で、ラベル本文・
//! `config` の値・パス文字列などデータ本文を含めない
//! （`.claude/rules/security.md`）。

use fandhe_edge_core::exitcode::ExitCode;

/// 学習リクエストの組み立て（[`crate::request::TrainRequest::new`]・
/// [`crate::request::TrainRequest::from_json_slice`]）で発生するエラー。
///
/// `reason_code()` は学習ワーカー（`contract.py`・`guard.py`）が同じ入力に
/// 対して返す `code` と揃える（`invalid_path`・`limit_exceeded`・
/// `invalid_request`）。パス以外の検証は Python 側の型検査・範囲検査より
/// 先に構文だけを見るため、Python が `invalid_path` を返すケースと Rust が
/// `invalid_request` を返すケースが一致しないことがある（本 crate はそれを
/// 「Rust 側の方が厳しい」区別として単体テストで確認し、共有 fixture には
/// 両者が一致するケースだけを載せる。issue #177 実装計画）。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TrainRequestError {
    /// リクエストの生バイト列が [`crate::limits::MAX_REQUEST_BYTES`] を超える
    /// （解析前・直列化後のいずれの検査でも到達しうる）。
    TooLarge { size: usize, limit: usize },
    /// `config` を直列化した長さが [`crate::limits::MAX_REQUEST_BYTES`] を超える。
    /// `TrainRequest::new` と探索の事前検証（`crate::search`）で、保持・複製の前に
    /// 検出する（REQ-39・issue #255）。`config` の内容は含めない。
    ConfigTooLarge { limit: usize },
    /// UTF-8 として読めない。
    NotUtf8,
    /// JSON として解析できない（構文エラー・重複キー・未知フィールド・型
    /// 不一致等）。`serde_json::Error` の位置情報（`line`／`column`）と
    /// `classify()` の分類のみを保持し、エラー文言（`err.to_string()`）は
    /// 保持しない。エラー文言には解析対象のキー名・値がそのまま埋め込まれ
    /// うるため（例: 未知フィールド名。`config` は利用者が任意のキーを
    /// 指定できる）、位置情報だけを渡すことでデータ本文の転記を防ぐ
    /// （`.claude/rules/security.md`「秘密情報の混入防止」。PR #220
    /// レビュー指摘 P0）。
    NotJson {
        line: usize,
        column: usize,
        category: serde_json::error::Category,
    },
    /// トップレベルが JSON オブジェクトでない。
    NotObject,
    /// `schema_version` が存在しない・整数でない・`1` 以外。
    UnsupportedSchemaVersion,
    /// 未知のフィールドを含む。
    UnknownField { field: &'static str },
    /// `kind` が空文字列、または存在しない。
    EmptyKind,
    /// `kind_version` が非負整数として読めない。
    InvalidKindVersion,
    /// `config` が JSON オブジェクトでない。
    ConfigNotObject,
    /// `label_order` が範囲外の件数。
    LabelOrderCount { actual: usize },
    /// `label_order[index]` が空文字列。
    LabelOrderEmptyLabel { index: usize },
    /// `label_order[index]` が [`crate::limits::MAX_LABEL_BYTES`] を超える。
    LabelOrderLabelTooLong { index: usize },
    /// `label_order` 内に重複するラベルがある。
    LabelOrderDuplicateLabel { index: usize },
    /// `label_order` がリストでない。
    LabelOrderNotArray,
    /// `max_bytes` が整数として読めない、または範囲外。
    InvalidMaxBytes,
    /// `seed` が整数として読めない、または範囲外。
    InvalidSeed,
    /// `device` が `"cpu"`／`"gpu"` のいずれでもない。
    InvalidDevice,
    /// `time_limit_seconds` が範囲外（指定された場合のみ検査）。
    InvalidTimeLimitSeconds,
    /// `rss_limit_bytes` が範囲外（指定された場合のみ検査）。
    InvalidRssLimitBytes,
    /// `root`・`train_path`・`out_dir` の経路の構文が不正
    /// （空・NUL・絶対 / 相対の取り違え・`..` 構成要素・`.` のみ等）。
    InvalidPath { field: &'static str },
    /// `serde_json` による直列化に失敗した（通常到達しない防御的な分岐）。
    SerializeFailed,
    /// `validation_inputs` が空配列（付けるなら 1 件以上）。
    ValidationInputsEmpty,
    /// `validation_inputs[index].id` が空、または
    /// `fandhe_edge_core::judgment::MAX_INPUT_ID_BYTES` を超える。
    ValidationInputInvalidId { index: usize },
    /// `validation_inputs[index].input` が
    /// `fandhe_edge_core::infer_input::MAX_INFER_INPUT_BYTES` を超える。
    ValidationInputTooLarge { index: usize },
    /// `validation_inputs` の `id`＋`input` の合計が
    /// [`crate::limits::MAX_VALIDATION_INPUT_TOTAL_BYTES`] を超える。
    ValidationInputsTotalBytesExceeded,
    /// `validation_inputs[index].id` が先行する要素と重複する。
    ValidationInputDuplicateId { index: usize },
    /// `validation_inputs` の結果 JSON の最大長（許可するラベル・id から計算。
    /// [`crate::request::validation_result_bytes_bound`]）が
    /// [`crate::limits::MAX_RESULT_BYTES_WITH_VALIDATION`] を超える。学習を始める
    /// 前に `limit_exceeded` で拒否する。
    ValidationResultTooLarge,
}

impl std::fmt::Display for TrainRequestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrainRequestError::TooLarge { size, limit } => {
                write!(f, "train request exceeds {limit} bytes limit (got {size})")
            }
            TrainRequestError::ConfigTooLarge { limit } => {
                write!(f, "train request config exceeds {limit} bytes limit")
            }
            TrainRequestError::NotUtf8 => write!(f, "train request is not valid utf-8"),
            TrainRequestError::NotJson {
                line,
                column,
                category,
            } => {
                write!(
                    f,
                    "train request is not valid JSON: {category:?} error at line {line}, column {column}"
                )
            }
            TrainRequestError::NotObject => write!(f, "train request must be a JSON object"),
            TrainRequestError::UnsupportedSchemaVersion => {
                write!(
                    f,
                    "train request schema_version must be integer {}",
                    crate::limits::REQUEST_SCHEMA_VERSION
                )
            }
            TrainRequestError::UnknownField { field } => {
                write!(f, "train request has an unknown field: {field}")
            }
            TrainRequestError::EmptyKind => write!(f, "kind must be a non-empty string"),
            TrainRequestError::InvalidKindVersion => {
                write!(f, "kind_version must be a non-negative integer")
            }
            TrainRequestError::ConfigNotObject => write!(f, "config must be a JSON object"),
            TrainRequestError::LabelOrderCount { actual } => write!(
                f,
                "label_order length out of range [{}, {}] (got {actual})",
                crate::limits::MIN_LABELS,
                crate::limits::MAX_LABELS
            ),
            TrainRequestError::LabelOrderEmptyLabel { index } => {
                write!(f, "label_order[{index}] must be a non-empty string")
            }
            TrainRequestError::LabelOrderLabelTooLong { index } => write!(
                f,
                "label_order[{index}] exceeds {} utf-8 bytes",
                crate::limits::MAX_LABEL_BYTES
            ),
            TrainRequestError::LabelOrderDuplicateLabel { index } => {
                write!(f, "label_order[{index}] is a duplicate")
            }
            TrainRequestError::LabelOrderNotArray => write!(f, "label_order must be an array"),
            TrainRequestError::InvalidMaxBytes => write!(
                f,
                "max_bytes must be an integer in [{}, {}]",
                crate::limits::MIN_MAX_BYTES,
                crate::limits::MAX_MAX_BYTES
            ),
            TrainRequestError::InvalidSeed => write!(
                f,
                "seed must be an integer in [{}, {}]",
                crate::limits::MIN_SEED,
                crate::limits::MAX_SEED
            ),
            TrainRequestError::InvalidDevice => write!(
                f,
                "device must be one of {:?}",
                crate::limits::ALLOWED_DEVICES
            ),
            TrainRequestError::InvalidTimeLimitSeconds => write!(
                f,
                "time_limit_seconds must be an integer in [1, {}]",
                crate::limits::MAX_TRAIN_WALL_SECONDS
            ),
            TrainRequestError::InvalidRssLimitBytes => write!(
                f,
                "rss_limit_bytes must be an integer in [1, {}]",
                crate::limits::MAX_TRAIN_RSS_BYTES
            ),
            TrainRequestError::InvalidPath { field } => {
                write!(f, "{field} has an invalid path")
            }
            TrainRequestError::SerializeFailed => {
                write!(f, "failed to serialize train request")
            }
            TrainRequestError::ValidationInputsEmpty => {
                write!(f, "validation_inputs must not be empty")
            }
            TrainRequestError::ValidationInputInvalidId { index } => {
                write!(f, "validation_inputs[{index}].id is empty or too long")
            }
            TrainRequestError::ValidationInputTooLarge { index } => {
                write!(f, "validation_inputs[{index}].input is too large")
            }
            TrainRequestError::ValidationInputsTotalBytesExceeded => {
                write!(f, "validation_inputs total bytes exceed the limit")
            }
            TrainRequestError::ValidationInputDuplicateId { index } => {
                write!(f, "validation_inputs[{index}].id is a duplicate")
            }
            TrainRequestError::ValidationResultTooLarge => write!(
                f,
                "the result for validation_inputs could exceed {} bytes",
                crate::limits::MAX_RESULT_BYTES_WITH_VALIDATION
            ),
        }
    }
}

impl std::error::Error for TrainRequestError {}

impl TrainRequestError {
    /// 学習ワーカー（`contract.py`・`guard.py`）と同じ語彙の機械可読コード。
    #[must_use]
    pub const fn reason_code(&self) -> &'static str {
        match self {
            TrainRequestError::InvalidPath { .. } => "invalid_path",
            TrainRequestError::TooLarge { .. }
            | TrainRequestError::ConfigTooLarge { .. }
            | TrainRequestError::ValidationResultTooLarge => "limit_exceeded",
            TrainRequestError::NotUtf8
            | TrainRequestError::NotJson { .. }
            | TrainRequestError::NotObject
            | TrainRequestError::UnsupportedSchemaVersion
            | TrainRequestError::UnknownField { .. }
            | TrainRequestError::EmptyKind
            | TrainRequestError::InvalidKindVersion
            | TrainRequestError::ConfigNotObject
            | TrainRequestError::LabelOrderCount { .. }
            | TrainRequestError::LabelOrderEmptyLabel { .. }
            | TrainRequestError::LabelOrderLabelTooLong { .. }
            | TrainRequestError::LabelOrderDuplicateLabel { .. }
            | TrainRequestError::LabelOrderNotArray
            | TrainRequestError::InvalidMaxBytes
            | TrainRequestError::InvalidSeed
            | TrainRequestError::InvalidDevice
            | TrainRequestError::InvalidTimeLimitSeconds
            | TrainRequestError::InvalidRssLimitBytes
            | TrainRequestError::ValidationInputsEmpty
            | TrainRequestError::ValidationInputInvalidId { .. }
            | TrainRequestError::ValidationInputTooLarge { .. }
            | TrainRequestError::ValidationInputsTotalBytesExceeded
            | TrainRequestError::ValidationInputDuplicateId { .. }
            | TrainRequestError::SerializeFailed => "invalid_request",
        }
    }

    /// REQ-21 の終了コードへの対応づけ。`TooLarge`・`ConfigTooLarge`・`ValidationResultTooLarge` のみ `LimitExceeded`（20）、
    /// それ以外はすべて `InvalidInput`（64）。
    #[must_use]
    pub const fn exit_code(&self) -> ExitCode {
        match self {
            TrainRequestError::TooLarge { .. }
            | TrainRequestError::ConfigTooLarge { .. }
            | TrainRequestError::ValidationResultTooLarge => ExitCode::LimitExceeded,
            _ => ExitCode::InvalidInput,
        }
    }
}

/// 学習ワーカーの標準出力（結果 JSON）の解析（[`crate::result::TrainOutcome`]）
/// で発生するエラー。ワーカー出力は信頼しない外部入力として扱う（fail-closed。
/// `.claude/rules/security.md`）。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TrainResultError {
    /// 標準出力が [`crate::limits::MAX_RESULT_BYTES`] を超える。
    TooLarge { size: usize, limit: usize },
    /// UTF-8 として読めない。
    NotUtf8,
    /// 空行を除いた行数がちょうど 1 行でない（0 行または 2 行以上）。
    NotExactlyOneLine { lines: usize },
    /// JSON として解析できない（構文エラー・重複キー・未知フィールド・型
    /// 不一致等）。`TrainRequestError::NotJson` と同じ理由で、位置情報
    /// （`line`／`column`）と `classify()` の分類のみを保持し、エラー文言
    /// は保持しない（`.claude/rules/security.md`。PR #220 レビュー指摘 P0）。
    NotJson {
        line: usize,
        column: usize,
        category: serde_json::error::Category,
    },
    /// トップレベルが JSON オブジェクトでない。
    NotObject,
    /// 未知のフィールドを含む。
    UnknownField { field: &'static str },
    /// `status` が `"ok"`／`"error"` のいずれでもない、または各 status に
    /// 必要なフィールドの組み合わせを満たさない。
    MalformedOutcome,
    /// `artifact` 内のフィールドが不正（型・範囲・許可値のいずれか）。
    MalformedArtifact { field: &'static str },
    /// `code`（[`crate::result::WorkerFailure::code`]）が
    /// [`crate::result::FailureCode`] の許可リストに一致しない（REQ-21・
    /// REQ-39。未知のコードを検証済みとして受理しない。PR #220 レビュー
    /// 指摘 P1）。
    InvalidFailureCode,
    /// `message` が上限（4 KiB）を超える。
    FailureMessageTooLong,
    /// 成果物の `kind`／`kind_version`／`label_order`／`max_bytes`／
    /// `candidate_label` が、この結果に対応する
    /// [`crate::request::TrainRequest`] の値と完全一致しない（REQ-39 ガード層
    /// 「完全性と版」・REQ-21 入出力契約。ワーカーが依頼と異なる種類・
    /// 未対応の版・別種類を名乗る成果物を返しても成功扱いにしないための検査。
    /// `kind` ごとの許可版一覧は本 crate の対象外〔未確定。PR #220 参照〕
    /// のため、ここでは「依頼内容と一致するか」だけを検査する）。`config`
    /// は「defaults(kind) に `request.config` を上書きした実効 config」との
    /// 完全一致で検査する（[`crate::kind_defaults::effective_config`] 参照。
    /// codex 指摘 PR #220 P1「成果物の追加 config 値を検証せず成功扱いに
    /// している」。旧・部分一致の `config_matches_explicit_keys` は撤去した）。
    ArtifactMismatch { field: &'static str },
    /// `kind` が [`crate::kind_defaults`] の共有 fixture
    /// （`fixtures/train_contract/kind_defaults.json`）に登録されていない
    /// ため、実効 config を算出できない（REQ-19・REQ-21・REQ-39・P1。
    /// fixture に無い種類の成果物は `config` を検証できないため成功扱いに
    /// しない。fail-closed）。
    UnsupportedKindForConfigDefaults,
}

impl std::fmt::Display for TrainResultError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrainResultError::TooLarge { size, limit } => {
                write!(f, "train result exceeds {limit} bytes limit (got {size})")
            }
            TrainResultError::NotUtf8 => write!(f, "train result is not valid utf-8"),
            TrainResultError::NotExactlyOneLine { lines } => write!(
                f,
                "train result must be exactly one JSON line (got {lines})"
            ),
            TrainResultError::NotJson {
                line,
                column,
                category,
            } => {
                write!(
                    f,
                    "train result is not valid JSON: {category:?} error at line {line}, column {column}"
                )
            }
            TrainResultError::NotObject => write!(f, "train result must be a JSON object"),
            TrainResultError::UnknownField { field } => {
                write!(f, "train result has an unknown field: {field}")
            }
            TrainResultError::MalformedOutcome => {
                write!(f, "train result has a malformed status/field combination")
            }
            TrainResultError::MalformedArtifact { field } => {
                write!(f, "train result artifact field is malformed: {field}")
            }
            TrainResultError::InvalidFailureCode => {
                write!(f, "train result code is not a known failure code")
            }
            TrainResultError::FailureMessageTooLong => {
                write!(f, "train result message exceeds size limit")
            }
            TrainResultError::ArtifactMismatch { field } => {
                write!(
                    f,
                    "train result artifact field does not match the request: {field}"
                )
            }
            TrainResultError::UnsupportedKindForConfigDefaults => {
                write!(f, "train result kind has no known config defaults")
            }
        }
    }
}

impl std::error::Error for TrainResultError {}

impl TrainResultError {
    /// 壊れたワーカー出力はすべて `runtime_error`（supervisor.py が
    /// 妥当な JSON 1 個・既知の終了コードでない出力を扱う場合と同じ扱い）。
    #[must_use]
    pub const fn reason_code(&self) -> &'static str {
        "runtime_error"
    }

    /// REQ-21 の終了コード。壊れたワーカー出力は常に `RuntimeError`（70）。
    #[must_use]
    pub const fn exit_code(&self) -> ExitCode {
        ExitCode::RuntimeError
    }
}

/// 学習ワーカーの子プロセス起動・監視・終了コード写像（REQ-21・REQ-34・
/// REQ-39・#178）で発生するエラー。[`crate::process::run_train`] が返す。
///
/// 呼び出し元・呼び出し先の文脈: Rust 側 CLI／ジョブ管理（未配線。TASK-33.x）
/// が学習ワーカー（`trainer/launch.py`）を子プロセスとして起動する経路の
/// エラーを表す。外部入力（子プロセスの終了コード・標準出力・OS のエラー）
/// の経路のため `panic`・`unwrap` はせず、すべて `Result` で返す
/// （`.claude/rules/coding-rust.md`）。
///
/// `Display` は英語で、フィールド名・数値・`io::ErrorKind` のみを含める。
/// ワーカーの標準出力・標準エラー出力の中身、リクエスト・ジョブディレクト
/// リのパス文字列は含めない（`.claude/rules/security.md`「秘密情報の混入
/// 防止」。学習・評価データ本文が含まれうる値をエラー経由で漏らさない）。
#[derive(Debug)]
#[non_exhaustive]
pub enum TrainProcessError {
    /// [`crate::process::WorkerLauncher::new`]／`from_trainer_dir` の検証
    /// 失敗（相対パス・`launch.py` 以外のファイル名・存在しない・通常
    /// ファイルでない等）。
    InvalidLauncher { field: &'static str },
    /// 呼び出し元が渡した `job_dir` が絶対パスのディレクトリでない。
    InvalidJobDir,
    /// `job_dir/request.json` の作成・書き込みに失敗した（既存ファイルの
    /// 上書き拒否〔`AlreadyExists`〕を含む）。
    RequestWrite { kind: std::io::ErrorKind },
    /// 子プロセスの起動（`Command::spawn`）に失敗した。
    Spawn { kind: std::io::ErrorKind },
    /// 子プロセスの終了待ち（`Child::try_wait`／`Child::wait`）に失敗した。
    /// Rust が扱うのは直接の子（supervisor）だけであり、子孫プロセスの
    /// 掃除は学習ワーカー側の lifeline に委ねる設計（issue #178 PR #233
    /// レビュー。`crates/train/src/process.rs` モジュール doc 参照）のため、
    /// 子孫の掃除確認フィールドは持たない。
    Wait { kind: std::io::ErrorKind },
    /// 外側の壁時計締め切り（[`crate::process::RunLimits`]）を超過した、
    /// または `try_wait()` の観測が締め切り後になった経路（`WaitOutcome::
    /// LateExit`。再利用されうる pid への誤った `kill()` を避けるため）
    /// で、子プロセスを強制終了した（REQ-39「資源の上限」）。
    ///
    /// `child_reaped` は、締め切り超過を検出して `Child::kill()` を送った
    /// 後、直接の子（supervisor）の終了を
    /// [`crate::process::ensure_reaped`] で回収できたかを示す
    /// （`LateExit` 経路では既に reap 済みのため常に `true`）。`false` は
    /// 回収自体
    /// （`KillWaitTimedOut`／`Wait`）が失敗したことを意味するが、その場合
    /// でも本バリアント（`exit_code()` は常に `LimitExceeded`）を返す。
    /// 回収の失敗を理由に `WallTimeout` の代わりに回収エラーを返すと、
    /// 壁時計タイムアウトが `LimitExceeded`（20）ではなく `RuntimeError`
    /// （70）に化けてしまい REQ-21 の終了コード契約に違反する（codex/review
    /// 指摘 P1「wall timeout 超過後 wait_after_kill(...)? 失敗で WallTimeout
    /// が KillWaitTimedOut／Wait に化ける」。issue #178 PR #233 レビュー。
    /// Cursor Bugbot 指摘 Medium も同一事象）。`child_reaped: false` は
    /// プロセスが OS 上にゾンビとして残り続ける可能性があることを呼び出し元
    /// へ伝える診断情報として使う。
    WallTimeout { limit_ms: u64, child_reaped: bool },
    /// `SIGKILL` 送出後の直接の子プロセスの終了待ちが
    /// [`crate::process::KILL_WAIT_TIMEOUT`] 以内に完了しなかった
    /// （割り込み不可能な OS 側の待ち〔D state〕等、極めて稀なケース。
    /// Cursor Bugbot 指摘 Medium「Timeout wait can block forever」。issue
    /// #178 PR #233 レビュー）。無期限に `Child::wait()` を待ち続けると
    /// 呼び出しスレッド自体が資源の上限なくブロックしてしまうため、上限で
    /// 打ち切って呼び出し元へ制御を返す（プロセス自体は OS 上にゾンビ
    /// として残り続ける可能性があり、確実な後始末を主張しない。fail-closed。
    /// REQ-39「資源の上限」）。
    KillWaitTimedOut,
    /// 子プロセスがシグナルで終了し、終了コードを取得できなかった
    /// （unix。`ExitStatus::code()` が `None` を返す場合）。
    TerminatedBySignal,
    /// 子プロセスの終了コードが REQ-21 の 7 種のいずれにも一致しない。
    UnknownExitCode(i32),
    /// 子プロセスの終了後、期限内に標準出力の読み取りが完了しなかった
    /// （孤児プロセスがパイプを閉じずに残っている等。REQ-39）。
    StdoutIncomplete,
    /// 子プロセスの終了後、期限内に標準エラー出力の読み取りが完了しなかった
    /// （`StdoutIncomplete` と同じ原因〔孤児プロセスがパイプを保持〕で
    /// 起こりうる。標準出力側だけを検査し標準エラー側のタイムアウトを
    /// `stderr_truncated: true` のまま成功扱いにすると、学習が継続中でも
    /// 成功として返してしまう〔codex/review 指摘。issue #178 PR #233
    /// レビュー〕ため、読み取り未完了は標準出力と同様にエラーとして扱う。
    /// REQ-39「資源の上限」）。
    StderrIncomplete,
    /// [`crate::process::RunLimits::with_wall_timeout`] に渡した値が不正
    /// （0、または既定の締め切り以上。「締める方向だけ」の制約に違反）。
    /// `WorkerLauncher` の検証失敗（[`TrainProcessError::InvalidLauncher`]）
    /// とは別の種類のため、独立したバリアントとして区別する。
    InvalidRunLimits,
    /// [`crate::request::TrainRequest::to_json_vec`] が直列化上限超過
    /// （`TrainRequestError::TooLarge`）を返した（`config` は
    /// `TrainRequest::new` の時点ではサイズ検査をしないため、巨大な
    /// `config` を持つリクエストで到達しうる。REQ-39「資源の上限」）。
    /// [`TrainRequestError::exit_code`] へそのまま委譲する（`LimitExceeded`
    /// になりうる。「通常到達しない」と誤って丸めない）。
    Request(TrainRequestError),
    /// 標準出力の内容が [`crate::result::TrainOutcome::from_worker_stdout`]
    /// の検証を通らなかった（壊れたワーカー出力。fail-closed）。
    Result(TrainResultError),
    /// 子プロセスの実終了コードと、標準出力の JSON が示す結果
    /// （[`crate::result::TrainOutcome::exit_code`]）が一致しない
    /// （REQ-39「完全性と版」・fail-closed）。
    ExitCodeMismatch {
        process: ExitCode,
        expected: ExitCode,
    },
    /// unix 以外（windows 等）の環境で [`crate::process::run_train`] が呼ば
    /// れた。子孫プロセス（`_worker` を含む）の確実な掃除は学習ワーカー側の
    /// lifeline（`_worker` が親の死を検知して自己終了する。
    /// `trainer/src/fandhe_edge_trainer/supervisor.py` モジュール docstring
    /// 参照）に委ねる設計へ移行したが、windows は本 issue の対象外のまま
    /// とし（未検証。将来対応時に改めて検討する）、
    /// [`crate::process::run_train`]（`cfg(not(unix))` 版）は子プロセスを
    /// 一切起動せず即座に本バリアントを返す（fail-closed）。
    UnsupportedPlatform,
}

impl std::fmt::Display for TrainProcessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrainProcessError::InvalidLauncher { field } => {
                write!(f, "invalid worker launcher field: {field}")
            }
            TrainProcessError::InvalidJobDir => {
                write!(f, "job_dir must be an existing absolute directory")
            }
            TrainProcessError::RequestWrite { kind } => {
                write!(f, "failed to write request file: {kind:?}")
            }
            TrainProcessError::Spawn { kind } => {
                write!(f, "failed to spawn worker process: {kind:?}")
            }
            TrainProcessError::Wait { kind } => {
                write!(f, "failed to wait for worker process: {kind:?}")
            }
            TrainProcessError::WallTimeout {
                limit_ms,
                child_reaped,
            } => {
                write!(
                    f,
                    "worker process exceeded wall timeout of {limit_ms} ms (child_reaped={child_reaped})"
                )
            }
            TrainProcessError::KillWaitTimedOut => {
                write!(
                    f,
                    "worker process did not exit within the bounded wait after SIGKILL"
                )
            }
            TrainProcessError::TerminatedBySignal => {
                write!(f, "worker process was terminated by a signal")
            }
            TrainProcessError::UnknownExitCode(code) => {
                write!(f, "worker process exited with unknown exit code {code}")
            }
            TrainProcessError::StdoutIncomplete => {
                write!(f, "worker process stdout was not fully read in time")
            }
            TrainProcessError::StderrIncomplete => {
                write!(f, "worker process stderr was not fully read in time")
            }
            TrainProcessError::InvalidRunLimits => {
                write!(f, "run limits must only be tightened, never loosened")
            }
            TrainProcessError::Request(inner) => {
                write!(f, "failed to serialize train request: {inner}")
            }
            TrainProcessError::Result(inner) => {
                write!(f, "worker result is invalid: {inner}")
            }
            TrainProcessError::ExitCodeMismatch { process, expected } => {
                write!(
                    f,
                    "worker process exit code {process:?} does not match result-derived exit code {expected:?}"
                )
            }
            TrainProcessError::UnsupportedPlatform => {
                write!(
                    f,
                    "worker process resource limits are not supported on this platform"
                )
            }
        }
    }
}

impl std::error::Error for TrainProcessError {}

impl From<TrainResultError> for TrainProcessError {
    fn from(value: TrainResultError) -> Self {
        TrainProcessError::Result(value)
    }
}

impl From<TrainRequestError> for TrainProcessError {
    fn from(value: TrainRequestError) -> Self {
        TrainProcessError::Request(value)
    }
}

impl TrainProcessError {
    /// REQ-21 の終了コードへの対応づけ。子プロセスの実終了コードを尊重する
    /// のは [`TrainProcessError::ExitCodeMismatch`] 以外に存在しない
    /// （`run_train` が実際の子プロセス終了コードと結果 JSON の一致を
    /// 検証済みの場合のみ `Ok(TrainRun)` を返すため）。
    #[must_use]
    pub const fn exit_code(&self) -> ExitCode {
        match self {
            TrainProcessError::InvalidLauncher { .. }
            | TrainProcessError::InvalidJobDir
            | TrainProcessError::InvalidRunLimits => ExitCode::InvalidInput,
            TrainProcessError::RequestWrite { .. }
            | TrainProcessError::Spawn { .. }
            | TrainProcessError::Wait { .. }
            | TrainProcessError::KillWaitTimedOut
            | TrainProcessError::TerminatedBySignal
            | TrainProcessError::UnknownExitCode(_)
            | TrainProcessError::StdoutIncomplete
            | TrainProcessError::StderrIncomplete
            | TrainProcessError::ExitCodeMismatch { .. }
            | TrainProcessError::UnsupportedPlatform => ExitCode::RuntimeError,
            TrainProcessError::WallTimeout { .. } => ExitCode::LimitExceeded,
            TrainProcessError::Request(inner) => inner.exit_code(),
            TrainProcessError::Result(inner) => inner.exit_code(),
        }
    }

    /// 学習ワーカーと同じ語彙の機械可読コード（[`ExitCode::name`] と同じ
    /// 語彙）。
    #[must_use]
    pub const fn reason_code(&self) -> &'static str {
        self.exit_code().name()
    }
}
