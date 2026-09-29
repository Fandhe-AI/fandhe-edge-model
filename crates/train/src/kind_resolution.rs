//! `kind` 省略時に既定候補の集合へ解決する層（REQ-19・TASK-19.2 の
//! 2026-09-24 追記。issue #77）。
//!
//! # 役割と呼び出し文脈
//!
//! 利用者がモデルの種類（`kind`）を指定しない場合、判別型の既定候補
//! （C1・C3）を探索候補として組み立てる。組み立てた [`SearchCandidate`] は
//! [`crate::search::run_search`] へそのまま渡せ、validation 正解率による選定
//! （TASK-18.x）が 1 つを選ぶ。`kind` を明示した場合は従来どおり 1 候補に
//! なる。将来の CLI `train`／`select` 工程（TASK-33.1-2・#136 /
//! TASK-33.2-2・#139）が `resolve_kind_candidates` → `run_search` の順に
//! 呼ぶ想定で、CLI 出力へ載せる形は [`KindResolutionRecord`]（直列化可能）。
//!
//! # 設計上の判断
//!
//! - 学習ワーカーには常に解決済みの具体的な `kind` を渡す。学習ワーカーの
//!   契約（`contract.py`）と成果物（`artifact.json`）は変えない。成果物の
//!   `kind` 記録は [`crate::result::TrainOutcome::from_worker_stdout`] が
//!   リクエストとの一致で保証する。
//! - **明示した `kind` には許可リスト検査を掛けない**。未知の `kind` は
//!   学習ワーカーが `unsupported_kind`（TASK-19.3）で拒否する挙動を保つ。
//!   検査するのは同梱の既定候補の集合だけ（fail-closed）。
//! - 共通コアの定義ファイルには `kind` フィールドが無い（現行スキーマでは
//!   すべての定義が「kind を省略した定義」）。追加は定義ファイルのスキーマ・
//!   正準化ハッシュの変更にあたりユーザー承認事項のため、本モジュールでは
//!   行わない。
//!
//! # 暫定値
//!
//! 既定候補の集合（C1・C3）と同率時の優先順（宣言順 c1 → c3）は、
//! **オーナー承認前の暫定値**。`fixtures/train_contract/default_candidates.json`
//! の 1 か所だけで差し替えられる。証拠種別: テストハーネス（PoC-24 実測との
//! 一致確認は人間担当）。

use std::sync::OnceLock;

use serde::Serialize;
use serde_json::{Map, Value};

use crate::error::TrainRequestError;
use crate::kind_defaults;
use crate::request::{Device, TrainRequest, TrainRequestParams};
use crate::search::{MAX_SEARCH_CANDIDATES, SearchCandidate, validate_candidate_id};

/// 既定候補の共有 fixture（Python 側の選択口との一致は
/// `trainer/tests/test_default_candidates_fixture.py` が照合する）。
const DEFAULT_CANDIDATES_JSON: &str =
    include_str!("../../../fixtures/train_contract/default_candidates.json");

/// 利用者が明示した種類。
#[derive(Debug, Clone)]
pub struct ExplicitKind {
    pub kind: String,
    pub kind_version: u32,
    pub config: Map<String, Value>,
}

/// 種類によらない共通の学習パラメータ（[`TrainRequestParams`] から
/// `kind`・`kind_version`・`config` を除いたもの）。
#[derive(Debug, Clone)]
pub struct CommonTrainParams {
    pub label_order: Vec<String>,
    pub max_bytes: u32,
    pub seed: u32,
    pub device: Device,
    pub root: String,
    pub train_path: String,
    pub out_dir: String,
    pub time_limit_seconds: Option<u32>,
    pub rss_limit_bytes: Option<u64>,
}

/// 種類の決まり方。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KindSource {
    /// 利用者が `kind` を明示した。
    Explicit,
    /// `kind` が省略され、既定候補の集合になった。
    Default,
}

/// 解決済みの候補一覧。
#[derive(Debug, Clone)]
pub struct KindResolution {
    source: KindSource,
    candidates: Vec<SearchCandidate>,
}

/// 解決記録の候補 1 件（CLI が JSON に載せる形）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResolvedCandidate {
    pub candidate_id: String,
    pub kind: String,
    pub kind_version: u32,
}

/// 解決記録（「既定候補が選ばれた」ことを出力に残すための直列化可能な形）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct KindResolutionRecord {
    pub kind_source: KindSource,
    pub candidates: Vec<ResolvedCandidate>,
}

impl KindResolution {
    #[must_use]
    pub fn source(&self) -> KindSource {
        self.source
    }

    #[must_use]
    pub fn candidates(&self) -> &[SearchCandidate] {
        &self.candidates
    }

    /// [`crate::search::SearchInput::candidates`] へ渡すために所有権を取り出す。
    #[must_use]
    pub fn into_candidates(self) -> Vec<SearchCandidate> {
        self.candidates
    }

    #[must_use]
    pub fn record(&self) -> KindResolutionRecord {
        KindResolutionRecord {
            kind_source: self.source,
            candidates: self
                .candidates
                .iter()
                .map(|c| ResolvedCandidate {
                    candidate_id: c.candidate_id.clone(),
                    kind: c.params.kind.clone(),
                    kind_version: c.params.kind_version,
                })
                .collect(),
        }
    }
}

/// [`resolve_kind_candidates`] のエラー。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum KindResolutionError {
    /// 同梱の既定候補の集合が壊れている（ビルド成果物の不整合）。
    DefaultCandidatesUnavailable { reason: String },
    /// 組み立てた学習リクエストの構文検査に失敗した。
    InvalidParams(TrainRequestError),
    /// 明示した `kind` が候補 ID として不正（空・`MAX_CANDIDATE_ID_BYTES` 超過・
    /// 制御文字を含む）。`run_search` が同じ規則で拒否するため、解決時点で
    /// 先に `invalid_input` にする。`kind` の内容はメッセージに含めない。
    InvalidKindId,
}

impl std::fmt::Display for KindResolutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DefaultCandidatesUnavailable { reason } => {
                write!(f, "default candidates are unavailable: {reason}")
            }
            Self::InvalidParams(e) => write!(f, "invalid train parameters: {e}"),
            Self::InvalidKindId => write!(f, "kind is not a valid candidate id"),
        }
    }
}

impl std::error::Error for KindResolutionError {}

impl KindResolutionError {
    /// REQ-21 の終了コードへの写像。同梱データの破損は `runtime_error`、
    /// 入力不正は `invalid_input`。
    #[must_use]
    pub fn exit_code(&self) -> fandhe_edge_core::exitcode::ExitCode {
        use fandhe_edge_core::exitcode::ExitCode;
        match self {
            Self::DefaultCandidatesUnavailable { .. } => ExitCode::RuntimeError,
            Self::InvalidParams(TrainRequestError::ConfigTooLarge { .. }) => {
                ExitCode::LimitExceeded
            }
            Self::InvalidParams(_) | Self::InvalidKindId => ExitCode::InvalidInput,
        }
    }

    /// 機械可読な理由コード。
    #[must_use]
    pub fn reason_code(&self) -> &'static str {
        match self {
            Self::DefaultCandidatesUnavailable { .. } => "default_candidates_unavailable",
            Self::InvalidParams(_) => "invalid_request",
            Self::InvalidKindId => "invalid_kind",
        }
    }
}

/// 既定候補の集合を文字列から解析して検査する（本番は同梱 fixture、テストは
/// 壊れた入力を渡す）。空・件数超過・空 kind・重複・`kind_defaults` で
/// 解決できない kind を拒否する。
fn parse_default_candidates(json: &str) -> Result<Vec<(String, u32)>, String> {
    let value: Value = serde_json::from_str(json).map_err(|e| format!("not valid JSON: {e}"))?;
    let array = value
        .get("default_candidates")
        .and_then(Value::as_array)
        .ok_or_else(|| "default_candidates must be an array".to_string())?;
    if array.is_empty() {
        return Err("default_candidates must not be empty".to_string());
    }
    if array.len() > MAX_SEARCH_CANDIDATES {
        return Err("default_candidates exceeds the candidate limit".to_string());
    }
    let mut out: Vec<(String, u32)> = Vec::with_capacity(array.len());
    for item in array {
        let kind = item
            .get("kind")
            .and_then(Value::as_str)
            .ok_or_else(|| "kind must be a string".to_string())?;
        let version = item
            .get("kind_version")
            .and_then(Value::as_u64)
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| "kind_version must be a u32".to_string())?;
        if kind.is_empty() {
            return Err("kind must not be empty".to_string());
        }
        if out.iter().any(|(k, _)| k == kind) {
            return Err("kind must be unique".to_string());
        }
        if kind_defaults::defaults_for_kind(kind).is_none() {
            return Err("kind has no entry in kind_defaults.json".to_string());
        }
        out.push((kind.to_string(), version));
    }
    Ok(out)
}

fn default_candidates() -> Result<&'static [(String, u32)], KindResolutionError> {
    static PARSED: OnceLock<Result<Vec<(String, u32)>, String>> = OnceLock::new();
    match PARSED.get_or_init(|| parse_default_candidates(DEFAULT_CANDIDATES_JSON)) {
        Ok(v) => Ok(v.as_slice()),
        Err(reason) => Err(KindResolutionError::DefaultCandidatesUnavailable {
            reason: reason.clone(),
        }),
    }
}

fn build_params(
    common: &CommonTrainParams,
    kind: String,
    kind_version: u32,
    config: Map<String, Value>,
    out_dir: String,
) -> Result<TrainRequestParams, KindResolutionError> {
    let params = TrainRequestParams {
        kind,
        kind_version,
        config,
        label_order: common.label_order.clone(),
        max_bytes: common.max_bytes,
        seed: common.seed,
        device: common.device,
        root: common.root.clone(),
        train_path: common.train_path.clone(),
        out_dir,
        time_limit_seconds: common.time_limit_seconds,
        rss_limit_bytes: common.rss_limit_bytes,
    };
    // 返す前に構文検査して fail-closed にする（結果の値は使わない）。
    TrainRequest::new(params.clone()).map_err(KindResolutionError::InvalidParams)?;
    Ok(params)
}

/// `kind` の有無から探索候補を組み立てる。
///
/// - `Some`: 1 候補（`candidate_id = kind`。`out_dir` は加工しない）。
///   許可リスト検査はしない（モジュール doc）。
/// - `None`: 既定候補の集合。`config` は空（既定値は学習ワーカー側の
///   `DEFAULT_CONFIG` が補う）。`out_dir` は `{out_dir}/{kind}` とし、
///   `run_search` の `(root, out_dir)` 衝突検出に掛からないよう候補ごとに
///   分ける。`out_dir` は学習リクエスト JSON に載る POSIX 形式の文字列
///   （構文検査は `TrainRequest::new`）なので、`Path::join` ではなく
///   文字列で結合する。
pub fn resolve_kind_candidates(
    explicit: Option<ExplicitKind>,
    common: CommonTrainParams,
) -> Result<KindResolution, KindResolutionError> {
    match explicit {
        Some(e) => {
            if !validate_candidate_id(&e.kind) {
                return Err(KindResolutionError::InvalidKindId);
            }
            let params = build_params(
                &common,
                e.kind.clone(),
                e.kind_version,
                e.config,
                common.out_dir.clone(),
            )?;
            Ok(KindResolution {
                source: KindSource::Explicit,
                candidates: vec![SearchCandidate {
                    candidate_id: e.kind,
                    params,
                }],
            })
        }
        None => {
            let mut candidates = Vec::new();
            for (kind, version) in default_candidates()? {
                let out_dir = format!("{}/{}", common.out_dir, kind);
                let params = build_params(&common, kind.clone(), *version, Map::new(), out_dir)?;
                candidates.push(SearchCandidate {
                    candidate_id: kind.clone(),
                    params,
                });
            }
            Ok(KindResolution {
                source: KindSource::Default,
                candidates,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn common() -> CommonTrainParams {
        CommonTrainParams {
            label_order: vec!["a".to_string(), "b".to_string()],
            max_bytes: 512,
            seed: 1,
            device: Device::Cpu,
            root: "/fandhe-edge-fixture-root".to_string(),
            train_path: "train.jsonl".to_string(),
            out_dir: "out".to_string(),
            time_limit_seconds: None,
            rss_limit_bytes: None,
        }
    }

    /// REQ-19: 同梱の既定候補は c1・c3 の順。
    #[test]
    fn req19_default_candidates_are_c1_then_c3() {
        let parsed = default_candidates().expect("bundled fixture is valid");
        assert_eq!(
            parsed,
            [("c1".to_string(), 1), ("c3".to_string(), 1)].as_slice()
        );
    }

    /// REQ-19・TASK-19.3: 明示 kind は素通し（未知 kind も Ok。拒否は学習ワーカー）。
    #[test]
    fn req19_explicit_kind_passes_through_unchanged() {
        let mut config = Map::new();
        config.insert("epochs".to_string(), Value::from(2));
        for kind in ["c3", "unknown-kind"] {
            let r = resolve_kind_candidates(
                Some(ExplicitKind {
                    kind: kind.to_string(),
                    kind_version: 7,
                    config: config.clone(),
                }),
                common(),
            )
            .expect("explicit kind resolves");
            assert_eq!(r.source(), KindSource::Explicit);
            assert_eq!(r.candidates().len(), 1);
            let c = &r.candidates()[0];
            assert_eq!(c.candidate_id, kind);
            assert_eq!(c.params.kind, kind);
            assert_eq!(c.params.kind_version, 7);
            assert_eq!(c.params.config, config);
            assert_eq!(c.params.out_dir, "out");
        }
    }

    /// REQ-19: kind 省略で既定候補になり、out_dir が候補ごとに分かれる。
    #[test]
    fn req19_omitted_kind_yields_default_candidates() {
        let r = resolve_kind_candidates(None, common()).expect("default resolves");
        assert_eq!(r.source(), KindSource::Default);
        let ids: Vec<&str> = r
            .candidates()
            .iter()
            .map(|c| c.candidate_id.as_str())
            .collect();
        assert_eq!(ids, ["c1", "c3"]);
        let dirs: Vec<&str> = r
            .candidates()
            .iter()
            .map(|c| c.params.out_dir.as_str())
            .collect();
        assert_eq!(dirs, ["out/c1", "out/c3"]);
        assert!(r.candidates().iter().all(|c| c.params.config.is_empty()));
    }

    /// REQ-19: 解決記録の JSON 形。
    #[test]
    fn req19_record_serializes_kind_source_default() {
        let r = resolve_kind_candidates(None, common()).expect("default resolves");
        assert_eq!(
            serde_json::to_value(r.record()).expect("serialize"),
            serde_json::json!({
                "kind_source": "default",
                "candidates": [
                    {"candidate_id": "c1", "kind": "c1", "kind_version": 1},
                    {"candidate_id": "c3", "kind": "c3", "kind_version": 1}
                ]
            })
        );
    }

    /// REQ-19・REQ-39: 既定候補の集合が壊れていれば fail-closed。
    #[test]
    fn req19_parse_default_candidates_rejects_broken_input() {
        let bad = [
            "not json",
            r#"{"default_candidates": []}"#,
            r#"{"default_candidates": [{"kind":"c1","kind_version":1},{"kind":"c1","kind_version":1}]}"#,
            r#"{"default_candidates": [{"kind":"nope","kind_version":1}]}"#,
            r#"{"default_candidates": [{"kind":"_meta","kind_version":1}]}"#,
            r#"{"default_candidates": [{"kind":"","kind_version":1}]}"#,
            r#"{"default_candidates": [{"kind":"c1"}]}"#,
        ];
        for json in bad {
            assert!(parse_default_candidates(json).is_err(), "{json}");
        }
        assert!(
            parse_default_candidates(r#"{"default_candidates":[{"kind":"c3","kind_version":1}]}"#)
                .is_ok()
        );
    }

    /// REQ-19・REQ-39: 候補 ID として不正な明示 kind は解決時に invalid_input。
    #[test]
    fn req19_explicit_kind_invalid_candidate_id_is_rejected() {
        for kind in [
            "x".repeat(crate::search::MAX_CANDIDATE_ID_BYTES + 1),
            "a\nb".to_string(),
        ] {
            let err = resolve_kind_candidates(
                Some(ExplicitKind {
                    kind,
                    kind_version: 1,
                    config: Map::new(),
                }),
                common(),
            )
            .unwrap_err();
            assert_eq!(err.reason_code(), "invalid_kind");
            assert_eq!(
                err.exit_code(),
                fandhe_edge_core::exitcode::ExitCode::InvalidInput
            );
        }
    }

    /// REQ-21: config 過大は limit_exceeded（20）。
    #[test]
    fn req21_config_too_large_maps_to_limit_exceeded() {
        let err =
            KindResolutionError::InvalidParams(TrainRequestError::ConfigTooLarge { limit: 1 });
        assert_eq!(
            err.exit_code(),
            fandhe_edge_core::exitcode::ExitCode::LimitExceeded
        );
    }

    /// REQ-21: 不正な共通パラメータは invalid_input（64）。
    #[test]
    fn req21_invalid_params_map_to_invalid_input() {
        let mut c = common();
        c.train_path = String::new();
        let err = resolve_kind_candidates(None, c).unwrap_err();
        assert_eq!(err.reason_code(), "invalid_request");
        assert_eq!(
            err.exit_code(),
            fandhe_edge_core::exitcode::ExitCode::InvalidInput
        );
    }
}
