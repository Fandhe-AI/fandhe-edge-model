//! 学習依存の無い ONNX 推論バックエンド（C1・C3。REQ-32・REQ-28・REQ-39・TASK-32.1-2・#113）。
//!
//! # 役割
//!
//! 学習ワーカー（`trainer/`）が書き出した C1（バイト n-gram TF-IDF + ロジスティック回帰）・
//! C3（バイト CNN）の ONNX を読み込み、[`crate::pipeline::ScoringBackend`] として推論経路へ
//! 差し込む。呼び出し元（想定）は CLI の `infer` 工程（TASK-33.x）と評価器の推論関数
//! （TASK-27.2 経由。[`crate::pipeline::InferencePipeline::as_predict_fn`]）。学習側（Python・
//! MLX・学習用 crate）には依存しない（REQ-32）。
//!
//! # 自作ランタイムを選んだ理由（依存の承認事項）
//!
//! TASK-32.1 は推論ランタイムに「自作・ONNX Runtime・tract」のいずれかを許す。`ort`（PoC-16 は
//! `download-binaries` でプリビルド ONNX Runtime を取得し REQ-38 と衝突しうる）・`tract-onnx`・
//! `prost` はいずれもユーザー未承認のため、本実装は std と承認済み依存だけで動く自作にした。
//! `ort` の採用は依存追加の承認事項であり、承認された場合の差し替え口は `ScoringBackend`。
//!
//! # 汎用インタプリタではなく「テンプレート照合＋種類別の実行器」
//!
//! 書き出し器（`kinds/c1.py::_export_c1_onnx`・`kinds/c3.py::_export_c3_onnx`）と同じリポ内で
//! グラフは固定の形のため、汎用の演算子実装は作らない。読み込み時にノード列（op_type・入出力名の
//! 順序）・属性・initializer の名前と形状を書き出し器のテンプレートと**完全一致**で照合し、
//! 一致しなければ fail-closed で拒否する（形式の許可制。REQ-39）。照合を通ったモデルだけを
//! 手書きの f32 順伝播で実行する。書き出し器を変えてテンプレートがずれた場合は
//! `tests/onnx_parity.rs` が失敗する（コードを直す。テストを緩めない）。
//!
//! # 数値
//!
//! `Gemm`・`Conv`・L2 ノルムは ONNX の型どおり f32 で計算する。**最後の Softmax だけは f32 の
//! ロジットから f64 で計算する**。f32 の softmax を f64 へ変換すると、クラス数が多い場合
//! （上限 `MAX_OPTIONS`=1024）に確率の合計が 1 から [`SCORE_SUM_TOLERANCE`]
//! （[`crate::pipeline`] の検査）を超えてずれうるため。argmax の結果はロジットの argmax と同じで
//! 変わらない。許容差を広げて通すことはしない。
//!
//! # 契約
//!
//! - `artifact.json` は本層で読まない（解釈は packaging・CLI〔TASK-33.x〕の責務）。`kind`・
//!   `max_bytes`・モデルファイルの経路・期待する sha256 は呼び出し側が渡す。ラベル ID への写像も
//!   呼び出し側（[`crate::pipeline::Prediction::label_index`]）
//! - [`OnnxBackend`] は `&self` の純粋計算で内部可変性を持たない（呼び出し間で結果に影響する
//!   状態を持たない。REQ-28）。`ScoringBackend` の「`&mut` を要求する場合の扱い」は不要になった
//! - エラーの 7 種終了コードへの写像は CLI 側の責務。想定は形式・内容の不正が
//!   `invalid_input`（64）、上限超過が `limit_exceeded`（20）、ファイル I/O 失敗が
//!   `runtime_error`（70）
//! - 対応するのは C1・C3 のみ。`autoregressive`（REQ-19b）等の ONNX 読み込みは後続 TASK
//!   （実装済みを装わない。[`ModelKind::parse`] は `unsupported_kind` で拒否する）

mod c1;
mod c3;
mod proto;
mod wire;

use crate::pipeline::{BackendError, InferencePipeline, ScoringBackend, TokenIds};
use crate::preprocess::ByteEncodingPreprocessor;
use fandhe_edge_core::fs::{FsError, read_bounded};
use fandhe_edge_core::hash::Sha256Digest;
use fandhe_edge_core::judgment::MAX_OPTIONS;
use proto::{GraphProto, NodeProto, TensorData, TensorProto};
use std::collections::HashMap;
use std::fmt;
use std::path::Path;

/// 読み込むモデルファイルのサイズ上限（バイト。読み込み前に確認。REQ-39）。学習ワーカーの
/// `limits.py::MAX_MODEL_BYTES`（40 MiB）に、`artifact.py::_ONNX_READ_CAP_BYTES` と同じ
/// 余裕 4 MiB を加えた値。
pub const MAX_MODEL_FILE_BYTES: u64 = 44 * 1024 * 1024;

/// `max_bytes`（前処理で切り詰めるバイト数）の下限。学習ワーカーの `limits.py`・共有 fixture
/// `fixtures/train_contract/limits.json` の `min_max_bytes` と一致する（テストで照合）。
pub const MIN_MAX_BYTES: usize = 1;

/// `max_bytes` の上限。`limits.json` の `max_max_bytes` と一致する（テストで照合）。C3 の計算量は
/// 系列長で有界にする必要があり、[`c3`] もこの値を系列長の上限に使う（REQ-39）。
pub const MAX_MAX_BYTES: usize = 4096;

/// 書き出し器が固定する ONNX の IR バージョン（`model_proto.ir_version = 8`）。
const EXPECTED_IR_VERSION: i64 = 8;
/// 書き出し器が固定する opset（既定 domain のバージョン 13）。
const EXPECTED_OPSET: i64 = 13;

/// 推論できるモデルの種類（[`crate::pipeline::ScoringBackend`] の実装が対応するもの）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModelKind {
    /// バイト n-gram TF-IDF + ロジスティック回帰（`kind="c1"`）。
    C1,
    /// バイト CNN（`kind="c3"`）。
    C3,
}

impl ModelKind {
    /// `artifact.json` の `kind` 文字列から種類を決める。未対応の種類（`autoregressive` を含む）は
    /// [`OnnxLoadError::UnsupportedKind`]（許可リスト方式。REQ-39）。
    pub fn parse(kind: &str) -> Result<Self, OnnxLoadError> {
        match kind {
            "c1" => Ok(Self::C1),
            "c3" => Ok(Self::C3),
            _ => Err(OnnxLoadError::UnsupportedKind),
        }
    }

    /// `kind` 文字列。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::C1 => "c1",
            Self::C3 => "c3",
        }
    }
}

/// モデル読み込みの失敗。モデルの重み・本文は保持しない（security.md）。
#[derive(Debug)]
#[non_exhaustive]
pub enum OnnxLoadError {
    /// ファイルを開けない・通常ファイルでない・上限超・読み込み失敗。
    File(FsError),
    /// sha256 が期待値と一致しない（REQ-39）。
    IntegrityMismatch,
    /// protobuf として不正（切り詰め・varint の桁あふれ・長さ不整合・非 ONNX 等）。
    MalformedProtobuf,
    /// `ir_version` が書き出し器の固定値と異なる。
    UnsupportedIrVersion,
    /// opset が書き出し器の固定値と異なる。
    UnsupportedOpset,
    /// グラフがテンプレートと一致しない（未知の op・順序・属性・入出力・initializer の不一致）。
    UnsupportedGraph,
    /// テンソルの形式が許可外（`raw_data` 以外・FLOAT/INT64 以外・外部参照・長さ不一致）。
    UnsupportedTensor,
    /// ノード数・名前長などの構造上の上限を超えた（REQ-39）。
    LimitExceeded,
    /// `kind` が推論対応の種類でない。
    UnsupportedKind,
    /// `max_bytes` が [`MIN_MAX_BYTES`]..=[`MAX_MAX_BYTES`] の範囲外。
    MaxBytesOutOfRange,
}

impl OnnxLoadError {
    /// 機械可読なエラーコード（英語）。
    pub fn code(&self) -> &'static str {
        match self {
            Self::File(FsError::TooLarge { .. }) => "limit_exceeded",
            Self::File(_) => "model_file_error",
            Self::IntegrityMismatch => "integrity_mismatch",
            Self::MalformedProtobuf => "malformed_protobuf",
            Self::UnsupportedIrVersion => "unsupported_ir_version",
            Self::UnsupportedOpset => "unsupported_opset",
            Self::UnsupportedGraph => "unsupported_graph",
            Self::UnsupportedTensor => "unsupported_tensor",
            Self::LimitExceeded => "limit_exceeded",
            Self::UnsupportedKind => "unsupported_kind",
            Self::MaxBytesOutOfRange => "max_bytes_out_of_range",
        }
    }
}

impl fmt::Display for OnnxLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::File(e) => write!(f, "model file error: {e}"),
            other => write!(f, "model rejected: {}", other.code()),
        }
    }
}

impl std::error::Error for OnnxLoadError {}

/// 種類別の実行器。
enum Inner {
    C1(c1::C1Model),
    C3(c3::C3Model),
}

/// 読み込み済みの ONNX モデル（C1 または C3）。[`ScoringBackend`] を実装する。
///
/// `&self` の純粋計算で、呼び出し間で結果に影響する状態を持たない（REQ-28）。
pub struct OnnxBackend {
    kind: ModelKind,
    inner: Inner,
}

impl fmt::Debug for OnnxBackend {
    /// 重みを出さず種類とクラス数のみ表示する。
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "OnnxBackend(kind={}, n_classes={})",
            self.kind.as_str(),
            self.n_classes()
        )
    }
}

impl OnnxBackend {
    /// ファイルを上限付きで読み、sha256 を照合してから復号する（REQ-39）。
    ///
    /// 経路の閉じ込め（`safe_join` 相当）は呼び出し側（CLI 統合。TASK-39.4-2）でガード層を通した
    /// 経路を渡す。本関数は通常ファイルの確認とサイズ上限の検査（`read_bounded`）を行う。
    pub fn load_path(
        path: &Path,
        kind: ModelKind,
        expected_sha256: &Sha256Digest,
    ) -> Result<Self, OnnxLoadError> {
        let bytes = read_bounded(path, MAX_MODEL_FILE_BYTES).map_err(OnnxLoadError::File)?;
        if Sha256Digest::of_bytes(&bytes) != *expected_sha256 {
            return Err(OnnxLoadError::IntegrityMismatch);
        }
        Self::from_bytes(&bytes, kind)
    }

    /// バイト列から復号する（sha256 の照合は呼び出し側。サイズ上限は [`MAX_MODEL_FILE_BYTES`]）。
    pub fn from_bytes(bytes: &[u8], kind: ModelKind) -> Result<Self, OnnxLoadError> {
        if u64::try_from(bytes.len()).map_or(true, |n| n > MAX_MODEL_FILE_BYTES) {
            return Err(OnnxLoadError::LimitExceeded);
        }
        let model = proto::decode_model(bytes)?;
        if model.ir_version != EXPECTED_IR_VERSION {
            return Err(OnnxLoadError::UnsupportedIrVersion);
        }
        let opset_ok = matches!(
            model.opsets.as_slice(),
            [o] if o.domain.is_empty() && o.version == EXPECTED_OPSET
        );
        if !opset_ok {
            return Err(OnnxLoadError::UnsupportedOpset);
        }
        let inner = match kind {
            ModelKind::C1 => Inner::C1(c1::C1Model::from_graph(&model.graph)?),
            ModelKind::C3 => Inner::C3(c3::C3Model::from_graph(&model.graph)?),
        };
        Ok(Self { kind, inner })
    }

    /// 読み込んだモデルの種類。
    pub fn kind(&self) -> ModelKind {
        self.kind
    }

    /// 出力するスコア数（選択肢の数）。
    pub fn n_classes(&self) -> usize {
        match &self.inner {
            Inner::C1(m) => m.n_classes(),
            Inner::C3(m) => m.n_classes(),
        }
    }
}

impl ScoringBackend for OnnxBackend {
    fn scores(&self, ids: &TokenIds) -> Result<Vec<f64>, BackendError> {
        let ids = ids.as_slice();
        check_token_ids(ids)?;
        match &self.inner {
            Inner::C1(m) => m.scores(ids),
            Inner::C3(m) => m.scores(ids),
        }
    }

    fn scores_limited(
        &self,
        ids: &TokenIds,
        limit: std::time::Duration,
    ) -> Result<Vec<f64>, BackendError> {
        let ids = ids.as_slice();
        check_token_ids(ids)?;
        match &self.inner {
            Inner::C1(m) => m.scores_limited(ids, limit),
            Inner::C3(m) => m.scores_limited(ids, limit),
        }
    }
}

/// 前処理（`max_bytes` の範囲検証つき）と ONNX バックエンドを束ねた推論パイプラインを組み立てる。
///
/// `max_bytes` は配布パッケージのメタデータ由来の外部入力として [`MIN_MAX_BYTES`]..=
/// [`MAX_MAX_BYTES`] を検証してから前処理へ渡す。
pub fn load_pipeline(
    kind: ModelKind,
    max_bytes: usize,
    path: &Path,
    expected_sha256: &Sha256Digest,
) -> Result<InferencePipeline<ByteEncodingPreprocessor, OnnxBackend>, OnnxLoadError> {
    if !(MIN_MAX_BYTES..=MAX_MAX_BYTES).contains(&max_bytes) {
        return Err(OnnxLoadError::MaxBytesOutOfRange);
    }
    let backend = OnnxBackend::load_path(path, kind, expected_sha256)?;
    Ok(InferencePipeline::new(
        ByteEncodingPreprocessor::new(max_bytes),
        backend,
    ))
}

// ---- C1・C3 共通の照合・計算の補助（子モジュールから使う） ----

/// トークン値の語彙サイズ（バイト値 0..=255 に 1 を足した 1..=256 と詰め物 0）。
const N_TOKENS: usize = 257;

/// 入力トークンが空・[`MAX_MAX_BYTES`] 超過、または `0..N_TOKENS` の範囲外なら拒否する（C1・C3 共通。fail-closed）。
fn check_token_ids(ids: &[i64]) -> Result<(), BackendError> {
    // 系列長の上限は C1・C3 共通の入口で検証する（REQ-39。C1 の n-gram 走査・集計量は長さに比例する）
    if ids.is_empty() || ids.len() > MAX_MAX_BYTES {
        return Err(BackendError::InvalidSequenceLength);
    }
    let in_range = |&id: &i64| usize::try_from(id).is_ok_and(|v| v < N_TOKENS);
    if !ids.iter().all(in_range) {
        return Err(BackendError::InvalidTokenId);
    }
    Ok(())
}

/// f32 のロジットから f64 で softmax を計算する（モジュール doc「数値」）。非有限なら拒否。
fn softmax_f64(logits: &[f32]) -> Result<Vec<f64>, BackendError> {
    if logits.is_empty() || logits.iter().any(|l| !l.is_finite()) {
        return Err(BackendError::Failed);
    }
    let max = logits
        .iter()
        .fold(f64::NEG_INFINITY, |m, &l| m.max(f64::from(l)));
    let exps: Vec<f64> = logits.iter().map(|&l| (f64::from(l) - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    Ok(exps.iter().map(|e| e / sum).collect())
}

/// ノードが書き出し器のテンプレート（op・入力名・出力名・属性名の集合）と一致することを確認する。
/// 属性は名前の集合が完全一致（過不足・重複なし）であること。値は呼び出し側が個別に確認する。
fn check_node(
    node: &NodeProto,
    op: &str,
    inputs: &[&str],
    outputs: &[&str],
    attrs: &[&str],
) -> Result<(), OnnxLoadError> {
    let names_eq = |actual: &[String], want: &[&str]| {
        actual.len() == want.len() && actual.iter().zip(want).all(|(a, w)| a == w)
    };
    let attrs_ok = node.attrs.len() == attrs.len()
        && attrs
            .iter()
            .all(|w| node.attrs.iter().filter(|a| a.name == *w).count() == 1);
    if node.op_type == op
        && node.domain.is_empty()
        && names_eq(&node.inputs, inputs)
        && names_eq(&node.outputs, outputs)
        && attrs_ok
    {
        Ok(())
    } else {
        Err(OnnxLoadError::UnsupportedGraph)
    }
}

/// INT 属性が期待値であること。
fn attr_int_is(node: &NodeProto, name: &str, want: i64) -> Result<(), OnnxLoadError> {
    match node.attr(name).and_then(|a| a.int()) {
        Some(v) if v == want => Ok(()),
        _ => Err(OnnxLoadError::UnsupportedGraph),
    }
}

/// INTS 属性が期待値であること。
fn attr_ints_are(node: &NodeProto, name: &str, want: &[i64]) -> Result<(), OnnxLoadError> {
    match node.attr(name).and_then(|a| a.ints()) {
        Some(v) if v == want => Ok(()),
        _ => Err(OnnxLoadError::UnsupportedGraph),
    }
}

/// FLOAT 属性が期待値とビット単位で一致すること（テンプレートの定数 1.0 の照合用）。
fn attr_f32_is(node: &NodeProto, name: &str, want: f32) -> Result<(), OnnxLoadError> {
    match node.attr(name).and_then(|a| a.float()) {
        Some(v) if v.to_bits() == want.to_bits() => Ok(()),
        _ => Err(OnnxLoadError::UnsupportedGraph),
    }
}

/// グラフの入出力が `ids: INT64[N, T]`・`probs: FLOAT[N, n_classes]` であること。
fn check_graph_io(graph: &GraphProto, n_classes: usize) -> Result<(), OnnxLoadError> {
    use proto::{DT_FLOAT, DT_INT64, Dim};
    let ok_in = matches!(
        graph.inputs.as_slice(),
        [i] if i.name == "ids"
            && i.elem_type == DT_INT64
            && i.dims == [Dim::Param("N".into()), Dim::Param("T".into())]
    );
    let want_k = i64::try_from(n_classes).map_err(|_| OnnxLoadError::UnsupportedGraph)?;
    let ok_out = matches!(
        graph.outputs.as_slice(),
        [o] if o.name == "probs"
            && o.elem_type == DT_FLOAT
            && o.dims == [Dim::Param("N".into()), Dim::Value(want_k)]
    );
    if ok_in && ok_out {
        Ok(())
    } else {
        Err(OnnxLoadError::UnsupportedGraph)
    }
}

/// クラス数の範囲（1..=`MAX_OPTIONS`）。
fn check_n_classes(k: usize) -> Result<(), OnnxLoadError> {
    if (1..=MAX_OPTIONS).contains(&k) {
        Ok(())
    } else {
        Err(OnnxLoadError::UnsupportedGraph)
    }
}

/// 名前で引ける initializer の集合。名前の重複は拒否する。参照専用（出力順に影響しない）。
struct Inits<'a> {
    map: HashMap<&'a str, &'a TensorProto>,
}

impl<'a> Inits<'a> {
    fn new(graph: &'a GraphProto) -> Result<Self, OnnxLoadError> {
        let mut map = HashMap::new();
        for t in &graph.initializers {
            if map.insert(t.name.as_str(), t).is_some() {
                return Err(OnnxLoadError::UnsupportedGraph);
            }
        }
        Ok(Self { map })
    }

    /// initializer の名前集合が `names` と完全一致すること（過不足なし）。
    fn require_exactly(&self, names: &[String]) -> Result<(), OnnxLoadError> {
        if self.map.len() == names.len() && names.iter().all(|n| self.map.contains_key(n.as_str()))
        {
            Ok(())
        } else {
            Err(OnnxLoadError::UnsupportedGraph)
        }
    }

    /// FLOAT テンソル（全値が有限）と形状を返す。
    fn f32(&self, name: &str) -> Result<(&'a [i64], &'a [f32]), OnnxLoadError> {
        match self.map.get(name).map(|t| (&t.dims, &t.data)) {
            Some((dims, TensorData::F32(v))) if v.iter().all(|x| x.is_finite()) => {
                Ok((dims.as_slice(), v.as_slice()))
            }
            _ => Err(OnnxLoadError::UnsupportedGraph),
        }
    }

    /// INT64 テンソルと形状を返す。
    fn i64(&self, name: &str) -> Result<(&'a [i64], &'a [i64]), OnnxLoadError> {
        match self.map.get(name).map(|t| (&t.dims, &t.data)) {
            Some((dims, TensorData::I64(v))) => Ok((dims.as_slice(), v.as_slice())),
            _ => Err(OnnxLoadError::UnsupportedGraph),
        }
    }

    /// FLOAT のスカラー。書き出し器は `np.ascontiguousarray` を通すため 0 次元も形状 `[1]` で出る。
    fn f32_scalar(&self, name: &str) -> Result<f32, OnnxLoadError> {
        match self.f32(name)? {
            ([1], [v]) => Ok(*v),
            _ => Err(OnnxLoadError::UnsupportedGraph),
        }
    }

    /// FLOAT テンソルで、形状が `dims` と一致すること。
    fn f32_shaped(&self, name: &str, dims: &[usize]) -> Result<&'a [f32], OnnxLoadError> {
        let (d, v) = self.f32(name)?;
        shape_is(d, dims)?;
        Ok(v)
    }
}

/// 形状が期待どおりであること。
fn shape_is(actual: &[i64], want: &[usize]) -> Result<(), OnnxLoadError> {
    let same = actual.len() == want.len()
        && actual
            .iter()
            .zip(want)
            .all(|(a, w)| usize::try_from(*a).is_ok_and(|a| a == *w));
    if same {
        Ok(())
    } else {
        Err(OnnxLoadError::UnsupportedGraph)
    }
}

/// スカラーがテンプレートの定数とビット単位で一致すること。
fn scalar_is(v: f32, want: f32) -> Result<(), OnnxLoadError> {
    if v.to_bits() == want.to_bits() {
        Ok(())
    } else {
        Err(OnnxLoadError::UnsupportedGraph)
    }
}
