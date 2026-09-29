//! C1（バイト n-gram TF-IDF + ロジスティック回帰）の ONNX テンプレート照合と順伝播
//! （REQ-32・REQ-28・REQ-39・TASK-32.1-2・#113）。
//!
//! # 役割
//!
//! [`super::OnnxBackend`] の C1 用の実行器。学習ワーカーの `kinds/c1.py::_export_c1_onnx` が書き出す
//! 14 ノードのグラフ（`TfIdfVectorizer`(mode=TF) → sublinear TF → idf → L2 正規化 → `Gemm` →
//! `Softmax`）とノード列・入出力名・属性・initializer を完全一致で照合し、一致したものだけを
//! 実行する（許可制。書き出し器を変えたらこの照合と `tests/onnx_parity.rs` を合わせて直す）。
//!
//! # 計算
//!
//! 1 系列（バッチ次元なし。REQ-28）について、サイズ n ごとの連続 n-gram（`max_skip_count=0`）の
//! 出現回数 tf を数え、`(ln(max(tf,1)) + [tf>0]) * idf` → L2 正規化（`max(norm, eps)` で割る）→
//! `x·W + b` を f32 で計算し、最後だけ f64 で softmax する（[`super`] の「数値」）。tf=0 の列は
//! sumsq にも `Gemm` にも 0 しか寄与しないため、出現した列だけを列番号の昇順（密なグラフの
//! 添字順と同じ順序）で処理する。語彙の n-gram に詰め物 id=0 は含まれない（書き出し器が保証し、
//! 読み込み時にも検査する）ので、詰め物位置が結果に影響しない。
//!
//! # 資源
//!
//! 語彙表は読み込み時に n-gram を 64 bit 整数（n ≤ 7、1 バイト 1 トークン）へ詰めて作る
//! （照合専用の `HashMap`。出力順に影響しない）。表のサイズはモデルファイルの大きさで有界。

use super::proto::GraphProto;
use super::{
    Inits, OnnxLoadError, attr_f32_is, attr_int_is, check_graph_io, check_n_classes, check_node,
    scalar_is, shape_is, softmax_f64,
};
use crate::pipeline::BackendError;
use std::collections::{BTreeMap, HashMap};
use std::time::{Duration, Instant};

/// n-gram の最大長（学習ワーカーの `limits.py::MAX_C1_NGRAM`）。64 bit キーに収まる上限でもある。
const MAX_NGRAM: usize = 7;

/// 1 系列の計算時間の上限（REQ-39。C3 の `MAX_INFER_DURATION`・
/// `latency::DEFAULT_LATENCY_PER_INFER_TIMEOUT_NS` と同値）。入力長の上限だけでは、
/// 大きな語彙表・重みを持つモデルの走査時間を抑えられない。
const MAX_INFER_DURATION: Duration = Duration::from_secs(10);
/// 経過時間を検査する間隔（反復回数）。`Instant::now` の呼び出しを間引くための値。
const CHECK_INTERVAL: usize = 256;

/// 書き出し器のノード列テンプレート: (op_type, 入力名, 出力名, 属性名)。
type NodeSpec = (
    &'static str,
    &'static [&'static str],
    &'static [&'static str],
    &'static [&'static str],
);

const TEMPLATE: [NodeSpec; 14] = [
    (
        "TfIdfVectorizer",
        &["ids"],
        &["tf"],
        &[
            "mode",
            "min_gram_length",
            "max_gram_length",
            "max_skip_count",
            "ngram_counts",
            "ngram_indexes",
            "pool_int64s",
        ],
    ),
    ("Max", &["tf", "one_f32"], &["tf_clip"], &[]),
    ("Log", &["tf_clip"], &["log_tf"], &[]),
    ("Greater", &["tf", "zero_f32"], &["presence_bool"], &[]),
    ("Cast", &["presence_bool"], &["presence_f"], &["to"]),
    ("Add", &["log_tf", "presence_f"], &["sublinear"], &[]),
    ("Mul", &["sublinear", "idf"], &["weighted"], &[]),
    ("Mul", &["weighted", "weighted"], &["sq"], &[]),
    (
        "ReduceSum",
        &["sq", "reduce_axes_1"],
        &["sumsq"],
        &["keepdims"],
    ),
    ("Sqrt", &["sumsq"], &["norm"], &[]),
    ("Max", &["norm", "eps_f32"], &["denom"], &[]),
    ("Div", &["weighted", "denom"], &["xnorm"], &[]),
    (
        "Gemm",
        &["xnorm", "weight", "bias"],
        &["logits"],
        &["alpha", "beta"],
    ),
    ("Softmax", &["logits"], &["probs"], &["axis"]),
];

/// (n-gram の長さ, n-gram → 列番号)。照合専用の表。
type NgramTable = (usize, HashMap<u64, u32>);

/// 照合済みの C1 モデル。
pub(super) struct C1Model {
    /// (n-gram の長さ, n-gram → 列番号)。長さの昇順。照合専用。
    tables: Vec<NgramTable>,
    idf: Vec<f32>,
    /// 形状 `[n_features, n_classes]` の行優先。
    weight: Vec<f32>,
    bias: Vec<f32>,
    eps: f32,
    n_classes: usize,
}

/// 1..=256 のトークン列を 1 バイト 1 トークンで 64 bit に詰める（範囲外なら `None`）。
fn pack_ngram<'a>(ids: impl IntoIterator<Item = &'a i64>) -> Option<u64> {
    let mut key = 0u64;
    for &id in ids {
        let byte = u8::try_from(id.checked_sub(1)?).ok()?;
        key = (key << 8) | u64::from(byte);
    }
    Some(key)
}

impl C1Model {
    pub(super) fn n_classes(&self) -> usize {
        self.n_classes
    }

    /// グラフをテンプレートと照合し、パラメータを取り出す。
    pub(super) fn from_graph(graph: &GraphProto) -> Result<Self, OnnxLoadError> {
        if graph.nodes.len() != TEMPLATE.len() {
            return Err(OnnxLoadError::UnsupportedGraph);
        }
        for (node, (op, ins, outs, attrs)) in graph.nodes.iter().zip(TEMPLATE.iter()) {
            check_node(node, op, ins, outs, attrs)?;
        }
        let nodes = &graph.nodes;
        let node = |i: usize| nodes.get(i).ok_or(OnnxLoadError::UnsupportedGraph);
        attr_int_is(node(4)?, "to", 1)?;
        attr_int_is(node(8)?, "keepdims", 1)?;
        attr_f32_is(node(12)?, "alpha", 1.0)?;
        attr_f32_is(node(12)?, "beta", 1.0)?;
        attr_int_is(node(13)?, "axis", 1)?;

        let inits = Inits::new(graph)?;
        let names: Vec<String> = [
            "idf",
            "one_f32",
            "zero_f32",
            "eps_f32",
            "reduce_axes_1",
            "weight",
            "bias",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
        inits.require_exactly(&names)?;

        let (wdims, _) = inits.f32("weight")?;
        let (n_features, n_classes) = match wdims {
            [f, k] => (
                usize::try_from(*f).map_err(|_| OnnxLoadError::UnsupportedGraph)?,
                usize::try_from(*k).map_err(|_| OnnxLoadError::UnsupportedGraph)?,
            ),
            _ => return Err(OnnxLoadError::UnsupportedGraph),
        };
        check_n_classes(n_classes)?;
        if n_features == 0 {
            return Err(OnnxLoadError::UnsupportedGraph);
        }
        let weight = inits.f32_shaped("weight", &[n_features, n_classes])?;
        let bias = inits.f32_shaped("bias", &[n_classes])?;
        let idf = inits.f32_shaped("idf", &[n_features])?;
        scalar_is(inits.f32_scalar("one_f32")?, 1.0)?;
        scalar_is(inits.f32_scalar("zero_f32")?, 0.0)?;
        let eps = inits.f32_scalar("eps_f32")?;
        if eps <= 0.0 {
            return Err(OnnxLoadError::UnsupportedGraph);
        }
        match inits.i64("reduce_axes_1")? {
            (d, [1]) => shape_is(d, &[1])?,
            _ => return Err(OnnxLoadError::UnsupportedGraph),
        }
        check_graph_io(graph, n_classes)?;

        let tables = parse_tfidf(node(0)?, n_features)?;
        Ok(Self {
            tables,
            idf: idf.to_vec(),
            weight: weight.to_vec(),
            bias: bias.to_vec(),
            eps,
            n_classes,
        })
    }

    /// 1 系列のスコア（確率。選択肢の宣言順）。
    pub(super) fn scores(&self, ids: &[i64]) -> Result<Vec<f64>, BackendError> {
        self.scores_within(ids, MAX_INFER_DURATION)
    }

    /// 呼び出し側の残り時間（`limit`）と 1 件の上限 `MAX_INFER_DURATION` の小さい方で打ち切るスコア計算
    /// （照合全体の期限を推論中にも強制するため。REQ-39）。
    pub(super) fn scores_limited(
        &self,
        ids: &[i64],
        limit: Duration,
    ) -> Result<Vec<f64>, BackendError> {
        self.scores_within(ids, limit.min(MAX_INFER_DURATION))
    }

    /// 計算時間の上限付きのスコア計算。n-gram の走査・重み行の加算の反復境界で経過時間を検査し、
    /// 超過したら [`BackendError::TimeLimitExceeded`] で打ち切る（REQ-39）。
    fn scores_within(&self, ids: &[i64], limit: Duration) -> Result<Vec<f64>, BackendError> {
        let started = Instant::now();
        let mut steps = 0usize;
        let mut check = || -> Result<(), BackendError> {
            if steps.is_multiple_of(CHECK_INTERVAL) && started.elapsed() >= limit {
                return Err(BackendError::TimeLimitExceeded);
            }
            steps = steps.wrapping_add(1);
            Ok(())
        };
        // 列番号 → 出現回数。昇順の反復で密なグラフの添字順と同じ累積順にする。
        let mut tf: BTreeMap<u32, u32> = BTreeMap::new();
        for (n, table) in &self.tables {
            for window in ids.windows(*n) {
                check()?;
                if let Some(col) = pack_ngram(window).and_then(|key| table.get(&key)) {
                    let c = tf.entry(*col).or_insert(0);
                    *c = c.saturating_add(1);
                }
            }
        }

        let mut weighted: Vec<(usize, f32)> = Vec::with_capacity(tf.len());
        let mut sumsq = 0.0f32;
        for (&col, &count) in &tf {
            check()?;
            let col = usize::try_from(col).map_err(|_| BackendError::Failed)?;
            let idf = *self.idf.get(col).ok_or(BackendError::Failed)?;
            // count > 0 なので sublinear TF は ln(max(count, 1)) + 1
            let sublinear = (count as f32).max(1.0).ln() + 1.0;
            let w = sublinear * idf;
            sumsq += w * w;
            weighted.push((col, w));
        }
        let denom = sumsq.sqrt().max(self.eps);

        let k = self.n_classes;
        let mut logits = vec![0.0f32; k];
        for &(col, w) in &weighted {
            check()?;
            let x = w / denom;
            let start = col.checked_mul(k).ok_or(BackendError::Failed)?;
            let end = start.checked_add(k).ok_or(BackendError::Failed)?;
            let row = self.weight.get(start..end).ok_or(BackendError::Failed)?;
            for (acc, wk) in logits.iter_mut().zip(row) {
                *acc += x * wk;
            }
        }
        for (acc, b) in logits.iter_mut().zip(&self.bias) {
            *acc += b;
        }
        softmax_f64(&logits)
    }
}

/// `TfIdfVectorizer` の属性を照合し、n-gram の長さごとの語彙表を作る。
///
/// レイアウト: `ngram_counts[n-1]` は長さ n の n-gram が `pool_int64s` 内で始まる要素位置で、
/// 長さ 1 から `max_gram_length` まで隙間なく数える（`min_gram_length` 未満の長さは要素 0 件）。
/// `ngram_indexes` は `0..n_features` の恒等列、`pool_int64s` はトークン値 1..=256 のみ
/// （詰め物 0 を含む n-gram は拒否）。同じ長さ内の重複 n-gram は拒否する。
fn parse_tfidf(
    node: &super::proto::NodeProto,
    n_features: usize,
) -> Result<Vec<NgramTable>, OnnxLoadError> {
    let bad = || OnnxLoadError::UnsupportedGraph;
    let attr = |name: &str| node.attr(name).ok_or_else(bad);
    if attr("mode")?.string() != Some(b"TF".as_slice()) {
        return Err(bad());
    }
    attr_int_is(node, "max_skip_count", 0)?;
    let to_len = |v: Option<i64>| -> Result<usize, OnnxLoadError> {
        v.and_then(|v| usize::try_from(v).ok()).ok_or_else(bad)
    };
    let min_n = to_len(attr("min_gram_length")?.int())?;
    let max_n = to_len(attr("max_gram_length")?.int())?;
    if !(1..=MAX_NGRAM).contains(&min_n) || !(min_n..=MAX_NGRAM).contains(&max_n) {
        return Err(bad());
    }
    let counts = attr("ngram_counts")?.ints().ok_or_else(bad)?;
    let indexes = attr("ngram_indexes")?.ints().ok_or_else(bad)?;
    let pool = attr("pool_int64s")?.ints().ok_or_else(bad)?;
    if counts.len() != max_n || indexes.len() != n_features {
        return Err(bad());
    }
    if !indexes
        .iter()
        .enumerate()
        .all(|(i, &v)| usize::try_from(v).is_ok_and(|v| v == i))
    {
        return Err(bad());
    }
    if !pool.iter().all(|&v| (1..=256).contains(&v)) {
        return Err(bad());
    }
    let start_of = |n: usize| -> Result<usize, OnnxLoadError> {
        // n は 1..=max_n。n == max_n + 1 のときは pool の終端
        if n > max_n {
            return Ok(pool.len());
        }
        to_len(counts.get(n - 1).copied())
    };
    if start_of(1)? != 0 {
        return Err(bad());
    }
    let mut tables = Vec::new();
    let mut col: usize = 0;
    for n in 1..=max_n {
        let (start, end) = (start_of(n)?, start_of(n + 1)?);
        if start > end || end > pool.len() {
            return Err(bad());
        }
        let seg = pool.get(start..end).ok_or_else(bad)?;
        if n < min_n {
            if !seg.is_empty() {
                return Err(bad());
            }
            continue;
        }
        if seg.len() % n != 0 {
            return Err(bad());
        }
        if seg.is_empty() {
            continue;
        }
        let mut table: HashMap<u64, u32> = HashMap::new();
        for gram in seg.chunks_exact(n) {
            let key = pack_ngram(gram).ok_or_else(bad)?;
            let id = u32::try_from(col).map_err(|_| bad())?;
            if table.insert(key, id).is_some() {
                return Err(bad());
            }
            col += 1;
        }
        tables.push((n, table));
    }
    if col != n_features {
        return Err(bad());
    }
    Ok(tables)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-39: 時間上限 0 では最初の反復で打ち切られ、通常の上限では成功する。
    #[test]
    fn req39_c1_scores_abort_when_time_limit_exceeded() {
        let mut table = HashMap::new();
        table.insert(1u64, 0u32);
        let model = C1Model {
            tables: vec![(1, table)],
            idf: vec![2.0],
            weight: vec![1.0, 0.0],
            bias: vec![0.0, 0.0],
            eps: 1e-12,
            n_classes: 2,
        };
        assert_eq!(
            model.scores_within(&[2], Duration::ZERO).err(),
            Some(BackendError::TimeLimitExceeded)
        );
        assert!(model.scores_within(&[2], MAX_INFER_DURATION).is_ok());
    }

    /// REQ-32: n-gram のパックは 1..=256 のみ受理し、0・257 は拒否する。
    #[test]
    fn req32_pack_ngram_range() {
        assert_eq!(pack_ngram(&[1i64, 256]), Some(0x00ff));
        assert_eq!(pack_ngram(&[0i64]), None);
        assert_eq!(pack_ngram(&[257i64]), None);
        assert_eq!(pack_ngram(&[-1i64]), None);
    }

    /// REQ-32: 出現する n-gram が無い（空入力 `[0]`）ときのスコアはバイアスの softmax になる。
    #[test]
    fn req32_empty_input_scores_are_softmax_of_bias() {
        let model = C1Model {
            tables: vec![],
            idf: vec![1.0],
            weight: vec![5.0, -5.0],
            bias: vec![0.0, 0.0],
            eps: 1e-12,
            n_classes: 2,
        };
        let p = model.scores(&[0]).unwrap_or_default();
        assert_eq!(p.len(), 2);
        assert!((p.first().copied().unwrap_or(0.0) - 0.5).abs() < 1e-12);
    }

    /// REQ-32: 出現した n-gram の列だけが寄与し、L2 正規化後の値で `Gemm` される。
    #[test]
    fn req32_single_ngram_is_l2_normalized() {
        // 1-gram [ids 2] → col 0。tf=1 → sublinear=1、weighted=idf=2、正規化後 x=1。
        let mut table = HashMap::new();
        table.insert(1u64, 0u32);
        let model = C1Model {
            tables: vec![(1, table)],
            idf: vec![2.0],
            weight: vec![1.0, 0.0],
            bias: vec![0.0, 0.0],
            eps: 1e-12,
            n_classes: 2,
        };
        let p = model.scores(&[2]).unwrap_or_default();
        let e = 1.0f64.exp();
        let want = e / (e + 1.0);
        assert!((p.first().copied().unwrap_or(0.0) - want).abs() < 1e-6);
    }
}
