//! 単体・バッチ・評価器経路の予測ラベル全件一致テスト（REQ-28・TASK-28.1-2・#118）。
//!
//! 証拠種別: テストハーネス（模擬の前処理・模擬バックエンド）。本テストは比較の規則を模擬で検証するもので、
//! 実前処理（#112）・ONNX 推論（#113）の実モデルでの全件一致は `tests/onnx_parity.rs` が担う。
//! 実機・E2E の証拠ではない。比較ハーネスは前処理・バックエンドに対して generic。
//!
//! PoC-16 の不一致原因（バッチ内パディング位置の混入）を再現しうる模擬バックエンド
//! （系列長・窓に敏感）で、650 件について 3 経路（単体・バッチ・評価器内）の予測ラベルを
//! 突き合わせ、不一致 index の一覧が空であることを具体値で確認する（fail-closed）。
//! 失敗メッセージは index・件数のみで入力本文を出さない（security.md）。
//! 不一致を検出したら `docs/design/runtime-batch-mismatch-procedure.md`（REQ-28 異常系・TASK-28.2）に従う。

use fandhe_edge_core::infer_input::MAX_INFER_INPUT_BYTES;
use fandhe_edge_eval::input_only::{EvalItem, run_inference_input_only};
use fandhe_edge_runtime::pipeline::*;
use std::rc::Rc;

#[allow(dead_code)]
#[path = "common/qwen_scoring.rs"]
mod qwen_scoring;
use qwen_scoring::{QwenRefBackend, bytes_to_ids};

const TOL: f64 = 1e-9;
const LABELS: usize = 4;
const CORPUS_LEN: usize = 650;

/// UTF-8 バイト + 1、空なら [0]（実前処理の代用。NFKC は行わない。#112 で置換）。
struct MockPre;
fn mock_ids(input: &str) -> Vec<i64> {
    if input.is_empty() {
        return vec![0];
    }
    input.bytes().map(|b| i64::from(b) + 1).collect()
}
impl Preprocessor for MockPre {
    fn preprocess(&self, input: &str) -> Result<TokenIds, PreprocessError> {
        Ok(TokenIds::new(mock_ids(input)))
    }
}

/// 系列長・位置・隣接要素（幅 3 の窓）に敏感な決定的スコア。softmax で合計 1。
fn window_scores(ids: &[i64]) -> Vec<f64> {
    let mut raw = [0.0f64; LABELS];
    for (k, r) in raw.iter_mut().enumerate() {
        let w = (k + 1) as f64;
        for i in 0..ids.len() {
            let lo = i.saturating_sub(1);
            let hi = (i + 2).min(ids.len());
            let s: i64 = ids.get(lo..hi).map_or(0, |x| x.iter().sum());
            *r += (s as f64) * w * (((i % 5) + 1) as f64) / 1000.0;
        }
        *r /= ids.len() as f64;
        *r = (*r * (k as f64 + 1.0)).sin();
    }
    let exp: Vec<f64> = raw.iter().map(|x| x.exp()).collect();
    let sum: f64 = exp.iter().sum();
    exp.iter().map(|x| x / sum).collect()
}

/// 模擬バックエンド（ONNX 推論の代用。#113 で置換）。
struct MockBackend;
impl ScoringBackend for MockBackend {
    /// テスト用スタブ: 計算は即時で打ち切り対象の反復を持たないため、`scores` と同じ結果を返す。
    fn scores_limited(
        &self,
        ids: &fandhe_edge_runtime::pipeline::TokenIds,
        _limit: std::time::Duration,
    ) -> Result<Vec<f64>, fandhe_edge_runtime::pipeline::BackendError> {
        self.scores(ids)
    }
    fn scores(&self, ids: &TokenIds) -> Result<Vec<f64>, BackendError> {
        Ok(window_scores(ids.as_slice()))
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 33
}

/// 固定 seed の LCG で決定的に合成した 650 件（実データは使わない）。
fn corpus() -> Vec<String> {
    let pool: Vec<&str> = vec![
        "a", "b", "z", "0", "9", " ", "\t", "\n", "-", "_", "あ", "い", "漢", "字", "テ", "スト",
        "Ａ", "ｂ", "１", "ﾃ", "😀", "🎉", "é", "ß",
    ];
    let mut st = 0x1188_2026_u64;
    let mut out: Vec<String> = vec![String::new(), " ".into(), "   \t\n".into()];
    while out.len() < CORPUS_LEN {
        let n = lcg(&mut st);
        // 長短を混ぜる（0〜約 4000 バイト）。
        let chars = match n % 10 {
            0 => 0,
            1..=5 => (lcg(&mut st) % 12) as usize,
            6..=8 => (lcg(&mut st) % 200) as usize,
            _ => (lcg(&mut st) % 1000) as usize,
        };
        let mut s = String::new();
        for _ in 0..chars {
            let idx = (lcg(&mut st) % pool.len() as u64) as usize;
            s.push_str(pool.get(idx).copied().unwrap_or("a"));
        }
        out.push(s);
    }
    // 同一入力の重複を含める。
    let dup = out.get(10).cloned().unwrap_or_default();
    if let Some(slot) = out.get_mut(400) {
        *slot = dup;
    }
    out
}

fn shuffled(n: usize, seed: u64) -> Vec<usize> {
    let mut v: Vec<usize> = (0..n).collect();
    let mut st = seed;
    for i in (1..n).rev() {
        let j = (lcg(&mut st) % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
    v
}

/// 不一致 index の一覧。長さ違いは fail-closed で失敗させる。
fn label_mismatches(a: &[usize], b: &[usize]) -> Vec<usize> {
    assert_eq!(a.len(), b.len(), "label vector length differs");
    a.iter()
        .zip(b)
        .enumerate()
        .filter(|(_, (x, y))| x != y)
        .map(|(i, _)| i)
        .collect()
}

fn ok(r: Result<Prediction, InferError>, i: usize) -> Prediction {
    match r {
        Ok(p) => p,
        Err(e) => panic!("inference failed at index {i}: code={}", e.code()),
    }
}

fn labels_of(preds: &[Prediction]) -> Vec<usize> {
    preds.iter().map(Prediction::label_index).collect()
}

fn scores_close(a: &[Prediction], b: &[Prediction]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.scores().len() == y.scores().len()
                && x.scores()
                    .iter()
                    .zip(y.scores())
                    .all(|(p, q)| (p - q).abs() <= TOL)
        })
}

fn single<P: Preprocessor, B: ScoringBackend>(
    pl: &InferencePipeline<P, B>,
    inputs: &[String],
) -> Vec<Prediction> {
    inputs
        .iter()
        .enumerate()
        .map(|(i, x)| ok(pl.infer_one(x), i))
        .collect()
}

/// 並べ替え・チャンク分割したバッチ推論の結果を元の index へ戻す。
fn batched<P: Preprocessor, B: ScoringBackend>(
    pl: &InferencePipeline<P, B>,
    inputs: &[String],
    order: &[usize],
    chunk: usize,
) -> Vec<Prediction> {
    let mut slots: Vec<Option<Prediction>> = vec![None; inputs.len()];
    for idxs in order.chunks(chunk) {
        let refs: Vec<&str> = idxs
            .iter()
            .map(|&i| inputs.get(i).map_or("", String::as_str))
            .collect();
        let res = pl.infer_batch(&refs).expect("batch within limits");
        assert_eq!(res.len(), idxs.len());
        for (&i, r) in idxs.iter().zip(res) {
            if let Some(s) = slots.get_mut(i) {
                *s = Some(ok(r, i));
            }
        }
    }
    slots
        .into_iter()
        .enumerate()
        .map(|(i, p)| p.unwrap_or_else(|| panic!("missing result at index {i}")))
        .collect()
}

/// 評価器の入力隔離経路（`run_inference_input_only`）で得た予測。
fn via_evaluator<P: Preprocessor + 'static, B: ScoringBackend + 'static>(
    pl: InferencePipeline<P, B>,
    inputs: &[String],
) -> Vec<Prediction> {
    let pl = Rc::new(pl);
    let q = Rc::clone(&pl);
    let tags: Vec<String> = Vec::new();
    let ids: Vec<String> = (0..inputs.len()).map(|i| format!("r{i}")).collect();
    let items: Vec<EvalItem<'_>> = inputs
        .iter()
        .zip(&ids)
        .map(|(x, id)| EvalItem {
            id,
            input: x,
            gold: "dummy",
            tags: &tags,
        })
        .collect();
    let preds = run_inference_input_only(&items, move |s: &str| (q.as_predict_fn())(s))
        .unwrap_or_else(|e| panic!("evaluator path aborted: {e:?}"));
    preds
        .into_iter()
        .enumerate()
        .map(|(i, r)| ok(r, i))
        .collect()
}

fn pipeline() -> InferencePipeline<MockPre, MockBackend> {
    InferencePipeline::new(MockPre, MockBackend)
}

/// パイプライン外の素朴なパディング付きバッチ（陰性対照。最大長へ 0 詰め）。
fn naive_padded_labels(inputs: &[String]) -> Vec<usize> {
    let seqs: Vec<Vec<i64>> = inputs.iter().map(|x| mock_ids(x)).collect();
    let max = seqs.iter().map(Vec::len).max().unwrap_or(0);
    seqs.into_iter()
        .map(|mut s| {
            s.resize(max, 0);
            window_scores(&s)
                .iter()
                .enumerate()
                .fold(
                    (0usize, f64::MIN),
                    |(bi, bv), (i, &v)| {
                        if v > bv { (i, v) } else { (bi, bv) }
                    },
                )
                .0
        })
        .collect()
}

/// REQ-28・TASK-28.1-2・#118。コーパスが上限内で 650 件であること（証拠種別: テストハーネス）。
#[test]
fn req28_corpus_is_650_within_limits() {
    let c = corpus();
    assert_eq!(c.len(), 650);
    assert!(c.iter().all(|x| x.len() < MAX_INFER_INPUT_BYTES));
    let total: usize = c.iter().map(String::len).sum();
    assert!(total < MAX_INFER_BATCH_TOTAL_BYTES);
}

/// REQ-28・TASK-28.1-2・#118。全 650 件で 3 経路の予測ラベルが一致し不一致 0 件
/// （証拠種別: テストハーネス。模擬前処理・模擬バックエンド）。
#[test]
fn req28_single_batch_evaluator_labels_match_all_650() {
    let c = corpus();
    let s = single(&pipeline(), &c);
    let s_labels = labels_of(&s);
    assert_eq!(s_labels.len(), 650);

    // 全件 1 バッチ。
    let identity: Vec<usize> = (0..c.len()).collect();
    let b = batched(&pipeline(), &c, &identity, c.len());
    assert_eq!(
        label_mismatches(&s_labels, &labels_of(&b)),
        Vec::<usize>::new()
    );
    assert!(scores_close(&s, &b));

    // 並べ替え × チャンクサイズ。
    for seed in [1u64, 2, 3] {
        let order = shuffled(c.len(), seed);
        for chunk in [1usize, 7, 64, 650] {
            let b = batched(&pipeline(), &c, &order, chunk);
            assert_eq!(
                label_mismatches(&s_labels, &labels_of(&b)),
                Vec::<usize>::new(),
                "seed={seed} chunk={chunk}"
            );
            assert!(scores_close(&s, &b), "seed={seed} chunk={chunk}");
        }
    }

    // 評価器内の推論経路。
    let e = via_evaluator(pipeline(), &c);
    assert_eq!(e.len(), 650);
    assert_eq!(
        label_mismatches(&s_labels, &labels_of(&e)),
        Vec::<usize>::new()
    );
    assert!(scores_close(&s, &e));
}

/// REQ-28・TASK-28.1-2・#118。全ラベルが出現し、一致が全件同一ラベルによる自明な一致でない
/// こと（証拠種別: テストハーネス）。
#[test]
fn req28_corpus_labels_cover_all_classes() {
    let s_labels = labels_of(&single(&pipeline(), &corpus()));
    for l in 0..LABELS {
        let n = s_labels.iter().filter(|&&x| x == l).count();
        assert!(n > 0, "label {l} never predicted");
    }
}

/// REQ-28・TASK-28.1-2・#118。陰性対照: 素朴なパディング付きバッチは同じ比較器で不一致が
/// 検出される（比較器が常に 0 を返す空振りの防止。証拠種別: テストハーネス）。
#[test]
fn req28_negative_control_padded_batch_detected_by_same_comparator() {
    let c = corpus();
    let s_labels = labels_of(&single(&pipeline(), &c));
    let naive = naive_padded_labels(&c);
    let mm = label_mismatches(&s_labels, &naive);
    // 496 は模擬バックエンド上の観測値（実モデルの値ではない）。
    assert_eq!(mm.len(), 496);
}

// --- Qwen の KV 使い回し採点（REQ-41・TASK-41.7-7・#421）。証拠種別: テストハーネス（合成重み・極小構成）---

/// 参照順伝播用の前処理（バイト列を語彙内の id へ。実トークナイザーは `tokenizer_golden.rs` が担う）。
struct QwenPre;
impl Preprocessor for QwenPre {
    fn preprocess(&self, input: &str) -> Result<TokenIds, PreprocessError> {
        Ok(TokenIds::new(bytes_to_ids(input)))
    }
}

fn qwen_pipeline() -> InferencePipeline<QwenPre, QwenRefBackend> {
    InferencePipeline::new(QwenPre, QwenRefBackend)
}

/// REQ-28・REQ-41・#421。Qwen の KV 使い回し採点でも全 650 件で 3 経路の予測ラベルが一致し
/// 不一致 0 件（`req28_single_batch_evaluator_labels_match_all_650` と同じ形。証拠種別: テストハーネス）。
#[test]
fn req28_req41_qwen_kv_scoring_single_batch_evaluator_match_all_650() {
    let c = corpus();
    let s = single(&qwen_pipeline(), &c);
    let s_labels = labels_of(&s);
    assert_eq!(s_labels.len(), 650);

    let identity: Vec<usize> = (0..c.len()).collect();
    let b = batched(&qwen_pipeline(), &c, &identity, c.len());
    assert_eq!(
        label_mismatches(&s_labels, &labels_of(&b)),
        Vec::<usize>::new()
    );
    assert!(scores_close(&s, &b));

    // 1 周が重い（debug で数秒）ため、seed 1 のみ・恒等順の全件チャンクは上で確認済み。
    for seed in [1u64] {
        let order = shuffled(c.len(), seed);
        for chunk in [1usize, 7, 64] {
            let b = batched(&qwen_pipeline(), &c, &order, chunk);
            assert_eq!(
                label_mismatches(&s_labels, &labels_of(&b)),
                Vec::<usize>::new(),
                "seed={seed} chunk={chunk}"
            );
            assert!(scores_close(&s, &b), "seed={seed} chunk={chunk}");
        }
    }

    let e = via_evaluator(qwen_pipeline(), &c);
    assert_eq!(
        label_mismatches(&s_labels, &labels_of(&e)),
        Vec::<usize>::new()
    );
    assert!(scores_close(&s, &e));
}

/// REQ-28・REQ-41・#421。全選択肢が 1 回以上予測され、全件同一ラベルによる自明な一致でないこと。
#[test]
fn req41_qwen_kv_scoring_labels_cover_all_choices() {
    let s_labels = labels_of(&single(&qwen_pipeline(), &corpus()));
    for l in 0..qwen_scoring::K {
        let n = s_labels.iter().filter(|&&x| x == l).count();
        assert!(n > 0, "choice {l} never predicted");
    }
}

/// REQ-41・#421。KV 使い回しの採点が、選択肢ごとに全系列を流す素朴な採点と一致する
/// （マスク・位置・行のずらしの正しさの照合）。確率差の上限 1e-5 は f32 の計算順の違い
/// （prefix の K・V を共有するか全系列で再計算するか）に由来する差の目安で、f64 用の 1e-9 は使わない。
#[test]
fn req41_kv_reuse_matches_naive_full_sequence_scoring() {
    for (i, x) in corpus().iter().take(64).enumerate() {
        let ids = bytes_to_ids(x);
        let a = qwen_scoring::kv_reuse_scores(&ids).expect("kv reuse");
        let b = qwen_scoring::naive_full_scores(&ids).expect("naive");
        assert_eq!(a.len(), b.len());
        let argmax = |v: &[f64]| (0..v.len()).max_by(|&p, &q| v[p].total_cmp(&v[q]));
        assert_eq!(argmax(&a), argmax(&b), "label differs at index {i}");
        let worst = a
            .iter()
            .zip(&b)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0, f64::max);
        assert!(worst <= 1e-5, "prob diff {worst} at index {i}");
    }
}

/// REQ-28・REQ-41・#421。陰性対照: チャンク内の最大長まで pad の id で右詰めしマスクなしで採点した
/// 素朴なバッチは、同じ比較器で不一致が検出される（比較器の空振り防止。証拠種別: テストハーネス）。
#[test]
fn req28_qwen_negative_control_padded_batch_detected() {
    let c = corpus();
    let s_labels = labels_of(&single(&qwen_pipeline(), &c));
    let mut naive = Vec::with_capacity(c.len());
    for chunk in c.chunks(64) {
        let seqs: Vec<Vec<i64>> = chunk.iter().map(|x| bytes_to_ids(x)).collect();
        let max = seqs.iter().map(Vec::len).max().unwrap_or(0);
        for mut s in seqs {
            s.resize(max, qwen_scoring::PAD);
            let p = qwen_scoring::kv_reuse_scores(&s).expect("padded");
            let best = (0..p.len())
                .max_by(|&a, &b| p[a].total_cmp(&p[b]))
                .unwrap_or(0);
            naive.push(best);
        }
    }
    let mm = label_mismatches(&s_labels, &naive);
    // 模擬上の観測値（実モデルの値ではない）。
    assert_eq!(mm.len(), 119);
}
