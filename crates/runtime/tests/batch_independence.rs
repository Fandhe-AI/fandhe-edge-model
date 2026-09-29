//! 単体/バッチ共通経路の結合テスト（REQ-28・TASK-28.1-1・#117）。
//! 証拠種別: テストハーネス（模擬の前処理・バックエンド。実前処理は #112、ONNX は #113）。

use fandhe_edge_core::infer_input::MAX_INFER_INPUT_BYTES;
use fandhe_edge_runtime::pipeline::*;
use std::cell::RefCell;
use std::rc::Rc;

const TOL: f64 = 1e-9;
const LABELS: usize = 4;

/// UTF-8 バイト + 1、空なら [0]（実前処理の代用。NFKC は行わない）。
struct TestPreprocessor;
fn test_ids(input: &str) -> TokenIds {
    if input.is_empty() {
        return TokenIds::new(vec![0]);
    }
    TokenIds::new(input.bytes().map(|b| i64::from(b) + 1).collect())
}
impl Preprocessor for TestPreprocessor {
    fn preprocess(&self, input: &str) -> Result<TokenIds, PreprocessError> {
        Ok(test_ids(input))
    }
}

/// 系列長・位置・隣接要素に敏感な決定的スコア（幅 3 の窓和）。
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

struct WindowBackend;
impl ScoringBackend for WindowBackend {
    fn scores(&self, ids: &TokenIds) -> Result<Vec<f64>, BackendError> {
        Ok(window_scores(ids.as_slice()))
    }
}

struct SpyPre(Rc<RefCell<usize>>);
impl Preprocessor for SpyPre {
    fn preprocess(&self, input: &str) -> Result<TokenIds, PreprocessError> {
        *self.0.borrow_mut() += 1;
        Ok(test_ids(input))
    }
}
struct SpyBackend(Rc<RefCell<Vec<Vec<i64>>>>);
impl ScoringBackend for SpyBackend {
    fn scores(&self, ids: &TokenIds) -> Result<Vec<f64>, BackendError> {
        self.0.borrow_mut().push(ids.as_slice().to_vec());
        Ok(window_scores(ids.as_slice()))
    }
}

fn pipeline() -> InferencePipeline<TestPreprocessor, WindowBackend> {
    InferencePipeline::new(TestPreprocessor, WindowBackend)
}

fn same(a: &Prediction, b: &Prediction) -> bool {
    a.label_index() == b.label_index()
        && a.scores().len() == b.scores().len()
        && a.scores()
            .iter()
            .zip(b.scores())
            .all(|(x, y)| (x - y).abs() <= TOL)
}

fn corpus() -> Vec<String> {
    vec![
        String::new(),
        " ".into(),
        "a".into(),
        "こんにちは世界".into(),
        "abc".into(),
        "abc".into(),
        "long input ".repeat(200),
        "zzzzzzzzzz".into(),
        "0123456789".into(),
    ]
}

/// 固定 seed の LCG による決定的な置換。
fn shuffled(mut v: Vec<String>, mut seed: u64) -> Vec<String> {
    for i in (1..v.len()).rev() {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        v.swap(i, ((seed >> 33) as usize) % (i + 1));
    }
    v
}

/// REQ-28: バッチの各要素が単体推論と一致する（不一致 0 件）。
#[test]
fn req28_batch_matches_single_for_every_element() {
    let p = pipeline();
    let inputs = corpus();
    let refs: Vec<&str> = inputs.iter().map(String::as_str).collect();
    let batch = p.infer_batch(&refs).unwrap();
    assert_eq!(batch.len(), refs.len());
    let mismatches = refs
        .iter()
        .zip(&batch)
        .filter(|(x, b)| !same(&p.infer_one(x).unwrap(), b.as_ref().unwrap()))
        .count();
    assert_eq!(mismatches, 0);
}

/// REQ-28: 同居要素の件数・長さ・順序・重複を変えても対象の予測が変わらない。
#[test]
fn req28_target_prediction_invariant_to_companions() {
    let p = pipeline();
    let target = "target input";
    let expected = p.infer_one(target).unwrap();
    let filler = corpus();
    for n in [0usize, 1, 2, 16, 127] {
        for seed in [1u64, 2, 3] {
            let mut items: Vec<String> = (0..n)
                .map(|i| filler.get(i % filler.len()).cloned().unwrap_or_default())
                .collect();
            items.push(target.to_string());
            items.push(target.to_string());
            let items = shuffled(items, seed);
            let refs: Vec<&str> = items.iter().map(String::as_str).collect();
            let out = p.infer_batch(&refs).unwrap();
            let mut found = 0;
            for (x, r) in refs.iter().zip(&out) {
                if *x == target {
                    found += 1;
                    assert!(same(&expected, r.as_ref().unwrap()));
                }
            }
            assert_eq!(found, 2);
        }
    }
}

/// REQ-28: 前処理・バックエンドは 1 件につきちょうど 1 回、パディングなしの列で呼ばれる。
#[test]
fn req28_batch_calls_shared_path_once_per_item() {
    let pre_calls = Rc::new(RefCell::new(0usize));
    let seen = Rc::new(RefCell::new(Vec::new()));
    let p = InferencePipeline::new(SpyPre(pre_calls.clone()), SpyBackend(seen.clone()));
    let inputs = corpus();
    let refs: Vec<&str> = inputs.iter().map(String::as_str).collect();
    p.infer_batch(&refs).unwrap();
    assert_eq!(*pre_calls.borrow(), refs.len());
    let seen = seen.borrow();
    assert_eq!(seen.len(), refs.len());
    for (x, ids) in refs.iter().zip(seen.iter()) {
        assert_eq!(ids.as_slice(), test_ids(x).as_slice());
    }
}

/// REQ-28: 呼び出しを跨いで状態を持ち回らない。
#[test]
fn req28_no_state_carried_across_calls() {
    let p = pipeline();
    let first = p.infer_one("x").unwrap();
    let _ = p.infer_batch(&["long long long", "", "y"]).unwrap();
    assert!(same(&first, &p.infer_one("x").unwrap()));
}

/// REQ-28・REQ-39: 上限超過の要素だけが失敗し、他要素へ波及しない。
#[test]
fn req28_item_error_does_not_affect_neighbors() {
    let p = pipeline();
    let big = "a".repeat(MAX_INFER_INPUT_BYTES + 1);
    let out = p.infer_batch(&["ok1", &big, "ok2"]).unwrap();
    assert_eq!(
        out.get(1).unwrap().as_ref().unwrap_err().code(),
        "input_too_large"
    );
    assert!(same(
        &p.infer_one("ok1").unwrap(),
        out.first().unwrap().as_ref().unwrap()
    ));
    assert!(same(
        &p.infer_one("ok2").unwrap(),
        out.get(2).unwrap().as_ref().unwrap()
    ));
}

/// REQ-28 陰性対照: 素朴なパディング付きバッチ実装を同じ照合で検出できる（照合の空振り防止）。
#[test]
fn req28_negative_control_padded_batch_is_detected() {
    let p = pipeline();
    let inputs = corpus();
    let ids: Vec<Vec<i64>> = inputs
        .iter()
        .map(|s| test_ids(s).as_slice().to_vec())
        .collect();
    let max = ids.iter().map(Vec::len).max().unwrap_or(0);
    let mismatches = inputs
        .iter()
        .zip(&ids)
        .filter(|(s, v)| {
            let mut padded = (*v).clone();
            padded.resize(max, 0);
            let naive = window_scores(&padded);
            let single = p.infer_one(s).unwrap();
            single
                .scores()
                .iter()
                .zip(&naive)
                .any(|(a, b)| (a - b).abs() > TOL)
        })
        .count();
    assert!(mismatches >= 1);
}

/// REQ-39: バッチ件数上限超過は前処理より前に拒否する。
#[test]
fn batch_len_limit_rejected_before_allocation() {
    let pre_calls = Rc::new(RefCell::new(0usize));
    let p = InferencePipeline::new(SpyPre(pre_calls.clone()), WindowBackend);
    let inputs = vec![""; MAX_INFER_BATCH_LEN + 1];
    let err = p.infer_batch(&inputs).unwrap_err();
    assert_eq!(err.code(), "too_many_inputs");
    assert_eq!(*pre_calls.borrow(), 0);
}
/// REQ-39: 総入力バイト数が上限を超えるバッチは、処理前（前処理を 1 回も呼ばず）に拒否する。
#[test]
fn req39_batch_total_bytes_rejected_before_processing() {
    let calls = Rc::new(RefCell::new(0usize));
    let p = InferencePipeline::new(SpyPre(calls.clone()), WindowBackend);
    let chunk = "a".repeat(MAX_INFER_INPUT_BYTES);
    let n = MAX_INFER_BATCH_TOTAL_BYTES / MAX_INFER_INPUT_BYTES + 1;
    let inputs: Vec<&str> = vec![chunk.as_str(); n];
    let err = p.infer_batch(&inputs).unwrap_err();
    assert_eq!(
        err,
        BatchError::TotalInputTooLarge {
            total: n * MAX_INFER_INPUT_BYTES,
            limit: MAX_INFER_BATCH_TOTAL_BYTES
        }
    );
    assert_eq!(err.code(), "total_input_too_large");
    assert_eq!(*calls.borrow(), 0);
}

struct HugeTokensPre;
impl Preprocessor for HugeTokensPre {
    fn preprocess(&self, _input: &str) -> Result<TokenIds, PreprocessError> {
        Ok(TokenIds::new(vec![1; MAX_INFER_TOKEN_IDS + 1]))
    }
}

/// REQ-39: トークン数が上限を超える場合、バックエンドを呼ばずに拒否する。
#[test]
fn req39_token_count_limit_rejected_before_backend() {
    let seen = Rc::new(RefCell::new(Vec::new()));
    let p = InferencePipeline::new(HugeTokensPre, SpyBackend(seen.clone()));
    let err = p.infer_one("x").unwrap_err();
    assert_eq!(
        err,
        InferError::TooManyTokens {
            len: MAX_INFER_TOKEN_IDS + 1,
            limit: MAX_INFER_TOKEN_IDS
        }
    );
    assert!(seen.borrow().is_empty());
}

struct FixedBackend(Vec<f64>);
impl ScoringBackend for FixedBackend {
    fn scores(&self, _ids: &TokenIds) -> Result<Vec<f64>, BackendError> {
        Ok(self.0.clone())
    }
}

/// REQ-28: 確率として不正なスコア（範囲外・合計不正）は成功扱いにしない。
#[test]
fn req28_invalid_probability_scores_rejected() {
    for bad in [vec![-0.5, 1.5], vec![0.9, 0.9], vec![0.2, 0.2]] {
        let p = InferencePipeline::new(TestPreprocessor, FixedBackend(bad));
        assert_eq!(p.infer_one("x").unwrap_err(), InferError::InvalidScores);
    }
}

/// REQ-39: スコア数が上限を超えるバックエンド出力は拒否する。
#[test]
fn req39_too_many_scores_rejected() {
    let n = MAX_INFER_SCORES + 1;
    let p = InferencePipeline::new(TestPreprocessor, FixedBackend(vec![1.0 / n as f64; n]));
    assert_eq!(
        p.infer_one("x").unwrap_err(),
        InferError::TooManyScores {
            len: n,
            limit: MAX_INFER_SCORES
        }
    );
}

/// REQ-39: 結果スコアの総数が上限を超えたら、バッチ全体を失敗とする。
#[test]
fn req39_batch_result_retention_limited() {
    let k = MAX_INFER_SCORES;
    let p = InferencePipeline::new(TestPreprocessor, FixedBackend(vec![1.0 / k as f64; k]));
    let n = MAX_INFER_BATCH_TOTAL_SCORES / k + 1;
    let inputs = vec![""; n];
    assert_eq!(
        p.infer_batch(&inputs).unwrap_err(),
        BatchError::ResultTooLarge {
            limit: MAX_INFER_BATCH_TOTAL_SCORES
        }
    );
}
