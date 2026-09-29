//! `infer --input-file` の一括推論と 1 行 1 JSON 出力（REQ-33 の例外。TASK-33.4・#141）の
//! 結合テスト。
//!
//! 証拠種別: テストハーネス（模擬の前処理・バックエンド。バイナリでの完走は #136、実前処理は
//! #112、ONNX は #113）。

use fandhe_edge_cli::infer_batch::{
    BatchLimits, emit_infer_batch, emit_infer_batch_with_limits, judgment_from_prediction,
};
use fandhe_edge_cli::output::write_ok_judgment;
use fandhe_edge_core::definition::Definition;
use fandhe_edge_core::exitcode::ExitCode;
use fandhe_edge_core::infer_input::MAX_INFER_INPUT_BYTES;
use fandhe_edge_runtime::pipeline::{
    BackendError, InferencePipeline, MAX_INFER_BATCH_LEN, PreprocessError, Preprocessor,
    ScoringBackend, TokenIds,
};
use std::io::{self, Cursor, Read, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

const DEFINITION_JSON: &str = r#"{
  "schema": "fandhe-edge-model-definition/v1",
  "name": "infer-batch-output-test",
  "version": 1,
  "judgment_type": "single_select",
  "options": [
    {"id": "a", "display_name": "A", "description": "a"},
    {"id": "b", "display_name": "B", "description": "b"},
    {"id": "c", "display_name": "C", "description": "c"}
  ],
  "io": {"input": "bytes"}
}"#;

fn definition() -> Definition {
    Definition::parse(DEFINITION_JSON).expect("valid definition")
}

/// 入力の文字列で固定スコアを決める模擬。前処理は入力をそのまま保持する。
struct Pre(Arc<Mutex<Vec<String>>>);
impl Preprocessor for Pre {
    fn preprocess(&self, input: &str) -> Result<TokenIds, PreprocessError> {
        self.0.lock().unwrap().push(input.to_string());
        Ok(TokenIds::new(input.bytes().map(i64::from).collect()))
    }
}

/// 先頭バイトが `a` なら a 優勢、`b` なら b 優勢、`f` ならバックエンド失敗、
/// `w` なら選択肢数と合わないスコア長を返す。
struct Backend;
impl ScoringBackend for Backend {
    fn scores(&self, ids: &TokenIds) -> Result<Vec<f64>, BackendError> {
        match ids.as_slice().first().copied() {
            Some(102) => Err(BackendError::Failed),
            Some(119) => Ok(vec![0.25, 0.25, 0.25, 0.25]),
            Some(98) => Ok(vec![0.25, 0.5, 0.25]),
            _ => Ok(vec![0.5, 0.25, 0.25]),
        }
    }
}

type Seen = Arc<Mutex<Vec<String>>>;

fn pipeline() -> (Arc<InferencePipeline<Pre, Backend>>, Seen) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    (
        Arc::new(InferencePipeline::new(Pre(Arc::clone(&seen)), Backend)),
        seen,
    )
}

fn run(input: &[u8]) -> (ExitCode, String) {
    let definition = definition();
    let (pipeline, _) = pipeline();
    let mut out: Vec<u8> = Vec::new();
    let code = emit_infer_batch(
        &mut out,
        Cursor::new(input.to_vec()),
        definition.io(),
        definition.options(),
        pipeline,
    )
    .expect("write must succeed");
    (code, String::from_utf8(out).expect("utf-8 output"))
}

fn assert_single_error(code: ExitCode, text: &str, expected: ExitCode, name: &str) {
    assert_eq!(code, expected);
    assert_eq!(text.matches('\n').count(), 1, "exactly one line: {text}");
    assert!(
        text.starts_with(&format!("{{\"code\":\"{name}\"")),
        "unexpected error line: {text}"
    );
}

/// REQ-33: 3 レコードは 3 行・入力順・id 保存で出る。
#[test]
fn req33_batch_three_records_emit_three_json_lines() {
    let (code, text) = run(b"{\"id\":\"r1\",\"input\":\"a1\"}\n{\"id\":\"r2\",\"input\":\"b2\"}\n{\"id\":\"r3\",\"input\":\"c3\"}\n");
    assert_eq!(code, ExitCode::Ok);
    assert_eq!(
        text,
        "{\"id\":\"r1\",\"status\":\"ok\",\"predicted_label\":\"a\",\"scores\":{\"a\":0.5,\"b\":0.25,\"c\":0.25}}\n\
         {\"id\":\"r2\",\"status\":\"ok\",\"predicted_label\":\"b\",\"scores\":{\"a\":0.25,\"b\":0.5,\"c\":0.25}}\n\
         {\"id\":\"r3\",\"status\":\"ok\",\"predicted_label\":\"a\",\"scores\":{\"a\":0.5,\"b\":0.25,\"c\":0.25}}\n"
    );
}

/// REQ-33: 1 レコードは 1 行。
#[test]
fn req33_batch_single_record_emits_one_line() {
    let (code, text) = run(b"{\"id\":\"only\",\"input\":\"b\"}");
    assert_eq!(code, ExitCode::Ok);
    assert_eq!(
        text,
        "{\"id\":\"only\",\"status\":\"ok\",\"predicted_label\":\"b\",\"scores\":{\"a\":0.25,\"b\":0.5,\"c\":0.25}}\n"
    );
}

/// REQ-33: 空行・空白行を読み飛ばし、CRLF も受理する。
#[test]
fn req33_batch_skips_blank_lines_and_accepts_crlf() {
    let (code, text) =
        run(b"\r\n{\"id\":\"x\",\"input\":\"a\"}\r\n   \r\n\n{\"id\":\"y\",\"input\":\"b\"}\r\n");
    assert_eq!(code, ExitCode::Ok);
    let ids: Vec<&str> = text
        .lines()
        .map(|l| if l.contains("\"id\":\"x\"") { "x" } else { "y" })
        .collect();
    assert_eq!(ids, ["x", "y"]);
}

/// REQ-21: 有効レコード 0 件は成功を装わず `invalid_input`。
#[test]
fn req33_batch_empty_input_is_invalid_input() {
    for input in [&b""[..], b"\n  \n"] {
        let (code, text) = run(input);
        assert_single_error(code, &text, ExitCode::InvalidInput, "invalid_input");
    }
}

/// REQ-21: 途中の不正レコードは結果行を 1 行も出さずエラー 1 行のみ。
#[test]
fn req21_batch_malformed_record_emits_single_error_and_no_rows() {
    let (code, text) =
        run(b"{\"id\":\"r1\",\"input\":\"a\"}\nnot json\n{\"id\":\"r3\",\"input\":\"a\"}\n");
    assert_single_error(code, &text, ExitCode::InvalidInput, "invalid_input");
    assert!(!text.contains("predicted_label"));
}

/// REQ-39: 1 行が上限超過なら `limit_exceeded`。
#[test]
fn req39_batch_oversized_record_is_limit_exceeded() {
    let big = "a".repeat(MAX_INFER_INPUT_BYTES + 1);
    let line = format!("{{\"id\":\"big\",\"input\":\"{big}\"}}\n");
    let (code, text) = run(line.as_bytes());
    assert_single_error(code, &text, ExitCode::LimitExceeded, "limit_exceeded");
}

/// REQ-39: 入力全体が上限超過なら上限 + 1 バイトで読み取りを止め `limit_exceeded`。
#[test]
fn req39_batch_input_over_byte_cap_is_limit_exceeded() {
    struct Endless(Arc<AtomicUsize>);
    impl Read for Endless {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            buf.fill(b'\n');
            self.0.fetch_add(buf.len(), Ordering::SeqCst);
            Ok(buf.len())
        }
    }
    let read = Arc::new(AtomicUsize::new(0));
    let definition = definition();
    let (pipeline, _) = pipeline();
    let mut out: Vec<u8> = Vec::new();
    let code = emit_infer_batch(
        &mut out,
        Endless(Arc::clone(&read)),
        definition.io(),
        definition.options(),
        pipeline,
    )
    .unwrap();
    let text = String::from_utf8(out).unwrap();
    assert_single_error(code, &text, ExitCode::LimitExceeded, "limit_exceeded");
    let limit = fandhe_edge_runtime::pipeline::MAX_INFER_BATCH_TOTAL_BYTES;
    assert_eq!(read.load(Ordering::SeqCst), limit + 1);
}

/// REQ-39: 改行の無い無限入力は 1 行の上限付近で読み取りを止め `limit_exceeded`
/// （入力全体をメモリへ読み込んでから検証しない）。
#[test]
fn req39_batch_endless_single_line_stops_reading_near_line_cap() {
    struct Endless(Arc<AtomicUsize>);
    impl Read for Endless {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            buf.fill(b' ');
            self.0.fetch_add(buf.len(), Ordering::SeqCst);
            Ok(buf.len())
        }
    }
    let read = Arc::new(AtomicUsize::new(0));
    let definition = definition();
    let (pipeline, _) = pipeline();
    let mut out: Vec<u8> = Vec::new();
    let code = emit_infer_batch(
        &mut out,
        Endless(Arc::clone(&read)),
        definition.io(),
        definition.options(),
        pipeline,
    )
    .unwrap();
    let text = String::from_utf8(out).unwrap();
    assert_single_error(code, &text, ExitCode::LimitExceeded, "limit_exceeded");
    // 1 行の上限 + 2 バイトと BufReader の 1 回分（8 KiB）を超えて読まない。
    assert!(read.load(Ordering::SeqCst) <= MAX_INFER_INPUT_BYTES + 2 + 8192);
}

/// REQ-39: 件数が上限 + 1 なら `limit_exceeded`。
#[test]
fn req39_batch_record_count_over_limit_is_limit_exceeded() {
    let line = "{\"id\":\"i\",\"input\":\"a\"}\n";
    let input = line.repeat(MAX_INFER_BATCH_LEN + 1);
    let (code, text) = run(input.as_bytes());
    assert_single_error(code, &text, ExitCode::LimitExceeded, "limit_exceeded");
}

/// REQ-21: 途中 1 件のバックエンド失敗は結果行なしの `runtime_error` 1 行。
#[test]
fn req21_batch_backend_failure_emits_single_runtime_error_and_no_rows() {
    let (code, text) = run(b"{\"id\":\"r1\",\"input\":\"a\"}\n{\"id\":\"r2\",\"input\":\"f\"}\n");
    assert_single_error(code, &text, ExitCode::RuntimeError, "runtime_error");
}

/// REQ-21: スコア長が選択肢数と合わない場合は `runtime_error` ではなく判定検証の写像に従い
/// 結果行を出さない。
#[test]
fn req21_batch_score_length_mismatch_emits_single_error_and_no_rows() {
    let (code, text) = run(b"{\"id\":\"r1\",\"input\":\"w\"}\n");
    assert_single_error(code, &text, ExitCode::RuntimeError, "runtime_error");
}

/// REQ-21: 予測 index が選択肢範囲外なら `runtime_error`。
#[test]
fn req21_label_index_out_of_options_is_runtime_error() {
    struct Wide;
    impl ScoringBackend for Wide {
        fn scores(&self, _ids: &TokenIds) -> Result<Vec<f64>, BackendError> {
            Ok(vec![0.0, 0.0, 0.0, 1.0])
        }
    }
    let definition = definition();
    let pre = Pre(Arc::new(Mutex::new(Vec::new())));
    let pipeline = InferencePipeline::new(pre, Wide);
    let prediction = pipeline.infer_one("x").unwrap();
    let error = judgment_from_prediction(definition.options(), "id", &prediction).unwrap_err();
    assert_eq!(error.code, ExitCode::RuntimeError);
}

/// REQ-21: 非 UTF-8 は `invalid_input`。
#[test]
fn req21_batch_non_utf8_is_invalid_input() {
    let (code, text) = run(&[0xff, 0xfe, b'\n']);
    assert_single_error(code, &text, ExitCode::InvalidInput, "invalid_input");
}

/// 一部だけ書いて失敗する Write（`output.rs` テストと同型）。
struct FailAfterFirstLine {
    calls: usize,
    flush_calls: usize,
}
impl Write for FailAfterFirstLine {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.calls += 1;
        if self.calls == 1 {
            Ok(buf.len())
        } else {
            Err(io::Error::other("simulated"))
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        self.flush_calls += 1;
        Ok(())
    }
}

/// REQ-21: 書き込み失敗後は打ち切り、以降の write（残り行・エラー行）をしない。
#[test]
fn req21_batch_stops_after_write_failure() {
    let definition = definition();
    let (pipeline, _) = pipeline();
    let mut out = FailAfterFirstLine {
        calls: 0,
        flush_calls: 0,
    };
    let input = b"{\"id\":\"1\",\"input\":\"a\"}\n{\"id\":\"2\",\"input\":\"a\"}\n{\"id\":\"3\",\"input\":\"a\"}\n";
    let result = emit_infer_batch(
        &mut out,
        Cursor::new(input.to_vec()),
        definition.io(),
        definition.options(),
        pipeline,
    );
    assert!(result.is_err());
    assert_eq!(out.calls, 2, "no write after the failing one");
    assert_eq!(out.flush_calls, 1, "flush only after the successful write");
}

/// REQ-28: バッチ出力は、同じ模擬で 1 件ずつ推論した行の連結とバイト一致する。
#[test]
fn req28_batch_lines_match_single_inference_lines() {
    let definition = definition();
    let inputs = ["a1", "b2", "c3", "b"];
    let mut jsonl = String::new();
    for (i, input) in inputs.iter().enumerate() {
        jsonl.push_str(&format!("{{\"id\":\"r{i}\",\"input\":\"{input}\"}}\n"));
    }
    let (_, batch_text) = run(jsonl.as_bytes());

    let (pipeline, _) = pipeline();
    let mut single: Vec<u8> = Vec::new();
    for (i, input) in inputs.iter().enumerate() {
        let prediction = pipeline.infer_one(input).unwrap();
        let result =
            judgment_from_prediction(definition.options(), &format!("r{i}"), &prediction).unwrap();
        write_ok_judgment(&mut single, &result).unwrap();
    }
    assert_eq!(batch_text.as_bytes(), single.as_slice());
}

/// REQ-27: 推論側へ渡るのは `input` の値だけで、`id` は渡らない。
#[test]
fn req27_batch_passes_only_input_to_pipeline() {
    let definition = definition();
    let (pipeline, seen) = pipeline();
    let mut out: Vec<u8> = Vec::new();
    emit_infer_batch(
        &mut out,
        Cursor::new(
            b"{\"id\":\"secret-id-1\",\"input\":\"alpha\"}\n{\"id\":\"secret-id-2\",\"input\":\"beta\"}\n".to_vec(),
        ),
        definition.io(),
        definition.options(),
        pipeline,
    )
    .unwrap();
    assert_eq!(
        *seen.lock().unwrap(),
        ["alpha".to_string(), "beta".to_string()]
    );
}

/// REQ-39: 改行が来ないまま停止する `Read` でも、期限で `limit_exceeded` を返して戻る
/// （読み取りが期限後も戻らないことはない）。
#[test]
fn req39_batch_stalled_reader_returns_limit_exceeded_at_deadline() {
    struct Stalled;
    impl Read for Stalled {
        fn read(&mut self, _buf: &mut [u8]) -> io::Result<usize> {
            std::thread::sleep(Duration::from_secs(30));
            Ok(0)
        }
    }
    let definition = definition();
    let (pipeline, _) = pipeline();
    let mut out: Vec<u8> = Vec::new();
    let started = std::time::Instant::now();
    let code = emit_infer_batch_with_limits(
        &mut out,
        Stalled,
        definition.io(),
        definition.options(),
        pipeline,
        BatchLimits {
            duration: Duration::from_millis(200),
            ..BatchLimits::default()
        },
    )
    .unwrap();
    assert!(started.elapsed() < Duration::from_secs(10));
    let text = String::from_utf8(out).unwrap();
    assert_single_error(code, &text, ExitCode::LimitExceeded, "limit_exceeded");
}

/// REQ-39: 期限を確認できない 1 件の推論が停止しても、期限で `limit_exceeded` を返して戻る。
#[test]
fn req39_batch_stalled_inference_returns_limit_exceeded_at_deadline() {
    struct Stalled;
    impl ScoringBackend for Stalled {
        fn scores(&self, _ids: &TokenIds) -> Result<Vec<f64>, BackendError> {
            std::thread::sleep(Duration::from_secs(30));
            Ok(vec![0.5, 0.25, 0.25])
        }
    }
    let definition = definition();
    let pipeline = Arc::new(InferencePipeline::new(
        Pre(Arc::new(Mutex::new(Vec::new()))),
        Stalled,
    ));
    let mut out: Vec<u8> = Vec::new();
    let started = std::time::Instant::now();
    let code = emit_infer_batch_with_limits(
        &mut out,
        Cursor::new(b"{\"id\":\"r1\",\"input\":\"a\"}\n".to_vec()),
        definition.io(),
        definition.options(),
        pipeline,
        BatchLimits {
            duration: Duration::from_millis(200),
            ..BatchLimits::default()
        },
    )
    .unwrap();
    assert!(started.elapsed() < Duration::from_secs(10));
    let text = String::from_utf8(out).unwrap();
    assert_single_error(code, &text, ExitCode::LimitExceeded, "limit_exceeded");
}

/// REQ-39: 総出力バイト数が上限を超えるなら、結果行を 1 行も出さず `limit_exceeded`。
/// 上限ちょうどは受理する。
#[test]
fn req39_batch_total_output_over_limit_is_limit_exceeded_before_any_row() {
    let definition = definition();
    let input = b"{\"id\":\"r1\",\"input\":\"a\"}\n{\"id\":\"r2\",\"input\":\"a\"}\n";
    let (_, full) = run(input);
    let run_with = |output_bytes: usize| {
        let (pipeline, _) = pipeline();
        let mut out: Vec<u8> = Vec::new();
        let code = emit_infer_batch_with_limits(
            &mut out,
            Cursor::new(input.to_vec()),
            definition.io(),
            definition.options(),
            pipeline,
            BatchLimits {
                output_bytes,
                ..BatchLimits::default()
            },
        )
        .unwrap();
        (code, String::from_utf8(out).unwrap())
    };
    let (code, text) = run_with(full.len());
    assert_eq!(code, ExitCode::Ok);
    assert_eq!(text, full);
    let (code, text) = run_with(full.len() - 1);
    assert_single_error(code, &text, ExitCode::LimitExceeded, "limit_exceeded");
    assert!(!text.contains("predicted_label"));
}

/// REQ-39: 出力段階の期限を超えたら、残りの行もエラー行も書かず `io::Error`（TimedOut）。
#[test]
fn req39_batch_output_deadline_stops_writing() {
    let definition = definition();
    let (pipeline, _) = pipeline();
    let mut out: Vec<u8> = Vec::new();
    let err = emit_infer_batch_with_limits(
        &mut out,
        Cursor::new(b"{\"id\":\"r1\",\"input\":\"a\"}\n".to_vec()),
        definition.io(),
        definition.options(),
        pipeline,
        BatchLimits {
            output_duration: Duration::ZERO,
            ..BatchLimits::default()
        },
    )
    .unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::TimedOut);
    assert!(out.is_empty());
}
