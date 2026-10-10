//! `write_ok_judgment`（TASK-21.1-2・REQ-21 正常系）の結合テスト。
//!
//! 定義ファイル（`Definition::parse`）の選択肢一覧から
//! `fandhe_edge_core::judgment::JudgmentResult` を組み立て、
//! `fandhe_edge_cli::output::write_ok_judgment` で書いた JSON が
//! 期待どおり（1 行・宣言順・改行 1 つ）であることを確認する。
//!
//! 本テストは `fandhe-edge-cli` の lib ターゲット（`fandhe_edge_cli`）を直
//! 接呼ぶ。CLI バイナリ（`fandhe-edge`）の 7 工程への接続（TASK-33.1-2・#136）とは独立に、
//! この出力契約だけを確認する。

use fandhe_edge_cli::output::write_ok_judgment;
use fandhe_edge_core::definition::Definition;
use fandhe_edge_core::exitcode::ExitCode;
use fandhe_edge_core::judgment::JudgmentResult;

/// テスト内で完結させる最小の定義ファイル JSON（`docs/spec` は読まない。
/// spec-reference.md「ビルド・テストは docs/spec 抜きで成立させる」）。
const DEFINITION_JSON: &str = r#"{
  "schema": "fandhe-edge-model-definition/v1",
  "name": "ok-judgment-output-test",
  "version": 1,
  "judgment_type": "single_select",
  "options": [
    {"id": "reject", "display_name": "Reject", "description": "reject the input"},
    {"id": "accept", "display_name": "Accept", "description": "accept the input"},
    {"id": "review", "display_name": "Review", "description": "needs human review"}
  ],
  "io": {"input": "bytes"}
}"#;

/// REQ-21・TASK-21.1-2: 判定成功時、JSON 1 行＋改行 1 つのみが書かれ、
/// `predicted_label`／`scores` のキー順が定義ファイルの `options` 宣言順
/// （reject, accept, review）と一致し、戻り値が `ExitCode::Ok` であること。
#[test]
fn req21_ok_judgment_written_as_single_json_line_with_declaration_order() {
    let definition = Definition::parse(DEFINITION_JSON).expect("valid definition file");
    let options = definition.options();

    let result = JudgmentResult::new(options, "input-001", "accept", &[0.1, 0.8, 0.1])
        .expect("valid judgment result");

    let mut buffer: Vec<u8> = Vec::new();
    let exit_code = write_ok_judgment(&mut buffer, &result).expect("write must succeed");

    assert_eq!(exit_code, ExitCode::Ok);
    assert_eq!(exit_code.code(), 0);

    let text = String::from_utf8(buffer).expect("output must be valid UTF-8");
    assert_eq!(
        text,
        "{\"id\":\"input-001\",\"status\":\"ok\",\"predicted_label\":\"accept\",\"scores\":{\"reject\":0.1,\"accept\":0.8,\"review\":0.1}}\n"
    );
    assert_eq!(
        text.lines().count(),
        1,
        "output must be exactly one JSON line"
    );
    assert!(
        text.ends_with('\n'),
        "output must end with a single newline"
    );
}

/// REQ-21: 書き込みに失敗する `Write` を渡すと `Err` になり、
/// `ExitCode::Ok` は返らないこと。
#[test]
fn req21_write_failure_does_not_yield_ok_exit_code() {
    use std::io::{self, Write};

    struct FailingWriter;
    impl Write for FailingWriter {
        fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
            Err(io::Error::other("simulated write failure"))
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    let definition = Definition::parse(DEFINITION_JSON).expect("valid definition file");
    let options = definition.options();
    let result = JudgmentResult::new(options, "input-002", "reject", &[0.9, 0.05, 0.05]).unwrap();

    let mut writer = FailingWriter;
    let outcome = write_ok_judgment(&mut writer, &result);
    assert!(outcome.is_err(), "write failure must propagate as Err");
}
