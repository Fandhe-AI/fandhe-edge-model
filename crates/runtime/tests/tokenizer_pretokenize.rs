//! 事前分割の golden 全件一致（REQ-41・TASK-41.4-3・#411。証拠種別: テストハーネス）。
//!
//! 期待値は Python 参照実装（`trainer/tools/poc26/qwen2_tokenizer.py`）の `pre_tokenize` を
//! `trainer/tools/gen_pretokenize_golden.py` が記録したもの。Rust の状態機械（自作）がこれと
//! 全件一致することを確かめる。不一致は Rust 側の誤りとして直す（golden は直さない）。

use fandhe_edge_runtime::tokenizer::pre_tokenize;

const GOLDEN: &str = include_str!("../../../fixtures/poc26/pretokenize_golden.json");

#[test]
fn golden_all_cases_match() {
    let doc: serde_json::Value = serde_json::from_str(GOLDEN).expect("golden json");
    let cases = doc["cases"].as_array().expect("cases");
    assert!(cases.len() > 300, "golden is unexpectedly small");
    for case in cases {
        let text = case["text"].as_str().expect("text");
        let nfc = case["nfc"].as_bool().expect("nfc");
        let want: Vec<&str> = case["pieces"]
            .as_array()
            .expect("pieces")
            .iter()
            .map(|p| p.as_str().expect("piece"))
            .collect();
        assert_eq!(
            pre_tokenize(text, nfc).expect("within limit"),
            want,
            "{text:?} nfc={nfc}"
        );
    }
}

/// 判定表・`unicode-normalization`・golden の Unicode 版が 15.0.0 で揃っている。
#[test]
fn unicode_versions_agree() {
    let doc: serde_json::Value = serde_json::from_str(GOLDEN).expect("golden json");
    assert_eq!(doc["source"]["python_unicodedata_version"], "15.0.0");
    assert_eq!(unicode_normalization::UNICODE_VERSION, (15, 0, 0));
}
