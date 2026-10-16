//! トークナイザー JSON の読み込みと検証の結合テスト（REQ-41・REQ-39・TASK-41.4-2・#410）。
//!
//! 合成の最小 `tokenizer.json`（実物と同じ pipeline）で受理と拒否を固定する。証拠種別: テストハーネス。
//! 実物を読む確認は `#[ignore]` の `real_tokenizer_json_loads`（実機）。

use fandhe_edge_core::hash::Sha256Digest;
use fandhe_edge_runtime::tokenizer::{
    MAX_ADDED_CONTENT_CHARS, MAX_TOKENIZER_JSON_BYTES, MAX_VOCAB_TOKEN_CHARS, SPLIT_REGEX,
    SpecialIds, TokenizerLoadError, TokenizerSpec, bytes_to_unicode,
};
use serde_json::{Value, json};

type Case = (&'static str, Box<dyn Fn(&mut Value)>);

/// 256 個の byte 文字 + 併合結果 "ab"（id 256）+ 特殊トークン 3 件（id 257..=259）。
fn base() -> Value {
    let mut vocab = serde_json::Map::new();
    for (i, c) in bytes_to_unicode().iter().enumerate() {
        vocab.insert(c.to_string(), json!(i));
    }
    vocab.insert("ab".into(), json!(256));
    let special = |id: u32, content: &str| {
        json!({"id": id, "content": content, "single_word": false, "lstrip": false,
               "rstrip": false, "normalized": false, "special": true})
    };
    json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
            special(257, "<|im_start|>"), special(258, "<|im_end|>"), special(259, "<|endoftext|>")
        ],
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": SPLIT_REGEX}, "behavior": "Isolated", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}
        ]},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": false,
                           "trim_offsets": false, "use_regex": false},
        "decoder": {"type": "ByteLevel", "add_prefix_space": false,
                    "trim_offsets": false, "use_regex": false},
        "model": {"type": "BPE", "dropout": null, "unk_token": null,
                  "continuing_subword_prefix": "", "end_of_word_suffix": "",
                  "fuse_unk": false, "byte_fallback": false, "vocab": vocab, "merges": ["a b"]}
    })
}

fn load(v: &Value) -> Result<TokenizerSpec, TokenizerLoadError> {
    TokenizerSpec::from_bytes(v.to_string().as_bytes())
}

fn code(v: &Value) -> &'static str {
    load(v).expect_err("must be rejected").code()
}

#[test]
fn accepts_minimal_valid_tokenizer() {
    let spec = load(&base()).expect("valid");
    assert!(spec.nfc());
    assert_eq!(spec.vocab().len(), 257);
    assert_eq!(spec.merges().len(), 1);
    assert_eq!(spec.merges().get(&(97, 98)), Some(&(0, 256)));
    assert_eq!(spec.added_tokens().len(), 3);
    assert_eq!(
        spec.special_ids(),
        SpecialIds {
            im_start: 257,
            im_end: 258,
            endoftext: 259
        }
    );
    let bytes = base().to_string();
    assert_eq!(
        *spec.source_sha256(),
        Sha256Digest::of_bytes(bytes.as_bytes())
    );
    assert!(format!("{spec:?}").contains("vocab: 257"));
}

#[test]
fn accepts_null_normalizer_and_array_merges() {
    let mut v = base();
    v["normalizer"] = Value::Null;
    v["model"]["merges"] = json!([["a", "b"]]);
    let spec = load(&v).expect("valid");
    assert!(!spec.nfc());
    assert_eq!(spec.merges().len(), 1);
}

#[test]
fn rejects_malformed_json() {
    for text in [
        r#"{"a":1,}"#,
        r#"{"a":1,"a":2}"#,
        r#"{"a":NaN}"#,
        "",
        "[1]x",
    ] {
        let e = TokenizerSpec::from_bytes(text.as_bytes()).expect_err("rejected");
        assert_eq!(e.code(), "malformed_json", "{text}");
    }
    let e = TokenizerSpec::from_bytes(b"[]").expect_err("rejected");
    assert_eq!(e.code(), "invalid_tokenizer");
}

#[test]
fn rejects_limit_exceeded() {
    // ファイル（バイト列）の上限 +1
    let big = vec![b' '; usize::try_from(MAX_TOKENIZER_JSON_BYTES).unwrap() + 1];
    assert_eq!(
        TokenizerSpec::from_bytes(&big)
            .expect_err("rejected")
            .code(),
        "limit_exceeded"
    );
    // 入れ子の深さ
    let deep = format!("{}{}", "[".repeat(40), "]".repeat(40));
    assert_eq!(
        TokenizerSpec::from_bytes(deep.as_bytes())
            .expect_err("rejected")
            .code(),
        "limit_exceeded"
    );
    // vocab キーの長さ
    let mut v = base();
    let long = "a".repeat(MAX_VOCAB_TOKEN_CHARS + 1);
    v["model"]["vocab"][long] = json!(260);
    assert_eq!(code(&v), "limit_exceeded");
    // added_tokens の content の長さ
    let mut v = base();
    v["added_tokens"][0]["content"] = json!("x".repeat(MAX_ADDED_CONTENT_CHARS + 1));
    assert_eq!(code(&v), "limit_exceeded");
}

#[test]
fn load_path_rejects_oversized_file_and_hash_mismatch() {
    let dir = std::env::temp_dir().join(format!("tok_load_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("tokenizer.json");

    let ok = base().to_string();
    std::fs::write(&path, &ok).unwrap();
    let digest = Sha256Digest::of_bytes(ok.as_bytes());
    assert!(TokenizerSpec::load_path(&path, &digest).is_ok());
    let wrong = Sha256Digest::of_bytes(b"other");
    assert_eq!(
        TokenizerSpec::load_path(&path, &wrong)
            .expect_err("rejected")
            .code(),
        "integrity_mismatch"
    );

    let f = std::fs::File::create(&path).unwrap();
    f.set_len(MAX_TOKENIZER_JSON_BYTES + 1).unwrap();
    assert_eq!(
        TokenizerSpec::load_path(&path, &digest)
            .expect_err("rejected")
            .code(),
        "limit_exceeded"
    );
    assert_eq!(
        TokenizerSpec::load_path(&dir, &digest)
            .expect_err("rejected")
            .code(),
        "tokenizer_file_error"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn rejects_unsupported_with_section() {
    let cases: Vec<Case> = vec![
        (
            "model",
            Box::new(|v| v["model"]["type"] = json!("WordPiece")),
        ),
        (
            "model",
            Box::new(|v| v["model"]["byte_fallback"] = json!(true)),
        ),
        ("model", Box::new(|v| v["model"]["unknown_key"] = json!(1))),
        (
            "normalizer",
            Box::new(|v| v["normalizer"] = json!({"type": "NFKC"})),
        ),
        (
            "pre_tokenizer",
            Box::new(|v| {
                v["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = json!("\\w+")
            }),
        ),
        (
            "pre_tokenizer",
            Box::new(|v| v["pre_tokenizer"]["pretokenizers"][1]["add_prefix_space"] = json!(true)),
        ),
        (
            "post_processor",
            Box::new(|v| v["post_processor"] = json!({"type": "TemplateProcessing"})),
        ),
        (
            "decoder",
            Box::new(|v| v["decoder"] = json!({"type": "WordPiece"})),
        ),
        ("top-level", Box::new(|v| v["extra"] = json!(1))),
        ("version", Box::new(|v| v["version"] = json!("2.0"))),
        (
            "truncation",
            Box::new(|v| v["truncation"] = json!({"max_length": 8})),
        ),
    ];
    for (section, mutate) in cases {
        let mut v = base();
        mutate(&mut v);
        match load(&v).expect_err("rejected") {
            TokenizerLoadError::Unsupported {
                section: s,
                supported,
            } => {
                assert_eq!(s, section);
                assert!(!supported.is_empty());
            }
            other => panic!("unexpected {other:?} for {section}"),
        }
    }
    let msg = load(&{
        let mut v = base();
        v["normalizer"] = json!({"type": "NFKC"});
        v
    })
    .expect_err("rejected")
    .to_string();
    assert_eq!(
        msg,
        "unsupported tokenizer: normalizer (supported: NFC or null)"
    );
}

#[test]
fn rejects_invalid_contents() {
    let cases: Vec<Case> = vec![
        (
            "duplicate vocab id",
            Box::new(|v| v["model"]["vocab"]["ab"] = json!(0)),
        ),
        (
            "id out of range",
            Box::new(|v| v["model"]["vocab"]["ab"] = json!(9999)),
        ),
        (
            "byte char missing",
            Box::new(|v| {
                v["model"]["vocab"].as_object_mut().unwrap().remove("a");
            }),
        ),
        (
            "non byte-level char",
            Box::new(|v| v["model"]["vocab"]["あ"] = json!(260)),
        ),
        (
            "merge result missing",
            Box::new(|v| v["model"]["merges"] = json!(["a c"])),
        ),
        (
            "merge side missing",
            Box::new(|v| v["model"]["merges"] = json!(["a zz"])),
        ),
        (
            "merge not a pair",
            Box::new(|v| v["model"]["merges"] = json!(["a b c"])),
        ),
        (
            "duplicate merge",
            Box::new(|v| v["model"]["merges"] = json!(["a b", ["a", "b"]])),
        ),
        (
            "added id duplicates vocab",
            Box::new(|v| v["added_tokens"][0]["id"] = json!(5)),
        ),
        (
            "duplicate added content",
            Box::new(|v| v["added_tokens"][1]["content"] = json!("<|im_start|>")),
        ),
        (
            "special missing",
            Box::new(|v| v["added_tokens"][2]["special"] = json!(false)),
        ),
        (
            "added flag type",
            Box::new(|v| v["added_tokens"][0]["lstrip"] = json!("no")),
        ),
        (
            "added unknown key",
            Box::new(|v| v["added_tokens"][0]["extra"] = json!(1)),
        ),
    ];
    for (name, mutate) in cases {
        let mut v = base();
        mutate(&mut v);
        assert_eq!(code(&v), "invalid_tokenizer", "{name}");
    }
}

#[test]
fn error_message_does_not_echo_input_tokens() {
    let mut v = base();
    v["model"]["vocab"]["SECRET_TOKEN_あ"] = json!(260);
    let msg = load(&v).expect_err("rejected").to_string();
    assert!(!msg.contains("SECRET_TOKEN"), "{msg}");
}

/// 実物の `tokenizer.json`（Qwen2.5-1.5B-Instruct）を読む実機確認。実物は重みと同じ扱いで
/// リポに同梱しない（`fixtures/poc26/PROVENANCE.md`）ため、`FANDHE_EDGE_TOKENIZER_JSON` が
/// 示すファイルを読む。REQ-41・TASK-41.4-2。証拠種別: 実機（人が実行）。
#[test]
#[ignore = "needs the real tokenizer.json via FANDHE_EDGE_TOKENIZER_JSON"]
fn real_tokenizer_json_loads() {
    let path =
        std::env::var_os("FANDHE_EDGE_TOKENIZER_JSON").expect("set FANDHE_EDGE_TOKENIZER_JSON");
    let digest: Sha256Digest = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
        .parse()
        .expect("digest");
    let spec = TokenizerSpec::load_path(std::path::Path::new(&path), &digest).expect("loads");
    assert_eq!(spec.vocab().len(), 151_643);
    assert_eq!(spec.merges().len(), 151_387);
    assert_eq!(spec.added_tokens().len(), 22);
    assert!(spec.nfc());
    assert_eq!(spec.special_ids().im_start, 151_644);
    assert_eq!(spec.special_ids().im_end, 151_645);
    assert_eq!(spec.special_ids().endoftext, 151_643);
}
