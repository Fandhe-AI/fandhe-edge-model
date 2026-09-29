//! 公開 API 経由の結合テスト（REQ-39・TASK-39.2-3・#155・PoC-20 ケース 2。証拠種別: テストハーネス）。

use fandhe_edge_core::exitcode::ExitCode;
use fandhe_edge_guard::kind::{KindAllowlist, KindRejection, MAX_KIND_LEN};

fn all() -> Vec<&'static str> {
    vec!["autoregressive", "c1", "c3"]
}

fn assert_malformed(input: &str, expected: KindRejection) {
    let err = KindAllowlist::supported().check(input).unwrap_err();
    assert_eq!(err, expected, "input: {input:?}");
    assert_eq!(err.reason_code(), "malformed_kind");
    assert_eq!(err.exit_code(), ExitCode::InvalidInput);
    assert_eq!(err.exit_code() as i32, 64);
}

/// REQ-39: 対応済みの kind は通り、許可リスト側の文字列を返す。
#[test]
fn req39_supported_kinds_accepted() {
    let al = KindAllowlist::supported();
    for k in ["c1", "c3", "autoregressive"] {
        assert_eq!(al.check(k).unwrap().as_str(), k);
    }
}

/// REQ-39: 非対応の kind は `unsupported_kind`・64。
#[test]
fn req39_unsupported_kinds_rejected() {
    for k in ["c2", "c4", "onnx", "pickle"] {
        let err = KindAllowlist::supported().check(k).unwrap_err();
        assert_eq!(err, KindRejection::NotAllowed { allowed: all() });
        assert_eq!(err.reason_code(), "unsupported_kind");
        assert_eq!(err.exit_code() as i32, 64);
    }
    let only_c3 = KindAllowlist::new(["c3"]).unwrap();
    assert_eq!(
        only_c3.check("c1").unwrap_err(),
        KindRejection::NotAllowed {
            allowed: vec!["c3"]
        }
    );
}

/// REQ-39・PoC-20 ケース c: インジェクション文字列・不正形式の拒否。
#[test]
fn req39_injection_and_malformed_rejected() {
    let cases: [(&str, usize); 14] = [
        ("c3; rm -rf ~", 2),
        ("../c3", 0),
        ("c3\n", 2),
        ("c3\0", 2),
        ("C3", 0),
        (" c3", 0),
        ("c3 ", 2),
        ("c3$(id)", 2),
        ("c3`id`", 2),
        ("c3|id", 2),
        ("c1,c3", 2),
        ("ｃ３", 0),
        ("c\u{200b}3", 1),
        ("_meta", 0),
    ];
    for (input, off) in cases {
        assert_malformed(input, KindRejection::InvalidCharacter { byte_offset: off });
    }
}

/// REQ-39: 空・長さ上限。上限超過は構文検査より先に判定する。
#[test]
fn req39_empty_and_length_limits() {
    assert_malformed("", KindRejection::Empty);
    assert_malformed(&"a".repeat(65), KindRejection::TooLong { len: 65, max: 64 });
    let big = "!".repeat(1 << 20);
    assert_malformed(
        &big,
        KindRejection::TooLong {
            len: 1 << 20,
            max: MAX_KIND_LEN,
        },
    );
    let boundary = "a".repeat(MAX_KIND_LEN);
    assert_eq!(
        KindAllowlist::supported().check(&boundary).unwrap_err(),
        KindRejection::NotAllowed { allowed: all() }
    );
}

/// REQ-39: 拒否メッセージに入力値を含めない。
#[test]
fn req39_display_does_not_echo_input() {
    let err = KindAllowlist::supported()
        .check("c3; rm -rf ~")
        .unwrap_err();
    let msg = err.to_string();
    assert!(!msg.contains("rm -rf"));
    assert!(!msg.contains(';'));
}

/// REQ-39: 許可リストの構築は不正要素で失敗し、空集合は全拒否（fail-closed）。
#[test]
fn req39_allowlist_construction() {
    assert_eq!(
        KindAllowlist::new(["c3", "bad kind"]).unwrap_err(),
        KindRejection::InvalidCharacter { byte_offset: 3 }
    );
    assert!(KindAllowlist::new(["_meta"]).is_err());
    assert!(KindAllowlist::new([""]).is_err());
    let empty = KindAllowlist::new([]).unwrap();
    assert_eq!(
        empty.check("c3").unwrap_err(),
        KindRejection::NotAllowed { allowed: vec![] }
    );
    let supported = KindAllowlist::supported();
    assert_eq!(supported.iter().collect::<Vec<_>>(), all());
    assert!(supported.contains("c1") && !supported.contains("c2"));
}
