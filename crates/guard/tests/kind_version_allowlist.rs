//! `kind_version` 許可リストの公開 API 結合テスト（REQ-39・TASK-39.6-1・#174・PoC-20 ケース 7。証拠種別: テストハーネス）。

use fandhe_edge_core::exitcode::ExitCode;
use fandhe_edge_guard::kind::{CheckedKind, KindAllowlist};
use fandhe_edge_guard::kind_version::{KindVersionAllowlist, KindVersionRejection};

fn checked(kind: &str) -> CheckedKind {
    KindAllowlist::supported()
        .check(kind)
        .expect("allowed kind")
}

/// REQ-39: 許可された `(c3, 1)` は合格し、kind と版を返す。
#[test]
fn req39_allowed_version_passes() {
    let ok = KindVersionAllowlist::supported()
        .check(checked("c3"), 1)
        .expect("ok");
    assert_eq!(ok.kind().as_str(), "c3");
    assert_eq!(ok.version(), 1);
}

/// REQ-39・PoC-20 ケース 7: `kind_version: 99` は全 kind で拒否される。
#[test]
fn req39_version_99_is_rejected() {
    for k in ["c1", "c3", "autoregressive"] {
        let e = KindVersionAllowlist::supported()
            .check(checked(k), 99)
            .expect_err("rejected");
        assert_eq!(e.reason_code(), "unsupported_kind_version");
        assert_eq!(e.exit_code() as i32, 64);
    }
    let e = KindVersionAllowlist::supported()
        .check(checked("c3"), 99)
        .expect_err("rejected");
    assert_eq!(
        e,
        KindVersionRejection::NotAllowed {
            kind: "c3",
            allowed: vec![1]
        }
    );
    assert_eq!(e.exit_code(), ExitCode::InvalidInput);
}

/// REQ-39: 境界値（0・2・u32::MAX）も拒否する。
#[test]
fn req39_boundary_versions_are_rejected() {
    let al = KindVersionAllowlist::supported();
    for v in [0, 2, u32::MAX] {
        assert!(al.check(checked("c3"), v).is_err());
        assert!(!al.contains("c3", v));
    }
}

/// REQ-39: 空の許可リストはすべて拒否する（fail-closed）。
#[test]
fn req39_empty_allowlist_rejects_everything() {
    let al = KindVersionAllowlist::new([]).expect("empty ok");
    let e = al.check(checked("c3"), 1).expect_err("rejected");
    assert_eq!(
        e,
        KindVersionRejection::NotAllowed {
            kind: "c3",
            allowed: vec![]
        }
    );
}

/// REQ-39: 構文違反の kind 名で許可リストを作れない。
#[test]
fn req39_new_rejects_malformed_kind() {
    assert!(KindVersionAllowlist::new([("C3; rm", 1)]).is_err());
}

/// REQ-39: `Display` は入力値（版番号）を含まない。
#[test]
fn req39_display_does_not_echo_input() {
    let e = KindVersionAllowlist::supported()
        .check(checked("c3"), 99)
        .expect_err("rejected");
    let s = e.to_string();
    assert!(!s.contains("99"), "{s}");
    assert_eq!(
        s,
        "kind_version is not supported for kind c3 (supported: 1)"
    );
}
