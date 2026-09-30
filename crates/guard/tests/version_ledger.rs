//! 版管理台帳の結合テスト（REQ-39・PoC-20 ケース 7・TASK-39.3-1・#167・TASK-39.3-2・#168。
//! 証拠種別: テストハーネス。合成ファイル・一時ディレクトリ。PoC-20 の実モデルのハッシュは
//! 本リポに無いため、構造（v1 記録 → 壊した v2 記録 → v1 へ戻す）だけを再現し、具体値は合成バイト列の sha256）。

// 閉じ込め検証（open_confined）が Unix 前提のため、Windows ではこのテスト全体を対象外とする。
#![cfg(unix)]

use std::fs;
use std::path::{Path, PathBuf};

use fandhe_edge_core::exitcode::ExitCode;
use fandhe_edge_core::fs::FsError;
use fandhe_edge_guard::path::PathRejection;
use fandhe_edge_guard::version_ledger::{
    ArtifactKind, CreatedAt, LedgerError, VersionId, VersionLedger,
};

const V1_HEX: &str = "1a1f4502024df8a68d12e64bb2364ad6308d04ed0a7d5e8300a676ec70867140";
const EMPTY_HEX: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

/// 一時ディレクトリ（Drop で削除）。
struct Sandbox(PathBuf);

impl Sandbox {
    fn new(label: &str) -> Self {
        let p = std::env::temp_dir().join(format!(
            "fandhe-guard-ledger-{}-{}",
            std::process::id(),
            label
        ));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).expect("mkdir");
        Sandbox(p)
    }
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn id(s: &str) -> VersionId {
    VersionId::new(s).expect("valid id")
}

fn at() -> CreatedAt {
    CreatedAt::from_unix_seconds(1_790_000_000).expect("valid time")
}

/// REQ-39: ファイルから記録したハッシュが具体値と一致する。
#[test]
fn req39_record_file_hash_and_time() {
    let sb = Sandbox::new("ok");
    let p = sb.0.join("model.onnx");
    fs::write(&p, b"model-v1").unwrap();
    fs::write(sb.0.join("empty"), b"").unwrap();
    let mut l = VersionLedger::new();
    let e = l
        .record_file(
            ArtifactKind::Model,
            id("v1"),
            &sb.0,
            Path::new("model.onnx"),
            1024,
            at(),
        )
        .unwrap();
    assert_eq!(e.sha256().to_hex(), V1_HEX);
    assert_eq!(e.created_at().to_rfc3339_utc(), "2026-09-21T14:13:20Z");
    let e = l
        .record_file(
            ArtifactKind::Data,
            id("d1"),
            &sb.0,
            Path::new("empty"),
            1024,
            at(),
        )
        .unwrap();
    assert_eq!(e.sha256().to_hex(), EMPTY_HEX);
    assert_eq!(l.len(), 2);
}

/// REQ-39: サイズ超過・非通常ファイル・不存在は台帳を変えずに拒否される。
#[test]
fn req39_record_file_failures_leave_ledger_empty() {
    let sb = Sandbox::new("fail");
    let big = sb.0.join("big");
    fs::write(&big, b"12345678").unwrap();
    let mut l = VersionLedger::new();

    let err = l
        .record_file(
            ArtifactKind::Model,
            id("v1"),
            &sb.0,
            Path::new("big"),
            4,
            at(),
        )
        .unwrap_err();
    assert!(matches!(err, LedgerError::Io(FsError::TooLarge { .. })));
    assert_eq!(err.exit_code(), ExitCode::LimitExceeded);

    let err = l
        .record_file(
            ArtifactKind::Model,
            id("v1"),
            &sb.0,
            Path::new("."),
            1024,
            at(),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        LedgerError::Path(PathRejection::NotRegularFile { .. })
    ));
    assert_eq!(err.exit_code(), ExitCode::InvalidInput);

    let err = l
        .record_file(
            ArtifactKind::Model,
            id("v1"),
            &sb.0,
            Path::new("nope"),
            1024,
            at(),
        )
        .unwrap_err();
    assert_eq!(err.exit_code(), ExitCode::InvalidInput);
    assert!(l.is_empty());
}

/// REQ-39: 重複は、ファイルを読む前に拒否される。
#[test]
fn req39_duplicate_checked_before_reading_file() {
    let sb = Sandbox::new("dup");
    let p = sb.0.join("model.onnx");
    fs::write(&p, b"model-v1").unwrap();
    let mut l = VersionLedger::new();
    l.record_file(
        ArtifactKind::Model,
        id("v1"),
        &sb.0,
        Path::new("model.onnx"),
        1024,
        at(),
    )
    .unwrap();
    let err = l
        .record_file(
            ArtifactKind::Model,
            id("v1"),
            &sb.0,
            Path::new("nope"),
            1024,
            at(),
        )
        .unwrap_err();
    assert!(matches!(err, LedgerError::DuplicateVersion { .. }));
    assert_eq!(l.len(), 1);
}

/// REQ-39: ルート外参照（`../`・絶対パス・symlink）は台帳を変えずに拒否される（open_confined 経由）。
#[test]
fn req39_record_file_rejects_escapes_from_root() {
    let sb = Sandbox::new("escape");
    let root = sb.0.join("root");
    let outside = sb.0.join("outside");
    fs::create_dir_all(&root).unwrap();
    fs::create_dir_all(&outside).unwrap();
    fs::write(outside.join("secret"), b"secret").unwrap();
    std::os::unix::fs::symlink(&outside, root.join("link_dir")).unwrap();
    std::os::unix::fs::symlink(outside.join("secret"), root.join("link_file")).unwrap();

    let mut l = VersionLedger::new();
    let abs = outside.join("secret");
    for cand in [
        Path::new("../outside/secret"),
        abs.as_path(),
        Path::new("link_dir/secret"),
        Path::new("link_file"),
    ] {
        let err = l
            .record_file(ArtifactKind::Model, id("v1"), &root, cand, 1024, at())
            .unwrap_err();
        assert!(matches!(err, LedgerError::Path(_)), "{cand:?}: {err}");
        assert_eq!(err.exit_code(), ExitCode::InvalidInput);
    }
    assert!(l.is_empty());
}

const TRUNCATED_HEX: &str = "bb5693af0cac6727384aac093fb596d71630ea0b55b18e1673399c0dff2e12ce";
const TAMPERED_HEX: &str = "d121be3103007b41edf96f8262925f8c7d61894afe9a041843b631f69445bc57";

/// v1（model-v1）と v2（切り詰め model-v）を記録した台帳と、その配置を作る。
fn ledger_v1_v2(sb: &Sandbox) -> VersionLedger {
    fs::create_dir_all(sb.0.join("v1")).unwrap();
    fs::create_dir_all(sb.0.join("v2")).unwrap();
    fs::write(sb.0.join("v1/model.onnx"), b"model-v1").unwrap();
    fs::write(sb.0.join("v2/model.onnx"), b"model-v").unwrap();
    let mut l = VersionLedger::new();
    for (v, p) in [("v1", "v1/model.onnx"), ("v2", "v2/model.onnx")] {
        l.record_file(ArtifactKind::Model, id(v), &sb.0, Path::new(p), 1024, at())
            .unwrap();
    }
    l
}

/// REQ-39・TASK-39.3-2: 前版へ戻すと、復元した成果物のハッシュが元の版のハッシュと一致する。
#[test]
fn req39_verify_rollback_to_previous_restores_v1_hash() {
    let sb = Sandbox::new("rb-prev");
    let l = ledger_v1_v2(&sb);
    assert_eq!(
        l.get(ArtifactKind::Model, &id("v2"))
            .unwrap()
            .sha256()
            .to_hex(),
        TRUNCATED_HEX
    );
    let r = l
        .verify_rollback_to_previous(
            ArtifactKind::Model,
            &id("v2"),
            &sb.0,
            Path::new("v1/model.onnx"),
            1024,
        )
        .unwrap();
    assert_eq!(r.recomputed_sha256().to_hex(), V1_HEX);
    assert_eq!(r.entry().sha256().to_hex(), V1_HEX);
    assert_eq!(r.entry().id().as_str(), "v1");
    assert_eq!(r.bytes(), b"model-v1");
    // 検証後にファイルを書き換えても、保持したバイト列は検証済みの内容のまま。
    fs::write(sb.0.join("v1/model.onnx"), b"tampered").unwrap();
    assert_eq!(r.into_bytes(), b"model-v1");
    assert_eq!(l.len(), 2);
}

/// REQ-39・TASK-39.3-2: 版を明示したロールバックでも一致する。
#[test]
fn req39_verify_rollback_to_explicit_version() {
    let sb = Sandbox::new("rb-explicit");
    let l = ledger_v1_v2(&sb);
    let r = l
        .verify_rollback_to(
            ArtifactKind::Model,
            &id("v1"),
            &sb.0,
            Path::new("v1/model.onnx"),
            1024,
        )
        .unwrap();
    assert_eq!(r.recomputed_sha256().to_hex(), V1_HEX);
}

/// REQ-39・TASK-39.3-2: 成果物が改ざんされていれば、ハッシュ不一致で拒否され台帳は不変。
#[test]
fn req39_rollback_detects_tampered_artifact() {
    let sb = Sandbox::new("rb-tamper");
    let l = ledger_v1_v2(&sb);
    fs::write(sb.0.join("v1/model.onnx"), b"tampered").unwrap();
    let err = l
        .verify_rollback_to_previous(
            ArtifactKind::Model,
            &id("v2"),
            &sb.0,
            Path::new("v1/model.onnx"),
            1024,
        )
        .unwrap_err();
    match &err {
        LedgerError::HashMismatch {
            expected, actual, ..
        } => {
            assert_eq!(expected.to_hex(), V1_HEX);
            assert_eq!(actual.to_hex(), TAMPERED_HEX);
        }
        other => panic!("unexpected {other:?}"),
    }
    assert_eq!(err.exit_code(), ExitCode::InvalidInput);
    assert_eq!(l.len(), 2);
    assert_eq!(
        l.get(ArtifactKind::Model, &id("v1"))
            .unwrap()
            .sha256()
            .to_hex(),
        V1_HEX
    );
}

/// REQ-39・TASK-39.3-2: 前版なし・未記録の版は、ファイルを読む前に拒否される。
#[test]
fn req39_rollback_without_previous_or_unknown() {
    let sb = Sandbox::new("rb-none");
    let mut l = VersionLedger::new();
    fs::write(sb.0.join("m"), b"model-v1").unwrap();
    l.record_file(
        ArtifactKind::Model,
        id("v1"),
        &sb.0,
        Path::new("m"),
        1024,
        at(),
    )
    .unwrap();
    let missing = Path::new("does-not-exist");
    let err = l
        .verify_rollback_to_previous(ArtifactKind::Model, &id("v1"), &sb.0, missing, 1024)
        .unwrap_err();
    assert!(
        matches!(err, LedgerError::NoPreviousVersion { .. }),
        "{err}"
    );
    let err = l
        .verify_rollback_to(ArtifactKind::Model, &id("v9"), &sb.0, missing, 1024)
        .unwrap_err();
    assert!(matches!(err, LedgerError::VersionNotFound { .. }), "{err}");
    assert_eq!(err.exit_code(), ExitCode::InvalidInput);
}

/// REQ-39・TASK-39.3-2: ロールバックでもルート外参照とサイズ超過は拒否される。
#[test]
fn req39_rollback_rejects_escapes_and_oversize() {
    let sb = Sandbox::new("rb-escape");
    let root = sb.0.join("root");
    let outside = sb.0.join("outside");
    fs::create_dir_all(&root).unwrap();
    fs::create_dir_all(&outside).unwrap();
    fs::write(root.join("m"), b"model-v1").unwrap();
    fs::write(outside.join("secret"), b"model-v1").unwrap();
    std::os::unix::fs::symlink(&outside, root.join("link_dir")).unwrap();
    std::os::unix::fs::symlink(outside.join("secret"), root.join("link_file")).unwrap();
    let mut l = VersionLedger::new();
    l.record_file(
        ArtifactKind::Model,
        id("v1"),
        &root,
        Path::new("m"),
        1024,
        at(),
    )
    .unwrap();
    let abs = outside.join("secret");
    for cand in [
        Path::new("../outside/secret"),
        abs.as_path(),
        Path::new("link_dir/secret"),
        Path::new("link_file"),
    ] {
        let err = l
            .verify_rollback_to(ArtifactKind::Model, &id("v1"), &root, cand, 1024)
            .unwrap_err();
        assert!(matches!(err, LedgerError::Path(_)), "{cand:?}: {err}");
        assert_eq!(err.exit_code(), ExitCode::InvalidInput);
    }
    let err = l
        .verify_rollback_to(ArtifactKind::Model, &id("v1"), &root, Path::new("m"), 4)
        .unwrap_err();
    assert!(matches!(err, LedgerError::Io(FsError::TooLarge { .. })));
    assert_eq!(err.exit_code(), ExitCode::LimitExceeded);
}
