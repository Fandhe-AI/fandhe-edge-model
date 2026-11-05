//! `package` の親のプロジェクト単位の排他ロック（`.package.lock` の `flock`）が、保持中の 2 つ目を拒否し、
//! 解放後は取れること（REQ-39・#662）。
//!
//! 証拠種別: テストハーネス。子プロセスを起動するテストと同じバイナリに置かない（#690）。同じプロセスの
//! 別スレッドが spawn すると、`drop` 直後の取り直しが失敗することがある（並行実行で 1/15 で再現、直列と
//! 単独では 0 件。実機の Mac）。子の生成から exec で CLOEXEC の fd が閉じるまでの間、子がロックの
//! open file description を参照するためと考えられる（推定）。本番の `package` の親は解放の後に spawn しない。

#![cfg(any(target_os = "linux", target_os = "macos"))]

use fandhe_edge_cli::infer_supervisor::PackageRemnants;
use fandhe_edge_core::exitcode::ExitCode;

/// REQ-39・#662: 同じプロジェクトの `package` の親は 1 つだけ。ロックを持つ間の 2 つ目は 64 で拒否し、
/// 解放後は取れる（後始末が他の `package` の作ったものを消さないための排他）。
#[test]
fn req39_issue662_second_package_on_same_project_is_rejected_while_locked() {
    let cwd = std::fs::canonicalize(std::env::temp_dir())
        .expect("temp")
        .join(format!("fandhe-pkglock-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&cwd);
    std::fs::create_dir_all(cwd.join("proj")).expect("proj");
    let dir = std::path::Path::new("proj");
    let first = PackageRemnants::snapshot(&cwd, dir).expect("first");
    assert!(first.is_some());
    let err = PackageRemnants::snapshot(&cwd, dir)
        .err()
        .expect("second is rejected");
    assert_eq!(err.code, ExitCode::InvalidInput);
    assert_eq!(err.message, "package already running");
    drop(first);
    assert!(
        PackageRemnants::snapshot(&cwd, dir)
            .expect("after release")
            .is_some()
    );
    let _ = std::fs::remove_dir_all(&cwd);
}
