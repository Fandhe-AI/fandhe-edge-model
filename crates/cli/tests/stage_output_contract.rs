//! 正常系 stdout JSON と stderr ログの分離の結合テスト（REQ-33・TASK-33.2-2・#139）。
//!
//! 証拠種別: テストハーネス（`package` の実処理への接続は #136 で `stages::package` に実装済み。バイナリでの完走は `pipeline_e2e.rs`）。
//! 期待値の `judgment:"pass"`・`status:"ok"` は PoC-16 vertical_a の package 工程の
//! 実測に基づく（`docs/spec` は読まない）。

use fandhe_edge_cli::args::Subcommand;
use fandhe_edge_cli::log::StderrLog;
use fandhe_edge_cli::stage_output::emit_package_outcome;
use fandhe_edge_core::exitcode::ExitCode;
use fandhe_edge_core::stage_report::{
    PackageCapacity, PackageCapacityComponents, PackageComponentSize, PackageMetrics,
};
use fandhe_edge_runtime::package_outcome::{PackageQualityJudgment, resolve_package_outcome};

const PASS_LINE: &str = "{\"step\":\"package\",\"status\":\"ok\",\"judgment\":\"pass\",\"acceptance_defined\":true,\"capacity\":{\"total_bytes\":10,\"limit_bytes\":40000000,\"exceeded\":false,\"components\":{\"weights\":{\"bytes\":10,\"file_count\":1},\"vocab_or_feature_transform\":{\"bytes\":0,\"file_count\":0},\"label_table\":{\"bytes\":0,\"file_count\":0},\"calibration\":{\"bytes\":0,\"file_count\":0},\"metadata\":{\"bytes\":0,\"file_count\":0}}},\"infer_p95\":null}\n";

/// 合成の計測値（重み 10 バイト・p95 の上限なし。#340）。
fn metrics() -> PackageMetrics {
    let c = PackageComponentSize::new;
    PackageMetrics {
        capacity: PackageCapacity::new(
            10,
            40_000_000,
            false,
            PackageCapacityComponents::new(c(10, 1), c(0, 0), c(0, 0), c(0, 0), c(0, 0)),
        ),
        infer_p95: None,
    }
}

/// AC1: Pass は exit 0 で judgment を含む JSON 1 行が stdout 側に書かれる。
#[test]
fn req33_package_pass_writes_single_json_and_exit_zero() {
    let mut stdout = Vec::new();
    let outcome = resolve_package_outcome(&[], PackageQualityJudgment::Pass);
    let code = emit_package_outcome(&mut stdout, &outcome, &metrics()).expect("emit");

    assert_eq!(code, ExitCode::Ok);
    assert_eq!(
        std::process::ExitCode::from(code),
        std::process::ExitCode::from(0u8)
    );
    assert_eq!(String::from_utf8(stdout).expect("utf8"), PASS_LINE);
}

/// AC2: ログは stderr 側だけに出て、stdout 側には混ざらない。
#[test]
fn req33_logs_go_to_stderr_and_never_reach_stdout() {
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    {
        let mut log = StderrLog::new(&mut stderr);
        log.info(Subcommand::Package, "start");
        log.info(Subcommand::Package, "measuring package");
        let outcome = resolve_package_outcome(&[], PackageQualityJudgment::Pass);
        emit_package_outcome(&mut stdout, &outcome, &metrics()).expect("emit");
        log.info(Subcommand::Package, "done");
    }

    let stdout = String::from_utf8(stdout).expect("utf8");
    assert_eq!(stdout, PASS_LINE);
    assert_eq!(stdout.matches('\n').count(), 1);
    assert!(!stdout.contains("fandhe-edge:"));

    let stderr = String::from_utf8(stderr).expect("utf8");
    assert_eq!(stderr.lines().count(), 3);
    for line in stderr.lines() {
        assert!(line.starts_with("fandhe-edge: package: "), "{line}");
        assert!(!line.contains("{\""), "{line}");
    }
}
