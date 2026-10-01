//! 実 `sandbox-exec` 下で macOS のメモリ監視が動くことの結合テスト
//! （REQ-38・REQ-39・TASK-39.5-2・#329・#327）。
//!
//! 従来の RSS 計測は setuid root の `/bin/ps` を起動しており、`sandbox-exec` 下では exec が
//! EPERM になって監視が常に `memory_probe_failed`（70）になった。libproc 版は外部プロセスを
//! 起動しないため sandbox 下でも完走する。証拠の種別: テストハーネス（実 `sandbox-exec`）。
//!
//! テストバイナリ自身を sandbox 内で再実行し（`helper`、`#[ignore]`）、そこで
//! `run_with_limits` を呼ぶ。`/bin/ps` が sandbox 下で使えるかは環境依存のためアサートせず、
//! 情報として出力するだけにする（#327 の Python 側回帰テストと同じ方針）。

#[cfg(target_os = "macos")]
mod macos {
    use fandhe_edge_guard::resource::{
        GuardedCommand, GuardedRunOutcome, MemoryLimit, RunConfig, TimeLimit, run_with_limits,
    };
    use std::process::Command;
    use std::time::Duration;

    const HELPER_ENV: &str = "FANDHE_GUARD_SANDBOX_HELPER";
    const HELPER_NAME: &str = "macos::sandbox_helper_run_with_limits";
    const OK_LINE: &str = "sandbox-helper-ok";
    const PROFILE: &str = "(version 1)(allow default)(deny network*)";

    /// sandbox 内で動く本体。環境変数が無い通常実行では何もしない（`#[ignore]` でも分離）。
    #[test]
    #[ignore = "invoked only by the outer sandbox-exec test"]
    fn sandbox_helper_run_with_limits() {
        if std::env::var(HELPER_ENV).is_err() {
            return;
        }
        let cfg = RunConfig::new(
            TimeLimit::new(Duration::from_secs(30)).unwrap(),
            1024 * 1024,
            64 * 1024,
        )
        .unwrap()
        .with_memory_limit(MemoryLimit::infer_default());
        let cmd = GuardedCommand::new("/bin/sleep").unwrap().arg("0.3");
        let outcome = run_with_limits(&cmd, &cfg)
            .unwrap_or_else(|e| panic!("run_with_limits failed under sandbox: {e}"));
        let GuardedRunOutcome::Exited { status, .. } = outcome else {
            panic!("expected Exited under sandbox");
        };
        assert!(status.success());
        println!("{OK_LINE}");
    }

    /// REQ-38・REQ-39・#329: 実 `sandbox-exec` 下でも RSS 監視つきの実行が `memory_probe_failed`
    /// にならず正常終了する。
    #[test]
    fn req38_req39_rss_probe_works_under_real_sandbox_exec() {
        let sandbox_exec = std::path::Path::new("/usr/bin/sandbox-exec");
        assert!(sandbox_exec.exists(), "/usr/bin/sandbox-exec is required");
        let exe = std::env::current_exe().unwrap();
        let out = Command::new(sandbox_exec)
            .args(["-p", PROFILE])
            .arg(&exe)
            .args(["--exact", HELPER_NAME, "--ignored", "--nocapture"])
            .env(HELPER_ENV, "1")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(
            out.status.code(),
            Some(0),
            "stdout: {stdout}\nstderr: {stderr}"
        );
        assert!(
            stdout.lines().any(|l| l == OK_LINE),
            "stdout: {stdout}\nstderr: {stderr}"
        );

        // `/bin/ps` が sandbox 下で使えるかは環境依存のためアサートしない（情報のみ）。
        let ps = Command::new(sandbox_exec)
            .args(["-p", PROFILE, "/bin/ps", "-o", "rss=", "-p"])
            .arg(std::process::id().to_string())
            .output();
        eprintln!(
            "info: /bin/ps under sandbox-exec exit={:?}",
            ps.ok().and_then(|o| o.status.code())
        );
    }
}
