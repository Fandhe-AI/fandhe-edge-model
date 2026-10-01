//! 実 `sandbox-exec` 下で macOS のメモリ監視が動くことの結合テスト
//! （REQ-38・REQ-39・TASK-39.5-2・#329・#327）。
//!
//! 従来の RSS 計測は setuid root の `/bin/ps` を起動しており、`sandbox-exec` 下では exec が
//! EPERM になって監視が常に `memory_probe_failed`（70）になった。libproc 版は外部プロセスを
//! 起動しないため sandbox 下でも完走する。証拠の種別: テストハーネス（実 `sandbox-exec`）。
//!
//! テストバイナリ自身を sandbox 内で再実行し（`helper`、`#[ignore]`）、そこで
//! `run_with_limits` を呼ぶ。正常終了だけでは RSS を一度も取得できなくても（`Ok(None)` が続いても）
//! 通ってしまうため、低い上限を超えて割り当てる子がメモリ超過で止まることも確かめる
//! （sandbox 内で RSS を実際に読めていることの確認。PR #336 の Codex レビュー指摘）。
//! `/bin/ps` が sandbox 下で使えるかは環境依存のためアサートせず、情報として出力するだけに
//! する（#327 の Python 側回帰テストと同じ方針）。

#[cfg(target_os = "macos")]
mod macos {
    use fandhe_edge_guard::resource::{
        GuardedCommand, GuardedRunOutcome, MemoryLimit, ResourceKind, RunConfig, TimeLimit,
        run_with_limits,
    };
    use std::process::Command;
    use std::time::Duration;

    const HELPER_ENV: &str = "FANDHE_GUARD_SANDBOX_HELPER";
    const HELPER_NAME: &str = "macos::sandbox_helper_run_with_limits";
    const OK_LINE: &str = "sandbox-helper-ok";
    const PROFILE: &str = "(version 1)(allow default)(deny network*)";
    const CHILD_ENV: &str = "FANDHE_GUARD_SANDBOX_ALLOC_CHILD";
    const CHILD_NAME: &str = "macos::sandbox_alloc_child";
    const MIB: u64 = 1024 * 1024;
    /// sandbox 内の割り当て子の上限と割り当て量（`memory_limit.rs` の 64 MiB 上限のテストと同じ値）。
    const SMALL_LIMIT: u64 = 64 * MIB;
    const ALLOC_MIB: usize = 512;

    /// sandbox 内で `run_with_limits` から起動される割り当て子。環境変数が無い通常実行では何もしない。
    ///
    /// 8 MiB ずつ確保して保持する。macOS のメモリ圧縮で RSS が縮まないよう、擬似乱数の雛形を
    /// 各チャンクへ複製する（`memory_limit.rs` の `child_entry` と同じ方式）。kill されなくても
    /// 20 秒で自走終了する（暴走防止）。
    #[test]
    #[ignore = "invoked only by the sandbox helper"]
    fn sandbox_alloc_child() {
        if std::env::var(CHILD_ENV).is_err() {
            return;
        }
        const CHUNK: usize = 8 * 1024 * 1024;
        let mut template = vec![0u8; CHUNK];
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        for b in template.chunks_mut(8) {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let bytes = x.to_le_bytes();
            let n = b.len();
            b.copy_from_slice(&bytes[..n]);
        }
        let mut chunks: Vec<Vec<u8>> = Vec::new();
        for i in 0..ALLOC_MIB.div_ceil(8) {
            let mut v = template.clone();
            if let Some(first) = v.first_mut() {
                *first = (i % 251) as u8;
            }
            chunks.push(std::hint::black_box(v));
            std::thread::sleep(Duration::from_millis(5));
        }
        std::thread::sleep(Duration::from_secs(20));
        std::hint::black_box(&chunks);
    }

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

        // RSS を実際に読めていれば、低い上限を超えた割り当て子はメモリ超過で止まる。
        let cfg = RunConfig::new(
            TimeLimit::new(Duration::from_secs(30)).unwrap(),
            1024 * 1024,
            64 * 1024,
        )
        .unwrap()
        .with_memory_limit(MemoryLimit::new(SMALL_LIMIT).unwrap());
        let cmd = GuardedCommand::new(std::env::current_exe().unwrap())
            .unwrap()
            .arg("--exact")
            .arg(CHILD_NAME)
            .arg("--ignored")
            .arg("--nocapture")
            .arg("--test-threads=1")
            .env(CHILD_ENV, "1");
        let outcome = run_with_limits(&cmd, &cfg)
            .unwrap_or_else(|e| panic!("run_with_limits failed under sandbox: {e}"));
        let GuardedRunOutcome::LimitExceeded(rec) = outcome else {
            panic!("expected LimitExceeded (memory) under sandbox");
        };
        assert_eq!(rec.kind(), ResourceKind::Memory);
        assert_eq!(rec.memory_limit_bytes(), Some(SMALL_LIMIT));
        let observed = rec.observed_rss_bytes().unwrap();
        assert!(observed > SMALL_LIMIT, "observed {observed}");
        assert!(observed < SMALL_LIMIT + 256 * MIB, "observed {observed}");
        println!("{OK_LINE}");
    }

    /// REQ-38・REQ-39・#329: 実 `sandbox-exec` 下でも RSS 監視つきの実行が `memory_probe_failed`
    /// にならず正常終了し、低い上限を超える子はメモリ超過（`ResourceKind::Memory`）で止まる。
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
