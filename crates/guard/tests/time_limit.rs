//! 実行時間上限（暫定 10 秒）の結合テスト（REQ-39・TASK-39.5-1・#170・PoC-20 ケース 3）。
//!
//! 子プロセスはテストバイナリ自身の再実行（`child_entry`）で用意し、環境変数
//! `FANDHE_GUARD_TEST_CHILD` で挙動を選ぶ（3 OS で動く）。証拠の種別: テストハーネス
//! （実プロセス・実時計・合成 sleeper）。実 CLI の推論が 10 秒を超える実測ではない。

use fandhe_edge_guard::resource::{
    DEFAULT_STDERR_CAP, DEFAULT_STDOUT_CAP, GuardedCommand, GuardedRunOutcome, INFER_TIME_LIMIT,
    ResourceKind, RunConfig, TimeLimit, run_with_limits,
};
use std::time::Duration;

const MODE_ENV: &str = "FANDHE_GUARD_TEST_CHILD";

/// 自己再実行される子の本体。環境変数が無い通常実行では何もしない。
#[test]
fn child_entry() {
    let Ok(mode) = std::env::var(MODE_ENV) else {
        return;
    };
    match mode.as_str() {
        "sleep60" => std::thread::sleep(Duration::from_secs(60)),
        "sleep5" => std::thread::sleep(Duration::from_secs(5)),
        "ok" => println!("child-ok"),
        "grandchild" => {
            // 自身は即終了し、stdout を継承した孫を残す。孫は 5 秒で自走終了する（後始末）。
            let _ = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "child_entry", "--nocapture", "--test-threads=1"])
                .env(MODE_ENV, "sleep5")
                .spawn();
            println!("child-exit");
        }
        "exit70" => std::process::exit(70),
        "flood" => {
            let line = "x".repeat(1024);
            for _ in 0..64 {
                println!("{line}");
            }
        }
        "spew" => {
            // 期限まで出力し続ける。kill されなくても 20 秒で自走終了する（テストの暴走防止）。
            use std::io::Write;
            let chunk = [b'x'; 8192];
            let stop = std::time::Instant::now() + Duration::from_secs(20);
            let mut out = std::io::stdout();
            while std::time::Instant::now() < stop {
                if out.write_all(&chunk).is_err() {
                    break;
                }
            }
        }
        _ => {}
    }
}

fn child(mode: &str) -> GuardedCommand {
    GuardedCommand::new(std::env::current_exe().unwrap())
        .unwrap()
        .arg("--exact")
        .arg("child_entry")
        .arg("--nocapture")
        .arg("--test-threads=1")
        .env(MODE_ENV, mode)
}

fn config(limit: Duration) -> RunConfig {
    RunConfig::new(TimeLimit::new(limit).unwrap(), 1024 * 1024, 64 * 1024)
        .unwrap()
        .without_memory_limit()
}

/// REQ-39: 暫定値は 10 秒で、既定設定にも反映される。
#[test]
fn req39_infer_time_limit_is_10_seconds() {
    assert_eq!(INFER_TIME_LIMIT, Duration::from_secs(10));
    assert_eq!(RunConfig::default().time_limit(), Duration::from_secs(10));
}

/// REQ-39・PoC-20 ケース 3: 10 秒を超える子は kill され、時間超過として記録される。
#[test]
fn req39_child_exceeding_10s_is_killed_and_recorded_as_time_limit() {
    // 時間上限だけを検証する（既定設定はメモリ上限も持ち、計測手段の無い OS では fail-closed になる）。
    let cfg = RunConfig::new(
        TimeLimit::infer_default(),
        DEFAULT_STDOUT_CAP,
        DEFAULT_STDERR_CAP,
    )
    .unwrap()
    .without_memory_limit();
    let outcome = run_with_limits(&child("sleep60"), &cfg).unwrap();
    let GuardedRunOutcome::LimitExceeded(rec) = outcome else {
        panic!("expected LimitExceeded");
    };
    assert_eq!(rec.kind(), ResourceKind::Time);
    assert_eq!(rec.limit(), Duration::from_secs(10));
    assert!(rec.child_reaped());
    assert_eq!(rec.exit_code().code(), 20);
    assert_eq!(rec.code(), "time_limit_exceeded");
    assert!(rec.elapsed() >= Duration::from_secs(10));
    assert!(rec.elapsed() < Duration::from_secs(15));
}

/// REQ-39: 上限値を変えても機構が効く。
#[test]
fn req39_short_limit_kills_sleeper() {
    let outcome = run_with_limits(&child("sleep5"), &config(Duration::from_millis(300))).unwrap();
    let GuardedRunOutcome::LimitExceeded(rec) = outcome else {
        panic!("expected LimitExceeded");
    };
    assert!(rec.elapsed() >= Duration::from_millis(300));
    assert!(rec.elapsed() < Duration::from_secs(3));
}

/// REQ-39: 期限内に終わる子は Exited で、出力が取れる。
#[test]
fn req39_child_within_limit_exits_normally() {
    let outcome = run_with_limits(&child("ok"), &config(Duration::from_secs(30))).unwrap();
    let GuardedRunOutcome::Exited { status, output, .. } = outcome else {
        panic!("expected Exited");
    };
    assert!(status.success());
    let text = String::from_utf8_lossy(output.stdout()).into_owned();
    assert!(text.contains("child-ok"));
    assert!(!output.stdout_truncated());
}

/// REQ-39: 非 0 終了は時間超過と区別される。
#[test]
fn req39_nonzero_exit_is_not_limit_exceeded() {
    let outcome = run_with_limits(&child("exit70"), &config(Duration::from_secs(30))).unwrap();
    let GuardedRunOutcome::Exited { status, .. } = outcome else {
        panic!("expected Exited");
    };
    assert_eq!(status.code(), Some(70));
}

/// REQ-39: 読み取り上限を超える出力は切り詰めフラグ付きで上限長に収まる。
#[test]
fn req39_stdout_is_capped_and_flagged() {
    let cfg = RunConfig::new(TimeLimit::new(Duration::from_secs(30)).unwrap(), 100, 100)
        .unwrap()
        .without_memory_limit();
    let outcome = run_with_limits(&child("flood"), &cfg).unwrap();
    let GuardedRunOutcome::Exited { output, .. } = outcome else {
        panic!("expected Exited");
    };
    assert_eq!(output.stdout().len(), 100);
    assert!(output.stdout_truncated());
}

/// REQ-39: 孫がパイプを保持しても runner は有界の時間で返る（子孫は kill しない制限の確認）。
#[cfg(unix)]
#[test]
fn req39_grandchild_holding_pipe_does_not_block_runner() {
    let cmd = GuardedCommand::new("/bin/sh")
        .unwrap()
        .arg("-c")
        .arg("sleep 3 & exec sleep 60");
    let started = std::time::Instant::now();
    let outcome = run_with_limits(&cmd, &config(Duration::from_millis(300))).unwrap();
    assert!(matches!(outcome, GuardedRunOutcome::LimitExceeded(_)));
    assert!(started.elapsed() < Duration::from_secs(5));
}

/// REQ-39: 子が出力し続けても、監視ループが期限を判定して kill する（pump が EOF まで居座らない）。
#[cfg(unix)]
#[test]
fn req39_continuous_output_does_not_bypass_time_limit() {
    let cfg = RunConfig::new(
        TimeLimit::new(Duration::from_millis(500)).unwrap(),
        1024 * 1024,
        64 * 1024,
    )
    .unwrap()
    .without_memory_limit();
    let started = std::time::Instant::now();
    let outcome = run_with_limits(&child("spew"), &cfg).unwrap();
    let GuardedRunOutcome::LimitExceeded(rec) = outcome else {
        panic!("expected LimitExceeded");
    };
    assert_eq!(rec.kind(), ResourceKind::Time);
    assert!(rec.child_reaped());
    assert!(started.elapsed() < Duration::from_secs(5));
}

/// REQ-39: 子が期限内に終了しても孫が stdout を保持して EOF が来ない場合、欠けた出力を
/// 正常終了として返さず `ReadOutput` にする（期限 30 秒より十分短い約 1 秒で判定される）。
#[test]
fn req39_grandchild_holding_pipe_is_read_output_error() {
    let err = run_with_limits(&child("grandchild"), &config(Duration::from_secs(30))).unwrap_err();
    assert_eq!(err, fandhe_edge_guard::resource::GuardRunError::ReadOutput);
}
