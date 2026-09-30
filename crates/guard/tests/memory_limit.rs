//! メモリ（RSS）上限（暫定 2 GiB）の結合テスト（REQ-39・TASK-39.5-2・#171・PoC-20 ケース 3）。
//!
//! 証拠の種別: テストハーネス（実プロセス・実割り当て・RSS ポーリングによる模擬の上限）。
//! 実 CLI の推論が 2 GiB を超える実測ではない。上限は `setrlimit` 等ではなく 50 ms 間隔の
//! ポーリングで強制するため、観測 RSS は上限をわずかに超えた値になる（PoC-20 は約 4% 超過）。
//! 子プロセスはテストバイナリ自身の再実行（`child_entry`）で用意し、環境変数
//! `FANDHE_GUARD_TEST_CHILD` で挙動を、`FANDHE_GUARD_TEST_ALLOC_MIB` で割り当て量を選ぶ。

use fandhe_edge_guard::resource::GuardedCommand;
#[cfg(any(target_os = "linux", target_os = "macos"))]
use fandhe_edge_guard::resource::{
    GuardedRunOutcome, MemoryLimit, ResourceKind, RunConfig, TimeLimit, run_with_limits,
};
use std::time::Duration;

const MODE_ENV: &str = "FANDHE_GUARD_TEST_CHILD";
const ALLOC_ENV: &str = "FANDHE_GUARD_TEST_ALLOC_MIB";
#[cfg(any(target_os = "linux", target_os = "macos"))]
const MIB: u64 = 1024 * 1024;

/// 自己再実行される子の本体。環境変数が無い通常実行では何もしない。
#[test]
fn child_entry() {
    let Ok(mode) = std::env::var(MODE_ENV) else {
        return;
    };
    match mode.as_str() {
        "alloc" => {
            let target_mib: usize = std::env::var(ALLOC_ENV)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            // 8 MiB ずつ確保する。macOS のメモリ圧縮で RSS が縮まないよう、圧縮されにくい
            // 擬似乱数の雛形を 1 つ作り、各チャンクへ複製する（ページ単位では非圧縮）。
            // 雛形の生成は 1 回だけにして、debug ビルドでも確保速度が律速にならないようにする。
            // 1 チャンクごとに sleep して割り当て速度を抑え、ポーリング間のオーバーシュートを
            // 有界にする。
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
            for i in 0..target_mib.div_ceil(8) {
                let mut v = template.clone();
                // チャンク間で内容を変える（重複排除・圧縮の余地を減らす）。
                if let Some(first) = v.first_mut() {
                    *first = (i % 251) as u8;
                }
                chunks.push(std::hint::black_box(v));
                std::thread::sleep(Duration::from_millis(5));
            }
            // 目標量に達したら保持したまま待つ。kill されなくても 20 秒で自走終了する（暴走防止）。
            std::thread::sleep(Duration::from_secs(20));
            std::hint::black_box(&chunks);
        }
        "sleep60" => std::thread::sleep(Duration::from_secs(60)),
        "ok" => println!("child-ok"),
        _ => {}
    }
}

fn child(mode: &str, alloc_mib: u64) -> GuardedCommand {
    GuardedCommand::new(std::env::current_exe().unwrap())
        .unwrap()
        .arg("--exact")
        .arg("child_entry")
        .arg("--nocapture")
        .arg("--test-threads=1")
        .env(MODE_ENV, mode)
        .env(ALLOC_ENV, alloc_mib.to_string())
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn config(limit: Duration, memory: MemoryLimit) -> RunConfig {
    RunConfig::new(TimeLimit::new(limit).unwrap(), 1024 * 1024, 64 * 1024)
        .unwrap()
        .with_memory_limit(memory)
}

/// REQ-39・PoC-20 ケース 3: RSS が 2 GiB を超える子は kill され、メモリ超過として記録される。
///
/// Linux 限定。macOS の GitHub ホステッド runner（搭載メモリが小さくメモリ圧縮が働く）では、
/// 2.25 GiB を確保しても `ps` の RSS が 2 GiB に達せず超過を観測できなかった（CI 実測。
/// 超過検知の機構自体は同 OS で 64 MiB 上限のテストが検証している）。2 GiB 既定値の
/// 保持は `resource` のユニットテストで全 OS 固定している。macOS 実機（十分なメモリ）での
/// 2 GiB 実測は実機前提の確認事項で、本テストは実機の代替にしない。
#[cfg(target_os = "linux")]
#[test]
fn req39_rss_over_2gib_is_killed_and_recorded_as_memory() {
    // 割り当て・RSS 計測（macOS は `ps` 起動）の所要時間で時間超過に反転しないよう、
    // 時間上限は 60 秒にする。割り当て量は上限を十分に超える 2.25 GiB とする。
    let cfg = config(Duration::from_secs(60), MemoryLimit::infer_default());
    let outcome = run_with_limits(&child("alloc", 2304), &cfg).unwrap();
    let GuardedRunOutcome::LimitExceeded(rec) = outcome else {
        panic!("expected LimitExceeded");
    };
    assert_eq!(rec.kind(), ResourceKind::Memory);
    assert_eq!(rec.code(), "memory_limit_exceeded");
    assert_eq!(rec.exit_code().code(), 20);
    assert_eq!(rec.memory_limit_bytes(), Some(2_147_483_648));
    let observed = rec.observed_rss_bytes().unwrap();
    assert!(observed > 2_147_483_648, "observed {observed}");
    assert!(observed < 2_147_483_648 + 512 * MIB, "observed {observed}");
    assert!(rec.child_reaped());
    eprintln!("observed_rss_bytes={observed}");
}

/// REQ-39: 上限値を変えても機構が効く（64 MiB 上限）。
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn req39_small_memory_limit_kills_allocator() {
    let cfg = config(Duration::from_secs(30), MemoryLimit::new(64 * MIB).unwrap());
    let outcome = run_with_limits(&child("alloc", 512), &cfg).unwrap();
    let GuardedRunOutcome::LimitExceeded(rec) = outcome else {
        panic!("expected LimitExceeded");
    };
    assert_eq!(rec.kind(), ResourceKind::Memory);
    assert_eq!(rec.memory_limit_bytes(), Some(64 * MIB));
    let observed = rec.observed_rss_bytes().unwrap();
    assert!(observed > 64 * MIB, "observed {observed}");
    assert!(observed < 64 * MIB + 256 * MIB, "observed {observed}");
}

/// REQ-39: 上限内で終わる子は Exited で、出力が取れる。
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn req39_child_within_memory_limit_exits_normally() {
    let cfg = config(Duration::from_secs(30), MemoryLimit::infer_default());
    let outcome = run_with_limits(&child("ok", 0), &cfg).unwrap();
    let GuardedRunOutcome::Exited { status, output, .. } = outcome else {
        panic!("expected Exited");
    };
    assert!(status.success());
    assert!(String::from_utf8_lossy(output.stdout()).contains("child-ok"));
}

/// REQ-39: メモリ監視が有効でも、sleep 中の子は時間超過になる（メモリの誤検知がない）。
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn req39_sleeper_with_memory_limit_is_time_limit() {
    let cfg = config(Duration::from_millis(500), MemoryLimit::infer_default());
    let outcome = run_with_limits(&child("sleep60", 0), &cfg).unwrap();
    let GuardedRunOutcome::LimitExceeded(rec) = outcome else {
        panic!("expected LimitExceeded");
    };
    assert_eq!(rec.kind(), ResourceKind::Time);
    assert_eq!(rec.memory_limit_bytes(), None);
}

/// REQ-39: 計測手段の無い OS ではメモリ上限つきの実行を起動前に拒否する（fail-closed）。
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
#[test]
fn req39_memory_limit_unsupported_os_fails_closed() {
    use fandhe_edge_guard::resource::{
        GuardRunError, MemoryLimit, RunConfig, TimeLimit, run_with_limits,
    };
    let cfg = RunConfig::new(TimeLimit::infer_default(), 1024, 1024)
        .unwrap()
        .with_memory_limit(MemoryLimit::infer_default());
    let err = run_with_limits(&child("ok", 0), &cfg).unwrap_err();
    assert_eq!(err, GuardRunError::MemoryLimitUnsupported);
    assert_eq!(err.exit_code().code(), 70);
    // 既定設定・`RunConfig::new` も上限を保持するため、上限なしで子を起動せず同じく拒否する。
    for cfg in [
        RunConfig::default(),
        RunConfig::new(TimeLimit::infer_default(), 1024, 1024).unwrap(),
    ] {
        assert_eq!(
            run_with_limits(&child("ok", 0), &cfg).unwrap_err(),
            GuardRunError::MemoryLimitUnsupported
        );
    }
}
