//! 容量上限超過 → `limit_exceeded` の結合テスト（REQ-30・REQ-21・TASK-30.2・#124）。
//! 証拠種別: テストハーネス（一時ディレクトリの実ファイル）。本番データでの再実演は未実施。

use fandhe_edge_core::exitcode::ExitCode;
use fandhe_edge_runtime::capacity::{
    CapacityBreakdown, PackageComponent, PackageFile, measure_package,
};
use fandhe_edge_runtime::capacity_limit::{
    CapacityLimit, CapacityLimitCheck, check_capacity_limit,
};
use fandhe_edge_runtime::package_outcome::{
    LimitBreach, PackageQualityJudgment, PackageVerdict, resolve_package_outcome,
};
use std::path::{Path, PathBuf};

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        for _ in 0..100 {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos());
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let p = std::env::temp_dir().join(format!(
                "fandhe-runtime-caplimit-{tag}-{}-{nanos}-{n}",
                std::process::id()
            ));
            match std::fs::create_dir(&p) {
                Ok(()) => return Self(p),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => panic!("create temp dir: {e}"),
            }
        }
        panic!("could not create a unique temp dir");
    }
    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Weights 1000B・LabelTable 300B・Metadata 249B（合計 1549B）を計測する。
fn measure(d: &TempDir) -> CapacityBreakdown {
    let mk = |name: &str, len: usize, component| {
        let path = d.path().join(name);
        std::fs::write(&path, vec![0u8; len]).unwrap();
        PackageFile { component, path }
    };
    let files = [
        mk("model.onnx", 1000, PackageComponent::Weights),
        mk("labels.json", 300, PackageComponent::LabelTable),
        mk("meta.json", 249, PackageComponent::Metadata),
    ];
    measure_package(&files).unwrap()
}

const ALL: [PackageQualityJudgment; 4] = [
    PackageQualityJudgment::Pass,
    PackageQualityJudgment::Fail,
    PackageQualityJudgment::Undeterminable,
    PackageQualityJudgment::NotDefined,
];

fn breaches(check: &CapacityLimitCheck) -> Vec<LimitBreach> {
    check.breach().into_iter().collect()
}

#[test]
fn req30_total_minus_one_is_limit_exceeded_for_every_judgment() {
    let d = TempDir::new("over");
    let b = measure(&d);
    let check = check_capacity_limit(&b, Some(CapacityLimit::from_bytes(1548).unwrap()));
    let br = breaches(&check);
    for q in ALL {
        let o = resolve_package_outcome(&br, q);
        assert_eq!(o.exit_code, ExitCode::LimitExceeded);
        assert_eq!(o.exit_code.code(), 20);
        assert_eq!(o.verdict, PackageVerdict::LimitExceeded);
        assert_eq!(
            o.breaches,
            vec![LimitBreach::Capacity {
                measured_bytes: 1549,
                limit_bytes: 1548
            }]
        );
    }
}

#[test]
fn req30_limit_equal_or_above_total_keeps_quality_exit_code() {
    let d = TempDir::new("within");
    let b = measure(&d);
    for limit in [1549, 1550] {
        let check = check_capacity_limit(&b, Some(CapacityLimit::from_bytes(limit).unwrap()));
        assert!(matches!(check, CapacityLimitCheck::Within { .. }));
        let br = breaches(&check);
        let codes: Vec<_> = ALL
            .iter()
            .map(|q| resolve_package_outcome(&br, *q).exit_code.code())
            .collect();
        assert_eq!(codes, vec![0, 10, 12, 0]);
    }
}

#[test]
fn req30_no_limit_configured_never_limit_exceeded() {
    let d = TempDir::new("none");
    let b = measure(&d);
    let check = check_capacity_limit(&b, None);
    assert_eq!(
        check,
        CapacityLimitCheck::NotConfigured { total_bytes: 1549 }
    );
    assert_eq!(
        resolve_package_outcome(&breaches(&check), PackageQualityJudgment::Fail)
            .exit_code
            .code(),
        10
    );
}
