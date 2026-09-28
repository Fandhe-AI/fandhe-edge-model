//! モデルパッケージの評価前後ハッシュ比較の結合テスト（REQ-27「評価の独立性」・
//! TASK-27.1-1・issue #69）。
//!
//! 出典: PoC-9 `InvarianceTest`（`docs/spec/03-poc/evaluation-contract/`）の
//! 「モデルパッケージ側」を移植する（`docs/spec` は読み込まず、手順の再現のみを
//! 行う。`.claude/rules/spec-reference.md` のビルド独立方針）。
//!
//! 証拠の種別: テストハーネス（合成データによる結合テスト）。実機ではない。
//!
//! `req27_evaluate_with_invariance_reports_*_removed_by_evaluation_step`
//! 系（削除系）は、評価クロージャの実行中に構成要素ファイルそのものを
//! 削除することで、評価後の取得が NotFound を [`ComponentChange::Removed`]
//! として報告することを確かめる（issue #226。TASK-27.1-1 の PR #214
//! レビューで挙がった P2 指摘への対応）。
//!
//! # 本ファイルの 2 種類のテスト
//!
//! 1. **[`fandhe_edge_eval::invariance::evaluate_with_invariance`] を使う結合
//!    テスト**（`evaluate_with_invariance_*`）: CLI の `evaluate` 工程（将来）が
//!    実際に使う想定の経路。評価前後のスナップショット取得は API 自身が
//!    ディスクから行うため、評価クロージャの内側でファイルを実際に書き換える
//!    ことで「評価処理そのものによるモデル改変」を模擬でき、API がそれを
//!    fail-closed で検出することを確かめられる（issue #214 の codex/review
//!    指摘への対応: 旧バージョンは評価クロージャの外側〔テスト自身〕が
//!    上書きしていたため、比較器〔`verify_unchanged`〕単体の動作しか
//!    確認できていなかった）
//! 2. **[`ModelPackageBytes`] / [`ModelPackageSnapshot`] を直接使う比較器の
//!    テスト**（`comparator_*`）: 前後のバイト列が与えられたときに
//!    `verify_unchanged` が正しく差分を検出することだけを確かめる、より
//!    低レベルなテスト。**評価経路には結合していない**ことを明示するため、
//!    テスト名に `comparator_` を付けている

use fandhe_edge_core::hash::Sha256Digest;
use fandhe_edge_eval::invariance::{
    ComponentChange, EvaluationInvarianceError, ModelComponent, ModelPackageBytes,
    ModelPackagePaths, ModelPackageSnapshot, evaluate_with_invariance,
};
use fandhe_edge_eval::metrics::{EvalRecord, Outcome, SingleSelectMetrics, evaluate_single_select};
use std::fmt;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

const EPSILON: f64 = 1e-9;

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < EPSILON
}

/// 合成のモデルパッケージ本体（`static` 相当のバイト列）。
fn synthetic_package() -> ModelPackageBytes<'static> {
    ModelPackageBytes {
        weights: Some(b"synthetic-weights-bytes-v1"),
        vocab: Some(b"synthetic-vocab-bytes-v1"),
        calibration: Some(b"synthetic-calibration-bytes-v1"),
        thresholds: Some(b"synthetic-thresholds-bytes-v1"),
    }
}

fn golds() -> Vec<&'static str> {
    vec!["A", "A", "A", "A", "B", "B"]
}

/// 最頻値（majority）スタブ予測器（PoC-9 の majority baseline に相当）。
///
/// 実際の推論関数と同じ形で、モデルパッケージ（ここでは重み
/// [`ModelComponent::Weights`]）のファイルを実際にディスクから読み取り、
/// その内容から予測ラベルを決める。`synthetic_package()` の重み（`-v1` 終わり）
/// を渡すと多数派の "A" を、それ以外（改変された重み）を渡すと "B" を返す。
///
/// [`evaluate_with_invariance`] の評価クロージャに渡された
/// [`ModelPackagePaths`] からさらに読み込む点が実運用の評価器（CLI の
/// `evaluate` 工程）の経路と同じで、固定値を返すだけのスタブでは検出できない
/// 「評価処理自体がモデルパッケージを書き換える回帰」をこのテストで
/// 検出できるようにする（issue #214 の codex/review 指摘への対応）。
fn predict_using_package(paths: &ModelPackagePaths<'_>) -> Vec<Outcome> {
    let weights = fs::read(paths.weights).expect("重みファイルの読み込みに失敗しないはず");
    let majority_label = if weights.ends_with(b"-v1") { "A" } else { "B" };
    vec![Outcome::Label(majority_label.to_string()); 6]
}

/// 評価器の本来の経路（`predict_using_package` → `evaluate_single_select`）を
/// 実行し、gold 6 件（A×4, B×2）に対する指標を返す。
fn run_real_evaluation_path(paths: &ModelPackagePaths<'_>) -> Result<SingleSelectMetrics, String> {
    let outcomes = predict_using_package(paths);
    let golds = golds();
    let records: Vec<EvalRecord<'_>> = golds
        .iter()
        .zip(outcomes.iter())
        .map(|(gold, outcome)| EvalRecord { gold, outcome })
        .collect();
    let labels = ["A", "B"];
    evaluate_single_select(&labels, &records).map_err(|err| err.to_string())
}

/// テスト用の一時ディレクトリを、成否に関わらず（panic 時も含めて）確実に
/// 削除するためのガード（RAII）。`Drop` に任せることで、途中の `expect` が
/// panic した場合でも後始末が漏れない（手動の `cleanup()` 呼び出し漏れを防ぐ。
/// create-plan の検証方法「後始末は必ず行う」）。
struct TempDirGuard(PathBuf);

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// 排他的に作成できる一意な一時ディレクトリを作る。
///
/// プロセス ID と固定文字列だけを名前に使うと、PID の再利用や同一プロセス内
/// での並行テスト実行時に既存ディレクトリと衝突しうる
/// （`create_dir_all` は既存ディレクトリも受け入れてしまうため検出できない。
/// codex/review 指摘。PRRT_kwDOUq-SxM6mhunp）。ここではナノ秒精度の
/// タイムスタンプと試行回数を名前へ加え、`fs::create_dir`（`create_dir_all`
/// と異なり既存ディレクトリではエラーを返す排他的な作成）が
/// `AlreadyExists` を返した場合だけ名前を変えて再試行することで、
/// 名前の衝突を検出できる形にする（外部 crate〔tempfile 等〕は追加しない）。
fn make_unique_temp_dir(label: &str) -> PathBuf {
    let pid = std::process::id();
    for attempt in 0..1000u32 {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let candidate = std::env::temp_dir().join(format!(
            "fandhe-edge-eval-invariance-test-{pid}-{label}-{attempt}-{nanos}"
        ));
        match fs::create_dir(&candidate) {
            Ok(()) => return candidate,
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(err) => panic!("一時ディレクトリの作成に失敗しないはず: {err}"),
        }
    }
    panic!("一意な一時ディレクトリを {pid} 回試行しても作成できなかった");
}

fn write_temp_file(path: &Path, bytes: &[u8]) {
    let mut file = fs::File::create(path).expect("一時ファイルの作成に失敗しないはず");
    file.write_all(bytes)
        .expect("一時ファイルへの書き込みに失敗しないはず");
}

/// テストの評価クロージャのエラー型（`run_real_evaluation_path` の `String` を包む）。
#[derive(Debug)]
struct EvalStepError(String);

impl fmt::Display for EvalStepError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for EvalStepError {}

/// 評価前後で使う 4 構成要素のファイル一式を準備する（`evaluate_with_invariance`
/// を使う結合テスト専用のフィクスチャ）。
struct PackageFixture {
    _guard: TempDirGuard,
    weights: PathBuf,
    vocab: PathBuf,
    calibration: PathBuf,
    thresholds: PathBuf,
}

impl PackageFixture {
    fn new(label: &str) -> Self {
        let dir = make_unique_temp_dir(label);
        let guard = TempDirGuard(dir.clone());
        let package = synthetic_package();
        let weights = dir.join("weights.bin");
        let vocab = dir.join("vocab.bin");
        let calibration = dir.join("calibration.bin");
        let thresholds = dir.join("thresholds.bin");
        write_temp_file(&weights, package.weights.expect("値あり"));
        write_temp_file(&vocab, package.vocab.expect("値あり"));
        write_temp_file(&calibration, package.calibration.expect("値あり"));
        write_temp_file(&thresholds, package.thresholds.expect("値あり"));
        PackageFixture {
            _guard: guard,
            weights,
            vocab,
            calibration,
            thresholds,
        }
    }

    fn paths(&self) -> ModelPackagePaths<'_> {
        ModelPackagePaths {
            weights: &self.weights,
            vocab: Some(&self.vocab),
            calibration: Some(&self.calibration),
            thresholds: Some(&self.thresholds),
        }
    }
}

#[test]
fn req27_evaluate_with_invariance_normal_path_returns_metrics() {
    // REQ-27 正常系: 評価器の本来の経路（predict_using_package →
    // evaluate_single_select）を evaluate_with_invariance の評価クロージャの
    // 中で実行し、パッケージに変化が無ければ具体的な指標がそのまま返る
    // ことを確かめる。
    let fixture = PackageFixture::new("normal-path");
    let paths = fixture.paths();

    let result: Result<SingleSelectMetrics, EvaluationInvarianceError<EvalStepError>> =
        evaluate_with_invariance(&paths, |p| {
            run_real_evaluation_path(p).map_err(EvalStepError)
        });

    let metrics = result.expect("パッケージが変化しなければ Ok のはず");
    assert_eq!(metrics.n_total, 6);
    assert!(approx_eq(metrics.accuracy.overall.value(), 4.0 / 6.0));
}

/// 評価クロージャの内側で `component` に対応するファイルを実際に書き換えた
/// 場合に、`evaluate_with_invariance` が `Changed` として検出し、評価結果を
/// 返さないことを確かめる（REQ-27。issue #214 の codex/review 指摘への対応:
/// 改変は評価クロージャの内側〔評価処理そのもの〕で起こし、テスト自身が
/// 評価の外側で上書きしない）。
fn assert_evaluation_path_tampering_detected(
    component: ModelComponent,
    tampered_bytes: &'static [u8],
    original_bytes: &'static [u8],
) {
    let fixture = PackageFixture::new(component.as_str());
    let paths = fixture.paths();
    let target_path: PathBuf = match component {
        ModelComponent::Weights => fixture.weights.clone(),
        ModelComponent::Vocab => fixture.vocab.clone(),
        ModelComponent::Calibration => fixture.calibration.clone(),
        ModelComponent::Thresholds => fixture.thresholds.clone(),
        other => panic!(
            "assert_evaluation_path_tampering_detected は {other:?} 用のパスを定義していない"
        ),
    };

    let result: Result<SingleSelectMetrics, EvaluationInvarianceError<EvalStepError>> =
        evaluate_with_invariance(&paths, |p| {
            // 評価器の本来の経路を実際に実行してから、評価処理の一部として
            // 対象ファイルを書き換えたことを模擬する。
            let metrics = run_real_evaluation_path(p).map_err(EvalStepError)?;
            write_temp_file(&target_path, tampered_bytes);
            Ok(metrics)
        });

    match result {
        Err(EvaluationInvarianceError::Changed(violation)) => {
            assert_eq!(violation.changes.len(), 1);
            match &violation.changes[0] {
                ComponentChange::Modified {
                    component: changed,
                    before,
                    after,
                } => {
                    assert_eq!(*changed, component);
                    assert_eq!(*before, Sha256Digest::of_bytes(original_bytes));
                    assert_eq!(*after, Sha256Digest::of_bytes(tampered_bytes));
                }
                other => panic!("Modified({component:?}) を期待したが {other:?} だった"),
            }
        }
        other => panic!("Changed({component:?}) を期待したが {other:?} だった"),
    }
}

#[test]
fn req27_evaluate_with_invariance_detects_weights_tampered_by_evaluation_step() {
    assert_evaluation_path_tampering_detected(
        ModelComponent::Weights,
        b"tampered-weights-bytes-v2",
        b"synthetic-weights-bytes-v1",
    );
}

#[test]
fn req27_evaluate_with_invariance_detects_vocab_tampered_by_evaluation_step() {
    assert_evaluation_path_tampering_detected(
        ModelComponent::Vocab,
        b"tampered-vocab-bytes-v2",
        b"synthetic-vocab-bytes-v1",
    );
}

#[test]
fn req27_evaluate_with_invariance_detects_calibration_tampered_by_evaluation_step() {
    assert_evaluation_path_tampering_detected(
        ModelComponent::Calibration,
        b"tampered-calibration-bytes-v2",
        b"synthetic-calibration-bytes-v1",
    );
}

#[test]
fn req27_evaluate_with_invariance_detects_thresholds_tampered_by_evaluation_step() {
    assert_evaluation_path_tampering_detected(
        ModelComponent::Thresholds,
        b"tampered-thresholds-bytes-v2",
        b"synthetic-thresholds-bytes-v1",
    );
}

/// 評価クロージャの内側で `component` に対応するファイルを実際に削除した
/// 場合に、`evaluate_with_invariance` が `Changed(Removed)` として検出する
/// ことを確かめる（issue #226。`assert_evaluation_path_tampering_detected` と
/// 同じ形の削除版）。
fn assert_evaluation_path_removal_detected(
    component: ModelComponent,
    original_bytes: &'static [u8],
) {
    let fixture = PackageFixture::new(&format!("{}-removed", component.as_str()));
    let paths = fixture.paths();
    let target_path: PathBuf = match component {
        ModelComponent::Weights => fixture.weights.clone(),
        ModelComponent::Vocab => fixture.vocab.clone(),
        ModelComponent::Calibration => fixture.calibration.clone(),
        ModelComponent::Thresholds => fixture.thresholds.clone(),
        other => {
            panic!("assert_evaluation_path_removal_detected は {other:?} 用のパスを定義していない")
        }
    };

    let result: Result<SingleSelectMetrics, EvaluationInvarianceError<EvalStepError>> =
        evaluate_with_invariance(&paths, |p| {
            let metrics = run_real_evaluation_path(p).map_err(EvalStepError)?;
            fs::remove_file(&target_path).expect("削除に失敗しないはず");
            Ok(metrics)
        });

    match result {
        Err(EvaluationInvarianceError::Changed(violation)) => {
            assert_eq!(violation.changes.len(), 1);
            match &violation.changes[0] {
                ComponentChange::Removed {
                    component: removed,
                    before,
                } => {
                    assert_eq!(*removed, component);
                    assert_eq!(*before, Sha256Digest::of_bytes(original_bytes));
                }
                other => panic!("Removed({component:?}) を期待したが {other:?} だった"),
            }
        }
        other => panic!("Changed({component:?}) を期待したが {other:?} だった"),
    }
}

#[test]
fn req27_evaluate_with_invariance_reports_weights_removed_by_evaluation_step() {
    // 重みの削除は必須構成要素であり、`Snapshot(MissingWeights)` へ落ちず
    // `Removed{Weights}` として報告されることが本 Issue の主眼（issue #226）。
    assert_evaluation_path_removal_detected(ModelComponent::Weights, b"synthetic-weights-bytes-v1");
}

#[test]
fn req27_evaluate_with_invariance_reports_vocab_removed_by_evaluation_step() {
    assert_evaluation_path_removal_detected(ModelComponent::Vocab, b"synthetic-vocab-bytes-v1");
}

#[test]
fn req27_evaluate_with_invariance_reports_calibration_removed_by_evaluation_step() {
    assert_evaluation_path_removal_detected(
        ModelComponent::Calibration,
        b"synthetic-calibration-bytes-v1",
    );
}

#[test]
fn req27_evaluate_with_invariance_reports_thresholds_removed_by_evaluation_step() {
    assert_evaluation_path_removal_detected(
        ModelComponent::Thresholds,
        b"synthetic-thresholds-bytes-v1",
    );
}

#[test]
fn req27_evaluate_with_invariance_reports_removed_and_modified_in_declared_order() {
    // issue #226: 削除（Removed）と改変（Modified）が同時に起きた場合、
    // `ModelComponent` の宣言順（Weights < Vocab < Calibration < Thresholds）
    // で `changes` に並ぶことを確認する。
    let fixture = PackageFixture::new("removed-and-modified");
    let paths = fixture.paths();
    let vocab_path = fixture.vocab.clone();
    let thresholds_path = fixture.thresholds.clone();

    let result: Result<SingleSelectMetrics, EvaluationInvarianceError<EvalStepError>> =
        evaluate_with_invariance(&paths, |p| {
            let metrics = run_real_evaluation_path(p).map_err(EvalStepError)?;
            fs::remove_file(&vocab_path).expect("削除に失敗しないはず");
            write_temp_file(&thresholds_path, b"tampered-thresholds-bytes-v2");
            Ok(metrics)
        });

    match result {
        Err(EvaluationInvarianceError::Changed(violation)) => {
            assert_eq!(violation.changes.len(), 2);
            match &violation.changes[0] {
                ComponentChange::Removed { component, .. } => {
                    assert_eq!(*component, ModelComponent::Vocab);
                }
                other => panic!("先頭は Removed(Vocab) を期待したが {other:?} だった"),
            }
            match &violation.changes[1] {
                ComponentChange::Modified { component, .. } => {
                    assert_eq!(*component, ModelComponent::Thresholds);
                }
                other => panic!("2 件目は Modified(Thresholds) を期待したが {other:?} だった"),
            }
        }
        other => panic!("Changed を期待したが {other:?} だった"),
    }
}

#[test]
fn req27_evaluate_with_invariance_prefers_removed_over_evaluation_error() {
    // issue #226: 評価クロージャが構成要素を削除したうえで Err を返しても、
    // 削除の検出（Changed）が評価エラーより優先されることを確認する
    // （`req27_evaluate_with_invariance_prefers_changed_over_evaluation_error`
    // の削除版）。
    let fixture = PackageFixture::new("removed-then-eval-error");
    let paths = fixture.paths();
    let calibration_path = fixture.calibration.clone();

    let result: Result<SingleSelectMetrics, EvaluationInvarianceError<EvalStepError>> =
        evaluate_with_invariance(&paths, |_p| {
            fs::remove_file(&calibration_path).expect("削除に失敗しないはず");
            Err(EvalStepError("boom".to_string()))
        });

    match result {
        Err(EvaluationInvarianceError::Changed(violation)) => {
            assert_eq!(violation.changes.len(), 1);
            assert!(matches!(
                &violation.changes[0],
                ComponentChange::Removed {
                    component: ModelComponent::Calibration,
                    ..
                }
            ));
        }
        other => panic!("Changed を期待したが {other:?} だった"),
    }
}

#[test]
fn req27_req39_evaluate_with_invariance_keeps_non_not_found_error_after_evaluation() {
    // 受け入れ条件 2（issue #226）: NotFound 以外の読み込み失敗（ここでは
    // 通常ファイルの代わりにディレクトリが置かれた状態。ENOTDIR/特殊ファイル
    // の一種）は `Changed` に丸めず、`NotRegularFile` として返すことを確認
    // する。chmod によるアクセス権テストは root で実行される CI 環境
    // （`.claude/rules/ci.md`）で成立しないため使わない。
    let fixture = PackageFixture::new("not-found-vs-not-regular");
    let paths = fixture.paths();
    let thresholds_path = fixture.thresholds.clone();

    let result: Result<SingleSelectMetrics, EvaluationInvarianceError<EvalStepError>> =
        evaluate_with_invariance(&paths, |p| {
            let metrics = run_real_evaluation_path(p).map_err(EvalStepError)?;
            fs::remove_file(&thresholds_path).expect("削除に失敗しないはず");
            fs::create_dir(&thresholds_path).expect("ディレクトリの作成に失敗しないはず");
            Ok(metrics)
        });

    match result {
        Err(EvaluationInvarianceError::NotRegularFile { component, .. }) => {
            assert_eq!(component, ModelComponent::Thresholds);
        }
        other => panic!("NotRegularFile(Thresholds) を期待したが {other:?} だった"),
    }
    // `TempDirGuard`（`_guard`）の `remove_dir_all` が、置き換えた
    // ディレクトリごと後始末できることを確認する（成否に関わらず削除）。
}

#[test]
fn req27_comparator_in_memory_snapshots_of_unchanged_package_match() {
    // 比較器（`ModelPackageSnapshot::verify_unchanged`）単体のテスト。
    // 同じバイト列から作った 2 つのスナップショットが一致することのみを
    // 確かめる（評価経路には結合していない。実際の評価経路の検証は
    // `evaluate_with_invariance_*` 系のテストを参照）。
    let package = synthetic_package();
    let before = ModelPackageSnapshot::capture(&package).expect("合成データは失敗しないはず");
    let after = ModelPackageSnapshot::capture(&package).expect("合成データは失敗しないはず");
    assert_eq!(before.verify_unchanged(&after), Ok(()));

    let component_names: Vec<&str> = before.components().map(|(c, _)| c.as_str()).collect();
    assert_eq!(
        component_names,
        vec!["weights", "vocab", "calibration", "thresholds"]
    );
}

#[test]
fn req27_comparator_disk_round_trip_of_unchanged_files_matches() {
    // 比較器のディスク往復版: ファイルへ書く → 読む → 読み直す、という経路
    // でも一致することを確認する（ファイルの中身は変えていないため、これも
    // 比較器の一致判定のみのテスト）。
    let fixture = PackageFixture::new("comparator-disk-round-trip");

    let read_all = |path: &Path| -> Vec<u8> {
        fs::read(path).expect("一時ファイルの読み込みに失敗しないはず")
    };

    let weights_bytes = read_all(&fixture.weights);
    let vocab_bytes = read_all(&fixture.vocab);
    let calibration_bytes = read_all(&fixture.calibration);
    let thresholds_bytes = read_all(&fixture.thresholds);
    let before = ModelPackageSnapshot::capture(&ModelPackageBytes {
        weights: Some(&weights_bytes),
        vocab: Some(&vocab_bytes),
        calibration: Some(&calibration_bytes),
        thresholds: Some(&thresholds_bytes),
    })
    .expect("合成データは失敗しないはず");

    let weights_bytes_after = read_all(&fixture.weights);
    let vocab_bytes_after = read_all(&fixture.vocab);
    let calibration_bytes_after = read_all(&fixture.calibration);
    let thresholds_bytes_after = read_all(&fixture.thresholds);
    let after = ModelPackageSnapshot::capture(&ModelPackageBytes {
        weights: Some(&weights_bytes_after),
        vocab: Some(&vocab_bytes_after),
        calibration: Some(&calibration_bytes_after),
        thresholds: Some(&thresholds_bytes_after),
    })
    .expect("合成データは失敗しないはず");

    assert_eq!(before.verify_unchanged(&after), Ok(()));
}

#[test]
fn req27_comparator_reports_modified_thresholds() {
    // 比較器のテスト: 「しきい値のバイト列が変わっていた」という前後の
    // スナップショットを直接与えると `Modified(Thresholds)` を検出する
    // （評価経路には結合していない）。
    let before_package = synthetic_package();
    let before = ModelPackageSnapshot::capture(&before_package).expect("失敗しないはず");

    let tampered_thresholds: &[u8] = b"tampered-thresholds-bytes-v2";
    let mut after_package = before_package;
    after_package.thresholds = Some(tampered_thresholds);
    let after = ModelPackageSnapshot::capture(&after_package).expect("失敗しないはず");

    let violation = before.verify_unchanged(&after).unwrap_err();
    assert_eq!(violation.changes.len(), 1);
    match &violation.changes[0] {
        ComponentChange::Modified { component, .. } => {
            assert_eq!(*component, ModelComponent::Thresholds);
        }
        other => panic!("Modified(Thresholds) を期待したが {other:?} だった"),
    }
}

#[test]
fn req27_predict_using_package_reflects_weights_content() {
    // `predict_using_package` が実際にモデルパッケージ（重み）の内容に依存する
    // ことを確かめる。これにより、`evaluate_with_invariance_*` 系のテストで
    // 評価クロージャに渡す評価経路が、モデルパッケージへ実質的に結合した
    // 経路であることを示す（固定値を返すだけの評価経路では検出できない
    // 回帰の対比。codex/review 指摘。PRRT_kwDOUq-SxM6mhunl）。
    let fixture = PackageFixture::new("predict-reflects-weights");
    let paths = fixture.paths();
    let original_outcomes = predict_using_package(&paths);
    assert_eq!(original_outcomes, vec![Outcome::Label("A".to_string()); 6]);

    write_temp_file(&fixture.weights, b"tampered-weights-bytes-v2");
    let tampered_outcomes = predict_using_package(&paths);
    assert_eq!(tampered_outcomes, vec![Outcome::Label("B".to_string()); 6]);

    assert_ne!(original_outcomes, tampered_outcomes);
}
