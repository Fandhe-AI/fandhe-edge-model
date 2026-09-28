//! モデルパッケージの評価前後ハッシュ比較の結合テスト。
//!
//! 出典: PoC-9 `InvarianceTest`（`docs/spec/03-poc/evaluation-contract/`）の
//! 「モデルパッケージ側」を移植する（REQ-27 正常系・TASK-27.1-1・issue #69。
//! `docs/spec` は読み込まず、手順の再現のみを行う。`.claude/rules/spec-reference.md`
//! のビルド独立方針）。
//!
//! 証拠の種別: テストハーネス（合成データによる結合テスト）。実機ではない。
//!
//! 手順:
//! 1. 合成のモデルパッケージ（重み・語彙・校正・しきい値の 4 構成要素）を用意する
//! 2. 評価前のスナップショットを作る
//! 3. そのパッケージを最頻値（majority）スタブ予測器で使い、
//!    `fandhe_edge_eval::metrics::evaluate_single_select` で評価を実行し、
//!    具体的な正解率を確かめる
//! 4. 評価後のスナップショットを（メモリ上・ディスク経由の 2 通りで）作り直し、
//!    `verify_unchanged` が `Ok(())` であることを確かめる
//! 5. 対比として、評価の途中でしきい値が書き換わったことを模擬すると
//!    `Modified { component: Thresholds, .. }` が検出されることを確かめる

use fandhe_edge_eval::invariance::{
    ComponentChange, ModelComponent, ModelPackageBytes, ModelPackageSnapshot,
};
use fandhe_edge_eval::metrics::{EvalRecord, Outcome, evaluate_single_select};
use std::fs;
use std::io::Write as _;

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

/// 最頻値（majority）スタブ予測器（PoC-9 の majority baseline に相当）を使い、
/// gold 6 件（A×4, B×2）に対する予測を作る。予測は常に多数派の "A" を返す
/// ため、正解率は 4/6 になる。
fn majority_predictions() -> Vec<Outcome> {
    vec![Outcome::Label("A".to_string()); 6]
}

fn golds() -> Vec<&'static str> {
    vec!["A", "A", "A", "A", "B", "B"]
}

#[test]
fn req27_task27_1_1_evaluation_does_not_change_model_package_hash() {
    let package = synthetic_package();

    // 1. 評価前のスナップショット。
    let before = ModelPackageSnapshot::capture(&package).expect("合成データは失敗しないはず");

    // 2. 評価器を経由して推論結果を集計する（モデルパッケージのバイト列自体は
    //    評価器へ渡さない。評価器が触れるのはラベル文字列のみ）。
    let outcomes = majority_predictions();
    let golds = golds();
    let records: Vec<EvalRecord<'_>> = golds
        .iter()
        .zip(outcomes.iter())
        .map(|(gold, outcome)| EvalRecord { gold, outcome })
        .collect();
    let labels = ["A", "B"];
    let metrics =
        evaluate_single_select(&labels, &records).expect("既知解データセットは失敗しないはず");
    assert_eq!(metrics.n_total, 6);
    assert!(approx_eq(metrics.accuracy.overall.value(), 4.0 / 6.0));

    // 3. 評価後のスナップショット（同じバイト列から作り直す。契約上は
    //    ディスクから読み直すべきだが、本テストでは合成データをそのまま使う
    //    ケースと、実際にファイルへ書いて読み直すケースの両方を確認する）。
    let after = ModelPackageSnapshot::capture(&package).expect("合成データは失敗しないはず");
    assert_eq!(before.verify_unchanged(&after), Ok(()));

    // components() が宣言順で走査でき、4 構成要素すべてが含まれることを確認する。
    let component_names: Vec<&str> = before.components().map(|(c, _)| c.as_str()).collect();
    assert_eq!(
        component_names,
        vec!["weights", "vocab", "calibration", "thresholds"]
    );
}

/// テスト用の一時ディレクトリを、成否に関わらず（panic 時も含めて）確実に
/// 削除するためのガード（RAII）。`Drop` に任せることで、途中の `expect` が
/// panic した場合でも後始末が漏れない（手動の `cleanup()` 呼び出し漏れを防ぐ。
/// create-plan の検証方法「後始末は必ず行う」）。
struct TempDirGuard(std::path::PathBuf);

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn req27_task27_1_1_disk_round_trip_preserves_hash() {
    // ファイルへ書く → 読む → 評価 → 読み直す、という経路でも一致することを
    // 確認する（外部 crate〔tempfile 等〕を追加せず、標準ライブラリの
    // `std::env::temp_dir` のみを使う）。
    let package = synthetic_package();
    let dir = std::env::temp_dir().join(format!(
        "fandhe-edge-eval-invariance-test-{}-{}",
        std::process::id(),
        "req27-task27-1-1"
    ));
    fs::create_dir_all(&dir).expect("一時ディレクトリの作成に失敗しないはず");
    let _guard = TempDirGuard(dir.clone());

    let write_component = |name: &str, bytes: &[u8]| -> std::path::PathBuf {
        let path = dir.join(name);
        let mut file = fs::File::create(&path).expect("一時ファイルの作成に失敗しないはず");
        file.write_all(bytes)
            .expect("一時ファイルへの書き込みに失敗しないはず");
        path
    };

    let weights_path = write_component("weights.bin", package.weights.expect("値あり"));
    let vocab_path = write_component("vocab.bin", package.vocab.expect("値あり"));
    let calibration_path = write_component("calibration.bin", package.calibration.expect("値あり"));
    let thresholds_path = write_component("thresholds.bin", package.thresholds.expect("値あり"));

    let read_all = |path: &std::path::Path| -> Vec<u8> {
        fs::read(path).expect("一時ファイルの読み込みに失敗しないはず")
    };

    // ディスクから読み込んだバイト列から評価前のスナップショットを作る。
    let weights_bytes = read_all(&weights_path);
    let vocab_bytes = read_all(&vocab_path);
    let calibration_bytes = read_all(&calibration_path);
    let thresholds_bytes = read_all(&thresholds_path);
    let read_back_before = ModelPackageBytes {
        weights: Some(&weights_bytes),
        vocab: Some(&vocab_bytes),
        calibration: Some(&calibration_bytes),
        thresholds: Some(&thresholds_bytes),
    };
    let before =
        ModelPackageSnapshot::capture(&read_back_before).expect("合成データは失敗しないはず");

    // このスナップショットとスナップショット取得元のバイト列とは無関係に、
    // 評価器を経由して推論結果を集計する（PoC-9 InvarianceTest の手順どおり、
    // スナップショット取得と評価の間に実際の評価工程を挟む）。
    let outcomes = majority_predictions();
    let golds = golds();
    let records: Vec<EvalRecord<'_>> = golds
        .iter()
        .zip(outcomes.iter())
        .map(|(gold, outcome)| EvalRecord { gold, outcome })
        .collect();
    let labels = ["A", "B"];
    let metrics =
        evaluate_single_select(&labels, &records).expect("既知解データセットは失敗しないはず");
    assert_eq!(metrics.n_total, 6);
    assert!(approx_eq(metrics.accuracy.overall.value(), 4.0 / 6.0));

    // 評価後、同じファイルを再度読み直す（評価前のバッファ
    // `weights_bytes` 等を使い回さず、新しい `Vec` へ読み直す）。
    let weights_bytes_after = read_all(&weights_path);
    let vocab_bytes_after = read_all(&vocab_path);
    let calibration_bytes_after = read_all(&calibration_path);
    let thresholds_bytes_after = read_all(&thresholds_path);
    let read_back_after = ModelPackageBytes {
        weights: Some(&weights_bytes_after),
        vocab: Some(&vocab_bytes_after),
        calibration: Some(&calibration_bytes_after),
        thresholds: Some(&thresholds_bytes_after),
    };
    let after =
        ModelPackageSnapshot::capture(&read_back_after).expect("合成データは失敗しないはず");

    assert_eq!(before.verify_unchanged(&after), Ok(()));
}

#[test]
fn req27_task27_1_1_threshold_tampering_during_evaluation_is_detected() {
    // 「評価の途中でしきい値のバイト列が書き換わった」ことを模擬する対比テスト。
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
