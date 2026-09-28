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
//! 3. **そのスナップショットの取得元と同じモデルパッケージ（[`ModelPackageBytes`]）を
//!    実際に読み取る** 最頻値（majority）スタブ予測器（[`predict_using_package`]）で
//!    予測を作り、`fandhe_edge_eval::metrics::evaluate_single_select` で評価を
//!    実行し、具体的な正解率を確かめる（評価経路がモデルパッケージから
//!    独立した固定値を返すだけだと、評価工程がモデルを書き換える回帰を
//!    このテストで検出できないため。codex/review 指摘。PRRT_kwDOUq-SxM6mhunl）
//! 4. 評価後のスナップショットを（メモリ上・ディスク経由の 2 通りで）作り直し、
//!    `verify_unchanged` が `Ok(())` であることを確かめる
//! 5. 対比として、評価の途中でしきい値が書き換わったことを模擬すると
//!    `Modified { component: Thresholds, .. }` が検出されることを確かめる
//! 6. 追加の対比として、[`predict_using_package`] が重みバイト列の内容に
//!    実際に依存すること（重みが変われば予測結果も変わること）を確かめ、
//!    手順 3 の評価がモデルパッケージへ実質的に結合していることを示す

use fandhe_edge_core::hash::Sha256Digest;
use fandhe_edge_eval::invariance::{
    ComponentChange, ModelComponent, ModelPackageBytes, ModelPackageSnapshot,
};
use fandhe_edge_eval::metrics::{EvalRecord, Outcome, evaluate_single_select};
use std::fs;
use std::io::Write as _;
use std::path::Path;

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

/// 最頻値（majority）スタブ予測器（PoC-9 の majority baseline に相当）。
///
/// 実際の推論関数と同じ形で、モデルパッケージ（ここでは重み
/// [`ModelComponent::Weights`]）のバイト列を実際に読み取り、その内容から
/// 予測ラベルを決める。`synthetic_package()` の重み（`-v1` 終わり）を渡すと
/// 多数派の "A" を、それ以外（改変された重み）を渡すと "B" を返す。
///
/// 評価前後のスナップショット比較（[`ModelPackageSnapshot::capture`]）の
/// 間に、実際にモデルパッケージのバイト列を消費する評価経路を挟むための
/// ヘルパー（codex/review 指摘。PRRT_kwDOUq-SxM6mhunl。固定値を返すだけの
/// スタブだと、評価工程がモデルパッケージを書き換える回帰をこのテストで
/// 検出できない）。gold 6 件（A×4, B×2）に対して使う前提で、多数派 "A" の
/// 場合の正解率は 4/6 になる。
fn predict_using_package(package: &ModelPackageBytes<'_>) -> Vec<Outcome> {
    let weights = package
        .weights
        .expect("呼び出し側は capture と同じく weights を渡す契約");
    let majority_label = if weights.ends_with(b"-v1") { "A" } else { "B" };
    vec![Outcome::Label(majority_label.to_string()); 6]
}

fn golds() -> Vec<&'static str> {
    vec!["A", "A", "A", "A", "B", "B"]
}

#[test]
fn req27_task27_1_1_evaluation_does_not_change_model_package_hash() {
    let package = synthetic_package();

    // 1. 評価前のスナップショット。
    let before = ModelPackageSnapshot::capture(&package).expect("合成データは失敗しないはず");

    // 2. モデルパッケージ（重み）を実際に読み取る予測器で推論結果を集計する。
    //    評価器（`evaluate_single_select`）自体が触れるのはラベル文字列のみ
    //    だが、その手前の予測器はモデルパッケージのバイト列を消費しており、
    //    before/after のスナップショット取得の間に実際の評価経路が挟まる。
    let outcomes = predict_using_package(&package);
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
fn make_unique_temp_dir(label: &str) -> std::path::PathBuf {
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

#[test]
fn req27_task27_1_1_disk_round_trip_preserves_hash() {
    // ファイルへ書く → 読む → 評価 → 読み直す、という経路でも一致することを
    // 確認する（外部 crate〔tempfile 等〕を追加せず、標準ライブラリの
    // `std::env::temp_dir` のみを使う）。
    let package = synthetic_package();
    let dir = make_unique_temp_dir("req27-task27-1-1");
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

    // ディスクから読み直したモデルパッケージ（`read_back_before`）を実際に
    // 読み取る予測器で推論結果を集計する（PoC-9 InvarianceTest の手順どおり、
    // スナップショット取得と評価の間に実際の評価工程を挟む。
    // codex/review 指摘。PRRT_kwDOUq-SxM6mhunl）。
    let outcomes = predict_using_package(&read_back_before);
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

#[test]
fn req27_predict_using_package_reflects_weights_content() {
    // `predict_using_package` が実際にモデルパッケージ（重み）の内容に依存する
    // ことを確かめる。これにより、手順 3（
    // `req27_task27_1_1_evaluation_does_not_change_model_package_hash` /
    // `req27_task27_1_1_disk_round_trip_preserves_hash`）で before/after の
    // スナップショット取得の間に挟む評価が、モデルパッケージへ実質的に
    // 結合した経路であることを示す（固定値を返すだけの評価経路では
    // 検出できない回帰の対比。codex/review 指摘。PRRT_kwDOUq-SxM6mhunl）。
    let original = synthetic_package();
    let original_outcomes = predict_using_package(&original);
    assert_eq!(original_outcomes, vec![Outcome::Label("A".to_string()); 6]);

    let tampered_weights: &[u8] = b"tampered-weights-bytes-v2";
    let mut tampered = original;
    tampered.weights = Some(tampered_weights);
    let tampered_outcomes = predict_using_package(&tampered);
    assert_eq!(tampered_outcomes, vec![Outcome::Label("B".to_string()); 6]);

    assert_ne!(original_outcomes, tampered_outcomes);
}

/// `synthetic_package()` の各構成要素の初期バイト列
/// （改変前にディスクへ書き込む内容と、期待するダイジェストの根拠に使う）。
fn original_component_bytes(component: ModelComponent) -> &'static [u8] {
    match component {
        ModelComponent::Weights => b"synthetic-weights-bytes-v1",
        ModelComponent::Vocab => b"synthetic-vocab-bytes-v1",
        ModelComponent::Calibration => b"synthetic-calibration-bytes-v1",
        ModelComponent::Thresholds => b"synthetic-thresholds-bytes-v1",
        // `ModelComponent` は `#[non_exhaustive]` だが、本 crate 外に新規
        // バリアントを追加する手段は無いため、この wildcard は将来
        // 構成要素が増えた場合にのみ到達しうる。到達したら「改変の再現に
        // 使う初期値が未定義」であることを明示して panic させ、テストの
        // 期待値が古いまま黙って通る（fail-open になる）のを防ぐ。
        other => panic!("original_component_bytes は {other:?} の初期値を定義していない"),
    }
}

/// バイト列を一時ファイルへ書き込む（新規作成・上書きの両方に使う）。
fn write_temp_file(path: &Path, bytes: &[u8]) {
    let mut file = fs::File::create(path).expect("一時ファイルの作成に失敗しないはず");
    file.write_all(bytes)
        .expect("一時ファイルへの書き込みに失敗しないはず");
}

/// 一時ファイルをディスクから読み込む。
fn read_temp_file(path: &Path) -> Vec<u8> {
    fs::read(path).expect("一時ファイルの読み込みに失敗しないはず")
}

/// 指定した構成要素だけをディスク上で改変した場合に、評価前後の
/// スナップショット比較（[`ModelPackageSnapshot::verify_unchanged`]）が
/// 実際に `ComponentChange::Modified { component, .. }` を検出することを
/// 確かめる（REQ-27「評価の独立性」。TASK-27.1-1・issue #69）。
///
/// これまでの `req27_task27_1_1_disk_round_trip_preserves_hash` は評価の
/// 前後で同じファイルを書き換えずに読み直すだけだったため、評価工程が
/// 実際にファイルを変更したときに前後比較が失敗することまでは検証できて
/// いなかった（codex/review 指摘。issue #214 の PR コメント）。本関数は
/// 評価前スナップショット取得 → 評価の実行 → **対象ファイルをディスク上で
/// 上書き** → 評価後スナップショット取得、という順序でディスクファイルを
/// 実際に変更し、変更した構成要素だけが `Modified` として検出されることを
/// 具体値（変更前後の sha256 ダイジェスト）で確かめる。
fn assert_disk_tampering_detected(component: ModelComponent, tampered_bytes: &'static [u8]) {
    let dir = make_unique_temp_dir(component.as_str());
    let _guard = TempDirGuard(dir.clone());

    let weights_path = dir.join("weights.bin");
    let vocab_path = dir.join("vocab.bin");
    let calibration_path = dir.join("calibration.bin");
    let thresholds_path = dir.join("thresholds.bin");

    write_temp_file(
        &weights_path,
        original_component_bytes(ModelComponent::Weights),
    );
    write_temp_file(&vocab_path, original_component_bytes(ModelComponent::Vocab));
    write_temp_file(
        &calibration_path,
        original_component_bytes(ModelComponent::Calibration),
    );
    write_temp_file(
        &thresholds_path,
        original_component_bytes(ModelComponent::Thresholds),
    );

    // 評価前: 改変前の内容をディスクから読み込んでスナップショットを作る。
    let weights_bytes = read_temp_file(&weights_path);
    let vocab_bytes = read_temp_file(&vocab_path);
    let calibration_bytes = read_temp_file(&calibration_path);
    let thresholds_bytes = read_temp_file(&thresholds_path);
    let before_package = ModelPackageBytes {
        weights: Some(&weights_bytes),
        vocab: Some(&vocab_bytes),
        calibration: Some(&calibration_bytes),
        thresholds: Some(&thresholds_bytes),
    };
    let before = ModelPackageSnapshot::capture(&before_package).expect("失敗しないはず");

    // 評価工程を実際に走らせ、改変前のバイト列を予測器に消費させる
    // （手順 3 と同じく、before/after のスナップショット取得の間に実際の
    // 評価経路を挟む）。
    let outcomes = predict_using_package(&before_package);
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

    // 評価の途中で対象の構成要素だけがディスク上で改変されたことを模擬する。
    let target_path: &Path = match component {
        ModelComponent::Weights => &weights_path,
        ModelComponent::Vocab => &vocab_path,
        ModelComponent::Calibration => &calibration_path,
        ModelComponent::Thresholds => &thresholds_path,
        other => panic!("assert_disk_tampering_detected は {other:?} 用のパスを定義していない"),
    };
    write_temp_file(target_path, tampered_bytes);

    // 評価後: 改変後のファイルをディスクから読み直す（改変前のバッファを
    // 使い回さない。契約はモジュール冒頭の doc のとおり）。
    let weights_bytes_after = read_temp_file(&weights_path);
    let vocab_bytes_after = read_temp_file(&vocab_path);
    let calibration_bytes_after = read_temp_file(&calibration_path);
    let thresholds_bytes_after = read_temp_file(&thresholds_path);
    let after_package = ModelPackageBytes {
        weights: Some(&weights_bytes_after),
        vocab: Some(&vocab_bytes_after),
        calibration: Some(&calibration_bytes_after),
        thresholds: Some(&thresholds_bytes_after),
    };
    let after = ModelPackageSnapshot::capture(&after_package).expect("失敗しないはず");

    let violation = before.verify_unchanged(&after).unwrap_err();
    assert_eq!(violation.changes.len(), 1);
    match &violation.changes[0] {
        ComponentChange::Modified {
            component: changed,
            before: before_digest,
            after: after_digest,
        } => {
            assert_eq!(*changed, component);
            assert_eq!(
                *before_digest,
                Sha256Digest::of_bytes(original_component_bytes(component))
            );
            assert_eq!(*after_digest, Sha256Digest::of_bytes(tampered_bytes));
        }
        other => panic!("Modified({component:?}) を期待したが {other:?} だった"),
    }
}

#[test]
fn req27_disk_tampering_of_weights_during_evaluation_is_detected() {
    assert_disk_tampering_detected(ModelComponent::Weights, b"tampered-weights-bytes-v2");
}

#[test]
fn req27_disk_tampering_of_vocab_during_evaluation_is_detected() {
    assert_disk_tampering_detected(ModelComponent::Vocab, b"tampered-vocab-bytes-v2");
}

#[test]
fn req27_disk_tampering_of_calibration_during_evaluation_is_detected() {
    assert_disk_tampering_detected(
        ModelComponent::Calibration,
        b"tampered-calibration-bytes-v2",
    );
}

#[test]
fn req27_disk_tampering_of_thresholds_during_evaluation_is_detected() {
    assert_disk_tampering_detected(ModelComponent::Thresholds, b"tampered-thresholds-bytes-v2");
}
