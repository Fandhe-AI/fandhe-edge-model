//! REQ-16 異常系・TASK-16.2-1: 漏洩・group 跨ぎ検出の結合テスト（PoC-9 の
//! `InjectionTest` 相当）。
//!
//! PoC-9（`03-poc/evaluation-contract/fixtures/injection/{clean,leak-duplicate,
//! group-straddle}`）と同じ構造の注入パターンを、本テスト内で組み立てた
//! **合成データ**で再現する。PoC の jsonl は本テストへコピーしない（データ本文は
//! LLM 生成物で、生成元・利用条件の記録が要る。licensing.md）。
//!
//! 証拠の種別: テストハーネス（合成データによる注入テスト）。PoC-9 の参照データを
//! 使った実測の再現ではなく、同じ注入パターンの構造的な再現である。

use fandhe_edge_data::leak::{LeakCheckable, Partitions, find_group_straddles, find_input_leaks};

/// テスト用の最小レコード。
struct Record {
    id: String,
    input: Vec<u8>,
    group_id: String,
}

impl LeakCheckable for Record {
    fn id(&self) -> &str {
        &self.id
    }
    fn input(&self) -> &[u8] {
        &self.input
    }
    fn group_id(&self) -> &str {
        &self.group_id
    }
}

fn record(id: &str, input: &str, group_id: &str) -> Record {
    Record {
        id: id.to_string(),
        input: input.as_bytes().to_vec(),
        group_id: group_id.to_string(),
    }
}

/// clean な train 10 行・test 10 行（入力・group_id とも互いに重ならない）。
fn clean_train_and_test() -> (Vec<Record>, Vec<Record>) {
    let train = (0..10)
        .map(|i| {
            record(
                &format!("train-{i}"),
                &format!("train input {i}"),
                &format!("train-group-{i}"),
            )
        })
        .collect();
    let test = (0..10)
        .map(|i| {
            record(
                &format!("test-{i}"),
                &format!("test input {i}"),
                &format!("test-group-{i}"),
            )
        })
        .collect();
    (train, test)
}

/// REQ-16 異常系・TASK-16.2-1（PoC-9 InjectionTest 相当・clean）: 互いに重ならない
/// train・test では漏洩 0 件・跨ぎ 0 件（誤検出 0 件）。
#[test]
fn req16_task16_2_1_clean_fixture_has_zero_leaks_and_zero_straddles() {
    let (train, test) = clean_train_and_test();
    let partitions = Partitions {
        train: &train,
        validation: None,
        test: Some(&test),
        evaluation: None,
    };

    let leak_report = find_input_leaks(&partitions).expect("上限以下の入力");
    let straddle_report = find_group_straddles(&partitions).expect("上限以下の入力");
    assert_eq!(leak_report.leak_pair_count(), 0, "誤検出（漏洩）が発生した");
    assert_eq!(leak_report.leaked_rows(), 0, "誤検出（漏洩）が発生した");
    assert_eq!(
        straddle_report.straddles.len(),
        0,
        "誤検出（group 跨ぎ）が発生した"
    );
}

/// REQ-16 異常系・TASK-16.2-1（PoC-9 InjectionTest 相当・leak-duplicate）: clean を
/// 基に、train の 2 行と byte 単位で同一の入力を持つ test 行を 2 件注入し、さらに
/// train 内に同一入力の重複を 1 件注入する。
///
/// 期待: 漏洩した入力の種類数 2、漏洩行数 2。train 内の重複が件数を増やしも
/// 減らしもしない。
#[test]
fn req16_task16_2_1_leak_duplicate_fixture_detects_exact_leak_count() {
    let (mut train, mut test) = clean_train_and_test();

    // train の 2 行（train-0, train-1）と同一入力を持つ test 行を 2 件注入する。
    test.push(record("leak-test-a", "train input 0", "leak-test-group-a"));
    test.push(record("leak-test-b", "train input 1", "leak-test-group-b"));

    // train 内の重複（train-0 と同一入力の追加行）: 漏洩件数を増やしも減らしもしない。
    train.push(record("train-0-dup", "train input 0", "train-group-0-dup"));

    let partitions = Partitions {
        train: &train,
        validation: None,
        test: Some(&test),
        evaluation: None,
    };

    let report = find_input_leaks(&partitions).expect("上限以下の入力");
    assert_eq!(
        report.leak_pair_count(),
        2,
        "漏洩した入力の種類数が一致しない"
    );
    assert_eq!(report.leaked_rows(), 2, "漏洩行数が一致しない");

    // train_ids・other_ids を具体的な ID 列で照合する。
    let mut leaks = report.leaks.clone();
    leaks.sort_by(|a, b| a.other_ids.cmp(&b.other_ids));

    let leak_a = &leaks[0];
    assert_eq!(leak_a.other_ids, vec!["leak-test-a".to_string()]);
    assert_eq!(
        &*leak_a.train_ids,
        ["train-0".to_string(), "train-0-dup".to_string()],
        "train 内の重複行も train_ids に含まれる"
    );

    let leak_b = &leaks[1];
    assert_eq!(leak_b.other_ids, vec!["leak-test-b".to_string()]);
    assert_eq!(&*leak_b.train_ids, ["train-1".to_string()]);

    // group 跨ぎは注入していないため 0 件のまま。
    let straddle_report = find_group_straddles(&partitions).expect("上限以下の入力");
    assert_eq!(straddle_report.straddles.len(), 0);
}

/// REQ-16 異常系・TASK-16.2-1（PoC-9 InjectionTest 相当・group-straddle）: clean を
/// 基に、train の 1 行と同じ group_id を持ち、入力は異なる test 行を 2 件注入する。
///
/// 期待: 漏洩 0 件、跨ぎ 1 件（`members` = train に 1 ID、test に 2 ID）。
#[test]
fn req16_task16_2_1_group_straddle_fixture_detects_straddle_without_input_leak() {
    let (train, mut test) = clean_train_and_test();

    // train-0 の group_id（"train-group-0"）を、入力は異なる test 行 2 件に付与する。
    test.push(record(
        "straddle-test-a",
        "completely different input a",
        "train-group-0",
    ));
    test.push(record(
        "straddle-test-b",
        "completely different input b",
        "train-group-0",
    ));

    let partitions = Partitions {
        train: &train,
        validation: None,
        test: Some(&test),
        evaluation: None,
    };

    let leak_report = find_input_leaks(&partitions).expect("上限以下の入力");
    assert_eq!(
        leak_report.leak_pair_count(),
        0,
        "入力が異なるため漏洩は検出されない"
    );

    let straddle_report = find_group_straddles(&partitions).expect("上限以下の入力");
    assert_eq!(straddle_report.straddles.len(), 1);
    let straddle = &straddle_report.straddles[0];
    assert_eq!(straddle.group_id, "train-group-0");
    assert_eq!(
        straddle
            .members
            .get(&fandhe_edge_data::leak::Partition::Train),
        Some(&vec!["train-0".to_string()])
    );
    assert_eq!(
        straddle
            .members
            .get(&fandhe_edge_data::leak::Partition::Test),
        Some(&vec![
            "straddle-test-a".to_string(),
            "straddle-test-b".to_string()
        ])
    );
}

/// REQ-16 異常系・TASK-16.2-1: 利用者の評価データ（Evaluation）側への漏洩と跨ぎも
/// partition = Evaluation として検出される。
#[test]
fn req16_task16_2_1_evaluation_partition_detects_leak_and_straddle() {
    let (train, test) = clean_train_and_test();
    let evaluation = vec![
        // 入力漏洩: train-0 と同一入力。
        record("eval-leak", "train input 0", "eval-leak-group"),
        // group 跨ぎ: train-1 と同一 group_id、入力は異なる。
        record(
            "eval-straddle",
            "unrelated evaluation input",
            "train-group-1",
        ),
    ];

    let partitions = Partitions {
        train: &train,
        validation: None,
        test: Some(&test),
        evaluation: Some(&evaluation),
    };

    let leak_report = find_input_leaks(&partitions).expect("上限以下の入力");
    let evaluation_leaks: Vec<_> = leak_report
        .leaks
        .iter()
        .filter(|leak| leak.partition == fandhe_edge_data::leak::Partition::Evaluation)
        .collect();
    assert_eq!(evaluation_leaks.len(), 1);
    assert_eq!(evaluation_leaks[0].other_ids, vec!["eval-leak".to_string()]);

    let straddle_report = find_group_straddles(&partitions).expect("上限以下の入力");
    let evaluation_straddles: Vec<_> = straddle_report
        .straddles
        .iter()
        .filter(|s| {
            s.members
                .contains_key(&fandhe_edge_data::leak::Partition::Evaluation)
        })
        .collect();
    assert_eq!(evaluation_straddles.len(), 1);
    assert_eq!(evaluation_straddles[0].group_id, "train-group-1");
}
