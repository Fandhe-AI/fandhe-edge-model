//! 評価データ凍結（[`fandhe_edge_data::eval_freeze`]）の結合テスト
//! （REQ-17・TASK-17.2-1・issue #47）。
//!
//! 公開 API を通した具体値の照合と、記録（`FreezeRecord`）の JSON 直列化・
//! 逆直列化が fail-closed になることを固定する。ゴールデン値は
//! `printf '%s' '<bytes>' | sha256sum`（証拠の種別: テストハーネス。
//! `.claude/rules/evaluation-contract.md`「決定性と証拠の種別」）で独立に
//! 計算した値と一致することを実装時に確認済み。

use fandhe_edge_data::eval_freeze::{
    EvalDataState, EvaluateGate, FreezeError, FreezeRecord, LoadFreezeRecordError,
    MAX_FREEZE_RECORD_JSON_BYTES, evaluate_gate, freeze_eval_data,
};

/// サンプル評価データ本体（JSONL・2 行・末尾 LF）。113 バイト。
const SAMPLE: &[u8] =
    b"{\"id\":\"e1\",\"input\":\"hello\",\"output\":{\"intent\":\"greet\"}}\n{\"id\":\"e2\",\"input\":\"bye\",\"output\":{\"intent\":\"farewell\"}}\n";

const SAMPLE_SHA256: &str = "edb189cef6cd5da4187a462786a777e93d58920566931e32d5dfb5d77ed39f62";

/// `SAMPLE` の `farewell` を 1 文字だけ変えた（`farewelL`）評価データ本体。
const SAMPLE_ONE_BYTE_CHANGED: &[u8] =
    b"{\"id\":\"e1\",\"input\":\"hello\",\"output\":{\"intent\":\"greet\"}}\n{\"id\":\"e2\",\"input\":\"bye\",\"output\":{\"intent\":\"farewelL\"}}\n";

const SAMPLE_ONE_BYTE_CHANGED_SHA256: &str =
    "306b04c17f66a94946eccceb1b4fdd0ceb1ec7dbcf2dc021be8adab8b56aa53a";

/// REQ-17: サンプル評価データを凍結すると、独立に計算した sha256・バイト長
/// と一致する。
#[test]
fn req17_freeze_eval_data_matches_independently_computed_sha256() {
    let record = freeze_eval_data(SAMPLE).expect("凍結は失敗しないはず");
    assert_eq!(record.sha256().to_hex(), SAMPLE_SHA256);
    assert_eq!(record.byte_len(), 113);
}

/// REQ-17: 1 バイトだけ変えた入力は異なる sha256 になる（#49 の不一致検知の
/// 前提となる性質）。バイト長は変わらない。
#[test]
fn req17_freeze_eval_data_differs_on_single_byte_change() {
    let original = freeze_eval_data(SAMPLE).expect("凍結は失敗しないはず");
    let changed = freeze_eval_data(SAMPLE_ONE_BYTE_CHANGED).expect("凍結は失敗しないはず");

    assert_eq!(changed.sha256().to_hex(), SAMPLE_ONE_BYTE_CHANGED_SHA256);
    assert_ne!(original.sha256(), changed.sha256());
    assert_eq!(original.byte_len(), changed.byte_len());
}

/// REQ-17: 同じ入力を 2 回凍結すると `FreezeRecord` が完全に一致する
/// （決定性。`.claude/rules/evaluation-contract.md`「決定性と証拠の種別」）。
#[test]
fn req17_freeze_eval_data_is_deterministic_across_calls() {
    let first = freeze_eval_data(SAMPLE).expect("凍結は失敗しないはず");
    let second = freeze_eval_data(SAMPLE).expect("凍結は失敗しないはず");
    assert_eq!(first, second);
}

/// REQ-17: `FreezeRecord` の JSON 表現がキー順を含め完全に一致し、
/// 逆直列化で元の値に戻る。
#[test]
fn req17_freeze_record_serializes_to_expected_json_and_round_trips() {
    let record = freeze_eval_data(SAMPLE).expect("凍結は失敗しないはず");
    let json = serde_json::to_string(&record).expect("serialize は成功するはず");
    assert_eq!(
        json,
        format!("{{\"algorithm\":\"sha256\",\"sha256\":\"{SAMPLE_SHA256}\",\"byte_len\":113}}")
    );

    let parsed: FreezeRecord = serde_json::from_str(&json).expect("deserialize は成功するはず");
    assert_eq!(parsed, record);
}

/// REQ-17: `FreezeRecord` の逆直列化は fail-closed になる。未知キー・
/// 不正なハッシュ方式・不正な sha256（大文字・桁数違い）・`byte_len` の
/// キー欠落は、いずれも `Err` になる。
#[test]
fn req17_freeze_record_deserialize_rejects_malformed_records() {
    let valid =
        format!("{{\"algorithm\":\"sha256\",\"sha256\":\"{SAMPLE_SHA256}\",\"byte_len\":113}}");
    assert!(serde_json::from_str::<FreezeRecord>(&valid).is_ok());

    let unknown_field = format!(
        "{{\"algorithm\":\"sha256\",\"sha256\":\"{SAMPLE_SHA256}\",\"byte_len\":113,\"path\":\"eval.jsonl\"}}"
    );
    assert!(serde_json::from_str::<FreezeRecord>(&unknown_field).is_err());

    let unknown_algorithm =
        format!("{{\"algorithm\":\"md5\",\"sha256\":\"{SAMPLE_SHA256}\",\"byte_len\":113}}");
    assert!(serde_json::from_str::<FreezeRecord>(&unknown_algorithm).is_err());

    let uppercase_hash = format!(
        "{{\"algorithm\":\"sha256\",\"sha256\":\"{}\",\"byte_len\":113}}",
        SAMPLE_SHA256.to_uppercase()
    );
    assert!(serde_json::from_str::<FreezeRecord>(&uppercase_hash).is_err());

    let short_hash = format!(
        "{{\"algorithm\":\"sha256\",\"sha256\":\"{}\",\"byte_len\":113}}",
        &SAMPLE_SHA256[..63]
    );
    assert!(serde_json::from_str::<FreezeRecord>(&short_hash).is_err());

    let missing_byte_len = format!("{{\"algorithm\":\"sha256\",\"sha256\":\"{SAMPLE_SHA256}\"}}");
    assert!(serde_json::from_str::<FreezeRecord>(&missing_byte_len).is_err());

    // `byte_len` は `u64` のため負数は表現できず、逆直列化時に拒否される。
    let negative_byte_len =
        format!("{{\"algorithm\":\"sha256\",\"sha256\":\"{SAMPLE_SHA256}\",\"byte_len\":-1}}");
    assert!(serde_json::from_str::<FreezeRecord>(&negative_byte_len).is_err());
}

/// REQ-17: 空入力は `NotProvided` ではなく `Frozen` として扱われ、
/// `actual_bytes` が記録と一致する限り `evaluate_gate` は `Proceed` に写る
/// （評価データなしの境界とは区別する。`crates/data/src/eval_freeze.rs`
/// モジュール doc 参照）。
#[test]
fn req17_empty_input_is_frozen_not_not_provided() {
    let record = freeze_eval_data(b"").expect("空入力の凍結は失敗しないはず");
    let state = EvalDataState::Frozen(record.clone());
    assert_eq!(
        evaluate_gate(&state, b""),
        Ok(EvaluateGate::Proceed(record))
    );
}

/// REQ-17: 凍結記録と実データが一致しない場合、`evaluate_gate` は
/// fail-closed で `HashMismatch` を返す（PR #209 レビュー指摘・issue #47
/// スレッド。`.claude/rules/evaluation-contract.md`「データの分割と凍結」）。
#[test]
fn req17_evaluate_gate_fails_closed_on_hash_mismatch() {
    let record = freeze_eval_data(SAMPLE).expect("凍結は失敗しないはず");
    let state = EvalDataState::Frozen(record);
    assert_eq!(
        evaluate_gate(&state, SAMPLE_ONE_BYTE_CHANGED),
        Err(FreezeError::HashMismatch)
    );
}

/// REQ-17・REQ-39: `MAX_FREEZE_RECORD_JSON_BYTES` を超える JSON 文字列は、
/// `serde_json` へ渡す前に `FreezeRecord::parse` が `TooLarge` として拒否する
/// （PR #209 の未解決 P0: 凍結記録の `Deserialize` は JSON 全体のサイズを
/// 制限しないため、公開の読み込み口〔`parse`・`load`〕で強制する）。
#[test]
fn req17_req39_parse_rejects_json_over_size_limit() {
    let record = freeze_eval_data(SAMPLE).expect("凍結は失敗しないはず");
    let json = serde_json::to_string(&record).expect("serialize は成功するはず");
    let padding = usize::try_from(MAX_FREEZE_RECORD_JSON_BYTES - json.len() as u64 + 1).unwrap();
    let oversized = format!("{json}{}", " ".repeat(padding));

    match FreezeRecord::parse(&oversized) {
        Err(LoadFreezeRecordError::TooLarge { size, limit }) => {
            assert_eq!(size, oversized.len() as u64);
            assert_eq!(limit, MAX_FREEZE_RECORD_JSON_BYTES);
        }
        other => panic!("TooLarge を期待したが {other:?} だった"),
    }
}

/// REQ-17・REQ-39: 上限内の凍結記録ファイルは `FreezeRecord::load` で
/// 読み込め、`freeze_eval_data` が返す値と一致する。
#[test]
fn req17_req39_load_round_trips_via_file() {
    let record = freeze_eval_data(SAMPLE).expect("凍結は失敗しないはず");
    let json = serde_json::to_string(&record).expect("serialize は成功するはず");

    let path = std::env::temp_dir().join(format!(
        "fandhe-edge-data-eval-freeze-integration-{}-{}.json",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::write(&path, &json).expect("テスト用ファイルを書き込めるはず");

    let loaded = FreezeRecord::load(&path);
    std::fs::remove_file(&path).ok();

    assert_eq!(loaded.expect("上限内のファイルは成功するはず"), record);
}
