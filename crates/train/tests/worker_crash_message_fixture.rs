//! 共有 fixture `fixtures/train_contract/worker_crash_message.json` と
//! `job_record` の worker クラッシュ文言の分類の一致照合（REQ-34・TASK-34.2・#146）。
//!
//! Python 側（`trainer/tests/test_supervisor.py`）も同じ fixture から接頭辞を照合する。
//! どちらか一方の文言が変わると、この照合が失敗する。

use fandhe_edge_train::job_record::{WORKER_SIGNAL_MESSAGE_PREFIX, parse_worker_signal_message};

fn fixture() -> serde_json::Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/train_contract/worker_crash_message.json");
    let text = std::fs::read_to_string(path).expect("read fixture");
    serde_json::from_str(&text).expect("parse fixture")
}

/// REQ-34: 接頭辞が fixture と一致する。
#[test]
fn req34_prefix_matches_shared_fixture() {
    assert_eq!(
        fixture()["prefix"].as_str(),
        Some(WORKER_SIGNAL_MESSAGE_PREFIX)
    );
    assert_eq!(WORKER_SIGNAL_MESSAGE_PREFIX, "worker terminated by signal ");
}

/// REQ-34: fixture のクラッシュ文言はシグナル番号つきで分類され、他は分類されない。
#[test]
fn req34_shared_fixture_messages_are_classified() {
    let fx = fixture();
    let crash = fx["crash_messages"].as_array().expect("crash_messages");
    assert_eq!(crash.len(), 2);
    for case in crash {
        let message = case["message"].as_str().expect("message");
        let signal = i32::try_from(case["signal"].as_i64().expect("signal")).expect("i32");
        assert_eq!(parse_worker_signal_message(message), Some(signal));
    }
    let non_crash = fx["non_crash_messages"].as_array().expect("non_crash");
    assert_eq!(non_crash.len(), 7);
    for message in non_crash {
        let message = message.as_str().expect("message");
        assert_eq!(parse_worker_signal_message(message), None, "{message}");
    }
}
