//! PR #202 レビュー指摘（P1）: `crates/data/src/eval_input.rs` の
//! `SCORE_SUM_TOLERANCE` と `crates/core/src/judgment.rs` の同名定数が
//! 値を重複定義しており、片方だけが変更されると乖離しうる、という指摘への
//! 対応。本 crate は `fandhe-edge-core` に依存しない設計
//! （`crates/data/src/lib.rs`「層の境界」参照）のため、値を 1 箇所の
//! 定数へ集約する代わりに共有 fixture
//! `fixtures/score_tolerance/score_sum_tolerance.json` を単一真実源とし、
//! 本ファイルと `fandhe-edge-core` 側の同名テスト
//! （`crates/core/tests/score_sum_tolerance_fixture.rs`）の双方が同じ
//! fixture を読み込んで自身の定数と照合する（`fixtures/exitcode/exit_codes.json`
//! による終了コードの Rust／学習ワーカー間照合〔#179〕と同じパターン）。
//!
//! 本テストはデータ契約層（data）の範囲に閉じており、他層（core・CLI 等）の
//! コードには依存しない。`docs/spec` も参照しない（本リポの CI は
//! `docs/spec` 抜きで成立させる方針のため）。

use std::fs;
use std::path::PathBuf;

use fandhe_edge_data::eval_input::SCORE_SUM_TOLERANCE;
use serde_json::Value;

/// 外部入力と同じ作法で扱うための読み込み前サイズ上限（REQ-39）。
/// リポ内固定 fixture だが、想定外の巨大化で無制限アロケーションに繋げない。
const MAX_FIXTURE_BYTES: u64 = 1024 * 1024;

/// リポジトリ直下からの相対パスを `Path::join` で組み立て、文字列連結・
/// 区切り文字のハードコードをしない（`.claude/rules/coding-rust.md`）。
/// `CARGO_MANIFEST_DIR` は `crates/data` を指すため、2 階層上がリポ直下。
fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("fixtures")
        .join("score_tolerance")
        .join("score_sum_tolerance.json")
}

fn load_fixture() -> Value {
    let path = fixture_path();
    let metadata = fs::metadata(&path)
        .unwrap_or_else(|e| panic!("failed to stat score tolerance fixture {path:?}: {e}"));
    assert!(
        metadata.len() <= MAX_FIXTURE_BYTES,
        "score tolerance fixture {path:?} exceeds size limit ({} bytes > {MAX_FIXTURE_BYTES})",
        metadata.len()
    );
    let bytes = fs::read(&path)
        .unwrap_or_else(|e| panic!("failed to read score tolerance fixture {path:?}: {e}"));
    serde_json::from_slice(&bytes)
        .unwrap_or_else(|e| panic!("failed to parse score tolerance fixture {path:?}: {e}"))
}

/// PR #202 レビュー指摘・P1: `fandhe_edge_data::eval_input::SCORE_SUM_TOLERANCE`
/// が共有 fixture の値と厳密に一致すること（`crates/core` 側の同名定数との
/// 乖離を、値そのものの直接比較ではなく共有 fixture 経由で検出する）。
#[test]
fn score_sum_tolerance_matches_shared_fixture() {
    let fixture = load_fixture();
    let fixture_value = fixture
        .get("score_sum_tolerance")
        .and_then(Value::as_f64)
        .unwrap_or_else(|| {
            panic!(
                "score tolerance fixture is missing a numeric \"score_sum_tolerance\" field: {fixture:?}"
            )
        });
    assert_eq!(
        SCORE_SUM_TOLERANCE.to_bits(),
        fixture_value.to_bits(),
        "fandhe_edge_data::eval_input::SCORE_SUM_TOLERANCE ({}) diverged from \
         fixtures/score_tolerance/score_sum_tolerance.json ({})",
        SCORE_SUM_TOLERANCE,
        fixture_value
    );
}
