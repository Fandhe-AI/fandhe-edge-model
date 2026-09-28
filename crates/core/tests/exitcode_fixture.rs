//! REQ-21・TASK-21.1（#179）: 終了コード 7 種の Rust ⇔ 学習ワーカー一致を
//! 共有 fixture（`fixtures/exitcode/exit_codes.json`）で照合する。
//!
//! `fandhe-edge-core`（本 crate）の [`fandhe_edge_core::exitcode::ExitCode`]
//! が単一真実源であり、学習ワーカー側ミラー
//! （`trainer/src/fandhe_edge_trainer/exitcode.py`）と数値・名前が食い違わ
//! ないことを、本ファイルと `trainer/tests/test_exitcode.py` の双方が同じ
//! JSON fixture を読み込んで照合する。`fixtures/exitcode/exit_codes.json`
//! というパスが Rust／trainer 間の唯一の結合点であり、このパスを変えると
//! 両テストの結合が切れる。
//!
//! 本テストは共通コア（core）の範囲に閉じており、他層（trainer・CLI 等）
//! のコードには依存しない。`docs/spec` も参照しない（本リポの CI は
//! `docs/spec` 抜きで成立させる方針のため）。

use std::fs;
use std::path::PathBuf;

use fandhe_edge_core::exitcode::ExitCode;
use serde::Deserialize;

/// 外部入力と同じ作法で扱うための読み込み前サイズ上限（REQ-39）。
/// リポ内固定 fixture だが、想定外の巨大化で無制限アロケーションに繋げない。
const MAX_FIXTURE_BYTES: u64 = 1024 * 1024;

/// fixture 内の 1 エントリ。`_meta` 等の未知フィールドが増えても構造体側の
/// フィールド以外は無視してよいが、エントリ自体は名前・数値以外を持たない
/// ことを検証するため `deny_unknown_fields` を付ける。
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    name: String,
    code: i32,
}

/// fixture 全体。`_meta` は説明用で照合対象に含めない。
#[derive(Debug, Deserialize)]
struct Fixture {
    exit_codes: Vec<Entry>,
}

/// リポジトリ直下からの相対パスを `Path::join` で組み立て、文字列連結・
/// 区切り文字のハードコードをしない（`.claude/rules/coding-rust.md`）。
/// `CARGO_MANIFEST_DIR` は `crates/core` を指すため、2 階層上がリポ直下。
fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("fixtures")
        .join("exitcode")
        .join("exit_codes.json")
}

fn load_fixture() -> Fixture {
    let path = fixture_path();
    let metadata = fs::metadata(&path)
        .unwrap_or_else(|e| panic!("failed to stat exit code fixture {path:?}: {e}"));
    assert!(
        metadata.len() <= MAX_FIXTURE_BYTES,
        "exit code fixture {path:?} exceeds size limit ({} bytes > {MAX_FIXTURE_BYTES})",
        metadata.len()
    );
    let bytes = fs::read(&path)
        .unwrap_or_else(|e| panic!("failed to read exit code fixture {path:?}: {e}"));
    serde_json::from_slice(&bytes)
        .unwrap_or_else(|e| panic!("failed to parse exit code fixture {path:?}: {e}"))
}

/// REQ-21: fixture がちょうど 7 件（`ExitCode::ALL` と同数）であること。
#[test]
fn req21_fixture_has_exactly_seven_entries() {
    let fixture = load_fixture();
    assert_eq!(fixture.exit_codes.len(), 7);
    assert_eq!(fixture.exit_codes.len(), ExitCode::ALL.len());
}

/// REQ-21: fixture の各エントリが `ExitCode` の対応 variant と数値・名前の
/// 両方で一致すること（fixture 側の数値・名前ずれを検出する）。
#[test]
fn req21_fixture_entries_match_rust_exit_code() {
    let fixture = load_fixture();
    for entry in &fixture.exit_codes {
        let parsed = ExitCode::try_from(entry.code).unwrap_or_else(|e| {
            panic!(
                "fixture entry {:?} (code={}) does not correspond to a known ExitCode: {e}",
                entry.name, entry.code
            )
        });
        assert_eq!(
            parsed.name(),
            entry.name,
            "fixture entry code={} has name {:?} but ExitCode::name() is {:?}",
            entry.code,
            entry.name,
            parsed.name()
        );
    }
}

/// REQ-21: `ExitCode::ALL` の各 variant が fixture にちょうど 1 件だけ現れ
/// ること（欠落・重複・余剰を検出する双方向照合）。
#[test]
fn req21_every_rust_variant_appears_once_in_fixture() {
    let fixture = load_fixture();
    for exit_code in ExitCode::ALL {
        let matches = fixture
            .exit_codes
            .iter()
            .filter(|entry| {
                entry.name == exit_code.name() && entry.code == i32::from(exit_code.code())
            })
            .count();
        assert_eq!(
            matches,
            1,
            "ExitCode {:?} (code={}) must appear exactly once in the fixture, found {matches}",
            exit_code.name(),
            exit_code.code()
        );
    }
}

/// REQ-21: fixture 内で名前・数値がそれぞれ重複していないこと。
#[test]
fn req21_fixture_has_no_duplicate_names_or_codes() {
    use std::collections::BTreeSet;

    let fixture = load_fixture();
    let names: BTreeSet<&str> = fixture.exit_codes.iter().map(|e| e.name.as_str()).collect();
    let codes: BTreeSet<i32> = fixture.exit_codes.iter().map(|e| e.code).collect();
    assert_eq!(
        names.len(),
        fixture.exit_codes.len(),
        "fixture has duplicate names"
    );
    assert_eq!(
        codes.len(),
        fixture.exit_codes.len(),
        "fixture has duplicate codes"
    );
}
