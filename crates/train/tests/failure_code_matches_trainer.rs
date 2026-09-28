//! REQ-21・REQ-39（PR #220 レビュー指摘 P1）: 学習ワーカー（Python）が実際に
//! 出す失敗 `code` の集合と、Rust 側の許可リスト
//! [`fandhe_edge_train::result::FailureCode`] が一致することを機械照合する。
//!
//! `trainer/src/fandhe_edge_trainer/` 配下（`kinds/` サブディレクトリを含め
//! 再帰的に走査。`kind` ごとの失敗コード〔`unsupported_kind`・`invalid_config`
//! 等〕は `kinds/__init__.py`・`kinds/c1.py`・`kinds/c3.py` にあるため、直下
//! だけを見ると取りこぼす）から、失敗コードを出す 2 つの経路をどちらも
//! 走査する:
//!
//! 1. `errors.py::WorkerError(code, ...)` 呼び出し（`raise`／`return` の両方）
//!    の第 1 引数
//! 2. `supervisor.py`／`cli.py` が `WorkerError` を経由せず直接組み立てる
//!    `{"status": "error", "code": "...", ...}`（`_emit` へ渡す辞書リテラル）
//!    の `"code":` の次の文字列リテラル
//!
//! いずれも文字列リテラルでない場合（例: `class WorkerError(Exception):` の
//! クラス定義、`"code": e.code` のように変数を渡す箇所）は該当なしとして
//! スキップする。値そのものは既知の固定コード（学習データ本文ではない）の
//! ため、本テストで読み込み・比較してよい。
//!
//! 依存追加なし（正規表現クレートを使わず、マーカー文字列の直後に続く最初
//! の文字列リテラルだけを抜き出す手書きスキャナで足りる。`.claude/rules/
//! dependency-policy.md`）。

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use fandhe_edge_train::result::FailureCode;

/// 外部入力と同じ作法で扱うための読み込み前サイズ上限（REQ-39）。
/// リポ内固定ファイルだが、想定外の巨大化で無制限アロケーションに繋げない。
const MAX_SOURCE_BYTES: u64 = 1024 * 1024;

/// `marker` の直後に続く最初の文字列リテラル（`"..."`）を全件抜き出す。
/// `marker` の直後が文字列リテラルでない出現（例: `class WorkerError
/// (Exception):` のクラス定義、`"code": e.code` の変数参照）は該当なしと
/// してスキップする。
fn extract_string_literals_after(source: &str, marker: &str) -> BTreeSet<String> {
    let mut codes = BTreeSet::new();
    let mut search_start = 0usize;
    while let Some(rel) = source.get(search_start..).and_then(|s| s.find(marker)) {
        let call_start = search_start + rel + marker.len();
        search_start = call_start;
        let Some(tail) = source.get(call_start..) else {
            break;
        };
        let after_ws = tail.trim_start();
        let Some(quoted) = after_ws.strip_prefix('"') else {
            continue;
        };
        let Some(end) = quoted.find('"') else {
            continue;
        };
        codes.insert(quoted[..end].to_string());
    }
    codes
}

/// モジュール doc の 2 経路（`WorkerError(...)` の第 1 引数・`"code": "..."`
/// の直接組み立て）の両方から失敗コードを集める。
fn extract_worker_error_codes(source: &str) -> BTreeSet<String> {
    let mut codes = extract_string_literals_after(source, "WorkerError(");
    codes.extend(extract_string_literals_after(source, "\"code\":"));
    codes
}

/// リポジトリ直下からの相対パスを `Path::join` で組み立てる
/// （`.claude/rules/coding-rust.md`。文字列連結・区切り文字のハードコード
/// をしない）。`CARGO_MANIFEST_DIR` は `crates/train` を指すため、2 階層上が
/// リポ直下。
fn trainer_src_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("trainer")
        .join("src")
        .join("fandhe_edge_trainer")
}

fn read_source_with_size_limit(path: &Path) -> String {
    let metadata = fs::metadata(path).unwrap_or_else(|e| panic!("failed to stat {path:?}: {e}"));
    assert!(
        metadata.len() <= MAX_SOURCE_BYTES,
        "{path:?} exceeds size limit ({} bytes > {MAX_SOURCE_BYTES})",
        metadata.len()
    );
    fs::read_to_string(path).unwrap_or_else(|e| panic!("failed to read {path:?}: {e}"))
}

/// `dir` 配下の `*.py` を再帰的に集める。`__pycache__`（コンパイル済み
/// キャッシュ。ソースではない）は除外する。
fn collect_py_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(dir).unwrap_or_else(|e| panic!("failed to read {dir:?}: {e}"));
    for entry in entries {
        let entry = entry.unwrap_or_else(|e| panic!("failed to read dir entry in {dir:?}: {e}"));
        let path = entry.path();
        let file_type = entry
            .file_type()
            .unwrap_or_else(|e| panic!("failed to stat dir entry {path:?}: {e}"));
        if file_type.is_dir() {
            if path.file_name().and_then(|n| n.to_str()) == Some("__pycache__") {
                continue;
            }
            collect_py_files(&path, out);
            continue;
        }
        if path.extension().and_then(|ext| ext.to_str()) == Some("py") {
            out.push(path);
        }
    }
}

/// `trainer/src/fandhe_edge_trainer/` 配下（`kinds/` を含め再帰的に走査した）
/// 全 `*.py` から `WorkerError(...)` の第 1 引数（文字列リテラル）を集める。
fn collect_trainer_failure_codes() -> BTreeSet<String> {
    let dir = trainer_src_dir();
    let mut files = Vec::new();
    collect_py_files(&dir, &mut files);
    let mut codes = BTreeSet::new();
    for path in files {
        let source = read_source_with_size_limit(&path);
        codes.extend(extract_worker_error_codes(&source));
    }
    codes
}

/// REQ-21・REQ-39: 学習ワーカーが実際に出す `code` の集合が、すべて
/// [`FailureCode::all`] の許可リストに含まれる（Rust 側の許可リストが
/// Python 側より狭くなっていないか）。
#[test]
fn req39_every_trainer_failure_code_is_in_rust_allowlist() {
    let trainer_codes = collect_trainer_failure_codes();
    assert!(
        !trainer_codes.is_empty(),
        "expected to find at least one WorkerError(...) call in trainer sources"
    );
    let rust_codes: BTreeSet<&str> = FailureCode::all().iter().map(|c| c.as_str()).collect();
    for code in &trainer_codes {
        assert!(
            rust_codes.contains(code.as_str()),
            "trainer emits code {code:?} which is not in FailureCode::all()"
        );
    }
}

/// REQ-21・REQ-39: [`FailureCode::all`] の各コードが、学習ワーカー側で実際に
/// 使われている（Rust 側の許可リストが Python 側より広くなっていないか。
/// 使われなくなったコードを許可リストに残さない）。
#[test]
fn req39_every_rust_allowlist_code_is_emitted_by_trainer() {
    let trainer_codes = collect_trainer_failure_codes();
    for code in FailureCode::all() {
        assert!(
            trainer_codes.contains(code.as_str()),
            "FailureCode::{code:?} ({:?}) is not emitted by any trainer source file",
            code.as_str()
        );
    }
}
