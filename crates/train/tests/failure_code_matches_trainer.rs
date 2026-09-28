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
//!
//! 本ファイルはさらに（issue #178 実装計画 3.6・step 3）、`WorkerError(code,
//! ..., ExitCode.<NAME>)` の第 3 引数（対応する終了コード）を抜き出し、
//! [`fandhe_edge_train::result::FailureCode::exit_code`] が学習ワーカー側の
//! 実際の対応と一致すること・1 つの `code` が複数の `ExitCode` に対応して
//! いないことを照合する（[`req178_failure_code_exit_code_matches_trainer_worker_error_calls`]）。

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use fandhe_edge_core::exitcode::ExitCode;
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

/// `WorkerError(code, ..., ExitCode.NAME)` 呼び出しから `(code, ExitCode 名)`
/// の組を抽出する（issue #178 実装計画 3.6・step 3）。`errors.py::WorkerError`
/// の第 1 引数はコード文字列リテラル、第 3 引数（呼び出しによっては複数行の
/// `message` を挟む）が対応する [`ExitCode`] 定数（`ExitCode.INVALID_INPUT`
/// 等）であるという `WorkerError.__init__` のシグネチャ（issue #178
/// 実装計画 2 章の現状調査）に基づく。1 回の呼び出し内で `ExitCode.` が
/// 複数回現れることはない（`message` に `ExitCode` という語を含む呼び出しは
/// 現状のソースに無い）ため、コード文字列リテラルの直後から次の
/// `WorkerError(`／`WorkerError` の呼び出し境界までの間で最初に現れる
/// `ExitCode.<NAME>` を対応する終了コードとみなす。
fn extract_code_exit_pairs(source: &str) -> BTreeMap<String, BTreeSet<String>> {
    let marker = "WorkerError(";
    let mut pairs: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut search_start = 0usize;
    while let Some(rel) = source.get(search_start..).and_then(|s| s.find(marker)) {
        let call_start = search_start + rel + marker.len();
        // 次回の検索開始位置は必ず前進させる（`code` 抽出の成否に関わらず
        // 無限ループを避ける）。
        search_start = call_start;
        let Some(tail) = source.get(call_start..) else {
            break;
        };
        let after_ws = tail.trim_start();
        let Some(quoted) = after_ws.strip_prefix('"') else {
            continue;
        };
        let Some(code_end) = quoted.find('"') else {
            continue;
        };
        let code = quoted[..code_end].to_string();
        let after_code = &quoted[code_end + 1..];
        // この呼び出しの範囲（次の `WorkerError(` 出現、または末尾まで）に
        // 限って `ExitCode.` を探す。
        let scope_end = after_code.find(marker).unwrap_or(after_code.len());
        let scope = &after_code[..scope_end];
        let Some(exit_rel) = scope.find("ExitCode.") else {
            continue;
        };
        let exit_tail = &scope[exit_rel + "ExitCode.".len()..];
        let name_end = exit_tail
            .find(|c: char| !(c.is_ascii_uppercase() || c == '_'))
            .unwrap_or(exit_tail.len());
        if name_end == 0 {
            continue;
        }
        let exit_name = exit_tail[..name_end].to_string();
        pairs.entry(code).or_default().insert(exit_name);
    }
    pairs
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

/// `trainer/src/fandhe_edge_trainer/` 配下全体から `(code, ExitCode 名)` の
/// 組を集める（[`extract_code_exit_pairs`] 参照）。
fn collect_trainer_code_exit_pairs() -> BTreeMap<String, BTreeSet<String>> {
    let dir = trainer_src_dir();
    let mut files = Vec::new();
    collect_py_files(&dir, &mut files);
    let mut pairs: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for path in files {
        let source = read_source_with_size_limit(&path);
        for (code, exit_names) in extract_code_exit_pairs(&source) {
            pairs.entry(code).or_default().extend(exit_names);
        }
    }
    pairs
}

/// Rust 側 [`ExitCode`] の `snake_case` 名（`ExitCode::name()`）を、Python
/// 側の `ExitCode.<NAME>`（`SCREAMING_SNAKE_CASE`）へ変換する。
fn python_exit_code_name(code: ExitCode) -> String {
    code.name().to_uppercase()
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

/// REQ-21・REQ-39・issue #178 実装計画 3.6: `FailureCode::exit_code()` が、
/// 学習ワーカー（`WorkerError(code, ..., ExitCode.<NAME>)`）の実際の対応と
/// 一致することを機械照合する。1 つの `code` が学習ワーカー側で複数の
/// 異なる `ExitCode` へ対応づけられていないこと（1 対 1 の対応）も確認する。
#[test]
fn req178_failure_code_exit_code_matches_trainer_worker_error_calls() {
    let pairs = collect_trainer_code_exit_pairs();
    assert!(
        !pairs.is_empty(),
        "expected to find at least one WorkerError(code, ..., ExitCode.NAME) call"
    );
    for code in FailureCode::all() {
        let Some(exit_names) = pairs.get(code.as_str()) else {
            panic!(
                "no WorkerError(...) call with ExitCode found for code {:?}",
                code.as_str()
            );
        };
        assert_eq!(
            exit_names.len(),
            1,
            "code {:?} maps to more than one ExitCode in the trainer: {exit_names:?}",
            code.as_str()
        );
        let trainer_exit_name = exit_names
            .iter()
            .next()
            .expect("checked non-empty above")
            .clone();
        let rust_exit_name = python_exit_code_name(code.exit_code());
        assert_eq!(
            trainer_exit_name, rust_exit_name,
            "FailureCode::{code:?}.exit_code() ({rust_exit_name}) does not match \
             the trainer's WorkerError(..., ExitCode.{trainer_exit_name})"
        );
    }
}
