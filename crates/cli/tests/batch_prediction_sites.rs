//! CLI がバッチ予測 API を参考測定に使っていないことの機械照合（REQ-28・TASK-28.3・#493）。
//!
//! # 固定する規則
//!
//! - runtime のバッチ予測 API（`infer_batch`・`infer_batch_until`・`infer_batch_partial_until`）の
//!   呼び出しは、判定経路 `crates/cli/src/infer_batch.rs`（`infer --input-file`）だけに置く。
//!   判定経路は単体とバッチの全件一致を保証済み（TASK-28.1）で、記録への注記は不要
//! - 参考測定の入口 `infer_batch_for_reference` の呼び出しは 0 件。CLI の出力・記録には参考測定が
//!   無いため、`batch_prediction` を明記すべき記録も存在しない（#493 の受け入れ条件
//!   「記録に明記される」を「該当する記録が存在しないことの機械照合」に読み替え。オーナー承認 2026-10-10）
//! - 今後 CLI が参考測定にバッチ API を使うときは `InferencePipeline::infer_batch_for_reference` を通し、
//!   その記録 JSON に `"batch_prediction":true` と `"batch_prediction_notice":"<BATCH_PREDICTION_NOTICE>"`
//!   （キー・注記は `fandhe_edge_runtime::prediction_provenance` の定数が SSOT）を出す。その変更で本テストが
//!   落ちるので、規則に沿って期待値を更新する
//!
//! # 走査の方法
//!
//! `crates/cli/src` 配下の `.rs` を読み、コメント（`//`・入れ子の `/* */`）と文字列・文字リテラル
//! （生文字列・バイト文字列を含む）を空白に置き換えてから識別子を拾う。対象名の識別子が完全一致で、
//! 直後（空白を挟んでよい）が `(` または turbofish の `::<` のときだけ呼び出しと数える。
//! そのため doc コメントや文字列中の言及、`emit_infer_batch(` のような別名、
//! `crate::infer_batch::...` のモジュールパス、`fn infer_batch(` の定義は数えない。
//! 走査器自体の誤検出・見逃しは本ファイルのユニットテストで固定する。証拠種別はテストハーネス。

use std::path::{Path, PathBuf};

/// runtime のバッチ予測 API（判定経路と共通の入口）。
const BATCH_APIS: [&str; 3] = [
    "infer_batch",
    "infer_batch_until",
    "infer_batch_partial_until",
];

/// 参考測定の入口（呼び出し 0 件を固定する）。
const REFERENCE_API: &str = "infer_batch_for_reference";

/// バッチ予測 API の呼び出しを許す唯一のファイル（`src` からの相対パス）。
const JUDGMENT_PATH_FILE: &str = "infer_batch.rs";

/// 呼び出し 1 件（`src` からの相対パス・1 始まりの行・API 名）。
#[derive(Debug, Clone, PartialEq, Eq)]
struct CallSite {
    file: PathBuf,
    line: usize,
    name: String,
}

/// コメントと文字列・文字リテラルを空白へ置き換える（改行は残し、行番号を保つ）。
fn strip_comments_and_literals(src: &str) -> String {
    let c: Vec<char> = src.chars().collect();
    let mut out = String::with_capacity(src.len());
    let blank = |out: &mut String, ch: char| out.push(if ch == '\n' { '\n' } else { ' ' });
    let is_ident = |ch: char| ch.is_alphanumeric() || ch == '_';
    let at = |i: usize| c.get(i).copied();
    let mut i = 0;
    while let Some(ch) = at(i) {
        let next = at(i + 1);
        let prev_ident = i > 0 && at(i - 1).is_some_and(is_ident);
        if ch == '/' && next == Some('/') {
            while let Some(x) = at(i) {
                if x == '\n' {
                    break;
                }
                blank(&mut out, x);
                i += 1;
            }
        } else if ch == '/' && next == Some('*') {
            let mut depth = 0usize;
            while let Some(x) = at(i) {
                if x == '/' && at(i + 1) == Some('*') {
                    depth += 1;
                    out.push_str("  ");
                    i += 2;
                } else if x == '*' && at(i + 1) == Some('/') {
                    depth -= 1;
                    out.push_str("  ");
                    i += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    blank(&mut out, x);
                    i += 1;
                }
            }
        } else if ch == 'r'
            && matches!(next, Some('"' | '#'))
            && (!prev_ident
                || (i >= 2
                    && matches!(at(i - 1), Some('b' | 'c'))
                    && !at(i - 2).is_some_and(is_ident)))
        {
            // 生文字列 `r"…"`・`r#"…"#`（`br`・`cr` も）。`r#ident` の生識別子は `"` が続かないので対象外。
            let mut j = i + 1;
            while at(j) == Some('#') {
                j += 1;
            }
            if at(j) != Some('"') {
                out.push(ch);
                i += 1;
                continue;
            }
            let hashes = j - i - 1;
            let mut k = j + 1;
            while let Some(x) = at(k) {
                if x == '"' && (1..=hashes).all(|h| at(k + h) == Some('#')) {
                    k += 1 + hashes;
                    break;
                }
                k += 1;
            }
            for x in c.get(i..k.min(c.len())).unwrap_or_default() {
                blank(&mut out, *x);
            }
            i = k;
        } else if ch == '"' {
            blank(&mut out, ch);
            i += 1;
            while let Some(x) = at(i) {
                blank(&mut out, x);
                i += 1;
                if x == '\\' {
                    if let Some(y) = at(i) {
                        blank(&mut out, y);
                        i += 1;
                    }
                } else if x == '"' {
                    break;
                }
            }
        } else if ch == '\'' && (next == Some('\\') || at(i + 2) == Some('\'')) {
            // 文字リテラル（`'"'` 等）。それ以外の `'` はライフタイムとしてそのまま残す。
            blank(&mut out, ch);
            i += 1;
            if at(i) == Some('\\') {
                blank(&mut out, '\\');
                i += 1;
                if let Some(y) = at(i) {
                    blank(&mut out, y);
                    i += 1;
                }
            }
            while let Some(x) = at(i) {
                blank(&mut out, x);
                i += 1;
                if x == '\'' {
                    break;
                }
            }
        } else {
            out.push(ch);
            i += 1;
        }
    }
    out
}

/// コメント・リテラル除去後のコードから、`names` の識別子の呼び出しを拾う。
fn find_calls(file: &Path, src: &str, names: &[&str]) -> Vec<CallSite> {
    let code = strip_comments_and_literals(src);
    let c: Vec<char> = code.chars().collect();
    let is_ident = |ch: char| ch.is_alphanumeric() || ch == '_';
    let mut sites = Vec::new();
    let mut line = 1;
    let mut i = 0;
    while let Some(&ch) = c.get(i) {
        if ch == '\n' {
            line += 1;
        }
        if !is_ident(ch) {
            i += 1;
            continue;
        }
        let start = i;
        while c.get(i).copied().is_some_and(is_ident) {
            i += 1;
        }
        let ident: String = c.get(start..i).unwrap_or_default().iter().collect();
        if !names.contains(&ident.as_str()) {
            continue;
        }
        let rest: String = c.get(i..).unwrap_or_default().iter().collect();
        let after = rest.trim_start();
        let is_call = after.starts_with('(') || after.starts_with("::<");
        let before: String = c.get(..start).unwrap_or_default().iter().collect();
        let prev_token = before
            .trim_end()
            .rsplit(|ch: char| !is_ident(ch))
            .next()
            .unwrap_or("");
        if is_call && prev_token != "fn" {
            sites.push(CallSite {
                file: file.to_path_buf(),
                line,
                name: ident,
            });
        }
    }
    sites
}

/// `crates/cli/src` 配下の `.rs` を再帰的に集める（相対パス順で決定的に返す）。
fn cli_sources() -> Vec<(PathBuf, String)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut stack = vec![root.clone()];
    let mut files = Vec::new();
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).expect("read_dir") {
            let path = entry.expect("dir entry").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                let text = std::fs::read_to_string(&path).expect("read source");
                let rel = path.strip_prefix(&root).expect("under src").to_path_buf();
                files.push((rel, text));
            }
        }
    }
    files.sort();
    files
}

fn calls_in_cli(names: &[&str]) -> Vec<CallSite> {
    cli_sources()
        .iter()
        .flat_map(|(rel, text)| find_calls(rel, text, names))
        .collect()
}

/// REQ-28・TASK-28.3: バッチ予測 API の呼び出しは判定経路 `infer_batch.rs` だけにあり、1 件以上ある
/// （0 件なら走査が空振りしている）。
#[test]
fn req28_task28_3_batch_api_calls_only_in_judgment_path() {
    let sites = calls_in_cli(&BATCH_APIS);
    let outside: Vec<&CallSite> = sites
        .iter()
        .filter(|s| s.file != Path::new(JUDGMENT_PATH_FILE))
        .collect();
    assert!(
        outside.is_empty(),
        "batch prediction API called outside the judgment path; reference measurements must use \
         {REFERENCE_API} and record batch_prediction (REQ-28): {outside:?}"
    );
    let names: Vec<&str> = sites.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["infer_batch_partial_until"]);
}

/// REQ-28・TASK-28.3: 参考測定の入口の呼び出しは 0 件（`batch_prediction` を明記すべき記録が無い）。
#[test]
fn req28_task28_3_no_reference_batch_predictions_in_cli() {
    assert_eq!(calls_in_cli(&[REFERENCE_API]), Vec::<CallSite>::new());
}

/// 走査器: コメント・文字列・別名・モジュールパス・定義を数えず、呼び出しだけを拾う。
#[test]
fn scanner_counts_calls_only() {
    let src = r##"
// pipeline.infer_batch(&x)
/// [`InferencePipeline::infer_batch`] infer_batch_for_reference(x)
/* outer /* pipeline.infer_batch(x) */ still comment infer_batch(x) */
use crate::infer_batch::{emit_infer_batch, judgment_from_prediction};
fn infer_batch(x: &str) {}
let s = "pipeline.infer_batch(&x) \" infer_batch(x)";
let r = r#"infer_batch_for_reference(x) "quoted""#;
let q = '"'; let e = '\''; fn f<'a>(v: &'a str) {}
emit_infer_batch(out, reader);
let a = pipeline.infer_batch (&x);
let b = InferencePipeline::infer_batch_until(&p, &x, d);
let c = p.infer_batch_for_reference::<()>(&x);
"##;
    let names = [
        "infer_batch",
        "infer_batch_until",
        "infer_batch_for_reference",
    ];
    let got: Vec<(usize, String)> = find_calls(Path::new("x.rs"), src, &names)
        .into_iter()
        .map(|s| (s.line, s.name))
        .collect();
    assert_eq!(
        got,
        [
            (11, "infer_batch".to_string()),
            (12, "infer_batch_until".to_string()),
            (13, "infer_batch_for_reference".to_string()),
        ]
    );
}
