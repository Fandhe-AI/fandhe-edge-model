//! `infer` が入力表現 `tokenizer` の `qwen2_lora` パッケージを、学習側（MLX 内推論）の予測ラベルと
//! 全件一致する形で推論すること（REQ-41・REQ-28・REQ-32・REQ-39・#626。証拠種別: テストハーネス）。
//!
//! 合成の極小ベースモデル（`fixtures/qwen2_parity/`。`trainer/tools/gen_qwen2_parity_fixture.py` が
//! 学習ワーカーの CLI で生成したパッケージそのもの）を一時ディレクトリへ複製して使う。実重みでの測定は
//! TASK-41.11〔#442〕の実機作業で、本テストは実機の証拠にならない。拒否の経路（`../` の
//! `tokenizer_file`・shard の改ざん）はガード層・ランタイムの照合が CLI の入口で効くことを確かめる。
//! 失敗メッセージはケース名・件数のみで入力本文を出さない（security.md）。

#![cfg(any(target_os = "linux", target_os = "macos"))]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// 単体推論（1 件 1 プロセス）で照合する件数。残りは `--input-file`（全件）で照合する。
const SINGLE_CASES: usize = 8;

struct Case {
    name: String,
    /// JSON 文字列リテラルの中身（エスケープ済み。`--input-file` の行へそのまま埋める）。
    raw_input: String,
    label: usize,
}

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/qwen2_parity")
}

/// `cases.json`（整形済み・1 欄 1 行）から必要な欄だけを読む。serde_json を cli の依存へ足さないため、
/// 行単位で読み、`\t` 以外のエスケープが現れたら失敗させる。
fn load_cases() -> Vec<Case> {
    let text = fs::read_to_string(fixture_dir().join("cases.json"))
        .expect("fixture の整形形式が想定と違う（cases.json の読み込み失敗。生成側の責務）");
    let (mut cases, mut name, mut raw) = (Vec::new(), None, None);
    for line in text.lines().map(str::trim) {
        if let Some(v) = line.strip_prefix("\"name\": \"") {
            name = Some(v.trim_end_matches(',').trim_end_matches('"').to_string());
        } else if let Some(v) = line.strip_prefix("\"input\": \"") {
            raw = Some(v.trim_end_matches(',').trim_end_matches('"').to_string());
        } else if let Some(v) = line.strip_prefix("\"mlx_label_index\": ") {
            cases.push(Case {
                name: name.take().expect(
                    "fixture の整形形式が想定と違う（name の後に label が来る前提。生成側の責務）",
                ),
                raw_input: raw.take().expect(
                    "fixture の整形形式が想定と違う（input の後に label が来る前提。生成側の責務）",
                ),
                label: v
                    .trim_end_matches(',')
                    .parse()
                    .expect("fixture の整形形式が想定と違う（label が整数でない。生成側の責務）"),
            });
        }
    }
    for c in &cases {
        assert!(
            !c.raw_input.replace("\\t", "").contains(['\\', '"']),
            "{}",
            c.name
        );
    }
    cases
}

/// 固定の定義（`label_order` と同じ選択肢の宣言順）とパッケージを一時 workspace へ複製する。
fn workspace(tag: &str) -> (PathBuf, Vec<String>) {
    let ws = std::env::temp_dir().join(format!("fandhe-qwen2-parity-{}-{tag}", std::process::id()));
    let _ = fs::remove_dir_all(&ws);
    let pkg = ws.join("p");
    fs::create_dir_all(&pkg).expect("mkdir");
    for f in [
        "artifact.json",
        "model.onnx",
        "tokenizer.json",
        "weights-00000.bin",
    ] {
        fs::copy(fixture_dir().join(f), pkg.join(f)).expect("copy");
    }
    let labels: Vec<String> = ["red", "tan", "sky", "map", "fox", "owl", "gem", "ink"]
        .map(String::from)
        .to_vec();
    let options: Vec<String> = labels
        .iter()
        .map(|l| format!(r#"{{"id":"{l}","display_name":"{l}","description":"d"}}"#))
        .collect();
    fs::write(
        pkg.join("definition.json"),
        format!(
            r#"{{"schema":"fandhe-edge-model-definition/v1","name":"q","version":1,"judgment_type":"single_select","options":[{}],"io":{{"input":"bytes"}}}}"#,
            options.join(",")
        ),
    )
    .expect("definition");
    (ws, labels)
}

fn infer(ws: &Path, args: &[&str]) -> (Option<i32>, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_fandhe-edge"))
        .arg("infer")
        .args(["--package", "p"])
        .args(args)
        .current_dir(ws)
        .stdin(Stdio::null())
        .output()
        .expect("run");
    (
        out.status.code(),
        String::from_utf8(out.stdout).expect("utf8"),
    )
}

fn predicted(line: &str) -> &str {
    let key = "\"predicted_label\":\"";
    let start = line.find(key).map(|i| i + key.len()).unwrap_or(0);
    line.get(start..)
        .and_then(|r| r.split('"').next())
        .unwrap_or("")
}

/// REQ-41・REQ-28: 単体（先頭 `SINGLE_CASES` 件）と `--input-file`（全件）の予測ラベルが学習側と一致し、
/// 同じケースの単体とバッチの出力行が全く同じになる。
#[test]
fn req41_req28_infer_matches_training_side_for_all_cases() {
    let (ws, labels) = workspace("ok");
    let cases = load_cases();
    assert_eq!(cases.len(), 120);

    let jsonl: String = cases
        .iter()
        .enumerate()
        .map(|(i, c)| format!("{{\"id\":\"{i}\",\"input\":\"{}\"}}\n", c.raw_input))
        .collect();
    fs::write(ws.join("in.jsonl"), jsonl).expect("jsonl");
    let (code, stdout) = infer(&ws, &["--input-file", "in.jsonl"]);
    assert_eq!(code, Some(0));
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), cases.len());
    let mismatched: Vec<&str> = cases
        .iter()
        .zip(&lines)
        .filter(|(c, l)| labels.get(c.label).map(String::as_str) != Some(predicted(l)))
        .map(|(c, _)| c.name.as_str())
        .collect();
    assert!(mismatched.is_empty(), "batch labels differ: {mismatched:?}");

    for (i, c) in cases.iter().take(SINGLE_CASES).enumerate() {
        let text = c.raw_input.replace("\\t", "\t");
        let (code, one) = infer(&ws, &["--text", &text]);
        assert_eq!(code, Some(0), "{}", c.name);
        // 単体の出力は `id` だけが違う（既定 id は `input`）。それ以外はバッチ行と同一。
        let batch_line = lines.get(i).copied().unwrap_or_default();
        let norm = |l: &str| l.split_once(",\"status\"").map(|(_, r)| r.to_string());
        assert_eq!(norm(one.trim_end()), norm(batch_line), "{}", c.name);
        assert_eq!(
            labels.get(c.label).map(String::as_str),
            Some(predicted(&one))
        );
    }
    let _ = fs::remove_dir_all(&ws);
}

/// REQ-39: `tokenizer_file` が `../` でパッケージ外を指すパッケージは拒否する。
#[test]
fn req39_qwen2_lora_tokenizer_path_escape_is_rejected() {
    let (ws, _) = workspace("escape");
    let meta = fs::read_to_string(ws.join("p/artifact.json")).expect("meta");
    fs::write(ws.join("outside.json"), "{}").expect("outside");
    let edited = meta.replace("\"tokenizer.json\"", "\"../outside.json\"");
    assert_ne!(meta, edited);
    fs::write(ws.join("p/artifact.json"), edited).expect("meta");
    let (code, stdout) = infer(&ws, &["--text", "x"]);
    assert_eq!(code, Some(64), "{stdout}");
    assert!(
        stdout.starts_with("{\"code\":\"invalid_input\""),
        "{stdout}"
    );
    let _ = fs::remove_dir_all(&ws);
}

/// REQ-39: shard の内容が記録された sha256 と違えば推論せず拒否する（完全性）。
#[test]
fn req39_qwen2_lora_tampered_shard_is_rejected() {
    let (ws, _) = workspace("shard");
    let shard = ws.join("p/weights-00000.bin");
    let mut bytes = fs::read(&shard).expect("shard");
    if let Some(b) = bytes.first_mut() {
        *b ^= 0xff;
    }
    fs::write(&shard, bytes).expect("shard");
    let (code, stdout) = infer(&ws, &["--text", "x"]);
    assert_eq!(code, Some(64), "{stdout}");
    assert_eq!(
        stdout,
        "{\"code\":\"invalid_input\",\"message\":\"model file cannot be loaded\"}\n"
    );
    let _ = fs::remove_dir_all(&ws);
}

/// REQ-39: tokenizer.json の改ざん（記録ハッシュ違い）は拒否する。
#[test]
fn req39_qwen2_lora_tampered_tokenizer_is_rejected() {
    let (ws, _) = workspace("tok");
    let tok = ws.join("p/tokenizer.json");
    let mut bytes = fs::read(&tok).expect("tok");
    bytes.push(b' ');
    fs::write(&tok, bytes).expect("tok");
    let (code, stdout) = infer(&ws, &["--text", "x"]);
    assert_eq!(code, Some(64), "{stdout}");
    assert_eq!(
        stdout,
        "{\"code\":\"invalid_input\",\"message\":\"package tokenizer does not match its recorded hash\"}\n"
    );
    let _ = fs::remove_dir_all(&ws);
}

/// `artifact.json` の文字列を置換して推論し、(終了コード, stdout) を返す。
fn infer_with_meta_edit(tag: &str, from: &str, to: &str) -> (Option<i32>, String) {
    let (ws, _) = workspace(tag);
    let meta = fs::read_to_string(ws.join("p/artifact.json")).expect("meta");
    let edited = meta.replace(from, to);
    assert_ne!(meta, edited, "fixture の整形形式が想定と違う: {from}");
    fs::write(ws.join("p/artifact.json"), edited).expect("meta");
    let r = infer(&ws, &["--text", "x"]);
    let _ = fs::remove_dir_all(&ws);
    r
}

/// REQ-41: kind が `qwen2_lora` でも入力表現が byte（明示・宣言なし）なら byte 前処理で黙って推論せず 64。
#[test]
fn req41_qwen2_lora_with_byte_representation_is_rejected() {
    for (tag, to) in [
        ("byte", "\"input_representation\": \"byte\""),
        ("undeclared", "\"_unused\": 0"),
    ] {
        let (code, stdout) =
            infer_with_meta_edit(tag, "\"input_representation\": \"tokenizer\"", to);
        assert_eq!(code, Some(64), "{tag}: {stdout}");
        assert_eq!(
            stdout, "{\"code\":\"invalid_input\",\"message\":\"model file cannot be loaded\"}\n",
            "{tag}"
        );
    }
}

/// REQ-41・REQ-39: 入力表現が tokenizer でも kind が `qwen2_lora` 以外（c3）は 64（外部データ付き ONNX は
/// c3 の許可リストで形式拒否される）。
#[test]
fn req41_tokenizer_with_c3_kind_is_rejected() {
    let (code, stdout) = infer_with_meta_edit("c3", "\"kind\": \"qwen2_lora\"", "\"kind\": \"c3\"");
    assert_eq!(code, Some(64), "{stdout}");
    assert_eq!(
        stdout,
        "{\"code\":\"invalid_input\",\"message\":\"format rejected: format_not_allowed\"}\n"
    );
}

/// 入力 `text` の prompt 長 + 最長の選択肢のトークン数（採点前の長さ検査の左辺。学習側の
/// `predict_records` の `len(ids) + longest` と同じ定義）。
fn required_len(ws: &Path, text: &str) -> usize {
    let meta = fs::read(ws.join("p/artifact.json")).expect("meta");
    let refs = fandhe_edge_core::artifact_meta::TokenizerPackageRefs::parse(&meta).expect("refs");
    let sha = refs.tokenizer_sha256.parse().expect("sha");
    let spec = fandhe_edge_runtime::tokenizer::TokenizerSpec::load_path(
        &ws.join("p").join(&refs.tokenizer_file),
        &sha,
    )
    .expect("tokenizer");
    let ids = spec
        .build_chat_ids(&refs.system_prompt, text, None)
        .expect("ids");
    ids.len() + refs.longest_choice_tokens
}

fn set_max_seq_length(ws: &Path, n: usize) {
    let path = ws.join("p/artifact.json");
    let meta = fs::read_to_string(&path).expect("meta");
    let key = "\"max_seq_length\": ";
    let start = meta
        .find(key)
        .map(|i| i + key.len())
        .expect("fixture の整形形式が想定と違う: max_seq_length");
    let len = meta[start..]
        .find(',')
        .expect("fixture の整形形式が想定と違う: max_seq_length の後");
    let edited = format!("{}{n}{}", &meta[..start], &meta[start + len..]);
    fs::write(&path, edited).expect("meta");
}

/// REQ-41・REQ-39: prompt + 最長の選択肢が `config.max_seq_length` ちょうどなら通り、1 超えると採点前に
/// `limit_exceeded`（20）。境界は学習側 `predict_records`（`len(ids) + longest > max_seq_length` で
/// `limit_exceeded`・等しければ通す）と同一。単体と `--input-file` の両方で効く。
#[test]
fn req41_max_seq_length_boundary_matches_training_side() {
    let text = "red3 red0 sky3 red1";
    for file_mode in [false, true] {
        let (ws, _) = workspace(if file_mode { "seqf" } else { "seqs" });
        let need = required_len(&ws, text);
        fs::write(
            ws.join("in.jsonl"),
            format!("{{\"id\":\"a\",\"input\":\"{text}\"}}\n"),
        )
        .expect("jsonl");
        let args: Vec<&str> = if file_mode {
            vec!["--input-file", "in.jsonl"]
        } else {
            vec!["--text", text]
        };
        set_max_seq_length(&ws, need);
        let (code, out) = infer(&ws, &args);
        assert_eq!(code, Some(0), "file_mode={file_mode}: {out}");
        set_max_seq_length(&ws, need - 1);
        let (code, out) = infer(&ws, &args);
        assert_eq!(code, Some(20), "file_mode={file_mode}: {out}");
        assert!(out.contains("limit_exceeded"), "{out}");
        let _ = fs::remove_dir_all(&ws);
    }
}
