//! `fixtures/qwen2_parity/cases.json`（学習側 MLX 内推論の予測ラベル）の読み取り。`infer_qwen2_parity.rs` と
//! `pipeline_e2e.rs` が `#[path]` で共有する（REQ-41・REQ-28・#626・#662。証拠種別: テストハーネス）。

use std::fs;
use std::path::Path;

/// `cases.json` の 1 ケース。
pub struct Case {
    pub name: String,
    /// JSON 文字列リテラルの中身（エスケープ済み。`--input-file` の行へそのまま埋める）。
    pub raw_input: String,
    /// 学習側（MLX 内推論）の予測ラベルの添字（`label_order` の位置）。
    pub label: usize,
}

/// `cases.json`（整形済み・1 欄 1 行）から必要な欄だけを読む。serde_json を cli の依存へ足さないため、
/// 行単位で読み、`\t` 以外のエスケープが現れたら失敗させる。
pub fn load_cases() -> Vec<Case> {
    let text = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/qwen2_parity/cases.json"),
    )
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
