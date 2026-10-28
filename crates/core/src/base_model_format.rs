//! ベースモデル（追加学習の種類。REQ-41）の入力ファイルの「形の検査」（REQ-39・TASK-41.2-1・#397）。
//!
//! ガード層の形式判定（`fandhe-edge-guard` の `format::detect_format`）から呼ばれる純関数を置く。
//! ガード層は serde を持たない（依存の配置を増やさない。`version_ledger_record` と同じ方針）ため、
//! JSON の解析はこの層で行う。
//!
//! - [`check_safetensors`]: safetensors のヘッダー（先頭 8 バイトの長さ＋JSON）とテンソルのオフセット整合だけを見る。
//!   データ部は解釈しない
//! - [`check_tokenizer_json`]: HF `tokenizer.json` の最小の形（`model.type` を持つ object）だけを見る
//!
//! いずれも意味の検証ではなく形の検査。確認しきれないもの（上限超過・不正構造・未知の dtype）は
//! `Err` を返す（fail-closed）。pickle 系（`.bin`・`.pt`）は呼び出し側の固定シグネチャ判定で拒否する。

use serde_json::Value;

/// safetensors ヘッダー JSON の上限（解析前に確認する。REQ-39）。
const MAX_SAFETENSORS_HEADER_BYTES: u64 = 16 * 1024 * 1024;

/// tokenizer JSON の上限（`Value` 化によるメモリ膨張を防ぐ。Qwen の tokenizer.json は約 7 MB）。
const MAX_TOKENIZER_JSON_BYTES: usize = 64 * 1024 * 1024;

/// バイト単位の dtype のサイズ。サブバイト型・未知の型は `None`。
fn dtype_size(dtype: &str) -> Option<u64> {
    match dtype {
        "BOOL" | "U8" | "I8" | "F8_E5M2" | "F8_E4M3" => Some(1),
        "I16" | "U16" | "F16" | "BF16" => Some(2),
        "I32" | "U32" | "F32" => Some(4),
        "I64" | "U64" | "F64" => Some(8),
        _ => None,
    }
}

/// テンソル 1 件のエントリを検査して `(begin, end)` を返す。
fn tensor_range(entry: &Value) -> Option<(u64, u64)> {
    let obj = entry.as_object()?;
    let size = dtype_size(obj.get("dtype")?.as_str()?)?;
    let mut elems: u64 = 1;
    for d in obj.get("shape")?.as_array()? {
        elems = elems.checked_mul(d.as_u64()?)?;
    }
    let [b, e] = obj.get("data_offsets")?.as_array()?.as_slice() else {
        return None;
    };
    let (b, e) = (b.as_u64()?, e.as_u64()?);
    (e.checked_sub(b)? == elems.checked_mul(size)?).then_some((b, e))
}

/// ベースモデル入力ファイルの形の検査で拒否した理由（REQ-39・REQ-41・TASK-41.2-1）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BaseModelFormatError {
    /// 先頭 8 バイトのヘッダー長が読めない・範囲外・上限超過・全長超過。
    SafetensorsHeaderLength,
    /// ヘッダーが `{` 始まりの JSON object ではない。
    SafetensorsHeaderJson,
    /// `__metadata__` が文字列→文字列の object でも `null` でもない。
    SafetensorsMetadata,
    /// テンソルのエントリが不正（未知 dtype・shape 溢れ・オフセットとサイズの不一致）。
    SafetensorsTensor,
    /// テンソルが 0 件。
    SafetensorsNoTensors,
    /// オフセットに隙間・重なりがある、またはデータ部の長さと一致しない。
    SafetensorsOffsets,
    /// tokenizer JSON が上限を超える。
    TokenizerTooLarge,
    /// tokenizer JSON の形（`model.type` を持つ object）ではない。
    TokenizerShape,
}

/// `bytes`（ファイル全体）が safetensors の形かを検査する。拒否理由を返す。REQ-39・REQ-41・TASK-41.2-1。
pub fn check_safetensors(bytes: &[u8]) -> Result<(), BaseModelFormatError> {
    check_safetensors_prefix(bytes, bytes.len() as u64)
}

/// safetensors のヘッダー長フィールド（先頭 8 バイト）の上限付き読み取り。ストリーム検査の入口
/// （ガード層が本体を保持せずにヘッダーだけ読むため。REQ-39・REQ-41・#672）。
pub fn safetensors_header_end(first8: [u8; 8]) -> Result<usize, BaseModelFormatError> {
    Some(u64::from_le_bytes(first8))
        .filter(|n| (2..=MAX_SAFETENSORS_HEADER_BYTES).contains(n))
        .and_then(|n| n.checked_add(8))
        .and_then(|v| usize::try_from(v).ok())
        .ok_or(BaseModelFormatError::SafetensorsHeaderLength)
}

/// [`check_safetensors`] の本体。`prefix` はファイル先頭から少なくとも `8 + ヘッダー長` バイト、
/// `total_len` はファイル全体の長さ（データ部の長さ検査に使う）。`prefix` がファイル全体なら
/// `check_safetensors` と同一の結果になる。
pub fn check_safetensors_prefix(prefix: &[u8], total_len: u64) -> Result<(), BaseModelFormatError> {
    use BaseModelFormatError as E;
    let bytes = prefix;
    let n = bytes
        .get(..8)
        .and_then(|h| <[u8; 8]>::try_from(h).ok())
        .map(u64::from_le_bytes)
        .filter(|n| (2..=MAX_SAFETENSORS_HEADER_BYTES).contains(n))
        .ok_or(E::SafetensorsHeaderLength)?;
    let data_start = n
        .checked_add(8)
        .and_then(|v| usize::try_from(v).ok())
        .ok_or(E::SafetensorsHeaderLength)?;
    let Some(header) = bytes.get(8..data_start) else {
        return Err(E::SafetensorsHeaderLength);
    };
    let Some(data_len) = total_len.checked_sub(data_start as u64) else {
        return Err(E::SafetensorsHeaderLength);
    };
    if header.first() != Some(&b'{') {
        return Err(E::SafetensorsHeaderJson);
    }
    let Ok(Value::Object(map)) = serde_json::from_slice::<Value>(header) else {
        return Err(E::SafetensorsHeaderJson);
    };
    let mut ranges = Vec::new();
    for (k, v) in &map {
        if k == "__metadata__" {
            // 任意。あれば文字列→文字列の object、または `null`（MLX の `save_safetensors` が metadata
            // なしで書く形。#628）。
            if !v.is_null()
                && !v
                    .as_object()
                    .is_some_and(|m| m.values().all(Value::is_string))
            {
                return Err(E::SafetensorsMetadata);
            }
            continue;
        }
        ranges.push(tensor_range(v).ok_or(E::SafetensorsTensor)?);
    }
    if ranges.is_empty() {
        return Err(E::SafetensorsNoTensors);
    }
    ranges.sort_unstable();
    // 0 から隙間も重なりもなく連続し、データ部の長さちょうどで終わること。
    let mut next = 0u64;
    for (b, e) in ranges {
        if b != next {
            return Err(E::SafetensorsOffsets);
        }
        next = e;
    }
    if data_len == next {
        Ok(())
    } else {
        Err(E::SafetensorsOffsets)
    }
}

/// `bytes`（ファイル全体）が HF `tokenizer.json` の形かを検査する。拒否理由を返す。REQ-39・REQ-41・TASK-41.2-1。
pub fn check_tokenizer_json(bytes: &[u8]) -> Result<(), BaseModelFormatError> {
    use BaseModelFormatError as E;
    if bytes.len() > MAX_TOKENIZER_JSON_BYTES {
        return Err(E::TokenizerTooLarge);
    }
    let Ok(Value::Object(top)) = serde_json::from_slice::<Value>(bytes) else {
        return Err(E::TokenizerShape);
    };
    top.get("model")
        .and_then(Value::as_object)
        .and_then(|m| m.get("type"))
        .is_some_and(Value::is_string)
        .then_some(())
        .ok_or(E::TokenizerShape)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_safetensors(b: &[u8]) -> bool {
        check_safetensors(b).is_ok()
    }
    fn is_tokenizer_json(b: &[u8]) -> bool {
        check_tokenizer_json(b).is_ok()
    }

    /// 試験用: ヘッダー JSON とデータ部から safetensors を組み立てる。
    fn st(header: &str, data: &[u8]) -> Vec<u8> {
        let mut v = (header.len() as u64).to_le_bytes().to_vec();
        v.extend_from_slice(header.as_bytes());
        v.extend_from_slice(data);
        v
    }

    const ONE: &str = r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;

    /// REQ-39・TASK-41.2-1: 正しい safetensors（最小・metadata・空白パディング・複数）を受理する。
    #[test]
    fn req39_safetensors_accepts_valid() {
        assert!(is_safetensors(&st(ONE, &[0; 4])));
        let meta = r#"{"__metadata__":{"format":"pt"},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        assert!(is_safetensors(&st(meta, &[0; 4])));
        // MLX の `mx.save_safetensors`（metadata なし）は `"__metadata__":null` を書く（#628）。
        let null_meta =
            r#"{"__metadata__":null,"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        assert!(is_safetensors(&st(null_meta, &[0; 4])));
        let pad = format!("{ONE}   ");
        assert!(is_safetensors(&st(&pad, &[0; 4])));
        let two = r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"U8","shape":[2,3],"data_offsets":[4,10]}}"#;
        assert!(is_safetensors(&st(two, &[0; 10])));
    }

    /// REQ-39・TASK-41.2-1: 壊れた safetensors は拒否する（fail-closed）。
    #[test]
    fn req39_safetensors_rejects_broken() {
        assert!(!is_safetensors(&[0; 7]));
        // 長さが全長超過
        let mut over = 1000u64.to_le_bytes().to_vec();
        over.extend_from_slice(b"{}");
        assert!(!is_safetensors(&over));
        // 長さが上限超過
        let mut huge = (MAX_SAFETENSORS_HEADER_BYTES + 1).to_le_bytes().to_vec();
        huge.extend_from_slice(b"{}");
        assert!(!is_safetensors(&huge));
        // 先頭が { でない・JSON 不正・配列トップ
        assert!(!is_safetensors(&st(" {}", &[])));
        assert!(!is_safetensors(&st("{not json", &[])));
        assert!(!is_safetensors(&st("[1]", &[])));
        // テンソル 0 件
        assert!(!is_safetensors(&st("{}", &[])));
        // 未知 dtype・サブバイト型
        let bad = r#"{"a":{"dtype":"F4","shape":[1],"data_offsets":[0,1]}}"#;
        assert!(!is_safetensors(&st(bad, &[0])));
        // shape の積のオーバーフロー
        let ovf = r#"{"a":{"dtype":"U8","shape":[4294967296,4294967296],"data_offsets":[0,0]}}"#;
        assert!(!is_safetensors(&st(ovf, &[])));
        // サイズ不一致
        let mism = r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#;
        assert!(!is_safetensors(&st(mism, &[0; 4])));
        // 隙間・重なり
        let gap = r#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"b":{"dtype":"U8","shape":[1],"data_offsets":[2,3]}}"#;
        assert!(!is_safetensors(&st(gap, &[0; 3])));
        let lap = r#"{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"b":{"dtype":"U8","shape":[2],"data_offsets":[1,3]}}"#;
        assert!(!is_safetensors(&st(lap, &[0; 3])));
        // 末尾の余り・不足
        assert!(!is_safetensors(&st(ONE, &[0; 5])));
        assert!(!is_safetensors(&st(ONE, &[0; 3])));
        // metadata が文字列以外
        let m = r#"{"__metadata__":{"k":1},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        assert!(!is_safetensors(&st(m, &[0; 4])));
        for m in ["1", "\"pt\"", "[]"] {
            let h = format!(
                r#"{{"__metadata__":{m},"a":{{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}}}"#
            );
            assert!(!is_safetensors(&st(&h, &[0; 4])), "{m}");
        }
    }

    /// REQ-39・TASK-41.2-1: tokenizer.json の形の検査。
    #[test]
    fn req39_tokenizer_json_shape() {
        assert!(is_tokenizer_json(
            br#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]}}"#
        ));
        assert!(!is_tokenizer_json(b"[]"));
        assert!(!is_tokenizer_json(br#"{"version":"1.0"}"#));
        assert!(!is_tokenizer_json(br#"{"model":[]}"#));
        assert!(!is_tokenizer_json(br#"{"model":{"type":1}}"#));
        assert!(!is_tokenizer_json(br#"{"model":{"type":"BPE"}} x"#));
        assert!(!is_tokenizer_json(br#"{"model":{"type":"BPE""#));
        assert!(!is_tokenizer_json(&vec![
            b' ';
            MAX_TOKENIZER_JSON_BYTES + 1
        ]));
    }

    /// REQ-39・TASK-41.2-1: 拒否理由が区別される。
    #[test]
    fn req39_reasons_distinguished() {
        assert_eq!(
            check_safetensors(&[0; 7]),
            Err(BaseModelFormatError::SafetensorsHeaderLength)
        );
        assert_eq!(
            check_safetensors(&st("{}", &[])),
            Err(BaseModelFormatError::SafetensorsNoTensors)
        );
        let bad = r#"{"a":{"dtype":"F4","shape":[1],"data_offsets":[0,1]}}"#;
        assert_eq!(
            check_safetensors(&st(bad, &[0])),
            Err(BaseModelFormatError::SafetensorsTensor)
        );
        assert_eq!(
            check_safetensors(&st(ONE, &[0; 5])),
            Err(BaseModelFormatError::SafetensorsOffsets)
        );
        assert_eq!(
            check_tokenizer_json(b"[]"),
            Err(BaseModelFormatError::TokenizerShape)
        );
    }
}
