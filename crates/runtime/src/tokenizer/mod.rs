//! Qwen2 系 `tokenizer.json`（HF 形式の byte-level BPE）の読み込みと検証
//! （REQ-41・REQ-39・REQ-32・TASK-41.4-2・#410。親 #408）。
//!
//! # 役割と境界
//!
//! 配布パッケージへ同梱するトークナイザー定義を、Python なしで推論ランタイムが使える形
//! （[`TokenizerSpec`]）に検証つきで読み込む。呼び出し元は後続の前段（Split 状態機械。#411）・
//! BPE と chat template（#412）で、本モジュールは **読み込みと検証だけ** を持つ。
//! 前処理・BPE・golden 照合は未実装（それぞれ #411・#412・#413）。
//!
//! 検証条件の参照実装は `trainer/tools/poc26/qwen2_tokenizer.py` の `Qwen2Tokenizer.__init__`・
//! `_check_pipeline`（同じ条件・同じ順序）。対応範囲外の項目は黙って誤変換せず、対応範囲を示す
//! [`TokenizerLoadError::Unsupported`] で拒否する（fail-closed）。項目は許可リストで照合し、
//! 未知のキーも拒否する（参照実装より厳しい）。
//!
//! # 資源上限（REQ-39）
//!
//! - ファイルは [`fandhe_edge_core::fs::read_bounded`] で [`MAX_TOKENIZER_JSON_BYTES`] まで
//!   読み込む前に確認する。経路の閉じ込め（`safe_join` 相当）は呼び出し側（ガード層）の責務
//!   （`OnnxBackend::load_path` と同じ分担）
//! - JSON の深さ・コンテナ要素数は [`json`] が確保前に判定する。vocab・merges・added_tokens の
//!   件数と文字列長も、変換先のコンテナを確保する前に判定する。一時的な値の木は検証後に捨てる
//!   （入力 16 MiB 由来の定数倍で頭打ち）
//! - `load_path` は sha256 を照合してから解析する。不一致は [`TokenizerLoadError::IntegrityMismatch`]
//!
//! エラーメッセージは英語で、入力中の token 文字列・キー名・値は載せない（security.md）。
//! 依存は追加していない（JSON は [`json`] の自作。#409 のオーナー判断）。

mod json;

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::Path;

use fandhe_edge_core::fs::{FsError, read_bounded};
use fandhe_edge_core::hash::Sha256Digest;

use json::{JsonError, Value};

/// `tokenizer.json` のファイルサイズ上限（参照実装の `MAX_TOKENIZER_JSON_BYTES` と同値。
/// 実物 Qwen2.5 は約 7 MB）。
pub const MAX_TOKENIZER_JSON_BYTES: u64 = 16 * 1024 * 1024;
/// vocab の件数上限（実物は 151,643 件。余裕を持たせた値）。
pub const MAX_VOCAB_ENTRIES: usize = 262_144;
/// merges の件数上限（実物は 151,387 件）。
pub const MAX_MERGES: usize = 262_144;
/// added_tokens の件数上限（実物は 22 件）。
pub const MAX_ADDED_TOKENS: usize = 1024;
/// vocab のキー（byte 表の文字列）の文字数上限。超過は [`TokenizerLoadError::LimitExceeded`]。
pub const MAX_VOCAB_TOKEN_CHARS: usize = 1024;
/// added_tokens の content の文字数上限。超過は [`TokenizerLoadError::LimitExceeded`]。
pub const MAX_ADDED_CONTENT_CHARS: usize = 256;

/// Qwen2 の Split 正規表現（`tokenizer.json` の `pre_tokenizer` との完全一致を要求する定数。
/// `qwen2_tokenizer.py` の `SPLIT_REGEX` と同一）。
pub const SPLIT_REGEX: &str = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

const TOP_KEYS: &[&str] = &[
    "version",
    "truncation",
    "padding",
    "added_tokens",
    "normalizer",
    "pre_tokenizer",
    "post_processor",
    "decoder",
    "model",
];
const BYTE_LEVEL_KEYS: &[&str] = &["type", "add_prefix_space", "trim_offsets", "use_regex"];
const MODEL_KEYS: &[&str] = &[
    "type",
    "dropout",
    "unk_token",
    "continuing_subword_prefix",
    "end_of_word_suffix",
    "fuse_unk",
    "byte_fallback",
    "ignore_merges",
    "vocab",
    "merges",
];
const ADDED_KEYS: &[&str] = &[
    "id",
    "content",
    "single_word",
    "lstrip",
    "rstrip",
    "normalized",
    "special",
];

/// `(左 id, 右 id)` -> `(順位, 併合後の id)`。
pub type MergeTable = HashMap<(u32, u32), (u32, u32)>;

/// 読み込みの失敗。入力中の本文は保持しない（security.md）。
#[derive(Debug)]
#[non_exhaustive]
pub enum TokenizerLoadError {
    /// ファイルを開けない・通常ファイルでない・上限超・読み込み失敗。
    File(FsError),
    /// sha256 が期待値と一致しない（REQ-39）。
    IntegrityMismatch,
    /// JSON として不正（構文・重複キー・孤立サロゲート・NaN・不正 UTF-8 等）。
    MalformedJson,
    /// 件数・長さ・深さの上限超過（REQ-39）。
    LimitExceeded { what: &'static str },
    /// 対応範囲外の種類・設定。`supported` に対応範囲を示す。
    Unsupported {
        section: &'static str,
        supported: &'static str,
    },
    /// 対応範囲の形式だが内容が不正（id の重複・範囲外、byte 文字の欠落、merges の不整合等）。
    Invalid { what: &'static str },
}

impl TokenizerLoadError {
    /// 機械可読なエラーコード（英語）。
    pub fn code(&self) -> &'static str {
        match self {
            Self::File(FsError::TooLarge { .. }) | Self::LimitExceeded { .. } => "limit_exceeded",
            Self::File(_) => "tokenizer_file_error",
            Self::IntegrityMismatch => "integrity_mismatch",
            Self::MalformedJson => "malformed_json",
            Self::Unsupported { .. } => "unsupported_tokenizer",
            Self::Invalid { .. } => "invalid_tokenizer",
        }
    }
}

impl fmt::Display for TokenizerLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::File(e) => write!(f, "tokenizer file error: {e}"),
            Self::IntegrityMismatch => f.write_str("tokenizer rejected: integrity_mismatch"),
            Self::MalformedJson => f.write_str("tokenizer rejected: malformed_json"),
            Self::LimitExceeded { what } => {
                write!(f, "tokenizer rejected: limit exceeded ({what})")
            }
            Self::Unsupported { section, supported } => {
                write!(
                    f,
                    "unsupported tokenizer: {section} (supported: {supported})"
                )
            }
            Self::Invalid { what } => write!(f, "invalid tokenizer: {what}"),
        }
    }
}

impl std::error::Error for TokenizerLoadError {}

fn invalid(what: &'static str) -> TokenizerLoadError {
    TokenizerLoadError::Invalid { what }
}

impl From<JsonError> for TokenizerLoadError {
    fn from(e: JsonError) -> Self {
        match e {
            JsonError::Malformed => Self::MalformedJson,
            JsonError::LimitExceeded => Self::LimitExceeded {
                what: "json nesting or container length",
            },
        }
    }
}

/// chat template が id で挿入する特殊トークンの id（`special: true` で存在することを検証済み）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecialIds {
    pub im_start: u32,
    pub im_end: u32,
    pub endoftext: u32,
}

/// 検証済みのトークナイザー定義。読み込み後は不変（`&self` のみ）。
///
/// 特殊トークン・added_tokens は id と content を保持するだけで、本文の照合表は作らない
/// （入力本文によるプロンプト注入を塞ぐ。参照実装と同じ方針。REQ-39）。
pub struct TokenizerSpec {
    nfc: bool,
    vocab: HashMap<String, u32>,
    merges: MergeTable,
    added: HashMap<u32, String>,
    special: SpecialIds,
    source_sha256: Sha256Digest,
}

impl fmt::Debug for TokenizerSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TokenizerSpec")
            .field("nfc", &self.nfc)
            .field("vocab", &self.vocab.len())
            .field("merges", &self.merges.len())
            .field("added", &self.added.len())
            .field("special", &self.special)
            .finish()
    }
}

impl TokenizerSpec {
    /// ファイルを上限付きで読み、sha256 を照合してから検証する（REQ-39）。
    ///
    /// 経路の閉じ込めは呼び出し側（ガード層）で通した経路を渡す。本関数は通常ファイルの確認と
    /// サイズ上限の検査（`read_bounded`）を行う。
    pub fn load_path(
        path: &Path,
        expected_sha256: &Sha256Digest,
    ) -> Result<Self, TokenizerLoadError> {
        let bytes =
            read_bounded(path, MAX_TOKENIZER_JSON_BYTES).map_err(TokenizerLoadError::File)?;
        if Sha256Digest::of_bytes(&bytes) != *expected_sha256 {
            return Err(TokenizerLoadError::IntegrityMismatch);
        }
        Self::from_bytes(&bytes)
    }

    /// バイト列から検証する（sha256 の照合は呼び出し側。サイズ上限は
    /// [`MAX_TOKENIZER_JSON_BYTES`]）。
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TokenizerLoadError> {
        if u64::try_from(bytes.len()).map_or(true, |n| n > MAX_TOKENIZER_JSON_BYTES) {
            return Err(TokenizerLoadError::LimitExceeded {
                what: "tokenizer json bytes",
            });
        }
        let Value::Object(mut top) = json::parse(bytes)? else {
            return Err(invalid("top-level is not an object"));
        };
        if top.keys().any(|k| !TOP_KEYS.contains(&k.as_str())) {
            return Err(TokenizerLoadError::Unsupported {
                section: "top-level",
                supported: "version, truncation, padding, added_tokens, normalizer, \
                            pre_tokenizer, post_processor, decoder, model",
            });
        }
        check_top_scalars(&top)?;
        let nfc = check_normalizer(top.get("normalizer"))?;
        check_pre_tokenizer(top.get("pre_tokenizer"))?;
        check_byte_level(top.get("decoder"), false, "decoder")?;
        check_byte_level(top.get("post_processor"), true, "post_processor")?;

        let Some(Value::Object(mut model)) = top.remove("model") else {
            return Err(invalid("model is not an object"));
        };
        check_model_settings(&model)?;
        let (Some(Value::Object(vocab_raw)), Some(Value::Array(merges_raw))) =
            (model.remove("vocab"), model.remove("merges"))
        else {
            return Err(invalid("vocab or merges has the wrong type"));
        };
        let Some(Value::Array(added_raw)) = top.remove("added_tokens") else {
            return Err(invalid("added_tokens is not an array"));
        };
        if vocab_raw.len() > MAX_VOCAB_ENTRIES {
            return Err(TokenizerLoadError::LimitExceeded {
                what: "vocab entries",
            });
        }
        if merges_raw.len() > MAX_MERGES {
            return Err(TokenizerLoadError::LimitExceeded { what: "merges" });
        }
        if added_raw.len() > MAX_ADDED_TOKENS {
            return Err(TokenizerLoadError::LimitExceeded {
                what: "added tokens",
            });
        }

        let id_limit = vocab_raw.len() + added_raw.len();
        let byte_chars: HashSet<char> = bytes_to_unicode().into_iter().collect();
        let vocab = build_vocab(vocab_raw, id_limit, &byte_chars)?;
        let (added, special) = build_added(&added_raw, id_limit, &vocab)?;
        let merges = build_merges(&merges_raw, &vocab)?;

        Ok(Self {
            nfc,
            vocab,
            merges,
            added,
            special,
            source_sha256: Sha256Digest::of_bytes(bytes),
        })
    }

    /// normalizer が NFC のとき true（null・キー無しなら false）。
    pub fn nfc(&self) -> bool {
        self.nfc
    }

    /// byte 表の文字列 -> token id。
    pub fn vocab(&self) -> &HashMap<String, u32> {
        &self.vocab
    }

    /// `(左 id, 右 id)` -> `(順位, 併合後の id)`。順位は小さいほど優先。
    pub fn merges(&self) -> &MergeTable {
        &self.merges
    }

    /// added_tokens の id -> content。
    pub fn added_tokens(&self) -> &HashMap<u32, String> {
        &self.added
    }

    /// chat template が挿入する特殊トークンの id。
    pub fn special_ids(&self) -> SpecialIds {
        self.special
    }

    /// 解析したバイト列の sha256（golden 照合で実ファイルとの一致確認に使う。#413）。
    pub fn source_sha256(&self) -> &Sha256Digest {
        &self.source_sha256
    }
}

/// GPT-2 の byte -> 可視 Unicode 文字の対応表（ByteLevel 前処理と同一。添字がバイト値）。
/// 後続の BPE（#412）も使う。
pub fn bytes_to_unicode() -> [char; 256] {
    let mut out = ['\0'; 256];
    let mut extra = 0u32;
    for (b, slot) in (0u8..=255).zip(out.iter_mut()) {
        *slot = if matches!(b, 33..=126 | 161..=172 | 174..=255) {
            char::from(b)
        } else {
            let c = char::from_u32(256 + extra).unwrap_or(char::REPLACEMENT_CHARACTER);
            extra += 1;
            c
        };
    }
    out
}

fn unsupported_pipeline(section: &'static str, supported: &'static str) -> TokenizerLoadError {
    TokenizerLoadError::Unsupported { section, supported }
}

/// 許可キーだけを持つオブジェクトならその中身を返す。
fn obj_with<'a>(v: Option<&'a Value>, allowed: &[&str]) -> Option<&'a HashMap<String, Value>> {
    match v {
        Some(Value::Object(m)) if m.keys().all(|k| allowed.contains(&k.as_str())) => Some(m),
        _ => None,
    }
}

fn is_str(v: Option<&Value>, want: &str) -> bool {
    matches!(v, Some(Value::Str(s)) if s == want)
}

fn is_bool(v: Option<&Value>, want: bool) -> bool {
    matches!(v, Some(Value::Bool(b)) if *b == want)
}

fn is_null_or_absent(v: Option<&Value>) -> bool {
    matches!(v, None | Some(Value::Null))
}

fn check_top_scalars(top: &HashMap<String, Value>) -> Result<(), TokenizerLoadError> {
    if top.get("version").is_some_and(|v| !is_str(Some(v), "1.0")) {
        return Err(unsupported_pipeline("version", "1.0"));
    }
    for key in ["truncation", "padding"] {
        if !is_null_or_absent(top.get(key)) {
            return Err(unsupported_pipeline(
                if key == "truncation" {
                    "truncation"
                } else {
                    "padding"
                },
                "null",
            ));
        }
    }
    Ok(())
}

fn check_normalizer(v: Option<&Value>) -> Result<bool, TokenizerLoadError> {
    if is_null_or_absent(v) {
        return Ok(false);
    }
    match obj_with(v, &["type"]) {
        Some(m) if is_str(m.get("type"), "NFC") => Ok(true),
        _ => Err(unsupported_pipeline("normalizer", "NFC or null")),
    }
}

fn check_pre_tokenizer(v: Option<&Value>) -> Result<(), TokenizerLoadError> {
    const SUPPORTED: &str = "Sequence of Split (Qwen2 regex, Isolated, no invert) and ByteLevel \
                             (no prefix space, no regex)";
    let err = || unsupported_pipeline("pre_tokenizer", SUPPORTED);
    let seq = obj_with(v, &["type", "pretokenizers"]).ok_or_else(err)?;
    let Some(Value::Array(parts)) = seq.get("pretokenizers") else {
        return Err(err());
    };
    if !is_str(seq.get("type"), "Sequence") || parts.len() != 2 {
        return Err(err());
    }
    let (split, byte_level) = (parts.first(), parts.get(1));
    let split = obj_with(split, &["type", "pattern", "behavior", "invert"]).ok_or_else(err)?;
    let pattern = obj_with(split.get("pattern"), &["Regex"]).ok_or_else(err)?;
    if !(is_str(split.get("type"), "Split")
        && is_str(pattern.get("Regex"), SPLIT_REGEX)
        && is_str(split.get("behavior"), "Isolated")
        && is_bool(split.get("invert"), false))
    {
        return Err(err());
    }
    let bl = obj_with(byte_level, BYTE_LEVEL_KEYS).ok_or_else(err)?;
    if !(is_str(bl.get("type"), "ByteLevel")
        && is_bool(bl.get("add_prefix_space"), false)
        && is_bool(bl.get("use_regex"), false)
        && matches!(bl.get("trim_offsets"), None | Some(Value::Bool(_))))
    {
        return Err(err());
    }
    Ok(())
}

/// decoder（必須）・post_processor（null・キー無しも可）の ByteLevel を検証する。
fn check_byte_level(
    v: Option<&Value>,
    nullable: bool,
    section: &'static str,
) -> Result<(), TokenizerLoadError> {
    if nullable && is_null_or_absent(v) {
        return Ok(());
    }
    let ok = obj_with(v, BYTE_LEVEL_KEYS).is_some_and(|m| {
        is_str(m.get("type"), "ByteLevel")
            && ["add_prefix_space", "trim_offsets", "use_regex"]
                .iter()
                .all(|k| matches!(m.get(*k), None | Some(Value::Bool(_))))
    });
    if ok {
        Ok(())
    } else {
        Err(unsupported_pipeline(
            section,
            if nullable {
                "ByteLevel or null"
            } else {
                "ByteLevel"
            },
        ))
    }
}

fn check_model_settings(model: &HashMap<String, Value>) -> Result<(), TokenizerLoadError> {
    const SUPPORTED: &str = "BPE without dropout, unk_token, subword prefix/suffix, fuse_unk, \
                             byte_fallback or ignore_merges";
    let err = || unsupported_pipeline("model", SUPPORTED);
    if model.keys().any(|k| !MODEL_KEYS.contains(&k.as_str())) || !is_str(model.get("type"), "BPE")
    {
        return Err(err());
    }
    // 欠落は HF の既定値と同じ意味なので受理する。
    let ok = is_null_or_absent(model.get("dropout"))
        && is_null_or_absent(model.get("unk_token"))
        && ["continuing_subword_prefix", "end_of_word_suffix"]
            .iter()
            .all(|k| model.get(*k).is_none() || is_str(model.get(*k), ""))
        && ["fuse_unk", "byte_fallback", "ignore_merges"]
            .iter()
            .all(|k| matches!(model.get(*k), None | Some(Value::Bool(false))));
    if ok { Ok(()) } else { Err(err()) }
}

fn as_id(v: &Value, id_limit: usize) -> Option<u32> {
    match v {
        Value::Int(i) => usize::try_from(*i)
            .ok()
            .filter(|&n| n < id_limit)
            .and_then(|n| u32::try_from(n).ok()),
        _ => None,
    }
}

fn build_vocab(
    raw: HashMap<String, Value>,
    id_limit: usize,
    byte_chars: &HashSet<char>,
) -> Result<HashMap<String, u32>, TokenizerLoadError> {
    let mut vocab = HashMap::with_capacity(raw.len());
    let mut seen_ids = HashSet::with_capacity(raw.len());
    for (token, v) in raw {
        if token.chars().count() > MAX_VOCAB_TOKEN_CHARS {
            return Err(TokenizerLoadError::LimitExceeded {
                what: "vocab token length",
            });
        }
        let id =
            as_id(&v, id_limit).ok_or_else(|| invalid("vocab id is not an in-range integer"))?;
        if !seen_ids.insert(id) {
            return Err(invalid("duplicate vocab id"));
        }
        if !token.chars().all(|c| byte_chars.contains(&c)) {
            return Err(invalid("vocab token has a non byte-level character"));
        }
        vocab.insert(token, id);
    }
    let mut buf = [0u8; 4];
    if byte_chars
        .iter()
        .any(|c| !vocab.contains_key(&*c.encode_utf8(&mut buf)))
    {
        return Err(invalid("a byte-level character is missing from vocab"));
    }
    Ok(vocab)
}

fn build_added(
    raw: &[Value],
    id_limit: usize,
    vocab: &HashMap<String, u32>,
) -> Result<(HashMap<u32, String>, SpecialIds), TokenizerLoadError> {
    let vocab_ids: HashSet<u32> = vocab.values().copied().collect();
    let mut added: HashMap<u32, String> = HashMap::with_capacity(raw.len());
    let mut contents: HashSet<&str> = HashSet::with_capacity(raw.len());
    let mut special: HashMap<&str, u32> = HashMap::new();
    for entry in raw {
        let m = obj_with(Some(entry), ADDED_KEYS)
            .ok_or_else(|| invalid("added token entry is not a known object"))?;
        let id = m
            .get("id")
            .and_then(|v| as_id(v, id_limit))
            .ok_or_else(|| invalid("added token id is not an in-range integer"))?;
        let Some(Value::Str(content)) = m.get("content") else {
            return Err(invalid("added token content is not a string"));
        };
        if ["single_word", "lstrip", "rstrip", "normalized", "special"]
            .iter()
            .any(|k| !matches!(m.get(*k), None | Some(Value::Bool(_))))
        {
            return Err(invalid("added token flag is not a boolean"));
        }
        if content.chars().count() > MAX_ADDED_CONTENT_CHARS {
            return Err(TokenizerLoadError::LimitExceeded {
                what: "added token content length",
            });
        }
        if vocab_ids.contains(&id) || added.contains_key(&id) {
            return Err(invalid("duplicate token id"));
        }
        if !contents.insert(content.as_str()) {
            return Err(invalid("duplicate added token content"));
        }
        added.insert(id, content.clone());
        if is_bool(m.get("special"), true) {
            special.insert(content.as_str(), id);
        }
    }
    let find = |name: &str| {
        special
            .get(name)
            .copied()
            .ok_or_else(|| invalid("required special token is missing"))
    };
    let ids = SpecialIds {
        im_start: find("<|im_start|>")?,
        im_end: find("<|im_end|>")?,
        endoftext: find("<|endoftext|>")?,
    };
    Ok((added, ids))
}

fn build_merges(
    raw: &[Value],
    vocab: &HashMap<String, u32>,
) -> Result<MergeTable, TokenizerLoadError> {
    let mut merges = HashMap::with_capacity(raw.len());
    for (rank, m) in raw.iter().enumerate() {
        let (l, r) = match m {
            Value::Str(s) => {
                let mut it = s.split(' ');
                match (it.next(), it.next(), it.next()) {
                    (Some(l), Some(r), None) => (l, r),
                    _ => return Err(invalid("merge is not a pair")),
                }
            }
            Value::Array(a) => match a.as_slice() {
                [Value::Str(l), Value::Str(r)] => (l.as_str(), r.as_str()),
                _ => return Err(invalid("merge is not a pair")),
            },
            _ => return Err(invalid("merge is not a pair")),
        };
        let (Some(&li), Some(&ri)) = (vocab.get(l), vocab.get(r)) else {
            return Err(invalid("merge side is not in vocab"));
        };
        let Some(&merged) = vocab.get(&format!("{l}{r}")) else {
            return Err(invalid("merge result is not in vocab"));
        };
        let rank = u32::try_from(rank).map_err(|_| invalid("merge rank overflow"))?;
        if merges.insert((li, ri), (rank, merged)).is_some() {
            return Err(invalid("duplicate merge pair"));
        }
    }
    Ok(merges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_to_unicode_matches_gpt2_table() {
        let t = bytes_to_unicode();
        assert_eq!(t[usize::from(b' ')], '\u{120}'); // Ġ
        assert_eq!(t[usize::from(b'\n')], '\u{10a}'); // Ċ
        assert_eq!(t[usize::from(b'a')], 'a');
        assert_eq!(t[0], '\u{100}');
        assert_eq!(t[255], 'ÿ');
        assert_eq!(t.iter().collect::<HashSet<_>>().len(), 256);
    }

    #[test]
    fn split_regex_matches_reference_constant() {
        assert!(SPLIT_REGEX.starts_with("(?i:'s|'t|'re|'ve|'m|'ll|'d)|"));
        assert!(SPLIT_REGEX.ends_with("|\\s+(?!\\S)|\\s+"));
    }
}
