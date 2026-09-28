//! 来歴レコード型（REQ-40・TASK-40.1-1・issue #74）。
//!
//! 学習・評価データを外部 LLM で生成した場合に、モデル名・指示文のハッシュ・
//! 生成日時・token 数を後から追跡できるようにするための型を定義する。
//! 本モジュール（`provenance.rs`）は型と検証のみを提供する。取り込み時に
//! 来歴を記録する機構（来歴 JSON の検証・記録 JSON 生成・データ検査
//! コマンドとの接続）は [`ingest`] サブモジュールで実装済み（TASK-40.1-2・
//! issue #75）。
//!
//! # スコープ境界
//!
//! - 指示文本文から sha256 を計算する処理は持たない。[`PromptHash`] は
//!   既に計算済みのハッシュ値（16 進文字列または生バイト列）を受け取る
//!   （`sha2` の本 crate への配置または `fandhe-edge-core` のハッシュ計算の
//!   再利用はユーザー承認事項のため、issue #75 では見送った。[`ingest`]
//!   モジュール doc の「範囲外」を参照）
//! - `serde` の `Serialize`/`Deserialize` は派生しない（`crates/data/Cargo.toml`
//!   に `serde` を追加していない。[`ingest::provenance_to_json`] は
//!   `serde_json::Value`／`Map` を手動で組み立てて JSON 化する）
//! - 生成元（`source`）フィールド・外部 LLM 出力の既定拒否は TASK-40.2 の範囲
//!   （[`ProvenanceRecord`] は `#[non_exhaustive]` のため非破壊で追加できる）
//! - データ本文・指示文本文そのものは保持しない（[`PromptHash`] のみ保持する）
//!
//! # 呼び出し文脈
//!
//! [`ingest::parse_provenance_json`]（→ [`crate::ingest::ingest_records`]）
//! から呼ばれ、将来は CLI のデータ検査コマンド（パス未確定・TASK-33.x）
//! から接続される想定。本 crate の他モジュール（[`crate::inspect`] 等）と
//! 同様、ファイル読み込み・サイズ上限検査（REQ-39）はガード層
//! （呼び出し側）の責務とし、本モジュールはガード層を通過済みの値
//! （文字列・バイト列）を受け取るところから始まる。
//!
//! # 出典
//!
//! PoC-10／PoC-11／PoC-25 の生成ログ（`meta.json` の `model_requested`・
//! `prompt_sha256`・`started_utc`・`usage_observed`）の形状を踏襲した
//! （テストの具体値は PoC-11 の生成ログからリテラルとして移植。
//! `docs/spec` は参照しない。spec-reference のビルド独立方針）。
//! ハッシュ型 [`PromptHash`] は `crates/core/src/canonical.rs` の
//! `DefinitionHash` の形（`[u8; 32]`・`to_hex`／`Display`／`Debug` の書き方）
//! を模倣しているが、本 crate は `fandhe-edge-core` に依存しない
//! （`crates/data/src/lib.rs` の「層の境界」節を参照）。

use std::fmt;

pub mod ingest;

/// モデル名の許容バイト長の上限。
const MODEL_NAME_MAX_BYTES: usize = 256;

/// 生成日時の入力文字列の許容バイト長の上限（解析前に検査する）。
const GENERATED_AT_MAX_INPUT_BYTES: usize = 64;

/// 来歴レコードの構築・検証時のエラー。
///
/// `Display` は固定の英語文言のみを返し、入力値（モデル名・日時文字列等）を
/// 含めない（security.md「秘密情報の混入防止」。`crates/core` の
/// `CanonicalError` と同じ方針）。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ProvenanceError {
    /// モデル名が空文字列だった。
    EmptyModelName,
    /// モデル名が [`MODEL_NAME_MAX_BYTES`] バイトを超えていた。
    ModelNameTooLong,
    /// モデル名に制御文字（`char::is_control`）が含まれていた。
    ModelNameHasControlChar,
    /// モデル名の前後に空白が含まれていた（値を黙って書き換えない方針の
    /// ため、トリムせず拒否する）。
    ModelNameHasSurroundingWhitespace,
    /// 指示文ハッシュの 16 進表現の長さが 64 桁でなかった。
    PromptHashInvalidLength,
    /// 指示文ハッシュの 16 進表現に小文字 16 進以外の文字が含まれていた。
    PromptHashInvalidHex,
    /// 生成日時が RFC 3339（UTC 表記）として不正だった
    /// （形式・範囲・長さ超過のいずれか。入力値は含めない）。
    InvalidGeneratedAt,
}

impl fmt::Display for ProvenanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            ProvenanceError::EmptyModelName => "model name must not be empty",
            ProvenanceError::ModelNameTooLong => "model name exceeds the maximum byte length",
            ProvenanceError::ModelNameHasControlChar => {
                "model name must not contain control characters"
            }
            ProvenanceError::ModelNameHasSurroundingWhitespace => {
                "model name must not have surrounding whitespace"
            }
            ProvenanceError::PromptHashInvalidLength => {
                "prompt hash must be exactly 64 hex characters"
            }
            ProvenanceError::PromptHashInvalidHex => "prompt hash must be lowercase hexadecimal",
            ProvenanceError::InvalidGeneratedAt => {
                "generated_at must be a valid UTC RFC 3339 timestamp"
            }
        };
        f.write_str(message)
    }
}

impl std::error::Error for ProvenanceError {}

/// 生成時に指定したモデル名（PoC の `model_requested` に相当）。
///
/// PoC-10 では存在しないモデル名を指定すると HTTP 400 で失敗し、別モデルへ
/// 黙って切り替わらないことを確認済みのため、ここでは「実際に応答した
/// モデル」（PoC の `model_observed`。多くの場合 `null`）ではなく
/// **指定値**を保持する。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModelName(String);

impl ModelName {
    /// モデル名を検証して構築する。
    ///
    /// 空文字列・[`MODEL_NAME_MAX_BYTES`] バイト超過・制御文字・前後の空白を
    /// 拒否する（値を黙って書き換えない。トリムしない）。
    pub fn new(value: &str) -> Result<Self, ProvenanceError> {
        if value.is_empty() {
            return Err(ProvenanceError::EmptyModelName);
        }
        if value.len() > MODEL_NAME_MAX_BYTES {
            return Err(ProvenanceError::ModelNameTooLong);
        }
        if value.chars().any(char::is_control) {
            return Err(ProvenanceError::ModelNameHasControlChar);
        }
        let trimmed = value.trim();
        if trimmed.len() != value.len() {
            return Err(ProvenanceError::ModelNameHasSurroundingWhitespace);
        }
        Ok(ModelName(value.to_string()))
    }

    /// モデル名の文字列表現。
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// 指示文（プロンプト）本文の sha256 ハッシュ（32 バイト）。
///
/// 指示文本文そのものは保持しない。本 crate では指示文本文からハッシュを
/// 計算する処理を持たない（呼び出し側が計算済みの値を渡す。上記
/// スコープ境界を参照）。
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct PromptHash([u8; 32]);

impl PromptHash {
    /// 小文字 16 進 64 桁の文字列から構築する（PoC の `prompt_sha256` と
    /// 同じ表記）。長さ不一致・大文字混在・非 16 進文字を拒否する。
    pub fn from_hex(hex: &str) -> Result<Self, ProvenanceError> {
        if hex.len() != 64 {
            return Err(ProvenanceError::PromptHashInvalidLength);
        }
        let ascii = hex.as_bytes();
        let mut bytes = [0u8; 32];
        for (index, byte) in bytes.iter_mut().enumerate() {
            // 2 桁ずつ読み取る。`get` で境界外アクセスを避け、
            // 小文字 16 進以外の文字（大文字・非 16 進を含む）は fail-closed で拒否する。
            let high = ascii
                .get(index * 2)
                .copied()
                .and_then(lowercase_hex_digit_value)
                .ok_or(ProvenanceError::PromptHashInvalidHex)?;
            let low = ascii
                .get(index * 2 + 1)
                .copied()
                .and_then(lowercase_hex_digit_value)
                .ok_or(ProvenanceError::PromptHashInvalidHex)?;
            *byte = (high << 4) | low;
        }
        Ok(PromptHash(bytes))
    }

    /// 生の 32 バイトから構築する（issue #75 がハッシュを計算して渡す経路用。
    /// この経路は形式検証を必要としないため `Result` を返さない）。
    #[must_use]
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        PromptHash(bytes)
    }

    /// 生の 32 バイトを返す。
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// 小文字 16 進 64 桁の文字列表現。
    #[must_use]
    pub fn to_hex(&self) -> String {
        use fmt::Write as _;
        let mut out = String::with_capacity(self.0.len() * 2);
        for byte in self.0 {
            let _ = write!(out, "{byte:02x}");
        }
        out
    }
}

impl fmt::Display for PromptHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl fmt::Debug for PromptHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PromptHash({})", self.to_hex())
    }
}

/// ASCII 1 バイトを小文字 16 進の値（0〜15）に変換する
/// （`PromptHash::from_hex` が大文字・非 16 進文字を拒否するために使う。
/// `char::to_digit(16)` は大文字も受理してしまうため使わない）。
fn lowercase_hex_digit_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

/// RFC 3339（UTC 表記）の生成日時。
///
/// 受理する形式は `YYYY-MM-DDTHH:MM:SS[.f{1,9}](Z|+00:00)` のみ。UTC 以外の
/// オフセットは拒否し、タイムゾーン変換は呼び出し側の責務とする。
/// フィールドの並び順（年→月→日→時→分→秒→ナノ秒）で `PartialOrd`／`Ord`
/// を派生しており、比較順がそのまま時系列順になる（文字列比較に依存しない。
/// 小数秒の桁数が異なっても正しく比較できる）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GeneratedAt {
    year: u16,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: u8,
    nanos: u32,
}

impl GeneratedAt {
    /// RFC 3339（UTC 表記）の文字列を解析する。
    ///
    /// `Z` と `+00:00` は同じ値として正規化する。UTC 以外のオフセット・
    /// オフセット無し・`T` 以外の区切り・小数秒 0 桁または 10 桁以上・
    /// うるう秒（60）・範囲外の月日時分秒・非 ASCII 数字を拒否する。
    /// 入力の長さ（[`GENERATED_AT_MAX_INPUT_BYTES`] バイト）を解析前に
    /// 検査する。
    pub fn parse_rfc3339_utc(input: &str) -> Result<Self, ProvenanceError> {
        if input.len() > GENERATED_AT_MAX_INPUT_BYTES {
            return Err(ProvenanceError::InvalidGeneratedAt);
        }
        if !input.is_ascii() {
            return Err(ProvenanceError::InvalidGeneratedAt);
        }
        let bytes = input.as_bytes();

        // 固定長プレフィックス "YYYY-MM-DDTHH:MM:SS"（19 バイト）を要求する。
        if bytes.len() < 19 {
            return Err(ProvenanceError::InvalidGeneratedAt);
        }
        let year = parse_fixed_digits(bytes, 0, 4)?;
        expect_byte(bytes, 4, b'-')?;
        let month = parse_fixed_digits(bytes, 5, 2)?;
        expect_byte(bytes, 7, b'-')?;
        let day = parse_fixed_digits(bytes, 8, 2)?;
        expect_byte(bytes, 10, b'T')?;
        let hour = parse_fixed_digits(bytes, 11, 2)?;
        expect_byte(bytes, 13, b':')?;
        let minute = parse_fixed_digits(bytes, 14, 2)?;
        expect_byte(bytes, 16, b':')?;
        let second = parse_fixed_digits(bytes, 17, 2)?;

        let rest = bytes.get(19..).ok_or(ProvenanceError::InvalidGeneratedAt)?;
        let (nanos, offset_part) = parse_fraction_and_offset(rest)?;
        validate_utc_offset(offset_part)?;

        let month_u8: u8 = month
            .try_into()
            .map_err(|_| ProvenanceError::InvalidGeneratedAt)?;
        let day_u8: u8 = day
            .try_into()
            .map_err(|_| ProvenanceError::InvalidGeneratedAt)?;
        let hour_u8: u8 = hour
            .try_into()
            .map_err(|_| ProvenanceError::InvalidGeneratedAt)?;
        let minute_u8: u8 = minute
            .try_into()
            .map_err(|_| ProvenanceError::InvalidGeneratedAt)?;
        let second_u8: u8 = second
            .try_into()
            .map_err(|_| ProvenanceError::InvalidGeneratedAt)?;
        let year_u16: u16 = year
            .try_into()
            .map_err(|_| ProvenanceError::InvalidGeneratedAt)?;

        if !(1..=12).contains(&month_u8) {
            return Err(ProvenanceError::InvalidGeneratedAt);
        }
        let days_in_month = days_in_month(year_u16, month_u8);
        if day_u8 < 1 || day_u8 > days_in_month {
            return Err(ProvenanceError::InvalidGeneratedAt);
        }
        if hour_u8 > 23 {
            return Err(ProvenanceError::InvalidGeneratedAt);
        }
        if minute_u8 > 59 {
            return Err(ProvenanceError::InvalidGeneratedAt);
        }
        // うるう秒（60）は拒否する（本モジュール doc に明記済みの方針）。
        if second_u8 > 59 {
            return Err(ProvenanceError::InvalidGeneratedAt);
        }

        Ok(GeneratedAt {
            year: year_u16,
            month: month_u8,
            day: day_u8,
            hour: hour_u8,
            minute: minute_u8,
            second: second_u8,
            nanos,
        })
    }

    /// 正準形の RFC 3339（UTC）文字列を返す。`nanos == 0` なら
    /// `YYYY-MM-DDTHH:MM:SSZ`、それ以外は小数 9 桁固定
    /// `YYYY-MM-DDTHH:MM:SS.nnnnnnnnnZ`（決定的・固定幅）。
    #[must_use]
    pub fn to_rfc3339_utc(&self) -> String {
        use fmt::Write as _;
        let mut out = String::with_capacity(30);
        let _ = write!(
            out,
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
            self.year, self.month, self.day, self.hour, self.minute, self.second
        );
        if self.nanos != 0 {
            let _ = write!(out, ".{:09}", self.nanos);
        }
        out.push('Z');
        out
    }
}

impl fmt::Display for GeneratedAt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_rfc3339_utc())
    }
}

/// `bytes[offset..offset+len]` を 10 進整数として解析する（添字アクセス・
/// `unwrap` を使わず `get` と checked 演算で行う）。
fn parse_fixed_digits(bytes: &[u8], offset: usize, len: usize) -> Result<u32, ProvenanceError> {
    let slice = bytes
        .get(offset..offset + len)
        .ok_or(ProvenanceError::InvalidGeneratedAt)?;
    let mut value: u32 = 0;
    for &b in slice {
        if !b.is_ascii_digit() {
            return Err(ProvenanceError::InvalidGeneratedAt);
        }
        let digit = u32::from(b - b'0');
        value = value
            .checked_mul(10)
            .and_then(|v| v.checked_add(digit))
            .ok_or(ProvenanceError::InvalidGeneratedAt)?;
    }
    Ok(value)
}

/// `bytes[offset]` が期待するバイトと一致するか確認する。
fn expect_byte(bytes: &[u8], offset: usize, expected: u8) -> Result<(), ProvenanceError> {
    match bytes.get(offset) {
        Some(&b) if b == expected => Ok(()),
        _ => Err(ProvenanceError::InvalidGeneratedAt),
    }
}

/// 秒の後続部分（小数秒とオフセット）を解析する。小数点が無ければ
/// `nanos = 0` とし、残り全体をオフセット部分として返す。
fn parse_fraction_and_offset(rest: &[u8]) -> Result<(u32, &[u8]), ProvenanceError> {
    if let Some(&first) = rest.first()
        && first == b'.'
    {
        let digits_start = 1;
        let mut end = digits_start;
        while rest.get(end).is_some_and(u8::is_ascii_digit) {
            end += 1;
        }
        let digit_count = end - digits_start;
        if !(1..=9).contains(&digit_count) {
            return Err(ProvenanceError::InvalidGeneratedAt);
        }
        let digits = rest
            .get(digits_start..end)
            .ok_or(ProvenanceError::InvalidGeneratedAt)?;
        let mut value: u32 = 0;
        for &b in digits {
            let digit = u32::from(b - b'0');
            value = value
                .checked_mul(10)
                .and_then(|v| v.checked_add(digit))
                .ok_or(ProvenanceError::InvalidGeneratedAt)?;
        }
        // ナノ秒（9 桁）に正規化するため、桁数に応じて 10 のべき乗を掛ける。
        let scale = 10u32.pow((9 - digit_count) as u32);
        let nanos = value
            .checked_mul(scale)
            .ok_or(ProvenanceError::InvalidGeneratedAt)?;
        let offset = rest.get(end..).ok_or(ProvenanceError::InvalidGeneratedAt)?;
        return Ok((nanos, offset));
    }
    Ok((0, rest))
}

/// オフセット部分が `Z` または `+00:00` のいずれかであることを確認する
/// （UTC 以外のオフセット・オフセット無しは拒否する）。
fn validate_utc_offset(offset: &[u8]) -> Result<(), ProvenanceError> {
    if offset == b"Z" || offset == b"+00:00" {
        Ok(())
    } else {
        Err(ProvenanceError::InvalidGeneratedAt)
    }
}

/// うるう年判定（4 で割り切れ、100 で割り切れないか 400 で割り切れる年）。
fn is_leap_year(year: u16) -> bool {
    (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400)
}

/// 指定した年月の日数（うるう年の 2 月を考慮する）。呼び出し側
/// （[`GeneratedAt::parse_rfc3339_utc`]）で `month` が 1〜12 の範囲で
/// あることを確認してから呼ばれる前提のため、`match` の `None` 腕
/// （`month` が範囲外の場合のフォールバック `31`）は到達しない。
/// 呼び出し順が変わった場合に誤った日数（黙って `30` や `0` 等）を
/// 返さないよう、`unwrap` の代わりに明示的な分岐として残している。
fn days_in_month(year: u16, month: u8) -> u8 {
    const DAYS: [u8; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    if month == 2 && is_leap_year(year) {
        return 29;
    }
    match DAYS.get(usize::from(month.saturating_sub(1))) {
        Some(&d) => d,
        None => 31,
    }
}

/// 観測できた入力・出力 token 数。
///
/// cached／reasoning の内訳（PoC の `cached_input_tokens`・
/// `reasoning_output_tokens`）は保持しない。必要になれば
/// [`TokenUsage`] へフィールドを追加する（本型は `#[non_exhaustive]` を
/// 持たないため、追加時は破壊的変更として扱う）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TokenUsage {
    input_tokens: u64,
    output_tokens: u64,
}

impl TokenUsage {
    /// 入力・出力 token 数から構築する。
    #[must_use]
    pub fn new(input_tokens: u64, output_tokens: u64) -> Self {
        TokenUsage {
            input_tokens,
            output_tokens,
        }
    }

    /// 入力 token 数。
    #[must_use]
    pub fn input_tokens(&self) -> u64 {
        self.input_tokens
    }

    /// 出力 token 数。
    #[must_use]
    pub fn output_tokens(&self) -> u64 {
        self.output_tokens
    }

    /// 入力・出力の合計。オーバーフロー時は `None`（panic しない。
    /// `checked_add` を使う）。
    #[must_use]
    pub fn total(&self) -> Option<u64> {
        self.input_tokens.checked_add(self.output_tokens)
    }
}

/// token 数の観測状態。
///
/// 生成ツールが呼び出しに失敗した場合など、token 数が報告されないことが
/// ある（PoC のログでは `usage_observed: null`）。これを `0` で埋めると
/// 「0 token だった」と「観測できなかった」を区別できなくなるため、
/// 評価契約の方針（分母 0 を 0 で埋めない）・状態を enum で表す規約
/// （coding-rust.md）に合わせて明示的な variant を持つ。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TokenCount {
    /// token 数が観測できた。
    Observed(TokenUsage),
    /// 生成ツールが token 数を報告しなかった（0 件扱いにしない）。
    Unobserved,
}

/// 外部 LLM で生成したデータ 1 件分の来歴レコード（REQ-40・TASK-40.1-1）。
///
/// データ本文・指示文本文そのものは保持しない（[`PromptHash`] のみ保持）。
/// 生成元（`source`）・データ種別（学習／評価）は TASK-40.2 で追加予定
/// （`#[non_exhaustive]` のため非破壊で拡張できる）。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ProvenanceRecord {
    model_name: ModelName,
    prompt_hash: PromptHash,
    generated_at: GeneratedAt,
    token_count: TokenCount,
}

impl ProvenanceRecord {
    /// 検証済みの各項目から来歴レコードを構築する。各項目は構築時点で
    /// 検証済みのためここでは `Result` を返さない。
    #[must_use]
    pub fn new(
        model_name: ModelName,
        prompt_hash: PromptHash,
        generated_at: GeneratedAt,
        token_count: TokenCount,
    ) -> Self {
        ProvenanceRecord {
            model_name,
            prompt_hash,
            generated_at,
            token_count,
        }
    }

    /// 生成時に指定したモデル名。
    #[must_use]
    pub fn model_name(&self) -> &ModelName {
        &self.model_name
    }

    /// 指示文のハッシュ。
    #[must_use]
    pub fn prompt_hash(&self) -> &PromptHash {
        &self.prompt_hash
    }

    /// 生成日時（UTC）。
    #[must_use]
    pub fn generated_at(&self) -> &GeneratedAt {
        &self.generated_at
    }

    /// token 数の観測状態。
    #[must_use]
    pub fn token_count(&self) -> &TokenCount {
        &self.token_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn req40_leap_year_and_month_length_table() {
        assert!(is_leap_year(2024));
        assert!(is_leap_year(2000));
        assert!(!is_leap_year(1900));
        assert!(!is_leap_year(2026));
        assert_eq!(days_in_month(2024, 2), 29);
        assert_eq!(days_in_month(2026, 2), 28);
        assert_eq!(days_in_month(2026, 1), 31);
        assert_eq!(days_in_month(2026, 4), 30);
    }

    #[test]
    fn req40_generated_at_canonical_output_with_fraction() {
        let ts = GeneratedAt::parse_rfc3339_utc("2026-09-24T00:14:33.672492+00:00")
            .expect("有効な RFC 3339 のはず");
        assert_eq!(ts.to_rfc3339_utc(), "2026-09-24T00:14:33.672492000Z");
    }

    #[test]
    fn req40_prompt_hash_rejects_uppercase() {
        let uppercase = "A".repeat(64);
        assert_eq!(
            PromptHash::from_hex(&uppercase),
            Err(ProvenanceError::PromptHashInvalidHex)
        );
    }

    #[test]
    fn req40_model_name_error_message_does_not_leak_input() {
        let marker = "X".repeat(300);
        let err = ModelName::new(&marker).expect_err("上限超過で拒否されるはず");
        assert_eq!(err, ProvenanceError::ModelNameTooLong);
        assert!(!err.to_string().contains('X'));
    }
}
