//! 生バイト列の sha256 ダイジェスト型（REQ-17・REQ-27）。
//!
//! [`crate::canonical::DefinitionHash`] は `Definition` を正準化 JSON へ
//! 変換してからハッシュする専用型だが、こちらは既に読み込んだ・受け取った
//! バイト列をそのままハッシュする汎用型で、評価データの凍結（REQ-17。
//! TASK-17.2-1・issue #47）やモデルパッケージの評価前後比較（REQ-27。
//! TASK-27.1-1・issue #69）など、「正準化を経ない生バイト列の同一性」を
//! 扱う上位層から使われる。
//!
//! 2 つを別の型にしているのは [`crate::canonical`] の doc と同じ理由で、
//! 「正準化済みハッシュ」と「生バイト列ハッシュ」を取り違えて比較する誤用を
//! コンパイルエラーで防ぐため（`PartialEq` は同じ型同士でしか比較できない）。
//!
//! # 呼び出し文脈
//!
//! データ契約層（`fandhe-edge-data`）の評価データ凍結（REQ-17・TASK-17.2-1）が
//! [`Sha256Digest::of_bytes`] を使って評価データ本体の sha256 を記録する
//! （`crates/data/src/eval_freeze.rs` の `FreezeRecord`）。評価器（REQ-27・
//! TASK-27.1-1）は [`crate::fs::sha256_file_bounded`] がストリームで計算した
//! 結果を [`Sha256Digest::from_array`] 経由で本型へ変換する。
//!
//! # 外部入力としての扱い（fail-closed。REQ-39）
//!
//! [`Sha256Digest`] の `Display`・[`Sha256Digest::to_hex`] は小文字 16 進 64 桁
//! を返す。記録（JSON）から読み戻す [`std::str::FromStr`] 実装・
//! [`serde::Deserialize`] 実装は、この形式に厳密に一致する文字列だけを
//! 受け付け、それ以外は `Err` にする
//! （`.claude/rules/evaluation-contract.md`「データの分割と凍結」）。
//!
//! [`Sha256Digest`] の `Deserialize` は `deserialize_str`（`String::deserialize`
//! を経由しない）で 64 文字ちょうどの検査を最初に行うことで、64 文字を
//! 大幅に超える文字列に対する `Sha256Digest` 単体の変換コストを長さ検査の
//! 時点で打ち切る。ただし、JSON デシリアライザ（`serde_json`）はエスケープを
//! 含む文字列を走査してからでないと `visit_str` を呼べないため、
//! **この型だけでは JSON ドキュメント全体の読み込みサイズを制限できない**。
//! 外部から来た JSON 全体（凍結記録ファイル等）を読み込む呼び出し側は、
//! パース前に [`crate::fs::read_bounded`] 等でバイト長を確認してから
//! `serde_json` へ渡すこと（`crates/data/src/eval_freeze.rs` の
//! `FreezeRecord` 読み込み口を参照。REQ-39・REQ-17）。

use std::fmt;
use std::str::FromStr;

use sha2::{Digest as _, Sha256};

/// 生バイト列の sha256 ダイジェスト（32 バイト）。
///
/// `Display`・[`Sha256Digest::to_hex`] は小文字 16 進 64 桁を返す。
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Sha256Digest([u8; 32]);

impl Sha256Digest {
    /// バイト列から sha256 を計算する。
    #[must_use]
    pub fn of_bytes(bytes: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        Sha256Digest(hasher.finalize().into())
    }

    /// 既に計算済みの生の 32 バイトから直接構築する（crate 内限定）。
    ///
    /// [`crate::fs::sha256_file_bounded`] がストリームで sha256 を計算した
    /// 結果（`sha2::Sha256::finalize()` の出力）を本型へ変換するために使う。
    /// フィールドを非公開にしたままこの用途だけを許すため、`pub(crate)` に
    /// 留める（外部 crate から任意の 32 バイトを「有効なダイジェスト」として
    /// 偽装できないようにする）。
    #[must_use]
    pub(crate) fn from_array(bytes: [u8; 32]) -> Self {
        Sha256Digest(bytes)
    }

    /// 生の 32 バイトを返す。
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// 小文字 16 進 64 桁の文字列表現。
    #[must_use]
    pub fn to_hex(&self) -> String {
        lower_hex(&self.0)
    }
}

/// バイト列を小文字 16 進文字列に変換する（[`Sha256Digest::to_hex`]・
/// [`crate::canonical::DefinitionHash::to_hex`] の共通実装）。
///
/// [`crate::canonical::DefinitionHash`] と本モジュールの [`Sha256Digest`] は
/// 「正準化済みハッシュ」と「生バイト列ハッシュ」の誤用防止のため別の型に
/// 分けているが（各モジュールの doc を参照）、16 進エンコードそのものは
/// どちらのハッシュ値にも意味の違いが無いため、ここへ 1 箇所に集約する
/// （crate 内限定。両モジュールとも sha256 の 32 バイト出力を渡す前提）。
pub(crate) fn lower_hex(bytes: &[u8]) -> String {
    use fmt::Write as _;
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        // `write!` は `String` への書き込みで失敗しないため `unwrap`/`expect`
        // を使わず戻り値を無視できる（この書き込み先に限り失敗しえない）。
        let _ = write!(out, "{byte:02x}");
    }
    out
}

impl fmt::Display for Sha256Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl fmt::Debug for Sha256Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sha256Digest({})", self.to_hex())
    }
}

/// [`Sha256Digest`] の文字列パース（外部から来た記録の読み戻し）が失敗した理由。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ParseDigestError {
    /// 長さが 64 文字ちょうどでなかった（`found` は実際の文字数）。
    InvalidLength { found: usize },
    /// 64 文字ではあるが、小文字 16 進以外の文字（大文字・非 16 進）を含んでいた。
    InvalidHexDigit,
}

impl fmt::Display for ParseDigestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseDigestError::InvalidLength { found } => {
                write!(
                    f,
                    "sha256 digest must be exactly 64 lowercase hex characters, found {found}"
                )
            }
            ParseDigestError::InvalidHexDigit => {
                write!(f, "sha256 digest must contain only lowercase hex digits")
            }
        }
    }
}

impl std::error::Error for ParseDigestError {}

impl FromStr for Sha256Digest {
    type Err = ParseDigestError;

    /// 小文字 16 進ちょうど 64 桁の文字列だけを受け付ける（外部入力の経路。
    /// `unwrap`/`expect`/添字アクセスを使わず `get`・`char::to_digit` で処理する。
    /// `.claude/rules/coding-rust.md`「外部入力」）。
    ///
    /// 長さの確認（`s.len()`。バイト長、`O(1)`）を可変長バッファの確保より
    /// 先に行う。64 文字を超える巨大な文字列を渡されても、長さ不一致を検出した
    /// 時点で直ちに拒否し、これ以上の走査・確保を行わない
    /// （`.claude/rules/security.md`「資源の上限」・`.claude/rules/coding-rust.md`
    /// 「サイズ・件数を上限検証してからアロケーションに使う」）。ただし `s` 自体は
    /// 呼び出し元が既に確保済みの文字列であるため、本関数はその確保を防げない
    /// （JSON デシリアライズ経路の残存リスクはモジュール doc・
    /// [`Sha256Digest`] の `Deserialize` 実装の doc を参照）。
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // バイト長を先に確認する。有効な入力（小文字 16 進 64 桁）は 1 バイト
        // 1 文字の ASCII のみのため、この時点でバイト長と文字数は一致する。
        // 一致しない入力はこの後の文字列走査を行わずに拒否するため、`found`
        // には文字数の代わりにバイト長を使う（`chars().count()` による走査を
        // 避けるため）。
        if s.len() != 64 {
            return Err(ParseDigestError::InvalidLength { found: s.len() });
        }

        // ここまでで長さは 64 バイトに固定されているため、以降のバッファは
        // すべて固定長（32 バイト）で済み、入力長に比例した確保は発生しない。
        let ascii = s.as_bytes();
        let mut bytes = [0u8; 32];
        for (index, byte_slot) in bytes.iter_mut().enumerate() {
            let high_index = index * 2;
            // `get` で範囲外アクセスを避ける（64 バイトであることは上で確認済み
            // だが、添字アクセス `[]` を使わない規約に合わせる）。
            let high_ascii = *ascii
                .get(high_index)
                .ok_or(ParseDigestError::InvalidHexDigit)?;
            let low_ascii = *ascii
                .get(high_index + 1)
                .ok_or(ParseDigestError::InvalidHexDigit)?;

            // 大文字 hex を拒否するため、`is_ascii_hexdigit` ではなく明示的に
            // 小文字と数字だけを許可する `char::to_digit` の結果を使いつつ、
            // 元のバイトが大文字でないことも確認する。
            if high_ascii.is_ascii_uppercase() || low_ascii.is_ascii_uppercase() {
                return Err(ParseDigestError::InvalidHexDigit);
            }
            let high_digit = char::from(high_ascii)
                .to_digit(16)
                .ok_or(ParseDigestError::InvalidHexDigit)?;
            let low_digit = char::from(low_ascii)
                .to_digit(16)
                .ok_or(ParseDigestError::InvalidHexDigit)?;

            // `to_digit(16)` は 0..=15 の範囲を保証するため、u8 への変換で
            // オーバーフローしない（`try_into` で明示的に扱う）。
            let high_u8: u8 = high_digit
                .try_into()
                .map_err(|_| ParseDigestError::InvalidHexDigit)?;
            let low_u8: u8 = low_digit
                .try_into()
                .map_err(|_| ParseDigestError::InvalidHexDigit)?;
            *byte_slot = (high_u8 << 4) | low_u8;
        }

        Ok(Sha256Digest(bytes))
    }
}

impl serde::Serialize for Sha256Digest {
    /// 16 進文字列として直列化する（記録の JSON 表現。
    /// `crates/data/src/eval_freeze.rs` の `FreezeRecord` が使う）。
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_hex())
    }
}

impl<'de> serde::Deserialize<'de> for Sha256Digest {
    /// [`FromStr`] を通して検証する（外部から来た記録は fail-closed。
    /// `.claude/rules/evaluation-contract.md`）。
    ///
    /// `String::deserialize` は経由しない。`String::deserialize` は入力長に
    /// 比例したオーナー付き `String` を必ず確保してから返すため、64 文字の
    /// 長さ検査（[`FromStr`] 実装の先頭）より先に、外部入力（巨大な文字列）に
    /// 比例するメモリを消費してしまう
    /// （`.claude/rules/security.md`「ガード層: 資源の上限」・REQ-39）。
    /// 代わりに [`serde::de::Visitor`] で `&str` を直接受け取り、確保せずに
    /// [`Sha256Digest::from_str`] へ渡す（`from_str` は長さ検査を最初に行う）。
    ///
    /// **残存リスク（このデシリアライザだけでは閉じない。REQ-39）**:
    /// `serde_json` はエスケープ（`\"` 等）を含まない文字列であれば入力
    /// バッファを借用したまま `visit_str` を呼べるが、エスケープを含む
    /// 文字列は展開後の内容を保持する `String` を先に確保してから
    /// `visit_str` を呼ぶ。したがって、この `Visitor` 自体は追加の確保を
    /// 行わないものの、`serde_json` 側が展開のために確保するバッファの
    /// 大きさは、この実装の 64 文字検査より前に決まってしまう
    /// （巨大なエスケープ済み JSON 文字列を防げない）。この残存リスクは、
    /// **JSON ドキュメント全体を読み込む呼び出し側**が
    /// [`crate::fs::read_bounded`] 等でパース前にバイト長を制限すること
    /// で閉じる契約とする（`crates/data/src/eval_freeze.rs` の
    /// `FreezeRecord` 読み込み関数を参照）。
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct DigestVisitor;

        impl<'de> serde::de::Visitor<'de> for DigestVisitor {
            type Value = Sha256Digest;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a 64-character lowercase hex sha256 digest string")
            }

            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Sha256Digest::from_str(v).map_err(E::custom)
            }
        }

        deserializer.deserialize_str(DigestVisitor)
    }
}

/// 生バイト列（正準化を経ない）の sha256 を計算する。
///
/// [`crate::canonical::sha256_hex_bytes`] はこの関数へ委譲する（sha256 の
/// 実装を 1 か所にまとめる。`crates/core/src/canonical.rs` の doc 参照）。
pub(crate) fn sha256_hex_bytes(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 既知ベクタ（NIST 標準の sha256 空文字列・`"abc"`）。証拠の種別: 一次資料。
    #[test]
    fn req17_req27_of_bytes_known_vectors() {
        assert_eq!(
            Sha256Digest::of_bytes(b"").to_hex(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            Sha256Digest::of_bytes(b"abc").to_hex(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn req17_req27_display_is_lowercase_hex_64_chars() {
        let digest = Sha256Digest::of_bytes(b"abc");
        let s = digest.to_string();
        assert_eq!(s.len(), 64);
        assert!(
            s.chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        );
    }

    #[test]
    fn req17_req27_debug_contains_hex() {
        let digest = Sha256Digest::of_bytes(b"abc");
        let debug = format!("{digest:?}");
        assert!(debug.contains(&digest.to_hex()));
    }

    #[test]
    fn req17_req27_different_bytes_differ() {
        assert_ne!(
            Sha256Digest::of_bytes(b"abc"),
            Sha256Digest::of_bytes(b"abd")
        );
    }

    /// REQ-17: `to_hex` と `from_str` の往復で元の値に戻る（決定性）。
    #[test]
    fn req17_hex_round_trip() {
        let digest = Sha256Digest::of_bytes(b"round trip");
        let hex = digest.to_hex();
        let parsed: Sha256Digest = hex.parse().expect("valid hex は成功するはず");
        assert_eq!(digest, parsed);
    }

    /// REQ-17: 大文字 hex・63 桁・65 桁・非 16 進文字・空文字は fail-closed で拒否する。
    #[test]
    fn req17_from_str_rejects_malformed_input() {
        let valid = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        let uppercase = valid.to_uppercase();
        assert_eq!(
            Sha256Digest::from_str(&uppercase),
            Err(ParseDigestError::InvalidHexDigit)
        );

        let too_short = &valid[..63];
        assert_eq!(
            Sha256Digest::from_str(too_short),
            Err(ParseDigestError::InvalidLength { found: 63 })
        );

        let too_long = format!("{valid}f");
        assert_eq!(
            Sha256Digest::from_str(&too_long),
            Err(ParseDigestError::InvalidLength { found: 65 })
        );

        let non_hex = format!("{}g", &valid[..63]);
        assert_eq!(
            Sha256Digest::from_str(&non_hex),
            Err(ParseDigestError::InvalidHexDigit)
        );

        assert_eq!(
            Sha256Digest::from_str(""),
            Err(ParseDigestError::InvalidLength { found: 0 })
        );
    }

    /// REQ-17: serde で 16 進文字列として直列化・逆直列化できる。
    #[test]
    fn req17_serde_round_trip() {
        let digest = Sha256Digest::of_bytes(b"serde");
        let json = serde_json::to_string(&digest).expect("serialize は成功するはず");
        assert_eq!(json, format!("\"{}\"", digest.to_hex()));

        let parsed: Sha256Digest = serde_json::from_str(&json).expect("deserialize は成功するはず");
        assert_eq!(digest, parsed);
    }

    /// REQ-17: 不正な文字列の Deserialize は `Err` になる（fail-closed）。
    #[test]
    fn req17_serde_deserialize_rejects_malformed_string() {
        let result: Result<Sha256Digest, _> = serde_json::from_str("\"not-a-hash\"");
        assert!(result.is_err());
    }

    /// REQ-17: `Display` が小文字 16 進 64 桁になる。
    #[test]
    fn req17_display_is_lowercase_hex_64_chars() {
        let digest = Sha256Digest([0u8; 32]);
        let s = digest.to_string();
        assert_eq!(s.len(), 64);
        assert!(
            s.chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        );
        assert_eq!(s, "0".repeat(64));
    }
}
