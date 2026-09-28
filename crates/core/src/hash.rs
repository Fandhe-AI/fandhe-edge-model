//! 生バイト列の sha256 ダイジェスト型（REQ-17・REQ-27）。
//!
//! [`crate::canonical::DefinitionHash`] は `Definition` を正準化 JSON へ
//! 変換してからハッシュする専用型だが、こちらは既に読み込んだ・受け取った
//! バイト列をそのままハッシュする汎用型で、評価データの凍結（REQ-17。TASK-17.2）や
//! モデルパッケージの評価前後比較（REQ-27。TASK-27.1-1・issue #69）など、
//! 「正準化を経ない生バイト列の同一性」を扱う上位層から使われる。
//!
//! 2 つを別の型にしているのは [`crate::canonical`] の doc と同じ理由で、
//! 「正準化済みハッシュ」と「生バイト列ハッシュ」を取り違えて比較する誤用を
//! コンパイルエラーで防ぐため（`PartialEq` は同じ型同士でしか比較できない）。
//!
//! # 出典・経緯
//!
//! TASK-17.2-1（issue #47）向けに提案された公開 API と同じ形にしてある
//! （`of_bytes`・`as_bytes`・`to_hex`・`Display`・`Debug`）。当該 issue の PR が
//! 先にマージされた場合はそちらを正とし、本ファイルとの add/add 衝突は
//! そちらの実装を採用して解消する（issue #69 の PR 本文にも明記する）。

use sha2::{Digest as _, Sha256};
use std::fmt;

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
            // `write!` は `String` への書き込みで失敗しないため `unwrap`/`expect`
            // を使わず戻り値を無視できる（この書き込み先に限り失敗しえない）。
            let _ = write!(out, "{byte:02x}");
        }
        out
    }
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
                .all(|c| c.is_ascii_hexdigit() && !c.is_uppercase())
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
}
