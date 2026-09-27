//! 生バイト列の sha256 計算ヘルパー（低レベル・正準化を経ない）。
//!
//! # 位置づけ
//!
//! ここで提供するハッシュは、入力（ファイル・バイト列）をそのまま sha256 に
//! 通しただけの値であり、[`crate::definition`] が TASK-15.5 で実装する
//! 「定義ファイルの正準化ハッシュ」（表示名・説明文のみの変更では ID が
//! 変わらないよう正準化してから計算するハッシュ）とは別物である。
//! 評価データの凍結（REQ-17。`crates/data` の `eval_freeze` モジュールから
//! 呼ばれる想定）のように、内容の一字一句が変わればハッシュも変わってほしい
//! 用途に使う。正準化ハッシュが必要な呼び出し元は、このモジュールではなく
//! `definition` 側の正準化ハッシュ関数を使うこと。
//!
//! # 呼び出し文脈
//!
//! 評価データの凍結（データ契約層。REQ-17）・将来の配布パッケージの完全性検証
//! （REQ-28/30〜32・security.md「完全性と版」）など、生バイト列の同一性を
//! 判定したい層から呼ばれる想定。共通コアはどの層にも依存しないため、
//! 呼び出し元（データ契約・推論ランタイム等）から一方向に参照される。

use std::io::{self, Read};

/// バッファサイズ（ストリーミング読み込みの 1 チャンク分）。
/// ファイル全体をメモリに一括ロードしないための固定長バッファ。
const CHUNK_SIZE: usize = 64 * 1024;

/// 任意の [`Read`] からストリーミングで sha256 を計算し、小文字 16 進 64 桁で返す。
///
/// ファイル全体を一度にメモリへ載せず `CHUNK_SIZE` ごとに読み進めるため、
/// 大きな評価データ・学習データに対しても一定のメモリ使用量で計算できる。
/// 呼び出し元がサイズ上限の検証（REQ-39）を済ませたリーダーを渡す想定で、
/// 本関数自体はサイズ上限を検査しない。
///
/// # エラー
///
/// 読み込み中の I/O エラーはそのまま呼び出し元へ伝播する（panic しない）。
pub fn sha256_hex_of_reader<R: Read>(mut reader: R) -> io::Result<String> {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    let mut buf = [0u8; CHUNK_SIZE];
    loop {
        let read_len = reader.read(&mut buf)?;
        if read_len == 0 {
            break;
        }
        // read() は要求した長さより短く読むことがあるため、実際に読めた
        // 範囲（先頭 read_len バイト）だけをハッシュへ入力する。
        if let Some(chunk) = buf.get(..read_len) {
            hasher.update(chunk);
        }
    }
    // sha2 0.11 の `finalize()` は `hybrid-array` の `Array` を返し `LowerHex` を
    // 実装しないため、バイト列を 1 バイトずつ 16 進 2 桁へ整形して連結する
    // （新規依存（`hex` crate 等）を追加しないための最小実装）。
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest.as_slice() {
        hex.push_str(&format!("{byte:02x}"));
    }
    Ok(hex)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// REQ-17: 空バイト列の sha256 は公知の具体値
    /// (`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`) と一致する。
    #[test]
    fn sha256_hex_of_reader_empty_matches_known_digest() {
        let digest = sha256_hex_of_reader(Cursor::new(b"")).expect("in-memory read must not fail");
        assert_eq!(
            digest,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    /// REQ-17: "hello world\n" の sha256 は公知の具体値と一致する
    /// (`sha256sum` で事前確認済み)。
    #[test]
    fn sha256_hex_of_reader_hello_world_matches_known_digest() {
        let digest = sha256_hex_of_reader(Cursor::new(b"hello world\n"))
            .expect("in-memory read must not fail");
        assert_eq!(
            digest,
            "a948904f2f0f479b8f8197694b30184b0d2ed1c1cd2a1ec0fb85d299a192a447"
        );
    }

    /// 複数チャンクにまたがる長い入力でも一致すること（`CHUNK_SIZE` 境界の確認）。
    #[test]
    fn sha256_hex_of_reader_multi_chunk_matches_known_digest() {
        let data = vec![0x41u8; CHUNK_SIZE * 2 + 123];
        let digest =
            sha256_hex_of_reader(Cursor::new(&data)).expect("in-memory read must not fail");
        // 独立に sha2 クレートで直接計算した値と一致することを確認する
        // （ストリーミング実装が一括計算と同じ結果になることの検証）。
        use sha2::{Digest as _, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(&data);
        let finalized = hasher.finalize();
        let mut expected = String::with_capacity(finalized.len() * 2);
        for byte in finalized.as_slice() {
            expected.push_str(&format!("{byte:02x}"));
        }
        assert_eq!(digest, expected);
    }
}
