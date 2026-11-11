//! パッケージの配置の指紋（`PackageIdentity::layout_sha256`。#708・REQ-39）。
//!
//! 常駐は shard など GiB 級のファイルの sha256 を起動時に 1 回だけ取る。委ねる側が毎回取り直すと常駐の意味が
//! なくなるため、構成ファイルごとの「相対名・dev・ino・サイズ・mtime（ns）・ctime（ns）」を決まった順に並べた
//! sha256 を指紋とし、読み込み時の指紋と今の指紋が一致するときだけ委ねる。書き換え・差し替え・削除は
//! ctime・ino・サイズのどれかに現れ、一致しなければ委ねずに単独へ戻る（単独が全量検証して拒否する）。
//!
//! 常駐側は `prepare` が実際に読んだ fd の `metadata`（[`FileStat::of`]）から作り、パスを開き直さない
//! （開き直す間の差し替えの隙を作らない）。委ねる側も軽い検査と同じ閉じ込めた開き方で得た fd から作る。
//! 計算は [`layout_sha256`] の 1 箇所に集約する。ctime は OS が更新するため、mtime を戻す改竄でも変わる。
//! 限界: ctime を含めても、同一 inode のまま同一サイズで ctime の粒度内に書き換える競合は検出できない
//! （証拠種別: 設計上の推定）。
//! mmap での書き込みは、msync や writeback まで mtime・ctime を更新しない OS があり、その間の変更は検出できない。
//! 同じ権限（同一ユーザー）を持つ攻撃者は脅威モデルの範囲外とする（その権限なら常駐やソケットを直接操作できる）。

use std::fs::File;
use std::io;

use fandhe_edge_core::hash::Sha256Digest;

/// 構成ファイル 1 件の識別情報。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileStat {
    name: String,
    dev: u64,
    ino: u64,
    size: u64,
    mtime_ns: i128,
    ctime_ns: i128,
}

impl FileStat {
    /// 開いた fd の `metadata` から作る（`name` はパッケージ内の相対名）。
    ///
    /// # Errors
    /// `metadata` を読めないとき、または非対応 OS。
    #[cfg(unix)]
    pub fn of(name: &str, file: &File) -> io::Result<Self> {
        use std::os::unix::fs::MetadataExt;
        let m = file.metadata()?;
        let ns = |sec: i64, nsec: i64| i128::from(sec) * 1_000_000_000 + i128::from(nsec);
        Ok(Self {
            name: name.to_string(),
            dev: m.dev(),
            ino: m.ino(),
            size: m.len(),
            mtime_ns: ns(m.mtime(), m.mtime_nsec()),
            ctime_ns: ns(m.ctime(), m.ctime_nsec()),
        })
    }

    /// 非対応 OS（ガード層が先に拒否するため通常は呼ばれない）。
    ///
    /// # Errors
    /// 常に `Unsupported`。
    #[cfg(not(unix))]
    pub fn of(name: &str, file: &File) -> io::Result<Self> {
        let _ = (name, file);
        Err(io::ErrorKind::Unsupported.into())
    }
}

/// 構成ファイル群の指紋（sha256 の小文字 16 進）。名前順に並べ、各欄を区切って連結する。
#[must_use]
pub fn layout_sha256(mut stats: Vec<FileStat>) -> String {
    stats.sort_by(|a, b| a.name.cmp(&b.name));
    let mut buf = String::new();
    for s in &stats {
        buf.push_str(&format!(
            "{}:{}|{}|{}|{}|{}|{}\n",
            s.name.len(),
            s.name,
            s.dev,
            s.ino,
            s.size,
            s.mtime_ns,
            s.ctime_ns
        ));
    }
    Sha256Digest::of_bytes(buf.as_bytes()).to_hex()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stat(name: &str, ino: u64, ctime_ns: i128) -> FileStat {
        FileStat {
            name: name.into(),
            dev: 1,
            ino,
            size: 10,
            mtime_ns: 5,
            ctime_ns,
        }
    }

    /// REQ-39: 順序に依存せず、どの欄の変化も指紋を変える。
    #[test]
    fn layout_is_order_independent_and_field_sensitive() {
        let a = layout_sha256(vec![stat("a", 1, 2), stat("b", 3, 4)]);
        assert_eq!(a, layout_sha256(vec![stat("b", 3, 4), stat("a", 1, 2)]));
        assert_ne!(a, layout_sha256(vec![stat("a", 1, 3), stat("b", 3, 4)]));
        assert_ne!(a, layout_sha256(vec![stat("a", 9, 2), stat("b", 3, 4)]));
        assert_ne!(a, layout_sha256(vec![stat("a", 1, 2)]));
        assert_eq!(a.len(), 64);
    }
}
