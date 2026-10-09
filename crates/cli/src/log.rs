//! stderr へ人間向けのテキストログを出す経路（REQ-33・TASK-33.2-2・#139）。
//!
//! # 呼び出し文脈
//!
//! `main.rs` が工程を実行する分岐で使う。stdout は「1 呼び出し 1 JSON」の結果専用、
//! 進捗・ログは stderr へ、という契約（REQ-33）を、書き込み先を別の `Write` として
//! 持つ構造で守る（stdout 用の `Write` をここへ渡す経路が無い）。help と引数エラーの
//! 経路ではログを出さない（stderr を空に保つ既存テストの契約）。
//!
//! # security.md との関係
//!
//! メッセージは `&'static str` に限り、利用者の入力値・パス・データ本文・学習ワーカー
//! 由来の文字列を渡せないようにする（型による防御）。JSON にはしない。
//!
//! # 書き込み失敗
//!
//! ログの失敗で stdout の JSON と終了コードを変えないため（fail-safe）、書き込み
//! エラーは無視する。

use crate::args::Subcommand;
use std::io::Write;

/// 行頭に付ける固定の接頭辞。
const PREFIX: &str = "fandhe-edge: ";

/// stderr 相当の `Write` へテキスト 1 行ずつ書くロガー。
pub struct StderrLog<W: Write> {
    out: W,
}

impl<W: Write> StderrLog<W> {
    /// 書き込み先を受け取る（main では `std::io::stderr().lock()`）。
    pub fn new(out: W) -> Self {
        Self { out }
    }

    /// `fandhe-edge: <stage>: <msg>` を 1 行書く。失敗は無視する。
    pub fn info(&mut self, stage: Subcommand, msg: &'static str) {
        let _ = writeln!(self.out, "{PREFIX}{}: {msg}", stage.name());
    }

    /// `fandhe-edge: <stage>: <msg>: <count>` を 1 行書く（件数だけを添える。失敗は無視する）。
    pub fn info_count(&mut self, stage: Subcommand, msg: &'static str, count: usize) {
        let _ = writeln!(self.out, "{PREFIX}{}: {msg}: {count}", stage.name());
    }

    /// 工程が決まる前の行 `fandhe-edge: <msg>` を書く。失敗は無視する。
    pub fn info_top(&mut self, msg: &'static str) {
        let _ = writeln!(self.out, "{PREFIX}{msg}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    struct FailingWriter;

    impl Write for FailingWriter {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::Error::other("closed"))
        }
        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::other("closed"))
        }
    }

    /// REQ-33: ログ行の形式が固定でテキスト（JSON でない）。
    #[test]
    fn req33_log_line_format_is_exact() {
        let mut buf = Vec::new();
        {
            let mut log = StderrLog::new(&mut buf);
            log.info(Subcommand::Package, "start");
            log.info_top("hello");
        }
        assert_eq!(
            String::from_utf8(buf).expect("utf8"),
            "fandhe-edge: package: start\nfandhe-edge: hello\n"
        );
    }

    /// REQ-33: 書き込みに失敗しても panic せず呼び出し側へ影響しない。
    #[test]
    fn req33_log_write_failure_is_ignored() {
        let mut log = StderrLog::new(FailingWriter);
        log.info(Subcommand::Train, "x");
        log.info_top("y");
    }
}
