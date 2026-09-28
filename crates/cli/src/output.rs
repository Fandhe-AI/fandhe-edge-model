//! ok 終了時の判定結果 JSON を stdout 相当の `Write` へ書き出す出力関数
//! （REQ-21 正常系・TASK-21.1-2）。
//!
//! # 呼び出し文脈
//!
//! 呼び出し元は CLI の `infer` サブコマンド（TASK-33.1。現状は未配線。
//! `main.rs` は依然として引数を読まず exit 70 を返すスタブのまま）。配線後
//! は `std::io::stdout().lock()` を渡し、戻り値の [`ExitCode::Ok`] を
//! `main` の戻り値としてそのまま使う想定。`--input-file` 経由の一括推論
//! （evaluation-contract.md が認める「1 行 1 JSON」の例外）でも、入力 1 件
//! ごとに本関数を 1 回呼ぶ形で同じ契約を再利用する想定。
//!
//! 業務ロジック（選択肢・スコアの検証）は `fandhe-edge-core` の
//! `judgment::JudgmentResult::new` 側が担い、本関数は「検証済みの値を 1 行
//! の JSON として書く」だけの薄いアダプターに留める
//! （`.claude/rules/coding-rust.md`「操作アダプターは薄く保ち、業務ロジッ
//! クは下位層に置く」）。

use fandhe_edge_core::exitcode::ExitCode;
use fandhe_edge_core::judgment::JudgmentResult;
use std::io::{self, Write};

/// [`JudgmentResult`] を JSON 1 行＋改行として `out` へ書き、
/// [`ExitCode::Ok`] を返す。
///
/// 直列化に失敗した場合（`JudgmentResult::to_json_line` が `Err` を返す場
/// 合）は `out` へ何も書かずに `io::Error` を返す（stdout に不完全な JSON
/// を残さない。REQ-21「出力は 1 呼び出しにつき JSON 1 つ」）。
///
/// # 部分書き込み失敗時の方針（PR #202 レビュー指摘・P1）
///
/// `Write::write_all` は `out` の実装によっては、行の一部を書き込んだ後に
/// エラーを返すことがある（例: パイプの相手側が先に閉じた stdout）。この
/// 場合、`out` の内部状態には不完全な JSON バイト列が残り得るが、それを
/// 本関数が検知・巻き戻す手段は `Write` トレイトの外側から存在しない
/// （`out` は任意の書き込み先を表す汎用型で、シーク・トランザクション等
/// の取り消し操作を持たない）。本関数が保証するのは次の 1 点のみである:
/// **1 回の呼び出しにつき `write_all` を高々 1 回しか実行しない**（リト
/// ライや、エラー後に別の内容を重ねて書くことをしない）。これにより
/// 「不完全な JSON の後ろに別の JSON が連結される」事態を本関数の中では
/// 起こさない。
///
/// 呼び出し側（CLI の `infer` 一括推論。TASK-33.1。現状は未配線）は、本
/// 関数が `Err` を返した時点で **その入力ファイルに対する後続の呼び出し
/// を打ち切り**、以降の入力に対して本関数を呼ばないこと。打ち切らずに
/// 次の入力へ処理を進めると、直前の不完全な行の直後に次の JSON 行が書か
/// れてしまい、「1 行 1 JSON」という出力契約を読み手側が復元できなくなる
/// （壊れた行と正常な行の境界が改行だけでは判別できないため）。
///
/// # Errors
/// 直列化エラー、または `out` への書き込み・flush の失敗を
/// `io::Error`（`ErrorKind::Other` または下位の I/O エラー）として返す。
pub fn write_ok_judgment<W: Write>(out: &mut W, result: &JudgmentResult) -> io::Result<ExitCode> {
    let mut line = result
        .to_json_line()
        .map_err(|error| io::Error::other(error.to_string()))?;
    line.push('\n');

    out.write_all(line.as_bytes())?;
    out.flush()?;

    Ok(result.exit_code())
}

#[cfg(test)]
mod tests {
    use super::*;
    use fandhe_edge_core::definition::Choice;

    fn choice(id: &str) -> Choice {
        Choice {
            id: id.to_string(),
            display_name: id.to_string(),
            description: String::new(),
        }
    }

    /// REQ-21: 書き込み成功時、バッファに JSON 1 行＋改行のみが書かれ、
    /// 戻り値が `ExitCode::Ok`（`.code() == 0`）であること。
    #[test]
    fn req21_writes_single_json_line_and_returns_ok() {
        let options = [choice("a"), choice("b"), choice("c")];
        let result = JudgmentResult::new(&options, "row-1", "a", &[0.7, 0.2, 0.1]).unwrap();

        let mut buffer: Vec<u8> = Vec::new();
        let exit_code = write_ok_judgment(&mut buffer, &result).unwrap();

        assert_eq!(exit_code, ExitCode::Ok);
        assert_eq!(exit_code.code(), 0);
        assert_eq!(
            std::process::ExitCode::from(exit_code),
            std::process::ExitCode::from(0u8)
        );

        let text = String::from_utf8(buffer).unwrap();
        assert_eq!(
            text,
            "{\"id\":\"row-1\",\"status\":\"ok\",\"predicted_label\":\"a\",\"scores\":{\"a\":0.7,\"b\":0.2,\"c\":0.1}}\n"
        );
        assert_eq!(text.matches('\n').count(), 1, "must write exactly one line");
    }

    /// REQ-21: 定義ファイルの宣言順と一致した `predicted_label`／`scores`
    /// のキー順で書かれること。
    #[test]
    fn req21_preserves_declaration_order_from_definition() {
        let options = [choice("z"), choice("a"), choice("m")];
        let result = JudgmentResult::new(&options, "row-2", "z", &[0.5, 0.3, 0.2]).unwrap();

        let mut buffer: Vec<u8> = Vec::new();
        write_ok_judgment(&mut buffer, &result).unwrap();

        let text = String::from_utf8(buffer).unwrap();
        assert_eq!(
            text,
            "{\"id\":\"row-2\",\"status\":\"ok\",\"predicted_label\":\"z\",\"scores\":{\"z\":0.5,\"a\":0.3,\"m\":0.2}}\n"
        );
    }

    /// REQ-21: `id` に二重引用符・改行・制御文字を含む場合でも、
    /// 書き込まれるのは JSON 1 行＋末尾の改行 1 つのみであること
    /// （Review 指摘: id の値に依らず「出力は 1 呼び出しにつき JSON 1 つ」
    /// という出力契約を保つ）。
    #[test]
    fn req21_writes_single_line_when_id_contains_quotes_and_control_chars() {
        let options = [choice("a")];
        let raw_id = "row\"with\nquote\tand\u{0007}control";
        let result = JudgmentResult::new(&options, raw_id, "a", &[1.0]).unwrap();

        let mut buffer: Vec<u8> = Vec::new();
        let exit_code = write_ok_judgment(&mut buffer, &result).unwrap();
        assert_eq!(exit_code, ExitCode::Ok);

        let text = String::from_utf8(buffer).unwrap();
        assert_eq!(
            text.matches('\n').count(),
            1,
            "must write exactly one line even when id contains raw control chars"
        );
        assert!(text.ends_with('\n'));

        let line = text.trim_end_matches('\n');
        assert_eq!(
            line,
            r#"{"id":"row\"with\nquote\tand\u0007control","status":"ok","predicted_label":"a","scores":{"a":1.0}}"#
        );
    }

    /// 書き込みに失敗する `Write` を渡した場合、`Err` が返り
    /// `ExitCode::Ok` は返らないこと。
    #[test]
    fn req21_propagates_write_failure() {
        struct FailingWriter;
        impl Write for FailingWriter {
            fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
                Err(io::Error::other("simulated write failure"))
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }

        let options = [choice("a")];
        let result = JudgmentResult::new(&options, "row-3", "a", &[1.0]).unwrap();

        let mut writer = FailingWriter;
        let outcome = write_ok_judgment(&mut writer, &result);
        assert!(outcome.is_err());
    }

    /// 部分書き込み後に失敗する `Write` を渡した場合でも、`out` には途中
    /// までの不完全なバイト列以上のものが書かれず（リトライ・上書きをし
    /// ない）、`flush` が呼ばれないこと（PR #202 レビュー指摘・P1: 部分
    /// 書き込み失敗の経路を確認できていなかった）。
    ///
    /// `write_all` は内部で複数回 `write` を呼びうるため、このテストは
    /// 「最初の `write` 呼び出しで成功して一部バイトを書き、2 回目の
    /// `write` 呼び出しで失敗する」という部分書き込みの経路を模擬する。
    #[test]
    fn req21_stops_after_partial_write_failure_without_flush() {
        struct PartialThenFailingWriter {
            written: Vec<u8>,
            write_calls: usize,
            flush_calls: usize,
        }
        impl Write for PartialThenFailingWriter {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                self.write_calls += 1;
                if self.write_calls == 1 {
                    // 最初の呼び出しは先頭 4 バイトだけを書いたことにして
                    // 成功させる（部分書き込み）。
                    let n = buf.len().min(4);
                    self.written.extend_from_slice(&buf[..n]);
                    Ok(n)
                } else {
                    Err(io::Error::other("simulated partial write failure"))
                }
            }
            fn flush(&mut self) -> io::Result<()> {
                self.flush_calls += 1;
                Ok(())
            }
        }

        let options = [choice("a")];
        let result = JudgmentResult::new(&options, "row-4", "a", &[1.0]).unwrap();

        let mut writer = PartialThenFailingWriter {
            written: Vec::new(),
            write_calls: 0,
            flush_calls: 0,
        };
        let outcome = write_ok_judgment(&mut writer, &result);

        assert!(
            outcome.is_err(),
            "partial write followed by failure must surface as Err"
        );
        assert_eq!(
            writer.write_calls, 2,
            "write_all の内部リトライ以外に本関数が追加で write を呼んではならない"
        );
        assert_eq!(
            writer.flush_calls, 0,
            "書き込みに失敗した場合は flush を呼んではならない（不完全な行を確定させない）"
        );
        assert_eq!(
            writer.written.len(),
            4,
            "out 側に残るのは write_all が内部で書いた分のみで、本関数がそれ以上書き足してはならない"
        );
    }
}
