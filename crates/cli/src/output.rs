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
}
