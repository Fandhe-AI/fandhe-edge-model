//! 判定結果・エラーの JSON を stdout／stderr 相当の `Write` へ書き出す出力
//! 関数（REQ-21 正常系・TASK-21.1-2。異常系・TASK-21.2）。
//!
//! # 呼び出し文脈
//!
//! 呼び出し元は CLI の `infer` サブコマンド（TASK-33.1。現状は未配線。
//! `main.rs` は引数解析までで、解析成功後は工程を実行せず未実装の
//! `ErrorReport` を書いて exit 70 を返す）。配線後
//! は `std::io::stdout().lock()` を渡し、戻り値の [`ExitCode`] を
//! `main` の戻り値としてそのまま使う想定。`--input-file` 経由の一括推論
//! （evaluation-contract.md が認める「1 行 1 JSON」の例外）でも、入力 1 件
//! ごとに本関数を 1 回呼ぶ形で同じ契約を再利用する想定。
//!
//! 業務ロジック（選択肢・スコアの検証、推論入力の型検証）は
//! `fandhe-edge-core` の `judgment::JudgmentResult::new`・
//! `infer_input::InferInput::parse` 側が担い、本モジュールは「検証済みの
//! 値・エラーを 1 行の JSON として書く」だけの薄いアダプターに留める
//! （`.claude/rules/coding-rust.md`「操作アダプターは薄く保ち、業務ロジッ
//! クは下位層に置く」）。
//!
//! # package の正常系（TASK-33.2-2）
//!
//! [`write_package_report`] は `package` 工程の exit 0 の結果 JSON を書く。呼び出し元は
//! `stage_output::emit_package_outcome`（配線は TASK-33.1-2・#136）。
//!
//! # evaluate の skipped（TASK-33.3・#140）
//!
//! [`write_evaluate_report`] は評価データ未定義の `evaluate` の exit 0 の JSON を書く。呼び出し元は
//! `stage_output::emit_evaluate_skipped`（配線は TASK-33.1-2・#136）。
//!
//! # 異常系（TASK-21.2）
//!
//! [`write_error_report`] は [`ErrorReport`] を JSON 1 行として書き出す。
//! `ErrorReport` のスキーマ（`{"code","message"}`）はここでは拡張しない
//! （REQ-21・REQ-33 の入出力契約の変更はユーザー承認事項。
//! `.claude/rules/evaluation-contract.md`）。`infer_input_error_report`・
//! `judgment_error_report`・`definition_error_report` は各層のエラー型を
//! `ErrorReport` へ変換するだけの薄い関数で、`message` の生成規則自体は
//! 各エラー型の実装に委ねる（`InferInputError`／`JudgmentError` は
//! `Display`、`DefinitionError` はパス・利用者指定値を含まない
//! `public_message`。PR #217 レビュー指摘・P0）。
//!
//! # 容量内訳（TASK-30.1-2・#123）
//!
//! [`package_capacity_json`]・[`write_package_capacity`] は容量計測（`fandhe-edge-runtime`）の
//! 内訳を `package` 工程（TASK-33.1。未配線）の JSON へ出す部品。`status`・`judgment` は
//! TASK-33.x・TASK-30.2 の責務で、ここでは出さない。
//!
//! # 除外記録（TASK-32.2・#114）
//!
//! [`export_exclusions_json`] は配布候補から外した構成と理由（`fandhe-edge-runtime` の
//! `export_exclusion`）を `package`・`select` 工程（TASK-33.x。未配線）の JSON へ出す部品。
//! 入出力契約は変えず、`package` 出力への埋め込みは配線 TASK の責務。

use fandhe_edge_core::definition::DefinitionError;
use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};
use fandhe_edge_core::infer_input::InferInputError;
use fandhe_edge_core::judgment::{JudgmentError, JudgmentResult};
use fandhe_edge_core::stage_report::{EvaluateReport, PackageReport};
use fandhe_edge_runtime::capacity::{CapacityBreakdown, CapacityError};
use fandhe_edge_runtime::export_exclusion::{ExclusionReason, ExclusionRecord};
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

/// [`ErrorReport`] を JSON 1 行＋改行として `out` へ書き、
/// `report.code` を返す（TASK-21.2）。
///
/// [`write_ok_judgment`] と同じ保証を持つ: 直列化に失敗した場合は `out` へ
/// 何も書かず `Err` を返し、`write_all` は 1 回の呼び出しにつき高々 1 回
/// しか実行しない（部分書き込み失敗時にリトライ・追記・flush をしない。
/// `write_ok_judgment` のドキュメントコメント「部分書き込み失敗時の方
/// 針」と同じ理由・同じ呼び出し側の責務）。
///
/// # Errors
/// 直列化エラー、または `out` への書き込み・flush の失敗を
/// `io::Error`（`ErrorKind::Other` または下位の I/O エラー）として返す。
pub fn write_error_report<W: Write>(out: &mut W, report: &ErrorReport) -> io::Result<ExitCode> {
    let mut line = report
        .to_json_line()
        .map_err(|error| io::Error::other(error.to_string()))?;
    line.push('\n');

    out.write_all(line.as_bytes())?;
    out.flush()?;

    Ok(report.code)
}

/// [`PackageReport`]（`package` 工程の exit 0 の結果）を JSON 1 行＋改行として
/// `out` へ書き、[`ExitCode::Ok`] を返す（TASK-33.2-2）。
///
/// [`write_error_report`] と同じ保証を持つ: 直列化に失敗したら何も書かず `Err`、
/// `write_all` は高々 1 回で、部分書き込み失敗時にリトライ・追記・flush をしない。
///
/// # Errors
/// 直列化エラー、または `out` への書き込み・flush の失敗を `io::Error` として返す。
pub fn write_package_report<W: Write>(out: &mut W, report: &PackageReport) -> io::Result<ExitCode> {
    let mut line = report
        .to_json_line()
        .map_err(|error| io::Error::other(error.to_string()))?;
    line.push('\n');

    out.write_all(line.as_bytes())?;
    out.flush()?;

    Ok(ExitCode::Ok)
}

/// [`EvaluateReport`]（`evaluate` 工程の評価データ未定義 skipped。exit 0）を JSON 1 行＋改行として
/// `out` へ書き、[`ExitCode::Ok`] を返す（REQ-17・REQ-33・TASK-33.3・#140）。
///
/// [`write_package_report`] と同じ保証を持つ（直列化失敗時は何も書かず `Err`、`write_all` は
/// 高々 1 回、部分書き込み失敗時にリトライ・追記・flush をしない）。
///
/// # Errors
/// 直列化エラー、または `out` への書き込み・flush の失敗を `io::Error` として返す。
pub fn write_evaluate_report<W: Write>(
    out: &mut W,
    report: &EvaluateReport,
) -> io::Result<ExitCode> {
    let mut line = report
        .to_json_line()
        .map_err(|error| io::Error::other(error.to_string()))?;
    line.push('\n');

    out.write_all(line.as_bytes())?;
    out.flush()?;

    Ok(ExitCode::Ok)
}

/// [`InferInputError`] を [`ErrorReport`] へ変換する薄い関数。
///
/// 業務ロジックは持たない（`code = err.exit_code()`、
/// `message = err.to_string()`）。`InferInputError::Display` は入力本文・
/// `id` の値・未知キー名を含めないため（`infer_input.rs` のドキュメント
/// 参照）、本関数を経由しても秘密情報の混入防止（security.md）は保たれ
/// る。
#[must_use]
pub fn infer_input_error_report(err: &InferInputError) -> ErrorReport {
    ErrorReport::new(err.exit_code(), err.to_string())
}

/// [`JudgmentError`] を [`ErrorReport`] へ変換する薄い関数
/// （[`infer_input_error_report`] と対称）。
#[must_use]
pub fn judgment_error_report(err: &JudgmentError) -> ErrorReport {
    ErrorReport::new(err.exit_code(), err.to_string())
}

/// [`DefinitionError`] を [`ErrorReport`] へ変換する薄い関数
/// （[`infer_input_error_report`] と対称）。
///
/// `message` には `DefinitionError::Display`（パス・`UnsupportedSchema` の
/// 値・`DuplicateOptionId` の ID 等、秘密情報混入防止の対象になりうる値を
/// 含む内部診断表現）ではなく、[`DefinitionError::public_message`]（バリ
/// アントごとの固定文のみ）を使う（PR #217 レビュー指摘・P0。
/// security.md「秘密情報の混入防止（P0）」）。
#[must_use]
pub fn definition_error_report(err: &DefinitionError) -> ErrorReport {
    ErrorReport::new(err.exit_code(), err.public_message())
}

/// 容量内訳（REQ-30・TASK-30.1-2・#123）を `package` 出力へ埋め込む `capacity` オブジェクト
/// （JSON 文字列。改行なし）へ直列化する。
///
/// 後続の `package` 工程（TASK-33.1。未配線）が出力へ埋め込む部品。`status`・`judgment`・
/// 上限照合（TASK-30.2・#124）はここでは扱わない。手組みで足りるのは、キーが
/// `PackageComponent::as_str()` の ASCII snake_case と固定リテラルだけ、値が整数だけで、
/// エスケープが要らないため（前提はテストで固定）。構成要素は宣言順に 5 件とも常に出す。
#[must_use]
pub fn package_capacity_json(breakdown: &CapacityBreakdown) -> String {
    let components = breakdown
        .entries()
        .into_iter()
        .map(|(component, entry)| {
            format!(
                "\"{}\":{{\"bytes\":{},\"file_count\":{}}}",
                component.as_str(),
                entry.bytes,
                entry.file_count
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"total_bytes\":{},\"components\":{{{}}}}}",
        breakdown.total_bytes(),
        components
    )
}

/// `{"capacity":{...}}` を JSON 1 行＋改行として `out` へ書き、[`ExitCode::Ok`] を返す。
///
/// [`write_ok_judgment`] と同じ保証を持つ: 行全体を先に組み立て、`write_all` は 1 回の
/// 呼び出しにつき高々 1 回、失敗時にリトライ・追記・flush をしない。
///
/// # Errors
/// `out` への書き込み・flush の失敗を `io::Error` として返す。
pub fn write_package_capacity<W: Write>(
    out: &mut W,
    breakdown: &CapacityBreakdown,
) -> io::Result<ExitCode> {
    let mut line = format!("{{\"capacity\":{}}}", package_capacity_json(breakdown));
    line.push('\n');

    out.write_all(line.as_bytes())?;
    out.flush()?;

    Ok(ExitCode::Ok)
}

/// [`CapacityError`] を [`ErrorReport`] へ変換する薄い関数。`message` はパス・io エラー本文を
/// 含まない [`CapacityError::public_message`] を使う（security.md P0）。
#[must_use]
pub fn capacity_error_report(err: &CapacityError) -> ErrorReport {
    ErrorReport::new(err.exit_code(), err.public_message())
}

/// 配布候補の除外記録（REQ-32・TASK-32.2・#114）を `{"excluded":[...]}` へ直列化する
/// （JSON 文字列。改行なし）。
///
/// 後続の `package`・`select` 工程（TASK-33.x。未配線）が出力へ埋め込む部品。値は ASCII の
/// snake_case 固定リテラルと整数だけで、エスケープは要らない（前提は runtime 側のテストで固定）。
/// `detail`・`source`・`load_code`・`mismatched`・`total` は該当する場合だけ出し、キー順は固定。
/// 入力本文・パスは記録に含まれない。
#[must_use]
pub fn export_exclusions_json(records: &[ExclusionRecord]) -> String {
    let items = records
        .iter()
        .map(|r| {
            let mut s = format!(
                "{{\"kind\":\"{}\",\"runtime\":\"{}\",\"format\":\"{}\",\"code\":\"{}\"",
                r.config.kind.as_str(),
                r.config.runtime.as_str(),
                r.config.format.as_str(),
                r.code()
            );
            if let Some(d) = r.detail_code() {
                s.push_str(&format!(",\"detail\":\"{d}\""));
            }
            if let ExclusionReason::PredictionMismatch { mismatched, total } = &r.reason {
                s.push_str(&format!(",\"mismatched\":{mismatched},\"total\":{total}"));
            }
            s.push_str(&format!(",\"evidence\":\"{}\"", r.evidence.as_str()));
            if let Some(src) = r.source() {
                s.push_str(&format!(",\"source\":\"{src}\""));
            }
            s.push('}');
            s
        })
        .collect::<Vec<_>>()
        .join(",");
    format!("{{\"excluded\":[{items}]}}")
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

    // ------------------------------------------------------------------
    // TASK-21.2: write_error_report・エラー変換関数
    // ------------------------------------------------------------------

    /// TASK-21.2: `write_error_report` がバッファに JSON 1 行＋改行のみを
    /// 書き、戻り値が `ExitCode::InvalidInput`（`.code() == 64`）であるこ
    /// と（厳密な文字列一致）。
    #[test]
    fn req21_write_error_report_writes_single_json_line_and_returns_code() {
        let report = fandhe_edge_core::exitcode::ErrorReport::new(
            ExitCode::InvalidInput,
            "missing required field: input",
        );

        let mut buffer: Vec<u8> = Vec::new();
        let exit_code = write_error_report(&mut buffer, &report).unwrap();

        assert_eq!(exit_code, ExitCode::InvalidInput);
        assert_eq!(exit_code.code(), 64);

        let text = String::from_utf8(buffer).unwrap();
        assert_eq!(
            text,
            "{\"code\":\"invalid_input\",\"message\":\"missing required field: input\"}\n"
        );
        assert_eq!(text.matches('\n').count(), 1, "must write exactly one line");
    }

    /// TASK-21.2: 書き込みに失敗する `Write` を渡した場合、`Err` が返るこ
    /// と（`write_ok_judgment` の同種テストと同じ確認）。
    #[test]
    fn req21_write_error_report_propagates_write_failure() {
        struct FailingWriter;
        impl Write for FailingWriter {
            fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
                Err(io::Error::other("simulated write failure"))
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }

        let report = fandhe_edge_core::exitcode::ErrorReport::new(ExitCode::InvalidInput, "boom");
        let mut writer = FailingWriter;
        let outcome = write_error_report(&mut writer, &report);
        assert!(outcome.is_err());
    }

    /// TASK-33.2-2: package の結果は JSON 1 行＋改行で書かれ exit 0 を返し、
    /// 書き込み失敗は `Err` で伝わる（リトライ・追記なし）。
    #[test]
    fn req33_write_package_report_writes_line_and_propagates_failure() {
        let mut buffer: Vec<u8> = Vec::new();
        let code = write_package_report(&mut buffer, &PackageReport::pass()).unwrap();
        assert_eq!(code, ExitCode::Ok);
        assert_eq!(
            String::from_utf8(buffer).unwrap(),
            "{\"step\":\"package\",\"status\":\"ok\",\"judgment\":\"pass\",\"acceptance_defined\":true}\n"
        );

        struct FailingWriter;
        impl Write for FailingWriter {
            fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
                Err(io::Error::other("simulated write failure"))
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        assert!(write_package_report(&mut FailingWriter, &PackageReport::pass()).is_err());
    }

    /// TASK-21.2: 部分書き込み後に失敗する `Write` を渡した場合でも、本
    /// 関数が追加で `write` を呼ばず `flush` を呼ばないこと
    /// （`write_ok_judgment` の同種テストと同じ確認）。
    #[test]
    fn req21_write_error_report_stops_after_partial_write_failure_without_flush() {
        struct PartialThenFailingWriter {
            written: Vec<u8>,
            write_calls: usize,
            flush_calls: usize,
        }
        impl Write for PartialThenFailingWriter {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                self.write_calls += 1;
                if self.write_calls == 1 {
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

        let report = fandhe_edge_core::exitcode::ErrorReport::new(ExitCode::InvalidInput, "boom");
        let mut writer = PartialThenFailingWriter {
            written: Vec::new(),
            write_calls: 0,
            flush_calls: 0,
        };
        let outcome = write_error_report(&mut writer, &report);

        assert!(outcome.is_err());
        assert_eq!(writer.write_calls, 2);
        assert_eq!(writer.flush_calls, 0);
        assert_eq!(writer.written.len(), 4);
    }

    /// TASK-21.2: `infer_input_error_report` が `code`／`message` を
    /// `InferInputError` から正しく写すこと。
    #[test]
    fn req21_infer_input_error_report_maps_code_and_message() {
        let err = InferInputError::EmptyId;
        let report = infer_input_error_report(&err);
        assert_eq!(report.code, ExitCode::InvalidInput);
        assert_eq!(report.message, err.to_string());
    }

    /// TASK-21.2: `judgment_error_report` が `code`／`message` を
    /// `JudgmentError` から正しく写すこと。
    #[test]
    fn req21_judgment_error_report_maps_code_and_message() {
        let err = JudgmentError::EmptyOptions;
        let report = judgment_error_report(&err);
        assert_eq!(report.code, ExitCode::InvalidInput);
        assert_eq!(report.message, err.to_string());
    }

    /// TASK-21.2: `definition_error_report` が `code`／`message` を
    /// `DefinitionError` から正しく写すこと（`message` は
    /// `DefinitionError::public_message`。PR #217 レビュー指摘後は
    /// `Display` と一致しないため、`Display` との比較はしない）。
    #[test]
    fn req21_definition_error_report_maps_code_and_message() {
        let err = DefinitionError::EmptyOptions;
        let report = definition_error_report(&err);
        assert_eq!(report.code, ExitCode::InvalidInput);
        assert_eq!(report.message, err.public_message());
    }

    /// security.md「秘密情報の混入防止（P0）」: `definition_error_report`
    /// の `message` に、`DefinitionError::Display` が含みうるパス・利用者
    /// 指定値（`UnsupportedSchema` のスキーマ値・`DuplicateOptionId` の
    /// ID）が混入しないこと（PR #217 レビュー指摘）。
    #[test]
    fn req21_definition_error_report_does_not_leak_path_or_user_values() {
        let secret_path = "/home/alice/.ssh/secret-definition.json";
        let secret_schema = "sk-test-dummy-schema-marker";
        let secret_id = "row-secret-option-id-marker";

        let read_err = DefinitionError::Read {
            path: std::path::PathBuf::from(secret_path),
            source: io::Error::other("boom"),
        };
        let schema_err = DefinitionError::UnsupportedSchema {
            schema: secret_schema.to_string(),
        };
        let dup_err = DefinitionError::DuplicateOptionId {
            id: secret_id.to_string(),
        };
        let not_regular_err = DefinitionError::NotRegularFile {
            path: std::path::PathBuf::from(secret_path),
        };

        for err in [&read_err, &schema_err, &dup_err, &not_regular_err] {
            // 前提確認: `Display` は実際に秘匿すべき値を含む（本テストが
            // 意味のある回帰検知になっていることの確認）。
            let display = err.to_string();
            assert!(
                display.contains(secret_path)
                    || display.contains(secret_schema)
                    || display.contains(secret_id),
                "Display のフィクスチャ想定が崩れている（テストが無意味化していないか確認）: {display}"
            );

            let report = definition_error_report(err);
            assert!(!report.message.contains(secret_path));
            assert!(!report.message.contains(secret_schema));
            assert!(!report.message.contains(secret_id));

            let line = report.to_json_line().unwrap();
            assert!(!line.contains(secret_path));
            assert!(!line.contains(secret_schema));
            assert!(!line.contains(secret_id));
        }
    }

    // ------------------------------------------------------------------
    // TASK-30.1-2: 容量内訳の JSON 出力（REQ-30・#123）
    // ------------------------------------------------------------------

    use fandhe_edge_runtime::capacity::PackageComponent;

    const EXPECTED_CAPACITY: &str = concat!(
        "{\"capacity\":{\"total_bytes\":1549,\"components\":{",
        "\"weights\":{\"bytes\":1000,\"file_count\":1},",
        "\"vocab_or_feature_transform\":{\"bytes\":0,\"file_count\":0},",
        "\"label_table\":{\"bytes\":37,\"file_count\":1},",
        "\"calibration\":{\"bytes\":0,\"file_count\":0},",
        "\"metadata\":{\"bytes\":512,\"file_count\":1}}}}\n"
    );

    fn sample_breakdown() -> CapacityBreakdown {
        CapacityBreakdown::from_sizes([
            (PackageComponent::Weights, 1000),
            (PackageComponent::LabelTable, 37),
            (PackageComponent::Metadata, 512),
        ])
        .unwrap()
    }

    #[test]
    fn req30_write_package_capacity_exact_output() {
        let mut buf = Vec::new();
        let code = write_package_capacity(&mut buf, &sample_breakdown()).unwrap();
        assert_eq!(code.code(), 0);
        let text = String::from_utf8(buf).unwrap();
        assert_eq!(text, EXPECTED_CAPACITY);
        assert_eq!(text.matches('\n').count(), 1);
    }

    #[test]
    fn req30_zero_byte_file_vs_absent_and_summed() {
        let b = CapacityBreakdown::from_sizes([
            (PackageComponent::Weights, 300),
            (PackageComponent::Weights, 200),
            (PackageComponent::Calibration, 0),
        ])
        .unwrap();
        let json = package_capacity_json(&b);
        assert!(
            json.contains("\"weights\":{\"bytes\":500,\"file_count\":2}"),
            "{json}"
        );
        assert!(
            json.contains("\"calibration\":{\"bytes\":0,\"file_count\":1}"),
            "{json}"
        );
        assert!(
            json.contains("\"metadata\":{\"bytes\":0,\"file_count\":0}"),
            "{json}"
        );
        assert!(json.starts_with("{\"total_bytes\":500,"), "{json}");
    }

    #[test]
    fn req30_component_names_need_no_json_escaping() {
        for c in PackageComponent::all() {
            assert!(
                c.as_str()
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b == b'_'),
                "{}",
                c.as_str()
            );
        }
    }

    #[test]
    fn req30_write_package_capacity_propagates_write_failure() {
        struct Failing;
        impl Write for Failing {
            fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
                Err(io::Error::other("simulated"))
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        assert!(write_package_capacity(&mut Failing, &sample_breakdown()).is_err());
    }

    #[test]
    fn req30_write_package_capacity_stops_after_partial_write_failure() {
        struct PartialThenFailing {
            written: Vec<u8>,
            write_calls: usize,
            flush_calls: usize,
        }
        impl Write for PartialThenFailing {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                self.write_calls += 1;
                if self.write_calls == 1 {
                    let n = buf.len().min(4);
                    self.written
                        .extend_from_slice(buf.get(..n).unwrap_or_default());
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
        let mut w = PartialThenFailing {
            written: Vec::new(),
            write_calls: 0,
            flush_calls: 0,
        };
        assert!(write_package_capacity(&mut w, &sample_breakdown()).is_err());
        assert_eq!(w.write_calls, 2);
        assert_eq!(w.flush_calls, 0);
        assert_eq!(w.written.len(), 4);
    }

    #[test]
    fn req30_capacity_error_report_maps_code_and_message_without_leak() {
        let err = CapacityError::DuplicatePath {
            path: std::path::PathBuf::from("/home/alice/secret-marker.onnx"),
        };
        let report = capacity_error_report(&err);
        assert_eq!(report.code.code(), 64);
        assert_eq!(report.message, "package file is listed more than once");
        let line = report.to_json_line().unwrap();
        assert!(!line.contains("secret-marker"), "{line}");
        assert_eq!(
            capacity_error_report(&CapacityError::Overflow).code.code(),
            20
        );
    }

    fn export_cfg(
        kind: fandhe_edge_runtime::onnx::ModelKind,
        runtime: fandhe_edge_runtime::export_exclusion::InferenceRuntime,
        format: fandhe_edge_runtime::export_exclusion::NumericFormat,
    ) -> fandhe_edge_runtime::export_exclusion::ExportConfig {
        fandhe_edge_runtime::export_exclusion::ExportConfig {
            kind,
            runtime,
            format,
        }
    }

    /// REQ-32: 除外記録の JSON は完全一致で固定する（空・既知制約・予測ずれ・読み込み拒否）。
    #[test]
    fn req32_export_exclusions_json_exact() {
        use fandhe_edge_runtime::export_exclusion::{
            EvidenceKind, InferenceRuntime as Rt, KNOWN_INFEASIBLE, NumericFormat as Nf,
        };
        use fandhe_edge_runtime::onnx::ModelKind;
        assert_eq!(export_exclusions_json(&[]), "{\"excluded\":[]}");
        let known = ExclusionRecord {
            config: export_cfg(ModelKind::C3, Rt::Tract, Nf::Int8Dynamic),
            reason: ExclusionReason::ExportInfeasible(&KNOWN_INFEASIBLE[0]),
            evidence: EvidenceKind::Measured,
        };
        let mismatch = ExclusionRecord {
            config: export_cfg(ModelKind::C1, Rt::Own, Nf::F32),
            reason: ExclusionReason::PredictionMismatch {
                mismatched: 1,
                total: 120,
            },
            evidence: EvidenceKind::TestHarness,
        };
        let rejected = ExclusionRecord {
            config: export_cfg(ModelKind::C3, Rt::Own, Nf::F32),
            reason: ExclusionReason::RuntimeRejectedModel {
                load_code: "unsupported_graph",
            },
            evidence: EvidenceKind::TestHarness,
        };
        assert_eq!(
            export_exclusions_json(&[known, mismatch, rejected]),
            concat!(
                "{\"excluded\":[",
                "{\"kind\":\"c3\",\"runtime\":\"tract\",\"format\":\"int8_dynamic\",",
                "\"code\":\"export_infeasible\",\"detail\":\"type_unification_failed\",",
                "\"evidence\":\"measured\",\"source\":\"PoC-14\"},",
                "{\"kind\":\"c1\",\"runtime\":\"own\",\"format\":\"f32\",",
                "\"code\":\"prediction_mismatch\",\"mismatched\":1,\"total\":120,",
                "\"evidence\":\"test_harness\"},",
                "{\"kind\":\"c3\",\"runtime\":\"own\",\"format\":\"f32\",",
                "\"code\":\"runtime_rejected_model\",\"detail\":\"unsupported_graph\",",
                "\"evidence\":\"test_harness\"}]}"
            )
        );
    }
}
