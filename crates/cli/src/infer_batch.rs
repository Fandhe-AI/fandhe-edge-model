//! `infer --input-file` の一括推論と、その出力形（1 行 1 JSON）の例外を担う CLI 層の
//! モジュール（REQ-33・REQ-21・REQ-39・TASK-33.4・#141）。
//!
//! # 呼び出し文脈
//!
//! CLI の `infer` 工程が `--input-file` の入力を開き、[`emit_infer_batch`] へ `Read` として
//! 渡す想定（`main.rs` への配線は TASK-33.1-2・#136。入力ファイルを開く際の経路の閉じ込めも
//! guard 層の責務で、ここではパスを扱わない）。推論は `fandhe-edge-runtime` の
//! `InferencePipeline::infer_batch` に委ね、単体推論と同じ 1 系列の経路を通す（REQ-28）。
//! 推論関数へ渡すのは各レコードの `input` だけで、`id` は出力の組み立てにのみ使う
//! （評価契約「推論関数には input だけを渡す」）。
//!
//! # 出力契約
//!
//! REQ-33 は「1 呼び出しにつき JSON 1 つ」を原則とし、唯一の例外が `infer --input-file` の
//! 1 行 1 JSON である（[`OutputMode::JsonLines`]。PoC-16 事前登録規約）。JSON 以外を stdout へ
//! 混ぜない制約は例外でも維持する。新しいスキーマは作らない。
//!
//! - 成功（exit 0）: レコードごとに `JudgmentResult` の 1 行を入力順に出す（形は `--text` と同一）
//! - 失敗: 全件の解析・検証・推論・`JudgmentResult` 構築を書き込み前に終えるため、結果行は
//!   1 行も出さず、`ErrorReport`（`{"code","message"}`）を 1 行だけ出す。失敗が複数あれば
//!   入力順で最初のものを採る（決定的）
//! - 空行・空白のみの行は読み飛ばす。有効レコードが 0 件なら `invalid_input`（成功を装わない）
//!
//! # 実装しないもの（入出力契約の変更・後続作業。REQ-21・REQ-33）
//!
//! - PoC-16 の行単位エラー行（`status:"error"`）: `JudgmentStatus` の variant 追加を伴うため未実装
//! - `--out`（行をファイルへ書き stdout に要約を出す）: 要約スキーマ未確定・書き込み先の
//!   ガード（REQ-39）が要るため未実装。出力関数は `Write` に汎用化してあり、後続で差し替えられる
//!
//! # 資源上限（REQ-39・暫定）
//!
//! 入力は 1 行ずつ上限付きで読む（`read_until` を 1 行 `MAX_INFER_INPUT_BYTES + 2` バイトで
//! 打ち切る）ため、バッファは高々 1 行分で、入力全体をメモリへ載せない。入力全体のバイト数は
//! `MAX_INFER_BATCH_TOTAL_BYTES + 1` で読み取りを打ち切る。各レコードの `input` の復号後バイト長は
//! その JSON 行以下なので、ファイルが上限内なら総入力バイトも上限内（JSON のオーバーヘッド分だけ
//! 保守側）。件数は `MAX_INFER_BATCH_LEN`、1 行は `MAX_INFER_INPUT_BYTES` を、レコードを保持する
//! 前に検査する。
//!
//! 証拠種別: テストハーネス（バイナリでの完走は #136、実バックエンドは #112/#113）。

use crate::args::{Command, InferSource};
use crate::error_report::{ToErrorReport, default_message, emit_error_report};
use crate::output::{infer_input_error_report, judgment_error_report, write_ok_judgment};
use fandhe_edge_core::definition::{Choice, IoSchema};
use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};
use fandhe_edge_core::infer_input::{InferInput, MAX_INFER_INPUT_BYTES};
use fandhe_edge_core::judgment::JudgmentResult;
use fandhe_edge_runtime::pipeline::{
    InferencePipeline, MAX_INFER_BATCH_LEN, MAX_INFER_BATCH_TOTAL_BYTES, Prediction, Preprocessor,
    ScoringBackend,
};
use std::io::{self, BufRead, BufReader, Read, Write};

/// stdout の出力形（REQ-33）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputMode {
    /// 1 呼び出しにつき JSON 1 つ（原則）。
    SingleDocument,
    /// 1 行 1 JSON（`infer --input-file` の例外）。
    JsonLines,
}

/// コマンドの stdout 出力形を返す。
///
/// 全 variant を網羅 `match` し、コマンドが増えたらコンパイルエラーで気付けるようにする
/// （REQ-33）。`JsonLines` は `--out` なしの `infer --input-file` のみ。`--out` 付きは
/// 「行はファイル・stdout は単一 JSON」の想定（実処理は後続）で `SingleDocument` とする。
#[must_use]
pub const fn output_mode(command: &Command) -> OutputMode {
    match command {
        Command::Infer(args) => match &args.source {
            InferSource::InputFile { out: None, .. } => OutputMode::JsonLines,
            InferSource::InputFile { out: Some(_), .. } | InferSource::Text { .. } => {
                OutputMode::SingleDocument
            }
        },
        Command::Register(_)
        | Command::Inspect(_)
        | Command::Train(_)
        | Command::Evaluate(_)
        | Command::Select(_)
        | Command::Package(_) => OutputMode::SingleDocument,
    }
}

fn report(code: ExitCode) -> ErrorReport {
    ErrorReport::new(code, default_message(code))
}

/// `reader` から 1 行 1 JSON の推論入力を読み、検証済みレコードを入力順に返す。
///
/// 上限は [`MAX_INFER_BATCH_TOTAL_BYTES`]。
///
/// # Errors
/// 上限超過は `limit_exceeded`、非 UTF-8・不正レコード・有効 0 件は `invalid_input`、
/// 読み取り失敗は `runtime_error`。message は固定語彙でデータを含まない。
pub fn read_batch_records<R: Read>(
    reader: R,
    io: &IoSchema,
) -> Result<Vec<InferInput>, ErrorReport> {
    read_batch_records_with_limit(reader, io, MAX_INFER_BATCH_TOTAL_BYTES)
}

/// [`read_batch_records`] のバイト上限を指定できる版（上限の境界テスト用）。
fn read_batch_records_with_limit<R: Read>(
    reader: R,
    io: &IoSchema,
    byte_limit: usize,
) -> Result<Vec<InferInput>, ErrorReport> {
    // 入力全体は上限 + 1 バイトで打ち切る（無制限の読み取りの防止。REQ-39）。
    let cap = u64::try_from(byte_limit)
        .unwrap_or(u64::MAX)
        .saturating_add(1);
    let mut reader = BufReader::new(reader.take(cap));
    let mut consumed: usize = 0;
    let mut records = Vec::new();
    // 1 行ずつ、行長の上限 + 1 バイトまでしかバッファへ確保しない。巨大な単一行・空白行のみの
    // 入力でも、拒否前にファイル全体をメモリへ保持しない（REQ-39）。
    loop {
        let (line_bytes, has_newline) = read_bounded_line(&mut reader)?;
        if line_bytes.is_empty() && !has_newline {
            break;
        }
        consumed = consumed.saturating_add(line_bytes.len());
        if consumed > byte_limit {
            return Err(report(ExitCode::LimitExceeded));
        }
        let line_bytes = strip_line_ending(&line_bytes);
        if line_bytes.len() > MAX_INFER_INPUT_BYTES {
            return Err(report(ExitCode::LimitExceeded));
        }
        let line = std::str::from_utf8(line_bytes).map_err(|_| report(ExitCode::InvalidInput))?;
        if line.trim().is_empty() {
            continue;
        }
        if records.len() >= MAX_INFER_BATCH_LEN {
            return Err(report(ExitCode::LimitExceeded));
        }
        let record =
            InferInput::parse(line, io).map_err(|error| infer_input_error_report(&error))?;
        records.push(record);
    }
    if records.is_empty() {
        return Err(report(ExitCode::InvalidInput));
    }
    Ok(records)
}

/// 1 行（改行を含む）を最大 `MAX_INFER_INPUT_BYTES + 2` バイトまで読む。
///
/// 戻り値の bool は改行で終わったか。上限までに改行が無ければ超過行として、そこで打ち切った
/// 断片を返す（呼び出し側が長さで `limit_exceeded` にする。残りは読まない）。
fn read_bounded_line<R: BufRead>(reader: &mut R) -> Result<(Vec<u8>, bool), ErrorReport> {
    // 改行（CRLF なら 2 バイト）を含めて 1 行の上限 + 2 バイト。
    let line_cap = u64::try_from(MAX_INFER_INPUT_BYTES)
        .unwrap_or(u64::MAX)
        .saturating_add(2);
    let mut buf = Vec::new();
    reader
        .by_ref()
        .take(line_cap)
        .read_until(b'\n', &mut buf)
        .map_err(|_| report(ExitCode::RuntimeError))?;
    let has_newline = buf.last() == Some(&b'\n');
    Ok((buf, has_newline))
}

/// 行末の `\n` / `\r\n` を除く（`str::lines` と同じ扱い）。
fn strip_line_ending(line: &[u8]) -> &[u8] {
    let line = line.strip_suffix(b"\n").unwrap_or(line);
    line.strip_suffix(b"\r").unwrap_or(line)
}

/// 予測（選択肢 index とスコア）を `JudgmentResult` へ写す。`--text` 経路と共有する想定。
///
/// # Errors
/// index が選択肢の範囲外なら `runtime_error`。`JudgmentResult::new` の検証エラーはその
/// 終了コードへ写す。
pub fn judgment_from_prediction(
    options: &[Choice],
    id: &str,
    prediction: &Prediction,
) -> Result<JudgmentResult, ErrorReport> {
    let choice = options
        .get(prediction.label_index())
        .ok_or_else(|| report(ExitCode::RuntimeError))?;
    JudgmentResult::new(options, id, &choice.id, prediction.scores())
        .map_err(|error| judgment_error_report(&error))
}

/// 全レコードを `infer_batch` で推論し、入力順の `JudgmentResult` を作る（書き込みはしない）。
///
/// # Errors
/// バッチ全体の失敗、または入力順で最初の 1 件の失敗を `ErrorReport` で返す。
pub fn run_infer_batch<P: Preprocessor, B: ScoringBackend>(
    pipeline: &InferencePipeline<P, B>,
    options: &[Choice],
    records: &[InferInput],
) -> Result<Vec<JudgmentResult>, ErrorReport> {
    // 推論側へ渡すのは input のみ（id・ラベル・分割情報は渡さない）。
    let inputs: Vec<&str> = records.iter().map(InferInput::input).collect();
    let predictions = pipeline
        .infer_batch(&inputs)
        .map_err(|error| error.to_error_report())?;
    if predictions.len() != records.len() {
        return Err(report(ExitCode::RuntimeError));
    }
    let mut results = Vec::with_capacity(records.len());
    for (record, prediction) in records.iter().zip(predictions) {
        let prediction = prediction.map_err(|error| error.to_error_report())?;
        results.push(judgment_from_prediction(options, record.id(), &prediction)?);
    }
    Ok(results)
}

/// 入力を読み・推論し、成功なら結果を 1 行 1 JSON で、失敗なら `ErrorReport` を 1 行で書く。
///
/// # Errors
/// 書き込み・flush の失敗を `io::Error` で返す。部分書き込み後は出力が壊れているため、
/// 追加の書き込み（残りの行・`ErrorReport`）はせず即座に打ち切る（`output` の契約）。
/// 呼び出し側は `Err` を exit 70 に写し、何も書かない。
pub fn emit_infer_batch<W: Write, R: Read, P: Preprocessor, B: ScoringBackend>(
    out: &mut W,
    reader: R,
    io: &IoSchema,
    options: &[Choice],
    pipeline: &InferencePipeline<P, B>,
) -> io::Result<ExitCode> {
    let outcome = read_batch_records(reader, io)
        .and_then(|records| run_infer_batch(pipeline, options, &records));
    match outcome {
        Ok(results) => {
            for result in &results {
                write_ok_judgment(out, result)?;
            }
            Ok(ExitCode::Ok)
        }
        Err(error) => emit_error_report(out, &error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::{
        EvaluateArgs, InferArgs, InspectArgs, PackageArgs, RegisterArgs, SelectArgs, TrainArgs,
    };
    use fandhe_edge_core::definition::InputRepresentation;
    use std::path::PathBuf;

    fn p() -> PathBuf {
        PathBuf::from("p")
    }

    /// REQ-33: `JsonLines` は `--out` なしの `infer --input-file` のみ。
    #[test]
    fn req33_output_mode_is_single_document_except_infer_input_file() {
        let infer = |source| {
            Command::Infer(InferArgs {
                package: p(),
                source,
            })
        };
        let single = [
            Command::Register(RegisterArgs {
                definition: p(),
                project_dir: p(),
            }),
            Command::Inspect(InspectArgs { project_dir: p() }),
            Command::Train(TrainArgs {
                project_dir: p(),
                candidate: 0,
                smoke: false,
            }),
            Command::Evaluate(EvaluateArgs {
                project_dir: p(),
                candidate: 0,
            }),
            Command::Select(SelectArgs { project_dir: p() }),
            Command::Package(PackageArgs { project_dir: p() }),
            infer(InferSource::Text {
                text: "t".to_string(),
                id: None,
            }),
            infer(InferSource::InputFile {
                path: p(),
                out: Some(p()),
            }),
        ];
        for command in &single {
            assert_eq!(output_mode(command), OutputMode::SingleDocument);
        }
        assert_eq!(
            output_mode(&infer(InferSource::InputFile {
                path: p(),
                out: None
            })),
            OutputMode::JsonLines
        );
    }

    fn io_schema() -> IoSchema {
        IoSchema {
            input: InputRepresentation::Bytes,
        }
    }

    /// REQ-39: 上限ちょうどは受理し、上限 + 1 は `limit_exceeded`（小さい上限で境界を確認）。
    #[test]
    fn req39_byte_limit_boundary() {
        let line = "{\"id\":\"a\",\"input\":\"x\"}";
        let exact = line.len();
        let ok = read_batch_records_with_limit(line.as_bytes(), &io_schema(), exact).unwrap();
        assert_eq!(ok.len(), 1);
        let err =
            read_batch_records_with_limit(line.as_bytes(), &io_schema(), exact - 1).unwrap_err();
        assert_eq!(err.code, ExitCode::LimitExceeded);
    }
}
