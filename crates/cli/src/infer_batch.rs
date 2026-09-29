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
//! - 失敗: 全件の解析・検証・推論・`JudgmentResult` 構築の検証を書き込み前に終えるため、結果行は
//!   1 行も出さず、`ErrorReport`（`{"code","message"}`）を 1 行だけ出す。失敗が複数あれば
//!   入力順で最初のものを採る（決定的）
//! - `JudgmentResult` は検証の段階では作って即捨て、書き込みの段階で 1 件ずつ作り直して
//!   書く。結果行を全件保持しないため、選択肢 ID が結果ごとに複製されて大きくなるメモリ消費
//!   （選択肢 ID の合計長 × 件数）が入力に比例して膨らまない。保持するのは
//!   `Prediction`（スコア総数に runtime 側の上限あり）と検証済みレコードのみ
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
//! 処理時間は、入力の読み取り開始から出力開始までを 1 つの期限（`MAX_INFER_BATCH_DURATION`）で
//! 数える。読み取り・推論・出力量の検証は専用スレッドで行い、呼び出し側が期限で待つため、
//! 改行が来ない低速な `Read` や 1 件の推論の内部でブロックしても、超過は `limit_exceeded`
//! で返る（スレッドは強制終了できず切り離して残る。CLI は 1 呼び出し 1 プロセスで、結果を
//! 書いたら終了する前提）。協調的な確認（1 行ごと・1 件ごと）も併用する。
//!
//! 総出力量は書き込み前に全行の長さを合計して `MAX_INFER_BATCH_OUTPUT_BYTES` で拒否する。
//! 出力段階は行の間で `MAX_INFER_BATCH_OUTPUT_DURATION` を確認し、超過は書き込み失敗と同じ
//! `io::Error`（出力が壊れるため `ErrorReport` は追記しない）。1 回の `write` 自体が読み手の
//! 停止でブロックする場合は中断できないが、量は上限内に有界。
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
    InferencePipeline, MAX_INFER_BATCH_DURATION, MAX_INFER_BATCH_LEN, MAX_INFER_BATCH_TOTAL_BYTES,
    Prediction, Preprocessor, ScoringBackend,
};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::{Duration, Instant};

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

fn deadline_passed(deadline: Option<Instant>) -> bool {
    deadline.is_some_and(|d| Instant::now() >= d)
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
    read_batch_records_until(
        reader,
        io,
        Instant::now().checked_add(MAX_INFER_BATCH_DURATION),
    )
}

/// [`read_batch_records`] の期限を指定できる版。1 行読むごとに期限を確認し、超過なら
/// `limit_exceeded`（REQ-39）。`deadline` が `None` なら期限なし。
///
/// # Errors
/// [`read_batch_records`] と同じ。
pub fn read_batch_records_until<R: Read>(
    reader: R,
    io: &IoSchema,
    deadline: Option<Instant>,
) -> Result<Vec<InferInput>, ErrorReport> {
    read_batch_records_with_limit(reader, io, MAX_INFER_BATCH_TOTAL_BYTES, deadline)
}

/// [`read_batch_records`] のバイト上限を指定できる版（上限の境界テスト用）。
fn read_batch_records_with_limit<R: Read>(
    reader: R,
    io: &IoSchema,
    byte_limit: usize,
    deadline: Option<Instant>,
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
        if deadline_passed(deadline) {
            return Err(report(ExitCode::LimitExceeded));
        }
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

/// バッチ全体の総出力バイト数の上限（暫定。REQ-39）。
///
/// 入力は最大 [`MAX_INFER_BATCH_LEN`] 件で、選択肢 ID は 1 件あたり最大 64 KiB になりうるため、
/// 出力は入力より桁違いに大きくなりえる。全行の長さを書き込み前に合計してこの値で拒否し、
/// 出力量（と、その書き込みに要する時間）を有界にする。
pub const MAX_INFER_BATCH_OUTPUT_BYTES: usize = 256 * 1024 * 1024;

/// 出力段階（結果行の書き込み）の時間上限（暫定。REQ-39）。行と行の間で確認する。
pub const MAX_INFER_BATCH_OUTPUT_DURATION: Duration = Duration::from_secs(60);

/// [`emit_infer_batch_with_limits`] の資源上限。既定は本モジュールの定数（REQ-39）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchLimits {
    /// 読み取り開始から出力開始までの時間上限。
    pub duration: Duration,
    /// 総出力バイト数の上限。
    pub output_bytes: usize,
    /// 出力段階の時間上限。
    pub output_duration: Duration,
}

impl Default for BatchLimits {
    fn default() -> Self {
        Self {
            duration: MAX_INFER_BATCH_DURATION,
            output_bytes: MAX_INFER_BATCH_OUTPUT_BYTES,
            output_duration: MAX_INFER_BATCH_OUTPUT_DURATION,
        }
    }
}

/// 全レコードを `infer_batch_until` で推論し、各予測が `JudgmentResult` として組み立てられる
/// ことを検証したうえで、入力順の予測を返す（書き込みはしない）。
///
/// `JudgmentResult` は検証のためだけに作って捨てる。全件を保持すると選択肢 ID が結果ごとに
/// 複製されメモリが入力に比例して膨らむため、書き込み側（[`emit_infer_batch`]）が 1 件ずつ
/// 作り直す（REQ-39）。検証時に各行の直列化後の長さ（改行込み）を合計し、`output_byte_limit` を
/// 超えたら `limit_exceeded` とする（書き込み前に総出力量を拒否するため。REQ-39）。
///
/// # Errors
/// バッチ全体の失敗（期限超過を含む）、または入力順で最初の 1 件の失敗を `ErrorReport` で返す。
pub fn predict_batch<P: Preprocessor, B: ScoringBackend>(
    pipeline: &InferencePipeline<P, B>,
    options: &[Choice],
    records: &[InferInput],
    deadline: Option<Instant>,
    output_byte_limit: usize,
) -> Result<Vec<Prediction>, ErrorReport> {
    // 推論側へ渡すのは input のみ（id・ラベル・分割情報は渡さない）。
    let inputs: Vec<&str> = records.iter().map(InferInput::input).collect();
    let predictions = pipeline
        .infer_batch_until(&inputs, deadline)
        .map_err(|error| error.to_error_report())?;
    if predictions.len() != records.len() {
        return Err(report(ExitCode::RuntimeError));
    }
    let mut validated = Vec::with_capacity(records.len());
    let mut output_bytes: usize = 0;
    for (record, prediction) in records.iter().zip(predictions) {
        let prediction = prediction.map_err(|error| error.to_error_report())?;
        let result = judgment_from_prediction(options, record.id(), &prediction)?;
        let line = result
            .to_json_line()
            .map_err(|_| report(ExitCode::RuntimeError))?;
        // 改行 1 バイトを加える。
        output_bytes = output_bytes.saturating_add(line.len()).saturating_add(1);
        if output_bytes > output_byte_limit {
            return Err(report(ExitCode::LimitExceeded));
        }
        validated.push(prediction);
    }
    Ok(validated)
}

/// 読み取り・推論・出力量の検証までを行う計算段階。書き込みはしない。
fn compute_batch<R: Read, P: Preprocessor, B: ScoringBackend>(
    reader: R,
    io: &IoSchema,
    options: &[Choice],
    pipeline: &InferencePipeline<P, B>,
    deadline: Option<Instant>,
    output_byte_limit: usize,
) -> Result<(Vec<InferInput>, Vec<Prediction>), ErrorReport> {
    let records = read_batch_records_until(reader, io, deadline)?;
    let predictions = predict_batch(pipeline, options, &records, deadline, output_byte_limit)?;
    // 出力の途中で打ち切ると出力が壊れるため、書き始める前にだけ期限を確認する。
    if deadline_passed(deadline) {
        return Err(report(ExitCode::LimitExceeded));
    }
    Ok((records, predictions))
}

/// 入力を読み・推論し、成功なら結果を 1 行 1 JSON で、失敗なら `ErrorReport` を 1 行で書く。
/// 資源上限は [`BatchLimits::default`]。
///
/// # Errors
/// [`emit_infer_batch_with_limits`] と同じ。
pub fn emit_infer_batch<W, R, P, B>(
    out: &mut W,
    reader: R,
    io: &IoSchema,
    options: &[Choice],
    pipeline: Arc<InferencePipeline<P, B>>,
) -> io::Result<ExitCode>
where
    W: Write,
    R: Read + Send + 'static,
    P: Preprocessor + Send + Sync + 'static,
    B: ScoringBackend + Send + Sync + 'static,
{
    emit_infer_batch_with_limits(out, reader, io, options, pipeline, BatchLimits::default())
}

/// [`emit_infer_batch`] の資源上限を指定できる版（REQ-39）。
///
/// 読み取り・推論・出力量の検証は専用スレッドで行い、呼び出し側は `limits.duration` で
/// `recv_timeout` する。改行が来ない低速な `Read` や、期限を確認できない 1 件の推論の途中でも、
/// 期限超過で `limit_exceeded` を返せる（スレッドは強制終了できないため、超過時は切り離して
/// 残す。呼び出し側は結果を書いたらプロセスを終了する前提で、CLI は 1 呼び出し 1 プロセス）。
/// 書き込みは総量（`limits.output_bytes`）を事前に検査済みで、行の間で `limits.output_duration`
/// を確認する。ただし 1 回の `write` 自体が読み手の停止でブロックする場合は中断できない
/// （OS のパイプ背圧。出力量が有界なので待つのは最大でも上限内の量）。
///
/// # Errors
/// 書き込み・flush の失敗、および出力段階の期限超過（`ErrorKind::TimedOut`）を `io::Error` で
/// 返す。部分書き込み後は出力が壊れているため、追加の書き込み（残りの行・`ErrorReport`）は
/// せず即座に打ち切る（`output` の契約）。呼び出し側は `Err` を exit 70 に写し、何も書かない。
pub fn emit_infer_batch_with_limits<W, R, P, B>(
    out: &mut W,
    reader: R,
    io: &IoSchema,
    options: &[Choice],
    pipeline: Arc<InferencePipeline<P, B>>,
    limits: BatchLimits,
) -> io::Result<ExitCode>
where
    W: Write,
    R: Read + Send + 'static,
    P: Preprocessor + Send + Sync + 'static,
    B: ScoringBackend + Send + Sync + 'static,
{
    // 読み取り開始から出力開始までを 1 つの期限で数える（REQ-39）。
    let start = Instant::now();
    let deadline = start.checked_add(limits.duration);
    let (tx, rx) = mpsc::channel();
    let worker_io = io.clone();
    let worker_options = options.to_vec();
    let worker_pipeline = Arc::clone(&pipeline);
    let output_bytes = limits.output_bytes;
    let spawned = thread::Builder::new()
        .name("infer-batch".to_string())
        .spawn(move || {
            let outcome = compute_batch(
                reader,
                &worker_io,
                &worker_options,
                &worker_pipeline,
                deadline,
                output_bytes,
            );
            // 受信側が期限で去っていれば送信は失敗する。捨てる。
            let _ = tx.send(outcome);
        });
    let outcome = match spawned {
        Err(_) => Err(report(ExitCode::RuntimeError)),
        Ok(_) => {
            let received = match deadline {
                Some(d) => rx.recv_timeout(d.saturating_duration_since(Instant::now())),
                None => rx.recv().map_err(|_| mpsc::RecvTimeoutError::Disconnected),
            };
            match received {
                Ok(outcome) => outcome,
                Err(mpsc::RecvTimeoutError::Timeout) => Err(report(ExitCode::LimitExceeded)),
                // 計算スレッドの panic。
                Err(mpsc::RecvTimeoutError::Disconnected) => Err(report(ExitCode::RuntimeError)),
            }
        }
    };
    match outcome {
        Ok((records, predictions)) => {
            let output_deadline = Instant::now().checked_add(limits.output_duration);
            for (record, prediction) in records.iter().zip(&predictions) {
                if deadline_passed(output_deadline) {
                    return Err(io::Error::new(
                        io::ErrorKind::TimedOut,
                        "infer batch output deadline exceeded",
                    ));
                }
                // predict_batch で検証済みのため、ここでの再構築は失敗しない想定。
                // 万一失敗しても部分出力のまま続けず、書き込み失敗と同じく打ち切る。
                let result = judgment_from_prediction(options, record.id(), prediction)
                    .map_err(|error| io::Error::other(error.message))?;
                write_ok_judgment(out, &result)?;
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
        let ok = read_batch_records_with_limit(line.as_bytes(), &io_schema(), exact, None).unwrap();
        assert_eq!(ok.len(), 1);
        let err = read_batch_records_with_limit(line.as_bytes(), &io_schema(), exact - 1, None)
            .unwrap_err();
        assert_eq!(err.code, ExitCode::LimitExceeded);
    }

    /// REQ-39: 総量の上限が次の JSON 行の途中に当たっても、断片を解析せず `limit_exceeded`
    /// （`invalid_input` にしない）。
    #[test]
    fn req39_limit_inside_next_line_is_limit_exceeded_not_invalid_input() {
        let first = "{\"id\":\"a\",\"input\":\"x\"}\n";
        let input = format!("{first}{first}");
        // 上限を 2 行目の途中（不完全な JSON になる位置）に置く。
        let limit = first.len() + 5;
        let err =
            read_batch_records_with_limit(input.as_bytes(), &io_schema(), limit, None).unwrap_err();
        assert_eq!(err.code, ExitCode::LimitExceeded);
    }

    /// REQ-39: 読み取り開始前に期限が過ぎていれば `limit_exceeded`（読み取りも期限の対象）。
    #[test]
    fn req39_read_deadline_is_enforced() {
        let line = "{\"id\":\"a\",\"input\":\"x\"}\n";
        let err = read_batch_records_until(line.as_bytes(), &io_schema(), Some(Instant::now()))
            .unwrap_err();
        assert_eq!(err.code, ExitCode::LimitExceeded);
    }
}
