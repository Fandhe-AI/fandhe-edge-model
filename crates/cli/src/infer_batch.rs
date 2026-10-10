//! `infer --input-file` の一括推論と、その出力形（1 行 1 JSON）の例外を担う CLI 層の
//! モジュール（REQ-33・REQ-21・REQ-39・TASK-33.4・#141）。
//!
//! # 呼び出し文脈
//!
//! CLI の `infer` 工程が `--input-file` の入力を開き、[`emit_infer_batch`] へ `Read` として
//! 渡す（`stages::infer` から配線済み。#136・#141。入力ファイルを開く際の経路の閉じ込めも
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
//!   入力順で最初のものを採る（決定的。優先規則は `compute_batch` の 1 つのループに集約。
//!   1 レコードずつ読み・推論し、最初の失敗が出た時点で打ち切って即座に受け手へ送る）
//! - `JudgmentResult` は検証の段階では作って即捨て、書き込みの段階で 1 件ずつ作り直して
//!   書く。結果行を全件保持しないため、選択肢 ID が結果ごとに複製されて大きくなるメモリ消費
//!   （選択肢 ID の合計長 × 件数）が入力に比例して膨らまない。保持するのは
//!   `Prediction`（スコア総数に runtime 側の上限あり）と検証済みレコードのみ
//! - 空行・空白のみの行は読み飛ばす。有効レコードが 0 件なら `invalid_input`（成功を装わない）
//! - `--out`（#459。オーナー承認 2026-10-09）: 成功時だけ結果行（上と同一バイト列）を OUT へ書き、
//!   stdout は要約 JSON 1 つ。失敗は `ErrorReport` を stdout へ出し OUT を作らない。結果行と
//!   `ErrorReport` の書き分けは [`emit_infer_batch_split`]、OUT の閉じ込め・遅延作成は
//!   `stages::infer` が担う
//!
//! # 実装しないもの（入出力契約の変更・後続作業。REQ-21・REQ-33）
//!
//! - PoC-16 の行単位エラー行（`status:"error"`）: `JudgmentStatus` の variant 追加を伴うため未実装
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
//! 書いたら終了する前提）。協調的な確認（1 行ごと・1 件ごと）も併用する。CLI 経路は
//! 公開経路は常にプロセス終了で回収する（下の「公開経路は回収を保証するモードだけ」）。
//!
//! 総出力量は書き込み前に全行の長さを合計して `MAX_INFER_BATCH_OUTPUT_BYTES` で拒否する。
//!
//! 時間上限の確定点は計算段階（読み取り・推論・出力量の検証）の 1 つだけで、期限内に全件の
//! 計算が済んだ時点で結果は成功として確定する。計算中の超過は結果行なしの `ErrorReport`
//! （`limit_exceeded`・exit 20）。書き出しと flush の完了後に期限を理由に失敗へ変えない。
//! 書き出しの段階は、CLI 経路のウォッチドッグだけが扱う: 出力先が停止して
//! `MAX_INFER_BATCH_OUTPUT_DURATION` を超えたらプロセスを exit 20（`limit_exceeded`）で終える。
//! 既知の制限: 出力先が停止しているため `ErrorReport` は書けず、途中までの行が残りうる。この
//! 場合は終了コードが唯一の判定根拠になる（書き込みをブロックしたまま待たないため）。
//! ウォッチドッグの起動に失敗したら、上限なしで書かず `io::Error`（exit 70）で終える。
//!
//! # 公開経路は回収を保証するモードだけ（REQ-39）
//!
//! Rust のスレッドは外から止められない。期限超過後は計算スレッドを join せず結果を捨てて戻り
//! （計算スレッドは 1 呼び出しにつき 1 本で、期限超過後に次の計算を起動しない）、停止した
//! スレッドの回収は「エラー JSON を書いて flush した直後の `process::exit`」に頼る。この回収を
//! 保証するため、crate の外から呼べる [`emit_infer_batch`]・[`emit_infer_batch_with_limits`] は
//! 常にプロセス終了で回収するモード（CLI の 1 呼び出し 1 プロセス専用）で動き、回収しない
//! モード（`StallPolicy::Leak`）は crate 内部（`pub(crate)`）とそのユニットテストに閉じ込める。
//! 同様に、止まりうる同期処理（`Read::read` の読み取り・推論）を直接実行する関数
//! （逐次読み取り・`compute_batch`）は crate 内部に限り、公開するのは停止を回収する `emit_*`
//! だけとする（公開関数から直接呼ばれると、`Read::read` が返らない場合に時間上限を守れないため）。
//!
//! 長く動き続けるプロセス（将来の MCP サーバ。REQ-36・REQ-37）から推論する場合は、スレッドを
//! 強制終了できないため、本関数を直接呼ばず CLI を子プロセスとして起動して隔離する前提とする
//! （配線は #136 で `stages::infer` に実装済み）。
//!
//! 証拠種別: テストハーネス（バイナリでの完走は `tests/pipeline_e2e.rs`、実バックエンドは #112/#113）。

use crate::args::{Command, InferSource};
use crate::error_report::{ToErrorReport, default_message, emit_error_report};
use crate::output::{infer_input_error_report, judgment_error_report, write_ok_judgment};
use fandhe_edge_core::definition::{Choice, IoSchema};
use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};
use fandhe_edge_core::infer_input::{InferInput, MAX_INFER_INPUT_BYTES};
use fandhe_edge_core::judgment::JudgmentResult;
use fandhe_edge_eval::abstention::{IndexedDecision, decide_abstention_with_parameters};
use fandhe_edge_runtime::pipeline::{
    InferencePipeline, MAX_INFER_BATCH_DURATION, MAX_INFER_BATCH_LEN, MAX_INFER_BATCH_TOTAL_BYTES,
    MAX_INFER_BATCH_TOTAL_SCORES, Prediction, Preprocessor, ScoringBackend,
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
/// 「行はファイル・stdout は単一 JSON」で `SingleDocument` とする（#459）。
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

/// 1 行 1 JSON の推論入力を、1 レコードずつ上限付きで読む（同期。REQ-33・REQ-39）。
///
/// 計算スレッドが 1 レコード読むごとに推論へ進めるための逐次読み取りで、入力順の最初の失敗が
/// 確定した時点で打ち切れる。`Read::read` が返らない場合の時間上限は保証しない（呼び出し側の
/// 計算スレッドの期限待ちとプロセス終了が強制する。モジュール doc）ため、crate 内部に限る。
struct RecordReader<'a, R: Read> {
    reader: BufReader<io::Take<R>>,
    io: &'a IoSchema,
    byte_limit: usize,
    deadline: Option<Instant>,
    consumed: usize,
    count: usize,
}

impl<'a, R: Read> RecordReader<'a, R> {
    fn new(reader: R, io: &'a IoSchema, byte_limit: usize, deadline: Option<Instant>) -> Self {
        // 入力全体は上限 + 1 バイトで打ち切る（無制限の読み取りの防止。REQ-39）。
        let cap = u64::try_from(byte_limit)
            .unwrap_or(u64::MAX)
            .saturating_add(1);
        Self {
            reader: BufReader::new(reader.take(cap)),
            io,
            byte_limit,
            deadline,
            consumed: 0,
            count: 0,
        }
    }

    /// 次の有効レコード。入力の終わりは `Ok(None)`。空行・空白のみの行は読み飛ばす。
    ///
    /// 1 行ずつ、行長の上限 + 1 バイトまでしかバッファへ確保しない。巨大な単一行・空白行のみの
    /// 入力でも、拒否前にファイル全体をメモリへ保持しない（REQ-39）。
    fn next_record(&mut self) -> Result<Option<InferInput>, ErrorReport> {
        loop {
            if deadline_passed(self.deadline) {
                return Err(report(ExitCode::LimitExceeded));
            }
            let (line_bytes, has_newline) = read_bounded_line(&mut self.reader)?;
            if line_bytes.is_empty() && !has_newline {
                return Ok(None);
            }
            self.consumed = self.consumed.saturating_add(line_bytes.len());
            if self.consumed > self.byte_limit {
                return Err(report(ExitCode::LimitExceeded));
            }
            let line_bytes = strip_line_ending(&line_bytes);
            if line_bytes.len() > MAX_INFER_INPUT_BYTES {
                return Err(report(ExitCode::LimitExceeded));
            }
            let line =
                std::str::from_utf8(line_bytes).map_err(|_| report(ExitCode::InvalidInput))?;
            if line.trim().is_empty() {
                continue;
            }
            if self.count >= MAX_INFER_BATCH_LEN {
                return Err(report(ExitCode::LimitExceeded));
            }
            let record = InferInput::parse(line, self.io)
                .map_err(|error| infer_input_error_report(&error))?;
            self.count += 1;
            return Ok(Some(record));
        }
    }
}

/// 読み取りだけを行って検証済みレコードを返す（バイト上限・期限の境界テスト用）。
#[cfg(test)]
fn read_batch_records_with_limit<R: Read>(
    reader: R,
    io: &IoSchema,
    byte_limit: usize,
    deadline: Option<Instant>,
) -> Result<Vec<InferInput>, ErrorReport> {
    let mut records = RecordReader::new(reader, io, byte_limit, deadline);
    let mut out = Vec::new();
    while let Some(record) = records.next_record()? {
        out.push(record);
    }
    if out.is_empty() {
        return Err(report(ExitCode::InvalidInput));
    }
    Ok(out)
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
    judgment_from_prediction_scoped(options, &JudgmentScope::default(), id, prediction)
}

/// 配布パッケージの校正（`calibration.json` の T・τ。REQ-22・#497）。値の範囲は読み込み時
/// （`stages::infer`）と評価器の判定関数の両方で検証する。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InferCalibration {
    /// 温度 T。
    pub temperature: f64,
    /// 保留しきい値 τ。
    pub threshold: f64,
}

/// `infer` の判定行の `status` を決める指定（定義の `out_of_scope_label` とパッケージの校正。
/// REQ-22・#478・#497）。既定（どちらも無し）は常に `ok`。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct JudgmentScope {
    /// 定義の `out_of_scope_label`。
    pub out_of_scope_label: Option<String>,
    /// パッケージの校正（`calibration.json` が無ければ `None`。保留を返さない）。
    pub calibration: Option<InferCalibration>,
}

/// [`judgment_from_prediction`] に [`JudgmentScope`] を渡す版（`infer` 用。単件・バッチ共通。
/// REQ-22・REQ-28・#478・#497）。
///
/// 優先順は 対象外 > 保留 > ok。校正が無ければ argmax が対象外ラベルのときだけ `out_of_scope`。
/// 校正があれば、スコア（確率 p）から `ln(p)` をロジットとして評価器の
/// [`decide_abstention_with_parameters`] に渡し、`evaluate` の保留判定と同じ規則で区分する
/// （規則は CLI で再実装しない）。行の `scores` は ONNX の生の確率のまま。
///
/// # Errors
/// [`judgment_from_prediction`] と同じ。評価器の判定が失敗する・argmax が食い違う場合は `runtime_error`。
pub fn judgment_from_prediction_scoped(
    options: &[Choice],
    scope: &JudgmentScope,
    id: &str,
    prediction: &Prediction,
) -> Result<JudgmentResult, ErrorReport> {
    let choice = options
        .get(prediction.label_index())
        .ok_or_else(|| report(ExitCode::RuntimeError))?;
    let result = JudgmentResult::new(options, id, &choice.id, prediction.scores())
        .map_err(|error| judgment_error_report(&error))?;
    let out_of_scope_label = scope.out_of_scope_label.as_deref();
    let Some(calibration) = scope.calibration else {
        return Ok(if out_of_scope_label == Some(choice.id.as_str()) {
            result.into_out_of_scope()
        } else {
            result
        });
    };
    let out_of_scope_index =
        out_of_scope_label.and_then(|label| options.iter().position(|c| c.id == label));
    let logits: Vec<f64> = prediction.scores().iter().map(|p| p.ln()).collect();
    let decision = decide_abstention_with_parameters(
        calibration.temperature,
        calibration.threshold,
        out_of_scope_index,
        &logits,
    )
    .map_err(|_| report(ExitCode::RuntimeError))?;
    // 評価器の argmax（同値は宣言順の先頭）と行の `predicted_label` が食い違えば止める（fail-closed）。
    match decision {
        IndexedDecision::Abstain { .. } => Ok(result.into_abstain()),
        IndexedDecision::Adopt { label_index, .. } if label_index == prediction.label_index() => {
            Ok(result)
        }
        IndexedDecision::OutOfScope { label_index, .. }
            if label_index == prediction.label_index() =>
        {
            Ok(result.into_out_of_scope())
        }
        _ => Err(report(ExitCode::RuntimeError)),
    }
}

/// バッチ全体の総出力バイト数の上限（暫定。REQ-39）。
///
/// 入力は最大 [`MAX_INFER_BATCH_LEN`] 件で、選択肢 ID は 1 件あたり最大 64 KiB になりうるため、
/// 出力は入力より桁違いに大きくなりえる。全行の長さを書き込み前に合計してこの値で拒否し、
/// 出力量（と、その書き込みに要する時間）を有界にする。
pub const MAX_INFER_BATCH_OUTPUT_BYTES: usize = fandhe_edge_core::limits::MAX_OUTPUT_BYTES;

/// 出力段階（結果行の書き込み）の時間上限（暫定。REQ-39）。CLI 経路のウォッチドッグが強制する。
pub const MAX_INFER_BATCH_OUTPUT_DURATION: Duration = Duration::from_secs(60);

/// [`emit_infer_batch_with_limits`] の資源上限。既定は本モジュールの定数（REQ-39）。
///
/// 停止時の回収方式（プロセス終了）は公開の型からは変えられない（`StallPolicy` は crate 内部）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchLimits {
    /// 読み取り開始から出力開始までの時間上限。
    pub duration: Duration,
    /// 総出力バイト数の上限。
    pub output_bytes: usize,
    /// 出力段階の時間上限。超過して書き込みがブロックしたら、ウォッチドッグが exit 20
    /// （`limit_exceeded`。ErrorReport は書けない）でプロセスを終える。
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

/// 中断できない停止（出力先への書き込みのブロック・期限超過で切り離した計算スレッド）の回収方式。
/// crate 内部の型で、公開経路は常に [`StallPolicy::TerminateProcess`]（REQ-39）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StallPolicy {
    /// プロセスの終了で回収する（CLI の 1 呼び出し 1 プロセス）。
    TerminateProcess,
    /// 回収しない（切り離したスレッド・停止した書き込みが残りうる）。ユニットテスト専用。
    #[cfg(test)]
    Leak,
}

impl StallPolicy {
    pub(crate) const fn terminates(self) -> bool {
        match self {
            Self::TerminateProcess => true,
            #[cfg(test)]
            Self::Leak => false,
        }
    }
}

/// 出力段階のウォッチドッグ。drop（正常完了・エラー）で解除される。
///
/// `write_all` / `flush` が出力先の停止でブロックしても、協調的な期限確認は戻らないため、
/// 別スレッドが期限で `process::exit` して回収する（REQ-39。CLI の 1 プロセス専用）。
pub(crate) struct OutputWatchdog {
    _disarm: mpsc::Sender<()>,
}

impl OutputWatchdog {
    /// `enabled` なら `duration` 後にプロセスを終了するウォッチドッグを起動する。
    ///
    /// # Errors
    /// スレッドを起動できなければ `Err`。上限なしで続行せず、呼び出し側は fail-closed で
    /// 終える（REQ-39）。
    pub(crate) fn arm(enabled: bool, duration: Duration) -> io::Result<Option<Self>> {
        Self::arm_with(enabled, duration, |task| {
            thread::Builder::new()
                .name("infer-batch-watchdog".to_string())
                .spawn(task)
                .map(|_| ())
        })
    }

    /// 起動手段を差し替えられる版（起動失敗の回帰テスト用）。
    fn arm_with(
        enabled: bool,
        duration: Duration,
        spawn: impl FnOnce(Box<dyn FnOnce() + Send>) -> io::Result<()>,
    ) -> io::Result<Option<Self>> {
        if !enabled {
            return Ok(None);
        }
        let (tx, rx) = mpsc::channel::<()>();
        spawn(Box::new(move || {
            // 送信側の drop（解除）は Disconnected で返る。Timeout のときだけ終了する。
            if let Err(mpsc::RecvTimeoutError::Timeout) = rx.recv_timeout(duration) {
                // 時間上限の超過は経路によらず limit_exceeded（REQ-21・REQ-39）。出力先が停止して
                // いるため ErrorReport は書けず、終了コードが唯一の判定根拠になる。
                std::process::exit(i32::from(ExitCode::LimitExceeded.code()));
            }
        }))?;
        Ok(Some(Self { _disarm: tx }))
    }
}

/// 読み取り・推論・出力量の検証までを 1 レコードずつ入力順に行う計算段階。書き込みはしない。
///
/// 「入力順で最初の失敗」（読み取り・検証・推論・`JudgmentResult` 構築・出力量・バッチ全体の
/// 上限）が出た時点でそれが確定するため、即座に打ち切って `Err` を返す（呼び出し側の受け手へ
/// 直ちに届く。より前の位置で後から失敗が起きることはない）。判断規則はこの 1 つのループに
/// 集約する。1 件の推論は単体推論と同じ `run_single` を通す（REQ-28）。推論側へ渡すのは
/// `input` のみ（id・ラベル・分割情報は渡さない）。
///
/// `JudgmentResult` は検証のためだけに作って捨てる。全件を保持すると選択肢 ID が結果ごとに
/// 複製されメモリが入力に比例して膨らむため、書き込み側が 1 件ずつ作り直す（REQ-39）。
fn compute_batch<R: Read, P: Preprocessor, B: ScoringBackend>(
    reader: R,
    io: &IoSchema,
    options: &[Choice],
    pipeline: &InferencePipeline<P, B>,
    scope: &JudgmentScope,
    deadline: Option<Instant>,
    output_byte_limit: usize,
) -> Result<(Vec<InferInput>, Vec<Prediction>), ErrorReport> {
    let mut reader = RecordReader::new(reader, io, MAX_INFER_BATCH_TOTAL_BYTES, deadline);
    let mut records = Vec::new();
    let mut predictions = Vec::new();
    let mut retained_scores: usize = 0;
    let mut output_bytes: usize = 0;
    while let Some(record) = reader.next_record()? {
        let partial = pipeline.infer_batch_partial_until(&[record.input()], deadline);
        let mut results = partial.results.into_iter();
        let prediction = match results.next() {
            Some(result) => result.map_err(|error| error.to_error_report())?,
            None => {
                return Err(partial
                    .failure
                    .map_or_else(|| report(ExitCode::RuntimeError), |e| e.to_error_report()));
            }
        };
        // 書き込み（`write_results`）と同じ関数・同じ値で行を作り、出力量を照合する（REQ-39・REQ-22。
        // 対象外・保留の行は `status` が長いため、ok 前提の長さで数えると実出力が上限を超えうる）。
        let result = judgment_from_prediction_scoped(options, scope, record.id(), &prediction)?;
        let line = result
            .to_json_line()
            .map_err(|_| report(ExitCode::RuntimeError))?;
        // 改行 1 バイトを加える。
        output_bytes = output_bytes.saturating_add(line.len()).saturating_add(1);
        if output_bytes > output_byte_limit {
            return Err(report(ExitCode::LimitExceeded));
        }
        retained_scores = retained_scores.saturating_add(prediction.scores().len());
        if retained_scores > MAX_INFER_BATCH_TOTAL_SCORES {
            return Err(report(ExitCode::LimitExceeded));
        }
        // この件が期限を超えて完了した場合の DeadlineExceeded は partial.failure に載る。
        if let Some(failure) = partial.failure {
            return Err(failure.to_error_report());
        }
        records.push(record);
        predictions.push(prediction);
    }
    if records.is_empty() {
        return Err(report(ExitCode::InvalidInput));
    }
    // 出力の途中で打ち切ると出力が壊れるため、書き始める前にだけ期限を確認する。
    if deadline_passed(deadline) {
        return Err(report(ExitCode::LimitExceeded));
    }
    Ok((records, predictions))
}

/// 入力を読み・推論し、成功なら結果を 1 行 1 JSON で、失敗なら `ErrorReport` を 1 行で書く。
/// 資源上限は [`BatchLimits::default`]。停止はプロセス終了で回収する（CLI 専用。モジュール doc）。
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

/// [`emit_infer_batch`] に [`JudgmentScope`]（対象外ラベル・校正）を渡す版（REQ-22・#478・#497）。
/// 行の `status` だけが変わり、終了コードは全行を計算できれば 0 のまま。
pub(crate) fn emit_infer_batch_scoped<W, R, P, B>(
    out: &mut W,
    reader: R,
    io: &IoSchema,
    options: &[Choice],
    scope: &JudgmentScope,
    pipeline: Arc<InferencePipeline<P, B>>,
) -> io::Result<ExitCode>
where
    W: Write,
    R: Read + Send + 'static,
    P: Preprocessor + Send + Sync + 'static,
    B: ScoringBackend + Send + Sync + 'static,
{
    emit_infer_batch_inner(
        out,
        None,
        reader,
        io,
        options,
        scope,
        pipeline,
        BatchLimits::default(),
        StallPolicy::TerminateProcess,
    )
}

/// [`emit_infer_batch`] の資源上限を指定できる版（REQ-39）。
///
/// 読み取り・推論・出力量の検証は専用スレッドで行い、呼び出し側は `limits.duration` で
/// `recv_timeout` する。改行が来ない低速な `Read` や、期限を確認できない 1 件の推論の途中でも、
/// 期限超過で `limit_exceeded` を返せる（スレッドは強制終了できないため、超過時は切り離して
/// 残す。呼び出し側は結果を書いたらプロセスを終了する前提で、CLI は 1 呼び出し 1 プロセス）。
/// 書き込みは総量（`limits.output_bytes`）を事前に検査済み。計算段階で確定した成功は、書き出し後に
/// 期限を理由に覆さない。読み手の停止で `write` がブロックする場合、`Write` は中断できない。
/// 停止はウォッチドッグとプロセス終了で回収する（常に。呼び出し側は CLI の 1 呼び出し 1
/// プロセス。長寿命プロセスからは直接呼ばず CLI を子プロセスで起動する。モジュール doc）。
///
/// # Errors
/// 書き込み・flush の失敗（および 1 行ごとの再構築失敗）を `io::Error` で返す。部分書き込み後は
/// 出力が壊れているため、追加の書き込み（残りの行・`ErrorReport`）は
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
    emit_infer_batch_inner(
        out,
        None,
        reader,
        io,
        options,
        &JudgmentScope::default(),
        pipeline,
        limits,
        StallPolicy::TerminateProcess,
    )
}

/// 結果行の出力先（`infer --out`。REQ-33・#459）。
///
/// [`BatchResults::finish`] は出力段階のウォッチドッグを解除する前に呼ばれ、公開・要約の出力・
/// 失敗時の報告まで同じ時間上限の対象にする（REQ-39）。
pub(crate) trait BatchResults: Write {
    /// 書き出し（または計算失敗の `ErrorReport` 出力）の結果 `result` を受けて後処理をし、
    /// 最終的な戻り値を返す。`out` は stdout。
    fn finish(
        &mut self,
        out: &mut dyn Write,
        result: Result<ExitCode, WriteFailure>,
    ) -> io::Result<ExitCode>;
}

/// 書き出し失敗の発生源（REQ-21・REQ-33）。stdout の失敗は部分書き込みの可能性があるため
/// 追加の出力をせず、結果ファイル側の失敗だけを `ErrorReport` にできるよう型で区別する。
#[derive(Debug)]
pub(crate) enum WriteFailure {
    /// stdout（`out`）への書き込み・flush の失敗。
    Stdout(io::Error),
    /// 結果の出力先（`--out` の一時ファイル）への書き込みの失敗。
    Results(io::Error),
}

impl WriteFailure {
    fn into_io(self) -> io::Error {
        match self {
            Self::Stdout(e) | Self::Results(e) => e,
        }
    }
}

/// [`emit_infer_batch`] の結果行（成功時のみ）を `results` へ、`ErrorReport` を `out` へ
/// 分けて書く版（`infer --out`。REQ-33・#459）。上限・停止の回収は既定どおり（CLI 専用）。
///
/// 失敗時は `results` へ何も書かない（遅延作成の出力先がファイルを作らずに済む）。
///
/// # Errors
/// [`emit_infer_batch_with_limits`] と同じ（`results` への書き込み失敗を含む）。
pub(crate) fn emit_infer_batch_split<W, R, P, B>(
    out: &mut W,
    results: &mut dyn BatchResults,
    reader: R,
    io: &IoSchema,
    options: &[Choice],
    scope: &JudgmentScope,
    pipeline: Arc<InferencePipeline<P, B>>,
) -> io::Result<ExitCode>
where
    W: Write,
    R: Read + Send + 'static,
    P: Preprocessor + Send + Sync + 'static,
    B: ScoringBackend + Send + Sync + 'static,
{
    emit_infer_batch_inner(
        out,
        Some(results),
        reader,
        io,
        options,
        scope,
        pipeline,
        BatchLimits::default(),
        StallPolicy::TerminateProcess,
    )
}

/// [`emit_infer_batch_with_limits`] の本体。回収方式を選べる（crate 内部。REQ-39）。
/// `results` が `Some` なら結果行はそこへ、`ErrorReport` は常に `out` へ書く（REQ-33・#459）。
// 引数が 8 個になるのは、公開 API 3 本（既定・上限指定・分割）が共有する本体で、入力・出力先・
// 定義・パイプライン・上限・回収方式をそれぞれ独立に差し替えるため。束ねる型を増やすと
// 公開経路ごとの呼び出しが読みにくくなるので、ここでは許容する。
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_infer_batch_inner<W, R, P, B>(
    out: &mut W,
    mut results: Option<&mut dyn BatchResults>,
    reader: R,
    io: &IoSchema,
    options: &[Choice],
    scope: &JudgmentScope,
    pipeline: Arc<InferencePipeline<P, B>>,
    limits: BatchLimits,
    policy: StallPolicy,
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
    let worker_scope = scope.clone();
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
                &worker_scope,
                deadline,
                output_bytes,
            );
            // 受信側が期限で去っていれば送信は失敗する。捨てる。
            let _ = tx.send(outcome);
        });
    let mut abandoned = false;
    let outcome = match spawned {
        Err(_) => Err(report(ExitCode::RuntimeError)),
        Ok(_) => {
            let received = match deadline {
                Some(d) => rx.recv_timeout(d.saturating_duration_since(Instant::now())),
                None => rx.recv().map_err(|_| mpsc::RecvTimeoutError::Disconnected),
            };
            match received {
                Ok(outcome) => outcome,
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    abandoned = true;
                    Err(report(ExitCode::LimitExceeded))
                }
                // 計算スレッドの panic。
                Err(mpsc::RecvTimeoutError::Disconnected) => Err(report(ExitCode::RuntimeError)),
            }
        }
    };
    // 計算スレッドを切り離した後は、どの失敗経路を通っても最後に必ず process::exit へ到達する
    // （`?`・return で公開関数から戻らない。切り離したスレッドはプロセス終了でしか回収できない）。
    if abandoned && policy.terminates() {
        let watchdog = OutputWatchdog::arm(true, limits.output_duration);
        std::process::exit(finish_abandoned(out, &outcome, watchdog));
    }
    let has_results = results.is_some();
    // 書き込み（結果行・ErrorReport とも）を対象に、停止を期限でプロセス終了へ倒す。
    // 起動に失敗したら上限なしで書かず、何も書かずに Err（exit 70）で終える。
    let _watchdog = OutputWatchdog::arm(policy.terminates(), limits.output_duration)?;
    let written = match outcome {
        Ok((records, predictions)) => {
            // 計算段階で期限内に全件が済んだ時点で結果は成功として確定している。書き出しと
            // flush の完了後に期限を理由に失敗へ変えない（確定点は 1 つ）。書き出しの停止は
            // ウォッチドッグ（CLI 経路）だけが扱う。
            let mut sink: &mut dyn Write = match results.as_deref_mut() {
                Some(results) => results,
                None => &mut *out,
            };
            let to_failure = if has_results {
                WriteFailure::Results
            } else {
                WriteFailure::Stdout
            };
            write_results(&mut sink, options, scope, &records, &predictions)
                .map(|()| ExitCode::Ok)
                .map_err(to_failure)
        }
        Err(error) => emit_error_report(out, &error).map_err(WriteFailure::Stdout),
    };
    // 公開・要約の出力・失敗時の報告も、ウォッチドッグを解除する前に行う（REQ-39）。
    match results {
        Some(results) => results.finish(&mut *out, written),
        None => written.map_err(WriteFailure::into_io),
    }
}

/// 検証済みの全件を 1 行ずつ作り直して書く。
fn write_results(
    sink: &mut &mut dyn Write,
    options: &[Choice],
    scope: &JudgmentScope,
    records: &[InferInput],
    predictions: &[Prediction],
) -> io::Result<()> {
    for (record, prediction) in records.iter().zip(predictions) {
        // predict_batch で検証済みのため、ここでの再構築は失敗しない想定。
        // 万一失敗しても部分出力のまま続けず、書き込み失敗と同じく打ち切る。
        let result = judgment_from_prediction_scoped(options, scope, record.id(), prediction)
            .map_err(|error| io::Error::other(error.message))?;
        write_ok_judgment(sink, &result)?;
    }
    Ok(())
}

/// 計算スレッドを切り離した後の終了処理。ErrorReport を書いて flush し、終了コード（常に
/// `limit_exceeded` の 20。時間上限の超過）を返す。呼び出し側が `process::exit` する。
///
/// ウォッチドッグの起動に失敗している（`watchdog` が `Err`）場合は、停止した出力先への書き込みを
/// 中断できないため ErrorReport を書かず、終了コードだけで返す（無期限に待たない。REQ-39）。
/// 書き込みの間はウォッチドッグを保持し、停止を終了へ倒す。
fn finish_abandoned<W: Write, T>(
    out: &mut W,
    outcome: &Result<T, ErrorReport>,
    watchdog: io::Result<Option<OutputWatchdog>>,
) -> i32 {
    if let Ok(_guard) = watchdog {
        // 切り離した場合の outcome は常に Err(limit_exceeded)。念のため同じ報告にする。
        let error = match outcome {
            Err(error) => error.clone(),
            Ok(_) => report(ExitCode::LimitExceeded),
        };
        let _ = emit_error_report(out, &error);
        let _ = out.flush();
    }
    i32::from(ExitCode::LimitExceeded.code())
}

/// 単件推論の協調的な期限（runtime）に足す、CLI 側のハード上限までの猶予（暫定）。協調的な期限が
/// 先に効けば通常の `limit_exceeded`、止まって返らないときだけ CLI の見張りが効く（REQ-39）。
pub const SINGLE_HARD_LIMIT_GRACE: Duration = Duration::from_millis(500);

/// 単件推論の既定の上限。`duration` は `core::limits::INFER_TIME_LIMIT`（推論 1 件の上限。REQ-39）。
#[must_use]
pub fn default_single_limits() -> BatchLimits {
    BatchLimits {
        duration: fandhe_edge_core::limits::INFER_TIME_LIMIT,
        ..BatchLimits::default()
    }
}

/// 計算スレッドで `work` を実行し、`duration` 以内に返らなければ切り離す（バッチと共通の見張り。
/// REQ-39）。
///
/// 期限超過時、`policy` がプロセス終了で回収するなら [`OutputWatchdog`] を起動して
/// [`finish_abandoned`] が `limit_exceeded` の JSON を書き、`process::exit(20)` する（戻らない）。
/// 回収しない方式（テスト専用）では `Err(limit_exceeded)` を返す。スレッドを起動できない・
/// panic は `runtime_error`。
pub(crate) fn run_with_stall_guard<W, T>(
    out: &mut W,
    duration: Duration,
    output_duration: Duration,
    policy: StallPolicy,
    work: impl FnOnce() -> Result<T, ErrorReport> + Send + 'static,
) -> Result<T, ErrorReport>
where
    W: Write,
    T: Send + 'static,
{
    let deadline = Instant::now().checked_add(duration);
    let (tx, rx) = mpsc::channel();
    let spawned = thread::Builder::new()
        .name("infer-single".to_string())
        .spawn(move || {
            let _ = tx.send(work());
        });
    if spawned.is_err() {
        return Err(report(ExitCode::RuntimeError));
    }
    let received = match deadline {
        Some(d) => rx.recv_timeout(d.saturating_duration_since(Instant::now())),
        None => rx.recv().map_err(|_| mpsc::RecvTimeoutError::Disconnected),
    };
    match received {
        Ok(outcome) => outcome,
        Err(mpsc::RecvTimeoutError::Timeout) => {
            let outcome: Result<T, ErrorReport> = Err(report(ExitCode::LimitExceeded));
            if policy.terminates() {
                let watchdog = OutputWatchdog::arm(true, output_duration);
                std::process::exit(finish_abandoned(out, &outcome, watchdog));
            }
            outcome
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => Err(report(ExitCode::RuntimeError)),
    }
}

/// `--text` の単件推論を実行して 1 行の JSON を書く（既定の上限。REQ-39・REQ-33）。
///
/// # Errors
/// 出力先への書き込み失敗（exit 70 に写す）。
pub fn emit_infer_single<W, P, B>(
    out: &mut W,
    io: &IoSchema,
    options: &[Choice],
    pipeline: Arc<InferencePipeline<P, B>>,
    id: &str,
    text: &str,
) -> io::Result<ExitCode>
where
    W: Write,
    P: Preprocessor + Send + Sync + 'static,
    B: ScoringBackend + Send + Sync + 'static,
{
    emit_infer_single_with_limits(
        out,
        io,
        options,
        pipeline,
        id,
        text,
        default_single_limits(),
    )
}

/// [`emit_infer_single`] に [`JudgmentScope`] を渡す版（REQ-22・#478・#497）。対象外なら
/// `status:"out_of_scope"`・終了コード 11、保留なら `status:"abstain"`・12（11 を優先）。
pub(crate) fn emit_infer_single_scoped<W, P, B>(
    out: &mut W,
    call: SingleCall<'_>,
    pipeline: Arc<InferencePipeline<P, B>>,
) -> io::Result<ExitCode>
where
    W: Write,
    P: Preprocessor + Send + Sync + 'static,
    B: ScoringBackend + Send + Sync + 'static,
{
    emit_infer_single_inner(
        out,
        pipeline,
        call,
        default_single_limits(),
        StallPolicy::TerminateProcess,
    )
}

/// [`emit_infer_single`] の上限を指定できる版。停止の回収は常にプロセス終了（CLI の 1 呼び出し
/// 1 プロセス専用。[`emit_infer_batch_with_limits`] と同じ）。
///
/// runtime へは `limits.duration` を協調的な期限として渡し（`infer_one_within`）、止まって返らない
/// ときだけ `duration + SINGLE_HARD_LIMIT_GRACE` で見切って `limit_exceeded`・exit 20 で終える。
///
/// # Errors
/// 出力先への書き込み失敗。
pub fn emit_infer_single_with_limits<W, P, B>(
    out: &mut W,
    io: &IoSchema,
    options: &[Choice],
    pipeline: Arc<InferencePipeline<P, B>>,
    id: &str,
    text: &str,
    limits: BatchLimits,
) -> io::Result<ExitCode>
where
    W: Write,
    P: Preprocessor + Send + Sync + 'static,
    B: ScoringBackend + Send + Sync + 'static,
{
    emit_infer_single_inner(
        out,
        pipeline,
        SingleCall {
            io,
            options,
            scope: &JudgmentScope::default(),
            id,
            text,
        },
        limits,
        StallPolicy::TerminateProcess,
    )
}

/// 単件推論 1 回分の入力（[`emit_infer_single_inner`] の引数をまとめたもの）。
pub(crate) struct SingleCall<'a> {
    pub(crate) io: &'a IoSchema,
    pub(crate) options: &'a [Choice],
    /// 定義の `out_of_scope_label` とパッケージの校正（REQ-22・#478・#497）。
    pub(crate) scope: &'a JudgmentScope,
    pub(crate) id: &'a str,
    pub(crate) text: &'a str,
}

/// [`emit_infer_single_with_limits`] の本体。回収方式を選べる（crate 内部。REQ-39）。
pub(crate) fn emit_infer_single_inner<W, P, B>(
    out: &mut W,
    pipeline: Arc<InferencePipeline<P, B>>,
    call: SingleCall<'_>,
    limits: BatchLimits,
    policy: StallPolicy,
) -> io::Result<ExitCode>
where
    W: Write,
    P: Preprocessor + Send + Sync + 'static,
    B: ScoringBackend + Send + Sync + 'static,
{
    let SingleCall {
        io,
        options,
        scope,
        id,
        text,
    } = call;
    let input = match InferInput::from_text(id, text, io) {
        Ok(input) => input,
        Err(e) => return emit_error_report(out, &infer_input_error_report(&e)),
    };
    let cooperative = limits.duration;
    let hard = cooperative.saturating_add(SINGLE_HARD_LIMIT_GRACE);
    let worker_options = options.to_vec();
    let worker_scope = scope.clone();
    let outcome = run_with_stall_guard(out, hard, limits.output_duration, policy, move || {
        // 推論側へ渡すのは `input` のみ（REQ-27）。
        let prediction = pipeline
            .infer_one_within(input.input(), cooperative)
            .map_err(|e| e.to_error_report())?;
        judgment_from_prediction_scoped(&worker_options, &worker_scope, input.id(), &prediction)
    });
    // 書き込みの停止も期限でプロセス終了へ倒す（バッチと同じ見張り）。
    let _watchdog = OutputWatchdog::arm(policy.terminates(), limits.output_duration)?;
    match outcome {
        Ok(result) => write_ok_judgment(out, &result),
        Err(error) => emit_error_report(out, &error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::{
        EvaluateArgs, InferArgs, InspectArgs, PackageArgs, RegisterArgs, SelectArgs, TrainArgs,
        TrainOp, TrainTarget,
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
                previous_project_dir: None,
            }),
            Command::Inspect(InspectArgs {
                project_dir: p(),
                seed: 42,
            }),
            Command::Train(TrainArgs {
                project_dir: p(),
                op: TrainOp::Run(TrainTarget::Candidate(0)),
                smoke: false,
                train_seed: None,
            }),
            Command::Evaluate(EvaluateArgs {
                project_dir: p(),
                candidate: 0,
                previous_project_dir: None,
            }),
            Command::Select(SelectArgs { project_dir: p() }),
            Command::Package(PackageArgs {
                project_dir: p(),
                allow_smoke: false,
            }),
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
        let err = read_batch_records_with_limit(
            line.as_bytes(),
            &io_schema(),
            MAX_INFER_BATCH_TOTAL_BYTES,
            Some(Instant::now()),
        )
        .unwrap_err();
        assert_eq!(err.code, ExitCode::LimitExceeded);
    }

    /// REQ-39: 期限内に解除（drop）したウォッチドッグはプロセスを終了しない。
    #[test]
    fn req39_watchdog_disarmed_before_deadline_does_not_exit() {
        let guard = OutputWatchdog::arm(true, Duration::from_millis(50)).unwrap();
        assert!(guard.is_some());
        drop(guard);
        std::thread::sleep(Duration::from_millis(150));
        assert!(
            OutputWatchdog::arm(false, Duration::from_millis(1))
                .unwrap()
                .is_none()
        );
    }

    /// REQ-39: ウォッチドッグの起動に失敗したら、上限なしで続行せず `Err`（fail-closed）。
    #[test]
    fn req39_watchdog_spawn_failure_is_error_not_unbounded() {
        let result = OutputWatchdog::arm_with(true, Duration::from_secs(1), |_task| {
            Err(io::Error::other("spawn failed"))
        });
        assert!(result.is_err());
    }

    const UNIT_DEFINITION: &str = r#"{"schema":"fandhe-edge-model-definition/v1","name":"t","version":1,
"judgment_type":"single_select","options":[
{"id":"a","display_name":"A","description":"a"},
{"id":"b","display_name":"B","description":"b"},
{"id":"c","display_name":"C","description":"c"}],"io":{"input":"bytes"}}"#;

    /// `f` は推論失敗、`s` は 500 ms 止まる（期限をまたぐ）バックエンド。
    struct FailOrSlow;
    impl ScoringBackend for FailOrSlow {
        fn scores(
            &self,
            ids: &fandhe_edge_runtime::pipeline::TokenIds,
        ) -> Result<Vec<f64>, fandhe_edge_runtime::pipeline::BackendError> {
            match ids.as_slice().first().copied() {
                Some(102) => Err(fandhe_edge_runtime::pipeline::BackendError::Failed),
                Some(115) => {
                    std::thread::sleep(Duration::from_millis(500));
                    Ok(vec![0.5, 0.25, 0.25])
                }
                _ => Ok(vec![0.5, 0.25, 0.25]),
            }
        }
        fn scores_limited(
            &self,
            ids: &fandhe_edge_runtime::pipeline::TokenIds,
            _limit: Duration,
        ) -> Result<Vec<f64>, fandhe_edge_runtime::pipeline::BackendError> {
            // テスト用スタブ: 時間上限は対象外のため委譲する。
            self.scores(ids)
        }
    }

    fn compute_with_deadline(inputs: &[&str], deadline_ms: u64) -> Result<usize, ExitCode> {
        let definition = fandhe_edge_core::definition::Definition::parse(UNIT_DEFINITION)
            .expect("valid definition");
        let pipeline = InferencePipeline::new(UnitPre, FailOrSlow);
        let text: String = inputs
            .iter()
            .enumerate()
            .map(|(i, input)| format!("{{\"id\":\"r{i}\",\"input\":\"{input}\"}}\n"))
            .collect();
        let deadline = Instant::now().checked_add(Duration::from_millis(deadline_ms));
        compute_batch(
            std::io::Cursor::new(text.into_bytes()),
            definition.io(),
            definition.options(),
            &pipeline,
            &JudgmentScope::default(),
            deadline,
            usize::MAX,
        )
        .map(|(records, _)| records.len())
        .map_err(|e| e.code)
    }

    /// REQ-33: 1 件目が推論失敗なら、後続で期限を超えることになっても入力順で先の 1 件目の
    /// `runtime_error` が確定し、以降の読み取り・推論には進まない。
    #[test]
    fn req33_earlier_record_failure_wins_over_later_deadline() {
        // 1 件目の失敗で即座に打ち切るため、2 件目の 500 ms の停止には到達しない。
        let started = Instant::now();
        assert_eq!(
            compute_with_deadline(&["f", "s", "a"], 250),
            Err(ExitCode::RuntimeError)
        );
        assert!(started.elapsed() < Duration::from_millis(400));
    }

    /// REQ-33: 1 件目で期限を超えたら、3 件目が推論失敗でも先の期限超過（`limit_exceeded`）を採る。
    #[test]
    fn req33_earlier_deadline_wins_over_later_record_failure() {
        assert_eq!(
            compute_with_deadline(&["s", "a", "f"], 250),
            Err(ExitCode::LimitExceeded)
        );
    }

    /// 書き込みのたびに一定時間止まる出力先。
    struct SlowWriter(Vec<u8>);
    impl Write for SlowWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            std::thread::sleep(Duration::from_millis(150));
            self.0.extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    struct UnitPre;
    impl Preprocessor for UnitPre {
        fn preprocess(
            &self,
            input: &str,
        ) -> Result<
            fandhe_edge_runtime::pipeline::TokenIds,
            fandhe_edge_runtime::pipeline::PreprocessError,
        > {
            Ok(fandhe_edge_runtime::pipeline::TokenIds::new(
                input.bytes().map(i64::from).collect(),
            ))
        }
    }

    struct UnitBackend;
    impl ScoringBackend for UnitBackend {
        fn scores(
            &self,
            _ids: &fandhe_edge_runtime::pipeline::TokenIds,
        ) -> Result<Vec<f64>, fandhe_edge_runtime::pipeline::BackendError> {
            Ok(vec![0.5, 0.25, 0.25])
        }
        fn scores_limited(
            &self,
            ids: &fandhe_edge_runtime::pipeline::TokenIds,
            _limit: Duration,
        ) -> Result<Vec<f64>, fandhe_edge_runtime::pipeline::BackendError> {
            // テスト用スタブ: 時間上限は対象外のため委譲する。
            self.scores(ids)
        }
    }

    /// REQ-21・REQ-39: 計算段階で全件が期限内に済んだら成功が確定し、書き出しが出力期限より
    /// 遅くても結果行は全件そろって exit 0 のまま（失敗へ覆さない）。回収しない内部モードで
    /// 確かめる（公開経路ではウォッチドッグが停止を終了させるため）。
    #[test]
    fn req39_success_is_committed_after_compute_even_if_writing_is_slow() {
        let definition = fandhe_edge_core::definition::Definition::parse(
            r#"{"schema":"fandhe-edge-model-definition/v1","name":"t","version":1,
"judgment_type":"single_select","options":[
{"id":"a","display_name":"A","description":"a"},
{"id":"b","display_name":"B","description":"b"},
{"id":"c","display_name":"C","description":"c"}],"io":{"input":"bytes"}}"#,
        )
        .expect("valid definition");
        let pipeline = Arc::new(InferencePipeline::new(UnitPre, UnitBackend));
        let mut out = SlowWriter(Vec::new());
        let code = emit_infer_batch_inner(
            &mut out,
            None,
            std::io::Cursor::new(
                b"{\"id\":\"r1\",\"input\":\"a\"}\n{\"id\":\"r2\",\"input\":\"b\"}\n".to_vec(),
            ),
            definition.io(),
            definition.options(),
            &JudgmentScope::default(),
            pipeline,
            BatchLimits {
                output_duration: Duration::from_millis(50),
                ..BatchLimits::default()
            },
            StallPolicy::Leak,
        )
        .unwrap();
        assert_eq!(code, ExitCode::Ok);
        let text = String::from_utf8(out.0).unwrap();
        assert_eq!(text.matches('\n').count(), 2);
        assert!(!text.contains("\"code\""));
    }

    /// 先頭バイトが `b` なら b 優勢、それ以外は a 優勢（対象外ラベルの argmax 制御用）。
    struct MixBackend;
    impl ScoringBackend for MixBackend {
        fn scores(
            &self,
            ids: &fandhe_edge_runtime::pipeline::TokenIds,
        ) -> Result<Vec<f64>, fandhe_edge_runtime::pipeline::BackendError> {
            Ok(if ids.as_slice().first() == Some(&98) {
                vec![0.25, 0.5, 0.25]
            } else {
                vec![0.5, 0.25, 0.25]
            })
        }
        fn scores_limited(
            &self,
            ids: &fandhe_edge_runtime::pipeline::TokenIds,
            _limit: Duration,
        ) -> Result<Vec<f64>, fandhe_edge_runtime::pipeline::BackendError> {
            self.scores(ids)
        }
    }

    fn abc_definition() -> fandhe_edge_core::definition::Definition {
        fandhe_edge_core::definition::Definition::parse(
            r#"{"schema":"fandhe-edge-model-definition/v1","name":"t","version":1,
"judgment_type":"single_select","options":[
{"id":"a","display_name":"A","description":"a"},
{"id":"b","display_name":"B","description":"b"},
{"id":"c","display_name":"C","description":"c"}],"io":{"input":"bytes"}}"#,
        )
        .expect("valid definition")
    }

    /// 対象外ラベルと校正（T=1・τ）の指定。`tau` が `None` なら校正なし。
    fn scope(oos: Option<&str>, tau: Option<f64>) -> JudgmentScope {
        JudgmentScope {
            out_of_scope_label: oos.map(str::to_string),
            calibration: tau.map(|threshold| InferCalibration {
                temperature: 1.0,
                threshold,
            }),
        }
    }

    /// `--text` の単件推論（MixBackend）を `scope` で実行し、終了コードと出力を返す。
    fn run_single(scope: &JudgmentScope, text: &str) -> (ExitCode, String) {
        let definition = abc_definition();
        let mut out: Vec<u8> = Vec::new();
        let code = emit_infer_single_scoped(
            &mut out,
            SingleCall {
                io: definition.io(),
                options: definition.options(),
                scope,
                id: "t",
                text,
            },
            Arc::new(InferencePipeline::new(UnitPre, MixBackend)),
        )
        .unwrap();
        (code, String::from_utf8(out).unwrap())
    }

    /// REQ-22・REQ-21: `--text` は argmax が対象外ラベルなら exit 11・`status:"out_of_scope"`、
    /// それ以外は exit 0・`ok`。`out_of_scope_label` なしは従来と同じ（exit 0）。
    #[test]
    fn req22_single_out_of_scope_label_argmax_exits_11() {
        let run = |oos: Option<&str>, text: &str| run_single(&scope(oos, None), text);
        let row = |status: &str, label: &str, scores: &str| {
            format!(
                "{{\"id\":\"t\",\"status\":\"{status}\",\"predicted_label\":\"{label}\",\"scores\":{scores}}}\n"
            )
        };
        let a_scores = "{\"a\":0.5,\"b\":0.25,\"c\":0.25}";
        let b_scores = "{\"a\":0.25,\"b\":0.5,\"c\":0.25}";
        assert_eq!(
            run(Some("a"), "x"),
            (ExitCode::OutOfScope, row("out_of_scope", "a", a_scores))
        );
        assert_eq!(
            run(Some("a"), "b"),
            (ExitCode::Ok, row("ok", "b", b_scores))
        );
        assert_eq!(run(None, "x"), (ExitCode::Ok, row("ok", "a", a_scores)));
    }

    /// REQ-22・REQ-21・#497: 校正があると、確信度（T=1 で top1 = 0.5）が τ 未満なら exit 12・
    /// `status:"abstain"`（scores は生の確率のまま）、τ ちょうどは採用（`>=`）。対象外の argmax は
    /// τ 未満でも 11 を優先し、校正なしは従来どおり 0。
    #[test]
    fn req22_issue497_single_abstain_exits_12_and_out_of_scope_wins() {
        let row = |status: &str, label: &str, scores: &str| {
            format!(
                "{{\"id\":\"t\",\"status\":\"{status}\",\"predicted_label\":\"{label}\",\"scores\":{scores}}}\n"
            )
        };
        let a_scores = "{\"a\":0.5,\"b\":0.25,\"c\":0.25}";
        assert_eq!(
            run_single(&scope(None, Some(0.6)), "x"),
            (ExitCode::Pending, row("abstain", "a", a_scores))
        );
        assert_eq!(
            run_single(&scope(None, Some(0.5)), "x"),
            (ExitCode::Ok, row("ok", "a", a_scores))
        );
        assert_eq!(
            run_single(&scope(Some("a"), Some(0.6)), "x"),
            (ExitCode::OutOfScope, row("out_of_scope", "a", a_scores))
        );
        assert_eq!(
            run_single(&scope(None, None), "x"),
            (ExitCode::Ok, row("ok", "a", a_scores))
        );
    }

    /// REQ-22・REQ-28・#497: `--input-file` は保留・対象外の行が混じっても exit 0 で、各行は同じ入力の
    /// `--text`（同じ id）の出力と完全一致する。
    #[test]
    fn req28_issue497_batch_rows_match_single_rows_including_abstain() {
        let definition = abc_definition();
        let scope = scope(Some("b"), Some(0.6));
        let inputs = ["x", "b", "y"];
        let text: String = inputs
            .iter()
            .map(|input| format!("{{\"id\":\"t\",\"input\":\"{input}\"}}\n"))
            .collect();
        let mut out: Vec<u8> = Vec::new();
        let code = emit_infer_batch_scoped(
            &mut out,
            std::io::Cursor::new(text.into_bytes()),
            definition.io(),
            definition.options(),
            &scope,
            Arc::new(InferencePipeline::new(UnitPre, MixBackend)),
        )
        .unwrap();
        assert_eq!(code, ExitCode::Ok);
        let singles: Vec<(ExitCode, String)> =
            inputs.iter().map(|i| run_single(&scope, i)).collect();
        assert_eq!(
            singles.iter().map(|(c, _)| *c).collect::<Vec<_>>(),
            vec![ExitCode::Pending, ExitCode::OutOfScope, ExitCode::Pending]
        );
        assert_eq!(
            String::from_utf8(out).unwrap(),
            singles
                .into_iter()
                .map(|(_, line)| line)
                .collect::<String>()
        );
    }

    /// REQ-22・REQ-21・REQ-28: `--input-file` は対象外の行が混じっても exit 0 で、該当行だけ
    /// `out_of_scope`。scores は単件と同じ。
    #[test]
    fn req22_batch_out_of_scope_rows_keep_exit_0_and_mark_only_those_rows() {
        let definition = abc_definition();
        let mut out: Vec<u8> = Vec::new();
        let code = emit_infer_batch_scoped(
            &mut out,
            std::io::Cursor::new(
                b"{\"id\":\"r1\",\"input\":\"x\"}\n{\"id\":\"r2\",\"input\":\"b\"}\n".to_vec(),
            ),
            definition.io(),
            definition.options(),
            &scope(Some("a"), None),
            Arc::new(InferencePipeline::new(UnitPre, MixBackend)),
        )
        .unwrap();
        assert_eq!(code, ExitCode::Ok);
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "{\"id\":\"r1\",\"status\":\"out_of_scope\",\"predicted_label\":\"a\",\"scores\":{\"a\":0.5,\"b\":0.25,\"c\":0.25}}\n\
             {\"id\":\"r2\",\"status\":\"ok\",\"predicted_label\":\"b\",\"scores\":{\"a\":0.25,\"b\":0.5,\"c\":0.25}}\n"
        );
    }

    /// REQ-39・REQ-22: 出力量の上限は書き込みと同じ行（対象外は `out_of_scope`・保留は `abstain`）で照合する。
    /// ok 前提の長さでは上限内でも実出力は超える上限で `limit_exceeded`（20）となり、何も書かない。
    #[test]
    fn req39_output_limit_counts_out_of_scope_rows_as_written() {
        let definition = abc_definition();
        let input = b"{\"id\":\"r1\",\"input\":\"x\"}\n{\"id\":\"r2\",\"input\":\"x\"}\n".to_vec();
        let run_with = |scope: JudgmentScope, output_bytes: usize| {
            let mut out: Vec<u8> = Vec::new();
            let code = emit_infer_batch_inner(
                &mut out,
                None,
                std::io::Cursor::new(input.clone()),
                definition.io(),
                definition.options(),
                &scope,
                Arc::new(InferencePipeline::new(UnitPre, MixBackend)),
                BatchLimits {
                    output_bytes,
                    ..BatchLimits::default()
                },
                StallPolicy::Leak,
            )
            .unwrap();
            (code, String::from_utf8(out).unwrap())
        };
        let (code, ok_text) = run_with(scope(None, None), usize::MAX);
        assert_eq!(code, ExitCode::Ok);
        // 2 行とも対象外なら、`"ok"` が `"out_of_scope"` になり 1 行あたり 10 バイト増える。
        let ok_len = ok_text.len();
        let (code, text) = run_with(scope(Some("a"), None), ok_len + 19);
        assert_eq!(code, ExitCode::LimitExceeded);
        assert!(!text.contains("predicted_label"));
        let (code, text) = run_with(scope(Some("a"), None), ok_len + 20);
        assert_eq!(code, ExitCode::Ok);
        assert_eq!(text.len(), ok_len + 20);
        // 2 行とも保留（#497）なら `"abstain"` で 1 行あたり 5 バイト増える。境界は書く行と同じ長さ。
        let (code, text) = run_with(scope(None, Some(0.6)), ok_len + 9);
        assert_eq!(code, ExitCode::LimitExceeded);
        assert!(!text.contains("predicted_label"));
        let (code, text) = run_with(scope(None, Some(0.6)), ok_len + 10);
        assert_eq!(code, ExitCode::Ok);
        assert_eq!(text.len(), ok_len + 10);
        assert_eq!(text.matches("\"status\":\"abstain\"").count(), 2);
    }

    /// REQ-39: 回収しない方式（テスト専用）では、返らない処理を期限で見切って `limit_exceeded`（20）を
    /// 返し、期限内に返る処理の結果はそのまま返す。
    #[test]
    fn req39_stall_guard_reports_limit_exceeded_without_exit_under_leak_policy() {
        let mut out: Vec<u8> = Vec::new();
        let stalled: Result<u8, ErrorReport> = run_with_stall_guard(
            &mut out,
            Duration::from_millis(50),
            Duration::from_secs(1),
            StallPolicy::Leak,
            || {
                thread::sleep(Duration::from_secs(5));
                Ok(1)
            },
        );
        assert_eq!(
            stalled.expect_err("must time out"),
            report(ExitCode::LimitExceeded)
        );
        let fast = run_with_stall_guard(
            &mut out,
            Duration::from_secs(5),
            Duration::from_secs(1),
            StallPolicy::Leak,
            || Ok(7_u8),
        );
        assert_eq!(fast.expect("in time"), 7);
        assert!(out.is_empty());
    }

    /// REQ-39: 切り離しの後は、ウォッチドッグの起動に失敗しても終了コードは 20 で、停止しうる
    /// 書き込みはしない。起動できていれば ErrorReport を 1 行書いて 20。
    #[test]
    fn req39_finish_abandoned_always_yields_limit_exceeded_exit_code() {
        let outcome: Result<(), ErrorReport> = Err(report(ExitCode::LimitExceeded));
        let mut out: Vec<u8> = Vec::new();
        let code = finish_abandoned(&mut out, &outcome, Err(io::Error::other("spawn failed")));
        assert_eq!(code, 20);
        assert!(out.is_empty());
        let code = finish_abandoned(&mut out, &outcome, Ok(None));
        assert_eq!(code, 20);
        let text = String::from_utf8(out).unwrap();
        assert_eq!(text.matches('\n').count(), 1);
        assert!(text.starts_with("{\"code\":\"limit_exceeded\""));
    }
}
