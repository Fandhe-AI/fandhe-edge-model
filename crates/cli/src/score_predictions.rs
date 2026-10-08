//! 外部の予測ファイルを凍結 test で採点する PoC 用の入口（バイナリ `fandhe-edge-score` の本体）。
//!
//! **PoC-26 用。7 工程（`fandhe-edge`）の入出力契約の外**にあり、7 工程の引数・JSON・終了コードは
//! 変えない（REQ-41・TASK-41.1・#445）。
//!
//! # 役割
//!
//! PoC-26 の追加学習候補 P（Python の PoC スクリプトが `pred.jsonl` を出す）を、現行候補
//! （`evaluate` が保存する `evaluation_predictions.jsonl`。[`fandhe_edge_core::stage_report::PredictionLine`]）と同じ評価器・
//! 同じ凍結 test で採点する。指標・検定は評価器（`fandhe-edge-eval`）のものをそのまま使い、
//! 本モジュールは入力の接続と JSON 出力だけを持つ（評価ロジックを再実装しない。REQ-24）。
//!
//! # 手順
//!
//! 1. 凍結 test のハッシュ照合（[`load_frozen_evaluation`]。不一致・凍結記録の欠落は `invalid_input`）。
//!    評価データが無ければ `invalid_input`（`skipped` にはしない。採点対象が無いため）
//! 2. 評価対象は `evaluate` と共有のデコード（`evaluation_records`。矛盾入力も除外しない）から作り、各予測を
//!    `prepare_evaluation_input` へ通す（id 重複・欠落・不正 JSON は停止。不正な `scores` は
//!    不正解。評価対象の id の欠け・余分・空ファイルは停止し、台帳を消費しない）。全予測で評価対象の id 列が一致することを確認
//! 3. majority は train 分割のラベルだけから作る（[`majority_from_train`]。`evaluate` と同じ関数）
//! 4. 各予測: 正解率・Macro-F1・ラベル別・混同行列・Wilson 95% 区間・対 majority の McNemar
//! 5. Holm: 採点対象の [対 majority, 対 `--compare` 各相手] を族の大きさ 3 固定で補正（相手が欠けても
//!    m は 3。事前登録。脱落は保守側）。`--reference` は族に入れず McNemar の生の値だけ出す
//!    P 以外（C1・C3・AR。`evaluate` が保存した予測）は、予測ファイルが（cwd 配下の）
//!    `…/candidates/<N>/evaluation_predictions.jsonl` にあり、同じディレクトリの `evaluation_record.json` が
//!    次をすべて満たすことを必須にする（満たさなければ `invalid_input`。手編集した予測・別の凍結 test や
//!    別候補の予測を比較相手にできない。#445・REQ-27）。比較対象は採点側（`--project-dir`）の凍結記録と
//!    定義。予測ファイルは `--project-dir` の外（同じ凍結 test の複製プロジェクト）でもよい（下の運用）。P は評価記録を持たず対象外
//!    （P の seed・来歴は自己申告のまま。PoC スクリプトの出力は評価記録で束縛できない）
//!    - `predictions_sha256` が予測ファイルの sha256 と一致
//!    - `evaluation_sha256`・`evaluation_bytes` が現在の凍結記録と一致
//!    - `definition_sha256` が現在の定義の正準化ハッシュと一致（`evaluate` と同じ計算）
//!    - `config_id`（`<candidate_id>:seed<N>`）の seed が `--seed` と一致（形式不正も拒否）
//!    - `candidate_index` が `<N>` と一致し、`candidate_id`（kind）が NAME に対応（C1→c1・C3→c3・AR→autoregressive）
//! 6. 台帳: `poc26_score_ledger/<evaluation_sha256>/seed-<seed>/<NAME>.sha256` に予測ファイルの sha256 を
//!    記録し、採点結果を同じディレクトリの `<採点対象 NAME>.report.json` に保存する（いずれも一時名への
//!    書き込み・fsync・上書きしない名前替えで原子的に作る。確定は呼び出し期限の内側ではなく [`emit`] が行う）
//!
//! # 運用（PoC-26。REQ-27・REQ-41・#445）
//!
//! `evaluate` は選定固定により 1 プロジェクト 1 候補しか評価できない。そこで、凍結済み（`register` →
//! `inspect` 済み）のプロジェクトを学習前に候補×seed ごとに複製し（C1×3・C3×3）、各複製で 1 候補だけ
//! `train --train-seed` → `select` → `evaluate` を行う。採点は原本プロジェクト 1 か所だけで行い、比較相手は
//! 複製側の `candidates/<N>/evaluation_predictions.jsonl` を読むだけ（複製を開かず、書かない）。
//! `--reference AR` の経路は残すが、AR は 7 工程から学習・評価できないため PoC-26 では欠測になる。
//!
//! # 1 回限りの担保（REQ-27）
//!
//! `--candidate` の NAME が台帳に既にあれば、同じ sha256 でも拒否する（採点対象としての適用は 1 回限り。
//! 結果を見て予測ファイルを作り直して再採点できない）。`--compare`・`--reference` は同じ sha256 の
//! 再読込だけ許可し、別の sha256 は拒否する（比較相手の差し替えを許さない）。台帳の確定は全検証が
//! 通り、期限内に採点が返り、出力のウォッチドッグ起動が済んだ後（直列化は台帳確定の直前）に [`emit`] のメイン
//! スレッドだけが行う（比較相手の台帳 → 採点対象の台帳 → 結果 `<NAME>.report.json` の順）。適用済みの判定は台帳の
//! `.sha256` だけが正で、結果ファイルは検査に使わない。拒否 message には結果があればその project
//! 相対パスを含め、無ければ `result was not saved` と明示する（stdout が失敗しても結果を取り戻せる）。
//!
//! # 適用回数の機械的な制限（事前登録 4〜5 節）
//!
//! NAME は許可リスト {P, C1, C3, AR}、`--seed` は {0, 1, 2} に限り、台帳は凍結 test の sha256 単位に分ける。
//! 採点対象としての適用は 4 候補 × 3 seed = 最大 12 回に機械的に制限される。予測ファイルを作り直して
//! 同名・別 sha256 で出すと拒否し、別名は許可リストで塞ぐ。
//! 別 NAME の予測が同じバイト列でも拒否しない（異なる候補が同じ予測を返す正当な比較がある。同じバイト列を
//! 別 NAME で再採点しても新しい情報は得られず、C1・C3・AR は評価記録の来歴照合で束縛される）。
//! `--reference` は最大 1 個、`--compare` は最大 2 個。役割も固定: `--compare` は C1・C3 のみ、
//! `--reference` は AR のみで、どちらも `--candidate P` のときだけ許す（`--candidate` は 4 つのいずれでもよい）。
//! 台帳の確認から書き込みまでは `<evaluation_sha256>/seed-<n>/.lock` の排他ロック（`flock`。プロセス終了で
//! 解放される）の中で行い、同時実行による二重適用を防ぐ。許可リスト外の候補を足すには、事前登録の追補と
//! コード変更が要る。
//!
//! ponytail: PoC-26 専用の固定値。汎用化（候補名・seed を定義から取る）は TASK-33.x の配線時。
//!
//! # 必要件数の仮定（事前登録 4〜5 節）
//!
//! b 側 0.15・c 側 0.05・検出力 0.8・α = 0.05 / 3（Holm の族 m = 3）。名前付き定数は
//! `ASSUMED_B_RATE` 等（出典は事前登録）。
//!
//! # 入出力
//!
//! 引数は [`parse_args`]、結果は [`score`] が構造化型（core の `ScoreReport`）で返す。確定前の結果は
//! crate 内部の `PendingScore` に閉じ、`commit`（台帳確定）を経ないと読めない。
//! 出力は [`emit`] が 7 工程と同じ出口（`write_stage_line`・`emit_error_report`）で JSON 1 行にし、
//! 呼び出し全体に時間上限（[`MAX_SCORE_DURATION`]）を持つ。エラーは既存 CLI と同じ
//! [`ErrorReport`]（`code`・`message`。終了コード 7 種）。予測・評価データの本文は出さない。
//! JSON の直列化は core の型に閉じる（cli は `serde_json` に依存しない）。
//!
//! # 既知の限界
//!
//! - 採点対象の台帳確定から結果の公開までの間（ごく小さい窓）でプロセスが終了すると、結果は失われ再採点も
//!   できない（1 回限りを優先する fail-closed）。比較相手の台帳だけ確定した状態の再実行は成功し、
//!   比較相手は同じ sha256 のみ許可される。電源断に対するディレクトリエントリの永続化までは保証しない
//! - 出力は選択肢数 L の 2 乗に比例する（混同行列）。L は定義の検査で `MAX_OPTIONS`（1024。core の
//!   `judgment`）以下に限られるため、1 候補あたり約 100 万セル、最大 4 候補で有界
//! - 台帳は採点を行うプロジェクト（`--project-dir`）単位。採点は原本 1 か所で行う運用で 1 回限りを守る。
//!   プロジェクトを複製して台帳を空にする操作、再 `register` による別台帳は塞げない
//!   （`final_test_ledger` と同じ既知の限界。オーナー了承の残余リスク）
//! - RSS 上限はプロセス内で強制しない（7 工程と同じ）。入力は予測ファイル 1 つ 64 MiB
//!   （`MAX_PROJECT_FILE_BYTES`）× 最大 4 ファイルと評価データで有界

use std::collections::BTreeSet;
use std::ffi::OsString;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use fandhe_edge_core::evaluation_record::{
    BaselineComparisonVerdict, EvaluationRecord, MAX_EVALUATION_RECORD_BYTES,
};
use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};
use fandhe_edge_core::fs::read_bounded_open_file;
use fandhe_edge_core::hash::Sha256Digest;
use fandhe_edge_core::stage_report::{
    ScoreCandidate, ScoreConfusionMatrix, ScoreHolm, ScoreHolmComparison, ScorePerLabel,
    ScoreReference, ScoreReport, ScoreRole, ScoreVsMajority,
};
use fandhe_edge_data::eval_input::{PredictionOutcome, prepare_evaluation_input};
use fandhe_edge_data::inspect::ValidRecord;
use fandhe_edge_eval::holm::{FamilySize, compare_candidates_with_holm};
use fandhe_edge_eval::metrics::{
    ConfusionColumn, EvalRecord, Outcome, SingleSelectMetrics, evaluate_single_select,
};
use fandhe_edge_eval::sample_size::{McNemarSampleSizeAssumption, required_sample_size_mcnemar};
use fandhe_edge_eval::significance::{
    BaselineComparison, BaselineVerdict, PairedRecord, RequiredSampleSize, compare_with_baseline,
};
use fandhe_edge_eval::wilson::wilson_ci95;
use fandhe_edge_guard::path::open_confined;

use crate::error_report::{ToErrorReport, emit_error_report};
use crate::infer_batch::{
    MAX_INFER_BATCH_OUTPUT_DURATION, OutputWatchdog, StallPolicy, run_with_stall_guard,
};
use crate::output::write_stage_line;
use crate::project::{
    CANDIDATES_DIR, EVALUATION_PREDICTIONS_FILE, EVALUATION_RECORD_FILE, MAX_PROJECT_FILE_BYTES,
    Project, fs_report, invalid, runtime,
};
use crate::stages::baseline::majority_from_train;
use crate::stages::evaluate::evaluation_records;
use crate::stages::inspect::load_frozen_evaluation;
use crate::stages::train::verified_split;

/// 採点 1 呼び出し全体（読み込み・計算・台帳の書き込み）の時間上限（暫定 600 秒。REQ-39）。
///
/// 暫定値（推定。実機での採点時間は未測定）。入力は有界（予測 64 MiB × 最大 4 ファイルと評価データ）。
/// 実測後に見直す。
pub const MAX_SCORE_DURATION: Duration = Duration::from_secs(600);
/// 台帳ロックの最大待ち時間（無限待ちを作らない。REQ-39）。
const LEDGER_LOCK_TIMEOUT: Duration = Duration::from_secs(10);
/// 台帳のディレクトリ名（プロジェクト直下。`seed-<seed>/<NAME>.sha256` を置く）。
const LEDGER_DIR: &str = "poc26_score_ledger";
/// 族の大きさ（事前登録: 対 majority・対 2 相手。相手が欠けても固定）。
const FAMILY_SIZE: usize = 3;
/// `--compare` の最大数（族の大きさ 3 から対 majority の 1 を引いた数）。
const MAX_COMPARES: usize = FAMILY_SIZE - 1;
/// 事前登録 4 節の候補名（P・C1・C3・AR〔autoregressive〕）。大文字小文字は区別する。
///
/// ponytail: PoC-26 専用の固定値。汎用化（候補名・seed を定義から取る）は TASK-33.x の配線時。
const ALLOWED_NAMES: [&str; 4] = ["P", "C1", "C3", "AR"];
/// 事前登録 5 節の seed。
const ALLOWED_SEEDS: [u32; 3] = [0, 1, 2];
/// `--reference` の最大数。
const MAX_REFERENCES: usize = 1;
/// 台帳ファイルの読み込み上限（sha256 の 16 進 64 文字と改行の余裕）。
const MAX_LEDGER_BYTES: u64 = 128;

/// 名前つきの予測ファイル。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamedPath {
    /// `[A-Za-z0-9_-]{1,32}`。台帳のファイル名にも使うため、この文字種に限る。
    pub name: String,
    /// カレントディレクトリ配下へ閉じ込めて開く予測ファイル。
    pub path: PathBuf,
}

/// `fandhe-edge-score` の引数。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScoreArgs {
    pub project_dir: PathBuf,
    pub seed: u32,
    /// 採点対象（Holm の族の主体）。
    pub candidate: NamedPath,
    /// Holm の族に入れる比較相手（0〜2 件）。
    pub compares: Vec<NamedPath>,
    /// 族に入れず McNemar の生の値だけを出す相手。
    pub references: Vec<NamedPath>,
}

impl ScoreArgs {
    /// 事前登録の制限（seed・NAME の許可リスト・件数・NAME の重複）を検査する。`parse_args` と `run` の
    /// 双方が呼ぶ（公開 API から直接組んだ値でも制限を迂回させない。REQ-27）。
    fn validate(&self) -> Result<(), ErrorReport> {
        if !ALLOWED_SEEDS.contains(&self.seed) {
            return Err(invalid("seed is not in the preregistered list"));
        }
        if self.references.len() > MAX_REFERENCES {
            return Err(invalid("too many --reference options"));
        }
        if self.compares.len() > MAX_COMPARES {
            return Err(invalid("too many --compare options"));
        }
        let mut names = BTreeSet::new();
        for (_, n) in self.entries() {
            validate_name(&n.name)?;
            if n.path.as_os_str().is_empty() {
                return Err(invalid("name must match [A-Za-z0-9_-]{1,32}"));
            }
            if !names.insert(n.name.as_str()) {
                return Err(invalid("duplicate NAME"));
            }
        }
        // 役割の制約（事前登録 5 節）: compare は C1・C3、reference は AR、どちらも採点対象が P のときだけ。
        let has_others = !self.compares.is_empty() || !self.references.is_empty();
        if has_others && self.candidate.name != "P" {
            return Err(invalid("--compare and --reference require --candidate P"));
        }
        if self
            .compares
            .iter()
            .any(|n| !matches!(n.name.as_str(), "C1" | "C3"))
        {
            return Err(invalid("--compare accepts only C1 or C3"));
        }
        if self.references.iter().any(|n| n.name != "AR") {
            return Err(invalid("--reference accepts only AR"));
        }
        Ok(())
    }
}

/// NAME の文字種・長さと許可リストを検査する。
fn validate_name(name: &str) -> Result<(), ErrorReport> {
    let name_ok = (1..=32).contains(&name.len())
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-');
    if !name_ok {
        return Err(invalid("name must match [A-Za-z0-9_-]{1,32}"));
    }
    if !ALLOWED_NAMES.contains(&name) {
        return Err(invalid("name is not in the preregistered list"));
    }
    Ok(())
}

/// `NAME=PATH` を分解する（NAME の文字種・長さを検査する）。
fn parse_named(value: &str) -> Result<NamedPath, ErrorReport> {
    let (name, path) = value
        .split_once('=')
        .ok_or_else(|| invalid("argument must be NAME=PATH"))?;
    if path.is_empty() {
        return Err(invalid("name must match [A-Za-z0-9_-]{1,32}"));
    }
    validate_name(name)?;
    Ok(NamedPath {
        name: name.to_string(),
        path: PathBuf::from(path),
    })
}

/// 引数列を解析する（`--key value` 形式のみ。未知・重複・欠落は `invalid_input`）。
///
/// # Errors
/// 引数の不備（固定 message。入力の値は載せない）、NAME の重複は `invalid_input`（64）。
pub fn parse_args<I: IntoIterator<Item = OsString>>(args: I) -> Result<ScoreArgs, ErrorReport> {
    let mut project_dir: Option<PathBuf> = None;
    let mut seed: Option<u32> = None;
    let mut candidate: Option<NamedPath> = None;
    let mut compares = Vec::new();
    let mut references = Vec::new();
    let mut iter = args.into_iter();
    while let Some(flag) = iter.next() {
        let flag = flag
            .into_string()
            .map_err(|_| invalid("unknown argument"))?;
        let value = iter
            .next()
            .ok_or_else(|| invalid("option requires a value"))?;
        match flag.as_str() {
            "--project-dir" => {
                if project_dir.replace(PathBuf::from(value)).is_some() {
                    return Err(invalid("duplicate option"));
                }
            }
            "--seed" => {
                let parsed = value
                    .to_str()
                    .and_then(|v| v.parse::<u32>().ok())
                    .ok_or_else(|| invalid("seed must be a u32"))?;
                if !ALLOWED_SEEDS.contains(&parsed) {
                    return Err(invalid("seed is not in the preregistered list"));
                }
                if seed.replace(parsed).is_some() {
                    return Err(invalid("duplicate option"));
                }
            }
            "--candidate" => {
                let named = parse_named(value.to_str().ok_or_else(|| invalid("invalid value"))?)?;
                if candidate.replace(named).is_some() {
                    return Err(invalid("duplicate option"));
                }
            }
            "--compare" | "--reference" => {
                let named = parse_named(value.to_str().ok_or_else(|| invalid("invalid value"))?)?;
                if flag == "--compare" {
                    compares.push(named);
                } else {
                    references.push(named);
                }
            }
            _ => return Err(invalid("unknown argument")),
        }
    }
    let args = ScoreArgs {
        project_dir: project_dir.ok_or_else(|| invalid("--project-dir is required"))?,
        seed: seed.ok_or_else(|| invalid("--seed is required"))?,
        candidate: candidate.ok_or_else(|| invalid("--candidate is required"))?,
        compares,
        references,
    };
    args.validate()?;
    Ok(args)
}

/// 予測ファイルの役割（出力の `role`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    Candidate,
    Compare,
    Reference,
}

impl Role {
    fn report_role(self) -> ScoreRole {
        match self {
            Role::Candidate => ScoreRole::Candidate,
            Role::Compare => ScoreRole::Compare,
            Role::Reference => ScoreRole::Reference,
        }
    }
}

impl ScoreArgs {
    /// 採点対象・比較相手・参照相手を、この順で返す。
    fn entries(&self) -> impl Iterator<Item = (Role, &NamedPath)> {
        std::iter::once((Role::Candidate, &self.candidate))
            .chain(self.compares.iter().map(|n| (Role::Compare, n)))
            .chain(self.references.iter().map(|n| (Role::Reference, n)))
    }
}

/// 読み込み・分類済みの 1 つの予測ファイル。
#[derive(Debug)]
struct Loaded<'a> {
    role: Role,
    named: &'a NamedPath,
    sha256: String,
    ids: Vec<String>,
    golds: Vec<String>,
    outcomes: Vec<Outcome>,
}

/// 評価記録の来歴照合に使う、現在のプロジェクトの値（#445・REQ-27）。
struct Provenance<'a> {
    /// 予測ファイルと評価記録を cwd 配下へ閉じ込めて開く起点。
    cwd: &'a Path,
    evaluation_sha256: String,
    evaluation_bytes: u64,
    definition_sha256: String,
    seed: u32,
}

/// NAME に対応する `candidate_id`（= kind）。
fn expected_candidate_id(name: &str) -> Option<&'static str> {
    match name {
        "C1" => Some("c1"),
        "C3" => Some("c3"),
        "AR" => Some("autoregressive"),
        _ => None,
    }
}

/// 予測ファイルが `…/candidates/<N>/evaluation_predictions.jsonl`（`evaluate` の保存先。別プロジェクト
/// でもよい）にあり、同じディレクトリの評価記録が予測ファイルの sha256・**採点側**（`--project-dir`）の
/// 凍結記録・定義・候補に束縛されていることを確認する（#445・REQ-27・REQ-41。いずれの不一致・欠落も
/// `invalid_input`）。
///
/// `evaluate` は選定固定により 1 プロジェクト 1 候補しか評価できないため、PoC-26 では候補×seed ごとに
/// 凍結済みプロジェクトを学習前に複製して評価し、採点は原本 1 か所で行う。比較相手の複製側は
/// 読むだけで、`Project::open` も書き込みもしない（cwd 配下への閉じ込めで 2 ファイルを開く）。
fn verify_bound_to_record(
    prov: &Provenance<'_>,
    named: &NamedPath,
    real: &Path,
    sha256: &str,
) -> Result<(), ErrorReport> {
    let outside = || invalid("prediction file is not under a candidates directory");
    let parts: Vec<&str> = real.iter().filter_map(|c| c.to_str()).collect();
    let [.., dir, index, file] = parts.as_slice() else {
        return Err(outside());
    };
    let index: usize = index.parse().map_err(|_| outside())?;
    if *dir != CANDIDATES_DIR || *file != EVALUATION_PREDICTIONS_FILE {
        return Err(outside());
    }
    // 同じディレクトリの評価記録を、予測ファイルと同じく cwd 配下へ閉じ込めて上限付きで読む。
    let record_path = real.with_file_name(EVALUATION_RECORD_FILE);
    let missing = || invalid("evaluation record is missing for the prediction file");
    let (record_file, record_real) =
        open_confined(prov.cwd, &record_path).map_err(|_| missing())?;
    if record_real.as_path().parent() != real.parent() {
        return Err(missing());
    }
    let bytes = read_bounded_open_file(
        record_file,
        record_real.as_path(),
        MAX_EVALUATION_RECORD_BYTES,
    )
    .map_err(|e| fs_report(&e))?;
    let record = EvaluationRecord::from_json_slice(&bytes)
        .map_err(|_| invalid("evaluation record is malformed"))?;
    if record.predictions_sha256.as_deref() != Some(sha256) {
        return Err(invalid(
            "prediction file does not match the evaluation record",
        ));
    }
    if record.evaluation_sha256 != prov.evaluation_sha256
        || record.evaluation_bytes != prov.evaluation_bytes
        || record.definition_sha256 != prov.definition_sha256
    {
        return Err(invalid(
            "evaluation record does not belong to this evaluation data",
        ));
    }
    // 代表構成 ID は `<candidate_id>:seed<N>`。N が `--seed` と一致しなければ別 seed の予測。
    let seed_ok = record
        .config_id
        .strip_prefix(record.candidate_id.as_str())
        .and_then(|r| r.strip_prefix(":seed"))
        .and_then(|n| n.parse::<u32>().ok())
        == Some(prov.seed);
    if !seed_ok {
        return Err(invalid("evaluation record seed does not match --seed"));
    }
    if record.candidate_index != index
        || expected_candidate_id(&named.name) != Some(record.candidate_id.as_str())
    {
        return Err(invalid(
            "evaluation record does not match the candidate name",
        ));
    }
    Ok(())
}

/// 評価対象のレコードから、`prepare_evaluation_input` へ渡す gold の JSONL（`{"id","label"}`）を作る。
///
/// `input` を載せないため、評価器側の正規化 input による重複・矛盾グループの除外が働かず、
/// `evaluate` と同じ全レコードが評価対象になる（REQ-27・#445）。
fn gold_jsonl(records: &[ValidRecord]) -> String {
    let mut out = String::new();
    for r in records {
        out.push_str("{\"id\":");
        push_json_string(&mut out, &r.id);
        out.push_str(",\"label\":");
        push_json_string(&mut out, &r.label_id);
        out.push_str("}\n");
    }
    out
}

/// JSON の文字列リテラルとして `value` を追記する（cli は `serde_json` に依存しない）。
fn push_json_string(out: &mut String, value: &str) {
    out.push('"');
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if u32::from(c) < 0x20 => out.push_str(&format!("\\u{:04x}", u32::from(c))),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// 予測ファイルを閉じ込めつきで上限付きに読み、評価入力として分類する。
fn load_prediction<'a>(
    cwd: &Path,
    prov: &Provenance<'_>,
    gold_text: &str,
    labels: &BTreeSet<String>,
    role: Role,
    named: &'a NamedPath,
) -> Result<Loaded<'a>, ErrorReport> {
    let (file, confined) = open_confined(cwd, &named.path).map_err(|e| e.to_error_report())?;
    let bytes = read_bounded_open_file(file, confined.as_path(), MAX_PROJECT_FILE_BYTES)
        .map_err(|e| fs_report(&e))?;
    let sha256 = Sha256Digest::of_bytes(&bytes).to_hex();
    if named.name != "P" {
        verify_bound_to_record(prov, named, confined.as_path(), &sha256)?;
    }
    let text =
        std::str::from_utf8(&bytes).map_err(|_| invalid("prediction file is not valid UTF-8"))?;
    let outcome = prepare_evaluation_input(gold_text, text, labels).map_err(|stop| {
        invalid(&format!(
            "prediction input rejected for {}: {}",
            named.name,
            stop.code()
        ))
    })?;
    // 評価対象の全 id をちょうど 1 回ずつ含むこと（欠け・余分は停止。空も拒否）。予測の失敗（invalid・
    // abstain・error の行）は不正解として数える既存の扱いのままで、欠落とは区別する。台帳の確定
    // （適用権の消費）より前の検査のため、不完全なファイルで 1 回限りの枠を失わない（REQ-27・#445）。
    // id の重複は `prepare_evaluation_input` が停止し、欠けは `pred_line == None`、余分は行数の超過で分かる。
    let pred_rows = text.lines().filter(|l| !l.trim().is_empty()).count();
    if outcome.active.iter().any(|row| row.pred_line.is_none()) || pred_rows != outcome.active.len()
    {
        return Err(invalid(&format!(
            "prediction file must contain every evaluation id exactly once for {}",
            named.name
        )));
    }
    let mut loaded = Loaded {
        role,
        named,
        sha256,
        ids: Vec::with_capacity(outcome.active.len()),
        golds: Vec::with_capacity(outcome.active.len()),
        outcomes: Vec::with_capacity(outcome.active.len()),
    };
    for row in outcome.active {
        loaded.outcomes.push(match row.prediction {
            PredictionOutcome::Label(l) => Outcome::Label(l),
            PredictionOutcome::Invalid(_) => Outcome::Invalid,
            PredictionOutcome::Abstain => Outcome::Abstain,
            PredictionOutcome::Error(_) => Outcome::Error,
        });
        loaded.ids.push(row.id);
        loaded.golds.push(row.gold_label);
    }
    Ok(loaded)
}

/// 事前登録の仮定: b 側（候補のみ正解）の割合。
///
/// 出典: 事前登録 4〜5 節（`poc26-preregistration.md`。REQ-41・TASK-41.1・#386）。
const ASSUMED_B_RATE: f64 = 0.15;
/// 事前登録の仮定: c 側（下限基準のみ正解）の割合（出典は [`ASSUMED_B_RATE`] と同じ）。
const ASSUMED_C_RATE: f64 = 0.05;
/// 事前登録の検出力（出典は [`ASSUMED_B_RATE`] と同じ）。
const TARGET_POWER: f64 = 0.8;
/// Holm の族 m = 3 に対する有意水準 α = 0.05 / 3（出典は [`ASSUMED_B_RATE`] と同じ）。
/// [`FAMILY_SIZE`] を変えるときはここも合わせる（下の const assert が食い違いを止める）。
const HOLM_ALPHA: f64 = 0.05 / 3.0;
const _: () = assert!(FAMILY_SIZE == 3);

/// 事前登録の仮定から必要件数を求める。
fn required_sample_size() -> Result<RequiredSampleSize, ErrorReport> {
    let model =
        McNemarSampleSizeAssumption::new(ASSUMED_B_RATE, ASSUMED_C_RATE, HOLM_ALPHA, TARGET_POWER)
            .map_err(|_| invalid("sample size cannot be computed"))?;
    required_sample_size_mcnemar(&model).map_err(|_| invalid("sample size cannot be computed"))
}

/// 評価器の判定を出力の語彙へ写す（将来 variant が増えたら黙って通さず失敗させる）。
fn verdict_of(v: BaselineVerdict) -> Result<BaselineComparisonVerdict, ErrorReport> {
    match v {
        BaselineVerdict::SignificantlyBetter => Ok(BaselineComparisonVerdict::SignificantlyBetter),
        BaselineVerdict::NotSignificantlyBetter => {
            Ok(BaselineComparisonVerdict::NotSignificantlyBetter)
        }
        BaselineVerdict::Undeterminable(_) => Ok(BaselineComparisonVerdict::Undeterminable),
        _ => Err(runtime("unsupported verdict")),
    }
}

/// `candidate` を `other` と対応づけて McNemar 比較する（`other` を下限基準側に置く。b＝候補のみ正解）。
fn compare_with(
    labels: &[&str],
    golds: &[String],
    candidate: &[Outcome],
    other: &[Outcome],
    required: RequiredSampleSize,
) -> Result<BaselineComparison, ErrorReport> {
    let paired: Vec<PairedRecord<'_>> = golds
        .iter()
        .zip(candidate)
        .zip(other)
        .map(|((gold, candidate), baseline)| PairedRecord {
            gold,
            candidate,
            baseline,
        })
        .collect();
    compare_with_baseline(labels, &paired, required)
        .map_err(|_| runtime("cannot compute comparison"))
}

/// ラベル別指標と混同行列を出力用の型へ写す。
fn metrics_report(
    labels: &[&str],
    m: &SingleSelectMetrics,
) -> Result<(Vec<ScorePerLabel>, ScoreConfusionMatrix), ErrorReport> {
    let per_label = m
        .per_label
        .iter()
        .map(|l| ScorePerLabel {
            label: l.label.clone(),
            support: l.support,
            predicted_count: l.predicted_count,
            tp: l.tp,
            fp: l.fp,
            fn_: l.fn_,
            precision: l.precision,
            recall: l.recall,
            f1: l.f1,
        })
        .collect();
    let mut rows = Vec::with_capacity(labels.len());
    for gold in 0..labels.len() {
        let row = (0..labels.len())
            .map(ConfusionColumn::Label)
            .chain([
                ConfusionColumn::Invalid,
                ConfusionColumn::Abstain,
                ConfusionColumn::Error,
            ])
            .map(|c| {
                m.confusion
                    .get(gold, c)
                    .ok_or_else(|| runtime("cannot read confusion matrix"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        rows.push(row);
    }
    let confusion =
        ScoreConfusionMatrix::new(labels.iter().map(|l| (*l).to_string()).collect(), rows)
            .ok_or_else(|| runtime("cannot read confusion matrix"))?;
    Ok((per_label, confusion))
}

/// 台帳のディレクトリ（凍結 test の sha256 単位・seed 単位）のプロジェクト内の相対パス。
fn ledger_dir(eval_sha: &str, seed: u32) -> PathBuf {
    Path::new(LEDGER_DIR)
        .join(eval_sha)
        .join(format!("seed-{seed}"))
}

/// 採点対象の結果の保存先（台帳と同じディレクトリ。台帳確定の直前に原子的に作る。REQ-27）。
fn report_rel(eval_sha: &str, seed: u32, name: &str) -> PathBuf {
    ledger_dir(eval_sha, seed).join(format!("{name}.report.json"))
}

/// 台帳ファイルのプロジェクト内の相対パス。
fn ledger_rel(eval_sha: &str, seed: u32, name: &str) -> PathBuf {
    ledger_dir(eval_sha, seed).join(format!("{name}.sha256"))
}

/// 採点対象が適用済み（台帳に NAME がある）のときの `invalid_input`。判定は台帳の `.sha256` だけが正で、
/// 結果ファイルは検査に使わない。結果があればその project 相対パス（データ本文は含めない）を message に
/// 添え、無ければ保存されていないことを明示する（REQ-27）。
fn already_scored(project: &Project, eval_sha: &str, seed: u32, name: &str) -> ErrorReport {
    let rel = report_rel(eval_sha, seed, name);
    match project.exists(&rel) {
        Ok(true) => invalid(&format!(
            "candidate has already been scored; result saved in {}",
            rel.display()
        )),
        _ => invalid("candidate has already been scored; result was not saved"),
    }
}

/// 台帳を照合し、書くべきもの（名前・sha256）を返す。違反は書き込み前に `invalid_input`。
///
/// 採点対象は同名が台帳にあれば拒否する。別 NAME の予測が同じバイト列でも拒否しない（異なる候補が
/// 同じ予測を返す正当な比較がある。1 回限りは NAME × seed の台帳と、C1・C3・AR の来歴照合で担保する）。
fn check_ledger<'a>(
    project: &Project,
    eval_sha: &str,
    seed: u32,
    loaded: &'a [Loaded<'_>],
) -> Result<Vec<&'a Loaded<'a>>, ErrorReport> {
    let mut to_write = Vec::new();
    for l in loaded {
        match project.read_optional(ledger_rel(eval_sha, seed, &l.named.name), MAX_LEDGER_BYTES)? {
            None => {
                to_write.push(l);
            }
            Some(_) if l.role == Role::Candidate => {
                return Err(already_scored(project, eval_sha, seed, &l.named.name));
            }
            Some(recorded) => {
                if recorded.trim_ascii() != l.sha256.as_bytes() {
                    return Err(invalid(
                        "prediction file differs from the one recorded for this name",
                    ));
                }
            }
        }
    }
    Ok(to_write)
}

/// `<ledger>/<evaluation_sha256>/seed-<n>/.lock` に排他ロック（`flock`）を取る。戻り値を保持している間
/// 有効で、drop またはプロセス終了で解放される（クラッシュ後に残って次を止めない）。最大 10 秒待つ
/// （無限待ちを作らない。REQ-39）。
fn lock_ledger(project: &Project, eval_sha: &str, seed: u32) -> Result<std::fs::File, ErrorReport> {
    let root = Path::new(LEDGER_DIR);
    let by_eval = root.join(eval_sha);
    let by_seed = ledger_dir(eval_sha, seed);
    for d in [root, by_eval.as_path(), by_seed.as_path()] {
        project.ensure_dir(d)?;
    }
    let rel = by_seed.join(".lock");
    // 既存なら作成が失敗するだけ（無視）。開けなければ次の open_file が拒否する。
    let _ = project.write_new(&rel, b"");
    let (file, _) = project.open_file(&rel)?;
    let deadline = Instant::now() + LEDGER_LOCK_TIMEOUT;
    loop {
        match file.try_lock() {
            Ok(()) => return Ok(file),
            Err(std::fs::TryLockError::WouldBlock) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(_) => return Err(runtime("cannot lock score ledger")),
        }
    }
}

/// 採点が済み、台帳への確定を待っている状態（REQ-27・REQ-39）。
///
/// [`run`] が返し、確定は [`emit`] のメインスレッドが期限内の返却を確認した後に [`PendingScore::commit`]
/// で行う。期限超過で切り離されたワーカーはこの型を作っても台帳を書けない（書くコードは `commit` だけで、
/// 呼ぶのは emit 側）。ロック（`flock` の fd）を保持し続け、確定または drop で解放される。
#[derive(Debug)]
pub(crate) struct PendingScore {
    project: Project,
    _lock: std::fs::File,
    eval_sha: String,
    seed: u32,
    candidate: String,
    /// 書くべき台帳エントリ（NAME・sha256）。比較相手を先、採点対象を最後に並べる。
    entries: Vec<(String, String)>,
    report: ScoreReport,
}

impl PendingScore {
    /// 台帳エントリを原子的に確定し（比較相手が先、採点対象が最後）、最後に結果
    /// `<NAME>.report.json` を原子的に置く。ロックの中。先に台帳ディレクトリの `.tmp-*` 残骸を消す。
    /// 途中で終了しても各ファイルは「無い」か「完全」で、適用済みの判定は台帳だけが正のため、採点対象の
    /// 台帳が確定した後は別 NAME・別 sha256 での再適用はできない。
    ///
    /// 限界: 採点対象の台帳確定から結果の公開までの間に終了すると、結果は失われ再採点もできない
    /// （ごく小さい窓。1 回限りを優先する fail-closed）。その状態の再実行の拒否 message は
    /// `result was not saved` になる。
    ///
    /// # Errors
    /// 書き込み失敗は `runtime_error`、既存は `invalid_input`。採点対象の台帳確定後の失敗は、確定済みで
    /// 結果が保存されていないことを message に含める。
    pub(crate) fn commit(self) -> Result<(ScoreReport, String), ErrorReport> {
        // 直列化は台帳を消費する前に済ませる（失敗しても適用権を使わない）。
        let report_line = self
            .report
            .to_json_line()
            .map_err(|_| runtime("cannot serialize score report"))?;
        self.project
            .remove_tmp_files(ledger_dir(&self.eval_sha, self.seed))?;
        for (name, sha) in &self.entries {
            self.project.publish_new_file(
                ledger_rel(&self.eval_sha, self.seed, name),
                format!("{sha}\n").as_bytes(),
            )?;
        }
        let rel = report_rel(&self.eval_sha, self.seed, &self.candidate);
        self.project
            .publish_new_file(&rel, format!("{report_line}\n").as_bytes())
            .map_err(|e| {
                ErrorReport::new(
                    e.code,
                    "candidate ledger committed but the result was not saved",
                )
            })?;
        Ok((self.report, report_line))
    }
}

/// 採点して台帳を確定し、結果を返す（`run` ＋ `commit`）。公開 API はこれだけで、確定前の結果には
/// 触れられない（台帳を消費せずに結果だけを取得する経路を作らない。REQ-27）。
///
/// 時間上限は持たない。呼び出し側が期限を持つこと（バイナリは [`emit`] が持つ）。
///
/// # Errors
/// [`run`]・[`PendingScore::commit`] と同じ。
pub fn score(args: &ScoreArgs, cwd: &Path) -> Result<ScoreReport, ErrorReport> {
    run(args, cwd)?.commit().map(|(report, _)| report)
}

/// 採点して、台帳確定待ちの結果を返す（出力・時間上限・台帳の書き込みは持たない。ロックの取得と台帳の
/// 検査までを行い、書き込みは [`PendingScore::commit`] が行う）。
///
/// 呼び出し全体の時間上限（[`MAX_SCORE_DURATION`]）と出力の期限は [`emit`] が持つ（`infer_batch` の
/// `run_with_stall_guard` と同じ見張り。REQ-39）。crate 内部専用で、確定前の結果（`PendingScore`）は
/// `commit` を経ないと読めない。
///
/// # Errors
/// 凍結ハッシュ不一致・評価データなし・予測の不備・台帳違反は `invalid_input`（64）、
/// 上限超過は `limit_exceeded`（20）、評価器・I/O の失敗は `runtime_error`（70）。
pub(crate) fn run(args: &ScoreArgs, cwd: &Path) -> Result<PendingScore, ErrorReport> {
    args.validate()?;
    let project = Project::open(cwd, &args.project_dir)?;
    let Some((freeze, eval_bytes)) = load_frozen_evaluation(&project)? else {
        return Err(invalid("evaluation data is not provided"));
    };
    let definition = project.load_definition()?;
    let label_set: BTreeSet<String> = definition.options().iter().map(|c| c.id.clone()).collect();
    let labels: Vec<&str> = definition.options().iter().map(|c| c.id.as_str()).collect();
    // 評価対象は `evaluate` と同じデコード経路（`evaluation_records`）から取る。矛盾入力グループも
    // 除外しない（件数・ID・順序が evaluate の評価記録と一致する。REQ-27・#445）。
    let eval_records =
        evaluation_records(&eval_bytes, &definition).map_err(|e| e.to_error_report())?;
    let gold_text = gold_jsonl(&eval_records);
    let gold_text = gold_text.as_str();

    let prov = Provenance {
        cwd,
        evaluation_sha256: freeze.sha256().to_hex(),
        evaluation_bytes: freeze.byte_len(),
        seed: args.seed,
        definition_sha256: definition
            .canonical_hash()
            .map_err(|_| runtime("cannot hash definition"))?
            .to_hex(),
    };
    let loaded = args
        .entries()
        .map(|(role, named)| load_prediction(cwd, &prov, gold_text, &label_set, role, named))
        .collect::<Result<Vec<_>, _>>()?;
    let first = loaded
        .first()
        .ok_or_else(|| runtime("no prediction loaded"))?;
    if loaded
        .iter()
        .any(|l| l.ids != first.ids || l.golds != first.golds)
    {
        return Err(invalid("prediction files cover different records"));
    }
    // 予測側の評価対象が evaluate の評価対象と同じ ID・同じ並びであること（厳密に保つ）。
    if first.ids.len() != eval_records.len()
        || first
            .ids
            .iter()
            .zip(&eval_records)
            .any(|(id, r)| *id != r.id)
    {
        return Err(invalid("prediction files do not match the evaluation data"));
    }
    let n_total = u64::try_from(first.ids.len()).map_err(|_| runtime("too many records"))?;

    // majority は train 分割のラベルだけから作る（評価データを渡せない。REQ-27）。
    let records = project.load_records(&definition)?;
    let (split, _) = verified_split(&project, &records)?;
    let majority = Outcome::Label(majority_from_train(&definition, &records, &split)?);
    let majority_outcomes = vec![majority; first.golds.len()];
    let required = required_sample_size()?;

    let mut candidates = Vec::with_capacity(loaded.len());
    for l in &loaded {
        let eval_records: Vec<EvalRecord<'_>> = l
            .golds
            .iter()
            .zip(&l.outcomes)
            .map(|(gold, outcome)| EvalRecord { gold, outcome })
            .collect();
        let m = evaluate_single_select(&labels, &eval_records)
            .map_err(|_| runtime("cannot compute evaluation metrics"))?;
        let correct = m.accuracy.overall.numerator();
        let wilson =
            wilson_ci95(correct, n_total).ok_or_else(|| runtime("cannot compute interval"))?;
        let vs_majority =
            compare_with(&labels, &l.golds, &l.outcomes, &majority_outcomes, required)?;
        let counts = vs_majority.counts();
        let (per_label, confusion_matrix) = metrics_report(&labels, &m)?;
        candidates.push(ScoreCandidate {
            name: l.named.name.clone(),
            role: l.role.report_role(),
            pred_sha256: l.sha256.clone(),
            correct,
            accuracy: m.accuracy.overall.value(),
            accuracy_wilson95: [wilson.lo(), wilson.hi()],
            macro_f1: m.macro_f1.value(),
            per_label,
            confusion_matrix,
            vs_majority: ScoreVsMajority {
                b: counts.b_candidate_only,
                c: counts.c_baseline_only,
                p: vs_majority.test().p_two_sided().value(),
                verdict: verdict_of(vs_majority.verdict())?,
            },
        });
    }

    // Holm: 採点対象の [対 majority, 対 compare 各相手] を m = 3 固定で補正する。
    let target = first;
    let mut comparisons = vec![compare_with(
        &labels,
        &target.golds,
        &target.outcomes,
        &majority_outcomes,
        required,
    )?];
    let mut against = vec!["majority".to_string()];
    for l in loaded.iter().filter(|l| l.role == Role::Compare) {
        comparisons.push(compare_with(
            &labels,
            &target.golds,
            &target.outcomes,
            &l.outcomes,
            required,
        )?);
        against.push(l.named.name.clone());
    }
    let family = FamilySize::new(FAMILY_SIZE).ok_or_else(|| runtime("invalid family size"))?;
    let adjusted = compare_candidates_with_holm(&comparisons, family)
        .map_err(|_| runtime("cannot adjust p values"))?;
    let mut holm_comparisons = Vec::with_capacity(adjusted.len());
    for (name, h) in against.into_iter().zip(&adjusted) {
        let c = h.comparison();
        holm_comparisons.push(ScoreHolmComparison {
            against: name,
            b: c.counts().b_candidate_only,
            c: c.counts().c_baseline_only,
            p_raw: c.test().p_two_sided().value(),
            p_adjusted: h.adjusted_p().value(),
            verdict: verdict_of(h.verdict())?,
        });
    }

    let mut references = Vec::new();
    for l in loaded.iter().filter(|l| l.role == Role::Reference) {
        let c = compare_with(
            &labels,
            &target.golds,
            &target.outcomes,
            &l.outcomes,
            required,
        )?;
        references.push(ScoreReference {
            candidate: target.named.name.clone(),
            against: l.named.name.clone(),
            b: c.counts().b_candidate_only,
            c: c.counts().c_baseline_only,
            p_raw: c.test().p_two_sided().value(),
        });
    }

    // 台帳は全検証が通った後に照合する。書き込みは emit 側の `commit`（検証の失敗・期限超過で適用権を
    // 使わない）。照合から確定までを seed 単位の排他ロックで保持する（同時実行による二重適用を防ぐ）。
    let eval_sha = freeze.sha256().to_hex();
    let lock = lock_ledger(&project, &eval_sha, args.seed)?;
    let to_write = check_ledger(&project, &eval_sha, args.seed, &loaded)?;
    let entries = to_write
        .iter()
        .filter(|l| l.role != Role::Candidate)
        .chain(to_write.iter().filter(|l| l.role == Role::Candidate))
        .map(|l| (l.named.name.clone(), l.sha256.clone()))
        .collect();

    let report = ScoreReport::new(
        args.seed,
        eval_sha.clone(),
        n_total,
        required.get(),
        candidates,
        ScoreHolm {
            candidate: target.named.name.clone(),
            m: FAMILY_SIZE,
            comparisons: holm_comparisons,
        },
        references,
    );
    Ok(PendingScore {
        project,
        _lock: lock,
        eval_sha,
        seed: args.seed,
        candidate: args.candidate.name.clone(),
        entries,
        report,
    })
}

/// 採点を [`MAX_SCORE_DURATION`] 以内に実行し、結果を 7 工程と同じ出口で書く（成功は 1 行 JSON を
/// `write_stage_line`、失敗は `emit_error_report`）。バイナリ `fandhe-edge-score` の本体。
///
/// 期限超過は `limit_exceeded`（20）。採点を切り離してプロセスを終了する（`infer_batch` の
/// `run_with_stall_guard` と同じ）。台帳を書くのは切り離されないメインスレッドだけなので、切り離された
/// ワーカーが走り切っても台帳・結果ファイルは書かれず、期限超過の再実行は適用権を消費していない。
///
/// 順序（REQ-27・REQ-39）: (b) 出力のウォッチドッグ起動 → (a) 直列化（`commit` の先頭。台帳の前）→ (c) 台帳エントリを原子的に
/// 確定（比較相手が先、採点対象が最後） → (d) 結果を `<NAME>.report.json` へ原子的に保存 → (e) ロック
/// 解放 → (f) stdout へ出力。失敗しうる (a)(b) は台帳を消費する前に行う。(f) が失敗・停止しても結果は
/// (d) に残り、再実行の「already scored」拒否の message にその project 相対パスが入る。
///
/// 残る限界: (c) の採点対象の確定と (d) の間で終了すると、結果は失われ再採点もできない
/// （`result was not saved`。ごく小さい窓。1 回限りを優先する fail-closed）。ファイルは各々が一時名への
/// 書き込み・fsync・`NOREPLACE` の名前替えで作られ、最終名は「無い」か「完全」のどちらかである。
///
/// # Errors
/// 出力先への書き込み失敗（呼び出し側は exit 70 に写す）。
pub fn emit<W: Write>(out: &mut W, args: &ScoreArgs, cwd: &Path) -> io::Result<ExitCode> {
    let (args, cwd) = (args.clone(), cwd.to_path_buf());
    emit_with(
        out,
        MAX_SCORE_DURATION,
        StallPolicy::TerminateProcess,
        move || run(&args, &cwd),
    )
}

/// [`emit`] の本体。計算と回収方式を差し替えられる（crate 内部。テストが実時間を待たないため）。
pub(crate) fn emit_with<W: Write>(
    out: &mut W,
    duration: Duration,
    policy: StallPolicy,
    work: impl FnOnce() -> Result<PendingScore, ErrorReport> + Send + 'static,
) -> io::Result<ExitCode> {
    let outcome =
        run_with_stall_guard(out, duration, MAX_INFER_BATCH_OUTPUT_DURATION, policy, work);
    // (b) 書き込みの停止も期限でプロセス終了へ倒す（7 工程の `infer` と同じ見張り）。
    // 起動に失敗しても、ここでは台帳をまだ書いていない（`outcome` は drop されロックが解ける）。
    let _watchdog = OutputWatchdog::arm(policy.terminates(), MAX_INFER_BATCH_OUTPUT_DURATION)?;
    // (a)(c)(d)(e): commit が直列化 → 台帳 → 結果を確定し、pending（ロック）を消費して解放する。
    match outcome.and_then(PendingScore::commit) {
        Ok((_, line)) => write_stage_line(out, Ok::<_, std::convert::Infallible>(line)),
        Err(error) => emit_error_report(out, &error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 台帳の確定・復旧のテスト。ガード層の閉じ込め（`openat` 等）は Linux・macOS 限定のため、
    /// Windows では `unsupported_platform` になり成立しない（プラットフォーム前提を明示する）。
    #[cfg(unix)]
    mod ledger {
        use super::*;

        /// 単体テスト用の作業ディレクトリ（cwd）と、その下の project を作る。
        fn scratch(case: &str) -> PathBuf {
            let root =
                std::env::temp_dir().join(format!("fe-score-unit-{case}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&root);
            std::fs::create_dir_all(root.join("proj")).expect("mkdir");
            root.canonicalize().expect("canonicalize")
        }

        const EVAL_SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

        /// ロックを取り済みの確定待ち（比較相手 C1 → 採点対象 P の順）。
        fn pending(root: &Path) -> Result<PendingScore, ErrorReport> {
            let project = Project::open(root, Path::new("proj"))?;
            let lock = lock_ledger(&project, EVAL_SHA, 0)?;
            let report = ScoreReport::new(
                0,
                EVAL_SHA.to_string(),
                1,
                1,
                Vec::new(),
                ScoreHolm {
                    candidate: "P".into(),
                    m: FAMILY_SIZE,
                    comparisons: Vec::new(),
                },
                Vec::new(),
            );
            Ok(PendingScore {
                project,
                _lock: lock,
                eval_sha: EVAL_SHA.to_string(),
                seed: 0,
                candidate: "P".into(),
                entries: vec![
                    ("C1".into(), "aa".repeat(32)),
                    ("P".into(), "bb".repeat(32)),
                ],
                report,
            })
        }

        fn ledger_path(root: &Path, file: &str) -> PathBuf {
            root.join("proj").join(ledger_dir(EVAL_SHA, 0)).join(file)
        }

        /// REQ-27・REQ-39: 期限を超えて切り離されたワーカーが走り切っても台帳・結果は書かれず、
        /// 期限超過の再実行は適用権を消費していないので成功する。
        #[test]
        fn req27_req39_timed_out_worker_never_writes_ledger() {
            let root = scratch("timeout");
            let (done_tx, done_rx) = std::sync::mpsc::channel();
            let worker_root = root.clone();
            let mut out: Vec<u8> = Vec::new();
            let code = emit_with(
                &mut out,
                Duration::from_millis(50),
                StallPolicy::Leak,
                move || {
                    std::thread::sleep(Duration::from_millis(300));
                    let pending = pending(&worker_root);
                    let _ = done_tx.send(());
                    pending
                },
            )
            .expect("write");
            assert_eq!(code, ExitCode::LimitExceeded);
            done_rx
                .recv_timeout(Duration::from_secs(30))
                .expect("worker finished");
            for f in ["C1.sha256", "P.sha256", "P.report.json"] {
                assert!(!ledger_path(&root, f).exists(), "{f}");
            }
            let worker_root = root.clone();
            let mut out: Vec<u8> = Vec::new();
            let code = emit_with(
                &mut out,
                Duration::from_secs(30),
                StallPolicy::Leak,
                move || pending(&worker_root),
            )
            .expect("write");
            assert_eq!(code, ExitCode::Ok);
            let line = String::from_utf8(out).expect("utf8");
            assert_eq!(
                std::fs::read_to_string(ledger_path(&root, "P.report.json")).expect("report"),
                line
            );
            assert_eq!(
                std::fs::read_to_string(ledger_path(&root, "C1.sha256")).expect("c1"),
                format!("{}\n", "aa".repeat(32))
            );
            let _ = std::fs::remove_dir_all(&root);
        }

        /// REQ-27・REQ-39: 孤立した一時ファイル（途中終了の残骸）があっても確定でき、残骸は台帳として
        /// 読まれない。確定後に同じ名前を確定しようとすると上書きせず拒否する。
        #[test]
        fn req27_leftover_tmp_files_do_not_block_commit() {
            let root = scratch("tmp");
            let p = pending(&root).expect("pending");
            let pid = std::process::id();
            std::fs::write(
                ledger_path(&root, &format!(".tmp-P.sha256-{pid}")),
                b"partial",
            )
            .expect("tmp");
            std::fs::write(
                ledger_path(&root, &format!(".tmp-P.report.json-{pid}")),
                b"{",
            )
            .expect("tmp");
            let (_, line) = p.commit().expect("commit");
            assert_eq!(
                std::fs::read_to_string(ledger_path(&root, "P.sha256")).expect("p"),
                format!("{}\n", "bb".repeat(32))
            );
            assert_eq!(
                std::fs::read_to_string(ledger_path(&root, "P.report.json")).expect("report"),
                format!("{line}\n")
            );
            // 一時名は片付けられ、2 回目の確定（二重適用）は上書きせず拒否される。
            assert!(!ledger_path(&root, &format!(".tmp-P.sha256-{pid}")).exists());
            // 残骸は `.tmp-*` を pid に関係なく消す。
            std::fs::write(ledger_path(&root, ".tmp-X-1"), b"old").expect("tmp");
            let again = pending(&root).expect("pending");
            let err = again.commit().expect_err("must not overwrite");
            assert_eq!(err.code, ExitCode::InvalidInput);
            assert!(!ledger_path(&root, ".tmp-X-1").exists());
            assert_eq!(
                std::fs::read_to_string(ledger_path(&root, "P.report.json")).expect("report"),
                format!("{line}\n")
            );
            let _ = std::fs::remove_dir_all(&root);
        }

        fn loaded<'a>(role: Role, named: &'a NamedPath, sha: &str) -> Loaded<'a> {
            Loaded {
                role,
                named,
                sha256: sha.to_string(),
                ids: Vec::new(),
                golds: Vec::new(),
                outcomes: Vec::new(),
            }
        }

        fn named(name: &str) -> NamedPath {
            NamedPath {
                name: name.to_string(),
                path: PathBuf::from("x.jsonl"),
            }
        }

        /// REQ-27: 採点対象の台帳確定後・結果の公開前に中断しても、同名の再実行は拒否され、message は結果が
        /// 保存されていないことを示す。別 NAME は独立。
        #[test]
        fn req27_interrupt_after_candidate_ledger_rejects_same_name() {
            let root = scratch("after-candidate");
            let project = Project::open(&root, Path::new("proj")).expect("open");
            let _lock = lock_ledger(&project, EVAL_SHA, 0).expect("lock");
            let sha = "bb".repeat(32);
            project
                .publish_new_file(ledger_rel(EVAL_SHA, 0, "P"), format!("{sha}\n").as_bytes())
                .expect("ledger");
            let p = named("P");
            // 別 NAME は同じバイト列でも採点できる（NAME × seed が 1 回限りの単位。REQ-27）。
            let ar = named("AR");
            check_ledger(&project, EVAL_SHA, 0, &[loaded(Role::Candidate, &ar, &sha)])
                .expect("other name is independent");
            let err = check_ledger(&project, EVAL_SHA, 0, &[loaded(Role::Candidate, &p, &sha)])
                .expect_err("same name");
            assert_eq!(
                err.message,
                "candidate has already been scored; result was not saved"
            );
            let _ = std::fs::remove_dir_all(&root);
        }

        /// REQ-27: 比較相手の台帳だけ確定して中断した状態の再実行は成功し、比較相手は同じ sha256 のみ許可。
        #[test]
        fn req27_interrupt_after_compare_ledger_allows_rerun() {
            let root = scratch("after-compare");
            let project = Project::open(&root, Path::new("proj")).expect("open");
            let lock = lock_ledger(&project, EVAL_SHA, 0).expect("lock");
            let (c1_sha, p_sha) = ("aa".repeat(32), "bb".repeat(32));
            project
                .publish_new_file(
                    ledger_rel(EVAL_SHA, 0, "C1"),
                    format!("{c1_sha}\n").as_bytes(),
                )
                .expect("ledger");
            let (p, c1) = (named("P"), named("C1"));
            let both = [
                loaded(Role::Candidate, &p, &p_sha),
                loaded(Role::Compare, &c1, &c1_sha),
            ];
            let ok = check_ledger(&project, EVAL_SHA, 0, &both).expect("rerun");
            assert_eq!(ok.len(), 1);
            let other = "cc".repeat(32);
            let err = check_ledger(
                &project,
                EVAL_SHA,
                0,
                &[
                    loaded(Role::Candidate, &p, &p_sha),
                    loaded(Role::Compare, &c1, &other),
                ],
            )
            .expect_err("swapped compare");
            assert_eq!(
                err.message,
                "prediction file differs from the one recorded for this name"
            );
            drop(lock);
            let _ = std::fs::remove_dir_all(&root);
        }

        /// REQ-39: 結果の公開に失敗したら、採点対象の台帳は確定済みで結果が保存されていないと message に示す。
        #[test]
        fn req27_report_failure_after_ledger_is_explicit() {
            let root = scratch("report-fail");
            let p = pending(&root).expect("pending");
            std::fs::write(ledger_path(&root, "P.report.json"), b"x").expect("squat");
            let err = p.commit().expect_err("report exists");
            assert_eq!(
                err.message,
                "candidate ledger committed but the result was not saved"
            );
            assert!(ledger_path(&root, "P.sha256").is_file());
            let _ = std::fs::remove_dir_all(&root);
        }
    }

    /// REQ-39: 採点が期限内に返らなければ `limit_exceeded`（20）の `ErrorReport` を 1 行書いて返す
    /// （実時間 600 秒を待たず、回収しない方式で検証する）。
    #[test]
    fn req39_score_deadline_returns_limit_exceeded() {
        let mut out: Vec<u8> = Vec::new();
        let code = emit_with(
            &mut out,
            Duration::from_millis(50),
            StallPolicy::Leak,
            || {
                std::thread::sleep(Duration::from_secs(5));
                Err(invalid("unreachable"))
            },
        )
        .expect("write");
        assert_eq!(code, ExitCode::LimitExceeded);
        let text = String::from_utf8(out).expect("utf8");
        assert_eq!(text.matches('\n').count(), 1);
        assert!(text.starts_with("{\"code\":\"limit_exceeded\""), "{text}");
    }

    /// REQ-39: 期限内に返る失敗はその `ErrorReport` の終了コードで 1 行書く。
    #[test]
    fn req39_score_in_time_failure_keeps_its_code() {
        let mut out: Vec<u8> = Vec::new();
        let code = emit_with(&mut out, Duration::from_secs(5), StallPolicy::Leak, || {
            Err(invalid("bad"))
        })
        .expect("write");
        assert_eq!(code, ExitCode::InvalidInput);
        assert!(
            String::from_utf8(out)
                .unwrap()
                .starts_with("{\"code\":\"invalid_input\"")
        );
    }
}
