//! 外部の予測ファイルを凍結 test で採点する PoC 用の入口（バイナリ `fandhe-edge-score` の本体）。
//!
//! **PoC-26 用。7 工程（`fandhe-edge`）の入出力契約の外**にあり、7 工程の引数・JSON・終了コードは
//! 変えない（REQ-41・TASK-41.1・#445）。
//!
//! # 役割
//!
//! PoC-26 の追加学習候補 P（Python の PoC スクリプトが `pred.jsonl` を出す）を、現行候補
//! （`evaluate` が保存する `evaluation_predictions.jsonl`。[`crate::prediction_lines`]）と同じ評価器・
//! 同じ凍結 test で採点する。指標・検定は評価器（`fandhe-edge-eval`）のものをそのまま使い、
//! 本モジュールは入力の接続と JSON 出力だけを持つ（評価ロジックを再実装しない。REQ-24）。
//!
//! # 手順
//!
//! 1. 凍結 test のハッシュ照合（[`load_frozen_evaluation`]。不一致・凍結記録の欠落は `invalid_input`）。
//!    評価データが無ければ `invalid_input`（`skipped` にはしない。採点対象が無いため）
//! 2. 各予測を `prepare_evaluation_input` へ通す（id 重複・欠落・不正 JSON は停止。不正な `scores` は
//!    不正解）。全予測で評価対象の id 列が一致することを確認
//! 3. majority は train 分割のラベルだけから作る（[`majority_from_train`]。`evaluate` と同じ関数）
//! 4. 各予測: 正解率・Macro-F1・ラベル別・混同行列・Wilson 95% 区間・対 majority の McNemar
//! 5. Holm: 採点対象の [対 majority, 対 `--compare` 各相手] を族の大きさ 3 固定で補正（相手が欠けても
//!    m は 3。事前登録。脱落は保守側）。`--reference` は族に入れず McNemar の生の値だけ出す
//!    P 以外（C1・C3・AR。`evaluate` が保存した予測）は、予測ファイルが現在の `--project-dir` の
//!    `candidates/<N>/evaluation_predictions.jsonl` にあり、同じディレクトリの `evaluation_record.json` が
//!    次をすべて満たすことを必須にする（満たさなければ `invalid_input`。手編集した予測・別プロジェクトや
//!    別候補の予測を比較相手にできない。#445・REQ-27）。P は評価記録を持たず対象外
//!    （P の seed・来歴は自己申告のまま。PoC スクリプトの出力は評価記録で束縛できない）
//!    - `predictions_sha256` が予測ファイルの sha256 と一致
//!    - `evaluation_sha256`・`evaluation_bytes` が現在の凍結記録と一致
//!    - `definition_sha256` が現在の定義の正準化ハッシュと一致（`evaluate` と同じ計算）
//!    - `config_id`（`<candidate_id>:seed<N>`）の seed が `--seed` と一致（形式不正も拒否）
//!    - `candidate_index` が `<N>` と一致し、`candidate_id`（kind）が NAME に対応（C1→c1・C3→c3・AR→autoregressive）
//! 6. 台帳: `poc26_score_ledger/<evaluation_sha256>/seed-<seed>/<NAME>.sha256` に予測ファイルの sha256 を新規作成で記録
//!
//! # 1 回限りの担保（REQ-27）
//!
//! `--candidate` の NAME が台帳に既にあれば、同じ sha256 でも拒否する（採点対象としての適用は 1 回限り。
//! 結果を見て予測ファイルを作り直して再採点できない）。`--compare`・`--reference` は同じ sha256 の
//! 再読込だけ許可し、別の sha256 は拒否する（比較相手の差し替えを許さない）。台帳の書き込みは全検証が
//! 通った後、出力の直前に行う（比較相手を先に、採点対象を最後に書く）。
//!
//! # 適用回数の機械的な制限（事前登録 4〜5 節）
//!
//! NAME は許可リスト {P, C1, C3, AR}、`--seed` は {0, 1, 2} に限り、台帳は凍結 test の sha256 単位に分ける。
//! 採点対象としての適用は 4 候補 × 3 seed = 最大 12 回に機械的に制限される。予測ファイルを作り直して
//! 同名・別 sha256 で出すと拒否し、別名は許可リストで、同じ sha256 の別名での再採点は台帳の照合で塞ぐ。
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
//! b 側 0.15・c 側 0.05・検出力 0.8・α = 0.05 / 3（Holm の族 m = 3）。
//!
//! # 入出力
//!
//! 引数は [`parse_args`]、結果は [`run`] が JSON 1 行の文字列で返す。エラーは既存 CLI と同じ
//! [`ErrorReport`]（`code`・`message`。終了コード 7 種）。予測・評価データの本文は出さない。
//! cli は `serde_json` に依存しないため JSON は手組み（任意の文字列は [`json_string`] で
//! エスケープ）。

use std::collections::BTreeSet;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

use fandhe_edge_core::evaluation_record::{EvaluationRecord, MAX_EVALUATION_RECORD_BYTES};
use fandhe_edge_core::exitcode::ErrorReport;
use fandhe_edge_core::fs::read_bounded_open_file;
use fandhe_edge_core::hash::Sha256Digest;
use fandhe_edge_data::eval_input::{PredictionOutcome, prepare_evaluation_input};
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

use crate::error_report::ToErrorReport;
use crate::prediction_lines::{json_f64, json_string};
use crate::project::{
    CANDIDATES_DIR, EVALUATION_PREDICTIONS_FILE, EVALUATION_RECORD_FILE, MAX_PROJECT_FILE_BYTES,
    Project, fs_report, invalid, runtime,
};
use crate::stages::baseline::majority_from_train;
use crate::stages::inspect::load_frozen_evaluation;
use crate::stages::train::verified_split;

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
    fn as_str(self) -> &'static str {
        match self {
            Role::Candidate => "candidate",
            Role::Compare => "compare",
            Role::Reference => "reference",
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
    project: &'a Project,
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

/// 予測ファイルが現在のプロジェクトの `candidates/<N>/evaluation_predictions.jsonl` にあり、同じ
/// ディレクトリの評価記録が予測ファイルの sha256・現在の凍結記録・定義・候補に束縛されていることを確認する
/// （#445・REQ-27。いずれの不一致・欠落も `invalid_input`）。
fn verify_bound_to_record(
    prov: &Provenance<'_>,
    named: &NamedPath,
    real: &Path,
    sha256: &str,
) -> Result<(), ErrorReport> {
    let outside = || invalid("prediction file is not under the project candidates directory");
    let rel = real
        .strip_prefix(prov.project.dir())
        .map_err(|_| outside())?;
    let parts: Vec<&str> = rel.iter().filter_map(|c| c.to_str()).collect();
    let [dir, index, file] = parts.as_slice() else {
        return Err(outside());
    };
    let index: usize = index.parse().map_err(|_| outside())?;
    if *dir != CANDIDATES_DIR || *file != EVALUATION_PREDICTIONS_FILE || rel.iter().count() != 3 {
        return Err(outside());
    }
    let record_rel = Path::new(CANDIDATES_DIR)
        .join(index.to_string())
        .join(EVALUATION_RECORD_FILE);
    let Some(bytes) = prov
        .project
        .read_optional(&record_rel, MAX_EVALUATION_RECORD_BYTES)?
    else {
        return Err(invalid(
            "evaluation record is missing for the prediction file",
        ));
    };
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
            "evaluation record does not belong to this project state",
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

/// 事前登録の仮定（b 側 0.15・c 側 0.05・検出力 0.8・α = 0.05 / 3）から必要件数を求める。
fn required_sample_size() -> Result<RequiredSampleSize, ErrorReport> {
    #[allow(clippy::cast_precision_loss)]
    let alpha = 0.05 / FAMILY_SIZE as f64;
    let model = McNemarSampleSizeAssumption::new(0.15, 0.05, alpha, 0.8)
        .map_err(|_| invalid("sample size cannot be computed"))?;
    required_sample_size_mcnemar(&model).map_err(|_| invalid("sample size cannot be computed"))
}

/// 評価器の判定を出力の語彙へ写す（将来 variant が増えたら黙って通さず失敗させる）。
fn verdict_str(v: BaselineVerdict) -> Result<&'static str, ErrorReport> {
    match v {
        BaselineVerdict::SignificantlyBetter => Ok("significantly_better"),
        BaselineVerdict::NotSignificantlyBetter => Ok("not_significantly_better"),
        BaselineVerdict::Undeterminable(_) => Ok("undeterminable"),
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

fn opt(v: Option<f64>) -> String {
    v.map_or_else(|| "null".to_string(), json_f64)
}

/// ラベル別指標と混同行列の JSON（`per_label`・`confusion_matrix`）。
fn metrics_json(labels: &[&str], m: &SingleSelectMetrics) -> Result<(String, String), ErrorReport> {
    let per_label: Vec<String> = m
        .per_label
        .iter()
        .map(|l| {
            format!(
                "{{\"label\":{},\"support\":{},\"predicted_count\":{},\"tp\":{},\"fp\":{},\"fn\":{},\"precision\":{},\"recall\":{},\"f1\":{}}}",
                json_string(&l.label),
                l.support,
                l.predicted_count,
                l.tp,
                l.fp,
                l.fn_,
                opt(l.precision),
                opt(l.recall),
                opt(l.f1)
            )
        })
        .collect();
    let mut rows = Vec::with_capacity(labels.len());
    for gold in 0..labels.len() {
        let columns = (0..labels.len())
            .map(ConfusionColumn::Label)
            .chain([
                ConfusionColumn::Invalid,
                ConfusionColumn::Abstain,
                ConfusionColumn::Error,
            ])
            .map(|c| {
                m.confusion
                    .get(gold, c)
                    .map(|n| n.to_string())
                    .ok_or_else(|| runtime("cannot read confusion matrix"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        rows.push(format!("[{}]", columns.join(",")));
    }
    let label_list: Vec<String> = labels.iter().map(|l| json_string(l)).collect();
    let confusion = format!(
        "{{\"labels\":[{}],\"columns\":[{},\"invalid\",\"abstain\",\"error\"],\"rows\":[{}]}}",
        label_list.join(","),
        label_list.join(","),
        rows.join(",")
    );
    Ok((format!("[{}]", per_label.join(",")), confusion))
}

/// 台帳のディレクトリ（凍結 test の sha256 単位・seed 単位）のプロジェクト内の相対パス。
fn ledger_dir(eval_sha: &str, seed: u32) -> PathBuf {
    Path::new(LEDGER_DIR)
        .join(eval_sha)
        .join(format!("seed-{seed}"))
}

/// 台帳ファイルのプロジェクト内の相対パス。
fn ledger_rel(eval_sha: &str, seed: u32, name: &str) -> PathBuf {
    ledger_dir(eval_sha, seed).join(format!("{name}.sha256"))
}

/// 今回の入力同士で同じ sha256 が別 NAME に使われていないこと（未記録の相手と同じ予測を採点対象に
/// して別名での再採点を迂回させない）。入力だけで決まるため、台帳ディレクトリを作る前に呼ぶ。
fn check_distinct_inputs(loaded: &[Loaded<'_>]) -> Result<(), ErrorReport> {
    for (i, a) in loaded.iter().enumerate() {
        if loaded
            .iter()
            .skip(i + 1)
            .any(|b| b.sha256 == a.sha256 && b.named.name != a.named.name)
        {
            return Err(invalid(
                "the same prediction file is used under more than one name",
            ));
        }
    }
    Ok(())
}

/// 台帳を照合し、書くべきもの（名前・sha256）を返す。違反は書き込み前に `invalid_input`。
///
/// 採点対象は、同名が台帳にある場合に加え、同じ seed の台帳に同じ sha256 が別の NAME で既にある場合も
/// 拒否する（別名での再採点を塞ぐ）。
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
                if l.role == Role::Candidate {
                    for other in ALLOWED_NAMES.iter().filter(|n| **n != l.named.name) {
                        let recorded = project
                            .read_optional(ledger_rel(eval_sha, seed, other), MAX_LEDGER_BYTES)?;
                        if recorded.is_some_and(|r| r.trim_ascii() == l.sha256.as_bytes()) {
                            return Err(invalid(
                                "prediction file has already been scored under another name",
                            ));
                        }
                    }
                }
                to_write.push(l);
            }
            Some(_) if l.role == Role::Candidate => {
                return Err(invalid("candidate has already been scored"));
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
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        match file.try_lock() {
            Ok(()) => return Ok(file),
            Err(std::fs::TryLockError::WouldBlock) if std::time::Instant::now() < deadline => {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            Err(_) => return Err(runtime("cannot lock score ledger")),
        }
    }
}

/// 台帳へ書く（比較相手を先、採点対象を最後。親ディレクトリが無ければ作る）。
fn write_ledger(
    project: &Project,
    eval_sha: &str,
    seed: u32,
    to_write: &[&Loaded<'_>],
) -> Result<(), ErrorReport> {
    let root = Path::new(LEDGER_DIR);
    let by_eval = root.join(eval_sha);
    let by_seed = ledger_dir(eval_sha, seed);
    for d in [root, by_eval.as_path(), by_seed.as_path()] {
        project.ensure_dir(d)?;
    }
    let ordered = to_write
        .iter()
        .filter(|l| l.role != Role::Candidate)
        .chain(to_write.iter().filter(|l| l.role == Role::Candidate));
    for l in ordered {
        project.write_new(
            ledger_rel(eval_sha, seed, &l.named.name),
            format!("{}\n", l.sha256).as_bytes(),
        )?;
    }
    Ok(())
}

/// 採点して結果の JSON 1 行（改行なし）を返す。
///
/// # Errors
/// 凍結ハッシュ不一致・評価データなし・予測の不備・台帳違反は `invalid_input`（64）、
/// 上限超過は `limit_exceeded`（20）、評価器・I/O の失敗は `runtime_error`（70）。
pub fn run(args: &ScoreArgs, cwd: &Path) -> Result<String, ErrorReport> {
    args.validate()?;
    let project = Project::open(cwd, &args.project_dir)?;
    let Some((freeze, eval_bytes)) = load_frozen_evaluation(&project)? else {
        return Err(invalid("evaluation data is not provided"));
    };
    let definition = project.load_definition()?;
    let label_set: BTreeSet<String> = definition.options().iter().map(|c| c.id.clone()).collect();
    let labels: Vec<&str> = definition.options().iter().map(|c| c.id.as_str()).collect();
    let gold_text = std::str::from_utf8(&eval_bytes)
        .map_err(|_| invalid("evaluation data is not valid UTF-8"))?;

    let prov = Provenance {
        project: &project,
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
    let n_total = u64::try_from(first.ids.len()).map_err(|_| runtime("too many records"))?;

    // majority は train 分割のラベルだけから作る（評価データを渡せない。REQ-27）。
    let records = project.load_records(&definition)?;
    let (split, _) = verified_split(&project, &records)?;
    let majority = Outcome::Label(majority_from_train(&definition, &records, &split)?);
    let majority_outcomes = vec![majority; first.golds.len()];
    let required = required_sample_size()?;

    let mut candidates_json = Vec::with_capacity(loaded.len());
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
        let (per_label, confusion) = metrics_json(&labels, &m)?;
        candidates_json.push(format!(
            "{{\"name\":{},\"role\":\"{}\",\"pred_sha256\":\"{}\",\"correct\":{},\"accuracy\":{},\"accuracy_wilson95\":[{},{}],\"macro_f1\":{},\"per_label\":{},\"confusion_matrix\":{},\"vs_majority\":{{\"b\":{},\"c\":{},\"p\":{},\"verdict\":\"{}\"}}}}",
            json_string(&l.named.name),
            l.role.as_str(),
            l.sha256,
            correct,
            json_f64(m.accuracy.overall.value()),
            json_f64(wilson.lo()),
            json_f64(wilson.hi()),
            opt(m.macro_f1.value()),
            per_label,
            confusion,
            counts.b_candidate_only,
            counts.c_baseline_only,
            json_f64(vs_majority.test().p_two_sided().value()),
            verdict_str(vs_majority.verdict())?,
        ));
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
    let mut holm_json = Vec::with_capacity(adjusted.len());
    for (name, h) in against.iter().zip(&adjusted) {
        let c = h.comparison();
        holm_json.push(format!(
            "{{\"against\":{},\"b\":{},\"c\":{},\"p_raw\":{},\"p_adjusted\":{},\"verdict\":\"{}\"}}",
            json_string(name),
            c.counts().b_candidate_only,
            c.counts().c_baseline_only,
            json_f64(c.test().p_two_sided().value()),
            json_f64(h.adjusted_p().value()),
            verdict_str(h.verdict())?,
        ));
    }

    let mut references_json = Vec::new();
    for l in loaded.iter().filter(|l| l.role == Role::Reference) {
        let c = compare_with(
            &labels,
            &target.golds,
            &target.outcomes,
            &l.outcomes,
            required,
        )?;
        references_json.push(format!(
            "{{\"candidate\":{},\"against\":{},\"b\":{},\"c\":{},\"p_raw\":{}}}",
            json_string(&target.named.name),
            json_string(&l.named.name),
            c.counts().b_candidate_only,
            c.counts().c_baseline_only,
            json_f64(c.test().p_two_sided().value()),
        ));
    }

    // 台帳は全検証が通った後、出力の直前に書く（検証の失敗で適用権を使わない）。
    let eval_sha = freeze.sha256().to_hex();
    // 確認から書き込みまでを seed 単位の排他ロックの中で行う（同時実行による二重適用を防ぐ）。
    check_distinct_inputs(&loaded)?;
    let _lock = lock_ledger(&project, &eval_sha, args.seed)?;
    let to_write = check_ledger(&project, &eval_sha, args.seed, &loaded)?;
    write_ledger(&project, &eval_sha, args.seed, &to_write)?;

    Ok(format!(
        "{{\"step\":\"score_predictions\",\"status\":\"ok\",\"seed\":{},\"evaluation_sha256\":\"{}\",\"n_total\":{},\"required_sample_size\":{},\"candidates\":[{}],\"holm\":{{\"candidate\":{},\"m\":{},\"comparisons\":[{}]}},\"references\":[{}]}}",
        args.seed,
        freeze.sha256().to_hex(),
        n_total,
        required.get(),
        candidates_json.join(","),
        json_string(&target.named.name),
        FAMILY_SIZE,
        holm_json.join(","),
        references_json.join(","),
    ))
}
