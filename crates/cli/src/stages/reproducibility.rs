//! `evaluate --seed-run-project` の再現性（3 seed 以上の Wilson 95% 区間の重なり。REQ-26・REQ-27・
//! REQ-39・TASK-26.3・#490）。
//!
//! # 呼び出し文脈
//!
//! [`super::evaluate`] が使う。seed ごとの学習は人が複製プロジェクト（`register → inspect` 後に複製し、
//! 各複製で `train --train-seed S → select → evaluate`）で行い、本モジュールはその評価記録を読むだけ
//! （再推論しない・複製プロジェクトの台帳に触れない）。
//!
//! - [`check_run_count`]: 自分を含む run 数が `MIN_REPRODUCIBILITY_RUNS..=MAX_REPRODUCIBILITY_RUNS` か
//!   （引数だけで決まるので最初に呼ぶ）
//! - [`load_seed_runs`]: 各複製プロジェクトを `Project::open`（cwd 配下・保持 fd 起点）で開き、選定記録 →
//!   評価記録を読んで自分と照合する。**適用権（`apply_once`）より前に呼ぶ**ため、違反（`invalid_input`）で
//!   適用権を失わない
//! - [`SeedRuns::judge`]: 自分の `correct/total` 確定後（台帳への完了記録の前）に評価器の
//!   `judge_reproducibility` を呼ぶ（判定は再実装しない）。残る失敗は `runtime_error`
//!
//! 判定は記録・報告のみで、終了コードと `package` の照合には使わない。

use std::path::{Path, PathBuf};

use fandhe_edge_core::evaluation_record::{
    ReproducibilityRecord, ReproducibilityRunRecord, ReproducibilityVerdict,
};
use fandhe_edge_core::exitcode::ErrorReport;
use fandhe_edge_core::hash::Sha256Digest;
use fandhe_edge_core::stage_report::{EvaluateInterval, EvaluateReproducibility, EvaluateSeedRun};
use fandhe_edge_data::eval_freeze::FreezeRecord;
use fandhe_edge_data::split_record::SplitRecord;
use fandhe_edge_eval::reproducibility::{
    MAX_REPRODUCIBILITY_RUNS, MIN_REPRODUCIBILITY_RUNS, OverlapVerdict, SeedRun,
    judge_reproducibility,
};
use fandhe_edge_eval::wilson::wilson_ci95;

use crate::project::{
    DATA_DIR, MAX_PROJECT_FILE_BYTES, Project, TRAIN_DATA_FILE, invalid, runtime,
};

use super::previous_comparison::{SelectedRecordMessages, read_selected_evaluation_record};
use super::train::read_split_record;

/// 自分を含む run 数（`seed_run_projects + 1`）を検証する。指定なし（0 件）は判定しないので通す。
///
/// # Errors
/// 自分を含めて `MIN_REPRODUCIBILITY_RUNS` 未満・`MAX_REPRODUCIBILITY_RUNS` 超は `invalid_input`。
pub(super) fn check_run_count(seed_run_projects: usize) -> Result<(), ErrorReport> {
    if seed_run_projects == 0 {
        return Ok(());
    }
    let runs = seed_run_projects.saturating_add(1);
    if runs < MIN_REPRODUCIBILITY_RUNS {
        return Err(invalid(
            "too few seed run projects for a reproducibility check",
        ));
    }
    if runs > MAX_REPRODUCIBILITY_RUNS {
        return Err(invalid(
            "too many seed run projects for a reproducibility check",
        ));
    }
    Ok(())
}

/// 自分（評価する候補）の照合用の値。
pub(super) struct OwnRun<'a> {
    /// 自分のプロジェクト（学習データと分割記録の照合に使う。読むだけ）。
    pub project: &'a Project,
    pub freeze: &'a FreezeRecord,
    pub definition_sha256: &'a str,
    pub candidate_id: &'a str,
    pub config_id: &'a str,
    /// 凍結した評価データの件数（適用前に数えた値）。
    pub total: u64,
}

/// 1 seed 分の件数。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RunCount {
    seed: u32,
    correct: u64,
    total: u64,
}

/// 照合済みの複製プロジェクトの件数と自分の seed（[`SeedRuns::judge`] の入力）。
#[derive(Debug)]
pub(super) struct SeedRuns {
    own_seed: u32,
    others: Vec<RunCount>,
}

/// 学習条件の照合に使う値（取り込み済みの学習データのバイト列の sha256 と分割記録）。
///
/// 分割記録（`split.json`。seed・分割規則・各分割のレコード内容ハッシュ。REQ-17）は train / validation の
/// 割付を、学習データの sha256 は行順を含む学習データ全体を固定する。学習 seed 以外の条件が違う複製の結果を
/// 再現性として扱わないため、自分と複製で両方が一致することを求める（PR #516 指摘）。
#[derive(Debug, PartialEq, Eq)]
struct TrainingFingerprint {
    train_sha256: Sha256Digest,
    split: SplitRecord,
}

/// プロジェクトの学習データ（上限つき・保持 fd 起点で読む）と分割記録から [`TrainingFingerprint`] を作る。
fn training_fingerprint(project: &Project) -> Result<TrainingFingerprint, ErrorReport> {
    let train = project.read(
        Path::new(DATA_DIR).join(TRAIN_DATA_FILE),
        MAX_PROJECT_FILE_BYTES,
    )?;
    Ok(TrainingFingerprint {
        train_sha256: Sha256Digest::of_bytes(&train),
        split: read_split_record(project)?,
    })
}

/// `"<candidate_id>:seed<S>"` から S を取り出す（正準形の u32 だけ。`train` の seed 記録と同じ規則）。
fn seed_from_config_id(config_id: &str, candidate_id: &str) -> Option<u32> {
    let text = config_id
        .strip_prefix(candidate_id)?
        .strip_prefix(":seed")?;
    text.parse::<u32>()
        .ok()
        .filter(|seed| seed.to_string() == text)
}

/// 各複製プロジェクトの評価記録を読み、自分と照合する（適用権を取る前に呼ぶ）。
///
/// 照合: 評価データの sha256・バイト長（凍結記録）・定義の正準化ハッシュ・学習データの sha256 と分割記録・
/// 候補 ID・評価件数が自分と一致し、`config_id` から取り出した seed が自分とも他の複製とも重複しないこと。
///
/// # Errors
/// 経路の拒否・記録が無い / 不正・照合の不一致は `invalid_input`、記録の上限超過は `limit_exceeded`。
pub(super) fn load_seed_runs(
    cwd: &Path,
    seed_run_projects: &[PathBuf],
    own: &OwnRun<'_>,
) -> Result<SeedRuns, ErrorReport> {
    let own_seed = seed_from_config_id(own.config_id, own.candidate_id)
        .ok_or_else(|| runtime("cannot determine train seed"))?;
    let evaluation_sha256 = own.freeze.sha256().to_hex();
    let own_training = training_fingerprint(own.project)?;
    let mut others: Vec<RunCount> = Vec::with_capacity(seed_run_projects.len());
    for dir in seed_run_projects {
        let project = Project::open(cwd, dir)?;
        let (index, record) = read_selected_evaluation_record(
            &project,
            &SelectedRecordMessages {
                no_selection: "seed run project has no selection record",
                invalid_selection: "seed run selection record is invalid",
                not_evaluated: "seed run project has not been evaluated",
                malformed: "seed run evaluation record is invalid",
            },
        )?;
        if record.candidate_index != index || record.correct > record.total {
            return Err(invalid("seed run evaluation record is invalid"));
        }
        if record.evaluation_sha256 != evaluation_sha256
            || record.evaluation_bytes != own.freeze.byte_len()
        {
            return Err(invalid("seed run evaluation data does not match"));
        }
        if record.definition_sha256 != own.definition_sha256 {
            return Err(invalid("seed run definition does not match"));
        }
        if training_fingerprint(&project)? != own_training {
            return Err(invalid(
                "seed run project was trained on different data or split",
            ));
        }
        if record.candidate_id != own.candidate_id {
            return Err(invalid("seed run candidate does not match"));
        }
        let seed = seed_from_config_id(&record.config_id, own.candidate_id)
            .ok_or_else(|| invalid("seed run config id is invalid"))?;
        if seed == own_seed || others.iter().any(|r| r.seed == seed) {
            return Err(invalid("seed run seed is duplicated"));
        }
        if record.total != own.total {
            return Err(invalid("seed run total does not match"));
        }
        others.push(RunCount {
            seed,
            correct: record.correct,
            total: record.total,
        });
    }
    Ok(SeedRuns { own_seed, others })
}

impl SeedRuns {
    /// 自分の件数を加えて seed 昇順に並べ、評価器の `judge_reproducibility` で判定する。
    ///
    /// # Errors
    /// 評価器の判定・区間の算出に失敗したら `runtime_error`（照合は適用前に済んでいるため想定外）。
    pub(super) fn judge(
        &self,
        correct: u64,
        total: u64,
        eval_data_hash: Sha256Digest,
    ) -> Result<(EvaluateReproducibility, ReproducibilityRecord), ErrorReport> {
        let fail = || runtime("reproducibility check failed");
        let mut runs = self.others.clone();
        runs.push(RunCount {
            seed: self.own_seed,
            correct,
            total,
        });
        runs.sort_by_key(|r| r.seed);
        let seed_runs: Vec<SeedRun> = runs
            .iter()
            .map(|r| SeedRun {
                seed: u64::from(r.seed),
                correct: r.correct,
                total: r.total,
                eval_data_hash,
            })
            .collect();
        let report = judge_reproducibility(&seed_runs).map_err(|_| fail())?;
        let verdict = match report.verdict() {
            OverlapVerdict::AllPairsOverlap => ReproducibilityVerdict::AllPairsOverlap,
            OverlapVerdict::SomePairsDisjoint => ReproducibilityVerdict::SomePairsDisjoint,
            _ => return Err(fail()),
        };
        let seed_at = |i: usize| runs.get(i).map(|r| r.seed).ok_or_else(fail);
        let disjoint_pairs = report
            .disjoint_pairs()
            .iter()
            .map(|p| Ok([seed_at(p.first())?, seed_at(p.second())?]))
            .collect::<Result<Vec<_>, ErrorReport>>()?;
        let stdout_runs = runs
            .iter()
            .map(|r| {
                let ci = wilson_ci95(r.correct, r.total).ok_or_else(fail)?;
                Ok(EvaluateSeedRun {
                    seed: r.seed,
                    correct: r.correct,
                    total: r.total,
                    ci95: EvaluateInterval {
                        lo: ci.lo(),
                        hi: ci.hi(),
                    },
                })
            })
            .collect::<Result<Vec<_>, ErrorReport>>()?;
        Ok((
            EvaluateReproducibility {
                runs: stdout_runs,
                verdict,
                disjoint_pairs,
            },
            ReproducibilityRecord {
                seeds: runs.iter().map(|r| r.seed).collect(),
                runs: runs
                    .iter()
                    .map(|r| ReproducibilityRunRecord {
                        seed: r.seed,
                        correct: r.correct,
                        total: r.total,
                    })
                    .collect(),
                verdict,
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fandhe_edge_core::exitcode::ExitCode;

    fn runs(own_seed: u32, others: &[(u32, u64)]) -> SeedRuns {
        SeedRuns {
            own_seed,
            others: others
                .iter()
                .map(|&(seed, correct)| RunCount {
                    seed,
                    correct,
                    total: 100,
                })
                .collect(),
        }
    }

    /// REQ-26・#490: 正解数が近い 3 seed は全組が重なる。seed 昇順に並び、区間は Wilson 95%
    /// （80/100 → [0.711_169_038_073_497_7, 0.866_634_077_440_901_3]。許容差 1e-9）。
    #[test]
    fn req26_issue490_close_runs_all_pairs_overlap() {
        let (report, record) = runs(42, &[(7, 82), (1, 78)])
            .judge(80, 100, Sha256Digest::of_bytes(b"e"))
            .expect("judge");
        assert_eq!(report.verdict, ReproducibilityVerdict::AllPairsOverlap);
        assert!(report.disjoint_pairs.is_empty());
        assert_eq!(
            report
                .runs
                .iter()
                .map(|r| (r.seed, r.correct))
                .collect::<Vec<_>>(),
            vec![(1, 78), (7, 82), (42, 80)]
        );
        let own = report.runs.get(2).expect("own");
        let ci = wilson_ci95(80, 100).expect("ci");
        assert!((own.ci95.lo - ci.lo()).abs() < 1e-9);
        assert!((own.ci95.hi - ci.hi()).abs() < 1e-9);
        assert!((own.ci95.lo - 0.711_169_038_073_497_7).abs() < 1e-9);
        assert!((own.ci95.hi - 0.866_634_077_440_901_3).abs() < 1e-9);
        assert_eq!(record.seeds, vec![1, 7, 42]);
        assert_eq!(record.verdict, ReproducibilityVerdict::AllPairsOverlap);
    }

    /// REQ-26・#490: 離れた正解数（20/100 と 90/100）の組は重ならず、seed の組で報告する。
    #[test]
    fn req26_issue490_far_runs_report_disjoint_seed_pairs() {
        let (report, _) = runs(42, &[(1, 20), (7, 85)])
            .judge(90, 100, Sha256Digest::of_bytes(b"e"))
            .expect("judge");
        assert_eq!(report.verdict, ReproducibilityVerdict::SomePairsDisjoint);
        assert_eq!(report.disjoint_pairs, vec![[1, 7], [1, 42]]);
    }

    /// REQ-26・REQ-39・#490: 自分を含め 3 run 未満・上限超は `invalid_input`。指定なしは通す。
    #[test]
    fn req26_issue490_run_count_is_checked() {
        assert!(check_run_count(0).is_ok());
        assert_eq!(
            check_run_count(1).expect_err("few").code,
            ExitCode::InvalidInput
        );
        assert!(check_run_count(MIN_REPRODUCIBILITY_RUNS - 1).is_ok());
        assert!(check_run_count(MAX_REPRODUCIBILITY_RUNS - 1).is_ok());
        assert_eq!(
            check_run_count(MAX_REPRODUCIBILITY_RUNS)
                .expect_err("many")
                .code,
            ExitCode::InvalidInput
        );
    }

    /// REQ-26・#490: seed は `<candidate_id>:seed<S>` の正準形の u32 だけを受理する。
    #[test]
    fn req26_issue490_seed_is_parsed_from_config_id() {
        assert_eq!(seed_from_config_id("c1:seed42", "c1"), Some(42));
        assert_eq!(seed_from_config_id("c1:seed042", "c1"), None);
        assert_eq!(seed_from_config_id("c1:seed+1", "c1"), None);
        assert_eq!(seed_from_config_id("c3:seed42", "c1"), None);
        assert_eq!(seed_from_config_id("c1:seed4294967296", "c1"), None);
    }
}
