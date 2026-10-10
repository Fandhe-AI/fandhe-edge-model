//! 中断した学習ジョブへの「やり直し」案内（REQ-34・TASK-34.3・issue #147）。
//!
//! # 役割
//!
//! チェックポイントからの再開は提供しない。中断（キャンセル・失敗）したジョブには、
//! 常に「再開不可・最初からやり直す」を機械可読な [`RestartGuidance`] で返す。
//! 加えて、次の実行で `out_dir` をどう扱うかを [`OutDirAction`] で案内する。
//! 本モジュールは純粋関数だけを持ち、ファイルシステムには触れない
//! （削除・rename・書き込みをしない）。
//!
//! # 再開を提供しない根拠（証拠種別: 実機の実測。PoC-19 の結果の引用）
//!
//! PoC-19 の実測（Conditional Go 条件「追加-4」・TASK-34.3 備考の表項目 8）では、
//! MLX の既定 GPU で再開後の予測が全件一致したのは 4 試行中 3 試行で、残る
//! 1 試行は 191/288 件（66.3%）しか一致しなかった。PoC-19 の `lifecycle_jobctl.py
//! status` はチェックポイントがあれば `resumable: true` を返していたが、本実装は
//! その逆で常に再開不可とする。これらの数値はコメントにだけ記し、出力には含めない。
//! 決定的な再開を将来提供する場合は、CPU 実行など別条件での実測（表項目 8 の
//! 未検証条件）が前提になる。
//!
//! # 呼び出し元の想定
//!
//! CLI `train` 工程（`stages::train`）と、#146（TASK-34.2）の状態確認（`train --status`・
//! #485）が呼ぶ。出力の JSON 契約への組み込みは `stages::train` で済み（#486）。
//!
//! # 自動掃除をしない理由
//!
//! `SIGKILL` フォールバック後に残る `out_dir`・tmp は自動では削除しない。本 crate は
//! ガード層の経路の閉じ込め（fd 起点の操作）を持たず、`NonEmpty` は公開済みの成果物
//! の可能性があり、削除は TOCTOU・誤削除の危険を伴うため。代わりに
//! [`OutDirAction`] で呼び出し元・人間へ案内する。自動掃除はスコープ外として
//! 別途追跡する（ガード層の fd 起点の rmdir など、閉じ込めを保証した設計が前提）。

use std::convert::Infallible;

use serde::{Serialize, Serializer};

use crate::error::TrainProcessError;
use crate::job::JobState;
use crate::process::{CancelStop, CancelledRun, OutDirResidue, TrainRunEnd};
use crate::result::TrainOutcome;

/// 固定の `reason_code`（再開を提供しない）。
pub const REASON_RESUME_NOT_SUPPORTED: &str = "resume_not_supported";

/// 再開の可否。値は 1 つだけで、直列化すると常に `false`。`true` を表現できない
/// （再開を誤って提供する経路を型で作らない。REQ-34）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResumeNotProvided;

impl Serialize for ResumeNotProvided {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bool(false)
    }
}

/// 案内する行動。最初からのやり直しの 1 つだけ。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RestartAction {
    /// 最初からやり直す。
    RestartFromScratch,
}

/// 次の実行での `out_dir` の扱いの案内。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OutDirAction {
    /// 何も残っていない。同じ `out_dir` でやり直してよい。
    ReuseAllowed,
    /// 空の予約ディレクトリが残っている。空ディレクトリを消すか別の `out_dir` を使う
    /// （そのままだと排他的な作成が `output_conflict` になる）。
    RemoveEmptyOrChooseNew,
    /// 公開済みの成果物の可能性、または判定不能。再利用せず、手で確かめて別の
    /// `out_dir` を使う。
    DoNotReuseInspectManually,
    /// 残ったものを観測していない。`out_dir` の状態は不明。
    NotInspected,
}

impl OutDirAction {
    /// 固定の英語メッセージ（パス・データ由来の文字列は含めない）。
    const fn message(self) -> &'static str {
        match self {
            Self::ReuseAllowed => {
                "Resume is not supported. Restart the job from scratch; the same out_dir can be reused."
            }
            Self::RemoveEmptyOrChooseNew => {
                "Resume is not supported. Restart the job from scratch; remove the empty out_dir left behind or choose a new out_dir."
            }
            Self::DoNotReuseInspectManually => {
                "Resume is not supported. Restart the job from scratch; do not reuse the out_dir, inspect it manually and choose a new out_dir."
            }
            Self::NotInspected => {
                "Resume is not supported. Restart the job from scratch; the state of out_dir was not inspected, so check it before reuse."
            }
        }
    }

    /// 観測した残置から案内へ写す。
    const fn from_residue(residue: OutDirResidue) -> Self {
        match residue {
            OutDirResidue::Absent => Self::ReuseAllowed,
            OutDirResidue::EmptyReservation => Self::RemoveEmptyOrChooseNew,
            OutDirResidue::NonEmpty | OutDirResidue::Unknown => Self::DoNotReuseInspectManually,
        }
    }
}

/// やり直しの案内（機械可読）。`resumable` は型の上で `false` 固定。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct RestartGuidance {
    resumable: ResumeNotProvided,
    action: RestartAction,
    reason_code: &'static str,
    out_dir_action: OutDirAction,
    message: &'static str,
}

impl RestartGuidance {
    const fn new(out_dir_action: OutDirAction) -> Self {
        Self {
            resumable: ResumeNotProvided,
            action: RestartAction::RestartFromScratch,
            reason_code: REASON_RESUME_NOT_SUPPORTED,
            out_dir_action,
            message: out_dir_action.message(),
        }
    }

    /// 再開可否（常に `false`）。
    #[must_use]
    pub const fn resumable(&self) -> bool {
        false
    }

    /// 案内する行動。
    #[must_use]
    pub const fn action(&self) -> RestartAction {
        self.action
    }

    /// 固定の理由コード。
    #[must_use]
    pub const fn reason_code(&self) -> &'static str {
        self.reason_code
    }

    /// `out_dir` の扱いの案内。
    #[must_use]
    pub const fn out_dir_action(&self) -> OutDirAction {
        self.out_dir_action
    }

    /// 固定の英語メッセージ。
    #[must_use]
    pub const fn message(&self) -> &'static str {
        self.message
    }
}

/// 再開の要求が拒否されたこと。[`RestartGuidance`] を持つ。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResumeRefused {
    guidance: RestartGuidance,
}

impl ResumeRefused {
    /// 拒否に添えるやり直しの案内。
    #[must_use]
    pub const fn guidance(&self) -> RestartGuidance {
        self.guidance
    }
}

/// 再開の要求を必ず拒否する口。成功値の型が [`Infallible`] のため、再開が成功する
/// 経路は型の上で存在しない（REQ-34）。
///
/// # Errors
///
/// 常に [`ResumeRefused`]。
pub const fn request_resume(out_dir_action: OutDirAction) -> Result<Infallible, ResumeRefused> {
    Err(ResumeRefused {
        guidance: RestartGuidance::new(out_dir_action),
    })
}

/// キャンセルで止めた実行への案内。
///
/// 残置の観測が必要なのに無い不整合は、安全側の [`OutDirAction::NotInspected`]。
#[must_use]
pub fn guidance_for_cancelled(run: &CancelledRun) -> RestartGuidance {
    let action = match (run.stop(), run.out_dir_residue()) {
        (CancelStop::BeforeStart | CancelStop::Cooperative, _) => OutDirAction::ReuseAllowed,
        (CancelStop::ForcedKill | CancelStop::Unconfirmed, Some(residue)) => {
            OutDirAction::from_residue(residue)
        }
        (CancelStop::ForcedKill | CancelStop::Unconfirmed, None) => OutDirAction::NotInspected,
    };
    RestartGuidance::new(action)
}

/// 実行の結果への案内。成功（中断ではない）は `None`。
#[must_use]
pub fn guidance_for_run(
    result: &Result<TrainRunEnd, TrainProcessError>,
) -> Option<RestartGuidance> {
    match result {
        Ok(TrainRunEnd::Completed(run)) => match run.outcome() {
            TrainOutcome::Ok(_) => None,
            TrainOutcome::Error(_) => Some(RestartGuidance::new(OutDirAction::NotInspected)),
        },
        Ok(TrainRunEnd::Cancelled(run)) => Some(guidance_for_cancelled(run)),
        Err(TrainProcessError::CancelOutcomeUnconfirmed { residue }) => {
            Some(RestartGuidance::new(OutDirAction::from_residue(*residue)))
        }
        Err(_) => Some(RestartGuidance::new(OutDirAction::NotInspected)),
    }
}

/// 状態しか分からない呼び出し元（状態確認コマンド等）向けの案内。
///
/// `JobState` にバリアントが増えたとき、判断を強制するため網羅的に列挙する。
#[must_use]
pub const fn guidance_for_state(state: JobState) -> Option<RestartGuidance> {
    match state {
        JobState::Cancelled | JobState::Failed => {
            Some(RestartGuidance::new(OutDirAction::NotInspected))
        }
        JobState::Queued | JobState::Running | JobState::Cancelling | JobState::Succeeded => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::time::Duration;

    const ALL_RESIDUES: [OutDirResidue; 4] = [
        OutDirResidue::Absent,
        OutDirResidue::EmptyReservation,
        OutDirResidue::NonEmpty,
        OutDirResidue::Unknown,
    ];

    fn assert_fixed(g: &RestartGuidance) {
        assert!(!g.resumable());
        assert_eq!(g.action(), RestartAction::RestartFromScratch);
        assert_eq!(g.reason_code(), "resume_not_supported");
    }

    /// REQ-34・TASK-34.3: 全ての止め方と残置の組で再開不可・やり直しになる。
    #[test]
    fn req34_cancelled_all_combinations() {
        let d = Duration::ZERO;
        for stop in [CancelStop::BeforeStart, CancelStop::Cooperative] {
            let g = guidance_for_cancelled(&CancelledRun::for_test(stop, None, d));
            assert_fixed(&g);
            assert_eq!(g.out_dir_action(), OutDirAction::ReuseAllowed);
        }
        for stop in [CancelStop::ForcedKill, CancelStop::Unconfirmed] {
            let g = guidance_for_cancelled(&CancelledRun::for_test(stop, None, d));
            assert_fixed(&g);
            assert_eq!(g.out_dir_action(), OutDirAction::NotInspected);
            let expected = [
                OutDirAction::ReuseAllowed,
                OutDirAction::RemoveEmptyOrChooseNew,
                OutDirAction::DoNotReuseInspectManually,
                OutDirAction::DoNotReuseInspectManually,
            ];
            for (residue, want) in ALL_RESIDUES.into_iter().zip(expected) {
                let g = guidance_for_cancelled(&CancelledRun::for_test(stop, Some(residue), d));
                assert_fixed(&g);
                assert_eq!(g.out_dir_action(), want);
            }
        }
    }

    /// REQ-34・TASK-34.3: JSON は具体値で固定される（`resumable` は false）。
    #[test]
    fn req34_json_snapshots() {
        let cases = [
            (
                OutDirAction::ReuseAllowed,
                "reuse_allowed",
                "Resume is not supported. Restart the job from scratch; the same out_dir can be reused.",
            ),
            (
                OutDirAction::RemoveEmptyOrChooseNew,
                "remove_empty_or_choose_new",
                "Resume is not supported. Restart the job from scratch; remove the empty out_dir left behind or choose a new out_dir.",
            ),
            (
                OutDirAction::DoNotReuseInspectManually,
                "do_not_reuse_inspect_manually",
                "Resume is not supported. Restart the job from scratch; do not reuse the out_dir, inspect it manually and choose a new out_dir.",
            ),
            (
                OutDirAction::NotInspected,
                "not_inspected",
                "Resume is not supported. Restart the job from scratch; the state of out_dir was not inspected, so check it before reuse.",
            ),
        ];
        for (action, name, message) in cases {
            let value = serde_json::to_value(RestartGuidance::new(action)).expect("serialize");
            assert_eq!(
                value,
                json!({
                    "resumable": false,
                    "action": "restart_from_scratch",
                    "reason_code": "resume_not_supported",
                    "out_dir_action": name,
                    "message": message,
                })
            );
        }
    }

    /// REQ-34・TASK-34.3: PoC-19 の jobctl はチェックポイントがあれば
    /// `resumable: true` を返したが、本実装は Failed・Cancelled とも false。
    #[test]
    fn req34_state_guidance_table() {
        let table = [
            (JobState::Queued, false),
            (JobState::Running, false),
            (JobState::Cancelling, false),
            (JobState::Succeeded, false),
            (JobState::Cancelled, true),
            (JobState::Failed, true),
        ];
        for (state, some) in table {
            let g = guidance_for_state(state);
            assert_eq!(g.is_some(), some, "{state:?}");
            if let Some(g) = g {
                assert_fixed(&g);
                assert_eq!(g.out_dir_action(), OutDirAction::NotInspected);
            }
        }
    }

    /// REQ-34・TASK-34.3: 再開の要求は必ず拒否され、案内を返す。
    #[test]
    fn req34_request_resume_always_refused() {
        match request_resume(OutDirAction::ReuseAllowed) {
            Ok(never) => match never {},
            Err(refused) => assert_fixed(&refused.guidance()),
        }
    }

    /// REQ-34・TASK-34.3: 実行結果からの写像。
    #[test]
    fn req34_guidance_for_run() {
        let failed = Ok(TrainRunEnd::Completed(
            crate::process::TrainRun::for_test_error(),
        ));
        let g = guidance_for_run(&failed).expect("some");
        assert_fixed(&g);
        assert_eq!(g.out_dir_action(), OutDirAction::NotInspected);

        let cancelled = Ok(TrainRunEnd::Cancelled(CancelledRun::for_test(
            CancelStop::ForcedKill,
            Some(OutDirResidue::NonEmpty),
            Duration::ZERO,
        )));
        let g = guidance_for_run(&cancelled).expect("some");
        assert_eq!(g.out_dir_action(), OutDirAction::DoNotReuseInspectManually);

        let unconfirmed = Err(TrainProcessError::CancelOutcomeUnconfirmed {
            residue: OutDirResidue::EmptyReservation,
        });
        let g = guidance_for_run(&unconfirmed).expect("some");
        assert_eq!(g.out_dir_action(), OutDirAction::RemoveEmptyOrChooseNew);

        let timeout = Err(TrainProcessError::WallTimeout {
            limit_ms: 1,
            child_reaped: true,
        });
        let g = guidance_for_run(&timeout).expect("some");
        assert_eq!(g.out_dir_action(), OutDirAction::NotInspected);
    }
}
