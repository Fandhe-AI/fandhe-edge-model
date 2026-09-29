//! 正常系（exit 0）の工程結果 JSON の型と直列化（REQ-21・REQ-33・TASK-33.2-2・#139）。
//!
//! # 呼び出し文脈
//!
//! CLI の各工程（`register → … → infer`）が exit 0 で終わるとき、stdout へ出す
//! 「工程ごとの結果フィールドを持つ JSON オブジェクト」の型をここに置く。異常系
//! （exit ≠ 0）の `{"code","message"}` は [`crate::exitcode::ErrorReport`] が担い、
//! 本モジュールはそれを変更しない。cli crate は `serde_json` に依存しないため、
//! 直列化を core 側に閉じる（`ErrorReport::to_json_line` と同じ理由。
//! `.claude/rules/dependency-policy.md`）。I/O は行わない。
//!
//! # 現状（実装済みの範囲）
//!
//! `package` 工程の [`PackageReport`] のみ。フィールドは PoC-16 の package 工程の
//! 出力名（`step`・`status`・`judgment`・`acceptance_defined`）に揃えた最小集合で、
//! パス・データ本文・計測値は載せない（security.md。容量・p95 等の追加は各結線
//! TASK で main が判断する入出力契約の変更）。加えて `evaluate` 工程の評価データ
//! 未定義時の [`EvaluateReport`]（`status:"skipped"`。TASK-33.3・#140）を持つ。
//! 評価完了の結果型（指標を持つ）は評価器の結線 TASK で追加する（未実装）。
//!
//! 合否判定は exit 0 になる `pass` のみを表す。`fail`・`limit_exceeded`・判定不能は
//! exit ≠ 0 であり `ErrorReport` 側へ流すため、本型では表現できない（壊れた値を
//! 表現できない型にする。coding-rust.md）。

use serde::Serialize;

/// CLI の 7 工程（REQ-33。工程順）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    /// 定義ファイルの登録。
    Register,
    /// データ検査。
    Inspect,
    /// 学習。
    Train,
    /// 評価。
    Evaluate,
    /// 選定。
    Select,
    /// 配布パッケージ化。
    Package,
    /// 推論。
    Infer,
}

/// 工程の状態（`ok`・`skipped`）。`skipped` は評価データ未定義の `evaluate` のみ（REQ-17・TASK-33.3）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StageStatus {
    /// 工程が完了した。
    Ok,
    /// 工程を実行せず終えた（exit 0。評価済みを装わない）。
    Skipped,
}

/// `evaluate` を skipped で終えた理由（機械可読な英語の固定語彙）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluateSkipReason {
    /// 評価データが定義されていない（REQ-17）。
    EvaluationDataNotDefined,
}

/// `evaluate` 工程が評価データ未定義で返す JSON（REQ-17・REQ-33・TASK-33.3・#140）。
///
/// 指標・合否のフィールドを持たず、コンストラクタは [`Self::skipped`] のみのため
/// 「評価済みを装う」値を作れない。フィールドは宣言順（`step`・`status`・`reason`）に直列化する。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EvaluateReport {
    step: Stage,
    status: StageStatus,
    reason: EvaluateSkipReason,
}

impl EvaluateReport {
    /// 評価データ未定義の skipped 結果（exit 0。PoC-16 縦断 2 の `status:"skipped"`）。
    #[must_use]
    pub const fn skipped() -> Self {
        Self {
            step: Stage::Evaluate,
            status: StageStatus::Skipped,
            reason: EvaluateSkipReason::EvaluationDataNotDefined,
        }
    }

    /// JSON 1 行（末尾の改行なし）へ直列化する。[`PackageReport::to_json_line`] と対称。
    ///
    /// # Errors
    /// `serde_json` 側の直列化エラーをそのまま返す。
    pub fn to_json_line(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

/// 合否判定（exit 0 になるものだけ）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PackageJudgment {
    /// 合否基準を満たした。
    Pass,
}

/// `package` 工程が exit 0 で返す JSON（フィールドは宣言順に直列化する）。
///
/// フィールドは非公開で、コンストラクタ経由でのみ作る。`step`・`status` を固定し、
/// `judgment` と `acceptance_defined` の矛盾した組み合わせを作れないようにする。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PackageReport {
    step: Stage,
    status: StageStatus,
    /// 合否基準が未設定のときは `null`（`skip_serializing_if` を付けずスキーマを固定する）。
    judgment: Option<PackageJudgment>,
    acceptance_defined: bool,
}

impl PackageReport {
    /// 合否基準を満たした結果（PoC-16 実測の `judgment:"pass"`）。
    #[must_use]
    pub const fn pass() -> Self {
        Self {
            step: Stage::Package,
            status: StageStatus::Ok,
            judgment: Some(PackageJudgment::Pass),
            acceptance_defined: true,
        }
    }

    /// 合否基準が未設定の結果（`judgment` は `null`。exit 0）。
    #[must_use]
    pub const fn acceptance_not_defined() -> Self {
        Self {
            step: Stage::Package,
            status: StageStatus::Ok,
            judgment: None,
            acceptance_defined: false,
        }
    }

    /// JSON 1 行（末尾の改行なし）へ直列化する。
    ///
    /// [`crate::exitcode::ErrorReport::to_json_line`] と対称の API。改行は
    /// 呼び出し側（cli の出力関数）が付ける。
    ///
    /// # Errors
    /// `serde_json` 側の直列化エラーをそのまま返す。
    pub fn to_json_line(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-33: Pass の JSON が PoC-16 の名前・値と完全一致する。
    #[test]
    fn req33_pass_report_json_is_exact() {
        assert_eq!(
            PackageReport::pass().to_json_line().expect("json"),
            r#"{"step":"package","status":"ok","judgment":"pass","acceptance_defined":true}"#
        );
    }

    /// REQ-33: 合否基準未設定は `judgment:null`・`acceptance_defined:false`。
    #[test]
    fn req33_not_defined_report_json_is_exact() {
        assert_eq!(
            PackageReport::acceptance_not_defined()
                .to_json_line()
                .expect("json"),
            r#"{"step":"package","status":"ok","judgment":null,"acceptance_defined":false}"#
        );
    }

    /// REQ-33: 7 工程が snake_case の名前で直列化される。
    #[test]
    fn req33_stage_names_are_snake_case() {
        let all = [
            (Stage::Register, "register"),
            (Stage::Inspect, "inspect"),
            (Stage::Train, "train"),
            (Stage::Evaluate, "evaluate"),
            (Stage::Select, "select"),
            (Stage::Package, "package"),
            (Stage::Infer, "infer"),
        ];
        for (stage, name) in all {
            assert_eq!(
                serde_json::to_string(&stage).expect("json"),
                format!("\"{name}\"")
            );
        }
    }

    /// REQ-33・REQ-17: evaluate の skipped JSON が完全一致する。
    #[test]
    fn req33_evaluate_skipped_report_json_is_exact() {
        assert_eq!(
            EvaluateReport::skipped().to_json_line().expect("json"),
            r#"{"step":"evaluate","status":"skipped","reason":"evaluation_data_not_defined"}"#
        );
    }

    /// REQ-33: skipped の JSON は 1 行。
    #[test]
    fn req33_evaluate_skipped_report_is_single_line() {
        assert!(
            !EvaluateReport::skipped()
                .to_json_line()
                .expect("json")
                .contains('\n')
        );
    }

    /// REQ-33: 工程状態は snake_case。
    #[test]
    fn req33_stage_status_names_are_snake_case() {
        assert_eq!(
            serde_json::to_string(&StageStatus::Ok).expect("json"),
            "\"ok\""
        );
        assert_eq!(
            serde_json::to_string(&StageStatus::Skipped).expect("json"),
            "\"skipped\""
        );
    }

    /// REQ-33: 出力は 1 行（改行を含まない）。
    #[test]
    fn req33_report_is_single_line() {
        assert!(
            !PackageReport::pass()
                .to_json_line()
                .expect("json")
                .contains('\n')
        );
    }
}
