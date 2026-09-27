//! 終了コード契約（REQ-21・TASK-21.1）。
//!
//! CLI の 7 工程（`register → inspect → train → evaluate → select → package
//! → infer`）・推論経路すべてが共有する、固定 7 種の終了コードを表す型と、
//! 機械可読な `code` / `message` の JSON エラー型を提供する
//! （`.claude/rules/coding-rust.md`「エラーは 7 種の終了コードと機械可読な
//! `code` / `message` の JSON に揃える」）。
//!
//! 出典は PoC-16 の `core/src/exitcode.rs`
//! （`docs/spec/03-poc/core-cli-vertical-slice/core/src/exitcode.rs`）だが、
//! PoC-16 は `i32` 定数 + `name()` に `_ => "unknown"` を許す実装であり、
//! 「判定結果・終了コード・状態は enum で表し、壊れた値を表現できない型に
//! する」（`.claude/rules/coding-rust.md`）に反するため、そのまま移植せず
//! `enum` として作り直している。
//!
//! # 本モジュールのスコープ
//!
//! ここで定義するのは終了コードの型と最小の JSON エラー型のみ。CLI の各工
//! 程が `ok` 時に実際に出す判定結果 JSON（選択肢 ID・スコア等）の配線は
//! TASK-21.1-2（TASK-33.2）の対象で、本モジュールには含めない。PoC-16 の
//! `fail()` が出していた `step` / `status` / `exit_code_name` / `error` と
//! いった CLI 工程ごとの出力契約もここでは踏襲しない。
//!
//! # セキュリティ上の注意
//!
//! [`ErrorReport::message`] は英語の人が読むための文字列で、資格情報・学習
//! / 評価データ本文などの生データを含めない（`.claude/rules/security.md`）。

use std::fmt;

/// REQ-21 で固定された 7 種類の終了コード。
///
/// CLI・MCP・TUI（操作アダプター）が返す `std::process::ExitCode` と、
/// エラー JSON の `code` フィールドの両方をこの型で表す。`#[repr(u8)]` で
/// 実プロセスの終了コード値と 1 対 1 に対応させ、`match` を全 variant 網羅
/// にすることで「未知の終了コード」を型レベルで排除する（PoC-16 の
/// `_ => "unknown"` のような黙殺を作らない）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum ExitCode {
    /// 正常終了。
    Ok = 0,
    /// 判定結果が「不合格」であることを表す終了（REQ-21）。
    JudgedFail = 10,
    /// 入力が判定対象の範囲外であることを表す終了。
    OutOfScope = 11,
    /// 判定が保留（未確定）であることを表す終了。
    Pending = 12,
    /// 資源上限（サイズ・件数・時間等）を超えたことを表す終了。
    LimitExceeded = 20,
    /// 外部入力（定義ファイル・CLI 引数・MCP リクエスト等）が不正。
    InvalidInput = 64,
    /// 上記以外の実行時エラー。
    RuntimeError = 70,
}

impl ExitCode {
    /// 7 種類すべてを列挙する。Rust ⇔ 学習ワーカー
    /// （`trainer/src/fandhe_edge_trainer/exitcode.py`）の値一致を fixture
    /// で照合するテスト（#179）が全件を走査するために公開する。
    pub const ALL: [ExitCode; 7] = [
        ExitCode::Ok,
        ExitCode::JudgedFail,
        ExitCode::OutOfScope,
        ExitCode::Pending,
        ExitCode::LimitExceeded,
        ExitCode::InvalidInput,
        ExitCode::RuntimeError,
    ];

    /// プロセス終了コードとしての数値（REQ-21 の値そのもの）。
    #[must_use]
    pub const fn code(self) -> u8 {
        self as u8
    }

    /// JSON `code` フィールド・ログに出す機械可読な名前。
    ///
    /// 学習ワーカー側ミラー（`trainer/src/fandhe_edge_trainer/exitcode.py`）
    /// と同じ語彙にする。全 variant を網羅した `match` にして、PoC-16 の
    /// ような未知値フォールバックを持たない。
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            ExitCode::Ok => "ok",
            ExitCode::JudgedFail => "judged_fail",
            ExitCode::OutOfScope => "out_of_scope",
            ExitCode::Pending => "pending",
            ExitCode::LimitExceeded => "limit_exceeded",
            ExitCode::InvalidInput => "invalid_input",
            ExitCode::RuntimeError => "runtime_error",
        }
    }
}

/// 未知の終了コード値。[`ExitCode`] の 7 種以外の `i32` から変換しようと
/// したときに返る（外部入力の経路では `unwrap` / `panic` しない。
/// `.claude/rules/coding-rust.md`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnknownExitCode(pub i32);

impl fmt::Display for UnknownExitCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unknown exit code: {}", self.0)
    }
}

impl std::error::Error for UnknownExitCode {}

impl TryFrom<i32> for ExitCode {
    type Error = UnknownExitCode;

    /// 子プロセス（学習ワーカー等）の終了コード
    /// （`std::process::ExitStatus::code() -> Option<i32>`）のような外部由
    /// 来の値を、fail-closed に `ExitCode` へ写す。7 種以外の値はすべて
    /// `Err` にする。
    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(ExitCode::Ok),
            10 => Ok(ExitCode::JudgedFail),
            11 => Ok(ExitCode::OutOfScope),
            12 => Ok(ExitCode::Pending),
            20 => Ok(ExitCode::LimitExceeded),
            64 => Ok(ExitCode::InvalidInput),
            70 => Ok(ExitCode::RuntimeError),
            other => Err(UnknownExitCode(other)),
        }
    }
}

impl From<ExitCode> for std::process::ExitCode {
    /// CLI（操作アダプター）の `main` 関数の戻り値として使うための変換。
    fn from(value: ExitCode) -> Self {
        std::process::ExitCode::from(value.code())
    }
}

/// 機械可読なエラー JSON（`code` / `message`）。
///
/// `code` は [`ExitCode`] を `#[serde(rename_all = "snake_case")]` で文字
/// 列化したもの（例: `"invalid_input"`）。`message` は英語の人が読むため
/// の文字列で、生の評価 / 学習データ・資格情報・秘匿すべきパス以外の情報
/// に留める（`.claude/rules/security.md`）。
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ErrorReport {
    pub code: ExitCode,
    pub message: String,
}

impl ErrorReport {
    /// `code` と `message` から [`ErrorReport`] を組み立てる。
    pub fn new(code: ExitCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-21: 7 種類の終了コードの数値を個別に固定する回帰テスト。
    #[test]
    fn req21_exit_code_values_are_fixed() {
        assert_eq!(ExitCode::Ok.code(), 0);
        assert_eq!(ExitCode::JudgedFail.code(), 10);
        assert_eq!(ExitCode::OutOfScope.code(), 11);
        assert_eq!(ExitCode::Pending.code(), 12);
        assert_eq!(ExitCode::LimitExceeded.code(), 20);
        assert_eq!(ExitCode::InvalidInput.code(), 64);
        assert_eq!(ExitCode::RuntimeError.code(), 70);
    }

    /// REQ-21: `name()` が学習ワーカー側ミラーと同じ語彙になることを固定する。
    #[test]
    fn req21_exit_code_names_are_fixed() {
        assert_eq!(ExitCode::Ok.name(), "ok");
        assert_eq!(ExitCode::JudgedFail.name(), "judged_fail");
        assert_eq!(ExitCode::OutOfScope.name(), "out_of_scope");
        assert_eq!(ExitCode::Pending.name(), "pending");
        assert_eq!(ExitCode::LimitExceeded.name(), "limit_exceeded");
        assert_eq!(ExitCode::InvalidInput.name(), "invalid_input");
        assert_eq!(ExitCode::RuntimeError.name(), "runtime_error");
    }

    /// REQ-21: `ALL` が 7 種すべてを重複なく含むこと。
    #[test]
    fn req21_all_contains_seven_variants() {
        assert_eq!(ExitCode::ALL.len(), 7);
        let mut codes: Vec<u8> = ExitCode::ALL.iter().map(|c| c.code()).collect();
        codes.sort_unstable();
        assert_eq!(codes, vec![0, 10, 11, 12, 20, 64, 70]);
    }

    /// REQ-21: serde の snake_case 表現が手書き `name()` と食い違わないこと
    /// を防ぐ回帰テスト。
    #[test]
    fn req21_serde_name_matches_name_method() {
        for exit_code in ExitCode::ALL {
            let json = serde_json::to_string(&exit_code).expect("serialize ExitCode");
            let expected = format!("\"{}\"", exit_code.name());
            assert_eq!(json, expected);
        }
    }

    /// REQ-21: 正しい 7 つの値はすべて round trip し、元の variant に戻る。
    #[test]
    fn req21_try_from_i32_round_trips_known_values() {
        for exit_code in ExitCode::ALL {
            let value = i32::from(exit_code.code());
            let parsed = ExitCode::try_from(value).expect("known exit code must parse");
            assert_eq!(parsed, exit_code);
        }
    }

    /// REQ-21: 未知の値は `Err` になる（fail-closed）。
    #[test]
    fn req21_try_from_i32_rejects_unknown_values() {
        for unknown in [1, 13, 21, 63, 71, 255, -1] {
            let result = ExitCode::try_from(unknown);
            assert_eq!(result, Err(UnknownExitCode(unknown)));
        }
    }

    /// REQ-21: `ErrorReport` のシリアライズがフィールド順を含め厳密に一致
    /// すること。
    #[test]
    fn req21_error_report_serializes_to_expected_json() {
        let report = ErrorReport::new(ExitCode::InvalidInput, "missing required field: labels");
        let json = serde_json::to_string(&report).expect("serialize ErrorReport");
        assert_eq!(
            json,
            r#"{"code":"invalid_input","message":"missing required field: labels"}"#
        );
    }
}
