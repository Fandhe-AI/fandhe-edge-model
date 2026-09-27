//! ok 終了時の判定結果型（REQ-21 正常系・TASK-21.1-2）。
//!
//! CLI の `infer` 工程（TASK-33.1。現状は未配線）が単一選択の判定に成功した
//! ときに stdout へ書く JSON 1 行の中身を表す。終了コード（`exitcode`
//! モジュール）とは別モジュールに分けているのは、終了コード自体は 7 種の
//! 状態を表す薄い型であるのに対し、判定結果は「どの選択肢が選ばれ、各選択
//! 肢のスコアは何か」という値を持つ型で、検証すべき不変条件（選択肢 ID の
//! 実在性・スコアの件数と範囲）が exitcode モジュールとは別だからである。
//!
//! # 呼び出し文脈
//!
//! - 呼び出し元（想定・TASK-33.1 で配線）: CLI の `infer` サブコマンド。
//!   [`crate::definition::Definition::options`] と、推論ランタイム
//!   （TASK-30.x/31.x。未実装）が返す確率の列から [`JudgmentResult::new`] を
//!   呼んで組み立てる
//! - 呼び出し先（想定）: `fandhe-edge-cli` の出力関数
//!   （`crates/cli/src/output.rs`）が [`JudgmentResult::to_json_line`] を
//!   stdout へ書き、`ExitCode::Ok` を返す
//!
//! # スキーマ（本 TASK で確定。呼び出し側で変更しない）
//!
//! ```json
//! {"id":"<入力の識別子>","status":"ok","predicted_label":"<Choice.id>","scores":{"<Choice.id>":<f64>,...}}
//! ```
//!
//! - 基準は PoC-16 `PredictionRow`（`{id, status, predicted_label, scores}`）。
//!   REQ-21 の受け入れ基準が「PoC-16 実測相当」のため、PoC-16 のキー名を維持する
//! - `predicted_label` は定義ファイルの `Choice.id`（不変 ID）。`display_name` は
//!   入れない
//! - `scores` のキー順は定義ファイルの `options` の宣言順で固定する（PoC-9
//!   追補 A-10 の majority タイブレークの根拠と同じ理由。`definition.rs` の
//!   `Definition` ドキュメント参照）。`serde_json::Map`／`HashMap` はキー順を
//!   保持しない（`preserve_order` 無しではソートされる）ため使わず、
//!   `Vec<(String, f64)>` を保持して独自の `Serialize` で宣言順に書き出す
//! - `id` には入力本文を入れない（security.md「データ本文を出力しない」）。
//!   長さの上限は [`MAX_INPUT_ID_BYTES`]（暫定値）
//!
//! `JudgmentStatus` の variant は現状 `Ok` のみ。保留・対象外
//! （REQ-22。`abstain`／`out_of_scope` 等）は後続 TASK で追加する。
//!
//! # 検証（fail-closed）
//!
//! [`JudgmentResult`] は `Deserialize` を実装しない。フィールドは非公開に
//! し、検証を通る唯一の構築経路は [`JudgmentResult::new`] のみとする
//! （`definition.rs` の `Definition`／`RawDefinition` の関係に倣う。ガード
//! 層の迂回を防ぐ。security.md）。

use crate::definition::Choice;
use crate::exitcode::ExitCode;
use serde::ser::SerializeMap;
use serde::{Serialize, Serializer};
use std::fmt;

/// `id` フィールドのバイト長上限（暫定値。REQ-39 の資源上限が正式に決まり
/// 次第、値を見直す）。
pub const MAX_INPUT_ID_BYTES: usize = 1024;

/// 判定結果の状態。現状は `Ok`（正常終了）のみを持つ（REQ-21 正常系）。
///
/// `#[non_exhaustive]` にはしない。CLI 側で全 variant を網羅した `match` を
/// 書けるようにし、REQ-22 で variant を追加する際にコンパイルエラーで
/// 気付けるようにするため（`exitcode::ExitCode` と同じ設計）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JudgmentStatus {
    /// 単一選択の判定に成功した（REQ-21 正常系。終了コード `ok` に対応）。
    Ok,
}

/// [`JudgmentResult::new`] が拒否する入力の種類。
///
/// 値そのものではなく種別と選択肢 ID のみを保持する（security.md「秘密情
/// 報の混入防止」: 入力本文・学習/評価データをエラー文へ漏らさない）。
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum JudgmentError {
    /// 定義ファイルの選択肢一覧が空。
    EmptyOptions,
    /// 選択肢 ID が空文字列。
    EmptyChoiceId,
    /// 選択肢 ID が重複している。
    DuplicateChoiceId { id: String },
    /// スコアの件数が選択肢の件数と一致しない。
    ScoreCountMismatch { expected: usize, actual: usize },
    /// スコアが非有限（NaN・±inf）。serde_json は NaN を `null` として書く
    /// ため、「スコアを含む」という契約が黙って壊れるのを防ぐ。
    NonFiniteScore { choice_id: String },
    /// スコアが確率として扱える範囲 `[0.0, 1.0]` の外。
    ScoreOutOfRange { choice_id: String },
    /// `predicted_choice_id` が選択肢一覧に存在しない。
    UnknownPredictedChoice { id: String },
    /// 入力の識別子（`id`）が空文字列。
    EmptyInputId,
    /// 入力の識別子が [`MAX_INPUT_ID_BYTES`] を超える。
    InputIdTooLong { len: usize, limit: usize },
    /// JSON への直列化に失敗した（`serde_json` 側のエラー）。
    Serialize(String),
}

impl fmt::Display for JudgmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JudgmentError::EmptyOptions => write!(f, "options must not be empty"),
            JudgmentError::EmptyChoiceId => write!(f, "choice id must not be empty"),
            JudgmentError::DuplicateChoiceId { id } => {
                write!(f, "duplicate choice id: {id}")
            }
            JudgmentError::ScoreCountMismatch { expected, actual } => {
                write!(f, "score count mismatch: expected {expected}, got {actual}")
            }
            JudgmentError::NonFiniteScore { choice_id } => {
                write!(f, "non-finite score for choice id: {choice_id}")
            }
            JudgmentError::ScoreOutOfRange { choice_id } => {
                write!(
                    f,
                    "score out of range [0.0, 1.0] for choice id: {choice_id}"
                )
            }
            JudgmentError::UnknownPredictedChoice { id } => {
                write!(f, "unknown predicted choice id: {id}")
            }
            JudgmentError::EmptyInputId => write!(f, "input id must not be empty"),
            JudgmentError::InputIdTooLong { len, limit } => {
                write!(f, "input id too long: {len} bytes (limit: {limit})")
            }
            JudgmentError::Serialize(message) => {
                write!(f, "failed to serialize judgment result: {message}")
            }
        }
    }
}

impl std::error::Error for JudgmentError {}

impl JudgmentError {
    /// REQ-21 の終了コードへの写像。入力 ID の不正（利用者が直せる外部入
    /// 力の誤り）は `InvalidInput`、それ以外（ランタイムが返すスコアと定義
    /// ファイルの不整合等、呼び出し側のバグに近い状態）は `RuntimeError`
    /// とする。
    #[must_use]
    pub const fn exit_code(&self) -> ExitCode {
        match self {
            JudgmentError::EmptyInputId | JudgmentError::InputIdTooLong { .. } => {
                ExitCode::InvalidInput
            }
            _ => ExitCode::RuntimeError,
        }
    }
}

/// ok 終了時の判定結果（選択肢 ID・スコア。REQ-21 正常系）。
///
/// フィールドは非公開。検証済みの値を作る経路は [`JudgmentResult::new`] に
/// 限定する（モジュール冒頭のドキュメント参照）。
#[derive(Debug, Clone, PartialEq)]
pub struct JudgmentResult {
    id: String,
    predicted_choice_id: String,
    /// 定義ファイルの `options` 宣言順を保った (選択肢 ID, スコア) の列。
    scores: Vec<(String, f64)>,
}

impl JudgmentResult {
    /// 定義ファイルの選択肢一覧・入力の識別子・推論ランタイムが返した確率
    /// の列（`options` と同じ宣言順）から [`JudgmentResult`] を組み立てる。
    ///
    /// `options` の各フィールドは `pub`（`Choice`）のため、呼び出し側が任
    /// 意のスライスを渡してくる前提ですべて再検証する（添字アクセスは使
    /// わず `iter().zip()`／`get()` で処理する。coding-rust.md「外部入力の
    /// 経路」）。
    ///
    /// # Errors
    /// [`JudgmentError`] の各 variant を参照。
    pub fn new(
        options: &[Choice],
        id: impl Into<String>,
        predicted_choice_id: &str,
        scores: &[f64],
    ) -> Result<Self, JudgmentError> {
        if options.is_empty() {
            return Err(JudgmentError::EmptyOptions);
        }
        if scores.len() != options.len() {
            return Err(JudgmentError::ScoreCountMismatch {
                expected: options.len(),
                actual: scores.len(),
            });
        }

        let id = id.into();
        if id.is_empty() {
            return Err(JudgmentError::EmptyInputId);
        }
        if id.len() > MAX_INPUT_ID_BYTES {
            return Err(JudgmentError::InputIdTooLong {
                len: id.len(),
                limit: MAX_INPUT_ID_BYTES,
            });
        }

        let mut seen_ids: Vec<&str> = Vec::with_capacity(options.len());
        let mut pairs: Vec<(String, f64)> = Vec::with_capacity(options.len());
        for (choice, score) in options.iter().zip(scores.iter()) {
            if choice.id.is_empty() {
                return Err(JudgmentError::EmptyChoiceId);
            }
            if seen_ids.contains(&choice.id.as_str()) {
                return Err(JudgmentError::DuplicateChoiceId {
                    id: choice.id.clone(),
                });
            }
            seen_ids.push(choice.id.as_str());

            if !score.is_finite() {
                return Err(JudgmentError::NonFiniteScore {
                    choice_id: choice.id.clone(),
                });
            }
            if *score < 0.0 || *score > 1.0 {
                return Err(JudgmentError::ScoreOutOfRange {
                    choice_id: choice.id.clone(),
                });
            }

            pairs.push((choice.id.clone(), *score));
        }

        if !seen_ids.contains(&predicted_choice_id) {
            return Err(JudgmentError::UnknownPredictedChoice {
                id: predicted_choice_id.to_string(),
            });
        }

        Ok(Self {
            id,
            predicted_choice_id: predicted_choice_id.to_string(),
            scores: pairs,
        })
    }

    /// 入力の識別子。入力本文は含まない（security.md）。
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// 判定結果の状態。現状は常に `Ok`（TASK-21.1-2 の対象は正常系のみ）。
    #[must_use]
    pub const fn status(&self) -> JudgmentStatus {
        JudgmentStatus::Ok
    }

    /// 選ばれた選択肢の ID（`Choice.id`）。
    #[must_use]
    pub fn predicted_choice_id(&self) -> &str {
        &self.predicted_choice_id
    }

    /// 各選択肢のスコアを、定義ファイルの `options` 宣言順で返す。
    pub fn scores(&self) -> impl Iterator<Item = (&str, f64)> {
        self.scores.iter().map(|(id, score)| (id.as_str(), *score))
    }

    /// 対応する終了コード。判定結果が構築できた時点で常に `ExitCode::Ok`。
    #[must_use]
    pub const fn exit_code(&self) -> ExitCode {
        ExitCode::Ok
    }

    /// JSON 1 行（末尾の改行なし）へ直列化する。
    ///
    /// 呼び出し側（`fandhe-edge-cli` の出力関数）が改行を付けて stdout へ
    /// 書く想定（本 crate は I/O を行わない。層の境界を保つため）。
    ///
    /// # Errors
    /// `serde_json` 側の直列化エラーを [`JudgmentError::Serialize`] として返す。
    pub fn to_json_line(&self) -> Result<String, JudgmentError> {
        serde_json::to_string(self).map_err(|error| JudgmentError::Serialize(error.to_string()))
    }
}

/// スキーマ順（`id` → `status` → `predicted_label` → `scores`）を固定する
/// ための手書き `Serialize`。`derive` にすると `scores` に `HashMap`／
/// `serde_json::Map` を使わない限りフィールド順は宣言順になるが、`scores`
/// 自体のキー順（宣言順）を保つために `Vec<(String, f64)>` を独自の
/// `serialize_map` で書く必要があるため、`JudgmentResult` 全体も手書きに
/// 揃えている。
impl Serialize for JudgmentResult {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;

        struct ScoresMap<'a>(&'a [(String, f64)]);
        impl Serialize for ScoresMap<'_> {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                let mut map = serializer.serialize_map(Some(self.0.len()))?;
                for (id, score) in self.0 {
                    map.serialize_entry(id, score)?;
                }
                map.end()
            }
        }

        let mut state = serializer.serialize_struct("JudgmentResult", 4)?;
        state.serialize_field("id", &self.id)?;
        state.serialize_field("status", &self.status())?;
        state.serialize_field("predicted_label", &self.predicted_choice_id)?;
        state.serialize_field("scores", &ScoresMap(&self.scores))?;
        state.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::definition::Choice;

    fn choice(id: &str) -> Choice {
        Choice {
            id: id.to_string(),
            display_name: id.to_string(),
            description: String::new(),
        }
    }

    /// REQ-21: 3 選択肢の完全一致（PoC-16 `PredictionRow` 相当のキー名）。
    #[test]
    fn req21_ok_judgment_serializes_to_expected_json() {
        let options = [choice("a"), choice("b"), choice("c")];
        let result = JudgmentResult::new(&options, "row-1", "a", &[0.7, 0.2, 0.1]).unwrap();

        assert_eq!(result.id(), "row-1");
        assert_eq!(result.status(), JudgmentStatus::Ok);
        assert_eq!(result.predicted_choice_id(), "a");
        assert_eq!(result.exit_code(), ExitCode::Ok);

        let json = result.to_json_line().unwrap();
        assert_eq!(
            json,
            r#"{"id":"row-1","status":"ok","predicted_label":"a","scores":{"a":0.7,"b":0.2,"c":0.1}}"#
        );
    }

    /// REQ-21: `scores` のキー順は辞書順ではなく `options` の宣言順を保つ。
    #[test]
    fn req21_scores_preserve_declaration_order_not_lexical_order() {
        let options = [choice("z"), choice("a"), choice("m")];
        let result = JudgmentResult::new(&options, "row-2", "z", &[0.5, 0.3, 0.2]).unwrap();

        let json = result.to_json_line().unwrap();
        assert_eq!(
            json,
            r#"{"id":"row-2","status":"ok","predicted_label":"z","scores":{"z":0.5,"a":0.3,"m":0.2}}"#
        );

        let collected: Vec<(&str, f64)> = result.scores().collect();
        assert_eq!(collected, vec![("z", 0.5), ("a", 0.3), ("m", 0.2)]);
    }

    /// REQ-21: `status` の直列化値が `ExitCode::Ok.name()` と一致すること。
    #[test]
    fn req21_status_matches_exit_code_name() {
        let json = serde_json::to_string(&JudgmentStatus::Ok).unwrap();
        assert_eq!(json, format!("\"{}\"", ExitCode::Ok.name()));
    }

    /// REQ-21: 空の `options` は拒否する。
    #[test]
    fn req21_rejects_empty_options() {
        let result = JudgmentResult::new(&[], "row", "a", &[]);
        assert_eq!(result, Err(JudgmentError::EmptyOptions));
    }

    /// REQ-21: スコアの件数不一致（多い・少ない両方）を拒否する。
    #[test]
    fn req21_rejects_score_count_mismatch() {
        let options = [choice("a"), choice("b"), choice("c")];
        assert_eq!(
            JudgmentResult::new(&options, "row", "a", &[0.5, 0.5]),
            Err(JudgmentError::ScoreCountMismatch {
                expected: 3,
                actual: 2
            })
        );
        assert_eq!(
            JudgmentResult::new(&options, "row", "a", &[0.5, 0.3, 0.1, 0.1]),
            Err(JudgmentError::ScoreCountMismatch {
                expected: 3,
                actual: 4
            })
        );
    }

    /// REQ-21: NaN・+inf・-inf を個別に拒否する。
    #[test]
    fn req21_rejects_non_finite_scores() {
        let options = [choice("a"), choice("b")];
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                JudgmentResult::new(&options, "row", "a", &[bad, 0.5]),
                Err(JudgmentError::NonFiniteScore {
                    choice_id: "a".to_string()
                })
            );
        }
    }

    /// REQ-21: 範囲外のスコア（負・1 超）を拒否する。
    #[test]
    fn req21_rejects_out_of_range_scores() {
        let options = [choice("a"), choice("b")];
        assert_eq!(
            JudgmentResult::new(&options, "row", "a", &[-0.1, 0.5]),
            Err(JudgmentError::ScoreOutOfRange {
                choice_id: "a".to_string()
            })
        );
        assert_eq!(
            JudgmentResult::new(&options, "row", "a", &[1.1, 0.5]),
            Err(JudgmentError::ScoreOutOfRange {
                choice_id: "a".to_string()
            })
        );
    }

    /// REQ-21: 未知の `predicted_choice_id` を拒否する。
    #[test]
    fn req21_rejects_unknown_predicted_choice() {
        let options = [choice("a"), choice("b")];
        assert_eq!(
            JudgmentResult::new(&options, "row", "z", &[0.5, 0.5]),
            Err(JudgmentError::UnknownPredictedChoice {
                id: "z".to_string()
            })
        );
    }

    /// REQ-21: 選択肢 ID の重複を拒否する。
    #[test]
    fn req21_rejects_duplicate_choice_id() {
        let options = [choice("a"), choice("a")];
        assert_eq!(
            JudgmentResult::new(&options, "row", "a", &[0.5, 0.5]),
            Err(JudgmentError::DuplicateChoiceId {
                id: "a".to_string()
            })
        );
    }

    /// REQ-21: 選択肢 ID の空文字列を拒否する。
    #[test]
    fn req21_rejects_empty_choice_id() {
        let options = [choice(""), choice("b")];
        assert_eq!(
            JudgmentResult::new(&options, "row", "b", &[0.5, 0.5]),
            Err(JudgmentError::EmptyChoiceId)
        );
    }

    /// REQ-21: 空の入力 ID を拒否する。
    #[test]
    fn req21_rejects_empty_input_id() {
        let options = [choice("a")];
        assert_eq!(
            JudgmentResult::new(&options, "", "a", &[0.5]),
            Err(JudgmentError::EmptyInputId)
        );
    }

    /// REQ-21: 入力 ID の長さ境界値（1024 バイトは受理、1025 バイトは拒否）。
    #[test]
    fn req21_input_id_length_boundary() {
        let options = [choice("a")];
        let at_limit = "x".repeat(MAX_INPUT_ID_BYTES);
        assert!(JudgmentResult::new(&options, at_limit, "a", &[0.5]).is_ok());

        let over_limit = "x".repeat(MAX_INPUT_ID_BYTES + 1);
        assert_eq!(
            JudgmentResult::new(&options, over_limit, "a", &[0.5]),
            Err(JudgmentError::InputIdTooLong {
                len: MAX_INPUT_ID_BYTES + 1,
                limit: MAX_INPUT_ID_BYTES
            })
        );
    }

    /// REQ-21: `id` に二重引用符・改行・制御文字を含む場合でも、
    /// `serde_json` のエスケープにより JSON 1 行（改行 1 つのみ）という
    /// 出力契約が保たれること（Review 指摘: id の値に依らず契約を守る）。
    #[test]
    fn req21_escapes_id_with_quotes_newlines_and_control_chars() {
        let options = [choice("a")];
        let raw_id = "row\"with\nquote\tand\u{0007}control";
        let result = JudgmentResult::new(&options, raw_id, "a", &[1.0]).unwrap();

        assert_eq!(result.id(), raw_id);

        let json = result.to_json_line().unwrap();
        // 出力全体が 1 行（改行を含まない）であること。
        assert_eq!(
            json.matches('\n').count(),
            0,
            "must not contain raw newline"
        );
        assert_eq!(
            json,
            r#"{"id":"row\"with\nquote\tand\u0007control","status":"ok","predicted_label":"a","scores":{"a":1.0}}"#
        );

        // 直列化した JSON をパースし戻すと元の `id` と一致すること
        // （エスケープが可逆であることの確認）。
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["id"], raw_id);
    }

    /// REQ-21: `exit_code()` の写像（入力 ID の不正 → 64、それ以外 → 70）。
    #[test]
    fn req21_error_exit_code_mapping() {
        assert_eq!(
            JudgmentError::EmptyInputId.exit_code(),
            ExitCode::InvalidInput
        );
        assert_eq!(
            JudgmentError::InputIdTooLong {
                len: 2000,
                limit: MAX_INPUT_ID_BYTES
            }
            .exit_code(),
            ExitCode::InvalidInput
        );
        assert_eq!(
            JudgmentError::EmptyOptions.exit_code(),
            ExitCode::RuntimeError
        );
        assert_eq!(
            JudgmentError::ScoreCountMismatch {
                expected: 1,
                actual: 2
            }
            .exit_code(),
            ExitCode::RuntimeError
        );
        assert_eq!(
            JudgmentError::UnknownPredictedChoice {
                id: "z".to_string()
            }
            .exit_code(),
            ExitCode::RuntimeError
        );
    }
}
