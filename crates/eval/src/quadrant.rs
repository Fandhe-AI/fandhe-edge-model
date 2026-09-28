//! 型の正しさと意味の正しさの分離集計（REQ-24 境界値・TASK-24.3・issue #62）。
//!
//! 評価契約（`.claude/rules/evaluation-contract.md`「有意性・指標」）は
//! 「型の正しさと意味の正しさは別々に数える（`type_meaning_quadrant`）」を
//! 不変条件としている。PoC-9（`03-poc/evaluation-contract` manifest.json
//! `type_and_meaning_definition`）の定義に従うと、単一選択（single-select）
//! では型不正の行は意味を判定できないため、常に「意味 NG」として扱う。
//! そのため単一選択の区分は次の 5 セルになる（構造上 0 になる
//! 「型不正だが意味は正しい」セルは作らない）:
//!
//! - `type_ok_meaning_ok`: 型が正しく（宣言済みラベルへの予測）、意味も正しい（gold と一致）
//! - `type_ok_meaning_ng`: 型が正しいが、意味が誤り（gold と不一致）
//! - `type_ng_count`: 型が不正（未知ラベル・空文字列・[`crate::metrics::Outcome::Invalid`]）
//! - `abstain`: 判定保留（[`crate::metrics::Outcome::Abstain`]）
//! - `error`: 推論エラー（[`crate::metrics::Outcome::Error`]）
//!
//! # multi-item（複数項目選択）の拡張点（未実装）
//!
//! PoC-9 の multi-item モードでは「型は不正だが意味（intent レベル）は正しい」
//! （`type_ng_intent_ok`）というセルが存在し、`type_ng_count` を
//! `type_ng_intent_ok` / `type_ng_intent_ng` の 2 つに分割した 4 区分＋
//! abstain/error になる（PoC-9 known/multi-item の実測値: 3/3/2/0/1/1）。
//! 本 crate が受け取る [`crate::metrics::Outcome`] は単一選択専用で、
//! `fandhe-edge-core::JudgmentType` も現状 `SingleSelect` のみを扱い
//! `multi_select` を `OutOfScope` として拒否するため、この分割は
//! core が multi-item に対応する時点までの後続事項とする（本モジュールでは
//! 実装しない。issue #62 実装計画「本 PR で行わないこと」）。
//!
//! # 呼び出し元・層の境界
//!
//! [`crate::metrics::evaluate_single_select`] の集計ループから呼ばれる。
//! JSON への直列化は CLI の `evaluate` 工程（TASK-33.x）の責務であり、
//! 本モジュールはフィールド名を PoC-9 の JSON キーに 1:1 で揃えるところ
//! までを担う。
//!
//! `status` が ok/abstain/error のいずれでもない行（PoC-9 追補 A-3 の
//! unknown_status）は、CLI 層・データ契約層で
//! [`crate::metrics::Outcome::Invalid`] へ正規化してから本 crate へ渡す前提
//! とする（本 crate は型付きのスライスだけを受け取る。`lib.rs` の既存方針）。
//!
//! 証拠の種別: テストハーネス（PoC-9 の手計算値・評価器出力の転記との照合。
//! 実機計測なし）。

use crate::metrics::EvalError;

/// 1 件の予測結果をどの区分（セル）へ数えるかを表す。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeMeaningCell {
    /// 型が正しく、意味も正しい（gold と一致）。
    TypeOkMeaningOk,
    /// 型は正しいが、意味が誤り（gold と不一致）。
    TypeOkMeaningNg,
    /// 型が不正（未知ラベル・空文字列・[`crate::metrics::Outcome::Invalid`]）。
    TypeNg,
    /// 判定保留。
    Abstain,
    /// 推論エラー。
    Error,
}

/// 型と意味の正しさの分離集計結果。
///
/// 外部から任意の値で構築させない（フィールドは非公開）。合計が
/// 評価件数（`n_total`）と一致するという不変条件を壊せないようにするため
/// （`.claude/rules/coding-rust.md`「壊れた値を表現できない型にする」。
/// [`crate::metrics::Ratio`] と同じ流儀）。生成・加算は [`TypeMeaningQuadrant::increment`]
/// に集約し、[`crate::metrics::evaluate_single_select`] がループ後に
/// [`TypeMeaningQuadrant::total`] と `n_total` の一致を検証する。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TypeMeaningQuadrant {
    type_ok_meaning_ok: u64,
    type_ok_meaning_ng: u64,
    type_ng_count: u64,
    abstain: u64,
    error: u64,
}

impl TypeMeaningQuadrant {
    /// 型が正しく意味も正しい件数。
    pub fn type_ok_meaning_ok(&self) -> u64 {
        self.type_ok_meaning_ok
    }

    /// 型は正しいが意味が誤りの件数。
    pub fn type_ok_meaning_ng(&self) -> u64 {
        self.type_ok_meaning_ng
    }

    /// 型が不正の件数（意味は判定不能。multi-item 対応前の単一選択では
    /// この 1 セルにまとめる。モジュールドキュメント参照）。
    pub fn type_ng_count(&self) -> u64 {
        self.type_ng_count
    }

    /// 判定保留の件数。
    pub fn abstain(&self) -> u64 {
        self.abstain
    }

    /// 推論エラーの件数。
    pub fn error(&self) -> u64 {
        self.error
    }

    /// 指定したセルの件数を返す。
    pub fn get(&self, cell: TypeMeaningCell) -> u64 {
        match cell {
            TypeMeaningCell::TypeOkMeaningOk => self.type_ok_meaning_ok,
            TypeMeaningCell::TypeOkMeaningNg => self.type_ok_meaning_ng,
            TypeMeaningCell::TypeNg => self.type_ng_count,
            TypeMeaningCell::Abstain => self.abstain,
            TypeMeaningCell::Error => self.error,
        }
    }

    /// 5 セルの合計。桁あふれ時は `None`（評価済みを装わず fail-closed）。
    pub fn total(&self) -> Option<u64> {
        self.type_ok_meaning_ok
            .checked_add(self.type_ok_meaning_ng)
            .and_then(|v| v.checked_add(self.type_ng_count))
            .and_then(|v| v.checked_add(self.abstain))
            .and_then(|v| v.checked_add(self.error))
    }

    /// 指定したセルを 1 件加算する。`record_index` はエラーメッセージにのみ
    /// 使う（`records` 内での位置）。桁あふれは
    /// [`EvalError::Internal`] を返す（fail-closed。`.claude/rules/coding-rust.md`
    /// 「エラーハンドリング・外部入力」）。
    pub(crate) fn increment(
        &mut self,
        cell: TypeMeaningCell,
        record_index: usize,
    ) -> Result<(), EvalError> {
        let slot = match cell {
            TypeMeaningCell::TypeOkMeaningOk => &mut self.type_ok_meaning_ok,
            TypeMeaningCell::TypeOkMeaningNg => &mut self.type_ok_meaning_ng,
            TypeMeaningCell::TypeNg => &mut self.type_ng_count,
            TypeMeaningCell::Abstain => &mut self.abstain,
            TypeMeaningCell::Error => &mut self.error,
        };
        *slot = slot.checked_add(1).ok_or_else(|| EvalError::Internal {
            detail: format!("type_meaning_quadrant overflow at record index {record_index}"),
        })?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-24 境界値・TASK-24.3: 各セルへ 1 回ずつ加算すると、各 getter が
    /// 1 を返し、合計が 5 になる。
    #[test]
    fn increment_each_cell_once_updates_getters_and_total() {
        let mut quadrant = TypeMeaningQuadrant::default();
        quadrant
            .increment(TypeMeaningCell::TypeOkMeaningOk, 0)
            .unwrap();
        quadrant
            .increment(TypeMeaningCell::TypeOkMeaningNg, 1)
            .unwrap();
        quadrant.increment(TypeMeaningCell::TypeNg, 2).unwrap();
        quadrant.increment(TypeMeaningCell::Abstain, 3).unwrap();
        quadrant.increment(TypeMeaningCell::Error, 4).unwrap();

        assert_eq!(quadrant.type_ok_meaning_ok(), 1);
        assert_eq!(quadrant.type_ok_meaning_ng(), 1);
        assert_eq!(quadrant.type_ng_count(), 1);
        assert_eq!(quadrant.abstain(), 1);
        assert_eq!(quadrant.error(), 1);
        assert_eq!(quadrant.get(TypeMeaningCell::TypeOkMeaningOk), 1);
        assert_eq!(quadrant.total(), Some(5));
    }

    /// REQ-39・TASK-24.3: 単一セルの桁あふれは `checked_add` で検出し、
    /// fail-closed で `EvalError::Internal` を返す（panic させない）。
    #[test]
    fn increment_overflow_on_single_cell_returns_internal_error() {
        let mut quadrant = TypeMeaningQuadrant {
            type_ok_meaning_ok: u64::MAX,
            ..TypeMeaningQuadrant::default()
        };
        let result = quadrant.increment(TypeMeaningCell::TypeOkMeaningOk, 0);
        assert_eq!(
            result,
            Err(EvalError::Internal {
                detail: "type_meaning_quadrant overflow at record index 0".to_string(),
            })
        );
    }

    /// REQ-39・TASK-24.3: 合計の桁あふれ（2 セルを `u64::MAX` にする）は
    /// `total()` が `None` を返す。
    #[test]
    fn total_overflow_returns_none() {
        let quadrant = TypeMeaningQuadrant {
            type_ok_meaning_ok: u64::MAX,
            type_ok_meaning_ng: 1,
            ..TypeMeaningQuadrant::default()
        };
        assert_eq!(quadrant.total(), None);
    }
}
