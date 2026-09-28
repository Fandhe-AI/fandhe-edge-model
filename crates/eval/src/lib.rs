//! 評価器層（`fandhe-edge-eval`）。
//!
//! 評価器は spec 上「TASK-24.1 の 1 つだけ」に集約する層で、学習ワーカー・
//! 推論ランタイム・CLI から見て評価指標の算出ロジックを再実装させないための
//! 唯一の実装場所となる（`.claude/rules/coding-rust.md`「crate 構成と層の境界」・
//! REQ-18・REQ-25・REQ-26・REQ-27・REQ-29 が本層に依存する）。
//!
//! 本 crate は推論経路には入らない。CLI の `evaluate` 工程（REQ-33）が、
//! データ契約層（`fandhe-edge-data`）で検証済みの gold・推論結果を本 crate の
//! 関数へ渡し、返ってきた指標を JSON へ整形して出力する想定。
//!
//! - [`metrics`][]: 正解率・ラベル別指標（適合率・再現率・F1）・Macro-F1・
//!   混同行列の算出（REQ-24 正常系・TASK-24.1-1・issue #59）
//! - [`mcnemar`][]: McNemar の正確検定（両側）の統計計算コア
//!   （REQ-25 正常系・TASK-25.1-1・issue #64）
//! - [`baseline`][]: 下限基準（majority）の予測生成（REQ-25・TASK-25.1-2・
//!   issue #65）
//! - [`significance`][]: 行ごとの正誤 → McNemar 検定 → α=0.05 での有意性
//!   判定への接続（REQ-25・TASK-25.1-2・issue #65）
//! - [`quadrant`][]: 型と意味の正しさの分離集計（REQ-24 境界値・TASK-24.3・
//!   issue #62。単一選択の 5 区分〔type_ok_meaning_ok / type_ok_meaning_ng /
//!   type_ng_count / abstain / error〕のみ。multi-item は未対応）
//! - [`wilson`][]: Wilson 95% 信頼区間の算出（REQ-24・REQ-26・TASK-24.1-2・
//!   issue #60。sklearn 照合 8/8 は `tests/sklearn_check.rs` を参照。
//!   Wilson 自体は sklearn に実装が無いため sklearn 照合の対象外）
//!
//! # 現状（実装済みを装わない）
//!
//! - 分母 0 の指標の `None`（未定義）表示・Macro-F1 の平均から除いた
//!   ラベルの列挙（[`metrics::MacroF1::excluded_labels`]）: 実装済み
//!   （REQ-24 異常系・TASK-24.2・issue #61）。`None` を JSON の `null` へ
//!   写す直列化は本 crate の責務ではなく、CLI の `evaluate` 工程
//!   （TASK-33.1）が担う
//! - `type_meaning_quadrant`（型と意味の正しさの分離集計）: 単一選択の
//!   3 区分＋abstain/error を実装済み（TASK-24.3・issue #62）。multi-item の
//!   型不正行の意味分割（`type_ng_intent_ok` / `type_ng_intent_ng`）は
//!   未実装（core の multi-item 対応待ち。[`quadrant`] モジュールの
//!   ドキュメントコメント参照）
//! - McNemar 検定（REQ-25）: 統計計算コアは実装済み（TASK-25.1-1・issue #64）。
//!   下限基準（majority）比較への接続・p<0.05 の判定は実装済み
//!   （TASK-25.1-2・issue #65。文字 n-gram 規則等の `simple_rule` 下限基準は
//!   未実装。依存 `unicode-normalization` の承認と入力表現の整合の判断が要る）。
//!   件数不足による「判定不能」の判定は未実装（TASK-25.2）。
//!   複数候補比較の Holm 補正（REQ-26）は未実装（TASK-25.3）
//! - coverage・abstain_rate・error_rate 等のレポート系（REQ-29）: 未実装
//!   （TASK-29.x）。[`metrics::SingleSelectMetrics::outcome_counts`] の件数を
//!   材料にして上位層が算出する
//!
//! # 層の境界・不変条件
//!
//! - 外部依存を持たない（`Cargo.toml` の `[dependencies]` は空）。ラベルは
//!   ID の文字列スライスで受け取り、`fandhe-edge-core::definition` には
//!   依存しない。呼び出し側が `Definition::options()` の `id` を宣言順で渡す
//! - 予測ファイル（JSONL）の読み込み・`status` 文字列の正規化はデータ契約層
//!   ・CLI 層の責務であり、本 crate は型付きのメモリ上のスライスだけを
//!   受け取る（REQ-27: 評価の前後で評価データのハッシュが一致すること。
//!   本 crate は入力を参照でのみ受け取り、書き換えない）
//! - レコード件数の主たる上限検証（REQ-39）は呼び出し側（データ検査層）の
//!   責務。ラベル（選択肢）数の上限検証は本 crate 自身が行う
//!   （[`metrics::evaluate_single_select`] がラベル索引の構築前に
//!   [`metrics::MAX_LABELS`] を検証し、超過時は確保せず
//!   [`metrics::EvalError::TooManyLabels`] を返す。詳細は [`metrics`]
//!   モジュールの「資源上限」節を参照）。ただし [`significance::correctness`]・
//!   [`significance::compare_with_baseline`] は外部入力由来の `records.len()`
//!   を正誤 `Vec` の容量へ直接使うため、確保前に
//!   [`significance::MAX_EVAL_RECORDS`] で件数を拒否する防御層を本 crate 側
//!   にも置く（Review 指摘。TASK-25.1-2・issue #65）
pub mod baseline;
pub mod mcnemar;
pub mod metrics;
pub mod quadrant;
pub mod significance;
pub mod wilson;
