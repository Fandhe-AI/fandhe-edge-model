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
//! - [`invariance`][]: モデルパッケージ（重み・語彙・校正・しきい値）の
//!   評価前後ハッシュ比較（REQ-27 正常系・TASK-27.1-1・issue #69）
//!
//! # 現状（実装済みを装わない）
//!
//! - Wilson 95% 信頼区間・sklearn 照合: 未実装（TASK-24.1-2・issue #60）
//! - 分母 0 の指標の JSON `null` 表示・平均から除いたラベルの列挙
//!   （`excluded_labels`）: 未実装（TASK-24.2・issue #61）。本 crate は
//!   `Option<f64>` として `None` を返すところまでを担う
//! - `type_meaning_quadrant`（型と意味の正しさの分離集計）: 未実装
//!   （TASK-24.3・issue #62）。[`metrics::Outcome`] を enum にしてあるのは
//!   この後続実装が予測の種類を判定しやすくするため
//! - McNemar 検定・Holm 補正（REQ-24・REQ-25）: 未実装（TASK-25.x）。
//!   同じ crate 内の後続モジュールとして追加する想定
//! - coverage・abstain_rate・error_rate 等のレポート系（REQ-29）: 未実装
//!   （TASK-29.x）。[`metrics::SingleSelectMetrics::outcome_counts`] の件数を
//!   材料にして上位層が算出する
//! - 評価データのハッシュの前後比較・不一致時の停止（REQ-17・TASK-17.3）への
//!   接続: 未実装（issue #70）。[`invariance`] と同じ
//!   [`fandhe_edge_core::hash::Sha256Digest`] とスナップショット比較の形を
//!   再利用できる想定
//! - 推論関数へ `input` 以外を渡さないことの記録・検査（TASK-27.2。PoC-9
//!   `ArgumentRecordingPredictor` 相当）: 未実装
//! - 凍結した最終 test への 1 回限り適用の強制（TASK-27.3）: 未実装
//! - パッケージ全体を 1 つにまとめた合成ダイジェスト・配布パッケージの
//!   マニフェスト形式（REQ-30・TASK-30.x）: 未実装。[`invariance`] は
//!   構成要素ごとのダイジェストの集合までを提供し、配布パッケージ形式の
//!   契約は先取りしない
//! - CLI `evaluate` 工程への配線（issue #140）: 未実装
//!
//! # 層の境界・不変条件
//!
//! - 外部 crate への直接依存は持たない。workspace 内の `fandhe-edge-core` には
//!   [`invariance`] が使う生バイト列の sha256 ダイジェスト型
//!   （`fandhe_edge_core::hash::Sha256Digest`）のためだけに依存する
//!   （`Cargo.toml` の `[dependencies]` を参照）
//! - ラベルは ID の文字列スライスで受け取り、`fandhe-edge-core::definition` には
//!   依存しない。呼び出し側が `Definition::options()` の `id` を宣言順で渡す
//! - 予測ファイル（JSONL）の読み込み・`status` 文字列の正規化・モデルパッケージの
//!   ファイル読み込みはデータ契約層・CLI 層の責務であり、本 crate は型付きの
//!   メモリ上のスライスだけを受け取る（REQ-27: 評価の前後でモデル・評価データの
//!   ハッシュが一致すること。本 crate は入力を参照でのみ受け取り、書き換えない）
//! - レコード件数・ファイルサイズの上限検証（REQ-39）は呼び出し側（データ検査層・
//!   ガード層）の責務。ラベル（選択肢）数の上限検証は本 crate 自身が行う
//!   （[`metrics::evaluate_single_select`] がラベル索引の構築前に
//!   [`metrics::MAX_LABELS`] を検証し、超過時は確保せず
//!   [`metrics::EvalError::TooManyLabels`] を返す。詳細は [`metrics`]
//!   モジュールの「資源上限」節を参照）
pub mod invariance;
pub mod metrics;
