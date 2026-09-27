//! データ契約層（`fandhe-edge-data`）。
//!
//! 利用者が用意した学習・評価データ（JSONL、1 行 1 レコード）の検査
//! （REQ-16）・group 単位分割と凍結（REQ-17）・来歴（REQ-40）を担う層。
//! 本 crate は現時点で以下の 2 つを実装する。
//!
//! - [`inspect`]: 型・必須項目・ラベル enum の検査ロジック
//!   （TASK-16.1（親 issue #37）のうち TASK-16.1-1・issue #38）。件数集計
//!   （行数・ユニーク数・ラベル別件数。TASK-16.1-2・issue #39）は対象外
//! - [`split`]: group 単位分割ロジック（TASK-17.1-1・issue #44）
//!
//! # 現状（実装済みを装わない）
//!
//! - データ検査のうち漏洩・group 跨ぎ・メタデータ混入・矛盾・正規化入力の
//!   重複検出（REQ-16・TASK-16.2）: 未実装
//! - 分割結果のハッシュ計算・記録・凍結（REQ-17・TASK-17.1-2/17.2/17.3）: 未実装
//!   （後続 #45 が [`split`] モジュールの出力を消費してハッシュ化する。
//!   [`inspect::inspect_records`] が返す [`inspect::ValidRecord`] は
//!   [`split::Groupable`] を実装しないため、[`split::split_by_group`] へ渡す際は
//!   呼び出し側（CLI 等）が変換する）
//! - 来歴（REQ-40）: 未実装
//!
//! # 層の境界
//!
//! データ契約層は共通コア（`fandhe-edge-core`）の下位に位置づく想定だが、
//! 本 crate 作成時点（issue #38 実装時）で共通コアの定義ファイルスキーマ
//! （TASK-15.3・issue #32〜#34）が未実装だったため、[`inspect`] は
//! `fandhe-edge-core` には依存せず、有効なラベル ID の集合を呼び出し側
//! （将来的には CLI 等の上位層）から `BTreeSet<String>` として受け取る形で
//! 設計を分離している。`fandhe-edge-core::definition` の型から
//! `BTreeSet<String>` へ変換する 1 行のアダプタを呼び出し側（CLI 等）に
//! 足す想定であり、本 crate 自体を `fandhe-edge-core` に依存させる変更は
//! 別 issue の範囲とする。[`split`] は `id`・`group_id`・ラベルだけを要求する
//! 独立したトレイト（[`split::Groupable`]）で結合を最小化しており、同様に
//! `fandhe-edge-core` には依存しない。
//!
//! # 出典
//!
//! [`inspect`] のレコード形状（`id`・`input`・`output.intent`・`tags`・
//! `group_id`）は PoC-16（`docs/spec/03-poc/core-cli-vertical-slice/`）の
//! `definitions/smoke_topic_a.json`（単一選択判定）・`data/smoke/train.jsonl`
//! に一致する形を採用した。異常種別のうち `malformed_json`・`malformed_record`
//! の 2 分類は PoC-9（`docs/spec/03-poc/evaluation-contract/`）の
//! `evaluator/records.py` を踏襲する。[`split`] の割付規則は
//! `docs/spec/03-poc/scratch-classifier/scripts/split_train.py`（PoC-9・PoC-10）
//! を移植したもの（乱数系列は独立実装。[`split`] モジュール doc 参照）。
//!
//! # 前提条件（呼び出し元が守るべきこと）
//!
//! 本 crate の関数はファイル読み込み・サイズ上限検査（REQ-39）を行わない。
//! 既に読み込み済みの JSONL 本文・レコード列を受け取るところから始まる。
//! ガード層（経路の閉じ込め・資源上限。REQ-39・パス未確定）を通過済みの
//! 入力を渡すことを前提とし、CLI 側（TASK-33.x）はガード層を経ずに
//! 生の外部入力を本 crate へ直結してはならない（各モジュールの
//! doc コメントも参照）。`docs/spec` は参照せず、ビルド・テストは
//! `docs/spec` 抜きで成立する（spec-reference のビルド独立方針）。

pub mod inspect;
pub mod split;
