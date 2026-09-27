//! データ契約層（`fandhe-edge-data`）。
//!
//! 利用者が用意した学習・評価データ（JSONL、1 行 1 レコード）を、後続工程
//! （分割・学習・評価）へ渡す前に検査する（REQ-16）。本 crate は
//! TASK-16.1（親 issue #37）のうち、型・必須項目・ラベル enum の検査ロジック
//! （TASK-16.1-1・issue #38）を担う。件数集計（行数・ユニーク数・ラベル別件数。
//! TASK-16.1-2・issue #39）は本 crate の対象外。
//!
//! # 層の境界
//!
//! データ契約層は共通コア（`fandhe-edge-core`）の下位に位置づく想定だが、
//! 本 crate 作成時点（issue #38 実装時）で共通コアの定義ファイルスキーマ
//! （TASK-15.3・issue #32〜#34）が未実装のため、`fandhe-edge-core` には
//! 依存しない。有効なラベル ID の集合は呼び出し側（将来的には CLI 等の
//! 上位層）から `BTreeSet<String>` として受け取る形に設計を分離している。
//! TASK-15.3 が実装された後は、`fandhe-edge-core::definition` の型から
//! `BTreeSet<String>` へ変換する 1 行のアダプタを呼び出し側（CLI 等）に
//! 足す想定であり、本 crate 自体を `fandhe-edge-core` に依存させる変更は
//! 別 issue の範囲とする。
//!
//! # 出典
//!
//! レコード形状（`id`・`input`・`output.intent`・`tags`・`group_id`）は
//! PoC-16（`docs/spec/03-poc/core-cli-vertical-slice/`）の
//! `definitions/smoke_topic_a.json`（単一選択判定）・`data/smoke/train.jsonl`
//! に一致する形を採用した。異常種別のうち `malformed_json`・`malformed_record`
//! の 2 分類は PoC-9（`docs/spec/03-poc/evaluation-contract/`）の
//! `evaluator/records.py` を踏襲する。
//!
//! # 前提条件（呼び出し元が守るべきこと）
//!
//! 本 crate の関数はファイル読み込み・サイズ上限検査（REQ-39）を行わない。
//! 既に読み込み済みの JSONL 本文（`&str`）を受け取るところから始まる。
//! ガード層（経路の閉じ込め・資源上限。REQ-39・パス未確定）を通過済みの
//! 入力を渡すことを前提とし、CLI 側（TASK-33.x）はガード層を経ずに
//! 生の外部入力を本 crate へ直結してはならない（`inspect` モジュールの
//! doc コメントも参照）。

pub mod inspect;
