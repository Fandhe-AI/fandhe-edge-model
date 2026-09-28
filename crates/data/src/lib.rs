//! データ契約層（`fandhe-edge-data`）。
//!
//! 利用者が用意した学習・評価データ（JSONL、1 行 1 レコード）の検査
//! （REQ-16）・group 単位分割と凍結（REQ-17）・来歴（REQ-40）を担う層。
//! 本 crate は現時点で以下を実装する。
//!
//! - [`inspect`]: 型・必須項目・ラベル enum の検査ロジック
//!   （TASK-16.1（親 issue #37）のうち TASK-16.1-1・issue #38）
//! - [`report`][]: 検査レポート（行数・ユニーク数・ラベル別件数の集計。
//!   TASK-16.1-2・issue #39）
//! - [`consistency`]・[`normalize`]: 矛盾レコード検出・メタデータ混入検出
//!   （TASK-16.2-2）。既定の正規化規則は学習ワーカーと同じ NFKC＋空白規則
//!   （[`normalize`] モジュールのドキュメントを参照）
//! - [`leak`]: 漏洩・group 跨ぎ検出（REQ-16・TASK-16.2-1・issue #41）
//! - [`split`]: group 単位分割ロジック（TASK-17.1-1・issue #44）
//! - [`split_record`]: 分割の seed・規則・各分割のハッシュの記録と永続化
//!   （TASK-17.1-2・issue #45）
//! - [`eval_input`][]: 評価入力（gold・pred）の異常系処理（REQ-23・
//!   TASK-23.1-1・issue #55（ケース 1〜6）・TASK-23.1-2・issue #56
//!   （ケース 7〜12。矛盾・ラベル順序・未出現クラス・不正なスコア・
//!   全件保留・全件失敗）で全 12 ケースの挙動を固定済み
//! - [`preprocess_boundary`][]: 空入力の前処理食い違いの検知・報告
//!   （REQ-23 境界値・TASK-23.2・issue #57）
//! - [`provenance`]: 来歴レコード型（REQ-40・TASK-40.1-1・issue #74）。
//!   [`provenance::ingest`] で取り込み時の JSON 検証・記録 JSON 生成を実装
//!   （TASK-40.1-2・issue #75）
//! - [`ingest`][]: データ検査（[`inspect::inspect_records`]）と来歴の取り込み
//!   （[`provenance::ingest::parse_provenance_json`]）の接続点
//!   （REQ-40・TASK-40.1-2・issue #75）
//! - [`eval_freeze`][]: 評価データ本体の凍結記録（sha256・バイト長）と、
//!   評価データなしの境界動作（`evaluate` が `status:"skipped"`・exit 0 で
//!   完走する型のゲート。REQ-17・TASK-17.2-1・issue #47）
//! - [`frozen_placement`][]: 凍結済み評価データの読み取り専用配置（unix:
//!   `chmod 444` 相当）と、直接の書き込み試行が権限エラーで拒否されることの
//!   非破壊的な確認（REQ-39・REQ-17・TASK-17.2-2・issue #48）
//!
//! # 現状（実装済みを装わない）
//!
//! - データ検査（REQ-16）
//!   - 矛盾レコード検出・メタデータ混入検出（TASK-16.2-2）: 実装済み
//!   - 漏洩・group 跨ぎ検出（TASK-16.2-1・[`leak`]）: 実装済み
//!   - 検査レポート（行数・ユニーク数・ラベル別件数の集計。TASK-16.1-2）:
//!     実装済み
//!   - JSONL の読み込み・型・必須項目の検査（TASK-16.1）: 未実装
//!     （TASK-16.1 のローダがファイル I/O・サイズ上限検査（REQ-39）を担い、
//!     [`consistency`]・[`leak`] の trait を実装したレコード列を渡す想定）
//!   - 正規化した入力での漏洩照合（NFKC 等）: 未実装（TASK-15.5 の完了が前提。
//!     [`leak`] のスコープの境界を参照）
//! - 分割結果のハッシュ計算・記録は [`split_record`] で実装済み
//!   （REQ-17・TASK-17.1-2・issue #45）。評価データの凍結（[`eval_freeze`]・
//!   REQ-17・TASK-17.2-1・issue #47）は「実データバイト列から独立に再計算
//!   したハッシュとの一致確認」「評価データなしの境界動作」までを実装済み。
//!   読み取り専用配置・直接書き込みの拒否確認（[`frozen_placement`]・
//!   REQ-39・TASK-17.2-2・issue #48）も実装済み。凍結後のハッシュ不一致検知
//!   で処理を止める分岐（過去の記録台帳との突き合わせ・版管理。REQ-17・
//!   TASK-17.3・issue #49）は未実装。CLI `evaluate` 工程
//!   への配線・学習・パッケージ化・推論を通した完走確認（TASK-33.3・
//!   issue #140）も未実装。（[`inspect::inspect_records`] が返す
//!   [`inspect::ValidRecord`] は [`split::Groupable`] を実装しないため、
//!   [`split::split_by_group`]・[`split_record::split_and_record`] へ渡す際は
//!   呼び出し側（CLI 等）が変換する）
//! - 来歴（REQ-40）: レコード型・各項目の検証（[`provenance`]・
//!   TASK-40.1-1）に加え、取り込み時の JSON 検証・記録 JSON 生成・
//!   データ検査との接続（[`provenance::ingest`]・[`ingest`]・
//!   TASK-40.1-2・issue #75）を実装済み。指示文本文からの sha256 計算
//!   （`sha2` の data 層配置はユーザー承認待ち。計算済みハッシュの取り込みの
//!   みサポート）・`source`（生成元）フィールド・外部 LLM 出力の既定拒否
//!   （TASK-40.2）・CLI `inspect` 工程への配線（TASK-33.x）は未実装
//!
//! # 層の境界
//!
//! データ契約層は共通コア（`fandhe-edge-core`）より上位に位置する想定だが、
//! 本 crate 作成時点（issue #38 実装時）で共通コアの定義ファイルスキーマ
//! （TASK-15.3・issue #32〜#34）が未実装だったため、[`inspect`] は
//! `fandhe-edge-core` には依存せず、有効なラベル ID の集合を呼び出し側
//! （将来的には CLI 等の上位層）から `BTreeSet<String>` として受け取る形で
//! 設計を分離している。`fandhe-edge-core::definition` の型から
//! `BTreeSet<String>` へ変換する 1 行のアダプタを呼び出し側（CLI 等）に
//! 足す想定であり、[`inspect`] 自体を `fandhe-edge-core` に依存させる変更は
//! 別 issue の範囲とする。[`split`] は `id`・`group_id`・ラベルだけを要求する
//! 独立したトレイト（[`split::Groupable`]）で結合を最小化しており、同様に
//! `fandhe-edge-core` には依存しない。[`consistency`] も同様に独立したトレイト
//! （`InputRecord` 等）でレコード形状を抽象化し、`fandhe-edge-core` には
//! 依存しない。[`leak`] も同様に独立したトレイト（[`leak::LeakCheckable`]）
//! で結合を最小化しており、学習ワーカー・評価器・操作アダプターから
//! 呼ばれる想定。一方 [`split_record`]（issue #45）は
//! 分割記録のハッシュ計算に `fandhe-edge-core::canonical::canonical_sha256_hex`
//! （sha2 を内包）を使い、[`eval_freeze`]（TASK-17.2-1・issue #47）は評価データの
//! 凍結記録（sha256・バイト長）に `fandhe-edge-core::hash::Sha256Digest` を、
//! `evaluate` 工程の終了コード対応に `fandhe-edge-core::exitcode::ExitCode` を
//! 再利用するため、いずれも workspace 内のパス依存として `fandhe-edge-core`
//! に依存する（dependency-policy.md「承認済みの依存（Rust）」2026-09-28 承認。
//! `inspect`・`split`・`leak`・`consistency` は引き続き `fandhe-edge-core` に
//! 依存しない。依存の向きは crate 全体として data → core の一方向のまま
//! 変わらない）。
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
//!
//! **例外**: [`frozen_placement`] は評価データ本体を読み込みはしないが、
//! ファイルシステムへの副作用（権限変更）を持つ点で他モジュールと異なる。
//! `path` はガード層（経路の閉じ込め。TASK-39.x）を通過済みであることを
//! 引き続き前提とする（[`frozen_placement`] モジュール doc「責務の境界」
//! 参照）。

pub mod consistency;
pub mod eval_freeze;
pub mod eval_input;
pub mod frozen_placement;
pub mod ingest;
pub mod inspect;
mod json_keys;
pub mod leak;
pub mod normalize;
pub mod preprocess_boundary;
pub mod provenance;
pub mod report;
pub mod split;
pub mod split_record;
