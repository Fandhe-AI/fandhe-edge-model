//! 共通コア層（`fandhe-edge-core`）。
//!
//! 定義ファイル（選択肢〔ラベル〕・入出力構造・モデルの種類の指定）のスキーマと
//! 判定型（判定結果・状態など、層を跨いで共有する型。終了コード契約は
//! `exitcode` モジュールで実装済み〔TASK-21.1-1〕、ok 終了時の判定結果型
//! （選択肢 ID・スコア）は `judgment` モジュールで実装済み〔TASK-21.1-2〕）、
//! 正準化・ハッシュ計算（定義・データ・モデルの同一性判定の土台）を担う（REQ-15）。
//!
//! # 層の境界
//!
//! 共通コアは他のどの層（データ契約・学習ワーカー・評価器・推論ランタイム・
//! 操作アダプター）にも依存しない最下層で、他層がこの crate を参照する
//! 一方向の依存とする。推論ランタイムからも直接参照されるため、学習側の
//! 依存（Python・MLX・学習用 crate）をこの crate に持ち込まない（REQ-32）。
//!
//! # 出典・移植方針
//!
//! PoC-16 の `edge_core`（`docs/spec/03-poc/core-cli-vertical-slice/core`）を
//! M6・M8・M9 の骨格として段階的に移植する計画（`06-roadmap.md`）。
//! ただし本 crate のコード・テストからは `docs/spec` 配下を参照しない
//! （spec-reference のビルド独立方針）。
//!
//! # 現状
//!
//! `definition` モジュールは TASK-15.3-1 で型定義・正常系パーサ、
//! TASK-15.3-2 で必須項目・型不整合の検証、TASK-15.4 でラベル定義
//! （`options`）非同梱時に `missing_labels` で停止する異常系を実装済み。
//! 定義の同一性判定・正準化ハッシュは TASK-15.5 で `canonical` モジュールに
//! 実装済み。
//! `exitcode` モジュールは TASK-21.1-1 で 7 種の終了コード enum と
//! 機械可読 JSON エラー型を実装済み。`judgment` モジュールは TASK-21.1-2 で
//! ok 終了時の判定結果型（選択肢 ID・スコア）を実装済み（CLI 側の出力配線
//! は TASK-33.1/33.2 の対象で未着手）。
//! `hash` モジュールは生バイト列の sha256 ダイジェスト型 [`hash::Sha256Digest`]
//! を実装済み（issue #69・TASK-27.1-1・issue #47・TASK-17.2-1。評価データの
//! 凍結〔REQ-17〕とモデルパッケージの評価前後比較〔REQ-27〕の双方が使う）。
//! `fs` モジュールはサイズ上限付きの安全な通常ファイル読み込み
//! （[`fs::open_regular_file_for_read`]・[`fs::read_bounded`]・
//! [`fs::sha256_file_bounded`]）を実装済み（REQ-39・issue #214 codex/review 指摘。
//! `definition::Definition::load` と評価器〔`crates/eval` の `invariance`
//! モジュール〕が個別に複製していた防御をここへ集約した）。
//! `infer_input` モジュールは TASK-21.2 で推論入力レコードの未知形式・型
//! 不正を検出し `invalid_input`／`limit_exceeded` へ写す型を実装済み
//! （CLI バイナリへの配線は TASK-33.x の対象で未着手）。
//! `rebuild` モジュールは TASK-20.1-1（issue #90）で作り直し判定
//! （[`rebuild::RebuildDecision`]）の型と比較骨格を、TASK-20.1-2（issue #91）
//! で追加・削除・統合の網羅的なパターン検出を、TASK-20.2（issue #92）で
//! `NotRequired` の差分詳細（[`rebuild::NotRequiredRebuild`]）を、
//! TASK-20.3（issue #93）でハッシュ完全一致時に比較処理を一切実行しない
//! ことの保証と専用テストを実装済み（REQ-20 の正常系・異常系・境界値）。
//! `artifact_meta` モジュールは TASK-39.4-2（issue #159）で、パッケージのメタデータ
//! `artifact.json` から経路参照 `onnx_file` だけを取り出す暫定リーダーを実装済み
//! （パッケージ形式の確定は TASK-28/32）。
//! `stage_report` モジュールは TASK-33.2-2（issue #139）で exit 0 時の工程結果
//! JSON（現状は `package` の [`stage_report::PackageReport`]）の型と直列化を実装済み。

pub mod artifact_meta;
pub mod canonical;
pub mod definition;
pub mod exitcode;
pub mod fs;
pub mod hash;
pub mod infer_input;
pub mod judgment;
pub mod rebuild;
pub mod stage_report;
