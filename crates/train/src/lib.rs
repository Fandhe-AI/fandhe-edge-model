//! 学習ワーカー層（Rust 側呼び出し元）の JSON 境界契約（`fandhe-edge-train`）。
//!
//! Rust 側 CLI／ジョブ管理（REQ-34）が、学習ワーカー（`trainer/`・Python）を
//! 子プロセスとして起動して JSON で学習を依頼する経路のうち、本 crate は
//! 「学習リクエスト」と「学習結果（成功時の成果物記録・失敗時のエラー）」の
//! 2 つの JSON を検証付きの型で表す（REQ-18・REQ-19・REQ-34。issue #177）。
//! さらに [`process`] モジュールが、その学習ワーカーを実際に子プロセスとして
//! 起動し、壁時計タイムアウト・出力読み取り上限を掛けたうえで、終了コードと
//! 結果 JSON を REQ-21 の 7 種の終了コードへ写す（REQ-34・REQ-39。issue #178）。
//!
//! # 単一真実源
//!
//! `coding-python.md`「入出力の JSON スキーマは Rust 側の定義を正とし、
//! Python 側で独自のフィールドを増やさない」の方針に基づき、本 crate が
//! スキーマの正となる。ただし実装（検証順序・フィールド名・上限値）は
//! 既存の Python 実装（`trainer/src/fandhe_edge_trainer/contract.py`・
//! `limits.py`・`artifact.py`）から書き起こしたものであり、両者の解釈が
//! 一致することを共有 fixture（`fixtures/train_contract/`）で機械照合する
//! （`crates/train/tests/train_contract_fixture.rs` と
//! `trainer/tests/test_train_contract_fixture.py`）。
//!
//! # 層の境界
//!
//! 学習ワーカー層（`trainer/` の呼び出し元）に位置し、共通コア
//! （`fandhe-edge-core`）に加え、評価器（`fandhe-edge-eval`）にも依存する
//! （[`selection_significance`] が選定結果への McNemar・Holm 有意性判定の
//! 付与に評価器を呼ぶ。評価ロジックは再実装しない。REQ-18・TASK-18.3-1・
//! #87）。操作アダプター（CLI・TUI・MCP）・推論ランタイムはこの crate に
//! 依存しない（推論経路に学習側の型を持ち込まない。REQ-32・
//! `.claude/rules/coding-rust.md`）。
//!
//! [`time_allotment`] は探索予算（REQ-18。既定 1 時間）のうち候補 1 件へ
//! 配分する持ち時間の算出・実行記録を担う（TASK-18.1-1・issue #83）。
//! [`search`] はそれを繰り返し呼び、探索予算全体の消費を追跡し、
//! validation 正解率が最も高い候補を選定する（TASK-18.1-2・issue #84）。
//! 正解率の算出は評価器 `fandhe-edge-eval`
//! （[`fandhe_edge_eval::metrics::evaluate_single_select`]）に委譲する
//! （評価器は 1 つに集約し、他の層で再実装しない。
//! `.claude/rules/coding-rust.md`「crate 構成と層の境界」）。
//!
//! 子プロセスの起動・タイムアウト・終了コード写像（ワーカーの `code` から
//! [`fandhe_edge_core::exitcode::ExitCode`] への対応づけ）は [`process`]
//! モジュールが実装する（issue #178。学習は [`process::run_train`]）。
//! [`time_allotment::CandidateRunner`] はこの実行器を差し込むための接合点
//! （trait）で、[`process::WorkerCandidateRunner`] が `run_train` を使う実装。
//! validation の採点は別プロセスではなく**学習ジョブの中**で行う（issue #84
//! PR #238 レビュー・選択肢 2）: 学習リクエストの任意項目 `validation_inputs`
//! （[`request::ValidationInput`]。`id`・`input` のみで正解ラベルは持たない。
//! REQ-27）を学習ワーカーが学習直後に予測し、結果の `validation_predictions`
//! （[`result::SuccessOutcome::validation_predictions`]）として返す。
//! [`search`] はその予測列を validation gold と突き合わせる。
//!
//! [`kind_resolution`] は利用者が `kind` を省略したとき、既定候補（C1・C3。
//! オーナー承認前の暫定値）を探索候補の集合へ解決する（REQ-19・TASK-19.2・
//! issue #77）。学習ワーカーには常に解決済みの `kind` を渡す。
//!
//! # キャンセル操作（TASK-34.1-1・issue #144）
//!
//! [`process::run_train_cancellable`] と [`job`]（状態遷移・
//! [`job::JobHandle::cancel`]）が、学習ジョブへのキャンセルと状態遷移
//! （実行中 → キャンセル中 → キャンセル済み）を担う（REQ-34）。キャンセルは
//! supervisor の stdin を閉じる協調キャンセルで始め、猶予内に止まらなければ直接の
//! 子への `SIGKILL` にフォールバックする（TASK-34.1-2・issue #145。協調では
//! supervisor が保持中の fd で予約を解放するため公開場所に何も残らず、確定済みなら
//! 成功として扱う。責務分担は [`process`] のモジュール doc）。後続・未実装:
//! `SIGTERM` による graceful cancel（依存追加か `unsafe` が要りユーザー承認事項）・
//! キャンセルの終了コード写像（TASK-33.x）・探索途中のキャンセル伝播・
//! `job.json` 永続化とクラッシュ検出（#146）。やり直しの案内は [`restart`]
//! （再開は提供せず、`SIGKILL` フォールバック後に残る予約は自動で掃除せず案内する。
//! TASK-34.3・#147）。
//!
//! # スコープ外（#178 以降も対象外）
//!
//! - 解決記録（[`kind_resolution::KindResolutionRecord`]）の CLI 出力への反映
//!   （TASK-33.1-2・#136 / TASK-33.2-2・#139）と、定義ファイルへの任意項目
//!   `kind` の追加（スキーマ・正準化ハッシュの変更。ユーザー承認事項）
//!
//! - `kind` ごとの `config` 検証（`config` は「JSON オブジェクトであること」
//!   だけを検査する。`kinds/c1.py`・`c3.py` の `_validate_config` は再実装
//!   しない）
//! - ファイルシステムへの経路の閉じ込め（`root`・`train_path`・`out_dir` は
//!   文字列としての構文検査に留め、存在確認・dir_fd による閉じ込めは学習
//!   ワーカー自身の多層防御（`trainer/src/fandhe_edge_trainer/guard.py`）と
//!   将来のガード層（TASK-39.x）が担う）。子プロセス起動後の `artifact_dir`
//!   の閉じ込め検証は [`result::TrainOutcome::from_worker_stdout`]（#177）が
//!   引き続き担う
//! - CLI `train`／`select` 工程への配線・`trainer_dir` の発見（CLI 引数・
//!   設定からの解決。TASK-33.x）
//! - Rust 側での RSS 監視（学習ワーカー自身の `supervisor.py` が担う）。
//!   外側の壁時計締め切り超過時、[`process::run_train`] は直接の子
//!   （supervisor）だけを `Child::kill()` で終了させる。`_worker` を含む
//!   子孫プロセスの確実な掃除は Rust 側の関与なしに、
//!   学習ワーカー自身が supervisor の死を検知して自己終了する「lifeline」
//!   方式に委ねる（issue #178 PR #233 レビュー: `process_group(0)`・
//!   `/bin/kill` 呼び出しによる Rust 側でのプロセスグループ管理は、PID
//!   再利用・ゾンビ起因の誤判定等の構造的な欠陥が収束しなかったため全面
//!   撤去した。[`process`] モジュール doc・`trainer/src/fandhe_edge_trainer/
//!   supervisor.py` モジュール docstring「lifeline」節参照。native
//!   `kill(2)` の直接呼び出し〔`libc`／`unsafe`〕は依存追加・`unsafe`
//!   新規導入のいずれもユーザー承認事項のため対象外）
//! - McNemar・Holm による有意性判定の選定記録への統合（TASK-18.3-1・issue #87）
//! - 探索記録のファイルへの永続化・CLI `select` 工程の JSON 出力・終了コード
//!   への写像（TASK-33.x）

pub mod error;
pub mod job;
mod kind_defaults;
// kind 省略時の既定候補の解決（REQ-19・TASK-19.2・#77）。
pub mod kind_resolution;
pub mod limits;
pub mod process;
pub mod request;
// 中断ジョブへの再開非提供・やり直し案内（REQ-34・TASK-34.3・#147）。
pub mod restart;
pub mod result;
pub mod search;
// 選定結果への McNemar・Holm 有意性判定の付与（評価器
// `fandhe-edge-eval` を呼ぶ。REQ-18・REQ-25・TASK-18.3-1・#87）。
pub mod selection_significance;
// CLI `train`・`select`・`package` 工程が置くファイルの読み書き補助（REQ-18・TASK-33.1-2・#136）。
pub mod stage_files;
pub mod time_allotment;
