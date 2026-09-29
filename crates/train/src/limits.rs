//! 学習リクエスト・結果 JSON の資源上限（REQ-39）。
//!
//! 値は学習ワーカー側の単一真実源
//! （`trainer/src/fandhe_edge_trainer/limits.py`・`contract.py`・
//! `supervisor.py`）をそのまま写したもので、オーナー確認済み
//! （2026-09-27・issue #11）。本ファイルは値を緩めず、変更する場合は
//! Python 側・共有 fixture（`fixtures/train_contract/limits.json`）と
//! 同時に見直す（`crates/train/tests/train_contract_fixture.rs` が
//! fixture との一致を機械照合する）。
//!
//! 例外: [`MAX_WORKER_STDERR_BYTES`]・[`SUPERVISOR_SHUTDOWN_GRACE_SECONDS`]
//! （issue #178）・[`COOPERATIVE_CANCEL_GRACE_SECONDS`]（issue #145）は Rust 側（子プロセスの起動・監視）固有の値で、
//! 学習ワーカー側に対応する単一真実源を持たない（各定数の doc に根拠を
//! 記す）。
//!
//! [`crate::limits`] の各定数は `fandhe_edge_core::judgment` の
//! `MAX_OPTIONS`・`MAX_CHOICE_ID_BYTES` と値がたまたま同じでも、
//! 契約としては別物であり結合しない（学習リクエストの `label_order` は
//! 学習ワーカー固有の契約であり、推論 1 件あたりの判定結果契約とは独立に
//! 変更されうる）。
//!
//! 例外: [`MAX_VALIDATION_INPUT_TOTAL_BYTES`] は Python 側（`limits.py`）に
//! 対応する値を持たず、共有 fixture（`fixtures/train_contract/limits.json`・
//! `train_contract_fixture.rs`）の機械照合の対象にも含めない。
//! `crate::search::SearchInput` は Rust 内部でのみ使う型（issue #84）で、
//! 学習ワーカーとの JSON 境界には現れないため。

/// 学習リクエスト JSON の `schema_version`（`contract.py::SCHEMA_VERSION`）。
pub const REQUEST_SCHEMA_VERSION: u32 = 1;

/// 学習リクエスト JSON ファイルの読み込み上限（バイト）。
/// `contract.py::limits.MAX_REQUEST_BYTES`。
pub const MAX_REQUEST_BYTES: usize = 1024 * 1024;

/// `label_order` の最小・最大件数。`limits.py::MIN_LABELS`・`MAX_LABELS`。
pub const MIN_LABELS: usize = 2;
pub const MAX_LABELS: usize = 1024;

/// 1 ラベルあたりの最大 UTF-8 バイト数。`limits.py::MAX_LABEL_BYTES`。
pub const MAX_LABEL_BYTES: usize = 256;

/// `max_bytes`（バイト入力の最大長）の許容範囲。
/// `limits.py::MIN_MAX_BYTES`・`MAX_MAX_BYTES`。
pub const MIN_MAX_BYTES: u32 = 1;
pub const MAX_MAX_BYTES: u32 = 4096;

/// `seed` の許容範囲。`limits.py::MIN_SEED`・`MAX_SEED`
/// （`mx.random.seed`・`numpy.random.default_rng` が受け付ける非負 32bit
/// 符号なし整数の範囲）。
pub const MIN_SEED: u32 = 0;
pub const MAX_SEED: u32 = u32::MAX;

/// 1 学習ジョブあたりの壁時計上限（秒）。`limits.py::MAX_TRAIN_WALL_SECONDS`。
pub const MAX_TRAIN_WALL_SECONDS: u32 = 3600;

/// 1 学習ジョブあたりの RSS 上限（バイト）。`limits.py::MAX_TRAIN_RSS_BYTES`。
pub const MAX_TRAIN_RSS_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// 学習結果（ワーカーの標準出力）の読み込み上限（バイト）。
/// `supervisor.py::_MAX_WORKER_STDOUT_BYTES`。
pub const MAX_RESULT_BYTES: usize = 1024 * 1024;

/// `device` に許可される値。`contract.py::_ALLOWED_DEVICES`。
pub const ALLOWED_DEVICES: [&str; 2] = ["cpu", "gpu"];

/// 成果物の `selector_version` に許可される値（`artifact.py::SELECTOR_VERSION`）。
/// 空文字列・未対応版の成功結果を検証済みとして受理しないための許可リスト
/// （REQ-39「完全性と版」・P1。codex review PR #220）。値を追加する場合は
/// `artifact.py::SELECTOR_VERSION` の版付け方針と合わせて見直す。
pub const ALLOWED_SELECTOR_VERSIONS: [&str; 1] = ["0.1"];

/// [`crate::search::SearchInput::validation_inputs`]（validation 入力
/// 1 件分）の合計バイト数の上限（REQ-39「資源の上限」・P0 指摘対応。
/// codex review PR #238）。
///
/// `SearchInput` は公開 API で、データ契約層（`crates/data`）を経由しない
/// 呼び出し元が直接値を渡せるため、`validate_input`（`crate::search`）が
/// 学習を始める前に（採点は学習ジョブの中で行うため。[`crate::search`]
/// モジュール doc 参照）合計バイト数を検証する。1 件あたりの上限は
/// `fandhe_edge_core::infer_input::MAX_INFER_INPUT_BYTES`（推論入力 1 件の
/// 上限。train・infer で共有）をそのまま使うため train 側に重複定義しない
/// が、合計バイト数の上限は train・data のいずれにも既存の定数が無いため
/// 本ファイルに新設する（承認事項として報告。issue #84 PR #238 レビュー）。
/// 値は `crates/data::leak::MAX_LEAK_CHECK_TOTAL_BYTES`（学習・評価データの
/// 矛盾検出で合計バイト数に用いる上限。64 MiB）と同じ値に揃えた
/// （`crates/train` は `crates/data` に依存しないため定数を共有できず、
/// 値のみ揃えて重複定義する。新規の crate 間依存の追加はユーザー承認が
/// 必要なため、値の一致に留めた）。
///
/// 1 件あたりの上限（1 MiB）と本定数（64 MiB）を単純に割ると、64 件までは
/// 1 件あたり上限ぎりぎりのサイズでも許容できる。一方 validation 件数の
/// 上限は [`fandhe_edge_eval::significance::MAX_EVAL_RECORDS`]（100 万件）
/// で、100 万件に本定数を均等配分すると 1 件あたり平均 67 バイト程度しか
/// 割り当てられない。実際の validation 入力の典型サイズ・件数の想定が
/// この 2 つの上限とどう両立するかは検証していないため、64 MiB という
/// 値自体の妥当性は承認事項として報告する。
///
/// `crate::search::SearchInput::validation_record_ids`（validation レコード
/// 識別子の列。1 件あたりの上限は別途
/// [`fandhe_edge_core::judgment::MAX_INPUT_ID_BYTES`] を使う）の合計バイト
/// 数上限にも本定数を流用する（P1 指摘対応。issue #84 PR #238 レビュー。
/// record_id 専用の新しい定数を追加で起こさず、「1 つの `SearchInput`
/// フィールドが保持できる合計バイト数」の共通の目安として扱う）。
///
/// 学習リクエスト JSON の `validation_inputs`（学習ジョブ内での採点用の
/// validation 入力。[`crate::request::ValidationInput`]）にも、同じ定数を
/// 合計バイト数（`id`＋`input`）の上限として使う（issue #84 PR #238・
/// 選択肢 2）。ただし validation 入力はリクエスト JSON の**内側**を通るため、
/// 実際に効く上限は [`MAX_REQUEST_BYTES`]（1 MiB。リクエスト全体）であり、
/// 本定数（64 MiB）はそれより緩い。すなわち 1 リクエストで運べる validation
/// 入力は、リクエストの他の項目を除いておよそ 1 MiB 以内（短いレコードで
/// 数千件規模）に限られる。1 件あたり [`fandhe_edge_core::infer_input::MAX_INFER_INPUT_BYTES`]
/// （1 MiB）ちょうどのレコードも、リクエスト全体の上限を超えるため 1 件でも
/// 拒否される。本定数は「`SearchInput` が保持できる合計」の上限として
/// 変更せず、リクエスト側の実効上限との差はオーナー判断で据え置いた
/// （選択肢 (a)。上限を緩めるなら `MAX_REQUEST_BYTES` の見直しが別途必要）。
pub const MAX_VALIDATION_INPUT_TOTAL_BYTES: usize = 64 * 1024 * 1024;

/// `validation_inputs` を含むリクエストの結果 JSON（`validation_predictions`
/// を含む）の読み取り上限の**天井**（バイト）。実際の上限はリクエストごとに
/// [`crate::request::validation_result_bytes_bound`] で正確に計算する
/// （`TrainRequest::max_result_bytes`。許可するラベル・id から求める）。計算値が
/// この天井を超えるリクエストは、学習を始める前に `limit_exceeded` で拒否する
/// （P1 指摘対応。issue #84 PR #238 レビュー: 以前の固定の見積もりは、エスケープで
/// 大きくなる許可済みのラベルの結果を収容できなかった）。したがって、受理された
/// リクエストの正常な結果が上限で弾かれることはない。
/// `validation_inputs` を持たないリクエストの結果には [`MAX_RESULT_BYTES`]（1 MiB）
/// を使い続ける。Python 側の対応は `limits.py::MAX_RESULT_BYTES_WITH_VALIDATION`。
///
/// 値 64 MiB は [`MAX_VALIDATION_INPUT_TOTAL_BYTES`] と同じで、新しい桁を増やさない
/// （根拠: 仮置き。実用上の validation 件数・ラベル長は、この天井の内側に収まる
/// ことをテストで確認している。ただし制御文字だけの最長ラベルと多数の短い入力の
/// 組み合わせは拒否されうる）。
pub const MAX_RESULT_BYTES_WITH_VALIDATION: usize = 64 * 1024 * 1024;
/// 学習ワーカーの標準エラー出力（stderr）の保持上限（バイト）。REQ-39
/// 「資源の上限」（#178）。`_worker` は supervisor の stderr を継承する
/// （`supervisor.py::_spawn_worker_and_finalize`）ため、Rust 側 stderr
/// パイプの書き手は supervisor 自身と `_worker` の両方になりうる。stdout
/// （[`MAX_RESULT_BYTES`]。契約上の結果 JSON）とは別に、診断用途の stderr
/// にも無制限の保持を許さないための独立した上限を設ける。先頭から保持し、
/// 超過分は読み捨てる（`crates/train/src/process.rs`）。
pub const MAX_WORKER_STDERR_BYTES: usize = 64 * 1024;

/// Rust 側の壁時計締め切りに足す猶予（秒）。REQ-34・REQ-39（#178）。
///
/// `supervisor.py`（内側監視者）は `_worker` を別セッション・別プロセス
/// グループ（W）で起動する（`start_new_session=True`）。内側の壁時計・RSS
/// タイムアウトでは supervisor 自身が `killpg(W)` でそのグループごと掃除
/// する。Rust 側の外側締め切りは、supervisor 自身が `time_limit_seconds`
/// 超過を検出してから、内側の後始末（`killpg`・`ps` 呼び出し・reader
/// スレッドの回収等）を終えるまでの時間を確実に上回る必要がある（さもない
/// と、まだ後始末中の supervisor を Rust 側が早期に打ち切ってしまう）。
///
/// Rust 側が外側締め切りで直接の子（supervisor）を `SIGKILL` した場合
/// （＝ supervisor が自身の後始末を終える前に終了させられた場合）でも、
/// `_worker` は孤児として残らない: `_worker` は起動直後から「lifeline」
/// （supervisor が握り続ける pipe の書き込み端）を監視しており、supervisor
/// がどのような形で終了しても、カーネルが書き込み端を自動的に閉じるため
/// `_worker` は必ず EOF を観測して自己終了する（issue #178 PR #233
/// レビュー。`trainer/src/fandhe_edge_trainer/supervisor.py` モジュール
/// docstring「lifeline」節参照。Rust 側でのプロセスグループ管理
/// 〔`process_group(0)`・`/bin/kill` 呼び出し〕は構造的な欠陥が収束せず
/// 全面撤去した）。
///
/// 内訳（`supervisor.py` の定数から算出。根拠を明示し、緩めずに保つ）:
/// 内側の猶予 `_TIME_LIMIT_GRACE_SECONDS`（5 秒）＋ `ps` 呼び出しの
/// タイムアウト（5 秒）＋ kill 後の `proc.wait()` 上限（10 秒）＋ reader
/// スレッドの join 上限（10 秒＋5 秒）＝ 35 秒に、Python プロセス起動・
/// OS のプロセス後始末のための余裕を加えて 60 秒とする。
///
/// この結果、既定値（`time_limit_seconds` = [`MAX_TRAIN_WALL_SECONDS`] =
/// 3600 秒）では Rust 側の外側締め切りは 3660 秒となり、字義どおりの
/// 「3600 秒を超えない」からは猶予分だけ外れる（issue #178 実装計画・
/// オーナー確認事項 1。孤児プロセスを残さないことを優先する設計判断）。
pub const SUPERVISOR_SHUTDOWN_GRACE_SECONDS: u32 = 60;

/// 協調キャンセルで、supervisor が自ら後始末して終了するのを待つ猶予（秒）。
/// REQ-34・REQ-39（TASK-34.1-2・#145）。
///
/// キャンセル時、Rust 側は supervisor の `stdin`（キャンセル用パイプ）を閉じ、
/// supervisor が worker の停止（`killpg` → 回収。`wait` の上限 10 秒）と予約の
/// 解放・結果 JSON の出力を終えて自ら終了するのを、この猶予まで待つ。上限は
/// `supervisor.py::_terminate_and_reap` の `wait(timeout=10)` に、予約解放と JSON
/// 出力の時間を加えた値を上回る 15 秒とする。猶予を超えたら従来どおり `SIGKILL`
/// にフォールバックする（無限待ちを作らない）。壁時計の期限は常にこの猶予より
/// 優先し、`SIGKILL` 経路の回収待ち（5 秒）が壁時計の内側に収まるよう、待ちを
/// 壁時計の期限から遡って切り詰める（`crates/train/src/process.rs`）。
///
/// Rust 側固有の値で、学習ワーカー側に対応する定数は無い（共有 fixture の対象外）。
/// 暫定値で、実機での測定に基づく見直しは別途。
pub const COOPERATIVE_CANCEL_GRACE_SECONDS: u32 = 15;
