//! 学習データと評価データの間の漏洩・group 跨ぎ検出（REQ-16 異常系・TASK-16.2-1）。
//!
//! 気づかないまま評価データが学習データに漏れて精度を過大評価することを防ぐ
//! （REQ-16 ユーザーストーリー）。本モジュールは 2 種類の異常を検出する。
//!
//! - **入力の漏洩**（[`find_input_leaks`]）: train の入力と byte 完全一致する
//!   入力が他の分割（validation・test・利用者の評価データ）に存在する。
//! - **group 跨ぎ**（[`find_group_straddles`]）: 同一 `group_id` を持つレコードが
//!   2 つ以上の分割に分散している（入力が異なっていても検出する。跨ぎ検出の
//!   存在意義そのもの）。
//!
//! 設計・受入基準の出典は PoC-9（`03-poc/evaluation-contract/evaluator/inspect.py`
//! の `find_train_test_leak`・`find_cross_group`。`docs/spec` は参照せず、本モジュールは
//! 同じ観点を独立に実装したものであり、PoC-9 の出力とビット互換であることは保証しない）。
//!
//! # スコープの境界（本 Issue #41 / TASK-16.2-1 が扱わないこと）
//!
//! 1. **group_id は受け取って使うだけで、導出しない**。`(intent, args)` の
//!    テンプレートキーから group を作る処理はデータ準備側の責務であり、
//!    本モジュールは既に付与された `group_id` をそのまま照合に使う。
//! 2. **照合は `input` の byte 完全一致のみ**。PoC-9 は NFKC 正規化・前後空白除去・
//!    連続空白の圧縮をした後の入力で照合していたが、`unicode-normalization` の
//!    導入は見送り中で、正準化規則は共通コアの 1 箇所に集約する方針
//!    （coding-rust.md・TASK-15.5）のため、本モジュールでは独自の正規化をしない。
//!    正規化した入力での照合は後続の課題（依存の承認と TASK-15.5 の完了が前提）。
//! 3. **同一分割内の重複は扱わない**（TASK-16.2-2 / #42 のデータ検査レポートの担当）。
//! 4. **終了コードや `inspect` CLI への接続はしない**。検出結果をどう終了コード
//!    （REQ-21）へ写すかは評価契約・入出力契約の設計判断であり、main が設計し
//!    ユーザー承認を経てから決める。本モジュールは構造化したレポートを返すところまで。
//! 5. TASK-16.2-2（#42。メタデータ混入・矛盾レコード検出）とはレポート型を分離しており、
//!    統合は #39（検査レポート全体の集約）が担う。
//!
//! # 資源上限について
//!
//! 本モジュールは `Partitions` に渡されたスライス長の合計に比例した処理のみを行い、
//! 入力 byte はすべて借用（コピーしない）。`split.rs` は件数・サイズの上限検査
//! （REQ-39）を呼び出し側（データ検査層・ガード層）の責務としているが、本モジュールの
//! 公開 API（[`find_input_leaks`]・[`find_group_straddles`]・[`inspect_leakage`]）は
//! train 側 ID 列を複製・索引化する経路（下記）を持つため、呼び出し側へ委ねるだけでは
//! この API を直接使う経路で上限を保証できない（reviewer 指摘 PR #195・
//! security.md「ガード層: 資源の上限」）。そこで [`validate_resource_limits`] で
//! レコード総件数・ID 長・入力 byte 長を入口で検証してから集計する（`crates/core/src/definition.rs`
//! の `MAX_DEFINITION_FILE_BYTES` と同じく、REQ-39 の資源上限が正式に決まるまでの
//! 暫定値。[`MAX_LEAK_CHECK_RECORDS`]・[`MAX_LEAK_CHECK_ID_BYTES`]・
//! [`MAX_LEAK_CHECK_INPUT_BYTES`]）。
//!
//! また出力側では、同一の train 入力が複数の相手分割（validation・test・
//! evaluation）へ漏洩した場合、その入力に対応する train 側 ID 列を分割の数だけ
//! 複製すると、外部データが持つ重複度に応じて出力サイズが増幅されうる
//! （reviewer 指摘 PR #195）。[`InputLeak::train_ids`] は `Rc<[String]>` で
//! 同じ ID 列を共有し、複製を ID 列の実体ではなく参照カウントの増加に留める。
//!
//! さらに `find_input_leaks` は train 側 ID の所有化（`&str` から `String` への
//! 複製）自体を、実際に他の分割へ漏洩したと判明した入力についてのみ遅延して行う。
//! 件数上限（[`MAX_LEAK_CHECK_RECORDS`]）と ID 長上限（[`MAX_LEAK_CHECK_ID_BYTES`]）を
//! 満たす入力であっても、漏洩が 0 件の場合に train 側の全 ID を無条件で複製すると、
//! 許容範囲内の件数・長さだけで数百 MB 規模のメモリを追加確保しうる
//! （reviewer 指摘 PR #195 追加分・security.md「ガード層: 資源の上限」）。
//!
//! 上記の件数上限（[`MAX_LEAK_CHECK_RECORDS`]）と 1 件あたりの ID・入力長上限
//! （[`MAX_LEAK_CHECK_ID_BYTES`]・[`MAX_LEAK_CHECK_INPUT_BYTES`]）はいずれも
//! 単独の値としては妥当でも、両方の上限いっぱいの入力（上限件数 ×
//! 上限長の ID）を大量に漏洩・group 跨ぎと判定させると、複製された ID が
//! 数百 MB 規模のメモリを確保しうる（reviewer 指摘 PR #195: 件数上限
//! `MAX_LEAK_CHECK_RECORDS` 分の ID を ID 長上限いっぱいまで漏洩させると
//! 約 800MB 超になりうる）。そこで ID・group ID・入力の byte 長の合計にも
//! 別途 [`MAX_LEAK_CHECK_TOTAL_BYTES`] を設け、[`validate_resource_limits`] の
//! 入口で拒否する（件数上限・個別長上限のどちらも満たしていても、合計が
//! 上限を超えれば処理しない）。

use std::collections::BTreeMap;
use std::rc::Rc;

/// 1 回の検査（[`find_input_leaks`]・[`find_group_straddles`]）で受け付ける
/// レコード総件数（train + validation + test + evaluation）の上限。
///
/// 暫定値（REQ-39 の資源上限が正式に決まり次第見直す。
/// `crates/core/src/definition.rs::MAX_DEFINITION_FILE_BYTES` と同じ方針）。
pub const MAX_LEAK_CHECK_RECORDS: usize = 200_000;

/// レコード ID・group ID 1 件あたりの byte 長の上限（暫定値。同上）。
pub const MAX_LEAK_CHECK_ID_BYTES: usize = 4096;

/// レコード 1 件あたりの入力（[`LeakCheckable::input`]）の byte 長の上限（暫定値。同上）。
///
/// `find_input_leaks` は train 側の入力を `BTreeMap` のキーとして比較するため、
/// 1 件の入力が極端に長いと、件数上限（[`MAX_LEAK_CHECK_RECORDS`]）を満たしていても
/// 入力サイズに応じて比較処理の時間が線形に増える（reviewer 指摘 PR #195・REQ-39
/// 「資源の上限」）。ID 長と同じ暫定値を流用する。
pub const MAX_LEAK_CHECK_INPUT_BYTES: usize = 4096;

/// ID・group ID・入力（[`LeakCheckable::id`]・[`LeakCheckable::group_id`]・
/// [`LeakCheckable::input`]）の byte 長の合計（train + validation + test +
/// evaluation 全レコード分）の上限（暫定値。同上）。
///
/// 件数上限（[`MAX_LEAK_CHECK_RECORDS`]）・個別長上限（[`MAX_LEAK_CHECK_ID_BYTES`]・
/// [`MAX_LEAK_CHECK_INPUT_BYTES`]）を両方満たす入力でも、上限いっぱいの長さの
/// レコードを上限件数ぶん並べると `find_input_leaks`・`find_group_straddles` が
/// ID を複製する経路で数百 MB 規模のメモリを追加確保しうる（reviewer 指摘 PR #195・
/// REQ-39「ガード層: 資源の上限」）。本モジュール先頭のドキュメントコメント参照。
pub const MAX_LEAK_CHECK_TOTAL_BYTES: usize = 64 * 1024 * 1024;

/// [`validate_resource_limits`] が検出する、公開 API の入口で拒否すべき違反
/// （REQ-39・security.md「ガード層: 資源の上限」。reviewer 指摘 PR #195）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LeakCheckError {
    /// train・validation・test・evaluation の合計レコード件数が上限を超えた。
    TooManyRecords { total: usize, limit: usize },
    /// レコード ID または group ID の byte 長が上限を超えた。
    ///
    /// ID の内容そのものはエラーに含めない（データ本文・識別子をログ・
    /// エラーメッセージへ転記しない。security.md「秘密情報の混入防止」）。
    IdTooLong { len: usize, limit: usize },
    /// レコードの入力（[`LeakCheckable::input`]）の byte 長が上限を超えた。
    ///
    /// 入力そのものはエラーに含めない（データ本文をログ・エラーメッセージへ
    /// 転記しない。security.md「秘密情報の混入防止」）。
    InputTooLong { len: usize, limit: usize },
    /// ID・group ID・入力の byte 長の合計（全レコード分）が
    /// [`MAX_LEAK_CHECK_TOTAL_BYTES`] を超えた。
    ///
    /// 件数・個別長の上限を両方満たしていても、合計 byte 数が大きいと
    /// ID の複製経路（[`InputLeak::train_ids`] 等）で数百 MB 規模のメモリを
    /// 追加確保しうるため、個別の上限とは別に検証する（reviewer 指摘 PR #195）。
    TotalBytesExceeded { total: usize, limit: usize },
}

/// 漏洩・group 跨ぎ検出の対象になるレコードが満たす最小の契約。
///
/// `split.rs` の [`crate::split::Groupable`] と同様に、TASK-16.1 の具体的な
/// レコード構造体には依存しない。将来そのレコード構造体は `Groupable` と
/// `LeakCheckable` の両方を実装する想定（#38）。
pub trait LeakCheckable {
    /// レコード ID（レポート内でレコードを特定するために使う）。
    fn id(&self) -> &str;

    /// 入力の byte 表現（README「実装方針（要点）」・REQ-15: 入力表現は byte のみ）。
    ///
    /// レポートにはこの byte 列そのものを含めない（security.md「データ本文を
    /// ログ・エラーメッセージへ転記しない」）。ID だけで特定する。
    fn input(&self) -> &[u8];

    /// group ID（データ準備側で付与済みのもの。本モジュールでは導出しない）。
    ///
    /// `inspect::ValidRecord::group_id` が `Option<String>`（group_id は任意項目。
    /// `inspect.rs`「group_id: 任意。必須化・自動算出は TASK-17.1 の検討事項」）で
    /// あるのに合わせて `Option<&str>` にする（reviewer 指摘 PR #195・crates/data/src/leak.rs:172）。
    /// 呼び出し側が欠損値を空文字列などの番兵値へ置き換えて実装すると、
    /// 無関係な複数レコードが同一 group（空文字列）として誤って群跨ぎ判定される
    /// おそれがあるため、欠損を表現できる契約にして [`find_group_straddles`] 側で
    /// `None` を判定から除外する（下記参照）。
    fn group_id(&self) -> Option<&str>;
}

/// レコードが属する分割の種類。
///
/// `split::Split`（train / validation / test の 3 値）とは異なり、利用者が
/// 独立に用意した評価データ（REQ-17 でツールが切り出す test と区別される）を
/// 表す `Evaluation` を持つため、別の型として定義する。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Partition {
    Train,
    Validation,
    Test,
    Evaluation,
}

/// 検査対象の分割一式。
///
/// 名前付きフィールドにすることで「同じ分割を 2 回渡す」状態を型の上で
/// 表現できなくする（coding-rust.md「壊れた値を表現できない型にする」）。
/// `train` は必須（漏洩・跨ぎの基準になる分割）、他は利用者の構成に応じて
/// 存在しない場合があるため `Option` とする。
pub struct Partitions<'a, R> {
    pub train: &'a [R],
    pub validation: Option<&'a [R]>,
    pub test: Option<&'a [R]>,
    pub evaluation: Option<&'a [R]>,
}

impl<'a, R> Partitions<'a, R> {
    /// train 以外の分割を `(Partition, &[R])` の一覧として返す（内部の走査順を集約する）。
    ///
    /// 順序は validation → test → evaluation に固定する（レポートの決定性のため。
    /// 分割自体の走査順は最終的な `Vec` のソートで担保するが、本メソッドの
    /// 固定順は入力側の反復にも決定性を持たせる）。
    fn other_partitions(&self) -> Vec<(Partition, &'a [R])> {
        let mut result = Vec::new();
        if let Some(validation) = self.validation {
            result.push((Partition::Validation, validation));
        }
        if let Some(test) = self.test {
            result.push((Partition::Test, test));
        }
        if let Some(evaluation) = self.evaluation {
            result.push((Partition::Evaluation, evaluation));
        }
        result
    }

    /// train を含む全分割のスライスを 1 つの一覧として返す（総件数・ID 長の
    /// 検証で全レコードを走査するための内部ヘルパー。[`validate_resource_limits`]
    /// から使う）。
    fn all_partitions(&self) -> Vec<&'a [R]> {
        let mut result = vec![self.train];
        result.extend(
            self.other_partitions()
                .into_iter()
                .map(|(_, records)| records),
        );
        result
    }
}

/// [`find_input_leaks`]・[`find_group_straddles`] の入口で、レコード総件数と
/// ID・group ID・入力の byte 長を検証する（REQ-39・security.md「ガード層: 資源の上限」。
/// reviewer 指摘 PR #195。本モジュール先頭のドキュメントコメント参照）。
///
/// `partitions` に含まれるスライス長の合計にのみ比例した処理で、入力件数に
/// 応じて増える追加アロケーションは行わない（分割数分〔最大 4 件〕の固定長
/// `Vec` のみ使う。違反を検出したら即座に打ち切る）。
pub fn validate_resource_limits<R: LeakCheckable>(
    partitions: &Partitions<'_, R>,
) -> Result<(), LeakCheckError> {
    let all_partitions = partitions.all_partitions();

    let total: usize = all_partitions.iter().map(|records| records.len()).sum();
    if total > MAX_LEAK_CHECK_RECORDS {
        return Err(LeakCheckError::TooManyRecords {
            total,
            limit: MAX_LEAK_CHECK_RECORDS,
        });
    }

    // 個別長の検証と並行して、ID・group ID・入力の byte 長の合計（全レコード分）を
    // 積算する。`saturating_add` を使い、万一の桁あふれでラップアラウンドして
    // 上限判定をすり抜けることを防ぐ（オーバーフロー時は `usize::MAX` に張り付き、
    // 以降 `MAX_LEAK_CHECK_TOTAL_BYTES` 判定に必ず抵触する）。
    let mut total_bytes: usize = 0;
    for records in all_partitions {
        for record in records {
            let id_len = record.id().len();
            if id_len > MAX_LEAK_CHECK_ID_BYTES {
                return Err(LeakCheckError::IdTooLong {
                    len: id_len,
                    limit: MAX_LEAK_CHECK_ID_BYTES,
                });
            }
            // group_id は任意項目（`inspect::ValidRecord::group_id` が
            // `Option<String>`）のため、存在する場合のみ長さを検証する。
            // 欠損（`None`）はここでは 0 byte として扱い、合計 byte 数
            // （`total_bytes`）にも加算しない。
            let group_id_len = record.group_id().map(str::len).unwrap_or(0);
            if group_id_len > MAX_LEAK_CHECK_ID_BYTES {
                return Err(LeakCheckError::IdTooLong {
                    len: group_id_len,
                    limit: MAX_LEAK_CHECK_ID_BYTES,
                });
            }
            let input_len = record.input().len();
            if input_len > MAX_LEAK_CHECK_INPUT_BYTES {
                return Err(LeakCheckError::InputTooLong {
                    len: input_len,
                    limit: MAX_LEAK_CHECK_INPUT_BYTES,
                });
            }

            total_bytes = total_bytes
                .saturating_add(id_len)
                .saturating_add(group_id_len)
                .saturating_add(input_len);
            if total_bytes > MAX_LEAK_CHECK_TOTAL_BYTES {
                return Err(LeakCheckError::TotalBytesExceeded {
                    total: total_bytes,
                    limit: MAX_LEAK_CHECK_TOTAL_BYTES,
                });
            }
        }
    }

    Ok(())
}

/// train の入力と byte 完全一致した 1 件（1 つの入力 × 1 つの相手分割）。
///
/// `train_ids`・`other_ids` は該当する **全 ID** を昇順で保持する（PoC-9 の
/// `find_train_test_leak` が入力をキーに `test_id` を上書きし、同一入力を持つ
/// 評価側の複数行のうち最後の 1 件しか残せなかった欠陥を、本実装では直す）。
///
/// `train_ids` は `Rc<[String]>` にしている。同一の train 入力が複数の相手分割
/// （validation・test・evaluation）へ漏洩すると、その入力に対応する
/// `InputLeak` が分割の数だけ生成されるが、`Rc` で共有することで ID 列本体の
/// 複製をせず参照カウントの増加のみに留める（reviewer 指摘 PR #195・REQ-39
/// 資源の上限の観点。外部データの重複度に比例して出力サイズが増幅するのを防ぐ）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputLeak {
    pub partition: Partition,
    pub train_ids: Rc<[String]>,
    pub other_ids: Vec<String>,
}

/// 入力漏洩の検出結果一式。
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct InputLeakReport {
    /// 並び順は `(partition, other_ids 全体の辞書順, train_ids 全体の辞書順)` の
    /// 昇順で決定的にする。入力 byte そのものの順には依存させない（本文の内容が
    /// 並び順から推測されることも避ける。security.md）。
    ///
    /// `other_ids` の最小値だけを比較キーにすると、ID の一意性が保証されない
    /// 場合（同一 ID を持つ異なるレコードが複数の分割に存在する等）に
    /// 最小値が同点になり得て、その際の順序が `other_index`（`BTreeMap`）の
    /// 入力 byte の走査順に依存してしまう（reviewer 指摘 PR #195）。
    /// `other_ids`（各要素は昇順ソート済み）を丸ごと辞書式比較しても、
    /// `partition` と `other_ids` の両方が同点になり得る（同一の `other_ids`
    /// 集合を持つ異なる入力が、異なる `train_ids` を伴って複数存在する場合。
    /// Bugbot / Codex 指摘 PR #195）。この場合に区別する情報を欠くと
    /// `other_index`（`BTreeMap`）の入力 byte 走査順が残ってしまうため、
    /// `train_ids` も比較キーへ加えて入力 byte 順への依存を断つ。
    pub leaks: Vec<InputLeak>,
}

impl InputLeakReport {
    /// 漏洩の組の件数（`(入力, 相手分割)` の組ごとに 1 件。同一入力が
    /// validation・test など複数分割へ漏洩した場合は、その分だけ重複して
    /// 数える。
    ///
    /// 「異なり入力数」（reviewer 指摘 PR #195: 旧名 `distinct_inputs` は
    /// この値を異なり入力数だと誤解させた）とは異なる。異なり入力数は
    /// `leaks` から入力 byte 単位で別途数え直す必要があり、本メソッドは
    /// その代わりにならない。
    pub fn leak_pair_count(&self) -> usize {
        self.leaks.len()
    }

    /// 相手側（train 以外）で漏洩と判定された行数の合計。
    pub fn leaked_rows(&self) -> usize {
        self.leaks.iter().map(|leak| leak.other_ids.len()).sum()
    }
}

/// train の入力と、他の分割（validation・test・evaluation）の入力が byte 完全一致する
/// 組を検出する（REQ-16 異常系・TASK-16.2-1。PoC-9 `find_train_test_leak` 相当）。
///
/// train 側の索引は入力 byte を借用したまま構築するため、`partitions` に含まれる
/// レコードの total 件数に比例した処理になる（コピーは発生しない）。
///
/// # Errors
///
/// [`validate_resource_limits`] がレコード総件数・ID 長・入力 byte 長の上限超過を検出した場合、
/// 集計処理（train 側 ID の複製・索引化を含む）を一切行わずに [`LeakCheckError`] を返す
/// （REQ-39・security.md「ガード層: 資源の上限」）。
pub fn find_input_leaks<R: LeakCheckable>(
    partitions: &Partitions<'_, R>,
) -> Result<InputLeakReport, LeakCheckError> {
    validate_resource_limits(partitions)?;

    // train: 入力 byte -> train 側の全 ID（同一入力の train 内重複も漏れなく保持する）。
    // この時点では借用（`&str`）のみで、String の複製は発生しない。
    let mut train_index: BTreeMap<&[u8], Vec<&str>> = BTreeMap::new();
    for record in partitions.train {
        train_index
            .entry(record.input())
            .or_default()
            .push(record.id());
    }
    for ids in train_index.values_mut() {
        ids.sort_unstable();
    }

    // 相手側（validation・test・evaluation）の索引を先に全分割分構築する
    // （これも借用のみで、まだ train 側 ID の複製はしない）。実際に漏洩と
    // 判定される入力の集合を先に確定させてから、その入力についてだけ
    // train 側 ID を所有化するため（reviewer 指摘 PR #195 追加分・REQ-39
    // 「資源の上限」: 漏洩が 0 件の入力に対しても train 側の全 ID を複製すると、
    // 許容件数の上限〔`MAX_LEAK_CHECK_RECORDS`〕いっぱいに ID 長の上限
    // 〔`MAX_LEAK_CHECK_ID_BYTES`〕の ID を与えるだけで漏洩の有無に関係なく
    // 数百 MB 規模のメモリを追加確保してしまう）。
    // 入力 byte -> その分割側の全 ID（借用）の索引。分割の種類ごとに 1 つ持つ。
    type InputIndex<'a> = BTreeMap<&'a [u8], Vec<&'a str>>;

    let other_indices: Vec<(Partition, InputIndex<'_>)> = partitions
        .other_partitions()
        .into_iter()
        .map(|(partition, records)| {
            let mut other_index: InputIndex<'_> = BTreeMap::new();
            for record in records {
                other_index
                    .entry(record.input())
                    .or_default()
                    .push(record.id());
            }
            (partition, other_index)
        })
        .collect();

    // 入力 byte -> その入力の train 側 ID 列（所有権を持つ String に変換済み）を
    // `Rc<[String]>` として遅延構築するキャッシュ。実際に他分割へ漏洩したと
    // 判明した入力についてのみ 1 度だけ構築し、以降は clone（参照カウントの
    // 増加のみ）で使い回す。漏洩していない train 入力は所有化しない。
    let mut train_ids_shared: BTreeMap<&[u8], Rc<[String]>> = BTreeMap::new();

    let mut leaks: Vec<InputLeak> = Vec::new();
    for (partition, other_index) in &other_indices {
        for (input, other_ids) in other_index {
            let Some(train_ids_borrowed) = train_index.get(input) else {
                continue;
            };
            let train_ids = train_ids_shared.entry(input).or_insert_with(|| {
                train_ids_borrowed
                    .iter()
                    .map(|id| (*id).to_string())
                    .collect()
            });
            let mut other_ids = other_ids.clone();
            other_ids.sort_unstable();
            leaks.push(InputLeak {
                partition: *partition,
                train_ids: Rc::clone(train_ids),
                other_ids: other_ids.into_iter().map(str::to_string).collect(),
            });
        }
    }

    leaks.sort_by(|a, b| {
        a.partition
            .cmp(&b.partition)
            .then_with(|| a.other_ids.cmp(&b.other_ids))
            .then_with(|| a.train_ids.cmp(&b.train_ids))
    });

    Ok(InputLeakReport { leaks })
}

/// 2 つ以上の分割に跨って現れた group を 1 件表す。
///
/// `members` は分割ごとの ID 一覧を保持し、各分割内の ID は昇順とする。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupStraddle {
    pub group_id: String,
    pub members: BTreeMap<Partition, Vec<String>>,
}

/// group 跨ぎの検出結果一式。
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct GroupStraddleReport {
    /// `group_id` の昇順で並べる。
    pub straddles: Vec<GroupStraddle>,
}

/// 同一 `group_id` を持つレコードが 2 つ以上の分割（train を含むどの組でもよい）に
/// 分散している group を検出する（REQ-16 異常系・TASK-16.2-1。PoC-9 `find_cross_group` 相当）。
///
/// 入力が異なる場合（[`find_input_leaks`] では検出できないケース）でも、group が
/// 一致していれば検出できることが本検出の存在意義（PoC-9 の group-straddle
/// フィクスチャが示す観点）。
///
/// `group_id` が欠損（[`LeakCheckable::group_id`] が `None`）のレコードは
/// 群跨ぎ判定の対象から除外する。欠損値を空文字列等の番兵値へ揃えて比較すると、
/// 元々 group が無関係な複数レコードが同一 group として誤検出されるため
/// （reviewer 指摘 PR #195・crates/data/src/leak.rs:172）、除外することで
/// 誤検出を避ける。
///
/// # Errors
///
/// [`validate_resource_limits`] がレコード総件数・ID 長・入力 byte 長の上限超過を検出した場合、
/// 索引化を一切行わずに [`LeakCheckError`] を返す（REQ-39・
/// security.md「ガード層: 資源の上限」）。
pub fn find_group_straddles<R: LeakCheckable>(
    partitions: &Partitions<'_, R>,
) -> Result<GroupStraddleReport, LeakCheckError> {
    validate_resource_limits(partitions)?;

    // group_id -> (分割 -> その分割内の ID 一覧)。
    let mut by_group: BTreeMap<&str, BTreeMap<Partition, Vec<&str>>> = BTreeMap::new();

    let mut all_partitions: Vec<(Partition, &[R])> = vec![(Partition::Train, partitions.train)];
    all_partitions.extend(partitions.other_partitions());

    for (partition, records) in all_partitions {
        for record in records {
            // group_id が欠損しているレコードは群跨ぎ判定から除外する
            // （上記ドキュメントコメント参照。欠損値を番兵値へ揃えないことで
            // 無関係なレコード同士を誤って同一 group と扱わない）。
            let Some(group_id) = record.group_id() else {
                continue;
            };
            by_group
                .entry(group_id)
                .or_default()
                .entry(partition)
                .or_default()
                .push(record.id());
        }
    }

    let mut straddles: Vec<GroupStraddle> = Vec::new();
    for (group_id, members_by_partition) in by_group {
        if members_by_partition.len() < 2 {
            continue;
        }
        let mut members: BTreeMap<Partition, Vec<String>> = BTreeMap::new();
        for (partition, mut ids) in members_by_partition {
            ids.sort_unstable();
            members.insert(partition, ids.into_iter().map(str::to_string).collect());
        }
        straddles.push(GroupStraddle {
            group_id: group_id.to_string(),
            members,
        });
    }

    Ok(GroupStraddleReport { straddles })
}

/// 漏洩と group 跨ぎをまとめて呼ぶ受け口。
///
/// #42（メタデータ混入・矛盾レコード検出）や #39（検査レポート全体）が
/// 統合するときの入口として用意する。
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LeakageReport {
    pub input_leaks: InputLeakReport,
    pub group_straddles: GroupStraddleReport,
}

/// [`find_input_leaks`] と [`find_group_straddles`] をまとめて実行する。
///
/// # Errors
///
/// [`validate_resource_limits`] がレコード総件数・ID 長・入力 byte 長の上限超過を検出した場合、
/// 両方の検出を実行せずに [`LeakCheckError`] を返す。
pub fn inspect_leakage<R: LeakCheckable>(
    partitions: &Partitions<'_, R>,
) -> Result<LeakageReport, LeakCheckError> {
    Ok(LeakageReport {
        input_leaks: find_input_leaks(partitions)?,
        group_straddles: find_group_straddles(partitions)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// テスト用の最小レコード（`split.rs` の `TestRecord` と同じ方針。動的な
    /// ID・group_id・input を扱うため `String`/`Vec<u8>` で保持する）。
    struct TestRecord {
        id: String,
        input: Vec<u8>,
        group_id: Option<String>,
    }

    impl LeakCheckable for TestRecord {
        fn id(&self) -> &str {
            &self.id
        }
        fn input(&self) -> &[u8] {
            &self.input
        }
        fn group_id(&self) -> Option<&str> {
            self.group_id.as_deref()
        }
    }

    fn record(id: &str, input: &str, group_id: &str) -> TestRecord {
        TestRecord {
            id: id.to_string(),
            input: input.as_bytes().to_vec(),
            group_id: Some(group_id.to_string()),
        }
    }

    /// `inspect::ValidRecord::group_id` が `None`（group_id 未指定）の場合に
    /// 相当するレコードを作る（reviewer 指摘 PR #195・crates/data/src/leak.rs:172
    /// の回帰防止用ヘルパー）。
    fn record_without_group_id(id: &str, input: &str) -> TestRecord {
        TestRecord {
            id: id.to_string(),
            input: input.as_bytes().to_vec(),
            group_id: None,
        }
    }

    fn empty_partitions<'a>(train: &'a [TestRecord]) -> Partitions<'a, TestRecord> {
        Partitions {
            train,
            validation: None,
            test: None,
            evaluation: None,
        }
    }

    /// REQ-16 異常系・TASK-16.2-1: 全分割が空なら漏洩 0 件・跨ぎ 0 件。
    #[test]
    fn req16_task16_2_1_all_empty_yields_no_findings() {
        let train: Vec<TestRecord> = Vec::new();
        let partitions = empty_partitions(&train);
        let leak_report = find_input_leaks(&partitions).expect("上限以下の入力");
        let straddle_report = find_group_straddles(&partitions).expect("上限以下の入力");
        assert_eq!(leak_report.leaks, Vec::new());
        assert_eq!(straddle_report.straddles, Vec::new());
    }

    /// REQ-16 異常系・TASK-16.2-1: train だけ存在し他分割が無い場合も漏洩 0 件・跨ぎ 0 件。
    #[test]
    fn req16_task16_2_1_train_only_yields_no_findings() {
        let train = vec![record("t1", "hello", "g1")];
        let partitions = empty_partitions(&train);
        assert_eq!(
            find_input_leaks(&partitions).expect("上限以下の入力").leaks,
            Vec::new()
        );
        assert_eq!(
            find_group_straddles(&partitions)
                .expect("上限以下の入力")
                .straddles,
            Vec::new()
        );
    }

    /// REQ-16 異常系・TASK-16.2-1（PoC-9 の上書き欠陥の回帰防止）: train の t1 と
    /// 同一入力を持つ評価側の複数行（e1・e2）が、どちらも欠落せず全件報告される。
    #[test]
    fn req16_task16_2_1_multiple_other_rows_with_same_input_all_reported() {
        let train = vec![record("t1", "same input", "g_train")];
        let evaluation = vec![
            record("e2", "same input", "g_e2"),
            record("e1", "same input", "g_e1"),
        ];
        let partitions = Partitions {
            train: &train,
            validation: None,
            test: None,
            evaluation: Some(&evaluation),
        };

        let report = find_input_leaks(&partitions).expect("上限以下の入力");
        assert_eq!(report.leaks.len(), 1);
        let leak = &report.leaks[0];
        assert_eq!(leak.partition, Partition::Evaluation);
        assert_eq!(&*leak.train_ids, ["t1".to_string()]);
        assert_eq!(leak.other_ids, vec!["e1".to_string(), "e2".to_string()]);
        assert_eq!(report.leak_pair_count(), 1);
        assert_eq!(report.leaked_rows(), 2);
    }

    /// REQ-16 異常系・TASK-16.2-1: train 側で複数行が同一入力の場合、`train_ids` に
    /// すべて含まれ、`leak_pair_count` は 1 のままになる（train 内の重複は漏洩件数を
    /// 増やしも減らしもしない。同一分割内の重複自体は #42 の担当）。
    #[test]
    fn req16_task16_2_1_duplicate_train_input_all_ids_kept_leak_pair_count_unchanged() {
        let train = vec![
            record("t1", "dup input", "g1"),
            record("t2", "dup input", "g2"),
        ];
        let test = vec![record("x1", "dup input", "g3")];
        let partitions = Partitions {
            train: &train,
            validation: None,
            test: Some(&test),
            evaluation: None,
        };

        let report = find_input_leaks(&partitions).expect("上限以下の入力");
        assert_eq!(report.leaks.len(), 1);
        assert_eq!(
            &*report.leaks[0].train_ids,
            ["t1".to_string(), "t2".to_string()]
        );
        assert_eq!(report.leak_pair_count(), 1);
        assert_eq!(report.leaked_rows(), 1);
    }

    /// REQ-16 異常系・TASK-16.2-1（reviewer 指摘の回帰防止）: 同一入力が
    /// validation・test の複数分割へ漏洩した場合、`leaks` は分割ごとに 1 件
    /// （組ごと）ずつ計 2 件になり、`leak_pair_count()`（= `leaks.len()`）は
    /// 「異なり入力数」ではなく「漏洩の組の件数」として 2 を返す
    /// （異なり入力数そのものは 1）。この差は `leak_pair_count` のドキュメント
    /// コメントに明記済み。#39（検査レポート集約）でこの値を異なり入力数として
    /// 使わないよう、値の意味をここで固定する。
    ///
    /// 同時に、`train_ids`（`Rc<[String]>`）が 2 件の `InputLeak` 間で同じ
    /// アロケーションを共有していること（`Rc::ptr_eq`）も確認し、分割数に応じて
    /// ID 列本体が複製されない（reviewer 指摘 PR #195・REQ-39）ことを回帰防止する。
    #[test]
    fn req16_task16_2_1_same_input_leaked_into_multiple_partitions_counts_per_partition() {
        let train = vec![record("t1", "x", "g_train")];
        let validation = vec![record("v1", "x", "g_v")];
        let test = vec![record("x1", "x", "g_test")];
        let partitions = Partitions {
            train: &train,
            validation: Some(&validation),
            test: Some(&test),
            evaluation: None,
        };

        let report = find_input_leaks(&partitions).expect("上限以下の入力");
        assert_eq!(report.leaks.len(), 2);
        assert_eq!(report.leak_pair_count(), 2);
        assert_eq!(report.leaked_rows(), 2);
        assert_eq!(report.leaks[0].partition, Partition::Validation);
        assert_eq!(report.leaks[1].partition, Partition::Test);
        assert!(
            Rc::ptr_eq(&report.leaks[0].train_ids, &report.leaks[1].train_ids),
            "同じ train 入力に対する train_ids は Rc で共有され、分割ごとに複製されない"
        );
    }

    /// REQ-16 異常系・TASK-16.2-1（reviewer 指摘 PR #195 追加分・REQ-39
    /// 「ガード層: 資源の上限」）: 漏洩していない train 入力の ID は、漏洩と無関係な
    /// 大量の train レコードが存在しても複製・報告されない。漏洩と判定されるのは
    /// 実際に他分割へ入力が一致した 1 件のみで、レポートにはその ID だけが現れる
    /// （train 側の全 ID を無条件で所有化する旧実装は、この境界に関係なく全件を
    /// 複製していた）。
    #[test]
    fn req16_task16_2_1_non_leaking_train_records_are_not_materialized_into_report() {
        let mut train: Vec<TestRecord> = (0..500)
            .map(|i| record(&format!("t{i}"), &format!("only-in-train-{i}"), "g"))
            .collect();
        train.push(record("t_leak", "shared-input", "g_leak"));
        let validation = vec![record("v1", "shared-input", "g_v")];
        let partitions = Partitions {
            train: &train,
            validation: Some(&validation),
            test: None,
            evaluation: None,
        };

        let report = find_input_leaks(&partitions).expect("上限以下の入力");
        assert_eq!(report.leak_pair_count(), 1);
        assert_eq!(report.leaks[0].train_ids.as_ref(), ["t_leak".to_string()]);
        assert_eq!(report.leaks[0].other_ids, vec!["v1".to_string()]);
    }

    /// REQ-16 異常系・TASK-16.2-1: byte が 1 つでも異なれば漏洩としない
    /// （末尾空白・全角半角の違いを含む byte 完全一致という現行仕様の固定）。
    /// 正規化した入力での照合は後続の課題（依存の承認・TASK-15.5 が前提）。
    #[test]
    fn req16_task16_2_1_byte_difference_is_not_a_leak() {
        let train = vec![record("t1", "hello", "g1")];
        let test = vec![
            record("x1", "hello ", "g2"), // 末尾空白の違い
            record("x2", "ｈello", "g3"), // 全角文字混入
        ];
        let partitions = Partitions {
            train: &train,
            validation: None,
            test: Some(&test),
            evaluation: None,
        };
        assert_eq!(
            find_input_leaks(&partitions).expect("上限以下の入力").leaks,
            Vec::new()
        );
    }

    /// REQ-16 異常系・TASK-16.2-1: validation・evaluation それぞれについて
    /// `partition` が正しく付く。
    #[test]
    fn req16_task16_2_1_partition_is_tagged_correctly_for_each_split() {
        let train = vec![record("t1", "v-leak", "g_v"), record("t2", "e-leak", "g_e")];
        let validation = vec![record("v1", "v-leak", "g_v2")];
        let evaluation = vec![record("e1", "e-leak", "g_e2")];
        let partitions = Partitions {
            train: &train,
            validation: Some(&validation),
            test: None,
            evaluation: Some(&evaluation),
        };

        let report = find_input_leaks(&partitions).expect("上限以下の入力");
        assert_eq!(report.leaks.len(), 2);
        assert_eq!(report.leaks[0].partition, Partition::Validation);
        assert_eq!(report.leaks[1].partition, Partition::Evaluation);
    }

    /// REQ-16 異常系・TASK-16.2-1: group が train・validation・test の 3 分割に
    /// 跨る場合、`members` に 3 分割すべてが並ぶ。
    #[test]
    fn req16_task16_2_1_group_straddling_three_partitions_detected() {
        let train = vec![record("t1", "input-t", "shared-group")];
        let validation = vec![record("v1", "input-v", "shared-group")];
        let test = vec![record("x1", "input-x", "shared-group")];
        let partitions = Partitions {
            train: &train,
            validation: Some(&validation),
            test: Some(&test),
            evaluation: None,
        };

        let report = find_group_straddles(&partitions).expect("上限以下の入力");
        assert_eq!(report.straddles.len(), 1);
        let straddle = &report.straddles[0];
        assert_eq!(straddle.group_id, "shared-group");
        assert_eq!(
            straddle.members.get(&Partition::Train),
            Some(&vec!["t1".to_string()])
        );
        assert_eq!(
            straddle.members.get(&Partition::Validation),
            Some(&vec!["v1".to_string()])
        );
        assert_eq!(
            straddle.members.get(&Partition::Test),
            Some(&vec!["x1".to_string()])
        );
        // 入力が全て異なっていても group 一致だけで検出できることを確認する
        // （入力完全一致では検出できないケース。跨ぎ検出の存在意義）。
        assert_eq!(
            find_input_leaks(&partitions).expect("上限以下の入力").leaks,
            Vec::new()
        );
    }

    /// REQ-16 異常系・TASK-16.2-1: validation と test の間だけの跨ぎ（train を
    /// 含まない）も検出する。
    #[test]
    fn req16_task16_2_1_straddle_between_validation_and_test_without_train() {
        let train = vec![record("t1", "unrelated", "g_train_only")];
        let validation = vec![record("v1", "input-v", "shared-group")];
        let test = vec![record("x1", "input-x", "shared-group")];
        let partitions = Partitions {
            train: &train,
            validation: Some(&validation),
            test: Some(&test),
            evaluation: None,
        };

        let report = find_group_straddles(&partitions).expect("上限以下の入力");
        assert_eq!(report.straddles.len(), 1);
        let straddle = &report.straddles[0];
        assert_eq!(straddle.group_id, "shared-group");
        assert_eq!(straddle.members.len(), 2);
        assert!(!straddle.members.contains_key(&Partition::Train));
    }

    /// REQ-16 異常系・TASK-16.2-1（reviewer 指摘 PR #195・crates/data/src/leak.rs:172
    /// の回帰防止）: `group_id` が欠損（`None`）しているレコードは、
    /// 他のレコードも `group_id` が欠損している場合であっても同一 group として
    /// 誤って群跨ぎ判定されない（欠損値を空文字列等の番兵値へ揃えて比較すると、
    /// 無関係なレコード同士が誤検出されてしまう懸念への回帰防止）。
    #[test]
    fn req16_task16_2_1_missing_group_id_is_excluded_from_straddle_detection() {
        let train = vec![record_without_group_id("t1", "train-input")];
        let test = vec![record_without_group_id("x1", "test-input")];
        let partitions = Partitions {
            train: &train,
            validation: None,
            test: Some(&test),
            evaluation: None,
        };

        let report = find_group_straddles(&partitions).expect("上限以下の入力");
        assert_eq!(
            report.straddles,
            Vec::new(),
            "group_id 欠損レコード同士が誤って同一 group として検出された"
        );
    }

    /// REQ-16 異常系・TASK-16.2-1（同上の回帰防止）: `group_id` が実際に一致する
    /// レコードの跨ぎ検出は、他に `group_id` 欠損のレコードが混在していても
    /// 影響を受けない（欠損レコードは判定から除外されるのみで、有効な group_id
    /// を持つレコード間の検出には影響しない）。
    #[test]
    fn req16_task16_2_1_missing_group_id_does_not_affect_other_straddle_detection() {
        let train = vec![
            record("t1", "input-t", "shared-group"),
            record_without_group_id("t2", "no-group-input"),
        ];
        let test = vec![
            record("x1", "input-x", "shared-group"),
            record_without_group_id("x2", "another-no-group-input"),
        ];
        let partitions = Partitions {
            train: &train,
            validation: None,
            test: Some(&test),
            evaluation: None,
        };

        let report = find_group_straddles(&partitions).expect("上限以下の入力");
        assert_eq!(report.straddles.len(), 1);
        assert_eq!(report.straddles[0].group_id, "shared-group");
    }

    /// REQ-16 異常系・TASK-16.2-1: 利用者の評価データ（Evaluation）側への漏洩と
    /// 跨ぎも、partition = Evaluation として検出される。
    #[test]
    fn req16_task16_2_1_evaluation_partition_leak_and_straddle_detected() {
        let train = vec![record("t1", "leak-input", "shared-group")];
        let evaluation = vec![record("e1", "leak-input", "shared-group")];
        let partitions = Partitions {
            train: &train,
            validation: None,
            test: None,
            evaluation: Some(&evaluation),
        };

        let leak_report = find_input_leaks(&partitions).expect("上限以下の入力");
        assert_eq!(leak_report.leaks.len(), 1);
        assert_eq!(leak_report.leaks[0].partition, Partition::Evaluation);

        let straddle_report = find_group_straddles(&partitions).expect("上限以下の入力");
        assert_eq!(straddle_report.straddles.len(), 1);
        assert!(
            straddle_report.straddles[0]
                .members
                .contains_key(&Partition::Evaluation)
        );
    }

    /// REQ-16 異常系・TASK-16.2-1: レコード順を入れ替えてもレポートが `==` で一致する
    /// （決定性。evaluation-contract.md「決定性と証拠の種別」）。
    #[test]
    fn req16_task16_2_1_report_is_order_independent() {
        let train_a = vec![
            record("t1", "input-1", "g1"),
            record("t2", "input-2", "g2"),
            record("t3", "input-3", "shared-group"),
        ];
        // train_a の逆順（入力・group_id は同じまま順序だけ変える）。
        let train_b = vec![
            record("t3", "input-3", "shared-group"),
            record("t2", "input-2", "g2"),
            record("t1", "input-1", "g1"),
        ];

        let test_a = vec![
            record("x1", "input-1", "gx1"),
            record("x2", "input-x", "shared-group"),
        ];
        let test_b = vec![
            record("x2", "input-x", "shared-group"),
            record("x1", "input-1", "gx1"),
        ];

        let partitions_a = Partitions {
            train: &train_a,
            validation: None,
            test: Some(&test_a),
            evaluation: None,
        };
        let partitions_b = Partitions {
            train: &train_b,
            validation: None,
            test: Some(&test_b),
            evaluation: None,
        };

        assert_eq!(
            find_input_leaks(&partitions_a),
            find_input_leaks(&partitions_b)
        );
        assert_eq!(
            find_group_straddles(&partitions_a),
            find_group_straddles(&partitions_b)
        );
    }

    /// REQ-16 異常系・TASK-16.2-1（reviewer 指摘 PR #195・Bugbot / Codex）:
    /// `partition`・`other_ids` が同点でも `train_ids` が異なれば、その
    /// `train_ids` の辞書順で並び順が決まり、`other_index`（`BTreeMap`）の
    /// 入力 byte 走査順には依存しない。
    ///
    /// 2 つの入力 `"leak-input-A"`・`"leak-input-B"`（byte 順で A < B）を
    /// それぞれ異なる train ID（"tZ"・"tA"）に対応させ、両方とも test 側で
    /// 同じ ID `"x1"` から漏洩したことにして `other_ids` を同点にする。
    /// `train_ids` を比較キーに加えないと、押し込み順（`other_index` の
    /// 入力 byte 順 = A→B）がそのまま安定ソートで残り、A（train_ids=["tZ"]）
    /// が先頭に来てしまう。`train_ids` の辞書順（"tA" < "tZ"）で正しく
    /// B・A の順になることを確認する。
    #[test]
    fn req16_task16_2_1_leak_order_breaks_tie_by_train_ids() {
        let train = vec![
            record("tZ", "leak-input-A", "gz"),
            record("tA", "leak-input-B", "ga"),
        ];
        let test = vec![
            record("x1", "leak-input-A", "gz"),
            record("x1", "leak-input-B", "ga"),
        ];
        let partitions = Partitions {
            train: &train,
            validation: None,
            test: Some(&test),
            evaluation: None,
        };

        let report = find_input_leaks(&partitions).expect("上限以下の入力");
        assert_eq!(report.leaks.len(), 2);
        // 同点（partition・other_ids）の 2 件が train_ids の辞書順（"tA" < "tZ"）で並ぶ。
        assert_eq!(report.leaks[0].train_ids.as_ref(), ["tA".to_string()]);
        assert_eq!(report.leaks[0].other_ids, vec!["x1".to_string()]);
        assert_eq!(report.leaks[1].train_ids.as_ref(), ["tZ".to_string()]);
        assert_eq!(report.leaks[1].other_ids, vec!["x1".to_string()]);
    }

    /// REQ-16 異常系・TASK-16.2-1: `inspect_leakage` が両方の検出結果をまとめて返す。
    #[test]
    fn req16_task16_2_1_inspect_leakage_combines_both_reports() {
        let train = vec![record("t1", "leak-input", "shared-group")];
        let test = vec![record("x1", "leak-input", "shared-group")];
        let partitions = Partitions {
            train: &train,
            validation: None,
            test: Some(&test),
            evaluation: None,
        };

        let combined = inspect_leakage(&partitions).expect("上限以下の入力");
        assert_eq!(
            combined.input_leaks,
            find_input_leaks(&partitions).expect("上限以下の入力")
        );
        assert_eq!(
            combined.group_straddles,
            find_group_straddles(&partitions).expect("上限以下の入力")
        );
    }

    /// REQ-16 異常系・TASK-16.2-1（reviewer 指摘 PR #195）: レコード総件数が
    /// `MAX_LEAK_CHECK_RECORDS` を超えると、索引化・複製を一切行わずに
    /// `LeakCheckError::TooManyRecords` を返す。
    #[test]
    fn req16_task16_2_1_rejects_record_count_over_limit() {
        let train: Vec<TestRecord> = (0..=MAX_LEAK_CHECK_RECORDS)
            .map(|i| record(&format!("t{i}"), "x", "g"))
            .collect();
        let partitions = empty_partitions(&train);

        assert_eq!(
            find_input_leaks(&partitions),
            Err(LeakCheckError::TooManyRecords {
                total: MAX_LEAK_CHECK_RECORDS + 1,
                limit: MAX_LEAK_CHECK_RECORDS,
            })
        );
        assert_eq!(
            find_group_straddles(&partitions),
            Err(LeakCheckError::TooManyRecords {
                total: MAX_LEAK_CHECK_RECORDS + 1,
                limit: MAX_LEAK_CHECK_RECORDS,
            })
        );
    }

    /// REQ-16 異常系・TASK-16.2-1（reviewer 指摘 PR #195）: ID が
    /// `MAX_LEAK_CHECK_ID_BYTES` を超える長さの場合、ID の内容を含めずに
    /// `LeakCheckError::IdTooLong` を返す（security.md「秘密情報の混入防止」）。
    #[test]
    fn req16_task16_2_1_rejects_id_over_byte_limit() {
        let long_id = "a".repeat(MAX_LEAK_CHECK_ID_BYTES + 1);
        let train = vec![record(&long_id, "x", "g")];
        let partitions = empty_partitions(&train);

        assert_eq!(
            find_input_leaks(&partitions),
            Err(LeakCheckError::IdTooLong {
                len: MAX_LEAK_CHECK_ID_BYTES + 1,
                limit: MAX_LEAK_CHECK_ID_BYTES,
            })
        );
    }

    /// REQ-16 異常系・TASK-16.2-1（reviewer 指摘 PR #195）: group ID が
    /// `MAX_LEAK_CHECK_ID_BYTES` を超える長さの場合も同様に拒否する。
    #[test]
    fn req16_task16_2_1_rejects_group_id_over_byte_limit() {
        let long_group_id = "g".repeat(MAX_LEAK_CHECK_ID_BYTES + 1);
        let train = vec![record("t1", "x", &long_group_id)];
        let partitions = empty_partitions(&train);

        assert_eq!(
            find_group_straddles(&partitions),
            Err(LeakCheckError::IdTooLong {
                len: MAX_LEAK_CHECK_ID_BYTES + 1,
                limit: MAX_LEAK_CHECK_ID_BYTES,
            })
        );
    }

    /// REQ-16 異常系・TASK-16.2-1（codex-review 指摘 PR #195）: 入力（`input()`）が
    /// `MAX_LEAK_CHECK_INPUT_BYTES` を超える長さの場合、`BTreeMap` への索引化を
    /// 行わずに `LeakCheckError::InputTooLong` を返す（security.md
    /// 「ガード層: 資源の上限」・「秘密情報の混入防止」）。
    #[test]
    fn req16_task16_2_1_rejects_input_over_byte_limit() {
        let long_input = "x".repeat(MAX_LEAK_CHECK_INPUT_BYTES + 1);
        let train = vec![record("t1", &long_input, "g")];
        let partitions = empty_partitions(&train);

        assert_eq!(
            find_input_leaks(&partitions),
            Err(LeakCheckError::InputTooLong {
                len: MAX_LEAK_CHECK_INPUT_BYTES + 1,
                limit: MAX_LEAK_CHECK_INPUT_BYTES,
            })
        );
        assert_eq!(
            find_group_straddles(&partitions),
            Err(LeakCheckError::InputTooLong {
                len: MAX_LEAK_CHECK_INPUT_BYTES + 1,
                limit: MAX_LEAK_CHECK_INPUT_BYTES,
            })
        );
    }

    /// REQ-16 異常系・TASK-16.2-1（reviewer 指摘 PR #195 P0）: 件数上限
    /// （`MAX_LEAK_CHECK_RECORDS`）・個別長上限（`MAX_LEAK_CHECK_ID_BYTES`・
    /// `MAX_LEAK_CHECK_INPUT_BYTES`）をいずれも満たす入力でも、ID・group ID・
    /// 入力の byte 長の合計が `MAX_LEAK_CHECK_TOTAL_BYTES` を超えると、索引化・
    /// 複製を行わずに `LeakCheckError::TotalBytesExceeded` を返す（複製された
    /// ID が数百 MB 規模のメモリを確保しうる懸念への回帰防止）。
    #[test]
    fn req16_task16_2_1_rejects_total_bytes_over_limit_even_within_per_record_limits() {
        // 1 件あたり id + group_id + input で MAX_LEAK_CHECK_ID_BYTES 分の長さを持つ
        // レコードを、合計が MAX_LEAK_CHECK_TOTAL_BYTES を超えるだけの件数だけ
        // 用意する（件数自体は MAX_LEAK_CHECK_RECORDS を大きく下回る）。
        let per_record_bytes = MAX_LEAK_CHECK_ID_BYTES;
        let record_count = MAX_LEAK_CHECK_TOTAL_BYTES / per_record_bytes + 1;
        assert!(
            record_count < MAX_LEAK_CHECK_RECORDS,
            "件数上限にはまだ余裕がある入力で合計 byte 数だけを超過させるテスト前提が崩れている"
        );

        let long_id = "a".repeat(per_record_bytes);
        let train: Vec<TestRecord> = (0..record_count)
            .map(|_| TestRecord {
                id: long_id.clone(),
                input: Vec::new(),
                group_id: Some(String::new()),
            })
            .collect();
        let partitions = empty_partitions(&train);

        let result = find_input_leaks(&partitions);
        match result {
            Err(LeakCheckError::TotalBytesExceeded { total, limit }) => {
                assert!(total > MAX_LEAK_CHECK_TOTAL_BYTES);
                assert_eq!(limit, MAX_LEAK_CHECK_TOTAL_BYTES);
            }
            other => panic!("TotalBytesExceeded を期待したが {other:?} だった"),
        }
        assert!(matches!(
            find_group_straddles(&partitions),
            Err(LeakCheckError::TotalBytesExceeded { .. })
        ));
    }

    /// REQ-16 異常系・TASK-16.2-1（reviewer 指摘 PR #195 P2）: `other_ids` の
    /// 最小値だけでなく列全体を比較キーにすることで、複数の漏洩の `other_ids`
    /// 最小値が同点でも、実際の ID 集合が異なれば入力 byte の走査順（`BTreeMap`
    /// の順）に依存せず一意に順序が決まる（ID の一意性が保証されない場合の
    /// 並び順が入力 byte 順に依存しうるという懸念への回帰防止）。
    #[test]
    fn req16_task16_2_1_leak_order_uses_full_other_ids_not_only_minimum() {
        // 2 つの train 入力がそれぞれ test 側の異なる入力へ漏洩し、両方の
        // other_ids の最小値が同じ "id1" になるが、2 件目の集合には最小値の
        // 他に "id2" が追加で含まれる（最小値だけの比較では順序が同点になり、
        // その場合の実装依存の tie-break が入力 byte 順に依存しうる）。
        let train = vec![
            record("t_z", "z-input", "g1"),
            record("t_a", "a-input", "g2"),
        ];
        let test = vec![
            // "z-input"（BTreeMap 走査順では後）が other_ids = ["id1"] のみ。
            record("id1", "z-input", "gx1"),
            // "a-input"（BTreeMap 走査順では先）が other_ids = ["id1", "id2"]
            // （同一 ID "id1" が別レコードにも使われている、一意性非保証のケース）。
            record("id1", "a-input", "gx2"),
            record("id2", "a-input", "gx2"),
        ];
        let partitions = Partitions {
            train: &train,
            validation: None,
            test: Some(&test),
            evaluation: None,
        };

        let report = find_input_leaks(&partitions).expect("上限以下の入力");
        assert_eq!(report.leaks.len(), 2);
        // 全体比較では ["id1"] < ["id1", "id2"]（短い方が辞書式に先）となり、
        // 入力 byte の走査順（"a-input" が先）とは無関係に一意に定まる。
        assert_eq!(report.leaks[0].other_ids, vec!["id1".to_string()]);
        assert_eq!(
            report.leaks[1].other_ids,
            vec!["id1".to_string(), "id2".to_string()]
        );
    }
}
