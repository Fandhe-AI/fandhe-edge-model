//! group 単位分割ロジック（REQ-17・TASK-17.1-1）。
//!
//! 学習データを `train` / `validation` / `test` の 3 分割に分ける際、
//! 同一 group（`(intent, args)` 由来の識別子）が複数の split に跨ると、
//! 学習データと評価データの間で実質的な情報漏洩が起き、精度を過大評価する
//! （REQ-16 の漏洩検出と対をなす不変条件）。本モジュールは group を跨がない
//! 決定的な分割アルゴリズムのみを提供する。
//!
//! 分割の seed・規則・各分割のハッシュの記録と永続化・凍結は後続
//! TASK-17.1-2（#45）・TASK-17.2・TASK-17.3 の責務であり、本モジュールは
//! それらが消費できる構造化された分割結果（[`SplitResult`]）を返すところまでを担う。
//!
//! # Python 実装との関係
//!
//! 割付規則（`alloc_counts`）は `docs/spec/03-poc/scratch-classifier/scripts/split_train.py`
//! （PoC-9・PoC-10。REQ-17 の実測根拠）と同じ規則を移植したものだが、
//! 乱数系列（[`SplitMix64`] と Fisher–Yates シャッフル）は独立実装であり、
//! Python 側の出力とビット互換であることは保証しない。受け入れ条件は
//! 「同一 seed で分割すると同じ分割結果になる」という Rust 内での決定性のみである。
//!
//! # 資源上限について
//!
//! 本モジュールは `records` に渡されたスライス長に比例した処理のみを行い、
//! 追加の無制限アロケーションは行わない。呼び出し元の件数上限検査（REQ-39）は
//! 本モジュールの責務外とする（呼び出し側であるデータ検査層 TASK-16.1 が担う）。

use std::collections::BTreeMap;

/// group 単位分割の対象になるレコードが満たす最小の契約。
///
/// TASK-16.1 のデータ検査モジュールが持つ想定の具体的なレコード構造体には
/// 依存せず、`id`・`group_id`・層化用ラベルだけを要求する（層間の結合を最小化する）。
pub trait Groupable {
    /// レコード ID（分割結果の対応づけに使う）。
    ///
    /// 空文字列・重複の禁止は呼び出し側の検証責務とし、本モジュールでは
    /// 形式検査しない。
    fn id(&self) -> &str;

    /// group ID。同一 group のレコードは必ず同じ split に属する（REQ-17）。
    fn group_id(&self) -> &str;

    /// 層化に使うレコード単位のラベル（例: intent）。
    ///
    /// group 内の最頻ラベルを group の代表ラベルとして扱う
    /// （PoC-9 の `group_intent` 相当）。
    fn label(&self) -> &str;
}

/// 分割先（REQ-17: train / validation / test の 3 分割）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Split {
    Train,
    Validation,
    Test,
}

/// 分割比率（既定 0.8 / 0.1 / 0.1。PoC-9・PoC-10 の実測値）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitRatios {
    pub train: f64,
    pub validation: f64,
    pub test: f64,
}

impl Default for SplitRatios {
    fn default() -> Self {
        Self {
            train: 0.8,
            validation: 0.1,
            test: 0.1,
        }
    }
}

/// 浮動小数の合計比較の許容差（[coding-rust](../../../.claude/rules/coding-rust.md)。1e-9）。
const RATIO_SUM_TOLERANCE: f64 = 1e-9;

impl SplitRatios {
    /// 各比率が `[0.0, 1.0]` 区間かつ合計がおよそ 1.0（許容差 1e-9）であることを検証する。
    ///
    /// NaN・負値・区間外・合計の逸脱を `unwrap` / `expect` を使わず `Result` で拒否する
    /// （外部由来の設定値であっても panic させない。coding-rust.md）。
    fn validate(&self) -> Result<(), SplitError> {
        let values = [self.train, self.validation, self.test];
        for value in values {
            if !(0.0..=1.0).contains(&value) {
                // NaN は範囲比較が常に false になるため、この分岐で自然に拒否される。
                return Err(SplitError::InvalidRatios);
            }
        }
        let sum = self.train + self.validation + self.test;
        if (sum - 1.0).abs() > RATIO_SUM_TOLERANCE {
            return Err(SplitError::InvalidRatios);
        }
        Ok(())
    }
}

/// 1 ラベルあたりの group 件数の割付内訳（#45 が分割規則の記録に使う）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabelAllocation {
    pub label: String,
    pub n_groups: usize,
    pub train: usize,
    pub validation: usize,
    pub test: usize,
}

/// 分割結果一式。
///
/// フィールドはすべて `BTreeMap` とし、反復順序が実行のたびに変わらないようにする
/// （`HashMap` は使わない。決定性の不変条件を壊す典型的な原因のため。
/// [evaluation-contract](../../../.claude/rules/evaluation-contract.md)）。
/// 後続 #45（TASK-17.1-2）はこの型から各 split のレコード ID 集合を取り出してハッシュ化する。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitResult {
    pub by_record: BTreeMap<String, Split>,
    pub by_group: BTreeMap<String, Split>,
    pub per_label: Vec<LabelAllocation>,
    /// 分割規則の識別子（#45 が記録する「規則」に含める）。
    /// アルゴリズム（割付規則・PRNG の種類）を変えたら値を変える。
    pub rule_id: &'static str,
}

/// 割付規則・PRNG 実装の識別子。値を変えたら過去の分割結果との互換性が失われる。
const RULE_ID: &str = "group-stratified-v1/alloc-poc9/splitmix64-fisher-yates";

/// 分割の入力検証エラー。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitError {
    /// `SplitRatios` が区間外・NaN・合計が 1.0 から許容差を超えて外れている。
    InvalidRatios,
}

/// `n` 件の group を train / validation / test へ割り付ける件数を決める。
///
/// `docs/spec/03-poc/scratch-classifier/scripts/split_train.py`（PoC-9・PoC-10。REQ-17 根拠）の
/// `alloc_counts` と同じ割付規則を移植したもの。常に `train + validation + test == n` を保つ
/// （呼び出し側の設定ミスでも panic しない。coding-rust.md）。
fn alloc_counts(n: usize, ratios: &SplitRatios) -> (usize, usize, usize) {
    match n {
        0 => (0, 0, 0),
        1 => (1, 0, 0),
        2 => (1, 0, 1),
        _ => {
            #[allow(clippy::cast_precision_loss)]
            let n_f64 = n as f64;
            let n_train_raw = (n_f64 * ratios.train).floor();
            let n_val_raw = (n_f64 * ratios.validation).floor();
            // 負値・NaN は validate() で既に拒否済みだが、境界値でも usize への
            // 変換が壊れないよう 0 未満にはならないことを前提にしつつ明示的に扱う。
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            let n_train = n_train_raw.max(0.0) as usize;
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            let n_val = (n_val_raw.max(0.0) as usize).max(1);
            let n_val = n_val.min(n);
            let mut n_train = n_train.min(n.saturating_sub(n_val));
            let mut n_test = n.saturating_sub(n_train).saturating_sub(n_val);
            if n_test < 1 {
                n_train = n_train.saturating_sub(1);
                n_test = n.saturating_sub(n_train).saturating_sub(n_val);
            }
            (n_train, n_val, n_test)
        }
    }
}

/// group 内のレコードのラベル最頻値を group の代表ラベルとする。
///
/// 同数の場合はラベル名の辞書順で最小のものを採る
/// （`split_train.py` の `sorted(counts.items(), key=(-count, label))[0]` と同じ規則）。
/// `labels` が空の場合は空文字列を返す（呼び出し元は group が必ず 1 件以上の
/// レコードを持つ前提で呼ぶため、通常この分岐には到達しない）。
fn group_label<'a>(labels: impl Iterator<Item = &'a str>) -> String {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for label in labels {
        *counts.entry(label).or_insert(0) += 1;
    }
    let mut best: Option<(&str, usize)> = None;
    for (label, count) in counts {
        best = match best {
            None => Some((label, count)),
            Some((best_label, best_count)) => {
                if count > best_count || (count == best_count && label < best_label) {
                    Some((label, count))
                } else {
                    Some((best_label, best_count))
                }
            }
        };
    }
    best.map(|(label, _)| label.to_string()).unwrap_or_default()
}

/// SplitMix64（Vigna, 2015）。単純で高速な決定的 PRNG。
///
/// 暗号用途ではなく、分割の再現性（REQ-17）のためだけに使う
/// （評価契約に関わる決定性の実装方式であり、他のセキュリティ用途の
/// 乱数生成に流用してはならない）。
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// FNV-1a（自作）。
///
/// `std::collections::hash_map::DefaultHasher`（SipHash）は実装詳細が
/// 将来変わりうるため、分割の再現性を保つ本モジュールでは使わない。
fn fnv1a(bytes: &[u8]) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 0xCBF2_9CE4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;
    let mut hash = FNV_OFFSET_BASIS;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// ラベルごとに独立な PRNG シードを派生させる。
///
/// ラベル集合の追加・削除が他ラベルのシャッフル結果に影響しないようにする
/// （ラベルごとに独立したシャッフル系列を持たせるため）。
fn derive_label_seed(base_seed: u64, label: &str) -> u64 {
    let mixed = base_seed ^ fnv1a(label.as_bytes());
    SplitMix64::new(mixed).next_u64()
}

/// Fisher–Yates シャッフル（決定的）。
fn shuffle<T>(items: &mut [T], rng: &mut SplitMix64) {
    for i in (1..items.len()).rev() {
        let bound = i as u64 + 1;
        let j = (rng.next_u64() % bound) as usize;
        items.swap(i, j);
    }
}

/// group 単位で `records` を train / validation / test に決定的に分割する（REQ-17）。
///
/// 同一 `group_id` を持つレコードは必ず同じ [`Split`] に属する。同一 `seed`・
/// 同一入力・同一 `ratios` であれば、実行のたびに同じ [`SplitResult`] を返す
/// （受け入れ条件。TASK-17.1-1）。
///
/// `records` が空の場合はエラーにせず、空の [`SplitResult`] を返す。
///
/// # Errors
///
/// `ratios` が区間外・NaN・合計が 1.0 から許容差を超えて外れている場合、
/// [`SplitError::InvalidRatios`] を返す。
pub fn split_by_group<T: Groupable>(
    records: &[T],
    seed: u64,
    ratios: &SplitRatios,
) -> Result<SplitResult, SplitError> {
    ratios.validate()?;

    if records.is_empty() {
        return Ok(SplitResult {
            by_record: BTreeMap::new(),
            by_group: BTreeMap::new(),
            per_label: Vec::new(),
            rule_id: RULE_ID,
        });
    }

    // 1. group_id -> レコードのラベル一覧。
    let mut group_labels: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for record in records {
        group_labels
            .entry(record.group_id())
            .or_default()
            .push(record.label());
    }

    // 2. group_id -> 代表ラベル。
    let group_representative: BTreeMap<&str, String> = group_labels
        .iter()
        .map(|(group_id, labels)| (*group_id, group_label(labels.iter().copied())))
        .collect();

    // 3. ラベル -> group_id のソート済み一覧（BTreeMap のキー順で構築するため、
    //    group_id は常に昇順で並ぶ）。
    let mut label_groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (group_id, label) in &group_representative {
        label_groups
            .entry(label.clone())
            .or_default()
            .push((*group_id).to_string());
    }

    let mut by_group: BTreeMap<String, Split> = BTreeMap::new();
    let mut per_label: Vec<LabelAllocation> = Vec::new();

    // 4. ラベルを昇順（BTreeMap の反復順）に処理し、各ラベルごとに独立したシード・
    //    シャッフル系列で割り付ける（ラベルの処理順が実行のたびに変わらないことが決定性の前提）。
    for (label, mut groups) in label_groups {
        let label_seed = derive_label_seed(seed, &label);
        let mut rng = SplitMix64::new(label_seed);
        shuffle(&mut groups, &mut rng);

        let n = groups.len();
        let (n_train, n_val, n_test) = alloc_counts(n, ratios);

        for (index, group_id) in groups.iter().enumerate() {
            let split = if index < n_train {
                Split::Train
            } else if index < n_train + n_val {
                Split::Validation
            } else {
                Split::Test
            };
            by_group.insert(group_id.clone(), split);
        }

        per_label.push(LabelAllocation {
            label,
            n_groups: n,
            train: n_train,
            validation: n_val,
            test: n_test,
        });

        debug_assert_eq!(n_train + n_val + n_test, n);
    }

    // 5. レコード ID -> split（group の割付結果をレコードへ展開する）。
    let mut by_record: BTreeMap<String, Split> = BTreeMap::new();
    for record in records {
        if let Some(split) = by_group.get(record.group_id()) {
            by_record.insert(record.id().to_string(), *split);
        }
    }

    Ok(SplitResult {
        by_record,
        by_group,
        per_label,
        rule_id: RULE_ID,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// テスト用の最小レコード（`Groupable` の実装確認を兼ねる）。
    struct TestRecord {
        id: &'static str,
        group_id: &'static str,
        label: &'static str,
    }

    impl Groupable for TestRecord {
        fn id(&self) -> &str {
            self.id
        }
        fn group_id(&self) -> &str {
            self.group_id
        }
        fn label(&self) -> &str {
            self.label
        }
    }

    fn record(id: &'static str, group_id: &'static str, label: &'static str) -> TestRecord {
        TestRecord {
            id,
            group_id,
            label,
        }
    }

    /// REQ-17・TASK-17.1-1: alloc_counts の境界値（PoC-9/10 の規則の移植確認）。
    #[test]
    fn req17_task17_1_1_alloc_counts_boundary_values() {
        let ratios = SplitRatios::default();
        assert_eq!(alloc_counts(0, &ratios), (0, 0, 0));
        assert_eq!(alloc_counts(1, &ratios), (1, 0, 0));
        assert_eq!(alloc_counts(2, &ratios), (1, 0, 1));
        assert_eq!(alloc_counts(3, &ratios), (1, 1, 1));
        assert_eq!(alloc_counts(4, &ratios), (2, 1, 1));
        assert_eq!(alloc_counts(10, &ratios), (8, 1, 1));
    }

    /// REQ-17・TASK-17.1-1: alloc_counts は常に train + validation + test == n を保つ。
    #[test]
    fn req17_task17_1_1_alloc_counts_sum_invariant() {
        let ratios = SplitRatios::default();
        for n in 0..200 {
            let (train, validation, test) = alloc_counts(n, &ratios);
            assert_eq!(train + validation + test, n, "n={n} で合計が一致しない");
        }
    }

    /// REQ-17・TASK-17.1-1: group_label は最頻値を返し、同数はラベル名の辞書順最小を採る。
    #[test]
    fn req17_task17_1_1_group_label_majority_and_tie_break() {
        assert_eq!(group_label(["a", "a", "b"].into_iter()), "a");
        // 同数（b, a が各 1 件）の場合は辞書順で小さい "a" を採る。
        assert_eq!(group_label(["b", "a"].into_iter()), "a");
        assert_eq!(group_label(["z", "z", "a", "a"].into_iter()), "a");
    }

    /// REQ-17・TASK-17.1-1: 同一 group_id を持つ全レコードが同じ split に属する
    /// （group を跨がない、という受け入れ条件の直接検証）。
    #[test]
    fn req17_task17_1_1_records_in_same_group_share_split() {
        let mut records = Vec::new();
        for group_index in 0..20 {
            let label = if group_index % 2 == 0 {
                "intent_a"
            } else {
                "intent_b"
            };
            let group_id: &'static str = Box::leak(format!("group-{group_index}").into_boxed_str());
            let n_records_in_group = 2 + (group_index % 4);
            for record_index in 0..n_records_in_group {
                let id: &'static str = Box::leak(
                    format!("group-{group_index}-record-{record_index}").into_boxed_str(),
                );
                records.push(record(id, group_id, label));
            }
        }

        let result = split_by_group(&records, 42, &SplitRatios::default()).expect("valid ratios");

        for r in &records {
            let expected = result
                .by_group
                .get(r.group_id())
                .expect("group の割付が存在する");
            let actual = result
                .by_record
                .get(r.id())
                .expect("レコードの割付が存在する");
            assert_eq!(
                actual,
                expected,
                "group {} を跨いだ割付になっている",
                r.group_id()
            );
        }
    }

    /// REQ-17・TASK-17.1-1: 同一 seed・同一入力で 2 回呼んだ結果が完全一致する
    /// （決定性。受け入れ条件そのもの）。ピン留めした期待値と比較し、PRNG の
    /// 実装を変えると失敗するようにする。
    #[test]
    fn req17_task17_1_1_deterministic_given_same_seed() {
        let records = vec![
            record("r1", "g1", "a"),
            record("r2", "g1", "a"),
            record("r3", "g2", "a"),
            record("r4", "g3", "a"),
            record("r5", "g4", "a"),
            record("r6", "g5", "b"),
            record("r7", "g6", "b"),
            record("r8", "g7", "b"),
        ];

        let ratios = SplitRatios::default();
        let result_1 = split_by_group(&records, 7, &ratios).expect("valid ratios");
        let result_2 = split_by_group(&records, 7, &ratios).expect("valid ratios");
        assert_eq!(result_1, result_2, "同一 seed・同一入力で結果が一致しない");

        // ピン留めした期待値（本実装の PRNG・シャッフル・割付規則から算出した具体値）。
        // アルゴリズムを変更した場合はこの値を更新する。
        let mut expected_by_group: BTreeMap<String, Split> = BTreeMap::new();
        for (group_id, split) in [
            ("g1", Split::Test),
            ("g2", Split::Train),
            ("g3", Split::Validation),
            ("g4", Split::Train),
            ("g5", Split::Test),
            ("g6", Split::Train),
            ("g7", Split::Validation),
        ] {
            expected_by_group.insert(group_id.to_string(), split);
        }
        assert_eq!(
            result_1.by_group, expected_by_group,
            "ピン留めした期待値と一致しない（PRNG の実装が変わった可能性がある）"
        );
    }

    /// REQ-17・TASK-17.1-1: 異なる seed では少なくとも 1 group の割付が変わる。
    #[test]
    fn req17_task17_1_1_different_seed_can_change_allocation() {
        let mut records = Vec::new();
        for group_index in 0..12 {
            let group_id: &'static str = Box::leak(format!("group-{group_index}").into_boxed_str());
            let id: &'static str = Box::leak(format!("record-{group_index}").into_boxed_str());
            records.push(record(id, group_id, "a"));
        }

        let ratios = SplitRatios::default();
        let result_seed_1 = split_by_group(&records, 1, &ratios).expect("valid ratios");
        let result_seed_2 = split_by_group(&records, 2, &ratios).expect("valid ratios");
        assert_ne!(
            result_seed_1.by_group, result_seed_2.by_group,
            "異なる seed でも割付が変わらなかった"
        );
    }

    /// REQ-17・TASK-17.1-1: SplitRatios の異常値は Err(InvalidRatios) になり panic しない。
    #[test]
    fn req17_task17_1_1_invalid_ratios_rejected() {
        let records = vec![record("r1", "g1", "a")];

        let negative = SplitRatios {
            train: -0.1,
            validation: 0.1,
            test: 1.0,
        };
        assert_eq!(
            split_by_group(&records, 0, &negative),
            Err(SplitError::InvalidRatios)
        );

        let nan = SplitRatios {
            train: f64::NAN,
            validation: 0.1,
            test: 0.1,
        };
        assert_eq!(
            split_by_group(&records, 0, &nan),
            Err(SplitError::InvalidRatios)
        );

        let sum_too_large = SplitRatios {
            train: 0.8,
            validation: 0.5,
            test: 0.5,
        };
        assert_eq!(
            split_by_group(&records, 0, &sum_too_large),
            Err(SplitError::InvalidRatios)
        );
    }

    /// REQ-17・TASK-17.1-1: 空配列は panic せず空の SplitResult を返す。
    #[test]
    fn req17_task17_1_1_empty_records_yield_empty_result() {
        let records: Vec<TestRecord> = Vec::new();
        let result = split_by_group(&records, 0, &SplitRatios::default()).expect("valid ratios");
        assert!(result.by_record.is_empty());
        assert!(result.by_group.is_empty());
        assert!(result.per_label.is_empty());
    }
}
