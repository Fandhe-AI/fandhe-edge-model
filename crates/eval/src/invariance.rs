//! モデルパッケージ（重み・語彙・校正・しきい値）の評価前後ハッシュ比較。
//!
//! CLI の `evaluate` 工程（REQ-33）から、評価の実行前後でモデルパッケージの
//! 構成要素ごとに sha256 を取り、変化していないことを機械的に確かめるために
//! 呼ばれる想定（REQ-27「評価の独立性」正常系・TASK-27.1-1・issue #69。
//! PoC-9 `InvarianceTest.test_argument_recording_and_hash` の「モデルパッケージ側」を
//! 移植する）。
//!
//! # 呼び出し側との契約（重要）
//!
//! - **評価後のスナップショットは、評価前のバッファを使い回さず、ディスクから
//!   読み直したバイト列から作ること。** 同じメモリ上のバイト列を 2 回ハッシュしても
//!   常に一致してしまい、検査そのものが無意味になる
//! - ファイルの読み込み・サイズ上限の検証（REQ-39）・経路の閉じ込め・形式の
//!   許可リストは、本モジュールではなく呼び出し側（ガード層・CLI `evaluate` 工程）の
//!   責務。本モジュールはメモリ上のバイト列を受け取ってハッシュを取るだけで、
//!   ファイル I/O・逆シリアル化は一切行わない
//!
//! # 資源上限
//!
//! 計算量は入力バイト列の長さの合計に比例し、追加のアロケーションは構成要素
//! （[`ModelComponent`]）の数に比例する。構成要素ごとの入力サイズの
//! 上限検証は、読み込み前に呼び出し側のガード層（REQ-39）で行う。
//!
//! # 現状（実装済みを装わない）
//!
//! - 評価データのハッシュの前後比較・不一致時の停止（REQ-17・TASK-17.3）への
//!   接続は本モジュールの範囲外（issue #70）。ただし後続実装は本モジュールと
//!   同じ [`fandhe_edge_core::hash::Sha256Digest`] とスナップショット比較の形を
//!   再利用できる
//! - 推論関数へ `input` 以外を渡さないことの記録・検査（TASK-27.2。PoC-9
//!   `ArgumentRecordingPredictor` 相当）は未実装
//! - 凍結した最終 test への 1 回限り適用の強制（TASK-27.3）は未実装
//! - パッケージ全体を 1 つにまとめた合成ダイジェスト・配布パッケージの
//!   マニフェスト形式（REQ-30・TASK-30.x）は本モジュールでは作らない
//!   （配布パッケージ形式の契約を先取りしないため）。構成要素ごとの
//!   ダイジェストの集合（[`ModelPackageSnapshot`]）までを提供する

use fandhe_edge_core::hash::Sha256Digest;
use std::collections::BTreeMap;
use std::fmt;

/// モデルパッケージを構成する要素の種類（REQ-27 が挙げる 4 構成要素）。
///
/// `#[non_exhaustive]` にして、将来構成要素が増えても外部 crate の
/// `match` を壊さないようにする。`Ord` は決定的な並び順（列挙順）を
/// 保証するために derive する（`ComponentChange` の一覧・`components()` の
/// 走査順を安定させるため）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum ModelComponent {
    /// モデルの重み（ONNX グラフの本体等）。
    Weights,
    /// 語彙（バイト n-gram の辞書等）。C3（バイト CNN）など、語彙を
    /// 持たない候補では [`ModelPackageBytes`] 側で `None` にする。
    Vocab,
    /// 校正パラメータ（温度スケーリング等）。
    Calibration,
    /// しきい値（判定境界）。
    Thresholds,
}

impl ModelComponent {
    /// 列挙順を辿る状態遷移。`current` が `None` なら最初の構成要素
    /// （[`ModelComponent::Weights`]）を返し、最後の構成要素
    /// （[`ModelComponent::Thresholds`]）に達すると `None` を返して終了する。
    ///
    /// 以前は宣言順の配列 `ALL: [ModelComponent; 4]` を別途持っていたが、
    /// 配列はただのリテラルなので新しいバリアントを追加しても配列側の
    /// 更新をコンパイラが強制できず、`entries()` が新しい構成要素を黙って
    /// 取りこぼす（fail-open になる）欠陥があった（codex/review 指摘。
    /// PRRT_kwDOUq-SxM6mh13u）。この `match` は wildcard を持たない
    /// exhaustive match で、`Some(...)` 側の各アームが全バリアントを
    /// 一度ずつ列挙するため、新しいバリアントを追加すると
    /// 「そのバリアントへ遷移してくる前段の枝」と「そのバリアントから
    /// 次へ遷移する枝」の両方の追加が要求され、追加しない限りコンパイルが
    /// 通らない。
    ///
    /// **この保証にも限界がある**: 新しいバリアント `Foo` を追加した際、
    /// 既存の終端アーム（`Some(ModelComponent::Thresholds) => None`）を
    /// `Some(ModelComponent::Thresholds) => Some(ModelComponent::Foo)` へ
    /// 直さずに `Some(ModelComponent::Foo) => None` という新しい独立した
    /// 終端アームだけを追加すれば構文的にはコンパイルが通り、`Foo` は
    /// 列挙から孤立したまま（`all()` が辿らないまま）になりうる。この
    /// 残存リスクはコンパイルだけでは塞ぎきれないため、テスト側
    /// （`tests` モジュールの `req27_all_lists_each_component_exactly_once_in_declared_order`）
    /// で列挙結果を具体値（宣言済みの全バリアントの一覧）と突き合わせ、
    /// バリアント追加時にテストの期待値更新を要求することで実質的に検出する。
    const fn next(current: Option<ModelComponent>) -> Option<ModelComponent> {
        match current {
            None => Some(ModelComponent::Weights),
            Some(ModelComponent::Weights) => Some(ModelComponent::Vocab),
            Some(ModelComponent::Vocab) => Some(ModelComponent::Calibration),
            Some(ModelComponent::Calibration) => Some(ModelComponent::Thresholds),
            Some(ModelComponent::Thresholds) => None,
        }
    }

    /// 全構成要素を宣言順で決定的に列挙するイテレータ
    /// （[`ModelPackageBytes::entries`] が走査に使う）。要素数を
    /// 別途ハードコードせず [`Self::next`] の状態遷移をそのまま辿るため、
    /// 要素数と列挙内容が二重管理にならない。
    fn all() -> impl Iterator<Item = ModelComponent> {
        std::iter::successors(Self::next(None), |&current| Self::next(Some(current)))
    }

    /// JSON のキー等に使う英語の識別子（プログラム出力文字列は英語。
    /// `.claude/rules/japanese-style.md`）。
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            ModelComponent::Weights => "weights",
            ModelComponent::Vocab => "vocab",
            ModelComponent::Calibration => "calibration",
            ModelComponent::Thresholds => "thresholds",
        }
    }
}

impl fmt::Display for ModelComponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// モデルパッケージの構成要素を、それぞれのバイト列への参照で束ねた入力。
///
/// 「構成要素が存在しない（`None`）」と「構成要素が空バイト列
/// （`Some(&[])`）」を区別する（[`fandhe_edge_core::hash`] が評価データの
/// 凍結〔REQ-17〕向けに引いた線と同じ。空バイト列は `Sha256Digest::of_bytes(&[])`
/// で計算できる有効な入力として扱う）。C3（バイト CNN）のように語彙を
/// 持たないモデルでは `vocab` を `None` にする。
#[derive(Clone, Copy, Default)]
pub struct ModelPackageBytes<'a> {
    /// モデルの重み。
    pub weights: Option<&'a [u8]>,
    /// 語彙。無い候補では `None`。
    pub vocab: Option<&'a [u8]>,
    /// 校正パラメータ。無ければ `None`。
    pub calibration: Option<&'a [u8]>,
    /// しきい値。無ければ `None`。
    pub thresholds: Option<&'a [u8]>,
}

impl fmt::Debug for ModelPackageBytes<'_> {
    /// バイト長のみを出し、モデル・語彙・校正・しきい値の生の中身は出さない
    /// （`.claude/rules/security.md`「秘密情報の混入防止」。derive Debug は
    /// `Option<&[u8]>` をそのまま出力するため、`{:?}` によるデバッグ出力が
    /// 将来ログ等へ追加された場合の漏洩を避ける）。
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn len_or_none(value: Option<&[u8]>) -> Option<usize> {
            value.map(<[u8]>::len)
        }
        f.debug_struct("ModelPackageBytes")
            .field("weights_len", &len_or_none(self.weights))
            .field("vocab_len", &len_or_none(self.vocab))
            .field("calibration_len", &len_or_none(self.calibration))
            .field("thresholds_len", &len_or_none(self.thresholds))
            .finish()
    }
}

impl<'a> ModelPackageBytes<'a> {
    /// 指定した構成要素に対応するフィールドの値を返す。wildcard 無しの
    /// exhaustive match にしているため、[`ModelComponent`] へ将来バリアントを
    /// 追加した際にこの match の更新漏れがあればコンパイルエラーになり、
    /// `entries()`（延いては [`ModelPackageSnapshot::capture`]）が新しい
    /// 構成要素を黙って取りこぼす（fail-open になる）のを防ぐ。
    fn component_bytes(&self, component: ModelComponent) -> Option<&'a [u8]> {
        match component {
            ModelComponent::Weights => self.weights,
            ModelComponent::Vocab => self.vocab,
            ModelComponent::Calibration => self.calibration,
            ModelComponent::Thresholds => self.thresholds,
        }
    }

    /// 構成要素とその値を、[`ModelComponent`] の宣言順で決定的に列挙する。
    fn entries(&self) -> impl Iterator<Item = (ModelComponent, Option<&'a [u8]>)> + 'a {
        let this = *self;
        ModelComponent::all().map(move |component| (component, this.component_bytes(component)))
    }
}

/// [`ModelPackageSnapshot::capture`] が返しうるエラー。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum InvarianceError {
    /// 重み（[`ModelComponent::Weights`]）が `None`。重みは判定結果を
    /// 直接左右する構成要素であり、REQ-27（評価の独立性）が求める
    /// 「評価前後でモデルのハッシュが一致すること」は重みを含めて初めて
    /// 意味を持つ。重み以外の構成要素（語彙・校正・しきい値）だけが
    /// 揃っていても、重みの改変を検出できないスナップショットを
    /// 「検証済み」と偽らないよう fail-closed で拒否する
    /// （codex/review 指摘。PRRT_kwDOUq-SxM6mhunj）。
    MissingWeights,
}

impl fmt::Display for InvarianceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InvarianceError::MissingWeights => {
                write!(f, "model package is missing required component: weights")
            }
        }
    }
}

impl std::error::Error for InvarianceError {}

/// モデルパッケージの構成要素ごとの sha256 ダイジェストの集合
/// （評価前・評価後のそれぞれで 1 つずつ作り、[`ModelPackageSnapshot::verify_unchanged`]
/// で比較する）。
///
/// `BTreeMap` で持つのは、内部の `HashMap` の走査順がプロセスごとに変わりうる
/// （非決定的になる）のを避けるため（[`ModelComponent`] は `Ord` を derive
/// 済みなのでそのままキーに使える）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelPackageSnapshot {
    digests: BTreeMap<ModelComponent, Sha256Digest>,
}

impl ModelPackageSnapshot {
    /// バイト列の束からスナップショットを作る。重み
    /// （[`ModelComponent::Weights`]）が無ければ
    /// [`InvarianceError::MissingWeights`] を返す。
    ///
    /// 重みを必須にするのは、重み以外の構成要素（語彙・校正・しきい値）
    /// だけが揃った状態を「検証済みのモデルパッケージ」として受理すると、
    /// 評価前後で重みが書き換わっても [`Self::verify_unchanged`] が
    /// 検出できず REQ-27（評価の独立性）を満たせないため
    /// （codex/review 指摘。PRRT_kwDOUq-SxM6mhunj）。重みが `Some` なら
    /// 必ず 1 件以上ダイジェストが入るため、空のパッケージという状態は
    /// この必須化によって構造的に起こりえない。
    pub fn capture(bytes: &ModelPackageBytes<'_>) -> Result<Self, InvarianceError> {
        if bytes.weights.is_none() {
            return Err(InvarianceError::MissingWeights);
        }
        let mut digests = BTreeMap::new();
        for (component, value) in bytes.entries() {
            if let Some(value) = value {
                digests.insert(component, Sha256Digest::of_bytes(value));
            }
        }
        Ok(ModelPackageSnapshot { digests })
    }

    /// 指定した構成要素のダイジェスト（無ければ `None`）。
    #[must_use]
    pub fn digest(&self, component: ModelComponent) -> Option<&Sha256Digest> {
        self.digests.get(&component)
    }

    /// 含まれる構成要素を [`ModelComponent`] の宣言順で決定的に走査する。
    pub fn components(&self) -> impl Iterator<Item = (ModelComponent, &Sha256Digest)> {
        self.digests.iter().map(|(&c, d)| (c, d))
    }

    /// `self`（評価前）と `after`（評価後）を比較し、差分が無ければ `Ok(())`、
    /// 1 件でもあれば構成要素ごとの変化を列挙した [`InvarianceViolation`] を返す。
    ///
    /// 真偽値は返さない（`.claude/rules/coding-rust.md`「公開 API・型設計」）。
    /// `after` は呼び出し側の契約（モジュール冒頭の doc）に従い、評価前の
    /// バッファを使い回さず読み直したバイト列から作られている前提。
    pub fn verify_unchanged(
        &self,
        after: &ModelPackageSnapshot,
    ) -> Result<(), InvarianceViolation> {
        let mut changes = Vec::new();
        // 両方のスナップショットに含まれるキーの和集合を走査する（片方にしか
        // 無い構成要素も Added/Removed として検出するため）。ハードコードした
        // 4 バリアントの配列にしないのは、将来 `ModelComponent` へバリアントが
        // 増えたときに本ループが黙って新バリアントを見逃す（fail-open になる）
        // のを防ぐため。`ModelComponent` は `Ord` を derive 済みで `BTreeSet` は
        // 常にソート済みで走査されるため、宣言順のまま決定的になる。
        let all_components: std::collections::BTreeSet<ModelComponent> = self
            .digests
            .keys()
            .chain(after.digests.keys())
            .copied()
            .collect();
        for component in all_components {
            match (self.digest(component), after.digest(component)) {
                (Some(before), Some(after)) => {
                    if before != after {
                        changes.push(ComponentChange::Modified {
                            component,
                            before: *before,
                            after: *after,
                        });
                    }
                }
                (Some(before), None) => {
                    changes.push(ComponentChange::Removed {
                        component,
                        before: *before,
                    });
                }
                (None, Some(after)) => {
                    changes.push(ComponentChange::Added {
                        component,
                        after: *after,
                    });
                }
                (None, None) => {}
            }
        }
        if changes.is_empty() {
            Ok(())
        } else {
            Err(InvarianceViolation { changes })
        }
    }
}

/// 構成要素 1 つの評価前後での変化。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ComponentChange {
    /// 評価前後の両方に存在したが、ダイジェストが変わった。
    Modified {
        /// 変化した構成要素。
        component: ModelComponent,
        /// 評価前のダイジェスト。
        before: Sha256Digest,
        /// 評価後のダイジェスト。
        after: Sha256Digest,
    },
    /// 評価前は無く、評価後に現れた。
    Added {
        /// 追加された構成要素。
        component: ModelComponent,
        /// 評価後のダイジェスト。
        after: Sha256Digest,
    },
    /// 評価前にはあったが、評価後に無くなった。
    Removed {
        /// 削除された構成要素。
        component: ModelComponent,
        /// 評価前のダイジェスト。
        before: Sha256Digest,
    },
}

/// [`ModelPackageSnapshot::verify_unchanged`] が検出した不変性違反。
///
/// `changes` は [`ModelComponent`] の宣言順で決定的に並ぶ。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct InvarianceViolation {
    /// 変化した構成要素の一覧（[`ModelComponent`] の宣言順）。
    pub changes: Vec<ComponentChange>,
}

impl fmt::Display for InvarianceViolation {
    /// 構成要素名と sha256 の 16 進表現だけを出し、モデル・データの中身は
    /// 出さない（`.claude/rules/security.md`「秘密情報の混入防止」。
    /// 学習・評価データには機密情報が含まれうる前提）。
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "model package changed during evaluation: ")?;
        for (index, change) in self.changes.iter().enumerate() {
            if index > 0 {
                write!(f, "; ")?;
            }
            match change {
                ComponentChange::Modified {
                    component,
                    before,
                    after,
                } => {
                    write!(f, "{component} modified (before={before}, after={after})")?;
                }
                ComponentChange::Added { component, after } => {
                    write!(f, "{component} added (after={after})")?;
                }
                ComponentChange::Removed { component, before } => {
                    write!(f, "{component} removed (before={before})")?;
                }
            }
        }
        Ok(())
    }
}

impl std::error::Error for InvarianceViolation {}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_bytes() -> ModelPackageBytes<'static> {
        ModelPackageBytes {
            weights: Some(b"weights-v1"),
            vocab: Some(b"vocab-v1"),
            calibration: Some(b"calibration-v1"),
            thresholds: Some(b"thresholds-v1"),
        }
    }

    #[test]
    fn req27_all_lists_each_component_exactly_once_in_declared_order() {
        // `ModelComponent::all()` が宣言順に構成要素を 1 回ずつ列挙することを、
        // 件数のみの検査（旧 `ALL.len() == 4`）ではなく実際の列挙内容との
        // 具体値比較で確認する。件数だけの検査では、新しいバリアントの追加を
        // `next()` の遷移に組み込み忘れて孤立させても検出できない
        // （codex/review 指摘。PRRT_kwDOUq-SxM6mh13u）。列挙を高々 16 件で
        // 打ち切るのは、`next()` の実装ミスで循環し無限に列挙し続ける事態を
        // 避けるため（16 は現在の構成要素数 4 に対して十分な安全域）。
        let components: Vec<ModelComponent> = ModelComponent::all().take(16).collect();
        assert_eq!(
            components,
            vec![
                ModelComponent::Weights,
                ModelComponent::Vocab,
                ModelComponent::Calibration,
                ModelComponent::Thresholds,
            ]
        );
    }

    #[test]
    fn req27_all_yields_strictly_increasing_order() {
        // `ModelComponent::all()` が宣言順（`Ord` derive の順）に厳密単調
        // 増加することを確認する。`next()` が誤って前の構成要素へ戻る
        // （循環する）実装ミスがあれば、重複または逆順として検出できる。
        let components: Vec<ModelComponent> = ModelComponent::all().take(16).collect();
        for pair in components.windows(2) {
            assert!(pair[0] < pair[1], "宣言順が単調増加でない: {pair:?}");
        }
    }

    #[test]
    fn req27_capture_rejects_missing_weights_even_with_other_components() {
        // 重み以外の 3 構成要素が揃っていても、重みが無ければ拒否する
        // （codex/review 指摘。重み以外だけの「検証済み」を偽装させない）。
        let mut bytes = sample_bytes();
        bytes.weights = None;
        assert_eq!(
            ModelPackageSnapshot::capture(&bytes),
            Err(InvarianceError::MissingWeights)
        );
    }

    #[test]
    fn req27_capture_rejects_empty_package() {
        // 全構成要素が `None`（重みも含む）の場合も MissingWeights で拒否する。
        let bytes = ModelPackageBytes::default();
        assert_eq!(
            ModelPackageSnapshot::capture(&bytes),
            Err(InvarianceError::MissingWeights)
        );
    }

    #[test]
    fn req27_capture_same_bytes_is_unchanged() {
        let bytes = sample_bytes();
        let before = ModelPackageSnapshot::capture(&bytes).expect("失敗しないはず");
        let after = ModelPackageSnapshot::capture(&bytes).expect("失敗しないはず");
        assert_eq!(before.verify_unchanged(&after), Ok(()));
    }

    #[test]
    fn req27_modified_single_component_is_reported() {
        let before_bytes = sample_bytes();
        let before = ModelPackageSnapshot::capture(&before_bytes).expect("失敗しないはず");

        let mut after_bytes = before_bytes;
        after_bytes.weights = Some(b"weights-v2-modified");
        let after = ModelPackageSnapshot::capture(&after_bytes).expect("失敗しないはず");

        let violation = before.verify_unchanged(&after).unwrap_err();
        assert_eq!(violation.changes.len(), 1);
        match &violation.changes[0] {
            ComponentChange::Modified {
                component,
                before: before_digest,
                after: after_digest,
            } => {
                assert_eq!(*component, ModelComponent::Weights);
                // 独立に計算したゴールデン値（証拠の種別: 一次資料。
                // `python3 -c "import hashlib;
                // print(hashlib.sha256(b'weights-v1').hexdigest())"` で再確認できる）。
                // `Sha256Digest::of_bytes` を両辺で使うと `of_bytes` 自体の誤りを
                // 検出できないため、ここでは具体値と比較する
                // （`.claude/rules/coding-rust.md`「テスト」）。
                assert_eq!(
                    before_digest.to_hex(),
                    "c0ab742f68a24ef362b5529351eb13561e746b70a0f815efe7b64b570b477851"
                );
                assert_eq!(
                    after_digest.to_hex(),
                    "606e9eda915b7e2460355dc03565ec4b446367152dd7c5c61061a7702924fe63"
                );
            }
            other => panic!("Modified を期待したが {other:?} だった"),
        }
    }

    #[test]
    fn req27_added_and_removed_component_is_reported() {
        let mut before_bytes = sample_bytes();
        before_bytes.vocab = None;
        let before = ModelPackageSnapshot::capture(&before_bytes).expect("失敗しないはず");

        let mut after_bytes = sample_bytes();
        after_bytes.thresholds = None;
        let after = ModelPackageSnapshot::capture(&after_bytes).expect("失敗しないはず");

        let violation = before.verify_unchanged(&after).unwrap_err();
        assert_eq!(violation.changes.len(), 2);
        assert!(violation.changes.iter().any(|c| matches!(
            c,
            ComponentChange::Added {
                component: ModelComponent::Vocab,
                ..
            }
        )));
        assert!(violation.changes.iter().any(|c| matches!(
            c,
            ComponentChange::Removed {
                component: ModelComponent::Thresholds,
                ..
            }
        )));
    }

    #[test]
    fn req27_absent_differs_from_empty() {
        let mut before_bytes = sample_bytes();
        before_bytes.calibration = None;
        let before = ModelPackageSnapshot::capture(&before_bytes).expect("失敗しないはず");

        let mut after_bytes = sample_bytes();
        after_bytes.calibration = Some(&[]);
        let after = ModelPackageSnapshot::capture(&after_bytes).expect("失敗しないはず");

        let violation = before.verify_unchanged(&after).unwrap_err();
        assert_eq!(violation.changes.len(), 1);
        match &violation.changes[0] {
            ComponentChange::Added { component, after } => {
                assert_eq!(*component, ModelComponent::Calibration);
                assert_eq!(
                    after.to_hex(),
                    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
                );
            }
            other => panic!("Added を期待したが {other:?} だった"),
        }
    }

    #[test]
    fn req27_changes_are_ordered_deterministically() {
        let before_bytes = sample_bytes();
        let before = ModelPackageSnapshot::capture(&before_bytes).expect("失敗しないはず");

        let after_bytes = ModelPackageBytes {
            weights: Some(b"weights-v2"),
            vocab: Some(b"vocab-v2"),
            calibration: Some(b"calibration-v2"),
            thresholds: Some(b"thresholds-v2"),
        };
        let after = ModelPackageSnapshot::capture(&after_bytes).expect("失敗しないはず");

        let violation = before.verify_unchanged(&after).unwrap_err();
        let components: Vec<ModelComponent> = violation
            .changes
            .iter()
            .map(|c| match c {
                ComponentChange::Modified { component, .. }
                | ComponentChange::Added { component, .. }
                | ComponentChange::Removed { component, .. } => *component,
            })
            .collect();
        assert_eq!(
            components,
            vec![
                ModelComponent::Weights,
                ModelComponent::Vocab,
                ModelComponent::Calibration,
                ModelComponent::Thresholds,
            ]
        );
    }

    #[test]
    fn req27_violation_display_contains_only_component_and_hex() {
        let before_bytes = sample_bytes();
        let before = ModelPackageSnapshot::capture(&before_bytes).expect("失敗しないはず");

        let mut after_bytes = before_bytes;
        // 識別しやすい機密風の文字列（テスト用）。Display にこの中身が
        // 出ないことを確認する。
        after_bytes.thresholds = Some(b"SECRET-VALUE-should-not-leak");
        let after = ModelPackageSnapshot::capture(&after_bytes).expect("失敗しないはず");

        let violation = before.verify_unchanged(&after).unwrap_err();
        let displayed = violation.to_string();
        assert!(displayed.contains("thresholds"));
        assert!(!displayed.contains("SECRET-VALUE-should-not-leak"));
        // ダイジェストの 16 進表現は含まれる。
        let after_digest = Sha256Digest::of_bytes(b"SECRET-VALUE-should-not-leak");
        assert!(displayed.contains(&after_digest.to_hex()));
    }

    #[test]
    fn req27_model_package_bytes_debug_does_not_leak_raw_content() {
        // `ModelPackageBytes` の Debug 出力にモデル・語彙・校正・しきい値の
        // 生バイト列（機密情報を含みうる）が出ないことを確認する
        // （`.claude/rules/security.md`「秘密情報の混入防止」）。バイト長のみが
        // 出ることを確かめる。
        let bytes = ModelPackageBytes {
            weights: Some(b"SECRET-WEIGHTS-should-not-leak"),
            vocab: None,
            calibration: Some(b"SECRET-CALIBRATION"),
            thresholds: Some(b""),
        };
        let debugged = format!("{bytes:?}");
        assert!(!debugged.contains("SECRET-WEIGHTS-should-not-leak"));
        assert!(!debugged.contains("SECRET-CALIBRATION"));
        assert!(debugged.contains("weights_len: Some(30)"));
        assert!(debugged.contains("vocab_len: None"));
        assert!(debugged.contains("calibration_len: Some(18)"));
        assert!(debugged.contains("thresholds_len: Some(0)"));
    }
}
