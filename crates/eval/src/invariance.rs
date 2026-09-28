//! モデルパッケージ（重み・語彙・校正・しきい値）の評価前後ハッシュ比較。
//!
//! CLI の `evaluate` 工程（REQ-33）から、評価の実行前後でモデルパッケージの
//! 構成要素ごとに sha256 を取り、変化していないことを機械的に確かめるために
//! 呼ばれる想定（REQ-27「評価の独立性」正常系・TASK-27.1-1・issue #69。
//! PoC-9 `InvarianceTest.test_argument_recording_and_hash` の「モデルパッケージ側」を
//! 移植する）。
//!
//! # 2 段構えの API
//!
//! - [`ModelPackageBytes`] / [`ModelPackageSnapshot`]（メモリ上のバイト列専用）:
//!   **評価後のスナップショットは、評価前のバッファを使い回さず、呼び出し側が
//!   ディスクから読み直したバイト列から作ること。** 同じメモリ上のバイト列を
//!   2 回ハッシュしても常に一致してしまい、検査そのものが無意味になる。
//!   本経路はファイル I/O・逆シリアル化を一切行わず、経路の閉じ込め・形式の
//!   許可リストは呼び出し側（ガード層・CLI `evaluate` 工程）の責務のまま残す
//! - [`evaluate_with_invariance`]（パス指定・構造的に安全）: 「評価前後で
//!   同じバッファを使い回す」という上のアンチパターンを、呼び出し側の規律に
//!   頼らず型で防ぐための上位 API。評価前後の両方でパスから実際にディスクを
//!   読み直すのは本関数自身であり、呼び出し側は評価クロージャへ渡された
//!   [`ModelPackagePaths`] からさらに読み込む必要はない（TASK-27.1-1・issue #69・
//!   issue #214 の codex/review 指摘: 予測器が重みバイト列しか読まず評価が
//!   ラベルしか受け取らない経路では、評価処理自体がパッケージを書き換える
//!   回帰を検出できないという指摘への対応）
//!
//! # 資源上限
//!
//! [`ModelPackageSnapshot::capture`] の計算量は入力バイト列の長さの合計に
//! 比例し、追加のアロケーションは構成要素（[`ModelComponent`]）の数に比例する。
//! この経路の入力サイズ上限検証は、読み込み前に呼び出し側のガード層
//! （REQ-39）で行う。一方 [`evaluate_with_invariance`] は自らファイルを開くため、
//! 構成要素 1 件あたり [`MAX_MODEL_COMPONENT_BYTES`] を超える読み込みを行わない
//! （詳細は当該定数と [`evaluate_with_invariance`] のドキュメントを参照）。
//!
//! [`ModelPackagePaths`] が指すパスに FIFO・ソケット・ディレクトリ等の特殊
//! ファイルが渡されても無期限に停止しない通常ファイル判定・サイズ上限・
//! `O_NONBLOCK` を用いた TOCTOU 対策は、共通コアの [`fandhe_edge_core::fs`]
//! モジュールへ集約済み（issue #214 codex/review 指摘: 本モジュールが独自に
//! 複製していた防御ロジック〔旧 `read_component_bounded`・
//! `open_component_for_read`〕を撤去し、`fandhe-edge-core::definition::Definition::load`
//! と同じ関数を共有する）。本モジュールは
//! [`fandhe_edge_core::fs::sha256_file_bounded`] を呼び、評価前後それぞれで
//! 構成要素ファイルの sha256 を**ストリームで**計算する。読み込んだ生バイト列
//! 全体は保持せず、保持するのは [`fandhe_edge_core::hash::Sha256Digest`]
//! （32 バイト）だけにすることで、評価前後のスナップショットを同時に持っても
//! 構成要素の生バイト列（最大で構成要素数 × [`MAX_MODEL_COMPONENT_BYTES`]）を
//! 二重に保持しない（issue #214 codex/review 指摘: 4 構成要素が各 64 MiB の
//! 場合、旧実装では評価前後で最大 512 MiB が同時にメモリへ残りえた）。
//! 経路の閉じ込め（`../` 等の拒否）は引き続き呼び出し側のガード層
//! （REQ-39・パス未確定）の責務であり、本モジュールはサイズ上限と通常
//! ファイルであることの検査のみを扱う。
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

use fandhe_edge_core::fs::FsError;
use fandhe_edge_core::hash::Sha256Digest;
use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

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
        Self::from_digests(digests)
    }

    /// 構成要素ごとに既に計算済みのダイジェストの集合から直接スナップショットを
    /// 作る（[`evaluate_with_invariance`] がファイルをストリームで読み込みながら
    /// [`fandhe_edge_core::fs::sha256_file_bounded`] で計算したダイジェストを
    /// 渡す用途。生バイト列を経由しないため [`ModelPackageBytes`] を介さない）。
    ///
    /// [`Self::capture`] と同じ不変条件（重み〔[`ModelComponent::Weights`]〕が
    /// 無ければ [`InvarianceError::MissingWeights`]）を課す。両関数とも最終的に
    /// 本関数へ委譲することで、この不変条件の検査箇所を 1 つに保つ。
    fn from_digests(
        digests: BTreeMap<ModelComponent, Sha256Digest>,
    ) -> Result<Self, InvarianceError> {
        if !digests.contains_key(&ModelComponent::Weights) {
            return Err(InvarianceError::MissingWeights);
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

/// モデルパッケージの構成要素それぞれのファイルパス。
///
/// [`evaluate_with_invariance`] へ渡す入力で、評価クロージャにもそのまま
/// 渡される。重み（[`ModelComponent::Weights`]）は
/// [`ModelPackageSnapshot::capture`] と同じ理由（評価の独立性を確かめるには
/// 重みを含めて初めて意味を持つ）で必須のフィールドにし、「重みの無い
/// モデルパッケージ」という壊れた値を型で表現できないようにする
/// （`.claude/rules/coding-rust.md`「公開 API・型設計」）。語彙が無い候補
/// （C3〔バイト CNN〕等）では `vocab` を `None` にする。
#[derive(Debug, Clone, Copy)]
pub struct ModelPackagePaths<'a> {
    /// モデルの重みファイルのパス。
    pub weights: &'a Path,
    /// 語彙ファイルのパス。無い候補では `None`。
    pub vocab: Option<&'a Path>,
    /// 校正パラメータファイルのパス。無ければ `None`。
    pub calibration: Option<&'a Path>,
    /// しきい値ファイルのパス。無ければ `None`。
    pub thresholds: Option<&'a Path>,
}

/// [`evaluate_with_invariance`] が構成要素 1 件あたりに読み込むバイト数の上限
/// （暫定値。REQ-30 が目標とする配布パッケージ全体の容量目安 40MB に対し、
/// 構成要素単位の上限として十分な余裕を見込んだ 64MiB とする。REQ-39 の
/// 資源上限が正式に決まり次第見直す。`fandhe-edge-core::definition::MAX_DEFINITION_FILE_BYTES`
/// と同じ位置づけの暫定値）。
pub const MAX_MODEL_COMPONENT_BYTES: u64 = 64 * 1024 * 1024;

/// 構成要素 1 件を [`fandhe_edge_core::fs::sha256_file_bounded`] でストリーム
/// ハッシュ化し、失敗を構成要素・パスの情報を添えた [`EvaluationInvarianceError`]
/// へ写す（[`capture_snapshot`] が評価前後それぞれで構成要素ごとに呼ぶ内部
/// ヘルパー）。通常ファイル判定・サイズ上限・`O_NONBLOCK` による TOCTOU 対策は
/// すべて共通コア側の実装に委ね、本 crate 側では複製しない
/// （issue #214 codex/review 指摘）。
fn hash_component<E>(
    component: ModelComponent,
    path: &Path,
) -> Result<Sha256Digest, EvaluationInvarianceError<E>> {
    fandhe_edge_core::fs::sha256_file_bounded(path, MAX_MODEL_COMPONENT_BYTES).map_err(|err| {
        match err {
            FsError::Read { path, source } => EvaluationInvarianceError::Read {
                component,
                path,
                source,
            },
            FsError::TooLarge { path, size, limit } => EvaluationInvarianceError::TooLarge {
                component,
                path,
                size,
                limit,
            },
            FsError::NotRegularFile { path } => {
                EvaluationInvarianceError::NotRegularFile { component, path }
            }
            // `FsError` は `#[non_exhaustive]`（共通コア側で将来バリアントが
            // 増えうる）。未知のバリアントを fail-open で無視せず、内容を
            // 保った I/O エラーとして fail-closed に扱う（`path` は本関数の
            // 引数から補う。将来バリアントが `path` を持たない可能性がある
            // ため、元エラーの `Display` 文字列だけを `source` に残す）。
            other => EvaluationInvarianceError::Read {
                component,
                path: path.to_path_buf(),
                source: std::io::Error::other(other.to_string()),
            },
        }
    })
}

/// `paths` が指す各構成要素ファイルを [`MAX_MODEL_COMPONENT_BYTES`] の上限付きで
/// ディスクから読み込み、構成要素ごとの sha256 ダイジェストのみを保持する
/// スナップショットを作る（[`evaluate_with_invariance`] が評価前後それぞれで
/// 呼ぶ内部ヘルパー）。
///
/// 読み込んだ生バイト列は構成要素 1 件ずつ処理が終わるたびに破棄され、
/// 保持されるのは [`Sha256Digest`]（32 バイト）だけになる。評価前・評価後の
/// 両方のスナップショットを同時に持っても、旧実装（`CapturedComponentBytes`
/// にバイト列を保持し続けていた実装）のように構成要素の生バイト列を二重に
/// 持たない（issue #214 codex/review 指摘）。
fn capture_snapshot<E>(
    paths: &ModelPackagePaths<'_>,
) -> Result<ModelPackageSnapshot, EvaluationInvarianceError<E>> {
    let mut digests = BTreeMap::new();
    digests.insert(
        ModelComponent::Weights,
        hash_component(ModelComponent::Weights, paths.weights)?,
    );
    if let Some(path) = paths.vocab {
        digests.insert(
            ModelComponent::Vocab,
            hash_component(ModelComponent::Vocab, path)?,
        );
    }
    if let Some(path) = paths.calibration {
        digests.insert(
            ModelComponent::Calibration,
            hash_component(ModelComponent::Calibration, path)?,
        );
    }
    if let Some(path) = paths.thresholds {
        digests.insert(
            ModelComponent::Thresholds,
            hash_component(ModelComponent::Thresholds, path)?,
        );
    }
    ModelPackageSnapshot::from_digests(digests).map_err(EvaluationInvarianceError::Snapshot)
}

/// [`evaluate_with_invariance`] が返しうるエラー。評価クロージャ自身の
/// エラー型 `E` を包んで一緒に返せるようにする。
///
/// `#[non_exhaustive]` にして、将来バリアントが増えても外部 crate の
/// `match` を壊さないようにする（[`InvarianceError`] と同じ方針）。
#[derive(Debug)]
#[non_exhaustive]
pub enum EvaluationInvarianceError<E> {
    /// 構成要素ファイルの読み込み（評価前・評価後どちらか一方）に失敗した。
    Read {
        /// 読み込みに失敗した構成要素。
        component: ModelComponent,
        /// 読み込みに失敗したファイルのパス。
        path: PathBuf,
        /// 元の I/O エラー。
        source: std::io::Error,
    },
    /// 構成要素ファイルが [`MAX_MODEL_COMPONENT_BYTES`] を超えていた。
    TooLarge {
        /// 上限を超えていた構成要素。
        component: ModelComponent,
        /// 上限を超えていたファイルのパス。
        path: PathBuf,
        /// 報告または実際に読んだバイト数。
        size: u64,
        /// 上限バイト数。
        limit: u64,
    },
    /// 構成要素のパス先が通常ファイルではなかった（FIFO・ソケット・
    /// キャラクタデバイス・ディレクトリ等。REQ-39・issue #214 codex/review
    /// 指摘）。
    NotRegularFile {
        /// 通常ファイルではなかった構成要素。
        component: ModelComponent,
        /// 通常ファイルではなかったパス。
        path: PathBuf,
    },
    /// スナップショットの作成に失敗した（[`ModelPackageSnapshot::capture`]
    /// が返す [`InvarianceError`]。現状は重み欠如のみ）。
    Snapshot(InvarianceError),
    /// 評価の前後でモデルパッケージが変化した（REQ-27 の不変条件違反）。
    /// fail-closed のため、この場合は評価クロージャの結果（`Ok` であっても）
    /// を呼び出し側へ返さない。
    Changed(InvarianceViolation),
    /// 評価クロージャ自体がエラーを返した。モデルパッケージに変化が無かった
    /// ことは確認済みで、評価そのものの失敗であることを示す。
    Evaluation(E),
}

impl<E: fmt::Display> fmt::Display for EvaluationInvarianceError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvaluationInvarianceError::Read {
                component,
                path,
                source,
            } => write!(
                f,
                "failed to read model package component {component} at {}: {source}",
                path.display()
            ),
            EvaluationInvarianceError::TooLarge {
                component,
                path,
                size,
                limit,
            } => write!(
                f,
                "model package component {component} at {} exceeds size limit ({size} > {limit} bytes)",
                path.display()
            ),
            EvaluationInvarianceError::NotRegularFile { component, path } => write!(
                f,
                "model package component {component} at {} is not a regular file",
                path.display()
            ),
            EvaluationInvarianceError::Snapshot(err) => write!(f, "{err}"),
            EvaluationInvarianceError::Changed(violation) => write!(f, "{violation}"),
            EvaluationInvarianceError::Evaluation(err) => write!(f, "evaluation failed: {err}"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for EvaluationInvarianceError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            EvaluationInvarianceError::Read { source, .. } => Some(source),
            EvaluationInvarianceError::Snapshot(err) => Some(err),
            EvaluationInvarianceError::Changed(err) => Some(err),
            EvaluationInvarianceError::Evaluation(err) => Some(err),
            EvaluationInvarianceError::TooLarge { .. } => None,
            EvaluationInvarianceError::NotRegularFile { .. } => None,
        }
    }
}

/// 評価経路そのものを前後のパッケージハッシュ比較で包む（REQ-27「評価の
/// 独立性」・TASK-27.1-1・issue #69）。
///
/// CLI の `evaluate` 工程（将来。REQ-33・issue #140 で配線予定）が、実際の
/// 推論・集計処理を `eval` クロージャとして本関数へ渡す想定。呼び出しの流れ:
///
/// 1. `paths` からディスク上の構成要素を読み込み、評価前スナップショットを作る
/// 2. `eval(&paths)` を実行する（評価クロージャは `paths` から必要な構成要素を
///    読み込んでよい。同じパスを指すファイルを読み直すだけで、書き込みは
///    しない前提）
/// 3. `eval` の成否に関わらず、**必ず** `paths` を改めてディスクから読み直して
///    評価後スナップショットを作る（fail-closed。評価が失敗した場合でも
///    パッケージが改変されていないかは確認する）
/// 4. 前後のスナップショットを比較する。変化があれば
///    [`EvaluationInvarianceError::Changed`] を返し、`eval` の結果（`Ok` で
///    あっても）は呼び出し側へ渡さない
/// 5. 変化が無ければ、`eval` の結果をそのまま返す（`Err` だった場合は
///    [`EvaluationInvarianceError::Evaluation`] として包む）
///
/// 構成要素 1 件あたりの読み込み上限は [`MAX_MODEL_COMPONENT_BYTES`]。経路の
/// 閉じ込め（`../` 等の拒否）・形式の許可リストは呼び出し側のガード層
/// （REQ-39・パス未確定）の責務で、本関数は行わない。
pub fn evaluate_with_invariance<T, E>(
    paths: &ModelPackagePaths<'_>,
    eval: impl FnOnce(&ModelPackagePaths<'_>) -> Result<T, E>,
) -> Result<T, EvaluationInvarianceError<E>> {
    let before = capture_snapshot(paths)?;

    let eval_result = eval(paths);

    // `eval_result` が `Err` でも、必ずディスクを読み直して評価後の状態を
    // 確認する（fail-closed。評価失敗時にパッケージ改変の検査を省略しない）。
    let after = capture_snapshot(paths)?;

    before
        .verify_unchanged(&after)
        .map_err(EvaluationInvarianceError::Changed)?;

    eval_result.map_err(EvaluationInvarianceError::Evaluation)
}

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

    /// テスト用の一時ファイルを、成否に関わらず削除するガード（RAII）。
    struct TempFileGuard(std::path::PathBuf);

    impl Drop for TempFileGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    /// 排他的に一意な一時ファイルパスを作り、`bytes` を書き込む。
    fn write_unique_temp_file(label: &str, bytes: &[u8]) -> TempFileGuard {
        let pid = std::process::id();
        for attempt in 0..1000u32 {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let candidate = std::env::temp_dir().join(format!(
                "fandhe-edge-eval-invariance-unit-{pid}-{label}-{attempt}-{nanos}"
            ));
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&candidate)
            {
                Ok(mut file) => {
                    use std::io::Write as _;
                    file.write_all(bytes).expect("書き込みに失敗しないはず");
                    return TempFileGuard(candidate);
                }
                Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(err) => panic!("一時ファイルの作成に失敗しないはず: {err}"),
            }
        }
        panic!("一意な一時ファイルを作成できなかった");
    }

    // 通常ファイル判定・FIFO での無期限停止回避・TOCTOU 対策そのものの単体
    // テストは、防御ロジックの実体である `fandhe_edge_core::fs` 側
    // （`crates/core/src/fs.rs` の `tests` モジュール）に集約済み
    // （issue #214 codex/review 指摘。本 crate は旧 `read_component_bounded`・
    // `ComponentReadError` を複製せず、`fandhe_edge_core::fs::sha256_file_bounded`
    // をそのまま呼ぶだけになったため）。ここでは本 crate 独自の関心事
    // （構成要素種別への写像・ストリーム計算したダイジェストが
    // `Sha256Digest::of_bytes` と一致すること）だけを確認する。

    #[test]
    fn req27_hash_component_matches_of_bytes_for_known_content() {
        // ストリームで計算したダイジェスト（`hash_component` 経由）が、
        // メモリ上のバイト列から計算した `Sha256Digest::of_bytes` と一致する
        // ことを確認する（issue #214 codex/review 指摘。ストリーム計算経路と
        // メモリ上バイト列の経路の食い違いを検出する）。
        let guard = write_unique_temp_file("hash-component-weights", b"weights-v1");
        let digest = hash_component::<StubEvalError>(ModelComponent::Weights, &guard.0)
            .expect("上限内は成功するはず");
        assert_eq!(digest, Sha256Digest::of_bytes(b"weights-v1"));
    }

    #[test]
    fn req27_hash_component_maps_missing_file_to_read_error_with_component() {
        // `fandhe_edge_core::fs::FsError::Read` が構成要素の情報を添えて
        // `EvaluationInvarianceError::Read` へ写ることを確認する（実際の
        // I/O エラーの中身は core 側でテスト済みのため、ここでは写像先の
        // `component` フィールドのみを確認する）。
        let missing = std::env::temp_dir().join("fandhe-edge-eval-invariance-does-not-exist");
        let err = hash_component::<StubEvalError>(ModelComponent::Vocab, &missing).unwrap_err();
        match err {
            EvaluationInvarianceError::Read { component, .. } => {
                assert_eq!(component, ModelComponent::Vocab);
            }
            other => panic!("Read(Vocab) を期待したが {other:?} だった"),
        }
    }

    /// REQ-39: 通常ファイル判定そのものは `fandhe_edge_core::fs` 側で単体
    /// テスト済み（`crates/core/src/fs.rs`）。ここでは `hash_component` が
    /// `FsError::NotRegularFile` を構成要素の情報を添えて
    /// `EvaluationInvarianceError::NotRegularFile` へ正しく写すことのみを
    /// 確認する（写像漏れの検出。issue #214 codex/review 指摘）。
    #[test]
    fn req39_hash_component_maps_directory_to_not_regular_file_with_component() {
        let dir = std::env::temp_dir().join(format!(
            "fandhe-edge-eval-invariance-unit-{}-hash-component-dir-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir(&dir).expect("テスト用ディレクトリを作成できるはず");

        let result = hash_component::<StubEvalError>(ModelComponent::Calibration, &dir);
        std::fs::remove_dir(&dir).expect("テスト用ディレクトリを削除できるはず");

        match result.expect_err("ディレクトリは通常ファイルではないため拒否されるはず")
        {
            EvaluationInvarianceError::NotRegularFile { component, path } => {
                assert_eq!(component, ModelComponent::Calibration);
                assert_eq!(path, dir);
            }
            other => panic!("NotRegularFile(Calibration) を期待したが {other:?} だった"),
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    struct StubEvalError(&'static str);

    impl fmt::Display for StubEvalError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "stub evaluation error: {}", self.0)
        }
    }
    impl std::error::Error for StubEvalError {}

    /// [`evaluate_with_invariance`] のテスト専用に 4 構成要素のファイルを
    /// 用意する（`weights` は必須、他は指定した内容で作る）。
    struct EvaluationFixture {
        _weights_guard: TempFileGuard,
        _vocab_guard: TempFileGuard,
        _calibration_guard: TempFileGuard,
        _thresholds_guard: TempFileGuard,
        weights_path: std::path::PathBuf,
        vocab_path: std::path::PathBuf,
        calibration_path: std::path::PathBuf,
        thresholds_path: std::path::PathBuf,
    }

    impl EvaluationFixture {
        fn new(label: &str) -> Self {
            let weights_guard = write_unique_temp_file(&format!("{label}-weights"), b"weights-v1");
            let vocab_guard = write_unique_temp_file(&format!("{label}-vocab"), b"vocab-v1");
            let calibration_guard =
                write_unique_temp_file(&format!("{label}-calibration"), b"calibration-v1");
            let thresholds_guard =
                write_unique_temp_file(&format!("{label}-thresholds"), b"thresholds-v1");
            let weights_path = weights_guard.0.clone();
            let vocab_path = vocab_guard.0.clone();
            let calibration_path = calibration_guard.0.clone();
            let thresholds_path = thresholds_guard.0.clone();
            EvaluationFixture {
                _weights_guard: weights_guard,
                _vocab_guard: vocab_guard,
                _calibration_guard: calibration_guard,
                _thresholds_guard: thresholds_guard,
                weights_path,
                vocab_path,
                calibration_path,
                thresholds_path,
            }
        }

        fn paths(&self) -> ModelPackagePaths<'_> {
            ModelPackagePaths {
                weights: &self.weights_path,
                vocab: Some(&self.vocab_path),
                calibration: Some(&self.calibration_path),
                thresholds: Some(&self.thresholds_path),
            }
        }
    }

    #[test]
    fn req27_evaluate_with_invariance_returns_ok_when_package_unchanged() {
        // REQ-27 正常系: 評価クロージャがパッケージへ触れても書き換えなければ、
        // 評価結果がそのまま返る。
        let fixture = EvaluationFixture::new("unchanged");
        let paths = fixture.paths();

        let result: Result<usize, EvaluationInvarianceError<StubEvalError>> =
            evaluate_with_invariance(&paths, |p| {
                let weights = std::fs::read(p.weights).expect("読み込みに失敗しないはず");
                Ok(weights.len())
            });

        assert_eq!(
            result.expect("変化が無ければ Ok のはず"),
            b"weights-v1".len()
        );
    }

    #[test]
    fn req27_evaluate_with_invariance_detects_weights_tampered_during_evaluation() {
        // REQ-27 異常系: 評価クロージャの中でパッケージのファイルを実際に
        // 書き換えると、API 自身が前後比較で検出し、評価結果（Ok であっても）
        // を返さないことを確かめる（issue #214 codex/review 指摘への対応。
        // 「テストが評価後に上書きする」旧テストと異なり、改変は評価クロージャの
        // 内側で起きる）。
        let fixture = EvaluationFixture::new("tamper-weights");
        let paths = fixture.paths();
        let weights_path = fixture.weights_path.clone();

        let result: Result<(), EvaluationInvarianceError<StubEvalError>> =
            evaluate_with_invariance(&paths, |_p| {
                std::fs::write(&weights_path, b"tampered-weights-v2")
                    .expect("書き込みに失敗しないはず");
                Ok(())
            });

        match result {
            Err(EvaluationInvarianceError::Changed(violation)) => {
                assert_eq!(violation.changes.len(), 1);
                match &violation.changes[0] {
                    ComponentChange::Modified {
                        component,
                        before,
                        after,
                    } => {
                        assert_eq!(*component, ModelComponent::Weights);
                        assert_eq!(*before, Sha256Digest::of_bytes(b"weights-v1"));
                        assert_eq!(*after, Sha256Digest::of_bytes(b"tampered-weights-v2"));
                    }
                    other => panic!("Modified(Weights) を期待したが {other:?} だった"),
                }
            }
            other => panic!("Changed を期待したが {other:?} だった"),
        }
    }

    #[test]
    fn req27_evaluate_with_invariance_detects_vocab_tampered_during_evaluation() {
        let fixture = EvaluationFixture::new("tamper-vocab");
        let paths = fixture.paths();
        let vocab_path = fixture.vocab_path.clone();

        let result: Result<(), EvaluationInvarianceError<StubEvalError>> =
            evaluate_with_invariance(&paths, |_p| {
                std::fs::write(&vocab_path, b"tampered-vocab-v2")
                    .expect("書き込みに失敗しないはず");
                Ok(())
            });

        match result {
            Err(EvaluationInvarianceError::Changed(violation)) => {
                assert_eq!(violation.changes.len(), 1);
                assert!(matches!(
                    &violation.changes[0],
                    ComponentChange::Modified {
                        component: ModelComponent::Vocab,
                        ..
                    }
                ));
            }
            other => panic!("Changed を期待したが {other:?} だった"),
        }
    }

    #[test]
    fn req27_evaluate_with_invariance_detects_calibration_tampered_during_evaluation() {
        let fixture = EvaluationFixture::new("tamper-calibration");
        let paths = fixture.paths();
        let calibration_path = fixture.calibration_path.clone();

        let result: Result<(), EvaluationInvarianceError<StubEvalError>> =
            evaluate_with_invariance(&paths, |_p| {
                std::fs::write(&calibration_path, b"tampered-calibration-v2")
                    .expect("書き込みに失敗しないはず");
                Ok(())
            });

        match result {
            Err(EvaluationInvarianceError::Changed(violation)) => {
                assert_eq!(violation.changes.len(), 1);
                assert!(matches!(
                    &violation.changes[0],
                    ComponentChange::Modified {
                        component: ModelComponent::Calibration,
                        ..
                    }
                ));
            }
            other => panic!("Changed を期待したが {other:?} だった"),
        }
    }

    #[test]
    fn req27_evaluate_with_invariance_detects_thresholds_tampered_during_evaluation() {
        let fixture = EvaluationFixture::new("tamper-thresholds");
        let paths = fixture.paths();
        let thresholds_path = fixture.thresholds_path.clone();

        let result: Result<(), EvaluationInvarianceError<StubEvalError>> =
            evaluate_with_invariance(&paths, |_p| {
                std::fs::write(&thresholds_path, b"tampered-thresholds-v2")
                    .expect("書き込みに失敗しないはず");
                Ok(())
            });

        match result {
            Err(EvaluationInvarianceError::Changed(violation)) => {
                assert_eq!(violation.changes.len(), 1);
                assert!(matches!(
                    &violation.changes[0],
                    ComponentChange::Modified {
                        component: ModelComponent::Thresholds,
                        ..
                    }
                ));
            }
            other => panic!("Changed を期待したが {other:?} だった"),
        }
    }

    #[test]
    fn req27_evaluate_with_invariance_surfaces_evaluation_error_when_unchanged() {
        // 評価クロージャが Err を返し、かつパッケージが変化していなければ、
        // Evaluation(E) として包んで返す。
        let fixture = EvaluationFixture::new("eval-err-unchanged");
        let paths = fixture.paths();

        let result: Result<(), EvaluationInvarianceError<StubEvalError>> =
            evaluate_with_invariance(&paths, |_p| Err(StubEvalError("boom")));

        match result {
            Err(EvaluationInvarianceError::Evaluation(StubEvalError(message))) => {
                assert_eq!(message, "boom");
            }
            other => panic!("Evaluation を期待したが {other:?} だった"),
        }
    }

    #[test]
    fn req27_evaluate_with_invariance_prefers_changed_over_evaluation_error() {
        // fail-closed の順序確認: 評価クロージャが Err を返した場合でも、
        // その間にパッケージが改変されていれば Changed を優先して返し、
        // 評価エラーで改変が覆い隠されないことを確かめる
        // （advisor 指摘: 評価失敗時こそ改変検査を省略しない）。
        let fixture = EvaluationFixture::new("eval-err-tampered");
        let paths = fixture.paths();
        let weights_path = fixture.weights_path.clone();

        let result: Result<(), EvaluationInvarianceError<StubEvalError>> =
            evaluate_with_invariance(&paths, |_p| {
                std::fs::write(&weights_path, b"tampered-weights-v2")
                    .expect("書き込みに失敗しないはず");
                Err(StubEvalError("boom"))
            });

        assert!(matches!(result, Err(EvaluationInvarianceError::Changed(_))));
    }

    #[test]
    fn req27_evaluate_with_invariance_accepts_missing_vocab_for_vocab_free_candidates() {
        // C3（バイト CNN）等、語彙を持たない候補では vocab: None を許す。
        let weights_guard = write_unique_temp_file("vocab-free-weights", b"weights-v1");
        let thresholds_guard = write_unique_temp_file("vocab-free-thresholds", b"thresholds-v1");
        let paths = ModelPackagePaths {
            weights: &weights_guard.0,
            vocab: None,
            calibration: None,
            thresholds: Some(&thresholds_guard.0),
        };

        let result: Result<(), EvaluationInvarianceError<StubEvalError>> =
            evaluate_with_invariance(&paths, |_p| Ok(()));

        assert!(result.is_ok());
    }

    #[test]
    fn req27_evaluate_with_invariance_reports_too_large_component() {
        // 構成要素が上限を超えている場合、評価クロージャを呼ぶ前に
        // TooLarge で拒否する（本テストは MAX_MODEL_COMPONENT_BYTES の実値では
        // 現実的な時間で再現できないため、`capture_snapshot` が使う上限
        // そのものではなく `fandhe_edge_core::fs::sha256_file_bounded` 側の
        // 単体テスト（`crates/core/src/fs.rs`
        // `req39_sha256_file_bounded_rejects_file_over_limit`）で上限超過の
        // 検出を確認する。ここでは読み込み失敗（存在しないパス）が Read
        // エラーとして伝播することを確認する）。
        let weights_guard = write_unique_temp_file("read-error-weights", b"weights-v1");
        let missing_vocab = std::env::temp_dir().join("fandhe-edge-eval-invariance-missing-vocab");
        let paths = ModelPackagePaths {
            weights: &weights_guard.0,
            vocab: Some(&missing_vocab),
            calibration: None,
            thresholds: None,
        };

        let result: Result<(), EvaluationInvarianceError<StubEvalError>> =
            evaluate_with_invariance(&paths, |_p| Ok(()));

        match result {
            Err(EvaluationInvarianceError::Read { component, .. }) => {
                assert_eq!(component, ModelComponent::Vocab);
            }
            other => panic!("Read(Vocab) を期待したが {other:?} だった"),
        }
    }

    /// REQ-39・REQ-27: 公開 API [`evaluate_with_invariance`] 自身が、
    /// `ModelPackagePaths` にディレクトリ（特殊ファイルの一種）が渡された
    /// 場合に [`EvaluationInvarianceError::NotRegularFile`] として拒否する
    /// ことを確認する（issue #214 codex/review 指摘: 共通コアの
    /// `fandhe_edge_core::fs` 側の単体テストだけでなく、`capture_snapshot` を
    /// 経由した公開 API レベルの伝播も確かめる）。
    #[test]
    fn req39_evaluate_with_invariance_rejects_directory_as_weights() {
        let dir = std::env::temp_dir().join(format!(
            "fandhe-edge-eval-invariance-unit-{}-weights-dir-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir(&dir).expect("テスト用ディレクトリを作成できるはず");

        let paths = ModelPackagePaths {
            weights: &dir,
            vocab: None,
            calibration: None,
            thresholds: None,
        };

        let result: Result<(), EvaluationInvarianceError<StubEvalError>> =
            evaluate_with_invariance(&paths, |_p| Ok(()));
        std::fs::remove_dir(&dir).expect("テスト用ディレクトリを削除できるはず");

        match result {
            Err(EvaluationInvarianceError::NotRegularFile { component, path }) => {
                assert_eq!(component, ModelComponent::Weights);
                assert_eq!(path, dir);
            }
            other => panic!("NotRegularFile(Weights) を期待したが {other:?} だった"),
        }
    }
}
