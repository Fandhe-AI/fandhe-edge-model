//! `kind_version`（モデルの種類ごとの版）の許可リスト判定（REQ-39「版の検証」・TASK-39.6-1・#174・PoC-20 ケース 7）。
//!
//! PoC-20（模擬＋テストハーネス）で、ガード層の無い CLI は `kind_version: 99` のパッケージを
//! そのまま推論へ通すことが実測されている。本モジュールは `kind` ごとの許可版
//! （学習ワーカーの選択口 `_registry` と同じ形 `c1:{1}`・`c3:{1}`・`autoregressive:{1}`）と照合し、
//! 合格した組だけを [`CheckedKindVersion`] として返す。
//!
//! # 責務境界
//!
//! - 呼び出し元は CLI の `infer`（`artifact.json` を読んだ直後・モデルのバイト列を開く前）。
//!   検査の順序は型で強制する: [`crate::kind::KindAllowlist::check`] を通った
//!   [`CheckedKind`] がなければ [`KindVersionAllowlist::check`] を呼べない
//! - 検査するのは版の許可リストだけ。sha256 による完全性検証は `version_ledger`（TASK-39.3）、
//!   破損パッケージの確認は TASK-39.6-2（#175）の範囲
//! - `artifact.json` 上で `kind_version` が非整数・範囲外のときは（欠落は版 1 として扱う）、呼び出し側（core の
//!   `ArtifactOnnxRef::parse`）の解析エラーとして拒否する。本 API は `u32` を受け取る
//! - 既定の許可集合は学習ワーカーの選択口と同じ。ずれは
//!   `crates/train/tests/guard_kind_version_allowlist_sync.rs` が共有 fixture 経由で検出する
//! - 拒否メッセージには入力値（版番号）を埋め込まない。許可側の値のみを出す
//! - 拒否は終了コード `invalid_input`（64）に写す（REQ-21）

use crate::kind::{CheckedKind, KindRejection, validate_kind_syntax};
use fandhe_edge_core::exitcode::ExitCode;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// 既定で対応する `(kind, 許可版)`。学習ワーカーの選択口 `_registry`（TASK-19.1）と一致させる。
/// 契約値の集約先はここ 1 箇所。
const SUPPORTED_KIND_VERSIONS: [(&str, &[u32]); 3] =
    [("c1", &[1]), ("c3", &[1]), ("autoregressive", &[1])];

/// `kind` ごとに許可する `kind_version` の集合。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KindVersionAllowlist {
    versions: BTreeMap<&'static str, BTreeSet<u32>>,
}

/// 版の検査に合格した `(kind, kind_version)` の証明。
///
/// 公開コンストラクタを持たず、[`KindVersionAllowlist::check`] だけが作る。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckedKindVersion {
    kind: CheckedKind,
    version: u32,
}

impl CheckedKindVersion {
    /// 検査済みの `kind`。
    pub const fn kind(&self) -> CheckedKind {
        self.kind
    }

    /// 許可リストに載っていた版。
    pub const fn version(&self) -> u32 {
        self.version
    }
}

/// `kind_version` の拒否理由。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum KindVersionRejection {
    /// その `kind` の許可版に含まれない（`kind` 自体が版の許可リストに無い場合を含む。その場合 `allowed` は空）。
    NotAllowed {
        /// 検査した `kind`（許可リスト側の名前）。
        kind: &'static str,
        /// その `kind` で許可されている版（昇順）。
        allowed: Vec<u32>,
    },
}

impl KindVersionRejection {
    /// 対応する終了コード（入力不正）。
    pub const fn exit_code(&self) -> ExitCode {
        ExitCode::InvalidInput
    }

    /// 機械可読な理由コード。学習ワーカーの `FailureCode::UnsupportedKindVersion` と語彙を揃える。
    pub const fn reason_code(&self) -> &'static str {
        match self {
            KindVersionRejection::NotAllowed { .. } => "unsupported_kind_version",
        }
    }
}

impl fmt::Display for KindVersionRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KindVersionRejection::NotAllowed { kind, allowed } => {
                let list: Vec<String> = allowed.iter().map(u32::to_string).collect();
                write!(
                    f,
                    "kind_version is not supported for kind {kind} (supported: {})",
                    list.join(", ")
                )
            }
        }
    }
}

impl std::error::Error for KindVersionRejection {}

impl KindVersionAllowlist {
    /// 許可する `(kind, 版)` の組から作る。`kind` が構文違反なら `Err`（綴り間違いを黙って通さない）。
    /// 空は `Ok` で、すべての入力を拒否する（fail-closed）。
    pub fn new(
        entries: impl IntoIterator<Item = (&'static str, u32)>,
    ) -> Result<Self, KindRejection> {
        let mut versions: BTreeMap<&'static str, BTreeSet<u32>> = BTreeMap::new();
        for (kind, version) in entries {
            validate_kind_syntax(kind)?;
            versions.entry(kind).or_default().insert(version);
        }
        Ok(Self { versions })
    }

    /// 既定の許可集合（`c1:{1}`・`c3:{1}`・`autoregressive:{1}`。学習ワーカーの選択口と同じ）。
    pub fn supported() -> Self {
        let mut versions: BTreeMap<&'static str, BTreeSet<u32>> = BTreeMap::new();
        for (kind, vs) in SUPPORTED_KIND_VERSIONS {
            versions.entry(kind).or_default().extend(vs.iter().copied());
        }
        Self { versions }
    }

    /// `(kind, 版)` が許可リストに含まれるか（完全一致）。
    pub fn contains(&self, kind: &str, kind_version: u32) -> bool {
        self.versions
            .get(kind)
            .is_some_and(|s| s.contains(&kind_version))
    }

    /// 許可されている `(kind, 版)` をソート順で返す。
    pub fn iter(&self) -> impl Iterator<Item = (&'static str, u32)> + '_ {
        self.versions
            .iter()
            .flat_map(|(k, vs)| vs.iter().map(move |v| (*k, *v)))
    }

    /// 検査済みの `kind` に対する `kind_version` を検査する。合格した場合だけ [`CheckedKindVersion`] を返す。
    pub fn check(
        &self,
        kind: CheckedKind,
        kind_version: u32,
    ) -> Result<CheckedKindVersion, KindVersionRejection> {
        let name = kind.as_str();
        match self.versions.get(name) {
            Some(set) if set.contains(&kind_version) => Ok(CheckedKindVersion {
                kind,
                version: kind_version,
            }),
            other => Err(KindVersionRejection::NotAllowed {
                kind: name,
                allowed: other
                    .map(|s| s.iter().copied().collect())
                    .unwrap_or_default(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kind::KindAllowlist;

    /// REQ-39: 既定の版の許可リストの kind 集合は kind の許可リストと一致する（片方だけ増減させない）。
    #[test]
    fn req39_supported_kinds_match_kind_allowlist() {
        let kinds: BTreeSet<&str> = KindAllowlist::supported().iter().collect();
        let versioned: BTreeSet<&str> = KindVersionAllowlist::supported()
            .iter()
            .map(|(k, _)| k)
            .collect();
        assert_eq!(kinds, versioned);
    }
}
