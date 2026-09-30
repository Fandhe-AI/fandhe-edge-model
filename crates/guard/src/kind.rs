//! `kind`（モデルの種類）値の許可リスト判定（REQ-39「形式・種別の検査」・TASK-39.2-3・#155・PoC-20 ケース 2）。
//!
//! PoC-20（テストハーネス）で、ガード層の無い CLI は `kind` に入ったインジェクション文字列
//! （例: `c3; rm -rf ~`）を検証せずに通すことが実測されている。本モジュールは `kind` を
//! 「構文検査（`[a-z][a-z0-9_]{0,63}`）→ 許可リストとの完全一致」の順で判定し、合格した値だけを
//! [`CheckedKind`] として返す。合格後に下流へ渡るのは許可リスト側が持つ `&'static str` であり、
//! 入力由来の文字列は流れない。
//!
//! # 責務境界
//!
//! - 検査するのは `kind` だけ。`kind_version` の許可リストは [`crate::kind_version`]（TASK-39.6-1・#174）で扱う
//! - JSON 上で `kind` が欠落している・文字列でない場合は呼び出し側（CLI の `infer` 統合 TASK-39.2-4・#156。欠落は必須違反として拒否）の
//!   解析エラーとして扱い、本 API は `&str` を受け取る
//! - 既定の許可集合は学習ワーカーの選択口（TASK-19.1 の `_registry`）と同じ。ずれは
//!   `crates/train/tests/guard_kind_allowlist_sync.rs` が共有 fixture 経由で機械的に検出する
//! - 拒否メッセージには入力値を埋め込まない（ログインジェクション・データ転記の防止）
//! - 拒否は既存の終了コード `invalid_input`（64）に写す（REQ-21）。CLI への接続は #156 で `infer` の `artifact.json` へ統合済み

use fandhe_edge_core::exitcode::ExitCode;
use std::collections::BTreeSet;
use std::fmt;

/// `kind` の長さの上限（バイト）。構文検査より先に判定し、巨大な入力を文字走査なしで拒否する
/// （REQ-39 資源の上限）。
pub const MAX_KIND_LEN: usize = 64;

/// 既定で対応する `kind`。学習ワーカーの選択口 `_registry`（TASK-19.1）と一致させる。
const SUPPORTED_KINDS: [&str; 3] = ["c1", "c3", "autoregressive"];

/// 許可する `kind` の集合。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KindAllowlist {
    kinds: BTreeSet<&'static str>,
}

/// 許可リストの検査に合格した `kind` の証明。
///
/// 公開コンストラクタを持たず、[`KindAllowlist::check`] だけが作る。値は入力の借用ではなく
/// 許可リスト側の `&'static str`。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckedKind(&'static str);

impl CheckedKind {
    /// 許可リストに載っている `kind` 名。
    pub const fn as_str(&self) -> &'static str {
        self.0
    }
}

/// `kind` の拒否理由。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum KindRejection {
    /// 空文字列。
    Empty,
    /// 長さが [`MAX_KIND_LEN`] バイトを超える。
    TooLong {
        /// 入力のバイト長。
        len: usize,
        /// 上限。
        max: usize,
    },
    /// 構文違反（先頭が `[a-z]` でない、または `[a-z0-9_]` 以外を含む）。
    InvalidCharacter {
        /// 最初の不正バイトの位置。
        byte_offset: usize,
    },
    /// 構文は正しいが許可リストにない。
    NotAllowed {
        /// 許可されている `kind`（ソート済み）。
        allowed: Vec<&'static str>,
    },
}

impl KindRejection {
    /// 対応する終了コード。いずれも入力不正（`invalid_input`）。
    pub const fn exit_code(&self) -> ExitCode {
        ExitCode::InvalidInput
    }

    /// 機械可読な理由コード。`unsupported_kind` は学習ワーカーのエラーコードと揃える。
    pub const fn reason_code(&self) -> &'static str {
        match self {
            KindRejection::NotAllowed { .. } => "unsupported_kind",
            _ => "malformed_kind",
        }
    }
}

impl fmt::Display for KindRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KindRejection::Empty => write!(f, "kind must not be empty"),
            KindRejection::TooLong { len, max } => {
                write!(f, "kind is too long ({len} bytes, max {max})")
            }
            KindRejection::InvalidCharacter { byte_offset } => write!(
                f,
                "kind contains a disallowed character at byte {byte_offset} (allowed: [a-z0-9_], must start with [a-z])"
            ),
            KindRejection::NotAllowed { allowed } => {
                write!(
                    f,
                    "kind is not supported (supported: {})",
                    allowed.join(", ")
                )
            }
        }
    }
}

impl std::error::Error for KindRejection {}

/// 構文検査の 1 箇所実装。長さ → 空 → 文字の順に判定する。
pub(crate) fn validate_kind_syntax(kind: &str) -> Result<(), KindRejection> {
    if kind.len() > MAX_KIND_LEN {
        return Err(KindRejection::TooLong {
            len: kind.len(),
            max: MAX_KIND_LEN,
        });
    }
    if kind.is_empty() {
        return Err(KindRejection::Empty);
    }
    for (i, b) in kind.bytes().enumerate() {
        let ok = if i == 0 {
            b.is_ascii_lowercase()
        } else {
            b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'
        };
        if !ok {
            return Err(KindRejection::InvalidCharacter { byte_offset: i });
        }
    }
    Ok(())
}

impl KindAllowlist {
    /// 許可する `kind` を指定して作る。構文違反が 1 件でもあれば `Err`（綴り間違いを黙って通さない）。
    /// 空集合は `Ok` で、すべての入力を拒否する（fail-closed）。
    pub fn new(kinds: impl IntoIterator<Item = &'static str>) -> Result<Self, KindRejection> {
        let mut set = BTreeSet::new();
        for k in kinds {
            validate_kind_syntax(k)?;
            set.insert(k);
        }
        Ok(Self { kinds: set })
    }

    /// 既定の許可集合（`c1`・`c3`・`autoregressive`。学習ワーカーの選択口と同じ）。
    pub fn supported() -> Self {
        Self {
            kinds: SUPPORTED_KINDS.into_iter().collect(),
        }
    }

    /// 許可リストに `kind` が含まれるか（構文検査は行わない完全一致）。
    pub fn contains(&self, kind: &str) -> bool {
        self.kinds.contains(kind)
    }

    /// 許可されている `kind` をソート順で返す。
    pub fn iter(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.kinds.iter().copied()
    }

    /// `kind` を検査する。合格した場合だけ [`CheckedKind`] を返す。
    pub fn check(&self, kind: &str) -> Result<CheckedKind, KindRejection> {
        validate_kind_syntax(kind)?;
        match self.kinds.get(kind) {
            Some(k) => Ok(CheckedKind(k)),
            None => Err(KindRejection::NotAllowed {
                allowed: self.iter().collect(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-39: 構文検査の境界（先頭の数字・先頭の `_`・DEL・非 ASCII）。
    #[test]
    fn req39_syntax_boundaries() {
        assert_eq!(validate_kind_syntax("a"), Ok(()));
        assert_eq!(validate_kind_syntax("a_9"), Ok(()));
        assert_eq!(
            validate_kind_syntax("1c"),
            Err(KindRejection::InvalidCharacter { byte_offset: 0 })
        );
        assert_eq!(
            validate_kind_syntax("_c"),
            Err(KindRejection::InvalidCharacter { byte_offset: 0 })
        );
        assert_eq!(
            validate_kind_syntax("c\u{7f}"),
            Err(KindRejection::InvalidCharacter { byte_offset: 1 })
        );
        assert_eq!(
            validate_kind_syntax("é"),
            Err(KindRejection::InvalidCharacter { byte_offset: 0 })
        );
    }

    /// REQ-39: 既定集合が構文検査に通る。
    #[test]
    fn req39_supported_kinds_pass_syntax() {
        for k in SUPPORTED_KINDS {
            assert_eq!(validate_kind_syntax(k), Ok(()));
        }
        assert_eq!(
            KindAllowlist::new(SUPPORTED_KINDS),
            Ok(KindAllowlist::supported())
        );
    }
}
