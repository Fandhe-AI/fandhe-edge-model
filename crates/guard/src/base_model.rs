//! ベースモデルのディレクトリ内ファイルを開く入口（REQ-41・REQ-39・TASK-41.2-2・#398）。
//!
//! 追加学習（REQ-41）で利用者がローカルパスで渡すベースモデルのディレクトリ
//! （`model.safetensors`・`tokenizer.json` 等）を、読む前にガード層で検査する。役割は
//! 「名前の字句検査 → 閉じ込め付き open → 読み込み前のサイズ確認 → 内容の形式照合」だけで、
//! 経路・サイズ・形式の判定は [`crate::package`]・[`crate::file_size`]・[`crate::format`] に委譲する。
//!
//! - ディレクトリの受付は [`crate::package::confine_package`] をそのまま使う。root は呼び出し側
//!   （CLI 配線時。TASK-41.5・#425）が決める。ディレクトリ自身に閉じ込めるなら
//!   `confine_package(dir, Path::new("."))`
//! - メンバーは保持 fd 起点の `O_NOFOLLOW` で開く。`..`・絶対パス・symlink（末尾も親も）は拒否し、
//!   パスを開き直さない（検証後の差し替え対策）。このため HF cache の `snapshots/<rev>/`
//!   （blobs への symlink）は拒否される。実ファイルとして置いたディレクトリを渡すこと
//! - 検査順序は 名前 → 経路 → サイズ（fstat。1 バイトも読まない）→ 形式。名前と内容が食い違う
//!   もの（pickle 偽装を含む）は `format_not_allowed` で拒否する
//! - 利用者由来のパス・ファイル名・内容はエラーの値（Debug・source を含む）にもメッセージにも含めない。
//!   経路・形式の拒否は終了コードと理由コードだけに畳む
//!
//! - サイズ上限は重み（safetensors）が 2 GiB（`split_files::MAX_SHARD_FILE_BYTES`。#655）、
//!   `tokenizer.json` が 1 GiB（`MAX_READ_FILE_BYTES`）
//!
//! ponytail: 形式判定のため上限（重み 2 GiB・他 1 GiB）までメモリへ読み切る（`check_open_file` の方式）。
//! RSS が問題になれば safetensors はヘッダーだけの検査へ切り替える（#435）。
//! ハードリンクで外部ファイルを指すメンバーは経路では検出できない（必要なら `st_nlink > 1` を拒否）。

use crate::file_size::{
    FileSizeRejection, MAX_READ_FILE_BYTES, check_file_size_unclamped, check_file_size_within,
};
use crate::format::{
    CheckedFile, FileFormat, FormatAllowlist, check_open_file, check_open_file_unclamped,
};
use crate::package::ConfinedPackage;
use crate::split_files::MAX_SHARD_FILE_BYTES;
use fandhe_edge_core::exitcode::ExitCode;
use std::fmt;
use std::path::Path;

/// ベースモデルのファイルを開く際の拒否理由。
#[derive(Debug)]
#[non_exhaustive]
pub enum BaseModelFileRejection {
    /// ファイル名が許可されていない（`*.safetensors`・`tokenizer.json` 以外）。ファイルには触れない。
    NameNotAllowed,
    /// 経路の閉じ込め・open の拒否（`../`・絶対パス・symlink・通常ファイル以外）。
    /// 元の [`PathRejection`] は利用者由来のパスを保持するため、終了コードと理由コードだけを残す。
    Path {
        exit_code: ExitCode,
        reason_code: &'static str,
    },
    /// 読み込み前のサイズ確認の拒否（上限超過等。パスを含まない）。
    Size(FileSizeRejection),
    /// 内容の形式の拒否（名前との不一致・pickle 偽装・許可リスト外・読み込み失敗）。
    /// 元の [`FormatRejection`] の I/O エラーはパスを保持しうるため、終了コードと理由コードだけを残す。
    Format {
        exit_code: ExitCode,
        reason_code: &'static str,
    },
}

impl BaseModelFileRejection {
    /// 終了コード（REQ-21）。名前不許可は `InvalidInput`、他は内包する拒否のもの。
    pub fn exit_code(&self) -> ExitCode {
        match self {
            Self::NameNotAllowed => ExitCode::InvalidInput,
            Self::Path { exit_code, .. } | Self::Format { exit_code, .. } => *exit_code,
            Self::Size(e) => e.exit_code(),
        }
    }

    /// 機械可読な理由コード（英語の snake_case）。
    pub fn reason_code(&self) -> &'static str {
        match self {
            Self::NameNotAllowed => "base_model_file_name_not_allowed",
            Self::Path { reason_code, .. } | Self::Format { reason_code, .. } => reason_code,
            Self::Size(e) => e.reason_code(),
        }
    }
}

impl fmt::Display for BaseModelFileRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NameNotAllowed => write!(f, "base model file name is not allowed"),
            Self::Path { .. } => write!(f, "base model file path was rejected"),
            Self::Size(e) => e.fmt(f),
            Self::Format { .. } => write!(f, "base model file format was rejected"),
        }
    }
}

impl std::error::Error for BaseModelFileRejection {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Size(e) => Some(e),
            _ => None,
        }
    }
}

/// ファイル名から期待する内容の形式を決める。大文字小文字を区別し、ドットファイル・拡張子なしは除外する。
fn expected_format(member: &Path) -> Option<FileFormat> {
    if member.file_name().is_some_and(|n| n == "tokenizer.json") {
        return Some(FileFormat::TokenizerJson);
    }
    if member.extension().is_some_and(|e| e == "safetensors") {
        return Some(FileFormat::Safetensors);
    }
    None
}

/// 閉じ込め済みのベースモデルのディレクトリから `member` を開き、名前と内容が一致するものだけを返す。
///
/// 学習ワーカー起動前の `train`（追加学習。REQ-41）から呼ばれる想定。
///
/// # Errors
/// [`BaseModelFileRejection`]。検査順序はモジュール doc を参照。
pub fn open_base_model_file(
    dir: &ConfinedPackage,
    member: &Path,
) -> Result<CheckedFile, BaseModelFileRejection> {
    let expected = expected_format(member).ok_or(BaseModelFileRejection::NameNotAllowed)?;
    let file = dir
        .open_regular_member(member)
        .map_err(|e| BaseModelFileRejection::Path {
            exit_code: e.exit_code(),
            reason_code: e.reason_code(),
        })?;
    let allow = FormatAllowlist::new([expected]);
    // 重み（safetensors）だけ重みファイル専用の 2 GiB。7B の埋め込み 1 テンソルが 1 GiB 超で
    // 分割できないため（REQ-39・REQ-41・#654・#655）。他のファイルは 1 GiB のまま
    let checked = if expected == FileFormat::Safetensors {
        check_file_size_unclamped(&file, MAX_SHARD_FILE_BYTES)
            .map_err(BaseModelFileRejection::Size)?;
        check_open_file_unclamped(file, member, &allow, MAX_SHARD_FILE_BYTES)
    } else {
        check_file_size_within(&file, MAX_READ_FILE_BYTES).map_err(BaseModelFileRejection::Size)?;
        check_open_file(file, member, &allow, MAX_READ_FILE_BYTES)
    };
    checked.map_err(|e| BaseModelFileRejection::Format {
        exit_code: e.exit_code(),
        reason_code: e.reason_code(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-41・REQ-39・TASK-41.2-2: 名前判定の境界。
    #[test]
    fn req41_expected_format_boundaries() {
        for p in [
            "model.safetensors",
            "model-00001-of-00002.safetensors",
            "sub/w.safetensors",
        ] {
            assert_eq!(
                expected_format(Path::new(p)),
                Some(FileFormat::Safetensors),
                "{p}"
            );
        }
        assert_eq!(
            expected_format(Path::new("tokenizer.json")),
            Some(FileFormat::TokenizerJson)
        );
        for p in [
            "pytorch_model.bin",
            "model.pt",
            "config.json",
            "model.SAFETENSORS",
            ".safetensors",
            "tokenizer.JSON",
            "my_tokenizer.json",
            "model",
        ] {
            assert_eq!(expected_format(Path::new(p)), None, "{p}");
        }
    }

    /// REQ-41・REQ-39: 拒否型の終了コードと理由コード。
    #[test]
    fn req41_name_rejection_codes() {
        let e = BaseModelFileRejection::NameNotAllowed;
        assert_eq!(e.exit_code(), ExitCode::InvalidInput);
        assert_eq!(e.reason_code(), "base_model_file_name_not_allowed");
    }
}
