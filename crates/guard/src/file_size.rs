//! 読み込み前のファイルサイズ上限の検証（REQ-39 境界値「読み込むファイルサイズ」・TASK-39.5-3・#172・
//! PoC-20 ケース 3・C-16）。
//!
//! 暫定上限 [`MAX_READ_FILE_BYTES`]（1 GiB）を超えるファイルを、中身を 1 バイトも読まずに拒否する。
//! 本実装は模擬ではなく、開いたハンドルへの `fstat`（[`File::metadata`]）による実際のサイズ確認である。
//! この暫定値は推論経路（ガード層）向けで、学習ワーカーの上限（`trainer` の `limits.py`）とは別。
//!
//! # 設計上の要点
//!
//! - 検査は**ハンドル基準**。PoC のパス基準の確認と違い、[`crate::path::open_confined`] が返した
//!   ハンドルを開き直さないため、確認と読み込みの間の差し替え（TOCTOU）を再び開かない
//! - 検査順序は 経路 → サイズ → 形式。通常ファイル以外（FIFO・デバイス等）も拒否する
//! - 天井 [`MAX_READ_FILE_BYTES`] は [`effective_read_limit`] で `format`・`version_ledger` の
//!   読み込み口にも適用され、呼び出し側は 1 GiB を超える上限を渡せない
//! - 確認後にファイルが伸びても、読み込み側（`core::fs` の `take(limit + 1)`）で頭打ちになる
//! - 拒否メッセージにパス・ファイル内容を含めない（size・limit の数値のみ）
//!
//! CLI の `infer --input-file` への配線と `ToErrorReport` 写像は #136 の範囲。

use crate::path::{ConfinedPath, PathRejection, open_confined};
use fandhe_edge_core::exitcode::ExitCode;
use std::fmt;
use std::fs::File;
use std::path::Path;

/// 読み込むファイルサイズの暫定上限（1 GiB。PoC-20 の `MAX_INPUT_FILE_BYTES` 相当。REQ-39）。
pub const MAX_READ_FILE_BYTES: u64 = 1024 * 1024 * 1024;

/// 呼び出し側の希望上限を、ガード層の天井 [`MAX_READ_FILE_BYTES`] で丸めた実効上限を返す。
pub const fn effective_read_limit(requested: u64) -> u64 {
    if requested < MAX_READ_FILE_BYTES {
        requested
    } else {
        MAX_READ_FILE_BYTES
    }
}

/// ファイルサイズ検査の拒否理由。
#[derive(Debug)]
#[non_exhaustive]
pub enum FileSizeRejection {
    /// サイズが実効上限を超える。`limit` は天井で丸めた実効値。
    TooLarge {
        /// ファイルの論理サイズ（バイト）。
        size: u64,
        /// 適用した実効上限（バイト）。
        limit: u64,
    },
    /// 通常ファイルではない（FIFO・ディレクトリ・デバイス等）。
    NotRegularFile,
    /// メタデータを取得できなかった。
    Metadata(std::io::Error),
}

impl FileSizeRejection {
    /// 終了コード（REQ-21）。超過は `LimitExceeded`、通常ファイル以外は `InvalidInput`、取得失敗は `RuntimeError`。
    pub fn exit_code(&self) -> ExitCode {
        match self {
            FileSizeRejection::TooLarge { .. } => ExitCode::LimitExceeded,
            FileSizeRejection::NotRegularFile => ExitCode::InvalidInput,
            FileSizeRejection::Metadata(_) => ExitCode::RuntimeError,
        }
    }

    /// 機械可読な理由コード（英語の snake_case）。
    pub const fn reason_code(&self) -> &'static str {
        match self {
            FileSizeRejection::TooLarge { .. } => "file_size_limit_exceeded",
            FileSizeRejection::NotRegularFile => "not_regular_file",
            FileSizeRejection::Metadata(_) => "file_metadata_unavailable",
        }
    }
}

impl fmt::Display for FileSizeRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FileSizeRejection::TooLarge { size, limit } => {
                write!(f, "file exceeds size limit ({size} > {limit} bytes)")
            }
            FileSizeRejection::NotRegularFile => write!(f, "not a regular file"),
            FileSizeRejection::Metadata(_) => write!(f, "file metadata is unavailable"),
        }
    }
}

impl std::error::Error for FileSizeRejection {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            FileSizeRejection::Metadata(e) => Some(e),
            _ => None,
        }
    }
}

/// 開いたハンドルのサイズを、読み込み前に `limit`（天井で丸め）と照合する。1 バイトも読まない。
///
/// 通れば論理サイズを返す。`limit` が天井を超える場合は [`MAX_READ_FILE_BYTES`] を適用する。
pub fn check_file_size_within(file: &File, limit: u64) -> Result<u64, FileSizeRejection> {
    let meta = file.metadata().map_err(FileSizeRejection::Metadata)?;
    if !meta.file_type().is_file() {
        return Err(FileSizeRejection::NotRegularFile);
    }
    let limit = effective_read_limit(limit);
    let size = meta.len();
    if size > limit {
        return Err(FileSizeRejection::TooLarge { size, limit });
    }
    Ok(size)
}

/// 既定の上限 [`MAX_READ_FILE_BYTES`] でサイズを検査する。
pub fn check_file_size(file: &File) -> Result<u64, FileSizeRejection> {
    check_file_size_within(file, MAX_READ_FILE_BYTES)
}

/// [`open_confined_size_checked`] の拒否理由。
#[derive(Debug)]
#[non_exhaustive]
pub enum SizeCheckedOpenRejection {
    /// 経路の閉じ込め・open の拒否。
    Path(PathRejection),
    /// サイズ検査の拒否。
    Size(FileSizeRejection),
}

impl SizeCheckedOpenRejection {
    /// 終了コード（REQ-21）。内包する拒否理由のものを返す。
    pub fn exit_code(&self) -> ExitCode {
        match self {
            SizeCheckedOpenRejection::Path(e) => e.exit_code(),
            SizeCheckedOpenRejection::Size(e) => e.exit_code(),
        }
    }

    /// 機械可読な理由コード（英語の snake_case）。
    pub const fn reason_code(&self) -> &'static str {
        match self {
            SizeCheckedOpenRejection::Path(e) => e.reason_code(),
            SizeCheckedOpenRejection::Size(e) => e.reason_code(),
        }
    }
}

impl fmt::Display for SizeCheckedOpenRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SizeCheckedOpenRejection::Path(e) => e.fmt(f),
            SizeCheckedOpenRejection::Size(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for SizeCheckedOpenRejection {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SizeCheckedOpenRejection::Path(e) => Some(e),
            SizeCheckedOpenRejection::Size(e) => Some(e),
        }
    }
}

/// 閉じ込め検証とサイズ確認を通ったハンドル。パスから開き直さず、このハンドルを読むこと。
#[derive(Debug)]
pub struct SizeCheckedFile {
    file: File,
    path: ConfinedPath,
    size: u64,
}

impl SizeCheckedFile {
    /// 確認時点のサイズ（バイト）。
    pub fn size(&self) -> u64 {
        self.size
    }

    /// ハンドル・検証済み経路・確認済みサイズに分解する。
    pub fn into_parts(self) -> (File, ConfinedPath, u64) {
        (self.file, self.path, self.size)
    }
}

/// `root` 配下の `candidate` を [`open_confined`] で開き、読み込み前にサイズを確認する
/// （`infer --input-file` 等の利用者入力ファイル向け入口。TASK-39.5-3・#172）。
pub fn open_confined_size_checked(
    root: &Path,
    candidate: &Path,
    limit: u64,
) -> Result<SizeCheckedFile, SizeCheckedOpenRejection> {
    let (file, path) = open_confined(root, candidate).map_err(SizeCheckedOpenRejection::Path)?;
    let size = check_file_size_within(&file, limit).map_err(SizeCheckedOpenRejection::Size)?;
    Ok(SizeCheckedFile { file, path, size })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_file(name: &str, len: usize) -> (std::path::PathBuf, File) {
        let p = std::env::temp_dir().join(format!("fe-guard-fsz-{}-{name}", std::process::id()));
        let mut f = File::create(&p).unwrap();
        f.write_all(&vec![0u8; len]).unwrap();
        drop(f);
        let f = File::open(&p).unwrap();
        (p, f)
    }

    /// REQ-39・TASK-39.5-3: 上限ちょうどは通り、+1 は拒否、0 バイトは通る。
    #[test]
    fn req39_boundaries_with_injected_limit() {
        let (p, f) = temp_file("b", 10);
        assert_eq!(check_file_size_within(&f, 10).unwrap(), 10);
        match check_file_size_within(&f, 9) {
            Err(FileSizeRejection::TooLarge { size: 10, limit: 9 }) => {}
            other => panic!("unexpected {other:?}"),
        }
        let (p0, f0) = temp_file("z", 0);
        assert_eq!(check_file_size_within(&f0, 0).unwrap(), 0);
        let _ = std::fs::remove_file(p);
        let _ = std::fs::remove_file(p0);
    }

    /// REQ-39・TASK-39.5-3: 天井の丸め。
    #[test]
    fn req39_effective_read_limit_clamps() {
        assert_eq!(effective_read_limit(u64::MAX), 1_073_741_824);
        assert_eq!(effective_read_limit(44 * 1024 * 1024), 44 * 1024 * 1024);
        assert_eq!(
            effective_read_limit(MAX_READ_FILE_BYTES),
            MAX_READ_FILE_BYTES
        );
    }

    /// REQ-39・TASK-39.5-3: ディレクトリのハンドルは通常ファイルではない。
    #[cfg(unix)]
    #[test]
    fn req39_directory_handle_is_not_regular_file() {
        let d = File::open(std::env::temp_dir()).unwrap();
        let e = check_file_size(&d).unwrap_err();
        assert!(matches!(e, FileSizeRejection::NotRegularFile));
        assert_eq!(e.exit_code(), ExitCode::InvalidInput);
        assert_eq!(e.reason_code(), "not_regular_file");
    }
}
