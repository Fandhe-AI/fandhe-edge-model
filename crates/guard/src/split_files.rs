//! 分割ファイル群（重みの shard 等）の受け付けとパッケージ合計の上限（REQ-39・REQ-41・TASK-41.2-5・#580・#400）。
//!
//! 1 ファイルの上限は重みファイル専用の [`MAX_SHARD_FILE_BYTES`]（2 GiB）。他のファイルの上限
//! [`MAX_READ_FILE_BYTES`]（1 GiB）は据え置き。分割された複数ファイルを
//! まとめて受け付けて、合計を [`MAX_PACKAGE_TOTAL_BYTES`] と照合し、各ファイルの sha256 を記録・照合する。
//! ベースモデルのディレクトリ受付（#398）・ONNX 外部データの読み込み（#581）・来歴への記録（#399）から
//! 呼ばれる想定で、現時点で CLI・trainer には未接続。形式（safetensors か）の判定は #397 の責務で扱わない。
//!
//! # 設計上の要点
//!
//! - 入力は [`crate::path::open_confined`] 等が返した開いたハンドル。パスから開き直さない（TOCTOU を作らない）
//! - 検査順序は 件数 → 各ファイルの fstat（通常ファイル・2 GiB 以下）→ 合計 → ハッシュ。合計超過までは 1 バイトも読まない
//! - 合計は飽和加算（溢れは超過扱い）。ハッシュは確認済み fstat サイズを上限とするストリームで、fstat 後に伸びたファイルは拒否する
//! - 拒否メッセージにパス・内容を含めない（数値と index のみ）
//!
//! # 合計上限の値（暫定。オーナー確認事項）
//!
//! 20 GiB。Qwen2.5-7B の bf16 重み（15,231 MB）の約 1.4 倍（#654。オーナー承認 2026-10-25）。
//! 1.5B（約 3.09GB）にも十分な余裕がある暴走止めの上限。証拠種別: 推定。

use crate::file_size::{FileSizeRejection, check_file_size_unclamped};
use crate::path::ConfinedPath;
use fandhe_edge_core::exitcode::ExitCode;
use fandhe_edge_core::fs::{FsError, sha256_open_file_bounded};
use fandhe_edge_core::hash::Sha256Digest;
use std::fmt;
use std::fs::File;

/// 重みファイル（外部データ shard）専用の 1 ファイル上限（2 GiB）。
///
/// 7B の埋め込み 1 テンソル（約 1.09 GB）が [`crate::file_size::MAX_READ_FILE_BYTES`]（1 GiB）に収まらないため。
/// オーナー判断 2026-10-25・#654（REQ-39・REQ-41）。他のファイルは 1 GiB のまま。
pub const MAX_SHARD_FILE_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// パッケージ（分割ファイル群）の合計サイズの上限（20 GiB。7B の bf16 重み 15,231 MB の約 1.4 倍。証拠種別: 推定）。
pub const MAX_PACKAGE_TOTAL_BYTES: u64 = 20 * 1024 * 1024 * 1024;

/// 受け付ける分割ファイルの最大件数（ハッシュ計算の時間と fd 数を有界にする）。
pub const MAX_SPLIT_FILES: usize = fandhe_edge_core::limits::MAX_EXTERNAL_DATA_SHARDS;

/// 呼び出し側の希望上限を天井 [`MAX_PACKAGE_TOTAL_BYTES`] で丸めた実効上限を返す。
pub const fn effective_total_limit(requested: u64) -> u64 {
    if requested < MAX_PACKAGE_TOTAL_BYTES {
        requested
    } else {
        MAX_PACKAGE_TOTAL_BYTES
    }
}

/// 検査を通った 1 ファイル。
#[derive(Debug)]
pub struct SplitFile {
    path: ConfinedPath,
    size: u64,
    sha256: Sha256Digest,
}

impl SplitFile {
    /// 検証済み経路。
    pub fn path(&self) -> &ConfinedPath {
        &self.path
    }
    /// fstat 時点のサイズ（バイト）。
    pub fn size(&self) -> u64 {
        self.size
    }
    /// 内容の sha256。
    pub fn sha256(&self) -> Sha256Digest {
        self.sha256
    }
}

/// 検査を通った分割ファイル群（入力と同じ順序）。
#[derive(Debug)]
pub struct SplitFiles {
    files: Vec<SplitFile>,
    total: u64,
    limit: u64,
}

impl SplitFiles {
    /// 各ファイル（入力順）。
    pub fn files(&self) -> &[SplitFile] {
        &self.files
    }
    /// 合計サイズ（バイト）。
    pub fn total(&self) -> u64 {
        self.total
    }
    /// 適用した合計の実効上限（バイト）。
    pub fn limit(&self) -> u64 {
        self.limit
    }
}

/// 分割ファイル群の拒否理由。
#[derive(Debug)]
#[non_exhaustive]
pub enum SplitFilesRejection {
    /// ファイルが 0 件。
    Empty,
    /// 件数が [`MAX_SPLIT_FILES`] を超える。
    TooMany {
        /// 渡された件数。
        count: usize,
        /// 件数の上限。
        limit: usize,
    },
    /// `index` 番目のファイルがサイズ検査（2 GiB・通常ファイル）で拒否された。
    File {
        /// 入力順の位置。
        index: usize,
        /// 内包する拒否理由。
        source: FileSizeRejection,
    },
    /// 合計が実効上限を超える（溢れた場合 `total` は `u64::MAX`）。
    TotalTooLarge {
        /// 合計サイズ（バイト）。
        total: u64,
        /// 適用した実効上限（バイト）。
        limit: u64,
    },
    /// `index` 番目のハッシュ計算に失敗した。
    Hash {
        /// 入力順の位置。
        index: usize,
        /// 内包するエラー。
        source: FsError,
    },
    /// 期待ハッシュの件数がファイル数と一致しない。
    DigestCountMismatch {
        /// 期待ハッシュの件数。
        expected: usize,
        /// ファイル数。
        actual: usize,
    },
    /// `index` 番目のハッシュが期待と一致しない。
    HashMismatch {
        /// 入力順の位置。
        index: usize,
    },
}

impl SplitFilesRejection {
    /// 終了コード（REQ-21）。ハッシュ不一致は `version_ledger` と同じく `InvalidInput`。
    pub fn exit_code(&self) -> ExitCode {
        match self {
            Self::Empty | Self::DigestCountMismatch { .. } | Self::HashMismatch { .. } => {
                ExitCode::InvalidInput
            }
            Self::TooMany { .. } | Self::TotalTooLarge { .. } => ExitCode::LimitExceeded,
            Self::File { source, .. } => source.exit_code(),
            Self::Hash { source, .. } => match source {
                FsError::TooLarge { .. } => ExitCode::LimitExceeded,
                FsError::NotRegularFile { .. } => ExitCode::InvalidInput,
                _ => ExitCode::RuntimeError,
            },
        }
    }

    /// 機械可読な理由コード（英語の snake_case）。
    pub fn reason_code(&self) -> &'static str {
        match self {
            Self::Empty => "split_files_empty",
            Self::TooMany { .. } => "split_files_count_exceeded",
            Self::File { source, .. } => source.reason_code(),
            Self::TotalTooLarge { .. } => "package_total_size_limit_exceeded",
            Self::Hash { source, .. } => match source {
                FsError::TooLarge { .. } => "file_size_limit_exceeded",
                FsError::NotRegularFile { .. } => "not_regular_file",
                _ => "file_read_failed",
            },
            Self::DigestCountMismatch { .. } => "split_file_digest_count_mismatch",
            Self::HashMismatch { .. } => "split_file_hash_mismatch",
        }
    }
}

impl fmt::Display for SplitFilesRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "no split files given"),
            Self::TooMany { count, limit } => {
                write!(f, "too many split files ({count} > {limit})")
            }
            Self::File { index, source } => write!(f, "split file #{index}: {source}"),
            Self::TotalTooLarge { total, limit } => write!(
                f,
                "package total exceeds size limit ({total} > {limit} bytes)"
            ),
            Self::Hash { index, .. } => write!(f, "split file #{index}: hash computation failed"),
            Self::DigestCountMismatch { expected, actual } => write!(
                f,
                "digest count mismatch ({expected} expected, {actual} files)"
            ),
            Self::HashMismatch { index } => write!(f, "split file #{index}: sha256 mismatch"),
        }
    }
}

impl std::error::Error for SplitFilesRejection {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::File { source, .. } => Some(source),
            Self::Hash { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// 分割ファイル群を受け付ける。1 ファイル 2 GiB 以下・合計が実効上限以下を fstat で確認してから
/// （この間 1 バイトも読まない）、各ファイルの sha256 を計算して返す。
pub fn check_split_files(
    members: Vec<(File, ConfinedPath)>,
    total_limit: u64,
) -> Result<SplitFiles, SplitFilesRejection> {
    if members.is_empty() {
        return Err(SplitFilesRejection::Empty);
    }
    if members.len() > MAX_SPLIT_FILES {
        return Err(SplitFilesRejection::TooMany {
            count: members.len(),
            limit: MAX_SPLIT_FILES,
        });
    }
    let limit = effective_total_limit(total_limit);
    let mut sizes = Vec::with_capacity(members.len());
    let mut total: u64 = 0;
    for (index, (file, _)) in members.iter().enumerate() {
        let size = check_file_size_unclamped(file, MAX_SHARD_FILE_BYTES)
            .map_err(|source| SplitFilesRejection::File { index, source })?;
        total = total.saturating_add(size);
        sizes.push(size);
    }
    if total > limit {
        return Err(SplitFilesRejection::TotalTooLarge { total, limit });
    }
    let mut files = Vec::with_capacity(members.len());
    for (index, ((file, path), size)) in members.into_iter().zip(sizes).enumerate() {
        // 上限は確認済みの fstat サイズ。fstat 後に伸びたファイルは拒否し、実際にハッシュする内容も
        // 合計上限の内側に収める。
        let sha256 = sha256_open_file_bounded(file, path.as_path(), size)
            .map_err(|source| SplitFilesRejection::Hash { index, source })?;
        files.push(SplitFile { path, size, sha256 });
    }
    Ok(SplitFiles {
        files,
        total,
        limit,
    })
}

/// [`check_split_files`] に加え、記録済みの sha256（入力と同じ順序）と照合する。不一致は fail-closed。
pub fn verify_split_files(
    members: Vec<(File, ConfinedPath)>,
    expected: &[Sha256Digest],
    total_limit: u64,
) -> Result<SplitFiles, SplitFilesRejection> {
    if members.len() != expected.len() {
        return Err(SplitFilesRejection::DigestCountMismatch {
            expected: expected.len(),
            actual: members.len(),
        });
    }
    let checked = check_split_files(members, total_limit)?;
    for (index, (f, want)) in checked.files.iter().zip(expected).enumerate() {
        if f.sha256 != *want {
            return Err(SplitFilesRejection::HashMismatch { index });
        }
    }
    Ok(checked)
}
