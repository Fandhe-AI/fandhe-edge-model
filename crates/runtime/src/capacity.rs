//! 容量計測コア（REQ-30・TASK-30.1-1・#122）。
//!
//! モデルパッケージの非圧縮合計バイト数を、構成要素（重み・語彙または特徴量変換の
//! 定義・選択肢表・校正設定・メタデータ）ごとの内訳として集計する。CLI の `package`
//! 工程（#123・TASK-30.1-2）が JSON 出力へ接続する際の下位ロジックで、本モジュールは
//! 値の型と集計だけを持つ。
//!
//! # 範囲外（後続 TASK の責務）
//!
//! - 上限との照合と `limit_exceeded`（TASK-30.2・#124）。40MB は目安であり、本モジュールは
//!   合否の真偽値を持たない
//! - JSON 出力・CLI への接続、7 種の終了コードへの写像（#123）
//! - 経路の閉じ込め（`../`・ルート外参照。ガード層 REQ-39）と sha256 検証（TASK-28・39）。
//!   呼び出し側が渡すパスは検証済みである前提
//! - 構成要素の分類。どのファイルをどの要素に数えるかは呼び出し側が明示する
//!   （配布パッケージ形式は TASK-28・32 で決まるため、ここでは推測しない）
//!
//! 実機での C1・C3 の実測は人の作業であり、本モジュールのテストは生成物によるテスト
//! ハーネスの証拠にとどまる。

use fandhe_edge_core::fs::{FsError, open_regular_file_for_read};
use std::collections::HashSet;
use std::fmt;
use std::path::PathBuf;

/// パッケージの構成要素（REQ-30 の内訳 5 種）。宣言順が内訳の並び順になる。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum PackageComponent {
    /// 重み。
    Weights,
    /// 語彙ファイルまたは特徴量変換の定義。
    VocabOrFeatureTransform,
    /// 選択肢表。
    LabelTable,
    /// 校正設定。
    Calibration,
    /// メタデータ。
    Metadata,
}

impl PackageComponent {
    /// 宣言順の全バリアント。
    pub fn all() -> [PackageComponent; 5] {
        [
            PackageComponent::Weights,
            PackageComponent::VocabOrFeatureTransform,
            PackageComponent::LabelTable,
            PackageComponent::Calibration,
            PackageComponent::Metadata,
        ]
    }

    /// 英語の識別子（#123 の JSON キーに使う想定）。
    pub fn as_str(self) -> &'static str {
        match self {
            PackageComponent::Weights => "weights",
            PackageComponent::VocabOrFeatureTransform => "vocab_or_feature_transform",
            PackageComponent::LabelTable => "label_table",
            PackageComponent::Calibration => "calibration",
            PackageComponent::Metadata => "metadata",
        }
    }

    fn index(self) -> usize {
        match self {
            PackageComponent::Weights => 0,
            PackageComponent::VocabOrFeatureTransform => 1,
            PackageComponent::LabelTable => 2,
            PackageComponent::Calibration => 3,
            PackageComponent::Metadata => 4,
        }
    }
}

/// 1 構成要素のバイト数とファイル数。
///
/// 要素が無い（`file_count == 0`）場合と 0 バイトのファイルがある場合を区別できる。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ComponentBytes {
    /// 非圧縮バイト数の合計。
    pub bytes: u64,
    /// 計上したファイル数。
    pub file_count: u32,
}

/// 構成要素ごとの内訳。5 要素すべてのエントリを常に持つ。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapacityBreakdown {
    entries: [ComponentBytes; 5],
    total: u64,
}

/// 容量計測のエラー。
#[derive(Debug)]
#[non_exhaustive]
pub enum CapacityError {
    /// 計測対象が 1 件も無い。
    EmptyPackage,
    /// 同じファイルが複数回渡された（開いたハンドルの同一性で判定し、Unix では `a` と `./a`
    /// の別表記・ハードリンクの別名も検出する）。
    DuplicatePath { path: PathBuf },
    /// 構成要素が symlink だった、または検査後に別ファイルへ差し替えられた。
    SymlinkRejected { path: PathBuf },
    /// 通常ファイル以外・I/O エラー。
    File(FsError),
    /// 合計が `u64`（またはファイル数が `u32`）を超えた。
    Overflow,
}

impl fmt::Display for CapacityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CapacityError::EmptyPackage => write!(f, "package has no files to measure"),
            CapacityError::DuplicatePath { path } => {
                write!(f, "duplicate package file path: {}", path.display())
            }
            CapacityError::SymlinkRejected { path } => {
                write!(f, "package file is a symlink: {}", path.display())
            }
            CapacityError::File(e) => write!(f, "{e}"),
            CapacityError::Overflow => write!(f, "package size counters overflow"),
        }
    }
}

impl std::error::Error for CapacityError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CapacityError::File(e) => Some(e),
            _ => None,
        }
    }
}

impl From<FsError> for CapacityError {
    fn from(e: FsError) -> Self {
        CapacityError::File(e)
    }
}

impl CapacityBreakdown {
    /// (構成要素, バイト数) の列から集計する。同じ要素は件数・バイト数を合算する。
    ///
    /// 入力が空なら [`CapacityError::EmptyPackage`]、加算があふれたら
    /// [`CapacityError::Overflow`]（合計を黙って丸めない。fail-closed）。
    pub fn from_sizes(
        sizes: impl IntoIterator<Item = (PackageComponent, u64)>,
    ) -> Result<Self, CapacityError> {
        let mut entries = [ComponentBytes::default(); 5];
        let mut any = false;
        let mut total: u64 = 0;
        for (component, size) in sizes {
            any = true;
            total = total.checked_add(size).ok_or(CapacityError::Overflow)?;
            let entry = entries
                .get_mut(component.index())
                .ok_or(CapacityError::Overflow)?;
            entry.bytes = entry
                .bytes
                .checked_add(size)
                .ok_or(CapacityError::Overflow)?;
            entry.file_count = entry
                .file_count
                .checked_add(1)
                .ok_or(CapacityError::Overflow)?;
        }
        if !any {
            return Err(CapacityError::EmptyPackage);
        }
        Ok(Self { entries, total })
    }

    /// 指定した構成要素のバイト数・ファイル数。
    pub fn component(&self, component: PackageComponent) -> ComponentBytes {
        self.entries
            .get(component.index())
            .copied()
            .unwrap_or_default()
    }

    /// 宣言順の全エントリ。
    pub fn entries(&self) -> Vec<(PackageComponent, ComponentBytes)> {
        PackageComponent::all()
            .into_iter()
            .map(|c| (c, self.component(c)))
            .collect()
    }

    /// 合計バイト数（内訳の総和。構築時に checked 加算で検証済み）。
    pub fn total_bytes(&self) -> u64 {
        self.total
    }
}

/// 計測対象ファイル。構成要素は呼び出し側が明示する。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackageFile {
    /// このファイルが属する構成要素。
    pub component: PackageComponent,
    /// ファイルのパス（検証済みである前提。モジュール doc 参照）。
    pub path: PathBuf,
}

/// 開いたファイルハンドル自体の同一性（重複判定の鍵）。
///
/// Unix では (デバイス, inode)。パスの別表記・ハードリンクの別名を同一ファイルとして
/// 検出できる。それ以外の OS（M10 時点で対象外）では、開いた後に正規化したパスで代用する。
#[cfg(unix)]
type FileId = (u64, u64);
#[cfg(not(unix))]
type FileId = PathBuf;

#[cfg(unix)]
fn file_id(_path: &std::path::Path, meta: &std::fs::Metadata) -> std::io::Result<FileId> {
    use std::os::unix::fs::MetadataExt as _;
    Ok((meta.dev(), meta.ino()))
}

#[cfg(not(unix))]
fn file_id(path: &std::path::Path, _meta: &std::fs::Metadata) -> std::io::Result<FileId> {
    std::fs::canonicalize(path)
}

/// 実ファイルのサイズを構成要素ごとに集計する。中身は読まない。
///
/// symlink は拒否する。検査から計測までの間にパスが差し替えられても（TOCTOU。REQ-39）
/// 検証済み以外のファイルを計上しないよう、次の順で行う。
///
/// 1. `symlink_metadata` で symlink を拒否し、そのメタデータを控える
/// 2. `open_regular_file_for_read` で通常ファイル検証つきに開く（FIFO 等で停止しない）
/// 3. 開いたハンドルの `fstat` 相当（`file.metadata()`）が 1 と同一のファイルであること
///    （Unix ではデバイス・inode の一致）を確認し、不一致なら symlink 差し替えとして拒否する
/// 4. サイズと重複判定の鍵は、パスではなく開いたハンドルから得る
///
/// 重複判定は Unix ではハンドルの (デバイス, inode) で行うため、`a` と `./a` の別表記や
/// ハードリンクの別名も [`CapacityError::DuplicatePath`] になる。
pub fn measure_package(files: &[PackageFile]) -> Result<CapacityBreakdown, CapacityError> {
    let mut seen: HashSet<FileId> = HashSet::new();
    let mut sizes = Vec::with_capacity(files.len());
    for f in files {
        let read_err = |source| {
            CapacityError::File(FsError::Read {
                path: f.path.clone(),
                source,
            })
        };
        let link_meta = std::fs::symlink_metadata(&f.path).map_err(read_err)?;
        if link_meta.file_type().is_symlink() {
            return Err(CapacityError::SymlinkRejected {
                path: f.path.clone(),
            });
        }
        let file = open_regular_file_for_read(&f.path)?;
        let meta = file.metadata().map_err(read_err)?;
        let opened_id = file_id(&f.path, &meta).map_err(read_err)?;
        let checked_id = file_id(&f.path, &link_meta).map_err(read_err)?;
        // 検査後に別ファイル（symlink 先など）へ差し替えられていたら、検証済みの
        // 対象ではないため計測しない。
        #[cfg(unix)]
        if opened_id != checked_id {
            return Err(CapacityError::SymlinkRejected {
                path: f.path.clone(),
            });
        }
        #[cfg(not(unix))]
        let _ = checked_id;
        if !seen.insert(opened_id) {
            return Err(CapacityError::DuplicatePath {
                path: f.path.clone(),
            });
        }
        sizes.push((f.component, meta.len()));
    }
    CapacityBreakdown::from_sizes(sizes)
}

#[cfg(test)]
mod tests {
    use super::PackageComponent::*;
    use super::*;

    #[test]
    fn req30_breakdown_individual_and_total() {
        let b = CapacityBreakdown::from_sizes([(Weights, 1000), (LabelTable, 37), (Metadata, 512)])
            .unwrap();
        assert_eq!(b.component(Weights).bytes, 1000);
        assert_eq!(b.component(LabelTable).bytes, 37);
        assert_eq!(b.component(Metadata).bytes, 512);
        assert_eq!(
            b.component(VocabOrFeatureTransform),
            ComponentBytes::default()
        );
        assert_eq!(b.component(Calibration).file_count, 0);
        assert_eq!(b.total_bytes(), 1549);
    }

    #[test]
    fn req30_same_component_is_summed() {
        let b = CapacityBreakdown::from_sizes([(Weights, 300), (Weights, 200)]).unwrap();
        assert_eq!(
            b.component(Weights),
            ComponentBytes {
                bytes: 500,
                file_count: 2
            }
        );
    }

    #[test]
    fn req30_zero_byte_file_differs_from_absent() {
        let b = CapacityBreakdown::from_sizes([(Weights, 1), (Calibration, 0)]).unwrap();
        assert_eq!(
            b.component(Calibration),
            ComponentBytes {
                bytes: 0,
                file_count: 1
            }
        );
        assert_ne!(b.component(Calibration), b.component(Metadata));
    }

    #[test]
    fn req30_empty_input_is_rejected() {
        assert!(matches!(
            CapacityBreakdown::from_sizes([]),
            Err(CapacityError::EmptyPackage)
        ));
    }

    #[test]
    fn req30_overflow_is_rejected() {
        assert!(matches!(
            CapacityBreakdown::from_sizes([(Weights, u64::MAX), (Metadata, 1)]),
            Err(CapacityError::Overflow)
        ));
    }

    #[test]
    fn req30_all_and_as_str_are_declaration_ordered() {
        let names: Vec<_> = PackageComponent::all().iter().map(|c| c.as_str()).collect();
        assert_eq!(
            names,
            [
                "weights",
                "vocab_or_feature_transform",
                "label_table",
                "calibration",
                "metadata"
            ]
        );
        let b = CapacityBreakdown::from_sizes([(Metadata, 2), (Weights, 1)]).unwrap();
        let order: Vec<_> = b.entries().iter().map(|(c, _)| *c).collect();
        assert_eq!(order, PackageComponent::all());
    }
}
