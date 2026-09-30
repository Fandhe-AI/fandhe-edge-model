//! 容量計測コア（REQ-30・TASK-30.1-1・#122）。
//!
//! モデルパッケージの非圧縮合計バイト数を、構成要素（重み・語彙または特徴量変換の
//! 定義・選択肢表・校正設定・メタデータ）ごとの内訳として集計する。CLI の `package`
//! 工程（#123・TASK-30.1-2）が JSON 出力へ接続する際の下位ロジックで、本モジュールは
//! 値の型と集計、およびエラーの終了コード・公開メッセージへの写像（#123）を持つ。
//! JSON への直列化は CLI 側の責務（`fandhe-edge-cli` の `output` モジュール）。
//!
//! # 範囲外（後続 TASK の責務）
//!
//! - 上限との照合（TASK-30.2・#124）。`limit_exceeded` への終了コード決定は
//!   [`crate::package_outcome`]（#132）。40MB は目安であり、本モジュールは
//!   合否の真偽値を持たない。語彙ファイル超過構成の目安との比較・除外記録は
//!   [`crate::vocab_exclusion`]（TASK-30.3・#125）
//! - JSON 直列化・CLI 工程への配線（CLI 側。#123・TASK-33.1）
//! - 経路の閉じ込め（`../`・ルート外参照。ガード層 REQ-39）と sha256 検証（TASK-28・39）。
//!   呼び出し側が渡すパスは検証済みである前提
//! - 構成要素の分類。どのファイルをどの要素に数えるかは呼び出し側が明示する
//!   （配布パッケージ形式は TASK-28・32 で決まるため、ここでは推測しない）
//!
//! 実機での C1・C3 の実測は人の作業であり、本モジュールのテストは生成物によるテスト
//! ハーネスの証拠にとどまる。

use fandhe_edge_core::exitcode::ExitCode;
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
    /// ハンドル由来のファイル同一性を取得できない環境（Unix 以外）で、重複がないことを
    /// 証明できない複数ファイルの計測を求められた。近似せず fail-closed で拒否する（REQ-39）。
    IdentityUnavailable,
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
            CapacityError::IdentityUnavailable => write!(
                f,
                "file identity is unavailable on this platform; cannot verify duplicates"
            ),
            CapacityError::Overflow => write!(f, "package size counters overflow"),
        }
    }
}

impl CapacityError {
    /// 7 種の終了コードへの写像（REQ-21・REQ-30・#123）。分類はここ 1 か所に集約する。
    ///
    /// - 資源超過 `LimitExceeded`（20）: 1 ファイルの上限超過（`FsError::TooLarge`）・合計の
    ///   あふれ（`Overflow`）。利用者が設定した合計容量の上限（TASK-30.2・#124）ではない
    /// - 実行時エラー `RuntimeError`（70）: 計測中の I/O 障害（権限・EIO 等。`NotFound`・
    ///   `InvalidInput` 以外の `io::ErrorKind` はすべて環境起因として扱う fail-closed）と、
    ///   プラットフォームの機能不足（`IdentityUnavailable`）。入力を直しても解消しない
    /// - 入力不正 `InvalidInput`（64）: 上記以外の利用者が直せる不正（不在・非通常ファイル・
    ///   symlink・重複・空のパッケージ）。将来増える `FsError` も入力不正に倒す
    #[must_use]
    pub fn exit_code(&self) -> ExitCode {
        match self {
            CapacityError::File(FsError::TooLarge { .. }) | CapacityError::Overflow => {
                ExitCode::LimitExceeded
            }
            CapacityError::IdentityUnavailable => ExitCode::RuntimeError,
            CapacityError::File(FsError::Read { source, .. }) => match source.kind() {
                std::io::ErrorKind::NotFound | std::io::ErrorKind::InvalidInput => {
                    ExitCode::InvalidInput
                }
                _ => ExitCode::RuntimeError,
            },
            _ => ExitCode::InvalidInput,
        }
    }

    /// パス・io エラー本文を含まない固定文（英語）。エラー JSON の `message` 用。
    ///
    /// `Display` は診断用にパス等を含むため `message` には使わない（security.md の
    /// 秘密情報混入防止 P0）。
    #[must_use]
    pub fn public_message(&self) -> String {
        let text = match self {
            CapacityError::EmptyPackage => "package has no files to measure",
            CapacityError::DuplicatePath { .. } => "package file is listed more than once",
            CapacityError::SymlinkRejected { .. } => {
                "package file is a symlink or was replaced during measurement"
            }
            CapacityError::File(FsError::TooLarge { .. }) => "package file exceeds size limit",
            CapacityError::File(FsError::NotRegularFile { .. }) => {
                "package file is not a regular file"
            }
            CapacityError::File(_) => "package file is not readable",
            CapacityError::IdentityUnavailable => "file identity is unavailable on this platform",
            CapacityError::Overflow => "package size counters overflow",
        };
        text.to_string()
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

/// 検査時と開いた後のメタデータが同一ファイルのものと見なせるかを近似判定する
/// （Unix 以外の差し替え検出用。REQ-39）。時刻が取得できない環境では取得可否の一致を要求する。
#[cfg(not(unix))]
fn metadata_snapshot_matches(a: &std::fs::Metadata, b: &std::fs::Metadata) -> bool {
    a.file_type() == b.file_type()
        && a.len() == b.len()
        && a.modified().ok() == b.modified().ok()
        && a.created().ok() == b.created().ok()
}

/// 計測対象 1 ファイルの大きさの上限（バイト。REQ-39）。配布物の容量目安（40MB。REQ-30）を
/// 大きく上回る値で、異常に大きい入力を計測値の段階で拒否する。
pub const MAX_FILE_BYTES: u64 = 1 << 30;

/// 計測で得たサイズを合計へ加算する前に上限と照合する（REQ-39）。事前検査には依存しない。
fn check_size(path: &std::path::Path, size: u64, limit: u64) -> Result<(), CapacityError> {
    if size > limit {
        return Err(CapacityError::File(FsError::TooLarge {
            path: path.to_path_buf(),
            size,
            limit,
        }));
    }
    Ok(())
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
    measure_package_with_limit(files, MAX_FILE_BYTES)
}

/// [`measure_package`] の 1 ファイル上限を指定する版（上限は計測で得たサイズに適用する）。
pub fn measure_package_with_limit(
    files: &[PackageFile],
    limit: u64,
) -> Result<CapacityBreakdown, CapacityError> {
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
        // Unix 以外では標準ライブラリだけでは開いたハンドルの同一性（ボリューム通し番号・
        // ファイル index）を安定版で取得できない（取得には依存追加か `unsafe` が要り、
        // どちらもユーザー承認事項）。代替として、検査時のメタデータと開いたハンドルの
        // メタデータの種別・サイズ・更新時刻・作成時刻を突き合わせ、差し替えの兆候があれば
        // 拒否する（fail-closed。近似であり同一性の証明ではない。M10 時点で対象外）。
        #[cfg(not(unix))]
        {
            let _ = checked_id;
            if !metadata_snapshot_matches(&link_meta, &meta) {
                return Err(CapacityError::SymlinkRejected {
                    path: f.path.clone(),
                });
            }
        }
        if !seen.insert(opened_id) {
            return Err(CapacityError::DuplicatePath {
                path: f.path.clone(),
            });
        }
        check_size(&f.path, meta.len(), limit)?;
        sizes.push((f.component, meta.len()));
    }
    CapacityBreakdown::from_sizes(sizes)
}

/// 呼び出し側が検証つきで開いた通常ファイルのハンドルから、構成要素ごとのサイズを集計する。
///
/// 検証（経路の閉じ込め等）とファイル取得の間でパスを再解決させないための入口（TOCTOU 対策。
/// REQ-39）。サイズと重複判定の鍵は渡されたハンドルの `fstat` 相当から得る。ハンドルが通常
/// ファイルであることの保証は呼び出し側（`open_regular_file_for_read` 等）が負う。
/// `label` はエラー表示用のパスで、再度開くことも再解決することもしない。
///
/// 制約（REQ-39）: 同一ファイルの重複検出（[`CapacityError::DuplicatePath`]）は Unix
/// （デバイス・inode）のみ。Unix 以外ではハンドルの同一性を標準ライブラリだけでは取得できない
/// ため、重複がないことを証明できない複数ファイル（2 件以上）は近似せず
/// [`CapacityError::IdentityUnavailable`] で拒否する（fail-closed。M10 時点で対象外）。
pub fn measure_opened_files(
    files: &[(PackageComponent, PathBuf, std::fs::File)],
) -> Result<CapacityBreakdown, CapacityError> {
    measure_opened_files_with_limit(files, MAX_FILE_BYTES)
}

/// [`measure_opened_files`] の 1 ファイル上限を指定する版。集計に使うメタデータのサイズ自体を
/// 上限と照合する（オープン後に成長したファイルも合計へ加算する前に拒否。REQ-39）。
pub fn measure_opened_files_with_limit(
    files: &[(PackageComponent, PathBuf, std::fs::File)],
    limit: u64,
) -> Result<CapacityBreakdown, CapacityError> {
    let mut seen: HashSet<FileId> = HashSet::new();
    let mut sizes = Vec::with_capacity(files.len());
    for (component, label, file) in files {
        let read_err = |source| {
            CapacityError::File(FsError::Read {
                path: label.clone(),
                source,
            })
        };
        let meta = file.metadata().map_err(read_err)?;
        if !meta.is_file() {
            return Err(CapacityError::File(FsError::NotRegularFile {
                path: label.clone(),
            }));
        }
        // 重複判定の鍵はハンドル由来の (dev, ino) のみ。Unix 以外では std だけでハンドルの
        // 同一性を取れず、`label` の再解決（canonicalize）は契約違反になるため判定しない。
        #[cfg(unix)]
        {
            let id = file_id(label, &meta).map_err(read_err)?;
            if !seen.insert(id) {
                return Err(CapacityError::DuplicatePath {
                    path: label.clone(),
                });
            }
        }
        #[cfg(not(unix))]
        {
            let _ = (&mut seen, read_err);
            if files.len() > 1 {
                return Err(CapacityError::IdentityUnavailable);
            }
        }
        check_size(label, meta.len(), limit)?;
        sizes.push((*component, meta.len()));
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

    /// REQ-30・REQ-39: 同一性を取得できない環境の拒否は InvalidInput・固定文で報告される。
    #[test]
    fn req39_identity_unavailable_maps_to_runtime_error() {
        let e = CapacityError::IdentityUnavailable;
        assert_eq!(e.exit_code(), ExitCode::RuntimeError);
        assert_eq!(
            e.public_message(),
            "file identity is unavailable on this platform"
        );
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

    /// REQ-30・REQ-39: ハンドル渡しの計測はサイズを集計し、同一ファイルの重複は拒否する。
    #[test]
    fn req30_measure_opened_files_sums_and_rejects_duplicates() {
        // 予測可能な名前へ上書きしない。一意なディレクトリを排他作成し、その中へ create_new する
        let dir_path = (0..100)
            .find_map(|i| {
                let nanos = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_nanos());
                let p = std::env::temp_dir().join(format!(
                    "fandhe_cap_opened_{}_{nanos}_{i}",
                    std::process::id()
                ));
                std::fs::create_dir(&p).ok().map(|()| p)
            })
            .expect("unique temp dir");
        let path = dir_path.join("f.bin");
        {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .unwrap();
            f.write_all(b"abcd").unwrap();
        }
        let open = || std::fs::File::open(&path).unwrap();
        let one = measure_opened_files(&[(Weights, path.clone(), open())]).unwrap();
        assert_eq!(one.total_bytes(), 4);
        // ディレクトリのハンドルは NotRegularFile（"not a regular file"）で拒否される。
        // Windows は File::open でディレクトリを開けないため Unix のみで検証する
        #[cfg(unix)]
        {
            let dir = std::fs::File::open(std::env::temp_dir()).unwrap();
            let not_regular = measure_opened_files(&[(Weights, path.clone(), dir)]);
            assert!(matches!(
                not_regular,
                Err(CapacityError::File(FsError::NotRegularFile { .. }))
            ));
        }
        let dup = measure_opened_files(&[
            (Weights, path.clone(), open()),
            (Metadata, path.clone(), open()),
        ]);
        #[cfg(unix)]
        assert!(matches!(dup, Err(CapacityError::DuplicatePath { .. })));
        // Unix 以外は同一性を証明できないため複数件を拒否する（API 契約。doc 参照）
        #[cfg(not(unix))]
        assert!(matches!(dup, Err(CapacityError::IdentityUnavailable)));
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir_path);
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

    #[test]
    fn req21_capacity_error_exit_codes() {
        let p = || PathBuf::from("/x");
        assert_eq!(CapacityError::EmptyPackage.exit_code().code(), 64);
        assert_eq!(
            CapacityError::DuplicatePath { path: p() }
                .exit_code()
                .code(),
            64
        );
        assert_eq!(
            CapacityError::SymlinkRejected { path: p() }
                .exit_code()
                .code(),
            64
        );
        assert_eq!(
            CapacityError::File(FsError::NotRegularFile { path: p() })
                .exit_code()
                .code(),
            64
        );
        assert_eq!(
            CapacityError::File(FsError::Read {
                path: p(),
                source: std::io::Error::from(std::io::ErrorKind::NotFound)
            })
            .exit_code()
            .code(),
            64
        );
        // 計測中の I/O 障害（EIO・権限等）は入力不正ではなく実行時エラー
        for kind in [
            std::io::ErrorKind::PermissionDenied,
            std::io::ErrorKind::TimedOut,
            std::io::ErrorKind::Other,
        ] {
            assert_eq!(
                CapacityError::File(FsError::Read {
                    path: p(),
                    source: std::io::Error::from(kind)
                })
                .exit_code()
                .code(),
                70,
                "{kind:?}"
            );
        }
        assert_eq!(
            CapacityError::File(FsError::TooLarge {
                path: p(),
                size: 2,
                limit: 1
            })
            .exit_code()
            .code(),
            20
        );
        assert_eq!(CapacityError::Overflow.exit_code().code(), 20);
        assert_eq!(CapacityError::IdentityUnavailable.exit_code().code(), 70);
    }

    #[test]
    fn req21_capacity_public_message_does_not_leak() {
        let p = || PathBuf::from("/home/alice/secret-marker.onnx");
        let errs = [
            CapacityError::DuplicatePath { path: p() },
            CapacityError::SymlinkRejected { path: p() },
            CapacityError::File(FsError::NotRegularFile { path: p() }),
            CapacityError::File(FsError::TooLarge {
                path: p(),
                size: 2,
                limit: 1,
            }),
            CapacityError::File(FsError::Read {
                path: p(),
                source: std::io::Error::other("secret-io-marker"),
            }),
        ];
        for e in &errs {
            let shown = e.to_string();
            assert!(shown.contains("secret"), "{shown}");
            let m = e.public_message();
            assert!(!m.contains("secret"), "{m}");
            assert!(!m.contains("alice"), "{m}");
        }
        assert_eq!(
            CapacityError::EmptyPackage.public_message(),
            "package has no files to measure"
        );
        assert_eq!(
            CapacityError::Overflow.public_message(),
            "package size counters overflow"
        );
    }

    /// REQ-39: 計測で得たサイズが上限を超えたら、合計へ加算する前に TooLarge（20）で拒否する。
    /// オープン後に成長したファイル（事前検査の後で値だけが上限を超える状況）を再現する。
    #[test]
    fn req39_measured_size_over_limit_is_limit_exceeded() {
        use std::io::Write as _;
        let dir = (0..100)
            .find_map(|i| {
                let nanos = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_nanos());
                let p = std::env::temp_dir().join(format!(
                    "fandhe_cap_limit_{}_{nanos}_{i}",
                    std::process::id()
                ));
                std::fs::create_dir(&p).ok().map(|()| p)
            })
            .expect("unique temp dir");
        let path = dir.join("f.bin");
        let mut w = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        w.write_all(b"ab").unwrap();
        let files = [(Weights, path.clone(), std::fs::File::open(&path).unwrap())];
        assert_eq!(
            measure_opened_files_with_limit(&files, 2)
                .unwrap()
                .total_bytes(),
            2
        );
        w.write_all(b"cd").unwrap();
        let e = measure_opened_files_with_limit(&files, 2).unwrap_err();
        assert!(matches!(
            e,
            CapacityError::File(FsError::TooLarge {
                size: 4,
                limit: 2,
                ..
            })
        ));
        assert_eq!(e.exit_code().code(), 20);
        let pf = [PackageFile {
            component: Weights,
            path: path.clone(),
        }];
        assert_eq!(
            measure_package_with_limit(&pf, 3)
                .unwrap_err()
                .exit_code()
                .code(),
            20
        );
        assert_eq!(measure_package_with_limit(&pf, 4).unwrap().total_bytes(), 4);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }
}
