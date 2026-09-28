//! サイズ上限付きの安全な通常ファイル読み込み（REQ-39）。
//!
//! [`definition::Definition::load`](crate::definition::Definition::load)
//! （定義ファイル。REQ-15）と評価器のモデルパッケージ読み込み
//! （REQ-27・`crates/eval` の `invariance` モジュール）は、どちらも
//! 「通常ファイルであることの確認・サイズ上限付きの読み込み・FIFO 等での
//! 無期限停止の回避」という同じ防御を必要とする。以前はこの防御を
//! `definition` モジュールと `fandhe-edge-eval` の両方に個別実装しており、
//! 一方だけを直しても他方が古いままになりうる複製状態だった
//! （issue #214 codex/review 指摘）。本モジュールへ集約し、両者が同じ関数を
//! 呼ぶことで防御を 1 箇所に保つ。
//!
//! # 提供する API
//!
//! - [`open_regular_file_for_read`][]: 通常ファイルであることを検証しつつ開く
//! - [`read_bounded`][]: `limit` バイトまでの上限付きでファイル全体を読み込む
//!   （[`definition::Definition::load`](crate::definition::Definition::load) が使う）
//! - [`sha256_file_bounded`][]: `limit` バイトまでの上限付きでファイルの sha256 を
//!   ストリームで計算する（読み込んだバイト列全体を保持しない。評価器の
//!   モデルパッケージ評価前後比較〔REQ-27〕が使う想定。構成要素 1 件あたりの
//!   バイト列を保持し続けないことで、構成要素数に比例したメモリ消費を避ける）
//!
//! # 防御の内容（呼び出し元で共通）
//!
//! - 通常ファイル判定: 開く前に `std::fs::metadata` で種別を確認し、通常
//!   ファイル以外（FIFO・ソケット・キャラクタデバイス・ディレクトリ等）は
//!   [`FsError::NotRegularFile`] として拒否する
//! - Linux・macOS では `O_NONBLOCK` 付きで開くことで、上の確認から実際に
//!   開くまでの間（TOCTOU）にパスが FIFO へ差し替えられても無期限に
//!   停止しない。開いた後に同一ハンドルで再度種別を確認し、通常ファイル
//!   以外を同じエラーで拒否する（両 OS 以外〔Windows 等〕では `O_NONBLOCK`
//!   相当を持たないが、開く前の事前チェックはどの OS でも効く。Windows は
//!   M10 時点で対象外。coding-rust.md「クロスプラットフォーム」）
//! - サイズ上限: `std::fs::metadata` で報告されたサイズと、実際に読み込んだ
//!   バイト数の両方を `limit` と照合する。メタデータ取得後にファイルが
//!   拡大・差し替えられても、読み込み量自体を上限近傍で頭打ちにする
//!   （`take(limit + 1)` で `limit` 超過を検出する。ストリームハッシュでは
//!   固定長バッファで読み進めながら累計を照合する）
//!
//! 経路の閉じ込め（`../` 等の拒否）・形式の許可リストは、本モジュールの
//! 対象外で呼び出し側のガード層（REQ-39・パス未確定）の責務とする。

use crate::hash::Sha256Digest;
use std::fmt;
use std::fs::{File, Metadata};
use std::io::Read as _;
use std::path::{Path, PathBuf};

/// ストリームでハッシュを計算する際の固定長バッファサイズ。読み込んだ
/// バイト列全体を保持せずに済むよう、この大きさの分だけ確保して使い回す。
const STREAM_BUFFER_BYTES: usize = 64 * 1024;

/// 本モジュールの読み込み関数が返しうるエラー。
///
/// 呼び出し側（[`definition::DefinitionError`](crate::definition::DefinitionError)・
/// `fandhe_edge_eval::invariance::EvaluationInvarianceError`）は、それぞれの
/// エラー型へ写して返す（`#[non_exhaustive]` にして、将来バリアントが増えても
/// 呼び出し側の `match` を壊さないようにする）。
#[derive(Debug)]
#[non_exhaustive]
pub enum FsError {
    /// ファイルを開く・読み込む際の I/O エラー。
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    /// 上限バイト数を超えていた（`size` は報告または実際に読んだバイト数）。
    TooLarge {
        path: PathBuf,
        size: u64,
        limit: u64,
    },
    /// パス先が通常ファイルではない（FIFO・ソケット・キャラクタデバイス・
    /// ディレクトリ等）。これらを許すと `File::open`・読み込みが書き手を
    /// 待って無期限に停止しうる（security.md「ガード層: 資源の上限」）。
    NotRegularFile { path: PathBuf },
}

impl fmt::Display for FsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FsError::Read { path, source } => {
                write!(f, "failed to read {}: {source}", path.display())
            }
            FsError::TooLarge { path, size, limit } => write!(
                f,
                "{} exceeds size limit ({size} > {limit} bytes)",
                path.display()
            ),
            FsError::NotRegularFile { path } => {
                write!(f, "{} is not a regular file", path.display())
            }
        }
    }
}

impl std::error::Error for FsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            FsError::Read { source, .. } => Some(source),
            FsError::TooLarge { .. } | FsError::NotRegularFile { .. } => None,
        }
    }
}

/// FIFO・ソケット・キャラクタデバイス等での無期限停止を避けるため
/// `O_NONBLOCK` 付きで開く（Linux・macOS）。`file_type().is_file()` の検査は
/// 呼び出し元（[`open_regular_file_with_metadata`]）で行う。
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn open_nonblocking(path: &Path) -> std::io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt as _;

    // `O_NONBLOCK` はカーネル ABI の安定値（Linux は全対応アーキテクチャで
    // 8進 0o4000、macOS（BSD 系）は 0x0004）。`libc` 等の新規依存を追加せず
    // （dependency-policy.md）標準ライブラリの `custom_flags` のみで実現する
    // ため、対応 OS を限定してハードコードする。
    #[cfg(target_os = "linux")]
    const O_NONBLOCK: i32 = 0o4000;
    #[cfg(target_os = "macos")]
    const O_NONBLOCK: i32 = 0x0004;

    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(O_NONBLOCK)
        .open(path)
}

/// Linux・macOS 以外（Windows 等）向けのフォールバック。`O_NONBLOCK` 相当の
/// 対策は持たないが、Windows は M10 時点で対象外（coding-rust.md「クロス
/// プラットフォーム」）であり、開く前の事前チェック（[`open_regular_file_with_metadata`]）
/// による通常ファイル以外の拒否は引き続き効く。
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn open_nonblocking(path: &Path) -> std::io::Result<File> {
    std::fs::File::open(path)
}

/// 通常ファイルであることを検証しつつ開き、開いた後に取得したメタデータも
/// 併せて返す（[`read_bounded`]・[`sha256_file_bounded`] がサイズ取得のために
/// 再度 `metadata()` を呼ばずに済むようにするための内部専用ヘルパー）。
///
/// 開く前に `std::fs::metadata` で種別を確認するのは、`O_NONBLOCK` を
/// 持たない OS（Windows 等）でも FIFO 等での無期限停止を防ぐため。
/// Linux・macOS ではさらに [`open_nonblocking`] で `O_NONBLOCK` を付けて
/// 開くことで、事前チェックの後にパスが FIFO へ差し替えられた場合の窓
/// （TOCTOU）でも無期限停止を避け、開いた後に再度種別を確認する。
fn open_regular_file_with_metadata(path: &Path) -> Result<(File, Metadata), FsError> {
    let pre_metadata = std::fs::metadata(path).map_err(|source| FsError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    if !pre_metadata.file_type().is_file() {
        return Err(FsError::NotRegularFile {
            path: path.to_path_buf(),
        });
    }

    let file = open_nonblocking(path).map_err(|source| FsError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    // 事前チェックとオープンの間に差し替えられていないかの TOCTOU 対策
    // （同一の `File` ハンドルへ再度問い合わせる）。
    let metadata = file.metadata().map_err(|source| FsError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    if !metadata.file_type().is_file() {
        return Err(FsError::NotRegularFile {
            path: path.to_path_buf(),
        });
    }
    Ok((file, metadata))
}

/// 通常ファイルであることを検証しつつ開く（REQ-39）。
///
/// FIFO・ソケット・キャラクタデバイス・ディレクトリ等を指すパスは
/// [`FsError::NotRegularFile`] として拒否する。呼び出し側は返ってきた
/// `File` を自前で読み込んでよいが、サイズ上限付きの読み込みが必要な場合は
/// [`read_bounded`]・[`sha256_file_bounded`] を使うこと（本関数はサイズ上限を
/// 検査しない）。
pub fn open_regular_file_for_read(path: &Path) -> Result<File, FsError> {
    let (file, _metadata) = open_regular_file_with_metadata(path)?;
    Ok(file)
}

/// 通常ファイルを `limit` バイトまでの上限付きで読み込む（REQ-39）。
///
/// `std::fs::metadata` でサイズを確認した後に別途読み込む実装は、確認後に
/// ファイルが拡大・差し替えられると上限を超えて無制限にメモリへ読み込みうる
/// （TOCTOU。security.md「ガード層: 資源の上限」）。ここでは 1 つの `File` から
/// メタデータ取得・`take` による打ち切り読み込みまで行い、その間の再オープンを
/// 避けることでこの窓を閉じる。加えて、報告されたサイズと実際に読み込んだ
/// バイト数の両方を `limit` と照合する。
///
/// [`definition::Definition::load`](crate::definition::Definition::load)・
/// `fandhe_edge_eval::invariance` の両方がこの関数を使う
/// （issue #214 codex/review 指摘: 防御ロジックの複製を解消する）。
pub fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>, FsError> {
    let (mut file, metadata) = open_regular_file_with_metadata(path)?;
    let reported_size = metadata.len();
    if reported_size > limit {
        return Err(FsError::TooLarge {
            path: path.to_path_buf(),
            size: reported_size,
            limit,
        });
    }

    // 上限+1バイトまでしか読まない。開いた後にファイルが上限超へ拡大・
    // 差し替えられていても、読み込み量自体を上限近傍で頭打ちにできる。
    let mut buf = Vec::new();
    (&mut file)
        .take(limit.saturating_add(1))
        .read_to_end(&mut buf)
        .map_err(|source| FsError::Read {
            path: path.to_path_buf(),
            source,
        })?;
    let actual_size = buf.len() as u64;
    if actual_size > limit {
        return Err(FsError::TooLarge {
            path: path.to_path_buf(),
            size: actual_size,
            limit,
        });
    }
    Ok(buf)
}

/// 通常ファイルの sha256 を `limit` バイトまでの上限付きでストリーム計算する
/// （REQ-27・REQ-39）。
///
/// [`read_bounded`] と異なり読み込んだバイト列全体を保持しない。固定長
/// バッファ（[`STREAM_BUFFER_BYTES`]）へ読み進めるたびにハッシュへ流し込み、
/// 保持するのは最終的な [`Sha256Digest`]（32 バイト）だけにする。評価器が
/// モデルパッケージの評価前後比較（REQ-27）で構成要素ごとに呼ぶと、
/// 評価前・評価後のスナップショットを同時に保持しても、構成要素の生
/// バイト列（最大で構成要素数 × `limit`）を二重に持たずに済む
/// （issue #214 codex/review 指摘）。
///
/// サイズ上限は、報告されたファイルサイズ（開いた直後のメタデータ）と、
/// 実際に読み進めた累計バイト数の両方で判定する。報告サイズが上限内でも、
/// 読み進める途中で累計が上限を超えた時点（メタデータ取得後にファイルが
/// 拡大・差し替えられた場合）で即座に拒否し、それ以上読み進めない。
pub fn sha256_file_bounded(path: &Path, limit: u64) -> Result<Sha256Digest, FsError> {
    use sha2::{Digest as _, Sha256};

    let (mut file, metadata) = open_regular_file_with_metadata(path)?;
    let reported_size = metadata.len();
    if reported_size > limit {
        return Err(FsError::TooLarge {
            path: path.to_path_buf(),
            size: reported_size,
            limit,
        });
    }

    let mut hasher = Sha256::new();
    let mut buf = [0u8; STREAM_BUFFER_BYTES];
    let mut total: u64 = 0;
    loop {
        let read = file.read(&mut buf).map_err(|source| FsError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        if read == 0 {
            break;
        }
        total = total.saturating_add(read as u64);
        if total > limit {
            return Err(FsError::TooLarge {
                path: path.to_path_buf(),
                size: total,
                limit,
            });
        }
        hasher.update(&buf[..read]);
    }
    Ok(Sha256Digest::from_array(hasher.finalize().into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// テスト用の一時ファイルを、成否に関わらず削除するガード（RAII）。
    struct TempFileGuard(PathBuf);

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
                "fandhe-edge-core-fs-unit-{pid}-{label}-{attempt}-{nanos}"
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

    #[test]
    fn req39_read_bounded_accepts_file_within_limit() {
        let guard = write_unique_temp_file("within-limit", b"hello");
        let bytes = read_bounded(&guard.0, 5).expect("上限内は成功するはず");
        assert_eq!(bytes, b"hello");
    }

    #[test]
    fn req39_read_bounded_rejects_file_over_limit() {
        // REQ-39 の資源上限: 上限を 1 バイトでも超えるファイルは拒否する。
        let guard = write_unique_temp_file("over-limit", b"hello-world");
        let err = read_bounded(&guard.0, 5).unwrap_err();
        match err {
            FsError::TooLarge { size, limit, .. } => {
                assert_eq!(size, 11);
                assert_eq!(limit, 5);
            }
            other => panic!("TooLarge を期待したが {other:?} だった"),
        }
    }

    #[test]
    fn req39_read_bounded_reports_io_error_for_missing_file() {
        let missing = std::env::temp_dir().join("fandhe-edge-core-fs-does-not-exist");
        let err = read_bounded(&missing, 1_024).unwrap_err();
        assert!(matches!(err, FsError::Read { .. }));
    }

    /// REQ-39: ディレクトリを指すパスは通常ファイルではないため、サイズ上限の
    /// 検査に進む前に [`FsError::NotRegularFile`] で拒否される。
    #[test]
    fn req39_read_bounded_rejects_directory() {
        let dir = std::env::temp_dir().join(format!(
            "fandhe-edge-core-fs-unit-{}-dir-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir(&dir).expect("テスト用ディレクトリを作成できるはず");

        let result = read_bounded(&dir, 1_024);
        std::fs::remove_dir(&dir).expect("テスト用ディレクトリを削除できるはず");

        match result.expect_err("ディレクトリは通常ファイルではないため拒否されるはず")
        {
            FsError::NotRegularFile { .. } => {}
            other => panic!("NotRegularFile を期待したが {other:?} だった"),
        }
    }

    /// REQ-39・REQ-27: FIFO（名前付きパイプ）を指すパスを渡しても、書き手が
    /// 現れなくても即座に拒否されること（無期限に停止しない）を確認する。
    /// テストが実際に無期限停止した場合は harness のタイムアウトで検出される。
    /// `open_regular_file_with_metadata` は開く前の `std::fs::metadata` に
    /// よる事前チェックで FIFO を拒否するため、`open(2)` 自体は呼ばれず、
    /// Linux・macOS 限定の `O_NONBLOCK` 経路（TOCTOU 対策の第 2 防御）には
    /// 到達しない。テスト自体は `mkfifo` コマンド（Unix 限定）に依存するため、
    /// 対象 OS を Linux・macOS に絞る。
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn req39_read_bounded_rejects_fifo_without_blocking() {
        let path = std::env::temp_dir().join(format!(
            "fandhe-edge-core-fs-unit-{}-fifo-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        // std に mkfifo 相当の API が無いため、テスト専用に `mkfifo` コマンドで
        // FIFO を作成する（本体コードでは子プロセスを起動しない）。
        let status = std::process::Command::new("mkfifo")
            .arg(&path)
            .status()
            .expect("mkfifo コマンドを起動できるはず");
        assert!(status.success(), "mkfifo が成功するはず");

        // 書き手が存在しない FIFO に対して呼ぶ。無条件の `File::open` は
        // ここでプロセスごと無期限に停止しうる。
        let result = read_bounded(&path, 1_024);
        std::fs::remove_file(&path).expect("FIFO を削除できるはず");

        match result.expect_err("FIFO は通常ファイルではないため拒否されるはず")
        {
            FsError::NotRegularFile { .. } => {}
            other => panic!("NotRegularFile を期待したが {other:?} だった"),
        }
    }

    #[test]
    fn req27_sha256_file_bounded_matches_of_bytes_known_vector() {
        // ストリームで計算したハッシュが `Sha256Digest::of_bytes` と一致する
        // ことを、独立に検証可能な具体値（NIST 標準の sha256 "abc"）で確かめる
        // （issue #214 codex/review 指摘: ストリーム計算経路自体の正しさを
        // 既存のメモリ上バイト列ハッシュと同じ既知ベクタで確認する）。
        let guard = write_unique_temp_file("stream-hash-abc", b"abc");
        let digest = sha256_file_bounded(&guard.0, 1_024).expect("上限内は成功するはず");
        assert_eq!(
            digest.to_hex(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(digest, Sha256Digest::of_bytes(b"abc"));
    }

    #[test]
    fn req27_sha256_file_bounded_matches_of_bytes_for_content_larger_than_stream_buffer() {
        // ストリームバッファ（64KiB）を跨ぐ複数回の read でも、1 回で
        // ハッシュ化した場合と一致することを確認する（バッファ境界の
        // 誤りがあれば検出できる）。
        let content = vec![0xabu8; STREAM_BUFFER_BYTES * 3 + 17];
        let guard = write_unique_temp_file("stream-hash-multi-buffer", &content);
        let digest =
            sha256_file_bounded(&guard.0, content.len() as u64).expect("上限内は成功するはず");
        assert_eq!(digest, Sha256Digest::of_bytes(&content));
    }

    #[test]
    fn req39_sha256_file_bounded_rejects_file_over_limit() {
        let guard = write_unique_temp_file("stream-hash-over-limit", b"hello-world");
        let err = sha256_file_bounded(&guard.0, 5).unwrap_err();
        match err {
            FsError::TooLarge { size, limit, .. } => {
                assert_eq!(size, 11);
                assert_eq!(limit, 5);
            }
            other => panic!("TooLarge を期待したが {other:?} だった"),
        }
    }

    #[test]
    fn req39_sha256_file_bounded_rejects_directory() {
        let dir = std::env::temp_dir().join(format!(
            "fandhe-edge-core-fs-unit-{}-hash-dir-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir(&dir).expect("テスト用ディレクトリを作成できるはず");

        let result = sha256_file_bounded(&dir, 1_024);
        std::fs::remove_dir(&dir).expect("テスト用ディレクトリを削除できるはず");

        match result.expect_err("ディレクトリは通常ファイルではないため拒否されるはず")
        {
            FsError::NotRegularFile { .. } => {}
            other => panic!("NotRegularFile を期待したが {other:?} だった"),
        }
    }

    #[test]
    fn req39_open_regular_file_for_read_rejects_directory() {
        let dir = std::env::temp_dir().join(format!(
            "fandhe-edge-core-fs-unit-{}-open-dir-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir(&dir).expect("テスト用ディレクトリを作成できるはず");

        let result = open_regular_file_for_read(&dir);
        std::fs::remove_dir(&dir).expect("テスト用ディレクトリを削除できるはず");

        match result.expect_err("ディレクトリは通常ファイルではないため拒否されるはず")
        {
            FsError::NotRegularFile { .. } => {}
            other => panic!("NotRegularFile を期待したが {other:?} だった"),
        }
    }

    #[test]
    fn req39_open_regular_file_for_read_accepts_regular_file() {
        let guard = write_unique_temp_file("open-regular", b"hello");
        let mut file = open_regular_file_for_read(&guard.0).expect("通常ファイルは成功するはず");
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)
            .expect("読み込みに失敗しないはず");
        assert_eq!(buf, b"hello");
    }
}
