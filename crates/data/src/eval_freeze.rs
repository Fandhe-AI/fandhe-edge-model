//! 評価データの凍結（REQ-17）。
//!
//! `evaluate` 工程（CLI 側の実体は TASK-33.1 未着手）が判定に評価データを使う前に、
//! そのファイルのハッシュを記録して凍結する処理を提供する。評価データが与えられて
//! いない場合は I/O を一切行わず [`fandhe_edge_core::eval_data::EvalDataStatus::NotProvided`]
//! を返し、`evaluate` が `status:"skipped"`・exit 0 で完走できるようにする
//! （PoC-16 縦断 2 で実測した挙動・評価契約「評価データが無い場合、`evaluate` は
//! `status:"skipped"`・exit 0 とし、評価済みを装わない」に対応する境界）。
//!
//! # 経路の閉じ込め（REQ-39・security.md「経路の閉じ込め」）
//!
//! [`freeze_eval_data`] は許可ルート（`root`）を受け取り、そのルート配下だけを
//! 開く。絶対パス・`..`（親ディレクトリ参照）は正規化前に拒否し、正規化
//! （`canonicalize`）後の実体パスがルート配下であることまで確認してから開くため、
//! ルート外の実体を指す symlink も拒否する（`safe_join` 相当）。
//!
//! # 対象外（本 issue のスコープ外）
//!
//! - ハッシュ不一致時の停止判定（TASK-17.3）。本関数は記録するのみで、既存の
//!   記録値との突き合わせ・検証は行わない
//! - 読み取り専用配置（書き込み拒否）への変更（おそらく TASK-17.2-2）。本関数は
//!   ファイルの権限を一切変更しない

use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

use fandhe_edge_core::eval_data::{EvalDataStatus, FreezeRecord};
use fandhe_edge_core::hash::sha256_hex_of_reader;

use crate::limits::MAX_EVAL_DATA_BYTES;

/// 評価データの凍結に失敗した理由（外部入力の経路のため panic せず enum で返す）。
#[derive(Debug)]
pub enum FreezeError {
    /// 指定されたパスが存在しない、または読み込めない（`io::ErrorKind::NotFound` 等）。
    NotFound(io::Error),
    /// ファイルサイズが上限（[`MAX_EVAL_DATA_BYTES`]）を超えている。
    /// REQ-39「読み込み前にサイズを確認する」ため、`File::open` の前に検出して拒否する。
    /// `metadata()` 取得後にファイルが成長する TOCTOU に備え、実読み込み量が上限を
    /// 超えた場合（後述の `.take()` 経由）にもこのバリアントで拒否する。
    TooLarge { limit: u64, actual: u64 },
    /// 通常ファイルではないパス（キャラクタデバイス・FIFO・ディレクトリ・symlink の
    /// 解決先等）が指定された。`/dev/zero` のようなキャラクタデバイスは `stat` 上の
    /// サイズが 0 のため [`TooLarge`](FreezeError::TooLarge) 検査を素通りし、
    /// ストリーミング読み込みが EOF に到達せず無限に読み続けてしまう
    /// （REQ-39「無制限の…無限待ちを作らない」）ため、サイズ検証の前に拒否する。
    NotAFile,
    /// `path` が許可ルート（`root`）の外を指している（絶対パス・`..` による脱出、
    /// または正規化後の実体がルート外にある symlink 等）。REQ-39「経路の閉じ込め」
    /// （security.md）に基づき、ファイルを開く前に拒否する。
    OutsideRoot,
    /// 上記以外の I/O エラー（メタデータ取得・読み込み中のエラー等）。
    Io(io::Error),
}

impl std::fmt::Display for FreezeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FreezeError::NotFound(err) => write!(f, "evaluation data not found: {err}"),
            FreezeError::TooLarge { limit, actual } => write!(
                f,
                "evaluation data exceeds size limit: limit={limit} actual={actual}"
            ),
            FreezeError::NotAFile => {
                write!(f, "evaluation data path is not a regular file")
            }
            FreezeError::OutsideRoot => {
                write!(f, "evaluation data path escapes the allowed root")
            }
            FreezeError::Io(err) => write!(f, "evaluation data I/O error: {err}"),
        }
    }
}

impl std::error::Error for FreezeError {}

/// 評価データを凍結する。
///
/// - `path` が `None` の場合: I/O を一切行わず `Ok(EvalDataStatus::NotProvided)` を返す。
/// - `path` が `Some` の場合: まず `root` 配下への経路の閉じ込めを検証し
///   （絶対パス・`..` を正規化前に拒否、正規化後の実体パスが `root` 配下である
///   ことを確認。REQ-39「経路の閉じ込め」）、その後に通常ファイルであること・
///   ファイルサイズを確認してから（非通常ファイル・上限超過ならファイルを
///   開かずに拒否）、ストリーミングで sha256 を計算し `EvalDataStatus::Frozen` を
///   返す。読み込みは `MAX_EVAL_DATA_BYTES + 1` バイトで打ち切り、`metadata()`
///   取得後にファイルが成長する TOCTOU（実読み込み量が事前チェックしたサイズを
///   上回るケース）も検出して拒否する（REQ-39「無制限のアロケーション・
///   無限待ちを作らない」）。
///
/// `root` はあらかじめ存在するディレクトリであること（呼び出し元が確定させた
/// 評価データの置き場。CLI 側配線時にどのディレクトリを渡すかは TASK-33.1 で
/// 決める）。`path` は `root` からの相対パスとして扱う。
///
/// ハッシュ計算対象のファイル内容（データ本文）は返り値・エラーメッセージに含めない
/// （security.md「秘密情報の混入防止」「機微情報の露出」。学習・評価データに
/// 個人情報が含まれうる前提のため）。
pub fn freeze_eval_data(root: &Path, path: Option<&Path>) -> Result<EvalDataStatus, FreezeError> {
    let Some(path) = path else {
        return Ok(EvalDataStatus::NotProvided);
    };

    let canonical_path = resolve_within_root(root, path)?;
    let path = canonical_path.as_path();

    let metadata = std::fs::metadata(path).map_err(|err| {
        if err.kind() == io::ErrorKind::NotFound {
            FreezeError::NotFound(err)
        } else {
            FreezeError::Io(err)
        }
    })?;
    // 通常ファイル以外（キャラクタデバイス・FIFO・ディレクトリ等）は、`stat` 上の
    // サイズがハッシュ対象の実データ量と無関係（`/dev/zero` は 0 バイトだが読むと
    // 終端しない）なため、サイズ検査より先に拒否する。この事前チェックは高速な
    // 門前払い用であり、下の `open_without_blocking` の後段で fd を取り直して
    // 再検証するまでがセキュリティ境界（TOCTOU 対策）である。
    if !metadata.is_file() {
        return Err(FreezeError::NotAFile);
    }
    let byte_len = metadata.len();
    if byte_len > MAX_EVAL_DATA_BYTES {
        return Err(FreezeError::TooLarge {
            limit: MAX_EVAL_DATA_BYTES,
            actual: byte_len,
        });
    }

    // `metadata(path)` から open までの間にパスが FIFO 等へ差し替えられる
    // TOCTOU に備え、Unix ではノンブロッキングで開く（書き手の無い FIFO の
    // open(2) で無限にブロックしない。REQ-39「無制限の…無限待ちを作らない」）。
    let file = open_without_blocking(path)?;
    // 開いた fd 自体を fstat で再検証する（path ではなく file descriptor の種別・
    // サイズを見るため、ここまでの間の差し替えを確実に検出できる）。
    let fd_metadata = file.metadata().map_err(FreezeError::Io)?;
    if !fd_metadata.is_file() {
        return Err(FreezeError::NotAFile);
    }
    let fd_byte_len = fd_metadata.len();
    if fd_byte_len > MAX_EVAL_DATA_BYTES {
        return Err(FreezeError::TooLarge {
            limit: MAX_EVAL_DATA_BYTES,
            actual: fd_byte_len,
        });
    }
    // 実読み込み量にも上限を課す（`.take(limit + 1)`）。fstat 後にファイルが
    // 成長した場合でも、上限超過分を読み進めた時点で確実に止まる。
    let read_limit = MAX_EVAL_DATA_BYTES.saturating_add(1);
    let mut counting = CountingReader::new(io::BufReader::new(file).take(read_limit));
    let sha256 = sha256_hex_of_reader(&mut counting).map_err(FreezeError::Io)?;
    let actual_read = counting.count();
    if actual_read > MAX_EVAL_DATA_BYTES {
        return Err(FreezeError::TooLarge {
            limit: MAX_EVAL_DATA_BYTES,
            actual: actual_read,
        });
    }

    Ok(EvalDataStatus::Frozen(FreezeRecord {
        // 呼び出し元が渡した相対パスではなく、経路の閉じ込め検証で確定させた
        // 正規化後の実体パスを記録する。symlink 解決前の表記より、実際に
        // ハッシュ対象にしたファイルの実体を一意に指す値の方が
        // 記録整合性（REQ-17）にかなうため（`byte_len` を stat 値ではなく
        // 実読み込み量にする判断と同じ理由）。
        path: path.to_path_buf(),
        sha256,
        // `metadata.len()` ではなく実際に読み込んだ（＝ハッシュした）バイト数を
        // 記録する。TOCTOU でファイルが縮んだ場合でも記録値とハッシュ対象が
        // 食い違わないようにするため（REQ-17 の記録整合性）。
        byte_len: actual_read,
    }))
}

/// `path`（`root` からの相対パス）が `root` 配下に閉じ込められていることを
/// 検証し、正規化後の実体パスを返す（REQ-39「経路の閉じ込め」・security.md
/// 「`safe_join` 相当の検証（正規化後にルート配下であることの確認）」）。
///
/// 二段階で検証する:
/// 1. 構文検証（ファイルシステムに触れる前）: `path` が絶対パスである場合、
///    または `..`（親ディレクトリ参照）を含む場合は拒否する。`root.join(path)`
///    は `path` が絶対パスだと `root` を無視して `path` そのものを返してしまう
///    ため、`join` の前に弾く。
/// 2. 意味検証（`canonicalize` 後）: `root` と `root.join(path)` の双方を
///    正規化し、後者が前者の配下であることを確認する。symlink はここで解決
///    されるため、`path` 自体は `..` を含まなくても、その実体（symlink の
///    解決先）がルート外にある場合はここで拒否できる。`root` 自体を
///    正規化するのは、`root` 自体が symlink（例: macOS の `/tmp`）の場合に
///    `starts_with` の比較が正しく機能するようにするため。
fn resolve_within_root(root: &Path, path: &Path) -> Result<std::path::PathBuf, FreezeError> {
    if path.is_absolute() {
        return Err(FreezeError::OutsideRoot);
    }
    for component in path.components() {
        match component {
            std::path::Component::Normal(_) | std::path::Component::CurDir => {}
            std::path::Component::ParentDir
            | std::path::Component::RootDir
            | std::path::Component::Prefix(_) => {
                return Err(FreezeError::OutsideRoot);
            }
        }
    }

    let joined = root.join(path);

    let canonical_root = canonicalize_for_confinement(root)?;
    let canonical_path = canonicalize_for_confinement(&joined)?;

    if !canonical_path.starts_with(&canonical_root) {
        return Err(FreezeError::OutsideRoot);
    }

    Ok(canonical_path)
}

/// [`resolve_within_root`] 用に `canonicalize` を呼び、`NotFound` を
/// [`FreezeError::NotFound`] として区別する（他の I/O エラーと違い、
/// 「評価データが見つからない」という既存の呼び出し元向けの意味を保つため）。
fn canonicalize_for_confinement(path: &Path) -> Result<std::path::PathBuf, FreezeError> {
    std::fs::canonicalize(path).map_err(|err| {
        if err.kind() == io::ErrorKind::NotFound {
            FreezeError::NotFound(err)
        } else {
            FreezeError::Io(err)
        }
    })
}

/// 評価データのパスを通常ファイルとして安全に開く。
///
/// `std::fs::metadata(path)` によるファイル種別チェックと `File::open(path)` は
/// 別々の操作であり、その間にパスが FIFO 等へ差し替えられると
/// （TOCTOU）、単純な `File::open` は書き手が現れるまで無期限にブロックし得る
/// （REQ-39「無制限の…無限待ちを作らない」への抵触）。Unix では `O_NONBLOCK` 付きで
/// 開くことでこのブロックを避ける。通常ファイルに対する `O_NONBLOCK` は読み取り
/// 挙動に影響しない（POSIX の規定）ため、正常系の動作は変わらない。
///
/// 値を `libc` クレートに頼らず OS ごとに直書きしているのは、新規依存の追加が
/// ユーザー承認事項（[dependency-policy](../../../.claude/rules/dependency-policy.md)）
/// であり、この 1 箇所のためだけに依存を増やさない判断による。
///
/// 呼び出し元（[`freeze_eval_data`]）はこの後さらに開いた fd 自体を fstat で
/// 再検証するため、ここで返す `File` の種別・サイズはまだ信頼しない。
#[cfg(unix)]
fn open_without_blocking(path: &Path) -> Result<File, FreezeError> {
    use std::os::unix::fs::OpenOptionsExt;

    #[cfg(any(target_os = "linux", target_os = "android"))]
    const O_NONBLOCK: i32 = 0o4000;
    #[cfg(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "freebsd",
        target_os = "dragonfly",
        target_os = "netbsd",
        target_os = "openbsd"
    ))]
    const O_NONBLOCK: i32 = 0x0004;

    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(O_NONBLOCK)
        .open(path)
        .map_err(|err| {
            if err.kind() == io::ErrorKind::NotFound {
                FreezeError::NotFound(err)
            } else {
                FreezeError::Io(err)
            }
        })
}

/// 評価データのパスを開く（Unix 以外）。
///
/// FIFO による無限ブロックは `mkfifo` が使える Unix 系環境に固有の攻撃経路のため、
/// それ以外の環境では通常の `File::open` で足りる（呼び出し元が fd を fstat で
/// 再検証する点は Unix 版と共通）。
#[cfg(not(unix))]
fn open_without_blocking(path: &Path) -> Result<File, FreezeError> {
    File::open(path).map_err(|err| {
        if err.kind() == io::ErrorKind::NotFound {
            FreezeError::NotFound(err)
        } else {
            FreezeError::Io(err)
        }
    })
}

/// 読み込んだバイト数を数えながら委譲する [`Read`] ラッパー。
///
/// `.take(limit)` と組み合わせ、上限ちょうどまで読ませた実バイト数を
/// 呼び出し元が事後に検査できるようにするために使う（このモジュールの
/// TOCTOU 対策専用の内部実装）。
struct CountingReader<R> {
    inner: R,
    count: u64,
}

impl<R: Read> CountingReader<R> {
    fn new(inner: R) -> Self {
        Self { inner, count: 0 }
    }

    fn count(&self) -> u64 {
        self.count
    }
}

impl<R: Read> Read for CountingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let read_len = self.inner.read(buf)?;
        // `usize` から `u64` への変換は 32bit 環境でも失敗しないが、REQ-39 系の
        // 外部入力パスでは `unwrap` / `as` の桁あふれ想定を避ける方針のため
        // `try_from` で明示し、万一の変換失敗時は `count` を `u64::MAX` に
        // 飽和させて `TooLarge` 判定側へ確実に倒す（panic させない）。
        let read_len_u64 = u64::try_from(read_len).unwrap_or(u64::MAX);
        self.count = self.count.saturating_add(read_len_u64);
        Ok(read_len)
    }
}
