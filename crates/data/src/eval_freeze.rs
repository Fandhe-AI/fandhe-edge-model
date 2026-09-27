//! 評価データの凍結（REQ-17）。
//!
//! `evaluate` 工程（CLI 側の実体は TASK-33.1 未着手）が判定に評価データを使う前に、
//! そのファイルのハッシュを記録して凍結する処理を提供する。評価データが与えられて
//! いない場合は I/O を一切行わず [`fandhe_edge_core::eval_data::EvalDataStatus::NotProvided`]
//! を返し、`evaluate` が `status:"skipped"`・exit 0 で完走できるようにする
//! （PoC-16 縦断 2 で実測した挙動・評価契約「評価データが無い場合、`evaluate` は
//! `status:"skipped"`・exit 0 とし、評価済みを装わない」に対応する境界）。
//!
//! # 対象外（本 issue のスコープ外）
//!
//! - ハッシュ不一致時の停止判定（TASK-17.3）。本関数は記録するのみで、既存の
//!   記録値との突き合わせ・検証は行わない
//! - 読み取り専用配置（書き込み拒否）への変更（おそらく TASK-17.2-2）。本関数は
//!   ファイルの権限を一切変更しない
//! - パストラバーサル対策（経路の閉じ込め）。呼び出し元が既に確定させた単一パスを
//!   受け取る前提とし、ガード層（REQ-39）相当の中途半端な検証をここでは行わない

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
            FreezeError::Io(err) => write!(f, "evaluation data I/O error: {err}"),
        }
    }
}

impl std::error::Error for FreezeError {}

/// 評価データを凍結する。
///
/// - `path` が `None` の場合: I/O を一切行わず `Ok(EvalDataStatus::NotProvided)` を返す。
/// - `path` が `Some` の場合: 通常ファイルであること・ファイルサイズを確認してから
///   （非通常ファイル・上限超過ならファイルを開かずに拒否）、ストリーミングで
///   sha256 を計算し `EvalDataStatus::Frozen` を返す。読み込みは
///   `MAX_EVAL_DATA_BYTES + 1` バイトで打ち切り、`metadata()` 取得後にファイルが
///   成長する TOCTOU（実読み込み量が事前チェックしたサイズを上回るケース）も
///   検出して拒否する（REQ-39「無制限のアロケーション・無限待ちを作らない」）。
///
/// ハッシュ計算対象のファイル内容（データ本文）は返り値・エラーメッセージに含めない
/// （security.md「秘密情報の混入防止」「機微情報の露出」。学習・評価データに
/// 個人情報が含まれうる前提のため）。
pub fn freeze_eval_data(path: Option<&Path>) -> Result<EvalDataStatus, FreezeError> {
    let Some(path) = path else {
        return Ok(EvalDataStatus::NotProvided);
    };

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
        path: path.to_path_buf(),
        sha256,
        // `metadata.len()` ではなく実際に読み込んだ（＝ハッシュした）バイト数を
        // 記録する。TOCTOU でファイルが縮んだ場合でも記録値とハッシュ対象が
        // 食い違わないようにするため（REQ-17 の記録整合性）。
        byte_len: actual_read,
    }))
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
