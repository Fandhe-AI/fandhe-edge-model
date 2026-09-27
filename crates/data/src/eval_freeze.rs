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
//! 開く。絶対パス・`..`（親ディレクトリ参照）は構文検証（ファイルシステムに
//! 触れる前）で拒否し、`path` は `root` 直下の 1 コンポーネント（`root` の
//! 直接の子）に限定する（[`validate_within_root`] で強制する）。
//!
//! **`canonicalize` による事前解決は行わない。** 以前の実装は
//! `canonicalize(root.join(path))` で実体パスを解決してから、その実体パスを
//! `open` していた。しかしこの 2 段階（`canonicalize` → `open`）の間に、
//! `path` の実体解決がまたがる中間ディレクトリ（`path` 自体が `root` 配下の
//! ネストしたファイルへの symlink である場合、その解決先の親ディレクトリ）を
//! ルート外への symlink に差し替えられると、最終コンポーネントにしか効かない
//! `O_NOFOLLOW` ではこのすり替えを検出できず、ルート外のファイルを読み取れて
//! しまう（TOCTOU。codex レビュー指摘 PRRT_kwDOUq-SxM6mb5Yq・REQ-39 P0）。
//!
//! この窓を作らないため、[`freeze_eval_data`] は `validate_within_root` が
//! 返す未加工の結合パス（`root.join(path)`。`canonicalize` を挟まない）を
//! **1 回の `open`** にそのまま渡す。安定版 Rust の `std` には
//! `openat`/`fstatat` 相当が無く、`libc` 等の依存追加も `unsafe extern "C"` の
//! FFI 追加もユーザー承認事項（[dependency-policy](../../../.claude/rules/dependency-policy.md)・
//! [coding-rust](../../../.claude/rules/coding-rust.md)「unsafe・FFI」）のため、
//! 代わりに検証対象の可変要素を「`root` 直下の 1 エントリ」だけに縮退させ、
//! `path` にネストしたコンポーネントを許さないことで、途中に差し替え可能な
//! ディレクトリが一切存在しない状態を作る。
//!
//! - **Unix**: この 1 回の `open` に `O_NOFOLLOW` を付けて（[`open_without_blocking`]）、
//!   最終エントリ自体が symlink であれば（`root` 内外どちらを指していても）
//!   拒否し、開いた fd 自体を `fstat` で再検証する。検証と読み込みが
//!   1 syscall に結合されるため、TOCTOU の窓は生まれない。
//! - **Unix 以外（`O_NOFOLLOW` 相当が無い環境）**: [`confine_within_root_by_parent`]
//!   で `canonicalize` 後の実体パスの親が `root` そのものであることを確認して
//!   から `open` する。`canonicalize` とその直後の `open` の間に最終エントリを
//!   差し替えられる TOCTOU の窓が残る（`.claude/rules/coding-rust.md`
//!   「クロスプラットフォーム」: 検証環境は Mac のみで Windows/Linux は
//!   M10 時点で対象外。この窓は本対応が導入される以前から Unix 以外に存在した
//!   防御水準と同等で、後退させないための最小対応にとどめる）。
//!
//! いずれの経路も `path` が `root` 直下の生のファイルでない限り
//! （＝symlink である限り、または実体の親が `root` でない限り）成功しない。
//! 記録用の正規化パス（`FreezeRecord` の `path`）は、ハッシュ対象を確定させた
//! **後**に `canonicalize` して求める（この時点の `canonicalize` は記録の
//! 一意性のためだけで、経路の閉じ込め判定には使わない）。`root` 配下に
//! ネストしたパスは受け付けない（将来 openat 経由の実装に切り替える際に承認を
//! 得て拡張する）。
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
    /// REQ-39「読み込み前にサイズを確認する」ため、開いた fd の `fstat` で
    /// 実データを読む前に検出して拒否する（パスへの `metadata()` 事前チェックは
    /// TOCTOU の窓を広げるため廃止し、fd 基準の 1 回の検証に統合した）。
    /// この fstat 後にファイルが成長する TOCTOU に備え、実読み込み量が上限を
    /// 超えた場合（後述の `.take()` 経由）にもこのバリアントで拒否する。
    TooLarge { limit: u64, actual: u64 },
    /// 通常ファイルではないパス（キャラクタデバイス・FIFO・ディレクトリ・symlink の
    /// 解決先等）が指定された。`/dev/zero` のようなキャラクタデバイスは `stat` 上の
    /// サイズが 0 のため [`TooLarge`](FreezeError::TooLarge) 検査を素通りし、
    /// ストリーミング読み込みが EOF に到達せず無限に読み続けてしまう
    /// （REQ-39「無制限の…無限待ちを作らない」）ため、開いた fd の種別検証
    /// （`fstat`）をサイズ検証より前に行う。
    NotAFile,
    /// `path` が許可ルート（`root`）の外を指している（絶対パス・`..` による脱出、
    /// `root` 直下の 1 コンポーネントに収まらないネストしたパス、正規化後の実体が
    /// ルート外にある symlink、または `open` 時点で symlink に差し替えられていた
    /// 場合〔`O_NOFOLLOW` の `ELOOP`〕）。REQ-39「経路の閉じ込め」（security.md）に
    /// 基づき、ファイルを開く前後で拒否する。
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
/// - `path` が `Some` の場合: まず `root` 配下への経路の閉じ込めを構文検証する
///   （絶対パス・`..` を拒否、`root` 直下の 1 コンポーネントに限定。
///   [`validate_within_root`]。REQ-39「経路の閉じ込め」）。この検証は
///   ファイルシステムに触れないため、ここまでの間に TOCTOU の窓は生まれない。
///   検証後は、`validate_within_root` が返す未加工の結合パス（`canonicalize`
///   を経由しない）を渡す `open` 1 回（Unix では `O_NOFOLLOW` 付き）で開いた
///   fd に対する `fstat` だけを信頼して通常ファイルであること・ファイルサイズを
///   確認し（非通常ファイル・上限超過ならデータを読まずに拒否）、その後
///   ストリーミングで sha256 を計算し `EvalDataStatus::Frozen` を返す。
///   `canonicalize` による事前解決を挟まないのは、解決後の実体パスを別途
///   `open` し直す 2 段階構成が、その間に中間ディレクトリを symlink へ
///   差し替えられる TOCTOU を生むため（モジュール冒頭「経路の閉じ込め」参照。
///   codex レビュー指摘 PRRT_kwDOUq-SxM6mb5Yq への対応）。読み込みは
///   `MAX_EVAL_DATA_BYTES + 1` バイトで打ち切り、`fstat` 後にファイルが
///   成長する TOCTOU（実読み込み量が事前チェックしたサイズを上回るケース）も
///   検出して拒否する（REQ-39「無制限のアロケーション・無限待ちを作らない」）。
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

    // 構文検証のみ（ファイルシステムに触れない）。返り値は `canonicalize` を
    // 経由しない未加工の結合パスで、これを直接 `open` に渡す（TOCTOU 対策。
    // モジュール冒頭「経路の閉じ込め」参照）。
    let joined = validate_within_root(root, path)?;

    // Windows 等（Unix 以外）には `O_NOFOLLOW` 相当が無いため、`open` の
    // `O_NOFOLLOW` 1 syscall だけでは経路の閉じ込めを完結できない。代わりに
    // `canonicalize` 後の実体パスの親が `root` そのものであることを確認する
    // （`starts_with` ではなく `parent()` 比較にすることで、`path` が `root`
    // 配下のネストしたファイルへの symlink だった場合も拒否し、中間ディレクトリを
    // 経由する経路自体を無くす）。この確認と直後の `open` の間には、なお
    // 最終エントリを差し替えられる TOCTOU の窓が残るが、これは本 PR 以前から
    // 変わらない Unix 以外の既存の防御水準であり、後退させない
    // （`open_without_blocking`〔Unix 以外〕のドキュメント参照）。
    #[cfg(not(unix))]
    confine_within_root_by_parent(root, &joined)?;

    // `validate_within_root` が返した未加工パスを、パス文字列で何度も
    // 解決し直さず 1 回の `open` に直結させる（Unix では `O_NONBLOCK` で FIFO の
    // 無限ブロックを、`O_NOFOLLOW` で最終エントリが symlink（`root` 内外いずれを
    // 指す場合も）であるケースを検出する。REQ-39「無制限の…無限待ちを作らない」
    // 「経路の閉じ込め」）。
    let file = open_without_blocking(&joined)?;
    // 開いた fd 自体を fstat で再検証する（path ではなく file descriptor の種別・
    // サイズを見るため、`open` が成功した時点の実体を確実に検証できる）。
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

    // 記録用の正規化後の実体パスは、ハッシュ対象を確定させた**後**に求める。
    // ここでの `canonicalize` は記録の一意性（REQ-17）のためだけであり、経路の
    // 閉じ込め判定にはもう使わない（判定は `open` 1 回で完結済み）ため、
    // ここでの path 再解決が TOCTOU の窓を広げることはない。
    let canonical_path = canonicalize_for_record(&joined)?;

    Ok(EvalDataStatus::Frozen(FreezeRecord {
        // 呼び出し元が渡した相対パスではなく、正規化後の実体パスを記録する。
        // symlink 解決前の表記より、実際にハッシュ対象にしたファイルの実体を
        // 一意に指す値の方が記録整合性（REQ-17）にかなうため（`byte_len` を
        // stat 値ではなく実読み込み量にする判断と同じ理由）。
        path: canonical_path,
        sha256,
        // `metadata.len()` ではなく実際に読み込んだ（＝ハッシュした）バイト数を
        // 記録する。TOCTOU でファイルが縮んだ場合でも記録値とハッシュ対象が
        // 食い違わないようにするため（REQ-17 の記録整合性）。
        byte_len: actual_read,
    }))
}

/// `path`（`root` からの相対パス）が `root` 配下に閉じ込められた構文であることを
/// 検証し、`canonicalize` を経由しない未加工の結合パス（`root.join(path)`）を
/// 返す（REQ-39「経路の閉じ込め」・security.md「`safe_join` 相当の検証」）。
///
/// 二段階で検証する（いずれもファイルシステムに触れない構文検証のみ）:
/// 1. `path` が絶対パスである場合、または `..`（親ディレクトリ参照）を含む
///    場合は拒否する。`root.join(path)` は `path` が絶対パスだと `root` を
///    無視して `path` そのものを返してしまうため、`join` の前に弾く。
/// 2. `path` の実体的なコンポーネント（`Normal`）がちょうど 1 個であることを
///    要求し、`root` 直下の直接の子以外（ネストしたパス）を拒否する。
///
/// 意図的に `canonicalize` による意味検証（実体パスの解決）は行わない。
/// 呼び出し元（[`freeze_eval_data`]）はこの関数が返す未加工パスをそのまま
/// 1 回の `open`（Unix では `O_NOFOLLOW` 付き）に渡し、その `open` 自体が
/// 「最終エントリが symlink なら拒否する」形で経路の閉じ込めを完結させる。
/// `path` を `root` 直下の 1 コンポーネントへ制限しているため、この `open` が
/// 辿るファイルシステム上の可変要素は「`root` 直下の 1 エントリ」だけであり、
/// 差し替え可能な中間ディレクトリが経路上に存在しない。もし `canonicalize` で
/// 事前に実体パスを解決し、その解決後のパスを別途 `open` し直す 2 段階構成に
/// すると、解決からその `open` までの間に（`path` が `root` 配下のネストした
/// ファイルへの symlink だった場合の）解決先の親ディレクトリを symlink へ
/// 差し替えられる TOCTOU が生まれる（`O_NOFOLLOW` は最終コンポーネントにしか
/// 効かないため検出できない。codex レビュー指摘 PRRT_kwDOUq-SxM6mb5Yq・
/// REQ-39 P0 への対応。モジュール冒頭「経路の閉じ込め」参照）。
fn validate_within_root(root: &Path, path: &Path) -> Result<std::path::PathBuf, FreezeError> {
    if path.is_absolute() {
        return Err(FreezeError::OutsideRoot);
    }
    let mut normal_components = 0u32;
    for component in path.components() {
        match component {
            std::path::Component::Normal(_) => {
                normal_components += 1;
            }
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir
            | std::path::Component::RootDir
            | std::path::Component::Prefix(_) => {
                return Err(FreezeError::OutsideRoot);
            }
        }
    }
    if normal_components != 1 {
        return Err(FreezeError::OutsideRoot);
    }

    Ok(root.join(path))
}

/// Unix 以外（`O_NOFOLLOW` 相当が無い環境）向けの意味検証: `joined`
/// （`root.join(path)`）を `canonicalize` した実体パスの親ディレクトリが、
/// 同じく `canonicalize` した `root` そのものであることを確認する
/// （REQ-39「経路の閉じ込め」）。
///
/// `starts_with` ではなく `parent()` の一致で比較するのは、`path` が `root`
/// 配下のネストしたファイルへの symlink であるケース（実体パスは `root`
/// 配下だが、その親ディレクトリが `root` 自身ではない）まで拒否するため。
/// これにより、この関数が通過した後に `open` が辿る経路は「`canonicalize`
/// した `root`」＋「1 つの最終エントリ」だけに絞られ、`open` に至るまでの
/// 間に差し替えられ得る中間ディレクトリを経路から無くす。
///
/// Unix では `open_without_blocking` の `O_NOFOLLOW` 1 syscall で経路の閉じ込め
/// 検証と読み込みを結合できる（この関数は呼ばない）が、Unix 以外にはその手段が
/// 無いため、この `canonicalize` と直後の `open` の間に最終エントリを
/// 差し替えられる TOCTOU の窓が残る。これは本モジュールの TOCTOU 対策（codex
/// レビュー指摘 PRRT_kwDOUq-SxM6mb5Yq）が導入される以前から Unix 以外に
/// 存在していた防御水準と同等であり、後退させないための最小対応
/// （`.claude/rules/coding-rust.md`「クロスプラットフォーム」: 検証環境は
/// Mac のみで Windows/Linux は M10 時点で対象外。完全な close は openat 相当の
/// 実装〔依存追加 or `unsafe extern "C"` FFI〕をユーザー承認のうえ拡張する）。
#[cfg(not(unix))]
fn confine_within_root_by_parent(root: &Path, joined: &Path) -> Result<(), FreezeError> {
    let canonical_root = canonicalize_for_record(root)?;
    let canonical_joined = canonicalize_for_record(joined)?;
    if canonical_joined.parent() != Some(canonical_root.as_path()) {
        return Err(FreezeError::OutsideRoot);
    }
    Ok(())
}

/// ハッシュ対象を確定させた**後**に、記録用の正規化済み実体パスを求める
/// （REQ-17 の記録整合性のためだけに使う。経路の閉じ込め判定はすでに `open`
/// 1 回で完結しているため、ここでの `canonicalize` が TOCTOU の窓を広げる
/// ことはない）。`NotFound` を [`FreezeError::NotFound`] として区別する。
fn canonicalize_for_record(path: &Path) -> Result<std::path::PathBuf, FreezeError> {
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
/// `O_NOFOLLOW` も付ける。呼び出し元（[`freeze_eval_data`]）は
/// [`validate_within_root`] が返す未加工パス（`canonicalize` を経由しない
/// `root.join(path)`）をこの 1 回の `open` にそのまま渡す。`open` 自体を
/// 「最終エントリが symlink なら失敗する」形にすることで、経路の閉じ込め検証と
/// 実際の読み込みを 1 syscall に結合し、検証後に別途 `open` し直す 2 段階構成が
/// 生む TOCTOU（`ELOOP` を [`FreezeError::OutsideRoot`] へ写す）を作らない。
/// 中間ディレクトリの差し替えは [`validate_within_root`] が `path` を
/// `root` 直下の 1 コンポーネントへ制限することで経路自体を無くしている
/// （モジュール冒頭「経路の閉じ込め」参照）。
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

    #[cfg(any(target_os = "linux", target_os = "android"))]
    const O_NOFOLLOW: i32 = 0o400_000;
    #[cfg(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "freebsd",
        target_os = "dragonfly",
        target_os = "netbsd",
        target_os = "openbsd"
    ))]
    const O_NOFOLLOW: i32 = 0x0100;

    // `io::ErrorKind::FilesystemLoop` は本ツールチェーンの安定版 Rust では未安定
    // （`io_error_more`、rust-lang/rust#86442）のため使えず、`raw_os_error()` を
    // OS ごとの `ELOOP` 値と比較する（`open_without_blocking` 冒頭のコメントと同じ
    // 「`libc` に頼らず直書きする」方針）。
    #[cfg(any(target_os = "linux", target_os = "android"))]
    const ELOOP: i32 = 40;
    #[cfg(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "freebsd",
        target_os = "dragonfly",
        target_os = "netbsd",
        target_os = "openbsd"
    ))]
    const ELOOP: i32 = 62;

    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(O_NONBLOCK | O_NOFOLLOW)
        .open(path)
        .map_err(|err| {
            if err.kind() == io::ErrorKind::NotFound {
                FreezeError::NotFound(err)
            } else if err.raw_os_error() == Some(ELOOP) {
                // `O_NOFOLLOW` が最終コンポーネントの symlink を検出したときの
                // 挙動（`ELOOP`）。検証後に symlink へ差し替えられた TOCTOU と
                // 同じ扱い（経路の閉じ込め違反）にする。
                FreezeError::OutsideRoot
            } else {
                FreezeError::Io(err)
            }
        })
}

/// 評価データのパスを開く（Unix 以外）。
///
/// FIFO による無限ブロックは `mkfifo` が使える Unix 系環境に固有の攻撃経路のため、
/// それ以外の環境では通常の `File::open` で足りる（呼び出し元が fd を fstat で
/// 再検証する点は Unix 版と共通）。`O_NOFOLLOW` 相当が無いため symlink を
/// そのまま辿ってしまうが、経路の閉じ込め（symlink がルート外・ネストした
/// ファイルを指すケースの拒否）は呼び出し元（[`freeze_eval_data`]）がこの
/// 関数を呼ぶ**前**に [`confine_within_root_by_parent`] で確定させている
/// （モジュール冒頭「経路の閉じ込め」参照。ここでの `open` はその確認済みの
/// 実体パスを開くだけで、新たな符号解決を行わない）。
///
/// Windows では `File::open` にディレクトリを渡すと（Unix と異なり fd は開けず）
/// `ErrorKind::PermissionDenied`（`Access is denied.`）を返す。これをそのまま
/// [`FreezeError::Io`] へ倒すと、ディレクトリを渡した場合に
/// [`FreezeError::NotAFile`] を期待する呼び出し元・テストが Windows でのみ
/// 失敗する（rust-ci windows-latest で実測。Cursor Bugbot 指摘
/// PRRT_kwDOUq-SxM6mb6t-）。`open` 失敗後にパスの種別を確認するこの再検証は、
/// 既に失敗した `open` の結果をどの [`FreezeError`] に分類するかの後始末に
/// すぎず、確認結果を使って別の実体を開き直すわけではないため、経路の
/// 閉じ込め判定を再び TOCTOU に晒すことはない。
#[cfg(not(unix))]
fn open_without_blocking(path: &Path) -> Result<File, FreezeError> {
    File::open(path).map_err(|err| {
        if err.kind() == io::ErrorKind::NotFound {
            FreezeError::NotFound(err)
        } else if err.kind() == io::ErrorKind::PermissionDenied
            && std::fs::symlink_metadata(path)
                .map(|metadata| metadata.is_dir())
                .unwrap_or(false)
        {
            FreezeError::NotAFile
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

/// `open_without_blocking` 自体の単体テスト（Cursor Bugbot 指摘への対応）。
///
/// `crates/data/tests/eval_freeze.rs` の結合テスト
/// `freeze_eval_data_returns_not_a_file_for_fifo_without_blocking` は、渡すパスが
/// 最初から FIFO であるため `freeze_eval_data` 冒頭の `metadata(path).is_file()`
/// （経路の閉じ込め検証の直後・`open_without_blocking` 呼び出しより前）の時点で
/// 既に `NotAFile` を返してしまい、`open_without_blocking` のノンブロッキング
/// 実装そのものは経由しない。そのため `open_without_blocking` を `File::open` へ
/// 差し戻す回帰が起きても、その結合テストは（別の分岐で）同じ `NotAFile` を返し
/// 続けて検知できない。ここでは private 関数 `open_without_blocking` を直接呼び、
/// 書き手の無い FIFO に対して実際にノンブロッキングで返ることを検証する
/// （TOCTOU 対策のロックイン。REQ-39「無制限の…無限待ちを作らない」）。
#[cfg(all(test, unix))]
mod open_without_blocking_tests {
    use super::{FreezeError, open_without_blocking};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    /// プロセス ID・現在時刻（ナノ秒）から一意な一時ディレクトリを作る
    /// （このモジュール専用。`std::env::temp_dir()` を親にし、テスト終了後に削除する）。
    fn unique_temp_dir(tag: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!(
            "fandhe-edge-data-open-without-blocking-{}-{tag}-{nanos}",
            std::process::id(),
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir for open_without_blocking test");
        dir
    }

    /// REQ-39 回帰テスト: 書き手の無い FIFO を `open_without_blocking` へ直接渡しても
    /// 無期限にブロックせず、fd を返した上でその fd が通常ファイルでないと
    /// 判定できることを確認する（Linux/macOS 実機・テストハーネス）。
    #[test]
    fn open_without_blocking_returns_promptly_for_writerless_fifo() {
        let dir = unique_temp_dir("fifo");
        let fifo_path = dir.join("eval.fifo");

        let status = std::process::Command::new("mkfifo")
            .arg(&fifo_path)
            .status()
            .expect("mkfifo command must be available on unix test runners");
        assert!(status.success(), "mkfifo must exit successfully");

        let (tx, rx) = std::sync::mpsc::channel();
        let fifo_path_for_thread = fifo_path.clone();
        std::thread::spawn(move || {
            let result = open_without_blocking(&fifo_path_for_thread).map(|file| file.metadata());
            // メインスレッドがタイムアウトで抜けた後に送信が失敗しても
            // （受信側が既に drop 済み）テストの成否には影響しないため無視する。
            let _ = tx.send(result);
        });

        let result = rx
            .recv_timeout(Duration::from_secs(5))
            .expect("open_without_blocking must not block indefinitely on a writerless FIFO");

        match result {
            Ok(Ok(metadata)) => {
                assert!(
                    !metadata.is_file(),
                    "opened FIFO fd must not report as a regular file"
                );
            }
            Ok(Err(err)) => panic!("fstat on opened FIFO fd failed: {err}"),
            Err(err) => panic!("open_without_blocking returned an error: {err}"),
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// REQ-39 回帰テスト（TOCTOU 修正・codex/Cursor Bugbot 指摘
    /// PRRT_kwDOUq-SxM6mbqce・PRRT_kwDOUq-SxM6mbtcp・PRRT_kwDOUq-SxM6mb5Yq への
    /// 対応）: `open_without_blocking` に symlink を直接渡すと `O_NOFOLLOW` により
    /// `ELOOP` となり、`FreezeError::OutsideRoot` を返すことを確認する。
    ///
    /// `validate_within_root` は構文検証のみで `canonicalize` を行わないため、
    /// `path`（`root` 直下の 1 エントリ）が symlink であるかどうかは、この
    /// `open` 1 回（`O_NOFOLLOW` 付き）でのみ判定される。`freeze_eval_data`
    /// 経由の結合テスト `..._for_symlink_escaping_root` はこの防御を実際に
    /// 経由して `OutsideRoot` を返す。ここでは `open_without_blocking` を直接
    /// 呼び、`open` 自体が symlink を拒否する挙動を単体レベルでロックインする。
    #[test]
    fn open_without_blocking_returns_outside_root_for_symlink() {
        let dir = unique_temp_dir("symlink");
        let target = dir.join("target.jsonl");
        std::fs::write(&target, b"target").expect("write symlink target file");
        let link = dir.join("link.jsonl");
        std::os::unix::fs::symlink(&target, &link).expect("create symlink");

        let result = open_without_blocking(&link);

        match result {
            Err(FreezeError::OutsideRoot) => {}
            Err(other) => panic!("expected OutsideRoot for symlink, got {other}"),
            Ok(_) => panic!("expected OutsideRoot error for symlink, got Ok"),
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}
