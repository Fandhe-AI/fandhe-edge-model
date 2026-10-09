//! 凍結済み評価データの読み取り専用配置と、直接書き込みの拒否確認
//! （REQ-39・REQ-17・TASK-17.2-2）。
//!
//! # 呼び出し文脈
//!
//! CLI の `register` 工程（`stages::register`。REQ-33・#136）から、
//! [`crate::eval_freeze::freeze_eval_data`] で凍結記録を作った直後に呼ばれる
//! 想定の順序は「凍結（[`crate::eval_freeze`]）→ 配置（本モジュール）→
//! 評価時の再計算（[`crate::eval_freeze::evaluate_gate`]）」。
//!
//! # 設計
//!
//! 以前の実装は、利用者が指定した `path`（＝評価データの置き場所）を
//! `chmod` するか、その `path` を `rename` で置き換えることで読み取り専用
//! 配置を実現しようとしていた。しかしどちらの方式も、利用者の `path` が
//! 指す実体を直接書き換える構造を持つ限り、以下の問題を避けられなかった
//! （issue #227 codex[bot]・cursor[bot] 指摘）:
//!
//! - `rename` の直前に対象の同一性を確認していなければ、差し替えられた
//!   別ファイルの上に配置してしまう
//! - `rename` の後に書き込み拒否を検証していると、検証が失敗した時点で
//!   既に利用者の `path` は書き換わってしまっている
//! - コピー量をあらかじめ業務上のサイズ上限で制限していないと、際限なく
//!   読み進めてしまう
//! - コピーしたバイト数だけを照合していると、同じ長さのまま内容だけ
//!   すり替えられたデータをそのまま配置できてしまう
//!
//! 本実装はこれらすべてを、**利用者の `src` には一切触れない**設計で解決
//! する。[`place_read_only`] は次のように動作する:
//!
//! 1. `src`（利用者の評価データファイル）を、[`fandhe_edge_core::fs::read_bounded`]
//!    で `max_bytes` までの上限付きで全量メモリへ読み込む（超過は
//!    [`FsError::TooLarge`] 経由で拒否。事前に `record.byte_len() <=
//!    max_bytes` も検証し、記録が主張するサイズの時点で上限超過を検出
//!    できる場合はファイルを読む前に拒否する）
//! 2. 読み込んだバイト列に [`crate::eval_freeze::freeze_eval_data`] を適用し、
//!    呼び出し側が渡した `record`（[`crate::eval_freeze::FreezeRecord`]）と
//!    完全一致するか照合する（[`crate::eval_freeze::evaluate_gate`] と同じ
//!    比較ロジック。sha256・バイト長のどちらかでも異なれば
//!    [`PlacementError::HashMismatch`] で拒否し、以降のファイルシステム
//!    操作を一切行わない）。**検証したバイト列そのものを次の手順で書き
//!    出す**ため、「検証時点」と「書き出し時点」の内容が食い違う窓は
//!    構造的に存在しない
//! 3. `dest_dir`（呼び出し側が用意した、本ツールが管理するディレクトリ）
//!    の mode・所有者を検証する（後述「`dest_dir` の機密性」。不適格なら
//!    [`PlacementError::InsecureDestDir`]）。次に `dest_dir` の中に
//!    `mkdir(0700)` で非公開の作業ディレクトリを作り、その中に
//!    `create_new`（`O_EXCL`）で新規ファイルを作って検証済みのバイト列を
//!    書き込み、`sync_all` してから `0o400`（所有者のみ読み取り可）へ
//!    `chmod` する
//! 4. 公開する前に、作業ディレクトリ内のそのファイルへ対して書き込み
//!    拒否のプローブ（非破壊的な `append` オープン試行）を行う。拒否され
//!    なければ（root 実行時など）作業ディレクトリを片付けて
//!    [`PlacementError::WriteNotRejected`] を返し、**何も公開しない**
//! 5. プローブに成功して初めて、`std::fs::hard_link` で
//!    `dest_dir/file_name` として公開する。`hard_link` は宛先が既に存在
//!    する場合に原子的に失敗するため、既存のファイルを上書きすることが
//!    ない（宛先が既にあれば [`PlacementError::AlreadyExists`]）。
//!    `rename` は使わない
//! 6. 公開直後、`dest_path` に現れた実体が検証・chmod したステージング
//!    inode と本当に同一であることを `(dev, ino)` で突き合わせる
//!    （後述「公開後の検証」）。**この検証が通るまで、また通らなかった
//!    場合も、`dest_path` を削除しない**。同一でなければ他者が既に
//!    差し替えたと判断し、そのファイルには触れず
//!    [`PlacementError::DestReplacedAfterPublish`] を返す。突き合わせ
//!    自体（`symlink_metadata`）が失敗した場合は、`dest_path` が本当に
//!    自分が公開した実体かをこの時点で確認できないため、撤去を試みず
//!    [`PlacementError::PublishedButUnverified`]（配置の成否・
//!    `dest_path` の残存状況が「不明」であることを表す）を返す。呼び出し
//!    側が単純に再試行すると [`PlacementError::AlreadyExists`] になって
//!    配置の成否を誤認する、という事態を避けるため、公開後の失敗経路は
//!    必ずこのいずれかへ着地させ、宙ぶらりんの成功を返さない
//! 7. 公開後、作業ディレクトリ（もう中身は無い）を片付ける。この後始末
//!    に失敗しても配置自体は有効なため、[`ReadOnlyPlacement::cleanup_failed`]
//!    に記録して成功を返す（無視はしない）
//!
//! この設計では、本関数が権限変更（`chmod`）を行う対象は、常に本関数
//! 自身が新規作成した（他のどのパスからも見えない）inode に限られる。
//! `src` は最初から最後まで読み込み専用で参照するだけで、`chmod`・
//! `rename`・`unlink` のいずれも行わない。
//!
//! # 責務の境界（本モジュールが行わないこと）
//!
//! - **経路の閉じ込め（`../`・絶対パス・symlink によるルート外参照の拒否）は
//!   行わない**（TASK-39.x・issue #157/#158 の対象。本モジュールが受け取る
//!   `src`・`dest_dir` はガード層を通過済みであることを前提とする。crate
//!   全体の前提条件。`crates/data/src/lib.rs`）
//! - **`max_bytes`（業務上の最大サイズ）の既定値は決めない**（issue #172の
//!   対象）。呼び出し側の方針として受け取るだけで、本モジュールが定数を
//!   新設することはしない
//! - **配置後の事後改変の検知は行わない**。本関数が返した
//!   [`ReadOnlyPlacement`] を受け取った「後」に、誰かが `dest_dir` 配下で
//!   `dest_path` を差し替える・別の内容で上書きする、といった事後の改変
//!   まではこのモジュールでは検出できない。その検出は、評価時に凍結記録
//!   の sha256 と再計算した値を突き合わせる
//!   [`crate::eval_freeze::evaluate_gate`]（REQ-17・TASK-17.3・issue #49）が担う
//! - **root 実行下での書き込み防止は保証しない**。root は mode `0o400` でも
//!   書き込めるため、[`place_read_only`] は書き込みを防げない配置を
//!   「配置済み」と装わず、[`PlacementError::WriteNotRejected`] を返して
//!   fail-closed に失敗させ、何も公開しない（後述「root・ACL の扱い」）
//! - **非 unix では読み取り専用配置そのものを拒否する**
//!   （[`PlacementError::UnsupportedPlatform`]）。M10 時点で Windows は
//!   対象外の OS のため、「実装済みを装わない」
//!   （`.claude/rules/coding-rust.md`）方針で fail-closed に拒否する
//! - **`dest_dir` の親ディレクトリの連鎖（さらに上位が他者に開かれて
//!   いないか）は検査しない**。`dest_dir` 自身を（後述の検証を通った）
//!   `0700` に保てば、その中に公開するファイルへ他者は到達できない。
//!   ルート配下の閉じ込め自体は TASK-39.x の担当
//! - **CLI の出力 JSON 全体の形は決めない**。入出力契約は TASK-33.3 に委ねる
//!
//! # `dest_dir`・`file_name` の扱い
//!
//! `dest_dir` は呼び出し側が **`0700`（所有者のみ読み書き・実行可）で
//! 用意する**契約の、本ツールが管理するディレクトリ（root 配下）である
//! ことを前提とする。本関数は `dest_dir` がディレクトリであり symlink で
//! ないことを確認し（[`PlacementError::InvalidDestDir`]）、それ以上の
//! 来歴は検証しない。`file_name` は `dest_dir` 直下の単一の通常コンポー
//! ネントに限定する。区切り文字（`/`・`\`）・NUL バイト・空文字列・`.`・
//! `..` はすべて [`PlacementError::InvalidFileName`] として拒否し、
//! `dest_dir` の外・作業ディレクトリの外を指すファイル名を受け付けない。
//!
//! # `dest_dir` の機密性（unix。issue #227 codex[bot] P0 指摘）
//!
//! 公開するファイルの mode を一律 `0o444`（誰でも読み取り可）にすると、
//! `dest_dir` を他のローカルユーザーが辿れる配置では評価データ本文が
//! 全ローカルユーザーに読めてしまう。本関数はこれを次の 2 段で防ぐ:
//!
//! 1. 公開するファイルの mode は `0o444` ではなく **`0o400`（所有者のみ
//!    読み取り可）**にする。読み取り専用であること・書き込み拒否の
//!    プローブ（非 root では `EACCES`）が成立することは変わらない
//! 2. `dest_dir` 自身の mode に group・other 向けの権限ビットが一つでも
//!    あれば（`mode & 0o077 != 0`）、または `dest_dir` の所有者が
//!    呼び出しプロセスの実効ユーザーでなければ、`0o400` にしたところで
//!    同じ `dest_dir` を書き込める・所有する別ユーザー経由で内容へ到達
//!    されうるため、[`PlacementError::InsecureDestDir`] で拒否し何も
//!    公開しない
//!
//! 実効ユーザー ID の取得に `getuid(2)` 相当の `unsafe` な FFI・新規依存
//! （`libc` 等）は使わない（`.claude/rules/coding-rust.md`「unsafe は
//! 原則禁止」・`.claude/rules/dependency-policy.md`）。代わりに、本関数が
//! 自分自身で新規作成した作業ディレクトリ（`mkdir` の呼び出し元プロセスの
//! 実効ユーザーが所有者になる）の所有者 uid を「自分の uid」の代わりに
//! 使い、`dest_dir` の所有者と突き合わせる。
//!
//! # root・ACL の扱い（安全側に倒した判断）
//!
//! mode ビットの設定だけでは、root（`CAP_DAC_OVERRIDE`）・書き込みを許す
//! POSIX ACL・mode を無視するファイルシステムのいずれでも書き込みを防げない
//! ことがある。[`place_read_only`] は mode を設定した「つもり」で終わらず、
//! 必ず公開前に実際の書き込み試行で拒否を確認し、拒否を確認できなければ
//! [`PlacementError::WriteNotRejected`] を返して fail-closed にする
//! （`.claude/rules/coding-rust.md`「未実装・簡易実装の箇所は実装済みを
//! 装わない」）。この確認は公開前（作業ディレクトリ内）に行うため、
//! 確認に失敗しても `dest_dir/file_name` には何も現れない。root 実行を
//! 許すかどうかの方針は CLI 配線（TASK-33.3）で決めるべき事項とする。
//!
//! # セキュリティ上の注意
//!
//! [`PlacementError`] の `Display` は英語固定で、評価データ本文を含めない
//! （`.claude/rules/security.md`）。本モジュールはパスをシェル・子プロセスへ
//! 渡さない（`PathBuf` のみで扱う）。
//!
//! # 出典
//!
//! 公開前の書き込み拒否プローブ（`append` モードで開いて即座に閉じる）は
//! PoC-20 ケース 5
//! （`docs/spec/03-poc/safety-hardening/scripts/case5_eval_integrity.py`）の
//! `open(eval_path, "a")` を移植したもの。PoC は `chmod 444` した評価
//! ファイルを追記モードで開くと `PermissionError`（errno 13）になることを
//! 実測している（証拠の種別: テストハーネス）。PoC-20 自体も、ツールが
//! 管理する場所へコピーしたうえでそのコピーを読み取り専用にする方式を
//! 採っており、利用者の元データには触れない。

use crate::eval_freeze::{FreezeRecord, freeze_eval_data};
use fandhe_edge_core::fs::{FsError, read_bounded};
use std::fmt;
#[cfg(unix)]
use std::fs::OpenOptions;
use std::io;
#[cfg(unix)]
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

/// [`place_read_only`] が成功した証跡（読み取り専用配置が成立したことを
/// 表す）。フィールドは非公開にし、[`place_read_only`] を通してしか作れない
/// （形だけの「配置済み」偽装を防ぐ）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadOnlyPlacement {
    // 公開先（`dest_dir.join(file_name)`）。
    path: PathBuf,
    // unix のみで意味を持つ値（非 unix では `place_read_only` が
    // `PlacementError::UnsupportedPlatform` を返すため本型は構築されない）。
    mode: u32,
    // 公開直後、`path` 自身を `symlink_metadata` して取得した `(dev, ino)`。
    // `hard_link` によって公開された inode の識別に使う。
    dev_ino: (u64, u64),
    // 公開後の作業ディレクトリの後始末（ファイル・ディレクトリの削除）に
    // 失敗したかどうか。配置自体は有効だが、呼び出し側が診断・監視で
    // 使えるよう無視せずに記録する。
    cleanup_failed: bool,
}

impl ReadOnlyPlacement {
    /// 公開先のパス（`dest_dir.join(file_name)`）。
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// 配置後の unix パーミッションビット（`mode & 0o7777`）。
    #[must_use]
    pub fn mode(&self) -> u32 {
        self.mode
    }

    /// 公開直後の `(dev, ino)`。呼び出し側が後から
    /// `std::fs::symlink_metadata` 等で同じ実体かどうかを突き合わせる
    /// ために公開する。
    #[must_use]
    pub fn dev_ino(&self) -> (u64, u64) {
        self.dev_ino
    }

    /// 公開後の作業ディレクトリの後始末に失敗したかどうか。`true` でも
    /// 配置自体（`path` が読み取り専用で存在すること）は有効。
    #[must_use]
    pub fn cleanup_failed(&self) -> bool {
        self.cleanup_failed
    }
}

/// [`place_read_only`] が失敗する理由。
#[derive(Debug)]
#[non_exhaustive]
pub enum PlacementError {
    /// `file_name` が `dest_dir` 直下の単一の通常コンポーネントでなかった
    /// （空文字列・`.`・`..`・区切り文字・NUL バイトのいずれかを含む）。
    InvalidFileName,
    /// `dest_dir` が存在しない・ディレクトリでない・symlink だった。
    InvalidDestDir { path: PathBuf },
    /// `dest_dir` が「所有者のみアクセス可」の契約を満たしていなかった
    /// （mode に group・other 向けの権限ビットがある、または所有者が
    /// 呼び出しプロセスの実効ユーザーでない。issue #227 codex[bot] P0
    /// 指摘: 評価データの機密性。モジュール doc「`dest_dir` の機密性」）。
    InsecureDestDir { path: PathBuf },
    /// 凍結記録が主張するバイト長（[`FreezeRecord::byte_len`]）が
    /// `max_bytes` を超えていた。`src` を読み込む前に検出する。
    TooLarge { size: u64, limit: u64 },
    /// `src` から読み込んだ実データを [`freeze_eval_data`] で再計算した
    /// 結果が、渡された `record` と一致しなかった（fail-closed。
    /// モジュール doc「設計」手順 2）。
    HashMismatch,
    /// `dest_dir/file_name` に既にファイルが存在していた
    /// （`std::fs::hard_link` が原子的に検出。上書きしない）。
    AlreadyExists { path: PathBuf },
    /// 権限を読み取り専用へ変更した後も、公開前の書き込みプローブが
    /// 拒否されなかった。root（`CAP_DAC_OVERRIDE`）等で mode ビットの
    /// 設定だけでは書き込みを防げない状況を検出する（fail-closed。
    /// モジュール doc「root・ACL の扱い」）。このプローブは公開前に
    /// 行うため、このエラーが返る時点で `dest_dir/file_name` には
    /// 何も存在しない。
    WriteNotRejected { path: PathBuf },
    /// 公開（`hard_link`）に成功した直後の検証で、`dest_path` に現れた
    /// 実体が検証・chmod したステージング inode と一致しなかった
    /// （`(dev, ino)` 不一致）。他者が既に `dest_path` を差し替えたと
    /// 判断し、そのファイルには触れずに返す（削除しない。issue #227 P1
    /// 指摘: 公開後の失敗経路を「これは他者のものだ」（本 variant）か
    /// 「確認できない」（[`PlacementError::PublishedButUnverified`]）の
    /// どちらかへ必ず着地させ、誤って上書き・削除しないこと。モジュール
    /// doc「設計」手順 6）。
    DestReplacedAfterPublish { path: PathBuf },
    /// 公開（`hard_link`）には成功したが、公開直後の検証
    /// （`symlink_metadata`）自体が失敗した。`dest_path` が本当に自分が
    /// 公開した実体かをこの時点では確認できないため、`remove_file` で
    /// 撤去を試みることもしない（issue #227 codex[bot] P1 指摘: 撤去を
    /// 試みると、無関係な別ファイルをたまたま同じパスに見つけて削除して
    /// しまう競合がありうる）。`dest_path` に何が残っているか本関数の
    /// 内部では確定できない状態のまま返す。[`PlacementError::AlreadyExists`]
    /// （呼び出し側が単純に再試行して「既にある」と誤認する）とは区別し、
    /// 呼び出し側に手動確認を促す。
    PublishedButUnverified { path: PathBuf },
    /// 非 unix プラットフォームでは読み取り専用配置そのものを拒否する
    /// （モジュール doc「責務の境界」）。
    UnsupportedPlatform { path: PathBuf },
    /// `src` の読み込みに伴う防御（通常ファイル判定・サイズ上限・
    /// TOCTOU 対策）由来のエラー（[`fandhe_edge_core::fs`]）。
    Fs(FsError),
    /// 作業ディレクトリの作成・ファイル作成・権限変更・`hard_link` などの
    /// I/O エラー。公開直後の検証（`symlink_metadata`）自体の失敗は
    /// ここに含めない（[`PlacementError::PublishedButUnverified`] を
    /// 返す。`dest_path` が本当に自分の公開した実体かをこの時点では
    /// 確認できないため、`Io` として片付けたり削除したりしない）。
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
}

impl fmt::Display for PlacementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlacementError::InvalidFileName => write!(
                f,
                "file name must be a single non-empty path component without separators"
            ),
            PlacementError::InvalidDestDir { path } => {
                write!(
                    f,
                    "{} is not a usable destination directory",
                    path.display()
                )
            }
            PlacementError::InsecureDestDir { path } => {
                write!(
                    f,
                    "{} is not owner-only (0700), refusing to publish into it",
                    path.display()
                )
            }
            PlacementError::TooLarge { size, limit } => {
                write!(f, "eval data size exceeds limit ({size} > {limit} bytes)")
            }
            PlacementError::HashMismatch => write!(
                f,
                "recomputed eval data hash does not match the frozen record"
            ),
            PlacementError::AlreadyExists { path } => {
                write!(
                    f,
                    "{} already exists, refusing to overwrite it",
                    path.display()
                )
            }
            PlacementError::WriteNotRejected { path } => write!(
                f,
                "{} would still be writable after setting read-only permissions",
                path.display()
            ),
            PlacementError::DestReplacedAfterPublish { path } => write!(
                f,
                "{} was replaced by another entity right after publishing",
                path.display()
            ),
            PlacementError::PublishedButUnverified { path } => write!(
                f,
                "{} was published but its post-publish state could not be verified or removed",
                path.display()
            ),
            PlacementError::UnsupportedPlatform { path } => write!(
                f,
                "{} cannot be placed read-only on this platform",
                path.display()
            ),
            PlacementError::Fs(source) => write!(f, "{source}"),
            PlacementError::Io { path, source } => {
                write!(
                    f,
                    "failed to place {} as read-only: {source}",
                    path.display()
                )
            }
        }
    }
}

impl std::error::Error for PlacementError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PlacementError::Fs(source) => Some(source),
            PlacementError::Io { source, .. } => Some(source),
            PlacementError::InvalidFileName
            | PlacementError::InvalidDestDir { .. }
            | PlacementError::InsecureDestDir { .. }
            | PlacementError::TooLarge { .. }
            | PlacementError::HashMismatch
            | PlacementError::AlreadyExists { .. }
            | PlacementError::WriteNotRejected { .. }
            | PlacementError::DestReplacedAfterPublish { .. }
            | PlacementError::PublishedButUnverified { .. }
            | PlacementError::UnsupportedPlatform { .. } => None,
        }
    }
}

impl From<FsError> for PlacementError {
    fn from(source: FsError) -> Self {
        PlacementError::Fs(source)
    }
}

/// `file_name` が `dest_dir` 直下の単一の通常コンポーネントであることを
/// 検証する。空文字列・`.`・`..`・区切り文字（`/`・`\`）・NUL バイトの
/// いずれかを含む場合は拒否する。
///
/// バイト単位の明示チェックに加えて `Path::components()` でも単一の
/// `Normal` コンポーネントになることを確認し、プラットフォーム固有の
/// プレフィックス（Windows の `C:` 等）や、バイト単位のチェックだけでは
/// 見落としうる解釈のずれを二重に防ぐ。
fn validate_file_name(file_name: &str) -> Result<(), PlacementError> {
    if file_name.is_empty() || file_name == "." || file_name == ".." {
        return Err(PlacementError::InvalidFileName);
    }
    if file_name.bytes().any(|b| b == b'/' || b == b'\\' || b == 0) {
        return Err(PlacementError::InvalidFileName);
    }
    let mut components = Path::new(file_name).components();
    match (components.next(), components.next()) {
        (Some(std::path::Component::Normal(_)), None) => Ok(()),
        _ => Err(PlacementError::InvalidFileName),
    }
}

/// `dest_dir` がディレクトリであり symlink でないことを確認する。
///
/// `symlink_metadata` はリンクを辿らないため、`dest_dir` 自身が symlink
/// である場合を正しく検出できる（`std::fs::metadata` はリンクを辿って
/// しまう）。存在しない場合はここでは区別せず、呼び出し元が `Io` として
/// 扱う。
fn check_dest_dir(dest_dir: &Path) -> Result<(), PlacementError> {
    let meta = std::fs::symlink_metadata(dest_dir).map_err(|source| PlacementError::Io {
        path: dest_dir.to_path_buf(),
        source,
    })?;
    if meta.file_type().is_symlink() || !meta.is_dir() {
        return Err(PlacementError::InvalidDestDir {
            path: dest_dir.to_path_buf(),
        });
    }
    Ok(())
}

/// `dest_dir` の mode に group・other 向けの権限ビットが無いことを確認する
/// （unix 限定。モジュール doc「`dest_dir` の機密性」）。
///
/// `dest_dir` を作成した呼び出し側が `0700` の契約を守っているかどうかを
/// 判定する 1 段目の検査（uid の突き合わせより前に、作業ディレクトリを
/// 一切作らずに済ませられる安価な検査を先に行う）。
#[cfg(unix)]
fn check_dest_dir_mode(dest_dir: &Path) -> Result<(), PlacementError> {
    use std::os::unix::fs::PermissionsExt as _;

    let meta = std::fs::symlink_metadata(dest_dir).map_err(|source| PlacementError::Io {
        path: dest_dir.to_path_buf(),
        source,
    })?;
    if meta.permissions().mode() & 0o077 != 0 {
        return Err(PlacementError::InsecureDestDir {
            path: dest_dir.to_path_buf(),
        });
    }
    Ok(())
}

/// 配置先ディレクトリの操作の抽象（[`place_read_only_bytes`] が使う。REQ-39・REQ-17）。
///
/// 凍結配置の手順（照合 → ステージングへ書き込み → 0400 化 → 書き込み拒否のプローブ → 既存を置き換えない
/// 公開 → 片付け）は [`place_read_only_bytes`] の 1 か所に置き、ファイル操作だけをこの trait で差し替える。
/// パス版は [`StdPlacementDir`]（std のパス操作。[`place_read_only`] が使う）、CLI は保持した
/// ディレクトリ fd 起点の実装（検証後のパス差し替えで外へ出ない）を渡す。data 層は guard・rustix に
/// 依存しないため、fd 起点の実装は呼び出し側（CLI）が書く。
///
/// 名前引数 `rel` は配置先直下からの相対（`<staging>/frozen` のように 1 段の入れ子のみ）で、
/// 実装は配置先の外へ出さないこと。すべてシンボリックリンクを辿らない操作にする。
pub trait PlacementDir {
    /// 配置先ディレクトリ自身の `(mode & 0o7777, 所有者 uid)`。
    fn dir_mode_and_uid(&self) -> io::Result<(u32, u32)>;
    /// 配置先直下の非公開ディレクトリを 0700 で新規作成する（既存なら `AlreadyExists`）。
    fn create_private_dir(&self, name: &str) -> io::Result<()>;
    /// 配置先直下のディレクトリ `name` の所有者 uid（辿らない）。
    fn entry_uid(&self, name: &str) -> io::Result<u32>;
    /// `rel` に新規ファイルを排他的に作る（既存なら `AlreadyExists`）。
    fn create_new_file(&self, rel: &str) -> io::Result<std::fs::File>;
    /// `rel` を追記モードで開けるかを試す。開ければ `Ok`（書き込みを拒否できていない）。
    fn probe_append(&self, rel: &str) -> io::Result<()>;
    /// `rel` を配置先直下の `name` として、**既存を置き換えずに**公開する（既存なら `AlreadyExists`）。
    fn publish_no_replace(&self, rel: &str, name: &str) -> io::Result<()>;
    /// 配置先直下の `name`（辿らない）の `(dev, ino, mode & 0o7777)`。
    fn stat_entry(&self, name: &str) -> io::Result<(u64, u64, u32)>;
    /// 配置先直下のディレクトリ `name` を中身ごと片付ける。
    fn remove_dir_tree(&self, name: &str) -> io::Result<()>;
}

/// `dest_dir` の中に非公開の作業ディレクトリを新規作成する（unix 限定）。
///
/// 作業ディレクトリは `PlacementDir::create_private_dir`（0700）で作る。名前は衝突時に付け直す。
#[cfg(unix)]
fn create_staging_dir<D: PlacementDir + ?Sized>(dest: &D) -> Result<String, PlacementError> {
    let pid = std::process::id();
    for attempt in 0..1000u32 {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let name = format!(".fandhe-edge-frozen-staging-{pid}-{attempt}-{nanos}");
        match dest.create_private_dir(&name) {
            Ok(()) => return Ok(name),
            Err(err) if err.kind() == ErrorKind::AlreadyExists => continue,
            Err(source) => {
                return Err(PlacementError::Io {
                    path: PathBuf::from(name),
                    source,
                });
            }
        }
    }
    Err(PlacementError::Io {
        path: PathBuf::new(),
        source: io::Error::other("failed to create a unique staging directory"),
    })
}

/// 作業ディレクトリのエラー経路の後始末（ベストエフォート）。成功経路では明示的に片付け、
/// 成否を [`ReadOnlyPlacement::cleanup_failed`] へ記録する。
#[cfg(unix)]
struct StagingGuard<'a, D: PlacementDir + ?Sized> {
    dest: &'a D,
    name: String,
}

#[cfg(unix)]
impl<D: PlacementDir + ?Sized> Drop for StagingGuard<'_, D> {
    fn drop(&mut self) {
        let _ = self.dest.remove_dir_tree(&self.name);
    }
}

/// 検証済みのバイト列を、`dest` の直下へ読み取り専用で配置する（REQ-39・REQ-17・TASK-17.2-2）。
///
/// 手順（モジュール doc「設計」）の実体はここに 1 つだけ置く。`bytes` を [`freeze_eval_data`] で
/// 再計算して `record` と照合（不一致は [`PlacementError::HashMismatch`]。以降の操作を一切しない）→
/// `dest` の mode・所有者の確認 → 非公開ステージングへ書き込み・`sync`・0400 化 → 書き込み拒否の
/// プローブ（拒否を確認できなければ [`PlacementError::WriteNotRejected`]。何も公開しない）→
/// 既存を置き換えない公開 → 公開後の `(dev, ino)` の突き合わせ → ステージングの片付け。
/// 検証したバイト列そのものを書き出すため、検証と書き出しの間に内容が変わる窓は無い。
///
/// パスを一切扱わないので、呼び出し側が保持 fd 起点の [`PlacementDir`] を渡せば、検証後の
/// パス差し替えでプロジェクトの外を読み書きしない。返す [`ReadOnlyPlacement::path`] は `file_name`
/// （配置先からの相対）。非 unix では常に [`PlacementError::UnsupportedPlatform`]。
pub fn place_read_only_bytes<D: PlacementDir + ?Sized>(
    bytes: &[u8],
    dest: &D,
    file_name: &str,
    record: &FreezeRecord,
    max_bytes: u64,
) -> Result<ReadOnlyPlacement, PlacementError> {
    validate_file_name(file_name)?;
    if record.byte_len() > max_bytes {
        return Err(PlacementError::TooLarge {
            size: record.byte_len(),
            limit: max_bytes,
        });
    }
    let actual_len = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    if actual_len > max_bytes {
        return Err(PlacementError::TooLarge {
            size: actual_len,
            limit: max_bytes,
        });
    }
    // `usize` から `u64` への変換失敗は実質到達しないが、外部入力の経路では `unwrap` を使わない。
    let recomputed = freeze_eval_data(bytes).map_err(|_| PlacementError::Io {
        path: PathBuf::from(file_name),
        source: io::Error::other("eval data byte length does not fit in u64"),
    })?;
    if &recomputed != record {
        return Err(PlacementError::HashMismatch);
    }
    let dest_path = PathBuf::from(file_name);

    #[cfg(unix)]
    {
        use std::io::Write as _;
        use std::os::unix::fs::MetadataExt as _;
        use std::os::unix::fs::PermissionsExt as _;

        let io_err = |source: io::Error| PlacementError::Io {
            path: dest_path.clone(),
            source,
        };
        // 1 段目: 配置先の mode に group・other 向けの権限ビットが無いこと。
        let (dir_mode, dir_uid) = dest.dir_mode_and_uid().map_err(io_err)?;
        if dir_mode & 0o077 != 0 {
            return Err(PlacementError::InsecureDestDir {
                path: dest_path.clone(),
            });
        }

        let staging_name = create_staging_dir(dest)?;
        let _staging = StagingGuard {
            dest,
            name: staging_name.clone(),
        };
        // 2 段目: 今作った作業ディレクトリの所有者（＝自分の uid）と配置先の所有者を突き合わせる
        // （`getuid` の unsafe FFI・依存を使わない。モジュール doc「`dest_dir` の機密性」）。
        let my_uid = dest.entry_uid(&staging_name).map_err(io_err)?;
        if dir_uid != my_uid {
            return Err(PlacementError::InsecureDestDir {
                path: dest_path.clone(),
            });
        }

        let staged_rel = format!("{staging_name}/frozen");
        let mut staging_file = dest.create_new_file(&staged_rel).map_err(io_err)?;
        staging_file
            .write_all(bytes)
            .and_then(|()| staging_file.sync_all())
            .map_err(io_err)?;
        // 権限変更するのはこの新規 inode だけ。`0o444` ではなく `0o400`（所有者のみ。issue #227）。
        const READ_ONLY_MODE: u32 = 0o400;
        staging_file
            .set_permissions(std::fs::Permissions::from_mode(READ_ONLY_MODE))
            .map_err(io_err)?;
        let staging_meta = staging_file.metadata().map_err(io_err)?;
        let staged_mode = staging_meta.permissions().mode() & 0o7777;
        if staged_mode != READ_ONLY_MODE {
            return Err(io_err(io::Error::other(format!(
                "permissions did not converge to {READ_ONLY_MODE:o} (got {staged_mode:o})"
            ))));
        }
        let staged_dev_ino = (staging_meta.dev(), staging_meta.ino());

        // 公開前の書き込み拒否プローブ（書き込みモードで開いたままのハンドルではなく、開き直して
        // 実際にファイルシステムが拒否することを確認する。非公開ディレクトリ内なので競合しない）。
        match dest.probe_append(&staged_rel) {
            Err(err) if err.kind() == ErrorKind::PermissionDenied => {}
            Ok(()) => {
                return Err(PlacementError::WriteNotRejected { path: dest_path });
            }
            Err(source) => {
                return Err(PlacementError::Io {
                    path: dest_path,
                    source,
                });
            }
        }

        // 公開。既存を置き換えない（原子的に失敗する）。
        match dest.publish_no_replace(&staged_rel, file_name) {
            Ok(()) => {}
            Err(source) if source.kind() == ErrorKind::AlreadyExists => {
                return Err(PlacementError::AlreadyExists { path: dest_path });
            }
            Err(source) => {
                return Err(PlacementError::Io {
                    path: dest_path,
                    source,
                });
            }
        }

        // 公開直後の検証。失敗経路は必ず「これは他者のものだ」か「確認できない」へ着地させ、公開後は
        // `dest_path` を削除しない（issue #227 codex[bot] P1 指摘。モジュール doc「設計」手順 6）。
        let (mode, (dev, ino)) = verify_and_finalize_publish(dest, file_name, staged_dev_ino)?;

        // 作業領域の後始末。公開は完了しているので、失敗しても配置は有効（`cleanup_failed` に記録）。
        let cleanup_failed = dest.remove_dir_tree(&staging_name).is_err();
        Ok(ReadOnlyPlacement {
            path: dest_path,
            mode,
            dev_ino: (dev, ino),
            cleanup_failed,
        })
    }

    #[cfg(not(unix))]
    {
        let _ = dest;
        Err(PlacementError::UnsupportedPlatform { path: dest_path })
    }
}

/// 公開直後の検証を行う（unix 限定。REQ-17・REQ-39・TASK-17.2-2。issue #227 P1 指摘）。
///
/// `file_name` に現れた実体が `expected_dev_ino`（検証・chmod 済みのステージング inode）と一致すれば
/// `(mode, dev_ino)` を返す。**公開後の経路では `file_name` を一切削除しない**（`stat` が失敗した
/// 時点で本当に自分が公開した inode かは確認できず、撤去すると無関係な別ファイルを消す競合が
/// ありうる）。失敗経路は次のいずれかへ必ず着地させ、呼び出し側が [`PlacementError::AlreadyExists`]
/// と誤認して単純に再試行する事態を避ける（モジュール doc「設計」手順 6）:
///
/// - `(dev, ino)` が一致しない: 他者が差し替えたと判断し、触れず
///   [`PlacementError::DestReplacedAfterPublish`]
/// - 突き合わせ自体（`stat_entry`）が失敗: 撤去を試みず [`PlacementError::PublishedButUnverified`]
#[cfg(unix)]
fn verify_and_finalize_publish<D: PlacementDir + ?Sized>(
    dest: &D,
    file_name: &str,
    expected_dev_ino: (u64, u64),
) -> Result<(u32, (u64, u64)), PlacementError> {
    let path = PathBuf::from(file_name);
    let (dev, ino, mode) = dest
        .stat_entry(file_name)
        .map_err(|_| PlacementError::PublishedButUnverified { path: path.clone() })?;
    if (dev, ino) != expected_dev_ino {
        return Err(PlacementError::DestReplacedAfterPublish { path });
    }
    Ok((mode, (dev, ino)))
}

/// パス（`dest_dir`）を std のパス操作で扱う [`PlacementDir`]（パス版 [`place_read_only`] 用）。
///
/// 検証後にパスが差し替えられる問題（TOCTOU）を避けたい呼び出し元は使わず、保持 fd 起点の
/// 実装を渡すこと（CLI の `register` がそうする）。
#[cfg(unix)]
struct StdPlacementDir<'a> {
    dir: &'a Path,
}

#[cfg(unix)]
impl PlacementDir for StdPlacementDir<'_> {
    fn dir_mode_and_uid(&self) -> io::Result<(u32, u32)> {
        use std::os::unix::fs::MetadataExt as _;
        let meta = std::fs::symlink_metadata(self.dir)?;
        Ok((meta.mode() & 0o7777, meta.uid()))
    }
    fn create_private_dir(&self, name: &str) -> io::Result<()> {
        use std::os::unix::fs::DirBuilderExt as _;
        let mut builder = std::fs::DirBuilder::new();
        builder.mode(0o700);
        builder.create(self.dir.join(name))
    }
    fn entry_uid(&self, name: &str) -> io::Result<u32> {
        use std::os::unix::fs::MetadataExt as _;
        Ok(std::fs::symlink_metadata(self.dir.join(name))?.uid())
    }
    fn create_new_file(&self, rel: &str) -> io::Result<std::fs::File> {
        use std::os::unix::fs::OpenOptionsExt as _;
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(self.dir.join(rel))
    }
    fn probe_append(&self, rel: &str) -> io::Result<()> {
        OpenOptions::new()
            .append(true)
            .open(self.dir.join(rel))
            .map(|_| ())
    }
    fn publish_no_replace(&self, rel: &str, name: &str) -> io::Result<()> {
        // `hard_link` は宛先が既に存在する場合に原子的に失敗する（`rename` は使わない）。
        std::fs::hard_link(self.dir.join(rel), self.dir.join(name))
    }
    fn stat_entry(&self, name: &str) -> io::Result<(u64, u64, u32)> {
        use std::os::unix::fs::MetadataExt as _;
        let meta = std::fs::symlink_metadata(self.dir.join(name))?;
        Ok((meta.dev(), meta.ino(), meta.mode() & 0o7777))
    }
    fn remove_dir_tree(&self, name: &str) -> io::Result<()> {
        std::fs::remove_dir_all(self.dir.join(name))
    }
}

/// 評価データ本体を読み取り専用配置にする（REQ-39・REQ-17・TASK-17.2-2）。パス版。
///
/// モジュール doc「設計」を参照。`src` は読み込むだけで、権限変更・
/// `rename`・`unlink` のいずれも行わない。`record` は `src` の内容から
/// 独立に計算済みの凍結記録（[`crate::eval_freeze::freeze_eval_data`]）で、
/// 本関数はそれを信用せず `src` から読み直した実データと突き合わせる。
/// `max_bytes` は呼び出し側の方針として受け取る上限で、本モジュールは
/// 既定値を持たない（モジュール doc「責務の境界」）。
///
/// 手順の実体は [`place_read_only_bytes`]（`src` の読み込みと `dest_dir` の事前検査の後、std の
/// パス操作の [`PlacementDir`] を渡す薄いアダプター）。パスを開き直すため、検証後の差し替えを
/// 避けたい呼び出し元は [`place_read_only_bytes`] を保持 fd 起点の実装で使うこと。
///
/// 非 unix では常に [`PlacementError::UnsupportedPlatform`] を返す
/// （モジュール doc「責務の境界」）。
pub fn place_read_only(
    src: &Path,
    dest_dir: &Path,
    file_name: &str,
    record: &FreezeRecord,
    max_bytes: u64,
) -> Result<ReadOnlyPlacement, PlacementError> {
    validate_file_name(file_name)?;

    if record.byte_len() > max_bytes {
        return Err(PlacementError::TooLarge {
            size: record.byte_len(),
            limit: max_bytes,
        });
    }

    check_dest_dir(dest_dir)?;
    #[cfg(unix)]
    {
        // mode の検査は `src` を読む前に済ませる（安価な検査を先に）。
        check_dest_dir_mode(dest_dir)?;
    }

    // `src` は読み込むだけ。上限は `max_bytes`（呼び出し側の方針）で、`read_bounded` は
    // メタデータ由来のサイズと実際に読んだバイト数の両方を照合する（TOCTOU 対策）。
    let bytes = read_bounded(src, max_bytes)?;
    let dest_path = dest_dir.join(file_name);

    #[cfg(unix)]
    {
        let mut placement = place_read_only_bytes(
            &bytes,
            &StdPlacementDir { dir: dest_dir },
            file_name,
            record,
            max_bytes,
        )
        .map_err(|e| with_dest_path(e, &dest_path))?;
        placement.path = dest_path;
        Ok(placement)
    }

    #[cfg(not(unix))]
    {
        let _ = &bytes;
        Err(PlacementError::UnsupportedPlatform { path: dest_path })
    }
}

/// [`place_read_only_bytes`] が返したエラーの相対パスを、パス版の絶対パスに置き換える
/// （パス版の既存の表示・挙動を保つ）。
#[cfg(unix)]
fn with_dest_path(error: PlacementError, dest_path: &Path) -> PlacementError {
    let path = dest_path.to_path_buf();
    match error {
        PlacementError::InsecureDestDir { .. } => PlacementError::InsecureDestDir { path },
        PlacementError::AlreadyExists { .. } => PlacementError::AlreadyExists { path },
        PlacementError::WriteNotRejected { .. } => PlacementError::WriteNotRejected { path },
        PlacementError::DestReplacedAfterPublish { .. } => {
            PlacementError::DestReplacedAfterPublish { path }
        }
        PlacementError::PublishedButUnverified { .. } => {
            PlacementError::PublishedButUnverified { path }
        }
        PlacementError::Io { source, .. } => PlacementError::Io { path, source },
        other => other,
    }
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

    fn write_unique_temp_file(label: &str, bytes: &[u8]) -> TempFileGuard {
        let pid = std::process::id();
        for attempt in 0..1000u32 {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let candidate = std::env::temp_dir().join(format!(
                "fandhe-edge-data-frozen-placement-unit-{pid}-{label}-{attempt}-{nanos}"
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

    /// テスト用の一時ディレクトリ（`dest_dir` 役）を、成否に関わらず
    /// 削除するガード（RAII）。`place_read_only` が内部で作るファイルは
    /// `0o400` になりうるが、削除に必要なのは親ディレクトリの書き込み
    /// 権限であって対象ファイル自身の権限ではないため、権限を戻す処理は
    /// 不要（POSIX の unlink の意味論）。
    struct TempDirGuard(PathBuf);

    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// `dest_dir` は呼び出し側が `0700` で用意する契約（モジュール doc
    /// 「`dest_dir` の機密性」）。テストヘルパーも同じ契約に従い、unix では
    /// 作成直後に明示的へ `0700` へ揃える（`std::fs::create_dir` の既定
    /// mode は umask 次第で group・other にビットが残りうるため）。
    /// `InsecureDestDir` を確認する専用テストだけは、この後で意図的に
    /// mode を緩める。
    fn make_temp_dir(label: &str) -> TempDirGuard {
        let pid = std::process::id();
        for attempt in 0..1000u32 {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let candidate = std::env::temp_dir().join(format!(
                "fandhe-edge-data-frozen-placement-dir-{pid}-{label}-{attempt}-{nanos}"
            ));
            match std::fs::create_dir(&candidate) {
                Ok(()) => {
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::PermissionsExt as _;
                        std::fs::set_permissions(
                            &candidate,
                            std::fs::Permissions::from_mode(0o700),
                        )
                        .expect("テスト用ディレクトリの権限設定に失敗しないはず");
                    }
                    return TempDirGuard(candidate);
                }
                Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(err) => panic!("テスト用ディレクトリの作成に失敗しないはず: {err}"),
            }
        }
        panic!("一意なテスト用ディレクトリを作成できなかった");
    }

    /// unix で実行ユーザーが root（uid 0）かどうかを判定する（`libc` 不要。
    /// 一時ファイルの `MetadataExt::uid()` が呼び出しユーザーの実効 uid と
    /// 一致することを利用する）。
    #[cfg(unix)]
    fn running_as_root(path: &Path) -> bool {
        use std::os::unix::fs::MetadataExt as _;
        std::fs::metadata(path)
            .map(|meta| meta.uid() == 0)
            .unwrap_or(false)
    }

    /// REQ-17・REQ-39・TASK-17.2-2: 非 root 環境での正常系。`dest_dir` 直下に読み取り
    /// 専用ファイルが 1 件だけ公開され、内容が一致し、直接の書き込み
    /// （append・truncate 双方）が拒否される。
    #[cfg(unix)]
    #[test]
    fn req17_req39_place_read_only_success() {
        let dest_dir = make_temp_dir("success-dest");
        let src = write_unique_temp_file("success-src", b"eval data");
        if running_as_root(&src.0) {
            // root 分岐は `req39_place_read_only_fails_closed_when_root` で扱う。
            return;
        }

        let record = freeze_eval_data(b"eval data").expect("失敗しないはず");
        let placement = place_read_only(&src.0, &dest_dir.0, "frozen.jsonl", &record, 1024)
            .expect("配置は成功するはず");

        assert_eq!(placement.mode(), 0o400);
        assert!(!placement.cleanup_failed());
        assert_eq!(placement.path(), dest_dir.0.join("frozen.jsonl"));

        use std::os::unix::fs::MetadataExt as _;
        let dest_meta = std::fs::symlink_metadata(placement.path())
            .expect("配置後のメタデータを取得できるはず");
        assert_eq!((dest_meta.dev(), dest_meta.ino()), placement.dev_ino());

        let content = std::fs::read(placement.path()).expect("読み取りは成功するはず");
        assert_eq!(content, b"eval data");

        match OpenOptions::new().append(true).open(placement.path()) {
            Err(err) => assert_eq!(err.kind(), ErrorKind::PermissionDenied),
            Ok(_) => panic!("append open は拒否されるはず"),
        }
        match std::fs::write(placement.path(), b"tampered") {
            Err(err) => assert_eq!(err.kind(), ErrorKind::PermissionDenied),
            Ok(()) => panic!("truncate write は拒否されるはず"),
        }

        // `dest_dir` には公開したファイルだけが残り、作業ディレクトリは
        // 後始末済み。
        let entries: Vec<_> = std::fs::read_dir(&dest_dir.0)
            .expect("dest_dir を読み取れるはず")
            .filter_map(Result::ok)
            .collect();
        assert_eq!(entries.len(), 1, "dest_dir には公開先だけが残るはず");
    }

    /// REQ-17・REQ-39・TASK-17.2-2: issue #227 の一連の指摘に対する回帰テスト。
    /// `place_read_only` の呼び出し中も `src` と同じ inode を指し続ける
    /// 読み取りハンドルを事前に握っておき、呼び出し後も mode・inode が
    /// 変わっていないことを確認する（`src` に `chmod`・`rename`・`unlink`
    /// のいずれも行わない直接証拠）。
    #[cfg(unix)]
    #[test]
    fn req17_place_read_only_never_touches_src() {
        use std::os::unix::fs::MetadataExt as _;
        use std::os::unix::fs::PermissionsExt as _;

        let dest_dir = make_temp_dir("src-untouched-dest");
        let src = write_unique_temp_file("src-untouched-src", b"eval data");
        if running_as_root(&src.0) {
            return;
        }

        let src_handle = std::fs::File::open(&src.0).expect("src を開けるはず");
        let before = src_handle
            .metadata()
            .expect("src のメタデータを取得できるはず");
        let mode_before = before.permissions().mode() & 0o7777;
        let ino_before = before.ino();
        assert_ne!(mode_before, 0o400, "src の初期 mode は 0o400 ではないはず");

        let record = freeze_eval_data(b"eval data").expect("失敗しないはず");
        let placement = place_read_only(&src.0, &dest_dir.0, "frozen.jsonl", &record, 1024)
            .expect("配置は成功するはず");
        assert_eq!(placement.mode(), 0o400);

        let after = src_handle
            .metadata()
            .expect("ハンドル越しのメタデータ取得に失敗しないはず");
        assert_eq!(
            after.permissions().mode() & 0o7777,
            mode_before,
            "src の mode が変わってはならない"
        );
        let src_meta_after =
            std::fs::symlink_metadata(&src.0).expect("src のメタデータを取得できるはず");
        assert_eq!(
            src_meta_after.ino(),
            ino_before,
            "src の inode が変わってはならない（rename・unlink されていない）"
        );

        let src_content = std::fs::read(&src.0).expect("src の読み取りは成功するはず");
        assert_eq!(src_content, b"eval data");
    }

    /// REQ-17・TASK-17.2-2: `src` の内容がハッシュと食い違うと `HashMismatch` になり、
    /// `dest_dir` には何も作られない（issue #227 P0 指摘: バイト数の一致
    /// だけでは同じ長さの差し替えを見逃す）。
    #[cfg(unix)]
    #[test]
    fn req17_place_read_only_rejects_hash_mismatch_without_creating_dest() {
        let dest_dir = make_temp_dir("hash-mismatch-dest");
        // 元の内容と同じ長さのまま 1 バイトだけ変えた内容を `src` に置く。
        let src = write_unique_temp_file("hash-mismatch-src", b"eval-data");

        let record = freeze_eval_data(b"eval data").expect("失敗しないはず");
        match place_read_only(&src.0, &dest_dir.0, "frozen.jsonl", &record, 1024) {
            Err(PlacementError::HashMismatch) => {}
            other => panic!("HashMismatch を期待したが {other:?} だった"),
        }

        assert!(!dest_dir.0.join("frozen.jsonl").exists());
        let entries: Vec<_> = std::fs::read_dir(&dest_dir.0)
            .expect("dest_dir を読み取れるはず")
            .collect();
        assert!(entries.is_empty(), "失敗時は dest_dir に何も残らないはず");
    }

    /// REQ-39・TASK-17.2-2: 凍結記録が主張するバイト長が `max_bytes` を超えると、
    /// `src` を読み込む前に `TooLarge` として拒否される。
    #[cfg(unix)]
    #[test]
    fn req39_place_read_only_rejects_when_declared_size_exceeds_max_bytes() {
        let dest_dir = make_temp_dir("too-large-dest");
        let src = write_unique_temp_file("too-large-src", b"eval data");

        let record = freeze_eval_data(b"eval data").expect("失敗しないはず");
        let max_bytes = record.byte_len() - 1;
        match place_read_only(&src.0, &dest_dir.0, "frozen.jsonl", &record, max_bytes) {
            Err(PlacementError::TooLarge { size, limit }) => {
                assert_eq!(size, record.byte_len());
                assert_eq!(limit, max_bytes);
            }
            other => panic!("TooLarge を期待したが {other:?} だった"),
        }
        assert!(!dest_dir.0.join("frozen.jsonl").exists());
    }

    /// REQ-17・TASK-17.2-2: 宛先に既にファイルが存在する場合、`place_read_only` は
    /// それを上書きせず `AlreadyExists` を返す（issue #227 P0 指摘:
    /// `rename` による置き換えの禁止）。
    #[cfg(unix)]
    #[test]
    fn req17_place_read_only_does_not_overwrite_existing_dest() {
        let dest_dir = make_temp_dir("already-exists-dest");
        let src = write_unique_temp_file("already-exists-src", b"eval data");
        let dest_path = dest_dir.0.join("frozen.jsonl");
        std::fs::write(&dest_path, b"existing").expect("既存ファイルを作成できるはず");

        let record = freeze_eval_data(b"eval data").expect("失敗しないはず");
        match place_read_only(&src.0, &dest_dir.0, "frozen.jsonl", &record, 1024) {
            Err(PlacementError::AlreadyExists { path }) => assert_eq!(path, dest_path),
            other => panic!("AlreadyExists を期待したが {other:?} だった"),
        }

        let content = std::fs::read(&dest_path).expect("既存ファイルは読み取れるはず");
        assert_eq!(
            content, b"existing",
            "既存ファイルの内容が変わってはならない"
        );
    }

    /// REQ-39・TASK-17.2-2: `file_name` が単一の通常コンポーネントでない場合は
    /// `InvalidFileName` として拒否される（区切り文字・`..`・空文字列・
    /// NUL バイトを含む代表的なケースをまとめて確認する）。
    #[test]
    fn req39_place_read_only_rejects_invalid_file_names() {
        for name in ["", ".", "..", "a/b", "a\\b", "a\0b"] {
            match validate_file_name(name) {
                Err(PlacementError::InvalidFileName) => {}
                other => panic!("{name:?} は InvalidFileName になるはず: {other:?}"),
            }
        }
    }

    /// REQ-39・TASK-17.2-2: `validate_file_name` の拒否が `place_read_only` 全体を
    /// 通しても効くこと（統合的な確認。区切り文字を含む代表例のみ）。
    #[cfg(unix)]
    #[test]
    fn req39_place_read_only_rejects_invalid_file_name_end_to_end() {
        let dest_dir = make_temp_dir("invalid-name-dest");
        let src = write_unique_temp_file("invalid-name-src", b"eval data");
        let record = freeze_eval_data(b"eval data").expect("失敗しないはず");
        match place_read_only(&src.0, &dest_dir.0, "a/b", &record, 1024) {
            Err(PlacementError::InvalidFileName) => {}
            other => panic!("InvalidFileName を期待したが {other:?} だった"),
        }
    }

    /// REQ-39・TASK-17.2-2: 存在しない `src` は `Fs(FsError::Read)` として報告される。
    #[cfg(unix)]
    #[test]
    fn req39_place_read_only_reports_fs_error_for_missing_src() {
        let dest_dir = make_temp_dir("missing-src-dest");
        let missing_src =
            std::env::temp_dir().join("fandhe-edge-data-frozen-placement-missing-src.jsonl");
        let record = freeze_eval_data(b"eval data").expect("失敗しないはず");
        match place_read_only(&missing_src, &dest_dir.0, "frozen.jsonl", &record, 1024) {
            Err(PlacementError::Fs(FsError::Read { .. })) => {}
            other => panic!("Fs(Read) を期待したが {other:?} だった"),
        }
    }

    /// REQ-39・TASK-17.2-2: ディレクトリを `src` に渡すと `Fs(FsError::NotRegularFile)`
    /// になる（FIFO 等での無期限停止と同じ防御。`fandhe_edge_core::fs`）。
    #[cfg(unix)]
    #[test]
    fn req39_place_read_only_reports_fs_error_for_directory_src() {
        let dest_dir = make_temp_dir("dir-src-dest");
        let dir_src = make_temp_dir("dir-src-src");
        let record = freeze_eval_data(b"").expect("失敗しないはず");
        match place_read_only(&dir_src.0, &dest_dir.0, "frozen.jsonl", &record, 1024) {
            Err(PlacementError::Fs(FsError::NotRegularFile { .. })) => {}
            other => panic!("Fs(NotRegularFile) を期待したが {other:?} だった"),
        }
    }

    /// REQ-39・TASK-17.2-2: `dest_dir` が存在しない場合は `Io` エラーになる。
    #[test]
    fn req39_place_read_only_reports_io_error_for_missing_dest_dir() {
        let src = write_unique_temp_file("missing-dest-dir-src", b"eval data");
        let missing_dest_dir =
            std::env::temp_dir().join("fandhe-edge-data-frozen-placement-missing-dest-dir");
        let record = freeze_eval_data(b"eval data").expect("失敗しないはず");
        match place_read_only(&src.0, &missing_dest_dir, "frozen.jsonl", &record, 1024) {
            Err(PlacementError::Io { .. }) => {}
            other => panic!("Io エラーを期待したが {other:?} だった"),
        }
    }

    /// REQ-39・TASK-17.2-2: `dest_dir` が通常ファイルだと `InvalidDestDir` になる。
    #[test]
    fn req39_place_read_only_rejects_dest_dir_that_is_a_regular_file() {
        let not_a_dir = write_unique_temp_file("dest-dir-is-file", b"not a directory");
        let src = write_unique_temp_file("dest-dir-is-file-src", b"eval data");
        let record = freeze_eval_data(b"eval data").expect("失敗しないはず");
        match place_read_only(&src.0, &not_a_dir.0, "frozen.jsonl", &record, 1024) {
            Err(PlacementError::InvalidDestDir { .. }) => {}
            other => panic!("InvalidDestDir を期待したが {other:?} だった"),
        }
    }

    /// REQ-39・TASK-17.2-2: `dest_dir` が symlink だと `InvalidDestDir` になる。
    #[cfg(unix)]
    #[test]
    fn req39_place_read_only_rejects_symlinked_dest_dir() {
        let real_dir = make_temp_dir("symlink-dest-dir-target");
        let src = write_unique_temp_file("symlink-dest-dir-src", b"eval data");
        let link_path = std::env::temp_dir().join(format!(
            "fandhe-edge-data-frozen-placement-unit-{}-dest-dir-symlink-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::os::unix::fs::symlink(&real_dir.0, &link_path).expect("symlink を作成できるはず");

        let record = freeze_eval_data(b"eval data").expect("失敗しないはず");
        let result = place_read_only(&src.0, &link_path, "frozen.jsonl", &record, 1024);
        let _ = std::fs::remove_file(&link_path);

        match result {
            Err(PlacementError::InvalidDestDir { .. }) => {}
            other => panic!("InvalidDestDir を期待したが {other:?} だった"),
        }
    }

    /// REQ-39・TASK-17.2-2: `dest_dir` の mode に group・other 向けの権限
    /// ビットが一つでもあると `InsecureDestDir` になり、何も公開されない
    /// （issue #227 codex[bot] P0 指摘: 評価データの機密性。モジュール doc
    /// 「`dest_dir` の機密性」）。
    #[cfg(unix)]
    #[test]
    fn req39_place_read_only_rejects_insecure_dest_dir_mode() {
        use std::os::unix::fs::PermissionsExt as _;

        let dest_dir = make_temp_dir("insecure-mode-dest");
        std::fs::set_permissions(&dest_dir.0, std::fs::Permissions::from_mode(0o755))
            .expect("dest_dir の権限を意図的に緩められるはず");
        let src = write_unique_temp_file("insecure-mode-src", b"eval data");

        let record = freeze_eval_data(b"eval data").expect("失敗しないはず");
        match place_read_only(&src.0, &dest_dir.0, "frozen.jsonl", &record, 1024) {
            Err(PlacementError::InsecureDestDir { path }) => assert_eq!(path, dest_dir.0),
            other => panic!("InsecureDestDir を期待したが {other:?} だった"),
        }
        assert!(
            !dest_dir.0.join("frozen.jsonl").exists(),
            "insecure な dest_dir には何も公開されないはず"
        );
    }

    /// REQ-17・REQ-39・TASK-17.2-2: 公開直後の検証で、`dest_path` の実体が
    /// 公開したはずの inode と一致しない場合（他者が差し替えた場合）は
    /// `DestReplacedAfterPublish` を返し、そのファイルには触れない（削除
    /// しない）。`place_read_only` 全体の中でこの競合を決定的に再現する
    /// ことはできないため、内部ヘルパー [`verify_and_finalize_publish`]
    /// を直接呼んで確認する（issue #227 P1 指摘）。
    #[cfg(unix)]
    #[test]
    fn req17_req39_verify_and_finalize_publish_detects_replacement_without_deleting() {
        use std::os::unix::fs::MetadataExt as _;

        let dest_dir = make_temp_dir("replaced-after-publish-dest");
        let dest_path = dest_dir.0.join("frozen.jsonl");
        // 「公開されたはず」のステージング inode の (dev, ino) を、実在する
        // 別ファイルの値で代用する（本テストの目的は「一致しない値が
        // 渡されたときに検出できるか」であり、staging 実体そのものの
        // 再現は不要）。
        let other = write_unique_temp_file("replaced-after-publish-staged", b"staged");
        let staged_meta = std::fs::symlink_metadata(&other.0).expect("メタデータを取得できるはず");
        let staged_dev_ino = (staged_meta.dev(), staged_meta.ino());

        // `dest_path` には「他者が置いた」ことにする別ファイルを用意する。
        std::fs::write(&dest_path, b"someone else's file").expect("dest を作成できるはず");

        match verify_and_finalize_publish(
            &StdPlacementDir { dir: &dest_dir.0 },
            "frozen.jsonl",
            staged_dev_ino,
        ) {
            Err(PlacementError::DestReplacedAfterPublish { path }) => {
                assert_eq!(path, PathBuf::from("frozen.jsonl"));
            }
            other => panic!("DestReplacedAfterPublish を期待したが {other:?} だった"),
        }

        let content = std::fs::read(&dest_path).expect("dest はまだ存在するはず（削除されない）");
        assert_eq!(content, b"someone else's file");
    }

    /// REQ-17・REQ-39・TASK-17.2-2: 公開直後の `symlink_metadata` 自体が
    /// 失敗した場合、`dest_path` を削除しようとせず `PublishedButUnverified`
    /// を返す（issue #227 codex[bot] P1 指摘: `dest_path` が本当に自分が
    /// 公開した実体かをこの時点では確認できないため、撤去を試みると
    /// 無関係な別ファイルを削除してしまう競合がありうる）。呼び出し側が
    /// `AlreadyExists` と誤認しないよう、専用の variant で区別する。
    ///
    /// `dest_path` が最初から存在しない場合（`NotFound`）だけでなく、
    /// **実際にはまだ存在するのに `symlink_metadata` が失敗する**状況
    /// （`dest_dir` の実行ビットを一時的に外し `EACCES` を再現する）でも、
    /// `dest_path` の内容が変更・削除されずに残ることを確認する（削除を
    /// 一切試みなくなったことの直接証拠）。
    #[cfg(unix)]
    #[test]
    fn req17_req39_verify_and_finalize_publish_reports_unverified_without_deleting_when_lstat_fails()
     {
        use std::os::unix::fs::PermissionsExt as _;

        let dest_dir = make_temp_dir("lstat-fails-dest");
        let dest_path = dest_dir.0.join("frozen.jsonl");
        std::fs::write(&dest_path, b"still there").expect("dest を作成できるはず");
        if running_as_root(&dest_path) {
            // root は DAC 検査を無視するため、実行ビットを外しても
            // `symlink_metadata` は失敗しない（別テストで扱う分岐ではない
            // ため、この決定的な再現方法自体を諦めて返す）。
            return;
        }
        let staged_dev_ino = (0, 0);

        // `dest_dir` の実行ビットを外し、`dest_path` 自体は存在するまま
        // `symlink_metadata(dest_path)` だけが `EACCES` で失敗する状況を
        // 決定的に再現する。
        std::fs::set_permissions(&dest_dir.0, std::fs::Permissions::from_mode(0o600))
            .expect("dest_dir の権限を一時的に変更できるはず");
        let result = verify_and_finalize_publish(
            &StdPlacementDir { dir: &dest_dir.0 },
            "frozen.jsonl",
            staged_dev_ino,
        );
        // 後始末（`TempDirGuard::drop` が中身を削除できるように）実行ビットを戻す。
        std::fs::set_permissions(&dest_dir.0, std::fs::Permissions::from_mode(0o700))
            .expect("dest_dir の権限を元に戻せるはず");

        match result {
            Err(PlacementError::PublishedButUnverified { path }) => {
                assert_eq!(path, PathBuf::from("frozen.jsonl"));
            }
            other => panic!("PublishedButUnverified を期待したが {other:?} だった"),
        }

        let content = std::fs::read(&dest_path).expect("dest はまだ存在するはず（削除されない）");
        assert_eq!(content, b"still there");
    }

    /// REQ-39・TASK-17.2-2: root 実行下では mode `0o400` でも書き込めるため、
    /// 公開前のプローブで検出され `WriteNotRejected` になり、`dest_dir` には
    /// 何も公開されない（モジュール doc「root・ACL の扱い」）。
    #[cfg(unix)]
    #[test]
    fn req39_place_read_only_fails_closed_when_root() {
        let dest_dir = make_temp_dir("root-dest");
        let src = write_unique_temp_file("root-src", b"eval data");
        if !running_as_root(&src.0) {
            // 非 root 分岐は `req17_req39_place_read_only_success` で扱う。
            return;
        }

        let record = freeze_eval_data(b"eval data").expect("失敗しないはず");
        match place_read_only(&src.0, &dest_dir.0, "frozen.jsonl", &record, 1024) {
            Err(PlacementError::WriteNotRejected { .. }) => {}
            other => panic!("root では WriteNotRejected を期待したが {other:?} だった"),
        }
        assert!(
            !dest_dir.0.join("frozen.jsonl").exists(),
            "root では何も公開されないはず"
        );
    }

    /// REQ-39・TASK-17.2-2: 非 unix では `place_read_only` が常に `UnsupportedPlatform`
    /// を返す（モジュール doc「責務の境界」）。
    #[cfg(not(unix))]
    #[test]
    fn req39_place_read_only_is_unsupported_on_non_unix() {
        let dest_dir = make_temp_dir("non-unix-dest");
        let src = write_unique_temp_file("non-unix-src", b"eval data");
        let record = freeze_eval_data(b"eval data").expect("失敗しないはず");
        match place_read_only(&src.0, &dest_dir.0, "frozen.jsonl", &record, 1024) {
            Err(PlacementError::UnsupportedPlatform { .. }) => {}
            other => panic!("UnsupportedPlatform を期待したが {other:?} だった"),
        }
    }

    /// REQ-39・TASK-17.2-2: `Display` が英語固定で、評価データ本文を含まない。
    #[test]
    fn req39_display_is_english_and_excludes_data_body() {
        let path = PathBuf::from("/tmp/example-eval.jsonl");
        let err = PlacementError::WriteNotRejected { path: path.clone() };
        let text = err.to_string();
        assert!(text.is_ascii());
        assert!(text.contains("writable"));

        // `HashMismatch` はフィールドを持たない unit variant なので、実際の
        // データ本文（テストが使う「tampered」等の具体的な内容）を
        // 構造的に含みえない。ここでは固定の英語メッセージであることだけ
        // 確認する。
        let hash_text = PlacementError::HashMismatch.to_string();
        assert!(hash_text.is_ascii());
        assert!(!hash_text.contains("tampered"));
    }

    /// 書き込み拒否のプローブを常に「書けた」と答える [`PlacementDir`]（root・ACL で拒否できない環境の模擬）。
    #[cfg(unix)]
    struct AlwaysWritable<'a>(StdPlacementDir<'a>);

    #[cfg(unix)]
    impl PlacementDir for AlwaysWritable<'_> {
        fn dir_mode_and_uid(&self) -> io::Result<(u32, u32)> {
            self.0.dir_mode_and_uid()
        }
        fn create_private_dir(&self, name: &str) -> io::Result<()> {
            self.0.create_private_dir(name)
        }
        fn entry_uid(&self, name: &str) -> io::Result<u32> {
            self.0.entry_uid(name)
        }
        fn create_new_file(&self, rel: &str) -> io::Result<std::fs::File> {
            self.0.create_new_file(rel)
        }
        fn probe_append(&self, _rel: &str) -> io::Result<()> {
            Ok(())
        }
        fn publish_no_replace(&self, rel: &str, name: &str) -> io::Result<()> {
            self.0.publish_no_replace(rel, name)
        }
        fn stat_entry(&self, name: &str) -> io::Result<(u64, u64, u32)> {
            self.0.stat_entry(name)
        }
        fn remove_dir_tree(&self, name: &str) -> io::Result<()> {
            self.0.remove_dir_tree(name)
        }
    }

    #[cfg(unix)]
    fn names_in(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .expect("read_dir")
            .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    /// REQ-17・REQ-39: バイト列版は、照合が一致すれば 0400 で公開し、作業領域を残さない（相対パスを返す）。
    #[cfg(unix)]
    #[test]
    fn req17_place_read_only_bytes_publishes_read_only_and_cleans_staging() {
        use std::os::unix::fs::PermissionsExt as _;

        if running_as_root(&std::env::temp_dir()) {
            return; // root は書き込みを拒否できず fail-closed になる（別テストで確認）。
        }
        let dest_dir = make_temp_dir("bytes-ok");
        let bytes = b"{\"id\":\"a\"}\n";
        let record = freeze_eval_data(bytes).expect("record");
        let placement = place_read_only_bytes(
            bytes,
            &StdPlacementDir { dir: &dest_dir.0 },
            "frozen.jsonl",
            &record,
            1024,
        )
        .expect("placed");
        assert_eq!(placement.path(), Path::new("frozen.jsonl"));
        assert_eq!(placement.mode(), 0o400);
        assert!(!placement.cleanup_failed());
        assert_eq!(names_in(&dest_dir.0), ["frozen.jsonl"]);
        let file = dest_dir.0.join("frozen.jsonl");
        assert_eq!(std::fs::read(&file).expect("read"), bytes);
        assert_eq!(
            std::fs::metadata(&file).expect("meta").permissions().mode() & 0o7777,
            0o400
        );
    }

    /// REQ-17・REQ-39: 記録と一致しないバイト列は `HashMismatch` で拒否し、配置先に何も作らない。
    #[cfg(unix)]
    #[test]
    fn req17_place_read_only_bytes_rejects_hash_mismatch_without_touching_dest() {
        let dest_dir = make_temp_dir("bytes-mismatch");
        let record = freeze_eval_data(b"original").expect("record");
        let result = place_read_only_bytes(
            b"tampered",
            &StdPlacementDir { dir: &dest_dir.0 },
            "frozen.jsonl",
            &record,
            1024,
        );
        assert!(
            matches!(result, Err(PlacementError::HashMismatch)),
            "{result:?}"
        );
        assert_eq!(names_in(&dest_dir.0), Vec::<String>::new());
    }

    /// REQ-39: 公開先に既にファイルがあれば置き換えず `AlreadyExists`。既存の内容は不変で、作業領域も残さない。
    #[cfg(unix)]
    #[test]
    fn req39_place_read_only_bytes_does_not_replace_existing_file() {
        if running_as_root(&std::env::temp_dir()) {
            return;
        }
        let dest_dir = make_temp_dir("bytes-exists");
        std::fs::write(dest_dir.0.join("frozen.jsonl"), b"old").expect("existing");
        let record = freeze_eval_data(b"new").expect("record");
        let result = place_read_only_bytes(
            b"new",
            &StdPlacementDir { dir: &dest_dir.0 },
            "frozen.jsonl",
            &record,
            1024,
        );
        assert!(
            matches!(result, Err(PlacementError::AlreadyExists { .. })),
            "{result:?}"
        );
        assert_eq!(
            std::fs::read(dest_dir.0.join("frozen.jsonl")).expect("read"),
            b"old"
        );
        assert_eq!(names_in(&dest_dir.0), ["frozen.jsonl"]);
    }

    /// REQ-39: 書き込み拒否を確認できない（プローブが「書けた」）場合は `WriteNotRejected` で
    /// fail-closed にし、何も公開せず作業領域も残さない。
    #[cfg(unix)]
    #[test]
    fn req39_place_read_only_bytes_fails_closed_when_probe_is_not_rejected() {
        let dest_dir = make_temp_dir("bytes-probe");
        let record = freeze_eval_data(b"data").expect("record");
        let result = place_read_only_bytes(
            b"data",
            &AlwaysWritable(StdPlacementDir { dir: &dest_dir.0 }),
            "frozen.jsonl",
            &record,
            1024,
        );
        assert!(
            matches!(result, Err(PlacementError::WriteNotRejected { .. })),
            "{result:?}"
        );
        assert_eq!(names_in(&dest_dir.0), Vec::<String>::new());
    }
}
