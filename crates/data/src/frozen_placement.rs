//! 凍結済み評価データの読み取り専用配置と、直接書き込みの拒否確認
//! （REQ-39・REQ-17・TASK-17.2-2）。
//!
//! # 呼び出し文脈
//!
//! 将来の CLI `evaluate` 工程（REQ-33・TASK-33.3・issue #140）から、
//! [`crate::eval_freeze::freeze_eval_data`] で凍結記録を作った直後に呼ばれる
//! 想定の順序は「凍結（[`crate::eval_freeze`]）→ 配置（本モジュール）→
//! 評価時の再計算（[`crate::eval_freeze::evaluate_gate`]）」。本モジュールは
//! ハッシュ照合を一切行わない（凍結記録との結び付けは呼び出し側の責務）。
//!
//! # 設計（不変条件）
//!
//! 以前の実装は `path` が指す既存 inode へ直接 `chmod` していたが、それでは
//! nlink 検査（`st_nlink == 1`）と `set_permissions` 呼び出しの間に、同じ
//! inode へのハードリンクを許可ルート外へ追加されると、その `chmod` が
//! ルート外の inode にも及んでしまう（issue #227 codex[bot] P0 指摘）。
//!
//! 本実装は既存 inode への `chmod` を一切行わない。代わりに [`place_read_only`]
//! は次の不変条件を維持する:
//!
//! > **本関数が権限変更（`chmod` 相当）を行う inode は、本関数自身が
//! > `create_new`（`O_EXCL`）で新規作成した inode に限る。その inode は、
//! > 本関数が呼び出し直前に `mkdir(0700)` で作成した非公開ディレクトリの
//! > 中にのみ存在し、本関数が `rename` で最終配置へ移すまで他のどのパスにも
//! > 現れない。**
//!
//! `O_EXCL` は常に新規 inode を保証するため、`mkdir` と `create_new` の間に
//! 親ディレクトリへの書き込み権限を持つ第三者がこの作業ディレクトリを
//! 別物へ差し替えたとしても（親ディレクトリ自体が書き込み可能な場合の
//! 残存リスク。後述）、`create_new` が返す inode は依然として第三者が
//! 事前に用意した既存 inode ではあり得ない（`O_EXCL` は「既存なら失敗」で
//! あり「既存を返す」ことはない）。作成直後に行う `st_nlink == 1` の再検査は、
//! 万一を想定した検出であって、この不変条件そのものを担う主防御ではない。
//!
//! 手順（詳細は次節）:
//!
//! 1. `path` の内容を、新規に `mkdir(0700)` した非公開ディレクトリ内の
//!    `create_new` ファイルへストリームでコピーする（事前に取得した
//!    メタデータのサイズまでに制限する。後述「サイズの扱い」）
//! 2. コピー先（他のどのパスにも露出していない新規 inode）だけを
//!    読み取り専用へ `chmod` する
//! 3. コピー先の nlink が 1 であることを確認する（想定外の共有を検出）
//! 4. `rename` でコピー先を `path` の位置へ原子的に移す（同一ファイル
//!    システム内。symlink の場合は symlink 自体を置き換えるだけでリンク先
//!    には触れない）
//! 5. 配置後の `path` に対して [`verify_direct_write_rejected`] 相当の
//!    非破壊プローブで、実際に書き込みが拒否されることを確認する
//!
//! # サイズの扱い（読み込み前提の変更点）
//!
//! 旧実装は評価データ本体を一切読み込まなかったが、本実装は `path` の内容を
//! ストリームでコピーする（[`std::io::copy`] を固定長バッファで進めるため、
//! 保持するメモリはファイル全体ではなく一定量に収まる）。コピー量は
//! `open` 直後に取得したメタデータのサイズ（`+1` バイト）で打ち切り、
//! それを超えて読み進めない（[`fandhe_edge_core::fs::read_bounded`] と
//! 同じ考え方）。ただし「業務上許容する最大サイズ」という上限方針そのものは
//! 依然として本モジュールの対象外（issue #172）で、ここでの上限はあくまで
//! 「メタデータ取得後にファイルが拡大・差し替えられても際限なく読み進め
//! ない」ための打ち切りに過ぎない。
//!
//! この変更により、[`place_read_only`] の呼び出しには `path` の親
//! ディレクトリへの書き込み・実行権限（一時ディレクトリの作成・
//! ファイルの作成・`rename` に必要）が新たに要る（旧実装は `path` 自体への
//! 権限操作だけで完結していた）。
//!
//! コピー内容の完全性は `std::io::copy` の返す転送バイト数と、事前に
//! 取得したメタデータのサイズとの一致で確認する。sha256 等での再ハッシュ
//! 照合は行わない（[`fandhe_edge_core::hash`] の sha256 実装は共通コア層に
//! 閉じており、任意バイト列のストリームハッシュを取るための公開 API は
//! 現時点で無い。本 crate から `sha2` を直接の依存に追加することは
//! `.claude/rules/dependency-policy.md`「承認済みの依存」表にある配置層
//! （共通コア）を超える新規追加になり、data-builder の裁量を超えるため
//! 見送った。加えて、開いたままのハンドルへ結び付けたストリームハッシュは
//! Linux の `/proc/self/fd` でしか安全に取れず〔macOS の `/dev/fd` は
//! `dup()` 相当でオフセットを共有するため、`io::copy` で末尾まで読み進めた
//! 後に取り直すと 0 バイトしか読めない〕、移植可能な形にできない。
//! コピー内容の完全性照合を厳密にしたい場合は、共通コアへストリーム
//! ハッシュの公開 API を追加するかどうかを main の設計判断とし、承認事項
//! として報告する）。下流の内容整合性は凍結記録の sha256 による事後検知
//! （[`crate::eval_freeze::evaluate_gate`]・issue #49）に委ねる。
//!
//! # 責務の境界（本モジュールが行わないこと）
//!
//! - **経路の閉じ込め（`../`・絶対パス・symlink によるルート外参照の拒否）は
//!   行わない**（TASK-39.x・issue #157/#158 の対象。本モジュールが受け取る
//!   `path` はガード層を通過済みであることを前提とする。crate 全体の前提
//!   条件。`crates/data/src/lib.rs`）。ただし symlink そのものへの権限変更は
//!   本モジュール自身の安全のため拒否する（後述）
//! - **読み込み前のサイズ上限（業務上の最大値）検査は行わない**
//!   （issue #172 の対象）。「サイズの扱い」節のとおり、コピー量は事前
//!   取得サイズまでに打ち切るが、その値自体への上限は課さない
//! - **ハッシュ不一致検知（凍結記録との突き合わせ）は行わない**
//!   （TASK-17.3・issue #49 の対象）
//! - **親ディレクトリ単位の読み取り専用化は行わない**。本関数の呼び出しが
//!   終わった後、その親ディレクトリ自体が書き込み可能であれば、`unlink`・
//!   `rename` による差し替えは防げない（PoC-20 もファイル単位の
//!   `chmod 444` のみを実測している）。差し替えは凍結記録の sha256 による
//!   事後検知（[`crate::eval_freeze::evaluate_gate`]・issue #49）で捕捉する。
//!   同じ理由で、`mkdir(0700)` から `create_new` までの間に親ディレクトリの
//!   書き込み権限を持つ第三者が作業ディレクトリ名を奪い取ろうとしても、
//!   「不変条件」節のとおり `O_EXCL` が新規 inode を保証するため、この
//!   モジュールが chmod する対象が既存の（ルート外を含む）inode に
//!   すり替わることはない
//! - **root 実行下での書き込み防止は保証しない**。root は mode `0o444` でも
//!   書き込めるため、[`place_read_only`] は書き込みを防げない配置を
//!   「配置済み」と装わず、[`PlacementError::WriteNotRejected`] で
//!   fail-closed に失敗させる（後述「root・ACL の扱い」）
//! - **ハードリンクされたコピー元ファイルへの配置は拒否する**（unix。
//!   `st_nlink != 1`）。以前は「chmod がルート外 inode へ及ぶことを防ぐ」
//!   ための検査だったが、本実装ではコピー元を chmod しないためその意味は
//!   なくなった。代わりに「凍結対象のコピー元が単独の実体でない（別の
//!   パスからも書き換えられうる可能性がある）場合は凍結を拒否する」という
//!   保守的な判断として維持する（コピー中にもう一方のリンク経由で内容が
//!   書き換えられる競合を避ける）
//! - **非 unix では読み取り専用配置そのものを拒否する**
//!   （[`PlacementError::UnsupportedPlatform`]）。Windows には
//!   `(dev, ino)` 相当の安価な同一性検査手段が無く、検査用ハンドルを
//!   閉じてパス経由で権限変更する実装は検査後の差し替え（TOCTOU）を
//!   防げない（issue #227 codex[bot] P1 指摘）。ハンドルに結び付けた
//!   権限変更（Win32 API・`unsafe` FFI が必要）は M10 時点で対象外の
//!   OS 向けの実装として見送り、「実装済みを装わない」
//!   （`.claude/rules/coding-rust.md`）ため fail-closed に拒否する
//! - **CLI の出力 JSON 全体の形は決めない**。入出力契約は TASK-33.3 に委ねる
//!
//! # 手順（[`place_read_only`]）
//!
//! 1. `symlink_metadata` で symlink・非通常ファイルを拒否する（リンク先が
//!    評価ディレクトリ外かもしれないファイルの権限を、本関数の副作用で
//!    書き換えないため）
//! 2. [`fandhe_edge_core::fs::open_regular_file_for_read`] でコピー元を開く
//!    （非 unix ではここで [`PlacementError::UnsupportedPlatform`] として
//!    拒否する）
//! 3. unix では、手順 1 の `(dev, ino)` と開いたハンドルのそれを突き合わせ、
//!    検査からオープンまでの差し替え（TOCTOU）を検出する。続けて
//!    ハンドルの `st_nlink` が 1 であることを確認し、ハードリンクされた
//!    ファイル（内容が別パス経由で書き換えられうる）への凍結を拒否する
//! 4. `path` の親ディレクトリの中に `mkdir(0700)` で非公開の作業
//!    ディレクトリを新規作成し、その中に `create_new` で新規ファイルを作る
//! 5. コピー元の内容を、事前取得したサイズまでに制限してストリームで
//!    新規ファイルへコピーする（「サイズの扱い」節）
//! 6. 新規ファイル（他のどのパスにも露出していない）だけを権限
//!    `0o444`（読み取り専用・setuid/setgid/sticky・実行ビットなし）に
//!    変更する（unix のみ。非 unix は手順 2 で既に拒否済み）
//! 7. 新規ファイルの nlink が 1 であることを再確認する
//! 8. `rename` で新規ファイルを `path` の位置へ原子的に移す
//! 9. [`verify_direct_write_rejected`] 相当の非破壊プローブで、配置後の
//!    `path` への直接書き込みが実際に拒否されることを確認できた場合に
//!    限り成功とする（fail-closed）
//!
//! # 書き込みプローブの TOCTOU 対策（Linux）
//!
//! Linux では、手順 6 で権限変更したのと同じ（まだ開いたままの）ハンドルを
//! そのまま使い、その書き込みプローブ（`append` での再オープン）をパス
//! 経由ではなく `/proc/self/fd/<fd>` というカーネル提供の疑似シンボリック
//! リンク経由で行う（[`reopen_append_via_fd`]）。これによりディレクトリ
//! エントリの再探索が発生しないため、`rename` の前後を通じて一貫して
//! 同じ inode を対象にできる。macOS の `/dev/fd`（`fdescfs`）は同種の
//! 疑似シンボリックリンクに見えるが `dup()` 相当の実装であり、要求した
//! フラグに関わらず元のディスクリプタのアクセスモードを越える再オープンを
//! 拒否するため、対象ファイルの実際の権限を検査できない（issue #227
//! cursor[bot] 指摘: Write probe broken on macOS。実際に CI の
//! macOS 実行で「書き込み可能なはずのファイルへの直接書き込みが拒否と
//! 誤判定される」形で顕在化した）。そのため macOS を含む Linux 以外の
//! unix・非 unix 環境（M10 時点で macOS 以外は検証対象外）では、この fd
//! 直参照が使えないためパス再オープン＋再検査のベストエフォート対策に
//! 留める（[`verify_identity_after_permission_denied`]）。
//!
//! # root・ACL の扱い（安全側に倒した判断）
//!
//! mode ビットの設定だけでは、root（`CAP_DAC_OVERRIDE`）・書き込みを許す
//! POSIX ACL・mode を無視するファイルシステムのいずれでも書き込みを防げない
//! ことがある。[`place_read_only`] は mode を設定した「つもり」で終わらず、
//! 必ず [`verify_direct_write_rejected`] の実際の書き込み試行で拒否を
//! 確認し、拒否を確認できなければ [`PlacementError::WriteNotRejected`] を
//! 返して fail-closed にする（`.claude/rules/coding-rust.md`
//! 「未実装・簡易実装の箇所は実装済みを装わない」）。root 実行を許すか
//! どうかの方針は CLI 配線（TASK-33.3）で決めるべき事項とする。
//!
//! # セキュリティ上の注意
//!
//! [`PlacementError`] の `Display` は英語固定で、評価データ本文を含めない
//! （`.claude/rules/security.md`）。本モジュールはパスをシェル・子プロセスへ
//! 渡さない（`PathBuf` のみで扱う。テストの FIFO 作成のみ例外的に子プロセス
//! `mkfifo` を使う）。
//!
//! # 出典
//!
//! [`verify_direct_write_rejected`] の非破壊プローブ（`append` モードで
//! 開いて即座に閉じる）は PoC-20 ケース 5
//! （`docs/spec/03-poc/safety-hardening/scripts/case5_eval_integrity.py`）の
//! `open(eval_path, "a")` を移植したもの。PoC は `chmod 444` した評価
//! ファイルを追記モードで開くと `PermissionError`（errno 13）になることを
//! 実測している（証拠の種別: テストハーネス）。

use fandhe_edge_core::fs::{FsError, open_regular_file_for_read};
use std::fmt;
use std::fs::OpenOptions;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

/// [`place_read_only`] が成功した証跡（読み取り専用配置が成立したことを
/// 表す）。フィールドは非公開にし、[`place_read_only`] を通してしか作れない
/// （形だけの「配置済み」偽装を防ぐ）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadOnlyPlacement {
    // unix のみ。非 unix では `place_read_only` が
    // `PlacementError::UnsupportedPlatform` を返すため本型は構築されない
    // （モジュール doc「責務の境界」）。
    mode: u32,
    // 配置確認時点（`place_read_only` 内の最終検査）の `(dev, ino)`。
    // 呼び出し側が「この戻り値が指す実体は今もこれか」を後から突き合わせ
    // られるようにする（issue #227 codex[bot] P0 指摘: 戻り値だけでは
    // 呼び出し元が使うパスの読み取り専用状態を保証できないため、検証手段
    // 自体を戻り値に持たせる）。関数の返り値が確定した「その瞬間」の実体を
    // 表すのみで、返り値を受け取った後の差し替え（TOCTOU）はこのフィールド
    // だけでは検出できない。その事後検知は凍結記録の sha256
    // （[`crate::eval_freeze::evaluate_gate`]・issue #49）に委ねる。
    //
    // 本関数は毎回 `create_new` で新規 inode を作ってから `rename` で
    // `path` へ配置するため（モジュール doc「設計（不変条件）」）、同じ
    // `path` に対して 2 回呼んでも `dev_ino` は毎回異なる（「冪等」とは
    // 「もう一度呼んでも成功し mode 0o444 が得られる」という意味であり、
    // 「同じ inode が返る」という意味ではない）。
    dev_ino: (u64, u64),
}

impl ReadOnlyPlacement {
    /// 配置後の unix パーミッションビット（`mode & 0o7777`）。
    #[must_use]
    pub fn mode(&self) -> u32 {
        self.mode
    }

    /// 配置確認時点の `(dev, ino)`。呼び出し側が後から
    /// `std::fs::symlink_metadata` 等で同じ実体かどうかを突き合わせるために
    /// 公開する（issue #227 codex[bot] P0 指摘への対応）。同じ `path` に
    /// 対して [`place_read_only`] を複数回呼ぶと、呼ぶたびに新しい inode が
    /// 作られるため値は変わる（構造体 doc参照）。
    #[must_use]
    pub fn dev_ino(&self) -> (u64, u64) {
        self.dev_ino
    }
}

/// [`place_read_only`]・[`verify_direct_write_rejected`] が失敗する理由。
#[derive(Debug)]
#[non_exhaustive]
pub enum PlacementError {
    /// パス先が symlink だった。リンク先（評価ディレクトリ外かもしれない
    /// ファイル）への副作用を避けるため、権限変更を一切行わずに拒否する。
    Symlink { path: PathBuf },
    /// パス先が通常ファイルではない（ディレクトリ・FIFO・ソケット等）。
    NotRegularFile { path: PathBuf },
    /// 検査時（`symlink_metadata`）とオープン後のハンドルとで実体が異なる
    /// （`(dev, ino)` 不一致。unix 限定の TOCTOU 対策）。検査からオープンの
    /// 間にパスが差し替えられた可能性がある。
    Replaced { path: PathBuf },
    /// 権限を読み取り専用へ変更した後も、実際の書き込み試行
    /// （[`verify_direct_write_rejected`]）が拒否されなかった。
    ///
    /// root（`CAP_DAC_OVERRIDE`）・書き込みを許す ACL・mode を無視する
    /// ファイルシステム等、mode ビットの設定だけでは書き込みを防げない
    /// 状況を検出する（fail-closed。モジュール doc「root・ACL の扱い」）。
    WriteNotRejected { path: PathBuf },
    /// コピー元がハードリンクされている（`st_nlink != 1`。unix 限定）か、
    /// 本関数が新規作成したコピー先が想定外に複数リンクを持っていた
    /// （後者は原理上起きないはずの検出用チェック。モジュール doc
    /// 「設計（不変条件）」）。
    ///
    /// コピー元について: 別パスからも書き換えられうる内容を凍結すると、
    /// コピー中に内容が変化する競合を許してしまうため、単独の実体
    /// （リンク数 1）であることを確認できない限り凍結を拒否する
    /// （fail-closed。issue #227 codex[bot] P0 指摘を受けて、以前の
    /// 「chmod の副作用防止」目的から「凍結対象の単独性確認」目的へ
    /// 意味を改めた）。
    HardLinked { path: PathBuf, nlink: u64 },
    /// 非 unix プラットフォームでは読み取り専用配置そのものを拒否する。
    ///
    /// Windows には `(dev, ino)` 相当の安価な同一性検査手段が無く、検査用
    /// ハンドルを閉じてパス経由で権限変更すると検査後の差し替え
    /// （TOCTOU）を防げない（issue #227 codex[bot] P1 指摘）。ハンドルに
    /// 結び付けた権限変更には Win32 API 呼び出し（`unsafe` FFI）が要るが
    /// M10 時点で Windows は対象外 OS のため実装せず、「実装済みを装わ
    /// ない」（`.claude/rules/coding-rust.md`）方針で fail-closed に拒否
    /// する。将来 Windows 対応する際は、ハンドルに結び付けた同一性検査
    /// 付きの実装に置き換える（TASK-17.2-2 の将来仕様）。
    UnsupportedPlatform { path: PathBuf },
    /// 本モジュールの前提条件（通常ファイル判定・開いての読み込み）を
    /// 満たせなかった（[`fandhe_edge_core::fs`] 由来）。
    Fs(FsError),
    /// 権限変更・メタデータ取得・コピー・`rename` などの I/O エラー
    /// （コピー時のサイズ不一致の検出を含む。モジュール doc「サイズの
    /// 扱い」）。
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
}

impl fmt::Display for PlacementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlacementError::Symlink { path } => {
                write!(
                    f,
                    "{} is a symlink, refusing to change permissions",
                    path.display()
                )
            }
            PlacementError::NotRegularFile { path } => {
                write!(f, "{} is not a regular file", path.display())
            }
            PlacementError::Replaced { path } => write!(
                f,
                "{} was replaced between the pre-check and opening it",
                path.display()
            ),
            PlacementError::WriteNotRejected { path } => write!(
                f,
                "{} is still writable after setting read-only permissions",
                path.display()
            ),
            PlacementError::HardLinked { path, nlink } => {
                write!(
                    f,
                    "{} has {nlink} hard links, refusing to freeze it",
                    path.display()
                )
            }
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
            PlacementError::Symlink { .. }
            | PlacementError::NotRegularFile { .. }
            | PlacementError::Replaced { .. }
            | PlacementError::WriteNotRejected { .. }
            | PlacementError::HardLinked { .. }
            | PlacementError::UnsupportedPlatform { .. } => None,
        }
    }
}

impl From<FsError> for PlacementError {
    fn from(source: FsError) -> Self {
        PlacementError::Fs(source)
    }
}

/// symlink・非通常ファイルを拒否する事前検査（[`place_read_only`]・
/// [`verify_direct_write_rejected`] の両方が使う共通の入口）。
///
/// `std::fs::symlink_metadata` はリンクを辿らないため、symlink 自身の種別を
/// 判定できる（`std::fs::metadata` はリンクを辿ってしまい、リンク先が
/// 通常ファイルであれば symlink であることを見逃す）。
///
/// 呼び出し側が TOCTOU 検査（`open` 後の `(dev, ino)` 突き合わせ）に使える
/// よう、取得したメタデータをそのまま返す（このメタデータを捨てて後で
/// 取り直すと、検査からオープンまでの間の差し替えを見逃す。issue #227
/// レビュー指摘）。
fn check_not_symlink(path: &Path) -> Result<std::fs::Metadata, PlacementError> {
    let meta = std::fs::symlink_metadata(path).map_err(|source| PlacementError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    if meta.file_type().is_symlink() {
        return Err(PlacementError::Symlink {
            path: path.to_path_buf(),
        });
    }
    if !meta.file_type().is_file() {
        return Err(PlacementError::NotRegularFile {
            path: path.to_path_buf(),
        });
    }
    Ok(meta)
}

#[cfg(unix)]
fn dev_ino(meta: &std::fs::Metadata) -> (u64, u64) {
    use std::os::unix::fs::MetadataExt as _;
    (meta.dev(), meta.ino())
}

/// ハードリンク数（`st_nlink`）を返す（unix 限定。[`PlacementError::HardLinked`]
/// の判定に使う）。
#[cfg(unix)]
fn nlink_count(meta: &std::fs::Metadata) -> u64 {
    use std::os::unix::fs::MetadataExt as _;
    meta.nlink()
}

/// [`place_read_only`] が使う非公開の作業領域（unix 限定）。
///
/// `dir` は `path` の親ディレクトリの中に `mkdir(0700)` で新規作成した
/// ディレクトリで、他のどの利用者からも到達できない（実行ビットが無い
/// ため、ディレクトリ名を知っていてもトラバースできない）。`file_path` は
/// その中に `create_new`（`O_EXCL`）で作る予定のファイルパスで、この
/// ディレクトリの外へ一切公開されない限り、他プロセスがこの inode への
/// ハードリンクを作ることはできない（モジュール doc「設計（不変条件）」）。
#[cfg(unix)]
struct StagingArea {
    dir: PathBuf,
    file_path: PathBuf,
}

#[cfg(unix)]
impl Drop for StagingArea {
    fn drop(&mut self) {
        // `place_read_only` が成功した経路では、コピー先ファイルは既に
        // `rename` で `path` へ移動済みのため、このディレクトリは空である。
        // 失敗した経路では中身（コピー途中のファイル）ごと削除する。
        //
        // この削除は「後始末」であって「原状回復」ではない点が、issue #227
        // codex[bot] P0 指摘が問題にした旧実装の `let _ = file.set_permissions(...)`
        // （外部・ルート外 inode の権限を元に戻そうとして失敗を無視していた）
        // とは性質が異なる。このディレクトリは作成された瞬間から 0700 で
        // 他者から到達できない内輪の作業領域であり、削除に失敗して
        // 残ったとしても外部（ルート外・呼び出し元が把握するパス）へは
        // 一切影響しない。そのため削除失敗を `place_read_only` 全体の
        // エラーへ混ぜず、ベストエフォートに留める。
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// `parent` の中に `mkdir(0700)` で非公開の作業ディレクトリを新規作成する
/// （unix 限定）。
///
/// `DirBuilderExt::mode` は `mkdir(2)` 呼び出し自体に渡すモードのため、
/// 「作成した瞬間から 0700」であることが保証される（作成後に別途
/// `chmod` する実装だと、その間だけ既定のモードで晒される窓ができる）。
#[cfg(unix)]
fn create_staging_area(parent: &Path) -> Result<StagingArea, PlacementError> {
    use std::os::unix::fs::DirBuilderExt as _;

    let pid = std::process::id();
    for attempt in 0..1000u32 {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = parent.join(format!(
            ".fandhe-edge-frozen-staging-{pid}-{attempt}-{nanos}"
        ));
        let mut builder = std::fs::DirBuilder::new();
        builder.mode(0o700);
        match builder.create(&dir) {
            Ok(()) => {
                let file_path = dir.join("frozen");
                return Ok(StagingArea { dir, file_path });
            }
            Err(err) if err.kind() == ErrorKind::AlreadyExists => continue,
            Err(source) => {
                return Err(PlacementError::Io { path: dir, source });
            }
        }
    }
    Err(PlacementError::Io {
        path: parent.to_path_buf(),
        source: std::io::Error::other("failed to create a unique staging directory"),
    })
}

/// 評価データ本体を読み取り専用配置にする（REQ-39・REQ-17・TASK-17.2-2）。
///
/// モジュール doc「設計（不変条件）」「手順」を参照。unix では冪等（既に
/// 読み取り専用のファイルに対して呼んでも成功する。ただし呼ぶたびに新しい
/// inode を作り直すため `dev_ino` は変わる。[`ReadOnlyPlacement`] doc
/// 参照）。非 unix では常に [`PlacementError::UnsupportedPlatform`] を返す
/// （モジュール doc「責務の境界」）。
pub fn place_read_only(path: &Path) -> Result<ReadOnlyPlacement, PlacementError> {
    let pre_meta = check_not_symlink(path)?;

    let file = open_regular_file_for_read(path)?;

    #[cfg(unix)]
    {
        use std::io::{Read as _, Write as _};
        use std::os::unix::fs::OpenOptionsExt as _;
        use std::os::unix::fs::PermissionsExt as _;

        // `pre_meta` は `check_not_symlink` が `open` より前に取得した
        // symlink_metadata（このスコープに入る前に別のファイルへ差し替え
        // られていないかの基準）。ここで新たに `symlink_metadata` を取り
        // 直すと、その取り直し自体が新しい TOCTOU 窓になる（issue #227
        // codex P0 指摘）ため、必ず `check_not_symlink` が返したメタデータ
        // を使う。
        let opened_meta = file.metadata().map_err(|source| PlacementError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        if dev_ino(&pre_meta) != dev_ino(&opened_meta) {
            return Err(PlacementError::Replaced {
                path: path.to_path_buf(),
            });
        }

        // コピー元がハードリンクされている（`st_nlink != 1`）場合は凍結を
        // 拒否する。別パス経由で内容が書き換えられうる実体をコピー中に
        // 参照すると、コピーの完全性を保証できないため（モジュール doc
        // 「責務の境界」。issue #227 codex[bot] P0 指摘を受けて、以前の
        // 「chmod の副作用防止」目的から意味を改めた）。
        let source_nlink = nlink_count(&opened_meta);
        if source_nlink != 1 {
            return Err(PlacementError::HardLinked {
                path: path.to_path_buf(),
                nlink: source_nlink,
            });
        }

        // `path` の親ディレクトリの中に、他のどのパスからも到達できない
        // 非公開の作業領域を作る（モジュール doc「設計（不変条件）」）。
        // 同一ファイルシステム内であることを保証するため、無関係な
        // ディレクトリ（`std::env::temp_dir()` 等）ではなく `path` と
        // 同じ親ディレクトリを使う（`rename` はファイルシステムをまたげ
        // ない）。
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        let staging = create_staging_area(parent)?;

        let mut staging_file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&staging.file_path)
            .map_err(|source| PlacementError::Io {
                path: path.to_path_buf(),
                source,
            })?;

        // コピー量は `open` 直後に取得した `opened_meta` のサイズ（+1
        // バイト）で打ち切る。メタデータ取得後にコピー元が拡大・差し替え
        // られても、読み進める量自体をここで頭打ちにする
        // （`fandhe_edge_core::fs::read_bounded` と同じ考え方。モジュール
        // doc「サイズの扱い」）。
        let expected_len = opened_meta.len();
        let mut bounded_source = (&file).take(expected_len.saturating_add(1));
        let copied = std::io::copy(&mut bounded_source, &mut staging_file).map_err(|source| {
            PlacementError::Io {
                path: path.to_path_buf(),
                source,
            }
        })?;
        if copied != expected_len {
            return Err(PlacementError::Io {
                path: path.to_path_buf(),
                source: std::io::Error::other(format!(
                    "copied {copied} bytes but expected {expected_len} bytes"
                )),
            });
        }
        staging_file
            .flush()
            .and_then(|()| staging_file.sync_all())
            .map_err(|source| PlacementError::Io {
                path: path.to_path_buf(),
                source,
            })?;

        // ここまでで書き込んだのは、他のどのパスからも到達できない新規
        // inode（`staging.file_path`）のみ。ここから先で権限変更するのは
        // この inode 一つだけであり、既存の（ルート外かもしれない）inode
        // には一切触れない（モジュール doc「設計（不変条件）」）。
        const READ_ONLY_MODE: u32 = 0o444;
        staging_file
            .set_permissions(std::fs::Permissions::from_mode(READ_ONLY_MODE))
            .map_err(|source| PlacementError::Io {
                path: path.to_path_buf(),
                source,
            })?;

        let staging_meta = staging_file
            .metadata()
            .map_err(|source| PlacementError::Io {
                path: path.to_path_buf(),
                source,
            })?;
        let mode = staging_meta.permissions().mode() & 0o7777;
        if mode != READ_ONLY_MODE {
            return Err(PlacementError::Io {
                path: path.to_path_buf(),
                source: std::io::Error::other(format!(
                    "permissions did not converge to {READ_ONLY_MODE:o} (got {mode:o})"
                )),
            });
        }

        // 新規作成した inode が想定外に複数リンクを持っていないかの検出用
        // 再確認（原理上は起こり得ないはずだが、モジュール doc「不変条件」
        // の想定が崩れていないことを最後まで確認する）。
        let staging_nlink = nlink_count(&staging_meta);
        if staging_nlink != 1 {
            return Err(PlacementError::HardLinked {
                path: path.to_path_buf(),
                nlink: staging_nlink,
            });
        }
        let confirmed_dev_ino = dev_ino(&staging_meta);

        // 読み取り専用にした新規ファイルを `path` の位置へ原子的に移す。
        // 同一ファイルシステム内の `rename` はアトミックで、`path` が
        // symlink であってもそのリンク自体を置き換えるだけでリンク先には
        // 触れない。
        std::fs::rename(&staging.file_path, path).map_err(|source| PlacementError::Io {
            path: path.to_path_buf(),
            source,
        })?;

        // `rename` 直後、`path` の実体が期待どおり配置したばかりの inode で
        // あることを確認する（`rename` 自体はアトミックだが、この関数が
        // 返り値を確定する直前にもう一段確認しておく。これでも本関数が
        // `Ok` を返した「後」の差し替えまでは防げない。その事後検知は
        // 凍結記録の sha256〔[`crate::eval_freeze::evaluate_gate`]・
        // issue #49〕に委ねる）。
        let final_meta = std::fs::symlink_metadata(path).map_err(|source| PlacementError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        if final_meta.file_type().is_symlink() {
            return Err(PlacementError::Symlink {
                path: path.to_path_buf(),
            });
        }
        if dev_ino(&final_meta) != confirmed_dev_ino {
            return Err(PlacementError::Replaced {
                path: path.to_path_buf(),
            });
        }

        #[cfg(target_os = "linux")]
        {
            // 手順 6 で権限変更したのと同じ、まだ開いたままのハンドル
            // `staging_file` を使い、`/proc/self/fd/<fd>` 経由で書き込み
            // プローブを行う。この inode は既に `rename` で `path` の位置に
            // ある実体そのものであり、パス経由の再探索を経由しないため
            // `rename` 前後を通じて対象がぶれない（issue #227 codex[bot]
            // P0 指摘の TOCTOU 対策）。
            match reopen_append_via_fd(&staging_file) {
                Err(err) if err.kind() == ErrorKind::PermissionDenied => {}
                Ok(_opened) => {
                    return Err(PlacementError::WriteNotRejected {
                        path: path.to_path_buf(),
                    });
                }
                Err(source) => {
                    return Err(PlacementError::Io {
                        path: path.to_path_buf(),
                        source,
                    });
                }
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            // macOS 等（Linux 以外の unix）では `/dev/fd` 等のハンドル直参照
            // 手法が使えない、または信頼できない（macOS の `/dev/fd` は
            // dup 相当で、元のディスクリプタのアクセスモードを越える
            // 再オープンを要求フラグに関わらず拒否し、かつオフセットを
            // 共有するため、コピー完了後の `staging_file` から読み直しても
            // 0 バイトしか得られない。issue #227 cursor[bot] 指摘: Write
            // probe broken on macOS）。そのためパスを開き直すベストエフォート
            // の確認に留める。`confirmed_dev_ino` を期待値として渡し、
            // 権限変更後・書き込み確認前にパスが別のファイルへ差し替え
            // られていないかも突き合わせる（issue #227 codex P1 指摘）。
            verify_direct_write_rejected_checked(path, Some(confirmed_dev_ino))?;
        }

        Ok(ReadOnlyPlacement {
            mode,
            dev_ino: confirmed_dev_ino,
        })
    }

    #[cfg(not(unix))]
    {
        // 非 unix（Windows）では読み取り専用配置そのものを拒否する
        // （[`PlacementError::UnsupportedPlatform`]。モジュール doc
        // 「手順」・「責務の境界」参照）。検査用ハンドルを閉じてパス経由で
        // 権限変更する実装は、検査（`check_not_symlink`）とオープン
        // （`open_regular_file_for_read`）の後に path が symlink や
        // 別ファイルへ差し替えられても検出できない（issue #227
        // codex[bot] P1 指摘）。Windows には unix の `(dev, ino)` に相当する
        // 安価な同一性検査手段が無く、ハンドルに結び付けた権限変更には
        // `unsafe` な Win32 API 呼び出しが要るため、M10 時点で対象外の
        // Windows 向けに未検証の実装を「実装済みを装う」形で残さず、
        // fail-closed に拒否する。
        let _ = (&pre_meta, file);
        Err(PlacementError::UnsupportedPlatform {
            path: path.to_path_buf(),
        })
    }
}

/// 直接の書き込み試行が権限エラーで拒否されることを、非破壊的に確認する
/// （REQ-39・TASK-17.2-2。出典: PoC-20 ケース 5）。
///
/// `OpenOptions::new().append(true)` のみを使い `create`・`truncate` は
/// 付けない。開けても 1 バイトも書かずに即座に閉じるため、呼び出し前後で
/// ファイルの内容・mtime は変わらない。
///
/// 単独呼び出し（`expected_dev_ino` なし）でも、事前検査
/// （`check_not_symlink`）で得た `(dev, ino)` と実際に書き込みを試みた
/// ハンドルのそれを突き合わせ、検査からオープンの間の差し替えを検出する
/// （unix 限定。issue #227 cursor[bot] 指摘: TOCTOU check uses post-open
/// stat）。
pub fn verify_direct_write_rejected(path: &Path) -> Result<(), PlacementError> {
    verify_direct_write_rejected_checked(path, None)
}

/// `append` 用の `open` が `PermissionDenied` を返した直後に、検査時点
/// （[`check_not_symlink`]）の `(dev, ino)`（`pre_dev_ino`）と、現在の
/// パスの実体が同一であることを確認する（unix 限定）。
///
/// `PermissionDenied` というだけでは、検査時と同一のファイルへの拒否
/// なのか、検査後に「別の読み取り専用ファイル」へ差し替えられ、その
/// 別ファイルへの拒否を検査対象への拒否と誤認しているのかを区別
/// できない（issue #227 codex P0 指摘。fail-closed の TOCTOU 対策）。
/// `open` 呼び出し直後に改めて `symlink_metadata` を取得し `(dev, ino)`
/// が検査時と一致することを確認できて初めて、この拒否を検査対象の
/// ファイルへの拒否とみなす。これでも「失敗した `open`」と「直後の
/// 再検査」の間の窓は残る。Linux では [`reopen_append_via_fd`]
/// がこの窓自体を構造的に閉じるため、本関数は Linux 以外の環境（macOS を
/// 含む。macOS の `/dev/fd` が書き込みプローブに使えない理由は
/// [`fd_magic_path`] のドキュメント参照。M10 時点で macOS 以外は検証
/// 対象外。`.claude/rules/coding-rust.md`「クロスプラットフォーム」）
/// 向けのベストエフォートな代替としてのみ使う
/// （[`verify_direct_write_rejected_checked`] のフォールバック分岐）。
#[cfg(all(unix, not(target_os = "linux")))]
fn verify_identity_after_permission_denied(
    path: &Path,
    pre_dev_ino: (u64, u64),
) -> Result<(), PlacementError> {
    let post_meta = check_not_symlink(path)?;
    if dev_ino(&post_meta) != pre_dev_ino {
        return Err(PlacementError::Replaced {
            path: path.to_path_buf(),
        });
    }
    Ok(())
}

/// `/proc/self/fd/<fd>`（Linux 限定）を返す。パス上のディレクトリエントリを
/// 再探索せず、既に開いているファイルディスクリプタが指す実体を直接指す
/// カーネル提供の疑似シンボリックリンクである。このパスを `open` すると、
/// 対象の inode そのものを（元のパスを一切介さずに）別のアクセスモードで
/// 開き直せる。
///
/// macOS の `/dev/fd`（`fdescfs`）は同種の見た目を持つが `dup()` 相当の
/// 実装で、要求したフラグに関わらず元のディスクリプタのアクセスモードを
/// 越える再オープンを拒否する（対象ファイルの実際の権限を検査できない。
/// issue #227 cursor[bot] 指摘: Write probe broken on macOS）。そのため
/// macOS 向けの `fd_magic_path` は用意しない。macOS は
/// [`verify_identity_after_permission_denied`] によるパス再オープン＋
/// 再検査のベストエフォート経路を使う。
#[cfg(target_os = "linux")]
fn fd_magic_path(fd: std::os::unix::io::RawFd) -> PathBuf {
    PathBuf::from(format!("/proc/self/fd/{fd}"))
}

/// 事前に開いた読み取り専用ハンドル `read_handle` が指す実体そのものを、
/// [`fd_magic_path`] 経由で `append` モードとして開き直す（Linux 限定。
/// 上記 `fd_magic_path` のドキュメント参照）。パスの再探索を一切行わない
/// ため、`read_handle` を得た後にパス上のエントリが別ファイルへ
/// 差し替えられても、この再オープンは影響を受けない（issue #227 codex P0
/// 指摘: `append` の `open` が `PermissionDenied` を返した後、再検査までの
/// 間にファイルが差し替えられると、`(dev, ino)` が一致していても実際に
/// 拒否されたのが別ファイルだった可能性がある、という TOCTOU 窓を構造的に
/// 閉じる）。
#[cfg(target_os = "linux")]
fn reopen_append_via_fd(read_handle: &std::fs::File) -> std::io::Result<std::fs::File> {
    use std::os::unix::io::AsRawFd as _;
    OpenOptions::new()
        .append(true)
        .open(fd_magic_path(read_handle.as_raw_fd()))
}

/// [`verify_direct_write_rejected`] の内部実装。`expected_dev_ino` が
/// `Some` の場合（[`place_read_only`] からの呼び出し）、事前検査で得た
/// `(dev, ino)` と本関数が改めて得た `(dev, ino)` の一致も要求する。これに
/// より、権限変更後・本確認前にパスが別のファイルへ差し替えられていた
/// 場合も差し替え前のファイルと同一であることを確認できなければ
/// `Replaced` で拒否する（issue #227 codex P1 指摘）。
///
/// Linux では、書き込みプローブそのものを [`reopen_append_via_fd`]
/// で行い、対象を識別してから書き込みを試みるまでの間、パス経由の再
/// オープンを一切行わない（issue #227 codex P0 指摘の TOCTOU 対策。上記
/// ドキュメント参照）。Linux 以外の unix・非 unix 環境（macOS を含む。
/// `/dev/fd` が使えない理由は [`fd_magic_path`] のドキュメント参照。M10
/// 時点で macOS 以外は検証対象外。`.claude/rules/coding-rust.md`
/// 「クロスプラットフォーム」）では、パスを直接開いて `PermissionDenied`
/// の直後に再検査するベストエフォートの対策に留める
/// （[`verify_identity_after_permission_denied`]）。
fn verify_direct_write_rejected_checked(
    path: &Path,
    #[cfg_attr(not(unix), allow(unused_variables))] expected_dev_ino: Option<(u64, u64)>,
) -> Result<(), PlacementError> {
    let pre_meta = check_not_symlink(path)?;
    #[cfg(unix)]
    let pre_dev_ino = dev_ino(&pre_meta);
    #[cfg(not(unix))]
    let _ = &pre_meta;

    #[cfg(unix)]
    if let Some(expected) = expected_dev_ino
        && pre_dev_ino != expected
    {
        return Err(PlacementError::Replaced {
            path: path.to_path_buf(),
        });
    }

    #[cfg(target_os = "linux")]
    {
        // 読み取り専用ハンドルを開いて対象の inode に結び付ける。この
        // ハンドルの (dev, ino) が検査時と一致することを確認できて
        // 初めて、以後の書き込みプローブ（`reopen_append_via_fd`）を
        // 「検査対象そのもの」への操作とみなせる。
        let read_handle = open_regular_file_for_read(path)?;
        let opened_meta = read_handle
            .metadata()
            .map_err(|source| PlacementError::Io {
                path: path.to_path_buf(),
                source,
            })?;
        if dev_ino(&opened_meta) != pre_dev_ino {
            return Err(PlacementError::Replaced {
                path: path.to_path_buf(),
            });
        }

        match reopen_append_via_fd(&read_handle) {
            Err(err) if err.kind() == ErrorKind::PermissionDenied => Ok(()),
            Ok(_opened) => Err(PlacementError::WriteNotRejected {
                path: path.to_path_buf(),
            }),
            Err(source) => Err(PlacementError::Io {
                path: path.to_path_buf(),
                source,
            }),
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        match OpenOptions::new().append(true).open(path) {
            Err(err) if err.kind() == ErrorKind::PermissionDenied => {
                #[cfg(unix)]
                verify_identity_after_permission_denied(path, pre_dev_ino)?;
                Ok(())
            }
            Ok(opened) => {
                // 開けてしまった（＝書き込み可能）場合でも、それが検査時と
                // 同一のファイルであることを確認してから `WriteNotRejected`
                // として報告する。差し替え後の別ファイルが書き込み可能
                // だっただけなら `Replaced` で区別する。
                #[cfg(unix)]
                {
                    let opened_meta = opened.metadata().map_err(|source| PlacementError::Io {
                        path: path.to_path_buf(),
                        source,
                    })?;
                    if dev_ino(&opened_meta) != pre_dev_ino {
                        return Err(PlacementError::Replaced {
                            path: path.to_path_buf(),
                        });
                    }
                }
                #[cfg(not(unix))]
                {
                    drop(opened);
                }
                Err(PlacementError::WriteNotRejected {
                    path: path.to_path_buf(),
                })
            }
            Err(source) => Err(PlacementError::Io {
                path: path.to_path_buf(),
                source,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// テスト用の一時ファイルを、成否に関わらず削除するガード（RAII）。
    /// 削除の前に権限を書き込み可へ戻す（読み取り専用のままだと環境に
    /// よっては削除できないため、また後始末を確実にするため）。
    ///
    /// unix の permission bit（`0o444`）に依存するテストからしか使わない
    /// ため `#[cfg(unix)]`。windows では未使用となり clippy `-D warnings`
    /// で fail する（issue #227 CI 指摘）。
    #[cfg(unix)]
    struct TempFileGuard(PathBuf);

    #[cfg(unix)]
    impl Drop for TempFileGuard {
        fn drop(&mut self) {
            use std::os::unix::fs::PermissionsExt as _;
            let _ = std::fs::set_permissions(&self.0, std::fs::Permissions::from_mode(0o644));
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[cfg(unix)]
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

    /// REQ-39・TASK-17.2-2: 非 root 環境で、読み取り専用配置後に直接の
    /// 書き込み試行（append・truncate 双方）が権限エラーで拒否される。
    #[cfg(unix)]
    #[test]
    fn req39_place_read_only_rejects_direct_writes_when_not_root() {
        let guard = write_unique_temp_file("accept-basic", b"eval data");
        if running_as_root(&guard.0) {
            // root 分岐は別テスト（`req39_place_read_only_fails_closed_when_root`）で扱う。
            return;
        }

        let placement = place_read_only(&guard.0).expect("非 root では配置が成功するはず");
        assert_eq!(placement.mode(), 0o444);

        match OpenOptions::new().append(true).open(&guard.0) {
            Err(err) => assert_eq!(err.kind(), ErrorKind::PermissionDenied),
            Ok(_) => panic!("append open は拒否されるはず"),
        }
        match std::fs::write(&guard.0, b"tampered") {
            Err(err) => assert_eq!(err.kind(), ErrorKind::PermissionDenied),
            Ok(()) => panic!("truncate write は拒否されるはず"),
        }

        // 内容が変わっていないこと（読み取りは読み取り専用でも成功する）。
        let content = std::fs::read(&guard.0).expect("読み取りは成功するはず");
        assert_eq!(content, b"eval data");
    }

    /// REQ-39・TASK-17.2-2: root 実行下では mode `0o444` でも書き込めるため、
    /// `place_read_only` は `WriteNotRejected` で fail-closed に失敗する
    /// （モジュール doc「root・ACL の扱い」）。
    #[cfg(unix)]
    #[test]
    fn req39_place_read_only_fails_closed_when_root() {
        let guard = write_unique_temp_file("root-branch", b"eval data");
        if !running_as_root(&guard.0) {
            // 非 root 分岐は `req39_place_read_only_rejects_direct_writes_when_not_root` で扱う。
            return;
        }

        match place_read_only(&guard.0) {
            Err(PlacementError::WriteNotRejected { .. }) => {}
            other => panic!("root では WriteNotRejected を期待したが {other:?} だった"),
        }
    }

    /// REQ-39: `place_read_only` は冪等（2 回呼んでも成功し mode が
    /// `0o444` のまま変わらない）。本実装は呼ぶたびに新規 inode を作り
    /// `rename` で配置し直すため、`dev_ino` は毎回変わる（[`ReadOnlyPlacement`]
    /// doc 参照。以前の「同じ inode を chmod するだけ」の実装では
    /// `dev_ino` も一致していたが、issue #227 codex[bot] P0 指摘への対応で
    /// 既存 inode への chmod をやめたことに伴う意味の変更）。
    #[cfg(unix)]
    #[test]
    fn req39_place_read_only_is_idempotent() {
        let guard = write_unique_temp_file("idempotent", b"eval data");
        if running_as_root(&guard.0) {
            return;
        }

        let first = place_read_only(&guard.0).expect("1 回目は成功するはず");
        let second = place_read_only(&guard.0).expect("2 回目も成功するはず");
        assert_eq!(first.mode(), 0o444);
        assert_eq!(second.mode(), 0o444);
        assert_ne!(
            first.dev_ino(),
            second.dev_ino(),
            "呼ぶたびに新規 inode を作り直すため dev_ino は変わるはず"
        );
        let content = std::fs::read(&guard.0).expect("読み取りは成功するはず");
        assert_eq!(content, b"eval data");
    }

    /// REQ-39: 書き込み可能なファイル（所有者に書き込み権限がある通常の
    /// mode）へ `verify_direct_write_rejected` を呼ぶと、root かどうかに
    /// 関わらず `WriteNotRejected` になる（所有者の書き込み権限は root で
    /// なくても append open を成功させるため、root 分岐で期待値を変える
    /// 理由がない。以前のテストは `result.is_ok()`〔＝書き込み拒否〕を
    /// root の「append open に成功するはず」という逆の主張の根拠にして
    /// おり期待値が反転していた。issue #227 cursor[bot] 指摘: Root test
    /// inverts write-rejection result）。
    #[cfg(unix)]
    #[test]
    fn req39_verify_direct_write_rejected_on_writable_file() {
        let guard = write_unique_temp_file("writable-probe", b"eval data");
        match verify_direct_write_rejected(&guard.0) {
            Err(PlacementError::WriteNotRejected { .. }) => {}
            other => panic!("WriteNotRejected を期待したが {other:?} だった"),
        }
        // 非破壊プローブ: append open が成功しても中身は変わらない。
        assert_eq!(
            std::fs::read(&guard.0).expect("読み取りに失敗しないはず"),
            b"eval data"
        );
    }

    /// REQ-39・TASK-17.2-2: `append` open が `PermissionDenied` を返した
    /// 場合でも、事前検査で得た `(dev, ino)`（`pre_dev_ino`）と現在のパス
    /// の実体が一致しなければ `Replaced` として拒否する（issue #227 codex
    /// P0 指摘: 書き込み拒否確認は `(dev, ino)` 未確認で `Ok` を返す）。
    ///
    /// 実際の TOCTOU（`check_not_symlink` と `open` の間の差し替え）を
    /// 再現する代わりに、`place_read_only` で読み取り専用配置済みの
    /// ファイル A の `pre_dev_ino` を握った状態で、パス上のファイルを
    /// 別の読み取り専用ファイル B へ差し替えてから呼び出すことで、
    /// 「開けなかった対象が検査時のファイルと異なる」状況を決定的に
    /// 再現する。
    ///
    /// `verify_identity_after_permission_denied` は Linux では
    /// フォールバック分岐でも使わなくなった（[`super::reopen_append_via_fd`]
    /// が TOCTOU 窓を構造的に閉じるため）。macOS は `/dev/fd` が書き込み
    /// プローブに使えない（issue #227 cursor[bot] 指摘）ためこの
    /// フォールバックを使い続ける。本関数の定義と同じ cfg でゲートし、
    /// macOS を含む Linux 以外の unix 環境向けのテストとして残す。
    #[cfg(all(unix, not(target_os = "linux")))]
    #[test]
    fn req39_verify_identity_after_permission_denied_detects_replacement() {
        use std::os::unix::fs::MetadataExt as _;

        let guard_a = write_unique_temp_file("permdenied-replace-a", b"file a");
        if running_as_root(&guard_a.0) {
            // root では append open 自体が PermissionDenied にならない。
            return;
        }
        let placement = place_read_only(&guard_a.0).expect("A の読み取り専用配置は成功するはず");
        assert_eq!(placement.mode(), 0o444);
        let pre_meta =
            std::fs::symlink_metadata(&guard_a.0).expect("A のメタデータを取得できるはず");
        let pre_dev_ino = (pre_meta.dev(), pre_meta.ino());

        // A とは別 inode の読み取り専用ファイル B を用意し、A のパスへ
        // rename で差し替える（検査からオープンの間の差し替えと同じ
        // 効果を、決定的な手順で再現する）。
        let guard_b = write_unique_temp_file("permdenied-replace-b", b"file b");
        let placement_b = place_read_only(&guard_b.0).expect("B の読み取り専用配置は成功するはず");
        assert_eq!(placement_b.mode(), 0o444);
        let b_meta = std::fs::symlink_metadata(&guard_b.0).expect("B のメタデータを取得できるはず");
        assert_ne!(
            (b_meta.dev(), b_meta.ino()),
            pre_dev_ino,
            "A と B の (dev, ino) は異なるはず"
        );
        std::fs::rename(&guard_b.0, &guard_a.0).expect("B を A のパスへ差し替えられるはず");
        // rename 済みで guard_b のパスはもう存在しないため、Drop での
        // 後始末を A 側の guard に任せる（guard_b の Drop は空振りになる）。
        std::mem::forget(guard_b);

        match verify_identity_after_permission_denied(&guard_a.0, pre_dev_ino) {
            Err(PlacementError::Replaced { .. }) => {}
            other => panic!("Replaced を期待したが {other:?} だった"),
        }
    }

    /// REQ-39・TASK-17.2-2: `reopen_append_via_fd` は、対象への読み取り専用
    /// ハンドルを取得した後にパス上のファイルが（書き込み可能な）別実体へ
    /// 差し替えられても、その差し替え後のファイルではなく、ハンドルが
    /// 指す元の実体（読み取り専用）を対象に書き込みを試みる（issue #227
    /// codex P0 指摘: `append` の `open` が `PermissionDenied` を返した後、
    /// 再検査までにファイルが差し替えられると `(dev, ino)` の再検査だけ
    /// では別ファイルへの拒否を見逃しうる、という TOCTOU 窓そのものを
    /// 構造的に閉じることを示す）。Linux 限定（macOS では
    /// `reopen_append_via_fd` 自体を用意していない。issue #227 cursor[bot]
    /// 指摘: Write probe broken on macOS）。
    #[cfg(target_os = "linux")]
    #[test]
    fn req39_reopen_append_via_fd_ignores_path_replacement_after_handle_open() {
        let guard_a = write_unique_temp_file("fdbound-replace-a", b"file a");
        if running_as_root(&guard_a.0) {
            // root では append open 自体が PermissionDenied にならない。
            return;
        }
        let placement = place_read_only(&guard_a.0).expect("A の読み取り専用配置は成功するはず");
        assert_eq!(placement.mode(), 0o444);

        // `verify_direct_write_rejected_checked` が内部で行う手順のうち、
        // 「識別済みハンドルを取得する」ところまでを模する。
        let read_handle = open_regular_file_for_read(&guard_a.0).expect("A を開けるはず");

        // ハンドル取得後にパス上のファイルを書き込み可能な B へ差し替える
        // （検査後・書き込みプローブ前の TOCTOU 窓を決定的な手順で再現する）。
        let guard_b = write_unique_temp_file("fdbound-replace-b", b"file b");
        std::fs::rename(&guard_b.0, &guard_a.0).expect("B を A のパスへ差し替えられるはず");
        std::mem::forget(guard_b);

        // fd 直参照はパスの再探索を行わないため、差し替え後の書き込み
        // 可能な B ではなく、依然として読み取り専用の A を対象にする。
        match reopen_append_via_fd(&read_handle) {
            Err(err) => assert_eq!(err.kind(), ErrorKind::PermissionDenied),
            Ok(_) => panic!("fd 直参照は差し替え後の書き込み可能な B の影響を受けないはず"),
        }
    }

    /// REQ-39・TASK-17.2-2: issue #227 codex[bot] P0 指摘の回帰テスト。
    /// `place_read_only` 呼び出し前から握り続けている、コピー元ファイルへの
    /// 読み取りハンドル越しに見える mode が、呼び出し後も変わっていない
    /// ことを確認する（＝コピー元の既存 inode を `chmod` していない直接
    /// 証拠）。あわせて、`path` が指す実体が呼び出し前とは異なる inode
    /// （新規に作って `rename` で配置した実体）に変わっていることも確認する。
    ///
    /// 旧実装（既存 inode へ直接 `chmod` する実装）に対して本テストを
    /// 実行すると、握り続けたハンドル越しの mode が `0o444` に変わって
    /// しまい、かつ `path` の inode 番号も変化しないため失敗する。
    #[cfg(unix)]
    #[test]
    fn req39_place_read_only_never_chmods_original_inode() {
        use std::os::unix::fs::MetadataExt as _;
        use std::os::unix::fs::PermissionsExt as _;

        let guard = write_unique_temp_file("original-inode-untouched", b"eval data");
        if running_as_root(&guard.0) {
            // root は mode に関わらず書き込めるため、他テストで扱う分岐と
            // 重複させない。
            return;
        }

        // `place_read_only` の呼び出し中もコピー元と同じ inode を指し
        // 続ける読み取りハンドルを事前に握っておく。
        let original_handle = std::fs::File::open(&guard.0).expect("元ファイルを開けるはず");
        let original_meta_before = original_handle
            .metadata()
            .expect("元ファイルのメタデータを取得できるはず");
        let original_mode_before = original_meta_before.permissions().mode() & 0o7777;
        let original_ino = original_meta_before.ino();
        assert_ne!(
            original_mode_before, 0o444,
            "元ファイルの初期 mode は 0o444 ではないはず"
        );

        let placement = place_read_only(&guard.0).expect("配置は成功するはず");
        assert_eq!(placement.mode(), 0o444);

        // 握り続けていたハンドル越しに見える元 inode の mode が変わって
        // いないこと（chmod が及んでいない直接証拠）。
        let original_meta_after = original_handle
            .metadata()
            .expect("ハンドル越しのメタデータ取得に失敗しないはず");
        assert_eq!(
            original_meta_after.permissions().mode() & 0o7777,
            original_mode_before,
            "place_read_only は元の inode を chmod してはならない"
        );

        // `path` は `rename` により新しい inode（読み取り専用配置後の実体）
        // を指すようになっており、握り続けていた元 inode とは異なる。
        let final_ino = std::fs::symlink_metadata(&guard.0)
            .expect("配置後のメタデータを取得できるはず")
            .ino();
        assert_ne!(
            final_ino, original_ino,
            "place_read_only は新しい inode へ差し替えるはず（既存 inode への chmod ではない）"
        );

        // 内容は保持されている（コピーが正しく行われたこと）。
        let content = std::fs::read(&guard.0).expect("読み取りは成功するはず");
        assert_eq!(content, b"eval data");
    }

    /// REQ-39: ディレクトリを渡すと `NotRegularFile` になる。
    #[test]
    fn req39_place_read_only_rejects_directory() {
        let dir = std::env::temp_dir().join(format!(
            "fandhe-edge-data-frozen-placement-unit-{}-dir-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir(&dir).expect("テスト用ディレクトリを作成できるはず");

        let result = place_read_only(&dir);
        std::fs::remove_dir(&dir).expect("テスト用ディレクトリを削除できるはず");

        match result.expect_err("ディレクトリは通常ファイルではないため拒否されるはず")
        {
            PlacementError::NotRegularFile { .. } => {}
            other => panic!("NotRegularFile を期待したが {other:?} だった"),
        }
    }

    /// REQ-39: 存在しないパスを渡すと `Io` エラーになる（`symlink_metadata`
    /// の失敗として検出される）。
    #[test]
    fn req39_place_read_only_reports_io_error_for_missing_path() {
        let missing =
            std::env::temp_dir().join("fandhe-edge-data-frozen-placement-does-not-exist.jsonl");
        match place_read_only(&missing) {
            Err(PlacementError::Io { .. }) => {}
            other => panic!("Io エラーを期待したが {other:?} だった"),
        }
    }

    /// REQ-39: symlink を渡すと `Symlink` として拒否され、リンク先の権限は
    /// 変更されない（`../` 等のルート外参照を防ぐガード層とは別に、本
    /// モジュール自身もリンク先への副作用を避ける）。
    #[cfg(unix)]
    #[test]
    fn req39_place_read_only_rejects_symlink_without_touching_target() {
        let target = write_unique_temp_file("symlink-target", b"eval data");
        let link_path = std::env::temp_dir().join(format!(
            "fandhe-edge-data-frozen-placement-unit-{}-symlink-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::os::unix::fs::symlink(&target.0, &link_path).expect("symlink を作成できるはず");

        let result = place_read_only(&link_path);
        let _ = std::fs::remove_file(&link_path);

        match result.expect_err("symlink は拒否されるはず") {
            PlacementError::Symlink { .. } => {}
            other => panic!("Symlink を期待したが {other:?} だった"),
        }

        // リンク先の mode が元のまま（書き込み可）であることを確認する。
        use std::os::unix::fs::PermissionsExt as _;
        let target_mode = std::fs::metadata(&target.0)
            .expect("リンク先のメタデータを取得できるはず")
            .permissions()
            .mode()
            & 0o7777;
        assert_ne!(
            target_mode, 0o444,
            "symlink 拒否がリンク先の権限を変えてはならない"
        );
    }

    /// REQ-39: ハードリンクされたファイルを渡すと `HardLinked` として
    /// 拒否され、リンク先（同一 inode を指すもう一方のパス）の権限も
    /// 変更されない（issue #227 codex[bot] P0 指摘。コピー元が単独の
    /// 実体でないと凍結の完全性を保証できないための検査であることを
    /// 確認する。モジュール doc「責務の境界」）。
    #[cfg(unix)]
    #[test]
    fn req39_place_read_only_rejects_hard_linked_file() {
        let original = write_unique_temp_file("hardlink-original", b"eval data");
        let link_path = std::env::temp_dir().join(format!(
            "fandhe-edge-data-frozen-placement-unit-{}-hardlink-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::hard_link(&original.0, &link_path).expect("ハードリンクを作成できるはず");

        let result = place_read_only(&link_path);
        let _ = std::fs::remove_file(&link_path);

        match result.expect_err("ハードリンクされたファイルは拒否されるはず") {
            PlacementError::HardLinked { nlink, .. } => assert_eq!(nlink, 2),
            other => panic!("HardLinked を期待したが {other:?} だった"),
        }

        // 元ファイル（同一 inode）の mode が元のまま（書き込み可）で
        // あることを確認する（本モジュールの副作用がルート外実体へ
        // 及んでいないことの直接証拠）。
        use std::os::unix::fs::PermissionsExt as _;
        let original_mode = std::fs::metadata(&original.0)
            .expect("元ファイルのメタデータを取得できるはず")
            .permissions()
            .mode()
            & 0o7777;
        assert_ne!(
            original_mode, 0o444,
            "ハードリンク拒否が元ファイルの権限を変えてはならない"
        );
    }

    /// REQ-39: 非 unix では `place_read_only` が常に `UnsupportedPlatform`
    /// を返す（issue #227 codex[bot] P1 指摘。ハンドルに結び付けない
    /// パス経由の権限変更は TOCTOU を防げないため fail-closed に拒否する
    /// 方針。モジュール doc「責務の境界」）。
    #[cfg(not(unix))]
    #[test]
    fn req39_place_read_only_is_unsupported_on_non_unix() {
        let path = std::env::temp_dir().join(format!(
            "fandhe-edge-data-frozen-placement-unit-{}-non-unix-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&path, b"eval data").expect("テスト用ファイルを作成できるはず");

        let result = place_read_only(&path);
        let _ = std::fs::remove_file(&path);

        match result.expect_err("非 unix では常に拒否されるはず") {
            PlacementError::UnsupportedPlatform { .. } => {}
            other => panic!("UnsupportedPlatform を期待したが {other:?} だった"),
        }
    }

    /// REQ-39: FIFO（名前付きパイプ）を渡しても無期限に停止せず
    /// `NotRegularFile`（`open_regular_file_for_read` 経由）になる。
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn req39_place_read_only_rejects_fifo_without_blocking() {
        let path = std::env::temp_dir().join(format!(
            "fandhe-edge-data-frozen-placement-unit-{}-fifo-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let status = std::process::Command::new("mkfifo")
            .arg(&path)
            .status()
            .expect("mkfifo コマンドを起動できるはず");
        assert!(status.success(), "mkfifo が成功するはず");

        let result = place_read_only(&path);
        std::fs::remove_file(&path).expect("FIFO を削除できるはず");

        match result.expect_err("FIFO は通常ファイルではないため拒否されるはず")
        {
            PlacementError::NotRegularFile { .. }
            | PlacementError::Fs(FsError::NotRegularFile { .. }) => {}
            other => panic!("NotRegularFile を期待したが {other:?} だった"),
        }
    }

    /// REQ-39: `Display` が英語固定で、評価データ本文を含まない。
    #[test]
    fn req39_display_is_english_and_excludes_data_body() {
        let path = PathBuf::from("/tmp/example-eval.jsonl");
        let err = PlacementError::WriteNotRejected { path: path.clone() };
        let text = err.to_string();
        assert!(text.is_ascii());
        assert!(text.contains("writable"));
    }
}
