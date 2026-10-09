//! 評価データ（正解ラベルを含む本体）の評価前後ハッシュ比較・凍結記録との接続
//! （REQ-27「評価の独立性」・REQ-17「データの分割と凍結」・TASK-27.1-2・issue #70）。
//!
//! CLI の `evaluate` 工程（REQ-33・#314 で接続済み）から、データ契約層
//! （`fandhe-edge-data`）が凍結した評価データの sha256・バイト長を受け取り、
//! 実際の評価処理（`eval` クロージャ）を前後のディスク再読み込み＋ハッシュ比較で
//! 包むために呼ばれる想定。[`invariance`][crate::invariance]（モデルパッケージ側。
//! TASK-27.1-1・issue #69）と対になる評価データ側の実装で、同じ
//! [`fandhe_edge_core::hash::Sha256Digest`] とスナップショット比較の形を使う。
//!
//! # 処理順（評価契約の中核。この順序を固定する）
//!
//! 1. [`fandhe_edge_core::fs::read_bounded`] で評価データを **1 回だけ** 読む
//! 2. 読んだバイト列から sha256・バイト長を計算し、[`FrozenEvalData`]（呼び出し側が
//!    データ契約層の凍結記録から作る期待値）と照合する。不一致なら
//!    [`EvalDataInvarianceError::FrozenRecordMismatch`] を返し、**`eval` を呼ばない**
//!    （TASK-17.3「凍結後にハッシュが変化した状態で評価を実行しようとしたら警告なく
//!    実行せず停止」と同じ条件。モデルパッケージ側〔[`crate::invariance`]〕と異なり
//!    評価データ側は呼び出し側が期待値を持っているため、評価前の時点でも停止できる）
//! 3. `eval(&bytes)` を実行する（ハッシュを取った時点で検証済みのバイト列そのものを
//!    渡すため、ハッシュ計算と評価実行の間に差し替えの余地が無い）
//! 4. **`bytes` を drop してから** [`fandhe_edge_core::fs::sha256_file_bounded`] で
//!    ディスクを読み直す（評価前バッファと評価後の読み直しを同時に保持しない。
//!    [`crate::invariance`] の「構成要素の生バイト列を二重に保持しない」設計と同じ
//!    観点。`eval` が `Err` を返しても、この手順は必ず実行する）
//! 5. 評価前ダイジェストと評価後ダイジェストが不一致なら
//!    [`EvalDataInvarianceError::ChangedDuringEvaluation`] を返し、`eval` の結果
//!    （`Ok` であっても）は捨てる（fail-closed）
//! 6. 一致すれば `eval` の結果を返す（`Err` は
//!    [`EvalDataInvarianceError::Evaluation`] として包む）
//!
//! 評価後の照合は sha256 の一致のみで行い、バイト長を別途照合しない。
//! [`fandhe_edge_core::fs::sha256_file_bounded`] はダイジェストしか返さず、
//! sha256 の一致は内容（＝長さ）の一致を含意する。評価後にもう一度
//! `read_bounded` で全体を読み直すとメモリ使用量が最大 2 倍になるため使わない。
//!
//! # `eval` へ渡すもの（責務の境界）
//!
//! `eval` クロージャが受け取る `&[u8]` は正解ラベルを含む評価データ本体そのもので、
//! 評価器側の経路（本モジュール）を通る。「推論関数には `input` だけを渡す」
//! （`.claude/rules/evaluation-contract.md`「評価の独立性」・TASK-27.2）は別の話で、
//! 推論関数へ渡す情報を絞るのは TASK-27.2（[`crate::input_only`]）の責務であり、本モジュールは
//! 評価データ本体の完全性（改変されていないこと）だけを保証する。
//!
//! # 資源上限
//!
//! 1 ファイルあたり [`MAX_EVAL_DATA_BYTES`] まで（暫定値。REQ-39 の資源上限が
//! 正式に決まり次第見直す。`crates/eval/src/invariance.rs` の
//! `MAX_MODEL_COMPONENT_BYTES` と同じ位置づけの暫定値）。通常ファイル判定・
//! サイズ上限・TOCTOU 対策（`O_NONBLOCK`）は共通コアの [`fandhe_edge_core::fs`]
//! に委ね、本モジュールでは複製しない。経路の閉じ込め（`../` 等の拒否）・
//! 形式の許可リストは呼び出し側のガード層（REQ-39・パス未確定）の責務。
//!
//! # 現状（実装済みを装わない）
//!
//! - 凍結記録との不一致・改変検出を**呼び出し側へ返す**ところまでが本モジュールの
//!   範囲。停止分岐の本体（台帳との突き合わせ・来歴の記録）は
//!   `fandhe_edge_data::eval_freeze`（TASK-17.2-1・issue #47）が既に実装しており、
//!   台帳連携（TASK-17.3・issue #49）は本モジュールの対象外
//! - 終了コード（[`fandhe_edge_core::exitcode::ExitCode`]）への写像は行わない
//!   （`fandhe_edge_data::eval_freeze::FreezeError` の doc と同様、TASK-33.3 に委ねる）
//! - CLI `evaluate` 工程へは #314 で接続済み（`stages::evaluate` が [`FrozenEvalData`] を組み立てる）

use fandhe_edge_core::fs::{self, FsError};
use fandhe_edge_core::hash::Sha256Digest;
use std::fmt;
use std::path::{Path, PathBuf};

/// [`evaluate_with_eval_data_invariance`] が 1 ファイルあたりに読み込むバイト数の
/// 上限（暫定値。REQ-39「読み込むファイル 1GB」目安。issue #172 で正式な資源上限が
/// 決まり次第見直す）。
pub const MAX_EVAL_DATA_BYTES: u64 = 1024 * 1024 * 1024;

/// 凍結済み評価データの所在と期待値（呼び出し側が
/// `fandhe_edge_data::eval_freeze::FreezeRecord::sha256()`/`byte_len()` から
/// 組み立てる想定）。
///
/// パスと期待するダイジェスト・バイト長の組を 1 つの型にまとめることで、
/// 「パスだけ渡して期待値を照合し忘れる」誤用をコンパイル時の必須引数で防ぐ
/// （`.claude/rules/coding-rust.md`「公開 API・型設計」）。
#[derive(Debug, Clone, Copy)]
pub struct FrozenEvalData<'a> {
    /// 評価データ本体ファイルのパス。
    pub path: &'a Path,
    /// 凍結記録に残っている sha256 の期待値。
    pub sha256: Sha256Digest,
    /// 凍結記録に残っているバイト長の期待値。
    pub byte_len: u64,
}

/// [`evaluate_with_eval_data_invariance`] が返しうるエラー。評価クロージャ自身の
/// エラー型 `E` を包んで一緒に返せるようにする。
///
/// `#[non_exhaustive]` にして、将来バリアントが増えても外部 crate の `match` を
/// 壊さないようにする（[`crate::invariance::EvaluationInvarianceError`] と同じ方針）。
#[derive(Debug)]
#[non_exhaustive]
pub enum EvalDataInvarianceError<E> {
    /// 評価データファイルの読み込み（評価前の 1 回読み込み、または評価後の
    /// 読み直し）に失敗した。
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    /// 評価データファイルが [`MAX_EVAL_DATA_BYTES`] を超えていた。
    TooLarge {
        path: PathBuf,
        size: u64,
        limit: u64,
    },
    /// 評価データのパス先が通常ファイルではなかった（FIFO・ソケット・
    /// ディレクトリ等。REQ-39）。
    NotRegularFile { path: PathBuf },
    /// バイト列の長さが `u64` の範囲を超えていた（64bit 環境では実質的に
    /// 到達しないが、`as` キャストを使わず `u64::try_from` で明示的に変換する
    /// 規約〔`.claude/rules/coding-rust.md`〕に従うため `Result` として扱う）。
    LengthOverflow,
    /// 評価前: 実データから計算した sha256・バイト長が [`FrozenEvalData`] の
    /// 期待値と一致しなかった。fail-closed のため **`eval` を呼ばない**
    /// （TASK-17.3「凍結後にハッシュが変化した状態で評価を実行しようとしたら
    /// 警告なく実行せず停止」と同じ条件）。
    FrozenRecordMismatch {
        expected_sha256: Sha256Digest,
        actual_sha256: Sha256Digest,
        expected_byte_len: u64,
        actual_byte_len: u64,
    },
    /// 評価後: `eval` の実行中（またはその前後）に評価データファイルの内容が
    /// 変化した。fail-closed のため、`eval` の結果（`Ok` であっても）は
    /// 呼び出し側へ返さない。
    ChangedDuringEvaluation {
        before: Sha256Digest,
        after: Sha256Digest,
    },
    /// 評価クロージャ自体がエラーを返した。評価データに変化が無かったことは
    /// 確認済みで、評価そのものの失敗であることを示す。
    Evaluation(E),
}

impl<E: fmt::Display> fmt::Display for EvalDataInvarianceError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvalDataInvarianceError::Read { path, source } => write!(
                f,
                "failed to read eval data at {}: {source}",
                path.display()
            ),
            EvalDataInvarianceError::TooLarge { path, size, limit } => write!(
                f,
                "eval data at {} exceeds size limit ({size} > {limit} bytes)",
                path.display()
            ),
            EvalDataInvarianceError::NotRegularFile { path } => {
                write!(f, "eval data at {} is not a regular file", path.display())
            }
            EvalDataInvarianceError::LengthOverflow => {
                write!(f, "eval data byte length does not fit in u64")
            }
            EvalDataInvarianceError::FrozenRecordMismatch {
                expected_sha256,
                actual_sha256,
                expected_byte_len,
                actual_byte_len,
            } => write!(
                f,
                "eval data does not match frozen record (expected sha256={expected_sha256} byte_len={expected_byte_len}, actual sha256={actual_sha256} byte_len={actual_byte_len})"
            ),
            EvalDataInvarianceError::ChangedDuringEvaluation { before, after } => write!(
                f,
                "eval data changed during evaluation (before sha256={before}, after sha256={after})"
            ),
            EvalDataInvarianceError::Evaluation(err) => write!(f, "evaluation failed: {err}"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for EvalDataInvarianceError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            EvalDataInvarianceError::Read { source, .. } => Some(source),
            EvalDataInvarianceError::Evaluation(err) => Some(err),
            EvalDataInvarianceError::TooLarge { .. }
            | EvalDataInvarianceError::NotRegularFile { .. }
            | EvalDataInvarianceError::LengthOverflow
            | EvalDataInvarianceError::FrozenRecordMismatch { .. }
            | EvalDataInvarianceError::ChangedDuringEvaluation { .. } => None,
        }
    }
}

/// [`fandhe_edge_core::fs::FsError`] を [`EvalDataInvarianceError`] へ写す
/// （[`crate::invariance::hash_component`] と同じ方針。`FsError` は
/// `#[non_exhaustive]` のため、未知のバリアントは fail-open で無視せず
/// `Read` として fail-closed に扱う）。
fn map_fs_error<E>(path: &Path, err: FsError) -> EvalDataInvarianceError<E> {
    match err {
        FsError::Read { path, source } => EvalDataInvarianceError::Read { path, source },
        FsError::TooLarge { path, size, limit } => {
            EvalDataInvarianceError::TooLarge { path, size, limit }
        }
        FsError::NotRegularFile { path } => EvalDataInvarianceError::NotRegularFile { path },
        other => EvalDataInvarianceError::Read {
            path: path.to_path_buf(),
            source: std::io::Error::other(other.to_string()),
        },
    }
}

/// [`evaluate_with_eval_data_invariance`] の本体。上限をテストから制御できるよう
/// `limit` を引数に取る非公開関数にし、公開関数は [`MAX_EVAL_DATA_BYTES`] で
/// 委譲する（1GiB 超のファイルを CI の 3 OS で作らずに `TooLarge` を検証するため）。
fn evaluate_with_eval_data_invariance_limited<T, E>(
    frozen: &FrozenEvalData<'_>,
    limit: u64,
    eval: impl FnOnce(&[u8]) -> Result<T, E>,
) -> Result<T, EvalDataInvarianceError<E>> {
    // 1. 評価データを 1 回だけ読む。
    let bytes =
        fs::read_bounded(frozen.path, limit).map_err(|err| map_fs_error(frozen.path, err))?;

    // 2. 凍結記録と照合する。不一致なら eval を呼ばずに停止する。
    let actual_byte_len =
        u64::try_from(bytes.len()).map_err(|_| EvalDataInvarianceError::LengthOverflow)?;
    let actual_sha256 = Sha256Digest::of_bytes(&bytes);
    if actual_sha256 != frozen.sha256 || actual_byte_len != frozen.byte_len {
        return Err(EvalDataInvarianceError::FrozenRecordMismatch {
            expected_sha256: frozen.sha256,
            actual_sha256,
            expected_byte_len: frozen.byte_len,
            actual_byte_len,
        });
    }

    // 3. 検証済みのバイト列で評価を実行する。
    let eval_result = eval(&bytes);

    // 4. 評価前バッファを drop してからディスクを読み直す（最大メモリを
    //    1 ファイル分に抑える。eval が Err でも必ず実行する）。
    drop(bytes);
    let after = fs::sha256_file_bounded(frozen.path, limit)
        .map_err(|err| map_fs_error(frozen.path, err))?;

    // 5. 評価前後のダイジェストを比較する。変化があれば eval の結果を捨てる。
    if after != frozen.sha256 {
        return Err(EvalDataInvarianceError::ChangedDuringEvaluation {
            before: frozen.sha256,
            after,
        });
    }

    // 6. 一致すれば eval の結果を返す。
    eval_result.map_err(EvalDataInvarianceError::Evaluation)
}

/// 評価経路そのものを評価データの評価前後ハッシュ比較で包む（REQ-27「評価の
/// 独立性」・REQ-17「データの分割と凍結」・TASK-27.1-2・issue #70）。
///
/// CLI の `evaluate` 工程（`stages::evaluate`。#314 で接続済み）が、データ契約層の
/// `fandhe_edge_data::eval_freeze::evaluate_gate` を通して得た
/// `FreezeRecord`（`Proceed` の場合のみ）から [`FrozenEvalData`] を組み立て、
/// 実際の評価処理を `eval` クロージャとして渡す想定。処理順はモジュール doc
/// 「処理順」節を参照。
///
/// [`crate::invariance::evaluate_with_invariance`]（モデルパッケージ側）と
/// 組み合わせる場合は、`evaluate_with_invariance(paths, |paths| { ... その
/// クロージャの中で本関数を呼ぶ ... })` のように入れ子にする（合成 API・
/// 合成エラー型は本 issue の範囲では作らない）。
pub fn evaluate_with_eval_data_invariance<T, E>(
    frozen: &FrozenEvalData<'_>,
    eval: impl FnOnce(&[u8]) -> Result<T, E>,
) -> Result<T, EvalDataInvarianceError<E>> {
    evaluate_with_eval_data_invariance_limited(frozen, MAX_EVAL_DATA_BYTES, eval)
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
                "fandhe-edge-eval-data-invariance-unit-{pid}-{label}-{attempt}-{nanos}"
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
    fn req39_too_large_is_rejected_before_eval_is_called() {
        // limit を実データより小さく設定し、TooLarge で拒否されること・
        // クロージャが呼ばれていないことを確認する。
        let guard = write_unique_temp_file("too-large", b"0123456789");
        let frozen = FrozenEvalData {
            path: &guard.0,
            sha256: Sha256Digest::of_bytes(b"0123456789"),
            byte_len: 10,
        };
        let mut called = false;
        let result = evaluate_with_eval_data_invariance_limited::<(), ()>(&frozen, 5, |_bytes| {
            called = true;
            Ok(())
        });
        assert!(!called, "上限超過時は eval を呼ばないはず");
        match result {
            Err(EvalDataInvarianceError::TooLarge { size, limit, .. }) => {
                assert_eq!(size, 10);
                assert_eq!(limit, 5);
            }
            other => panic!("TooLarge を期待したが {other:?} だった"),
        }
    }

    #[test]
    fn req17_req27_byte_len_mismatch_alone_stops_before_evaluation() {
        // sha256 が一致していても byte_len だけが偽装されていれば停止する
        // （評価前段で eval を呼ばない）。
        let guard = write_unique_temp_file("byte-len-mismatch", b"hello");
        let frozen = FrozenEvalData {
            path: &guard.0,
            sha256: Sha256Digest::of_bytes(b"hello"),
            byte_len: 999,
        };
        let mut called = false;
        let result = evaluate_with_eval_data_invariance::<(), ()>(&frozen, |_bytes| {
            called = true;
            Ok(())
        });
        assert!(!called);
        match result {
            Err(EvalDataInvarianceError::FrozenRecordMismatch {
                expected_byte_len,
                actual_byte_len,
                ..
            }) => {
                assert_eq!(expected_byte_len, 999);
                assert_eq!(actual_byte_len, 5);
            }
            other => panic!("FrozenRecordMismatch を期待したが {other:?} だった"),
        }
    }

    #[test]
    fn req27_display_does_not_leak_eval_data_content() {
        // Display にはパス・16 進ダイジェスト・バイト長のみを含み、
        // 評価データ本体は含めない（security.md「秘密情報の混入防止」）。
        let secret = b"SECRET-VALUE-should-not-leak";
        let guard = write_unique_temp_file("display-leak", secret);
        let frozen = FrozenEvalData {
            path: &guard.0,
            sha256: Sha256Digest::of_bytes(b"different-bytes"),
            byte_len: 999,
        };
        let result: Result<(), EvalDataInvarianceError<String>> =
            evaluate_with_eval_data_invariance(&frozen, |_bytes| Ok(()));
        let message = result.unwrap_err().to_string();
        assert!(
            !message.contains("SECRET-VALUE-should-not-leak"),
            "評価データ本体が Display に漏れている: {message}"
        );
        assert!(
            message.contains(&Sha256Digest::of_bytes(secret).to_hex()),
            "評価データの sha256 hex を含むはず: {message}"
        );
    }

    #[test]
    fn length_overflow_variant_is_unreachable_in_practice() {
        // 64bit 環境では `bytes.len(): usize` は常に `u64` へ変換できるため、
        // `LengthOverflow` は実質的に到達しない（`fandhe_edge_data::eval_freeze::freeze_eval_data`
        // の同名バリアントと同じ理由。`.claude/rules/coding-rust.md`「外部入力」の
        // `as` キャスト禁止規約に従い `u64::try_from` を使うために型として残す）。
        // 到達不能のため専用テストは置かず、この doc コメントで理由を記録する。
    }
}
