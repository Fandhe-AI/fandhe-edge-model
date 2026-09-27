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
use std::io;
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
    TooLarge { limit: u64, actual: u64 },
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
            FreezeError::Io(err) => write!(f, "evaluation data I/O error: {err}"),
        }
    }
}

impl std::error::Error for FreezeError {}

/// 評価データを凍結する。
///
/// - `path` が `None` の場合: I/O を一切行わず `Ok(EvalDataStatus::NotProvided)` を返す。
/// - `path` が `Some` の場合: ファイルサイズを確認してから（上限超過ならファイルを
///   開かずに拒否）、ストリーミングで sha256 を計算し `EvalDataStatus::Frozen` を返す。
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
    let byte_len = metadata.len();
    if byte_len > MAX_EVAL_DATA_BYTES {
        return Err(FreezeError::TooLarge {
            limit: MAX_EVAL_DATA_BYTES,
            actual: byte_len,
        });
    }

    let file = File::open(path).map_err(|err| {
        if err.kind() == io::ErrorKind::NotFound {
            FreezeError::NotFound(err)
        } else {
            FreezeError::Io(err)
        }
    })?;
    let sha256 = sha256_hex_of_reader(io::BufReader::new(file)).map_err(FreezeError::Io)?;

    Ok(EvalDataStatus::Frozen(FreezeRecord {
        path: path.to_path_buf(),
        sha256,
        byte_len,
    }))
}
