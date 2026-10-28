//! ベースモデルのディレクトリ内ファイルを開く入口（REQ-41・REQ-39・TASK-41.2-2・#398）。
//!
//! 追加学習（REQ-41）で利用者がローカルパスで渡すベースモデルのディレクトリ
//! （`model.safetensors`・`tokenizer.json` 等）を、読む前にガード層で検査する。役割は
//! 「名前の字句検査 → 閉じ込め付き open → 読み込み前のサイズ確認 → 内容の形式照合」だけで、
//! 経路・サイズ・形式の判定は [`crate::package`]・[`crate::file_size`]・[`crate::format`] に委譲する。
//!
//! - ディレクトリの受付は [`crate::package::confine_package`] をそのまま使う。root は呼び出し側
//!   （CLI 配線時。TASK-41.5・#425）が決める。ディレクトリ自身に閉じ込めるなら
//!   `confine_package(dir, Path::new("."))`
//! - メンバーは保持 fd 起点の `O_NOFOLLOW` で開く。`..`・絶対パス・symlink（末尾も親も）は拒否し、
//!   パスを開き直さない（検証後の差し替え対策）。このため HF cache の `snapshots/<rev>/`
//!   （blobs への symlink）は拒否される。実ファイルとして置いたディレクトリを渡すこと
//! - 検査順序は 名前 → 経路 → サイズ（fstat。1 バイトも読まない）→ 形式。名前と内容が食い違う
//!   もの（pickle 偽装を含む）は `format_not_allowed` で拒否する
//! - 利用者由来のパス・ファイル名・内容はエラーの値（Debug・source を含む）にもメッセージにも含めない。
//!   経路・形式の拒否は終了コードと理由コードだけに畳む
//!
//! - サイズ上限は重み（safetensors）が 5 GiB（[`MAX_BASE_MODEL_FILE_BYTES`]。パッケージ側の shard 上限 2 GiB とは別。#655）、
//!   `tokenizer.json` が 1 GiB（`MAX_READ_FILE_BYTES`）
//!
//! - `train --kind qwen2_lora` の起動前照合（#628）は [`verify_base_model_sha256`]。定義の `base_model.sha256` の
//!   全ファイルを開いて sha256 を比べる。`config.json` は形式の許可リストに無い（JSON の解釈は学習ワーカー）ため、
//!   名前・経路・サイズ（64 KiB。学習ワーカーの `MAX_BASE_MODEL_CONFIG_BYTES` と同値）だけを確かめてストリームで
//!   ハッシュする。重み・`tokenizer.json` は [`open_base_model_file`] の検査を通したバイト列をハッシュする
//!
//! 起動前照合の重み（safetensors）はヘッダーだけ検査して本体を固定長バッファでハッシュする（#672）。
//! ponytail: `tokenizer.json` と [`open_base_model_file`] は上限（1 GiB・重み 5 GiB）までメモリへ読み切る。
//! ハードリンクで外部ファイルを指すメンバーは経路では検出できない（必要なら `st_nlink > 1` を拒否）。

use crate::file_size::{
    FileSizeRejection, MAX_READ_FILE_BYTES, check_file_size_unclamped, check_file_size_within,
};
use crate::format::{
    CheckedFile, FileFormat, FormatAllowlist, check_open_file, check_open_file_unclamped,
};
use crate::package::ConfinedPackage;
/// ベースモデルの重み（safetensors）1 ファイルの上限（5 GiB。REQ-39・REQ-41・#665）。
/// 根拠: HF の分割既定が 5 GB で、Qwen 系の重み 1 ファイルは最大 3.95 GB。学習ワーカー側
/// `limits.py::MAX_BASE_MODEL_FILE_BYTES` と同値（共有 fixture `fixtures/lora_limits/base_model_file.json` で照合）。
/// パッケージ側の shard 上限（`split_files::MAX_SHARD_FILE_BYTES`、2 GiB）とは別で、そちらは変えない。
/// 上限緩和はオーナー判断 2026-10-27（上限の見直しを一任）・#665。
pub const MAX_BASE_MODEL_FILE_BYTES: u64 = 5 << 30;

use fandhe_edge_core::base_model_format::{check_safetensors_prefix, safetensors_header_end};
use fandhe_edge_core::definition::BASE_MODEL_CONFIG_FILE;
use fandhe_edge_core::exitcode::ExitCode;
use fandhe_edge_core::fs::{FsError, sha256_open_file_bounded};
use fandhe_edge_core::hash::Sha256Digest;
use fandhe_edge_core::hash::Sha256Stream;
use std::collections::BTreeMap;
use std::fmt;
use std::io::Read as _;
use std::path::Path;

/// ストリーム読み込みのバッファ長（1 MiB）。重みのサイズによらず RSS はこの長さ＋ヘッダー（最大 16 MiB）で頭打ち。
const STREAM_BUFFER_BYTES: usize = 1 << 20;

/// `config.json` の 1 ファイル上限（64 KiB。学習ワーカーの `MAX_BASE_MODEL_CONFIG_BYTES` と同値。#628）。
pub const MAX_BASE_MODEL_CONFIG_BYTES: u64 = 64 * 1024;

/// ベースモデルのファイルを開く際の拒否理由。
#[derive(Debug)]
#[non_exhaustive]
pub enum BaseModelFileRejection {
    /// ファイル名が許可されていない（`*.safetensors`・`tokenizer.json` 以外）。ファイルには触れない。
    NameNotAllowed,
    /// 経路の閉じ込め・open の拒否（`../`・絶対パス・symlink・通常ファイル以外）。
    /// 元の [`PathRejection`] は利用者由来のパスを保持するため、終了コードと理由コードだけを残す。
    Path {
        exit_code: ExitCode,
        reason_code: &'static str,
    },
    /// 読み込み前のサイズ確認の拒否（上限超過等。パスを含まない）。
    Size(FileSizeRejection),
    /// 内容の形式の拒否（名前との不一致・pickle 偽装・許可リスト外・読み込み失敗）。
    /// 元の [`FormatRejection`] の I/O エラーはパスを保持しうるため、終了コードと理由コードだけを残す。
    Format {
        exit_code: ExitCode,
        reason_code: &'static str,
    },
    /// 実ファイルの sha256 が定義の値と一致しない（#628）。どのファイルかは利用者由来の名前のため持たない。
    Sha256Mismatch,
}

impl BaseModelFileRejection {
    /// 終了コード（REQ-21）。名前不許可は `InvalidInput`、他は内包する拒否のもの。
    pub fn exit_code(&self) -> ExitCode {
        match self {
            Self::NameNotAllowed | Self::Sha256Mismatch => ExitCode::InvalidInput,
            Self::Path { exit_code, .. } | Self::Format { exit_code, .. } => *exit_code,
            Self::Size(e) => e.exit_code(),
        }
    }

    /// 機械可読な理由コード（英語の snake_case）。
    pub fn reason_code(&self) -> &'static str {
        match self {
            Self::NameNotAllowed => "base_model_file_name_not_allowed",
            Self::Sha256Mismatch => "base_model_sha256_mismatch",
            Self::Path { reason_code, .. } | Self::Format { reason_code, .. } => reason_code,
            Self::Size(e) => e.reason_code(),
        }
    }
}

impl fmt::Display for BaseModelFileRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NameNotAllowed => write!(f, "base model file name is not allowed"),
            Self::Path { .. } => write!(f, "base model file path was rejected"),
            Self::Size(e) => e.fmt(f),
            Self::Format { .. } => write!(f, "base model file format was rejected"),
            Self::Sha256Mismatch => write!(f, "base model file sha256 mismatch"),
        }
    }
}

impl std::error::Error for BaseModelFileRejection {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Size(e) => Some(e),
            _ => None,
        }
    }
}

/// ファイル名から期待する内容の形式を決める。大文字小文字を区別し、ドットファイル・拡張子なしは除外する。
fn expected_format(member: &Path) -> Option<FileFormat> {
    if member.file_name().is_some_and(|n| n == "tokenizer.json") {
        return Some(FileFormat::TokenizerJson);
    }
    if member.extension().is_some_and(|e| e == "safetensors") {
        return Some(FileFormat::Safetensors);
    }
    None
}

/// 閉じ込め済みのベースモデルのディレクトリから `member` を開き、名前と内容が一致するものだけを返す。
///
/// 学習ワーカー起動前の `train`（追加学習。REQ-41）から呼ばれる想定。
///
/// # Errors
/// [`BaseModelFileRejection`]。検査順序はモジュール doc を参照。
pub fn open_base_model_file(
    dir: &ConfinedPackage,
    member: &Path,
) -> Result<CheckedFile, BaseModelFileRejection> {
    let expected = expected_format(member).ok_or(BaseModelFileRejection::NameNotAllowed)?;
    let file = dir
        .open_regular_member(member)
        .map_err(|e| BaseModelFileRejection::Path {
            exit_code: e.exit_code(),
            reason_code: e.reason_code(),
        })?;
    let allow = FormatAllowlist::new([expected]);
    // 重み（safetensors）だけ重みファイル専用の 5 GiB。1.5B の model.safetensors は 3.09 GB で、
    // 7B の埋め込み 1 テンソルは 1 GiB 超で分割できないため（REQ-39・REQ-41・#654・#655）。他のファイルは 1 GiB のまま
    let checked = if expected == FileFormat::Safetensors {
        check_file_size_unclamped(&file, MAX_BASE_MODEL_FILE_BYTES)
            .map_err(BaseModelFileRejection::Size)?;
        check_open_file_unclamped(file, member, &allow, MAX_BASE_MODEL_FILE_BYTES)
    } else {
        check_file_size_within(&file, MAX_READ_FILE_BYTES).map_err(BaseModelFileRejection::Size)?;
        check_open_file(file, member, &allow, MAX_READ_FILE_BYTES)
    };
    checked.map_err(|e| BaseModelFileRejection::Format {
        exit_code: e.exit_code(),
        reason_code: e.reason_code(),
    })
}

/// 閉じ込め済みのベースモデルのディレクトリで、`expected`（定義の `base_model.sha256`）の全ファイルの sha256 を
/// 実ファイルと照合する（`train --kind qwen2_lora` が学習ワーカーの起動前・候補ディレクトリの作成前に呼ぶ。
/// REQ-41・REQ-39・#628）。
///
/// 名前順に 1 ファイルずつ開き、最初の拒否で止める。メモリに載せるのは 1 ファイル分だけ（重みは最大 5 GiB）。
///
/// # Errors
/// 欠落・symlink・経路外は `Path`、上限超過は `Size`（20）、形式の不一致は `Format`、`config.json`・
/// `tokenizer.json`・`*.safetensors` 以外の名前は `NameNotAllowed`、ハッシュの不一致は `Sha256Mismatch`（64）。
pub fn verify_base_model_sha256(
    dir: &ConfinedPackage,
    expected: &BTreeMap<String, Sha256Digest>,
) -> Result<(), BaseModelFileRejection> {
    for (name, want) in expected {
        let member = Path::new(name);
        let got = if name == BASE_MODEL_CONFIG_FILE {
            sha256_config(dir, member)?
        } else {
            sha256_streaming(dir, name)?
        };
        if &got != want {
            return Err(BaseModelFileRejection::Sha256Mismatch);
        }
    }
    Ok(())
}

/// 重み（safetensors）は先頭ヘッダーだけで形式を検査し、本体は固定長バッファで読みながらハッシュする
/// （全体を保持しない。#672）。`tokenizer.json` は従来どおり [`open_base_model_file`] の検査を通す。
/// 検査順序・拒否の終了コードは [`open_base_model_file`] と同じ（名前 → 経路 → サイズ → 形式）。
fn sha256_streaming(
    dir: &ConfinedPackage,
    name: &str,
) -> Result<Sha256Digest, BaseModelFileRejection> {
    let member = Path::new(name);
    let expected = expected_format(member).ok_or(BaseModelFileRejection::NameNotAllowed)?;
    if expected != FileFormat::Safetensors {
        return Ok(Sha256Digest::of_bytes(
            open_base_model_file(dir, member)?.as_bytes(),
        ));
    }
    let file = dir
        .open_regular_member(member)
        .map_err(|e| BaseModelFileRejection::Path {
            exit_code: e.exit_code(),
            reason_code: e.reason_code(),
        })?;
    let size = check_file_size_unclamped(&file, MAX_BASE_MODEL_FILE_BYTES)
        .map_err(BaseModelFileRejection::Size)?;
    sha256_safetensors_stream(file, size, MAX_BASE_MODEL_FILE_BYTES)
}

/// 形式の拒否（名前と内容の不一致等。`open_base_model_file` の `format_not_allowed` と同じ値）。
fn not_allowed() -> BaseModelFileRejection {
    BaseModelFileRejection::Format {
        exit_code: ExitCode::InvalidInput,
        reason_code: "format_not_allowed",
    }
}

/// 読み取りの失敗を従来経路（`FormatRejection::Io`）と同じ分類にする。短すぎ（`UnexpectedEof`）は
/// 形式の拒否、存在しない（`NotFound`）は 64、他の I/O エラーは 70。reason_code は `file_unreadable`。
fn read_failed(e: &std::io::Error) -> BaseModelFileRejection {
    match e.kind() {
        std::io::ErrorKind::UnexpectedEof => not_allowed(),
        kind => BaseModelFileRejection::Format {
            exit_code: if kind == std::io::ErrorKind::NotFound {
                ExitCode::InvalidInput
            } else {
                ExitCode::RuntimeError
            },
            reason_code: "file_unreadable",
        },
    }
}

/// 開いた fd（`reader`）から、ヘッダー（先頭 8 バイト＋ヘッダー長ぶん。上限 16 MiB）を検査しつつ最大
/// [`STREAM_BUFFER_BYTES`] ずつ読んで sha256 を計算する。`size` は fstat の長さで、読めた総量が
/// これと食い違えば（検査後の伸縮）拒否する。`limit` を超えて読まない（超過は従来の
/// `Io(TooLarge)` と同じ 20・`file_unreadable`）。
fn sha256_safetensors_stream<R: std::io::Read>(
    reader: R,
    size: u64,
    limit: u64,
) -> Result<Sha256Digest, BaseModelFileRejection> {
    let mut reader = reader.take(limit.saturating_add(1));
    let mut hasher = Sha256Stream::new();
    let mut first8 = [0u8; 8];
    reader
        .read_exact(&mut first8)
        .map_err(|e| read_failed(&e))?;
    let end = safetensors_header_end(first8).map_err(|_| not_allowed())?;
    // ヘッダー長がファイル全体を超える指定は、ヘッダーを確保・読み込みする前に拒否する
    if end as u64 > size {
        return Err(not_allowed());
    }
    let mut prefix = first8.to_vec();
    prefix.resize(end, 0);
    let rest = prefix.get_mut(first8.len()..).ok_or_else(not_allowed)?;
    reader.read_exact(rest).map_err(|e| read_failed(&e))?;
    check_safetensors_prefix(&prefix, size).map_err(|_| not_allowed())?;
    hasher.update(&prefix);
    let mut total = end as u64;
    drop(prefix);
    let mut buf = vec![0u8; STREAM_BUFFER_BYTES];
    loop {
        let n = reader.read(&mut buf).map_err(|e| read_failed(&e))?;
        if n == 0 {
            break;
        }
        total += n as u64;
        hasher.update(buf.get(..n).ok_or_else(not_allowed)?);
    }
    if total > limit {
        return Err(BaseModelFileRejection::Format {
            exit_code: ExitCode::LimitExceeded,
            reason_code: "file_unreadable",
        });
    }
    if total != size {
        return Err(not_allowed());
    }
    Ok(hasher.finish())
}

/// `config.json` を閉じ込め付きで開き、64 KiB を上限にストリームで sha256 を計算する（内容は解釈しない）。
fn sha256_config(
    dir: &ConfinedPackage,
    member: &Path,
) -> Result<Sha256Digest, BaseModelFileRejection> {
    let file = dir
        .open_regular_member(member)
        .map_err(|e| BaseModelFileRejection::Path {
            exit_code: e.exit_code(),
            reason_code: e.reason_code(),
        })?;
    check_file_size_within(&file, MAX_BASE_MODEL_CONFIG_BYTES)
        .map_err(BaseModelFileRejection::Size)?;
    // 検査後の増加は読み進めた累計で検出される（`FsError::TooLarge`）。I/O エラーはパスを保持するため畳む
    sha256_open_file_bounded(file, member, MAX_BASE_MODEL_CONFIG_BYTES).map_err(|e| match e {
        FsError::TooLarge { size, limit, .. } => {
            BaseModelFileRejection::Size(FileSizeRejection::TooLarge { size, limit })
        }
        FsError::NotRegularFile { .. } => {
            BaseModelFileRejection::Size(FileSizeRejection::NotRegularFile)
        }
        _ => BaseModelFileRejection::Format {
            exit_code: ExitCode::RuntimeError,
            reason_code: "base_model_file_unreadable",
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-39・REQ-41・#665: 5 GiB ちょうど（sparse）はサイズ検査を通り、+1 は拒否。読み込まない。
    #[test]
    fn req39_size_check_boundary_at_5gib() {
        let p = std::env::temp_dir().join(format!("fe-guard-bm5g-{}", std::process::id()));
        let f = std::fs::File::create(&p).unwrap();
        f.set_len(MAX_BASE_MODEL_FILE_BYTES).unwrap();
        let ok = check_file_size_unclamped(&f, MAX_BASE_MODEL_FILE_BYTES);
        f.set_len(MAX_BASE_MODEL_FILE_BYTES + 1).unwrap();
        let ng = check_file_size_unclamped(&f, MAX_BASE_MODEL_FILE_BYTES);
        let _ = std::fs::remove_file(&p);
        assert_eq!(ok.unwrap(), MAX_BASE_MODEL_FILE_BYTES);
        assert!(matches!(ng, Err(FileSizeRejection::TooLarge { .. })));
    }

    /// REQ-41・REQ-39・TASK-41.2-2: 名前判定の境界。
    #[test]
    fn req41_expected_format_boundaries() {
        for p in [
            "model.safetensors",
            "model-00001-of-00002.safetensors",
            "sub/w.safetensors",
        ] {
            assert_eq!(
                expected_format(Path::new(p)),
                Some(FileFormat::Safetensors),
                "{p}"
            );
        }
        assert_eq!(
            expected_format(Path::new("tokenizer.json")),
            Some(FileFormat::TokenizerJson)
        );
        for p in [
            "pytorch_model.bin",
            "model.pt",
            "config.json",
            "model.SAFETENSORS",
            ".safetensors",
            "tokenizer.JSON",
            "my_tokenizer.json",
            "model",
        ] {
            assert_eq!(expected_format(Path::new(p)), None, "{p}");
        }
    }

    /// REQ-41・REQ-39: 拒否型の終了コードと理由コード。
    #[test]
    fn req41_name_rejection_codes() {
        let e = BaseModelFileRejection::NameNotAllowed;
        assert_eq!(e.exit_code(), ExitCode::InvalidInput);
        assert_eq!(e.reason_code(), "base_model_file_name_not_allowed");
        let e = BaseModelFileRejection::Sha256Mismatch;
        assert_eq!(e.exit_code(), ExitCode::InvalidInput);
        assert_eq!(e.reason_code(), "base_model_sha256_mismatch");
        assert_eq!(e.to_string(), "base model file sha256 mismatch");
    }

    /// 先頭ヘッダー付きの合成 safetensors（U8 テンソル 1 個。データ部は `n` バイトの疑似データ）。
    fn st(n: usize) -> Vec<u8> {
        let h = format!(r#"{{"t":{{"dtype":"U8","shape":[{n}],"data_offsets":[0,{n}]}}}}"#);
        let mut v = (h.len() as u64).to_le_bytes().to_vec();
        v.extend_from_slice(h.as_bytes());
        v.extend((0..n).map(|i| (i % 251) as u8));
        v
    }

    fn stream(bytes: &[u8], size: u64, limit: u64) -> Result<Sha256Digest, BaseModelFileRejection> {
        sha256_safetensors_stream(bytes, size, limit)
    }

    /// 読み出し 1 回あたりの要求長の最大値を数える Reader。
    struct CountingReader<'a> {
        inner: &'a [u8],
        max_request: usize,
    }

    impl std::io::Read for CountingReader<'_> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            self.max_request = self.max_request.max(buf.len());
            self.inner.read(buf)
        }
    }

    /// 中身を読み切ったあとに `kind` のエラーを返す Reader。
    struct FailingReader<'a> {
        inner: &'a [u8],
        kind: std::io::ErrorKind,
    }

    impl std::io::Read for FailingReader<'_> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            match self.inner.read(buf)? {
                0 => Err(std::io::Error::from(self.kind)),
                n => Ok(n),
            }
        }
    }

    /// REQ-39・REQ-41・#672: バッファ長を超える（3 MiB 超）重みでも全体読みと同じ sha256。
    #[test]
    fn req41_issue672_stream_matches_whole_read() {
        let b = st(3 * STREAM_BUFFER_BYTES + 5);
        let got = stream(&b, b.len() as u64, MAX_BASE_MODEL_FILE_BYTES).unwrap();
        assert_eq!(got, Sha256Digest::of_bytes(&b));
    }

    /// REQ-39・#672: 読み出し 1 回の要求長が固定バッファ長以下（全体読みに戻ると大きな要求になり失敗する）。
    #[test]
    fn req41_issue672_stream_request_size_is_bounded() {
        let b = st(3 * STREAM_BUFFER_BYTES + 5);
        let mut r = CountingReader {
            inner: &b,
            max_request: 0,
        };
        sha256_safetensors_stream(&mut r, b.len() as u64, MAX_BASE_MODEL_FILE_BYTES).unwrap();
        assert!(r.max_request > 0 && r.max_request <= STREAM_BUFFER_BYTES);
    }

    /// REQ-39・#672: ヘッダー不正・伸縮・短すぎは従来と同じ 64・`format_not_allowed`。
    /// 読み込み中に上限を超えて伸びたファイルは 20・`file_unreadable`（従来の `Io(TooLarge)` と同じ）。
    #[test]
    fn req41_issue672_stream_rejections() {
        let b = st(100);
        let n = b.len() as u64;
        let mut bad = b.clone();
        bad[8] = b'[';
        // ヘッダー長が size を超える指定（先頭 8 バイトだけの偽ファイル）は読み取り前に拒否される
        let huge = (1u64 << 20).to_le_bytes().to_vec();
        for (bytes, size) in [
            (bad, n),
            (b.clone(), n - 1),
            (b.clone(), n + 1),
            (vec![1; 4], 4),
            (huge, 8),
        ] {
            let e = stream(&bytes, size, MAX_BASE_MODEL_FILE_BYTES).unwrap_err();
            assert_eq!(e.reason_code(), "format_not_allowed");
            assert_eq!(e.exit_code(), ExitCode::InvalidInput);
        }
        let e = stream(&b, n, n - 1).unwrap_err();
        assert_eq!(e.exit_code(), ExitCode::LimitExceeded);
        assert_eq!(e.reason_code(), "file_unreadable");
    }

    /// REQ-39・#672: 本体・ヘッダーの I/O エラーは 70・`file_unreadable`（`UnexpectedEof` だけ形式拒否）。
    #[test]
    fn req41_issue672_stream_io_error_classification() {
        let b = st(100);
        let n = b.len() as u64;
        for cut in [4, 20, b.len() - 10] {
            let r = FailingReader {
                inner: &b[..cut],
                kind: std::io::ErrorKind::Other,
            };
            let e = sha256_safetensors_stream(r, n, MAX_BASE_MODEL_FILE_BYTES).unwrap_err();
            assert_eq!(e.exit_code(), ExitCode::RuntimeError, "{cut}");
            assert_eq!(e.reason_code(), "file_unreadable", "{cut}");
        }
        let r = FailingReader {
            inner: &b[..20],
            kind: std::io::ErrorKind::UnexpectedEof,
        };
        let e = sha256_safetensors_stream(r, n, MAX_BASE_MODEL_FILE_BYTES).unwrap_err();
        assert_eq!(e.reason_code(), "format_not_allowed");
    }
}
