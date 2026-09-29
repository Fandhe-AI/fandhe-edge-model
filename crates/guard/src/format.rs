//! 許可リストによるファイル形式判定基盤（REQ-39「形式・種別の検査」・TASK-39.2-1・#153）。
//!
//! 読み込もうとするファイルの形式を、拡張子ではなく先頭バイトの構造から判定し、
//! 許可リストに含まれる形式だけを通す。許可リスト方式のため、判定できない
//! 内容（[`FileFormat::Unknown`]）は常に拒否する（fail-closed）。
//!
//! # 責務境界
//!
//! - 判定は先頭 [`FORMAT_PREFIX_BYTES`] だけを読む。pickle 等の中身は解釈・展開・実行しない
//! - ONNX の判定は protobuf の top-level 構造の「形の検査」であり、意味の検証ではない。
//!   最終的な解析の成否は読み込み時の ONNX Runtime に委ねる。`external_data` の拒否は対象外
//! - 経路の閉じ込め（TASK-39.4）・サイズ上限（TASK-39.5）は本モジュールの責務外。
//!   呼び出し側（CLI の統合は TASK-39.2-4・#156）が経路 → サイズ → 形式の順で先に適用する
//! - 拡張子と内容の照合による偽装拒否シナリオは TASK-39.2-2（#154）で上に重ねる
//!
//! pickle プロトコル 0/1（テキスト opcode 始まり）は誤検出が多いため固定シグネチャでは
//! 判定せず、ONNX の構造検査にも通らないので `Unknown` として拒否される。

use fandhe_edge_core::exitcode::ExitCode;
use fandhe_edge_core::fs::{FsError, open_regular_file_for_read};
use std::collections::BTreeSet;
use std::fmt;
use std::io::Read as _;
use std::path::Path;

/// 判定に読む先頭バイト数の上限（64 KiB）。固定シグネチャと ONNX の top-level 数フィールドを
/// 見るのに十分で、ファイル全体（最大 1GB 級）は読まない。
pub const FORMAT_PREFIX_BYTES: usize = 64 * 1024;

/// 判定できるファイル形式。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum FileFormat {
    /// ONNX（ModelProto の top-level 構造の検査に合格したもの）。
    Onnx,
    /// pickle（先頭 `0x80` とプロトコル版 2..=5）。任意コード実行につながる形式。
    Pickle,
    /// NumPy `.npy`（先頭 `\x93NUMPY`）。
    Npy,
    /// zip（torch の `.pt` は zip 内に pickle を持つ）。
    Zip,
    /// GGUF（先頭 `GGUF`）。
    Gguf,
    /// 上のいずれでもない（空ファイルを含む）。許可リストには入らない。
    Unknown,
}

impl FileFormat {
    /// JSON・メッセージ用の英語識別子。
    pub const fn name(self) -> &'static str {
        match self {
            FileFormat::Onnx => "onnx",
            FileFormat::Pickle => "pickle",
            FileFormat::Npy => "npy",
            FileFormat::Zip => "zip",
            FileFormat::Gguf => "gguf",
            FileFormat::Unknown => "unknown",
        }
    }
}

/// ModelProto（onnx.proto3）の top-level フィールドの wire type。未知の field 番号は `None`。
/// 1 ir_version・5 model_version は varint(0)、それ以外は LEN(2)。
fn model_field_wire_type(field: u64) -> Option<u64> {
    match field {
        1 | 5 => Some(0),
        2 | 3 | 4 | 6 | 7 | 8 | 14 | 20 | 25 => Some(2),
        _ => None,
    }
}

/// varint を読み `(値, 消費バイト数)` を返す。10 バイト以内に終わらなければ `Err(false)`、
/// prefix の終端で途切れたら `Err(true)`。
fn read_varint(buf: &[u8]) -> Result<(u64, usize), bool> {
    let mut value: u64 = 0;
    for i in 0..10usize {
        let Some(&b) = buf.get(i) else {
            return Err(true);
        };
        let shift = (i as u32).saturating_mul(7);
        // 10 バイト目は u64 の最上位 1 ビットだけを持てる。0x00・0x01 以外は u64 に収まらない
        // 値（または継続ビット付き）であり、切り詰めて受理せず拒否する。
        if i == 9 && b > 0x01 {
            return Err(false);
        }
        value |= u64::from(b & 0x7f).checked_shl(shift).unwrap_or(0);
        if b & 0x80 == 0 {
            return Ok((value, i.saturating_add(1)));
        }
    }
    Err(false)
}

/// prefix 上で ModelProto の top-level フィールドを走査し、ONNX の形かを判定する。
/// 先頭が `ir_version`（tag 0x08・値 1..=0xFFFF）で、全フィールドが既知の
/// (field, wire type) であり、LEN が `total_len` を超えず、`graph`（field 7）が現れれば真。
/// prefix の終端で途切れた場合は、それまでに条件を満たしていれば真とする。
fn looks_like_onnx_model(prefix: &[u8], total_len: u64) -> bool {
    if prefix.first() != Some(&0x08) {
        return false;
    }
    let mut pos: usize = 0;
    let mut seen_graph = false;
    let mut seen_ir_version = false;
    while pos < prefix.len() {
        let Some(rest) = prefix.get(pos..) else {
            break;
        };
        let (tag, n) = match read_varint(rest) {
            Ok(v) => v,
            Err(truncated) => return truncated && seen_ir_version && seen_graph,
        };
        let field = tag >> 3;
        let wire = tag & 7;
        if model_field_wire_type(field) != Some(wire) {
            return false;
        }
        let Some(body_pos) = pos.checked_add(n) else {
            return false;
        };
        let Some(body) = prefix.get(body_pos..) else {
            return false;
        };
        if wire == 0 {
            let (v, m) = match read_varint(body) {
                Ok(v) => v,
                Err(truncated) => return truncated && seen_ir_version && seen_graph,
            };
            if field == 1 {
                if v == 0 || v > 0xFFFF {
                    return false;
                }
                seen_ir_version = true;
            }
            let Some(next) = body_pos.checked_add(m) else {
                return false;
            };
            pos = next;
        } else {
            let (len, m) = match read_varint(body) {
                Ok(v) => v,
                Err(truncated) => return truncated && seen_ir_version && seen_graph,
            };
            let Some(start) = body_pos.checked_add(m) else {
                return false;
            };
            let Some(end) = u64::try_from(start).ok().and_then(|s| s.checked_add(len)) else {
                return false;
            };
            if end > total_len {
                return false;
            }
            // graph は長さを完全に読み、ファイル内に収まると確認できてから認める。
            if field == 7 {
                seen_graph = true;
            }
            match usize::try_from(end) {
                Ok(e) if e <= prefix.len() => pos = e,
                // LEN が prefix を超える場合は中身を読まず、ここまでの検査で判定する。
                _ => return seen_ir_version && seen_graph,
            }
        }
    }
    seen_ir_version && seen_graph
}

/// 先頭バイトからファイル形式を判定する純関数。`prefix` はファイル先頭の有限長、
/// `total_len` はメタデータ上のファイルサイズ。判定順は固定シグネチャ → ONNX 構造検査 → `Unknown`。
pub fn detect_format(prefix: &[u8], total_len: u64) -> FileFormat {
    if let [0x80, ver, ..] = prefix
        && (2..=5).contains(ver)
    {
        return FileFormat::Pickle;
    }
    if prefix.starts_with(b"\x93NUMPY") {
        return FileFormat::Npy;
    }
    if prefix.starts_with(b"PK\x03\x04") || prefix.starts_with(b"PK\x05\x06") {
        return FileFormat::Zip;
    }
    if prefix.starts_with(b"GGUF") {
        return FileFormat::Gguf;
    }
    if looks_like_onnx_model(prefix, total_len) {
        return FileFormat::Onnx;
    }
    FileFormat::Unknown
}

/// 許可する形式の集合。[`FileFormat::Unknown`] は入れられない（fail-closed）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormatAllowlist {
    formats: BTreeSet<FileFormat>,
}

impl FormatAllowlist {
    /// 許可リストを作る。`Unknown` は黙って除外する。
    pub fn new(formats: impl IntoIterator<Item = FileFormat>) -> Self {
        Self {
            formats: formats
                .into_iter()
                .filter(|f| *f != FileFormat::Unknown)
                .collect(),
        }
    }

    /// 推論・パッケージ用の既定（ONNX のみ）。
    pub fn onnx_only() -> Self {
        Self::new([FileFormat::Onnx])
    }

    /// 形式が許可されているか。
    pub fn contains(&self, format: FileFormat) -> bool {
        self.formats.contains(&format)
    }

    /// 判定済みの形式を許可リストと照合する。
    pub fn check(&self, detected: FileFormat) -> Result<AllowedFormat, FormatRejection> {
        if self.contains(detected) {
            Ok(AllowedFormat(detected))
        } else {
            Err(FormatRejection::NotAllowed {
                detected,
                allowed: self.formats.iter().copied().collect(),
            })
        }
    }
}

/// 許可リストの検査を通った形式。外部から直接は作れない。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllowedFormat(FileFormat);

impl AllowedFormat {
    /// 通った形式。
    pub const fn format(self) -> FileFormat {
        self.0
    }
}

/// 形式検査の拒否理由。メッセージは英語で、ファイル内容・パスを含めない。
#[derive(Debug)]
#[non_exhaustive]
pub enum FormatRejection {
    /// 検出した形式が許可リストにない（`allowed` はソート済み）。
    NotAllowed {
        detected: FileFormat,
        allowed: Vec<FileFormat>,
    },
    /// ファイルを開けない・通常ファイルでない・読み込めない。
    Io(FsError),
}

impl FormatRejection {
    /// 終了コード（REQ-21）。形式不許可・通常ファイルでない・存在しない → `InvalidInput`、
    /// サイズ超過 → `LimitExceeded`、その他の I/O 失敗 → `RuntimeError`。
    pub fn exit_code(&self) -> ExitCode {
        match self {
            FormatRejection::NotAllowed { .. } => ExitCode::InvalidInput,
            FormatRejection::Io(FsError::NotRegularFile { .. }) => ExitCode::InvalidInput,
            FormatRejection::Io(FsError::TooLarge { .. }) => ExitCode::LimitExceeded,
            FormatRejection::Io(FsError::Read { source, .. })
                if source.kind() == std::io::ErrorKind::NotFound =>
            {
                ExitCode::InvalidInput
            }
            FormatRejection::Io(_) => ExitCode::RuntimeError,
        }
    }

    /// 機械可読な理由コード（英語の snake_case）。
    pub const fn reason_code(&self) -> &'static str {
        match self {
            FormatRejection::NotAllowed { .. } => "format_not_allowed",
            FormatRejection::Io(_) => "file_unreadable",
        }
    }
}

impl fmt::Display for FormatRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FormatRejection::NotAllowed { detected, allowed } => {
                let names: Vec<&str> = allowed.iter().map(|a| a.name()).collect();
                write!(
                    f,
                    "file format '{}' is not allowed (allowed: {})",
                    detected.name(),
                    names.join(", ")
                )
            }
            FormatRejection::Io(_) => write!(f, "file could not be opened as a regular file"),
        }
    }
}

impl std::error::Error for FormatRejection {}

/// ファイルの先頭だけを読んで形式を判定し、許可リストと照合する。
///
/// 経路の検証・サイズ上限を通したパスを渡すこと（責務外。モジュール doc 参照）。
pub fn check_file_format(
    path: &Path,
    allowlist: &FormatAllowlist,
) -> Result<AllowedFormat, FormatRejection> {
    let file = open_regular_file_for_read(path).map_err(FormatRejection::Io)?;
    let read_err = |source| {
        FormatRejection::Io(FsError::Read {
            path: path.to_path_buf(),
            source,
        })
    };
    let total_len = file.metadata().map_err(read_err)?.len();
    let mut prefix = Vec::new();
    file.take(FORMAT_PREFIX_BYTES as u64)
        .read_to_end(&mut prefix)
        .map_err(read_err)?;
    allowlist.check(detect_format(&prefix, total_len))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(b: &[u8]) -> FileFormat {
        detect_format(b, b.len() as u64)
    }

    /// REQ-39・TASK-39.2-1: 最小の ONNX 形と実物に近い先頭を Onnx と判定する。
    #[test]
    fn req39_detects_onnx() {
        assert_eq!(d(&[0x08, 0x07, 0x3a, 0x00]), FileFormat::Onnx);
        let mut b = vec![0x08, 0x03, 0x12, 7];
        b.extend_from_slice(b"pytorch");
        b.extend_from_slice(&[0x1a, 3]);
        b.extend_from_slice(b"0.3");
        b.extend_from_slice(&[0x3a, 2, 0x0a, 0x00]);
        assert_eq!(d(&b), FileFormat::Onnx);
    }

    /// REQ-39・TASK-39.2-1: graph が prefix を超える巨大 LEN でも中身を読まず合格する。
    #[test]
    fn req39_onnx_large_graph_beyond_prefix() {
        let prefix = [0x08, 0x07, 0x3a, 0x80, 0x80, 0x80, 0x04];
        assert_eq!(detect_format(&prefix, 1 << 30), FileFormat::Onnx);
        // 実サイズより大きい LEN は拒否。
        assert_eq!(detect_format(&prefix, 100), FileFormat::Unknown);
    }

    /// REQ-39・TASK-39.2-1: 固定シグネチャ形式。
    #[test]
    fn req39_detects_signature_formats() {
        assert_eq!(d(&[0x80, 0x04, 0x95, 0x00]), FileFormat::Pickle);
        assert_eq!(d(&[0x80, 0x02]), FileFormat::Pickle);
        assert_eq!(d(&[0x80, 0x05]), FileFormat::Pickle);
        assert_eq!(d(&[0x80, 0x06]), FileFormat::Unknown);
        assert_eq!(d(b"\x93NUMPY\x01\x00"), FileFormat::Npy);
        assert_eq!(d(b"PK\x03\x04abc"), FileFormat::Zip);
        assert_eq!(d(b"PK\x05\x06"), FileFormat::Zip);
        assert_eq!(d(b"GGUF\x03"), FileFormat::Gguf);
    }

    /// REQ-39・TASK-39.2-1: 判定できないものと壊れた ONNX は Unknown（panic しない）。
    #[test]
    fn req39_unknown_and_broken_onnx() {
        assert_eq!(d(b""), FileFormat::Unknown);
        assert_eq!(d(b"hello"), FileFormat::Unknown);
        assert_eq!(d(b"(lp0\n."), FileFormat::Unknown);
        assert_eq!(d(&[0x08, 0x07]), FileFormat::Unknown); // graph なし
        assert_eq!(d(&[0x3a, 0x00]), FileFormat::Unknown); // 先頭が ir_version でない
        assert_eq!(d(&[0x08, 0x07, 0x3a, 0x05, 0x00]), FileFormat::Unknown); // LEN 超過
        assert_eq!(
            d(&[0x08, 0x07, 0x78, 0x01, 0x3a, 0x00]),
            FileFormat::Unknown
        ); // 未知 field 15
        assert_eq!(d(&[0x08, 0x00, 0x3a, 0x00]), FileFormat::Unknown); // ir_version 0
        let mut long = vec![0x08];
        long.extend_from_slice(&[0xff; 11]);
        assert_eq!(d(&long), FileFormat::Unknown);
    }

    /// REQ-39・TASK-39.2-1: graph の長さ varint が途中で切れた入力は ONNX と認めない。
    #[test]
    fn req39_truncated_graph_length_is_unknown() {
        assert_eq!(d(&[0x08, 0x07, 0x3a, 0x80]), FileFormat::Unknown);
        assert!(!FormatAllowlist::onnx_only().contains(d(&[0x08, 0x07, 0x3a, 0x80])));
    }

    /// REQ-39・TASK-39.2-1: 10 バイト目が 0x00・0x01 以外の varint は拒否し、0x01 は受理する。
    #[test]
    fn req39_varint_tenth_byte_must_be_0_or_1() {
        let mut bad = vec![0x08, 0x07, 0x3a];
        bad.extend_from_slice(&[0x80; 9]);
        bad.push(0x02);
        assert_eq!(detect_format(&bad, u64::MAX), FileFormat::Unknown);
        let mut ok = vec![0x08, 0x07, 0x3a];
        ok.extend_from_slice(&[0x80; 9]);
        ok.push(0x01);
        assert_eq!(detect_format(&ok, u64::MAX), FileFormat::Onnx);
    }

    /// REQ-39・TASK-39.2-1: onnx_only は ONNX だけを通し、他は InvalidInput(64) で拒否する。
    #[test]
    fn req39_onnx_only_allowlist() {
        let al = FormatAllowlist::onnx_only();
        let ok = al.check(FileFormat::Onnx).unwrap();
        assert_eq!(ok.format(), FileFormat::Onnx);
        for f in [
            FileFormat::Pickle,
            FileFormat::Npy,
            FileFormat::Zip,
            FileFormat::Gguf,
            FileFormat::Unknown,
        ] {
            let err = al.check(f).unwrap_err();
            assert_eq!(err.exit_code(), ExitCode::InvalidInput);
            assert_eq!(err.exit_code().code(), 64);
            assert_eq!(err.reason_code(), "format_not_allowed");
            match err {
                FormatRejection::NotAllowed { detected, allowed } => {
                    assert_eq!(detected, f);
                    assert_eq!(allowed, vec![FileFormat::Onnx]);
                }
                _ => panic!("unexpected variant"),
            }
        }
    }

    /// REQ-39・TASK-39.2-1: 許可リストの中身で結果が変わり、Unknown は入れられない。
    #[test]
    fn req39_allowlist_contents_and_unknown_excluded() {
        let al = FormatAllowlist::new([FileFormat::Onnx, FileFormat::Npy]);
        assert!(al.check(FileFormat::Npy).is_ok());
        assert!(al.check(FileFormat::Pickle).is_err());
        let al = FormatAllowlist::new([FileFormat::Unknown, FileFormat::Onnx]);
        assert!(!al.contains(FileFormat::Unknown));
        assert!(al.check(FileFormat::Unknown).is_err());
    }

    /// REQ-39・TASK-39.2-1: 拒否メッセージは英語の形式名だけで、入力バイトを含まない。
    #[test]
    fn req39_message_has_no_content() {
        let mut b = vec![0x80, 0x04];
        b.extend_from_slice(b"CANARY_SECRET");
        let f = d(&b);
        let msg = FormatAllowlist::onnx_only()
            .check(f)
            .unwrap_err()
            .to_string();
        assert_eq!(msg, "file format 'pickle' is not allowed (allowed: onnx)");
        assert!(!msg.contains("CANARY"));
    }
}
