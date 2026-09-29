//! 許可リストによるファイル形式判定基盤（REQ-39「形式・種別の検査」・TASK-39.2-1・#153）。
//!
//! 読み込もうとするファイルの形式を、拡張子ではなく先頭バイトの構造から判定し、
//! 許可リストに含まれる形式だけを通す。許可済みの証明は
//! [`open_checked_file`] が実ファイルの内容を検査して返す [`CheckedFile`] だけが持ち、バイト列と形式を切り離せない。
//! 検査したバイト列はメモリに保持して [`CheckedFile`] で渡し、検査後の書き換えを防ぐ。許可リスト方式のため、判定できない
//! 内容（[`FileFormat::Unknown`]）は常に拒否する（fail-closed）。
//!
//! # 責務境界
//!
//! - ファイルは呼び出し側が指定する上限（`max_bytes`）以内でメモリへ読み切り、そのバイト列で判定する。
//!   ONNX は top-level の tag と長さだけをたどり、LEN の中身は解釈しない。
//!   pickle 等の中身は解釈・展開・実行しない
//! - ONNX の判定は ModelProto と `graph`（GraphProto）の top-level 構造の「形の検査」であり、
//!   意味の検証ではない（空の graph・output の無い graph は拒否）。node / initializer の中身は検査しない。
//!   最終的な解析の成否は読み込み時の ONNX Runtime に委ねる。
//!   一方、テンソルの外部データ参照（`external_data`・`data_location`）は graph・サブグラフ・functions・
//!   training_info 内の TensorProto をすべて降りて検出し、1 件でもあれば拒否する（参照先は追わない）
//! - 経路の閉じ込め（TASK-39.4）は本モジュールの責務外。サイズ上限の値の決定（TASK-39.5）も
//!   呼び出し側（CLI の統合は TASK-39.2-4・#156）が行い、本モジュールは渡された上限を強制する
//! - 拡張子と内容の照合による偽装拒否は `model_file` モジュール（TASK-39.2-2・#154）で上に重ねた
//!
//! pickle プロトコル 0/1（テキスト opcode 始まり）は誤検出が多いため固定シグネチャでは
//! 判定せず、ONNX の構造検査にも通らないので `Unknown` として拒否される。

use fandhe_edge_core::exitcode::ExitCode;
use fandhe_edge_core::fs::{FsError, read_bounded};
use std::collections::BTreeSet;
use std::fmt;
use std::io::{self, SeekFrom};
use std::path::Path;

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
/// 26 configuration（repeated DeviceConfigurationProto。IR 10 以降）を含む。ローカルの onnx 1.23.0 の
/// onnx.proto から ModelProto の全フィールド（1〜8・14・20・25・26）を確認済み。
fn model_field_wire_type(field: u64) -> Option<u64> {
    match field {
        1 | 5 => Some(0),
        2 | 3 | 4 | 6 | 7 | 8 | 14 | 20 | 25 | 26 => Some(2),
        _ => None,
    }
}

/// varint の読み取り失敗の種類。
enum VarintError {
    /// 10 バイト以内に終わらない・10 バイト目が 0x00/0x01 以外（u64 に収まらない）。
    Invalid,
    /// 入力が途中で尽きた（入力元が `limited` なら判定用 prefix の打ち切り、そうでなければ実ファイル末尾）。
    Short,
}

/// varint を読み `(値, 消費バイト数)` を返す。
fn read_varint(buf: &[u8]) -> Result<(u64, usize), VarintError> {
    let mut value: u64 = 0;
    for i in 0..10usize {
        let Some(&b) = buf.get(i) else {
            return Err(VarintError::Short);
        };
        let shift = (i as u32).saturating_mul(7);
        // 10 バイト目は u64 の最上位 1 ビットだけを持てる。0x00・0x01 以外は u64 に収まらない
        // 値（または継続ビット付き）であり、切り詰めて受理せず拒否する。
        if i == 9 && b > 0x01 {
            return Err(VarintError::Invalid);
        }
        value |= u64::from(b & 0x7f).checked_shl(shift).unwrap_or(0);
        if b & 0x80 == 0 {
            return Ok((value, i.saturating_add(1)));
        }
    }
    Err(VarintError::Invalid)
}

/// ONNX 走査が任意オフセットの少量のバイトを読むための入力元。
trait ByteSource {
    /// `offset` から最大 `buf.len()` バイトを読み、読めた長さを返す。実ファイル末尾で短くなる。
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<usize>;
    /// 短い読み取りが判定用 prefix の人為的な打ち切りに由来しうるなら真
    /// （実ファイル末尾での切断と区別する）。
    fn limited(&self) -> bool;
}

/// 先頭 prefix だけを持つ入力元（純関数 [`detect_format`] 用）。
struct SliceSource<'a> {
    prefix: &'a [u8],
    total_len: u64,
}

impl ByteSource for SliceSource<'_> {
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        let Ok(start) = usize::try_from(offset) else {
            return Ok(0);
        };
        let Some(rest) = self.prefix.get(start..) else {
            return Ok(0);
        };
        let n = rest.len().min(buf.len());
        if let (Some(dst), Some(src)) = (buf.get_mut(..n), rest.get(..n)) {
            dst.copy_from_slice(src);
        }
        Ok(n)
    }

    fn limited(&self) -> bool {
        u64::try_from(self.prefix.len()).is_ok_and(|l| l < self.total_len)
    }
}

/// ONNX 走査の結果。
enum OnnxScan {
    /// ModelProto の形を最後まで確認できた。
    Onnx,
    /// ONNX ではない（不正な構造・実ファイル末尾での切断を含む）。
    NotOnnx,
    /// 判定用 prefix の打ち切り・走査回数の上限に達し、続きは未検査。
    Exhausted,
}

/// トップレベルのフィールド数の上限（`metadata_props` 等の繰り返しを考慮した余裕値）。
/// 超えたら未検査として扱う（無限走査を作らない。REQ-39）。
const MAX_TOP_LEVEL_FIELDS: usize = 65_536;

/// GraphProto 内のフィールド数の上限（`node` の繰り返しを考慮した余裕値）。超えたら未検査。
const MAX_GRAPH_FIELDS: usize = 1 << 20;

/// ValueInfoProto 内のフィールド数の上限（`metadata_props` の繰り返しを考慮した余裕値）。超えたら未検査。
const MAX_VALUE_INFO_FIELDS: usize = 4096;

/// GraphProto（onnx.proto3）のフィールドの wire type。全て LEN(2)。未知の field 番号は `None`。
/// 1 node・2 name・5 initializer・10 doc_string・11 input・12 output・13 value_info・
/// 14 quantization_annotation・15 sparse_initializer・16 metadata_props。
fn graph_field_wire_type(field: u64) -> Option<u64> {
    match field {
        1 | 2 | 5 | 10 | 11 | 12 | 13 | 14 | 15 | 16 => Some(2),
        _ => None,
    }
}

/// 1 フィールド分の `(field, wire, 値, 値の直後の位置)`。読めない場合は走査結果で打ち切る。
type FieldStep = Result<(u64, u64, u64, u64), OnnxScan>;

/// `pos` のフィールドの tag と最初の varint（varint 値、または LEN の長さ）を読む。
fn read_field<S: ByteSource>(src: &mut S, pos: u64) -> io::Result<FieldStep> {
    let mut buf = [0u8; 10];
    let n = src.read_at(pos, &mut buf)?;
    let Some(head) = buf.get(..n) else {
        return Ok(Err(OnnxScan::NotOnnx));
    };
    let (tag, tag_len) = match read_varint(head) {
        Ok(v) => v,
        Err(VarintError::Short) if src.limited() => return Ok(Err(OnnxScan::Exhausted)),
        Err(_) => return Ok(Err(OnnxScan::NotOnnx)),
    };
    let body_pos = pos.saturating_add(tag_len as u64);
    let mut vbuf = [0u8; 10];
    let vn = src.read_at(body_pos, &mut vbuf)?;
    let Some(vhead) = vbuf.get(..vn) else {
        return Ok(Err(OnnxScan::NotOnnx));
    };
    let (v, m) = match read_varint(vhead) {
        Err(VarintError::Short) if src.limited() => return Ok(Err(OnnxScan::Exhausted)),
        Err(_) => return Ok(Err(OnnxScan::NotOnnx)),
        Ok(v) => v,
    };
    Ok(Ok((
        tag >> 3,
        tag & 7,
        v,
        body_pos.saturating_add(m as u64),
    )))
}

/// ValueInfoProto（onnx.proto3）のフィールドの wire type。全て LEN(2)。未知の field 番号は `None`。
/// 1 name・2 type・3 doc_string・4 metadata_props。
fn value_info_field_wire_type(field: u64) -> Option<u64> {
    match field {
        1..=4 => Some(2),
        _ => None,
    }
}

/// ValueInfoProto の本体 `[start, end)` の top-level 構造を検査する。
/// 全フィールドが既知の LEN で `end` ちょうどで終わり、空でない `name`（field 1）を持てば妥当な形。
fn scan_value_info<S: ByteSource>(src: &mut S, start: u64, end: u64) -> io::Result<OnnxScan> {
    let mut pos = start;
    let mut seen_name = false;
    for _ in 0..MAX_VALUE_INFO_FIELDS {
        if pos == end {
            return Ok(if seen_name {
                OnnxScan::Onnx
            } else {
                OnnxScan::NotOnnx
            });
        }
        let (field, wire, len, after) = match read_field(src, pos)? {
            Ok(v) => v,
            Err(r) => return Ok(r),
        };
        if value_info_field_wire_type(field) != Some(wire) {
            return Ok(OnnxScan::NotOnnx);
        }
        let Some(next) = after.checked_add(len) else {
            return Ok(OnnxScan::NotOnnx);
        };
        if next > end {
            return Ok(OnnxScan::NotOnnx);
        }
        if field == 1 && len > 0 {
            seen_name = true;
        }
        pos = next;
    }
    Ok(OnnxScan::Exhausted)
}

/// GraphProto の本体 `[start, end)` を、LEN の中身を飛ばしながら走査する。
/// 全フィールドが既知の LEN で `end` を超えず、ちょうど `end` で終わり、
/// `output`（field 12。有効なグラフは 1 つ以上の出力を持つ）が現れ、各 output が有効な
/// ValueInfoProto（空でない name を持つ）であれば妥当な形。
fn scan_graph_proto<S: ByteSource>(src: &mut S, start: u64, end: u64) -> io::Result<OnnxScan> {
    let mut pos = start;
    let mut seen_output = false;
    for _ in 0..MAX_GRAPH_FIELDS {
        if pos == end {
            return Ok(if seen_output {
                OnnxScan::Onnx
            } else {
                OnnxScan::NotOnnx
            });
        }
        let (field, wire, len, after) = match read_field(src, pos)? {
            Ok(v) => v,
            Err(r) => return Ok(r),
        };
        if graph_field_wire_type(field) != Some(wire) {
            return Ok(OnnxScan::NotOnnx);
        }
        let Some(next) = after.checked_add(len) else {
            return Ok(OnnxScan::NotOnnx);
        };
        if next > end {
            return Ok(OnnxScan::NotOnnx);
        }
        if field == 12 {
            // output は ValueInfoProto。必須の name を持たない空の output を許可しない。
            match scan_value_info(src, after, next)? {
                OnnxScan::Onnx => seen_output = true,
                other => return Ok(other),
            }
        }
        pos = next;
    }
    Ok(OnnxScan::Exhausted)
}

/// ModelProto の top-level フィールドを、LEN の中身を飛ばしながら末尾まで走査し、ONNX の形かを判定する。
/// 先頭が `ir_version`（tag 0x08・値 1..=0xFFFF）で、全フィールドが既知の (field, wire type) であり、
/// LEN が `total_len` を超えず、`graph`（field 7）が現れ、ちょうど `total_len` で終われば ONNX。
/// `graph` の本体は [`scan_graph_proto`] で GraphProto の top-level 構造まで検査する
/// （空の graph・未知フィールドを含む graph は拒否）。
fn scan_onnx_model<S: ByteSource>(src: &mut S, total_len: u64) -> io::Result<OnnxScan> {
    let mut pos: u64 = 0;
    let mut seen_graph = false;
    let mut seen_ir_version = false;
    for _ in 0..MAX_TOP_LEVEL_FIELDS {
        if pos == total_len {
            return Ok(if seen_ir_version && seen_graph {
                OnnxScan::Onnx
            } else {
                OnnxScan::NotOnnx
            });
        }
        if pos == 0 {
            let mut first = [0u8; 1];
            src.read_at(0, &mut first)?;
            if first != [0x08] {
                return Ok(OnnxScan::NotOnnx);
            }
        }
        let (field, wire, v, after) = match read_field(src, pos)? {
            Ok(x) => x,
            Err(r) => return Ok(r),
        };
        if model_field_wire_type(field) != Some(wire) {
            return Ok(OnnxScan::NotOnnx);
        }
        if wire == 0 {
            if field == 1 {
                if v == 0 || v > 0xFFFF {
                    return Ok(OnnxScan::NotOnnx);
                }
                seen_ir_version = true;
            }
            pos = after;
        } else {
            let Some(end) = after.checked_add(v) else {
                return Ok(OnnxScan::NotOnnx);
            };
            if end > total_len {
                return Ok(OnnxScan::NotOnnx);
            }
            if field == 7 {
                match scan_graph_proto(src, after, end)? {
                    OnnxScan::Onnx => seen_graph = true,
                    other => return Ok(other),
                }
            }
            pos = end;
        }
    }
    Ok(OnnxScan::Exhausted)
}

/// 固定シグネチャ形式の判定。該当しなければ `None`。
fn detect_signature(prefix: &[u8]) -> Option<FileFormat> {
    if let [0x80, ver, ..] = prefix
        && (2..=5).contains(ver)
    {
        return Some(FileFormat::Pickle);
    }
    if prefix.starts_with(b"\x93NUMPY") {
        return Some(FileFormat::Npy);
    }
    if prefix.starts_with(b"PK\x03\x04") || prefix.starts_with(b"PK\x05\x06") {
        return Some(FileFormat::Zip);
    }
    if prefix.starts_with(b"GGUF") {
        return Some(FileFormat::Gguf);
    }
    None
}

/// テンソルの外部データ参照の検査で許すサブグラフの入れ子の深さ（超えたら拒否）。
const MAX_TENSOR_SCAN_DEPTH: usize = 32;

/// テンソルの外部データ参照の検査で走査するフィールド数の上限（超えたら拒否。REQ-39）。
const MAX_TENSOR_SCAN_FIELDS: usize = 1 << 22;

/// 外部データ参照の検査の走査状態。
struct ExternalScan {
    remaining_fields: usize,
}

/// 1 フィールド分の値。
enum WireValue<'a> {
    Varint(u64),
    Len(&'a [u8]),
    Fixed,
}

impl ExternalScan {
    /// `buf` の全フィールドを走査して `f(field, value)` を呼ぶ。形が壊れている・上限超過は `Err`。
    fn each_field<'a>(
        &mut self,
        buf: &'a [u8],
        mut f: impl FnMut(&mut Self, u64, WireValue<'a>) -> Result<(), ()>,
    ) -> Result<(), ()> {
        let mut pos = 0usize;
        while pos < buf.len() {
            self.remaining_fields = self.remaining_fields.checked_sub(1).ok_or(())?;
            let rest = buf.get(pos..).ok_or(())?;
            let (tag, n) = read_varint(rest).map_err(|_| ())?;
            pos = pos.checked_add(n).ok_or(())?;
            let rest = buf.get(pos..).ok_or(())?;
            let value = match tag & 7 {
                0 => {
                    let (v, n) = read_varint(rest).map_err(|_| ())?;
                    pos = pos.checked_add(n).ok_or(())?;
                    WireValue::Varint(v)
                }
                1 | 5 => {
                    let n = if tag & 7 == 1 { 8 } else { 4 };
                    pos = pos.checked_add(n).ok_or(())?;
                    if pos > buf.len() {
                        return Err(());
                    }
                    WireValue::Fixed
                }
                2 => {
                    let (len, n) = read_varint(rest).map_err(|_| ())?;
                    let start = pos.checked_add(n).ok_or(())?;
                    let end = start
                        .checked_add(usize::try_from(len).map_err(|_| ())?)
                        .ok_or(())?;
                    let body = buf.get(start..end).ok_or(())?;
                    pos = end;
                    WireValue::Len(body)
                }
                _ => return Err(()),
            };
            f(self, tag >> 3, value)?;
        }
        Ok(())
    }

    /// ModelProto: graph(7)・training_info(20)・functions(25) 内のテンソルを検査する。
    fn model(&mut self, buf: &[u8]) -> Result<(), ()> {
        self.each_field(buf, |me, field, v| match (field, v) {
            (7, WireValue::Len(b)) => me.graph(b, 1),
            (20, WireValue::Len(b)) => me.each_field(b, |me, f, v| match (f, v) {
                (1 | 2, WireValue::Len(g)) => me.graph(g, 1),
                _ => Ok(()),
            }),
            (25, WireValue::Len(b)) => me.each_field(b, |me, f, v| match (f, v) {
                (7, WireValue::Len(n)) => me.node(n, 1),
                (11, WireValue::Len(a)) => me.attribute(a, 1),
                _ => Ok(()),
            }),
            _ => Ok(()),
        })
    }

    /// GraphProto: node(1)・initializer(5)・sparse_initializer(15)。
    fn graph(&mut self, buf: &[u8], depth: usize) -> Result<(), ()> {
        if depth > MAX_TENSOR_SCAN_DEPTH {
            return Err(());
        }
        self.each_field(buf, |me, field, v| match (field, v) {
            (1, WireValue::Len(b)) => me.node(b, depth),
            (5, WireValue::Len(b)) => me.tensor(b),
            (15, WireValue::Len(b)) => me.sparse(b),
            _ => Ok(()),
        })
    }

    /// NodeProto: attribute(5)。
    fn node(&mut self, buf: &[u8], depth: usize) -> Result<(), ()> {
        self.each_field(buf, |me, field, v| match (field, v) {
            (5, WireValue::Len(b)) => me.attribute(b, depth),
            _ => Ok(()),
        })
    }

    /// AttributeProto: t(5)・tensors(10)・sparse_tensor(22)・sparse_tensors(23)・g(6)・graphs(11)。
    fn attribute(&mut self, buf: &[u8], depth: usize) -> Result<(), ()> {
        self.each_field(buf, |me, field, v| match (field, v) {
            (5 | 10, WireValue::Len(b)) => me.tensor(b),
            (22 | 23, WireValue::Len(b)) => me.sparse(b),
            (6 | 11, WireValue::Len(b)) => me.graph(b, depth.saturating_add(1)),
            _ => Ok(()),
        })
    }

    /// SparseTensorProto: values(1)・indices(2)。
    fn sparse(&mut self, buf: &[u8]) -> Result<(), ()> {
        self.each_field(buf, |me, field, v| match (field, v) {
            (1 | 2, WireValue::Len(b)) => me.tensor(b),
            _ => Ok(()),
        })
    }

    /// TensorProto: external_data(13) が 1 件でもある、または data_location(14) が DEFAULT(0) 以外なら拒否。
    fn tensor(&mut self, buf: &[u8]) -> Result<(), ()> {
        self.each_field(buf, |_, field, v| match (field, v) {
            (13, _) => Err(()),
            (14, WireValue::Varint(0)) => Ok(()),
            (14, _) => Err(()),
            _ => Ok(()),
        })
    }
}

/// ModelProto 内のすべての TensorProto を降りて、外部データ（外部ファイル）への参照が無いことを確認する。
/// 参照先を追って検査することはせず、参照があれば拒否する（経路の閉じ込め・サイズ上限の迂回を防ぐ。REQ-39）。
fn has_no_external_data(bytes: &[u8]) -> bool {
    ExternalScan {
        remaining_fields: MAX_TENSOR_SCAN_FIELDS,
    }
    .model(bytes)
    .is_ok()
}

/// バイト列全体からファイル形式を判定する純関数（非公開。公開すると検査を経ない判定値が出回る）。
/// 判定順は固定シグネチャ → ONNX 構造検査 → `Unknown`。`bytes` はファイル全体でなければならず、
/// 全バイトを走査して「ちょうど末尾で終わる」ことを確認する。長さ超過・途中切れ・走査回数の上限で
/// 確認しきれない場合は `Unknown`（fail-closed。REQ-39）。
fn detect_format(bytes: &[u8]) -> FileFormat {
    if let Some(f) = detect_signature(bytes) {
        return f;
    }
    let total_len = bytes.len() as u64;
    let mut src = SliceSource {
        prefix: bytes,
        total_len,
    };
    match scan_onnx_model(&mut src, total_len) {
        Ok(OnnxScan::Onnx) if has_no_external_data(bytes) => FileFormat::Onnx,
        _ => FileFormat::Unknown,
    }
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
    ///
    /// 非公開: 公開すると検査を経ない形式で許可済みを装える。
    /// 許可済みの証明は [`open_checked_file`] が返す [`CheckedFile`] だけが持つ（REQ-39）。
    fn check(&self, detected: FileFormat) -> Result<FileFormat, FormatRejection> {
        if self.contains(detected) {
            Ok(detected)
        } else {
            Err(FormatRejection::NotAllowed {
                detected,
                allowed: self.formats.iter().copied().collect(),
            })
        }
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
    /// モデルファイルの拡張子が許可されていない（TASK-39.2-2・#154）。
    /// `expected` は常に定数で、利用者由来の拡張子は保持しない。
    ExtensionNotAllowed { expected: &'static str },
    /// ファイルを開けない・通常ファイルでない・読み込めない。
    Io(FsError),
}

impl FormatRejection {
    /// 終了コード（REQ-21）。形式不許可・通常ファイルでない・存在しない → `InvalidInput`、
    /// サイズ超過 → `LimitExceeded`、その他の I/O 失敗 → `RuntimeError`。
    pub fn exit_code(&self) -> ExitCode {
        match self {
            FormatRejection::NotAllowed { .. } => ExitCode::InvalidInput,
            FormatRejection::ExtensionNotAllowed { .. } => ExitCode::InvalidInput,
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
            FormatRejection::ExtensionNotAllowed { .. } => "extension_not_allowed",
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
            FormatRejection::ExtensionNotAllowed { expected } => {
                write!(f, "file extension is not allowed (expected: .{expected})")
            }
            FormatRejection::Io(_) => write!(f, "file could not be opened as a regular file"),
        }
    }
}

impl std::error::Error for FormatRejection {}

/// 形式検査を通したバイト列。検査に使ったバイト列そのものをメモリに保持して渡すための型。
///
/// ファイルはハンドルを保持しても別の書き込みハンドルから上書きできる（検査後の内容差し替え）。
/// そのため検査対象を不変のメモリ上のバッファに読み切り、利用側にはこのバッファだけを渡す。
/// 利用側はパスを開き直さず、[`CheckedFile::as_bytes`]・[`std::io::Read`]・[`std::io::Seek`]
/// 経由で読む（REQ-39「形式の許可制」）。
#[derive(Debug)]
pub struct CheckedFile {
    cursor: io::Cursor<Vec<u8>>,
    format: FileFormat,
}

impl CheckedFile {
    /// 検査を通った形式。形式の証明は `CheckedFile` の中にだけあり、バイト列と切り離して取り出せない
    /// （`FileFormat` は判定結果のラベルで、許可済みの証明ではない）。
    pub const fn format(&self) -> FileFormat {
        self.format
    }

    /// 検査したバイト列全体。
    pub fn as_bytes(&self) -> &[u8] {
        self.cursor.get_ref()
    }
}

impl io::Read for CheckedFile {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.cursor.read(buf)
    }
}

impl io::Seek for CheckedFile {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.cursor.seek(pos)
    }
}

/// ファイルを `max_bytes` を上限にメモリへ読み切り、そのバイト列から形式を判定して許可リストと照合する。
///
/// 通った場合は検査したバイト列を保持する [`CheckedFile`] を返す。後続の読み込みは返された
/// バイト列で行い、パスを開き直さない（検査後の書き換え・差し替えへの対策。REQ-39）。
/// 上限超過は `FsError::TooLarge`（終了コード `LimitExceeded`）。経路の検証は責務外
/// （モジュール doc 参照）。
pub fn open_checked_file(
    path: &Path,
    allowlist: &FormatAllowlist,
    max_bytes: u64,
) -> Result<CheckedFile, FormatRejection> {
    let bytes = read_bounded(path, max_bytes).map_err(FormatRejection::Io)?;
    let detected = detect_format(&bytes);
    let format = allowlist.check(detected)?;
    Ok(CheckedFile {
        cursor: io::Cursor::new(bytes),
        format,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(b: &[u8]) -> FileFormat {
        detect_format(b)
    }

    /// REQ-39・TASK-39.2-1: 最小の ONNX 形と実物に近い先頭を Onnx と判定する。
    #[test]
    fn req39_detects_onnx() {
        assert_eq!(
            d(&[0x08, 0x07, 0x3a, 0x05, 0x62, 0x03, 0x0a, 0x01, 0x78]),
            FileFormat::Onnx
        );
        let mut b = vec![0x08, 0x03, 0x12, 7];
        b.extend_from_slice(b"pytorch");
        b.extend_from_slice(&[0x1a, 3]);
        b.extend_from_slice(b"0.3");
        b.extend_from_slice(&[0x3a, 5, 0x62, 0x03, 0x0a, 0x01, 0x78]);
        assert_eq!(d(&b), FileFormat::Onnx);
    }

    /// REQ-39・TASK-39.2-1: graph の中身も検査する。空の graph・output の無い graph・
    /// 未知フィールドや LEN 超過を含む graph は ONNX と認めない。
    #[test]
    fn req39_graph_contents_are_validated() {
        assert_eq!(d(&[0x08, 0x07, 0x3a, 0x00]), FileFormat::Unknown); // 空の graph
        assert_eq!(
            d(&[0x08, 0x07, 0x3a, 0x02, 0x0a, 0x00]),
            FileFormat::Unknown
        ); // output なし
        assert_eq!(
            d(&[0x08, 0x07, 0x3a, 0x02, 0x7a, 0x00]),
            FileFormat::Unknown
        ); // output 以外だけ（field 15）
        assert_eq!(
            d(&[0x08, 0x07, 0x3a, 0x02, 0x08, 0x01]),
            FileFormat::Unknown
        ); // varint は graph に無い
        assert_eq!(
            d(&[0x08, 0x07, 0x3a, 0x02, 0x62, 0x05]),
            FileFormat::Unknown
        ); // 子の LEN が graph 超過
        assert_eq!(
            d(&[
                0x08, 0x07, 0x3a, 0x07, 0x0a, 0x00, 0x62, 0x03, 0x0a, 0x01, 0x78
            ]),
            FileFormat::Onnx
        );
    }

    /// REQ-39・TASK-39.2-1: output の ValueInfoProto は空でない name が必須。空の output・
    /// name 無し・空 name・未知フィールドを含む output は ONNX と認めない。
    #[test]
    fn req39_output_value_info_is_validated() {
        // 空の output（LEN 0）
        assert_eq!(
            d(&[0x08, 0x07, 0x3a, 0x02, 0x62, 0x00]),
            FileFormat::Unknown
        );
        // name 無し（field 2 type だけ）
        assert_eq!(
            d(&[0x08, 0x07, 0x3a, 0x04, 0x62, 0x02, 0x12, 0x00]),
            FileFormat::Unknown
        );
        // 空 name
        assert_eq!(
            d(&[0x08, 0x07, 0x3a, 0x04, 0x62, 0x02, 0x0a, 0x00]),
            FileFormat::Unknown
        );
        // 未知フィールド（field 5）
        assert_eq!(
            d(&[
                0x08, 0x07, 0x3a, 0x07, 0x62, 0x05, 0x0a, 0x01, 0x78, 0x2a, 0x00
            ]),
            FileFormat::Unknown
        );
        // 有効な output は通る
        assert_eq!(
            d(&[0x08, 0x07, 0x3a, 0x05, 0x62, 0x03, 0x0a, 0x01, 0x78]),
            FileFormat::Onnx
        );
    }

    /// REQ-39・TASK-39.2-1: 許可済みの型はファイルの内容検査でのみ得られ、pickle は拒否される。
    #[test]
    fn req39_open_checked_file_inspects_content() {
        let dir = std::env::temp_dir().join(format!("fe-guard-fmt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let onnx = dir.join("m.onnx");
        std::fs::write(
            &onnx,
            [0x08, 0x07, 0x3a, 0x05, 0x62, 0x03, 0x0a, 0x01, 0x78],
        )
        .unwrap();
        let pkl = dir.join("m.onnx.pkl");
        std::fs::write(&pkl, [0x80, 0x04, 0x95]).unwrap();
        let al = FormatAllowlist::onnx_only();
        let ok = open_checked_file(&onnx, &al, 1 << 20).unwrap();
        assert_eq!(ok.format(), FileFormat::Onnx);
        let err = open_checked_file(&pkl, &al, 1 << 20).unwrap_err();
        assert_eq!(err.exit_code(), ExitCode::InvalidInput);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// protobuf の LEN フィールドを組み立てる試験用ヘルパー（tag・長さとも varint で書く）。
    fn ld(field: u8, payload: &[u8]) -> Vec<u8> {
        fn varint(v: &mut Vec<u8>, mut n: usize) {
            while n >= 0x80 {
                v.push((n & 0x7f) as u8 | 0x80);
                n >>= 7;
            }
            v.push(n as u8);
        }
        let mut v = Vec::new();
        varint(&mut v, (usize::from(field) << 3) | 2);
        varint(&mut v, payload.len());
        v.extend_from_slice(payload);
        v
    }

    /// 有効な output だけを持つ graph 本体に `extra` を足した ONNX を作る。
    fn model_with_graph_extra(extra: &[u8]) -> Vec<u8> {
        let mut g = ld(12, &[0x0a, 0x01, 0x78]);
        g.extend_from_slice(extra);
        let mut m = vec![0x08, 0x07];
        m.extend(ld(7, &g));
        m
    }

    fn tensor_with_external() -> Vec<u8> {
        ld(13, &[0x0a, 0x01, 0x6b]) // external_data(key)
    }

    /// REQ-39・TASK-39.2-1: 外部データを参照するテンソルは、どこに現れても ONNX と認めない。
    #[test]
    fn req39_external_data_is_rejected_everywhere() {
        let ext = tensor_with_external();
        let loc = vec![0x70, 0x01]; // data_location = EXTERNAL
        let clean_tensor = vec![0x40, 0x01]; // data_type=1（外部参照なし）
        // 通常モデル（外部参照の無い initializer）は通る。
        assert_eq!(
            d(&model_with_graph_extra(&ld(5, &clean_tensor))),
            FileFormat::Onnx
        );
        // data_location=DEFAULT(0) は通る。
        assert_eq!(
            d(&model_with_graph_extra(&ld(5, &[0x70, 0x00]))),
            FileFormat::Onnx
        );
        // initializer の external_data・data_location=EXTERNAL
        assert_eq!(
            d(&model_with_graph_extra(&ld(5, &ext))),
            FileFormat::Unknown
        );
        assert_eq!(
            d(&model_with_graph_extra(&ld(5, &loc))),
            FileFormat::Unknown
        );
        // sparse_initializer の values・indices
        for f in [1u8, 2] {
            let sp = ld(f, &ext);
            assert_eq!(
                d(&model_with_graph_extra(&ld(15, &sp))),
                FileFormat::Unknown
            );
        }
        // node の attribute の t・tensors・sparse_tensor・sparse_tensors
        for (af, wrap_sparse) in [(5u8, false), (10, false), (22, true), (23, true)] {
            let inner = if wrap_sparse {
                ld(1, &ext)
            } else {
                ext.clone()
            };
            let node = ld(5, &ld(af, &inner));
            assert_eq!(
                d(&model_with_graph_extra(&ld(1, &node))),
                FileFormat::Unknown
            );
        }
        // サブグラフ（属性 g・graphs）内の外部参照
        let sub = ld(5, &ext);
        for af in [6u8, 11] {
            let node = ld(5, &ld(af, &sub));
            assert_eq!(
                d(&model_with_graph_extra(&ld(1, &node))),
                FileFormat::Unknown
            );
        }
        // functions 内のノードの属性・attribute_proto、training_info の graph
        let node = ld(5, &ld(5, &ext));
        let mut m = model_with_graph_extra(&[]);
        m.extend(ld(25, &ld(7, &node)));
        assert_eq!(d(&m), FileFormat::Unknown);
        let mut m = model_with_graph_extra(&[]);
        m.extend(ld(25, &ld(11, &ld(5, &ext))));
        assert_eq!(d(&m), FileFormat::Unknown);
        let mut m = model_with_graph_extra(&[]);
        m.extend(ld(20, &ld(2, &ld(5, &ext))));
        assert_eq!(d(&m), FileFormat::Unknown);
        // 外部参照の無い functions は通る。
        let mut m = model_with_graph_extra(&[]);
        m.extend(ld(25, &ld(7, &ld(5, &ld(5, &clean_tensor)))));
        assert_eq!(d(&m), FileFormat::Onnx);
    }

    /// REQ-39・TASK-39.2-1: サブグラフの入れ子が深さの上限を超えたら拒否し、上限以内なら通る。
    #[test]
    fn req39_subgraph_depth_limit() {
        fn nested(levels: usize) -> Vec<u8> {
            // graph の node の attribute g に graph を `levels` 段入れる（長さ varint を 2 バイトで書く）。
            let mut inner: Vec<u8> = Vec::new();
            for _ in 0..levels {
                let attr = ld(6, &inner);
                let node = ld(5, &attr);
                inner = ld(1, &node);
            }
            inner
        }
        let build = |levels: usize| {
            let mut g = ld(12, &[0x0a, 0x01, 0x78]);
            g.extend(nested(levels));
            let mut m = vec![0x08, 0x07];
            m.extend(ld(7, &g));
            m
        };
        assert_eq!(d(&build(10)), FileFormat::Onnx);
        assert_eq!(d(&build(40)), FileFormat::Unknown);
    }

    /// REQ-39・TASK-39.2-1: ModelProto の field 26（configuration。LEN）を持つ ONNX を通し、
    /// wire type が違う field 26（varint）は拒否する。
    #[test]
    fn req39_model_field_26_configuration() {
        let base = [0x08, 0x07, 0x3a, 0x05, 0x62, 0x03, 0x0a, 0x01, 0x78];
        let mut ok = base.to_vec();
        ok.extend_from_slice(&[0xd2, 0x01, 0x02, 0x0a, 0x00]); // field 26, LEN 2
        assert_eq!(d(&ok), FileFormat::Onnx);
        let mut bad = base.to_vec();
        bad.extend_from_slice(&[0xd0, 0x01, 0x01]); // field 26, varint
        assert_eq!(d(&bad), FileFormat::Unknown);
    }

    /// REQ-39・TASK-39.2-1: graph の後ろに未検査の末尾（長さ付きフィールドの中身が無い・LEN が
    /// 実サイズ超過）を持つバイト列は ONNX と認めない（全バイトが検査できたものだけを通す）。
    #[test]
    fn req39_unverified_tail_is_never_onnx() {
        let mut b = vec![0x08, 0x07, 0x3a, 0x05, 0x62, 0x03, 0x0a, 0x01, 0x78];
        assert_eq!(d(&b), FileFormat::Onnx);
        b.extend_from_slice(&[0x32, 0x64]); // doc_string LEN 100 だが中身が無い
        assert_eq!(d(&b), FileFormat::Unknown);
        assert_eq!(
            d(&[0x08, 0x07, 0x3a, 0x80, 0x80, 0x80, 0x04]),
            FileFormat::Unknown
        );
    }

    /// REQ-39・TASK-39.2-1: 実ファイル末尾で切れた protobuf は、graph の後でも拒否する。
    #[test]
    fn req39_truncated_at_real_eof_after_graph_is_unknown() {
        assert_eq!(
            d(&[
                0x08, 0x07, 0x3a, 0x05, 0x62, 0x03, 0x0a, 0x01, 0x78, 0x12, 0x80
            ]),
            FileFormat::Unknown
        ); // 値の varint が途中
        assert_eq!(
            d(&[0x08, 0x07, 0x3a, 0x05, 0x62, 0x03, 0x0a, 0x01, 0x78, 0x92]),
            FileFormat::Unknown
        ); // tag が途中
        assert_eq!(
            d(&[0x08, 0x07, 0x3a, 0x05, 0x62, 0x03, 0x0a, 0x01, 0x78, 0x12]),
            FileFormat::Unknown
        ); // 長さが無い
        assert_eq!(
            d(&[0x08, 0x07, 0x3a, 0x05, 0x62, 0x03, 0x0a, 0x01, 0x78, 0x08]),
            FileFormat::Unknown
        ); // 値が無い
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
            d(&[
                0x08, 0x07, 0x78, 0x01, 0x3a, 0x05, 0x62, 0x03, 0x0a, 0x01, 0x78
            ]),
            FileFormat::Unknown
        ); // 未知 field 15
        assert_eq!(
            d(&[0x08, 0x00, 0x3a, 0x05, 0x62, 0x03, 0x0a, 0x01, 0x78]),
            FileFormat::Unknown
        ); // ir_version 0
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
        // model_version（field 5・varint）の 10 バイト varint を経由して graph まで完全に走査する。
        let mut bad = vec![0x08, 0x07, 0x28];
        bad.extend_from_slice(&[0x80; 9]);
        bad.push(0x02);
        bad.extend_from_slice(&[0x3a, 0x05, 0x62, 0x03, 0x0a, 0x01, 0x78]);
        assert_eq!(d(&bad), FileFormat::Unknown);
        let mut ok = vec![0x08, 0x07, 0x28];
        ok.extend_from_slice(&[0x80; 9]);
        ok.push(0x01);
        ok.extend_from_slice(&[0x3a, 0x05, 0x62, 0x03, 0x0a, 0x01, 0x78]);
        assert_eq!(d(&ok), FileFormat::Onnx);
    }

    /// REQ-39・TASK-39.2-1: onnx_only は ONNX だけを通し、他は InvalidInput(64) で拒否する。
    #[test]
    fn req39_onnx_only_allowlist() {
        let al = FormatAllowlist::onnx_only();
        let ok = al.check(FileFormat::Onnx).unwrap();
        assert_eq!(ok, FileFormat::Onnx);
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

    /// REQ-39・TASK-39.2-2: 拡張子拒否の契約（64・理由コード・固定メッセージ）。
    #[test]
    fn req39_extension_not_allowed_contract() {
        let err = FormatRejection::ExtensionNotAllowed { expected: "onnx" };
        assert_eq!(err.exit_code().code(), 64);
        assert_eq!(err.reason_code(), "extension_not_allowed");
        assert_eq!(
            err.to_string(),
            "file extension is not allowed (expected: .onnx)"
        );
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
