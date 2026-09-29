//! ONNX の protobuf メッセージのうち、C1・C3 の照合と実行に必要なフィールドだけを所有型へ
//! 復号する（REQ-32・REQ-39・TASK-32.1-2・#113）。
//!
//! # 役割と許可制
//!
//! [`super::c1`]・[`super::c3`] のテンプレート照合が読む入力表現。復号の段階で、任意の
//! グラフ・外部ファイル参照を実行系へ持ち込まないために次を拒否する（許可リスト方式。
//! 拒否は [`OnnxLoadError`] の分類で返し、本文・重みは保持しない）。
//!
//! - 部分グラフ・テンソル・型情報を値に持つ属性（GRAPH・TENSOR・SPARSE_TENSOR 等。再帰を作らない）
//! - `raw_data` 以外の形で値を持つテンソル、`external_data` / `data_location`（モデル外のファイル
//!   参照による経路外読み込みを塞ぐ）、FLOAT・INT64 以外の要素型
//! - ノード数・initializer 数・入出力名数・名前の長さ・テンソルの次元数が上限を超えるモデル
//!
//! 要素数は次元の積を `checked_mul` で求め、`raw_data` の長さと厳密に一致することを確認してから
//! 復号する。全体の量は呼び出し元が課すファイルサイズ上限で有界（[`super::MAX_MODEL_FILE_BYTES`]）。
//! 外部入力の経路のため `unwrap`・添字アクセスを使わない。

use super::OnnxLoadError;
use super::wire::{Reader, WireType};

/// ノード数の上限（C3 の最大構成は約 42、C1 は 14。余裕を見た暫定値。REQ-39）。
pub(super) const MAX_NODES: usize = 128;
/// initializer 数の上限（C3 の最大構成は約 23。REQ-39）。
pub(super) const MAX_INITIALIZERS: usize = 64;
/// 1 ノードの属性数の上限（REQ-39）。
const MAX_ATTRS_PER_NODE: usize = 16;
/// 1 ノードの入力・出力名数の上限（C3 の `Concat` は幅数 ≤ 8 個。REQ-39）。
const MAX_NODE_IO: usize = 16;
/// グラフの入出力・opset 記述の件数の上限（REQ-39）。
const MAX_GRAPH_IO: usize = 8;
/// 名前・文字列属性のバイト長の上限（REQ-39）。
const MAX_NAME_BYTES: usize = 128;
/// テンソル・入出力形状の次元数の上限（C3 の畳み込み重みが 3 次元。REQ-39）。
const MAX_DIMS: usize = 4;

/// AttributeProto.AttributeType の値のうち受理するもの。
const ATTR_FLOAT: i64 = 1;
const ATTR_INT: i64 = 2;
const ATTR_STRING: i64 = 3;
const ATTR_FLOATS: i64 = 6;
const ATTR_INTS: i64 = 7;

/// TensorProto.DataType の値のうち受理するもの。
pub(super) const DT_FLOAT: i64 = 1;
pub(super) const DT_INT64: i64 = 7;

#[derive(Debug)]
pub(super) struct ModelProto {
    pub ir_version: i64,
    pub opsets: Vec<OpsetId>,
    pub graph: GraphProto,
}

#[derive(Debug)]
pub(super) struct OpsetId {
    pub domain: String,
    pub version: i64,
}

#[derive(Debug, Default)]
pub(super) struct GraphProto {
    pub nodes: Vec<NodeProto>,
    pub initializers: Vec<TensorProto>,
    pub inputs: Vec<ValueInfo>,
    pub outputs: Vec<ValueInfo>,
}

#[derive(Debug, PartialEq)]
pub(super) enum Dim {
    Value(i64),
    Param(String),
}

#[derive(Debug, Default)]
pub(super) struct ValueInfo {
    pub name: String,
    pub elem_type: i64,
    pub dims: Vec<Dim>,
}

#[derive(Debug, Default)]
pub(super) struct NodeProto {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub op_type: String,
    pub domain: String,
    pub attrs: Vec<AttrProto>,
}

#[derive(Debug, Default)]
pub(super) struct AttrProto {
    pub name: String,
    pub ty: i64,
    pub f: Option<f32>,
    pub i: Option<i64>,
    pub s: Option<Vec<u8>>,
    pub floats: Vec<f32>,
    pub ints: Vec<i64>,
}

#[derive(Debug)]
pub(super) enum TensorData {
    F32(Vec<f32>),
    I64(Vec<i64>),
}

#[derive(Debug)]
pub(super) struct TensorProto {
    pub name: String,
    pub dims: Vec<i64>,
    pub data: TensorData,
}

fn malformed<T>(_: T) -> OnnxLoadError {
    OnnxLoadError::MalformedProtobuf
}

fn string_of(bytes: &[u8]) -> Result<String, OnnxLoadError> {
    if bytes.len() > MAX_NAME_BYTES {
        return Err(OnnxLoadError::LimitExceeded);
    }
    String::from_utf8(bytes.to_vec()).map_err(malformed)
}

fn push_limited<T>(v: &mut Vec<T>, item: T, limit: usize) -> Result<(), OnnxLoadError> {
    if v.len() >= limit {
        return Err(OnnxLoadError::LimitExceeded);
    }
    v.push(item);
    Ok(())
}

/// repeated int64（packed・非 packed の両方）を 1 フィールド分読み足す。
fn read_repeated_i64(
    r: &mut Reader<'_>,
    wt: WireType,
    out: &mut Vec<i64>,
    limit: usize,
) -> Result<(), OnnxLoadError> {
    match wt {
        WireType::Varint => {
            let v = r.read_varint().map_err(malformed)?;
            push_limited(out, v as i64, limit)
        }
        WireType::Len => {
            let mut inner = Reader::new(r.read_len().map_err(malformed)?);
            while !inner.is_empty() {
                let v = inner.read_varint().map_err(malformed)?;
                push_limited(out, v as i64, limit)?;
            }
            Ok(())
        }
        _ => Err(OnnxLoadError::MalformedProtobuf),
    }
}

/// repeated float（packed・非 packed の両方）を 1 フィールド分読み足す。
fn read_repeated_f32(
    r: &mut Reader<'_>,
    wt: WireType,
    out: &mut Vec<f32>,
    limit: usize,
) -> Result<(), OnnxLoadError> {
    match wt {
        WireType::Fixed32 => {
            let v = r.read_f32().map_err(malformed)?;
            push_limited(out, v, limit)
        }
        WireType::Len => {
            let bytes = r.read_len().map_err(malformed)?;
            if bytes.len() % 4 != 0 {
                return Err(OnnxLoadError::MalformedProtobuf);
            }
            // 長さが 4 の倍数であることは上で確認済み（余りは常に空）
            for arr in bytes.as_chunks::<4>().0 {
                push_limited(out, f32::from_le_bytes(*arr), limit)?;
            }
            Ok(())
        }
        _ => Err(OnnxLoadError::MalformedProtobuf),
    }
}

fn expect_len<'a>(r: &mut Reader<'a>, wt: WireType) -> Result<&'a [u8], OnnxLoadError> {
    if wt != WireType::Len {
        return Err(OnnxLoadError::MalformedProtobuf);
    }
    r.read_len().map_err(malformed)
}

fn expect_varint(r: &mut Reader<'_>, wt: WireType) -> Result<u64, OnnxLoadError> {
    if wt != WireType::Varint {
        return Err(OnnxLoadError::MalformedProtobuf);
    }
    r.read_varint().map_err(malformed)
}

/// ModelProto 全体を復号する。`limit_elems` は 1 属性・1 テンソルの要素数の上限（ファイルサイズ）。
pub(super) fn decode_model(bytes: &[u8]) -> Result<ModelProto, OnnxLoadError> {
    let elem_limit = bytes.len();
    let mut r = Reader::new(bytes);
    let mut ir_version = 0i64;
    let mut opsets = Vec::new();
    let mut graph: Option<GraphProto> = None;
    while let Some((num, wt)) = r.next_field().map_err(malformed)? {
        match num {
            1 => ir_version = expect_varint(&mut r, wt)? as i64,
            7 => {
                let body = expect_len(&mut r, wt)?;
                if graph.is_some() {
                    return Err(OnnxLoadError::MalformedProtobuf);
                }
                graph = Some(decode_graph(body, elem_limit)?);
            }
            8 => {
                let body = expect_len(&mut r, wt)?;
                push_limited(&mut opsets, decode_opset(body)?, MAX_GRAPH_IO)?;
            }
            _ => r.skip(wt).map_err(malformed)?,
        }
    }
    Ok(ModelProto {
        ir_version,
        opsets,
        graph: graph.ok_or(OnnxLoadError::MalformedProtobuf)?,
    })
}

fn decode_opset(bytes: &[u8]) -> Result<OpsetId, OnnxLoadError> {
    let mut r = Reader::new(bytes);
    let mut domain = String::new();
    let mut version = 0i64;
    while let Some((num, wt)) = r.next_field().map_err(malformed)? {
        match num {
            1 => domain = string_of(expect_len(&mut r, wt)?)?,
            2 => version = expect_varint(&mut r, wt)? as i64,
            _ => r.skip(wt).map_err(malformed)?,
        }
    }
    Ok(OpsetId { domain, version })
}

fn decode_graph(bytes: &[u8], elem_limit: usize) -> Result<GraphProto, OnnxLoadError> {
    let mut r = Reader::new(bytes);
    let mut g = GraphProto::default();
    while let Some((num, wt)) = r.next_field().map_err(malformed)? {
        match num {
            1 => {
                let node = decode_node(expect_len(&mut r, wt)?, elem_limit)?;
                push_limited(&mut g.nodes, node, MAX_NODES)?;
            }
            5 => {
                let t = decode_tensor(expect_len(&mut r, wt)?)?;
                push_limited(&mut g.initializers, t, MAX_INITIALIZERS)?;
            }
            11 => {
                let v = decode_value_info(expect_len(&mut r, wt)?)?;
                push_limited(&mut g.inputs, v, MAX_GRAPH_IO)?;
            }
            12 => {
                let v = decode_value_info(expect_len(&mut r, wt)?)?;
                push_limited(&mut g.outputs, v, MAX_GRAPH_IO)?;
            }
            // sparse_initializer・quantization_annotation は使わない書き出し器の出力には無い
            15 | 14 => return Err(OnnxLoadError::UnsupportedGraph),
            _ => r.skip(wt).map_err(malformed)?,
        }
    }
    Ok(g)
}

fn decode_node(bytes: &[u8], elem_limit: usize) -> Result<NodeProto, OnnxLoadError> {
    let mut r = Reader::new(bytes);
    let mut n = NodeProto::default();
    while let Some((num, wt)) = r.next_field().map_err(malformed)? {
        match num {
            1 => {
                let s = string_of(expect_len(&mut r, wt)?)?;
                push_limited(&mut n.inputs, s, MAX_NODE_IO)?;
            }
            2 => {
                let s = string_of(expect_len(&mut r, wt)?)?;
                push_limited(&mut n.outputs, s, MAX_NODE_IO)?;
            }
            4 => n.op_type = string_of(expect_len(&mut r, wt)?)?,
            5 => {
                let a = decode_attr(expect_len(&mut r, wt)?, elem_limit)?;
                push_limited(&mut n.attrs, a, MAX_ATTRS_PER_NODE)?;
            }
            7 => n.domain = string_of(expect_len(&mut r, wt)?)?,
            // name(3)・doc_string(6) は照合に使わない
            _ => r.skip(wt).map_err(malformed)?,
        }
    }
    Ok(n)
}

fn decode_attr(bytes: &[u8], elem_limit: usize) -> Result<AttrProto, OnnxLoadError> {
    let mut r = Reader::new(bytes);
    let mut a = AttrProto::default();
    while let Some((num, wt)) = r.next_field().map_err(malformed)? {
        match num {
            1 => a.name = string_of(expect_len(&mut r, wt)?)?,
            2 => {
                if wt != WireType::Fixed32 {
                    return Err(OnnxLoadError::MalformedProtobuf);
                }
                a.f = Some(r.read_f32().map_err(malformed)?);
            }
            3 => a.i = Some(expect_varint(&mut r, wt)? as i64),
            4 => a.s = Some(string_of(expect_len(&mut r, wt)?)?.into_bytes()),
            7 => read_repeated_f32(&mut r, wt, &mut a.floats, elem_limit)?,
            8 => read_repeated_i64(&mut r, wt, &mut a.ints, elem_limit)?,
            20 => a.ty = expect_varint(&mut r, wt)? as i64,
            // t・g・strings・tensors・graphs・ref_attr_name・sparse_tensor(s)・tp・type_protos:
            // 部分グラフ・テンソル・型情報を値に持つ属性は許可しない（再帰・外部参照を作らない）
            5 | 6 | 9 | 10 | 11 | 14 | 15 | 21 | 22 | 23 => {
                return Err(OnnxLoadError::UnsupportedGraph);
            }
            _ => r.skip(wt).map_err(malformed)?,
        }
    }
    if !matches!(
        a.ty,
        ATTR_FLOAT | ATTR_INT | ATTR_STRING | ATTR_FLOATS | ATTR_INTS
    ) {
        return Err(OnnxLoadError::UnsupportedGraph);
    }
    Ok(a)
}

fn decode_tensor(bytes: &[u8]) -> Result<TensorProto, OnnxLoadError> {
    let mut r = Reader::new(bytes);
    let mut name = String::new();
    let mut dims: Vec<i64> = Vec::new();
    let mut data_type = 0i64;
    let mut raw: Option<&[u8]> = None;
    while let Some((num, wt)) = r.next_field().map_err(malformed)? {
        match num {
            1 => read_repeated_i64(&mut r, wt, &mut dims, MAX_DIMS)?,
            2 => data_type = expect_varint(&mut r, wt)? as i64,
            8 => name = string_of(expect_len(&mut r, wt)?)?,
            9 => raw = Some(expect_len(&mut r, wt)?),
            // segment・float_data・int32_data・string_data・int64_data・double_data・uint64_data・
            // external_data・data_location: raw_data 以外の値の持ち方と外部参照は受理しない
            3..=7 | 10 | 11 | 13 | 14 => return Err(OnnxLoadError::UnsupportedTensor),
            _ => r.skip(wt).map_err(malformed)?,
        }
    }
    let elem_size: usize = match data_type {
        DT_FLOAT => 4,
        DT_INT64 => 8,
        _ => return Err(OnnxLoadError::UnsupportedTensor),
    };
    let mut count: usize = 1;
    for d in &dims {
        let d = usize::try_from(*d).map_err(|_| OnnxLoadError::UnsupportedTensor)?;
        count = count
            .checked_mul(d)
            .ok_or(OnnxLoadError::UnsupportedTensor)?;
    }
    let expected = count
        .checked_mul(elem_size)
        .ok_or(OnnxLoadError::UnsupportedTensor)?;
    let raw = raw.unwrap_or(&[]);
    if raw.len() != expected {
        return Err(OnnxLoadError::UnsupportedTensor);
    }
    let data = if data_type == DT_FLOAT {
        let mut v = Vec::with_capacity(count);
        // raw.len() == count * 4 は上で確認済み
        for arr in raw.as_chunks::<4>().0 {
            v.push(f32::from_le_bytes(*arr));
        }
        TensorData::F32(v)
    } else {
        let mut v = Vec::with_capacity(count);
        // raw.len() == count * 8 は上で確認済み
        for arr in raw.as_chunks::<8>().0 {
            v.push(i64::from_le_bytes(*arr));
        }
        TensorData::I64(v)
    };
    Ok(TensorProto { name, dims, data })
}

fn decode_value_info(bytes: &[u8]) -> Result<ValueInfo, OnnxLoadError> {
    let mut r = Reader::new(bytes);
    let mut v = ValueInfo::default();
    while let Some((num, wt)) = r.next_field().map_err(malformed)? {
        match num {
            1 => v.name = string_of(expect_len(&mut r, wt)?)?,
            2 => decode_type(expect_len(&mut r, wt)?, &mut v)?,
            _ => r.skip(wt).map_err(malformed)?,
        }
    }
    Ok(v)
}

/// TypeProto（`tensor_type` のみ受理）から要素型と形状を読む。
fn decode_type(bytes: &[u8], v: &mut ValueInfo) -> Result<(), OnnxLoadError> {
    let mut r = Reader::new(bytes);
    while let Some((num, wt)) = r.next_field().map_err(malformed)? {
        match num {
            1 => {
                let mut t = Reader::new(expect_len(&mut r, wt)?);
                while let Some((n2, w2)) = t.next_field().map_err(malformed)? {
                    match n2 {
                        1 => v.elem_type = expect_varint(&mut t, w2)? as i64,
                        2 => decode_shape(expect_len(&mut t, w2)?, &mut v.dims)?,
                        _ => t.skip(w2).map_err(malformed)?,
                    }
                }
            }
            // sequence・map・optional・sparse 等の型は受理しない
            2..=9 => return Err(OnnxLoadError::UnsupportedGraph),
            _ => r.skip(wt).map_err(malformed)?,
        }
    }
    Ok(())
}

fn decode_shape(bytes: &[u8], dims: &mut Vec<Dim>) -> Result<(), OnnxLoadError> {
    let mut r = Reader::new(bytes);
    while let Some((num, wt)) = r.next_field().map_err(malformed)? {
        if num != 1 {
            r.skip(wt).map_err(malformed)?;
            continue;
        }
        let mut d = Reader::new(expect_len(&mut r, wt)?);
        let mut dim: Option<Dim> = None;
        while let Some((n2, w2)) = d.next_field().map_err(malformed)? {
            match n2 {
                1 => dim = Some(Dim::Value(expect_varint(&mut d, w2)? as i64)),
                2 => dim = Some(Dim::Param(string_of(expect_len(&mut d, w2)?)?)),
                _ => d.skip(w2).map_err(malformed)?,
            }
        }
        push_limited(dims, dim.ok_or(OnnxLoadError::MalformedProtobuf)?, MAX_DIMS)?;
    }
    Ok(())
}

impl AttrProto {
    /// INT 属性の値（型と値の存在を確認する）。
    pub(super) fn int(&self) -> Option<i64> {
        (self.ty == ATTR_INT).then_some(self.i).flatten()
    }

    /// FLOAT 属性の値。
    pub(super) fn float(&self) -> Option<f32> {
        (self.ty == ATTR_FLOAT).then_some(self.f).flatten()
    }

    /// STRING 属性の値。
    pub(super) fn string(&self) -> Option<&[u8]> {
        (self.ty == ATTR_STRING)
            .then_some(self.s.as_deref())
            .flatten()
    }

    /// INTS 属性の値。
    pub(super) fn ints(&self) -> Option<&[i64]> {
        (self.ty == ATTR_INTS).then_some(self.ints.as_slice())
    }
}

impl NodeProto {
    /// 名前が一致する属性（同名の重複は照合側で拒否するため最初の 1 件）。
    pub(super) fn attr(&self, name: &str) -> Option<&AttrProto> {
        self.attrs.iter().find(|a| a.name == name)
    }
}
