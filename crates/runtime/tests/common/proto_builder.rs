//! 改変モデルを組み立てるための最小 protobuf ツリー編集器（テスト専用。REQ-39・#113）。
//!
//! コミット済みの正常な ONNX を wire 形式のフィールド列へ分解し、任意のフィールドを差し替え・
//! 追加・並べ替えて再符号化する。推論ランタイム側の復号器（`src/onnx/wire.rs`）とは独立に書き、
//! 復号器の誤りを同じ誤りで打ち消さないようにする。テスト専用のため `unwrap`・添字を使ってよい。

/// wire 上の値（group は扱わない）。
#[derive(Clone, Debug, PartialEq)]
pub enum Val {
    Varint(u64),
    Len(Vec<u8>),
    Fixed32([u8; 4]),
    Fixed64([u8; 8]),
}

/// フィールド番号と値の並び（出現順を保持する）。
pub type Msg = Vec<(u32, Val)>;

fn read_varint(b: &[u8], pos: &mut usize) -> u64 {
    let mut v = 0u64;
    let mut shift = 0;
    loop {
        let byte = b[*pos];
        *pos += 1;
        v |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return v;
        }
        shift += 7;
    }
}

fn write_varint(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// メッセージをフィールド列へ分解する。
pub fn parse(b: &[u8]) -> Msg {
    let mut pos = 0;
    let mut out = Vec::new();
    while pos < b.len() {
        let tag = read_varint(b, &mut pos);
        let num = u32::try_from(tag >> 3).unwrap();
        let val = match tag & 7 {
            0 => Val::Varint(read_varint(b, &mut pos)),
            1 => {
                let v = b[pos..pos + 8].try_into().unwrap();
                pos += 8;
                Val::Fixed64(v)
            }
            2 => {
                let len = usize::try_from(read_varint(b, &mut pos)).unwrap();
                let v = b[pos..pos + len].to_vec();
                pos += len;
                Val::Len(v)
            }
            5 => {
                let v = b[pos..pos + 4].try_into().unwrap();
                pos += 4;
                Val::Fixed32(v)
            }
            other => panic!("unsupported wire type {other}"),
        };
        out.push((num, val));
    }
    out
}

/// フィールド列をバイト列へ符号化する。
pub fn encode(m: &Msg) -> Vec<u8> {
    let mut out = Vec::new();
    for (num, val) in m {
        let wt = match val {
            Val::Varint(_) => 0,
            Val::Fixed64(_) => 1,
            Val::Len(_) => 2,
            Val::Fixed32(_) => 5,
        };
        write_varint(&mut out, (u64::from(*num) << 3) | wt);
        match val {
            Val::Varint(v) => write_varint(&mut out, *v),
            Val::Fixed64(v) => out.extend_from_slice(v),
            Val::Len(v) => {
                write_varint(&mut out, v.len() as u64);
                out.extend_from_slice(v);
            }
            Val::Fixed32(v) => out.extend_from_slice(v),
        }
    }
    out
}

/// `field` 番の n 番目の LEN フィールドの位置（`Msg` 内の添字）。
fn nth_index(m: &Msg, field: u32, nth: usize) -> usize {
    m.iter()
        .enumerate()
        .filter(|(_, (n, v))| *n == field && matches!(v, Val::Len(_)))
        .nth(nth)
        .map(|(i, _)| i)
        .unwrap()
}

/// `field` 番の n 番目の LEN フィールドを子メッセージとして取り出す。
pub fn sub(m: &Msg, field: u32, nth: usize) -> Msg {
    match &m[nth_index(m, field, nth)].1 {
        Val::Len(b) => parse(b),
        _ => unreachable!(),
    }
}

/// 子メッセージを編集して書き戻す。
pub fn edit_sub(m: &mut Msg, field: u32, nth: usize, f: impl FnOnce(&mut Msg)) {
    let i = nth_index(m, field, nth);
    let mut inner = sub(m, field, nth);
    f(&mut inner);
    m[i].1 = Val::Len(encode(&inner));
}

/// モデル全体（ModelProto）を編集して再符号化する。
pub fn edit_model(bytes: &[u8], f: impl FnOnce(&mut Msg)) -> Vec<u8> {
    let mut m = parse(bytes);
    f(&mut m);
    encode(&m)
}

/// グラフ（ModelProto.graph = 7）を編集して再符号化する。
pub fn edit_graph(bytes: &[u8], f: impl FnOnce(&mut Msg)) -> Vec<u8> {
    edit_model(bytes, |m| edit_sub(m, 7, 0, f))
}

/// 文字列フィールド（field 番号 `field`）の値。
pub fn string_field(m: &Msg, field: u32) -> Option<String> {
    m.iter().find_map(|(n, v)| match v {
        Val::Len(b) if *n == field => Some(String::from_utf8(b.clone()).unwrap()),
        _ => None,
    })
}

/// 名前が一致する initializer（GraphProto.initializer = 5）を編集する。
pub fn edit_initializer(bytes: &[u8], name: &str, f: impl FnOnce(&mut Msg)) -> Vec<u8> {
    edit_graph(bytes, |g| {
        let count = g.iter().filter(|(n, _)| *n == 5).count();
        let nth = (0..count)
            .find(|&i| string_field(&sub(g, 5, i), 8).as_deref() == Some(name))
            .expect("initializer not found");
        edit_sub(g, 5, nth, f);
    })
}

/// `node_index` 番目のノード（GraphProto.node = 1）の、名前が一致する属性（NodeProto.attribute = 5）を
/// 編集する。
pub fn edit_attr(bytes: &[u8], node_index: usize, attr: &str, f: impl FnOnce(&mut Msg)) -> Vec<u8> {
    edit_graph(bytes, |g| {
        edit_sub(g, 1, node_index, |node| {
            let count = node.iter().filter(|(n, _)| *n == 5).count();
            let nth = (0..count)
                .find(|&i| string_field(&sub(node, 5, i), 1).as_deref() == Some(attr))
                .expect("attribute not found");
            edit_sub(node, 5, nth, f);
        })
    })
}

/// ノード（GraphProto.node = 1）の op_type（NodeProto.op_type = 4）を書き換える。
pub fn set_op_type(bytes: &[u8], node_index: usize, op: &str) -> Vec<u8> {
    edit_graph(bytes, |g| {
        edit_sub(g, 1, node_index, |node| {
            let i = node.iter().position(|(n, _)| *n == 4).unwrap();
            node[i].1 = Val::Len(op.as_bytes().to_vec());
        })
    })
}
