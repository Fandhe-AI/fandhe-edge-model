//! ONNX の外部データ（分割した重み）の解決（REQ-32・REQ-39・REQ-41・TASK-41.7-8・#581）。
//!
//! # 役割と呼び出し文脈
//!
//! 追加学習の種類（Qwen2.5-1.5B 級。bf16 で約 3.1 GB〔推定〕）は重みを 1 GiB 以下の shard に
//! 分け、ONNX の `external_data` で参照する。本モジュールは [`super::proto`] が保留した参照を、
//! 呼び出し側が渡す閉じ込め済みの `File` から読んで実体に置き換える。**未接続**: 呼び出し元の
//! Qwen 実行器は #420・#424・#421 で追加する（kind 名は #424 で決まるため、[`super::ModelKind`]
//! にも未登録。C1・C3 は [`super::STRICT_MODEL_LIMITS`] で外部データを拒否したまま）。
//!
//! # 契約
//!
//! - 経路の閉じ込め: runtime はパスを開かない。呼び出し側が `ConfinedPackage::open_member`
//!   （ガード層。TOCTOU を避ける fd 起点の open）で開いた `File` だけを受け取る。字句の検査として
//!   location は単一のファイル名（`[A-Za-z0-9._-]`、先頭は `.` 以外）に限る
//! - 完全性: shard ごとに呼び出し側が渡す期待 sha256 と、読みながら計算した値を照合する。
//!   不一致なら復号済みの値をすべて捨てる（fail-closed）。ONNX 内の `checksum` は信用しない
//! - 資源: 件数・shard ごとの fstat サイズ・合計を、1 バイトも読む前に上限と照合する。
//!   参照範囲の重なりを拒否し（同じ領域を何度も参照させて確保量を増幅させない）、確保は
//!   `try_reserve_exact`。shard は固定長バッファで 1 回だけ流し読みし、生バイトの全量を保持しない
//! - 保持形式: bf16 は `u16` のまま保持する（f32 へ全量展開すると約 6.2 GB〔推定〕で RSS 安全上限
//!   8 GiB に近い）。f32 への展開は Gemm・Gather のカーネル内（#416・#417）。RSS の実測は #421・#435

use super::proto::{DT_BFLOAT16, DT_FLOAT, DT_INT64, ExternalRef, GraphProto, TensorData};
use super::{ExternalDataLimits, OnnxLoadError};
use fandhe_edge_core::fs::FsError;
use fandhe_edge_core::hash::{Sha256Digest, Sha256Stream};
use std::collections::BTreeSet;
use std::fs::File;
use std::io::{Read as _, Seek as _, SeekFrom};
use std::path::PathBuf;

/// 流し読みのバッファ長（8 の倍数）。
const BUF_BYTES: usize = 64 * 1024;
/// location の長さの上限。
const MAX_LOCATION_BYTES: usize = 128;

/// 呼び出し側が閉じ込めて開いた shard 1 つ。
#[derive(Debug)]
#[cfg_attr(not(test), allow(dead_code))]
pub(super) struct ExternalShard {
    pub location: String,
    pub file: File,
    /// 呼び出し側（`artifact.json` 由来）が渡す期待 sha256。
    pub sha256: Sha256Digest,
}

/// location が単一のファイル名であること（`/`・`\`・`..`・先頭 `.`・空・長すぎるものは拒否）。
fn location_ok(l: &str) -> bool {
    !l.is_empty()
        && l.len() <= MAX_LOCATION_BYTES
        && !l.starts_with('.')
        && !l.contains("..")
        && l.bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

/// 復号先。要素は `try_reserve_exact` 済みの容量内にだけ積む。
enum Out {
    F32(Vec<f32>),
    I64(Vec<i64>),
    Bf16(Vec<u16>),
}

/// 1 テンソル分の逐次デコーダ。チャンク境界で要素が分かれても `carry` で繋ぐ。
struct Sink {
    out: Out,
    elem: usize,
    carry: [u8; 8],
    clen: usize,
}

fn alloc<T>(n: usize) -> Result<Vec<T>, OnnxLoadError> {
    let mut v = Vec::new();
    v.try_reserve_exact(n)
        .map_err(|_| OnnxLoadError::LimitExceeded)?;
    Ok(v)
}

impl Sink {
    fn new(r: &ExternalRef) -> Result<Self, OnnxLoadError> {
        let (out, elem) = match r.data_type {
            DT_FLOAT => (Out::F32(alloc(r.count)?), 4),
            DT_INT64 => (Out::I64(alloc(r.count)?), 8),
            DT_BFLOAT16 => (Out::Bf16(alloc(r.count)?), 2),
            _ => return Err(OnnxLoadError::UnsupportedTensor),
        };
        Ok(Self {
            out,
            elem,
            carry: [0; 8],
            clen: 0,
        })
    }

    /// `b` は常に要素サイズ丁度。`try_into` は長さ一致で失敗しない（失敗時は積まず、
    /// `finish` の要素数検査で拒否される）。
    fn emit(&mut self, b: &[u8]) {
        match &mut self.out {
            Out::F32(v) => v.extend(b.try_into().map(f32::from_le_bytes)),
            Out::I64(v) => v.extend(b.try_into().map(i64::from_le_bytes)),
            Out::Bf16(v) => v.extend(b.try_into().map(u16::from_le_bytes)),
        }
    }

    fn push(&mut self, mut bytes: &[u8]) {
        if self.clen > 0 {
            let take = (self.elem - self.clen).min(bytes.len());
            let (head, rest) = bytes.split_at(take);
            if let Some(dst) = self.carry.get_mut(self.clen..self.clen + take) {
                dst.copy_from_slice(head);
            }
            self.clen += take;
            bytes = rest;
            if self.clen < self.elem {
                return;
            }
            let c = self.carry;
            if let Some(whole) = c.get(..self.elem) {
                self.emit(whole);
            }
            self.clen = 0;
        }
        let mut it = bytes.chunks_exact(self.elem);
        for c in it.by_ref() {
            self.emit(c);
        }
        let rem = it.remainder();
        if let Some(dst) = self.carry.get_mut(..rem.len()) {
            dst.copy_from_slice(rem);
        }
        self.clen = rem.len();
    }

    fn finish(self, want: usize) -> Result<TensorData, OnnxLoadError> {
        let (len, data) = match self.out {
            Out::F32(v) => (v.len(), TensorData::F32(v)),
            Out::I64(v) => (v.len(), TensorData::I64(v)),
            Out::Bf16(v) => (v.len(), TensorData::Bf16(v)),
        };
        if self.clen != 0 || len != want {
            return Err(OnnxLoadError::UnsupportedTensor);
        }
        Ok(data)
    }
}

fn read_err(location: &str, source: std::io::Error) -> OnnxLoadError {
    OnnxLoadError::File(FsError::Read {
        path: PathBuf::from(location),
        source,
    })
}

/// グラフの `External` を、閉じ込め済みの shard から読んだ値へ置き換える。成功時、`External` は
/// 1 つも残らない。失敗時は `graph` の中身を信用してはならない（呼び出し側は捨てる）。
#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn resolve_external(
    graph: &mut GraphProto,
    shards: Vec<ExternalShard>,
    limits: &ExternalDataLimits,
) -> Result<(), OnnxLoadError> {
    // 1. 件数
    if shards.len() > limits.max_shards {
        return Err(OnnxLoadError::LimitExceeded);
    }
    // 2・3. location の字句検証と、参照集合 = 渡された shard 集合
    let referenced: BTreeSet<&str> = graph
        .initializers
        .iter()
        .filter_map(|t| match &t.data {
            TensorData::External(e) => Some(e.location.as_str()),
            _ => None,
        })
        .collect();
    let given: BTreeSet<&str> = shards.iter().map(|s| s.location.as_str()).collect();
    if !referenced
        .iter()
        .chain(given.iter())
        .all(|l| location_ok(l))
        || given.len() != shards.len()
        || referenced != given
    {
        return Err(OnnxLoadError::UnsupportedTensor);
    }
    // 4. fstat（読む前に shard ごと・合計を確認）
    let mut sizes = Vec::with_capacity(shards.len());
    let mut total: u64 = 0;
    for s in &shards {
        let md = s.file.metadata().map_err(|e| read_err(&s.location, e))?;
        if !md.file_type().is_file() {
            return Err(OnnxLoadError::File(FsError::NotRegularFile {
                path: PathBuf::from(&s.location),
            }));
        }
        if md.len() > limits.max_shard_bytes {
            return Err(OnnxLoadError::LimitExceeded);
        }
        total = total
            .checked_add(md.len())
            .filter(|t| *t <= limits.max_total_bytes)
            .ok_or(OnnxLoadError::LimitExceeded)?;
        sizes.push(md.len());
    }
    // 5. 範囲: shard 内に収まり、互いに重ならない
    let mut refs: Vec<(usize, u64, u64, usize)> = Vec::new(); // (shard, offset, length, initializer)
    for (ti, t) in graph.initializers.iter().enumerate() {
        if let TensorData::External(e) = &t.data {
            let si = shards
                .iter()
                .position(|s| s.location == e.location)
                .ok_or(OnnxLoadError::UnsupportedTensor)?;
            let end = e
                .offset
                .checked_add(e.length)
                .ok_or(OnnxLoadError::UnsupportedTensor)?;
            if sizes.get(si).is_none_or(|sz| end > *sz) {
                return Err(OnnxLoadError::UnsupportedTensor);
            }
            refs.push((si, e.offset, e.length, ti));
        }
    }
    refs.sort_unstable();
    // 各範囲は上で checked_add 済みで、end は shard サイズ以下
    if refs
        .windows(2)
        .any(|w| matches!(w, [a, b] if a.0 == b.0 && a.1 + a.2 > b.1))
    {
        return Err(OnnxLoadError::UnsupportedTensor);
    }
    // 6. 確保（読む前）
    let mut sinks: Vec<Option<Sink>> = Vec::new();
    sinks.resize_with(graph.initializers.len(), || None);
    for (ti, t) in graph.initializers.iter().enumerate() {
        if let (TensorData::External(e), Some(slot)) = (&t.data, sinks.get_mut(ti)) {
            *slot = Some(Sink::new(e)?);
        }
    }
    // 7・8. shard ごとに 1 回だけ流し読みし、読みながらハッシュ・復号する
    for (si, shard) in shards.iter().enumerate() {
        let size = sizes.get(si).copied().unwrap_or(0);
        let mut file = &shard.file;
        file.seek(SeekFrom::Start(0))
            .map_err(|e| read_err(&shard.location, e))?;
        let mut mine = refs.iter().filter(|r| r.0 == si).peekable();
        let mut hasher = Sha256Stream::new();
        let mut buf = [0u8; BUF_BYTES];
        let mut pos: u64 = 0;
        loop {
            let n = file
                .read(&mut buf)
                .map_err(|e| read_err(&shard.location, e))?;
            if n == 0 {
                break;
            }
            let chunk = buf.get(..n).ok_or(OnnxLoadError::UnsupportedTensor)?;
            let next_pos = pos.saturating_add(n as u64);
            // fstat 後に伸びたファイルは拒否（時間・メモリの上限を守る）
            if next_pos > size {
                return Err(OnnxLoadError::File(FsError::TooLarge {
                    path: PathBuf::from(&shard.location),
                    size: next_pos,
                    limit: size,
                }));
            }
            hasher.update(chunk);
            // このチャンクと重なる参照へ配る（refs は範囲順で重ならない）
            while let Some(&&(_, off, len, ti)) = mine.peek() {
                let end = off + len;
                if off >= next_pos {
                    break;
                }
                let lo = off.max(pos) - pos;
                let hi = end.min(next_pos) - pos;
                if lo < hi {
                    // lo・hi は 64 KiB 以下（チャンク内の位置）
                    let part = chunk.get(lo as usize..hi as usize);
                    if let (Some(Some(sink)), Some(part)) = (sinks.get_mut(ti), part) {
                        sink.push(part);
                    }
                }
                if end <= next_pos {
                    mine.next();
                } else {
                    break;
                }
            }
            pos = next_pos;
        }
        // 縮んだ・差し替えられた場合も含め、完全性の違反として扱う
        if pos != size || hasher.finish() != shard.sha256 {
            return Err(OnnxLoadError::IntegrityMismatch);
        }
    }
    // 9. 置き換え（全 shard が通ってから）
    for (ti, slot) in sinks.into_iter().enumerate() {
        if let (Some(sink), Some(t)) = (slot, graph.initializers.get_mut(ti)) {
            t.data = sink.finish(match &t.data {
                TensorData::External(e) => e.count,
                _ => 0,
            })?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
