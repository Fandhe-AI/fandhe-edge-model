//! 1 piece の byte-level BPE（REQ-41・TASK-41.4-4・#412）。
//!
//! 呼び出し元は [`super::TokenizerSpec::encode`]。参照実装は
//! `trainer/src/fandhe_edge_trainer/qwen2_tokenizer.py` の `_bpe` で、優先度付きキュー
//! （順位・左位置・左右の版番号のタプル順）と双方向リンクにより、併合の順序まで同一にする
//! （順位が同じなら左を先に併合）。文字列ではなく id 同士で併合する。vocab は #410 で全単射と
//! 検証済みのため、文字列連結での照合と結果は変わらない。
//!
//! 計算量は piece のバイト数 n に対し O(n log n)。作業領域とヒープはどちらも n に比例し、
//! piece は呼び出し元で入力上限（`MAX_ENCODE_BYTES`）以下に収まる（REQ-39）。
//! 添字は `get` で扱い、不整合は [`TokenizeError::Internal`] で返す（panic させない）。

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::{MergeTable, TokenizeError};

type Heap = BinaryHeap<Reverse<(u32, usize, usize, u32, u32)>>;

/// 隣接 (i, j) が merges にあれば候補に積む。
fn push(
    heap: &mut Heap,
    merges: &MergeTable,
    parts: &[u32],
    ver: &[u32],
    i: usize,
    j: usize,
) -> Result<(), TokenizeError> {
    let l = *parts.get(i).ok_or(TokenizeError::Internal)?;
    let r = *parts.get(j).ok_or(TokenizeError::Internal)?;
    if let Some(&(rank, _)) = merges.get(&(l, r)) {
        let vi = *ver.get(i).ok_or(TokenizeError::Internal)?;
        let vj = *ver.get(j).ok_or(TokenizeError::Internal)?;
        heap.push(Reverse((rank, i, j, vi, vj)));
    }
    Ok(())
}

/// `piece` を UTF-8 バイト列として BPE し、token id 列を返す。
pub(super) fn bpe(
    piece: &str,
    byte_ids: &[u32; 256],
    merges: &MergeTable,
) -> Result<Vec<u32>, TokenizeError> {
    let mut parts: Vec<u32> = piece
        .bytes()
        .map(|b| byte_ids.get(usize::from(b)).copied())
        .collect::<Option<_>>()
        .ok_or(TokenizeError::Internal)?;
    let n = parts.len();
    // nxt[i] == n は終端。prev は None が先頭。
    let mut nxt: Vec<usize> = (1..=n).collect();
    let mut prev: Vec<Option<usize>> = (0..n).map(|i| i.checked_sub(1)).collect();
    let mut alive = vec![true; n];
    let mut ver = vec![0u32; n];
    let mut heap = Heap::new();

    for i in 0..n.saturating_sub(1) {
        push(&mut heap, merges, &parts, &ver, i, i + 1)?;
    }
    while let Some(Reverse((_, i, j, vi, vj))) = heap.pop() {
        let fresh = alive.get(i) == Some(&true)
            && nxt.get(i) == Some(&j)
            && ver.get(i) == Some(&vi)
            && ver.get(j) == Some(&vj);
        if !fresh {
            continue;
        }
        let l = *parts.get(i).ok_or(TokenizeError::Internal)?;
        let r = *parts.get(j).ok_or(TokenizeError::Internal)?;
        let &(_, merged) = merges.get(&(l, r)).ok_or(TokenizeError::Internal)?;
        let nj = *nxt.get(j).ok_or(TokenizeError::Internal)?;
        *parts.get_mut(i).ok_or(TokenizeError::Internal)? = merged;
        *ver.get_mut(i).ok_or(TokenizeError::Internal)? += 1;
        *alive.get_mut(j).ok_or(TokenizeError::Internal)? = false;
        *nxt.get_mut(i).ok_or(TokenizeError::Internal)? = nj;
        if nj < n {
            *prev.get_mut(nj).ok_or(TokenizeError::Internal)? = Some(i);
        }
        if let Some(p) = *prev.get(i).ok_or(TokenizeError::Internal)? {
            push(&mut heap, merges, &parts, &ver, p, i)?;
        }
        if nj < n {
            push(&mut heap, merges, &parts, &ver, i, nj)?;
        }
    }

    let mut ids = Vec::new();
    let mut i = 0;
    while i < n {
        ids.push(*parts.get(i).ok_or(TokenizeError::Internal)?);
        i = *nxt.get(i).ok_or(TokenizeError::Internal)?;
    }
    Ok(ids)
}
