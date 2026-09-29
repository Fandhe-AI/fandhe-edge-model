//! protobuf wire 形式の有界デコーダ（REQ-32・REQ-39・TASK-32.1-2・#113）。
//!
//! # 役割
//!
//! [`super::proto`] が ONNX の必要フィールドだけを取り出すための最下層。外部 crate（`prost` 等）は
//! 未承認のため使わず、`&[u8]` 上のカーソルで varint・長さ付き（LEN）・固定長の 3 種だけを復号する。
//! 入力（モデルファイル）は外部入力として扱い、添字アクセス・`unwrap` を使わず、長さが残りバイト数を
//! 超える・varint が 64 bit を超える・group（wire type 3/4）を含む場合はすべて [`WireError`] で拒否する。

/// wire 形式として不正（本文を保持しない）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WireError;

/// wire type（protobuf の 3 bit タグ）。group（3・4）は受理しない。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WireType {
    Varint,
    Fixed64,
    Len,
    Fixed32,
}

/// バイト列上のカーソル。
pub(crate) struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    /// 未読のバイトが残っていない。
    pub(crate) fn is_empty(&self) -> bool {
        self.pos >= self.buf.len()
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], WireError> {
        let end = self.pos.checked_add(n).ok_or(WireError)?;
        let s = self.buf.get(self.pos..end).ok_or(WireError)?;
        self.pos = end;
        Ok(s)
    }

    /// varint（最大 10 バイト。64 bit を超える桁は拒否）。
    pub(crate) fn read_varint(&mut self) -> Result<u64, WireError> {
        let mut value: u64 = 0;
        for shift in (0..70).step_by(7) {
            let byte = *self.buf.get(self.pos).ok_or(WireError)?;
            self.pos += 1;
            let low = u64::from(byte & 0x7f);
            if shift == 63 && low > 1 {
                return Err(WireError);
            }
            value |= low << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(WireError)
    }

    /// 次のフィールドのタグ（フィールド番号・wire type）。終端なら `None`。
    pub(crate) fn next_field(&mut self) -> Result<Option<(u32, WireType)>, WireError> {
        if self.is_empty() {
            return Ok(None);
        }
        let tag = self.read_varint()?;
        let number = u32::try_from(tag >> 3).map_err(|_| WireError)?;
        if number == 0 {
            return Err(WireError);
        }
        let wt = match tag & 7 {
            0 => WireType::Varint,
            1 => WireType::Fixed64,
            2 => WireType::Len,
            5 => WireType::Fixed32,
            _ => return Err(WireError),
        };
        Ok(Some((number, wt)))
    }

    /// LEN フィールドの中身（長さが残りを超えたら拒否）。
    pub(crate) fn read_len(&mut self) -> Result<&'a [u8], WireError> {
        let len = usize::try_from(self.read_varint()?).map_err(|_| WireError)?;
        self.take(len)
    }

    /// fixed32 を f32 として読む。
    pub(crate) fn read_f32(&mut self) -> Result<f32, WireError> {
        let b: [u8; 4] = self.take(4)?.try_into().map_err(|_| WireError)?;
        Ok(f32::from_le_bytes(b))
    }

    /// 期待しない・不要なフィールドを wire type に従って読み飛ばす。
    pub(crate) fn skip(&mut self, wt: WireType) -> Result<(), WireError> {
        match wt {
            WireType::Varint => self.read_varint().map(|_| ()),
            WireType::Fixed64 => self.take(8).map(|_| ()),
            WireType::Len => self.read_len().map(|_| ()),
            WireType::Fixed32 => self.take(4).map(|_| ()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-39: varint の境界値（1 バイト最大・2 バイト最小・u64::MAX）。
    #[test]
    fn req39_varint_boundaries() {
        assert_eq!(Reader::new(&[0x7f]).read_varint(), Ok(127));
        assert_eq!(Reader::new(&[0x80, 0x01]).read_varint(), Ok(128));
        let max = [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01];
        assert_eq!(Reader::new(&max).read_varint(), Ok(u64::MAX));
    }

    /// REQ-39: 64 bit 超・11 バイト以上・途中で切れた varint は拒否する。
    #[test]
    fn req39_varint_overflow_and_truncation_rejected() {
        let over = [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x02];
        assert_eq!(Reader::new(&over).read_varint(), Err(WireError));
        let long = [0x80; 11];
        assert_eq!(Reader::new(&long).read_varint(), Err(WireError));
        assert_eq!(Reader::new(&[0x80]).read_varint(), Err(WireError));
        assert_eq!(Reader::new(&[]).read_varint(), Err(WireError));
    }

    /// REQ-39: LEN の長さが残りを超えたら拒否し、group と番号 0 は拒否する。
    #[test]
    fn req39_len_and_tag_validation() {
        assert_eq!(Reader::new(&[0x05, 1, 2]).read_len(), Err(WireError));
        assert_eq!(Reader::new(&[0x02, 1, 2]).read_len(), Ok(&[1u8, 2][..]));
        // field 1・wire type 3（start group）
        assert_eq!(Reader::new(&[0x0b]).next_field(), Err(WireError));
        // field 0
        assert_eq!(Reader::new(&[0x00]).next_field(), Err(WireError));
        assert_eq!(
            Reader::new(&[0x0a]).next_field(),
            Ok(Some((1, WireType::Len)))
        );
    }

    /// 未知フィールドは wire type に従って読み飛ばせる。
    #[test]
    fn skip_unknown_fields() {
        let mut r = Reader::new(&[0x2d, 1, 2, 3, 4, 0x08, 0x01]);
        assert_eq!(r.next_field(), Ok(Some((5, WireType::Fixed32))));
        assert_eq!(r.skip(WireType::Fixed32), Ok(()));
        assert_eq!(r.next_field(), Ok(Some((1, WireType::Varint))));
        assert_eq!(r.skip(WireType::Varint), Ok(()));
        assert_eq!(r.next_field(), Ok(None));
    }
}
