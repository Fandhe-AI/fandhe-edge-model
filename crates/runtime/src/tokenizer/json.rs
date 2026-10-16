//! トークナイザー JSON 用の std のみの最小 JSON パーサー（REQ-41・REQ-39・TASK-41.4-2・#410）。
//!
//! 呼び出し元は [`super::TokenizerSpec::from_bytes`] だけ。`serde_json` は runtime では
//! dev-dependency にしか承認されていないため（#409 のオーナー判断: 依存を追加しない）、
//! RFC 8259 のうち `tokenizer.json` の読み込みに必要な範囲を自作する。
//!
//! 外部入力なので、入れ子の深さと各コンテナの要素数に上限を設け、要素を追加する前に判定する。
//! 重複キー・孤立サロゲート・先頭 0・`NaN` などは拒否する（後勝ちの黙認をしない）。
//! `unwrap`・`expect`・添字アクセスは本体では使わない。

use std::collections::HashMap;

/// 入れ子の深さ上限（実物の `tokenizer.json` は 5）。
pub(super) const MAX_JSON_DEPTH: usize = 16;
/// 配列・オブジェクト 1 つあたりの要素数上限（実物の最大は merges の約 15 万件）。
pub(super) const MAX_JSON_CONTAINER_LEN: usize = 262_144;

/// 構文上の値。小数・指数・i64 範囲外の数値は文法だけ検証して値を保持しない。
#[derive(Debug, PartialEq)]
pub(super) enum Value {
    Null,
    Bool(bool),
    Int(i64),
    NonIntNumber,
    Str(String),
    Array(Vec<Value>),
    Object(HashMap<String, Value>),
}

/// 構文解析の失敗。入力の内容は保持しない。
#[derive(Debug, PartialEq, Eq)]
pub(super) enum JsonError {
    Malformed,
    LimitExceeded,
}

/// 全体を 1 つの JSON 値として解析する（前後の空白のみ許可。BOM は拒否）。
pub(super) fn parse(bytes: &[u8]) -> Result<Value, JsonError> {
    let text = std::str::from_utf8(bytes).map_err(|_| JsonError::Malformed)?;
    let mut p = Parser {
        src: text.as_bytes(),
        text,
        pos: 0,
    };
    p.skip_ws();
    let v = p.value(0)?;
    p.skip_ws();
    if p.pos != p.src.len() {
        return Err(JsonError::Malformed);
    }
    Ok(v)
}

struct Parser<'a> {
    src: &'a [u8],
    text: &'a str,
    pos: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn eat(&mut self, b: u8) -> Result<(), JsonError> {
        if self.peek() == Some(b) {
            self.pos += 1;
            Ok(())
        } else {
            Err(JsonError::Malformed)
        }
    }

    fn literal(&mut self, word: &[u8], v: Value) -> Result<Value, JsonError> {
        let end = self.pos.saturating_add(word.len());
        if self.src.get(self.pos..end) == Some(word) {
            self.pos = end;
            Ok(v)
        } else {
            Err(JsonError::Malformed)
        }
    }

    fn value(&mut self, depth: usize) -> Result<Value, JsonError> {
        match self.peek().ok_or(JsonError::Malformed)? {
            b'n' => self.literal(b"null", Value::Null),
            b't' => self.literal(b"true", Value::Bool(true)),
            b'f' => self.literal(b"false", Value::Bool(false)),
            b'"' => Ok(Value::Str(self.string()?)),
            b'[' => self.array(depth),
            b'{' => self.object(depth),
            b'-' | b'0'..=b'9' => self.number(),
            _ => Err(JsonError::Malformed),
        }
    }

    fn array(&mut self, depth: usize) -> Result<Value, JsonError> {
        let depth = depth + 1;
        if depth > MAX_JSON_DEPTH {
            return Err(JsonError::LimitExceeded);
        }
        self.eat(b'[')?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Value::Array(items));
        }
        loop {
            if items.len() >= MAX_JSON_CONTAINER_LEN {
                return Err(JsonError::LimitExceeded);
            }
            self.skip_ws();
            items.push(self.value(depth)?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(Value::Array(items));
                }
                _ => return Err(JsonError::Malformed),
            }
        }
    }

    fn object(&mut self, depth: usize) -> Result<Value, JsonError> {
        let depth = depth + 1;
        if depth > MAX_JSON_DEPTH {
            return Err(JsonError::LimitExceeded);
        }
        self.eat(b'{')?;
        let mut map = HashMap::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Value::Object(map));
        }
        loop {
            if map.len() >= MAX_JSON_CONTAINER_LEN {
                return Err(JsonError::LimitExceeded);
            }
            self.skip_ws();
            let key = self.string()?;
            self.skip_ws();
            self.eat(b':')?;
            self.skip_ws();
            let v = self.value(depth)?;
            if map.insert(key, v).is_some() {
                return Err(JsonError::Malformed); // 重複キー
            }
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(Value::Object(map));
                }
                _ => return Err(JsonError::Malformed),
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, JsonError> {
        let end = self.pos.saturating_add(4);
        let s = self.src.get(self.pos..end).ok_or(JsonError::Malformed)?;
        let mut n = 0u32;
        for &b in s {
            let d = char::from(b).to_digit(16).ok_or(JsonError::Malformed)?;
            n = n * 16 + d;
        }
        self.pos = end;
        Ok(n)
    }

    fn string(&mut self) -> Result<String, JsonError> {
        self.eat(b'"')?;
        let mut out = String::new();
        loop {
            let b = self.peek().ok_or(JsonError::Malformed)?;
            match b {
                b'"' => {
                    self.pos += 1;
                    return Ok(out);
                }
                0..=0x1f => return Err(JsonError::Malformed),
                b'\\' => {
                    self.pos += 1;
                    let e = self.peek().ok_or(JsonError::Malformed)?;
                    self.pos += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            let cp = if (0xD800..0xDC00).contains(&hi) {
                                self.eat(b'\\')?;
                                self.eat(b'u')?;
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return Err(JsonError::Malformed);
                                }
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else {
                                hi
                            };
                            // 孤立した下位サロゲートはここで None になる。
                            out.push(char::from_u32(cp).ok_or(JsonError::Malformed)?);
                        }
                        _ => return Err(JsonError::Malformed),
                    }
                }
                _ => {
                    // 入力は検証済み UTF-8 で pos は文字境界にあるため、1 文字ずつ切り出す。
                    let rest = self.text.get(self.pos..).ok_or(JsonError::Malformed)?;
                    let c = rest.chars().next().ok_or(JsonError::Malformed)?;
                    out.push(c);
                    self.pos += c.len_utf8();
                }
            }
        }
    }

    fn digits(&mut self) -> usize {
        let start = self.pos;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.pos += 1;
        }
        self.pos - start
    }

    fn number(&mut self) -> Result<Value, JsonError> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        match self.peek() {
            Some(b'0') => self.pos += 1,
            Some(b'1'..=b'9') => {
                self.digits();
            }
            _ => return Err(JsonError::Malformed),
        }
        let mut is_int = true;
        if self.peek() == Some(b'.') {
            is_int = false;
            self.pos += 1;
            if self.digits() == 0 {
                return Err(JsonError::Malformed);
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            is_int = false;
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if self.digits() == 0 {
                return Err(JsonError::Malformed);
            }
        }
        if !is_int {
            return Ok(Value::NonIntNumber);
        }
        let s = self.text.get(start..self.pos).ok_or(JsonError::Malformed)?;
        Ok(s.parse::<i64>().map_or(Value::NonIntNumber, Value::Int))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> Result<Value, JsonError> {
        parse(s.as_bytes())
    }

    #[test]
    fn parses_all_value_kinds() {
        let v = p(
            r#" {"a":[1,-2,null,true,false,"x"],"b":{"c":1.5},"d":1e3,"e":99999999999999999999} "#,
        )
        .unwrap();
        let Value::Object(m) = v else { panic!() };
        assert_eq!(
            m["a"],
            Value::Array(vec![
                Value::Int(1),
                Value::Int(-2),
                Value::Null,
                Value::Bool(true),
                Value::Bool(false),
                Value::Str("x".into())
            ])
        );
        assert_eq!(m["d"], Value::NonIntNumber);
        assert_eq!(m["e"], Value::NonIntNumber);
    }

    #[test]
    fn decodes_escapes_and_surrogate_pairs() {
        assert_eq!(
            p(r#""a\n\"\\\/A😀é""#).unwrap(),
            Value::Str("a\n\"\\/A\u{1F600}é".into())
        );
    }

    #[test]
    fn rejects_malformed() {
        for s in [
            "[1,]",
            "[1",
            "{\"a\":1,}",
            "\"a\nb\"",
            r#""\ud83d""#,
            r#""\ude00""#,
            "NaN",
            "Infinity",
            "01",
            "+1",
            "1.",
            "-",
            "[1] x",
            "{\"a\":1,\"a\":2}",
            "\u{feff}1",
            "",
            r#""\x""#,
            r#""\u12""#,
            "tru",
        ] {
            assert_eq!(p(s), Err(JsonError::Malformed), "{s:?}");
        }
        assert_eq!(parse(&[b'"', 0xff, b'"']), Err(JsonError::Malformed));
    }

    #[test]
    fn rejects_depth_and_length_over_limit() {
        let n = MAX_JSON_DEPTH;
        assert!(p(&format!("{}{}", "[".repeat(n), "]".repeat(n))).is_ok());
        let m = n + 1;
        assert_eq!(
            p(&format!("{}{}", "[".repeat(m), "]".repeat(m))),
            Err(JsonError::LimitExceeded)
        );
        let wide = format!("[{}0]", "0,".repeat(MAX_JSON_CONTAINER_LEN));
        assert_eq!(p(&wide), Err(JsonError::LimitExceeded));
    }
}
