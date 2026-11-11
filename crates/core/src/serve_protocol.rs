//! 常駐プロセス（`fandhe-edge serve`）と CLI（`infer`）の間の 1 行 JSON プロトコル（issue #708）。
//!
//! cli の `serve`（要求の解析・応答の生成）と `infer` の委ねる処理（要求の生成・応答の解析）が使う。
//! cli は `serde_json` に依存しない方針のため、JSON の読み書きはこのモジュールに閉じる。
//!
//! 操作アダプター（cli の `serve` と `infer` の委ねる処理）専用の型で、推論・学習の層からは参照しない。
//!
//! - REQ-21・REQ-33: 応答の `stdout` は加工せずそのまま運び、単独実行と stdout のバイト列・終了コードを同一にする
//! - REQ-38: 通信はローカルの Unix ソケットだけ。このモジュールはバイト列の変換のみで I/O を持たない
//! - REQ-39: 要求・応答 1 行の大きさ、`id` の長さ、shard の件数を上限検証してから受理する
//!   （読み取り側は [`MAX_REQUEST_LINE_BYTES`]・[`MAX_RESPONSE_LINE_BYTES`] を超えて読まないこと）

use serde::{Deserialize, Serialize};

use crate::infer_input::MAX_INFER_INPUT_BYTES;
use crate::limits::MAX_EXTERNAL_DATA_SHARDS;

/// 要求 1 行の最大バイト数。`text` は JSON エスケープで最大 6 倍（`\uXXXX`）になるため
/// [`MAX_INFER_INPUT_BYTES`] の 6 倍に、`id`・`package`・キー名の分として 64 KiB を足す。
pub const MAX_REQUEST_LINE_BYTES: usize = MAX_INFER_INPUT_BYTES * 6 + 64 * 1024;

/// 応答 1 行の最大バイト数（16 MiB）。`infer` 1 件の stdout は判定結果 JSON 1 つで小さい。
/// `limits::MAX_OUTPUT_BYTES`（256 MiB）の 6 倍を許すと 1.5 GiB の確保を許すことになるため、
/// 実際に取りうる大きさに対して十分な値に抑えた（暫定。REQ-39）。
pub const MAX_RESPONSE_LINE_BYTES: usize = 16 * 1024 * 1024;

/// `id` の最大バイト数。
pub const MAX_ID_BYTES: usize = 64;

/// プロトコルの解析・検証エラー（外部入力のため panic せず返す）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProtocolError {
    /// 1 行が上限（バイト）を超えた。
    TooLarge { limit: usize },
    /// JSON として不正、または未知のフィールド・型不一致。
    InvalidJson(String),
    /// `id` が空または [`MAX_ID_BYTES`] 超過。
    InvalidId,
    /// sha256 が小文字 hex 64 桁でない。
    InvalidSha256,
    /// shard の件数が [`MAX_EXTERNAL_DATA_SHARDS`] を超えた。
    TooManyShards,
}

impl std::fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooLarge { limit } => write!(f, "serve protocol line exceeds {limit} bytes"),
            Self::InvalidJson(m) => write!(f, "invalid serve protocol JSON: {m}"),
            Self::InvalidId => write!(f, "invalid request id"),
            Self::InvalidSha256 => write!(f, "sha256 must be 64 lowercase hex digits"),
            Self::TooManyShards => write!(f, "too many shard hashes"),
        }
    }
}

impl std::error::Error for ProtocolError {}

/// パッケージの同一性（各ファイルのバイト列の sha256）。常駐が自分のものと `==` で比べる。
/// ONNX が同じでも定義・校正・メタが違うパッケージを同一視しないよう、これらも含める。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackageIdentity {
    pub onnx_sha256: String,
    pub shard_sha256: Vec<String>,
    /// `artifact.json` のバイト列。
    pub artifact_sha256: String,
    /// `definition.json` のバイト列。
    pub definition_sha256: String,
    /// 校正ファイルのバイト列。校正ファイルが無ければ `None`（JSON では null）。
    pub calibration_sha256: Option<String>,
    /// 構成ファイルの配置の指紋（inode・サイズ・mtime・ctime など）を cli 側でまとめた sha256。
    /// 計算は cli が行い、core は運んで照合するだけ。常駐の起動後にファイルが差し替えられたら
    /// 値が変わり、委ねる側が気づいて単独に戻れる（REQ-39 の完全性）。
    pub layout_sha256: String,
}

/// CLI から常駐への要求。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServeRequest {
    pub id: String,
    pub text: String,
    pub package: PackageIdentity,
}

/// 常駐から CLI への応答。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServeResponse {
    /// 判定を行った。`stdout` は単独実行と同一のバイト列（UTF-8）で、加工せず流す。
    Ok { exit_code: u8, stdout: String },
    /// 要求のパッケージが常駐のものと違う。CLI は単独へ戻る。
    PackageMismatch,
}

/// 応答の線上表現。内部タグ付き enum の単位 variant は `deny_unknown_fields` が効かないため、
/// `PackageMismatch {}` の形で受けて未知のフィールドを拒否する。
#[derive(Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
enum Wire {
    Ok { exit_code: u8, stdout: String },
    PackageMismatch {},
}

fn valid_sha256(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

fn validate_request(r: &ServeRequest) -> Result<(), ProtocolError> {
    if r.id.is_empty() || r.id.len() > MAX_ID_BYTES {
        return Err(ProtocolError::InvalidId);
    }
    if r.package.shard_sha256.len() > MAX_EXTERNAL_DATA_SHARDS {
        return Err(ProtocolError::TooManyShards);
    }
    let p = &r.package;
    let all = [
        &p.onnx_sha256,
        &p.artifact_sha256,
        &p.definition_sha256,
        &p.layout_sha256,
    ]
    .into_iter()
    .chain(&p.shard_sha256)
    .chain(p.calibration_sha256.as_ref());
    if all.into_iter().any(|h| !valid_sha256(h)) {
        return Err(ProtocolError::InvalidSha256);
    }
    if r.text.len() > MAX_INFER_INPUT_BYTES {
        return Err(ProtocolError::TooLarge {
            limit: MAX_INFER_INPUT_BYTES,
        });
    }
    Ok(())
}

fn trim_line(line: &[u8], limit: usize) -> Result<&[u8], ProtocolError> {
    let body = line.strip_suffix(b"\n").unwrap_or(line);
    if body.len() > limit {
        return Err(ProtocolError::TooLarge { limit });
    }
    Ok(body)
}

fn to_line<T: Serialize>(v: &T) -> Vec<u8> {
    // 文字列・整数・Option のみの型で、直列化は失敗しない。万一の失敗は空行ではなく
    // 不正 JSON の行として相手に伝え（受け側は InvalidJson で拒否する）、デバッグ時は検出する。
    let mut out = serde_json::to_vec(v).unwrap_or_else(|_| {
        debug_assert!(false, "serve protocol serialization failed");
        b"null".to_vec()
    });
    out.push(b'\n');
    out
}

/// 要求を 1 行（末尾 `\n`）へ直列化する。
pub fn encode_request(req: &ServeRequest) -> Vec<u8> {
    to_line(req)
}

/// 要求 1 行を解析・検証する（末尾 `\n` は有無とも可）。
pub fn decode_request(line: &[u8]) -> Result<ServeRequest, ProtocolError> {
    let req: ServeRequest = serde_json::from_slice(trim_line(line, MAX_REQUEST_LINE_BYTES)?)
        .map_err(|e| ProtocolError::InvalidJson(e.to_string()))?;
    validate_request(&req)?;
    Ok(req)
}

/// 応答を 1 行（末尾 `\n`）へ直列化する。
pub fn encode_response(res: &ServeResponse) -> Vec<u8> {
    to_line(&match res {
        ServeResponse::Ok { exit_code, stdout } => Wire::Ok {
            exit_code: *exit_code,
            stdout: stdout.clone(),
        },
        ServeResponse::PackageMismatch => Wire::PackageMismatch {},
    })
}

/// 応答 1 行を解析する（末尾 `\n` は有無とも可）。
pub fn decode_response(line: &[u8]) -> Result<ServeResponse, ProtocolError> {
    let wire: Wire = serde_json::from_slice(trim_line(line, MAX_RESPONSE_LINE_BYTES)?)
        .map_err(|e| ProtocolError::InvalidJson(e.to_string()))?;
    Ok(match wire {
        Wire::Ok { exit_code, stdout } => ServeResponse::Ok { exit_code, stdout },
        Wire::PackageMismatch {} => ServeResponse::PackageMismatch,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(c: char) -> String {
        c.to_string().repeat(64)
    }

    fn req(text: &str) -> ServeRequest {
        ServeRequest {
            id: "r1".into(),
            text: text.into(),
            package: PackageIdentity {
                onnx_sha256: h('a'),
                shard_sha256: vec![h('b')],
                artifact_sha256: h('c'),
                definition_sha256: h('d'),
                calibration_sha256: None,
                layout_sha256: h('f'),
            },
        }
    }

    #[test]
    fn request_roundtrip_and_wire_format() {
        let r = req("hi");
        let bytes = encode_request(&r);
        let expected = format!(
            "{{\"id\":\"r1\",\"text\":\"hi\",\"package\":{{\"onnx_sha256\":\"{}\",\"shard_sha256\":[\"{}\"],\"artifact_sha256\":\"{}\",\"definition_sha256\":\"{}\",\"calibration_sha256\":null,\"layout_sha256\":\"{}\"}}}}\n",
            h('a'),
            h('b'),
            h('c'),
            h('d'),
            h('f')
        );
        assert_eq!(bytes, expected.as_bytes());
        assert_eq!(decode_request(&bytes), Ok(r.clone()));
        assert_eq!(decode_request(&bytes[..bytes.len() - 1]), Ok(r));
    }

    #[test]
    fn escaped_text_roundtrip() {
        let r = req("a\"b\\c\nd\u{0}\u{1f600}日本語");
        let bytes = encode_request(&r);
        assert_eq!(bytes.iter().filter(|&&b| b == b'\n').count(), 1);
        assert_eq!(decode_request(&bytes).map(|x| x.text), Ok(r.text));
    }

    #[test]
    fn response_roundtrip_and_wire_format() {
        let ok = ServeResponse::Ok {
            exit_code: 10,
            stdout: "x\n\"y\"".into(),
        };
        let bytes = encode_response(&ok);
        assert_eq!(
            bytes,
            b"{\"status\":\"ok\",\"exit_code\":10,\"stdout\":\"x\\n\\\"y\\\"\"}\n"
        );
        assert_eq!(decode_response(&bytes), Ok(ok));
        let mm = encode_response(&ServeResponse::PackageMismatch);
        assert_eq!(mm, b"{\"status\":\"package_mismatch\"}\n");
        assert_eq!(
            decode_response(b"{\"status\":\"package_mismatch\"}"),
            Ok(ServeResponse::PackageMismatch)
        );
    }

    #[test]
    fn unknown_fields_rejected() {
        let mut v = String::from_utf8(encode_request(&req("t"))).unwrap();
        v = v.replacen("\"id\"", "\"extra\":1,\"id\"", 1);
        assert!(matches!(
            decode_request(v.as_bytes()),
            Err(ProtocolError::InvalidJson(_))
        ));
        let bad = b"{\"status\":\"package_mismatch\",\"x\":1}";
        assert!(matches!(
            decode_response(bad),
            Err(ProtocolError::InvalidJson(_))
        ));
        assert!(matches!(
            decode_response(b"{\"status\":\"ok\"}"),
            Err(ProtocolError::InvalidJson(_))
        ));
        assert!(matches!(
            decode_request(b"not json"),
            Err(ProtocolError::InvalidJson(_))
        ));
    }

    #[test]
    fn size_limits() {
        let big = vec![b' '; MAX_REQUEST_LINE_BYTES + 1];
        assert_eq!(
            decode_request(&big),
            Err(ProtocolError::TooLarge {
                limit: MAX_REQUEST_LINE_BYTES
            })
        );
        let big = vec![b' '; MAX_RESPONSE_LINE_BYTES + 1];
        assert_eq!(
            decode_response(&big),
            Err(ProtocolError::TooLarge {
                limit: MAX_RESPONSE_LINE_BYTES
            })
        );
        // 最大入力が全部 6 倍にエスケープされても上限に収まる。
        let r = req(&"\u{1}".repeat(MAX_INFER_INPUT_BYTES));
        let bytes = encode_request(&r);
        assert!(bytes.len() <= MAX_REQUEST_LINE_BYTES + 1);
        assert_eq!(
            decode_request(&bytes).map(|x| x.text.len()),
            Ok(MAX_INFER_INPUT_BYTES)
        );
    }

    #[test]
    fn invalid_fields_rejected() {
        let mut r = req("t");
        r.package.onnx_sha256 = "A".repeat(64);
        assert_eq!(
            decode_request(&encode_request(&r)),
            Err(ProtocolError::InvalidSha256)
        );
        r.package.onnx_sha256 = "a".repeat(63);
        assert_eq!(
            decode_request(&encode_request(&r)),
            Err(ProtocolError::InvalidSha256)
        );
        let mut r = req("t");
        r.package.shard_sha256 = vec!["g".repeat(64)];
        assert_eq!(
            decode_request(&encode_request(&r)),
            Err(ProtocolError::InvalidSha256)
        );
        let mut r = req("t");
        r.package.shard_sha256 = vec![h('c'); MAX_EXTERNAL_DATA_SHARDS + 1];
        assert_eq!(
            decode_request(&encode_request(&r)),
            Err(ProtocolError::TooManyShards)
        );
        r.package.shard_sha256 = vec![h('c'); MAX_EXTERNAL_DATA_SHARDS];
        assert!(decode_request(&encode_request(&r)).is_ok());
        let mut r = req("t");
        r.id = String::new();
        assert_eq!(
            decode_request(&encode_request(&r)),
            Err(ProtocolError::InvalidId)
        );
        r.id = "i".repeat(MAX_ID_BYTES + 1);
        assert_eq!(
            decode_request(&encode_request(&r)),
            Err(ProtocolError::InvalidId)
        );
        r.id = "i".repeat(MAX_ID_BYTES);
        assert!(decode_request(&encode_request(&r)).is_ok());
    }

    #[test]
    fn new_identity_fields_validated_and_calibration_roundtrip() {
        let mut r = req("t");
        r.package.calibration_sha256 = Some(h('e'));
        let bytes = encode_request(&r);
        assert!(
            String::from_utf8(bytes.clone())
                .unwrap()
                .contains(&format!("\"calibration_sha256\":\"{}\"", h('e')))
        );
        assert_eq!(decode_request(&bytes), Ok(r.clone()));
        r.package.calibration_sha256 = Some("E".repeat(64));
        assert_eq!(
            decode_request(&encode_request(&r)),
            Err(ProtocolError::InvalidSha256)
        );
        let mut r = req("t");
        r.package.artifact_sha256 = "a".repeat(10);
        assert_eq!(
            decode_request(&encode_request(&r)),
            Err(ProtocolError::InvalidSha256)
        );
        let mut r = req("t");
        r.package.layout_sha256 = "F".repeat(64);
        assert_eq!(
            decode_request(&encode_request(&r)),
            Err(ProtocolError::InvalidSha256)
        );
        let mut r = req("t");
        r.package.definition_sha256 = String::new();
        assert_eq!(
            decode_request(&encode_request(&r)),
            Err(ProtocolError::InvalidSha256)
        );
    }

    #[test]
    fn text_limit_boundary() {
        let ok = req(&"a".repeat(MAX_INFER_INPUT_BYTES));
        assert!(decode_request(&encode_request(&ok)).is_ok());
        let over = req(&"a".repeat(MAX_INFER_INPUT_BYTES + 1));
        assert_eq!(
            decode_request(&encode_request(&over)),
            Err(ProtocolError::TooLarge {
                limit: MAX_INFER_INPUT_BYTES
            })
        );
    }
}
