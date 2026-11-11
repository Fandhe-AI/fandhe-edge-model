//! 常駐プロセス（`fandhe-edge serve`）と CLI（`infer`）の間の 1 行 JSON プロトコル（issue #708・#709）。
//!
//! cli の `serve`（要求の解析・応答の生成）と `infer` の委ねる処理（要求の生成・応答の解析）が使う。
//! 常駐の親と、モデルを持つ worker 子の間（パイプ）も同じ 1 行 JSON を使う（#709。[`ServeResponse::Ready`] は
//! worker 子から親への最初の 1 行専用）。
//! cli は `serde_json` に依存しない方針のため、JSON の読み書きはこのモジュールに閉じる。
//!
//! # 版の照合（#709）
//!
//! 要求は [`PROTOCOL_VERSION`] と CLI の版（`cli_version`）を名乗る。[`decode_request`] はまずこの 2 欄だけを
//! 未知の欄を許して読み、食い違えば [`ProtocolError::VersionMismatch`] を返す（将来の版が欄を増やしても
//! 「不正な要求」ではなく「版の食い違い」として断り、CLI が単独へ戻れるようにする）。一致したときだけ厳密に読む。
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

/// 1 行 JSON の形の版。要求の欄・意味を変えるときに上げる（#709）。
/// cli のソケットのファイル名（`serve.v<版>.sock`）もこの値に合わせて上げる（版の違う常駐と CLI が
/// 互いのソケットを見ないようにする。cli 側の単体テストで照合）。
pub const PROTOCOL_VERSION: u32 = 1;

/// `cli_version` の最大バイト数。
pub const MAX_CLI_VERSION_BYTES: usize = 64;

/// プロトコルの解析・検証エラー（外部入力のため panic せず返す）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProtocolError {
    /// 1 行が上限（バイト）を超えた。
    TooLarge { limit: usize },
    /// JSON として不正、または未知のフィールド・型不一致。
    InvalidJson(String),
    /// `protocol`・`cli_version` が受け手と違う、または欄が無い・型が違う（#709）。
    VersionMismatch,
    /// `id` が空または [`MAX_ID_BYTES`] 超過。
    InvalidId,
    /// `cli_version` が空または [`MAX_CLI_VERSION_BYTES`] 超過。
    InvalidCliVersion,
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
            Self::VersionMismatch => write!(f, "serve protocol version mismatch"),
            Self::InvalidId => write!(f, "invalid request id"),
            Self::InvalidCliVersion => write!(f, "invalid cli version"),
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
    /// 1 行 JSON の形の版（[`PROTOCOL_VERSION`]）。
    pub protocol: u32,
    /// 送り手の CLI の版（`CARGO_PKG_VERSION`）。常駐は自分の版と完全一致のときだけ受ける。
    pub cli_version: String,
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
    /// 要求の `protocol`・`cli_version` が常駐のものと違う。CLI は単独へ戻る（#709）。
    VersionMismatch,
    /// worker 子から常駐の親への最初の 1 行専用（読み込みが済んだ。`package` は読み込んだ内容の同一性、
    /// `cli_version` は worker の版。稼働中にバイナリが差し替わったことを親が検出する）。
    /// ソケットには流れない。CLI が受け取ったら不正な応答として扱う（#709）。
    Ready {
        package: PackageIdentity,
        cli_version: String,
    },
    /// worker 子から常駐の親への応答専用。[`ServeResponse::Ok`] と同じ判定結果だが、worker はこの応答を
    /// 最後に自分で終わる（親は次の要求を渡さずに作り直す）。親はクライアントへ `Ok` として返す。
    /// ソケットには流れない。CLI が受け取ったら不正な応答として扱う（#709）。
    OkRetiring { exit_code: u8, stdout: String },
}

/// 応答の線上表現。内部タグ付き enum の単位 variant は `deny_unknown_fields` が効かないため、
/// `PackageMismatch {}` の形で受けて未知のフィールドを拒否する。
#[derive(Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
enum Wire {
    Ok {
        exit_code: u8,
        stdout: String,
    },
    PackageMismatch {},
    VersionMismatch {},
    Ready {
        package: PackageIdentity,
        cli_version: String,
    },
    OkRetiring {
        exit_code: u8,
        stdout: String,
    },
}

/// 版の 2 欄だけを先に読むための型。未知の欄を許し、型の違いも読めるよう値のまま受ける。
#[derive(Deserialize)]
struct VersionProbe {
    protocol: Option<serde_json::Value>,
    cli_version: Option<serde_json::Value>,
}

fn valid_sha256(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

fn validate_request(r: &ServeRequest) -> Result<(), ProtocolError> {
    if r.id.is_empty() || r.id.len() > MAX_ID_BYTES {
        return Err(ProtocolError::InvalidId);
    }
    validate_cli_version(&r.cli_version)?;
    validate_identity(&r.package)?;
    if r.text.len() > MAX_INFER_INPUT_BYTES {
        return Err(ProtocolError::TooLarge {
            limit: MAX_INFER_INPUT_BYTES,
        });
    }
    Ok(())
}

fn validate_cli_version(v: &str) -> Result<(), ProtocolError> {
    if v.is_empty() || v.len() > MAX_CLI_VERSION_BYTES {
        return Err(ProtocolError::InvalidCliVersion);
    }
    Ok(())
}

fn validate_identity(p: &PackageIdentity) -> Result<(), ProtocolError> {
    if p.shard_sha256.len() > MAX_EXTERNAL_DATA_SHARDS {
        return Err(ProtocolError::TooManyShards);
    }
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
///
/// まず `protocol`・`cli_version` だけを読み、[`PROTOCOL_VERSION`]・`expected_cli_version` と食い違えば
/// （欄が無い・型が違う場合も）[`ProtocolError::VersionMismatch`]。一致したら未知の欄を拒否して読む。
pub fn decode_request(
    line: &[u8],
    expected_cli_version: &str,
) -> Result<ServeRequest, ProtocolError> {
    let body = trim_line(line, MAX_REQUEST_LINE_BYTES)?;
    let probe: VersionProbe =
        serde_json::from_slice(body).map_err(|e| ProtocolError::InvalidJson(e.to_string()))?;
    let protocol_ok = probe.protocol.as_ref().and_then(serde_json::Value::as_u64)
        == Some(u64::from(PROTOCOL_VERSION));
    let cli_ok = probe
        .cli_version
        .as_ref()
        .and_then(serde_json::Value::as_str)
        == Some(expected_cli_version);
    if !protocol_ok || !cli_ok {
        return Err(ProtocolError::VersionMismatch);
    }
    let req: ServeRequest =
        serde_json::from_slice(body).map_err(|e| ProtocolError::InvalidJson(e.to_string()))?;
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
        ServeResponse::VersionMismatch => Wire::VersionMismatch {},
        ServeResponse::Ready {
            package,
            cli_version,
        } => Wire::Ready {
            package: package.clone(),
            cli_version: cli_version.clone(),
        },
        ServeResponse::OkRetiring { exit_code, stdout } => Wire::OkRetiring {
            exit_code: *exit_code,
            stdout: stdout.clone(),
        },
    })
}

/// 応答 1 行を解析する（末尾 `\n` は有無とも可）。
pub fn decode_response(line: &[u8]) -> Result<ServeResponse, ProtocolError> {
    let wire: Wire = serde_json::from_slice(trim_line(line, MAX_RESPONSE_LINE_BYTES)?)
        .map_err(|e| ProtocolError::InvalidJson(e.to_string()))?;
    Ok(match wire {
        Wire::Ok { exit_code, stdout } => ServeResponse::Ok { exit_code, stdout },
        Wire::PackageMismatch {} => ServeResponse::PackageMismatch,
        Wire::VersionMismatch {} => ServeResponse::VersionMismatch,
        Wire::Ready {
            package,
            cli_version,
        } => {
            validate_identity(&package)?;
            validate_cli_version(&cli_version)?;
            ServeResponse::Ready {
                package,
                cli_version,
            }
        }
        Wire::OkRetiring { exit_code, stdout } => ServeResponse::OkRetiring { exit_code, stdout },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// テスト用の CLI の版。
    const V: &str = "1.2.3";

    /// 受け手の版を [`V`] として読む。
    fn dec(line: &[u8]) -> Result<ServeRequest, ProtocolError> {
        decode_request(line, V)
    }

    fn h(c: char) -> String {
        c.to_string().repeat(64)
    }

    fn req(text: &str) -> ServeRequest {
        ServeRequest {
            protocol: PROTOCOL_VERSION,
            cli_version: V.into(),
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
            "{{\"protocol\":1,\"cli_version\":\"1.2.3\",\"id\":\"r1\",\"text\":\"hi\",\"package\":{{\"onnx_sha256\":\"{}\",\"shard_sha256\":[\"{}\"],\"artifact_sha256\":\"{}\",\"definition_sha256\":\"{}\",\"calibration_sha256\":null,\"layout_sha256\":\"{}\"}}}}\n",
            h('a'),
            h('b'),
            h('c'),
            h('d'),
            h('f')
        );
        assert_eq!(bytes, expected.as_bytes());
        assert_eq!(dec(&bytes), Ok(r.clone()));
        assert_eq!(dec(&bytes[..bytes.len() - 1]), Ok(r));
    }

    #[test]
    fn escaped_text_roundtrip() {
        let r = req("a\"b\\c\nd\u{0}\u{1f600}日本語");
        let bytes = encode_request(&r);
        assert_eq!(bytes.iter().filter(|&&b| b == b'\n').count(), 1);
        assert_eq!(dec(&bytes).map(|x| x.text), Ok(r.text));
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
            dec(v.as_bytes()),
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
            dec(b"not json"),
            Err(ProtocolError::InvalidJson(_))
        ));
    }

    #[test]
    fn size_limits() {
        let big = vec![b' '; MAX_REQUEST_LINE_BYTES + 1];
        assert_eq!(
            dec(&big),
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
        assert_eq!(dec(&bytes).map(|x| x.text.len()), Ok(MAX_INFER_INPUT_BYTES));
    }

    #[test]
    fn invalid_fields_rejected() {
        let mut r = req("t");
        r.package.onnx_sha256 = "A".repeat(64);
        assert_eq!(dec(&encode_request(&r)), Err(ProtocolError::InvalidSha256));
        r.package.onnx_sha256 = "a".repeat(63);
        assert_eq!(dec(&encode_request(&r)), Err(ProtocolError::InvalidSha256));
        let mut r = req("t");
        r.package.shard_sha256 = vec!["g".repeat(64)];
        assert_eq!(dec(&encode_request(&r)), Err(ProtocolError::InvalidSha256));
        let mut r = req("t");
        r.package.shard_sha256 = vec![h('c'); MAX_EXTERNAL_DATA_SHARDS + 1];
        assert_eq!(dec(&encode_request(&r)), Err(ProtocolError::TooManyShards));
        r.package.shard_sha256 = vec![h('c'); MAX_EXTERNAL_DATA_SHARDS];
        assert!(dec(&encode_request(&r)).is_ok());
        let mut r = req("t");
        r.id = String::new();
        assert_eq!(dec(&encode_request(&r)), Err(ProtocolError::InvalidId));
        r.id = "i".repeat(MAX_ID_BYTES + 1);
        assert_eq!(dec(&encode_request(&r)), Err(ProtocolError::InvalidId));
        r.id = "i".repeat(MAX_ID_BYTES);
        assert!(dec(&encode_request(&r)).is_ok());
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
        assert_eq!(dec(&bytes), Ok(r.clone()));
        r.package.calibration_sha256 = Some("E".repeat(64));
        assert_eq!(dec(&encode_request(&r)), Err(ProtocolError::InvalidSha256));
        let mut r = req("t");
        r.package.artifact_sha256 = "a".repeat(10);
        assert_eq!(dec(&encode_request(&r)), Err(ProtocolError::InvalidSha256));
        let mut r = req("t");
        r.package.layout_sha256 = "F".repeat(64);
        assert_eq!(dec(&encode_request(&r)), Err(ProtocolError::InvalidSha256));
        let mut r = req("t");
        r.package.definition_sha256 = String::new();
        assert_eq!(dec(&encode_request(&r)), Err(ProtocolError::InvalidSha256));
    }

    #[test]
    fn text_limit_boundary() {
        let ok = req(&"a".repeat(MAX_INFER_INPUT_BYTES));
        assert!(dec(&encode_request(&ok)).is_ok());
        let over = req(&"a".repeat(MAX_INFER_INPUT_BYTES + 1));
        assert_eq!(
            dec(&encode_request(&over)),
            Err(ProtocolError::TooLarge {
                limit: MAX_INFER_INPUT_BYTES
            })
        );
    }

    /// #709: 版の食い違い（protocol 違い・`cli_version` 違い・欄なし・型違い）は、未知の欄があっても
    /// `VersionMismatch` になる。JSON として読めない行は今までどおり `InvalidJson`。
    #[test]
    fn version_mismatch_detected_before_strict_parse() {
        let line = String::from_utf8(encode_request(&req("t"))).unwrap();
        let mm = |l: &str| {
            assert_eq!(
                dec(l.as_bytes()),
                Err(ProtocolError::VersionMismatch),
                "{l}"
            )
        };
        mm(&line.replacen("\"protocol\":1", "\"protocol\":2", 1));
        mm(&line.replacen("\"cli_version\":\"1.2.3\"", "\"cli_version\":\"1.2.4\"", 1));
        mm(&line.replacen("\"protocol\":1,", "", 1));
        mm(&line.replacen("\"cli_version\":\"1.2.3\",", "", 1));
        mm(&line.replacen("\"protocol\":1", "\"protocol\":\"1\"", 1));
        mm(&line.replacen("\"cli_version\":\"1.2.3\"", "\"cli_version\":3", 1));
        // 将来の版が欄を増やしても、不正な要求ではなく版の食い違いとして断る。
        mm(&line.replacen("\"protocol\":1", "\"future\":{\"x\":1},\"protocol\":2", 1));
        mm("{}");
        // 受け手の版が違えば同じ行でも食い違い。
        assert_eq!(
            decode_request(line.as_bytes(), "9.9.9"),
            Err(ProtocolError::VersionMismatch)
        );
        assert!(matches!(
            dec(b"not json"),
            Err(ProtocolError::InvalidJson(_))
        ));
        assert_eq!(dec(line.as_bytes()), Ok(req("t")));
    }

    /// #709: `cli_version` は空でなく 64 バイト以内。
    #[test]
    fn cli_version_length_validated() {
        let mut r = req("t");
        r.cli_version = "v".repeat(MAX_CLI_VERSION_BYTES + 1);
        assert_eq!(
            decode_request(&encode_request(&r), &r.cli_version),
            Err(ProtocolError::InvalidCliVersion)
        );
        r.cli_version = String::new();
        assert_eq!(
            decode_request(&encode_request(&r), ""),
            Err(ProtocolError::InvalidCliVersion)
        );
        r.cli_version = "v".repeat(MAX_CLI_VERSION_BYTES);
        assert_eq!(decode_request(&encode_request(&r), &r.cli_version), Ok(r));
    }

    /// #709: `version_mismatch`・`ready`・`ok_retiring` の線上表現と往復。`ready` の sha256 と版も検証する。
    #[test]
    fn version_mismatch_ready_and_retiring_roundtrip() {
        let vm = encode_response(&ServeResponse::VersionMismatch);
        assert_eq!(vm, b"{\"status\":\"version_mismatch\"}\n");
        assert_eq!(decode_response(&vm), Ok(ServeResponse::VersionMismatch));
        assert!(matches!(
            decode_response(b"{\"status\":\"version_mismatch\",\"x\":1}"),
            Err(ProtocolError::InvalidJson(_))
        ));
        let ready = |package: PackageIdentity, v: &str| ServeResponse::Ready {
            package,
            cli_version: v.into(),
        };
        let bytes = encode_response(&ready(req("t").package, V));
        let text = String::from_utf8(bytes.clone()).unwrap();
        assert!(
            text.starts_with("{\"status\":\"ready\",\"package\":{\"onnx_sha256\":\""),
            "{text}"
        );
        assert!(text.ends_with("},\"cli_version\":\"1.2.3\"}\n"), "{text}");
        assert_eq!(decode_response(&bytes), Ok(ready(req("t").package, V)));
        let mut bad = req("t").package;
        bad.onnx_sha256 = "A".repeat(64);
        assert_eq!(
            decode_response(&encode_response(&ready(bad, V))),
            Err(ProtocolError::InvalidSha256)
        );
        assert_eq!(
            decode_response(&encode_response(&ready(req("t").package, ""))),
            Err(ProtocolError::InvalidCliVersion)
        );
        // 版の欄が無い `ready`（古い worker）は読めない。
        assert!(matches!(
            decode_response(text.replace(",\"cli_version\":\"1.2.3\"", "").as_bytes()),
            Err(ProtocolError::InvalidJson(_))
        ));
        let retiring = ServeResponse::OkRetiring {
            exit_code: 20,
            stdout: "x\n".into(),
        };
        let bytes = encode_response(&retiring);
        assert_eq!(
            bytes,
            b"{\"status\":\"ok_retiring\",\"exit_code\":20,\"stdout\":\"x\\n\"}\n"
        );
        assert_eq!(decode_response(&bytes), Ok(retiring));
    }
}
