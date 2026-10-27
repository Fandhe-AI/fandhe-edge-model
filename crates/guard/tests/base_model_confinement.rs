//! ベースモデルのディレクトリ受付・経路の閉じ込め・読み込み前のサイズ確認・名前と内容の照合の結合テスト
//! （REQ-39・REQ-41・TASK-41.2-2・#398。証拠種別: テストハーネス。合成ディレクトリ）。

#![cfg(any(target_os = "linux", target_os = "macos"))]

use fandhe_edge_core::exitcode::ExitCode;
use fandhe_edge_core::hash::Sha256Digest;
use fandhe_edge_guard::base_model::{
    BaseModelFileRejection, MAX_BASE_MODEL_CONFIG_BYTES, open_base_model_file,
    verify_base_model_sha256,
};
use fandhe_edge_guard::file_size::{FileSizeRejection, MAX_READ_FILE_BYTES};
use fandhe_edge_guard::format::FileFormat;
use fandhe_edge_guard::package::{ConfinedPackage, confine_package};
use fandhe_edge_guard::split_files::MAX_SHARD_FILE_BYTES;
use std::collections::BTreeMap;
use std::fs;
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};

const TOKENIZER: &[u8] = br#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]}}"#;

fn safetensors() -> Vec<u8> {
    let hdr = r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut v = (hdr.len() as u64).to_le_bytes().to_vec();
    v.extend_from_slice(hdr.as_bytes());
    v.extend_from_slice(&[0; 4]);
    v
}

/// `<tmp>/workspace/models/qwen/` と `<tmp>/outside/`。Drop で片付ける。
struct Sandbox {
    base: PathBuf,
}

impl Sandbox {
    fn new(label: &str) -> Self {
        let base =
            std::env::temp_dir().join(format!("fandhe-guard-bmc-{}-{label}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(base.join("workspace/models/qwen")).unwrap();
        fs::create_dir_all(base.join("outside")).unwrap();
        Sandbox { base }
    }
    fn dir(&self) -> PathBuf {
        self.base.join("workspace/models/qwen")
    }
    fn outside(&self) -> PathBuf {
        self.base.join("outside")
    }
    fn confine(&self) -> ConfinedPackage {
        confine_package(&self.dir(), Path::new(".")).unwrap()
    }
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.base);
    }
}

fn reject(r: Result<impl std::fmt::Debug, BaseModelFileRejection>) -> BaseModelFileRejection {
    r.unwrap_err()
}

/// REQ-41・REQ-39: 正しいファイルは、ディレクトリ自身・workspace 配下のどちらに閉じ込めても通る。
#[test]
fn req41_accepts_valid_files() {
    let sb = Sandbox::new("ok");
    fs::write(sb.dir().join("model.safetensors"), safetensors()).unwrap();
    fs::write(sb.dir().join("tokenizer.json"), TOKENIZER).unwrap();
    let pkg = confine_package(&sb.base.join("workspace"), Path::new("models/qwen")).unwrap();
    for c in [sb.confine(), pkg] {
        let w = open_base_model_file(&c, Path::new("model.safetensors")).unwrap();
        assert_eq!(w.format(), FileFormat::Safetensors);
        let t = open_base_model_file(&c, Path::new("tokenizer.json")).unwrap();
        assert_eq!(t.format(), FileFormat::TokenizerJson);
    }
}

/// REQ-39: ディレクトリ自体が root 外なら受け付けない。
#[test]
fn req39_directory_outside_root_rejected() {
    let sb = Sandbox::new("dirout");
    let ws = sb.base.join("workspace");
    let e = confine_package(&ws, Path::new("../outside")).unwrap_err();
    assert_eq!(e.reason_code(), "path_escapes_root");
    symlink(sb.outside(), ws.join("link")).unwrap();
    let e = confine_package(&ws, Path::new("link")).unwrap_err();
    assert_eq!(e.reason_code(), "path_escapes_root");
    assert_eq!(e.exit_code(), ExitCode::InvalidInput);
}

/// REQ-39: `..`・絶対パスのメンバーは path_escapes_root(64)。
#[test]
fn req39_member_escape_rejected() {
    let sb = Sandbox::new("esc");
    fs::write(
        sb.base.join("workspace/models/outside.safetensors"),
        safetensors(),
    )
    .unwrap();
    let abs = sb.outside().join("x.safetensors");
    fs::write(&abs, safetensors()).unwrap();
    let c = sb.confine();
    for m in [Path::new("../outside.safetensors"), abs.as_path()] {
        let e = reject(open_base_model_file(&c, m));
        assert_eq!(e.reason_code(), "path_escapes_root");
        assert_eq!(e.exit_code(), ExitCode::InvalidInput);
    }
}

/// REQ-39: 末尾が symlink のメンバーは、指す先が外でも内でも拒否される（中身が正しくても）。
#[test]
fn req39_symlink_member_rejected() {
    let sb = Sandbox::new("sym");
    fs::write(sb.outside().join("real.safetensors"), safetensors()).unwrap();
    fs::write(sb.dir().join("real_tokenizer.json"), TOKENIZER).unwrap();
    symlink(
        sb.outside().join("real.safetensors"),
        sb.dir().join("model.safetensors"),
    )
    .unwrap();
    symlink("real_tokenizer.json", sb.dir().join("tokenizer.json")).unwrap();
    let c = sb.confine();
    for m in ["model.safetensors", "tokenizer.json"] {
        let e = reject(open_base_model_file(&c, Path::new(m)));
        assert_eq!(e.reason_code(), "not_regular_file", "{m}");
        assert_eq!(e.exit_code(), ExitCode::InvalidInput);
    }
}

/// REQ-39: 親ディレクトリが symlink のメンバーは path_escapes_root(64)。
#[test]
fn req39_symlink_parent_rejected() {
    let sb = Sandbox::new("symparent");
    fs::write(sb.outside().join("model.safetensors"), safetensors()).unwrap();
    symlink(sb.outside(), sb.dir().join("sub")).unwrap();
    let e = reject(open_base_model_file(
        &sb.confine(),
        Path::new("sub/model.safetensors"),
    ));
    assert_eq!(e.reason_code(), "path_escapes_root");
    assert_eq!(e.exit_code(), ExitCode::InvalidInput);
}

/// REQ-39: tokenizer.json は 1 GiB+1 で、読む前の fstat により file_size_limit_exceeded(20)。
#[test]
fn req39_oversize_rejected_before_read() {
    let sb = Sandbox::new("big");
    let f = fs::File::create(sb.dir().join("tokenizer.json")).unwrap();
    f.set_len(MAX_READ_FILE_BYTES + 1).unwrap();
    let e = reject(open_base_model_file(
        &sb.confine(),
        Path::new("tokenizer.json"),
    ));
    match &e {
        BaseModelFileRejection::Size(FileSizeRejection::TooLarge { size, limit }) => {
            assert_eq!(*size, MAX_READ_FILE_BYTES + 1);
            assert_eq!(*limit, MAX_READ_FILE_BYTES);
        }
        other => panic!("unexpected {other:?}"),
    }
    assert_eq!(e.exit_code(), ExitCode::LimitExceeded);
    assert_eq!(e.reason_code(), "file_size_limit_exceeded");
}

/// REQ-39・REQ-41・#655: 重みは 2 GiB まで。1 GiB+1 はサイズ検査を通り（形式段で拒否）、2 GiB+1 は Size で拒否。
#[test]
fn req39_weight_limit_is_2gib() {
    let sb = Sandbox::new("bigw");
    let f = fs::File::create(sb.dir().join("model.safetensors")).unwrap();
    f.set_len(MAX_READ_FILE_BYTES + 1).unwrap();
    let e = reject(open_base_model_file(
        &sb.confine(),
        Path::new("model.safetensors"),
    ));
    assert!(matches!(e, BaseModelFileRejection::Format { .. }), "{e:?}");
    f.set_len(MAX_SHARD_FILE_BYTES + 1).unwrap();
    let e = reject(open_base_model_file(
        &sb.confine(),
        Path::new("model.safetensors"),
    ));
    match &e {
        BaseModelFileRejection::Size(FileSizeRejection::TooLarge { size, limit }) => {
            assert_eq!(*size, MAX_SHARD_FILE_BYTES + 1);
            assert_eq!(*limit, MAX_SHARD_FILE_BYTES);
        }
        other => panic!("unexpected {other:?}"),
    }
    assert_eq!(e.reason_code(), "file_size_limit_exceeded");
}

/// REQ-41: 許可外の名前は、ファイルが存在しなくても開かずに base_model_file_name_not_allowed(64)。
#[test]
fn req41_name_not_allowed_without_touching_file() {
    let sb = Sandbox::new("name");
    let c = sb.confine();
    for m in ["pytorch_model.bin", "config.json", "model.pt"] {
        let e = reject(open_base_model_file(&c, Path::new(m)));
        assert_eq!(e.reason_code(), "base_model_file_name_not_allowed", "{m}");
        assert_eq!(e.exit_code(), ExitCode::InvalidInput);
    }
}

/// REQ-39・REQ-41: 名前と内容の不一致・pickle 偽装は format_not_allowed(64)。メッセージに漏えいなし。
#[test]
fn req41_name_content_mismatch_rejected() {
    let sb = Sandbox::new("mismatch");
    let mut pkl = vec![0x80, 0x02];
    pkl.extend_from_slice(b"canary-not-printed");
    fs::write(sb.dir().join("a.safetensors"), TOKENIZER).unwrap();
    fs::write(sb.dir().join("tokenizer.json"), safetensors()).unwrap();
    fs::write(sb.dir().join("b.safetensors"), &pkl).unwrap();
    let c = sb.confine();
    for m in ["a.safetensors", "tokenizer.json", "b.safetensors"] {
        let e = reject(open_base_model_file(&c, Path::new(m)));
        assert_eq!(e.reason_code(), "format_not_allowed", "{m}");
        assert_eq!(e.exit_code(), ExitCode::InvalidInput);
        let msg = e.to_string();
        assert!(!msg.contains(m) && !msg.contains("canary"), "{msg}");
        assert!(!msg.contains(sb.base.to_str().unwrap()), "{msg}");
    }
}

/// REQ-39: 絶対パス・`..` の member は、Display・Debug のどちらにも露出しない。
#[test]
fn req39_rejection_does_not_leak_member_path() {
    let sb = Sandbox::new("leak");
    for m in [
        "/private/customer/model.safetensors",
        "../customer.safetensors",
    ] {
        let e = reject(open_base_model_file(&sb.confine(), Path::new(m)));
        let shown = format!("{e} | {e:?} | {:?}", std::error::Error::source(&e));
        assert!(!shown.contains("customer"), "{shown}");
        assert!(!shown.contains("private"), "{shown}");
    }
}
const CONFIG: &[u8] = br#"{"model_type":"qwen2"}"#;

/// 合成のベースモデル 3 ファイルを置き、定義に書く sha256 の写像を返す。
fn write_base_model(dir: &Path) -> BTreeMap<String, Sha256Digest> {
    let files: [(&str, Vec<u8>); 3] = [
        ("config.json", CONFIG.to_vec()),
        ("model.safetensors", safetensors()),
        ("tokenizer.json", TOKENIZER.to_vec()),
    ];
    files
        .into_iter()
        .map(|(name, bytes)| {
            fs::write(dir.join(name), &bytes).unwrap();
            (name.to_string(), Sha256Digest::of_bytes(&bytes))
        })
        .collect()
}

fn verify_err(sb: &Sandbox, expected: &BTreeMap<String, Sha256Digest>) -> (ExitCode, &'static str) {
    let e = verify_base_model_sha256(&sb.confine(), expected).unwrap_err();
    (e.exit_code(), e.reason_code())
}

/// REQ-41・REQ-39・#628: 定義の sha256 と実ファイルが一致すれば通る。
#[test]
fn req41_issue628_verify_sha256_accepts_matching_files() {
    let sb = Sandbox::new("sha-ok");
    let expected = write_base_model(&sb.dir());
    verify_base_model_sha256(&sb.confine(), &expected).unwrap();
}

/// REQ-41・REQ-39・#628: 1 バイトの改変は sha256 不一致（64）。メッセージにファイル名を出さない。
#[test]
fn req41_issue628_verify_sha256_rejects_one_byte_change() {
    for name in ["config.json", "model.safetensors", "tokenizer.json"] {
        let sb = Sandbox::new(&format!("sha-ng-{name}"));
        let expected = write_base_model(&sb.dir());
        let mut bytes = fs::read(sb.dir().join(name)).unwrap();
        // safetensors はデータ部の末尾、JSON は末尾の `}` の直前に空白を足して形式は保つ
        if name == "model.safetensors" {
            *bytes.last_mut().unwrap() = 1;
        } else {
            bytes.insert(bytes.len() - 1, b' ');
        }
        fs::write(sb.dir().join(name), &bytes).unwrap();
        let e = verify_base_model_sha256(&sb.confine(), &expected).unwrap_err();
        assert!(
            matches!(e, BaseModelFileRejection::Sha256Mismatch),
            "{name}: {e:?}"
        );
        assert_eq!(e.exit_code(), ExitCode::InvalidInput);
        assert_eq!(e.reason_code(), "base_model_sha256_mismatch");
        assert!(!e.to_string().contains(name), "{e}");
    }
}

/// REQ-41・REQ-39・#628: ファイルの欠落・symlink のメンバーは経路の拒否（64）。
#[test]
fn req41_issue628_verify_sha256_rejects_missing_and_symlink() {
    let sb = Sandbox::new("sha-missing");
    let expected = write_base_model(&sb.dir());
    fs::remove_file(sb.dir().join("model.safetensors")).unwrap();
    assert_eq!(verify_err(&sb, &expected).0, ExitCode::InvalidInput);

    let sb = Sandbox::new("sha-symlink");
    let expected = write_base_model(&sb.dir());
    let real = sb.outside().join("config.json");
    fs::rename(sb.dir().join("config.json"), &real).unwrap();
    symlink(&real, sb.dir().join("config.json")).unwrap();
    assert_eq!(verify_err(&sb, &expected).0, ExitCode::InvalidInput);
}

/// REQ-41・REQ-39・#628: `config.json` は 64 KiB まで。超過は 20（読み込み前に止める）。
#[test]
fn req41_issue628_verify_sha256_config_size_limit() {
    let sb = Sandbox::new("sha-config-size");
    let mut expected = write_base_model(&sb.dir());
    let limit = usize::try_from(MAX_BASE_MODEL_CONFIG_BYTES).unwrap();
    let exact = vec![b' '; limit];
    fs::write(sb.dir().join("config.json"), &exact).unwrap();
    expected.insert("config.json".into(), Sha256Digest::of_bytes(&exact));
    verify_base_model_sha256(&sb.confine(), &expected).unwrap();

    let over = vec![b' '; limit + 1];
    fs::write(sb.dir().join("config.json"), &over).unwrap();
    expected.insert("config.json".into(), Sha256Digest::of_bytes(&over));
    assert_eq!(
        verify_err(&sb, &expected),
        (ExitCode::LimitExceeded, "file_size_limit_exceeded")
    );
}

/// REQ-41・REQ-39・#628: 許可外の名前は開かずに拒否する（定義の検証を迂回した写像への多層防御）。
#[test]
fn req41_issue628_verify_sha256_rejects_disallowed_name() {
    let sb = Sandbox::new("sha-name");
    let mut expected = write_base_model(&sb.dir());
    fs::write(sb.dir().join("pytorch_model.bin"), b"x").unwrap();
    expected.insert("pytorch_model.bin".into(), Sha256Digest::of_bytes(b"x"));
    assert_eq!(
        verify_err(&sb, &expected),
        (ExitCode::InvalidInput, "base_model_file_name_not_allowed")
    );
}
