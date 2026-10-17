//! `resolve_external` と外部データ参照の復号のテスト（REQ-39・REQ-41・TASK-41.7-8・#581）。
//! 証拠種別: テストハーネス（一時ディレクトリの実ファイルと、手書きの最小 protobuf）。

use super::super::proto::decode_model;
use super::super::{FINETUNE_MODEL_LIMITS, ModelLimits, STRICT_MODEL_LIMITS};
use super::*;
use std::io::Write as _;

// ---- 最小の protobuf 符号化 ----

fn varint(mut v: u64) -> Vec<u8> {
    let mut o = Vec::new();
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            o.push(b);
            return o;
        }
        o.push(b | 0x80);
    }
}
fn field_varint(num: u64, v: u64) -> Vec<u8> {
    [varint(num << 3), varint(v)].concat()
}
fn field_len(num: u64, body: &[u8]) -> Vec<u8> {
    [
        varint(num << 3 | 2),
        varint(body.len() as u64),
        body.to_vec(),
    ]
    .concat()
}
fn entry(k: &str, v: &str) -> Vec<u8> {
    field_len(
        13,
        &[field_len(1, k.as_bytes()), field_len(2, v.as_bytes())].concat(),
    )
}
/// 外部データを参照するテンソル（initializer）。
fn ext_tensor(name: &str, dims: &[u64], dt: u64, loc: &str, off: u64, len: u64) -> Vec<u8> {
    let mut b = Vec::new();
    for d in dims {
        b.extend(field_varint(1, *d));
    }
    b.extend(field_varint(2, dt));
    b.extend(field_len(8, name.as_bytes()));
    b.extend(entry("location", loc));
    b.extend(entry("offset", &off.to_string()));
    b.extend(entry("length", &len.to_string()));
    b.extend(field_varint(14, 1));
    b
}
/// ir_version=8 と initializer のみのグラフ。
fn model(inits: &[Vec<u8>]) -> Vec<u8> {
    let mut g = Vec::new();
    for t in inits {
        g.extend(field_len(5, t));
    }
    [field_varint(1, 8), field_len(7, &g)].concat()
}
fn decode(inits: &[Vec<u8>], limits: &ModelLimits) -> Result<GraphProto, OnnxLoadError> {
    decode_model(&model(inits), limits).map(|m| m.graph)
}

struct Dir(PathBuf);
impl Dir {
    fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "fandhe-onnx-ext-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
    fn shard(&self, name: &str, bytes: &[u8]) -> ExternalShard {
        let p = self.0.join(name);
        std::fs::File::create(&p).unwrap().write_all(bytes).unwrap();
        ExternalShard {
            location: name.into(),
            file: File::open(&p).unwrap(),
            sha256: Sha256Digest::of_bytes(bytes),
        }
    }
}
impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn lim() -> ExternalDataLimits {
    ExternalDataLimits {
        max_shard_bytes: 1024,
        max_shards: 4,
        max_total_bytes: 2048,
    }
}
const FT: ModelLimits = FINETUNE_MODEL_LIMITS;

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn code(r: Result<(), OnnxLoadError>) -> &'static str {
    r.unwrap_err().code()
}

#[test]
fn req39_strict_rejects_external_data() {
    let t = ext_tensor("w", &[2], 1, "a.bin", 0, 8);
    assert_eq!(
        decode(&[t], &STRICT_MODEL_LIMITS).unwrap_err().code(),
        "unsupported_tensor"
    );
    // bf16 も C1・C3 では拒否
    let raw = [
        field_varint(1, 1),
        field_varint(2, 16),
        field_len(8, b"b"),
        field_len(9, &[0, 0]),
    ]
    .concat();
    assert_eq!(
        decode(&[raw], &STRICT_MODEL_LIMITS).unwrap_err().code(),
        "unsupported_tensor"
    );
}

#[test]
fn req41_finetune_loads_external_f32_and_bf16_across_two_shards() {
    let d = Dir::new();
    // shard a: [pad 4B][f32 1.5, -2.0, 3.25] / shard b: [bf16 0x3f80, 0x4000, 0xc040][i64 -7]
    let a = [vec![9u8; 4], f32s(&[1.5, -2.0, 3.25])].concat();
    let b = [
        [0x80u8, 0x3f, 0x00, 0x40, 0x40, 0xc0].to_vec(),
        (-7i64).to_le_bytes().to_vec(),
    ]
    .concat();
    let inits = [
        ext_tensor("f", &[3], 1, "a.bin", 4, 12),
        ext_tensor("h", &[3], 16, "b.bin", 0, 6),
        ext_tensor("i", &[1], 7, "b.bin", 6, 8),
    ];
    let mut g = decode(&inits, &FT).unwrap();
    let shards = vec![d.shard("a.bin", &a), d.shard("b.bin", &b)];
    resolve_external(&mut g, shards, &lim()).unwrap();
    assert!(matches!(&g.initializers[0].data, TensorData::F32(v) if v == &[1.5, -2.0, 3.25]));
    assert!(
        matches!(&g.initializers[1].data, TensorData::Bf16(v) if v == &[0x3f80, 0x4000, 0xc040])
    );
    assert!(matches!(&g.initializers[2].data, TensorData::I64(v) if v == &[-7]));
}

#[test]
fn req41_external_decoding_is_chunk_boundary_independent() {
    // 64 KiB を超える shard で、要素がチャンク境界をまたいでも値が一致する
    let d = Dir::new();
    let n = 40_000usize;
    let vals: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let bytes = f32s(&vals);
    let l = ExternalDataLimits {
        max_shard_bytes: 1 << 20,
        max_shards: 4,
        max_total_bytes: 1 << 20,
    };
    let t = ext_tensor("w", &[n as u64], 1, "w.bin", 0, (n * 4) as u64);
    let mut g = decode(&[t], &FT).unwrap();
    resolve_external(&mut g, vec![d.shard("w.bin", &bytes)], &l).unwrap();
    assert!(matches!(&g.initializers[0].data, TensorData::F32(v) if v == &vals));
}

#[test]
fn req39_external_sha256_mismatch_is_integrity_mismatch() {
    let d = Dir::new();
    let mut s = d.shard("a.bin", &f32s(&[1.0, 2.0]));
    s.sha256 = Sha256Digest::of_bytes(b"other");
    let mut g = decode(&[ext_tensor("w", &[2], 1, "a.bin", 0, 8)], &FT).unwrap();
    assert_eq!(
        code(resolve_external(&mut g, vec![s], &lim())),
        "integrity_mismatch"
    );
}

#[test]
fn req39_external_location_escape_rejected() {
    let long = "x".repeat(129);
    for loc in ["../x", "/abs", "a/b", "", ".hidden", "a\\b", long.as_str()] {
        let d = Dir::new();
        // 名前の長さ上限（128 B）超過は復号の段階で拒否される
        let Ok(mut g) = decode(&[ext_tensor("w", &[1], 1, loc, 0, 4)], &FT) else {
            assert!(loc.len() > 128);
            continue;
        };
        let mut s = d.shard("ok.bin", &f32s(&[1.0]));
        s.location = loc.into();
        assert_eq!(
            code(resolve_external(&mut g, vec![s], &lim())),
            "unsupported_tensor",
            "{loc}"
        );
    }
}

#[test]
fn req39_external_location_set_mismatch_rejected() {
    let d = Dir::new();
    let inits = [ext_tensor("w", &[1], 1, "a.bin", 0, 4)];
    let bytes = f32s(&[1.0]);
    // 参照先が渡されていない
    let mut g = decode(&inits, &FT).unwrap();
    assert_eq!(
        code(resolve_external(&mut g, vec![], &lim())),
        "unsupported_tensor"
    );
    // 余分な shard
    let mut g = decode(&inits, &FT).unwrap();
    let v = vec![d.shard("a.bin", &bytes), d.shard("b.bin", &bytes)];
    assert_eq!(
        code(resolve_external(&mut g, v, &lim())),
        "unsupported_tensor"
    );
    // 重複
    let mut g = decode(&inits, &FT).unwrap();
    let v = vec![d.shard("a.bin", &bytes), d.shard("a.bin", &bytes)];
    assert_eq!(
        code(resolve_external(&mut g, v, &lim())),
        "unsupported_tensor"
    );
}

#[test]
fn req39_external_range_rejected() {
    let d = Dir::new();
    let bytes = f32s(&[1.0, 2.0, 3.0, 4.0]); // 16 B
    let run = |inits: &[Vec<u8>]| {
        let mut g = decode(inits, &FT).unwrap();
        code(resolve_external(
            &mut g,
            vec![d.shard("a.bin", &bytes)],
            &lim(),
        ))
    };
    // 末尾超過
    assert_eq!(
        run(&[ext_tensor("w", &[2], 1, "a.bin", 12, 8)]),
        "unsupported_tensor"
    );
    // offset + length の桁あふれ
    let t = ext_tensor("w", &[2], 1, "a.bin", u64::MAX - 3, 8);
    assert_eq!(run(&[t]), "unsupported_tensor");
    // 範囲の重なり
    let inits = [
        ext_tensor("p", &[2], 1, "a.bin", 0, 8),
        ext_tensor("q", &[2], 1, "a.bin", 4, 8),
    ];
    assert_eq!(run(&inits), "unsupported_tensor");
}

#[test]
fn req39_external_malformed_reference_rejected_at_decode() {
    // エントリ数上限（3）に当たる前に目的の検証へ届くよう、location・length に追加 1 件の計 3 件で組む
    let build = |extra: (&str, &str)| {
        [
            field_varint(1, 1),
            field_varint(2, 1),
            entry("location", "a.bin"),
            entry("length", "4"),
            entry(extra.0, extra.1),
            field_varint(14, 1),
        ]
        .concat()
    };
    let with = |extra: (&str, &str)| decode(&[build(extra)], &FT).unwrap_err().code();
    // length が次元と合わない
    assert!(decode(&[ext_tensor("w", &[2], 1, "a.bin", 0, 7)], &FT).is_err());
    // 未知の key（checksum）
    assert_eq!(with(("checksum", "00")), "unsupported_tensor");
    // 重複 key
    assert_eq!(with(("length", "4")), "unsupported_tensor");
    // 非数の offset
    assert_eq!(with(("offset", "-1")), "unsupported_tensor");
    // 正常系（同じ組み立てで offset が数なら通る）
    assert!(decode(&[build(("offset", "0"))], &FT).is_ok());
    // エントリ数上限超過は別経路で拒否される
    let base = ext_tensor("w", &[1], 1, "a.bin", 0, 4);
    assert_ne!(
        decode(&[[base.clone(), entry("checksum", "00")].concat()], &FT)
            .unwrap_err()
            .code(),
        "unsupported_tensor"
    );
    // raw_data との併用
    assert!(decode(&[[base.clone(), field_len(9, &[0; 4])].concat()], &FT).is_err());
    // data_location だけ
    let t = [field_varint(1, 1), field_varint(2, 1), field_varint(14, 1)].concat();
    assert!(decode(&[t], &FT).is_err());
    // external_data だけ（data_location 無し）
    let t = [
        field_varint(1, 1),
        field_varint(2, 1),
        entry("location", "a.bin"),
        entry("length", "4"),
    ]
    .concat();
    assert!(decode(&[t], &FT).is_err());
}

#[test]
fn req39_external_shard_over_limit_is_limit_exceeded() {
    let d = Dir::new();
    let inits = [ext_tensor("w", &[1], 1, "a.bin", 0, 4)];
    let l = |m| ExternalDataLimits {
        max_shard_bytes: m,
        max_shards: 4,
        max_total_bytes: 1 << 20,
    };
    let bytes = vec![0u8; 100];
    // ちょうどは通り、+1 は拒否
    let mut g = decode(&inits, &FT).unwrap();
    resolve_external(&mut g, vec![d.shard("a.bin", &bytes)], &l(100)).unwrap();
    let mut g = decode(&inits, &FT).unwrap();
    let r = resolve_external(&mut g, vec![d.shard("a.bin", &bytes)], &l(99));
    assert_eq!(code(r), "limit_exceeded");
}

#[test]
fn req39_external_total_and_count_over_limit_is_limit_exceeded() {
    let d = Dir::new();
    let inits = [
        ext_tensor("p", &[1], 1, "a.bin", 0, 4),
        ext_tensor("q", &[1], 1, "b.bin", 0, 4),
    ];
    let bytes = vec![0u8; 100];
    let shards = || vec![d.shard("a.bin", &bytes), d.shard("b.bin", &bytes)];
    let total = |t| ExternalDataLimits {
        max_shard_bytes: 100,
        max_shards: 4,
        max_total_bytes: t,
    };
    let mut g = decode(&inits, &FT).unwrap();
    resolve_external(&mut g, shards(), &total(200)).unwrap();
    let mut g = decode(&inits, &FT).unwrap();
    assert_eq!(
        code(resolve_external(&mut g, shards(), &total(199))),
        "limit_exceeded"
    );
    let mut g = decode(&inits, &FT).unwrap();
    let count = ExternalDataLimits {
        max_shards: 1,
        ..total(200)
    };
    assert_eq!(
        code(resolve_external(&mut g, shards(), &count)),
        "limit_exceeded"
    );
}

#[test]
fn req39_model_limits_by_kind() {
    use super::super::{MAX_MODEL_FILE_BYTES, ModelKind};
    assert_eq!(ModelKind::C1.limits(), &STRICT_MODEL_LIMITS);
    assert_eq!(ModelKind::C3.limits(), &STRICT_MODEL_LIMITS);
    assert_eq!(
        STRICT_MODEL_LIMITS.max_model_file_bytes,
        MAX_MODEL_FILE_BYTES
    );
    assert!(STRICT_MODEL_LIMITS.external.is_none() && !STRICT_MODEL_LIMITS.allow_bf16);
    let ext = FT.external.unwrap();
    assert_eq!(
        (ext.max_shard_bytes, ext.max_shards, ext.max_total_bytes),
        (1 << 30, 256, 8 << 30)
    );
}
