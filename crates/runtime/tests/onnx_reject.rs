//! 不正・改変された ONNX モデルの拒否テスト（REQ-39「形式・完全性の検査」・REQ-32・TASK-32.1-2・#113）。
//!
//! 証拠種別: テストハーネス。コミット済みの正常な C1・C3（`fixtures/onnx_parity/`）を
//! `common/proto_builder.rs`（復号器とは独立に書いた protobuf 編集器）で 1 か所だけ改変し、
//! 推論ランタイムが fail-closed で拒否する（読み込みに成功しない）ことを、エラーコードの具体値で確認する。
//! 許可リスト方式（テンプレートの完全一致）のため、未知の演算子・順序の違い・属性値の違い・
//! 外部ファイル参照・部分グラフ属性はすべて拒否される。

#[path = "common/proto_builder.rs"]
mod proto_builder;

use fandhe_edge_core::hash::Sha256Digest;
use fandhe_edge_runtime::onnx::{
    MAX_MAX_BYTES, MAX_MODEL_FILE_BYTES, MIN_MAX_BYTES, ModelKind, OnnxBackend, load_pipeline,
};
use proto_builder::{Val, edit_attr, edit_graph, edit_initializer, edit_model, set_op_type};
use std::fs;
use std::path::PathBuf;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("fixtures")
        .join("onnx_parity")
}

fn model_bytes(name: &str) -> Vec<u8> {
    fs::read(fixture_dir().join(name)).expect("fixture read")
}

fn code(bytes: &[u8], kind: ModelKind) -> &'static str {
    match OnnxBackend::from_bytes(bytes, kind) {
        Ok(_) => "ok",
        Err(e) => e.code(),
    }
}

/// 正常なモデルは読み込める（改変テストの前提）。
#[test]
fn req32_baseline_models_load() {
    assert_eq!(code(&model_bytes("c1.onnx"), ModelKind::C1), "ok");
    assert_eq!(code(&model_bytes("c3.onnx"), ModelKind::C3), "ok");
}

/// REQ-39: `kind` と中身が一致しないモデル（C1 を C3 として・C3 を C1 として）は拒否する。
#[test]
fn req39_kind_mismatch_rejected() {
    assert_eq!(
        code(&model_bytes("c1.onnx"), ModelKind::C3),
        "unsupported_graph"
    );
    assert_eq!(
        code(&model_bytes("c3.onnx"), ModelKind::C1),
        "unsupported_graph"
    );
}

/// REQ-39: 未対応の `kind`（autoregressive を含む）は `unsupported_kind`。
#[test]
fn req39_unsupported_kind_rejected() {
    for k in ["autoregressive", "", "C1", "pickle"] {
        let e = ModelKind::parse(k).expect_err("must reject");
        assert_eq!(e.code(), "unsupported_kind", "{k}");
    }
    assert_eq!(ModelKind::parse("c1").ok(), Some(ModelKind::C1));
    assert_eq!(ModelKind::parse("c3").ok(), Some(ModelKind::C3));
}

/// REQ-39: sha256 が期待値と一致しなければ `integrity_mismatch`。一致すれば読み込める。
#[test]
fn req39_sha256_mismatch_rejected() {
    let path = fixture_dir().join("c1.onnx");
    let zero: Sha256Digest = "0".repeat(64).parse().expect("digest");
    let err = OnnxBackend::load_path(&path, ModelKind::C1, &zero).expect_err("must reject");
    assert_eq!(err.code(), "integrity_mismatch");
    let good = Sha256Digest::of_bytes(&model_bytes("c1.onnx"));
    assert!(OnnxBackend::load_path(&path, ModelKind::C1, &good).is_ok());
}

/// REQ-39: 上限超のファイルは読み込む前にサイズで拒否する（`limit_exceeded`）。
/// 存在しないファイルは読み込みエラー。
#[test]
fn req39_oversized_and_missing_file_rejected() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let big = dir.join("onnx_reject_oversized.bin");
    let f = fs::File::create(&big).expect("create");
    f.set_len(MAX_MODEL_FILE_BYTES + 1).expect("set_len");
    drop(f);
    let digest = Sha256Digest::of_bytes(b"");
    let err = OnnxBackend::load_path(&big, ModelKind::C1, &digest).expect_err("must reject");
    let _ = fs::remove_file(&big);
    assert_eq!(err.code(), "limit_exceeded");

    let missing = dir.join("onnx_reject_missing.bin");
    let err = OnnxBackend::load_path(&missing, ModelKind::C1, &digest).expect_err("must reject");
    assert_eq!(err.code(), "model_file_error");
}

/// REQ-39: 非 ONNX のバイト列・空・途中で切れた protobuf・varint の桁あふれは `malformed_protobuf`。
#[test]
fn req39_malformed_protobuf_rejected() {
    let good = model_bytes("c1.onnx");
    let truncated = good.get(..good.len() / 2).expect("half");
    let overflow = [
        0x08u8, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x02,
    ];
    let cases: [(&str, &[u8]); 4] = [
        ("not_onnx", b"this is definitely not an onnx model"),
        ("empty", b""),
        ("truncated", truncated),
        ("varint_overflow", &overflow),
    ];
    for (name, bytes) in cases {
        assert_eq!(code(bytes, ModelKind::C1), "malformed_protobuf", "{name}");
    }
}

/// REQ-39: `ir_version`・opset が書き出し器の固定値と違えば拒否する。
#[test]
fn req39_version_mismatch_rejected() {
    let good = model_bytes("c1.onnx");
    let ir7 = edit_model(&good, |m| {
        let i = m.iter().position(|(n, _)| *n == 1).expect("ir_version");
        m[i].1 = Val::Varint(7);
    });
    assert_eq!(code(&ir7, ModelKind::C1), "unsupported_ir_version");
    let opset14 = edit_model(&good, |m| {
        let i = m.iter().position(|(n, _)| *n == 8).expect("opset");
        let mut inner = proto_builder::parse(match &m[i].1 {
            Val::Len(b) => b,
            _ => unreachable!(),
        });
        let v = inner.iter().position(|(n, _)| *n == 2).expect("version");
        inner[v].1 = Val::Varint(14);
        m[i].1 = Val::Len(proto_builder::encode(&inner));
    });
    assert_eq!(code(&opset14, ModelKind::C1), "unsupported_opset");
}

/// REQ-39: 未知の演算子・ノード順の入れ替え・属性値の変更は `unsupported_graph`（C1）。
#[test]
fn req39_c1_graph_template_violations_rejected() {
    let good = model_bytes("c1.onnx");
    // 未知の op（最後のノードの op_type を差し替え）
    let unknown_op = set_op_type(&good, 13, "Softmaz");
    assert_eq!(code(&unknown_op, ModelKind::C1), "unsupported_graph");
    // ノード順の入れ替え（Log と Greater）
    let swapped = edit_graph(&good, |g| {
        let idx: Vec<usize> = g
            .iter()
            .enumerate()
            .filter(|(_, (n, _))| *n == 1)
            .map(|(i, _)| i)
            .collect();
        g.swap(idx[2], idx[3]);
    });
    assert_eq!(code(&swapped, ModelKind::C1), "unsupported_graph");
    // mode="TFIDF" 相当（同じ長さ 2 の別値 "TX"）
    let mode = edit_attr(&good, 0, "mode", |a| {
        let i = a.iter().position(|(n, _)| *n == 4).expect("s");
        a[i].1 = Val::Len(b"TX".to_vec());
    });
    assert_eq!(code(&mode, ModelKind::C1), "unsupported_graph");
    // Cast の to を FLOAT(1) 以外へ
    let cast = edit_attr(&good, 4, "to", |a| {
        let i = a.iter().position(|(n, _)| *n == 3).expect("i");
        a[i].1 = Val::Varint(11);
    });
    assert_eq!(code(&cast, ModelKind::C1), "unsupported_graph");
    // min_gram_length = 0
    let min0 = edit_attr(&good, 0, "min_gram_length", |a| {
        let i = a.iter().position(|(n, _)| *n == 3).expect("i");
        a[i].1 = Val::Varint(0);
    });
    assert_eq!(code(&min0, ModelKind::C1), "unsupported_graph");
    // pool_int64s に詰め物 0 を含める
    let pool0 = edit_attr(&good, 0, "pool_int64s", |a| {
        let i = a.iter().position(|(n, _)| *n == 8).expect("ints");
        a[i].1 = Val::Varint(0);
    });
    assert_eq!(code(&pool0, ModelKind::C1), "unsupported_graph");
}

/// REQ-39: C3 のテンプレート違反（偶数カーネル・詰め物行が非ゼロ・未知の op）は拒否する。
#[test]
fn req39_c3_graph_template_violations_rejected() {
    let good = model_bytes("c3.onnx");
    // 7 番目のノードが最初の Conv。kernel_shape を偶数へ
    let even = edit_attr(&good, 7, "kernel_shape", |a| {
        let i = a.iter().position(|(n, _)| *n == 8).expect("ints");
        a[i].1 = Val::Varint(4);
    });
    assert_eq!(code(&even, ModelKind::C3), "unsupported_graph");
    // embed の 0 行目（詰め物）の先頭要素を 1.0f32 へ
    let pad = edit_initializer(&good, "embed", |t| {
        let i = t.iter().position(|(n, _)| *n == 9).expect("raw");
        if let Val::Len(b) = &mut t[i].1 {
            b[..4].copy_from_slice(&1.0f32.to_le_bytes());
        }
    });
    assert_eq!(code(&pad, ModelKind::C3), "unsupported_graph");
    // 未知の op
    let unknown_op = set_op_type(&good, 0, "Gathez");
    assert_eq!(code(&unknown_op, ModelKind::C3), "unsupported_graph");
    // 詰め物を除外できない負値（-0.001）のマスク値は、負数でも書き出し器の固定値（-1e9）と
    // 異なるため拒否する（REQ-28）
    let weak_mask = edit_initializer(&good, "neg_big_f32", |t| {
        let i = t.iter().position(|(n, _)| *n == 9).expect("raw");
        t[i].1 = Val::Len((-0.001f32).to_le_bytes().to_vec());
    });
    assert_eq!(code(&weak_mask, ModelKind::C3), "unsupported_graph");
}

/// REQ-39: テンソルの形式違反（長さ不一致・FLOAT/INT64 以外・外部参照・raw_data 以外の値）は
/// `unsupported_tensor`。
#[test]
fn req39_tensor_violations_rejected() {
    let good = model_bytes("c1.onnx");
    let short = edit_initializer(&good, "bias", |t| {
        let i = t.iter().position(|(n, _)| *n == 9).expect("raw");
        if let Val::Len(b) = &mut t[i].1 {
            b.pop();
        }
    });
    assert_eq!(code(&short, ModelKind::C1), "unsupported_tensor");
    let double = edit_initializer(&good, "bias", |t| {
        let i = t.iter().position(|(n, _)| *n == 2).expect("data_type");
        t[i].1 = Val::Varint(11);
    });
    assert_eq!(code(&double, ModelKind::C1), "unsupported_tensor");
    let external = edit_initializer(&good, "bias", |t| t.push((13, Val::Len(b"kv".to_vec()))));
    assert_eq!(code(&external, ModelKind::C1), "unsupported_tensor");
    let location = edit_initializer(&good, "bias", |t| t.push((14, Val::Varint(1))));
    assert_eq!(code(&location, ModelKind::C1), "unsupported_tensor");
    let float_data = edit_initializer(&good, "bias", |t| t.push((4, Val::Fixed32([0; 4]))));
    assert_eq!(code(&float_data, ModelKind::C1), "unsupported_tensor");
}

/// REQ-39: 部分グラフ（GRAPH 型）を値に持つ属性は拒否する（再帰を作らない）。
#[test]
fn req39_subgraph_attribute_rejected() {
    let good = model_bytes("c1.onnx");
    let with_graph = edit_attr(&good, 13, "axis", |a| a.push((6, Val::Len(Vec::new()))));
    assert_eq!(code(&with_graph, ModelKind::C1), "unsupported_graph");
}

/// REQ-39: `max_bytes` の範囲検証（境界値）。範囲は学習ワーカーの `limits.json` と一致する。
#[test]
fn req39_max_bytes_range_and_limits_fixture() {
    let path = fixture_dir().join("c1.onnx");
    let digest = Sha256Digest::of_bytes(&model_bytes("c1.onnx"));
    for bad in [0usize, MAX_MAX_BYTES + 1, usize::MAX] {
        let err = load_pipeline(ModelKind::C1, bad, &path, &digest)
            .err()
            .expect("must reject");
        assert_eq!(err.code(), "max_bytes_out_of_range", "{bad}");
    }
    for ok in [MIN_MAX_BYTES, MAX_MAX_BYTES] {
        assert!(
            load_pipeline(ModelKind::C1, ok, &path, &digest).is_ok(),
            "{ok}"
        );
    }
    let limits: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("..")
                .join("fixtures")
                .join("train_contract")
                .join("limits.json"),
        )
        .expect("limits read"),
    )
    .expect("limits json");
    assert_eq!(limits["min_max_bytes"].as_u64(), Some(MIN_MAX_BYTES as u64));
    assert_eq!(limits["max_max_bytes"].as_u64(), Some(MAX_MAX_BYTES as u64));
}

/// REQ-39: 語彙範囲外のトークン・空の系列・上限超過の系列はバックエンドが拒否する（panic しない）。
#[test]
fn req39_backend_rejects_invalid_token_sequences() {
    use fandhe_edge_runtime::pipeline::{ScoringBackend, TokenIds};
    for (kind, name) in [(ModelKind::C1, "c1.onnx"), (ModelKind::C3, "c3.onnx")] {
        let backend = OnnxBackend::from_bytes(&model_bytes(name), kind).expect("load");
        let cases: [(&str, Vec<i64>, &str); 4] = [
            ("empty", vec![], "invalid_sequence_length"),
            ("negative", vec![-1], "invalid_token_id"),
            ("too_large", vec![257], "invalid_token_id"),
            ("huge", vec![i64::MAX], "invalid_token_id"),
        ];
        for (label, ids, want) in cases {
            let err = backend
                .scores(&TokenIds::new(ids))
                .expect_err("must reject");
            assert_eq!(err.code(), want, "{name} {label}");
        }
    }
    // C1・C3 とも共通の入口で系列長の上限（MAX_MAX_BYTES）を検査する（境界値は受理）
    for (kind, name) in [(ModelKind::C1, "c1.onnx"), (ModelKind::C3, "c3.onnx")] {
        let backend = OnnxBackend::from_bytes(&model_bytes(name), kind).expect("load");
        let long = TokenIds::new(vec![1; MAX_MAX_BYTES + 1]);
        let err = backend.scores(&long).expect_err("must reject");
        assert_eq!(err.code(), "invalid_sequence_length", "{name} long");
        let at_limit = TokenIds::new(vec![1; MAX_MAX_BYTES]);
        assert!(backend.scores(&at_limit).is_ok(), "{name} at limit");
    }
}
