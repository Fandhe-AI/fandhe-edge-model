//! 最終 test の台帳が候補（重み）を見分ける識別の算出（REQ-27・REQ-39・#662）。
//!
//! # 位置づけ
//!
//! 識別の定義を評価器の 1 か所に集約する。台帳（[`crate::final_test_once`]）は、呼び出し側が申告した
//! 識別を信じず、本モジュールが**ファイルから自ら算出した値**を重みロック・代表構成ロックのキーに使う。
//! 呼び出し側が別の識別を名乗って同じモデルを再適用する、登録後に重みを変えて登録済みの識別を名乗る、
//! といった迂回を公開 API では作れない。CLI の `evaluate`（登録と適用）と `package`（評価済みの照会）は
//! 同じ関数を呼ぶ。
//!
//! # 識別の定義（オーナー決定 2026-10-29）
//!
//! - **byte 表現**（`manifest` が無い、または入力表現が `tokenizer` でない）: ONNX ファイルの sha256
//!   （従来どおり。値は変えない）。
//! - **tokenizer 表現**（`qwen2_lora`）: 領域名 `fandhe-edge/evaluate/weights-identity/v2\0` ‖ ONNX の
//!   sha256（32B）‖ `tokenizer.json` の sha256（32B）‖ shard 件数（u64 LE）‖ 各 shard の sha256（32B。
//!   `artifact.json` の `external_data` の記載順）の sha256。shard の location（ファイル名）は入れない。
//!   ONNX 本体は shard の中身が違っても同じバイト列になりうる（重みの大半は外部データ）ため、ONNX だけでは
//!   shard だけが違う 2 候補が同じ識別になる。
//!
//! 識別は**重みの中身だけ**で決める。`artifact.json` の生バイトは入れない（`created_utc`・来歴・空白だけが
//! 違う同一モデルが別の識別になり、別の構成名で最終 test に 2 回当てられるため）。`artifact.json` の生バイトの
//! 完全一致は、識別とは別に台帳の登録照合（[`crate::final_test_once::RegisteredConfig::with_manifest`]）が見る。
//!
//! # 前提
//!
//! shard の記録 sha256 と実体の一致は、`tokenizer.json` については評価器が `vocab` の sha256 と記録値を
//! 照合し（[`crate::final_test_once`]）、shard についてはランタイムが読みながら行う sha256 照合
//! （不一致は fail-closed）が保証する。評価器は GiB 級の shard を読まない。

use crate::invariance::{MAX_MODEL_COMPONENT_BYTES, ModelPackageBytes, ModelPackagePaths};
use fandhe_edge_core::artifact_meta::{
    ArtifactMeta, MAX_ARTIFACT_META_BYTES, PackageInputRepresentation, TokenizerPackageRefs,
};
use fandhe_edge_core::fs::FsError;
use fandhe_edge_core::hash::Sha256Digest;
use std::fmt;

const WEIGHTS_IDENTITY_DOMAIN: &[u8] = b"fandhe-edge/evaluate/weights-identity/v2\0";

/// 識別を算出できなかった理由。メッセージは英語で、ファイルの中身は含めない。
#[derive(Debug)]
#[non_exhaustive]
pub enum WeightsIdentityError {
    /// 重みまたは `artifact.json` を読めなかった（読み込み失敗・サイズ上限超過・通常ファイルでない）。
    Read(FsError),
    /// `artifact.json` を解析できない、または tokenizer 表現の参照欄（tokenizer・shard の sha256）が不正。
    /// 識別を作れないまま続けない（fail-closed）。
    ManifestInvalid,
}

impl fmt::Display for WeightsIdentityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WeightsIdentityError::Read(source) => {
                write!(f, "failed to read weights identity input: {source}")
            }
            WeightsIdentityError::ManifestInvalid => {
                write!(f, "artifact manifest is invalid for weights identity")
            }
        }
    }
}

impl std::error::Error for WeightsIdentityError {}

/// 算出した識別と、tokenizer 表現のときの `tokenizer.json` の記録 sha256。
pub(crate) struct ResolvedIdentity {
    pub(crate) digest: Sha256Digest,
    /// tokenizer 表現のときだけ `Some`（`vocab` の sha256 との照合に使う）。
    pub(crate) tokenizer_sha256: Option<Sha256Digest>,
}

/// ONNX の sha256 と `artifact.json` のバイト列（あれば）から識別を求める（定義は本モジュールの 1 か所）。
pub(crate) fn resolve(
    onnx: Sha256Digest,
    manifest: Option<&[u8]>,
) -> Result<ResolvedIdentity, WeightsIdentityError> {
    let Some(manifest) = manifest else {
        return Ok(ResolvedIdentity {
            digest: onnx,
            tokenizer_sha256: None,
        });
    };
    let meta = ArtifactMeta::parse(manifest).map_err(|_| WeightsIdentityError::ManifestInvalid)?;
    if meta.input_representation() != PackageInputRepresentation::Tokenizer {
        return Ok(ResolvedIdentity {
            digest: onnx,
            tokenizer_sha256: None,
        });
    }
    let bad = || WeightsIdentityError::ManifestInvalid;
    let refs = TokenizerPackageRefs::parse(manifest).map_err(|_| bad())?;
    let tokenizer: Sha256Digest = refs.tokenizer_sha256.parse().map_err(|_| bad())?;
    let shards = refs
        .external_data
        .iter()
        .map(|e| e.sha256.parse::<Sha256Digest>().map_err(|_| bad()))
        .collect::<Result<Vec<_>, _>>()?;
    let mut buf = Vec::with_capacity(WEIGHTS_IDENTITY_DOMAIN.len() + 72 + 32 * shards.len());
    buf.extend_from_slice(WEIGHTS_IDENTITY_DOMAIN);
    buf.extend_from_slice(onnx.as_bytes());
    buf.extend_from_slice(tokenizer.as_bytes());
    // 件数を固定長で先に置くので、境界のずらしで別の組が同じ入力にならない。
    buf.extend_from_slice(
        &u64::try_from(shards.len())
            .unwrap_or(u64::MAX)
            .to_le_bytes(),
    );
    for shard in &shards {
        buf.extend_from_slice(shard.as_bytes());
    }
    Ok(ResolvedIdentity {
        digest: Sha256Digest::of_bytes(&buf),
        tokenizer_sha256: Some(tokenizer),
    })
}

/// メモリ上の ONNX と `artifact.json` から識別を求める。`manifest` が `None`、または解析した入力表現が
/// tokenizer でなければ ONNX の sha256（byte 表現。従来どおり）。
///
/// CLI の `evaluate` が全候補の事前登録に使い、`package` が評価済みの照会に使う（REQ-27・#662）。
///
/// # Errors
/// `manifest` を解析できない、または tokenizer 表現の参照欄が不正（[`WeightsIdentityError::ManifestInvalid`]）。
pub fn weights_identity_from_bytes(
    onnx: &[u8],
    manifest: Option<&[u8]>,
) -> Result<Sha256Digest, WeightsIdentityError> {
    resolve(Sha256Digest::of_bytes(onnx), manifest).map(|r| r.digest)
}

/// `paths` が指すファイルから識別を求める（台帳の適用が使う。ファイルを自ら読む）。
///
/// ONNX は [`MAX_MODEL_COMPONENT_BYTES`] の上限付きでストリームハッシュし、`artifact.json` は
/// `MAX_ARTIFACT_META_BYTES` の上限付きで読む。
///
/// # Errors
/// ファイルを読めない（[`WeightsIdentityError::Read`]）、または [`weights_identity_from_bytes`] と同じ。
pub fn weights_identity_from_paths(
    paths: &ModelPackagePaths<'_>,
) -> Result<Sha256Digest, WeightsIdentityError> {
    resolve_from_paths(paths).map(|r| r.digest)
}

pub(crate) fn resolve_from_paths(
    paths: &ModelPackagePaths<'_>,
) -> Result<ResolvedIdentity, WeightsIdentityError> {
    let onnx = fandhe_edge_core::fs::sha256_file_bounded(paths.weights, MAX_MODEL_COMPONENT_BYTES)
        .map_err(WeightsIdentityError::Read)?;
    let manifest = paths
        .manifest
        .map(|p| fandhe_edge_core::fs::read_bounded(p, MAX_ARTIFACT_META_BYTES))
        .transpose()
        .map_err(WeightsIdentityError::Read)?;
    resolve(onnx, manifest.as_deref())
}

pub(crate) fn resolve_from_bytes(
    package: &ModelPackageBytes<'_>,
) -> Result<Option<ResolvedIdentity>, WeightsIdentityError> {
    let Some(weights) = package.weights else {
        return Ok(None);
    };
    resolve(Sha256Digest::of_bytes(weights), package.manifest).map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    const H1: &str = "1111111111111111111111111111111111111111111111111111111111111111";
    const H2: &str = "2222222222222222222222222222222222222222222222222222222222222222";
    const H3: &str = "3333333333333333333333333333333333333333333333333333333333333333";

    fn byte_manifest(kind: &str) -> String {
        let z = "0".repeat(64);
        format!(
            r#"{{"kind":"{kind}","kind_version":1,"max_bytes":64,"label_order":["a","b"],"onnx_file":"model.onnx","onnx_sha256":"{z}"}}"#
        )
    }

    fn tok_meta_json(created_utc: &str, shards: &[&str], tokenizer: &str, prompt: &str) -> String {
        let z = "0".repeat(64);
        let ext = shards
            .iter()
            .enumerate()
            .map(|(i, h)| format!(r#"{{"location":"w{i}.bin","sha256":"{h}"}}"#))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            r#"{{"kind":"qwen2_lora","kind_version":1,"max_bytes":64,"label_order":["a","b"],"onnx_file":"model.onnx","onnx_sha256":"{z}","input_representation":"tokenizer","created_utc":"{created_utc}","tokenizer_file":"tokenizer.json","tokenizer_sha256":"{tokenizer}","external_data":[{ext}],"scoring":{{"system_prompt":"{prompt}","choices":[{{"token_ids":[1]}},{{"token_ids":[2]}}]}},"config":{{"max_seq_length":64}}}}"#
        )
    }

    fn identity_of(json: &str) -> Sha256Digest {
        weights_identity_from_bytes(b"onnx", Some(json.as_bytes())).expect("identity")
    }

    /// REQ-27・#662: manifest が無い、または byte 表現の manifest では ONNX の sha256 のまま（従来の値）。
    #[test]
    fn req27_issue662_byte_identity_is_the_onnx_sha256() {
        let onnx = Sha256Digest::of_bytes(b"onnx");
        assert_eq!(weights_identity_from_bytes(b"onnx", None).unwrap(), onnx);
        for kind in ["c1", "c3"] {
            let manifest = byte_manifest(kind);
            assert_eq!(
                weights_identity_from_bytes(b"onnx", Some(manifest.as_bytes())).unwrap(),
                onnx,
                "{kind}"
            );
        }
    }

    /// REQ-27・#662 (f): 推論と無関係な欄（`created_utc`）・空白だけが違う `artifact.json` は同じ識別になる。
    #[test]
    fn req27_issue662_identity_ignores_unrelated_fields_and_whitespace() {
        let a = tok_meta_json("2026-10-01T00:00:00Z", &[H1, H2], H3, "p");
        let b = tok_meta_json("2026-12-31T23:59:59Z", &[H1, H2], H3, "p");
        let spaced = a.replace(',', " ,\n ");
        assert_ne!(a, b);
        assert_ne!(a, spaced);
        assert_eq!(identity_of(&a), identity_of(&b));
        assert_eq!(identity_of(&a), identity_of(&spaced));
        assert_ne!(identity_of(&a), Sha256Digest::of_bytes(b"onnx"));
    }

    /// REQ-27・#662: 識別の正準入力は領域名 ‖ ONNX ‖ tokenizer ‖ 件数(u64 LE) ‖ 各 shard。独立に組み立てた
    /// 値と一致する（定義の固定）。
    #[test]
    fn req27_issue662_identity_matches_the_v2_definition() {
        let json = tok_meta_json("t", &[H1, H2], H3, "p");
        let digest = |h: &str| h.parse::<Sha256Digest>().unwrap();
        let mut buf = b"fandhe-edge/evaluate/weights-identity/v2\0".to_vec();
        buf.extend_from_slice(Sha256Digest::of_bytes(b"onnx").as_bytes());
        buf.extend_from_slice(digest(H3).as_bytes());
        buf.extend_from_slice(&2u64.to_le_bytes());
        buf.extend_from_slice(digest(H1).as_bytes());
        buf.extend_from_slice(digest(H2).as_bytes());
        assert_eq!(identity_of(&json), Sha256Digest::of_bytes(&buf));
    }

    #[test]
    fn req27_issue662_identity_changes_with_shard_tokenizer_order_and_onnx() {
        let base = identity_of(&tok_meta_json("t", &[H1, H2], H3, "p"));
        assert_ne!(base, identity_of(&tok_meta_json("t", &[H1, H3], H3, "p")));
        assert_ne!(base, identity_of(&tok_meta_json("t", &[H1, H2], H1, "p")));
        assert_ne!(base, identity_of(&tok_meta_json("t", &[H2, H1], H3, "p")));
        let json = tok_meta_json("t", &[H1, H2], H3, "p");
        assert_ne!(
            base,
            weights_identity_from_bytes(b"onnx2", Some(json.as_bytes())).unwrap()
        );
        // system プロンプトだけが違う同一重みは同じ識別（評価済みとして断られる安全側）。
        assert_eq!(
            base,
            identity_of(&tok_meta_json("t", &[H1, H2], H3, "other"))
        );
    }

    /// REQ-27・#662: 解析できない manifest・参照欄が不正な tokenizer 表現は識別を作らない（fail-closed）。
    #[test]
    fn req27_issue662_unparsable_manifest_is_an_error() {
        for bad in ["not json", "{}"] {
            assert!(matches!(
                weights_identity_from_bytes(b"onnx", Some(bad.as_bytes())),
                Err(WeightsIdentityError::ManifestInvalid)
            ));
        }
        let no_refs = byte_manifest("qwen2_lora").replace(
            r#""onnx_file""#,
            r#""input_representation":"tokenizer","onnx_file""#,
        );
        assert!(matches!(
            weights_identity_from_bytes(b"onnx", Some(no_refs.as_bytes())),
            Err(WeightsIdentityError::ManifestInvalid)
        ));
    }

    /// REQ-27・#662: fixture の `qwen2_lora` 成果物で、`model.onnx` が同じまま shard の sha256 だけが違う候補は
    /// 別の識別になり、パス版はバイト列版と同じ値になる。
    #[test]
    fn req27_issue662_fixture_candidates_differing_only_in_shard_get_distinct_identities() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/qwen2_parity");
        let onnx = std::fs::read(dir.join("model.onnx")).expect("onnx");
        let meta = std::fs::read_to_string(dir.join("artifact.json")).expect("meta");
        let shard =
            Sha256Digest::of_bytes(&std::fs::read(dir.join("weights-00000.bin")).expect("shard"))
                .to_hex();
        assert!(meta.contains(&shard), "shard sha256 is recorded");
        let other = meta.replace(&shard, &"1".repeat(64));
        assert_ne!(meta, other);
        let id = |json: &str| weights_identity_from_bytes(&onnx, Some(json.as_bytes())).unwrap();
        assert_ne!(id(&meta), id(&other));
        assert_ne!(id(&meta), Sha256Digest::of_bytes(&onnx));
        // パス版: ファイルから読んでもバイト列版と同じ値。
        let paths = ModelPackagePaths {
            weights: &dir.join("model.onnx"),
            vocab: None,
            calibration: None,
            thresholds: None,
            manifest: Some(&dir.join("artifact.json")),
        };
        assert_eq!(weights_identity_from_paths(&paths).unwrap(), id(&meta));
    }
}
