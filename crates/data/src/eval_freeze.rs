//! 評価データの凍結記録と、評価データなしの境界動作（REQ-17・TASK-17.2-1）。
//!
//! # 呼び出し文脈
//!
//! 将来の CLI `evaluate` 工程（REQ-33・TASK-33.3・issue #140）から呼ばれる想定。
//! CLI はガード層（経路の閉じ込め・読み込み前のサイズ上限。REQ-39。issue
//! #157/#158・#172。いずれも未実装）を通した上で評価データ本体を 1 回だけ
//! 読み込み、そのバイト列を [`freeze_eval_data`] へ渡す。同じバイト列を
//! 検査（[`crate::inspect`]）・評価にも使うことで、ハッシュを取った時点と
//! 評価する時点の間に差し替えの余地を作らない（REQ-27: 評価の前後でモデルと
//! 評価データのハッシュが一致すること）。
//!
//! # 責務の境界（本モジュールが行わないこと）
//!
//! - **評価データ本体（大容量）の読み込み・経路の閉じ込め・サイズ上限検査は
//!   行わない**（crate 全体の前提条件。`crates/data/src/lib.rs`）。
//!   [`freeze_eval_data`] はパスもリーダーも受け取らず、既に読み込まれた
//!   `&[u8]` だけを受け取る。無制限に読み込む経路をそもそも持たない。
//!   一方、[`FreezeRecord`] 自体（凍結記録 JSON。数十〜百数十バイト程度の
//!   小さい構造）を外部ファイルから読み戻す経路は本モジュールが提供する
//!   （[`FreezeRecord::load`]・[`FreezeRecord::parse`]。REQ-39 の項を参照）
//! - **[`evaluate_gate`] は「凍結記録が実データ由来である」ことを、渡された
//!   実データバイト列から改めて sha256・バイト長を計算し直して確認する
//!   （fail-closed。`.claude/rules/evaluation-contract.md`「データの分割と
//!   凍結」）。[`FreezeRecord`] は [`Deserialize`] を実装するため、形式上
//!   正しいが実データとは無関係なハッシュ値を詰めた記録を組み立てて
//!   [`EvalDataState::Frozen`] に渡すことができてしまう（PR #209 レビュー
//!   指摘）。[`evaluate_gate`] はそれを無条件に信用せず、常に渡された実データ
//!   から再計算した値と突き合わせ、不一致なら [`FreezeError::HashMismatch`]
//!   で拒否する。同様に [`EvalDataState::NotProvided`] という状態フラグも
//!   単独では信用せず、渡された実データバイト列（`actual_bytes`）が非空
//!   なら状態と実データの不整合として [`FreezeError::NotProvidedButDataPresent`]
//!   で拒否する（PR #209 レビュー指摘）
//! - **凍結記録の“来歴”（過去に記録した台帳との突き合わせ・版管理）は
//!   実装しない**（REQ-17・TASK-17.3・issue #49 の対象）。[`EvalDataState`]
//!   は `#[non_exhaustive]` にしてあり、#49 が台帳との突き合わせ結果を表す
//!   variant を追加できる
//! - **読み取り専用配置・書き込み拒否は本モジュールの対象外**（[`crate::frozen_placement`]
//!   が担う。REQ-39・TASK-17.2-2・issue #48）
//! - **CLI の出力 JSON 全体の形（`step`・`reason` 等のキー構成）は決めない**。
//!   本モジュールが固定するのは「`"skipped"` という語彙」と「exit 0 への
//!   対応」だけで、それ以外の入出力契約は TASK-33.2/33.3 に委ねる
//!
//! # 評価データなしの境界（PoC-16 縦断 2 相当）
//!
//! 評価データが渡されなかったとき、`evaluate` は評価済みを装わず
//! `status:"skipped"`・exit 0 で完走しなければならない
//! （`.claude/rules/evaluation-contract.md`「データの分割と凍結」）。
//! [`EvalDataState::NotProvided`] → [`evaluate_gate`] → [`EvaluateGate::Skip`]
//! → [`EvalStatus::Skipped`]（`exit_code() == ExitCode::Ok`）という型の連なりで
//! この経路を固定する。[`EvalStatus`] は現時点で `Skipped` の 1 variant しか
//! 持たず、指標・合否のフィールドを持たないため「評価済みを装う」ことが
//! 型として不可能になっている。
//!
//! **空のバイト列は「評価データなし」とみなさない**。利用者がファイルを渡した
//! 意思を黙って skipped にすり替えないため、`freeze_eval_data(b"")` は
//! 正常に [`FreezeRecord`]（`byte_len: 0`）を返し、[`EvalDataState::Frozen`]
//! として扱う。件数 0 件（レコードが 1 件も無い）の扱いは検査・評価器の責務
//! （REQ-26 の「分母 0 は null」）で、本モジュールの対象外。
//!
//! # セキュリティ上の注意
//!
//! [`FreezeError`]・[`EvalStatus`] の `Display`/JSON 表現は英語固定で、評価
//! データ本文・パスを含めない（`.claude/rules/security.md`）。[`FreezeRecord`]
//! 自体もパスやファイル名を持たない（どのファイルに対応するかは、呼び出し側の
//! プロジェクト台帳〔TASK-33.x〕が記録の外で管理する）。
//!
//! **[`FreezeRecord`] の `Deserialize` 実装は JSON ドキュメント全体の
//! サイズを制限しない**（`fandhe_edge_core::hash::Sha256Digest` の
//! `Deserialize` 実装の doc「残存リスク」を参照。`serde_json` はエスケープを
//! 含む文字列を展開してから `visit_str` を呼ぶため、64 文字の検査より前に
//! 入力長に比例したメモリを確保しうる）。外部ファイル・外部文字列から
//! 凍結記録を読み戻す経路は、`serde_json::from_str::<FreezeRecord>` を直接
//! 呼ばず、必ず [`FreezeRecord::load`]・[`FreezeRecord::parse`] を使うこと。
//! どちらも `serde_json` へ渡す前に総バイト数を [`MAX_FREEZE_RECORD_JSON_BYTES`]
//! と照合し、超過分は fail-closed で拒否する（REQ-39・REQ-17）。

use fandhe_edge_core::exitcode::ExitCode;
use fandhe_edge_core::fs::{FsError, read_bounded};
use fandhe_edge_core::hash::Sha256Digest;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::Path;

/// 評価データの凍結に使うハッシュ方式。現時点では sha256 のみ。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum HashAlgorithm {
    Sha256,
}

/// 評価データ本体を凍結した記録（REQ-17）。
///
/// パスや名前を持たない（モジュール doc 参照）。フィールドは非公開にし、
/// getter で読ませる: 記録は [`freeze_eval_data`] を通してしか作れず、
/// 外部から任意のハッシュ値を詰めた `FreezeRecord` を組み立てられない
/// （検証済みの入力から作った記録という保証。JSON からの逆直列化
/// （[`Deserialize`]）は例外的に許すが、`Sha256Digest` の `FromStr` 経由の
/// 検証を通るため fail-closed）。
///
/// JSON の形は `{"algorithm":"sha256","sha256":"<64hex>","byte_len":N}` に
/// 固定する（`#[serde(deny_unknown_fields)]` で未知キーを拒否）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FreezeRecord {
    algorithm: HashAlgorithm,
    sha256: Sha256Digest,
    byte_len: u64,
}

impl FreezeRecord {
    /// ハッシュ方式。
    #[must_use]
    pub fn algorithm(&self) -> HashAlgorithm {
        self.algorithm
    }

    /// 評価データ本体の sha256。
    #[must_use]
    pub fn sha256(&self) -> Sha256Digest {
        self.sha256
    }

    /// 評価データ本体のバイト長。
    #[must_use]
    pub fn byte_len(&self) -> u64 {
        self.byte_len
    }
}

/// [`FreezeRecord`] の JSON 表現が取りうる最大バイト数（REQ-39・REQ-17）。
///
/// `{"algorithm":"sha256","sha256":"<64hex>","byte_len":<u64>}` の厳密な形式
/// （[`FreezeRecord`] の doc 参照）は、`byte_len` が `u64::MAX`
/// （`18446744073709551615`。20 桁）のときに最大 130 バイト程度になる
/// （`printf '%s' '{"algorithm":"sha256","sha256":"'"$(printf 'f%.0s' {1..64})"'","byte_len":18446744073709551615}' | wc -c`
/// で独立に確認済み。証拠の種別: テストハーネス）。1 KiB
/// （`MAX_FREEZE_RECORD_JSON_BYTES`）は、この最大値の 7 倍以上の余裕
/// （手書き・pretty-print 時の空白混入を許容する余地）を持たせつつ、
/// 攻撃者が送り込みうる巨大なエスケープ済み JSON 文字列
/// （`fandhe_edge_core::hash::Sha256Digest` の `Deserialize` doc「残存
/// リスク」参照）に対しては十分に小さい上限として選んだ値。
pub const MAX_FREEZE_RECORD_JSON_BYTES: u64 = 1024;

/// [`FreezeRecord::load`]・[`FreezeRecord::parse`] が失敗する理由。
#[derive(Debug)]
#[non_exhaustive]
pub enum LoadFreezeRecordError {
    /// JSON 表現のバイト数が [`MAX_FREEZE_RECORD_JSON_BYTES`] を超えていた
    /// （`serde_json` へ渡す前に検出。REQ-39）。
    TooLarge { size: u64, limit: u64 },
    /// ファイルの読み込みに失敗した（[`FreezeRecord::load`] のみ）。
    Fs(FsError),
    /// JSON として妥当だが [`FreezeRecord`] の形式（キー構成・`Sha256Digest`
    /// の 64 桁小文字 16 進・`HashAlgorithm` の既知値）に一致しなかった、
    /// または JSON 構文として不正だった。
    Parse(serde_json::Error),
}

impl fmt::Display for LoadFreezeRecordError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadFreezeRecordError::TooLarge { size, limit } => write!(
                f,
                "frozen record JSON exceeds size limit ({size} > {limit} bytes)"
            ),
            LoadFreezeRecordError::Fs(source) => write!(f, "{source}"),
            LoadFreezeRecordError::Parse(source) => {
                write!(f, "failed to parse frozen record JSON: {source}")
            }
        }
    }
}

impl std::error::Error for LoadFreezeRecordError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LoadFreezeRecordError::TooLarge { .. } => None,
            LoadFreezeRecordError::Fs(source) => Some(source),
            LoadFreezeRecordError::Parse(source) => Some(source),
        }
    }
}

impl From<FsError> for LoadFreezeRecordError {
    fn from(source: FsError) -> Self {
        LoadFreezeRecordError::Fs(source)
    }
}

impl FreezeRecord {
    /// 凍結記録の JSON 文字列を、パース前にバイト数を検証してから読み込む
    /// （REQ-39・REQ-17）。
    ///
    /// `serde_json::from_str::<FreezeRecord>(json)` を直接呼ぶ経路は、
    /// `FreezeRecord` の `Deserialize`（`Sha256Digest` の `Deserialize` を
    /// 経由）だけでは JSON ドキュメント全体のサイズを制限できない
    /// （モジュール doc「セキュリティ上の注意」参照）。本関数は
    /// `json.len()`（バイト長・`O(1)`）を [`MAX_FREEZE_RECORD_JSON_BYTES`] と
    /// 比較し、超過していれば `serde_json` を一切呼ばずに
    /// [`LoadFreezeRecordError::TooLarge`] を返す。
    pub fn parse(json: &str) -> Result<Self, LoadFreezeRecordError> {
        let size = json.len() as u64;
        if size > MAX_FREEZE_RECORD_JSON_BYTES {
            return Err(LoadFreezeRecordError::TooLarge {
                size,
                limit: MAX_FREEZE_RECORD_JSON_BYTES,
            });
        }
        serde_json::from_str(json).map_err(LoadFreezeRecordError::Parse)
    }

    /// パスから凍結記録の JSON を読み込む（REQ-39・REQ-17）。
    ///
    /// [`fandhe_edge_core::fs::read_bounded`] で通常ファイル判定・TOCTOU 対策
    /// （サイズ確認後のファイル拡大・差し替え）・FIFO 等での無期限停止の
    /// 回避を経た上で、[`MAX_FREEZE_RECORD_JSON_BYTES`] バイトまでの上限で
    /// 読み込む。読み込んだバイト列は UTF-8 として検証してから
    /// [`FreezeRecord::parse`] へ渡す（`parse` 側の長さ検査は
    /// `read_bounded` の上限検証と同じ定数を参照するため冗長だが、
    /// `parse` を単独で呼ぶ経路（文字列を直接渡す呼び出し）でも上限が
    /// 効くことを保証するために残す）。
    pub fn load(path: &Path) -> Result<Self, LoadFreezeRecordError> {
        let bytes = read_bounded(path, MAX_FREEZE_RECORD_JSON_BYTES)?;
        let text = String::from_utf8(bytes).map_err(|err| {
            LoadFreezeRecordError::Fs(FsError::Read {
                path: path.to_path_buf(),
                source: std::io::Error::new(std::io::ErrorKind::InvalidData, err),
            })
        })?;
        Self::parse(&text)
    }
}

/// [`freeze_eval_data`] が失敗する理由。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FreezeError {
    /// バイト列の長さが `u64` の範囲を超えていた。
    ///
    /// 64bit 環境では実質的に到達しないが（`usize` が `u64` に収まる）、
    /// `as` キャストを使わず `u64::try_from` で明示的に変換する規約
    /// （`.claude/rules/coding-rust.md`）に従うために `Result` として扱う。
    LengthOverflow,
    /// [`evaluate_gate`] で、実データから再計算した sha256・バイト長が
    /// [`EvalDataState::Frozen`] に渡された記録と一致しなかった。
    ///
    /// fail-closed の中核（`.claude/rules/evaluation-contract.md`
    /// 「データの分割と凍結」: 「ハッシュが記録と一致しなければ処理を
    /// 停止する」）。呼び出し側（将来の CLI `evaluate` 工程）はこの
    /// variant を受け取ったら評価を進めず、非ゼロ終了で停止すること
    /// （具体的な終了コードへの対応付けは TASK-33.3 に委ねる）。
    HashMismatch,
    /// [`evaluate_gate`] で、状態が [`EvalDataState::NotProvided`]
    /// （評価データなし）なのに、渡された `actual_bytes` が非空だった
    /// （状態フラグと実データの不整合。PR #209 レビュー指摘）。
    ///
    /// この組み合わせを無条件に `Skip`（`status:"skipped"`・exit 0）へ
    /// 通すと、評価データ本体が実際に渡されているにもかかわらず状態
    /// フラグの指定ミスだけで評価済みを装わず終わらせてしまい、
    /// `.claude/rules/evaluation-contract.md`「評価データが無い場合、
    /// `evaluate` は `status:"skipped"`・exit 0 とし、評価済みを装わない」
    /// （評価データが無い場合に限る）に反する。fail-closed で拒否し、
    /// 呼び出し側に状態と実データの矛盾を修正させる。
    NotProvidedButDataPresent,
}

impl fmt::Display for FreezeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FreezeError::LengthOverflow => {
                write!(f, "eval data byte length does not fit in u64")
            }
            FreezeError::HashMismatch => {
                write!(
                    f,
                    "recomputed eval data hash does not match the frozen record"
                )
            }
            FreezeError::NotProvidedButDataPresent => {
                write!(
                    f,
                    "eval data state is not_provided but actual_bytes is non-empty"
                )
            }
        }
    }
}

impl std::error::Error for FreezeError {}

/// 評価データ本体のバイト列を凍結する。
///
/// **パスもリーダーも受け取らない**（`&[u8]` のみ）。理由は 3 つ:
///
/// 1. crate 全体の前提条件（本 crate はファイル読み込み・サイズ上限検査を
///    行わない。読み込み済みの本文を受け取る）に、[`crate::inspect`]・
///    [`crate::eval_input`] と同じ流儀で合わせるため
/// 2. REQ-27（評価の前後で評価データのハッシュが一致すること）を構造で
///    満たすため。呼び出し側（将来の CLI `evaluate`）はガード層を通して
///    1 回だけ読み込んだバッファをハッシュし、**同じバッファ**を検査・評価に
///    渡す。`Read` を受け取る設計だと 2 回読むか tee が必要になり、ハッシュを
///    取った時点と評価する時点の間に差し替えの余地が生まれてしまう
/// 3. 資源上限（読み込み前のサイズ上限。暫定 1GB。REQ-39・issue #172）は
///    ガード層が読み込み前に課す責務であり、本関数の脅威モデルに含めない。
///    本関数は既に読み込まれたバイト列を受け取るため、無制限読み込みの経路を
///    持たない
///
/// `byte_len` はプラットフォームに依存しない記録にするため `u64` で持つ
/// （`u64::try_from(bytes.len())` で変換し、失敗したら
/// [`FreezeError::LengthOverflow`] を返す）。
///
/// **空のバイト列（`bytes.is_empty()`）も凍結する**（「評価データなし」とは
/// 区別する。モジュール doc「評価データなしの境界」参照）。
pub fn freeze_eval_data(bytes: &[u8]) -> Result<FreezeRecord, FreezeError> {
    let byte_len = u64::try_from(bytes.len()).map_err(|_| FreezeError::LengthOverflow)?;
    Ok(FreezeRecord {
        algorithm: HashAlgorithm::Sha256,
        sha256: Sha256Digest::of_bytes(bytes),
        byte_len,
    })
}

/// 評価データの有無・凍結状態（REQ-17）。
///
/// `#[non_exhaustive]` にしてあり、TASK-17.3（issue #49）がハッシュ不一致を
/// 表す `Mismatched` 等の variant を追加できる余地を残す。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum EvalDataState {
    /// 評価データが渡されなかった。
    NotProvided,
    /// 評価データが凍結済み（空入力でもここに含まれる）。
    Frozen(FreezeRecord),
}

/// [`EvalDataState`] から `evaluate` 工程が進めてよいかどうかを判定した結果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvaluateGate {
    /// 評価をスキップする（評価データなし）。
    Skip,
    /// 評価を進める（凍結済みの記録を伴う）。
    Proceed(FreezeRecord),
}

/// [`EvalDataState`] を [`EvaluateGate`] へ写す。
///
/// `NotProvided` は無条件に `Skip` へは写さない: `actual_bytes`
/// （呼び出し側が実際に読み込んだ評価データ本体。`state` を作った際と
/// 同じバッファであることを期待する）が空である場合に限り `Skip` を返す。
/// `Frozen(record)` も無条件に `Proceed` へは写さない: `actual_bytes` から
/// 改めて [`freeze_eval_data`] を計算し直し、`record` と完全一致する場合に
/// 限り `Proceed(record.clone())` を返す。一致しなければ
/// [`FreezeError::HashMismatch`] を返して fail-closed に拒否する
/// （`.claude/rules/evaluation-contract.md`「データの分割と凍結」）。
///
/// [`FreezeRecord`] は [`Deserialize`] を実装するため、形式（64hex の
/// sha256・非負の `byte_len`）だけが正しく値は実データと無関係な記録を
/// 組み立てて `EvalDataState::Frozen` に詰めることができてしまう。
/// この関数はそれを信用せず、`actual_bytes` から独立に再計算した値との
/// 一致を必ず確認する（PR #209 レビュー指摘・issue #47 スレッド）。
///
/// 同様に `state` が `NotProvided` なのに `actual_bytes` が非空である
/// 組み合わせも、状態フラグの指定ミスだけで評価データ本体が渡されている
/// 事実を無視して `status:"skipped"`・exit 0 に通してしまう
/// （評価データが無い場合に限りスキップを許す評価契約に反する）ため、
/// 無条件に信用せず [`FreezeError::NotProvidedButDataPresent`] で
/// fail-closed に拒否する（PR #209 レビュー指摘）。評価データが実際に
/// 渡されなかった場合、呼び出し側は空スライスを渡すこと。
///
/// `#[non_exhaustive]` な [`EvalDataState`] に対する `match` は、将来 #49 が
/// variant を追加した際にこの関数がコンパイルエラーで検出できるよう、
/// ワイルドカードアーム無しで網羅する（本 issue の範囲では `NotProvided`・
/// `Frozen` の 2 種のみ存在するため到達可能）。
pub fn evaluate_gate(
    state: &EvalDataState,
    actual_bytes: &[u8],
) -> Result<EvaluateGate, FreezeError> {
    match state {
        EvalDataState::NotProvided => {
            if actual_bytes.is_empty() {
                Ok(EvaluateGate::Skip)
            } else {
                Err(FreezeError::NotProvidedButDataPresent)
            }
        }
        EvalDataState::Frozen(record) => {
            let recomputed = freeze_eval_data(actual_bytes)?;
            if &recomputed == record {
                Ok(EvaluateGate::Proceed(record.clone()))
            } else {
                Err(FreezeError::HashMismatch)
            }
        }
    }
}

/// `evaluate` 工程の状態（REQ-21・評価契約）。
///
/// 現時点では `Skipped` の 1 variant のみ。指標・合否のフィールドを持たない
/// ため、「評価済みを装う」ことが型として表現できない
/// （`.claude/rules/coding-rust.md`「判定結果・終了コード・状態は enum で表し、
/// 壊れた値を表現できない型にする」）。指標を持つ variant（評価完了）は
/// TASK-33.3 以降、評価器（`fandhe-edge-eval`）の実装が揃った時点で追加する。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum EvalStatus {
    /// 評価データが無いため評価をスキップした（評価済みを装わない）。
    Skipped,
}

impl EvalStatus {
    /// この状態に対応する終了コード（REQ-21）。
    ///
    /// `Skipped` は正常終了（exit 0）として扱う
    /// （`.claude/rules/evaluation-contract.md`「データの分割と凍結」:
    /// 「評価データが無い場合、`evaluate` は `status:"skipped"`・exit 0 とし、
    /// 評価済みを装わない」）。
    #[must_use]
    pub const fn exit_code(self) -> ExitCode {
        match self {
            EvalStatus::Skipped => ExitCode::Ok,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-17: 空入力の凍結記録がゴールデン sha256（既知の空文字列ベクタ）
    /// と `byte_len: 0` になる。
    #[test]
    fn req17_freeze_empty_input() {
        let record = freeze_eval_data(b"").expect("空入力の凍結は失敗しないはず");
        assert_eq!(record.algorithm(), HashAlgorithm::Sha256);
        assert_eq!(
            record.sha256().to_hex(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(record.byte_len(), 0);
    }

    /// REQ-17: 同じ入力を 2 回凍結すると完全に一致する記録になる（決定性）。
    #[test]
    fn req17_freeze_is_deterministic() {
        let bytes = b"hello world";
        let first = freeze_eval_data(bytes).expect("失敗しないはず");
        let second = freeze_eval_data(bytes).expect("失敗しないはず");
        assert_eq!(first, second);
    }

    /// REQ-17: 1 バイトだけ変えると sha256 が変わる（不一致検知の前提。#49）。
    #[test]
    fn req17_freeze_differs_on_single_byte_change() {
        let original = freeze_eval_data(b"farewell").expect("失敗しないはず");
        let changed = freeze_eval_data(b"farewelL").expect("失敗しないはず");
        assert_ne!(original.sha256(), changed.sha256());
        assert_eq!(original.byte_len(), changed.byte_len());
    }

    /// REQ-21: `EvalStatus::Skipped` が exit 0 に対応する。
    #[test]
    fn req21_skipped_maps_to_exit_ok() {
        assert_eq!(EvalStatus::Skipped.exit_code(), ExitCode::Ok);
        assert_eq!(EvalStatus::Skipped.exit_code().code(), 0);
    }

    /// REQ-21: `EvalStatus::Skipped` の JSON 表現が `"skipped"` になる
    /// （語彙の固定。CLI 出力契約の詳細は TASK-33.3 に委ねる）。
    #[test]
    fn req21_skipped_serializes_to_expected_json() {
        let json = serde_json::to_string(&EvalStatus::Skipped).expect("serialize は成功するはず");
        assert_eq!(json, "\"skipped\"");
    }

    /// REQ-17: 評価データなし（`NotProvided`）かつ `actual_bytes` も空の
    /// 場合は `Skip` に写る。
    #[test]
    fn req17_evaluate_gate_skips_when_not_provided() {
        assert_eq!(
            evaluate_gate(&EvalDataState::NotProvided, b""),
            Ok(EvaluateGate::Skip)
        );
    }

    /// REQ-17: `NotProvided`（評価データなしの状態フラグ）なのに
    /// `actual_bytes` が非空の場合は、状態と実データの不整合として
    /// `Skip`（`status:"skipped"`・exit 0）を返さず fail-closed で
    /// `NotProvidedButDataPresent` を返す（PR #209 レビュー指摘・issue #47
    /// スレッド: 評価データ本体が実際に渡されているのに状態フラグの
    /// 指定ミスだけで評価済みを装わず終わらせないこと）。
    #[test]
    fn req17_evaluate_gate_rejects_not_provided_with_data_present() {
        assert_eq!(
            evaluate_gate(&EvalDataState::NotProvided, b"some eval data"),
            Err(FreezeError::NotProvidedButDataPresent)
        );
    }

    /// REQ-17: 凍結済み（空入力を含む）で `actual_bytes` が記録と一致する
    /// 場合は `Proceed` に写る。
    #[test]
    fn req17_evaluate_gate_proceeds_when_frozen_including_empty() {
        let empty_record = freeze_eval_data(b"").expect("失敗しないはず");
        let state = EvalDataState::Frozen(empty_record.clone());
        assert_eq!(
            evaluate_gate(&state, b""),
            Ok(EvaluateGate::Proceed(empty_record))
        );

        let non_empty_record = freeze_eval_data(b"some eval data").expect("失敗しないはず");
        let state = EvalDataState::Frozen(non_empty_record.clone());
        assert_eq!(
            evaluate_gate(&state, b"some eval data"),
            Ok(EvaluateGate::Proceed(non_empty_record))
        );
    }

    /// REQ-17: `state` に含まれる凍結記録と `actual_bytes` から再計算した
    /// ハッシュが一致しない場合、`Proceed` を返さず fail-closed で
    /// `HashMismatch` を返す（PR #209 レビュー指摘・issue #47 スレッド:
    /// 形式上正しいが実データと無関係な記録を無条件に信用しないこと）。
    #[test]
    fn req17_evaluate_gate_rejects_hash_mismatch() {
        let stale_record = freeze_eval_data(b"original eval data").expect("失敗しないはず");
        let state = EvalDataState::Frozen(stale_record);
        assert_eq!(
            evaluate_gate(&state, b"tampered eval data"),
            Err(FreezeError::HashMismatch)
        );
    }

    /// REQ-17: `byte_len` だけを偽装し `sha256` は実データ由来の値のままの
    /// 記録でも、`FreezeRecord` の完全一致比較（`derive(PartialEq)`）により
    /// 拒否される。
    #[test]
    fn req17_evaluate_gate_rejects_byte_len_only_mismatch() {
        let genuine = freeze_eval_data(b"eval data").expect("失敗しないはず");
        let forged = FreezeRecord {
            algorithm: genuine.algorithm(),
            sha256: genuine.sha256(),
            byte_len: genuine.byte_len() + 1,
        };
        let state = EvalDataState::Frozen(forged);
        assert_eq!(
            evaluate_gate(&state, b"eval data"),
            Err(FreezeError::HashMismatch)
        );
    }

    /// REQ-17・REQ-39: 上限内の妥当な JSON は `FreezeRecord::parse` で成功する。
    #[test]
    fn req17_req39_parse_accepts_record_within_limit() {
        let record = freeze_eval_data(b"eval data").expect("失敗しないはず");
        let json = serde_json::to_string(&record).expect("serialize は成功するはず");
        assert!(json.len() as u64 <= MAX_FREEZE_RECORD_JSON_BYTES);
        let parsed = FreezeRecord::parse(&json).expect("上限内の JSON は成功するはず");
        assert_eq!(parsed, record);
    }

    /// REQ-17・REQ-39: `MAX_FREEZE_RECORD_JSON_BYTES` を超える入力は、
    /// JSON として妥当かどうかに関わらず `serde_json` へ渡す前に
    /// `TooLarge` として拒否される（fail-closed。資源の上限）。
    ///
    /// 末尾の空白は JSON の構文としては無害（parser が無視する）だが、本関数の
    /// 長さ検査は内容を解釈せずバイト長だけで判定するため、この入力でも
    /// `serde_json` を呼ばずに拒否できることを確認する。
    #[test]
    fn req17_req39_parse_rejects_oversized_json_before_deserializing() {
        let record = freeze_eval_data(b"eval data").expect("失敗しないはず");
        let json = serde_json::to_string(&record).expect("serialize は成功するはず");
        let base_len = json.len() as u64;
        let padding = usize::try_from(MAX_FREEZE_RECORD_JSON_BYTES - base_len + 1)
            .expect("padding は usize に収まるはず");
        let oversized = format!("{json}{}", " ".repeat(padding));
        assert!(oversized.len() as u64 > MAX_FREEZE_RECORD_JSON_BYTES);

        match FreezeRecord::parse(&oversized) {
            Err(LoadFreezeRecordError::TooLarge { size, limit }) => {
                assert_eq!(size, oversized.len() as u64);
                assert_eq!(limit, MAX_FREEZE_RECORD_JSON_BYTES);
            }
            other => panic!("TooLarge を期待したが {other:?} だった"),
        }
    }

    /// REQ-17・REQ-39: 構文的に不正な JSON は `Parse` エラーになる
    /// （上限検査を通過した後の通常のパース失敗）。
    #[test]
    fn req17_req39_parse_rejects_malformed_json_within_limit() {
        match FreezeRecord::parse("not json") {
            Err(LoadFreezeRecordError::Parse(_)) => {}
            other => panic!("Parse エラーを期待したが {other:?} だった"),
        }
    }

    /// テスト用の一時ファイルを、成否に関わらず削除するガード（RAII）。
    struct TempFileGuard(std::path::PathBuf);

    impl Drop for TempFileGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn write_unique_temp_file(label: &str, bytes: &[u8]) -> TempFileGuard {
        let pid = std::process::id();
        for attempt in 0..1000u32 {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let candidate = std::env::temp_dir().join(format!(
                "fandhe-edge-data-eval-freeze-unit-{pid}-{label}-{attempt}-{nanos}"
            ));
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&candidate)
            {
                Ok(mut file) => {
                    use std::io::Write as _;
                    file.write_all(bytes).expect("書き込みに失敗しないはず");
                    return TempFileGuard(candidate);
                }
                Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(err) => panic!("一時ファイルの作成に失敗しないはず: {err}"),
            }
        }
        panic!("一意な一時ファイルを作成できなかった");
    }

    /// REQ-17・REQ-39: 上限内の凍結記録ファイルは `FreezeRecord::load` で
    /// 読み込める。
    #[test]
    fn req17_req39_load_reads_record_file_within_limit() {
        let record = freeze_eval_data(b"eval data").expect("失敗しないはず");
        let json = serde_json::to_string(&record).expect("serialize は成功するはず");
        let guard = write_unique_temp_file("within-limit", json.as_bytes());

        let loaded = FreezeRecord::load(&guard.0).expect("上限内のファイルは成功するはず");
        assert_eq!(loaded, record);
    }

    /// REQ-17・REQ-39: `MAX_FREEZE_RECORD_JSON_BYTES` を 1 バイトでも超える
    /// ファイルは、内容を全量読み込む前に `TooLarge` として拒否される
    /// （`fandhe_edge_core::fs::read_bounded` のメタデータ確認による
    /// 早期拒否。REQ-39「資源の上限」）。
    #[test]
    fn req17_req39_load_rejects_oversized_file_before_reading_content() {
        let oversized_len =
            usize::try_from(MAX_FREEZE_RECORD_JSON_BYTES + 1).expect("usize に収まるはず");
        let oversized = vec![b' '; oversized_len];
        let guard = write_unique_temp_file("over-limit", &oversized);

        match FreezeRecord::load(&guard.0) {
            Err(LoadFreezeRecordError::Fs(FsError::TooLarge { size, limit, .. })) => {
                assert_eq!(size, MAX_FREEZE_RECORD_JSON_BYTES + 1);
                assert_eq!(limit, MAX_FREEZE_RECORD_JSON_BYTES);
            }
            other => panic!("Fs(TooLarge) を期待したが {other:?} だった"),
        }
    }

    /// REQ-17・REQ-39: 存在しないパスの読み込みは `Fs` エラーになる。
    #[test]
    fn req17_req39_load_reports_fs_error_for_missing_file() {
        let missing = std::env::temp_dir().join("fandhe-edge-data-eval-freeze-does-not-exist.json");
        match FreezeRecord::load(&missing) {
            Err(LoadFreezeRecordError::Fs(_)) => {}
            other => panic!("Fs エラーを期待したが {other:?} だった"),
        }
    }
}
