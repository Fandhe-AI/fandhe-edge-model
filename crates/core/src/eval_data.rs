//! 評価契約（`.claude/rules/evaluation-contract.md`）に関わる型を集約するモジュール。
//!
//! `evaluate` 工程（CLI 側の実体は TASK-33.1 未着手）が参照する評価データの
//! 凍結記録・状態を表す型を置く。凍結処理そのもの（ファイル読み込み・サイズ検証・
//! ハッシュ計算）はデータ契約層（`crates/data` の `eval_freeze` モジュール）が担い、
//! 本モジュールは層を跨いで共有する型の定義のみを持つ（共通コアは他層に依存しない
//! 最下層のため、ここには I/O を含む処理を置かない）。

use std::path::PathBuf;

/// 評価データの凍結記録（REQ-17）。
///
/// ハッシュは [`crate::hash::sha256_hex_of_reader`] による生バイト列の sha256 で、
/// 定義ファイルの正準化ハッシュ（TASK-15.5）とは別物である。
///
/// 以後の一致確認（TASK-17.3・評価契約「評価データは凍結し、ハッシュが記録と
/// 一致しなければ処理を停止する」）は、この record の `sha256` を再計算値と
/// 突き合わせる形で実装する想定。読み取り専用配置への書き込み拒否（おそらく
/// TASK-17.2-2）は別タスクが担い、本 record 自体はファイルの権限を変更しない
/// （権限設定はハッシュ記録の後に追加できる構造にしてあり、record の形は
/// 変えない想定）。
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FreezeRecord {
    /// 凍結対象の評価データのパス（呼び出し元が確定させた単一パスをそのまま保持する。
    /// パストラバーサル対策〔経路の閉じ込め〕はガード層〔REQ-39〕の責務であり、
    /// 本型は検証済みの前提で受け取ったパスを記録するのみ）。
    pub path: PathBuf,
    /// 生バイト列の sha256（小文字 16 進 64 桁）。
    pub sha256: String,
    /// 実際に読み込んで `sha256` の計算対象にしたバイト数。`stat` 由来の
    /// メタデータ値ではなく、ストリーミング読み込みで実際にハッシュへ通したバイト数を
    /// 記録する（`crates/data` の `eval_freeze` 実装が担う TOCTOU 対策。
    /// `metadata()` 取得後にファイルが変化しても、この値は `sha256` と必ず対応する）。
    pub byte_len: u64,
}

/// `evaluate` 工程が参照する評価データの状態（REQ-17 境界値・評価契約）。
///
/// `NotProvided` は評価データが与えられなかったことを表し、`evaluate` は
/// これを受けて `status:"skipped"`・exit 0 を返すべきである（終了コードへの
/// 変換自体は TASK-21.1 の型が担う。ここでは対応関係をドキュメンテーション
/// コメントで示すに留め、CLI 配線・JSON スキーマは発明しない）。
///
/// 将来 TASK-17.3 でハッシュ不一致を表すバリアント（例: `Mismatched`）を
/// 追加できるよう `#[non_exhaustive]` にしてあり、呼び出し元は必ず
/// デフォルト分岐（`_ =>`）を伴う `match` を書く必要がある。
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvalDataStatus {
    /// 評価データが与えられなかった状態。
    NotProvided,
    /// 評価データが凍結され、ハッシュが記録された状態。
    Frozen(FreezeRecord),
}

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-17: `EvalDataStatus` は `match` で網羅的に分岐できる
    /// （`#[non_exhaustive]` によりデフォルト分岐が必須になることも含めて確認する）。
    #[test]
    fn eval_data_status_matches_not_provided_and_frozen() {
        let not_provided = EvalDataStatus::NotProvided;
        let frozen = EvalDataStatus::Frozen(FreezeRecord {
            path: PathBuf::from("/tmp/example.jsonl"),
            sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".to_string(),
            byte_len: 0,
        });

        // `#[non_exhaustive]` はクレート外からの `match` にのみワイルドカード腕を
        // 強制するため、定義元の本クレート内ではワイルドカードを書くと
        // `unreachable_patterns` になる（clippy `-D warnings` で検出済み）。
        let describe = |status: &EvalDataStatus| -> &'static str {
            match status {
                EvalDataStatus::NotProvided => "not_provided",
                EvalDataStatus::Frozen(_) => "frozen",
            }
        };

        assert_eq!(describe(&not_provided), "not_provided");
        assert_eq!(describe(&frozen), "frozen");
    }
}
