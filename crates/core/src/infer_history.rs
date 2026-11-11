//! `infer --text` の判定履歴 1 行（`history.jsonl`）の型と直列化（issue #711。REQ-33・REQ-38・REQ-39）。
//!
//! cli の `infer_history` が、環境変数 `FANDHE_EDGE_HISTORY=1` のときだけ手元のファイルへ 1 件 1 行で追記する際に使う。
//! cli は `serde_json` に依存しない方針のため、JSON の組み立てはこのモジュールに閉じる。
//! 操作アダプター（cli）専用で、判定・評価の契約には入らない。
//! 履歴には入力の全文が残る（既定は無効。有効化は利用者の明示操作）。I/O は持たない。

use serde::Serialize;

/// 履歴 1 行。欄の名前と順序は履歴ファイルの形式（利用側が読む）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct HistoryRecord {
    /// 判定を終えた時刻（Unix 秒）。
    pub time: u64,
    /// 入力の全文。
    pub text: String,
    /// `infer` が stdout に出した JSON（読めなければ文字列のまま）。読み直した値なので、キーの順・数値の表記は
    /// stdout のバイト列と一致しないことがある。
    pub output: serde_json::Value,
    /// 終了コード（数値）。
    pub exit_code: u8,
    /// 外側で測った所要時間（ミリ秒）。
    pub elapsed_ms: u64,
    /// `--package` に渡された値（非 UTF-8 は lossy）。
    pub package: String,
    /// CLI の版。
    pub cli_version: String,
}

/// stdout の控え（先頭 64 KiB まで）から `output` 欄の値を作る。JSON 1 つとして読めればその値、
/// 読めない（64 KiB で切れた・空など）なら文字列のまま（非 UTF-8 は lossy）。
#[must_use]
pub fn output_value(captured: &[u8]) -> serde_json::Value {
    serde_json::from_slice(captured).unwrap_or_else(|_| {
        serde_json::Value::String(String::from_utf8_lossy(captured).into_owned())
    })
}

/// 1 行（末尾の改行を含む）に直列化する。直列化に失敗したら `None`（呼び出し側は黙って捨てる）。
#[must_use]
pub fn encode(record: &HistoryRecord) -> Option<Vec<u8>> {
    let mut line = serde_json::to_vec(record).ok()?;
    line.push(b'\n');
    Some(line)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(output: serde_json::Value) -> HistoryRecord {
        HistoryRecord {
            time: 1_700_000_000,
            text: "a\"b\n日本".into(),
            output,
            exit_code: 12,
            elapsed_ms: 34,
            package: "pkg".into(),
            cli_version: "0.1.0".into(),
        }
    }

    /// REQ-33・#711: エスケープされた 1 行になり、output が JSON のときは値として入る。
    #[test]
    fn encode_escapes_and_embeds_json_output() {
        let out = output_value(br#"{"a":1}"#);
        let line = String::from_utf8(encode(&record(out)).expect("encode")).expect("utf8");
        assert_eq!(
            line,
            "{\"time\":1700000000,\"text\":\"a\\\"b\\n日本\",\"output\":{\"a\":1},\"exit_code\":12,\"elapsed_ms\":34,\"package\":\"pkg\",\"cli_version\":\"0.1.0\"}\n"
        );
    }

    /// #711: 読めない・切れた出力は文字列のまま入る。
    #[test]
    fn unreadable_output_is_kept_as_string() {
        assert_eq!(
            output_value(br#"{"a":"#),
            serde_json::Value::String("{\"a\":".into())
        );
    }
}
