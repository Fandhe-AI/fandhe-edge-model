# sandbox_deny_log fixture の来歴

- 生成元: 手作りの**合成データ**（実機の `log stream` の出力ではない）。証拠種別: テストハーネス
- 形式の出典: PoC-16 で観測した `log stream --style ndjson` の出力形式
  （先頭のヘッダ行 `Filtering the log data ...` と、`eventMessage` に
  `Sandbox: <プロセス名>(<pid>) deny(<n>) <操作> [対象]` を持つ行）
- 用途: `scripts/sandbox_deny_report.py` の判定規則のテスト（REQ-38・TASK-38.1-2・#163）。
  `crates/cli/tests/sandbox_monitor_script.rs` から参照する
- 実在のユーザー名・ホスト名・`/Users/` パスは含めない（IP アドレスは文書用の予約範囲）
- `clean.ndjson` の `networkserviceproxy`（操作 `system-info`）と `WeatherMenu`
  （`networkextension` を含むパスへの `file-read*`）は、部分文字列 `network` で拾うと
  誤検出する PoC-16 の 2 例を模したもの
- ファイル一覧: `clean`（通信拒否なし）・`tool_python`・`tool_duplicate`（本ツール起因）・
  `unattributed`（帰属不明）・`unrecognized`（形式外の deny 行）・`garbage`（JSON でない行）・
  `no_header`（ヘッダなし）・`edge_cases`（括弧を含むプロセス名・単数形の重複報告・`network*`）
- `duplicate_with_original`: 元の行と `3 duplicate reports for` の要約行が両方ある（発生回数は 4。
  二重計上の回帰）。`leak_probe`: 許可リスト外のプロセス名・通信先に、レポートへ
  出してはならない目印の文字列（`secret-host.example.invalid`・`PrivateAppName`）を含む
  （生文字列非保存の回帰。すべて架空の値）
- `unrecognized_no_network`: `network` を含まない形式外の拒否行（判定不能になることの回帰。合成データ）
- `tool_bigpid`: 7 桁の PID を持つ本ツール起因の拒否行（偽の `pgrep` が同じ PID を返すテストで使う。合成データ）
- 陽性対照の拒否行（PoC-16 実測の形 `Sandbox: curl(<pid>) deny(1) network-outbound /private/var/run/mDNSResponder` を模した合成データ。TASK-38.2・#164）は、テストが偽の `log` から出す行と `crates/cli/tests/sandbox_monitor_script.rs` 内の合成ストリームで持ち、ファイルとしては追加していない
- `system_policy_non_network`・`system_policy_network`・`system_policy_unrecognized`: `System Policy: <プロセス名>(<pid>) deny(<n>) <操作> [対象]` 形式の拒否行（Issue #331 で実機に出ると観測された形を模した**合成データ**。実機の生ログの転記ではない。証拠種別: テストハーネス。TASK-38.1-2・#163）。`non_network` は非通信操作（`file-read-data`）、`network` は `network-outbound` と重複要約行（発生回数 3）、`unrecognized` は `(<pid>) deny(<n>)` の構造を欠く行（判定不能の回帰）
