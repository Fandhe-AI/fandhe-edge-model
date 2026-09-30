# Rust コーディング規約

## ツールチェーン

- `rust-toolchain.toml`（stable・rustfmt・clippy）を単一真実源とする
- `make fmt`・`make lint`（clippy `-D warnings`）・`make test` を通してからコミットする（clippy 警告 0 件を維持。[ci](./ci.md)）
- ビルド・テストは `docs/spec` 抜きで成立させる。コード・`build.rs`・テストから `docs/spec` 配下を参照しない

## crate 構成と層の境界

- 層は spec の 6 層（共通コア・データ契約・学習ワーカー・評価器・成果物 / 推論 SDK・操作アダプター）に従う。crate は `crates/<name>/` の 1 階層に配置。確定済み: `crates/core/`（`fandhe-edge-core`）・`crates/guard/`（`fandhe-edge-guard`。許可リストによる形式判定・経路の閉じ込め〔`safe_join` 相当。TASK-39.4-1・#158〕・`infer` の `--package`・`onnx_file` への統合〔TASK-39.4-2・#159〕を実装済み。TASK-39.2-1・#153）・`crates/cli/`（`fandhe-edge-cli`）・`crates/data/`（`fandhe-edge-data`。検査・group 単位分割・凍結記録・読み取り専用配置・ハッシュ不一致時の停止・分割記録のレコード内容ハッシュ・来歴の記録型と取り込み記録・データ検査との接続を実装済み。TASK-16.1-1・#38・TASK-17.1-1・#44・TASK-17.2-2・#227・TASK-17.3・#251）・`crates/eval/`（`fandhe-edge-eval`。正解率・ラベル別指標・Macro-F1・混同行列・McNemar / Holm・下限基準比較・回帰・Wilson 区間と再現性・不変性・推論関数への input のみ受け渡し・凍結 test の 1 回限り適用・校正と棄権を実装済み。対象外ラベル〔TASK-22.2〕・coverage〔TASK-22.3〕・quadrant の multi-item・レポート系〔REQ-29〕・CLI 配線は未実装。TASK-24.1-1・#59）・`crates/train/`（`fandhe-edge-train`。学習ワーカー層の 学習リクエスト・結果 JSON の型・子プロセス起動と終了コード写像〔#178〕・探索予算内の候補選定〔TASK-18.1・#83・#84〕・選定結果の有意性判定〔TASK-18.3-1・#87〕・中断ジョブへの再開非提供とやり直し案内〔TASK-34.3・#147〕を実装済み。CLI `train` 工程への配線〔TASK-33.x〕は未着手。issue #177）。残りの層の crate は後続 TASK で追加する（PoC-16 の `03-poc/core-cli-vertical-slice/core`〔`edge_core`〕を M6・M8・M9 の骨格として移植する計画。`06-roadmap.md`）
- 依存は一方向に保つ（通常依存は data・eval → core。eval の data は dev-dependency のみ。train は core・data・eval に依存）: 共通コアはどの層にも依存しない。評価器は TASK-24.1 の 1 つだけに集約し、他の層で評価ロジックを再実装しない
- **推論ランタイムは学習側（学習ワーカー・学習用依存）に依存しない**（REQ-32。[dependency-policy](./dependency-policy.md)）。推論経路のコードに学習用 crate・Python 呼び出しを持ち込まない
- 操作アダプター（CLI・TUI・MCP）は薄く保ち、業務ロジックは下位層に置く。入出力契約は CLI の 1 つに集約し、TUI・MCP は CLI と同じ契約を使う（REQ-33・REQ-36・REQ-37）
- 循環依存を作らない。複数 crate が共有する型は下位 crate へ置く

## 公開 API・型設計

- 戻り値は将来拡張できる構造を持つ型にする（真偽値・フラットな文字列で済ませない）
- 判定結果・終了コード・状態（`skipped`・判定不能等）は enum で表し、壊れた値を表現できない型にする（[evaluation-contract](./evaluation-contract.md)）
- ハッシュは正準化した入力から計算し、正準化の規則を 1 箇所に集約する（REQ-15）
- 未実装・簡易実装の箇所は「実装済みを装わない」。ドキュメントコメントに将来仕様と対応する REQ-n を明記する（[code-comment-style](./code-comment-style.md)）

## エラーハンドリング・外部入力

- ライブラリコードでは `Result` を返し、panic させない
- 外部入力（定義ファイル・学習 / 評価データ・モデルファイル・CLI 引数・MCP リクエスト・学習ワーカーの出力）の経路では `unwrap` / `expect` / 添字アクセス（`[]`）を使わず、`get()`・`try_into()`・checked 演算で明示的に処理する
- サイズ・件数を上限検証してからアロケーションに使う（ファイルは読み込み前にサイズを確認する。REQ-39）
- 子プロセス（学習ワーカー等）・長時間処理には必ずタイムアウトと資源上限を設ける（REQ-39）
- エラーは 7 種の終了コードと機械可読な `code` / `message` の JSON に揃える（REQ-21）

## 数値・決定性

- 浮動小数の比較は許容差（1e-9）を明示し、`==` で比較しない
- 乱数は seed を引数で受け取り、グローバルな乱数状態に依存しない。並列処理で結果の順序・集計順が変わらないようにする

## unsafe・FFI

- `unsafe` は原則禁止。FFI 境界（ONNX Runtime 等）で必要な場合のみ、`// SAFETY:` コメントで理由と維持すべき不変条件を明記する
- `unsafe` の新規追加はユーザー承認を得る（レビューで P0 として扱う）

## クロスプラットフォーム

- 検証環境は Mac（Apple Silicon）のみ（`04-requirements.md` L107 付近）。Windows / Linux は M10 時点で対象外だが、将来の対応を妨げないよう OS 固有処理は `cfg(target_os = ...)` で局所化する
- パスは `PathBuf` / `Path::join` で組み立て、文字列連結・区切り文字のハードコードをしない
- 内部データファイルの改行は LF 固定

## テスト

- 挙動は REQ-n に対応づけてテストし、テスト名またはドキュメントコメントに ID を記す
- ユニットテストと結合テストを併置し、期待値は具体値で書く（真偽値のみの assert に頼らない）
- テストの skip・ignore・アサーション弱体化・許容差の拡大で CI を通さない。実機前提テストの扱いは [ci](./ci.md) に従う

## コメント

- [code-comment-style](./code-comment-style.md) に従う（`//!` / `///` のドキュメンテーションコメント）
