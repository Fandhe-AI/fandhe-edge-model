# Rust コーディング規約

## ツールチェーン

- `rust-toolchain.toml`（stable・rustfmt・clippy）を単一真実源とする
- `make fmt`・`make lint`（clippy `-D warnings`）・`make test` を通してからコミットする（clippy 警告 0 件を維持。[ci](./ci.md)）
- ビルド・テストは `docs/spec` 抜きで成立させる。コード・`build.rs`・テストから `docs/spec` 配下を参照しない

## crate 構成と層の境界

- 層は spec の 6 層（共通コア・データ契約・学習ワーカー・評価器・成果物 / 推論 SDK・操作アダプター）に従う。crate は `crates/<name>/` の 1 階層に配置。確定済み: `crates/core/`（`fandhe-edge-core`）・`crates/cli/`（`fandhe-edge-cli`）・`crates/data/`（`fandhe-edge-data`。検査・group 単位分割を実装済み。TASK-16.1-1・#38・TASK-17.1-1・#44）・`crates/eval/`（`fandhe-edge-eval`。現時点は正解率・ラベル別指標・Macro-F1・混同行列のみ実装済み。TASK-24.1-1・#59）。残りの層の crate は後続 TASK で追加する（PoC-16 の `03-poc/core-cli-vertical-slice/core`〔`edge_core`〕を M6・M8・M9 の骨格として移植する計画。`06-roadmap.md`）
- 依存は一方向に保つ: 共通コアはどの層にも依存しない。評価器は TASK-24.1 の 1 つだけに集約し、他の層で評価ロジックを再実装しない
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
