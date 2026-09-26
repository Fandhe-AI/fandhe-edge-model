# spec 参照規約（リポ固有）

## 前提

- 仕様・要件定義の SSOT は [fandhe-edge-model-spec](https://github.com/Fandhe-AI/fandhe-edge-model-spec)（`docs/spec` submodule）の `04-requirements.md`（REQ-15〜40。REQ-19b を含む）。タスク定義は `05-tasks.md`（TASK-15.1〜40.x）、マイルストーンは `06-roadmap.md`（M6〜M11）、PoC 成果物は `03-poc/`・`02-poc-plan.md`（PoC-n・C-n）、設計判断・疑問点は `01-brainstorm.md`（疑問点 n）
- v1・v2（REQ-1〜14 と対応 TASK・M1〜M5）は凍結済みで、本リポの実装対象外（`01-brainstorm.md` 疑問点 24）
- spec リポは private として維持するが、**spec の内容を本リポ（public）のコード・コメント・ドキュメント・Issue・PR・コミットメッセージに載せてよい**（オーナー判断 2026-09-27）。ただし要約・引用は必要な範囲に留める
- spec リポへのアクセス権がない環境では `docs/spec` を解決できない

## 参照の仕方

- 対応する REQ-n・TASK-n・M-n・PoC-n を必ず併記し、SSOT へ辿れるようにする
- 要約・引用は必要な範囲に留め、spec ファイルの丸ごとコピーはしない（spec 更新時に内容が乖離するため。詳細は ID から spec を参照させる）
- spec と本リポの記述が食い違った場合は spec を正とし、spec 側の変更が必要ならユーザーへ報告する
- 状態が「検討中」の要件（例: REQ-37 の MCP）や、spec が「確定しない」とした事項（例: 学習側の Rust 化範囲）は確定扱いしない。実装で判断が必要になったらユーザーへ確認する
- spec の証拠種別（テストハーネス・模擬・推定・実機）の区別を、本リポの記述・PR でも崩さない

## 運用

- `docs/spec` 配下のファイルを本リポ側で編集しない（spec リポ側で管理し、本リポは submodule 参照の更新のみ行う）
- ビルド・テストは `docs/spec` 抜きで成立させる。コード・`build.rs`・テスト・学習スクリプトから `docs/spec` 配下を読み込まない（PoC のコード・データを流用する場合は本リポへ移植し、出典の PoC-n を記す）
- spec 内に資格情報・個人情報・実機のホスト名 / アドレスなど本来公開すべきでないものを見つけた場合は転記せず、ユーザーに報告する
