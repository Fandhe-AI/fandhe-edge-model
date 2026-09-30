# 依存管理規約（リポ固有）

## 原則

- **依存最小方針**: 共通コアと推論ランタイムは最小に保ち、外部依存は必要なものに限る。PoC-16 の Rust コアは `ort`・`unicode-normalization`・`serde`・`serde_json` の 4 件で成立している（`04-requirements.md` L107 付近）
- **推論は学習に依存しない**: 推論ランタイム・配布パッケージに学習側の依存（Python・MLX・学習用 crate）を持ち込まない。Python・MLX が PATH に無い `env -i` 環境でも推論が成功すること（REQ-32）
- **完全固定**: Rust は `Cargo.toml` で `=x.y.z` の完全固定（`^`・`~`・範囲指定は禁止）。workspace 共通依存は `[workspace.dependencies]` に集約する。Python は `pyproject.toml` で `==x.y.z` 固定し、lock ファイルをコミットする
- **ユーザー承認制**: 依存の追加・更新・削除は必ずユーザーの明示承認を経てから行う
- **承認記録の台帳と機械照合**: 承認記録は `dependency-approvals.json` に残し、`make check-dependency-approvals`（`make ci`・lefthook pre-commit・python-ci の pytest）が manifest・lock と照合して、記録の無い依存変更を fail-closed で止める（REQ-38・TASK-38.3・#165。手順とチェックリストは [dependency-approval-flow](../../docs/design/dependency-approval-flow.md)）
- **通信はユーザー承認後**: 依存・モデル重み・データセットの取得など通信を伴う操作は、明示承認を経てから実行する。推論・学習・評価の実行時に本ツール起因の通信を発生させない（REQ-38）

## 承認を求める際に提示する情報

1. パッケージ名・バージョン（`=x.y.z` / `==x.y.z`）・目的（なぜ自作でなく依存か）・配置する層（推論側か学習側か）
2. ライセンス（[licensing](./licensing.md) の許可範囲に収まること）
3. メンテナンス状況（最終リリース日・リポジトリの活動状況）
4. 推移的依存の概要（大量の間接依存・ネイティブビルド（C/C++）・プリビルドバイナリの自動ダウンロードを引き込まないか）
5. 配布サイズ・推論レイテンシへの影響（容量の目安 40MB・p95 250ms 未満。REQ-30・REQ-31）

## 承認済みの依存（Rust）

実装前に承認を得た依存の版を示す補助資料。導入する PR は本表の版で `[workspace.dependencies]` に `=x.y.z` 固定し、PR 本文には従来どおり承認の記録（「承認を求める際に提示する情報」の 1〜5 と承認日）を残す（本節への参照だけで済ませない。AGENTS.md「依存の追加・更新」）。表に無い依存・版の変更は改めて承認を得る。

| crate | 版 | 目的 | 配置する層 | ライセンス | 承認 |
| ----- | -- | ---- | ---------- | ---------- | ---- |
| `serde`（`derive` feature） | `=1.0.229` | 定義ファイル・JSON 入出力の型への読み込み・データ契約の記録型（凍結記録等）の直列化・学習リクエスト / 結果 JSON の型 | 共通コア（TASK-15.3〜15.5・REQ-15）・データ契約（TASK-16.1〜・REQ-16/17。dev-dependency を含む）・学習ワーカー（Rust 側の学習リクエスト / 結果の型。`crates/train`・#177・REQ-18/19） | MIT OR Apache-2.0 | 2026-09-27 オーナー承認（データ契約への配置は 2026-09-28 承認。学習ワーカー層への配置も 2026-09-28 承認） |
| `serde_json` | `=1.0.151` | JSON の読み書き（定義ファイル・CLI の JSON 出力・データ契約のレコード読み込み・学習ワーカーとの JSON 境界） | 共通コア（TASK-15.3〜15.5・REQ-15）・データ契約（TASK-16.1〜・REQ-16）・学習ワーカー（Rust 側の学習リクエスト / 結果の型。`crates/train`・#177・REQ-18/19）・推論ランタイム（dev-dependency のみ。共有ゴールデンベクタ fixture の読み込み。`crates/runtime`・#112・REQ-32） | MIT OR Apache-2.0 | 2026-09-27 オーナー承認（データ契約への配置も同日承認。学習ワーカー層への配置は 2026-09-28 承認。推論ランタイム層への配置は Issue #112 コメント 2026-09-29 に承認記録あり） |
| `sha2` | `=0.11.0` | 正準化ハッシュ（sha256）の計算。PoC-16 の `shasum` 子プロセス起動を置き換え、OS・外部コマンドに依存させない | 共通コア（TASK-15.5・REQ-15） | MIT OR Apache-2.0 | 2026-09-27 オーナー承認 |
| `unicode-normalization` | `=0.1.22` | 矛盾検出の入力を NFKC 正規化し、学習ワーカー（Python `unicodedata`）の共通正規化規則と一致させる | データ契約（TASK-16.2-2・#42・REQ-16）・推論ランタイム（バイト前処理の NFKC 正規化。`crates/runtime`・TASK-32.1-1・#112・REQ-32） | MIT OR Apache-2.0（crate の表記は旧式の `MIT/Apache-2.0`） | 2026-09-28 オーナー承認（当初 `=0.1.25` で承認し、Unicode 版を学習ワーカーと揃えるため同日 `=0.1.22` へ変更を承認）。推論ランタイム層への配置は Issue #112 コメント 2026-09-29 に承認記録あり |
| `rustix`（`default-features = false`・features `fs`・`alloc`・`std`） | `=1.1.5` | ディレクトリ fd 起点の `openat`（`O_NOFOLLOW`）・`fstat`・`F_GETPATH`。std に openat 相当が無く、`unsafe` の FFI を自前で持たずに、経路の閉じ込めの検証後の差し替え（TOCTOU）を塞ぐため | ガード層（`crates/guard`。TASK-39.4-1・#158・REQ-39。Linux・macOS 限定の target 依存） | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | 2026-09-29 オーナー承認（`std` feature は `rustix::fd` を `std::os::fd` と同一の型にして `File` へ変換するために追加） |

承認時に確認した推移的依存（2026-09-27 時点の crates.io。証拠種別: 一次情報の調査）:

- `serde`（derive）: `serde_core`・`serde_derive`・`proc-macro2`・`quote`・`syn`（3.0 系）・`unicode-ident`（`(MIT OR Apache-2.0) AND Unicode-3.0`）。`syn` 3.0 系は PoC-16 の lock でも同じ
- `serde_json`: `itoa`・`memchr`（Unlicense OR MIT）・`zmij`（MIT 単独。`ryu` の後継）
- `sha2`: `cfg-if`・`cpufeatures`（Apple Silicon では `libc`）・`digest`・`block-buffer`・`crypto-common`・`hybrid-array`・`typenum`
- いずれも `deny.toml` の許可ライセンスに収まり、C/C++ のネイティブビルド・プリビルドバイナリの自動ダウンロードは確認されなかった
- メンテナンス状況: 最終リリースは `serde` 2026-07-18・`serde_json` 2026-07-20・`sha2` 2026-03-25（いずれも yank なし）
- 配布サイズ・推論レイテンシへの影響: 共通コアの定義ファイル読み込みとハッシュ計算に限られ、推論の 1 件あたりの経路には入らない見込み（推定。導入 PR で実測値があれば記録する）
- `unicode-normalization`（2026-09-28 承認。証拠種別: 一次情報の調査〔crates.io・GitHub・crate のソース〕）: 推移的依存は `tinyvec`（`Zlib OR Apache-2.0 OR MIT`。`alloc` feature のみ）の 1 件。`build.rs`・C/C++ のネイティブビルド・外部ダウンロード・プリビルドバイナリなし（正規化テーブルは事前生成の静的テーブル）。0.1.22 は 2022-09-16 公開・yank なし、unicode-rs が継続保守。配布サイズへの影響は数十〜百数十 KB 程度（推定）で、推論の 1 件あたりの経路には入らない
- 版は Unicode のバージョンで選ぶ: `unicode-normalization` 0.1.22 は `UNICODE_VERSION = (15, 0, 0)`、学習ワーカーの Python 3.12（`trainer/.python-version`）の `unicodedata.unidata_version` も 15.0.0 で、NFKC の結果が文字単位で一致する（0.1.23 は 15.1.0、0.1.24 は 16.0.0、0.1.25 は 17.0.0 のため採用しない）。両側の Unicode 版が一致することはテストで固定し、Python または本 crate の版を上げる場合は両者の Unicode 版を揃えて改めて承認を得る（2026-09-28 オーナー判断）
- `rustix`（2026-09-29 承認。証拠種別: 一次情報の調査〔`Cargo.lock`・crate のソース〕）: `Cargo.lock` で確定した版は rustix 1.1.5。推移的依存は Linux が `bitflags` 2.13.2・`linux-raw-sys` 0.12.1（既定の linux_raw バックエンドで libc 不要）、macOS が `bitflags`・`libc`（既存の lock 版）・`errno` 0.3.14。Windows 向けの `windows-sys`・`windows-link` は `errno` の target 依存として lock に載るが、guard は Linux・macOS 限定の target 依存のため Windows ではビルドされない。`build.rs` は機能検出のみで、C/C++ のビルドもプリビルドバイナリもない。配布サイズへの影響は数十〜百数十 KB 程度（推定）で、推論の 1 件あたりの経路には入らない

## 事前学習済み重み・外部データ

- 事前学習済みモデルの重み・語彙・外部データセットの取得・同梱は依存追加と同様にユーザー承認事項とする（ライセンス・配布可否・容量を併せて提示する）

## subagent への適用

- builder Agent は依存の追加・更新を行わず、`dependency-approvals.json` も独断で編集しない（承認の記録は承認後に main が反映する）。必要と判断した場合は「承認事項」として main へ報告し、main がユーザーに確認する
