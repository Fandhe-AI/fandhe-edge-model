# unsafe 許可リストの運用手順

REQ-39・#335（#329 の security-auditor 監査 P2）。ワークスペースは `unsafe_code = "deny"` で、やむを得ない FFI などだけを `#[allow(unsafe_code)]` と `// SAFETY:` コメントで局所的に許可する（[coding-rust](../../.claude/rules/coding-rust.md)）。「`unsafe` の新規追加はオーナー承認」を機械で担保するため、出現箇所を `unsafe-allowlist.json` と fail-closed で照合する。

証拠の種別: 照合ゲートの動作はテストハーネス（合成の変異を加えた pytest）で確認している。承認が実在するかどうかは機械では確かめられず、PR レビューで確認する。

## 構成要素

| 要素 | 役割 |
| ---- | ---- |
| `unsafe-allowlist.json`（リポジトリルート） | 許可リスト。`file`・`item`（enclosing `mod` と item 名。例 `darwin::pidinfo`）・`level`・`count`・`approved_on`・`approved_by`・`record`・`purpose` |
| `scripts/check_unsafe_allowlist.py` | `crates/` の全 `*.rs` を文字列走査して照合する（標準ライブラリのみ・読み取り専用・通信なし）。終了コードは 0・10（未承認・件数超過・陳腐化・lint 緩和）・64（形式不正）・70 |
| `make check-unsafe-allowlist` | 上記の実行口。`make ci` の前提に含まれる |
| lefthook の pre-commit `unsafe-allowlist` | Rust ソース・manifest・許可リストが staged のときに index の内容で照合する |
| `trainer/tests/test_unsafe_allowlist.py` | 実リポジトリの照合と陰性対照。python-ci（`make py-ci`）で全 PR に効く |

走査にした理由は、macOS 専用コードが Linux の clippy では除外され lint で見逃されるため。走査は cfg を問わず全ファイルを見る。

## 手順

1. `allow(unsafe_code)` の追加が必要になったら、実装前にオーナーの承認を得る（`unsafe` の新規追加。builder Agent は承認事項として main へ報告する）
2. 承認後、同じコミットで `unsafe-allowlist.json` に記録を追加・更新する（`approved_on`・`approved_by`・`record`〔Issue・PR 等の実在する参照〕・`purpose`）
3. 出現を削除・移動したときは該当記録も更新する（残すと `stale_entry` で失敗する）
4. `make check-unsafe-allowlist` を通す

## チェックリスト

- 記録の `record` が実在の承認記録を指しているか（レビューで確認する。同じ PR で許可リストを書き換えれば機械照合は通る）
- `// SAFETY:` コメントが理由と不変条件を述べているか
- ルート `[workspace.lints.rust] unsafe_code` が deny のままか、メンバー crate が `[lints] workspace = true` のみか

## 限界

- マクロ展開で生成される `allow` は対象外
- rustfmt 済みのソースを前提に item を判定する（`make fmt-check` が担保）
- `unsafe` ブロック自体は `deny` のコンパイルエラーが止める
