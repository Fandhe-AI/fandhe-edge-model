---
name: judge
description: 利用者が用意した判定モデル（fandhe-edge-model のパッケージ）で、入力テキストを選択肢に分類したいときに使う。ローカルの常駐 serve 経由で判定する。
---

# judge

入力テキストを、パッケージの選択肢のどれに当たるか判定する。

## 呼び方

```sh
"${CLAUDE_PLUGIN_ROOT}/bin/fandhe-edge-judge" --text "<入力>"
```

- `--package` は付けない。環境変数 `FANDHE_EDGE_PACKAGE` から渡される
- `--id`・`--input-file`・`--out`・`--version-ledger` は付けない。付けると常駐に委ねず単独の判定になる
- 入力は 1 MiB 以下にする（超えると常駐に委ねない）

## 出力

stdout に JSON が 1 つ出る。判定結果は `id`・`status`・`predicted_label`・`scores` を持つ。

## 終了コード

| 終了コード | code | 意味と対応 |
| ---------- | ---- | ---------- |
| 0 | ok | 判定結果。`status` が `ok` |
| 10 | judged_fail | 判定上の不合格 |
| 11 | out_of_scope | 範囲外。結果行は出る |
| 12 | pending | `status` が `abstain` の保留。判断を人へ返す |
| 20 | limit_exceeded | 時間・サイズ・RSS の上限超過 |
| 64 | invalid_input | 引数・環境変数の誤り。直して再実行する |
| 70 | runtime_error | `fandhe-edge` の不在など。利用者へ報告する |

## 前提

- `fandhe-edge` が PATH にあること（インストールは `plugin/README.md`）
- `FANDHE_EDGE_PACKAGE` にパッケージの絶対パスが設定されていること
- モデルはプラグインに同梱しない

## 常駐

呼ぶたびに裏で `serve` を起動する。使われない時間が `FANDHE_EDGE_IDLE_EXIT_SECS`（既定 600 秒）続くと、自分で終わる。手で止めるには `fandhe-edge serve --stop` を実行する。
