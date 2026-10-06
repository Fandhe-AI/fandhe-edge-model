# PoC-26 事前登録 追補 1（追加学習候補 P の学習・採点設定）

対応: REQ-41・TASK-41.1-5・PoC-26・M12・issue #390。親文書は `docs/design/poc26-preregistration.md`（本文は変更しない）。実装は `trainer/tools/poc26/`（入口 `lora_poc.py`、実装は `cli.py`・`train.py`・`predict.py`・`probe.py`・`score.py`・`assets.py`・`io_records.py`・`common.py`・`safe_io.py`、`qwen2_model.py`・`qwen2_tokenizer.py`）。

## 1. 日時と理由

- 日時: 2026-10-06（評価データはまだ存在せず、どのモデルもどの評価データにも当てていない）。
- 理由: 事前登録 4 節の「rank 等は #390 で固定して追補に記録」に従い、候補 P の学習・採点の設定を評価前に固定する。本文の確定内容（格子 iters ∈ {300, 600} × lr ∈ {1e-4, 5e-5}・seed 0/1/2・指標・有意性）は変えない。
- 証拠の種別: 以下は設計上の固定であり、実機の測定値ではない。実機で確認する項目は 6 節で、結果は記録欄へ証拠の種別付きで書く。

## 2. 固定した設定

| 項目 | 固定値 | 由来・理由 |
| ---- | ------ | ---------- |
| LoRA rank / scale / dropout | 8 / 20.0 / 0.0 | mlx-lm の既定。PoC-21 P4 と同じ |
| 適用層 | 末尾 8 ブロックの q/k/v/o/gate/up/down（全 Linear）。学習可能パラメータ 1.466M（float32） | P4 の `--num-layers 8` と一致（P4 のログ 1.466M と計算値が一致） |
| LoRA 初期化 | `a` ~ U(-1/√in, 1/√in)、`b` = 0。乱数は **`--seed` から導いた鍵**（`mx.random.key(seed)` を split。グローバルな乱数状態に依存しない）。`adapter_config.json` の `lora_init_seed` に記録する | 学習前は base と出力一致。同じ seed で同じ初期値 |
| optimizer / lr | Adam・定数 lr・weight decay なし。lr は格子 {1e-4, 5e-5} | P4 と同じ |
| batch / iters / max_seq_length | 4 / 格子 {300, 600} / 512 | 長さは 2048 から 512 へ下げる。超過は切り詰めず**エラー停止**（`run.json` の `truncated_count` は常に 0） |
| バッチの作り方 | 長さ順ソート → 4 件ずつ → `np.random.default_rng(seed).permutation` の順。端数は捨てる。pad は `<\|endoftext\|>`（損失から除外） | mlx-lm の `iterate_batches` に揃える |
| 損失 | pad 以外の全 token の cross_entropy 平均（prompt もマスクしない）。float32 | P4（`mask_prompt=False`） |
| dtype | 重みは bf16 のまま読む。LoRA・loss・対数尤度は float32。`--dtype float32` で全体を float32 に昇格 | mlx-lm と同じ。CPU で bf16 が動かない場合・決定性確認用 |
| seed | `random`・`numpy`・`mlx` の 3 系統。必須引数 | REQ-26 |
| 学習の壁時計予算 | `--max-wall-seconds`（既定 3600 = 事前登録 4 節の 1 時間。上限 86400）。**学習ループだけ**に適用し、到達したら学習を打ち切って adapter を保存し、`run.json` に `budget_reached: true`・実施 iters（`iters_done`）を記録して validation 採点へ進む（「予算到達は記録してその時点の最良で評価する」） | 事前登録 4 節。時間はコマンド開始（読み込みを含む）からで、1 ステップ以上は必ず行う。予算で打ち切った実行は決定的な再現の対象外。実機の CPU 決定性確認（6.4）では `--max-wall-seconds` を延ばして打ち切りを避ける |
| 資源上限（停止） | メモリ: RSS と MLX のピーク確保量（`mx.get_peak_memory()`）の大きい方が 8 GiB（`limits.MAX_TRAIN_RSS_BYTES`）を超えたら停止（ステップ・レコードの境界で確認。1 回の forward / backward の途中では止められない）。コマンド全体の壁時計 24 時間の天井。採点（train の validation 採点・predict・probe の forward）の壁時計上限 `--max-score-seconds`（既定 3600。上限 86400）は学習とは別枠で、到達は停止（20）。モデル読み込みは 1 回の処理なので壁時計上限の対象外。事前見積もり: 学習は batch_size x max_seq_length x vocab ≤ 5e8 要素（backward の保持を forward の約 3 倍と**推定**して 2 GB x 3 = 6 GB < 8 GiB。係数は実測ではない）、採点は 1 件の K x 系列長 ≤ 2^20 token。超過は終了コード 20。採点は K を 8 件ずつに分けて forward する（合算は分割に依らない） | REQ-39 |

## 3. 採点（P4 からの意図的な変更）

- 各ラベルの token 列に **`<|im_end|>` を加えた**条件付き対数尤度の合計を採点値とする。**長さ正規化はしない**。K=26 件を 8 件ずつ forward し（合算は分割に依らない）、softmax を `scores` に、既存の対応づけ (b)（`map_scores_to_choice`）で `predicted_label` を決める（同点は宣言順で先頭。非有限は `status:"error"`）。
- 理由: 26 ラベルに前方一致の対（`browser_navigate`／`browser_navigate_back`、`browser_network_request`／`browser_network_requests`）がある。EOS を含めないと P(接頭辞) ≥ P(接頭辞＋続き) が常に成り立ち、長い方を選べない。学習の completion は `{label}<|im_end|>\n` で終わるため、採点と整合する。
- この変更は評価データに当てる前に固定する。評価後には変えない。

## 4. system プロンプトと chat template

system プロンプト全文（`build_system_prompt`。ラベルは定義ファイルの宣言順。以下は事前登録 3 節の 26 ラベルの順）。定義ファイルの宣言順が異なる場合は、プロンプトの列挙順も変わり、`run.json` の `system_prompt_sha256` が変わる。

```text
You are a tool selector for Playwright MCP. The user describes a browser task in Japanese. Reply with exactly one name from the list below and nothing else. Reply none when no tool applies.
Names:
- browser_click
- browser_close
- browser_console_messages
- browser_drag
- browser_drop
- browser_emulate_media
- browser_evaluate
- browser_file_upload
- browser_fill_form
- browser_find
- browser_handle_dialog
- browser_hover
- browser_navigate
- browser_navigate_back
- browser_network_request
- browser_network_requests
- browser_press_key
- browser_resize
- browser_run_code_unsafe
- browser_select_option
- browser_snapshot
- browser_take_screenshot
- browser_type
- browser_wait_for
- browser_tabs
- none
```

- user メッセージは `input` をそのまま入れる（NFKC はしない。tokenizer の NFC のみ）。
- chat template は tools なし経路の 3 role 分だけを固定で組み立てる: `<|im_start|>{role}\n{content}<|im_end|>\n` の連結。採点時は末尾に `<|im_start|>assistant\n` を付け（`add_generation_prompt`）、学習時は assistant の `{label}<|im_end|>\n` を続ける。特殊トークンは id で挿入する。
- `tokenizer_config.json` との照合（PR #446 のレビュー対応）: model-dir の `tokenizer_config.json` を上限（1 MiB）つきで読み、`eos_token`=`<|im_end|>`・`pad_token`=`<|endoftext|>` と、`chat_template` が文字列で `<|im_start|>`・`<|im_end|>`・`assistant`・`add_generation_prompt` を含むことを確認する。これは**粗い整合確認**で、テンプレート全文の一致は保証しない。全文の一致は現実的に機械照合できない（Jinja の評価系は標準ライブラリになく、依存を足さない）ため、`chat_template` の sha256 を `run.json` の `chat_template_sha256` に記録し、**既知の sha256 との照合を人間の実機確認項目**とする（6 節の記録欄。`tokenizer.json` の golden 照合と同様）。

## 5. 特殊トークン文字列を本文で照合しない

`encode()` は本文中の `<|im_start|>` 等を通常の文字として分割する。HF `tokenizers` の既定（added_tokens を照合する）とは異なる。入力本文によるプロンプト注入を塞ぐ（REQ-39）。特殊トークンは template 側が id で挿入する。golden ベクタに特殊トークン文字列は含めない（`docs/design/poc26-tokenizer-golden-procedure.md`）。

## 6. 実重みでの確認（人間・実機）

前提: #387 でベースモデル一式（`config.json`・`model.safetensors`・`tokenizer.json`・`tokenizer_config.json`）をリポ外のディレクトリ `<dir>` に用意し、各 sha256 を記録してある。`train`・`predict`・`probe` はその 4 つの sha256（64 桁の小文字 16 進）を必須引数 `--model-sha256`・`--config-sha256`・`--tokenizer-sha256`・`--tokenizer-config-sha256` で受け、一致しなければ終了コード 64 で停止する（以下の例の `<sha>`）。以下の `lora_poc.py` は `uv run --locked --directory trainer python -m tools.poc26.lora_poc` で起動する（以下 `lora_poc`）。

### 6.1 往復と健全性（リポ内コードのみ）

```bash
lora_poc probe --model-dir <dir> --out <work>/probe_ours.json --dtype float32 --evidence real_machine \
  --model-sha256 <sha> --config-sha256 <sha> --tokenizer-sha256 <sha> --tokenizer-config-sha256 <sha>
```

- 合格条件: 全 case で `roundtrip_ok` が true（NFC 正規化後の本文と一致）。`greedy` の 3 件（LoRA なし）が日本語・英語として読める文になる。崩れていれば forward の誤り（RoPE・scale・bias）を疑う。

### 6.2 参照との突き合わせ（必須・リポ外の使い捨て venv）

`mlx-lm`・`tokenizers` の導入は通信を伴うため、オーナーの承認を得てから行う（REQ-38）。リポの依存は変えない。参照側のスニペット（リポ外の使い捨て venv で実行）:

```python
import base64
import json
import sys

import mlx.core as mx
import numpy as np
from mlx_lm import load
from tokenizers import Tokenizer

model_dir, ours_path, out_path = sys.argv[1:4]
ours = json.load(open(ours_path, encoding="utf-8"))
tok = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
model, _ = load(model_dir)
model.set_dtype(mx.float32)
cases = []
for case in ours["cases"]:
    ids = tok.encode(case["text"], add_special_tokens=False).ids
    logits = model(mx.array([ids]))[0, -1].astype(mx.float32)
    mx.eval(logits)
    raw = np.array(logits, dtype="<f4").tobytes()
    cases.append(
        {"text": case["text"], "ids": ids, "last_logits_f32_b64": base64.b64encode(raw).decode()}
    )
json.dump({"evidence": "real_machine", "cases": cases}, open(out_path, "w", encoding="utf-8"))
```

```bash
lora_poc compare-probe <work>/probe_ours.json <work>/probe_ref.json --atol 1e-3
```

- 合格条件: 終了コード 0（`status` が `match`）。すなわち token id が全 case で完全一致・logits の最大絶対差が 1e-3 以下・argmax が全 case で一致（両側 float32。bf16 同士なら `--atol 5e-2`）。
- 不合格（終了コード 10）なら、`id_mismatches` > 0 はトークナイザー、`logits_mismatches` > 0 は forward を疑い、`golden` 手順書と同様に原因を特定する。許容差を広げて通さない。

### 6.3 固定

- 6.2 の token id を `fixtures/poc26/tokenizer_golden.json` に固定する手順は `docs/design/poc26-tokenizer-golden-procedure.md`。
- `run.json` の `chat_template_sha256` を、HF の公式配布の `tokenizer_config.json` から取った `chat_template` の sha256 と照合し、記録欄に書く（4 節）。

### 6.4 CPU の決定性（#391 の前）

`--device cpu --seed 0` で `train` を同じ引数で 2 回（iters 300・validation 数十件で可）実行し、`pred.jsonl`・`adapters.safetensors` の sha256 が一致することを確認する。bf16 が CPU で動かなければ `--dtype float32` で行い、記録する。予算での打ち切り（`budget_reached`）が起きないよう `--max-wall-seconds` を延ばす。

### 6.5 記録欄（結果は証拠の種別付きで #390 のコメントにも書く）

iters 600 の実行（GPU 本番の前に 1 回。#391）の所要時間と最大 RSS を `run.json` から転記し、1 時間予算・RSS 8 GiB に収まるかを確認する。

| 項目 | 記入 |
| ---- | ---- |
| 6.1 往復・貪欲生成（実機） | 未実施 |
| 6.2 参照との突き合わせ（実機。件数・最大絶対差・argmax 一致・mlx-lm／tokenizers の版・日時） | 未実施 |
| 6.3 chat_template の sha256（実測・既知値・一致） | 未実施 |
| 6.4 CPU 決定性（実機。dtype・sha256 の一致） | 未実施 |
| iters 600 の所要時間（`train_seconds`）と `max_rss_bytes`（実機。`budget_reached` の有無・device・dtype） | 未実施 |

6.2 を省く場合はオーナー判断とし、P の結果に「forward・トークナイザーの一致は未確認（往復と貪欲生成のみ）」の注記を付ける。

## 7. 読み込み上限と実測バイト数

- `model.safetensors` ≤ 1 GiB（2^30）・`tokenizer.json` ≤ 16 MiB・`tokenizer_config.json` ≤ 1 MiB・`config.json` ≤ 64 KiB・定義ファイル ≤ 1 MiB・学習データは `limits.MAX_TRAIN_DATA_BYTES`（1 行は `MAX_TRAIN_LINE_BYTES`・件数は `MAX_TRAIN_EXAMPLES`）。読む前に `fstat` で検査する。symlink は辿らない。
- 実サイズと sha256 は `run.json` の `model_safetensors_bytes`・`model_safetensors_sha256` に記録する。#400 の判断材料（実測 988MB 前後。10 進でも 2 進でも 1 GiB 未満の見込み）。

| 項目 | 実測（実機） |
| ---- | ------------ |
| `model.safetensors` のバイト数 | 未実施 |

## 8. 1 件／バッチの一致と p95

- 1 件／バッチの一致は対象外（事前登録 5 節の再掲）。採点は常に 1 件 × K 選択肢（K=26。8 件ずつの forward）。
- 推論の p95（#393 の材料）: `predict` の 1 件あたり採点時間を `run.json` の `scoring`（`count`・`mean_seconds`・`p95_seconds`・`max_seconds`・`choices_per_prompt`〔1 件あたりの選択肢数 K=26〕・`forward_chunk`〔1 回の forward に入れる選択肢数 8〕）に記録する。`train` の validation 採点は `validation_scoring` に同じ形で記録する。実機の p95 は人間が実測する（#393）。

## 9. 入力の脅威モデル（P3）

- 入力は PoC を実行する人が渡すローカルの信頼できるパスだけを想定する。ガード層（REQ-39）の `safe_join` 相当は持たない。
- `O_NOFOLLOW` は最後の要素だけに効き、パスの途中のディレクトリの symlink は辿る。これは脅威モデル上許容する。
- ベースモデルは PR-B の `load_qwen2` が、#387 で記録した sha256（必須引数 `--model-sha256`・`--config-sha256`）と照合してから読む（検証済み fd から全バイトを 1 度だけ読んだバイト列の照合・形状と dtype の検証・ピークメモリの見積もりを含む）。`tokenizer.json`・`tokenizer_config.json` も同様に必須引数（`--tokenizer-sha256`・`--tokenizer-config-sha256`）で照合する。token id（学習・採点の入力）と chat template の前提を、記録したものと同一に固定するため。不一致は終了コード 64。照合済みの値を `run.json`（`sha256_pinned: true`）と `adapter_config.json` に記録する。`adapters.safetensors` は検証済みの fd から全バイトを 1 度だけ読み、そのバイト列から sha256 と配列（`mx.load` に `BytesIO` を渡す）の両方を作る（開き直さない）。
- adapter の設定（rank・scale・dropout・num_layers・keys・lora_init_seed・dtype・max_seq_length・ベースモデル / config / tokenizer の sha256 群・system プロンプトの sha256・ラベル順）は `adapters.safetensors` の **metadata**（キー `fandhe_adapter_config`、JSON 文字列 1 つ）に埋め込む。`predict` は必須引数 `--adapter-sha256`（学習時に `run.json` の `adapters_sha256` へ記録した、ファイル全体のハッシュ）と読んだバイト列を照合するので、重みと設定の両方が完全性で守られる。`predict` は metadata の設定を正とし、`adapter_config.json` は人間向けの写しで、metadata と一致しなければ終了コード 64（写しの書き換えだけでは設定を変えられない）。あわせてベースモデルと config・tokenizer の sha256（metadata の値と指定した期待値の両方）・ラベル順・system プロンプト・dtype・max_seq_length を照合し、重みの構造・形状・dtype（float32 / bfloat16 / float16 のみ）・有限性を検査する。JSON は NaN・Infinity・重複キーを拒否する。

## 10. 出力の確定と失敗後の再実行

- `--out-dir`（`probe` は `--out`）は**存在しない**ことを要求し、親ディレクトリは存在している必要がある（`mkdir -p` はしない）。
- `train`・`predict` は同じ親に `.<名前>.tmp-XXXX`（0700）の一時ディレクトリを作り、各ファイルをそのディレクトリ fd 起点で排他作成（0600）して、全出力が揃ってから `out-dir` へ確定する。確定は、宛先を `mkdir`（0700）で排他作成してからその空ディレクトリを `rename` で置き換える（POSIX の rename は宛先が空のときだけ置き換えるので、間に他者が宛先へファイルを置いても上書きせず失敗する。既存の宛先は mkdir で 64）。採点中などの失敗では一時ディレクトリを消すため、半端な出力は残らず、**同じ `--out-dir` でそのまま再実行できる**。プロセスの強制終了（SIGKILL）では一時ディレクトリが残りうる（隠し名のため手で消す）。
- 入力の読み込み・検証とモデル・LoRA の準備をすべて終えてから一時ディレクトリを作る。既存の `out-dir` は上書きしない（終了コード 64）。
- `probe` は出力ファイルを親ディレクトリ fd 起点で排他作成（0600）する。
- 採点・probe の壁時計上限は、各 forward の**後**にも確認する（最後の 1 件で超過しても終了コード 20）。
