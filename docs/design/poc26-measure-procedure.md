# PoC-26 候補 P の容量・RSS・推論のみ p95 の実機測定手順

対応: REQ-41・TASK-41.1-8・PoC-26 成功基準 2・M12・issue #393。実装は `trainer/tools/poc26/measure.py`（`predict --warmup N` は `predict.py`・`score.py`）、検査は `trainer/tests/test_poc26_measure.py`。

## 目的

追加学習候補 P（Qwen2.5-0.5B-Instruct＋LoRA）の容量・RSS・推論のみの p95 を、事前登録の基準（容量目安 40 MB・RSS 2 GB・p95 250 ms 未満）と比べて記録する。

- 担当: 人間（実機で実行する。Agent は計測スクリプトと手順の準備までに留める）
- 証拠の種別: 実機。ただし下の「静かな Mac」の申告が無い実行は参考扱い（`reference_only`）で、実機の値として確定しない（`scripts/real-machine-check.sh` と同じ運用。`docs/design/real-machine-check-procedure.md`）
- 評価データ（凍結 test）は使わない。入力は validation の入力のみ（ラベルを渡さない）

## 測定の定義（オーナー確定 2026-10-08）

| 項目 | 定義 |
| ---- | ---- |
| 容量 | MLX で推論するときに読むファイル一式の非圧縮合計バイト（10 進 MB）。base `model.safetensors`（bf16）・`adapters.safetensors`・`adapter_config.json`・`tokenizer.json`・`tokenizer_config.json`・`config.json`。目安 40 MB との差を記録し、超過は警告 |
| 容量（参考） | #392 の ONNX（`model.onnx`＋`model.onnx.data`）は `--onnx-dir` を渡したときだけ併記し、合計に含めない |
| p95 | 1 件＝1 入力について K=26 選択肢すべての採点（`score.py` の `score_labels` の計測範囲。トークナイズは含まない）。先頭 5 件（ウォームアップ）を統計から除外し、残りの全件で nearest-rank の p95。基準は 250 ms 未満 |
| 採点回数の伸び | 1 件あたりの選択肢数 K と forward chunk（8）、forward 回数（K を chunk で割り上げ）を併記 |
| 条件 | GPU・bf16 を主、CPU・bf16 を併記（同じ seed 0 アダプタ。既存アダプタはすべて bf16 学習で、`predict` の dtype 照合と整合させるため。オーナー決定 2026-10-08） |
| RSS | `/usr/bin/time -l` の maximum resident set size（モデル読み込み〜採点の実行全体）を主とし、2 GiB（ガード層の RSS 上限と同じ。REQ-39）と比べる。`run.json` の `max_rss_bytes`（`ru_maxrss` と MLX ピークの大きい方）・`mlx_peak_memory_bytes`・`peak memory footprint` を併記 |
| 注意 | RSS は読み込みと採点を分離できない（推論のみのピークではない）。`uv` 自体は `time -l` の外で起動するため、RSS は Python の子プロセスだけを表す |
| Jev の公表値 | E2E 70〜500 ms・RSS 約 571.7 MB は参考値。指標が違う（E2E と推論のみ）ため同列に比べない |

## 前提

1. 他のアプリをすべて終了し、電源に接続した静かな Mac で実行する。実行前に `uptime` の load average が低いことを確認する（記録に 1 分平均が入る）
2. ベースモデル一式（#387）と seed 0 のアダプタ（`~/poc26/work/grid_i600_lr5e-5`）、入力のみの validation（`~/poc26/work/val_input_only.jsonl`）がある
3. アダプタの dtype（bf16）と `--max-seq-length`（512）が学習時と一致すること。`predict` が `adapter_config.json` の記録と照合し、不一致は終了コード 64 で止まる（その条件は `status: error` で記録される）
4. 出力先は存在しない新しいディレクトリ（既存は上書きしない。0700 で作る）
5. 通信は発生しない（REQ-38）。Agent は実モデルでの本測定を実行しない

## 実行コマンド

`--quiet-machine` は前提 1 を満たしたときだけ付ける。付けないと `classification` は `reference_only` になる。

```bash
export PATH=$HOME/.local/bin:$PATH UV_NO_CONFIG=1
M=$HOME/poc26/qwen2.5-0.5b-instruct
A=$HOME/poc26/work/grid_i600_lr5e-5
sha(){ grep " $1\$" $M/SHA256SUMS | cut -d' ' -f1; }
ash=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['adapters_sha256'])" $A/run.json)
cd /Users/yuriyoshihara/orca/fandhe-edge-model
uv run --locked --directory trainer python -m tools.poc26.measure \
  --model-dir $M --model-sha256 $(sha model.safetensors) --config-sha256 $(sha config.json) \
  --tokenizer-sha256 $(sha tokenizer.json) --tokenizer-config-sha256 $(sha tokenizer_config.json) \
  --definition $HOME/poc26/project/definition.json \
  --adapter-dir $A --adapter-sha256 $ash \
  --input $HOME/poc26/work/val_input_only.jsonl \
  --out-dir $HOME/poc26/work/measure_$(date +%Y%m%d) \
  --evidence real_machine --quiet-machine --warmup 5
```

- `--onnx-dir <#392 の出力>` を足すと ONNX の容量を参考で併記する
- `--conditions gpu_bf16` のように条件を絞れる（既定は `gpu_bf16` と `cpu_bf16` の直列）
- 条件が失敗しても残りの条件を続行し、失敗は `record.json` の該当条件に `status: error` と固定の `code` で記録する（成功済みの条件は残る。全体の終了コードは 70）。中断（Ctrl-C）でも成功済みの条件は記録される
- 子ごとの上限時間は `--timeout-seconds`（既定 5400）。超過はプロセスグループごと止めて終了コード 70
- 出力: `record.json`（機械可読）・`record.md`（要約）と、条件ごとの `gpu_bf16/`・`cpu_bf16/`（`pred.jsonl`・`raw_scores.jsonl`・`run.json`）。`pred.jsonl` と `raw_scores.jsonl` には入力の id が入るため、リポ・Issue・PR へ転記しない。`record.json`・`record.md` には件数・ハッシュ・サイズ・秒・固定語彙だけが入り、パス・データ本文・id・子の stderr は入らない

## 合否の読み方

- 容量: `capacity.total_mb` が 40 を超えると `exceeds_target: true`（警告。0.5B の bf16 重みだけで約 988 MB のため、目安 40 MB は大きく超える見込み。値は実測で確定する）
- p95: `p95_ms < 250` なら `p95_ok`
- RSS: `rss_time_l_bytes < 2 GiB` なら `rss_ok`
- 判定は人が行い、`reference_only` の値を実機の成功基準の根拠にしない

## 記録欄

| 項目 | gpu/bf16（主） | cpu/bf16 | 基準 |
| ---- | -------------- | ----------- | ---- |
| 実施日・機種・macOS・commit | | | — |
| 証拠の種別（`classification`） | | | real_machine のみ確定 |
| load average（実行前 1 分平均） | | | 低いこと |
| 容量 合計 MB（10 進）／ 40 MB との差 | | | 目安 40 MB |
| 容量 参考: ONNX 合計 MB | | | 合計に含めない |
| 測定件数／ウォームアップ除外件数 | | | 除外 5 |
| p95 ms | | | 250 ms 未満 |
| mean ms／max ms | | | — |
| K／forward chunk／forward 回数 | | | 26／8／4 |
| RSS（`time -l`）MB | | | 2 GiB 未満 |
| 内部 max RSS MB／MLX peak MB／peak memory footprint MB | | | 併記 |
| Jev 参考値（E2E 70〜500 ms・RSS 約 571.7 MB） | 同列に比べない | | 参考 |
