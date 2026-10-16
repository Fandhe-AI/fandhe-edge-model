# Qwen2 採点グラフの ONNX 演算一覧

対応: REQ-41・TASK-41.7-1・M12・issue #415（依存 #394。前提の決定 #405・#409・#392）。関連 REQ: REQ-28（推論の一致）・REQ-32（学習非依存の推論）・REQ-39（ガード層）。

- 本書は自作の推論バックエンド（`crates/runtime/src/onnx/`）が Qwen2 系の採点グラフに対して許す演算・属性・データ型・rank の一覧と、範囲外を除外する方針を定める。**一覧が唯一の正**で、変更は本書の更新を先に行う
- 本書は文書のみ。コード・fixture・`docs/spec` は変更しない。演算の実装は #416〜#419・#581、書き出しは #431、除外理由の実装は #433
- 証拠の種別: 本書の構成は設計（参照グラフ）。ReferenceEvaluator 等による数値照合は**未検証**で、#431 で確認する。件数・容量の数値は「推定」と明記する
- autoregressive（#420・REQ-19b）の演算は対象外。本書の一覧に入れない

## 1. 前提と範囲

- 導出元は Qwen の KV 使い回し採点グラフ（#405 の案 A）。PoC-26 の書き出し（`trainer/tools/poc26/export_onnx.py`）は `logits` を出すだけで採点を含まず、未対応 11 種（Cos・MatMul・Range・Reciprocal・ReduceMean・Reshape・Shape・Sigmoid・Sin・Split・Where。#392）だけでは一覧は確定しない
- 入出力契約は現行のまま: `ids: INT64 [N,T]` → `probs: FLOAT [N,K]`。N は常に 1
- 採点（ラベル token 列と `<|im_end|>` の対数尤度の合計 → 選択肢間 softmax。`trainer/tools/poc26/score.py` と同じ）はグラフ内で行う。選択肢 token 列は定数で持ち、特殊トークンは id で挿入する（#409）
- 対応する config は `qwen2_model.py` の `_SUPPORTED` と同じ: 密結合・`hidden_act=silu`・`rope_scaling` なし・sliding window なし・tie と非 tie の両方。例: 0.5B（24 層・kv 2）、1.5B（28 層・kv 2・head_dim 128・vocab 151936・tie）、7B（非 tie・`lm_head` あり）
- opset 13（既定 domain のみ）、IR version 8（`EXPECTED_OPSET`・`EXPECTED_IR_VERSION` と同じ）
- 数値: 計算は f32。重みは bf16 で保存し f32 で計算する。出力直前の Softmax だけ f64（C3 と同じ）

## 2. 採点グラフの構成（演算を導く参照設計）

隠れ状態は 2 次元 `[rows, H]` で持つ。prefix は `rows=T`、選択肢ブロックは `rows=K*M`。線形層がすべて `Gemm(transB=1)` になり、重みを HF の (out, in) のまま bf16 で持てる（転置コピーを作らない）。

1. 共通の前処理: `Reshape(ids,[-1])`。`Shape`・`Gather`（INT64 の形状から INT64 スカラー T を取る。出力は INT64 のまま `Range(0,T,1)` へ渡す）で位置を作り、`Cast`（INT64 → FLOAT）・`Unsqueeze`・`Mul(inv_freq)` から `Cos`・`Sin` で RoPE 表を作る。prefix の因果マスクは `Unsqueeze`・`Greater` → `Where(future, -1e30, s)`（Trilu は opset 14 のため使わない）
2. 埋め込み: 論理グラフは `Gather(Cast(embed_bf16), ids)`（BFLOAT16 は initializer と Cast の入力にだけ現れ、Gather の出力は FLOAT）。実行時は Cast と Gather を融合し、参照された行だけを f32 に展開する。埋め込み全体は f32 に展開しない
3. 各層（prefix）
   - RMSNorm: `Mul(x,x)` → `ReduceMean` → `Add(eps)` → `Sqrt` → `Reciprocal` → `Mul` → `Mul(Cast(w_bf16))`
   - q・k・v: `Gemm(x, Cast(W_bf16), Cast(b_bf16), transB=1)`。o・gate・up・down も同じ（bias なし）
   - heads: `Reshape` → `Transpose`。GQA は q を `[1,nkv,grp,T,hd]`（5 次元）、k・v を `Unsqueeze` で `[1,nkv,1,T,hd]` にし、MatMul のバッチ方向ブロードキャストで扱う
   - RoPE: `Split`（等分 2 つ）→ `Mul`・`Sub`・`Add` → `Concat`
   - 注意: `Transpose(k)` → `MatMul` → `Mul(scale)` → `Where` → `Softmax` → `MatMul(v)` → `Transpose` → `Reshape`。残差は `Add`
   - SwiGLU: `Sigmoid` → `Mul`（SiLU）、`Mul(up)` → `Gemm(down)`
   - 各層で RoPE 後の K と V をブロック側へ渡す
4. 選択肢ブロック（K 件。長さ M に右 pad した定数 `cand: INT64 [K,M]`。pad は `endoftext`）
   - 位置は `Add(Cast(T), arange(M))`（arange は定数）から RoPE 表
   - prefix の KV を `Expand` で K 件に広げ、ブロックの KV と `Concat`（seq 軸）で連結する
   - スコアに加算マスクを足す。prefix は全位置を見せ、ブロック内は因果（`Where`）。pad は実 token の後ろにあるため、因果マスクで実 token からは見えない
   - 残りは prefix の層と同じ演算
5. 採点
   - 予測に使う行: prefix の最後の行（`Slice` → `Expand` で K 件）とブロックの先頭 M-1 行（`Slice`）を `Concat` → `Reshape([K*M,H])`
   - 最終 RMSNorm → `Gemm(h, Cast(embed_bf16 または lm_head_bf16), transB=1)` → `Reshape([K,M,V])`。tie でも非 tie でも同じ形で、Transpose は要らない
   - `LogSoftmax` → `GatherElements(index=cand に次元を足した [K,M,1])`（行 j の予測対象は cand[j]。上の「予測に使う行」で prefix の最終行とブロック先頭 M-1 行を連結済みのため、ずらさない。`score.py::score_chunk` が位置 p-1+j に対し c[j] を採点するのと一致。REQ-41） → `Mul`（有効位置の重み定数。pad 位置は 0）→ `ReduceSum` → `[K]`
   - `Unsqueeze` → `Softmax` で `probs [1,K]`。この Softmax だけ f64
   - logits は `K*M` 行だけ計算する（PoC の全 T 行より少ない）
6. 不採用の案
   - OneHot＋ReduceSum による対数尤度の取り出し: GatherElements の 1 演算で足りる
   - スコアを Concat して KV の Expand を避ける案: 演算が増える。Expand のコピーは層ごとに一時的（推定: 数十 MB 程度）
   - `MatMul(h, Transpose(embed))`: 実行時に転置コピー（約 0.9 GB。推定）が要る
   - system プロンプト部の KV を書き出し時に定数化する案（#405）: 対象外・将来

## 3. 演算一覧（opset 13）

属性・入力の扱いは opset 13 に従う（axes を属性で受けるか入力で受けるかは演算ごとに違う）。

| 演算 | 用途 | 許す範囲 | 担当 |
| ---- | ---- | -------- | ---- |
| MatMul | 注意のスコアと文脈 | FLOAT。rank 2〜5。バッチ次元は numpy 方式でブロードキャスト | #416 |
| Gemm | 線形層・LM head | A は 2 次元 FLOAT。B は `Cast(BFLOAT16 の initializer)`。`transA=0`・`transB=1`・`alpha=1.0`・`beta=1.0`。C は省略可で、ある場合は 1 次元 | #416 |
| Add / Sub / Mul | 要素ごと | FLOAT（INT64 の Add は位置計算に限る）。多方向ブロードキャスト | #416 |
| Sqrt / Reciprocal | RMSNorm | FLOAT | #416 |
| Cast | bf16 の重みの展開・位置の型変換 | `to=FLOAT` だけ。入力は BFLOAT16（initializer のみ）か INT64 | #416 |
| Shape | T の取得 | 属性 start・end は使わない | #417 |
| Gather | 埋め込み・スカラー T | `axis=0`。data は FLOAT か `Cast(BFLOAT16 の initializer)`（埋め込み。2 節の融合実行）か、Shape の出力である INT64（形状取得用。出力は INT64 のまま Range へ渡す）。index は INT64 で、範囲外は拒否 | #417 |
| GatherElements | 目標 token の対数確率 | `axis=-1`。index は INT64 で、範囲外は拒否 | #417 |
| Reshape | heads・平坦化 | shape は INT64 の入力。0 と -1 の意味は opset 13 のとおり | #417 |
| Transpose | heads・k の転置 | `perm` は rank と一致する順列 | #417 |
| Unsqueeze | 次元の追加 | axes は入力（opset 13） | #417 |
| Concat | RoPE・KV の seq 連結・行の連結 | axis を明示する。入力数に上限 | #417 |
| Slice | 最後の行・ブロックの前半 | starts・ends・axes は入力。steps は省略か 1 | #417 |
| Split | RoPE の半分割 | `axis=-1`。split は入力で、等分 2 つ | #417 |
| Expand | prefix の KV・最後の行を K 件に複製 | shape は INT64 の入力 | #417 |
| ReduceMean | RMSNorm | axes は属性で `[-1]`。`keepdims=1` | #418 |
| ReduceSum | 対数尤度の合計 | axes は入力（opset 13）。`keepdims=0`。`noop_with_empty_axes=0` | #418 |
| Sigmoid | SiLU | FLOAT | #418 |
| Softmax | 注意・最終出力 | `axis=-1`。グラフ出力を作る Softmax だけ f64 | #418 |
| LogSoftmax | 採点 | `axis=-1`。max を引く数値安定化を行う | #418 |
| Range | 位置 | INT64 のスカラー start・limit・delta。要素数に上限 | #419 |
| Greater | 因果マスク | INT64 の入力。BOOL を出力 | #419 |
| Where | マスク | cond は BOOL、値は FLOAT | #419 |
| Sin / Cos | RoPE | FLOAT | #419 |

- 合計 28 種。内訳: PoC の未対応 11 種、既存の C1・C3 にある 11 種（Add・Mul・Gemm・Gather・Softmax・Cast・Concat・Transpose・Unsqueeze・Sub・ReduceSum）、新規 6 種（Slice・Expand・GatherElements・LogSoftmax・Sqrt・Greater）
- 既存の C1・C3 にある演算も、汎用の形（ブロードキャスト・高い rank）では実装し直す。C1・C3 の既存テンプレート照合と上限は変えない
- 実装者向けの確認: 2 節の各手順で使う演算の和集合と本表が一致すること。autoregressive 専用の演算（Erf・CumSum・And・Equal・Less・GreaterOrEqual・Div 等）は本表に無い

## 4. データ型・rank・件数

- FLOAT: 計算。INT64: id・形状・index。BOOL: 中間値に限る（Greater の出力・Where の cond。initializer には使わない）。BFLOAT16: initializer に限る（`proto.rs` に `DT_BFLOAT16 = 16` を足す。#581）。それ以外（FLOAT16・DOUBLE・INT32 等）は拒否
- bf16: `Cast(BFLOAT16 の initializer)` の消費者が Gemm の B か Gather の data のときは、読み込み時に f32 の全量を作らずカーネル内で展開する。norm の重みなど小さいものは読み込み時に f32 へ畳んでよい。全体は f32 に展開しない（1.5B で約 6.2 GB〔推定〕となり RSS 安全上限 8 GiB に近いため。#581 の決定）。bf16 は `u16` のまま保持し、外部データは shard 単位で流し読みして生バイトの全量を持たない。RSS の実測は #421・#435
- rank: 中間値は 5 以下（GQA の `[1,nkv,grp,T,hd]`）。initializer は 2 以下。C1・C3 の `MAX_DIMS=4` は変えず、上限は種類ごとに持つ
- 件数: 1 層あたりのノードは 2 節から約 50〜60、initializer は約 12（q・k・v の重みと bias、o・gate・up・down、norm 2 つ）。28 層で数千ノード・数百 initializer になる（推定。#431 で実測）。現行の `MAX_NODES=128`・`MAX_INITIALIZERS=64` には収まらないので、種類別の上限が要る。#581 の暫定値（推定。#431 の実測で見直す）: ノード 4096・initializer 1024・rank 5・本体 64 MiB、外部データは shard 1 GiB・256 件・合計 8 GiB（ガード層の #580 と同じ）。C1・C3 は 128・64・4・44 MiB のまま外部データと bf16 を拒否する
- 動的な次元は T だけ（`max_seq_length` と `max_position_embeddings` で上限を持つ）。K と M は定数

## 5. 範囲外を除外する方針

- 学習側（#431・#433）: 書き出した ONNX の演算・属性・dtype・rank・opset が一覧の範囲外なら、配布候補から外して理由を記録する（TASK-32.2 の仕組み。`export_exclusion.rs` の理由に未対応演算を足す）。config の範囲外（rope_scaling・sliding window・silu 以外）は書き出しの前に止める。#431 の受け入れテストは本書の表を基準にする
- 推論側（#416〜#419・#581）: 読み込み時に許可リスト（演算 × 属性 × dtype × rank）で照合し、外れたら fail-closed で拒否する（`invalid_input`=64。上限超過は `limit_exceeded`=20。REQ-39）。既定以外の domain・部分グラフ属性・opset ≠ 13 も拒否する
- 資源の上限: T・Range の要素数・Concat の入力数・ノード数・initializer 数・rank を許可リストの一部として持つ。上限の無いアロケーションを作らない
- 完全性: 外部データ（bf16 の重み）は経路の閉じ込めと sha256 照合を通す（#581）。入力 id が語彙外なら Gather で拒否する

## 6. 数値と決定性

- 行列積の縮約順はスレッド数に依存させない（REQ-28。#416）
- 演算の単体テストの許容差は各実装 issue に任せる（1e-9 は f64 で計算する場合に限る）
- 1 件ずつの推論とバッチ推論の全件一致（REQ-28）は、N=1 固定の契約で保つ
