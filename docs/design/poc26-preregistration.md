# PoC-26 事前登録（確定 2026-10-06）

対応: REQ-41・TASK-41.1（PoC-26「公開モデルの追加学習と現行候補の比較」検証方法 1）・M12・issue #386。
spec の内容は要約であり、詳細は spec の `02-poc-plan.md` PoC-26 節・`04-requirements.md` REQ-41・`05-tasks.md` TASK-41.1・`06-roadmap.md` M12 を参照する。様式は PoC-10 の事前登録（spec `03-poc/scratch-classifier/preregistration.md`）に合わせる。

**状態: 確定（2026-10-06、オーナー判断）**。Agent が判断材料として下書きし、オーナーが下の「確定記録」のとおり確定した。以後は本文を変えず、変更は追補ファイル（`poc26-preregistration-addendum-N.md`）に理由と日時を書く。

## 0. 確定記録

| 項目 | 記入 |
| ---- | ---- |
| 確定日 | 2026-10-06（評価データはまだ存在せず、どのモデルもどの評価データにも当てていない） |
| 確定者 | yu-doly（オーナー） |
| 確定した Playwright MCP の版 | `@playwright/mcp` 0.0.83 |
| 確定したラベル集合 | 案 A（既定ツール 25 ＋ `none` ＝ 26。3 節の共通の規則を含む） |
| Holm の族 | seed ごとに m=3（P 対 majority・C1・C3）。autoregressive は参考 |
| ベースモデル | Qwen2.5-0.5B-Instruct（ライセンスの最終判断は #387） |
| 既定案のまま登録した項目 | 9 節のすべて |
| 本文のハッシュ | 本書内に書くと自己参照になるため、確定コミット後に `shasum -a 256` の値を issue #386 へ記録する |

- 固定の時点: 学習・評価データの用意（#387）より前。
- 確定より後に評価データを作る場合、評価データの凍結ハッシュ（#388）は本書ではなく PoC-26 の結果記録（#394・`docs/design/poc26-result.md`）に書く。

## 1. 題材

- 入力: 日本語の作業内容（1 件 1 文〜数文）。出力: 呼ぶべき Playwright MCP のツール名 1 つ、または `none`。
- 判定型は選択肢から 1 つを選ぶ単一ラベル（REQ-15）。自由生成はしない（追加学習の候補も選択肢の採点で選ぶ。REQ-19b の対応づけ (b) と同じ考え方）。
- `none` は「対象とする版のどのツールにも対応しない作業内容」で、対象外入力（REQ-22）の受け皿を兼ねる。

## 2. Playwright MCP の対象版（案）

| 項目 | 内容 |
| ---- | ---- |
| パッケージ | `@playwright/mcp`（microsoft/playwright-mcp） |
| 版（確定） | **0.0.83**（npm の `latest`。`next` は 0.0.79-alpha 系のため採らない） |
| 公開日 | 2026-09-28（GitHub Release `v0.0.83` の `published_at` 2026-09-28T23:17:11Z） |
| ライセンス | Apache-2.0（tag v0.0.83 の `package.json`。ツール名の一覧を本リポに載せるだけで、コードは同梱しない） |
| 取得日・出典 | 2026-10-06 に読み取りのみで確認。npm registry（https://registry.npmjs.org/@playwright/mcp の dist-tags）、https://github.com/microsoft/playwright-mcp/releases/tag/v0.0.83、https://raw.githubusercontent.com/microsoft/playwright-mcp/v0.0.83/README.md |

- 版を固定する理由: ツール集合が版で変わるため（spec REQ-20 の → 2026-10-06 の注記）。PoC-26 では版の変化は扱わない。
- 確定までに新しい版が出た場合は、オーナーが確定時の `latest` に差し替えてよい（ラベル集合の案も合わせて再確認する）。
- パッケージの取得・実行はしていない（`npm install`・`npx` なし）。ツール名は README の記載で、ソースコードとの照合は**未確認**（tag v0.0.83 の `src/` には README.md しか見当たらず、実装の置き場所を特定していない）。#387 のデータ作成前に、実環境で 0.0.83 を起動して `tools/list` の結果と README を照合し、差異があれば評価データに当てる前に追補で記録する。

## 3. ラベル集合（案）

tag v0.0.83 の README「Tools」節の分類に従う（README の勘定: 全ツール 72 ＝ 既定 25 ＋ capability 付き 47）。README には `--caps` の説明「possible values: vision, pdf, devtools」と、節ごとの `--caps=config`・`network`・`storage`・`testing` の表記が併存し、**整合は未確認**。#387 のデータ作成前に実環境の `tools/list` で照合し、差異があれば評価データに当てる前に追補で記録する。

### 案 A: 既定で有効なツールのみ＋`none`（26 ラベル。**確定**）

Core automation（24）＋ Tab management（1）＋ `none`。

| 分類 | ツール名 |
| ---- | -------- |
| Core automation（24） | `browser_click`・`browser_close`・`browser_console_messages`・`browser_drag`・`browser_drop`・`browser_emulate_media`・`browser_evaluate`・`browser_file_upload`・`browser_fill_form`・`browser_find`・`browser_handle_dialog`・`browser_hover`・`browser_navigate`・`browser_navigate_back`・`browser_network_request`・`browser_network_requests`・`browser_press_key`・`browser_resize`・`browser_run_code_unsafe`・`browser_select_option`・`browser_snapshot`・`browser_take_screenshot`・`browser_type`・`browser_wait_for` |
| Tab management（1） | `browser_tabs` |
| 対象外（1） | `none` |

採用の理由:

- 利用者が追加設定なしで使う集合で、題材（日本語の作業内容から呼ぶツールを返す）の既定の利用場面に対応する。
- 26 ラベルは PoC-10 の題材 A（9 ラベル。spec `03-poc/scratch-classifier/README.md`）より多く、同じ評価件数なら 1 ラベルあたりの件数は少なくなる。それでも案 B（73 ラベル）より必要件数（5 節）とデータの用意（#387）の負担が小さい。
- capability 付きのツール（案 B）は名前だけで区別しにくい類似対（例: `browser_click` と `browser_mouse_click_xy`、`browser_find` と `browser_verify_*`）を増やし、品質差の比較より「ラベル定義の曖昧さ」が結果を左右しやすい。

### 案 B: 全 capability 込み＋`none`（73 ラベル。不採用）

案 A に、Configuration（1: `browser_get_config`）・Network（4）・Storage（17）・DevTools（13）・Coordinate-based〔vision〕（6）・PDF（1）・Test assertions〔testing〕（5）の 47 件を加える（26 ＋ 47 ＝ 73）。ツール名は 2 節の README を参照。

- 向く場合: 利用者が capability を有効にした環境を想定するとき。
- 欠点: ラベル数が約 3 倍になり、Storage の get／set／delete／clear／list のような対（5 × 3 系統）が入る。必要件数（5 節）とデータの用意（#387）の負担が大きい。

### 案 C: 案 A ＋ 特定 capability（折衷。不採用）

案 A に vision（6）か testing（5）だけを加える案。

### 共通の規則

- 本 PoC の結論は、0.0.83 の既定ツール集合（25）＋ `none` に限る。`--caps` で有効になるツールに対応する作業内容は `none` とする。capability 付きへの拡張は別版の PoC（新しい事前登録・新しい凍結 test）で検証し、本 PoC の結果と混ぜない。
- ラベル ID はツール名をそのまま使い、`none` を最後に置く（定義ファイルの宣言順。majority の同数時の優先順に使う）。
- 評価データ・学習データに、対象版に存在しないツール名を入れない（データ検査 REQ-16 で選択肢外のラベルは不正とする）。
- 作業内容が複数ツールの連続（例: 開いてからクリック）を含む場合、**最初に呼ぶツール**をラベルとする。データ作成の指示文（#387）にこの規則を含める。

## 4. 候補と各候補の構成の決め方

| ID | 方式 | 学習経路 | 入力表現 | 調整する範囲（seed 0 の validation で選ぶ。**[仮定]**） |
| -- | ---- | -------- | -------- | ------------------------------------------------- |
| C1 | バイト n-gram TF-IDF＋ロジスティック回帰（TF-IDF は ONNX グラフ内） | CLI `train`（TASK-33.x の配線後） | byte | `kind` の既定の探索範囲（trainer `kinds/c1.py`） |
| C3 | バイト CNN | CLI `train` | byte | 同上（`kinds/c3.py`） |
| autoregressive | バイト単位の小型自己回帰 decoder（対応づけ (b)） | CLI `train` | byte | 同上（`kinds/autoregressive.py`） |
| P（追加学習） | 公開モデル＋LoRA、選択肢の採点で選ぶ | PoC 用スクリプト `trainer/tools/poc26/`（#390。PoC-21 の P4 の経路を参考） | ベースモデルのトークナイザー | iters ∈ {300, 600}、lr ∈ {1e-4, 5e-5}（PoC-21 の P4 と同じ格子。rank 等は #390 で固定して追補に記録） |

- 追加学習のベースモデルは Qwen2.5-0.5B-Instruct（確定。spec PoC-26 の初期例）。ライセンスの最終判断・ローカルへの用意は人間の担当（#387）で、本ツールはダウンロードしない（REQ-38）。
- ベースモデル候補の比較調査（2026-10-06、Hugging Face の一次情報）の要約: Qwen2.5-0.5B-Instruct は Apache-2.0・標準の qwen2 アーキテクチャ・BPE の `tokenizer.json`・bf16 988MB・日本語対応を明記・PoC-21 の P4 の経路を流用できるため維持。Qwen3-0.6B（1.5GB）・sarashina2.2（SentencePiece・1.59GB）・llm-jp-3-440m（Unigram）・Gemma／LFM（独自ライセンス）・Qwen3.5／Gemma 4（構造が特殊）は不採用。副候補は ibm-granite/granite-4.0-350m（Apache-2.0・705MB。mlx-lm の LoRA は未検証）。bf16 988MB は読み込むファイルの上限 1GB（REQ-39・#400）とほぼ同じで、書き出し形式しだいで超えうる。この点は #400 の判断材料とする。
- 下限基準: majority（学習データの最頻ラベル。同数は宣言順。`crates/eval` の `fit_majority`）。
- **代表構成の決め方**: 各候補で、seed 0 の validation 正解率が最も高い設定を代表構成とする（同値は Macro-F1、さらに同値は配布パッケージが小さい方）。seed 1・2 は代表構成のまま学習する。
- 候補と下限基準は train だけで学習し、validation は調整にだけ使う。
- **凍結 test への適用は候補 × seed ごとに 1 回限り**（REQ-27）。test の結果を見て構成・データ・規則・ラベル集合を変えない。
- 学習・調整の予算は 1 構成・1 seed あたり 1 時間（REQ-18 の既定。到達した構成は「予算到達」と記録し、その時点の最良で評価する）。

## 5. seed・指標・有意性

- seed: **0・1・2** の 3 つ（REQ-26。すべての乱数を seed で固定。MLX の GPU 学習は同一 seed でも完全再現しないため、決定性の確認は 7 節）。
- 主指標: 正解率（全件を分母。abstain・error を不正解に数える）。
- 併記: Macro-F1、ラベル別 precision／recall／F1、混同行列、件数、Wilson 95% 区間（`crates/eval` の `metrics`・`wilson`）。分母が 0 の指標は `null` とし、平均から除外する（0 や 1 で埋めない）。
- majority との比較: seed ごとに、各候補の代表構成と majority を同じ凍結 test で McNemar 正確検定（両側）で比べる。b＝候補のみ正解、c＝majority のみ正解。「有意に上回る」は b > c かつ p < 0.05（`crates/eval` の `significance::judge`）。評価件数が必要件数未満なら**判定不能**（`Undeterminable`）とし、合格扱いにしない。
- 候補間の比較: 追加学習の候補 P と C1・C3（spec REQ-41 の受け入れ基準の相手）を、seed ごとに「P 対 majority・P 対 C1・P 対 C3」の **m=3** を 1 つの族として Holm 補正する（`holm::compare_candidates_with_holm`）。脱落した候補があっても m=3 のまま補正する（保守側）。autoregressive は参考として P 対 autoregressive の生の p も記録するが、族に入れない（確定）。
- **「有意に上回る」の成立**: Holm 補正後 p < 0.05 かつ b > c が **3 seed すべて**で成り立つこと（PoC-10・PoC-21 と同じ規則）。
- 再現性: 候補ごとに 3 seed の正解率の Wilson 95% 区間の重なりを判定する（`reproducibility`。全ペアが重なれば再現性あり）。
- 必要件数: `crates/eval` の `required_sample_size_mcnemar`（PoC-10 と同じ仮定: 不一致の割合 b 側 0.15・c 側 0.05・検出力 0.8）を、α=0.05 と Holm の最も厳しい段（α=0.05/3）の両方で算出し、後者を凍結 test の下限とする。算出値は #388 の時点で結果記録に書く（**[仮定]**。PoC-10 の m=4 では 221 件）。
- 推論の一致: 現行候補は 1 件ずつの推論とバッチ推論の予測ラベルが全件一致すること（REQ-28）。追加学習の候補は PoC 用スクリプトのため対象外とし、その旨を記録する。
- P の p95: 選択肢数 K=26 での値として記録し、K に対する伸び（1 件あたりの採点回数）を併記する。ラベル数の多い用途への外挿は線形の見積もりに留める（6 節の成功基準 2）。

## 6. 成功基準との対応

| PoC-26 成功基準（暫定） | 本書での確認方法 | 担当 |
| ----------------------- | ---------------- | ---- |
| 1. 品質差を同じ評価器・同じ凍結 test で、信頼区間と有意性付きで示す | 5 節（正解率・Wilson・McNemar・Holm を候補 × 3 seed で記録） | Agent（#389・#391） |
| 2. 追加学習の候補の容量・RSS・p95 を目安・上限との関係で示す | 容量（非圧縮の合計バイト。目安 40MB との差。超過は警告。10 進 MB）・RSS（上限 2GB）・推論のみ p95（250ms 未満）を証拠の種別付きで記録 | 人間（#393。Agent は手順の準備まで） |
| 3. 書き出しと自作の推論バックエンドに要る演算・トークナイザーの範囲を示す | ONNX への書き出し可否、演算の一覧と `crates/runtime/src/onnx/` が扱えない演算、トークナイザーの種類 | Agent（#392） |
| 4. REQ-41 の受け入れ基準と初期アーキテクチャの材料を出す | 1〜3 を `docs/design/poc26-result.md` にまとめる | 共同（#394） |

- 追加学習の候補が現行候補を有意に上回ることは、成功基準でも REQ-41 の受け入れ条件でもない（spec REQ-41・M12 の完了基準）。上回らなかった場合の扱いはオーナーが判断する。

## 7. 評価データの条件

- データは利用者が用意する（#387）。LLM による言い換えの生成は可とし、生成元・指示文・利用条件を来歴（REQ-40）に記録する。Jev の出力は使わない。
- 分割はテンプレート単位の group で行う（REQ-17。同じテンプレートの言い換えが train と test に跨らない）。評価データが学習データとテンプレートを共有しないなら、別 group として扱う。
- `inspect`（#388）でデータ検査（REQ-16。選択肢外ラベル・重複・group の跨ぎ）を通し、group 分割の seed・規則・各分割のハッシュを記録して test を凍結する。ハッシュ不一致は停止（fail-closed）。
- 推論関数には `input` だけを渡す（REQ-27）。
- `none` のレコードには、`input`・`label` とは別のメタデータ欄（例 `none_reason`。値は `capability:<ツール名>` または `out_of_scope`）を付けて来歴に残す。推論関数へは渡さず（REQ-27）、`input` 本文に混ぜない。本 PoC の評価・学習では使わない。将来この欄で再ラベルした凍結 test は新しい PoC 版として扱い、旧版の結果を見て候補・構成を選び直さない。`none` の割合と内訳（`none_reason` の集計）は結果記録（#394）に書く。
- データ本文を本書・結果記録・issue・PR へ転記しない。記録してよいのは件数・ラベル分布・ハッシュ・group 数だけ。
- 決定性: 代表構成を seed 0 でもう一度学習し、validation の予測ラベルが全件一致するかを確かめる（凍結 test は使わない）。不一致は件数を記録し、GPU の非決定性を許容幅の拡大で吸収しない。決定的な再現が要る確認は CPU で行う。

## 8. 証拠の種別・中止・変更

- 記録する数値には証拠の種別（テストハーネス／模擬／推定／実機）を必ず付ける。学習と評価は実機（Mac）。Zenn 記事の RSS 約 571.7MB と Jev の公表値（E2E 70〜500ms）は外部の公表値で、比較対象ではなく参考値。
- 中止: ベースモデルのライセンスが用途に合わないと分かった候補はその時点で外す。損失の発散が seed を変えて 3 回続いた候補は「不成立」と記録する。
- 判定不能: TASK-33.x の配線がない、データまたはベースモデルを用意できない、凍結 test が必要件数未満、評価器が動かない。
- 変更: 確定後の変更は本文を書き換えず、追補ファイルに理由・日時・変更前後を書く。評価データに当てた後の変更は、新しい版の PoC として扱い、旧版の結果と混ぜない。

## 9. 既定案で登録した項目（2026-10-06 にオーナーがすべて採用）

- Playwright MCP の版: 0.0.83
- ラベル集合: 案 A（26 ラベル）
- 追加学習の候補: Qwen2.5-0.5B-Instruct＋LoRA、格子 iters ∈ {300, 600} × lr ∈ {1e-4, 5e-5}
- seed: 0・1・2。すべての seed で有意であることを求める
- Holm の族: seed ごとに m=3（P 対 majority・C1・C3）。autoregressive は参考
- 必要件数の仮定: PoC-10 と同じ（b 側 0.15・c 側 0.05・検出力 0.8）

## 10. オーナーが確定した事項（2026-10-06、yu-doly）

1. Playwright MCP の版: 0.0.83（2 節）
2. ラベル集合: 案 A ＋ 3 節「共通の規則」の capability 付きツールの扱い。README とソースの整合は #387 の前に `tools/list` で照合（2・3 節）
3. ベースモデル: Qwen2.5-0.5B-Instruct。ライセンスの最終判断は #387（4 節）
4. Holm の族: m=3。autoregressive は参考（5 節）
5. 既定案（9 節）はすべて採用
6. 本文ハッシュは確定コミット後に issue #386 へ記録（0 節）
