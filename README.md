# fandhe-edge-model

用途専用の小型ローカル判定モデルを作成・評価するツールの実装リポジトリです。利用者が用意した選択肢（ラベル）・入出力構造・学習データ・独立した評価データから、数十 MB 以下を志向する判定モデルを作り、Claude Code・Codex などから呼び出せるようにすることを目指します。

## 位置づけ

- **本リポジトリは public** です（OSS として公開する方針）
- **仕様・要件定義**: [fandhe-edge-model-spec](https://github.com/Fandhe-AI/fandhe-edge-model-spec)（`docs/spec` に submodule 参照。**private リポジトリとして意図的に非公開を維持**する方針であり、アクセス権のない環境からは submodule を解決できません）

## ステータス

実装は進行中です（CLI 7 工程は結線済み。進捗は Issue で管理します）。要件は spec リポの [`04-requirements.md`](https://github.com/Fandhe-AI/fandhe-edge-model-spec/blob/main/04-requirements.md)（REQ-15〜40）、タスク定義は [`05-tasks.md`](https://github.com/Fandhe-AI/fandhe-edge-model-spec/blob/main/05-tasks.md)（87 件・35.0 人日）、マイルストーンは [`06-roadmap.md`](https://github.com/Fandhe-AI/fandhe-edge-model-spec/blob/main/06-roadmap.md)（M6〜M11。初期スコープは M6〜M10）を参照してください。

## 実装方針（要点）

- **薄い統合ツール**: 定義・データ検査・評価契約・成果物管理・CLI・推論ランタイムを自作し、学習は差し替え可能なワーカー（MLX など、または Rust の candle／burn）として呼び出します（暫定の推奨構成）
- **推論は学習に依存しない**: 推論ランタイムと配布パッケージに学習側の依存を持ち込みません
- **モデルの種類の選択口**: 判別型・生成型などの種類を同じ選択口から選べるようにし、種類は PoC の結果で増やします
- **評価契約**: 評価データの凍結・ハッシュの不変・下限基準に対する有意性の判定を、評価器の正しさとあわせて保証します
- **入力表現**: byte のみで確定しています
- **呼び出し口**: CLI＋JSON を基本とします。ローカル MCP・Codex からの呼び出しも要件に含みますが、実クライアントとの接続は初期スコープ外です
- **ローカル完結**: 推論・学習・評価はローカルで完結させます
- **実装言語の境界**: 共通コアを Rust 中心とする案を暫定で推奨しています。学習側をどこまで Rust に寄せるかは未決です

詳細な要件は spec リポの [`04-requirements.md`](https://github.com/Fandhe-AI/fandhe-edge-model-spec/blob/main/04-requirements.md) を唯一の正（SSOT）とします。

## `evaluate` 工程の出力（REQ-24・REQ-27・REQ-33・#314）

`fandhe-edge evaluate`（`--candidate <N>`）は、凍結した評価データへ選定済み候補を **1 回だけ** 適用し、JSON を 1 つ出力します。出力スキーマは 2026-09-30 にオーナー承認済みです（入出力契約への加算的な追加）。

評価データがあり評価が完了した場合（exit 0）:

```json
{"step":"evaluate","status":"ok","candidate":0,"kind":"c1","n_total":100,"correct":90,"accuracy":0.9,"macro_f1":0.88}
```

| フィールド | 意味 |
| ---------- | ---- |
| `candidate` | 評価した候補の番号 |
| `kind` | 候補のモデルの種類 |
| `n_total` | 評価データの件数（分母。0 件は完了にならない） |
| `correct` | 正解した件数 |
| `accuracy` | 正解率（`correct / n_total`） |
| `macro_f1` | Macro-F1（評価器で算出。分母 0 のラベルは平均から除外し、全ラベルで未定義なら `null`） |

評価データが未定義のときは `{"step":"evaluate","status":"skipped",...}`（exit 0）で、評価済みを装いません。成功時は `candidates/<N>/evaluation_record.json` を書き、`package` は記録と最終 test の台帳での適用完了を確認してから公開します。値は例示です。出力にはこのほか `per_label`・`type_meaning_quadrant`・`calibration`・`abstention`・`comparison`・`reproducibility`・診断レポート `diagnostics`（REQ-29・#492）が加わります。正解率の Wilson 区間は評価の出力には含めず、`package` の合否照合で使います。

定義ファイルに省略可能な `baseline_comparison`（例 `{"assumed_p_b_bp":1500,"assumed_p_c_bp":500,"power_bp":8000}`）を書いたときだけ、`evaluate` は下限基準（majority）との McNemar 比較を行い、結果（`majority_label`・`baseline_correct`・`b`・`c`・`required_n`・`verdict`）を評価記録 `evaluation_record.json` の `baseline_comparison` に残します（出力 JSON は変わりません。REQ-25・#339）。3 つの値は 1 万分率の整数で、`assumed_p_b_bp`・`assumed_p_c_bp` は 0..=10000（`p_b > p_c`・和は 10000 以下）、`power_bp` は 1..=9999、有意水準 α は 0.05 で固定です。`verdict` は `significantly_better`・`not_significantly_better`・`undeterminable`（評価件数が仮定から事前計算した必要件数に満たない）です。majority は train 分割のラベルだけから作り（validation・評価データは使わない）、最終 test の適用前に確定します。`package` は記録の比較欄を計算し直して照合し、majority・必要件数・`baseline_correct`（凍結した評価データから数え直し）・件数の整合・`verdict` の改変、欄の有無の食い違いは `invalid_input` で止めます。`b` と `c` を同じ量だけずらす改変は検出できません（候補の予測の封印記録が要る。#168 の範囲）。`verdict` は `package` の合否（`judgment`・終了コード）には使いません（記録と照合のみ。`baseline_comparison` だけの定義の `package` は `judgment:null`・`acceptance_defined:false`・exit 0 で、`acceptance` と併記しても合否は正解率だけで決まります。#344）。

定義ファイルに省略可能な `out_of_scope_label`（`options[].id` のいずれか）を書くと、`infer` は argmax がそのラベルの行の `status` を `out_of_scope` にします（行の形は `ok` と同じで、`scores` も変わりません。`JudgmentStatus` の値は `ok`・`out_of_scope`・`abstain`〔#497。下記〕）。`infer --text` はその行を stdout に出して exit 11（`out_of_scope`。`{"code","message"}` 形ではない）で終わり、`infer --input-file`（`--out` を含む）は全行を計算できれば exit 0 のままで、行の `status` で区別します（`--out` の要約は不変）。存在しない id・`null`・空文字・文字列以外は `invalid_input`、省略時は従来と同じ出力で定義の正準化ハッシュも変わりません（REQ-22・REQ-21・TASK-22.2・#478。`evaluate` は `out_of_scope_label` と `abstention` の `out_of_scope` に出します）。

評価データがあり `evaluate` が校正（温度 T・保留しきい値 τ）を記録したプロジェクトでは、`package` が `package/calibration.json`（1 行 JSON。`{"onnx_sha256","label_order","temperature","threshold"}`）を同梱し、容量内訳の `calibration` 枠に計上します（`package` の stdout の形は不変）。`infer` は `calibration.json` があれば、校正後の確信度が τ 未満の行の `status` を `abstain` にします（行の形・`scores` は `ok` と同じ。判定規則は `evaluate` と共通）。優先順は `out_of_scope` > `abstain` > `ok` で、`infer --text` の `abstain` は exit 12（`pending`）、`--input-file`（`--out` を含む）は全行を計算できれば exit 0 のままです。`calibration.json` が配布 ONNX の sha256・定義の宣言順と一致しない、T・τ が範囲外、未知キーは `invalid_input`。評価データが無いプロジェクトには同梱されず、`infer` は保留を返しません。同梱するときは配布用の `package/artifact.json` に `calibration_sha256`（`calibration.json` 全体の sha256）を追記し（候補側の `artifact.json` は変えない）、`infer` は記録があるのにファイルが無い・記録が無いのにファイルがある・sha256 の不一致を `invalid_input` で拒否します（τ の改変や削除をパッケージ内の自己整合で検出。REQ-39）。`artifact.json` ごとの改変の検出は外部台帳（#168）の範囲です（REQ-22・REQ-28・REQ-30・#497）。

`package` は配布用の `package/artifact.json` に `definition_sha256`（`package/definition.json` のバイト列の sha256）も追記し、`infer` は欄があれば `definition.json` と照合して、不一致を `invalid_input`（`package definition does not match its recorded hash`）で拒否します（欄の無い旧パッケージは照合しません）。あわせて `package` は版管理台帳 `<project>/version_ledger.json` に、配布用 `artifact.json`・`data/train.jsonl`・`selection_record.json` の sha256 を同じ版 ID（`v1`・`v2`…）で記録して読み取り専用にし、stdout の末尾に `"version":{"id","previous"}` を載せます（`--previous-project-dir OLD` で OLD の台帳を照合して引き継ぎます）。`infer --version-ledger L [--version-id V]` は配布用 `artifact.json` を台帳の model 版と照合してから推論します（前版への復帰。定義・ONNX・校正は `artifact.json` の記録を介して束縛されます。台帳ファイル自体の改変は検出しません。REQ-39・#491）。

## 開発環境構築

```bash
git clone git@github.com:Fandhe-AI/fandhe-edge-model.git
cd fandhe-edge-model
git submodule update --init   # docs/spec（private・要アクセス権）
```

`docs/spec`（`fandhe-edge-model-spec`）は private リポジトリのため、アクセス権のない環境では submodule 取得が失敗します。実装コードのビルド・テストは `docs/spec` 抜きでも成立するよう維持します。

学習ワーカー（`trainer/`）は `PYTHONPATH` の手動設定なしに、唯一の起動口 `trainer/launch.py` から起動できます（`-I` 隔離モード必須。Issue #12）:

```bash
trainer/.venv/bin/python3 -I trainer/launch.py train --request <request.json のパス>
```

## ライセンス

MIT OR Apache-2.0 のデュアルライセンスです（[LICENSE-MIT](./LICENSE-MIT) / [LICENSE-APACHE](./LICENSE-APACHE)）。
