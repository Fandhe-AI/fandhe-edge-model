# Mac 実機での動作確認（項目 A〜F）の手順と記録簿

対応: REQ-21・REQ-28・REQ-30・REQ-31・REQ-32・REQ-33・REQ-39（横断の確認。spec の特定の TASK には対応しない）・#354。REQ-38（sandbox 下の通信 0 件）は対象外（`sandbox-offline-check-procedure.md` の担当）。
spec の内容は要約であり、詳細は spec の REQ-21・REQ-28・REQ-30・REQ-31・REQ-32・REQ-33・REQ-39 を参照する。

## 1. 目的と範囲

PC を変えても同じ手順で main の動作確認（項目 A〜F）を再現できるようにする。
スクリプト（`scripts/real-machine-check.sh`・`scripts/real_machine_check_record.py`）は実行と記録の整形までを行い、
「実機」の証拠としての確定は人が行う。Agent の範囲はスクリプトと手順の準備までで、実機の結果を Agent が確定させない（`evidence_hint` は `requires_human_review` か `test_harness` のみ）。

## 2. 新しい PC での前提

1. リポジトリのルートで `make doctor` を実行して環境診断を行う（何も導入しない）。
2. Python 3.9 以上が必要。`python3 --version` で確認。
3. 以下は通信を伴うため、実行前に承認を取得する：
   - `make setup`: Rust ツールチェーン（`rust-toolchain.toml` で管理）。
   - `make py-sync`: `trainer/.venv`（Python 3、uv）・MLX（CPU）。
   - **`cargo fetch --locked`**: 依存を前もってダウンロードする（通信を伴うため承認を得てから実行する。`make setup` は cargo の依存を取得しない）。未実行の場合、実行時に `cargo build --locked` が失敗する（オフラインモード。REQ-38）
4. 前提を満たしたら `make real-machine-check` を実行できる。

## 3. 実行方法

### 3-1. コマンドラインの形式

```bash
make real-machine-check ARGS="--work-dir <DIR> [--items <LIST>] [--repeat N] [--quiet-machine] [--with-ci] [--p95-limit-us N] [--package-limit-bytes N]"
```

または、スクリプトを直接実行する場合：

```bash
<repo>/scripts/real-machine-check.sh --work-dir <DIR> [options...]
```

### 3-2. 引数

値を取るオプション（`--work-dir` / `--items` / `--repeat` / `--p95-limit-us` / `--package-limit-bytes`）は `--key VALUE` と `--key=VALUE` の両方を受け付けます。重複・空要素・未知のオプションは拒否（exit 64）。

| 引数 | 説明 | 既定値 / 必須 |
| ---- | ---- | ---------- |
| `--work-dir DIR` | 作業と記録の置き場。存在しないか空であること。**作業ディレクトリがリポジトリ自身・その配下・その祖先のどれかなら拒否される。symlink・非ディレクトリ・空でないディレクトリも拒否される**（物理パスへ正規化して比較） | 必須 |
| `--items LIST` | 実行する項目。`A,B,C,D,E,F` の部分集合（カンマ区切り・大文字・重複不可） | `B,C,D,E,F`（A は既定では含まない） |
| `--with-ci` | A（`make ci`）を実行する明示の同意。通信を伴いうる（`uv sync`・advisory DB・`npx`）。`--items` に A があり `--with-ci` が無ければ引数エラー（exit 64）で、何も実行しない | — |
| `--repeat N` | F（ガード層の時間制限テスト）の実行回数。1 以上 1000 以下の整数 | 50 |
| `--quiet-machine` | 「他のアプリを閉じた静かな状態」という人の申告。p95 の分類が `real_machine` になるのは、これがあり、かつ `make`・`cargo` の代役も `FANDHE_EDGE_BIN` の差し替えも無いときだけ。それ以外は `reference_only` | — |
| `--p95-limit-us N` | C-1 の推論 p95 上限（マイクロ秒）。1 以上 3600000000 以下の整数（上限は定義ファイルの `limits.max_infer_p95_us` の上限と同じ。REQ-31）。範囲外は CLI・make・cargo を起動する前に引数エラー（exit 64） | 50000 |
| `--package-limit-bytes N` | C-2 の容量上限（バイト） | 1000 |
| `--help` | 使い方を JSON 1 行（`{"code":"ok","message":"usage: ..."}` の形）で stdout へ出して exit 0 | — |

**A を実行する場合の注意**: `--items` に `A` を含めても、`--with-ci` が無ければ以下の JSON を出して exit 64（`invalid_input`）で停止します。実行順は常に A→F です。

```json
{"code":"invalid_input","message":"<固定メッセージ>"}
```

**`make real-machine-check ARGS=...` の制約**: シェルが `$(ARGS)` を単語分割するため、パスなど空白を含む値は渡せません。その場合はスクリプトを直接実行してください：

```bash
scripts/real-machine-check.sh --work-dir '/path with space' ...
```

**`--work-dir` の制約**: リポジトリ外に置く必要があります。物理パス正規化後、以下は拒否されます：

- リポジトリ自身・その配下（リポジトリへは何も書かない）
- リポジトリの祖先（リポジトリを含むディレクトリ）
- symlink
- 非ディレクトリ
- 空でないディレクトリ

### 3-3. 環境変数

| 環境変数 | 説明 | 備考 |
| ------- | ---- | ---- |
| `FANDHE_EDGE_BIN` | CLI のバイナリパス。`/` を含む相対・絶対パスで指定。相対パスは呼び出し時のカレント基準で絶対化される。未設定なら `cargo build --locked --release -p fandhe-edge-cli --bin fandhe-edge` でビルドして、`compiler-artifact` の executable を特定する（出力先は cargo が報告したパス）。ビルドの上限時間は 1800 秒 | `cli_profile:"release"`（未設定時）または `null`（設定時）。`evidence_hint` は MAKE_CMD / CARGO_CMD の有無で決まる（FANDHE_EDGE_BIN の有無は影響しない）。**差し替えた CLI が、cargo がビルドした CLI とバイト単位で同一でない限り D は失敗する**（`make`・`cargo` の代役の下では照合しないため対象外）: `make check-runtime-linkage` が検査するのは cargo がビルドした CLI（cargo の報告から取る。`build.target-dir`・`CARGO_BUILD_TARGET_DIR`・`CARGO_TARGET_DIR` のどれで出力先を変えていても追従する）で、その sha256 が差し替えた CLI と一致しなければ `linkage_target_mismatch`（§4-D） |
| `FANDHE_EDGE_MAKE_CMD`・`FANDHE_EDGE_CARGO_CMD` | 検査用（テスト専用の上書き）。**絶対パスの実行ファイル**（PATH で探さない）。どちらかを設定すると `evidence_hint` が `test_harness` に変わり、`otool -L` は実行されず `direct_libraries` が `null` になる。絶対パスでない場合や実行可能でなければ引数エラー（exit 64） | テスト用のみ。両方同時に設定可能 |

### 3-4. 実行時の上限（REQ-39）

スクリプトは各子プロセスに以下の上限を設けます。超過時は `timeout` または `output_limit` で失敗します。

| 対象 | 上限時間 | stdout | stderr | 用途 |
| ---- | ------- | ------ | ------ | ---- |
| CLI 1 工程（register / inspect / train / select / evaluate / package / infer） | 600 秒 | 1 MiB | 8 MiB | B・C・E |
| `make ci` | 3600 秒 | 64 MiB | 64 MiB | A（通信するため `--locked` なし） |
| `make check-runtime-linkage` | 1800 秒 | 64 MiB | 64 MiB | D（オフラインモード） |
| `cargo test --locked` / 1 回（F） | 300 秒 | 64 MiB | 64 MiB | F（N 回の各回） |
| `cargo build --locked`（FANDHE_EDGE_BIN 未設定時） | 1800 秒 | 64 MiB | 64 MiB | ビルド（オフラインモード。REQ-38） |
| `cargo test --locked ... --no-run`（F の事前ビルド） | 1800 秒 | 64 MiB | 64 MiB | F（オフラインモード。REQ-38） |
| `otool -L`・`git` コマンド・環境採取 | 30 秒 | 64 KiB | 64 KiB | 環境情報 |
| 入力ファイル（定義・学習・評価） | — | 16 MiB | — | 読み込み前に確認 |
| CLI バイナリの sha256 計算 | — | 1 GiB | — | 計算時に確認（超過は計算せず `null`） |
| パッケージファイル個別（model.onnx・vocab.json 等） | — | 256 MiB | — | sha256 計算時に確認 |
| E の推論件数 | — | 1000 件 | — | `train.jsonl` 件数上限 |

スクリプトと `check-runtime-linkage.sh` が自分で起動する `cargo` は、すべて `--locked`（`Cargo.lock` を暗黙に更新しない）。A の `make ci` は make の中の cargo であり、スクリプトは `--locked` を付けない。

## 4. 各項目 A〜F が何を確かめるか

### 4-A. `make ci`（ローカルゲート）

リポジトリのルートで `make ci` を実行し、ビルド・テスト・依存確認の全件成功を確かめます。

- **確かめること**: fmt・clippy・test・py-ci・deny・check-dependency-approvals・check-unsafe-allowlist・lint-docs が全件成功し、プロジェクト全体の品質ゲートを通ること。
- **想定する終了コード**: 0（成功）
- **A の合格条件**:
  - exit 0
  - stdout が読めた
  - `skip:` で始まる行が 0 件（skip は検証済みと扱わない）
  - Rust のテストが 1 件以上通り（`passed` の合計が 1 以上）、failed 0 件
  - pytest の要約行が読めて、`passed` が 1 以上・failed 0 件
- **失敗の理由**: `skipped`（skip 行あり）、`no_test_results`（Rust の `passed` が 0、または pytest の要約行が無い・`passed` が 0）、`test_failures`（failed あり）、`unexpected_exit_code`（0 以外）、`output_limit`（stdout が上限超過で読めない）、`timeout`・`spawn_error`・`killed`（実行の失敗）
- **記録する `record.json` フィールド**: 終了コード（`exit_code`）、`skip_lines`（`skip:` で始まる行の件数）、`rust_tests`（`passed`・`failed`・`ignored`）、`pytest`（`passed`・`skipped`・`failed`）、`stderr_bytes`
- **pytest の skip**: Mac では Linux 用の POSIX ACL テスト 3 件が skip になる。検証済みと扱わない。記録には残す

**注意**: A の中で実行されるテスト（Rust の unit・integration テスト、pytest）はテストハーネスで、実機の証拠にはなりません。

### 4-B. 評価データありの 7 工程（`register → inspect → train → select → evaluate → package → infer`）

入力データ（`fixtures/sandbox_run_eval/`）をコピーして、7 工程を順に実行し、正常系での完走を確かめます。学習は CPU で、`--smoke` は付けません。候補は 1 個（ID 0）。

- **確かめること**: 各工程が連続して終了コード 0 で成功し、出力される JSON の報告値が、作業ディレクトリへコピーした fixture から導いた値と整合すること。`package` の容量内訳（5 項目）が合計と一致すること。
- **想定する終了コード**: 0（全工程成功）
- **B の成功条件**: 7 工程すべてが exit 0 で `status` が `ok`、かつ各工程の報告値が次のとおり整合し、`capacity` の 5 要素（`weights`・`vocab_or_feature_transform`・`label_table`・`calibration`・`metadata`）が各々 `bytes` と `file_count`（0 以上の整数）を持ち、5 要素の合計が `capacity.total_bytes` と一致する（C も同じ照合を通る。REQ-21・REQ-33）。さらに、公開された `package/` 直下の通常ファイルのバイト数の合計が `capacity.total_bytes` と一致する（REQ-30）。`==` で比べる報告値（`options`・`candidate`・`n_total` など）は、比べる前に真偽値を除く整数であることを確かめる
  - `register`: `options` が定義の選択肢の数、`evaluation_defined` が「評価データの行数が 1 以上」と一致
  - `inspect`: `valid_records` が `train.jsonl` の行数（空行を除く）と一致し、`split` の `train`・`validation`・`test` の和が `valid_records` と一致
  - `train`・`select`: `candidate` が 0（B・C は候補 0 を学習する）で、`kind` が `c1`・`c3`・`autoregressive` のどれか（`crates/guard/src/kind.rs` の `SUPPORTED_KINDS` と一致することを pytest が機械照合する）。`select` の `kind` は `train` と同じ値
  - `evaluate`: `candidate` が `select` の報告値、`kind` が `select` と同じ値、`n_total` が `evaluation.jsonl` の行数（1 以上）、`correct` が 0 以上 `n_total` 以下、`accuracy` が `correct / n_total` と 1e-9 以内で一致、キー `macro_f1` が存在して `null` か 0 以上 1 以下（分母 0 の指標は `null`。REQ-24）
  - `package`: キー `judgment`・`infer_p95` が値が `null` でも存在する。`capacity` は各値が 0 以上の整数で、`limit_bytes` が定義の `limits.max_package_bytes`（無ければ既定値 40000000。`crates/cli/src/stages/package.rs` の `DEFAULT_CAPACITY_LIMIT_BYTES` と一致することを pytest が機械照合する）と一致し、`exceeded == (total_bytes > limit_bytes)`、exit 0 なら超過していない。B の定義は合否基準と上限を持たないため、`judgment` が `null`・`acceptance_defined` が `false`・`infer_p95` が `null`
  - `infer`（B の単発）: `predicted_label` が定義の選択肢 ID のどれか、`scores` のキー集合が選択肢 ID と一致し、各値が有限で 0 以上 1 以下、和が 1 から 1e-6 以内（`SCORE_SUM_TOLERANCE` は `fixtures/score_tolerance/score_sum_tolerance.json` と一致することを pytest が機械照合する）、`predicted_label` が最大スコアの選択肢（同点は定義の宣言順で先頭）
- **失敗の理由**: `input_unreadable`（fixture が読めない）、`missing_field`（JSON に必須フィールドなし）、`package_unreadable`（`package/` が読めない）、`capacity_sum_mismatch`（容量の 5 要素の合計と `total_bytes` が一致しない）、工程別の失敗（`step` に工程名・`exit_code` に終了コード・`reason` に `timeout`・`output_limit`・`spawn_error`・`killed`・`invalid_json`・`unexpected_exit_code`・`unexpected_output`。報告値の不整合は `unexpected_output`。stdout の JSON の入れ子が深すぎて読めない場合も `invalid_json`）。`package/` のファイルの合計が `total_bytes` と合わない場合は `unexpected_output`（`step` は `package`）
- **記録する `record.json` フィールド**:
  - `status`: `"ok"` または `"failed"`
  - `steps`: 工程の配列（各工程：`step`（固定語彙）・`command`（固定語彙）・`exit_code`・`stderr_bytes`・`summary`）。`summary` は工程ごとに決まった欄だけを新しく組み立てた要約で、CLI の JSON の他の欄は捨てる（欄は常に出し、無い・型が違う値は `null`、閉じた語彙の欄で語彙外の文字列は `<unexpected>`）
    - `register`: `step`・`status`・`options`・`evaluation_defined`・`definition_sha256`
    - `inspect`: `step`・`status`・`valid_records`・`split`（`train`・`validation`・`test`）
    - `train`・`select`: `step`・`status`・`candidate`・`kind`
    - `evaluate`: `train` の 4 欄に加えて `n_total`・`correct`・`accuracy`・`macro_f1`
    - `package`: `step`・`status`・`code`・`judgment`・`acceptance_defined`・`capacity`・`infer_p95`
    - `infer`: `status`・`scores_keys`（`scores` のキー数）・`predicted_index`（`predicted_label` が定義の選択肢 ID の何番目か。0 始まり。選択肢に無ければ `null`）。選択肢 ID は利用者が決める文字列のため `predicted_label` は記録しない
  - `capacity`: `total_bytes`・`limit_bytes`・`exceeded`・`components`（5 項目。各々 `bytes` と `file_count`）
  - `capacity_sum_matches_total`: 5 要素の合計が `total_bytes` に一致したか（boolean）
  - `package_files`: `package/` 内ファイルの配列（各ファイル：`name`・`bytes`・`sha256`。`name` が `^[A-Za-z0-9._-]{1,64}$` に一致しなければ `<unrecognized>`）
  - `reason`（失敗時のみ）: 上記の失敗理由。工程の `unexpected_exit_code`・`unexpected_output` では、あわせて `code`（CLI の stdout の `code` が 7 種の語彙の値ならその値、語彙外の文字列なら `<unexpected>`、文字列でなければ欄なし）と、`message` が文字列のときの `message_bytes`（UTF-8 のバイト数）・`message_sha256` を出す。`message` の本文は記録せず、`<work-dir>` の工程の stdout のファイルに残る

### 4-C. `package` の上限照合

2 つの通りを別の project で実行します。C 全体で 1 つの `status` を持ちます。

#### 4-C-1. `limits.max_infer_p95_us`

定義へ `limits: {"max_infer_p95_us": <値>}` を足して実行します。

- **確かめること**: `package` が exit 0 または 20 で成功し、`infer_p95`（推論 p95 の値・上限・超過フラグ）が記録されること。指定した上限が CLI に伝わること。
- **想定する終了コード**: 0 または 20
- **C-1 の合格条件**:
  - exit 0 または 20（どちらでも ok。ただし下の対応を満たすこと）
  - `infer_p95` の `p95_us`・`limit_us` が 0 以上の整数（真偽値・負数は不可）で、`exceeded` が boolean
  - `exceeded` が `p95_us > limit_us` の値と一致
  - `exceeded` が exit コード 20 の有無と一致：`exceeded == true` ↔ `exit 20`
  - exit 20 のときは `code == "limit_exceeded"` で、`package/` が作られていない。exit 0 のときは `status == "ok"` で、`package/` が公開されている
  - `limit_us` が指定値（`--p95-limit-us`）と一致
  - C-1 の `capacity` が要約でき（各値が 0 以上の整数で `exceeded == (total_bytes > limit_bytes)`、5 項目の `bytes` の合計が `total_bytes` と一致、`limit_bytes` が既定値 40000000 と一致）、超過していない
  - exit 0 のときは、公開された `package/` 直下の通常ファイルのバイト数の合計が `capacity.total_bytes` と一致する（C-1 は一覧を記録へ足さず、照合だけ行う。REQ-30）
  - 工程の検査（B と同じ。`kind` の一貫性・`evaluate` の `accuracy`・`macro_f1`・`package` のキー `judgment`・`infer_p95` の存在）を通る
- **失敗の理由**: `unexpected_output`（`exceeded` と計算値・終了コードの不一致、`code`・`status` の不整合、`package/` の有無と終了コードの不一致、`limit_us` の不一致、`p95_us`・`limit_us` が負数、`capacity` の欠落・超過・内訳の合計の不一致・`limit_bytes` の不一致、`package/` のファイルの合計と `total_bytes` の不一致のいずれか）、`missing_field`（C-1 の判定〔`judge_p95`〕が `infer_p95` の型違い・欠落を見つけたとき。通常は工程の検査が先に `unexpected_output` で止める）、工程別の失敗
- **記録する `record.json` フィールド**:
  - `p95.p95_us`・`p95.limit_us`・`p95.exceeded`・`p95.classification`（`real_machine` / `reference_only`）・`p95.package_exit_code`（0 または 20）・`p95.package_published`（C-1 の `package/` の有無）
- **注意**: p95 の値は参考値です。`--quiet-machine` を付け、他のアプリを実際に閉じた状態でのみ「実機の p95」として扱います。`--quiet-machine` があっても、`make`・`cargo` の代役（`FANDHE_EDGE_MAKE_CMD`・`FANDHE_EDGE_CARGO_CMD`）か `FANDHE_EDGE_BIN` の差し替えの下では `classification` は `reference_only` で、それ以外でも静かな状態は人の申告です。

#### 4-C-2. `limits.max_package_bytes`

定義へ `limits: {"max_package_bytes": <値>}` を足して実行し、容量上限超過を確かめます。

- **確かめること**: `package` が exit 20 で reject され、理由が容量超過であること。`package/` ディレクトリが公開されないこと。
- **想定する終了コード**: 20
- **C-2 の成功条件**:
  - exit 20
  - `code == "limit_exceeded"`
  - `capacity.exceeded == true`
  - `capacity.total_bytes > capacity.limit_bytes`（超過判定の一致）
  - `capacity.limit_bytes` が指定値（`--package-limit-bytes`）と一致
  - `package/` ディレクトリが存在しない
- **失敗の理由**: `unexpected_output`（工程の検査が先に止めるもの。`package` の JSON の `step` の不一致、`capacity` の欠落・上限値の不一致・内訳の合計の不一致・`exceeded` と計算値の不一致、`infer_p95` が null でない〔C-2 の定義に p95 の上限は無い〕など報告値の不整合。exit 20 で `code` が `limit_exceeded` でない場合、または容量も p95 も超過と報告されていない場合もここで `unexpected_output` になる。`case` は `C-2`）、`capacity_limit_not_enforced`（工程の検査を通ったあとの C-2 の判定〔`judge_capacity_limit`〕で落ちるもの。実際に届くのは 2 つだけで、`package` が exit 0 で報告値に不整合がない〔上限を超えていないと報告された〕場合と、exit 20 なのに `package/` が公開されている場合）、工程別の失敗（`package` が exit 0・20 以外のときは `unexpected_exit_code`。exit 0 は工程の検査では許され、C-2 の判定で落ちる）。C-2 の判定にも `capacity` の欠落・上限値の不一致で `unexpected_output` を返す分岐があるが、工程の検査が先に同じ条件で止めるため通常は届かない。工程の検査で止まった失敗には `capacity_limit` の欄は付かず（`case`・`steps`・`p95` のみ）、`capacity_limit` は C-2 の判定まで届いたときだけ記録される
- **記録する `record.json` フィールド**:
  - `status`: `"ok"` または `"failed"`
  - `p95`: C-1 の結果（再利用）
  - `capacity_limit.package_exit_code`（20）、`.code`（7 種の語彙の値。語彙外の文字列は `<unexpected>`、文字列でなければ `null`）、`.capacity_exceeded`（boolean または null）、`.total_bytes`・`.limit_bytes`、`.infer_p95_exceeded`（boolean または null）、`.package_published`（`package/` ディレクトリの存在）
  - `reason`（失敗時のみ）: `capacity_limit_not_enforced`（`capacity_limit` 欄を伴う場合。判定が `unexpected_output` を返す分岐は上記のとおり通常届かない）。工程の検査で止まった失敗は `reason` が `unexpected_output`・`unexpected_exit_code` などで、`capacity_limit` 欄は付かない

### 4-D. 推論が学習に依存しないこと（REQ-32）

リポジトリのルートで `make check-runtime-linkage` を実行し、CLI が Python・MLX への動的リンクを持たないこと、および環境変数ゼロの環境での実行が成功することを確かめます。

- **確かめること**: `make check-runtime-linkage` の exit 0 と、その成功の印（env -i テスト 3 件・`OK: tool=...` の行）。リンクを確認したバイナリが、B・C・E で実行した CLI と同一であること。macOS の場合は、実行した CLI への `otool -L` の結果を記録する（`check-runtime-linkage.sh` の判定とは別の、情報としての記録）。
- **想定する終了コード**: 0
- **D の合格条件**:
  - exit 0
  - stdout が読めた
  - `skip:` で始まる行が 0 件
  - `ok: req32_*` の行がちょうど 3 件（env -i テストが成功。件数は `check-runtime-linkage.sh` の env -i テスト数の定数 `LINKAGE_ENV_I_TESTS`）
  - `OK: tool=<otool|ldd> ...` の行があり（stdout を後ろから見て最初に一致した行を使う）、macOS では tool が `otool`（`ldd` は実機の確認にならない）
  - リンクを確認したバイナリ（`check-runtime-linkage.sh` が cargo の報告〔`--message-format=json-render-diagnostics` の `compiler-artifact` の `executable`〕から取り、ログ `<work-dir>/D/linkage.log` に `cli_bin: <絶対パス>` の 1 行で出す。記録側はこの行からパスを読んで sha256 を計算する。パスは `record.json`・`record.md` に出ない）の sha256 が、いま実行した CLI の sha256 と、開始時に記録した `cli_sha256` の両方に一致する。`make`・`cargo` の代役の下では偽の `make` が何も検査しないため照合しない（`linkage_target_matches_cli` は `null`）
- **失敗の理由**: `unexpected_exit_code`（0 以外）、`skipped`（skip 行あり）、`no_test_results`（env -i テストの `ok:` 行が 0 件）、`unexpected_output`（`ok: req32_*` が 3 件でない・`OK: tool=` の行が無い・macOS で tool が `otool` でない）、`linkage_target_unreadable`（`cli_bin:` 行が無い・2 行以上ある・絶対パスでない、または検査対象を読めない・上限超過）、`cli_changed`（いまの CLI が開始時と異なる、または読めない）、`linkage_target_mismatch`（検査対象が実行した CLI と異なる）、`otool_failed`（macOS で `otool -L` に失敗）、実行の失敗（`timeout`・`output_limit`・`spawn_error`・`killed`）
- **記録する `record.json` フィールド**:
  - `status`: `"ok"` または `"failed"`
  - `exit_code`: make の終了コード
  - `skip_lines`: stdout に `skip:` で始まる行の件数
  - `env_i_tests_ok`: stdout の `ok: req32_*` の件数
  - `linkage_tool`: `OK: tool=...` の行（後ろから見て最初に一致した行）の tool（`otool`・`ldd`。読み取れなければ `null`）
  - `linkage_target_sha256`: リンクを確認したバイナリの sha256（照合しない場合・照合前に失敗した場合は `null`）
  - `linkage_target_matches_cli`: 検査対象が実行した CLI と一致したか（boolean。照合しない場合・照合前に失敗した場合は `null`）
  - `direct_libraries`: 実行した CLI への `otool -L` の出力から抽出した直接リンク（macOS のみ。記録では `/usr/lib/`・`/System/` で始まり安全な文字だけの 128 字以内のものだけが名前のまま残り、それ以外は `<redacted>` に置き換わる。代役の下では取らず `null`）
  - `reason`（失敗時のみ）: 上記の失敗理由

### 4-E. 1 件ずつとバッチ推論の一致（REQ-28）

B で成功した `package/` に対し、学習データの入力だけ（`train.jsonl` の `id` と `input` フィールドのみ）を 2 つの方法で推論し、結果が一致することを確かめます。

**重要**: `evaluation.jsonl`（凍結した評価データ）は使いません。学習データ（`train.jsonl`）を使うことで、REQ-27（評価の独立性）を保ちます。

- **確かめること**: 1 件ずつの推論とバッチ推論の予測ラベルが全件一致し、スコアに NaN・無限大がなく、差が許容値以下であること。
- **想定する終了コード**: 0
- **E の合格条件**: B が `ok` + 次の整合 + 予測ラベル不一致 0 件 + NaN・無限大 0 件 + スコア差の最大値が 1e-9 以下
  - バッチ出力の行数が入力の件数と一致し、各行が JSON で `status` が `ok`・`id` が文字列で重複なし
  - バッチの各行と単体の出力が、どちらも infer の規則（`predicted_label` が選択肢 ID・`scores` のキー集合が選択肢 ID と一致・各値が有限で 0 以上 1 以下・和が 1 から 1e-6 以内・`predicted_label` が最大スコアの選択肢〔同点は定義の宣言順で先頭〕）を満たし、単体の出力の `id` が入力の `id` と一致
- **E が実行されない場合**（`not_run`）: B が要求されていない（`requires_B`）、または前の項目が失敗している（`previous_item_failed`。B の失敗を含む）
- **失敗の理由**: `input_unreadable`（学習データが読めない）、`record_count_out_of_range`（件数 0 または 1000 超）、`duplicate_id`（id の重複）、`mismatch`（予測またはスコアが一致しない。NaN・無限大も含む）、`unexpected_output`（バッチ側は行数の不一致・`status` が `ok` でない・出力に `step` の欄がある・`id` が文字列でない・`id` の重複・infer の規則の不整合〔`step` は `infer-batch`〕。単体側は stdout が JSON オブジェクトとして読めない〔JSON でない・入れ子が深すぎる・空の dict〕・`status` が `ok` でない・出力に `step` の欄がある・`id` が入力と一致しない・infer の規則の不整合〔`step` は `infer-single`〕）、`invalid_json`（バッチの行が JSON として読めない〔JSON でない・入れ子が深すぎる〕場合だけ。`step` は `infer-batch`。単体側は `invalid_json` にならない）、工程別の失敗（`step:"infer-batch"` / `"infer-single"`。`timeout`・`output_limit`・`spawn_error`・`killed`・`unexpected_exit_code` を含む）
- **記録する `record.json` フィールド**:
  - `status`: `"ok"` / `"failed"` / `"not_run"`
  - `reason`（`not_run` 時）: `requires_B`・`previous_item_failed`
  - `records`: 推論した件数
  - `label_match`・`label_mismatch`: 予測ラベルの一致数・不一致数
  - `scores_exact_match`: スコアが完全に一致した件数
  - `scores_nonfinite`: スコアに NaN・無限大があった件数
  - `max_abs_score_diff`: 有限なスコア差の最大値（非有限は除外）
  - `input_sha256`: 推論に使った入力ファイルのハッシュ
  - `reason`（失敗時のみ）: `mismatch`
  - 失敗時の詳細は `.../E/mismatch-ids.txt` に以下のいずれかに該当する `id` を記録（1 行ずつ）：予測ラベル不一致・スコアに NaN・無限大・スコア差が許容差超過

### 4-F. ガード層の時間制限テスト再発確認（#346）

`cargo test --locked -p fandhe-edge-guard --test time_limit` を N 回実行し、1 回も失敗しないこと（#346 の再発〔並行 spawn の fd 継承競合〕が無いこと）を確かめます。

- **確かめること**: テストバイナリのコンパイル（初回）が成功し、その後 N 回のテスト実行がすべて成功すること。#346 の再発の兆候として、失敗した回のログにある `ReadOutput`・`ReadOutputIncomplete` を数えて記録する（成功した回のログは数えない。成功の判定は各回の exit 0 と `test result: ok. N passed`〔N >= 1〕で、これらの語が成功した回に出ても失敗にしない）。
- **想定する終了コード**: 0（失敗 0 回）または 10（失敗あり）
- **F の成功条件**: テストバイナリビルド成功 + 実行回数 N 回 + 失敗 0 回（各回が exit 0・期限内・出力上限内で、ログに `test result: ok. N passed`（N >= 1）がある）
- **失敗の理由**: `build_failed`（`--no-run` のビルドが失敗・期限超過・出力上限超過）、`test_failures`（N 回のうち 1 回以上の失敗。失敗の内訳は `timeouts`・`no_tests` の件数欄に出る）。`no_tests`・`timeout` は `reason` ではなく件数の欄
- **記録する `record.json` フィールド**:
  - `status`: `"ok"` / `"failed"`
  - `runs`: 実行回数（N）
  - `passed`・`failed`: 成功・失敗の回数
  - `timeouts`: 制限時間超過の回数
  - `no_tests`: exit 0 で終わったが `test result: ok. N passed`（N >= 1）が無かった回数（失敗に数える）
  - `read_output`: 失敗した回のうち、ログに `ReadOutput`（Incomplete 以外）が出現した回数（成功した回は数えない）
  - `read_output_incomplete`: 失敗した回のうち、ログに `ReadOutputIncomplete` が出現した回数（成功した回は数えない）
  - `load_start`・`load_end`: 実行開始・終了時の load average（3 要素の配列。`os.getloadavg()` で取れなければ `null`）
  - `reason`（失敗時のみ）: `build_failed` / `test_failures`
- **実行特性**: F だけは全 N 回を数えてから成否を決めます（失敗があっても止めず最後まで実行する）。失敗 0 回で `ok`、1 件以上で `failed`。
- **初回実行の自動準備**: スクリプトが N 回実行の前に `cargo test --locked ... --no-run` を 1 回実行し、テストバイナリをコンパイルします（回数に数えない。失敗は `build_failed`）。事前準備は不要ですが、初回実行は通常より時間がかかります。

## 5. 判定の読み方

スクリプトの終了コードは 4 種です。

| 終了コード | 意味 | 行動 |
| -------- | ---- | ---- |
| 0 | 要求したすべての項目が `ok` | 実機確認成功。`record.json`・`record.md` を PR に記録 |
| 10 | 1 つ以上の要求項目が `failed` または `not_run`（E が B の失敗で `not_run` の場合を含む） | 失敗した項目の記録を確認し、原因を特定する。原因不明のまま再実行しない |
| 64 | 引数エラー（前処理で検出） | `--items A` 指定時に `--with-ci` が無い、`--work-dir` がリポジトリ内、など。エラーメッセージ（JSON）から原因を確認して引数を修正。`record.json` は出力されない |
| 70 | 実行環境エラー・中断 | python3 が無い・3.9 未満、作業ディレクトリを作成できない、`FANDHE_EDGE_BIN` が実行不可、fixture が読めない、`record.json` を書き込めない、`FANDHE_EDGE_BIN` 未設定で CLI をビルドできない、実行中に SIGINT・SIGTERM・SIGHUP で中断、項目に想定外の例外が出た（`reason` が `internal_error`。1 件でもあれば stdout の `code` は `runtime_error`）、など。環境を確認またはスクリプトを再実行。**中断時・`internal_error` 時は `record.json` が書かれる**（その時点までの項目の結果を記録。中断した項目の `reason` は `interrupted`）。その時点までの `record.md` も出力される |

**項目の実行フロー**:

1. 要求した項目（`--items LIST`）を A→F の順で実行する
2. 各項目の実行：
   - 要求されていない → `not_run` / `not_selected`
   - 中断済み → `not_run` / `interrupted`
   - 前の項目が `failed` → `not_run` / `previous_item_failed`（B の失敗で止まった E もこれ）
   - E で B が要求されておらず成功していない → `not_run` / `requires_B`
   - それ以外 → 実行して結果を記録
3. 最初に `failed` になった項目があれば、以降の要求項目は実行されず `not_run` になる（`not_run` の `reason` は §8）
4. 終了コード 0 = 要求したすべての項目が `ok`、10 = 1 つ以上が `failed` / `not_run`

**項目ごとの成否判定**:

各項目の合格条件は §4 のとおりで、満たす = `ok`、満たさない = `failed` です（終了コード 0 だけでは `ok` にならない）。要点:

- **A**: exit 0 + `skip:` 行 0 件 + Rust と pytest のテストが 1 件以上通り failed 0 件
- **B**: 全 7 工程が exit 0 で報告値が整合 + 容量の 5 要素の合計が `total_bytes` と一致 = `ok`。1 工程でも非 0・報告値の不整合・容量の不一致 = `failed`
- **C**: C-1 と C-2 の両方が成功（C-1 は exit 0 または 20 で p95 と容量・`package/` の有無が整合、C-2 は exit 20 + 容量超過 + `package/` 非公開）= `ok`。1 つでも失敗 = `failed`。C 全体で 1 つの `status` を持つ
- **D**: exit 0 + 成功の印（env -i テスト 3 件・`OK: tool=...`）+ リンクを確認したバイナリが実行した CLI と一致 = `ok`
- **E**: B が要求されていない、または前の項目が失敗 → `not_run`。それ以外で、バッチと単体が整合し予測・スコアが全一致 = `ok`。一致しない = `failed`
- **F**: `failed == 0` = `ok`、`failed > 0` = `failed`

**失敗時の報告の 5 点**:

1. 終了コード
2. `reason`（失敗の理由。固定語彙：`timeout`・`output_limit`・`spawn_error`・`killed`・`invalid_json`・`unexpected_exit_code`・`unexpected_output`・`missing_field`・`input_unreadable` など。項目ごとの語彙は §4。想定外の例外は `internal_error`〔`error_type` に例外の型名〕）
3. `step`（工程名。固定語彙：`register`・`inspect`・`train`・`select`・`evaluate`・`package`・`infer` など。工程が無い場合は省略）
4. stdout の JSON（あれば）から `code`（7 種の語彙の値）・`message_bytes`・`message_sha256`（`message` の本文は記録されず、`<work-dir>` の stdout のファイルに残る）
5. 実行したコマンド（固定語彙。パスは含めない）

## 6. 証拠の種別の扱い

`record.json` の `evidence_hint` は以下の 2 値のみです。

| 値 | 意味 |
| ---- | ---- |
| `requires_human_review` | 実機で実行。人が確認して「実機」と記録 |
| `test_harness` | スクリプトまたはテスト内に偽の `make`・`cargo`・CLI を使用（`FANDHE_EDGE_MAKE_CMD`・`FANDHE_EDGE_CARGO_CMD` 設定時） |

**項目ごとの扱い**:

- **A**: 終了のみが実機証拠。中のテスト（Rust unit・integration、pytest）はテストハーネス。記録には区別して記載する
- **C-1 の p95**: `--quiet-machine` フラグが **あり、かつ実際に他のアプリを閉じた状態** でのみ `classification: "real_machine"`。フラグが無い、フラグはあるが実際に静かでない、`make`・`cargo` の代役か `FANDHE_EDGE_BIN` の差し替えの下で実行した = `reference_only`（最後の 2 つはフラグがあっても `reference_only`）。参考値扱いで「実機の p95」としない
- **E**: 学習データ（`train.jsonl`）のみ使用し、評価データ（`evaluation.jsonl`）は使わない（REQ-27。評価の独立性）
- **F**: テストハーネスを実機で実行したもの。テストの実装（偽の sleeper、実時計）は実機ですが、判定対象（予測値・スコア）は合成に依存しているため、実機の証拠としては「テストハーネス」扱い

**B・C のプロジェクト**: 複数項目で同じ評価データを使う場合は、project ごとに 1 回だけ適用します。B・C-1・C-2 は各々異なる project を使い、各 project へ 1 回ずつ評価データを適用します（REQ-27）。

## 7. `record.json` / `record.md` の保存と PR・Issue への転記

### 保存先と形式

`record.json` は `<work-dir>/` の直下に保存され、実行時のマスク（umask 077）で 0600 の権限を持ちます。各項目の詳細ログは `<work-dir>/<項目>/steps/` などに `*.log`・`*.err` として保存されますが、このログは `record.json` に含まれず、PR・Issue へも転記されません。

### 記録してよい項目（転記を許可）

- 終了コード・`code` 値（固定語彙）
- 件数・ハッシュ・サイズ・ライブラリ名（`/usr/lib/`・`/System/` で始まるもの）
- load average・開始・終了の時刻
- 工程の要約（工程ごとに決まった欄だけ。固定語彙・検証済みの数値と真偽値）。`message` の `message_bytes`・`message_sha256`
- 機種・メモリ・OS・コミット SHA・実行日（`hw_model`・`cpu`・`os_name`・`os_version`・`os_build` は `^[A-Za-z0-9 ._,()+-]{1,64}$` に一致しなければ `null`。`ncpu`・`memory_bytes` は ASCII の数字 1〜20 桁だけを整数にし、それ以外は `null`）
- `record.json` / `record.md` の内容（伏せ処理済み）

**記録の作り方（許可リスト方式）**: 記録は、記録してよい欄を先に決めて組み立てる。工程の要約（`steps[].summary`）は、工程ごとに決まった欄だけを新しい dict へ組み立て、CLI の JSON の他の欄は捨てる（§4-B）。閉じた語彙の欄で、文字列だが語彙外の値は `<unexpected>`、`package_files[].name` が規則外なら `<unrecognized>`、型が違う欄は `null` になる。最後の関門（`sanitize_record`）で、文字列の値を持ってよいキーの閉じた集合に無い位置の文字列と、200 字を超える文字列を `<redacted>` に置き換える（個別の要約で漏れても止まる）。パス区切りを含む文字列も、`items.D.direct_libraries` の要素（`/usr/lib/`・`/System/` で始まる 128 字以内のもの）を除いて `<redacted>` になる。非有限の浮動小数は `null` になる。値が `<unexpected>`・`<unrecognized>`・`<redacted>` の欄は、記録の規則が働いた印であり、原因は `<work-dir>` のログで確かめる

### 転記してはいけないもの（記録簿から除外）

- **パス**（入力ファイル・作業ディレクトリ・出力先）。例外: `/usr/lib/`・`/System/` で始まるシステムライブラリ名のみ可
- **データ本文**（`input`・`text` の値、`train.jsonl`・`evaluation.jsonl` の内容、`id` の値）
- **生ログ**（`<work-dir>/*/` 配下の `*.log`・`*.err`・`*.ndjson` ファイルの内容）。バイト数・出力有無だけ記録
- 失敗時の完全なコマンド文字列（パスを含む）
- CLI の `message` の本文と、`infer` の `predicted_label`・選択肢 ID（利用者が決める文字列。記録には `message_bytes`・`message_sha256`・`predicted_index` だけが出る）

### 作業ディレクトリのクリーンアップ

`record.json`・`record.md` を PR へ転記した後は、作業ディレクトリ全体（`<work-dir>/`）は削除してかまいません。`record.json`・`record.md` のバックアップが必要な場合は、PR・Issue・wiki など別の場所に保存してください。

## 8. `record.json` と `record.md` の構造

### `record.json`

スクリプトが `<work-dir>/record.json` に出力する JSON の構造は以下のとおりです。

```json
{
  "schema": "real-machine-check/1",
  "evidence_hint": "requires_human_review" | "test_harness",
  "bin_override": true | false,
  "environment": {
    "hw_model": "string (^[A-Za-z0-9 ._,()+-]{1,64}$) or null",
    "cpu": "string (same rule) or null",
    "ncpu": "int (ASCII digits, 1-20) or null",
    "memory_bytes": "int (ASCII digits, 1-20) or null",
    "os_name": "string (same rule) or null",
    "os_version": "string (same rule) or null",
    "os_build": "string (same rule) or null",
    "commit": "40-char hex or null",
    "worktree_clean": "true | false | null",
    "started_local": "ISO8601 with offset",
    "ended_local": "ISO8601 with offset",
    "cli_sha256": "64-char hex or null",
    "cli_bytes": "int or null",
    "cli_profile": "release" | null
  },
  "inputs": {
    "train_records": "int",
    "evaluation_records": "int",
    "definition_sha256": "64-char hex",
    "train_sha256": "64-char hex",
    "evaluation_sha256": "64-char hex"
  },
  "options": {
    "items": ["A" | "B" | ... | "F"],
    "repeat": "int",
    "quiet_machine": "true | false",
    "p95_limit_us": "int",
    "package_limit_bytes": "int",
    "with_ci": "true | false",
    "cargo_offline": true
  },
  "items": {
    "A": { "status": "ok" | "failed" | "not_run", ... },
    "B": { "status": "ok" | "failed" | "not_run", ... },
    ...
    "F": { "status": "ok" | "failed" | "not_run", ... }
  }
}
```

各項目の詳細フィールドは §4-A 〜 §4-F 参照。記録の文字列の欄は許可リスト方式で組み立てられ、語彙外の値は `<unexpected>`・`<unrecognized>`・`<redacted>`、型が違う欄は `null` になる（§7）。`schema` は `real-machine-check/1` のまま。`not_run` 項目の `reason` は：

- `not_selected`: `--items` に指定されなかった
- `previous_item_failed`: 前の項目が `failed` になった
- `interrupted`: 実行中に SIGINT・SIGTERM・SIGHUP で中断
- `requires_B`: E で B が要求されていない（B が失敗した場合は `previous_item_failed`）

### `record.md`

自動生成の Markdown テンプレート。冒頭に `bin_override`・`evidence_hint`・harness の有無が表示され、以下は注意行が出ます：

- `bin_override: true` の場合：「注意: CLI を `FANDHE_EDGE_BIN` で差し替えた（このスクリプトがビルドした CLI ではない）」
- 証拠の種別は「人が確認して記入」と指示されます
- 「項目ごとの結果」の表の「要点」には、各項目の `record.json` の欄から `status`・`steps`・`package_files`・`capacity`・`p95`・`capacity_limit`・`message_sha256` を除いたものを JSON で出します（B は `total_bytes` と `capacity_sum_matches_total`、C は `p95` と `capacity_limit` を足す）。失敗の要点は `reason`・`step`・`exit_code` と、`code`・`message_bytes` です（`message_sha256` は `record.json` だけ）

## 9. 記録簿

1 度の実行（PC・コミット・条件）ごとに 1 行を追加します。

| 日付・機種・OS・コミット | 実行方法 | 全体結果 | A | B | C-1 | C-2 | D | E | F | 証拠種別 | 記録先 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 2026-10-04〜05 / Mac16,6（Apple M4 Max）・64 GiB / macOS 27.0（26A428） / 132a7da | 手動コマンド（スクリプト未使用） | 全項目成功 | ok | ok | ok | ok | ok | ok | ok（再発なし） | 実機（項目別注記あり） | #354 |

## 10. 1 件目の詳細（2026-10-04〜05、コミット 132a7da）

本実行は契約確定前に手動コマンドで実行されたもので、`record.json` が無く、スクリプトの `record.md` も生成されていません。以下は当時の確認内容の要点です。

### 9-A. ローカルゲート（`make ci`）

`make ci` は終了コード 0 で完走しました（約 4 分）。

- `cargo test --workspace`: 2146 passed・0 failed・6 ignored
- `test-trainer-integration`: 4 件すべて 1 passed（c1・c3 の実 trainer、sandbox 監視チェーンの評価なし・評価あり）
- `py-ci`: ruff 指摘 0 件（56 ファイル）、pytest 682 passed・3 skipped
- `check-dependency-approvals`・`check-unsafe-allowlist`: 違反 0 件
- `deny`: 指摘 0 件（警告 6 件は license-not-encountered で許可リスト外の稀少ライセンス）
- `fmt-check`・clippy・lint-docs: 指摘・警告 0 件

**検証済みと扱わないもの**:

- pytest skip 3 件（Linux 用 POSIX ACL テスト。Mac では実行されない）
- Rust ignored 6 件のうち 2 件（子プロセスの補助関数。別のテストから呼ばれる）。残る 4 件は `test-trainer-integration` で実行

### 9-B. 評価データありの 7 工程

`register → inspect → train → select → evaluate → package → infer` を順に実行。学習は CPU、候補は 1 個。

| 工程 | 終了コード | JSON 要約 |
| ---- | -------- | ------ |
| register | 0 | `status: ok`・選択肢 3・`evaluation_defined: true` |
| inspect | 0 | `status: ok`・有効 90 件・分割 train/validation/test |
| train | 0 | `status: ok`・候補 0・`kind: c1` |
| select | 0 | `status: ok`・候補 0 |
| evaluate | 0 | `status: ok`・`n_total` 12・正解率 1.0 |
| package | 0 | `status: ok`・`judgment: null` |
| infer | 0 | `status: ok`・予測ラベル・スコア |

**`capacity` の 5 項目**（B のみ）:

| 項目 | bytes | file_count |
| ---- | ----- | ---------- |
| weights | 4568 | 1 |
| vocab_or_feature_transform | 0 | 0 |
| label_table | 328 | 1 |
| calibration | 0 | 0 |
| metadata | 531 | 1 |

- `total_bytes` 5427。5 項目の合計と一致
- `package/` の 3 ファイル: `model.onnx`（4568 B、sha256 `d063...821`）、`definition.json`（328 B）、`artifact.json`（531 B）

### 9-C. `package` の上限照合

新しい project 2 つで実行。`register → evaluate` までは exit 0。

**C-1**（`max_infer_p95_us: 50000`）:

- `package` exit 0
- `infer_p95`: `p95_us` 2・`limit_us` 50000・`exceeded` false
- `classification: reference_only`（他アプリが動作中。load avg 8.10。`--quiet-machine` 無し）

**C-2**（`max_package_bytes: 1000`）:

- `package` exit 20
- `code: limit_exceeded`
- `capacity.exceeded` true（`total_bytes` 5615 > 1000）
- `infer_p95` null

### 9-D. 推論が学習に依存しないこと（REQ-32）

`make check-runtime-linkage` exit 0。

- `otool -L`: Python・MLX リンクなし。直接リンク 2 件（`libSystem.B.dylib`・`libiconv.2.dylib`）
- `env -i` テスト 3 件: すべて ok
  - `req32_inference_succeeds_under_env_i_without_path`
  - `req32_inference_succeeds_under_env_i_with_system_path`
  - `req32_inference_does_not_invoke_python_from_path`

### 9-E. 1 件ずつとバッチ推論の一致（REQ-28）

B の `package/` に対し、学習データ 90 件で確認。評価データ（`evaluation.jsonl`）は使わない。

- `infer --input-file`: 1 回・exit 0・出力 90 行
- `infer --text`: 90 回・全回 exit 0・出力 90 行
- 予測ラベル: 90/90 件一致・不一致 0 件
- スコア: 完全一致 90/90・差の最大値 0

### 9-F. #346 の再発確認（guard）

`cargo test -p fandhe-edge-guard --test time_limit` を 50 回実行。

- 失敗: 0 回
- `ReadOutput`: 0 件・`ReadOutputIncomplete`: 0 件
- #346 に対応する 2 テスト（`req39_concurrent_spawns_do_not_inherit_each_others_pipes`・`req39_grandchild_holding_pipe_is_read_output_error`）も 50 回とも ok
- 開始時 load avg 2.58・終了時 3.81

## 11. 今回の対象外

- sandbox 下の通信 0 件確認（REQ-38。`sandbox-offline-check-procedure.md` の実機実行。人が実行）
- 静かな状態での実機 p95 計測（`--quiet-machine` フラグ使用時）
- 実データでの容量計測（REQ-30。人が実機で実行）
- GPU を使う学習
- C-3（自己回帰モデル。学習ワーカー側未実装）
