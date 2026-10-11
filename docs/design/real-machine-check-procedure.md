# Mac 実機での動作確認（項目 A〜J）の手順と記録簿

対応: REQ-18・REQ-21・REQ-26・REQ-27・REQ-28・REQ-30・REQ-31・REQ-32・REQ-33・REQ-34・REQ-39（横断の確認。spec の特定の TASK には対応しない）・#354・#469（G〜J と B の拡張）。REQ-38（sandbox 下の通信 0 件）は対象外（`sandbox-offline-check-procedure.md` の担当）。
spec の内容は要約であり、詳細は spec の REQ-18・REQ-21・REQ-26・REQ-27・REQ-28・REQ-30・REQ-31・REQ-32・REQ-33・REQ-34・REQ-39 を参照する。

## 1. 目的と範囲

PC を変えても同じ手順で main の動作確認（項目 A〜J）を再現できるようにする。G〜J は #469 の CLI 結線（`train --all`・`--cancel`・`--status`・再現性・旧モデルとの比較・版管理台帳）で増えた機能の確認で、実 trainer を使う（G・H は既定の項目、I は明示したときだけ）。
スクリプト（`scripts/real-machine-check.sh`・`scripts/real_machine_check_record.py`）は実行と記録の整形までを行い、
「実機」の証拠としての確定は人が行う。Agent の範囲はスクリプトと手順の準備までで、実機の結果を Agent が確定させない（`evidence_hint` は `requires_human_review` か `test_harness` のみ）。

## 2. 新しい PC での前提

1. リポジトリのルートで `make doctor` を実行して環境診断を行う（何も導入しない）。
2. Python 3.9 以上が必要。`python3 --version` で確認。
3. 以下は通信を伴うため、実行前に承認を取得する：
   - `make setup`: Rust ツールチェーン（`rust-toolchain.toml` で管理）。
   - `make py-sync`: `trainer/.venv`（Python 3、uv）・MLX（CPU）。
   - **`cargo fetch --locked`**: 依存を前もってダウンロードする（通信を伴うため承認を得てから実行する。`make setup` は cargo の依存を取得しない）。未実行の場合、実行時に `cargo build --locked` が失敗する（オフラインモード。REQ-38）
   - **ツールチェーンの導入**: `rust-toolchain.toml` の指すツールチェーンが未導入だと、実行は `cargo` の失敗（`toolchain ... is not installed`）で止まる（スクリプトは取得を止めて通信しないため。#375）。先に `make setup` か `rustup toolchain install` を、通信を伴うので承認を得てから実行する
4. 前提を満たしたら `make real-machine-check` を実行できる。

## 3. 実行方法

### 3-1. コマンドラインの形式

```bash
make real-machine-check ARGS="--work-dir <DIR> [--items <LIST>] [--repeat N] [--quiet-machine] [--with-ci] [--p95-limit-us N] [--package-limit-bytes N] [--overall-timeout-sec N] [--g-budget-seconds N] [--i-device cpu|gpu]"
```

または、スクリプトを直接実行する場合：

```bash
<repo>/scripts/real-machine-check.sh --work-dir <DIR> [options...]
```

### 3-2. 引数

値を取るオプション（`--work-dir` / `--items` / `--repeat` / `--p95-limit-us` / `--package-limit-bytes` / `--overall-timeout-sec` / `--g-budget-seconds` / `--i-device`）は `--key VALUE` と `--key=VALUE` の両方を受け付けます。重複・空要素・未知のオプションは拒否（exit 64）。

| 引数 | 説明 | 既定値 / 必須 |
| ---- | ---- | ---------- |
| `--work-dir DIR` | 作業と記録の置き場。存在しないか空であること。**作業ディレクトリがリポジトリ自身・その配下・その祖先のどれかなら拒否される。symlink・非ディレクトリ・空でないディレクトリも拒否される**（物理パスへ正規化して比較。末尾の `/`・`/.` は判定の前に取り除くため、`link/` や `link/.` でも symlink は拒否される。#364） | 必須 |
| `--items LIST` | 実行する項目。`A,B,C,D,E,F,G,H,I,J` の部分集合（カンマ区切り・大文字・重複不可）。E・J は B の成果物を使うため B と一緒に指定する（B なしの E・J は引数エラー〔exit 64。何も実行しない〕。#362）。`A`〜`J` とカンマ以外の文字（`*`・`?`・`[` 等）を含む値は、分割の前に拒否する（パス名展開でカレントのファイル名に化けない。#364）。A は通信しうる（`--with-ci`）、I は学習を 4 回行い GPU・時間を長く占有しうる（#103）ため、既定に入れず `--items I` で明示したときだけ実行する | `B,C,D,E,F,G,H,J`（A・I は既定では含まない） |
| `--with-ci` | A（`make ci`）を実行する明示の同意。通信を伴いうる（`uv sync`・advisory DB・`npx`）。`--items` に A があり `--with-ci` が無ければ引数エラー（exit 64）で、何も実行しない | — |
| `--repeat N` | F（ガード層の時間制限テスト）の実行回数。1 以上 1000 以下の整数 | 50 |
| `--quiet-machine` | 「他のアプリを閉じた静かな状態」という人の申告。p95 の分類が `real_machine` になるのは、これがあり、かつ `make`・`cargo` の代役も `FANDHE_EDGE_BIN` の差し替えも無いときだけ。それ以外は `reference_only` | — |
| `--p95-limit-us N` | C-1 の推論 p95 上限（マイクロ秒）。1 以上 3600000000 以下の整数（上限は定義ファイルの `limits.max_infer_p95_us` の上限と同じ。REQ-31）。範囲外は CLI・make・cargo を起動する前に引数エラー（exit 64） | 50000 |
| `--package-limit-bytes N` | C-2 の容量上限（バイト）。1 以上 999999999999999（15 桁）以下の整数（桁あふれを避けるための上限。シェルと Python が同じ値で検査する）。範囲外は引数エラー（exit 64） | 1000 |
| `--overall-timeout-sec N` | 実行全体の上限時間（秒。入力の採取・CLI のビルド・開始時の環境採取を含み、終了時の再採取と記録の書き出しは含まない）。1 以上 86400 以下の整数。範囲外は引数エラー（exit 64）。契約に定めのない暫定値（自分で決めた点。REQ-39）。超えたら §5 のとおり打ち切る。`--repeat` を大きくする場合は併せて上げる | 14400（4 時間） |
| `--g-budget-seconds N` | G（`train --all`）の探索予算（秒）。1 以上 921600 以下の整数（CLI の `--budget-seconds` の上限 `MAX_SEARCH_BUDGET_SECONDS` と同じ。範囲外は引数エラー〔exit 64〕）。小さくすると全候補が予算到達（exit 20）になりうる。それも想定内の結果として記録する（§4-G） | 3600（CLI の既定） |
| `--i-device cpu\|gpu` | I の証拠種別の申告。`cpu` → `cpu_real_machine`、`gpu` → `gpu_real_machine_declared`。**CLI の `train` は現状 CPU 固定**（`crates/cli/src/stages/train.rs` の `Device::Cpu`）のため、`gpu` は人が別の手段で GPU 学習へ切り替えたときだけ指定する（切り替えの有無はスクリプトが確認できない）。MLX の GPU 学習は同一 seed でも完全再現しないため、再現性の確認は CPU で行う（REQ-26・#103） | `cpu` |
| `--help` | 使い方を JSON 1 行（`{"code":"ok","message":"usage: ..."}` の形）で stdout へ出して exit 0 | — |

**A を実行する場合の注意**: `--items` に `A` を含めても、`--with-ci` が無ければ以下の JSON を出して exit 64（`invalid_input`）で停止します。実行順は常に A→J です。

```json
{"code":"invalid_input","message":"<固定メッセージ>"}
```

**`make real-machine-check ARGS=...` の制約**: シェルが `$(ARGS)` を単語分割するため、パスなど空白を含む値は渡せません。`ARGS` は **make のコマンドラインで渡した値だけ**を受け付けます（環境変数の `ARGS` は拾わず、スクリプトを起動せず固定のエラーで非 0 終了します。#364）。レシピは `set -f` でパス名展開を止めてから起動します（止まるのはパス名展開だけで、変数展開・コマンド置換は残るため、`ARGS` には自分で書いたリテラル値だけを渡してください）。空白を含む値はスクリプトを直接実行してください：

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
| `FANDHE_EDGE_BIN` | CLI のバイナリパス。`/` を含む相対・絶対パスで指定。相対パスは呼び出し時のカレント基準で絶対化される。存在しない・通常ファイルでない・実行権限が無い場合は、何も起動する前に引数エラー（exit 64。`FANDHE_EDGE_MAKE_CMD`・`FANDHE_EDGE_CARGO_CMD` の不正と同じ分類。#362）。未設定なら `cargo build --locked --release -p fandhe-edge-cli --bin fandhe-edge` でビルドして、`compiler-artifact` の executable を特定する（出力先は cargo が報告したパス）。ビルドの上限時間は 1800 秒 | `cli_profile:"release"`（未設定時）または `null`（設定時）。`evidence_hint` は MAKE_CMD / CARGO_CMD の有無で決まる（FANDHE_EDGE_BIN の有無は影響しない）。**差し替えた CLI が、cargo がビルドした CLI とバイト単位で同一でない限り D は失敗する**（`make`・`cargo` の代役の下では照合しないため対象外）: `make check-runtime-linkage` が検査するのは cargo がビルドした CLI（cargo の報告から取る。`build.target-dir`・`CARGO_BUILD_TARGET_DIR`・`CARGO_TARGET_DIR` のどれで出力先を変えていても追従する）で、その sha256 が差し替えた CLI と一致しなければ `linkage_target_mismatch`（§4-D） |
| `FANDHE_EDGE_MAKE_CMD`・`FANDHE_EDGE_CARGO_CMD` | 検査用（テスト専用の上書き）。**絶対パスの実行ファイル**（PATH で探さない）。どちらかを設定すると `evidence_hint` が `test_harness` に変わり、`otool -L` は実行されず `direct_libraries` が `null` になる。絶対パスでない場合や実行可能でなければ引数エラー（exit 64） | テスト用のみ。両方同時に設定可能 |
| `FANDHE_EDGE_TRAINER_DIR` | 学習ワーカー（`trainer/`）の場所。CLI の `train` 工程が読む（設定の有無は空文字も設定扱い）。このスクリプトは値を読まず、設定の有無だけを `environment.trainer_origin`（`env` または `build_default`）へ記録する。パスは記録しない | 未設定なら CLI をビルドしたツリーの `trainer/`（ビルド時の既定）を使う。`FANDHE_EDGE_BIN` で差し替えた CLI の既定は、その CLI をビルドしたツリーを指し、このリポジトリの `trainer/` とは限らない。`train` を使わない項目（A・D・F のみ）の実行でも、この欄は環境変数の状態を表す |

### 3-4. 実行時の上限（REQ-39）

スクリプトは各子プロセスに以下の上限を設けます。超過時は `timeout` または `output_limit` で失敗します。

| 対象 | 上限時間 | stdout | stderr | 用途 |
| ---- | ------- | ------ | ------ | ---- |
| CLI 1 工程（register / inspect / train / select / evaluate / package / infer） | 600 秒 | 1 MiB | 8 MiB | B・C・E・H・I・J |
| `train --all`（G） | `--g-budget-seconds` + 300 秒 | 1 MiB | 8 MiB | G（予算内の学習が終わり次第戻るため、余裕 300 秒を足す。契約に定めのない値） |
| `ps`・H のキャンセル要求（`train --cancel`） | 30 秒 | — | — | H（子孫の確認は固定の `/bin/ps`。待つ上限は running の出現 120 秒・子孫の終了 30 秒。契約に定めのない値） |
| `make ci` | 3600 秒 | 64 MiB | 64 MiB | A（通信するため `--locked` なし。`RUSTUP_AUTO_INSTALL=0` のみ渡す。#375） |
| `make check-runtime-linkage` | 1800 秒 | 64 MiB | 64 MiB | D（オフラインモード。`CARGO_NET_OFFLINE=true`・`RUSTUP_AUTO_INSTALL=0`） |
| `cargo test --locked` / 1 回（F） | 300 秒 | 64 MiB | 64 MiB | F（N 回の各回） |
| `cargo build --locked`（FANDHE_EDGE_BIN 未設定時） | 1800 秒 | 64 MiB | 64 MiB | ビルド（オフラインモード。`CARGO_NET_OFFLINE=true`・`RUSTUP_AUTO_INSTALL=0`。REQ-38） |
| `cargo test --locked ... --no-run`（F の事前ビルド） | 1800 秒 | 64 MiB | 64 MiB | F（オフラインモード。`CARGO_NET_OFFLINE=true`・`RUSTUP_AUTO_INSTALL=0`。REQ-38） |
| `otool -L`・`git` コマンド・環境採取 | 30 秒 | 64 KiB | 64 KiB | 環境情報 |
| 実行全体（`--overall-timeout-sec`） | 既定 14400 秒 | — | — | 全項目の合計。超過は §5 |
| 入力ファイル（定義・学習・評価） | — | 16 MiB | — | 読み込み前に確認 |
| CLI バイナリの sha256 計算 | — | 1 GiB | — | 計算時に確認（超過は計算せず `null`） |
| パッケージファイル個別（model.onnx・vocab.json 等） | — | 256 MiB | — | sha256 計算時に確認 |
| E の推論件数 | — | 1000 件 | — | `train.jsonl` 件数上限 |

**環境採取の子（`git`・`sysctl`・`sw_vers`・`otool`）**は、`PATH` を探さず固定の絶対パス（macOS は `/usr/bin/git`・`/usr/sbin/sysctl`・`/usr/bin/sw_vers`・`/usr/bin/otool`。Linux のテストハーネスは `/usr/bin/git` か `/bin/git` だけで、実機の証拠にならない）で起動し、最小の環境（`PATH` 固定・`LC_ALL=C`・親にあるときだけ `HOME`）だけを渡します。親の `GIT_DIR`・`GIT_WORK_TREE`・`DEVELOPER_DIR` 等は届かないため、`PATH` の先頭の同名の実行ファイルや環境変数で、別物・別リポジトリの値が `commit`・機種・OS・直接リンクの欄に入ることはありません（#364）。`HOME` を残すのは、利用者のグローバル設定（`safe.directory` 等）が効かないと所有者の違うチェックアウトで `commit` が取れなくなるためです。固定パスに無い場合は「使えない」として扱い、`PATH` へは戻りません（`git` が無ければ `commit`・`worktree_clean` が `null` → `stable` が `null` → exit 10。`sysctl`・`sw_vers` が無ければ該当欄が `null`。macOS で `otool` が無ければ D は `failed` / `otool_failed`）。`make`・`cargo`・`python3` は利用者のツールチェーンで場所が決まるため `PATH` のままです。

スクリプトと `check-runtime-linkage.sh` が自分で起動する `cargo` は、すべて `--locked`（`Cargo.lock` を暗黙に更新しない）。A の `make ci` は make の中の cargo であり、スクリプトは `--locked` を付けない。

**rustup のツールチェーン自動取得を止める（REQ-38・#375）**: スクリプトは A を含む全ての子に `RUSTUP_AUTO_INSTALL=0` を渡す（親に同名の変数があっても `0` で上書きする）。`CARGO_NET_OFFLINE` は cargo 自身の設定で rustup プロキシを止めないため、別に必要になる。A の `make ci` には `CARGO_NET_OFFLINE` を渡さず（`--with-ci` の同意が覆う通信は許す）、ツールチェーンの自動取得だけを止める。`rustup set auto-install disable` は `settings.toml` を書き換えるため使わない。ツールチェーンが無いときの失敗は既存の分類で現れる（CLI のビルド失敗は exit 70・`cannot build the CLI`、F は `build_failed`・`list_failed`）。`record.json` のスキーマは変えず、この抑止は `options.cargo_offline` とは別で、記録には出ない。

- **確かめた内容**（証拠種別: 模擬。Linux x86_64・rustup 1.29.1 の実バイナリ。通信は loopback の閉じたポートだけ）: 空の `RUSTUP_HOME`・`RUSTUP_DIST_SERVER=http://127.0.0.1:1` で `rust-toolchain.toml`（stable）のあるディレクトリから `~/.cargo/bin/cargo --version` を実行すると、環境変数なしでは `syncing channel updates` の後にダウンロードを試みた（自動取得は起きる）。`RUSTUP_AUTO_INSTALL=0` では `toolchain ... is not installed` で即停止し、ダウンロードを試みない。出典: rustup 1.29.1 の文言（`you may opt out with RUSTUP_AUTO_INSTALL=0`）と rust-lang/rustup#4836。macOS 実機では未確認
- **既知の限界**（いずれも証拠種別: 推定・未確認）: ① `RUSTUP_AUTO_INSTALL` を認識しない古い rustup では止められない（版の境界は公式文書を参照しておらず未確認）。② rustup を経由しない cargo（直接配置・`FANDHE_EDGE_CARGO_CMD` の代役）には効果がない。③ `RUSTUP_TOOLCHAIN` や `+toolchain` で別のツールチェーンを指す場合の挙動は未確認。④ コンポーネント（rustfmt・clippy）欠落時の自動導入が同じ変数で止まるかは未確認

### 3-5. 運用上の注意（#365）

実行の前に知っておくと原因の切り分けが早くなる点です。証拠種別は各項に書きます。

- **出力先を変えた環境でも D は追従する**（証拠種別: 推定。コードの読解で、出力先を変えた実行での確認はしていない）: CLI のビルド（`build_cli`）も `check-runtime-linkage.sh` も、CLI の場所を cargo の報告（`compiler-artifact` の `executable`）から取る。どちらもリポジトリのルートで同じ環境の cargo を起動するため、`build.target-dir`・`CARGO_BUILD_TARGET_DIR`・`CARGO_TARGET_DIR` のどれで出力先を変えても同じ実物を指す。`target/` を決め打ちする箇所はない。D が `linkage_target_mismatch` になるのは、`FANDHE_EDGE_BIN` で差し替えた CLI が cargo のビルドした CLI とバイト単位で違うときである（§3-3・§4-D）
- **色を強制する環境では A が `no_test_results` になりうる**: A の合否は `make ci` の stdout を行頭一致で読んで決める（Rust の `test result:` 行・pytest の要約行）。ANSI エスケープ（色）は取り除かないため、どちらかの行に色が付くと読めず、exit 0 でも `no_test_results` で `failed` になる（`rust_tests.passed` が 0、または `pytest` が `null`）。
  - 確認できた発生条件（証拠種別: 模擬。計画時の小さな試行で、`make ci` 全体での再現ではない）: pytest 9.1.1 は `FORCE_COLOR=1` または `PY_COLORS=1` で、出力先がファイルでも要約行に色が付く（`NO_COLOR=1` は `FORCE_COLOR=1` に勝つ）。cargo 1.98.1 の libtest の `test result:` 行は、`CARGO_TERM_COLOR=always`・`.cargo/config.toml` の `[term] color`・`FORCE_COLOR`・`CLICOLOR_FORCE` のいずれでも、出力先がファイルなら色が付かなかった（色が付いたのは `-- --color always` を渡したときだけで、`make ci` も F もこれを渡さない）
  - 見分け方: A が `exit_code: 0`・`reason: no_test_results` で、`<work-dir>/A/make-ci.log` に ESC（0x1b）が含まれる
  - 対処: `FORCE_COLOR`・`PY_COLORS` を外して、新しい `--work-dir` で実行し直す。判定は `failed` のままで、色つきの結果を合格と読み替えない。スクリプト・Makefile 側で色を無効にする改善は本手順の範囲外（別課題の候補）
- **stdout を `head` などのパイプへつなぎ、先に閉じられると 70 になる**（証拠種別: テストハーネス。実機の確認ではなく、macOS 実機での実測は人が行う）: 最終 JSON を stdout へ書けなかった場合の終了コードは、本来の値によらず 70（REQ-21）。判定の中身は `<work-dir>/record.json` で確認する（記録を作った後の最終出力に失敗した場合に限り、`record.json`・`record.md` は stdout の書き込みより前に書き終えている。引数エラーは記録を作る前に終了するため、両ファイルは存在しない）。traceback や `Exception ignored` は stderr に出ない
- **スクリプト自身が SIGKILL・SIGQUIT で落ちると、子が残る**（証拠種別: 推定。コードの読解で、この場合を固定するテストはない）: ハンドラを置くのは SIGINT・SIGTERM・SIGHUP だけで、SIGQUIT（端末の Ctrl-\）は既定の動作（終了）、SIGKILL は捕捉できない。子は別セッション・別プロセスグループで動くため、スクリプトの終了も端末からのシグナルも子へ届かない。`record.json`・`record.md` は書かれず、stdout の最終 JSON も出ない。実行中の子は自然に終わるまで走り続け、上限時間・出力上限の監視も効かなくなる。`ps` で残りを確認して手で止め（`pkill -f` のような広い一致で無関係なプロセスを止めない）、同じ `--work-dir` は空でないため使えない（exit 64）ので新しいディレクトリで実行し直す。§5「中断の方式」の既知の限界を参照
- **stdout が詰まっていると、中断シグナルでは止まらない**（証拠種別: テストハーネス。偽の読み手で stdout のパイプを詰まらせた再現。Mac16,6・64 GiB・macOS 27.0・Python 3.12.12・2026-10-05 で、SIGINT・SIGTERM・SIGHUP のどれも 5 回中 5 回止まらなかった）: 読み手が stdout を読まないと、最後の JSON を書く `emit` の書き込みでブロックする。この間は 3 つの中断シグナルを送っても終わらず、SIGKILL でしか止められない（REQ-21・REQ-39）。
  - 原因: ハンドラは印を立てるだけで、印を見るのは子を待つループと項目の境目だけなので、`emit` の書き込みの待機には効かない。書き込みの待機にも上限がない。さらに `_entry` は `main` を抜ける直前に 3 つのシグナルを `SIG_IGN` にするため、終了処理中のシグナルは無視される
  - 時期: #370 より前から同じ挙動で、#370 による後退ではない
  - 対処 (a): 読み手側を読ませる。呼び出し元が stdout を読み続ける（`cat` へパイプで繋ぐ、ファイルへリダイレクトするなど）
  - 対処 (b): 止まったままなら SIGKILL でスクリプトを止める。直前の項目と同じく子が残りうるため、`ps` で残りを確認して手で止め（`pkill -f` のような広い一致は使わない）、同じ `--work-dir` は空でないため使えない（exit 64）ので新しいディレクトリで実行し直す
  - 補足（証拠種別: 推定。コードの読解）: `record.json`・`record.md` は最終 JSON の `emit` より前に書かれる（`main` の書き出しの後に `emit` を呼ぶ）ため、書き込みで止まった時点では書き終わっている場合が多い
  - 実装で止められるようにする（stdout 書き込みへの上限時間など）かは本手順では決めない。必要になれば別 Issue で扱う

## 4. 各項目 A〜J が何を確かめるか

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
- **失敗の理由**: `skipped`（skip 行あり）、`no_test_results`（Rust の `passed` が 0、または pytest の要約行が無い・`passed` が 0。色つきの行は読めないためこれになりうる。§3-5）、`test_failures`（failed あり）、`unexpected_exit_code`（0 以外）、`output_limit`（stdout が上限超過）、`output_unreadable`（ログが読めない）、`timeout`・`spawn_error`・`killed`（実行の失敗）
- **記録する `record.json` フィールド**: 終了コード（`exit_code`）、`skip_lines`（`skip:` で始まる行の件数）、`rust_tests`（`passed`・`failed`・`ignored`）、`pytest`（`passed`・`skipped`・`failed`）、`stderr_bytes`
- **pytest の skip**: Mac では Linux 用の POSIX ACL テスト 3 件が skip になる。検証済みと扱わない。記録には残す

**注意**: A の中で実行されるテスト（Rust の unit・integration テスト、pytest）はテストハーネスで、実機の証拠にはなりません。

### 4-B. 評価データありの 7 工程（`register → inspect → train → select → evaluate → package → infer`）

入力データ（`fixtures/sandbox_run_eval/`）をコピーして、7 工程を順に実行し、正常系での完走を確かめます。学習は CPU で、`--smoke` は付けません。候補は 1 個（ID 0）。

- **確かめること**: 各工程が連続して終了コード 0 で成功し、出力される JSON の報告値が、作業ディレクトリへコピーした fixture から導いた値と整合すること。`package` の容量内訳（5 項目）が合計と一致すること。
- **想定する終了コード**: 0（全工程成功）
- **B の成功条件**: 7 工程すべてが exit 0 で `status` が `ok`（`infer` だけは exit 11・12 も可で、判定行の `status` がそれぞれ `out_of_scope`・`abstain` と一致すること。#506）、かつ各工程の報告値が次のとおり整合し、`capacity` の 5 要素（`weights`・`vocab_or_feature_transform`・`label_table`・`calibration`・`metadata`）が各々 `bytes` と `file_count`（0 以上の整数）を持ち、5 要素の合計が `capacity.total_bytes` と一致する（C も同じ照合を通る。REQ-21・REQ-33）。さらに、公開された `package/` 直下が通常ファイルだけ（symlink・ディレクトリ・FIFO 等があれば失敗）で、そのバイト数の合計が `capacity.total_bytes` と、ファイル数が `capacity` の 5 要素の `file_count` の合計と一致する（REQ-30。計測の対象は組み立て先から公開される集合そのものなので、`package/` の全ファイルが数えられる。#362）。`==` で比べる報告値（`options`・`candidate`・`n_total` など）は、比べる前に真偽値を除く整数であることを確かめる
  - `register`: `options` が定義の選択肢の数、`evaluation_defined` が「評価データの行数が 1 以上」と一致
  - `inspect`: `valid_records` が `train.jsonl` の行数（空行を除く）と一致し、`split` の `train`・`validation`・`test` の和が `valid_records` と一致
  - `train`・`select`: `candidate` が 0（B・C は候補 0 を学習する）で、`kind` が `c1`・`c3`・`autoregressive` のどれか（`crates/guard/src/kind.rs` の `SUPPORTED_KINDS` と一致することを pytest が機械照合する）。`select` の `kind` は `train` と同じ値
  - `evaluate`: `candidate` が `select` の報告値、`kind` が `select` と同じ値、`n_total` が `evaluation.jsonl` の行数（1 以上）、`correct` が 0 以上 `n_total` 以下、`accuracy` が `correct / n_total` と 1e-9 以内で一致、キー `macro_f1` が存在して `null` か 0 以上 1 以下（分母 0 の指標は `null`。REQ-24）
  - `package`: キー `judgment`・`infer_p95` が値が `null` でも存在する。`capacity` は `total_bytes`・`exceeded`・`guideline_bytes`・`over_guideline` が正しい型で、`limit_bytes` が定義の `limits.max_package_bytes` と一致する。定義に無いときは `limit_bytes` が `null` で `exceeded` が `false`（上限を照合しない。REQ-30・TASK-41.9・#406）。設定があるときは `exceeded == (total_bytes > limit_bytes)`、exit 0 なら `exceeded` は `false`。`guideline_bytes` は pytest が runtime の `REFERENCE_CAPACITY_BYTES` と目安の値を照合し、`over_guideline == (total_bytes > guideline_bytes)` を検査する（目安超過は警告で、exit 0 に影響しない）。B の定義は合否基準と上限を持たないため、`judgment` が `null`・`acceptance_defined` が `false`・`infer_p95` が `null`
  - `infer`（B の単発）: `predicted_label` が定義の選択肢 ID のどれか、`scores` のキー集合が選択肢 ID と一致し、各値が有限で 0 以上 1 以下、和が 1 から 1e-6 以内（`SCORE_SUM_TOLERANCE` は `fixtures/score_tolerance/score_sum_tolerance.json` と一致することを pytest が機械照合する）、`predicted_label` が最大スコアの選択肢（同点は定義の宣言順で先頭）。B の単発は `--id` を付けないので、`id` が既定値 `input`（`crates/cli/src/stages/infer.rs` の `DEFAULT_TEXT_ID` と一致することを pytest が機械照合する。#362）
- **#469 で増えた出力の照合（B のみ。`contract_checks`）**: 7 工程の報告値の照合に加えて、次を確かめる（欄名・形は `crates/core/src/stage_report.rs` の `EvaluateCompletedReport`・`EvaluateCalibration`・`EvaluateAbstention`・`EvaluateDiagnostics`・`SelectReport`・`PackageReport`・`PackageVersion`。バッチの契約は #469 のコメント）。REQ-22・REQ-25・REQ-29・REQ-30・REQ-39
  - `evaluate` の `calibration`: null でなく、`temperature` が正の有限数・`adopted` が真偽値・`threshold` が 0〜1・`n_validation` が 1 以上・`validation_coverage` が 0〜1（B は評価データありのため null を失敗にする）
  - `evaluate` の `abstention`: `answered`・`abstained`・`out_of_scope`・`correct_answered` が 0 以上の整数、`coverage` が 0〜1、**`answered + abstained == n_total`**（`out_of_scope` は `answered` の内数で、`out_of_scope <= answered`。`stage_report.rs` の `EvaluateAbstention` の doc に従う。B は対象外ラベルを定義しないため `out_of_scope` は 0 で、`answered + abstained + out_of_scope` としても同じ値になる）、`correct_answered <= answered`
  - `evaluate` の `diagnostics`: 非 null で、`train`・`eval`（`eval.n_rows == n_total`）・`confusable_pairs`・`limitations`・`data_volume` を持つ
  - `select` の `significance`: キーが存在し、null（定義に `baseline_comparison` が無い。B の定義はこれ）か `verdict` が `significantly_better`・`not_significantly_better`・`undeterminable` のどれか。**`undeterminable` も正常**（件数不足は合格扱いにしないが、失敗にもしない。REQ-25）
  - `package` の `version`: `{"id":"v1","previous":null}`（B は `--previous-project-dir` なし）。`project/version_ledger.json` が通常ファイルで、`schema_version` が 1・今回の版の `model`・`data`・`experiment` の 3 件（`sha256` は hex 64 桁）を持つ（台帳ファイル自体の改変検出は範囲外。#491）
  - 校正の束縛: `package/artifact.json` の `calibration_sha256` が `package/calibration.json` の実際の sha256 と一致し、`capacity.components.calibration` が `calibration.json` の実サイズ・`file_count` 1（REQ-39・#497）
- **失敗の理由**: `input_unreadable`（fixture が読めない）、`missing_field`（JSON に必須フィールドなし）、`package_unreadable`（`package/` が読めない）、`package_entry_not_regular`（`package/` 直下に通常ファイル以外がある。#362）、`capacity_sum_mismatch`（容量の 5 要素の合計と `total_bytes` が一致しない）、`calibration_invalid`・`abstention_invalid`・`diagnostics_invalid`・`significance_invalid`・`version_invalid`・`version_ledger_invalid`・`calibration_binding_mismatch`・`calibration_capacity_mismatch`（上の #469 の照合。順にそれぞれの欄の不整合）、工程別の失敗（`step` に工程名・`exit_code` に終了コード・`reason` に `timeout`・`output_limit`・`spawn_error`・`killed`・`invalid_json`・`unexpected_exit_code`・`unexpected_output`。報告値の不整合は `unexpected_output`。非 0 終了で stdout が JSON でない場合は `invalid_json` のまま、許容外の終了コードであることを補助欄 `exit_code_unexpected: true` で示す。stdout の JSON の入れ子が深すぎて読めない場合も `invalid_json`）。`package/` のファイルの合計が `total_bytes` と、ファイル数が `file_count` の合計と合わない場合は `unexpected_output`（`step` は `package`）。単発 `infer` の `id` が既定値でない場合も `unexpected_output`（`step` は `infer`）
- **記録する `record.json` フィールド**:
  - `status`: `"ok"` または `"failed"`
  - `steps`: 工程の配列（各工程：`step`（固定語彙）・`command`（固定語彙。infer は `infer --package <package-dir> --text <fixed-sample>` で、引数の値・パスは出さない）・`exit_code`・`stderr_bytes`・`summary`）。`summary` は工程ごとに決まった欄だけを新しく組み立てた要約で、CLI の JSON の他の欄は捨てる（欄は常に出し、無い・型が違う値は `null`、閉じた語彙の欄で語彙外の文字列は `<unexpected>`）
    - `register`: `step`・`status`・`options`・`evaluation_defined`・`definition_sha256`
    - `inspect`: `step`・`status`・`valid_records`・`split`（`train`・`validation`・`test`）
    - `train`・`select`: `step`・`status`・`candidate`・`kind`
    - `evaluate`: `train` の 4 欄に加えて `n_total`・`correct`・`accuracy`・`macro_f1`
    - `package`: `step`・`status`・`code`・`judgment`・`acceptance_defined`・`capacity`・`infer_p95`
    - `infer`: `status`・`scores_keys`（`scores` のキー数）・`predicted_index`（`predicted_label` が定義の選択肢 ID の何番目か。0 始まり。選択肢に無ければ `null`）。選択肢 ID は利用者が決める文字列のため `predicted_label` は記録しない
  - `capacity`: `total_bytes`・`limit_bytes`（未設定なら `null`）・`exceeded`・`guideline_bytes`・`over_guideline`・`components`（5 項目。各々 `bytes` と `file_count`）
  - `capacity_sum_matches_total`: 5 要素の合計が `total_bytes` に一致したか（boolean）
  - `contract_checks`: 上の #469 の照合の要約。`calibration`（`n_validation`・`adopted`）、`abstention`（`answered`・`abstained`・`out_of_scope`・`coverage`）、`diagnostics_present`、`significance_verdict`（null か語彙内の値）、`version_number`（1）、`version_ledger_present`、`calibration_sha256_matches`。選択肢 ID・ラベルは出さない
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
  - C-1 の `capacity` が要約でき（`total_bytes`・`guideline_bytes` が 0 以上の整数、`exceeded` と `over_guideline` が boolean で `over_guideline == (total_bytes > guideline_bytes)`、5 項目の `bytes` の合計が `total_bytes` と一致）、定義に `limits.max_package_bytes` が無いため `limit_bytes` が `null` で `exceeded` が `false`（上限を設定していない C-1 の定義での検査。目安超過の `over_guideline` は警告として記録するだけで、合否には使わない）
  - exit 0 のときは、公開された `package/` 直下が通常ファイルだけで、そのバイト数の合計が `capacity.total_bytes` と、ファイル数が `file_count` の合計と一致する（C-1 は一覧を記録へ足さず、照合だけ行う。REQ-30）
  - exit 0・20 のどちらでも、`package` の実行後に `package.staging/`（公開前の組み立て先）が残っていない（公開後・失敗後に残らない設計。REQ-30。#362）
  - 工程の検査（B と同じ。`kind` の一貫性・`evaluate` の `accuracy`・`macro_f1`・`package` のキー `judgment`・`infer_p95` の存在）を通る
- **失敗の理由**: `unexpected_output`（`exceeded` と計算値・終了コードの不一致、`code`・`status` の不整合、`package/` の有無と終了コードの不一致、`limit_us` の不一致、`p95_us`・`limit_us` が負数、`capacity` の欠落・`exceeded` が `true`（上限を設定していない C-1 では起きない）・内訳の合計の不一致・`limit_bytes` の不一致（`null` でない・定義と違う）・`over_guideline` と計算値の不一致、`package/` のファイルの合計・件数と `total_bytes`・`file_count` の合計の不一致、`package/` に通常ファイル以外があるのいずれか）、`staging_left`（`package.staging/` が残っている。#362）、`missing_field`（C-1 の判定〔`judge_p95`〕が `infer_p95` の型違い・欠落を見つけたとき。通常は工程の検査が先に `unexpected_output` で止める）、工程別の失敗
- **記録する `record.json` フィールド**:
  - `p95.p95_us`・`p95.limit_us`・`p95.exceeded`・`p95.classification`（`real_machine` / `reference_only`）・`p95.package_exit_code`（0 または 20）・`p95.package_published`（C-1 の `package/` の有無）・`p95.package_staging_present`（C-1 の実行後に `package.staging/` が残っていたか。成功時は常に `false`。#362）
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
- **失敗の理由**: `unexpected_output`（工程の検査が先に止めるもの。`package` の JSON の `step` の不一致、`capacity` の欠落・上限値の不一致・内訳の合計の不一致・`exceeded` と計算値の不一致、`infer_p95` が null でない〔C-2 の定義に p95 の上限は無い〕など報告値の不整合。exit 20 で `code` が `limit_exceeded` でない場合、または容量も p95 も超過と報告されていない場合もここで `unexpected_output` になる。`case` は `C-2`）、`capacity_limit_not_enforced`（工程の検査を通ったあとの C-2 の判定〔`judge_capacity_limit`〕で落ちるもの。実際に届くのは 2 つだけ〔ほかに `package.staging/` が残っていれば `staging_left`〕で、`package` が exit 0 で報告値に不整合がない〔上限を超えていないと報告された〕場合と、exit 20 なのに `package/` が公開されている場合）、工程別の失敗（`package` が exit 0・20 以外のときは `unexpected_exit_code`。C-2 の許容終了コードは 20 だけだが、exit 0 は許容外として工程失敗にはならず、工程の検査〔`_step_check`〕へ進む。そこで `status` が `ok` でない・報告値が不整合ならば `unexpected_output` で止まり、整合していれば C-2 の判定へ届いて `capacity_limit_not_enforced` になる）。C-2 の判定にも `capacity` の欠落・上限値の不一致で `unexpected_output` を返す分岐があるが、工程の検査が先に同じ条件で止めるため通常は届かない。工程の検査で止まった失敗には `capacity_limit` の欄は付かず（`case`・`steps`・`p95` のみ）、`capacity_limit` は C-2 の判定まで届いたときだけ記録される
- **記録する `record.json` フィールド**:
  - `status`: `"ok"` または `"failed"`
  - `p95`: C-1 の結果（再利用）
  - `capacity_limit.package_exit_code`（20）、`.code`（7 種の語彙の値。語彙外の文字列は `<unexpected>`、文字列でなければ `null`）、`.capacity_exceeded`（boolean または null）、`.total_bytes`・`.limit_bytes`、`.infer_p95_exceeded`（boolean または null）、`.package_published`（`package/` ディレクトリの存在）、`.package_staging_present`（実行後に `package.staging/` が残っていたか）
  - `reason`（失敗時のみ）: `capacity_limit_not_enforced`（`capacity_limit` 欄を伴う場合。判定が `unexpected_output` を返す分岐は上記のとおり通常届かない）、`staging_left`（`package.staging/` が残っている。`capacity_limit` 欄を伴う）。工程の検査で止まった失敗は `reason` が `unexpected_output`・`unexpected_exit_code` などで、`capacity_limit` 欄は付かない

#### 4-C-3. C の失敗記録の形（`case`・`steps`）

失敗した場所で `case`・`steps` の付き方が変わる（成功時はどちらも付かない）。`case` の値は `C-1`・`C-2` の 2 つだけ。`steps` の各要素の形は B と同じ（§4-B）。`record.md` の「要点」は `case` を出すが `steps` は出さない。

| 失敗した場所 | `case` | `steps` | ほかの欄 |
| ---- | ---- | ---- | ---- |
| C-1 の入力の複製 | なし | なし | `reason: input_unreadable` だけ |
| C-1 の工程 | `C-1` | あり（失敗した工程まで。fixture を読めず始められなければ空の配列） | `reason`・`step`・`exit_code` ほか工程失敗の欄 |
| C-1 の判定（p95・`package/` の有無と終了コード・`package/` の合計と件数・`staging_left`） | `C-1` | なし | `step: package`・`exit_code`・`reason` |
| C-2 の入力の複製 | なし | なし | `reason: input_unreadable` だけ（`p95` も付かない） |
| C-2 の工程 | `C-2` | あり | `p95`（C-1 の結果）＋工程失敗の欄 |
| C-2 の判定（`capacity_limit_not_enforced`・`staging_left`） | なし | なし | `p95`・`capacity_limit`・`reason` |

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
- **失敗の理由**: `unexpected_exit_code`（0 以外）、`skipped`（skip 行あり）、`no_test_results`（env -i テストの `ok:` 行が 0 件）、`unexpected_output`（`ok: req32_*` が 3 件でない・`OK: tool=` の行が無い・macOS で tool が `otool` でない）、`linkage_target_unreadable`（`cli_bin:` 行が無い・2 行以上ある・絶対パスでない、または検査対象を読めない・上限超過）、`cli_changed`（いまの CLI が開始時と異なる、または読めない）、`linkage_target_mismatch`（検査対象が実行した CLI と異なる）、`otool_failed`（macOS で `otool -L` に失敗）、`output_unreadable`（`make` のログが読めない）、実行の失敗（`timeout`・`output_limit`・`spawn_error`・`killed`）
- **出力先の変更**: `build.target-dir` などで出力先を変えていても、検査対象は cargo の報告から取るため追従する（§3-3・§3-5）
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

- **確かめること**: 1 件ずつの推論とバッチ推論の予測ラベルが全件一致し、スコアに NaN・無限大がなく、スコアが全件で完全に一致すること。
- **想定する終了コード**: 0
- **E の合格条件**: B が `ok` + 次の整合 + 予測ラベル不一致 0 件 + NaN・無限大 0 件 + スコアが全件で完全に一致（`scores_exact_match` が件数と同じ）
  - 根拠: 同一実装の単体対バッチはスコアも含めて完全一致を要求する（`runtime-batch-mismatch-procedure.md`。REQ-28）。スコア差の最大値は参考値で、合否には使わない
  - バッチ出力の行数が入力の件数と一致し、各行が JSON で `status` が `ok`・`out_of_scope`・`abstain` のいずれか（#506）・`id` が文字列で重複なし、かつ出力が入力順（`i` 行目の `id` が入力の `i` 件目の `id` と一致。`crates/cli/src/infer_batch.rs` が入力順に出す。#362）
  - バッチの各行と単体の出力が、どちらも infer の規則（`predicted_label` が選択肢 ID・`scores` のキー集合が選択肢 ID と一致・各値が有限で 0 以上 1 以下・和が 1 から 1e-6 以内・`predicted_label` が最大スコアの選択肢〔同点は定義の宣言順で先頭〕）を満たし、単体の出力の `id` が入力の `id` と一致
- **E が実行されない場合**（`not_run`）: 前の項目が失敗している（`previous_item_failed`。B の失敗を含む）。E を B なしで指定した場合は起動前に引数エラー（exit 64。§3-2）で、記録は作られない
- **失敗の理由**: `input_unreadable`（学習データが読めない）、`record_count_out_of_range`（件数 0 または 1000 超）、`duplicate_id`（id の重複）、`mismatch`（予測ラベル・`status` が一致しない、またはスコアが完全一致でない〔少しでも違えば失敗〕。NaN・無限大も含む）、`unexpected_output`（バッチ側は行数の不一致・`status` が 3 値〔`ok`・`out_of_scope`・`abstain`〕のいずれでもない・出力に `step` の欄がある・`id` が文字列でない・`id` の重複・出力順が入力順でない・infer の規則の不整合〔`step` は `infer-batch`〕。単体側は stdout が JSON オブジェクトとして読めない〔JSON でない・入れ子が深すぎる・空の dict〕・`status` が終了コード〔0→`ok`・11→`out_of_scope`・12→`abstain`〕と対応しない・出力に `step` の欄がある・`id` が入力と一致しない・infer の規則の不整合〔`step` は `infer-single`〕）、`invalid_json`（バッチの行が JSON として読めない〔JSON でない・入れ子が深すぎる〕場合だけ。`step` は `infer-batch`。単体側は `invalid_json` にならない）、工程別の失敗（`step:"infer-batch"` / `"infer-single"`。`timeout`・`output_limit`・`spawn_error`・`killed`・`unexpected_exit_code` を含む）
- **記録する `record.json` フィールド**:
  - `status`: `"ok"` / `"failed"` / `"not_run"`
  - `reason`（`not_run` 時）: `previous_item_failed`
  - `records`: 推論した件数
  - `label_match`・`label_mismatch`: 予測ラベルの一致数・不一致数
  - `scores_exact_match`: スコアが完全に一致した件数（合否に使う。件数と同じでなければ失敗）
  - `scores_nonfinite`: スコアに NaN・無限大があった件数（合否に使う。1 件でもあれば失敗）
  - `max_abs_score_diff`: 有限なスコア差の最大値（非有限は除外）。参考値で、合否には使わない
  - `input_sha256`: 推論に使った入力ファイルのハッシュ
  - `reason`（失敗時のみ）: 上記の失敗の理由のいずれか（`mismatch` は予測・スコアの不一致。語彙の一覧は §5）。`records` の欄は `record_count_out_of_range`・`duplicate_id` のときに付く
  - 失敗時の詳細は `.../E/mismatch-ids.txt` に以下のいずれかに該当する `id` を記録（1 行ずつ）：予測ラベル不一致・スコアに NaN・無限大・スコアが完全一致でない（少しでも違う）

### 4-F. ガード層の時間制限テスト再発確認（#346）

`cargo test --locked -p fandhe-edge-guard --test time_limit` を N 回実行し、1 回も失敗しないこと（#346 の再発〔並行 spawn の fd 継承競合〕が無いこと）を確かめます。

- **確かめること**: テストバイナリのコンパイル（初回）が成功し、その後 N 回のテスト実行がすべて成功すること。#346 の再発の兆候として、失敗した回のログにある `ReadOutput`・`ReadOutputIncomplete` を数えて記録する（成功した回のログは数えない。成功の判定は後述の各回の合格条件で、これらの語が成功した回に出ても失敗にしない）。1 回あたりに実行されるべき件数は、事前に `cargo test --locked -p fandhe-edge-guard --test time_limit -- --list` を 1 回実行して得る（固定値と比べない。#362）。
- **想定する終了コード**: 0（失敗 0 回）または 10（失敗あり）
- **F の成功条件**: テストバイナリビルド成功 + 実行回数 N 回 + 失敗 0 回（各回が exit 0・期限内・出力上限内で、ログの `test result: ok.` 行の `passed` の合計が `--list` の件数（`expected_tests`）と一致し、`failed`・`ignored` が 0）
- **失敗の理由**: `build_failed`（`--no-run` のビルドが失敗・期限超過・出力上限超過）、`list_failed`（`--list` が失敗・期限超過・出力上限超過。#362）、`no_tests_listed`（`--list` が 0 件。#362）、`test_failures`（N 回のうち 1 回以上の失敗。失敗の内訳は `timeouts`・`no_tests`・`count_mismatch`・`output_limit`・`killed`・`spawn_error` の件数欄に出る）。これらの件数の語は `reason` ではなく件数の欄
- **記録する `record.json` フィールド**:
  - `status`: `"ok"` / `"failed"`
  - `runs`: 実行回数（N）
  - `expected_tests`: `--list` で得た 1 回あたりの件数（整数。#362）
  - `passed`・`failed`: 成功・失敗の回数
  - `timeouts`: 制限時間超過の回数
  - `no_tests`: exit 0 で終わったが 0 件しか実行されなかった（`passed` も `failed` も 0）回数（失敗に数える）
  - `count_mismatch`: exit 0 で終わったが `passed` が `expected_tests` と違う、または `failed`・`ignored` が 0 でなかった回数（失敗に数える。#362）
  - `output_limit`・`killed`・`spawn_error`: 出力上限超過・子の回収の失敗（rc の欠落）・起動失敗の回数（それぞれ失敗に数える。#362）
  - `read_output`: 失敗した回のうち、ログに `ReadOutput`（Incomplete 以外）が出現した回数（成功した回は数えない）
  - `read_output_incomplete`: 失敗した回のうち、ログに `ReadOutputIncomplete` が出現した回数（成功した回は数えない）
  - `load_start`・`load_end`: 実行開始・終了時の load average（3 要素の配列。`os.getloadavg()` で取れなければ `null`）
  - `reason`（失敗時のみ）: `build_failed` / `list_failed` / `no_tests_listed` / `test_failures` / `unreaped`（KILL 後の回収が上限時間内に終わらず、その回で打ち切る。`runs` が `--repeat` 未満になる。`child_may_remain` が `true`。§5）
- **実行特性**: F だけは全 N 回を数えてから成否を決めます（失敗があっても止めず最後まで実行する）。失敗 0 回で `ok`、1 件以上で `failed`。
- **初回実行の自動準備**: スクリプトが N 回実行の前に `cargo test --locked ... --no-run` を 1 回実行し、テストバイナリをコンパイルします（回数に数えない。失敗は `build_failed`）。続けて `-- --list` を 1 回実行して件数を得ます（回数に数えない。失敗は `list_failed`）。事前準備は不要ですが、初回実行は通常より時間がかかります。

### 4-G. `train --all`（探索予算内の全候補の学習。REQ-18・#482・#483）

新しい project（`G/project`）で `register → inspect` の後に `train --project-dir project --all --budget-seconds N` を実行します（`N` は `--g-budget-seconds`、既定 3600）。実 trainer（CPU）で既定候補を宣言順に学習するため、予算の分だけ時間がかかります。

- **確かめること**: 候補ごとの結果（`candidates[].result`）と予算到達（`budget_reached`）が契約の語彙で出ること、探索記録 `search_record.json` が残ること、exit 0（`evaluated` が 1 件以上）か exit 20（全件が予算到達）であること。
- **想定する終了コード**: 0 または 20（どちらも想定内。exit 20 の stdout は `code:"limit_exceeded"` のエラー JSON で、`candidates` を持たない）
- **G の合格条件**:
  - exit 0 または 20。それ以外は `unexpected_exit_code`
  - どちらでも `project/search_record.json` が通常ファイルの JSON オブジェクトとして読める（無ければ `search_record_missing`）
  - exit 20: `code` が `limit_exceeded`（`outcome:"budget_exhausted"`）
  - exit 0: `step:"train"`・`status:"ok"`・`budget_seconds` が指定値・`budget_reached` が真偽値・`total_elapsed_ms` が 0 以上の整数・`candidates` が空でない配列。各要素は `candidate` が 0 からの連番・`kind` が語彙内・`result` が `evaluated`・`training_not_completed`・`scoring_failed`・`scoring_exceeded_budget`・`scoring_skipped_budget_exhausted`・`training_exceeded_time_limit`・`training_timed_out`・`not_started` のどれか・`budget_reached` が null か `search_budget`・`candidate_time_limit`（`TrainSearchResult`・`TrainBudgetScope`。pytest が Rust の enum と機械照合する）。`evaluated` が 1 件以上
  - exit 0: `evaluated` の候補だけ `candidates/<N>/result.json` が残り、それ以外の候補ディレクトリは片付いている（選定対象にしない契約）
- **失敗の理由**: `unexpected_exit_code`・`unexpected_output`（形・語彙の不整合）・`search_record_missing`・`candidate_result_missing`・`candidate_dir_not_cleaned`・`no_candidate_evaluated`、工程別の失敗（`step:"train"`・`case:"train-all"`）
- **記録する `record.json` フィールド**: `status`・`exit_code`・`outcome`（`evaluated` / `budget_exhausted`）・`budget_seconds`・`budget_reached`・`total_elapsed_ms`・`candidates`（`candidate`・`kind`・`result`・`budget_reached`）・`search_record_present`・`steps`
- **人が判断すること**: 予算が妥当か、どの候補がどの結果だったか、exit 20 のとき予算を増やして再実行するか。スクリプトは語彙・整合の確認だけで、探索結果の良し悪しは判定しない

### 4-H. `train --cancel`・`--status` とクラッシュ検出（REQ-34・REQ-39・#484・#485・#486）

新しい project（`H/project`）で `register → inspect` の後に、次の 2 つを順に行います。実 trainer（CPU）を起動し、`job.json` が running になるまで待つため、学習の立ち上がりの分だけ時間がかかります。

1. **キャンセル**: `train --candidate 0` を起動し、`candidates/0/job/job.json` が `running` になり、CLI の下に学習ワーカー（コマンドラインに `launch.py` と `_worker` を持つプロセス。supervisor の `worker_argv`）が現れたら（CLI と supervisor だけの状態では送らない）、別プロセスで `train --project-dir project --cancel` を送る。
2. **クラッシュ検出**: もう一度 `train --candidate 0`（`cancelled` の候補は丸ごと消して新規に始まる。REQ-34）を起動し、同じく running を待って train 本体の CLI プロセスへ `SIGKILL` を送る。

- **確かめること**: キャンセルの応答・train の終了コードと `message`・`--status` の状態・`package/` が生成されないこと・子孫が 0 件になること。続けて、`SIGKILL` した train の記録が `--status` で `failed`・`owner_lost` として検出されること。
- **H の合格条件**:
  - キャンセルの応答: exit 0・`cancellations` が 1 件で `candidate` が 0・`cancel` が `requested`（`already_finished` は学習が先に終わった印で失敗。`cancel_not_requested`）
  - キャンセルされた train: exit 70・`code:"runtime_error"`・`message` が `training cancelled`・`step:"train"`・`candidate` 0・`job.state:"cancelled"`・`restart` が `resumable:false`・`action:"restart_from_scratch"`・`reason_code:"resume_not_supported"`（`train_exit_code_not_70`・`train_output_invalid`）
  - `--status --candidate 0`: exit 0・`jobs[0]` が `state:"cancelled"`・`crash_detected:false`・`failure:null`・`record_updated:false`、`restart` が同じ案内（`status_unexpected`・`status_invalid`）
  - `project/package` が存在しない（`package_created`）
  - 観測した子孫（`ps` の `ppid` を辿って求めた pid と開始時刻の組。操作の後も train の終了まで観測し続ける。ワーカーを 1 件以上観測できなければ失敗）が、train の終了後 30 秒以内にすべて終わる。**ラッパーの後始末（グループ KILL）を外して起動する**ため、残っていなければ CLI 自身（協調キャンセル・`SIGKILL`・supervisor の `killpg`・lifeline。REQ-39）が止めたことになる。残っていれば `descendants_remain_after_cancel`（残ったものは、pid と開始時刻が控えと一致することを送る直前に確かめてから KILL し、件数だけを記録する。pid が再利用された別のプロセスへは送らない。中断・全体の期限の例外の経路でも同じ後始末を行い、止められなければ `child_may_remain` を立てる）。観測が 2 件未満なら `no_descendant_observed`。running にならないまま 120 秒待ったら `job_not_running`
  - クラッシュ: `SIGKILL` した train は JSON を出さない（終了コードは 137 の見込みで、記録に残す）。子孫は 30 秒以内に終わる（`descendants_remain_after_crash`）。その後の `--status --candidate 0` が exit 0・`state:"failed"`・`crash_detected:true`・`failure.kind:"crashed"`・`failure.cause:"owner_lost"`・`record_updated:true`（running の残骸を書き戻した初回）・`restart` が同じ案内
- **失敗の理由**: 上の括弧内の語、`driver_error`・`ps_unavailable`（監視の失敗）、工程別の失敗（`step:"train"`・`case`: `train-cancelled`・`status-cancelled`・`train-killed`・`status-crashed`）
- **記録する `record.json` フィールド**: `cancel_check`（`cancel`・`train_exit_code`・`descendants_observed`・`descendants_remaining`・`package_absent`・`state`・`restart_action`）・`crash_check`（`train_exit_code`・`descendants_observed`・`descendants_remaining`・`state`・`cause`・`restart_action`）・`steps`
- **既知の限界**: Ctrl-C（`SIGINT`）にはハンドラを置かない契約（`owner_lost` として次の `--status` で検出）。H は `SIGKILL` で同じ状態を作る。子孫は H が起動した CLI の下だけを見る（`setsid` で別セッションへ移った孫は、`ppid` の連鎖が切れれば観測できない）

### 4-I. 3 seed の再現性と前のモデルとの比較（REQ-26・REQ-27・#488〜#490・#103）

**既定の項目に入れない**（`--items I` で明示したときだけ実行する）。学習を 4 回行うため、実機で時間がかかる。

新しい project（`I/project`）で `register → inspect` の後にプロジェクトを複製し（`p1`・`p2`・`p3`・`old`）、各複製で `train --train-seed S → select → evaluate` を実行する（`p1`〜`p3` は seed 1・2・3、`old` は旧モデルの代役で seed 4）。最後の `p1` の `evaluate` だけに `--previous-project-dir old --seed-run-project p2 --seed-run-project p3` を付ける（`p2`・`p3`・`old` を先に評価する。凍結 test への適用は複製ごとに 1 回で、REQ-27 の 1 回限りは複製ごとに守られる）。

- **確かめること**: `reproducibility`（`verdict`・`runs`・`disjoint_pairs`）と、旧モデルとの比較（`comparison`）の出力が契約の形で出ること。
- **I の合格条件**:
  - 7 工程の各報告値の整合（B と同じ `_step_check`）。`train` は `--train-seed` を受ける
  - `reproducibility`: `runs` が seed 昇順で `[1,2,3]`、各 run の `total` が評価件数・`correct` が 0 以上 `total` 以下・`ci95` が 0〜1 で `lo <= hi`。`disjoint_pairs` は `runs` の seed の昇順の組で、空 ⇔ `verdict` が `all_pairs_overlap`。**`some_pairs_disjoint` も正常な出力**（再現性の合否の解釈は人が行う）
  - `comparison`: 旧モデルは同じ定義・同じ凍結 test の別 seed のため、`premise` が `same_label_set`・`evaluation_data` が `same`・`removed_labels`・`added_labels` が空・`n_common` が評価件数・`n_previous_only`・`n_current_only` が 0。`counts` は `n` が評価件数・4 区分（`both_correct`・`correct_to_incorrect`・`incorrect_to_correct`・`both_wrong`）の合計が `n`・両方の `ci95` が 0〜1。p 値・有意性は持たない契約のため、回帰の多寡の解釈は人が行う
- **失敗の理由**: `copy_failed`（複製の失敗）・`reproducibility_invalid`・`comparison_invalid`・工程別の失敗（`case`: `train-<複製>`・`select-<複製>`・`evaluate-<複製>`）
- **証拠種別（`--i-device`）**: `cpu`（既定）→ `evidence: "cpu_real_machine"`、`gpu` → `evidence: "gpu_real_machine_declared"`。**CLI の `train` は現状 CPU 固定のため、`gpu` は人の申告で、スクリプトは GPU で学習したことを確認できない**。MLX の GPU 学習は同一 seed でも完全再現しないため、決定的な再現の確認は CPU で行う（#103 の限界「事前登録の条件の GPU ではなく CPU で確認した」を成果物へ書く）。`test_harness`（偽 CLI）の実行は実機の証拠にならない（`evidence_hint` が `test_harness`）
- **記録する `record.json` フィールド**: `evidence`・`device`・`seeds`（`[1,2,3,4]`）・`reproducibility`（`verdict`・`runs`〔`seed`・`correct`・`total`〕・`disjoint_pairs`）・`comparison`（`premise`・`evaluation_data`・`n_common`・`counts`）・`steps`

### 4-J. 前の版への復帰（REQ-39・REQ-27・#491）

B の `project`（`package` 済み。版 v1）を旧プロジェクトとして使う（**B と一緒に指定する**）。カレントは作業ディレクトリ（`B/`・`J/` の親）にする（CLI はカレント配下の相対パスだけを受けるため）。

新しい project（`J/project`）を `package` まで実行し（`package --project-dir J/project --previous-project-dir B/project`）、版 v2 を作る。続けて次を確かめる。

- **J の合格条件**:
  - `package` の `version` が `{"id":"v2","previous":"v1"}`。`J/project/version_ledger.json` が v1・v2 それぞれの `model`・`data`・`experiment` を持つ（`version_invalid`・`version_ledger_invalid`）
  - `infer --package B/project/package --version-ledger J/project/version_ledger.json --version-id v1`（v1 の package を新しい台帳の v1 で使う）が exit 0 で、判定の整合を満たす
  - `infer --package J/project/package --version-ledger <台帳>`（`--version-id` 省略 = 最新の v2）が exit 0
  - `infer --package J/project/package --version-id v1`（`--version-ledger` なし）が exit 64・`code:"invalid_input"`
  - `B/project/package` を複製した `J/tampered` の `artifact.json` を 1 バイト改変し、`--version-id v1` で `infer` すると exit 64・`code:"invalid_input"`（`package does not match version ledger`）。**元の `package/` は変更しない**（改変は複製）
  - 64 の確認で、終了コードが 64 でない・`code` が `invalid_input` でないときは `rejection_not_64`
- `infer` の確認（復帰・最新版）は、校正つきのパッケージでは対象外（exit 11・`out_of_scope`）・保留（exit 12・`abstain`）も正常として、B と同じく終了コードと `status` の対応で照合する（対応しなければ `unexpected_output`）
- **失敗の理由**: `previous_package_missing`（B の `package/` が無い）・`copy_failed`・`version_invalid`・`version_ledger_invalid`・`rejection_not_64`・工程別の失敗（`case`: `package-v2`・`infer-rollback-v1`・`infer-latest`・`infer-version-id-only`・`infer-tampered-artifact`）
- **記録する `record.json` フィールド**: `version_number`（2）・`previous_version_number`（1）・`rollback_to_v1_ok`・`latest_ok`・`version_id_only_exit_code`（64）・`tampered_artifact_exit_code`（64）・`steps`
- **限界**: 台帳ファイル自体の改変は検出できない（読み取り専用化のみ。#491 の範囲外）

## 5. 判定の読み方

スクリプトの終了コードは 4 種です（120・141・1 など 7 種の外の値は出さず、70 へ写します。下の 70 の行の「stdout へ書けなかった場合」）。

| 終了コード | 意味 | 行動 |
| -------- | ---- | ---- |
| 0 | 要求したすべての項目が `ok` | 実機確認成功。`record.json`・`record.md` を PR に記録 |
| 10 | 1 つ以上の要求項目が `failed` または `not_run`（E が B の失敗で `not_run` の場合を含む）。または、実行全体の上限時間（`--overall-timeout-sec`）を超えた（stdout の `message` は `overall time limit exceeded`。実行中の項目は `failed`・残りは `not_run`〔`reason` は `overall_timeout`〕。子のグループは止めて回収する。終了時の再採取と記録の書き出しは上限の外で行うため、全体の所要は上限 ＋ 再採取〔最大 60 秒〕＋ 回収待ち〔10 秒〕で頭打ちになる。#364）。または、開始時と終了時で `commit`・`worktree_clean`・CLI の sha256 のいずれかが一致しない（`environment.stable` が `false`。stdout の `message` は `environment changed during the run`。項目の `status` は書き換えず、全項目が `ok` でも 10 にする。#360） | 失敗した項目の記録を確認し、原因を特定する。原因不明のまま再実行しない |
| 64 | 引数エラー（前処理で検出） | `--items A` 指定時に `--with-ci` が無い、`--items E`・`J` で B が無い、`--g-budget-seconds`・`--i-device` が範囲外、`--work-dir` がリポジトリ内、`FANDHE_EDGE_BIN` が無い・実行できない、など。エラーメッセージ（JSON）から原因を確認して引数を修正。`record.json` は出力されない |
| 70 | 実行環境エラー・中断 | 最終 JSON を stdout へ書けなかった（読み手が先に閉じたパイプ・書き込めないファイル・閉じた stdout）場合は、本来が 0・10・64 でも 70（記録作成後の最終出力に失敗した 0・10 では `record.json`・`record.md` を書き終えているので work-dir で確認する。引数エラー〔64〕は記録作成前に終了するため両ファイルは存在しない）。python3 が無い・3.9 未満、作業ディレクトリを作成できない、fixture が読めない、`record.json` を書き込めない、`FANDHE_EDGE_BIN` 未設定で CLI をビルドできない、実行中に SIGINT・SIGTERM・SIGHUP で中断、項目に想定外の例外が出た（`reason` が `internal_error`。1 件でもあれば stdout の `code` は `runtime_error`）、子が残りうる状態で（全体の上限時間を超えた、または全項目が `ok`）終わった（stdout の `message` は `a child process may remain`。上限超過の 10 には隠さない）、など。環境を確認またはスクリプトを再実行。**中断時・`internal_error` 時は `record.json` が書かれる**（その時点までの項目の結果を記録。中断した項目の `reason` は `interrupted`）。その時点までの `record.md` も出力される。項目の開始前（入力の採取・CLI のビルド・環境の採取の途中）に中断された場合も `record.json`・`record.md` を書く（選んだ項目はすべて `not_run` / `interrupted`、選んでいない項目は `not_run` / `not_selected`、未採取の `environment`・`inputs` は `null`。CLI のビルド中の中断では `inputs` は値あり・`environment` は `null`）。この中断では stdout が `{"code":"runtime_error","message":"interrupted","record":"record.json"}`・終了コードが 70 になる。同じシグナルを 2 回受けた強制終了では `record.json` を書かず、stdout は `{"code":"runtime_error","message":"interrupted (forced exit)"}` になる（「中断の方式」）。中断以外の失敗（fixture が読めない・CLI のビルド失敗）は `record.json` なしで終了コード 70 になる |

**項目の実行フロー**:

1. 要求した項目（`--items LIST`）を A→J の順で実行する
2. 各項目の実行：
   - 要求されていない → `not_run` / `not_selected`
   - 中断済み → `not_run` / `interrupted`（前の項目が `failed` の場合よりも優先する）
   - 全体の上限時間の超過済み → `not_run` / `overall_timeout`（`interrupted` の次、`previous_item_failed` より優先）。実行中に超えた項目は `failed` / `overall_timeout`。終了コードは中断（70）が内部エラー（70）より、内部エラーが上限超過（10）より優先する
   - 前の項目が `failed` → `not_run` / `previous_item_failed`（B の失敗で止まった E もこれ）
   - それ以外 → 実行して結果を記録
3. 最初に `failed` になった項目があれば、以降の要求項目は実行されず `not_run` になる（`not_run` の `reason` は §8）
4. 終了コード 0 = 要求したすべての項目が `ok`、10 = 1 つ以上が `failed` / `not_run`

**中断の方式**: SIGINT・SIGTERM・SIGHUP を受けたハンドラは印を立てるだけで、子プロセスを待つループ（20 ミリ秒ごと）と項目の境目で印を見て、子のプロセスグループを止めて回収してから中断として扱う（止まるまでの遅れは待機の周期程度）。中断のシグナルを受けていれば、項目がすべて完了していても最終結果は中断（終了コード 70・`interrupted`）になる（項目の結果は記録に残る）。既知の限界: `setsid` で別セッションへ移った孫プロセスは止められず残る。子の回収の待機には上限（10 秒。`REAP_WAIT_LIMIT_SECONDS`）があり、KILL が失敗して回収できなければ待つのを諦めて先へ進む。このとき `record.json` の `child_may_remain` を `true` にし（既定は `false`）、その項目は `failed`（`reason` は `unreaped`）にして通常の合否判定へ流さず（fail-closed）、stderr に `a child process may remain` を 1 行出す。`record.md` にも警告行が出る。`ps` で残りを確認し、残っていれば手で止める。もう 1 つの既知の限界として、スクリプト自身が SIGKILL・SIGQUIT で落ちた場合は何も後始末されない（§3-5。証拠種別: 推定）。また、stdout のパイプが詰まっていると最終の JSON の書き込みで止まり、中断シグナルが効かない（§3-5。証拠種別: テストハーネス）。後始末が終わらないときのため、**同じ中断シグナルを 2 回受けると**、子のグループへ KILL を送って即座に終了コード 70 で終える（`record.json`・`record.md` は書かれず、stdout は `{"code":"runtime_error","message":"interrupted (forced exit)"}` の 1 行。証拠種別: テストハーネス）。最終の JSON を出した後に届いたシグナルは 2 行目の JSON を出さず、プロセスの入口（`_entry`）は `main` から抜ける直前に 3 つの中断シグナルを無視（`SIG_IGN`）へ設定するので、終了処理中にシグナルが届いても、シグナルで終わらず終了コードは 70 のままになる。

**項目ごとの成否判定**:

各項目の合格条件は §4 のとおりで、満たす = `ok`、満たさない = `failed` です（終了コード 0 だけでは `ok` にならない）。要点:

- **A**: exit 0 + `skip:` 行 0 件 + Rust と pytest のテストが 1 件以上通り failed 0 件
- **B**: 全 7 工程が exit 0 で報告値が整合 + 容量の 5 要素の合計が `total_bytes` と一致 = `ok`。1 工程でも非 0・報告値の不整合・容量の不一致 = `failed`
- **C**: C-1 と C-2 の両方が成功（C-1 は exit 0 または 20 で p95 と容量・`package/` の有無が整合、C-2 は exit 20 + 容量超過 + `package/` 非公開）= `ok`。1 つでも失敗 = `failed`。C 全体で 1 つの `status` を持つ
- **D**: exit 0 + 成功の印（env -i テスト 3 件・`OK: tool=...`）+ リンクを確認したバイナリが実行した CLI と一致 = `ok`
- **E**: B が要求されていない、または前の項目が失敗 → `not_run`。それ以外で、バッチと単体が整合し予測・スコアが全一致 = `ok`。一致しない = `failed`
- **F**: `failed == 0` = `ok`、`failed > 0` = `failed`
- **G**: exit 0（`evaluated` が 1 件以上で語彙・後始末が整合）または exit 20（`limit_exceeded`）で、どちらも `search_record.json` が残る = `ok`
- **H**: キャンセル（応答・exit 70・`cancelled`・`package/` なし・子孫 0 件）とクラッシュ検出（`failed`・`owner_lost`）がすべて契約どおり = `ok`
- **I**: 4 複製の 7 工程と、再現性・比較の出力が契約の形 = `ok`（`some_pairs_disjoint` も `ok`。解釈は人）
- **J**: v2 の作成・v1 での復帰・最新版・`--version-id` 単独の 64・改変の 64 がすべて契約どおり = `ok`

**失敗時の報告の 5 点**:

1. 終了コード
2. `reason`（失敗の理由。固定語彙の一覧と意味は下の「`reason` の語彙」。項目ごとの語彙は §4。想定外の例外は `internal_error`〔`error_type` に例外の型名。組み込みの閉じた語彙で、語彙外は `<unexpected>`〕）
3. `step`（工程名。固定語彙：`register`・`inspect`・`train`・`select`・`evaluate`・`package`・`infer` など。工程が無い場合は省略）
4. stdout の JSON（あれば）から `code`（7 種の語彙の値）・`message_bytes`・`message_sha256`（`message` の本文は記録されず、`<work-dir>` の stdout のファイルに残る）
5. 実行したコマンド（固定語彙。パスは含めない）

### `reason` の語彙（一覧。#365）

`reason` は文字列の欄で、値はすべてスクリプト内の固定文字列である（閉じた語彙の機械検査はない。語彙外は記録の許可リストが `<unexpected>` に置き換える。§7）。出所ごとに分けて示す。語彙に載っていても、原因を確かめずに再実行しない（fail-closed）。

**(a) 子プロセスの起動・回収**

| reason | 意味 | 出る場所 |
| ---- | ---- | ---- |
| `timeout` | 子ごとの上限時間（§3-4）を超え、スクリプトがプロセスグループを KILL した | 全項目 |
| `output_limit` | stdout・stderr のファイルが上限を超えた（実行中・終了後）。A・D はログを読む時点のサイズ超過でも同じ語 | 全項目 |
| `spawn_error` | 実行ファイルが見つからない・実行できない、終了コードのファイルを事前に消せない、ログを開けない、起動に失敗した | 全項目 |
| `killed` | 子は期限内・上限内で終わったが、**終了コードを確認できなかった**。ラッパーの sh が終了コードを書く前に外から止められた、終了コードのファイルが無い・通常ファイルでない（symlink 等への差し替え）・1〜3 桁の数字でない・255 超。成功扱いにしない（#359） | 全項目（F は件数欄） |
| `unreaped` | KILL 後の回収が 10 秒（`REAP_WAIT_LIMIT_SECONDS`）で終わらない、またはグループへ KILL を送れない。`child_may_remain` が `true` になり、stderr に `a child process may remain` を出す | 全項目（F では `reason` になり、その回で打ち切る） |

`killed` の読み方（誤読されやすい点。証拠種別: テストハーネス）:

- スクリプト自身が上限超過で止めた場合は `timeout`・`output_limit` で、`killed` ではない。
- 子のコマンドがシグナルで終わった場合、ラッパーは `$?`（128 ＋ シグナル番号）を終了コードとして書くので `killed` にならず、終了コード側の失敗になる（A・D は `unexpected_exit_code`。B・C の工程は stdout が JSON でなければ `invalid_json` ＋ `exit_code_unexpected: true`）。
- 終了コードのファイルの欠落・形式不正（数字でない・桁数超過・255 超）・symlink 等への差し替えは `killed` になる。ただし 0〜255 の数字を持つ通常ファイルは書き手の正当性を検証せずそのまま受け入れるため、有効な形式で 0 などに書き換えられた場合は検出できず `ok` になりうる（テストハーネスの範囲。改ざん耐性の保証ではない）。

**(b) 項目ごとの判定**

| 項目 | reason |
| ---- | ---- |
| A | `output_unreadable`・`unexpected_exit_code`・`skipped`・`test_failures`・`no_test_results`（判定順は skipped → test_failures → no_test_results） |
| B | `input_unreadable`・`invalid_json`（補助欄 `exit_code_unexpected`）・`unexpected_exit_code`・`unexpected_output`・`missing_field`・`package_unreadable`・`package_entry_not_regular`・`capacity_sum_mismatch`・`calibration_invalid`・`abstention_invalid`・`diagnostics_invalid`・`significance_invalid`・`version_invalid`・`version_ledger_invalid`・`calibration_binding_mismatch`・`calibration_capacity_mismatch` |
| C | B の工程の語、`staging_left`（C-1・C-2 の両方）、`capacity_limit_not_enforced`（C-2）、C-1 の判定の `missing_field`・`unexpected_output` |
| D | `output_unreadable`・`unexpected_exit_code`・`skipped`・`no_test_results`・`unexpected_output`・`linkage_target_unreadable`・`cli_changed`・`linkage_target_mismatch`・`otool_failed` |
| E | `input_unreadable`・`record_count_out_of_range`・`duplicate_id`・`unexpected_exit_code`・`unexpected_output`・`invalid_json`・`mismatch`。`e-inputs.jsonl` を書けないときは `step` なしの `spawn_error`（子の起動失敗ではない。現状の挙動で、語の見直しは別課題の候補） |
| F | `build_failed`・`list_failed`・`no_tests_listed`・`test_failures`・`unreaped` |
| G | `input_unreadable`・`unexpected_exit_code`・`unexpected_output`・`invalid_json`・`search_record_missing`・`candidate_result_missing`・`candidate_dir_not_cleaned`・`no_candidate_evaluated` |
| H | `input_unreadable`・`driver_error`・`ps_unavailable`・`job_not_running`・`cancel_response_invalid`・`cancel_not_requested`・`train_exit_code_not_70`・`train_output_invalid`・`status_invalid`・`status_unexpected`・`package_created`・`no_descendant_observed`・`descendants_remain_after_cancel`・`descendants_remain_after_crash`・工程別の失敗 |
| I | `input_unreadable`・`copy_failed`・`missing_field`・`reproducibility_invalid`・`comparison_invalid`・工程別の失敗 |
| J | `previous_package_missing`・`input_unreadable`・`copy_failed`・`missing_field`・`version_invalid`・`version_ledger_invalid`・`rejection_not_64`・`unexpected_exit_code`・`unexpected_output`・工程別の失敗 |

**(c) 実行の制御**

- `failed` の項目: `interrupted`・`overall_timeout`・`internal_error`（`error_type` つき）
- `not_run` の項目: `not_selected`・`previous_item_failed`・`interrupted`・`overall_timeout`

**(d) 付随する欄と、畳まれる箇所**

- 付随する欄: `step`・`exit_code`（B・C の工程の起動失敗・回収失敗では `null`）・`exit_code_unexpected`・`code`・`message_bytes`・`message_sha256`（`unexpected_exit_code`・`unexpected_output` の工程失敗）・`case`・`steps`（§4-C-3）・`p95`・`capacity_limit`（C）・`records`（E の `record_count_out_of_range`・`duplicate_id`）・`error_type`
- F の各回の `timeout`・`output_limit`・`killed`・`spawn_error` は `reason` にならず件数欄（`timeouts`・`output_limit`・`killed`・`spawn_error`）に出る。F の `--no-run` の失敗は `build_failed`、`--list` の失敗は `list_failed` に畳まれる
- D の `otool -L` の失敗は `otool_failed` に畳まれる。項目の開始前の CLI のビルド失敗は記録なしの exit 70。環境採取の失敗は該当欄が `null`

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
- **G・H・J**: 実 CLI・実 trainer（CPU）を実機で実行した結果。`FANDHE_EDGE_MAKE_CMD`・`FANDHE_EDGE_CARGO_CMD` の下（偽 CLI）の実行は `test_harness` で、実機の証拠にならない。H の子孫の観測は `ps`（固定パス）の `ppid` の連鎖によるもので、証拠種別は実機（人が確認して記入）
- **I**: `--i-device` で証拠種別を区別して記録する（`cpu_real_machine`・`gpu_real_machine_declared`）。後者は人の申告で、CLI が GPU で学習した確認はスクリプトにはできない（CLI は CPU 固定）。再現性の合否・回帰の多寡の解釈は人が行い、スクリプトは出力の整合までを確かめる
- **F**: テストハーネスを実機で実行したもの。テストの実装（偽の sleeper、実時計）は実機ですが、判定対象（予測値・スコア）は合成に依存しているため、実機の証拠としては「テストハーネス」扱い

**B・C のプロジェクト**: 複数項目で同じ評価データを使う場合は、project ごとに 1 回だけ適用します。B・C-1・C-2 は各々異なる project を使い、各 project へ 1 回ずつ評価データを適用します（REQ-27）。

## 7. `record.json` / `record.md` の保存と PR・Issue への転記

### 保存先と形式

`record.json` は `<work-dir>/` の直下に保存され、実行時のマスク（umask 077）で 0600 の権限を持ちます。各項目の詳細ログは `<work-dir>/<項目>/steps/` などに `*.log`・`*.err` として保存されますが、このログは `record.json` に含まれず、PR・Issue へも転記されません。

`record.json`・`record.md` は同じディレクトリの一時ファイルへ書いてから置き換えるため、中断しても壊れた JSON は残りません。出力先のファイルが symlink などの通常ファイルでない場合は辿らずに失敗し、`runtime_error`（exit 70）になります。ログ・入力ファイルの書き込みも symlink を辿りません。

### 記録してよい項目（転記を許可）

- 終了コード・`code` 値（固定語彙）
- 件数・ハッシュ・サイズ・ライブラリ名（`/usr/lib/`・`/System/` で始まるもの）
- load average・開始・終了の時刻
- 工程の要約（工程ごとに決まった欄だけ。固定語彙・検証済みの数値と真偽値）。`message` の `message_bytes`・`message_sha256`
- 終了時の再採取と比較（`commit_end`・`worktree_clean_end`・`cli_end_sha256`・`commit_unchanged`・`worktree_clean_unchanged`・`cli_unchanged`・`stable`）と、CLI・trainer の出所（`cli_origin`・`trainer_origin`。閉じた語彙。パスは記録しない）
- 機種・メモリ・OS・コミット SHA・実行日（`hw_model`・`cpu`・`os_name`・`os_version`・`os_build` は `^[A-Za-z0-9 ._,()+-]{1,64}$` に一致しなければ `null`。`ncpu`・`memory_bytes` は ASCII の数字 1〜20 桁だけを整数にし、それ以外は `null`）
- `record.json` / `record.md` の内容（伏せ処理済み）

**記録の作り方（許可リスト方式）**: 記録は、記録してよい欄を先に決めて組み立てる。工程の要約（`steps[].summary`）は、工程ごとに決まった欄だけを新しい dict へ組み立て、CLI の JSON の他の欄は捨てる（§4-B）。閉じた語彙の欄で、文字列だが語彙外の値は `<unexpected>`、`package_files[].name` が規則外なら `<unrecognized>`、型が違う欄は `null` になる。最後の関門（`sanitize_record`）で、文字列の値を持ってよいキーの閉じた集合に無い位置の文字列と、200 字を超える文字列を `<redacted>` に置き換える（個別の要約で漏れても止まる）。パス区切りを含む文字列も、`items.D.direct_libraries` の要素（`/usr/lib/`・`/System/` で始まる 128 字以内のもの）を除いて `<redacted>` になる。非有限の浮動小数は `null` になる。`record.md` では、セルへ入る文字列から制御文字・bidi 制御文字・ゼロ幅文字（Unicode カテゴリ Cc・Cf・Cs）を除き、行区切り類とタブは空白 1 個にする（表示の欺瞞と行の増殖を防ぐ）。JSONL・ログの行は LF だけで割る（U+2028 等では割らない）。値が `<unexpected>`・`<unrecognized>`・`<redacted>` の欄は、記録の規則が働いた印であり、原因は `<work-dir>` のログで確かめる

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
  "child_may_remain": true | false (子の回収が上限時間内に終わらず、子・孫が残っている可能性。既定は false),
  "overall_timeout_exceeded": true (実行全体の上限時間を超えたときだけ付く欄。超えなければ欄が無い),
  "environment": null (項目の開始前に中断された場合) | {
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
    "cli_profile": "release" | null,
    "commit_end": "40-char hex or null (終了時。取れなければ null)",
    "worktree_clean_end": "true | false | null (終了時)",
    "cli_end_sha256": "64-char hex or null (終了時)",
    "commit_unchanged": "true | false | null",
    "worktree_clean_unchanged": "true | false | null",
    "cli_unchanged": "true | false | null",
    "stable": "true | false | null",
    "cli_origin": "built_by_script | env_override",
    "trainer_origin": "env | build_default"
  },
  "inputs": null (項目の開始前に中断された場合) | {
    "train_records": "int",
    "evaluation_records": "int",
    "definition_sha256": "64-char hex",
    "train_sha256": "64-char hex",
    "evaluation_sha256": "64-char hex"
  },
  "options": {
    "items": ["A" | "B" | ... | "J"],
    "repeat": "int",
    "quiet_machine": "true | false",
    "p95_limit_us": "int",
    "package_limit_bytes": "int",
    "overall_timeout_sec": "int (実行全体の上限時間。#364)",
    "with_ci": "true | false",
    "g_budget_seconds": "int (G の探索予算。既定 3600)",
    "i_device": "cpu | gpu (I の証拠種別の申告)",
    "cargo_offline": "true | false (A を含まなければ true)"
  },
  "items": {
    "A": { "status": "ok" | "failed" | "not_run", ... },
    "B": { "status": "ok" | "failed" | "not_run", ... },
    ...
    "F": { "status": "ok" | "failed" | "not_run", ... },
    "G": { "status": "ok" | "failed" | "not_run", ... },
    "H": { "status": "ok" | "failed" | "not_run", ... },
    "I": { "status": "ok" | "failed" | "not_run", ... },
    "J": { "status": "ok" | "failed" | "not_run", ... }
  }
}
```

各項目の詳細フィールドは §4-A 〜 §4-J 参照。記録の文字列の欄は許可リスト方式で組み立てられ、語彙外の値は `<unexpected>`・`<unrecognized>`・`<redacted>`、型が違う欄は `null` になる（§7）。`schema` は `real-machine-check/1` のまま。`not_run` 項目の `reason` は：

- `not_selected`: `--items` に指定されなかった
- `previous_item_failed`: 前の項目が `failed` になった
- `interrupted`: 実行中（項目の開始前を含む）に SIGINT・SIGTERM・SIGHUP で中断。前の項目が `failed` で止まった後に中断を受けた場合も、未実行の項目はこの値になる（`previous_item_failed` より優先）
- `overall_timeout`: 実行全体の上限時間（`options.overall_timeout_sec`）を超えた（`failed` の項目の `reason` にも使う）。`record.md` に注意行が出る（#364）

### 終了時の環境の再採取（#360）

`environment` の `commit`・`worktree_clean`・`cli_sha256` は開始時の値のままです（意味は変えません）。実行の終わり（記録を書く直前）に同じ採取をやり直し、`*_end`・`cli_end_sha256` と比較結果（`*_unchanged`）、集約の `stable` を足します。

- 比較は fail-closed: 片方だけ取れなければ「一致しない」（`false`）、両方取れていなければ `null`（比較できない）
- `stable` は 1 つでも `false` なら `false`。`false` のときの最終結果は `judged_fail`（10）。判定の優先は、中断・内部エラー（70）、`stable` が `false`（10）、項目の失敗（10）、ok（0）の順
- 中断の印が立っているときは終了時の採取をせず、`*_end`・`*_unchanged`・`stable` は `null`
- `options.cargo_offline` は「すべての子プロセスが offline で起動したか」を表す。A の `make ci` は offline を強制しないため、A を含む実行では `false`（以前の記録は A 実行時も `true` で、実態を表していなかった）。rustup の自動取得の抑止（`RUSTUP_AUTO_INSTALL=0`）は A を含め常に効き、この欄とは別でスキーマも不変（#375）
- 限界: `worktree_clean` が開始時も終了時も `false` のとき、差分の中身が変わっても検出できない（差分のハッシュは本文を扱う危険があるため取らない）

### `record.md`

自動生成の Markdown テンプレート。冒頭に `bin_override`・`evidence_hint`・harness の有無が表示され、以下は注意行が出ます：

- `bin_override: true` の場合：「注意: CLI を `FANDHE_EDGE_BIN` で差し替えた（このスクリプトがビルドした CLI ではない）」
  あわせて「`commit`・`worktree_clean` は CLI の出所を表さない」旨が出ます
- `environment.stable` が `false` の場合：「注意: 開始時と終了時で commit・worktree_clean・CLI の sha256 のいずれかが一致しない」
- `child_may_remain` が `true` の場合：「注意: 子プロセスの回収が上限時間内に終わらなかった」（子・孫が残っている可能性。結果を採用しない）
- 実行全体の上限時間を超えた場合は 2 通り：項目が `overall_timeout` で打ち切られたときは「上限時間を超えて打ち切った」、全項目の完了後に超えていたときは「全項目の完了後に超えていた（各項目の `status` は変えていない）」（どちらも結果を採用しない。#364）
- `environment.stable` が `null`（開始時か終了時の値を採取できず比較できない）の場合：「注意: …一致を確認できなかった（採取不能。成功扱いにしない）」
- 証拠の種別は「人が確認して記入」と指示されます
- 「項目ごとの結果」の表の「要点」には、各項目の `record.json` の欄から `status`・`steps`・`package_files`・`capacity`・`p95`・`capacity_limit`・`message_sha256` を除いたものを JSON で出します（B は `total_bytes` と `capacity_sum_matches_total`、C は `p95` と `capacity_limit` を足す）。失敗の要点は `reason`・`step`・`exit_code` と、`code`・`message_bytes` です（`message_sha256` は `record.json` だけ）

## 9. 記録簿

1 度の実行（PC・コミット・条件）ごとに 1 行を追加します。

| 日付・機種・OS・コミット | 実行方法 | 全体結果 | A | B | C-1 | C-2 | D | E | F | G | H | I | J | 証拠種別 | 記録先 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 2026-10-04〜05 / Mac16,6（Apple M4 Max）・64 GiB / macOS 27.0（26A428） / 132a7da | 手動コマンド（スクリプト未使用） | 全項目成功 | ok | ok | ok | ok | ok | ok | ok（再発なし） | 未実施 | 未実施 | 未実施 | 未実施 | 実機（項目別注記あり） | #354 |
| 2026-10-05 / Mac16,6（Apple M4 Max）・64 GiB / macOS 27.0（26A428） / 62e52da | スクリプト（`--items A,B,C,D,E,F --with-ci --repeat 50`。`--quiet-machine` なし） | 全項目成功（exit 0） | ok | ok | ok（p95 は参考値） | ok | ok | ok | ok（50 / 50） | 未実施 | 未実施 | 未実施 | 未実施 | 実機（項目別注記あり） | #354 |

G〜J の記録欄（人が実行した結果を記入する。スクリプトの `record.json` の値を転記し、証拠種別は人が確認して書く）:

| 項目 | 記入する内容 |
| ---- | ------------ |
| G | `outcome`（`evaluated` / `budget_exhausted`）・`--g-budget-seconds` の値・`budget_reached`・各候補の `result`（`kind` つき）・`total_elapsed_ms` |
| H | `cancel_check`（応答・train の終了コード・`state`・`descendants_observed` / `descendants_remaining`・`package_absent`）・`crash_check`（train の終了コード・`state`・`cause`・`descendants_remaining`） |
| I | `evidence`（`cpu_real_machine` / `gpu_real_machine_declared`）・`reproducibility.verdict`・`runs`（seed ごとの `correct` / `total`）・`disjoint_pairs`・`comparison.counts`（回帰・改善の件数）。再現性・回帰の解釈（人の判断）を別に書く |
| J | `version_number` / `previous_version_number`・復帰・最新版・`--version-id` 単独・改変の各結果 |

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
