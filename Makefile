# fandhe-edge-model の開発タスクランナー。
#
# `make setup` 一発で開発環境（サブモジュール・rustup・lefthook・学習ワーカーの
# 仮想環境同期）を構築し、`make ci` でローカル検証（CI の ci.yml・python-ci.yml と
# 同等のローカルゲート）を一括実行する。`make doctor` は環境診断のみを行い、
# 何も導入・変更しない。
#
# Makefile は薄い入口（thin entry point）であり、実処理の定義は cargo・uv・各 lint ツール・
# CI（.github/workflows/ci.yml → Fandhe-AI/actions の reusable workflow・
# .github/workflows/python-ci.yml）が持つ。Make 側で独自の依存グラフ・増分判定は
# 持たない（make スキルの責務分離方針）。
#
# 実装は未着手（Cargo.toml・`crates/` 配下の実クレート未追加）のため、cargo 系ターゲットは
# HAS_CARGO / HAS_MEMBERS 判定でスキップし、workspace 作成後に自動で有効化される
# （冪等セルフヒール。deny も deny.toml + Cargo.toml + メンバー crate が揃った時点で、
# hooks も lefthook.yml が追加された時点で有効化）。学習ワーカー（trainer/。Python・uv）の
# py-* 系ターゲットも同様に HAS_PY（trainer/pyproject.toml の有無）判定でスキップし、
# 追加時点で自動有効化される。スキップ時は必ず `skip:` を表示し、実行したかのように
# 黙って成功扱いにはしない。
# Fandhe-AI/fandhe-container の Makefile と同一方針。GNU Make 3.81 で動作する構文のみを使う。

.DEFAULT_GOAL := help
SHELL := /bin/bash

# Cargo.toml の有無（無ければ cargo 系をスキップ。workspace 作成後に有効化）
HAS_CARGO := $(wildcard Cargo.toml)
HAS_DENY := $(wildcard deny.toml)
# lefthook.yml の有無（無ければ hooks をスキップ。設定無しで `lefthook install` しない）
HAS_LEFTHOOK := $(wildcard lefthook.yml)
# workspace のメンバー crate（`crates/*/Cargo.toml`）の有無。member crate が
# 1 つも無い仮想 workspace（`members = []`）に対しては `cargo fmt --all --check`・
# `cargo clippy --workspace`・`cargo test --workspace`・`cargo deny check ...` の
# いずれも「対象パッケージが無い」エラーで落ちる（cargo の仕様）ため、これらの
# ターゲットは HAS_MEMBERS でスキップする。
# 一方 Cargo.toml 自体の構文・workspace 定義としての妥当性は member の有無に
# 依存せず常に検証可能なため、`check-workspace-manifest`（下記）は HAS_CARGO のみで
# 判定する。
HAS_MEMBERS := $(wildcard crates/*/Cargo.toml)

# 学習ワーカー（trainer/。Python・uv 管理。REQ-18〜20/19b）の有無。
# 推論ランタイムは学習側に依存しない（REQ-32）ため、将来の Rust workspace ルートとは
# 別ディレクトリに切り出している。未追加の間 py-* 系ターゲットはスキップする
PY_DIR := trainer
HAS_PY := $(wildcard trainer/pyproject.toml)
# Windows（GitHub Actions の windows runner では OS 環境変数が Windows_NT）判定。
# 実 trainer 結合テスト（real_trainer.rs は #![cfg(unix)]）は Windows でテストが
# 0 件になるため、test-trainer-integration を Windows では実行対象から外す（#258）
IS_WINDOWS := $(filter Windows_NT,$(OS))

# lint ツールの固定バージョン。CI（Fandhe-AI/actions の lint-docs reusable workflow）の
# 既定値に合わせる（CI 側が正。乖離したらこちらを追従させる）。
# EC_NPM_VERSION は npm ラッパーパッケージの版、EC_BIN_VERSION はラッパーが取得する Go バイナリの
# release タグ（CI は v3.8.0 を直接取得する）。ラッパーは EC_VERSION 未指定だと latest を取得し、
# v4 以降の asset 名変更（ec-* → editorconfig-checker-*）で 6.1.1 がバイナリを見つけられず
# 失敗するうえ、CI と版がずれるため、EC_VERSION で CI と同じタグに固定する。
MARKDOWNLINT_VERSION := 0.49.1
YAMLLINT_VERSION := 1.38.0
EC_NPM_VERSION := 6.1.1
EC_BIN_VERSION := 3.8.0
COMMITLINT_VERSION := 21.2.1
COMMITLINT_CONFIG_VERSION := 21.2.0

# 導入系ツールの固定バージョン（`=x.y.z` 完全固定方針に合わせ exact 固定）。
LEFTHOOK_VERSION := 2.1.10
CARGO_DENY_VERSION := 0.20.2
# trainer/pyproject.toml の [tool.uv] required-version と同一の値を維持する
# （pyproject.toml 側を正とし、乖離したらこちらを追従させる。乖離したまま放置すると
# uv 自身が required-version 違反で fail するため、更新時は 2 箇所を同時に直す）
UV_VERSION := 0.12.19

.PHONY: help
help: ## ターゲット一覧を表示する
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'

# --------------------------------------------------
# 環境診断・環境構築
# --------------------------------------------------

# 診断のみ。何も導入・修復しない（導入は setup の責務）。必須ツールが欠けていれば
# 非 0 で終了し、任意ツール（lint-docs 系・lefthook）は欠けていても警告に留める。
.PHONY: doctor
doctor: ## 開発環境を診断する（導入・変更は一切しない。必須ツール欠落時は非 0 終了）
	@missing=0; \
	for tool in git rustup cargo rustc; do \
		if command -v "$$tool" >/dev/null 2>&1; then \
			printf 'ok    %-16s %s\n' "$$tool" "$$("$$tool" --version 2>/dev/null | head -n 1)"; \
		else \
			printf 'miss  %-16s （必須）\n' "$$tool"; missing=1; \
		fi; \
	done; \
	for comp in rustfmt cargo-clippy; do \
		if command -v "$$comp" >/dev/null 2>&1 && "$$comp" --version >/dev/null 2>&1; then \
			printf 'ok    %-16s %s\n' "$$comp" "$$("$$comp" --version 2>/dev/null | head -n 1)"; \
		else \
			printf 'miss  %-16s （必須。rust-toolchain.toml の components。rustup show で導入）\n' "$$comp"; missing=1; \
		fi; \
	done; \
	for tool in npx lefthook yamllint uvx cargo-deny; do \
		if command -v "$$tool" >/dev/null 2>&1; then \
			printf 'ok    %-16s\n' "$$tool"; \
		else \
			printf 'warn  %-16s （任意。未導入の場合は該当ターゲットが導入案内または自動導入を行う）\n' "$$tool"; \
		fi; \
	done; \
	if command -v uv >/dev/null 2>&1; then \
		uv_ver=$$(uv --version 2>/dev/null | awk '{print $$2}'); \
		if [ "$$uv_ver" = "$(UV_VERSION)" ]; then \
			printf 'ok    %-16s %s\n' uv "$$uv_ver"; \
		else \
			printf 'warn  %-16s %s （固定版 $(UV_VERSION) と不一致。trainer/pyproject.toml の required-version が拒否します）\n' uv "$$uv_ver"; \
		fi; \
	else \
		printf 'warn  %-16s （任意。py-* ターゲットに必要。brew install uv や https://docs.astral.sh/uv/ を参照）\n' uv; \
	fi; \
	if [ -f docs/spec/.git ] || [ -d docs/spec/.git ]; then \
		echo 'ok    docs/spec        submodule 取得済み'; \
	else \
		echo 'warn  docs/spec        submodule 未取得（private。make submodule で取得を試行）'; \
	fi; \
	if [ "$$missing" -ne 0 ]; then \
		echo "NG: 必須ツールが不足しています（make setup または公式手順で導入してください）" >&2; \
		exit 1; \
	fi

# 依存ターゲット並記だと -j 実行時に順序が保証されず、cargo フォールバックを持つ hooks が
# rustup より先に走りうるため、再帰 make で「submodule → rustup → hooks → py-sync」の
# 順を明示する（py-sync は uv 未導入・trainer/pyproject.toml 未追加ならそれぞれ
# 自身で fail-closed / skip する。hooks より後ろに置く積極的な理由は無いが、
# 既存 3 ステップの後ろに素直に追加する）。
.PHONY: setup
setup: ## 開発環境を一括構築する（サブモジュール → rustup → lefthook → 学習ワーカー同期の順を保証）
	$(MAKE) submodule
	$(MAKE) rustup
	$(MAKE) hooks
	$(MAKE) py-sync
	@echo "setup 完了"

# rustup は前提条件として確認のみ行い、自動導入はしない。取得したインストーラを検証なしに
# 実行する経路（curl | sh）を作らないため（サプライチェーン対策）。未導入時は
# 公式の導入手順を案内して停止する。toolchain は rust-toolchain.toml が単一真実源。
.PHONY: rustup
rustup: ## rustup（cargo）の導入を確認する（未導入なら公式手順を案内して停止）
	@if ! command -v rustup >/dev/null 2>&1 && [ ! -x "$$HOME/.cargo/bin/rustup" ]; then \
		echo "error: rustup が見つかりません。公式手順（https://rustup.rs/）で導入してから再実行してください" >&2; \
		exit 1; \
	fi

# docs/spec（fandhe-edge-model-spec）は private リポジトリのため、アクセス権のない環境では
# 取得に失敗する。実装コードのビルド・テストは docs/spec 抜きでも成立させる方針
# （README.md）のため、失敗しても setup 全体は止めない。
.PHONY: submodule
submodule: ## docs/spec サブモジュールを初期化・更新する（private・アクセス権が無ければ警告のみ）
	@git submodule update --init || \
		echo "警告: docs/spec（private）の取得に失敗しました。アクセス権のない環境では想定内です（ビルド・テストは spec 抜きで成立します）"

# lefthook（Go 製。crates.io には存在しないため cargo フォールバックは置かない）は
# brew（バージョン固定不可だが常用導線）を優先し、無ければ npm 配布版を exact 固定の
# npx ワンショットで実行する（lefthook が生成する hook スクリプトは PATH → npx の順で
# 本体を解決するため、npx 経由の導入でもコミット時にフックが機能する）。
# lefthook.yml が未追加の間は、設定の無い hooks を導入しないようスキップする。
.PHONY: hooks
hooks: ## lefthook の git hooks を導入する（未導入なら lefthook 本体も導入。lefthook.yml 未追加ならスキップ）
ifneq ($(HAS_LEFTHOOK),)
	@if command -v lefthook >/dev/null 2>&1; then \
		lefthook install; \
	elif command -v brew >/dev/null 2>&1; then \
		echo "lefthook を導入します"; \
		brew install lefthook && lefthook install; \
	elif command -v npx >/dev/null 2>&1; then \
		echo "lefthook（npx 固定版）で hooks を導入します"; \
		npx --yes lefthook@$(LEFTHOOK_VERSION) install; \
	else \
		echo "brew / npx が見つかりません。https://lefthook.dev/installation/ を参照してください" >&2; \
		exit 1; \
	fi
else
	@echo "skip: lefthook.yml 未追加のため hooks をスキップ"
endif

# --------------------------------------------------
# ドキュメント／設定ファイル系 lint（CI の lint-docs ジョブと同等の内容）
# --------------------------------------------------

.PHONY: lint-md
lint-md: ## markdownlint（.markdownlint.jsonc / .markdownlintignore 参照）
	npx --yes markdownlint-cli@$(MARKDOWNLINT_VERSION) --ignore-path .markdownlintignore "**/*.md"

# yamllint は Python 製のため npx で賄えない。導入済みの実体（brew / pip）を優先し、
# uvx があれば固定版のワンショット実行で代替する。いずれも無ければ fail-closed で
# 導入方法を案内して失敗する（silent skip は CI との false-green 乖離になるため行わない）。
.PHONY: lint-yaml
lint-yaml: ## yamllint（.yamllint 参照）
	@if command -v yamllint >/dev/null 2>&1; then \
		yamllint .; \
	elif command -v uvx >/dev/null 2>&1; then \
		uvx yamllint==$(YAMLLINT_VERSION) .; \
	else \
		echo "yamllint 未導入: brew install yamllint / pip install yamllint==$(YAMLLINT_VERSION) で導入してください" >&2; \
		exit 1; \
	fi

.PHONY: lint-editorconfig
lint-editorconfig: ## editorconfig-checker（.editorconfig + .editorconfig-checker.json 参照）
	EC_VERSION=v$(EC_BIN_VERSION) npx --yes editorconfig-checker@$(EC_NPM_VERSION)

# main からの分岐点以降のコミットを CI（lint-docs の commitlint ジョブ）と同じ
# extends 構成で検証する。origin/main が未取得の環境では範囲を決められないためスキップする。
# `git rev-parse --verify --quiet` は「参照が存在しない」場合に終了コード 1 を返すため、
# これだけを skip 条件にし、それ以外の非 0（`fatal: detected dubious ownership` 等。
# 典型的には終了コード 128）はエラーメッセージを表示して非 0 終了する（fail-closed）。
.PHONY: lint-commits
lint-commits: ## commitlint（origin/main からの分岐点以降のコミットを検証）
	@out=$$(git rev-parse --verify --quiet refs/remotes/origin/main 2>&1 >/dev/null); st=$$?; \
	if [ "$$st" -eq 1 ]; then \
		echo "skip: origin/main が未取得のため commitlint をスキップ"; \
		exit 0; \
	elif [ "$$st" -ne 0 ]; then \
		printf '%s\n' "$$out" >&2; \
		echo "NG: origin/main の参照確認に失敗しました（git rev-parse exit=$$st ）" >&2; \
		exit 1; \
	fi; \
	base=$$(git merge-base origin/main HEAD) || { \
		echo "NG: git merge-base の実行に失敗しました" >&2; \
		exit 1; \
	}; \
	npx --yes -p @commitlint/cli@$(COMMITLINT_VERSION) -p @commitlint/config-conventional@$(COMMITLINT_CONFIG_VERSION) \
		commitlint --extends @commitlint/config-conventional --from "$$base" --to HEAD

.PHONY: lint-docs
lint-docs: lint-md lint-yaml lint-editorconfig lint-commits ## ドキュメント／設定ファイル系 lint を一括実行する

# --------------------------------------------------
# 品質チェック（Rust。Cargo.toml 追加後に有効化）
# --------------------------------------------------

# workspace 仮想 manifest（Cargo.toml）自体の構文・定義としての妥当性を検証する。
# `cargo verify-project` は member crate が 0 件の仮想 workspace でも成功するため、
# HAS_MEMBERS を条件にせず HAS_CARGO のみで常時実行する。
.PHONY: check-workspace-manifest
check-workspace-manifest: ## cargo verify-project で workspace manifest の妥当性を検証する
ifneq ($(HAS_CARGO),)
	@out=$$(cargo verify-project 2>&1) || { \
		echo "$$out" >&2; \
		echo "NG: Cargo.toml が cargo にとって不正な manifest です" >&2; \
		exit 1; \
	}; \
	if ! printf '%s\n' "$$out" | grep -q '"success"'; then \
		echo "$$out" >&2; \
		echo "NG: cargo verify-project が success を返しませんでした" >&2; \
		exit 1; \
	fi
else
	@echo "skip: Cargo.toml 未追加のため check-workspace-manifest をスキップ"
endif

# ソースを書き換える唯一のターゲット（fmt-check / lint / test / ci は書き換えない）
.PHONY: fmt
fmt: ## cargo fmt --all で整形する（ソースを書き換える）
ifneq ($(and $(HAS_CARGO),$(HAS_MEMBERS)),)
	cargo fmt --all
else
	@echo "skip: Cargo.toml 未追加、または workspace にメンバー crate が無いため fmt をスキップ"
endif

.PHONY: fmt-check
fmt-check: ## cargo fmt --check（整形差分の検出。書き換えない）
ifneq ($(and $(HAS_CARGO),$(HAS_MEMBERS)),)
	cargo fmt --all --check
else
	@echo "skip: Cargo.toml 未追加、または workspace にメンバー crate が無いため fmt-check をスキップ"
endif

# 既定 feature のみで検証する（`--all-features` 込みの検証は CI の rust-ci ジョブが
# 担う。CI の rust-ci-default-features ジョブと同一コマンド）。
.PHONY: lint
lint: ## cargo clippy -D warnings（既定 feature。lint ゲート）
ifneq ($(and $(HAS_CARGO),$(HAS_MEMBERS)),)
	cargo clippy --workspace --all-targets -- -D warnings
else
	@echo "skip: Cargo.toml 未追加、または workspace にメンバー crate が無いため lint をスキップ"
endif

.PHONY: test
test: ## cargo test（既定 feature。workspace 全体）
ifneq ($(and $(HAS_CARGO),$(HAS_MEMBERS)),)
	cargo test --workspace
else
	@echo "skip: Cargo.toml 未追加、または workspace にメンバー crate が無いため test をスキップ"
endif

# Mac 実機で人間が実行する実機前提の確認（`ci` には含めない。REQ-32・TASK-32.3・#115）
.PHONY: check-runtime-linkage
check-runtime-linkage: ## 推論ランタイムの動的リンク確認（Mac 実機前提・人間が実行。REQ-32・#115）
ifneq ($(and $(HAS_CARGO),$(HAS_MEMBERS)),)
	sh scripts/check-runtime-linkage.sh
else
	@echo "skip: Cargo.toml 未追加、または workspace にメンバー crate が無いため check-runtime-linkage をスキップ"
endif

.PHONY: deny
deny: ## cargo deny check advisories bans licenses sources（依存監査。cargo-deny 未導入なら自動導入）
ifneq ($(and $(HAS_CARGO),$(HAS_DENY),$(HAS_MEMBERS)),)
	@export PATH="$$HOME/.cargo/bin:$$PATH"; \
	command -v cargo-deny >/dev/null 2>&1 || { \
		echo "cargo-deny を導入します"; \
		cargo install cargo-deny@$(CARGO_DENY_VERSION) --locked; \
	}; \
	cargo deny --locked check advisories bans licenses sources
else
	@echo "skip: Cargo.toml・deny.toml のいずれか未追加、または workspace にメンバー crate が無いため deny をスキップ"
endif

# --------------------------------------------------
# 品質チェック（Python 学習ワーカー。trainer/pyproject.toml 追加後に有効化）
# --------------------------------------------------
# uv 0.12.19 を経由してのみ実行する（uv 未導入時は curl|sh 等の自動導入をしない。
# rustup ターゲットと同一のサプライチェーン方針）。`--locked` を必ず付け、
# lock ファイルと pyproject.toml が食い違う場合は fail-closed で止める
# （uv sync が lock を無言で書き換えることを防ぐ）。

# uv 自体の有無を確認するヘルパ（無ければ導入方法を案内して停止する）
define require_uv
	if ! command -v uv >/dev/null 2>&1; then \
		echo "error: uv が見つかりません。brew install uv（バージョン $(UV_VERSION) 系）または https://docs.astral.sh/uv/getting-started/installation/ の公式手順で導入してから再実行してください" >&2; \
		exit 1; \
	fi
endef

.PHONY: py-sync
py-sync: ## uv sync --locked で学習ワーカーの仮想環境を lock どおりに同期する
ifneq ($(HAS_PY),)
	@$(require_uv)
	uv sync --locked --directory $(PY_DIR)
else
	@echo "skip: trainer/pyproject.toml 未追加のため py-sync をスキップ"
endif

.PHONY: py-fmt
py-fmt: ## ruff format で学習ワーカー・scripts/ の Python を整形する（書き換える）
ifneq ($(HAS_PY),)
	@$(require_uv)
	uv run --locked --directory $(PY_DIR) ruff format . ../scripts
else
	@echo "skip: trainer/pyproject.toml 未追加のため py-fmt をスキップ"
endif

.PHONY: py-fmt-check
py-fmt-check: ## ruff format --check（整形差分の検出。書き換えない）
ifneq ($(HAS_PY),)
	@$(require_uv)
	uv run --locked --directory $(PY_DIR) ruff format --check . ../scripts
else
	@echo "skip: trainer/pyproject.toml 未追加のため py-fmt-check をスキップ"
endif

.PHONY: py-lint
py-lint: ## ruff check（学習ワーカーの lint ゲート。S ルールで危険な逆シリアル化等を検出）
ifneq ($(HAS_PY),)
	@$(require_uv)
	uv run --locked --directory $(PY_DIR) ruff check . ../scripts
else
	@echo "skip: trainer/pyproject.toml 未追加のため py-lint をスキップ"
endif

.PHONY: py-test
py-test: ## pytest（学習ワーカーのテスト）
ifneq ($(HAS_PY),)
	@$(require_uv)
	uv run --locked --directory $(PY_DIR) pytest
else
	@echo "skip: trainer/pyproject.toml 未追加のため py-test をスキップ"
endif

.PHONY: py-ci
py-ci: py-fmt-check py-lint py-test ## 学習ワーカーのローカルゲートを一括実行する

.PHONY: test-trainer-integration
# 実 trainer（MLX CPU）を Rust の run_train から起動する結合テスト（issue #258。
# REQ-18/19/34/39）。crates/train/tests/real_trainer.rs は #[ignore] で既定の
# `make test` から分離してあり、ここで --ignored 付きで実行する（python-ci.yml と
# `make ci` の両方で実行するため、CI を通すための skip ではない）。
# libtest のテスト名フィルタは 1 回の起動につき 1 つしか渡せないため、テストごとに
# 個別に起動する（「パッケージ:テストバイナリ:テスト名」の組で回す。sandbox_pipeline_real_trainer は
# sandbox 監視チェーンの完走確認。REQ-38・TASK-38.1・#161）。cargo test 自体の終了状態を保持するためパイプ（tee）は使わず、
# 一時ファイルへ出力してから表示する。--exact のテスト名がずれると 0 件実行で
# 成功してしまうため、出力の「1 passed」も検査して fail-closed にする
# （テスト名を変えたらここも更新する）。
# real_trainer.rs は unix 限定（`#![cfg(unix)]`）のため、Windows では skip を表示して
# 成功扱いにする（Windows は MLX の実機検証対象外。`.claude/rules/coding-rust.md`）。
ifneq ($(IS_WINDOWS),)
test-trainer-integration: ## 実 trainer を Rust から起動する結合テスト（Windows では対象外。issue #258）
	@echo "skip: real_trainer は unix 限定のため Windows では test-trainer-integration をスキップ"
else
test-trainer-integration: py-sync ## 実 trainer を Rust から起動する結合テスト（#[ignore] 分離分。issue #258）
ifneq ($(and $(HAS_CARGO),$(HAS_MEMBERS),$(HAS_PY)),)
	@$(require_uv)
	@out="$$(mktemp)"; overall=0; \
	for spec in \
		fandhe-edge-train:real_trainer:req18_real_trainer_c1_job_completes_with_typed_outcome \
		fandhe-edge-train:real_trainer:req18_real_trainer_c3_job_completes_with_validation_predictions \
		fandhe-edge-cli:sandbox_pipeline_real_trainer:req38_real_pipeline_completes_under_monitor_with_zero_tool_denials; do \
		pkg="$${spec%%:*}"; rest="$${spec#*:}"; bin="$${rest%%:*}"; t="$${rest#*:}"; \
		cargo test -p "$$pkg" --test "$$bin" -- --ignored --exact "$$t" >"$$out" 2>&1; \
		status=$$?; \
		cat "$$out"; \
		if [ "$$status" -ne 0 ]; then \
			echo "error: $$bin の $$t が失敗しました（終了コード $$status）" >&2; overall=1; \
		elif ! grep -q "test result: ok. 1 passed" "$$out"; then \
			echo "error: $$bin の $$t が実行・成功していません（テスト名の不一致の可能性）" >&2; overall=1; \
		fi; \
	done; \
	rm -f "$$out"; exit $$overall
else
	@echo "skip: Cargo.toml / メンバー crate / trainer/pyproject.toml のいずれかが無いため test-trainer-integration をスキップ"
endif
endif

.PHONY: check-dependency-approvals
# 依存の承認台帳（dependency-approvals.json）と manifest・lock を照合し、承認記録のない
# 依存の追加・更新・削除を止める（REQ-38・TASK-38.3・#165。手順: docs/design/dependency-approval-flow.md）。
# tomllib が要るため trainer の Python（3.12。.python-version）で実行する。`uv run --locked` は環境未構築時に
# 依存の解決・同期（通信）を起こしうるため使わず、`--no-project --offline` でプロジェクト同期と通信を
# 抑止する（Python 本体が未導入ならダウンロードせず失敗する＝fail-closed）。標準ライブラリのみ。
check-dependency-approvals: ## 依存の承認台帳と manifest・lock を照合する（未承認の依存変更で失敗。#165）
ifneq ($(HAS_PY),)
	@$(require_uv)
	uv run --no-project --offline --directory $(PY_DIR) python -I ../scripts/check_dependency_approvals.py --root ..
else
	@echo "skip: trainer/pyproject.toml 未追加のため check-dependency-approvals をスキップ"
endif

.PHONY: ci
ci: lint-docs check-workspace-manifest check-dependency-approvals fmt-check lint test deny py-ci test-trainer-integration ## ローカルゲート（CI の ci.yml・python-ci.yml と同等のチェック）を一括実行する

# --------------------------------------------------
# 後片付け
# --------------------------------------------------

# 素の `cargo clean` は CARGO_TARGET_DIR / build.target-dir が他 worktree と共有の
# ディレクトリを指している場合にそれを丸ごと消すため、`--target-dir target` で本リポの
# ./target に限定する。target/ が symlink（共有ディレクトリへの参照の可能性）なら
# 辿らずに拒否する（make スキル rust-crate サンプルと同一方針）。
.PHONY: clean
clean: ## 本リポの ./target のみ削除する（共有 CARGO_TARGET_DIR には触れない）
ifneq ($(HAS_CARGO),)
	@if [ -L target ]; then echo "error: target/ が symlink です（共有ビルドディレクトリの可能性）。削除を拒否します" >&2; exit 1; fi
	cargo clean --target-dir target
else
	@echo "skip: Cargo.toml 未追加のため clean をスキップ"
endif
