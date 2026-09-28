# train_contract フィクスチャの出典

issue #177（学習リクエスト・結果 JSON の Rust 型と共有 fixture による一致確認。
REQ-18・REQ-19・REQ-34・REQ-39）向けに手作成した fixture 一式。実データ・
個人情報・秘密情報は含まない（すべてダミーの ASCII 値）。証拠種別: テスト
ハーネス（Rust の `cargo test`・Python の `pytest` の両方が同じ fixture を
読み、同じ解釈になることを機械照合する）。

## 単一真実源

`coding-python.md`「入出力の JSON スキーマは Rust 側の定義を正とし、Python
側で独自のフィールドを増やさない」に基づき、スキーマの正は
`crates/train/src/`（`fandhe-edge-train`）とする。ただし各定数・検証規則の
**値そのもの**は既存の Python 実装（`trainer/src/fandhe_edge_trainer/
contract.py`・`limits.py`・`artifact.py`・`guard.py`・`supervisor.py`）を
書き起こしたもので、両実装が独立に同じ値を持つことを本 fixture で照合する。

## ファイル一覧

- `limits.json`: `crates/train/src/limits.rs` の各定数・
  `trainer/src/fandhe_edge_trainer/limits.py` の対応定数・
  `contract.py::SCHEMA_VERSION`・`contract.py::_ALLOWED_DEVICES`・
  `supervisor.py::_MAX_WORKER_STDOUT_BYTES` の値を並べたもの。
- `kind_defaults.json`: `kind` ごとの config 既定値（REQ-18・REQ-19・
  REQ-19b・REQ-21・REQ-39。codex review PR #220 P1「成果物の追加 config 値を
  検証せず成功扱いにしている」対応）。選択口（`trainer/src/
  fandhe_edge_trainer/kinds/__init__.py::_registry`）に登録済みで
  `DEFAULT_CONFIG` を持つ種類（`c1`・`c3`）をすべて含む。値は各
  `kinds/<kind>.py::DEFAULT_CONFIG` をそのまま `json.dump` して生成した
  （生成コマンドは下記）。`crates/train/src/kind_defaults.rs` が
  `include_str!` でビルド時に埋め込み、
  `TrainOutcome::from_worker_stdout`（`crates/train/src/result.rs`）が
  「`kind_defaults.json` の既定値に `request.config` を上書きした実効
  config」と成果物の `artifact.config` の完全一致を検査するために使う
  （未知のキーの追加・既定値の書き換え・明示値の書き換えをすべて拒否する）。
  `trainer/tests/test_kind_defaults_fixture.py` が Python 側の
  `DEFAULT_CONFIG` との一致を機械照合し、`kind` を選択口へ追加して本 fixture
  の更新を忘れると同テストが落ちる。生成コマンドは下記「生成コマンド」節の
  `kind_defaults.json` の項を参照。
- `definition.json`: `fandhe-edge-model-definition/v1` 形式のダミー定義
  （選択肢 3 件: `positive`・`negative`・`neutral`）。
  `label_order_from_definition`（Rust）と `options[].id` の宣言順（pytest）が
  一致することの確認に使う。
- `request_full.json`: 13 項目すべてを含む正常な学習リクエスト。`root` は
  プレースホルダー `/fandhe-edge-fixture-root`（絶対パス文字列）で、
  `trainer/tests/test_train_contract_fixture.py` はこの値と完全一致する
  場合に限り `str(tmp_path)` へ置き換えてから `validate_request` を呼ぶ
  （`guard.resolve_root` が実際にディレクトリを開くため）。`crates/train`
  側は文字列としての構文検査のみ行うため置き換えない。
- `request_minimal.json`: 任意項目（`config`・`time_limit_seconds`・
  `rss_limit_bytes`）を省略した正常なリクエスト。
- `request_reject_cases.json`: `base`（`request_full.json` と同じ内容）に
  1 フィールドずつ `patch` または `raw_text`（JSON 外の数値トークンを含む
  生テキスト）を適用した異常系一覧。各ケースの `expected_code`／
  `expected_exit` は Python（`contract.py`／`guard.py`）と Rust
  （`fandhe-edge-train`）の両方が独立に実装した検証ロジックが一致した値。
  一致しないケース（Rust の方が厳格な `kind_version` の負値・JSON の重複
  キー等）はここに含めず、`crates/train/src/request.rs` の単体テストへ
  分離した（issue #177 実装計画「Rust の方が厳しいケース」）。
  `time_limit_seconds_null`・`rss_limit_bytes_null`（PR #220 Bugbot 指摘対応）:
  キーが存在し値が JSON `null` の場合、両実装とも「省略時の既定値」へは
  解決せず `invalid_request` で拒否する（`contract.py::validate_request` の
  `raw.get(field, DEFAULT)` は値が `null` なら `DEFAULT` を使わず
  `isinstance(None, int)` で弾く。Rust 側は `crates/train/src/request.rs`
  の `deserialize_present` で同じ区別を行う）。
- `result_ok.json`: 成功時の結果 JSON。`onnx_sha256` はダミーバイト列
  `b"dummy-onnx-bytes-for-fixture"` の SHA-256（形式が妥当な値であること
  だけを確認するためのダミーで、実際の `model.onnx` とは対応しない）。
  `artifact` の各フィールドは `trainer/src/fandhe_edge_trainer/artifact.py::
  build_artifact` を実行して得た出力をそのまま採用し、`created_utc` のみ
  固定値 `2026-09-28T00:00:00Z` に置き換えた（生成コマンドは下記）。
  `kind`・`kind_version`・`label_order`・`max_bytes`・`candidate_label` は
  `request_full.json` と意図的に一致させてある（REQ-39・P1。PR #220
  レビュー対応: `TrainOutcome::from_worker_stdout` がワーカー出力を
  依頼内容と照合するため。`crates/train/tests/train_contract_fixture.rs`
  参照）。`artifact_dir` は `cli.py::run_worker_train` が実際に出す形
  （`str(request.out_dir.display)` = `guard.py::confine` の
  `root_handle.root_real.joinpath(*rel.parts)`。絶対パス）に合わせ、
  `request_full.json` の `root`（`/fandhe-edge-fixture-root`）と `out_dir`
  （`out`）を結合した `/fandhe-edge-fixture-root/out` にした（REQ-39・P0）。
  `config` は `request_full.json` の `config`（`{"epochs": 2}`）と完全一致
  させず、`kinds/c3.py::train` が実際に返す実効 config（`{**DEFAULT_CONFIG,
  **request.config}`。`request_full.json` の `epochs: 2` で上書きした
  `c3.DEFAULT_CONFIG`）を反映した（REQ-19・REQ-21・REQ-39・P1。codex 指摘
  PR #220「既定値で補完された config を正常な学習結果として受理する」）。
  `TrainOutcome::from_worker_stdout` は `config` を「`kind_defaults.json`
  （既定値の正本である学習ワーカー側 `kinds/c1.py`・`kinds/c3.py` の
  `DEFAULT_CONFIG` をそのまま書き出した共有 fixture）に `request.config` を
  上書きした実効 config」との完全一致で検査する
  （[`kind_defaults.json`](#ファイル一覧) 参照。codex review PR #220 P1
  「成果物の追加 config 値を検証せず成功扱いにしている」対応。旧・部分一致
  検査〔`request` の明示キーのみ照合〕は撤去した）ため、この fixture は
  「`request.config` の明示キーである `epochs` の値は保たれているが、他の
  キーは既定値のまま増えている」という実際のワーカー挙動を表す。既定値
  そのものの正本は引き続き学習ワーカー側であり、本 crate 側では値を
  再定義しない（層の境界。`.claude/rules/dependency-policy.md`）。
- `result_error.json`: `cli.main(["train", "--request",
  "/tmp/does-not-exist-train-request.json"])` を実行して得た標準出力
  （終了コード 64）をそのまま採用した（生成コマンドは下記）。

## 生成コマンド（証拠種別: テストハーネス。本開発機で実行・確認済み）

`kind_defaults.json`（`c1`・`c3` の `DEFAULT_CONFIG` をそのまま書き出す）:

```console
$ cd trainer && uv run --locked python3 -c "
import sys, json
sys.path.insert(0, 'src')
from fandhe_edge_trainer.kinds import c1, c3
out = {
    '_meta': {
        'description': 'Per-kind default config values (REQ-18/19/19b/21/39). ...'
    },
    'c1': c1.DEFAULT_CONFIG,
    'c3': c3.DEFAULT_CONFIG,
}
with open('../fixtures/train_contract/kind_defaults.json', 'w') as f:
    json.dump(out, f, indent=2)
    f.write('\n')
"
```

```console
$ cd trainer && python3 -c "
import sys
sys.path.insert(0, 'src')
from fandhe_edge_trainer import cli
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rc = cli.main(['train', '--request', '/tmp/does-not-exist-train-request.json'])
print('RC=', rc)
print('OUT=', repr(buf.getvalue()))
"
RC= 64
OUT= '{"status": "error", "code": "invalid_request", "message": "file not readable: FileNotFoundError"}\n'
```

```console
$ cd trainer && uv run python3 -c "
import sys, hashlib, json
sys.path.insert(0, 'src')
from fandhe_edge_trainer import artifact
from fandhe_edge_trainer.kinds import c3
sha = hashlib.sha256(b'dummy-onnx-bytes-for-fixture').hexdigest()
# cli.py::run_worker_train が実際に artifact へ渡す config は
# kinds/c3.py::train が返す trained.config（= {**DEFAULT_CONFIG,
# **request.config}）であり、request.config そのものではない。
cfg = {**c3.DEFAULT_CONFIG, **{'epochs': 2}}
art = artifact.build_artifact(
    kind='c3', kind_version=1, config=cfg,
    label_order=['positive', 'negative', 'neutral'], output_type='choice',
    max_bytes=512, candidate_label='c3', onnx_sha256=sha,
)
art['created_utc'] = '2026-09-28T00:00:00Z'
# artifact_dir は cli.py::run_worker_train が実際に出す形（root + out_dir の
# 絶対パス）に合わせた手作業の値（request_full.json の root/out_dir と対応）。
print(json.dumps(
    {'status': 'ok', 'artifact_dir': '/fandhe-edge-fixture-root/out', 'artifact': art},
    indent=2,
))
"
```

## `label_order_above_max` ケースの 1025 件の `label_order`

`["l0", "l1", ..., "l1024"]`（`f"l{i}" for i in range(1025)`）として生成した。
生成スクリプトは残していない（値は fixture へ直接書き込み済み）。
