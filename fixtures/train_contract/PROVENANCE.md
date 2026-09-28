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
  `kind`・`kind_version`・`config`・`label_order`・`max_bytes` は
  `request_full.json` と意図的に一致させてある（REQ-39・P1。PR #220
  レビュー対応: `TrainOutcome::from_worker_stdout` がワーカー出力を
  依頼内容と照合するため。`crates/train/tests/train_contract_fixture.rs`
  参照）。`artifact_dir` は `cli.py::run_worker_train` が実際に出す形
  （`str(request.out_dir.display)` = `guard.py::confine` の
  `root_handle.root_real.joinpath(*rel.parts)`。絶対パス）に合わせ、
  `request_full.json` の `root`（`/fandhe-edge-fixture-root`）と `out_dir`
  （`out`）を結合した `/fandhe-edge-fixture-root/out` にした（REQ-39・P0）。
- `result_error.json`: `cli.main(["train", "--request",
  "/tmp/does-not-exist-train-request.json"])` を実行して得た標準出力
  （終了コード 64）をそのまま採用した（生成コマンドは下記）。

## 生成コマンド（証拠種別: テストハーネス。本開発機で実行・確認済み）

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
$ cd trainer && python3 -c "
import sys, hashlib, json
sys.path.insert(0, 'src')
from fandhe_edge_trainer import artifact
sha = hashlib.sha256(b'dummy-onnx-bytes-for-fixture').hexdigest()
art = artifact.build_artifact(
    kind='c3', kind_version=1, config={'epochs': 2},
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
