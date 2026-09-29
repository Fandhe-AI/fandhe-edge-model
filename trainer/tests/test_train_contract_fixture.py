"""学習リクエスト・結果 JSON の Rust ⇔ Python 一致テスト（REQ-18・REQ-19・
REQ-34・REQ-39・issue #177）。

単一真実源は Rust 側のスキーマ定義（`crates/train/src/`。`fandhe-edge-train`）
だが、値・検証規則は本ワーカーの既存実装（`contract.py`・`limits.py`・
`artifact.py`）から書き起こしたものである。両実装が独立に同じ解釈へ到達する
ことを、共有 fixture（`fixtures/train_contract/`）を介して
`crates/train/tests/train_contract_fixture.rs`（Rust）と本ファイル（pytest）の
双方から照合する。`fixtures/train_contract/` というパスが唯一の結合点であり、
このパスを変えると両テストの結合が切れる。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fandhe_edge_trainer import artifact, cli, contract, limits
from fandhe_edge_trainer.errors import WorkerError
from fandhe_edge_trainer.exitcode import ExitCode
from fandhe_edge_trainer.kinds import c3

# 読み込み前にファイルサイズの上限を確認する（外部入力読み込みの作法。
# coding-python.md）。ローカル固定 fixture だが、想定外に巨大化した場合に
# 無制限アロケーションへ繋げない。
_MAX_FIXTURE_BYTES = 1 * 1024 * 1024

# 本ファイル（trainer/tests/test_train_contract_fixture.py）からリポジトリ
# 直下までの階層: parents[0]=trainer/tests, parents[1]=trainer,
# parents[2]=リポジトリ直下。
_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE_DIR = _REPO_ROOT / "fixtures" / "train_contract"

#: `request_full.json`／`request_reject_cases.json` の `base.root` に入っている
#: プレースホルダー。この値と完全一致する場合に限り `tmp_path` へ置き換える
#: （issue #177 実装計画 0-2）。
_ROOT_PLACEHOLDER = "/fandhe-edge-fixture-root"


def _load_json_fixture(name: str) -> Any:
    path = _FIXTURE_DIR / name
    if not path.is_file():
        raise FileNotFoundError(
            f"共有 train_contract fixture が見つからない: {path}"
            f"（リポジトリ直下 {_REPO_ROOT} 起点で fixtures/train_contract/{name} を"
            "解決できるか確認する）"
        )
    size = path.stat().st_size
    assert size <= _MAX_FIXTURE_BYTES, (
        f"{name} が上限（{_MAX_FIXTURE_BYTES} バイト）を超えている: {size}"
    )
    raw = path.read_bytes()
    raw.decode("ascii")  # fixture は ASCII のみで書く（非 ASCII は \uXXXX）契約。
    return json.loads(raw)


_LIMITS = _load_json_fixture("limits.json")
_DEFINITION = _load_json_fixture("definition.json")
_REQUEST_FULL = _load_json_fixture("request_full.json")
_REQUEST_MINIMAL = _load_json_fixture("request_minimal.json")
_REQUEST_WITH_VALIDATION = _load_json_fixture("request_with_validation.json")
_RESULT_OK_WITH_VALIDATION = _load_json_fixture("result_ok_with_validation.json")
_RESULT_CAP_CASES = _load_json_fixture("result_cap_cases.json")
_REJECT_CASES = _load_json_fixture("request_reject_cases.json")
_RESULT_OK = _load_json_fixture("result_ok.json")
_RESULT_ERROR = _load_json_fixture("result_error.json")


def test_req39_limits_fixture_matches_python_constants() -> None:
    """REQ-39: fixture の各上限値が `limits.py`／`contract.py`／`supervisor.py`
    の対応定数と一致すること。
    """
    assert _LIMITS["request_schema_version"] == contract.SCHEMA_VERSION
    assert _LIMITS["max_request_bytes"] == limits.MAX_REQUEST_BYTES
    assert _LIMITS["min_labels"] == limits.MIN_LABELS
    assert _LIMITS["max_labels"] == limits.MAX_LABELS
    assert _LIMITS["max_label_bytes"] == limits.MAX_LABEL_BYTES
    assert _LIMITS["min_max_bytes"] == limits.MIN_MAX_BYTES
    assert _LIMITS["max_max_bytes"] == limits.MAX_MAX_BYTES
    assert _LIMITS["min_seed"] == limits.MIN_SEED
    assert _LIMITS["max_seed"] == limits.MAX_SEED
    assert _LIMITS["max_train_wall_seconds"] == limits.MAX_TRAIN_WALL_SECONDS
    assert _LIMITS["max_train_rss_bytes"] == limits.MAX_TRAIN_RSS_BYTES

    from fandhe_edge_trainer import supervisor

    assert _LIMITS["max_result_bytes"] == supervisor._MAX_WORKER_STDOUT_BYTES
    # 学習ジョブ内採点用の validation 入力（REQ-18・REQ-27。issue #84 PR #238）。
    assert _LIMITS["max_validation_input_bytes"] == limits.MAX_VALIDATION_INPUT_BYTES
    assert _LIMITS["max_validation_id_bytes"] == limits.MAX_VALIDATION_ID_BYTES
    assert _LIMITS["max_validation_input_total_bytes"] == limits.MAX_VALIDATION_INPUT_TOTAL_BYTES
    assert _LIMITS["max_result_bytes_with_validation"] == limits.MAX_RESULT_BYTES_WITH_VALIDATION
    assert (
        _LIMITS["max_result_bytes_with_validation"]
        == supervisor._MAX_WORKER_STDOUT_BYTES_WITH_VALIDATION
    )
    assert list(_LIMITS["allowed_devices"]) == list(contract._ALLOWED_DEVICES)


def test_req18_request_full_fixture_has_exactly_the_contract_fields() -> None:
    """REQ-18: `request_full.json` のキー集合が `contract.py::_REQUEST_FIELDS`
    と完全一致する（Python 側でフィールドが増減したら検出する）。
    """
    # `validation_inputs` は任意項目のため `request_full.json` には含めず、
    # `request_with_validation.json` 側で全項目を網羅する。
    assert set(_REQUEST_FULL.keys()) == contract._REQUEST_FIELDS - {"validation_inputs"}
    assert set(_REQUEST_WITH_VALIDATION.keys()) == contract._REQUEST_FIELDS


def test_req27_validate_request_accepts_validation_inputs_without_gold(tmp_path: Path) -> None:
    """REQ-27: `request_with_validation.json` の `validation_inputs` が
    `(id, input)` の列として検証を通り、正解ラベルを持たないこと。
    """
    resolved = _resolve_root(_REQUEST_WITH_VALIDATION, tmp_path)
    req = contract.validate_request(resolved)
    try:
        assert req.validation_inputs == (
            ("val-001", "great product, works well"),
            ("val-002", "terrible, broke on day one"),
            ("val-003", "it is fine"),
        )
    finally:
        req.close_resources()


def test_req18_validation_inputs_absent_resolves_to_none(tmp_path: Path) -> None:
    """`validation_inputs` を省略したリクエストは `None`（採点なし）になる。"""
    req = contract.validate_request(_resolve_root(_REQUEST_FULL, tmp_path))
    try:
        assert req.validation_inputs is None
    finally:
        req.close_resources()


def test_req27_result_ok_with_validation_fixture_matches_request_ids() -> None:
    """REQ-27: `result_ok_with_validation.json` は `result_ok.json` に
    `validation_predictions` だけを足したもので、予測の `id` は
    `request_with_validation.json` の `validation_inputs[].id` と同順・同件数、
    各予測は `id`・`status`・`predicted_label` のみを持つ（`scores` は含めない）。
    """
    without = {k: v for k, v in _RESULT_OK_WITH_VALIDATION.items() if k != "validation_predictions"}
    assert without == _RESULT_OK
    predictions = _RESULT_OK_WITH_VALIDATION["validation_predictions"]
    assert [p["id"] for p in predictions] == [
        v["id"] for v in _REQUEST_WITH_VALIDATION["validation_inputs"]
    ]
    for prediction in predictions:
        assert list(prediction) == ["id", "status", "predicted_label"]


def _resolve_root(request_obj: dict, tmp_path: Path) -> dict:
    """`root` がプレースホルダーと一致する場合に限り `tmp_path` へ置き換える。"""
    resolved = dict(request_obj)
    if resolved.get("root") == _ROOT_PLACEHOLDER:
        resolved["root"] = str(tmp_path)
    return resolved


def test_req18_validate_request_accepts_request_full(tmp_path: Path) -> None:
    """REQ-18: `request_full.json`（root を tmp_path に置換）が
    `contract.validate_request` を通り、各フィールドが fixture の値と
    一致すること。
    """
    resolved = _resolve_root(_REQUEST_FULL, tmp_path)
    req = contract.validate_request(resolved)
    try:
        assert req.kind == _REQUEST_FULL["kind"]
        assert req.kind_version == _REQUEST_FULL["kind_version"]
        assert req.config == _REQUEST_FULL["config"]
        assert req.label_order == _REQUEST_FULL["label_order"]
        assert req.max_bytes == _REQUEST_FULL["max_bytes"]
        assert req.seed == _REQUEST_FULL["seed"]
        assert req.device == _REQUEST_FULL["device"]
        assert req.time_limit_seconds == _REQUEST_FULL["time_limit_seconds"]
        assert req.rss_limit_bytes == _REQUEST_FULL["rss_limit_bytes"]
    finally:
        req.close_resources()


def test_req18_validate_request_accepts_request_minimal_with_defaults(tmp_path: Path) -> None:
    """REQ-18: `request_minimal.json`（任意項目省略）が既定値
    （`MAX_TRAIN_WALL_SECONDS`／`MAX_TRAIN_RSS_BYTES`）で解決されること。
    """
    resolved = _resolve_root(_REQUEST_MINIMAL, tmp_path)
    req = contract.validate_request(resolved)
    try:
        assert req.config == {}
        assert req.time_limit_seconds == limits.MAX_TRAIN_WALL_SECONDS
        assert req.rss_limit_bytes == limits.MAX_TRAIN_RSS_BYTES
    finally:
        req.close_resources()


def test_req18_definition_label_order_matches_request_full_label_order() -> None:
    """`definition.json` の `options[].id`（宣言順）が `request_full.json` の
    `label_order` と一致すること（Rust 側 `label_order_from_definition` と
    同じ投影を pytest 側でも確認する）。
    """
    ids = [option["id"] for option in _DEFINITION["options"]]
    assert ids == _REQUEST_FULL["label_order"]


def _apply_case(case: dict[str, Any]) -> Any:
    """`base` に `patch` または `raw_text` を適用した結果を返す。"""
    if "raw_text" in case:
        return case["raw_text"]
    resolved = dict(_REJECT_CASES["base"])
    resolved.update(case["patch"])
    return resolved


def _validate_applied_case(applied: Any, tmp_path: Path) -> None:
    """`_apply_case` の結果を検証経路（生テキストなら parse も経る）へ通す。

    到達しないはず（`validate_request` は例外を送出する）だが、万一成功した
    場合に fd をリークさせないよう `close_resources()` を呼ぶ。
    """
    if isinstance(applied, str):
        parsed = contract.parse_request_bytes(applied.encode("utf-8"))
        contract.validate_request(parsed)
        return
    resolved = _resolve_root(applied, tmp_path)
    req = contract.validate_request(resolved)
    req.close_resources()


@pytest.mark.parametrize("case", _REJECT_CASES["cases"], ids=lambda c: c["name"])
def test_req39_reject_cases_match_expected_code_and_exit(
    case: dict[str, Any], tmp_path: Path
) -> None:
    """REQ-39: `request_reject_cases.json` の各ケースが、Python 側の検証でも
    fixture の `expected_code`／`expected_exit` と一致する終了コード・
    エラーコードで拒否されること。
    """
    applied = _apply_case(case)
    with pytest.raises(WorkerError) as exc_info:
        _validate_applied_case(applied, tmp_path)
    err = exc_info.value
    assert err.code == case["expected_code"], f"case={case['name']}"
    assert int(err.exit_code) == case["expected_exit"], f"case={case['name']}"


def test_req39_oversized_request_is_limit_exceeded() -> None:
    """REQ-39: `MAX_REQUEST_BYTES` を超えるリクエストは `limit_exceeded`／
    exit 20 になる（fixture には巨大な文字列を置かず、テスト内で生成する）。
    """
    huge = b"a" * (limits.MAX_REQUEST_BYTES + 1)
    with pytest.raises(WorkerError) as exc_info:
        contract.parse_request_bytes(huge)
    assert exc_info.value.code == "limit_exceeded"
    assert int(exc_info.value.exit_code) == 20


def test_req21_build_artifact_matches_result_ok_fixture() -> None:
    """REQ-21: `artifact.build_artifact` の出力が `result_ok.json` の
    `artifact` と完全一致すること（`created_utc` は monkeypatch で固定する）。
    """
    expected = _RESULT_OK["artifact"]
    art = artifact.build_artifact(
        kind=expected["kind"],
        kind_version=expected["kind_version"],
        config=expected["config"],
        label_order=expected["label_order"],
        output_type=expected["output_type"],
        max_bytes=expected["max_bytes"],
        candidate_label=expected["candidate_label"],
        onnx_sha256=expected["onnx_sha256"],
    )
    art["created_utc"] = expected["created_utc"]  # now_utc() は時刻依存のため固定する。
    assert art == expected


def test_req19_result_ok_config_matches_c3_default_merged_with_request_full() -> None:
    """REQ-19・REQ-21・REQ-39（codex 指摘 PR #220 P1「成果物の追加 config 値を
    検証せず成功扱いにしている」）: `result_ok.json` の `artifact.config`
    は `kinds/c3.py::train` が実際に返す実効 config
    （`{**DEFAULT_CONFIG, **request.config}`）と一致すること。

    Rust 側（`crates/train/src/result.rs`・`crates/train/src/
    kind_defaults.rs::effective_config`）は `config` を「`kind_defaults.json`
    （`DEFAULT_CONFIG` をそのまま書き出した共有 fixture）に `request.config`
    を上書きした実効 config」との完全一致で検査するが、その検査が想定する
    「既定値補完済みの実際のワーカー出力」の形がこのテストで固定される
    （`result_ok.json` を手で書き換えても、`kinds/c3.py::DEFAULT_CONFIG` と
    乖離すれば検出できる）。
    """
    expected_config = {**c3.DEFAULT_CONFIG, **_REQUEST_FULL["config"]}
    assert _RESULT_OK["artifact"]["config"] == expected_config


def test_req21_result_ok_candidate_label_matches_request_full_kind() -> None:
    """REQ-21・REQ-39（codex 指摘 PR #220「成果物の candidate_label が依頼と
    異なっても成功扱いになる」）: `cli.py::run_worker_train` は
    `candidate_label=request.kind` を記録するため、`result_ok.json` の
    `artifact.candidate_label` は `request_full.json` の `kind` と一致する
    こと（Rust 側 `TrainOutcome::from_worker_stdout` の一致検査が想定する
    Python 側の出力規則をここで固定する）。
    """
    assert _RESULT_OK["artifact"]["candidate_label"] == _REQUEST_FULL["kind"]


def test_req21_missing_request_file_matches_result_error_fixture(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """REQ-21: 存在しないリクエストパスを渡した `cli.main` の標準出力が
    `result_error.json` と完全一致し、戻り値が 64 であること。

    `result_error.json`（`fixtures/train_contract/PROVENANCE.md` 記載の
    生成コマンド）は固定パスで生成したが、本テスト自体は `tmp_path`
    （pytest が用意する使い捨てディレクトリ）配下の未作成パスを使う
    （`S108`: 固定の `/tmp` パスへ決め打ちしない）。エラーメッセージは
    `FileNotFoundError` の型名のみで、パス文字列を含まないため fixture との
    一致に影響しない。
    """
    missing_path = tmp_path / "does-not-exist-train-request.json"
    rc = cli.main(["train", "--request", str(missing_path)])
    assert rc == int(ExitCode.INVALID_INPUT)
    assert rc == 64
    captured = capsys.readouterr()
    printed = json.loads(captured.out.strip())
    assert printed == _RESULT_ERROR


@pytest.mark.parametrize("case", _RESULT_CAP_CASES["cases"], ids=lambda c: c["name"])
def test_req39_result_cap_cases_match_bound_function(case: dict[str, Any]) -> None:
    """P1（issue #84 PR #238 レビュー）: リクエストごとの結果上限の計算が、独立に
    計算した共有 fixture `result_cap_cases.json` と一致する（Rust 側
    `validation_result_bytes_bound` と同じ fixture で照合する）。
    """
    actual = contract.validation_result_bytes_bound(case["label_order"], case["ids"])
    assert actual == case["expected_max_result_bytes"], case["name"]


def test_req39_request_with_validation_fixture_has_computed_result_cap(tmp_path: Path) -> None:
    """`request_with_validation.json` の結果上限は、1 MiB ＋ 余裕 64 ＋ 3 件 ×
    (固定 50 ＋ id 7 ＋ 最長ラベル 8) = 1_048_835。supervisor の標準出力上限はこれを使う。
    """
    req = contract.validate_request(_resolve_root(_REQUEST_WITH_VALIDATION, tmp_path))
    try:
        assert req.max_result_bytes == 1_048_835
    finally:
        req.close_resources()
    plain = contract.validate_request(_resolve_root(_REQUEST_FULL, tmp_path))
    try:
        assert plain.max_result_bytes == limits.MAX_RESULT_BYTES
    finally:
        plain.close_resources()


def test_req39_control_char_label_with_many_inputs_is_rejected_before_training(
    tmp_path: Path,
) -> None:
    """P1: 制御文字だけの最長ラベル（エスケープで 1536 バイト）と多数の短い入力の
    組み合わせは、結果の最大長が天井を超えるため `limit_exceeded`／exit 20 で拒否される。
    件数が少なければ受理され、上限は天井以下で全件が最長ラベルの結果を収容できる。
    """
    labels = ["\u0001" * 256, "b"]

    def request_with(n: int) -> dict:
        resolved = _resolve_root(_REQUEST_FULL, tmp_path)
        resolved["label_order"] = labels
        resolved["validation_inputs"] = [{"id": f"a{i}", "input": ""} for i in range(n)]
        return resolved

    with pytest.raises(WorkerError) as exc_info:
        contract.validate_request(request_with(43_000))
    assert exc_info.value.code == "limit_exceeded"
    assert int(exc_info.value.exit_code) == 20

    accepted = contract.validate_request(request_with(1_000))
    try:
        assert accepted.max_result_bytes <= limits.MAX_RESULT_BYTES_WITH_VALIDATION
        worst_case = (
            limits.MAX_RESULT_BYTES
            + 28
            + sum(
                len(
                    json.dumps(
                        {"id": f"a{i}", "status": "abstain", "predicted_label": labels[0]},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
                + 1
                for i in range(1_000)
            )
        )
        assert accepted.max_result_bytes >= worst_case
    finally:
        accepted.close_resources()
