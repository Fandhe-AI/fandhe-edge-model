"""閲覧用 HTML レポート（scripts/report_html.py）の単体テスト。

REQ-35（閲覧用の表示）・REQ-39（入力はエスケープ・出力先 symlink を辿らない）。
証拠種別: テストハーネス（合成データ）。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "report_html.py"
spec = importlib.util.spec_from_file_location("report_html", SCRIPT)
assert spec is not None
assert spec.loader is not None
mod = importlib.util.module_from_spec(spec)
sys.modules["report_html"] = mod
spec.loader.exec_module(mod)


def jl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def make_project(root: Path, bad_input: str = "x one") -> Path:
    (root / "candidates" / "0").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "package").mkdir()
    (root / "definition.json").write_text(
        json.dumps({"name": "demo", "version": 3, "options": [{"id": "a", "display_name": "Aaa"}]}),
        encoding="utf-8",
    )
    (root / "split.json").write_text(
        json.dumps({"seed": 1, "splits": {"test": {"record_count": 3}}}), encoding="utf-8"
    )
    (root / "selection_record.json").write_text(
        json.dumps(
            {"candidate_index": 0, "rule": "r1", "validation_correct": 2, "validation_total": 3}
        ),
        encoding="utf-8",
    )
    c = root / "candidates" / "0"
    (c / "result.json").write_text(
        json.dumps({"status": "ok", "artifact": {"kind": "c1"}}), encoding="utf-8"
    )
    (c / "evaluation_report.json").write_text(
        json.dumps({"accuracy": 0.5, "mystery": {"z": [1, 2]}}), encoding="utf-8"
    )
    jl(
        c / "evaluation_predictions.jsonl",
        [
            {"id": "e1", "status": "ok", "predicted_label": "a", "scores": {"a": 0.9, "b": 0.1}},
            {
                "id": "e2",
                "status": "ok",
                "predicted_label": "b",
                "scores": {"a": 0.2, "b": 0.7, "c": 0.1},
            },
            {"id": "e3", "status": "abstained", "predicted_label": None, "scores": {}},
        ],
    )
    jl(
        root / "data" / "evaluation.jsonl",
        [
            {"id": "e1", "input": "x zero", "output": {"intent": "a"}},
            {"id": "e2", "input": bad_input, "output": {"intent": "a"}},
            {"id": "e3", "input": "x two", "output": {"intent": "c"}},
        ],
    )
    (root / "package" / "artifact.json").write_text(
        json.dumps({"kind": "c1", "max_bytes": 512}), encoding="utf-8"
    )
    (root / "version_ledger.json").write_text(
        json.dumps({"entries": [{"kind": "model", "id": "v1"}]}), encoding="utf-8"
    )
    return root


def run(root: Path, *extra: str) -> str:
    assert mod.main(["--project", str(root), *extra]) == 0
    return (root / "report.html").read_text(encoding="utf-8")


def test_all_sections_and_mistakes_req35(tmp_path: Path) -> None:
    """全節が出て、誤判定は e2 だけ・e3 は保留表に出る。"""
    h = run(make_project(tmp_path))
    for t in (
        "概要",
        "候補と選定",
        "評価",
        "誤判定",
        "パッケージ",
        "demo",
        "Aaa",
        "mystery",
        "validation_correct",
    ):
        assert t in h
    wrong, held = h.split("保留・対象外")
    wrong = wrong.split("不一致")[1]
    assert "e2" in wrong
    assert "b: 0.700, a: 0.200, c: 0.100" in wrong
    assert "e1" not in wrong
    assert "e3" not in wrong
    assert "abstained" in held


def test_script_input_escaped_req35(tmp_path: Path) -> None:
    h = run(make_project(tmp_path, bad_input="<script>alert(1)</script>"))
    assert "<script>" not in h
    assert "&lt;script&gt;" in h


def test_missing_files_show_none_exit0_req35(tmp_path: Path) -> None:
    h = run(tmp_path)
    assert h.count("なし") >= 4


def test_no_input_text_req35(tmp_path: Path) -> None:
    h = run(make_project(tmp_path), "--no-input-text")
    assert "<th>入力</th>" not in h
    assert "x one" not in h


def test_bad_project_exit64_req35(tmp_path: Path) -> None:
    assert mod.main(["--project", str(tmp_path / "nope")]) == 64


def test_out_symlink_not_followed_req39(tmp_path: Path) -> None:
    target = tmp_path / "t.txt"
    target.write_text("keep")
    link = tmp_path / "l.html"
    link.symlink_to(target)
    assert mod.main(["--project", str(tmp_path), "--out", str(link)]) == 70
    assert target.read_text() == "keep"
