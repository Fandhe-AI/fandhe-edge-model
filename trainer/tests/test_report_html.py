"""閲覧用 HTML レポート（scripts/report_html.py）の単体テスト。

REQ-35（閲覧用の表示）・REQ-39（入力はエスケープ・出力先 symlink を辿らない）。
証拠種別: テストハーネス（合成データ）。
"""

from __future__ import annotations

import importlib.util
import json
import os
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
    """全節が出て、誤判定は e2 だけ・e3 は予測できなかった行の表に出る。"""
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
    wrong, held = h.split("予測できなかった行")
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
    (tmp_path / "p").mkdir()
    assert mod.main(["--project", str(tmp_path / "p"), "--out", str(link)]) == 70
    assert target.read_text() == "keep"


def test_reads_stay_inside_the_project_req39(tmp_path: Path) -> None:
    """symlink でプロジェクト外を指すファイル・候補ディレクトリは読まない（経路の閉じ込め）。"""
    root = make_project(tmp_path / "p")
    outside = tmp_path / "outside"
    (outside / "9").mkdir(parents=True)
    (outside / "secret.json").write_text(json.dumps({"kind": "LEAKED"}), encoding="utf-8")
    (outside / "9" / "result.json").write_text(json.dumps({"status": "LEAKED2"}), encoding="utf-8")
    art = root / "package" / "artifact.json"
    art.unlink()
    art.symlink_to(outside / "secret.json")
    (root / "candidates" / "9").symlink_to(outside / "9")
    h = run(root)
    assert "LEAKED" not in h
    assert "artifact.json: なし" in h


def test_fifo_input_does_not_hang_req39(tmp_path: Path) -> None:
    """通常ファイルでない入力（FIFO）は開いても待たず「なし」になる（資源の上限）。"""
    root = make_project(tmp_path)
    (root / "definition.json").unlink()
    os.mkfifo(root / "definition.json")
    h = run(root)
    assert "definition.json: なし" in h


def test_out_into_project_is_refused_req39(tmp_path: Path) -> None:
    """--out にプロジェクトの入力（report.html 以外）を指定したら 64 で拒否し、上書きしない。"""
    root = make_project(tmp_path)
    before = (root / "selection_record.json").read_bytes()
    assert mod.main(["--project", str(root), "--out", str(root / "selection_record.json")]) == 64
    assert (root / "selection_record.json").read_bytes() == before


def test_generic_tables_are_bounded_req39(tmp_path: Path) -> None:
    """互いに異なるキーの辞書が大量にあっても、行・列・セル数の上限で出力が膨らまない。"""
    root = make_project(tmp_path)
    big = [{f"k{i}": i} for i in range(20000)]
    (root / "selection_record.json").write_text(json.dumps({"x": big}), encoding="utf-8")
    h = run(root)
    assert len(h) < 2_000_000
    assert "20000 件中 先頭 500 件を表示" in h


def test_out_hardlink_does_not_truncate_the_shared_file_req39(tmp_path: Path) -> None:
    """report.html が入力への hardlink でも入力を書き換えない（新しい inode へ置き換える）。"""
    root = make_project(tmp_path)
    before = (root / "definition.json").read_bytes()
    os.link(root / "definition.json", root / "report.html")
    run(root)
    assert (root / "definition.json").read_bytes() == before


def test_report_is_private_by_default_req39(tmp_path: Path) -> None:
    """レポートは入力本文を含みうるため、所有者だけが読める 0600 で作る。"""
    root = make_project(tmp_path)
    run(root)
    assert (root / "report.html").stat().st_mode & 0o777 == 0o600
