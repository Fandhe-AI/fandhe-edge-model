#!/usr/bin/env python3
"""プロジェクトディレクトリから閲覧用の HTML レポートを 1 枚作る補助スクリプト。

7 工程の CLI 契約の外の補助ツール（REQ-35・M11）。標準ライブラリのみ・通信なし。
プロジェクトは読み取りのみ。指標は計算せず、evaluate が保存した JSON を表示するだけ
（評価ロジックは評価器 1 か所という評価契約。REQ-24〜26）。
読み込む各ファイルにはサイズ上限を設ける（REQ-39）。

使い方: python3 scripts/report_html.py --project DIR [--out FILE] [--no-input-text]
終了コード: 0 = 出力成功 / 64 = 入力不正 / 70 = 出力失敗。stdout は出力パス 1 行のみ。
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

MAX_BYTES = 64 * 1024 * 1024  # 1 ファイルの読み込み上限
MAX_ROWS = 500  # 誤判定表の最大行数
INPUT_CHARS = 200  # 入力の表示文字数
MAX_DEPTH = 6  # 汎用表示の入れ子上限（超えたら整形 JSON）

CSS = """
:root{color-scheme:light dark;--bg:#fff;--fg:#1a1a1a;--line:#ccc;--head:#eee}
@media (prefers-color-scheme:dark){:root{--bg:#1c1c1e;--fg:#e8e8e8;--line:#444;--head:#2c2c2e}}
body{background:var(--bg);color:var(--fg);font:14px/1.6 sans-serif;
max-width:1100px;margin:2em auto;padding:0 1em}
table{border-collapse:collapse;margin:.5em 0}
th,td{border:1px solid var(--line);padding:2px 8px;text-align:left;vertical-align:top}
th{background:var(--head)}pre{background:var(--head);padding:.5em;overflow:auto}
.none{opacity:.7}h2{border-bottom:1px solid var(--line)}
"""


class Skip(Exception):
    """節を出せない理由（表示用の日本語文）を運ぶ。"""


def e(v: Any) -> str:
    """全ての値を通す HTML エスケープ。"""
    return html.escape(str(v), quote=True)


def read_text(path: Path) -> str:
    """上限つきで読む。無い・壊れている・大きすぎるは Skip。"""
    try:
        if path.stat().st_size > MAX_BYTES:
            raise Skip("大きすぎるため省略")
        return path.read_text(encoding="utf-8")
    except Skip:
        raise
    except (OSError, UnicodeDecodeError):
        raise Skip("なし") from None


def read_json(path: Path) -> Any:
    """JSON ファイルを読む。"""
    try:
        return json.loads(read_text(path))
    except ValueError:
        raise Skip("なし") from None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """JSONL を行ごとに読む（壊れた行・オブジェクト以外は捨てる）。"""
    try:
        if path.stat().st_size > MAX_BYTES:
            raise Skip("大きすぎるため省略")
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
        return rows
    except Skip:
        raise
    except (OSError, UnicodeDecodeError):
        raise Skip("なし") from None


def pretty(v: Any) -> str:
    return f"<pre>{e(json.dumps(v, ensure_ascii=False, indent=2))}</pre>"


def render(v: Any, depth: int = 0) -> str:
    """JSON 値を汎用に表示する。未知の形・深すぎる入れ子は整形 JSON に落とす。"""
    if depth > MAX_DEPTH:
        return pretty(v)
    if isinstance(v, dict):
        if not v:
            return "-"
        rows = "".join(
            f"<tr><th>{e(k)}</th><td>{render(x, depth + 1)}</td></tr>" for k, x in v.items()
        )
        return f"<table>{rows}</table>"
    if isinstance(v, list):
        if not v:
            return "-"
        if all(isinstance(x, dict) for x in v):
            cols: list[str] = []
            for x in v:
                cols += [k for k in x if k not in cols]
            head = "".join(f"<th>{e(c)}</th>" for c in cols)
            body = "".join(
                "<tr>"
                + "".join(f"<td>{render(x[c], depth + 1) if c in x else ''}</td>" for c in cols)
                + "</tr>"
                for x in v
            )
            return f"<table><tr>{head}</tr>{body}</table>"
        return "<ul>" + "".join(f"<li>{render(x, depth + 1)}</li>" for x in v) + "</ul>"
    return e("null" if v is None else v)


def section(title: str, build: Callable[[], str]) -> str:
    """節を作る。失敗は「なし」等で表示して続行する。"""
    try:
        body = build()
    except Skip as s:
        body = f'<p class="none">{e(s)}</p>'
    except Exception:  # 想定外の形でも他の節を止めない
        body = '<p class="none">なし</p>'
    return f"<h2>{e(title)}</h2>{body}"


def overview(p: Path) -> str:
    out = ""
    try:
        d = read_json(p / "definition.json")
        out += f"<p>名前: {e(d.get('name'))} / 版: {e(d.get('version'))}</p>"
        opts = [
            {"id": o.get("id"), "display_name": o.get("display_name")}
            for o in d.get("options", [])
            if isinstance(o, dict)
        ]
        out += render(opts)
    except Skip as s:
        out += f'<p class="none">definition.json: {e(s)}</p>'
    try:
        sp = read_json(p / "split.json")
        rows = [
            {"split": k, "record_count": v.get("record_count")} for k, v in sp["splits"].items()
        ]
        out += f"<p>分割 (seed: {e(sp.get('seed'))})</p>" + render(rows)
    except (Skip, KeyError, AttributeError, TypeError):
        out += '<p class="none">split.json: なし</p>'
    return out


def selected_index(p: Path) -> int | None:
    try:
        i = read_json(p / "selection_record.json").get("candidate_index")
        return i if isinstance(i, int) else None
    except (Skip, AttributeError):
        return None


def candidates(p: Path) -> str:
    rows = []
    try:
        dirs = sorted(
            (p / "candidates").iterdir(),
            key=lambda x: (not x.name.isdigit(), int(x.name) if x.name.isdigit() else 0, x.name),
        )
    except OSError:
        dirs = []
    for d in dirs:
        try:
            r = read_json(d / "result.json")
            a = r.get("artifact") or {}
            rows.append(
                {
                    "candidate": d.name,
                    "status": r.get("status"),
                    "kind": a.get("kind"),
                    "kind_version": a.get("kind_version"),
                }
            )
        except Skip:
            rows.append({"candidate": d.name, "status": "なし"})
    out = render(rows) if rows else '<p class="none">なし</p>'
    try:
        out += "<h3>選定</h3>" + render(read_json(p / "selection_record.json"))
    except Skip:
        out += '<p class="none">selection_record.json: なし</p>'
    return out


def evaluation(p: Path) -> str:
    i = selected_index(p)
    if i is None:
        raise Skip("なし")
    cdir = p / "candidates" / str(i)
    try:
        rep = read_json(cdir / "evaluation_report.json")
    except Skip:
        out = '<p class="none">evaluate の結果が保存されていません</p>'
        rec = read_json(cdir / "evaluation_record.json")
        return out + render({"correct": rec.get("correct"), "total": rec.get("total")})
    if not isinstance(rep, dict):
        return pretty(rep)
    return "".join(f"<h3>{e(k)}</h3>{render(v)}" for k, v in rep.items())


def correct_label(output: Any) -> str:
    """正解を output から取る。1 値のオブジェクトはその値、文字列はそのまま。"""
    if isinstance(output, dict) and len(output) == 1:
        output = next(iter(output.values()))
    return (
        output
        if isinstance(output, str)
        else json.dumps(output, ensure_ascii=False, sort_keys=True)
    )


def top3(scores: Any) -> str:
    if not isinstance(scores, dict):
        return ""
    items = sorted(
        ((k, v) for k, v in scores.items() if isinstance(v, (int, float))), key=lambda kv: -kv[1]
    )[:3]
    return ", ".join(f"{k}: {v:.3f}" for k, v in items)


def mistakes(p: Path, show_input: bool) -> str:
    i = selected_index(p)
    if i is None:
        raise Skip("なし")
    preds = read_jsonl(p / "candidates" / str(i) / "evaluation_predictions.jsonl")
    data = {r.get("id"): r for r in read_jsonl(p / "data" / "evaluation.jsonl")}
    wrong, held = [], []
    for pr in preds:
        rec = data.get(pr.get("id"), {})
        row = {
            "id": pr.get("id"),
            "正解": correct_label(rec.get("output")) if "output" in rec else "",
            "予測": pr.get("predicted_label") or "",
            "スコア上位3": top3(pr.get("scores")),
        }
        if show_input:
            row["入力"] = str(rec.get("input", ""))[:INPUT_CHARS]
        if pr.get("status") != "ok":
            row["status"] = pr.get("status")
            held.append(row)
        elif "output" in rec and row["正解"] != row["予測"]:
            wrong.append(row)

    def table(rows: list[dict[str, Any]]) -> str:
        if not rows:
            return '<p class="none">なし</p>'
        note = (
            f"<p>{len(rows)} 件中 先頭 {MAX_ROWS} 行を表示</p>"
            if len(rows) > MAX_ROWS
            else f"<p>{len(rows)} 件</p>"
        )
        return note + render(rows[:MAX_ROWS])

    return f"<h3>不一致</h3>{table(wrong)}<h3>保留・対象外</h3>{table(held)}"


def package(p: Path) -> str:
    out = ""
    try:
        out += "<h3>artifact.json</h3>" + render(read_json(p / "package" / "artifact.json"))
    except Skip as s:
        out += f'<p class="none">artifact.json: {e(s)}</p>'
    try:
        out += "<h3>version_ledger.json</h3>" + render(
            read_json(p / "version_ledger.json").get("entries")
        )
    except (Skip, AttributeError):
        out += '<p class="none">version_ledger.json: なし</p>'
    return out


def build_html(p: Path, show_input: bool) -> str:
    secs = [
        section("概要", lambda: overview(p)),
        section("候補と選定", lambda: candidates(p)),
        section("評価", lambda: evaluation(p)),
        section("誤判定", lambda: mistakes(p, show_input)),
        section("パッケージ", lambda: package(p)),
    ]
    return (
        '<!doctype html><html lang="ja"><head><meta charset="utf-8">'
        f"<title>レポート</title><style>{CSS}</style></head><body><h1>レポート</h1>{''.join(secs)}</body></html>\n"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="閲覧用の HTML レポートを作る（REQ-35）")
    ap.add_argument("--project", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--no-input-text", action="store_true")
    try:
        a = ap.parse_args(argv)
    except SystemExit:
        return 64
    if not a.project.is_dir():
        print("error: --project is not a directory", file=sys.stderr)
        return 64
    out = a.out or a.project / "report.html"
    text = build_html(a.project, not a.no_input_text)
    try:
        # 出力先が symlink なら辿らず失敗させる
        fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o644)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
    except OSError:
        print("error: failed to write the report", file=sys.stderr)
        return 70
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
