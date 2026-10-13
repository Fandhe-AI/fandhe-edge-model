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
import contextlib
import html
import json
import os
import stat
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

MAX_BYTES = 16 * 1024 * 1024  # JSONL 1 ファイルの読み込み上限
# JSON 1 ファイルの上限。一括で解析するため JSONL より小さくし、解析後のオブジェクト数も抑える
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_ROWS = 500  # 誤判定表の最大行数
INPUT_CHARS = 200  # 入力の表示文字数
MAX_DEPTH = 6  # 汎用表示の入れ子上限（超えたら整形 JSON）
# 汎用表示の上限（REQ-39）。入力が 64 MiB 未満でも、表の行 × 列の掛け算で出力が膨らまないようにする
MAX_ITEMS = 500  # 1 つの表・リストの行数
MAX_COLS = 50  # 辞書の配列を表にするときの列数
MAX_CELLS = 50_000  # レポート全体で表示する値の数（超えた分は「省略」）
MAX_VALUE_CHARS = 1_000  # 1 つの値の表示文字数（超えた分は「…」）
MAX_TOTAL_READ = 64 * 1024 * 1024  # 1 回の実行で読む合計バイト数
MAX_SECTION_CHARS = 16 * 1024 * 1024  # 1 節の HTML の文字数（超えたら節ごと省略）
MAX_PRE_CHARS = 100_000  # 整形 JSON の表示文字数
MAX_RECORDS = 100_000  # JSONL の行数（超えたら節ごと省略。読みながら数え、全件は溜めない）
MAX_ENTRIES = 1_000  # ディレクトリの項目数（候補の列挙）

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
    """全ての値を通す HTML エスケープ。長すぎる値は切り詰める（REQ-39）。"""
    t = str(v)
    if len(t) > MAX_VALUE_CHARS:
        t = t[:MAX_VALUE_CHARS] + "…"
    return html.escape(t, quote=True)


# 読み込みの起点（--project）。build_html が設定する。読み込みはすべてこの配下に閉じ込める（REQ-39）
ROOT = Path(".")


def _open_under_root(path: Path, directory: bool = False) -> int:
    """`ROOT` 配下の `path` を、途中も含めて symlink を辿らずに開いた fd を返す。

    ルート外・`..`・symlink は OSError（呼び出し側で「なし」）。FIFO で止まらないよう
    O_NONBLOCK で開き、通常ファイルかどうかは呼び出し側が fstat で確かめる。
    """
    parts = path.relative_to(ROOT).parts
    if not parts or any(x in ("..", ".") for x in parts):
        raise OSError("outside of the project")
    fd = os.open(ROOT, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for name in parts[:-1]:
            nfd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = nfd
        last = os.O_DIRECTORY if directory else os.O_NONBLOCK
        return os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | last, dir_fd=fd)
    finally:
        os.close(fd)


def list_dir(path: Path) -> list[str]:
    """`ROOT` 配下のディレクトリの名前一覧（symlink のディレクトリは辿らない）。"""
    fd = _open_under_root(path, directory=True)
    names: list[str] = []
    try:
        with os.scandir(fd) as it:
            for ent in it:
                if len(names) >= MAX_ENTRIES:
                    raise Skip("項目が多すぎるため省略")
                names.append(ent.name)
        return names
    finally:
        os.close(fd)


def read_text(path: Path, max_bytes: int = MAX_BYTES) -> str:
    """上限つきで読む。無い・通常ファイルでない・壊れている・大きすぎるは Skip。"""
    try:
        fd = _open_under_root(path)
    except (OSError, ValueError):
        raise Skip("なし") from None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise Skip("なし")
        global _read_left
        limit = min(max_bytes, _read_left)  # 1 ファイルと実行全体の両方の上限
        chunks, size = [], 0
        while size <= limit:  # 確認後に伸びても上限 + 1 バイトまでしか読まない
            b = os.read(fd, min(1 << 20, limit + 1 - size))
            if not b:
                break
            chunks.append(b)
            size += len(b)
        _read_left -= size
        if size > limit:
            raise Skip("大きすぎるため省略")
        return b"".join(chunks).decode("utf-8")
    except UnicodeDecodeError:
        raise Skip("なし") from None
    except OSError:
        raise Skip("なし") from None
    finally:
        os.close(fd)


def read_json(path: Path) -> Any:
    """JSON ファイルを読む。"""
    try:
        return json.loads(read_text(path, MAX_JSON_BYTES))
    except ValueError:
        raise Skip("なし") from None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """JSONL を行ごとに読む（壊れた行・オブジェクト以外は捨てる）。"""
    rows: list[dict[str, Any]] = []
    text, pos = read_text(path), 0
    while pos < len(text):
        end = text.find("\n", pos)
        end = len(text) if end < 0 else end
        line, pos = text[pos:end], end + 1
        if not line.strip():
            continue
        if len(rows) >= MAX_RECORDS:  # 溜める前に件数を確かめる（REQ-39）
            raise Skip("件数が多すぎるため省略")
        try:
            obj = json.loads(line)
        except (ValueError, RecursionError):
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


# 残りの表示セル数・読み込みバイト数。build_html が初期化し、render・read_text が減らす
_cells_left = MAX_CELLS
_read_left = MAX_TOTAL_READ


def pretty(v: Any) -> str:
    text = json.dumps(v, ensure_ascii=False, indent=2)
    if len(text) > MAX_PRE_CHARS:
        text = text[:MAX_PRE_CHARS] + "\n…（省略）"
    # 整形 JSON は MAX_PRE_CHARS で切ってあるので、値 1 つの上限（e）はかけない
    return f"<pre>{html.escape(text, quote=True)}</pre>"


def _omitted(total: int, shown: int) -> str:
    return f'<p class="none">{total} 件中 先頭 {shown} 件を表示</p>' if total > shown else ""


def render(v: Any, depth: int = 0) -> str:
    """JSON 値を汎用に表示する。未知の形・深すぎる入れ子は整形 JSON に落とす。

    行数・列数・全体のセル数に上限を置き、超えた分は省略と明示する（REQ-39）。
    """
    global _cells_left
    if _cells_left <= 0:
        return "…（省略）"
    _cells_left -= 1
    if depth > MAX_DEPTH:
        return pretty(v)
    if isinstance(v, dict):
        if not v:
            return "-"
        items = list(v.items())[:MAX_ITEMS]
        rows = "".join(f"<tr><th>{e(k)}</th><td>{render(x, depth + 1)}</td></tr>" for k, x in items)
        return f"<table>{rows}</table>{_omitted(len(v), len(items))}"
    if isinstance(v, list):
        if not v:
            return "-"
        head_items = v[:MAX_ITEMS]
        if all(isinstance(x, dict) for x in head_items):
            cols: list[str] = []
            for x in head_items:
                cols += [k for k in x if k not in cols][: MAX_COLS - len(cols)]
                if len(cols) >= MAX_COLS:
                    break
            head = "".join(f"<th>{e(c)}</th>" for c in cols)
            body = "".join(
                "<tr>"
                + "".join(f"<td>{render(x[c], depth + 1) if c in x else ''}</td>" for c in cols)
                + "</tr>"
                for x in head_items
            )
            return f"<table><tr>{head}</tr>{body}</table>{_omitted(len(v), len(head_items))}"
        lis = "".join(f"<li>{render(x, depth + 1)}</li>" for x in head_items)
        return f"<ul>{lis}</ul>{_omitted(len(v), len(head_items))}"
    return e("null" if v is None else v)


def section(title: str, build: Callable[[], str]) -> str:
    """節を作る。失敗は「なし」等で表示して続行する。"""
    try:
        body = build()
    except Skip as s:
        body = f'<p class="none">{e(s)}</p>'
    except Exception:  # 想定外の形でも他の節を止めない
        body = '<p class="none">なし</p>'
    if len(body) > MAX_SECTION_CHARS:
        body = '<p class="none">大きすぎるため省略</p>'
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
        dirs = [
            p / "candidates" / n
            for n in sorted(
                list_dir(p / "candidates"),
                key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 0, x),
            )
        ]
    except (OSError, ValueError):
        dirs = []
    for d in dirs:
        try:
            r = read_json(d / "result.json")
            a = r.get("artifact") or {}
            rows.append(
                {
                    "candidate": d.name,
                    # 保持する前に短くする（表示用の値だけを持つ。REQ-39）
                    "status": str(r.get("status"))[: MAX_VALUE_CHARS + 1],
                    "kind": str(a.get("kind"))[: MAX_VALUE_CHARS + 1],
                    "kind_version": str(a.get("kind_version"))[: MAX_VALUE_CHARS + 1],
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
    ab = rep.get("abstention") if isinstance(rep.get("abstention"), dict) else {}
    cal = rep.get("calibration") if isinstance(rep.get("calibration"), dict) else {}
    summary = {
        "候補": rep.get("candidate"),
        "種類": rep.get("kind"),
        "正解 / 件数": f"{rep.get('correct')} / {rep.get('n_total')}",
        "正解率": rep.get("accuracy"),
        "Macro-F1": rep.get("macro_f1"),
        "回答した割合（coverage）": ab.get("coverage"),
        "校正の採用": cal.get("adopted"),
    }
    out = "<h3>要点</h3>" + render(summary)
    # 要点に出したキーと見出しに不要なキーを除き、残りを評価器の出力のまま並べる（指標は計算しない）
    titles = {
        "per_label": "ラベル別",
        "macro_f1_excluded_labels": "Macro-F1 から除いたラベル",
        "out_of_scope_label": "対象外ラベル",
        "type_meaning_quadrant": "型と意味",
        "calibration": "校正",
        "abstention": "棄権",
        "comparison": "旧モデルとの比較",
        "reproducibility": "再現性",
        "diagnostics": "診断",
    }
    shown = {"step", "status", "candidate", "kind", "n_total", "correct", "accuracy", "macro_f1"}
    for k, v in rep.items():
        if k not in shown:
            out += f"<h3>{e(titles.get(k, k))}</h3>{render(v)}"
    return out


def correct_label(output: Any) -> str:
    """正解を output から取る。データ契約どおり `output.intent` を優先する（data の inspect）。

    intent が無い形は、1 値のオブジェクトならその値、文字列はそのまま。
    """
    if isinstance(output, dict) and "intent" in output:
        output = output["intent"]
    elif isinstance(output, dict) and len(output) == 1:
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
        # evaluation_predictions.jsonl は校正前の予測（status は ok・abstain・error）。
        # 校正による保留・対象外は行ごとに保存されていない。ここで判定し直さない
        # （評価ロジックは評価器 1 か所）
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

    note = (
        '<p class="none">校正前の予測を正解と突き合わせたもの。校正による保留・対象外の件数は'
        "「評価」の節の「棄権」を見てください（行ごとの内訳は保存されていません）。</p>"
    )
    return (
        f"{note}<h3>不一致（校正前の予測）</h3>{table(wrong)}"
        f"<h3>予測できなかった行（abstain・error）</h3>{table(held)}"
    )


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
    global ROOT, _cells_left, _read_left
    ROOT = p
    _cells_left = MAX_CELLS
    _read_left = MAX_TOTAL_READ
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
    # プロジェクトは読み取りのみ。配下へ書くのは report.html だけにし、入力・成果物の上書きを防ぐ
    proj = a.project.resolve()
    target = out.parent.resolve() / out.name
    if target.is_relative_to(proj) and target != proj / "report.html":
        print("error: --out must not point into the project except report.html", file=sys.stderr)
        return 64
    text = build_html(a.project, not a.no_input_text)
    tmp = None
    try:
        # 既存の出力先が通常ファイルでない（symlink 等）なら書かない。既存の inode は書き換えず、
        # 同じディレクトリの新しい一時ファイルへ書いて名前を置き換える（hardlink 先を壊さない）
        if out.is_symlink() or (out.exists() and not out.is_file()):
            raise OSError("output is not a regular file")
        fd, tmp = tempfile.mkstemp(dir=out.parent, prefix=".report-", suffix=".html")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        # mkstemp の 0600 のまま置く（誤判定の入力本文を含みうるため。共有するなら利用者が変える）
        os.replace(tmp, out)
        tmp = None
    except OSError:
        print("error: failed to write the report", file=sys.stderr)
        return 70
    finally:
        if tmp is not None:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
