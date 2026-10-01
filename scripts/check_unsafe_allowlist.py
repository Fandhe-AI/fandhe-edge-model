"""`allow(unsafe_code)` の出現箇所を許可リスト（unsafe-allowlist.json）と照合する。

REQ-39・#335（#329 の security-auditor 監査 P2）。ワークスペースは `unsafe_code = "deny"` で、
FFI などやむを得ない箇所だけを `#[allow(unsafe_code)]` と `// SAFETY:` コメントで局所的に許可する
（`.claude/rules/coding-rust.md`「unsafe・FFI」）。「`unsafe` の新規追加はオーナー承認」の規約を
機械的に担保するため、出現箇所（ファイル・対象 item）を承認記録付きの許可リストと fail-closed で
照合する。手順は `docs/design/unsafe-allowlist-flow.md`。

呼び出し元: `make check-unsafe-allowlist`（`make ci` の前提）・lefthook の pre-commit・
`trainer/tests/test_unsafe_allowlist.py`（python-ci の `make py-ci` 経由で全 PR に効く）。
製品には入らない検証用スクリプトで、標準ライブラリだけを使う（依存の追加なし）。通信・子プロセス
起動・書き込みは一切しない。`python3 -I` で起動する想定（`tomllib` は Python 3.11 以上）。

rustc / clippy の lint ではなくソースの文字列走査にしている理由: macOS 専用コード
（`#[cfg(target_os = "macos")]`）は Linux のビルドで除外されコンパイルされないため、lint では OS に
よって見逃す。走査なら cfg を問わず全ファイルを見るので、どの OS の CI でも検出できる。

入力（引数 `--root` のみ。環境変数は読まない）: `crates/` 配下の全 `*.rs`（`target/` を除く）・
ルートとメンバー crate の `Cargo.toml`・`.cargo/config*`・`unsafe-allowlist.json`。読み込む前に
`lstat` で通常ファイルであること（symlink を拒否）とサイズ上限を確認する（REQ-39）。

出力: stdout に JSON 1 行（キーは英語。ソース本文は含めず、ファイル・item・種別のみ）。
終了コードは 0=ok・10=judged_fail・64=invalid_input・70=runtime_error（REQ-21）。

照合規則（fail-closed）:
- 許可リストに無い (file, item, level) の出現は `unlisted_allow`、`count` 超過は `count_exceeded`、
  許可リストにあって出現が無い記録は `stale_entry`（削除・移動にも承認記録の更新を要求する）。
- 属性に `unsafe_code` が現れ、`allow` / `expect` / `warn` / `deny` / `forbid` のどれとも解釈
  できないものは `unparsed_attribute`。`deny` / `forbid` のみは通す。
- ルート `[workspace.lints.rust] unsafe_code` が deny / forbid 以外なら `lint_level_relaxed`。
  メンバー crate の `[lints]` が `workspace = true` のみでなければ `lints_override`。
  `.cargo/config*` に `unsafe_code` が現れたら `cargo_config_lint`（迂回経路の封じ込め）。

item の特定は rustfmt 済みのソースを前提にする（enclosing `mod` はインデントで判定。
`make fmt-check` が前提を担保する）。

承認済み item の本体の変更検出: 各記録は `body_sha256`（対象 item の本文を空白正規化したものの
sha256。複数出現は各ハッシュを昇順に改行連結して再ハッシュ）を持ち、item 内に `unsafe` ブロックを
足すなどの本体の変更は `body_changed` で止まる（再承認と記録の更新を要求する）。新しい値は
`--print-hashes` で得る（照合はせず、現在の出現ごとのハッシュを JSON で出す）。

走査対象内の symlink ディレクトリは Cargo が辿れて走査を回避できるため `src/` を含め入力不正とする。

限界（証拠種別: テストハーネス〔合成リポジトリでの陰性対照〕）: 許可リストを同じ PR で書き換えれば
機械照合は通る（承認の実在は PR レビューで確認する。AGENTS.md）。マクロ展開で生成される `allow` は
対象外。`unsafe` ブロック自体は `deny` のコンパイルエラーが担う。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tomllib
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

# 各入力ファイルの読み込み上限。現行の最大の .rs は数十 KB 程度。
MAX_FILE_BYTES = 4 * 1024 * 1024
# 走査するファイル数の上限（無制限の走査を作らない。REQ-39）。
MAX_FILES = 20000
# 出力の肥大化を防ぐ違反の列挙上限。
MAX_VIOLATIONS = 200

ALLOWLIST_NAME = "unsafe-allowlist.json"
SCHEMA_VERSION = 1

EXIT_OK = 0
EXIT_JUDGED_FAIL = 10
EXIT_INVALID_INPUT = 64
EXIT_RUNTIME_ERROR = 70

LEVELS = ("allow", "expect", "warn")
ENTRY_FIELDS = {
    "file",
    "item",
    "level",
    "count",
    "body_sha256",
    "approved_on",
    "approved_by",
    "record",
    "purpose",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DATE_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
WORD_UNSAFE_RE = re.compile(r"\bunsafe_code\b")
LEVEL_RE = re.compile(r"\b(allow|expect|warn|deny|forbid)\b")
RAW_STR_RE = re.compile(r"b?r(#*)\"")
ITEM_RE = re.compile(
    r"(?:pub(?:\s*\([^)]*\))?\s+|unsafe\s+|async\s+|default\s+"
    r"|const\s+(?=(?:unsafe\s+|async\s+)*fn\b)"
    r"|extern\s*(?:\"[^\"]*\")?\s+)*"
    r"(fn|mod|impl|struct|enum|trait|static|const|use|type|union)\b\s*(?:mut\s+)?([^{;(<=\s]*)"
)
MOD_RE = re.compile(r"^\s*(?:pub(?:\s*\([^)]*\))?\s+)?mod\s+(\w+)\s*\{")


class InputError(Exception):
    """許可リスト・manifest・ソースの形式不正・欠落・上限超過（終了コード 64）。"""


class Violation:
    """照合で見つけた 1 件の違反（JSON へ直列化する固定の語彙。本文は含めない）。"""

    def __init__(self, kind: str, file: str, item: str = "", level: str = "") -> None:
        """違反の種別・ファイル・item・レベルを保持する。"""
        self.kind = kind
        self.file = file
        self.item = item
        self.level = level

    def as_dict(self) -> dict[str, str]:
        """JSON 出力用の辞書へ変換する。"""
        return {"kind": self.kind, "file": self.file, "item": self.item, "level": self.level}


def read_bytes_limited(root: Path, rel: str) -> bytes:
    """ルート相対のパスを、通常ファイル・サイズ上限を確認してから読む（REQ-39）。"""
    path = root / rel
    try:
        st = path.lstat()
    except OSError as exc:
        raise InputError(f"cannot stat {rel}") from exc
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise InputError(f"{rel} is not a regular file")
    if st.st_size > MAX_FILE_BYTES:
        raise InputError(f"{rel} exceeds the size limit")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise InputError(f"cannot read {rel}") from exc


def read_text(root: Path, rel: str) -> str:
    """UTF-8 テキストとして読む。"""
    try:
        return read_bytes_limited(root, rel).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InputError(f"{rel} is not valid UTF-8") from exc


def load_toml(root: Path, rel: str) -> dict[str, Any]:
    """TOML を読み込む。構文エラーは入力不正として扱う。"""
    try:
        return tomllib.loads(read_text(root, rel))
    except tomllib.TOMLDecodeError as exc:
        raise InputError(f"{rel} is not valid TOML") from exc


def _ident(ch: str) -> bool:
    """識別子の一部になる文字か。"""
    return ch.isalnum() or ch == "_"


def _blank(chunk: str) -> str:
    """改行以外を空白にする（行番号とインデントを維持）。"""
    return "".join(c if c == "\n" else " " for c in chunk)


def mask_rust(src: str) -> str:
    """コメント・文字列・文字リテラルの内容を空白にする（改行は保つ）。

    ドキュメントコメントや文字列中の `allow(unsafe_code)` への言及を誤検出させないための前処理。
    ライフタイム（`'a`）と文字リテラル（`'x'`・`'\\n'`）は先読みで区別する。
    """
    out: list[str] = []
    i = 0
    n = len(src)
    while i < n:
        c = src[i]
        two = src[i : i + 2]
        prev_ident = i > 0 and _ident(src[i - 1])
        raw = RAW_STR_RE.match(src, i) if c in "rb" and not prev_ident else None
        if two == "//":
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append(_blank(src[i:j]))
            i = j
        elif two == "/*":
            depth, j = 1, i + 2
            while j < n and depth > 0:
                if src.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif src.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            out.append(_blank(src[i:j]))
            i = j
        elif raw is not None:
            end = src.find('"' + raw.group(1), raw.end())
            j = n if end < 0 else end + 1 + len(raw.group(1))
            out.append(_blank(src[i:j]))
            i = j
        elif c == '"' or (two == 'b"' and not prev_ident):
            j = i + (2 if c == "b" else 1)
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            j = min(j + 1, n)
            out.append(_blank(src[i:j]))
            i = j
        elif c == "'":
            if src[i + 1 : i + 2] == "\\":
                j = src.find("'", i + 3)
                j = n if j < 0 else j + 1
                out.append(_blank(src[i:j]))
                i = j
            elif src[i + 2 : i + 3] == "'" and src[i + 1 : i + 2] != "\n":
                out.append(_blank(src[i : i + 3]))
                i += 3
            else:
                out.append(c)
                i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _match_bracket(text: str, pos: int) -> tuple[int, bool]:
    """`[` の直後 pos から対応する `]` の次の位置を返す（対応が無ければ (len, False)）。"""
    depth, j = 1, pos
    while j < len(text) and depth > 0:
        if text[j] == "[":
            depth += 1
        elif text[j] == "]":
            depth -= 1
        j += 1
    return j, depth == 0


def find_attributes(masked: str) -> list[tuple[int, int, bool, str]]:
    """`#[...]` / `#![...]` を括弧対応の線形走査で抽出し (開始, 終端, 内部属性か, 本文) を返す。"""
    found: list[tuple[int, int, bool, str]] = []
    for m in re.finditer(r"#\s*(!?)\s*\[", masked):
        end, closed = _match_bracket(masked, m.end())
        body = masked[m.end() : end - 1] if closed else masked[m.end() :]
        found.append((m.start(), end, m.group(1) == "!", body))
    return found


def enclosing_mods(lines: list[str], line_idx: int) -> list[str]:
    """属性行より浅いインデントの `mod X {` を上方向に辿り、外側から順の mod 名を返す。"""
    cur = len(lines[line_idx]) - len(lines[line_idx].lstrip())
    mods: list[str] = []
    for k in range(line_idx - 1, -1, -1):
        text = lines[k]
        if not text.strip():
            continue
        indent = len(text) - len(text.lstrip())
        if indent < cur:
            cur = indent
            m = MOD_RE.match(text)
            if m:
                mods.append(m.group(1))
    return list(reversed(mods))


def _item_name(rest: str) -> str:
    """属性の直後のテキストから対象 item の名前を得る（後続の属性は読み飛ばす）。"""
    while True:
        rest = rest.lstrip()
        am = re.match(r"#\s*!?\s*\[", rest)
        if not am:
            break
        end, _ = _match_bracket(rest, am.end())
        rest = rest[end:]
    im = ITEM_RE.match(rest)
    if im and im.group(1) == "impl":
        header = re.split(r"[{;]", rest, maxsplit=1)[0]
        return " ".join(header.split())[:120]
    if im and im.group(2):
        return im.group(2)
    return "<other>"


def _item_end(masked: str, pos: int) -> int:
    """属性の直後 pos から対象 item の終端位置を返す（`;` か対応する `}`。閉じなければ末尾）。"""
    n = len(masked)
    depth = 0
    j = pos
    while j < n:
        ch = masked[j]
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(depth - 1, 0)
        elif ch == ";" and depth == 0:
            return j + 1
        elif ch == "{" and depth == 0:
            braces, j = 1, j + 1
            while j < n and braces > 0:
                braces += {"{": 1, "}": -1}.get(masked[j], 0)
                j += 1
            return j
        j += 1
    return n


def body_hash(src: str, start: int, end: int) -> str:
    """item の本文（属性を含む元ソースの区間）を空白正規化して sha256 を返す。"""
    return hashlib.sha256(" ".join(src[start:end].split()).encode("utf-8")).hexdigest()


def combine_hashes(hashes: list[str]) -> str:
    """同一キーの複数出現のハッシュを昇順に改行連結して再ハッシュする。"""
    return hashlib.sha256("\n".join(sorted(hashes)).encode("utf-8")).hexdigest()


def scan_source(
    rel: str, src: str, violations: list[Violation]
) -> list[tuple[tuple[str, str, str], str]]:
    """1 ファイルの許可の出現を ((file, item, level), 本体ハッシュ) の列で返す。

    解釈不能な形は違反に積む。
    """
    masked = mask_rust(src)
    lines = masked.split("\n")
    occurrences: list[tuple[tuple[str, str, str], str]] = []
    for start, end, inner, body in find_attributes(masked):
        if not WORD_UNSAFE_RE.search(body):
            continue
        levels = set(LEVEL_RE.findall(body))
        if not levels:
            violations.append(Violation("unparsed_attribute", rel))
            continue
        relaxed = [lv for lv in LEVELS if lv in levels]
        if not relaxed:
            continue  # deny / forbid のみ
        mods = enclosing_mods(lines, masked.count("\n", 0, start))
        name = "<inner>" if inner else _item_name(masked[end:])
        item = "::".join([*mods, name])
        # 内部属性はファイル全体、外部属性は属性から item 終端までを本体とする
        digest = (
            body_hash(src, 0, len(src)) if inner else body_hash(src, start, _item_end(masked, end))
        )
        for lv in relaxed:
            occurrences.append(((rel, item, lv), digest))
    return occurrences


def collect_rs_files(root: Path) -> list[str]:
    """`crates/` 配下の全 `*.rs` のルート相対 POSIX パス（symlink は辿らず、`target/` は除外）。"""
    crates = root / "crates"
    if crates.is_symlink() or not crates.is_dir():
        raise InputError("crates directory is missing")
    files: list[str] = []
    for dirpath, dirnames, filenames in os.walk(crates, followlinks=False):
        for d in dirnames:
            if d != "target" and Path(dirpath, d).is_symlink():
                raise InputError("symlinked directory under crates/ is not allowed")
        dirnames[:] = sorted(d for d in dirnames if d != "target")
        for fn in sorted(filenames):
            if fn.endswith(".rs"):
                files.append(PurePosixPath(*Path(dirpath, fn).relative_to(root).parts).as_posix())
                if len(files) > MAX_FILES:
                    raise InputError("too many source files")
    return files


def load_allowlist(root: Path) -> dict[tuple[str, str, str], tuple[int, str]]:
    """許可リストを検証して {(file, item, level): (count, body_sha256)} を返す。

    形式不正は InputError。
    """
    try:
        data = json.loads(read_text(root, ALLOWLIST_NAME))
    except json.JSONDecodeError as exc:
        raise InputError(f"{ALLOWLIST_NAME} is not valid JSON") from exc
    if not isinstance(data, dict) or set(data) != {"schema_version", "entries"}:
        raise InputError(f"{ALLOWLIST_NAME} must have exactly schema_version and entries")
    if data["schema_version"] != SCHEMA_VERSION or isinstance(data["schema_version"], bool):
        raise InputError(f"{ALLOWLIST_NAME} schema_version must be {SCHEMA_VERSION}")
    entries = data["entries"]
    if not isinstance(entries, list):
        raise InputError("entries must be a list")
    result: dict[tuple[str, str, str], tuple[int, str]] = {}
    for e in entries:
        if not isinstance(e, dict) or set(e) != ENTRY_FIELDS:
            raise InputError("each entry must have exactly the documented fields")
        for k in (
            "file",
            "item",
            "level",
            "body_sha256",
            "approved_on",
            "approved_by",
            "record",
            "purpose",
        ):
            if not isinstance(e[k], str) or not e[k].strip():
                raise InputError(f"entry field {k} must be a non-empty string")
        p = PurePosixPath(e["file"])
        if p.is_absolute() or ".." in p.parts or "\\" in e["file"] or p.suffix != ".rs":
            raise InputError("entry file must be a relative POSIX path to a .rs file")
        if p.parts[:1] != ("crates",):
            raise InputError("entry file must be under crates/")
        if e["level"] not in LEVELS:
            raise InputError("entry level must be allow, expect or warn")
        if isinstance(e["count"], bool) or not isinstance(e["count"], int) or e["count"] < 1:
            raise InputError("entry count must be a positive integer")
        if not SHA256_RE.match(e["body_sha256"]):
            raise InputError("entry body_sha256 must be 64 lowercase hex digits")
        if not DATE_RE.match(e["approved_on"]):
            raise InputError("entry approved_on must be YYYY-MM-DD")
        try:
            date.fromisoformat(e["approved_on"])
        except ValueError as exc:
            raise InputError("entry approved_on is not a valid date") from exc
        key = (e["file"], e["item"], e["level"])
        if key in result:
            raise InputError("duplicate allowlist entry")
        result[key] = (e["count"], e["body_sha256"])
    return result


def check_lint_levels(root: Path, violations: list[Violation]) -> None:
    """lint レベルの迂回経路（ルート・メンバー crate の manifest・.cargo/config*）を検査する。"""
    root_toml = load_toml(root, "Cargo.toml")
    level = root_toml.get("workspace", {}).get("lints", {}).get("rust", {}).get("unsafe_code")
    if isinstance(level, dict):
        level = level.get("level")
    if level not in ("deny", "forbid"):
        violations.append(Violation("lint_level_relaxed", "Cargo.toml"))
    try:
        names = sorted(
            e.name for e in os.scandir(root / "crates") if e.is_dir(follow_symlinks=False)
        )
    except OSError as exc:
        raise InputError("cannot list crates") from exc
    for name in names:
        rel = f"crates/{name}/Cargo.toml"
        if os.path.lexists(root / rel) and load_toml(root, rel).get("lints") != {"workspace": True}:
            violations.append(Violation("lints_override", rel))
    for rel in (".cargo/config.toml", ".cargo/config"):
        if os.path.lexists(root / rel) and WORD_UNSAFE_RE.search(read_text(root, rel)):
            violations.append(Violation("cargo_config_lint", rel))


def collect_occurrences(
    root: Path, violations: list[Violation]
) -> dict[tuple[str, str, str], list[str]]:
    """全 `*.rs` を走査し {(file, item, level): [本体ハッシュ, ...]} を返す。"""
    found: dict[tuple[str, str, str], list[str]] = {}
    for rel in collect_rs_files(root):
        for key, digest in scan_source(rel, read_text(root, rel), violations):
            found.setdefault(key, []).append(digest)
    return found


def print_hashes(root: Path) -> tuple[int, dict[str, Any]]:
    """現在の出現ごとに記録へ書く `body_sha256` を出力する（照合はしない。記録更新の補助）。"""
    try:
        found = collect_occurrences(root, [])
    except InputError as exc:
        return EXIT_INVALID_INPUT, {
            "status": "invalid_input",
            "code": EXIT_INVALID_INPUT,
            "message": str(exc),
        }
    items = [
        {
            "file": k[0],
            "item": k[1],
            "level": k[2],
            "count": len(v),
            "body_sha256": combine_hashes(v),
        }
        for k, v in sorted(found.items())
    ]
    return EXIT_OK, {"status": "ok", "code": EXIT_OK, "occurrences": items}


def run(root: Path) -> tuple[int, dict[str, Any]]:
    """照合を実行し、(終了コード, 出力 JSON の辞書) を返す。"""
    try:
        allowed = load_allowlist(root)
        violations: list[Violation] = []
        found = collect_occurrences(root, violations)
        for key in sorted(found):
            if key not in allowed:
                violations.append(Violation("unlisted_allow", key[0], key[1], key[2]))
            elif len(found[key]) > allowed[key][0]:
                violations.append(Violation("count_exceeded", key[0], key[1], key[2]))
            elif (
                combine_hashes(found[key]) != allowed[key][1] and len(found[key]) == allowed[key][0]
            ):
                violations.append(Violation("body_changed", key[0], key[1], key[2]))
        for key in sorted(allowed):
            if len(found.get(key, [])) < allowed[key][0]:
                violations.append(Violation("stale_entry", key[0], key[1], key[2]))
        check_lint_levels(root, violations)
    except InputError as exc:
        return EXIT_INVALID_INPUT, {
            "status": "invalid_input",
            "code": EXIT_INVALID_INPUT,
            "message": str(exc),
            "violation_count": 0,
            "violations": [],
        }
    if violations:
        return EXIT_JUDGED_FAIL, {
            "status": "judged_fail",
            "code": EXIT_JUDGED_FAIL,
            "message": "allow(unsafe_code) occurrences do not match the approved allowlist",
            "violation_count": len(violations),
            "violations_truncated": len(violations) > MAX_VIOLATIONS,
            "violations": [x.as_dict() for x in violations[:MAX_VIOLATIONS]],
        }
    return EXIT_OK, {"status": "ok", "code": EXIT_OK, "violation_count": 0, "violations": []}


class _ArgumentError(Exception):
    """コマンドライン引数の不正（argparse の usage 出力・SystemExit を避けるための例外）。"""


class _JsonArgumentParser(argparse.ArgumentParser):
    """引数エラーで stderr へ usage を出さず `_ArgumentError` を送出する parser。"""

    def error(self, message: str) -> NoReturn:
        raise _ArgumentError(message)


def main(argv: list[str]) -> int:
    """エントリポイント。例外は最後に runtime_error(70) へ写す（REQ-21）。"""
    try:
        parser = _JsonArgumentParser(description=__doc__.splitlines()[0])
        parser.add_argument("--root", required=True, type=Path, help="repository root")
        parser.add_argument(
            "--print-hashes", action="store_true", help="print body_sha256 values and exit"
        )
        try:
            args = parser.parse_args(argv)
        except _ArgumentError as exc:
            payload = {
                "status": "invalid_input",
                "code": EXIT_INVALID_INPUT,
                "message": f"invalid arguments: {exc}",
                "violation_count": 0,
                "violations": [],
            }
            sys.stdout.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
            return EXIT_INVALID_INPUT
        except SystemExit as exc:  # --help（終了コード 0）のみ
            return EXIT_OK if exc.code in (0, None) else EXIT_INVALID_INPUT
        code, payload = print_hashes(args.root) if args.print_hashes else run(args.root)
    except Exception:  # 予期しない例外も JSON 1 つと終了コード 70 に揃える
        code, payload = (
            EXIT_RUNTIME_ERROR,
            {
                "status": "runtime_error",
                "code": EXIT_RUNTIME_ERROR,
                "message": "unexpected error",
                "violation_count": 0,
                "violations": [],
            },
        )
    sys.stdout.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
