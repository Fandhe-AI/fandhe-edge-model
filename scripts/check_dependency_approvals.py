"""依存の承認台帳（dependency-approvals.json）と manifest・lock を照合し、未承認の依存変更を止める。

REQ-38・TASK-38.3・#165。依存の追加・更新・削除は通信を伴う操作（crates.io・PyPI からの取得）で、
ユーザーの明示承認が要る（`.claude/rules/dependency-policy.md`）。承認を口頭や PR 本文だけに
任せると記録の欠落に気付けないため、承認記録を機械可読な台帳に残し、manifest・lock との差分が
あれば fail-closed で失敗させる。手順は `docs/design/dependency-approval-flow.md`。

呼び出し元: `make check-dependency-approvals`（`make ci` の前提）・lefthook の pre-commit・
`trainer/tests/test_dependency_approvals.py`（python-ci の `make py-ci` 経由で全 PR に効く）。
製品（CLI・推論経路・配布物）には入らない検証用スクリプトで、標準ライブラリだけを使う
（依存の追加なし）。通信・子プロセス起動・書き込みは一切しない。`python3 -I` で起動する想定
（tomllib は Python 3.11 以上が必要なため、Makefile は trainer の Python を `uv run` で使う）。

入力（引数 `--root` のみ。環境変数は読まない）: ルート相対の固定パス
`Cargo.toml`・各メンバー crate の `Cargo.toml`・`Cargo.lock`・`trainer/pyproject.toml`・
`trainer/uv.lock`・`dependency-approvals.json`。読み込む前に `lstat` で通常ファイルであること
（symlink を拒否）とサイズ上限（REQ-39 と同じ方針）を確認する。

出力: stdout に JSON 1 行（キーは英語）。終了コードは 7 種のうち 4 つを使う（REQ-21）。
0=全件承認済み（ok）・10=未承認・固定違反・余分な記録（judged_fail）・
64=台帳・manifest の形式不正や欠落（invalid_input）・70=予期しない例外（runtime_error）。

照合規則の要点:
- Rust: ルートの `[workspace.dependencies]` は `=x.y.z` の完全固定（git・registry・範囲指定は
  拒否）で、
  台帳の `cargo.direct` に記録があること。メンバー crate は `workspace = true` のみ（依存の集約を
  機械で担保）。`Cargo.lock` の registry パッケージは台帳（direct ∪ locked）に (name, version) が
  あること。source の無いパッケージはワークスペースメンバー名に限る（内部 crate の追加は台帳不要）。
- Python: `pyproject.toml` の依存は `name[extras]==x.y.z` のみ。`uv.lock` の registry パッケージも
  台帳に (name, version) があること。
- ルート manifest 自身の `[dependencies]` 等もメンバーと同じ規則（`workspace = true` のみ）で
  検査する。
  ワークスペース依存の `path` は、実在するメンバーのパスで、依存名がその package 名と一致すること。
- 配置層・機能: メンバー crate の依存は、台帳 `direct[].layers` に crate の層（`crates/<層>`。
  dev-dependencies のみなら `<層>(dev)` も可）が含まれること。ルートの features・
  default-features は台帳の `features`・`default_features` と一致すること。メンバー側の
  features・default-features・optional による上書きは拒否する。
- Python: `[build-system].requires` も固定と台帳記録を照合する（uv.lock には現れないので
  lock 照合は対象外）。
- 逆方向: manifest の直接依存が lock に同じ版で現れること（`missing_in_lock`）。
- 台帳の余分な記録（manifest・lock に無い記録）も失敗（削除にも承認記録の更新が要る）。

限界: 台帳を同じ PR で書き換えれば機械照合は通る。承認の実在は PR レビューで確認する
（AGENTS.md「依存の追加・更新」）。証拠種別: テストハーネス（合成リポジトリでの陰性対照）。
"""

from __future__ import annotations

import argparse
import json
import re
import stat
import sys
import tomllib
from datetime import date
from pathlib import Path
from typing import Any, NoReturn

# 各入力ファイルの読み込み上限。現行の最大は Cargo.lock・uv.lock の数十 KB 程度。
MAX_FILE_BYTES = 4 * 1024 * 1024
# 出力の肥大化を防ぐ違反の列挙上限。
MAX_VIOLATIONS = 200

LEDGER_NAME = "dependency-approvals.json"
SCHEMA_VERSION = 1
CRATES_IO_SOURCE = "registry+https://github.com/rust-lang/crates.io-index"
PYPI_SOURCE = "https://pypi.org/simple"

EXIT_OK = 0
EXIT_JUDGED_FAIL = 10
EXIT_INVALID_INPUT = 64
EXIT_RUNTIME_ERROR = 70

CARGO_PIN_RE = re.compile(r"^=([0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?)$")
PY_REQ_RE = re.compile(
    r"^([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)"
    r"(?:\[([A-Za-z0-9,._ -]+)\])?"
    r"==([0-9]+\.[0-9]+\.[0-9]+)$"
)
DATE_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
CARGO_DEP_TABLES = ("dependencies", "dev-dependencies", "build-dependencies")

DIRECT_FIELDS = {"name", "version", "approved_on", "approved_by", "record", "purpose", "layers"}
# cargo の直接依存は、承認した機能構成（features・default-features）も台帳に持つ。
CARGO_DIRECT_FIELDS = DIRECT_FIELDS | {"features", "default_features"}
LOCKED_FIELDS = {"name", "version", "basis"}


class InputError(Exception):
    """台帳・manifest・lock の形式不正・欠落・上限超過（終了コード 64）。"""


class Violation:
    """照合で見つけた 1 件の違反（JSON へ直列化する固定の語彙）。"""

    def __init__(self, kind: str, ecosystem: str, name: str, version: str, file: str) -> None:
        """違反の種別・対象の生態系・名前・版・検出したファイルを保持する。"""
        self.kind = kind
        self.ecosystem = ecosystem
        self.name = name
        self.version = version
        self.file = file

    def as_dict(self) -> dict[str, str]:
        """JSON 出力用の辞書へ変換する。"""
        return {
            "kind": self.kind,
            "ecosystem": self.ecosystem,
            "name": self.name,
            "version": self.version,
            "file": self.file,
        }


def read_bytes_limited(root: Path, rel: str) -> bytes:
    """ルート相対の固定パスを、通常ファイル・サイズ上限を確認してから読む（REQ-39）。"""
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


def load_toml(root: Path, rel: str) -> dict[str, Any]:
    """TOML を読み込む。構文エラーは入力不正として扱う。"""
    try:
        return tomllib.loads(read_bytes_limited(root, rel).decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise InputError(f"{rel} is not valid TOML") from exc


def norm_py(name: str) -> str:
    """PyPI のパッケージ名を PEP 503 で正規化する。"""
    return re.sub(r"[-_.]+", "-", name).lower()


def _nonempty_str(value: Any, what: str) -> str:
    """空でない文字列であることを検証する。"""
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"ledger field {what} must be a non-empty string")
    return value


def load_ledger(root: Path) -> dict[str, dict[str, dict[tuple[str, str], dict[str, Any]]]]:
    """台帳を読み込んで検証し、{生態系: {direct|locked: {(name, version): entry}}} で返す。"""
    try:
        raw = json.loads(read_bytes_limited(root, LEDGER_NAME).decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise InputError(f"{LEDGER_NAME} is not valid JSON") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
        raise InputError("ledger schema_version is unknown")
    if set(raw) != {"schema_version", "cargo", "pypi"}:
        raise InputError("ledger top-level keys are invalid")
    out: dict[str, dict[str, dict[tuple[str, str], dict[str, Any]]]] = {}
    for eco in ("cargo", "pypi"):
        section = raw[eco]
        if not isinstance(section, dict) or set(section) != {"direct", "locked"}:
            raise InputError(f"ledger {eco} section is invalid")
        out[eco] = {}
        direct_fields = CARGO_DIRECT_FIELDS if eco == "cargo" else DIRECT_FIELDS
        for kind, fields in (("direct", direct_fields), ("locked", LOCKED_FIELDS)):
            entries = section[kind]
            if not isinstance(entries, list):
                raise InputError(f"ledger {eco}.{kind} must be a list")
            table: dict[tuple[str, str], dict[str, Any]] = {}
            for entry in entries:
                if not isinstance(entry, dict) or set(entry) != fields:
                    raise InputError(f"ledger {eco}.{kind} entry has invalid fields")
                name = _nonempty_str(entry["name"], "name")
                version = _nonempty_str(entry["version"], "version")
                if kind == "direct":
                    for f in ("approved_by", "record", "purpose"):
                        _nonempty_str(entry[f], f)
                    on = _nonempty_str(entry["approved_on"], "approved_on")
                    try:
                        if not DATE_RE.match(on):
                            raise ValueError
                        date.fromisoformat(on)
                    except ValueError as exc:
                        raise InputError("ledger approved_on must be YYYY-MM-DD") from exc
                    layers = entry["layers"]
                    if not isinstance(layers, list) or not layers:
                        raise InputError("ledger layers must be a non-empty list")
                    for layer in layers:
                        _nonempty_str(layer, "layers")
                    if eco == "cargo":
                        feats = entry["features"]
                        if not isinstance(feats, list) or not all(
                            isinstance(f, str) and f for f in feats
                        ):
                            raise InputError("ledger features must be a list of strings")
                        if not isinstance(entry["default_features"], bool):
                            raise InputError("ledger default_features must be a boolean")
                else:
                    basis = _nonempty_str(entry["basis"], "basis")
                    # 承認記録が未確認と明記された basis を承認済みとして通さない（fail-closed）
                    if "未確認" in basis or "要オーナー確認" in basis:
                        raise InputError("ledger basis states the approval record is unconfirmed")
                key = (norm_py(name) if eco == "pypi" else name, version)
                if key in table:
                    raise InputError(f"ledger has a duplicate entry in {eco}.{kind}")
                table[key] = entry
            out[eco][kind] = table
        overlap = set(out[eco]["direct"]) & set(out[eco]["locked"])
        if overlap:
            raise InputError(f"ledger {eco} has entries in both direct and locked")
    return out


def _cargo_workspace_dep(
    name: str, spec: Any, v: list[Violation], member_paths: dict[str, str]
) -> str | None:
    """ルートの 1 依存を検査し、registry 依存なら固定版（`=` なし）を返す。

    path 依存は、ワークスペースメンバーの実在するパスで、かつ依存名がそのメンバーの
    package 名と一致するものに限る（任意の crates/ 配下を内部 crate と装えない）。
    """
    rel = "Cargo.toml"
    if isinstance(spec, str):
        spec = {"version": spec}
    if not isinstance(spec, dict):
        v.append(Violation("pin_violation", "cargo", name, "", rel))
        return None
    if "path" in spec:
        path = spec["path"]
        ok = (
            isinstance(path, str)
            and path.startswith("crates/")
            and ".." not in Path(path).parts
            and not any(k in spec for k in ("git", "registry", "version", "package"))
            and member_paths.get(Path(path).as_posix().rstrip("/")) == name
        )
        if not ok:
            v.append(Violation("forbidden_source", "cargo", name, "", rel))
        return None
    if any(k in spec for k in ("git", "registry", "registry-index", "package")):
        v.append(Violation("forbidden_source", "cargo", name, "", rel))
        return None
    version = spec.get("version")
    m = CARGO_PIN_RE.match(version) if isinstance(version, str) else None
    if m is None:
        v.append(Violation("pin_violation", "cargo", name, str(version or ""), rel))
        return None
    return m.group(1)


def _cargo_feature_config(spec: Any) -> tuple[list[str], bool]:
    """registry 依存の (ソート済み features, default-features) を返す（既定は [] と True）。"""
    if not isinstance(spec, dict):
        return [], True
    feats = spec.get("features", [])
    names = sorted({f for f in feats if isinstance(f, str)}) if isinstance(feats, list) else []
    default = spec.get("default-features", spec.get("default_features", True))
    return names, default is not False


def _member_deps(
    rel: str,
    manifest: dict[str, Any],
    v: list[Violation],
    layer: str,
    usages: list[tuple[str, str, str, bool]],
) -> None:
    """crate の全依存表が `workspace = true` だけであることを検査し、利用先を集める。

    メンバー crate に加え、ルート manifest 自身の `[dependencies]` 等にも適用する。
    機能（features・default-features・optional）のメンバー側での上書きも拒否する（機能構成は
    ルートの宣言と台帳でだけ決める）。利用先は (依存名, manifest, 層, dev のみか) で `usages` へ
    追記し、台帳の layers との照合は呼び出し側が行う（REQ-38・#165）。
    """
    tables: list[tuple[str, dict[str, Any]]] = []
    for t in CARGO_DEP_TABLES:
        if isinstance(manifest.get(t), dict):
            tables.append((t, manifest[t]))
    targets = manifest.get("target")
    if isinstance(targets, dict):
        for cfg in targets.values():
            if isinstance(cfg, dict):
                for t in CARGO_DEP_TABLES:
                    if isinstance(cfg.get(t), dict):
                        tables.append((t, cfg[t]))
    forbidden = (
        "version",
        "git",
        "path",
        "registry",
        "package",
        "features",
        "default-features",
        "default_features",
        "optional",
    )
    for table_name, table in tables:
        for dep, spec in table.items():
            if not (isinstance(spec, dict) and spec.get("workspace") is True):
                v.append(Violation("member_dependency_not_workspace", "cargo", dep, "", rel))
            elif any(k in spec for k in forbidden):
                v.append(Violation("member_dependency_not_workspace", "cargo", dep, "", rel))
            else:
                usages.append((dep, rel, layer, table_name == "dev-dependencies"))


def check_cargo(
    root: Path, ledger: dict[str, dict[tuple[str, str], dict[str, Any]]], v: list[Violation]
) -> None:
    """Rust 側（ルート・メンバー・Cargo.lock）を台帳と照合する。"""
    top = load_toml(root, "Cargo.toml")
    ws = top.get("workspace")
    if not isinstance(ws, dict):
        raise InputError("Cargo.toml has no [workspace]")
    for forbidden in ("patch", "replace"):
        if forbidden in top:
            v.append(Violation("forbidden_source", "cargo", forbidden, "", "Cargo.toml"))
    members = ws.get("members", [])
    if not isinstance(members, list):
        raise InputError("workspace members must be a list")
    usages: list[tuple[str, str, str, bool]] = []
    _member_deps("Cargo.toml", top, v, "root", usages)
    member_names: set[str] = set()
    member_paths: dict[str, str] = {}
    for m in members:
        if not isinstance(m, str) or not m.startswith("crates/") or ".." in Path(m).parts:
            raise InputError("workspace members must be plain paths under crates/")
        rel = f"{m}/Cargo.toml"
        manifest = load_toml(root, rel)
        pkg = manifest.get("package")
        if not isinstance(pkg, dict) or not isinstance(pkg.get("name"), str):
            raise InputError(f"{rel} has no package name")
        member_names.add(pkg["name"])
        member_paths[Path(m).as_posix().rstrip("/")] = pkg["name"]
        _member_deps(rel, manifest, v, Path(m).name, usages)

    manifest_direct: set[tuple[str, str]] = set()
    ext_versions: dict[str, str] = {}
    wdeps = ws.get("dependencies", {})
    if not isinstance(wdeps, dict):
        raise InputError("workspace.dependencies must be a table")
    for name, spec in wdeps.items():
        ver = _cargo_workspace_dep(name, spec, v, member_paths)
        if ver is None:
            continue
        manifest_direct.add((name, ver))
        ext_versions[name] = ver
        entry = ledger["direct"].get((name, ver))
        if entry is None:
            v.append(Violation("unapproved_dependency", "cargo", name, ver, "Cargo.toml"))
        elif _cargo_feature_config(spec) != (
            sorted(set(entry["features"])),
            entry["default_features"],
        ):
            # 同じ版のまま機能を有効化・無効化しても承認記録の更新を要求する
            v.append(Violation("feature_mismatch", "cargo", name, ver, "Cargo.toml"))
    for dep, rel, layer, dev_only in usages:
        ver = ext_versions.get(dep)
        entry = ledger["direct"].get((dep, ver)) if ver is not None else None
        if entry is None:
            continue  # 内部 crate、または台帳未記録（unapproved_dependency で別途報告済み）
        allowed = set(entry["layers"])
        if layer not in allowed and not (dev_only and f"{layer}(dev)" in allowed):
            v.append(Violation("unapproved_layer", "cargo", dep, ver or "", rel))
    for key in ledger["direct"]:
        if key not in manifest_direct:
            v.append(Violation("stale_record", "cargo", key[0], key[1], LEDGER_NAME))

    lock = load_toml(root, "Cargo.lock")
    packages = lock.get("package", [])
    if not isinstance(packages, list):
        raise InputError("Cargo.lock package list is invalid")
    locked_seen: set[tuple[str, str]] = set()
    for p in packages:
        if not isinstance(p, dict) or not isinstance(p.get("name"), str):
            raise InputError("Cargo.lock has an invalid package entry")
        name, version, source = p["name"], str(p.get("version", "")), p.get("source")
        if source is None:
            if name not in member_names:
                v.append(Violation("forbidden_source", "cargo", name, version, "Cargo.lock"))
        elif source != CRATES_IO_SOURCE:
            v.append(Violation("forbidden_source", "cargo", name, version, "Cargo.lock"))
        else:
            key = (name, version)
            locked_seen.add(key)
            if key not in ledger["direct"] and key not in ledger["locked"]:
                v.append(
                    Violation("unapproved_locked_package", "cargo", name, version, "Cargo.lock")
                )
    for key in ledger["locked"]:
        if key not in locked_seen:
            v.append(Violation("stale_record", "cargo", key[0], key[1], LEDGER_NAME))
    # 逆方向: manifest にある承認済みの直接依存が、lock に同じ版で現れること
    for key in sorted(manifest_direct):
        if key not in locked_seen:
            v.append(Violation("missing_in_lock", "cargo", key[0], key[1], "Cargo.lock"))


def _py_requirements(pyproject: dict[str, Any]) -> list[Any]:
    """pyproject の全依存宣言（dependencies・optional・dependency-groups）を平坦化する。"""
    reqs: list[Any] = []
    project = pyproject.get("project", {})
    if isinstance(project, dict):
        reqs.extend(project.get("dependencies", []) or [])
        opt = project.get("optional-dependencies", {}) or {}
        if isinstance(opt, dict):
            for items in opt.values():
                reqs.extend(items or [])
    groups = pyproject.get("dependency-groups", {}) or {}
    if isinstance(groups, dict):
        for items in groups.values():
            reqs.extend(items or [])
    return reqs


def _py_build_requirements(pyproject: dict[str, Any]) -> list[Any]:
    """`[build-system].requires`（ビルド時依存）を返す。uv.lock には現れない。"""
    bs = pyproject.get("build-system", {})
    if isinstance(bs, dict):
        req = bs.get("requires", []) or []
        if isinstance(req, list):
            return list(req)
    return []


def check_pypi(
    root: Path, ledger: dict[str, dict[tuple[str, str], dict[str, Any]]], v: list[Violation]
) -> None:
    """Python 側（pyproject・uv.lock）を台帳と照合する。"""
    rel_py, rel_lock = "trainer/pyproject.toml", "trainer/uv.lock"
    pyproject = load_toml(root, rel_py)
    project = pyproject.get("project")
    if not isinstance(project, dict) or not isinstance(project.get("name"), str):
        raise InputError("pyproject.toml has no project name")
    own = norm_py(project["name"])

    manifest_direct: set[tuple[str, str]] = set()
    for req in _py_requirements(pyproject):
        m = PY_REQ_RE.match(req) if isinstance(req, str) else None
        if m is None:
            v.append(Violation("pin_violation", "pypi", str(req)[:80], "", rel_py))
            continue
        key = (norm_py(m.group(1)), m.group(3))
        manifest_direct.add(key)
        if key not in ledger["direct"]:
            v.append(Violation("unapproved_dependency", "pypi", key[0], key[1], rel_py))
    # ビルド時依存は uv.lock に載らないため、固定と承認記録だけを照合する
    build_direct: set[tuple[str, str]] = set()
    for req in _py_build_requirements(pyproject):
        m = PY_REQ_RE.match(req) if isinstance(req, str) else None
        if m is None:
            v.append(Violation("pin_violation", "pypi", str(req)[:80], "", rel_py))
            continue
        key = (norm_py(m.group(1)), m.group(3))
        build_direct.add(key)
        if key not in ledger["direct"]:
            v.append(Violation("unapproved_dependency", "pypi", key[0], key[1], rel_py))
    for key in ledger["direct"]:
        if key not in manifest_direct and key not in build_direct:
            v.append(Violation("stale_record", "pypi", key[0], key[1], LEDGER_NAME))

    lock = load_toml(root, rel_lock)
    packages = lock.get("package", [])
    if not isinstance(packages, list):
        raise InputError("uv.lock package list is invalid")
    locked_seen: set[tuple[str, str]] = set()
    for p in packages:
        if not isinstance(p, dict) or not isinstance(p.get("name"), str):
            raise InputError("uv.lock has an invalid package entry")
        name, version = norm_py(p["name"]), str(p.get("version", ""))
        source = p.get("source")
        if not isinstance(source, dict):
            raise InputError("uv.lock package has no source")
        if set(source) == {"registry"} and source["registry"] == PYPI_SOURCE:
            key = (name, version)
            locked_seen.add(key)
            if key not in ledger["direct"] and key not in ledger["locked"]:
                v.append(Violation("unapproved_locked_package", "pypi", name, version, rel_lock))
        elif set(source) == {"virtual"} and source["virtual"] == "." and name == own:
            continue
        else:
            v.append(Violation("forbidden_source", "pypi", name, version, rel_lock))
    for key in ledger["locked"]:
        if key not in locked_seen:
            v.append(Violation("stale_record", "pypi", key[0], key[1], LEDGER_NAME))
    # 逆方向: manifest にある承認済みの直接依存が、lock に同じ版で現れること
    for key in sorted(manifest_direct):
        if key not in locked_seen:
            v.append(Violation("missing_in_lock", "pypi", key[0], key[1], rel_lock))


def run(root: Path) -> tuple[int, dict[str, Any]]:
    """照合を実行し、(終了コード, 出力 JSON の辞書) を返す。"""
    try:
        ledger = load_ledger(root)
        violations: list[Violation] = []
        check_cargo(root, ledger["cargo"], violations)
        check_pypi(root, ledger["pypi"], violations)
    except InputError as exc:
        return EXIT_INVALID_INPUT, {
            "status": "invalid_input",
            "code": EXIT_INVALID_INPUT,
            "message": str(exc),
            "violation_count": 0,
            "violations": [],
        }
    shown = [x.as_dict() for x in violations[:MAX_VIOLATIONS]]
    if violations:
        return EXIT_JUDGED_FAIL, {
            "status": "judged_fail",
            "code": EXIT_JUDGED_FAIL,
            "message": "dependency changes without an approval record",
            "violation_count": len(violations),
            "violations_truncated": len(violations) > MAX_VIOLATIONS,
            "violations": shown,
        }
    return EXIT_OK, {
        "status": "ok",
        "code": EXIT_OK,
        "violation_count": 0,
        "violations": [],
    }


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
        try:
            args = parser.parse_args(argv)
        except _ArgumentError as exc:
            # 引数不正も「stdout に JSON 1 つ」の契約に揃え、invalid_input(64) で返す
            payload = {
                "status": "invalid_input",
                "code": EXIT_INVALID_INPUT,
                "message": f"invalid arguments: {exc}",
                "violation_count": 0,
                "violations": [],
            }
            sys.stdout.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
            return EXIT_INVALID_INPUT
        except SystemExit as exc:  # --help（終了コード 0）のみ。usage は argparse が出力済み
            return EXIT_OK if exc.code in (0, None) else EXIT_INVALID_INPUT
        code, payload = run(args.root)
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
