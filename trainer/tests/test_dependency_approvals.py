"""依存承認台帳の照合スクリプト（scripts/check_dependency_approvals.py）のテスト。

REQ-38・TASK-38.3・#165。実リポジトリの manifest・lock・台帳を tmp_path へ複製し、
変異を加えて陰性対照を取る（証拠種別: テストハーネス）。実リポジトリそのものの照合テストは
python-ci（`make py-ci`）経由で全 PR に効き、承認記録のない依存変更を CI で止める。
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "check_dependency_approvals.py"

spec = importlib.util.spec_from_file_location("check_dependency_approvals", SCRIPT)
assert spec is not None
assert spec.loader is not None
mod = importlib.util.module_from_spec(spec)
sys.modules["check_dependency_approvals"] = mod
spec.loader.exec_module(mod)

Capture = pytest.CaptureFixture[str]


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """実リポジトリの照合対象ファイルだけを複製した作業用ルート。"""
    rels = ["Cargo.toml", "Cargo.lock", "dependency-approvals.json"]
    rels += ["trainer/pyproject.toml", "trainer/uv.lock"]
    rels += [str(m.relative_to(REPO)) for m in (REPO / "crates").glob("*/Cargo.toml")]
    for rel in rels:
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, dst)
    return tmp_path


def run(root: Path, capsys: Capture) -> tuple[int, dict[str, Any]]:
    """main を実行し (終了コード, stdout の JSON 1 つ) を返す。"""
    code = mod.main(["--root", str(root)])
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    return code, json.loads(lines[0])


def kinds(payload: dict[str, Any]) -> set[tuple[str, str]]:
    """違反を (kind, name) の集合にする。"""
    return {(x["kind"], x["name"]) for x in payload["violations"]}


def edit(root: Path, rel: str, old: str, new: str) -> None:
    """ファイル中の文字列を 1 か所置換する（対象が無ければテスト自体が誤りなので失敗させる）。"""
    p = root / rel
    text = p.read_text(encoding="utf-8")
    assert old in text
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


def append(root: Path, rel: str, text: str) -> None:
    """ファイル末尾へ追記する。"""
    with (root / rel).open("a", encoding="utf-8") as f:
        f.write(text)


def test_req38_real_repository_is_fully_approved(capsys: Capture) -> None:
    """REQ-38・TASK-38.3: 実リポジトリの依存は全件台帳に承認記録がある。"""
    code, payload = run(REPO, capsys)
    assert (code, payload["status"], payload["violations"]) == (0, "ok", [])


def test_req38_copied_repo_passes(repo: Path, capsys: Capture) -> None:
    """複製した状態は exit 0（以降の陰性対照の基準）。"""
    assert run(repo, capsys)[0] == 0


def test_req38_unapproved_workspace_dependency_fails(repo: Path, capsys: Capture) -> None:
    """台帳に無い crate を [workspace.dependencies] に足すと exit 10。"""
    edit(repo, "Cargo.toml", "[workspace.lints.rust]", 'foo = "=1.0.0"\n\n[workspace.lints.rust]')
    code, payload = run(repo, capsys)
    assert code == 10
    assert ("unapproved_dependency", "foo") in kinds(payload)


def test_req38_version_bump_without_record_fails(repo: Path, capsys: Capture) -> None:
    """Cargo.lock の serde の版だけ変える（cargo update 相当）と exit 10。"""
    edit(
        repo,
        "Cargo.lock",
        'name = "serde"\nversion = "1.0.229"',
        'name = "serde"\nversion = "1.0.230"',
    )
    code, payload = run(repo, capsys)
    assert code == 10
    assert ("unapproved_locked_package", "serde") in kinds(payload)


def test_req38_new_locked_package_fails(repo: Path, capsys: Capture) -> None:
    """Cargo.lock に未記録の registry パッケージを足すと exit 10。"""
    append(
        repo,
        "Cargo.lock",
        '\n[[package]]\nname = "evil"\nversion = "0.1.0"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n',
    )
    code, payload = run(repo, capsys)
    assert code == 10
    assert ("unapproved_locked_package", "evil") in kinds(payload)


def test_req38_git_source_in_lock_fails(repo: Path, capsys: Capture) -> None:
    """crates.io 以外の source（git）は拒否する。"""
    append(
        repo,
        "Cargo.lock",
        '\n[[package]]\nname = "g"\nversion = "0.1.0"\nsource = "git+https://x/y"\n',
    )
    code, payload = run(repo, capsys)
    assert code == 10
    assert ("forbidden_source", "g") in kinds(payload)


@pytest.mark.parametrize("spec_text", ['"^1.0.151"', '">=1.0"', '"1.0.151"'])
def test_req38_range_pin_fails(repo: Path, capsys: Capture, spec_text: str) -> None:
    """範囲指定・`=` 無しの版は完全固定違反で exit 10。"""
    edit(repo, "Cargo.toml", 'serde_json = "=1.0.151"', f"serde_json = {spec_text}")
    code, payload = run(repo, capsys)
    assert code == 10
    assert ("pin_violation", "serde_json") in kinds(payload)


def test_req38_git_workspace_dependency_fails(repo: Path, capsys: Capture) -> None:
    """git 依存は拒否する。"""
    edit(repo, "Cargo.toml", 'sha2 = "=0.11.0"', 'sha2 = { git = "https://x/y" }')
    code, payload = run(repo, capsys)
    assert code == 10
    assert ("forbidden_source", "sha2") in kinds(payload)


def test_req38_member_direct_version_fails(repo: Path, capsys: Capture) -> None:
    """メンバー crate の直接 version 指定は workspace 集約違反で exit 10。"""
    edit(repo, "crates/core/Cargo.toml", "sha2.workspace = true", 'sha2 = "=0.11.0"')
    code, payload = run(repo, capsys)
    assert code == 10
    assert ("member_dependency_not_workspace", "sha2") in kinds(payload)


def test_req38_member_target_table_is_checked(repo: Path, capsys: Capture) -> None:
    """[target.*.dependencies] の直接指定も検査する（workspace = true は許可。guard の rustix）。"""
    append(
        repo, "crates/core/Cargo.toml", "\n[target.'cfg(unix)'.dependencies]\nbar = \"=1.0.0\"\n"
    )
    code, payload = run(repo, capsys)
    assert code == 10
    assert ("member_dependency_not_workspace", "bar") in kinds(payload)


def test_req38_new_workspace_member_needs_no_record(repo: Path, capsys: Capture) -> None:
    """境界値: 内部 crate（lock の source 無し）の追加は台帳更新なしで exit 0。"""
    (repo / "crates/newc").mkdir()
    (repo / "crates/newc/Cargo.toml").write_text(
        '[package]\nname = "fandhe-edge-newc"\nversion = "0.1.0"\n', encoding="utf-8"
    )
    edit(repo, "Cargo.toml", '"crates/guard"]', '"crates/guard", "crates/newc"]')
    append(repo, "Cargo.lock", '\n[[package]]\nname = "fandhe-edge-newc"\nversion = "0.1.0"\n')
    assert run(repo, capsys)[0] == 0


def test_req38_unapproved_python_dependency_fails(repo: Path, capsys: Capture) -> None:
    """pyproject に台帳に無い依存を足すと exit 10。"""
    edit(repo, "trainer/pyproject.toml", '"onnx==1.23.0"]', '"onnx==1.23.0", "requests==2.0.0"]')
    code, payload = run(repo, capsys)
    assert code == 10
    assert ("unapproved_dependency", "requests") in kinds(payload)


@pytest.mark.parametrize("req", ['"requests>=2"', '"requests"', '"requests==2.*"'])
def test_req38_python_unpinned_fails(repo: Path, capsys: Capture, req: str) -> None:
    """範囲指定・版無し・ワイルドカードは拒否する。"""
    edit(repo, "trainer/pyproject.toml", '"onnx==1.23.0"]', f'"onnx==1.23.0", {req}]')
    code, payload = run(repo, capsys)
    assert code == 10
    assert any(k == "pin_violation" for k, _ in kinds(payload))


def test_req38_uv_lock_version_change_fails(repo: Path, capsys: Capture) -> None:
    """uv.lock の numpy の版だけ変えると exit 10。"""
    edit(
        repo,
        "trainer/uv.lock",
        'name = "numpy"\nversion = "2.5.3"',
        'name = "numpy"\nversion = "2.5.4"',
    )
    code, payload = run(repo, capsys)
    assert code == 10
    assert ("unapproved_locked_package", "numpy") in kinds(payload)


def test_req38_uv_lock_non_pypi_source_fails(repo: Path, capsys: Capture) -> None:
    """PyPI 以外の registry・git source は拒否する。"""
    append(
        repo,
        "trainer/uv.lock",
        '\n[[package]]\nname = "x"\nversion = "1"\nsource = { git = "https://x/y" }\n',
    )
    code, payload = run(repo, capsys)
    assert code == 10
    assert ("forbidden_source", "x") in kinds(payload)


def mutate_ledger(repo: Path, fn: Callable[[dict[str, Any]], object]) -> None:
    """台帳を読み、fn で変更して書き戻す。"""
    p = repo / mod.LEDGER_NAME
    ledger = json.loads(p.read_text(encoding="utf-8"))
    fn(ledger)
    p.write_text(json.dumps(ledger), encoding="utf-8")


def test_req38_stale_ledger_record_fails(repo: Path, capsys: Capture) -> None:
    """manifest・lock に無い台帳の記録は exit 10（削除にも記録の更新が要る）。"""
    mutate_ledger(
        repo,
        lambda g: g["cargo"]["locked"].append({"name": "ghost", "version": "1.0.0", "basis": "t"}),
    )
    code, payload = run(repo, capsys)
    assert code == 10
    assert ("stale_record", "ghost") in kinds(payload)


MUTATIONS: dict[str, Callable[[dict[str, Any]], object]] = {
    "schema": lambda g: g.update(schema_version=2),
    "missing-record": lambda g: g["cargo"]["direct"][0].pop("record"),
    "date-format": lambda g: g["cargo"]["direct"][0].update(approved_on="2026/09/27"),
    "date-invalid": lambda g: g["cargo"]["direct"][0].update(approved_on="2026-13-40"),
    "blank-approver": lambda g: g["cargo"]["direct"][0].update(approved_by=" "),
    "extra-field": lambda g: g["cargo"]["direct"][0].update(extra="x"),
    "duplicate": lambda g: g["cargo"]["direct"].append(dict(g["cargo"]["direct"][0])),
    "missing-basis": lambda g: g["cargo"]["locked"][0].pop("basis"),
    "missing-section": lambda g: g.pop("pypi"),
    "unconfirmed-basis": lambda g: g["cargo"]["locked"][0].update(
        basis="個別の承認記録は未確認（要オーナー確認）"
    ),
}


@pytest.mark.parametrize("name", list(MUTATIONS))
def test_req38_malformed_ledger_is_invalid_input(repo: Path, capsys: Capture, name: str) -> None:
    """台帳の形式不正は exit 64（fail-closed）。"""
    mutate_ledger(repo, MUTATIONS[name])
    code, payload = run(repo, capsys)
    assert (code, payload["status"]) == (64, "invalid_input")


def test_req38_broken_json_and_missing_files(repo: Path, capsys: Capture) -> None:
    """台帳の JSON 破損・ファイル欠落は exit 64。"""
    (repo / mod.LEDGER_NAME).write_text("{", encoding="utf-8")
    assert run(repo, capsys)[0] == 64
    (repo / mod.LEDGER_NAME).unlink()
    assert run(repo, capsys)[0] == 64


def test_req38_symlink_and_oversize_rejected(repo: Path, capsys: Capture) -> None:
    """symlink の入力とサイズ上限超過（読み込み前に判定）は exit 64。"""
    (repo / "Cargo.lock").rename(repo / "real.lock")
    (repo / "Cargo.lock").symlink_to(repo / "real.lock")
    assert run(repo, capsys)[0] == 64
    (repo / "Cargo.lock").unlink()
    (repo / "real.lock").rename(repo / "Cargo.lock")
    with (repo / "Cargo.lock").open("ab") as f:
        f.truncate(mod.MAX_FILE_BYTES + 1)
    assert run(repo, capsys)[0] == 64


def test_req38_missing_root_argument_is_invalid_input() -> None:
    """引数不足は invalid_input(64)。"""
    assert mod.main([]) == 64


def test_req38_unexpected_exception_maps_to_70(
    monkeypatch: pytest.MonkeyPatch, capsys: Capture
) -> None:
    """予期しない例外は JSON 1 つと runtime_error(70)。"""

    def boom(_root: Path) -> None:
        raise RuntimeError

    monkeypatch.setattr(mod, "run", boom)
    code = mod.main(["--root", "."])
    payload = json.loads(capsys.readouterr().out)
    assert (code, payload["status"]) == (70, "runtime_error")


def test_req38_script_has_no_network_or_process_modules() -> None:
    """REQ-38: 照合スクリプトは通信・子プロセス系のモジュールを import しない（静的確認）。"""
    src = SCRIPT.read_text(encoding="utf-8")
    for banned in ("socket", "urllib", "http.client", "requests", "subprocess", "os.system"):
        assert f"import {banned}" not in src
        assert f"from {banned}" not in src


def test_req38_root_manifest_dependency_tables_are_checked(repo: Path, capsys: Capture) -> None:
    """ルート manifest 自身の [dependencies]・[dev-dependencies] も workspace 集約違反になる。"""
    append(repo, "Cargo.toml", '\n[dependencies]\nrootdep = "=1.0.0"\n')
    append(repo, "Cargo.toml", '\n[dev-dependencies]\nrootdev = { git = "https://x/y" }\n')
    code, payload = run(repo, capsys)
    assert code == 10
    found = kinds(payload)
    assert ("member_dependency_not_workspace", "rootdep") in found
    assert ("member_dependency_not_workspace", "rootdev") in found


def test_req38_path_dependency_must_match_member(repo: Path, capsys: Capture) -> None:
    """path 依存が実在メンバーのパスと package 名に対応しなければ forbidden_source。"""
    edit(
        repo,
        "Cargo.toml",
        'fandhe-edge-core = { path = "crates/core" }',
        'fandhe-edge-core = { path = "crates/nonexistent" }',
    )
    code, payload = run(repo, capsys)
    assert code == 10
    assert ("forbidden_source", "fandhe-edge-core") in kinds(payload)


def test_req38_path_dependency_name_mismatch_fails(repo: Path, capsys: Capture) -> None:
    """実在するパスでも依存名が別 crate の package 名なら forbidden_source。"""
    edit(
        repo,
        "Cargo.toml",
        'fandhe-edge-core = { path = "crates/core" }',
        'fandhe-edge-core = { path = "crates/data" }',
    )
    code, payload = run(repo, capsys)
    assert code == 10
    assert ("forbidden_source", "fandhe-edge-core") in kinds(payload)


def test_req38_direct_dependency_missing_from_cargo_lock_fails(repo: Path, capsys: Capture) -> None:
    """manifest と台帳にあるが Cargo.lock に無い直接依存は missing_in_lock で exit 10。"""
    text = (repo / "Cargo.lock").read_text(encoding="utf-8")
    start = text.index('[[package]]\nname = "sha2"')
    end = text.index("[[package]]", start + 1)
    (repo / "Cargo.lock").write_text(text[:start] + text[end:], encoding="utf-8")
    code, payload = run(repo, capsys)
    assert code == 10
    assert ("missing_in_lock", "sha2") in kinds(payload)


def test_req38_direct_dependency_missing_from_uv_lock_fails(repo: Path, capsys: Capture) -> None:
    """pyproject と台帳にあるが uv.lock に無い直接依存は missing_in_lock で exit 10。"""
    text = (repo / "trainer/uv.lock").read_text(encoding="utf-8")
    start = text.index('[[package]]\nname = "numpy"')
    end = text.index("[[package]]", start + 1)
    (repo / "trainer/uv.lock").write_text(text[:start] + text[end:], encoding="utf-8")
    code, payload = run(repo, capsys)
    assert code == 10
    assert ("missing_in_lock", "numpy") in kinds(payload)
