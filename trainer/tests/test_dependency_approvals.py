"""依存承認台帳の照合スクリプト（scripts/check_dependency_approvals.py）のテスト。

REQ-38・TASK-38.3・#165。実リポジトリの manifest・lock・台帳を tmp_path へ複製し、
変異を加えて陰性対照を取る（証拠種別: テストハーネス）。実リポジトリそのものの照合テストは
python-ci（`make py-ci`）経由で全 PR に効き、承認記録のない依存変更を CI で止める。
"""

from __future__ import annotations

import importlib.util
import json
import re
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
    "pypi-missing-extras": lambda g: g["pypi"]["direct"][0].pop("extras"),
    "pypi-unnormalized-extras": lambda g: g["pypi"]["direct"][0].update(extras=["CPU"]),
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


def test_req38_missing_root_argument_is_invalid_input(capsys: Capture) -> None:
    """引数不足は stdout に invalid_input の JSON 1 つを出し、stderr は空で exit 64。"""
    assert mod.main([]) == 64
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert (payload["status"], payload["code"]) == ("invalid_input", 64)
    assert captured.out.count("\n") == 1
    assert captured.err == ""


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


@pytest.mark.parametrize("req", ["numpy==1", "numpy==1.0", "numpy==1.0.0.post1", "numpy==1.*"])
def test_req38_python_short_or_loose_pin_is_rejected(req: str) -> None:
    """Python の版は `==x.y.z`（数字 3 組）のみ完全固定として受理する。"""
    assert mod.PY_REQ_RE.match(req) is None


def test_req38_python_full_pin_is_accepted() -> None:
    """`==x.y.z` と extras 付きは受理する。"""
    assert mod.PY_REQ_RE.match("numpy==2.5.3") is not None
    assert mod.PY_REQ_RE.match("mlx[cpu]==0.32.2") is not None


def test_req38_unapproved_layer_for_existing_dependency_fails(repo: Path, capsys: Capture) -> None:
    """台帳の layers に無い層（crates/eval）で承認済み依存を使うと exit 10（REQ-38・#165）。"""
    edit(repo, "crates/eval/Cargo.toml", "[lints]", "serde.workspace = true\n\n[lints]")
    code, payload = run(repo, capsys)
    assert code == 10
    assert ("unapproved_layer", "serde") in kinds(payload)


def test_req38_dev_only_layer_needs_dev_record(repo: Path, capsys: Capture) -> None:
    """境界値: 通常依存への昇格は `<層>(dev)` 記録だけでは通らない（runtime）。"""
    edit(
        repo,
        "crates/runtime/Cargo.toml",
        "[dev-dependencies]",
        "serde_json.workspace = true\n\n[dev-dependencies]",
    )
    # 重複キーを避けるため dev 側の既存記述を除去する
    text = (repo / "crates/runtime/Cargo.toml").read_text(encoding="utf-8")
    head, _, tail = text.partition("[dev-dependencies]")
    tail = tail.replace("serde_json.workspace = true\n", "", 1)
    (repo / "crates/runtime/Cargo.toml").write_text(
        head + "[dev-dependencies]" + tail, encoding="utf-8"
    )
    code, payload = run(repo, capsys)
    assert code == 10
    assert ("unapproved_layer", "serde_json") in kinds(payload)


def test_req38_workspace_feature_change_fails(repo: Path, capsys: Capture) -> None:
    """同じ版のまま features を増やすと、台帳の features と不一致で exit 10。"""
    edit(
        repo,
        "Cargo.toml",
        'serde_json = "=1.0.151"',
        'serde_json = { version = "=1.0.151", features = ["preserve_order"] }',
    )
    code, payload = run(repo, capsys)
    assert code == 10
    assert ("feature_mismatch", "serde_json") in kinds(payload)


def test_req38_default_features_change_fails(repo: Path, capsys: Capture) -> None:
    """default-features の有効化（rustix は台帳で false）も exit 10。"""
    edit(repo, "Cargo.toml", "default-features = false, ", "")
    code, payload = run(repo, capsys)
    assert code == 10
    assert ("feature_mismatch", "rustix") in kinds(payload)


def test_req38_member_feature_override_fails(repo: Path, capsys: Capture) -> None:
    """メンバー側の features 上書きは exit 10。"""
    edit(
        repo,
        "crates/core/Cargo.toml",
        "sha2.workspace = true",
        'sha2 = { workspace = true, features = ["oid"] }',
    )
    code, payload = run(repo, capsys)
    assert code == 10
    assert ("member_dependency_not_workspace", "sha2") in kinds(payload)


def test_req38_build_system_requires_is_checked(repo: Path, capsys: Capture) -> None:
    """build-system.requires の未承認・未固定の依存も exit 10。"""
    append(
        repo,
        "trainer/pyproject.toml",
        '\n[build-system]\nrequires = ["hatchling==1.0.0", "setuptools>=1"]\n',
    )
    code, payload = run(repo, capsys)
    assert code == 10
    found = kinds(payload)
    assert ("unapproved_dependency", "hatchling") in found
    assert any(k == "pin_violation" for k, _ in found)


def test_req38_python_extras_removed_fails(repo: Path, capsys: Capture) -> None:
    """承認済みの extras を外す（`mlx[cpu]` -> `mlx`）と exit 10（REQ-38）。"""
    edit(repo, "trainer/pyproject.toml", '"mlx[cpu]==0.32.2"', '"mlx==0.32.2"')
    code, payload = run(repo, capsys)
    assert code == 10
    assert ("extras_mismatch", "mlx") in kinds(payload)


def test_req38_python_extras_added_fails(repo: Path, capsys: Capture) -> None:
    """extras の無い承認済み依存へ extras を足す（`numpy` -> `numpy[x]`）と exit 10（REQ-38）。"""
    edit(repo, "trainer/pyproject.toml", '"numpy==2.5.3"', '"numpy[extra]==2.5.3"')
    code, payload = run(repo, capsys)
    assert code == 10
    assert ("extras_mismatch", "numpy") in kinds(payload)


def test_req38_python_extras_changed_fails(repo: Path, capsys: Capture) -> None:
    """extras を別のものへ変える・追加すると exit 10（REQ-38）。"""
    edit(repo, "trainer/pyproject.toml", "mlx[cpu]==0.32.2", "mlx[cuda]==0.32.2")
    code, payload = run(repo, capsys)
    assert code == 10
    assert ("extras_mismatch", "mlx") in kinds(payload)
    edit(repo, "trainer/pyproject.toml", "mlx[cuda]==0.32.2", "mlx[cpu,cuda]==0.32.2")
    code, payload = run(repo, capsys)
    assert ("extras_mismatch", "mlx") in kinds(payload)


def test_req38_python_extras_normalization_is_tolerated(repo: Path, capsys: Capture) -> None:
    """extras の大文字小文字・空白・順序・区切りの差は同一として通る（PEP 685）。"""
    edit(repo, "trainer/pyproject.toml", "mlx[cpu]==0.32.2", "mlx[ CPU ]==0.32.2")
    code, payload = run(repo, capsys)
    assert code == 0, payload
    assert mod.norm_extras("B_x, a.y,A-Y") == ["a-y", "b-x"]


@pytest.mark.parametrize(
    ("anchor", "old", "new"),
    [
        ("tool.uv", "package = false", 'package = false\ndev-dependencies = ["requests==2.0.0"]'),
        ("group", '"pytest==9.1.1"]', '"pytest==9.1.1", "requests==2.0.0"]'),
        (
            "group-marker",
            '"pytest==9.1.1"]',
            '"pytest==9.1.1", "requests==2.0.0; os_name == \'x\'"]',
        ),
    ],
)
def test_req38_python_dev_dependency_sections_are_checked(
    repo: Path, capsys: Capture, anchor: str, old: str, new: str
) -> None:
    """`[tool.uv].dev-dependencies`・`[dependency-groups]` の未承認依存も exit 10（REQ-38）。"""
    edit(repo, "trainer/pyproject.toml", old, new)
    code, payload = run(repo, capsys)
    assert code == 10
    assert ("unapproved_dependency", "requests") in kinds(payload)


def test_req38_python_optional_dependencies_are_checked(repo: Path, capsys: Capture) -> None:
    """`[project.optional-dependencies]` の未承認依存も exit 10（REQ-38）。"""
    edit(
        repo,
        "trainer/pyproject.toml",
        "[dependency-groups]",
        '[project.optional-dependencies]\nx = ["requests==2.0.0"]\n\n[dependency-groups]',
    )
    code, payload = run(repo, capsys)
    assert code == 10
    assert ("unapproved_dependency", "requests") in kinds(payload)


def test_req38_python_include_group_is_understood(repo: Path, capsys: Capture) -> None:
    """`include-group` は実在する参照先なら通り、未知の参照先・未知の形式は exit 64（REQ-38）。"""
    edit(
        repo,
        "trainer/pyproject.toml",
        "[tool.uv]",
        "[dependency-groups.all]\nx = 1\n\n[tool.uv]",
    )
    assert run(repo, capsys)[0] == 64
    edit(repo, "trainer/pyproject.toml", "[dependency-groups.all]\nx = 1\n", "")
    edit(
        repo,
        "trainer/pyproject.toml",
        'dev = ["ruff==0.16.9", "pytest==9.1.1"]',
        'dev = ["ruff==0.16.9", "pytest==9.1.1", {include-group = "nope"}]',
    )
    assert run(repo, capsys)[0] == 64
    edit(repo, "trainer/pyproject.toml", '"nope"', '"dev"')
    assert run(repo, capsys)[0] == 0


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("package = false", 'package = false\noverride-dependencies = ["numpy==9.9.9"]'),
        ("package = false", 'package = false\nindex = [{ url = "https://example.invalid" }]'),
        ('dependencies = ["mlx', 'dynamic = ["dependencies"]\ndependencies = ["mlx'),
    ],
)
def test_req38_python_unverifiable_sections_are_invalid_input(
    repo: Path, capsys: Capture, old: str, new: str
) -> None:
    """未知の `[tool.uv]` キー・動的依存は黙って読み飛ばさず exit 64（fail-closed。REQ-38）。"""
    edit(repo, "trainer/pyproject.toml", old, new)
    assert run(repo, capsys)[0] == 64


@pytest.mark.parametrize("table", ["dev-dependencies", "build-dependencies"])
def test_req38_member_dev_and_build_dependencies_are_checked(
    repo: Path, capsys: Capture, table: str
) -> None:
    """dev / build の依存表と target 別の dev 表も直接指定を拒否する（REQ-38）。"""
    append(repo, "crates/core/Cargo.toml", f'\n[{table}]\nbar = "=1.0.0"\n')
    code, payload = run(repo, capsys)
    assert code == 10
    assert ("member_dependency_not_workspace", "bar") in kinds(payload)


def test_req38_member_target_dev_dependencies_are_checked(repo: Path, capsys: Capture) -> None:
    """[target.*.dev-dependencies] の直接指定も拒否する（REQ-38）。"""
    append(
        repo,
        "crates/core/Cargo.toml",
        "\n[target.'cfg(unix)'.dev-dependencies]\nbar = \"=1.0.0\"\n",
    )
    code, payload = run(repo, capsys)
    assert code == 10
    assert ("member_dependency_not_workspace", "bar") in kinds(payload)


def test_req38_python_layer_not_in_ledger_fails(repo: Path, capsys: Capture) -> None:
    """台帳 layers に trainer が無い依存は、名前・版・extras が一致しても exit 10（REQ-38）。"""
    mutate_ledger(repo, lambda g: g["pypi"]["direct"][1].update(layers=["core"]))
    code, payload = run(repo, capsys)
    assert code == 10
    assert ("unapproved_layer", "numpy") in kinds(payload)


def test_req38_python_dev_layer_only_covers_dev_usage(repo: Path, capsys: Capture) -> None:
    """`trainer(dev)` は dev 区分の利用だけを許す。本番依存 numpy への利用は exit 10（REQ-38）。"""

    def to_dev(g: dict[str, Any]) -> None:
        for e in g["pypi"]["direct"]:
            if e["name"] in ("ruff", "numpy"):
                e["layers"] = ["trainer(dev)"]

    mutate_ledger(repo, to_dev)
    code, payload = run(repo, capsys)
    assert code == 10
    assert kinds(payload) == {("unapproved_layer", "numpy")}


def test_req38_python_build_requires_needs_build_layer(repo: Path, capsys: Capture) -> None:
    """build-system の利用は `trainer(dev)` では許されず、`trainer(build)` なら通る（REQ-38）。"""
    append(repo, "trainer/pyproject.toml", '\n[build-system]\nrequires = ["ruff==0.16.9"]\n')
    mutate_ledger(
        repo,
        lambda g: next(e for e in g["pypi"]["direct"] if e["name"] == "ruff").update(
            layers=["trainer(dev)"]
        ),
    )
    code, payload = run(repo, capsys)
    assert code == 10
    assert ("unapproved_layer", "ruff") in kinds(payload)
    mutate_ledger(
        repo,
        lambda g: next(e for e in g["pypi"]["direct"] if e["name"] == "ruff").update(
            layers=["trainer(dev)", "trainer(build)"]
        ),
    )
    assert run(repo, capsys)[0] == 0


@pytest.mark.parametrize(
    "spec",
    [
        '{ version = "=1.0.151", features = "preserve_order" }',
        '{ version = "=1.0.151", features = ["a", 1] }',
        '{ version = "=1.0.151", features = 1 }',
        '{ version = "=1.0.151", default-features = "false" }',
        '{ version = "=1.0.151", default-features = 0 }',
        '{ version = "=1.0.151", default-features = false, default_features = false }',
    ],
)
def test_req38_cargo_malformed_feature_config_is_invalid_input(
    repo: Path, capsys: Capture, spec: str
) -> None:
    """features・default-features の型不正は黙って読み替えず exit 64（REQ-38）。"""
    edit(repo, "Cargo.toml", 'serde_json = "=1.0.151"', f"serde_json = {spec}")
    assert run(repo, capsys)[0] == 64


@pytest.mark.parametrize(
    "text",
    [
        "\n[build-system]\nrequires = 1\n",
        '\n[build-system]\nrequires = ""\n',
        '\n[build-system]\nbuild-backend = "x"\n',
    ],
)
def test_req38_python_malformed_build_system_is_invalid_input(
    repo: Path, capsys: Capture, text: str
) -> None:
    """`[build-system]` の `requires` が無い / リストでないと exit 64（REQ-38）。"""
    append(repo, "trainer/pyproject.toml", text)
    assert run(repo, capsys)[0] == 64


def prepend(root: Path, rel: str, text: str) -> None:
    """ファイル先頭（トップレベル）へ追記する。"""
    p = root / rel
    p.write_text(text + p.read_text(encoding="utf-8"), encoding="utf-8")


def test_req38_python_build_system_not_a_table_is_invalid_input(
    repo: Path, capsys: Capture
) -> None:
    """`build-system` が表でないと exit 64（REQ-38）。"""
    prepend(repo, "trainer/pyproject.toml", "build-system = 1\n")
    assert run(repo, capsys)[0] == 64


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ('dependencies = ["mlx', 'dynamic = "dependencies"\ndependencies = ["mlx'),
        ('dependencies = ["mlx', 'dynamic = [1]\ndependencies = ["mlx'),
        ('dependencies = ["mlx', 'optional-dependencies = ""\ndependencies = ["mlx'),
        ("[tool.uv]", "[tool]\nuv = 1\n[tool.x]"),
    ],
)
def test_req38_python_wrongly_typed_sections_are_invalid_input(
    repo: Path, capsys: Capture, old: str, new: str
) -> None:
    """pyproject の各セクションの型不正は空扱いにせず exit 64（REQ-38）。"""
    p = repo / "trainer/pyproject.toml"
    text = p.read_text(encoding="utf-8")
    assert old in text
    p.write_text(text.replace(old, new, 1), encoding="utf-8")
    assert run(repo, capsys)[0] == 64


@pytest.mark.parametrize("target", ["crates/core/Cargo.toml", "Cargo.toml"])
@pytest.mark.parametrize(
    "text",
    [
        "dev-dependencies = 1\n",
        "build-dependencies = []\n",
        "target = 1\n",
        "target = { x = 1 }\n",
        "target = { x = { dependencies = 1 } }\n",
    ],
)
def test_req38_cargo_wrongly_typed_dependency_tables_are_invalid_input(
    repo: Path, capsys: Capture, target: str, text: str
) -> None:
    """依存表・target の型不正は読み飛ばさず exit 64（REQ-38）。"""
    prepend(repo, target, text)
    assert run(repo, capsys)[0] == 64


@pytest.mark.parametrize("lock", ["Cargo.lock", "trainer/uv.lock"])
def test_req38_lock_without_package_list_or_version_is_invalid_input(
    repo: Path, capsys: Capture, lock: str
) -> None:
    """lock に package 一覧が無い・version が無い / 文字列でないと exit 64（REQ-38）。"""
    p = repo / lock
    original = p.read_text(encoding="utf-8")
    p.write_text("version = 1\n", encoding="utf-8")
    assert run(repo, capsys)[0] == 64
    head, sep, rest = original.partition("[[package]]\n")
    no_version = re.sub(r"(?m)^version = .*\n", "", rest, count=1)
    int_version = re.sub(r"(?m)^version = .*\n", "version = 1\n", rest, count=1)
    for body in (no_version, int_version):
        p.write_text(head + sep + body, encoding="utf-8")
        assert run(repo, capsys)[0] == 64


def test_req38_python_dependency_groups_not_a_table_is_invalid_input(
    repo: Path, capsys: Capture
) -> None:
    """トップレベルの `dependency-groups` が表でないと exit 64（REQ-38）。"""
    prepend(repo, "trainer/pyproject.toml", "dependency-groups = 0\n")
    assert run(repo, capsys)[0] == 64


@pytest.mark.parametrize("req", ['"numpy[]==2.5.3"', '"numpy[a,]==2.5.3"', '"numpy[,a]==2.5.3"'])
def test_req38_python_malformed_extras_is_rejected(repo: Path, capsys: Capture, req: str) -> None:
    """空の extras 要素は黙って捨てず、解釈できない要求として exit 10（REQ-38）。"""
    edit(repo, "trainer/pyproject.toml", '"numpy==2.5.3"', req)
    code, payload = run(repo, capsys)
    assert code == 10
    assert any(k == "pin_violation" for k, _ in kinds(payload))


def test_req38_trailing_newline_pin_is_rejected() -> None:
    """末尾の改行付きの要求・固定版は完全一致でないため受理しない（REQ-38）。"""
    assert mod.PY_REQ_RE.fullmatch("numpy==2.5.3\n") is None
    assert mod.CARGO_PIN_RE.fullmatch("=1.0.0\n") is None


@pytest.mark.parametrize("pin", ["=1.2.3-alpha", "=1.2.3+meta", "=1.2.3-rc.1+b", "=1.2", "1.2.3"])
def test_req38_cargo_pin_only_accepts_plain_x_y_z(pin: str) -> None:
    """Cargo の固定は `=x.y.z` のみ。pre-release・build metadata などは拒否する（REQ-38）。"""
    assert mod.CARGO_PIN_RE.fullmatch(pin) is None
    assert mod.CARGO_PIN_RE.fullmatch("=1.2.3") is not None


@pytest.mark.parametrize("pin", ["=1.0.151-alpha", "=1.0.151+meta"])
def test_req38_cargo_prerelease_pin_in_manifest_fails(
    repo: Path, capsys: Capture, pin: str
) -> None:
    """manifest の `=x.y.z-pre` / `=x.y.z+meta` は pin_violation で exit 10（REQ-38）。"""
    edit(repo, "Cargo.toml", 'serde_json = "=1.0.151"', f'serde_json = "{pin}"')
    code, payload = run(repo, capsys)
    assert code == 10
    assert ("pin_violation", "serde_json") in kinds(payload)


@pytest.mark.parametrize(
    "req",
    [
        "numpy==2.5.3rc1",
        "numpy==2.5.3.post1",
        "numpy==2.5.3.dev0",
        "numpy==2.5.3+local",
        "numpy==2.5.*",
    ],
)
def test_req38_python_pre_post_dev_local_and_wildcard_pins_are_rejected(req: str) -> None:
    """Python の固定は `==x.y.z` のみ（pre・post・dev・local・ワイルドカードは拒否。REQ-38）。"""
    assert mod.PY_REQ_RE.fullmatch(req) is None
