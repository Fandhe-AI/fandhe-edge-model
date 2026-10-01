"""unsafe 許可リストの照合スクリプト（scripts/check_unsafe_allowlist.py）のテスト。

REQ-39・#335。実リポジトリの照合対象を tmp_path へ複製し、変異を加えて陰性対照を取る
（証拠種別: テストハーネス）。実リポジトリそのものの照合テストは python-ci（`make py-ci`）
経由で全 PR に効き、承認記録のない `allow(unsafe_code)` の追加を CI で止める。
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "check_unsafe_allowlist.py"

spec = importlib.util.spec_from_file_location("check_unsafe_allowlist", SCRIPT)
assert spec is not None
assert spec.loader is not None
mod = importlib.util.module_from_spec(spec)
sys.modules["check_unsafe_allowlist"] = mod
spec.loader.exec_module(mod)

Capture = pytest.CaptureFixture[str]
GUARD = "crates/guard/src/resource.rs"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """実リポジトリの crates/（.rs と Cargo.toml）・ルート Cargo.toml・許可リストの複製。"""
    for rel in ["Cargo.toml", "unsafe-allowlist.json"]:
        shutil.copy(REPO / rel, tmp_path / rel)
    for src in (REPO / "crates").rglob("*"):
        if "target" in src.parts or not src.is_file():
            continue
        if src.suffix == ".rs" or src.name == "Cargo.toml":
            dst = tmp_path / src.relative_to(REPO)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
    return tmp_path


def run(root: Path, capsys: Capture) -> tuple[int, dict[str, Any]]:
    """main を実行し (終了コード, stdout の JSON 1 つ) を返す。"""
    code = mod.main(["--root", str(root)])
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    return code, json.loads(lines[0])


def kinds(payload: dict[str, Any]) -> set[tuple[str, str]]:
    """違反を (kind, item) の集合にする。"""
    return {(x["kind"], x["item"]) for x in payload["violations"]}


def append(root: Path, rel: str, text: str) -> None:
    """ファイル末尾へ追記する。"""
    p = root / rel
    p.write_text(p.read_text() + text)


def set_ledger(root: Path, fn: Any) -> None:
    """許可リスト JSON を読み、fn で変異させて書き戻す。"""
    p = root / "unsafe-allowlist.json"
    data = json.loads(p.read_text())
    fn(data)
    p.write_text(json.dumps(data))


def test_real_repository_matches_allowlist(capsys: Capture) -> None:
    """REQ-39・#335: 現在の出現箇所（darwin::pidinfo の 1 件）は許可リストと一致する。"""
    code, payload = run(REPO, capsys)
    assert (code, payload["status"], payload["violation_count"]) == (0, "ok", 0)


def test_new_file_allow_is_rejected(repo: Path, capsys: Capture) -> None:
    """未承認のファイルへの追加は unlisted_allow（exit 10）。"""
    (repo / "crates/core/src/extra.rs").write_text("#[allow(unsafe_code)]\nfn x() {}\n")
    code, payload = run(repo, capsys)
    assert code == 10
    assert ("unlisted_allow", "x") in kinds(payload)


def test_other_function_in_listed_file_is_rejected(repo: Path, capsys: Capture) -> None:
    """許可済みファイルでも別 item への追加は unlisted_allow。"""
    append(repo, GUARD, "\n#[allow(unsafe_code)]\nfn other() {}\n")
    code, payload = run(repo, capsys)
    assert code == 10
    assert ("unlisted_allow", "other") in kinds(payload)


def test_same_item_second_occurrence_exceeds_count(repo: Path, capsys: Capture) -> None:
    """同一 item の 2 つ目は count_exceeded。"""
    p = repo / GUARD
    text = p.read_text()
    p.write_text(
        text.replace(
            "    #[allow(unsafe_code)]\n    fn pidinfo",
            "    #[allow(unsafe_code)]\n    #[allow(unsafe_code)]\n    fn pidinfo",
            1,
        )
    )
    code, payload = run(repo, capsys)
    assert code == 10
    assert ("count_exceeded", "darwin::pidinfo") in kinds(payload)


@pytest.mark.parametrize(
    ("snippet", "item"),
    [
        ("#![allow(unsafe_code)]\n", "<inner>"),
        ("#[allow(dead_code, unsafe_code)]\nfn f() {}\n", "f"),
        ("#[allow(\n    dead_code,\n    unsafe_code,\n)]\nfn f() {}\n", "f"),
        ('#[cfg_attr(target_os = "macos", allow(unsafe_code))]\npub(crate) fn f() {}\n', "f"),
        ("#[expect(unsafe_code)]\nconst fn f() {}\n", "f"),
        ("#[warn(unsafe_code)]\n#[inline]\nunsafe fn f() {}\n", "f"),
        (
            "mod a {\n    pub mod b {\n        #[allow(unsafe_code)]\n"
            "        fn f() {}\n    }\n}\n",
            "a::b::f",
        ),
    ],
)
def test_attribute_forms_are_detected(repo: Path, capsys: Capture, snippet: str, item: str) -> None:
    """各種の属性形（内部属性・複数行・cfg_attr・expect・warn・入れ子 mod）を検出する。"""
    (repo / "crates/core/src/extra.rs").write_text(snippet)
    code, payload = run(repo, capsys)
    assert code == 10
    assert ("unlisted_allow", item) in kinds(payload)


@pytest.mark.parametrize("rel", ["crates/core/tests/t.rs", "crates/core/build.rs"])
def test_tests_and_build_script_are_scanned(repo: Path, capsys: Capture, rel: str) -> None:
    """tests/・build.rs への追加も検出する。"""
    (repo / rel).parent.mkdir(parents=True, exist_ok=True)
    (repo / rel).write_text("#[allow(unsafe_code)]\nfn t() {}\n")
    code, payload = run(repo, capsys)
    assert code == 10
    assert ("unlisted_allow", "t") in kinds(payload)


def test_mentions_and_deny_are_not_flagged(repo: Path, capsys: Capture) -> None:
    """コメント・文字列・raw 文字列中の言及と deny / forbid は誤検出しない。"""
    src = (
        "//! `#[allow(unsafe_code)]` の説明\n"
        "/// #[allow(unsafe_code)]\n"
        "/* #[allow(unsafe_code)] /* nested */ */\n"
        "#![forbid(unsafe_code)]\n#[deny(unsafe_code)]\nfn a<'a>(x: &'a str) -> char {\n"
        '    let _s = "#[allow(unsafe_code)]";\n'
        '    let _r = r#"#[allow(unsafe_code)] "quoted""#;\n'
        "    let _c = '\"';\n    'x'\n}\n"
    )
    (repo / "crates/core/src/extra.rs").write_text(src)
    code, payload = run(repo, capsys)
    assert (code, payload["violation_count"]) == (0, 0)


def test_stale_entry_is_rejected(repo: Path, capsys: Capture) -> None:
    """許可リストにあって出現の無い記録は stale_entry。"""

    def mutate(d: dict[str, Any]) -> None:
        d["entries"][0]["item"] = "darwin::gone"

    set_ledger(repo, mutate)
    code, payload = run(repo, capsys)
    assert code == 10
    assert ("stale_entry", "darwin::gone") in kinds(payload)
    assert ("unlisted_allow", "darwin::pidinfo") in kinds(payload)


def test_removed_entry_is_rejected(repo: Path, capsys: Capture) -> None:
    """記録の削除は unlisted_allow。"""
    set_ledger(repo, lambda d: d["entries"].clear())
    code, payload = run(repo, capsys)
    assert code == 10
    assert ("unlisted_allow", "darwin::pidinfo") in kinds(payload)


def test_member_lints_override_is_rejected(repo: Path, capsys: Capture) -> None:
    """メンバー crate の manifest で unsafe_code を緩めると lints_override。"""
    p = repo / "crates/core/Cargo.toml"
    p.write_text(p.read_text() + '\n[lints.rust]\nunsafe_code = "allow"\n')
    code, payload = run(repo, capsys)
    assert code == 10
    assert any(x["kind"] == "lints_override" for x in payload["violations"])


def test_root_level_relaxed_is_rejected(repo: Path, capsys: Capture) -> None:
    """ルートの deny を緩めると lint_level_relaxed。"""
    p = repo / "Cargo.toml"
    p.write_text(p.read_text().replace('unsafe_code = "deny"', 'unsafe_code = "warn"'))
    code, payload = run(repo, capsys)
    assert code == 10
    assert any(x["kind"] == "lint_level_relaxed" for x in payload["violations"])


def test_cargo_config_mention_is_rejected(repo: Path, capsys: Capture) -> None:
    """.cargo/config.toml に unsafe_code が現れたら cargo_config_lint。"""
    (repo / ".cargo").mkdir()
    (repo / ".cargo/config.toml").write_text('[build]\nrustflags = ["-A", "unsafe_code"]\n')
    code, payload = run(repo, capsys)
    assert code == 10
    assert any(x["kind"] == "cargo_config_lint" for x in payload["violations"])


def test_unparsable_attribute_fails_closed(repo: Path, capsys: Capture) -> None:
    """レベル語の無い unsafe_code の属性は unparsed_attribute。"""
    (repo / "crates/core/src/extra.rs").write_text("#[some_tool(unsafe_code)]\nfn f() {}\n")
    code, payload = run(repo, capsys)
    assert code == 10
    assert any(x["kind"] == "unparsed_attribute" for x in payload["violations"])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(schema_version=2),
        lambda d: d.update(extra=1),
        lambda d: d["entries"][0].update(unknown=1),
        lambda d: d["entries"][0].pop("record"),
        lambda d: d["entries"][0].update(approved_on="2026/10/01"),
        lambda d: d["entries"][0].update(approved_on="2026-13-45"),
        lambda d: d["entries"][0].update(count=0),
        lambda d: d["entries"][0].pop("body_sha256"),
        lambda d: d["entries"][0].update(body_sha256="XYZ"),
        lambda d: d["entries"][0].update(level="deny"),
        lambda d: d["entries"][0].update(file="../x.rs"),
        lambda d: d["entries"][0].update(file="/abs/x.rs"),
        lambda d: d["entries"].append(dict(d["entries"][0])),
    ],
)
def test_malformed_allowlist_is_invalid_input(repo: Path, capsys: Capture, mutate: Any) -> None:
    """許可リストの形式不正は invalid_input（64）。"""
    set_ledger(repo, mutate)
    code, payload = run(repo, capsys)
    assert (code, payload["status"]) == (64, "invalid_input")


def test_missing_allowlist_and_bad_args(repo: Path, capsys: Capture) -> None:
    """許可リスト欠落・不正な引数は 64。"""
    (repo / "unsafe-allowlist.json").unlink()
    code, _ = run(repo, capsys)
    assert code == 64
    assert mod.main([]) == 64
    capsys.readouterr()


def test_symlinked_source_is_rejected(repo: Path, capsys: Capture) -> None:
    """symlink の .rs は読まずに invalid_input（経路の閉じ込め。REQ-39）。"""
    (repo / "crates/core/src/link.rs").symlink_to(repo / GUARD)
    code, _ = run(repo, capsys)
    assert code == 64


def test_symlinked_source_directory_is_rejected(repo: Path, capsys: Capture) -> None:
    """crates/ 配下の symlink ディレクトリ（src/ 差し替え含む）は invalid_input（REQ-39・#335）。"""
    real = repo / "crates/core/src"
    moved = repo / "crates/core/src_real"
    real.rename(moved)
    real.symlink_to(moved, target_is_directory=True)
    code, payload = run(repo, capsys)
    assert (code, payload["status"]) == (64, "invalid_input")


def test_unsafe_block_added_in_approved_item_is_detected(repo: Path, capsys: Capture) -> None:
    """承認済み item の本体に unsafe を足すと件数が同じでも body_changed（REQ-39・#335）。"""
    p = repo / GUARD
    text = p.read_text()
    needle = "    fn pidinfo(pid: c_int, flavor: Flavor) -> Option<Info> {\n"
    assert needle in text
    p.write_text(text.replace(needle, needle + "        let _x = unsafe { 0u8 };\n", 1))
    code, payload = run(repo, capsys)
    assert code == 10
    assert ("body_changed", "darwin::pidinfo") in kinds(payload)


def test_print_hashes_matches_allowlist(repo: Path, capsys: Capture) -> None:
    """--print-hashes の出力は許可リストの body_sha256 と一致する。"""
    code = mod.main(["--root", str(repo), "--print-hashes"])
    payload = json.loads(capsys.readouterr().out)
    ledger = json.loads((repo / "unsafe-allowlist.json").read_text())
    assert code == 0
    assert [o["body_sha256"] for o in payload["occurrences"]] == [
        e["body_sha256"] for e in ledger["entries"]
    ]


def test_unexpected_exception_is_runtime_error(
    monkeypatch: pytest.MonkeyPatch, capsys: Capture
) -> None:
    """予期しない例外は runtime_error（70）。"""

    def boom(_: Path) -> None:
        raise RuntimeError("x")

    monkeypatch.setattr(mod, "run", boom)
    assert mod.main(["--root", str(REPO)]) == 70
    capsys.readouterr()


def test_mask_preserves_layout() -> None:
    """マスクは行数と各行の長さ（インデント）を保つ。"""
    src = '    let a = "x\\"y"; // c\n    /* a\n b */ fn f<\'a>() {}\n'
    masked = mod.mask_rust(src)
    assert [len(x) for x in masked.split("\n")] == [len(x) for x in src.split("\n")]
    assert "fn f<'a>() {}" in masked
    assert "x" not in masked.split("\n")[0].split("=")[1]


def test_cfg_attr_deny_after_other_lint_is_not_flagged(repo: Path, capsys: Capture) -> None:
    """cfg_attr の複数 lint で unsafe_code が deny に属すなら誤検出しない（REQ-39・#335）。"""
    src = (
        '#[cfg_attr(feature = "x", allow(dead_code), deny(unsafe_code))]\n'
        "fn a() {}\n"
        "#[cfg_attr(all(), allow(dead_code), forbid(unsafe_code))]\nfn b() {}\n"
    )
    (repo / "crates/core/src/extra.rs").write_text(src)
    code, payload = run(repo, capsys)
    assert (code, payload["violation_count"]) == (0, 0)


def test_cfg_attr_allow_with_other_lints_is_detected(repo: Path, capsys: Capture) -> None:
    """複数 lint 指定の allow(dead_code, unsafe_code) は allow として検出する（REQ-39・#335）。"""
    (repo / "crates/core/src/extra.rs").write_text(
        '#[cfg_attr(feature = "x", deny(dead_code), allow(dead_code, unsafe_code))]\nfn a() {}\n'
    )
    code, payload = run(repo, capsys)
    assert code == 10
    assert ("unlisted_allow", "a") in kinds(payload)


def test_const_generic_brace_in_header_is_included_in_body_hash(repo: Path) -> None:
    """ヘッダの const generic の `{}` で本体が途切れず、本体の変更を検出する（REQ-39・#335）。"""
    src = "#[allow(unsafe_code)]\nfn g<const N: usize = { 1 + 1 }>() {\n    let _a = 0;\n}\n"
    target = repo / "crates/core/src/extra.rs"
    target.write_text(src)
    _, before = mod.print_hashes(repo)
    target.write_text(src.replace("_a = 0", "_a = 1"))
    _, after = mod.print_hashes(repo)

    def digest(payload: dict[str, Any]) -> str:
        found = [o for o in payload["occurrences"] if o["file"].endswith("extra.rs")]
        return str(found[0]["body_sha256"])

    assert digest(before) != digest(after)


def test_out_of_line_mod_allow_is_rejected(repo: Path, capsys: Capture) -> None:
    """`mod name;` への allow は別ファイルに及ぶため fail-closed で拒否する（REQ-39・#335）。"""
    (repo / "crates/core/src/extra.rs").write_text("#[allow(unsafe_code)]\nmod inner;\n")
    code, payload = run(repo, capsys)
    assert code == 10
    assert "unparsed_attribute" in {v["kind"] for v in payload["violations"]}


def test_lefthook_unsafe_hook_covers_cargo_config() -> None:
    """hook の unsafe-allowlist が .cargo/config* を glob と抽出に含む（REQ-39・#335）。"""
    text = (REPO / "lefthook.yml").read_text()
    block = text[text.index("name: unsafe-allowlist") : text.index("commit-msg:")]
    assert block.count(".cargo/config.toml") == 2
    assert block.count('".cargo/config"') == 1
    assert block.count(" .cargo/config ") == 1
