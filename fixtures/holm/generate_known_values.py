#!/usr/bin/env python3
"""Holm 法による多重比較補正の既知値ゴールデンベクタを再生成するスクリプト。

`crates/eval/src/holm.rs`・`crates/eval/tests/holm_known_answer.rs` に
ハードコードされた期待値の再現手段として `fixtures/holm/` に置く
（`fixtures/mcnemar/generate_known_values.py` と同じ方針。issue #67・
TASK-25.3）。

standard library のみを使い、`fractions.Fraction` と `math.comb` による
厳密な有理数計算で McNemar の両側正確 p 値を求め、その上で Holm 補正
（`crates/eval/src/holm.rs::holm_adjust` と同じ手順: 昇順ソート →
`(m - k) * p` → 累積最大による単調化 → 1 で打ち切り）を行う。Rust 実装の
`ln`/`exp` を使った近似計算とは独立した参照実装であり、証拠の種別は
テストハーネス・独立実装との照合。

このスクリプトは記録・再現専用であり、Rust のビルド・テストからは呼び出さ
れない（`crates/eval` の `Cargo.toml` に依存追加は無い。`.claude/rules/
dependency-policy.md`）。`trainer/`（uv プロジェクト）の規約（`.claude/rules/
coding-python.md`）は `trainer/` 配下限定のため、本スクリプトはその対象外
（stdlib のみで依存を持たない単発の生成ツール）。

使い方::

    python3 fixtures/holm/generate_known_values.py > fixtures/holm/known_values.json

出力される値は `known_values.json` の内容と一致する（このスクリプトの
再実行で再現できることを確認済み）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, getcontext
from fractions import Fraction
from math import comb

# 指数部が極端に小さい値でも桁落ちしないよう、余裕を持った精度で Decimal 化する。
getcontext().prec = 60


def two_sided_p_exact(b: int, c: int) -> Fraction:
    """`Binomial(n, 0.5)` の下での McNemar 両側正確検定 p 値を厳密に求める。

    `n = b + c`、`k = min(b, c)` として `p = min(1, 2 * P[X <= k])`
    （`crates/eval/src/mcnemar.rs` の `two_sided_p_value` と同じ定義。
    `fixtures/mcnemar/generate_known_values.py` と同一実装）。
    """
    n = b + c
    if n == 0:
        return Fraction(1)
    k = min(b, c)
    tail = sum(Fraction(comb(n, i)) for i in range(k + 1))
    p = 2 * tail / Fraction(2) ** n
    return p if p <= 1 else Fraction(1)


def holm_adjust_exact(p_values: list[Fraction], m: int) -> list[Fraction]:
    """有理数のまま Holm 補正を行う（`crates/eval/src/holm.rs::holm_adjust` と同じ手順）。

    1. 昇順に安定ソート（同値は入力順を保つ）
    2. 0 始まりで k 番目に `(m - k)` を掛ける
    3. 累積最大で単調化する
    4. 1 で打ち切る

    戻り値は入力順に並べ直す。
    """
    indexed = list(enumerate(p_values))
    # Python の sort は安定なので、同値は入力順（元のインデックス順）を保つ。
    indexed.sort(key=lambda pair: pair[1])

    result: dict[int, Fraction] = {}
    running = Fraction(0)
    for k, (idx, p) in enumerate(indexed):
        multiplier = m - k
        adjusted = multiplier * p
        if adjusted > 1:
            adjusted = Fraction(1)
        running = max(running, adjusted)
        result[idx] = running

    return [result[i] for i in range(len(p_values))]


@dataclass(frozen=True)
class Candidate:
    """1 候補分の分割表（不一致ペア数）。"""

    name: str
    b: int
    c: int


@dataclass(frozen=True)
class Family:
    """Holm 補正の対象となる 1 つの族（m と候補の集合）。"""

    key: str
    family_size: int
    candidates: list[Candidate]
    note: str


# `crates/eval/tests/holm_known_answer.rs` の §5.2 各ケースに対応する族。
FAMILIES: list[Family] = [
    Family(
        key="poc10_eval_incat_seed0",
        family_size=4,
        candidates=[
            Candidate("C1", 130, 29),
            Candidate("C2", 123, 26),
            Candidate("C3", 161, 52),
            Candidate("C4", 108, 36),
        ],
        note="PoC-10 eval_incat seed0（m=4）。すべて SignificantlyBetter の想定",
    ),
    Family(
        key="poc10_eval_incat_seed1",
        family_size=4,
        candidates=[
            Candidate("C1", 130, 29),
            Candidate("C2", 123, 26),
            Candidate("C3", 143, 43),
            Candidate("C4", 125, 58),
        ],
        note="PoC-10 eval_incat seed1（m=4）",
    ),
    Family(
        key="poc10_eval_incat_seed2",
        family_size=4,
        candidates=[
            Candidate("C1", 130, 29),
            Candidate("C2", 123, 26),
            Candidate("C3", 144, 40),
            Candidate("C4", 134, 58),
        ],
        note="PoC-10 eval_incat seed2（m=4）",
    ),
    Family(
        key="poc10_ref_test_seed0",
        family_size=4,
        candidates=[
            Candidate("C1", 8, 79),
            Candidate("C2", 260, 82),
            Candidate("C3", 279, 97),
            Candidate("C4", 97, 78),
        ],
        note=(
            "PoC-10 ref_test seed0（m=4）。C1 は b<c で向きの確認、"
            "C4 は補正後 p が α 付近になる境界の確認"
        ),
    ),
    Family(
        key="poc25_primary_seed0",
        family_size=4,
        candidates=[
            Candidate("C1", 156, 36),
            Candidate("C2", 138, 35),
            Candidate("C3", 177, 89),
            Candidate("C4", 79, 4),
        ],
        note="PoC-25 primary seed0（m=4）。C1 の補正後が累積最大で C4 と等しくなる想定",
    ),
    Family(
        key="alpha_boundary_solo",
        family_size=1,
        candidates=[Candidate("solo", 13, 4)],
        note="(13,4) を単独（m=1）で補正 → 補正前と同じ値のまま SignificantlyBetter",
    ),
    Family(
        key="alpha_boundary_pair_flips",
        family_size=2,
        candidates=[Candidate("a", 13, 4), Candidate("b", 22, 10)],
        note="(13,4) と (22,10) を m=2 の族にすると (13,4) が ×2 され判定が反転する",
    ),
    Family(
        key="alpha_boundary_pair_holds",
        family_size=2,
        candidates=[Candidate("a", 13, 4), Candidate("b", 130, 29)],
        note="(13,4) と十分小さい p の (130,29) を m=2 にしても (13,4) は SignificantlyBetter を保つ",
    ),
    Family(
        key="dropout_3_of_4",
        family_size=4,
        candidates=[
            Candidate("C1", 130, 29),
            Candidate("C2", 123, 26),
            Candidate("C3", 161, 52),
        ],
        note="PoC-10 seed0 の 4 候補のうち 3 候補だけを m=4 で渡す（脱落）",
    ),
]


def main() -> None:
    families_out = []
    for family in FAMILIES:
        exact_p = [two_sided_p_exact(c.b, c.c) for c in family.candidates]
        adjusted = holm_adjust_exact(exact_p, family.family_size)

        candidates_out = []
        for cand, p, adj in zip(family.candidates, exact_p, adjusted):
            p_repr = format(Decimal(p.numerator) / Decimal(p.denominator), ".17g")
            adj_repr = format(Decimal(adj.numerator) / Decimal(adj.denominator), ".17g")
            candidates_out.append(
                {
                    "name": cand.name,
                    "b": cand.b,
                    "c": cand.c,
                    "raw_p_exact_fraction": f"{p.numerator}/{p.denominator}",
                    "raw_p_f64_actual": float(p_repr),
                    "adjusted_p_exact_fraction": f"{adj.numerator}/{adj.denominator}",
                    "adjusted_p_f64_actual": float(adj_repr),
                }
            )

        families_out.append(
            {
                "key": family.key,
                "family_size": family.family_size,
                "note": family.note,
                "candidates": candidates_out,
            }
        )

    # advisor 指摘: poc25_primary_seed0 で C1×3 が C4×4 の累積最大にちょうど
    # 巻き取られることの余裕（相対差）を明示する。差が小さすぎる場合、
    # Rust 側の libm 近似誤差（相対 1e-14 程度）でも判定が入れ替わりうる。
    c1_raw = two_sided_p_exact(156, 36)
    c4_raw = two_sided_p_exact(79, 4)
    c1_times_3 = 3 * c1_raw
    c4_times_4 = 4 * c4_raw
    gap = (c4_times_4 - c1_times_3) / c4_times_4
    poc25_margin = {
        "c1_times_3_exact_fraction": f"{c1_times_3.numerator}/{c1_times_3.denominator}",
        "c4_times_4_exact_fraction": f"{c4_times_4.numerator}/{c4_times_4.denominator}",
        "relative_gap_c4x4_minus_c1x3_over_c4x4": float(
            format(Decimal(gap.numerator) / Decimal(gap.denominator), ".6g")
        ),
        "note": (
            "正なら C1×3 < C4×4 で、累積最大により C1 の補正後値は C4×4 に"
            "巻き取られる。相対差が libm 近似誤差（相対 1e-14 程度）を"
            "十分上回ることを示す診断用フィールド"
        ),
    }

    json.dump(
        {"families": families_out, "poc25_primary_seed0_margin_check": poc25_margin},
        __import__("sys").stdout,
        ensure_ascii=False,
        indent=2,
    )
    print()


if __name__ == "__main__":
    main()
