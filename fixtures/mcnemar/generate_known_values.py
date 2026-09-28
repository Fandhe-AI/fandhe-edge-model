#!/usr/bin/env python3
"""McNemar の正確検定（両側）の既知値ゴールデンベクタを再生成するスクリプト。

`crates/eval/src/mcnemar.rs`・`crates/eval/tests/mcnemar_known_answer.rs` に
ハードコードされた期待値の再現手段として fixtures/mcnemar/ に置く
（review 指摘。issue #64。生成元スクリプト自体がリポジトリに無いと
19 個の期待値を独立に再現できないため）。

standard library のみを使い、`fractions.Fraction` と `math.comb` による
厳密な有理数計算で p 値を求める（Rust 実装の `ln`/`exp` を使った近似計算とは
独立した参照実装。証拠の種別: テストハーネス・独立実装との照合）。

このスクリプトは記録・再現専用であり、Rust のビルド・テストからは呼び出さ
れない（`crates/eval` の `Cargo.toml` に `[dependencies]` は追加していない。
`.claude/rules/dependency-policy.md`）。Python 本体（uv プロジェクト）の
規約（`.claude/rules/coding-python.md`）は `trainer/` 配下限定のため、本
スクリプトはその対象外（stdlib のみで依存を持たない単発の生成ツール）。

使い方::

    python3 fixtures/mcnemar/generate_known_values.py > fixtures/mcnemar/known_values.json

出力される値は `known_values.json` の内容と一致する（このスクリプトの
再実行で再現できることを確認済み）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, getcontext
from fractions import Fraction
from math import comb

# 指数部が極端に小さい値（例: 2^-1099）でも桁落ちしないよう、余裕を持った
# 精度で Decimal 化する。
getcontext().prec = 60


@dataclass(frozen=True)
class Case:
    """1 件の既知値ケース。"""

    b: int
    c: int
    note: str


def two_sided_p_exact(b: int, c: int) -> Fraction:
    """`Binomial(n, 0.5)` の下での McNemar 両側正確検定 p 値を厳密に求める。

    `n = b + c`、`k = min(b, c)` として `p = min(1, 2 * P[X <= k])`
    （`crates/eval/src/mcnemar.rs` の `two_sided_p_value` と同じ定義）。
    """
    n = b + c
    if n == 0:
        return Fraction(1)
    k = min(b, c)
    tail = sum(Fraction(comb(n, i)) for i in range(k + 1))
    p = 2 * tail / Fraction(2) ** n
    return p if p <= 1 else Fraction(1)


# `crates/eval/src/mcnemar.rs`（ユニットテスト）・
# `crates/eval/tests/mcnemar_known_answer.rs`（結合テスト）で使用している
# 既知値ケースを網羅する。値を変更する場合は両方のテストファイルを合わせて
# 更新すること。
CASES: list[Case] = [
    Case(0, 0, "b+c==0 は p=1.0 の規約（エラーにしない）"),
    Case(1, 0, "小さい既知値"),
    Case(3, 1, "小さい既知値。mcnemar.rs::known_value_b3_c1 と重複"),
    Case(6, 0, "小さい既知値。mcnemar.rs::known_value_b6_c0 と重複"),
    Case(10, 0, "小さい既知値"),
    Case(5, 5, "小さい既知値（b == c）"),
    Case(30, 29, "小さい既知値"),
    Case(12, 3, "小さい既知値"),
    Case(9, 3, "小さい既知値。mcnemar.rs::symmetric_in_b_and_c の (9,3) と重複"),
    Case(25, 10, "中程度の既知値"),
    Case(60, 40, "中程度の既知値"),
    Case(400, 250, "大きい既知値"),
    Case(5000, 4700, "大きい既知値"),
    Case(1000, 0, "大きい既知値（極端に小さい p）"),
    Case(1100, 0, "アンダーフローで f64 が 0.0 になる境界（真値 2^-1099）"),
    Case(130, 29, "PoC-10 seed0 C1 vs majority"),
    Case(123, 26, "PoC-10 seed0 C2 vs majority"),
    Case(161, 52, "PoC-10 seed0 C3 vs majority"),
    Case(108, 36, "PoC-10 seed0 C4 vs majority"),
]


def main() -> None:
    records = []
    for case in CASES:
        p = two_sided_p_exact(case.b, case.c)
        # 厳密な有理数値を 17 桁精度で丸めた十進表現（真値の記録用。
        # 指数部が f64 の表現範囲〔約 4.9e-324 未満〕を下回る場合、
        # この文字列を `f64` リテラルとしてパースすると 0.0 に
        # アンダーフローする ―― その場合でも本フィールドは丸め前の
        # 厳密値の 17 桁表現をそのまま記録する）。
        p_f64_repr = format(Decimal(p.numerator) / Decimal(p.denominator), ".17g")
        # 実際に `f64` へ変換した後の値（Python の `float` も IEEE754
        # binary64 なので Rust の `f64` と同じ丸め・アンダーフロー規則に
        # 従う）。`p_two_sided_f64_repr` が真値の 17 桁表現であるのに対し、
        # こちらは「テストコードの `f64` リテラルに実際に格納される値」
        # そのもの。指数部がアンダーフローする既知値ケース（例: (1100, 0)）
        # では `p_two_sided_f64_repr` は非ゼロの文字列のまま、本フィールドは
        # `0.0` になり、両者が一致しないのは仕様どおり（PROVENANCE.md 参照）。
        p_f64_actual = float(p_f64_repr)
        records.append(
            {
                "b": case.b,
                "c": case.c,
                "p_two_sided_exact_fraction": f"{p.numerator}/{p.denominator}",
                "p_two_sided_f64_repr": p_f64_repr,
                "p_two_sided_f64_actual": p_f64_actual,
                "note": case.note,
            }
        )
    json.dump(
        {"cases": records}, __import__("sys").stdout, ensure_ascii=False, indent=2
    )
    print()


if __name__ == "__main__":
    main()
