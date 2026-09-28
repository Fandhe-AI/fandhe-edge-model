#!/usr/bin/env python3
"""McNemar 検定の必要件数（Connor 式）の既知値ゴールデンベクタを再生成する
スクリプト。

`crates/eval/src/sample_size.rs`・`crates/eval/tests/required_sample_size.rs`
にハードコードされた期待値の再現手段として fixtures/sample_size/ に置く
（issue #66・TASK-25.2。`fixtures/mcnemar/generate_known_values.py` と同じ
流儀）。

standard library のみを使い、逆正規分布関数（分位点）を 2 系統で計算して
一致を確認してから出力する（Rust 実装〔Wichura (1988) AS241〕とは独立な
照合。証拠の種別: テストハーネス・独立実装との照合）:

- (i) `statistics.NormalDist().inv_cdf`: CPython の実装自体が AS241 を
  移植したものであり、係数はほぼ同一アルゴリズム（Rust 実装との照合として
  は係数転記チェックの意味合いが強い）
- (ii) `math.erfc` による `Φ(x) = 0.5 * erfc(-x / sqrt(2))` の二分法:
  分位点探索のアルゴリズム自体が (i) と独立している

PoC-10・PoC-24 の事前登録値（155・188・221・272）は、上記 2 系統いずれとも
独立に記録された第三の参照値であり、本スクリプトの出力がこれらと一致する
ことも別途確認する（`PROVENANCE.md` 参照）。

このスクリプトは記録・再現専用であり、Rust のビルド・テストからは呼び出さ
れない（`crates/eval` の `Cargo.toml` に依存追加はしていない）。Python 本体
（uv プロジェクト）の規約（`.claude/rules/coding-python.md`）は `trainer/`
配下限定のため、本スクリプトはその対象外（stdlib のみで依存を持たない単発
の生成ツール）。

使い方::

    python3 fixtures/sample_size/generate_known_values.py > fixtures/sample_size/known_values.json

出力される値は `known_values.json` の内容と一致する（このスクリプトの
再実行で再現できることを確認済み）。
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from statistics import NormalDist

_ND = NormalDist()

# 分位点探索の二分法の許容誤差。(i)(ii) の相対差が本許容以下であることを
# 確認してから出力する（一致確認の閾値であり、Rust 実装の許容差 1e-9 とは
# 別物）。
_QUANTILE_CROSS_CHECK_RTOL = 1e-12


def _normal_cdf_erfc(x: float) -> float:
    """`math.erfc` による標準正規分布の累積分布関数（独立実装）。"""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def _inv_cdf_bisection(
    p: float, lo: float = -40.0, hi: float = 40.0, iters: int = 200
) -> float:
    """`_normal_cdf_erfc` に対する二分法での逆関数（独立実装の分位点探索）。"""
    for _ in range(iters):
        mid = (lo + hi) / 2
        if _normal_cdf_erfc(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _cross_checked_quantile(p: float) -> float:
    """(i) `NormalDist.inv_cdf` と (ii) `erfc` 二分法の両方で分位点を求め、
    相対差が `_QUANTILE_CROSS_CHECK_RTOL` を超えたら異常終了する。
    """
    a = _ND.inv_cdf(p)
    b = _inv_cdf_bisection(p)
    if a == 0.0:
        diff_ok = abs(a - b) <= _QUANTILE_CROSS_CHECK_RTOL
    else:
        diff_ok = abs(a - b) / abs(a) <= _QUANTILE_CROSS_CHECK_RTOL
    if not diff_ok:
        raise SystemExit(
            f"quantile cross-check failed for p={p!r}: NormalDist={a!r} bisection={b!r}"
        )
    return a


@dataclass(frozen=True)
class QuantileCase:
    """分位点の既知値 1 件。"""

    p: float
    note: str


QUANTILE_CASES: list[QuantileCase] = [
    QuantileCase(0.5, "中央値（0.0 の対称点）"),
    QuantileCase(0.8, "power=0.8 の z_b"),
    QuantileCase(0.975, "alpha=0.05 の z_a（1 - alpha/2）"),
    QuantileCase(1 - 0.025 / 2, "alpha=0.025 の z_a（Holm m=2 最厳段）"),
    QuantileCase(1 - 0.0125 / 2, "alpha=0.0125 の z_a（Holm m=4 最厳段。PoC-10）"),
    QuantileCase(
        1 - (0.05 / 12) / 2, "alpha=0.05/12 の z_a（4 候補 x 3 seed を 1 族）"
    ),
    QuantileCase(1e-10, "定義域境界に近い極小値"),
]


@dataclass(frozen=True)
class SampleSizeCase:
    """必要件数の既知値 1 件（PoC-10・PoC-24 の事前登録の仮定）。"""

    p_b: float
    p_c: float
    alpha: float
    power: float
    note: str


# p_b=0.15・p_c=0.05・power=0.8 は PoC-10 の事前登録における仮定であり、
# 実測に基づく値ではない（`crates/eval/src/sample_size.rs` のドキュメント
# コメント参照）。
SAMPLE_SIZE_CASES: list[SampleSizeCase] = [
    SampleSizeCase(0.15, 0.05, 0.05, 0.8, "単純比較（alpha=0.05）"),
    SampleSizeCase(0.15, 0.05, 0.025, 0.8, "Holm m=2 最厳段（PoC-24）"),
    SampleSizeCase(0.15, 0.05, 0.0125, 0.8, "Holm m=4 最厳段（PoC-10 事前登録の下限）"),
    SampleSizeCase(0.15, 0.05, 0.05 / 12, 0.8, "4 候補 x 3 seed を 1 族"),
]


def _required_n(p_b: float, p_c: float, alpha: float, power: float) -> float:
    """Connor (1987) の McNemar サンプルサイズ公式（丸め前の f64）。"""
    z_a = _cross_checked_quantile(1 - alpha / 2)
    z_b = _cross_checked_quantile(power)
    d = p_b - p_c
    s = p_b + p_c
    return (z_a * math.sqrt(s) + z_b * math.sqrt(s - d * d)) ** 2 / (d * d)


# PoC-10・PoC-24 の事前登録記録（本スクリプトとは独立に記録された値。
# 一致を確認するための第三の参照値）。
_POC_REGISTERED_CEIL = {
    0.05: 155,
    0.025: 188,
    0.0125: 221,
    0.05 / 12: 272,
}


def main() -> None:
    quantiles = []
    for case in QUANTILE_CASES:
        value = _cross_checked_quantile(case.p)
        quantiles.append(
            {
                "p": case.p,
                "quantile": value,
                "note": case.note,
            }
        )

    sample_sizes = []
    for case in SAMPLE_SIZE_CASES:
        n = _required_n(case.p_b, case.p_c, case.alpha, case.power)
        ceil_n = math.ceil(n)
        expected_ceil = _POC_REGISTERED_CEIL.get(case.alpha)
        if expected_ceil is not None and expected_ceil != ceil_n:
            raise SystemExit(
                f"PoC registered ceil mismatch for alpha={case.alpha!r}: "
                f"computed={ceil_n} registered={expected_ceil}"
            )
        sample_sizes.append(
            {
                "p_b": case.p_b,
                "p_c": case.p_c,
                "alpha": case.alpha,
                "power": case.power,
                "n": n,
                "ceil_n": ceil_n,
                "note": case.note,
            }
        )

    json.dump(
        {"quantiles": quantiles, "sample_sizes": sample_sizes},
        sys.stdout,
        ensure_ascii=False,
        indent=2,
    )
    print()


if __name__ == "__main__":
    main()
