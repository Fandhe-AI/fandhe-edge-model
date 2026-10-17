"""コミットメッセージの各行が 100 文字以内かを検査する（lefthook commit-msg から呼ぶ）。

CI の commitlint（config-conventional の header-max-length・body-max-line-length・
footer-max-line-length）と同じ上限・数え方に揃える。文字数は JS の string length
（UTF-16 コード単位。絵文字などの BMP 外は 2）で、バイト数ではない。
`#` で始まる行は git が commit 時に除去するため対象外。標準ライブラリのみ。
"""

import sys

MAX_LEN = 100


def check(text: str) -> list[tuple[int, int]]:
    """上限を超えた行の (行番号, 文字数) を返す。"""
    bad = []
    for no, line in enumerate(text.split("\n"), 1):
        if line.startswith("#"):
            continue
        n = len(line.rstrip("\r").encode("utf-16-le")) // 2
        if n > MAX_LEN:
            bad.append((no, n))
    return bad


def main(argv: list[str]) -> int:
    with open(argv[1], encoding="utf-8") as f:
        bad = check(f.read())
    for no, n in bad:
        msg = f"NG: {no} 行目が {n} 文字です（上限 {MAX_LEN} 文字。commitlint と同じ数え方）"
        print(msg, file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
