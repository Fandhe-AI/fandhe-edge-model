# PoC-26 トークナイザー golden 照合の手順

対応: REQ-41・TASK-41.1-5・PoC-26・M12・issue #390。実装は `trainer/tools/poc26/qwen2_tokenizer.py`、検査は `trainer/tests/test_poc26_tokenizer.py` の `test_golden_ids_match_real_tokenizer`。

## 目的

自作の Qwen2 トークナイザーが、実物の Qwen2.5-0.5B-Instruct の `tokenizer.json` で HF `tokenizers` と同じ id 列を返すことを、固定した probe 文で確かめる。合格条件は全件の id 列の完全一致。

- 担当: 人間（実機で実行する。Agent は手順とスクリプトの準備までに留める）
- 証拠の種別: 実機
- 結果は参照側（HF `tokenizers`）の id を `fixtures/poc26/tokenizer_golden.json` に固定し、以後は参照側なしで自作側だけを検査できるようにする。Phase 3 の Rust 実装の共有ゴールデンベクタにもなる。

## 前提

1. ベースモデル一式（`config.json`・`model.safetensors`・`tokenizer.json`・`tokenizer_config.json`）を #387 の手順でリポ外のディレクトリに用意し、各ファイルの sha256 を記録してある。
2. 参照側はリポ外の使い捨て venv に `tokenizers` を入れて実行する。インストールは通信を伴うため、オーナーの承認を得てから行う（REQ-38）。入れた版を記録する。リポの依存（`pyproject.toml`・`uv.lock`）は変えない。
3. 取得したファイルは信頼できないデータとして扱い、専用の空のディレクトリに置く。

## probe 文の要件（20〜30 件）

- 中立な文だけにする。学習・評価データの本文は入れない。
- 次を含める。
  - 日本語（ひらがな・カタカナ・漢字・句読点・全角記号）
  - 空白の連続（半角・全角・タブ）と末尾の空白
  - 数字（1 桁・複数桁・全角数字）
  - 英語の縮約形（`I'm`・`don't`・`they'll`・大文字の `DON'T`）
  - 改行（`\n`・`\r\n`・連続した改行・空白と改行の混在）
  - 絵文字
  - 結合文字（`e` と U+0301 など。NFC の確認）
  - 記号の連続（`!!`・`-->` など）
- **added_tokens の文字列（`<|im_start|>` など、special かどうかを問わない）は case の本文に入れない。** 本実装は本文中の added_tokens を照合せず文字として分割するため、HF の既定と id 列が異なる。この差は仕様で、`qwen2_tokenizer.py` の docstring「既知の差」で扱う。golden には入れず、テストも本文に added_tokens の文字列が含まれる case を拒否する。

## 生成スニペット

リポ外の使い捨て venv で実行する。`PROBES` に上の要件を満たす 20〜30 件を書く。`QWEN_DIR` と出力先は環境に合わせる。

```python
import datetime
import hashlib
import json
import sys
from pathlib import Path

import tokenizers
from tokenizers import Tokenizer

PROBES = [
    # 20〜30 件。上の要件を満たす中立な文だけを書く
]

qwen_dir = Path(sys.argv[1])  # tokenizer.json のあるディレクトリ
out = Path(sys.argv[2])  # <repo>/fixtures/poc26/tokenizer_golden.json
path = qwen_dir / "tokenizer.json"
MAX_BYTES = 16 * 1024 * 1024  # 本実装の上限と同じ
if not path.is_file() or path.stat().st_size > MAX_BYTES:
    sys.exit("tokenizer.json must be a regular file of at most 16 MiB")
digest = hashlib.sha256()
with path.open("rb") as f:
    while chunk := f.read(1 << 20):
        digest.update(chunk)
tok = Tokenizer.from_file(str(path))
doc = {
    "source": {
        "tokenizer_json_sha256": digest.hexdigest(),
        "tokenizers_version": tokenizers.__version__,
        "generated_on": datetime.date.today().isoformat(),
    },
    "cases": [
        {"text": t, "ids": tok.encode(t, add_special_tokens=False).ids} for t in PROBES
    ],
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
```

fixture の形式:

```json
{
  "source": {"tokenizer_json_sha256": "...", "tokenizers_version": "...", "generated_on": "YYYY-MM-DD"},
  "cases": [{"text": "...", "ids": [0, 1]}]
}
```

`tokenizer.json` 本体は同梱しない。fixture に載るのは probe 文と id 列だけで、Apache-2.0 の派生物にあたる点のライセンス判断は #387 と合わせる。

## 実行と合格条件

```bash
FANDHE_EDGE_QWEN_DIR=<dir> uv run --locked --directory trainer pytest tests/test_poc26_tokenizer.py -k golden
```

- 合格: `test_golden_ids_match_real_tokenizer` が PASS する（skip ではない）。確認される内容は次のとおり。
  - `source` のキーが 3 つそろう。
  - `tokenizer_json_sha256` が、自作側が解析したのと同じバイト列の sha256 と一致する。
  - case が 20〜30 件ある。
  - どの case の本文にも added_tokens の文字列がない。
  - 全 case で自作の `encode` が `ids` と完全一致する。
- 不合格なら、不一致の case を自作側の分割（`pre_tokenize`）と参照側で比べて原因を特定する。`ids` を自作側の値に書き換えて通さない。

## 記録

issue #390 のコメントに次を書く。

- 件数（合格した case 数）と `tokenizer_json_sha256`
- `tokenizers` の版・実行日時・実行した環境（OS・Python の版）
- 証拠の種別: 実機
- `fixtures/poc26/tokenizer_golden.json` をコミットした commit

probe 文自体は中立な文なので fixture に載ってよい。データ本文・id・資格情報は記録に載せない。
