//! 空入力の前処理食い違いの検知・報告（REQ-23 境界値・TASK-23.2・issue #57）。
//!
//! # 背景（PoC-16 の既知の食い違い）
//!
//! REQ-23 の受け入れ基準「境界値」は、空文字列の入力に対する前処理が
//! 推論の経路と評価の経路とで結果が一致するとは限らないことを認め、
//! その食い違いを「検知・報告できること」を最小の受け入れ基準とする
//! （両経路の統一は「検討中」のままで対象外。`.claude/rules/spec-reference.md`
//! 「状態が『検討中』の要件は確定扱いしない」）。PoC-16 では Rust 側
//! `preprocess::encode_bytes("")` が詰め物トークン 1 個の `[0]` を、
//! Python 側 `train_mlx.encode("")` が空列 `[]` を返す食い違いが実測された。
//!
//! # 本リポの現状（実装済みを装わない）
//!
//! - 本リポの学習ワーカー（`trainer/src/fandhe_edge_trainer/encoding.py`）の
//!   `encode_bytes` は既に `[0]` を返すよう置き換え済みで、SSOT ベクタ
//!   （`fixtures/preprocess/byte_encoding_vectors.json` の `empty_input`
//!   ・`whitespace_only_spaces_tabs_newlines` 等）と `trainer/tests/test_encoding.py`
//!   がそれを照合している。`[]` を返していたのは PoC-16 時点の `train_mlx.encode`
//!   であり、本リポには存在しない
//! - 本リポには Rust の前処理・推論ランタイムが未作成（推論 SDK 層のパスは
//!   未確定）のため、「2 つの実装を実際に走らせて比較する」検知は本 TASK
//!   時点ではできない。本モジュールが提供するのは (a) 学習ワーカー経路の
//!   契約（正規化後に空となる入力は `[0]`）を SSOT ベクタから Rust 側でも
//!   固定する手段（[`classify_empty_input_encoding`]・
//!   `crates/data/tests/preprocess_boundary.rs` の結合テスト）と、
//!   (b) 2 経路（推論・評価）が実際に出したトークン列を受け取って
//!   一致／食い違いを分類する純粋関数（[`compare_empty_input_encodings`]。
//!   PoC-16 の `[0]` vs `[]` を検知できる）の 2 点である。Rust 推論
//!   ランタイムを作成したうえでの実地の一致検証（`normalize_input`／
//!   `encode_bytes` の Rust 移植・SSOT ベクタ全件照合）は Chore #10
//!   Deliverable B（REQ-28）の担当とする
//! - [`EmptyInputConsistency`]・データ契約層の [`crate::eval_input::WarningCode::EmptyInput`]
//!   から 7 種終了コード（REQ-21）・CLI の JSON 出力への対応付けは CLI 側
//!   （TASK-21.2・TASK-33.x）の責務であり、本 crate はコードの値を
//!   ハードコードしない
//!
//! # なぜ NFKC 正規化が不要か（空判定の根拠）
//!
//! 学習ワーカーの `normalize_input`（NFKC 正規化 → 空白の圧縮）が空文字列を
//! 返すのは「全文字が空白」の場合に限られる。これは Unicode の合成規則上、
//! 空白文字（結合クラス 0）が NFKC 正規化で他の文字と結合しないため、
//! 1 文字ごとに「NFKC 後に空白として消える」かどうかを判定すれば文字列全体
//! に拡張できる。Python の `unicodedata`（Unicode 16.0.0）で全コードポイント
//! を走査し、「`NFKC(c)` を `str.split()` した結果が空」⇔「`c.isspace()`」が
//! 全件一致することを確認済み（不一致 0 件。証拠種別: テストハーネス
//! 〔計画時の走査スクリプト〕）。したがって [`is_empty_after_normalization`]
//! は NFKC 正規化を実装せずに「全文字が Python の空白」だけで判定でき、
//! `unicode-normalization` という未承認の依存（`.claude/rules/dependency-policy.md`）
//! を追加する必要がない。
//!
//! # Python の空白集合と Rust の差
//!
//! Python の `str.isspace()` が真になる文字は 29 個
//! （U+0009–U+000D、U+001C–U+001F、U+0020、U+0085、U+00A0、U+1680、
//! U+2000–U+200A、U+2028、U+2029、U+202F、U+205F、U+3000）。Rust 標準
//! ライブラリの `char::is_whitespace`（Unicode `White_Space` プロパティ）
//! には U+001C–U+001F（情報分離文字）が含まれない。PoC-16 の Rust 側も
//! 独自の `is_py_whitespace` でこの差を吸収していた（学習ワーカー
//! `encoding.py` の docstring に記載）。本モジュールの [`is_py_whitespace`]
//! は同じ差を吸収する独自の述語であり、`str::split_whitespace` を使う
//! [`crate::eval_input::normalize_input`]（重複・矛盾検出用。U+001C–U+001F
//! を空白扱いしない）とは意図的に別の空白集合を持つ（スコープ外の観察事項。
//! 本 issue では `eval_input::normalize_input` を変更しない）。

/// Python の `str.isspace()` と同じ空白集合で `c` を判定する。
///
/// Rust 標準の [`char::is_whitespace`]（Unicode `White_Space`）に
/// U+001C〜U+001F（情報分離文字 4 種）を加えたものが、学習ワーカーの
/// `str.split()`（Python の空白集合）と一致する（モジュール doc「Python の
/// 空白集合と Rust の差」参照）。
pub fn is_py_whitespace(c: char) -> bool {
    c.is_whitespace() || ('\u{1C}'..='\u{1F}').contains(&c)
}

/// 学習ワーカーの `normalize_input`（NFKC 正規化 → 空白圧縮）を適用した結果が
/// 空文字列になるかどうかを、NFKC 正規化を実装せずに判定する。
///
/// 空文字列自身も真を返す（0 文字は「全文字が空白」の空虚な真）。判定根拠は
/// モジュール doc「なぜ NFKC 正規化が不要か」を参照。
pub fn is_empty_after_normalization(input: &str) -> bool {
    input.chars().all(is_py_whitespace)
}

/// 空入力（[`is_empty_after_normalization`] が真の入力）に対して前処理が
/// 出したトークン列の分類。
///
/// 学習ワーカー（`trainer/src/fandhe_edge_trainer/encoding.py`）は
/// `PaddingOnly { len: 1 }`（`[0]`）を返す契約になっている
/// （`crates/data/tests/preprocess_boundary.rs` が SSOT ベクタから固定）。
/// PoC-16 時点の Python 側 `train_mlx.encode("")` は `EmptySequence`（`[]`）
/// を返していた（本リポには存在しない実装。モジュール doc「背景」参照）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmptyInputEncoding {
    /// トークン列が空（`[]`）。
    EmptySequence,
    /// 詰め物トークン（値 `0`）のみで構成される（`[0, 0, ...]`）。`len` は
    /// 長さ（1 以上）。
    PaddingOnly { len: usize },
}

/// [`classify_empty_input_encoding`] が「空入力に対する前処理として不正」と
/// 判定した理由。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmptyInputEncodingError {
    /// 詰め物トークン（`0`）以外の値が含まれていた（`index` はその位置。
    /// 0 始まり）。空入力なのに実データ相当のトークンが出た、前処理側の
    /// 不具合を疑わせる兆候。
    NonPaddingToken { index: usize },
}

impl EmptyInputEncodingError {
    pub fn code(&self) -> &'static str {
        match self {
            EmptyInputEncodingError::NonPaddingToken { .. } => "non_padding_token",
        }
    }
}

/// 空入力に対する前処理のトークン列（`ids`）を分類する。
///
/// `ids` の値そのものは戻り値・エラーのいずれにも含めない（位置情報のみ。
/// `.claude/rules/security.md`「データ本文をログ・エラーメッセージへ転記
/// しない」に準じ、評価契約の診断情報の方針
/// （[`crate::eval_input`] モジュール doc「PoC-9 との差分」）を踏襲する）。
/// 追加のアロケーションは行わない。
pub fn classify_empty_input_encoding(
    ids: &[i64],
) -> Result<EmptyInputEncoding, EmptyInputEncodingError> {
    if ids.is_empty() {
        return Ok(EmptyInputEncoding::EmptySequence);
    }
    if let Some(index) = ids.iter().position(|&id| id != 0) {
        return Err(EmptyInputEncodingError::NonPaddingToken { index });
    }
    Ok(EmptyInputEncoding::PaddingOnly { len: ids.len() })
}

/// 前処理を実行した経路（推論経路か評価経路か）。
///
/// [`compare_empty_input_encodings`] が食い違いの報告先を区別するために使う。
/// 現時点では Rust 推論ランタイムが未作成のため、`Inference` 側の実測値は
/// 呼び出し元（将来のランタイム・テストハーネス）が用意する想定である
/// （モジュール doc「本リポの現状」参照）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreprocessPath {
    /// 推論経路（学習非依存の推論ランタイム。REQ-28/30〜32。未作成）。
    Inference,
    /// 評価経路（学習ワーカーの前処理、または評価器が用いる前処理）。
    Evaluation,
}

impl PreprocessPath {
    pub fn code(&self) -> &'static str {
        match self {
            PreprocessPath::Inference => "inference",
            PreprocessPath::Evaluation => "evaluation",
        }
    }
}

/// 推論経路・評価経路の空入力前処理結果を比較した結論。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmptyInputConsistency {
    /// 両経路が同じ分類のトークン列を出した（食い違いなし）。
    Consistent(EmptyInputEncoding),
    /// 両経路が異なる分類のトークン列を出した（PoC-16 の `[0]` vs `[]` の
    /// ような食い違い。REQ-23 境界値が定める「検知・報告」の対象）。
    Diverged {
        inference: EmptyInputEncoding,
        evaluation: EmptyInputEncoding,
    },
}

impl EmptyInputConsistency {
    /// CLI・上位層が機械判定に使うキー（出力文字列は英語。
    /// `.claude/rules/japanese-style.md`「プログラムの出力文字列は英語」）。
    /// 7 種終了コード（REQ-21）への対応付けは CLI 側（TASK-21.2・TASK-33.x）
    /// の責務であり、本関数はその数値をハードコードしない。
    pub fn code(&self) -> &'static str {
        match self {
            EmptyInputConsistency::Consistent(_) => "consistent",
            EmptyInputConsistency::Diverged { .. } => "empty_input_preprocess_divergence",
        }
    }
}

/// 推論経路・評価経路それぞれが空入力に対して出したトークン列を分類し、
/// 一致するかどうかを比較する（REQ-23 境界値・TASK-23.2 の中心的な検知
/// ロジック）。
///
/// 呼び出し元は事前に対象の入力が [`is_empty_after_normalization`] を満たす
/// ことを確認しておく（本関数自体は入力文字列を受け取らず、既に前処理を
/// 実行した後のトークン列だけを扱う）。いずれかの経路のトークン列が
/// [`classify_empty_input_encoding`] でエラーになった場合は、その経路を
/// 示す [`PreprocessPath`] とエラーの組を返す（比較を打ち切る。両経路とも
/// 不正な場合は `inference` 側を優先して報告する）。
pub fn compare_empty_input_encodings(
    inference: &[i64],
    evaluation: &[i64],
) -> Result<EmptyInputConsistency, (PreprocessPath, EmptyInputEncodingError)> {
    let inference_encoding =
        classify_empty_input_encoding(inference).map_err(|e| (PreprocessPath::Inference, e))?;
    let evaluation_encoding =
        classify_empty_input_encoding(evaluation).map_err(|e| (PreprocessPath::Evaluation, e))?;

    if inference_encoding == evaluation_encoding {
        Ok(EmptyInputConsistency::Consistent(inference_encoding))
    } else {
        Ok(EmptyInputConsistency::Diverged {
            inference: inference_encoding,
            evaluation: evaluation_encoding,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-23・TASK-23.2: Python `str.isspace()` が真になる 29 文字すべてで
    /// [`is_py_whitespace`] が真になること（モジュール doc「Python の空白
    /// 集合と Rust の差」の列挙に基づく具体値）。
    #[test]
    fn req23_is_py_whitespace_matches_python_isspace_29_chars() {
        const PY_WHITESPACE: [char; 29] = [
            '\u{09}', '\u{0A}', '\u{0B}', '\u{0C}', '\u{0D}', '\u{1C}', '\u{1D}', '\u{1E}',
            '\u{1F}', '\u{20}', '\u{85}', '\u{A0}', '\u{1680}', '\u{2000}', '\u{2001}', '\u{2002}',
            '\u{2003}', '\u{2004}', '\u{2005}', '\u{2006}', '\u{2007}', '\u{2008}', '\u{2009}',
            '\u{200A}', '\u{2028}', '\u{2029}', '\u{202F}', '\u{205F}', '\u{3000}',
        ];
        for c in PY_WHITESPACE {
            assert!(
                is_py_whitespace(c),
                "expected whitespace: U+{:04X}",
                c as u32
            );
        }
    }

    /// 非空白文字（Rust の `is_whitespace` が拾わない境界含む）は偽になること。
    #[test]
    fn req23_is_py_whitespace_rejects_non_whitespace() {
        for c in ['\u{0000}', '\u{200B}', '\u{FEFF}', '\u{00A8}', 'a', 'あ'] {
            assert!(
                !is_py_whitespace(c),
                "expected non-whitespace: U+{:04X}",
                c as u32
            );
        }
    }

    /// U+0000–U+3000 の範囲で「29 文字の集合に含まれる ⇔ 述語が真」を全件
    /// 照合する（部分一致ではなく全件一致であることを保証する）。
    #[test]
    fn req23_is_py_whitespace_matches_python_isspace_over_range() {
        const PY_WHITESPACE_SET: [u32; 29] = [
            0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x85, 0xA0, 0x1680, 0x2000,
            0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200A, 0x2028,
            0x2029, 0x202F, 0x205F, 0x3000,
        ];
        for code_point in 0u32..=0x3000 {
            let Some(c) = char::from_u32(code_point) else {
                continue;
            };
            let expected = PY_WHITESPACE_SET.contains(&code_point);
            assert_eq!(
                is_py_whitespace(c),
                expected,
                "mismatch at U+{code_point:04X}"
            );
        }
    }

    /// REQ-23・TASK-23.2: 正規化後に空となる代表的な入力が真になること。
    #[test]
    fn req23_is_empty_after_normalization_true_cases() {
        for input in ["", "   ", "\t\n", "\u{3000}", "\u{1C}", "\u{85}\u{A0}"] {
            assert!(
                is_empty_after_normalization(input),
                "expected empty after normalization: {input:?}"
            );
        }
    }

    /// 非空白文字を含む入力は偽になること（U+200B はゼロ幅スペースだが
    /// Python の空白集合には含まれない）。
    #[test]
    fn req23_is_empty_after_normalization_false_cases() {
        for input in ["a", " a ", "\u{200B}"] {
            assert!(
                !is_empty_after_normalization(input),
                "expected non-empty after normalization: {input:?}"
            );
        }
    }

    /// REQ-23・TASK-23.2: 空トークン列は `EmptySequence` に分類される
    /// （PoC-16 時点の Python 側 `train_mlx.encode("")` の出力形）。
    #[test]
    fn req23_classify_empty_sequence() {
        assert_eq!(
            classify_empty_input_encoding(&[]),
            Ok(EmptyInputEncoding::EmptySequence)
        );
    }

    /// 詰め物トークンのみのトークン列は `PaddingOnly` に分類される
    /// （本リポの学習ワーカー `encode_bytes` の出力形）。
    #[test]
    fn req23_classify_padding_only() {
        assert_eq!(
            classify_empty_input_encoding(&[0]),
            Ok(EmptyInputEncoding::PaddingOnly { len: 1 })
        );
        assert_eq!(
            classify_empty_input_encoding(&[0, 0, 0]),
            Ok(EmptyInputEncoding::PaddingOnly { len: 3 })
        );
    }

    /// 詰め物トークン以外の値が混ざると `NonPaddingToken` エラーになり、
    /// その位置（0 始まり）を返す。
    #[test]
    fn req23_classify_non_padding_token_error() {
        assert_eq!(
            classify_empty_input_encoding(&[0, 5]),
            Err(EmptyInputEncodingError::NonPaddingToken { index: 1 })
        );
        assert_eq!(
            classify_empty_input_encoding(&[0, 5]).unwrap_err().code(),
            "non_padding_token"
        );
    }

    /// REQ-23・TASK-23.2: PoC-16 で実測された食い違い（推論側 `[0]`・
    /// 評価側 `[]`）を `Diverged` として検知できること。
    #[test]
    fn req23_compare_detects_poc16_divergence() {
        let result = compare_empty_input_encodings(&[0], &[]);
        assert_eq!(
            result,
            Ok(EmptyInputConsistency::Diverged {
                inference: EmptyInputEncoding::PaddingOnly { len: 1 },
                evaluation: EmptyInputEncoding::EmptySequence,
            })
        );
        assert_eq!(result.unwrap().code(), "empty_input_preprocess_divergence");
    }

    /// 両経路が同じ分類なら `Consistent` になる。
    #[test]
    fn req23_compare_consistent_when_equal() {
        let result = compare_empty_input_encodings(&[0], &[0]);
        assert_eq!(
            result,
            Ok(EmptyInputConsistency::Consistent(
                EmptyInputEncoding::PaddingOnly { len: 1 }
            ))
        );
        assert_eq!(result.unwrap().code(), "consistent");
    }

    /// 不正なトークン列は経路情報付きのエラーとして返り、比較を打ち切る。
    #[test]
    fn req23_compare_propagates_error_with_side() {
        assert_eq!(
            compare_empty_input_encodings(&[0, 9], &[]),
            Err((
                PreprocessPath::Inference,
                EmptyInputEncodingError::NonPaddingToken { index: 1 }
            ))
        );
        assert_eq!(
            compare_empty_input_encodings(&[0], &[0, 9]),
            Err((
                PreprocessPath::Evaluation,
                EmptyInputEncodingError::NonPaddingToken { index: 1 }
            ))
        );
    }
}
