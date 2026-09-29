//! 推論関数へ `input` だけを渡す呼び出し境界と、渡した引数の事後照合。
//!
//! CLI の `evaluate` 工程（REQ-33。配線は issue #140 で未実装）が、評価データの
//! 各レコードを推論へ流す際に呼ぶ想定の層である（REQ-27 異常系「推論関数に
//! `input` 以外の情報〔正解・ID・タグ〕が渡っていないことを確認できる」・
//! TASK-27.2・issue #71。PoC-9 `evaluator/harness.py` の
//! `ArgumentRecordingPredictor` を移植する）。評価契約「推論関数には `input`
//! だけを渡す」（`.claude/rules/evaluation-contract.md`）を、共通コア
//! `fandhe_edge_core::infer_input` の「推論ランタイムへは `input()` のみを渡す」
//! 想定と対応させて機械照合できる形にする。
//!
//! # コンパイル時保証と実行時照合の分担
//!
//! PoC-9 は Python の動的型のため「位置引数 1 個・キーワード引数 0 個・型が
//! `str`」を実行時に確認していた。Rust では推論関数の型を `FnMut(&str) -> P`
//! とすることで、この 3 点はコンパイル時に保証される（実行時検査として再発明
//! しない）。実行時に残る検査は、各呼び出しの引数が対応するレコードの `input`
//! とバイト単位で完全一致すること、および呼び出し回数がレコード件数と一致する
//! ことである。部分一致（contains）判定は、正当な `input` が正解ラベル文字列を
//! 含む場合に誤検出するため採用しない。
//!
//! # 記録の扱い
//!
//! 記録器は生の文字列を保持せず、バイト長と sha256 のみを残す（学習・評価データ
//! の本文をログ・エラーへ転記しない。`.claude/rules/security.md`）。違反型も
//! index・件数のみを持ち、`id`・`input`・`gold`・`tags` の実値を含まない。
//! 空文字列の `input` は有効な入力として扱う（TASK-23.2・issue #212）。
//!
//! # 資源上限
//!
//! レコード件数は [`crate::significance::MAX_EVAL_RECORDS`] を超えると確保前に
//! 拒否する。記録器の記録件数も同じ上限で打ち切り、超過は違反として扱う。

use fandhe_edge_core::hash::Sha256Digest;

use crate::significance::MAX_EVAL_RECORDS;

/// 評価器が保持する 1 件の借用ビュー。所有権は取らない（REQ-27）。
///
/// 推論関数へ渡すのは `input` だけで、`id`・`gold`・`tags` は評価器側にのみ残す。
/// data 層の `ValidRecord` からの変換は呼び出し側が行う（eval は data に通常依存しない）。
#[derive(Debug, Clone, Copy)]
pub struct EvalItem<'a> {
    /// レコード ID（推論へは渡さない）。
    pub id: &'a str,
    /// 推論へ渡す入力。
    pub input: &'a str,
    /// 正解ラベル（推論へは渡さない）。
    pub gold: &'a str,
    /// タグ（推論へは渡さない）。
    pub tags: &'a [String],
}

/// 推論関数の 1 呼び出し分の記録（生の文字列は保持しない）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArgumentRecord {
    /// 渡された引数のバイト長。
    pub byte_len: usize,
    /// 渡された引数の sha256。
    pub digest: Sha256Digest,
}

/// PoC-9 `ArgumentRecordingPredictor` 相当。推論関数をラップして引数を記録する。
pub struct ArgumentRecorder<F> {
    predict: F,
    records: Vec<ArgumentRecord>,
    limit: usize,
    overflowed: bool,
}

impl<F> ArgumentRecorder<F> {
    /// 記録上限を [`MAX_EVAL_RECORDS`] として作る。
    #[must_use]
    pub fn new(predict: F) -> Self {
        Self::with_limit(predict, MAX_EVAL_RECORDS)
    }

    fn with_limit(predict: F, limit: usize) -> Self {
        Self {
            predict,
            records: Vec::new(),
            limit,
            overflowed: false,
        }
    }

    /// 呼び出しごとの記録（呼び出し順）。
    #[must_use]
    pub fn records(&self) -> &[ArgumentRecord] {
        &self.records
    }

    /// 記録上限を超えて呼び出されたか。
    #[must_use]
    pub fn overflowed(&self) -> bool {
        self.overflowed
    }

    /// 引数を記録してから推論関数を呼ぶ。
    pub fn call<P>(&mut self, input: &str) -> P
    where
        F: FnMut(&str) -> P,
    {
        if self.records.len() < self.limit {
            self.records.push(ArgumentRecord {
                byte_len: input.len(),
                digest: Sha256Digest::of_bytes(input.as_bytes()),
            });
        } else {
            self.overflowed = true;
        }
        (self.predict)(input)
    }
}

/// 引数照合の違反。実値（`id`・`input`・`gold`・`tags`）は含まない。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum InputIsolationViolation {
    /// 呼び出し回数がレコード件数と一致しない。
    CallCountMismatch {
        /// 期待した呼び出し回数（レコード件数）。
        expected: usize,
        /// 実際の呼び出し回数。
        actual: usize,
    },
    /// 記録上限を超えて呼び出された。
    RecorderOverflow {
        /// 記録上限。
        limit: usize,
    },
    /// 引数が対応するレコードの `input` と一致しない index の一覧（昇順）。
    ArgumentMismatch {
        /// 不一致のレコード index。
        indices: Vec<usize>,
    },
}

impl std::fmt::Display for InputIsolationViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CallCountMismatch { expected, actual } => write!(
                f,
                "inference call count {actual} does not match record count {expected}"
            ),
            Self::RecorderOverflow { limit } => {
                write!(f, "inference calls exceeded recorder limit {limit}")
            }
            Self::ArgumentMismatch { indices } => write!(
                f,
                "{} inference call(s) received an argument other than the record input",
                indices.len()
            ),
        }
    }
}

impl std::error::Error for InputIsolationViolation {}

/// [`run_inference_input_only`] のエラー。
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum InputIsolationError {
    /// レコード件数が上限を超えた（確保前に拒否）。
    TooManyRecords {
        /// 渡された件数。
        count: usize,
        /// 上限。
        limit: usize,
    },
    /// 引数照合に違反があった（予測は返さない。fail-closed）。
    Violation(InputIsolationViolation),
}

impl std::fmt::Display for InputIsolationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooManyRecords { count, limit } => {
                write!(f, "record count {count} exceeds limit {limit}")
            }
            Self::Violation(v) => write!(f, "input isolation violated: {v}"),
        }
    }
}

impl std::error::Error for InputIsolationError {}

/// 記録器の記録が `items` の `input` と順序込みで完全一致することを照合する。
///
/// 同一 `input` が複数あっても順序ベースなので誤検出しない。上限超過・件数不一致は
/// 引数不一致より優先して報告する（fail-closed）。
///
/// # Errors
///
/// 違反があれば [`InputIsolationViolation`] を返す。
pub fn verify_input_only<F>(
    items: &[EvalItem<'_>],
    recorder: &ArgumentRecorder<F>,
) -> Result<(), InputIsolationViolation> {
    if recorder.overflowed {
        return Err(InputIsolationViolation::RecorderOverflow {
            limit: recorder.limit,
        });
    }
    let actual = recorder.records.len();
    if actual != items.len() {
        return Err(InputIsolationViolation::CallCountMismatch {
            expected: items.len(),
            actual,
        });
    }
    let indices: Vec<usize> = items
        .iter()
        .zip(recorder.records.iter())
        .enumerate()
        .filter(|(_, (item, rec))| {
            rec.byte_len != item.input.len()
                || rec.digest != Sha256Digest::of_bytes(item.input.as_bytes())
        })
        .map(|(i, _)| i)
        .collect();
    if indices.is_empty() {
        Ok(())
    } else {
        Err(InputIsolationViolation::ArgumentMismatch { indices })
    }
}

/// 各レコードの `input` だけを推論関数へ渡し、レコード順の予測を返す。
///
/// 記録器越しに呼び、最後に [`verify_input_only`] で照合する。違反があれば
/// 予測を返さない（fail-closed）。`id` との対応付けは呼び出し側が添字で行う。
///
/// 注意: この関数は常に `item.input` だけを渡すため、この関数経由では内部の照合は
/// 恒真になる（実装済みを装わない）。違反検出が実効性を持つのは、呼び出し側が
/// [`ArgumentRecorder`] を直接駆動して [`verify_input_only`] へ渡す経路であり、
/// CLI 配線（issue #140）でその形になる想定（REQ-27）。
///
/// # Errors
///
/// 件数が [`MAX_EVAL_RECORDS`] 超なら [`InputIsolationError::TooManyRecords`]、
/// 照合違反なら [`InputIsolationError::Violation`]。
pub fn run_inference_input_only<P, F>(
    items: &[EvalItem<'_>],
    predict: F,
) -> Result<Vec<P>, InputIsolationError>
where
    F: FnMut(&str) -> P,
{
    run_with_limit(items, predict, MAX_EVAL_RECORDS)
}

fn run_with_limit<P, F>(
    items: &[EvalItem<'_>],
    predict: F,
    limit: usize,
) -> Result<Vec<P>, InputIsolationError>
where
    F: FnMut(&str) -> P,
{
    if items.len() > limit {
        return Err(InputIsolationError::TooManyRecords {
            count: items.len(),
            limit,
        });
    }
    let mut recorder = ArgumentRecorder::with_limit(predict, limit);
    let mut preds = Vec::with_capacity(items.len());
    for item in items {
        preds.push(recorder.call(item.input));
    }
    verify_input_only(items, &recorder).map_err(InputIsolationError::Violation)?;
    Ok(preds)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn items<'a>(inputs: &[&'a str], tags: &'a [String]) -> Vec<EvalItem<'a>> {
        inputs
            .iter()
            .map(|s| EvalItem {
                id: "id",
                input: s,
                gold: "g",
                tags,
            })
            .collect()
    }

    /// REQ-27: 記録器は長さと sha256 だけを残す。
    #[test]
    fn req27_recorder_records_len_and_digest() {
        let mut r = ArgumentRecorder::new(|s: &str| s.len());
        assert_eq!(r.call("abc"), 3);
        assert_eq!(r.call(""), 0);
        assert_eq!(r.records()[0].byte_len, 3);
        assert_eq!(r.records()[0].digest, Sha256Digest::of_bytes(b"abc"));
        assert_eq!(r.records()[1].byte_len, 0);
        assert!(!r.overflowed());
    }

    /// REQ-27: 記録上限の境界（上限ちょうどは可、超過で overflow）。
    #[test]
    fn req27_recorder_overflow_boundary() {
        let mut r = ArgumentRecorder::with_limit(|_: &str| 0u8, 2);
        r.call("a");
        r.call("b");
        assert!(!r.overflowed());
        r.call("c");
        assert!(r.overflowed());
        assert_eq!(r.records().len(), 2);
    }

    /// REQ-27: 件数上限超過は確保前に拒否する。
    #[test]
    fn req27_too_many_records_rejected() {
        let tags: Vec<String> = vec![];
        let it = items(&["a", "b", "c"], &tags);
        let e = run_with_limit(&it, |_: &str| 0u8, 2).unwrap_err();
        assert_eq!(
            e,
            InputIsolationError::TooManyRecords { count: 3, limit: 2 }
        );
        assert_eq!(e.to_string(), "record count 3 exceeds limit 2");
    }

    /// REQ-27: 違反の Display は実値を含まない。
    #[test]
    fn req27_display_has_no_values() {
        let v = InputIsolationViolation::ArgumentMismatch {
            indices: vec![3, 7],
        };
        assert_eq!(
            v.to_string(),
            "2 inference call(s) received an argument other than the record input"
        );
        let c = InputIsolationViolation::CallCountMismatch {
            expected: 20,
            actual: 19,
        };
        assert_eq!(
            c.to_string(),
            "inference call count 19 does not match record count 20"
        );
    }
}
