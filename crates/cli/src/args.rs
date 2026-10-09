//! 7 工程サブコマンドの引数定義とパーサ（REQ-33・TASK-33.1-1）。
//!
//! # 役割と呼び出し文脈
//!
//! CLI バイナリ（`main.rs`）が `args_os()` から取り出した引数列を [`parse`]
//! に渡し、型付きの [`Invocation`] を受け取る。本モジュールは「文字列を型に
//! 変える」ことだけを担い、ファイルを開かない・工程を実行しない。下位層
//! （core / data / train / eval / runtime）への接続と 7 工程の完走は
//! TASK-33.1-2（#136）、経路の閉じ込め・資源上限などのガード層は REQ-39
//! （TASK-39.x）、`--text` の長さ検証は `fandhe-edge-core` の `InferInput`
//! 側の責務で、いずれも本モジュールでは行わない（二重実装しない）。
//!
//! # 設計
//!
//! - オプション表（[`options`]）を 1 箇所に置き、パーサの受理判定と
//!   [`render_help`] の両方がここを参照する。help とパーサの食い違いを構造的
//!   に防ぐ。
//! - 値を取るオプションは次のトークンを無条件に消費する（`--text` の値が
//!   `--` で始まっても値として扱う）。`--key=value` 形式も受理する。
//! - 重複・未知のオプション・余分な位置引数は黙殺せず拒否する（fail-closed）。
//! - `--text` と `--input-file` は [`InferSource`] の enum で排他を型に表す。
//! - argv 経路では `unwrap` / `expect` / 添字アクセスを使わない。パス値は
//!   `OsString` から損失なく `PathBuf` へ変換し（`--key=value` 形式でも同様）、
//!   それ以外は UTF-8 を要求する。
//! - [`ArgsError`] の `Display` は固定の英語文で、利用者が渡したトークンや値を
//!   含めない（表に載る既知のオプション名・サブコマンド名のみ）。

use crate::project::DEFAULT_SEED;
use fandhe_edge_core::exitcode::{ErrorReport, ExitCode};
use fandhe_edge_train::search::SearchBudget;
use std::ffi::OsString;
use std::fmt;
use std::path::PathBuf;

/// 7 工程のサブコマンド（REQ-33。工程順に並べる）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Subcommand {
    Register,
    Inspect,
    Train,
    Evaluate,
    Select,
    Package,
    Infer,
}

impl Subcommand {
    /// 工程順の全サブコマンド。
    pub const ALL: [Subcommand; 7] = [
        Subcommand::Register,
        Subcommand::Inspect,
        Subcommand::Train,
        Subcommand::Evaluate,
        Subcommand::Select,
        Subcommand::Package,
        Subcommand::Infer,
    ];

    /// コマンドラインで使う名前。
    pub const fn name(self) -> &'static str {
        match self {
            Subcommand::Register => "register",
            Subcommand::Inspect => "inspect",
            Subcommand::Train => "train",
            Subcommand::Evaluate => "evaluate",
            Subcommand::Select => "select",
            Subcommand::Package => "package",
            Subcommand::Infer => "infer",
        }
    }

    fn from_name(name: &str) -> Option<Subcommand> {
        Subcommand::ALL.into_iter().find(|s| s.name() == name)
    }

    const fn summary(self) -> &'static str {
        match self {
            Subcommand::Register => "Register a definition file into a project directory",
            Subcommand::Inspect => "Inspect the training data of a project",
            Subcommand::Train => "Train a candidate model",
            Subcommand::Evaluate => "Evaluate a trained candidate on the frozen evaluation data",
            Subcommand::Select => "Select a candidate",
            Subcommand::Package => "Build a distribution package",
            Subcommand::Infer => "Run inference with a package",
        }
    }
}

/// オプション 1 件の定義。パーサと help の単一の情報源。
#[derive(Debug, Clone, Copy)]
pub struct OptSpec {
    /// `--` 付きのオプション名。
    pub name: &'static str,
    /// 値のプレースホルダ。`None` は値を取らないフラグ。
    pub value: Option<&'static str>,
    /// 必須か（`infer` の `--text` / `--input-file` は排他必須のため表上は任意）。
    pub required: bool,
    /// help 用の 1 行説明（英語）。
    pub help: &'static str,
}

const fn opt(
    name: &'static str,
    value: &'static str,
    required: bool,
    help: &'static str,
) -> OptSpec {
    OptSpec {
        name,
        value: Some(value),
        required,
        help,
    }
}

const REGISTER_OPTS: &[OptSpec] = &[
    opt("--definition", "PATH", true, "Path to the definition file"),
    opt("--project-dir", "DIR", true, "Project directory"),
];
const PROJECT_OPTS: &[OptSpec] = &[opt("--project-dir", "DIR", true, "Project directory")];
const PACKAGE_OPTS: &[OptSpec] = &[
    opt("--project-dir", "DIR", true, "Project directory"),
    OptSpec {
        name: "--allow-smoke",
        value: None,
        required: false,
        help: "Allow packaging a smoke-trained candidate (for verification only; not for distribution)",
    },
];
const INSPECT_OPTS: &[OptSpec] = &[
    opt("--project-dir", "DIR", true, "Project directory"),
    opt(
        "--seed",
        "N",
        false,
        "Project seed for the split and training (u32, default 42)",
    ),
];
const TRAIN_OPTS: &[OptSpec] = &[
    opt("--project-dir", "DIR", true, "Project directory"),
    opt(
        "--candidate",
        "N",
        false,
        "Candidate number (exactly one of --candidate / --all)",
    ),
    OptSpec {
        name: "--all",
        value: None,
        required: false,
        help: "Train all default candidates within the search budget (exactly one of --candidate / --all)",
    },
    opt(
        "--budget-seconds",
        "N",
        false,
        "Search budget in seconds for --all (1 to 921600, default 3600)",
    ),
    OptSpec {
        name: "--smoke",
        value: None,
        required: false,
        help: "Run a short smoke training",
    },
    opt(
        "--train-seed",
        "N",
        false,
        "Training seed override (u32, default: the seed recorded in split.json; the split is unchanged)",
    ),
];
const EVALUATE_OPTS: &[OptSpec] = &[
    opt("--project-dir", "DIR", true, "Project directory"),
    opt("--candidate", "N", true, "Candidate number"),
];
const INFER_OPTS: &[OptSpec] = &[
    opt("--package", "PATH", true, "Path to the package"),
    opt(
        "--text",
        "TEXT",
        false,
        "Input text (exactly one of --text / --input-file)",
    ),
    opt(
        "--input-file",
        "PATH",
        false,
        "Input file (exactly one of --text / --input-file)",
    ),
    opt("--id", "ID", false, "Input id (only with --text)"),
    opt(
        "--out",
        "PATH",
        false,
        "Output path (only with --input-file)",
    ),
];

/// サブコマンドが受理するオプションの表。
pub const fn options(sub: Subcommand) -> &'static [OptSpec] {
    match sub {
        Subcommand::Register => REGISTER_OPTS,
        Subcommand::Inspect => INSPECT_OPTS,
        Subcommand::Select => PROJECT_OPTS,
        Subcommand::Package => PACKAGE_OPTS,
        Subcommand::Train => TRAIN_OPTS,
        Subcommand::Evaluate => EVALUATE_OPTS,
        Subcommand::Infer => INFER_OPTS,
    }
}

/// `register` の引数。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisterArgs {
    pub definition: PathBuf,
    pub project_dir: PathBuf,
}
/// `inspect` の引数。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InspectArgs {
    pub project_dir: PathBuf,
    /// プロジェクトの seed（分割と学習で共通。`split.json` に記録され、`train` はその値を使う。
    /// 既定は [`DEFAULT_SEED`]。REQ-17）。
    pub seed: u32,
}
/// `train` の対象。`--candidate` と `--all` の排他を型で表す（REQ-18・#482）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainTarget {
    /// `--candidate N`: 候補 1 件を学習する。
    Candidate(usize),
    /// `--all [--budget-seconds N]`: 既定候補の全件を探索予算内で学習する（持ち時間は均等割り固定）。
    All { budget: SearchBudget },
}

/// `train` の引数。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainArgs {
    pub project_dir: PathBuf,
    pub target: TrainTarget,
    /// `--smoke`（`--all` では全候補に適用する）。
    pub smoke: bool,
    /// `--train-seed`: 学習 seed の上書き（省略時は `split.json` の seed。分割は変えない。REQ-17・REQ-41）。
    pub train_seed: Option<u32>,
}
/// `evaluate` の引数。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvaluateArgs {
    pub project_dir: PathBuf,
    pub candidate: usize,
}
/// `select` の引数。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectArgs {
    pub project_dir: PathBuf,
}
/// `package` の引数。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackageArgs {
    pub project_dir: PathBuf,
    /// `--allow-smoke`: `train --smoke` で短縮学習した候補の package を許す（検証専用。配布用ではない。
    /// 指定しなければ拒否する。REQ-27）。
    pub allow_smoke: bool,
}

/// `infer` の入力源。同時指定・不整合な組み合わせを型で表現できなくする。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InferSource {
    /// `--text`（任意で `--id`）。
    Text { text: String, id: Option<String> },
    /// `--input-file`（任意で `--out`）。1 行 1 JSON の一括推論は `infer_batch` モジュール（TASK-33.4）。`--out` は結果行を
    /// ファイルへ書き stdout に要約を出す（#459）。
    InputFile { path: PathBuf, out: Option<PathBuf> },
}

/// `infer` の引数。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferArgs {
    pub package: PathBuf,
    pub source: InferSource,
}

/// 解析済みのコマンド。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    Register(RegisterArgs),
    Inspect(InspectArgs),
    Train(TrainArgs),
    Evaluate(EvaluateArgs),
    Select(SelectArgs),
    Package(PackageArgs),
    Infer(InferArgs),
}

/// 解析結果。`Help(None)` はトップレベルの help。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Invocation {
    Run(Command),
    Help(Option<Subcommand>),
}

/// 引数エラー。すべて REQ-21 の `invalid_input`（64）に写る。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArgsError {
    NoSubcommand,
    UnknownSubcommand,
    UnknownOption {
        subcommand: Subcommand,
    },
    UnexpectedPositional {
        subcommand: Subcommand,
    },
    MissingValue {
        option: &'static str,
    },
    FlagTakesNoValue {
        option: &'static str,
    },
    DuplicateOption {
        option: &'static str,
    },
    MissingRequired {
        option: &'static str,
    },
    InvalidCandidate,
    /// `--seed` が `u32` の範囲の非負整数でない。
    InvalidSeed,
    /// `--train-seed` が `u32` の範囲の非負整数でない。
    InvalidTrainSeed,
    /// `--budget-seconds` が `1..=MAX_SEARCH_BUDGET_SECONDS` の整数でない。
    InvalidBudgetSeconds,
    /// `train` に `--candidate` と `--all` の両方が指定された。
    ConflictingTrainTarget,
    /// `train` に `--candidate` も `--all` も指定されていない。
    MissingTrainTarget,
    ConflictingInferSource,
    MissingInferSource,
    /// `--id` は `--text` とだけ、`--out` は `--input-file` とだけ併用できる。
    IncompatibleOption {
        option: &'static str,
    },
    /// パス値（`PATH` / `DIR`）が空文字列。空の `PathBuf` を後段へ渡さない。
    EmptyValue {
        option: &'static str,
    },
    NonUtf8Argument,
}

impl ArgsError {
    /// 常に `invalid_input`（REQ-21）。
    pub const fn exit_code(&self) -> ExitCode {
        ExitCode::InvalidInput
    }
}

impl fmt::Display for ArgsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArgsError::NoSubcommand => f.write_str("a subcommand is required"),
            ArgsError::UnknownSubcommand => f.write_str("unknown subcommand"),
            ArgsError::UnknownOption { subcommand } => {
                write!(f, "unknown option for subcommand {}", subcommand.name())
            }
            ArgsError::UnexpectedPositional { subcommand } => write!(
                f,
                "unexpected positional argument for subcommand {}",
                subcommand.name()
            ),
            ArgsError::MissingValue { option } => write!(f, "option {option} requires a value"),
            ArgsError::FlagTakesNoValue { option } => {
                write!(f, "option {option} does not take a value")
            }
            ArgsError::DuplicateOption { option } => {
                write!(f, "option {option} was specified more than once")
            }
            ArgsError::MissingRequired { option } => {
                write!(f, "required option {option} is missing")
            }
            ArgsError::InvalidCandidate => {
                f.write_str("option --candidate must be a non-negative integer")
            }
            ArgsError::InvalidSeed => {
                f.write_str("option --seed must be an integer in the range 0 to 4294967295")
            }
            ArgsError::InvalidTrainSeed => {
                f.write_str("option --train-seed must be an integer in the range 0 to 4294967295")
            }
            ArgsError::InvalidBudgetSeconds => {
                f.write_str("option --budget-seconds must be an integer in the range 1 to 921600")
            }
            ArgsError::ConflictingTrainTarget => {
                f.write_str("options --candidate and --all cannot be used together")
            }
            ArgsError::MissingTrainTarget => f.write_str("one of --candidate or --all is required"),
            ArgsError::ConflictingInferSource => {
                f.write_str("options --text and --input-file cannot be used together")
            }
            ArgsError::MissingInferSource => {
                f.write_str("one of --text or --input-file is required")
            }
            ArgsError::IncompatibleOption { option } => {
                write!(
                    f,
                    "option {option} cannot be used with the chosen input source"
                )
            }
            ArgsError::EmptyValue { option } => {
                write!(f, "option {option} requires a non-empty value")
            }
            ArgsError::NonUtf8Argument => f.write_str("argument is not valid UTF-8"),
        }
    }
}

impl std::error::Error for ArgsError {}

/// [`ArgsError`] を `{"code","message"}` の `ErrorReport` へ変換する（`output.rs`
/// の `*_error_report` と対称。`message` は固定文で利用者の値を含まない）。
pub fn args_error_report(err: &ArgsError) -> ErrorReport {
    ErrorReport::new(err.exit_code(), err.to_string())
}

fn is_help(tok: &OsString) -> bool {
    tok.to_str().is_some_and(|s| s == "-h" || s == "--help")
}

/// argv（プログラム名を除く）を解析する。
pub fn parse<I: IntoIterator<Item = OsString>>(args: I) -> Result<Invocation, ArgsError> {
    let mut it = args.into_iter();
    let first = it.next().ok_or(ArgsError::NoSubcommand)?;
    if is_help(&first) {
        return Ok(Invocation::Help(None));
    }
    let name = first.to_str().ok_or(ArgsError::NonUtf8Argument)?;
    let sub = Subcommand::from_name(name).ok_or(ArgsError::UnknownSubcommand)?;
    let rest: Vec<OsString> = it.collect();
    let specs = options(sub);

    // help は他の検証より優先する。値として消費されるトークンは除外して走査する。
    let mut scan = rest.iter();
    while let Some(tok) = scan.next() {
        if is_help(tok) {
            return Ok(Invocation::Help(Some(sub)));
        }
        if let Some(s) = tok.to_str()
            && !s.contains('=')
            && specs.iter().any(|o| o.name == s && o.value.is_some())
        {
            scan.next();
        }
    }

    let mut values: Vec<(&'static str, OsString)> = Vec::new();
    let mut toks = rest.into_iter();
    while let Some(tok) = toks.next() {
        let (key, inline) = split_option(&tok, sub)?;
        let spec = specs
            .iter()
            .find(|o| o.name == key.as_str())
            .ok_or(ArgsError::UnknownOption { subcommand: sub })?;
        if values.iter().any(|(n, _)| *n == spec.name) {
            return Err(ArgsError::DuplicateOption { option: spec.name });
        }
        let value = if spec.value.is_some() {
            match inline {
                Some(v) => v,
                None => toks
                    .next()
                    .ok_or(ArgsError::MissingValue { option: spec.name })?,
            }
        } else if inline.is_some() {
            return Err(ArgsError::FlagTakesNoValue { option: spec.name });
        } else {
            OsString::new()
        };
        if matches!(spec.value, Some("PATH" | "DIR")) && value.is_empty() {
            return Err(ArgsError::EmptyValue { option: spec.name });
        }
        values.push((spec.name, value));
    }

    for spec in specs.iter().filter(|o| o.required) {
        if !values.iter().any(|(n, _)| *n == spec.name) {
            return Err(ArgsError::MissingRequired { option: spec.name });
        }
    }
    build(sub, values).map(Invocation::Run)
}

/// トークンを `--key` と `=` 以降の値に分ける。キーは UTF-8 を要求し、値は
/// `OsString` のまま（unix ではバイト列レベルで分割）返す。空白区切り形式と
/// `--key=value` 形式でパス値の扱いを揃えるため。
fn split_option(tok: &OsString, sub: Subcommand) -> Result<(String, Option<OsString>), ArgsError> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::{OsStrExt, OsStringExt};
        let bytes = tok.as_bytes();
        if !bytes.starts_with(b"--") {
            // 位置引数。非 UTF-8 の位置引数は従来どおり NonUtf8Argument とする。
            return match tok.to_str() {
                Some(_) => Err(ArgsError::UnexpectedPositional { subcommand: sub }),
                None => Err(ArgsError::NonUtf8Argument),
            };
        }
        let (k, v) = match bytes.iter().position(|b| *b == b'=') {
            Some(i) => (
                bytes.get(..i).unwrap_or_default(),
                Some(OsString::from_vec(
                    bytes.get(i + 1..).unwrap_or_default().to_vec(),
                )),
            ),
            None => (bytes, None),
        };
        let key = std::str::from_utf8(k).map_err(|_| ArgsError::NonUtf8Argument)?;
        Ok((key.to_string(), v))
    }
    #[cfg(not(unix))]
    {
        // UTF-16 単位で `=` を探し、キー部分だけ UTF-8 を要求する。値は
        // `OsString` のまま保持し、非 UTF-8 のパスを損失なく受理する。
        use std::os::windows::ffi::{OsStrExt, OsStringExt};
        let wide: Vec<u16> = tok.encode_wide().collect();
        let dashes = [u16::from(b'-'), u16::from(b'-')];
        if !wide.starts_with(&dashes) {
            return match tok.to_str() {
                Some(_) => Err(ArgsError::UnexpectedPositional { subcommand: sub }),
                None => Err(ArgsError::NonUtf8Argument),
            };
        }
        let (k, v) = match wide.iter().position(|c| *c == u16::from(b'=')) {
            Some(i) => (
                wide.get(..i).unwrap_or_default(),
                Some(OsString::from_wide(wide.get(i + 1..).unwrap_or_default())),
            ),
            None => (wide.as_slice(), None),
        };
        let key = String::from_utf16(k).map_err(|_| ArgsError::NonUtf8Argument)?;
        Ok((key, v))
    }
}

/// 値の取り出し手。表で必須検証済みのため、欠落は `MissingRequired` で返す。
struct Values(Vec<(&'static str, OsString)>);

impl Values {
    fn take(&mut self, name: &'static str) -> Option<OsString> {
        let pos = self.0.iter().position(|(n, _)| *n == name)?;
        Some(self.0.swap_remove(pos).1)
    }
    fn path(&mut self, name: &'static str) -> Result<PathBuf, ArgsError> {
        self.take(name)
            .map(PathBuf::from)
            .ok_or(ArgsError::MissingRequired { option: name })
    }
    fn opt_path(&mut self, name: &'static str) -> Option<PathBuf> {
        self.take(name).map(PathBuf::from)
    }
    fn opt_string(&mut self, name: &'static str) -> Result<Option<String>, ArgsError> {
        self.take(name)
            .map(|v| v.into_string().map_err(|_| ArgsError::NonUtf8Argument))
            .transpose()
    }
    fn seed(&mut self) -> Result<u32, ArgsError> {
        match self.take("--seed") {
            None => Ok(DEFAULT_SEED),
            Some(v) => v
                .into_string()
                .map_err(|_| ArgsError::NonUtf8Argument)?
                .parse::<u32>()
                .map_err(|_| ArgsError::InvalidSeed),
        }
    }
    fn train_seed(&mut self) -> Result<Option<u32>, ArgsError> {
        self.take("--train-seed")
            .map(|v| {
                v.into_string()
                    .map_err(|_| ArgsError::NonUtf8Argument)?
                    .parse::<u32>()
                    .map_err(|_| ArgsError::InvalidTrainSeed)
            })
            .transpose()
    }
    fn budget(&mut self) -> Result<SearchBudget, ArgsError> {
        match self.take("--budget-seconds") {
            None => Ok(SearchBudget::default()),
            Some(v) => v
                .into_string()
                .map_err(|_| ArgsError::NonUtf8Argument)?
                .parse::<u64>()
                .ok()
                .and_then(SearchBudget::new)
                .ok_or(ArgsError::InvalidBudgetSeconds),
        }
    }
    fn train_target(&mut self) -> Result<TrainTarget, ArgsError> {
        let all = self.take("--all").is_some();
        let has_candidate = self.0.iter().any(|(n, _)| *n == "--candidate");
        match (has_candidate, all) {
            (true, true) => Err(ArgsError::ConflictingTrainTarget),
            (false, false) => Err(ArgsError::MissingTrainTarget),
            (true, false) => {
                if self.take("--budget-seconds").is_some() {
                    return Err(ArgsError::IncompatibleOption {
                        option: "--budget-seconds",
                    });
                }
                Ok(TrainTarget::Candidate(self.candidate()?))
            }
            (false, true) => Ok(TrainTarget::All {
                budget: self.budget()?,
            }),
        }
    }
    fn candidate(&mut self) -> Result<usize, ArgsError> {
        let v = self
            .take("--candidate")
            .ok_or(ArgsError::MissingRequired {
                option: "--candidate",
            })?
            .into_string()
            .map_err(|_| ArgsError::NonUtf8Argument)?;
        v.parse::<usize>().map_err(|_| ArgsError::InvalidCandidate)
    }
}

fn build(sub: Subcommand, values: Vec<(&'static str, OsString)>) -> Result<Command, ArgsError> {
    let mut v = Values(values);
    Ok(match sub {
        Subcommand::Register => Command::Register(RegisterArgs {
            definition: v.path("--definition")?,
            project_dir: v.path("--project-dir")?,
        }),
        Subcommand::Inspect => Command::Inspect(InspectArgs {
            project_dir: v.path("--project-dir")?,
            seed: v.seed()?,
        }),
        Subcommand::Train => Command::Train(TrainArgs {
            project_dir: v.path("--project-dir")?,
            target: v.train_target()?,
            smoke: v.take("--smoke").is_some(),
            train_seed: v.train_seed()?,
        }),
        Subcommand::Evaluate => Command::Evaluate(EvaluateArgs {
            project_dir: v.path("--project-dir")?,
            candidate: v.candidate()?,
        }),
        Subcommand::Select => Command::Select(SelectArgs {
            project_dir: v.path("--project-dir")?,
        }),
        Subcommand::Package => Command::Package(PackageArgs {
            project_dir: v.path("--project-dir")?,
            allow_smoke: v.take("--allow-smoke").is_some(),
        }),
        Subcommand::Infer => {
            let package = v.path("--package")?;
            let text = v.opt_string("--text")?;
            let input_file = v.opt_path("--input-file");
            let id = v.opt_string("--id")?;
            let out = v.opt_path("--out");
            let source = match (text, input_file) {
                (Some(_), Some(_)) => return Err(ArgsError::ConflictingInferSource),
                (None, None) => return Err(ArgsError::MissingInferSource),
                (Some(text), None) => {
                    if out.is_some() {
                        return Err(ArgsError::IncompatibleOption { option: "--out" });
                    }
                    InferSource::Text { text, id }
                }
                (None, Some(path)) => {
                    if id.is_some() {
                        return Err(ArgsError::IncompatibleOption { option: "--id" });
                    }
                    InferSource::InputFile { path, out }
                }
            };
            Command::Infer(InferArgs { package, source })
        }
    })
}

/// help テキスト（英語の固定文）を表から生成する。
pub fn render_help(topic: Option<Subcommand>) -> String {
    let mut s = String::new();
    match topic {
        None => {
            s.push_str("Usage: fandhe-edge <SUBCOMMAND> [OPTIONS]\n\nSubcommands:\n");
            for sub in Subcommand::ALL {
                s.push_str(&format!("  {:<10}{}\n", sub.name(), sub.summary()));
            }
            s.push_str(
                "\nRun `fandhe-edge <SUBCOMMAND> --help` for the options of a subcommand.\n",
            );
        }
        Some(sub) => {
            s.push_str(&format!(
                "{}\n\nUsage: fandhe-edge {} [OPTIONS]\n\nOptions:\n",
                sub.summary(),
                sub.name()
            ));
            for o in options(sub) {
                let head = match o.value {
                    Some(p) => format!("{} <{}>", o.name, p),
                    None => o.name.to_string(),
                };
                let req = if o.required { "required" } else { "optional" };
                s.push_str(&format!("  {head:<26}[{req}] {}\n", o.help));
            }
            s.push_str("  -h, --help                Print this help\n");
        }
    }
    s
}

#[cfg(test)]
mod tests {
    //! REQ-33・TASK-33.1-1 のパーサ検証。
    use super::*;

    fn p(args: &[&str]) -> Result<Invocation, ArgsError> {
        parse(args.iter().map(OsString::from))
    }
    fn run(args: &[&str]) -> Command {
        match p(args) {
            Ok(Invocation::Run(c)) => c,
            other => panic!("expected Run, got {other:?}"),
        }
    }

    #[test]
    fn req33_register_parses() {
        assert_eq!(
            run(&["register", "--definition", "d.json", "--project-dir=proj"]),
            Command::Register(RegisterArgs {
                definition: "d.json".into(),
                project_dir: "proj".into()
            })
        );
    }

    #[test]
    fn req33_project_only_subcommands_parse() {
        let d = PathBuf::from("proj");
        assert_eq!(
            run(&["inspect", "--project-dir", "proj"]),
            Command::Inspect(InspectArgs {
                project_dir: d.clone(),
                seed: 42
            })
        );
        assert_eq!(
            run(&["select", "--project-dir", "proj"]),
            Command::Select(SelectArgs {
                project_dir: d.clone()
            })
        );
        assert_eq!(
            run(&["package", "--project-dir", "proj"]),
            Command::Package(PackageArgs {
                project_dir: d,
                allow_smoke: false
            })
        );
    }

    /// REQ-17: `inspect --seed` は u32 の範囲で受理し、省略時は 42。範囲外・非数値は `InvalidSeed`。
    #[test]
    fn req17_inspect_seed_parses_with_default_and_range() {
        let seeded = |v: &str| p(&["inspect", "--project-dir", "proj", "--seed", v]);
        assert_eq!(
            run(&["inspect", "--project-dir", "proj", "--seed", "7"]),
            Command::Inspect(InspectArgs {
                project_dir: PathBuf::from("proj"),
                seed: 7
            })
        );
        assert_eq!(
            run(&["inspect", "--project-dir", "proj"]),
            Command::Inspect(InspectArgs {
                project_dir: PathBuf::from("proj"),
                seed: 42
            })
        );
        assert_eq!(
            run(&["inspect", "--project-dir", "proj", "--seed=4294967295"]),
            Command::Inspect(InspectArgs {
                project_dir: PathBuf::from("proj"),
                seed: u32::MAX
            })
        );
        assert_eq!(seeded("4294967296"), Err(ArgsError::InvalidSeed));
        assert_eq!(seeded("-1"), Err(ArgsError::InvalidSeed));
        assert_eq!(seeded("abc"), Err(ArgsError::InvalidSeed));
        assert_eq!(ArgsError::InvalidSeed.exit_code(), ExitCode::InvalidInput);
    }

    /// REQ-27・REQ-33: `package --allow-smoke` はフラグ（値なし）で、省略時は false。
    #[test]
    fn req27_package_allow_smoke_flag_parses() {
        let d = PathBuf::from("proj");
        assert_eq!(
            run(&["package", "--project-dir", "proj", "--allow-smoke"]),
            Command::Package(PackageArgs {
                project_dir: d,
                allow_smoke: true
            })
        );
        assert_eq!(
            err(&["package", "--project-dir", "proj", "--allow-smoke=1"]),
            ArgsError::FlagTakesNoValue {
                option: "--allow-smoke"
            }
        );
    }

    /// REQ-17・REQ-41: `train --train-seed` は u32 の範囲で受理し、省略時は `None`。範囲外・負数・非数は
    /// `InvalidTrainSeed`（`invalid_input`）。
    #[test]
    fn req17_train_seed_parses_with_range() {
        let t = |v: &str| {
            p(&[
                "train",
                "--project-dir",
                "proj",
                "--candidate",
                "0",
                "--train-seed",
                v,
            ])
        };
        let seed_of = |v: &str| match t(v) {
            Ok(Invocation::Run(Command::Train(a))) => a.train_seed,
            other => panic!("unexpected {other:?}"),
        };
        assert_eq!(seed_of("2"), Some(2));
        assert_eq!(seed_of("4294967295"), Some(u32::MAX));
        assert_eq!(t("4294967296"), Err(ArgsError::InvalidTrainSeed));
        assert_eq!(t("-1"), Err(ArgsError::InvalidTrainSeed));
        assert_eq!(t("abc"), Err(ArgsError::InvalidTrainSeed));
        assert_eq!(
            ArgsError::InvalidTrainSeed.exit_code(),
            ExitCode::InvalidInput
        );
    }

    #[test]
    fn req33_train_and_evaluate_parse() {
        assert_eq!(
            run(&[
                "train",
                "--project-dir",
                "proj",
                "--candidate",
                "2",
                "--smoke"
            ]),
            Command::Train(TrainArgs {
                project_dir: "proj".into(),
                target: TrainTarget::Candidate(2),
                smoke: true,
                train_seed: None
            })
        );
        assert_eq!(
            run(&["train", "--candidate=0", "--project-dir", "proj"]),
            Command::Train(TrainArgs {
                project_dir: "proj".into(),
                target: TrainTarget::Candidate(0),
                smoke: false,
                train_seed: None
            })
        );
        assert_eq!(
            run(&["evaluate", "--project-dir", "proj", "--candidate", "3"]),
            Command::Evaluate(EvaluateArgs {
                project_dir: "proj".into(),
                candidate: 3
            })
        );
    }

    /// REQ-18・#482: `train --all` は `--budget-seconds` を `1..=921600`（既定 3600）で受理し、
    /// `--smoke`・`--train-seed` と併用できる。`--candidate` との併用・どちらも無し・範囲外の予算・
    /// `--candidate` と `--budget-seconds` の併用はいずれも `invalid_input`（64）。
    #[test]
    fn req18_train_all_parses_and_rejects_conflicts() {
        let all = |extra: &[&str]| {
            let mut a = vec!["train", "--project-dir", "proj", "--all"];
            a.extend_from_slice(extra);
            p(&a)
        };
        let target = |r: Result<Invocation, ArgsError>| match r {
            Ok(Invocation::Run(Command::Train(a))) => Ok(a.target),
            Err(e) => Err(e),
            other => panic!("unexpected {other:?}"),
        };
        let budget = |s: u64| TrainTarget::All {
            budget: SearchBudget::new(s).expect("budget"),
        };
        assert_eq!(target(all(&[])), Ok(budget(3600)));
        assert_eq!(target(all(&["--budget-seconds", "1"])), Ok(budget(1)));
        assert_eq!(
            target(all(&["--budget-seconds=921600"])),
            Ok(budget(921_600))
        );
        assert_eq!(
            run(&[
                "train",
                "--project-dir",
                "proj",
                "--all",
                "--smoke",
                "--train-seed",
                "7"
            ]),
            Command::Train(TrainArgs {
                project_dir: "proj".into(),
                target: budget(3600),
                smoke: true,
                train_seed: Some(7)
            })
        );
        for bad in ["0", "921601", "-1", "abc", "18446744073709551616"] {
            assert_eq!(
                target(all(&["--budget-seconds", bad])),
                Err(ArgsError::InvalidBudgetSeconds),
                "{bad}"
            );
        }
        assert_eq!(
            target(all(&["--candidate", "0"])),
            Err(ArgsError::ConflictingTrainTarget)
        );
        assert_eq!(
            target(p(&["train", "--project-dir", "proj"])),
            Err(ArgsError::MissingTrainTarget)
        );
        assert_eq!(
            target(p(&[
                "train",
                "--project-dir",
                "proj",
                "--candidate",
                "0",
                "--budget-seconds",
                "10"
            ])),
            Err(ArgsError::IncompatibleOption {
                option: "--budget-seconds"
            })
        );
        for e in [
            ArgsError::InvalidBudgetSeconds,
            ArgsError::ConflictingTrainTarget,
            ArgsError::MissingTrainTarget,
        ] {
            assert_eq!(args_error_report(&e).code, ExitCode::InvalidInput);
        }
        // 上限の表示値は型の上限と一致する（help・エラー文の 921600）。
        assert_eq!(
            fandhe_edge_train::search::MAX_SEARCH_BUDGET_SECONDS,
            921_600
        );
    }

    #[test]
    fn req33_infer_four_shapes() {
        let pkg = PathBuf::from("pkg");
        assert_eq!(
            run(&["infer", "--package", "pkg", "--text", "hello"]),
            Command::Infer(InferArgs {
                package: pkg.clone(),
                source: InferSource::Text {
                    text: "hello".into(),
                    id: None
                }
            })
        );
        assert_eq!(
            run(&["infer", "--package", "pkg", "--text", "hello", "--id", "a1"]),
            Command::Infer(InferArgs {
                package: pkg.clone(),
                source: InferSource::Text {
                    text: "hello".into(),
                    id: Some("a1".into())
                }
            })
        );
        assert_eq!(
            run(&["infer", "--package", "pkg", "--input-file", "in.jsonl"]),
            Command::Infer(InferArgs {
                package: pkg.clone(),
                source: InferSource::InputFile {
                    path: "in.jsonl".into(),
                    out: None
                }
            })
        );
        assert_eq!(
            run(&[
                "infer",
                "--package",
                "pkg",
                "--input-file",
                "in.jsonl",
                "--out",
                "o.jsonl"
            ]),
            Command::Infer(InferArgs {
                package: pkg,
                source: InferSource::InputFile {
                    path: "in.jsonl".into(),
                    out: Some("o.jsonl".into())
                }
            })
        );
    }

    #[test]
    fn req33_value_starting_with_dashes_is_consumed() {
        assert_eq!(
            run(&["infer", "--package", "pkg", "--text", "--leading"]),
            Command::Infer(InferArgs {
                package: "pkg".into(),
                source: InferSource::Text {
                    text: "--leading".into(),
                    id: None
                }
            })
        );
    }

    fn err(args: &[&str]) -> ArgsError {
        match p(args) {
            Err(e) => {
                assert_eq!(e.exit_code().code(), 64);
                e
            }
            other => panic!("expected error, got {other:?}"),
        }
    }

    #[test]
    fn req33_errors_map_to_expected_variants() {
        assert_eq!(err(&[]), ArgsError::NoSubcommand);
        assert_eq!(err(&["bogus"]), ArgsError::UnknownSubcommand);
        assert_eq!(
            err(&["inspect", "--project-dir", "p", "--nope"]),
            ArgsError::UnknownOption {
                subcommand: Subcommand::Inspect
            }
        );
        assert_eq!(
            err(&["inspect", "--project-dir"]),
            ArgsError::MissingValue {
                option: "--project-dir"
            }
        );
        // 空のパス値は invalid_input（空 PathBuf を通さない）。
        for args in [
            &["inspect", "--project-dir="][..],
            &["inspect", "--project-dir", ""][..],
            &["register", "--definition", "", "--project-dir", "p"][..],
            &["infer", "--package", "p", "--input-file="][..],
        ] {
            assert!(
                matches!(err(args), ArgsError::EmptyValue { .. }),
                "{args:?}"
            );
        }
        assert_eq!(
            err(&["inspect", "--project-dir", "a", "--project-dir", "b"]),
            ArgsError::DuplicateOption {
                option: "--project-dir"
            }
        );
        assert_eq!(
            err(&["inspect", "extra"]),
            ArgsError::UnexpectedPositional {
                subcommand: Subcommand::Inspect
            }
        );
        assert_eq!(
            err(&["inspect"]),
            ArgsError::MissingRequired {
                option: "--project-dir"
            }
        );
        assert_eq!(
            err(&["train", "--project-dir", "p", "--candidate", "abc"]),
            ArgsError::InvalidCandidate
        );
        assert_eq!(
            err(&["train", "--project-dir", "p", "--candidate", "-1"]),
            ArgsError::InvalidCandidate
        );
        assert_eq!(
            err(&[
                "train",
                "--project-dir",
                "p",
                "--candidate",
                "1",
                "--smoke=1"
            ]),
            ArgsError::FlagTakesNoValue { option: "--smoke" }
        );
        assert_eq!(
            err(&[
                "infer",
                "--package",
                "p",
                "--text",
                "a",
                "--input-file",
                "b"
            ]),
            ArgsError::ConflictingInferSource
        );
        assert_eq!(
            err(&["infer", "--package", "p"]),
            ArgsError::MissingInferSource
        );
        assert_eq!(
            err(&["infer", "--package", "p", "--input-file", "b", "--id", "x"]),
            ArgsError::IncompatibleOption { option: "--id" }
        );
        assert_eq!(
            err(&["infer", "--package", "p", "--text", "a", "--out", "x"]),
            ArgsError::IncompatibleOption { option: "--out" }
        );
    }

    #[cfg(unix)]
    #[test]
    fn req33_non_utf8_is_error_not_panic() {
        use std::os::unix::ffi::OsStrExt;
        let bad = std::ffi::OsStr::from_bytes(&[0xff]).to_os_string();
        let r = parse([
            OsString::from("infer"),
            OsString::from("--package"),
            OsString::from("p"),
            OsString::from("--text"),
            bad.clone(),
        ]);
        assert_eq!(r, Err(ArgsError::NonUtf8Argument));
        // パス値は非 UTF-8 でも損失なく受理する。
        let r = parse([
            OsString::from("inspect"),
            OsString::from("--project-dir"),
            bad.clone(),
        ]);
        assert_eq!(
            r,
            Ok(Invocation::Run(Command::Inspect(InspectArgs {
                project_dir: PathBuf::from(bad.clone()),
                seed: 42
            })))
        );
        // `--key=value` 形式でも同じく損失なく受理する。
        let mut eq = b"--project-dir=".to_vec();
        eq.push(0xff);
        let r = parse([
            OsString::from("inspect"),
            std::ffi::OsStr::from_bytes(&eq).to_os_string(),
        ]);
        assert_eq!(
            r,
            Ok(Invocation::Run(Command::Inspect(InspectArgs {
                project_dir: PathBuf::from(bad.clone()),
                seed: 42
            })))
        );
        assert_eq!(parse([bad]), Err(ArgsError::NonUtf8Argument));
    }

    #[test]
    fn req33_messages_do_not_leak_user_input() {
        const M: &str = "should-not-leak";
        let cases = [
            p(&[M]),
            p(&["inspect", &format!("--{M}")]),
            p(&["inspect", M]),
            p(&["train", "--project-dir", "p", "--candidate", M]),
            p(&["infer", "--package", M, "--text", M, "--input-file", M]),
        ];
        for c in cases {
            let e = c.expect_err("must be an error");
            assert!(!e.to_string().contains(M), "leaked: {e}");
            assert!(!args_error_report(&e).message.contains(M));
        }
    }

    #[test]
    fn req33_help_lists_all_options_and_subcommands() {
        let train = render_help(Some(Subcommand::Train));
        for o in [
            "--project-dir",
            "--candidate",
            "--all",
            "--budget-seconds",
            "--smoke",
        ] {
            assert!(train.contains(o), "{train}");
        }
        for sub in Subcommand::ALL {
            let h = render_help(Some(sub));
            for o in options(sub) {
                assert!(h.contains(o.name), "{} help lacks {}", sub.name(), o.name);
            }
        }
        let top = render_help(None);
        for sub in Subcommand::ALL {
            assert!(top.contains(sub.name()));
        }
    }

    #[test]
    fn req33_help_takes_priority_over_validation() {
        assert_eq!(
            p(&["train", "--help"]),
            Ok(Invocation::Help(Some(Subcommand::Train)))
        );
        assert_eq!(
            p(&["train", "-h"]),
            Ok(Invocation::Help(Some(Subcommand::Train)))
        );
        assert_eq!(p(&["--help"]), Ok(Invocation::Help(None)));
        assert_eq!(
            p(&["train", "--bogus", "--help"]),
            Ok(Invocation::Help(Some(Subcommand::Train)))
        );
    }
}
