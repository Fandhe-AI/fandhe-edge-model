//! `scripts/real-machine-check.sh`（Mac 実機での動作確認 A〜F の入口）と
//! `scripts/real_machine_check_record.py`（実行・伏せ処理・記録の生成）の結合テスト
//! （REQ-21・REQ-27・REQ-28・REQ-30・REQ-31・REQ-33・REQ-39・Issue #354）。
//!
//! 証拠種別: テストハーネス。偽の make・偽の cargo・偽の CLI を一時ディレクトリへ書き出して使うため、
//! **実 make・実 cargo・実 CLI・実機の測定は一切行っていない**。ここで検証するのは
//! スクリプトの制御（引数検証・項目の順序と停止・記録の構造・パスとデータ本文の伏せ・終了コード）で、
//! 実機の証拠にならない。`record.json` の `evidence_hint` も上書きの経路では `test_harness` になる。
//! 上書きなしの経路（実 make・実 cargo を呼ぶ `requires_human_review`）はここでは走らせない。
//! `FANDHE_EDGE_BIN` を空にした経路だけは、偽 cargo の `build` が報告した実行ファイルを使う。
//! `record.json` の値の検査は、標準ライブラリだけの `python3 -I` の小さな問い合わせで行う
//! （cli crate に JSON の依存を足さないため。python3 はスクリプト本体も必要とする前提で、
//! `sandbox_monitor_script.rs` と同じく無い環境の分岐は持たない）。
//! Windows では `sh` を前提にできないため unix に限定する。

#![cfg(unix)]

use std::fs;
use std::io::Read;
use std::os::unix::fs::{PermissionsExt, symlink};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use fandhe_edge_core::definition::MAX_LIMIT_INFER_P95_US;

/// 子プロセスの上限時間（資源上限。REQ-39）。
const TIMEOUT: Duration = Duration::from_secs(120);
const LEAK_BODY: &str = "SECRET_BODY_7c1";
const LEAK_ID: &str = "leakid-4711";

static SEQ: AtomicU32 = AtomicU32::new(0);

/// 偽の CLI。`FAKE_DIR` へ呼び出しを記録し、環境変数で挙動を切り替える
/// （`FAKE_FAIL_STAGE`・`FAKE_LEAK`・`FAKE_C1_EXCEED`・`FAKE_C2_MODE`・`FAKE_E_MISMATCH`）。
/// 既定の出力は実 CLI の形（`crates/core/src/stage_report.rs`）に揃え、スクリプトの判定
/// （REQ-21・REQ-28・REQ-30・REQ-31・REQ-33）を満たす。矛盾した出力は `FAKE_BAD=<種別>:<case>`
/// （case は cwd 末尾の B・C1・C2。種別は `bad` 判定の箇所を参照）で個別に作る。
/// cwd の末尾（B・C1・C2）で package の挙動を変え、定義ファイルの `limits` を読んで上限を返す。
const FAKE_CLI: &str = r##"#!/bin/sh
stage=$1
here=$(basename "$PWD")
echo "$stage $here" >> "$FAKE_DIR/cli.log"
printf '%s\n' "$*" >> "$FAKE_DIR/cli.args"
echo "${CARGO_NET_OFFLINE:-unset}" >> "$FAKE_DIR/cli.env"
[ "$stage" = register ] && cp definition.json "$FAKE_DIR/def-$here.json"
SC='{"alpha":0.5,"beta":0.25,"gamma":0.25}'
bad=${FAKE_BAD:-}
[ "$bad" = scores_high ] && SC='{"alpha":1.5,"beta":0.0,"gamma":-0.5}'
[ "$bad" = scores_neg ] && SC='{"alpha":0.75,"beta":0.5,"gamma":-0.25}'
SHA=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
p95lim=$(sed -n 's/.*"max_infer_p95_us": *\([0-9]*\).*/\1/p' definition.json 2>/dev/null)
[ -n "$p95lim" ] || p95lim=50000
pkglim=$(sed -n 's/.*"max_package_bytes": *\([0-9]*\).*/\1/p' definition.json 2>/dev/null)
[ -n "$pkglim" ] || pkglim=40000000
extra=
extra_infer=
if [ -n "${FAKE_LEAK:-}" ] && [ "$stage" != infer ]; then
  extra=",\"project_dir\":\"$PWD/project\",\"rel\":\"a/b\",\"win\":\"C:\\\\Users\\\\x\",\"input\":\"SECRET_BODY_7c1\",\"text\":\"SECRET_BODY_7c1\",\"id\":\"leakid-4711\",\"note\":\"SECRET_BODY_7c1 in $PWD\",\"body\":\"SECRET_BODY_7c1\",\"nested\":{\"k\":\"SECRET_BODY_7c1\",\"l\":[\"SECRET_BODY_7c1\"]},\"arr\":[\"SECRET_BODY_7c1\",{\"x\":\"SECRET_BODY_7c1\"}]"
  extra_infer=",\"body\":\"SECRET_BODY_7c1\",\"nested\":{\"k\":\"SECRET_BODY_7c1\"},\"arr\":[\"SECRET_BODY_7c1\"]"
fi
if [ "${FAKE_FAIL_STAGE:-}" = "$stage" ]; then
  printf '{"code":"runtime_error","message":"%s","step":"%s"}\n' "${FAKE_FAIL_MSG:-stage failed at /secret/dir}" "$stage"
  exit "${FAKE_FAIL_RC:-70}"
fi
comps='"components":{"weights":{"bytes":100,"file_count":1},"vocab_or_feature_transform":{"bytes":20,"file_count":1},"label_table":{"bytes":5,"file_count":1},"calibration":{"bytes":3,"file_count":1},"metadata":{"bytes":7,"file_count":1}}'
case "$stage" in
register)
  printf '{"step":"register","status":"ok","definition_sha256":"%s","options":3,"evaluation_defined":true%s}\n' "$SHA" "$extra" ;;
inspect)
  printf '{"step":"inspect","status":"ok","valid_records":90,"split":{"train":72,"validation":9,"test":9}%s}\n' "$extra" ;;
train)
  cand=0
  [ "$bad" = cand_false ] && cand=false
  printf '{"step":"train","status":"ok","candidate":%s,"kind":"c1"%s}\n' "$cand" "$extra" ;;
select)
  k=c1
  [ "$bad" = select_kind ] && k=c3
  printf '{"step":"select","status":"ok","candidate":0,"kind":"%s"%s}\n' "$k" "$extra" ;;
evaluate)
  acc=1.0
  [ "$bad" = accuracy ] && acc=0.5
  printf '{"step":"evaluate","status":"ok","candidate":0,"kind":"c1","n_total":%s,"correct":12,"accuracy":%s,"macro_f1":1.0%s}\n' "${FAKE_EVAL_N:-12}" "$acc" "$extra" ;;
package)
  c2mode=${FAKE_C2_MODE:-limit}
  # 内訳の合計。bad=sum:<case> のときだけ total_bytes を 1 多く返す
  extrab=0
  [ "$bad" = "sum:$here" ] && extrab=1
  if [ "$here" = C2 ] && [ "$c2mode" != exit0 ]; then
    [ "$c2mode" = keepdir ] && mkdir -p project/package
    # 実 CLI と同じく total_bytes > limit_bytes で超過。内訳の合計は total_bytes と一致させる。
    # small は total_bytes <= limit_bytes なのに exceeded:true を返す不正な出力
    wb=$((pkglim + 1))
    [ "$c2mode" = small ] && wb=100
    total=$((wb + 35 + extrab))
    rl=$pkglim
    [ -n "${FAKE_C2_LIMIT_WRONG:-}" ] && rl=$((pkglim + 7))
    code='"limit_exceeded"'
    [ -n "${FAKE_C2_CODE_NESTED:-}" ] && code='{"k":"SECRET_BODY_7c1"}'
    c2comps="\"components\":{\"weights\":{\"bytes\":$wb,\"file_count\":1},\"vocab_or_feature_transform\":{\"bytes\":20,\"file_count\":1},\"label_table\":{\"bytes\":5,\"file_count\":1},\"calibration\":{\"bytes\":3,\"file_count\":1},\"metadata\":{\"bytes\":7,\"file_count\":1}}"
    printf '{"code":%s,"message":"resource limit exceeded","step":"package","capacity":{"total_bytes":%s,"limit_bytes":%s,"exceeded":true,%s},"infer_p95":null}\n' "$code" "$total" "$rl" "$c2comps"
    exit 20
  fi
  if [ "$here" = C1 ] && [ -n "${FAKE_C1_EXCEED:-}" ]; then
    # low は p95_us が上限未満なのに exceeded:true・exit 20 を返す不正な出力
    pv=99999
    [ "$FAKE_C1_EXCEED" = low ] && pv=2
    [ -n "${FAKE_C1_KEEPDIR:-}" ] && mkdir -p project/package
    printf '{"code":"limit_exceeded","message":"resource limit exceeded","step":"package","capacity":{"total_bytes":135,"limit_bytes":40000000,"exceeded":false,%s},"infer_p95":{"p95_us":%s,"limit_us":%s,"exceeded":true}}\n' "$comps" "$pv" "$p95lim"
    exit 20
  fi
  total=$((135 + extrab))
  lim=$pkglim
  [ "$bad" = "limit:$here" ] && lim=39999999
  # package/ の通常ファイルの合計は total_bytes と一致させる（pkgsum は 1 バイト少なくする）
  odd=0
  [ -n "${FAKE_PKG_ODD:-}" ] && odd=10
  mb=$((total - 35 - odd))
  [ "$bad" = "pkgsum:$here" ] && mb=$((mb - 1))
  mkdir -p project/package
  head -c "$mb" /dev/zero > project/package/model.onnx
  head -c 35 /dev/zero > project/package/artifact.json
  [ "$odd" != 0 ] && head -c "$odd" /dev/zero > "project/package/odd name.bin"
  p95=null
  pv=2
  [ "$bad" = "p95neg:$here" ] && pv=-1
  [ "$here" = C1 ] && p95=$(printf '{"p95_us":%s,"limit_us":%s,"exceeded":false}' "$pv" "$p95lim")
  jfield='"judgment":null,'
  [ "$bad" = "nojudgment:$here" ] && jfield=
  printf '{"step":"package","status":"ok",%s"acceptance_defined":false,"capacity":{"total_bytes":%s,"limit_bytes":%s,"exceeded":false,%s},"infer_p95":%s%s}\n' "$jfield" "$total" "$lim" "$comps" "$p95" "$extra" ;;
infer)
  file=
  id=input
  prev=
  for a in "$@"; do
    [ "$prev" = --input-file ] && file=$a
    [ "$prev" = --id ] && id=$a
    prev=$a
  done
  if [ -n "$file" ]; then
    cp "$file" "$FAKE_DIR/batch-input.jsonl"
    lab=alpha
    BSC=$SC
    # バッチだけ最大スコアの選択肢が変わる（スコアと label は整合。単体と食い違う）
    [ -n "${FAKE_E_MISMATCH:-}" ] && lab=beta && BSC='{"alpha":0.25,"beta":0.5,"gamma":0.25}'
    # label だけが最大スコアでない（スコアは単体と同じ）
    [ "$bad" = batch_notmax ] && lab=beta
    # 2 行目のスコアだけ 1e-12 ずらす（label・最大スコア・合計の規則は保つ。許容差内だが完全一致ではない）
    SC2=$BSC
    [ "$bad" = batch_score_shift ] && SC2='{"alpha":0.500000000001,"beta":0.25,"gamma":0.25}'
    first=
    n=0
    while IFS= read -r line; do
      rid=$(printf '%s' "$line" | sed -n 's/^{"id": "\([^"]*\)".*/\1/p')
      [ -n "$first" ] || first=$rid
      n=$((n + 1))
      [ -n "${FAKE_E_DUP:-}" ] && [ "$n" = 2 ] && rid=$first
      rowsc=$BSC
      [ "$n" = 2 ] && rowsc=$SC2
      printf '{"id":"%s","status":"ok","predicted_label":"%s","scores":'"$rowsc"'}\n' "$rid" "$lab"
    done < "$file"
  else
    lab=alpha
    [ -n "${FAKE_INFER_BAD_LABEL:-}" ] && lab=zzz
    [ "$bad" = infer_notmax ] && lab=beta
    st='"ok"'
    [ -n "${FAKE_INFER_STATUS_NESTED:-}" ] && st='{"k":"SECRET_BODY_7c1"}'
    printf '{"id":"%s","status":%s,"predicted_label":"%s","scores":'"$SC"'%s}\n' "$id" "$st" "$lab" "$extra_infer"
  fi ;;
*) exit 99 ;;
esac
"##;

/// 偽の make。`ci`（A）と `check-runtime-linkage`（D）だけを受け付ける。出力は実際の形式で、
/// `make ci` の実際の順序どおり pytest の要約行の後に cargo の `test result:` 行を出す。
/// `FAKE_CI_MODE`（ok・skip・noresults・rustfail・fail）・`FAKE_LINK_MODE`（ok・skip・none・fail）・
/// `FAKE_MAKE_SLEEP`（D で長く待つ。中断のテスト用）で切り替える。
const FAKE_MAKE: &str = r##"#!/bin/sh
echo "$*" >> "$FAKE_DIR/make.log"
echo "$1 ${CARGO_NET_OFFLINE:-unset}" >> "$FAKE_DIR/make.env"
case "$1" in
ci)
  mode=${FAKE_CI_MODE:-ok}
  [ "$mode" = skip ] && echo "skip: no trainer"
  if [ "$mode" != noresults ]; then
    echo "======= 7 passed, 1 skipped in 1.20s ======="
    if [ "$mode" = rustfail ]; then
      echo "test result: FAILED. 11 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s"
    else
      echo "test result: ok. 12 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.01s"
    fi
    echo "test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s"
  fi
  [ "$mode" = fail ] && exit 2
  exit 0 ;;
check-runtime-linkage)
  if [ -n "${FAKE_MAKE_SLEEP:-}" ]; then
    echo $$ > "$FAKE_DIR/make.pid"
    sleep 60 &
    echo $! > "$FAKE_DIR/make.cpid"
    : > "$FAKE_DIR/make.started"
    wait
  fi
  if [ -n "${FAKE_MAKE_ORPHAN:-}" ]; then
    # TERM を無視する孫を残して正常終了する（ラッパーが最後に KILL で片付けること）
    sh -c 'trap "" TERM; echo $$ > "$FAKE_DIR/orphan.pid"; exec sleep 60' &
    while [ ! -s "$FAKE_DIR/orphan.pid" ]; do sleep 0.05; done
  fi
  mode=${FAKE_LINK_MODE:-ok}
  [ "$mode" = skip ] && echo "skip: no cargo"
  [ "$(uname -s)" = Darwin ] && tool=otool || tool=ldd
  if [ "$mode" != none ]; then
    echo "ok: req32_one"
    echo "ok: req32_two"
    [ "$mode" = two ] || echo "ok: req32_three"
    [ "$mode" = fail ] || echo "OK: tool=$tool evidence=transitive targets=fandhe-edge,env_isolation"
  fi
  [ "$mode" = fail ] && exit 2
  exit 0 ;;
*) exit 99 ;;
esac
"##;

/// 偽の cargo。`build`（`--message-format=json`）は `FAKE_CLI_PATH` を `compiler-artifact` で報告する。
/// `test ... --no-run` は回数に数えない。それ以外の `test` は `FAKE_CARGO_PATTERN`
/// （カンマ区切り。`ok`・`inc`・`plain`・`zero`）の n 番目で n 回目の結果を決める。
const FAKE_CARGO: &str = r##"#!/bin/sh
printf '%s\n' "$*" >> "$FAKE_DIR/cargo.args"
echo "${CARGO_NET_OFFLINE:-unset}" >> "$FAKE_DIR/cargo.env"
if [ "$1" = build ]; then
  printf '{"reason":"compiler-artifact","target":{"name":"fandhe-edge","kind":["bin"]},"executable":"%s"}\n' "$FAKE_CLI_PATH"
  exit 0
fi
case "$*" in *--no-run*) exit 0 ;; esac
n=$(( $(cat "$FAKE_DIR/cargo.count" 2>/dev/null || echo 0) + 1 ))
echo "$n" > "$FAKE_DIR/cargo.count"
mode=$(echo "${FAKE_CARGO_PATTERN:-}" | cut -d, -f"$n")
case "$mode" in
inc) echo "error: ReadOutputIncomplete"; exit 101 ;;
plain) echo "error: ReadOutput(Io)"; exit 101 ;;
zero) echo "test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s"; exit 0 ;;
*) echo "test result: ok. 12 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s"; exit 0 ;;
esac
"##;

struct Out {
    code: Option<i32>,
    stdout: String,
}

/// テストごとの作業ディレクトリと偽のコマンド群。
struct Env {
    dir: PathBuf,
    work: PathBuf,
    cli: PathBuf,
    make: PathBuf,
    cargo: PathBuf,
}

impl Env {
    fn new() -> Self {
        let n = SEQ.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("fandhe-rmc-{}-{n}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("mkdir");
        let cli = dir.join("fake-cli");
        let make = dir.join("fake-make");
        let cargo = dir.join("fake-cargo");
        write_exe(&cli, FAKE_CLI);
        write_exe(&make, FAKE_MAKE);
        write_exe(&cargo, FAKE_CARGO);
        Env {
            work: dir.join("work"),
            dir,
            cli,
            make,
            cargo,
        }
    }

    fn work_arg(&self) -> Vec<String> {
        vec!["--work-dir".into(), self.work.display().to_string()]
    }

    fn lines(&self, name: &str) -> Vec<String> {
        fs::read_to_string(self.dir.join(name))
            .unwrap_or_default()
            .lines()
            .map(str::to_string)
            .collect()
    }

    /// スクリプトを起動する Command。`envs` の値が空文字ならスクリプトからは未設定と同じに見える。
    fn command(&self, args: &[String], envs: &[(&str, &str)], cwd: Option<&Path>) -> Command {
        let mut cmd = Command::new("sh");
        cmd.process_group(0)
            .arg(repo_root().join("scripts").join("real-machine-check.sh"))
            .args(args)
            .env_remove("CARGO_NET_OFFLINE")
            .env("FAKE_DIR", &self.dir)
            .env("FAKE_CLI_PATH", &self.cli)
            .env("FANDHE_EDGE_BIN", &self.cli)
            .env("FANDHE_EDGE_MAKE_CMD", &self.make)
            .env("FANDHE_EDGE_CARGO_CMD", &self.cargo)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        if let Some(d) = cwd {
            cmd.current_dir(d);
        }
        for (k, v) in envs {
            cmd.env(k, v);
        }
        cmd
    }

    fn run(&self, args: &[String], envs: &[(&str, &str)]) -> Out {
        self.run_cwd(args, envs, None)
    }

    fn run_cwd(&self, args: &[String], envs: &[(&str, &str)], cwd: Option<&Path>) -> Out {
        let mut child = self.command(args, envs, cwd).spawn().expect("spawn sh");
        let mut so = child.stdout.take().expect("stdout");
        let pgid = child.id();
        let h_out = std::thread::spawn(move || {
            let mut s = String::new();
            so.read_to_string(&mut s).ok();
            s
        });
        let start = Instant::now();
        let status = loop {
            if let Some(st) = child.try_wait().expect("try_wait") {
                break st;
            }
            if start.elapsed() > TIMEOUT {
                Command::new("kill")
                    .args(["-KILL", &format!("-{pgid}")])
                    .status()
                    .ok();
                child.kill().ok();
                child.wait().ok();
                h_out.join().ok();
                panic!("script timed out");
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        Out {
            code: status.code(),
            stdout: h_out.join().expect("join"),
        }
    }

    fn text(&self, name: &str) -> String {
        fs::read_to_string(self.work.join(name)).expect("read record")
    }

    /// 偽 CLI のログから `<stage> <cwd 末尾>` の行の数を数える。
    fn count_calls(&self, entry: &str) -> usize {
        self.lines("cli.log").iter().filter(|l| *l == entry).count()
    }

    /// `record.json` の値を `a.b.0.*.c` 形式のパスで取り出し、コンパクト JSON（キー順固定）で返す。
    fn q(&self, path: &str) -> String {
        const PY: &str = "import json,sys\n\
            def walk(v,p):\n\
            \x20   if not p: return v\n\
            \x20   if p[0]=='*': return [walk(x,p[1:]) for x in v]\n\
            \x20   if isinstance(v,list): return walk(v[int(p[0])],p[1:])\n\
            \x20   return walk(v[p[0]],p[1:])\n\
            v=json.load(open(sys.argv[1]))\n\
            print(json.dumps(walk(v,sys.argv[2].split('.')),separators=(',',':'),sort_keys=True))\n";
        let o = Command::new("python3")
            .args(["-I", "-c", PY])
            .arg(self.work.join("record.json"))
            .arg(path)
            .stdin(Stdio::null())
            .output()
            .expect("python3");
        assert!(
            o.status.success(),
            "query {path} failed: {}",
            String::from_utf8_lossy(&o.stderr)
        );
        String::from_utf8(o.stdout)
            .expect("utf8")
            .trim()
            .to_string()
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

fn write_exe(path: &Path, body: &str) {
    fs::write(path, body).expect("write");
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).expect("chmod");
}

fn s(v: &[&str]) -> Vec<String> {
    v.iter().map(|x| x.to_string()).collect()
}

fn with_work(e: &Env, extra: &[&str]) -> Vec<String> {
    let mut a = e.work_arg();
    a.extend(s(extra));
    a
}

/// `{"id": "x", ...}`（空白の有無を問わない）から id の値を取り出す。
fn extract_id(line: &str) -> Option<String> {
    let rest = line.split("\"id\":").nth(1)?.trim_start();
    let rest = rest.strip_prefix('"')?;
    Some(rest.split('"').next()?.to_string())
}

fn fixture_ids(name: &str) -> Vec<String> {
    let p = repo_root()
        .join("fixtures")
        .join("sandbox_run_eval")
        .join(name);
    fs::read_to_string(p)
        .expect("fixture")
        .lines()
        .filter_map(extract_id)
        .collect()
}

const INVALID: &str = "{\"code\":\"invalid_input\",\"message\":\"";
const ITEMS_MSG: &str = "--items must be a comma-separated subset of A,B,C,D,E,F";
const REPEAT_MSG: &str = "--repeat must be an integer from 1 to 1000";
const JUDGED_FAIL: &str = "{\"code\":\"judged_fail\",\"message\":\"one or more requested items failed or were not run\",\"record\":\"record.json\"}\n";

/// 引数と envs の組で検証失敗（exit 64・固定メッセージ）になり、何も起動されず、
/// 渡した `--work-dir` も作られていないことを確かめる。
fn assert_rejected_before_start(e: &Env, args: &[String], envs: &[(&str, &str)], message: &str) {
    let o = e.run(args, envs);
    assert_eq!(o.code, Some(64), "args={args:?} stdout={}", o.stdout);
    assert_eq!(
        o.stdout,
        format!("{INVALID}{message}\"}}\n"),
        "args={args:?}"
    );
    for log in ["make.log", "cargo.args", "cli.log"] {
        assert!(e.lines(log).is_empty(), "{log} was written for {args:?}");
    }
    let work = args
        .iter()
        .position(|a| a == "--work-dir")
        .and_then(|i| args.get(i + 1));
    if let Some(w) = work {
        assert!(!Path::new(w).exists(), "work dir created for {args:?}");
    }
}

/// REQ-21・REQ-39: 引数の誤りは make・cargo・CLI を起動する前に exit 64 と固定メッセージの 1 行 JSON。
/// 実際に渡した `--work-dir` が作られていないことも、その呼び出しの Env で確かめる。
#[test]
fn req21_invalid_args_exit_64_before_starting_anything() {
    let repo_inside = repo_root()
        .join("crates")
        .join("cli")
        .join("rmc-test-nonexistent")
        .display()
        .to_string();
    let cases: Vec<(Vec<&str>, &str)> = vec![
        (
            vec!["--items", "A"],
            "item A requires --with-ci (make ci may use the network)",
        ),
        (vec!["--items", "X"], ITEMS_MSG),
        (vec!["--items", "b"], ITEMS_MSG),
        (vec!["--items", "B,"], ITEMS_MSG),
        (
            vec!["--items", "B,B"],
            "--items must not contain duplicates",
        ),
        (vec!["--repeat", "0"], REPEAT_MSG),
        (vec!["--repeat=1001"], REPEAT_MSG),
        (vec!["--repeat", "abc"], REPEAT_MSG),
        (vec!["--repeat", "05"], REPEAT_MSG),
    ];
    for (extra, message) in cases {
        let e = Env::new();
        assert_rejected_before_start(&e, &with_work(&e, &extra), &[], message);
    }
    let e = Env::new();
    assert_rejected_before_start(&e, &[], &[], "--work-dir is required");
    assert_rejected_before_start(
        &e,
        &s(&["--work-dir", &repo_inside]),
        &[],
        "work directory must be outside the repository",
    );
    assert!(!Path::new(&repo_inside).exists());
}

/// REQ-39: 代役コマンドの指定誤り・作業ディレクトリの誤り（空でない・symlink 経由のリポジトリ配下）は
/// exit 64。相対パスの CLI は物理パスへ正規化されて動き、裸の名前は 64。
#[test]
fn req39_invalid_overrides_and_work_dirs_are_rejected() {
    let abs = "must be an absolute path to an executable";
    let e = Env::new();
    assert_rejected_before_start(
        &e,
        &with_work(&e, &["--items", "D"]),
        &[("FANDHE_EDGE_MAKE_CMD", "fake-make")],
        &format!("FANDHE_EDGE_MAKE_CMD {abs}"),
    );
    assert_rejected_before_start(
        &e,
        &with_work(&e, &["--items", "D"]),
        &[("FANDHE_EDGE_MAKE_CMD", "./fake-make")],
        &format!("FANDHE_EDGE_MAKE_CMD {abs}"),
    );
    assert_rejected_before_start(
        &e,
        &with_work(&e, &["--items", "F"]),
        &[("FANDHE_EDGE_CARGO_CMD", "fake-cargo")],
        &format!("FANDHE_EDGE_CARGO_CMD {abs}"),
    );
    assert_rejected_before_start(
        &e,
        &with_work(&e, &["--items", "B"]),
        &[("FANDHE_EDGE_BIN", "fake-cli")],
        "FANDHE_EDGE_BIN must be a path containing a slash",
    );

    // 空でない作業ディレクトリ
    let e = Env::new();
    fs::create_dir_all(&e.work).expect("mkdir");
    fs::write(e.work.join("keep"), "x").expect("write");
    let o = e.run(&e.work_arg(), &[]);
    assert_eq!(o.code, Some(64), "{}", o.stdout);
    assert_eq!(
        o.stdout,
        format!("{INVALID}work directory is not empty\"}}\n")
    );
    assert!(e.lines("cli.log").is_empty());

    // symlink 経由でリポジトリ配下を指す
    let e = Env::new();
    let link = e.dir.join("link-to-repo");
    symlink(repo_root().join("crates"), &link).expect("symlink");
    let target = repo_root().join("crates").join("rmc-test-via-link");
    let args = s(&[
        "--work-dir",
        &link.join("rmc-test-via-link").display().to_string(),
    ]);
    assert_rejected_before_start(
        &e,
        &args,
        &[],
        "work directory must be outside the repository",
    );
    assert!(!target.exists());

    // 作業ディレクトリ自体が symlink
    let e = Env::new();
    let real = e.dir.join("real-empty");
    fs::create_dir_all(&real).expect("mkdir");
    let link = e.dir.join("link-work");
    symlink(&real, &link).expect("symlink");
    let o = e.run(&s(&["--work-dir", &link.display().to_string()]), &[]);
    assert_eq!(o.code, Some(64), "{}", o.stdout);
    assert_eq!(
        o.stdout,
        format!("{INVALID}work directory is not a directory\"}}\n")
    );
    assert!(e.lines("cli.log").is_empty());
}

/// REQ-33: `FANDHE_EDGE_BIN` が `./` 付きの相対パスでも、呼び出し時の cwd を基準に解決されて成功する。
#[test]
fn req33_relative_bin_path_is_resolved() {
    let e = Env::new();
    let o = e.run_cwd(
        &with_work(&e, &["--items", "B"]),
        &[("FANDHE_EDGE_BIN", "./fake-cli")],
        Some(&e.dir),
    );
    assert_eq!(o.code, Some(0), "stdout={}", o.stdout);
    assert_eq!(e.q("bin_override"), "true");
    assert_eq!(e.count_calls("register B"), 1);
}

/// REQ-21・REQ-28・REQ-30・REQ-31・REQ-33: B〜F が通り、記録の構造が契約どおりになる。
#[test]
fn req33_normal_run_records_all_items() {
    let e = Env::new();
    let o = e.run(
        &with_work(&e, &["--items", "B,C,D,E,F", "--repeat", "3"]),
        &[],
    );
    assert_eq!(o.code, Some(0), "stdout={}", o.stdout);
    assert_eq!(
        o.stdout,
        "{\"code\":\"ok\",\"message\":\"all requested items completed\",\"record\":\"record.json\"}\n"
    );
    assert!(e.work.join("record.md").is_file());

    assert_eq!(e.q("schema"), "\"real-machine-check/1\"");
    assert_eq!(e.q("evidence_hint"), "\"test_harness\"");
    assert_eq!(e.q("bin_override"), "true");
    assert_eq!(e.q("environment.cli_profile"), "null");
    assert_eq!(e.q("options.items"), "[\"B\",\"C\",\"D\",\"E\",\"F\"]");
    assert_eq!(e.q("options.repeat"), "3");
    assert_eq!(e.q("options.with_ci"), "false");
    assert_eq!(e.q("options.cargo_offline"), "true");
    assert_eq!(e.q("items.A.status"), "\"not_run\"");
    for item in ["B", "C", "D", "E", "F"] {
        assert_eq!(e.q(&format!("items.{item}.status")), "\"ok\"", "{item}");
    }

    // B: 7 工程の順序。select は evaluate より先（REQ-27・REQ-33）
    assert_eq!(
        e.q("items.B.steps.*.step"),
        "[\"register\",\"inspect\",\"train\",\"select\",\"evaluate\",\"package\",\"infer\"]"
    );
    let calls = e.lines("cli.log");
    assert_eq!(
        calls[..7],
        [
            "register B",
            "inspect B",
            "train B",
            "select B",
            "evaluate B",
            "package B",
            "infer B"
        ]
    );
    // B(7) + C-1(6) + C-2(6) + E のバッチ 1 + 単体 90
    assert_eq!(calls.len(), 7 + 6 + 6 + 1 + 90);
    let args = e.lines("cli.args");
    assert_eq!(args[2], "train --project-dir project --candidate 0");
    assert_eq!(args[4], "evaluate --project-dir project --candidate 0");

    // 容量内訳の 5 項目と合計（REQ-30）
    assert_eq!(e.q("items.B.capacity.total_bytes"), "135");
    assert_eq!(e.q("items.B.capacity_sum_matches_total"), "true");
    for (name, bytes) in [
        ("weights", 100),
        ("vocab_or_feature_transform", 20),
        ("label_table", 5),
        ("calibration", 3),
        ("metadata", 7),
    ] {
        assert_eq!(
            e.q(&format!("items.B.capacity.components.{name}.bytes")),
            bytes.to_string(),
            "{name}"
        );
    }
    assert_eq!(
        e.q("items.B.package_files.*.name"),
        "[\"artifact.json\",\"model.onnx\"]"
    );

    // C: p95 と容量上限（REQ-30・REQ-31）
    assert_eq!(e.q("items.C.p95.p95_us"), "2");
    assert_eq!(e.q("items.C.p95.limit_us"), "50000");
    assert_eq!(e.q("items.C.p95.classification"), "\"reference_only\"");
    assert_eq!(e.q("items.C.p95.package_exit_code"), "0");
    assert_eq!(e.q("items.C.capacity_limit.package_exit_code"), "20");
    assert_eq!(e.q("items.C.capacity_limit.capacity_exceeded"), "true");
    assert_eq!(e.q("items.C.capacity_limit.limit_bytes"), "1000");
    assert_eq!(e.q("items.C.capacity_limit.package_published"), "false");

    // D・E・F（REQ-28）
    assert_eq!(e.q("items.D.env_i_tests_ok"), "3");
    assert_eq!(
        e.q("items.D.linkage_tool"),
        if cfg!(target_os = "macos") {
            "\"otool\""
        } else {
            "\"ldd\""
        }
    );
    assert_eq!(e.q("items.E.records"), "90");
    assert_eq!(e.q("items.E.label_match"), "90");
    assert_eq!(e.q("items.E.label_mismatch"), "0");
    assert_eq!(e.q("items.E.scores_exact_match"), "90");
    assert_eq!(e.q("items.F.runs"), "3");
    assert_eq!(e.q("items.F.passed"), "3");
    assert_eq!(e.q("items.F.failed"), "0");
    assert_eq!(e.q("items.F.no_tests"), "0");
    // `--no-run` の 1 回 + 本番 3 回。`--no-run` は回数に数えない
    assert_eq!(e.lines("cargo.count"), ["3"]);
    let cargo = e.lines("cargo.args");
    assert_eq!(
        cargo[0],
        "test --locked -p fandhe-edge-guard --test time_limit --no-run"
    );
    assert_eq!(cargo.len(), 4);
    assert!(
        cargo[1..]
            .iter()
            .all(|l| l == "test --locked -p fandhe-edge-guard --test time_limit")
    );
}

/// REQ-38: A 以外の子（make の D・cargo・CLI）には `CARGO_NET_OFFLINE=true` が渡り、
/// A の `make ci` には渡らない。
#[test]
fn req38_offline_env_is_passed_except_to_make_ci() {
    let e = Env::new();
    let o = e.run(
        &with_work(&e, &["--items", "A,B,D,F", "--repeat", "1", "--with-ci"]),
        &[],
    );
    assert_eq!(o.code, Some(0), "stdout={}", o.stdout);
    assert_eq!(
        e.lines("make.env"),
        ["ci unset", "check-runtime-linkage true"]
    );
    assert_eq!(e.lines("cargo.env"), ["true", "true"]);
    let cli_env = e.lines("cli.env");
    assert_eq!(cli_env.len(), 7);
    assert!(cli_env.iter().all(|l| l == "true"));
}

/// security.md: 記録・stdout に、作業ディレクトリのパス・データ本文・id の値が現れない。
#[test]
fn req39_record_does_not_leak_path_body_or_id() {
    let e = Env::new();
    let o = e.run(
        &with_work(&e, &["--items", "B,C,D,E,F", "--repeat", "2"]),
        &[("FAKE_LEAK", "1")],
    );
    assert_eq!(o.code, Some(0), "stdout={}", o.stdout);
    let json = e.text("record.json");
    let md = e.text("record.md");
    let mut forbidden = vec![
        e.dir.display().to_string(),
        LEAK_BODY.to_string(),
        LEAK_ID.to_string(),
        "C:\\\\Users".to_string(),
        "C:\\Users".to_string(),
        "a/b".to_string(),
        // train.jsonl の id と入力本文
        "alpha-0".to_string(),
        "alpha sample".to_string(),
    ];
    forbidden.push(
        fs::canonicalize(&e.work)
            .expect("canon")
            .display()
            .to_string(),
    );
    forbidden.push(
        fs::canonicalize(repo_root())
            .expect("canon")
            .display()
            .to_string(),
    );
    for (name, text) in [
        ("record.json", &json),
        ("record.md", &md),
        ("stdout", &o.stdout),
    ] {
        for f in &forbidden {
            assert!(!text.contains(f.as_str()), "{name} contains {f:?}");
        }
    }
    // 契約 §1-1: 許可リスト外の欄（パス値の欄を含む）は工程の要約へ組み立てられず、キーごと落ちる
    for key in ["project_dir", "rel", "win", "input", "text", "id", "note"] {
        assert!(!json.contains(&format!("\"{key}\"")), "{key} key kept");
    }
    assert_eq!(e.q("items.B.steps.0.summary.options"), "3");
    assert_eq!(e.q("items.B.steps.0.summary.evaluation_defined"), "true");
}

/// REQ-21: 工程が失敗したら B を failed と記録し、後続の項目・工程を実行せず exit 10。
#[test]
fn req21_step_failure_stops_following_steps_and_items() {
    let e = Env::new();
    let o = e.run(
        &with_work(&e, &["--items", "B,C,D,E,F", "--repeat", "2"]),
        &[("FAKE_FAIL_STAGE", "train")],
    );
    assert_eq!(o.code, Some(10), "stdout={}", o.stdout);
    assert_eq!(o.stdout, JUDGED_FAIL);
    assert_eq!(e.q("items.B.status"), "\"failed\"");
    assert_eq!(e.q("items.B.reason"), "\"unexpected_exit_code\"");
    assert_eq!(e.q("items.B.step"), "\"train\"");
    assert_eq!(e.q("items.B.exit_code"), "70");
    assert_eq!(e.q("items.B.code"), "\"runtime_error\"");
    // 契約: `message` は記録せず、UTF-8 のバイト数と sha256 だけを出す（本文は work-dir の stdout に残る）
    assert_eq!(e.q("items.B.message_bytes"), "27");
    assert_eq!(
        e.q("items.B.message_sha256"),
        "\"4273680ba876c1fa9378313783d827c53318f42ada32574d6a111a9e12974c8a\""
    );
    assert!(!e.text("record.json").contains("stage failed"));
    assert!(!e.text("record.md").contains("stage failed"));
    assert_eq!(
        e.q("items.B.steps.*.step"),
        "[\"register\",\"inspect\",\"train\"]"
    );
    for item in ["C", "D", "E", "F"] {
        assert_eq!(e.q(&format!("items.{item}.status")), "\"not_run\"");
        assert_eq!(
            e.q(&format!("items.{item}.reason")),
            "\"previous_item_failed\""
        );
    }
    // select 以降の工程・make・cargo は起動されていない
    assert_eq!(e.lines("cli.log"), ["register B", "inspect B", "train B"]);
    assert!(e.lines("make.log").is_empty());
    assert!(e.lines("cargo.args").is_empty());
}

/// REQ-27: `evaluate` は B・C-1・C-2 の project ごとにちょうど 1 回で、失敗後に再実行されない。
#[test]
fn req27_evaluate_runs_exactly_once_per_project() {
    let e = Env::new();
    let o = e.run(&with_work(&e, &["--items", "B,C"]), &[]);
    assert_eq!(o.code, Some(0), "stdout={}", o.stdout);
    for entry in ["evaluate B", "evaluate C1", "evaluate C2"] {
        assert_eq!(e.count_calls(entry), 1, "{entry}");
    }
    let e = Env::new();
    let o = e.run(
        &with_work(&e, &["--items", "B,C"]),
        &[("FAKE_FAIL_STAGE", "evaluate")],
    );
    assert_eq!(o.code, Some(10), "stdout={}", o.stdout);
    assert_eq!(e.count_calls("evaluate B"), 1);
    assert_eq!(e.count_calls("evaluate C1"), 0);
    assert_eq!(e.count_calls("package B"), 0);
    assert_eq!(e.q("items.B.step"), "\"evaluate\"");
}

/// REQ-39・#346: F は一部が失敗しても全回実行し、`ReadOutput` 系の件数を数えて exit 10。
#[test]
fn req39_item_f_runs_all_repeats_even_when_some_fail() {
    let e = Env::new();
    let o = e.run(
        &with_work(&e, &["--items", "F", "--repeat", "6"]),
        &[("FAKE_CARGO_PATTERN", "ok,inc,plain,ok,inc,ok")],
    );
    assert_eq!(o.code, Some(10), "stdout={}", o.stdout);
    assert_eq!(e.q("items.F.status"), "\"failed\"");
    assert_eq!(e.q("items.F.reason"), "\"test_failures\"");
    assert_eq!(e.q("items.F.runs"), "6");
    assert_eq!(e.q("items.F.passed"), "3");
    assert_eq!(e.q("items.F.failed"), "3");
    assert_eq!(e.q("items.F.read_output"), "1");
    assert_eq!(e.q("items.F.read_output_incomplete"), "2");
    assert_eq!(e.q("items.F.timeouts"), "0");
    assert_eq!(e.q("items.F.no_tests"), "0");
    assert_eq!(e.lines("cargo.count"), ["6"]);
}

/// REQ-39: 0 件しか実行しなかった回（`test result: ok. 0 passed`）は失敗に数え `no_tests` に載せる。
/// `--no-run` の呼び出しは回数に数えない。
#[test]
fn req39_item_f_counts_zero_tests_as_failure() {
    let e = Env::new();
    let o = e.run(
        &with_work(&e, &["--items", "F", "--repeat", "3"]),
        &[("FAKE_CARGO_PATTERN", "ok,zero,zero")],
    );
    assert_eq!(o.code, Some(10), "stdout={}", o.stdout);
    assert_eq!(e.q("items.F.status"), "\"failed\"");
    assert_eq!(e.q("items.F.runs"), "3");
    assert_eq!(e.q("items.F.passed"), "1");
    assert_eq!(e.q("items.F.failed"), "2");
    assert_eq!(e.q("items.F.no_tests"), "2");
    assert_eq!(e.q("items.F.read_output"), "0");
    // `--no-run` 1 回 + 本番 3 回の計 4 回呼ばれるが、数えるのは本番だけ
    assert_eq!(e.lines("cargo.args").len(), 4);
    assert_eq!(e.lines("cargo.count"), ["3"]);
}

/// REQ-27: E は B の `train.jsonl` の入力だけを使い、凍結した `evaluation.jsonl` を `infer` へ渡さない。
#[test]
fn req27_item_e_uses_train_inputs_not_evaluation_data() {
    let e = Env::new();
    let o = e.run(&with_work(&e, &["--items", "B,E"]), &[]);
    assert_eq!(o.code, Some(0), "stdout={}", o.stdout);
    let train_ids = fixture_ids("train.jsonl");
    let eval_ids = fixture_ids("evaluation.jsonl");
    assert_eq!(train_ids.len(), 90);
    assert_eq!(eval_ids.len(), 12);

    let batch = fs::read_to_string(e.dir.join("batch-input.jsonl")).expect("batch input");
    let mut batch_ids: Vec<String> = batch.lines().filter_map(extract_id).collect();
    assert_eq!(batch_ids.len(), 90);
    batch_ids.sort();
    let mut expected = train_ids;
    expected.sort();
    assert_eq!(batch_ids, expected);
    assert!(batch_ids.iter().all(|i| !eval_ids.contains(i)));

    for line in e.lines("cli.args") {
        assert!(!line.contains("evaluation.jsonl"), "{line}");
        for id in &eval_ids {
            assert!(!line.contains(id.as_str()), "{line}");
        }
    }
}

/// REQ-28: バッチと単体でラベルが違えば E は failed・exit 10。記録に載るのは件数だけで id は載らない。
#[test]
fn req28_item_e_mismatch_fails_without_ids_in_record() {
    let e = Env::new();
    let o = e.run(
        &with_work(&e, &["--items", "B,E"]),
        &[("FAKE_E_MISMATCH", "1")],
    );
    assert_eq!(o.code, Some(10), "stdout={}", o.stdout);
    assert_eq!(o.stdout, JUDGED_FAIL);
    assert_eq!(e.q("items.E.status"), "\"failed\"");
    assert_eq!(e.q("items.E.reason"), "\"mismatch\"");
    assert_eq!(e.q("items.E.records"), "90");
    assert_eq!(e.q("items.E.label_match"), "0");
    assert_eq!(e.q("items.E.label_mismatch"), "90");
    for (name, text) in [
        ("record.json", e.text("record.json")),
        ("record.md", e.text("record.md")),
        ("stdout", o.stdout.clone()),
    ] {
        assert!(!text.contains("alpha-0"), "{name} contains an id");
        assert!(!text.contains("mismatch_ids"), "{name} lists ids");
    }
}

/// REQ-27・REQ-28: `--items E` 単独は B が無いので `not_run`（requires_B）で exit 10。
/// B が失敗したときも E は実行されない。
#[test]
fn req28_item_e_requires_successful_b() {
    let e = Env::new();
    let o = e.run(&with_work(&e, &["--items", "E"]), &[]);
    assert_eq!(o.code, Some(10), "stdout={}", o.stdout);
    assert_eq!(e.q("items.E.status"), "\"not_run\"");
    assert_eq!(e.q("items.E.reason"), "\"requires_B\"");
    assert!(e.lines("cli.log").is_empty());

    let e = Env::new();
    let o = e.run(
        &with_work(&e, &["--items", "B,E"]),
        &[("FAKE_FAIL_STAGE", "train")],
    );
    assert_eq!(o.code, Some(10), "stdout={}", o.stdout);
    assert_eq!(e.q("items.E.status"), "\"not_run\"");
    assert_eq!(e.q("items.E.reason"), "\"previous_item_failed\"");
    assert_eq!(e.count_calls("infer B"), 0);
}

/// REQ-39: `--with-ci` が無ければ `make ci` を呼ばない。あれば `ci` で 1 回だけ呼び `skip:` 行を数える。
/// `make ci` の実際の順序（pytest の要約行の後に cargo の `test result:` 行）でも、
/// pytest の件数が Rust の行で上書きされない。
#[test]
fn req39_make_ci_runs_only_with_with_ci() {
    let e = Env::new();
    let o = e.run(&with_work(&e, &["--items", "B,D"]), &[]);
    assert_eq!(o.code, Some(0), "stdout={}", o.stdout);
    assert_eq!(e.lines("make.log"), ["check-runtime-linkage"]);
    assert_eq!(e.q("items.A.status"), "\"not_run\"");
    assert_eq!(e.q("items.A.reason"), "\"not_selected\"");

    let e = Env::new();
    let o = e.run(&with_work(&e, &["--items", "A", "--with-ci"]), &[]);
    assert_eq!(o.code, Some(0), "stdout={}", o.stdout);
    assert_eq!(e.lines("make.log"), ["ci"]);
    assert_eq!(e.q("options.with_ci"), "true");
    assert_eq!(e.q("items.A.status"), "\"ok\"");
    assert_eq!(e.q("items.A.skip_lines"), "0");
    assert_eq!(
        e.q("items.A.rust_tests"),
        "{\"failed\":0,\"ignored\":1,\"passed\":15}"
    );
    assert_eq!(
        e.q("items.A.pytest"),
        "{\"failed\":0,\"passed\":7,\"skipped\":1}"
    );
}

/// REQ-39: A は exit 非 0・`skip:` 行あり・テスト結果の行なし・Rust のテスト失敗のいずれも failed。
#[test]
fn req39_item_a_fails_unless_tests_really_ran() {
    for (mode, reason) in [
        ("fail", "unexpected_exit_code"),
        ("skip", "skipped"),
        ("noresults", "no_test_results"),
        ("rustfail", "test_failures"),
    ] {
        let e = Env::new();
        let o = e.run(
            &with_work(&e, &["--items", "A", "--with-ci"]),
            &[("FAKE_CI_MODE", mode)],
        );
        assert_eq!(o.code, Some(10), "{mode}: {}", o.stdout);
        assert_eq!(e.q("items.A.status"), "\"failed\"", "{mode}");
        assert_eq!(e.q("items.A.reason"), format!("\"{reason}\""), "{mode}");
    }
    let e = Env::new();
    e.run(
        &with_work(&e, &["--items", "A", "--with-ci"]),
        &[("FAKE_CI_MODE", "skip")],
    );
    assert_eq!(e.q("items.A.skip_lines"), "1");
}

/// REQ-32・REQ-39: D は `skip:` 行がある・`ok: req32_` が 0 行・exit 非 0 のいずれも failed。
#[test]
fn req39_item_d_fails_unless_env_i_tests_ran() {
    for (mode, reason) in [
        ("skip", "skipped"),
        ("none", "no_test_results"),
        ("fail", "unexpected_exit_code"),
    ] {
        let e = Env::new();
        let o = e.run(
            &with_work(&e, &["--items", "D"]),
            &[("FAKE_LINK_MODE", mode)],
        );
        assert_eq!(o.code, Some(10), "{mode}: {}", o.stdout);
        assert_eq!(e.q("items.D.status"), "\"failed\"", "{mode}");
        assert_eq!(e.q("items.D.reason"), format!("\"{reason}\""), "{mode}");
    }
}

/// REQ-31: C-1 の package が exit 20（p95 超過）でも C は failed にならず、超過と終了コードが記録される。
#[test]
fn req31_c1_p95_exceeded_is_recorded_not_failed() {
    let e = Env::new();
    let o = e.run(
        &with_work(&e, &["--items", "C"]),
        &[("FAKE_C1_EXCEED", "1")],
    );
    assert_eq!(o.code, Some(0), "stdout={}", o.stdout);
    assert_eq!(e.q("items.C.status"), "\"ok\"");
    assert_eq!(e.q("items.C.p95.exceeded"), "true");
    assert_eq!(e.q("items.C.p95.package_exit_code"), "20");
    assert_eq!(e.q("items.C.p95.p95_us"), "99999");
}

/// REQ-30: C-2 で上限が効かない（package が exit 0、または `package/` が残る）と C は failed。
#[test]
fn req30_c2_unenforced_capacity_limit_fails() {
    for mode in ["exit0", "keepdir"] {
        let e = Env::new();
        let o = e.run(&with_work(&e, &["--items", "C"]), &[("FAKE_C2_MODE", mode)]);
        assert_eq!(o.code, Some(10), "{mode}: {}", o.stdout);
        assert_eq!(e.q("items.C.status"), "\"failed\"", "{mode}");
        assert_eq!(
            e.q("items.C.reason"),
            "\"capacity_limit_not_enforced\"",
            "{mode}"
        );
    }
}

/// REQ-31: 偽 make・偽 cargo の代役（と `FANDHE_EDGE_BIN` の差し替え）の下では、`--quiet-machine` が
/// あっても p95 の区分は `reference_only`（実機の測定でないものを `real_machine` にしない）。
/// `real_machine` になる経路は実 make・実 cargo を要するため、Python 側のテストで固定している。
#[test]
fn req31_quiet_machine_under_harness_stays_reference_only() {
    let e = Env::new();
    let o = e.run(&with_work(&e, &["--items", "C", "--quiet-machine"]), &[]);
    assert_eq!(o.code, Some(0), "stdout={}", o.stdout);
    assert_eq!(e.q("items.C.p95.classification"), "\"reference_only\"");
    assert_eq!(e.q("options.quiet_machine"), "true");
    assert_eq!(e.q("evidence_hint"), "\"test_harness\"");
    let e = Env::new();
    let o = e.run(&with_work(&e, &["--items", "C"]), &[]);
    assert_eq!(o.code, Some(0), "stdout={}", o.stdout);
    assert_eq!(e.q("items.C.p95.classification"), "\"reference_only\"");
    assert_eq!(e.q("options.quiet_machine"), "false");
}

/// REQ-30・REQ-31: `--p95-limit-us`・`--package-limit-bytes` が定義ファイルの `limits` に反映される。
#[test]
fn req30_limit_options_reach_the_definition_limits() {
    let e = Env::new();
    let o = e.run(
        &with_work(
            &e,
            &[
                "--items",
                "C",
                "--p95-limit-us",
                "1234",
                "--package-limit-bytes=4321",
            ],
        ),
        &[],
    );
    assert_eq!(o.code, Some(0), "stdout={}", o.stdout);
    let d1 = fs::read_to_string(e.dir.join("def-C1.json")).expect("C1 definition");
    let d2 = fs::read_to_string(e.dir.join("def-C2.json")).expect("C2 definition");
    assert!(
        d1.contains("\"limits\": {\"max_infer_p95_us\": 1234}"),
        "{d1}"
    );
    assert!(
        d2.contains("\"limits\": {\"max_package_bytes\": 4321}"),
        "{d2}"
    );
    assert_eq!(e.q("items.C.p95.limit_us"), "1234");
    assert_eq!(e.q("items.C.capacity_limit.limit_bytes"), "4321");
    assert_eq!(e.q("options.p95_limit_us"), "1234");
    assert_eq!(e.q("options.package_limit_bytes"), "4321");
}

/// REQ-21・REQ-31: `--p95-limit-us` は 1 から `MAX_LIMIT_INFER_P95_US` の整数に限り、範囲外
/// （上限超過・0・16 桁）は何も起動せず exit 64。上限は core の定数から組み立てるため、
/// core が上限を変えるとスクリプト側のリテラルとの食い違いでこのテストが落ちる。
#[test]
fn req31_p95_limit_out_of_range_is_rejected_before_start() {
    let message = format!("--p95-limit-us must be an integer from 1 to {MAX_LIMIT_INFER_P95_US}");
    let over = (MAX_LIMIT_INFER_P95_US + 1).to_string();
    for bad in [over.as_str(), "0", "1234567890123456"] {
        let e = Env::new();
        assert_rejected_before_start(
            &e,
            &with_work(&e, &["--items", "C", "--p95-limit-us", bad]),
            &[],
            &message,
        );
    }
}

/// REQ-31: `--p95-limit-us` の上限ちょうど（`MAX_LIMIT_INFER_P95_US`）は引数の検証を通り、
/// 項目 C の記録へその値が入る（core の定数との対応。範囲外側は上のテスト）。
#[test]
fn req31_p95_limit_at_max_is_accepted() {
    let max = MAX_LIMIT_INFER_P95_US.to_string();
    let e = Env::new();
    let o = e.run(
        &with_work(&e, &["--items", "C", "--p95-limit-us", &max]),
        &[],
    );
    assert_eq!(o.code, Some(0), "stdout={}", o.stdout);
    assert_eq!(e.q("items.C.status"), "\"ok\"");
    assert_eq!(e.q("items.C.p95.limit_us"), max);
    assert_eq!(e.q("options.p95_limit_us"), max);
}

/// REQ-33: `FANDHE_EDGE_BIN` を設定しない経路では、偽 cargo の `compiler-artifact` の
/// `executable` が CLI として使われ、`bin_override` が false・`cli_profile` が `release` になる。
#[test]
fn req33_unset_bin_uses_cargo_reported_executable() {
    let e = Env::new();
    let o = e.run(
        &with_work(&e, &["--items", "B"]),
        &[("FANDHE_EDGE_BIN", "")],
    );
    assert_eq!(o.code, Some(0), "stdout={}", o.stdout);
    assert_eq!(
        e.lines("cargo.args"),
        ["build --locked --release -p fandhe-edge-cli --bin fandhe-edge --message-format=json"]
    );
    assert_eq!(e.count_calls("register B"), 1);
    assert_eq!(e.q("bin_override"), "false");
    assert_eq!(e.q("environment.cli_profile"), "\"release\"");
    assert_eq!(e.q("evidence_hint"), "\"test_harness\"");
}

/// 証拠種別: 上書き経路の `evidence_hint` は `test_harness` で、実機を表す値を出さず、
/// `record.md` の冒頭に `bin_override`・`evidence_hint`・harness かどうかが出る。
/// `bin_override` が true のときだけ差し替えの注意行が出る。
/// 上書きなしの `requires_human_review` 経路は実 make・実 cargo を呼ぶため、ここでは走らせない。
#[test]
fn evidence_hint_and_record_md_header_for_harness_run() {
    let e = Env::new();
    let o = e.run(&with_work(&e, &["--items", "D"]), &[]);
    assert_eq!(o.code, Some(0), "stdout={}", o.stdout);
    let hint = e.q("evidence_hint");
    assert_eq!(hint, "\"test_harness\"");
    assert!(!hint.contains("real_machine"));
    let md = e.text("record.md");
    assert!(md.contains("- bin_override: True\n"), "{md}");
    assert!(md.contains("- evidence_hint: test_harness\n"), "{md}");
    assert!(
        md.contains("- harness（make・cargo の代役）: はい\n"),
        "{md}"
    );
    assert!(
        md.contains("- 注意: CLI を `FANDHE_EDGE_BIN` で差し替えた"),
        "{md}"
    );

    let e = Env::new();
    let o = e.run(
        &with_work(&e, &["--items", "D"]),
        &[("FANDHE_EDGE_BIN", "")],
    );
    assert_eq!(o.code, Some(0), "stdout={}", o.stdout);
    let md = e.text("record.md");
    assert!(md.contains("- bin_override: False\n"), "{md}");
    assert!(!md.contains("注意: CLI を"), "{md}");
}

/// REQ-21: SIGTERM を受けたら子のグループを止め、実行中の項目を `interrupted`、残りを `not_run` として
/// record を書き、exit 70 と固定メッセージを返す。子プロセスは残らない。
#[test]
fn req21_sigterm_stops_children_and_records_interrupted() {
    let e = Env::new();
    let mut child = e
        .command(
            &with_work(&e, &["--items", "D,F", "--repeat", "2"]),
            &[("FAKE_MAKE_SLEEP", "1")],
            None,
        )
        .spawn()
        .expect("spawn");
    let started = e.dir.join("make.started");
    let start = Instant::now();
    while !started.exists() {
        assert!(
            child.try_wait().expect("try_wait").is_none(),
            "script exited early"
        );
        assert!(
            start.elapsed() < Duration::from_secs(30),
            "make not started"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    Command::new("kill")
        .args(["-TERM", &child.id().to_string()])
        .status()
        .expect("kill");
    let status = loop {
        if let Some(st) = child.try_wait().expect("try_wait") {
            break st;
        }
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "script did not stop"
        );
        std::thread::sleep(Duration::from_millis(20));
    };
    let mut out = String::new();
    child
        .stdout
        .take()
        .expect("stdout")
        .read_to_string(&mut out)
        .expect("read");
    assert_eq!(status.code(), Some(70), "stdout={out}");
    assert_eq!(
        out,
        "{\"code\":\"runtime_error\",\"message\":\"interrupted\",\"record\":\"record.json\"}\n"
    );
    assert_eq!(e.q("items.D.status"), "\"failed\"");
    assert_eq!(e.q("items.D.reason"), "\"interrupted\"");
    assert_eq!(e.q("items.F.status"), "\"not_run\"");
    assert_eq!(e.q("items.F.reason"), "\"interrupted\"");
    assert!(e.lines("cargo.args").is_empty());
    // 偽 make と、その子の sleep が残っていない（kill -0 が失敗するまで短く待つ）
    for name in ["make.pid", "make.cpid"] {
        let pid = e.lines(name).first().cloned().expect("pid");
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let alive = Command::new("kill")
                .args(["-0", &pid])
                .stderr(Stdio::null())
                .status()
                .expect("kill -0")
                .success();
            if !alive {
                break;
            }
            assert!(Instant::now() < deadline, "{name} ({pid}) still alive");
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

/// REQ-30: C-2 で CLI が指定値と違う `limit_bytes` を返したら（上限が伝わっていない）C は failed
/// （`unexpected_output`）・exit 10。`total_bytes <= limit_bytes` なのに `exceeded:true`・exit 20 を
/// 返した場合も failed（package の報告値の不整合。`unexpected_output`）。
#[test]
fn req30_c2_wrong_limit_or_inconsistent_excess_fails() {
    let e = Env::new();
    let o = e.run(
        &with_work(&e, &["--items", "C"]),
        &[("FAKE_C2_LIMIT_WRONG", "1")],
    );
    assert_eq!(o.code, Some(10), "stdout={}", o.stdout);
    assert_eq!(o.stdout, JUDGED_FAIL);
    assert_eq!(e.q("items.C.status"), "\"failed\"");
    assert_eq!(e.q("items.C.reason"), "\"unexpected_output\"");

    let e = Env::new();
    let o = e.run(
        &with_work(&e, &["--items", "C"]),
        &[("FAKE_C2_MODE", "small")],
    );
    assert_eq!(o.code, Some(10), "stdout={}", o.stdout);
    // total_bytes <= limit_bytes の exceeded:true は package の報告値の不整合として工程で止まる
    assert_eq!(e.q("items.C.status"), "\"failed\"");
    assert_eq!(e.q("items.C.reason"), "\"unexpected_output\"");
    assert_eq!(e.q("items.C.case"), "\"C-2\"");
    assert_eq!(e.q("items.C.step"), "\"package\"");
    assert_eq!(e.q("items.C.exit_code"), "20");
}

/// REQ-31: C-1 で `p95_us` が上限未満なのに `exceeded:true`・exit 20 を返したら C は failed
/// （`unexpected_output`）。
#[test]
fn req31_c1_inconsistent_p95_excess_fails() {
    let e = Env::new();
    let o = e.run(
        &with_work(&e, &["--items", "C"]),
        &[("FAKE_C1_EXCEED", "low")],
    );
    assert_eq!(o.code, Some(10), "stdout={}", o.stdout);
    assert_eq!(e.q("items.C.status"), "\"failed\"");
    assert_eq!(e.q("items.C.reason"), "\"unexpected_output\"");
    assert_eq!(e.q("items.C.case"), "\"C-1\"");
}

/// REQ-39: 偽 make が TERM を無視する孫を残して正常終了しても、スクリプトの終了後に孫が残らない。
#[test]
fn req39_term_ignoring_grandchild_does_not_outlive_script() {
    let e = Env::new();
    let o = e.run(
        &with_work(&e, &["--items", "D"]),
        &[("FAKE_MAKE_ORPHAN", "1")],
    );
    assert_eq!(o.code, Some(0), "stdout={}", o.stdout);
    let pid = e.lines("orphan.pid").first().cloned().expect("pid");
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let alive = Command::new("kill")
            .args(["-0", &pid])
            .stderr(Stdio::null())
            .status()
            .expect("kill -0")
            .success();
        if !alive {
            break;
        }
        assert!(Instant::now() < deadline, "grandchild {pid} still alive");
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// REQ-33: evaluate の `n_total` が評価データの行数（12）と違えば B は failed（`unexpected_output`）で
/// exit 10。後続の工程は起動されない。
#[test]
fn req33_evaluate_n_total_mismatch_fails_b() {
    let e = Env::new();
    let o = e.run(&with_work(&e, &["--items", "B"]), &[("FAKE_EVAL_N", "13")]);
    assert_eq!(o.code, Some(10), "stdout={}", o.stdout);
    assert_eq!(o.stdout, JUDGED_FAIL);
    assert_eq!(e.q("items.B.status"), "\"failed\"");
    assert_eq!(e.q("items.B.reason"), "\"unexpected_output\"");
    assert_eq!(e.q("items.B.step"), "\"evaluate\"");
    assert_eq!(e.count_calls("package B"), 0);
}

/// REQ-33: infer の `predicted_label` が定義の選択肢 ID でなければ B は failed（`unexpected_output`）。
#[test]
fn req33_infer_label_outside_options_fails_b() {
    let e = Env::new();
    let o = e.run(
        &with_work(&e, &["--items", "B"]),
        &[("FAKE_INFER_BAD_LABEL", "1")],
    );
    assert_eq!(o.code, Some(10), "stdout={}", o.stdout);
    assert_eq!(e.q("items.B.status"), "\"failed\"");
    assert_eq!(e.q("items.B.reason"), "\"unexpected_output\"");
    assert_eq!(e.q("items.B.step"), "\"infer\"");
}

/// REQ-31: C-1 が exit 20 なのに `package/` が残っていれば C は failed（`unexpected_output`）。
#[test]
fn req31_c1_exit_20_with_published_package_fails() {
    let e = Env::new();
    let o = e.run(
        &with_work(&e, &["--items", "C"]),
        &[("FAKE_C1_EXCEED", "1"), ("FAKE_C1_KEEPDIR", "1")],
    );
    assert_eq!(o.code, Some(10), "stdout={}", o.stdout);
    assert_eq!(e.q("items.C.status"), "\"failed\"");
    assert_eq!(e.q("items.C.reason"), "\"unexpected_output\"");
    assert_eq!(e.q("items.C.case"), "\"C-1\"");
    assert_eq!(e.q("items.C.exit_code"), "20");
}

/// REQ-32: D は `ok: req32_` が 3 件でない（2 件）と failed（`unexpected_output`）で exit 10。
#[test]
fn req32_item_d_requires_exactly_three_env_i_tests() {
    let e = Env::new();
    let o = e.run(
        &with_work(&e, &["--items", "D"]),
        &[("FAKE_LINK_MODE", "two")],
    );
    assert_eq!(o.code, Some(10), "stdout={}", o.stdout);
    assert_eq!(o.stdout, JUDGED_FAIL);
    assert_eq!(e.q("items.D.status"), "\"failed\"");
    assert_eq!(e.q("items.D.reason"), "\"unexpected_output\"");
    assert_eq!(e.q("items.D.env_i_tests_ok"), "2");
}

/// REQ-28: バッチ出力に id の重複があれば（行数は合っていても）E は failed（`unexpected_output`）。
#[test]
fn req28_item_e_duplicate_batch_id_fails() {
    let e = Env::new();
    let o = e.run(&with_work(&e, &["--items", "B,E"]), &[("FAKE_E_DUP", "1")]);
    assert_eq!(o.code, Some(10), "stdout={}", o.stdout);
    assert_eq!(e.q("items.E.status"), "\"failed\"");
    assert_eq!(e.q("items.E.reason"), "\"unexpected_output\"");
    assert_eq!(e.q("items.E.step"), "\"infer-batch\"");
}

/// REQ-28: 同一実装の単体対バッチはスコアも含めて完全一致する。バッチの 1 行だけスコアが 1e-12
/// ずれていれば、許容差（1e-9）以内でも E は failed（`mismatch`）で、その行の id だけが
/// `mismatch-ids.txt` に載る。
#[test]
fn req28_item_e_score_difference_within_1e9_fails() {
    let e = Env::new();
    let o = e.run(
        &with_work(&e, &["--items", "B,E"]),
        &[("FAKE_BAD", "batch_score_shift")],
    );
    assert_eq!(o.code, Some(10), "stdout={}", o.stdout);
    assert_eq!(o.stdout, JUDGED_FAIL);
    assert_eq!(e.q("items.E.status"), "\"failed\"");
    assert_eq!(e.q("items.E.reason"), "\"mismatch\"");
    assert_eq!(e.q("items.E.records"), "90");
    assert_eq!(e.q("items.E.label_mismatch"), "0");
    assert_eq!(e.q("items.E.scores_nonfinite"), "0");
    assert_eq!(e.q("items.E.scores_exact_match"), "89");
    let diff: f64 = e.q("items.E.max_abs_score_diff").parse().expect("diff");
    assert!(diff > 0.0 && diff <= 1e-9, "diff={diff}");
    // ずらした行は 2 行目。その id だけが載る
    let batch = fs::read_to_string(e.dir.join("batch-input.jsonl")).expect("batch input");
    let second = batch.lines().nth(1).expect("second row");
    let id = second
        .split("\"id\": \"")
        .nth(1)
        .and_then(|r| r.split('"').next())
        .expect("id");
    assert_eq!(e.text("E/mismatch-ids.txt"), format!("{id}\n"));
}

/// 偽 CLI へ `FAKE_BAD` を与えて `items` を実行し、`item` が failed・exit 10 で、`reason`・`step`
/// が期待値どおりであることを確かめる（観測用の補助。期待値は実行して確かめた値）。
fn assert_bad_output_fails(
    items: &str,
    envs: &[(&str, &str)],
    item: &str,
    reason: &str,
    step: Option<&str>,
) -> Env {
    let e = Env::new();
    let o = e.run(&with_work(&e, &["--items", items]), envs);
    assert_eq!(o.code, Some(10), "{envs:?}: {}", o.stdout);
    assert_eq!(o.stdout, JUDGED_FAIL, "{envs:?}");
    assert_eq!(
        e.q(&format!("items.{item}.status")),
        "\"failed\"",
        "{envs:?}"
    );
    assert_eq!(
        e.q(&format!("items.{item}.reason")),
        format!("\"{reason}\""),
        "{envs:?}"
    );
    if let Some(st) = step {
        assert_eq!(
            e.q(&format!("items.{item}.step")),
            format!("\"{st}\""),
            "{envs:?}"
        );
    }
    e
}

/// REQ-30・REQ-31・REQ-33: CLI の出力が契約と矛盾すると（容量の内訳が合計と違う・`select` の `kind` が
/// `train` と違う・`predicted_label` が最大スコアでない・スコアが範囲外・`p95_us` が負数・`candidate` が
/// 真偽値・`accuracy` が `correct / n_total` と違う・`judgment` の欠落・`limit_bytes` が既定値でない・
/// `package/` の合計が `total_bytes` と違う）、その項目は failed（`unexpected_output`）・exit 10。
#[test]
fn req33_contradicting_cli_output_fails_the_item() {
    // (FAKE_BAD, items, 失敗する項目, 失敗した工程)
    let cases: [(&str, &str, &str, &str); 13] = [
        ("sum:C1", "C", "C", "package"),
        ("sum:C2", "C", "C", "package"),
        ("select_kind", "B", "B", "select"),
        ("infer_notmax", "B", "B", "infer"),
        ("scores_high", "B", "B", "infer"),
        ("scores_neg", "B", "B", "infer"),
        ("p95neg:C1", "C", "C", "package"),
        ("cand_false", "B", "B", "train"),
        ("accuracy", "B", "B", "evaluate"),
        ("nojudgment:B", "B", "B", "package"),
        ("nojudgment:C1", "C", "C", "package"),
        ("limit:B", "B", "B", "package"),
        ("pkgsum:B", "B", "B", "package"),
    ];
    for (bad, items, item, step) in cases {
        assert_bad_output_fails(
            items,
            &[("FAKE_BAD", bad)],
            item,
            "unexpected_output",
            Some(step),
        );
    }
}

/// REQ-28: E のバッチ出力の `predicted_label` が最大スコアでなければ（スコアは単体と同じ）E は failed
/// （`unexpected_output`）で、不一致（`mismatch`）とは区別される。
#[test]
fn req28_item_e_batch_label_not_max_score_fails() {
    assert_bad_output_fails(
        "B,E",
        &[("FAKE_BAD", "batch_notmax")],
        "E",
        "unexpected_output",
        Some("infer-batch"),
    );
}

/// REQ-21・REQ-33・security.md: 許可リスト外の欄（本文らしい文字列・入れ子の dict・list）は、
/// 各工程の JSON へ足しても `record.json`・`record.md`・stdout のどこにも出ない。
/// 工程の要約は決まった欄だけの新しい dict になる。
#[test]
fn req39_summary_is_built_from_allowlisted_fields_only() {
    let e = Env::new();
    let o = e.run(
        &with_work(&e, &["--items", "B,C,E", "--repeat", "1"]),
        &[("FAKE_LEAK", "1")],
    );
    assert_eq!(o.code, Some(0), "stdout={}", o.stdout);
    let json = e.text("record.json");
    let md = e.text("record.md");
    for (name, text) in [
        ("record.json", &json),
        ("record.md", &md),
        ("stdout", &o.stdout),
    ] {
        assert!(!text.contains(LEAK_BODY), "{name} contains the body");
    }
    for key in ["body", "nested", "arr", "note"] {
        assert!(!json.contains(&format!("\"{key}\"")), "{key} key kept");
    }
    assert_eq!(
        e.q("items.B.steps.0.summary"),
        "{\"definition_sha256\":\"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\",\
         \"evaluation_defined\":true,\"options\":3,\"status\":\"ok\",\"step\":\"register\"}"
    );
    assert_eq!(
        e.q("items.B.steps.2.summary"),
        "{\"candidate\":0,\"kind\":\"c1\",\"status\":\"ok\",\"step\":\"train\"}"
    );
    // infer の要約は選択肢 ID（利用者が決める文字列）を記録せず、何番目かだけを残す
    assert_eq!(
        e.q("items.B.steps.6.summary"),
        "{\"predicted_index\":0,\"scores_keys\":3,\"status\":\"ok\"}"
    );
}

/// REQ-21・security.md: 失敗した工程の `message`（パス文字を含まない本文らしい文字列）は記録せず、
/// `message_bytes`（UTF-8 のバイト数）と `message_sha256` だけが出る。
#[test]
fn req39_failed_stage_message_is_recorded_as_size_and_hash_only() {
    let e = Env::new();
    let o = e.run(
        &with_work(&e, &["--items", "B"]),
        &[
            ("FAKE_FAIL_STAGE", "inspect"),
            ("FAKE_FAIL_MSG", "SECRET_BODY_7c1 and more"),
        ],
    );
    assert_eq!(o.code, Some(10), "stdout={}", o.stdout);
    for (name, text) in [
        ("record.json", e.text("record.json")),
        ("record.md", e.text("record.md")),
        ("stdout", o.stdout.clone()),
    ] {
        assert!(!text.contains(LEAK_BODY), "{name} contains the message");
    }
    assert_eq!(e.q("items.B.step"), "\"inspect\"");
    assert_eq!(e.q("items.B.code"), "\"runtime_error\"");
    assert_eq!(e.q("items.B.message_bytes"), "24");
    assert_eq!(
        e.q("items.B.message_sha256"),
        "\"93f946f9a72f3acc1597af6ab33b70624eed2a7ad7556da5ff3567c1a7c84f9f\""
    );
}

/// REQ-21・security.md: `infer` の `status` や C-2 の `code` が入れ子の dict でも、中の文字列は記録に出ない。
#[test]
fn req39_nested_status_and_code_do_not_reach_record() {
    let e = Env::new();
    let o = e.run(
        &with_work(&e, &["--items", "B"]),
        &[("FAKE_INFER_STATUS_NESTED", "1")],
    );
    assert_eq!(o.code, Some(10), "stdout={}", o.stdout);
    assert_eq!(e.q("items.B.status"), "\"failed\"");
    assert_eq!(e.q("items.B.reason"), "\"unexpected_output\"");
    assert_eq!(e.q("items.B.step"), "\"infer\"");
    for text in [e.text("record.json"), e.text("record.md"), o.stdout.clone()] {
        assert!(!text.contains(LEAK_BODY));
    }
    assert_eq!(e.q("items.B.steps.6.summary.status"), "null");

    let e = Env::new();
    let o = e.run(
        &with_work(&e, &["--items", "C"]),
        &[("FAKE_C2_CODE_NESTED", "1")],
    );
    assert_eq!(o.code, Some(10), "stdout={}", o.stdout);
    assert_eq!(e.q("items.C.status"), "\"failed\"");
    for text in [e.text("record.json"), e.text("record.md"), o.stdout.clone()] {
        assert!(!text.contains(LEAK_BODY));
    }
}

/// REQ-30・security.md: `package/` に規則外の名前（空白を含む）のファイルがあると、
/// 記録の `package_files[].name` は `<unrecognized>` になり、元の名前は出ない。
#[test]
fn req39_unrecognized_package_file_name_is_not_recorded() {
    let e = Env::new();
    let o = e.run(&with_work(&e, &["--items", "B"]), &[("FAKE_PKG_ODD", "1")]);
    assert_eq!(o.code, Some(0), "stdout={}", o.stdout);
    let json = e.text("record.json");
    assert!(!json.contains("odd name"), "{json}");
    assert!(!e.text("record.md").contains("odd name"));
    assert_eq!(
        e.q("items.B.package_files.*.name"),
        "[\"artifact.json\",\"model.onnx\",\"<unrecognized>\"]"
    );
}
