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
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{self, Receiver};
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
echo "${RUSTUP_AUTO_INSTALL:-unset}" >> "$FAKE_DIR/cli.rustup"
[ "$stage" = register ] && cp definition.json "$FAKE_DIR/def-$here.json"
SC='{"alpha":0.5,"beta":0.25,"gamma":0.25}'
bad=${FAKE_BAD:-}
[ "$bad" = scores_high ] && SC='{"alpha":1.5,"beta":0.0,"gamma":-0.5}'
[ "$bad" = scores_neg ] && SC='{"alpha":0.75,"beta":0.5,"gamma":-0.25}'
SHA=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
p95lim=$(sed -n 's/.*"max_infer_p95_us": *\([0-9]*\).*/\1/p' definition.json 2>/dev/null)
[ -n "$p95lim" ] || p95lim=50000
pkglim=$(sed -n 's/.*"max_package_bytes": *\([0-9]*\).*/\1/p' definition.json 2>/dev/null)
# 定義に上限が無ければ実 CLI は `limit_bytes:null`（REQ-30）。算術用の既定値は別に持つ
pkgjson=${pkglim:-null}
[ -n "$pkglim" ] || pkglim=40000000
G='"guideline_bytes":40000000,"over_guideline":false'
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
# 内訳の file_count は、公開される package/ の実ファイル数（model.onnx と artifact.json の 2 つ。
# FAKE_PKG_ODD なら `odd name.bin` を足した 3 つ）と一致させる。FAKE_FC_SHIFT は weights を 1 多く報告する
wfc=1
[ -n "${FAKE_FC_SHIFT:-}" ] && wfc=2
ltfc=0
[ -n "${FAKE_PKG_ODD:-}" ] && ltfc=1
comps=$(printf '"components":{"weights":{"bytes":100,"file_count":%s},"vocab_or_feature_transform":{"bytes":20,"file_count":0},"label_table":{"bytes":5,"file_count":%s},"calibration":{"bytes":3,"file_count":0},"metadata":{"bytes":7,"file_count":1}}' "$wfc" "$ltfc")
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
  # 組み立て先が残る欠陥の再現（値は残す cwd 名 C1・C2。exit 0・20 のどちらでも残る）
  [ "${FAKE_STAGING_LEFT:-}" = "$here" ] && mkdir -p project/package.staging
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
    printf '{"code":%s,"message":"resource limit exceeded","step":"package","capacity":{"total_bytes":%s,"limit_bytes":%s,"exceeded":true,%s,%s},"infer_p95":null}\n' "$code" "$total" "$rl" "$G" "$c2comps"
    exit 20
  fi
  if [ "$here" = C1 ] && [ -n "${FAKE_C1_EXCEED:-}" ]; then
    # low は p95_us が上限未満なのに exceeded:true・exit 20 を返す不正な出力
    pv=99999
    [ "$FAKE_C1_EXCEED" = low ] && pv=2
    [ -n "${FAKE_C1_KEEPDIR:-}" ] && mkdir -p project/package
    printf '{"code":"limit_exceeded","message":"resource limit exceeded","step":"package","capacity":{"total_bytes":135,"limit_bytes":%s,"exceeded":false,%s,%s},"infer_p95":{"p95_us":%s,"limit_us":%s,"exceeded":true}}\n' "$pkgjson" "$G" "$comps" "$pv" "$p95lim"
    exit 20
  fi
  total=$((135 + extrab))
  lim=$pkgjson
  [ "$bad" = "limit:$here" ] && lim=39999999
  # package/ の通常ファイルの合計は total_bytes と一致させる（pkgsum は 1 バイト少なくする）
  odd=0
  [ -n "${FAKE_PKG_ODD:-}" ] && odd=10
  mb=$((total - 35 - odd))
  [ "$bad" = "pkgsum:$here" ] && mb=$((mb - 1))
  mkdir -p project/package
  head -c "$mb" /dev/zero > project/package/model.onnx
  head -c 35 /dev/zero > project/package/artifact.json
  # 通常ファイル以外の混入の再現（計測対象外のまま見逃されないことの確認）
  [ "${FAKE_PKG_KIND:-}" = symlink ] && ln -s model.onnx project/package/link
  [ "${FAKE_PKG_KIND:-}" = dir ] && mkdir project/package/sub
  [ "$odd" != 0 ] && head -c "$odd" /dev/zero > "project/package/odd name.bin"
  p95=null
  pv=2
  [ "$bad" = "p95neg:$here" ] && pv=-1
  [ "$here" = C1 ] && p95=$(printf '{"p95_us":%s,"limit_us":%s,"exceeded":false}' "$pv" "$p95lim")
  jfield='"judgment":null,'
  [ "$bad" = "nojudgment:$here" ] && jfield=
  printf '{"step":"package","status":"ok",%s"acceptance_defined":false,"capacity":{"total_bytes":%s,"limit_bytes":%s,"exceeded":false,%s,%s},"infer_p95":%s%s}\n' "$jfield" "$total" "$lim" "$G" "$comps" "$p95" "$extra" ;;
infer)
  file=
  id=${FAKE_DEFAULT_ID:-input}
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
    {
    while IFS= read -r line; do
      rid=$(printf '%s' "$line" | sed -n 's/^{"id": "\([^"]*\)".*/\1/p')
      [ -n "$first" ] || first=$rid
      n=$((n + 1))
      [ -n "${FAKE_E_DUP:-}" ] && [ "$n" = 2 ] && rid=$first
      rowsc=$BSC
      [ "$n" = 2 ] && rowsc=$SC2
      printf '{"id":"%s","status":"ok","predicted_label":"%s","scores":'"$rowsc"'}\n' "$rid" "$lab"
    done < "$file"
    } | if [ "${FAKE_E_ORDER:-}" = swap ]; then
      # 1 行目と 2 行目の出力順を入れ替える（入力順の保証の違反）
      awk 'NR==1{a=$0;next} NR==2{print $0; print a; next} {print}'
    else
      cat
    fi
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
echo "$1 ${RUSTUP_AUTO_INSTALL:-unset}" >> "$FAKE_DIR/make.rustup"
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
    sleep 300 &
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
/// （カンマ区切り。`ok`・`inc`・`plain`・`zero`・`short`・`ignored`・`killed`・`overflow`・`unexec`）の
/// n 番目で n 回目の結果を決める。`-- --list` も回数に数えず、`FAKE_LIST_COUNT`（既定 12）件を出す
/// （`FAKE_LIST_RC` で終了コードを変える）。
/// `FAKE_CARGO_BUILD_SLEEP`（`build` で長く待つ。ビルド中の中断のテスト用）・
/// `FAKE_CARGO_BUILD_FAIL`（`build` が失敗する）・
/// `FAKE_TOOLCHAIN_MISSING`（rustup プロキシの代役。ツールチェーン未導入で、`RUSTUP_AUTO_INSTALL` が
/// `0` 以外なら取得を試みた印 `rustup.download` を作り、`0` なら取得せず exit 1。REQ-38・#375）。
const FAKE_CARGO: &str = r##"#!/bin/sh
printf '%s\n' "$*" >> "$FAKE_DIR/cargo.args"
echo "${CARGO_NET_OFFLINE:-unset}" >> "$FAKE_DIR/cargo.env"
echo "${RUSTUP_AUTO_INSTALL:-unset}" >> "$FAKE_DIR/cargo.rustup"
if [ -n "${FAKE_TOOLCHAIN_MISSING:-}" ]; then
  if [ "${RUSTUP_AUTO_INSTALL:-}" != 0 ]; then
    : > "$FAKE_DIR/rustup.download"
    exit 0
  fi
  echo "error: toolchain 'stable' is not installed" >&2
  exit 1
fi
if [ "$1" = build ]; then
  # FAKE_CARGO_BUILD_FAIL: ビルド失敗（スクリプト自身の実行不能 70 の経路）
  [ -n "${FAKE_CARGO_BUILD_FAIL:-}" ] && exit 1
  if [ -n "${FAKE_CARGO_BUILD_SLEEP:-}" ]; then
    echo $$ > "$FAKE_DIR/cargo.pid"
    sleep 300 &
    echo $! > "$FAKE_DIR/cargo.cpid"
    : > "$FAKE_DIR/cargo.started"
    wait
  fi
  printf '{"reason":"compiler-artifact","target":{"name":"fandhe-edge","kind":["bin"]},"executable":"%s"}\n' "$FAKE_CLI_PATH"
  exit 0
fi
case "$*" in *--no-run*) exit 0 ;; esac
case "$*" in
*--list*)
  # `-- --list` は回数に数えない。FAKE_LIST_COUNT 件のテストを `<name>: test` で出す
  c=${FAKE_LIST_COUNT:-12}
  i=1
  while [ "$i" -le "$c" ]; do echo "tests::t$i: test"; i=$((i + 1)); done
  echo
  echo "$c tests, 0 benchmarks"
  exit "${FAKE_LIST_RC:-0}" ;;
esac
n=$(( $(cat "$FAKE_DIR/cargo.count" 2>/dev/null || echo 0) + 1 ))
echo "$n" > "$FAKE_DIR/cargo.count"
mode=$(echo "${FAKE_CARGO_PATTERN:-}" | cut -d, -f"$n")
case "$mode" in
inc) echo "error: ReadOutputIncomplete"; exit 101 ;;
plain) echo "error: ReadOutput(Io)"; exit 101 ;;
zero) echo "test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s"; exit 0 ;;
short) echo "test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s"; exit 0 ;;
ignored) echo "test result: ok. 12 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.01s"; exit 0 ;;
killed) kill -9 $PPID; exit 0 ;;
overflow) head -c 70000000 /dev/zero | tr '\0' 'x'; exit 0 ;;
unexec)
  # 自分自身の実行権限を外す。次の回の起動が spawn_error になる
  chmod 000 "$0"
  echo "test result: ok. 12 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s"; exit 0 ;;
*) echo "test result: ok. 12 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s"; exit 0 ;;
esac
"##;

struct Out {
    code: Option<i32>,
    stdout: String,
    stderr: String,
}

impl Out {
    /// 失敗時の表示用。stdout と stderr（上限つきで取得したもの）を並べる。
    fn diag(&self) -> String {
        format!("stdout={} stderr={}", self.stdout, self.stderr)
    }
}

/// 子の出力の読み取り上限（失敗時の診断用。超えた分は読み捨ててパイプを詰まらせない。REQ-39）。
const CAP_DIAG: u64 = 64 * 1024;
/// 読み取りスレッドの結果を待つ上限（孫がパイプを握り続けても固まらない）。
const DRAIN_WAIT: Duration = Duration::from_secs(10);

/// 子の出力を別スレッドで上限つきで読む。先頭 `CAP_DIAG` バイトを結果として送り、残りは読み捨てる。
fn drain<R: Read + Send + 'static>(mut r: R) -> Receiver<String> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = (&mut r).take(CAP_DIAG).read_to_end(&mut buf);
        let _ = tx.send(String::from_utf8_lossy(&buf).into_owned());
        let _ = std::io::copy(&mut r, &mut std::io::sink());
    });
    rx
}

fn collect(rx: &Receiver<String>) -> String {
    rx.recv_timeout(DRAIN_WAIT).unwrap_or_default()
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
        self.command_inner(args, envs, cwd, false)
    }

    /// `closed_fd` が真なら、補助の `sh -c 'exec ... >&-'` 経由で fd 1 を閉じた状態で起動する
    /// （`sys.stdout is None` の経路。macOS・Linux 共通で動く）。
    fn command_inner(
        &self,
        args: &[String],
        envs: &[(&str, &str)],
        cwd: Option<&Path>,
        closed_fd: bool,
    ) -> Command {
        let mut cmd = Command::new("sh");
        if closed_fd {
            cmd.args(["-c", "exec sh \"$@\" >&-", "sh"]);
        }
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
            .stderr(Stdio::piped());
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
        let rx_out = drain(child.stdout.take().expect("stdout"));
        let rx_err = drain(child.stderr.take().expect("stderr"));
        let pgid = child.id();
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
                panic!(
                    "script timed out: stdout={} stderr={}",
                    collect(&rx_out),
                    collect(&rx_err)
                );
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        Out {
            code: status.code(),
            stdout: collect(&rx_out),
            stderr: collect(&rx_err),
        }
    }

    /// stdout を `kind` の状態にして起動し、終了コードと stderr だけを返す（REQ-21）。
    fn run_with_stdout(
        &self,
        args: &[String],
        envs: &[(&str, &str)],
        kind: StdoutKind,
    ) -> (Option<i32>, String) {
        let mut cmd = self.command_inner(args, envs, None, matches!(kind, StdoutKind::ClosedFd));
        match kind {
            StdoutKind::ClosedPipe => {
                let (reader, writer) = std::io::pipe().expect("pipe");
                drop(reader);
                cmd.stdout(Stdio::from(writer));
            }
            StdoutKind::ClosedFd => {}
            // macOS に /dev/full は無いため Linux に局所化する。閉じたパイプ・閉じた fd の検証は全 OS で行う
            #[cfg(target_os = "linux")]
            StdoutKind::Full => {
                let f = fs::OpenOptions::new()
                    .write(true)
                    .open("/dev/full")
                    .expect("open /dev/full");
                cmd.stdout(Stdio::from(f));
            }
        }
        let mut child = cmd.spawn().expect("spawn sh");
        let rx_err = drain(child.stderr.take().expect("stderr"));
        let pgid = child.id();
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
                panic!("script timed out: stderr={}", collect(&rx_err));
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        (status.code(), collect(&rx_err))
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
/// stdout の状態（REQ-21: 最終 JSON を書けない場合の終了コードの確認用）。
#[derive(Clone, Copy)]
enum StdoutKind {
    /// 読み手が先に閉じたパイプ（EPIPE）
    ClosedPipe,
    /// fd 1 が閉じている（Python 側は `sys.stdout is None`）
    ClosedFd,
    /// 書き込みが ENOSPC になる `/dev/full`
    #[cfg(target_os = "linux")]
    Full,
}

fn assert_stdout_failure_exit_70(e: &Env, kind: StdoutKind) {
    let mut args = e.work_arg();
    args.extend(["--items".into(), "D".into()]);
    let (code, err) = e.run_with_stdout(&args, &[], kind);
    assert_eq!(code, Some(70), "stderr={err}");
    assert!(!err.contains("Traceback"), "stderr={err}");
    assert!(!err.contains("Exception ignored"), "stderr={err}");
    assert!(e.work.join("record.json").is_file());
    assert!(e.work.join("record.md").is_file());
    // stdout の失敗が項目の判定を書き換えない
    assert_eq!(e.q("items.D.status"), "\"ok\"");
}

/// REQ-21: 読み手が閉じたパイプでも exit 70 で、record は書き終えている。
#[test]
fn req21_stdout_closed_pipe_exits_70_after_writing_record() {
    assert_stdout_failure_exit_70(&Env::new(), StdoutKind::ClosedPipe);
}

/// REQ-21: stdout が閉じていても exit 70 で、record は書き終えている。
#[test]
fn req21_stdout_closed_fd_exits_70_after_writing_record() {
    assert_stdout_failure_exit_70(&Env::new(), StdoutKind::ClosedFd);
}

/// REQ-21: 書き込めない stdout（`/dev/full`）でも exit 70。
#[cfg(target_os = "linux")]
#[test]
fn req21_stdout_full_exits_70_after_writing_record() {
    assert_stdout_failure_exit_70(&Env::new(), StdoutKind::Full);
}

/// REQ-21: 早期終了（shell の `fail()`）・引数エラー・判定失敗でも、stdout が書けなければ 141・1 でなく 70。
#[test]
fn req21_stdout_failure_in_early_exits_maps_to_70() {
    type Case = (Vec<String>, Vec<(&'static str, &'static str)>);
    let cases: Vec<Case> = vec![
        (vec!["--help".into()], vec![]),
        (vec!["--items".into(), "Z".into()], vec![]),
        (vec!["--items".into(), "A".into()], vec![]),
        (
            vec!["--items".into(), "D".into()],
            vec![("FAKE_FAIL_STAGE", "train")],
        ),
    ];
    for kind in [StdoutKind::ClosedPipe, StdoutKind::ClosedFd] {
        for (extra, envs) in &cases {
            let e = Env::new();
            let mut args = e.work_arg();
            args.extend(extra.clone());
            let (code, err) = e.run_with_stdout(&args, envs, kind);
            assert_eq!(code, Some(70), "args={args:?} stderr={err}");
            assert!(!err.contains("Traceback"), "args={args:?} stderr={err}");
            assert!(
                !err.contains("Exception ignored"),
                "args={args:?} stderr={err}"
            );
        }
    }
}

const REPEAT_MSG: &str = "--repeat must be an integer from 1 to 1000";
const JUDGED_FAIL: &str = "{\"code\":\"judged_fail\",\"message\":\"one or more requested items failed or were not run\",\"record\":\"record.json\"}\n";

/// 引数と envs の組で検証失敗（exit 64・固定メッセージ）になり、何も起動されず、
/// 渡した `--work-dir` も作られていないことを確かめる。
fn assert_rejected_before_start(e: &Env, args: &[String], envs: &[(&str, &str)], message: &str) {
    let o = e.run(args, envs);
    assert_eq!(o.code, Some(64), "args={args:?} {}", o.diag());
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
    // 存在しない・実行権限が無い・ディレクトリ・親が無い FANDHE_EDGE_BIN は起動前に 64（70 ではない）
    let bin_msg = "FANDHE_EDGE_BIN must be a path to an executable file";
    let e = Env::new();
    let not_exec = e.dir.join("not-exec");
    fs::write(&not_exec, "#!/bin/sh\n").expect("write");
    fs::set_permissions(&not_exec, fs::Permissions::from_mode(0o600)).expect("chmod");
    let a_dir = e.dir.join("a-dir");
    fs::create_dir_all(&a_dir).expect("mkdir");
    let missing = e.dir.join("missing");
    let no_parent = e.dir.join("no-such-dir").join("fandhe-edge");
    for bad in [&missing, &not_exec, &a_dir, &no_parent] {
        assert_rejected_before_start(
            &e,
            &with_work(&e, &["--items", "B"]),
            &[("FANDHE_EDGE_BIN", &bad.display().to_string())],
            bin_msg,
        );
    }

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
    assert_eq!(o.code, Some(0), "{}", o.diag());
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
    assert_eq!(o.code, Some(0), "{}", o.diag());
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
    // 終了時の再採取（#360）: CLI は差し替え（env_override）で、開始時と終了時のハッシュが一致する
    assert_eq!(e.q("environment.cli_origin"), "\"env_override\"");
    assert_eq!(e.q("environment.cli_unchanged"), "true");
    assert_eq!(
        e.q("environment.cli_end_sha256"),
        e.q("environment.cli_sha256")
    );
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
    assert_eq!(e.q("items.F.expected_tests"), "12");
    for k in ["count_mismatch", "output_limit", "killed", "spawn_error"] {
        assert_eq!(e.q(&format!("items.F.{k}")), "0", "{k}");
    }
    // `--no-run` と `--list` の各 1 回 + 本番 3 回。前者 2 つは回数に数えない
    assert_eq!(e.lines("cargo.count"), ["3"]);
    let cargo = e.lines("cargo.args");
    assert_eq!(
        cargo[0],
        "test --locked -p fandhe-edge-guard --test time_limit --no-run"
    );
    assert_eq!(
        cargo[1],
        "test --locked -p fandhe-edge-guard --test time_limit -- --list"
    );
    assert_eq!(cargo.len(), 5);
    assert!(
        cargo[2..]
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
    assert_eq!(o.code, Some(0), "{}", o.diag());
    assert_eq!(
        e.lines("make.env"),
        ["ci unset", "check-runtime-linkage true"]
    );
    // `--no-run`・`--list`・本番 1 回
    assert_eq!(e.lines("cargo.env"), ["true", "true", "true"]);
    // A を含む実行は make ci が offline で起動されないため cargo_offline は false（#360）
    assert_eq!(e.q("options.cargo_offline"), "false");
    let cli_env = e.lines("cli.env");
    assert_eq!(cli_env.len(), 7);
    assert!(cli_env.iter().all(|l| l == "true"));
}

/// REQ-38・#375: rustup の自動取得を止める `RUSTUP_AUTO_INSTALL=0` が、A の `make ci` を含む
/// 全ての子（make・cargo・CLI）へ届く。A には `CARGO_NET_OFFLINE` を渡さないまま。
#[test]
fn req38_rustup_auto_install_is_disabled_for_every_child() {
    let e = Env::new();
    let o = e.run(
        &with_work(
            &e,
            &["--items", "A,B,C,D,E,F", "--repeat", "1", "--with-ci"],
        ),
        &[],
    );
    assert_eq!(o.code, Some(0), "{}", o.diag());
    assert_eq!(e.lines("make.env")[0], "ci unset");
    assert_eq!(e.lines("make.rustup"), ["ci 0", "check-runtime-linkage 0"]);
    let cargo = e.lines("cargo.rustup");
    assert!(!cargo.is_empty());
    assert!(cargo.iter().all(|l| l == "0"), "{cargo:?}");
    let cli = e.lines("cli.rustup");
    assert!(!cli.is_empty());
    assert!(cli.iter().all(|l| l == "0"), "{cli:?}");
}

/// REQ-38・#375: 親が `RUSTUP_AUTO_INSTALL=1` を渡しても、子には `0` が届く。
#[test]
fn req38_parent_rustup_auto_install_is_overridden() {
    let e = Env::new();
    let o = e.run(
        &with_work(&e, &["--items", "A,D,F", "--repeat", "1", "--with-ci"]),
        &[("RUSTUP_AUTO_INSTALL", "1")],
    );
    assert_eq!(o.code, Some(0), "{}", o.diag());
    assert_eq!(e.lines("make.rustup"), ["ci 0", "check-runtime-linkage 0"]);
    assert!(e.lines("cargo.rustup").iter().all(|l| l == "0"));
}

/// REQ-38・#375: ツールチェーンが無くても取得は試みられず（`rustup.download` が作られない）、
/// 失敗は成功を装わず既存の分類（CLI ビルド失敗は exit 70）で見える。
#[test]
fn req38_missing_toolchain_does_not_download() {
    let e = Env::new();
    let o = e.run(
        &with_work(&e, &["--items", "D"]),
        &[("FANDHE_EDGE_BIN", ""), ("FAKE_TOOLCHAIN_MISSING", "1")],
    );
    assert_eq!(o.code, Some(70), "{}", o.diag());
    assert!(!e.dir.join("rustup.download").exists());
    assert_eq!(
        o.stdout,
        "{\"code\":\"runtime_error\",\"message\":\"cannot build the CLI\"}\n",
        "{}",
        o.diag()
    );

    // F でも取得は試みられず、record と stdout に rustup の語（環境変数名を含む）が出ない（スキーマ不変）
    let e = Env::new();
    let o = e.run(
        &with_work(&e, &["--items", "F", "--repeat", "1"]),
        &[("FAKE_TOOLCHAIN_MISSING", "1")],
    );
    assert!(!e.dir.join("rustup.download").exists());
    let rec = fs::read_to_string(e.work.join("record.json")).unwrap_or_default();
    assert!(!rec.contains("RUSTUP"), "{}", o.diag());
    assert!(!rec.contains("rustup"), "{}", o.diag());
    assert!(!o.stdout.contains("RUSTUP"), "{}", o.diag());
}

/// security.md: 記録・stdout に、作業ディレクトリのパス・データ本文・id の値が現れない。
#[test]
fn req39_record_does_not_leak_path_body_or_id() {
    let e = Env::new();
    let o = e.run(
        &with_work(&e, &["--items", "B,C,D,E,F", "--repeat", "2"]),
        &[("FAKE_LEAK", "1")],
    );
    assert_eq!(o.code, Some(0), "{}", o.diag());
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
        ("stderr", &o.stderr),
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
    assert_eq!(o.code, Some(10), "{}", o.diag());
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
    assert_eq!(o.code, Some(0), "{}", o.diag());
    for entry in ["evaluate B", "evaluate C1", "evaluate C2"] {
        assert_eq!(e.count_calls(entry), 1, "{entry}");
    }
    let e = Env::new();
    let o = e.run(
        &with_work(&e, &["--items", "B,C"]),
        &[("FAKE_FAIL_STAGE", "evaluate")],
    );
    assert_eq!(o.code, Some(10), "{}", o.diag());
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
    assert_eq!(o.code, Some(10), "{}", o.diag());
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
    assert_eq!(o.code, Some(10), "{}", o.diag());
    assert_eq!(e.q("items.F.status"), "\"failed\"");
    assert_eq!(e.q("items.F.runs"), "3");
    assert_eq!(e.q("items.F.passed"), "1");
    assert_eq!(e.q("items.F.failed"), "2");
    assert_eq!(e.q("items.F.no_tests"), "2");
    assert_eq!(e.q("items.F.read_output"), "0");
    // `--no-run`・`--list` 各 1 回 + 本番 3 回の計 5 回呼ばれるが、数えるのは本番だけ
    assert_eq!(e.lines("cargo.args").len(), 5);
    assert_eq!(e.lines("cargo.count"), ["3"]);
}

/// REQ-27: E は B の `train.jsonl` の入力だけを使い、凍結した `evaluation.jsonl` を `infer` へ渡さない。
#[test]
fn req27_item_e_uses_train_inputs_not_evaluation_data() {
    let e = Env::new();
    let o = e.run(&with_work(&e, &["--items", "B,E"]), &[]);
    assert_eq!(o.code, Some(0), "{}", o.diag());
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
    assert_eq!(o.code, Some(10), "{}", o.diag());
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

/// REQ-27・REQ-28・REQ-21: `--items E` 単独は B が無いので、起動前に引数エラー（exit 64）にする。
/// B が失敗したときも E は実行されない。
#[test]
fn req28_item_e_requires_successful_b() {
    let e = Env::new();
    assert_rejected_before_start(
        &e,
        &with_work(&e, &["--items", "E"]),
        &[],
        "item E requires item B",
    );
    assert_rejected_before_start(
        &e,
        &with_work(&e, &["--items", "C,D,E,F"]),
        &[],
        "item E requires item B",
    );

    let e = Env::new();
    let o = e.run(
        &with_work(&e, &["--items", "B,E"]),
        &[("FAKE_FAIL_STAGE", "train")],
    );
    assert_eq!(o.code, Some(10), "{}", o.diag());
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
    assert_eq!(o.code, Some(0), "{}", o.diag());
    assert_eq!(e.lines("make.log"), ["check-runtime-linkage"]);
    assert_eq!(e.q("items.A.status"), "\"not_run\"");
    assert_eq!(e.q("items.A.reason"), "\"not_selected\"");

    let e = Env::new();
    let o = e.run(&with_work(&e, &["--items", "A", "--with-ci"]), &[]);
    assert_eq!(o.code, Some(0), "{}", o.diag());
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
    assert_eq!(o.code, Some(0), "{}", o.diag());
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
    assert_eq!(o.code, Some(0), "{}", o.diag());
    assert_eq!(e.q("items.C.p95.classification"), "\"reference_only\"");
    assert_eq!(e.q("options.quiet_machine"), "true");
    assert_eq!(e.q("evidence_hint"), "\"test_harness\"");
    let e = Env::new();
    let o = e.run(&with_work(&e, &["--items", "C"]), &[]);
    assert_eq!(o.code, Some(0), "{}", o.diag());
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
    assert_eq!(o.code, Some(0), "{}", o.diag());
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
    assert_eq!(o.code, Some(0), "{}", o.diag());
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
    assert_eq!(o.code, Some(0), "{}", o.diag());
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
    assert_eq!(o.code, Some(0), "{}", o.diag());
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
    assert_eq!(o.code, Some(0), "{}", o.diag());
    let md = e.text("record.md");
    assert!(md.contains("- bin_override: False\n"), "{md}");
    assert!(!md.contains("注意: CLI を"), "{md}");
}

/// 起動済みのスクリプトと、stdout・stderr の読み取り口。
struct Running {
    child: Child,
    out: Receiver<String>,
    err: Receiver<String>,
}

/// スクリプトを起動し、偽コマンドが `started` ファイルを置く（子が走り出す）まで待つ。
fn start_until(e: &Env, args: &[String], envs: &[(&str, &str)], started: &str) -> Running {
    let mut child = e.command(args, envs, None).spawn().expect("spawn");
    let out = drain(child.stdout.take().expect("stdout"));
    let err = drain(child.stderr.take().expect("stderr"));
    let mut r = Running { child, out, err };
    let path = e.dir.join(started);
    let start = Instant::now();
    while !path.exists() {
        if r.child.try_wait().expect("try_wait").is_some() {
            panic!("script exited early: stderr={}", collect(&r.err));
        }
        if start.elapsed() > Duration::from_secs(30) {
            r.kill_group();
            panic!("{started} not created: stderr={}", collect(&r.err));
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    r
}

impl Running {
    fn kill_group(&mut self) {
        Command::new("kill")
            .args(["-KILL", &format!("-{}", self.child.id())])
            .stderr(Stdio::null())
            .status()
            .ok();
        self.child.kill().ok();
        self.child.wait().ok();
    }

    /// スクリプト本体（python3 へ exec 済み）へシグナルを送る。送り先が既に無くても失敗にしない。
    fn signal(&self, sig: &str) {
        Command::new("kill")
            .args([&format!("-{sig}"), &self.child.id().to_string()])
            .stderr(Stdio::null())
            .status()
            .ok();
    }

    /// 終了を待つ。期限は呼び出した時点（シグナルを送った後）から測るので、止め損ねて
    /// 偽コマンドの sleep が自然終了するのを待った場合は期限内に終わらず失敗する。
    fn finish(mut self) -> (std::process::ExitStatus, String, String) {
        let from = Instant::now();
        let status = loop {
            if let Some(st) = self.child.try_wait().expect("try_wait") {
                break st;
            }
            if from.elapsed() > Duration::from_secs(60) {
                self.kill_group();
                panic!(
                    "script did not stop: stdout={} stderr={}",
                    collect(&self.out),
                    collect(&self.err)
                );
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        (status, collect(&self.out), collect(&self.err))
    }
}

/// 偽コマンドが書いた pid が、短い猶予のうちに消える（子・孫が残らない）ことを確かめる。
fn assert_pids_gone(e: &Env, names: &[&str]) {
    for name in names {
        let pid = e.lines(name).first().cloned().expect("pid");
        let deadline = Instant::now() + Duration::from_secs(10);
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

const INTERRUPTED: &str =
    "{\"code\":\"runtime_error\",\"message\":\"interrupted\",\"record\":\"record.json\"}\n";

/// 項目 D の `make` の実行中に `sig` を 1 回送り、子のグループを止めて record を書き exit 70 で終える。
fn interrupt_during_make(sig: &str) {
    let e = Env::new();
    let r = start_until(
        &e,
        &with_work(&e, &["--items", "D,F", "--repeat", "2"]),
        &[("FAKE_MAKE_SLEEP", "1")],
        "make.started",
    );
    r.signal(sig);
    let (status, out, err) = r.finish();
    let code = status.code();
    assert_eq!(code, Some(70), "stdout={out} stderr={err}");
    assert_eq!(out, INTERRUPTED, "stderr={err}");
    assert_eq!(e.q("items.D.status"), "\"failed\"");
    assert_eq!(e.q("items.D.reason"), "\"interrupted\"");
    assert_eq!(e.q("items.F.status"), "\"not_run\"");
    assert_eq!(e.q("items.F.reason"), "\"interrupted\"");
    assert!(e.lines("cargo.args").is_empty());
    assert_pids_gone(&e, &["make.pid", "make.cpid"]);
}

/// REQ-21: SIGTERM を受けたら子のグループを止め、実行中の項目を `interrupted`、残りを `not_run` として
/// record を書き、exit 70 と固定メッセージを返す。子プロセスは残らない。
#[test]
fn req21_sigterm_stops_children_and_records_interrupted() {
    interrupt_during_make("TERM");
}

/// REQ-21: SIGINT でも SIGTERM と同じ（Ctrl-C。子を止め、record を書き、exit 70）。
#[test]
fn req21_sigint_stops_children_and_records_interrupted() {
    interrupt_during_make("INT");
}

/// REQ-21: SIGHUP でも SIGTERM と同じ（端末を閉じた場合）。
#[test]
fn req21_sighup_stops_children_and_records_interrupted() {
    interrupt_during_make("HUP");
}

/// REQ-21・REQ-39: 同じシグナルを続けて 2 回受けても、exit 70・固定 JSON 1 行（record あり、または
/// 強制終了の固定メッセージ）で終わり、偽 make とその子は残らない。2 回目が先に処理されて強制終了に
/// なるか、1 回目の後始末が先に終わって record が書かれるかは時間次第で、仕様はどちらも許す。
/// シグナルによる終了は許さない（終了処理の窓では中断シグナルを無視して exit 70 を保つ）。
/// 決定的な強制終了の検証は Python 側（`test_second_signal_forces_exit_while_cleanup_is_stuck`）。
#[test]
fn req21_double_sigterm_exits_70_without_leftover_children() {
    let e = Env::new();
    let r = start_until(
        &e,
        &with_work(&e, &["--items", "D,F"]),
        &[("FAKE_MAKE_SLEEP", "1")],
        "make.started",
    );
    r.signal("TERM");
    r.signal("TERM");
    let (status, out, err) = r.finish();
    let code = status.code();
    assert_eq!(code, Some(70), "stdout={out} stderr={err}");
    let forced = "{\"code\":\"runtime_error\",\"message\":\"interrupted (forced exit)\"}\n";
    assert!(out == INTERRUPTED || out == forced, "stdout={out:?}");
    assert_pids_gone(&e, &["make.pid", "make.cpid"]);
}

/// REQ-21: CLI のビルド中（項目の開始前）に SIGTERM を受けても、記録の骨格から `record.json`・`record.md` を書き、
/// exit 70 と `record` 付きの固定メッセージを返す。選んだ項目は `interrupted`、他は `not_selected`。
/// 環境の採取は未了のため `environment` は null。偽 cargo と子の sleep は残らない。
#[test]
fn req21_sigterm_during_cli_build_still_writes_record() {
    let e = Env::new();
    let r = start_until(
        &e,
        &with_work(&e, &["--items", "D,F"]),
        &[("FANDHE_EDGE_BIN", ""), ("FAKE_CARGO_BUILD_SLEEP", "1")],
        "cargo.started",
    );
    r.signal("TERM");
    let (status, out, err) = r.finish();
    let code = status.code();
    assert_eq!(code, Some(70), "stdout={out} stderr={err}");
    assert_eq!(out, INTERRUPTED, "stderr={err}");
    for item in ["D", "F"] {
        assert_eq!(e.q(&format!("items.{item}.status")), "\"not_run\"");
        assert_eq!(e.q(&format!("items.{item}.reason")), "\"interrupted\"");
    }
    assert_eq!(e.q("items.A.status"), "\"not_run\"");
    assert_eq!(e.q("items.A.reason"), "\"not_selected\"");
    assert_eq!(e.q("environment"), "null");
    assert_eq!(e.q("inputs.train_records"), "90");
    assert!(e.work.join("record.md").exists(), "record.md missing");
    assert_pids_gone(&e, &["cargo.pid", "cargo.cpid"]);
}

/// REQ-21: ラッパーの終了コードと stdout の `code` が対応する（0=ok・10=judged_fail・64=invalid_input・
/// 70=runtime_error。ビルド失敗）。
#[test]
fn req21_script_propagates_exit_code_and_stdout_json() {
    let one_line = |o: &Out, code: &str| {
        assert_eq!(o.stdout.lines().count(), 1, "{}", o.diag());
        assert!(
            o.stdout.starts_with(&format!("{{\"code\":\"{code}\"")),
            "{}",
            o.diag()
        );
    };
    let e = Env::new();
    let o = e.run(&with_work(&e, &["--items", "D"]), &[]);
    assert_eq!(o.code, Some(0), "{}", o.diag());
    one_line(&o, "ok");

    let e = Env::new();
    let o = e.run(
        &with_work(&e, &["--items", "B"]),
        &[("FAKE_FAIL_STAGE", "train")],
    );
    assert_eq!(o.code, Some(10), "{}", o.diag());
    one_line(&o, "judged_fail");

    let e = Env::new();
    let o = e.run(&with_work(&e, &["--items", "Z"]), &[]);
    assert_eq!(o.code, Some(64), "{}", o.diag());
    one_line(&o, "invalid_input");

    // Python 側の 70: CLI のビルドに失敗する
    let e = Env::new();
    let o = e.run(
        &with_work(&e, &["--items", "D"]),
        &[("FANDHE_EDGE_BIN", ""), ("FAKE_CARGO_BUILD_FAIL", "1")],
    );
    assert_eq!(o.code, Some(70), "{}", o.diag());
    assert_eq!(
        o.stdout,
        "{\"code\":\"runtime_error\",\"message\":\"cannot build the CLI\"}\n",
        "{}",
        o.diag()
    );
}

/// REQ-39: 記号・空白・改行を含む `--work-dir` と `FANDHE_EDGE_BIN` が、シェルに解釈されずそのまま
/// 渡る。注入のための副作用（`touch pwned`）が起きず、record は指定した場所にできる。
#[test]
fn req39_symbols_and_newlines_in_paths_pass_through_verbatim() {
    let e = Env::new();
    let weird = "we ird$(touch pwned)'\"\n;`touch pwned2`name";
    let work = e.dir.join(weird);
    let bin_dir = e.dir.join("b in$(touch pwned3)");
    fs::create_dir_all(&bin_dir).expect("mkdir");
    let bin = bin_dir.join("fake-cli");
    write_exe(&bin, FAKE_CLI);
    let o = e.run_cwd(
        &s(&["--work-dir", &work.display().to_string(), "--items", "B"]),
        &[("FANDHE_EDGE_BIN", &bin.display().to_string())],
        Some(&e.dir),
    );
    assert_eq!(o.code, Some(0), "{}", o.diag());
    assert!(work.join("record.json").is_file(), "record.json missing");
    for name in ["pwned", "pwned2", "pwned3"] {
        assert!(!e.dir.join(name).exists(), "{name} was created");
        assert!(!work.join(name).exists(), "{name} was created");
    }
}

/// REQ-39: 記号を含む不正な値は、値をエコーせずに 64 で拒否される。注入の副作用も起きない。
#[test]
fn req39_invalid_values_with_symbols_are_rejected_without_echo() {
    let e = Env::new();
    let cases: [&[&str]; 4] = [
        &["--items", "B;touch pwned"],
        &["--items", "B\nC"],
        &["--repeat", "1;touch pwned"],
        &["--p95-limit-us", "1 2"],
    ];
    for extra in cases {
        let o = e.run(&with_work(&e, extra), &[]);
        assert_eq!(o.code, Some(64), "{}", o.diag());
        assert!(o.stdout.starts_with(INVALID), "{}", o.diag());
        assert_eq!(o.stdout.lines().count(), 1, "{}", o.diag());
        for v in extra.iter().skip(1) {
            assert!(!o.stdout.contains(v), "value echoed: {}", o.diag());
            assert!(!o.stderr.contains(v), "value echoed: {}", o.diag());
        }
        assert!(!e.dir.join("pwned").exists());
        assert!(!e.work.exists(), "work dir created for {extra:?}");
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
    assert_eq!(o.code, Some(10), "{}", o.diag());
    assert_eq!(o.stdout, JUDGED_FAIL);
    assert_eq!(e.q("items.C.status"), "\"failed\"");
    assert_eq!(e.q("items.C.reason"), "\"unexpected_output\"");

    let e = Env::new();
    let o = e.run(
        &with_work(&e, &["--items", "C"]),
        &[("FAKE_C2_MODE", "small")],
    );
    assert_eq!(o.code, Some(10), "{}", o.diag());
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
    assert_eq!(o.code, Some(10), "{}", o.diag());
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
    assert_eq!(o.code, Some(0), "{}", o.diag());
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
    assert_eq!(o.code, Some(10), "{}", o.diag());
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
    assert_eq!(o.code, Some(10), "{}", o.diag());
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
    assert_eq!(o.code, Some(10), "{}", o.diag());
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
    assert_eq!(o.code, Some(10), "{}", o.diag());
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
    assert_eq!(o.code, Some(10), "{}", o.diag());
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
    assert_eq!(o.code, Some(10), "{}", o.diag());
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
    assert_eq!(o.code, Some(0), "{}", o.diag());
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
    assert_eq!(o.code, Some(10), "{}", o.diag());
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
    assert_eq!(o.code, Some(10), "{}", o.diag());
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
    assert_eq!(o.code, Some(10), "{}", o.diag());
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
    assert_eq!(o.code, Some(0), "{}", o.diag());
    let json = e.text("record.json");
    assert!(!json.contains("odd name"), "{json}");
    assert!(!e.text("record.md").contains("odd name"));
    assert_eq!(
        e.q("items.B.package_files.*.name"),
        "[\"artifact.json\",\"model.onnx\",\"<unrecognized>\"]"
    );
}
/// REQ-30・#362: `package` の実行後（exit 0・20 のどちらでも）に `package.staging/` が残っていれば
/// C は failed（`staging_left`）・exit 10。残っていない通常の実行では記録の欄が false になる。
#[test]
fn req30_c_staging_left_fails() {
    // C-1 の exit 0、C-1 の exit 20、C-2 の exit 20
    let cases: [&[(&str, &str)]; 3] = [
        &[("FAKE_STAGING_LEFT", "C1")],
        &[("FAKE_STAGING_LEFT", "C1"), ("FAKE_C1_EXCEED", "1")],
        &[("FAKE_STAGING_LEFT", "C2")],
    ];
    for envs in cases {
        assert_bad_output_fails("C", envs, "C", "staging_left", None);
    }
    let e = Env::new();
    let o = e.run(&with_work(&e, &["--items", "C"]), &[]);
    assert_eq!(o.code, Some(0), "{}", o.diag());
    assert_eq!(e.q("items.C.p95.package_staging_present"), "false");
    assert_eq!(
        e.q("items.C.capacity_limit.package_staging_present"),
        "false"
    );
}

/// REQ-30・#362: `package/` に通常ファイル以外（symlink・ディレクトリ）があれば B は
/// `package_entry_not_regular`、C-1 は `unexpected_output`。黙って飛ばして合格にしない。
#[test]
fn req30_package_entry_not_regular_fails() {
    for kind in ["symlink", "dir"] {
        assert_bad_output_fails(
            "B",
            &[("FAKE_PKG_KIND", kind)],
            "B",
            "package_entry_not_regular",
            None,
        );
        assert_bad_output_fails(
            "C",
            &[("FAKE_PKG_KIND", kind)],
            "C",
            "unexpected_output",
            Some("package"),
        );
    }
}

/// REQ-30・#362: `package/` の通常ファイル数が `capacity` の `file_count` の合計と違えば
/// B・C-1 は `unexpected_output`。
#[test]
fn req30_file_count_mismatch_fails() {
    for items in ["B", "C"] {
        assert_bad_output_fails(
            items,
            &[("FAKE_FC_SHIFT", "1")],
            items,
            "unexpected_output",
            Some("package"),
        );
    }
}

/// REQ-33・#362: B の単発 `infer`（`--id` なし）の `id` が既定値 `input` でなければ B は failed。
#[test]
fn req33_infer_default_id_must_be_input() {
    assert_bad_output_fails(
        "B",
        &[("FAKE_DEFAULT_ID", "x")],
        "B",
        "unexpected_output",
        Some("infer"),
    );
}

/// REQ-28・#362: バッチ出力が入力順でなければ E は failed（`unexpected_output`）。
#[test]
fn req28_item_e_batch_order_must_match_input() {
    assert_bad_output_fails(
        "B,E",
        &[("FAKE_E_ORDER", "swap")],
        "E",
        "unexpected_output",
        Some("infer-batch"),
    );
}

/// REQ-39・#362: F の `--list` が失敗・0 件なら F は failed（本番の cargo は呼ばない）。
#[test]
fn req39_item_f_list_failures_fail_closed() {
    assert_bad_output_fails("F", &[("FAKE_LIST_RC", "101")], "F", "list_failed", None);
    assert_bad_output_fails(
        "F",
        &[("FAKE_LIST_COUNT", "0")],
        "F",
        "no_tests_listed",
        None,
    );
    let e = Env::new();
    let o = e.run(
        &with_work(&e, &["--items", "F"]),
        &[("FAKE_LIST_COUNT", "0")],
    );
    assert_eq!(o.code, Some(10), "{}", o.diag());
    assert!(e.lines("cargo.count").is_empty());
}

/// REQ-39・#362: F の各回は `passed` が `--list` の件数（`expected_tests`）と一致し、`ignored` が 0
/// のときだけ合格。件数違い・ignored 付き・killed・出力上限超過・起動失敗はそれぞれの欄に数える。
#[test]
fn req39_item_f_counts_each_failure_kind() {
    let e = Env::new();
    let o = e.run(
        &with_work(&e, &["--items", "F", "--repeat", "3"]),
        &[("FAKE_CARGO_PATTERN", "ok,short,ignored")],
    );
    assert_eq!(o.code, Some(10), "{}", o.diag());
    assert_eq!(e.q("items.F.status"), "\"failed\"");
    assert_eq!(e.q("items.F.reason"), "\"test_failures\"");
    assert_eq!(e.q("items.F.expected_tests"), "12");
    assert_eq!(e.q("items.F.passed"), "1");
    assert_eq!(e.q("items.F.failed"), "2");
    assert_eq!(e.q("items.F.count_mismatch"), "2");
    assert_eq!(e.q("items.F.no_tests"), "0");

    // `--list` の件数に追随する（固定値の 12 とは比べない）
    let e = Env::new();
    let o = e.run(
        &with_work(&e, &["--items", "F", "--repeat", "2"]),
        &[("FAKE_LIST_COUNT", "5"), ("FAKE_CARGO_PATTERN", "short,ok")],
    );
    assert_eq!(o.code, Some(10), "{}", o.diag());
    assert_eq!(e.q("items.F.expected_tests"), "5");
    assert_eq!(e.q("items.F.passed"), "1");
    assert_eq!(e.q("items.F.count_mismatch"), "1");

    for (mode, key) in [
        ("killed", "killed"),
        ("overflow", "output_limit"),
        ("unexec", "spawn_error"),
    ] {
        let pattern = if mode == "unexec" {
            "unexec,ok".to_string()
        } else {
            format!("ok,{mode}")
        };
        let e = Env::new();
        let o = e.run(
            &with_work(&e, &["--items", "F", "--repeat", "2"]),
            &[("FAKE_CARGO_PATTERN", &pattern)],
        );
        assert_eq!(o.code, Some(10), "{mode}: stdout={}", o.stdout);
        assert_eq!(e.q("items.F.status"), "\"failed\"", "{mode}");
        assert_eq!(e.q("items.F.passed"), "1", "{mode}");
        assert_eq!(e.q("items.F.failed"), "1", "{mode}");
        assert_eq!(e.q(&format!("items.F.{key}")), "1", "{mode}");
    }
}

// ---- #364: 子プロセスの環境・全体の上限時間・ARGS・--work-dir の末尾スラッシュ ----

const OVERALL_MSG: &str = "--overall-timeout-sec must be an integer from 1 to 86400";
const OVERALL_EXCEEDED: &str = "{\"code\":\"judged_fail\",\"message\":\"overall time limit exceeded\",\"record\":\"record.json\"}\n";

/// `assert_rejected_before_start` の cwd 指定版（パス名展開の対象になるファイルがある cwd で確かめる）。
fn assert_rejected_in(e: &Env, args: &[String], cwd: &Path, message: &str) {
    let o = e.run_cwd(args, &[], Some(cwd));
    assert_eq!(o.code, Some(64), "args={args:?} {}", o.diag());
    assert_eq!(
        o.stdout,
        format!("{INVALID}{message}\"}}\n"),
        "args={args:?}"
    );
    for log in ["make.log", "cargo.args", "cli.log"] {
        assert!(e.lines(log).is_empty(), "{log} was written for {args:?}");
    }
    assert!(!e.work.exists(), "work dir created for {args:?}");
}

/// REQ-39・#364: `--items` のパス名展開の文字（`*`・`?`・`[B]`）は、cwd に `B`・`C` という名のファイルが
/// あっても、そのファイル名に化けて検証を通らず、起動前に 64 になる（作業ディレクトリも作られない）。
#[test]
fn req39_items_glob_characters_do_not_expand_to_file_names() {
    for bad in ["*", "?", "[B]", "B,*", "B-C"] {
        let e = Env::new();
        let cwd = e.dir.join("globdir");
        fs::create_dir_all(&cwd).expect("mkdir");
        fs::write(cwd.join("B"), "x").expect("write");
        fs::write(cwd.join("C"), "x").expect("write");
        assert_rejected_in(&e, &with_work(&e, &["--items", bad]), &cwd, ITEMS_MSG);
    }
}

/// REQ-39・#364: `--work-dir` が symlink なら、末尾の `/`・`//`・`/.`・`/./` を付けても拒否される。
/// 実ディレクトリの末尾スラッシュは通る。
#[test]
fn req39_work_dir_symlink_is_rejected_with_trailing_slash_forms() {
    for suffix in ["/", "//", "/.", "/./"] {
        let e = Env::new();
        let real = e.dir.join("real-empty");
        fs::create_dir_all(&real).expect("mkdir");
        let link = e.dir.join("link-work");
        symlink(&real, &link).expect("symlink");
        let arg = format!("{}{suffix}", link.display());
        let o = e.run(&s(&["--work-dir", &arg]), &[]);
        assert_eq!(o.code, Some(64), "{suffix}: {}", o.diag());
        assert_eq!(
            o.stdout,
            format!("{INVALID}work directory is not a directory\"}}\n"),
            "{suffix}"
        );
        assert!(e.lines("cli.log").is_empty(), "{suffix}");
    }
    let e = Env::new();
    let real = e.dir.join("real-empty");
    fs::create_dir_all(&real).expect("mkdir");
    let arg = format!("{}/", real.display());
    let o = e.run(&s(&["--work-dir", &arg, "--items", "B"]), &[]);
    assert_eq!(o.code, Some(0), "{}", o.diag());
    assert!(real.join("record.json").is_file());
}

/// REQ-39・#364: `--overall-timeout-sec` の範囲外・形の誤りは起動前に 64（固定メッセージ）。
#[test]
fn req39_overall_timeout_out_of_range_is_rejected_before_start() {
    for bad in ["0", "86401", "01", "abc", "123456", "-1", ""] {
        let e = Env::new();
        assert_rejected_before_start(
            &e,
            &with_work(&e, &["--overall-timeout-sec", bad]),
            &[],
            OVERALL_MSG,
        );
    }
    let e = Env::new();
    assert_rejected_before_start(
        &e,
        &with_work(&e, &["--overall-timeout-sec=0"]),
        &[],
        OVERALL_MSG,
    );
    let e = Env::new();
    assert_rejected_before_start(
        &e,
        &with_work(
            &e,
            &["--overall-timeout-sec", "5", "--overall-timeout-sec", "6"],
        ),
        &[],
        "duplicate option",
    );
    // 範囲の端（86400）は通り、記録へ入る
    let e = Env::new();
    let o = e.run(
        &with_work(&e, &["--items", "B", "--overall-timeout-sec=86400"]),
        &[],
    );
    assert_eq!(o.code, Some(0), "{}", o.diag());
    assert_eq!(e.q("options.overall_timeout_sec"), "86400");
}

/// REQ-39・#364: 既定の全体の上限時間（14400 秒）が記録される。
#[test]
fn req39_default_overall_timeout_is_recorded() {
    let e = Env::new();
    let o = e.run(&with_work(&e, &["--items", "B"]), &[]);
    assert_eq!(o.code, Some(0), "{}", o.diag());
    assert_eq!(e.q("options.overall_timeout_sec"), "14400");
}

/// REQ-39・#364: 全体の上限時間を超えたら子のグループを止め、実行中の項目は failed、残りは not_run
/// （reason は `overall_timeout`）で record を書き、exit 10 と固定メッセージを返す。子は残らない。
#[test]
fn req39_overall_timeout_stops_running_item_and_marks_rest_not_run() {
    let e = Env::new();
    let o = e.run(
        &with_work(&e, &["--items", "D,F", "--overall-timeout-sec", "5"]),
        &[("FAKE_MAKE_SLEEP", "1")],
    );
    assert_eq!(o.code, Some(10), "{}", o.diag());
    assert_eq!(o.stdout, OVERALL_EXCEEDED, "stderr={}", o.stderr);
    assert_eq!(e.q("options.overall_timeout_sec"), "5");
    assert_eq!(e.q("items.D.status"), "\"failed\"");
    assert_eq!(e.q("items.D.reason"), "\"overall_timeout\"");
    assert_eq!(e.q("items.F.status"), "\"not_run\"");
    assert_eq!(e.q("items.F.reason"), "\"overall_timeout\"");
    assert!(e.lines("cargo.args").is_empty());
    assert!(e.text("record.md").contains("上限時間"));
    assert_pids_gone(&e, &["make.pid", "make.cpid"]);
}

/// REQ-38・REQ-39・#364: 環境採取の `git`・`sysctl`・`sw_vers` は PATH を探さず固定パスで起動され、
/// 親の `GIT_DIR`・`GIT_WORK_TREE` も届かない。PATH の先頭に囮を置き、`GIT_*` を存在しない場所へ向けても、
/// `environment.commit` は環境を空にした固定パスの git で取った HEAD と一致し、囮は起動されない。
#[test]
fn req38_probe_commands_ignore_path_and_git_env() {
    let e = Env::new();
    let decoy = e.dir.join("decoy");
    fs::create_dir_all(&decoy).expect("mkdir");
    let body = "#!/bin/sh\necho \"$0\" >> \"$FAKE_DIR/decoy.log\"\necho 0000000000000000000000000000000000000000\n";
    for name in ["git", "sysctl", "sw_vers", "otool"] {
        write_exe(&decoy.join(name), body);
    }
    let path = format!(
        "{}:{}",
        decoy.display(),
        std::env::var("PATH").unwrap_or_default()
    );
    let nowhere = e.dir.join("no-such-git-dir").display().to_string();
    let o = e.run(
        &with_work(&e, &["--items", "B"]),
        &[
            ("PATH", &path),
            ("GIT_DIR", &nowhere),
            ("GIT_WORK_TREE", &nowhere),
        ],
    );
    assert_eq!(o.code, Some(0), "{}", o.diag());
    let head = Command::new("/usr/bin/git")
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .arg("-C")
        .arg(repo_root())
        .args(["rev-parse", "HEAD"])
        .output()
        .expect("git rev-parse");
    let head = String::from_utf8(head.stdout)
        .expect("utf8")
        .trim()
        .to_string();
    assert_eq!(head.len(), 40, "{head}");
    assert_eq!(e.q("environment.commit"), format!("\"{head}\""));
    assert!(e.lines("decoy.log").is_empty(), "decoy was started");
}

/// REQ-39・#364: `make real-machine-check` は環境変数の `ARGS` を拾わず（スクリプトを起動せず非 0 終了）、
/// コマンドラインの `ARGS` はパス名展開されない（`[5]` が `5` というファイルに化けない）。
/// 正しい経路（`ARGS=--help`）は通る。雛形は一時ディレクトリに作り、リポジトリへは何も書かない。
#[test]
fn req39_make_target_rejects_env_args_and_stops_glob_expansion() {
    let e = Env::new();
    let tmpl = e.dir.join("tmpl");
    fs::create_dir_all(tmpl.join("crates").join("x")).expect("mkdir");
    fs::write(tmpl.join("Cargo.toml"), "").expect("write");
    fs::write(tmpl.join("crates").join("x").join("Cargo.toml"), "").expect("write");
    fs::write(tmpl.join("5"), "x").expect("write");
    symlink(repo_root().join("scripts"), tmpl.join("scripts")).expect("symlink");
    let makefile = repo_root().join("Makefile");
    let make = |args: &[&str], env_args: Option<&str>| {
        let mut cmd = Command::new("make");
        cmd.arg("-s")
            .arg("-C")
            .arg(&tmpl)
            .arg("-f")
            .arg(&makefile)
            .arg("real-machine-check")
            .args(args)
            .env_remove("MAKEFLAGS")
            .env_remove("MFLAGS")
            .env_remove("MAKELEVEL")
            .env_remove("ARGS")
            .env("FAKE_DIR", &e.dir)
            .env("FAKE_CLI_PATH", &e.cli)
            .env("FANDHE_EDGE_BIN", &e.cli)
            .env("FANDHE_EDGE_MAKE_CMD", &e.make)
            .env("FANDHE_EDGE_CARGO_CMD", &e.cargo)
            .stdin(Stdio::null());
        if let Some(v) = env_args {
            cmd.env("ARGS", v);
        }
        cmd.output().expect("make")
    };
    let work = e.work.display().to_string();

    // 環境変数だけの ARGS: スクリプトを起動せず非 0 終了（作業ディレクトリも作られない）
    let o = make(&[], Some("--help"));
    assert!(!o.status.success());
    assert_eq!(String::from_utf8_lossy(&o.stdout), "");
    assert!(
        String::from_utf8_lossy(&o.stderr)
            .contains("ARGS must be given on the make command line, not through the environment")
    );
    let o = make(&[], Some(&format!("--work-dir {work}")));
    assert!(!o.status.success());
    assert!(!e.work.exists());

    // コマンドラインの ARGS: `[5]` はパス名展開されず --repeat の検証で 64 になる
    let args = format!("ARGS=--work-dir {work} --repeat [5]");
    let o = make(&[&args], None);
    assert!(!o.status.success());
    assert_eq!(
        String::from_utf8_lossy(&o.stdout),
        format!("{INVALID}{REPEAT_MSG}\"}}\n")
    );
    assert!(!e.work.exists());

    // 正しい経路は通る
    let o = make(&["ARGS=--help"], None);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(
        String::from_utf8_lossy(&o.stdout).starts_with("{\"code\":\"ok\",\"message\":\"usage:")
    );
}
