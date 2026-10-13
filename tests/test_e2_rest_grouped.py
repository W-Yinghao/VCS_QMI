"""r7 E2 grouped rest path (slurm/e2_rest_grouped.sh + slurm/e2_unit_group.sbatch) on CPU — fake per-unit wrappers and stubbed sbatch / squeue; no
real record, prediction or result is read and nothing is submitted.  The group wrapper (frozen line lookup by unit name, exact frozen wrapper
arguments, skip-complete, a failing unit does not stop the group, exit = missing markers, 'unit_exit=' renaming, refusals); the driver dry run
(the frozen rest stages and CPU steps in order, the frozen planner's lines files, selector groups of 3 on RTX6000PRO,H100 with 12 h, one quality
group per benchmark on L40S, the choose step between stage 1 and stage 2, CAP 20 wait, skip-complete, resubmission of a group with a missing
marker and the retry limit, switch-over adoption of queued per-unit / group / CPU-step jobs by name, gates and refusals).
Run: python -I tests/test_e2_rest_grouped.py -v"""
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))
import r7_d1_40c_plan as PL  # noqa: E402
import r7_e2_plan as EP  # noqa: E402

GROUP = R / "slurm" / "e2_unit_group.sbatch"
DRIVER = R / "slurm" / "e2_rest_grouped.sh"
LR = {"vcs": {"chosen_lr": 3e-4}, "logistic": {"chosen_lr": 1e-4}, "direct": {"chosen_lr": 3e-4}}

FAKE_WRAPPER = r'''#!/usr/bin/env bash
echo "$(basename "$0") $*" >> "$FAKE_LOG"; echo "exit=7"
case "$1" in
  train) m="$FAKE_ROOT/selectors/$2_lr$3_s$4.done.json";;
  run) i=${3%%/*}; n=${3##*/}; m="$FAKE_ROOT/eval/pred/$2/done_shard${i}of${n}.json";;
  *) exit 9;;
esac
if [ -n "$FAKE_FAIL" ] && [[ "$*" == *"$FAKE_FAIL"* ]]; then exit 3; fi
mkdir -p "$(dirname "$m")"; echo '{}' > "$m"; exit 0
'''


def fake_repo(t: Path):
    repo = t / "repo"; (repo / "slurm").mkdir(parents=True)
    for w in ("r7_e2_sel.sbatch", "r7_e2_gpu.sbatch", "other.sbatch"):
        f = repo / "slurm" / w; f.write_text(FAKE_WRAPPER); f.chmod(f.stat().st_mode | stat.S_IEXEC)
    root = t / "E2"; ld = root / "lines"
    EP.write_lines(ld / "stage1_lines.txt", EP.stage1(root), "test"); EP.write_lines(ld / "stage2_lines.txt", EP.stage2(LR, root), "test")
    EP.write_lines(ld / "eval_lines.txt", EP.eval_lines(root), "test")
    return repo, root, ld


def run_group(repo, root, lines, units, fail=""):
    log = root / "fake.log"
    env = dict(os.environ, E2_REPO=str(repo), FAKE_ROOT=str(root), FAKE_LOG=str(log), FAKE_FAIL=fail)
    p = subprocess.run(["bash", str(GROUP), str(lines), *units], env=env, capture_output=True, text=True, timeout=120)
    calls = log.read_text().splitlines() if log.exists() else []
    log.unlink(missing_ok=True)
    return p, calls


class GroupWrapper(unittest.TestCase):
    def test_runs_frozen_wrappers_in_order_and_skips_complete(self):
        with tempfile.TemporaryDirectory() as t:
            repo, root, ld = fake_repo(Path(t)); s1 = ld / "stage1_lines.txt"
            units = ["r7e2s_vcs_lr0.0001_s0", "r7e2s_vcs_lr0.0003_s0", "r7e2s_logistic_lr0.0001_s0"]
            p, calls = run_group(repo, root, s1, units)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertEqual(calls, ["r7_e2_sel.sbatch train vcs 0.0001 0", "r7_e2_sel.sbatch train vcs 0.0003 0", "r7_e2_sel.sbatch train logistic 0.0001 0"])
            for u in ("vcs_lr0.0001_s0", "vcs_lr0.0003_s0", "logistic_lr0.0001_s0"):
                self.assertTrue((root / "selectors" / f"{u}.done.json").exists())
            self.assertIn("unit_exit=7", p.stdout); self.assertEqual([x for x in p.stdout.splitlines() if x.startswith("exit=")], [p.stdout.splitlines()[-1]])
            self.assertTrue(p.stdout.splitlines()[-1].startswith("exit=0 "))
            p, calls = run_group(repo, root, s1, units)                                              # everything complete: nothing runs
            self.assertEqual(p.returncode, 0); self.assertEqual(calls, []); self.assertEqual(p.stdout.count("already complete"), 3)

    def test_failing_unit_does_not_stop_the_group(self):
        with tempfile.TemporaryDirectory() as t:
            repo, root, ld = fake_repo(Path(t)); s1 = ld / "stage1_lines.txt"
            units = ["r7e2s_logistic_lr0.0003_s0", "r7e2s_direct_lr0.0001_s0", "r7e2s_direct_lr0.0003_s0"]
            p, calls = run_group(repo, root, s1, units, fail="direct 0.0001")
            self.assertEqual(p.returncode, 1); self.assertEqual(len(calls), 3); self.assertIn("r7e2s_direct_lr0.0001_s0: NO marker", p.stdout)
            self.assertFalse((root / "selectors" / "direct_lr0.0001_s0.done.json").exists()); self.assertTrue((root / "selectors" / "direct_lr0.0003_s0.done.json").exists())
            p, calls = run_group(repo, root, s1, units)                                              # the retry runs only the missing unit
            self.assertEqual(p.returncode, 0); self.assertEqual(calls, ["r7_e2_sel.sbatch train direct 0.0001 0"])

    def test_quality_group_and_refusals(self):
        with tempfile.TemporaryDirectory() as t:
            repo, root, ld = fake_repo(Path(t)); ev = ld / "eval_lines.txt"
            p, calls = run_group(repo, root, ev, [f"r7e2q_gqa_dev_{i}" for i in range(6)])
            self.assertEqual(p.returncode, 0, p.stdout); self.assertEqual(calls, [f"r7_e2_gpu.sbatch run gqa_dev {i}/6" for i in range(6)])
            self.assertEqual(len(list((root / "eval" / "pred" / "gqa_dev").glob("done_shard*of6.json"))), 6)
            bad = ld / "bad_lines.txt"
            bad.write_text("sbatch --parsable --job-name=r7e2q_x_0 --partition=L40S --exclude=node51,node60 slurm/other.sbatch run x 0/1 #MARK /tmp/none.json\n")
            p, calls = run_group(repo, root, bad, ["r7e2q_x_0", "r7e2s_missing_lr1_s0", "rm -rf"])
            self.assertEqual(p.returncode, 3); self.assertEqual(calls, []); self.assertIn("not a frozen E2", p.stdout)
            self.assertIn("no line", p.stdout); self.assertIn("not an E2 selector / quality unit name", p.stdout)
            p, calls = run_group(repo, root, ld / "nope.txt", ["r7e2q_gqa_dev_0"]); self.assertEqual(p.returncode, 1); self.assertIn("missing", p.stdout)


# ------------------------------------------------------------------------------------------------------------------------------ driver
STUB_SBATCH = r'''#!/usr/bin/env bash
echo "$*" >> "$STUB_DIR/sbatch.log"
n=$(cat "$STUB_DIR/counter" 2>/dev/null || echo 4000); n=$((n + 1)); echo $n > "$STUB_DIR/counter"
args="$*"; E=$E2_ROOT
if [ -n "$STUB_FAIL_ON" ] && [[ "$args" == *"$STUB_FAIL_ON"* ]]; then exit 1; fi
mkunit() { mkdir -p "$(dirname "$1")"; echo '{"stub": true}' > "$1"
  if [[ "$1" == */selectors/*.done.json ]]; then t=$(basename "$1" .done.json); touch "${1%.done.json}.pt"
    v=$(python3 -c "import zlib; print(zlib.crc32(b'$t') % 1000 / 1000)"); echo "{\"selected_cal_spearman\": $v}" > "${1%.done.json}.json"; fi; }
case "$args" in
  *"e2_unit_group.sbatch "*)
    set -- $args; while [ "$1" != "slurm/e2_unit_group.sbatch" ]; do shift; done; shift; L=$1; shift
    for u in "$@"; do
      line=$(grep -F -- "--job-name=$u " "$L" | head -1); m=${line##* #MARK }
      if [ -f "$m" ]; then echo "skip $u" >> "$STUB_DIR/units.log"; continue; fi
      k=$(cat "$STUB_DIR/fail_$u" 2>/dev/null || echo 0)
      if [[ " ${STUB_UNIT_FAILS:-} " == *" $u "* ]] && [ "$k" -lt "${STUB_UNIT_FAIL_TIMES:-1}" ]; then echo $((k + 1)) > "$STUB_DIR/fail_$u"; echo "fail $u" >> "$STUB_DIR/units.log"; continue; fi
      mkunit "$m"; echo "ran $u" >> "$STUB_DIR/units.log"
    done;;
  *"r7_e2_gpu.sbatch cache "*) mkdir -p "$E/cache"; echo "{\"max_images\": ${args##* }}" > "$E/cache/cache_d1_40.json";;
  *"r7_e2_cpu.sbatch cachecheck "*) mkdir -p "$E/cache"; echo "{\"pass\": ${STUB_CACHE_PASS:-true}}" > "$E/cache/e2_cache_check.json";;
  *"r7_e2_cpu.sbatch choose"*) "$PY" scripts/r7_e2_plan.py choose --selectors-dir "$E/selectors" >> "$STUB_DIR/wrap.log" 2>&1 || exit 1;;
  *"r7_e2_cpu.sbatch check"*) "$PY" scripts/r7_d1_40c_plan.py configs --lr-choice "$E/selectors/lr_choice.json" --out "$E/eval/configs" --rows core >> "$STUB_DIR/wrap.log" 2>&1 || exit 1
    mkdir -p "$E/eval/check"; echo "{\"pass\": ${STUB_CHECK_PASS:-true}}" > "$E/eval/check/check.json";;
  *"r7_e2_cpu.sbatch runcomplete"*) mkdir -p "$E/eval/run"; echo '{"complete": true}' > "$E/eval/run/completeness.json";;
  *"r7_e2_cpu.sbatch reuserows "*) mkdir -p "$E/eval/reuse"; echo "{\"pass\": ${STUB_REUSE_PASS:-true}}" > "$E/eval/reuse/reuse_check.json";;
  *"r7_e2_cpu.sbatch corecomplete"*) mkdir -p "$E/eval"; echo '{"complete": true}' > "$E/eval/completeness.json";;
  *) echo "unexpected: $args" >> "$STUB_DIR/errors";;
esac
echo $n
'''
STUB_SQUEUE = r'''#!/usr/bin/env bash
args="$*"; Q=$STUB_DIR/queue
if [[ "$args" == *"-j "* ]]; then                                                    # one job's state: pre-existing (adoptable) jobs live in $Q
  id=$(sed -E 's/.*-j ([0-9]+).*/\1/' <<< "$args"); line=$(grep -E "^$id " "$Q" 2>/dev/null)
  [ -z "$line" ] && exit 0                                                             # stub-submitted jobs end at once
  read -r qid qname qpolls qmark qcontent <<< "$line"
  if [ "$qpolls" -gt 0 ]; then sed -i "s|^$id $qname $qpolls |$id $qname $((qpolls - 1)) |" "$Q"; echo RUNNING; exit 0; fi
  if [ "$qmark" != "-" ]; then mkdir -p "$(dirname "$qmark")"; echo "${qcontent:-{\"stub\": true\}}" > "$qmark"
    if [[ "$qmark" == */selectors/*.done.json ]]; then t=$(basename "$qmark" .done.json); touch "${qmark%.done.json}.pt"; echo '{"selected_cal_spearman": 0.5}' > "${qmark%.done.json}.json"; fi; fi
  sed -i "/^$id /d" "$Q"; echo "$qid" >> "$STUB_DIR/adopted_done"; exit 0
fi
if [[ "$args" == *"%i %j"* ]]; then [ -f "$Q" ] && awk '{print $1, $2}' "$Q"; exit 0; fi   # my jobs' names
c=$(cat "$STUB_DIR/qcalls" 2>/dev/null || echo 0); c=$((c + 1)); echo $c > "$STUB_DIR/qcalls"
n=${STUB_QUEUE_N:-0}; [ "$c" -le "${STUB_FULL_CALLS:-0}" ] && n=20
for i in $(seq 1 $n); do echo "job$i"; done
'''


def run_driver(tmp: Path, keys=True, data=True, teacher=True, queue=(), pre=None, **env_extra):
    stub = tmp / "stub"; stub.mkdir(parents=True)
    for name, body in (("sbatch", STUB_SBATCH), ("squeue", STUB_SQUEUE)):
        f = stub / name; f.write_text(body); f.chmod(f.stat().st_mode | stat.S_IEXEC)
    e2 = tmp / "E2"; e2.mkdir()
    if data:
        (e2 / "data").mkdir(); (e2 / "data" / "data_ready.json").write_text("{}"); (e2 / "data" / "d1_manifest_meta.json").write_text('{"counts": {"FIT": {"images": 1234}}}')
    if teacher is not None:
        (e2 / "teacher").mkdir(); (e2 / "teacher" / "r7_40a_completeness.json").write_text(json.dumps({"complete": bool(teacher)}))
    if pre:
        pre(e2)
    (stub / "queue").write_text("".join(" ".join(str(x) for x in q) + "\n" for q in queue))
    proto = tmp / "proto.yaml"; proto.write_text("r7_40c_staged_freeze: >\n  x\nr7_e2a_staged_freeze: >\n  y\n" + ("r7_e2_staged_freeze: >\n  z\n" if keys else "note: r7_e2_staged_freeze later\n"))
    env = dict(os.environ, PATH=f"{stub}:{os.environ['PATH']}", STUB_DIR=str(stub), E2_ROOT=str(e2), PROTOCOL=str(proto), SUBMIT_LOG=str(tmp / "grouped.log"),
               LINES_DIR=str(e2 / "lines"), POLL="0", WAIT_H="1", SOON_S="0", SOON_POLL="1", PY=sys.executable, RETRIES="2",
               FROZEN_DRIVER_PATTERN=f"no-such-process-{os.getpid()}-{tmp.name}")
    env.update({k: str(v) for k, v in env_extra.items()})
    p = subprocess.run(["bash", str(DRIVER)], env=env, capture_output=True, text=True, timeout=600)
    rd = lambda f: (stub / f).read_text().splitlines() if (stub / f).exists() else []  # noqa: E731
    return p, (tmp / "grouped.log").read_text() if (tmp / "grouped.log").exists() else "", rd("sbatch.log"), rd("units.log"), rd("errors"), e2


def kind(c):
    for k, pat in (("cache", "gpu.sbatch cache"), ("cachecheck", "cpu.sbatch cachecheck"), ("choose", "cpu.sbatch choose"), ("check", "cpu.sbatch check"),
                   ("runcomplete", "cpu.sbatch runcomplete"), ("reuserows", "cpu.sbatch reuserows"), ("corecomplete", "cpu.sbatch corecomplete")):
        if pat in c:
            return k
    if "e2_unit_group.sbatch" in c:
        return c.split("--job-name=")[1].split()[0]
    return "?"


FULL = ["cache", "cachecheck", "r7e2g_s1_1", "r7e2g_s1_2", "choose", "r7e2g_s2_1", "r7e2g_s2_2", "check",
        "r7e2g_q_gqa_dev", "r7e2g_q_textvqa_val", "r7e2g_q_chartqa_val", "runcomplete", "reuserows", "corecomplete"]


def sel_marks(e2, tags):
    for t in tags:
        (e2 / "selectors").mkdir(parents=True, exist_ok=True)
        for ext, body in ((".done.json", "{}"), (".json", '{"selected_cal_spearman": 0.3}'), (".pt", "")):
            (e2 / "selectors" / f"{t}{ext}").write_text(body)


class DriverDryRun(unittest.TestCase):
    def test_full_sequence_groups_and_cap_wait(self):
        with tempfile.TemporaryDirectory() as t:
            p, log, calls, units, errs, e2 = run_driver(Path(t), STUB_FULL_CALLS=3)
            self.assertEqual(p.returncode, 0, log + p.stderr); self.assertEqual(errs, []); self.assertIn("done: E2 evaluation complete", log)
            self.assertEqual([kind(c) for c in calls], FULL)
            self.assertFalse(any("--export" in c or "--dependency" in c for c in calls)); self.assertFalse(any("score" in c for c in calls))
            self.assertTrue(all("--exclude=node51,node60" in c for c in calls if "--partition=CPU" not in c))
            self.assertTrue(all("--partition=CPU" in c and "--wait" in c for c in calls if kind(c) in ("cachecheck", "choose", "check", "runcomplete", "reuserows", "corecomplete")))
            sel = [c for c in calls if "r7e2g_s" in c]; q = [c for c in calls if "r7e2g_q_" in c]
            self.assertTrue(all("--partition=RTX6000PRO,H100 --exclude=node51,node60 --time=12:00:00 slurm/e2_unit_group.sbatch " in c for c in sel))
            self.assertTrue(all(len(c.split("slurm/e2_unit_group.sbatch ")[1].split()) == 4 for c in sel))   # lines file + 3 units
            self.assertIn(str(e2 / "lines" / "stage1_lines.txt") + " r7e2s_vcs_lr0.0001_s0 r7e2s_vcs_lr0.0003_s0 r7e2s_logistic_lr0.0001_s0", sel[0])
            self.assertIn(str(e2 / "lines" / "stage2_lines.txt"), sel[2])
            tq = {c.split("--job-name=")[1].split()[0]: c for c in q}
            self.assertIn("--partition=L40S --exclude=node51,node60 --time=10:00:00", tq["r7e2g_q_gqa_dev"]); self.assertIn("--time=16:00:00", tq["r7e2g_q_textvqa_val"])
            self.assertIn("--time=04:30:00", tq["r7e2g_q_chartqa_val"]); self.assertTrue(tq["r7e2g_q_gqa_dev"].endswith(" ".join(f"r7e2q_gqa_dev_{i}" for i in range(6))))
            self.assertTrue(tq["r7e2g_q_chartqa_val"].endswith("r7e2q_chartqa_val_0 r7e2q_chartqa_val_1"))
            self.assertEqual(len([u for u in units if u.startswith("ran r7e2s_")]), 12); self.assertEqual(len([u for u in units if u.startswith("ran r7e2q_")]), 12)
            ch = json.loads((e2 / "selectors" / "lr_choice.json").read_text()); s2 = (e2 / "lines" / "stage2_lines.txt").read_text()
            self.assertTrue(all(f"train {o} {ch[o]['chosen_lr']:g} 1" in s2 for o in ch))                      # stage 2 at the chosen rates
            gl = (e2 / "lines" / "groups_eval.txt").read_text().splitlines(); self.assertEqual(len([x for x in gl if not x.startswith("#")]), 3)
            self.assertEqual(len(gl[1].split(" #MARKS ")[1].split()), 6)
            run = PL.read_configs(e2 / "eval" / "run" / "configs_gqa_dev.txt"); self.assertEqual(len(run), 46)
            self.assertTrue(next(c for c in calls if kind(c) == "cache").endswith("cache 1234"))

    def test_skip_complete_and_resubmit_missing_marker(self):
        with tempfile.TemporaryDirectory() as t:                                                 # stage-1 group 1 complete; one unit of group 2 fails once
            done = ["vcs_lr0.0001_s0", "vcs_lr0.0003_s0", "logistic_lr0.0001_s0"]
            p, log, calls, units, errs, e2 = run_driver(Path(t), pre=lambda e: sel_marks(e, done), STUB_UNIT_FAILS="r7e2s_direct_lr0.0003_s0")
            self.assertEqual(p.returncode, 0, log); ks = [kind(c) for c in calls]
            self.assertNotIn("r7e2g_s1_1", ks); self.assertEqual(ks.count("r7e2g_s1_2"), 2); self.assertIn("RETRY 1/2", log)
            self.assertLess(max(i for i, k in enumerate(ks) if k == "r7e2g_s1_2"), ks.index("choose")); self.assertLess(ks.index("choose"), ks.index("r7e2g_s2_1"))
            i = units.index("fail r7e2s_direct_lr0.0003_s0")
            self.assertEqual(units[i + 1:i + 4], ["skip r7e2s_logistic_lr0.0003_s0", "skip r7e2s_direct_lr0.0001_s0", "ran r7e2s_direct_lr0.0003_s0"])
        with tempfile.TemporaryDirectory() as t:                                                 # a unit that never gets its marker: 3 group jobs, then STOP
            p, log, calls, units, errs, e2 = run_driver(Path(t), STUB_UNIT_FAILS="r7e2q_textvqa_val_2", STUB_UNIT_FAIL_TIMES=9)
            self.assertEqual(p.returncode, 1); self.assertEqual(sum(kind(c) == "r7e2g_q_textvqa_val" for c in calls), 3)
            self.assertIn("group r7e2g_q_textvqa_val failed 3 times (missing: done_shard2of4.json", log); self.assertNotIn("done:", log)
        with tempfile.TemporaryDirectory() as t:                                                 # everything already done: nothing is submitted
            def all_done(e):
                sel_marks(e, [f"{o}_lr{lr:g}_s0" for o in ("vcs", "logistic", "direct") for lr in (1e-4, 3e-4)]); (e / "cache").mkdir()
                (e / "cache" / "cache_d1_40.json").write_text("{}"); (e / "cache" / "e2_cache_check.json").write_text('{"pass": true}')
            p, log, calls, units, errs, e2 = run_driver(Path(t), pre=all_done, STUB_FAIL_ON="cpu.sbatch check")
            ks = [kind(c) for c in calls]; self.assertEqual(ks[:3], ["choose", "r7e2g_s2_1", "r7e2g_s2_2"])            # stage 1 and the cache skipped
            self.assertEqual(p.returncode, 1); self.assertIn("STOP: no job id", log)

    def test_switch_over_adopts_queued_jobs(self):
        with tempfile.TemporaryDirectory() as t:
            tt = Path(t); e2 = tt / "E2"
            queue = [(901, "r7e2c_cache", 2, e2 / "cache" / "cache_d1_40.json"), (902, "r7e2_cachecheck", 1, e2 / "cache" / "e2_cache_check.json", '{"pass":true}'),
                     (903, "r7e2s_vcs_lr0.0003_s0", 3, e2 / "selectors" / "vcs_lr0.0003_s0.done.json")]
            p, log, calls, units, errs, e2 = run_driver(tt, queue=queue)
            self.assertEqual(p.returncode, 0, log + p.stderr); ks = [kind(c) for c in calls]
            self.assertNotIn("cache", ks); self.assertNotIn("cachecheck", ks)                      # the running cache and its check are adopted, not duplicated
            self.assertIn("waiting for queued job(s) 901 (r7e2c_cache)", log); self.assertIn("waiting for queued r7e2_cachecheck job(s) 902", log)
            self.assertIn("waiting for queued job(s) 903 (group r7e2g_s1_1 or its units)", log)
            self.assertEqual(ks[:2], ["r7e2g_s1_2", "r7e2g_s1_1"]); self.assertIn("skip r7e2s_vcs_lr0.0003_s0", units)   # s1_1 only after unit job 903 left
            self.assertEqual(sorted((tt / "stub" / "adopted_done").read_text().split()), ["901", "902", "903"])

    def test_gates(self):
        for kw, msg in (({"keys": False}, "is not frozen"), ({"ROWS": "all"}, "bad ROWS"), ({"PCA_BASIS": "other"}, "bad PCA_BASIS"),
                        ({"CACHE_PARTITION": "L40S;x"}, "bad CACHE_PARTITION"), ({"data": False}, "data_ready.json not ready after")):
            with tempfile.TemporaryDirectory() as t:
                p, log, calls, units, errs, e2 = run_driver(Path(t), **kw)
                self.assertEqual(p.returncode, 1, (kw, log)); self.assertIn(msg, log); self.assertEqual(calls, []); self.assertNotIn("done:", log)
        with tempfile.TemporaryDirectory() as t:                                                 # teacher incomplete: cache + its check, then the gate times out
            p, log, calls, units, errs, e2 = run_driver(Path(t), teacher=False)
            self.assertEqual(p.returncode, 1); self.assertEqual([kind(c) for c in calls], ["cache", "cachecheck"]); self.assertIn("r7_40a_completeness.json not ready after", log)
        with tempfile.TemporaryDirectory() as t:                                                 # failing CPU checks stop as in the frozen path
            for kw, msg in (({"STUB_CACHE_PASS": "false"}, "e2_cache_check.json does not say"), ({"STUB_REUSE_PASS": "false"}, "failed the reuse check")):
                with tempfile.TemporaryDirectory() as t2:
                    p, log, calls, units, errs, e2 = run_driver(Path(t2), **kw); self.assertEqual(p.returncode, 1, kw); self.assertIn(msg, log)
            p, log, calls, units, errs, e2 = run_driver(Path(t), ROWS="core", STUB_REUSE_PASS="false")
            self.assertEqual(p.returncode, 0, log); self.assertIn("descriptive only", log); self.assertEqual(len(PL.read_configs(e2 / "eval" / "run" / "configs_chartqa_val.txt")), 50)
        with tempfile.TemporaryDirectory() as t:                                                 # a frozen rest driver still running: refuse
            marker = f"fake-frozen-driver-{os.getpid()}"
            s = subprocess.Popen(["bash", "-c", f"exec -a {marker} sleep 60"])
            try:
                p, log, calls, units, errs, e2 = run_driver(Path(t), FROZEN_DRIVER_PATTERN=marker)
                self.assertEqual(p.returncode, 1); self.assertIn("a frozen rest driver is still running", log); self.assertEqual(calls, [])
            finally:
                s.kill(); s.wait()


if __name__ == "__main__":
    unittest.main()
