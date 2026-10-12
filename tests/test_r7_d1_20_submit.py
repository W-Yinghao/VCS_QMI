"""Dry run of slurm/r7_d1_20_submit.sh with stub sbatch / squeue (no SLURM contact): stage order, file gates, the CPU choice job (executed
locally by the stub), no afterok, CAP-wait path, and the stop on an empty job id."""
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

R = Path(__file__).resolve().parents[1]

STUB_SBATCH = r'''#!/usr/bin/env bash
echo "$*" >> "$STUB_DIR/sbatch.log"
n=$(cat "$STUB_DIR/counter" 2>/dev/null || echo 1000); n=$((n + 1)); echo $n > "$STUB_DIR/counter"
args="$*"
if [ -n "$STUB_FAIL_ON" ] && [[ "$args" == *"$STUB_FAIL_ON"* ]]; then exit 1; fi
if [[ "$args" == *"cache d1_20"* ]]; then mkdir -p "$D1_ROOT/cache"
  for f in pca.pt feats_FIT.npy feats_CAL.npy images_FIT.json images_CAL.json question_ids.jsonl embed_table.pt; do touch "$D1_ROOT/cache/$f"; done; fi
if [[ "$args" =~ train\ ([a-z]+)\ ([0-9.e-]+)\ ([0-9]+) ]]; then
  t="${BASH_REMATCH[1]}_lr${BASH_REMATCH[2]}_s${BASH_REMATCH[3]}"; [[ "$args" == *"--no-question"* ]] && t="${t}_noq"; [[ "$args" == *"--target d"* ]] && t="${t}_dtarget"
  mkdir -p "$D1_ROOT/selectors"; touch "$D1_ROOT/selectors/$t.pt"
  v=$(python3 -c "import zlib; print(zlib.crc32(b'$t') % 1000 / 1000)"); echo "{\"selected_cal_spearman\": $v}" > "$D1_ROOT/selectors/$t.json"; fi
if [[ "$args" == *"--wrap"* ]]; then bash -c "${args#*--wrap }" >> "$STUB_DIR/wrap.log" 2>&1 || exit 1; fi
echo $n
'''
STUB_SQUEUE = r'''#!/usr/bin/env bash
if [[ "$*" == *"-j"* ]]; then exit 0; fi
c=$(cat "$STUB_DIR/qcalls" 2>/dev/null || echo 0); c=$((c + 1)); echo $c > "$STUB_DIR/qcalls"
n=${STUB_QUEUE_N:-0}; [ "$c" -le "${STUB_FULL_CALLS:-0}" ] && n=27
for i in $(seq 1 $n); do echo "job$i"; done
'''


def run_driver(tmp: Path, fail_on="", full_calls=0):
    stub = tmp / "stub"; stub.mkdir()
    for name, body in (("sbatch", STUB_SBATCH), ("squeue", STUB_SQUEUE)):
        f = stub / name; f.write_text(body); f.chmod(f.stat().st_mode | stat.S_IEXEC)
    d1 = tmp / "D1"; (d1 / "data").mkdir(parents=True); (d1 / "data" / "d1_manifest_meta.json").write_text('{"frozen": true}')
    env = dict(os.environ, PATH=f"{stub}:{os.environ['PATH']}", STUB_DIR=str(stub), D1_ROOT=str(d1), SUBMIT_LOG=str(tmp / "submit.log"), POLL="0",
               WAIT_H="1", PY=sys.executable, S1_LINES=str(tmp / "s1.txt"), S2_LINES=str(tmp / "s2.txt"), STUB_FAIL_ON=fail_on, CAP="27",
               STUB_FULL_CALLS=str(full_calls))
    p = subprocess.run(["bash", str(R / "slurm" / "r7_d1_20_submit.sh")], env=env, capture_output=True, text=True, timeout=300)
    calls = (stub / "sbatch.log").read_text().splitlines() if (stub / "sbatch.log").exists() else []
    return p, (tmp / "submit.log").read_text(), calls, d1


class DriverDryRun(unittest.TestCase):
    def test_full_sequence(self):
        with tempfile.TemporaryDirectory() as t:
            p, log, calls, d1 = run_driver(Path(t), full_calls=3)                          # the first CAP checks see a full queue
            self.assertEqual(p.returncode, 0, log + p.stderr)
            self.assertEqual(len(calls), 1 + 6 + 1 + 12, calls)
            self.assertIn("cache d1_20", calls[0]); self.assertTrue(all("train" in c for c in calls[1:7])); self.assertIn("--wrap", calls[7])
            self.assertTrue(all("train" in c for c in calls[8:]))
            self.assertFalse(any("--dependency" in c for c in calls))                         # file gates, no afterok chains
            self.assertTrue(all("--exclude=node51,node52,node60" in c for c in calls if "--partition=CPU" not in c))
            self.assertTrue(all("RTX6000PRO,H100,L40S" in c for c in calls if " train " in c))
            self.assertIn("--partition=H100,L40S", calls[0])                                  # LLaVA env: no Blackwell
            ch = json.loads((d1 / "selectors" / "lr_choice.json").read_text()); self.assertEqual(set(ch), {"vcs", "logistic", "direct"})
            self.assertEqual(sum("--no-question" in c for c in calls), 3); self.assertEqual(sum("--target d" in c for c in calls), 3)
            self.assertIn("done: stage 2 submitted (12 units)", log)

    def test_stops_on_empty_job_id(self):
        with tempfile.TemporaryDirectory() as t:
            p, log, calls, _ = run_driver(Path(t), fail_on="train logistic 0.0001")
            self.assertEqual(p.returncode, 1); self.assertIn("STOP: no job id from", log); self.assertNotIn("stage 1 complete", log)


if __name__ == "__main__":
    unittest.main()
