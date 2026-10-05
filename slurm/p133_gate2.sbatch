#!/usr/bin/env bash
#SBATCH --job-name=p133_gate2
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/p133_gate2_%j.out
# P133 gate, round 2 (main-session decision 2026-10-05): the constant-online-map gate of job 1021175, same protocol (real CIFAR-10, 20 P133 steps
# entered at peak lr, 256 random directions + mean-key / current-minus-queue directions + gradient ascent on J, P100 positive control), on
#   (b) q1024m999: queue 1024 view-0 keys, m 0.999        (c) noqueue: current-batch momentum keys only, m 0.99
# pass rule: max J over constant maps <= 2 x half-batch SE.  Also: P133 test suite (13), config-hash invariance, 6-step smokes + stop/resume.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
set -o pipefail
J=${SLURM_JOB_ID:-local}; GATE=/home/infres/yinwang/CS_QMI/outputs/P133_GATE2_$J; REP=$REPO_ROOT/reports/P133_GATE2_$J
REAL_OUT=$OUTPUT_ROOT; mkdir -p "$GATE" "$REP"; export OUTPUT_ROOT=$GATE
banner "P133 gate 2 -> $REP"
rc_all=0
fail() { echo "FAIL: $1"; rc_all=1; }
echo "### tests"; OUTPUT_ROOT=$REAL_OUT "$PYTHON" -m pytest -q tests/test_vcs_ssl_p133.py -p no:cacheprovider > "$REP/pytest_p133.txt" 2>&1 || fail "p133 tests"; tail -1 "$REP/pytest_p133.txt"
echo "### config-hash invariance (HEAD vs working tree)"
git -C "$REPO_ROOT" archive HEAD src reference | tar -x -C "$GATE" && mkdir -p "$GATE/headcode" && mv "$GATE/src" "$GATE/reference" "$GATE/headcode/"
cat > "$GATE/hashes.py" <<'PY'
import glob, json, os, sys
from vcs_ssl.config import load_config
env = {"DATA_ROOT": os.environ["DATA_ROOT"], "OUTPUT_ROOT": "/home/infres/yinwang/CS_QMI/outputs", "MANIFEST_ROOT": os.environ["MANIFEST_ROOT"]}
out = {}
for f in sorted(glob.glob("configs/*.yaml")):
    if "hpMC_" in f or "hpR50_" in f:
        continue
    try:
        out[f] = load_config(f, env=env)["_meta"]["config_hash"]
    except Exception as e:  # noqa: BLE001
        out[f] = f"ERROR {type(e).__name__}"
json.dump(out, open(sys.argv[1], "w"), indent=1)
PY
PYTHONPATH="$GATE/headcode/src:$GATE/headcode" "$PYTHON" "$GATE/hashes.py" "$GATE/hash_head.json" && "$PYTHON" "$GATE/hashes.py" "$GATE/hash_now.json"
"$PYTHON" -c "
import json; a=json.load(open('$GATE/hash_head.json')); b=json.load(open('$GATE/hash_now.json')); d=[f for f in a if a[f]!=b.get(f)]
json.dump({'configs':len(a),'changed':d},open('$REP/config_hash_invariance.json','w'),indent=1); print(len(a),'configs; changed',len(d),d[:5]); raise SystemExit(1 if d else 0)" || fail "hash invariance"
for v in q1024m999 noqueue; do
  echo "### constant-map gate: $v"
  "$PYTHON" scripts/p133_gate_constant_map.py --config configs/cifar10_hpMC_vcs_${v}_views4_800ep_seed0.yaml --steps 20 --run-dir "$GATE/constmap_$v" \
      --out "$REP/constant_map_$v.json" 2>&1 | tail -28; echo "gate_$v rc=${PIPESTATUS[0]}"
  for m in vcs simclr; do
    c=configs/cifar10_hpMC_${m}_${v}_views4_800ep_seed0.yaml; n=gate_${m}_${v}
    "$PYTHON" -m vcs_ssl.train --config "$c" --run-id "$n" --allow-cpu --smoke-steps 6 --smoke-epoch-steps 3 --no-epoch-eval 2>&1 | tail -1 || fail "smoke $n"
    "$PYTHON" -m vcs_ssl.train --config "$c" --run-id "${n}_r" --allow-cpu --smoke-steps 4 --smoke-epoch-steps 2 --no-epoch-eval --stop-after-steps 2 2>&1 | tail -1
    "$PYTHON" -m vcs_ssl.train --config "$c" --run-id "${n}_r" --allow-cpu --smoke-steps 4 --smoke-epoch-steps 2 --no-epoch-eval --resume "$GATE/${n}_r/checkpoints/last.pt" 2>&1 | tail -1 || fail "resume $n"
    echo "$n: $("$PYTHON" -c "import json;print(json.load(open('$GATE/$n/status.json'))['status'], json.load(open('$GATE/${n}_r/status.json'))['status'])")"
  done
done
"$PYTHON" - "$REP" <<'PY'
import json, sys, os
rep = sys.argv[1]; out = {}
for v in ("q1024m999", "noqueue"):
    f = f"{rep}/constant_map_{v}.json"
    if os.path.isfile(f):
        c = json.load(open(f)); out[v] = {k: c[k] for k in ("use_queue", "momentum_m", "queue_size", "max_J_p133_constant_map", "argmax", "half_batch_se",
                                                         "J_p133_gradient_ascent", "J_p133_adversarial", "random_J_p133_quantiles",
                                                         "max_J_p100_construction_same_candidates", "infonce_best_constant", "infonce_chance_log1pN",
                                                         "pass_p133", "pass_positive_control", "train_J_first_last", "step_seconds_cpu")}
json.dump(out, open(f"{rep}/summary.json", "w"), indent=1); print(json.dumps(out, indent=1))
PY
rm -rf "$GATE"
echo "gate_rc=$rc_all"; echo "exit=$rc_all end=$(date -u +%FT%TZ)"; exit $rc_all
