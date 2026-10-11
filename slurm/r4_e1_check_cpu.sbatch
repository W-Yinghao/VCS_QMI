#!/usr/bin/env bash
#SBATCH --job-name=r4_e1_check
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# R4-E1 pre-launch check: unit tests, smoke of the runner (seeds 0 and 1, smoke roles, 60 updates, branch at 30) and of the aggregator (written to
# outputs/R4_E1_smoke, never to reports/), and a full-size throughput timing (nothing saved).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16; rc=0; O=$OUTPUT_ROOT/R4_E1_smoke
echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} start=$(date -u +%FT%TZ)"
"$PYTHON" -m unittest tests/test_r4_e1_solver.py -v 2>&1 | tail -25; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
for s in 0 1; do t0=$(date +%s); "$PYTHON" scripts/r4_e1_solver.py --seed $s --out $O --smoke 2>&1 | tail -16; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1; echo "[timing] smoke seed $s: $(( $(date +%s) - t0 )) s"; done
"$PYTHON" scripts/r4_e1_aggregate.py --smoke --in-dir $O --out-dir $O 2>&1 | tail -40; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
"$PYTHON" - <<'PY' || rc=1
# full-size throughput: 100 updates of each fit criterion + 3 readouts on the real seed-0 roles; throwaway critic, nothing written
import sys, time, torch; sys.path.insert(0, "scripts"); import os
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
import r4_e1_solver as E1
from vcs_estim import benchmark as BM; from vcs_estim.critics import JointMLP; from vcs_estim.data import setting_from_mi
st = setting_from_mi("gaussian", 6.0, 20); roles, _ = BM.build_roles(st, 20, 20, 4096, 0); tr = BM.truths(st, 20, roles["TRUTH"])
E = roles["EVAL"]; ctx = E1.Ctx(roles, tr, (BM.eta_of(st, E.xp, E.yp, 20), BM.eta_of(st, E.xq, E.yq, 20)), False)
for obj, half in (("js", 0.5), ("J", 0.5), ("J", 1.0)):
    torch.manual_seed(99); c = JointMLP(20, 20); o = torch.optim.Adam(c.parameters(), lr=1e-4); g = torch.Generator().manual_seed(99)
    t0 = time.time(); E1.train(ctx, c, o, g, obj, half, "js", 0, 99, [], {}, [], []); tu = (time.time() - t0) / 99
    t0 = time.time(); [ctx.readout(c, half, "js") for _ in range(3)]; tr_ = (time.time() - t0) / 3
    print(f"[timing full size] {obj} half {half}: {tu * 1000:.1f} ms / update, {tr_:.2f} s / readout -> 2000-update trajectory with 21 readouts ~ {(2000 * tu + 21 * tr_) / 60:.1f} min", flush=True)
PY
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
