#!/usr/bin/env bash
#SBATCH --job-name=p149_gate
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P149 CPU gate: unit tests + a smoke of the fixture builder and the measurement runner on one encoder (outputs under outputs/P149_smoke).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
G=reports/P149_GATE_${SLURM_JOB_ID:-local}; mkdir -p $G; rc=0; S=$OUTPUT_ROOT/P149_smoke; mkdir -p $S
"$PYTHON" -m pytest -q tests/test_p149.py 2>&1 | tee $G/pytest_p149.txt | tail -3; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
"$PYTHON" -m pytest -q tests/test_p116.py 2>&1 | tee $G/pytest_regression.txt | tail -2; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
t0=$(date +%s); "$PYTHON" scripts/p149_twoview.py fixture --dataset cifar10 --run P107_AP3_views4_800ep_seed1 --out $S --smoke 2>&1 | tail -4; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
echo "[smoke] fixture seconds $(( $(date +%s) - t0 ))"
t0=$(date +%s); "$PYTHON" scripts/p149_twoview.py measure --fixture $S/P107_AP3_views4_800ep_seed1 --out $S/measure_P107_AP3_views4_800ep_seed1 --smoke 2>&1 | tail -8; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
echo "[smoke] measure seconds $(( $(date +%s) - t0 ))"
# full-size timing probe: one estimator x one setting at the registered sizes would need the full fixture; instead time 1000 steps of one MLP fit on random 4096 x 1024 pairs
"$PYTHON" - <<'PY' 2>&1 | tee $G/timing.txt
import time, torch, sys; sys.path.insert(0,'scripts'); sys.path.insert(0,'src'); import p149_twoview as P
torch.set_num_threads(16); P.DEV=torch.device('cpu'); d=512; g=torch.Generator().manual_seed(0)
mk=lambda n:(torch.randn(n,d,generator=g),torch.randn(n,d,generator=g))
Pf,Qf,Pt,Qt=mk(2048),mk(2048),mk(500),mk(500)
t=time.perf_counter(); r=P.fit_mlp('vcs',Pf,Qf,Pt,Qt,d,0,1000); print('mlp 3 candidates x 1000 steps:', round(time.perf_counter()-t,1),'s')
t=time.perf_counter(); r=P.fit_rff(Pf,Qf,Pt,Qt,d,0); print('rff 3 candidates:', round(time.perf_counter()-t,1),'s')
PY
echo "{\"gate_rc\": $rc}" > $G/summary.json; echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
