#!/bin/bash
#SBATCH --qos=runfill
#SBATCH --partition=V100-32GB,L40S,A100,A40
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r32_native_ratio_20261101/logs/%x_%j.out
# NRCS-v1 end-to-end smoke of the formal chain path in a SMOKE dir (nothing here is reused by formal units):
# W2 (2 ep x 20 steps) -> F1 export -> one F1 row (package runner) -> E2 children (2 ep x 20) incl. an interrupted+resumed one
# -> TEST-entry views -> Nrcs evaluator on DEV-as-TEST (no TEST file staged or read).
set -uo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r32_native_ratio_20261101
O=/projects/EEG-foundation-model/yinghao/outputs_v9/r32_native_ratio_20261101
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=$W/package/src:$W/nrcs:$W/upstream
S=$O/smoke/chain_$SLURM_JOB_ID; mkdir -p $S/runs
source $W/sbatch/stage_ims.sh
P=nrcs_v1_w2_n60_s1001_cosine_hinge_b0_i0p1
step() { echo "== $1"; shift; "$@"; rc=$?; echo "rc=$rc"; [ $rc -eq 0 ] || echo "SMOKE_STEP_FAILED"; }
export NRCS_SMOKE=2,20
step W2 $PY -u -B $W/nrcs/train_gpo.py --unit $P --data $O/data --ims $IMS --runs $S/runs
step EXPORT $PY -u -B $W/nrcs/export_f1.py --parent $P --data $O/data --ims $IMS --runs $S/runs --cache $S/f1_cache --bindings $S/bindings --binding-template $W/RUN_BINDING.json
step F1 bash -c "cd $W/package && $PY -B tools/run_registered.py --unit-id nrcs_v1_f1_n60_s1001_ratio_J_b1_i0 --binding $S/bindings/BINDING_n60_s1001.json --runs-root $S/runs"
for U in nrcs_v1_e2_n60_s1001_ratio_J_b1_i0p1 nrcs_v1_e2_n60_s1001_cosine_hinge_b0_i0p1; do
  step E2_$U $PY -u -B $W/nrcs/train_gpo.py --unit $U --data $O/data --ims $IMS --runs $S/runs; done
# interrupted + resumed unit (3 epochs): kill after epoch 1 is saved, then rerun the same command
U=nrcs_v1_e2_n60_s1001_cosine_J_b0_i0
NRCS_SMOKE=3,20 $PY -u -B $W/nrcs/train_gpo.py --unit $U --data $O/data --ims $IMS --runs $S/runs > $S/interrupt.log 2>&1 &
PID=$!; until grep -q "^epoch 1/3" $S/interrupt.log || ! kill -0 $PID 2>/dev/null; do sleep 1; done; kill $PID; wait $PID; echo "killed after: $(grep '^epoch' $S/interrupt.log)"
NRCS_SMOKE=3,20 step RESUME $PY -u -B $W/nrcs/train_gpo.py --unit $U --data $O/data --ims $IMS --runs $S/runs
step RERUN_DONE $PY -u -B $W/nrcs/train_gpo.py --unit $U --data $O/data --ims $IMS --runs $S/runs
step VIEWS $PY -B -c "import sys; sys.path.insert(0,'$W/nrcs'); import test_entry as t; t.build_views('$S/runs', '$S/test_view')"
D=$TMPDIR/nrcs_te_$SLURM_JOB_ID/f30k_precomp; mkdir -p $D; cp /projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data/f30k_precomp/dev_{caps.txt,ims.npy} $D/
step NRCS_EVAL_DEV_AS_TEST $PY -B - <<PYEOF
import sys, glob, json, torch, io
sys.path.insert(0, '$W/nrcs'); import test_entry as t
ev = t.Nrcs('$(dirname $D)'); ev.test_split = 'dev'
out = {}
for v in sorted(glob.glob('$S/test_view/*/*')):
    ck = torch.load(v + '/model_best.pth.tar', map_location='cpu', weights_only=False); dn = json.load(open(v + '/DONE.json'))
    r = ev.evaluate(None, ck)
    out[v] = {'logged_rsum': dn['logged_dev']['rsum'], 'recomputed_rsum': r['dev_recomputed']['rsum'], 'dev_as_test_rsum': r['test']['rsum'],
              'max_abs_diff_pp': max(abs(r['dev_recomputed'][k] - dn['logged_dev'][k]) for k in t.KEYS), 'n': [r['test']['n_images'], r['test']['n_captions']]}
json.dump(out, open('$S/NRCS_EVAL_SELFCHECK.json', 'w'), indent=1); print(json.dumps(out, indent=1))
PYEOF
ls -R $S/runs | head -60
echo SMOKE_CHAIN_END
