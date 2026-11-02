#!/bin/bash
#SBATCH --job-name=te_nrcs_rulecheck_rtx
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/test_eval_20261101/diag/%x_%j.out
# RTX6000PRO (training GPU type) variant for the f60 cosine_hinge_b0_i0p1 views only. DEV only (no TEST read). For every NRCS-v1 TEST view: compute the full-DEV score matrix twice (bitwise determinism check),
# save it (float32 npz), and compute the six Recalls / rSum under four rank rules on the SAME matrix:
#   upstream  = IN2R evaluation.py i2t/t2i (argsort, the paper/TEST-table program)
#   package   = native_ratio.retrieval_metrics (descending score, ties by increasing candidate index; the training readout)
#   tie_best  = no tied candidate counted ahead (optimistic bound)   tie_worst = every tied candidate counted ahead (pessimistic bound)
set -uo pipefail
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
SRC=/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data/f30k_precomp
LOC=$TMPDIR/rc_$SLURM_JOB_ID/f30k_precomp; mkdir -p $LOC; trap 'rm -rf $TMPDIR/rc_$SLURM_JOB_ID' EXIT
cp $SRC/dev_ims.npy $SRC/dev_caps.txt $LOC/
OUT=/projects/EEG-foundation-model/yinghao/outputs_v9/r32_native_ratio_20261101/dev_scores_rulecheck_rtx; mkdir -p $OUT
cd /home/infres/yinwang/CS_QMI/handoff/r32_native_ratio_20261101/nrcs
/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python -B -u - <<PY
import sys, os, glob, json, hashlib, numpy as np, torch
sys.path.insert(0, '/home/infres/yinwang/CS_QMI/handoff/r32_native_ratio_20261101/package/src')
from test_entry import Nrcs
from native_ratio.retrieval import retrieval_metrics
V = '/projects/EEG-foundation-model/yinghao/outputs_v9/r32_native_ratio_20261101/test_view'
OUT = '$OUT'
KEYS = ('i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10')
ev = Nrcs('$TMPDIR/rc_$SLURM_JOB_ID'); pool = ev._split('dev'); gt = np.asarray(pool.mapping)
def bounds(s, worst):
    ir = []
    for i in range(s.shape[0]):
        pos = np.flatnonzero(gt == i)
        if not len(pos): continue
        row = s[i]; b = row[pos].max()
        ir.append(1 + np.count_nonzero(row > b) + ((np.count_nonzero(row == b) - np.count_nonzero(row[pos] == b)) if worst else 0))
    tr = []
    for j, t in enumerate(gt):
        col = s[:, j]; v = col[t]
        tr.append(1 + np.count_nonzero(col > v) + ((np.count_nonzero(col == v) - 1) if worst else 0))
    d = {}
    for name, r in (('i2t', np.asarray(ir)), ('t2i', np.asarray(tr))):
        for k in (1, 5, 10): d[f'{name}_r{k}'] = float(100 * np.mean(r <= k))
    d['rsum'] = sum(d[k] for k in KEYS); return d
rows = []
for v in sorted(glob.glob(f'{V}/f60_s100*/e2_cosine_hinge_b0_i0p1*')):
    name = v.split('test_view/')[1]
    ck = torch.load(os.path.realpath(f'{v}/model_best.pth.tar'), map_location='cpu', weights_only=False)
    m = ev.ga.GPOTwoTower(ck['nrcs_arm'], ev.w2i); m.load_state_dict(ck['model']); m.cuda().eval()
    with torch.no_grad():
        s1 = ev._scores(m, pool); s2 = ev._scores(m, pool)
    del m; torch.cuda.empty_cache()
    f = f"{OUT}/{name.replace('/', '__')}.npz"; np.savez_compressed(f, scores=s1)
    up = ev._recalls(s1); pk = retrieval_metrics(s1, gt); pk['rsum'] = sum(pk[k] for k in KEYS)
    tb, tw = bounds(s1, False), bounds(s1, True)
    logged = json.load(open(f'{v}/DONE.json'))['logged_dev']
    rows.append(dict(view=name, scores_npz=f, scores_sha256=hashlib.sha256(s1.tobytes()).hexdigest(), rerun_bitwise_equal=bool(np.array_equal(s1, s2)),
                     n_unique=int(len(np.unique(s1))), logged_rsum=logged['rsum'],
                     upstream=up, package={k: pk[k] for k in KEYS + ('rsum',)}, tie_best=tb, tie_worst=tw,
                     package_minus_logged_max_pp=max(abs(pk[k] - logged[k]) for k in KEYS),
                     upstream_minus_package_rsum=up['rsum'] - pk['rsum'], tie_span_rsum=tb['rsum'] - tw['rsum']))
    print(json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in rows[-1].items() if k in ('view', 'rerun_bitwise_equal', 'logged_rsum', 'package_minus_logged_max_pp', 'upstream_minus_package_rsum', 'tie_span_rsum')}), flush=True)
json.dump(dict(note='DEV-only rank-rule cross-check for NRCS-v1 TEST views; no TEST read', gpu=torch.cuda.get_device_name(0), rows=rows),
          open('/home/infres/yinwang/CS_QMI/handoff/test_eval_20261101/diag/NRCS_RANK_RULE_CROSSCHECK_RTX_hinge60.json', 'w'), indent=1)
print('DONE', len(rows))
PY
