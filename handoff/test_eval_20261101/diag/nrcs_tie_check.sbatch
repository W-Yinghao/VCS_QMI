#!/bin/bash
#SBATCH --job-name=te_nrcs_tiecheck
#SBATCH --qos=normal
#SBATCH --partition=L40S,RTX6000PRO,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/test_eval_20261101/diag/%x_%j.out
# DEV only (no TEST read): for R32 views that failed the TEST tool's DEV gate, count exact score ties and compare the
# package retrieval_metrics (pessimistic stable ties = training readout) with the upstream i2t/t2i used by the TEST tool.
set -uo pipefail
SRC=/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data/f30k_precomp
LOC=$TMPDIR/tc_$SLURM_JOB_ID/f30k_precomp; mkdir -p $LOC; trap 'rm -rf $TMPDIR/tc_$SLURM_JOB_ID' EXIT
cp $SRC/dev_ims.npy $SRC/dev_caps.txt $LOC/
cd /home/infres/yinwang/CS_QMI/handoff/r32_native_ratio_20261101/nrcs
/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python -B -u - <<PY
import sys, json, numpy as np, torch
sys.path.insert(0, '/home/infres/yinwang/CS_QMI/handoff/r32_native_ratio_20261101/package/src')
from test_entry import Nrcs
from native_ratio.retrieval import retrieval_metrics
V='/projects/EEG-foundation-model/yinghao/outputs_v9/r32_native_ratio_20261101/test_view'
ev = Nrcs('$TMPDIR/tc_$SLURM_JOB_ID'); pool = ev._split('dev')
for v in ['f00_s1001/e2_ratio_J_b1_i0', 'f00_s1001/e2_ratio_J_b1_i0p1', 'f00_s1001/e2_ratio_logistic_b1_i0p1', 'f60_s1001/e2_ratio_J_b1_i0', 'f00_s1001/e2_cosine_J_b0_i0']:
    import glob, os
    cands = [f'{V}/{v}/model_best.pth.tar']
    ck = torch.load(os.path.realpath(cands[0]), map_location='cpu', weights_only=False)
    m = ev.ga.GPOTwoTower(ck['nrcs_arm'], ev.w2i); m.load_state_dict(ck['model']); m.cuda().eval()
    with torch.no_grad(): s = ev._scores(m, pool)
    pkg = retrieval_metrics(s, pool.mapping); up = ev._recalls(s)
    rows_with_ties = int(sum(len(np.unique(r)) < len(r) for r in s)); uniq = len(np.unique(s))
    print(json.dumps(dict(view=v, ckpt=os.path.basename(cands[0]), unique_scores=uniq, n_scores=int(s.size), rows_with_ties=rows_with_ties,
          pkg_rsum=round(sum(pkg[k] for k in ('i2t_r1','i2t_r5','i2t_r10','t2i_r1','t2i_r5','t2i_r10')), 3), upstream_rsum=round(up['rsum'], 3),
          logged=json.load(open(f'{V}/{v}/DONE.json'))['logged_dev']['rsum'])), flush=True)
    del m; torch.cuda.empty_cache()
PY
