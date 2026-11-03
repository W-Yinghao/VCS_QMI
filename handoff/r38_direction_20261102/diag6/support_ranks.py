"""diag6 item 1/5 (GPU, evaluator process, observed information only, no TEST file opened): native-H rank of the OBSERVED partner of
every CC152K training record over the full 150000 pool, both directions, with the candidate-table score rule of the parent's prep
(peer-mean native fp32 score (xA.yA + xB.yB)/2, same chunking as prep.candidate_tables). Record k = (image k, caption k).
rank = number of pool items with a strictly larger score (0 = first); ties = number of other items with an equal score.
Usage: python -B support_ranks.py <parent_key>   -> OUT6/support/ranks_<parent>.npz + .json"""
from __future__ import annotations
import sys, time, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import units6 as U6
pk = sys.argv[1]
P = U6.PARENTS[pk]
sys.path.insert(0, str(U6.TREES[P['tree']]))
import numpy as np
import torch
from common import install_guards, guard_record, N_TRAIN_IMG, N_TRAIN_CAP, PEERS, CONDS, sha256
install_guards(role='audit')
from data35 import load_cond

torch.backends.cuda.matmul.allow_tf32 = False
dev = torch.device('cuda')
t0 = time.time()
d = load_cond(P['cond'], dev, verify=True, need_prep=True)
obs = d['obs_np']; assert np.array_equal(obs, np.arange(N_TRAIN_CAP))
xs = [d[('fit', 'img', p)] for p in PEERS]; ys = [d[('fit', 'cap', p)] for p in PEERS]
r_i = np.zeros(N_TRAIN_IMG, np.int32); t_i = np.zeros(N_TRAIN_IMG, np.int32)
r_t = np.zeros(N_TRAIN_CAP, np.int32); t_t = np.zeros(N_TRAIN_CAP, np.int32)
with torch.no_grad():
    for i in range(0, N_TRAIN_IMG, 1000):
        sc = sum(x[i:i + 1000] @ y.T for x, y in zip(xs, ys)) / 2
        n = sc.shape[0]; ar = torch.arange(n, device=dev)
        so = sc[ar, ar + i][:, None]
        r_i[i:i + n] = (sc > so).sum(1).int().cpu().numpy(); t_i[i:i + n] = ((sc == so).sum(1) - 1).int().cpu().numpy()
    for j in range(0, N_TRAIN_CAP, 4000):
        sc = sum(y[j:j + 4000] @ x.T for x, y in zip(xs, ys)) / 2
        n = sc.shape[0]; ar = torch.arange(n, device=dev)
        so = sc[ar, ar + j][:, None]
        r_t[j:j + n] = (sc > so).sum(1).int().cpu().numpy(); t_t[j:j + n] = ((sc == so).sum(1) - 1).int().cpu().numpy()
ci = d['cand_i2t'].astype(np.int64); ct = d['cand_t2i'].astype(np.int64)
in_i = (ci == np.arange(N_TRAIN_IMG)[:, None]).any(1); in_t = (ct == obs[:, None]).any(1)
pos_i = np.where(ci == np.arange(N_TRAIN_IMG)[:, None], np.arange(32)[None], -1).max(1)
pos_t = np.where(ct == obs[:, None], np.arange(32)[None], -1).max(1)
chk = {}
for dn, r, t, inn, pos in (('i2t', r_i, t_i, in_i, pos_i), ('t2i', r_t, t_t, in_t, pos_t)):
    mism = (r < 32) != inn
    chk[dn] = dict(n_mismatch_rank_lt32_vs_table=int(mism.sum()), n_mismatch_with_ties=int((mism & (t > 0)).sum()),
                   n_table_position_ne_rank=int(((pos >= 0) & (pos != r)).sum()), n_table_position_ne_rank_without_ties=int(((pos >= 0) & (pos != r) & (t == 0)).sum()),
                   n_records_with_ties=int((t > 0).sum()))
out = U6.OUT6 / 'support'; out.mkdir(parents=True, exist_ok=True)
np.savez(out / f'ranks_{pk}.npz', rank_i2t=r_i, ties_i2t=t_i, rank_t2i=r_t, ties_t2i=t_t, in_top32_i2t=in_i, in_top32_t2i=in_t, trust=d['trust_subset'])
c = CONDS[P['cond']]
U6.write_json(out / f'ranks_{pk}.json', dict(parent=pk, cond=P['cond'], tree=P['tree'], h_unit=c['h_unit'], cache_manifest_sha256=d['manifest_sha256'],
                                             prep_manifest_sha256=d['prep_manifest_sha256'], candidate_table_sha256=d['cand_sha256'], trust_sha256=d['trust_sha256'],
                                             table_consistency=chk, seconds=time.time() - t0, gpu=torch.cuda.get_device_name(0), host=os.uname().nodename,
                                             job=os.environ.get('SLURM_JOB_ID'), test_guard=guard_record(),
                                             rule='rank of the observed partner (record identity) under the peer-mean native fp32 score over all 150000 pool items; '
                                                  'strictly-greater count; observed relation only (CC152K has no truth)'))
print('SUPPORT_RANKS_DONE', pk, chk, time.time() - t0)
