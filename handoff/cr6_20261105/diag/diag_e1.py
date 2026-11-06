"""Diagnose acceptance 1090990 E1 mismatch (V100): CR1 prod run-to-run vs CR6 clone vs clone-after-acceptance-imports. Writes only under argv[1]."""
import json, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, '/home/infres/yinwang/CS_QMI/handoff/cr6_20261105/code/site')
import torch, numpy as np
import cr6_common as K
W = Path(sys.argv[1]); dev = sys.argv[2]; mode = sys.argv[3] if len(sys.argv) > 3 else 'all'; W.mkdir(parents=True, exist_ok=True)
bank, prep = K.bank_path('Flickr60'), K.prep_dir('Flickr60'); out = {}
def ld(p): return torch.load(p, map_location='cpu', weights_only=False)
def cmp(a, b):
    x, y = ld(a), ld(b); m = {k: float((x['model'][k]-y['model'][k]).abs().max()) for k in x['model']}
    return dict(bitwise=all(v == 0 for v in m.values()), max_abs=max(m.values()), per_tensor_nonzero={k: v for k, v in m.items() if v > 0})
def prod(tag):
    d = W/f'prod_{tag}'
    if not (d/'DONE.json').exists():
        r = subprocess.run([K.PY, '-I', '-B', str(K.CR1_TREE/'code/site/cr1_prod.py'), 'fit-estimator', '--bank', str(bank), '--prepared', str(prep),
                            '--unit', 'CR1_Flickr60_EST_J_E01_s1001', '--out', str(d), '--device', dev], capture_output=True, text=True)
        if r.returncode: raise RuntimeError(r.stderr[-2000:])
    return d
def clone_fresh(tag):
    d = W/f'clone_{tag}'
    if not (d/'REPLAY.json').exists():
        code = ("import sys;sys.path.insert(0,'/home/infres/yinwang/CS_QMI/handoff/cr6_20261105/code/site');import cr6_source as S,cr6_common as K;"
                f"S.fit(K.bank_path('Flickr60'),K.prep_dir('Flickr60'),'J','E01','{d}','{dev}',1001,[1001,0,51],steps=250,saves=(0,1,2,10,250),unit='DIAG')")
        r = subprocess.run([K.PY, '-I', '-B', '-c', code], capture_output=True, text=True)
        if r.returncode: raise RuntimeError(r.stderr[-2000:])
    return d
t = time.time()
pa, pb = prod('a'), prod('b'); out['prod_a_vs_prod_b_step250'] = cmp(pa/'step_000250.pt', pb/'step_000250.pt'); out['prod_a_vs_prod_b_step8000'] = cmp(pa/'step_008000.pt', pb/'step_008000.pt')
ca, cb = clone_fresh('a'), clone_fresh('b'); out['clone_a_vs_clone_b_step250'] = cmp(ca/'step_000250.pt', cb/'step_000250.pt')
out['clone_a_vs_prod_a_step0'] = cmp(ca/'step_000000.pt', pa/'step_000000.pt'); out['clone_a_vs_prod_a_step250'] = cmp(ca/'step_000250.pt', pa/'step_000250.pt')
out['clone_a_vs_prod_b_step250'] = cmp(ca/'step_000250.pt', pb/'step_000250.pt')
h = K.CR1/'Flickr60'/'experts'/'J'/'E01'
out['prod_a_vs_history_step250'] = cmp(pa/'step_000250.pt', h/'step_000250.pt'); out['clone_a_vs_history_step250'] = cmp(ca/'step_000250.pt', h/'step_000250.pt')
# first-step trajectories: log of prod has step1 loss/grad; clone step 1/2/10 checkpoints vs each other
out['clone_a_vs_clone_b_step1'] = cmp(ca/'step_000001.pt', cb/'step_000001.pt'); out['clone_a_vs_clone_b_step10'] = cmp(ca/'step_000010.pt', cb/'step_000010.pt')
la = [json.loads(l) for l in (pa/'TRAIN.jsonl').read_text().splitlines()[:3]]; lb = [json.loads(l) for l in (pb/'TRAIN.jsonl').read_text().splitlines()[:3]]
lc = [json.loads(l) for l in (ca/'TRAIN.jsonl').read_text().splitlines()[:3]]
out['train_log_head'] = dict(prod_a=la, prod_b=lb, clone_a=lc)
out.update(device=dev, gpu=torch.cuda.get_device_name() if dev.startswith('cuda') else 'cpu', seconds=time.time()-t, utc=K.utc())
K.write_json(Path(f'/home/infres/yinwang/CS_QMI/handoff/cr6_20261105/diag/DIAG_E1_{dev.replace(":","")}_{K.job()[0]}.json'), out)
print(json.dumps({k: (v if not isinstance(v, dict) or 'bitwise' not in v else {kk: v[kk] for kk in ('bitwise', 'max_abs')}) for k, v in out.items() if k != 'train_log_head'}, indent=1))
