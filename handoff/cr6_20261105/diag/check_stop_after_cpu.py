"""CPU check of the E1 fix: clone (CR1 seeds, production T_max=8000, stopped after 250) vs the frozen CR1 fit_estimator run in the same job."""
import subprocess, sys, json
from pathlib import Path
sys.path.insert(0, '/home/infres/yinwang/CS_QMI/handoff/cr6_20261105/code/site')
import torch, cr6_common as K, cr6_source as SRC
sys.path.insert(0, str(K.CR5_TREE/'code'/'site')); import cr5_student as S5
W = Path(sys.argv[1]); W.mkdir(parents=True, exist_ok=True); bank, prep = K.bank_path('Flickr60'), K.prep_dir('Flickr60')
with K.stop_after(250): SRC.fit(bank, prep, 'J', 'E01', W/'clone', 'cpu', 1001, [1001, 0, 51], steps=8000, unit='CHK')
r = subprocess.run([K.PY, '-I', '-B', str(K.CR1_TREE/'code/site/cr1_prod.py'), 'fit-estimator', '--bank', str(bank), '--prepared', str(prep),
                    '--unit', 'CR1_Flickr60_EST_J_E01_s1001', '--out', str(W/'prod'), '--device', 'cpu'], capture_output=True, text=True); assert r.returncode == 0, r.stderr[-2000:]
x, y = [torch.load(W/d/'step_000250.pt', map_location='cpu', weights_only=False) for d in ('clone', 'prod')]
print('E1 cpu model bitwise', all(torch.equal(x['model'][k], y['model'][k]) for k in x['model']), 'sched', x['scheduler'] == y['scheduler'], 'has step 8000 in clone', (W/'clone'/'step_008000.pt').exists())
g = K.read_json(K.CR5_TREE/'PROTOCOL_FROZEN_CR5.json')['units']['CR5_Flickr60_OBS_COMPLETE_PC_s1002']['graph_path']
for t in ('a', 'b'):
    with K.stop_after(250): S5.train(str(bank), str(prep), g, W/f's5{t}', 'CHK', 'COMPLETE_PC', 1002, 20261104202, 'cpu')
a, b = [torch.load(W/f's5{t}'/'step_000250.pt', map_location='cpu', weights_only=False) for t in 'ab']
print('S5 stop_after determinism', all(torch.equal(a['model'][k], b['model'][k]) for k in a['model']), 'T_max', a['scheduler']['T_max'], 'done?', (W/'s5a'/'DONE.json').exists())
