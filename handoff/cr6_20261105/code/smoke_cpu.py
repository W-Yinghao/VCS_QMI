"""CPU code smoke for CR6 (not a formal unit; not frozen-bound). Writes only under the given scratch dir."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, '/home/infres/yinwang/CS_QMI/handoff/cr6_20261105/code/site')
import torch
import cr6_common as K
W = Path(sys.argv[1]); W.mkdir(parents=True, exist_ok=True); dev = 'cpu'; t0 = time.time()
import cr6_source as SRC, cr6_tail as TL, cr6_eval as EV, acceptance_cr6 as AC, freeze_cr6 as FR
SRC.fit(K.bank_path('Flickr60'), K.prep_dir('Flickr60'), 'J', 'E01', W/'est', dev, 2002000, 20261105200200, steps=3, saves=(0, 3), unit='SMOKE'); print('est ok', time.time()-t0, flush=True)
cr5 = K.read_json(K.CR5_TREE/'PROTOCOL_FROZEN_CR5.json')['units']
gj = cr5['CR5_Flickr60_J_KEEP_TOP1_COMPLETE_PC_s1002']['graph_path']
S = TL.TailStudent('Flickr60', gj, 1002, 1002, dev); AC.light_tail_checks('Flickr60', gj, W, dev, student=S, prefix='S'); print('light ok', AC.R, time.time()-t0, flush=True)
TL.train('Flickr60', gj, W/'tail', 'SMOKE', 1002, 20261104202, 1002, dev, max_steps=2, student=S); print('tail train ok', time.time()-t0, flush=True)
ref = AC.cr5_dir('Flickr60', 'CR5_Flickr60_OBS_COMPLETE_PC_s1002')
r = EV.evaluate_ckpt('Flickr60', ref/'step_000000.pt', W/'ev0', dev); old = K.read_json(ref/'eval_000000/METRICS.json')
print('eval frozen equal', r['metrics'] == old['metrics'], r['metrics']['rSum'], old['metrics']['rSum'], time.time()-t0, flush=True)
gc = cr5['CR5_CC152K_RECOVERED_20261104_OBS_COMPLETE_PC_s1002']['graph_path']
Sc = TL.TailStudent('CC152K_RECOVERED_20261104', gc, 1002, 1002, dev)
TL.train('CC152K_RECOVERED_20261104', gc, W/'tailcc', 'SMOKE', 1002, 20261104202, 1002, dev, max_steps=1, student=Sc)
rt = EV.evaluate_ckpt('CC152K_RECOVERED_20261104', W/'tailcc'/'step_000000.pt', W/'evcc_tail0', dev)
refc = AC.cr5_dir('CC152K_RECOVERED_20261104', 'CR5_CC152K_RECOVERED_20261104_OBS_COMPLETE_PC_s1002'); oc = K.read_json(refc/'eval_000000/METRICS.json')
print('tail eval cc step0', rt['metrics']['rSum'], 'frozen', oc['metrics']['rSum'], rt['features'], time.time()-t0, flush=True)
# freeze + summarize dry run in scratch
K.FROZEN = W/'PROTOCOL_FROZEN_CR6.json'; K.TREE_REAL = K.TREE
fake = W/'ACC.json'; fake.write_text(json.dumps(dict(status='ALL_PASS', env={}, job=['x', 'y'])))
sys.argv = ['freeze', str(fake)]
import cr6_common
orig_tree = K.TREE
FR.main(); print('freeze ok', K.sha256(K.FROZEN), flush=True)
fz = K.frozen_check(); print('frozen_check ok', len(fz['units']), len(fz['references']), flush=True)
import summarize_cr6 as SM
K.TREE = W; SM.main(); print('summarize ok', time.time()-t0, flush=True)
