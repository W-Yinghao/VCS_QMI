"""CR5 acceptance (GPU recorded; no TEST, no truth): seed-1001 replays vs CR3 RECT / CR4 PC trainers (same job, bitwise);
seeds 1002/1003 change init and draws; RECT and PC share draws per seed; exact resume; CC OBS/J_MASS_OBS single-positive."""
import json, shutil, sys, time
from pathlib import Path
import numpy as np, torch
from scipy.sparse import load_npz
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import cr5_student as S5                                                  # noqa: E402
sys.path.insert(0, '/home/infres/yinwang/CS_QMI/handoff/cr3_20261104/code/site')
import cr3_student as S3                                                  # noqa: E402
S4 = S5.S4; C1 = S5.C1
CR1 = Path('/projects/EEG-foundation-model/yinghao/outputs_cr1/CR1_20261104'); R = {}
fz4 = json.load(open('/home/infres/yinwang/CS_QMI/handoff/cr4_20261104/PROTOCOL_FROZEN_CR4.json'))
gp = lambda c, r: next(i['graph_path'] for i in fz4['units'].values() if i['condition'] == c and i['relation'] == r)
def rec(n, ok, **kw): R[n] = dict(ok=bool(ok), **kw); print(n, 'PASS' if ok else 'FAIL', json.dumps(kw, default=str)[:700], flush=True)
def same(a, b):
    x = torch.load(a, map_location='cpu', weights_only=False)['model']; y = torch.load(b, map_location='cpu', weights_only=False)['model']
    return all(torch.equal(x[k], y[k]) for k in x), max(float((x[k]-y[k]).abs().max()) for k in x)
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); dev = 'cuda'; t0 = time.perf_counter()
rec('GPU_RECORD', True, gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())
c = 'Flickr60'; g = gp(c, 'J_KEEP_TOP1'); bank, prep = CR1/'banks'/c/'fit_bank.npz', CR1/c/'prepared'; sv = (0, 250)
S5.train(bank, prep, g, out/'r5', 'A', 'RECT', 1001, 20261104201, dev, 250, sv); S3.train(bank, prep, g, out/'r3', 'A', 'RECT', dev, steps=250, check_frozen=False, saves=sv)
b1 = same(out/'r5'/'step_000250.pt', out/'r3'/'step_000250.pt'); rec('R1_seed1001_RECT_equals_CR3_trainer', b1[0], max_abs=b1[1])
S5.train(bank, prep, g, out/'p5', 'A', 'COMPLETE_PC', 1001, 20261104201, dev, 250, sv); S4.train(bank, prep, g, out/'p4', 'A', 'PC', dev, steps=250, saves=sv)
b2 = same(out/'p5'/'step_000250.pt', out/'p4'/'step_000250.pt'); rec('R2_seed1001_PC_equals_CR4_trainer', b2[0], max_abs=b2[1])
for s in (1002, 1003):
    for cons in ('RECT', 'COMPLETE_PC'):
        S5.train(bank, prep, g, out/f'{cons}_{s}', 'A', cons, s, 20261104201+(s-1001), dev, 50, (0, 50), draw_log=out/f'{cons}_{s}.jsonl')
dr = {s: (out/f'RECT_{s}.jsonl').read_text() == (out/f'COMPLETE_PC_{s}.jsonl').read_text() for s in (1002, 1003)}
i0 = same(out/'RECT_1002'/'step_000000.pt', out/'RECT_1003'/'step_000000.pt'); i1 = same(out/'RECT_1002'/'step_000000.pt', out/'r5'/'step_000000.pt')
dd = (out/'RECT_1002.jsonl').read_text() != (out/'RECT_1003.jsonl').read_text()
rec('R3_seeds', dr[1002] and dr[1003] and not i0[0] and not i1[0] and dd, rect_pc_same_draws=dr, init_1002_vs_1003_differs=not i0[0], init_1002_vs_1001_differs=not i1[0], draws_1002_vs_1003_differ=dd)
S5.train(bank, prep, g, out/'rA', 'A', 'COMPLETE_PC', 1002, 20261104202, dev, 300, (0, 250, 300)); (out/'rB').mkdir(exist_ok=True)
for f in ('step_000000.pt', 'step_000250.pt', 'TRAIN.jsonl'): shutil.copy(out/'rA'/f, out/'rB'/f)
rB = S5.train(bank, prep, g, out/'rB', 'A', 'COMPLETE_PC', 1002, 20261104202, dev, 300, (0, 250, 300)); b3 = same(out/'rA'/'step_000300.pt', out/'rB'/'step_000300.pt')
rec('R4_exact_resume', b3[0] and rB['resumed_from_step'] == 250, max_abs=b3[1])
sp = {}
for r in ('OBS', 'J_MASS_OBS'):
    m = load_npz(gp('CC152K_RECOVERED_20261104', r)).tocsr(); rows = np.diff(m.indptr); cols = np.diff(m.tocsc().indptr)
    sp[r] = dict(rows_all_one=bool((rows == 1).all()), cols_le_one=bool(((cols == 0) | (cols == 1)).all()))
rec('R5_CC_single_positive', all(v['rows_all_one'] and v['cols_le_one'] for v in sp.values()), graphs=sp)
allok = all(v['ok'] for v in R.values())
(out/'ACCEPTANCE_CR5.json').write_text(json.dumps(dict(all_pass=allok, checks=R, seconds=time.perf_counter()-t0, test_read=False, truth_read=False), indent=1, default=str)); print('ALL_PASS' if allok else 'ACCEPTANCE_FAIL')
