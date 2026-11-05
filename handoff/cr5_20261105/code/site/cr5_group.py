"""CR5 matched group = one condition x one seed in ONE job on ONE GPU (CR3_A1/CR4_A1 rule): all 6 graphs x {RECT, COMPLETE_PC}.
CC OBS/J_MASS_OBS PC: single-positive check + 250-step PC trajectory vs the same job's RECT step-250 (bitwise) -> reuse RECT, else train."""
import argparse, json, os, socket, subprocess, sys, time
from pathlib import Path
import numpy as np, torch
from scipy.sparse import load_npz
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import cr5_student as S5                                                   # noqa: E402
from cr1.io import read_json, write_json, sha256                          # noqa: E402
TREE = HERE.parents[1]; OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_cr5/CR5_20261105'); CR1 = Path('/projects/EEG-foundation-model/yinghao/outputs_cr1/CR1_20261104')
PY = '/home/infres/yinwang/CS_QMI/env_nr1/bin/python'; EVAL = '/home/infres/yinwang/CS_QMI/handoff/cr1_20261104/code/site/eval_student.py'
CCD = '/projects/EEG-foundation-model/yinghao/data/cc152k_raw/nr1/CC152K_RECOVERED_20261104'; NR1 = '/home/infres/yinwang/CS_QMI/outputs_nr1/NR1_20261104'
JOB = os.environ.get('SLURM_JOB_ID'); HOST = socket.gethostname(); utc = lambda: time.strftime('%FT%TZ', time.gmtime())
def env():
    drv = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], capture_output=True, text=True).stdout.split('\n')[0].strip()
    return {'gpu_name': torch.cuda.get_device_name(), 'driver': drv, 'torch': torch.__version__, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version()}
def evaluate(d, c):
    dv, tr = (f'{CCD}/cache/dev', f'{CCD}/evaluator_only/dev_positives.json') if c.startswith('CC152K') else (f'{NR1}/cache/dev', f'{NR1}/evaluator_only/dev_positives.json')
    p = subprocess.run([PY, '-I', '-B', EVAL, '--no-frozen-check', '--student', str(d), '--dev-cache', dv, '--truth', tr, '--prepared', str(CR1/c/'prepared'), '--device', 'cuda'], capture_output=True, text=True)
    (Path(d)/'EVAL_STDOUT.txt').write_text(p.stdout + p.stderr[-4000:])
    if p.returncode: raise RuntimeError(f'eval failed {d}: {p.stderr[-2000:]}')
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--condition', required=True); ap.add_argument('--seed', type=int, required=True); ap.add_argument('--dry-run', action='store_true'); a = ap.parse_args()
    fz = read_json(TREE/'PROTOCOL_FROZEN_CR5.json')
    for rel, h in fz['code_sha256'].items():
        if sha256(Path(rel) if rel.startswith('/') else TREE/rel) != h: raise RuntimeError(f'frozen code changed {rel}')
    for p, h in fz['data_sha256'].items():
        if sha256(p) != h: raise RuntimeError(f'frozen data changed {p}')
    units = [u for u, i in fz['units'].items() if i['condition'] == a.condition and i['head_seed'] == a.seed]
    order = sorted(units, key=lambda u: (fz['units'][u]['relation'], fz['units'][u]['consumer'] != 'RECT'))   # RECT before PC per graph
    if a.dry_run: print('DRY_RUN_OK', a.condition, a.seed, len(order), [fz['units'][u]['graph_path'] for u in order][:2]); return
    g = OUT/'groups'/f'{a.condition}_s{a.seed}'; g.mkdir(parents=True, exist_ok=True); e = env(); ge = g/'GROUP_ENV.json'
    if ge.exists():
        prev = read_json(ge)
        if {k: prev[k] for k in e} != e: raise RuntimeError(f'GROUP ENV MISMATCH {prev} vs {e}')
    else: write_json(ge, dict(e, condition=a.condition, seed=a.seed, first_job=JOB, host=HOST, utc=utc(), units=order))
    bank, prep = str(CR1/'banks'/a.condition/'fit_bank.npz'), str(CR1/a.condition/'prepared')
    for u in order:
        i = fz['units'][u]; d = OUT/'students'/a.condition/u
        if (d/'UNIT_COMPLETE.json').exists(): continue
        d.mkdir(parents=True, exist_ok=True)
        with open(d/'ATTEMPTS.txt', 'a') as f: f.write(f"{JOB} {HOST} {e['gpu_name']}, {e['driver']} {utc()}\n")
        if i['plan'].startswith('IDENTITY'):
            m = load_npz(i['graph_path']).tocsr(); rows = np.diff(m.indptr); cols = np.diff(m.tocsc().indptr); single = bool((rows == 1).all() and ((cols == 0) | (cols == 1)).all())
            ru = u.replace('COMPLETE_PC', 'RECT'); rd = OUT/'students'/a.condition/ru; t = d/'identity_trajectory'
            if not (t/'step_000250.pt').exists(): S5.train(bank, prep, i['graph_path'], t, u, 'COMPLETE_PC', i['head_seed'], i['sampler_seed'], 'cuda', 250, (0, 250))
            x = torch.load(t/'step_000250.pt', map_location='cpu', weights_only=False)['model']; y = torch.load(rd/'step_000250.pt', map_location='cpu', weights_only=False)['model']
            bit = all(torch.equal(x[k], y[k]) for k in x); dmax = max(float((x[k]-y[k]).abs().max()) for k in x); ok = single and bit
            write_json(d/'IDENTITY_RECEIPT.json', dict(unit=u, single_positive=single, trajectory250_bitwise_vs_same_job_RECT=bit, max_abs=dmax, decision='REUSE_SAME_JOB_RECT' if ok else 'TRAIN_NEW_FALLBACK', reused_dir=str(rd) if ok else None))
            if ok: write_json(d/'UNIT_COMPLETE.json', {'unit': u, 'status': 'REUSE_SAME_JOB_RECT', 'reused_dir': str(rd), 'job': JOB, 'utc': utc()}); print('IDENTITY_REUSE', u, flush=True); continue
        if not (d/'DONE.json').exists():
            r = S5.train(bank, prep, i['graph_path'], d, u, i['consumer'], i['head_seed'], i['sampler_seed'], 'cuda'); r['group_env'] = e; write_json(d/'DONE.json', r)
        evaluate(d, a.condition); write_json(d/'UNIT_COMPLETE.json', {'unit': u, 'status': 'COMPLETE', 'job': JOB, 'utc': utc(), 'env': e}); print('UNIT_COMPLETE', u, flush=True)
    print('GROUP_COMPLETE', a.condition, a.seed, flush=True)
if __name__ == '__main__': main()
