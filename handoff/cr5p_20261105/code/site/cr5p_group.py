"""CR5_P1105 matched group = one condition x one student seed in ONE job on ONE GPU: 6 graphs x COMPLETE_PC.
Trainer: frozen handoff/cr5_20261105/code/site/cr5_student.py (CR4 PC step; seeded), imported read-only; accepted in CR5 job 1089910."""
import argparse, json, os, socket, subprocess, sys, time
from pathlib import Path
import torch
sys.path.insert(0, '/home/infres/yinwang/CS_QMI/handoff/cr5_20261105/code/site')
import cr5_student as S5                                                   # noqa: E402
from cr1.io import read_json, write_json, sha256                          # noqa: E402
TREE = Path(__file__).resolve().parents[2]; OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_cr5p/CR5P1105'); CR1 = Path('/projects/EEG-foundation-model/yinghao/outputs_cr1/CR1_20261104')
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
    fz = read_json(TREE/'PROTOCOL_FROZEN_CR5_P1105.json')
    for rel, h in fz['code_sha256'].items():
        if sha256(Path(rel) if rel.startswith('/') else TREE/rel) != h: raise RuntimeError(f'frozen code changed {rel}')
    for p, h in fz['data_sha256'].items():
        if sha256(p) != h: raise RuntimeError(f'frozen data changed {p}')
    units = [u for u, i in fz['units'].items() if i['condition'] == a.condition and i['student_seed'] == a.seed]
    if a.dry_run: print('DRY_RUN_OK', a.condition, a.seed, len(units)); return
    g = OUT/'groups'/f'{a.condition}_t{a.seed}'; g.mkdir(parents=True, exist_ok=True); e = env(); ge = g/'GROUP_ENV.json'
    if ge.exists():
        prev = read_json(ge)
        if {k: prev[k] for k in e} != e: raise RuntimeError(f'GROUP ENV MISMATCH {prev} vs {e}')
    else: write_json(ge, dict(e, condition=a.condition, seed=a.seed, first_job=JOB, host=HOST, utc=utc(), units=units))
    bank, prep = str(CR1/'banks'/a.condition/'fit_bank.npz'), str(CR1/a.condition/'prepared')
    for u in units:
        i = fz['units'][u]; d = OUT/'students'/a.condition/u
        if (d/'UNIT_COMPLETE.json').exists(): continue
        d.mkdir(parents=True, exist_ok=True)
        with open(d/'ATTEMPTS.txt', 'a') as f: f.write(f"{JOB} {HOST} {e['gpu_name']}, {e['driver']} {utc()}\n")
        if not (d/'DONE.json').exists():
            r = S5.train(bank, prep, i['graph_path'], d, u, 'COMPLETE_PC', i['student_seed'], i['sampler_seed'], 'cuda'); r['group_env'] = e; r['package_unit_id'] = i['package_unit_id']; write_json(d/'DONE.json', r)
        evaluate(d, a.condition); write_json(d/'UNIT_COMPLETE.json', {'unit': u, 'status': 'COMPLETE', 'job': JOB, 'utc': utc(), 'env': e}); print('UNIT_COMPLETE', u, flush=True)
    print('GROUP_COMPLETE', a.condition, a.seed, flush=True)
if __name__ == '__main__': main()
