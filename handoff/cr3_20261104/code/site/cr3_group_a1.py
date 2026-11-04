"""CR3 amendment A1: one Slurm job per condition (matched group) runs all its cells sequentially on the single GPU it got.

GPU lock is per matched group (not global): the first attempt writes GROUP_ENV.json (GPU name, driver, torch, CUDA, cuDNN, host);
any later attempt of the same group must match it exactly, else it stops (no mixed-GPU group). Training uses the frozen,
unchanged cr3_student.train (byte-identical to PROTOCOL_FROZEN_CR3.json) with check_frozen=False, after this script has
performed the same code/config/data hash checks against PROTOCOL_FROZEN_CR3.json plus the A1 amendment hashes.
Each cell's DEV evaluation is a separate process (frozen CR1 eval_student.py, NR1 evaluator). Outputs: students_A1/.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import cr3_student as S                                    # unchanged frozen trainer
from cr1.io import read_json, write_json, sha256          # noqa: E402

TREE = HERE.parents[1]; OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_cr3/CR3_20261104')
CR1 = Path('/projects/EEG-foundation-model/yinghao/outputs_cr1/CR1_20261104')
PY = '/home/infres/yinwang/CS_QMI/env_nr1/bin/python'; EVAL = '/home/infres/yinwang/CS_QMI/handoff/cr1_20261104/code/site/eval_student.py'
DEV = {'CC152K_RECOVERED_20261104': ('/projects/EEG-foundation-model/yinghao/data/cc152k_raw/nr1/CC152K_RECOVERED_20261104/cache/dev',
                                     '/projects/EEG-foundation-model/yinghao/data/cc152k_raw/nr1/CC152K_RECOVERED_20261104/evaluator_only/dev_positives.json')}
FL = ('/home/infres/yinwang/CS_QMI/outputs_nr1/NR1_20261104/cache/dev', '/home/infres/yinwang/CS_QMI/outputs_nr1/NR1_20261104/evaluator_only/dev_positives.json')


def env():
    drv = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], capture_output=True, text=True).stdout.split('\n')[0].strip()
    return {'gpu_name': torch.cuda.get_device_name(), 'driver': drv, 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version()}


def check_a1(fz, a1):
    if sha256(TREE/'PROTOCOL_FROZEN_CR3.json') != a1['base_protocol_sha256']: raise RuntimeError('base protocol changed')
    for rel, h in fz['code_sha256'].items():
        if sha256(Path(rel) if rel.startswith('/') else TREE/rel) != h: raise RuntimeError(f'frozen code changed: {rel}')
    for rel, h in a1['code_sha256'].items():
        if sha256(TREE/rel) != h: raise RuntimeError(f'A1 code changed: {rel}')
    for p, h in fz['data_sha256'].items():
        if sha256(p) != h: raise RuntimeError(f'frozen data changed: {p}')
    if torch.__version__ != a1['numeric_env']['torch']: raise RuntimeError('torch differs from A1 numeric environment')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--condition', required=True); ap.add_argument('--units', default=None); a = ap.parse_args()
    fz = read_json(TREE/'PROTOCOL_FROZEN_CR3.json'); a1 = read_json(TREE/'AMENDMENT_CR3_A1.json'); check_a1(fz, a1)
    units = a1['groups'][a.condition] if a.units is None else a.units.split(',')
    gdir = OUT/'students_A1'/a.condition; gdir.mkdir(parents=True, exist_ok=True)
    e = env(); ge = gdir/'GROUP_ENV.json'
    if ge.exists():
        prev = read_json(ge)
        if {k: prev[k] for k in e} != e: raise RuntimeError(f'GROUP GPU/ENV MISMATCH: {prev} vs {e}; stop (no mixed-GPU matched group)')
    else:
        write_json(ge, dict(e, condition=a.condition, first_job=__import__('os').environ.get('SLURM_JOB_ID'), host=__import__('socket').gethostname(),
                            utc=time.strftime('%FT%TZ', time.gmtime()), units=units))
    cdev = DEV.get(a.condition, FL)
    for u in units:
        cfg = read_json(TREE/'package'/'configs'/'units'/f'{u}.json')
        if cfg['condition'] != a.condition or sha256(TREE/'package'/'configs'/'units'/f'{u}.json') != fz['unit_config_sha256'][u]: raise RuntimeError(u)
        d = gdir/u; d.mkdir(exist_ok=True)
        if (d/'UNIT_COMPLETE.json').exists(): print('skip complete', u, flush=True); continue
        with open(d/'ATTEMPTS.txt', 'a') as f: f.write(f"{__import__('os').environ.get('SLURM_JOB_ID')} {__import__('socket').gethostname()} {e['gpu_name']}, {e['driver']} {time.strftime('%FT%TZ', time.gmtime())}\n")
        rel = fz['graph_paths'][u]; bank = str(CR1/'banks'/a.condition/'fit_bank.npz'); prep = str(CR1/a.condition/'prepared')
        if not (d/'DONE.json').exists():
            r = S.train(bank, prep, rel, d, u, cfg['consumer'], 'cuda', check_frozen=False)
            r['amendment'] = 'CR3_A1'; r['group_env'] = e; write_json(d/'DONE.json', r)
        p = subprocess.run([PY, '-I', '-B', EVAL, '--no-frozen-check', '--student', str(d), '--dev-cache', cdev[0], '--truth', cdev[1],
                            '--prepared', prep, '--device', 'cuda'], capture_output=True, text=True)
        (d/'EVAL_STDOUT.txt').write_text(p.stdout + p.stderr[-4000:])
        if p.returncode: raise RuntimeError(f'eval failed {u}: {p.stderr[-2000:]}')
        write_json(d/'UNIT_COMPLETE.json', {'unit': u, 'status': 'COMPLETE', 'job': __import__('os').environ.get('SLURM_JOB_ID'), 'utc': time.strftime('%FT%TZ', time.gmtime()), 'env': e})
        print('UNIT_COMPLETE', u, flush=True)
    print('GROUP_COMPLETE', a.condition, flush=True)


if __name__ == '__main__': main()
