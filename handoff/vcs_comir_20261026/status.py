"""Stage status for pipeline.sh: prints 'COMPLETE n/N', 'FAILED name' or 'WAITING n/N' for a stage. A unit failed when an ATTEMPT is still
RUNNING but its Slurm job has left the queue, or an ATTEMPT recorded status FAILED."""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v10/vcs_comir_20261026')
sys.path.insert(0, str(HERE))
from select_lock import cal_grid, cfg_name  # noqa: E402

OBJ = ('nce_mse', 'nce_cos', 'vcs_exp', 'vcs_cos', 'bce_exp', 'bce_cos')


def units(stage):
    if stage.startswith('smoke_'):
        ds = stage.split('_')[1]
        return [OUT / 'smoke' / f"{ds}_cal_{cfg_name(dict(obj=o, tau=(0.5 if o in ('nce_mse', 'nce_cos', 'vcs_exp', 'bce_exp') else None), lr=0.01))}_s1017_smoke"
                for o in OBJ]
    if stage == 'cal':
        return [OUT / 'runs' / f'bf_cal_{cfg_name(c)}_s1017' for c in cal_grid()]
    ds = stage.split('_')[1]
    lock = json.loads((HERE / 'LOCK.json').read_text())
    return [OUT / 'runs' / f'{ds}_final_{n}_s{s}' for n in lock['finals'] for s in (1001, 1002, 1003)]


def main(stage):
    q = subprocess.run(['squeue', '-u', 'yinwang', '-h', '-o', '%i %j'], capture_output=True, text=True).stdout.split('\n')
    queued = {line.split()[0] for line in q if line.strip()}; names = {line.split()[1] for line in q if line.strip()}
    us = units(stage); done = 0
    if stage.startswith('smoke_') and f"r24_comir_{stage}" not in names and not any((u / 'COMPLETE.json').exists() for u in us):
        if not any(list(u.glob('ATTEMPT_*.json')) for u in us):
            print('FAILED no_attempt_and_smoke_job_gone'); return
    for u in us:
        if (u / 'COMPLETE.json').exists():
            done += 1; continue
        for a in u.glob('ATTEMPT_*.json'):
            rec = json.loads(a.read_text()); job = a.stem.split('_', 1)[1]
            if rec.get('status') == 'FAILED' or (rec.get('status') == 'RUNNING' and job not in queued):
                print('FAILED', u.name); return
    print(('COMPLETE' if done == len(us) else 'WAITING'), f'{done}/{len(us)}')


if __name__ == '__main__':
    main(sys.argv[1])
