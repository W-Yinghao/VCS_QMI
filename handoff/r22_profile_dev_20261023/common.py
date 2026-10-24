"""Small frozen CAL/FIT diagnostics; no training or scheduler side effects here."""
import hashlib
import json
import os
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)

def verify():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Numerical diagnostics require a Slurm allocation')
    for path, digest in read(HERE / 'FREEZE.json')['files'].items():
        if sha(BASE / path) != digest:
            raise RuntimeError('Frozen input/source changed: ' + path)
    return sha(HERE / 'FREEZE.json')

def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

def receipt(start, status, **extra):
    return dict(status=status, utc=utc(), job=os.environ['SLURM_JOB_ID'],
                node=os.environ.get('SLURMD_NODENAME'), seconds=time.time()-start,
                freeze_sha256=sha(HERE/'FREEZE.json'), **extra)
