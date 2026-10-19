"""Frozen experiment bookkeeping; imports no scientific runtime on the login host."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(p)


def frozen():
    m = read(ROOT / 'FROZEN.json')
    for rel, digest in m['code_sha256'].items():
        if sha(ROOT / rel) != digest:
            raise ValueError(f'Frozen code/protocol changed: {rel}')
    return m


def verify_input(m, path):
    path = str(path)
    if path not in m['input_sha256'] or sha(path) != m['input_sha256'][path]:
        raise ValueError(f'Input changed or not registered: {path}')


def oracle_pool(rows, labels, drop):
    # Return only the observed schema. Never send the truth mapping to WeightedStudent.
    for r in rows:
        if 'mismatched' in r or 'original_image_id' in r:
            raise ValueError('Labels leaked into an observed row')
        if r['role'] != 'U' and labels[r['pair_id']]['mismatched']:
            raise ValueError('An anchor was corrupted')
    pool = [dict(r) for r in rows if not drop or not labels[r['pair_id']]['mismatched']]
    weights = {r['pair_id']: float(not labels[r['pair_id']]['mismatched']) for r in pool if r['role'] == 'U'}
    return pool, weights
