import hashlib
import importlib.util
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def write(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(p)


def verify():
    for p, h in read(HERE/'FREEZE.json')['files'].items():
        assert sha(BASE/p) == h, p
    return sha(HERE/'FREEZE.json')


def geometry():
    spec = importlib.util.spec_from_file_location('reg_trajectory_long_host', BASE/'handoff/r22_geometry_long_v2_20261024/run.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m
