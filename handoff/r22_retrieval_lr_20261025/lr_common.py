import hashlib
import importlib.util
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
OLD = BASE / 'handoff/r22_other_apps_20261023'
def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
def write(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(p.name + f'.{os.getpid()}.tmp')
    t.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n'); t.replace(p)
def verify():
    for rel, h in read(HERE / 'FREEZE.json')['files'].items(): assert sha(BASE / rel) == h, rel
    return sha(HERE / 'FREEZE.json')
def load(name, p):
    import sys
    spec = importlib.util.spec_from_file_location(name, p)
    obj = importlib.util.module_from_spec(spec); sys.modules[name] = obj; spec.loader.exec_module(obj)
    return obj
def reference(seed, method):
    item = read(HERE / 'SOURCE_LOCK.json')['controls'][f'{method}_{seed}']
    p = Path(item['directory'])
    for name, h in item['files'].items(): assert sha(p / name) == h, (seed, method, name)
    return p, read(p / 'COMPLETE.json'), read(p / 'run_manifest.json')
def cpu_pass():
    r = read(HERE / 'checks/CPU.json'); assert r['status'] == 'PASS' and r['freeze_sha256'] == verify()
    assert sha(HERE / 'checks/CPU_TRACE.json') == r['trace_sha256']
    return r
