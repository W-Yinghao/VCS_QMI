"""A1 seed completion, immutable sources and admission to the existing pool."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
OLD = BASE/'handoff/r22_mechanism_review_20261024'
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v10/r22_a1_complete_20261025')
ARMS = ('vcs_exact', 'bce', 'nce')
SEEDS = (1002, 1003)
DATASET = {'bf': 'BF_SHG', 'epfl': 'EPFL_RGB_NIR'}

def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
def write(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n'); tmp.replace(p)
def verify():
    for p, h in read(HERE/'FREEZE.json')['files'].items(): assert sha(BASE/p) == h, p
    return sha(HERE/'FREEZE.json')
def units(): return read(HERE/'SOURCE_LOCK.json')['units']
def source_check(key):
    row = units()[key]
    for p, h in row['files'].items(): assert sha(p) == h, p
    return row
def original():
    assert os.environ.get('SLURM_JOB_ID'), 'Numerical work requires Slurm'
    sys.path.insert(0, str(OLD))
    spec = importlib.util.spec_from_file_location('a1_original_completion', OLD/'geometry_a1.py')
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
    return mod
def storage():
    h = read(BASE/'handoff/r22_monitor_codex_20261023/HOME_USAGE.json')
    assert time.time()-h['unix'] < 7200 and h['total_gib'] < 1450, 'home storage guard'
    st = os.statvfs('/projects/EEG-foundation-model'); free = st.f_bavail*st.f_frsize
    assert free >= 300_000_000_000+4*(1 << 30), 'projects floor plus complete output budget'
    return dict(home_gib=h['total_gib'], home_utc=h['utc'], projects_free_bytes=free, reserved_output_gib=4)
def cpu_pass():
    r = read(HERE/'checks/CPU.json')
    assert r['status'] == 'PASS' and r['freeze_sha256'] == verify()
    assert set(r['units']) == set(units()) and sha(HERE/'checks/CPU_TRACE.json') == r['trace_sha256']
    return r
def gate_pass(group):
    f = verify(); cpu_pass(); r = read(OUT/'gates'/group/'COMPLETE.json')
    assert r['status'] == 'COMPLETE' and r['freeze_sha256'] == f
    assert r['cpu_sha256'] == sha(HERE/'checks/CPU.json')
    expected = {k for k, v in units().items() if v['gate_group'] == group}
    assert {x['key'] for x in r['checks']} == expected
    assert sha(OUT/'gates'/group/'INPUT_TRACE.json') == r['trace_sha256']
    assert set(r['roles_read']) <= {'FIT', 'CAL'}
    for x in r['checks']:
        assert len(x['steps']) == 3 and x['model_changed'] and x['optimizer_rng_roundtrip'] and x['cal_gradient_finite']
        assert sha(OUT/'gates'/group/(x['key']+'.pt')) == x['checkpoint_sha256']
        assert x['environment_matched'] and x['baseline_batches_matched']
    return r
