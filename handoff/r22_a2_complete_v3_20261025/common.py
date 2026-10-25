"""A2 completion: immutable source checks and the existing pool's admission guards."""
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
REPO = BASE / 'ssl_pilot_v9_r22a2_dev'
SRC = REPO / 'v9_r22/vcsloss'
OLD = BASE / 'outputs_v9_r22/vcsloss'
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r22_a2_complete_v3_20261025')
ARMS = ('VCS_SHARP', 'VCSW_SHARP', 'BCE_SHARP')
SEEDS = (902, 903)

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
    for p, h in read(HERE/'FREEZE.json')['files'].items(): assert sha(p) == h, p
    for p, rec in read(HERE/'INPUT_LOCK.json')['large_data_stat'].items():
        st = Path(p).stat(); assert st.st_size == rec['size'] and st.st_mtime_ns == rec['mtime_ns'], p
    return sha(HERE/'FREEZE.json')
def source():
    spec = importlib.util.spec_from_file_location('a2_original', SRC/'r22_student.py')
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
    return mod
def reference(seed, arm):
    rec = read(HERE/'CONTROL_LOCK.json')['units'][f'{seed}_{arm}']
    return read(Path(rec['directory'])/'unit_result.json'), read(Path(rec['directory'])/'run_manifest.json')
def storage():
    h = read(BASE/'handoff/r22_monitor_codex_20261023/HOME_USAGE.json')
    assert time.time()-h['unix'] < 7200 and h['total_gib'] < 1450, 'home storage guard'
    st = os.statvfs('/projects/EEG-foundation-model'); free = st.f_bavail*st.f_frsize
    assert free >= 300_000_000_000 + 6*(1 << 30), 'reserve complete six-GiB output budget above projects floor'
    return dict(home_gib=h['total_gib'], home_utc=h['utc'], projects_free_bytes=free, reserved_output_gib=6)
def cpu_pass():
    r=read(HERE/'checks/CPU.json')
    assert r['status']=='PASS' and r['freeze_sha256']==verify()
    assert sha(HERE/'checks/CPU_TRACE.json')==r['trace_sha256']
    return r
def gate_pass():
    f=verify();c=cpu_pass();r=read(OUT/'gates/COMPLETE.json')
    assert r['status']=='COMPLETE' and r['freeze_sha256']==f and r['cpu_sha256']==sha(HERE/'checks/CPU.json')
    assert r['trace_sha256']==sha(OUT/'gates/OPEN_TRACE.json') and read(OUT/'gates/OPEN_TRACE.json')['status']=='PASS'
    assert {(x['seed'],x['arm']) for x in r['rows']}=={(s,a) for s in SEEDS for a in ARMS}
    for x in r['rows']:
        assert x['epochs']==40 and x['updates']==47 and x['pool_checks']['ok'] and x['model_changed'] and x['b_changed']
        d=OUT/'gates'/f"s{x['seed']}__{x['arm']}"
        assert sha(d/'train_log.json')==x['log_sha256'] and sha(d/'last.pt')==x['checkpoint_sha256']
    return r
