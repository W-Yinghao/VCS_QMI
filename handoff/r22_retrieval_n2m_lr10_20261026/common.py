"""Shared paths/identity for the N2M hinge-host lr_update 10 study. Frozen N2M files are only read, never edited."""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
PKG = BASE / 'ssl_pilot_v9_n2m/v9_n2m/VCS_V9_N2M_Server_Package_v1_20261019'
N2M = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/n2m_20261019_v1')
EXP = N2M / 'experiment.json'
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r22_retrieval_n2m_lr10_20261026')
TRACE = BASE / 'ssl_pilot_v9_r22a2_dev/v9_n2p/c_server/trace_open.py'
SEEDS, ARMS, PERIOD = (901, 902, 903), ('VCS', 'ConstantU'), 10
UPDATES_PER_EPOCH, EPOCHS = 1133, 40


def read(p): return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()


def write(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(p.name + f'.{os.getpid()}.tmp')
    t.write_text(json.dumps(obj, ensure_ascii=False, indent=1, allow_nan=False) + '\n'); t.replace(p)


def resolve(key): return Path(key) if key.startswith('/') else BASE / key


def verify():
    for key, h in read(HERE / 'FREEZE.json')['files'].items(): assert sha(resolve(key)) == h, key
    return sha(HERE / 'FREEZE.json')


def lr_vector(period): return [2e-4 * .1 ** (e // period) for e in range(EPOCHS)]


def trace_module():
    spec = importlib.util.spec_from_file_location('r22_trace_open', TRACE)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def n2m():
    """The frozen N2M package modules (package src on sys.path; runtime src added by its own loader)."""
    if str(PKG / 'src') not in sys.path: sys.path.insert(0, str(PKG / 'src'))
    from v9n2m import controls, identity, runner, runtime, prepare, aggregate, common
    return dict(controls=controls, identity=identity, runner=runner, runtime=runtime, prepare=prepare, aggregate=aggregate, common=common)


def runtime_src(exp):
    p = str(Path(exp['runtime_dir']) / 'src')
    if p not in sys.path: sys.path.insert(0, p)


def experiment():
    """N2M experiment identity without opening noise manifests (the read trace forbids them): receipt, package code, runtime files."""
    M = n2m()
    assert sha(EXP) == (N2M / 'EXPERIMENT.sha256').read_text().split()[0]
    exp = read(EXP)
    assert M['runtime'].package_code_identity() == exp['package_code'], 'N2M package code changed'
    M['common'].assert_same_file(Path(exp['runtime_dir']) / 'RUNTIME.json', exp['runtime_receipt_sha256'])
    assert M['runtime'].verify_runtime_files(exp['runtime_dir']) == exp['source_identity'], 'N2M runtime changed'
    return exp


def unit(exp, seed, arm):
    u = [x for x in exp['units'] if x['seed'] == seed and x['arm'] == arm]; assert len(u) == 1 and u[0]['action'] == 'RUN'
    return u[0]


def training_inputs(exp, seed, arm):
    """Same inputs as the N2M unit: observed pool, control weights + lock (unit hashes), vocab/token cache (frozen hashes),
    feature files (inherited size/mtime). The noise manifest is not opened (evaluator-only content)."""
    M = n2m(); u = unit(exp, seed, arm); same = M['common'].assert_same_file
    for pk, hk in (('observed_path', 'observed_sha256'), ('weights_path', 'weights_sha256'), ('weight_lock_path', 'weight_lock_sha256')):
        same(u[pk], u[hk])
    frozen = M['identity'].pinned_freeze(exp['base_handoff']); inp = exp['inputs']
    same(inp['vocab'], frozen['input_sha256'][frozen['vocab']])
    cache = Path(inp['cache']); assert {p.name for p in cache.iterdir() if p.is_file()} == set(frozen['cache_names'])
    for name in frozen['cache_names']: same(cache / name, frozen['code_sha256']['source/token_cache/' + name])
    for old, exp_stat in frozen['data_files'].items():
        st = (Path(inp['data']) / Path(old).name).stat(); assert (st.st_size, st.st_mtime_ns) == (exp_stat['size'], exp_stat['mtime_ns']), old
    rows = M['controls'].load_observed(u['observed_path'])
    u_ids = [r['pair_id'] for r in rows if r['role'] == 'U']; assert len(rows) == 145000 and len(u_ids) == 130500
    weights = M['controls'].load_weights(u['weights_path'], u_ids, u['weight_lock_path'])
    return u, rows, weights, read(u['weight_lock_path'])


def control(seed, arm):
    """The reused lr_update-30 N2M run of the same seed/arm (files frozen in FREEZE.json)."""
    d = N2M / 'runs' / f's{seed}__{arm}'; r = read(d / 'full_result.json')
    assert r['status'] == 'COMPLETE' and r['identity']['seed'] == seed and r['identity']['arm'] == arm
    for n in ('train_log.json', 'run_manifest.json', 'identity.json'): assert sha(d / n) == r['artifact_sha256'][n], (seed, arm, n)
    return d, r, read(d / 'train_log.json'), read(d / 'run_manifest.json')


def state_sha256(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def manifest_matches(M, control_manifest, manifest, period):
    ref = json.loads(json.dumps(control_manifest)); ref['cfg']['lr_update'] = period
    return M['runner'].manifests_compatible(ref, manifest)
