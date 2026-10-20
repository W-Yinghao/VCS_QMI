"""BF1_CAPACITY_EXT_K8_K32 hardware amendment (2026-10-19): the 6 not-yet-started seed-1003 units may run on RTX6000PRO, L40S or
H100.  Only the hardware rule changes.  The scientific path (frozen BF1 tree b8f59ea + the frozen K override in run.py) and the base
bookkeeping (capacity_common under base FREEZE e4157194) are imported unchanged.  No Torch at import.
"""
from __future__ import annotations

import os
from pathlib import Path

import capacity_common as C

NAME = 'BF1_CAPACITY_EXT_K8_K32_HW_AMENDMENT_20261019'
BASE_FREEZE = 'e4157194c5421e406c5abdc3b0decd733c0e1dcb554db1dc6d5def0c39a8023b'
BASE_PREREG = 'f366a8053e0099487586e2e49e10af1fbdc2e30d971914a610b3acbfce5c0d18'
DIR = C.BUNDLE / 'hw_amendment_20261019'
OUT_HW = Path('/projects/EEG-foundation-model/yinghao/outputs_v10/capacity_ext_hw_20261019')
RECORD_GIT_PATH = 'v10/capacity_ext/HARDWARE_AMENDMENT_20261019.json'
SCHEMA_UNIT = 'v10.capacity_ext.unit.hw_amendment.v1'
GPUS = {'RTX6000PRO': 'NVIDIA RTX PRO 6000 Blackwell Server Edition', 'L40S': 'NVIDIA L40S', 'H100': 'NVIDIA H100 NVL'}
PARTITIONS = 'RTX6000PRO,L40S,H100'
EXCLUDED = ('node51', 'node53', 'node60')
GATED = ('RTX6000PRO', 'H100')          # K8/K32 CUDA gate required before a unit may run on these types
UNIT_KEYS = tuple(C.unit_key(f, K, 1003) for f in C.FAMILIES for K in C.KS)
CODE_FILES = ('scripts/v10_capacity/hw_amendment.py', 'scripts/v10_capacity/run_hw.py', 'scripts/v10_capacity/hw_gate.py',
              'scripts/v10_capacity/check_hw.py', 'scripts/v10_capacity/capacity_table_hw.py',
              'scripts/v10_capacity/hw_amendment_record.py', 'scripts/v10_capacity/freeze_hw.py',
              'slurm/v10_capacity/run_hw.sbatch', 'slurm/v10_capacity/hw_gate.sbatch')


# ------------------------------------------------------------------------------------------------ freeze / record
def frozen_hw(expected=None):
    """The amendment freeze as submitted, on top of the unchanged base freeze."""
    expected = expected or os.environ.get('V10_CAPACITY_HW_FREEZE')
    C.require(expected, 'V10_CAPACITY_HW_FREEZE (sha256 of the amendment FREEZE.json) is required')
    C.require(C.sha(DIR / 'FREEZE.json') == expected, 'Hardware-amendment freeze changed after submission')
    f = C.read(DIR / 'FREEZE.json')
    C.require(f['base_freeze_sha256'] == BASE_FREEZE and C.sha(C.BUNDLE / 'FREEZE.json') == BASE_FREEZE, 'Base freeze changed')
    for name, digest in f['code_files'].items():
        C.require(C.sha(C.REPO / name) == digest, 'Frozen amendment code changed: ' + name)
    C.require(C.sha(DIR / 'HARDWARE_AMENDMENT.json') == f['record_sha256'] == C.sha(C.REPO / RECORD_GIT_PATH),
              'Amendment record changed or differs from the committed copy')
    return C.read(DIR / 'HARDWARE_AMENDMENT.json')


def checks_passed_hw(hw_freeze):
    p = DIR / 'CHECKS.json'
    C.require(p.exists(), 'Amendment control checks have not been recorded')
    c = C.read(p)
    C.require(c.get('status') == 'PASS' and c.get('hw_freeze_sha256') == hw_freeze, 'Amendment control checks did not pass for this freeze')
    return c


def hw_units(prereg):
    units = [u for u in prereg['units'] if u['key'] in UNIT_KEYS]
    C.require([u['key'] for u in units] == list(UNIT_KEYS), 'Amendment unit set differs from the 6 seed-1003 units')
    return units


def hw_unit(prereg, key):
    C.require(key in UNIT_KEYS, 'Not an amended unit (the 12 completed L40S units are never re-run): ' + str(key))
    return C.unit_by_key(prereg, key)


def validate_record(rec, prereg):
    C.require(rec['schema'] == 'v10.capacity_ext.hardware_amendment.v1' and rec['name'] == NAME, 'Wrong amendment record')
    C.require(rec['amends']['freeze']['sha256'] == BASE_FREEZE and rec['amends']['preregistration']['sha256'] == BASE_PREREG
              and C.sha(C.BUNDLE / 'PREREGISTRATION.json') == BASE_PREREG, 'Amendment does not cite the frozen base')
    C.require(rec['hardware']['accepted_gpus'] == GPUS and rec['hardware']['partition'] == PARTITIONS
              and tuple(rec['hardware']['exclude']) == EXCLUDED and tuple(rec['hardware']['gate_required_for']) == GATED,
              'Hardware rule differs from the frozen code')
    C.require([u['key'] for u in rec['scope']['units']] == list(UNIT_KEYS), 'Amendment scope differs')
    for u, p in zip(rec['scope']['units'], hw_units(prereg)):
        C.require(u['job_name'] == p['job_name'] and u['time_limit_seconds'] == p['time_limit_seconds'], 'Job name or time limit changed')
    C.require(rec['scientific_changes'] == [] and rec['budget']['ceiling_seconds'] == C.CEILING_SECONDS
              and rec['budget']['time_limit_seconds'] == C.TIME_LIMIT_SECONDS, 'Science or budget changed')
    C.require(rec['outputs']['root'] == str(OUT_HW), 'Output root changed')
    return rec


# ------------------------------------------------------------------------------------------------ hardware
def hardware_problem(env):
    """None if the recorded environment is an accepted GPU / partition / node, else the reason."""
    partition, gpu, host = env.get('slurm_partition'), env.get('gpu_name'), env.get('host')
    if partition not in GPUS:
        return f'partition {partition!r} not in {PARTITIONS}'
    if gpu != GPUS[partition]:
        return f'GPU {gpu!r} is not the accepted {GPUS[partition]!r} of partition {partition}'
    if host in EXCLUDED or not isinstance(host, str):
        return f'node {host!r} is excluded'
    return None


def runtime_environment(bf1):
    """BF1 runtime_check with the GPU-name rule replaced by the three accepted types; every other BF1 version, thread,
    precision-setting and environment-variable check is kept against the BF1 PROTOCOL."""
    C.require(os.environ.get('SLURM_JOB_ID'), 'Hardware inspection must run inside Slurm')
    env = bf1.A.environment('cuda')
    p = C.read(C.BF1 / 'PROTOCOL.json')['precision']
    problem = hardware_problem(env)
    C.require(problem is None, 'Hardware outside the amendment: ' + str(problem))
    for key in ('torch', 'cuda_runtime', 'cudnn', 'torch_threads'):
        C.require(env[key] == p['environment'][key], 'Environment mismatch: ' + key)
    C.require(bf1.A.precision_settings(env['precision']) == p['settings'], 'Precision settings differ from accepted T02')
    C.require(env['precision']['env'] == p['env'], 'Precision environment variables differ')
    env['hardware_amendment'] = {'record_sha256': C.sha(DIR / 'HARDWARE_AMENDMENT.json'), 'partition': env['slurm_partition'],
                                 'gpu_name': env['gpu_name'], 'node': env['host'], 'runtime_status': 'PASS'}
    return env


def gate_passed(partition, hw_freeze):
    """RTX6000PRO / H100 need a passing K8/K16/K32 CUDA gate of this amendment freeze on the same GPU type."""
    if partition not in GATED:
        return None
    gates = [C.read(p) for p in sorted((DIR / 'gpu_gates').glob('*.json'))] if (DIR / 'gpu_gates').exists() else []
    ok = [g for g in gates if g.get('status') == 'PASS' and g.get('hw_freeze_sha256') == hw_freeze
          and g.get('partition') == partition and g.get('gpu_name') == GPUS[partition]]
    if not ok:
        raise C.Refused('HW_GATE_MISSING', f'no passing K8/K32 CUDA gate for {partition} under this amendment freeze')
    return ok[-1]


def base_untouched(unit):
    """An amended unit must never have started under the base freeze (its base output directory must not exist)."""
    if (C.OUT / unit['key']).exists():
        raise C.Refused('BASE_UNIT_STARTED', f"{unit['key']} has a base-freeze output directory; it stays a base unit")


def expected_identity_hw(unit, prereg, base_freeze, hw_freeze):
    return {**C.expected_identity(unit, prereg, base_freeze), 'schema': SCHEMA_UNIT,
            'hardware_amendment_sha256': C.sha(DIR / 'HARDWARE_AMENDMENT.json'), 'capacity_hw_freeze_sha256': hw_freeze}


def hardware_of(directory, artifacts):
    envs = [C.read(Path(directory) / n) for n in artifacts if n.startswith('ENV_') and n.endswith('.json')]
    return sorted({(e['gpu_name'], e['slurm_partition'], e['host']) for e in envs})


# ------------------------------------------------------------------------------------------------ verification
def verify_result(unit, prereg, capacity_freeze_sha, require_receipt=True, *, hw_freeze_sha):
    """capacity_common.verify_result with exactly three declared replacements (check_hw.py compares the sources): the output root,
    the amended identity and the hardware rule (three accepted GPU types, partition-name pairs, node51/53/60 excluded)."""
    d = OUT_HW / unit['key']
    r = C.read(d / 'COMPLETE.json')
    ident = expected_identity_hw(unit, prereg, capacity_freeze_sha, hw_freeze_sha)
    C.require(r['status'] == 'COMPLETE' and r['identity'] == ident and r['steps'] == 3000, 'Wrong or incomplete unit identity')
    C.require(C.read(d / 'IDENTITY.json') == ident, 'IDENTITY.json differs')
    cps = r['checkpoints']
    C.require(tuple(c['step'] for c in cps) == C.CAL_STEPS and all(c['metrics']['n'] == 80 for c in cps), 'Incomplete CAL checkpoints')
    for c in cps:
        rows = C.lines(d / 'CAL' / f"step_{c['step']:04d}.jsonl")
        C.require(len({(x['core'], x['perturbation']) for x in rows}) == 80 and C.metrics(rows) == c['metrics']
                  and all(x['role'] == 'CAL' and x['checkpoint_sha256'] == c['checkpoint_sha256'] for x in rows), 'CAL rows mismatch')
    chosen = min(cps, key=C.selection_key)
    C.require(r['selected'] == chosen, 'Selected checkpoint is not the CAL lexicographic choice')
    dev = C.lines(d / 'DEV.jsonl')
    C.require(len({(x['core'], x['perturbation']) for x in dev}) == 320 and len(dev) == 320, 'Incomplete DEV32')
    C.require(all(x['role'] == 'DEV' and x['checkpoint_sha256'] == chosen['checkpoint_sha256'] for x in dev), 'DEV not at the selected checkpoint')
    C.require(C.read(d / 'DEV.identity.json')['checkpoint_sha256'] == chosen['checkpoint_sha256'], 'DEV identity checkpoint')
    C.require(C.metrics(dev) == r['DEV32'] and r['DEV32']['n'] == 320 and r['DEV24']['n'] == 240, 'DEV metrics mismatch')
    k16 = C.read(Path(unit['paired_k16_final']) / 'COMPLETE.json')
    C.require(r['DEV24_cores'] == k16['DEV24_cores'] and len(r['DEV24_cores']) == 24
              and not set(r['DEV24_cores']) & set(C.DEV24_EXCLUDED), 'DEV24 core set differs from the paired K16 final')
    C.require(set(r['per_core']) == set(k16['per_core']) and len(r['per_core']) == 32, 'DEV32 core set differs')
    required = {'IDENTITY.json', 'CONFIG.json', 'TRAIN_STEPS.jsonl', 'DATA_READS.json', 'DEV.jsonl', 'DEV.summary.json', 'DEV.identity.json'}
    for step in C.CAL_STEPS:
        required |= {f'checkpoints/step_{step:04d}.pt', f'checkpoints/step_{step:04d}.json', f'CAL/step_{step:04d}.jsonl',
                     f'CAL/step_{step:04d}.summary.json', f'CAL/step_{step:04d}.identity.json'}
    artifacts = r['artifacts']
    C.require(required <= set(artifacts), 'Missing required artifacts: ' + str(sorted(required - set(artifacts))))
    for name, digest in artifacts.items():
        path = (d / name).resolve()
        C.require(path.is_relative_to(d.resolve()) and C.sha(path) == digest, 'Result artifact changed: ' + name)
    steps = [x['step'] for x in C.lines(d / 'TRAIN_STEPS.jsonl')]
    C.require(steps == list(range(1, 3001)), 'TRAIN_STEPS is not steps 1..3000 exactly once')
    config = C.read(d / 'CONFIG.json')
    c = prereg['families'][unit['family']]['candidate']
    C.require(config['K'] == unit['K'] and config['method'] == unit['method'] and config['seed'] == unit['seed']
              and config['encoder_lr'] == c['lr'] and config['head_lr'] == c['lr']
              and config['m1']['max_iterations'] == (c['cap'] or 30), 'CONFIG differs from the unit')
    settings = C.read(prereg['bf1_basis']['protocol']['path'])['precision']
    envs = [C.read(d / n) for n in artifacts if n.startswith('ENV_') and n.endswith('.json')]
    C.require(envs, 'No runtime environment record')
    for e in envs:
        C.require(hardware_problem(e) is None, 'Unit ran outside the amended hardware: ' + str(hardware_problem(e)))
        C.require(e['precision']['env'] == settings['env'], 'Precision environment variables differ')
        C.require({k: e['precision'].get(k) for k in settings['settings']} == settings['settings'], 'Precision settings differ')
    reads = [C.read(d / n) for n in artifacts if n == 'DATA_READS.json' or (n.startswith('READS_') and n.endswith('.json'))]
    C.require(all(x['role'] in ('FIT', 'CAL', 'DEV') for rs in reads for x in rs), 'Unexpected data role read')
    C.require(not (C.BUNDLE / 'failures' / f"{r['job_id']}.json").exists(), 'The completing job recorded a failure')
    if require_receipt:
        path = C.BUNDLE / 'exits' / f"{r['job_id']}.json"
        C.require(path.exists(), 'No exit receipt for the completing job')
        receipt = C.read(path)
        C.require(receipt['exit_code'] == 0 and receipt['unit'] == unit['key'], 'No successful exit receipt for the completing job')
    return r
