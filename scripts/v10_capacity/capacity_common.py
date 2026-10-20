"""BF1_CAPACITY_EXT_K8_K32: units, identities, cost clock, admission (budget ceiling) and result verification.

No Torch and no scientific imports here.  The scientific path is the frozen BF1 tree, imported only by run.py.
"""
from __future__ import annotations

import contextlib
import datetime
import fcntl
import hashlib
import json
import math
import os
import re
import statistics
import time
from pathlib import Path

NAME = 'BF1_CAPACITY_EXT_K8_K32'
SCHEMA_UNIT = 'v10.capacity_ext.unit.v1'
REPO = Path(__file__).resolve().parents[2]
BUNDLE = Path('/home/infres/yinwang/CS_QMI/handoff/v10_capacity_ext_20261019')
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v10/capacity_ext_20261019')
BF1 = Path('/home/infres/yinwang/CS_QMI/handoff/v10_bf1_20261017')
BF1_SOURCE = BF1 / 'source'
BF1_COMMIT = 'b8f59ea825dc7a2064457d47a628bef09ca4e185'
CORE_HASH, HOST_HASH = 'e9b4239450c4e2eb', '6c6f094684fbc50a'
PREREG_GIT_PATH = 'v10/capacity_ext/PREREGISTRATION.json'

FAMILIES = ('MF_VCS_perm1', 'MF_VCS_M0', 'MF_VCS_M1')
METHODS = {'MF_VCS_perm1': 'MF_perm1', 'MF_VCS_M0': 'MF_M0', 'MF_VCS_M1': 'MF_M1'}
SHORT = {'MF_VCS_perm1': 'p1', 'MF_VCS_M0': 'm0', 'MF_VCS_M1': 'm1'}
KS = (8, 32)
SEEDS = (1001, 1002, 1003)
TIME_LIMIT_SECONDS = {'MF_VCS_perm1': 5400, 'MF_VCS_M0': 5400, 'MF_VCS_M1': 10800}
RESERVE_MARGIN_SECONDS = 120          # kill grace + receipt after the Slurm limit; part of every reservation
NON_ADMITTED_PROVISIONAL_SECONDS = 300  # a refused/early-exit job before its receipt exists
CEILING_SECONDS = 30 * 3600           # 30 allocated L40S GPU-hours for all 18 units including any resume jobs
CAL_STEPS = tuple(range(250, 3001, 250))
DEV24_EXCLUDED = ('1B_F4', '1B_H4', '1B_H9', '2B_D5', '2B_G3', '2B_G5', '2B_H2', '2B_I5')
ALLOWED_NODES_EXCLUDED = ('node51', 'node60')
EXIT_REFUSED = 3


class Refused(RuntimeError):
    """A pre-registered refusal (budget ceiling, duplicate, pause): reason-coded, not a scientific failure."""

    def __init__(self, code, detail):
        super().__init__(f'{code}: {detail}')
        self.code, self.detail = code, detail


# ------------------------------------------------------------------------------------------------ io
def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2 ** 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def lines(path):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]


def require(value, reason):
    if not value:
        raise RuntimeError(reason)


@contextlib.contextmanager
def exclusive(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


# ------------------------------------------------------------------------------------------------ units
def unit_key(family, K, seed):
    return f'{family}_K{K}_seed{seed}'


def job_name(family, K, seed):
    return f'v10k_{SHORT[family]}_k{K}_{seed}'


def slurm_time(seconds):
    return f'{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}'


def planned_units():
    """Frozen priority order: seed-major (1001 first), then family, then K.  If the ceiling binds, whole later seeds are lost
    first instead of whole family-K cells; the order never depends on any result."""
    out = []
    for seed in SEEDS:
        for family in FAMILIES:
            for K in KS:
                out.append({'order': len(out), 'key': unit_key(family, K, seed), 'family': family, 'method': METHODS[family],
                            'K': K, 'seed': seed, 'job_name': job_name(family, K, seed),
                            'time_limit_seconds': TIME_LIMIT_SECONDS[family], 'time_limit': slurm_time(TIME_LIMIT_SECONDS[family]),
                            'paired_k16_final': str(BF1 / 'results' / 'FINAL_DEV' / family / f'seed_{seed}')})
    return out


def unit_by_key(prereg, key):
    matches = [u for u in prereg['units'] if u['key'] == key]
    require(len(matches) == 1, 'Unknown or duplicated unit key: ' + str(key))
    return matches[0]


# ------------------------------------------------------------------------------------------------ freeze / pre-registration
def checked(ref):
    require(sha(ref['path']) == ref['sha256'], 'Pinned reference changed: ' + ref['path'])
    return read(ref['path'])


def frozen(expected_freeze=None):
    """The bundle freeze as submitted: FREEZE.json bytes (sha passed at submission), every frozen code file, the pre-registration
    record (bundle copy == committed git copy) and the passing CPU checks of this freeze."""
    expected_freeze = expected_freeze or os.environ.get('V10_CAPACITY_FREEZE')
    require(expected_freeze, 'V10_CAPACITY_FREEZE (sha256 of FREEZE.json) is required')
    require(sha(BUNDLE / 'FREEZE.json') == expected_freeze, 'Capacity freeze changed after submission')
    freeze = read(BUNDLE / 'FREEZE.json')
    for name, digest in freeze['code_files'].items():
        require(sha(REPO / name) == digest, 'Frozen capacity code changed: ' + name)
    require(sha(BUNDLE / 'PREREGISTRATION.json') == freeze['preregistration_sha256'], 'Pre-registration changed')
    require(sha(REPO / PREREG_GIT_PATH) == freeze['preregistration_sha256'], 'Committed pre-registration differs from the bundle copy')
    return read(BUNDLE / 'PREREGISTRATION.json')


def checks_passed(freeze_sha):
    p = BUNDLE / 'CHECKS.json'
    require(p.exists(), 'CPU checks have not been recorded for this freeze')
    c = read(p)
    require(c.get('status') == 'PASS' and c.get('freeze_sha256') == freeze_sha, 'CPU checks did not pass for this freeze')
    return c


def validate_prereg(p):
    require(p['schema'] == 'v10.bf1_capacity_extension.preregistration.v1' and p['name'] == NAME, 'Wrong pre-registration')
    require(p['status'] == 'PREREGISTERED_BEFORE_ANY_RUN', 'Pre-registration status changed')
    require(tuple(p['families']) == FAMILIES and tuple(p['K']) == KS and tuple(p['seeds']) == SEEDS, 'Matrix changed')
    require(p['units'] == planned_units(), 'Unit table differs from the frozen code (keys, order, names, time limits)')
    require(len({u['key'] for u in p['units']}) == 18 and len({u['job_name'] for u in p['units']}) == 18, 'Duplicate units')
    require(p['tuning']['new_candidates'] == 0 and p['tuning']['K_specific_tuning'] is False, 'K8/K32 tuning is not allowed')
    require(p['data_access']['TEST134'] == 'never read' and p['data_access']['EPFL'] == 'never read', 'Held data access')
    require(p['budget']['ceiling_seconds'] == CEILING_SECONDS and p['budget']['time_limit_seconds'] == TIME_LIMIT_SECONDS
            and p['budget']['reserve_margin_seconds'] == RESERVE_MARGIN_SECONDS, 'Budget rule changed')
    b = p['bf1_basis']
    require(b['source_commit'] == BF1_COMMIT and b['core_hash'] == CORE_HASH and b['host_hash'] == HOST_HASH, 'BF1 basis changed')
    for name in ('freeze', 'gpu_gate', 'protocol', 'data_lock'):
        checked(b[name])
    for family in FAMILIES:
        f = p['families'][family]
        sel = checked(f['selection'])
        require(sel['status'] == 'SELECTED' and sel['family'] == family and sel['selected']['candidate'] == f['candidate'],
                'Inherited K16 selection changed: ' + family)
        require(f['method'] == METHODS[family], 'Method name changed')
        for seed in SEEDS:
            ref = p['K_reference']['units'][family][str(seed)]
            checked(ref)
    return p


def expected_identity(unit, prereg, capacity_freeze_sha):
    """The full unit identity, reconstructed from the pre-registration only (run.py builds it through BF1 identity() and
    must obtain exactly this)."""
    f = prereg['families'][unit['family']]
    c = f['candidate']
    return {'schema': SCHEMA_UNIT, 'freeze_sha256': prereg['bf1_basis']['freeze']['sha256'], 'family': unit['family'],
            'method': unit['method'], 'seed': unit['seed'], 'lr': c['lr'], 'cap': c['cap'], 'phase': 'FINAL_DEV', 'steps': 3000,
            'checkpoint_interval': 250, 'K': unit['K'], 'dependency': f['selection']['sha256'],
            'data_lock_sha256': prereg['bf1_basis']['data_lock']['sha256'], 'extension': NAME,
            'preregistration_sha256': sha(BUNDLE / 'PREREGISTRATION.json'), 'capacity_freeze_sha256': capacity_freeze_sha,
            'result_scope': 'CAPACITY_EXTENSION_SEPARATE_TABLE'}


# ------------------------------------------------------------------------------------------------ Slurm clock and cost
def slurm_seconds(text):
    """Slurm durations as squeue prints them: D-HH:MM:SS, HH:MM:SS, MM:SS (also H:MM:SS, M:SS)."""
    text = (text or '').strip()
    m = re.fullmatch(r'(?:(\d+)-)?(\d+):(\d+)(?::(\d+))?', text)
    if not m:
        raise ValueError('Unparseable Slurm duration: ' + repr(text))
    days, a, b, c = m.groups()
    if c is None:
        if days is not None:
            raise ValueError('Unexpected D-HH:MM duration: ' + text)
        h, mi, s = 0, int(a), int(b)
    else:
        h, mi, s = int(a), int(b), int(c)
    if mi >= 60 or s >= 60:
        raise ValueError('Invalid Slurm duration: ' + text)
    return ((int(days or 0) * 24 + h) * 60 + mi) * 60 + s


def slurm_start_unix(text):
    """squeue %S printed with TZ=UTC (the sbatch exports TZ=UTC before calling squeue)."""
    text = (text or '').strip()
    return datetime.datetime.strptime(text, '%Y-%m-%dT%H:%M:%S').replace(tzinfo=datetime.timezone.utc).timestamp()


class Clock:
    """Allocated time from the Slurm StartTime without mixing clocks: node-clock elapsed since the batch shell started plus the
    controller-StartTime -> batch-start delay, the latter clipped at 0 because a node clock that lags the controller (node57 lagged
    ~90 s during BF1) would otherwise make it negative.  The controller's own elapsed (squeue %M at exit) is recorded in the receipt."""

    def __init__(self, sched_start_raw, batch_start_node):
        self.sched_start_raw = sched_start_raw
        self.batch_start = float(batch_start_node)
        try:
            self.sched_start = slurm_start_unix(sched_start_raw)
            self.source = 'slurm_starttime'
        except ValueError:
            self.sched_start = None
            self.source = 'batch_start_only_starttime_unavailable'
        self.startup = 0.0 if self.sched_start is None else max(0.0, self.batch_start - self.sched_start)

    @classmethod
    def from_env(cls):
        return cls(os.environ.get('V10K_SCHED_START_RAW', ''), os.environ['V10K_BATCH_START'])

    def elapsed(self, now=None):
        now = time.time() if now is None else now
        return max(0.0, now - self.batch_start) + self.startup

    def record(self):
        return {'sched_start_raw': self.sched_start_raw, 'sched_start_unix': self.sched_start, 'batch_start_unix_node': self.batch_start,
                'startup_seconds_clipped': self.startup, 'clock_source': self.source}


class Allocation:
    """BF1 run_unit's allocation interface (check(), record()); FINAL-type unit, so no in-run budget alarm: the ceiling is
    guaranteed by admission reservations (each reservation = the unit's Slurm limit + margin)."""

    def __init__(self, unit, job, clock):
        self.unit, self.job, self.clock = unit, job, clock
        self.last = 0.0
        self.record('START', time_limit_seconds=unit['time_limit_seconds'], node=os.environ.get('SLURMD_NODENAME'),
                    clock=clock.record())

    def elapsed(self):
        return self.clock.elapsed()

    def record(self, event, **extra):
        append(BUNDLE / 'COST_LEDGER.jsonl', {'utc': utc(), 'event': event, 'job': self.job, 'unit': self.unit['key'],
                                             'family': self.unit['family'], 'K': self.unit['K'], 'seed': self.unit['seed'],
                                             'allocated_seconds': self.elapsed(), 'gpu_count': 1,
                                             'accounting': 'Slurm StartTime clock (skew-clipped); receipt adds controller elapsed', **extra})
        self.last = time.monotonic()

    def check(self):
        if time.monotonic() - self.last > 30:
            self.record('HEARTBEAT')


def receipt_cost(receipt):
    return float(receipt['allocated_seconds'])


def committed_seconds(exclude_job=None):
    """Budget committed by every job of this extension: the receipt's allocated seconds when the job has exited; otherwise its
    reservation when admitted, or a small provisional amount for a job that was refused / is still before admission."""
    detail = {}
    for a in lines(BUNDLE / 'ADMISSIONS.jsonl'):
        if a['job'] != exclude_job:
            detail[a['job']] = a
    exits = {p.stem: read(p) for p in (BUNDLE / 'exits').glob('*.json')} if (BUNDLE / 'exits').exists() else {}
    total, rows = 0.0, []
    for job in sorted(set(detail) | set(exits)):
        if job == exclude_job:
            continue
        if job in exits:
            cost, kind = receipt_cost(exits[job]), 'actual_receipt'
        elif detail[job]['decision'] == 'ADMITTED':
            cost, kind = float(detail[job]['reserved_seconds']), 'reserved_until_receipt'
        else:
            cost, kind = float(NON_ADMITTED_PROVISIONAL_SECONDS), 'provisional_non_admitted'
        total += cost
        rows.append({'job': job, 'unit': (detail.get(job) or exits.get(job) or {}).get('unit'), 'seconds': cost, 'basis': kind})
    return total, rows


def admit(unit, job, reserved_seconds, already_complete=False):
    """Atomic admission under the bundle lock.  Refusal codes: PAUSED, DUPLICATE_OR_UNRECONCILED (another admitted job of this unit
    has no receipt), INCOMPLETE_BUDGET_CEILING (committed + this reservation > ceiling).  Never extended based on results."""
    with exclusive(BUNDLE / 'ADMISSION.lock'):
        committed, rows = committed_seconds(exclude_job=job)
        exits = {p.stem for p in (BUNDLE / 'exits').glob('*.json')} if (BUNDLE / 'exits').exists() else set()
        active = [a['job'] for a in lines(BUNDLE / 'ADMISSIONS.jsonl')
                  if a['unit'] == unit['key'] and a['decision'] == 'ADMITTED' and a['job'] != job and a['job'] not in exits]
        if (BUNDLE / 'PAUSE').exists():
            decision = 'PAUSED'
        elif already_complete:
            decision = 'ALREADY_COMPLETE'
        elif active:
            decision = 'DUPLICATE_OR_UNRECONCILED'
        elif committed + reserved_seconds > CEILING_SECONDS:
            decision = 'INCOMPLETE_BUDGET_CEILING'
        else:
            decision = 'ADMITTED'
        record = {'utc': utc(), 'job': job, 'unit': unit['key'], 'decision': decision, 'reserved_seconds': reserved_seconds,
                  'committed_other_seconds': committed, 'ceiling_seconds': CEILING_SECONDS, 'active_same_unit': active,
                  'committed_breakdown': rows}
        append(BUNDLE / 'ADMISSIONS.jsonl', record)
        if decision == 'INCOMPLETE_BUDGET_CEILING':
            write(BUNDLE / 'budget' / f"{unit['key']}.json", {**record, 'status': 'INCOMPLETE_BUDGET_CEILING',
                  'rule': 'units beyond the ceiling are marked incomplete and never extended based on results'})
    return record


# ------------------------------------------------------------------------------------------------ resume preparation
def read_step_rows(path):
    """TRAIN_STEPS rows; only the final line may be a partial append (killed mid-write) and is then dropped and reported."""
    p = Path(path)
    if not p.exists():
        return [], False
    raw = p.read_text().split('\n')
    while raw and not raw[-1].strip():
        raw.pop()
    rows, partial = [], False
    for i, s in enumerate(raw):
        if not s.strip():
            raise RuntimeError('Blank line inside the step log')
        try:
            rows.append(json.loads(s))
        except json.JSONDecodeError:
            if i != len(raw) - 1:
                raise RuntimeError('Corrupt step-log line before the end') from None
            partial = True
    return rows, partial


def latest_checkpoint_step(directory):
    """The checkpoint BF1 run_unit will resume from: max 'step' over checkpoints/step_*.json (identical rule)."""
    d = Path(directory) / 'checkpoints'
    metas = sorted(d.glob('step_*.json')) if d.exists() else []
    return max((read(p)['step'] for p in metas), default=0)


def prepare_resume(directory, job):
    """Before BF1 run_unit resumes, cut TRAIN_STEPS.jsonl back to the resume checkpoint so no step is logged twice (BF1 appends
    again from the checkpoint).  The original file is kept as TRAIN_STEPS.pre_resume_<job>.jsonl.  Returns a record or None."""
    directory = Path(directory)
    path = directory / 'TRAIN_STEPS.jsonl'
    step = latest_checkpoint_step(directory)
    rows, partial = read_step_rows(path)
    if not rows and not partial and step == 0:
        return None
    keep = [r for r in rows if r['step'] <= step]
    require([r['step'] for r in keep] == list(range(1, step + 1)),
            f'Step log is not exactly steps 1..{step} once before the resume checkpoint')
    dropped = len(rows) - len(keep)
    archive = None
    if dropped or partial:
        archive = directory / f'TRAIN_STEPS.pre_resume_{job}.jsonl'
        require(not archive.exists(), 'Resume archive already exists')
        os.replace(path, archive)
        if keep:
            temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
            temp.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in keep))
            os.replace(temp, path)
    return {'resume_checkpoint_step': step, 'kept_rows': len(keep), 'dropped_rows': dropped, 'partial_final_line': partial,
            'archive': None if archive is None else archive.name}


# ------------------------------------------------------------------------------------------------ verification
def selection_key(rec):
    """BF1 selection items 1-4 for one run (the candidate ordinal is constant within a run)."""
    m = rec['metrics']
    return (-m['success42'], -m['success9'], m['median_err'], rec['step'])


def metrics(rows):
    errs = [float(r['err_full']) for r in rows]
    require(errs and all(math.isfinite(x) and x >= 0 for x in errs), 'Empty or non-finite registration results')
    return {'n': len(errs), 'success42': sum(x < 42 for x in errs) / len(errs), 'success9': sum(x < 9 for x in errs) / len(errs),
            'median_err': statistics.median(errs)}


def verify_result(unit, prereg, capacity_freeze_sha, require_receipt=True):
    """Complete-unit audit (no Torch): identity, 12 CAL checkpoints x 80, CAL-only checkpoint choice (items 1-4), DEV32/DEV24 at
    that checkpoint, every hashed artifact, unique steps 1..3000, L40S + precision in every ENV, FIT/CAL/DEV reads only."""
    d = OUT / unit['key']
    r = read(d / 'COMPLETE.json')
    ident = expected_identity(unit, prereg, capacity_freeze_sha)
    require(r['status'] == 'COMPLETE' and r['identity'] == ident and r['steps'] == 3000, 'Wrong or incomplete unit identity')
    require(read(d / 'IDENTITY.json') == ident, 'IDENTITY.json differs')
    cps = r['checkpoints']
    require(tuple(c['step'] for c in cps) == CAL_STEPS and all(c['metrics']['n'] == 80 for c in cps), 'Incomplete CAL checkpoints')
    for c in cps:
        rows = lines(d / 'CAL' / f"step_{c['step']:04d}.jsonl")
        require(len({(x['core'], x['perturbation']) for x in rows}) == 80 and metrics(rows) == c['metrics']
                and all(x['role'] == 'CAL' and x['checkpoint_sha256'] == c['checkpoint_sha256'] for x in rows), 'CAL rows mismatch')
    chosen = min(cps, key=selection_key)
    require(r['selected'] == chosen, 'Selected checkpoint is not the CAL lexicographic choice')
    dev = lines(d / 'DEV.jsonl')
    require(len({(x['core'], x['perturbation']) for x in dev}) == 320 and len(dev) == 320, 'Incomplete DEV32')
    require(all(x['role'] == 'DEV' and x['checkpoint_sha256'] == chosen['checkpoint_sha256'] for x in dev), 'DEV not at the selected checkpoint')
    require(read(d / 'DEV.identity.json')['checkpoint_sha256'] == chosen['checkpoint_sha256'], 'DEV identity checkpoint')
    require(metrics(dev) == r['DEV32'] and r['DEV32']['n'] == 320 and r['DEV24']['n'] == 240, 'DEV metrics mismatch')
    k16 = read(Path(unit['paired_k16_final']) / 'COMPLETE.json')
    require(r['DEV24_cores'] == k16['DEV24_cores'] and len(r['DEV24_cores']) == 24
            and not set(r['DEV24_cores']) & set(DEV24_EXCLUDED), 'DEV24 core set differs from the paired K16 final')
    require(set(r['per_core']) == set(k16['per_core']) and len(r['per_core']) == 32, 'DEV32 core set differs')
    required = {'IDENTITY.json', 'CONFIG.json', 'TRAIN_STEPS.jsonl', 'DATA_READS.json', 'DEV.jsonl', 'DEV.summary.json', 'DEV.identity.json'}
    for step in CAL_STEPS:
        required |= {f'checkpoints/step_{step:04d}.pt', f'checkpoints/step_{step:04d}.json', f'CAL/step_{step:04d}.jsonl',
                     f'CAL/step_{step:04d}.summary.json', f'CAL/step_{step:04d}.identity.json'}
    artifacts = r['artifacts']
    require(required <= set(artifacts), 'Missing required artifacts: ' + str(sorted(required - set(artifacts))))
    for name, digest in artifacts.items():
        path = (d / name).resolve()
        require(path.is_relative_to(d.resolve()) and sha(path) == digest, 'Result artifact changed: ' + name)
    steps = [x['step'] for x in lines(d / 'TRAIN_STEPS.jsonl')]
    require(steps == list(range(1, 3001)), 'TRAIN_STEPS is not steps 1..3000 exactly once')
    config = read(d / 'CONFIG.json')
    c = prereg['families'][unit['family']]['candidate']
    require(config['K'] == unit['K'] and config['method'] == unit['method'] and config['seed'] == unit['seed']
            and config['encoder_lr'] == c['lr'] and config['head_lr'] == c['lr']
            and config['m1']['max_iterations'] == (c['cap'] or 30), 'CONFIG differs from the unit')
    settings = read(prereg['bf1_basis']['protocol']['path'])['precision']
    envs = [read(d / n) for n in artifacts if n.startswith('ENV_') and n.endswith('.json')]
    require(envs, 'No runtime environment record')
    for e in envs:
        require(e['gpu_name'] == 'NVIDIA L40S' and e['slurm_partition'] == 'L40S'
                and e['host'] not in ALLOWED_NODES_EXCLUDED, 'Unit ran outside the pre-registered L40S hardware')
        require(e['precision']['env'] == settings['env'], 'Precision environment variables differ')
        require({k: e['precision'].get(k) for k in settings['settings']} == settings['settings'], 'Precision settings differ')
    reads = [read(d / n) for n in artifacts if n == 'DATA_READS.json' or (n.startswith('READS_') and n.endswith('.json'))]
    require(all(x['role'] in ('FIT', 'CAL', 'DEV') for rs in reads for x in rs), 'Unexpected data role read')
    require(not (BUNDLE / 'failures' / f"{r['job_id']}.json").exists(), 'The completing job recorded a failure')
    if require_receipt:
        path = BUNDLE / 'exits' / f"{r['job_id']}.json"
        require(path.exists(), 'No exit receipt for the completing job')
        receipt = read(path)
        require(receipt['exit_code'] == 0 and receipt['unit'] == unit['key'], 'No successful exit receipt for the completing job')
    return r
