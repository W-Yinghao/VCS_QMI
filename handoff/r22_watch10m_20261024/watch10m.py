"""Ten-minute GPU snapshots and bounded reminders to the existing Codex session.

No GPU submissions, scientific choices, model spawning, or TEST access occur here.
The existing pool submits registered experiments; the original session reviews them.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
POOL = BASE / 'handoff/r22_task_diagnostics_20261023'
OLD = BASE / 'handoff/r22_monitor_codex_20261023'


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def series(jobs, old):
    groups = {}
    for source, job in jobs:
        group = job['group']
        if 'gate' in group:
            continue
        dense = re.match(r'r22_dense(\d+)_(bf|epfl)_', job['name'])
        suffix = f'{dense[1]}_{dense[2]}' if dense else group
        if group == 'other_kd':
            suffix += '_' + job.get('phase', 'unknown')
        key = Path(source).parent.name + ':' + suffix
        row = groups.setdefault(key, dict(expected=0, complete=0, source=source))
        row['expected'] += 1
        result = read(job['completion'], {})
        row['complete'] += int(result.get('status') == 'COMPLETE' and
                               result.get('freeze_sha256') == job['freeze_sha256'])
    for key, done, total in [('original_A3_seed901', min(old.get('a3_verified', 0), 2), 2),
                             ('original_A3_all_seeds', old.get('a3_verified', 0), 6),
                             ('original_NCR', old.get('ncr_complete', 0), 6),
                             ('original_KD_CRD', old.get('kd_complete', 0), 28),
                             ('original_C3', old.get('c3_complete', 0), 12)]:
        groups[key] = dict(expected=total, complete=done, source=str(OLD / 'ANALYSIS.json'))
    return groups


def transition(state, groups):
    finished = {k for k, v in groups.items() if v['complete'] == v['expected'] and v['expected'] > 0}
    # Already-reviewed results at installation are a baseline, not new discoveries.
    if 'seen_finished' not in state:
        state['baseline_finished'] = sorted(finished)
        state['unreviewed_series'] = []
    else:
        new = finished - set(state['seen_finished'])
        state['unreviewed_series'] = sorted(set(state.get('unreviewed_series', [])) | new)
    state['seen_finished'] = sorted(finished)


def probe(job):
    cmd = ['srun', '--jobid=' + job['id'], '--overlap', '--ntasks=1',
           '--cpus-per-task=1', '--mem=256M', '--time=00:01:00',
           '/usr/bin/python3', str(HERE / 'gpu_probe.py')]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=18)
        if result.returncode:
            return dict(job=job['id'], status='UNAVAILABLE', error=result.stderr[-800:])
        return dict(job=job['id'], name=job['name'], status='OBSERVED', **json.loads(result.stdout))
    except Exception as exc:
        return dict(job=job['id'], status='UNAVAILABLE', error=repr(exc))


def snapshot(sample_gpus=True):
    result = subprocess.run(['squeue', '-u', 'yinwang', '-h', '-o', '%i|%j|%T|%b|%R'],
                            capture_output=True, text=True, timeout=20)
    result.check_returncode()
    queue = []
    for line in result.stdout.splitlines():
        jid, name, state, gres, node = line.split('|', 4)
        if name.startswith('r22_') and 'gpu' in gres:
            queue.append(dict(id=jid, name=name, state=state, gres=gres, node=node))
    running = [j for j in queue if j['state'] == 'RUNNING']
    probes = []
    if sample_gpus:
        with ThreadPoolExecutor(max_workers=4) as executor:
            probes = list(executor.map(probe, running))
    old = read(OLD / 'STATUS.json', {})
    sources = read(POOL / 'POOL_SOURCES.json', [])
    jobs = [(source, j) for source in sources for j in read(source)]
    status = read(POOL / 'POOL_STATUS.json', {})
    summary = read(POOL / 'RESEARCH_SUMMARY.json', {})
    return dict(utc=utc(), running=len(running), pending=sum(j['state'] == 'PENDING' for j in queue),
                target_running=8, target_pending=12, queue=queue, gpu_samples=probes,
                gpu_note='Instantaneous samples, not sustained utilization; unavailable/idle samples require investigation, never automatic cancellation.',
                reserve=status.get('reserve'), alerts=status.get('alerts', []), storage=old.get('storage'),
                pool_utc=status.get('utc'), research_summary_utc=summary.get('utc'),
                old_monitor_utc=old.get('utc'), series=series(jobs, old))


def pending_acked(state):
    pending = state.get('pending')
    if not pending:
        return True
    if pending['token'] == 'R22_WATCH10M_SETUP_TEST':
        return (HERE / 'SETUP_ACK.json').exists()
    return (HERE / 'acks' / (pending['token'] + '.json')).exists()


def tick():
    if (HERE / 'STOP').exists():
        return
    with (HERE / 'LOCK').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read(HERE / 'STATE.json', {})
        config = read(HERE / 'CONFIG.json')
        now = time.time()
        if now < state.get('next_check_unix', 0):
            return
        snap = snapshot()
        transition(state, snap['series'])
        state.update(last_check_utc=snap['utc'], next_check_unix=now + config['interval_seconds'])
        if pending_acked(state):
            if state.get('pending'):
                state['last_acknowledged'] = state.pop('pending')
        snap['unreviewed_series'] = state['unreviewed_series']
        write(HERE / 'LATEST.json', snap)
        with (HERE / 'HISTORY.jsonl').open('a') as stream:
            stream.write(json.dumps(snap, ensure_ascii=False) + '\n')
        if state.get('pending'):
            state['delivery_status'] = 'WAITING_FOR_EXISTING_MESSAGE_ACK; no duplicate reminder'
            write(HERE / 'STATE.json', state)
            return
        token = str(uuid.uuid4())
        message = (HERE / 'PROMPT.txt').read_text().replace('{TOKEN}', token)
        message += '\n本次检查摘要（需在行动前核对实时状态）：' + json.dumps({
            k: snap[k] for k in ('utc', 'running', 'pending', 'reserve', 'unreviewed_series', 'alerts')
        }, ensure_ascii=False)
        request = dict(token=token, thread_id=config['thread_id'], message=message, utc=utc())
        (HERE / 'deliveries').mkdir(exist_ok=True)
        path = HERE / 'deliveries' / (token + '.json')
        write(path, request)
        # Persist intent before delivery. An ambiguous timeout is not blindly retried.
        state['pending'] = dict(token=token, utc=utc(), request=str(path))
        state['delivery_status'] = 'DELIVERING'
        write(HERE / 'STATE.json', state)
        cmd = ['srun', '--jobid=' + config['host_allocation'], '--overlap', '--ntasks=1',
               '--cpus-per-task=1', '--mem=256M', '--time=00:02:00',
               '/usr/bin/python3', str(HERE / 'deliver.py'), str(path)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
            receipt = dict(utc=utc(), returncode=result.returncode, stdout=result.stdout[-2000:], stderr=result.stderr[-2000:])
            state['delivery_status'] = 'QUEUED_AWAITING_ACK' if result.returncode == 0 else 'DELIVERY_FAILED_REQUIRES_REVIEW'
        except Exception as exc:
            receipt = dict(utc=utc(), error=repr(exc))
            state['delivery_status'] = 'DELIVERY_UNCERTAIN_REQUIRES_REVIEW'
        write(path.with_suffix('.receipt.json'), receipt)
        write(HERE / 'STATE.json', state)


def acknowledge(token):
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', token):
        raise ValueError('invalid token')
    (HERE / 'acks').mkdir(exist_ok=True)
    write(HERE / 'acks' / (token + '.json'), dict(token=token, utc=utc(), received_by_existing_session=True))
    if token == 'R22_WATCH10M_SETUP_TEST':
        write(HERE / 'SETUP_ACK.json', dict(token=token, utc=utc(), received_by_existing_session=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ack')
    parser.add_argument('--reviewed', nargs='+')
    args = parser.parse_args()
    if args.ack:
        acknowledge(args.ack)
    elif args.reviewed:
        with (HERE / 'LOCK').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = read(HERE / 'STATE.json', {})
            sources = read(POOL / 'POOL_SOURCES.json', [])
            groups = series([(source, job) for source in sources for job in read(source)],
                            read(OLD / 'STATUS.json', {}))
            for name in args.reviewed:
                row = groups.get(name, {})
                if not row.get('expected') or row.get('complete') != row['expected']:
                    raise ValueError('Cannot mark incomplete/unknown series reviewed: ' + name)
            state['unreviewed_series'] = sorted(set(state.get('unreviewed_series', [])) - set(args.reviewed))
            # A manual review can precede the next ten-minute discovery tick.
            # Record discovery too, so that tick does not re-add this reviewed series.
            state['seen_finished'] = sorted(set(state.get('seen_finished', [])) | set(args.reviewed))
            history = state.setdefault('reviewed_utc', {})
            history.update({name: utc() for name in args.reviewed})
            write(HERE / 'STATE.json', state)
    else:
        tick()
