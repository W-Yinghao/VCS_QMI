"""Maintain eight pending frozen V9 groups using the existing ledger and lock.

This dispatcher lives outside the accepted scientific bundle. Neither the
training implementation nor the original FROZEN.json is edited by this change.
"""
from __future__ import annotations
import fcntl
import getpass
import os
from pathlib import Path
import subprocess
import sys
import time

BUNDLE = Path(__file__).resolve().parent
EXPERIMENT = BUNDLE.parent / 'v9_full_20261017'
sys.path.insert(0, str(EXPERIMENT))
from common import ROOT, frozen, read, sha, write
from feed import gpu_count
from summarize import summarize


def utc():
    return time.strftime('%FT%TZ', time.gmtime())


def queue():
    p = subprocess.run(['squeue', '-r', '-u', getpass.getuser(), '-h', '-o', '%i|%j|%T|%q|%b|%R'],
                       capture_output=True, text=True, timeout=30, check=True)
    jobs = []
    for line in p.stdout.splitlines():
        if not line.strip():
            continue
        fields = line.split('|', 5)
        if len(fields) != 6:
            raise ValueError('Malformed scheduler row: ' + line)
        jobs.append(dict(zip(('id', 'name', 'state', 'qos', 'tres', 'reason'), fields)))
    return jobs


def queue_summary(jobs, groups):
    names = {'v9f_' + g['id'] for g in groups}
    normal = [j for j in jobs if j['qos'] == 'normal']
    own = [j for j in normal if j['name'] in names]
    return {
        'queue_total': len(jobs), 'normal_queue_total': len(normal),
        'normal_gpu_running': sum(gpu_count(j) for j in normal if j['state'] == 'RUNNING'),
        'own_live': len(own), 'own_running': sum(j['state'] == 'RUNNING' for j in own),
        'own_pending': sum(j['state'] == 'PENDING' for j in own),
        'own_pending_held': [j['id'] for j in own if j['reason'] in ('(JobHeldUser)', '(JobHeldAdmin)')],
    }


def has_capacity(status, cfg):
    return (status['normal_queue_total'] < cfg['max_normal_jobs']
            and status['own_live'] < cfg['max_own_live']
            and status['own_pending'] < cfg['target_pending'])


def verify_dispatch():
    manifest = read(BUNDLE / 'DISPATCH_FREEZE.json')
    for name, digest in manifest['files'].items():
        if sha(BUNDLE / name) != digest:
            raise ValueError('Dispatcher changed: ' + name)
    cfg = read(BUNDLE / 'CONFIG.json')
    if Path(cfg['experiment_root']) != ROOT or sha(ROOT / 'FROZEN.json') != cfg['experiment_freeze_sha256']:
        raise ValueError('Original experiment identity mismatch')
    frozen()
    gate = read(ROOT / 'VALIDATION.json')
    if gate['status'] != 'PASS' or gate['full_units'] != cfg['matrix_full_units']:
        raise ValueError('Original execution validation missing')
    return cfg


def main():
    with (ROOT / 'FEED.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg = verify_dispatch()
        groups = read(ROOT / 'GROUPS.json')
        assert len(groups) == cfg['matrix_groups']
        assert sum(len(g['units']) for g in groups) == cfg['matrix_full_units']
        lp = ROOT / 'LAUNCH.json'
        ledger = read(lp) if lp.exists() else {}
        started = time.monotonic()
        missing_since = {}
        while time.monotonic() - started < cfg['max_hours'] * 3600:
            try:
                jobs = queue()
            except (subprocess.SubprocessError, OSError) as exc:
                write(ROOT / 'FEED_STATUS.json', {'utc': utc(), 'status': 'QUEUE_UNAVAILABLE_NO_SUBMISSION',
                      'feeder_job': os.environ['SLURM_JOB_ID'], 'error': repr(exc)})
                time.sleep(cfg['poll_seconds'])
                continue
            ids = {j['id'] for j in jobs}
            for g in groups:
                matches = [j for j in jobs if j['name'] == 'v9f_' + g['id']]
                if len(matches) > 1:
                    raise RuntimeError('Duplicate live group: ' + g['id'])
                if matches:
                    rec = ledger.get(g['id'])
                    if rec is None:
                        ledger[g['id']] = {'job': matches[0]['id'], 'attempt': 1, 'recovered': True, 'utc': utc()}
                        write(lp, ledger)
                    elif rec['job'] != matches[0]['id']:
                        raise RuntimeError('Live job disagrees with ledger: ' + g['id'])
            candidates, failed = [], []
            for g in groups:
                gd = ROOT / 'results' / g['id']
                rec = ledger.get(g['id'])
                if (gd / 'DONE.json').exists():
                    continue
                if rec is None:
                    candidates.append(g)
                    continue
                if rec['job'] in ids:
                    missing_since.pop(g['id'], None)
                    continue
                first_missing = missing_since.setdefault(g['id'], time.monotonic())
                if time.monotonic() - first_missing < cfg['exit_record_grace_seconds']:
                    continue
                ep = gd / 'SCHED_EXIT.json'
                exit_rec = read(ep) if ep.exists() else {}
                # Preserve the accepted single unchanged infrastructure-resume policy.
                if (exit_rec.get('job_id') == rec['job'] and exit_rec.get('exit_code') in (137, 143)
                        and rec.get('attempt', 1) < 2):
                    candidates.append(g)
                else:
                    failed.append(g['id'])
            report = summarize()
            status = {'utc': utc(), **queue_summary(jobs, groups), 'groups_submitted': len(ledger),
                      'full_complete': sum(r['status'] == 'COMPLETE' for r in report['units']),
                      'failed_groups': failed, 'candidates': len(candidates),
                      'groups_not_submitted': sum(g['id'] not in ledger for g in groups),
                      'feeder_job': os.environ['SLURM_JOB_ID'], 'target_pending': cfg['target_pending'],
                      'max_normal_jobs': cfg['max_normal_jobs'], 'max_own_live': cfg['max_own_live'],
                      'poll_seconds': cfg['poll_seconds'], 'array_tasks_counted_separately': True,
                      'dispatch_record': str(BUNDLE / 'TRANSITION.json')}
            status['submission_capacity_available'] = has_capacity(status, cfg)
            write(ROOT / 'FEED_STATUS.json', status)
            if all((ROOT / 'results' / g['id'] / 'DONE.json').exists() or g['id'] in failed for g in groups):
                write(ROOT / 'FEED_FINISHED.json', status)
                return 1 if failed else 0
            if candidates and has_capacity(status, cfg):
                g = candidates[0]
                previous = ledger.get(g['id'], {})
                intent = {'group': g['id'], 'attempt': previous.get('attempt', 1) + 1 if previous else 1,
                          'utc': utc(), 'freeze_sha256': cfg['experiment_freeze_sha256'],
                          'dispatcher': str(BUNDLE / 'DISPATCH_FREEZE.json')}
                write(ROOT / 'SUBMIT_INTENT.json', intent)
                command = ['sbatch', '--parsable', '--qos=normal', '--job-name=v9f_' + g['id'],
                           str(ROOT / 'full.sbatch'), g['id']]
                try:
                    p = subprocess.run(command, capture_output=True, text=True, timeout=45)
                except (subprocess.SubprocessError, OSError) as exc:
                    write(ROOT / 'SUBMIT_UNCERTAIN.json', {**intent, 'error': repr(exc)})
                    raise
                if p.returncode:
                    write(ROOT / 'LAST_SUBMIT_REJECTED.json', {**intent, 'stdout': p.stdout, 'stderr': p.stderr})
                    time.sleep(cfg['poll_seconds'])
                    continue
                job = p.stdout.strip().split(';')[0]
                if not job.isdigit():
                    write(ROOT / 'SUBMIT_UNCERTAIN.json', {**intent, 'stdout': p.stdout, 'stderr': p.stderr})
                    raise RuntimeError('Ambiguous submission; inspect before retry')
                ledger[g['id']] = {**intent, 'job': job}
                write(lp, ledger)
                print('SUBMITTED', g['id'], job, flush=True)
                continue
            time.sleep(cfg['poll_seconds'])
        raise RuntimeError('Dispatcher exceeded 72 hours; inspect remaining groups')


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        write(BUNDLE / 'LOOP_ERROR.json', {'utc': utc(), 'job': os.environ.get('SLURM_JOB_ID'), 'error': repr(exc)})
        raise
