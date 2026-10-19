"""Slurm-only dispatch: unchanged sequential CAL, independent full FINAL seeds."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

BUNDLE = Path(__file__).resolve().parent
PROJECT = BUNDLE.parents[1]
V10 = PROJECT / 'handoff/v10_bf1_20261017'
SCIENCE_FREEZE = 'a03e88277ba61159d4cd70255d72bf13515b8cac9843465329f6f9bcb15f58a2'
SOURCE_COMMIT = 'b8f59ea825dc7a2064457d47a628bef09ca4e185'
GATE = '1045975'
OLD_ARRAY = '1045976'
sys.path.insert(0, str(V10 / 'source/scripts/v10_bf1'))
from common import FAMILIES, append, candidates, freeze, identity, read, selection_key, sha, utc, verify_complete, write


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def command(args):
    result = subprocess.run(args, text=True, capture_output=True, timeout=45)
    if result.returncode:
        raise RuntimeError(f'{args}: rc={result.returncode}: {result.stderr}')
    return result.stdout


def queue():
    output = command(['squeue', '-u', 'yinwang', '-r', '-h', '-o', '%i|%j|%T|%q|%b|%E|%R'])
    rows = []
    for line in output.splitlines():
        fields = line.split('|')
        require(len(fields) == 7, 'Malformed expanded queue row')
        rows.append(dict(zip(('id', 'name', 'state', 'qos', 'gres', 'dependency', 'reason'), fields)))
    return rows


def units():
    out = [dict(key=f'cal_{i}', family=f, phase='cal', seed=None,
                name=f'v10p8_c{i}', status='PLANNED') for i, f in enumerate(FAMILIES)]
    # Seed-major order balances family coverage when the normal submit limit binds.
    out += [dict(key=f'final_{i}_{seed}', family=f, phase='final', seed=seed,
                 name=f'v10p8_f{i}_{seed}', status='PLANNED')
            for seed in (1001, 1002, 1003) for i, f in enumerate(FAMILIES)]
    return out


def eligible(unit, ledger):
    if unit['status'] != 'PLANNED':
        return False
    same = [u for u in ledger['units'] if u['family'] == unit['family']]
    if any(u['status'] == 'FAILED_REVIEW_REQUIRED' for u in same):
        return False
    return unit['phase'] == 'cal' or any(u['phase'] == 'cal' and u['status'] == 'COMPLETE_VERIFIED' for u in same)


def capacity(rows):
    own = [r for r in rows if r['name'].startswith('v10p8_') or r['id'] == GATE]
    # The CPU dispatcher is named dispatch_bf1 and does not consume a GPU slot.
    return (sum(r['qos'] == 'normal' for r in rows) < 29
            and len(own) < 16
            and sum(r['state'] == 'PENDING' for r in own) < 8)


def check_selection(family):
    path = V10 / 'results/CAL' / family / 'SELECTION.json'
    selection = read(path)
    require(selection['status'] == 'SELECTED' and selection['family'] == family
            and selection['freeze_sha256'] == SCIENCE_FREEZE, 'Selection identity mismatch')
    records = selection['completed_candidates']
    require(bool(records), 'Empty CAL selection')
    expected = candidates(family)
    require([r['candidate'] for r in records] == expected[:len(records)], 'CAL order changed')
    require(selection['planned_candidates'] == len(expected), 'CAL grid changed')
    require(selection['budget_exhausted'] or len(records) == len(expected), 'Missing CAL candidates')
    for record in records:
        directory = (V10 / record['unit']).resolve()
        require(directory.parent == (V10 / 'results/CAL' / family).resolve(), 'Invalid CAL artifact path')
        require(sha(directory / 'COMPLETE.json') == record['complete_sha256'], 'CAL dependency changed')
        candidate = record['candidate']
        complete = verify_complete(directory, identity(family, 1017, candidate['lr'], candidate['cap'], 'CAL'))
        require(complete is not None and {**complete['selected'], 'order': candidate['order']} == record['choice'],
                'CAL candidate incomplete or selection mismatch')
    require(min(records, key=lambda r: selection_key(r['choice'])) == selection['selected'], 'CAL argmax mismatch')
    return selection, sha(path)


def verify_unit(unit):
    selection, selection_sha = check_selection(unit['family'])
    if unit['phase'] == 'final':
        candidate = selection['selected']['candidate']
        expected = identity(unit['family'], unit['seed'], candidate['lr'], candidate['cap'], 'FINAL_DEV', selection_sha)
        directory = V10 / 'results/FINAL_DEV' / unit['family'] / f"seed_{unit['seed']}"
        require(verify_complete(directory, expected) is not None, 'Missing full FINAL COMPLETE')
    return {'utc': utc(), 'selection_sha256': selection_sha, 'full_artifact_hashes_verified': True}


def verify_bundle():
    manifest = read(BUNDLE / 'DISPATCH_FREEZE.json')
    require(sha(BUNDLE / 'DISPATCH_FREEZE.json') == os.environ['V10_PARALLEL8_FREEZE'], 'Dispatch manifest changed')
    for name, digest in manifest['files'].items():
        require(sha(BUNDLE / name) == digest, 'Dispatch source changed: ' + name)
    require(sha(V10 / 'FROZEN.json') == SCIENCE_FREEZE, 'Scientific freeze changed')
    require(read(V10 / 'FROZEN.json')['source_commit'] == SOURCE_COMMIT, 'Scientific commit changed')
    frozen = freeze()
    require(sha(V10 / 'PROTOCOL.json') == read(BUNDLE / 'SCHEDULING_AMENDMENT.json')['base_protocol_sha256'],
            'Protocol changed')
    return {'utc': utc(), 'source_files': len(frozen['source_files']), 'run_inputs': len(frozen['run_inputs']),
            'freeze_sha256': SCIENCE_FREEZE, 'status': 'PASS'}


def controller_action(args):
    write(BUNDLE / 'CONTROL_INTENT.json', {'utc': utc(), 'args': args})
    output = command(args)
    append(BUNDLE / 'CONTROL_ACTIONS.jsonl', {'utc': utc(), 'args': args, 'stdout': output})
    return output


def activate():
    require(not (BUNDLE / 'CONTROL_INTENT.json').exists(), 'Prior activation action: inspect before retry')
    require(not (BUNDLE / 'LEDGER.json').exists(), 'Already activated; do not duplicate dispatcher')
    rows = queue()
    by_id = {r['id']: r for r in rows}
    launch = read(V10 / 'LAUNCH.json')
    require(launch['gate_job'] == GATE and launch['array_job'] == OLD_ARRAY, 'Unexpected active V10 jobs')
    require(by_id[GATE]['state'] == 'PENDING' and by_id[GATE]['dependency'] == '(null)', 'GPU gate state changed')
    require(not any(r['name'].startswith('v10p8_') for r in rows), 'New V10 jobs already exist')
    for i in range(4):
        row = by_id[f'{OLD_ARRAY}_{i}']
        require(row['state'] == 'PENDING' and row['dependency'].startswith('afterok:' + GATE), 'Old array started or changed')
    require(not (V10 / 'results/CAL').exists(), 'CAL artifacts already exist: reconcile before split')
    write(BUNDLE / 'BEFORE.json', {'utc': utc(), 'queue': rows, 'launch': launch})
    controller_action(['scontrol', 'hold', OLD_ARRAY])
    held = command(['scontrol', 'show', 'job', '-o', OLD_ARRAY])
    require('JobState=PENDING' in held and 'Reason=JobHeldUser' in held, 'Old array not safely held')
    controller_action(['scancel', OLD_ARRAY])
    after = command(['scontrol', 'show', 'job', '-o', OLD_ARRAY])
    require('JobState=CANCELLED' in after and 'RunTime=00:00:00' in after, 'Old array cancellation not verified')
    require(not any(r['id'].startswith(OLD_ARRAY + '_') for r in queue()), 'Old array still active')
    ledger = {'utc': utc(), 'status': 'ACTIVE', 'feeder_job': os.environ['SLURM_JOB_ID'],
              'gate_job': GATE, 'retired_array': OLD_ARRAY, 'units': units()}
    write(BUNDLE / 'LEDGER.json', ledger)
    write(BUNDLE / 'ACTIVATION.json', {'utc': utc(), 'status': 'PARALLEL8_ACTIVE',
          'dispatcher_job': ledger['feeder_job'], 'retired_array': OLD_ARRAY,
          'retired_array_controller': after, 'gate_job': GATE, 'gate_dependency': None,
          'max_gpu_concurrency': 8, 'enforcement': 'shared normal QoS GPU limit',
          'cal_jobs': 4, 'final_jobs': 12, 'formal_trainings': 30, 'cal_candidates': 18,
          'v9_unchanged': True, 'v9_closeout_blocks_v10': False,
          'science_freeze': SCIENCE_FREEZE, 'source_commit': SOURCE_COMMIT,
          'ledger': str(BUNDLE / 'LEDGER.json'), 'amendment': str(BUNDLE / 'SCHEDULING_AMENDMENT.json')})
    launch.update(array_job=None, array=None, array_job_status='REPLACED_BEFORE_START',
                  retired_array_job=OLD_ARRAY, max_gpu_concurrency=8,
                  status='PHASE_SPLIT_PARALLEL8_ACTIVE', submission_status='FEEDER_MANAGED',
                  scheduling_updated_utc=utc(), scheduling_decision=str(BUNDLE / 'ACTIVATION.json'),
                  scheduling_amendment=str(BUNDLE / 'SCHEDULING_AMENDMENT.json'),
                  dispatcher_job=ledger['feeder_job'], dispatch_ledger=str(BUNDLE / 'LEDGER.json'),
                  dispatch_status=str(BUNDLE / 'STATUS.json'),
                  dependency='CAL: afterok:1045975; FINAL: verified successful family CAL in dispatcher',
                  final_submission_rule='Only after CAL exit=0, END/COMPLETE ledger and all selection artifact hashes pass.',
                  active_gpu_jobs=[GATE])
    launch['previous_cancelled_jobs'] = list(dict.fromkeys(launch['previous_cancelled_jobs'] + [OLD_ARRAY]))
    write(V10 / 'LAUNCH.json', launch)
    return ledger


def submit(unit, ledger, gate_live):
    require(unit['status'] == 'PLANNED' and not unit.get('job'), 'Duplicate unit submission')
    if not gate_live:
        verify_gate()
    args = ['sbatch', '--parsable', '--job-name=' + unit['name'], '--time=23:59:00',
            '--export=ALL,V10_BF1_EXPECT_FREEZE=' + SCIENCE_FREEZE + ',V10_BF1_SOURCE_COMMIT=' + SOURCE_COMMIT]
    if unit['phase'] == 'cal' and gate_live:
        args.append('--dependency=afterok:' + GATE)
    args += [str(V10 / 'source/slurm/v10_bf1/gpu.sbatch'), unit['phase'], unit['family']]
    if unit['phase'] == 'final':
        args += ['--seed', str(unit['seed'])]
    # Persist intent before sbatch; ambiguous returns must never be blindly retried.
    unit.update(status='SUBMIT_INTENT', command=args, intent_utc=utc())
    write(BUNDLE / 'LEDGER.json', ledger)
    result = subprocess.run(args, text=True, capture_output=True, timeout=45)
    append(BUNDLE / 'SUBMISSIONS.jsonl', {'utc': utc(), 'key': unit['key'], 'args': args,
          'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
    if result.returncode != 0:
        if not result.stdout.strip() and ('QOSMaxSubmitJobPerUserLimit' in result.stderr or
                                        'Job violates accounting/QOS policy' in result.stderr):
            unit.update(status='PLANNED', last_deferral=result.stderr)
            write(BUNDLE / 'LEDGER.json', ledger)
            return False
        raise RuntimeError('Submission failed or ambiguous: inspect SUBMISSIONS and queue before retry')
    job = result.stdout.strip().split(';')[0]
    require(re.fullmatch(r'\d+', job), 'Ambiguous job ID')
    unit.update(status='SUBMITTED', job=job, submitted_utc=utc())
    write(BUNDLE / 'LEDGER.json', ledger)
    launch = read(V10 / 'LAUNCH.json')
    launch['submitted_phase_jobs'] = {u['key']: u['job'] for u in ledger['units'] if u.get('job')}
    launch['active_gpu_jobs'] = [GATE] + [u['job'] for u in ledger['units'] if u['status'] == 'SUBMITTED']
    write(V10 / 'LAUNCH.json', launch)
    print('SUBMITTED', unit['key'], job, flush=True)
    return True


def end_record(job, family, phase, status):
    path = V10 / 'COST_LEDGER.jsonl'
    if not path.exists():
        return False
    with path.open() as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        records = [json.loads(line) for line in stream if line.strip()]
    return any(r.get('event') == 'END' and str(r.get('job_id')) == job and r.get('family') == family
               and r.get('phase') == phase and r.get('status') == status
               for r in records)


def verify_gate():
    gate = read(V10 / 'GPU_GATE.json')
    exit_info = read(V10 / 'exits' / f'{GATE}.json')
    require(gate['status'] == 'PASS' and gate['freeze_sha256'] == SCIENCE_FREEZE
            and str(gate['job_id']) == GATE and exit_info['exit_code'] == 0
            and end_record(GATE, 'SHARED_EXECUTION_VALIDATION', 'VALIDATION', 'PASS'), 'GPU gate not verified PASS')


def observe(ledger, rows):
    by_id = {r['id']: r for r in rows}
    if GATE not in by_id:
        verify_gate()
    for unit in ledger['units']:
        if unit['status'] != 'SUBMITTED':
            continue
        job = unit['job']
        if job in by_id:
            unit['queue_state'] = by_id[job]['state']
            unit['queue_reason'] = by_id[job]['reason']
            continue
        unit.setdefault('first_absent_unix', time.time())
        path = V10 / 'exits' / f'{job}.json'
        if not path.exists() and time.time() - unit['first_absent_unix'] < 300:
            continue  # Shared-filesystem finalization grace, never evidence of success.
        try:
            exit_info = read(path)
            phase = 'CAL' if unit['phase'] == 'cal' else 'FINAL_DEV'
            require(str(exit_info['job_id']) == job and exit_info['exit_code'] == 0, 'Nonzero or mismatched batch exit')
            require(end_record(job, unit['family'], phase, 'COMPLETE'), 'Missing COMPLETE allocation END')
            unit['verification'] = verify_unit(unit)
            unit['status'] = 'COMPLETE_VERIFIED'
        except Exception as error:
            unit.update(status='FAILED_REVIEW_REQUIRED', error=repr(error), failure_utc=utc())
    write(BUNDLE / 'LEDGER.json', ledger)


def loop(ledger):
    while True:
        verify_bundle()
        rows = queue()
        observe(ledger, rows)
        for unit in ledger['units']:
            if not eligible(unit, ledger):
                continue
            rows = queue()
            if not capacity(rows):
                break
            if (BUNDLE / 'PAUSE_SUBMISSIONS').exists():
                break
            if not submit(unit, ledger, any(r['id'] == GATE for r in rows)):
                break
        rows = queue()
        completed = sum(u['status'] == 'COMPLETE_VERIFIED' for u in ledger['units'])
        failed = [u['key'] for u in ledger['units'] if u['status'] == 'FAILED_REVIEW_REQUIRED']
        own = [r for r in rows if r['name'].startswith('v10p8_') or r['id'] == GATE]
        status = 'COMPLETE_VERIFIED' if completed == 16 else ('REVIEW_REQUIRED' if failed else 'ACTIVE')
        write(BUNDLE / 'STATUS.json', {'utc': utc(), 'status': status, 'feeder_job': ledger['feeder_job'],
              'completed_phase_jobs': completed, 'total_phase_jobs': 16, 'failed': failed,
              'submitted_phase_jobs': sum(bool(u.get('job')) for u in ledger['units']),
              'pending': sum(r['state'] == 'PENDING' for r in own),
              'running': sum(r['state'] == 'RUNNING' for r in own), 'queue': own,
              'max_gpu_concurrency': 8, 'pending_target': 8, 'normal_submit_total': sum(r['qos'] == 'normal' for r in rows),
              'waiting_final_rule': 'Submit only after successful same-family CAL and verified artifacts.',
              'pause_submissions': (BUNDLE / 'PAUSE_SUBMISSIONS').exists()})
        launch = read(V10 / 'LAUNCH.json')
        launch.update(dispatch_status_value=status, dispatch_observed_utc=utc(), active_gpu_jobs=[r['id'] for r in own])
        write(V10 / 'LAUNCH.json', launch)
        if completed == 16:
            ledger['status'] = status
            write(BUNDLE / 'LEDGER.json', ledger)
            return
        if failed and not any(u['status'] == 'SUBMITTED' or eligible(u, ledger) for u in ledger['units']):
            raise RuntimeError('Affected family requires review; all other runnable work has finished')
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activate', action='store_true')
    args = parser.parse_args()
    require(bool(os.environ.get('SLURM_JOB_ID')), 'Dispatch controller runs only through Slurm')
    with (BUNDLE / 'DISPATCH.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(BUNDLE / 'PREFLIGHT.json', verify_bundle())
        ledger = activate() if args.activate else read(BUNDLE / 'LEDGER.json')
        require(not any(u['status'] == 'SUBMIT_INTENT' for u in ledger['units']), 'Ambiguous prior submission; manual reconcile required')
        loop(ledger)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        error = {'utc': utc(), 'status': 'FAIL_REVIEW_REQUIRED', 'traceback': traceback.format_exc()}
        write(BUNDLE / 'ERROR.json', error)
        write(BUNDLE / 'STATUS.json', error)
        raise
