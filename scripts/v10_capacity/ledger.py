"""Capacity-extension ledger view and reconciliation (no Torch, login-node safe).

  python ledger.py status            per-unit decision / jobs / exit codes / COMPLETE, committed budget vs the ceiling
  python ledger.py reconcile JOB     a job that left the queue without an exit receipt (e.g. node failure): writes a conservative
                                     receipt charging its full reservation, so its unit can be resubmitted under the resume rule
"""
import subprocess
import sys

import capacity_common as C


def in_queue(job):
    r = subprocess.run(['squeue', '-h', '-j', str(job), '-o', '%i'], capture_output=True, text=True, timeout=60)
    C.require(r.returncode == 0 or 'Invalid job id' in r.stderr, 'squeue failed: ' + r.stderr)
    return bool(r.stdout.strip())


def status():
    adm = C.lines(C.BUNDLE / 'ADMISSIONS.jsonl')
    exits = {p.stem: C.read(p) for p in (C.BUNDLE / 'exits').glob('*.json')} if (C.BUNDLE / 'exits').exists() else {}
    committed, _ = C.committed_seconds()
    print(f'committed {committed / 3600:.3f} of {C.CEILING_SECONDS / 3600:.0f} allocated L40S GPU-h')
    for u in C.planned_units():
        mine = [a for a in adm if a['unit'] == u['key']]
        jobs = ' '.join(f"{a['job']}:{a['decision']}:exit={exits.get(a['job'], {}).get('exit_code', '-')}" for a in mine)
        done = 'COMPLETE' if (C.OUT / u['key'] / 'COMPLETE.json').exists() else ''
        budget = 'INCOMPLETE_BUDGET_CEILING' if (C.BUNDLE / 'budget' / f"{u['key']}.json").exists() else ''
        print(f"{u['order']:2d} {u['key']:28s} {u['job_name']:18s} {done:8s} {budget} {jobs}")


def reconcile(job):
    C.require(not in_queue(job), 'Job is still in the queue')
    C.require(not (C.BUNDLE / 'exits' / f'{job}.json').exists(), 'Job already has a receipt')
    adm = [a for a in C.lines(C.BUNDLE / 'ADMISSIONS.jsonl') if a['job'] == job]
    C.require(len(adm) == 1, 'Job has no unique admission record')
    a = adm[0]
    seconds = a['reserved_seconds'] if a['decision'] == 'ADMITTED' else C.NON_ADMITTED_PROVISIONAL_SECONDS
    C.write(C.BUNDLE / 'exits' / f'{job}.json', {'job': job, 'unit': a['unit'], 'exit_code': None, 'reconciled': True,
            'allocated_seconds': seconds, 'allocated_seconds_basis': 'reconciled_full_reservation_no_receipt', 'utc': C.utc(),
            'node': None})
    print('reconciled', job, a['unit'], seconds)


if __name__ == '__main__':
    if sys.argv[1:] == ['status']:
        status()
    elif len(sys.argv) == 3 and sys.argv[1] == 'reconcile':
        reconcile(sys.argv[2])
    else:
        raise SystemExit(__doc__)
