"""One-shot transition: retire the old policy naturally and resume owned BF1 jobs."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

B = Path(__file__).resolve().parent
P = B.parents[1]
D = P / 'handoff/v10_parallel8_20261018'
OLD = P / 'handoff/v9_first_20261018'
POLICY = 'v10_fill_and_validate_20261019'

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok, why):
    if not ok: raise RuntimeError(why)
def write(p, x):
    tmp = p.with_name('.' + p.name + '.tmp')
    tmp.write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(p)
def utc(): return time.strftime('%FT%TZ', time.gmtime())
def current():
    require(read(P / 'handoff/ACTIVE_SCHEDULING_POLICY.json')['policy_id'] == POLICY,
            'Transition superseded; stop without further actions')
def command(args):
    r = subprocess.run(args, capture_output=True, text=True, timeout=45)
    require(r.returncode == 0, repr(args) + ': ' + r.stderr)
    return r.stdout
def queue():
    return [dict(zip(('id', 'name', 'state', 'qos', 'gres', 'reason'), s.split('|', 5)))
            for s in command(['squeue', '-u', 'yinwang', '-r', '-h', '-o', '%i|%j|%T|%q|%b|%R']).splitlines()]

def main():
    require('SLURM_JOB_ID' in os.environ, 'Slurm controller job required')
    require(sha(B / 'FREEZE.json') == os.environ['V10_TRANSITION_FREEZE'], 'Freeze mismatch')
    for n, h in read(B / 'FREEZE.json')['files'].items():
        require(sha(B / n) == h, 'Changed transition source: ' + n)
    with (B / 'ACTIVATE.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (B / 'ACTIVATED.json').exists(), 'Already activated')
        current()
        # The old policy sees the new selector and exits itself. Do not cancel it.
        for _ in range(20):
            current()
            rows = queue()
            if not any(r['id'] == '1047062' for r in rows): break
            time.sleep(15)
        else: raise RuntimeError('Old policy did not exit; leave pause and hold intact')
        require(read(OLD / 'STATUS.json')['status'] == 'SUPERSEDED_NO_RELEASE', 'Old policy exit unverified')
        require(any(r['id'] == '1046189' and r['state'] == 'RUNNING' for r in rows), 'Existing BF1 dispatcher not running')
        pause = D / 'PAUSE_SUBMISSIONS'
        require(read(pause)['policy_id'] == 'v9_first_20261018_2143', 'Pause ownership changed')
        ledger = read(D / 'LEDGER.json')
        expected = {'final_2_1001', 'final_2_1002', 'final_2_1003', 'final_3_1003'}
        require({u['key'] for u in ledger['units'] if u['status'] != 'COMPLETE_VERIFIED'} == expected,
                'BF1 remaining matrix changed; review before resume')
        require(set(read(OLD / 'LEDGER.json')['held_by_this_policy']) == {'1046917'}, 'Held-job ownership changed')
        os.environ['V10_PARALLEL8_FREEZE'] = '00f43e15e21a7ee03d934c3518b4913748ea93cda577d6fce676988bf0778ce3'
        spec = importlib.util.spec_from_file_location('existing_dispatch', D / 'dispatch.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.verify_bundle()
        module.verify_gate()
        for family in module.FAMILIES: module.check_selection(family)
        row = next(r for r in rows if r['id'] == '1046917')
        require(row['state'] == 'PENDING' and row['reason'] == '(JobHeldUser)', 'Owned hold state changed')
        write(B / 'BEFORE_RELEASE.json', {'utc': utc(), 'queue': rows, 'ledger': ledger})
        current()
        command(['scontrol', 'release', '1046917'])
        write(B / 'RELEASED.json', {'utc': utc(), 'job': '1046917', 'owner_policy': 'v9_first_20261018_2143'})
        current()
        require(read(pause)['policy_id'] == 'v9_first_20261018_2143', 'Pause changed during release')
        pause.replace(B / 'RETIRED_PAUSE_SUBMISSIONS.json')
        write(B / 'ACTIVATED.json', {'utc': utc(), 'status': 'EXISTING_BF1_RESUMED_NEW_VALIDATION_AUTHORIZED',
              'policy_id': POLICY, 'controller_job': os.environ['SLURM_JOB_ID'], 'released': ['1046917'],
              'dispatcher': '1046189', 'cancelled_jobs': [], 'queue': queue()})

if __name__ == '__main__':
    try: main()
    except Exception as exc:
        write(B / 'ERROR.json', {'utc': utc(), 'error': repr(exc), 'traceback': traceback.format_exc()})
        raise
