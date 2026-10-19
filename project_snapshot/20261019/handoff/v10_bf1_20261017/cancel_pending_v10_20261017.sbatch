#!/usr/bin/env bash
#SBATCH --job-name=v10_cancel_pending
#SBATCH --partition=CPU
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
cd /home/infres/yinwang/CS_QMI
python3 - <<'PY'
from pathlib import Path
import datetime, hashlib, json, os, subprocess, time
root = Path('handoff/v10_bf1_20261017')
record = {'user_context': 'Cancel held V10 queue entries until V9 closeout; retain frozen code and resubmission parameters.',
          'control_job': os.environ['SLURM_JOB_ID'], 'actions': []}
def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)
def save():
    record['updated_utc'] = utc()
    write(root / 'CANCELLED_PENDING.json', record)
def command(args):
    return subprocess.run(args, text=True, capture_output=True, timeout=30, check=True).stdout.strip()
def queue():
    output = command(['squeue', '-r', '-u', 'yinwang', '-h', '-o', '%i|%j|%T|%q|%b|%R'])
    return [dict(zip(('id', 'name', 'state', 'qos', 'tres', 'reason'), line.split('|', 5)))
            for line in output.splitlines() if line]
def targets(jobs):
    return [j for j in jobs if j['id'] == '1044420' or j['id'].startswith('1044424_')]
try:
    launch = json.loads((root / 'LAUNCH.json').read_text())
    actual = hashlib.sha256((root / 'FROZEN.json').read_bytes()).hexdigest()
    assert actual == launch['freeze_sha256']
    before = queue()
    selected = targets(before)
    assert len(selected) == 5, selected
    assert all(j['state'] == 'PENDING' and j['reason'] == '(JobHeldUser)' and
               j['name'] in ('v10bf_gate', 'v10bf_full') for j in selected), selected
    assert not any(j['id'] == '1044425' for j in before), 'Old priority watcher unexpectedly live'
    write(root / 'LAUNCH_BEFORE_CANCELLATION.json', launch)
    record.update(status='CANCELLING_PENDING_ONLY', before=selected, freeze_sha256=actual)
    save()
    for jid in ('1044424', '1044420'):
        output = command(['scancel', jid])
        record['actions'].append({'utc': utc(), 'command': ['scancel', jid], 'stdout': output})
        save()
    deadline = time.monotonic() + 40
    after = queue()
    while targets(after):
        if time.monotonic() > deadline:
            raise RuntimeError('Cancelled V10 entries still visible; inspect state')
        time.sleep(2)
        after = queue()
    record.update(status='CANCELLED_BEFORE_START', after_v10_jobs=targets(after),
                  v9_jobs_after=[j for j in after if j['name'].startswith('v9f_')],
                  next_action='Resubmit the same frozen gate and array only after V9 closeout; use a new afterok gate ID.')
    save()
    print(record['status'], flush=True)
except BaseException as exc:
    record.update(status='CANCELLATION_ERROR', error=repr(exc))
    save()
    raise
PY
