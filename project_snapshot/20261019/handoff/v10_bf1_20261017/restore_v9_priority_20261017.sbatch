#!/usr/bin/env bash
#SBATCH --job-name=v9_priority_restore
#SBATCH --partition=CPU
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
set -euo pipefail
export PYTHONUNBUFFERED=1 TZ=UTC
cd /home/infres/yinwang/CS_QMI
python3 - <<'PY'
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

root = Path('handoff/v10_bf1_20261017')
record = {'user_instruction': '不，还是要让V9的作业跑完',
          'control_job': os.environ['SLURM_JOB_ID'], 'actions': [],
          'policy': 'Finish the existing V9 round before releasing V10 GPU jobs.'}

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def save():
    record['updated_utc'] = utc()
    path = root / 'RESTORE_V9_PRIORITY.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)

def command(args):
    result = subprocess.run(args, text=True, capture_output=True, timeout=30, check=True)
    return result.stdout.strip()

def queue():
    text = command(['squeue', '-r', '-u', 'yinwang', '-h', '-o', '%i|%j|%T|%q|%b|%R'])
    return [dict(zip(('id', 'name', 'state', 'qos', 'tres', 'reason'), line.split('|', 5)))
            for line in text.splitlines() if line]

def action(args):
    result = command(args)
    item = {'utc': utc(), 'command': args, 'stdout': result}
    record['actions'].append(item)
    save()
    print(json.dumps(item), flush=True)

def held_ids():
    path = root / 'V9_PENDING_HELD.json'
    return set(json.loads(path.read_text()).get('jobs', [])) if path.exists() else set()

try:
    before = queue()
    record['before'] = before
    record['status'] = 'RESTORING'
    save()
    selected = [j for j in before if j['id'] == '1044420' or j['id'].startswith('1044424_')]
    assert len(selected) == 5, selected
    assert all(j['state'] == 'PENDING' and j['name'] in ('v10bf_gate', 'v10bf_full') for j in selected), selected
    action(['scontrol', 'hold', '1044420'])
    action(['scontrol', 'hold', '1044424'])
    ours = held_ids()
    watcher = [j for j in before if j['id'] == '1044425']
    if watcher:
        assert watcher[0]['name'] == 'v10bf_watch', watcher
        action(['scancel', '1044425'])
    deadline = time.monotonic() + 40
    while any(j['id'] == '1044425' for j in queue()):
        if time.monotonic() > deadline:
            raise RuntimeError('Watcher has not exited; do not race its hold policy')
        time.sleep(2)
    ours |= held_ids()
    for j in queue():
        if j['id'] in ours and j['state'] == 'PENDING' and j['reason'] == '(JobHeldUser)':
            assert j['name'].startswith('v9f_') and j['name'] != 'v9f_feed', j
            action(['scontrol', 'release', j['id']])
    after = queue()
    record['after'] = after
    remaining = [j['id'] for j in after if j['id'] in ours and j['reason'] == '(JobHeldUser)']
    assert not remaining, remaining
    assert not any(j['id'] == '1044425' for j in after)
    v10 = [j for j in after if j['id'] == '1044420' or j['id'].startswith('1044424_')]
    assert len(v10) == 5 and all(j['state'] == 'PENDING' and j['reason'] == '(JobHeldUser)' for j in v10), v10
    (root / 'V9_PENDING_HELD.json').write_text(json.dumps({'utc': utc(), 'jobs': [],
        'released_by_control_job': os.environ['SLURM_JOB_ID']}, indent=2) + '\n')
    record['status'] = 'RESTORED_V9_FIRST_V10_HELD'
    save()
    print(record['status'], flush=True)
except BaseException as exc:
    record['status'] = 'RESTORE_FAILED'
    record['error'] = repr(exc)
    save()
    raise
PY
