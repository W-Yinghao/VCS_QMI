#!/usr/bin/env bash
#SBATCH --job-name=v9_replace_feeder
#SBATCH --partition=CPU
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
cd /home/infres/yinwang/CS_QMI/handoff/v9_dispatch_pending8_20261017
python3 - <<'PY'
import fcntl, os, subprocess, time
from feed_pending8 import BUNDLE, ROOT, queue, read, utc, verify_dispatch, write
cfg = verify_dispatch()
record = {'utc': utc(), 'control_job': os.environ['SLURM_JOB_ID'], 'old_feeder': cfg['old_feeder'],
          'status': 'PREPARED', 'scientific_freeze_unchanged': cfg['experiment_freeze_sha256']}
write(BUNDLE / 'TRANSITION.json', record)
try:
    before = queue()
    old = [j for j in before if j['id'] == cfg['old_feeder']]
    if old:
        assert old[0]['name'] == 'v9f_feed' and old[0]['tres'] in ('N/A', 'gres/gpu:0'), old
        subprocess.run(['scancel', cfg['old_feeder']], check=True, timeout=30)
    deadline = time.monotonic() + 45
    while any(j['id'] == cfg['old_feeder'] for j in queue()):
        if time.monotonic() > deadline:
            raise RuntimeError('Old feeder still present; replacement not submitted')
        time.sleep(2)
    assert not any(j['name'] == 'v9f_feed' for j in queue()), 'Another feeder is already live'
    with (ROOT / 'FEED.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(BUNDLE / 'PREVIOUS_FEED_STATUS.json', read(ROOT / 'FEED_STATUS.json'))
        write(BUNDLE / 'LAUNCH_AT_TRANSITION.json', read(ROOT / 'LAUNCH.json'))
    record.update(status='OLD_FEEDER_STOPPED', old_feeder_stopped_utc=utc())
    write(BUNDLE / 'TRANSITION.json', record)
    p = subprocess.run(['sbatch', '--parsable', str(BUNDLE / 'feed_pending8.sbatch')],
                       text=True, capture_output=True, timeout=45, check=True)
    jid = p.stdout.strip().split(';')[0]
    if not jid.isdigit():
        raise RuntimeError('Uncertain replacement submission: ' + p.stdout)
    record.update(status='REPLACEMENT_SUBMITTED', new_feeder=jid, utc=utc())
    write(BUNDLE / 'TRANSITION.json', record)
    print('NEW_FEEDER', jid, flush=True)
except BaseException as exc:
    record.update(status='TRANSITION_ERROR', error=repr(exc), utc=utc())
    write(BUNDLE / 'TRANSITION.json', record)
    raise
PY
