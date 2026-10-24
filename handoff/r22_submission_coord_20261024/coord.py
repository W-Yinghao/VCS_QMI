"""Serialize the existing pool and legacy A3 dispatcher; reserve admitted A3 slots."""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path

BASE = Path('/home/infres/yinwang/CS_QMI')
HERE = Path(__file__).resolve().parent
WATCH = BASE / 'handoff/r22_monitor_codex_20261023'


@contextmanager
def submission_lock():
    with (HERE / 'SUBMISSION.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def outstanding_a3(analysis, submitted, live_names):
    a3 = analysis.get('a3', {})
    if (a3.get('verified', 0) < 2 or a3.get('identity') != 'REUSE_IDENTICAL_EXCEPT_HARDWARE'
            or a3.get('rule', {}).get('decision') != 'ADD_SEEDS_902_903'):
        return []
    expected = {f'r22_vl3_{arm}_s{seed}' for seed in (902, 903) for arm in ('nce_vcsw', 'nce_bcew')}
    return sorted(expected - set(submitted) - set(live_names))


def reserved_a3_slots(live):
    if (WATCH / 'STOP').exists() or (WATCH / 'PAUSE_SUBMISSIONS').exists():
        return 0
    path = WATCH / 'ANALYSIS.json'
    if not path.exists():
        return 0
    analysis = json.loads(path.read_text())
    submitted = set()
    for path in [WATCH / 'SUBMITTED.tsv', BASE / 'handoff/review_claude_20261019/deps_watch/APPS_r22_vl3_SUBMITTED.tsv']:
        if path.exists():
            for line in path.read_text().splitlines():
                fields = line.split('\t')
                if len(fields) >= 3 and fields[2].isdigit():
                    submitted.add(fields[1])
    return len(outstanding_a3(analysis, submitted, {row['name'] for row in live}))


def project_gpu_count(live):
    return sum(row['name'].startswith('r22_') and 'gpu:' in row['gres'] for row in live)
