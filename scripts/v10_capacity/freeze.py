"""Freeze the capacity extension code + pre-registration into the bundle FREEZE.json (no Torch).

Requires every frozen file committed and clean on branch v10/capacity-k8-k32 and the bundle pre-registration byte-identical to the
committed copy.  Re-freezing is allowed only before any GPU job of the extension has run; the old freeze and its checks are archived.
"""
import subprocess
import sys

import capacity_common as C

CODE_FILES = ('scripts/v10_capacity/capacity_common.py', 'scripts/v10_capacity/run.py', 'scripts/v10_capacity/receipt.py',
              'scripts/v10_capacity/preregister.py', 'scripts/v10_capacity/freeze.py', 'scripts/v10_capacity/check_control.py',
              'scripts/v10_capacity/smoke_k.py', 'scripts/v10_capacity/capacity_table.py', 'scripts/v10_capacity/ledger.py',
              'slurm/v10_capacity/run.sbatch', 'slurm/v10_capacity/cpu_check.sbatch')


def git(*args):
    return subprocess.run(['git', '-C', str(C.REPO), *args], capture_output=True, text=True, check=True).stdout.strip()


def main():
    C.require(not (C.BUNDLE / 'ADMISSIONS.jsonl').exists(), 'A GPU job has been admitted: this freeze is final')
    paths = list(CODE_FILES) + [C.PREREG_GIT_PATH]
    C.require(git('status', '--porcelain', '--', *paths) == '', 'Frozen files must be committed and clean')
    tracked = set(git('ls-files', '--', *paths).splitlines())
    C.require(tracked == set(paths), 'Untracked frozen file(s): ' + str(sorted(set(paths) - tracked)))
    C.require(C.sha(C.BUNDLE / 'PREREGISTRATION.json') == C.sha(C.REPO / C.PREREG_GIT_PATH), 'Bundle and committed pre-registration differ')
    C.validate_prereg(C.read(C.BUNDLE / 'PREREGISTRATION.json'))
    old = C.BUNDLE / 'FREEZE.json'
    if old.exists():
        stamp = C.utc().replace(':', '')
        old.rename(C.BUNDLE / f'FREEZE.superseded_{stamp}.json')
        if (C.BUNDLE / 'CHECKS.json').exists():
            (C.BUNDLE / 'CHECKS.json').rename(C.BUNDLE / f'CHECKS.superseded_{stamp}.json')
    freeze = {'schema': 'v10.capacity_ext.freeze.v1', 'utc': C.utc(), 'name': C.NAME, 'branch': git('rev-parse', '--abbrev-ref', 'HEAD'),
              'commit': git('rev-parse', 'HEAD'), 'code_files': {p: C.sha(C.REPO / p) for p in CODE_FILES},
              'preregistration_sha256': C.sha(C.BUNDLE / 'PREREGISTRATION.json'),
              'preregistration_commit': git('log', '-1', '--format=%H', '--', C.PREREG_GIT_PATH),
              'bf1_frozen_tree': {'path': str(C.BF1_SOURCE), 'commit': C.BF1_COMMIT,
                                  'freeze_sha256': C.sha(C.BF1 / 'FROZEN.json')},
              'gate': 'GPU jobs require CHECKS.json status PASS for this freeze sha (CPU control checks + CPU smoke of K8/K32)'}
    C.write(C.BUNDLE / 'FREEZE.json', freeze)
    print(C.sha(C.BUNDLE / 'FREEZE.json'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
