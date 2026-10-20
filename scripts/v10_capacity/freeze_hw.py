"""Freeze the hardware amendment: DELTA_REVIEW.json and FREEZE.json in hw_amendment_20261019, computed from the frozen tree.

Requires: the frozen tree (detached worktree) clean at a commit that contains every amendment file and the committed record copy;
the base freeze unchanged; no amended job yet.  No Torch.
"""
import subprocess
import sys

import capacity_common as C
import check_hw as CH
import hw_amendment as H


def git(*args):
    return subprocess.run(['git', '-C', str(C.REPO), *args], capture_output=True, text=True, check=True).stdout.strip()


def main():
    C.require(str(C.REPO) == CH.FROZEN_TREE, 'Freeze from the frozen tree ' + CH.FROZEN_TREE)
    C.require(not (H.DIR / 'jobs').exists(), 'An amended job has already run: the amendment freeze is final')
    C.require(git('status', '--porcelain') == '', 'The frozen tree must be clean')
    paths = list(H.CODE_FILES) + [H.RECORD_GIT_PATH]
    C.require(set(git('ls-files', '--', *paths).splitlines()) == set(paths), 'Every amendment file must be committed')
    prereg = C.validate_prereg(C.frozen(H.BASE_FREEZE))
    rec = H.validate_record(C.read(H.DIR / 'HARDWARE_AMENDMENT.json'), prereg)
    C.require(C.sha(H.DIR / 'HARDWARE_AMENDMENT.json') == C.sha(C.REPO / H.RECORD_GIT_PATH), 'Record differs from the committed copy')
    bf1 = C.read(C.BF1 / 'FROZEN.json')
    science = {name: C.sha(C.BF1_SOURCE / name) == digest for name, digest in bf1['source_files'].items()}
    base = C.read(C.BUNDLE / 'FREEZE.json')
    base_code = {name: C.sha(C.REPO / name) == digest for name, digest in base['code_files'].items()}
    C.require(all(science.values()) and all(base_code.values()), 'Science or base code is not byte-identical')
    deltas = CH.source_deltas()
    C.require(deltas['run.main -> run_hw.main'] == CH.RUN_MAIN_DELTA
              and deltas['capacity_common.verify_result -> hw_amendment.verify_result'] == CH.VERIFY_DELTA
              and deltas['run.sbatch -> run_hw.sbatch (code lines)'] == CH.SBATCH_DELTA_CODE_LINES, 'Undeclared source delta')
    review = {'schema': 'v10.capacity_ext.hw_delta_review.v1', 'utc': C.utc(), 'frozen_tree': str(C.REPO), 'commit': git('rev-parse', 'HEAD'),
              'base_freeze_sha256': H.BASE_FREEZE,
              'science_code': {'bf1_frozen_tree': str(C.BF1_SOURCE), 'bf1_commit': C.BF1_COMMIT, 'files_checked': len(science),
                               'byte_identical_to_bf1_FROZEN': all(science.values()), 'core_hash': C.CORE_HASH, 'host_hash': C.HOST_HASH},
              'capacity_code': {'base_frozen_files_checked': len(base_code), 'byte_identical_to_base_FREEZE': all(base_code.values()),
                                'files': sorted(base_code), 'preregistration_git_copy_sha256': C.sha(C.REPO / C.PREREG_GIT_PATH)},
              'reused_unchanged': {'run.py': ['import_bf1', 'capacity_config_factory', 'require_only_k_differs', 'cpu_first_loader',
                                              'capacity_identity', 'git_head', 'interrupted'],
                                   'capacity_common.py': ['frozen', 'validate_prereg', 'checks_passed', 'unit_by_key', 'Clock', 'Allocation',
                                                          'admit', 'committed_seconds', 'prepare_resume', 'expected_identity', 'metrics',
                                                          'selection_key', 'slurm_seconds', 'write', 'append', 'Refused'],
                                   'receipt.py': ['whole file (exit receipt)'], 'smoke_k.py': ['Recorder', 'fp64_head_reference'],
                                   'capacity_table.py': ['k16_reference', 'm1_diagnostics', 'unit_cost', 'interval', 'grid']},
              'declared_deltas': deltas,
              'new_files': list(H.CODE_FILES),
              'no_scientific_change': ('Every changed line concerns the accepted partitions / GPU names / nodes, the amendment freeze, '
                                       'gates, identity fields or the output root; the training, selection, evaluation, data access and '
                                       'accounting code paths are the unchanged frozen functions.')}
    C.write(H.DIR / 'DELTA_REVIEW.json', review)
    old = H.DIR / 'FREEZE.json'
    if old.exists():
        stamp = C.utc().replace(':', '')
        old.rename(H.DIR / f'FREEZE.superseded_{stamp}.json')
        if (H.DIR / 'CHECKS.json').exists():
            (H.DIR / 'CHECKS.json').rename(H.DIR / f'CHECKS.superseded_{stamp}.json')
    freeze = {'schema': 'v10.capacity_ext.hw_freeze.v1', 'utc': C.utc(), 'name': H.NAME, 'frozen_tree': str(C.REPO),
              'commit': git('rev-parse', 'HEAD'), 'branch_of_commit': 'v10/capacity-k8-k32', 'base_freeze_sha256': H.BASE_FREEZE,
              'record_sha256': C.sha(H.DIR / 'HARDWARE_AMENDMENT.json'), 'delta_review_sha256': C.sha(H.DIR / 'DELTA_REVIEW.json'),
              'code_files': {p: C.sha(C.REPO / p) for p in H.CODE_FILES},
              'gates': 'units need CHECKS.json PASS for this freeze; RTX6000PRO / H100 units also need a gpu_gates/<partition>_<job>.json PASS'}
    C.write(H.DIR / 'FREEZE.json', freeze)
    print(C.sha(H.DIR / 'FREEZE.json'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
