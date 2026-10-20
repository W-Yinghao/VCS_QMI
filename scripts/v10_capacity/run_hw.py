"""BF1_CAPACITY_EXT_K8_K32 unit runner under the 2026-10-19 hardware amendment (one Slurm GPU job = one of the 6 seed-1003 units).

run.py main() with only hardware-related replacements (check_hw.py compares the two sources): the three accepted partitions /
GPU types instead of L40S, the hardware-amended runtime check instead of BF1 runtime_check (every BF1 version, precision and
environment check kept), the amendment freeze and checks, a passing K8/K32 CUDA gate for RTX6000PRO / H100, a refusal if the unit
ever started under the base freeze, the amended identity and output root.  The scientific path is the unchanged BF1 tree with the
unchanged K override and CPU-first loader of run.py; the 12 completed L40S units are never re-run.
"""
from __future__ import annotations

import os
import signal
import sys
import traceback

import capacity_common as C
import hw_amendment as H
import run as R


def main(key):
    C.require(os.environ.get('SLURM_JOB_ID'), 'Scientific execution requires Slurm')
    job = os.environ['SLURM_JOB_ID']
    freeze_sha = os.environ.get('V10_CAPACITY_FREEZE')
    hw_freeze_sha = os.environ.get('V10_CAPACITY_HW_FREEZE')
    prereg = C.validate_prereg(C.frozen(freeze_sha))
    C.checks_passed(freeze_sha)
    record = H.validate_record(H.frozen_hw(hw_freeze_sha), prereg)
    H.checks_passed_hw(hw_freeze_sha)
    unit = H.hw_unit(prereg, key)
    C.require(os.environ.get('SLURM_JOB_NAME') == unit['job_name'], 'Job name must be ' + unit['job_name'])
    partition = os.environ.get('SLURM_JOB_PARTITION')
    C.require(partition in H.GPUS, 'Amended hardware is one of the partitions ' + H.PARTITIONS)
    limit = C.slurm_seconds(os.environ.get('V10K_TIME_LIMIT_RAW', ''))
    C.require(limit == unit['time_limit_seconds'], f"Slurm time limit {limit}s != pre-registered {unit['time_limit_seconds']}s")
    allocation = C.Allocation(unit, job, C.Clock.from_env())
    state = 'FAIL'
    try:
        directory = H.OUT_HW / unit['key']
        H.base_untouched(unit)
        gate = H.gate_passed(partition, hw_freeze_sha)
        done = (directory / 'COMPLETE.json').exists()
        if done:
            H.verify_result(unit, prereg, freeze_sha, require_receipt=False, hw_freeze_sha=hw_freeze_sha)
        admission = C.admit(unit, job, limit + C.RESERVE_MARGIN_SECONDS, already_complete=done)
        if admission['decision'] == 'ALREADY_COMPLETE':
            state = 'ALREADY_COMPLETE'
            print('ALREADY_COMPLETE', key, flush=True)
            return 0
        if admission['decision'] != 'ADMITTED':
            raise C.Refused(admission['decision'], f"committed={admission['committed_other_seconds']:.0f}s "
                            f"reserve={admission['reserved_seconds']}s ceiling={C.CEILING_SECONDS}s")
        os.environ['V10_BF1_EXPECT_FREEZE'] = prereg['bf1_basis']['freeze']['sha256']
        bf1 = R.import_bf1()
        bf1.freeze()
        bf1.require_gpu_gate()
        C.require(bf1.A.code_hash() == C.CORE_HASH and bf1.A.host_code_hash() == C.HOST_HASH, 'Accepted BF core/host changed')
        bf1.A.check_job_pin()
        env = H.runtime_environment(bf1)
        C.require(env['git_commit'] == C.BF1_COMMIT and env['git_dirty_code'] == '' and env['code_hash_src_scripts_v10'] == C.CORE_HASH
                  and env['host'] not in H.EXCLUDED, 'Frozen BF1 tree or node is not the pre-registered one')
        ident = R.capacity_identity(bf1, unit, prereg, freeze_sha)
        ident.update(schema=H.SCHEMA_UNIT, hardware_amendment_sha256=C.sha(H.DIR / 'HARDWARE_AMENDMENT.json'),
                     capacity_hw_freeze_sha256=hw_freeze_sha)
        C.require(ident == H.expected_identity_hw(unit, prereg, freeze_sha, hw_freeze_sha), 'Identity differs from the amendment reconstruction')
        config = R.capacity_config_factory(bf1)
        resolved = R.require_only_k_differs(bf1, config, ident)
        bf1.config = config
        bf1.load_checkpoint = R.cpu_first_loader(bf1)
        C.require(not (directory / 'IDENTITY.json').exists() or C.read(directory / 'IDENTITY.json') == ident,
                  'Refusing to touch a unit directory that holds another identity')
        resume = C.prepare_resume(directory, job)
        if resume:
            allocation.record('RESUME_PREPARED', **resume)
        C.write(C.BUNDLE / 'runtime' / f'{job}.json', {'utc': C.utc(), 'job': job, 'unit': unit, 'identity': ident,
                'train_config': resolved, 'environment': env, 'admission': admission, 'resume': resume,
                'capacity_commit': R.git_head(C.REPO), 'capacity_freeze_sha256': freeze_sha,
                'patches': ['run.config: K from identity', "run.load_checkpoint: map_location='cpu'"],
                'capacity_hw_freeze_sha256': hw_freeze_sha, 'hardware_amendment': record['name'], 'gpu_gate': gate,
                'gpu_name': env['gpu_name'], 'partition': env['slurm_partition'], 'node': env['host']})
        C.write(H.DIR / 'jobs' / f'{job}.json', {'utc': C.utc(), 'job': job, 'unit': unit['key'], 'gpu_name': env['gpu_name'],
                'partition': env['slurm_partition'], 'node': env['host'], 'hw_freeze_sha256': hw_freeze_sha})
        allocation.record('RUNTIME_VERIFIED', gpu=env['gpu_name'], node=env['host'], resume=resume, partition=env['slurm_partition'])
        print('RUNTIME_VERIFIED', key, env['gpu_name'], env['host'], 'K=' + str(unit['K']), flush=True)
        bf1.run_unit(ident, directory, allocation, env)
        H.verify_result(unit, prereg, freeze_sha, require_receipt=False, hw_freeze_sha=hw_freeze_sha)
        state = 'COMPLETE'
        print('UNIT_COMPLETE', key, flush=True)
        return 0
    except C.Refused as refusal:
        state = refusal.code
        C.write(C.BUNDLE / 'refusals' / f'{job}.json', {'utc': C.utc(), 'job': job, 'unit': key, 'code': refusal.code,
                                                        'detail': refusal.detail})
        print('REFUSED', key, refusal, flush=True)
        return C.EXIT_REFUSED
    except BaseException as error:
        state = 'INTERRUPTED' if isinstance(error, InterruptedError) else 'FAIL'
        C.write(C.BUNDLE / 'failures' / f'{job}.json', {'utc': C.utc(), 'job': job, 'unit': key, 'error': repr(error),
                                                        'traceback': traceback.format_exc()})
        raise
    finally:
        allocation.record('END', status=state)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, R.interrupted)
    signal.signal(signal.SIGUSR1, R.interrupted)
    raise SystemExit(main(sys.argv[1]))
