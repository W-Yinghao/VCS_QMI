"""Exit receipt written by run.sbatch after the runner exits (no Torch): the allocation measured from the Slurm StartTime.

allocated_seconds = the controller's own elapsed time (squeue %M sampled at exit, i.e. now - StartTime on the controller clock)
plus the node-measured seconds between that sample and this receipt.  If squeue is unavailable, the skew-clipped node estimate is
used (node elapsed since the batch shell started + max(0, batch start - StartTime)).  Both are recorded.
"""
import os
import socket
import sys
import time

import capacity_common as C


def main(key):
    job = os.environ['SLURM_JOB_ID']
    end = time.time()
    clock = C.Clock(os.environ.get('V10K_SCHED_START_RAW', ''), os.environ['V10K_BATCH_START'])
    node_estimate = clock.elapsed(end)
    controller = None
    try:
        used = C.slurm_seconds(os.environ.get('V10K_CONTROLLER_USED_RAW', ''))
        controller = used + max(0.0, end - float(os.environ['V10K_USED_SAMPLE_NODE']))
    except (ValueError, KeyError):
        pass
    r = {'job': job, 'unit': key, 'exit_code': int(os.environ['V10K_EXIT_CODE']), 'node': socket.gethostname(),
         'freeze_sha256': os.environ.get('V10_CAPACITY_FREEZE'), 'job_name': os.environ.get('SLURM_JOB_NAME'),
         'partition': os.environ.get('SLURM_JOB_PARTITION'), 'time_limit_raw': os.environ.get('V10K_TIME_LIMIT_RAW'),
         'controller_used_raw': os.environ.get('V10K_CONTROLLER_USED_RAW'), 'controller_elapsed_seconds': controller,
         'node_estimate_seconds': node_estimate, 'end_unix_node': end, **clock.record(),
         'allocated_seconds': controller if controller is not None else node_estimate,
         'allocated_seconds_basis': 'controller_elapsed' if controller is not None else 'node_estimate_skew_clipped',
         'gpu_count': 1, 'utc': C.utc()}
    C.write(C.BUNDLE / 'exits' / f'{job}.json', r)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1]))
