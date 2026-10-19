"""Prequeue V10 behind the CPU V9 closeout job; never modify V9 jobs."""
import fcntl
import os
from pathlib import Path
import subprocess
import traceback

from bridge import BUNDLE, PROJECT, V9, V10, V10_FREEZE, SOURCE_COMMIT
from bridge import command, queue, read, require, utc, verify_bundle, verify_science, write


def submit(key, args):
    record = {'utc': utc(), 'step': key, 'command': args}
    write(BUNDLE / f'{key}_INTENT.json', record)
    try:
        stdout = command(args)
    except Exception as exc:
        write(BUNDLE / 'SUBMISSION_UNCERTAIN.json', {**record, 'error': repr(exc),
              'rule': 'Inspect squeue and INTENT files before any retry; do not submit a duplicate.'})
        raise
    job = stdout.strip().split(';')[0]
    require(job.isdigit(), f'Ambiguous sbatch result: {stdout!r}')
    write(BUNDLE / f'{key}_SUBMITTED.json', {**record, 'job': job})
    return job


def build_commands(bridge_id, gate_id=None):
    plan = read(BUNDLE / 'PLAN.json')
    gate = list(plan['gate_command'])
    gate.insert(2, f'--dependency=afterok:{bridge_id}')
    array = [s.replace('<NEW_GATE_JOB_ID>', str(gate_id)) for s in plan['array_command_template']]
    return gate, array


def main():
    verify_bundle()
    with (BUNDLE / 'ACTIVATE.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (BUNDLE / 'ACTIVATION.json').exists(), 'Already activated; inspect existing jobs')
        require(not list(BUNDLE.glob('*_INTENT.json')), 'Previous submission intent exists; reconcile before retry')
        prior = read(V10 / 'LAUNCH.json')
        require(prior['status'] == 'CANCELLED_BEFORE_START_WAIT_V9_CLOSEOUT' and not prior['active_gpu_jobs'], 'V10 already has another launch')
        rows = queue()
        require(not any(r['name'].startswith('v10bf_') or r['name'] == 'handoff_v9v10' for r in rows), 'Existing live V10/handoff job')
        # +1 monitor, +1 gate, +4 array tasks. Keep the existing normal ceiling of29.
        require(sum(r['qos'] == 'normal' for r in rows) + 6 <= 29, 'Insufficient normal submission capacity')
        write(BUNDLE / 'QUEUE_BEFORE.json', {'utc': utc(), 'jobs': rows})
        preflight = verify_science(check_data=True)
        preflight['job_id'] = os.environ.get('SLURM_JOB_ID')
        write(BUNDLE / 'PREFLIGHT.json', preflight)
        (BUNDLE / 'V10_LAUNCH_BEFORE.json').write_bytes((V10 / 'LAUNCH.json').read_bytes())
        bridge_id = submit('BRIDGE', ['sbatch', '--parsable', str(BUNDLE / 'bridge.sbatch')])
        gate_command, _ = build_commands(bridge_id)
        gate_id = submit('GATE', gate_command)
        _, array_command = build_commands(bridge_id, gate_id)
        array_id = submit('ARRAY', array_command)
        launch = {**prior, 'utc': utc(), 'gate_job': gate_id, 'array_job': array_id,
                  'gate_dependency': f'afterok:{bridge_id}', 'dependency': f'afterok:{gate_id}',
                  'closeout_job': bridge_id, 'status': 'QUEUED_WAIT_V9_CLOSEOUT_DEPENDENCY',
                  'submission_status': 'SUBMITTED_CLOSEOUT_AND_GATE_DEPENDENT',
                  'scheduling_updated_utc': utc(), 'scheduling_decision': str(BUNDLE / 'ACTIVATION.json'),
                  'active_gpu_jobs': [gate_id] + [f'{array_id}_{i}' for i in range(4)],
                  'gate_job_status': 'PENDING_V9_CLOSEOUT', 'array_job_status': 'PENDING_GATE',
                  'previous_cancelled_jobs': ['1044420', '1044424'],
                  'resume_condition': 'V9 CPU closeout job must PASS, then gate must PASS; Slurm afterok dependencies enforce both.',
                  'closeout_status': str(BUNDLE / 'STATUS.json')}
        write(V10 / 'LAUNCH.json', launch)
        activation = {'status': 'SUBMITTED', 'utc': utc(), 'control_job': os.environ.get('SLURM_JOB_ID'),
                      'closeout_job': bridge_id, 'gate_job': gate_id, 'array_job': array_id,
                      'gate_dependency': launch['gate_dependency'], 'array_dependency': launch['dependency'],
                      'array': launch['array'], 'gpu_tasks_added': 5,
                      'v10_freeze_sha256': V10_FREEZE, 'source_commit': SOURCE_COMMIT,
                      'scope': 'Queue V10 now; GPU execution only after full V9 closeout. No V9 hold/cancel or new V9 grid.'}
        write(BUNDLE / 'ACTIVATION.json', activation)
        state_path = PROJECT / 'handoff/V10_TAKEOVER_STATE.json'
        state = read(state_path)
        state.update(updated_utc=utc(), status='BF1_QUEUED_WAIT_V9_CLOSEOUT_DEPENDENCY',
                     execution_priority='V10 prequeued behind V9 CPU closeout and GPU gate; no V10 GPU execution before V9 closeout.',
                     ready_to_submit_bf1=False, ready_to_submit_bf1_note='Already submitted with dependencies; do not duplicate.',
                     auto_handoff=activation,
                     bf1_scheduling={'gate_job': gate_id, 'array_job': array_id, 'closeout_job': bridge_id,
                                     'status': launch['status'], 'gate_dependency': launch['gate_dependency'],
                                     'array_dependency': launch['dependency'], 'active_gpu_jobs': launch['active_gpu_jobs'],
                                     'old_watch_job': '1044425', 'old_watch_status': 'STOPPED_DO_NOT_RESTART'})
        for item in state['next_required_work']:
            if item['id'] == 'BF1-03': item['state'] = 'GPU_GATE_QUEUED_AFTER_V9_CLOSEOUT'
            if item['id'] in ('BF1-04', 'BF1-05'): item['state'] = 'ARRAY_QUEUED_AFTER_V9_CLOSEOUT_AND_GPU_GATE'
        write(state_path, state)
        state_path = PROJECT / 'handoff/V9_TAKEOVER_STATE.json'
        state = read(state_path)
        state.update(updated_utc=utc(), auto_handoff=activation, auto_handoff_status='ARMED_WAIT_V9',
                     v10_prequeued_gpu_jobs=launch['active_gpu_jobs'])
        write(state_path, state)
        condition_path = PROJECT / 'handoff/V9_CLOSEOUT_BEFORE_V10_20261017.json'
        conditions = read(condition_path)
        conditions.update(updated_utc=utc(), status='V10_PREQUEUED_WAIT_V9_CLOSEOUT', auto_handoff=activation,
                          latest_user_instruction='也别只排4个任务，V9完成后可以提交V10',
                          submission_authorization='V10 may be prequeued with afterok dependencies now; GPU execution remains gated by V9 closeout.')
        conditions['v10_release_conditions'][-1] = 'V10 gate is prequeued afterok of the new CPU closeout job; its array is afterok of the new gate. GPU execution starts only after all preceding closeout conditions pass. Old IDs stay cancelled.'
        write(condition_path, conditions)
        write(BUNDLE / 'QUEUE_AFTER.json', {'utc': utc(), 'jobs': queue()})
        print(activation, flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        write(BUNDLE / 'ACTIVATION_ERROR.json', {'utc': utc(), 'traceback': traceback.format_exc()})
        raise
