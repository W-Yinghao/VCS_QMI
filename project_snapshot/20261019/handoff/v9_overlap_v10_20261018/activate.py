"""Apply the user's overlap policy to existing jobs, retaining the GPU gate."""
import fcntl
import os
import re
import traceback

from bridge import BUNDLE, PROJECT, V9, V10, V10_FREEZE
from bridge import command, queue, read, require, sha, utc, verify_bundle, verify_science, write


def action(args):
    record = {'utc': utc(), 'command': args}
    write(BUNDLE / 'ACTION_INTENT.json', record)
    stdout = command(args)
    record['stdout'] = stdout
    with (BUNDLE / 'ACTIONS.jsonl').open('a') as stream:
        import json
        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    return stdout


def main():
    verify_bundle()
    with (BUNDLE / 'ACTIVATE.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (BUNDLE / 'ACTION_INTENT.json').exists(), 'Previous action exists; inspect before retry')
        require(not (BUNDLE / 'ACTIVATION.json').exists(), 'Overlap already activated')
        amendment = read(BUNDLE / 'SCHEDULING_AMENDMENT.json')
        require(sha(V10 / 'PROTOCOL.json') == amendment['base_protocol_sha256'], 'Unexpected base V10 protocol')
        preflight = verify_science()
        write(BUNDLE / 'PREFLIGHT.json', preflight)
        launch = read(V10 / 'LAUNCH.json')
        require(launch['gate_job'] == '1045975' and launch['array_job'] == '1045976', 'Unexpected V10 IDs')
        rows = queue()
        by_id = {r['id']: r for r in rows}
        for jid in ['1045975'] + [f'1045976_{i}' for i in range(4)]:
            require(by_id[jid]['state'] == 'PENDING', f'Job no longer pending: {jid}')
        require(by_id['1045974']['name'] == 'handoff_v9v10', 'Unexpected old CPU monitor')
        require(sum(r['qos'] == 'normal' for r in rows) + 1 <= 29, 'No spare slot for independent CPU audit')
        before = {j: command(['scontrol', 'show', 'job', '-o', j]) for j in ('1045975','1045976')}
        require('Dependency=afterok:1045974' in before['1045975'], 'Unexpected gate dependency')
        require('Dependency=afterok:1045975' in before['1045976'], 'Unexpected formal dependency')
        write(BUNDLE / 'BEFORE.json', {'utc': utc(), 'queue': rows, 'controller': before, 'v10_launch': launch})
        audit_id = action(['sbatch', '--parsable', str(BUNDLE / 'bridge.sbatch')]).strip().split(';')[0]
        require(audit_id.isdigit(), 'Ambiguous auditor submission; inspect ACTIONS before retry')
        write(BUNDLE / 'AUDITOR_SUBMITTED.json', {'utc': utc(), 'job': audit_id})
        action(['scontrol', 'update', 'JobId=1045976', 'ArrayTaskThrottle=4'])
        action(['scontrol', 'update', 'JobId=1045975', 'Dependency='])
        after = {j: command(['scontrol', 'show', 'job', '-o', j]) for j in ('1045975','1045976')}
        require('Dependency=(null)' in after['1045975'], 'Gate dependency not cleared')
        require('Dependency=afterok:1045975' in after['1045976'], 'Formal gate dependency changed')
        require(re.search(r'ArrayTaskId=\S*%4(?:\s|$)', after['1045976']) or 'ArrayTaskThrottle=4' in after['1045976'], 'Array throttle not4')
        # The gate is already independent; cancelling its former CPU predecessor cannot block it.
        action(['scancel', '1045974'])
        record = {'utc': utc(), 'status': 'OVERLAP_ACTIVE', 'control_job': os.environ.get('SLURM_JOB_ID'),
                  'user_instruction': amendment['user_instruction'], 'independent_v9_audit_job': audit_id,
                  'retired_closeout_gate_job': '1045974', 'gate_job': '1045975', 'gate_dependency': None,
                  'array_job': '1045976', 'array': '0-3%4', 'array_dependency': 'afterok:1045975',
                  'v9_feeder_job': '1045056', 'v10_freeze_sha256': V10_FREEZE,
                  'scheduling_amendment': str(BUNDLE / 'SCHEDULING_AMENDMENT.json'),
                  'v9_closeout_blocks_v10': False, 'controller_after': after}
        write(BUNDLE / 'ACTIVATION.json', record)
        launch.update(status='QUEUED_FOR_V9_TAIL_OVERLAP', submission_status='EXISTING_JOBS_UPDATED_FOR_OVERLAP',
                      scheduling_updated_utc=utc(), scheduling_decision=str(BUNDLE / 'ACTIVATION.json'),
                      gate_dependency=None, gate_job_status='ELIGIBLE_WAIT_RESOURCES', array='0-3%4',
                      closeout_job=audit_id, closeout_status=str(BUNDLE / 'STATUS.json'),
                      closeout_blocks_v10=False, scheduling_amendment=str(BUNDLE / 'SCHEDULING_AMENDMENT.json'),
                      resume_condition='GPU gate may run as L40S capacity frees during V9 tail; formal families require successful GPU gate. V9 closeout is independent.')
        write(V10 / 'LAUNCH.json', launch)
        state_path = PROJECT / 'handoff/V10_TAKEOVER_STATE.json'
        state = read(state_path)
        state.update(updated_utc=utc(), status='BF1_QUEUED_FOR_V9_TAIL_OVERLAP', v9_closeout_required_before_v10=False,
                     execution_priority='V9 tail and V10 may overlap within the shared8 normal GPUs; V9 continues and closes independently.',
                     auto_handoff=record, ready_to_submit_bf1=False,
                     ready_to_submit_bf1_note='Existing jobs updated; do not duplicate. GPU gate no longer waits for V9 closeout.')
        state['bf1_scheduling'].update(status=launch['status'], closeout_job=audit_id, closeout_blocks_v10=False,
                                      gate_dependency=None, array_dependency='afterok:1045975', array='0-3%4',
                                      scheduling_amendment=record['scheduling_amendment'])
        for item in state['next_required_work']:
            if item['id'] == 'BF1-03': item['state'] = 'GPU_GATE_ELIGIBLE_WAIT_FREE_L40S'
            if item['id'] in ('BF1-04','BF1-05'): item['state'] = 'ARRAY_WAIT_GPU_GATE_WITH_OVERLAP_ALLOWED'
        write(state_path, state)
        state_path = PROJECT / 'handoff/V9_TAKEOVER_STATE.json'
        state = read(state_path)
        state.update(updated_utc=utc(), v9_closeout_required_before_v10=False,
                     active_priority='Complete all frozen V9 work and analyze it; allow V10 to use GPUs released during the V9 tail. No new V9 expansion grids.',
                     auto_handoff=record, auto_handoff_status='INDEPENDENT_V9_AUDIT_OVERLAP_ALLOWED',
                     status_note='V9 continues its frozen matrix and retries; independent closeout no longer blocks V10 GPU execution.')
        write(state_path, state)
        condition_path = PROJECT / 'handoff/V9_CLOSEOUT_BEFORE_V10_20261017.json'
        conditions = read(condition_path)
        write(BUNDLE / 'PREVIOUS_CLOSEOUT_POLICY.json', conditions)
        conditions.update(updated_utc=utc(), status='SUPERSEDED_BY_V9_TAIL_OVERLAP', latest_user_instruction=amendment['user_instruction'],
                          v9_closeout_required_before_v10=False, v9_closeout_still_required=True, auto_handoff=record,
                          submission_authorization='Existing V10 gate eligible immediately; formal array still requires GPU gate. V9 completion is no longer an execution dependency.')
        conditions['historical_v10_release_conditions'] = conditions.pop('v10_release_conditions')
        conditions['v9_independent_closeout_conditions'] = conditions['historical_v10_release_conditions'][:4]
        conditions['current_v10_release_conditions'] = ['Available normal L40S allocation for the GPU gate.', 'Successful same-freeze GPU gate before formal families.']
        write(condition_path, conditions)
        plan_path = V10 / 'RESUBMISSION_PLAN.json'
        plan = read(plan_path)
        plan.update(status='EXISTING_JOBS_UPDATED_FOR_V9_TAIL_OVERLAP', updated_utc=utc(), closeout_job=audit_id,
                    closeout_blocks_v10=False, activation=str(BUNDLE / 'ACTIVATION.json'),
                    scheduling_amendment=record['scheduling_amendment'], array_task_throttle=4,
                    note='Existing IDs1045975/1045976 remain active. Do not resubmit; formal GPU gate dependency retained.')
        write(plan_path, plan)
        write(BUNDLE / 'QUEUE_AFTER.json', {'utc': utc(), 'jobs': queue()})
        print({k:v for k,v in record.items() if k!='controller_after'}, flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        write(BUNDLE / 'ACTIVATION_ERROR.json', {'utc': utc(), 'traceback': traceback.format_exc()})
        raise
