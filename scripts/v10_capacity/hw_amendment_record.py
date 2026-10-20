"""Write the labelled hardware amendment of BF1_CAPACITY_EXT_K8_K32 (bundle copy + committed git copy).  No Torch.

The frozen base files (FREEZE.json e4157194, PREREGISTRATION.json f366a805, CHECKS.json) are cited, never edited.  Every number is
read from the records: the 12 completed units' receipts / runtime records, the 6 pending base jobs, the seed replication's measured
K16 costs on RTX PRO 6000 and H100 NVL, and the cross-hardware T01 notes.
"""
import json
import subprocess

import capacity_common as C
import hw_amendment as H

OWNER_CHOICE = '「final 放开，CAL 留 L40S」'
OPTION_OFFERED = ('EPFL final、BF 和 EPFL 的 RuLSIF v2 final、K8/K32 剩下的 6 个单元，改为三分区，记录实际硬件、分层报告。CAL 和计时留在 '
                  'L40S，保持每族 8 个 L40S GPU 小时的同等预算。先跑跨硬件门禁，再撤下未开始的作业重新提交。')
STANDING = '「后续待运行GPU任务统一 --partition=RTX6000PRO,L40S,H100，不要再按seed、方法或GPU型号人为分组/限卡；记录实际硬件与成本即可」'
PENDING_BASE_JOBS = {'MF_VCS_perm1_K8_seed1003': '1049238', 'MF_VCS_perm1_K32_seed1003': '1049271', 'MF_VCS_M0_K8_seed1003': '1049728',
                     'MF_VCS_M0_K32_seed1003': '1049729', 'MF_VCS_M1_K8_seed1003': '1049730', 'MF_VCS_M1_K32_seed1003': '1049731'}
XHW_NOTES = C.Path('/home/infres/yinwang/CS_QMI/handoff/review_claude_20261019/xhw_t01/NOTES.md')
REPLICATION = C.Path('/home/infres/yinwang/CS_QMI/handoff/v10_seed_replication_pool_20261019')
REPLICATION_INHERITED = C.Path('/home/infres/yinwang/CS_QMI/handoff/v10_seed_replication_hw_20261019')


def ref(path):
    return {'path': str(path), 'sha256': C.sha(path)}


def completed_units(prereg):
    runtime = {p.stem: C.read(p) for p in (C.BUNDLE / 'runtime').glob('*.json')}
    exits = {p.stem: C.read(p) for p in (C.BUNDLE / 'exits').glob('*.json')}
    out = []
    for u in prereg['units']:
        if u['key'] in H.UNIT_KEYS:
            C.require(not (C.OUT / u['key']).exists(), 'A seed-1003 unit already started under the base freeze: ' + u['key'])
            continue
        complete = C.read(C.OUT / u['key'] / 'COMPLETE.json')
        job = complete['job_id']
        env = runtime[job]['environment']
        C.require(exits[job]['exit_code'] == 0 and env['gpu_name'] == 'NVIDIA L40S', 'Completed unit is not an L40S success: ' + u['key'])
        out.append({'key': u['key'], 'job': job, 'gpu_name': env['gpu_name'], 'node': env['host'],
                    'allocated_seconds': exits[job]['allocated_seconds'], 'complete_sha256': C.sha(C.OUT / u['key'] / 'COMPLETE.json'),
                    'identity_capacity_freeze_sha256': complete['identity']['capacity_freeze_sha256']})
    C.require(len(out) == 12 and all(x['identity_capacity_freeze_sha256'] == H.BASE_FREEZE for x in out), 'Expected 12 base units')
    return out


def replication_costs():
    """Measured K16 BF1 final costs of the seed replication by GPU type (its clock started at batch start: slightly low)."""
    ledger = C.read(REPLICATION / 'LEDGER.json')
    rows = []
    for u in ledger['units']:
        root = REPLICATION_INHERITED if u.get('inherited_from') else REPLICATION
        rt, ex = root / 'runtime' / f"{u['job']}.json", root / 'exits' / f"{u['job']}.json"
        if u['family'] in C.FAMILIES and rt.exists() and ex.exists():
            rows.append({'family': u['family'], 'seed': u['seed'], 'job': u['job'], 'gpu_name': C.read(rt)['gpu_name'],
                         'node': C.read(rt)['host'], 'allocated_seconds': C.read(ex)['allocated_seconds']})
    return rows


def main():
    C.require(not (H.DIR / 'jobs').exists() and not (H.DIR / 'FREEZE.json').exists(), 'The amendment is already frozen or in use')
    prereg = C.validate_prereg(C.frozen(H.BASE_FREEZE))
    done = completed_units(prereg)
    units = H.hw_units(prereg)
    committed = sum(x['allocated_seconds'] for x in done)
    reserve = sum(u['time_limit_seconds'] + C.RESERVE_MARGIN_SECONDS for u in units)
    l40s = {f: [x['allocated_seconds'] for x in done if x['key'].startswith(f + '_K')] for f in C.FAMILIES}
    rep = replication_costs()
    k16 = prereg['budget']['k16_measured_basis']['family_summary']
    head = subprocess.run(['git', '-C', str(C.REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    rec = {
        'schema': 'v10.capacity_ext.hardware_amendment.v1',
        'name': H.NAME,
        'label': 'HARDWARE AMENDMENT to BF1_CAPACITY_EXT_K8_K32 (hardware / scheduling only; no scientific change)',
        'status': 'FROZEN_BEFORE_ANY_AMENDED_RUN',
        'utc': C.utc(),
        'generator': {'script': 'scripts/v10_capacity/hw_amendment_record.py', 'sha256': C.sha(C.Path(__file__)), 'code_commit': head},
        'amends': {'freeze': ref(C.BUNDLE / 'FREEZE.json'), 'preregistration': ref(C.BUNDLE / 'PREREGISTRATION.json'),
                   'base_cpu_checks': ref(C.BUNDLE / 'CHECKS.json'),
                   'base_files_edited': False, 'base_hardware_clause_replaced_for': list(H.UNIT_KEYS)},
        'authority': {
            'owner_choice_verbatim': OWNER_CHOICE,
            'option_as_offered_verbatim': OPTION_OFFERED,
            'standing_instruction_verbatim': STANDING, 'standing_instruction_time': '2026-10-19 09:31 UTC',
            'decision_date': '2026-10-19', 'relayed_by': 'main-session coordinator',
            'application': ('The remaining 6 K8/K32 units become three-partition jobs with recorded hardware and stratified reporting. '
                            'This extension has no CAL tuning of its own: it inherits the K16 CAL selections, which were made on '
                            'L40S and stay unchanged, so the "CAL 留 L40S" part needs no change here.')},
        'scope': {'units': [{'key': u['key'], 'job_name': u['job_name'], 'time_limit': u['time_limit'],
                             'time_limit_seconds': u['time_limit_seconds'], 'replaces_pending_base_job': PENDING_BASE_JOBS[u['key']],
                             'paired_k16_final': u['paired_k16_final']} for u in units],
                  'pending_base_jobs': ('never started (no base output directory, no admission); the main session cancels them '
                                        'and resubmits the units under this amendment; run_hw.py refuses any unit that started '
                                        'under the base freeze (BASE_UNIT_STARTED)'),
                  'completed_base_units': done,
                  'completed_units_rule': 'the 12 completed L40S units stay valid under the base freeze and are never re-run'},
        'scientific_changes': [],
        'unchanged': ['every scientific rule of the pre-registration: families, K, seeds, inherited K16 selections, training, '
                      'CAL selection items 1-4, DEV32 / DEV24, statistics, interpretation grid, forbidden claims',
                      'the science code: frozen BF1 tree b8f59ea (core e9b4239450c4e2eb, host 6c6f094684fbc50a) and the K override / '
                      'CPU-first loader of run.py, byte-identical (DELTA_REVIEW.json)',
                      'budget: 30 GPU-h ceiling, admission reservations, per-unit L40S-based limits 1:30 / 3:00, resume rule'],
        'hardware': {
            'accepted_gpus': H.GPUS, 'partition': H.PARTITIONS, 'exclude': list(H.EXCLUDED),
            'exclude_reason': {'node51': 'project-wide exclusion', 'node60': 'project-wide exclusion',
                               'node53': ('its free H100 NVL GPUs failed twice for this project with "CUDA-capable device(s) is/are '
                                          'busy or unavailable" (xhw_t01 jobs 1048501, 1048508)')},
            'gpu_name_rule': ('torch.cuda.get_device_name must equal the accepted name of the partition the job runs in; no prefix '
                              'matching, no renaming'),
            'runtime_checks_kept': ['BF1 freeze() and the GPU_GATE record', 'core / host code hashes and the BF1 job pin',
                                    'torch 2.9.0+cu128, CUDA 12.8, cuDNN 91002, 4 torch threads',
                                    'the BF1 PROTOCOL precision settings (matmul highest, cuDNN conv tf32, determinism switches)',
                                    'CUBLAS_WORKSPACE_CONFIG=:4096:8 and both TF32 override variables unset',
                                    'clean frozen BF1 tree at b8f59ea', 'job name, partition and Slurm time limit of the unit'],
            'gate_required_for': list(H.GATED),
            'gate_rule': ('run_hw.py refuses (HW_GATE_MISSING, exit 3) to start a unit on RTX6000PRO or H100 unless a hw_gate.py job '
                          'of this amendment freeze passed on that GPU type; L40S needs none (t01_coverage)')},
        't01_coverage': {
            'evidence': {'xhw_t01_notes': ref(XHW_NOTES),
                         'RTX6000PRO': 'job 1048500 node61: verify_torch 3/3 required, core_tol 654/654, adapter 319 pass / 0 required failures, geometry 94/94',
                         'H100': 'job 1048596 nodesumo01: same counts (node53 attempts 1048501 / 1048508 were device faults)',
                         'code': 'ssl_pilot_v10 @4336bb6, core hash e9b4239450c4e2eb = the BF1 accepted core'},
            'covered_on_both_types': ['MF backend numerics (objective terms vs dense, gradients, QuadraticHead / HVP, box solver vs QP '
                                      'oracle) at head widths (3,2), (7,6), (16,16)',
                                      'host adapter (training step, M1 relative zero-head rule, real-feature checks) and registration '
                                      'geometry gradients at K = 16',
                                      'complete K16 BF1 runs of the seed replication (MF perm1 / M0 / M1 on RTX PRO 6000, MF M0 on H100 NVL)'],
            'not_covered': ['K = 8 and K = 32 feature maps, M1 solver on 8 x 8 / 32 x 32 heads and the K-wide registration energy on '
                            'RTX PRO 6000 / H100 NVL (different GEMM shapes, so different CUDA kernels)',
                            'the CPU-first checkpoint restore on any GPU (resume is a fallback; no unit has resumed)'],
            'decision': ('A short CUDA gate per new GPU type (hw_gate.py: K8/K16/K32 x 3 families on real FIT data, numerical '
                         'cross-checks, CPU-first resume, runner dry run) must pass before units run there. L40S needs no new gate: '
                         'the 12 completed L40S units ran exactly this K8/K32 code end to end and passed the complete-unit audit, on '
                         'top of the L40S BF1 GPU gate and the L40S T01.'),
            'scope': 'execution and numerical gate only; not scientific evidence and not a cross-hardware equivalence claim'},
        'recording': ['per job: GPU name (torch.cuda.get_device_name), partition, node, Slurm StartTime and controller-elapsed '
                      'allocated seconds (runtime/<job>.json, ENV_<job>.json, exits/<job>.json, hw_amendment_20261019/jobs/<job>.json)',
                      'per unit: the set of GPU types / nodes it ran on (a unit resumed on another type is "mixed")',
                      'costs in native allocated GPU-seconds of the actual type, never converted to L40S-equivalents'],
        'outputs': {'root': str(H.OUT_HW), 'identity': ('base identity with schema v10.capacity_ext.unit.hw_amendment.v1 plus '
                                                       'hardware_amendment_sha256 and capacity_hw_freeze_sha256; the GPU model is '
                                                       'recorded, not part of the identity'),
                    'shared_ledgers': 'ADMISSIONS / COST_LEDGER / exits / runtime / failures / refusals / budget of the base bundle (one ceiling)'},
        'budget': {
            'ceiling_seconds': C.CEILING_SECONDS, 'time_limit_seconds': C.TIME_LIMIT_SECONDS, 'reserve_margin_seconds': C.RESERVE_MARGIN_SECONDS,
            'accounting': 'native allocated GPU-seconds of every job on any GPU type count 1:1 against the unchanged ceiling',
            'committed_by_12_completed_units_seconds': committed, 'reservation_of_6_remaining_units_seconds': reserve,
            'committed_plus_reservations_gpu_hours': (committed + reserve) / 3600,
            'measured_l40s_k8_k32_seconds': {f: {'min': min(v), 'max': max(v)} for f, v in l40s.items()},
            'k16_cost_by_type_seed_replication': rep,
            'k16_l40s_corrected_mean_seconds': {f: k16[f]['corrected_mean_seconds'] for f in C.FAMILIES},
            'time_limit_argument': ('The L40S-based limits stay. RTX PRO 6000 ran the K16 BF1 finals 1.7-2.4x faster than L40S; H100 NVL '
                                    'ran MF M0 at L40S speed. The longest completed unit (M1 K32 on L40S) used '
                                    f"{max(l40s['MF_VCS_M1']):.0f} s of its 10800 s limit."),
            'never_extended': 'unchanged: no limit, ceiling, step count or seed is increased after any result is seen'},
        'reporting': {
            'pairing_caveat': ('The K16 references and the 12 completed units ran on NVIDIA L40S. A seed-1003 K8/K32 unit that runs on '
                               'RTX PRO 6000 or H100 NVL is compared with its L40S K16 reference, so its difference includes the '
                               'hardware difference. K effects for seed 1003 are reported with their hardware, by hardware stratum and '
                               'pooled. No claim of cross-hardware equivalence.'),
            'pooled': 'the pre-registered statistics and bootstrap unchanged (default_rng(20261019), same order), every unit labelled with GPU and node',
            'strata': ('the same statistics within each GPU type of the K8/K32 unit (default_rng(20261020)); the L40S stratum is the '
                       'hardware-matched sensitivity; a one-seed stratum is descriptive only, without a grid label'),
            'interpretation': ('the pre-registered grid labels the pooled view; when the pooled label and the L40S-stratum label differ, '
                               'both are reported and the result is called hardware-sensitive'),
            'costs': 'native allocated GPU-h per unit with GPU type and node, plus totals per type',
            'implementation': 'scripts/v10_capacity/capacity_table_hw.py (frozen with this amendment); capacity_table.py remains the '
                              'base view, in which the seed-1003 units (amendment root) appear as INCOMPLETE'},
        'execution': {'frozen_tree': ('/home/infres/yinwang/CS_QMI/ssl_pilot_v10_capacity_hw_20261019 (detached worktree of branch '
                                      'v10/capacity-k8-k32 at the commit pinned in hw_amendment_20261019/FREEZE.json)'),
                      'sbatch': 'slurm/v10_capacity/run_hw.sbatch (one job per unit), slurm/v10_capacity/hw_gate.sbatch (one per GPU type)',
                      'order': ['GPU gate on RTX6000PRO and on H100 (both must PASS before units can start there)',
                                'cancel the 6 pending base jobs (never started)', 'submit the 6 lines of SUBMIT_LINES.txt'],
                      'no_requeue': True, 'automatic_retry': False},
        'delta_review': 'hw_amendment_20261019/DELTA_REVIEW.json (written by freeze_hw.py from the frozen tree)',
    }
    C.write(H.DIR / 'HARDWARE_AMENDMENT.json', rec)
    git_copy = C.REPO / H.RECORD_GIT_PATH
    git_copy.write_bytes((H.DIR / 'HARDWARE_AMENDMENT.json').read_bytes())
    H.validate_record(rec, prereg)
    print(json.dumps({'record': str(H.DIR / 'HARDWARE_AMENDMENT.json'), 'sha256': C.sha(H.DIR / 'HARDWARE_AMENDMENT.json'),
                      'committed_gpu_h': round(committed / 3600, 3), 'with_reservations_gpu_h': round((committed + reserve) / 3600, 3),
                      'l40s_k8k32_seconds': rec['budget']['measured_l40s_k8_k32_seconds'],
                      'replication': [(r['family'], r['gpu_name'][:20], round(r['allocated_seconds'])) for r in rep]}, indent=1))


if __name__ == '__main__':
    main()
