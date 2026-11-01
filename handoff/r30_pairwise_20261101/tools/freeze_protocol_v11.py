"""Write the v1.1 (+ amendment A1) PROTOCOL_FROZEN.json before the v1.1 GPU gate result, the 7 parents, the 4 calibrations and any
branch. Usage: freeze_protocol_v11.py CPU_JOB"""
import csv, hashlib, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = '/home/infres/yinwang/CS_QMI/r30_v11/VCS_R30_v1_1_Merged_Server_20261101'
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
cpu_job = sys.argv[1]
cpu = json.load(open(os.path.join(ROOT, f'CPU_CHECK_V11_{cpu_job}.json')))
im = json.load(open(os.path.join(ROOT, 'code_r30v11', 'INSTALL_MANIFEST.json')))
assert cpu['all_pass'] and cpu['code_sha256'] == im['files_sha256'], 'v1.1 CPU check must PASS on this exact snapshot'
man = json.load(open(os.path.join(PKG, 'MANIFEST_SHA256.json')))['files']
for f, hh in man.items():
    assert sha(os.path.join(PKG, f)) == hh, f
for name, rel in (('MATRIX.csv', 'MATRIX.csv'), ('WARMUP_PARENTS.csv', 'configs/WARMUP_PARENTS.csv'), ('PROTOCOL.json', 'configs/PROTOCOL.json'),
                  ('STAGE_VIEW.csv', 'configs/STAGE_VIEW.csv'), ('LEGACY_TO_MERGED.csv', 'configs/LEGACY_TO_MERGED.csv')):
    assert sha(os.path.join(ROOT, 'v11_protocol', name)) == man[rel], name
lines = os.path.join(ROOT, 'LINES_R30.txt')
plan = os.path.join(ROOT, 'plan_A1', 'EXECUTION_PLAN_A1.csv')
plan_pkg = os.path.join(ROOT, 'plan_v11', 'EXECUTION_PLAN.csv')
amend = os.path.join(ROOT, 'PROTOCOL_AMENDMENT_A1.json')
AM = json.load(open(amend))
assert AM['frozen'] and sha(plan) == AM['plan_a1']['sha256'] and sha(os.path.join(ROOT, 'MATRIX_A1.csv')) == AM['matrix_a1']['sha256']
pkg_math = os.path.join(ROOT, 'v11_package_checks', f'math_checks_{cpu_job}.json')
P = dict(
 version='R30-v1.1-merged-20261101', amendments=['A1'], amendment_a1_sha256=sha(amend), amendment_a1_file='PROTOCOL_AMENDMENT_A1.json', frozen=True, frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
 supersedes=[dict(file='PROTOCOL_FROZEN_v1_SUPERSEDED.json', sha256=sha(os.path.join(ROOT, 'PROTOCOL_FROZEN_v1_SUPERSEDED.json')),
                  note='R30 v1 (12-unit) protocol; never used for a formal job'),
             dict(file='PROTOCOL_FROZEN_v11_preA1_SUPERSEDED.json', sha256=sha(os.path.join(ROOT, 'PROTOCOL_FROZEN_v11_preA1_SUPERSEDED.json')),
                  note='v1.1 freeze before amendment A1 (30 paths); used by no parent / calibration / branch / gate result (only the protocol-independent early stages of smoke 1080122)'),
             dict(file='PROTOCOL_FROZEN_v11A1_r0_SUPERSEDED.json', sha256=sha(os.path.join(ROOT, 'PROTOCOL_FROZEN_v11A1_r0_SUPERSEDED.json')),
                  note='v1.1+A1 freeze on the snapshot before the milestone RNG-restore fix (smoke 1080122 failed at its first branch, no gate result); '
                       'rules and matrix unchanged, only code sha256s / CPU check / smoke references differ')],
 frozen_before='the v1.1 GPU gate result, all 7 parent states, all 4 calibrations and every branch unit; no v1.1 GPU result seen. '
               'Only v1.1 CPU checks (synthetic), the pre-migration v1 runfill smoke 1080050 and the failed smoke 1080122 (stopped by the '
               'milestone RNG check at its first branch, no gate) had run.',
 authority='package configs/PROTOCOL.json + MATRIX.csv (byte-identical copies in v11_protocol/, verified against the package manifest) + frozen amendment A1 (MATRIX_A1.csv overlay)',
 package=dict(dir=PKG, MANIFEST_SHA256=sha(os.path.join(PKG, 'MANIFEST_SHA256.json')), files_verified=len(man),
              local_checks=dict(check_package='v11_package_checks/check_package.json', resolve_preview='v11_package_checks/resolve_preview.out',
                                unittest='v11_package_checks/unittest.out (34 tests OK)', run_math_checks=os.path.relpath(pkg_math, ROOT),
                                run_math_checks_status=json.load(open(pkg_math))['status'],
                                note='package checks are local CPU checks of protocol/matrix/reference math, not production gates')),
 code_snapshot=os.path.join(ROOT, 'code_r30v11'), install_manifest_sha256=sha(os.path.join(ROOT, 'code_r30v11', 'INSTALL_MANIFEST.json')),
 base_snapshot=im['base_snapshot'], v1_snapshot=im['v1_snapshot'],
 migration_diff=dict(file='R30_V11_DIFF.patch', sha256=sha(os.path.join(ROOT, 'R30_V11_DIFF.patch')), v1_changed=im['v1_changed'], v1_added=im['v1_added']),
 diff_vs_r29=dict(file='R30_V11_DIFF_VS_R29.patch', sha256=sha(os.path.join(ROOT, 'R30_V11_DIFF_VS_R29.patch')), added=im['added_vs_base'],
                  byte_identical=im['unchanged_byte_identical']),
 code_files_sha256=im['files_sha256'],
 cpu_check=dict(job=cpu_job, file=f'CPU_CHECK_V11_{cpu_job}.json', sha256=sha(os.path.join(ROOT, f'CPU_CHECK_V11_{cpu_job}.json')), all_pass=True,
                reference=dict(file=os.path.join(ROOT, 'reference', 'objective_checks.py'), sha256=sha(os.path.join(ROOT, 'reference', 'objective_checks.py')),
                               identical_to_package=sha(os.path.join(ROOT, 'reference', 'objective_checks.py')) == man['reference/objective_checks.py'])),
 cells=json.load(open(os.path.join(ROOT, 'CELLS_R30.json'))),
 parents=list(csv.DictReader(open(os.path.join(ROOT, 'v11_protocol', 'WARMUP_PARENTS.csv')))),
 matrix=dict(file='v11_protocol/MATRIX.csv', sha256=sha(os.path.join(ROOT, 'v11_protocol', 'MATRIX.csv')), unique_paths=30),
 matrix_a1=AM['matrix_a1'],
 reuse_decisions=dict(file='REUSE_DECISIONS.json', sha256=sha(os.path.join(ROOT, 'REUSE_DECISIONS.json')), all_run=True),
 execution_plan_package=dict(file='plan_v11/EXECUTION_PLAN.csv', sha256=sha(plan_pkg), tool='package tools/resolve_plan.py --reuse-decisions REUSE_DECISIONS.json'),
 execution_plan=dict(file='plan_A1/EXECUTION_PLAN_A1.csv', sha256=sha(plan), tool='tools/apply_amendment_a1.py (package plan + A1 overlay)',
                     actions={a: sum(1 for r in csv.DictReader(open(plan)) if r['action'] == a) for a in ('RUN40', 'RUN10', 'RUN10_CONTROL')}),
 execution_list=dict(file='LINES_R30.txt', sha256=sha(lines), n=36, lines=open(lines).read().splitlines(),
                     order='stage-view FULL40 (H0/L0 + A1-extended s1001 HN/HB), the 12 10-epoch branches/controls, FULL40 s1001 Ls/PN/PV/PB, '
                           'then the A1 L40S block (HH s1002/s1003 extended to 40 + PN/PV/PB s1002/s1003); queue order only'),
 rules=dict(deviations='DEVIATIONS.json', deviations_sha256=sha(os.path.join(ROOT, 'DEVIATIONS.json')),
            tau=0.05, batch=128, K=5, warmup_epochs=5, co_train_epochs={'FULL40 incl. A1': 40, 'SHORT10/CONTROL10': 10}, lr_update=15,
            milestone_completed_epochs=10, raw_epoch_of_milestone=9,
            stage_paths='N/L/B coefficient 1, R29 co-train reduction; H native', fit_static='Ls/PN/PV/PB on the matching H parent (s1001; A1: PN/PV/PB also s1002/s1003, own calibration per cell), 64 FIT batches per peer, frozen',
            summary_rule=AM['summary_rule']['rule'],
            endpoints=dict(stage='full-DEV rSum + six recalls at completed_cotrain_epochs=10 (never best-in-first-10); paired by noise/seed; interaction only for seeds with all four paths',
                           full40='best full-DEV rSum over 40 epochs + same-checkpoint six recalls, final, last5, ranks, cost'),
            completion='all technically valid registered units run; no result-based stop / extension; 10-epoch branches never extended'),
 budget_note=dict(epoch_units=dict(warmup=35, cotrain=AM['budget']['cotrain_epoch_units'], package_range='610-660', with_A1='1030-1080'),
                  estimate_from_R29_receipts=('RTX PRO 6000: warmup epoch ~120 s (both peers), co-train epoch ~300 s at 0% and ~150-260 s at 60%; '
                                              'L40S assumed ~1.6x slower (no R29 Flickr L40S receipt). Parents ~0.75 GPUh (3 RTX) + ~1.5 GPUh (4 L40S); '
                                              'calibrations 4 x ~0.15 GPUh; RTX: 12 FULL40 + 4 A1-extended HN/HB x 2.6-3.4 h ~ 47 GPUh, 2 x 10-epoch ~1.4 GPUh; '
                                              'L40S: 8 x 10-epoch ~1 h ~ 8 GPUh + 8 A1 FULL40 x ~4 h ~ 32 GPUh; total ~90 GPUh plus continuation / '
                                              'staging overhead. Real costs from scheduler receipts.')),
 test='never read (stage.sh copies only train/dev; no TEST entry is run).')
json.dump(P, open(os.path.join(ROOT, 'PROTOCOL_FROZEN.json'), 'w'), indent=1)
print('frozen', sha(os.path.join(ROOT, 'PROTOCOL_FROZEN.json')))
