"""R30 v1.1 protocol amendment A1 (main-session adoption of the user's 2026-11-01 supplementary review; registered before any parent,
calibration or formal branch job). Writes MATRIX_A1.csv (package MATRIX.csv + A1 overlay), plan_A1/EXECUTION_PLAN_A1.csv (the
package resolve_plan output plan_v11/EXECUTION_PLAN.csv + A1 overlay) and PROTOCOL_AMENDMENT_A1.json (frozen). The package files
themselves are not modified. Usage: apply_amendment_a1.py"""
import csv, hashlib, json, os, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CS = '/home/infres/yinwang/CS_QMI'
SRC_CSV = f'{CS}/R30_v1_1_建议延长4项_20261101.csv'
SRC_MEMO = f'{CS}/R30_v1_1_补充审查与最小实验计划_20261101_CN.md'
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
assert sha(SRC_CSV).startswith('444b6511') and sha(SRC_MEMO).startswith('5fd43936')

EXTEND_STAGE = ['r30v11_f00_s1001_HN', 'r30v11_f00_s1001_HB', 'r30v11_f60_s1001_HN', 'r30v11_f60_s1001_HB']     # memo §三
EXTEND_CTRL = ['r30v11_f60_s1002_HH', 'r30v11_f60_s1003_HH']                                                       # memo §八
NEW = [(seed, obj) for seed in (1002, 1003) for obj in ('PN', 'PV', 'PB')]                                         # memo §八
CAL_PARENTS = ['wp_f00_s1001_H_e5', 'wp_f60_s1001_H_e5', 'wp_f60_s1002_H_e5', 'wp_f60_s1003_H_e5']

matrix = list(csv.DictReader(open(f'{ROOT}/v11_protocol/MATRIX.csv', newline='', encoding='utf-8')))
fields = list(matrix[0])
plan = list(csv.DictReader(open(f'{ROOT}/plan_v11/EXECUTION_PLAN.csv', newline='', encoding='utf-8')))
pfields = list(plan[0]) + ['amendment']
byid = {r['unit_id']: r for r in matrix}
assert set(EXTEND_STAGE + EXTEND_CTRL) <= set(byid)
for u in EXTEND_STAGE:
    assert byid[u]['co_train_epochs'] == '10' and byid[u]['execution_class'] == 'SHORT10_REQUIRED' and byid[u]['continuation_objective'] in ('N', 'B')
for u in EXTEND_CTRL:
    assert byid[u]['co_train_epochs'] == '10' and byid[u]['execution_class'] == 'CONTROL10_REUSE_OR_RUN'
newrows = []
for seed, obj in NEW:
    newrows.append(dict(unit_id=f'r30a1_f60_s{seed}_{obj}', noise_percent='60', seed=str(seed), parent_id=f'wp_f60_s{seed}_H_e5',
                        warmup_source='H', continuation_objective=obj, scale_rule='fit_static', co_train_epochs='40',
                        execution_class='FULL40_SEEDCONF_A1', stage_view='no', stage_role='seed_confirmation', legacy_unit_id='',
                        change='ADD_A1', reuse_policy='NEW_PAIRED_FULL_RUN', output_subdir=f'r30a1_f60_s{seed}_{obj}'))
m_a1 = []
for r in matrix:
    r = dict(r)
    if r['unit_id'] in EXTEND_STAGE + EXTEND_CTRL:
        r.update(co_train_epochs='40', execution_class='FULL40_EXTENDED_A1', change='EXTEND_A1_KEEP_E10',
                 reuse_policy='RUN_REQUIRED_40_A1' if r['unit_id'] in EXTEND_CTRL else r['reuse_policy'])
    m_a1.append(r)
m_a1 += newrows
ids = [r['unit_id'] for r in m_a1]
sem = [(r['noise_percent'], r['seed'], r['warmup_source'], r['continuation_objective'], r['scale_rule']) for r in m_a1]
assert len(ids) == len(set(ids)) == 36 and len(sem) == len(set(sem)) and len({r['output_subdir'] for r in m_a1}) == 36
with open(f'{ROOT}/MATRIX_A1.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(m_a1)

p_a1 = []
for r in plan:
    r = dict(r)
    m = next(x for x in m_a1 if x['unit_id'] == r['unit_id'])
    if r['unit_id'] in EXTEND_STAGE + EXTEND_CTRL:
        prev = r['action']
        r.update({k: m[k] for k in fields}); r['action'] = 'RUN40'; r['amendment'] = f'A1 (was {prev})'
    else:
        r['amendment'] = ''
    p_a1.append(r)
for m in newrows:
    p_a1.append(dict(m, action='RUN40', source_run_id='', readiness='PLAN_ONLY_REQUIRES_BOUND_PARENTS_AND_GPU_GATES', amendment='A1 (new)'))
assert [r['unit_id'] for r in p_a1] == ids and all(r['action'] in ('RUN40', 'RUN10', 'RUN10_CONTROL') for r in p_a1)
assert all((r['action'] == 'RUN40') == (r['co_train_epochs'] == '40') for r in p_a1)
os.makedirs(f'{ROOT}/plan_A1', exist_ok=True)
with open(f'{ROOT}/plan_A1/EXECUTION_PLAN_A1.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=pfields); w.writeheader(); w.writerows(p_a1)

ep = lambda rows: sum(int(r['co_train_epochs']) for r in rows)
A = dict(
 amendment='A1', base_version='R30-v1.1-merged-20261101', frozen=True, frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
 frozen_before='any R30 parent state, calibration or formal branch job, and before the v1.1 GPU gate result; no R30 v1.1 result of any kind seen',
 sources=dict(proposal_csv=dict(path=SRC_CSV, sha256=sha(SRC_CSV)), review_memo=dict(path=SRC_MEMO, sha256=sha(SRC_MEMO)),
              adoption='main session decision relayed 2026-11-01: adopt memo §三, §八 (seed confirmation), §九 (formal-mean NA rule; gate coverage), '
                       '§七 registered as a trigger only (not run now)'),
 package_files_unchanged=dict(MATRIX=sha(f'{ROOT}/v11_protocol/MATRIX.csv'), plan_v11=sha(f'{ROOT}/plan_v11/EXECUTION_PLAN.csv')),
 changes={
  'sec3_extend_list_NB_to_40': dict(units=EXTEND_STAGE, from_epochs=10, to_epochs=40, execution_class='FULL40_EXTENDED_A1',
      rule='same unit id, same parent, same run (no separate short copy); fixed E10 checkpoint/readout kept (completed 10 = raw 9); native 40-epoch '
           'LR schedule (lr_update 15 unchanged); no new parent; N/B keep coefficient 1 and the R29 reduction; RTX6000PRO blocks'),
  'sec8_extend_HH_controls_to_40': dict(units=EXTEND_CTRL, from_epochs=10, to_epochs=40, execution_class='FULL40_EXTENDED_A1',
      rule='these controls were resolved RUN10_CONTROL (no strict reuse possible); the 40-epoch extension needs a real run anyway; E10 kept; L40S blocks'),
  'sec8_seed_confirmation_new_units': dict(units=[r['unit_id'] for r in newrows], epochs=40, parents=['wp_f60_s1002_H_e5', 'wp_f60_s1003_H_e5'],
      rule='PN/PV/PB x 60% x seed1002/1003 from the existing matching H parents (no new warmup); each cell has its OWN one-time 64-FIT-batch '
           'calibration on its matching H parent (seed1001 coefficients never reused); same formulas / runner / gates; L40S blocks; '
           'ordered after the 30 main paths in LINES_R30.txt (queue order only)'),
  'calibration_parents': CAL_PARENTS,
 },
 matrix_a1=dict(file='MATRIX_A1.csv', sha256=sha(f'{ROOT}/MATRIX_A1.csv'), rows=len(m_a1), unique_paths=36,
                counts={c: sum(r['execution_class'] == c for r in m_a1) for c in sorted({r['execution_class'] for r in m_a1})}),
 plan_a1=dict(file='plan_A1/EXECUTION_PLAN_A1.csv', sha256=sha(f'{ROOT}/plan_A1/EXECUTION_PLAN_A1.csv'),
              actions={a: sum(r['action'] == a for r in p_a1) for a in ('RUN40', 'RUN10', 'RUN10_CONTROL')}),
 budget=dict(cotrain_epoch_units=ep(m_a1), cotrain_epoch_units_if_all_controls_reused=ep(m_a1) - 30,
             note='package 610-660 + sec3 120 + sec8 300 = 1030-1080 (here all 5 controls run: 1080); warmup epoch units <= 35 (7 parents x 5, '
                  'unchanged); calibrations 4 (2 RTX s1001, 2 L40S s1002/1003). An epoch unit is one dual-peer native co-train epoch, not a GPU hour.'),
 summary_rule=dict(memo='§九', rule='formal cross-seed mean = NA unless n_complete == expected_n (0%: 1 seed; 60%: 3 seeds); per-seed results always '
                               'reported; a partial mean only as a separately labelled exploratory_partial_mean (never in the formal column)',
                   implementation='tools/report_stage_r30.py (patched copy of the package tools/report_stage.py; reads MATRIX_A1.csv) + '
                                  'tests/test_report_stage_r30.py (package test_missing_pair + formal-mean-is-None assertion); the package tools are not modified'),
 feedback_intervention_trigger=dict(
  memo='§七', status='REGISTERED_NOT_RUN', executed_only_if='the trigger below fires on its registered readouts',
  pairs=[dict(name='list', vcs='r30_f60_s1001_L0 (H->L)', control='r30v11_f60_s1001_HN (H->N)'),
         dict(name='pairwise', vcs='r30_f60_s1001_PV', control='r30_f60_s1001_PN')],
  seed_and_state='60% seed1001 only; evaluated once per pair after both units have their E10_RECORD (completed 10); list pair first',
  trigger='A and B and C (all three)',
  A_one_step_favorable=('on the common H parent wp_f60_s1001_H_e5, a fixed diagnostic list of 64 FIT labeled batches (selected once from the parent '
                        'with a fixed seed, ids/hash registered before evaluation, disjoint from the 64 calibration batches); group correct-hard = '
                        'observed pair correct (offline evaluator) and the observed positive not ranked first by the parent; layer C one-step '
                        'probe (main + native intra/noisy terms, real clipping and the inherited AdamW, one step from an isolated copy): '
                        'mean delta(observed margin) of the VCS objective >= the control objective and > 0, with >= 100 correct-hard comparisons'),
  B_natural_continuation_worse='E10 full-DEV rSum (same parent, same seed, fixed completed epoch 10) of the VCS unit <= control unit - 2.0',
  C_feedback_divergence=('at any completed epoch 1..10, for either peer: relative difference of the GMM clean-split size n_pred_clean >= 0.10, or '
                         'absolute difference of the noisy-branch usable-candidate fraction (i2t or t2i, epoch mean) >= 0.05, or absolute '
                         'difference of the split precision (offline evaluator only) >= 0.05'),
  intervention=dict(
   design='2x2: {VCS objective, matched control objective} x {native dynamic feedback, common reference feedback}',
   common_start=('the control unit\'s immutable completed-10 checkpoint (state_completed10.pt), copied for every cell; the VCS cell switches '
                 'only the main objective (registered coefficient for PV)'),
   window='co-train epoch raw 10: native prelude (adjust_learning_rate, GMM split from the common state) then exactly 200 training iterations per '
          'peer (A then B, native order); identical budget in all four cells; each peer\'s actual optimizer.step count recorded',
   native_dynamic_feedback='each cell generates its own split (identical at the window start), queue writes, other-peer candidate identities '
                           'and detached candidate re-encodings / refiner targets',
   common_reference_feedback=('replay, from the control cell under native feedback in the same window: GMM split + reliability probabilities '
                              '+ avg_clean_prob, labeled / unlabeled batch ids in order, queue contents and IDs written, noisy-branch candidate '
                              'identities (cols / inv / A) and their detached re-encoded embeddings; each cell still encodes its own inputs '
                              'with its own encoder and refiner'),
   readouts='full-DEV rSum + six recalls after the window, main/intra/noisy gradient norms, split stats, fixed-list margins when available',
   claims='answers only whether the VCS-control gap changes under this artificial fixed-feedback intervention; not a causal mediation '
          'decomposition of natural training; fixed feedback is not a candidate method'),
  budget='4 cells x 200 iterations x 2 peers per triggered pair (~0.3 GPUh), not part of the 1080 epoch units'),
 gate_coverage=dict(memo='§九 last paragraph', checks={
  'real dtype / AMP': 'dtype_and_score_range (float32 params, scores, losses; autocast disabled; matmul TF32 off; cudnn flags recorded)',
  'actual cosine score range': 'dtype_and_score_range (min/max of the production score matrix within [-1, 1]; implied pair-margin range)',
  'multi-positive / no-negative / tie': 'rules_gpu, math_gpu (CUDA argmax first index), CPU 05_rules',
  'calibration state isolation': 'calibration_inprocess (encoders / AdamW moments / counters unchanged, .grad untouched), calibration_records '
                                 '(parent file sha256 and fingerprint unchanged before/after)',
  'production loss / gradient equality with the reference': 'math_gpu (H/L/N/B/PN/PV/PB vs package reference on a real batch, float32 and '
                                                            'float64), CPU 02/03/06'}),
 not_adopted_now=['memo §四 three-layer probes, §五 competitor-identity audit, §六 fixed-object J/gap table: diagnostics after the first delivery '
                  '(the §七 trigger A depends on the layer-C probe; it is evaluated only when that probe exists)'],
)
json.dump(A, open(f'{ROOT}/PROTOCOL_AMENDMENT_A1.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps(dict(matrix_a1=A['matrix_a1'], plan_a1=A['plan_a1']['actions'], budget=A['budget']['cotrain_epoch_units']), ensure_ascii=False))
