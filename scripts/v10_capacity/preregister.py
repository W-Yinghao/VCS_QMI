"""Write the BF1_CAPACITY_EXT_K8_K32 pre-registration record (bundle copy + committed git copy) before any job runs.

Every inherited value is read from its record, never retyped: the K16 CAL selections (SELECTION.json), the paired K16 finals
(COMPLETE.json), their measured costs (BF1 exit receipts, cost ledger, TRAIN_STEPS) and the K8/K16/K32 T02 timing rows.
Login-node safe: JSON reading and arithmetic only (no Torch, no image reads).
"""
import argparse
import datetime
import json
import statistics
import subprocess
from pathlib import Path

import capacity_common as C

DECISION = Path('/home/infres/yinwang/CS_QMI/V10_PHASE1_DECISIONS_20261017_CN.md')
PROTOCOL_P2 = C.BF1_SOURCE / 'v10' / 'protocol' / 'V10_PROTOCOL_P2_20261017.json'
T02 = C.BF1_SOURCE / 'v10' / 'T02' / 'T02_TIMING_cuda.json'
QUOTES = {
    'decision_8_1_fixed': '本批 MF 主表固定 **K=16、temperature=1、ridge=0、W weight decay=0**。',
    'decision_8_1_extension': '这是对原包 K×lr 网格的明确修改：先用共同 K16 回答机制问题。K8/K32 如用于自然图像，另记容量扩展，不暗中并入主表，也不增加没有入账的 27 格。',
    'decision_9_4_design': '三个 MF 族增加 K8 和 K32，继承各自 K16 在独立调参套选定的 lr/cap；同为 2000 步、每 250 步 checkpoint，按该重复的 CAL 样本 U-J 选 checkpoint，不访问 oracle。使用相同 10 个 data seed 和算法 seed1001，不另加算法重复或调参网格。',
    'decision_9_4_meaning': '其含义是**固定训练规则下的容量响应**，包含优化因素，不称为每个 K 独立充分优化后的最优容量曲线。',
}


def ref(path):
    return {'path': str(path), 'sha256': C.sha(path)}


def epoch(text):
    return datetime.datetime.strptime(text, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc).timestamp()


def k16_costs():
    """Measured BF1 K16 final costs per family/seed, a node-clock lag correction and the training / remainder split."""
    ledger = C.lines(C.BF1 / 'COST_LEDGER.jsonl')
    out = {}
    for family in C.FAMILIES:
        out[family] = {}
        for seed in C.SEEDS:
            d = C.BF1 / 'results' / 'FINAL_DEV' / family / f'seed_{seed}'
            complete = C.read(d / 'COMPLETE.json')
            job = complete['job_id']
            receipt = C.read(C.BF1 / 'exits' / f'{job}.json')
            start = next(r for r in ledger if r['job_id'] == job and r['event'] == 'START' and r['phase'] == 'FINAL_DEV')
            env = C.read(d / f'ENV_{job}.json')
            rows = C.lines(d / 'TRAIN_STEPS.jsonl')
            by = {r['step']: r for r in rows}
            rates, its = [], []
            for k in range(12):          # pure-training intervals: steps 250k+2 .. 250(k+1) contain no CAL readout
                a, b = 250 * k + 2, 250 * (k + 1)
                rates.append((epoch(by[b]['utc']) - epoch(by[a]['utc'])) / (b - a))
                if family == 'MF_VCS_M1':
                    its.append(statistics.mean(by[s]['m1_iterations'] for s in range(a + 1, b + 1)))
            rec = {'job': job, 'node': env['host'], 'gpu': env['gpu_name'], 'recorded_allocated_seconds': receipt['allocated_seconds'],
                   'start_unix': receipt['start_unix'], 'start_event_node_utc': start['utc'],
                   'start_event_allocated_seconds': start['allocated_seconds'],
                   'training_seconds': 250 * sum(rates), 'registration_seconds_logged':
                   sum(r['seconds'] for s in C.CAL_STEPS for r in C.lines(d / 'CAL' / f'step_{s:04d}.jsonl'))
                   + sum(r['seconds'] for r in C.lines(d / 'DEV.jsonl'))}
            if family == 'MF_VCS_M1':
                mx, my = statistics.mean(its), statistics.mean(rates)
                slope = sum((x - mx) * (y - my) for x, y in zip(its, rates)) / sum((x - mx) ** 2 for x in its)
                rec.update({'m1_mean_iterations': statistics.mean(r['m1_iterations'] for r in rows),
                            'm1_cap_hit_fraction': sum(r['m1_stop_reason'] == 'iteration_budget_exhausted' for r in rows) / len(rows),
                            'step_seconds_fit_fixed': my - slope * mx, 'step_seconds_fit_per_iteration': slope,
                            'step_seconds_at_cap100': my - slope * mx + 100 * slope})
            out[family][str(seed)] = rec
    unlagged = [r['start_event_allocated_seconds'] for f in out.values() for r in f.values() if r['start_event_allocated_seconds'] > 0]
    startup = statistics.median(unlagged)
    for f in out.values():
        for r in f.values():
            lag = 0.0
            if r['start_event_allocated_seconds'] == 0:            # node clock behind the controller StartTime at START
                lag = r['start_unix'] - epoch(r['start_event_node_utc']) + startup
            r['node_clock_lag_correction_seconds'] = lag
            r['corrected_allocated_seconds'] = r['recorded_allocated_seconds'] + lag
            r['remainder_seconds'] = r['corrected_allocated_seconds'] - r['training_seconds']
    return out, startup


def t02_ratios():
    t = C.read(T02)
    cfg, reg, env = t['parts']['train']['configs'], t['parts']['register']['energies'], t['environment']
    name = {'MF_VCS_perm1': 'MF_perm1_K{}', 'MF_VCS_M0': 'MF_M0_K{}', 'MF_VCS_M1': 'MF_M1_K{}_it30_capbound'}
    train = {f: {str(K): cfg[name[f].format(K)]['median_ms'] / cfg[name[f].format(16)]['median_ms'] for K in C.KS} for f in C.FAMILIES}
    registration = {str(K): reg[f'MF_moments_K{K}']['seconds_median'] / reg['MF_moments_K16']['seconds_median'] for K in C.KS}
    raw = {'train_median_ms': {k: cfg[k]['median_ms'] for k in sorted(cfg) if k.startswith('MF_') and 'checks' not in k and 'tol' not in k},
           'registration_seconds_median': {k: v['seconds_median'] for k, v in reg.items()}}
    return train, registration, raw, ref(T02), {'job': env.get('slurm_job_id'), 'gpu': env.get('gpu_name'), 'node': env.get('host')}


def budget():
    costs, startup = k16_costs()
    train, registration, raw, t02_ref, t02_env = t02_ratios()
    per = {}
    for f in C.FAMILIES:
        per[f] = {}
        recs = list(costs[f].values())
        for K in C.KS:
            rt, rr = train[f][str(K)], registration[str(K)]
            expected = statistics.mean(r['training_seconds'] * rt + r['remainder_seconds'] * rr for r in recs)
            if f == 'MF_VCS_M1':
                train_cap = 3000 * max(r['step_seconds_at_cap100'] for r in recs) * max(1.0, rt)
                pessimistic = train_cap + max(r['remainder_seconds'] for r in recs) * max(1.0, rr)
                rule = 'every one of the 3000 batches at the iteration cap 100 (fit of step time on mean iterations per interval, max over K16 seeds) x max(1, T02 K ratio) + max K16 remainder x max(1, registration ratio)'
            else:
                pessimistic = max(r['training_seconds'] * max(1.0, rt) + r['remainder_seconds'] * max(1.0, rr) for r in recs)
                rule = 'max over K16 seeds of training x max(1, T02 K ratio) + remainder x max(1, registration ratio)'
            per[f][str(K)] = {'expected_seconds': expected, 'pessimistic_seconds': pessimistic, 'pessimistic_rule': rule,
                              'time_limit_seconds': C.TIME_LIMIT_SECONDS[f],
                              'time_limit_over_pessimistic': C.TIME_LIMIT_SECONDS[f] / pessimistic,
                              'train_ratio_vs_K16': rt, 'registration_ratio_vs_K16': rr}
    n = len(C.SEEDS)
    exp_total = n * sum(v['expected_seconds'] for f in per.values() for v in f.values())
    pess_total = n * sum(v['pessimistic_seconds'] for f in per.values() for v in f.values())
    limits = sum(u['time_limit_seconds'] for u in C.planned_units())
    k16 = {f: {'recorded_mean_seconds': statistics.mean(r['recorded_allocated_seconds'] for r in costs[f].values()),
               'corrected_mean_seconds': statistics.mean(r['corrected_allocated_seconds'] for r in costs[f].values()),
               'corrected_max_seconds': max(r['corrected_allocated_seconds'] for r in costs[f].values())} for f in C.FAMILIES}
    long_units = [u for u in C.planned_units() if u['family'] == 'MF_VCS_M1']
    concurrent = (sum(u['time_limit_seconds'] + C.RESERVE_MARGIN_SECONDS for u in long_units)
                  + 2 * (C.TIME_LIMIT_SECONDS['MF_VCS_perm1'] + C.RESERVE_MARGIN_SECONDS))
    return {
        'unit': 'allocated L40S GPU-seconds (1 GPU per job), from the Slurm StartTime',
        'k16_measured_basis': {'per_unit': costs, 'family_summary': k16,
                               'startup_seconds_median_unlagged': startup,
                               'note': 'BF1 receipts = node end clock - controller StartTime; node57 lagged the controller (~90 s), '
                                       'so its receipts were low; corrected = recorded + lag + median unlagged startup.'},
        'k_scaling_basis': {'source': t02_ref, 'measured_on': t02_env, 'train_step_ratio_vs_K16': train,
                            'registration_ratio_vs_K16': registration, 'raw': raw,
                            'argument': ('K enters only the per-side Linear(16, K) feature maps, the K x K head W and the head '
                                         'moments; the host encoders (K-independent) dominate every training step and every '
                                         'registration energy. T02 measured train-step time at K8/K16/K32 within 0.4% (perm1, M0) '
                                         'and 1.2% (M1 at a fixed 30 iterations), and the moments registration energy within 2.7% '
                                         '(K32). perm1 registration has a smaller K-dependent part (O(CNK), moments O(CNK^2)), so '
                                         'the moments ratio bounds it. The unbounded quantity is the M1 relative-rule iteration '
                                         'count per batch, which may change with K; it is bounded by the inherited cap 100 '
                                         '(K16 finals: mean 77-82 iterations, 10-16% of batches at the cap).')},
        'per_family_K': per,
        'expected_total_seconds': exp_total, 'expected_total_gpu_hours': exp_total / 3600,
        'pessimistic_total_seconds': pess_total, 'pessimistic_total_gpu_hours': pess_total / 3600,
        'ceiling_seconds': C.CEILING_SECONDS, 'ceiling_gpu_hours': C.CEILING_SECONDS / 3600,
        'time_limit_seconds': C.TIME_LIMIT_SECONDS, 'reserve_margin_seconds': C.RESERVE_MARGIN_SECONDS,
        'sum_of_time_limits_gpu_hours': limits / 3600,
        'max_concurrent_reservation_gpu_hours': concurrent / 3600,
        'admission_rule': ('At each job start, under one bundle lock: committed = sum over every other job of this extension of its '
                           'receipt allocated seconds if it has exited, else its reservation (Slurm limit + 120 s) if admitted, '
                           f'else {C.NON_ADMITTED_PROVISIONAL_SECONDS} s; the job runs only if committed + its own reservation <= the '
                           'ceiling. Otherwise the unit is marked INCOMPLETE_BUDGET_CEILING (budget/<unit>.json) and the job exits 3.'),
        'refused_unit_rule': ('A unit refused only because of outstanding reservations may be resubmitted after those jobs exit; '
                              'it is final INCOMPLETE_BUDGET_CEILING if refused when no other job of the extension is running. '
                              'Resubmission never depends on CAL/DEV results.'),
        'resume_rule': ('A unit interrupted by its time limit or infrastructure may be resubmitted once, unconditionally on '
                        'results, subject to admission; it resumes from its last verified checkpoint (step log cut to that step, '
                        'checkpoint deserialised on CPU). A second interruption leaves the unit INCOMPLETE.'),
        'never_extended': 'No time limit, ceiling, step count or seed is increased after any CAL/DEV result is seen.',
        'queue_wait': 'reported separately, not budgeted',
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--generated-by', default='main-session delegate (Claude)')
    a = ap.parse_args()
    C.require(not (C.BUNDLE / 'ADMISSIONS.jsonl').exists() and not (C.BUNDLE / 'exits').exists(),
              'A job has already run: the pre-registration is immutable')
    text = DECISION.read_text()
    for k, q in QUOTES.items():
        C.require(q in text, 'Quote not found verbatim in the decision: ' + k)
    p2 = C.read(PROTOCOL_P2)
    head = subprocess.run(['git', '-C', str(C.REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    families = {}
    for f in C.FAMILIES:
        sp = C.BF1 / 'results' / 'CAL' / f / 'SELECTION.json'
        s = C.read(sp)
        C.require(s['status'] == 'SELECTED' and s['family'] == f, 'Selection status')
        families[f] = {'method': C.METHODS[f], 'selection': ref(sp), 'candidate': s['selected']['candidate'],
                       'k16_selected_cal_candidate_unit': s['selected']['unit'],
                       'inherited': 'lr (and M1 cap) of the K16 CAL selection on tuning seed 1017; no K8/K32 candidate'}
    k_reference = {f: {str(seed): {**ref(C.BF1 / 'results' / 'FINAL_DEV' / f / f'seed_{seed}' / 'COMPLETE.json'),
                                   'job': C.read(C.BF1 / 'results' / 'FINAL_DEV' / f / f'seed_{seed}' / 'COMPLETE.json')['job_id']}
                       for seed in C.SEEDS} for f in C.FAMILIES}
    prereg = {
        'schema': 'v10.bf1_capacity_extension.preregistration.v1',
        'name': C.NAME,
        'status': 'PREREGISTERED_BEFORE_ANY_RUN',
        'utc': C.utc(),
        'generated_by': a.generated_by,
        'generator': {'script': 'scripts/v10_capacity/preregister.py', 'sha256': C.sha(Path(__file__)), 'code_commit': head},
        'authority': {
            'decision': {**ref(DECISION), 'sections': ['8.1', '9.4'], 'quotes': QUOTES},
            'protocol_p2': {**ref(PROTOCOL_P2), 'bf_batch1.capacity_extension': p2['bf_batch1']['capacity_extension'],
                            't03.capacity.interpretation': p2['t03']['capacity']['interpretation']},
            'scope': ('The owner asked for the experiments to be completed and left the scope to the main session, which decided '
                      '(2026-10-19) to run this separately named natural-image capacity extension, modelled on the T03 capacity '
                      'block (decision 9.4): fixed training rule, hyperparameters inherited from K16, no new tuning grid, '
                      'checkpoint chosen on each run\'s own CAL.'),
        },
        'design_context': ('Designed on 2026-10-19 after the BF1 K16 CAL selections, the twelve K16 final DEV32/DEV24 readouts '
                           'and the 1004/1005 seed replication had been seen. A secondary, separately named DEV capacity extension: '
                           'not part of the BF1 primary table, not a confirmation, not an input to any method or K selection.'),
        'question': ('Under the fixed BF1 training rule, with each MF family\'s K16 CAL-selected learning rate (and M1 cap), how do '
                     'DEV registration outcomes respond when the MF head width K is 8 or 32 instead of 16, paired by training seed '
                     'with the existing K16 finals?'),
        'interpretation_scope': ('固定训练规则下的容量响应 / capacity response under a fixed training rule: includes optimisation '
                                 'effects (learning rate and M1 cap tuned at K16; M1 solver convergence at a fixed cap; the same '
                                 '3000 steps); not an optimal-capacity curve and not evidence about independently tuned K.'),
        'bf1_basis': {'source': str(C.BF1_SOURCE), 'source_commit': C.BF1_COMMIT, 'core_hash': C.CORE_HASH, 'host_hash': C.HOST_HASH,
                      'freeze': ref(C.BF1 / 'FROZEN.json'), 'gpu_gate': ref(C.BF1 / 'GPU_GATE.json'),
                      'protocol': ref(C.BF1 / 'PROTOCOL.json'), 'data_lock': ref(C.BF1 / 'DATA_LOCK.json'),
                      'use': 'scientific code imported only from this frozen tree; BF1 freeze(), GPU gate, runtime_check and checked_selection run in every job'},
        'families': families,
        'K': list(C.KS),
        'K_reference': {'K': 16, 'source': 'existing BF1 FINAL_DEV units of the same family and seed (not rerun)', 'units': k_reference},
        'seeds': list(C.SEEDS),
        'pairing': ('Same algorithm seed => same host encoder initialisation (built before the head under torch.manual_seed(seed)), '
                    'same HostBatchSampler stream (numpy default_rng(1000 + seed)): identical crops, positions and permutations '
                    'at every K. Only the feature-map widths, W and everything they influence differ. Same L40S model; nodes may '
                    'differ; CUDA kernels are not deterministic, so pairs are not bitwise replays.'),
        'tuning': {'new_candidates': 0, 'K_specific_tuning': False,
                   'rule': 'K8/K32 inherit the K16 selection; there are no K8/K32 tuning candidates (never 27 unbudgeted cells)'},
        'fixed_identical_to_bf1': {
            'protocol': ref(C.BF1 / 'PROTOCOL.json'), 'temperature': 1.0, 'ridge': 0.0, 'W_weight_decay': 0.0,
            'optimizer': 'AdamW, one learning rate tied across encoders, feature maps and (perm1/M0) W', 'non_W_weight_decay': 1e-4,
            'clip_norm': 1.0, 'scenes_per_step': 16, 'positions_per_scene': 128, 'crop': 160, 'w_init': 'zeros', 'dtype': 'float32',
            'M1': 'relative_zero_head: tau_b = 0.01 G_b(0), warm start from the previous batch W, W0 = 0, no floor, cap from the K16 selection',
            'steps': 3000, 'checkpoint_interval': 250, 'checkpoints': 12,
            'reference': 'within-scene V, ordinary permutations including fixed points',
            'registration': 'unchanged R6-REG-P2 searcher; MF_VCS_perm1 J_perm1 energy, M0/M1 exact V moments; frozen encoder/feature/W pair',
            'precision': 'BF1 PROTOCOL precision settings and environment variables, enforced by BF1 runtime_check on NVIDIA L40S'},
        'k_override': {
            'only_change': 'BF1 run.config hard-codes TrainConfig(K=16); the capacity runner takes K from the unit identity',
            'how_K_enters': ['TrainConfig.K -> Trainer: MFHead(16, 16, K, K) = per-side Linear(16, K) + softmax and W zeros(K, K)',
                             'perm1/M0: W in its own AdamW group (lr tied, decay 0), clipped with all parameters, projected to [-1, 1] after every step',
                             'M1: QuadraticHead/solve_box_head shapes follow the features (K x K); the zero-head reference, threshold and fp64-diagnostic trigger are K-independent',
                             'registration: MFRigidEnergy features_a/features_b of width K and W K x K; mf_energy_rows is width-generic',
                             'checkpoints: K-shaped state dicts loaded strictly; identity and CONFIG carry K'],
            'verification': ['run.py: dataclasses.asdict(capacity config) == BF1 config with only K replaced, at runtime for every unit',
                             'smoke_k.py (CPU Slurm, FIT data through the DataGuard): one real training step per family at K8/K16/K32'],
            'resume_fix': "BF1 run.load_checkpoint used with map_location='cpu' (only token changed); step log cut to the checkpoint on resume"},
        'selection_rule': {'scope': 'each run on its own CAL (8 cores x 10 perturbations at each of the 12 checkpoints)',
                           'lexicographic': ['success < 42 native px, descending', 'success < 9 native px, descending',
                                             'median corner error, ascending', 'earlier checkpoint step'],
                           'items': '1-4 of the common rule (item 5, candidate order, is constant within a run)'},
        'evaluation': {'DEV32': '32 DEV cores x 10 perturbations, only at the selected checkpoint',
                       'DEV24': {'rule': 'exclude DEV cores whose TMA position occurs in the included FIT or CAL roles',
                                 'excluded': list(C.DEV24_EXCLUDED), 'cores': 24}},
        'units': C.planned_units(),
        'hardware': {'partition': 'L40S', 'exclude': list(C.ALLOWED_NODES_EXCLUDED), 'gpu_name_required': 'NVIDIA L40S',
                     'gres': 'gpu:1', 'cpus_per_task': 8, 'mem': '64G', 'requeue': False,
                     'job_name_prefix': 'v10k_', 'k16_reference_nodes': 'node52 and node57 (L40S); K8/K32 may run on any non-excluded L40S node'},
        'execution': {'one_job_per_unit': True, 'units': 18, 'dispatcher': None,
                      'submission': 'by the main session after review, in the frozen unit order (bundle SUBMISSION.json)',
                      'automatic_retry': False,
                      'why_one_job_per_unit': 'per-unit limits of 1.5 h / 3 h keep backfill short and isolate failures; a family-K job with three sequential seeds would need 4.5-9 h limits'},
        'budget': budget(),
        'qc': ['all 3000 training steps logged exactly once (verify_result)', '12 CAL readouts x 80 registrations, finite',
               'selected checkpoint = CAL lexicographic minimum (recomputed)', 'DEV32 = 320 rows at the selected checkpoint; DEV24 = 240 rows / 24 cores',
               'CONFIG.K equals the unit K; lr/cap equal the inherited selection', 'every ENV: NVIDIA L40S, partition L40S, BF1 precision settings',
               'data reads only FIT/CAL/DEV through the DataGuard; TEST134 and EPFL never read',
               'all artifact hashes recorded in COMPLETE.json verified', 'exit receipt exit_code 0 for the completing job'],
        'data_access': {'roles': ['FIT', 'CAL', 'DEV'], 'TEST134': 'never read', 'EPFL': 'never read', 'evaluator_only': 'never read'},
        'test_repeat': ('TEST_REPEAT remains HOLD. This extension is not an input to any method or K selection; any held-out use of '
                        'K8/K32 would need its own pre-registration before images are read.'),
        'reporting': {
            'table': 'BF1 capacity extension (fixed training rule): separate table, never merged into the BF1 main table or the seed-replication table',
            'rows': 'family x K in {8, 16 (existing paired BF1 finals, reference), 32}',
            'per_unit': ['selected checkpoint step', 'DEV32 success42 / success9 / median error', 'DEV24 same',
                         'allocated L40S GPU-h and node', 'M1: fraction of batches at the cap, median iterations'],
            'paired': 'per seed: K minus K16 of the same family and seed; seed mean',
            'incomplete': 'a family-K cell with fewer than 3 complete seeds is reported as INCOMPLETE with its available seeds only'},
        'statistics': {
            'primary_metric': 'DEV32 success42 (as the BF1 primary readout)',
            'secondary': ['DEV32 success9', 'DEV32 median corner error', 'DEV24 sensitivity of each'],
            'contrast': 'Delta_s = metric(K, seed s) - metric(K16, seed s), s in 1001..1003; Delta_bar = mean over seeds',
            'uncertainty': ('success rates: core-cluster bootstrap of Delta_bar using per-core success fractions (each core keeps its '
                            '10 perturbations), B = 10000 resamples of the 32 (DEV24: 24) cores with replacement, numpy '
                            'default_rng(20261019), percentile 95% interval; seeds are not resampled (n = 3, shown individually); '
                            'median error: per-seed paired differences only'),
            'multiplicity': '3 families x 2 K x 2 rates x 2 core sets, descriptive; no family-wise error control and no significance wording',
            'implementation': 'scripts/v10_capacity/capacity_table.py (frozen with the runner)'},
        'interpretation_grid': {
            'applies_to': 'each family x K, DEV32 success42 primary; the same grid is reported descriptively for success9 and DEV24',
            'A_consistent_response': 'the 95% interval excludes 0 and all three Delta_s share its sign (none zero): "K differs from K16 under the fixed training rule (includes optimisation effects)", with the direction',
            'B_no_detectable_response': 'the 95% interval includes 0: "no capacity response detectable at this sample size under the fixed rule"',
            'C_mixed': 'the interval excludes 0 but the three Delta_s disagree in sign or one is 0: "mixed: core-level difference not consistent across seeds"',
            'all_outcomes': 'no optimal-K claim; the BF1 K16 main table, its selections and the method comparison rules (decision 5) are unchanged'},
        'forbidden_claims': ['optimal or best K', 'capacity curve independent of optimisation', 'K16 is optimal',
                             'merging K8/K32 into the BF1 main table or ranking families with them',
                             'significance wording for any cell', 'independent confirmation', 'cross-hardware equivalence',
                             'completed matrix when any unit is INCOMPLETE'],
        'limitations': ['3 training seeds; DEV is a development benchmark (32 cores; perturbations are not independent cores)',
                        'designed after K16 DEV readouts were seen (see design_context)',
                        'CAL is saturated for many K16 checkpoints (success42 = success9 = 1), so the checkpoint choice is driven by the median error',
                        'K16 hyperparameters may favour K16; a K8/K32 difference can reflect optimisation, not capacity alone'],
    }
    C.write(C.BUNDLE / 'PREREGISTRATION.json', prereg)
    git_copy = C.REPO / C.PREREG_GIT_PATH
    git_copy.parent.mkdir(parents=True, exist_ok=True)
    git_copy.write_bytes((C.BUNDLE / 'PREREGISTRATION.json').read_bytes())
    C.validate_prereg(prereg)
    b = prereg['budget']
    print(json.dumps({'written': str(C.BUNDLE / 'PREREGISTRATION.json'), 'sha256': C.sha(C.BUNDLE / 'PREREGISTRATION.json'),
                      'expected_gpu_h': round(b['expected_total_gpu_hours'], 3), 'pessimistic_gpu_h': round(b['pessimistic_total_gpu_hours'], 3),
                      'ceiling_gpu_h': b['ceiling_gpu_hours'], 'sum_limits_gpu_h': b['sum_of_time_limits_gpu_hours'],
                      'per_family_K': {f: {k: (round(v['expected_seconds']), round(v['pessimistic_seconds']), round(v['time_limit_over_pessimistic'], 2))
                                           for k, v in fk.items()} for f, fk in b['per_family_K'].items()}}, indent=1))


if __name__ == '__main__':
    main()
