"""Control checks for BF1_CAPACITY_EXT_K8_K32 without Torch or image reads: matrix, pre-registration, identities, the BF1 patch
contract, Slurm clock/cost parsing, admission against the ceiling, resume step-log truncation and the complete-unit audit."""
import ast
import copy
import importlib.util
import json
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

import capacity_common as C

HERE = Path(__file__).resolve().parent
SBATCH = C.REPO / 'slurm' / 'v10_capacity' / 'run.sbatch'
CPU_SBATCH = C.REPO / 'slurm' / 'v10_capacity' / 'cpu_check.sbatch'


def bf1_common():
    spec = importlib.util.spec_from_file_location('bf1_common_for_checks', C.BF1_SOURCE / 'scripts' / 'v10_bf1' / 'common.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def function_source(path, outer, inner=None):
    text = Path(path).read_text()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.FunctionDef) and node.name == outer:
            if inner is None:
                return ast.get_source_segment(text, node)
            for sub in ast.walk(node):
                if isinstance(sub, ast.FunctionDef) and sub.name == inner:
                    return textwrap.dedent(' ' * sub.col_offset + ast.get_source_segment(text, sub))
    raise KeyError(outer)


def prereg():
    return C.read(C.BUNDLE / 'PREREGISTRATION.json')


def make_fake_unit(out, bundle, unit, p, ident, job='999'):
    """A synthetic but structurally complete unit (no science): exercises every verify_result check."""
    d = out / unit['key']
    k16 = C.read(Path(unit['paired_k16_final']) / 'COMPLETE.json')
    C.write(d / 'IDENTITY.json', ident)
    c = p['families'][unit['family']]['candidate']
    C.write(d / 'CONFIG.json', {'method': unit['method'], 'seed': unit['seed'], 'K': unit['K'], 'encoder_lr': c['lr'],
                                'head_lr': c['lr'], 'm1': {'max_iterations': c['cap'] or 30}})
    (d / 'TRAIN_STEPS.jsonl').write_text(''.join(json.dumps({'step': s, 'loss': 0.0}) + '\n' for s in range(1, 3001)))
    cal = sorted(k16['per_core'])[:8]
    cps = []
    for step in C.CAL_STEPS:
        pt = d / 'checkpoints' / f'step_{step:04d}.pt'
        pt.parent.mkdir(parents=True, exist_ok=True)
        pt.write_bytes(f'fake checkpoint {unit["key"]} {step}'.encode())
        C.write(pt.with_suffix('.json'), {'identity': ident, 'step': step, 'sha256': C.sha(pt)})
        rows = [{'core': core, 'perturbation': i, 'err_full': 3.0 + abs(step - 1500) / 1000 + 0.01 * i, 'role': 'CAL',
                 'checkpoint_sha256': C.sha(pt), 'seconds': 1.0} for core in cal for i in range(10)]
        (d / 'CAL').mkdir(exist_ok=True)
        (d / 'CAL' / f'step_{step:04d}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        C.write(d / 'CAL' / f'step_{step:04d}.identity.json', {'unit': ident, 'checkpoint_sha256': C.sha(pt)})
        m = C.metrics(rows)
        C.write(d / 'CAL' / f'step_{step:04d}.summary.json', {'metrics': m})
        cps.append({'step': step, 'metrics': m, 'checkpoint': f'checkpoints/step_{step:04d}.pt', 'checkpoint_sha256': C.sha(pt)})
    chosen = min(cps, key=C.selection_key)
    dev = [{'core': core, 'perturbation': i, 'err_full': 2.0 + sum(map(ord, core)) % 50 + i, 'role': 'DEV',
            'checkpoint_sha256': chosen['checkpoint_sha256'], 'seconds': 1.0} for core in sorted(k16['per_core']) for i in range(10)]
    (d / 'DEV.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in dev))
    C.write(d / 'DEV.identity.json', {'unit': ident, 'checkpoint_sha256': chosen['checkpoint_sha256']})
    C.write(d / 'DEV.summary.json', {'metrics': C.metrics(dev)})
    protocol = C.read(p['bf1_basis']['protocol']['path'])['precision']
    C.write(d / f'ENV_{job}.json', {'gpu_name': 'NVIDIA L40S', 'slurm_partition': 'L40S', 'host': 'node52',
                                     'precision': {**protocol['settings'], 'env': protocol['env']}})
    C.write(d / 'DATA_READS.json', [{'core': 'x', 'role': r} for r in ('FIT', 'CAL', 'DEV')])
    sensitive = [r for r in dev if r['core'] in k16['DEV24_cores']]
    artifacts = {str(x.relative_to(d)): C.sha(x) for x in d.rglob('*') if x.is_file()}
    C.write(d / 'COMPLETE.json', {'status': 'COMPLETE', 'identity': ident, 'steps': 3000, 'checkpoints': cps, 'selected': chosen,
                                  'job_id': job, 'artifacts': artifacts, 'DEV32': C.metrics(dev), 'DEV24': C.metrics(sensitive),
                                  'DEV24_cores': k16['DEV24_cores'],
                                  'per_core': {core: C.metrics([r for r in dev if r['core'] == core]) for core in sorted(k16['per_core'])}})
    C.write(bundle / 'exits' / f'{job}.json', {'job': job, 'unit': unit['key'], 'exit_code': 0, 'allocated_seconds': 100.0})
    return d


class Matrix(unittest.TestCase):
    def test_eighteen_units_seed_major_names_and_limits(self):
        units = C.planned_units()
        self.assertEqual(len(units), 18)
        self.assertEqual(len({u['key'] for u in units}), 18)
        self.assertEqual({(u['family'], u['K'], u['seed']) for u in units},
                         {(f, k, s) for f in C.FAMILIES for k in (8, 32) for s in (1001, 1002, 1003)})
        self.assertEqual([u['seed'] for u in units], [1001] * 6 + [1002] * 6 + [1003] * 6)
        for u in units:
            self.assertTrue(u['job_name'].startswith('v10k_') and not u['job_name'].startswith('v9'))
            self.assertNotIn(16, (u['K'],))
            self.assertEqual(u['time_limit_seconds'], 10800 if u['family'] == 'MF_VCS_M1' else 5400)
            self.assertTrue((Path(u['paired_k16_final']) / 'COMPLETE.json').exists())

    def test_slurm_durations_and_start_times(self):
        cases = {'1:30:00': 5400, '01:30:00': 5400, '03:00:00': 10800, '1-00:00:00': 86400, '22:34': 1354, '0:05': 5, '1:02:03': 3723,
                 '2-03:04:05': 183845}
        for text, value in cases.items():
            self.assertEqual(C.slurm_seconds(text), value, text)
        for bad in ('', 'UNLIMITED', '90', '1:60:00', '1-02:03', 'N/A'):
            with self.assertRaises(ValueError):
                C.slurm_seconds(bad)
        self.assertEqual(C.slurm_start_unix('2026-10-19T00:00:00'), 1792368000.0)
        with self.assertRaises(ValueError):
            C.slurm_start_unix('N/A')

    def test_clock_uses_starttime_and_clips_a_lagging_node(self):
        start = C.slurm_start_unix('2026-10-19T12:00:00')
        ahead = C.Clock('2026-10-19T12:00:00', start + 7)
        self.assertEqual(ahead.startup, 7)
        self.assertAlmostEqual(ahead.elapsed(start + 107), 107)
        lag = C.Clock('2026-10-19T12:00:00', start - 90)          # node57-like clock behind the controller
        self.assertEqual(lag.startup, 0)
        self.assertAlmostEqual(lag.elapsed(start - 90 + 50), 50)
        missing = C.Clock('', start)
        self.assertEqual((missing.startup, missing.source), (0.0, 'batch_start_only_starttime_unavailable'))


class Preregistration(unittest.TestCase):
    def setUp(self):
        self.p = prereg()

    def test_record_matches_code_records_and_selections(self):
        C.validate_prereg(self.p)
        for f in C.FAMILIES:
            sel = C.read(C.BF1 / 'results' / 'CAL' / f / 'SELECTION.json')
            self.assertEqual(self.p['families'][f]['candidate'], sel['selected']['candidate'])
            self.assertEqual(self.p['families'][f]['selection']['sha256'], C.sha(C.BF1 / 'results' / 'CAL' / f / 'SELECTION.json'))
        got = {f: (self.p['families'][f]['candidate']['lr'], self.p['families'][f]['candidate']['cap']) for f in C.FAMILIES}
        self.assertEqual(got, {'MF_VCS_perm1': (3e-4, None), 'MF_VCS_M0': (1e-4, None), 'MF_VCS_M1': (3e-4, 100)})
        text = Path(self.p['authority']['decision']['path']).read_text()
        for q in self.p['authority']['decision']['quotes'].values():
            self.assertIn(q, text)

    def test_changes_are_rejected(self):
        def mutate(fn):
            q = copy.deepcopy(self.p); fn(q)
            with self.assertRaises((RuntimeError, KeyError)):
                C.validate_prereg(q)
        mutate(lambda q: q['units'].append({**q['units'][0], 'K': 16}))
        mutate(lambda q: q['units'][0].update(time_limit_seconds=86400))
        mutate(lambda q: q['budget'].update(ceiling_seconds=40 * 3600))
        mutate(lambda q: q['tuning'].update(new_candidates=3))
        mutate(lambda q: q['data_access'].update(TEST134='read'))
        mutate(lambda q: q['families']['MF_VCS_M1']['candidate'].update(cap=30))
        mutate(lambda q: q['bf1_basis']['freeze'].update(sha256='0' * 64))
        mutate(lambda q: q.update(K=[8, 16, 32]))

    def test_budget_numbers_follow_from_the_records(self):
        b = self.p['budget']
        self.assertLessEqual(b['pessimistic_total_gpu_hours'], b['ceiling_gpu_hours'])
        self.assertLessEqual(b['max_concurrent_reservation_gpu_hours'], b['ceiling_gpu_hours'])
        for f, ks in b['per_family_K'].items():
            for k, v in ks.items():
                self.assertGreater(v['time_limit_over_pessimistic'], 1.5, (f, k))
        recorded = {f: C.read(C.BF1 / 'exits' / f"{b['k16_measured_basis']['per_unit'][f]['1001']['job']}.json")['allocated_seconds']
                    for f in C.FAMILIES}
        for f in C.FAMILIES:
            self.assertEqual(b['k16_measured_basis']['per_unit'][f]['1001']['recorded_allocated_seconds'], recorded[f])


class Identity(unittest.TestCase):
    def test_identity_is_bf1_identity_with_capacity_fields(self):
        p = prereg()
        bc = bf1_common()
        for u in p['units']:
            c = p['families'][u['family']]['candidate']
            ident = bc.identity(u['family'], u['seed'], c['lr'], c['cap'], 'FINAL_DEV', dependency=p['families'][u['family']]['selection']['sha256'])
            self.assertEqual(ident['K'], 16)
            ident.update(schema=C.SCHEMA_UNIT, K=u['K'], extension=C.NAME, preregistration_sha256=C.sha(C.BUNDLE / 'PREREGISTRATION.json'),
                         capacity_freeze_sha256='a' * 64, result_scope='CAPACITY_EXTENSION_SEPARATE_TABLE')
            self.assertEqual(ident, C.expected_identity(u, p, 'a' * 64))

    def test_cpu_first_loader_is_bf1_load_checkpoint_with_one_token_changed(self):
        bf1 = function_source(C.BF1_SOURCE / 'scripts' / 'v10_bf1' / 'run.py', 'load_checkpoint')
        ours = function_source(HERE / 'run.py', 'cpu_first_loader', 'load_checkpoint')
        old, new = "map_location=t.device", "map_location='cpu'"
        self.assertEqual(bf1.count(old), 1)
        self.assertEqual(bf1.replace(old, new), ours)

    def test_capacity_config_is_bf1_config_with_k_from_identity(self):
        bf1 = function_source(C.BF1_SOURCE / 'scripts' / 'v10_bf1' / 'run.py', 'config')
        ours = function_source(HERE / 'run.py', 'capacity_config_factory', 'config')
        norm = lambda s: ''.join(s.split()).replace('return', '')
        self.assertEqual(norm(bf1).replace("K=16", "K=ident['K']"), norm(ours))

    def test_bf1_frozen_tree_and_worktree_science_are_byte_identical(self):
        f = C.read(C.BF1 / 'FROZEN.json')
        for name, digest in f['source_files'].items():
            self.assertEqual(C.sha(C.BF1_SOURCE / name), digest, name)
            self.assertEqual(C.sha(C.REPO / name), digest, 'worktree ' + name)


class Admission(unittest.TestCase):
    def run_in(self, fn):
        with tempfile.TemporaryDirectory() as t, patch.object(C, 'BUNDLE', Path(t)):
            fn(Path(t))

    def test_reservations_receipts_duplicates_pause_and_ceiling(self):
        def body(b):
            units = {u['key']: u for u in C.planned_units()}
            m1 = units['MF_VCS_M1_K8_seed1001']; p1 = units['MF_VCS_perm1_K8_seed1001']
            r = C.admit(m1, '1', 10920)
            self.assertEqual(r['decision'], 'ADMITTED')
            self.assertEqual(C.admit(m1, '2', 10920)['decision'], 'DUPLICATE_OR_UNRECONCILED')
            C.write(b / 'exits' / '1.json', {'job': '1', 'unit': m1['key'], 'exit_code': 1, 'allocated_seconds': 4000.0})
            C.write(b / 'exits' / '2.json', {'job': '2', 'unit': m1['key'], 'exit_code': 3, 'allocated_seconds': 20.0})
            r = C.admit(m1, '3', 10920)                            # resume after a receipt: allowed, receipts replace reservations
            self.assertEqual(r['decision'], 'ADMITTED')
            self.assertAlmostEqual(r['committed_other_seconds'], 4020.0)
            for j in range(4, 13):                                 # nine more reservations without receipts
                C.append(b / 'ADMISSIONS.jsonl', {'job': str(j), 'unit': f'other{j}', 'decision': 'ADMITTED', 'reserved_seconds': 10920})
            committed, rows = C.committed_seconds()
            self.assertAlmostEqual(committed, 4020 + 10 * 10920)    # receipts 4000 + 20, job 3 and jobs 4-12 reserved
            r = C.admit(p1, '20', 5520)
            self.assertEqual(r['decision'], 'INCOMPLETE_BUDGET_CEILING')
            self.assertTrue((b / 'budget' / f"{p1['key']}.json").exists())
            for j in ('11', '12'):
                C.write(b / 'exits' / f'{j}.json', {'job': j, 'unit': f'other{j}', 'exit_code': 0, 'allocated_seconds': 1000.0})
            r = C.admit(p1, '21', 5520)                             # 4020 + 8 x 10920 + 2000 + 300 (refused job 20) + 5520 <= ceiling
            self.assertEqual(r['decision'], 'ADMITTED')
            self.assertAlmostEqual(r['committed_other_seconds'], 4020 + 8 * 10920 + 2000 + C.NON_ADMITTED_PROVISIONAL_SECONDS)
            committed, rows = C.committed_seconds()
            self.assertIn('provisional_non_admitted', {x['basis'] for x in rows})
            (b / 'PAUSE').write_text('')
            self.assertEqual(C.admit(units['MF_VCS_M0_K8_seed1001'], '30', 5520)['decision'], 'PAUSED')
            (b / 'PAUSE').unlink()
            self.assertEqual(C.admit(units['MF_VCS_M0_K8_seed1001'], '31', 5520, already_complete=True)['decision'], 'ALREADY_COMPLETE')
        self.run_in(body)

    def test_frozen_limits_fit_the_ceiling_with_realistic_concurrency(self):
        units = C.planned_units()
        m1 = [u for u in units if u['family'] == 'MF_VCS_M1']
        self.assertEqual(sum(u['time_limit_seconds'] for u in units) / 3600, 36.0)
        worst_concurrent = sum(u['time_limit_seconds'] + C.RESERVE_MARGIN_SECONDS for u in m1) + 2 * (5400 + C.RESERVE_MARGIN_SECONDS)
        self.assertLess(worst_concurrent, C.CEILING_SECONDS)


class Resume(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def steps(self, n, partial=False, duplicate=None):
        text = ''.join(json.dumps({'step': s}) + '\n' for s in range(1, n + 1))
        if duplicate:
            text += json.dumps({'step': duplicate}) + '\n'
        if partial:
            text += '{"step": ' + str(n + 1)
        (self.d / 'TRAIN_STEPS.jsonl').write_text(text)

    def checkpoint(self, step):
        C.write(self.d / 'checkpoints' / f'step_{step:04d}.json', {'step': step})

    def test_fresh_directory_is_untouched(self):
        self.assertIsNone(C.prepare_resume(self.d, '1'))

    def test_rows_after_the_checkpoint_and_a_partial_line_are_cut(self):
        self.checkpoint(250); self.checkpoint(500); self.steps(613, partial=True)
        r = C.prepare_resume(self.d, '77')
        self.assertEqual((r['resume_checkpoint_step'], r['kept_rows'], r['dropped_rows'], r['partial_final_line']), (500, 500, 113, True))
        self.assertEqual([x['step'] for x in C.lines(self.d / 'TRAIN_STEPS.jsonl')], list(range(1, 501)))
        self.assertTrue((self.d / 'TRAIN_STEPS.pre_resume_77.jsonl').exists())
        self.assertIsNotNone(C.prepare_resume(self.d, '78'))   # idempotent: nothing more to cut
        self.assertFalse((self.d / 'TRAIN_STEPS.pre_resume_78.jsonl').exists())

    def test_crash_before_the_first_checkpoint_restarts_the_log(self):
        self.steps(120)
        r = C.prepare_resume(self.d, '5')
        self.assertEqual((r['resume_checkpoint_step'], r['kept_rows'], r['dropped_rows']), (0, 0, 120))
        self.assertFalse((self.d / 'TRAIN_STEPS.jsonl').exists())

    def test_a_gap_or_duplicate_before_the_checkpoint_is_refused(self):
        self.checkpoint(250); self.steps(249)
        with self.assertRaises(RuntimeError):
            C.prepare_resume(self.d, '1')
        self.steps(260, duplicate=17)
        with self.assertRaises(RuntimeError):
            C.prepare_resume(self.d, '2')
        (self.d / 'TRAIN_STEPS.jsonl').write_text('{"step": 1}\n{"st\n{"step": 3}\n')
        with self.assertRaises(RuntimeError):
            C.prepare_resume(self.d, '3')


class Audit(unittest.TestCase):
    def test_complete_unit_audit_accepts_a_valid_unit_and_rejects_each_defect(self):
        p = prereg()
        u = next(x for x in p['units'] if x['key'] == 'MF_VCS_M1_K32_seed1002')
        with tempfile.TemporaryDirectory() as t:
            out, bundle = Path(t) / 'out', Path(t) / 'bundle'
            real_prereg = C.BUNDLE / 'PREREGISTRATION.json'
            bundle.mkdir()
            (bundle / 'PREREGISTRATION.json').write_bytes(real_prereg.read_bytes())
            with patch.object(C, 'OUT', out), patch.object(C, 'BUNDLE', bundle):
                ident = C.expected_identity(u, p, 'b' * 64)
                d = make_fake_unit(out, bundle, u, p, ident)
                C.verify_result(u, p, 'b' * 64)

                def snapshot():
                    return {x: x.read_bytes() for root in (d, bundle) for x in root.rglob('*') if x.is_file()}

                def broken(fn, rehash=True):
                    backup = snapshot()
                    try:
                        fn()
                        if rehash:
                            r = C.read(d / 'COMPLETE.json')
                            r['artifacts'] = {k: C.sha(d / k) for k in r['artifacts']}
                            C.write(d / 'COMPLETE.json', r)
                        with self.assertRaises((RuntimeError, KeyError, OSError)):
                            C.verify_result(u, p, 'b' * 64)
                    finally:
                        for x in snapshot():
                            if x not in backup:
                                x.unlink()
                        for x, data in backup.items():
                            x.write_bytes(data)
                    C.verify_result(u, p, 'b' * 64)

                def edit(name, fn):
                    def go():
                        v = C.read(d / name); fn(v); C.write(d / name, v)
                    return go
                broken(lambda: (d / 'TRAIN_STEPS.jsonl').write_text((d / 'TRAIN_STEPS.jsonl').read_text() + '{"step": 3000}\n'))
                broken(edit('CONFIG.json', lambda v: v.update(K=16)))
                broken(edit('ENV_999.json', lambda v: v.update(gpu_name='NVIDIA H100 NVL')))
                broken(edit('ENV_999.json', lambda v: v['precision'].update(cudnn_conv_fp32_precision='ieee')))
                broken(edit('DEV.identity.json', lambda v: v.update(checkpoint_sha256='0' * 64)))
                broken(edit('COMPLETE.json', lambda v: v['identity'].update(K=8)), rehash=False)
                broken(edit('COMPLETE.json', lambda v: v.update(selected=v['checkpoints'][0])), rehash=False)
                broken(lambda: (d / 'checkpoints' / 'step_0750.pt').write_bytes(b'changed'), rehash=False)
                broken(lambda: C.write(bundle / 'failures' / '999.json', {'error': 'x'}), rehash=False)
                broken(edit('DATA_READS.json', lambda v: v.append({'role': 'TEST'})))
                broken(lambda: (bundle / 'exits' / '999.json').unlink(), rehash=False)
                sel = C.read(d / 'COMPLETE.json')['selected']['checkpoint_sha256']
                broken(lambda: (d / 'DEV.jsonl').write_text((d / 'DEV.jsonl').read_text().replace(sel, '1' * 64, 1)))
                broken(lambda: (d / 'CAL' / 'step_0500.jsonl').write_text(''.join((d / 'CAL' / 'step_0500.jsonl').read_text().splitlines(True)[:-1])))
                broken(edit('ENV_999.json', lambda v: v.update(host='node51')))
                broken(edit('COMPLETE.json', lambda v: v.update(DEV24_cores=v['DEV24_cores'][:-1] + ['1B_F4'])), rehash=False)


class SlurmContract(unittest.TestCase):
    def test_gpu_sbatch(self):
        text = SBATCH.read_text()
        for line in ('#SBATCH --partition=L40S', '#SBATCH --qos=normal', '#SBATCH --exclude=node51,node60', '#SBATCH --gres=gpu:1',
                     '#SBATCH --no-requeue', '#SBATCH --signal=B:USR1@60', 'unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE NVIDIA_TF32_OVERRIDE PYTHONPATH',
                     'CUBLAS_WORKSPACE_CONFIG=:4096:8', "squeue -j \"$SLURM_JOB_ID\" -h -o '%S|%l'", 'receipt.py'):
            self.assertIn(line, text)
        self.assertNotIn('#SBATCH --time', text)
        self.assertNotIn('#SBATCH --job-name', text)
        for path in (SBATCH, CPU_SBATCH):
            self.assertEqual(subprocess.run(['bash', '-n', str(path)]).returncode, 0)
        cpu = CPU_SBATCH.read_text()
        self.assertIn('#SBATCH --partition=cpu-high', cpu)
        self.assertNotIn('--gres', cpu)


if __name__ == '__main__':
    unittest.main(verbosity=2)
