"""Control checks of the 2026-10-19 hardware amendment, no Torch (login-node safe): record, scope, the exact GPU / partition / node
rule, the declared source deltas against the frozen base, byte identity of the science code, identities, the amended unit audit,
the stratified table, shared admission, and the Slurm contract.

  python check_hw.py            run the checks
  python check_hw.py --record   run them and write hw_amendment_20261019/CHECKS.json for the current amendment freeze
"""
import ast
import copy
import difflib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import capacity_common as C
import capacity_table as T
import capacity_table_hw as TH
import check_control as CC
import hw_amendment as H

HERE = Path(__file__).resolve().parent
FROZEN_TREE = '/home/infres/yinwang/CS_QMI/ssl_pilot_v10_capacity_hw_20261019'

# The complete normalised line delta (ast.unparse, docstrings removed, R./C. module prefixes stripped) of the two functions the
# amendment re-implements.  Any other change fails test_declared_source_deltas.
RUN_MAIN_DELTA = {
    'removed': [
        "    unit = C.unit_by_key(prereg, key)",
        "    C.require(os.environ.get('SLURM_JOB_PARTITION') == 'L40S', 'Pre-registered hardware is the L40S partition')",
        "        directory = C.OUT / unit['key']",
        "            C.verify_result(unit, prereg, freeze_sha, require_receipt=False)",
        "        env = bf1.runtime_check('cuda')",
        "        C.require(env['git_commit'] == C.BF1_COMMIT and env['git_dirty_code'] == '' and (env['code_hash_src_scripts_v10'] == C.CORE_HASH) and (env['host'] not in C.ALLOWED_NODES_EXCLUDED), 'Frozen BF1 tree or node is not the pre-registered one')",
        "        C.write(C.BUNDLE / 'runtime' / f'{job}.json', {'utc': C.utc(), 'job': job, 'unit': unit, 'identity': ident, 'train_config': resolved, 'environment': env, 'admission': admission, 'resume': resume, 'capacity_commit': git_head(C.REPO), 'capacity_freeze_sha256': freeze_sha, 'patches': ['run.config: K from identity', \"run.load_checkpoint: map_location='cpu'\"]})",
        "        allocation.record('RUNTIME_VERIFIED', gpu=env['gpu_name'], node=env['host'], resume=resume)",
        "        C.verify_result(unit, prereg, freeze_sha, require_receipt=False)"],
    'added': [
        "    hw_freeze_sha = os.environ.get('V10_CAPACITY_HW_FREEZE')",
        "    record = H.validate_record(H.frozen_hw(hw_freeze_sha), prereg)",
        "    H.checks_passed_hw(hw_freeze_sha)",
        "    unit = H.hw_unit(prereg, key)",
        "    partition = os.environ.get('SLURM_JOB_PARTITION')",
        "    C.require(partition in H.GPUS, 'Amended hardware is one of the partitions ' + H.PARTITIONS)",
        "        directory = H.OUT_HW / unit['key']",
        "        H.base_untouched(unit)",
        "        gate = H.gate_passed(partition, hw_freeze_sha)",
        "            H.verify_result(unit, prereg, freeze_sha, require_receipt=False, hw_freeze_sha=hw_freeze_sha)",
        "        env = H.runtime_environment(bf1)",
        "        C.require(env['git_commit'] == C.BF1_COMMIT and env['git_dirty_code'] == '' and (env['code_hash_src_scripts_v10'] == C.CORE_HASH) and (env['host'] not in H.EXCLUDED), 'Frozen BF1 tree or node is not the pre-registered one')",
        "        ident.update(schema=H.SCHEMA_UNIT, hardware_amendment_sha256=C.sha(H.DIR / 'HARDWARE_AMENDMENT.json'), capacity_hw_freeze_sha256=hw_freeze_sha)",
        "        C.require(ident == H.expected_identity_hw(unit, prereg, freeze_sha, hw_freeze_sha), 'Identity differs from the amendment reconstruction')",
        "        C.write(C.BUNDLE / 'runtime' / f'{job}.json', {'utc': C.utc(), 'job': job, 'unit': unit, 'identity': ident, 'train_config': resolved, 'environment': env, 'admission': admission, 'resume': resume, 'capacity_commit': git_head(C.REPO), 'capacity_freeze_sha256': freeze_sha, 'patches': ['run.config: K from identity', \"run.load_checkpoint: map_location='cpu'\"], 'capacity_hw_freeze_sha256': hw_freeze_sha, 'hardware_amendment': record['name'], 'gpu_gate': gate, 'gpu_name': env['gpu_name'], 'partition': env['slurm_partition'], 'node': env['host']})",
        "        C.write(H.DIR / 'jobs' / f'{job}.json', {'utc': C.utc(), 'job': job, 'unit': unit['key'], 'gpu_name': env['gpu_name'], 'partition': env['slurm_partition'], 'node': env['host'], 'hw_freeze_sha256': hw_freeze_sha})",
        "        allocation.record('RUNTIME_VERIFIED', gpu=env['gpu_name'], node=env['host'], resume=resume, partition=env['slurm_partition'])",
        "        H.verify_result(unit, prereg, freeze_sha, require_receipt=False, hw_freeze_sha=hw_freeze_sha)"]}
VERIFY_DELTA = {
    'removed': [
        "def verify_result(unit, prereg, capacity_freeze_sha, require_receipt=True):",
        "    d = OUT / unit['key']",
        "    ident = expected_identity(unit, prereg, capacity_freeze_sha)",
        "        require(e['gpu_name'] == 'NVIDIA L40S' and e['slurm_partition'] == 'L40S' and (e['host'] not in ALLOWED_NODES_EXCLUDED), 'Unit ran outside the pre-registered L40S hardware')"],
    'added': [
        "def verify_result(unit, prereg, capacity_freeze_sha, require_receipt=True, *, hw_freeze_sha):",
        "    d = OUT_HW / unit['key']",
        "    ident = expected_identity_hw(unit, prereg, capacity_freeze_sha, hw_freeze_sha)",
        "        require(hardware_problem(e) is None, 'Unit ran outside the amended hardware: ' + str(hardware_problem(e)))"]}
SBATCH_DELTA_CODE_LINES = {
    'removed': ['#SBATCH --partition=L40S', '#SBATCH --exclude=node51,node60',
                ': "${V10_CAPACITY_FREEZE:?sha256 of the bundle FREEZE.json is required (--export=ALL,V10_CAPACITY_FREEZE=...)}"',
                'WT=/home/infres/yinwang/CS_QMI/ssl_pilot_v10_capacity',
                'OUT=/projects/EEG-foundation-model/yinghao/outputs_v10/capacity_ext_20261019',
                '"$PYTHON" -u "$WT/scripts/v10_capacity/run.py" "${1:?unit key}" &'],
    'added': ['#SBATCH --partition=RTX6000PRO,L40S,H100', '#SBATCH --exclude=node51,node53,node60',
              ': "${V10_CAPACITY_FREEZE:?sha256 of the base FREEZE.json is required (--export=ALL,V10_CAPACITY_FREEZE=...)}"',
              ': "${V10_CAPACITY_HW_FREEZE:?sha256 of the amendment FREEZE.json is required (--export=ALL,...,V10_CAPACITY_HW_FREEZE=...)}"',
              'WT=' + FROZEN_TREE, 'OUT=/projects/EEG-foundation-model/yinghao/outputs_v10/capacity_ext_hw_20261019',
              '"$PYTHON" -u "$WT/scripts/v10_capacity/run_hw.py" "${1:?unit key}" &']}


def function_source(path, name, strip=()):
    tree = ast.parse(Path(path).read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    if ast.get_docstring(fn) is not None:
        fn.body = fn.body[1:]

    class Strip(ast.NodeTransformer):
        def visit_Attribute(self, node):
            self.generic_visit(node)
            if isinstance(node.value, ast.Name) and node.value.id in strip:
                return ast.copy_location(ast.Name(id=node.attr, ctx=node.ctx), node)
            return node
    return ast.unparse(Strip().visit(fn))


def line_delta(a_text, b_text, code_only=False):
    keep = (lambda s: s.strip() and not s.lstrip().startswith('#') or s.startswith('#SBATCH')) if code_only else (lambda s: True)
    a = [s for s in a_text.splitlines() if keep(s)]
    b = [s for s in b_text.splitlines() if keep(s)]
    diff = list(difflib.unified_diff(a, b, lineterm='', n=0))
    return {'removed': [s[1:] for s in diff if s.startswith('-') and not s.startswith('---')],
            'added': [s[1:] for s in diff if s.startswith('+') and not s.startswith('+++')]}


def source_deltas():
    return {'run.main -> run_hw.main': line_delta(function_source(HERE / 'run.py', 'main'),
                                                   function_source(HERE / 'run_hw.py', 'main', strip=('R',))),
            'capacity_common.verify_result -> hw_amendment.verify_result': line_delta(
                function_source(HERE / 'capacity_common.py', 'verify_result'), function_source(HERE / 'hw_amendment.py', 'verify_result', strip=('C',))),
            'run.sbatch -> run_hw.sbatch (code lines)': line_delta((C.REPO / 'slurm/v10_capacity/run.sbatch').read_text(),
                                                                   (C.REPO / 'slurm/v10_capacity/run_hw.sbatch').read_text(), code_only=True)}


def record():
    return C.read(H.DIR / 'HARDWARE_AMENDMENT.json')


def prereg():
    return C.read(C.BUNDLE / 'PREREGISTRATION.json')


class Record(unittest.TestCase):
    def test_record_cites_the_frozen_base_and_matches_the_code(self):
        r, p = record(), prereg()
        H.validate_record(r, p)
        self.assertEqual(C.sha(C.BUNDLE / 'FREEZE.json'), H.BASE_FREEZE)
        self.assertEqual(r['amends']['freeze']['sha256'], 'e4157194c5421e406c5abdc3b0decd733c0e1dcb554db1dc6d5def0c39a8023b')
        self.assertEqual(r['authority']['owner_choice_verbatim'], '「final 放开，CAL 留 L40S」')
        self.assertIn('--partition=RTX6000PRO,L40S,H100', r['authority']['standing_instruction_verbatim'])
        self.assertEqual(r['scientific_changes'], [])
        self.assertEqual(C.sha(C.REPO / H.RECORD_GIT_PATH), C.sha(H.DIR / 'HARDWARE_AMENDMENT.json'))

    def test_changes_are_rejected(self):
        r, p = record(), prereg()

        def mutate(fn):
            q = copy.deepcopy(r); fn(q)
            with self.assertRaises((RuntimeError, KeyError)):
                H.validate_record(q, p)
        mutate(lambda q: q['hardware']['accepted_gpus'].update(H100='NVIDIA H100'))
        mutate(lambda q: q['hardware'].update(exclude=['node51', 'node60']))
        mutate(lambda q: q['hardware'].update(partition='L40S'))
        mutate(lambda q: q['scope']['units'].append({'key': 'MF_VCS_M1_K32_seed1001', 'job_name': 'v10k_m1_k32_1001', 'time_limit_seconds': 10800}))
        mutate(lambda q: q['scope']['units'][4].update(time_limit_seconds=86400))
        mutate(lambda q: q['budget'].update(ceiling_seconds=40 * 3600))
        mutate(lambda q: q['scientific_changes'].append('K16 cap 30'))
        mutate(lambda q: q['amends']['freeze'].update(sha256='0' * 64))
        mutate(lambda q: q['outputs'].update(root=str(C.OUT)))

    def test_scope_is_the_six_unstarted_seed_1003_units(self):
        p = prereg()
        units = H.hw_units(p)
        self.assertEqual([u['key'] for u in units], [C.unit_key(f, K, 1003) for f in C.FAMILIES for K in C.KS])
        self.assertEqual([(u['job_name'], u['time_limit']) for u in units],
                         [('v10k_p1_k8_1003', '01:30:00'), ('v10k_p1_k32_1003', '01:30:00'), ('v10k_m0_k8_1003', '01:30:00'),
                          ('v10k_m0_k32_1003', '01:30:00'), ('v10k_m1_k8_1003', '03:00:00'), ('v10k_m1_k32_1003', '03:00:00')])
        for u in units:
            self.assertFalse((C.OUT / u['key']).exists(), u['key'])
        with self.assertRaises(RuntimeError):
            H.hw_unit(p, 'MF_VCS_M1_K32_seed1001')
        r = record()
        self.assertEqual(len(r['scope']['completed_base_units']), 12)


class Hardware(unittest.TestCase):
    def test_exact_names_partition_pairs_and_excluded_nodes(self):
        ok = [('RTX6000PRO', 'NVIDIA RTX PRO 6000 Blackwell Server Edition', 'node58'), ('L40S', 'NVIDIA L40S', 'node52'),
              ('H100', 'NVIDIA H100 NVL', 'nodesumo01')]
        for part, gpu, host in ok:
            self.assertIsNone(H.hardware_problem({'slurm_partition': part, 'gpu_name': gpu, 'host': host}))
        bad = [('H100', 'NVIDIA H100 NVL', 'node53'), ('L40S', 'NVIDIA L40S', 'node51'), ('RTX6000PRO', 'NVIDIA RTX PRO 6000 Blackwell Server Edition', 'node60'),
               ('H100', 'NVIDIA H100 80GB HBM3', 'nodesumo01'), ('RTX6000PRO', 'NVIDIA RTX PRO 6000', 'node58'),
               ('L40S', 'NVIDIA H100 NVL', 'node52'), ('A100', 'NVIDIA A100-SXM4-80GB', 'node01'), (None, 'NVIDIA L40S', 'node52')]
        for part, gpu, host in bad:
            self.assertIsNotNone(H.hardware_problem({'slurm_partition': part, 'gpu_name': gpu, 'host': host}), (part, gpu, host))

    def test_gate_requirement(self):
        with tempfile.TemporaryDirectory() as t, patch.object(H, 'DIR', Path(t)):
            self.assertIsNone(H.gate_passed('L40S', 'x'))
            for part in ('RTX6000PRO', 'H100'):
                with self.assertRaises(C.Refused):
                    H.gate_passed(part, 'x')
            C.write(Path(t) / 'gpu_gates' / 'RTX6000PRO_1.json', {'status': 'PASS', 'hw_freeze_sha256': 'y', 'partition': 'RTX6000PRO',
                                                                  'gpu_name': H.GPUS['RTX6000PRO']})
            with self.assertRaises(C.Refused):
                H.gate_passed('RTX6000PRO', 'x')
            C.write(Path(t) / 'gpu_gates' / 'RTX6000PRO_2.json', {'status': 'PASS', 'hw_freeze_sha256': 'x', 'partition': 'RTX6000PRO',
                                                                  'gpu_name': H.GPUS['RTX6000PRO']})
            self.assertEqual(H.gate_passed('RTX6000PRO', 'x')['partition'], 'RTX6000PRO')
            with self.assertRaises(C.Refused):
                H.gate_passed('H100', 'x')


class Delta(unittest.TestCase):
    def test_declared_source_deltas(self):
        d = source_deltas()
        self.assertEqual(d['run.main -> run_hw.main'], RUN_MAIN_DELTA)
        self.assertEqual(d['capacity_common.verify_result -> hw_amendment.verify_result'], VERIFY_DELTA)
        self.assertEqual(d['run.sbatch -> run_hw.sbatch (code lines)'], SBATCH_DELTA_CODE_LINES)

    def test_science_and_base_code_byte_identical(self):
        f = C.read(C.BF1 / 'FROZEN.json')
        for name, digest in f['source_files'].items():
            self.assertEqual(C.sha(C.BF1_SOURCE / name), digest, name)
        base = C.read(C.BUNDLE / 'FREEZE.json')
        for name, digest in base['code_files'].items():
            self.assertEqual(C.sha(C.REPO / name), digest, name)
        self.assertEqual(C.sha(C.REPO / C.PREREG_GIT_PATH), H.BASE_PREREG)

    def test_identity_is_the_base_identity_plus_amendment_fields(self):
        p = prereg()
        for u in H.hw_units(p):
            base = C.expected_identity(u, p, H.BASE_FREEZE)
            hw = H.expected_identity_hw(u, p, H.BASE_FREEZE, 'h' * 64)
            self.assertEqual({k: v for k, v in hw.items() if k not in ('schema', 'hardware_amendment_sha256', 'capacity_hw_freeze_sha256')},
                             {k: v for k, v in base.items() if k != 'schema'})
            self.assertEqual((hw['schema'], hw['capacity_hw_freeze_sha256']), (H.SCHEMA_UNIT, 'h' * 64))
            self.assertNotIn('gpu', ' '.join(hw))


class Audit(unittest.TestCase):
    def make(self, t):
        p = prereg()
        out, bundle = Path(t) / 'out', Path(t) / 'bundle'
        bundle.mkdir(exist_ok=True)
        (bundle / 'PREREGISTRATION.json').write_bytes((C.BUNDLE / 'PREREGISTRATION.json').read_bytes())
        (bundle / 'hw').mkdir(exist_ok=True)
        (bundle / 'hw' / 'HARDWARE_AMENDMENT.json').write_bytes((H.DIR / 'HARDWARE_AMENDMENT.json').read_bytes())
        return p, out, bundle

    def test_amended_audit_accepts_each_type_and_rejects_defects(self):
        p = prereg()
        u = C.unit_by_key(p, 'MF_VCS_M1_K32_seed1003')
        for part, gpu in H.GPUS.items():
            with tempfile.TemporaryDirectory() as t:
                _, out, bundle = self.make(t)
                with patch.object(C, 'BUNDLE', bundle), patch.object(H, 'DIR', bundle / 'hw'), patch.object(H, 'OUT_HW', out):
                    ident = H.expected_identity_hw(u, p, H.BASE_FREEZE, 'h' * 64)
                    d = CC.make_fake_unit(out, bundle, u, p, ident)
                    env = C.read(d / 'ENV_999.json')
                    env.update(gpu_name=gpu, slurm_partition=part, host='node58' if part == 'RTX6000PRO' else 'nodesumo01' if part == 'H100' else 'node52')
                    C.write(d / 'ENV_999.json', env)
                    r = C.read(d / 'COMPLETE.json')
                    r['artifacts']['ENV_999.json'] = C.sha(d / 'ENV_999.json')
                    C.write(d / 'COMPLETE.json', r)
                    H.verify_result(u, p, H.BASE_FREEZE, hw_freeze_sha='h' * 64)
                    self.assertEqual(H.hardware_of(d, r['artifacts'])[0][0], gpu)
                    for change in ({'host': 'node53'}, {'gpu_name': 'NVIDIA A100-SXM4-80GB'}, {'slurm_partition': 'L40S' if part != 'L40S' else 'H100'}):
                        bad = {**env, **change}
                        C.write(d / 'ENV_999.json', bad)
                        r['artifacts']['ENV_999.json'] = C.sha(d / 'ENV_999.json')
                        C.write(d / 'COMPLETE.json', r)
                        with self.assertRaises(RuntimeError):
                            H.verify_result(u, p, H.BASE_FREEZE, hw_freeze_sha='h' * 64)
                    C.write(d / 'ENV_999.json', env)
                    r['artifacts']['ENV_999.json'] = C.sha(d / 'ENV_999.json')
                    C.write(d / 'COMPLETE.json', r)
                    with self.assertRaises(RuntimeError):                     # wrong amendment freeze in the identity
                        H.verify_result(u, p, H.BASE_FREEZE, hw_freeze_sha='z' * 64)
                    with patch.object(C, 'OUT', out), self.assertRaises(RuntimeError):   # the base verifier never accepts it
                        C.verify_result(u, p, H.BASE_FREEZE)

    def test_the_twelve_completed_base_units_verify_unchanged(self):
        p = prereg()
        done = [u for u in p['units'] if u['key'] not in H.UNIT_KEYS]
        self.assertEqual(len(done), 12)
        for u in done:
            r = C.verify_result(u, p, H.BASE_FREEZE)
            self.assertEqual({e[0] for e in H.hardware_of(C.OUT / u['key'], r['artifacts'])}, {'NVIDIA L40S'})


class Table(unittest.TestCase):
    def test_pooled_and_strata_on_mocked_units(self):
        p = prereg()
        gpus = {1001: 'NVIDIA L40S', 1002: 'NVIDIA L40S', 1003: H.GPUS['RTX6000PRO']}

        def fake_row(u, prereg_, base_freeze, hw_freeze):
            ref, ref_dir = T.k16_reference(p, u['family'], u['seed'])
            r = copy.deepcopy(ref)
            if u['seed'] == 1003:
                c = sorted(r['per_core'])[0]
                r['per_core'][c]['success42'] -= 0.1
                r['DEV32']['success42'] -= 0.1 / 32
                if c in r['DEV24_cores']:
                    r['DEV24']['success42'] -= 0.1 / 24
            row = {'seed': u['seed'], 'unit': u['key'], 'stratum': gpus[u['seed']], 'hardware': [(gpus[u['seed']], 'x', 'nodeX')],
                   '_per_core': r['per_core'], '_per_core_k16': ref['per_core'], '_dev24': r['DEV24_cores']}
            for s in TH.SUBSETS:
                row[s], row[s + '_k16'] = r[s], ref[s]
                row[s + '_delta'] = {m: r[s][m] - ref[s][m] for m in ('success42', 'success9', 'median_err')}
            return row
        with patch.object(TH, 'unit_row', fake_row):
            cells = TH.build(p, H.BASE_FREEZE, 'h' * 64)
        self.assertEqual(len(cells), 6)
        for c in cells:
            self.assertEqual(c['status'], 'COMPLETE')
            self.assertEqual(set(c['strata']), {'NVIDIA L40S', H.GPUS['RTX6000PRO']})
            self.assertEqual(c['strata']['NVIDIA L40S']['seeds'], [1001, 1002])
            self.assertEqual(c['strata'][H.GPUS['RTX6000PRO']]['DEV32']['success42']['grid'], 'single_seed_descriptive')
            self.assertEqual(c['pooled']['seeds'], [1001, 1002, 1003])
            self.assertAlmostEqual(c['pooled']['DEV32']['success42']['delta_per_seed'][2], -0.1 / 32)
            self.assertEqual(c['strata']['NVIDIA L40S']['DEV32']['success42']['delta_mean'], 0.0)


class Admission(unittest.TestCase):
    def test_one_ceiling_and_cross_freeze_duplicates(self):
        p = prereg()
        u = C.unit_by_key(p, 'MF_VCS_M1_K32_seed1003')
        with tempfile.TemporaryDirectory() as t, patch.object(C, 'BUNDLE', Path(t)):
            for p_ in (Path('/home/infres/yinwang/CS_QMI/handoff/v10_capacity_ext_20261019/exits')).glob('*.json'):
                (Path(t) / 'exits').mkdir(exist_ok=True)
                (Path(t) / 'exits' / p_.name).write_bytes(p_.read_bytes())
            committed, _ = C.committed_seconds()
            self.assertAlmostEqual(committed, sum(C.read(x)['allocated_seconds'] for x in (Path(t) / 'exits').glob('*.json')))
            r = C.admit(u, 'base_job', 10920)                         # a base-freeze job of the same unit, still running
            self.assertEqual(r['decision'], 'ADMITTED')
            self.assertEqual(C.admit(u, 'hw_job', 10920)['decision'], 'DUPLICATE_OR_UNRECONCILED')
            C.write(Path(t) / 'exits' / 'base_job.json', {'job': 'base_job', 'unit': u['key'], 'exit_code': 1, 'allocated_seconds': 60.0})
            r = C.admit(u, 'hw_job2', 10920)
            self.assertEqual(r['decision'], 'ADMITTED')
            self.assertLessEqual(r['committed_other_seconds'] + 10920, C.CEILING_SECONDS)


class SlurmContract(unittest.TestCase):
    def test_sbatch_files(self):
        run = (C.REPO / 'slurm/v10_capacity/run_hw.sbatch').read_text()
        gate = (C.REPO / 'slurm/v10_capacity/hw_gate.sbatch').read_text()
        for line in ('#SBATCH --partition=RTX6000PRO,L40S,H100', '#SBATCH --exclude=node51,node53,node60', '#SBATCH --gres=gpu:1',
                     '#SBATCH --no-requeue', '#SBATCH --signal=B:USR1@60', 'WT=' + FROZEN_TREE, 'receipt.py',
                     'unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE NVIDIA_TF32_OVERRIDE PYTHONPATH', 'CUBLAS_WORKSPACE_CONFIG=:4096:8'):
            self.assertIn(line, run)
        for line in ('#SBATCH --exclude=node51,node53,node60', '#SBATCH --gres=gpu:1', 'WT=' + FROZEN_TREE, 'hw_gate.py',
                     'unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE NVIDIA_TF32_OVERRIDE PYTHONPATH', 'CUBLAS_WORKSPACE_CONFIG=:4096:8'):
            self.assertIn(line, gate)
        self.assertNotIn('#SBATCH --time', run)
        self.assertNotIn('#SBATCH --job-name', run)
        self.assertNotIn('#SBATCH --partition', gate)
        for path in ('slurm/v10_capacity/run_hw.sbatch', 'slurm/v10_capacity/hw_gate.sbatch'):
            self.assertEqual(subprocess.run(['bash', '-n', str(C.REPO / path)]).returncode, 0)


def record_checks():
    hw_freeze = C.sha(H.DIR / 'FREEZE.json')
    H.validate_record(H.frozen_hw(hw_freeze), C.validate_prereg(C.frozen(H.BASE_FREEZE)))
    C.require(str(C.REPO) == FROZEN_TREE, 'Record the checks from the frozen tree')
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    with tempfile.TemporaryFile('w+') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        stream.seek(0)
        log = stream.read()
    print(log)
    ok = result.wasSuccessful() and C.sha(H.DIR / 'FREEZE.json') == hw_freeze
    (H.DIR / 'check_hw.log').write_text(log)
    C.write(H.DIR / 'CHECKS.json', {'status': 'PASS' if ok else 'FAIL', 'utc': C.utc(), 'hw_freeze_sha256': hw_freeze,
            'base_freeze_sha256': H.BASE_FREEZE, 'tests_run': result.testsRun, 'failures': len(result.failures),
            'errors': len(result.errors), 'log': str(H.DIR / 'check_hw.log'), 'log_sha256': C.sha(H.DIR / 'check_hw.log'),
            'executed_on': 'login node (no Torch, single thread); the coordinator asked for no job submission',
            'scope': 'control checks only; the numerical K8/K32 gates on RTX6000PRO / H100 are hw_gate.py GPU jobs'})
    print('CHECKS', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    if sys.argv[1:] == ['--record']:
        raise SystemExit(record_checks())
    unittest.main(verbosity=2)
