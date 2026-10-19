"""Metadata-only software tests. No synthetic experiment results or GPU work."""
import json
from pathlib import Path
import tempfile
import unittest

import bridge as b
from activate import build_commands


class HandoffChecks(unittest.TestCase):
    def setUp(self):
        self.groups = b.read(b.V9 / 'GROUPS.json')
        self.launch = {g['id']: {'job': str(2000 + i)} for i, g in enumerate(self.groups)}
        self.done = list(self.launch)
        self.finished = {'failed_groups': [], 'full_complete': 33}

    def test_empty_queue_does_not_mean_complete(self):
        self.assertFalse(b.readiness([], self.groups, self.launch, self.done[:-1], self.finished)['ready'])
        self.assertFalse(b.readiness([], self.groups, self.launch, self.done, None)['ready'])
        self.assertFalse(b.readiness([], self.groups, self.launch, self.done, {'failed_groups': ['902_2'], 'full_complete': 32})['ready'])

    def test_latest_retry_job_blocks_even_with_done_markers(self):
        self.launch['902_2']['job'] = '9000'
        live = [{'id': '9000', 'name': 'resumed_group', 'state': 'RUNNING'}]
        self.assertFalse(b.readiness(live, self.groups, self.launch, self.done, self.finished)['ready'])
        self.assertTrue(b.readiness([], self.groups, self.launch, self.done, self.finished)['ready'])

    def test_arrays_are_counted_as_separate_rows(self):
        rows = b.parse_queue('100_0|v10bf_full|PENDING|normal|gres/gpu:1|N/A|0:00|Dependency\n100_1|v10bf_full|PENDING|normal|gres/gpu:1|N/A|0:00|Dependency\n')
        self.assertEqual(len(rows), 2)
        with self.assertRaises(RuntimeError):
            b.parse_queue('broken response')

    def test_new_dependencies_do_not_reference_cancelled_jobs(self):
        gate, array = build_commands('7000', '7001')
        self.assertIn('--dependency=afterok:7000', gate)
        self.assertIn('--dependency=afterok:7001', array)
        self.assertIn('--array=0-3%2', array)
        self.assertFalse(any('1044420' in s or '1044424' in s for s in gate + array))

    def fixture(self, path):
        group = {'id': '901_1', 'seed': 901}
        unit = {'method': 'VCS_Weight', 'scale': 'sum'}
        ident = {**group, 'group': group['id'], **unit, 'epochs': 40, 'freeze_sha256': b.V9_FREEZE}
        log = {'epochs': [{'epoch': i, 'seconds': i + 1, 'dev': {'rsum': 10 if i >= 15 else 1}} for i in range(40)], 'steps': []}
        b.write(path / 'identity.json', ident)
        b.write(path / 'run_manifest.json', {'fixture': 'metadata test only'})
        b.write(path / 'train_log.json', log)
        (path / 'best.pt').write_bytes(b'tiny validation fixture')
        (path / 'last.pt').write_bytes(b'tiny validation fixture')
        hashes = {n: b.sha(path / n) for n in ('best.pt','last.pt','train_log.json','run_manifest.json','identity.json')}
        result = {'identity': ident, 'full_epochs': 40, 'test_read': False, 'artifact_sha256': hashes,
                  'done': {'best': {'epoch': 15, 'rsum': 10}, 'seconds': 40, 'manifest_sha256': hashes['run_manifest.json']},
                  'hardware': {'gpu': 'test', 'host': 'test'}}
        b.write(path / 'full_result.json', result)
        return group, unit, result

    def test_corrupt_checkpoint_rejects_handoff(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            group, unit, result = self.fixture(path)
            self.assertEqual(b.verify_unit(path, group, unit)['best_epoch'], 15)
            (path / 'last.pt').write_bytes(b'corrupt')
            with self.assertRaisesRegex(RuntimeError, 'Artifact hash mismatch'):
                b.verify_unit(path, group, unit)

    def test_late_tie_checkpoint_rejects_handoff(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            group, unit, result = self.fixture(path)
            result['done']['best']['epoch'] = 39
            b.write(path / 'full_result.json', result)
            with self.assertRaisesRegex(RuntimeError, 'Checkpoint selection mismatch'):
                b.verify_unit(path, group, unit)

    def test_resume_at_finished_checkpoint_counts_previous_segment(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            group, unit, result = self.fixture(path)
            log = b.read(path / 'train_log.json')
            log['resumed_at'] = [40]
            b.write(path / 'train_log.json', log)
            result['artifact_sha256']['train_log.json'] = b.sha(path / 'train_log.json')
            result['done']['seconds'] = 0.1
            b.write(path / 'full_result.json', result)
            self.assertAlmostEqual(b.verify_unit(path, group, unit)['measured_training_seconds'], 40.1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
