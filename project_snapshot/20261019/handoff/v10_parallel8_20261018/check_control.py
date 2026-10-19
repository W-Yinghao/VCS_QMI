"""Control-plane tests only; no Torch, GPU or scientific result simulation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dispatch as D


class DispatchChecks(unittest.TestCase):
    def test_exact_work_matrix(self):
        rows = D.units()
        self.assertEqual(len(rows), 16)
        self.assertEqual(len({u['key'] for u in rows}), 16)
        self.assertEqual(sum(u['phase'] == 'cal' for u in rows), 4)
        self.assertEqual(sum(len(D.candidates(u['family'])) for u in rows if u['phase'] == 'cal'), 18)
        for family in D.FAMILIES:
            self.assertEqual(sorted(u['seed'] for u in rows if u['phase'] == 'final' and u['family'] == family),
                             [1001, 1002, 1003])

    def test_final_requires_successfully_verified_own_cal(self):
        ledger = {'units': D.units()}
        final = ledger['units'][4]
        self.assertFalse(D.eligible(final, ledger))
        ledger['units'][0]['status'] = 'SUBMITTED'
        self.assertFalse(D.eligible(final, ledger))
        ledger['units'][1]['status'] = 'COMPLETE_VERIFIED'
        self.assertFalse(D.eligible(final, ledger))
        ledger['units'][0]['status'] = 'COMPLETE_VERIFIED'
        self.assertTrue(D.eligible(final, ledger))

    def test_no_duplicates_or_further_submissions_after_family_failure(self):
        ledger = {'units': D.units()}
        ledger['units'][0]['status'] = 'COMPLETE_VERIFIED'
        final = ledger['units'][4]
        final['status'] = 'SUBMITTED'
        self.assertFalse(D.eligible(final, ledger))
        final['status'] = 'FAILED_REVIEW_REQUIRED'
        self.assertFalse(D.eligible(ledger['units'][8], ledger))
        self.assertTrue(D.eligible(ledger['units'][1], ledger))

    def test_capacity_counts_array_tasks_and_other_normal_jobs(self):
        other = [dict(id=str(i), name='other', state='RUNNING', qos='normal') for i in range(29)]
        self.assertFalse(D.capacity(other))
        self.assertTrue(D.capacity(other[:28]))
        pending = [dict(id=f'123_{i}', name=f'v10p8_f{i}', state='PENDING', qos='normal') for i in range(8)]
        self.assertFalse(D.capacity(pending))
        running = [dict(id=str(i), name=f'v10p8_f{i}', state='RUNNING', qos='normal') for i in range(8)]
        self.assertTrue(D.capacity(running + pending[:7]))
        self.assertFalse(D.capacity(running + pending))
        runfill = [dict(id=str(i), name='other', state='RUNNING', qos='runfill') for i in range(40)]
        self.assertTrue(D.capacity(runfill))

    def test_submitted_job_disappearing_is_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = {'units': D.units()}
            unit = ledger['units'][0]
            unit.update(status='SUBMITTED', job='123', first_absent_unix=0)
            with patch.object(D, 'V10', Path(tmp)), patch.object(D, 'BUNDLE', Path(tmp)):
                D.observe(ledger, [dict(id=D.GATE, state='PENDING')])
            self.assertEqual(unit['status'], 'FAILED_REVIEW_REQUIRED')
            self.assertFalse(D.eligible(ledger['units'][4], ledger))

    def test_zero_exit_without_complete_end_is_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            D.write(base / 'exits/123.json', {'job_id': '123', 'exit_code': 0})
            ledger = {'units': D.units()}
            unit = ledger['units'][0]
            unit.update(status='SUBMITTED', job='123')
            with patch.object(D, 'V10', base), patch.object(D, 'BUNDLE', base):
                D.observe(ledger, [dict(id=D.GATE, state='PENDING')])
            self.assertEqual(unit['status'], 'FAILED_REVIEW_REQUIRED')

    def test_gate_identity_and_exit_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            D.write(base / 'GPU_GATE.json', {'status': 'PASS', 'freeze_sha256': D.SCIENCE_FREEZE, 'job_id': D.GATE})
            D.write(base / f'exits/{D.GATE}.json', {'exit_code': 1})
            with patch.object(D, 'V10', base), self.assertRaises(RuntimeError):
                D.verify_gate()


if __name__ == '__main__':
    unittest.main(verbosity=2)
