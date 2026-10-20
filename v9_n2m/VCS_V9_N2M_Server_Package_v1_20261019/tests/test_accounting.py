import tempfile
import unittest
from pathlib import Path

from v9n2m.accounting import parse_sacct, collect_accounting
from v9n2m.common import write_json


class AccountingTests(unittest.TestCase):
    def test_parent_allocation_not_batch_extern(self):
        raw="10|COMPLETED|3600|cpu=8,gres/gpu=1|0:0|a|b\n10.batch|COMPLETED|3600|cpu=8|0:0|a|b\n10.extern|COMPLETED|3600|cpu=8|0:0|a|b\n"
        rows=parse_sacct(raw)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]["gpu_allocation_hours"],1)
        self.assertTrue(rows[0]["terminal"])

    def test_missing_gpu_count_is_unknown_not_zero(self):
        row=parse_sacct("10|RUNNING|3600|cpu=8|0:0|a|Unknown\n")[0]
        self.assertIsNone(row["gpu_allocation_hours"])
        self.assertFalse(row["terminal"])

    def test_duplicate_parent_record_needs_explicit_reconciliation(self):
        with self.assertRaises(ValueError):
            parse_sacct("10|FAILED|3600|gres/gpu=1|1:0|a|b\n10|COMPLETED|4000|gres/gpu=1|0:0|b|c\n")

    def test_three_arms_one_job_is_counted_once(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);units=[]
            for arm in ["VCS","ConstantU","ShuffleVCS"]:
                output=root/arm
                write_json(output/"attempts/one.json",{"job_id":"10","status":"SUCCEEDED","elapsed_s":1000})
                units.append({"action":"RUN","output_dir":str(output)})
            exp=root/"experiment.json";write_json(exp,{"units":units})
            raw=root/"raw.txt";raw.write_text("10|COMPLETED|3600|cpu=8,gres/gpu=1|0:0|a|b\n")
            result=collect_accounting(exp,root/"accounting",raw)
            self.assertEqual(result["gpu_allocation_hours_known"],1)
            self.assertTrue(result["complete_terminal_accounting"])
            self.assertEqual(len(result["attempt_receipts"]),3)


if __name__ == "__main__": unittest.main()
