"""R4-01 regression tests for scripts/p151_e0_aggregate_r4.py: the common readout never takes a native value."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from p151_e0_aggregate_r4 import common_fields, native_fields  # noqa: E402


class CommonReadout(unittest.TestCase):
    def test_js_native_not_used(self):
        row = {"estimand": "JS2", "native": {"value": 1.1, "own_truth": 1.3, "signed_error": -0.2},
               "posterior": {"defined": True, "J_eval": 0.7, "J_eval_minus_S": -0.1, "R_eval": 0.3, "posterior_mse": 0.12, "J_eval_boot_se": 0.01}}
        cf = common_fields(row)
        self.assertEqual(cf["common_J"], 0.7)
        self.assertEqual(cf["common_J_minus_S"], -0.1)
        self.assertTrue(cf["identity_R_eq_1_minus_J"])
        self.assertEqual(native_fields(row)["native_value"], 1.1)

    def test_missing_common_is_null_with_reason(self):
        row = {"native": {"value": 4.9}, "posterior": {"defined": False, "reason": "no balanced posterior"}}
        cf = common_fields(row)
        self.assertIsNone(cf["common_J"])
        self.assertEqual(cf["status"], "missing_common_readout")
        self.assertTrue(cf["reason"])

    def test_absent_posterior_block(self):
        cf = common_fields({"native": {"value": 0.5}})
        self.assertIsNone(cf["common_J"])
        self.assertNotEqual(cf.get("common_J"), 0.5)

    def test_nonfinite_flagged(self):
        cf = common_fields({"posterior": {"defined": True, "J_eval": float("nan")}})
        self.assertIsNone(cf["common_J"])
        self.assertEqual(cf["status"], "nonfinite_common_readout")

    def test_identity_violation_detected(self):
        cf = common_fields({"posterior": {"defined": True, "J_eval": 0.7, "R_eval": 0.5}})
        self.assertFalse(cf["identity_R_eq_1_minus_J"])


if __name__ == "__main__":
    unittest.main()
