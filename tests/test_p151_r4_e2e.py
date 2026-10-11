"""R4-01 end-to-end regression (package docs/05 step 4): with a JS row whose native value is 1.1 and common J_eval 0.7, the aggregated common panel must
read 0.7 (seed sd from the common values), the native panel 1.1; an identity-violating row and a family without a single registered selection must be
excluded and listed.  Runs the real main() on a fixture via monkeypatched load_cells / O."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import p151_e0_aggregate_r4 as agg  # noqa: E402


def row(method, native_value, j_eval, mse, r_eval=None, selected=True, own_truth=1.3, s=0.8):
    return {"method": method, "selected": selected, "estimand": "JS2" if "js" in method else "S", "evaluation_readout": "native", "fit_seconds": 1.0, "n_params": 10,
            "numerical": {"nonfinite_steps": 0}, "select": {"curve": [[k, 0.5] for k in range(6)]}, "negative_construction": method.split(":")[2], "split_manifest_hash": "h",
            "native": {"value": native_value, "own_truth": own_truth, "signed_error": native_value - own_truth, "abs_error": abs(native_value - own_truth), "boot_se": 0.01, "boot_values": [native_value] * 5},
            "posterior": {"defined": True, "J_eval": j_eval, "J_eval_minus_S": j_eval - s, "R_eval": (1 - j_eval) if r_eval is None else r_eval, "posterior_mse": mse, "J_eval_boot_se": 0.01}}


def fixture():
    cells = {}
    for I in (2, 4):
        for seed, (jn, jj) in enumerate(((1.1, 0.70), (1.3, 0.72), (0.9, 0.68))):
            S = 0.8 + 0.001 * seed                                                                                       # per-seed truth S, as in P85 cells
            rows = [row("neural:js:inbatch:lr0.0001", jn, jj, 0.12, s=S), row("neural:vcs:inbatch:lr0.0001", 0.69, 0.69, 0.13, r_eval=0.5 if (I == 2 and seed == 0) else None, s=S),
                    row("neural:vcs:product:lr0.0001", 0.6, 0.6, 0.2, s=S), row("neural:vcs:product:lr0.0005", 0.61, 0.61, 0.2, s=S)]      # product: two selected rows -> excluded
            name = f"P85_gaussian_ds20_dt20_I{I}_N4096_B256_U2000_s{seed}"
            cells[name] = {"cell": {"setting": "gaussian", "d_signal": 20, "d_total": 20, "I": float(I), "N": 4096, "B": 256, "updates": 2000, "seed": seed, "methods": "all", "name": name},
                           "truth": {"S": S, "S_se": 0.001, "MI": float(I), "J_oracle": 0.79, "JS2": 1.3}, "rows": rows}
    return cells


class EndToEnd(unittest.TestCase):
    def test_common_panel_reads_common_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            agg.load_cells = fixture; agg.O = Path(tmp) / "no_inputs"; sys.argv = ["p151_e0_aggregate_r4.py", "--out", tmp + "/out", "--old", tmp + "/none"]
            self.assertEqual(agg.main(), 0)
            common = json.load(open(Path(tmp) / "out" / "e_precision_common.json")); native = json.load(open(Path(tmp) / "out" / "e_precision_native.json"))
            stab = json.load(open(Path(tmp) / "out" / "e_stability_r4.json")); meta = json.load(open(Path(tmp) / "out" / "metadata.json"))
            js = common["gaussian"]["4"]["families"]["neural:js:inbatch"]
            self.assertAlmostEqual(js["common_J"]["mean"], 0.70, places=9); self.assertAlmostEqual(js["common_J_minus_S"]["mean"], 0.70 - 0.801, places=9)
            self.assertAlmostEqual(native["gaussian"]["4"]["families"]["neural:js:inbatch"]["native_value"]["mean"], 1.1, places=9)
            self.assertAlmostEqual(stab["gaussian"]["4"]["neural:js:inbatch"]["seed_sd_common_J"], 0.02, places=9)       # common sd, not the native 0.2
            self.assertAlmostEqual(common["gaussian"]["4"]["S_oracle"], 0.801, places=9)                                  # seed mean of truth.S
            vcs2 = common["gaussian"]["2"]["families"]["neural:vcs:inbatch"]
            self.assertEqual(vcs2["common_J"]["n"], 2)                                                                       # the identity-violating seed is excluded
            self.assertEqual(len(meta["identity_R_eq_1_minus_J_failures"]), 1)
            self.assertNotIn("neural:vcs:product", common["gaussian"]["4"]["families"])                                     # two selected rows -> no registered selection
            self.assertTrue(any(p["family"] == "neural:vcs:product" for p in meta["selection_problems"]))


if __name__ == "__main__":
    unittest.main()
