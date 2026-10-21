"""CPU checks of the N2P C0/C1 server pieces on synthetic inputs only (never experimental evidence).

    cd /tmp && PYTHON -I -B -m unittest discover -s <this dir> -p test_c_server.py -v
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c1_student as C  # noqa: E402


class TraceTests(unittest.TestCase):
    def run_traced(self, tmp, body, stage="c0"):
        script = tmp / "script.py"
        (tmp / "sibling.py").write_text("VALUE = 7\n")
        script.write_text("import sys\nfrom sibling import VALUE\n" + body)
        out = tmp / f"trace_{len(list(tmp.glob('trace_*')))}.json"
        proc = subprocess.run([sys.executable, "-I", "-B", str(HERE / "trace_open.py"), "--out", str(out), "--stage", stage,
                               "--", str(script), "arg1"], capture_output=True, text=True)
        return proc.returncode, json.loads(out.read_text())

    def test_clean_run_keeps_exit_code_and_lists_files(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            (tmp / "train_caps.txt").write_text("x")
            code, trace = self.run_traced(tmp, f"open({str(tmp / 'train_caps.txt')!r}).read()\nsys.exit(VALUE - 4)\n")
            self.assertEqual(code, 3)
            self.assertEqual(trace["status"], "PASS")
            self.assertIn(str((tmp / "train_caps.txt").resolve()), [x["path"] for x in trace["opened"]])
            self.assertEqual(trace["argv"], ["arg1"])

    def test_forbidden_reads_turn_status_and_exit_code(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            (tmp / "eval_only").mkdir()
            (tmp / "eval_only" / "pairs.eval_only.json").write_text("{}")
            (tmp / "dev_caps.txt").write_text("x")
            code, trace = self.run_traced(tmp, f"open({str(tmp / 'eval_only' / 'pairs.eval_only.json')!r}).read()\n")
            self.assertEqual((code, trace["status"]), (9, "FORBIDDEN_ACCESS"))
            code, trace = self.run_traced(tmp, f"open({str(tmp / 'dev_caps.txt')!r}).read()\n", stage="c0")
            self.assertEqual(code, 9)
            code, trace = self.run_traced(tmp, f"open({str(tmp / 'dev_caps.txt')!r}).read()\n", stage="c1")
            self.assertEqual((code, trace["status"]), (0, "PASS"))

    def test_forbidden_hits_are_visible_before_finish(self):
        import trace_open
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "eval_only" / "pairs.eval_only.json"
            bad.parent.mkdir()
            bad.write_text("{}")
            finish = trace_open.install("c1")
            self.assertEqual(finish.forbidden_hits(), [])
            bad.read_text()
            self.assertEqual(finish.forbidden_hits(), [str(bad.resolve())])
            self.assertEqual(finish(Path(d) / "trace.json")["status"], "FORBIDDEN_ACCESS")

    def test_script_failure_is_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            code, trace = self.run_traced(Path(d), "raise ValueError('boom')\n")
            self.assertEqual(code, 1)
            self.assertIn("boom", trace["error"])


class InvestmentGateTests(unittest.TestCase):
    @staticmethod
    def case(deltas, valid, collapse):
        units = {f"s{s}__{a}": {"severe_collapse": collapse} for s in C.SEEDS for a in C.ARMS}
        pairs = [{"seed": s, **{k: valid for k in C.PAIR_VALIDITY}, "vcs_minus_bce_best_rsum": d} for s, d in zip(C.SEEDS, deltas)]
        return C.rich_gate(units, pairs)

    def test_reviewer_counterexample_is_not_a_signal(self):
        gate = self.case([5.0, 5.0, 5.0], False, "UNKNOWN")      # all pairs mismatched, all collapse UNKNOWN
        self.assertEqual(gate["status"], "PAIRING_INVALID")
        self.assertIsNone(gate["investment_signal"])

    def test_unknown_collapse_or_missing_pair_is_incomplete(self):
        for gate in (self.case([5.0, 5.0, 5.0], True, "UNKNOWN"), self.case([5.0, 5.0], True, "NO")):
            self.assertEqual(gate["status"], "INCOMPLETE")
            self.assertIsNone(gate["investment_signal"])

    def test_valid_pairs_compute_the_rule(self):
        self.assertIs(self.case([5.0, 5.0, -1.0], True, "NO")["investment_signal"], True)
        gate = self.case([-4.596, -11.716, 6.095], True, "NO")
        self.assertEqual((gate["status"], gate["positive_seeds"], gate["investment_signal"]), ("COMPUTED", 1, False))


class CrosscheckReuseTests(unittest.TestCase):
    """c0_extract.sbatch accepts or reuses a TORCH_CROSSCHECK receipt only when it says PASS."""

    def run_extract(self, tmp, out_receipt=None, records_receipt=None, block_rerun=False):
        root, records = tmp / "root", tmp / "records"
        (root / "c0" / "traces").mkdir(parents=True, exist_ok=True)
        (records / "c0").mkdir(parents=True, exist_ok=True)
        if out_receipt is not None:
            C.write_json(root / "c0" / "TORCH_CROSSCHECK_SERVER.json", out_receipt)
        if records_receipt is not None:
            C.write_json(records / "c0" / "TORCH_CROSSCHECK_SERVER.json", records_receipt)
        if block_rerun:                    # trace_open refuses an existing trace, so no real crosscheck starts
            (root / "c0" / "traces" / "crosscheck_t.json").write_text("{}")
        env = dict(os.environ, N2PC_ROOT=str(root), N2PC_RECORDS=str(records), SLURM_JOB_ID="t", SEEDS=" ")
        proc = subprocess.run(["bash", str(HERE / "c0_extract.sbatch")], env=env, capture_output=True, text=True)
        return proc.returncode, proc.stdout, root / "c0" / "TORCH_CROSSCHECK_SERVER.json"

    def test_existing_non_pass_receipt_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            code, stdout, _ = self.run_extract(Path(d), out_receipt={"status": "FAIL"})
            self.assertEqual(code, 3)
            self.assertIn("REFUSED", stdout)

    def test_only_a_pass_receipt_is_reused(self):
        with tempfile.TemporaryDirectory() as d:
            code, stdout, receipt = self.run_extract(Path(d), records_receipt={"status": "PASS"})
            self.assertEqual(code, 0, stdout)
            self.assertEqual(json.loads(receipt.read_text())["status"], "PASS")
        with tempfile.TemporaryDirectory() as d:
            code, stdout, receipt = self.run_extract(Path(d), records_receipt={"status": "FAIL"}, block_rerun=True)
            self.assertNotEqual(code, 0)
            self.assertFalse(receipt.exists())


class GateTests(unittest.TestCase):
    def test_gate_requires_eligible_decision_and_acceptance(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with self.assertRaises(C.Refused) as ctx:
                C.read_gate(root, False)
            self.assertEqual(ctx.exception.code, 3)
            decision = {"status": "NO_FULL_TRAINING_FROM_THIS_PILOT", "full_training_eligible": False}
            C.write_json(root / "c0" / "PILOT_DECISION.json", decision)
            with self.assertRaises(C.Refused):
                C.read_gate(root, False)
            decision = {"status": C.ELIGIBLE, "full_training_eligible": True,
                        "verified_weight_identities": [{"seed": 901, "arm": "rich_vcs", "sha256": "a" * 64}]}
            C.write_json(root / "c0" / "PILOT_DECISION.json", decision)
            gate, verified = C.read_gate(root, False)
            self.assertEqual(verified[(901, "rich_vcs")], "a" * 64)
            with self.assertRaises(C.Refused):
                C.read_gate(root, True)
            C.write_json(root / "c1" / "C1_ACCEPTANCE.json", {"status": "FAIL"})
            with self.assertRaises(C.Refused):
                C.read_gate(root, True)
            C.write_json(root / "c1" / "C1_ACCEPTANCE.json", {"status": "PASS"})          # no decision binding
            with self.assertRaises(C.Refused):
                C.read_gate(root, True)
            C.write_json(root / "c1" / "C1_ACCEPTANCE.json", {"status": "PASS", "gate": {"decision_sha256": gate["decision_sha256"]}})
            self.assertIn("acceptance_sha256", C.read_gate(root, True)[0])
            decision["note"] = "a later, different decision file"
            C.write_json(root / "c0" / "PILOT_DECISION.json", decision)
            with self.assertRaises(C.Refused) as ctx:                       # acceptance made under another decision
                C.read_gate(root, True)
            self.assertIn("different PILOT_DECISION", str(ctx.exception))

    def test_pair_environment_must_not_change_between_attempts(self):
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            env = {"gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "driver": "615.71.09", "torch": "2.9.0+cu128",
                   "cuda": "12.8", "cudnn": 91002, "python": "3.12.12", "numpy": "2.5.2"}
            C.attempt_log(runs / "s901__RICH_VCS", {"event": "start", "environment": env})
            C.check_paired_environment(runs, 901, dict(env, host="other"))          # host may differ
            with self.assertRaises(C.Refused) as ctx:
                C.check_paired_environment(runs, 901, dict(env, gpu="NVIDIA L40S"))
            self.assertEqual(ctx.exception.code, 76)
            C.check_paired_environment(runs, 902, dict(env, gpu="NVIDIA L40S"))     # other seeds independent

    def test_precision_matches_protocol(self):
        effective = C.configure_precision(C.protocol()["precision"])
        self.assertTrue(effective["deterministic_algorithms"] and effective["cudnn_deterministic"])
        self.assertFalse(effective["cuda_matmul_tf32"] or effective["cudnn_tf32"] or effective["cudnn_benchmark"])
        self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")


class StudentSmokeTests(unittest.TestCase):
    """Tiny synthetic SGR through the original WeightedStudent on CPU (code path only)."""

    @classmethod
    def setUpClass(cls):
        import numpy as np
        from v9.n2.fixtures import collate_pairs, make_fixture
        from v9.n2.teacher import HostConfig
        fx = make_fixture(n_images=12, n_caps=5, n_region=3, img_dim=8, vocab=40, seed=0)
        cls.cfg = HostConfig(vocab_size=40, img_dim=8, word_dim=8, embed_size=16, sim_dim=8, num_region=3, sgr_step=1,
                             batch_size=8, num_epochs=2, warmup_epochs=1, lr_update=30)
        cls.rows = [{"pair_id": f"p{c}", "caption_id": str(c), "observed_image_id": str(c // 5),
                     "role": "critic_fit" if c < 10 else "U"} for c in range(60)]
        u = [r["pair_id"] for r in cls.rows if r["role"] == "U"]
        cls.weights = {"RICH_VCS": dict(zip(u, np.linspace(.05, .95, len(u)).tolist())),
                       "RICH_BCE": dict(zip(u, np.linspace(.9, .3, len(u)).tolist()))}
        cls.ctx = {"cfg": cls.cfg, "collate": partial(collate_pairs, fx["images"], fx["captions"]),
                   "dev_images": fx["images"], "dev_captions": [fx["captions"][c] for c in range(60)]}

    def test_both_arms_train_with_paired_identity_and_summary(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            res = {}
            for arm in ("RICH_VCS", "RICH_BCE"):
                out = root / "c1" / "runs" / f"s901__{arm}"
                inputs = {"rows": self.rows, "weights": self.weights[arm], "lock": {"sha256": "SYNTHETIC"}}
                C.attempt_log(out, {"event": "start", "environment": {}})
                checks, done = C.train_arm(self.ctx, inputs, 901, out, 2, "cpu")
                C.attempt_log(out, {"event": "end", "status": "COMPLETE", "elapsed_s": 1.0})
                post, _ = C.log_checks(out, 2, 8)
                self.assertTrue(checks["ok"] and post["ok"], (checks, post))
                self.assertEqual((checks["n_A"], checks["n_U"], checks["expected_updates"], post["updates"]), (10, 50, 16, 16))
                self.assertEqual(checks["last_batch_sizes"], [4])
                res[arm] = {"checks": checks, "init": C.initial_model_sha256(self.cfg, 901), "done": done}
                C.write_json(out / "unit_result.json", {
                    "status": "COMPLETE_VERIFIED", "environment": {"gpu": "cpu", "host": "test"},
                    "initial_model_sha256": res[arm]["init"], "data_order_sha256": checks["data_order_sha256"],
                    "handoff_receipt": {"weight_sha256": arm}, "backend_sha256": "b", "optimizer_sha256": "o"})
            self.assertEqual(res["RICH_VCS"]["init"], res["RICH_BCE"]["init"])
            self.assertEqual(res["RICH_VCS"]["checks"]["data_order_sha256"], res["RICH_BCE"]["checks"]["data_order_sha256"])
            self.assertNotEqual(res["RICH_VCS"]["checks"]["sum_w_U"], res["RICH_BCE"]["checks"]["sum_w_U"])
            summary = C.summarize(root)
            self.assertEqual([p["seed"] for p in summary["pairs"]], [901])
            pair = summary["pairs"][0]
            self.assertTrue(pair["same_initial_model"] and pair["same_data_order"])
            self.assertAlmostEqual(pair["vcs_minus_bce_best_rsum"],
                                   summary["units"]["s901__RICH_VCS"]["best_rsum"] - summary["units"]["s901__RICH_BCE"]["best_rsum"])
            self.assertEqual(summary["new_rich_gate"]["status"], "INCOMPLETE")      # needs all three seeds
            self.assertIsNone(summary["new_rich_gate"]["investment_signal"])
            self.assertEqual(summary["units"]["s902__RICH_VCS"]["status"], "NOT_COMPLETE")
            self.assertTrue((root / "c1" / "C1_EPOCHS.csv").is_file())

    def test_no_completion_record_after_forbidden_read(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "unit"
            inputs = {"rows": self.rows, "weights": self.weights["RICH_VCS"], "lock": {"sha256": "SYNTHETIC"}}
            with self.assertRaises(C.Refused) as ctx:
                C.train_arm(self.ctx, inputs, 901, out, 2, "cpu", trace_hits=lambda: ["/x/eval_only/pairs.eval_only.json"])
            self.assertEqual(ctx.exception.code, 9)
            self.assertFalse((out / "done.json").exists() or (out / "last.pt").exists() or (out / "best.pt").exists())
            with self.assertRaises(C.Refused):
                C.require_clean_trace(lambda: ["/x/test_ims.npy"])
            C.require_clean_trace(lambda: [])

    def test_pool_checks_catch_wrong_anchor_weight_or_normaliser(self):
        import numpy as np

        class Fake:
            pass
        fake = Fake()
        is_u = np.array([r["role"] == "U" for r in self.rows])
        w = self.weights["RICH_VCS"]
        fake.v = np.array([1.0 if r["role"] != "U" else w[r["pair_id"]] for r in self.rows])
        fake.mean_v_full = float(fake.v.sum() / len(fake.v))
        self.assertTrue(C.pool_checks(fake, self.rows, w, 901, self.cfg)["ok"])
        fake.v[~is_u] = .5
        self.assertFalse(C.pool_checks(fake, self.rows, w, 901, self.cfg)["ok"])
        fake.v[~is_u] = 1.0
        fake.mean_v_full *= 1.001
        self.assertFalse(C.pool_checks(fake, self.rows, w, 901, self.cfg)["ok"])
        fake.mean_v_full /= 1.001
        expect = {**C.PROTOCOL_POOL, "updates": 45320}
        self.assertFalse(C.pool_checks(fake, self.rows, w, 901, self.cfg, expect)["ok"])   # not the protocol pool


if __name__ == "__main__":
    unittest.main()
