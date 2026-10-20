"""Bookkeeping tests are CPU-only; actual SGR equivalence is a server check.

Set V9N2M_BASELINE_ROOT to the frozen ``source`` directory to also exercise the
exact source transformation and its unchanged scientific expressions.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from v9n2m.trainer_patch import (
    BASELINE_TRAIN_SHA256,
    _PATCH_HELPERS,
    patch_train_source,
    telemetry_patch_identity,
)


def helpers():
    namespace = {"Path": Path, "json": json, "hashlib": hashlib}
    exec(compile(_PATCH_HELPERS, "<telemetry-helpers>", "exec"), namespace)
    return namespace


class UpdateAccountingTests(unittest.TestCase):
    def setUp(self):
        self.h = helpers()

    def test_counts_updates_not_sparse_rows(self):
        log = {
            "epochs": [{"epoch": 0, "steps_in_epoch": 1133}, {"epoch": 1, "steps_in_epoch": 1133}],
            "steps": [{"step": 20 * (i + 1)} for i in range(113)],
        }
        self.assertEqual(self.h["_n2m_completed_updates"](log), 2266)
        self.assertNotEqual(len(log["steps"]), 2266)

    def test_full_40_epoch_update_count(self):
        log = {"epochs": [{"epoch": e, "steps_in_epoch": 1133, "global_step": (e + 1) * 1133} for e in range(40)]}
        self.assertEqual(self.h["_n2m_completed_updates"](log), 45320)

    def test_missing_or_corrupted_history_is_not_guessed(self):
        invalid = [
            {"epochs": [{"epoch": 0}]},
            {"epochs": [{"epoch": 1, "steps_in_epoch": 1133}]},
            {"epochs": [{"epoch": 0, "steps_in_epoch": 1133, "global_step": 56}]},
            {"epochs": [{"epoch": 0, "steps_in_epoch": 0}]},
            {"epochs": [{"epoch": 0, "steps_in_epoch": True}]},
            {"epochs": [{"epoch": 0, "steps_in_epoch": 1.5}]},
        ]
        for log in invalid:
            with self.subTest(log=log), self.assertRaises(ValueError):
                self.h["_n2m_completed_updates"](log)

    def test_interruption_is_checkpoint_boundary_not_completion(self):
        log = {"epochs": [{"epoch": 0, "steps_in_epoch": 1133}], "steps": []}
        best = {"rsum": 450.0, "epoch": 0}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            result = self.h["_n2m_interrupted"](out, best, log, 40, 1133)
            self.assertEqual(result["status"], "INTERRUPTED")
            self.assertEqual(result["epochs_run"], 1)
            self.assertEqual(result["target_epochs"], 40)
            self.assertEqual(json.loads((out / "train_log.json").read_text()), log)
            self.assertFalse((out / "done.json").exists())
            self.assertFalse((out / "train_log.json.tmp").exists())

    def test_existing_done_is_checked_without_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            log = {"epochs": [{"epoch": 0, "steps_in_epoch": 3, "dev": {"rsum": 410.0}}, {"epoch": 1, "steps_in_epoch": 3, "dev": {"rsum": 410.0}}], "steps": []}
            done = {"epochs_run": 2, "global_step": 6, "best": {"rsum": 410.0, "epoch": 0}}
            (out / "train_log.json").write_text(json.dumps(log))
            (out / "done.json").write_text(json.dumps(done))
            (out / "best.pt").write_bytes(b"structural-test-only")
            (out / "last.pt").write_bytes(b"structural-test-only")
            self.assertEqual(self.h["_n2m_read_completed"](out, 2), done)
            with self.assertRaises(ValueError):
                self.h["_n2m_read_completed"](out, 40)
            done["best"]["epoch"] = 1
            (out / "done.json").write_text(json.dumps(done))
            with self.assertRaises(ValueError):
                self.h["_n2m_read_completed"](out, 2)
            (out / "last.pt").unlink()
            with self.assertRaises(ValueError):
                self.h["_n2m_read_completed"](out, 2)

    def test_checkpoint_is_bound_to_arm_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            identity = out / "identity.json"
            manifest = out / "run_manifest.json"
            identity.write_text('{"arm":"VCS","seed":901}')
            manifest.write_text('{"mean_v_full_pool":0.61}')
            checkpoint = {
                "execution_identity_sha256": hashlib.sha256(identity.read_bytes()).hexdigest(),
                "run_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            }
            self.h["_n2m_validate_resume_identity"](out, checkpoint)
            identity.write_text('{"arm":"ShuffleVCS","seed":901}')
            with self.assertRaisesRegex(ValueError, "execution identity"):
                self.h["_n2m_validate_resume_identity"](out, checkpoint)
            checkpoint["execution_identity_sha256"] = hashlib.sha256(identity.read_bytes()).hexdigest()
            manifest.write_text('{"mean_v_full_pool":0.99}')
            with self.assertRaisesRegex(ValueError, "run manifest"):
                self.h["_n2m_validate_resume_identity"](out, checkpoint)

    def test_legacy_checkpoint_is_not_adopted_as_production(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "run_manifest.json").write_text('{}')
            self.h["_n2m_validate_resume_identity"](out, {})
            (out / "identity.json").write_text('{"unit_id":"s901__ConstantU"}')
            with self.assertRaisesRegex(ValueError, "execution identity"):
                self.h["_n2m_validate_resume_identity"](out, {})
            checkpoint = {"execution_identity_sha256": hashlib.sha256((out / "identity.json").read_bytes()).hexdigest()}
            with self.assertRaisesRegex(ValueError, "run manifest"):
                self.h["_n2m_validate_resume_identity"](out, checkpoint)
            (out / "identity.json").unlink()
            with self.assertRaisesRegex(ValueError, "lost"):
                self.h["_n2m_validate_resume_identity"](out, checkpoint)


class PatchTests(unittest.TestCase):
    def test_unknown_source_is_refused(self):
        with self.assertRaisesRegex(ValueError, "unrecognised trainer"):
            patch_train_source("print('not the executed trainer')\n")

    def test_identity_does_not_claim_gpu_pass(self):
        identity = telemetry_patch_identity()
        self.assertEqual(identity["baseline_train_sha256"], BASELINE_TRAIN_SHA256)
        self.assertFalse(identity["scientific_loop_changes"])
        self.assertIn("requires", identity["gpu_equivalence"])

    def _source(self):
        root = os.environ.get("V9N2M_BASELINE_ROOT")
        if not root:
            self.skipTest("set V9N2M_BASELINE_ROOT for the exact frozen-source check")
        path = Path(root) / "src/v9/n2/train.py"
        self.assertTrue(path.is_file(), str(path))
        return path.read_text(encoding="utf-8")

    def test_pinned_source_transformation_compiles(self):
        original = self._source()
        self.assertEqual(hashlib.sha256(original.encode()).hexdigest(), BASELINE_TRAIN_SHA256)
        patched = patch_train_source(original)
        compile(patched, "<patched>", "exec")
        self.assertEqual(patched, patch_train_source(original))
        self.assertNotIn('step = len(log["steps"])', patched)
        self.assertIn('"step_in_epoch": in_epoch, "loss": float(loss)', patched)
        self.assertIn('"global_step": _n2m_completed_updates(log)', patched)
        with self.assertRaises(ValueError):
            patch_train_source(patched)

    def test_scientific_expressions_and_rng_functions_are_identical(self):
        original = self._source()
        patched = patch_train_source(original)
        protected = [
            "seed_everything(self.seed); host = build_host(cfg).to(self.device); opt = torch.optim.Adam(host.parameters(), lr=cfg.learning_rate * self.lr_mult)",
            "lr = cfg.learning_rate * self.lr_mult * (0.1 ** (epoch // cfg.lr_update))",
            "rng = np.random.default_rng([self.seed, 9_020, epoch])",
            "for idx in iterate_batches(len(self.rows), cfg.batch_size, rng):",
            "sims = host(ims.to(self.device), caps.to(self.device), lens)",
            "hard = epoch >= cfg.warmup_epochs",
            "er, ec = hinge_per_anchor(sims, cfg.margin, hard_negative=hard); ell = er + ec",
            'loss = ((v.detach() * ell).sum() if self.loss_scale == "sum" else (v.detach() * ell).mean()) / self.mean_v_full',
            "opt.zero_grad(set_to_none=True); loss.backward()",
            "gn_t = torch.nn.utils.clip_grad_norm_(host.parameters(), cfg.grad_clip)",
            "opt.step(); step += 1; in_epoch += 1",
            'if val["rsum"] > best["rsum"]:',
        ]
        for expression in protected:
            with self.subTest(expression=expression):
                self.assertEqual(original.count(expression), 1)
                self.assertEqual(patched.count(expression), 1)
        for name in ("_rng_state", "_set_rng_state"):
            before = next(n for n in ast.parse(original).body if isinstance(n, ast.FunctionDef) and n.name == name)
            after = next(n for n in ast.parse(patched).body if isinstance(n, ast.FunctionDef) and n.name == name)
            self.assertEqual(ast.dump(before), ast.dump(after))


if __name__ == "__main__":
    unittest.main()
