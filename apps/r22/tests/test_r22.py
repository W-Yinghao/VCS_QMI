"""r22 CPU tests (torch, no GPU, no pretrained assets): VCS-CRD and banked / MOS-weighted VCS.

Run: python -I test_r22.py --report <json>. The engine next to this directory is put first on sys.path.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
ENGINE = HERE / "engine"
REPDISTILLER = "/home/infres/yinwang/CS_QMI/outputs_apps/kd_20261020/external/RepDistiller"
BASE_PKG = Path("/home/infres/yinwang/CS_QMI/ssl_pilot_app_r22/apps/VCS_QMI_KD_AIGIQA_Server_Package_v1_20261019")
AIG_FIT = "/projects/EEG-foundation-model/yinghao/data/agiqa_3k/prepared/AGIQA-3k/fit.csv"
MEASUREMENTS = {}


def policy(mode="alternating_target_sg", warmup=2, ramp=4):
    return dict(mode=mode, warmup_epochs=warmup, ramp_epochs=ramp, critic_steps=1,
                critic_lr=.001, critic_weight_decay=0., critic_lr_step_epochs=4, critic_lr_gamma=.5)


def crd_cfg(**kw):
    cfg = dict(source_root=REPDISTILLER, feat_dim=8, negatives=12, temperature=.07, momentum=.5,
               negative_mode="uniform_other_instances", objective="vcs")
    cfg.update(kw)
    return cfg


class KDTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2022)
        torch.set_num_threads(1)

    def build(self, n=40, **kw):
        from kd.crd_adapter import OfficialCRD
        ids = list(range(100, 100 + n))
        labels = [i % 4 for i in range(n)]
        crd = OfficialCRD(6, 5, ids, labels, crd_cfg(**kw), seed=31).double()
        return crd, ids

    def negatives(self, crd, global_ids):
        rows = []
        for g in global_ids:
            pos = crd.lookup[int(g)]
            pool = np.arange(crd.n_data)[np.arange(crd.n_data) != pos]
            r = np.random.default_rng(crd.seed + crd.epoch * 1000003 + int(g) * 9176)
            rows.append(r.choice(pool, size=crd.k, replace=crd.k > len(pool)))
        return np.stack(rows)

    def test_vcs_crd_equals_brute_force_J_with_memory_bank(self):
        crd, ids = self.build()
        crd.set_epoch(3)
        batch = torch.tensor(ids[5:13])
        fs, ft = torch.randn(8, 6, dtype=torch.float64), torch.randn(8, 5, dtype=torch.float64)
        mem = crd.loss.contrast
        with torch.no_grad():   # non-trivial memory contents (unit rows), as after a first epoch
            mem.memory_v1.copy_(torch.nn.functional.normalize(torch.randn_like(mem.memory_v1), dim=1))
            mem.memory_v2.copy_(torch.nn.functional.normalize(torch.randn_like(mem.memory_v2), dim=1))
        m1, m2 = mem.memory_v1.clone(), mem.memory_v2.clone()
        official = copy.deepcopy(mem)
        loss = crd(fs, ft, batch)
        neg = self.negatives(crd, batch.tolist())
        pos = np.array([crd.lookup[int(g)] for g in batch])
        self.assertFalse((neg == pos[:, None]).any(), "Q must not contain the anchor's own instance")
        with torch.no_grad():
            es, et = crd.loss.embed_s(fs), crd.loss.embed_t(ft)
            tau = .07
            tp, tqs, tqt = [], [], []
            for i in range(len(batch)):
                tp.append(float(torch.tanh(es[i] @ et[i] / tau)))
                for j in neg[i]:
                    tqs.append(float(torch.tanh(es[i] @ m2[j] / tau)))   # student anchor x teacher memory
                    tqt.append(float(torch.tanh(et[i] @ m1[j] / tau)))   # teacher anchor x student memory
            tp, tqs, tqt = map(lambda v: torch.tensor(v, dtype=torch.float64), (tp, tqs, tqt))
            J = lambda a, b: a.mean() - b.mean() - .5 * a.square().mean() - .5 * b.square().mean()
            brute = -(J(tp, tqs) + J(tp, tqt))
        self.assertEqual(len(tqs), len(batch) * crd.k)
        torch.testing.assert_close(loss.detach(), brute, rtol=1e-12, atol=1e-12)
        MEASUREMENTS["kd_vcs_crd_bruteforce_abs_error"] = float((loss.detach() - brute).abs())
        # Memory update equals the official ContrastMemory update for the same embeddings and positives.
        contrast = torch.from_numpy(np.concatenate([pos[:, None], neg], 1))
        with contextlib.redirect_stdout(io.StringIO()):
            official(es, et, torch.from_numpy(pos), contrast)
        torch.testing.assert_close(mem.memory_v1, official.memory_v1, rtol=0, atol=1e-15)
        torch.testing.assert_close(mem.memory_v2, official.memory_v2, rtol=0, atol=1e-15)
        untouched = np.setdiff1d(np.arange(crd.n_data), pos)
        self.assertTrue(torch.equal(mem.memory_v1[untouched], m1[untouched]))

    def test_vcs_crd_needs_product_of_marginals_Q(self):
        with self.assertRaises(ValueError):
            self.build(negative_mode="class_excluding_original")
        with self.assertRaises(ValueError):
            self.build(objective="infonce")

    def test_vcs_crd_gradient_routes_teacher_and_bank_detached(self):
        crd, ids = self.build()
        fs = torch.randn(8, 6, dtype=torch.float64, requires_grad=True)
        ft = torch.randn(8, 5, dtype=torch.float64, requires_grad=True)   # forward must detach teacher features
        loss = crd(fs, ft, torch.tensor(ids[:8]))
        loss.backward()
        self.assertGreater(float(fs.grad.norm()), 0.)
        self.assertIsNone(ft.grad)
        self.assertGreater(float(crd.loss.embed_s.linear.weight.grad.norm()), 0.)
        self.assertGreater(float(crd.loss.embed_t.linear.weight.grad.norm()), 0.)
        mem = crd.loss.contrast
        self.assertFalse(mem.memory_v1.requires_grad or mem.memory_v2.requires_grad)
        self.assertNotIn("memory_v1", dict(crd.named_parameters()))
        for key in ("vcs_j_s", "vcs_j_t", "vcs_tp_mean", "vcs_tqs_mean", "vcs_tqt_mean"):
            self.assertIn(key, crd.last_parts)

    def test_official_nce_path_unchanged(self):
        """objective absent -> official CRDLoss; same value as the unmodified base adapter."""
        from kd.crd_adapter import OfficialCRD
        base_path = BASE_PKG
        src = (base_path / "kd" / "crd_adapter.py").read_text().replace("from .models import", "from kd.models import")
        ns = {}
        exec(compile(src, str(base_path / "kd" / "crd_adapter.py"), "exec"), ns)
        ids, labels = list(range(40)), [i % 4 for i in range(40)]
        cfg = crd_cfg(); cfg.pop("objective")
        torch.manual_seed(5); new = OfficialCRD(6, 5, ids, labels, cfg, seed=3)
        torch.manual_seed(5); old = ns["OfficialCRD"](6, 5, ids, labels, cfg, seed=3)
        fs, ft = torch.randn(8, 6), torch.randn(8, 5)
        with contextlib.redirect_stdout(io.StringIO()):
            a, b = new(fs, ft, torch.tensor(ids[:8])), old(fs, ft, torch.tensor(ids[:8]))
        self.assertTrue(torch.equal(a, b))

    def test_runner_synthetic_vcscrd_and_dkd_vcscrd_teacher_frozen_fit_cal_only(self):
        from kd.data import prepare, write_json
        import kd.data
        from kd.runner import train
        roles = []
        original = kd.data.RoleDataset.__init__
        def record(self, root, manifest, role, *a, **k):
            roles.append(role)
            return original(self, root, manifest, role, *a, **k)
        with tempfile.TemporaryDirectory() as tmp, unittest.mock.patch.object(kd.data.RoleDataset, "__init__", record):
            root = Path(tmp)
            prepare(root / "data", root / "roles.json", synthetic=True)
            cfg = dict(schema="kd_config_v1", name="cpu_teacher", seed=88, device="cpu", threads=1,
                       track="synthetic_smoke", data=dict(root=str(root / "data"), manifest=str(root / "roles.json")),
                       teacher=dict(backend="tiny", name="tiny_teacher"), student=dict(backend="tiny", name="tiny_student"),
                       loss=dict(anchor="none", feature="none"),
                       optimizer=dict(lr=.03, momentum=.9, weight_decay=.0005, head_weight_decay=0.,
                                      head_optimizer="sgd", head_lr=.03, milestones=[2], gamma=.1),
                       training=dict(epochs=1, batch_size=16, workers=0))
            path = root / "teacher.json"; write_json(path, cfg)
            with contextlib.redirect_stdout(io.StringIO()):
                train(path, root / "teacher", teacher_training=True)
            cfg["teacher"].update(checkpoint=str(root / "teacher" / "teacher.pt"),
                                  provenance=str(root / "teacher" / "teacher_provenance.json"))
            cfg["training"]["epochs"] = 2
            for name, extra in (("vcscrd", dict(anchor="none")),
                                ("dkd_vcscrd", dict(anchor="dkd", temperature=4., dkd_alpha=1., dkd_beta=6., dkd_warmup=20))):
                c = copy.deepcopy(cfg); c["name"] = name
                c["loss"].update(extra, ce_weight=1., crd_weight=3., crd=crd_cfg(negatives=16))
                path = root / f"{name}.json"; write_json(path, c)
                with contextlib.redirect_stdout(io.StringIO()):
                    result = train(path, root / name)
                self.assertEqual(result["status"], "complete")
                self.assertTrue(result["teacher_frozen_verified"])
                history = json.loads((root / name / "history.json").read_text())
                parts = history[-1]["loss_parts"]
                self.assertIn("vcs_j_s", parts); self.assertIn("crd", parts)
                self.assertIsNotNone(history[0]["gradient_probe"]["weighted_representation_gradient_norm"])
                MEASUREMENTS[f"kd_synthetic_{name}_last_crd_loss"] = parts["crd"]
        self.assertEqual(set(roles), {"fit", "cal"})


class AIGTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4101)
        torch.set_num_threads(1)

    def banked(self, method="vcs_exact", bank=7, weighting="mos"):
        from aigiqa.losses import BankedDependence
        reg = BankedDependence(4, 3, method, rank=4, bank_size=bank, p_weighting=weighting).double()
        with torch.no_grad():
            reg.W.uniform_(-.6, .6)
        return reg

    def test_banked_J_and_bce_equal_brute_force(self):
        for method in ("vcs_exact", "bce"):
            reg = self.banked(method)
            history = [torch.randn(5, 3, dtype=torch.float64) for _ in range(3)]
            for y_old in history:
                reg.enqueue(y_old)
            contents = torch.cat(history)[-7:]     # FIFO keeps the 7 most recent rows
            self.assertEqual(int(reg.bank_count), 7)
            self.assertEqual({tuple(r.tolist()) for r in reg.bank}, {tuple(r.tolist()) for r in contents})
            x = torch.randn(5, 4, dtype=torch.float64); y = torch.randn(5, 3, dtype=torch.float64)
            labels = torch.tensor([.2, .9, .5, .7, .1], dtype=torch.float64)
            reg.set_labels(labels)
            loss = reg(x, y)
            w = labels / labels.mean()
            T = lambda xi, yj: float(torch.softmax(reg.head.left(xi), -1) @ reg.W @ torch.softmax(reg.head.right(yj), -1))
            with torch.no_grad():
                pos = torch.tensor([T(x[i], y[i]) for i in range(5)], dtype=torch.float64)
                neg = torch.tensor([T(x[i], y[j]) for i in range(5) for j in range(5) if i != j]
                                   + [T(x[i], contents[k]) for i in range(5) for k in range(7)], dtype=torch.float64)
            self.assertEqual(neg.numel(), 5 * 4 + 5 * 7)
            if method == "vcs_exact":
                brute = -((w * pos).mean() - neg.mean() - .5 * (w * pos.square()).mean() - .5 * neg.square().mean())
            else:
                brute = -((w * torch.log1p(pos)).mean() + torch.log1p(-neg).mean())
            torch.testing.assert_close(loss.detach(), brute, rtol=1e-12, atol=1e-12)
            MEASUREMENTS[f"aig_banked_{method}_bruteforce_abs_error"] = float((loss.detach() - brute).abs())

    def test_bank_zero_uniform_equals_package_objective(self):
        from vcs_common.dependence import DependenceRegularizer
        from vcs_common.application_update import ApplicationUpdate
        for method in ("vcs_exact", "bce"):
            reg = self.banked(method, bank=0, weighting="uniform")
            pkg = DependenceRegularizer(4, 3, method=method, rank=4, q_mode="u").double()
            pkg.head.load_state_dict(reg.head.state_dict())
            x = torch.randn(6, 4, dtype=torch.float64); y = torch.randn(6, 3, dtype=torch.float64)
            torch.testing.assert_close(reg(x, y), pkg(x, y), rtol=1e-12, atol=1e-12)
            staged = ApplicationUpdate({"dep": reg}, policy(), .1).outer_loss("dep", x, y)
            torch.testing.assert_close(staged, pkg(x, y), rtol=1e-12, atol=1e-12)

    def test_mos_weighted_P_is_a_probability_measure(self):
        reg = self.banked()
        labels = torch.tensor([.04, .6, .9, .3, .2], dtype=torch.float64)
        reg.set_labels(labels)
        self.assertTrue(bool((reg.weights >= 0).all()))
        torch.testing.assert_close(reg.weights.sum() / len(labels), torch.tensor(1., dtype=reg.weights.dtype))
        # constant labels give the uniform P
        reg.set_labels(torch.full((5,), .4, dtype=torch.float64))
        torch.testing.assert_close(reg.weights, torch.ones(5, dtype=reg.weights.dtype))
        for bad in (torch.tensor([-.1, .5, .5, .5, .5]), torch.zeros(5), torch.tensor([float("nan"), 1, 1, 1, 1])):
            with self.assertRaises(ValueError):
                reg.set_labels(bad)
        x = torch.randn(5, 4, dtype=torch.float64); y = torch.randn(5, 3, dtype=torch.float64)
        reg.weights = None
        with self.assertRaises(RuntimeError):
            reg(x, y)

    def test_alternating_bank_detached_outer_x_only(self):
        from vcs_common.application_update import ApplicationUpdate
        reg = self.banked()
        reg.enqueue(torch.randn(7, 3, dtype=torch.float64))
        host_x, host_y = torch.nn.Linear(6, 4).double(), torch.nn.Linear(6, 3).double()
        inp = torch.randn(5, 6, dtype=torch.float64)
        x, y = host_x(inp), host_y(inp)
        reg.set_labels(torch.tensor([.3, .5, .8, .2, .6], dtype=torch.float64))
        controller = ApplicationUpdate({"dep": reg}, policy(), .1)
        right_before = reg.head.right.weight.detach().clone()
        controller.fit_critics({"dep": (x, y)})
        self.assertFalse(torch.equal(reg.head.right.weight, right_before))   # critic fit used the (detached) bank
        self.assertTrue(all(p.grad is None for p in list(host_x.parameters()) + list(host_y.parameters())))
        loss = controller.outer_loss("dep", x, y)
        loss.backward()
        self.assertGreater(float(host_x.weight.grad.norm()), 0.)
        self.assertIsNone(host_y.weight.grad)
        self.assertTrue(all(p.grad is None for p in reg.parameters()))
        self.assertFalse(reg.bank.requires_grad)
        reg.enqueue(y)
        self.assertFalse(reg.bank.requires_grad)
        self.assertEqual(int(reg.bank_count), 7)

    def test_critic_init_and_host_rng_match_package_arms(self):
        from aigiqa.models import TinyCMIM
        from aigiqa.engine import build_losses
        host = TinyCMIM()
        draws, heads = [], {}
        for name, extra in (("task_only", dict(method="none", weight=0.)),
                            ("inbatch", dict(method="vcs_exact", weight=.1)),
                            ("exact_bank_mosw", dict(method="vcs_exact", weight=.1, bank_size=8, p_weighting="mos")),
                            ("bce_bank_mosw", dict(method="bce", weight=.1, bank_size=8, p_weighting="mos"))):
            torch.manual_seed(765)
            cfg = dict(seed=4101, loss=dict(branches=["text"], forward="none", rank=4,
                                            application_update=policy(), **extra))
            losses = build_losses(host, cfg, "cpu")
            draws.append(torch.rand(6))
            if name != "task_only":
                heads[name] = losses.regularizers["text"].head.state_dict()
        for value in draws[1:]:
            torch.testing.assert_close(value, draws[0], rtol=0., atol=0.)
        for name in ("exact_bank_mosw", "bce_bank_mosw"):
            for k, v in heads["inbatch"].items():
                torch.testing.assert_close(heads[name][k], v, rtol=0., atol=0.)

    def test_engine_synthetic_banked_mos_FIT_CAL_only(self):
        import numpy as np
        from PIL import Image
        from unittest.mock import patch
        from aigiqa.manifest import prepare, write_csv
        from aigiqa.engine import train
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); images = root / "images"; images.mkdir()
            rng = np.random.default_rng(42)
            rows = []
            for i, word in enumerate(["amber", "beacon", "cactus", "dahlia", "ember", "fjord", "galaxy", "harbor",
                                      "indigo", "jungle", "koala", "lotus", "maple", "nebula", "opal", "prairie",
                                      "quartz", "raven", "saffron", "tundra"]):
                for j in range(3):
                    name = f"{i}_{j}.png"
                    Image.fromarray(rng.integers(0, 80, (24, 24, 3), dtype=np.uint8) + j * 50).save(images / name)
                    rows.append(dict(image=name, prompt=word, quality_mos=1 + j + .02 * i,
                                     alignment_mos=1 + .8 * j + .025 * i, generator=f"g{j}"))
            write_csv(root / "annotations.csv", rows, list(rows[0]))
            prepared = root / "prepared"
            prepare(root / "annotations.csv", images, prepared, "canonical")
            for method in ("vcs_exact", "bce"):
                out = root / f"run_{method}"
                cfg = dict(model={"name": "tiny_synthetic"}, data_dir=str(prepared), output_dir=str(out),
                           target="alignment", evidence_role="SYNTHETIC_INTEGRATION_ONLY", seed=95, device="cpu",
                           image_size=24, batch_size=5, num_workers=0, epochs=3, learning_rate=.001,
                           loss=dict(method=method, forward="none", branches=["text"], rank=4, weight=.1,
                                     bank_size=8, p_weighting="mos", application_update=policy(warmup=1, ramp=1)))
                original_open = Path.open
                def guarded_open(path, *args, **kwargs):
                    if path.name in {"test_inputs.csv", "test_labels.csv", "all_inputs.csv", "all_labels.csv"}:
                        raise AssertionError("Training attempted to read TEST")
                    return original_open(path, *args, **kwargs)
                with patch.object(Path, "open", guarded_open):
                    result = train(cfg)
                self.assertEqual(result["status"], "COMPLETED_FIT_CAL_ONLY")
                self.assertGreater(result["critic_updates_total"], 0)
                saved = torch.load(out / "best.pt", weights_only=True)
                self.assertEqual(int(saved["losses"]["regularizers.text.bank_count"]), 8)
                history = json.loads((out / "history.json").read_text())
                diag = history[-1]["train_diagnostics"]
                self.assertEqual(diag["text_bank_count"], 8)
                self.assertEqual(diag["text_q_pairs"], 5 * 4 + 5 * 8)
                self.assertGreater(diag["text_p_weight_max"], 1.)
                MEASUREMENTS[f"aig_synthetic_{method}_last_raw_j"] = diag["text_raw_j"]

    @unittest.skipUnless(os.access(AIG_FIT, os.R_OK), "real FIT csv not readable here")
    def test_real_fit_alignment_mos_nonnegative(self):
        with open(AIG_FIT, newline="") as f:
            mos = [float(r["alignment_mos"]) for r in csv.DictReader(f)]
        MEASUREMENTS["aig_fit_alignment_mos_min"] = min(mos)
        MEASUREMENTS["aig_fit_alignment_mos_zero_count"] = sum(m == 0 for m in mos)
        MEASUREMENTS["aig_fit_n"] = len(mos)
        self.assertGreaterEqual(min(mos), 0.)


class GuardTests(unittest.TestCase):
    def test_guard_refuses_unlisted_files_in_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"; root.mkdir()
            (root / "train").write_text("ok"); (root / "test").write_text("held out")
            outside = Path(tmp) / "elsewhere.txt"; outside.write_text("x")
            code = f"""
import sys; sys.path.insert(0, {str(HERE)!r})
import guard
guard.install([({str(root)!r}, [{str(root / 'train')!r}])])
open({str(root / 'train')!r}).read(); open({str(outside)!r}).read()
try:
    open({str(root / 'test')!r}).read(); print('OPENED')
except PermissionError:
    print('REFUSED')
import os
try:
    os.open({str(root / 'test')!r}, os.O_RDONLY); print('OPENED')
except PermissionError:
    print('REFUSED')
"""
            out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=60)
            self.assertEqual(out.stdout.split(), ["REFUSED", "REFUSED"], out.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--only", choices=["kd", "aig", "all"], default="all")
    args = parser.parse_args()
    sys.path[:0] = [str(HERE), str(ENGINE), str(ENGINE / "shared")]
    global torch, np
    import numpy as np
    import torch
    import unittest.mock  # noqa: F401
    cases = {"kd": [KDTests, GuardTests], "aig": [AIGTests, GuardTests], "all": [KDTests, AIGTests, GuardTests]}[args.only]
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in cases)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(schema="r22_cpu_tests_v1", status="PASS" if result.wasSuccessful() else "FAIL",
                  tests_run=result.testsRun, errors=[str(e[0]) for e in result.errors],
                  failures=[str(f[0]) for f in result.failures], skipped=[str(s[0]) for s in result.skipped],
                  measurements=MEASUREMENTS, torch=torch.__version__, python=sys.executable,
                  host=os.uname().nodename, slurm_job=os.environ.get("SLURM_JOB_ID"))
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
