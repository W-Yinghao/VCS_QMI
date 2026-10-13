"""r7 E2 (Qwen2.5-VL-7B on FIT_FULL) on CPU — synthetic data, the frozen fake Qwen host and stubbed sbatch / squeue; no GQA / CAL / DEV / FIT_FULL
record, prediction or result is read.  Unit plan (partitions, exclusions, time limits, markers, no --export; stage 1 / choice / stage 2 with the
frozen rule; run configs for ROWS selectors / core and their guards; eval lines); the data stage on a synthetic frozen + full manifest whose
R7-40a-style records are written by the frozen teacher with the fake host (verification, static reuse check, tampering, replay switch, verified
copies, teacher lines, refusals) and the E2 teacher bundles + the frozen completeness on its output; the cache check; the replay comparison; the
memory-smoke data builder with the frozen trainer (tiny, CPU); the R7-40c row reuse check (pass + copy, mismatch, GPU / set guards, ROWS=core);
the readout (frozen R7-40c scorer on both scales, claim rule, scale contrast, refusals); the driver dry run; the protocol draft parses.
Run: python -I tests/test_r7_e2.py -v   (main env; also vl_baselines / SparseVLM via slurm/r7_e2_cpu.sbatch tests_vl / tests_sv)."""
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts")); sys.path.insert(0, str(R / "tests")); sys.path.insert(0, str(R / "r7" / "package_r7_20261011" / "reference"))
import r7_d1_30_run as RUN  # noqa: E402
import r7_d1_40_analyze as AN40  # noqa: E402
import r7_d1_40_masks as M4  # noqa: E402
import r7_d1_40_selectors as S4  # noqa: E402
import r7_d1_40_teacher as TE40  # noqa: E402
import r7_d1_40c_plan as PL  # noqa: E402
import r7_d1_40c_score as SCO  # noqa: E402
import r7_d1_analyze as AN  # noqa: E402
import r7_d1_data as DA  # noqa: E402
import r7_d1full_data as FD  # noqa: E402
import r7_e2_data as ED  # noqa: E402
import r7_e2_plan as EP  # noqa: E402
import r7_e2_score as ES  # noqa: E402
import r7_e2_smoke as ESM  # noqa: E402
from test_r7_d1_40 import FakeQwen, grid_of_image  # noqa: E402   (frozen R7-40 CPU stand-ins)
from test_r7_d1full import frozen_and_full  # noqa: E402            (frozen D1-FULL synthetic frozen + full manifests)

try:
    import llava.eval.m4c_evaluator  # noqa: F401
    HAS_LLAVA = True
except Exception:
    HAS_LLAVA = False

LR = {"vcs": {"chosen_lr": 3e-4}, "logistic": {"chosen_lr": 1e-4}, "direct": {"chosen_lr": 3e-4}}
L40S = "NVIDIA L40S"


# ------------------------------------------------------------------------------------------------------------------------------ plan
class Plan(unittest.TestCase):
    def test_lines_partitions_markers(self):
        root = Path("/x/E2"); plan = {"bundles": {"f000": ["FIT-040", "FIT-041"], "f001": ["FIT-042"]}}
        for line, mark in EP.teacher_lines(plan, root):
            self.assertIn("--partition=L40S --exclude=node51,node60 --time=02:00:00 slurm/r7_e2_gpu.sbatch bundle f00", line); self.assertNotIn("--export", line)
            self.assertEqual(mark, root / "teacher" / f"bundle_{line.split()[-1]}.done.json")
        with self.assertRaises(SystemExit):                                                    # CAL is reused, DEV never computed
            EP.teacher_lines({"bundles": {"f000": ["FIT-040", "DEV-000"]}}, root)
        meta = {"counts": {"FIT": {"images": 70256}}}
        (c, m), = EP.cache_lines(meta, root); self.assertTrue(c.endswith("slurm/r7_e2_gpu.sbatch cache 70256")); self.assertEqual(m, root / "cache" / "cache_d1_40.json")
        self.assertIn("--partition=RTX6000PRO,H100,L40S --exclude=node51,node60 --time=10:00:00", c)
        self.assertTrue(EP.cache_lines(meta, root, "frozen10k")[0][0].endswith("cache 10000"))
        for bad in ("RTX6000PRO;rm", "H100,A100", ""):
            with self.assertRaises(SystemExit):
                EP.cache_lines(meta, root, "fitfull", bad)
        with self.assertRaises(SystemExit):
            EP.pca_images(meta, "other")
        s1 = EP.stage1(root); self.assertEqual(len(s1), 6)
        self.assertEqual(sorted(m.name for _, m in s1), sorted(f"{o}_lr{lr:g}_s0.done.json" for o in ("vcs", "logistic", "direct") for lr in (1e-4, 3e-4)))
        for line, _ in s1:
            self.assertIn("--partition=RTX6000PRO,H100 --exclude=node51,node60 --time=04:30:00 slurm/r7_e2_sel.sbatch train ", line); self.assertNotIn("L40S", line)
        s2 = EP.stage2(LR, root); self.assertEqual([m.name for _, m in s2], ["vcs_lr0.0003_s1.done.json", "vcs_lr0.0003_s2.done.json", "logistic_lr0.0001_s1.done.json",
                                                                              "logistic_lr0.0001_s2.done.json", "direct_lr0.0003_s1.done.json", "direct_lr0.0003_s2.done.json"])
        self.assertTrue(s2[0][0].endswith("train vcs 0.0003 1")); self.assertEqual(len(set(EP.unit_tags(LR))), 12)
        ev = EP.eval_lines(root); self.assertEqual(len(ev), 12)
        self.assertEqual(sum("run gqa_dev " in c for c, _ in ev), 6); self.assertEqual(sum("run textvqa_val " in c for c, _ in ev), 4); self.assertEqual(sum("run chartqa_val " in c for c, _ in ev), 2)
        for c, m in ev:
            self.assertIn("--partition=L40S --exclude=node51,node60", c); self.assertNotIn("RTX6000PRO", c); self.assertTrue(m.name.startswith("done_shard"))
        self.assertIn("--time=03:45:00", next(c for c, _ in ev if "textvqa_val" in c))

    def test_choose_frozen_rule(self):
        with tempfile.TemporaryDirectory() as t:
            sd = Path(t)
            for o, (a, b) in {"vcs": (0.10, 0.20), "logistic": (0.30, 0.20), "direct": (0.15, 0.15005)}.items():
                for lr, v in ((1e-4, a), (3e-4, b)):
                    (sd / f"{o}_lr{lr:g}_s0.json").write_text(json.dumps({"selected_cal_spearman": v})); (sd / f"{o}_lr{lr:g}_s0.pt").write_bytes(b"x")
            ch = EP.choose(sd); self.assertEqual({o: v["chosen_lr"] for o, v in ch.items()}, {"vcs": 3e-4, "logistic": 1e-4, "direct": 3e-4})  # tie within 1e-4 -> 3e-4
            self.assertEqual(json.loads((sd / "lr_choice.json").read_text())["vcs"]["chosen_lr"], 3e-4); self.assertIn("frozen", ch["vcs"]["rule"])
            (sd / "vcs_lr0.0001_s0.pt").unlink()
            with self.assertRaises(SystemExit):
                EP.choose(sd)

    def test_run_configs(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); PL.write_configs(LR, t / "cfg", "core")
            res = EP.write_run_configs(t / "cfg", t / "run", "selectors"); self.assertEqual(res["gqa_dev"], 46)
            run = PL.read_configs(t / "run" / "configs_gqa_dev.txt"); self.assertEqual(run[0], "full"); self.assertFalse(any(c.startswith("static:") for c in run))
            self.assertEqual(sum(c.startswith("sel:") for c in run), 36); self.assertEqual(sum(c.startswith("dyn:vcs_lr0.0003_s") for c in run), 9)
            EP.write_run_configs(t / "cfg", t / "run", "selectors")                              # idempotent
            with self.assertRaises(SystemExit):                                                    # another ROWS needs a fresh eval dir
                EP.write_run_configs(t / "cfg", t / "run", "core")
            self.assertEqual(EP.write_run_configs(t / "cfg", t / "run2", "core")["chartqa_val"], 50)
            PL.write_configs({**LR, "vcs": {"chosen_lr": 1e-4}}, t / "cfg", "core")
            with self.assertRaises(SystemExit):                                                    # lr choice changed under existing run lists
                EP.write_run_configs(t / "cfg", t / "run", "selectors")
            PL.write_configs(LR, t / "cfg3", "static")
            with self.assertRaises(SystemExit):                                                    # only the R7-40c core list is accepted
                EP.write_run_configs(t / "cfg3", t / "run3", "selectors")


# ------------------------------------------------------------------------------------------------------------------------------ data stage
class Src:
    def open(self, i):
        return {"grid": grid_of_image(i)}


def r40a_records(fd: Path, td: Path, plan: dict):
    """R7-40a-style records of the frozen CAL / FIT shards: the frozen teacher (cmd_bundle) with the fake host, one bundle per shard; done records
    then carry the R7-40a GPU class (the fake host on CPU records none)."""
    gen = fd.parent / "gen"; gen.mkdir(exist_ok=True)
    for f in ("d1_manifest.jsonl", "d1_manifest_meta.json"):
        shutil.copy(fd / f, gen / f)
    shards = [s for s in plan["shards"] if s.startswith(("CAL-", "FIT-"))]
    (gen / "d1_shards.json").write_text(json.dumps(dict(plan, bundles={f"g{i:03d}": [s] for i, s in enumerate(shards)}))); (gen / "probe.json").write_text('{"pass": true}')
    for i in range(len(shards)):
        TE40.cmd_bundle(Namespace(bundle=f"g{i:03d}", manifest_dir=str(gen), out=str(td), images_per_chunk=2, max_tokens=4096, max_seqs=16, allow_dev=False,
                                  probe_report=str(gen / "probe.json")), host_factory=lambda: FakeQwen(), image_source=Src())
    for s in shards:
        p = td / f"{s}.done.json"; d = json.loads(p.read_text()); d["gpu"] = L40S; p.write_text(json.dumps(d))
    return shards


def e2_world(t: Path, replay_pass=True, frozen=True):
    fd, dd, man, plan, fman, fmeta = frozen_and_full(t)
    meta = json.loads((dd / "d1_manifest_meta.json").read_text()); meta["frozen"] = True; (dd / "d1_manifest_meta.json").write_text(json.dumps(meta))
    (dd / "d1full_verify.json").write_text(json.dumps({"ok": True, "d1full_manifest_sha256": DA.sha256_file(dd / "d1_manifest.jsonl")}))
    dplan = FD.make_plan(fman, plan, True, 5, 3); (dd / "d1_shards.json").write_text(json.dumps(dplan))
    td = t / "R40" / "teacher"; td.mkdir(parents=True); r40a_records(fd, td, plan)
    e2 = t / "E2"; (e2 / "smoke").mkdir(parents=True); (e2 / "smoke" / "reuse_replay.json").write_text(json.dumps({"pass": replay_pass, "gpu": L40S}))
    proto = t / "proto.yaml"; proto.write_text("r7_40c_staged_freeze: >\n  x\n" + ("r7_e2a_staged_freeze: >\n  y\n" if frozen else "notes: r7_e2a_staged_freeze comes later\n"))
    return {"fd": fd, "dd": dd, "td": td, "e2": e2, "proto": proto, "plan": plan, "dplan": dplan, "fman": fman}


def data_args(w, cmd="data", extra=()):
    a = ["--root", str(w["e2"]), cmd, "--df-data", str(w["dd"]), "--d1-data", str(w["fd"]), "--r40-teacher", str(w["td"]), "--full-sha", "", "--frozen-sha", "", "--no-code"]
    if cmd == "data":
        a += ["--lines", str(w["e2"] / "lines" / "teacher_lines.txt"), "--protocol", str(w["proto"])]
    return a + list(extra)


class DataStage(unittest.TestCase):
    def test_verify_static_reuse_copy_lines_and_e2_bundles(self):
        with tempfile.TemporaryDirectory() as t:
            w = e2_world(Path(t)); v, rows, fplan = ED.verify_sources(w["dd"], w["fd"], None, None)
            self.assertTrue(v["ok"], v["problems"]); self.assertTrue(v["frozen_is_byte_prefix"]); self.assertTrue(v["frozen_fit_are_first_fit_images"])
            self.assertEqual(ED.main(data_args(w, "precheck", ["--out", str(Path(t) / "pre")])), 0)
            pre = json.loads((Path(t) / "pre" / "precheck.json").read_text()); self.assertTrue(pre["reuse"]["fit"]["static_ok"]); self.assertEqual(pre["outcome_now"], "reuse_fit")
            self.assertGreater(pre["plan_dryrun"]["recompute_fit"]["bundles"], pre["plan_dryrun"]["reuse_fit"]["bundles"])
            self.assertFalse((w["e2"] / "data").exists())                                          # the precheck copies and plans nothing
            self.assertEqual(ED.main(data_args(w)), 0)
            dr = json.loads((w["e2"] / "data" / "data_ready.json").read_text()); self.assertTrue(dr["fit_reuse"])
            self.assertEqual(DA.sha256_file(w["e2"] / "data" / "d1_manifest.jsonl"), DA.sha256_file(w["dd"] / "d1_manifest.jsonl"))
            ep = json.loads((w["e2"] / "data" / "d1_shards.json").read_text()); self.assertEqual(ep["bundles"], w["dplan"]["bundles"]); self.assertEqual(ep["shards"], w["dplan"]["shards"])
            self.assertNotIn("d1full", ep); self.assertTrue(ep["e2"]["reuse_fit"]); self.assertFalse(any(s.startswith("DEV-") for s in ep["shards"]))
            reused = [s for s in w["plan"]["shards"] if s.startswith(("CAL-", "FIT-"))]
            for s in reused:
                for ext in (".jsonl", ".done.json"):
                    self.assertEqual(DA.sha256_file(w["e2"] / "teacher" / f"{s}{ext}"), DA.sha256_file(w["td"] / f"{s}{ext}"))
            lines = [x for x in (w["e2"] / "lines" / "teacher_lines.txt").read_text().splitlines() if not x.startswith("#")]
            self.assertEqual(len(lines), len(ep["bundles"])); self.assertTrue(all("--partition=L40S --exclude=node51,node60" in x and " #MARK " in x for x in lines))
            self.assertEqual(ED.main(data_args(w)), 0)                                             # re-run: identical files kept
            for b in ep["bundles"]:                                                                # E2's bundles with the frozen teacher on E2's data dir
                TE40.cmd_bundle(Namespace(bundle=b, manifest_dir=str(w["e2"] / "data"), out=str(w["e2"] / "teacher"), images_per_chunk=2, max_tokens=4096,
                                          max_seqs=16, allow_dev=False, probe_report=str(w["fd"].parent / "gen" / "probe.json")), host_factory=lambda: FakeQwen(), image_source=Src())
            for f in (w["e2"] / "teacher").glob("*.done.json"):                                  # the GPU class of the real E2 bundles (the fake host records none)
                if f.name.startswith(("FIT-", "CAL-")):
                    d = json.loads(f.read_text()); d["gpu"] = L40S; f.write_text(json.dumps(d))
            sha = json.loads((w["e2"] / "data" / "d1_manifest_meta.json").read_text())["manifest_sha256"]
            self.assertEqual(AN40.cmd_complete(Namespace(teacher_dir=str(w["e2"] / "teacher"), manifest_dir=str(w["e2"] / "data"), frozen_sha256=sha, roles=["CAL", "FIT"])), 0)
            rows_fit, cov = AN.load_role(w["e2"] / "teacher", w["e2"] / "data", "FIT"); self.assertEqual(cov["present"], cov["expected"])
            self.assertEqual(cov["expected"], sum(r["role"] == "FIT" for r in w["fman"]))

    def test_tampering_replay_and_refusals(self):
        with tempfile.TemporaryDirectory() as t:                                                  # a changed FIT record -> FIT recomputed, CAL still copied
            w = e2_world(Path(t)); f = w["td"] / "FIT-001.jsonl"; recs = [json.loads(x) for x in f.read_text().splitlines()]
            recs[0]["masked"]["grid"]["0.25"][0] += 1e-3; f.write_text("".join(json.dumps(r) + "\n" for r in recs))
            self.assertEqual(ED.main(data_args(w)), 0); ep = json.loads((w["e2"] / "data" / "d1_shards.json").read_text())
            self.assertFalse(ep["e2"]["reuse_fit"]); allb = {s for v in ep["bundles"].values() for s in v}
            self.assertTrue({s for s in w["plan"]["shards"] if s.startswith("FIT-")} <= allb); self.assertFalse(any(s.startswith(("CAL-", "DEV-")) for s in allb))
            self.assertTrue((w["e2"] / "teacher" / "CAL-000.jsonl").exists()); self.assertFalse((w["e2"] / "teacher" / "FIT-000.jsonl").exists())
            rc = json.loads((w["e2"] / "teacher" / "reuse_check.json").read_text()); self.assertIn("FIT-001", [s for s, v in rc["shards"].items() if not v["ok"]])
        with tempfile.TemporaryDirectory() as t:                                                  # replay not passing -> recompute
            w = e2_world(Path(t), replay_pass=False); ED.main(data_args(w))
            self.assertFalse(json.loads((w["e2"] / "data" / "data_ready.json").read_text())["fit_reuse"])
        with tempfile.TemporaryDirectory() as t:                                                  # a changed CAL record stops the data stage; a mask digest too
            w = e2_world(Path(t)); f = w["td"] / "CAL-000.jsonl"; recs = [json.loads(x) for x in f.read_text().splitlines()]
            recs[1]["order_digest"]["random"] = "0" * 16; f.write_text("".join(json.dumps(r) + "\n" for r in recs))
            with self.assertRaises(SystemExit):
                ED.main(data_args(w))
            self.assertFalse((w["e2"] / "data" / "data_ready.json").exists())
        with tempfile.TemporaryDirectory() as t:                                                  # no freeze key (only a mention) / no replay file / not L40S
            w = e2_world(Path(t), frozen=False)
            with self.assertRaises(SystemExit):
                ED.main(data_args(w))
            w["proto"].write_text("r7_e2_staged_freeze: >\n  y\n")                                # the later-stage key alone does not open the teacher stage
            with self.assertRaises(SystemExit):
                ED.main(data_args(w))
            w["proto"].write_text("r7_e2a_staged_freeze: >\n  y\n"); (w["e2"] / "smoke" / "reuse_replay.json").unlink()
            with self.assertRaises(SystemExit):
                ED.main(data_args(w))
            (w["e2"] / "smoke" / "reuse_replay.json").write_text('{"pass": true}'); p = w["td"] / "FIT-000.done.json"; d = json.loads(p.read_text()); d["gpu"] = "NVIDIA H100 NVL"
            p.write_text(json.dumps(d)); ED.main(data_args(w)); self.assertFalse(json.loads((w["e2"] / "data" / "data_ready.json").read_text())["fit_reuse"])

    def test_identity_checks_on_this_repo(self):
        import time
        code, ok = ED.code_identity40(time.time() + 3600); self.assertTrue(ok, code)               # the frozen teacher code is clean at HEAD
        s = ED.settings_check40(); self.assertTrue(s["ok"], s); self.assertEqual(s["teacher_defaults"], ED.TEACHER_DEFAULTS)
        with tempfile.TemporaryDirectory() as t:
            fz = Path(t) / "FREEZE.txt"; fz.write_text("x"); md = Path(t) / "model"; md.mkdir(); (md / "w.bin").write_text("y")
            rec, ok = ED.env_identity40(time.time() + 60, fz, md); self.assertTrue(ok); self.assertEqual(len(rec["files"]), 2)
            self.assertFalse(ED.env_identity40(time.time() - 3600, fz, md)[1])

    def test_runner_code_unchanged_since_r7_40c(self):
        out, ok = ED.runner_code_identity(); self.assertTrue(ok, {k: v for k, v in out.items() if not v["same"]})


class CacheCheck(unittest.TestCase):
    def test_cache_check(self):
        with tempfile.TemporaryDirectory() as t:
            w = e2_world(Path(t)); ED.main(data_args(w)); c = w["e2"] / "cache"; c.mkdir(); rows = DA.load_manifest(w["e2"] / "data")
            n_fit = len(FD.role_images(rows, "FIT")); rec = {"mode": "d1_40", "pca_tokens": 64 * n_fit, "explained_variance": 0.9}
            for role in ("FIT", "CAL"):
                ids = FD.role_images(rows, role); g = [list(grid_of_image(i)) for i in ids]; n = [a * b for a, b in g]
                (c / f"images_{role}.json").write_text(json.dumps({"images": ids, "offsets": np.concatenate([[0], np.cumsum(n)[:-1]]).tolist(), "n": n, "grid": g}))
                np.save(c / f"feats_{role}.npy", np.zeros((sum(n), 512), np.float16)); rec[f"{role}_images"] = len(ids); rec[f"{role}_tokens"] = sum(n)
            (c / "question_ids.jsonl").write_text("".join(json.dumps({"qid": r["qid"], "ids": [1]}) + "\n" for r in rows if r["role"] in ("FIT", "CAL")))
            for f in ("pca.pt", "embed_table.pt"):
                (c / f).write_bytes(b"x")
            (c / "cache_d1_40.json").write_text(json.dumps(rec))
            r = ED.cache_check(c, w["e2"] / "data", "fitfull"); self.assertTrue(r["pass"], r["problems"])
            r = ED.cache_check(c, w["e2"] / "data", "frozen10k", Path(t) / "r40cache")          # PCA image count of the other basis
            self.assertFalse(r["pass"]); self.assertNotIn("pca_vs_r7_40b", r)
            (Path(t) / "r40cache").mkdir(); (Path(t) / "r40cache" / "pca.pt").write_bytes(b"y")    # an unreadable reference is reported, not fatal
            self.assertIn("error", ED.cache_check(c, w["e2"] / "data", "frozen10k", Path(t) / "r40cache")["pca_vs_r7_40b"])
            ix = json.loads((c / "images_FIT.json").read_text()); ix["images"] = ix["images"][::-1]; (c / "images_FIT.json").write_text(json.dumps(ix))
            self.assertFalse(ED.cache_check(c, w["e2"] / "data", "fitfull")["pass"])


# ------------------------------------------------------------------------------------------------------------------------------ smoke helpers
class SmokeHelpers(unittest.TestCase):
    def test_replay_compare(self):
        with tempfile.TemporaryDirectory() as t:
            w = e2_world(Path(t))
            rep = ESM.replay(FakeQwen(), w["dd"], w["td"], ("FIT-000", "FIT-001"), 2, Src(), L40S); self.assertTrue(rep["pass"], rep); self.assertGreater(rep["records"], 0)
            self.assertEqual(rep["max_abs_dP_yes"], 0.0); self.assertFalse(ESM.replay(FakeQwen(), w["dd"], w["td"], ("FIT-000",), 2, Src(), "NVIDIA H100 NVL")["pass"])
            with self.assertRaises(SystemExit):
                ESM.replay(FakeQwen(), w["dd"], w["td"], ("DEV-000",), 2, Src(), L40S)
            f = w["td"] / "FIT-000.jsonl"; recs = [json.loads(x) for x in f.read_text().splitlines()]; recs[0]["full"][0] += 0.01
            f.write_text("".join(json.dumps(r) + "\n" for r in recs))
            rep = ESM.replay(FakeQwen(), w["dd"], w["td"], ("FIT-000",), 2, Src(), L40S); self.assertFalse(rep["pass"]); self.assertAlmostEqual(rep["max_abs_dP_yes"], 0.01, places=6)

    def test_memsmoke_builder_with_frozen_trainer(self):
        cfg = S4.Cfg40(emb_dim=8, vis_dim=4, d=16, heads=2, layers=1, ffn=32, dropout=0.0, max_rows=8, max_cols=8, steps_a=4, steps_b=2, eval_every=2, warmup=1, q_per_step=8)
        emb = torch.randn(30, 8, dtype=torch.bfloat16); ntok, cols = [12, 16, 10, 20], [4, 4, 5, 5]
        fit = ESM.synth(ntok, cols, 6, 21, 3, 20, emb, cfg, "cpu", seed=1); cal = ESM.synth(ntok, cols, 4, 15, 3, 20, emb, cfg, "cpu", seed=2)
        self.assertEqual(fit.n, 21); self.assertEqual(int(fit.ntok.sum()), 12 + 16 + 10 + 20 + 12 + 16); self.assertEqual(tuple(fit.rank.shape), (21, 3, 20))
        for q in range(fit.n):
            n = int(fit.ntok[fit.img_of[q]])
            for m in range(3):
                r = fit.rank[q, m].long(); self.assertEqual(sorted(r[:n].tolist()), list(range(n))); self.assertTrue(bool((r[n:] == S4.PAD_RANK).all()))
            self.assertEqual(fit.kk[q].tolist(), [M4.budget(f, n) for f in M4.FRACTIONS])
        model, log = S4.train40(cfg, "direct", fit, cal, ESM.W_FIT_FULL); self.assertEqual(len(log["evals"]), 2)
        self.assertIn("tau", S4.dynamic_tau40(model, cal, cfg, 0.5)); self.assertGreater(ESM.unit_hours(log["timing"])["unit_hours_estimate"], 0)
        with self.assertRaises(SystemExit):
            ESM.synth(ntok, cols, 6, 30, 3, 20, emb, cfg, "cpu")                                 # 3-4 questions per image


# ------------------------------------------------------------------------------------------------------------------------------ R7-40c rows
def rows_world(t: Path, rows="selectors", gpu=L40S, mismatch=False):
    sets = t / "sets"; sets.mkdir(); summ = {}; r40c = t / "R40C"; e2p = t / "E2" / "eval" / "pred"
    (r40c).mkdir(parents=True); (r40c / "completeness.json").write_text('{"complete": true}')
    for b in PL.BENCHES:
        items = [{"bench": b, "qid": f"{b}_{i}", "image_id": f"img{i // 2}", "image_file": "x", "prompt": "?", "answer": "yes"} for i in range(6)]
        (sets / f"{b}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in items)); s = DA.sha256_file(sets / f"{b}.jsonl"); summ[b] = {"sha256": s, "items": 6}
        n = PL.QUALITY_SHARDS[b]
        for side, d in (("r", r40c / "pred" / b), ("e", e2p / b)):
            d.mkdir(parents=True)
            for i in range(n):
                (d / f"done_shard{i}of{n}.json").write_text(json.dumps({"env": {"gpu": gpu if side == "r" else L40S}, "sets_file_sha256": s}))
            cfgs = ["full"] + (list(ED.GRID_ROWS) if (side == "r" or rows == "core") else [])
            for c in cfgs:
                recs = [{"qid": x["qid"], "image_id": x["image_id"], "pred": "yes" if c == "full" else c[-4:], "kept": 10, "n_vis": 20, "f": 1.0, "n_new": 2, "hit_max": False} for x in items]
                if mismatch and side == "e" and c == "full":
                    recs[3]["pred"] = "no"
                (d / f"{RUN.cfg_file(c)}@s0of{n}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    (sets / "d1_30_sets.json").write_text(json.dumps(summ))
    return sets, r40c, e2p


class ReuseRows(unittest.TestCase):
    def test_pass_and_copy(self):
        with tempfile.TemporaryDirectory() as t:
            sets, r40c, e2p = rows_world(Path(t)); res = ED.reuse_rows(e2p, r40c, sets, "selectors", code=False)
            self.assertTrue(res["pass"], res["problems"]); self.assertTrue(res["gate"]); self.assertEqual(len(res["copied"]), 4 * len(PL.BENCHES))
            for b in PL.BENCHES:
                for c in ED.GRID_ROWS:
                    f = f"{RUN.cfg_file(c)}@s0of{PL.QUALITY_SHARDS[b]}.jsonl"; self.assertEqual(DA.sha256_file(e2p / b / f), DA.sha256_file(r40c / "pred" / b / f))
                self.assertEqual(res["benches"][b]["full"]["identical_fraction"], 1.0)
            res2 = ED.reuse_rows(e2p, r40c, sets, "selectors", code=False); self.assertTrue(all(c.get("already_present") for c in res2["copied"]))
            f = e2p / "gqa_dev" / f"{RUN.cfg_file(ED.GRID_ROWS[0])}@s0of6.jsonl"; f.write_text(f.read_text() + "\n")
            with self.assertRaises(SystemExit):                                                    # never overwrites a different file
                ED.reuse_rows(e2p, r40c, sets, "selectors", code=False)

    def test_failures_and_core_mode(self):
        with tempfile.TemporaryDirectory() as t:
            sets, r40c, e2p = rows_world(Path(t), mismatch=True); res = ED.reuse_rows(e2p, r40c, sets, "selectors", code=False)
            self.assertFalse(res["pass"]); self.assertFalse(res["full_row_identity_ok"]); self.assertEqual(res["copied"], [])
            self.assertAlmostEqual(res["benches"]["gqa_dev"]["full"]["identical_fraction"], 5 / 6)
        with tempfile.TemporaryDirectory() as t:
            sets, r40c, e2p = rows_world(Path(t), gpu="NVIDIA H100 NVL"); res = ED.reuse_rows(e2p, r40c, sets, "selectors", code=False)
            self.assertFalse(res["static_ok"]); self.assertFalse(res["pass"])
        with tempfile.TemporaryDirectory() as t:
            sets, r40c, e2p = rows_world(Path(t)); (sets / "gqa_dev.jsonl").write_text((sets / "gqa_dev.jsonl").read_text() + "\n")
            self.assertFalse(ED.reuse_rows(e2p, r40c, sets, "selectors", code=False)["static_ok"])    # set file changed since the markers
        with tempfile.TemporaryDirectory() as t:
            sets, r40c, e2p = rows_world(Path(t), rows="core"); res = ED.reuse_rows(e2p, r40c, sets, "core", code=False)
            self.assertTrue(res["pass"]); self.assertFalse(res["gate"]); self.assertEqual(res["copied"], [])
            self.assertEqual(res["benches"]["chartqa_val"][ED.GRID_ROWS[1]]["identical_fraction"], 1.0)


# ------------------------------------------------------------------------------------------------------------------------------ readout
class Readout(unittest.TestCase):
    BENCHES = ("gqa_dev", "chartqa_val") + (("textvqa_val",) if HAS_LLAVA else ())
    LRF = {"vcs": {"chosen_lr": 1e-4}, "logistic": {"chosen_lr": 1e-4}, "direct": {"chosen_lr": 3e-4}}

    def test_verdicts(self):
        c = lambda d, p: {"delta": d, "ci95": [d - 1, d + 1], "p_holm": p}  # noqa: E731
        self.assertEqual(ES.verdict(c(2, 0.01), c(1, 0.02)), "claim: S1 above"); self.assertEqual(ES.verdict(c(-2, 0.01), c(-1, 0.001)), "claim: S1 below")
        self.assertTrue(ES.verdict(c(2, 0.01), c(-1, 0.01)).startswith("scale-dependent")); self.assertIn("10k scale only", ES.verdict(c(2, 0.01), c(1, 0.2)))
        self.assertIn("full scale only", ES.verdict(c(2, 0.3), c(1, 0.01))); self.assertIn("neither", ES.verdict(c(2, 0.3), c(1, 0.2)))
        self.assertEqual(ES.verdict(None, c(1, 0.01)), "missing")

    def world(self, t: Path):
        sets, e2, r40 = t / "sets", t / "E2", t / "R40"; r40c = r40 / "eval_40c"; sets.mkdir(); g = np.random.default_rng(0)
        for lr, d in ((self.LRF, e2 / "selectors"), (LR, r40 / "selectors")):
            d.mkdir(parents=True); (d / "lr_choice.json").write_text(json.dumps(lr))
        PL.write_configs(self.LRF, e2 / "eval" / "configs", "core", self.BENCHES); PL.write_configs(LR, r40c / "configs", "core", self.BENCHES)
        acc = {0.125: 0.5, 0.25: 0.6, 0.5: 0.7, 0.75: 0.8, 1.0: 0.9}
        for bench in self.BENCHES:
            items = []
            for i in range(60):
                it = {"bench": bench, "qid": f"{bench}_{i}", "image_id": f"img{i // 3}", "question": "?", "prompt": "?"}
                it.update(answer="yes", structural=["verify", "query"][i % 2]) if bench.startswith("gqa") else (
                    it.update(answer="12", type=["human_test", "augmented_test"][i % 2]) if bench.startswith("chartqa") else it.update(answers=["yes"] * 10))
                items.append(it)
            (sets / f"{bench}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in items)); right = "12" if bench.startswith("chartqa") else "yes"
            for pred, lr, bonus in ((e2 / "eval" / "pred", self.LRF, 0.15), (r40c / "pred", LR, 0.05)):
                pd = pred / bench; pd.mkdir(parents=True)
                for c in PL.config_list(PL.unit_names(lr), "core"):
                    k, a, f = PL.parse(c); f = 1.0 if k == "full" else f; recs = []
                    for it in items:
                        n = 300 + (int(it["image_id"][3:]) % 50); fr = f if k != "dyn" else [0.25, 0.5, 0.75, 1.0][int(g.integers(4))]
                        p = acc.get(fr, 0.75) + (bonus if (k == "sel" and a.startswith("vcs")) else 0.0)
                        recs.append({"qid": it["qid"], "image_id": it["image_id"], "pred": right if g.uniform() < p else "no", "kept": n if fr >= 1 else M4.budget(fr, n),
                                     "n_vis": n, "f": fr, "n_new": 2, "hit_max": False})
                    (pd / f"{RUN.cfg_file(c)}@s0of1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        proto = t / "proto.yaml"; proto.write_text("r7_40c_staged_freeze: >\n  x\nr7_e2a_staged_freeze: >\n  y\nr7_e2_staged_freeze: >\n  z\n")
        rep = t / "rep.json"; rep.write_text(json.dumps({"gpu": "NVIDIA H100 NVL", "host": "nodesumo01", "B_star": None}))
        SCO.main(["--sets-dir", str(sets), "--pred-dir", str(r40c / "pred"), "--configs-dir", str(r40c / "configs"), "--lr-choice", str(r40 / "selectors" / "lr_choice.json"),
                  "--out", str(r40c / "results"), "--benches", *self.BENCHES, "--reps", "200", "--protocol", str(proto), "--timing-report", str(rep), "--anchor", "none"])
        return ["--sets-dir", str(sets), "--e2-root", str(e2), "--r40c", str(r40c), "--r40-lr-choice", str(r40 / "selectors" / "lr_choice.json"),
                "--r40c-results", str(r40c / "results" / "r7_40c_results.json"), "--out", str(e2 / "eval" / "results"), "--benches", *self.BENCHES, "--reps", "200",
                "--protocol", str(proto), "--timing-report", str(rep)], proto

    def test_claims_and_scale_contrast(self):
        with tempfile.TemporaryDirectory() as t:
            args, proto = self.world(Path(t)); ES.main(args); out = Path(args[args.index("--out") + 1]); nb = len(self.BENCHES)
            rf = json.loads((out / "r7_40c_results.json").read_text()); self.assertEqual(rf["primary_basis"], "fraction")
            res = json.loads((out / "e2_claims.json").read_text()); self.assertEqual(len(res["claims"]), 9 * nb)
            ok = {"claim: S1 above", "claim: S1 below", "scale-dependent: significant with opposite signs", "not claimed: neither scale significant",
                  "not claimed: significant at the 10k scale only (scale-dependent)", "not claimed: significant at the full scale only (scale-dependent)"}
            self.assertTrue({r["verdict"] for r in res["claims"]} <= ok); self.assertTrue(all(r["full"]["holm_family_size"] == 9 if nb == 3 else 3 * nb for r in res["claims"]))
            for r in res["claims"]:
                self.assertEqual(r["verdict"], ES.verdict(r["k10"], r["full"])); self.assertEqual(r["full"]["holm_family_size"], 3 * nb)
            self.assertEqual(sorted({r["contrast"] for r in res["claims"]}), sorted(f"S1-{o}@f{f}" for o in ("S2", "S3", "grid") for f in ("0.25", "0.5", "0.75")))
            sc = res["scale_contrast"]["gqa_dev"]; self.assertEqual(len(sc["fixed_fraction"]), 12); c = sc["fixed_fraction"]["S1@f0.5"]
            self.assertEqual(len(c["per_seed"]), 3); self.assertTrue(c["ci95"][0] - 1e-9 <= c["delta"] <= c["ci95"][1] + 1e-9)
            self.assertAlmostEqual(c["delta"], c["full_quality"] - c["k10_quality"], places=6); self.assertEqual(len(sc["dynamic"]["S1_dyn@op0.5"]["full"]["quality"]), 3)
            self.assertTrue((out / "e2_claims.md").read_text().startswith("# r7 E2 claims"))
            with self.assertRaises(SystemExit):                                                     # never overwrites
                ES.main(args)
        with tempfile.TemporaryDirectory() as t:
            args, proto = self.world(Path(t)); proto.write_text("r7_40c_staged_freeze: >\n  x\nr7_e2a_staged_freeze: >\n  y\nnote: r7_e2_staged_freeze pending\n")
            with self.assertRaises(SystemExit):
                ES.main(args)
            self.assertFalse(Path(args[args.index("--out") + 1]).exists())


# ------------------------------------------------------------------------------------------------------------------------------ driver
STUB_SBATCH = r'''#!/usr/bin/env bash
echo "$*" >> "$STUB_DIR/sbatch.log"
n=$(cat "$STUB_DIR/counter" 2>/dev/null || echo 3000); n=$((n + 1)); echo $n > "$STUB_DIR/counter"
args="$*"; E=$E2_ROOT
if [ -n "$STUB_FAIL_ON" ] && [[ "$args" == *"$STUB_FAIL_ON"* ]]; then exit 1; fi
mk() { mkdir -p "$(dirname "$1")"; echo '{"stub": true}' > "$1"; }
case "$args" in
  *"r7_e2_cpu.sbatch data "*) lines=${args##* }; mkdir -p "$E/data" "$(dirname "$lines")"; echo '{"counts": {"FIT": {"images": 1234}}}' > "$E/data/d1_manifest_meta.json"
    { echo "# teacher"; for b in f000 f001; do echo "sbatch --parsable --job-name=r7e2t_$b --partition=L40S --exclude=node51,node60 --time=02:00:00 slurm/r7_e2_gpu.sbatch bundle $b #MARK $E/teacher/bundle_$b.done.json"; done; } > "$lines"
    mk "$E/data/data_ready.json";;
  *"bundle f001"*) c=$(cat "$STUB_DIR/f001" 2>/dev/null || echo 0); echo $((c + 1)) > "$STUB_DIR/f001"; [ "$c" -ge "${STUB_F001_FAILS:-0}" ] && mk "$E/teacher/bundle_f001.done.json";;
  *"bundle f000"*) mk "$E/teacher/bundle_f000.done.json";;
  *"r7_e2_cpu.sbatch complete") mkdir -p "$E/teacher"; echo "{\"complete\": ${STUB_TEACHER_COMPLETE:-true}}" > "$E/teacher/r7_40a_completeness.json";;
  *"r7_e2_gpu.sbatch cache "*) mkdir -p "$E/cache"; echo "{\"max_images\": ${args##* }}" > "$E/cache/cache_d1_40.json";;
  *"r7_e2_cpu.sbatch cachecheck "*) mkdir -p "$E/cache"; echo "{\"pass\": ${STUB_CACHE_PASS:-true}}" > "$E/cache/e2_cache_check.json";;
  *"r7_e2_sel.sbatch train "*)
    if [[ "$args" =~ train\ ([a-z]+)\ ([0-9.e-]+)\ ([0-9]+) ]]; then
      t="${BASH_REMATCH[1]}_lr${BASH_REMATCH[2]}_s${BASH_REMATCH[3]}"; mkdir -p "$E/selectors"; touch "$E/selectors/$t.pt"; mk "$E/selectors/$t.done.json"
      v=$(python3 -c "import zlib; print(zlib.crc32(b'$t') % 1000 / 1000)"); echo "{\"selected_cal_spearman\": $v}" > "$E/selectors/$t.json"; fi;;
  *"r7_e2_cpu.sbatch choose"*) "$PY" scripts/r7_e2_plan.py choose --selectors-dir "$E/selectors" >> "$STUB_DIR/wrap.log" 2>&1 || exit 1;;
  *"r7_e2_cpu.sbatch check"*) "$PY" scripts/r7_d1_40c_plan.py configs --lr-choice "$E/selectors/lr_choice.json" --out "$E/eval/configs" --rows core >> "$STUB_DIR/wrap.log" 2>&1 || exit 1
    mkdir -p "$E/eval/check"; if [ -n "$STUB_CHECK_FAILS" ]; then echo '{"pass": false}' > "$E/eval/check/check.json"; else echo '{"pass": true}' > "$E/eval/check/check.json"; fi;;
  *"r7_e2_gpu.sbatch run "*) set -- $args; i=${@: -1}; b=${@: -2:1}; mk "$E/eval/pred/$b/done_shard${i%%/*}of${i##*/}.json";;
  *"r7_e2_cpu.sbatch runcomplete"*) mkdir -p "$E/eval/run"; echo '{"complete": true}' > "$E/eval/run/completeness.json";;
  *"r7_e2_cpu.sbatch reuserows "*) mkdir -p "$E/eval/reuse"; echo "{\"pass\": ${STUB_REUSE_PASS:-true}}" > "$E/eval/reuse/reuse_check.json";;
  *"r7_e2_cpu.sbatch corecomplete"*) mkdir -p "$E/eval"; echo '{"complete": true}' > "$E/eval/completeness.json";;
esac
echo $n
'''
STUB_SQUEUE = r'''#!/usr/bin/env bash
if [[ "$*" == *"-j"* ]]; then exit 0; fi
c=$(cat "$STUB_DIR/qcalls" 2>/dev/null || echo 0); c=$((c + 1)); echo $c > "$STUB_DIR/qcalls"
n=${STUB_QUEUE_N:-0}; [ "$c" -le "${STUB_FULL_CALLS:-0}" ] && n=27
for i in $(seq 1 $n); do echo "job$i"; done
'''


def run_driver(tmp: Path, stage="all", keys=("r7_e2a", "r7_e2"), probe=True, replay=True, teacher_done=False, teacher_complete=True, **env_extra):
    stub = tmp / "stub"; stub.mkdir(parents=True)
    for name, body in (("sbatch", STUB_SBATCH), ("squeue", STUB_SQUEUE)):
        f = stub / name; f.write_text(body); f.chmod(f.stat().st_mode | stat.S_IEXEC)
    e2 = tmp / "E2"; (e2 / "smoke" / "probe").mkdir(parents=True)
    if probe:
        (e2 / "smoke" / "probe" / "probe_report.json").write_text('{\n "pass": true\n}')
    if replay:
        (e2 / "smoke" / "reuse_replay.json").write_text('{"pass": true}')
    if teacher_done:                                                                        # the teacher stage's outputs (for 'rest' alone)
        (e2 / "data").mkdir(parents=True); (e2 / "data" / "data_ready.json").write_text("{}"); (e2 / "data" / "d1_manifest_meta.json").write_text('{"counts": {"FIT": {"images": 1234}}}')
        (e2 / "teacher").mkdir(parents=True); (e2 / "teacher" / "r7_40a_completeness.json").write_text(json.dumps({"complete": teacher_complete}))
    proto = tmp / "proto.yaml"; proto.write_text("r7_40c_staged_freeze: >\n  x\nnote: r7_e2a_staged_freeze and r7_e2_staged_freeze are added at the freezes\n"
                                                  + "".join(f"{k}_staged_freeze: >\n  y\n" for k in keys))
    env = dict(os.environ, PATH=f"{stub}:{os.environ['PATH']}", STUB_DIR=str(stub), E2_ROOT=str(e2), PROTOCOL=str(proto), SUBMIT_LOG=str(tmp / "submit.log"),
               LINES_DIR=str(tmp / "lines"), POLL="0", WAIT_H="1", SOON_S="0", SOON_POLL="1", PY=sys.executable, CAP="27", RETRIES="2", **{k: str(v) for k, v in env_extra.items()})
    p = subprocess.run(["bash", str(R / "slurm" / "r7_e2_submit.sh")] + ([stage] if stage else []), env=env, capture_output=True, text=True, timeout=600)
    calls = (stub / "sbatch.log").read_text().splitlines() if (stub / "sbatch.log").exists() else []
    return p, (tmp / "submit.log").read_text() if (tmp / "submit.log").exists() else "", calls, e2


def kind(c):
    for k, pat in (("data", "cpu.sbatch data"), ("bundle", "gpu.sbatch bundle"), ("cache", "gpu.sbatch cache"), ("cachecheck", "cpu.sbatch cachecheck"),
                   ("complete", "cpu.sbatch complete"), ("train", "sel.sbatch train"), ("choose", "cpu.sbatch choose"), ("check", "cpu.sbatch check"),
                   ("run", "gpu.sbatch run"), ("runcomplete", "cpu.sbatch runcomplete"), ("reuserows", "cpu.sbatch reuserows"), ("corecomplete", "cpu.sbatch corecomplete")):
        if pat in c:
            return k
    return "?"


TEACHER = ["data", "bundle", "bundle", "bundle", "complete"]                                 # bundle f001 fails once and is retried
REST = ["cache", "cachecheck"] + ["train"] * 6 + ["choose"] + ["train"] * 6 + ["check"] + ["run"] * 12 + ["runcomplete", "reuserows", "corecomplete"]


class DriverDryRun(unittest.TestCase):
    def common(self, calls):
        self.assertFalse(any("--dependency" in c or "--export" in c for c in calls)); self.assertFalse(any("score" in c for c in calls))
        self.assertTrue(all("--partition=CPU" in c and "--wait" in c for c in calls if kind(c) in ("data", "complete", "cachecheck", "choose", "check", "runcomplete", "reuserows", "corecomplete")))
        self.assertTrue(all("--exclude=node51,node60" in c for c in calls if "--partition=CPU" not in c))
        self.assertTrue(all("--partition=L40S --exclude" in c for c in calls if kind(c) in ("bundle", "run")))
        self.assertTrue(all("--partition=RTX6000PRO,H100 " in c and "L40S" not in c for c in calls if kind(c) == "train"))

    def test_teacher_stage_alone(self):
        with tempfile.TemporaryDirectory() as t:                                                   # only the teacher-stage key: the teacher stage runs, nothing else
            p, log, calls, e2 = run_driver(Path(t), stage="teacher", keys=("r7_e2a",), STUB_F001_FAILS=1, STUB_FULL_CALLS=3)
            self.assertEqual(p.returncode, 0, log + p.stderr); self.assertEqual([kind(c) for c in calls], TEACHER); self.common(calls)
            self.assertIn("RETRY 1/2", log); self.assertIn("done: E2 teacher stage complete", log); self.assertTrue(calls[-1].endswith("cpu.sbatch complete"))
            self.assertIn("bundle f000", calls[1]); self.assertIn("--time=02:00:00", calls[1])
            p, log2, calls2, _ = run_driver(Path(t) / "b", stage="teacher", keys=("r7_e2a",), STUB_TEACHER_COMPLETE="false")
            self.assertEqual(p.returncode, 1); self.assertIn("does not say", log2)

    def test_rest_stage_waits_for_the_teacher_and_never_submits_it(self):
        with tempfile.TemporaryDirectory() as t:
            p, log, calls, e2 = run_driver(Path(t), stage="rest", teacher_done=True)
            self.assertEqual(p.returncode, 0, log + p.stderr); self.assertEqual([kind(c) for c in calls], REST); self.common(calls)
            cache = next(c for c in calls if kind(c) == "cache"); self.assertIn("--partition=RTX6000PRO,H100,L40S", cache); self.assertTrue(cache.endswith("cache 1234"))
            self.assertTrue(next(c for c in calls if kind(c) == "cachecheck").endswith("cachecheck fitfull")); self.assertTrue(next(c for c in calls if kind(c) == "reuserows").endswith("reuserows selectors"))
            run = PL.read_configs(e2 / "eval" / "run" / "configs_gqa_dev.txt"); self.assertEqual(len(run), 46); self.assertEqual(run[0], "full")
            self.assertEqual(len(PL.read_configs(e2 / "eval" / "configs" / "configs_textvqa_val.txt")), 50); self.assertIn("done: E2 evaluation complete", log)
            self.assertEqual(set(json.loads((e2 / "selectors" / "lr_choice.json").read_text())), {"vcs", "logistic", "direct"})
        with tempfile.TemporaryDirectory() as t:                                                   # teacher not complete: the cache runs, then the gate times out
            p, log, calls, e2 = run_driver(Path(t), stage="rest", teacher_done=True, teacher_complete=False)
            self.assertEqual(p.returncode, 1); self.assertEqual([kind(c) for c in calls], ["cache", "cachecheck"]); self.assertIn("r7_40a_completeness.json not ready after", log)
        with tempfile.TemporaryDirectory() as t:
            tt = Path(t); p, log, calls, e2 = run_driver(tt, stage="rest", teacher_done=False)
            self.assertEqual(p.returncode, 1); self.assertIn("data_ready.json not ready after", log); self.assertEqual(calls, [])

    def test_all_and_options(self):
        with tempfile.TemporaryDirectory() as t:
            p, log, calls, e2 = run_driver(Path(t), STUB_F001_FAILS=1)
            self.assertEqual(p.returncode, 0, log + p.stderr); self.assertEqual([kind(c) for c in calls], TEACHER + REST); self.common(calls)
        with tempfile.TemporaryDirectory() as t:                                                   # frozen10k basis, core rows, a failing reuse check is descriptive
            p, log, calls, e2 = run_driver(Path(t), PCA_BASIS="frozen10k", ROWS="core", STUB_REUSE_PASS="false", CACHE_PARTITION="RTX6000PRO")
            self.assertEqual(p.returncode, 0, log); self.assertTrue(next(c for c in calls if kind(c) == "cache").endswith("cache 10000"))
            self.assertIn("--partition=RTX6000PRO --exclude", next(c for c in calls if kind(c) == "cache")); self.assertIn("descriptive only", log)
            self.assertEqual(len(PL.read_configs(e2 / "eval" / "run" / "configs_chartqa_val.txt")), 50)

    def test_stops(self):
        for kw, msg, none in (({"stage": "teacher", "keys": ()}, "teacher stage is not frozen", True), ({"stage": "teacher", "keys": ("r7_e2",)}, "teacher stage is not frozen", True),
                              ({"stage": "teacher", "probe": False}, "no passing probe report", True), ({"stage": "teacher", "replay": False}, "no replay file", True),
                              ({"stage": "rest", "keys": ("r7_e2a",), "teacher_done": True}, "is not frozen", True), ({"stage": ""}, "usage", True),
                              ({"stage": "all", "keys": ("r7_e2a",)}, "is not frozen", True), ({"ROWS": "all"}, "bad ROWS", True), ({"PCA_BASIS": "other"}, "bad PCA_BASIS", True),
                              ({"CACHE_PARTITION": "L40S;x"}, "bad CACHE_PARTITION", True), ({"STUB_F001_FAILS": 9}, "unit failed 3 times", False),
                              ({"STUB_FAIL_ON": "train logistic 0.0001"}, "no job id", False), ({"STUB_CACHE_PASS": "false"}, "does not say", False),
                              ({"STUB_CHECK_FAILS": 1}, "check.json does not say", False), ({"STUB_REUSE_PASS": "false"}, "failed the reuse check", False)):
            with tempfile.TemporaryDirectory() as t:
                p, log, calls, _ = run_driver(Path(t), **kw)
                self.assertEqual(p.returncode, 1, (kw, log)); self.assertIn(msg, log); self.assertNotIn("done:", log)
                if none:
                    self.assertEqual(calls, [], kw)
        with tempfile.TemporaryDirectory() as t:                                                   # a failing bundle: exactly 3 submissions, then STOP
            p, log, calls, _ = run_driver(Path(t), stage="teacher", keys=("r7_e2a",), STUB_F001_FAILS=9); self.assertEqual(sum("bundle f001" in c for c in calls), 3)


class ProtocolDraft(unittest.TestCase):
    def test_draft_parses(self):
        p = R / "reports" / "R7" / "R7_E2_QWEN_FULL_PROTOCOL_DRAFT_20261013.yaml"
        if not p.exists():
            self.skipTest("no draft")
        import yaml
        d = yaml.safe_load(p.read_text()); self.assertEqual(d["status"], "draft"); self.assertIn("feasibility", d); self.assertNotIn("TBD", p.read_text())
        self.assertIn("stage_E2a_teacher", d); self.assertEqual(d["stage_E2a_teacher"]["freeze_key"], "r7_e2a_staged_freeze")
        self.assertTrue({f"E2a-{i}" for i in range(1, 7)} <= set(d["open_decisions"]["teacher_stage_E2a"]))


if __name__ == "__main__":
    unittest.main()
