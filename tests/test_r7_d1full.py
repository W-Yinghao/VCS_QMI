"""R7-30 D1-FULL on CPU (synthetic data and the frozen fake host; no GQA / CAL / DEV read): the frozen D1 builder extended to every eligible image
(byte prefix, CAL / DEV identity, FIT superset; verification and its failure modes); linear-time shard plan == the frozen plan; bundles and lines;
strict image source (a missing image stops loudly through the frozen bundle loop); bundle wrapper end to end; reuse check (pass, tampering,
copy) and completeness; cache streaming (npy bytes == np.save, frozen encode arithmetic, ordered prefetch, resumable run); selector / evaluation
plan; train wrapper (frozen trainer as a subprocess on a tiny root; finish path); D1-FULL readout (6-contrast primary family, Holm, scale
contrasts, 10k family == the frozen D1-30 readout); driver dry run with stub sbatch / squeue."""
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts")); sys.path.insert(0, str(R / "tests")); sys.path.insert(0, str(R / "r7" / "package_r7_20261011" / "reference"))
import r7_d1_30_run as RUN  # noqa: E402
import r7_d1_30_score as SC  # noqa: E402
import r7_d1_30_sparsevlm as SV  # noqa: E402
import r7_d1_data as DA  # noqa: E402
import r7_d1_plan20 as P20  # noqa: E402
import r7_d1_teacher as TE  # noqa: E402
import r7_d1full_cache as FC  # noqa: E402
import r7_d1full_data as FD  # noqa: E402
import r7_d1full_plan as FP  # noqa: E402
import r7_d1full_score as FS  # noqa: E402
import r7_d1full_teacher as FT  # noqa: E402
import r7_d1full_train as FTR  # noqa: E402
from test_r7_d1 import FakeLlava  # noqa: E402   (frozen D1-10 CPU stand-in for the host)


def synth_rows(n_img=300, seed=3, p_yes=0.5, balanced=True):
    """Synthetic train_all-like rows.  balanced: 2 or >= 4 binary questions per image, the first four alternating yes / no, so every role's
    alternation picks are exactly balanced and the frozen band trim never fires (as on the real data: trimmed 0 in both builds).  balanced=False:
    random answers (small roles can then trigger the frozen trim)."""
    g = np.random.default_rng(seed); rows = []
    for i in range(n_img):
        k = int(g.choice([2, 4, 5, 6, 7])) if balanced else int(g.integers(1, 8)); first = int(g.integers(2))
        for j in range(k):
            ans = ("yes", "no")[(first + j) % 2] if (balanced and j < 4) else ("yes" if g.uniform() < p_yes else "no")
            rows.append({"qid": f"{i}_{j}", "image_id": f"img{i:04d}", "question": f"Is thing {j} on {i}?", "answer": ans,
                         "structural": "verify", "semantic": "attr", "detailed": "d"})
        if g.uniform() < 0.3:
            rows.append({"qid": f"{i}_o", "image_id": f"img{i:04d}", "question": "What colour?", "answer": "red", "structural": "query", "semantic": "attr", "detailed": "d"})
    for i in range(n_img, n_img + 7):                                                   # images without any binary question: never eligible
        rows.append({"qid": f"{i}_o", "image_id": f"img{i:04d}", "question": "What?", "answer": "left", "structural": "query", "semantic": "rel", "detailed": "d"})
    return rows


def write_manifest(d: Path, man, meta):
    d.mkdir(parents=True, exist_ok=True); (d / "d1_manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in man))
    meta = dict(meta, questions_sha256={"train_all_questions_0.json": "q"}, mix665k_sha256="m", manifest_sha256=DA.sha256_file(d / "d1_manifest.jsonl"))
    (d / "d1_manifest_meta.json").write_text(json.dumps(meta)); return meta


def frozen_and_full(t: Path, rows=None, n_fit=60, n_cal=12, n_dev=12, ips=5, require_no_trim=True):
    rows = rows or synth_rows(); fd, dd = t / "D1" / "data", t / "DF" / "data"
    man, meta = DA.build_manifest(iter(rows), n_fit, n_cal, n_dev, 4, None, balance=True); meta = write_manifest(fd, man, dict(meta, frozen=True))
    plan = DA.plan_shards(man, ips, 2); (fd / "d1_shards.json").write_text(json.dumps(plan))
    fman, fmeta = FD.build(iter(rows), meta, None); fmeta = write_manifest(dd, fman, fmeta)
    if require_no_trim:                                                                 # precondition, as on the real data (trimmed 0 in both builds)
        assert all(v["trimmed"] == 0 for m in (meta, fmeta) for v in m["balance"].values()), (meta["balance"], fmeta["balance"])
    return fd, dd, man, plan, fman, fmeta


class ManifestExtension(unittest.TestCase):
    def test_full_manifest_extends_frozen(self):
        with tempfile.TemporaryDirectory() as t:
            fd, dd, man, plan, fman, fmeta = frozen_and_full(Path(t))
            R_ = FD.verify(dd, fd, images_dir=None, expected_sha=None)
            self.assertTrue(R_["ok"], json.dumps(R_, indent=1)[:3000]); self.assertTrue(R_["frozen_manifest_is_byte_prefix"])
            for role in ("CAL", "DEV"):
                self.assertTrue(R_["roles"][role]["identical"]); self.assertEqual(R_["roles"][role]["images_sha256_frozen"], R_["roles"][role]["images_sha256_d1full"])
            self.assertTrue(R_["roles"]["FIT"]["superset"]); self.assertEqual(R_["roles"]["FIT"]["frozen_images"], 60)
            self.assertEqual(R_["structure"]["assigned_images"], fmeta["eligible_images"]); self.assertEqual(fmeta["eligible_images"], 300)
            self.assertEqual(R_["roles"]["FIT"]["questions"], R_["roles"]["FIT"]["frozen_questions"] + R_["roles"]["FIT"]["additional_questions"])
            self.assertEqual(fmeta["counts"]["FIT"]["images"], 300 - 24); self.assertLessEqual(R_["structure"]["max_q_per_image"], 4)
            self.assertTrue(all(r["answer"] in ("yes", "no") for r in fman))

    def test_frozen_trim_breaks_prefix_and_is_flagged(self):
        """If the frozen build had trimmed its FIT role (the real one did not: trimmed 0 in both builds), FIT_FULL would not extend it and verify
        must say so.  Deterministic construction: roles depend only on the sha order of eligible images, so the frozen FIT images get 3 'yes' + 1
        'no' each (pi_yes 0.75: the frozen role must trim), 30 additional FIT images 4 'no' each and the rest 2 / 2 (the full role is exactly 0.5)."""
        base = synth_rows(); man0, meta0 = DA.build_manifest(iter(base), 60, 12, 12, 4, None, balance=True)
        frozen_fit = {r["image_id"] for r in man0 if r["role"] == "FIT"}
        full0, _ = FD.build(iter(base), dict(meta0, manifest_sha256="x"), None); add_fit = sorted({r["image_id"] for r in full0 if r["role"] == "FIT"} - frozen_fit)
        rows = []
        for img in sorted({r["image_id"] for r in man0} | {r["image_id"] for r in full0}):
            answers = ["yes", "no", "yes", "yes"] if img in frozen_fit else (["no"] * 4 if img in add_fit[:30] else ["yes", "no", "yes", "no"])
            rows += [{"qid": f"{img}_{j}", "image_id": img, "question": f"q{j} {img}?", "answer": a, "structural": "verify", "semantic": "s", "detailed": "d"}
                     for j, a in enumerate(answers)]
        with tempfile.TemporaryDirectory() as t:
            fd, dd, man, plan, fman, fmeta = frozen_and_full(Path(t), rows=rows, require_no_trim=False)
            self.assertGreater(json.loads((fd / "d1_manifest_meta.json").read_text())["balance"]["FIT"]["trimmed"], 0); self.assertEqual(fmeta["balance"]["FIT"]["trimmed"], 0)
            R_ = FD.verify(dd, fd, images_dir=None, expected_sha=None)
            self.assertFalse(R_["ok"]); self.assertFalse(R_["frozen_manifest_is_byte_prefix"]); self.assertFalse(R_["roles"]["FIT"]["superset"])
            self.assertTrue(R_["roles"]["FIT"]["frozen_qids_subset"]); self.assertTrue(R_["roles"]["CAL"]["identical"] and R_["roles"]["DEV"]["identical"])

    def test_verify_detects_changes_and_missing_images(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); fd, dd, man, plan, fman, fmeta = frozen_and_full(t)
            imgs = t / "images"; imgs.mkdir(); ids = sorted({r["image_id"] for r in fman})
            for i in ids[1:]:
                (imgs / f"{i}.jpg").write_bytes(b"x")
            R_ = FD.verify(dd, fd, images_dir=imgs, expected_sha=None); self.assertFalse(R_["ok"]); self.assertEqual(sum(R_["images"]["missing"].values()), 1)
            cal = next(i for i, r in enumerate(fman) if r["role"] == "CAL"); bad = [dict(r) for r in fman]; bad[cal]["question"] += " changed"
            write_manifest(dd, bad, fmeta); R2 = FD.verify(dd, fd, images_dir=None, expected_sha=None)
            self.assertFalse(R2["ok"]); self.assertFalse(R2["roles"]["CAL"]["identical"]); self.assertFalse(R2["frozen_manifest_is_byte_prefix"])
            fit = [i for i, r in enumerate(fman) if r["role"] == "FIT"]; bad = [r for i, r in enumerate(fman) if i != fit[3]]   # drop a frozen FIT question
            write_manifest(dd, bad, fmeta); R3 = FD.verify(dd, fd, images_dir=None, expected_sha=None); self.assertFalse(R3["roles"]["FIT"]["superset"])


class ShardPlan(unittest.TestCase):
    def test_fast_plan_equals_frozen_plan(self):
        with tempfile.TemporaryDirectory() as t:
            _, _, man, plan, fman, _ = frozen_and_full(Path(t))
            for m in (man, fman):
                ref = DA.plan_shards(m, 5, 4); sh, q = FD.shard_plan(m, 5)
                self.assertEqual(sh, ref["shards"]); self.assertEqual(q, ref["questions_per_shard"]); self.assertEqual(list(sh), list(ref["shards"]))

    def test_bundles_lines_and_reuse_switch(self):
        with tempfile.TemporaryDirectory() as t:
            _, _, man, plan, fman, _ = frozen_and_full(Path(t))
            p = FD.make_plan(fman, plan, True, 5, 3); frozen_fit = [s for s in plan["shards"] if s.startswith("FIT-")]
            self.assertFalse(any(s.startswith("DEV-") for s in p["shards"])); allb = [s for v in p["bundles"].values() for s in v]
            self.assertEqual(sorted(allb), sorted(s for s in p["shards"] if s.startswith("FIT-") and s not in frozen_fit))
            self.assertTrue(all(len(v) <= 3 for v in p["bundles"].values())); self.assertEqual(set(p["d1full"]["reused_shards"]), set(frozen_fit) | {s for s in plan["shards"] if s.startswith("CAL-")})
            p2 = FD.make_plan(fman, plan, False, 5, 3); self.assertTrue(set(frozen_fit) <= {s for v in p2["bundles"].values() for s in v})
            self.assertFalse(any(s.startswith("CAL-") for v in p2["bundles"].values() for s in v))
            bad = json.loads(json.dumps(plan)); s0 = frozen_fit[0]; bad["shards"][s0] = bad["shards"][s0][::-1]
            with self.assertRaises(SystemExit):
                FD.make_plan(fman, bad, True, 5, 3)
            for line, mark in FD.teacher_lines(p, Path("/x/teacher")):
                self.assertIn("--partition=H100,L40S", line); self.assertIn("--exclude=node51,node52,node60", line); self.assertNotIn("--export", line)
                self.assertNotIn("RTX6000PRO", line); self.assertTrue(str(mark).endswith(f"bundle_{line.split()[-1]}.ok.json")); self.assertIn("--time=04:00:00", line)


class ProtocolDraft(unittest.TestCase):
    def test_draft_parses(self):
        p = R / "reports" / "R7" / "R7_D1_FULL_PROTOCOL_DRAFT_20261012.yaml"
        if not p.exists():
            self.skipTest("no draft")
        import yaml
        d = yaml.safe_load(p.read_text()); self.assertEqual(d["status"], "draft"); self.assertIn("open_decisions", d)
        self.assertNotIn("TBD", p.read_text())


class FakeSource:
    """Strict-source stand-in: every listed image exists (the frozen fake host ignores pixels)."""
    folder = "fake"

    def __init__(self, missing=()):
        self.missing = set(missing)

    def exists(self, i):
        return i not in self.missing

    def open(self, i):
        if i in self.missing:
            raise FT.MissingImage(i)
        return None


def probe_ok(t: Path):
    p = t / "probe_report.json"; p.write_text(json.dumps({"pass": True})); return str(p)


def frozen_teacher(t: Path, fd: Path, plan):
    """The frozen D1-10 teacher (fake host) on the frozen CAL / FIT shards."""
    td = t / "D1" / "teacher"; pr = probe_ok(t)
    for b, shards in plan["bundles"].items():
        if any(s.startswith(("CAL-", "FIT-")) for s in shards):
            TE.cmd_bundle(Namespace(bundle=b, manifest_dir=str(fd), out=str(td), probe_report=pr, **FT.SETTINGS), host_factory=lambda: FakeLlava(), image_source=FakeSource())
    return td


class StrictTeacher(unittest.TestCase):
    def test_missing_image_is_loud(self):
        self.assertFalse(issubclass(FT.MissingImage, Exception))
        with tempfile.TemporaryDirectory() as t:
            src = FT.StrictImageSource(t); self.assertFalse(src.exists("nope"))
            with self.assertRaises(FT.MissingImage):
                src.open("nope")
            t = Path(t); fd, dd, man, plan, fman, fmeta = frozen_and_full(t); DA.cmd_freeze(Namespace(out=str(dd), protocol="t", go="t"))
            p = FD.make_plan(fman, plan, True, 5, 3); (dd / "d1_shards.json").write_text(json.dumps(p)); b = next(iter(p["bundles"]))
            gone = p["shards"][p["bundles"][b][0]][1]
            with self.assertRaises(FT.MissingImage):                                       # the frozen loop's 'except Exception' does not swallow it
                TE.cmd_bundle(Namespace(bundle=b, manifest_dir=str(dd), out=str(t / "tA"), probe_report=probe_ok(t), **FT.SETTINGS),
                              host_factory=lambda: FakeLlava(), image_source=FakeSource([gone]))
            called = []
            with self.assertRaises(SystemExit):                                            # preflight stops before the host loads
                FT.cmd_bundle(Namespace(bundle=b, manifest_dir=str(dd), out=str(t / "tB"), probe_report=probe_ok(t)),
                              host_factory=lambda: called.append(1) or FakeLlava(), image_source=FakeSource([gone]))
            self.assertEqual(called, [])

    def test_bundle_reuse_complete_end_to_end(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); fd, dd, man, plan, fman, fmeta = frozen_and_full(t); ftd = frozen_teacher(t, fd, plan)
            DA.cmd_freeze(Namespace(out=str(dd), protocol="R7-30-D1-FULL-test", go="unit test"))
            res = FD.reuse_check(dd, fd, ftd, replay=None)
            self.assertTrue(res["fit"]["static_ok"], json.dumps({k: res[k] for k in ("code", "env", "settings_check")}, indent=1)[:3000])
            self.assertTrue(res["cal"]["ok"]); self.assertFalse(res["fit_reuse"])                    # no GPU replay -> recompute
            rp = t / "replay.json"; rp.write_text(json.dumps({"pass": True})); res = FD.reuse_check(dd, fd, ftd, replay=rp); self.assertTrue(res["fit_reuse"])
            td = t / "DF" / "teacher"; copied = FD.copy_verified(res, ftd, td); self.assertEqual(len(copied), 2 * len(res["shards"]))
            self.assertEqual(FD.copy_verified(res, ftd, td), [])                                       # identical files are kept
            p = FD.make_plan(fman, plan, True, 5, 3); (dd / "d1_shards.json").write_text(json.dumps(p))
            self.assertFalse(FD.completeness(dd, td)["complete"])
            for b in p["bundles"]:
                FT.cmd_bundle(Namespace(bundle=b, manifest_dir=str(dd), out=str(td), probe_report=probe_ok(t)), host_factory=lambda: FakeLlava(), image_source=FakeSource())
                self.assertTrue((td / f"bundle_{b}.ok.json").exists())
            c = FD.completeness(dd, td); self.assertTrue(c["complete"], c["problems"]); self.assertEqual(c["present_per_role"]["FIT"], fmeta["counts"]["FIT"]["questions"])
            p["bundles"]["fx"] = ["FIT-000"]; (dd / "d1_shards.json").write_text(json.dumps(p))
            with self.assertRaises(SystemExit):                                                          # a reused shard is never recomputed
                FT.cmd_bundle(Namespace(bundle="fx", manifest_dir=str(dd), out=str(td), probe_report=probe_ok(t)), host_factory=lambda: FakeLlava(), image_source=FakeSource())
            f0 = sorted(td.glob("FIT-000.jsonl"))[0]; f0.write_text(f0.read_text() + f0.read_text().splitlines()[0] + "\n")
            with self.assertRaises(SystemExit):
                FD.copy_verified(res, ftd, td)                                                           # a different existing file is never overwritten

    def test_reuse_check_detects_tampering(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); fd, dd, man, plan, fman, fmeta = frozen_and_full(t); ftd = frozen_teacher(t, fd, plan)
            rows = [json.loads(x) for x in (ftd / "FIT-000.jsonl").read_text().splitlines()]; rows[0]["order_digest"]["random"] = "0" * 16
            (ftd / "FIT-000.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            res = FD.reuse_check(dd, fd, ftd, replay=None); e = res["shards"]["FIT-000"]
            self.assertFalse(e["ok"]); self.assertTrue(any("sha256" in p for p in e["problems"])); self.assertTrue(any("random" in p for p in e["problems"]))
            self.assertFalse(res["fit"]["static_ok"]); self.assertTrue(res["cal"]["ok"])
            (ftd / "CAL-000.jsonl").write_text((ftd / "CAL-000.jsonl").read_text().splitlines()[0] + "\n")
            self.assertFalse(FD.reuse_check(dd, fd, ftd, replay=None)["cal"]["ok"])


class FakeEnc:
    """CPU host for the cache code: deterministic 'visual tokens' [n, 576, 16] from the image id; tokenizer / embedding for the questions."""

    def __init__(self):
        self.torch = torch; self.embed = torch.nn.Embedding(50, 4096)

    def encode(self, pils):
        return torch.stack([torch.sin(torch.arange(576 * 16, dtype=torch.float32).reshape(576, 16) * (1 + (hash(p) % 97) / 97.0)) for p in pils]).half()

    def tok(self, q, add_special_tokens=False):
        return Namespace(input_ids=[1 + (ord(c) % 40) for c in q[:7]])


class Cache(unittest.TestCase):
    def test_npystream_equals_np_save(self):
        with tempfile.TemporaryDirectory() as t:
            a = np.random.default_rng(0).normal(size=(37, 3, 4)).astype(np.float16); p, q = Path(t) / "a.npy", Path(t) / "b.npy"; np.save(q, a)
            w = FC.NpyStream(p, a.shape)
            for c in range(0, 37, 16):
                w.write(a[c:c + 16])
            w.close(); self.assertEqual(p.read_bytes(), q.read_bytes()); self.assertFalse(p.with_name("a.npy.tmp").exists())
            w = FC.NpyStream(Path(t) / "c.npy", a.shape); w.write(a[:5])
            with self.assertRaises(ValueError):
                w.close()
            with self.assertRaises(ValueError):
                w.write(np.zeros((40, 3, 4)))

    def test_prefetch_order_and_missing(self):
        def opener(i):
            time.sleep(0.001 * ((i * 7) % 5))
            if i == 77:
                raise FT.MissingImage("77")
            return i * 2
        self.assertEqual([v for _, v in FC.prefetch(range(50), opener, workers=4, ahead=8)], [2 * i for i in range(50)])
        with self.assertRaises(FT.MissingImage):
            list(FC.prefetch(range(100), opener, workers=4, ahead=8))

    def test_run_frozen_basis_streaming_and_resume(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); rows = []
            for i in range(21):
                rows.append({"qid": f"q{i}", "image_id": f"i{i}", "question": f"is it {i}?", "role": "CAL" if i % 5 == 0 else "FIT", "C": 1})
            md = t / "data"; md.mkdir(); (md / "d1_manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            (md / "d1_manifest_meta.json").write_text(json.dumps({"frozen": True, "manifest_sha256": DA.sha256_file(md / "d1_manifest.jsonl")}))
            g = torch.Generator().manual_seed(0); pca = {"mean": torch.randn(16, generator=g), "basis": torch.randn(16, 8, generator=g), "explained_variance": 0.5, "tokens": 1}
            torch.save(pca, t / "frozen_pca.pt"); h = FakeEnc(); src = Namespace(open=lambda i: i)
            rec = FC.run(md, t / "cache", "frozen", t / "frozen_pca.pt", host=h, src=src, device="cpu")
            self.assertTrue(rec["pca_equals_frozen"]); fit = [r["image_id"] for r in rows if r["role"] == "FIT"]
            ref = np.concatenate([((h.encode(fit[c:c + 16]).float() - pca["mean"]) @ pca["basis"]).half().numpy() for c in range(0, len(fit), 16)])
            self.assertTrue(np.array_equal(np.load(t / "cache" / "feats_FIT.npy"), ref)); self.assertEqual(json.loads((t / "cache" / "images_FIT.json").read_text()), fit)
            q = [json.loads(x) for x in (t / "cache" / "question_ids.jsonl").read_text().splitlines()]; self.assertEqual([x["qid"] for x in q], [r["qid"] for r in rows])
            self.assertEqual(tuple(torch.load(t / "cache" / "embed_table.pt").shape), (50, 4096))
            m0 = (t / "cache" / "feats_FIT.npy").stat().st_mtime_ns; (t / "cache" / "cache_d1_full.json").unlink(); (t / "cache" / "feats_CAL.npy").unlink()
            FC.run(md, t / "cache", "frozen", t / "frozen_pca.pt", host=h, src=src, device="cpu")
            self.assertEqual((t / "cache" / "feats_FIT.npy").stat().st_mtime_ns, m0); self.assertTrue((t / "cache" / "feats_CAL.npy").exists())


class Plans(unittest.TestCase):
    def test_selector_lines(self):
        L1 = FP.stage1(Path("/x/sel")); self.assertEqual(len(L1), 6)
        ch = {"vcs": {"chosen_lr": 3e-4}, "logistic": {"chosen_lr": 1e-4}, "direct": {"chosen_lr": 1e-4}}; L2 = FP.stage2(ch, Path("/x/sel")); self.assertEqual(len(L2), 12)
        for line, mark in L1 + L2:
            self.assertIn("--partition=RTX6000PRO,H100 ", line); self.assertNotIn("L40S", line); self.assertIn("--exclude=node51,node52,node60", line)
            self.assertNotIn("--export", line); self.assertNotIn("--dependency", line); self.assertTrue(str(mark).endswith(".complete.json"))
            self.assertIn("--time=08:00:00", line)
        tags = lambda L: sorted(l.split("--job-name=")[1].split()[0].split("_", 1)[1] for l in L)  # noqa: E731
        self.assertEqual(tags([l for l, _ in L2]), tags(P20.stage2(ch))); self.assertEqual(tags([l for l, _ in L1]), tags(P20.stage1()))
        self.assertEqual(len({str(m) for _, m in L1 + L2}), 18)

    def test_eval_configs_and_lines(self):
        ch = {"vcs": {"chosen_lr": 3e-4}, "logistic": {"chosen_lr": 1e-4}, "direct": {"chosen_lr": 1e-4}}
        with tempfile.TemporaryDirectory() as t:
            res = FP.write_eval_configs(ch, Path(t)); self.assertEqual(res, {"configs_per_bench": 87, "timing_configs_per_bench": 27})
            cf = (Path(t) / "configs_textvqa_val.txt").read_text().split(); self.assertFalse(any(c.startswith(("full", "static:")) for c in cf))
            self.assertEqual(len([c for c in cf if c.startswith("sel:")]), 60); self.assertEqual(sorted(cf), sorted(c for c in RUN.config_list(
                [u for o in RUN.OBJECTIVES for u in RUN.unit_names(ch)[0][o]], RUN.unit_names(ch)[1]["A1"] + RUN.unit_names(ch)[1]["A2"], True) if c.startswith(("sel:", "dyn:"))))
        L = FP.eval_lines(Path("/x/d1_30")); self.assertEqual(len(L), 14); Ln = FP.eval_lines(Path("/x/d1_30"), devrisk=False, timing=False); self.assertEqual(len(Ln), 12)
        Ld = FP.eval_lines(Path("/x/d1_30"), devrisk_only=True); self.assertEqual(len(Ld), 1); self.assertIn("devrisk", Ld[0][0])
        self.assertTrue(str(Ld[0][1]).endswith("/x/d1_30/devrisk/d1_30_devrisk.json"))
        for line, _ in L:
            self.assertIn("--exclude=node51,node52,node60", line); self.assertNotIn("RTX6000PRO", line); self.assertNotIn("--export", line); self.assertIn("--time=", line)
        tl = [(l, m) for l, m in L if " timing_all" in l]; self.assertEqual(len(tl), 1)   # one bundled nodesumo01 job (owner rule 2026-10-13)
        self.assertTrue(all("--partition=H100 --nodelist=nodesumo01" in l for l, _ in tl)); self.assertTrue(str(tl[0][1]).endswith("timing/timing_bundle.done.json"))
        self.assertTrue(all("H100,L40S" in l for l, _ in L if " eval " in l)); self.assertTrue(all("/x/d1_30/pred/" in str(m) for l, m in L if " eval " in l))


def tiny_root(t: Path, n_fit=5, n_cal=3):
    """A D1-FULL-shaped root on CPU: manifest / plan / teacher records (frozen fake host) / cache (random PCA features)."""
    rows = synth_rows(40, seed=5, p_yes=0.5); man, meta = DA.build_manifest(iter(rows), n_fit, n_cal, 1, 4, None, balance=True)
    root = t / "root"; meta = write_manifest(root / "data", man, dict(meta, frozen=True)); plan = DA.plan_shards(man, 250, 4)
    (root / "data" / "d1_shards.json").write_text(json.dumps(plan)); pr = probe_ok(t)
    for b in plan["bundles"]:
        TE.cmd_bundle(Namespace(bundle=b, manifest_dir=str(root / "data"), out=str(root / "teacher"), probe_report=pr, **FT.SETTINGS),
                      host_factory=lambda: FakeLlava(), image_source=FakeSource())
    c = root / "cache"; c.mkdir(); g = np.random.default_rng(0)
    for role in ("FIT", "CAL"):
        ids = list(dict.fromkeys(r["image_id"] for r in man if r["role"] == role)); np.save(c / f"feats_{role}.npy", g.normal(size=(len(ids), 576, 512)).astype(np.float16))
        (c / f"images_{role}.json").write_text(json.dumps(ids))
    (c / "question_ids.jsonl").write_text("".join(json.dumps({"qid": r["qid"], "ids": [1 + i % 30 for i in range(3 + len(r["qid"]) % 4)]}) + "\n" for r in man if r["role"] != "DEV"))
    torch.save((torch.randn(64, 4096) * 0.02).half(), c / "embed_table.pt")
    return root


class TrainWrapper(unittest.TestCase):
    def test_frozen_trainer_marker_and_finish(self):
        with tempfile.TemporaryDirectory() as t:
            root = tiny_root(Path(t)); a = Namespace(objective="direct", lr=1e-4, seed=0, no_question=False, target="h", root=str(root))
            rc = FTR.cmd_train(a, ["--steps-a", "2", "--steps-b", "1", "--eval-every", "1"]); self.assertEqual(rc, 0)
            sel = root / "selectors"; tag = "direct_lr0.0001_s0"; mk = json.loads((sel / f"{tag}.complete.json").read_text())
            log = json.loads((sel / f"{tag}.json").read_text()); self.assertTrue(FTR.has_taus(log)); self.assertIsNone(mk["finished_by"])
            self.assertEqual(log["cfg"]["steps_a"], 2); self.assertEqual(log["coverage"]["FIT"]["present"], log["coverage"]["FIT"]["expected"])
            taus0 = {f: log["dynamic"][f]["tau"] for f in log["dynamic"]}
            (sel / f"{tag}.json").write_text(json.dumps({k: v for k, v in log.items() if k != "dynamic"})); (sel / f"{tag}.complete.json").unlink()
            self.assertEqual(FTR.cmd_train(a, []), 0); new = json.loads((sel / f"{tag}.json").read_text())
            self.assertTrue(new["finished_by"].startswith("r7_d1full_train finish")); self.assertTrue((sel / f"{tag}.complete.json").exists())
            for f, v in taus0.items():
                self.assertAlmostEqual(new["dynamic"][f]["tau"], v, places=6)
            self.assertEqual(FTR.cmd_train(a, []), 0)                                                    # marker present: nothing to do


class Readout(unittest.TestCase):
    def setup(self, t: Path, drop_full=None, with_full_timing=True):
        bench = "gqa_dev"; items = [{"bench": bench, "qid": f"q{i}", "image_id": f"img{i // 2}", "question": "?", "answer": "yes", "prompt": "?"} for i in range(40)]
        d30 = t / "d30"; (d30 / "sets").mkdir(parents=True); (d30 / "sets" / f"{bench}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in items))
        g = np.random.default_rng(0); acc = {72: 0.55, 144: 0.65, 288: 0.75, 432: 0.85, 576: 0.95}
        ch10 = {"vcs": {"chosen_lr": 3e-4}, "logistic": {"chosen_lr": 1e-4}, "direct": {"chosen_lr": 3e-4}}; chF = {"vcs": {"chosen_lr": 1e-4}, "logistic": {"chosen_lr": 1e-4}, "direct": {"chosen_lr": 3e-4}}
        (t / "lr10.json").write_text(json.dumps(ch10)); (t / "lrF.json").write_text(json.dumps(chF))
        def write(pd, cfg, a, kept=576):
            pd.mkdir(parents=True, exist_ok=True); rows = [{"qid": it["qid"], "image_id": it["image_id"], "pred": "yes" if g.uniform() < a else "no", "kept": kept} for it in items]
            (pd / f"{RUN.cfg_file(cfg)}@s0of1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        m10, a10 = RUN.unit_names(ch10); cfgs = RUN.config_list([u for o in RUN.OBJECTIVES for u in m10[o]], a10["A1"] + a10["A2"])
        sv = [SV.row_tag(T, o) for T, o in SV.rows_for(bench, list(SV.ANCHORS + SV.ADAPTED))]
        for c in cfgs + sv:
            if c == "full":
                write(d30 / "pred" / bench, c, acc[576])
            elif c.startswith(("static:", "sel:")):
                k = int(c.split(":")[2]); write(d30 / "pred" / bench, c, acc[k], k)
            elif c.startswith("dyn:"):
                write(d30 / "pred" / bench, c, 0.8, {"0.25": 144, "0.5": 288, "0.75": 432}[c.split(":")[2]])
            else:
                write(d30 / "pred" / bench, c, 0.7, int(c.split(":")[1]))
        mF, aF = RUN.unit_names(chF)
        for c in [c for c in RUN.config_list([u for o in RUN.OBJECTIVES for u in mF[o]], aF["A1"] + aF["A2"]) if c.startswith(("sel:", "dyn:"))]:
            if c == drop_full:
                continue
            if c.startswith("sel:"):
                k = int(c.split(":")[2]); write(t / "df" / "pred" / bench, c, 1.0 if c.startswith("sel:vcs") else acc[k], k)
            else:
                write(t / "df" / "pred" / bench, c, 0.82, {"0.25": 144, "0.5": 288, "0.75": 432}[c.split(":")[2]])
        lat = {72: 40.0, 144: 55.0, 288: 70.0, 432: 85.0}; tm = []
        for B in (1, 4):
            tm.append({"config": "full", "B": B, "per_item_ms_median": 100.0})
            for c in cfgs:
                if c.startswith("static:") or c.startswith(f"sel:{m10['vcs'][0]}:"):
                    tm.append({"config": c, "B": B, "per_item_ms_median": lat[int(c.split(":")[2])] + (2.0 if c.startswith("sel:") else 0.0)})
                elif c.startswith("dyn:"):
                    tm.append({"config": c, "B": B, "per_item_ms_median": {"0.25": 60.0, "0.5": 75.0, "0.75": 90.0}[c.split(":")[2]]})
        (d30 / "timing" / bench).mkdir(parents=True); (d30 / "timing" / bench / f"timing_{bench}.json").write_text(json.dumps({"rows": tm}))
        (d30 / "timing" / f"sparsevlm_timing_{bench}.json").write_text(json.dumps({"rows": [{"config": c, "B": 1, "per_item_ms_median": 45.0 + int(c.split(":")[1]) / 10} for c in sv]}))
        if with_full_timing:
            ft = [{"config": c, "B": B, "per_item_ms_median": {"0.25": 58.0, "0.5": 73.0, "0.75": 88.0}[c.split(":")[2]]} for B in (1, 4)
                  for c in RUN.config_list([u for o in RUN.OBJECTIVES for u in mF[o]], []) if c.startswith("dyn:")]
            (t / "df" / "timing" / bench).mkdir(parents=True); (t / "df" / "timing" / bench / f"timing_{bench}.json").write_text(json.dumps({"rows": ft}))
        proto = t / "proto.yaml"; proto.write_text("d1_30_staged_freeze: x\nd1_full_staged_freeze: y\n")
        return ["--d30-root", str(d30), "--d30-lr-choice", str(t / "lr10.json"), "--df-root", str(t / "df"), "--df-lr-choice", str(t / "lrF.json"),
                "--out", str(t / "res"), "--benches", bench, "--reps", "200", "--protocol", str(proto)]

    def test_primary_family_scale_and_k10_equals_d1_30(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); args = self.setup(t); FS.main(args); r = json.loads((t / "res" / "d1full_results.json").read_text())["benches"]["gqa_dev"]
            C = r["full_scale"]["contrasts"]
            for L in ("L25", "L50", "L75"):
                prim = [k for k, v in C.items() if f"@{L}" in k and v.get("role") == "primary"]
                self.assertEqual(sorted(prim), sorted([f"S1-S2@{L}", f"S1-S3@{L}", f"S1-SparseVLM@{L}[budget]"]))   # the frozen D1-30 family
                self.assertTrue(all(C[k]["holm_family_size"] == 3 for k in prim))                                  # 3 per benchmark (1 benchmark here)
                self.assertEqual(C[f"S1-SparseVLM@{L}[budget]"]["role"], "primary")
                for o in ("S1", "S2", "S3"):
                    s = r["scale_contrasts"][f"{o}:full-10k@{L}"]; self.assertEqual(s["role"], "descriptive"); self.assertIn("ci95", s)
                self.assertGreater(r["scale_contrasts"][f"S1:full-10k@{L}"]["delta"], 0)
            self.assertIn("A1:full-10k@k144", r["scale_contrasts"]); self.assertIn("S1_dyn@f0.5:full", r["dynamic_rows_by_scale"])
            self.assertEqual(r["full_scale_dynamic_rows_timed"], 27 * 2); self.assertIsNotNone(r["full_scale"]["at_latency"]["4"]["S1_dyn"]["L50"])
            out30 = t / "res30"
            SC.main(["--sets-dir", str(t / "d30" / "sets"), "--pred-dir", str(t / "d30" / "pred"), "--timing-dir", str(t / "d30" / "timing"), "--lr-choice", str(t / "lr10.json"),
                     "--out", str(out30), "--benches", "gqa_dev", "--reps", "200"])
            ref = {k: v for k, v in json.loads((out30 / "d1_30_results.json").read_text())["benches"]["gqa_dev"]["contrasts"].items() if v.get("role") == "primary"}
            self.assertEqual(set(ref), set(r["k10_primary_family"]))
            for k, v in ref.items():
                for f in ("delta", "ci95", "p", "p_holm", "holm_family_size"):
                    self.assertEqual(v[f], r["k10_primary_family"][k][f], (k, f))

    def test_guards(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); args = self.setup(t, drop_full="sel:vcs_lr0.0001_s1:288")
            with self.assertRaises(SystemExit):
                FS.main(args)
            (t / "proto.yaml").write_text("d1_30_staged_freeze: x\n")
            with self.assertRaises(SystemExit):
                FS.main(args)
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); args = self.setup(t, with_full_timing=False)
            with self.assertRaises(SystemExit):                                            # the D1-FULL dynamic rows need their own latency
                FS.main(args)
            FS.main(args + ["--allow-missing-dyn-timing"]); r = json.loads((t / "res" / "d1full_results.json").read_text())["benches"]["gqa_dev"]
            self.assertEqual(r["full_scale_dynamic_rows_timed"], 0); self.assertIsNone(r["full_scale"]["at_latency"]["4"]["S1_dyn"]["L50"])


STUB_SBATCH = r'''#!/usr/bin/env bash
echo "$*" >> "$STUB_DIR/sbatch.log"
n=$(cat "$STUB_DIR/counter" 2>/dev/null || echo 2000); n=$((n + 1)); echo $n > "$STUB_DIR/counter"
args="$*"
if [ -n "$STUB_FAIL_ON" ] && [[ "$args" == *"$STUB_FAIL_ON"* ]]; then exit 1; fi
mk() { mkdir -p "$(dirname "$1")"; echo '{"stub": true}' > "$1"; }
case "$args" in
  *"r7_d1full_cpu.sbatch data"*) mk "$DF_ROOT/data/data_ready.json"; mkdir -p "$LINES_DIR"
    { echo "# teacher"; for b in f000 f001; do echo "sbatch --parsable --job-name=r7dFt_$b --partition=H100,L40S --exclude=node51,node52,node60 slurm/r7_d1full_gpu.sbatch bundle $b #MARK $DF_ROOT/teacher/bundle_$b.ok.json"; done; } > "$LINES_DIR/teacher_lines.txt";;
  *"bundle f001"*) c=$(cat "$STUB_DIR/f001" 2>/dev/null || echo 0); echo $((c + 1)) > "$STUB_DIR/f001"; [ "$c" -ge "${STUB_F001_FAILS:-0}" ] && mk "$DF_ROOT/teacher/bundle_f001.ok.json";;
  *"bundle f000"*) mk "$DF_ROOT/teacher/bundle_f000.ok.json";;
  *"r7_d1full_cpu.sbatch complete"*) mkdir -p "$DF_ROOT/teacher"; echo '{"complete": true}' > "$DF_ROOT/teacher/d1full_completeness.json";;
  *"r7_d1full_gpu.sbatch cache"*) mkdir -p "$DF_ROOT/cache"; b=${args##*--pca-basis }; echo "{\"basis\": \"${b%% *}\"}" > "$DF_ROOT/cache/cache_d1_full.json";;
  *"r7_d1full_sel.sbatch train"*)
    if [[ "$args" =~ train\ ([a-z]+)\ ([0-9.e-]+)\ ([0-9]+) ]]; then
      t="${BASH_REMATCH[1]}_lr${BASH_REMATCH[2]}_s${BASH_REMATCH[3]}"; [[ "$args" == *"--no-question"* ]] && t="${t}_noq"; [[ "$args" == *"--target d"* ]] && t="${t}_dtarget"
      mkdir -p "$DF_ROOT/selectors"; touch "$DF_ROOT/selectors/$t.pt"; mk "$DF_ROOT/selectors/$t.complete.json"
      v=$(python3 -c "import zlib; print(zlib.crc32(b'$t') % 1000 / 1000)"); echo "{\"selected_cal_spearman\": $v}" > "$DF_ROOT/selectors/$t.json"; fi;;
  *"r7_d1full_cpu.sbatch choose"*) "$PY" scripts/r7_d1full_plan.py choose --selectors-dir "$DF_ROOT/selectors" >> "$STUB_DIR/wrap.log" 2>&1 || exit 1;;
  *"r7_d1full_gpu.sbatch eval"*) set -- $args; i=${@: -1}; b=${@: -2:1}; mk "$DF_ROOT/d1_30/pred/$b/done_shard${i%%/*}of${i##*/}.json";;
  *"r7_d1full_gpu.sbatch devrisk"*) mk "$DF_ROOT/d1_30/devrisk/d1_30_devrisk.json";;
  *"r7_d1full_gpu.sbatch timing_all"*) for b in gqa_dev textvqa_val chartqa_val; do mk "$DF_ROOT/d1_30/timing/$b/timing_$b.json"; done; mk "$DF_ROOT/d1_30/timing/timing_bundle.done.json";;
  *"r7_d1full_gpu.sbatch timing"*) set -- $args; b=${@: -1}; mk "$DF_ROOT/d1_30/timing/$b/timing_$b.json";;
esac
echo $n
'''
STUB_SQUEUE = r'''#!/usr/bin/env bash
if [[ "$*" == *"-j"* ]]; then exit 0; fi
c=$(cat "$STUB_DIR/qcalls" 2>/dev/null || echo 0); c=$((c + 1)); echo $c > "$STUB_DIR/qcalls"
n=${STUB_QUEUE_N:-0}; [ "$c" -le "${STUB_FULL_CALLS:-0}" ] && n=27
for i in $(seq 1 $n); do echo "job$i"; done
'''


def run_driver(tmp: Path, frozen=True, d30_devrisk_done=True, **env_extra):
    stub = tmp / "stub"; stub.mkdir()
    for name, body in (("sbatch", STUB_SBATCH), ("squeue", STUB_SQUEUE)):
        f = stub / name; f.write_text(body); f.chmod(f.stat().st_mode | stat.S_IEXEC)
    df, d1 = tmp / "DF", tmp / "D1"; (df / "smoke" / "probe").mkdir(parents=True); (df / "smoke" / "probe" / "probe_report.json").write_text('{"pass": true}')
    (d1 / "d1_30" / "sets").mkdir(parents=True); (d1 / "d1_30" / "sets" / "d1_30_sets.json").write_text("{}")
    if d30_devrisk_done:
        (d1 / "d1_30" / "devrisk").mkdir(parents=True); (d1 / "d1_30" / "devrisk" / "d1_30_devrisk.json").write_text("{}")
    proto = tmp / "proto.yaml"; proto.write_text("d1_30_staged_freeze: x\n" + ("d1_full_staged_freeze: y\n" if frozen else ""))
    env = dict(os.environ, PATH=f"{stub}:{os.environ['PATH']}", STUB_DIR=str(stub), DF_ROOT=str(df), D1_ROOT=str(d1), PROTOCOL=str(proto), SUBMIT_LOG=str(tmp / "submit.log"),
               LINES_DIR=str(tmp / "lines"), POLL="0", WAIT_H="1", PY=sys.executable, CAP="27", RETRIES="2", **{k: str(v) for k, v in env_extra.items()})
    p = subprocess.run(["bash", str(R / "slurm" / "r7_d1full_submit.sh")], env=env, capture_output=True, text=True, timeout=600)
    calls = (stub / "sbatch.log").read_text().splitlines() if (stub / "sbatch.log").exists() else []
    return p, (tmp / "submit.log").read_text() if (tmp / "submit.log").exists() else "", calls, df


class DriverDryRun(unittest.TestCase):
    def test_full_sequence_with_retry_and_cap_wait(self):
        with tempfile.TemporaryDirectory() as t:
            p, log, calls, df = run_driver(Path(t), STUB_F001_FAILS=1, STUB_FULL_CALLS=3)
            self.assertEqual(p.returncode, 0, log + p.stderr); self.assertIn("RETRY 1/2", log); self.assertIn("done: D1-FULL evaluation complete", log)
            kinds = [("cpu.sbatch data" in c and "data") or ("bundle" in c and "bundle") or ("cpu.sbatch complete" in c and "complete") or ("sbatch cache" in c and "cache")
                     or (" train " in c and "train") or ("choose" in c and "choose") or (" eval " in c and "eval") or ("devrisk" in c and "devrisk")
                     or ("sbatch timing" in c and "timing") or "?" for c in calls]
            self.assertEqual(kinds, ["data"] + ["bundle"] * 3 + ["complete", "cache"] + ["train"] * 6 + ["choose"] + ["train"] * 12 + ["eval"] * 12
                             + ["timing"] * 1 + ["devrisk"], kinds)   # one bundled nodesumo01 timing job (owner rule 2026-10-13)
            self.assertTrue(all("--partition=H100 --nodelist=nodesumo01 --exclude=node51,node52,node60" in c for c in calls if "sbatch timing" in c))
            self.assertFalse(any("--dependency" in c or "--export" in c for c in calls))
            self.assertTrue(all("--exclude=node51,node52,node60" in c for c in calls if "--partition=CPU" not in c))
            self.assertTrue(all("--partition=RTX6000PRO,H100 " in c and "L40S" not in c for c in calls if " train " in c))
            self.assertTrue(all("--partition=H100,L40S" in c and "RTX6000PRO" not in c for c in calls if "bundle" in c or "sbatch cache" in c or " eval " in c or "devrisk" in c))
            self.assertTrue(all("--wait" in c for c in calls if "--partition=CPU" in c)); self.assertIn("--pca-basis fitfull", next(c for c in calls if "sbatch cache" in c))
            ch = json.loads((df / "selectors" / "lr_choice.json").read_text()); self.assertEqual(set(ch), {"vcs", "logistic", "direct"})
            self.assertEqual(len(list((df / "d1_30" / "configs").glob("configs_*.txt"))), 3); self.assertEqual(len(list((df / "d1_30" / "configs").glob("timing_*.txt"))), 3)

    def test_stops(self):
        with tempfile.TemporaryDirectory() as t:
            p, log, calls, _ = run_driver(Path(t), frozen=False); self.assertEqual(p.returncode, 1); self.assertIn("not frozen", log); self.assertEqual(calls, [])
        with tempfile.TemporaryDirectory() as t:
            p, log, calls, _ = run_driver(Path(t), STUB_FAIL_ON="train logistic 0.0001"); self.assertEqual(p.returncode, 1); self.assertIn("STOP: no job id", log)
            self.assertNotIn("lr choice", log)
        with tempfile.TemporaryDirectory() as t:
            p, log, calls, _ = run_driver(Path(t), STUB_F001_FAILS=9); self.assertEqual(p.returncode, 1); self.assertIn("unit failed 3 times", log)
            self.assertEqual(sum("bundle f001" in c for c in calls), 3)
        with tempfile.TemporaryDirectory() as t:
            p, log, calls, _ = run_driver(Path(t), PCA_BASIS="other"); self.assertEqual(p.returncode, 1); self.assertIn("PCA_BASIS", log)
        with tempfile.TemporaryDirectory() as t:                                           # D1-30's DEV-risk read comes first: D1-FULL's waits for it
            p, log, calls, _ = run_driver(Path(t), d30_devrisk_done=False); self.assertEqual(p.returncode, 1); self.assertIn("D1-30 DEV-risk output absent", log)
            self.assertFalse(any("devrisk" in c for c in calls)); self.assertEqual(sum(" eval " in c for c in calls), 12)


if __name__ == "__main__":
    unittest.main()
