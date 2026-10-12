"""R7-30 D1-30 pieces on CPU with fakes / tiny models: pinned ChartQA evaluator; freeze gate and evaluation-set rules; runner bookkeeping,
resume, OCR anchor and completion marker; SelectorBank fixed / dynamic budgets against the frozen D1-20 rule; SparseVLM official counter and
adapted lists; timing rows; scorer (coverage guard, latency interpolation, contrasts, Holm, ablations, bootstrap); unit plan."""
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))
import r7_d1_30_data as DD  # noqa: E402
import r7_d1_30_eval as EV  # noqa: E402
import r7_d1_30_plan as PL  # noqa: E402
import r7_d1_30_run as RUN  # noqa: E402
import r7_d1_30_score as SC  # noqa: E402
import r7_d1_30_sparsevlm as SV  # noqa: E402
import r7_d1_30_timing as TM  # noqa: E402
import r7_d1_selectors as SEL  # noqa: E402


class Evaluator(unittest.TestCase):
    def test_relaxed_correctness_pinned_semantics(self):
        self.assertEqual(EV.chartqa_score("10.4", "10"), 1.0); self.assertEqual(EV.chartqa_score("10.6", "10"), 0.0)
        self.assertEqual(EV.chartqa_score("12%", "0.12"), 1.0); self.assertEqual(EV.chartqa_score(" Yes ", "yes"), 1.0)
        self.assertEqual(EV.chartqa_score("0", "0"), 1.0)                                   # zero target -> exact-match branch
        self.assertEqual(EV.chartqa_score("0.01", "0"), 0.0); self.assertEqual(EV.chartqa_score("Paris", "London"), 0.0)
        self.assertEqual(EV.chartqa_score("14.", "14"), 1.0)                                 # float('14.') parses: numeric branch
        self.assertEqual(EV.PINNED["commit"], "6fe25c1dc8151823ee3b479519d8d5948812fee4")


class Sets(unittest.TestCase):
    def test_freeze_gate(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "proto.yaml"; p.write_text("status: frozen_D1-10__staged_D1-20_D1-30\nD1_30:\n  x: 1\n")
            with self.assertRaises(SystemExit):
                DD.require_freeze(p)
            p.write_text("d1_30_staged_freeze_20261013:\n  x: 1\n"); DD.require_freeze(p)

    def test_gqa_dev_rule(self):
        rows = [{"qid": f"{i}_{j}", "image_id": f"img{i}", "question": f"Q{i} {j}?", "answer": ("yes" if j % 2 else f"obj{j}"), "structural": "s",
                 "semantic": "m", "detailed": "d"} for i in range(6) for j in range(9)]
        seen = {("img0", DD.DA.norm_q("Q0 0?"))}; order = ["img2", "img0", "img1", "img9"]
        out, st = DD.pick_gqa_dev(rows, order, seen)
        self.assertEqual([r["image_id"] for r in out][::4], ["img2", "img0", "img1"])          # manifest order; img9 has no questions
        self.assertNotIn("0_0", {r["qid"] for r in out}); self.assertEqual(st["excluded_llava_seen_pairs_on_dev_images"], 1)
        self.assertEqual(st["images_without_unseen_questions"], 1)
        for img in ("img0", "img1", "img2"):
            got = [r["qid"] for r in out if r["image_id"] == img]
            cand = sorted([r for r in rows if r["image_id"] == img and r["qid"] != "0_0"], key=lambda r: DD.DA.h("d1-q", r["qid"]))[:4]
            self.assertEqual(got, [r["qid"] for r in cand])
        self.assertTrue(all(r["prompt"].endswith(DD.SHORT) for r in out))
        self.assertEqual(out, DD.pick_gqa_dev(list(reversed(rows)), order, seen)[0])         # independent of row order

    def test_textvqa_exclusion_and_chartqa_types(self):
        rows = [{"question_id": i, "image_id": f"im{i % 3}", "question": "q?", "answers": ["a"] * 10, "ocr_tokens": ["X", "Y"]} for i in range(9)]
        it = DD.textvqa_items(rows, {"im1"}); self.assertEqual(len(it), 6); self.assertTrue(all(x["image_id"] != "im1" for x in it))
        self.assertEqual(it[0]["prompt_ocr_anchor"], "q?\nReference OCR token: X, Y" + DD.SHORT)
        ch = DD.chartqa_items([{"imgname": "a.png", "query": "How many?", "label": "3", "type": "human_test"},
                               {"imgname": "b.png", "query": "Max?", "label": "7", "type": "augmented_val"}])
        self.assertEqual([c["split"] for c in ch], ["human", "augmented"]); self.assertEqual(ch[0]["prompt"], "How many?" + DD.CHART_SUFFIX)
        with self.assertRaises(SystemExit):
            DD.chartqa_items([{"imgname": "c.png", "query": "?", "label": "1", "type": "other"}])


class BuildSynthetic(unittest.TestCase):
    """The post-freeze build path end to end on synthetic files with the production schemas (no real data read)."""

    def test_build(self):
        import io
        import pyarrow as pa
        import pyarrow.parquet as pq
        from PIL import Image
        def png(c):
            b = io.BytesIO(); Image.new("RGB", (8, 8), (c, 0, 0)).save(b, "PNG"); return b.getvalue()
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); data, d1, fx = t / "data", t / "d1", t / "fx"
            (d1 / "data").mkdir(parents=True); (data / "gqa" / "train_all_questions").mkdir(parents=True); (data / "gqa" / "images").mkdir(parents=True)
            man = [{"qid": f"m{i}", "image_id": img, "role": role, "question": "Is it?"} for i, (img, role) in enumerate([("10", "FIT"), ("20", "DEV"), ("30", "DEV")])]
            (d1 / "data" / "d1_manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in man))
            (d1 / "data" / "d1_manifest_meta.json").write_text(json.dumps({"frozen": True, "manifest_sha256": "abc"}))
            qs = {f"q{i}": {"imageId": img, "question": f"What is {i}?", "answer": ("yes" if i % 2 else "cat"), "types": {"structural": "query"}}
                  for i, img in enumerate(["10", "20", "20", "20", "20", "20", "30", "30"])}
            (data / "gqa" / "train_all_questions" / "train_all_questions_0.json").write_text(json.dumps(qs))
            for img in ("20", "30"):
                (data / "gqa" / "images" / f"{img}.jpg").write_bytes(b"x")
            (data / "llava_mix665k").mkdir(); (data / "llava_mix665k" / "llava_v1_5_mix665k.json").write_text(json.dumps(
                [{"image": "gqa/images/20.jpg", "conversations": [{"from": "human", "value": "<image>\nWhat is 1?" + DD.SHORT}, {"from": "gpt", "value": "yes"}]}]))
            (data / "textvqa_lmms" / "data").mkdir(parents=True); fx.mkdir()
            tv = [{"question_id": i, "image_id": f"t{i % 3}", "question": f"q{i}?", "answers": ["a"] * 10, "ocr_tokens": ["X"], "image": {"bytes": png(i % 3), "path": None}}
                  for i in range(6)]
            pq.write_table(pa.Table.from_pylist(tv), data / "textvqa_lmms" / "data" / "validation-00000-of-00001.parquet")
            (fx / "textvqa.jsonl").write_text(json.dumps({"image_id": "t1", "qid": "1"}) + "\n")
            (data / "chartqa" / "data").mkdir(parents=True)
            ch = [{"imgname": f"c{i % 2}.png", "query": f"Q{i}?", "label": str(i), "type": ("human_test" if i < 2 else "augmented_test"), "image": png(i % 2)} for i in range(4)]
            pq.write_table(pa.Table.from_pylist(ch), data / "chartqa" / "data" / "val-00000-of-00001.parquet")
            proto = t / "proto.yaml"; proto.write_text("status: x\n")
            with self.assertRaises(SystemExit):
                DD.build(t / "out", proto, ["gqa_dev"], data=data, d1=d1, fixtures=fx)
            proto.write_text("d1_30_staged_freeze_test:\n  a: 1\n")
            s = DD.build(t / "out", proto, ["gqa_dev", "textvqa_val", "chartqa_val"], data=data, d1=d1, fixtures=fx)
            g = [json.loads(x) for x in (t / "out" / "gqa_dev.jsonl").read_text().splitlines()]
            self.assertEqual({r["image_id"] for r in g}, {"20", "30"}); self.assertNotIn("q1", {r["qid"] for r in g})      # seen pair excluded
            self.assertEqual(sum(r["image_id"] == "20" for r in g), 4); self.assertEqual(s["gqa_dev_rule"]["excluded_llava_seen_pairs_on_dev_images"], 1)
            tvi = [json.loads(x) for x in (t / "out" / "textvqa_val.jsonl").read_text().splitlines()]
            self.assertEqual(len(tvi), 4); self.assertTrue(all(r["image_id"] != "t1" and Path(r["image_file"]).exists() for r in tvi))
            self.assertEqual(s["textvqa_rule"]["excluded_questions"], 2)
            c = [json.loads(x) for x in (t / "out" / "chartqa_val.jsonl").read_text().splitlines()]
            self.assertEqual([r["split"] for r in c], ["human", "human", "augmented", "augmented"]); self.assertEqual(s["chartqa_rule"]["charts"], 2)
            self.assertTrue(all(Path(r["image_file"]).read_bytes()[:4] == b"\x89PNG" for r in c))


class FakeHost:
    def __init__(self):
        self.torch, self.prompts = torch, []

    def encode(self, images):
        t = torch.zeros(len(images), 576, 8); t[:, :, 0] = 1.0; t[:, :, 1] = torch.arange(576.0) / 576; return t

    def segments(self, prompt, with_image=True):
        self.prompts.append(prompt); ids = torch.tensor([5, 6, 7]); return ids[:1], ids[1:]

    def embed_ids(self, ids):
        e = torch.zeros(len(ids), 8); e[:, 2] = 1.0; return e

    def question_vector(self, q):
        return torch.tensor([0.0, 1.0] + [0.0] * 6)

    class _Tok:
        def __call__(self, q, add_special_tokens=False, return_tensors=None):
            return Namespace(input_ids=torch.tensor([[1, 2, 3]]))
    tok = _Tok()

    def generate(self, seqs, max_new_tokens=16, min_new_tokens=0):
        return [f"v{int(s[:, 0].sum())}" for s in seqs]

    def last_logits(self, seqs):
        return torch.zeros(len(seqs), 4)


class FakeBank:
    """Fixed budgets as asked; dynamic budgets alternate between two grid values (two kept-token buckets)."""

    def __init__(self):
        self.tau = {"u1": {0.25: 0.0, 0.5: 0.0, 0.75: 0.0}}

    def prepare(self, feats, q_list):
        return {"n": feats.shape[0]}

    def kept(self, unit, st, budget=None, frac=None):
        if budget is not None:
            return [np.arange(budget)] * st["n"]
        k = int(576 * frac); return [np.arange(k if i % 2 == 0 else 2 * k) for i in range(st["n"])]


def tiny_sets(d: Path, bench="gqa_fixture", n=5):
    d.mkdir(parents=True, exist_ok=True); rows = []
    for i in range(n):
        rows.append({"bench": bench, "qid": f"q{i}", "image_id": f"img{i // 2}", "image_file": f"f{i // 2}", "question": f"q{i}?", "prompt": f"q{i}?\nShort",
                     "answer": "v576" if i % 2 == 0 else "no", "prompt_ocr_anchor": f"q{i}?\nOCR\nShort", "answers": ["v576"] * 10})
    (d / f"{bench}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows)); return rows


class Runner(unittest.TestCase):
    def test_configs_resume_anchor_marker(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); tiny_sets(t / "sets"); cf = t / "cfg.txt"
            cf.write_text("full\nfull_ocr\nstatic:random:72\nstatic:qsim:144\nsel:u1:288\ndyn:u1:0.5\n")
            a = Namespace(sets_dir=str(t / "sets"), bench="gqa_fixture", configs_file=str(cf), shard="0/1", out=str(t / "out"), chunk=2, batch=3, limit=0)
            h = FakeHost(); RUN.run(a, host=h, bank=FakeBank(), opener=lambda fp: None)
            got = {c: [r for p in RUN.cfg_paths(t / "out" / "gqa_fixture", c) for r in json.loads("[" + ",".join(p.read_text().splitlines()) + "]")] for c in
                   ("full", "full_ocr", "static:random:72", "static:qsim:144", "sel:u1:288", "dyn:u1:0.5")}
            self.assertEqual(sorted(x.name for x in (t / "out" / "gqa_fixture").glob("full*.jsonl")), ["full@s0of1.jsonl", "full_ocr@s0of1.jsonl"])
            self.assertTrue(all(len(v) == 5 for v in got.values()))
            self.assertEqual({r["pred"] for r in got["full"]}, {"v576"}); self.assertEqual({r["pred"] for r in got["static:random:72"]}, {"v72"})
            self.assertEqual({r["kept"] for r in got["sel:u1:288"]}, {288}); self.assertEqual({r["kept"] for r in got["dyn:u1:0.5"]}, {288, 576})
            self.assertEqual(sum("OCR" in p for p in h.prompts), 5)                           # anchor prompt only for full_ocr
            self.assertTrue(all(p.endswith("\nShort") for p in h.prompts))                  # the item's prompt as given: no second suffix
            self.assertTrue((t / "out" / "gqa_fixture" / "done_shard0of1.json").exists())
            h2 = FakeHost(); RUN.run(a, host=h2, bank=FakeBank(), opener=lambda fp: None); self.assertEqual(h2.prompts, [])  # resume: nothing redone

    def test_single_writer_files_and_union(self):
        """Each shard appends only to its own <cfg>@s<i>of<n>.jsonl; answers already in the plain file (or another shard's file) are not redone."""
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); tiny_sets(t / "sets", n=12); cf = t / "cfg.txt"; cf.write_text("full\nstatic:grid:72\n"); od = t / "out" / "gqa_fixture"; od.mkdir(parents=True)
            early = RUN.load_items(t / "sets", "gqa_fixture")[:5]
            (od / "full.jsonl").write_text("".join(json.dumps({"qid": it["qid"], "image_id": it["image_id"], "pred": "early", "kept": 576}) + "\n" for it in early))
            for sh in ("0/2", "1/2"):
                h = FakeHost(); RUN.run(Namespace(sets_dir=str(t / "sets"), bench="gqa_fixture", configs_file=str(cf), shard=sh, out=str(t / "out"), chunk=4, batch=4, limit=0),
                                        host=h, bank=FakeBank(), opener=lambda fp: None)
            names = sorted(x.name for x in od.glob("*.jsonl"))
            self.assertEqual(names, ["full.jsonl", "full@s0of2.jsonl", "full@s1of2.jsonl", "static__grid__72@s0of2.jsonl", "static__grid__72@s1of2.jsonl"])
            full = [json.loads(x) for p in RUN.cfg_paths(od, "full") for x in p.read_text().splitlines()]
            self.assertEqual(len(full), 12); self.assertEqual(len({r["qid"] for r in full}), 12)                  # early answers kept, none redone
            self.assertEqual(sum(r["pred"] == "early" for r in full), 5)
            self.assertTrue((od / "done_shard0of2.json").exists() and (od / "done_shard1of2.json").exists())

    def test_shards_partition_groups(self):
        with tempfile.TemporaryDirectory() as t:
            rows = tiny_sets(Path(t), n=40); parts = [RUN.load_items(t, "gqa_fixture", f"{i}/3") for i in range(3)]
            self.assertEqual(sorted(q["qid"] for p in parts for q in p), sorted(r["qid"] for r in rows))
            self.assertEqual(len({p["image_id"] for p in parts[0]} & {p["image_id"] for p in parts[1]}), 0)

    def test_config_grid(self):
        main, abl = RUN.unit_names({o: {"chosen_lr": 3e-4} for o in RUN.OBJECTIVES})
        mu = [u for o in RUN.OBJECTIVES for u in main[o]]; au = abl["A1"] + abl["A2"]
        cfg = RUN.config_list(mu, au); self.assertEqual(len(cfg), 1 + 12 + 36 + 24 + 27); self.assertEqual(len(RUN.config_list(mu, au, True)), 101)
        self.assertIn("sel:vcs_lr0.0003_s2:432", cfg); self.assertIn("sel:direct_lr0.0003_s1_dtarget:72", cfg)
        self.assertFalse(any(c.startswith("dyn:") and c.split(":")[1].endswith(("_noq", "_dtarget")) for c in cfg))


class Bank(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory(); d = Path(self.t.name); torch.manual_seed(0)
        cfg = SEL.Cfg(emb_dim=8, vis_dim=4, d=16, heads=2, layers=1, ffn=32, dropout=0.0)
        from dataclasses import asdict
        for obj in ("vcs", "direct"):
            m = SEL.Selector(cfg, obj); torch.save({"state_dict": m.state_dict(), "cfg": asdict(cfg), "objective": obj}, d / f"{obj}_u.pt")
        torch.save({"mean": torch.zeros(8), "basis": torch.randn(8, 4)}, d / "pca.pt")
        (d / "vcs_u.json").write_text(json.dumps({"dynamic": {"0.5": {"tau": 0.0}}}))
        self.bank = RUN.SelectorBank(d, d / "pca.pt", ["vcs_u", "direct_u"], device="cpu")
        self.feats = torch.randn(6, 576, 8); self.q = [torch.randn(3 + i % 2, 8) for i in range(6)]

    def tearDown(self):
        self.t.cleanup()

    def test_fixed_budget_is_hard_topk(self):
        st = self.bank.prepare(self.feats, self.q)
        for k in (72, 432):
            kept = self.bank.kept("vcs_u", st, budget=k); self.assertTrue(all(len(x) == k and np.all(np.diff(x) > 0) for x in kept))
        g, _ = self.bank._gates("vcs_u", st); top = torch.argsort(g[0], descending=True)[:72].sort().values.numpy()
        self.assertTrue(np.array_equal(self.bank.kept("vcs_u", st, budget=72)[0], top))

    def test_dynamic_rule_matches_d1_20(self):
        st = self.bank.prepare(self.feats, self.q); R = self.bank.risks("direct_u", st)
        for tau in (np.percentile(R, 30), np.percentile(R, 70)):
            self.bank.tau["direct_u"] = {0.5: float(tau)}; ks = self.bank.dynamic_budgets("direct_u", st, 0.5)
            ok = R <= tau; ok[:, -1] = True; ref = np.asarray(RUN.GRID)[np.argmax(ok, 1)]          # r7_d1_selectors.dynamic_tau's mean_frac rule
            self.assertTrue(np.array_equal(ks, ref))
            kept = self.bank.kept("direct_u", st, frac=0.5); self.assertEqual([len(x) for x in kept], ks.tolist())
        taus = RUN.calibrate_tau(R)
        for f, tau in taus.items():
            ok = R <= tau; ok[:, -1] = True; frac = np.mean(np.asarray(RUN.GRID)[np.argmax(ok, 1)] / 576)
            self.assertLessEqual(frac, f + 1e-9)

    def test_inputs_match_cache_rounding(self):
        st = self.bank.prepare(self.feats, self.q); ref = ((self.feats.float() - self.bank.mean) @ self.bank.basis).half().float()
        self.assertTrue(torch.equal(st["v"], ref)); self.assertTrue(bool(st["pad"][0, 3])); self.assertFalse(bool(st["pad"][1, 3]))


class SparseVLMLists(unittest.TestCase):
    def test_counter_reproduces_r7_20(self):
        self.assertEqual(SV.official_equal_tokens(SV.OFFICIAL_V1[192]), 192.03125); self.assertEqual(SV.official_equal_tokens(SV.OFFICIAL_V1[64]), 64.8125)
        self.assertEqual(SV.official_equal_tokens([576, 576, 576]), 576.0)

    def test_adapted_lists(self):
        for T in SV.ANCHORS:
            self.assertEqual(SV.adapted_list(T), SV.OFFICIAL_V1[T])
        for T in SV.ADAPTED:
            self.assertLess(abs(SV.official_equal_tokens(SV.adapted_list(T)) - T), 0.5); self.assertLessEqual(max(SV.adapted_list(T)), 576)
        self.assertEqual(SV.row_tag(144), "sparsevlm:144:adapted"); self.assertEqual(SV.row_tag(192, True), "sparsevlm:192:official:ocr")
        self.assertEqual(len(SV.rows_for("textvqa_val", [192, 64, 72])), 5); self.assertEqual(len(SV.rows_for("gqa_dev", [192, 64, 72])), 3)
        self.assertEqual(SV.ANCHORS, (192, 128, 96, 64)); self.assertEqual(len(SV.rows_for("textvqa_val", list(SV.ANCHORS + SV.ADAPTED))), 12)


class Timing(unittest.TestCase):
    def test_rows(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); tiny_sets(t / "sets", n=40); cf = t / "cfg.txt"; cf.write_text("full\nstatic:grid:72\nsel:u1:144\ndyn:u1:0.25\n")
            TM.main(["--sets-dir", str(t / "sets"), "--bench", "gqa_fixture", "--configs-file", str(cf), "--out", str(t / "o"), "--n", "32", "--n-big", "16",
                     "--n1", "4", "--batches", "1", "4", "16"], host=FakeHost(), bank=FakeBank(), opener=lambda fp: None)
            rows = json.loads((t / "o" / "timing_gqa_fixture.json").read_text())["rows"]
            self.assertEqual(len(rows), 12); self.assertEqual({r["B"] for r in rows}, {1, 4, 16})
            self.assertEqual({r["pool"] for r in rows if r["B"] == 1}, {4}); self.assertEqual({r["pool"] for r in rows if r["B"] == 4}, {32})
            self.assertEqual({r["pool"] for r in rows if r["B"] == 16}, {16})
            self.assertEqual({r["n_batches"] for r in rows if r["B"] == 4}, {8}); self.assertEqual({r["n_batches"] for r in rows if r["B"] == 16}, {1})
            r = next(r for r in rows if r["config"] == "sel:u1:144" and r["B"] == 4); self.assertEqual(r["mean_visual_tokens"], 144.0)
            lo, hi = r["per_item_ms_median_ci95"]; self.assertTrue(lo <= r["per_item_ms_median"] <= hi)
            d = next(r for r in rows if r["config"] == "dyn:u1:0.25" and r["B"] == 4)                 # bucketed: budgets 144 / 288 alternate
            self.assertTrue(d["bucketed"]); self.assertEqual(d["budget_share"], {"144": 0.5, "288": 0.5}); self.assertEqual(d["mean_visual_tokens"], 216.0)
            self.assertEqual(d["stage2_batches"], {"144": 4, "288": 4}); self.assertEqual(d["stage2_cycled_items"], {"144": 0, "288": 0})
            d16 = next(r for r in rows if r["config"] == "dyn:u1:0.25" and r["B"] == 16)               # 8 items per bucket cycled to one full batch of 16
            self.assertEqual(d16["stage2_batches"], {"144": 1, "288": 1}); self.assertEqual(d16["stage2_cycled_items"], {"144": 8, "288": 8})


class Scoring(unittest.TestCase):
    def setup_bench(self, t: Path, drop=None, n=40):
        bench = "gqa_dev"; items = [{"bench": bench, "qid": f"q{i}", "image_id": f"img{i // 2}", "question": "?", "answer": "yes", "prompt": "?"} for i in range(n)]
        (t / "sets").mkdir(parents=True); (t / "sets" / f"{bench}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in items))
        pd = t / "pred" / bench; pd.mkdir(parents=True); g = np.random.default_rng(0)
        choice = {"vcs": {"chosen_lr": 3e-4}, "logistic": {"chosen_lr": 1e-4}, "direct": {"chosen_lr": 3e-4}}; (t / "lr.json").write_text(json.dumps(choice))
        main, abl = RUN.unit_names(choice); cfgs = RUN.config_list([u for o in RUN.OBJECTIVES for u in main[o]], abl["A1"] + abl["A2"])
        sv = [SV.row_tag(T, o) for T, o in SV.rows_for(bench, list(SV.ANCHORS + SV.ADAPTED))]
        acc = {72: 0.55, 144: 0.65, 288: 0.75, 432: 0.85, 576: 0.95}
        def write(cfg, a, kept=576):
            rows = [{"qid": it["qid"], "image_id": it["image_id"], "pred": "yes" if g.uniform() < a else "no", "kept": kept} for it in items]
            if drop == cfg:
                rows = rows[:-1]
            (pd / f"{RUN.cfg_file(cfg)}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        for c in cfgs + sv:
            if c == "full":
                write(c, acc[576])
            elif c.startswith(("static:", "sel:")):
                k = int(c.split(":")[2]); write(c, acc[k] + (0.05 if c.startswith("sel:vcs") else 0.0), k)
            elif c.startswith("dyn:"):
                write(c, 0.8, {"0.25": 144, "0.5": 288, "0.75": 432}[c.split(":")[2]])
            else:
                T = int(c.split(":")[1]); write(c, 0.7, T)
        lat = {72: 40.0, 144: 55.0, 288: 70.0, 432: 85.0}; tm = []
        for B in (1, 4):
            tm.append({"config": "full", "B": B, "per_item_ms_median": 100.0})
            for c in cfgs:
                if c.startswith("static:") or c.startswith(f"sel:{main['vcs'][0]}:"):
                    tm.append({"config": c, "B": B, "per_item_ms_median": lat[int(c.split(":")[2])] + (2.0 if c.startswith("sel:") else 0.0)})
                elif c.startswith("dyn:"):
                    tm.append({"config": c, "B": B, "per_item_ms_median": {"0.25": 60.0, "0.5": 75.0, "0.75": 90.0}[c.split(":")[2]]})
        (t / "tim").mkdir(); (t / "tim" / f"timing_{bench}.json").write_text(json.dumps({"rows": tm}))
        svt = [{"config": c, "B": 1, "per_item_ms_median": 45.0 + int(c.split(":")[1]) / 10} for c in sv]
        (t / "tim" / f"sparsevlm_timing_{bench}.json").write_text(json.dumps({"rows": svt}))
        return ["--sets-dir", str(t / "sets"), "--pred-dir", str(t / "pred"), "--timing-dir", str(t / "tim"), "--lr-choice", str(t / "lr.json"),
                "--out", str(t / "res"), "--benches", bench, "--reps", "200"]

    def test_scores_contrasts_holm(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); SC.main(self.setup_bench(t)); res = json.loads((t / "res" / "d1_30_results.json").read_text())["benches"]["gqa_dev"]
            self.assertEqual(res["latency_points_ms"]["4"], {"L25": 55.0, "L50": 70.0, "L75": 85.0})
            C = res["contrasts"]
            for L in ("L25", "L50", "L75"):
                for k in (f"S1-S2@{L}", f"S1-S3@{L}", f"S1-SparseVLM@{L}[budget]"):
                    self.assertEqual(C[k]["role"], "primary"); self.assertIn("p_holm", C[k]); self.assertEqual(C[k]["holm_family_size"], 3)
                    self.assertTrue(C[k]["ci95"][0] - 1e-9 <= C[k]["delta"] <= C[k]["ci95"][1] + 1e-9)
                self.assertEqual(C[f"S1-SparseVLM@{L}[latency_b1]"]["role"], "secondary")
            self.assertIn("A1-S1@k72", C); self.assertIn("A3:S1_dyn-fixed@f0.5", C)
            self.assertEqual(res["at_latency"]["4"]["grid"]["L50"]["seed_sd"], 0.0)
            self.assertAlmostEqual(res["at_latency"]["4"]["grid"]["L50"]["mean"], res["fixed_budget"]["static:grid:288"]["quality"])
            self.assertEqual(res["dynamic_budget_distribution"]["dyn:vcs_lr0.0003_s0:0.5"]["288"], 1.0)

    def test_integrity_first_record_wins(self):
        """Duplicates across files resolve to the first complete record (plain file first, then shard files by name); torn / blank lines are
        dropped and counted; the readout still passes coverage."""
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); args = self.setup_bench(t); pd = t / "pred" / "gqa_dev"; c = "static:grid:144"
            first = (pd / f"{RUN.cfg_file(c)}.jsonl").read_text().splitlines()
            (pd / f"{RUN.cfg_file(c)}.jsonl").write_text("\n".join(first[:20]) + "\n")
            dup = json.loads(first[3]); dup["pred"] = "flipped"
            (pd / f"{RUN.cfg_file(c)}@s0of2.jsonl").write_text("\n".join(first[20:] + [json.dumps(dup), '{"qid": "q7", "image_id": "img3", "pr', ""]) + "\n")
            recs, n = SC.load_config(pd, c)
            self.assertEqual(n["duplicate_qids"], 1); self.assertEqual(n["dropped_lines"], 1); self.assertEqual(n["blank_lines"], 1)
            self.assertEqual(recs[dup["qid"]]["pred"], json.loads(first[3])["pred"])                     # the first complete record wins
            SC.main(args); res = json.loads((t / "res" / "d1_30_results.json").read_text())["benches"]["gqa_dev"]
            self.assertEqual(res["integrity"]["totals"], {"dropped_lines": 1, "blank_lines": 1, "duplicate_qids": 1})
            self.assertIn(c, res["integrity"]["configs_with_issues"])

    def test_coverage_guard(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            with self.assertRaises(SystemExit):
                SC.main(self.setup_bench(t, drop="static:grid:144"))

    def test_helpers(self):
        self.assertEqual(SC.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        self.assertEqual(SC.bracket([1.0, 3.0], 2.0), (0, 1, 0.5)); self.assertIsNone(SC.bracket([1.0, 3.0], 4.0)); self.assertEqual(SC.bracket([3.0, 1.0], 3.0), (0, 0, 0.0))
        self.assertAlmostEqual(SC.first_within([10, 20, 30], [0.5, 0.7, 0.9], 0.8), 25.0); self.assertIsNone(SC.first_within([10, 20], [0.1, 0.2], 0.5))
        g = ["a", "a", "b", "c", "c", "c"]; x = np.array([1, 0, 1, 1, 0, 0.5]); bt = SC.Boot(g, reps=50, seed=1)
        for r in range(5):                                                                   # replicate mean = explicit resample mean
            idx = np.concatenate([np.flatnonzero(bt.inv == j).repeat(int(c)) for j, c in enumerate(bt.W[r])]); self.assertAlmostEqual(bt.means(x)[r], x[idx].mean())


class Plan(unittest.TestCase):
    def test_plan(self):
        with tempfile.TemporaryDirectory() as t:
            res = PL.write_configs({o: {"chosen_lr": 3e-4} for o in RUN.OBJECTIVES}, Path(t))
            self.assertEqual(res["gqa_dev"], {"configs": 100, "timing_configs": 1 + 12 + 4 + 27}); self.assertEqual(res["textvqa_val"]["configs"], 101)
            L = PL.lines(Path("/x")); self.assertEqual(len(L), 6 + 4 + 2 + 3 + 3 + 1 + 3 + 1)
            self.assertTrue(all("--partition=H100,L40S" in l or "--nodelist=nodesumo01" in l for l, _ in L))
            self.assertTrue(all("RTX6000PRO" not in l and "--export" not in l for l, _ in L))


if __name__ == "__main__":
    unittest.main()
