"""R7-50 sealed-test plumbing on CPU with synthetic files only (no real data, no closed/ file): build (inputs without answers, sealed labels,
freeze / record / size / hash / count / instruction-mix gates, GQA image rule, ChartQA rules, id prefix), §A.2 rows of phases L and F from the
frozen D1-30 grid, unit lines, generation guards and the D1-30 runner on fakes, the full-scale integrity gate's plan (GPU pins from the reference
markers) and comparison, manifest coverage / hashing, the scoring gate (incl. the committed-manifest check) and the readout of phases L and F,
and stubbed dry runs of the driver (phases L / F / Q).
    python -I tests/test_r7_50.py -v        (slurm/r7_50_cpu.sbatch tests)"""
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))
import r7_50_test as T  # noqa: E402
import r7_d1_30_data as DD  # noqa: E402
import r7_d1_30_plan as PL  # noqa: E402
import r7_d1_30_run as RUN  # noqa: E402

FROZEN = "status: frozen\nr7_50_freeze_20261013:\n  by: test\n"
LR = {"vcs": {"chosen_lr": 3e-4}, "logistic": {"chosen_lr": 1e-4}, "direct": {"chosen_lr": 1e-4}}
LR_F = {"vcs": {"chosen_lr": 1e-4}, "logistic": {"chosen_lr": 3e-4}, "direct": {"chosen_lr": 3e-4}}     # a D1-FULL-like choice (other tags)


def png(c, size=8):
    from PIL import Image
    b = io.BytesIO(); Image.new("RGB", (size, size), (c, 0, 0)).save(b, "PNG"); return b.getvalue()


def jpg_file(p: Path, c):
    from PIL import Image
    p.parent.mkdir(parents=True, exist_ok=True); Image.new("RGB", (8, 8), (0, c, 0)).save(p, "JPEG")


def synth_sources(t: Path, n_img=6, q_per=3, n_chart=8):
    """A GQA balanced-format file with distinctive answers, its images, a ChartQA-format parquet (human + augmented) and a tiny instruction mix."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    gq = {}
    for i in range(n_img):
        jpg_file(t / "images" / f"{1000 + i}.jpg", 10 * i)
        for j in range(q_per):
            gq[f"{i}{j:02d}"] = {"imageId": f"{1000 + i}", "question": f"Is thing {i}-{j} red?", "answer": ("ansyes" if j % 2 else f"zebra{i}{j}"),
                                 "types": {"structural": ("verify" if j % 2 else "query"), "semantic": "attr", "detailed": "d"}, "isBalanced": True}
    (t / "src").mkdir(parents=True, exist_ok=True); (t / "src" / "gqa_questions.json").write_text(json.dumps(gq))
    ch = [{"imgname": f"chart{i // 2}.png", "query": f"What is value {i}?", "label": f"{40 + i}", "type": ("human_test" if i % 2 == 0 else "augmented_test"),
           "image": png(i // 2)} for i in range(n_chart)]
    pq.write_table(pa.Table.from_pylist(ch), t / "src" / "chartqa.parquet")
    mix = [{"image": "gqa/images/1000.jpg", "conversations": [{"from": "human", "value": "<image>\nIs thing 0-1 red?" + DD.SHORT}, {"from": "gpt", "value": "x"}]},
           {"image": "vg/VG_100K/1001.jpg", "conversations": []}, {"image": "vg/VG_100K_2/1000.jpg", "conversations": []},
           {"image": "coco/train2017/1.jpg", "conversations": []}, {"image": "chartqa/train/png/a.png", "conversations": []}]
    (t / "src" / "mix.json").write_text(json.dumps(mix))
    return t / "src" / "gqa_questions.json", t / "src" / "chartqa.parquet", gq, ch


class Build(unittest.TestCase):
    def test_dev_build_inputs_without_answers(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); g, c, gq, ch = synth_sources(t); s = T.build(t / "root", g, c, images_dir=t / "images", mix665k=t / "src" / "mix.json"); L = T.Layout(t / "root")
            ex = s["benches"]["gqa_testdev"]["exposure_check"]
            self.assertEqual((ex["images_in_mix_gqa_paths"], ex["images_in_mix_vg_paths"], ex["images_in_mix_any"], ex["numeric_image_ids"]), (1, 2, 2, 6))
            self.assertEqual(ex["questions_seen_in_mix_gqa_turns"], 1); self.assertEqual(s["benches"]["chartqa_test"]["exposure_check"]["mix_entries_with_chart_image_paths"], 1)
            gi, ci = T.read_jsonl_strict(L.inputs_file("gqa_testdev")), T.read_jsonl_strict(L.inputs_file("chartqa_test"))
            for items in (gi, ci):
                self.assertTrue(all(set(it) == {"bench", "qid", "image_id", "image_file", "question", "prompt"} for it in items))
            self.assertEqual(len(gi), 18); self.assertEqual(len(ci), 8)
            self.assertTrue(all(it["prompt"] == it["question"] + DD.SHORT for it in gi))
            self.assertTrue(all(it["prompt"] == it["question"] + DD.CHART_SUFFIX for it in ci))
            self.assertEqual([it["qid"] for it in ci], [f"chartqa_test_{i:05d}" for i in range(8)])
            for it in ci:                                                                    # extracted chart = original bytes
                self.assertEqual(Path(it["image_file"]).read_bytes(), png(int(it["image_id"][5:-4])))
            texts = [L.inputs_file(b).read_text() for b in T.BENCHES] + [L.summary.read_text()] + [L.bench_summary(b).read_text() for b in T.BENCHES]
            texts += [L.images_file(b).read_text() for b in T.BENCHES]
            for a in [r["answer"] for r in gq.values()] + [r["label"] for r in ch]:
                self.assertFalse(any(f'"{a}"' in x for x in texts), a)                 # no answer anywhere outside the sealed labels
            lab = T.read_jsonl_strict(L.labels_file("gqa_testdev")); self.assertEqual({r["answer"] for r in lab}, {r["answer"] for r in gq.values()})
            self.assertEqual(stat.S_IMODE(os.stat(L.labels_file("gqa_testdev")).st_mode), 0o400)
            cl = T.read_jsonl_strict(L.labels_file("chartqa_test")); self.assertEqual([r["split"] for r in cl], ["human", "augmented"] * 4)
            self.assertEqual(s["benches"]["chartqa_test"]["split_counts"], {"human": 4, "augmented": 4}); self.assertTrue(s["ok"])
            self.assertEqual(s["benches"]["gqa_testdev"]["images"], 6); self.assertIsNone(s["freeze"])
            self.assertEqual(L.pred, t / "root" / "sealed" / "pred_L"); self.assertEqual(T.Layout(t / "root", "F").allow, t / "root" / "ALLOW_SCORING_F")
            with self.assertRaises(SystemExit):                                              # one-shot
                T.build(t / "root", g, c, images_dir=t / "images")
            T.build(t / "root2", g, c, images_dir=t / "images", chartqa_qid_prefix="chartqa_val")    # development-only id prefix
            self.assertEqual([it["qid"] for it in T.read_jsonl_strict(T.Layout(t / "root2").inputs_file("chartqa_test"))][:2], ["chartqa_val_00000", "chartqa_val_00001"])

    def test_exposure_counts_by_prefix(self):
        """D50-5: test-dev image ids counted by prefix ('n'-prefixed vs numeric Visual Genome ids) and against the mix; ids only."""
        mix = {"file": "m", "sha256": "s", "gqa_images": {"2"}, "vg_images": {"2", "7"}, "gqa_pairs": {("2", DD.DA.norm_q("Is it red?"))}, "chart_entries": 0}
        items = [{"image_id": i, "question": q} for i, q in (("n161313", "Is it red?"), ("2", "Is it red?"), ("7", "What?"), ("x9", "Why?"), ("n5", "Is it red?"))]
        ex = T.exposure("gqa_testdev", items, mix)
        self.assertEqual((ex["images"], ex["n_prefixed_image_ids"], ex["numeric_image_ids"], ex["other_image_ids"]), (5, 2, 2, 1))
        self.assertEqual((ex["images_in_mix_gqa_paths"], ex["images_in_mix_vg_paths"], ex["images_in_mix_any"], ex["questions_seen_in_mix_gqa_turns"]), (1, 2, 2, 1))
        s = T.exposure_sentence({"benches": {"gqa_testdev": {"exposure_check": ex}, "chartqa_test": {"exposure_check": {"mix_entries_with_chart_image_paths": 0}}}})
        self.assertIn("2 of 5 images have 'n'-prefixed ids", s[0]); self.assertIn("ChartQA test: 0 entries", s[1])

    def test_missing_gqa_image_stops(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); g, c, _, _ = synth_sources(t); (t / "images" / "1003.jpg").unlink()
            with self.assertRaises(SystemExit) as e:
                T.build(t / "root", g, c, images_dir=t / "images")
            self.assertIn("missing", str(e.exception)); self.assertFalse(T.Layout(t / "root").inputs_file("gqa_testdev").exists())

    def test_closed_sources_gates(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); g, c, _, _ = synth_sources(t); data = t / "data"; cl = data / "gqa" / "closed"; cl.mkdir(parents=True)
            (data / "chartqa" / "closed" / "data").mkdir(parents=True)
            gq, cq = cl / "testdev_balanced_questions.json", data / "chartqa" / "closed" / "data" / "test-00000-of-00001.parquet"
            gq.write_bytes(g.read_bytes()); cq.write_bytes(c.read_bytes()); os.chmod(gq, 0); os.chmod(cq, 0)   # unreadable: refusals must come first
            csvp = t / "rec.csv"

            def rec(gb=None, gs=None):
                gq_b, gq_s = (gb if gb is not None else len(g.read_bytes())), (gs or T.hashlib.sha256(g.read_bytes()).hexdigest())
                csvp.write_text("item_dir,path,bytes,sha256\n" + f"gqa,gqa/closed/testdev_balanced_questions.json,{gq_b},{gq_s}\n" +
                                f"chartqa,chartqa/closed/data/test-00000-of-00001.parquet,{len(c.read_bytes())},{T.hashlib.sha256(c.read_bytes()).hexdigest()}\n")
            rec(); kw = dict(images_dir=t / "images", data_root=data, csv_path=csvp, expected={"gqa_testdev": 18, "chartqa_test": 8}, mix665k=t / "src" / "mix.json")
            draft = t / "draft.yaml"; draft.write_text('status: draft\nfreeze_key_expected: "r7_50_freeze_20261013: added by the coordinator"\n')
            for proto in (None, draft):
                with self.assertRaises(SystemExit) as e:
                    T.build(t / "r0", gq, cq, protocol=proto, **kw)
                self.assertIn("frozen", str(e.exception))
            fz = t / "frozen.yaml"; fz.write_text(FROZEN)
            with self.assertRaises(SystemExit) as e:                                      # official ChartQA ids are chartqa_test_<row>
                T.build(t / "r0", gq, cq, protocol=fz, chartqa_qid_prefix="chartqa_val", **kw)
            self.assertIn("no prefix override", str(e.exception))
            with self.assertRaises(SystemExit) as e:                                      # the overlap check is mandatory for the official build
                T.build(t / "r0", gq, cq, protocol=fz, **dict(kw, mix665k=None))
            self.assertIn("instruction mix", str(e.exception))
            with self.assertRaises(SystemExit) as e:                                      # official GQA images only from the production folder
                T.build(t / "r1", gq, cq, protocol=fz, **kw)
            self.assertIn("official GQA images", str(e.exception))
            saved, T.GQA_IMAGES = T.GQA_IMAGES, t / "images"
            try:
                rec(gb=1)
                with self.assertRaises(SystemExit) as e:
                    T.build(t / "r2", gq, cq, protocol=fz, **kw)
                self.assertIn("not opened", str(e.exception))                             # size check before the open (file is unreadable)
                os.chmod(gq, 0o400); os.chmod(cq, 0o400); rec(gs="0" * 64)
                with self.assertRaises(SystemExit) as e:
                    T.build(t / "r3", gq, cq, protocol=fz, **kw)
                self.assertIn("not parsed", str(e.exception))
                rec()
                with self.assertRaises(SystemExit) as e:
                    T.build(t / "r4", gq, cq, protocol=fz, **dict(kw, expected={"gqa_testdev": 12578, "chartqa_test": 8}))
                self.assertIn("12578 expected", str(e.exception)); self.assertFalse(T.Layout(t / "r4").labels_file("gqa_testdev").exists())
                s = T.build(t / "r5", gq, cq, protocol=fz, **kw)
                self.assertTrue(s["benches"]["gqa_testdev"]["sha256_matches_record"]); self.assertEqual(s["freeze"]["freeze_key"], "r7_50_freeze_20261013")
                self.assertTrue(s["benches"]["chartqa_test"]["mode"].startswith("official test"))
                self.assertEqual(s["benches"]["gqa_testdev"]["exposure_check"]["questions_seen_in_mix_gqa_turns"], 1)
            finally:
                T.GQA_IMAGES = saved

    def test_dev_mode_refuses_test_names_and_bad_chart_rows(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); g, c, _, _ = synth_sources(t); tn = t / "src" / "my_testdev_questions.json"; tn.write_bytes(g.read_bytes()); os.chmod(tn, 0)
            with self.assertRaises(SystemExit) as e:
                T.build(t / "r0", tn, c, images_dir=t / "images")
            self.assertIn("test-type", str(e.exception))
            bad = [{"imgname": "a.png", "query": "q?", "label": "1", "type": "other", "image": png(1)}]
            pq.write_table(pa.Table.from_pylist(bad), t / "src" / "bad.parquet")
            with self.assertRaises(SystemExit):
                T.build(t / "r1", g, t / "src" / "bad.parquet", images_dir=t / "images")
            clash = [{"imgname": "a.png", "query": "q?", "label": "1", "type": "human_x", "image": png(1)},
                     {"imgname": "a.png", "query": "r?", "label": "2", "type": "human_x", "image": png(2)}]
            pq.write_table(pa.Table.from_pylist(clash), t / "src" / "clash.parquet")
            with self.assertRaises(SystemExit) as e:
                T.build(t / "r2", g, t / "src" / "clash.parquet", images_dir=t / "images")
            self.assertIn("different image bytes", str(e.exception))


def make_selectors(d: Path, choice=LR, extra_units=()):
    d.mkdir(parents=True, exist_ok=True); (d / "lr_choice.json").write_text(json.dumps(choice)); main, abl = RUN.unit_names(choice)
    for u in [u for o in RUN.OBJECTIVES for u in main[o]] + list(extra_units):
        (d / f"{u}.pt").write_bytes(b"ckpt-" + u.encode())
        (d / f"{u}.json").write_text(json.dumps({"selected_step": 1000, "dynamic": {str(f): {"tau": 0.1 * f} for f in RUN.FRACTIONS}}))
    (d / "pca.pt").write_bytes(b"pca")
    return main


class Rows(unittest.TestCase):
    def test_a2_rows_from_the_frozen_grid(self):
        with tempfile.TemporaryDirectory() as t:
            PL.write_configs(LR, Path(t)); main, _ = RUN.unit_names(LR)
            for b, d in T.D30_BENCH.items():
                grid = T.read_rows(Path(t) / f"configs_{d}.txt"); rows = T.a2_rows(grid, main)
                self.assertEqual(len(rows), 50); self.assertEqual(rows[0], "full")
                self.assertEqual(sum(r.startswith("static:grid:") for r in rows), 4); self.assertEqual(sum(r.startswith("sel:") for r in rows), 36)
                self.assertEqual(sorted(r for r in rows if r.startswith("dyn:")), sorted(f"dyn:{u}:{f}" for u in main["vcs"] for f in RUN.FRACTIONS))
                self.assertFalse(any(x in r for r in rows for x in ("random", "qsim", "_noq", "_dtarget", "full_ocr")))
                self.assertEqual(rows, [g for g in grid if g in set(rows)])                  # D1-30 order kept
                f_rows = T.phase_rows(grid, main, "F"); self.assertEqual(len(f_rows), 45); self.assertTrue(all(r.startswith(("sel:", "dyn:")) for r in f_rows))
                sv = T.sparsevlm_rows(b); self.assertEqual(len(sv), 8); self.assertFalse(any(r.endswith(":ocr") for r in sv))
                self.assertEqual(sum(r.endswith(":official") for r in sv), 4)

    def test_real_lr_choice_if_present(self):
        if not T.LR_CHOICE.is_file():
            self.skipTest("D1-20 lr_choice.json not present")
        choice = json.loads(T.LR_CHOICE.read_text()); main, _ = RUN.unit_names(choice)
        with tempfile.TemporaryDirectory() as t:
            PL.write_configs(choice, Path(t)); rows = T.a2_rows(T.read_rows(Path(t) / "configs_gqa_dev.txt"), main)
            self.assertEqual(len(rows), 50); self.assertIn(f"dyn:{main['vcs'][2]}:0.75", rows); self.assertIn(f"sel:{main['direct'][1]}:72", rows)

    def test_configs_and_lines_phases(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); main = make_selectors(t / "sel", extra_units=["x"]); d30 = t / "d30cfg"; PL.write_configs(LR, d30)
            r = T.configs(t / "root", "L", t / "sel" / "lr_choice.json", t / "sel", t / "sel" / "pca.pt", d30)
            self.assertEqual(r, {b: {"pipeline_rows": 50, "sparsevlm_rows": 8} for b in T.BENCHES})
            rec = json.loads((t / "root" / "configs_L" / "rows.json").read_text()); self.assertEqual(len(rec["units"]), 9)
            self.assertEqual(rec["benches"]["gqa_testdev"]["d1_30_driver_configs"]["file"], str(d30 / "configs_gqa_dev.txt")); self.assertEqual(rec["selectors_dir"], str(t / "sel"))
            self.assertEqual(rec["units"][main["vcs"][1]]["seed"], 1); self.assertEqual(rec["benches"]["chartqa_test"]["rows"]["sparsevlm:72:adapted"]["label"], "adapted")
            mf = make_selectors(t / "self", LR_F)                                           # phase F: D1-FULL-like units
            rf = T.configs(t / "root", "F", t / "self" / "lr_choice.json", t / "self", t / "self" / "pca.pt", d30)
            self.assertEqual(rf, {b: {"pipeline_rows": 45, "sparsevlm_rows": 0} for b in T.BENCHES})
            recf = json.loads((t / "root" / "configs_F" / "rows.json").read_text()); self.assertIsNone(recf["benches"]["gqa_testdev"]["d1_30_driver_configs"])
            self.assertEqual(len(recf["benches"]["gqa_testdev"]["baseline_rows_from_phase_L"]), 13); self.assertIn(f"dyn:{mf['vcs'][0]}:0.5", recf["benches"]["chartqa_test"]["rows"])
            with self.assertRaises(SystemExit):
                T.configs(t / "root", "Q")
            (d30 / "configs_gqa_dev.txt").write_text("full\n")
            with self.assertRaises(SystemExit):                                              # not a subset of the D1-30 driver's grid
                T.configs(t / "root", "L", t / "sel" / "lr_choice.json", t / "sel", t / "sel" / "pca.pt", d30)
            (t / "sel" / f"{main['direct'][2]}.json").write_text("{}")
            with self.assertRaises(SystemExit):                                              # D1-20 incomplete: no CAL taus
                T.configs(t / "root2", "L", t / "sel" / "lr_choice.json", t / "sel", t / "sel" / "pca.pt", None)
            L = T.lines(Path("/x/root"), "L"); self.assertEqual(len(L), 6 + 3 + 2 + 1)
            F = T.lines(Path("/x/root"), "F"); self.assertEqual(len(F), 6 + 2); self.assertTrue(all(l.endswith(" /x/root F") and "pred_F" in str(m) for l, m in F))
            for line, mark in L + F:
                self.assertIn("--partition=H100,L40S", line); self.assertIn("--exclude=node51,node52,node60", line)
                self.assertNotIn("--export", line); self.assertNotIn("RTX6000PRO", line); self.assertTrue(str(mark).startswith("/x/root/sealed/pred_"))
            self.assertEqual(sum("run gqa_testdev" in l for l, _ in L), 6); self.assertEqual(sum("sparsevlm chartqa_test 0/1" in l for l, _ in L), 1)
            self.assertIn(Path("/x/root/sealed/pred_L/gqa_testdev/done_shard5of6.json"), [m for _, m in L])
            self.assertIn(Path("/x/root/sealed/pred_L/gqa_testdev/sparsevlm_run_gqa_testdev_shard2of3.json"), [m for _, m in L])
            self.assertFalse(any(("gqa" in l and "chartqa" in l) for l, _ in L + F))        # GQA-only / ChartQA-only lines


class FakeHost:
    def __init__(self):
        import torch
        self.torch, self.max_new = torch, set()

    def encode(self, images):
        t = self.torch.zeros(len(images), 576, 8); t[:, :, 0] = 1.0; return t

    def segments(self, prompt, with_image=True):
        ids = self.torch.tensor([5, 6, 7]); return ids[:1], ids[1:]

    def embed_ids(self, ids):
        e = self.torch.zeros(len(ids), 8); e[:, 2] = 1.0; return e

    def question_vector(self, q):
        return self.torch.tensor([0.0, 1.0] + [0.0] * 6)

    def generate(self, seqs, max_new_tokens=16, min_new_tokens=0):
        self.max_new.add(max_new_tokens); return [f"v{int(s[:, 0].sum())}" for s in seqs]


class Generation(unittest.TestCase):
    def test_guards_and_runner(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); g, c, _, _ = synth_sources(t)
            with self.assertRaises(SystemExit):
                T.run_quality(t / "root", "gqa_testdev", "0/1", host=FakeHost())            # not built
            T.build(t / "root", g, c, images_dir=t / "images"); L = T.Layout(t / "root")
            for b in T.BENCHES:
                T.write_atomic(L.configs_file(b), "full\nstatic:grid:72\n"); T.write_atomic(L.sv_rows_file(b), "\n".join(T.sparsevlm_rows(b)) + "\n")
            for b, n in (("gqa_testdev", 128), ("chartqa_test", 16)):
                for sh in ("0/2", "1/2"):
                    h = FakeHost(); T.run_quality(t / "root", b, sh, host=h); self.assertEqual(h.max_new, {n})
                files = sorted(p.name for p in (L.pred / b).glob("*.jsonl"))
                self.assertEqual(files, ["full@s0of2.jsonl", "full@s1of2.jsonl", "static__grid__72@s0of2.jsonl", "static__grid__72@s1of2.jsonl"])
                self.assertTrue((L.pred / b / "done_shard1of2.json").exists())
            alt = t / "alt.txt"; alt.write_text("static:grid:144\n"); T.run_quality(t / "root", "gqa_testdev", "0/1", host=FakeHost(), configs_file=alt)
            self.assertTrue((L.pred / "gqa_testdev" / "static__grid__144@s0of1.jsonl").exists())   # a configs-file override (integrity groups)
            self.assertEqual(RUN.MAX_NEW["gqa_testdev"], RUN.MAX_NEW["gqa_dev"]); self.assertEqual(RUN.MAX_NEW["chartqa_test"], RUN.MAX_NEW["chartqa_val"])
            with self.assertRaises(SystemExit):
                T.run_sparsevlm(t / "root", ["gqa_testdev"], "0/1", rows=[192])               # rows differ from the sealed row file
            with self.assertRaises(SystemExit):
                T.run_sparsevlm(t / "root", ["gqa_testdev"], "0/1", phase="F")                # SparseVLM rows: phase L only
            L.manifest.parent.mkdir(parents=True); L.manifest.write_text("{}")
            with self.assertRaises(SystemExit):
                T.run_quality(t / "root", "gqa_testdev", "0/2", host=FakeHost())            # sealed: no generation after the manifest
            with self.assertRaises(SystemExit):
                T.run_sparsevlm(t / "root", ["gqa_testdev"], "0/1")


def write_preds(d: Path, items, cfg, answer, kept, shards=2):
    d.mkdir(parents=True, exist_ok=True); rows = [{"qid": it["qid"], "image_id": it["image_id"], "pred": answer(it), "kept": kept(i)} for i, it in enumerate(items)]
    k = len(rows) // shards + 1
    for s in range(shards):
        (d / f"{RUN.cfg_file(cfg)}@s{s}of{shards}.jsonl").write_text(T.jsonl(rows[s * k:(s + 1) * k]))


def synth_predictions(L: T.Layout, rows_of, seed=0, full_correct=True):
    """Rows of a phase for both benchmarks (two shard files per row); accuracy rising with the budget."""
    g = np.random.default_rng(seed); acc = {72: 0.4, 144: 0.55, 288: 0.7, 432: 0.8, 576: 0.9}
    lab = {b: {r["qid"]: r for r in T.read_jsonl_strict(L.labels_file(b))} for b in T.BENCHES}
    for b in T.BENCHES:
        items = T.read_jsonl_strict(L.inputs_file(b))
        for c in rows_of(b):
            p = c.split(":"); k = 576 if c == "full" else (int(p[1]) if p[0] == "sparsevlm" else (int(p[2]) if p[0] in ("static", "sel") else None))
            a = 1.0 if (c == "full" and full_correct) else None
            write_preds(L.pred / b, items, c, lambda it: lab[b][it["qid"]]["answer"] if g.uniform() < (a or acc.get(k or 288, 0.6)) else "wrong",
                        lambda i: k if k is not None else (144 if i % 2 else 576))
        if L.phase == "L":
            (L.pred / b / f"sparsevlm_run_{b}_shard0of1.json").write_text(json.dumps({"rows": [{"config": "sparsevlm:432:adapted", "official_forwards": 9,
                "predicted_equal_tokens": 432.21875, "equal_visual_tokens": 432.21875, "counter_deviation": 0.0, "counter_check": True,
                "deviation_is_float32_accumulation": True}], "rows_without_counter_record": []}))
    (L.pred / "hub_links" / "1_2").mkdir(parents=True, exist_ok=True)
    if not (L.pred / "hub_links" / "1_2" / "clip").exists():
        (L.pred / "hub_links" / "1_2" / "clip").symlink_to(L.root, target_is_directory=True)


def synth_timing(tdir: Path, main, tbench):
    lat = {72: 40.0, 144: 55.0, 288: 70.0, 432: 85.0}; rows = []
    for B in (1, 4, 16):
        rows.append({"config": "full", "B": B, "per_item_ms_median": 100.0, "per_item_ms_median_ci95": [99.0, 101.0]})
        for k in lat:
            rows.append({"config": f"static:grid:{k}", "B": B, "per_item_ms_median": lat[k], "per_item_ms_median_ci95": [lat[k] - 1, lat[k] + 1]})
            rows.append({"config": f"sel:{main['vcs'][0]}:{k}", "B": B, "per_item_ms_median": lat[k] + 2.0})
        for o in RUN.OBJECTIVES:
            for u in main[o]:
                for f, v in zip(RUN.FRACTIONS, (60.0, 75.0, 90.0)):
                    rows.append({"config": f"dyn:{u}:{f}", "B": B, "per_item_ms_median": v, "budget_share": {"144": 0.5, "576": 0.5}, "mean_visual_tokens": 360.0})
    d = tdir / tbench; d.mkdir(parents=True, exist_ok=True); (d / f"timing_{tbench}.json").write_text(json.dumps({"rows": rows, "gpu": "NVIDIA H100 NVL"}))
    (d / f"sparsevlm_timing_{tbench}.json").write_text(json.dumps({"rows": [{"config": c, "B": 1, "per_item_ms_median": 45.0 + int(c.split(":")[1]) / 10}
                                                                            for c in T.sparsevlm_rows("gqa_testdev")]}))


class ManifestAndScore(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory(); t = Path(self.td.name); g, c, _, _ = synth_sources(t, n_img=10, q_per=4, n_chart=24)
        T.build(t / "root", g, c, images_dir=t / "images"); self.main = make_selectors(t / "sel"); self.mainF = make_selectors(t / "self", LR_F)
        T.configs(t / "root", "L", t / "sel" / "lr_choice.json", t / "sel", t / "sel" / "pca.pt", None)
        T.configs(t / "root", "F", t / "self" / "lr_choice.json", t / "self", t / "self" / "pca.pt", None)
        self.L, self.LF, self.t = T.Layout(t / "root", "L"), T.Layout(t / "root", "F"), t
        synth_predictions(self.L, lambda b: T.read_rows(self.L.configs_file(b)) + T.read_rows(self.L.sv_rows_file(b)))
        synth_predictions(self.LF, lambda b: T.read_rows(self.LF.configs_file(b)), seed=1)
        for tb in ("gqa_dev", "chartqa_val"):
            synth_timing(t / "timing", self.main, tb)

    def tearDown(self):
        self.td.cleanup()

    def test_manifest_coverage_and_hashes(self):
        L, t = self.L, self.t; victim = L.pred / "gqa_testdev" / "static__grid__144@s1of2.jsonl"; keep = victim.read_text()
        victim.write_text("\n".join(keep.splitlines()[:-1]) + "\n")
        with self.assertRaises(SystemExit) as e:
            T.manifest(L.root, "L", t / "timing")
        self.assertIn("coverage", str(e.exception)); self.assertFalse(L.manifest.exists())
        victim.write_text(keep)
        with self.assertRaises(SystemExit):
            T.manifest(L.root, "L", t / "nowhere")                                           # profiling-workload timing required
        digest, m = T.manifest(L.root, "L", t / "timing")
        self.assertEqual(digest, T.sha(L.manifest)); self.assertTrue(L.manifest_sha.read_text().startswith(digest)); self.assertEqual(m["phase"], "L")
        self.assertEqual(m["rows"]["gqa_testdev"]["n_rows"], 58); self.assertFalse(any("hub_links" in f["path"] for f in m["files"]))
        self.assertEqual(m["n_files"], 2 * (58 * 2 + 1)); mt = L.manifest.read_text()
        self.assertFalse(any(a in mt for a in ("zebra", "ansyes")))                         # labels never opened / copied by the manifest
        for tb in ("gqa_dev", "chartqa_val"):
            synth_timing(t / "ftiming", self.mainF, tb)
        dF, mF = T.manifest(L.root, "F", t / "timing", None, t / "ftiming"); self.assertEqual(mF["rows"]["chartqa_test"]["n_rows"], 45); self.assertEqual(mF["n_files"], 2 * 45 * 2)
        with self.assertRaises(SystemExit):
            T.manifest(L.root, "L", t / "timing")                                            # sealed once

    def test_gate_and_readout_phase_L(self):
        L, t = self.L, self.t; digest, _ = T.manifest(L.root, "L", t / "timing")
        sc = lambda out, **kw: T.score(L.root, "L", None, t / "timing", None, 200, out, **kw)
        with self.assertRaises(SystemExit) as e:
            sc(t / "o0")
        self.assertIn("absent", str(e.exception))
        L.allow.write_text("manifest_sha256=" + "0" * 64 + "\n")
        with self.assertRaises(SystemExit):
            sc(t / "o1")
        L.allow.write_text(f"# coordinator\nmanifest_sha256={digest}\ncommit=abc\n"); f = L.pred / "chartqa_test" / "full@s0of2.jsonl"; orig = f.read_bytes()
        f.write_bytes(orig + b"\n")
        with self.assertRaises(SystemExit):
            sc(t / "o2")
        f.write_bytes(orig); tf = t / "timing" / "gqa_dev" / "timing_gqa_dev.json"; tt = tf.read_text(); tf.write_text(tt + " ")
        with self.assertRaises(SystemExit):
            sc(t / "o3")                                                                     # latency inputs must equal the manifest's
        tf.write_text(tt)
        repo = t / "repo"; repo.mkdir(); g = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True).stdout.strip()
        g("init", "-q"); g("config", "user.email", "t@t"); g("config", "user.name", "t"); (repo / "m.json").write_bytes(L.manifest.read_bytes())
        g("add", "m.json"); g("commit", "-q", "-m", "m"); head = g("rev-parse", "HEAD"); saved, T.REPO = T.REPO, repo
        try:
            with self.assertRaises(SystemExit) as e:                                      # production rule: the manifest must be committed
                sc(t / "o5", require_commit=True)
            self.assertIn("do not hold the manifest", str(e.exception))
            (repo / "m.json").write_text("{}"); g("commit", "-qam", "changed"); bad = g("rev-parse", "HEAD")
            L.allow.write_text(f"manifest_sha256={digest}\ncommit={bad}\nmanifest_copy=m.json\n")
            with self.assertRaises(SystemExit):
                sc(t / "o6", require_commit=True)
            L.allow.write_text(f"manifest_sha256={digest}\ncommit={head}\nmanifest_copy=m.json\n"); res = sc(t / "o4", require_commit=True)
        finally:
            T.REPO = saved
        self.assertFalse(any((t / f"o{i}").exists() for i in (0, 1, 2, 3, 5, 6)))
        for b, r in res["benches"].items():
            self.assertEqual(len(r["rows"]), 58); self.assertAlmostEqual(r["rows"]["full"]["quality"], 100.0)
            prim = {k: v for k, v in r["contrasts"].items() if v.get("role") == "primary"}; pts = ("L25", "L50", "L75")
            self.assertEqual(sorted(prim), sorted([f"S1-S2@{p}" for p in pts] + [f"S1-S3@{p}" for p in pts] + [f"S1-SparseVLM@{p}[budget]" for p in pts]))
            for v in prim.values():
                self.assertEqual(v["holm_family_size"], 6); self.assertTrue(v["ci95"][0] - 1e-9 <= v["delta"] <= v["ci95"][1] + 1e-9)
                self.assertGreaterEqual(v["p_holm"], v["p"])
            self.assertEqual(r["contrasts"]["S1-grid@L50"]["role"], "secondary"); self.assertEqual(r["contrasts"]["S1-SparseVLM@L25[latency_b1]"]["role"], "secondary")
            self.assertIn("A3:S1_dyn-fixed@f0.5", r["contrasts"]); self.assertEqual(set(r["row_sources"].values()), {"phase L"})
            self.assertEqual(r["latency_points_ms"]["4"], {"L25": 55.0, "L50": 70.0, "L75": 85.0})
            self.assertEqual(r["rows"][f"sel:{self.main['logistic'][2]}:144"]["latency_profiling_workload_ms"]["B4"]["median"], 57.0)   # representative S1 unit
            self.assertEqual(set(r["rows"]["sparsevlm:144:adapted"]["latency_profiling_workload_ms"]), {"B1"})
            dyn = r["dynamic_rows_on_test"]; self.assertEqual(len(dyn), 9)
            for v in dyn.values():
                self.assertAlmostEqual(sum(v["kept_tokens_share_test"].values()), 1.0); self.assertEqual(v["kept_tokens_share_profiling_workload_B4"], {"144": 0.5, "576": 0.5})
            self.assertEqual(len(r["sparsevlm_counters"]), 1); self.assertEqual(r["integrity"]["files_per_config_max"], 2)
        self.assertEqual(set(res["benches"]["gqa_testdev"]["subsets"]), {"structural:query", "structural:verify"})
        ch = res["benches"]["chartqa_test"]["subsets"]; self.assertEqual(ch["human"]["n"], 12); self.assertEqual(ch["augmented"]["n"], 12)
        self.assertIn("S1-S2@L50", ch["human"]["contrasts_secondary"]); self.assertTrue(all(v["role"].startswith("secondary") for v in ch["human"]["contrasts_secondary"].values()))
        self.assertAlmostEqual(ch["mean_of_splits"]["full"], 100.0)
        self.assertIn("profiling workload", res["latency_label"]); self.assertEqual(len(res["exposure_statement"]), 3)
        md = (t / "o4" / "r7_50_results.md").read_text(); self.assertIn("Primary contrasts", md); self.assertIn("phase L", md)
        with self.assertRaises(SystemExit):
            sc(t / "o4")                                                                     # never overwrites a readout

    def test_phase_F_readout_uses_sealed_L_baselines(self):
        L, LF, t = self.L, self.LF, self.t
        with self.assertRaises(SystemExit):
            T.manifest(L.root, "F", t / "timing")                                            # F3: D1-FULL's own timing run is required
        for tb in ("gqa_dev", "chartqa_val"):
            synth_timing(t / "ftiming", self.mainF, tb)
        dF, mF = T.manifest(L.root, "F", t / "timing", None, t / "ftiming"); LF.allow.write_text(f"manifest_sha256={dF}\n")
        self.assertIn("extra_files", mF["profiling_timing"]["gqa_testdev"])
        scF = lambda out, **kw: T.score(L.root, "F", None, t / "timing", None, 200, out, extra_timing_dir=t / "ftiming", **kw)
        with self.assertRaises(SystemExit) as e:
            scF(t / "f0")                                                                    # phase L's baselines must be sealed first
        self.assertIn("phase L predictions not sealed", str(e.exception))
        T.manifest(L.root, "L", t / "timing"); res = scF(t / "f1")
        for b, r in res["benches"].items():
            self.assertEqual(len(r["rows"]), 45 + 5 + 8)
            self.assertEqual({c for c, s in r["row_sources"].items() if s == "phase L"}, set(T.BASELINE_FROM_L) | set(T.sparsevlm_rows(b)))
            prim = [v for v in r["contrasts"].values() if v.get("role") == "primary" and "p" in v]
            self.assertEqual(len(prim), 9); self.assertTrue(all(v["holm_family_size"] == 6 for v in prim))
            self.assertEqual(r["rows"][f"dyn:{self.mainF['vcs'][1]}:0.5"]["latency_profiling_workload_ms"]["B4"]["median"], 75.0)   # D1-FULL timing run
            self.assertIsNotNone(r["at_latency"]["4"]["S1_dyn"]["L75"])
            self.assertEqual(r["rows"][f"sel:{self.mainF['direct'][0]}:288"]["latency_profiling_workload_ms"]["B4"]["median"], 72.0)   # 10k representative unit
        self.assertEqual(res["phase"], "F"); self.assertEqual(res["representative_latency_unit"], self.main["vcs"][0])
        with self.assertRaises(SystemExit):
            T.score(L.root, "F", None, t / "timing", None, 200, t / "f3")                    # F3: no scoring of phase F without D1-FULL's timing
        f = L.pred / "gqa_testdev" / "full@s0of2.jsonl"; f.write_bytes(f.read_bytes() + b"\n")
        with self.assertRaises(SystemExit):
            scF(t / "f2")                                                                    # a changed phase-L baseline file is refused


def synth_reference(t: Path, main, gpu_b=("NVIDIA L40S", "NVIDIA H100 NVL")):
    """A D1-30-like reference: sets (DEV GQA items with the synthetic GQA images, ChartQA val items with extracted charts) and predictions /
    markers for full, grid-144 (shard 0/1), S1 seed 0 at 144 (D1-30 shard counts) and SparseVLM-192."""
    g, c, gq, ch = synth_sources(t); sets = t / "ref" / "sets"; pred = t / "ref" / "pred"; sets.mkdir(parents=True)
    gi = [{"bench": "gqa_dev", "qid": q, "image_id": r["imageId"], "image_file": str(t / "images" / f"{r['imageId']}.jpg"), "question": r["question"],
           "answer": r["answer"], "prompt": r["question"] + DD.SHORT} for q, r in gq.items()]
    (sets / "gqa_dev.jsonl").write_text(T.jsonl(gi)); paths = {}
    for i, r in enumerate(ch):
        DD.write_image("chartqa_val", r["imgname"], r["image"], sets, paths)
    ci = [{"bench": "chartqa_val", "qid": f"chartqa_val_{i:05d}", "image_id": r["imgname"], "image_file": paths[r["imgname"]], "question": r["query"],
           "answer": r["label"], "prompt": r["query"] + DD.CHART_SUFFIX} for i, r in enumerate(ch)]
    (sets / "chartqa_val.jsonl").write_text(T.jsonl(ci))
    for d30b, items in (("gqa_dev", gi), ("chartqa_val", ci)):
        d = pred / d30b; d.mkdir(parents=True)
        for cfg, k in (("full", 576), ("static:grid:144", 144), (f"sel:{main['vcs'][0]}:144", 144), ("sparsevlm:192:official", 192)):
            (d / f"{RUN.cfg_file(cfg)}@s0of1.jsonl").write_text(T.jsonl({"qid": it["qid"], "image_id": it["image_id"], "pred": f"a{j % 3}", "kept": k}
                                                                        for j, it in enumerate(items)))
        (d / "done_shard0of1.json").write_text(json.dumps({"env": {"gpu": "NVIDIA L40S"}}))
        n = PL.SHARDS[d30b]
        for i in range(n):
            (d / f"done_shard{i}of{n}.json").write_text(json.dumps({"env": {"gpu": gpu_b[i % 2]}}))
    (t / "ref" / "d30.log").write_text(f"x RETRY 1/2: job 7 left the queue without {pred / 'gqa_dev' / 'done_shard1of6.json'}\n")
    return g, c, sets, pred, gi, ci


class IntegrityGate(unittest.TestCase):
    def test_plan_pins_and_compare(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); main = make_selectors(t / "sel"); _, c, sets, pred, gi, ci = synth_reference(t, main)
            u = T.integrity_prepare(t / "root", "L", t / "sel" / "lr_choice.json", t / "sel", t / "sel" / "pca.pt", pred, sets, t / "ref" / "d30.log",
                                    chartqa_src=c, images_dir=t / "images")
            self.assertEqual(u, {"pipe_L40S": 2 + 3 + 1, "pipe_H100": 3 + 1, "sv_L40S": 2})
            base = T.Layout(t / "root").integrity; plan = json.loads((base / "plan.json").read_text())
            self.assertEqual(plan["groups"]["B"], [f"sel:{main['vcs'][0]}:144"]); self.assertEqual(plan["retried_reference_markers"], [str(pred / "gqa_dev" / "done_shard1of6.json")])
            self.assertTrue(any(x.get("reference_retried") for x in plan["units"]["pipe_H100"]["tasks"]))
            lines = (base / "lines.txt").read_text()
            self.assertIn("--partition=L40S", lines); self.assertIn("--partition=H100 ", lines); self.assertNotIn("--export", lines)
            self.assertEqual(lines.count("slurm/r7_50_gpu.sbatch integrity"), 3)
            IB = T.Layout(base, "L"); mine = T.read_jsonl_strict(IB.inputs_file("chartqa_test")); self.assertEqual(mine[0]["qid"], "chartqa_val_00000")
            for b, d30b, items in (("gqa_testdev", "gqa_dev", gi), ("chartqa_test", "chartqa_val", ci)):   # our side: copies of the reference answers
                for cfg in ("full", "static:grid:144", f"sel:{main['vcs'][0]}:144", "sparsevlm:192:official"):
                    src = pred / d30b / f"{RUN.cfg_file(cfg)}@s0of1.jsonl"; (IB.pred / b).mkdir(parents=True, exist_ok=True)
                    (IB.pred / b / f"{RUN.cfg_file(cfg)}@s0of1.jsonl").write_text(src.read_text())
                (IB.pred / b / f"sparsevlm_run_{b}_shard0of1.json").write_text(json.dumps({"rows": [{"config": "sparsevlm:192:official", "official_forwards": len(items),
                    "predicted_equal_tokens": 192.03125, "equal_visual_tokens": 192.03125, "counter_deviation": 0.0, "counter_check": True}]}))
            self.assertEqual(T.integrity_compare(t / "root", "L"), 0); res = json.loads((base / "integrity.json").read_text())
            self.assertTrue(res["pass"]); self.assertEqual(res["benches"]["gqa_testdev"]["inputs_identical"], {"question": 18, "prompt": 18, "image": 18})
            self.assertEqual(res["benches"]["chartqa_test"]["static:grid:144"]["identical"], 8)
            f = IB.pred / "chartqa_test" / f"{RUN.cfg_file('sparsevlm:192:official')}@s0of1.jsonl"; rows = T.read_jsonl_strict(f); rows[3]["pred"] = "flipped"
            f.write_text(T.jsonl(rows)); self.assertEqual(T.integrity_compare(t / "root", "L"), 1)
            res = json.loads((base / "integrity.json").read_text()); x = res["benches"]["chartqa_test"]["sparsevlm:192:official"]
            self.assertFalse(res["pass"]); self.assertEqual((x["identical"], x["mismatched_qids_first_20"]), (7, ["chartqa_val_00003"]))
            self.assertNotIn("flipped", (base / "integrity.json").read_text())             # no prediction text in the record
            saved, T.SV_AGREEMENT_MIN = T.SV_AGREEMENT_MIN, 0.85                            # SparseVLM rule: agreement threshold, not identity
            try:
                self.assertEqual(T.integrity_compare(t / "root", "L"), 0)                    # 7 / 8 = 0.875 >= 0.85 passes
                g = IB.pred / "gqa_testdev" / f"{RUN.cfg_file('static:grid:144')}@s0of1.jsonl"; rows = T.read_jsonl_strict(g); rows[0]["pred"] = "x"; g.write_text(T.jsonl(rows))
                self.assertEqual(T.integrity_compare(t / "root", "L"), 1)                    # a deterministic row: any mismatch fails
                rows[0]["pred"] = T.read_jsonl_strict(pred / "gqa_dev" / f"{RUN.cfg_file('static:grid:144')}@s0of1.jsonl")[0]["pred"]; g.write_text(T.jsonl(rows))
                m = IB.pred / "chartqa_test" / "sparsevlm_run_chartqa_test_shard0of1.json"; mm = json.loads(m.read_text()); mm["rows"][0]["counter_check"] = False
                m.write_text(json.dumps(mm)); self.assertEqual(T.integrity_compare(t / "root", "L"), 1)   # E3 counter outside tolerance fails
            finally:
                T.SV_AGREEMENT_MIN = saved
            self.assertEqual(T.SV_AGREEMENT_MIN, 0.98)


STUB_SBATCH = r'''#!/usr/bin/env bash
ST=$STUB_STATE; mkdir -p "$ST"; n=$(( $(cat "$ST/n" 2>/dev/null || echo 1000) + 1 )); echo $n > "$ST/n"; echo "$*" >> "$ST/calls"
[ "${STUB_NOID:-0}" = 1 ] && exit 0
args=("$@"); s=""; rest=()
for i in "${!args[@]}"; do case "${args[$i]}" in *.sbatch) s=${args[$i]}; rest=("${args[@]:$((i+1))}");; esac; done
case "$s" in
  *r7_50_cpu.sbatch)
    R=${rest[1]}; P=${rest[2]:-L}
    case "${rest[0]}" in
      prepare) mkdir -p "$R/sealed/inputs"; echo '{"ok": true}' > "$R/sealed/inputs/build_summary.json"; "$STUB_PY" -I scripts/r7_50_test.py lines --root "$R" --phase L > /dev/null;;
      configs) "$STUB_PY" -I scripts/r7_50_test.py lines --root "$R" --phase "$P" > /dev/null;;
      integrity-prepare) I="$R/integrity_$P"; mkdir -p "$I"; { echo "# stub"; for u in pipe_L40S sv_L40S; do
          echo "sbatch --parsable --job-name=r7_50i_$u --partition=L40S --exclude=node51,node52,node60 slurm/r7_50_gpu.sbatch integrity $R $P $u #MARK $I/done_$u.json"; done; } > "$I/lines.txt";;
      integrity-compare) I="$R/integrity_$P"; if [ "${STUB_INTEGRITY:-pass}" = pass ]; then echo '{"pass": true}' > "$I/integrity.json"; else echo '{"pass": false}' > "$I/integrity.json"; echo $n; exit 1; fi;;
      manifest) M="$R/manifest_$P"; mkdir -p "$M"; echo "{\"files\": [], \"phase\": \"$P\"}" > "$M/predictions_manifest.json"; h=$(sha256sum "$M/predictions_manifest.json" | cut -d' ' -f1)
                echo "$h  predictions_manifest.json" > "$M/predictions_manifest.sha256"
                case "${STUB_ALLOW:-none}" in good) printf '# test\nmanifest_sha256=%s\n' "$h" > "$R/ALLOW_SCORING_$P";; bad) echo "manifest_sha256=00" > "$R/ALLOW_SCORING_$P";; esac;;
      score) mkdir -p "$R/results_$P"; echo '{}' > "$R/results_$P/r7_50_results.json";;
    esac;;
  *r7_50_gpu.sbatch)
    what=${rest[0]}
    if [ "$what" = integrity ]; then R=${rest[1]}; P=${rest[2]}; u=${rest[3]}; key="integrity_$u"; m="$R/integrity_$P/done_$u.json"
    else b=${rest[1]}; sh=${rest[2]}; R=${rest[3]}; P=${rest[4]:-L}; i=${sh%/*}; k=${sh#*/}; key="${what}_${b}_${i}"
      if [ "$what" = run ]; then m="$R/sealed/pred_$P/$b/done_shard${i}of${k}.json"; else m="$R/sealed/pred_L/$b/sparsevlm_run_${b}_shard${i}of${k}.json"; fi; fi
    t=$(( $(cat "$ST/try_$key" 2>/dev/null || echo 0) + 1 )); echo $t > "$ST/try_$key"
    if [ "$key" = "${STUB_FAIL_UNIT:-}" ] && [ "$t" -le "${STUB_FAIL_TIMES:-0}" ]; then :; else mkdir -p "$(dirname "$m")"; echo '{}' > "$m"; fi;;
esac
echo $n
'''


class DriverDryRun(unittest.TestCase):
    """slurm/r7_50_submit.sh with stubbed sbatch / squeue (R750_DRYRUN=1, temporary roots; nothing is submitted)."""

    def env(self, t: Path, **kw):
        no_d30_markers = kw.pop("_no_d30_markers", None)                              # execution note r7_50_1: the gate waits for these shards
        b = t / "bin"; b.mkdir(exist_ok=True); (b / "sbatch").write_text(STUB_SBATCH); (b / "squeue").write_text("#!/usr/bin/env bash\nexit 0\n")
        for f in ("sbatch", "squeue"):
            os.chmod(b / f, 0o755)
        d1 = t / "d1"; make_selectors(d1 / "selectors", extra_units=[u for v in RUN.unit_names(LR)[1].values() for u in v]); make_selectors(t / "full_sel", LR_F)
        for tb in ("gqa_dev", "chartqa_val"):
            (d1 / "d1_30" / "timing" / tb).mkdir(parents=True, exist_ok=True)
            for f in (f"timing_{tb}.json", f"sparsevlm_timing_{tb}.json"):
                (d1 / "d1_30" / "timing" / tb / f).write_text("{}")
        if not no_d30_markers:
            for tb, n, nsv in (("gqa_dev", 6, 3), ("chartqa_val", 2, 1)):
                pd = d1 / "d1_30" / "pred" / tb; pd.mkdir(parents=True, exist_ok=True)
                for i in range(n):
                    (pd / f"done_shard{i}of{n}.json").write_text(json.dumps({"env": {"gpu": "NVIDIA L40S"}}))
                for i in range(nsv):
                    (pd / f"sparsevlm_run_{tb}_shard{i}of{nsv}.json").write_text(json.dumps({"env": {"gpu": "NVIDIA L40S"}}))
        (t / "d30.log").write_text("x stage 1 complete (23 units)\n"); (t / "proto.yaml").write_text(FROZEN)
        e = dict(os.environ, PATH=f"{b}:{os.environ['PATH']}", STUB_STATE=str(t / "state"), STUB_PY=sys.executable, PY=sys.executable, R750_DRYRUN="1",
                 R750_ROOT=str(t / "root"), D1_ROOT=str(d1), D30_LOG=str(t / "d30.log"), SUBMIT_LOG=str(t / "drv.log"), PROTOCOL=str(t / "proto.yaml"),
                 POLL="1", WAIT_H="1", WAIT_SCORE_H="1", RETRIES="2", STUB_ALLOW="good", F_LR_CHOICE=str(t / "full_sel" / "lr_choice.json"),
                 F_SELECTORS=str(t / "full_sel"), F_PCA=str(t / "full_sel" / "pca.pt"), F_INTEGRITY_REF=str(t / "fref"), F_TIMING_DIR=str(t / "ftiming"))
        for tb in ("gqa_dev", "chartqa_val"):
            (t / "ftiming" / tb).mkdir(parents=True, exist_ok=True); (t / "ftiming" / tb / f"timing_{tb}.json").write_text("{}")
        e.update({k: str(v) for k, v in kw.items()})
        return e

    def drive(self, t, **kw):
        p = subprocess.run(["bash", str(R / "slurm" / "r7_50_submit.sh")], env=self.env(t, **kw), cwd=str(R), capture_output=True, text=True, timeout=180)
        log = (t / "drv.log").read_text() if (t / "drv.log").exists() else ""; calls = (t / "state" / "calls").read_text() if (t / "state" / "calls").exists() else ""
        return p.returncode, log, calls

    def test_phase_L_full_pass_with_integrity_and_one_retry(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); rc, log, calls = self.drive(t, STUB_FAIL_UNIT="run_chartqa_test_1", STUB_FAIL_TIMES=1)
            self.assertEqual(rc, 0, log); self.assertIn("RETRY 1/2", log); self.assertIn("generation L complete (12 units)", log); self.assertIn("done:", log)
            self.assertIn("integrity complete (2 units)", log); self.assertIn("integrity gate passed", log)
            self.assertLess(calls.index("integrity-compare"), calls.index("r7_50_cpu.sbatch prepare"))   # the gate precedes the one test read
            self.assertEqual(calls.count("r7_50_cpu.sbatch prepare"), 1); self.assertEqual(calls.count("r7_50_cpu.sbatch score"), 1)
            self.assertEqual(calls.count("r7_50_gpu.sbatch run") + calls.count("r7_50_gpu.sbatch sparsevlm"), 13); self.assertNotIn("--export", calls)
            self.assertIn("D1-20 complete: 15 units", log); self.assertLess(log.index("stage 2 complete"), log.index("stage 3: ALLOW_SCORING_L present"))
            self.assertTrue(all("--exclude=node51,node52,node60" in c for c in calls.splitlines() if "r7_50_gpu" in c))

    def test_phase_F_and_Q(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); rc, log, _ = self.drive(t)                                        # phase L first (builds the sealed inputs)
            self.assertEqual(rc, 0, log); (t / "drv.log").unlink(); (t / "state" / "calls").unlink()
            rc, log, calls = self.drive(t, PHASE="F")
            self.assertEqual(rc, 0, log); self.assertIn("integrity gate passed", log); self.assertIn("r7_50_cpu.sbatch integrity-prepare", calls)
            self.assertIn("D1-FULL complete: 9 units", log); self.assertIn("generation F complete (8 units)", log); self.assertNotIn("r7_50_cpu.sbatch prepare", calls)
            runs = [c for c in calls.splitlines() if "r7_50_gpu.sbatch run" in c]; self.assertEqual(len(runs), 8); self.assertTrue(all(c.endswith(" F") for c in runs))
            self.assertNotIn("sparsevlm", calls); self.assertIn("r7_50_cpu.sbatch configs", calls); self.assertIn("r7_50_cpu.sbatch score", calls)
            self.assertTrue((t / "root" / "results_F" / "r7_50_results.json").exists())
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); rc, log, calls = self.drive(t, PHASE="F")
            self.assertEqual(rc, 1); self.assertIn("phase F needs phase L's sealed inputs", log); self.assertEqual(calls, "")
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); rc, log, calls = self.drive(t, PHASE="Q"); self.assertEqual(rc, 1); self.assertIn("phase Q", log); self.assertEqual(calls, "")

    def test_phase_F_timing_gated_before_manifest(self):
        """Execution note r7_50_8: D1-FULL's timing is gated just before phase F's manifest (it is needed by the manifest and the scorer), so
        the integrity gate and the 8 generation units run first; a missing timing run stops before any manifest or label is touched."""
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); rc, log, _ = self.drive(t)                                        # phase L first (builds the sealed inputs)
            self.assertEqual(rc, 0, log); (t / "drv.log").unlink(); (t / "state" / "calls").unlink()
            rc, log, calls = self.drive(t, PHASE="F", F_TIMING_DIR="/nonexistent", WAIT_H="0")
            self.assertEqual(rc, 1, log); self.assertIn("STOP", log); self.assertIn("timing_gqa_dev.json missing", log)
            self.assertIn("integrity gate passed", log); self.assertIn("generation F complete (8 units)", log)
            self.assertNotIn("r7_50_cpu.sbatch manifest", calls); self.assertNotIn("r7_50_cpu.sbatch score", calls)

    def test_stops(self):
        cases = [(dict(PROTOCOL="/nonexistent.yaml"), "not a frozen R7-50 protocol", ""),
                 (dict(R750_DRYRUN="0"), "not committed", ""),
                 (dict(R750_ROOT=str(T.ROOT)), "refuses the production root", ""),
                 (dict(WAIT_H="0", _no_d30_markers="1"), "done_shard0of6.json missing", ""),
                 (dict(STUB_NOID="1"), "no job id", ""),
                 (dict(STUB_INTEGRITY="fail"), "full-scale integrity gate FAILED", "integrity-compare"),
                 (dict(STUB_FAIL_UNIT="integrity_sv_L40S", STUB_FAIL_TIMES=99), "integrity unit failed 3 times", "integrity-prepare"),
                 (dict(STUB_FAIL_UNIT="sparsevlm_gqa_testdev_2", STUB_FAIL_TIMES=99), "unit failed 3 times", "prepare"),
                 (dict(STUB_ALLOW="bad"), "does not carry manifest_sha256", "manifest"),
                 (dict(STUB_ALLOW="none", WAIT_SCORE_H="0"), "ALLOW_SCORING_L after 0 h", "manifest"),
                 (dict(PHASE="F", F_LR_CHOICE=""), "phase F needs F_LR_CHOICE", ""),
                 (dict(PHASE="F", F_INTEGRITY_REF="none"), "F_INTEGRITY_REF=none is not allowed", "")]
        for kw, msg, last in cases:
            with tempfile.TemporaryDirectory() as t:
                t = Path(t); rc, log, calls = self.drive(t, **kw)
                self.assertEqual(rc, 1, (kw, log)); self.assertIn("STOP", log, (kw, log)); self.assertIn(msg, log, (kw, log))
                self.assertNotIn("r7_50_cpu.sbatch score", calls, kw)                       # labels never opened on a stop
                if kw.get("STUB_INTEGRITY") or "integrity_" in kw.get("STUB_FAIL_UNIT", ""):
                    self.assertNotIn("r7_50_cpu.sbatch prepare", calls)                       # no test file opened when the gate fails
                if kw.get("R750_ROOT") == str(T.ROOT):
                    self.assertEqual(calls, "")
                if last:
                    self.assertIn(f"r7_50_cpu.sbatch {last}", calls)


if __name__ == "__main__":
    unittest.main()
