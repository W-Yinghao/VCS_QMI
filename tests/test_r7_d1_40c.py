"""R7-40c (Qwen2.5-VL evaluation stage): configuration grammar and unit plan; the quality runner with a fake host (single-writer files,
markers, resume, answers never held, token-count ordering, header-estimate and missing-image stops); the selector bank against the frozen
R7-40b rules (hard top-k on valid tokens, risk_grid / dynamic rule, batched == per-item selection); HF generate with injected 3-D rotary ids
== QwenCore on pruned rows (tiny random Qwen2.5-VL; vl_baselines env); smart_resize == transformers'; timing rows (pools around the primary B,
bucketing by chosen fraction, reference-report checks); the scorer (freeze gate, coverage guard, latency / fraction primary bases, Holm over
9, A3, refusal to overwrite, cross-host side-by-side); DEV completeness and the DEV-risk summary; the driver (stubbed sbatch / squeue).  CPU."""
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts")); sys.path.insert(0, str(R / "r7" / "package_r7_20261011" / "reference"))
import r7_d1_30_run as RUN  # noqa: E402
import r7_d1_40_masks as M4  # noqa: E402
import r7_d1_40_selectors as S4  # noqa: E402
import r7_d1_40c_devrisk as DRK  # noqa: E402
import r7_d1_40c_plan as PL  # noqa: E402
import r7_d1_40c_run as RN  # noqa: E402
import r7_d1_40c_score as SCO  # noqa: E402
import r7_d1_40c_timing as TMG  # noqa: E402
import r7_d1_selectors as SEL  # noqa: E402

try:                                                                                   # Qwen2.5-VL classes (vl_baselines env; SparseVLM's 4.37 lacks them)
    from transformers import Qwen2_5_VLConfig  # noqa: F401
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize as _sr  # noqa: F401
    HAS_TF = True
except ImportError:
    HAS_TF = False
try:
    import llava.eval.m4c_evaluator  # noqa: F401
    HAS_LLAVA = True
except Exception:
    HAS_LLAVA = False

LR = {"vcs": {"chosen_lr": 3e-4}, "logistic": {"chosen_lr": 1e-4}, "direct": {"chosen_lr": 1e-4}}


# ------------------------------------------------------------------------------------------------------------------------------ plan
class Plan(unittest.TestCase):
    def test_grammar_and_row_sets(self):
        main = PL.unit_names(LR)
        self.assertEqual(main["vcs"], ["vcs_lr0.0003_s0", "vcs_lr0.0003_s1", "vcs_lr0.0003_s2"]); self.assertEqual(main["direct"][2], "direct_lr0.0001_s2")
        n = {r: len(PL.config_list(main, r)) for r in PL.ROW_SETS}; self.assertEqual(n, {"core": 50, "static": 58, "dynamic": 68, "all": 76})
        core = PL.config_list(main); self.assertEqual(core[:5], ["full", "static:grid:0.125", "static:grid:0.25", "static:grid:0.5", "static:grid:0.75"])
        self.assertEqual(sum(c.startswith("dyn:vcs_") for c in core), 9); self.assertFalse(any(c.startswith(("dyn:logistic", "dyn:direct", "static:random", "static:qsim")) for c in core))
        self.assertEqual(len(set(PL.config_list(main, "all"))), 76)
        for c in PL.config_list(main, "all"):
            k, a, f = PL.parse(c); self.assertIn(k, ("full", "static", "sel", "dyn"))
        self.assertEqual(PL.parse("sel:vcs_lr0.0003_s1:0.125"), ("sel", "vcs_lr0.0003_s1", 0.125)); self.assertEqual(PL.parse("dyn:x:0.5"), ("dyn", "x", 0.5))
        for bad in ("static:grid:0.3", "dyn:u:0.125", "sel:u:1", "static:edge:0.25", "foo:a:0.25"):
            with self.assertRaises(ValueError):
                PL.parse(bad)
        tl = PL.timing_list(core, main["vcs"][0]); self.assertEqual(len(tl), 1 + 4 + 4 + 9); self.assertTrue(all(c.startswith(("full", "static:", f"sel:{main['vcs'][0]}:", "dyn:")) for c in tl))
        tags = PL.r7_40b_tags(LR); self.assertEqual(len(tags), 12); self.assertEqual(len(set(tags)), 12)
        self.assertIn("vcs_lr0.0001_s0", tags); self.assertIn("logistic_lr0.0003_s0", tags); self.assertIn("direct_lr0.0001_s2", tags); self.assertNotIn("vcs_lr0.0001_s1", tags)
        with tempfile.TemporaryDirectory() as t:
            res = PL.write_configs(LR, Path(t), "core"); self.assertEqual(res["gqa_dev"], {"configs": 50, "timing_configs": 18})
            self.assertEqual(PL.read_configs(Path(t) / "configs_chartqa_val.txt"), core)

    def test_lines(self):
        L = PL.lines(Path("/x"), r40=Path("/r40"), timing_report=Path("/r40/timing/timing_report.json"))
        self.assertEqual(len(L), 12 + 3 + 1 + 2 + 1)
        o = [x for x in L if "r7d40co_" in x[0]]; self.assertEqual(len(o), 3)                         # decision 15: one anchor unit per benchmark
        self.assertTrue(all("--partition=L40S " in x[0] and " official " in x[0] and x[1].name == "done_official_shard0of1.json" and not x[2] for x in o))
        self.assertEqual(len(PL.lines(Path("/x"), official=False)), 12 + 1 + 2 + 1)
        for line, mark, gates in L:
            self.assertTrue(line.startswith("sbatch --parsable ")); self.assertIn("--exclude=node51,node60", line); self.assertNotIn("node52", line)
            self.assertNotIn("--export", line); self.assertNotIn("--dependency", line); self.assertIn("--time=", line)
        q = [x for x in L if "r7d40cq_" in x[0]]; self.assertEqual(len(q), 12); self.assertTrue(all("--partition=L40S " in x[0] and not x[2] for x in q))
        self.assertEqual(sorted(str(m) for _, m, _ in q)[0], "/x/pred/chartqa_val/done_shard0of2.json")
        t = [x for x in L if "r7d40ct_" in x[0]]; self.assertEqual(len(t), 1)                          # ONE nodesumo01 job (owner rule 2026-10-13)
        self.assertIn("--partition=H100 --nodelist=nodesumo01", t[0][0]); self.assertTrue(t[0][0].endswith("timing_bundle report"))     # decision 14
        self.assertEqual(t[0][2], [Path("/r40/timing/timing_report.json")])
        self.assertEqual(t[0][1], Path("/x/timing/timing_bundle.done.json")); self.assertIn("--time=08:00:00", t[0][0])
        tr = [x for x in PL.lines(Path("/x"), bstar="in_bundle", timing_report=Path("/r40/timing/timing_report.json")) if "r7d40ct_" in x[0]]
        self.assertEqual(len(tr), 1); self.assertTrue(tr[0][0].endswith("timing_bundle in_bundle")); self.assertEqual(tr[0][2], [])
        with self.assertRaises(SystemExit):
            PL.lines(Path("/x"), bstar="twice")
        d = [x for x in L if "devteacher" in x[0]]; self.assertEqual([x[0].split()[-1] for x in d], ["b012", "b013"]); self.assertTrue(all("--partition=L40S " in x[0] for x in d))
        dr = [x for x in L if "devrisk" in x[0]][0]; self.assertEqual(dr[2], [Path("/r40/teacher/bundle_b012.done.json"), Path("/r40/teacher/bundle_b013.done.json")])
        self.assertEqual(len(PL.lines(Path("/x"), dev=False, timing=False, official=False)), 12)
        self.assertIn("--partition=RTX6000PRO,H100,L40S ", PL.lines(Path("/x"), "RTX6000PRO,H100,L40S")[0][0])
        for bad in ("A100", "L40S --export=X", "", "L40S;rm", "L40S node52"):
            with self.assertRaises(SystemExit):
                PL.lines(Path("/x"), bad)
        with tempfile.TemporaryDirectory() as tt:
            PL.write_lines(L, Path(tt) / "l.txt", "t"); txt = (Path(tt) / "l.txt").read_text().splitlines()
            self.assertTrue(txt[0].startswith("#")); self.assertTrue(all(" #MARK " in x for x in txt[1:])); self.assertEqual(sum(" #GATE " in x for x in txt), 2)


# ------------------------------------------------------------------------------------------------------------------------------ fakes
E = 8                                                                                   # fake host width (embeddings and merged tokens)


class FakeTok:
    def __call__(self, text, add_special_tokens=False):
        return Namespace(input_ids=[3 + (ord(c) % 20) for c in text[:6]] or [3])


class FakeCore:
    merge, image_token_id, eos_ids, pad_id, dev = 2, 99, [7, 8], 8, "cpu"

    def __init__(self):
        self.torch, self.gen_calls, self.rows = torch, 0, 0

    def embed_ids(self, ids):
        ids = torch.as_tensor(np.asarray(ids), dtype=torch.long); e = torch.zeros(len(ids), E); e[:, 2] = 1.0; e[:, 3] = ids.float() / 50
        return e

    def full_positions(self, pre, n_vis, post, thw):
        rows, cols = int(thw[1]) // 2, int(thw[2]) // 2; s = len(pre); r, c = np.divmod(np.arange(n_vis), cols)
        return torch.as_tensor(np.concatenate([np.tile(np.arange(s), (3, 1)), np.stack([np.full(n_vis, s), s + r, s + c]),
                                               np.tile(s + max(rows, cols) + np.arange(len(post)), (3, 1))], 1))

    def generate(self, seqs, positions, prompt_ids, max_new_tokens=128, min_new_tokens=0):
        self.gen_calls += 1; self.rows += len(seqs); out = []
        for s, p, pid in zip(seqs, positions, prompt_ids):
            assert tuple(p.shape) == (3, s.shape[0]) and len(pid) == s.shape[0]
            assert int((np.asarray(pid) == self.image_token_id).sum()) == int(s[:, 0].sum())      # one image id per visual token
            out.append([100 + int(s[:, 0].sum()), 7] if max_new_tokens > 1 else [100])
        return out, None

    def last_logits(self, seqs, positions):
        return torch.zeros(len(seqs), 4)


class FakeHost:
    """CPU stand-in for QwenPrefill: images are dicts {'grid': (rows, cols)}; merged token i of an image has dim 0 = 1 (visual flag) and a
    position-dependent pattern; answers are 'v<number of visual tokens seen>'."""
    image_token_id, load_seconds, info, stage_notes = 99, 0.0, {}, []

    def __init__(self):
        self.torch, self.core, self.tok, self.prompts = torch, FakeCore(), FakeTok(), []

    def segments(self, prompt, with_image=True):
        self.prompts.append(prompt); ids = np.full(3 + len(prompt) % 5, 5)
        return ids[:2], ids[2:]

    def question_vector(self, q):
        v = torch.zeros(E); v[1] = 1.0; v[4] = (len(q) % 7) / 7
        return v

    def encode(self, images):
        out = []
        for im in images:
            rows, cols = im["grid"]; n = rows * cols; g = torch.Generator().manual_seed(rows * 100 + cols)
            f = torch.randn(n, E, generator=g); f[:, 0] = 1.0; out.append((f, (rows, cols), torch.tensor([1, 2 * rows, 2 * cols])))
        return out

    def decode(self, toks):
        return [f"v{t[0] - 100}" for t in toks]


GRIDS = {"img0": (3, 4), "img1": (4, 4), "img2": (2, 5), "img3": (4, 5), "img4": (5, 6)}


def tiny_sets(d: Path, bench="gqa_dev", n=13):
    d.mkdir(parents=True, exist_ok=True); rows = []
    for i in range(n):
        img = f"img{i % 5}"
        rows.append({"bench": bench, "qid": f"q{i:02d}", "image_id": img, "image_file": img, "question": f"Is thing {i} here?", "prompt": f"Is thing {i} here?\nShort",
                     "answer": "v12" if i % 2 else "no", "answers": ["v12"] * 10, "structural": "verify" if i % 3 else "query", "type": "human_test" if i % 2 else "augmented_test"})
    (d / f"{bench}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def tiny_units(d: Path, objectives=("vcs", "logistic", "direct"), seeds=(0, 1, 2), lr_choice=LR, taus=True):
    """Tiny real Selector40 checkpoints (emb E, vis 4) + PCA [E, 4] + R7-40b-style logs with dynamic taus."""
    d.mkdir(parents=True, exist_ok=True); torch.manual_seed(0)
    cfg = S4.Cfg40(emb_dim=E, vis_dim=4, d=16, heads=2, layers=1, ffn=32, dropout=0.0, max_rows=8, max_cols=8)
    for o in objectives:
        for s in seeds:
            u = PL.unit_tag(o, lr_choice[o]["chosen_lr"], s); m = S4.Selector40(cfg, o)
            torch.save({"state_dict": m.state_dict(), "cfg": asdict(cfg), "objective": o, "host": "Qwen2.5-VL-7B"}, d / f"{u}.pt")
            if taus:
                (d / f"{u}.json").write_text(json.dumps({"dynamic": {"0.25": {"tau": 0.002}, "0.5": {"tau": 0.0005}, "0.75": {"tau": 0.0001}}}))
    g = torch.Generator().manual_seed(1); torch.save({"mean": torch.zeros(E), "basis": torch.randn(E, 4, generator=g)}, d / "pca.pt")
    (d / "lr_choice.json").write_text(json.dumps(lr_choice))
    return cfg


def run_args(t: Path, cf: Path, bench="gqa_dev", shard="0/1", chunk=4, batch=3):
    return Namespace(sets_dir=str(t / "sets"), bench=bench, configs_file=str(cf), shard=shard, out=str(t / "out"), chunk=chunk, batch=batch, limit=0,
                     selectors_dir=str(t / "sel"), pca=str(t / "sel" / "pca.pt"))


def bank_for(t: Path, units):
    return RN.QwenSelectorBank(t / "sel", t / "sel" / "pca.pt", units, device="cpu")


class FakeProcOut(dict):
    def to(self, dev):
        return self


class FakeProc:
    """The processor's official call: chat text + image -> input_ids with one image token per merged token."""
    def __call__(self, text, images, return_tensors):
        rows, cols = images[0]["grid"]
        return FakeProcOut(input_ids=torch.tensor([[5, 5] + [99] * (rows * cols) + [5, 5, 5]]))

    def batch_decode(self, new, skip_special_tokens=True):
        return [f"o{int(t[0]) - 100}" for t in new]


class FakeModel:
    def __init__(self, fail_at=None):
        self.calls, self.fail_at = 0, fail_at

    def generate(self, input_ids, do_sample, num_beams, max_new_tokens, use_cache, output_logits, return_dict_in_generate):
        assert not do_sample and num_beams == 1 and use_cache and return_dict_in_generate
        self.calls += 1
        if self.fail_at is not None and self.calls >= self.fail_at:
            raise RuntimeError("simulated crash")
        n = int((input_ids == 99).sum()); return Namespace(sequences=torch.cat([input_ids, torch.tensor([[100 + n, 7]])], 1))


class FakeOfficialHost(FakeHost):
    def __init__(self, fail_at=None):
        super().__init__(); self.proc, self.model = FakeProc(), FakeModel(fail_at)

    def chat_text(self, prompt, with_image=True):
        self.prompts.append(prompt); return prompt


class OfficialAnchor(unittest.TestCase):
    def test_rows_marker_resume_crash(self):
        """Decision 15: one official generate per item at batch 1 -> full_official@s<i>of<n>.jsonl (single writer) with the common row fields;
        marker only when complete; resume after a crash redoes nothing and writes no duplicate."""
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); tiny_sets(t / "sets", n=13); a = Namespace(sets_dir=str(t / "sets"), bench="gqa_dev", shard="0/1", out=str(t / "out"), limit=0)
            with self.assertRaises(RuntimeError):
                RN.run_official(a, host=FakeOfficialHost(fail_at=6), opener=lambda f: {"grid": GRIDS[f]})
            od = t / "out" / "gqa_dev"; self.assertFalse((od / "done_official_shard0of1.json").exists())
            h = FakeOfficialHost(); RN.run_official(a, host=h, opener=lambda f: {"grid": GRIDS[f]}); self.assertEqual(h.model.calls, 13 - 5)
            rows = [json.loads(x) for x in (od / "full_official@s0of1.jsonl").read_text().splitlines()]; self.assertEqual(len({r["qid"] for r in rows}), 13)
            self.assertEqual(len(rows), 13)
            for r in rows:
                n = int(np.prod(GRIDS[r["image_id"]])); self.assertEqual((r["pred"], r["kept"], r["n_vis"], r["f"], r["n_new"], r["hit_max"]), (f"o{n}", n, n, 1.0, 2, False))
                self.assertIn("official", r["path"])
            self.assertTrue(all(p.endswith("\nShort") for p in h.prompts))                  # the item's pinned prompt, through the chat template
            mk = json.loads((od / "done_official_shard0of1.json").read_text()); self.assertEqual((mk["items"], mk["row"]), (13, RN.OFFICIAL))
            h2 = FakeOfficialHost(); RN.run_official(a, host=h2, opener=lambda f: {"grid": GRIDS[f]}); self.assertEqual(h2.model.calls, 0)
            with self.assertRaises(RN.MissingImage):
                RN.run_official(Namespace(sets_dir=str(t / "sets"), bench="gqa_dev", shard="0/1", out=str(t / "out2"), limit=0), host=FakeOfficialHost())

    def test_stage2_completeness_covers_the_anchor(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); cfgdir = t / "cfg"; PL.write_configs(LR, cfgdir, "core"); pred = t / "pred"
            for b in PL.BENCHES:
                tiny_sets(t / "sets", bench=b, n=7); od = pred / b; od.mkdir(parents=True)
                qids = [json.loads(x)["qid"] for x in (t / "sets" / f"{b}.jsonl").read_text().splitlines()]
                for c in PL.read_configs(cfgdir / f"configs_{b}.txt") + [RN.OFFICIAL]:
                    (od / f"{RUN.cfg_file(c)}@s0of1.jsonl").write_text("".join(json.dumps({"qid": q, "pred": "x"}) + "\n" for q in qids))
                for i in range(PL.QUALITY_SHARDS[b]):
                    (od / f"done_shard{i}of{PL.QUALITY_SHARDS[b]}.json").write_text("{}")
                (od / "done_official_shard0of1.json").write_text("{}")
            a = Namespace(sets_dir=str(t / "sets"), configs_dir=str(cfgdir), pred_dir=str(pred), out=str(t / "c"), benches=list(PL.BENCHES), official=True)
            self.assertEqual(RN.complete(a), 0)
            (pred / "textvqa_val" / "full_official@s0of1.jsonl").unlink(); self.assertEqual(RN.complete(a), 7)
            res = json.loads((t / "c" / "completeness.json").read_text()); self.assertIn(RN.OFFICIAL, res["benches"]["textvqa_val"]["examples"])
            a.official = False; self.assertEqual(RN.complete(a), 0)


# ------------------------------------------------------------------------------------------------------------------------------ runner
class Runner(unittest.TestCase):
    def test_run_rows_files_markers_resume(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); tiny_sets(t / "sets"); tiny_units(t / "sel"); main = PL.unit_names(LR); cf = t / "cfg.txt"
            cfgs = ["full", "static:grid:0.25", "static:random:0.5", "static:qsim:0.125", f"sel:{main['vcs'][0]}:0.25", f"sel:{main['direct'][1]}:0.75",
                    f"dyn:{main['vcs'][2]}:0.5", f"dyn:{main['logistic'][0]}:0.25"]
            cf.write_text("\n".join(cfgs) + "\n"); h = FakeHost(); units = sorted({PL.parse(c)[1] for c in cfgs if c.startswith(("sel", "dyn"))})
            RN.run(run_args(t, cf), host=h, bank=bank_for(t, units), opener=lambda f: {"grid": GRIDS[f]}, grid_of=lambda f: GRIDS[f])
            od = t / "out" / "gqa_dev"; names = sorted(x.name for x in od.glob("*.jsonl")); self.assertEqual(len(names), len(cfgs))
            self.assertTrue(all(n.endswith("@s0of1.jsonl") for n in names)); self.assertTrue((od / "done_shard0of1.json").exists())
            for c in cfgs:
                rows = [json.loads(x) for x in (od / f"{RUN.cfg_file(c)}@s0of1.jsonl").read_text().splitlines()]; self.assertEqual(len(rows), 13)
                for r in rows:
                    n = r["n_vis"]; self.assertEqual(n, int(np.prod(GRIDS[r["image_id"]]))); self.assertEqual(r["pred"], f"v{r['kept']}")
                    self.assertEqual(set(r), {"qid", "image_id", "pred", "kept", "n_vis", "f", "n_new", "hit_max"}); self.assertEqual(r["n_new"], 2); self.assertFalse(r["hit_max"])
                    k, a, f = PL.parse(c)
                    if k == "full":
                        self.assertEqual(r["kept"], n)
                    elif k in ("static", "sel"):
                        self.assertEqual(r["kept"], M4.budget(f, n)); self.assertEqual(r["f"], f)
                    else:
                        self.assertIn(r["f"], PL.DYN_GRID); self.assertEqual(r["kept"], n if r["f"] >= 1 else M4.budget(r["f"], n))
            self.assertTrue(all(p.endswith("\nShort") for p in h.prompts))                    # the item's pinned prompt as given (no second suffix)
            h2 = FakeHost(); RN.run(run_args(t, cf), host=h2, bank=bank_for(t, units), opener=lambda f: {"grid": GRIDS[f]}, grid_of=lambda f: GRIDS[f])
            self.assertEqual(h2.core.gen_calls, 0)                                              # resume: nothing redone
            mk = json.loads((od / "done_shard0of1.json").read_text()); self.assertEqual(mk["items"], 13); self.assertIn("selector_files", mk)

    def test_crash_resume_and_single_writer(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); tiny_sets(t / "sets", n=20); tiny_units(t / "sel"); cf = t / "cfg.txt"; cf.write_text("full\nstatic:grid:0.5\n")

            class Crash(FakeHost):
                def __init__(self, at):
                    super().__init__(); self.at = at

                def decode(self, toks):
                    if self.core.gen_calls >= self.at:
                        raise RuntimeError("simulated crash")
                    return super().decode(toks)
            for sh in ("0/2", "1/2"):
                with self.assertRaises(RuntimeError):
                    RN.run(run_args(t, cf, shard=sh, chunk=4, batch=4), host=Crash(3), bank=None, opener=lambda f: {"grid": GRIDS[f]}, grid_of=lambda f: GRIDS[f])
                RN.run(run_args(t, cf, shard=sh, chunk=4, batch=4), host=FakeHost(), bank=None, opener=lambda f: {"grid": GRIDS[f]}, grid_of=lambda f: GRIDS[f])
            od = t / "out" / "gqa_dev"
            self.assertEqual(sorted(x.name for x in od.glob("*.jsonl")), ["full@s0of2.jsonl", "full@s1of2.jsonl", "static__grid__0.5@s0of2.jsonl", "static__grid__0.5@s1of2.jsonl"])
            for c in ("full", "static:grid:0.5"):
                recs = [json.loads(x) for p in RUN.cfg_paths(od, c) for x in p.read_text().splitlines()]
                self.assertEqual(len(recs), 20); self.assertEqual(len({r["qid"] for r in recs}), 20)
            s0 = {json.loads(x)["image_id"] for x in (od / "full@s0of2.jsonl").read_text().splitlines()}
            s1 = {json.loads(x)["image_id"] for x in (od / "full@s1of2.jsonl").read_text().splitlines()}
            self.assertFalse(s0 & s1)                                                           # D1-30 sharding by image group

    def test_items_without_answers_ordering_and_stops(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); tiny_sets(t / "sets", n=13); items = RN.load_items(t / "sets", "gqa_dev")
            self.assertFalse(any(k in it for it in items for k in ("answer", "answers", "label")))
            srt = RN.sort_by_tokens(items, {f: GRIDS[f] for f in GRIDS}); n = [np.prod(GRIDS[it["image_file"]]) for it in srt]
            self.assertEqual(n, sorted(n)); self.assertEqual(srt, RN.sort_by_tokens(items, {f: GRIDS[f] for f in GRIDS}))      # deterministic
            for img in GRIDS:                                                                   # same-image items stay contiguous
                pos = [j for j, it in enumerate(srt) if it["image_id"] == img]; self.assertEqual(pos, list(range(pos[0], pos[0] + len(pos))))
            cf = t / "cfg.txt"; cf.write_text("full\n")
            with self.assertRaises(SystemExit):                                                 # header estimate != encoder grid -> stop
                RN.run(run_args(t, cf), host=FakeHost(), opener=lambda f: {"grid": GRIDS[f]}, grid_of=lambda f: (9, 9))
            with self.assertRaises(RN.MissingImage):
                RN.merged_grid(t / "nope.jpg")
            with self.assertRaises(RN.MissingImage):
                RN.open_image(t / "nope.jpg")
            from PIL import Image
            Image.new("RGB", (1120, 896)).save(t / "a.png"); self.assertEqual(RN.merged_grid(t / "a.png"), (32, 40))
            Image.new("RGB", (640, 480)).save(t / "b.png"); self.assertEqual(RN.merged_grid(t / "b.png"), (17, 23))
            Image.new("RGB", (100, 60)).save(t / "c.png"); r, c = RN.merged_grid(t / "c.png"); self.assertGreaterEqual(r * c, 256)
            src = [x.name for x in (R / "scripts").glob("r7_d1_40c_*.py")]
            for f in src:
                code = (R / "scripts" / f).read_text(); self.assertNotIn("import zipfile", code, f); self.assertNotIn("TE.ImageSource(", code, f)

    def test_stage0_check_and_gridcheck(self):
        """Stage 0 on synthetic sets: sha256 / counts against the build summary, missing images, merged grids beyond 64 × 64, the 9 units'
        files and taus, configuration files and rows.json; the header-only grid summary."""
        import hashlib
        from PIL import Image
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); sets = t / "sets"; sets.mkdir(); (t / "img").mkdir(); summ = {}
            sizes = {"a": (640, 480), "b": (1120, 896), "c": (300, 200)}
            for k, (w, h) in sizes.items():
                Image.new("RGB", (w, h)).save(t / "img" / f"{k}.png")
            for b in PL.BENCHES:
                rows = [{"qid": f"{b}_{i}", "image_id": k, "image_file": str(t / "img" / f"{k}.png"), "question": "q?", "prompt": "q?", "answer": "x"}
                        for i, k in enumerate("aabbc")]
                (sets / f"{b}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
                summ[b] = {"items": 5, "groups": 3, "sha256": hashlib.sha256((sets / f"{b}.jsonl").read_bytes()).hexdigest()}
            (sets / "d1_30_sets.json").write_text(json.dumps(summ)); sel = t / "sel"; tiny_units(sel)
            a = Namespace(sets_dir=str(sets), configs_dir=str(t / "cfg"), lr_choice=str(sel / "lr_choice.json"), selectors_dir=str(sel), pca=str(sel / "pca.pt"),
                          out=str(t / "check"), rows="core")
            self.assertEqual(RN.check(a), 0); res = json.loads((t / "check" / "check.json").read_text()); self.assertTrue(res["pass"])
            g = res["benches"]["gqa_dev"]; self.assertEqual((g["images"], g["grid_max_rows"], g["grid_max_cols"], g["images_over_64x64"]), (3, 32, 40, 0))
            self.assertEqual(g["tokens_per_item"]["max"], 1280); rows_rec = json.loads((t / "cfg" / "rows.json").read_text())
            self.assertEqual(len(rows_rec["units"]), 9); self.assertEqual(PL.read_configs(t / "cfg" / "configs_gqa_dev.txt"), PL.config_list(PL.unit_names(LR)))
            self.assertEqual(RN.grid_summary(sets)["chartqa_val"]["tokens_per_item"]["sum"], g["tokens_per_item"]["sum"])
            Image.new("RGB", (4000, 200)).save(t / "img" / "c.png")                          # 7 × 143 merged grid: beyond the tables
            (t / "img" / "a.png").unlink(); (sets / "textvqa_val.jsonl").write_text((sets / "textvqa_val.jsonl").read_text() + "\n")
            self.assertEqual(RN.check(a), 5); res = json.loads((t / "check" / "check.json").read_text()); self.assertFalse(res["pass"]); txt = " ".join(res["problems"])
            self.assertIn("beyond the 64 × 64", txt); self.assertIn("missing images", txt); self.assertIn("textvqa_val: sha256", txt)
            (sel / f"{PL.unit_names(LR)['vcs'][1]}.json").write_text(json.dumps({"dynamic": {"0.5": {"tau": 0.1}}}))
            RN.check(a); self.assertIn("CAL taus missing", " ".join(json.loads((t / "check" / "check.json").read_text())["problems"]))

    @unittest.skipUnless(HAS_TF, "needs transformers with Qwen2.5-VL (vl_baselines env)")
    def test_smart_resize_matches_transformers(self):
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
        g = np.random.default_rng(0)
        for _ in range(3000):
            h, w = int(g.integers(20, 4000)), int(g.integers(20, 4000))
            if max(h, w) / min(h, w) > 200:
                continue
            self.assertEqual(RN.smart_resize(h, w), smart_resize(h, w, factor=28, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28))


# ------------------------------------------------------------------------------------------------------------------------------ selectors
class Bank(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory(); self.d = Path(self.t.name); self.cfg = tiny_units(self.d / "sel"); main = PL.unit_names(LR)
        self.units = [main["vcs"][0], main["logistic"][1], main["direct"][2]]; self.bank = bank_for(self.d, self.units)
        g = torch.Generator().manual_seed(5); self.items = []
        for rows, cols, lq in ((3, 4, 5), (4, 5, 3), (2, 7, 6), (5, 5, 4)):
            self.items.append((torch.randn(rows * cols, E, generator=g).bfloat16(), (rows, cols), torch.randn(lq, E, generator=g).bfloat16()))

    def tearDown(self):
        self.t.cleanup()

    def test_fixed_is_hard_topk_and_inputs_match_cache(self):
        for feat, grid, q in self.items:
            st = self.bank.prepare(feat, grid, q); n = feat.shape[0]
            self.assertTrue(torch.equal(st["v"][0], ((feat.float() - self.bank.mean) @ self.bank.basis).half().float()))   # the cache's fp16 rounding
            for u in self.units:
                g, _ = self.bank._gates(u, st)
                for f in (0.125, 0.25, 0.5, 0.75):
                    k = M4.budget(f, n); kept = self.bank.kept(u, st, f); top = np.sort(torch.argsort(g[0], descending=True)[:k].numpy())
                    self.assertTrue(np.array_equal(kept, top)); self.assertEqual(len(kept), k)
                self.assertTrue(np.array_equal(self.bank.kept(u, st, 1.0), np.arange(n)))

    def test_dynamic_rule_matches_frozen_risk_grid(self):
        """Per-item risks / choice == r7_d1_40_selectors.risk_grid + the dynamic_tau40 rule on TupleData40 holding the same items."""
        feats = torch.cat([f for f, _, _ in self.items]).float(); ntok = [f.shape[0] for f, _, _ in self.items]; off = np.concatenate([[0], np.cumsum(ntok)[:-1]])
        Lq = max(q.shape[0] for _, _, q in self.items); emb = torch.cat([q for _, _, q in self.items]).float(); q_ids = np.zeros((4, Lq), np.int64); q_len = []
        o = 0
        for i, (_, _, q) in enumerate(self.items):
            q_ids[i, :q.shape[0]] = np.arange(o, o + q.shape[0]); q_len.append(q.shape[0]); o += q.shape[0]
        pf = (((feats - self.bank.mean) @ self.bank.basis).half())
        data = S4.TupleData40(pf, off, ntok, [g[1] for _, g, _ in self.items], np.arange(4), q_ids, np.array(q_len), emb.bfloat16(),
                              np.zeros((4, 3, max(ntok)), np.int16), np.ones((4, 4), np.int32), np.ones(4, np.int64), np.zeros((4, 12)), np.zeros((4, 12)), "cpu", 8, 8)
        for u in self.units:
            m = self.bank.models[u]; ref = S4.risk_grid(m, data, m.cfg); m.eval()                  # risk_grid leaves the model in train mode
            for i, (feat, grid, q) in enumerate(self.items):
                mine = self.bank.risks(u, self.bank.prepare(feat, grid, q)); self.assertTrue(np.allclose(mine, ref[i], atol=1e-5, rtol=1e-4), (u, mine, ref[i]))
            for op in PL.OPS:
                tau = float(np.median(ref[:, 1])); self.bank.tau[u][op] = tau; ok = ref <= tau; ok[:, -1] = True
                exp = np.asarray(PL.DYN_GRID)[np.argmax(ok, 1)]
                got = [self.bank.dynamic_fraction(u, self.bank.prepare(f, g, q), float(op)) for f, g, q in self.items]
                self.assertEqual(got, exp.tolist())

    def test_batched_selection_equals_per_item(self):
        for u in self.units:
            states = [self.bank.prepare(f, g, q) for f, g, q in self.items]
            for f in (0.125, 0.5):
                kb, fb = self.bank.select_batch(u, states, f=f)
                for st, k, x in zip([self.bank.prepare(ff, g, q) for ff, g, q in self.items], kb, fb):
                    self.assertTrue(np.array_equal(k, self.bank.kept(u, st, f))); self.assertEqual(x, f)
            self.bank.tau[u]["0.5"] = float(np.median([self.bank.risks(u, self.bank.prepare(*it))[2] for it in self.items]))
            kb, fb = self.bank.select_batch(u, states, op=0.5)
            for (ff, g, q), k, x in zip(self.items, kb, fb):
                st = self.bank.prepare(ff, g, q); self.assertEqual(x, self.bank.dynamic_fraction(u, st, 0.5)); self.assertTrue(np.array_equal(k, self.bank.kept(u, st, x)))

    def test_grid_beyond_tables_stops(self):
        with self.assertRaises(SystemExit):
            self.bank.prepare(torch.zeros(9 * 2, E), (9, 2), torch.zeros(3, E))
        with self.assertRaises(RuntimeError):
            self.bank.prepare(torch.zeros(10, E), (3, 4), torch.zeros(3, E))


# ------------------------------------------------------------------------------------------------------------------------------ tiny Qwen
def tiny_qwen():
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
    torch.manual_seed(0)
    cfg = Qwen2_5_VLConfig(text_config=dict(vocab_size=160, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                                            num_key_value_heads=2, max_position_embeddings=2048, rope_theta=10000.0, rms_norm_eps=1e-6,
                                            rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]}),
                           vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64, patch_size=14,
                                              spatial_merge_size=2, temporal_patch_size=2, window_size=56, fullatt_block_indexes=[1], tokens_per_second=2),
                           image_token_id=150, video_token_id=151, vision_start_token_id=152, vision_end_token_id=153, attn_implementation="sdpa")
    m = Qwen2_5_VLForConditionalGeneration(cfg).eval(); m.generation_config.repetition_penalty = 1.05
    m.generation_config.eos_token_id = [7, 8]; m.generation_config.pad_token_id = 9; m.generation_config.do_sample = False
    return m


@unittest.skipUnless(HAS_TF, "needs transformers with Qwen2.5-VL (vl_baselines env)")
class InjectedHFGenerate(unittest.TestCase):
    """The smoke's 'official' reference for pruned rows: HF generate with the spliced ids injected through get_rope_index equals QwenCore at
    batch 1 and in a left-padded batch, on full and pruned rows (tiny random Qwen2.5-VL, fp32, CPU)."""

    def test_pruned_rows(self):
        import r7_d1_40_host as QH
        import r7_d1_40c_smoke as SMK
        model = tiny_qwen(); core = QH.QwenCore(model, [7, 8], 9, 1.05); g = torch.Generator().manual_seed(3)
        host = Namespace(torch=torch, model=model, core=core, decode=lambda toks: [" ".join(str(int(x)) for x in t if int(x) not in (7, 8, 9)) for t in toks])
        seqs, poss, pids, refs = [], [], [], []
        for gh, gw, n_pre, n_post, kept in ((4, 6, 5, 7, None), (6, 4, 3, 9, [0, 2, 5]), (2, 8, 6, 4, [1, 3]), (4, 4, 4, 5, [0, 1, 2])):
            thw = torch.tensor([1, gh, gw]); n = gh * gw // 4; pre = torch.cat([torch.randint(10, 140, (n_pre - 1,), generator=g), torch.tensor([152])])
            post = torch.cat([torch.tensor([153]), torch.randint(10, 140, (n_post - 1,), generator=g)]); pv = torch.randn(gh * gw, 3 * 2 * 14 * 14, generator=g)
            f = core.visual_tokens(pv, thw[None])[0]; kk = np.arange(n) if kept is None else np.asarray(kept); pos = core.full_positions(pre, n, post, thw)
            seq = torch.cat([core.embed_ids(pre), f[torch.as_tensor(kk)], core.embed_ids(post)]); pp = QH.prune_positions(pos, len(pre), n, kk)
            pid = np.concatenate([pre.numpy(), np.full(len(kk), 150), post.numpy()]); seqs.append(seq); poss.append(pp); pids.append(pid)
            hf = SMK.hf_generate_spliced(host, seq, pp, pid, 6); new, _ = core.generate([seq], [pp], [pid], max_new_tokens=6)
            self.assertEqual(hf, host.decode(new)[0]); refs.append(hf)
            self.assertFalse(hasattr(model.model, "__dict__") and "get_rope_index" in model.model.__dict__)       # hook removed
        new, _ = core.generate(seqs, poss, pids, max_new_tokens=6)
        self.assertEqual(host.decode(new), refs)


# ------------------------------------------------------------------------------------------------------------------------------ timing
class Timing(unittest.TestCase):
    def test_b_star_report_checks_and_pools(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "rep.json"; p.write_text(json.dumps({"gpu": "NVIDIA H100 NVL", "host": "nodesumo01", "B_star": 16, "rule": "x"}))
            b, rec = TMG.read_b_star(p); self.assertEqual(b, 16); self.assertEqual(rec["host"], "nodesumo01")
            p.write_text(json.dumps({"gpu": "NVIDIA L40S", "host": "node50", "B_star": 16}))
            with self.assertRaises(SystemExit):
                TMG.read_b_star(p)
            self.assertEqual(TMG.read_b_star(p, allow_any=True)[0], 16)
            p.write_text(json.dumps({"gpu": "NVIDIA H100 NVL", "host": "nodesumo01", "B_star": 16, "rule": "x"}))
            with self.assertRaisesRegex(SystemExit, "primary batch 16"):                       # B* must be among the measured batches
                TMG.main(["--sets-dir", "x", "--bench", "gqa_dev", "--configs-file", "x", "--out", "x", "--timing-report", str(p), "--batches", "1", "4"])
        self.assertEqual(TMG.primary_batch(None), 4); self.assertEqual(TMG.primary_batch(16), 16)
        self.assertEqual([TMG.pool_size(B, 16) for B in (1, 4, 16, 32)], [64, 256, 512, 256]); self.assertEqual([TMG.pool_size(B, 4) for B in (1, 4, 16, 32)], [64, 512, 256, 256])
        with self.assertRaises(SystemExit):
            TMG.main(["--sets-dir", "x", "--bench", "gqa_dev", "--configs-file", "x", "--out", "x", "--b-star", "4"])

    def test_rows_with_fake_host(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); tiny_sets(t / "sets", n=40); tiny_units(t / "sel"); main = PL.unit_names(LR); cf = t / "tim.txt"
            u = main["vcs"][0]; cf.write_text(f"full\nstatic:grid:0.25\nsel:{u}:0.5\ndyn:{u}:0.5\n")
            bank = bank_for(t, [u]); bank.tau[u]["0.5"] = 1e9                              # every item -> 0.125 (smallest fraction)
            TMG.main(["--sets-dir", str(t / "sets"), "--bench", "gqa_dev", "--configs-file", str(cf), "--out", str(t / "o"), "--n", "32", "--n-big", "16", "--n1", "4",
                      "--batches", "1", "4", "16", "--allow-any-gpu", "--b-star", "16"], host=FakeHost(), bank=bank, opener=lambda f: {"grid": GRIDS[f]})
            j = json.loads((t / "o" / "timing_gqa_dev.json").read_text()); rows = j["rows"]
            self.assertEqual(j["b_star"], 16); self.assertEqual(j["primary_batch"], 16); self.assertEqual(j["pools"], {"B1": 4, "B4": 16, "B16": 32})
            self.assertEqual(len(rows), 12); self.assertEqual({r["pool"] for r in rows if r["B"] == 16}, {32}); self.assertEqual({r["n_batches"] for r in rows if r["B"] == 4}, {4})
            r = next(r for r in rows if r["config"] == "static:grid:0.25" and r["B"] == 4); self.assertAlmostEqual(r["mean_kept_fraction"], 0.25, delta=0.06)
            self.assertTrue(0 <= r["vision_share_of_per_item"] <= 1); lo, hi = r["per_item_ms_median_ci95"]; self.assertTrue(lo <= r["per_item_ms_median"] <= hi)
            d = next(r for r in rows if r["config"] == f"dyn:{u}:0.5" and r["B"] == 4)
            self.assertTrue(d["bucketed"]); self.assertEqual(d["budget_share"], {"0.125": 1.0}); self.assertEqual(d["stage2_batches"], {"0.125": 4})
            bank.tau[u]["0.5"] = -1e9                                                         # every item -> 1 (full)
            TMG.main(["--sets-dir", str(t / "sets"), "--bench", "gqa_dev", "--configs-file", str(cf), "--out", str(t / "o2"), "--n", "32", "--n-big", "16", "--n1", "4",
                      "--batches", "16", "--allow-any-gpu", "--b-star", "none"], host=FakeHost(), bank=bank, opener=lambda f: {"grid": GRIDS[f]})
            j2 = json.loads((t / "o2" / "timing_gqa_dev.json").read_text()); d2 = next(r for r in j2["rows"] if r["config"].startswith("dyn:"))
            self.assertEqual(j2["primary_batch"], 4); self.assertEqual(d2["budget_share"], {"1": 1.0}); self.assertEqual(d2["mean_kept_fraction"], 1.0)
            self.assertEqual(d2["stage2_cycled_items"], {"1": 0})


class TimingBundle(unittest.TestCase):
    def test_bundle_bstar_parts_marker_retry(self):
        """One job for every reference-node unit: B* first (the frozen procedure stubbed here; never re-measured once its report exists), then the
        three benchmarks back-to-back, one marker; a retried bundle keeps the parts an earlier attempt wrote."""
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); tiny_units(t / "sel"); cfg = t / "cfg"; PL.write_configs(LR, cfg, "core")
            for b in PL.BENCHES:
                tiny_sets(t / "sets", bench=b, n=24)
            calls = []

            def fake_bstar(host, report, allow_any):
                calls.append(1); Path(report).parent.mkdir(parents=True, exist_ok=True)
                Path(report).write_text(json.dumps({"gpu": "NVIDIA H100 NVL", "host": "nodesumo01", "B_star": 4, "rule": "stub"}))
            bank = bank_for(t, sorted({PL.parse(c)[1] for c in PL.read_configs(cfg / "timing_gqa_dev.txt") if c.startswith(("sel:", "dyn:"))}))
            argv = ["--sets-dir", str(t / "sets"), "--configs-dir", str(cfg), "--out", str(t / "tim"), "--timing-report", str(t / "r40" / "timing_report.json"),
                    "--b-star-mode", "in_bundle", "--selectors-dir", str(t / "sel"), "--pca", str(t / "sel" / "pca.pt"), "--n", "8", "--n-big", "8", "--n1", "2", "--batches", "1", "4"]
            with self.assertRaises(SystemExit):                                                # 'report' mode with no report: the bundle stops
                TMG.bundle([x if x != "in_bundle" else "report" for x in argv], host=FakeHost(), bank=bank, opener=lambda f: {"grid": GRIDS[f]}, bstar_fn=fake_bstar)
            TMG.bundle(argv, host=FakeHost(), bank=bank, opener=lambda f: {"grid": GRIDS[f]}, bstar_fn=fake_bstar)
            mk = json.loads((t / "tim" / TMG.BUNDLE_MARK).read_text()); self.assertEqual([p["part"] for p in mk["parts"]], ["b_star", *PL.BENCHES])
            self.assertTrue(mk["parts"][0]["status"].startswith("measured in this bundle")); self.assertEqual(calls, [1])
            for b in PL.BENCHES:
                j = json.loads((t / "tim" / b / f"timing_{b}.json").read_text()); self.assertEqual(j["b_star"], 4); self.assertEqual(j["pools"], {"B1": 2, "B4": 8})
                self.assertEqual(len(j["rows"]), 18 * 2)
            (t / "tim" / TMG.BUNDLE_MARK).unlink(); (t / "tim" / "chartqa_val" / "timing_chartqa_val.json").unlink()   # a retry after a failure in part 3
            TMG.bundle(argv, host=FakeHost(), bank=bank, opener=lambda f: {"grid": GRIDS[f]}, bstar_fn=fake_bstar)
            mk = json.loads((t / "tim" / TMG.BUNDLE_MARK).read_text()); st = [p["status"] for p in mk["parts"]]
            self.assertEqual(calls, [1]); self.assertTrue(st[0].startswith("reused")); self.assertTrue(st[1].startswith("kept") and st[2].startswith("kept"))
            self.assertEqual(st[3], "measured")


# ------------------------------------------------------------------------------------------------------------------------------ scorer
class Scoring(unittest.TestCase):
    BENCHES = ("gqa_dev", "chartqa_val") + (("textvqa_val",) if HAS_LLAVA else ())

    def setup(self, t: Path, b_star=16, drop=None, rows="core", timing=True):
        main = PL.unit_names(LR); g = np.random.default_rng(0); (t / "sets").mkdir(parents=True); cfgdir = t / "cfg"; PL.write_configs(LR, cfgdir, rows, self.BENCHES)
        acc = {0.125: 0.5, 0.25: 0.6, 0.5: 0.7, 0.75: 0.8, 1.0: 0.9}
        for bench in self.BENCHES:
            items = []
            for i in range(60):
                it = {"bench": bench, "qid": f"{bench}_{i}", "image_id": f"img{i // 3}", "question": "?", "prompt": "?"}
                if bench.startswith("gqa"):
                    it.update(answer="yes", structural=["verify", "query"][i % 2])
                elif bench.startswith("chartqa"):
                    it.update(answer="12", type=["human_test", "augmented_test"][i % 2])
                else:
                    it.update(answers=["yes"] * 10)
                items.append(it)
            (t / "sets" / f"{bench}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in items)); pd = t / "pred" / bench; pd.mkdir(parents=True)
            right = "12" if bench.startswith("chartqa") else "yes"
            for c in PL.read_configs(cfgdir / f"configs_{bench}.txt"):
                k, a, f = PL.parse(c); f = 1.0 if k == "full" else f; recs = []
                for it in items:
                    n = 300 + (hash(it["image_id"]) % 50); fr = f if k != "dyn" else [0.25, 0.5, 0.75, 1.0][int(g.integers(4))]
                    p = acc.get(fr, 0.75) + (0.05 if (k == "sel" and a.startswith("vcs")) else 0.0)
                    recs.append({"qid": it["qid"], "image_id": it["image_id"], "pred": right if g.uniform() < p else "no", "kept": n if fr >= 1 else M4.budget(fr, n),
                                 "n_vis": n, "f": fr, "n_new": 2, "hit_max": False})
                if drop == (bench, c):
                    recs = recs[:-1]
                (pd / f"{RUN.cfg_file(c)}@s0of1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
                if c == "full" and drop != (bench, RN.OFFICIAL):                              # the batch-1 official anchor: same answers but one
                    (pd / "full_official@s0of1.jsonl").write_text("".join(json.dumps(dict(r, pred=("zz" if i == 0 else r["pred"]))) + "\n" for i, r in enumerate(recs)))
            if timing:
                lat = {0.125: 40.0, 0.25: 55.0, 0.5: 70.0, 0.75: 85.0}; tm = []
                for B in (1, 4, 16, 32):
                    tm.append({"config": "full", "B": B, "per_item_ms_median": 100.0, "per_item_ms_median_ci95": [99.0, 101.0]})
                    for c in PL.read_configs(cfgdir / f"timing_{bench}.txt"):
                        k, a, f = PL.parse(c)
                        if k == "static":
                            tm.append({"config": c, "B": B, "per_item_ms_median": lat[f]})
                        elif k == "sel":
                            tm.append({"config": c, "B": B, "per_item_ms_median": lat[f] + 2.0})
                        elif k == "dyn":
                            tm.append({"config": c, "B": B, "per_item_ms_median": 60.0 + 20 * f})
                (t / "tim" / bench).mkdir(parents=True, exist_ok=True)
                (t / "tim" / bench / f"timing_{bench}.json").write_text(json.dumps({"rows": tm, "b_star": b_star, "primary_batch": b_star or 4, "gpu": "NVIDIA H100 NVL"}))
        (t / "lr.json").write_text(json.dumps(LR)); (t / "proto.yaml").write_text("status: x\nr7_40c_staged_freeze_20261013:\n  a: 1\n")
        (t / "rep.json").write_text(json.dumps({"gpu": "NVIDIA H100 NVL", "host": "nodesumo01", "B_star": b_star}))
        return ["--sets-dir", str(t / "sets"), "--pred-dir", str(t / "pred"), "--timing-dir", str(t / "tim"), "--configs-dir", str(cfgdir), "--lr-choice", str(t / "lr.json"),
                "--out", str(t / "res"), "--benches", *self.BENCHES, "--reps", "200", "--protocol", str(t / "proto.yaml"), "--timing-report", str(t / "rep.json")]

    def test_latency_basis(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); args = self.setup(t, b_star=16); SCO.main(args); res = json.loads((t / "res" / "r7_40c_results.json").read_text())
            self.assertEqual(res["primary_basis"], "latency"); self.assertEqual(res["b_star"], 16); nb = len(self.BENCHES)
            for b, Rb in res["benches"].items():
                prim = {k: v for k, v in Rb["contrasts"].items() if v.get("role") == "primary"}
                self.assertEqual(sorted(prim), sorted(f"S1-{o}@{L}" for o in ("S2", "S3", "grid") for L in ("L25", "L50", "L75")))
                for v in prim.values():
                    self.assertEqual(v["holm_family_size"], 3 * nb); self.assertEqual(v["batch"], 16); self.assertTrue(v["ci95"][0] - 1e-9 <= v["delta"] <= v["ci95"][1] + 1e-9)
                    self.assertEqual(len(v["per_seed"]), 3); self.assertGreaterEqual(v["p_holm"], v["p"])
                self.assertEqual(Rb["contrasts"]["S1-S2@f0.25"]["role"], "secondary"); self.assertIn("A3:S1_dyn-fixed@op0.5", Rb["contrasts"])
                self.assertAlmostEqual(sum(Rb["dynamic_fraction_distribution"][next(iter(Rb["dynamic_fraction_distribution"]))].values()), 1.0)
                self.assertEqual(Rb["latency_points_ms"]["16"], {"L25": 55.0, "L50": 70.0, "L75": 85.0}); self.assertIn("within_1pt_of_full_ms_B16", Rb)
                self.assertAlmostEqual(Rb["fixed_fraction"]["static:grid:0.25"]["mean_kept_fraction"], 0.25, delta=0.01)
            self.assertEqual(set(res["benches"]["gqa_dev"]["subsets"]), {"verify", "query"}); self.assertEqual(set(res["benches"]["chartqa_val"]["subsets"]), {"human", "augmented"})
            A = res["benches"]["gqa_dev"]["official_anchor"]; self.assertEqual(A["row"], "full_official"); self.assertAlmostEqual(A["identical_answer_fraction"], 59 / 60)
            self.assertEqual(A["full_batched_minus_official"]["role"], "secondary"); self.assertFalse(any(k.startswith("S1-") and "official" in k for k in res["benches"]["gqa_dev"]["contrasts"]))
            with self.assertRaises(SystemExit):                                               # never overwrite a readout
                SCO.main(args)

    def test_fraction_basis_without_b_star(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); args = self.setup(t, b_star=None, timing=False); args[args.index("--timing-dir") + 1] = str(t / "none")
            SCO.main(args); res = json.loads((t / "res" / "r7_40c_results.json").read_text()); nb = len(self.BENCHES)
            self.assertEqual(res["primary_basis"], "fraction"); self.assertIsNone(res["b_star"])
            for b, Rb in res["benches"].items():
                prim = sorted(k for k, v in Rb["contrasts"].items() if v.get("role") == "primary")
                self.assertEqual(prim, sorted(f"S1-{o}@f{f}" for o in ("S2", "S3", "grid") for f in ("0.25", "0.5", "0.75")))
                self.assertTrue(all(Rb["contrasts"][k]["holm_family_size"] == 3 * nb for k in prim)); self.assertIn("S1-S2@f0.125", Rb["contrasts"])
                self.assertFalse(any(k.endswith(("@L25", "@L50", "@L75")) for k in Rb["contrasts"]))
                c = Rb["contrasts"]["S1-grid@f0.5"]; self.assertEqual(len(c["seed_deltas"]), 3); self.assertEqual(len(c["per_seed"]), 3)

    def test_guards(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); args = self.setup(t, drop=("gqa_dev", "static:grid:0.5"))
            with self.assertRaises(SystemExit) as cm:
                SCO.main(args)
            self.assertIn("coverage guard", str(cm.exception))
        with tempfile.TemporaryDirectory() as t:                                              # the anchor is under the coverage guard too
            t = Path(t); args = self.setup(t, drop=("chartqa_val", RN.OFFICIAL))
            with self.assertRaises(SystemExit) as cm:
                SCO.main(args)
            self.assertIn("full_official", str(cm.exception)); SCO.main(args + ["--anchor", "none"])
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); args = self.setup(t); (t / "proto.yaml").write_text("status: frozen\n")
            with self.assertRaises(SystemExit) as cm:
                SCO.main(args)
            self.assertIn("not frozen", str(cm.exception)); self.assertFalse((t / "res").exists())
        with tempfile.TemporaryDirectory() as t:                                              # B* but no timing rows -> stop
            t = Path(t); args = self.setup(t, b_star=16, timing=False); args[args.index("--timing-dir") + 1] = str(t / "none")
            with self.assertRaises(SystemExit):
                SCO.main(args)
        with tempfile.TemporaryDirectory() as t:                                              # timing B* disagrees with the reference report
            t = Path(t); args = self.setup(t, b_star=16); (t / "rep.json").write_text(json.dumps({"gpu": "NVIDIA H100 NVL", "host": "nodesumo01", "B_star": 4}))
            with self.assertRaises(SystemExit):
                SCO.main(args)

    def test_cross_host_side_by_side(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); args = self.setup(t, b_star=16)
            d130 = {"benches": {b: {"contrasts": {f"S1-{o}@{L}": {"delta": 1.5, "ci95": [0.1, 2.9], "role": "primary" if o != "grid" else "secondary"}
                                                  for o in ("S2", "S3", "grid") for L in ("L25", "L50", "L75")}} for b in self.BENCHES}}
            (t / "d130.json").write_text(json.dumps(d130)); SCO.main(args + ["--d1-30-results", str(t / "d130.json")])
            ch = json.loads((t / "res" / "r7_40c_results.json").read_text())["cross_host_descriptive"]
            self.assertIn("no pooling", ch["note"]); self.assertEqual(len(ch["rows"]), 9 * len(self.BENCHES)); self.assertTrue(all(r["llava_delta"] == 1.5 for r in ch["rows"]))
        fb = {"fixed_budget": {"sel:vcs_lr0.0003_s0:144": {"quality": 60.0}, "sel:vcs_lr0.0003_s1:144": {"quality": 62.0}, "sel:vcs_lr0.0003_s0_noq:144": {"quality": 0.0},
                               "sel:logistic_lr0.0001_s0:144": {"quality": 58.0}, "static:grid:144": {"quality": 55.0}, "sel:vcs_lr0.0003_s0:288": {"quality": 99.0}}}
        self.assertEqual(SCO.llava_fixed_delta(fb, "S2", 144), 3.0); self.assertEqual(SCO.llava_fixed_delta(fb, "grid", 144), 6.0); self.assertIsNone(SCO.llava_fixed_delta(fb, "S3", 144))


# ------------------------------------------------------------------------------------------------------------------------------ DEV secondaries
class DevSecondaries(unittest.TestCase):
    def test_dev_complete(self):
        import hashlib
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); md, td = t / "m", t / "teacher"; md.mkdir(); td.mkdir()
            man = [{"qid": f"q{i}", "image_id": f"i{i // 2}", "role": "DEV" if i < 8 else "CAL", "question": "?"} for i in range(12)]
            (md / "d1_manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in man)); sha = hashlib.sha256((md / "d1_manifest.jsonl").read_bytes()).hexdigest()
            (md / "d1_manifest_meta.json").write_text(json.dumps({"frozen": True, "manifest_sha256": sha}))
            (md / "d1_shards.json").write_text(json.dumps({"shards": {"DEV-000": [], "DEV-001": [], "CAL-000": []}}))
            r = DRK.dev_complete(td, md, sha); self.assertFalse(r["complete"]); self.assertEqual(len(r["problems"]), 2)
            for s, n in (("DEV-000", 5), ("DEV-001", 3)):
                (td / f"{s}.done.json").write_text(json.dumps({"expected": n, "present": n, "duplicates": 0, "skipped_images": [], "gpu": "NVIDIA L40S"}))
            r = DRK.dev_complete(td, md, sha); self.assertTrue(r["complete"]); self.assertEqual(r["present"], 8)
            self.assertFalse(DRK.dev_complete(td, md, "0" * 64)["complete"])
            (td / "DEV-001.done.json").write_text(json.dumps({"expected": 3, "present": 2, "duplicates": 0, "skipped_images": [], "gpu": "x"}))
            self.assertFalse(DRK.dev_complete(td, md, sha)["complete"])

    def test_summarize40(self):
        g = np.random.default_rng(0); n = 40; r = g.normal(size=n * 12); h = 2 * r + g.normal(size=n * 12) * 0.1
        frac = np.tile(np.array([k for _ in M4.MECHANISMS for k in M4.FKEYS]), n); mech = np.tile(np.array([m for m in M4.MECHANISMS for _ in M4.FKEYS]), n)
        s = DRK.summarize40(r, h, frac, mech, np.repeat([f"i{i // 2}" for i in range(n)], 12), reps=100)
        self.assertGreater(s["spearman_r_h"], 0.9); self.assertAlmostEqual(s["slope_h_on_r"], 2.0, delta=0.05); self.assertEqual(set(s["by_fraction"]), set(M4.FKEYS))
        self.assertEqual(s["n_images"], 20); self.assertLessEqual(s["spearman_ci95"][0], s["spearman_r_h"])


# ------------------------------------------------------------------------------------------------------------------------------ driver
STUB_SBATCH = r'''#!/usr/bin/env bash
echo "$*" >> "$STUB_DIR/sbatch.log"
n=$(cat "$STUB_DIR/counter" 2>/dev/null || echo 3000); n=$((n + 1)); echo $n > "$STUB_DIR/counter"
args="$*"; R=$R40_ROOT; RC=$R40C_ROOT
if [ -n "$STUB_FAIL_ON" ] && [[ "$args" == *"$STUB_FAIL_ON"* ]]; then exit 1; fi
if [[ "$args" == *"cpu.sbatch check"* ]]; then mkdir -p "$RC/check" "$RC/configs"; p=true; [ -n "$STUB_CHECK_FAILS" ] && p=false
  echo "{\"pass\": $p, \"configs\": {\"rows\": \"$(echo $args | awk '{print $NF}')\"}}" > "$RC/check/check.json"; fi
if [[ "$args" =~ gpu\.sbatch\ run\ ([a-z_]+)\ ([0-9]+)/([0-9]+) ]]; then b=${BASH_REMATCH[1]}; i=${BASH_REMATCH[2]}; k=${BASH_REMATCH[3]}
  c=$(cat "$STUB_DIR/tries_${b}_$i" 2>/dev/null || echo 0); echo $((c + 1)) > "$STUB_DIR/tries_${b}_$i"
  if [ "${b}_$i" != "$STUB_FLAKY" ] || [ "$c" -ge "${STUB_FLAKY_TIMES:-1}" ]; then mkdir -p "$RC/pred/$b"; echo '{}' > "$RC/pred/$b/done_shard${i}of${k}.json"; fi; fi
if [[ "$args" =~ gpu\.sbatch\ official\ ([a-z_]+)\ ([0-9]+)/([0-9]+) ]]; then mkdir -p "$RC/pred/${BASH_REMATCH[1]}"
  echo '{}' > "$RC/pred/${BASH_REMATCH[1]}/done_official_shard${BASH_REMATCH[2]}of${BASH_REMATCH[3]}.json"; fi
if [[ "$args" =~ gpu\.sbatch\ timing_bundle\ ([a-z_]+) ]]; then m=${BASH_REMATCH[1]}
  [ "$m" = "report" ] && [ ! -f "$TIMING_REPORT" ] && echo "timing bundle submitted before its gate" >> "$STUB_DIR/errors"
  mkdir -p "$RC/timing"; echo "{\"mode\": \"$m\"}" > "$RC/timing/timing_bundle.done.json"; fi
if [[ "$args" =~ gpu\.sbatch\ devteacher\ (b01[23]) ]]; then mkdir -p "$R/teacher"; echo '{}' > "$R/teacher/bundle_${BASH_REMATCH[1]}.done.json"; fi
if [[ "$args" == *"gpu.sbatch devrisk"* ]]; then [ -f "$R/teacher/bundle_b012.done.json" ] && [ -f "$R/teacher/bundle_b013.done.json" ] || echo "devrisk before its gate" >> "$STUB_DIR/errors"
  mkdir -p "$RC/devrisk"; echo '{}' > "$RC/devrisk/r7_40c_devrisk.json"; fi
if [[ "$args" == *"cpu.sbatch complete"* ]]; then mkdir -p "$RC"; echo '{"complete": true}' > "$RC/completeness.json"; fi
echo $n
'''
STUB_SQUEUE = r'''#!/usr/bin/env bash
if [[ "$*" == *"-n r7d40_timing"* ]]; then [ -n "$STUB_BTIMING_QUEUED" ] && echo 1035608; exit 0; fi
if [[ "$*" == *"-j"* ]]; then exit 0; fi
c=$(cat "$STUB_DIR/qcalls" 2>/dev/null || echo 0); c=$((c + 1)); echo $c > "$STUB_DIR/qcalls"
n=0; [ "$c" -le "${STUB_FULL_CALLS:-0}" ] && n=27
for i in $(seq 1 $n); do echo "job$i"; done
'''


def run_driver(tmp: Path, protocol_key=True, b40_complete=True, timing_report=True, **env_extra):
    stub = tmp / "stub"; stub.mkdir(exist_ok=True)
    for name, body in (("sbatch", STUB_SBATCH), ("squeue", STUB_SQUEUE)):
        f = stub / name; f.write_text(body); f.chmod(f.stat().st_mode | stat.S_IEXEC)
    d1 = tmp / "D1"; r40 = d1 / "r7_40"; (r40 / "selectors").mkdir(parents=True, exist_ok=True); (r40 / "timing").mkdir(parents=True, exist_ok=True)
    (r40 / "selectors" / "lr_choice.json").write_text(json.dumps(LR))
    if b40_complete:
        for tg in PL.r7_40b_tags(LR):
            (r40 / "selectors" / f"{tg}.done.json").write_text("{}")
    if timing_report:
        (r40 / "timing" / "timing_report.json").write_text(json.dumps({"gpu": "NVIDIA H100 NVL", "host": "nodesumo01", "B_star": 16}))
    proto = tmp / "protocol.yaml"; proto.write_text("r7_40c_staged_freeze: x\n" if protocol_key else "r7_40b_staged_freeze: y\n"); (tmp / "lines").mkdir(exist_ok=True)
    env = dict(os.environ, PATH=f"{stub}:{os.environ['PATH']}", STUB_DIR=str(stub), D1_ROOT=str(d1), R40_ROOT=str(r40), R40C_ROOT=str(r40 / "eval_40c"),
               SUBMIT_LOG=str(tmp / "submit.log"), POLL="0", WAIT_H="1", SOON_S="0", SOON_POLL="1", PY=sys.executable, PROTOCOL=str(proto),
               LINES_DIR=str(tmp / "lines"), CAP="27", TIMING_REPORT=str(r40 / "timing" / "timing_report.json"), **{k: str(v) for k, v in env_extra.items()})
    p = subprocess.run(["bash", str(R / "slurm" / "r7_d1_40c_submit.sh")], env=env, capture_output=True, text=True, timeout=600)
    calls = (stub / "sbatch.log").read_text().splitlines() if (stub / "sbatch.log").exists() else []
    errs = (stub / "errors").read_text() if (stub / "errors").exists() else ""
    return p, (tmp / "submit.log").read_text() if (tmp / "submit.log").exists() else "", calls, errs


class DriverDryRun(unittest.TestCase):
    def test_full_sequence_with_retry(self):
        with tempfile.TemporaryDirectory() as t:
            p, log, calls, errs = run_driver(Path(t), STUB_FULL_CALLS=3, STUB_FLAKY="gqa_dev_2", ROWS="all", QUALITY_PARTITION="L40S")
            self.assertEqual(p.returncode, 0, log + p.stderr); self.assertEqual(errs, "")
            self.assertIn("cpu.sbatch check all", calls[0]); self.assertIn("--wait", calls[0])
            q = [c for c in calls if "gpu.sbatch run " in c]; self.assertEqual(len(q), 13); self.assertEqual(sum("run gqa_dev 2/6" in c for c in q), 2)
            self.assertTrue(all("--partition=L40S --exclude=node51,node60" in c for c in q))
            tb = [c for c in calls if "gpu.sbatch timing_bundle" in c]; self.assertEqual(len(tb), 1); self.assertTrue(tb[0].endswith("timing_bundle report"))
            of = [c for c in calls if "gpu.sbatch official" in c]; self.assertEqual(len(of), 3); self.assertTrue(all("--partition=L40S " in c for c in of))
            self.assertIn("--partition=H100 --nodelist=nodesumo01", tb[0]); self.assertEqual(sum("devteacher" in c for c in calls), 2)
            self.assertEqual(sum("gpu.sbatch devrisk" in c for c in calls), 1); self.assertIn("cpu.sbatch complete", calls[-1])
            self.assertFalse(any("--dependency" in c or "--export" in c for c in calls)); self.assertIn("RETRY 1/2", log)
            self.assertTrue(all("--exclude=node51,node60" in c for c in calls if "--partition=CPU" not in c))
            self.assertIn("done: R7-40c generation complete", log); self.assertIn("scoring is the coordinator's step", log)
            self.assertFalse(any("cpu.sbatch score" in c for c in calls))                    # the driver never scores

    def test_stops(self):
        for kw, msg in (({"protocol_key": False}, "is not frozen"), ({"b40_complete": False}, "missing after"), ({"STUB_CHECK_FAILS": 1}, "stage-0 check failed"),
                        ({"STUB_FAIL_ON": "run textvqa_val 1/4"}, "STOP: no job id"), ({"STUB_FLAKY": "chartqa_val_0", "STUB_FLAKY_TIMES": 5}, "failed 3 times"),
                        ({"timing_report": False, "BSTAR": "report"}, "incomplete (or gated)"), ({"QUALITY_PARTITION": "L40S;x"}, "bad QUALITY_PARTITION"),
                        ({"ROWS": "most"}, "bad ROWS"), ({"BSTAR": "twice"}, "bad BSTAR"),
                        ({"timing_report": False, "STUB_BTIMING_QUEUED": 1, "BSTAR": "in_bundle"}, "still queued")):
            with tempfile.TemporaryDirectory() as t:
                p, log, calls, errs = run_driver(Path(t), **kw)
                self.assertEqual(p.returncode, 1, (kw, log)); self.assertIn(msg, log); self.assertNotIn("done:", log); self.assertEqual(errs, "")

    def test_dev_and_timing_off(self):
        with tempfile.TemporaryDirectory() as t:
            p, log, calls, errs = run_driver(Path(t), DEV=0, TIMING=0, timing_report=False, STUB_BTIMING_QUEUED=1)
            self.assertEqual(p.returncode, 0, log); self.assertFalse(any("devteacher" in c or "devrisk" in c or "gpu.sbatch timing" in c for c in calls))

    def test_bstar_in_bundle_and_report_modes(self):
        with tempfile.TemporaryDirectory() as t:                                              # no report, nothing queued: B* inside the bundle
            p, log, calls, errs = run_driver(Path(t), timing_report=False, BSTAR="in_bundle")
            self.assertEqual(p.returncode, 0, log); self.assertEqual(errs, ""); self.assertEqual(sum("timing_bundle in_bundle" in c for c in calls), 1)
        with tempfile.TemporaryDirectory() as t:                                              # report mode (the default, decision 14) with the report present
            p, log, calls, errs = run_driver(Path(t))
            self.assertEqual(p.returncode, 0, log); self.assertEqual(errs, ""); self.assertEqual(sum("timing_bundle report" in c for c in calls), 1)


class ProcessorBounds(unittest.TestCase):
    """Execution note 40c_1: the load-time check reads the processor's effective bounds (min_pixels / max_pixels attributes and the grids its
    call produces), not image_processor.size, which transformers 4.57.6 leaves at the checkpoint's values."""

    class IP:
        def __init__(self, lo, hi):
            self.size = {"shortest_edge": 3136, "longest_edge": 12845056}
            self.min_pixels, self.max_pixels, self.patch_size, self.merge_size = lo, hi, 14, 2

    class Proc:
        def __init__(self, lo, hi):
            self.image_processor = ProcessorBounds.IP(lo, hi)

        def __call__(self, text, images, return_tensors):
            w, h = images[0].size
            hb, wb = RN.smart_resize(h, w, min_pixels=self.image_processor.min_pixels, max_pixels=self.image_processor.max_pixels)
            return {"image_grid_thw": torch.tensor([[1, hb // 14, wb // 14]])}

    def test_frozen_bounds_applied_pass(self):
        probe = RN.check_processor(self.Proc(256 * 28 * 28, 1280 * 28 * 28), 256 * 28 * 28, 1280 * 28 * 28, "t")
        self.assertEqual([p["grid"] for p in probe], [[14, 19], [14, 19], [30, 41], [21, 58]])     # measured on the real processor (job 1036829)

    def test_checkpoint_bounds_stop(self):
        with self.assertRaises(SystemExit):
            RN.check_processor(self.Proc(3136, 12845056), 256 * 28 * 28, 1280 * 28 * 28, "t")


if __name__ == "__main__":
    unittest.main()
