"""r7 E1 (external Qwen2.5-VL baselines MMTok / HiPrune): stubbed CPU tests of the I/O glue scripts/r7_e1_external.py — configuration names,
the fraction -> token mapping of each method (checked against the fetched official source when it is present), the prediction format read by the
R7-40c loaders (r7_d1_30_score.load_config, r7_d1_40c_score.Bench40), single-writer files, crash / torn-line resume, one-build-per-file identity,
shard markers, the per-fraction interpreters (command, cwd, knob environment), unit lines, stage-2 completeness and the wrappers.  No GPU, no model
weights, no method env: a fake processor / model drive the real Runtime hook and worker loop.  Run: python -I tests/test_r7_e1_external.py -v"""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import numpy as np
import torch

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))
import r7_d1_30_run as RUN  # noqa: E402
import r7_d1_30_score as SC  # noqa: E402
import r7_d1_40_masks as M4  # noqa: E402
import r7_d1_40c_plan as PL  # noqa: E402
import r7_d1_40c_run as RN  # noqa: E402
import r7_d1_40c_score as SCO  # noqa: E402
import r7_e1_external as EX  # noqa: E402

IMG = 99
GRIDS = {"img0": (3, 4), "img1": (4, 4), "img2": (2, 5), "img3": (4, 5), "img4": (5, 6)}


def tiny_sets(d: Path, bench="gqa_dev", n=13):
    d.mkdir(parents=True, exist_ok=True); rows = []
    for i in range(n):
        img = f"img{i % 5}"
        rows.append({"bench": bench, "qid": f"q{i:02d}", "image_id": img, "image_file": img, "question": f"Is thing {i} here?",
                     "prompt": f"Is thing {i} here?\nShort", "answer": "o3" if i % 2 else "no", "answers": ["o3"] * 10,
                     "structural": "verify", "type": "human_test" if i % 2 else "augmented_test"})
    (d / f"{bench}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


# ------------------------------------------------------------------------------------------------------------------------------ fakes
class FakeOut(dict):
    def to(self, dev):
        return self


class FakeProc:
    """The official processor calls: chat template (records the user text), processor (one image token per merged token), decode."""
    def __init__(self):
        self.user_texts = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert not tokenize and add_generation_prompt and messages[0]["role"] == "user" and messages[0]["content"][0] == {"type": "image"}
        t = messages[0]["content"][1]["text"]; self.user_texts.append(t)
        return "<s>" + t

    def __call__(self, text, images, return_tensors):
        rows, cols = images[0]["grid"]
        return FakeOut(input_ids=torch.tensor([[5, 5] + [IMG] * (rows * cols) + [5, 5, 5]]))

    def batch_decode(self, new, skip_special_tokens=True):
        return [f" o{int(new[0][0]) - 100} "]


class FakeLLM(torch.nn.Module):
    def forward(self, inputs_embeds=None, **kw):
        return None


class FakeModel:
    """Simulates a pruning method inside generate: the LLM module is called with the pruned prefill (k_rule tokens kept), then one decode step."""
    def __init__(self, method, f, fail_at=None, fail_after_prefill=False):
        self.method, self.f, self.fail_at, self.calls, self.kwargs, self.fail_after_prefill = method, f, fail_at, 0, [], fail_after_prefill
        self.config = Namespace(image_token_id=IMG); self.generation_config = Namespace(eos_token_id=[7, 8], pad_token_id=8, repetition_penalty=1.05)
        self.llm = FakeLLM(); self.dtype = torch.bfloat16

    def generate(self, input_ids, **kw):
        self.calls += 1; self.kwargs.append(kw)
        if self.fail_at is not None and self.calls >= self.fail_at:
            raise RuntimeError("simulated crash")
        n = int((input_ids == IMG).sum()); L = int(input_ids.shape[1]); k = EX.k_rule(self.method, self.f, n)
        self.llm(inputs_embeds=torch.zeros(1, L - n + k, 2))
        if self.fail_after_prefill and k < n:                                           # HiPrune + eager: the shipped 4D mask fails after the selection
            raise RuntimeError("The size of tensor a (70) must match the size of tensor b (303) at non-singleton dimension 0")
        self.llm(inputs_embeds=torch.zeros(1, 1, 2))
        return Namespace(sequences=torch.cat([input_ids, torch.tensor([[100 + k, 7]])], 1))


def fake_runtime(method, f, attn=None, fail_at=None, processor="FakeProcessorFast", fail_after_prefill=False):
    model = FakeModel(method, f, fail_at, fail_after_prefill)
    info = {"method": method, "fraction": float(f), "attn": attn or EX.METHODS[method]["attn"], "dtype": "torch.bfloat16", "transformers": EX.METHODS[method]["transformers"],
            "torch": "fake", "knobs": {"knob": float(f)}, "load_seconds": 0.0, "bounds": {"image_processor": {"class": processor}},
            "attn_built": {"config": attn or EX.METHODS[method]["attn"]}}
    return EX.Runtime(torch, FakeProc(), model, model.llm, "cpu", info)


def wargs(t: Path, method="mmtok", f="0.25", bench="gqa_dev", shard="0/1", limit=0):
    return Namespace(method=method, fraction=f, bench=bench, shard=shard, sets_dir=str(t / "sets"), out=str(t / "out"), limit=limit, model_dir="/x",
                     proc_dir=None, attn=None, device="cpu", dtype="bfloat16")


def rargs(t: Path, method="mmtok", anchor=False, bench="gqa_dev", shard="0/1", limit=0, model_dir=None, fractions=None):
    return Namespace(method=method, bench=bench, shard=shard, sets_dir=str(t / "sets"), out=str(t / "out"), anchor=anchor, limit=limit,
                     model_dir=model_dir, proc_dir=None, attn=None, device="cpu", dtype="bfloat16", fractions=fractions)


OPEN = (lambda f: {"grid": GRIDS[f]})
GRID = (lambda f: GRIDS[f])


# ------------------------------------------------------------------------------------------------------------------------------ names / budgets
class NamesAndBudgets(unittest.TestCase):
    def test_config_names(self):
        self.assertEqual(EX.config_list(), [f"ext:{m}:{k}" for m in ("mmtok", "hiprune") for k in ("0.125", "0.25", "0.5", "0.75")])
        self.assertEqual(EX.config_list(anchor=True)[4], "ext:mmtok:1"); self.assertEqual(len(EX.config_list(anchor=True)), 10)
        self.assertEqual(RUN.cfg_file("ext:mmtok:0.25"), "ext__mmtok__0.25"); self.assertEqual(RUN.cfg_file("ext:hiprune:1"), "ext__hiprune__1")
        for c in EX.config_list(anchor=True):
            m, f = EX.parse_cfg(c); self.assertEqual(EX.cfg_name(m, f), c)
        for bad in ("ext:foo:0.25", "ext:mmtok:0.3", "sel:x:0.25", "ext:mmtok:0.250", "ext:hiprune:2"):
            with self.assertRaises(ValueError):
                EX.parse_cfg(bad)

    def test_fraction_to_tokens(self):
        cases = {(0.125, 260): (32, 32, 33), (0.125, 300): (37, 38, 38), (0.25, 333): (83, 83, 83), (0.5, 1001): (500, 500, 501),
                 (0.75, 1258): (943, 944, 944), (0.75, 1001): (750, 751, 751), (0.125, 1276): (159, 160, 160)}
        for (f, n), (mm, hp, r40) in cases.items():
            self.assertEqual((EX.k_rule("mmtok", f, n), EX.k_rule("hiprune", f, n), EX.r7_40c_k(f, n)), (mm, hp, r40), (f, n))
        for n in range(256, 1281):
            for f in EX.FRACTIONS:
                mm, hp, r40 = EX.k_rule("mmtok", f, n), EX.k_rule("hiprune", f, n), M4.budget(f, n)
                self.assertTrue(mm == int(np.floor(f * n)) and r40 - 1 <= mm <= r40 and r40 - 1 <= hp <= r40, (f, n))
            self.assertEqual((EX.k_rule("mmtok", 1.0, n), EX.k_rule("hiprune", 1.0, n), EX.r7_40c_k(1.0, n)), (n, n, n))
        with self.assertRaises(ValueError):
            EX.k_rule("other", 0.5, 300)

    def test_rules_match_the_fetched_official_code(self):
        mm, hp = EX.METHODS["mmtok"]["dir"], EX.METHODS["hiprune"]["dir"]
        if not (mm.exists() and hp.exists()):
            self.skipTest("fetched official trees not present")
        fwd = (mm / "mmtok/qwen/qwen2_5_VLmodel_mmtok.py").read_text(); wrap = (mm / "mmtok/qwen/qwen2_5_vl_mmtok.py").read_text()
        sel = (mm / "mmtok/core/semantic_selector.py").read_text(); core = (mm / "mmtok/core/mmtok_core.py").read_text()
        self.assertIn('token_retain_ratio = getattr(self._mmtok_core, "retain_ratio", float(os.environ.get("TOKEN_RETAIN_RATIO", "0.1")))', fwd)
        self.assertIn("target_vision_tokens = int(token_retain_ratio * image_embeds.shape[0])", fwd)
        self.assertIn("mmtok_core.retain_ratio = retain_ratio", wrap); self.assertIn("k_max = min(self.target_vision_tokens, n)", sel)
        self.assertIn("if num_tokens <= self.target_vision_tokens:", core)
        q = (hp / "Qwen2_5_VL/qwen2_5_vl_HiPrune.py").read_text()
        for s in ("OBJECT_LAYER = int(os.environ['HIPRUNE_OBJECT_LAYER'])", "ALPHA = float(os.environ['HIPRUNE_ALPHA'])",
                  "RETAIN = float(os.environ['HIPRUNE_QWEN_RETENTION'])", "visual_token_num = round(n_image_tokens * RETAIN)",
                  "deep_token_num = visual_token_num - shallow_token_indices.shape[0]", "shallow_attention = attn_list[OBJECT_LAYER - 1]"):
            self.assertIn(s, q)
        readme = (hp / "README.md").read_text()
        self.assertIn("|Qwen2.5-VL|HIPRUNE_QWEN_RETENTION|0-1|0.334/0.223/0.112|", readme)
        self.assertIn("|   |HIPRUNE_OBJECT_LAYER|1-24|16|", readme.split("|Qwen2.5-VL|")[1]); self.assertIn("|   |HIPRUNE_ALPHA|0-1|0.1|", readme.split("|Qwen2.5-VL|")[1])

    def test_method_registry_and_fetched_trees(self):
        self.assertEqual({m: s["licence"] for m, s in EX.METHODS.items()}, {"mmtok": "Apache-2.0", "hiprune": "MIT"})
        self.assertEqual((EX.METHODS["mmtok"]["attn"], EX.METHODS["hiprune"]["attn"]), ("sdpa", "flash_attention_2"))
        for m, s in EX.METHODS.items():
            self.assertTrue(s["commit"].startswith(s["dir"].name.split("@")[1]))
            if not s["dir"].exists():
                continue
            self.assertTrue((s["dir"] / "LICENSE").is_file())
            self.assertEqual((s["dir"] / ".git" / "HEAD").read_text().strip(), s["commit"])            # detached at the recorded commit
            self.assertFalse(os.access(s["dir"], os.W_OK)); self.assertFalse(os.access(s["dir"] / s["files"][-1], os.W_OK))   # fetched read-only
            self.assertTrue(all(v and len(v) == 64 for v in EX.method_files_sha(m).values()))


# ------------------------------------------------------------------------------------------------------------------------------ worker / rows
class Worker(unittest.TestCase):
    def test_rows_and_r7_40c_loaders(self):
        """Rows in R7-40c's layout, read by r7_d1_30_score.load_config and by r7_d1_40c_score.Bench40 (coverage guard, kept / N, scores)."""
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); items = tiny_sets(t / "sets"); cfgs = EX.config_list(anchor=True)
            for c in cfgs:
                m, f = EX.parse_cfg(c); rt = fake_runtime(m, f)
                self.assertEqual(EX.worker(wargs(t, m, EX.PL.fkey(f)), rt=rt, opener=OPEN, grid_of=GRID), 0)
                self.assertEqual(rt.proc.user_texts, [it["prompt"] for it in EX.shard_items(t / "sets", "gqa_dev")])   # pinned prompt via the chat template
                for kw in rt.model.kwargs:
                    self.assertEqual(kw, {"do_sample": False, "num_beams": 1, "max_new_tokens": 128, "use_cache": True, "output_logits": True,
                                          "return_dict_in_generate": True})
            od = t / "out" / "gqa_dev"
            self.assertTrue((od / "ext__mmtok__0.25@s0of1.jsonl").exists()); self.assertTrue((od / "_e1_meta" / "ext__hiprune__1@s0of1.json").exists())
            for c in cfgs:
                recs, integ = SC.load_config(od, c); m, f = EX.parse_cfg(c)
                self.assertEqual(len(recs), 13); self.assertEqual((integ["dropped_lines"], integ["duplicate_qids"], len(integ["files"])), (0, 0, 1))
                for it in items:
                    r = recs[it["qid"]]; n = int(np.prod(GRIDS[it["image_id"]])); k = EX.k_rule(m, f, n)
                    self.assertEqual((r["pred"], r["kept"], r["n_vis"], r["f"], r["n_new"], r["hit_max"], r["k_rule"], r["image_id"]),
                                     (f"o{k}", k, n, f, 2, False, k, it["image_id"]))
                    self.assertGreaterEqual(r["ms"], 0.0); self.assertNotIn("answer", r)
            bn = SCO.Bench40("gqa_dev", str(t / "sets"), str(t / "out"), None, cfgs, 50, None)
            for c in cfgs:
                m, f = EX.parse_cfg(c); want = np.mean([EX.k_rule(m, f, np.prod(GRIDS[it["image_id"]])) / np.prod(GRIDS[it["image_id"]]) for it in items])
                self.assertAlmostEqual(bn.kept_fraction(c), want); self.assertEqual(len(bn.S[c]), 13)
            self.assertEqual(float(bn.S["ext:mmtok:1"].mean()), np.mean([float(f"o{np.prod(GRIDS[it['image_id']])}" == it["answer"]) for it in items]))

    def test_crash_torn_line_and_resume(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); tiny_sets(t / "sets"); od = t / "out" / "gqa_dev"; f = od / "ext__hiprune__0.5@s0of1.jsonl"
            with self.assertRaises(RuntimeError):
                EX.worker(wargs(t, "hiprune", "0.5"), rt=fake_runtime("hiprune", 0.5, fail_at=6), opener=OPEN, grid_of=GRID)
            self.assertEqual(len(f.read_text().splitlines()), 5)
            meta = json.loads((od / "_e1_meta" / "ext__hiprune__0.5@s0of1.json").read_text()); self.assertEqual(meta["sessions"][0]["answers"], 5)
            with open(f, "a") as fh:
                fh.write('{"qid": "q0')                                                  # a crash mid-write
            rt = fake_runtime("hiprune", 0.5); EX.worker(wargs(t, "hiprune", "0.5"), rt=rt, opener=OPEN, grid_of=GRID); self.assertEqual(rt.model.calls, 8)
            recs, integ = SC.load_config(od, "ext:hiprune:0.5")
            self.assertEqual((len(recs), integ["dropped_lines"], integ["duplicate_qids"]), (13, 1, 0))
            rt3 = fake_runtime("hiprune", 0.5); EX.worker(wargs(t, "hiprune", "0.5"), rt=rt3, opener=OPEN, grid_of=GRID); self.assertEqual(rt3.model.calls, 0)
            self.assertEqual(len(json.loads((od / "_e1_meta" / "ext__hiprune__0.5@s0of1.json").read_text())["sessions"]), 2)

    def test_one_build_per_file_and_stops(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); tiny_sets(t / "sets"); a = wargs(t, "hiprune", "0.25", limit=4)
            EX.worker(a, rt=fake_runtime("hiprune", 0.25), opener=OPEN, grid_of=GRID)
            with self.assertRaises(SystemExit):                                         # another attention backend for the same file: stop
                EX.worker(wargs(t, "hiprune", "0.25"), rt=fake_runtime("hiprune", 0.25, attn="eager"), opener=OPEN, grid_of=GRID)
            with self.assertRaises(SystemExit):                                         # another image processor class for the same file: stop
                EX.worker(wargs(t, "hiprune", "0.25"), rt=fake_runtime("hiprune", 0.25, processor="SlowProcessor"), opener=OPEN, grid_of=GRID)
            self.assertEqual(json.loads((t / "out" / "gqa_dev" / "_e1_meta" / "ext__hiprune__0.25@s0of1.json").read_text())["identity"]["processor"], "FakeProcessorFast")
            with self.assertRaises(SystemExit):                                         # processor N != header estimate: stop
                EX.worker(wargs(t, "mmtok", "0.25"), rt=fake_runtime("mmtok", 0.25), opener=OPEN, grid_of=lambda f: (1, 1))
            with self.assertRaises(RN.MissingImage):
                EX.worker(wargs(t, "mmtok", "0.5"), rt=fake_runtime("mmtok", 0.5), grid_of=GRID)            # real opener: the file is missing
            rt = fake_runtime("mmtok", 0.75); rt.model.llm = FakeLLM()                  # the hooked module is never called: stop
            with self.assertRaises(SystemExit):
                EX.worker(wargs(t, "mmtok", "0.75"), rt=rt, opener=OPEN, grid_of=GRID)
            with self.assertRaises(SystemExit):
                EX.worker(wargs(t, "mmtok", "0.25", bench="unknown_bench"), rt=fake_runtime("mmtok", 0.25), opener=OPEN, grid_of=GRID)

    def test_probe_records_kept_when_the_llm_fails(self):
        """The CPU probe (HiPrune with eager attention): the selection runs, the shipped LLM mask fails; kept comes from the prefill hook."""
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); sets = t / "sets"; sets.mkdir(); (sets / "img").mkdir(); items = []
            from PIL import Image
            for i, (w, h) in enumerate(EX.SELF_WH):
                fp = sets / "img" / f"i{i}.png"; Image.new("RGB", (w, h)).save(fp)
                items.append({"qid": f"sc{i}", "image_id": f"i{i}", "image_file": str(fp), "question": "q", "prompt": "q" + EX.V.SHORT})
            (sets / "e1_selfcheck.jsonl").write_text("".join(json.dumps(r) + "\n" for r in items))

            class GridProc(FakeProc):
                def __call__(self, text, images, return_tensors):
                    rows, cols = RN.merged_grid(images[0].filename)
                    return FakeOut(input_ids=torch.tensor([[5, 5] + [IMG] * (rows * cols) + [5, 5, 5]]))
            for f in (0.125, 1.0):
                rt = fake_runtime("hiprune", f, fail_after_prefill=True); rt.proc = GridProc()
                buf = io.StringIO()
                with mock.patch.object(EX, "load_runtime", return_value=rt), mock.patch.object(EX.RN, "open_image", side_effect=lambda pth: Image.open(pth)), contextlib.redirect_stdout(buf):
                    EX.probe(Namespace(method="hiprune", fraction=EX.PL.fkey(f), bench="e1_selfcheck", sets_dir=str(sets), model_dir="/m", proc_dir=None, attn="eager"))
                pr = json.loads([x for x in buf.getvalue().splitlines() if x.startswith("PROBE ")][-1][6:])
                for x in pr["items"]:
                    n = int(np.prod(RN.merged_grid(sets / "img" / f"{x['qid'].replace('sc', 'i')}.png")))
                    self.assertEqual((x["n_vis"], x["kept"], x["k_rule"]), (n, EX.k_rule("hiprune", f, n), EX.k_rule("hiprune", f, n)))
                    self.assertEqual(x["error"] is None, f == 1.0)

    def test_ensure_newline(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "x.jsonl"; EX.ensure_newline(p); self.assertFalse(p.exists())
            p.write_text('{"a": 1}\n'); EX.ensure_newline(p); self.assertEqual(p.read_text(), '{"a": 1}\n')
            p.write_text('{"a": 1}\n{"b'); EX.ensure_newline(p); self.assertEqual(p.read_text(), '{"a": 1}\n{"b\n')


# ------------------------------------------------------------------------------------------------------------------------------ unit / markers
class Unit(unittest.TestCase):
    def spawner(self, t, log, fail=None):
        def spawn(a, f, qdir):
            log.append((EX.cfg_name(a.method, f), str(qdir)))
            if fail == f:
                return 3
            return EX.worker(Namespace(**{**vars(a), "fraction": EX.PL.fkey(f), "model_dir": str(qdir)}), rt=fake_runtime(a.method, f), opener=OPEN, grid_of=GRID)
        return spawn

    def test_run_marker_resume(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); tiny_sets(t / "sets", bench="textvqa_val", n=13); log, staged = [], []
            stage = lambda src: (staged.append(src) or (t / "staged", None, "fake stage"))   # noqa: E731
            a = rargs(t, "mmtok", anchor=True, bench="textvqa_val", shard="0/1")
            self.assertEqual(EX.run(a, spawn=self.spawner(t, log), stage=stage, grid_of=GRID), 0)
            self.assertEqual([c for c, _ in log], EX.config_list(("mmtok",), anchor=True)); self.assertEqual(staged, [EX.QH.QWEN_DIR])
            self.assertTrue(all(q == str(t / "staged") for _, q in log))
            mk = json.loads((t / "out" / "textvqa_val" / "done_e1_mmtok_shard0of1.json").read_text())
            self.assertEqual((mk["method"], mk["items"], len(mk["configs"]), mk["answers_this_job"], mk["limit"]), ("mmtok", 13, 5, 65, 0))
            self.assertEqual(mk["method_commit"], EX.METHODS["mmtok"]["commit"]); self.assertEqual(len(mk["code"]["sha256"]), 64)
            for k in ("env", "gpus", "node", "job", "repo", "seconds", "sets_file_sha256", "method_files_sha256", "knob", "k_rule", "sessions", "utc"):
                self.assertIn(k, mk)
            self.assertIn("torch", mk["env"]); self.assertIn("commit", mk["repo"])
            for c, kc in mk["kept_check"].items():
                self.assertEqual((kc["rows"], kc["kept_equals_method_rule"], kc["mismatch_examples"]), (13, 13, []))
            self.assertEqual(mk["kept_check"]["ext:mmtok:1"]["mean_kept_fraction"], 1.0)
            log.clear(); staged.clear(); EX.run(a, spawn=self.spawner(t, log), stage=stage, grid_of=GRID)
            self.assertEqual((log, staged), ([], []))                                   # complete: nothing staged or run, marker rewritten
            self.assertEqual(json.loads((t / "out" / "textvqa_val" / "done_e1_mmtok_shard0of1.json").read_text())["answers_this_job"], 0)

    def test_one_fraction_smoke_and_readback(self):
        """The coordinator's smoke: one fraction on the first items of a shard, then smokecheck reads the rows back with R7-40c's loader."""
        self.assertEqual(EX.run_fractions(Namespace(method="mmtok", fractions=["0.25"], anchor=False)), [0.25])
        self.assertEqual(EX.run_fractions(Namespace(method="hiprune", fractions=None, anchor=True)), [0.125, 0.25, 0.5, 0.75, 1.0])
        with self.assertRaises(ValueError):
            EX.run_fractions(Namespace(method="mmtok", fractions=["0.3"], anchor=False))
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); tiny_sets(t / "sets", bench="gqa_dev", n=40); tiny_sets(t / "sets", bench="chartqa_val", n=20); log = []
            for b, sh in (("gqa_dev", "0/6"), ("chartqa_val", "0/2")):
                a = rargs(t, "mmtok", bench=b, shard=sh, limit=5, model_dir=str(t), fractions=["0.25"]); buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    EX.run(a, spawn=self.spawner(t, log), grid_of=GRID)
                lines = [x for x in buf.getvalue().splitlines() if " kept " in x]; self.assertEqual(len(lines), 5)   # one line per smoke item
                self.assertTrue(all("(rule " in x and "R7-40c " in x for x in lines))
            self.assertEqual([c for c, _ in log], ["ext:mmtok:0.25", "ext:mmtok:0.25"])
            self.assertEqual(sorted(p.name for p in (t / "out" / "gqa_dev").glob("*.jsonl")), ["ext__mmtok__0.25@s0of6.jsonl"])
            sc = Namespace(method="mmtok", out=str(t / "out"), sets_dir=str(t / "sets"), benches=["gqa_dev@0/6", "chartqa_val@0/2"], fractions=["0.25"], anchor=False, limit=5)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(EX.smokecheck(sc), 0)
            res = json.loads(buf.getvalue().split("SMOKECHECK ", 1)[1])
            self.assertTrue(res["pass"]); self.assertEqual(res["benches"]["chartqa_val"]["configs"]["ext:mmtok:0.25"]["rows"], 5)
            self.assertEqual(res["benches"]["gqa_dev"]["configs"]["ext:mmtok:0.25"]["kept_equals_rule"], 5)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(EX.smokecheck(Namespace(**{**vars(sc), "fractions": ["0.5"]})), 8)   # nothing at 0.5: fails
                (t / "out" / "chartqa_val" / "done_e1_mmtok_shard0of2.json").unlink(); self.assertEqual(EX.smokecheck(sc), 8)

    def test_run_stops_without_marker(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); tiny_sets(t / "sets"); od = t / "out" / "gqa_dev"; log = []
            with self.assertRaises(SystemExit):                                         # a worker that fails: stop, no marker
                EX.run(rargs(t, "hiprune", model_dir=str(t)), spawn=self.spawner(t, log, fail=0.5), grid_of=GRID)
            self.assertEqual([c for c, _ in log], ["ext:hiprune:0.125", "ext:hiprune:0.25", "ext:hiprune:0.5"])
            self.assertFalse((od / "done_e1_hiprune_shard0of1.json").exists())
            with self.assertRaises(SystemExit):                                         # a worker that answers nothing: incomplete, no marker
                EX.run(rargs(t, "hiprune", model_dir=str(t)), spawn=lambda a, f, q: 0, grid_of=GRID)
            self.assertFalse((od / "done_e1_hiprune_shard0of1.json").exists())
            staged = []
            with self.assertRaises(RN.MissingImage):                                    # a missing image stops before staging
                EX.run(rargs(t, "mmtok"), spawn=lambda a, f, q: 0, stage=lambda s: staged.append(s))
            self.assertEqual(staged, [])
            with self.assertRaises(SystemExit):
                EX.run(rargs(t, "other"), spawn=lambda a, f, q: 0, grid_of=GRID)

    def test_sharding_is_r7_40c(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); tiny_sets(t / "sets", n=40)
            parts = [EX.shard_items(t / "sets", "gqa_dev", f"{i}/3") for i in range(3)]
            self.assertEqual([[it["qid"] for it in p] for p in parts], [[it["qid"] for it in RUN.load_items(t / "sets", "gqa_dev", f"{i}/3")] for i in range(3)])
            self.assertEqual(sorted(it["qid"] for p in parts for it in p), sorted(f"q{i:02d}" for i in range(40)))
            self.assertTrue(all("answer" not in it and "answers" not in it for p in parts for it in p))
            self.assertEqual(len(EX.shard_items(t / "sets", "gqa_dev", "0/1", limit=7)), 7)

    def test_child_interpreter(self):
        base = {"PATH": "/bin", "HIPRUNE_ALPHA": "0.9", "MMTok_JIT_N_THRESHOLD": "1", "TOKEN_RETAIN_RATIO": "0.3", "SLURM_JOB_ID": "7"}
        e = EX.child_env("hiprune", 0.125, base)
        self.assertEqual({k: e[k] for k in ("HIPRUNE_QWEN_RETENTION", "HIPRUNE_ALPHA", "HIPRUNE_OBJECT_LAYER")}, {"HIPRUNE_QWEN_RETENTION": "0.125", "HIPRUNE_ALPHA": "0.1", "HIPRUNE_OBJECT_LAYER": "16"})
        self.assertNotIn("MMTok_JIT_N_THRESHOLD", e); self.assertNotIn("TOKEN_RETAIN_RATIO", e); self.assertEqual((e["PATH"], e["SLURM_JOB_ID"]), ("/bin", "7"))
        e = EX.child_env("mmtok", 1.0, base)
        self.assertEqual((e["TOKEN_RETAIN_RATIO"], e["SELECTION_METHOD"]), ("1.0", "mmtok")); self.assertFalse(any(k.startswith("HIPRUNE_") for k in e))
        self.assertEqual(float(EX.child_env("hiprune", 0.75, base)["HIPRUNE_QWEN_RETENTION"]), 0.75)
        with mock.patch.object(EX.subprocess, "run", return_value=Namespace(returncode=0)) as sp:
            a = rargs(Path("/tmp/x"), "hiprune"); self.assertEqual(EX.spawn_worker(a, 0.125, Path("/stage/q")), 0)
            cmd, kw = sp.call_args[0][0], sp.call_args[1]
            self.assertEqual(cmd[:4], [sys.executable, "-I", str(R / "scripts" / "r7_e1_external.py"), "worker"])
            self.assertEqual(cmd[cmd.index("--fraction") + 1], "0.125"); self.assertEqual(cmd[cmd.index("--attn") + 1], "flash_attention_2")
            self.assertEqual(cmd[cmd.index("--model-dir") + 1], "/stage/q"); self.assertEqual(cmd[cmd.index("--proc-dir") + 1], "/stage/q")
            self.assertEqual(kw["cwd"], str(R)); self.assertEqual(kw["env"]["HIPRUNE_QWEN_RETENTION"], "0.125")
            self.assertFalse(str(kw["cwd"]).startswith(str(EX.THIRD)))
        with mock.patch.object(EX, "worker", return_value=0) as w:
            EX.main(["worker", "--method", "hiprune", "--fraction", "0.125", "--bench", "gqa_dev", "--shard", "1/6", "--model-dir", "/m"])
            ns = w.call_args[0][0]; self.assertEqual((ns.method, ns.fraction, ns.shard, ns.device, ns.dtype, ns.attn), ("hiprune", "0.125", "1/6", "cuda", "bfloat16", None))


# ------------------------------------------------------------------------------------------------------------------------------ plan / stage 2 / wrappers
class PlanCompleteWrappers(unittest.TestCase):
    def test_unit_lines(self):
        L = EX.unit_lines(Path("/x")); self.assertEqual(len(L), 2 * (6 + 4 + 2))
        for line, mark, gates in L:
            self.assertTrue(line.startswith("sbatch --parsable --job-name=r7e1_")); self.assertIn("--partition=L40S --exclude=node51,node60 ", line)
            self.assertIn("--time=", line); self.assertNotIn("--export", line); self.assertNotIn("--dependency", line); self.assertEqual(gates, [])
            args = line.split("slurm/r7_e1_gpu.sbatch ")[1].split(); self.assertEqual(args[0], "run"); self.assertNotIn(",", " ".join(args))
            m, b, s = args[1:4]; self.assertEqual(mark, Path("/x") / "pred" / b / f"done_e1_{m}_shard{s.replace('/', 'of')}.json")
            self.assertEqual(int(s.split("/")[1]), PL.QUALITY_SHARDS[b]); self.assertIn(f"--time={EX.TIME[(m, b)]} ", line)
        self.assertTrue(all(x[0].endswith(" anchor") for x in EX.unit_lines(Path("/x"), anchor=True)))
        with self.assertRaises(SystemExit):
            EX.unit_lines(Path("/x"), partition="L40S,P100")
        with tempfile.TemporaryDirectory() as t:
            self.assertEqual(EX.plan(Namespace(root=t, lines=str(Path(t) / "lines.txt"), anchor=True, partition="L40S")), 0)
            self.assertEqual(PL.read_configs(Path(t) / "configs" / "configs_e1_chartqa_val.txt"), EX.config_list(anchor=True))
            lines = [x for x in (Path(t) / "lines.txt").read_text().splitlines() if not x.startswith("#")]
            self.assertEqual(len(lines), 24); self.assertTrue(all(" #MARK " in x for x in lines))

    def test_complete(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); pred = t / "pred"
            for b in PL.BENCHES:
                tiny_sets(t / "sets", bench=b, n=7); od = pred / b; od.mkdir(parents=True)
                qids = [json.loads(x)["qid"] for x in (t / "sets" / f"{b}.jsonl").read_text().splitlines()]
                for c in EX.config_list():
                    (od / f"{RUN.cfg_file(c)}@s0of1.jsonl").write_text("".join(json.dumps({"qid": q, "pred": "x"}) + "\n" for q in qids))
                for m in EX.METHODS:
                    for i in range(PL.QUALITY_SHARDS[b]):
                        (od / EX.marker_name(m, f"{i}/{PL.QUALITY_SHARDS[b]}")).write_text("{}")
            a = Namespace(sets_dir=str(t / "sets"), pred_dir=str(pred), out=str(t / "c"), anchor=False, benches=list(PL.BENCHES), methods=list(EX.METHODS))
            self.assertEqual(EX.complete(a), 0)
            a.anchor = True; self.assertEqual(EX.complete(a), 7)
            a.anchor = False; (pred / "chartqa_val" / "done_e1_hiprune_shard1of2.json").unlink(); self.assertEqual(EX.complete(a), 7)
            res = json.loads((t / "c" / "completeness_e1.json").read_text()); self.assertEqual(res["benches"]["chartqa_val"]["markers_missing"], ["done_e1_hiprune_shard1of2.json"])

    def test_wrappers(self):
        g, c = (R / "slurm" / "r7_e1_gpu.sbatch").read_text(), (R / "slurm" / "r7_e1_cpu.sbatch").read_text()
        for s in (g, c):
            self.assertIn("#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out", s); self.assertNotIn("--export", s.replace("(no --export)", ""))
            self.assertIn("cd /home/infres/yinwang/CS_QMI/ssl_pilot", s); self.assertIn('code: $(cat scripts/r7_e1_external.py tests/test_r7_e1_external.py slurm/r7_e1_*', s)
            self.assertNotIn("cd /projects/EEG-foundation-model/yinghao/third_party", s)
        self.assertIn("#SBATCH --exclude=node51,node60", g); self.assertIn("#SBATCH --partition=L40S", g); self.assertIn("#SBATCH --partition=CPU", c)
        self.assertIn('"$PY" -I scripts/r7_e1_external.py run', g); self.assertIn("/home/infres/yinwang/CS_QMI/outputs/R7_E1/smoke/$m", g)
        self.assertIn("--benches gqa_dev@0/6 chartqa_val@0/2", g); self.assertIn('"$PYTHON" -I tests/test_r7_e1_external.py -v', c)
        self.assertIn("--uploaded-prior-to 2026-02-20T14:24:18Z", c); self.assertIn("--uploaded-prior-to 2025-09-24T15:29:24Z", c)
        self.assertIn("transformers==4.52.4", c); self.assertIn("transformers==4.52.0", c); self.assertIn("torch==2.8.0", c)
        r = subprocess.run(["bash", "-n", str(R / "slurm" / "r7_e1_gpu.sbatch")]); self.assertEqual(r.returncode, 0)
        r = subprocess.run(["bash", "-n", str(R / "slurm" / "r7_e1_cpu.sbatch")]); self.assertEqual(r.returncode, 0)


if __name__ == "__main__":
    unittest.main()
