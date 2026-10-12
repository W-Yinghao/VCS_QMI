"""R7-40 (Qwen2.5-VL second host): masks on variable merged grids, host-interface helpers, the spliced prefill / decode path against HF
generate on a tiny random Qwen2.5-VL (vl_baselines env only), teacher bookkeeping with a fake host (exact counts, batching invariance, resume,
coverage, DEV refusal, gates), the readout, the selector adapter (padding, equivalence with the frozen D1-20 selector, exact micro-batch
accumulation, top-k on valid tokens, end-to-end training, resume), the unit plan and the driver (stubbed sbatch / squeue).  CPU only."""
import itertools
import json
import math
import os
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
sys.path.insert(0, str(R / "scripts")); sys.path.insert(0, str(R / "r7" / "package_r7_20261011" / "reference"))
import r7_d1_40_analyze as AN40  # noqa: E402
import r7_d1_40_host as QH  # noqa: E402
import r7_d1_40_masks as M4  # noqa: E402
import r7_d1_40_plan as PL  # noqa: E402
import r7_d1_40_selectors as S4  # noqa: E402
import r7_d1_40_teacher as TE40  # noqa: E402
import r7_d1_40_train as TR40  # noqa: E402
import r7_d1_analyze as AN  # noqa: E402
import r7_d1_data as DA  # noqa: E402
import r7_d1_masks as MK  # noqa: E402
import r7_d1_selectors as SEL  # noqa: E402
import r7_d1_teacher as TE  # noqa: E402

try:
    import transformers  # noqa: F401
    HAS_TF = True
except ImportError:                                                                    # main env: no transformers
    HAS_TF = False

GRIDS = ((14, 19), (17, 23), (15, 16), (24, 24), (3, 4), (5, 5), (9, 30))


# ------------------------------------------------------------------------------------------------------------------------------ masks
class Masks(unittest.TestCase):
    def test_budgets_round_half_up_monotone(self):
        self.assertEqual(M4.budget(0.125, 266), 33); self.assertEqual(M4.budget(0.125, 260), 33); self.assertEqual(M4.budget(0.5, 3), 2)
        self.assertEqual(M4.budget(0.125, 4), 1); self.assertEqual(M4.budget(0.125, 1), 1)
        for n in range(1, 1300):
            ks = [M4.budget(f, n) for f in M4.FRACTIONS]; self.assertTrue(all(a <= b for a, b in zip(ks, ks[1:]))); self.assertLessEqual(ks[-1], n)
        self.assertEqual(M4.budgets(576), {"0.125": 72, "0.25": 144, "0.5": 288, "0.75": 432})

    def test_orderings_permutation_nested_deterministic(self):
        for rows, cols in GRIDS:
            n = rows * cols; sc = np.random.default_rng(n).normal(size=n)
            o1, o2 = M4.orderings("q1", rows, cols, sc), M4.orderings("q1", rows, cols, sc)
            for m in M4.MECHANISMS:
                order = o1[m][0]; self.assertEqual(sorted(order.tolist()), list(range(n))); self.assertTrue(np.array_equal(order, o2[m][0]))
                ks = M4.budgets(n); sets = [set(M4.kept(order, ks[key]).tolist()) for key in M4.FKEYS]
                self.assertEqual([len(x) for x in sets], [ks[key] for key in M4.FKEYS]); self.assertTrue(all(a <= b for a, b in zip(sets, sets[1:])))
                kk = M4.kept(order, ks["0.25"]); self.assertTrue(np.all(np.diff(kk) > 0))                       # raster order
                rk = M4.ranks(order); self.assertTrue(np.array_equal(np.flatnonzero(rk < ks["0.5"]), np.sort(order[:ks["0.5"]])))
            self.assertFalse(np.array_equal(M4.orderings("q2", rows, cols, sc)["random"][0], o1["random"][0]))
            self.assertFalse(np.array_equal(MK.order_random("q1")[0][:10], o1["random"][0][:10]) and n == 576)   # separate seed stream

    def test_grid_on_24x24_recovers_frozen_structure(self):
        for qid in ("a", "b", "c"):
            order, lp = M4.order_grid(qid, 24, 24); first = set(M4.kept(order, 144).tolist())
            self.assertIn(first, [set(MK.lattice(ox, oy).tolist()) for ox in (0, 1) for oy in (0, 1)])
            r, c = np.divmod(M4.kept(order, 72), 24); self.assertEqual(len(set((((r // 2) + (c // 2)) % 2).tolist())), 1)
            for key, frozen in zip(M4.FKEYS, (-math.log(8), -math.log(4), -math.log(6), -math.log(4))):
                self.assertAlmostEqual(lp[key], frozen, places=10)

    def test_grid_logp_exact_by_enumeration(self):
        """Every reachable kept set at every k on small odd / even grids: the mechanism's exact probability (48 configurations × uniform
        subsets of the boundary block) sums to 1 and equals exp(grid_logp)."""
        for rows, cols in ((3, 4), (5, 5), (2, 7)):
            n = rows * cols
            for k in range(1, n):
                dist = {}
                for perm, par in M4.CONFIGS:
                    blocks = M4.grid_blocks(rows, cols, perm, par); cum, j = 0, 0
                    while j < len(blocks) and cum + len(blocks[j]) <= k:
                        cum += len(blocks[j]); j += 1
                    base = set(int(t) for b in blocks[:j] for t in b); r = k - cum
                    subs = [()] if r == 0 else list(itertools.combinations(blocks[j].tolist(), r))
                    for sub in subs:
                        key = frozenset(base | set(sub)); dist[key] = dist.get(key, 0.0) + 1.0 / 48 / len(subs)
                self.assertAlmostEqual(sum(dist.values()), 1.0, places=12)
                for S, p in dist.items():
                    order = np.array(sorted(S) + sorted(set(range(n)) - S)); lp = M4.grid_logp(rows, cols, order, {"x": k})["x"]
                    self.assertAlmostEqual(math.exp(lp), p, places=10, msg=f"{rows}x{cols} k={k}")

    def test_random_and_qsim_logp(self):
        order, lp = M4.order_random("x", 14, 19)
        for key, k in M4.budgets(266).items():
            self.assertAlmostEqual(lp[key], -math.log(math.comb(266, k)), places=6)
        sc = np.random.default_rng(1).normal(size=266); o, lq = M4.order_qsim("x", sc, 266); z = (sc - sc.mean()) / (sc.std() + 1e-12)
        ref = MK.pl_prefix_logp(z, o, tuple(M4.budgets(266).values()))
        self.assertEqual(lq, {key: ref[k] for key, k in M4.budgets(266).items()})
        with self.assertRaises(ValueError):
            M4.order_qsim("x", sc[:-1], 266)


# ------------------------------------------------------------------------------------------------------------------------------ host helpers
class HostHelpers(unittest.TestCase):
    def test_split_and_prune(self):
        pre, post = QH.split_image_segment([1, 2, 99, 3, 4, 5], 99); self.assertEqual(pre.tolist(), [1, 2]); self.assertEqual(post.tolist(), [3, 4, 5])
        for bad in ([1, 2, 3], [99, 1, 99]):
            with self.assertRaises(ValueError):
                QH.split_image_segment(bad, 99)
        rows, cols, s, n_post = 3, 4, 2, 3; n = rows * cols; r, c = np.divmod(np.arange(n), cols)
        pos = torch.as_tensor(np.concatenate([np.tile(np.arange(s), (3, 1)), np.stack([np.full(n, s), s + r, s + c]),
                                              np.tile(s + max(rows, cols) + np.arange(n_post), (3, 1))], 1))
        kept = np.array([0, 5, 11]); pp = QH.prune_positions(pos, s, n, kept)
        self.assertEqual(pp.shape, (3, s + 3 + n_post)); self.assertTrue(torch.equal(pp[:, s:s + 3], pos[:, s + kept]))
        self.assertTrue(torch.equal(pp[:, -n_post:], pos[:, -n_post:])); self.assertEqual(QH.next_position(pp), QH.next_position(pos))
        self.assertTrue(torch.equal(QH.prune_positions(pos, s, n, np.arange(n)), pos))
        with self.assertRaises(ValueError):
            QH.kept_columns(s, n, n_post, [0, 0])

    def test_repetition_penalty_semantics(self):
        sc = torch.tensor([[1.0, -2.0, 0.5, 3.0], [-1.0, 2.0, -0.5, 0.0]]); out = QH.repetition_penalty(sc, [torch.tensor([0, 1, 1]), torch.tensor([3])], 1.05)
        exp = sc.clone(); exp[0, 0] = 1.0 / 1.05; exp[0, 1] = -2.0 * 1.05; exp[1, 3] = 0.0
        self.assertTrue(torch.allclose(out, exp)); self.assertTrue(torch.equal(QH.repetition_penalty(sc, [torch.tensor([0])] * 2, 1.0), sc))
        if HAS_TF:
            from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor
            g = torch.Generator().manual_seed(0); s2 = torch.randn(3, 50, generator=g); ids = torch.randint(0, 50, (3, 7), generator=g)
            ref = RepetitionPenaltyLogitsProcessor(1.05)(ids, s2.clone()); self.assertTrue(torch.equal(QH.repetition_penalty(s2, list(ids), 1.05), ref))


def tiny_qwen():
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
    torch.manual_seed(0)
    cfg = Qwen2_5_VLConfig(text_config=dict(vocab_size=160, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                                            num_key_value_heads=2, max_position_embeddings=2048, rope_theta=10000.0, rms_norm_eps=1e-6,
                                            rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]}),
                           vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64, patch_size=14,
                                              spatial_merge_size=2, temporal_patch_size=2, window_size=56, fullatt_block_indexes=[1], tokens_per_second=2),
                           image_token_id=150, video_token_id=151, vision_start_token_id=152, vision_end_token_id=153, attn_implementation="sdpa")
    return Qwen2_5_VLForConditionalGeneration(cfg).eval()


@unittest.skipUnless(HAS_TF, "needs transformers (vl_baselines env)")
class TinyQwenEquivalence(unittest.TestCase):
    """The spliced path (QwenCore) against the official model on a tiny random Qwen2.5-VL (fp32, CPU): rope ids, prefill logits, greedy
    decoding with the repetition penalty (single and left-padded batches), forced length, pruned positions."""

    @classmethod
    def setUpClass(cls):
        cls.model = tiny_qwen(); cls.core = QH.QwenCore(cls.model, [7, 8], 9, 1.05); g = torch.Generator().manual_seed(3); cls.items = []
        for gh, gw, n_pre, n_post in ((4, 6, 5, 7), (6, 4, 3, 9), (2, 8, 6, 4)):
            thw = torch.tensor([1, gh, gw]); n = gh * gw // 4
            pre = torch.cat([torch.randint(10, 140, (n_pre - 1,), generator=g), torch.tensor([152])])
            post = torch.cat([torch.tensor([153]), torch.randint(10, 140, (n_post - 1,), generator=g)])
            pv = torch.randn(gh * gw, 3 * 2 * 14 * 14, generator=g); ids = torch.cat([pre, torch.full((n,), 150), post])
            cls.items.append({"thw": thw, "n": n, "pre": pre, "post": post, "pv": pv, "ids": ids})

    def spliced(self, it, kept=None):
        f = self.core.visual_tokens(it["pv"], it["thw"][None])[0]; n = it["n"]; kept = np.arange(n) if kept is None else kept
        pos = self.core.full_positions(it["pre"], n, it["post"], it["thw"])
        seq = torch.cat([self.core.embed_ids(it["pre"]), f[torch.as_tensor(kept)], self.core.embed_ids(it["post"])])
        return seq, QH.prune_positions(pos, len(it["pre"]), n, kept), pos

    def official(self, it, **kw):
        with torch.inference_mode():
            return self.model.generate(input_ids=it["ids"][None], attention_mask=torch.ones(1, len(it["ids"]), dtype=torch.long), pixel_values=it["pv"],
                                       image_grid_thw=it["thw"][None], do_sample=False, num_beams=1, repetition_penalty=1.05, eos_token_id=[7, 8],
                                       pad_token_id=9, output_logits=True, return_dict_in_generate=True, **kw)

    def test_rope_ids_structure(self):
        for it in self.items:
            _, _, pos = self.spliced(it); s = len(it["pre"]); rows, cols = self.core.grid_of(it["thw"]); n = it["n"]; r, c = np.divmod(np.arange(n), cols)
            self.assertTrue(np.array_equal(pos[:, :s].numpy(), np.tile(np.arange(s), (3, 1))))
            self.assertTrue(np.array_equal(pos[:, s:s + n].numpy(), np.stack([np.full(n, s), s + r, s + c])))
            self.assertTrue(np.array_equal(pos[:, s + n:].numpy(), np.tile(s + max(rows, cols) + np.arange(len(it["post"])), (3, 1))))
            ref, delta = self.model.model.get_rope_index(it["ids"][None], image_grid_thw=it["thw"][None], attention_mask=torch.ones(1, len(it["ids"])))
            self.assertTrue(torch.equal(ref[:, 0], pos)); self.assertEqual(int(delta) + len(it["ids"]), QH.next_position(pos))

    def test_prefill_matches_official_forward(self):
        seqs, poss = [], []
        for it in self.items:
            seq, pp, _ = self.spliced(it); seqs.append(seq); poss.append(pp)
            with torch.inference_mode():
                ref = self.model(input_ids=it["ids"][None], pixel_values=it["pv"], image_grid_thw=it["thw"][None]).logits[:, -1].float()
            self.assertTrue(torch.allclose(self.core.last_logits([seq], [pp]), ref, atol=1e-5, rtol=1e-5))
        batched = self.core.last_logits(seqs, poss)                                         # right padding changes nothing
        for i, (s, p) in enumerate(zip(seqs, poss)):
            self.assertTrue(torch.allclose(batched[i], self.core.last_logits([s], [p])[0], atol=1e-5, rtol=1e-5))

    def test_decode_matches_hf_generate(self):
        for kw in ({"max_new_tokens": 6}, {"max_new_tokens": 6, "min_new_tokens": 6}):
            singles = []
            for it in self.items:
                out = self.official(it, **kw); new_ref = out.sequences[0, len(it["ids"]):].tolist(); seq, pp, _ = self.spliced(it)
                new, first = self.core.generate([seq], [pp], [it["ids"]], max_new_tokens=kw["max_new_tokens"], min_new_tokens=kw.get("min_new_tokens", 0))
                self.assertEqual(new[0][:len(new_ref)], new_ref); self.assertTrue(all(t == 9 for t in new[0][len(new_ref):]))
                self.assertTrue(torch.allclose(first[0], out.logits[0][0].float(), atol=1e-5, rtol=1e-5)); singles.append(new_ref)
            seqs = [self.spliced(it)[0] for it in self.items]; poss = [self.spliced(it)[1] for it in self.items]
            new, _ = self.core.generate(seqs, poss, [it["ids"] for it in self.items], max_new_tokens=kw["max_new_tokens"], min_new_tokens=kw.get("min_new_tokens", 0))
            for got, ref in zip(new, singles):                                             # left-padded batch == official batch-1 answers
                self.assertEqual(got[:len(ref)], ref)
            if "min_new_tokens" in kw:
                self.assertTrue(all(len(x) == 6 for x in singles))

    def test_pruned_sequences(self):
        it = self.items[0]; n = it["n"]; kept = np.array([0, 2, 5]); seq, pp, pos = self.spliced(it, kept)
        self.assertEqual(seq.shape[0], len(it["pre"]) + 3 + len(it["post"])); self.assertEqual(QH.next_position(pp), QH.next_position(pos))
        new, _ = self.core.generate([seq], [pp], [torch.cat([it["pre"], torch.full((3,), 150), it["post"]])], max_new_tokens=3, min_new_tokens=3)
        self.assertEqual(len(new[0]), 3)
        full_seq, full_pp, _ = self.spliced(it, np.arange(n)); seq0, pp0, _ = self.spliced(it)
        self.assertTrue(torch.equal(full_seq, seq0) and torch.equal(full_pp, pp0))


# ------------------------------------------------------------------------------------------------------------------------------ teacher (fake host)
class FakeCore:
    merge, image_token_id = 2, 99

    def __init__(self, fail_at=None):
        self.torch, self.calls, self.fail_at = torch, 0, fail_at

    def embed_ids(self, ids):
        e = torch.zeros(len(ids), 6); e[:, 2] = 1.0
        return e

    def full_positions(self, pre, n_vis, post, thw):
        rows, cols = int(thw[1]) // 2, int(thw[2]) // 2; s = len(pre); r, c = np.divmod(np.arange(n_vis), cols)
        return torch.as_tensor(np.concatenate([np.tile(np.arange(s), (3, 1)), np.stack([np.full(n_vis, s), s + r, s + c]),
                                               np.tile(s + max(rows, cols) + np.arange(len(post)), (3, 1))], 1))

    def text_positions(self, L):
        return torch.arange(L).view(1, -1).expand(3, -1).clone()

    def last_logits(self, seqs, positions):
        self.calls += 1
        if self.fail_at is not None and self.calls >= self.fail_at:
            raise RuntimeError("simulated crash")
        out = torch.zeros(len(seqs), 4)
        for i, (s, p) in enumerate(zip(seqs, positions)):
            assert tuple(p.shape) == (3, s.shape[0])
            out[i, 1] = s[:, 0].sum() / 100.0
        return out


class FakeQwen:
    """CPU stand-in for QwenPrefill: dim 0 of an embedding flags a visual token, so P(yes) is an exact function of the visual tokens seen;
    images are dicts {'grid': (rows, cols)}."""
    yes_ids, no_ids, load_seconds, image_token_id, info, stage_notes = [1], [2], 0.0, 99, {}, []

    def __init__(self, fail_at=None):
        self.torch, self.core = torch, FakeCore(fail_at)

    def segments(self, prompt, with_image=True):
        ids = np.full(3 + len(prompt) % 4, 5)
        return (ids[:2], ids[2:]) if with_image else (ids, ids[:0])

    def question_vector(self, q):
        return torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    def encode(self, images):
        out = []
        for im in images:
            rows, cols = im["grid"]; n = rows * cols; f = torch.zeros(n, 6); f[:, 0] = 1.0; f[:, 1] = (torch.arange(n, dtype=torch.float32) + 1) / n
            out.append((f, (rows, cols), torch.tensor([1, 2 * rows, 2 * cols])))
        return out


def grid_of_image(image_id):
    k = int(str(image_id).replace("img", "")) % 4
    return ((3, 4), (4, 4), (2, 5), (4, 5))[k]


def synth_rows(n_img=70, seed=0):
    g = np.random.default_rng(seed); rows = []
    for i in range(n_img):
        for j in range(int(g.integers(2, 7))):
            ans = ["yes", "no", "red", "left"][int(g.integers(4))]
            rows.append({"qid": f"{i}_{j}", "image_id": f"img{i}", "question": f"Is thing {j} here?", "answer": ans,
                         "structural": "verify" if ans in ("yes", "no") else "query", "semantic": "attr", "detailed": "x"})
    return rows


class TeacherBookkeeping(unittest.TestCase):
    def test_variants_exact_counts_and_batching_invariance(self):
        host = FakeQwen(); items = [{"qid": f"q{i}", "image_id": f"img{i % 4}", "question": "q" * i, "C": 1} for i in range(8)]
        imgs = {f"img{i}": {"grid": grid_of_image(f"img{i}")} for i in range(4)}
        a = TE40.run_items(host, items, imgs, max_tokens=64, max_seqs=5); b = TE40.run_items(FakeQwen(), items, imgs, 10 ** 6, 1000)
        self.assertEqual(a, b)
        p = lambda n: math.exp(n / 100.0) / (math.exp(n / 100.0) + 3.0)  # noqa: E731
        for r in a:
            n = r["n_vis"]; self.assertEqual(n, int(np.prod(r["grid"]))); self.assertEqual(r["budgets"], M4.budgets(n))
            self.assertAlmostEqual(r["full"][0], p(n), places=5); self.assertAlmostEqual(r["text"][0], p(0), places=5)
            for m in M4.MECHANISMS:
                for key in M4.FKEYS:
                    self.assertAlmostEqual(r["masked"][m][key][0], p(r["budgets"][key]), places=5)
            q = TE.unb64(r["qsim_order"]); self.assertEqual(sorted(q.tolist()), list(range(n)))
            rows, cols = r["grid"]
            self.assertEqual(TE.digest(M4.order_grid(r["qid"], rows, cols, False)[0]), r["order_digest"]["grid"])

    def _bundle_setup(self, root, n_img=60):
        man, _ = DA.build_manifest(iter(synth_rows(n_img)), 20, 4, 4, 2)
        (root / "d1_manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in man))
        (root / "d1_manifest_meta.json").write_text(json.dumps({"manifest_sha256": DA.sha256_file(root / "d1_manifest.jsonl")}))
        plan = DA.plan_shards(man, images_per_shard=5, shards_per_bundle=2); (root / "d1_shards.json").write_text(json.dumps(plan))
        return man, plan                                         # bundles: [CAL-000, FIT-000], [FIT-001, FIT-002], [FIT-003, DEV-000]

    def test_bundle_gates_dev_crash_resume_coverage(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); man, plan = self._bundle_setup(root)
            self.assertEqual(plan["bundles"], {"b000": ["CAL-000", "FIT-000"], "b001": ["FIT-001", "FIT-002"], "b002": ["FIT-003", "DEV-000"]})

            class Src:
                def __init__(self, bad=()):
                    self.bad = set(bad)

                def open(self, i):
                    if i in self.bad:
                        raise OSError("unreadable")
                    return {"grid": grid_of_image(i)}
            bad = plan["shards"]["FIT-001"][0]
            a = Namespace(bundle="b001", manifest_dir=str(root), out=str(root / "teacher"), images_per_chunk=2, max_tokens=4096, max_seqs=16, allow_dev=False)
            with self.assertRaises(SystemExit):                                                    # gate: unfrozen manifest
                TE40.cmd_bundle(a, host_factory=lambda: FakeQwen(), image_source=Src([bad]))
            DA.cmd_freeze(Namespace(out=str(root), protocol="R7-40a-test", go="unit test"))
            with self.assertRaises(SystemExit):                                                    # gate: probe report missing
                TE40.cmd_bundle(Namespace(**vars(a), probe_report=str(root / "nope.json")), host_factory=lambda: FakeQwen(), image_source=Src([bad]))
            (root / "probe.json").write_text(json.dumps({"pass": False}))
            with self.assertRaises(SystemExit):                                                    # gate: probe report not passing
                TE40.cmd_bundle(Namespace(**vars(a), probe_report=str(root / "probe.json")), host_factory=lambda: FakeQwen(), image_source=Src([bad]))
            with self.assertRaises(RuntimeError):                                                  # crash mid-shard
                TE40.cmd_bundle(a, host_factory=lambda: FakeQwen(fail_at=9), image_source=Src([bad]))
            self.assertFalse((root / "teacher" / "bundle_b001.done.json").exists())
            TE40.cmd_bundle(a, host_factory=lambda: FakeQwen(), image_source=Src([bad]))           # resume
            self.assertTrue((root / "teacher" / "bundle_b001.done.json").exists())
            for shard in plan["bundles"]["b001"]:
                d = json.loads((root / "teacher" / f"{shard}.done.json").read_text()); self.assertEqual(d["duplicates"], 0)
                n_bad = sum(1 for r in man if r["image_id"] == bad and shard == "FIT-001"); self.assertEqual(d["present"] + n_bad, d["expected"])
            with self.assertRaises(SystemExit):                                                    # DEV sealed: a bundle holding a DEV shard is refused
                TE40.cmd_bundle(Namespace(**{**vars(a), "bundle": "b002"}), host_factory=lambda: FakeQwen(), image_source=Src())
            self.assertFalse((root / "teacher" / "FIT-003.jsonl").exists())
            TE40.cmd_bundle(Namespace(**{**vars(a), "bundle": "b000"}), host_factory=lambda: FakeQwen(), image_source=Src())
            rows, cov = AN.load_role(root / "teacher", root, "CAL"); self.assertEqual(cov["present"], cov["expected"])
            res = AN40.analyse40(rows, {"w_yes": 1.0, "w_no": 1.0}); self.assertEqual(res["n_questions"], len(rows))
            self.assertLess(max(c["identity_max_abs_error"] for c in res["cells"].values()), 1e-12)
            sha = json.loads((root / "d1_manifest_meta.json").read_text())["manifest_sha256"]
            self.assertEqual(AN40.cmd_complete(Namespace(teacher_dir=str(root / "teacher"), manifest_dir=str(root), frozen_sha256=sha, roles=["CAL"])), 0)
            self.assertEqual(AN40.cmd_complete(Namespace(teacher_dir=str(root / "teacher"), manifest_dir=str(root), frozen_sha256=sha, roles=["CAL", "FIT"])), 7)
            with self.assertRaises(SystemExit):
                AN40.cmd_readout(Namespace(teacher_dir=str(root / "teacher"), manifest_dir=str(root), roles=["DEV"], allow_dev=False))

    def test_images_from_extracted_folder_only(self):
        """Coordinator 2026-10-12: the image archives were deleted; R7-40 reads the extracted folder only and stops on a missing file."""
        from PIL import Image
        with tempfile.TemporaryDirectory() as t:
            Image.new("RGB", (40, 30), (10, 20, 30)).save(Path(t) / "123.jpg"); src = TE40.ExtractedImageSource(t)
            self.assertEqual(src.open("123").size, (40, 30))
            with self.assertRaises(TE40.MissingImage) as cm:
                src.open("456")
            self.assertIn("never reads a zip", str(cm.exception))
            root = Path(t) / "m"; root.mkdir(); man, plan = self._bundle_setup(root); DA.cmd_freeze(Namespace(out=str(root), protocol="t", go="t"))

            class Missing:
                def open(self, i):
                    raise TE40.MissingImage(f"missing {i}")
            with self.assertRaises(TE40.MissingImage):                                           # a missing image stops the bundle (no skip)
                TE40.cmd_bundle(Namespace(bundle="b001", manifest_dir=str(root), out=str(root / "teacher"), images_per_chunk=2, max_tokens=4096,
                                          max_seqs=16, allow_dev=False), host_factory=lambda: FakeQwen(), image_source=Missing())
            for i in sorted({r["image_id"] for r in man if r["role"] == "CAL"}):
                Image.new("RGB", (8, 8)).save(Path(t) / f"{i}.jpg")
            rc = TE40.cmd_imagecheck(Namespace(images=t, manifest_dir=str(root), out=str(root / "ic"), roles=["CAL", "FIT"]))
            rep = json.loads((root / "ic" / "imagecheck.json").read_text())
            self.assertEqual(rc, 3); self.assertEqual(rep["roles"]["CAL"]["missing"], 0); self.assertGreater(rep["roles"]["FIT"]["missing"], 0)
        for f in sorted((R / "scripts").glob("r7_d1_40_*.py")):                                   # no archive access anywhere in R7-40
            code = f.read_text(); self.assertNotIn("import zipfile", code, f.name); self.assertNotIn("TE.ImageSource(", code, f.name)

    def test_s4_rules(self):
        self.assertFalse(TE40.s4_pass([0.99, 0.05], "min")); self.assertTrue(TE40.s4_pass([0.99] * 19 + [0.05], "rate:0.9"))
        self.assertFalse(TE40.s4_pass([0.99] * 8 + [0.05] * 2, "rate:0.9"))
        with self.assertRaises(ValueError):
            TE40.s4_pass([1.0], "nope")


class Analysis(unittest.TestCase):
    def test_known_values(self):
        mk = lambda pm: {m: {key: pm for key in M4.FKEYS} for m in M4.MECHANISMS}  # noqa: E731
        rows = [{"qid": "a", "image_id": "i1", "C": 1, "structural": "verify", "semantic": "rel", "full": (0.9, 0.1), "text": (0.5, 0.5),
                 "masked": mk((0.6, 0.4)), "n_vis": 266, "budgets": M4.budgets(266)},
                {"qid": "b", "image_id": "i2", "C": -1, "structural": "logical", "semantic": "obj", "full": (0.03, 0.07), "text": (0.5, 0.5),
                 "masked": mk((0.5, 0.5)), "n_vis": 280, "budgets": M4.budgets(280)}]
        out = AN40.analyse40(rows, {"w_yes": 1.0, "w_no": 1.0}); c = out["cells"]["random:0.125"]
        Tb, Ta, C = np.array([0.8, -0.4]), np.array([0.2, 0.0]), np.array([1.0, -1.0]); h = (C - Ta) ** 2 - (C - Tb) ** 2; d = (Tb - Ta) ** 2
        self.assertAlmostEqual(c["mean_h"], h.mean(), places=12); self.assertAlmostEqual(c["mean_d"], d.mean(), places=12)
        self.assertAlmostEqual(c["budget_tokens"], (33 + 35) / 2); self.assertEqual(c["budget_fraction"], 0.125)
        g = np.random.default_rng(5); more = []
        for i in range(12):                                                                # secondary: low-mass items excluded and counted by type
            C = 1 if i % 2 else -1; pf = g.uniform(0.1, 0.9); mass = 0.3 if i < 2 else 0.99
            more.append({"qid": f"m{i}", "image_id": f"i{i // 2}", "C": C, "structural": "logical" if i < 2 else "verify", "semantic": "obj",
                         "full": (pf * mass, (1 - pf) * mass), "text": (0.5, 0.5), "masked": mk((g.uniform(0.2, 0.8), 0.5)), "n_vis": 280, "budgets": M4.budgets(280)})
        sec = AN40.secondary_mass40({"CAL": more}, {"w_yes": 1.0, "w_no": 1.0})
        self.assertEqual(sec["excluded_per_role"]["CAL"], {"excluded": 2, "of": 12}); self.assertEqual(sec["low_mass_by_type"]["CAL"], {"logical/obj": {"low": 2, "of": 2}})
        self.assertEqual(sec["tables"]["CAL"]["n_questions"], 10)
        self.assertIn("Low-mass items by type (CAL): logical/obj 2/2", AN40.markdown({"roles": {"CAL": out}, "secondary_mass_ge_0p9": sec}))


# ------------------------------------------------------------------------------------------------------------------------------ selectors
TINY = dict(emb_dim=8, vis_dim=6, d=16, heads=2, layers=1, ffn=32, dropout=0.0, q_per_step=16, masks_per_q=3, warmup=5, max_rows=8, max_cols=8)
SYN_GRIDS = ((3, 4), (4, 4), (2, 5), (4, 5))


def synth_data(n=48, seed=0, grids=SYN_GRIDS):
    """Variable-N tuples: the label lives in visual token 0; h / d from a teacher that sees the label iff token 0 is kept."""
    g = np.random.default_rng(seed); ns = [grids[i % len(grids)][0] * grids[i % len(grids)][1] for i in range(n)]
    cols = [grids[i % len(grids)][1] for i in range(n)]; offsets = np.concatenate([[0], np.cumsum(ns)[:-1]])
    feats = g.normal(size=(sum(ns), 6)).astype(np.float32); C = np.where(feats[offsets, 0] > 0, 1, -1); nmax = max(ns)
    rank = np.full((n, 3, nmax), S4.PAD_RANK, np.int16); kk = np.array([[M4.budget(f, ns[i]) for f in M4.FRACTIONS] for i in range(n)])
    h = np.zeros((n, 12), np.float32); d = np.zeros((n, 12), np.float32)
    for i in range(n):
        for m in range(3):
            rank[i, m, :ns[i]] = M4.ranks(g.permutation(ns[i]))
            for j in range(4):
                b = 0.8 * C[i]; a = b if rank[i, m, 0] < kk[i, j] else 0.0; h[i, 4 * m + j] = (C[i] - a) ** 2 - (C[i] - b) ** 2; d[i, 4 * m + j] = (b - a) ** 2
    q_ids = g.integers(1, 30, size=(n, 5)); q_len = g.integers(3, 6, size=n); emb = torch.tensor(g.normal(size=(30, 8)), dtype=torch.float32)
    return S4.TupleData40(torch.tensor(feats), offsets, ns, cols, np.arange(n), q_ids, q_len, emb, rank, kk, C, h, d, "cpu", 8, 8)


class SelectorAdapter(unittest.TestCase):
    def test_batch_masks_and_padding_invisible(self):
        data = synth_data(12, 0); b = data.batch(np.arange(12), all_masks=True); Q, M, Nb = b["keep"].shape
        self.assertEqual((M, Nb), (12, 20))
        for i in range(Q):
            n = int(b["n"][i]); self.assertFalse(b["keep"][i, :, n:].any()); self.assertTrue(b["valid"][i, :n].all() and not b["valid"][i, n:].any())
            for m in range(3):
                for j in range(4):
                    k = int(data.kk[i, j]); self.assertEqual(int(b["keep"][i, 4 * m + j].sum()), k)
                    self.assertTrue(torch.equal(b["keep"][i, 4 * m + j, :n], data.rank[i, m, :n].long() < k))
        mi = np.array([[0, 5, 11]] * 12); bs = data.batch(np.arange(12), mi); self.assertTrue(torch.equal(bs["keep"], b["keep"][:, [0, 5, 11]]))
        perm = np.array([7, 2, 11, 0]); mp = np.array([[1, 4, 9], [0, 3, 6], [11, 2, 5], [8, 7, 10]]); bp = data.batch(perm, mp)   # global rows
        for i, (q, ms) in enumerate(zip(perm, mp)):
            self.assertTrue(torch.equal(bp["h"][i], data.h[q, torch.as_tensor(ms)]) and torch.equal(bp["d"][i], data.d[q, torch.as_tensor(ms)]))
            self.assertTrue(torch.equal(bp["keep"][i, :, :int(data.ntok[q])], data.batch([q], all_masks=True)["keep"][0, torch.as_tensor(ms)]))
            self.assertEqual(int(bp["C"][i]), int(data.C[q]))
        torch.manual_seed(0); m = S4.Selector40(S4.Cfg40(**TINY), "vcs").eval()
        with torch.no_grad():                                                              # a padded row equals the same row alone
            pb_all = m.encode(b["q_emb"], b["q_pad"], b["v"], S4.full_bias(b["valid"]), b["rows"], b["cols"])[1]
            one = data.batch([0], all_masks=True); pb_one = m.encode(one["q_emb"], one["q_pad"], one["v"], S4.full_bias(one["valid"]), one["rows"], one["cols"])[1]
            self.assertTrue(torch.allclose(pb_all[0], pb_one[0], atol=1e-5))
            keep = b["keep"][:, 3]; v2 = b["v"].clone(); v2[~keep] += 100.0                  # dropped tokens change nothing
            p1 = m.encode(b["q_emb"], b["q_pad"], b["v"], SEL.hard_bias(keep), b["rows"], b["cols"])[1]
            p2 = m.encode(b["q_emb"], b["q_pad"], v2, SEL.hard_bias(keep), b["rows"], b["cols"])[1]
            self.assertTrue(torch.allclose(p1, p2, atol=1e-5))

    def test_equivalence_with_frozen_selector(self):
        """Constant 4 × 4 grid: with the frozen 16 × d table set to row + col, the adapter reproduces the frozen encoder, every objective's
        loss and its gradients (positions: grad_row[r] = Σ_c grad_pos[r·4 + c])."""
        cf = SEL.Cfg(emb_dim=8, vis_dim=6, n_vis=16, grid=4, d=16, heads=2, layers=1, ffn=32, dropout=0.0)
        c40 = S4.Cfg40(**{**TINY, "max_rows": 4, "max_cols": 4})
        data = synth_data(8, 1, grids=((4, 4),)); qi = np.arange(8); mi = np.array([[0, 5, 11]] * 8); b = data.batch(qi, mi)
        bf = {k: b[k] for k in ("q_emb", "q_pad", "v", "keep", "C", "h", "d")}; wts = torch.tensor([1.1, 0.9])
        for obj in ("vcs", "logistic", "direct"):
            torch.manual_seed(2); m40 = S4.Selector40(c40, obj); mf = SEL.Selector(cf, obj)
            sd = {k: v for k, v in m40.state_dict().items() if k not in ("row_pos", "col_pos")}; r_, c_ = np.divmod(np.arange(16), 4)
            sd["pos"] = (m40.row_pos[r_] + m40.col_pos[c_]).detach().clone(); mf.load_state_dict(sd)
            Cq = b["C"].float(); W_b = torch.where(Cq > 0, wts[0], wts[1]).sum()
            l40 = S4.estimator_terms(m40, b, wts, W_b, 3 * W_b); lf = SEL.estimator_loss(mf, bf, wts)
            self.assertAlmostEqual(float(l40), float(lf), places=5); l40.backward(); lf.backward()
            for name, p in mf.named_parameters():
                if name == "pos":
                    gp = p.grad.view(4, 4, -1); self.assertTrue(torch.allclose(m40.row_pos.grad, gp.sum(1), atol=1e-5)); self.assertTrue(torch.allclose(m40.col_pos.grad, gp.sum(0), atol=1e-5))
                elif p.grad is not None:
                    self.assertTrue(torch.allclose(dict(m40.named_parameters())[name].grad, p.grad, atol=1e-5), (obj, name))

    def test_microbatch_accumulation_is_exact(self):
        """Any partition of a step into micro-batches gives the single-batch loss and gradient (global normalisers; dropout off)."""
        data = synth_data(16, 2); qi = np.arange(16); mi = np.stack([np.random.default_rng(i).choice(12, 3, replace=False) for i in qi]); wts = torch.tensor([1.0, 1.0])
        for obj in ("vcs", "logistic", "direct"):
            torch.manual_seed(4); m = S4.Selector40(S4.Cfg40(**TINY), obj); grads = []
            Cq = data.C[torch.as_tensor(qi)].float(); W_b = torch.where(Cq > 0, wts[0], wts[1]).sum()
            parts = [[np.arange(16)], S4.plan_groups(data.lengths[qi], 4, 1.0), S4.plan_groups(data.lengths[qi], 4, 3000.0),
                     S4.plan_groups(data.lengths[qi], 4, 1e18, overhead=0.0), [np.array([3, 1]), np.arange(4, 16), np.array([0, 2])]]
            self.assertEqual(len(parts[1]), 16)
            for groups in parts:
                m.zero_grad(set_to_none=True); tot = 0.0
                for g in groups:
                    lg = S4.estimator_terms(m, data.batch(qi[g], mi[g]), wts, W_b, 3 * W_b); lg.backward(); tot += float(lg.detach())
                grads.append((tot, {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}))
            for tot, gr in grads[1:]:
                self.assertAlmostEqual(tot, grads[0][0], places=5)
                for n in gr:
                    self.assertTrue(torch.allclose(gr[n], grads[0][1][n], atol=1e-6, rtol=1e-4), (obj, n))

    def test_plan_groups(self):
        L = np.array([30, 10, 25, 12, 40, 11]); groups = S4.plan_groups(L, 5, 5 * 2 * 26 ** 2)
        self.assertEqual(sorted(int(i) for g in groups for i in g), list(range(6)))
        for g in groups:
            self.assertTrue(len(g) == 1 or len(g) * 5 * L[g].max() ** 2 <= 5 * 2 * 26 ** 2)
        flat = [int(i) for g in groups for i in g]; self.assertEqual(flat, list(np.argsort(L, kind="stable")))
        big = np.array([300] * 60 + [619] * 2 + [1009] * 2); gr = S4.plan_groups(big, 5, S4.Cfg40().cost_train)  # GQA-like step
        self.assertEqual([len(x) for x in gr], [60, 2, 2])

    def test_plan_groups_optimal_contiguous_partition(self):
        g = np.random.default_rng(0)
        for trial in range(40):
            n = int(g.integers(1, 8)); L = g.integers(5, 60, size=n); s, cap, ov = 5, float(g.choice([2e3, 2e4, 1e9])), float(g.choice([0.0, 50.0]))
            groups = S4.plan_groups(L, s, cap, gamma=0.01, overhead=ov); order = np.argsort(L, kind="stable")
            self.assertEqual([int(i) for gr in groups for i in gr], [int(i) for i in order])            # contiguous in the sorted order
            for gr in groups:
                self.assertTrue(len(gr) == 1 or len(gr) * s * L[gr].max() ** 2 <= cap)
            cost = lambda gs: sum(len(x) * s * (L[x].max() + 0.01 * L[x].max() ** 2) + ov for x in gs)  # noqa: E731
            best = np.inf                                                                              # brute force over all cuts
            for mask in range(2 ** max(n - 1, 0)):
                cuts = [0] + [k + 1 for k in range(n - 1) if mask >> k & 1] + [n]; gs = [order[a:b] for a, b in zip(cuts, cuts[1:])]
                if all(len(x) == 1 or len(x) * s * L[x].max() ** 2 <= cap for x in gs):
                    best = min(best, cost(gs))
            self.assertAlmostEqual(cost(groups), best, places=6)

    def test_topk_on_valid_tokens(self):
        g = torch.randn(4, 20); valid = torch.arange(20)[None] < torch.tensor([20, 12, 16, 10])[:, None]; k = torch.tensor([5, 12, 2, 7])
        hk = S4.hard_topk40(g, k, valid); self.assertEqual(hk.sum(1).tolist(), k.tolist()); self.assertFalse((hk & ~valid).any())
        for r in range(4):
            nv = int(valid[r].sum()); self.assertEqual(set(torch.topk(g[r, :nv], int(k[r])).indices.tolist()), set(torch.nonzero(hk[r]).flatten().tolist()))
        soft = S4.relaxed_topk40(g, k, 1e-4, valid, torch.Generator().manual_seed(0))
        self.assertTrue(torch.allclose((soft * valid).sum(1), k.float(), atol=1e-2))

    def test_end_to_end_all_objectives_resume_native(self):
        fit, cal = synth_data(48, 0), synth_data(24, 1)
        for obj, name, better in (("vcs", "J", "higher"), ("logistic", "bce", "lower"), ("direct", "mse_h", "lower")):
            cfg = S4.Cfg40(**TINY, steps_a=40, steps_b=10, eval_every=10, lr=3e-3, seed=0)
            model, log = S4.train40(cfg, obj, fit, cal, (1.0, 1.0))
            self.assertEqual([e["step"] for e in log["evals"]], [10, 20, 30, 40]); self.assertIn(log["selected_step"], (10, 20, 30, 40))
            self.assertAlmostEqual(log["selected_cal_spearman"], max(e["cal_spearman_r_h"] for e in log["evals"]), places=12)
            self.assertTrue(all(e["native"]["name"] == name and e["native"]["better"] == better for e in log["evals"]))
            self.assertEqual(set(log["soft_hard_gap"]), {"0.125", "0.25", "0.5", "0.75"})
            dyn = S4.dynamic_tau40(model, cal, cfg, 0.5); self.assertLessEqual(abs(dyn["mean_kept_fraction"] - 0.5), 0.26)
            R_ = S4.risk_grid(model, cal, cfg)
            if obj != "direct":
                self.assertTrue(np.allclose(R_[:, -1], 0.0, atol=1e-6))                     # all tokens kept: T_A = T_B
        cfg = S4.Cfg40(**TINY, steps_a=40, steps_b=8, eval_every=10, lr=3e-3, seed=5); fit, cal = synth_data(32, 2), synth_data(16, 3)
        straight, _ = S4.train40(cfg, "vcs", fit, cal, (1.0, 1.0))
        with tempfile.TemporaryDirectory() as t:
            rp = Path(t) / "unit.resume.pt"
            S4.train40(cfg, "vcs", fit, cal, (1.0, 1.0), resume_path=rp, stop_after=20)
            resumed, log = S4.train40(cfg, "vcs", fit, cal, (1.0, 1.0), resume_path=rp)
        self.assertEqual(log["resumed_at"], [20])
        for (k1, v1), (k2, v2) in zip(straight.state_dict().items(), resumed.state_dict().items()):
            self.assertTrue(torch.equal(v1, v2), k1)

    def test_assemble_from_teacher_records(self):
        host = FakeQwen(); items = [{"qid": f"q{i}", "image_id": f"img{i % 4}", "question": "q" * (i + 1), "C": 1 if i % 2 else -1} for i in range(8)]
        imgs = {f"img{i}": {"grid": grid_of_image(f"img{i}")} for i in range(4)}; recs = TE40.run_items(host, items, imgs)
        index = {"images": [f"img{i}" for i in range(4)], "n": [int(np.prod(grid_of_image(f"img{i}"))) for i in range(4)],
                 "grid": [list(grid_of_image(f"img{i}")) for i in range(4)]}
        qid_ids = {it["qid"]: list(range(1, 2 + i)) for i, it in enumerate(items)}
        img_of, q_ids, q_len, rank, kk, C, h, d = TR40.assemble40(recs, index, qid_ids)
        for i, r in enumerate(recs):
            n = r["n_vis"]; self.assertTrue((rank[i, :, n:] == S4.PAD_RANK).all()); self.assertEqual(kk[i].tolist(), [r["budgets"][k] for k in M4.FKEYS])
            q = TE.unb64(r["qsim_order"]); self.assertTrue(np.array_equal(np.flatnonzero(rank[i, 2, :n] < kk[i, 1]), np.sort(q[:kk[i, 1]])))
            tb = TR40.T(r["full"]); ta = TR40.T(r["masked"]["grid"]["0.5"]); self.assertAlmostEqual(float(h[i, 6]), (r["C"] - ta) ** 2 - (r["C"] - tb) ** 2, places=6)
        bad = json.loads(json.dumps(recs)); bad[0]["grid"] = [1, int(np.prod(bad[0]["grid"]))]
        with self.assertRaises(RuntimeError):
            TR40.assemble40(bad, index, qid_ids)


# ------------------------------------------------------------------------------------------------------------------------------ plan and driver
class Plan(unittest.TestCase):
    def test_lines(self):
        plan = {"bundles": {"b000": ["CAL-000"], "b001": ["FIT-000", "FIT-001"], "b002": ["DEV-000"]}}
        tl = PL.teacher_lines(Path("/x"), plan); self.assertEqual([m.name for _, m in tl], ["bundle_b000.done.json", "bundle_b001.done.json"])
        s1 = PL.stage1(Path("/x")); s2 = PL.stage2(Path("/x"), {o: {"chosen_lr": 1e-4} for o in PL.OBJECTIVES})
        self.assertEqual(len(s1), 6); self.assertEqual(len(s2), 6)
        for line, mark in tl + s1 + s2 + PL.probe_lines(Path("/x"), "rate:0.9") + PL.cache_lines(Path("/x")):
            self.assertNotIn("--export", line); self.assertIn("--exclude=node51,node52,node60", line); self.assertNotIn("--dependency", line)
            self.assertTrue(line.startswith("sbatch --parsable"))
            self.assertIn("--partition=RTX6000PRO,H100" + (",L40S" if "r7_d1_40_gpu" in line else " "), line + " ")
        self.assertTrue(all(" 0.0001 " in ln for ln, _ in s2)); self.assertTrue(all(m.name.endswith(".done.json") for _, m in s1 + s2))
        pr = PL.probe_lines(Path("/x"), "rate:0.9", "same_gpu", "H100")[0][0]
        self.assertTrue(pr.endswith("gpu.sbatch probe rate:0.9 same_gpu") and "--partition=H100 --exclude=node51,node52,node60" in pr)
        self.assertIn("--partition=L40S ", PL.teacher_lines(Path("/x"), plan, "L40S")[0][0])
        self.assertIn("--partition=RTX6000PRO,H100,L40S ", PL.stage1(Path("/x"), "RTX6000PRO,H100,L40S")[0][0])
        for bad in ("A100", "H100 --export=X", "", "L40S;rm"):
            with self.assertRaises(SystemExit):
                PL.teacher_lines(Path("/x"), plan, bad)


STUB_SBATCH = r'''#!/usr/bin/env bash
echo "$*" >> "$STUB_DIR/sbatch.log"
n=$(cat "$STUB_DIR/counter" 2>/dev/null || echo 2000); n=$((n + 1)); echo $n > "$STUB_DIR/counter"
args="$*"; R=$R40_ROOT
if [ -n "$STUB_FAIL_ON" ] && [[ "$args" == *"$STUB_FAIL_ON"* ]]; then exit 1; fi
if [[ "$args" =~ gpu\.sbatch\ probe\ ([^ ]+) ]]; then mkdir -p "$R/probe"; p=true; [ "${BASH_REMATCH[1]}" = "min" ] && [ -n "$STUB_PROBE_FAILS_MIN" ] && p=false
  echo "{\"pass\": $p}" > "$R/probe/probe_report.json"; fi
if [[ "$args" =~ gpu\.sbatch\ bundle\ (b[0-9]+) ]]; then b=${BASH_REMATCH[1]}; c=$(cat "$STUB_DIR/tries_$b" 2>/dev/null || echo 0); echo $((c + 1)) > "$STUB_DIR/tries_$b"
  if [ "$b" != "$STUB_FLAKY" ] || [ "$c" -ge "${STUB_FLAKY_TIMES:-1}" ]; then mkdir -p "$R/teacher"; echo '{}' > "$R/teacher/bundle_$b.done.json"; fi; fi
if [[ "$args" == *"gpu.sbatch cache"* ]]; then mkdir -p "$R/cache"; echo '{}' > "$R/cache/cache_d1_40.json"; fi
if [[ "$args" == *"cpu.sbatch complete"* ]]; then mkdir -p "$R/teacher"; echo '{"complete": true}' > "$R/teacher/r7_40a_completeness.json"; fi
if [[ "$args" == *"cpu.sbatch readout"* ]]; then echo '{}' > "$R/teacher/r7_40a_results.json"; fi
if [[ "$args" == *"cpu.sbatch choose"* ]]; then "$PY_STUB" "$REPO/scripts/r7_d1_40_plan.py" --root "$R" choose --selectors-dir "$R/selectors" >> "$STUB_DIR/choose.log" 2>&1 || exit 1; fi
if [[ "$args" =~ train\ ([a-z]+)\ ([0-9.e-]+)\ ([0-9]+) ]]; then t="${BASH_REMATCH[1]}_lr${BASH_REMATCH[2]}_s${BASH_REMATCH[3]}"; mkdir -p "$R/selectors"
  v=$(python3 -c "import zlib; print(zlib.crc32(b'$t') % 1000 / 1000)"); echo "{\"selected_cal_spearman\": $v}" > "$R/selectors/$t.json"
  touch "$R/selectors/$t.pt"; echo '{}' > "$R/selectors/$t.done.json"; fi
echo $n
'''
STUB_SQUEUE = r'''#!/usr/bin/env bash
if [[ "$*" == *"-j"* ]]; then exit 0; fi
c=$(cat "$STUB_DIR/qcalls" 2>/dev/null || echo 0); c=$((c + 1)); echo $c > "$STUB_DIR/qcalls"
n=0; [ "$c" -le "${STUB_FULL_CALLS:-0}" ] && n=27
for i in $(seq 1 $n); do echo "job$i"; done
'''


def run_driver(tmp: Path, stage, protocol_key=True, **env_extra):
    stub = tmp / "stub"; stub.mkdir(exist_ok=True)
    for name, body in (("sbatch", STUB_SBATCH), ("squeue", STUB_SQUEUE)):
        f = stub / name; f.write_text(body); f.chmod(f.stat().st_mode | stat.S_IEXEC)
    d1 = tmp / "D1"; (d1 / "data").mkdir(parents=True, exist_ok=True); (d1 / "data" / "d1_manifest_meta.json").write_text('{"frozen": true}')
    (d1 / "data" / "d1_shards.json").write_text(json.dumps({"bundles": {f"b{i:03d}": ([f"CAL-{i:03d}"] if i < 2 else [f"FIT-{i:03d}"]) for i in range(12)}
                                                            | {"b012": ["DEV-000"], "b013": ["DEV-001"]}}))
    proto = tmp / "protocol.yaml"; proto.write_text(("r7_40a_staged_freeze: x\nr7_40b_staged_freeze: y\n" if protocol_key else "nothing: 1\n"))
    (tmp / "lines").mkdir(exist_ok=True)
    env = dict(os.environ, PATH=f"{stub}:{os.environ['PATH']}", STUB_DIR=str(stub), D1_ROOT=str(d1), R40_ROOT=str(d1 / "r7_40"), SUBMIT_LOG=str(tmp / "submit.log"),
               POLL="0", WAIT_H="1", SOON_S="0", SOON_POLL="1", PY=sys.executable, PY_STUB=sys.executable, REPO=str(R), PROTOCOL=str(proto),
               LINES_DIR=str(tmp / "lines"), CAP="27", **{k: str(v) for k, v in env_extra.items()})
    p = subprocess.run(["bash", str(R / "slurm" / "r7_d1_40_submit.sh"), stage], env=env, capture_output=True, text=True, timeout=300)
    calls = (stub / "sbatch.log").read_text().splitlines() if (stub / "sbatch.log").exists() else []
    return p, (tmp / "submit.log").read_text() if (tmp / "submit.log").exists() else "", calls, d1


class DriverDryRun(unittest.TestCase):
    def test_40a_full_sequence_with_retry(self):
        with tempfile.TemporaryDirectory() as t:
            p, log, calls, d1 = run_driver(Path(t), "40a", STUB_FULL_CALLS=3, STUB_FLAKY="b005", S4_RULE="rate:0.9", S1_RULE="same_gpu",
                                           PROBE_PARTITION="H100", TEACHER_PARTITION="L40S")
            self.assertEqual(p.returncode, 0, log + p.stderr)
            self.assertIn("--partition=H100 --exclude=node51,node52,node60 slurm/r7_d1_40_gpu.sbatch probe rate:0.9 same_gpu", calls[0])
            bundles = [c for c in calls if "gpu.sbatch bundle" in c]; self.assertTrue(all("--partition=L40S " in c for c in bundles))
            self.assertEqual(len(bundles), 13); self.assertEqual(sum("bundle b005" in c for c in bundles), 2)        # 12 bundles + 1 retry
            self.assertFalse(any("bundle b012" in c or "bundle b013" in c for c in calls))                          # DEV not in 40a
            self.assertTrue(bundles[0].endswith("bundle b000") and bundles[1].endswith("bundle b001"))             # CAL first
            self.assertFalse(any("--dependency" in c or "--export" in c for c in calls))
            self.assertTrue(all("--exclude=node51,node52,node60" in c for c in calls if "--partition=CPU" not in c))
            self.assertIn("cpu.sbatch complete", calls[-2]); self.assertIn("cpu.sbatch readout", calls[-1]); self.assertIn("RETRY 1/2", log)
            self.assertIn("done: R7-40a complete", log)

    def test_40a_stops(self):
        for kw, msg in (({"protocol_key": False}, "is not frozen"), ({"STUB_PROBE_FAILS_MIN": 1}, "launch probe did not pass"),
                        ({"STUB_FAIL_ON": "bundle b003"}, "STOP: no job id"), ({"STUB_FLAKY": "b004", "STUB_FLAKY_TIMES": 5}, "failed 3 times")):
            with tempfile.TemporaryDirectory() as t:
                p, log, calls, _ = run_driver(Path(t), "40a", **kw)
                self.assertEqual(p.returncode, 1, (kw, log)); self.assertIn(msg, log); self.assertNotIn("done:", log)

    def test_40b_sequence(self):
        with tempfile.TemporaryDirectory() as t:
            p0, log0, _, d1 = run_driver(Path(t), "40b")
            self.assertEqual(p0.returncode, 1); self.assertIn("R7-40a is not complete", log0)
            (d1 / "r7_40" / "teacher").mkdir(parents=True); (d1 / "r7_40" / "teacher" / "r7_40a_completeness.json").write_text('{"complete": true}')
            (Path(t) / "stub" / "sbatch.log").unlink(missing_ok=True)
            p, log, calls, d1 = run_driver(Path(t), "40b")
            self.assertEqual(p.returncode, 0, log + p.stderr); self.assertIn("gpu.sbatch cache", calls[0])
            trains = [c for c in calls if " train " in c]; self.assertEqual(len(trains), 12)
            self.assertTrue(all("--partition=RTX6000PRO,H100 " in c for c in trains)); self.assertIn("cpu.sbatch choose", calls[7])
            ch = json.loads((d1 / "r7_40" / "selectors" / "lr_choice.json").read_text()); self.assertEqual(set(ch), {"vcs", "logistic", "direct"})
            for c in trains[6:]:
                o = c.split(" train ")[1].split()[0]; self.assertIn(f" {ch[o]['chosen_lr']:g} ", c + " ")
            self.assertFalse(any("--dependency" in c for c in calls)); self.assertIn("done: R7-40b complete", log)


if __name__ == "__main__":
    unittest.main()
