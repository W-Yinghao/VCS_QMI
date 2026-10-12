"""R7-30 D1-10: mask mechanisms, data roles, teacher bookkeeping (fake host, CPU), resume / coverage, analysis.  Main env or SparseVLM env."""
import itertools
import json
import math
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts")); sys.path.insert(0, str(R / "r7" / "package_r7_20261011" / "reference"))
import r7_d1_analyze as AN  # noqa: E402
import r7_d1_data as DA  # noqa: E402
import r7_d1_masks as MK  # noqa: E402
import r7_d1_teacher as TE  # noqa: E402
import vcs_vl_reference as REF  # noqa: E402


class Masks(unittest.TestCase):
    def test_permutation_nested_counts_deterministic(self):
        sc = np.random.default_rng(0).normal(size=MK.N)
        for qid in ("q1", "q2"):
            o1, o2 = MK.orderings(qid, sc), MK.orderings(qid, sc)
            for m in MK.MECHANISMS:
                order = o1[m][0]; self.assertEqual(sorted(order.tolist()), list(range(MK.N))); self.assertTrue(np.array_equal(order, o2[m][0]))
                sets = [set(MK.kept(order, k).tolist()) for k in MK.BUDGETS]
                self.assertEqual([len(x) for x in sets], list(MK.BUDGETS))
                self.assertTrue(all(a <= b for a, b in zip(sets, sets[1:])))
                self.assertTrue(np.all(np.diff(MK.kept(order, 144)) > 0))                         # raster order kept
        self.assertFalse(np.array_equal(MK.orderings("q1", sc)["random"][0], MK.orderings("q2", sc)["random"][0]))

    def test_grid_structure_and_set_probabilities(self):
        for qid in ("a", "b", "c"):
            order = MK.order_grid(qid)[0]; first = set(MK.kept(order, 144).tolist())
            self.assertIn(first, [set(MK.lattice(ox, oy).tolist()) for ox in (0, 1) for oy in (0, 1)])
            r, c = np.divmod(MK.kept(order, 72), MK.GRID); par = ((r // 2) + (c // 2)) % 2; self.assertEqual(len(set(par.tolist())), 1)
        # independent enumeration of the mechanism's randomness: 4! lattice orders × 2 checkerboard parities
        cnt = {k: {} for k in (72, 144, 288, 432)}
        for perm in itertools.permutations(range(4)):
            for parity in (0, 1):
                for k, key in ((72, (perm[0], parity)), (144, (perm[0],)), (288, frozenset(perm[:2])), (432, frozenset(perm[:3]))):
                    cnt[k][key] = cnt[k].get(key, 0) + 1
        lp = MK.order_grid("x")[1]
        for k in cnt:
            p = next(iter(cnt[k].values())) / 48; self.assertAlmostEqual(lp[k], math.log(p), places=12)
            self.assertEqual(len(set(cnt[k].values())), 1)

    def test_random_and_pl_logp(self):
        lp = MK.order_random("x")[1]
        for k in MK.BUDGETS:
            self.assertAlmostEqual(lp[k], -math.log(math.comb(MK.N, k)), places=6)
        z = np.array([0.3, -1.0, 2.0, 0.1]); tot = {k: 0.0 for k in (1, 2, 3, 4)}; seen = {k: set() for k in (1, 2, 3, 4)}
        for perm in itertools.permutations(range(4)):                                      # brute force: PL probabilities sum to 1
            lps = MK.pl_prefix_logp(z, perm, (1, 2, 3, 4))
            for k in (1, 2, 3, 4):
                if perm[:k] not in seen[k]:
                    seen[k].add(perm[:k]); tot[k] += math.exp(lps[k])
        for k in tot:
            self.assertAlmostEqual(tot[k], 1.0, places=12)
        p = np.exp(z) / np.exp(z).sum(); self.assertAlmostEqual(math.exp(MK.pl_prefix_logp(z, (2, 0, 1, 3), (1,))[1]), p[2], places=12)


def synth_rows(n_img=70, seed=0):
    g = np.random.default_rng(seed); rows = []
    for i in range(n_img):
        for j in range(int(g.integers(2, 7))):
            ans = ["yes", "no", "red", "left"][int(g.integers(4))]
            rows.append({"qid": f"{i}_{j}", "image_id": f"img{i}", "question": f"Is thing {j} here?", "answer": ans,
                         "structural": "verify" if ans in ("yes", "no") else "query", "semantic": "attr", "detailed": "x"})
    return rows


class DataRoles(unittest.TestCase):
    def test_roles_exact_disjoint_deterministic(self):
        rows = synth_rows(); man, meta = DA.build_manifest(iter(rows), n_fit=20, n_cal=4, n_dev=4, per_image=2)
        man2, _ = DA.build_manifest(iter(list(reversed(rows))), n_fit=20, n_cal=4, n_dev=4, per_image=2)
        self.assertEqual([(r["qid"], r["role"]) for r in man], [(r["qid"], r["role"]) for r in man2])                 # input order irrelevant
        self.assertEqual({k: v["images"] for k, v in meta["counts"].items()}, {"FIT": 20, "CAL": 4, "DEV": 4})
        REF.assert_group_disjoint({k: [r["image_id"] for r in man if r["role"] == k] for k in ("FIT", "CAL", "DEV")})
        self.assertTrue(all(r["answer"] in ("yes", "no") and r["C"] in (1, -1) for r in man))
        per = {}
        for r in man:
            per[r["image_id"]] = per.get(r["image_id"], 0) + 1
        self.assertLessEqual(max(per.values()), 2)
        w = meta["class_weights_from_FIT"]; fit = [r for r in man if r["role"] == "FIT"]
        self.assertAlmostEqual(sum(w["w_yes"] if r["C"] > 0 else w["w_no"] for r in fit if r["C"] > 0) / len(fit), 0.5, places=12)

    def test_llava_exclusion_and_shards_feed(self):
        rows = synth_rows(); binq = next(r for r in rows if r["answer"] in ("yes", "no"))
        with tempfile.TemporaryDirectory() as t:
            mix = Path(t) / "mix.json"
            mix.write_text(json.dumps([{"image": f"gqa/images/{binq['image_id']}.jpg", "conversations": [
                {"from": "human", "value": "<image>\n" + binq["question"] + DA.SHORT}, {"from": "gpt", "value": "Yes"}]},
                {"image": "coco/train2017/1.jpg", "conversations": [{"from": "human", "value": binq["question"]}]}]))
            seen, st = DA.llava_seen_pairs(mix, with_stats=True); self.assertEqual(seen, {(binq["image_id"], DA.norm_q(binq["question"]))})
            self.assertEqual((st["mix665k_gqa_conversations"], st["mix665k_gqa_human_turns"]), (1, 1))
            man, meta = DA.build_manifest(iter(rows), 20, 4, 4, 2, seen=seen)
            self.assertNotIn(binq["qid"], {r["qid"] for r in man}); self.assertGreaterEqual(meta["excluded_llava_seen_qa"], 1)
            self.assertEqual(meta["llava_exclusion"]["seen_pairs_matching_a_train_balanced_question"], 1)
        plan = DA.plan_shards(man, images_per_shard=3, shards_per_bundle=2)
        self.assertTrue(next(iter(plan["shards"])).startswith("CAL-"))
        allimg = [i for v in plan["shards"].values() for i in v]; self.assertEqual(len(allimg), len(set(allimg))); self.assertEqual(len(allimg), 28)
        self.assertEqual(sorted(s for v in plan["bundles"].values() for s in v), sorted(plan["shards"]))
        for line in DA.feed_lines(plan):
            self.assertNotIn("--export", line); self.assertIn("--partition=H100,L40S", line); self.assertIn("--exclude=node51,node52,node60", line)


class ClosedSources(unittest.TestCase):
    def test_only_train_member_is_opened(self):
        import zipfile
        with tempfile.TemporaryDirectory() as t:
            z = Path(t) / "questions1.2.zip"
            with zipfile.ZipFile(z, "w") as zz:
                zz.writestr("train_balanced_questions.json", json.dumps({"1": {"imageId": "i1", "question": "Is it red?", "answer": "yes", "types": {}}}))
                zz.writestr("testdev_balanced_questions.json", "NOT JSON: must never be opened")
                zz.writestr("test_balanced_questions.json", "NOT JSON: must never be opened")
            self.assertEqual([r["qid"] for r in DA.read_questions(z)], ["1"])
            for bad in ("testdev_balanced_questions.json", "closed/x.json", "val_balanced_questions.json"):
                with self.assertRaises(SystemExit):
                    list(DA.read_questions(Path(t) / bad))


class TrainAllAndBalance(unittest.TestCase):
    def test_train_all_reader_opens_only_allow_listed_files(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t) / "train_all_questions"; d.mkdir()
            for n in range(2):
                (d / f"train_all_questions_{n}.json").write_text(json.dumps({f"{n}a": {"imageId": "i1", "question": "Is it red?", "answer": "no", "types": {}}}))
            (d / "testdev_all_questions.json").write_text("NOT JSON: must never be opened")
            self.assertEqual(sorted(r["qid"] for r in DA.read_questions(d, "all")), ["0a", "1a"])
            with self.assertRaises(SystemExit):
                list(DA.read_questions(Path(t) / "closed" / "train_all_questions", "all"))

    def test_balanced_selection_band_cap_determinism(self):
        g = np.random.default_rng(3); rows = []
        for i in range(400):
            for j in range(int(g.integers(1, 8))):
                ans = "yes" if g.uniform() < 0.75 else "no"                                    # strong yes skew
                rows.append({"qid": f"{i}_{j}", "image_id": f"img{i}", "question": f"q{j}?", "answer": ans, "structural": "verify",
                             "semantic": "s", "detailed": "d"})
        man, meta = DA.build_manifest(iter(rows), 100, 20, 20, 4, balance=True)
        man2, _ = DA.build_manifest(iter(list(reversed(rows))), 100, 20, 20, 4, balance=True)
        self.assertEqual([(r["qid"], r["role"]) for r in man], [(r["qid"], r["role"]) for r in man2])
        for role in ("FIT", "CAL", "DEV"):
            sel = [r for r in man if r["role"] == role]; pi = sum(r["C"] == 1 for r in sel) / len(sel)
            self.assertTrue(0.48 <= pi <= 0.52, (role, pi)); self.assertAlmostEqual(meta["balance"][role]["pi_yes"], pi, places=12)
            per = {}
            for r in sel:
                per[r["image_id"]] = per.get(r["image_id"], 0) + 1
            self.assertEqual(len(per), meta["counts"][role]["images"]); self.assertLessEqual(max(per.values()), 4); self.assertGreaterEqual(min(per.values()), 1)
        self.assertEqual(DA.pick_alternating(["y1", "y2"], ["n1"], 4, start_yes=False), ["n1", "y1", "y2"])


class FiniteL1(unittest.TestCase):
    def test_identities_and_calibrated_full_teacher(self):
        import r7_d1_l1 as L1
        res = L1.run(n_cal=6168)
        self.assertLess(res["identity_I1_max_abs_error"], 1e-12); self.assertLess(res["identity_I2_max_abs_mean_residual_calibrated_full"], 1e-10)
        self.assertLess(res["max_pointwise_identity_error"], 1e-12)
        miscal = [r for r in res["table"] if r["teacher"].startswith("tau_full=2.0")]
        self.assertTrue(any(abs(r["E_res"]) > 1e-4 for r in miscal))                 # an overconfident full teacher does leave a residual


class FakeLlava:
    """CPU stand-in: dim 0 of every embedding flags a visual token, so P(yes) is an exact function of the number of visual tokens seen."""
    yes_ids, no_ids, load_seconds = [1], [2], 0.0

    def __init__(self, fail_at=None):
        import torch
        self.torch, self.calls, self.fail_at = torch, 0, fail_at

    def encode(self, images):
        t = self.torch.zeros(len(images), MK.N, 4); t[:, :, 0] = 1.0
        t[:, :, 1] = self.torch.arange(MK.N, dtype=self.torch.float32) / MK.N
        return t

    def segments(self, question, with_image=True):
        ids = self.torch.tensor([5] * (3 + len(question) % 4))
        return (ids[:2], ids[2:]) if with_image else (ids, ids[:0])

    def embed_ids(self, ids):
        e = self.torch.zeros(len(ids), 4); e[:, 2] = 1.0
        return e

    def question_vector(self, q):
        return self.torch.tensor([0.0, 1.0, 0.0, 0.0])

    def last_logits(self, seqs):
        self.calls += 1
        if self.fail_at is not None and self.calls >= self.fail_at:
            raise RuntimeError("simulated crash")
        out = self.torch.zeros(len(seqs), 4)
        for i, s in enumerate(seqs):
            out[i, 1] = s[:, 0].sum() / 100.0
        return out


class TeacherBookkeeping(unittest.TestCase):
    def test_variants_use_exact_token_counts_and_batching_invariance(self):
        host = FakeLlava(); items = [{"qid": f"q{i}", "image_id": "img0", "question": "q" * i, "C": 1} for i in range(5)]
        a = TE.run_items(host, items, {"img0": None}, max_tokens=4096, max_seqs=7); b = TE.run_items(FakeLlava(), items, {"img0": None}, 10 ** 6, 1000)
        self.assertEqual(a, b)
        p = lambda n: math.exp(n / 100.0) / (math.exp(n / 100.0) + 3.0)  # noqa: E731   softmax over 4 logits: yes = n/100, three others 0
        for r in a:
            self.assertAlmostEqual(r["full"][0], p(576), places=5); self.assertAlmostEqual(r["text"][0], p(0), places=5)
            for m in MK.MECHANISMS:
                for k in MK.BUDGETS:
                    self.assertAlmostEqual(r["masked"][m][str(k)][0], p(k), places=5)
            self.assertEqual(len(TE.unb64(r["qsim_order"])), MK.N)

    def test_bundle_crash_resume_coverage(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); man, _ = DA.build_manifest(iter(synth_rows(60)), 20, 4, 4, 2)
            (root / "d1_manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in man))
            (root / "d1_manifest_meta.json").write_text(json.dumps({"manifest_sha256": DA.sha256_file(root / "d1_manifest.jsonl")}))
            plan = DA.plan_shards(man, images_per_shard=5, shards_per_bundle=10); (root / "d1_shards.json").write_text(json.dumps(plan))

            class Src:
                def __init__(self, bad=()):
                    self.bad = set(bad)

                def open(self, i):
                    if i in self.bad:
                        raise OSError("unreadable")
                    return None
            bad = plan["shards"]["FIT-001"][0]; a = Namespace(bundle="b000", manifest_dir=str(root), out=str(root / "teacher"), images_per_chunk=2, max_tokens=4096, max_seqs=16)
            with self.assertRaises(SystemExit):                                                    # gate: unfrozen manifest
                TE.cmd_bundle(a, host_factory=lambda: FakeLlava(), image_source=Src([bad]))
            DA.cmd_freeze(Namespace(out=str(root), protocol="R7-30-D1-test", go="unit test"))
            with self.assertRaises(SystemExit):                                                    # gate: probe report missing
                TE.cmd_bundle(Namespace(**vars(a), probe_report=str(root / "nope.json")), host_factory=lambda: FakeLlava(), image_source=Src([bad]))
            with self.assertRaises(RuntimeError):
                TE.cmd_bundle(a, host_factory=lambda: FakeLlava(fail_at=9), image_source=Src([bad]))
            TE.cmd_bundle(a, host_factory=lambda: FakeLlava(), image_source=Src([bad]))
            for shard in plan["bundles"]["b000"]:
                d = json.loads((root / "teacher" / f"{shard}.done.json").read_text()); self.assertEqual(d["duplicates"], 0)
                n_bad = sum(1 for r in man if r["image_id"] == bad and shard == "FIT-001")
                self.assertEqual(d["present"] + n_bad, d["expected"])
            rows, cov = AN.load_role(root / "teacher", root, "CAL"); self.assertEqual(cov["present"], cov["expected"])
            rows_f, cov_f = AN.load_role(root / "teacher", root, "FIT"); self.assertEqual(cov_f["present"] + cov_f["skipped_questions"], cov_f["expected"])
            (root / "teacher" / "CAL-000.jsonl").write_text("".join(json.dumps(r) + "\n" for r in V_read(root / "teacher" / "CAL-000.jsonl")[1:]))
            with self.assertRaises(RuntimeError):
                AN.load_role(root / "teacher", root, "CAL")


def V_read(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


class Analysis(unittest.TestCase):
    def test_known_values(self):
        rows = [{"qid": "a", "image_id": "i1", "C": 1, "structural": "verify", "full": (0.9, 0.1), "text": (0.5, 0.5),
                 "masked": {m: {str(k): (0.6, 0.4) for k in MK.BUDGETS} for m in MK.MECHANISMS}},
                {"qid": "b", "image_id": "i2", "C": -1, "structural": "logical", "full": (0.3, 0.7), "text": (0.5, 0.5),
                 "masked": {m: {str(k): (0.5, 0.5) for k in MK.BUDGETS} for m in MK.MECHANISMS}}]
        out = AN.analyse(rows, {"w_yes": 1.0, "w_no": 1.0}); c = out["cells"]["random:72"]
        Tb, Ta, C = np.array([0.8, -0.4]), np.array([0.2, 0.0]), np.array([1.0, -1.0])
        h = (C - Ta) ** 2 - (C - Tb) ** 2; d = (Tb - Ta) ** 2
        self.assertAlmostEqual(c["mean_h"], h.mean(), places=12); self.assertAlmostEqual(c["mean_d"], d.mean(), places=12)
        self.assertAlmostEqual(c["mean_residual"], (h - d).mean(), places=12); self.assertLess(c["identity_max_abs_error"], 1e-12)
        self.assertAlmostEqual(out["teacher"]["brier_full"], ((C - Tb) ** 2).mean(), places=12)

    def test_rank_ceiling_decision_rule(self):
        """The frozen rule ('d does not rank' iff observed + image-clustered margin < ceiling band's lower end) must not fire for a calibrated
        teacher (20 simulated datasets: at most 1 firing) and must fire for an anti-informative one.  The false-firing rate is printed."""
        fires, inside, n_sets = 0, 0, 20
        for seed in range(n_sets):
            g = np.random.default_rng(100 + seed); n = 3000; pt = g.uniform(0.02, 0.98, n); Tb = 2 * pt - 1
            Ta = np.clip(Tb * g.uniform(0.0, 1.0, n) + g.normal(0, 0.1, n), -1, 1)
            C = np.where(g.uniform(size=n) < pt, 1.0, -1.0)                                        # labels drawn FROM the full teacher
            rc = AN.rank_ceiling(C, Ta, Tb, [i // 3 for i in range(n)], draws=200, boot=200, seed=seed)
            fires += rc["verdict"] == "d_does_not_rank_per_question_losses"; inside += rc["ceiling_band"][0] <= rc["observed"] <= rc["ceiling_band"][1]
            self.assertEqual(rc["verdict"] == "d_does_not_rank_per_question_losses", rc["observed_boot_ci"][1] < rc["ceiling_band"][0])
        print(f"\n[rank-ceiling rule] calibrated teacher: false-firing {fires}/{n_sets} = {fires / n_sets:.2f}; "
              f"observed inside its own 95 % band {inside}/{n_sets} (informational)")
        self.assertLessEqual(fires, 1)
        g = np.random.default_rng(7); n = 2000; Tb = g.uniform(-0.9, 0.9, n); Ta = -Tb; C = np.sign(Ta)    # anti-informative: h = −4|Tb|, d = 4 Tb²
        rb = AN.rank_ceiling(C, Ta, Tb, [i // 2 for i in range(n)], draws=100, boot=200, seed=1)
        print(f"[rank-ceiling rule] anti-informative teacher: observed {rb['observed']:.3f}, band {rb['ceiling_band']}, verdict {rb['verdict']}")
        self.assertEqual(rb["verdict"], "d_does_not_rank_per_question_losses")

    def test_secondary_excludes_low_mass(self):
        g = np.random.default_rng(5); rows = []
        for i in range(60):
            C = 1 if i % 2 else -1; pf = g.uniform(0.1, 0.9); mass = 0.4 if i < 4 else 0.99                  # 4 low-mass items
            rows.append({"qid": f"q{i}", "image_id": f"i{i // 3}", "C": C, "structural": "verify", "full": (pf * mass, (1 - pf) * mass), "text": (0.5, 0.5),
                         "masked": {m: {str(k): (0.5, 0.5) for k in MK.BUDGETS} for m in MK.MECHANISMS}})
        sec = AN.secondary_mass({"CAL": rows, "FIT": rows[10:]}, {"w_yes": 1.0, "w_no": 1.0})
        self.assertEqual(sec["excluded_per_role"]["CAL"], {"excluded": 4, "of": 60}); self.assertEqual(sec["excluded_per_role"]["FIT"], {"excluded": 0, "of": 50})
        self.assertEqual(sec["tables"]["CAL"]["n_questions"], 56); self.assertNotIn("FIT", sec["tables"])
        prim = AN.analyse(rows, {"w_yes": 1.0, "w_no": 1.0}); self.assertEqual(prim["n_questions"], 60)               # primary keeps every item
        md = AN.markdown({"roles": {"CAL": prim}, "secondary_mass_ge_0p9": sec}); self.assertIn("Secondary: excludes yes/no mass < 0.9", md)

    def test_dev_sealed(self):
        old = sys.argv
        try:
            sys.argv = ["x", "--teacher-dir", "/nonexistent", "--manifest-dir", "/nonexistent", "--roles", "DEV"]
            with self.assertRaises(SystemExit):
                AN.main()
        finally:
            sys.argv = old


if __name__ == "__main__":
    unittest.main()
