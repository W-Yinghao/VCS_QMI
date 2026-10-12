"""Unit tests for scripts/r5_final_eval.py on SYNTHETIC files only (no official test file, no real annotation file is opened):
guard, REFER / MDETR parsers, ranking rules, hit / metric definitions, Holm, t interval, bootstrap, and a CPU end-to-end pass of
build -> (synthetic proposals and predictions) -> manifest -> gate -> score, including tamper detection.
    python -I -m unittest tests/test_r5_final_eval.py -v      (repo root; sbatch slurm/r5_final_eval_cpu.sbatch tests)
"""
import argparse
import json
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
import r5_final_eval as FE  # noqa: E402


def refer_files(d: Path, split_test="test"):
    """Three images: 1 (two referred objects, one with two sentences; split test), 2 (one referred object; test), 3 (train, must be ignored)."""
    refs = [{"ref_id": 0, "ann_id": 11, "image_id": 1, "split": split_test, "sentences": [{"sent_id": 100, "raw": "The Man.", "sent": "the man", "tokens": []},
                                                                                      {"sent_id": 101, "raw": "left guy", "sent": "left guy", "tokens": []}]},
            {"ref_id": 1, "ann_id": 10, "image_id": 1, "split": split_test, "sentences": [{"sent_id": 102, "raw": "A dog", "sent": "a dog", "tokens": []}]},
            {"ref_id": 2, "ann_id": 20, "image_id": 2, "split": split_test, "sentences": [{"sent_id": 200, "raw": "cat", "sent": "cat", "tokens": []}]},
            {"ref_id": 3, "ann_id": 30, "image_id": 3, "split": "train", "sentences": [{"sent_id": 300, "raw": "TRAIN ONLY", "sent": "train only", "tokens": []}]}]
    inst = {"images": [{"id": 1, "width": 100, "height": 80}, {"id": 2, "width": 50, "height": 50}, {"id": 3, "width": 10, "height": 10}],
            "annotations": [{"id": 10, "image_id": 1, "bbox": [0, 0, 40, 40]}, {"id": 11, "image_id": 1, "bbox": [50, 10, 30, 60]},
                            {"id": 20, "image_id": 2, "bbox": [5, 5, 20, 20]}, {"id": 30, "image_id": 3, "bbox": [1, 1, 5, 5]}]}
    d.mkdir(parents=True, exist_ok=True); (d / "refs(umd).p").write_bytes(pickle.dumps(refs)); (d / "instances.json").write_text(json.dumps(inst))
    return {"refs": str(d / "refs(umd).p"), "instances": str(d / "instances.json")}


class TestGuard(unittest.TestCase):
    def test_protected(self):
        self.assertTrue(FE.is_protected(FE.REFER_DIR / "refs(umd).p"))
        self.assertTrue(FE.is_protected(FE.MDETR_DIR / "finetune_refcoco_testA.json"))
        self.assertTrue(FE.is_protected(FE.MDETR_DIR / "finetune_refcocog_test.json"))
        self.assertTrue(FE.is_protected("/x/closed/anything.json"))
        self.assertTrue(FE.is_protected(FE.DATA / "refcoco_unc" / "scene_inputs.pkl"))
        self.assertFalse(FE.is_protected(FE.MDETR_DIR / "finetune_refcoco_train.json"))
        with self.assertRaises(PermissionError):           # raised before any open()
            FE.read_source(FE.REFER_DIR / "refs(umd).p", official=False)
        with self.assertRaises(PermissionError):
            FE.read_source("/nonexistent/finetune_refcoco+_testB.json", official=False)


class TestParsers(unittest.TestCase):
    def test_refer(self):
        with tempfile.TemporaryDirectory() as t:
            src = refer_files(Path(t) / "r")
            images, labels, counts, srcs, echo = FE.parse_refer("g_test", {"sources": src, "split": "test"}, False, {})
            self.assertEqual([im["image_id"] for im in images], [1, 2])
            self.assertEqual([q["qid"] for q in images[0]["queries"]], ["g_test:102", "g_test:100", "g_test:101"])   # ann 10 before ann 11; file order within
            self.assertEqual(images[0]["queries"][1]["text"], "The Man.")                                           # raw form, not `sent`
            self.assertEqual(labels["queries"]["g_test:100"], [50.0, 10.0, 80.0, 70.0])
            self.assertEqual(labels["images"], {"1": {"ge2": True}, "2": {"ge2": False}})
            self.assertEqual((counts["images"], counts["referred"], counts["expressions"], counts["images_ge2"]), (2, 3, 4, 1))
            self.assertNotIn("g_test:300", labels["queries"])
            for im in images:                                                                                      # inputs carry no label
                self.assertEqual(set(im), {"image_id", "path", "width", "height", "queries"})
                for q in im["queries"]:
                    self.assertEqual(set(q), {"qid", "text"})
            self.assertEqual(srcs["refs"]["sha256"], FE.sha256_file(src["refs"]))

    def test_mdetr(self):
        d = {"images": [{"id": 7, "original_id": 5, "caption": "b first", "width": 64, "height": 48},
                        {"id": 3, "original_id": 5, "caption": "a second", "width": 64, "height": 48},
                        {"id": 9, "original_id": 4, "caption": "solo", "width": 32, "height": 32}],
                "annotations": [{"image_id": 7, "original_id": 77, "bbox": [1, 2, 3, 4]}, {"image_id": 3, "original_id": 66, "bbox": [10, 10, 5, 5]},
                                {"image_id": 9, "original_id": 55, "bbox": [0, 0, 0.5, 8]}]}
        with tempfile.TemporaryDirectory() as t:
            f = Path(t) / "finetune_x_train.json"; f.write_text(json.dumps(d))
            images, labels, counts, srcs, _ = FE.parse_mdetr("x_DEV", {"sources": {"mdetr": str(f)}}, False, {})
            self.assertEqual([im["image_id"] for im in images], [4, 5])
            self.assertEqual([q["qid"] for q in images[1]["queries"]], ["x_DEV:3", "x_DEV:7"])                      # ann 66 before ann 77
            self.assertEqual(labels["queries"]["x_DEV:7"], [1.0, 2.0, 4.0, 6.0])
            self.assertEqual(labels["images"]["5"]["ge2"], True); self.assertEqual(labels["images"]["4"]["ge2"], False)
            self.assertEqual(counts["degenerate_referred_boxes"], 1)                                               # 0.5 px wide box
            self.assertEqual(counts["expressions"], 3)


class TestOfficialBuildPath(unittest.TestCase):
    """The official (test-mode) build path on SYNTHETIC files: RG.dev_roles DEV echo, recorded counts, archive hash + member identity, and the
    all-or-nothing stop before any input / label file is written."""

    def make(self, d: Path, n_train=60):
        import torch
        refs, images, anns, sid = [], [], [], 1000
        for iid in range(1, n_train + 6):
            sp = "train" if iid <= n_train else ("val" if iid <= n_train + 2 else "test")
            images.append({"id": iid, "width": 100, "height": 100})
            for k in range(2):                                                      # two referred objects per image
                aid = iid * 10 + k; anns.append({"id": aid, "image_id": iid, "bbox": [10.0 * k, 5.0, 20.0, 30.0]})
                refs.append({"ref_id": aid, "ann_id": aid, "image_id": iid, "split": sp,
                             "sentences": [{"sent_id": sid, "raw": f"Object {k} of image {iid}.", "sent": "", "tokens": []}]}); sid += 1
        d.mkdir(parents=True, exist_ok=True); (d / "refs(umd).p").write_bytes(pickle.dumps(refs)); (d / "instances.json").write_text(json.dumps({"images": images, "annotations": anns}))
        objs = [type("S", (), {"image_id": i, "umd_split": "train" if i <= n_train else "test"})() for i in range(1, n_train + 6)]
        role = FE.RG.dev_roles(objs); meta = []
        for iid in sorted(i for i, r in role.items() if r == "DEV"):
            rr = [r for r in refs if r["image_id"] == iid]
            meta.append({"image_id": iid, "sent_id": np.array([r["sentences"][0]["sent_id"] for r in rr]), "texts": [r["sentences"][0]["raw"] for r in rr],
                         "target": np.array([0, 1]), "ref_boxes": np.array([FE.xyxy(a["bbox"]) for a in anns if a["image_id"] == iid])})
        torch.save(meta, d / "meta.pt")
        import zipfile as zf
        with zf.ZipFile(d / "release.zip", "w") as z:
            z.write(d / "refs(umd).p", "refcocog/refs(umd).p"); z.write(d / "instances.json", "refcocog/instances.json")
        return refs, meta

    def plan(self, d: Path, recorded_zip_sha: str, expected: dict):
        return {"mode": "test", "eval_sets": {"refcocog_test": {
            "dataset": "refcocog", "split": "test", "format": "refer", "official": True, "sources": {"refs": str(d / "refs(umd).p"), "instances": str(d / "instances.json")},
            "expected": expected, "dev_echo": str(d / "meta.pt"),
            "archive_check": {"path": str(d / "release.zip"), "members": {"refs": "refcocog/refs(umd).p", "instances": "refcocog/instances.json"},
                              "recorded_sha256": recorded_zip_sha, "recorded_in": "synthetic"}}}}

    def run_build(self, root: Path, plan: dict) -> int:
        root.mkdir(parents=True, exist_ok=True); (root / "plan.json").write_text(json.dumps(plan))
        orig = FE.RG.coco_path; FE.RG.coco_path = lambda iid: f"/synthetic/{iid}.jpg"
        try:
            return FE.cmd_build(argparse.Namespace(root=str(root), plan=None))
        finally:
            FE.RG.coco_path = orig

    def test_official_build(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t) / "src"; refs, meta = self.make(d); zsha = FE.sha256_file(d / "release.zip")
            self.assertGreater(len(meta), 0)
            ok = {"images": 3, "referred": 6, "expressions": 6, "images_ge2": 3}
            self.assertEqual(self.run_build(Path(t) / "ok", self.plan(d, zsha, ok)), 0)
            b = json.loads((Path(t) / "ok" / "logs" / "build.json").read_text())["eval_sets"]["refcocog_test"]["checks"]
            self.assertTrue(b["dev_echo_refs_umd"]["pass"]); self.assertEqual(b["dev_echo_refs_umd"]["matching_expressions"], 2 * len(meta))
            self.assertTrue(b["archive"]["pass"]); self.assertTrue(b["archive"]["archive_matches_record"])
            self.assertTrue((Path(t) / "ok" / "sealed" / "labels" / "refcocog_test.json").exists())
            # any failed check -> nothing written
            for name, plan in (("count", self.plan(d, zsha, {**ok, "expressions": 7})), ("archive", self.plan(d, "0" * 64, ok))):
                r = Path(t) / name
                self.assertEqual(self.run_build(r, plan), FE.EXIT_CHECK, name)
                self.assertFalse((r / "inputs").exists()); self.assertFalse((r / "sealed").exists()); self.assertTrue((r / "logs" / "build_FAILED.json").exists())
            meta[0]["texts"][0] = "changed"; import torch; torch.save(meta, d / "meta.pt")
            r = Path(t) / "echo"; self.assertEqual(self.run_build(r, self.plan(d, zsha, ok)), FE.EXIT_CHECK); self.assertFalse((r / "sealed").exists())

    def test_tar_members(self):
        import tarfile as tf
        with tempfile.TemporaryDirectory() as t:
            d = Path(t); (d / "a_testA.json").write_text("{}"); (d / "other.json").write_text("[1]")
            with tf.open(d / "arc.tar.gz", "w:gz") as z:
                z.add(d / "a_testA.json", "mdetr_annotations/a_testA.json"); z.add(d / "other.json", "mdetr_annotations/other.json")
            h = FE.archive_member_hashes(str(d / "arc.tar.gz"), {"a_testA.json"})
            self.assertEqual(h["members"], {"a_testA.json": [FE.sha256_file(d / "a_testA.json")]}); self.assertEqual(h["sha256"], FE.sha256_file(d / "arc.tar.gz"))


class TestRankAndScore(unittest.TestCase):
    def test_rank_rules(self):
        cos = np.array([[0.30, 0.10], [0.29, 0.40], [0.30, 0.00]]); det = np.array([0.05, 0.9, 0.5])
        r = FE.rank_rows(cos, det, 10.0, {"c": {"kind": "cos"}, "f0": {"kind": "score_fusion", "lambda": 0.0},
                                          "f1": {"kind": "score_fusion", "lambda": 1.0}, "f4": {"kind": "score_fusion", "lambda": 4.0}})
        self.assertEqual(r["c"].tolist(), [0, 1])                   # tie 0.30 / 0.30 -> first in detector order
        self.assertEqual(r["f0"].tolist(), [0, 1])
        self.assertEqual(r["f1"].tolist(), [1, 1])                  # 6.0 + log .05 < 5.8 + log .9
        self.assertEqual(r["f4"].tolist(), [1, 1])
        with self.assertRaises(ValueError):
            FE.rank_rows(cos, np.array([0.1, 0.0, 0.2]), 10.0, {"f": {"kind": "score_fusion", "lambda": 1.0}})

    def test_clean_proposals(self):
        P = np.array([[-5, -5, 20, 20, 0.9, 1], [10, 10, 10.5, 30, 0.8, 1], [0, 0, 200, 200, 0.7, 3]], float)
        boxes, det = FE.clean_proposals(P, 100, 50)
        self.assertEqual(boxes, [(0.0, 0.0, 20.0, 20.0), (0.0, 0.0, 100.0, 50.0)]); self.assertEqual(det, [0.9, 0.7])

    def test_hits_metrics(self):
        inp = {"images": [{"image_id": 1, "queries": [{"qid": "a"}, {"qid": "b"}]}, {"image_id": 2, "queries": [{"qid": "c"}]}]}
        lab = {"queries": {"a": [0, 0, 10, 10], "b": [0, 0, 10, 10], "c": [0, 0, 4, 4]}, "images": {"1": {"ge2": True}, "2": {"ge2": False}}}
        A = FE.es_arrays(inp, lab)
        preds = {"a": [0, 0, 0, 10, 5], "b": None, "c": [0, 0, 0, 4, 4]}          # IoU(a) = 0.5 exactly -> hit (>=); b uncovered -> error
        h = FE.hits_of(preds, A); self.assertEqual(h.tolist(), [True, False, True])
        m = FE.metrics_of(h, A)
        self.assertAlmostEqual(m["query_all"], 2 / 3); self.assertAlmostEqual(m["image_macro_ge2"], 0.5); self.assertAlmostEqual(m["query_ge2"], 0.5)

    def test_holm_and_t(self):
        adj = FE.holm({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertAlmostEqual(adj["a"][0], 0.03); self.assertTrue(adj["a"][1]); self.assertAlmostEqual(adj["c"][0], 0.06); self.assertAlmostEqual(adj["b"][0], 0.06)
        self.assertFalse(adj["b"][1])
        from scipy import stats
        from vl1_10_aggregate import tint
        x = [0.011, 0.019, 0.007]; t = FE.t_stats(x); ref = tint(x)
        self.assertAlmostEqual(t["ci95"][0], ref["ci95"][0]); self.assertAlmostEqual(t["ci95"][1], ref["ci95"][1])
        self.assertAlmostEqual(t["p_two_sided"], stats.ttest_1samp(x, 0).pvalue)

    def test_bootstrap(self):
        rng = np.random.default_rng(0); n = 200; nq = rng.integers(1, 6, n).astype(float); d = rng.normal(0.1, 0.5, n) * nq
        idx = FE.boot_matrix(n, 500, 0); ci = FE.boot_ratio_ci(d, nq, idx); pt = d.sum() / nq.sum()
        self.assertTrue(ci[0] <= pt <= ci[1]); self.assertEqual(ci, FE.boot_ratio_ci(d, nq, FE.boot_matrix(n, 500, 0)))


class TestEndToEndCPU(unittest.TestCase):
    """build -> synthetic proposals / predictions (the GPU stages' file formats) -> manifest -> gate -> score; tamper detection."""

    def test_pipeline(self):
        import torch
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "root"; src = refer_files(Path(t) / "r")
            plan = {"mode": "dev", "eval_sets": {"g_DEV": {"dataset": "refcocog", "split": "test", "format": "refer", "sources": src, "official": False,
                                                           "expected": {"images": 2, "referred": 3, "expressions": 4, "images_ge2": 1}}},
                    "models": {"VQ_s0": {"dataset": "refcocog", "seed": 0}, "VCS_s0": {"dataset": "refcocog", "seed": 0}},
                    "rows": [{"row": "VQ|g_DEV|s0", "method": "VQ", "model": "VQ_s0", "eval_set": "g_DEV", "seed": 0, "rank": {"kind": "cos"}},
                             {"row": "VCS|g_DEV|s0", "method": "VCS", "model": "VCS_s0", "eval_set": "g_DEV", "seed": 0, "rank": {"kind": "cos"}}],
                    "shards": {"sh": ["VQ_s0", "VCS_s0"]}, "contrasts": [["primary", "VQ", "VCS"]], "missing": [],
                    "statistics": {"bootstrap": {"B": 50}, "holm": {"alpha": 0.05, "family": "VQ - VCS"}}}
            root.mkdir(parents=True); (root / "plan.json").write_text(json.dumps(plan))
            ns = argparse.Namespace(root=str(root), plan=None, allow_code_change=False, out=None)
            orig = FE.RG.coco_path; FE.RG.coco_path = lambda iid: f"/synthetic/{iid}.jpg"          # no image needed on CPU
            try:
                self.assertEqual(FE.cmd_build(ns), 0)
            finally:
                FE.RG.coco_path = orig
            self.assertEqual(oct(os.stat(root / "sealed" / "labels" / "g_DEV.json").st_mode & 0o777), "0o400")
            bl = FE.build_log(root); inp = FE.load_inputs(root, "g_DEV", bl["eval_sets"]["g_DEV"]["inputs"]["sha256"])
            props = {1: torch.tensor([[0, 0, 40, 40, 0.9, 1], [50, 10, 80, 70, 0.8, 1]], dtype=torch.float32), 2: torch.zeros((0, 6))}
            (root / "proposals").mkdir(); torch.save(props, root / "proposals" / "g_DEV.pt")
            FE.stage_record(root, "proposals", {"eval_sets": {"g_DEV": {"sha256": FE.sha256_file(root / "proposals" / "g_DEV.pt")}}})
            (root / "proposals" / "PROPOSALS_DONE").write_text("x")
            # VQ predicts the right box for every query of image 1; VCS swaps them; image 2 has no proposal (error for both)
            gt_of = {"g_DEV:102": 0, "g_DEV:100": 1, "g_DEV:101": 1}
            pd = root / "sealed" / "predictions"
            for mid, flip in (("VQ_s0", False), ("VCS_s0", True)):
                rows = {}
                for r in plan["rows"]:
                    if r["model"] != mid:
                        continue
                    pr = {}
                    for im in inp["images"]:
                        for q in im["queries"]:
                            if im["image_id"] == 2:
                                pr[q["qid"]] = None
                            else:
                                k = gt_of[q["qid"]] ^ int(flip); pr[q["qid"]] = [k] + props[1][k, :4].tolist()
                    rows[r["row"]] = {"rank": r["rank"], "predictions": pr}
                FE.write_atomic(pd / f"pred__{mid}__g_DEV.json", FE.canon({"model": mid, "eval_set": "g_DEV", "rows": rows}), mode=0o444)
            FE.stage_record(root, "rank_sh", {}); FE.write_atomic(pd / "shards" / "sh.done", b"{}")
            self.assertEqual(FE.cmd_manifest(ns), 0)
            self.assertEqual(FE.cmd_score(ns), FE.EXIT_CHECK)                                    # no gate yet
            msha = FE.sha256_file(pd / "MANIFEST.sha256"); (root / "ALLOW_SCORING").write_text(f"manifest_sha256={msha}\n")
            self.assertEqual(FE.cmd_score(ns), 0)
            res = json.loads((root / "score" / "results.json").read_text())
            self.assertAlmostEqual(res["rows"]["VQ|g_DEV|s0"]["query_all"], 3 / 4); self.assertAlmostEqual(res["rows"]["VCS|g_DEV|s0"]["query_all"], 0.0)
            self.assertAlmostEqual(res["rows"]["VQ|g_DEV|s0"]["image_macro_ge2"], 1.0); self.assertAlmostEqual(res["eval_sets"]["g_DEV"]["ceiling_any_valid_proposal_query_all"], 0.75)
            c = [x for x in res["contrasts"] if x["metric"] == "query_all"][0]; self.assertAlmostEqual(c["mean"], 0.75)
            self.assertTrue((root / "score" / "tables.md").exists())
            self.assertEqual(FE.cmd_score(ns), FE.EXIT_CHECK)                                    # a second default scoring is refused
            rv = argparse.Namespace(**{**vars(ns), "out": str(root / "score_reverify")})           # deterministic re-verification elsewhere
            self.assertEqual(FE.cmd_score(rv), 0)
            self.assertEqual(json.loads((root / "score_reverify" / "results.json").read_text())["rows"], res["rows"])
            # tamper: a changed prediction file must stop the score
            os.chmod(pd / "pred__VQ_s0__g_DEV.json", 0o644); (pd / "pred__VQ_s0__g_DEV.json").write_bytes(b"{}")
            self.assertEqual(FE.cmd_score(argparse.Namespace(**{**vars(ns), "out": str(root / "score_tampered")})), FE.EXIT_CHECK)


if __name__ == "__main__":
    unittest.main()
