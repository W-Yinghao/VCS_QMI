"""R7-50 phase Q (Qwen2.5-VL rows on the official test splits; addendum reports/R7/R7_50_PHASE_Q_ADDENDUM_DRAFT_20261012.yaml) on CPU with
synthetic files only (tmp dirs; no real data, no sealed / closed file): configs_Q (the 41 rows of Q1, unit / PCA / lr_choice / addendum hashes,
R7-40c cross-check, image-grid check and refusals), lines_Q, the generation guard and the R7-40c runner (imported, unchanged) on R7-40c's fake
Qwen host writing into sealed/pred_Q, the full-scale integrity gate (plan pins from R7-40c's reference markers, the runner on both sides, pass
and every STOP case: answer, kept count, input, missing item on either side, GPU class, reference sets, marker; the development-mode build when
pyarrow is present), manifest_Q (coverage, shard markers, R7-40c's timing bundle files, marker and B* report; refusals), score_Q on a synthetic
labelled set (ALLOW_SCORING_Q incl. the committed-manifest check, fraction and latency bases, Holm over 6, row latencies, ChartQA splits), and
stubbed dry runs of slurm/r7_50_q_submit.sh (reaching the ALLOW_SCORING_Q wait; every STOP).
    python -I tests/test_r7_50_q.py -v        (slurm/r7_50_q_cpu.sbatch tests | tests_vl)"""
import hashlib
import importlib.util
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

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts")); sys.path.insert(0, str(R / "tests"))
import r7_50_q as Q  # noqa: E402
import r7_d1_30_data as DD  # noqa: E402
import r7_d1_30_run as RUN  # noqa: E402
import r7_d1_40_masks as M4  # noqa: E402
import r7_d1_40c_plan as P40  # noqa: E402
import r7_d1_40c_run as RN  # noqa: E402
import r7_vl_common as V  # noqa: E402
import test_r7_d1_40c as T40  # noqa: E402   (R7-40c's fake Qwen host and tiny R7-40b-style units, imported)

T = Q.T
LR = T40.LR
HAS_PYARROW = importlib.util.find_spec("pyarrow") is not None
FROZEN = "status: frozen\nr7_50_freeze_20261013:\n  by: test\n"
ADD_FROZEN = "addendum_id: R7-50-Q\nstatus: frozen_20261013   # test\n"
CLASS = {"L40S": "NVIDIA L40S", "H100": "NVIDIA H100 NVL"}
GRIDS = [(3, 4), (4, 4), (2, 5), (4, 5), (5, 6)]


def grid_for(f):
    """Small merged grids for the fake host (the tiny selectors' tables are 8 × 8), keyed by the image BYTES: an extracted copy of a chart gets
    the grid of its original."""
    return GRIDS[int(hashlib.sha256(Path(f).read_bytes()).hexdigest(), 16) % len(GRIDS)]


def opener(f):
    return {"grid": grid_for(f)}


def ns(argv, chunk=32, batch=32):
    """The runner's Namespace for the arguments the GPU wrapper passes to 'r7_d1_40c_run.py run' (chunk / batch = the runner's defaults)."""
    assert argv[0] == "run"; d = {argv[i][2:].replace("-", "_"): argv[i + 1] for i in range(1, len(argv), 2)}
    return Namespace(**d, chunk=chunk, batch=batch, limit=0)


def run_unit(a, units, gpu):
    """The R7-40c quality runner (unchanged) on the fake host; the marker's env.gpu as on the given GPU."""
    with mock.patch.object(V, "env_info", lambda: {"gpu": gpu}):
        return RN.run(a, host=T40.FakeHost(), bank=RN.QwenSelectorBank(Path(a.selectors_dir), Path(a.pca), units, device="cpu"), opener=opener, grid_of=grid_for)


def seal(root: Path, img: Path, n_img=6, q_per=3, n_chart=8):
    """Sealed inputs / labels / build summary in phase L's format (bench names gqa_testdev / chartqa_test; images as files), synthetic, no parquet."""
    from PIL import Image
    L = T.Layout(root, "Q"); img.mkdir(parents=True, exist_ok=True); items = {b: [] for b in Q.BENCHES}; labels = {b: [] for b in Q.BENCHES}
    for i in range(n_img):
        f = img / f"{2000 + i}.jpg"; Image.new("RGB", (640 - 20 * i, 480), (10 * i, 50, 0)).save(f, "JPEG")
        for j in range(q_per):
            q = f"t{i}{j:02d}"
            items["gqa_testdev"].append({"bench": "gqa_testdev", "qid": q, "image_id": f"{2000 + i}", "image_file": str(f), "question": f"Is it {i}-{j}?",
                                         "prompt": f"Is it {i}-{j}?" + DD.SHORT})
            labels["gqa_testdev"].append({"qid": q, "answer": f"zebra{i}{j}" if j % 2 else "yes", "structural": "verify" if j % 2 else "query", "semantic": "a", "detailed": "d"})
    for i in range(n_chart):
        f = img / f"chart{i // 2}.png"
        if not f.exists():
            Image.new("RGB", (800, 600 - 10 * (i // 2)), (0, 0, 20 * (i // 2))).save(f)
        q = f"chartqa_test_{i:05d}"
        items["chartqa_test"].append({"bench": "chartqa_test", "qid": q, "image_id": f.name, "image_file": str(f), "question": f"Value {i}?", "prompt": f"Value {i}?" + DD.CHART_SUFFIX})
        labels["chartqa_test"].append({"qid": q, "answer": str(40 + i), "type": "human_test" if i % 2 == 0 else "augmented_test", "split": "human" if i % 2 == 0 else "augmented"})
    benches = {}
    for b in Q.BENCHES:
        T.write_atomic(L.inputs_file(b), T.jsonl(items[b])); T.write_atomic(L.labels_file(b), T.jsonl(labels[b]), mode=0o400)
        benches[b] = {"bench": b, "mode": "development (synthetic test fixture)", "source": "synthetic", "sha256": None, "recorded": None, "items": len(items[b]),
                      "groups": len({it["image_id"] for it in items[b]}), "inputs_sha256": T.sha(L.inputs_file(b)), "labels_sha256": T.sha(L.labels_file(b))}
    benches["chartqa_test"]["split_counts"] = {k: sum(x["split"] == k for x in labels["chartqa_test"]) for k in ("human", "augmented")}
    T.write_atomic(L.summary, json.dumps({"ok": True, "freeze": None, "benches": benches}))
    return L, items, labels


def configured(t: Path, **kw):
    """Sealed root + tiny R7-40b-style units (9 units, PCA, lr_choice) + a frozen addendum -> configs_Q."""
    L, items, labels = seal(t / "root", t / "img", **kw); T40.tiny_units(t / "sel"); (t / "add.yaml").write_text(ADD_FROZEN)
    Q.configs(L.root, t / "sel" / "lr_choice.json", t / "sel", t / "sel" / "pca.pt", None, t / "add.yaml")
    return L, json.loads((L.configs / "rows.json").read_text()), labels


def gate_pass(L):
    T.write_atomic(L.integrity / "integrity.json", json.dumps({"pass": True}))


# ------------------------------------------------------------------------------------------------------------------------------ rows / configs / lines
class Rows(unittest.TestCase):
    def test_q1_rows(self):
        main = P40.unit_names(LR); rows = Q.q_rows(main)
        self.assertEqual(len(rows), 41); self.assertEqual(rows[0], "full"); self.assertEqual(rows, [c for c in P40.config_list(main, "core") if not c.startswith("dyn:")])
        self.assertEqual([r for r in rows if r.startswith("static:")], [f"static:grid:{f}" for f in P40.FKEYS])
        for o in P40.OBJECTIVES:
            self.assertEqual(sorted(r for r in rows if r.startswith(f"sel:{o}_")), sorted(f"sel:{u}:{f}" for u in main[o] for f in P40.FKEYS))
        self.assertFalse(any(x in r for r in rows for x in ("dyn:", "random", "qsim", "sparsevlm", "official")))
        self.assertEqual(Q.row_meta(f"sel:{main['logistic'][2]}:0.125", main), {"family": "S2", "objective": "logistic", "seed": 2, "unit": main["logistic"][2],
                                                                                 "fraction": 0.125, "stress_point": True})

    def test_configs_rows_json_hashes(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); L, items, _ = seal(t / "root", t / "img"); T40.tiny_units(t / "sel"); (t / "add.yaml").write_text(ADD_FROZEN)
            r40c = t / "r40c_cfg"; P40.write_configs(LR, r40c)                               # R7-40c's own configuration files (50 core rows)
            self.assertEqual(Q.configs(L.root, t / "sel" / "lr_choice.json", t / "sel", t / "sel" / "pca.pt", r40c, t / "add.yaml"), {b: 41 for b in Q.BENCHES})
            rec = json.loads((L.configs / "rows.json").read_text()); main = P40.unit_names(LR)
            self.assertEqual(sorted(rec["units"]), sorted(u for o in P40.OBJECTIVES for u in main[o]))
            for u, r in rec["units"].items():
                self.assertEqual(r["pt_sha256"], hashlib.sha256((t / "sel" / f"{u}.pt").read_bytes()).hexdigest())
                self.assertEqual(r["json_sha256"], hashlib.sha256((t / "sel" / f"{u}.json").read_bytes()).hexdigest())
            self.assertEqual(rec["pca"]["sha256"], T.sha(t / "sel" / "pca.pt")); self.assertEqual(rec["lr_choice"]["sha256"], T.sha(t / "sel" / "lr_choice.json"))
            self.assertEqual(rec["addendum"]["status"], "frozen_20261013"); self.assertEqual(rec["representative_unit"], main["vcs"][0])
            self.assertEqual(rec["units"][main["direct"][1]]["seed"], 1); self.assertEqual(rec["units"][main["vcs"][0]]["name"], "S1")
            for b in Q.BENCHES:
                self.assertEqual(P40.read_configs(L.configs_file(b)), Q.q_rows(main)); self.assertEqual(rec["benches"][b]["configs_sha256"], T.sha(L.configs_file(b)))
                self.assertEqual(rec["benches"][b]["r7_40c_configs"]["file"], str(r40c / f"configs_{Q.VAL[b]}.txt")); self.assertEqual(rec["benches"][b]["shards"], Q.SHARDS[b])
                g = rec["grid_check"][b]; self.assertEqual((g["missing_images"], g["images_over_64x64"]), (0, 0)); self.assertEqual(g["items"], len(items[b]))
            self.assertEqual((rec["benches"]["gqa_testdev"]["max_new_tokens"], rec["benches"]["chartqa_test"]["max_new_tokens"]), (128, 16))
            self.assertNotIn("zebra", (L.configs / "rows.json").read_text())                  # no label anywhere in configs_Q

    def test_configs_refusals(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); L, _, _ = seal(t / "root", t / "img"); T40.tiny_units(t / "sel"); (t / "add.yaml").write_text(ADD_FROZEN); r40c = t / "r40c_cfg"
            args = lambda **kw: dict(dict(lr_choice=t / "sel" / "lr_choice.json", selectors_dir=t / "sel", pca=t / "sel" / "pca.pt", r40c_configs=None,  # noqa: E731
                                          addendum=t / "add.yaml"), **kw)
            (t / "draft.yaml").write_text("status: draft\n")
            for k, msg in ((dict(addendum=t / "draft.yaml"), "not the frozen phase-Q addendum"), (dict(addendum=t / "none.yaml"), "not the frozen")):
                with self.assertRaisesRegex(SystemExit, msg):
                    Q.configs(L.root, **args(**k))
            P40.write_configs(LR, r40c); (r40c / "configs_chartqa_val.txt").write_text("full\n")
            with self.assertRaisesRegex(SystemExit, "rows absent from R7-40c"):
                Q.configs(L.root, **args(r40c_configs=r40c))
            main = P40.unit_names(LR); js = t / "sel" / f"{main['direct'][2]}.json"; keep = js.read_bytes(); js.unlink()
            with self.assertRaisesRegex(SystemExit, "incomplete"):
                Q.configs(L.root, **args())
            js.write_bytes(keep); im = t / "img" / "2001.jpg"; keep = im.read_bytes(); Image.new("RGB", (4000, 200)).save(im, "JPEG")   # 7 × 143 merged grid
            with self.assertRaisesRegex(SystemExit, "64 × 64"):
                Q.configs(L.root, **args())
            self.assertFalse(json.loads((L.configs / "grid_check.json").read_text())["pass"]); self.assertFalse((L.configs / "rows.json").exists())
            im.write_bytes(keep); f = L.inputs_file("chartqa_test"); keep = f.read_bytes(); f.write_bytes(keep + b"\n")
            with self.assertRaisesRegex(SystemExit, "changed since the build"):
                Q.configs(L.root, **args())
            f.write_bytes(keep); Q.configs(L.root, **args()); (L.pred / "gqa_testdev").mkdir(parents=True); (L.pred / "gqa_testdev" / "full@s0of6.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(SystemExit, "fixed once"):
                Q.configs(L.root, **args())

    def test_lines(self):
        U = Q.lines(Path("/x/root")); self.assertEqual(len(U), 6 + 2)
        for line, mark in U:
            self.assertTrue(line.startswith("sbatch --parsable --job-name=r7_50qq_")); self.assertIn("--partition=L40S --exclude=node51,node60 ", line)
            self.assertFalse(any(x in line for x in ("--export", "--dependency", "--nodelist", "node52", "H100", "RTX")))             # Q8: no node pin
            self.assertTrue(str(mark).startswith("/x/root/sealed/pred_Q/")); self.assertFalse("gqa" in line and "chartqa" in line)
        self.assertEqual(sum("run gqa_testdev " in l and "--time=03:00:00" in l for l, _ in U), 6); self.assertEqual(sum("run chartqa_test " in l and "--time=04:00:00" in l for l, _ in U), 2)
        self.assertIn(("sbatch --parsable --job-name=r7_50qq_chartqa_test_1 --partition=L40S --exclude=node51,node60 --time=04:00:00 slurm/r7_50_q_gpu.sbatch run chartqa_test 1/2 /x/root",
                       Path("/x/root/sealed/pred_Q/chartqa_test/done_shard1of2.json")), U)
        with tempfile.TemporaryDirectory() as t:
            self.assertEqual(Q.write_lines(Path(t)), 8); txt = (Path(t) / "lines_Q.txt").read_text().splitlines()
            self.assertTrue(txt[0].startswith("#")); self.assertEqual(sum(" #MARK " in x and not x.startswith("#") for x in txt), 8)


# ------------------------------------------------------------------------------------------------------------------------------ generation
class Generation(unittest.TestCase):
    def test_guard_runner_markers(self):
        """The guard refuses before the integrity gate (top-level 'pass' only), on a wrong shard, a changed selector file, an edited configs
        file and after manifest_Q; its arguments drive the R7-40c runner into sealed/pred_Q with markers that record configs_Q, the sealed
        inputs and the selector hashes."""
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); L, rec, _ = configured(t); main = P40.unit_names(LR)
            with self.assertRaisesRegex(SystemExit, "integrity gate"):
                Q.generation_args(L.root, "gqa_testdev", "0/6")
            T.write_atomic(L.integrity / "integrity.json", json.dumps({"benches": {"x": {"pass": True}}, "pass": False}))
            with self.assertRaisesRegex(SystemExit, "integrity gate"):
                Q.generation_args(L.root, "gqa_testdev", "0/6")
            gate_pass(L)
            for b, sh in (("gqa_testdev", "0/5"), ("gqa_testdev", "6/6"), ("chartqa_test", "0/6"), ("textvqa_val", "0/4")):
                with self.assertRaises(SystemExit):
                    Q.generation_args(L.root, b, sh)
            argv = Q.generation_args(L.root, "chartqa_test", "1/2")
            self.assertEqual(argv, ["run", "--sets-dir", str(L.inputs), "--bench", "chartqa_test", "--configs-file", str(L.configs_file("chartqa_test")), "--shard", "1/2",
                                    "--out", str(L.pred), "--selectors-dir", str(t / "sel"), "--pca", str(t / "sel" / "pca.pt")])
            units = sorted(rec["units"])
            for b in Q.BENCHES:
                for i in range(Q.SHARDS[b]):
                    run_unit(ns(Q.generation_args(L.root, b, f"{i}/{Q.SHARDS[b]}")), units, CLASS["L40S"])
                od = L.pred / b; self.assertEqual(len(list(od.glob("*.jsonl"))), 41 * Q.SHARDS[b])
                for i in range(Q.SHARDS[b]):
                    mk = json.loads((od / f"done_shard{i}of{Q.SHARDS[b]}.json").read_text())
                    self.assertEqual((mk["configs"], mk["configs_sha256"], mk["sets_file_sha256"]), (41, T.sha(L.configs_file(b)), T.sha(L.inputs_file(b))))
                    self.assertEqual(mk["selector_files"]["pca"]["sha256"], rec["pca"]["sha256"]); self.assertEqual(mk["env"]["gpu"], "NVIDIA L40S")
                    self.assertTrue(all(mk["selector_files"][u]["pt_sha256"] == rec["units"][u]["pt_sha256"] for u in units))
                for c in Q.q_rows(main):
                    recs, n = Q.SC.load_config(od, c); self.assertEqual(set(recs), {it["qid"] for it in T.read_jsonl_strict(L.inputs_file(b))})
                    k, _, f = P40.parse(c)
                    for r in recs.values():
                        self.assertEqual(r["kept"], r["n_vis"] if k == "full" else M4.budget(f, r["n_vis"])); self.assertNotIn("answer", r)
            self.assertEqual(json.loads((L.pred / "gqa_testdev" / "full@s0of6.jsonl").read_text().splitlines()[0])["pred"][0], "v")
            pt = t / "sel" / f"{main['logistic'][1]}.pt"; keep = pt.read_bytes(); pt.write_bytes(keep + b"x")
            with self.assertRaisesRegex(SystemExit, "differ from their recorded sha256"):
                Q.generation_args(L.root, "gqa_testdev", "0/6")
            pt.write_bytes(keep); cf = L.configs_file("gqa_testdev"); keep = cf.read_text(); cf.write_text(keep.replace("static:grid:0.125\n", ""))
            with self.assertRaisesRegex(SystemExit, "differs from configs_Q"):
                Q.generation_args(L.root, "gqa_testdev", "0/6")
            cf.write_text(keep); L.manifest.parent.mkdir(parents=True); L.manifest.write_text("{}")
            with self.assertRaisesRegex(SystemExit, "sealed"):
                Q.generation_args(L.root, "gqa_testdev", "0/6")


# ------------------------------------------------------------------------------------------------------------------------------ integrity gate
def reference(t: Path):
    """An R7-40c-like reference: D1-30-format sets (GQA DEV items on JPEGs named by image id; ChartQA val items with chartqa_val_<row> ids and
    charts written as the D1-30 build writes them), R7-40c's 50 core rows run by the R7-40c runner on the fake host, shard by shard (gqa_dev 6,
    chartqa_val 2; gqa_dev shard 3 on an H100, the others on L40S), and an R7-40c driver log with one RETRY line."""
    from PIL import Image
    sel = t / "sel"; T40.tiny_units(sel); main = P40.unit_names(LR); sets, pred, img = t / "ref" / "sets", t / "ref" / "pred", t / "images"
    sets.mkdir(parents=True); img.mkdir(); gi, ci, chart_rows, paths = [], [], [], {}
    for i in range(7):
        Image.new("RGB", (600 + 13 * i, 450), (20 * i, 90, 30)).save(img / f"{3000 + i}.jpg", "JPEG")
        for j in range(3):
            q = f"What is {i}-{j}?"
            gi.append({"bench": "gqa_dev", "qid": f"d{i}{j:02d}", "image_id": f"{3000 + i}", "image_file": str(img / f"{3000 + i}.jpg"), "question": q,
                       "answer": f"ans{i}{j}", "prompt": q + DD.SHORT})
    for i in range(10):
        name, buf = f"c{i // 2}.png", io.BytesIO(); Image.new("RGB", (700, 500 - 7 * (i // 2)), (0, 30 * (i // 2), 60)).save(buf, "PNG"); raw = buf.getvalue()
        DD.write_image("chartqa_val", name, raw, sets, paths); q = f"How many {i}?"
        chart_rows.append({"imgname": name, "query": q, "label": str(i), "type": "human_test" if i % 2 == 0 else "augmented_test", "image": raw})
        ci.append({"bench": "chartqa_val", "qid": f"chartqa_val_{i:05d}", "image_id": name, "image_file": paths[name], "question": q, "answer": str(i), "prompt": q + DD.CHART_SUFFIX})
    (sets / "gqa_dev.jsonl").write_text(T.jsonl(gi)); (sets / "chartqa_val.jsonl").write_text(T.jsonl(ci))
    core = P40.config_list(main); cf = t / "ref" / "configs_core.txt"; cf.write_text("\n".join(core) + "\n")
    units = sorted({P40.parse(c)[1] for c in core if c.startswith(("sel:", "dyn:"))})
    for v, n in (("gqa_dev", 6), ("chartqa_val", 2)):
        for i in range(n):
            run_unit(Namespace(sets_dir=str(sets), bench=v, configs_file=str(cf), shard=f"{i}/{n}", out=str(pred), chunk=32, batch=32, limit=0,
                               selectors_dir=str(sel), pca=str(sel / "pca.pt")), units, CLASS["H100"] if (v, i) == ("gqa_dev", 3) else CLASS["L40S"])
    log = t / "ref" / "r40c.log"; log.write_text(f"x [40c] RETRY 1/2: job 7 left the queue without {pred / 'gqa_dev' / 'done_shard1of6.json'}\n")
    return {"sel": sel, "main": main, "sets": sets, "pred": pred, "img": img, "log": log, "chart_rows": chart_rows}


def integrity_inputs(root, ref):
    """The gate inputs as the development-mode build writes them (test bench names, R7-40c's ids, no answers), for envs without pyarrow."""
    IB = T.Layout(T.Layout(root, "Q").integrity, "Q"); benches = {}
    for b, v in Q.VAL.items():
        items = [dict({k: x for k, x in it.items() if k != "answer"}, bench=b) for it in T.read_jsonl_strict(ref["sets"] / f"{v}.jsonl")]
        T.write_atomic(IB.inputs_file(b), T.jsonl(items)); benches[b] = {"inputs_sha256": T.sha(IB.inputs_file(b)), "items": len(items)}
    T.write_atomic(IB.summary, json.dumps({"ok": True, "benches": benches}))
    return IB


def plan_args(ref):
    return (ref["sel"] / "lr_choice.json", ref["sel"], ref["sel"] / "pca.pt", ref["pred"], ref["sets"], ref["log"])


def run_gate(root, ref):
    """Our side of the gate: every planned unit through its guard (on its pinned GPU class) and the R7-40c runner."""
    plan = json.loads((T.Layout(root, "Q").integrity / "plan.json").read_text())
    for u in plan["units"].values():
        run_unit(ns(Q.integrity_args(root, u["bench"], u["shard"], device_name=CLASS[u["class"]])), [ref["main"]["vcs"][0]], CLASS[u["class"]])
    return plan


class IntegrityGate(unittest.TestCase):
    @unittest.skipUnless(HAS_PYARROW, "the development-mode build reads the ChartQA val parquet (SparseVLM env)")
    def test_prepare_build_plan_and_pass(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); ref = reference(t); pq.write_table(pa.Table.from_pylist(ref["chart_rows"]), t / "val-00000-of-00001.parquet")
            u = Q.integrity_prepare(t / "root", *plan_args(ref), chartqa_src=t / "val-00000-of-00001.parquet", images_dir=ref["img"])
            self.assertEqual(len(u), 8); self.assertEqual(u["gqa_testdev_3of6"], "H100"); self.assertEqual(sum(c == "L40S" for c in u.values()), 7)
            IB = T.Layout(T.Layout(t / "root", "Q").integrity, "Q"); mine = T.read_jsonl_strict(IB.inputs_file("chartqa_test"))
            self.assertEqual([it["qid"] for it in mine], [f"chartqa_val_{i:05d}" for i in range(10)]); self.assertTrue(all("answer" not in it for it in mine))
            self.assertNotEqual(mine[0]["image_file"], T.read_jsonl_strict(ref["sets"] / "chartqa_val.jsonl")[0]["image_file"])   # extracted copy, same bytes
            run_gate(t / "root", ref); self.assertEqual(Q.integrity_compare(t / "root"), 0)
            res = json.loads((IB.root / "integrity.json").read_text()); self.assertTrue(res["pass"])
            self.assertEqual(res["benches"]["gqa_testdev"]["inputs_identical"], {"question": 21, "prompt": 21, "image": 21})
            self.assertEqual(res["benches"]["chartqa_test"]["inputs_identical"], {"question": 10, "prompt": 10, "image": 10})

    def test_plan_pins_guard_compare_and_every_stop(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); ref = reference(t); root = t / "root"; IB = integrity_inputs(root, ref); base = IB.root; s1 = ref["main"]["vcs"][0]
            u = Q.integrity_plan(root, *plan_args(ref)); plan = json.loads((base / "plan.json").read_text())
            self.assertEqual(sorted(u), sorted([f"gqa_testdev_{i}of6" for i in range(6)] + ["chartqa_test_0of2", "chartqa_test_1of2"]))
            self.assertEqual(plan["rows"], ["full", "static:grid:0.25", f"sel:{s1}:0.25"]); self.assertEqual(P40.read_configs(IB.configs_file("gqa_testdev")), plan["rows"])
            self.assertTrue(plan["units"]["gqa_testdev_1of6"]["reference_retried"]); self.assertFalse(plan["units"]["gqa_testdev_2of6"]["reference_retried"])
            self.assertTrue(all(x["reference_resumed"] is False for x in plan["units"].values())); self.assertEqual(plan["units"]["gqa_testdev_0of6"]["reference_configs"], 50)
            lines = (base / "lines.txt").read_text().splitlines()[1:]; self.assertEqual(len(lines), 8)
            self.assertEqual(sum("--partition=H100 --exclude=node51,node60 " in x for x in lines), 1); self.assertEqual(sum("--partition=L40S --exclude=node51,node60 " in x for x in lines), 7)
            self.assertTrue(all("--time=00:45:00" in x and "slurm/r7_50_q_gpu.sbatch integrity " in x and "--export" not in x for x in lines))
            self.assertIn(f"integrity {root} gqa_testdev 3/6 #MARK {IB.pred / 'gqa_testdev' / 'done_shard3of6.json'}", [x for x in lines if "H100" in x][0])
            for dev, msg in (("NVIDIA H100 NVL", "pinned to L40S"), ("NVIDIA RTX PRO 6000 Blackwell", "unexpected or missing GPU class"), (None, "no CUDA")):
                with self.assertRaisesRegex(SystemExit, msg):
                    Q.integrity_args(root, "gqa_testdev", "0/6", device_name=dev or "")
            with self.assertRaisesRegex(SystemExit, "pinned to H100"):
                Q.integrity_args(root, "gqa_testdev", "3/6", device_name="NVIDIA L40S")
            js = ref["sel"] / f"{s1}.json"; keep = js.read_bytes(); js.write_bytes(keep + b" ")
            with self.assertRaisesRegex(SystemExit, "differ from their recorded sha256"):
                Q.integrity_args(root, "chartqa_test", "0/2", device_name="NVIDIA L40S")
            js.write_bytes(keep); a = Q.integrity_args(root, "chartqa_test", "0/2", device_name="NVIDIA L40S")
            self.assertEqual(a[a.index("--sets-dir") + 1], str(IB.inputs)); self.assertEqual(a[a.index("--out") + 1], str(IB.pred))
            run_gate(root, ref); self.assertEqual(Q.integrity_compare(root), 0); res = json.loads((base / "integrity.json").read_text())
            self.assertTrue(res["pass"]); self.assertTrue(all(x["ok"] and x["resumed_here"] is False for x in res["units"].values()))
            for b, n in (("gqa_testdev", 21), ("chartqa_test", 10)):
                self.assertEqual(res["benches"][b]["inputs_identical"], {"question": n, "prompt": n, "image": n}); self.assertTrue(res["benches"][b]["reference_sets_match_r7_40c_markers"])
                for c in plan["rows"]:
                    x = res["benches"][b][c]; self.assertEqual((x["identical"], x["items"], x["n_vis_identical"]), (n, n, n)); self.assertEqual(len(x["files_reference"]), Q.SHARDS[b])

            def stop(path, edit, check):
                """Edit one file, the gate must STOP (return 1, top-level pass false, no prediction text in the record), restore, the gate passes again."""
                keep = path.read_bytes(); edit(path)
                try:
                    self.assertEqual(Q.integrity_compare(root), 1); r = json.loads((base / "integrity.json").read_text()); self.assertFalse(r["pass"]); check(r)
                    self.assertNotIn("FLIPPED", (base / "integrity.json").read_text())
                finally:
                    path.write_bytes(keep)
                self.assertEqual(Q.integrity_compare(root), 0)

            def lines_edit(i, **kv):
                def f(p):
                    rows = T.read_jsonl_strict(p); rows[i].update(kv); p.write_text(T.jsonl(rows))
                return f
            g = IB.pred / "gqa_testdev"; first = lambda p: T.read_jsonl_strict(p)[0]["qid"]  # noqa: E731
            grid_file = g / "static__grid__0.25@s0of6.jsonl"
            stop(grid_file, lines_edit(0, pred="FLIPPED"), lambda r: self.assertEqual(
                (r["benches"]["gqa_testdev"]["static:grid:0.25"]["answer_differs"], r["benches"]["gqa_testdev"]["static:grid:0.25"]["mismatched_qids_first_20"]), (1, [first(grid_file)])))
            stop(g / "full@s2of6.jsonl", lines_edit(1, kept=1), lambda r: self.assertEqual(r["benches"]["gqa_testdev"]["full"]["kept_differs"], 1))
            stop(IB.inputs_file("chartqa_test"), lines_edit(4, question="How many 99?"), lambda r: self.assertEqual(r["benches"]["chartqa_test"]["inputs_identical"]["question"], 9))
            stop(IB.inputs_file("gqa_testdev"), lines_edit(2, image_file=str(ref["img"] / "3001.jpg")),
                 lambda r: self.assertEqual(r["benches"]["gqa_testdev"]["inputs_identical"]["image"], 20))
            sel_file = IB.pred / "chartqa_test" / f"sel__{s1}__0.25@s1of2.jsonl"
            stop(sel_file, lambda p: p.write_text("".join(p.read_text().splitlines(True)[1:])),
                 lambda r: self.assertEqual(r["benches"]["chartqa_test"][f"sel:{s1}:0.25"]["missing_here"], 1))
            stop(ref["pred"] / "gqa_dev" / "full@s4of6.jsonl", lambda p: p.write_text("".join(p.read_text().splitlines(True)[:-1])),
                 lambda r: self.assertEqual(r["benches"]["gqa_testdev"]["full"]["missing_reference"], 1))
            stop(g / "done_shard0of6.json", lambda p: p.write_text(p.read_text().replace("NVIDIA L40S", "NVIDIA H100 NVL")),
                 lambda r: self.assertEqual((r["units"]["gqa_testdev_0of6"]["class_here"], r["units"]["gqa_testdev_0of6"]["ok"]), ("H100", False)))
            stop(ref["sets"] / "gqa_dev.jsonl", lambda p: p.write_text(p.read_text() + "\n"),
                 lambda r: self.assertFalse(r["benches"]["gqa_testdev"]["reference_sets_match_r7_40c_markers"]))
            mk = g / "done_shard5of6.json"; keep = mk.read_bytes(); mk.unlink()
            self.assertEqual(Q.integrity_compare(root), 1); self.assertFalse(json.loads((base / "integrity.json").read_text())["units"]["gqa_testdev_5of6"]["marker_present"])
            mk.write_bytes(keep)
            rm = ref["pred"] / "gqa_dev" / "done_shard2of6.json"; m = json.loads(rm.read_text()); m["answers_this_session"] = 7; rm.write_text(json.dumps(m))
            Q.integrity_plan(root, *plan_args(ref)); self.assertEqual(Q.integrity_compare(root), 0)                 # a resumed reference shard is reported, not waived
            self.assertTrue(json.loads((base / "integrity.json").read_text())["units"]["gqa_testdev_2of6"]["reference_resumed"])
            (ref["pred"] / "chartqa_val" / "done_shard1of2.json").unlink()
            with self.assertRaisesRegex(SystemExit, "reference marker missing"):
                Q.integrity_plan(root, *plan_args(ref))


# ------------------------------------------------------------------------------------------------------------------------------ manifest / score
def synth_q_preds(L, rec, seed=0):
    """Runner-format predictions of the 41 rows (D1-30 sharding, one file per row and shard) with accuracy rising with the kept fraction (S1 a
    little higher), and runner-format shard markers (configs_Q, the sealed inputs and the recorded selector hashes)."""
    g = np.random.default_rng(seed); acc = {0.125: 0.45, 0.25: 0.55, 0.5: 0.7, 0.75: 0.8, 1.0: 0.9}
    lab = {b: {r["qid"]: r for r in T.read_jsonl_strict(L.labels_file(b))} for b in Q.BENCHES}
    for b in Q.BENCHES:
        cfgs, n, od = P40.read_configs(L.configs_file(b)), Q.SHARDS[b], L.pred / b; od.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            items = RN.load_items(L.inputs, b, f"{i}/{n}")
            for c in cfgs:
                k, a, f = P40.parse(c); f = 1.0 if k == "full" else f; rows = []
                for it in items:
                    nv = 300 + int(hashlib.sha256(it["image_id"].encode()).hexdigest(), 16) % 50; p = acc[f] + (0.05 if k == "sel" and a.startswith("vcs") else 0.0)
                    rows.append({"qid": it["qid"], "image_id": it["image_id"], "pred": lab[b][it["qid"]]["answer"] if g.uniform() < p else "wrong",
                                 "kept": nv if f >= 1 else M4.budget(f, nv), "n_vis": nv, "f": f, "n_new": 2, "hit_max": False})
                (od / f"{RUN.cfg_file(c)}@s{i}of{n}.jsonl").write_text(T.jsonl(rows))
            sf = {"pca": {"path": rec["pca"]["file"], "sha256": rec["pca"]["sha256"]}, **{u: {"pt_sha256": r["pt_sha256"], "json_sha256": r["json_sha256"]} for u, r in rec["units"].items()}}
            (od / f"done_shard{i}of{n}.json").write_text(json.dumps({"items": len(items), "configs": len(cfgs), "seconds": 1.0, "answers_this_session": len(items) * len(cfgs),
                                                                     "env": {"gpu": "NVIDIA L40S"}, "sets_file_sha256": T.sha(L.inputs_file(b)),
                                                                     "configs_sha256": T.sha(L.configs_file(b)), "selector_files": sf}))


def bundle(t: Path, rep, b_star=None, report_b_star="same"):
    """R7-40c's timing bundle as r7_d1_40c_timing.bundle writes it (BSTAR=report: the reused R7-40a report as part 'b_star', then gqa_dev,
    textvqa_val, chartqa_val) and the R7-40a report; rows: full, the uniform grid and the representative S1 seed-0 unit at B = 1 / 4 / 16 / 32."""
    td, rp = t / "tim", t / "r40" / "timing" / "timing_report.json"; rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps({"gpu": "NVIDIA H100 NVL", "host": "nodesumo01", "B_star": b_star if report_b_star == "same" else report_b_star, "rule": "test"}))
    lat = {0.125: 40.0, 0.25: 55.0, 0.5: 70.0, 0.75: 85.0}; parts = [{"part": "b_star", "status": "reused existing R7-40a report", "file": str(rp), "sha256": T.sha(rp)}]
    for v in ("gqa_dev", "textvqa_val", "chartqa_val"):
        rows = []
        for B in (1, 4, 16, 32):
            rows.append({"config": "full", "B": B, "per_item_ms_median": 100.0, "per_item_ms_median_ci95": [99.0, 101.0]})
            for f, x in lat.items():
                rows.append({"config": f"static:grid:{P40.fkey(f)}", "B": B, "per_item_ms_median": x, "per_item_ms_median_ci95": [x - 1, x + 1]})
                rows.append({"config": f"sel:{rep}:{P40.fkey(f)}", "B": B, "per_item_ms_median": x + 2.0, "per_item_ms_median_ci95": [x + 1, x + 3]})
        (td / v).mkdir(parents=True, exist_ok=True); p = td / v / f"timing_{v}.json"
        p.write_text(json.dumps({"rows": rows, "b_star": b_star, "primary_batch": b_star or 4, "gpu": "NVIDIA H100 NVL", "host": "nodesumo01"}))
        parts.append({"part": v, "status": "measured", "file": str(p), "sha256": T.sha(p)})
    (td / "timing_bundle.done.json").write_text(json.dumps({"parts": parts, "gpu": "NVIDIA H100 NVL", "host": "nodesumo01", "job": "1036770"}))
    return td, rp


def scored_root(t: Path, b_star=None, **kw):
    L, rec, labels = configured(t, n_img=10, q_per=4, n_chart=24); gate_pass(L); synth_q_preds(L, rec); td, rp = bundle(t, rec["representative_unit"], b_star, **kw)
    (t / "p40.yaml").write_text("status: frozen\nr7_40c_staged_freeze:\n  by: test\n")
    return L, rec, td, rp, labels


class ManifestAndScore(unittest.TestCase):
    def test_manifest_contents_and_refusals(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); L, rec, td, rp, labels = scored_root(t)
            victim = L.pred / "chartqa_test" / "static__grid__0.5@s1of2.jsonl"; keep = victim.read_text(); victim.write_text("".join(keep.splitlines(True)[:-1]))
            with self.assertRaisesRegex(SystemExit, "coverage"):
                Q.manifest(L.root, td, rp)
            victim.write_text(keep); mark = td / "timing_bundle.done.json"; mk = mark.read_text(); mark.unlink()
            with self.assertRaisesRegex(SystemExit, "missing"):
                Q.manifest(L.root, td, rp)
            mark.write_text(mk); tg = td / "gqa_dev" / "timing_gqa_dev.json"; keep = tg.read_text(); tg.write_text(keep + " ")
            with self.assertRaisesRegex(SystemExit, "differs from the sha256 recorded"):
                Q.manifest(L.root, td, rp)                                                   # a timing file changed after the bundle wrote it
            tg.write_text(keep); mark.write_text(mk.replace(str(rp), str(t / "elsewhere.json")))
            with self.assertRaisesRegex(SystemExit, "part 'b_star' does not list"):
                Q.manifest(L.root, td, rp)                                                   # the bundle's B* must be the report the scorer reads
            mark.write_text(mk); sm = L.pred / "gqa_testdev" / "done_shard2of6.json"; keep = sm.read_text(); sm.write_text(keep.replace(T.sha(L.configs_file("gqa_testdev")), "0" * 64))
            with self.assertRaisesRegex(SystemExit, "other configs or inputs"):
                Q.manifest(L.root, td, rp)
            m = json.loads(keep); m["selector_files"][rec["representative_unit"]]["pt_sha256"] = "1" * 64; sm.write_text(json.dumps(m))
            with self.assertRaisesRegex(SystemExit, "selector / PCA files differ"):
                Q.manifest(L.root, td, rp)
            m["selector_files"] = None; m["seconds"] = None; m["answers_this_session"] = None; sm.write_text(json.dumps(m))   # rewritten by a no-op session
            self.assertFalse(L.manifest.exists())
            for b in Q.BENCHES:
                os.chmod(L.labels_file(b), 0)                                                 # the manifest never opens a label file
            try:
                digest, man = Q.manifest(L.root, td, rp)
            finally:
                for b in Q.BENCHES:
                    os.chmod(L.labels_file(b), 0o400)
            self.assertEqual(digest, T.sha(L.manifest)); self.assertTrue(L.manifest_sha.read_text().startswith(digest)); self.assertEqual(man["phase"], "Q")
            self.assertEqual({b: v["n_rows"] for b, v in man["rows"].items()}, {b: 41 for b in Q.BENCHES}); self.assertEqual(man["n_files"], 41 * 8 + 8)
            self.assertEqual(set(man["profiling_timing"]["files"]), {"timing_bundle.done.json", "timing_report.json", "timing_gqa_dev.json", "timing_chartqa_val.json",
                                                                     "timing_textvqa_val.json"})
            for k, v in man["profiling_timing"]["files"].items():
                self.assertEqual(v["sha256"], T.sha(v["path"]))
            self.assertEqual(man["profiling_timing"]["files"]["timing_report.json"]["path"], str(rp)); self.assertEqual(man["profiling_timing"]["map"], Q.VAL)
            self.assertEqual((man["rows_json_sha256"], man["build_summary_sha256"], man["integrity_gate"], man["integrity_json_sha256"]),
                             (T.sha(L.configs / "rows.json"), T.sha(L.summary), True, T.sha(L.integrity / "integrity.json")))
            self.assertEqual(len(man["shard_markers"]["gqa_testdev"]), 6); self.assertFalse(man["shard_markers"]["gqa_testdev"]["done_shard2of6.json"]["selector_record"])
            self.assertEqual(man["coverage"]["chartqa_test"]["full"]["answers"], 24); self.assertEqual(man["addendum"]["status"], "frozen_20261013")
            mt = L.manifest.read_text(); self.assertFalse(any(f'"{a["answer"]}"' in mt for a in labels["gqa_testdev"] if a["answer"] != "yes"))
            with self.assertRaisesRegex(SystemExit, "already sealed"):
                Q.manifest(L.root, td, rp)
            with self.assertRaisesRegex(SystemExit, "sealed"):
                Q.generation_args(L.root, "gqa_testdev", "0/6")                              # no generation after hashing

    def test_score_gate_fraction_basis_holm6(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); L, rec, td, rp, _ = scored_root(t); digest, _ = Q.manifest(L.root, td, rp); main = P40.unit_names(LR); rep = rec["representative_unit"]
            sc = lambda out, **kw: Q.score(L.root, td, rp, 200, t / out, protocol_40c=t / "p40.yaml", **kw)  # noqa: E731
            with self.assertRaisesRegex(SystemExit, "absent"):
                sc("o0")
            L.allow.write_text("manifest_sha256=" + "0" * 64 + "\n")
            with self.assertRaisesRegex(SystemExit, "labels stay sealed"):
                sc("o1")
            L.allow.write_text(f"# coordinator\nmanifest_sha256={digest}\n"); f = L.pred / "chartqa_test" / "full@s0of2.jsonl"; orig = f.read_bytes(); f.write_bytes(orig + b"\n")
            with self.assertRaisesRegex(SystemExit, "differ from the manifest"):
                sc("o2")
            f.write_bytes(orig); tc = td / "textvqa_val" / "timing_textvqa_val.json"; keep = tc.read_text(); tc.write_text(keep + " ")
            with self.assertRaises(SystemExit):
                sc("o3")                                                                     # every sealed timing file must still match
            tc.write_text(keep); (t / "p40.yaml").write_text("status: frozen\n")
            with self.assertRaisesRegex(SystemExit, "not frozen"):
                sc("o4")
            (t / "p40.yaml").write_text("r7_40c_staged_freeze: x\n")
            repo = t / "repo"; repo.mkdir(); git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True).stdout.strip()  # noqa: E731
            git("init", "-q"); git("config", "user.email", "t@t"); git("config", "user.name", "t"); (repo / "m.json").write_bytes(L.manifest.read_bytes())
            git("add", "m.json"); git("commit", "-q", "-m", "m"); head = git("rev-parse", "HEAD")
            with mock.patch.object(T, "REPO", repo):
                with self.assertRaisesRegex(SystemExit, "do not hold the manifest"):
                    sc("o5", require_commit=True)                                            # production rule: the manifest must be committed
                L.allow.write_text(f"manifest_sha256={digest}\ncommit={head}\nmanifest_copy=m.json\n"); res = sc("o6", require_commit=True)
            self.assertFalse(any((t / f"o{i}").exists() for i in range(6)))
            self.assertEqual((res["primary_basis"], res["b_star"], res["primary_points"]), ("fraction", None, ["f0.25", "f0.5", "f0.75"]))
            self.assertEqual(res["b_star_source"]["file"], str(rp)); self.assertEqual(res["representative_latency_unit"], main["vcs"][0])
            for b, r in res["benches"].items():
                self.assertEqual(len(r["fixed_fraction"]), 41); self.assertAlmostEqual(r["fixed_fraction"]["static:grid:0.25"]["mean_kept_fraction"], 0.25, delta=0.01)
                prim = {k: v for k, v in r["contrasts"].items() if v.get("role") == "primary"}
                self.assertEqual(sorted(prim), sorted(f"S1-{o}@f{x}" for o in ("S2", "S3", "grid") for x in ("0.25", "0.5", "0.75")))
                for v in prim.values():
                    self.assertEqual(v["holm_family_size"], 6); self.assertGreaterEqual(v["p_holm"], v["p"] - 1e-12); self.assertEqual(len(v["per_seed"]), 3)
                    self.assertTrue(v["ci95"][0] - 1e-9 <= v["delta"] <= v["ci95"][1] + 1e-9)
                self.assertEqual(r["contrasts"]["S1-S2@f0.125"]["role"], "secondary"); self.assertEqual((r["contrasts"]["S1-grid@L50"]["role"], r["contrasts"]["S1-grid@L50"]["batch"]), ("secondary", 4))
                rl = r["row_latency_profiling_workload_ms"]
                self.assertEqual(rl[f"sel:{main['logistic'][2]}:0.25"]["B4"]["median"], 57.0); self.assertEqual(rl[f"sel:{main['direct'][0]}:0.75"]["B1"]["ci95"], [86.0, 88.0])
                self.assertEqual(rl["static:grid:0.5"]["B16"]["median"], 70.0); self.assertEqual(set(rl["full"]), {"B1", "B4", "B16", "B32"})
                self.assertEqual(r["profiling_timing"]["validation_bench"], Q.VAL[b]); self.assertNotIn("official_anchor", r); self.assertEqual(r["dynamic_fraction_distribution"], {})
            self.assertEqual(set(res["benches"]["gqa_testdev"]["subsets"]), {"query", "verify"})
            sp = res["benches"]["chartqa_test"]["splits_secondary"]; self.assertEqual((sp["human"]["n"], sp["augmented"]["n"]), (12, 12))
            self.assertTrue(all(v["role"] == "secondary (subset)" for v in sp["human"]["contrasts_secondary"].values()))
            self.assertEqual(len(sp["mean_of_splits"]), 41); self.assertEqual(set(res["benches"]["chartqa_test"]["subsets"]), {"human", "augmented"})
            self.assertIn("profiling workload", res["latency_label"]); self.assertEqual(len(res["exposure_statement"]), 2); self.assertIn("undisclosed", res["exposure_statement"][1])
            md = (t / "o6" / "r7_50_results.md").read_text(); self.assertIn("phase Q", md); self.assertIn("| gqa_testdev | S1-S3@f0.5 |", md)
            for b in Q.BENCHES:
                os.chmod(L.labels_file(b), 0)
            try:
                with self.assertRaisesRegex(SystemExit, "exists"):
                    sc("o6")                                                                 # never overwrites a readout, refused before any label is opened
            finally:
                for b in Q.BENCHES:
                    os.chmod(L.labels_file(b), 0o400)

    def test_latency_basis_and_b_star_mismatch(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); L, rec, td, rp, _ = scored_root(t, b_star=4); digest, _ = Q.manifest(L.root, td, rp); L.allow.write_text(f"manifest_sha256={digest}\n")
            res = Q.score(L.root, td, rp, 200, t / "o", protocol_40c=t / "p40.yaml")
            self.assertEqual((res["primary_basis"], res["b_star"]), ("latency", 4))
            for r in res["benches"].values():
                prim = {k: v for k, v in r["contrasts"].items() if v.get("role") == "primary"}
                self.assertEqual(sorted(prim), sorted(f"S1-{o}@{x}" for o in ("S2", "S3", "grid") for x in ("L25", "L50", "L75")))
                self.assertTrue(all(v["holm_family_size"] == 6 and v["batch"] == 4 for v in prim.values())); self.assertEqual(r["latency_points_ms"][4], {"L25": 55.0, "L50": 70.0, "L75": 85.0})
                self.assertEqual(r["contrasts"]["S1-S3@f0.5"]["role"], "secondary")
        with tempfile.TemporaryDirectory() as t:                                              # the timing files' B* must equal the R7-40a report's
            t = Path(t); L, rec, td, rp, _ = scored_root(t, b_star=None, report_b_star=4); digest, _ = Q.manifest(L.root, td, rp); L.allow.write_text(f"manifest_sha256={digest}\n")
            with self.assertRaisesRegex(SystemExit, "B\\*"):
                Q.score(L.root, td, rp, 200, t / "o", protocol_40c=t / "p40.yaml")
            self.assertFalse((t / "o").exists())


# ------------------------------------------------------------------------------------------------------------------------------ driver
STUB_SBATCH = r'''#!/usr/bin/env bash
ST=$STUB_STATE; mkdir -p "$ST"; n=$(( $(cat "$ST/n" 2>/dev/null || echo 2000) + 1 )); echo $n > "$ST/n"; echo "$*" >> "$ST/calls"
[ "${STUB_NOID:-0}" = 1 ] && exit 0
args=("$@"); s=""; rest=()
for i in "${!args[@]}"; do case "${args[$i]}" in *.sbatch) s=${args[$i]}; rest=("${args[@]:$((i+1))}");; esac; done
case "$s" in
  *r7_50_q_cpu.sbatch)
    R=${rest[1]}; I="$R/integrity_Q"
    case "${rest[0]}" in
      integrity-prepare) mkdir -p "$I"; { echo "# stub"; for bk in gqa_testdev:6 chartqa_test:2; do b=${bk%:*}; k=${bk#*:}; for i in $(seq 0 $((k - 1))); do
          echo "sbatch --parsable --job-name=r7_50iq_${b}_$i --partition=L40S --exclude=node51,node60 --time=00:45:00 slurm/r7_50_q_gpu.sbatch integrity $R $b $i/$k #MARK $I/sealed/pred_Q/$b/done_shard${i}of${k}.json"
          done; done; } > "$I/lines.txt";;
      integrity-compare) if [ "${STUB_INTEGRITY:-pass}" = pass ]; then echo '{"pass": true}' > "$I/integrity.json"
                         else printf '{"benches": {"gqa_testdev": {"full": {"ok": true, "pass": true}}},\n "pass": false}\n' > "$I/integrity.json"; echo $n; exit 1; fi;;
      configs) "$STUB_PY" -I scripts/r7_50_q.py lines --root "$R" > /dev/null;;
      manifest) M="$R/manifest_Q"; mkdir -p "$M"; echo '{"files": [], "phase": "Q"}' > "$M/predictions_manifest.json"; h=$(sha256sum "$M/predictions_manifest.json" | cut -d' ' -f1)
                echo "$h  predictions_manifest.json" > "$M/predictions_manifest.sha256"
                case "${STUB_ALLOW:-none}" in good) printf '# test\nmanifest_sha256=%s\ncommit=abc\nmanifest_copy=reports/R7/x.json\n' "$h" > "$R/ALLOW_SCORING_Q";;
                                              bad) echo "manifest_sha256=00" > "$R/ALLOW_SCORING_Q";; esac;;
      score) mkdir -p "$R/results_Q"; echo '{}' > "$R/results_Q/r7_50_results.json";;
    esac;;
  *r7_50_q_gpu.sbatch)
    if [ "${rest[0]}" = integrity ]; then R=${rest[1]}; b=${rest[2]}; sh=${rest[3]}; m="$R/integrity_Q/sealed/pred_Q/$b/done_shard${sh%/*}of${sh#*/}.json"; key="integrity_${b}_${sh%/*}"
    else b=${rest[1]}; sh=${rest[2]}; R=${rest[3]}; m="$R/sealed/pred_Q/$b/done_shard${sh%/*}of${sh#*/}.json"; key="run_${b}_${sh%/*}"; fi
    t=$(( $(cat "$ST/try_$key" 2>/dev/null || echo 0) + 1 )); echo $t > "$ST/try_$key"
    if [ "$key" = "${STUB_FAIL_UNIT:-}" ] && [ "$t" -le "${STUB_FAIL_TIMES:-0}" ]; then :; else mkdir -p "$(dirname "$m")"; echo '{}' > "$m"; fi;;
esac
echo $n
'''


class DriverDryRun(unittest.TestCase):
    """slurm/r7_50_q_submit.sh with stubbed sbatch / squeue (R750_DRYRUN=1, temporary roots; nothing is submitted)."""

    def env(self, t: Path, **kw):
        skip = {k for k in ("_no_inputs", "_no_done", "_no_bundle", "_incomplete", "_draft_add") if kw.pop(k, None)}
        b = t / "bin"; b.mkdir(exist_ok=True); (b / "sbatch").write_text(STUB_SBATCH); (b / "squeue").write_text("#!/usr/bin/env bash\nexit 0\n")
        for f in ("sbatch", "squeue"):
            os.chmod(b / f, 0o755)
        if "_no_inputs" not in skip:
            (t / "root" / "sealed" / "inputs").mkdir(parents=True, exist_ok=True); (t / "root" / "sealed" / "inputs" / "build_summary.json").write_text('{"ok": true}')
        r40 = t / "d1" / "r7_40"; sel = r40 / "selectors"; sel.mkdir(parents=True, exist_ok=True); (sel / "lr_choice.json").write_text(json.dumps(LR))
        for u in [u for v in P40.unit_names(LR).values() for u in v]:
            (sel / f"{u}.pt").write_bytes(b"x"); (sel / f"{u}.json").write_text("{}")
        (r40 / "timing").mkdir(exist_ok=True); (r40 / "timing" / "timing_report.json").write_text("{}"); rc = r40 / "eval_40c"
        (rc / "timing").mkdir(parents=True, exist_ok=True); (rc / "completeness.json").write_text(json.dumps({"complete": "_incomplete" not in skip}))
        for v, n in (("gqa_dev", 6), ("chartqa_val", 2)):
            (rc / "pred" / v).mkdir(parents=True, exist_ok=True); (rc / "timing" / v).mkdir(exist_ok=True); (rc / "timing" / v / f"timing_{v}.json").write_text("{}")
            for i in range(n):
                (rc / "pred" / v / f"done_shard{i}of{n}.json").write_text("{}")
        if "_no_bundle" not in skip:
            (rc / "timing" / "timing_bundle.done.json").write_text("{}")
        (t / "r40c.log").write_text("x [40c] start\n" + ("" if "_no_done" in skip else "y [40c] done: R7-40c generation complete; scoring is the coordinator's step\n"))
        (t / "proto.yaml").write_text(FROZEN); (t / "add.yaml").write_text("status: draft\n" if "_draft_add" in skip else ADD_FROZEN)
        (t / "p40.yaml").write_text("r7_40c_staged_freeze:\n  by: test\n")
        e = dict(os.environ, PATH=f"{b}:{os.environ['PATH']}", STUB_STATE=str(t / "state"), STUB_PY=sys.executable, PY=sys.executable, R750_DRYRUN="1",
                 R750_ROOT=str(t / "root"), D1_ROOT=str(t / "d1"), R40C_LOG=str(t / "r40c.log"), SUBMIT_LOG=str(t / "drv.log"), PROTOCOL=str(t / "proto.yaml"),
                 ADDENDUM=str(t / "add.yaml"), PROTOCOL_40C=str(t / "p40.yaml"), POLL="0", WAIT_H="1", WAIT_SCORE_H="1", RETRIES="2", SOON_S="0", SOON_POLL="1",
                 STUB_ALLOW="good")
        e.update({k: str(v) for k, v in kw.items()})
        return e

    def drive(self, t, **kw):
        p = subprocess.run(["bash", str(R / "slurm" / "r7_50_q_submit.sh")], env=self.env(t, **kw), cwd=str(R), capture_output=True, text=True, timeout=300)
        log = (t / "drv.log").read_text() if (t / "drv.log").exists() else ""; calls = (t / "state" / "calls").read_text() if (t / "state" / "calls").exists() else ""
        return p.returncode, log, calls

    def test_full_pass_with_integrity_and_one_retry(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); rc, log, calls = self.drive(t, STUB_FAIL_UNIT="run_chartqa_test_1", STUB_FAIL_TIMES=1)
            self.assertEqual(rc, 0, log); self.assertIn("RETRY 1/2", log); self.assertIn("integrity complete (8 units)", log); self.assertIn("integrity gate passed", log)
            self.assertIn("generation Q complete (8 units)", log); self.assertIn("R7-40c complete", log); self.assertIn("done:", log)
            self.assertLess(calls.index("integrity-compare"), calls.index("r7_50_q_cpu.sbatch configs"))      # the gate precedes any test generation
            self.assertEqual(calls.count("r7_50_q_cpu.sbatch score"), 1); self.assertEqual(calls.count("r7_50_q_gpu.sbatch integrity"), 8)
            self.assertEqual(calls.count("r7_50_q_gpu.sbatch run"), 9); self.assertNotIn("--export", calls); self.assertNotIn("--dependency", calls)
            gpu = [c for c in calls.splitlines() if "r7_50_q_gpu" in c]; self.assertTrue(all("--partition=L40S --exclude=node51,node60" in c for c in gpu))
            self.assertLess(log.index("stage 2 complete"), log.index("stage 3: ALLOW_SCORING_Q present")); self.assertLess(log.index("stage 3"), log.index("done:"))
            self.assertTrue((t / "root" / "results_Q" / "r7_50_results.json").exists()); self.assertIn("WAITING for", log)
            (t / "state" / "calls").unlink(); rc, log, calls = self.drive(t)                    # relaunched after scoring: scored once, nothing submitted
            self.assertEqual(rc, 0, log); self.assertIn("done already", log); self.assertNotIn("score", calls); self.assertNotIn("r7_50_q_gpu", calls)

    def test_waits_for_allow_scoring_and_refuses_to_score_without_it(self):
        for kw, msg in ((dict(STUB_ALLOW="none", WAIT_SCORE_H="0"), "ALLOW_SCORING_Q after 0 h"), (dict(STUB_ALLOW="bad"), "does not carry manifest_sha256")):
            with tempfile.TemporaryDirectory() as t:
                t = Path(t); rc, log, calls = self.drive(t, **kw)
                self.assertEqual(rc, 1, log); self.assertIn("WAITING for", log); self.assertIn(msg, log); self.assertIn("r7_50_q_cpu.sbatch manifest", calls)
                self.assertNotIn("r7_50_q_cpu.sbatch score", calls); self.assertFalse((t / "root" / "results_Q").exists())

    def test_stops(self):
        cases = [(dict(PHASE="L"), "phase Q only", None),
                 (dict(PROTOCOL="/nonexistent.yaml"), "not a frozen R7-50 protocol", None),
                 (dict(_draft_add=1), "not the frozen phase-Q addendum", None),
                 (dict(ADDENDUM="/nonexistent.yaml"), "not the frozen phase-Q addendum", None),
                 (dict(PROTOCOL_40C="/nonexistent.yaml"), "R7-40c is not frozen", None),
                 (dict(R750_DRYRUN="0"), "not committed", None),
                 (dict(R750_ROOT=str(T.ROOT)), "refuses the production root", None),
                 (dict(_no_inputs=1), "phase L's sealed inputs", None),
                 (dict(_no_done=1, WAIT_H="0"), "done: R7-40c generation complete' not in", None),
                 (dict(_incomplete=1, WAIT_H="0"), "no 'complete': true", None),
                 (dict(_no_bundle=1, WAIT_H="0"), "timing_bundle.done.json missing", None),
                 (dict(STUB_NOID="1"), "no job id", "integrity-prepare"),
                 (dict(STUB_INTEGRITY="fail"), "full-scale integrity gate FAILED", "integrity-compare"),
                 (dict(STUB_FAIL_UNIT="integrity_chartqa_test_0", STUB_FAIL_TIMES=99), "integrity unit failed 3 times", "integrity-prepare"),
                 (dict(STUB_FAIL_UNIT="run_gqa_testdev_4", STUB_FAIL_TIMES=99), "generation Q unit failed 3 times", "configs")]
        for kw, msg, last in cases:
            with tempfile.TemporaryDirectory() as t:
                t = Path(t); rc, log, calls = self.drive(t, **kw)
                self.assertEqual(rc, 1, (kw, log)); self.assertIn("STOP", log, (kw, log)); self.assertIn(msg, log, (kw, log))
                self.assertNotIn("r7_50_q_cpu.sbatch score", calls, kw); self.assertNotIn("r7_50_q_cpu.sbatch manifest", calls, kw)   # labels never opened on a stop
                if last is None:
                    self.assertEqual(calls, "", kw)                                          # stopped before any submission
                else:
                    self.assertIn(f"r7_50_q_cpu.sbatch {last}", calls, kw)
                if "integrity" in msg:
                    self.assertNotIn("r7_50_q_cpu.sbatch configs", calls); self.assertNotIn("r7_50_q_gpu.sbatch run", calls)   # nothing generated on test


if __name__ == "__main__":
    unittest.main()
