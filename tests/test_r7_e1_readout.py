"""r7 E1 readout (scripts/r7_e1_readout.py): stubbed CPU tests on synthetic predictions — the freeze / completeness gates, refusal to overwrite,
the numbers (per E1 row quality and CI from the frozen Bench40 / item_scores / Boot, realised kept fraction, cut answers, retention vs the in-env
anchor and vs R7-40c's full_official, kept == rule, median ms), the env diagnostic (paired CI, identical answers, no p-value), the host-column
table (R7-40c grid / S1 / S2 / S3 seed means next to the E1 columns, f = 1 row), the absence of any contrast or test against S1 / S2 / S3 / grid,
the coverage / build-identity / configuration guards, a disclosed-gap readout (one method), the optional equality cross-check with R7-40c's own
readout, and the wrapper (outside slurm/r7_e1_*, frozen E1 code hash unchanged).  TextVQA (m4c evaluator) only where llava is importable
(SparseVLM env).  Run: python -I tests/test_r7_e1_readout.py -v"""
import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))
import r7_d1_30_run as RUN  # noqa: E402
import r7_d1_30_score as SC  # noqa: E402
import r7_d1_40_masks as M4  # noqa: E402
import r7_d1_40c_plan as PL  # noqa: E402
import r7_e1_external as EX  # noqa: E402
import r7_e1_readout as RO  # noqa: E402

try:
    import llava.eval.m4c_evaluator  # noqa: F401
    HAS_LLAVA = True
except Exception:
    HAS_LLAVA = False

LR = {"vcs": {"chosen_lr": 3e-4}, "logistic": {"chosen_lr": 1e-4}, "direct": {"chosen_lr": 1e-4}}
BENCHES = ["gqa_dev", "chartqa_val"] + (["textvqa_val"] if HAS_LLAVA else [])
REPS = 60


def h(*parts) -> int:
    return int(hashlib.sha256("/".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def gold(bench, it):
    return it["answers"][0] if bench.startswith("textvqa") else it["answer"]


def pred(cfg, bench, it):
    """Deterministic synthetic answer: right with a configuration-dependent rate, else a wrong string."""
    return gold(bench, it) if h(cfg, it["qid"]) % 100 < 40 + h(cfg) % 50 else "zzz"


def tiny_sets(d: Path, bench, n=30):
    d.mkdir(parents=True, exist_ok=True); rows = []
    for i in range(n):
        it = {"bench": bench, "qid": f"{bench[:2]}{i:03d}", "image_id": f"img{i % 7}", "image_file": f"img{i % 7}", "question": f"q{i}", "prompt": f"q{i}\nShort"}
        if bench.startswith("gqa"):
            it.update(answer="yes" if i % 3 else "no", structural="verify" if i % 2 else "query")
        elif bench.startswith("chartqa"):
            it.update(answer=str(10 + i), type="human_test" if i % 2 else "augmented_test")
        else:
            it.update(answers=[f"w{i}"] * 10)
        rows.append(it)
    (d / f"{bench}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def nvis(it):
    return 300 + h("n", it["qid"]) % 61


def write_cfg(od: Path, cfg, rows_by_shard):
    od.mkdir(parents=True, exist_ok=True)
    for shard, rows in rows_by_shard.items():
        (od / f"{RUN.cfg_file(cfg)}@{RUN.shard_tag(shard)}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


IDENT = {"method": "m", "attn": "sdpa", "transformers": "4.52.4", "processor": "Qwen2VLImageProcessorFast"}


def fixture(t: Path, benches=BENCHES, methods=tuple(EX.METHODS), bad_kept=True):
    """Sets, R7-40c predictions (its 50 configured rows + full_official), E1 predictions in two shards with meta, configuration lists, gates."""
    sets = t / "sets"; e1p, r40p = t / "e1" / "pred", t / "r40c" / "pred"; main = PL.unit_names(LR)
    PL.write_configs(LR, t / "r40c" / "configs", "core")
    EX.plan(Namespace(root=str(t / "e1"), lines=str(t / "e1" / "lines.txt"), anchor=True, partition="L40S"))
    (t / "lr.json").write_text(json.dumps(LR)); proto = t / "frozen.yaml"; proto.write_text("r7_e1_staged_freeze: >\n  frozen\n")
    comp = t / "e1" / "completeness_e1.json"; comp.write_text(json.dumps({"complete": True, "utc": "x", "problems": []}))
    items = {}
    for b in benches:
        its = tiny_sets(sets, b); items[b] = its
        for c in PL.read_configs(t / "r40c" / "configs" / f"configs_{b}.txt") + [RO.OFFICIAL]:
            f = 1.0 if c in ("full", RO.OFFICIAL) or c.startswith("dyn:") else PL.parse(c)[2]
            write_cfg(r40p / b, c, {"0/1": [{"qid": it["qid"], "image_id": it["image_id"], "pred": pred(c, b, it), "n_vis": nvis(it), "f": f,
                                            "kept": nvis(it) if f >= 1 else M4.budget(f, nvis(it)), "n_new": 2, "hit_max": False} for it in its]})
        for c in EX.config_list(tuple(EX.METHODS), anchor=True):
            m, f = EX.parse_cfg(c); rows = []
            for j, it in enumerate(its):
                k = EX.k_rule(m, f, nvis(it)) - (1 if (bad_kept and j == 0 and c == "ext:mmtok:0.5") else 0)
                rows.append({"qid": it["qid"], "image_id": it["image_id"], "pred": pred(c, b, it), "kept": k, "n_vis": nvis(it), "f": f,
                             "n_new": 2, "hit_max": j % 11 == 0, "k_rule": EX.k_rule(m, f, nvis(it)), "ms": 100.0 + j})
            if m in methods:
                write_cfg(e1p / b, c, {"0/2": rows[:12], "1/2": rows[12:]})
                for s in ("0/2", "1/2"):
                    mp = EX.meta_path(e1p / b, c, s); mp.parent.mkdir(parents=True, exist_ok=True)
                    mp.write_text(json.dumps({"identity": {**IDENT, "method": m, "fraction": f}, "sessions": [{"gpu": "NVIDIA L40S"}]}))
    args = ["--sets-dir", str(sets), "--pred-dir", str(e1p), "--configs-dir", str(t / "e1" / "configs"), "--r40c-pred-dir", str(r40p),
            "--r40c-configs-dir", str(t / "r40c" / "configs"), "--lr-choice", str(t / "lr.json"), "--completeness", str(comp), "--protocol", str(proto),
            "--reps", str(REPS), "--benches", *benches]
    return args, items, main


def run_ro(args):
    with contextlib.redirect_stdout(io.StringIO()):
        return RO.main(args)


def scores(bench, its, cfg):
    sc = SC.item_scores(bench, its, {it["qid"]: pred(cfg, bench, it) for it in its})
    return np.array([sc[it["qid"]] for it in its])


def walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield path + "/" + str(k), k, v
            yield from walk(v, path + "/" + str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk(v, path + f"[{i}]")


class Gates(unittest.TestCase):
    def test_freeze_completeness_overwrite(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); args, _, _ = fixture(t, benches=["gqa_dev"])
            proto, comp = Path(args[args.index("--protocol") + 1]), Path(args[args.index("--completeness") + 1])
            proto.write_text("r7_40c_staged_freeze: x\n")
            with self.assertRaises(SystemExit):
                run_ro(args + ["--out", str(t / "o1")])
            proto.write_text("r7_e1_staged_freeze: x\n"); comp.write_text(json.dumps({"complete": False, "problems": ["gqa_dev: 1 incomplete"]}))
            with self.assertRaises(SystemExit):
                run_ro(args + ["--out", str(t / "o1")])
            comp.unlink()
            with self.assertRaises(SystemExit):
                run_ro(args + ["--out", str(t / "o1")])
            comp.write_text(json.dumps({"complete": True})); self.assertEqual(run_ro(args + ["--out", str(t / "o1")]), 0)
            with self.assertRaises(SystemExit):                                         # never overwrites a readout
                run_ro(args + ["--out", str(t / "o1")])
            self.assertFalse((t / "o2").exists())


class Numbers(unittest.TestCase):
    def test_rows_diagnostic_table_and_no_contrast(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); args, items, main = fixture(t); self.assertEqual(run_ro(args + ["--out", str(t / "out")]), 0)
            res = json.loads((t / "out" / "r7_e1_results.json").read_text()); md = (t / "out" / "r7_e1_results.md").read_text()
            self.assertEqual(res["methods"], ["mmtok", "hiprune"]); self.assertEqual(res["gaps"], [])
            for b in BENCHES:
                its = items[b]; Rb = res["benches"][b]; boot = SC.Boot([it["image_id"] for it in its], REPS)
                self.assertEqual(sorted(Rb["e1_rows"]), sorted(EX.config_list(anchor=True)))
                off = scores(b, its, RO.OFFICIAL)
                for c, r in Rb["e1_rows"].items():
                    m, f = EX.parse_cfg(c); s = scores(b, its, c); n = np.array([nvis(it) for it in its], float)
                    k = np.array([EX.k_rule(m, f, x) for x in n]); k[0] -= (c == "ext:mmtok:0.5")
                    self.assertAlmostEqual(r["quality"], 100 * s.mean()); self.assertEqual(r["ci95"], [100 * v for v in SC.ci(boot.means(s))])
                    self.assertAlmostEqual(r["mean_kept_fraction"], float(np.mean(k / n))); self.assertAlmostEqual(r["mean_visual_tokens"], float(k.mean()))
                    self.assertEqual(r["answers_cut_at_max_new_tokens"], sum(1 for j in range(len(its)) if j % 11 == 0))
                    self.assertEqual(r["kept_equals_method_rule"], len(its) - (c == "ext:mmtok:0.5")); self.assertEqual(r["ms_per_item_median_descriptive"], float(np.median(100.0 + np.arange(len(its)))))
                    anc = scores(b, its, EX.cfg_name(m, 1.0))
                    if f < 1:
                        self.assertAlmostEqual(r["retention_vs_in_env_anchor"]["rel"], s.mean() / anc.mean()); self.assertAlmostEqual(r["retention_vs_in_env_anchor"]["abs_drop"], 100 * (anc.mean() - s.mean()))
                    else:
                        self.assertIsNone(r["retention_vs_in_env_anchor"])
                    self.assertAlmostEqual(r["retention_vs_r7_40c_full_official"]["rel"], s.mean() / off.mean())
                self.assertEqual(Rb["integrity"]["kept_not_equal_rule"], {"ext:mmtok:0.5": 1}); self.assertEqual(Rb["integrity"]["files_per_config_max"], 2)
                for m in EX.METHODS:                                                    # env diagnostic: paired, no p-value
                    d = Rb["env_diagnostic"][m]; anc = scores(b, its, EX.cfg_name(m, 1.0))
                    self.assertAlmostEqual(d["anchor_minus_official"]["delta"], 100 * float((anc - off).mean()))
                    self.assertEqual(d["anchor_minus_official"]["ci95"], [100 * v for v in SC.ci(boot.means(anc - off))]); self.assertNotIn("p", d["anchor_minus_official"])
                    same = np.mean([pred(EX.cfg_name(m, 1.0), b, it) == pred(RO.OFFICIAL, b, it) for it in its])
                    self.assertAlmostEqual(d["identical_answer_fraction"], float(same)); self.assertEqual(d["items_with_different_score"], int(np.sum(anc != off)))
                T = Rb["host_column_table"]; self.assertEqual(sorted(T), sorted(list(PL.FKEYS) + ["1"]))
                for fk in PL.FKEYS:
                    for fam, o in RO.FAMILIES.items():
                        cfgs = [f"static:grid:{fk}"] if o is None else [f"sel:{u}:{fk}" for u in main[o]]; x = np.mean([scores(b, its, c) for c in cfgs], 0)
                        cl = T[fk]["r7_40c"][fam]; self.assertEqual(cl["rows"], cfgs); self.assertAlmostEqual(cl["quality"], 100 * x.mean())
                        self.assertEqual(cl["ci95"], [100 * v for v in SC.ci(boot.means(x))])
                        if o is not None:
                            self.assertEqual(cl["per_seed"], [100 * scores(b, its, c).mean() for c in cfgs])
                    for m in EX.METHODS:
                        self.assertAlmostEqual(T[fk]["e1"][m]["quality"], Rb["e1_rows"][EX.cfg_name(m, float(fk))]["quality"])
                self.assertEqual(sorted(T["1"]["r7_40c"]), sorted(["full (batched, R7-40c)", "full_official (batch 1)"]))
                self.assertAlmostEqual(T["1"]["r7_40c"]["full_official (batch 1)"]["quality"], 100 * off.mean())
                if b == "gqa_dev":
                    self.assertEqual(sorted(Rb["subsets"]["e1"]), ["query", "verify"])
                if b == "chartqa_val":
                    self.assertEqual(sorted(Rb["subsets"]["e1"]), ["augmented", "human"])
                self.assertEqual(Rb["builds"]["ext:hiprune:0.25"]["shard_records"], 2); self.assertEqual(Rb["builds"]["ext:hiprune:0.25"]["gpus"], ["NVIDIA L40S"])
            for path, k, v in walk(res):                                               # no contrast / test with S1 / S2 / S3 / grid anywhere
                self.assertNotIn(k, ("p", "p_holm", "holm_family_size", "contrasts", "role_primary"), path)
                self.assertFalse(str(k).startswith(("S1-", "S2-", "S3-", "grid-", "mmtok-", "hiprune-")), path)
                if k == "delta":
                    self.assertIn("/env_diagnostic/", path)
            self.assertIn("| f | grid | S1 | S2 | S3 | mmtok (E1) | hiprune (E1) |", md); self.assertIn("no contrast or test with S1 / S2 / S3 / grid", md)
            self.assertIn("Env diagnostic (not an effect)", md)


class Guards(unittest.TestCase):
    def test_coverage_identity_and_configuration_lists(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); args, items, _ = fixture(t, benches=["gqa_dev"]); od = t / "e1" / "pred" / "gqa_dev"
            mp = EX.meta_path(od, "ext:hiprune:0.25", "1/2"); rec = json.loads(mp.read_text()); good = mp.read_text()
            rec["identity"]["attn"] = "eager"; mp.write_text(json.dumps(rec))
            with self.assertRaises(SystemExit):                                         # two builds in one configuration
                run_ro(args + ["--out", str(t / "o")])
            mp.write_text(good); f = od / "ext__mmtok__0.25@s1of2.jsonl"; lines = f.read_text().splitlines(); f.write_text("\n".join(lines[:-1]) + "\n")
            with self.assertRaises(SystemExit):                                         # Bench40's coverage guard on an E1 row
                run_ro(args + ["--out", str(t / "o")])
            f.write_text("\n".join(lines) + "\n"); cf = t / "e1" / "configs" / "configs_e1_gqa_dev.txt"; keep = cf.read_text()
            cf.write_text("\n".join(c for c in keep.splitlines() if c != "ext:hiprune:1") + "\n")
            with self.assertRaises(SystemExit):                                         # D-E1-1: every method with its anchor
                run_ro(args + ["--out", str(t / "o")])
            cf.write_text(keep); rf = t / "r40c" / "configs" / "configs_gqa_dev.txt"; keep2 = rf.read_text()
            rf.write_text("\n".join(c for c in keep2.splitlines() if c != "static:grid:0.5") + "\n")
            with self.assertRaises(SystemExit):                                         # the R7-40c rows of the table must be configured
                run_ro(args + ["--out", str(t / "o")])
            rf.write_text(keep2); (t / "r40c" / "pred" / "gqa_dev" / "full_official@s0of1.jsonl").unlink()
            with self.assertRaises(SystemExit):                                         # and covered (R7-40c's anchor)
                run_ro(args + ["--out", str(t / "o")])
            self.assertFalse((t / "o" / "r7_e1_results.json").exists())

    def test_disclosed_gap_one_method(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); args, _, _ = fixture(t, benches=["chartqa_val"], methods=("mmtok",))
            with self.assertRaises(SystemExit):                                         # HiPrune rows absent: both methods cannot be read out
                run_ro(args + ["--out", str(t / "o1")])
            self.assertEqual(run_ro(args + ["--methods", "mmtok", "--out", str(t / "o2")]), 0)
            res = json.loads((t / "o2" / "r7_e1_results.json").read_text())
            self.assertEqual((res["methods"], res["gaps"]), (["mmtok"], ["hiprune"]))
            self.assertEqual(sorted(res["benches"]["chartqa_val"]["e1_rows"]), sorted(EX.config_list(("mmtok",), anchor=True)))
            self.assertIn("Not read out (disclosed gap): hiprune", (t / "o2" / "r7_e1_results.md").read_text())

    def test_r7_40c_readout_cross_check(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); args, items, main = fixture(t, benches=["gqa_dev"]); its = items["gqa_dev"]
            cfgs = ["full"] + [f"static:grid:{f}" for f in PL.FKEYS] + [f"sel:{u}:{f}" for o in PL.OBJECTIVES for u in main[o] for f in PL.FKEYS]
            ref = {"benches": {"gqa_dev": {"fixed_fraction": {c: {"quality": 100 * scores("gqa_dev", its, c).mean()} for c in cfgs},
                                           "official_anchor": {"quality": 100 * scores("gqa_dev", its, RO.OFFICIAL).mean()}}}}
            rp = t / "r40c_results.json"; rp.write_text(json.dumps(ref))
            self.assertEqual(run_ro(args + ["--r7-40c-results", str(rp), "--out", str(t / "o1")]), 0)
            self.assertEqual(json.loads((t / "o1" / "r7_e1_results.json").read_text())["r7_40c_results_check"]["max_abs_diff_points"], 0.0)
            ref["benches"]["gqa_dev"]["fixed_fraction"]["static:grid:0.25"]["quality"] += 0.5; rp.write_text(json.dumps(ref))
            with self.assertRaises(SystemExit):
                run_ro(args + ["--r7-40c-results", str(rp), "--out", str(t / "o2")])


class Wrapper(unittest.TestCase):
    def test_wrapper_and_frozen_e1_hash(self):
        w = R / "slurm" / "r7_readout_e1_cpu.sbatch"; s = w.read_text()
        self.assertFalse(w.name.startswith("r7_e1_"))                                   # outside the frozen wrappers' glob slurm/r7_e1_*
        frozen = sorted((R / "slurm").glob("r7_e1_*")); self.assertEqual([p.name for p in frozen], ["r7_e1_cpu.sbatch", "r7_e1_gpu.sbatch"])
        hsh = hashlib.sha256(b"".join(p.read_bytes() for p in [R / "scripts/r7_e1_external.py", R / "tests/test_r7_e1_external.py", *frozen])).hexdigest()
        self.assertEqual(hsh[:16], "519635f47ebbe9c0")                                  # the frozen E1 code is untouched
        self.assertIn("#SBATCH --partition=CPU", s); self.assertIn("envs/sparsevlm", s); self.assertIn("-I scripts/r7_e1_readout.py", s)
        self.assertIn('"$PYTHON" -I tests/test_r7_e1_readout.py -v', s); self.assertIn('complete --pred-dir "$E1/pred" --out "$E1" --anchor --methods "$@"', s)
        self.assertNotIn("--export", s.replace("(no --export)", ""))
        self.assertEqual(subprocess.run(["bash", "-n", str(w)]).returncode, 0)


if __name__ == "__main__":
    unittest.main()
