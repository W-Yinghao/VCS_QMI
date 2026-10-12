"""r7 R7-30 D1-FULL GPU smoke — code checks only, one LLaVA host load (SparseVLM env, L40S: most frozen FIT shards and the frozen D1-20 cache were
made on L40S, so bit-level comparisons are possible).  Never reads DEV; reads frozen FIT / CAL records and the R7-20 fixtures only.
  probe   : the frozen D1-10 probe (sentinels S1-S6 on the R7-20 GQA fixture) -> <out>/probe/probe_report.json = launch gate of the D1-FULL bundles
  replay  : first chunks of frozen FIT shards recomputed with the frozen teacher (reuse check) -> <out>/reuse_replay.json
  mini    : a mini D1-FULL root (<out>/mini: 4 frozen FIT + 4 additional FIT images, 4 CAL images): frozen FIT / CAL records copied (filtered to those
            images), the strict bundle on the 4 additional images (frozen settings, probe gate); a plan naming a missing image stops in preflight
  cache   : (a) the first 16 frozen FIT images encoded with the frozen basis vs the frozen D1-20 cache rows (bit check); (b) the D1-FULL cache code on
            the mini root with the FIT_FULL basis (streamed file == the frozen in-memory encode_role result) and with the frozen basis (copied pca.pt)
  eval    : the frozen D1-30 runner with the D1-FULL selector configs (87 rows) on 16 items of each R7-20 fixture set (gqa_fixture, textvqa_fixture,
            chartqa_proxy), seeded freshly initialised units under the D1-FULL unit tags (taus calibrated on the fixture: smoke only), the mini root's
            FIT_FULL-basis PCA, a separate output root; the frozen D1-30 timing of two D1-FULL dynamic rows at B = 1 / 4 / 16 / 32 (small pools, any
            GPU, labelled smoke; per-row cost extrapolated to the frozen pools); the D1-30 DEV-risk code path on the D1-10 probe records (fixture items)
  selector: after the host is released, the selector wrapper + the frozen trainer in the MAIN env (subprocess, as production) on the mini root (5 unit
            kinds, tiny steps) and the finish path (CAL taus dropped and recomputed).  The FIT_FULL-size memory smoke is separate (slurm/r7_d1full_sel.sbatch memsmoke).
    python scripts/r7_d1full_smoke.py --out DF/smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r7_d1_30_data as DD  # noqa: E402
import r7_d1_30_devrisk as DR  # noqa: E402
import r7_d1_30_run as RUN  # noqa: E402
import r7_d1_cache as CA  # noqa: E402
import r7_d1_teacher as TE  # noqa: E402
import r7_d1full_cache as FC  # noqa: E402
import r7_d1full_plan as FP  # noqa: E402
import r7_d1full_teacher as FT  # noqa: E402
import r7_vl_common as V  # noqa: E402

OUT = Path("/home/infres/yinwang/CS_QMI/outputs")
D1, DF = OUT / "R7_D1", OUT / "R7_D1_FULL"
FX = Path("/projects/EEG-foundation-model/yinghao/data/r7_fixtures")
R720 = OUT / "R7_fixtures" / "d1" / "pred_full.jsonl"
MAIN_PY = "/home/infres/yinwang/CS_QMI/env/bin/python"                                # main env (torch 2.9), as the production selector units
BENCH_SRC = {"gqa_fixture": "gqa_dev", "textvqa_fixture": "textvqa_val", "chartqa_proxy": "chartqa_val"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def role_images(rows, role):
    return list(dict.fromkeys(r["image_id"] for r in rows if r["role"] == role))


def build_mini(root: Path, df_data=DF / "data", frozen_teacher=D1 / "teacher", n_old=4, n_new=4, n_cal=4):
    """Mini D1-FULL root: manifest rows of 4 frozen FIT + 4 additional FIT + 4 CAL images; shards CAL-000 / FIT-000 (frozen records, filtered) and
    FIT-040 (to compute); bundle f000 = [FIT-040]."""
    rows = [json.loads(x) for x in (df_data / "d1_manifest.jsonl").read_text().splitlines() if x.strip()]
    fit, cal = role_images(rows, "FIT"), role_images(rows, "CAL"); n_frozen_fit = 10000
    pick = {"FIT-000": fit[:n_old], "FIT-040": fit[n_frozen_fit:n_frozen_fit + n_new], "CAL-000": cal[:n_cal]}
    keep = {i for v in pick.values() for i in v}; mrows = [r for r in rows if r["image_id"] in keep]
    (root / "data").mkdir(parents=True, exist_ok=True); (root / "teacher").mkdir(parents=True, exist_ok=True)
    (root / "data" / "d1_manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in mrows))
    f = [r for r in mrows if r["role"] == "FIT"]; pi = sum(r["C"] == 1 for r in f) / len(f)
    meta = {"smoke": True, "frozen": True, "manifest_sha256": sha(root / "data" / "d1_manifest.jsonl"),
            "class_weights_from_FIT": {"pi_yes": pi, "w_yes": 0.5 / pi, "w_no": 0.5 / (1 - pi)}}
    (root / "data" / "d1_manifest_meta.json").write_text(json.dumps(meta, indent=1))
    qps = {s: sum(r["image_id"] in set(v) for r in mrows) for s, v in pick.items()}
    plan = {"images_per_shard": 250, "shards_per_bundle": 1, "shards": {s: pick[s] for s in ("CAL-000", "FIT-000", "FIT-040")}, "bundles": {"f000": ["FIT-040"]},
            "questions_per_shard": qps, "d1full": {"reused_shards": ["CAL-000", "FIT-000"], "computed_shards": ["FIT-040"], "smoke": True}}
    (root / "data" / "d1_shards.json").write_text(json.dumps(plan, indent=1))
    for s in ("CAL-000", "FIT-000"):                                                     # frozen records of the picked images (FIT / CAL only)
        src = [r for r in V.read_jsonl(frozen_teacher / f"{s}.jsonl") if r["image_id"] in set(pick[s])]; done = json.loads((frozen_teacher / f"{s}.done.json").read_text())
        p = root / "teacher" / f"{s}.jsonl"; p.write_text("".join(V.jdump(r) + "\n" for r in src))
        (root / "teacher" / f"{s}.done.json").write_text(json.dumps({"shard": s, "expected": len(src), "present": len(src), "duplicates": 0, "new_this_session": len(src),
                                                                    "skipped_images": [], "seconds": 0.0, "gpu": done.get("gpu"), "jsonl_sha256": sha(p),
                                                                    "smoke_filtered_from": str(frozen_teacher / f"{s}.jsonl")}, indent=1))
    return plan


def smoke_selectors(torch, out: Path, lr=3e-4):
    """Freshly initialised frozen-architecture units (seeded) saved in the frozen checkpoint format under the D1-FULL unit tags (A1 / A2 with
    the ablation's cfg flag) — a wiring check of the D1-30 runner with a D1-FULL selectors dir; the D1-20 smoke checkpoints were removed by the
    D4 cleanup.  Returns the smoke lr_choice."""
    from dataclasses import asdict
    import r7_d1_selectors as SEL
    out.mkdir(parents=True, exist_ok=True); choice = {o: {"chosen_lr": lr, "smoke": True} for o in RUN.OBJECTIVES}; main, abl = RUN.unit_names(choice)
    todo = [(u, o, {}) for o in RUN.OBJECTIVES for u in main[o]] + [(u, "vcs", {"no_question": True}) for u in abl["A1"]] + [(u, "direct", {"target": "d"}) for u in abl["A2"]]
    for j, (u, o, flag) in enumerate(todo):
        torch.manual_seed(1000 + j); cfg = SEL.Cfg(lr=lr, **flag); m = SEL.Selector(cfg, o)
        torch.save({"state_dict": m.state_dict(), "cfg": asdict(cfg), "objective": o}, out / f"{u}.pt")
    (out / "lr_choice.json").write_text(json.dumps(choice, indent=1))
    return choice


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(DF / "smoke")); ap.add_argument("--replay-shards", nargs="+", default=["FIT-000", "FIT-001", "FIT-020"])
    ap.add_argument("--n-eval", type=int, default=16); a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rep = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "checks": {}}; t0 = time.time()
    host = TE.LlavaPrefill(out / "host"); torch = host.torch; rep["gpu"] = torch.cuda.get_device_name(0); rep["load_seconds"] = host.load_seconds
    src = FT.StrictImageSource(); C = rep["checks"]
    # probe (frozen sentinels) -> launch gate of the D1-FULL bundles
    rc = TE.cmd_probe(Namespace(out=str(out / "probe"), fixture=str(FX / "gqa.jsonl"), r720_preds=str(R720), check_generation=True), host)
    pr = json.loads((out / "probe" / "probe_report.json").read_text()); C["probe_pass"] = pr["pass"] and rc == 0
    C["probe_sentinels"] = pr["sentinels"]
    # replay of frozen FIT chunks (reuse check)
    rp = FT.replay(host, a.replay_shards, src); rp["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / "reuse_replay.json").write_text(V.jdump(rp, indent=1)); C["replay"] = {k: rp[k] for k in ("pass", "gpu", "same_gpu_shards", "cross_gpu_shards",
                                                                                                      "bit_identical_same_gpu", "max_abs_dP_same_gpu", "max_abs_dP_cross_gpu")}
    C["replay_shards"] = rp["shards"]
    # mini root: strict bundle (frozen settings, probe gate) and the missing-image stop
    mini = out / "mini"
    if mini.exists():
        shutil.rmtree(mini)
    plan = build_mini(mini)
    bad = mini / "data_missing"; shutil.copytree(mini / "data", bad); bp = json.loads((bad / "d1_shards.json").read_text())
    bp["shards"]["FIT-040"] = bp["shards"]["FIT-040"] + ["r7d1full_smoke_nonexistent_image"]; (bad / "d1_shards.json").write_text(json.dumps(bp))
    try:
        FT.cmd_bundle(Namespace(bundle="f000", manifest_dir=str(bad), out=str(mini / "teacher_missing"), probe_report=str(out / "probe" / "probe_report.json")),
                      host_factory=lambda: host, image_source=src); C["missing_image_stops"] = False
    except SystemExit as e:
        C["missing_image_stops"] = "preflight" in str(e)
    try:
        src.open("r7d1full_smoke_nonexistent_image"); C["strict_source_raises"] = False
    except FT.MissingImage:
        C["strict_source_raises"] = True
    t1 = time.time()
    FT.cmd_bundle(Namespace(bundle="f000", manifest_dir=str(mini / "data"), out=str(mini / "teacher"), probe_report=str(out / "probe" / "probe_report.json")),
                  host_factory=lambda: host, image_source=src)
    ok = json.loads((mini / "teacher" / "bundle_f000.ok.json").read_text())
    C["bundle"] = {"questions": ok["questions"], "expected": plan["questions_per_shard"]["FIT-040"], "seconds": time.time() - t1,
                   "s_per_question": (time.time() - t1) / max(ok["questions"], 1), "marker": True}
    # question token lengths over every FIT_FULL + CAL question (the selector memory smoke assumes <= 40 tokens; frozen FIT + CAL max 33)
    t5 = time.time(); allrows = [json.loads(x) for x in (DF / "data" / "d1_manifest.jsonl").read_text().splitlines() if x.strip()]
    lens = [len(host.tok(r["question"], add_special_tokens=False).input_ids) for r in allrows if r["role"] in ("FIT", "CAL")]
    C["question_tokens_fit_full_cal"] = {"n": len(lens), "max": max(lens), "p99": float(np.percentile(lens, 99)), "mean": float(np.mean(lens)),
                                         "n_over_40": int(sum(x > 40 for x in lens)), "seconds": time.time() - t5}
    # cache (a): frozen basis on the first 16 frozen FIT images vs the frozen D1-20 cache rows
    ids16 = json.loads((D1 / "cache" / "images_FIT.json").read_text())[:16]; pz = torch.load(D1 / "cache" / "pca.pt", map_location="cpu")
    mine = FC.encode_chunk(host, pz["mean"].cuda(), pz["basis"].cuda(), [src.open(i) for i in ids16]); ref = np.load(D1 / "cache" / "feats_FIT.npy", mmap_mode="r")[:16]
    C["cache_frozen_basis_vs_frozen_cache"] = {"bit_identical": bool(np.array_equal(mine, ref)), "max_abs_diff": float(np.abs(mine.astype(np.float32) - ref.astype(np.float32)).max()),
                                               "frozen_cache_gpu": json.loads((D1 / "cache" / "cache_d1_20.json").read_text())["env"]["gpu"]}
    # cache (b): D1-FULL cache code on the mini root, FIT_FULL basis and frozen basis
    for basis, d in (("fitfull", mini / "cache"), ("frozen", mini / "cache_frozen_basis")):
        t2 = time.time(); rec = FC.run(mini / "data", d, basis, host=host, src=src); e = {"seconds": time.time() - t2, "shapes": {r: rec[f"feats_{r}_shape"] for r in ("FIT", "CAL")},
                                                                                          "max_question_tokens": rec["max_question_tokens"]}
        pca = torch.load(d / "pca.pt", map_location="cpu"); rows = [json.loads(x) for x in (mini / "data" / "d1_manifest.jsonl").read_text().splitlines() if x.strip()]
        for role in ("FIT", "CAL"):
            ids = role_images(rows, role); ref = CA.encode_role(host, pca, [src.open(i) for i in ids]); got = np.load(d / f"feats_{role}.npy")
            e[f"{role}_stream_equals_frozen_encode_role"] = bool(np.array_equal(ref, got)); e[f"{role}_images_json_ok"] = json.loads((d / f"images_{role}.json").read_text()) == ids
        q = [json.loads(x) for x in (d / "question_ids.jsonl").read_text().splitlines() if x.strip()]
        e["question_ids_ok"] = [x["qid"] for x in q] == [r["qid"] for r in rows if r["role"] in ("FIT", "CAL")] and all(
            x["ids"] == host.tok(r["question"], add_special_tokens=False).input_ids for x, r in zip(q, [r for r in rows if r["role"] in ("FIT", "CAL")]))
        e["embed_table_shape"] = list(torch.load(d / "embed_table.pt").shape)
        if basis == "frozen":
            e["pca_is_frozen_copy"] = sha(d / "pca.pt") == sha(D1 / "cache" / "pca.pt")
        else:
            e["pca_fit_images"] = pca["fit_images"]; e["pca_tokens"] = pca["tokens"]
        C[f"cache_{basis}"] = e
    # eval: frozen D1-30 runner with D1-FULL selector configs, separate output root
    ev = out / "eval"
    if ev.exists():
        shutil.rmtree(ev)
    rep["sets"] = DD.fixture_sets(ev / "sets"); choice = smoke_selectors(torch, ev / "selectors")
    units = FP.units(choice); pca_p = mini / "cache" / "pca.pt"; bank = RUN.SelectorBank(ev / "selectors", pca_p, units)
    gq = RUN.load_items(ev / "sets", "gqa_fixture")[: a.n_eval]
    feats = RUN.encode_unique(host, gq, lambda fp: __import__("PIL.Image", fromlist=["Image"]).open(fp).convert("RGB"))
    st = bank.prepare(feats, RUN.question_embeddings(host, [it["question"] for it in gq]))
    for u in units:
        if not u.endswith(("_noq", "_dtarget")):
            t = RUN.calibrate_tau(bank.risks(u, st))
            (ev / "selectors" / f"{u}.json").write_text(json.dumps({"dynamic": {str(f): {"tau": v, "smoke_fixture_calibration": True} for f, v in t.items()}}, indent=1))
    C["eval_configs"] = FP.write_eval_configs(choice, ev / "configs"); C["eval"] = {}
    for b, sb in BENCH_SRC.items():
        t3 = time.time()
        RUN.run(Namespace(sets_dir=str(ev / "sets"), bench=b, configs_file=str(ev / "configs" / f"configs_{sb}.txt"), shard="0/1", out=str(ev / "pred"), chunk=32,
                          batch=32, limit=a.n_eval, selectors_dir=str(ev / "selectors"), pca=str(pca_p)), host=host)
        cfgs = [c for c in (ev / "configs" / f"configs_{sb}.txt").read_text().splitlines() if c.strip()]
        n = {c: len(RUN.answered(ev / "pred" / b, c)) for c in cfgs}
        C["eval"][b] = {"configs": len(cfgs), "all_configs_complete": all(v == a.n_eval for v in n.values()), "marker": (ev / "pred" / b / "done_shard0of1.json").exists(),
                        "no_shared_rows": not any(c.startswith(("full", "static:")) for c in cfgs), "seconds": time.time() - t3}
    # timing of the D1-FULL dynamic rows: the frozen D1-30 timing procedure with the D1-FULL selectors / PCA (any GPU here, labelled smoke; small
    # pools), timed per batch size to extrapolate the reference-node cost of the frozen pools (512 at B = 4, 256 at B = 16 / 32, 64 at B = 1)
    import r7_d1_30_timing as TM
    dyn = [c for c in (ev / "configs" / "timing_gqa_dev.txt").read_text().splitlines() if c.strip()]; pick = [dyn[0], dyn[4]]
    (ev / "configs" / "timing_smoke.txt").write_text("\n".join(pick) + "\n"); per_b = {}; n_rows = 0
    for B, n in ((1, 16), (4, 64), (16, 64), (32, 64)):
        t4 = time.time()
        TM.main(["--sets-dir", str(ev / "sets"), "--bench", "gqa_fixture", "--configs-file", str(ev / "configs" / "timing_smoke.txt"), "--out", str(ev / "timing" / f"B{B}"),
                 "--n", str(n), "--n-big", str(n), "--n1", str(n), "--batches", str(B), "--allow-any-gpu", "--selectors-dir", str(ev / "selectors"), "--pca", str(pca_p)], host=host)
        rows = json.loads((ev / "timing" / f"B{B}" / "timing_gqa_fixture.json").read_text())["rows"]; n_rows += len(rows)
        per_b[B] = {"pool": n, "seconds_per_row": (time.time() - t4) / len(pick), "bucketed": all(r.get("bucketed") for r in rows),
                    "per_item_ms_median": [r["per_item_ms_median"] for r in rows]}
    frozen_pool = {1: 64, 4: 512, 16: 256, 32: 256}
    est = sum(v["seconds_per_row"] * frozen_pool[B] / v["pool"] for B, v in per_b.items())
    C["timing_path"] = {"configs": pick, "rows": n_rows, "per_batch_size": per_b, "gpu": rep["gpu"],
                        "estimate_seconds_per_dynamic_row_at_frozen_pools": est, "estimate_hours_per_benchmark_27_rows": 27 * est / 3600}
    DR.main(["--out", str(ev / "devrisk"), "--records", str(D1 / "smoke" / "probe" / "probe_records.jsonl"), "--images-from-fixture", str(FX / "gqa.jsonl"),
             "--selectors-dir", str(ev / "selectors"), "--pca", str(pca_p), "--reps", "200", "--lr-choice", str(ev / "selectors" / "lr_choice.json")], host=host)
    dv = json.loads((ev / "devrisk" / "d1_30_devrisk_smoke.json").read_text()); C["devrisk_path"] = {"units": len(dv["units"]), "records": dv["coverage"]}
    # selector path in the main env (as production): the wrapper + frozen trainer on the mini root (tiny steps; reused FIT-000 / CAL-000 records + the
    # FIT-040 records computed above + the FIT_FULL-basis mini cache), then the finish path; the LLaVA host is released first
    del host, bank, feats, st; import gc; gc.collect(); torch.cuda.empty_cache(); env = dict(os.environ, PYTHONNOUSERSITE="1"); runs = {}
    for u in (["vcs", "0.0003", "0"], ["logistic", "0.0001", "0"], ["direct", "0.0001", "0"], ["vcs", "0.0003", "0", "--no-question"], ["direct", "0.0001", "0", "--target", "d"]):
        t6 = time.time(); rc = subprocess.call([MAIN_PY, str(HERE / "r7_d1full_train.py"), "train", *u, "--root", str(mini), "--steps-a", "20", "--steps-b", "10",
                                                 "--eval-every", "10"], env=env); runs[" ".join(u)] = {"rc": rc, "seconds": time.time() - t6}
    rc = subprocess.call([MAIN_PY, str(HERE / "r7_d1full_train.py"), "smokefinish", "vcs", "0.0003", "0", "--root", str(mini)], env=env)
    fin = json.loads((mini / "smoke_finish.json").read_text()) if (mini / "smoke_finish.json").exists() else {}
    marks = sorted(x.name for x in (mini / "selectors").glob("*.complete.json"))
    logs = {x.stem: json.loads(x.read_text()) for x in (mini / "selectors").glob("*.json") if not x.name.endswith(".complete.json")}
    C["selector_path_main_env"] = {"runs": runs, "markers": marks, "finish": {k: fin.get(k) for k in ("pass", "max_abs_tau_diff", "finished_by", "coverage")},
                                   "coverage_FIT": {t: l.get("coverage", {}).get("FIT") for t, l in logs.items()},
                                   "gpu_in_logs": sorted({(l.get("env") or {}).get("gpu") for l in logs.values()} - {None}), "torch": sorted({(l.get("env") or {}).get("torch") for l in logs.values()} - {None})}
    rep["pass"] = bool(C["probe_pass"] and C["missing_image_stops"] and C["strict_source_raises"] and C["bundle"]["questions"] == C["bundle"]["expected"]
                       and all(v["rc"] == 0 for v in runs.values()) and len(marks) == 5 and fin.get("pass") is True
                       and all(C[f"cache_{b}"][k] for b in ("fitfull", "frozen") for k in ("FIT_stream_equals_frozen_encode_role", "CAL_stream_equals_frozen_encode_role",
                                                                                              "FIT_images_json_ok", "CAL_images_json_ok", "question_ids_ok"))
                       and C["cache_frozen"]["pca_is_frozen_copy"] and all(v["all_configs_complete"] and v["marker"] for v in C["eval"].values())
                       and C["timing_path"]["rows"] == 8 and all(v["bucketed"] for v in C["timing_path"]["per_batch_size"].values()))
    rep["reuse_replay_pass"] = rp["pass"]; rep["seconds"] = time.time() - t0; rep["env"] = V.env_info()
    (out / "smoke_report_llava.json").write_text(V.jdump(rep, indent=1)); print(V.jdump(rep, indent=1))
    return 0 if rep["pass"] else 6


if __name__ == "__main__":
    sys.exit(main())
