"""r7 R7-50 smoke on NON-test data only: no file under any closed/ directory is opened, listed or hashed.
  prep       CPU  smoke sources in the official formats and development-mode builds through r7_50_test.build:
                  integrity root: the D1-30 GQA DEV set re-expressed as a GQA balanced-format questions file (answers withheld: this root is never
                    scored) and the full ChartQA val parquet; rows full + static:grid:144 + sparsevlm:192:official only
                  scored root: a GQA balanced-VAL subset (24 images in sha256 order, <= 4 questions each) and a ChartQA TRAIN subset parquet (48 human +
                    48 augmented rows, the official schema); neither is a D1-30 benchmark, so scoring them reads no D1-30 result
                  smoke selectors: the D1-20 architecture (cfg from the chosen-rate unit logs) with RANDOM seeded weights, tagged with the D1-30 smoke's
                    lr_choice (3e-4 for every objective) so the D1-30 smoke timing files provide profiling latencies — numbers are not results
                  selector record: sha256 of the D1-20 chosen-rate units that exist now, the PCA and lr_choice.json (protocol draft)
  gpu        GPU  one LLaVA host: smoke taus (calibrated on the scored GQA subset, smoke only), configs + lines for the scored root, the quality
                  runner on both roots (integrity: the first 96 items in D1-30 shard order = the early D1-30 rows' first three chunks); phase F
                  leg: a second random-init selector set ('D1-FULL-like', other tags) generating the 45 phase-F rows on the scored root
  allow      CPU  the scorer's gate refusals (no ALLOW_SCORING; a wrong sha256; a changed prediction file), then the smoke's ALLOW_SCORING
  integrity  CPU  integrity root vs the early D1-30 rows on the same items: inputs (question, prompt, image bytes) and prediction strings / kept counts;
                  no label opened, nothing scored, no prediction printed
  report     CPU  smoke_summary.json: build / configs / lines / manifest / scorer code checks, integrity numbers, measured costs
    python -I scripts/r7_50_smoke.py {prep|gpu|allow|integrity|report} --smoke /home/infres/yinwang/CS_QMI/outputs/R7_50_smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7_50_test as T  # noqa: E402
import r7_d1_30_run as RUN  # noqa: E402
import r7_vl_common as V  # noqa: E402

DATA, D1, D30 = T.DATA, T.D1, T.D30
D30_SMOKE = D1 / "d1_30_smoke"
SMOKE_TIMING = D30_SMOKE / "timing"
SMOKE_TMAP = {"gqa_testdev": "gqa_fixture", "chartqa_test": "chartqa_proxy"}
LR_F_SMOKE = {o: {"chosen_lr": 1e-4, "smoke": "D1-FULL-like tags for the phase-F leg"} for o in ("vcs", "logistic", "direct")}
INTEGRITY_ROWS = ("full", "static:grid:144")
INTEGRITY_SV = 192
N_INTEGRITY = 96
TOL = {"pipeline": 0.98, "sparsevlm": 0.96}             # pre-declared: frozen QC S5 (>= 98 % for batched generation); SparseVLM run-to-run 99.1 %


def h(tag, key):
    return hashlib.sha256(f"r7-50-smoke/{tag}/{key}".encode()).hexdigest()


def roots(smoke, integrity_root="integrity"):
    S = Path(smoke)
    return S, T.Layout(S / integrity_root), T.Layout(S / "scored"), S / "selectors", S / "sources"


def selectors_f(smoke):
    return Path(smoke) / "selectors_F"


# ------------------------------------------------------------------------------------------------------------------------------ prep
def gqa_integrity_source(dst: Path):
    """The D1-30 GQA DEV set in the GQA balanced format; answers withheld (this root is compared, never scored)."""
    dev = T.read_jsonl_strict(D30 / "sets" / "gqa_dev.jsonl")
    dst.write_text(json.dumps({it["qid"]: {"imageId": it["image_id"], "question": it["question"], "answer": "withheld", "types": {}} for it in dev}))
    return len(dev)


def gqa_val_subset(dst: Path, n_images=24, per_image=4):
    vb = json.loads((DATA / "gqa" / "val_balanced_questions.json").read_text()); by = {}
    for q, r in vb.items():
        by.setdefault(r["imageId"], []).append(q)
    sub = {}
    for img in sorted(by, key=lambda i: h("img", i))[:n_images]:
        for q in sorted(by[img], key=lambda q: h("q", q))[:per_image]:
            sub[q] = vb[q]
    dst.write_text(json.dumps(sub)); return {"questions": len(sub), "images": n_images, "source": "gqa/val_balanced_questions.json"}


def chartqa_train_subset(dst: Path, per_split=48):
    import pyarrow as pa
    import pyarrow.parquet as pq
    src = DATA / "chartqa" / "data" / "train-00000-of-00003.parquet"; pf = pq.ParquetFile(src); take = {"human": [], "augmented": []}
    for b in pf.iter_batches(batch_size=256):
        for r in b.to_pylist():
            k = [x for x in take if x in str(r["type"]).lower()]
            if len(k) == 1 and len(take[k[0]]) < per_split:
                take[k[0]].append(r)
        if all(len(v) == per_split for v in take.values()):
            break
    pq.write_table(pa.Table.from_pylist(take["human"] + take["augmented"], schema=pf.schema_arrow), dst)
    return {"rows": sum(len(v) for v in take.values()), "source": str(src), "schema": {f.name: str(f.type) for f in pf.schema_arrow}}


def smoke_selectors(sel: Path, choice=None, seed0=50_000):
    """Random-init selectors with the D1-20 architecture (cfg from the chosen-rate s0 unit logs) under the D1-30 smoke's unit tags (or the given
    lr_choice's tags, for the phase-F leg)."""
    import torch
    import r7_d1_selectors as SEL
    sel.mkdir(parents=True, exist_ok=True)
    if choice is None:
        shutil.copy(D30_SMOKE / "selectors" / "lr_choice.json", sel / "lr_choice.json")
    else:
        (sel / "lr_choice.json").write_text(json.dumps(choice, indent=1))
    main, _ = RUN.unit_names(json.loads((sel / "lr_choice.json").read_text())); real, _ = RUN.unit_names(json.loads(T.LR_CHOICE.read_text())); out = {}
    for oi, o in enumerate(RUN.OBJECTIVES):
        cfg = json.loads((T.SELECTORS / f"{real[o][0]}.json").read_text())["cfg"]
        for s, u in enumerate(main[o]):
            torch.manual_seed(seed0 + 10 * oi + s)
            m = SEL.Selector(SEL.Cfg(**{k: (tuple(v) if k == "budgets" else v) for k, v in cfg.items()}), o)
            torch.save({"cfg": cfg, "objective": o, "state_dict": m.state_dict(), "smoke": "R7-50 smoke: random init, not a trained selector"}, sel / f"{u}.pt")
            out[u] = {"objective": o, "cfg_from": f"{real[o][0]}.json", "seed": seed0 + 10 * oi + s}
    return out


def selector_record():
    """sha256 of the D1-20 chosen-rate units present now (protocol draft); the rest are 'after D1-20'."""
    choice = json.loads(T.LR_CHOICE.read_text()); main, abl = RUN.unit_names(choice); rec = {"lr_choice": {"file": str(T.LR_CHOICE), "sha256": T.sha(T.LR_CHOICE)},
                                                                                             "pca": {"file": str(T.PCA), "sha256": T.sha(T.PCA)}, "units": {}}
    for o in RUN.OBJECTIVES:
        for s, u in enumerate(main[o]):
            pt, js = T.SELECTORS / f"{u}.pt", T.SELECTORS / f"{u}.json"
            if pt.is_file() and js.is_file() and '"dynamic"' in js.read_text():
                lg = json.loads(js.read_text())
                rec["units"][u] = {"pt_sha256": T.sha(pt), "json_sha256": T.sha(js), "selected_step": lg.get("selected_step"),
                                   "dynamic_tau": {f: v.get("tau") for f, v in lg["dynamic"].items()}}
            else:
                rec["units"][u] = "after D1-20 (not yet complete)"
    rec["ablation_units_not_used_by_r7_50"] = abl
    return rec


def prep_integrity(smoke, integrity_root, chartqa_qid_prefix):
    """A further integrity root (e.g. with D1-30's chartqa_val ids, so that qid-seeded grid masks coincide); sources from 'prep'."""
    S, I, _, _, src = roots(smoke, integrity_root); rep = {"started_utc": T.utc(), "chartqa_qid_prefix": chartqa_qid_prefix}; t1 = time.time()
    rep["build_integrity"] = T.build(I.root, src / "gqa_dev_d1_30_integrity_questions.json", DATA / "chartqa" / "data" / "val-00000-of-00001.parquet",
                                     mix665k=T.MIX665K, chartqa_qid_prefix=chartqa_qid_prefix)
    rep["build_integrity"]["wall_s"] = time.time() - t1
    for b in T.BENCHES:
        T.write_atomic(I.configs_file(b), "\n".join(INTEGRITY_ROWS) + "\n"); T.write_atomic(I.sv_rows_file(b), "\n".join(T.sparsevlm_rows(b, [INTEGRITY_SV])) + "\n")
    T.write_atomic(S / f"smoke_prep_{integrity_root}.json", json.dumps(rep, indent=1)); print(json.dumps({"items": {b: v["items"] for b, v in rep["build_integrity"]["benches"].items()}}))
    return 0


def prep(smoke):
    S, I, SC_, sel, src = roots(smoke); src.mkdir(parents=True, exist_ok=True); rep = {"started_utc": T.utc()}; t0 = time.time()
    rep["sources"] = {"gqa_integrity": {"questions": gqa_integrity_source(src / "gqa_dev_d1_30_integrity_questions.json"), "answers": "withheld"},
                      "gqa_scored": gqa_val_subset(src / "gqa_val_balanced_smoke_questions.json"),
                      "chartqa_scored": chartqa_train_subset(src / "chartqa_train_smoke.parquet"),
                      "chartqa_integrity": {"source": str(DATA / "chartqa" / "data" / "val-00000-of-00001.parquet")}}
    t1 = time.time()
    rep["build_integrity"] = T.build(I.root, src / "gqa_dev_d1_30_integrity_questions.json", DATA / "chartqa" / "data" / "val-00000-of-00001.parquet",
                                     mix665k=T.MIX665K, chartqa_qid_prefix="chartqa_val")   # D1-30's ids: qid-seeded grid masks coincide (like-for-like)
    rep["build_integrity"]["wall_s"] = time.time() - t1; t1 = time.time()
    rep["build_scored"] = T.build(SC_.root, src / "gqa_val_balanced_smoke_questions.json", src / "chartqa_train_smoke.parquet", mix665k=T.MIX665K)
    rep["build_scored"]["wall_s"] = time.time() - t1
    for b in T.BENCHES:                                                                 # integrity rows only
        T.write_atomic(I.configs_file(b), "\n".join(INTEGRITY_ROWS) + "\n"); T.write_atomic(I.sv_rows_file(b), "\n".join(T.sparsevlm_rows(b, [INTEGRITY_SV])) + "\n")
    rep["smoke_selectors"] = smoke_selectors(sel); rep["smoke_selectors_F"] = smoke_selectors(selectors_f(smoke), LR_F_SMOKE, 60_000)
    rep["selector_record"] = selector_record()
    fake = src / "closed"                                                            # synthetic closed/ paths (never the real test files)
    for name, fn in (("refuse_closed_without_freeze", lambda: T.build(S / "refusal_check", fake / "testdev_balanced_questions.json", fake / "test.parquet", None)),
                     ("refuse_test_name_outside_closed", lambda: T.build(S / "refusal_check", src / "x_test_questions.json", src / "chartqa_train_smoke.parquet"))):
        try:
            fn(); rep[name] = "NOT REFUSED"
        except SystemExit as e:
            rep[name] = f"refused: {str(e)[:160]}"
    rep["seconds"] = time.time() - t0; T.write_atomic(S / "smoke_prep.json", json.dumps(rep, indent=1)); print(json.dumps({k: rep[k] for k in ("sources", "seconds")}, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------------------------------------ gpu
def gpu_integrity(smoke, integrity_root):
    """The integrity rows only (full, static:grid:144; first 96 items in D1-30 shard order) on a further integrity root."""
    import r7_d1_teacher as TE
    S, I, _, _, _ = roots(smoke, integrity_root); rep = {"started_utc": T.utc()}; t0 = time.time()
    host = TE.LlavaPrefill(I.pred); rep["gpu"] = host.torch.cuda.get_device_name(0); rep["load_seconds"] = host.load_seconds
    for b in T.BENCHES:
        t1 = time.time(); T.run_quality(I.root, b, "0/1", host=host, limit=N_INTEGRITY); rep[f"wall_{integrity_root}_{b}"] = time.time() - t1
    rep["seconds"] = time.time() - t0; rep["env"] = V.env_info(); T.write_atomic(S / f"smoke_gpu_{integrity_root}.json", V.jdump(rep, indent=1)); print(V.jdump(rep))
    return 0


def gpu(smoke):
    from PIL import Image
    import r7_d1_teacher as TE
    S, I, SC_, sel, _ = roots(smoke); selF = selectors_f(smoke); rep = {"started_utc": T.utc()}; t0 = time.time()
    host = TE.LlavaPrefill(SC_.pred); torch = host.torch; rep["gpu"] = torch.cuda.get_device_name(0); rep["load_seconds"] = host.load_seconds
    items = RUN.load_items(SC_.inputs, "gqa_testdev"); feats = RUN.encode_unique(host, items, lambda fp: Image.open(fp).convert("RGB"))
    q = RUN.question_embeddings(host, [it["question"] for it in items]); rep["smoke_tau"] = {}
    for sd in (sel, selF):                                                              # smoke taus on the scored GQA subset (smoke only)
        _, main = T.main_units(sd / "lr_choice.json"); units = [u for o in RUN.OBJECTIVES for u in main[o]]
        bank = RUN.SelectorBank(sd, T.PCA, units); st = bank.prepare(feats, q)
        for u in units:
            tt = RUN.calibrate_tau(bank.risks(u, st)); rep["smoke_tau"][u] = tt; cfg = torch.load(str(sd / f"{u}.pt"), map_location="cpu", weights_only=False)["cfg"]
            T.write_atomic(sd / f"{u}.json", json.dumps({"cfg": cfg, "dynamic": {str(f): {"tau": v, "smoke_calibration": True} for f, v in tt.items()},
                                                         "smoke": "R7-50 smoke: random-init selector, taus calibrated on the scored GQA smoke subset"}, indent=1))
        del bank, st
    del feats; torch.cuda.empty_cache()
    rep["configs_scored_L"] = T.configs(SC_.root, "L", sel / "lr_choice.json", sel, T.PCA, d30_configs=None); rep["lines_scored_L"] = T.write_lines(SC_.root, "L")
    rep["configs_scored_F"] = T.configs(SC_.root, "F", selF / "lr_choice.json", selF, T.PCA, d30_configs=None); rep["lines_scored_F"] = T.write_lines(SC_.root, "F")
    for root, phase, kw in ((I.root, "L", {"limit": N_INTEGRITY}), (SC_.root, "L", {}), (SC_.root, "F", {})):
        for b in T.BENCHES:
            t1 = time.time(); T.run_quality(root, b, "0/1", phase, host=host, **kw); rep[f"wall_{Path(root).name}_{phase}_{b}"] = time.time() - t1
    rep["seconds"] = time.time() - t0; rep["env"] = V.env_info(); T.write_atomic(S / "smoke_gpu.json", V.jdump(rep, indent=1)); print(V.jdump(rep, indent=1))
    return 0


def ftiming(smoke):
    """Smoke only: a D1-FULL-like timing file per smoke benchmark (D1-30 layout) for the phase-F leg — the D1-30 smoke timing's dynamic rows
    relabelled with the phase-F smoke units (same objective / seed).  Synthetic latencies, labelled; never used outside the smoke."""
    S = Path(smoke); _, mL = T.main_units(roots(smoke)[3] / "lr_choice.json"); _, mF = T.main_units(selectors_f(smoke) / "lr_choice.json")
    ren = {mL[o][s]: mF[o][s] for o in RUN.OBJECTIVES for s in range(3)}; n = {}
    for tb in sorted(set(SMOKE_TMAP.values())):
        j = json.loads((SMOKE_TIMING / tb / f"timing_{tb}.json").read_text())
        rows = [dict(r, config=f"dyn:{ren[r['config'].split(':')[1]]}:{r['config'].split(':')[2]}", smoke_relabelled_from=r["config"]) for r in j["rows"]
                if r["config"].startswith("dyn:") and r["config"].split(":")[1] in ren]
        T.write_atomic(S / "timing_F" / tb / f"timing_{tb}.json", json.dumps({"rows": rows, "gpu": j.get("gpu"), "note": "R7-50 smoke only: the D1-30 smoke "
                       "timing's dynamic rows relabelled to the phase-F smoke units (synthetic; stands in for D1-FULL's own timing run)"}, indent=1))
        n[tb] = len(rows)
    print(json.dumps(n)); return 0


# ------------------------------------------------------------------------------------------------------------------------------ allow
def allow(smoke):
    """The scorer's refusals on the real code path, then the smoke's gate file (the coordinator writes the production one)."""
    S, _, SC_, sel, _ = roots(smoke); rep = {}
    def refused(fn):
        try:
            fn(); return "NOT REFUSED"
        except SystemExit as e:
            return f"refused: {str(e)[:200]}"
    sc = lambda: T.score(SC_.root, "L", None, SMOKE_TIMING, SMOKE_TMAP, 50, S / "refusal_results")
    if SC_.allow.exists():
        SC_.allow.unlink()
    rep["no_allow_file"] = refused(sc)
    SC_.allow.write_text("manifest_sha256=" + "0" * 64 + "\n"); rep["wrong_manifest_sha"] = refused(sc)
    digest = T.sha(SC_.manifest); SC_.allow.write_text(f"manifest_sha256={digest}\n")
    victim = next(p for p in T.walk_files(SC_.pred) if p.name.startswith("full@")); orig = victim.read_bytes()
    victim.write_bytes(orig + b"\n"); rep["changed_prediction_file"] = refused(sc); victim.write_bytes(orig)
    extra = SC_.pred / "gqa_testdev" / "stray.jsonl"; extra.write_text("{}\n"); rep["extra_prediction_file"] = refused(sc); extra.unlink()
    rep["generation_after_manifest"] = refused(lambda: T.run_quality(SC_.root, "gqa_testdev", "0/1"))
    SC_.allow.write_text(f"# smoke only (written by r7_50_smoke.py allow; the coordinator writes the production file)\nmanifest_sha256={digest}\n")
    LF = T.Layout(SC_.root, "F"); LF.allow.write_text(f"# smoke only\nmanifest_sha256={T.sha(LF.manifest)}\n")
    rep["ok"] = all(v.startswith("refused") for v in rep.values()); T.write_atomic(S / "smoke_allow.json", json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


# ------------------------------------------------------------------------------------------------------------------------------ integrity
def integrity(smoke, integrity_root="integrity"):
    """Integrity root vs the early D1-30 rows on the same items.  Reads prediction STRINGS and kept counts for equality only; no label, no score."""
    S, I, _, _, _ = roots(smoke, integrity_root); rep = {"tolerance": TOL, "rule": "identical answer strings and identical kept-token counts per item",
                                                         "root": str(I.root)}; ok = True
    for b, d30b in T.D30_BENCH.items():
        qmap = (lambda q: q) if b.startswith("gqa") else (lambda q: q.replace("chartqa_test_", "chartqa_val_"))
        mine = {it["qid"]: it for it in T.read_jsonl_strict(I.inputs_file(b))}; ref = {it["qid"]: it for it in T.read_jsonl_strict(D30 / "sets" / f"{d30b}.jsonl")}
        r = {"items_in_inputs": len(mine), "d1_30_items": len(ref), "inputs_missing_in_d1_30": sum(qmap(q) not in ref for q in mine)}
        same_q = same_p = same_img = 0
        for q, it in mine.items():
            o = ref[qmap(q)]; same_q += it["question"] == o["question"]; same_p += it["prompt"] == o["prompt"]; same_img += it["image_id"] == o["image_id"] and (
                it["image_file"] == o["image_file"] if b.startswith("gqa") else T.sha(it["image_file"]) == T.sha(o["image_file"]))
        r["inputs_identical"] = {"question": same_q, "prompt": same_p, "image": same_img, "of": len(mine)}
        ok &= same_q == same_p == same_img == len(mine) == len(ref)
        for cfg in INTEGRITY_ROWS + (f"sparsevlm:{INTEGRITY_SV}:official",):
            got = {x["qid"]: x for p in RUN.cfg_paths(I.pred / b, cfg) for x in V.read_jsonl(p)}
            old = {x["qid"]: x for p in RUN.cfg_paths(D30 / "pred" / d30b, cfg) for x in V.read_jsonl(p)}
            pairs = [(x, old.get(qmap(q))) for q, x in got.items()]; n = sum(o is not None for _, o in pairs)
            same = sum(o is not None and x["pred"] == o["pred"] for x, o in pairs); kept = sum(o is not None and x.get("kept") == o.get("kept") for x, o in pairs)
            tol = TOL["sparsevlm" if cfg.startswith("sparsevlm") else "pipeline"]; passed = n == len(got) > 0 and same >= tol * n and kept == n
            r[cfg] = {"compared": n, "identical_answers": same, "identical_kept": kept, "fraction": same / n if n else None, "tolerance": tol, "pass": passed}
            if cfg.startswith("static:"):                                               # qid-seeded masks: same kept SET under both qid strings?
                import r7_d1_masks as MK
                k = int(cfg.split(":")[2]); order = MK.order_grid if cfg.split(":")[1] == "grid" else MK.order_random
                sm = [set(MK.kept(order(q)[0], k).tolist()) == set(MK.kept(order(qmap(q))[0], k).tolist()) for q in got]
                eq = [x["pred"] == old[qmap(q)]["pred"] for q, x in got.items()]
                r[cfg]["qid_seeded_masks"] = {"qids_differ": any(q != qmap(q) for q in got), "same_kept_set": sum(sm),
                                              "identical_answers_where_same_set": sum(a and b for a, b in zip(sm, eq)),
                                              "identical_answers_where_set_differs": sum((not a) and b for a, b in zip(sm, eq))}
            ok &= passed
        rep[b] = r
    rep["pass"] = bool(ok); name = "smoke_integrity.json" if integrity_root == "integrity" else f"smoke_integrity_{integrity_root}.json"
    T.write_atomic(S / name, json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------------------------------------------ report
def report(smoke):
    """Code checks only: quality numbers of the smoke (random-init selectors on GQA val / ChartQA train subsets) are not results and are not quoted."""
    S, I, SC_, sel, _ = roots(smoke); out = {"prep": {}, "gpu": {}, "costs": {}, "checks": {}}
    prep_ = json.loads((S / "smoke_prep.json").read_text()); gpu_ = json.loads((S / "smoke_gpu.json").read_text())
    for k in ("build_integrity", "build_scored"):
        out["prep"][k] = {b: {x: v.get(x) for x in ("mode", "items", "groups", "images", "split_counts", "bytes_read", "seconds", "inputs_fields", "exposure_check")}
                          for b, v in prep_[k]["benches"].items()}
        out["prep"][k]["wall_s"] = prep_[k].get("wall_s")
    out["prep"]["refusals"] = {k: prep_[k] for k in ("refuse_closed_without_freeze", "refuse_test_name_outside_closed")}
    meta = json.loads((D1 / "data" / "d1_manifest_meta.json").read_text()).get("image_exposure", {}).get("DEV")   # frozen D1 disclosure (DEV)
    ex = prep_["build_integrity"]["benches"]["gqa_testdev"].get("exposure_check") or {}
    out["checks"]["exposure_check_vs_d1_dev_disclosure"] = {"d1_frozen": meta, "r7_50_check": {k: ex.get(k) for k in ("images", "images_in_mix_gqa_paths",
                                                            "questions_seen_in_mix_gqa_turns")},
                                                            "match": bool(meta) and ex.get("images") == meta.get("images") and
                                                            ex.get("images_in_mix_gqa_paths") == meta.get("in_llava_gqa_mix") and ex.get("questions_seen_in_mix_gqa_turns") == 0}
    out["prep"]["selector_record"] = prep_["selector_record"]
    out["gpu"] = {k: gpu_.get(k) for k in ("gpu", "load_seconds", "configs_scored_L", "lines_scored_L", "configs_scored_F", "lines_scored_F", "seconds")}
    out["gpu"]["walls"] = {k: v for k, v in gpu_.items() if k.startswith("wall_")}
    for L, tag in ((I, "integrity"), (SC_, "scored"), (T.Layout(SC_.root, "F"), "scored_F")):     # measured per-answer costs
        for b in T.BENCHES:
            d = L.pred / b
            for m in sorted(d.glob("done_shard*.json")):
                j = json.loads(m.read_text()); n = j["items"] * j["configs"]
                out["costs"][f"{tag}:{b}:pipeline"] = {"answers": n, "seconds": j["seconds"], "s_per_answer": j["seconds"] / n if n and j["seconds"] else None,
                                                       "gpu": j["env"].get("gpu"), "items": j["items"], "rows": j["configs"]}
            rows = [json.loads(x) for f in sorted(d.glob("sparsevlm_rows_*.jsonl")) for x in f.read_text().splitlines() if x.strip()]
            if rows:
                n = sum(r["items"] for r in rows); s = sum(r["seconds"] for r in rows)
                out["costs"][f"{tag}:{b}:sparsevlm"] = {"answers": n, "seconds": s, "s_per_answer": s / n if n else None, "rows": len(rows),
                                                        "counter_checks": [bool(r.get("counter_check")) for r in rows]}
    out["checks"]["manifest"] = {}
    for ph in ("L", "F"):
        Lp = T.Layout(SC_.root, ph); man = json.loads(Lp.manifest.read_text())
        out["checks"]["manifest"][ph] = {"n_files": man["n_files"], "rows": {b: v["n_rows"] for b, v in man["rows"].items()}, "items": {b: v["items"] for b, v in man["inputs"].items()},
                                         "sha256": T.sha(Lp.manifest), "coverage_complete": all(all(c["answers"] == man["inputs"][b]["items"] for c in cov.values())
                                                                                                 for b, cov in man["coverage"].items())}
    out["checks"]["gate_refusals"] = json.loads((S / "smoke_allow.json").read_text()); chk = {}
    for ph in ("L", "F"):
        res = json.loads((T.Layout(SC_.root, ph).results / "r7_50_results.json").read_text())
        for b, r in res["benches"].items():
            prim = {k: v for k, v in r["contrasts"].items() if v.get("role") == "primary"}
            chk[f"{ph}:{b}"] = {"rows": len(r["rows"]), "primary_contrasts": len(prim), "primary_reached": sum("p" in v for v in prim.values()),
                                "holm_family_sizes": sorted({v.get("holm_family_size") for v in prim.values() if "p" in v}),
                                "ci_contains_point": all(v["ci95"][0] - 1e-9 <= v["delta"] <= v["ci95"][1] + 1e-9 for v in prim.values() if "p" in v),
                                "quality_in_range": all(0 <= v["quality"] <= 100 for v in r["rows"].values()),
                                "latency_columns": sum(bool(v["latency_profiling_workload_ms"]) for v in r["rows"].values()),
                                "dynamic_rows": len(r["dynamic_rows_on_test"]), "dynamic_shares_sum_to_1": all(abs(sum(v["kept_tokens_share_test"].values()) - 1) < 1e-9
                                                                                                              for v in r["dynamic_rows_on_test"].values()),
                                "subsets": sorted(r.get("subsets", {})), "sparsevlm_counters": len(r["sparsevlm_counters"]),
                                "sparsevlm_counters_ok": all(c.get("within_tolerance") is True for c in r["sparsevlm_counters"]),
                                "row_sources": sorted(set(r.get("row_sources", {}).values())), "notes": r.get("notes", []),
                                "integrity_totals": r["integrity"]["totals"], "secondary_contrasts": sum(v.get("role") != "primary" for v in r["contrasts"].values())}
        out["checks"][f"scorer_latency_label_present_{ph}"] = bool(res.get("latency_label"))
    out["checks"]["scorer"] = chk; out["checks"]["markdown_bytes"] = {ph: (T.Layout(SC_.root, ph).results / "r7_50_results.md").stat().st_size for ph in ("L", "F")}
    out["checks"]["integrity"] = {f.stem.replace("smoke_integrity", "") or "test_ids": json.loads(f.read_text()) for f in sorted(S.glob("smoke_integrity*.json"))}
    import r7_d1_30_sparsevlm as SV                                                     # E3 counter drift at the production shard sizes
    drift = {}
    for row in T.sparsevlm_rows("gqa_testdev"):
        Tk = int(row.split(":")[1]); pred = SV.official_equal_tokens(SV.adapted_list(Tk))
        drift[row] = {str(n): SV.official_counter_float32(pred, n) - pred for n in (2500, 4200, 5000, 6500, 12578)}
    out["checks"]["sparsevlm_float32_counter_drift"] = {"tolerance": SV.COUNTER_TOL, "max_abs": max(abs(v) for d in drift.values() for v in d.values()), "rows": drift}
    T.write_atomic(S / "smoke_summary.json", json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["prep", "prep-integrity", "gpu", "gpu-integrity", "ftiming", "allow", "integrity", "report"])
    ap.add_argument("--smoke", required=True); ap.add_argument("--integrity-root", default="integrity"); ap.add_argument("--chartqa-qid-prefix", default=None)
    a = ap.parse_args()
    if a.cmd == "prep-integrity":
        return prep_integrity(a.smoke, a.integrity_root, a.chartqa_qid_prefix)
    if a.cmd == "gpu-integrity":
        return gpu_integrity(a.smoke, a.integrity_root)
    if a.cmd == "integrity":
        return integrity(a.smoke, a.integrity_root)
    return {"prep": prep, "gpu": gpu, "ftiming": ftiming, "allow": allow, "report": report}[a.cmd](a.smoke)


if __name__ == "__main__":
    sys.exit(main())
