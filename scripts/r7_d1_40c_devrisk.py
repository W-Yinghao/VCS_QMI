"""r7 R7-40c DEV secondaries on Qwen2.5-VL-7B (vl_baselines env, one GPU; run after the two sealed DEV teacher bundles b012 / b013, which the
frozen R7-40a teacher code produces with --allow-dev on the 40a GPU class, D40a-4 / D40a-5):
  (a) DEV completeness: every DEV shard's done-record present == expected, no duplicate, no skipped image; per-role total == the frozen
      manifest; manifest sha256 unchanged (reads done-records and the manifest only) -> <out>/dev_completeness.json (the R7-40a CAL / FIT
      file teacher/r7_40a_completeness.json is never touched)
  (b) the yes / no-mass QC line for DEV promised by D40a-1 (minimum mass and count below 0.9 per pass, low-mass items by GQA type)
  (c) the D1-30 secondary 'selector risk calibration on DEV masks' re-applied: every R7-40b unit's predicted r(K) for the DEV binary
      questions × their 12 R7-40a masks (3 mechanisms × 4 fractions) against the realised teacher h = (C − a)² − (C − b)²: Spearman with an
      image-bootstrap CI (2,000), least-squares slope / intercept of h on r, per fraction and per mechanism.  r(K) is computed with the frozen
      training-side code (feature cache encode_role40 with the 40b PCA -> r7_d1_40_train.assemble40 -> r7_d1_40_selectors.predict_r40), i.e.
      exactly as the CAL Spearman that selected the checkpoints.  This is the first and only read of the sealed DEV teacher records (through
      the frozen D1-10 analyser's coverage guard, r7_d1_analyze.load_role).
    python -I scripts/r7_d1_40c_devrisk.py run --out O [--lr-choice L]                          (production)
    python -I scripts/r7_d1_40c_devrisk.py smoke --out O --records R --fixture F --selectors-dir S --pca P --units U ...   (fixture probe records)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts")); sys.path.insert(0, str(R / "r7" / "package_r7_20261011" / "reference"))
import r7_d1_30_devrisk as DR  # noqa: E402  (frozen D1-30: Spearman)
import r7_d1_40_masks as M4  # noqa: E402
import r7_d1_40c_plan as PL  # noqa: E402
import r7_vl_common as V  # noqa: E402

MANIFEST_DIR = PL.D1 / "data"
TEACHER = PL.R40 / "teacher"
FROZEN_SHA = "32c72126ec9d40a362c67883f3a0b92a7212ae063b7ab9a369c47a696706d794"
EVAL_COST = 64 * 13 * 606 ** 2 / 8      # predict_r40's grouping cap / 8: the same values (grouping is exact up to float order), explicit-attention
                                        # memory ≈ 1.2 GB per layer tensor instead of ≈ 9.8 GB, so it fits next to the 16 GB host on a 44 GB L40S


def dev_complete(teacher_dir: Path, manifest_dir: Path, frozen_sha=FROZEN_SHA):
    """(a): reads the DEV done-records and the manifest only (no teacher probability)."""
    plan = json.loads((manifest_dir / "d1_shards.json").read_text()); meta = json.loads((manifest_dir / "d1_manifest_meta.json").read_text())
    sha = hashlib.sha256((manifest_dir / "d1_manifest.jsonl").read_bytes()).hexdigest()
    expected = sum(1 for x in (manifest_dir / "d1_manifest.jsonl").read_text().splitlines() if x.strip() and json.loads(x)["role"] == "DEV")
    shards, bad, tot = {}, [], 0
    for s in plan["shards"]:
        if not s.startswith("DEV-"):
            continue
        f = teacher_dir / f"{s}.done.json"
        if not f.exists():
            bad.append(f"{s}: missing done-record"); continue
        d = json.loads(f.read_text()); shards[s] = {k: d[k] for k in ("expected", "present", "duplicates", "gpu")}; shards[s]["skipped"] = len(d["skipped_images"])
        if d["present"] != d["expected"] or d["duplicates"] or d["skipped_images"]:
            bad.append(f"{s}: {shards[s]}")
        tot += d["present"]
    res = {"role": "DEV", "shards": len([s for s in plan["shards"] if s.startswith("DEV-")]), "done_records": len(shards), "problems": bad, "present": tot,
           "manifest_dev_questions": expected, "manifest_sha256": sha, "sha256_unchanged": sha == frozen_sha == meta.get("manifest_sha256"),
           "gpus": sorted({str(v["gpu"]) for v in shards.values()})}
    res["complete"] = (not bad and res["sha256_unchanged"] and tot == expected and meta.get("frozen") is True)
    return res


def summarize40(r, h, frac, mech, groups, reps=2000):
    """(c) per unit: the D1-30 devrisk summary on fractions (Spearman with an image-bootstrap CI, slope / intercept of h on r, by fraction and
    by mechanism)."""
    r, h, frac, mech, groups = (np.asarray(x) for x in (r, h, frac, mech, groups)); r, h = r.astype(float), h.astype(float)
    A = np.vstack([r, np.ones_like(r)]).T; slope, icpt = np.linalg.lstsq(A, h, rcond=None)[0]
    ug, inv = np.unique(groups, return_inverse=True); mem = [np.flatnonzero(inv == j) for j in range(len(ug))]; rng = np.random.default_rng(0); bs = []
    for _ in range(reps):
        idx = np.concatenate([mem[j] for j in rng.integers(0, len(ug), len(ug))]); v = DR.spearman(r[idx], h[idx])
        if v is not None:
            bs.append(v)
    return {"n_pairs": int(len(r)), "n_images": int(len(ug)), "spearman_r_h": DR.spearman(r, h),
            "spearman_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if bs else None, "slope_h_on_r": float(slope), "intercept": float(icpt),
            "by_fraction": {k: {"mean_r": float(r[frac == k].mean()), "mean_h": float(h[frac == k].mean())} for k in M4.FKEYS},
            "by_mechanism": {m: {"mean_r": float(r[mech == m].mean()), "mean_h": float(h[mech == m].mean()), "spearman": DR.spearman(r[mech == m], h[mech == m])}
                             for m in M4.MECHANISMS}}


def risk_calibration(host, records, questions, opener, sel_dir: Path, pca_path: Path, units, reps=2000, device="cuda"):
    """(c) with the frozen training-side code on freshly encoded features of the records' images."""
    import torch
    import r7_d1_40_cache as C4
    import r7_d1_40_selectors as S4
    import r7_d1_40_train as TR40
    pca = torch.load(str(pca_path), map_location="cpu", weights_only=False); ids = sorted({r["image_id"] for r in records})
    feats, index = C4.encode_role40(host, pca, ids, opener)
    qid_ids = {r["qid"]: host.tok(questions[r["qid"]], add_special_tokens=False).input_ids for r in records}
    emb = host.core.embed.weight.detach().cpu().clone(); out = {}
    img_of, q_ids, q_len, rank, kk, C, h, d = TR40.assemble40(records, index, qid_ids)
    base = S4.Cfg40()
    data = S4.TupleData40(torch.from_numpy(feats), index["offsets"], index["n"], [g[1] for g in index["grid"]], img_of, q_ids, q_len, emb, rank, kk, C, h, d,
                          device, base.max_rows, base.max_cols)
    mech = np.array([m for m in M4.MECHANISMS for _ in M4.FKEYS]); frac = np.array([k for _ in M4.MECHANISMS for k in M4.FKEYS])
    grp = np.array([r["image_id"] for r in records])
    for u in units:
        ck = torch.load(str(Path(sel_dir) / f"{u}.pt"), map_location=device, weights_only=False)
        cfg = S4.Cfg40(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in ck["cfg"].items()})
        m = S4.Selector40(cfg, ck["objective"]).to(device); m.load_state_dict(ck["state_dict"]); m.eval()
        rr = S4.predict_r40(m, data, np.arange(data.n), cost=EVAL_COST)                  # [n, 12], mask order mechanism-major × fractions
        out[u] = summarize40(rr.reshape(-1), h.reshape(-1), np.tile(frac, data.n), np.tile(mech, data.n), np.repeat(grp, rr.shape[1]), reps)
        out[u]["objective"] = ck["objective"]
    return out, {"images": len(ids), "tokens": int(feats.shape[0]), "grid_max": [max(g[0] for g in index["grid"]), max(g[1] for g in index["grid"])]}


def main(argv=None, host=None) -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--out", default=str(PL.R40C / "devrisk")); r.add_argument("--lr-choice", default=str(PL.R40 / "selectors" / "lr_choice.json"))
    r.add_argument("--teacher-dir", default=str(TEACHER)); r.add_argument("--manifest-dir", default=str(MANIFEST_DIR))
    r.add_argument("--selectors-dir", default=str(PL.R40 / "selectors")); r.add_argument("--pca", default=str(PL.R40 / "cache" / "pca.pt")); r.add_argument("--reps", type=int, default=2000)
    s = sub.add_parser("smoke"); s.add_argument("--out", required=True); s.add_argument("--records", required=True); s.add_argument("--fixture", required=True)
    s.add_argument("--selectors-dir", required=True); s.add_argument("--pca", required=True); s.add_argument("--units", nargs="+", required=True); s.add_argument("--reps", type=int, default=200)
    a = ap.parse_args(argv); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    import r7_d1_analyze as AN
    import r7_d1_40_analyze as AN40
    if a.cmd == "run":
        comp = dev_complete(Path(a.teacher_dir), Path(a.manifest_dir)); (out / "dev_completeness.json").write_text(json.dumps(comp, indent=1))
        if not comp["complete"]:
            print(json.dumps(comp, indent=1)); raise SystemExit("DEV teacher records incomplete: the DEV secondaries do not run")
        rows, cov = AN.load_role(Path(a.teacher_dir), Path(a.manifest_dir), "DEV")       # the first and only read of the sealed DEV records
        man = {}
        for x in (Path(a.manifest_dir) / "d1_manifest.jsonl").read_text().splitlines():
            if x.strip():
                j = json.loads(x); man[j["qid"]] = j["question"]
        import r7_d1_40_teacher as TE40
        opener = TE40.ExtractedImageSource().open; units = [u for o in PL.OBJECTIVES for u in PL.unit_names(json.loads(Path(a.lr_choice).read_text()))[o]]
        name = "r7_40c_devrisk.json"
    else:
        rows = V.read_jsonl(Path(a.records)); cov = {"smoke_records": len(rows)}; man = {}
        for x in Path(a.fixture).read_text().splitlines():
            if x.strip():
                j = json.loads(x); man[j["qid"]] = j["question"]; man.setdefault("__files__", {})[j["image_id"]] = j["image_file"]
        files = man.pop("__files__"); from PIL import Image
        opener = lambda i: Image.open(files[i]).convert("RGB")  # noqa: E731
        units = a.units; name = "r7_40c_devrisk_smoke.json"
    if host is None:
        import r7_d1_40_host as QH
        host = QH.QwenPrefill(out)
    calib, feat_info = risk_calibration(host, rows, man, opener, Path(a.selectors_dir), Path(a.pca), units, a.reps)
    res = {"coverage": cov, "qc_yes_no_mass": AN.mass_qc(rows), "qc_low_mass_by_type": AN40.low_mass_by_type(rows), "features": feat_info,
           "units": calib, "env": V.env_info(), "seconds": time.time() - t0,
           "note": "D1-30 secondary 'selector risk calibration on DEV masks' re-applied to the R7-40b units; DEV yes / no-mass QC line (D40a-1)"}
    (out / name).write_text(V.jdump(res, indent=1)); print(f"[r7_d1_40c_devrisk] {a.cmd}: {len(units)} units, {len(rows)} questions -> {out / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
