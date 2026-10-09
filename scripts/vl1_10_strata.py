"""VL1-10 frozen secondary reading 5 (descriptive): DEV Top-1 per query in two strata, from the saved checkpoints (evaluation only).
  same_category : the target shares its COCO category with another referred object of the image (the category shortcut cannot solve it)
  long_expr     : expression longer than the DEV median token count (REFER tokens)
Routes: raw (cosine), VCS / JS / SigLIP-style (estimator-selected checkpoint), softmax (task-selected, selected on DEV: optimistic); RFF keeps no
checkpoint and is omitted.  Paired-by-seed contrasts per stratum: VCS − JS, VCS − raw, softmax − VCS.  Also the fit-time column (seconds of the
selected candidate and of the whole lr grid).
    python scripts/vl1_10_strata.py [--dir outputs/VL1_10] [--features <cache>] [--ns all] [--out reports/VL1/VL1_10_strata.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vl1_10_fit as FT  # noqa: E402
from vl1_10_aggregate import tint  # noqa: E402

RG = FT.RG; SEEDS = (0, 1, 2)


def query_meta(dev_recs):
    """Per DEV query (rec order, column order of Scene.compatibility()): same-category flag and token count."""
    scenes, _ = RG.load_scenes(resolve_paths=False); sc = {s.image_id: s for s in scenes}; same, ntok = [], []
    for r in dev_recs:
        s = sc[r["image_id"]]; cats = [o.category_id for o in s.referred]
        for i, o in enumerate(s.referred):
            for e in o.expressions:
                same.append(cats.count(cats[i]) > 1); ntok.append(e["n_tokens"])
    return np.array(same), np.array(ntok)


@torch.no_grad()
def hits(score_fn, recs) -> np.ndarray:
    out = []
    for s in range(0, len(recs), 64):
        chunk = recs[s:s + 64]; b = FT.pad(chunk); f = score_fn(b["U"], b["V"]).masked_fill(~b["rmask"][:, :, None], -torch.inf)
        h = (f.argmax(1) == b["tgt"])
        for i, r in enumerate(chunk):
            out.append(h[i, :len(r["V"])].numpy())
    return np.concatenate(out)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="outputs/VL1_10"); ap.add_argument("--features", default=str(FT.FEAT))
    ap.add_argument("--ns", nargs="+", default=["all"]); ap.add_argument("--routes", nargs="+", default=["vcs", "js", "siglip", "softmax"])
    ap.add_argument("--out", default="reports/VL1/VL1_10_strata.json"); a = ap.parse_args()
    D = Path("/home/infres/yinwang/CS_QMI") / a.dir; out_path = HERE.parent / a.out
    data, _ = FT.load_scene_tensors(False, Path(a.features), "region_uniform"); dev = data["DEV"]
    same, ntok = query_meta(dev); med = float(np.median(ntok)); long_ = ntok > med
    assert len(same) == sum(len(r["V"]) for r in dev)
    strata = {"all": np.ones_like(same), "same_category": same, "other_category": ~same, "long_expr": long_, "short_expr": ~long_}
    res = {"n_queries": int(len(same)), "median_tokens": med, "stratum_sizes": {k: int(v.sum()) for k, v in strata.items()}, "cells": {}, "contrasts": {}, "fit_seconds": {}}
    d = data["DEV"][0]["U"].shape[1]
    raw_h = hits(lambda U, V: torch.einsum("nrd,nwd->nrw", U, V), dev)
    res["cells"]["raw"] = {k: float(raw_h[m].mean()) for k, m in strata.items()}
    per = {}
    for n in a.ns:
        for route in a.routes:
            hs = []
            for s in SEEDS:
                f = D / f"{route}_N{n}_s{s}_states.pt"
                if not f.exists():
                    continue
                st = torch.load(f, weights_only=False)["task" if route == "softmax" else "est"]
                m = FT.PairMLP(d=d, kind="residual"); m.load_state_dict(st); m.eval(); hs.append(hits(m, dev))
                j = json.load(open(D / f"{route}_N{n}_s{s}.json")); sel = j["task_selected" if route == "softmax" else "estimator_selected"]
                fs = [c["fit_seconds"] for c in j["candidates"]]
                res["fit_seconds"].setdefault(f"{route}/N{n}", []).append({"selected_lr": sel["lr"], "selected": next(c["fit_seconds"] for c in j["candidates"] if c["lr"] == sel["lr"]), "grid_total": float(sum(fs))})
            if len(hs) == len(SEEDS):
                per[(route, n)] = hs
                res["cells"][f"{route}/N{n}"] = {k: tint([h[mk].mean() for h in hs]) for k, mk in strata.items()}
        for x, y in (("vcs", "js"), ("vcs", "raw"), ("softmax", "vcs")):
            if (x, n) in per and (y == "raw" or (y, n) in per):
                ys = [raw_h] * 3 if y == "raw" else per[(y, n)]
                res["contrasts"][f"{x}_minus_{y}/N{n}"] = {k: tint([hx[mk].mean() - hy[mk].mean() for hx, hy in zip(per[(x, n)], ys)]) for k, mk in strata.items()}
    for k, v in list(res["fit_seconds"].items()):
        res["fit_seconds"][k] = {"selected_mean": float(np.mean([x["selected"] for x in v])), "grid_total_mean": float(np.mean([x["grid_total"] for x in v]))}
    json.dump(res, open(out_path, "w"), indent=1)
    p = lambda t: f"{100 * t['mean']:.2f}" + (f" [{100 * t['ci95'][0]:+.2f},{100 * t['ci95'][1]:+.2f}]" if t.get("ci95") else "")
    print("strata sizes", res["stratum_sizes"], "median tokens", med)
    print("raw", {k: round(100 * v, 2) for k, v in res["cells"]["raw"].items()})
    for k, c in res["cells"].items():
        if k != "raw":
            print(k, {kk: p(v) for kk, v in c.items()})
    for k, c in res["contrasts"].items():
        print(k, {kk: p(v) for kk, v in c.items()})
    for k, v in res["fit_seconds"].items():
        print(f"fit seconds {k}: selected {v['selected_mean']:.0f}, lr grid {v['grid_total_mean']:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
