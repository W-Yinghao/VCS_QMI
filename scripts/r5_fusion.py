"""r5 R5-10 (package r5 docs/04 §2; machine/fusion_grid.json): R4-VL2 read with symmetric CAL fusion.  Exploratory (DEV has been reused).
F0  the frozen R4-VL2 readings, copied verbatim from reports/R4/R4_VL2_results.json (C1–C4, per-seed rows, missing items); never recomputed.
F1  per fixed CAL-selected checkpoint — R4-VL1 VCS / JS / set softmax (ν uniform), R4-VL2 VCS(ν) / JS(ν) — the score 2a·cos + λ·log s_r with
    λ ∈ {0, 0.25, 0.5, 1, 2, 4} chosen on CAL all-proposal image-macro Acc@0.5 (ties: smallest λ; DEV never used); DEV reported only at λ = 0, λ = 1
    and the CAL-chosen λ.  For ν models λ = 1 is the native rank (log ν and log s differ by a per-image constant); any other λ is labelled
    posthoc_task_calibration.  For uniform models λ > 0 is test-time information fusion, not their density-ratio correction.  Contrasts paired by
    seed (95 % t over seeds) mirroring R4-VL2 C1–C4 under F1, plus every model at the fixed λ = 1.  Reproduction check: DEV at λ = 0 (uniform) or
    λ = 1 (ν) against each run's recorded DEV value.  Wording: an interval containing 0 only contains 0 (no equivalence margin was pre-specified).
F2  joint epoch × λ selection: not identifiable from the saved states (only each run's CAL-selected state is saved) — checked by listing the
    checkpoint directories.
Writes reports/R5/fusion_F0_F1.json, reports/R5/selection_F2_availability.json and reports/R5/fusion_tables.md; missing inputs are listed, never filled.
    python scripts/r5_fusion.py [--datasets refcocog refcocoplus] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r5" / "package_r5_20261011" / "reference"))
from vl1_10_aggregate import tint  # noqa: E402
from vl1_22_tableC import iou_xyxy  # noqa: E402
from r4_vl0_scores import meta_path  # noqa: E402
from r5_diagnostics import BB, MODELS, OUT, SEEDS, critic, key, load_scores, run_json  # noqa: E402
from r5_reference import choose_cal_lambda  # noqa: E402

GRID = json.load(open(Path(__file__).resolve().parents[1] / "r5" / "package_r5_20261011" / "machine" / "fusion_grid.json"))
LAMBDAS = tuple(float(x) for x in GRID["lambda_grid"]); FUSION_MODELS = [m for m in MODELS if m[0] in ("r4vl1", "r4vl2")]
CKPTS = {"r4vl1": Path("/projects/EEG-foundation-model/yinghao/models/r4_vl1"), "r4vl2": Path("/projects/EEG-foundation-model/yinghao/models/r4_vl2")}
CONTRASTS = {"C1_F1 VCS(nu) - set softmax": ("r4vl2:vcs", "r4vl1:softmax"), "C2_F1 VCS(nu) - VCS": ("r4vl2:vcs", "r4vl1:vcs"),
             "C4_F1 VCS(nu) - JS(nu)": ("r4vl2:vcs", "r4vl2:js"), "JS(nu) - JS": ("r4vl2:js", "r4vl1:js"), "JS(nu) - set softmax": ("r4vl2:js", "r4vl1:softmax"),
             "VCS - set softmax (both fused on CAL)": ("r4vl1:vcs", "r4vl1:softmax")}


def valid_matrices(meta):
    """Per image [n_p, n_q] boolean: IoU(proposal, the query's target box) >= 0.5."""
    out = []
    for m in meta:
        P = m["prop_boxes"]
        out.append(np.zeros((0, len(m["target"])), bool) if len(P) == 0 else np.stack([iou_xyxy(m["ref_boxes"][t], P) >= 0.5 for t in m["target"]], 1))
    return out


def accuracy(cps, meta, valid, a, lam):
    """Image-macro Acc@0.5 of argmax(2a cos + λ log s); uncovered queries and images without proposals count as errors."""
    per = []
    for cp, m, v in zip(cps, meta, valid):
        if cp.shape[0] == 0:
            per.append(0.0); continue
        S = 2 * a * cp + (lam * np.log(np.asarray(m["det_score"], float))[:, None] if lam else 0.0); pred = S.argmax(0)
        per.append(float(v[pred, np.arange(cp.shape[1])].mean()))
    return float(np.mean(per))


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--datasets", nargs="+", default=["refcocog", "refcocoplus"]); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    out = OUT / "smoke" if a.smoke else OUT; out.mkdir(parents=True, exist_ok=True)
    res = {"grid": GRID, "F0": None, "F1": {}, "missing": [], "cal_evaluations": 0, "note": "an interval containing 0 only contains 0; no equivalence margin was pre-specified"}
    f0 = Path(__file__).resolve().parents[1] / "reports" / "R4" / "R4_VL2_results.json"
    if f0.exists():
        r0 = json.load(open(f0)); res["F0"] = {"source": str(f0), "status": "complete" if not r0.get("missing") else "incomplete", "missing": r0.get("missing"),
                                              "gates": r0.get("gates"), "cells": r0.get("cells"), "pooled": r0.get("pooled")}
    else:
        res["F0"] = {"source": str(f0), "status": "absent"}; res["missing"].append(str(f0))
    for ds in a.datasets:
        meta = {r: torch.load(meta_path(ds, r), weights_only=False) for r in ("CAL", "DEV")}
        if a.smoke:
            meta = {r: v[:150] for r, v in meta.items()}
        valid = {r: valid_matrices(v) for r, v in meta.items()}; cell = {"models": {}, "contrasts": {}}
        for m in FUSION_MODELS:
            for s in SEEDS:
                ab = critic(ds, m, s, res["missing"]); cps = {r: load_scores(ds, r, m[0], m[1], s, meta[r], res["missing"], a.smoke) for r in ("CAL", "DEV")}
                if ab is None or any(v is None for v in cps.values()):
                    continue
                cal = {lam: accuracy(cps["CAL"], meta["CAL"], valid["CAL"], ab[0], lam) for lam in LAMBDAS}; res["cal_evaluations"] += len(LAMBDAS)
                lam_c = choose_cal_lambda(cal); dev = {str(lam): accuracy(cps["DEV"], meta["DEV"], valid["DEV"], ab[0], lam) for lam in sorted({0.0, 1.0, lam_c})}
                rec_dev = json.load(open(run_json(ds, m, s)))["dev_at_selected"]["acc50"]["image_macro"]; native = 1.0 if m[2] == "score" else 0.0
                cell["models"].setdefault(key(m), {})[str(s)] = {"label": m[4], "critic_a_b": ab, "cal_by_lambda": {str(k): v for k, v in cal.items()}, "lambda_cal": lam_c,
                                                                  "lambda_label": "native" if lam_c == native else ("posthoc_task_calibration" if m[2] == "score" else "test_time_fusion"),
                                                                  "dev": dev, "dev_at_cal_lambda": dev[str(lam_c)], "dev_fixed_lambda1": dev["1.0"],
                                                                  "reproduction": {"recorded_dev": rec_dev, "recomputed_native": dev[str(native)], "diff": dev[str(native)] - rec_dev}}
        for name, (x, y) in CONTRASTS.items():
            for field in ("dev_at_cal_lambda", "dev_fixed_lambda1"):
                mx, my = cell["models"].get(x, {}), cell["models"].get(y, {}); ss = [s for s in mx if s in my]
                if ss:
                    cell["contrasts"].setdefault(name, {})[field] = {**tint([mx[s][field] - my[s][field] for s in ss]), "seeds": [int(s) for s in ss]}
        res["F1"][ds] = cell
    f2 = {"status": "not_identifiable_from_saved_states", "reason": "only each run's CAL-selected state is saved; per-epoch scores or weights do not exist", "checkpoint_dirs": {}}
    for src, d in CKPTS.items():
        files = sorted(p.name for p in d.glob("*.pt")) if d.exists() else []; f2["checkpoint_dirs"][src] = {"path": str(d), "n_files": len(files),
                                                                                                             "all_calAcc_single_state": all(n.endswith("_calAcc.pt") for n in files), "files": files}
    f2["history_note"] = "run JSONs keep per-epoch CAL metrics at the native rank only; no per-epoch scores, so a common epoch x lambda grid cannot be formed"
    json.dump(f2, open(out / "selection_F2_availability.json", "w"), indent=1); json.dump(res, open(out / "fusion_F0_F1.json", "w"), indent=1)
    p = lambda v: "—" if v is None else f"{100 * v:.2f}"
    q = lambda t: f"{100 * t['mean']:+.2f}" + (f" [{100 * t['ci95'][0]:+.2f}, {100 * t['ci95'][1]:+.2f}]" if t.get("ci95") else " (1 seed)")
    L = ["# r5 R5-10 — R4-VL2 with symmetric CAL fusion (F1); exploratory", "",
         f"F0 (frozen R4-VL2 readings): {res['F0']['status']}.  F2: {f2['status']}.  CAL evaluations added: {res['cal_evaluations']}.  {res['note']}.", ""]
    for ds, cell in res["F1"].items():
        L += [f"## {ds}", "", "| model | seed | λ chosen on CAL (label) | DEV λ = 0 | DEV λ = 1 | DEV at CAL λ | reproduction diff |", "|---|---|---|---|---|---|---|"]
        for k_, seeds in cell["models"].items():
            for s, r in sorted(seeds.items()):
                L.append(f"| {r['label']} | {s} | {r['lambda_cal']:g} ({r['lambda_label']}) | {p(r['dev'].get('0.0'))} | {p(r['dev'].get('1.0'))} | {p(r['dev_at_cal_lambda'])} | {100 * r['reproduction']['diff']:+.2f} |")
        L += ["", "| contrast (paired by seed) | at the CAL-chosen λ | at fixed λ = 1 |", "|---|---|---|"]
        for name, c in cell["contrasts"].items():
            L.append(f"| {name} | {q(c['dev_at_cal_lambda']) if 'dev_at_cal_lambda' in c else '—'} | {q(c['dev_fixed_lambda1']) if 'dev_fixed_lambda1' in c else '—'} |")
        L.append("")
    if res["missing"]:
        L += ["Missing inputs (not filled):", ""] + [f"- {x}" for x in sorted(set(res["missing"]))]
    (out / "fusion_tables.md").write_text("\n".join(L) + "\n"); print("\n".join(L[:60]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
