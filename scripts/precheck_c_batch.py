"""Second-application pre-check C — batch decoupling: degradation of VCS / InfoNCE / pairwise-logistic adapters as the batch shrinks.

    python scripts/precheck_c_batch.py --features <P49 dir> --out <prefix> [--pairing topic] [--smoke]

Re-uses pre-check A's pipeline (frozen CLIP features, identity-initialised 512×512 adapters, topic pairing = setting 2).  Two axes, reported
separately (brief §2 C):  FIXED UPDATES — every batch size trains for the same number of optimizer steps U;  FIXED EXPOSURE — every batch size
trains for the same number of epochs E (so small batches take more steps).  Batch sizes B ∈ {16, 32, 64, 128, 256, 512}.  At every (method, B, axis)
the learning rate is selected on SRC-CAL from lr ∈ {1e-3, 3e-4, 1e-4} (method's own loss) — equal tuning budget, budgets tabled; 3 seeds of the
selected lr.  VCS uses K = 8 independent-pool negatives regardless of B (its negatives do not depend on the batch); InfoNCE and logistic use the
in-batch pairs (their native form), so at B = 16 they have 15 negatives per anchor.
Metrics per point: retrieval R@1 (ranking, secondary); balanced error at the CAL-quantile threshold (FNR 5 %) on SRC-EVAL and TGT-EVAL
(non-ranking primary); ECE of score + Platt(CAL); held-out J (VCS).  Retention(B) = metric(B) / metric(256) for R@1 and 1 − balanced error.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from precheck_a_adapters import brier, ece, load_split, pair_sets, platt, retrieval, threshold_at_fnr, topic_partners, train, heldout_J  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--pairing", default="topic", choices=["exact", "topic"]); ap.add_argument("--index-dir", default="/home/infres/yinwang/CS_QMI/data/coco_index")
    ap.add_argument("--batches", default="16,32,64,128,256,512"); ap.add_argument("--lrs", default="1e-3,3e-4,1e-4"); ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--updates", type=int, default=3120, help="fixed-updates axis: optimizer steps for every batch size (= 40 epochs at B 256 on 20 000 images)")
    ap.add_argument("--exposure-epochs", type=int, default=40, help="fixed-exposure axis: epochs for every batch size")
    ap.add_argument("--fnr", type=float, default=0.05); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    fd = Path(a.features); raw = {k: load_split(fd, k) for k in ("SRC-FIT", "SRC-CAL", "SRC-EVAL", "TGT-EVAL")}
    batches = [int(x) for x in a.batches.split(",")]; lrs = [float(x) for x in a.lrs.split(",")]; seeds = [int(x) for x in a.seeds.split(",")]
    if a.smoke:
        batches, lrs, seeds, a.updates, a.exposure_epochs = [16, 256], [1e-3], [0], 30, 1
        raw = {k: (v[0][:2000], v[1][:2000], v[2][:2000]) for k, v in raw.items()}
    splits = {k: (v[0], v[1]) for k, v in raw.items()}; partners = {k: (None, None) for k in splits}
    if a.pairing == "topic":
        for k, v in raw.items():
            partners[k] = topic_partners(v[2], a.index_dir, 20260927)
    fit_img, fit_txt = splits["SRC-FIT"]; cal_img, cal_txt = splits["SRC-CAL"]; fit_p, cal_p = partners["SRC-FIT"][0], partners["SRC-CAL"][0]
    results = {"settings": vars(a), "device": str(device), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "axes": {}}
    t0 = time.time()
    for axis in ("fixed_updates", "fixed_exposure"):
        results["axes"][axis] = {}
        for method in ("vcs", "infonce", "logistic"):
            results["axes"][axis][method] = {}
            for B in batches:
                kw = {"max_steps": a.updates} if axis == "fixed_updates" else {}
                ep = 1 if axis == "fixed_updates" else a.exposure_epochs   # epochs ignored when max_steps is set
                grid = []
                for lr in lrs:
                    m, val = train(method, fit_img, fit_txt, cal_img, cal_txt, lr, ep, 0, device, batch=B, fit_partners=fit_p, cal_partners=cal_p, **kw)
                    grid.append({"lr": lr, "cal_loss": val})
                best = min(grid, key=lambda r: r["cal_loss"]); point = {"grid": grid, "selected_lr": best["lr"], "seeds": {}}
                for seed in seeds:
                    m, val = train(method, fit_img, fit_txt, cal_img, cal_txt, best["lr"], ep, seed, device, batch=B, fit_partners=fit_p, cal_partners=cal_p, **kw)
                    cal_s, cal_c, cal_y, _, _ = pair_sets(m, cal_img, cal_txt, device, 11, partners["SRC-CAL"][1]); th = threshold_at_fnr(cal_s[cal_y == 1], a.fnr)
                    r = {"cal_loss": val, "steps": (a.updates if axis == "fixed_updates" else a.exposure_epochs * (len(fit_img) // B))}
                    for split in ("SRC-EVAL", "TGT-EVAL"):
                        img, txt = splits[split]; s_, c_, y_, u, v = pair_sets(m, img, txt, device, 13, partners[split][1])
                        fnr = float((s_[y_ == 1] < th).float().mean()); fpr = float((s_[y_ == 0] >= th).float().mean())
                        ps, _, _ = platt(cal_s, cal_y, s_); e, mx, _ = ece(ps, y_)
                        r[split] = {"R1_R5": retrieval(u, v), "FNR": fnr, "FPR": fpr, "balanced_err": 0.5 * (fnr + fpr), "ECE_platt": e, "Brier_platt": brier(ps, y_)}
                        if method == "vcs":
                            r[split]["heldout_J"] = heldout_J(m, img, txt, device, 17, fixed_partner=partners[split][1])[0]
                    point["seeds"][str(seed)] = r
                results["axes"][axis][method][str(B)] = point
                mean = lambda sp, f: float(np.mean([r[sp][f] for r in point["seeds"].values()]))
                print(f"[{axis}] {method} B={B} lr={best['lr']:g}: SRC R@1 {np.mean([r['SRC-EVAL']['R1_R5'][0] for r in point['seeds'].values()]):.3f} bal.err {mean('SRC-EVAL', 'balanced_err'):.3f} | TGT R@1 {np.mean([r['TGT-EVAL']['R1_R5'][0] for r in point['seeds'].values()]):.3f} bal.err {mean('TGT-EVAL', 'balanced_err'):.3f} ({time.time() - t0:.0f}s)", flush=True)
    json.dump(results, open(a.out + ".json", "w"), indent=1, default=float)
    # markdown: per axis, metric vs batch with retention relative to B = 256
    L = [f"# Pre-check C — batch decoupling (pairing = {a.pairing}; frozen CLIP features; identity adapters) — {results['utc']}", "",
         f"Fixed updates: {a.updates} steps at every B; fixed exposure: {a.exposure_epochs} epochs at every B.  lr selected per (method, B, axis) on SRC-CAL from {lrs}; {len(seeds)} seeds. "
         "Balanced error at the SRC-CAL quantile threshold (FNR 5 %).  Retention = value(B) / value(256) for R@1 and for accuracy = 1 − balanced error.", ""]
    for axis, per_m in results["axes"].items():
        L += [f"## {axis}", "", "| method | B | steps | SRC R@1 (ret.) | SRC 1−bal.err (ret.) | TGT R@1 (ret.) | TGT 1−bal.err (ret.) | TGT ECE(Platt) | J (SRC/TGT) |", "|---|---|---|---|---|---|---|---|---|"]
        for method, per_b in per_m.items():
            ref = per_b.get("256") or per_b[max(per_b, key=int)]
            def val(pt, sp, f):
                return float(np.mean([r[sp][f] if f != "R1" else r[sp]["R1_R5"][0] for r in pt["seeds"].values()]))
            for B, pt in sorted(per_b.items(), key=lambda kv: int(kv[0])):
                cells = []
                for sp in ("SRC-EVAL", "TGT-EVAL"):
                    r1, r1r = val(pt, sp, "R1"), val(ref, sp, "R1"); acc, accr = 1 - val(pt, sp, "balanced_err"), 1 - val(ref, sp, "balanced_err")
                    cells += [f"{r1:.3f} ({r1 / max(r1r, 1e-9):.2f})", f"{acc:.3f} ({acc / max(accr, 1e-9):.2f})"]
                J = f"{val(pt, 'SRC-EVAL', 'heldout_J'):.3f}/{val(pt, 'TGT-EVAL', 'heldout_J'):.3f}" if method == "vcs" else "—"
                L.append(f"| {method} | {B} | {next(iter(pt['seeds'].values()))['steps']} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {val(pt, 'TGT-EVAL', 'ECE_platt'):.4f} | {J} |")
        L.append("")
    Path(a.out + ".md").write_text("\n".join(L) + "\n"); print("->", a.out + ".md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
