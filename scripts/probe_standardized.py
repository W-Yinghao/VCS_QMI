"""Evaluation-side diagnostic: re-run the frozen linear probe on *standardized* cached features (per-dimension z-score with fit-set
statistics) for every run that has a final evaluation.  Applied identically to all methods; nothing is retrained.

    python scripts/probe_standardized.py --output-root $OUTPUT_ROOT --out reports/PROBE_STANDARDIZED.md [--stages ...]

Uses the feature cache keys recorded in evaluations/evaluation_<tag>.json (fit_h / sel_hpz), the same probe hyper-parameters
(SGD 0.1/0.9/0, 100 epochs cosine->0.001x, seed 20260925, final probe epoch).  Writes evaluations/linear_standardized_<tag>.json per run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from vcs_ssl.config import load_resolved
from vcs_ssl.diagnostics import linear_probe
from vcs_ssl.utils import atomic_write_json, utc_now


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default="P3_pilot,P5_confirm200,P8_long800,P10_kstudy200,P12_vcs_hparamA,P14_vcs_hparamB_lr,P16_vcs_hparamB_set,P18_vcs_critic_variants,P20_vcs_ssl_wiring,P22_vcs_target_branch,P24_vcs_cosine_base")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    stages = set(a.stages.split(","))
    rows = []
    for rd in sorted(p for p in Path(a.output_root).iterdir() if p.is_dir()):
        st = rd / "status.json"
        if not st.is_file():
            continue
        stj = json.loads(st.read_text())
        if stj.get("stage") not in stages or stj.get("status") != "COMPLETED":
            continue
        evs = sorted((rd / "evaluations").glob("evaluation_epoch_*.json"))
        if not evs:
            continue
        ev = json.loads(evs[-1].read_text())
        tag = evs[-1].stem.replace("evaluation_", "")
        outp = rd / "evaluations" / f"linear_standardized_{tag}.json"
        if outp.is_file() and not a.force:
            res = json.loads(outp.read_text())
        else:
            fc = ev["feature_cache"]
            fit = torch.load(rd / "features" / f"{fc['fit_key']}.pt", weights_only=True)
            sel = torch.load(rd / "features" / f"{fc['sel_key']}.pt", weights_only=True)
            mu = fit["h"].mean(0, keepdim=True)
            sd = fit["h"].std(0, unbiased=True, keepdim=True).clamp_min(1e-6)
            hf, hs = (fit["h"] - mu) / sd, (sel["h"] - mu) / sd
            cfg = load_resolved(rd / "config.resolved.yaml")
            lp = linear_probe(hf, fit["labels"], hs, sel["labels"], cfg["evaluation"]["linear"], device=device)
            res = {"run_id": rd.name, "method": stj["method"], "stage": stj["stage"], "seed": stj["seed"], "checkpoint": ev["checkpoint"],
                   "standardization": "per-dim z-score with fit-set mean/std (ddof=1), applied to fit and selection", "h_norm_mean_raw": float(fit["h"].norm(dim=1).mean()),
                   "linear_val_top1_pct_standardized": lp["linear_val_top1_pct"], "linear_val_ce_standardized": lp["linear_val_ce"],
                   "curve": lp["curve"], "linear_val_top1_pct_raw": ev["linear_val_top1_pct"], "knn_val_top1_pct": ev["knn_val_top1_pct"], "utc": utc_now()}
            atomic_write_json(outp, res)
        rows.append(res)
        print(f"{rd.name:34s} raw {res['linear_val_top1_pct_raw']:.2f} -> std {res['linear_val_top1_pct_standardized']:.2f}  (kNN {res['knn_val_top1_pct']:.2f}, |h| {res['h_norm_mean_raw']:.1f})", flush=True)
    L = [f"# Standardized-feature linear probe (diagnostic) — {utc_now()}", "",
         "Same probe protocol on per-dimension standardized `h` (fit statistics). Same rule for every method; raw = frozen pilot protocol.", "",
         "| run | method | stage | ckpt | raw linear | standardized linear | Δ | kNN | mean ‖h‖ | curve (std probe, every 10 ep) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["stage"], r["run_id"])):
        L.append(f"| {r['run_id']} | {r['method']} | {r['stage']} | {r['checkpoint']} | {r['linear_val_top1_pct_raw']:.2f} | {r['linear_val_top1_pct_standardized']:.2f} | "
                 f"{r['linear_val_top1_pct_standardized'] - r['linear_val_top1_pct_raw']:+.2f} | {r['knn_val_top1_pct']:.2f} | {r['h_norm_mean_raw']:.1f} | "
                 f"{[round(c['val_top1_pct'], 1) for c in r['curve']]} |")
    Path(a.out).write_text("\n".join(L) + "\n")
    atomic_write_json(Path(a.out).with_suffix(".json"), rows)
    print(f"{len(rows)} runs -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
