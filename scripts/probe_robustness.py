"""P98 — linear-probe robustness of the P91 CIFAR-100 readings (evaluation only; no training, selection split only, official test never opened).

    python scripts/probe_robustness.py --out reports/P98_probe_robustness [--runs R1,R2,...] [--smoke] [--cpu]

For every checkpoint the frozen-h features are read from the run's own feature cache (the keys of `vcs_ssl.evaluate` protocol 'pilot'; a
cache miss re-extracts with the same function and seeds) and three probes are fitted on the FIT features and scored on the selection split:
  frozen  the pilot probe exactly (recipe hyper-parameters; reproduces the stored evaluation_<ckpt>.json value — checked);
  l2      the same probe on L2-normalised h (fit and selection normalised row-wise; nothing else changes);
  tuned   the recipe probe with 300 epochs and lr in {0.1, 0.3, 1.0} chosen on a seeded held-out 10 % of the FIT features (a probe trained on
          the other 90 %); the chosen lr is then refitted on all FIT features and scored once on the selection split.  The selection split
          never enters the lr choice.
Per probe: selection top-1, selection CE, final train CE.  Aggregates: P91 cells (method x budget) mean / sd over 3 seeds, the P92 readings
(a) ordering and (b) scaling re-read under each probe, and the pre-stated robustness verdict (sign of every |diff| >= 1.0 pairwise difference
identical under all three probes); the CIFAR-10 8x cells are reported as a reference.
"""
from __future__ import annotations

import argparse
import copy
import json
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vcs_ssl.checkpoint import load_checkpoint  # noqa: E402
from vcs_ssl.config import load_resolved  # noqa: E402
from vcs_ssl.data.cifar import load_train_partition  # noqa: E402
from vcs_ssl.data.splits import load_manifest  # noqa: E402
from vcs_ssl.data.transforms import build_clean_transform, clean_transform_signature  # noqa: E402
from vcs_ssl.diagnostics import extract_features, linear_probe  # noqa: E402
from vcs_ssl.evaluate import cached_features, feature_cache_key  # noqa: E402
from vcs_ssl.models import build_models  # noqa: E402
from vcs_ssl.utils import sha256_file, utc_now  # noqa: E402

O = Path("/home/infres/yinwang/CS_QMI/outputs")
P91 = [(m, b, s) for m in ("vcs_a5", "simclr", "vicreg") for b in ("200ep", "800ep") for s in range(3)]
C10 = [("P35_vcs_a5_views4_800ep_seed%d" % s, "epoch_800.pt") for s in range(3)] + \
      [(f"P41_{m}_views4_800ep_seed{s}", "epoch_800.pt") for m in ("simclr", "vicreg") for s in range(3)]
LRS = (0.1, 0.3, 1.0)
HOLDOUT_FRAC, HOLDOUT_SEED = 0.10, 20260930
TUNED_EPOCHS = 300


def default_runs():
    out = [(f"P91_c100_{m}_views4_{b}_seed{s}", "epoch_200.pt" if b == "200ep" else "epoch_800.pt") for m, b, s in P91]
    return out + C10


def features(run_dir: Path, ckpt: str, device, workers: int):
    cfg = load_resolved(run_dir / "config.resolved.yaml")
    manifest = load_manifest(run_dir / "manifest.json", expected_seed=cfg["data"]["split_seed"], expected_val_per_class=cfg["data"]["val_per_class"])
    ckpt_path = run_dir / "checkpoints" / ckpt; ckpt_sha = sha256_file(ckpt_path)
    fit_uids = np.asarray(manifest["fit_uids"], dtype=np.int64); sel_uids = np.asarray(manifest["selection_uids"], dtype=np.int64)
    tsig = clean_transform_signature(cfg["views"]); dtype = cfg["evaluation"]["linear"]["feature_cache_dtype"]
    k_fit = feature_cache_key(ckpt_sha=ckpt_sha, split_sha=manifest["manifest_sha256"], transform_sha=tsig, dtype=dtype, uids=fit_uids, which="fit_h")
    k_sel = feature_cache_key(ckpt_sha=ckpt_sha, split_sha=manifest["manifest_sha256"], transform_sha=tsig, dtype=dtype, uids=sel_uids, which="sel_hpz")
    lazy = {}

    def models():
        if not lazy:
            data = load_train_partition(cfg["data"]["name"], cfg["data"]["root"])
            ck = load_checkpoint(ckpt_path); torch.manual_seed(0)
            built = build_models(cfg, seed=int(cfg["run"]["seed"]), device=device)
            built["encoder"].load_state_dict(ck["encoder_state"]); built["projector"].load_state_dict(ck["projector_state"])
            for m in (built["encoder"], built["projector"]):
                m.eval(); [p.requires_grad_(False) for p in m.parameters()]
            lazy.update(data=data, enc=built["encoder"], proj=built["projector"])
        return lazy
    clean = build_clean_transform(cfg["views"])
    fit = cached_features(run_dir / "features", k_fit, lambda: extract_features(models()["enc"], None, models()["data"].data, models()["data"].targets,
                                                                                 fit_uids, clean, device=device, num_workers=workers, seed=11))
    sel = cached_features(run_dir / "features", k_sel, lambda: extract_features(models()["enc"], models()["proj"], models()["data"].data, models()["data"].targets,
                                                                                 sel_uids, clean, device=device, num_workers=workers, seed=12,
                                                                                 l2_eps=cfg["model"]["normalization"]["eps"]))
    return cfg, fit, sel, {"fit": fit["_cache"], "sel": sel["_cache"], "ckpt_sha256": ckpt_sha}


def holdout_split(n: int):
    """Seeded FIT-internal split (indices into the FIT feature rows); the selection split is never touched."""
    perm = np.random.default_rng(HOLDOUT_SEED).permutation(n); k = int(round(HOLDOUT_FRAC * n))
    return np.sort(perm[k:]), np.sort(perm[:k])


def probe_all(cfg, fit, sel, device, smoke=False):
    lc = dict(cfg["evaluation"]["linear"]); n_classes = int(fit["labels"].max()) + 1
    if smoke:
        lc["epochs"] = 2
    hf, yf, hs, ys = fit["h"].float(), fit["labels"], sel["h"].float(), sel["labels"]
    out = {}
    r = linear_probe(hf, yf, hs, ys, lc, device=device, n_classes=n_classes)
    out["frozen"] = {"top1": r["linear_val_top1_pct"], "val_ce": r["linear_val_ce"], "train_ce": r["final_train_ce"]}
    nf, ns = torch.nn.functional.normalize(hf, dim=1), torch.nn.functional.normalize(hs, dim=1)
    r = linear_probe(nf, yf, ns, ys, lc, device=device, n_classes=n_classes)
    out["l2"] = {"top1": r["linear_val_top1_pct"], "val_ce": r["linear_val_ce"], "train_ce": r["final_train_ce"]}
    tr, ho = holdout_split(len(hf)); lt = dict(lc); lt["epochs"] = 2 if smoke else TUNED_EPOCHS; scan = {}
    for lr in LRS:
        lt["lr"] = lr
        r = linear_probe(hf[tr], yf[tr], hf[ho], yf[ho], lt, device=device, n_classes=n_classes)
        scan[str(lr)] = {"holdout_top1": r["linear_val_top1_pct"], "holdout_ce": r["linear_val_ce"], "train_ce": r["final_train_ce"]}
    best = max(LRS, key=lambda lr: (scan[str(lr)]["holdout_top1"], -lr))        # ties -> smaller lr
    lt["lr"] = best
    r = linear_probe(hf, yf, hs, ys, lt, device=device, n_classes=n_classes)
    out["tuned"] = {"top1": r["linear_val_top1_pct"], "val_ce": r["linear_val_ce"], "train_ce": r["final_train_ce"], "lr": best, "epochs": lt["epochs"],
                    "lr_scan_on_fit_holdout": scan, "holdout": {"frac": HOLDOUT_FRAC, "seed": HOLDOUT_SEED, "n_train": int(len(tr)), "n_holdout": int(len(ho))}}
    return out


def aggregate(rows: dict) -> dict:
    agg = {}
    for probe in ("frozen", "l2", "tuned"):
        cells = {}
        for m, b, s in P91:
            rid = f"P91_c100_{m}_views4_{b}_seed{s}"
            if rid in rows:
                cells.setdefault((m, b), []).append(rows[rid]["probes"][probe]["top1"])
        mean = {f"{m}|{b}": (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0, len(v)) for (m, b), v in cells.items()}
        g = lambda m, b: mean.get(f"{m}|{b}", (float("nan"),))[0]
        diffs = {}
        for b in ("200ep", "800ep"):
            diffs[f"simclr-vcs|{b}"] = g("simclr", b) - g("vcs_a5", b); diffs[f"vcs-vicreg|{b}"] = g("vcs_a5", b) - g("vicreg", b)
            diffs[f"simclr-vicreg|{b}"] = g("simclr", b) - g("vicreg", b)
        gains = {m: g(m, "800ep") - g(m, "200ep") for m in ("vcs_a5", "simclr", "vicreg")}
        diffs["gain_vcs-gain_simclr"] = gains["vcs_a5"] - gains["simclr"]; diffs["gain_vcs-gain_vicreg"] = gains["vcs_a5"] - gains["vicreg"]
        agg[probe] = {"cells": mean, "pairwise": diffs, "gains": gains,
                      "c10_reference": {rid: rows[rid]["probes"][probe]["top1"] for rid, _ in C10 if rid in rows}}
    keys = list(agg["frozen"]["pairwise"])
    verdict = {}
    for k in keys:
        vals = [agg[p]["pairwise"][k] for p in ("frozen", "l2", "tuned")]
        big = [v for v in vals if abs(v) >= 1.0]
        verdict[k] = {"values": vals, "robust": all(np.sign(v) == np.sign(big[0]) for v in big) if big else True,
                      "note": "robust = every |diff| >= 1.0 has the same sign under all three probes (|diff| < 1.0 counts as a tie, compatible)"}
    agg["robustness"] = verdict; agg["all_robust"] = all(v["robust"] for v in verdict.values())
    return agg


def markdown(R) -> str:
    A = R["aggregate"]; L = [f"# P98 — linear-probe robustness of the P91 CIFAR-100 readings (results only; {R['utc']})", "",
                              "Probes: frozen (recipe), l2 (row-normalised h), tuned (300 epochs, lr from {0.1, 0.3, 1.0} on a 10 % FIT hold-out).  Selection split only.", "",
                              "## Per-checkpoint (selection top-1 / probe train CE)", "", "| run | frozen | l2 | tuned (lr) | stored pilot value |", "|---|---|---|---|---|"]
    for rid, r in R["rows"].items():
        p = r["probes"]
        L.append(f"| {rid} | {p['frozen']['top1']:.2f} / {p['frozen']['train_ce']:.3f} | {p['l2']['top1']:.2f} / {p['l2']['train_ce']:.3f} | "
                 f"{p['tuned']['top1']:.2f} / {p['tuned']['train_ce']:.3f} ({p['tuned']['lr']}) | {r.get('stored_pilot_top1')} |")
    L += ["", "## P91 cells (mean ± sd, 3 seeds)", "", "| probe | " + " | ".join(sorted(A["frozen"]["cells"])) + " |", "|---|" + "---|" * len(A["frozen"]["cells"])]
    for probe in ("frozen", "l2", "tuned"):
        c = A[probe]["cells"]; L.append(f"| {probe} | " + " | ".join(f"{c[k][0]:.2f} ± {c[k][1]:.2f}" for k in sorted(c)) + " |")
    L += ["", "## P92 readings re-read (pairwise differences; gains 2x -> 8x)", "", "| quantity | frozen | l2 | tuned | robust |", "|---|---|---|---|---|"]
    for k, v in A["robustness"].items():
        L.append(f"| {k} | " + " | ".join(f"{x:+.2f}" for x in v["values"]) + f" | {v['robust']} |")
    L += ["", f"**All pairwise signs robust across probes:** {A['all_robust']}"]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--runs", default=None)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--cpu", action="store_true"); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    device = torch.device("cuda", 0) if torch.cuda.is_available() and not a.cpu else torch.device("cpu")
    runs = [tuple(x.split(":")) for x in a.runs.split(",")] if a.runs else default_runs()
    R = {"utc": utc_now(), "device": str(device), "smoke": a.smoke, "probe_definitions": {"tuned_epochs": TUNED_EPOCHS, "lrs": LRS, "holdout_frac": HOLDOUT_FRAC,
         "holdout_seed": HOLDOUT_SEED}, "official_test_opened": False, "rows": {}}
    for rid, ck in runs:
        t0 = time.time(); rd = O / rid
        cfg, fit, sel, cache = features(rd, ck, device, a.workers)
        if a.smoke:
            fit = {k: (v[:2000] if isinstance(v, torch.Tensor) else v) for k, v in fit.items()}
            sel = {k: (v[:500] if isinstance(v, torch.Tensor) else v) for k, v in sel.items()}
        stored = None; ef = rd / "evaluations" / f"evaluation_{Path(ck).stem}.json"
        if ef.is_file():
            stored = json.load(open(ef)).get("linear_val_top1_pct")
        probes = probe_all(cfg, fit, sel, device, smoke=a.smoke)
        R["rows"][rid] = {"checkpoint": ck, "cache": cache, "stored_pilot_top1": stored, "probes": probes, "seconds": time.time() - t0,
                          "frozen_reproduces_stored": (None if (stored is None or a.smoke) else abs(probes["frozen"]["top1"] - stored) < 0.05)}
        print(f"{rid}: frozen {probes['frozen']['top1']:.2f} (stored {stored}) l2 {probes['l2']['top1']:.2f} tuned {probes['tuned']['top1']:.2f} "
              f"lr {probes['tuned']['lr']} ({time.time() - t0:.0f}s)", flush=True)
    R["aggregate"] = aggregate(R["rows"])
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(R, indent=1, default=str)); out.with_suffix(".md").write_text(markdown(R))
    print("wrote", out.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
