"""R6 REG-P1 readings (reports/R6/R6_REG_P1_PROTOCOL_*_20261011.yaml) over outputs/R6_REG/results/*.json (scripts/r6_reg_pilot.py register).

Per method: DEV image-macro success at < 42 px and < 9 px full-resolution error (per image over its 10 perturbations, then the mean over images),
median and 90th-percentile error, seconds per registration; mean ± sd over seeds for seeded methods (vcs, logistic, infonce, comir), deterministic
rows for mi / nmi / kernel.  Contrasts VCS − X (X ∈ mi, nmi, kernel, logistic, infonce, comir) on success < 42 px (primary: X = mi and X = comir):
paired by seed (seeded X) or VCS seed against the deterministic X; 95 % t over the 3 VCS seeds and an image bootstrap (2 000 resamples, image
indices shared across methods).  Wording: interval containing 0 -> no detectable difference; above / below 0 -> VCS registers x more / fewer.
    python scripts/r6_reg_aggregate.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path("/home/infres/yinwang/CS_QMI/outputs/R6_REG"); OUT = Path(__file__).resolve().parents[1] / "reports" / "R6"
SEEDED, DETERMINISTIC, SEEDS, THR = ("vcs", "logistic", "infonce", "comir"), ("mi", "nmi", "kernel"), (0, 1, 2), (42.0, 9.0)


def per_image(rec, thr):
    return {k: float(np.mean([r["err_full"] < thr for r in v])) for k, v in rec.items() if "#" not in k}


def tint(x):
    x = np.asarray(x, float); m = float(x.mean()); n = len(x)
    if n < 2:
        return {"mean": m, "sd": None, "ci95": None, "n": n}
    s = float(x.std(ddof=1)); h = stats.t.ppf(0.975, n - 1) * s / math.sqrt(n); return {"mean": m, "sd": s, "ci95": [m - h, m + h], "n": n}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--contrast-thr", type=float, default=42.0, help="secondary: contrasts at another success threshold (written with suffix _thr<N>)"); a = ap.parse_args()
    rd = (ROOT / "smoke" if a.smoke else ROOT) / "results"; sfx = "_smoke" if a.smoke else ""; osfx = sfx + ("" if a.contrast_thr == 42.0 else f"_thr{int(a.contrast_thr)}"); CT = a.contrast_thr
    runs, missing = {}, []
    for m in SEEDED:
        for s in SEEDS:
            f = rd / f"{m}_s{s}{sfx}.json"
            if f.exists():
                runs[(m, s)] = json.load(open(f))
            else:
                missing.append(f.name)
    for m in DETERMINISTIC:
        f = rd / (f"{m}_s0{sfx}.json" if m == "kernel" else f"{m}{sfx}.json")
        if f.exists():
            runs[(m, None)] = json.load(open(f))
        else:
            missing.append(f.name)
    res = {"protocol": "reports/R6/R6_REG_P1_PROTOCOL_FROZEN_20261011.yaml", "methods": {}, "contrasts": {}, "missing": missing}
    if CT != 42.0:
        res["contrast_threshold_px"] = CT; res["note"] = "secondary analysis: the frozen primary contrasts are at < 42 px"
    for m in SEEDED + DETERMINISTIC:
        keys = [k for k in runs if k[0] == m]
        if not keys:
            continue
        row = {"seeds": [k[1] for k in keys]}
        for thr in THR:
            row[f"success{int(thr)}"] = tint([float(np.mean(list(per_image(runs[k]["dev"], thr).values()))) for k in keys])
        for f in ("dev_median_err", "dev_p90_err", "dev_sec_per_registration"):
            row[f] = tint([runs[k][f] for k in keys if runs[k].get(f) is not None])
        if m in ("vcs", "logistic", "infonce"):
            row["chosen_ckpt"] = [runs[k].get("chosen_ckpt") for k in keys]
        if m == "kernel":
            row["chosen_mult"] = runs[keys[0]].get("chosen_mult")
        res["methods"][m] = row
    vk = sorted(k for k in runs if k[0] == "vcs"); g = np.random.default_rng(20261011)
    for other in ("mi", "nmi", "kernel", "logistic", "infonce", "comir"):
        ok = [k for k in runs if k[0] == other]
        if not vk or not ok:
            continue
        pairs = [(v, next((o for o in ok if o[1] == v[1]), None) if other in SEEDED else ok[0]) for v in vk]; pairs = [(v, o) for v, o in pairs if o is not None]
        imgs = sorted(set.intersection(*[set(per_image(runs[v]["dev"], CT)) & set(per_image(runs[o]["dev"], CT)) for v, o in pairs]))
        D = np.array([[per_image(runs[v]["dev"], CT)[i] - per_image(runs[o]["dev"], CT)[i] for i in imgs] for v, o in pairs])   # [pairs, images]
        boot = [float(D[:, g.integers(0, len(imgs), len(imgs))].mean()) for _ in range(2000)] if len(imgs) else []
        res["contrasts"][f"vcs_minus_{other}"] = {**tint(D.mean(1)), "image_boot95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))] if boot else None,
                                                  "n_images": len(imgs), "pairs": [[v[1], o[1]] for v, o in pairs]}
    OUT.mkdir(parents=True, exist_ok=True); json.dump(res, open(OUT / f"R6_REG_P1_results{osfx}.json", "w"), indent=1)
    p = lambda t: "—" if not t or t.get("mean") is None else f"{100 * t['mean']:.1f}" + (f" ± {100 * t['sd']:.1f}" if t.get("sd") is not None else "")
    q = lambda t: f"{100 * t['mean']:+.1f}" + (f" [{100 * t['ci95'][0]:+.1f}, {100 * t['ci95'][1]:+.1f}]" if t.get("ci95") else "") + (f"; image bootstrap [{100 * t['image_boot95'][0]:+.1f}, {100 * t['image_boot95'][1]:+.1f}]" if t.get("image_boot95") else "")
    L = [f"# R6 REG-P1 — BF/SHG translation-only registration: DEV results{' (SMOKE)' if a.smoke else ''}{'' if CT == 42.0 else f' — contrasts at < {CT:g} px (secondary)'}", "", f"Missing: {missing or 'none'}.", "",
         "| method | seeds | success < 42 px (%) | success < 9 px (%) | median error (px) | p90 error (px) | s / registration |", "|---|---|---|---|---|---|---|"]
    for m, r in res["methods"].items():
        L.append(f"| {m} | {r['seeds']} | {p(r['success42'])} | {p(r['success9'])} | {r['dev_median_err']['mean']:.1f} | {r['dev_p90_err']['mean']:.1f} | {r['dev_sec_per_registration']['mean']:.2f} |")
    L += ["", f"| contrast (success < {CT:g} px, image-macro){'' if CT == 42.0 else ' — secondary analysis'} | value |", "|---|---|"] + [f"| {k} | {q(v)} |" for k, v in res["contrasts"].items()]
    (OUT / f"R6_REG_P1_tables{osfx}.md").write_text("\n".join(L) + "\n"); print("\n".join(L)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
