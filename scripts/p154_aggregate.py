"""P154 ImageNet-100 pilot, seed 0 (frozen reading, P154_IN100_PILOT_PREREG_FROZEN_20261008.md §4): linear (primary) and kNN top-1 on the 5 000
development images at epochs 50 / 100 / 200 per method; Δ = VCS − SimCLR and VCS − JS at epoch 200 (linear; kNN alongside); trigger for seeds
1–2 of all three methods: |Δ| >= 1.0 for either contrast, in either direction; otherwise "single seed, within ±1.0".
Writes reports/P154_results.json; exits 1 if an epoch-200 evaluation is missing (no partial reading)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

O = Path("/home/infres/yinwang/CS_QMI/outputs"); REP = Path(__file__).resolve().parents[1] / "reports"
RUN = "P154_IN100_{m}_r18_200ep_seed0"; METHODS = ("vcs", "js", "simclr")


def ev(m, e):
    f = O / RUN.format(m=m) / "evaluations" / f"evaluation_epoch_{e}.json"
    if not f.exists():
        return None
    d = json.load(open(f)); return {"linear": d["linear_val_top1_pct"], "knn": d["knn_val_top1_pct"], "n_fit": d["n_fit"], "n_val": d["n_val"], "official_val_used": d["official_val_used"]}


def main() -> int:
    out = {"per_method": {m: {e: ev(m, e) for e in (50, 100, 200)} for m in METHODS}, "contrasts": {}, "trigger": None}
    missing = [m for m in METHODS if out["per_method"][m][200] is None]
    if missing:
        print("missing epoch-200 evaluations:", missing); return 1
    assert not any(out["per_method"][m][200]["official_val_used"] for m in METHODS)
    for other in ("simclr", "js"):
        a, b = out["per_method"]["vcs"][200], out["per_method"][other][200]
        out["contrasts"][f"vcs_minus_{other}"] = {"linear": a["linear"] - b["linear"], "knn": a["knn"] - b["knn"]}
    fires = [k for k, v in out["contrasts"].items() if abs(v["linear"]) >= 1.0]
    out["trigger"] = {"fires": bool(fires), "contrasts": fires, "rule": "|Δ linear| >= 1.0 for either contrast -> seeds 1-2 for all three methods"}
    json.dump(out, open(REP / "P154_results.json", "w"), indent=1)
    for m in METHODS:
        print(m, {e: (None if v is None else (round(v["linear"], 2), round(v["knn"], 2))) for e, v in out["per_method"][m].items()})
    for k, v in out["contrasts"].items():
        print(f"{k}: linear {v['linear']:+.2f}  kNN {v['knn']:+.2f}")
    print("trigger:", "FIRES " + ", ".join(fires) if fires else "no (single seed, within ±1.0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
