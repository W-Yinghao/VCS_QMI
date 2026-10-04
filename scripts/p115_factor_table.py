"""P115 2x2 augmentation-factor table (frozen reading: main effects, interaction, method contrast A-P3 - SimCLR); seed 0, descriptive."""
import json
from pathlib import Path
O = Path("/home/infres/yinwang/CS_QMI/outputs")
RUNS = {"AP3": {"std": "P107_AP3_views4_800ep_seed0", "crop": "P115_AP3_croponly_views4_800ep_seed0", "jit": "P115_AP3_jitteronly_views4_800ep_seed0", "both": "P107_AP3_augstrong_views4_800ep_seed0"},
        "SimCLR": {"std": "P41_simclr_views4_800ep_seed0", "crop": "P115_simclr_croponly_views4_800ep_seed0", "jit": "P115_simclr_jitteronly_views4_800ep_seed0", "both": "P89_simclr_views4_800ep_augstrong_seed0"}}
res = {}
for m, cells in RUNS.items():
    for metric, key in (("linear", "linear_val_top1_pct"), ("knn", "knn_val_top1_pct")):
        v = {c: float(json.load(open(O / r / "evaluations" / "evaluation_epoch_800.json"))[key]) for c, r in cells.items()}
        res[(m, metric)] = dict(v, crop_main=(v["crop"] + v["both"]) / 2 - (v["std"] + v["jit"]) / 2, jit_main=(v["jit"] + v["both"]) / 2 - (v["std"] + v["crop"]) / 2,
                                inter=v["both"] - v["crop"] - v["jit"] + v["std"], crop_single=v["crop"] - v["std"], jit_single=v["jit"] - v["std"], both_total=v["both"] - v["std"])
out = {}
for metric in ("linear", "knn"):
    a, s = res[("AP3", metric)], res[("SimCLR", metric)]
    for m, r in (("AP3", a), ("SimCLR", s)):
        print(f"{metric:6s} {m:6s} std {r['std']:.2f} crop {r['crop']:.2f} jit {r['jit']:.2f} both {r['both']:.2f} | crop main {r['crop_main']:+.2f} jit main {r['jit_main']:+.2f} inter {r['inter']:+.2f} | single crop {r['crop_single']:+.2f} jit {r['jit_single']:+.2f} both {r['both_total']:+.2f}")
    print(f"{metric:6s} contrast A-P3 - SimCLR: crop {a['crop_main']-s['crop_main']:+.2f} jitter {a['jit_main']-s['jit_main']:+.2f} interaction {a['inter']-s['inter']:+.2f}")
    out[metric] = {"AP3": a, "SimCLR": s}
json.dump(out, open("reports/P115_factor_table.json", "w"), indent=1)
