"""VL2 Tables B / C summary for RefCOCO and RefCOCO+: ReCLIP (official ensemble + relations, and IPS only) on UNC val (ReCLIP's candidate set: all
COCO objects) and on Table A's eligible DEV images with the referred-object candidates (exact target and IoU >= 0.5; query and image-macro), the
raw zero-shot rows of the five feature sets (Table A raw), and the Table C rows (Grounding DINO val / DEV, MDETR val).  Missing runs are listed.
Writes reports/VL2/VL2_tableBC.json.
    python scripts/vl2_tableB.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vl1_tableB_aggregate import score  # noqa: E402

O = Path("/home/infres/yinwang/CS_QMI/outputs"); REP = Path(__file__).resolve().parents[1] / "reports"; OUT = REP / "VL2" / "VL2_tableBC.json"
DATASETS, FEATS = ("refcoco", "refcocoplus"), ("clip", "siglip2", "fgclip2", "fgclip1", "reclip")


def main() -> int:
    res = {"reclip": {}, "raw_feature_rows": {}, "table_c": {}, "missing": []}
    for ds in DATASETS:
        for tag in ("parse", "baseline"):
            for sp in ("val", "dev_referred"):
                f = O / "VL2_reclip" / f"reclip_{ds}_{tag}_{sp}.json"
                if not f.exists():
                    res["missing"].append(f.name); continue
                rows = [json.loads(l) for l in open(f) if l.strip()]
                res["reclip"][f"{ds}/{tag}/{sp}"] = score(rows)
        for ft in FEATS:
            f = O / f"VL2_{ds}_{ft}" / "raw.json"
            if f.exists():
                r = json.load(open(f))["dev"]; res["raw_feature_rows"][f"{ds}/{ft}"] = {k: r[k] for k in ("top1_query", "top1_image_macro", "top1_all_objects", "n_images", "n_queries")}
            else:
                res["missing"].append(f"{ds}/{ft}/raw")
        for name in (f"gdino_tiny_{ds}_val", f"gdino_tiny_{ds}_dev", f"mdetr_r101_{ds}_val"):
            f = REP / "VL1" / f"VL1_20_{name}.json"
            if f.exists():
                d = json.load(open(f)); res["table_c"][name] = {k: d.get(k) for k in ("n", "acc_iou50", "acc_giou50", "acc_iou50_image_macro", "reference")}
            else:
                res["missing"].append(name)
    json.dump(res, open(OUT, "w"), indent=1)
    for k, v in res["reclip"].items():
        print(f"ReCLIP {k:32s} n {v['n_queries']:5d} query exact {100 * v['top1_query_exact']:.2f} IoU {100 * v['top1_query_iou']:.2f} | macro exact {100 * v['top1_image_macro_exact']:.2f} IoU {100 * v['top1_image_macro_iou']:.2f}")
    for k, v in res["raw_feature_rows"].items():
        print(f"raw {k:22s} macro {100 * v['top1_image_macro']:.2f} query {100 * v['top1_query']:.2f} all-obj {100 * v['top1_all_objects']:.2f} n {v['n_queries']}")
    for k, v in res["table_c"].items():
        print(f"Table C {k:28s} Acc@IoU0.5 {100 * v['acc_iou50']:.2f} (GIoU {100 * v['acc_giou50']:.2f}) n {v['n']} ref {v['reference']}")
    if res["missing"]:
        print("missing:", res["missing"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
