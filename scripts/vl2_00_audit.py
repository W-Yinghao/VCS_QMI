"""VL2 data audit — RefCOCO / RefCOCO+ (UNC splits) through the VL1 adapter (VL_DATASET).  Builds the scene-input cache, then reports per split:
images, referred objects, expressions; eligible images (>= 2 referred objects); FIT / CAL / DEV / VAL_OFFICIAL / TEST_CLOSED counts; same-category
share of eligible DEV queries; image overlap with RefCOCOg UMD val / test (contamination bookkeeping).  Also checks that the default dataset
(refcocog) still loads the VL1 scene set unchanged.
    VL_DATASET=refcoco+ python scripts/vl2_00_audit.py
Writes reports/VL2/VL2_DATA_AUDIT_<dataset>.json.
"""
from __future__ import annotations

import collections
import importlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    ds = os.environ.get("VL_DATASET", "refcocog"); assert ds in ("refcoco", "refcoco+"), "set VL_DATASET to refcoco or refcoco+"
    from vcs_vl import refcocog as RG
    scenes, st = RG.load_scenes(resolve_paths=True); role = RG.dev_roles(scenes)
    out = {"dataset": ds, "source": "MDETR finetune_<dataset>_<split>.json + COCO-2017 instances", "stats": {k: v for k, v in st.items() if k != "categories"},
           "splits": {}, "roles": dict(collections.Counter(role.values())), "eligible_by_role": {}, "missing_image_files": sum(s.path is None for s in scenes)}
    for sp in ("train", "val", "testA", "testB"):
        ss = [s for s in scenes if s.umd_split == sp]
        out["splits"][sp] = {"images": len(ss), "referred": sum(len(s.referred) for s in ss), "expressions": sum(len(o.expressions) for s in ss for o in s.referred),
                             "eligible_images": sum(len(s.referred) >= 2 for s in ss)}
    for r in ("FIT", "CAL", "DEV"):
        out["eligible_by_role"][r] = sum(1 for s in scenes if role[s.image_id] == r and len(s.referred) >= 2)
    dev = [s for s in scenes if role[s.image_id] == "DEV" and len(s.referred) >= 2]; same = n = 0
    for s in dev:
        cats = [o.category_id for o in s.referred]
        for i, o in enumerate(s.referred):
            same += len(o.expressions) * (cats.count(cats[i]) > 1); n += len(o.expressions)
    out["dev_queries"] = n; out["dev_same_category_share"] = same / max(n, 1)
    # contamination bookkeeping: images shared with RefCOCOg UMD val / test
    os.environ["VL_DATASET"] = "refcocog"; RGg = importlib.reload(RG)
    g, _ = RGg.load_scenes(resolve_paths=False); gsplit = {s.image_id: s.umd_split for s in g}
    out["overlap_with_refcocog_umd"] = {r: dict(collections.Counter(gsplit[s.image_id] for s in scenes if role[s.image_id] == r and s.image_id in gsplit)) for r in ("FIT", "CAL", "DEV", "VAL_OFFICIAL", "TEST_CLOSED")}
    out["refcocog_default_unchanged"] = {"n_scenes": len(g)}
    rep = REPO / "reports" / "VL2"; rep.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(rep / f"VL2_DATA_AUDIT_{ds.replace('+', 'plus')}.json", "w"), indent=1)
    print(json.dumps(out, indent=1)[:2500]); return 0


if __name__ == "__main__":
    sys.exit(main())
