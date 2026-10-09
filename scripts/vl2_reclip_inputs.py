"""VL2: ReCLIP input files for RefCOCO / RefCOCO+ (VL_DATASET), in the format of scripts/vl1_reclip_inputs.py (one JSON line per target object:
file_name COCO_train2014_<id>_<ann>.jpg, image_id, ann_id, anns, sentences).  Captions are MDETR's (REFER's lower-cased tokenised `sent` form;
no raw text is available for these datasets) — used as both `raw` and `sent`.
  <dataset>_val.jsonl          : UNC val (VAL_OFFICIAL), candidates = ALL COCO annotations of the image (ReCLIP's own protocol)
  <dataset>_dev_referred.jsonl : eligible DEV images (>= 2 referred), candidates = the referred objects only (Table A's primary set)
Also an image_root of COCO_train2014_<id>.jpg symlinks.  Writes under /projects/.../datasets/<dataset>_unc/.
    VL_DATASET=refcoco+ python scripts/vl2_reclip_inputs.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vcs_vl import refcocog as RG  # noqa: E402


def main() -> int:
    assert RG.DATASET in ("refcoco", "refcoco+"), "export VL_DATASET"
    inp = RG._unc_inputs(RG.DATASET); scenes, _ = RG.load_scenes(); role = RG.dev_roles(scenes)
    out_dir = RG.dataset_dir() / "reclip_inputs"; links = RG.dataset_dir() / "train2014_links"; out_dir.mkdir(parents=True, exist_ok=True); links.mkdir(exist_ok=True)
    tag = RG.DATASET.replace("+", "plus"); n = {"val": [0, 0], "dev_referred": [0, 0]}
    with open(out_dir / f"{tag}_val.jsonl", "w") as fv, open(out_dir / f"{tag}_dev_referred.jsonl", "w") as fd:
        for s in scenes:
            r = role[s.image_id]
            if r == "VAL_OFFICIAL":
                cand, fh, k = sorted(inp["anns"][s.image_id], key=lambda a: a["id"]), fv, "val"
            elif r == "DEV" and len(s.referred) >= 2:
                cand, fh, k = [{"id": o.ann_id, "bbox": list(o.box_xywh), "category_id": o.category_id, "iscrowd": o.iscrowd, "area": o.area} for o in s.referred], fd, "dev_referred"
            else:
                continue
            for o in s.referred:
                rec = {"file_name": f"COCO_train2014_{s.image_id:012d}_{o.ann_id}.jpg", "image_id": s.image_id, "ann_id": o.ann_id, "category_id": o.category_id,
                       "split": s.umd_split, "anns": cand, "sentences": [{"raw": e["raw"], "sent": e["raw"], "sent_id": e["sent_id"]} for e in o.expressions]}
                fh.write(json.dumps(rec) + "\n"); n[k][0] += 1; n[k][1] += len(o.expressions)
            dst = links / f"COCO_train2014_{s.image_id:012d}.jpg"
            if not dst.exists():
                os.symlink(s.path, dst)
    print({k: {"refs": v[0], "sentences": v[1]} for k, v in n.items()}, "links", len(os.listdir(links))); return 0


if __name__ == "__main__":
    sys.exit(main())
