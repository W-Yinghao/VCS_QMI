"""Rebuild ReCLIP's RefCOCOg input files (the official preprocessed data, gs://reclip-sanjays/reclip_data.tar.gz, is no longer accessible and
the generation code was never released — allenai/reclip issues #7 / #9-#12).  Format, read from the official main.py: one JSON line per REFER ref
with "file_name" (REFER style COCO_train2014_<id>_<ann>.jpg; main.py strips the ann suffix), "image_id", "ann_id", "anns" (ALL COCO annotations of
the image = the ground-truth-box candidate set, with "id" and "bbox" xywh), "sentences" ([{"raw", ...}]).  Also builds an image_root of
COCO_train2014_<id>.jpg symlinks onto the local COCO-2017 files (same image ids).
    python scripts/vl1_reclip_inputs.py
Writes /projects/EEG-foundation-model/yinghao/datasets/refcocog_umd/reclip_inputs/refcocog_umd_{val,test,dev}.jsonl and .../train2014_links/.
The "dev" file = the DEV role of the UMD train split (our development role); val / test = the UMD splits.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vcs_vl import refcocog as RG  # noqa: E402

OUT = Path("/projects/EEG-foundation-model/yinghao/datasets/refcocog_umd/reclip_inputs")
LINKS = Path("/projects/EEG-foundation-model/yinghao/datasets/refcocog_umd/train2014_links")


def main() -> int:
    refs = pickle.load(open(RG.ROOT / "refs(umd).p", "rb")); inst = json.load(open(RG.ROOT / "instances.json"))
    anns = {}
    for a in inst["annotations"]:
        anns.setdefault(a["image_id"], []).append({k: a[k] for k in ("id", "bbox", "category_id", "iscrowd", "area")})
    scenes, _ = RG.load_scenes(resolve_paths=False); role = RG.dev_roles(scenes)
    OUT.mkdir(parents=True, exist_ok=True); LINKS.mkdir(parents=True, exist_ok=True)
    files = {k: open(OUT / f"refcocog_umd_{k}.jsonl", "w") for k in ("val", "test", "dev")}; counts = {k: [0, 0] for k in files}
    for r in refs:
        k = {"val": "val", "test": "test"}.get(r["split"]) or ("dev" if role.get(r["image_id"]) == "DEV" else None)
        if k is None:
            continue
        rec = {"file_name": r["file_name"], "image_id": r["image_id"], "ann_id": r["ann_id"], "ref_id": r["ref_id"], "category_id": r["category_id"],
               "split": r["split"], "anns": sorted(anns[r["image_id"]], key=lambda a: a["id"]),
               "sentences": [{"raw": s["raw"], "sent": s["sent"], "sent_id": s["sent_id"]} for s in r["sentences"]]}
        files[k].write(json.dumps(rec) + "\n"); counts[k][0] += 1; counts[k][1] += len(rec["sentences"])
    for f in files.values():
        f.close()
    made = 0
    for iid in {r["image_id"] for r in refs}:
        dst = LINKS / f"COCO_train2014_{iid:012d}.jpg"
        if not dst.exists():
            src = RG.coco_path(iid)
            if src is None:
                raise FileNotFoundError(iid)
            os.symlink(src, dst); made += 1
    print({k: {"refs": v[0], "sentences": v[1]} for k, v in counts.items()}, "symlinks made", made, "total", len(os.listdir(LINKS)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
