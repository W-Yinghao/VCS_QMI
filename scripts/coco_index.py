"""Build a compact index of the local COCO-2017 copy for pre-checks A/C: per image -> file, split, n_captions, supercategories present.
Pure metadata (no images read); output JSON per split + a summary table.  Source: /projects/EEG-foundation-model/yinghao/FMCA-AV/coco."""
import json, sys, collections, hashlib
from pathlib import Path
ROOT = Path("/projects/EEG-foundation-model/yinghao/FMCA-AV/coco"); OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/infres/yinwang/CS_QMI/data/coco_index")
OUT.mkdir(parents=True, exist_ok=True); summary = {}
for split in ("train2017", "val2017"):
    cap = json.load(open(ROOT / "annotations" / f"captions_{split}.json")); ins = json.load(open(ROOT / "annotations" / f"instances_{split}.json"))
    cats = {c["id"]: c for c in ins["categories"]}
    caps = collections.defaultdict(list)
    for a in cap["annotations"]:
        caps[a["image_id"]].append(a["caption"].strip())
    sup = collections.defaultdict(set); ncat = collections.defaultdict(set)
    for a in ins["annotations"]:
        sup[a["image_id"]].add(cats[a["category_id"]]["supercategory"]); ncat[a["image_id"]].add(cats[a["category_id"]]["name"])
    idx = {}
    for im in cap["images"]:
        i = im["id"]; idx[i] = {"file": im["file_name"], "w": im["width"], "h": im["height"], "n_captions": len(caps[i]), "supercats": sorted(sup.get(i, [])), "n_instances_cats": len(ncat.get(i, []))}
    json.dump({"split": split, "images": idx, "captions": {str(k): v for k, v in caps.items()}, "licence_note": "COCO 2017: annotations CC BY 4.0; images under their Flickr licences (research use); local copy owned by the user (FMCA-AV project)"}, open(OUT / f"{split}_index.json", "w"))
    sc = collections.Counter(s for v in idx.values() for s in v["supercats"]); none = sum(1 for v in idx.values() if not v["supercats"])
    summary[split] = {"n_images": len(idx), "n_with_captions": sum(1 for v in idx.values() if v["n_captions"] > 0), "captions_per_image": collections.Counter(v["n_captions"] for v in idx.values()), "images_without_instances": none, "supercategory_image_counts": dict(sc.most_common())}
    print(split, "images", len(idx), "no-instance images", none, "supercats:", dict(sc.most_common()))
json.dump(summary, open(OUT / "summary.json", "w"), indent=2, default=str)
h = hashlib.sha256((ROOT / "annotations" / "captions_train2017.json").read_bytes()).hexdigest(); print("captions_train2017.json sha256", h[:16])
