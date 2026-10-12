"""Development-only inspection for scripts/r5_final_eval.py (never opens a test file): structure of the NON-test MDETR train files and the
R4-VL0 DEV meta, and whether the MDETR train-file DEV expressions reproduce the DEV inputs used in the r4 / r5 reports.
    python -I scripts/r5_final_eval_inspect.py
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

MDETR = Path("/projects/EEG-foundation-model/yinghao/datasets/mdetr_annotations")
META = Path("/home/infres/yinwang/CS_QMI/outputs/R4_VL0"); PROP = Path("/home/infres/yinwang/CS_QMI/outputs/VL1_21/proposals")


def main():
    for ds, mname in (("refcocoplus", "refcoco+"), ("refcoco", "refcoco"), ("refcocog", "refcocog")):
        f = MDETR / f"finetune_{mname}_train.json"
        assert "test" not in f.name.lower()
        d = json.load(open(f)); print(f"== {f.name}: top keys {sorted(d)}; images {len(d['images'])}; annotations {len(d['annotations'])}", flush=True)
        print("  image keys", sorted(d["images"][0])); print("  ann keys", sorted(d["annotations"][0]))
        im0 = {k: v for k, v in d["images"][0].items() if k not in ("tokens_negative", "tokens_positive_eval")}; print("  first image entry", im0)
        a0 = {k: v for k, v in d["annotations"][0].items() if k not in ("segmentation",)}; print("  first ann entry", a0)
        anns = {a["image_id"]: a for a in d["annotations"]}; print("  one ann per image entry:", len(anns) == len(d["images"]) == len(d["annotations"]))
        meta = torch.load(META / f"meta_{ds}.pt", weights_only=False); print(f"  meta_{ds}.pt: {len(meta)} DEV images; keys {sorted(meta[0])}")
        dev_keys = set(torch.load(PROP / f"{ds}_DEV.pt", map_location="cpu", weights_only=False).keys())
        by_img = defaultdict(list)
        for im in d["images"]:
            by_img[int(im["original_id"])].append(im)
        print(f"  DEV proposal-file images {len(dev_keys)}; in MDETR train {len(dev_keys & set(by_img))}; meta images in DEV keys {sum(m['image_id'] in dev_keys for m in meta)}")
        n_txt = n_txt_ok = n_box_ok = n_sid = 0; mism = []; n_ref_match = 0; ge2 = 0
        for m in meta:
            ims = by_img.get(m["image_id"], [])
            refs = {}
            for im in ims:
                a = anns[im["id"]]; refs.setdefault(int(a["original_id"]), a["bbox"])
            ge2 += len(refs) >= 2
            n_ref_match += sorted(refs) == sorted(int(x) for x in m["ref_ann"])
            mt = {int(s): (t, m["ref_boxes"][k]) for s, t, k in zip(m["sent_id"], m["texts"], m["target"])}
            md = {int(im["id"]): im for im in ims}
            for s, (t, rb) in mt.items():
                n_txt += 1
                if s in md:
                    n_sid += 1; im = md[s]; a = anns[im["id"]]; bx = a["bbox"]; xyxy = [bx[0], bx[1], bx[0] + bx[2], bx[1] + bx[3]]
                    n_txt_ok += im["caption"] == t; n_box_ok += bool(np.allclose(xyxy, rb, atol=1e-6))
                    if im["caption"] != t and len(mism) < 8:
                        mism.append((s, t, im["caption"]))
        print(f"  meta expressions {n_txt}; sent_id found in MDETR train {n_sid}; caption == meta text {n_txt_ok}; target box == meta ref box {n_box_ok}; "
              f"images whose referred-ann set matches {n_ref_match}/{len(meta)}; meta images with >= 2 MDETR referred anns {ge2}")
        for x in mism:
            print("   text mismatch (DEV, train split):", x)
        cnt = Counter(len(v) for v in by_img.values()); print("  expressions per train image (top)", cnt.most_common(5), flush=True)


def refcocog_text_forms():
    """RefCOCOg: MDETR ids are not REFER sent_ids; compare captions with the DEV meta raw texts per (image, target annotation)."""
    d = json.load(open(MDETR / "finetune_refcocog_train.json")); anns = {a["image_id"]: a for a in d["annotations"]}
    by = defaultdict(list)
    for im in d["images"]:
        by[(int(im["original_id"]), int(anns[im["id"]]["original_id"]))].append(im["caption"])
    meta = torch.load(META / "meta_refcocog.pt", weights_only=False); n = same = same_lower = same_norm = 0; ex = []
    norm = lambda t: " ".join(re.findall(r"[a-z0-9]+", t.lower()))
    for m in meta:
        for t, k in zip(m["texts"], m["target"]):
            caps = by.get((m["image_id"], int(m["ref_ann"][k])), []); n += 1
            same += t in caps; same_lower += t.lower() in caps; same_norm += norm(t) in [norm(c) for c in caps]
            if t not in caps and len(ex) < 6:
                ex.append((t, caps[:3]))
    print(f"refcocog DEV meta expressions {n}: raw in MDETR captions {same}; lower(raw) {same_lower}; alnum-normalized {same_norm}")
    for e in ex:
        print("  example (DEV, train split):", e)


def compare_props(new_path, old_path):
    """Set-level comparison of regenerated detector proposals with the stored DEV ones (order-insensitive)."""
    new = torch.load(new_path, map_location="cpu", weights_only=False); old = torch.load(old_path, map_location="cpu", weights_only=False)
    def iou(a, B):
        x1 = np.maximum(a[0], B[:, 0]); y1 = np.maximum(a[1], B[:, 1]); x2 = np.minimum(a[2], B[:, 2]); y2 = np.minimum(a[3], B[:, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None); return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - inter)
    st = defaultdict(list)
    for iid, x in new.items():
        y = old[iid]; x = x.numpy(); y = y.numpy(); st["n_new"].append(len(x)); st["n_old"].append(len(y))
        st["same_rows_sorted"].append(len(x) == len(y) and np.array_equal(x[np.lexsort(x.T[::-1])], y[np.lexsort(y.T[::-1])]))
        m = []
        for r in x:
            same = y[y[:, 5] == r[5]]
            m.append(iou(r, same).max() if len(same) else 0.0)
        m = np.array(m); st["frac_new_matched_iou99"].append(float((m >= 0.99).mean()) if len(m) else 1.0); st["frac_new_matched_iou90"].append(float((m >= 0.9).mean()) if len(m) else 1.0)
        st["top1_same_iou99"].append(bool(len(x) and len(y) and iou(x[0], y[:1])[0] >= 0.99))
        k = min(len(x), len(y), 100); st["abs_score_diff_top"].append(float(np.abs(np.sort(x[:, 4])[::-1][:k] - np.sort(y[:, 4])[::-1][:k]).max()) if k else 0.0)
    print(f"{new_path}: images {len(new)}; mean detections new {np.mean(st['n_new']):.2f} old {np.mean(st['n_old']):.2f}; same count {np.mean(np.array(st['n_new']) == np.array(st['n_old'])):.3f}; "
          f"identical row sets {np.mean(st['same_rows_sorted']):.3f}; new boxes matched (same class) IoU>=0.99 {np.mean(st['frac_new_matched_iou99']):.4f}, >=0.9 {np.mean(st['frac_new_matched_iou90']):.4f}; "
          f"top-1 box same {np.mean(st['top1_same_iou99']):.3f}; max |sorted score diff| median {np.median(st['abs_score_diff_top']):.4f} max {np.max(st['abs_score_diff_top']):.4f}", flush=True)


if __name__ == "__main__":
    import re
    if len(sys.argv) > 1 and sys.argv[1] == "compare_props":
        for i in range(2, len(sys.argv), 2):
            compare_props(sys.argv[i], sys.argv[i + 1])
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "refcocog_text":
        sys.exit(refcocog_text_forms())
    sys.exit(main())
