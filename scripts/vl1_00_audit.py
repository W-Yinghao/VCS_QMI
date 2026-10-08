"""VL1-00 data audit for RefCOCOg (UMD): counts per official split and development role, candidate histograms, alias / ambiguity counts,
CLIP token overflow, image presence, leakage checks by COCO and Flickr id, nested FIT prefixes, and a 100-image box check sheet.
Writes reports/VL1/VL1_DATA_AUDIT.json, reports/VL1/vl1_roles.json, reports/VL1/box_check_{0..3}.jpg.  CPU only.
    python scripts/vl1_00_audit.py
"""
from __future__ import annotations

import collections
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vcs_vl import refcocog as RG  # noqa: E402
from vcs_vl.pairlaw import assert_image_disjoint, pair_laws  # noqa: E402

OUT = REPO / "reports" / "VL1"


def hist(xs) -> dict:
    c = collections.Counter(xs); return {str(k): c[k] for k in sorted(c)}


def main() -> int:
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    scenes, st = RG.load_scenes()
    role = RG.dev_roles(scenes); pref = RG.fit_prefixes(scenes, role)
    by_role = collections.defaultdict(list)
    for s in scenes:
        by_role[role[s.image_id]].append(s)
    assert_image_disjoint({r: [str(s.image_id) for s in v] for r, v in by_role.items()})
    fl_roles = {r: [s.flickr_id for s in v if s.flickr_id] for r, v in by_role.items()}
    fl_leak = None
    try:
        assert_image_disjoint(fl_roles)
    except ValueError as e:
        fl_leak = str(e)
    fl_dup = sum(c - 1 for c in collections.Counter(s.flickr_id for s in scenes if s.flickr_id).values() if c > 1)
    # pair laws on every eligible scene (>= 2 referred objects): masses and uniform region marginal
    bad_law = 0
    for s in scenes:
        if len(s.referred) >= 2:
            A, _, _ = s.compatibility(); p, q = pair_laws(A)
            bad_law += int(not (np.allclose(p.sum(1), 1 / len(A)) and np.isclose(q.sum(), 1)))
    import open_clip
    tok = open_clip.get_tokenizer("ViT-B-16")
    enc = getattr(tok, "encode", None) or tok.tokenizer.encode
    ctx = 77
    per_role = {}
    for r, v in sorted(by_role.items()):
        elig = [s for s in v if len(s.referred) >= 2]
        exprs = [e for s in v for o in s.referred for e in o.expressions]
        lens = [len(enc(e["raw"])) + 2 for e in exprs]
        same_cat = sum(1 for s in elig if len({o.category_id for o in s.referred}) < len(s.referred))
        amb = 0
        for s in v:
            seen = collections.defaultdict(set)
            for o in s.referred:
                for e in o.expressions:
                    seen[e["raw"].strip().lower()].add(o.ann_id)
            amb += sum(1 for ids in seen.values() if len(ids) > 1)
        per_role[r] = {"images": len(v), "eligible_images_ge2_referred": len(elig), "referred_objects": sum(len(s.referred) for s in v),
                       "expressions": len(exprs), "distractors": sum(len(s.distractors) for s in v),
                       "hist_referred_per_image": hist(len(s.referred) for s in v), "hist_all_objects_per_eligible_image": hist(len(s.referred) + len(s.distractors) for s in elig),
                       "hist_expressions_per_object": hist(len(o.expressions) for s in v for o in s.referred),
                       "eligible_images_with_same_category_referred_pair": same_cat,
                       "identical_expression_text_for_different_objects_in_one_image": amb,
                       "clip_token_length_quantiles": dict(zip(["q50", "q90", "q99", "max"], np.percentile(lens, [50, 90, 99, 100]).tolist())) if lens else None,
                       "clip_token_overflow_fraction": float(np.mean([l > ctx for l in lens])) if lens else None,
                       "images_missing_on_disk": sum(1 for s in v if s.path is None)}
    sizes = [os.path.getsize(s.path) for s in scenes if s.path]
    audit = {"dataset": "RefCOCOg (UMD split), COCO-2014-train images from the local COCO-2017 copy", "official_url": "https://github.com/lichengunc/refer",
             "revision": RG.ARCHIVE, "license_confirmed": False,
             "license_note": "refer API code Apache-2.0; RefCOCOg annotations (Mao et al. 2016; UMD split Nagaraja et al. 2016) distributed by the authors for research; images: COCO (Flickr terms per image). Owner confirmation pending.",
             "downloaded": True, "images_found": len(sizes), "images_missing": sum(1 for s in scenes if s.path is None),
             "image_size_bytes": int(sum(sizes)), "split_counts": per_role, "load_stats": {k: v for k, v in st.items() if k != "categories"},
             "canonical_source_ids_available": True, "canonical_ids": "COCO image id + Flickr photo id (from flickr_url)",
             "overlap_audit": {"roles_disjoint_by_coco_id": True, "roles_disjoint_by_flickr_id": fl_leak is None, "flickr_leak": fl_leak,
                               "duplicate_flickr_ids_within_dataset": fl_dup, "flickr30k_overlap": "pending (Flickr30k images not yet obtained)"},
             "n_entities": sum(len(s.referred) for s in scenes), "n_multibox": 0, "n_nobox": 0,
             "multibox_note": "RefCOCOg: every ref points to one COCO box; multi-target expressions do not occur",
             "candidate_histogram": {r: v["hist_referred_per_image"] for r, v in per_role.items()},
             "phrase_token_overflow_fraction": {r: v["clip_token_overflow_fraction"] for r, v in per_role.items()},
             "pair_law_checks": {"eligible_scenes_failing_mass_or_uniform_region": bad_law},
             "fit_prefixes": {str(n): {"n": len(v), "sha256": RG.digest(v)} for n, v in pref.items()},
             "annotation_to_crop_check": "reports/VL1/box_check_{0..3}.jpg (100 FIT images; see intake)", "cache_bytes_measured": None,
             "raw_baseline_dev": None, "seconds": None, "split_seed": RG.SPLIT_SEED}
    json.dump({"roles": {str(k): v for k, v in sorted(role.items())}, "fit_prefixes": {str(n): v for n, v in pref.items()}, "sha256": RG.digest(role)},
              open(OUT / "vl1_roles.json", "w"))
    # 100-image box check (FIT, eligible, fixed seed), 4 sheets of 25
    from PIL import Image, ImageDraw
    fit = sorted(s.image_id for s in by_role["FIT"] if len(s.referred) >= 2 and s.path)
    pick = [fit[i] for i in np.random.default_rng(7).choice(len(fit), 100, replace=False)]
    sm = {s.image_id: s for s in scenes}
    for sheet in range(4):
        canvas = Image.new("RGB", (5 * 320, 5 * 300), "white")
        for k, iid in enumerate(pick[sheet * 25:(sheet + 1) * 25]):
            s = sm[iid]; im = Image.open(s.path).convert("RGB"); sc = 320 / max(im.size); im = im.resize((int(im.width * sc), int(im.height * sc)))
            d = ImageDraw.Draw(im)
            for j, o in enumerate(s.referred[:3]):
                x0, y0, x1, y1 = RG.clip_box(o.box_xywh, s.width, s.height)
                col = ["red", "lime", "cyan"][j]; d.rectangle([x0 * sc, y0 * sc, x1 * sc, y1 * sc], outline=col, width=3)
            tile = Image.new("RGB", (320, 300), "white"); tile.paste(im, (0, 0)); dt = ImageDraw.Draw(tile)
            dt.text((2, 242), f"{iid}", fill="black")
            for j, o in enumerate(s.referred[:3]):
                dt.text((2, 254 + 14 * j), (["R", "G", "C"][j] + ": " + o.expressions[0]["raw"])[:52], fill="black")
            canvas.paste(tile, ((k % 5) * 320, (k // 5) * 300))
        canvas.save(OUT / f"box_check_{sheet}.jpg", quality=80)
    audit["seconds"] = time.time() - t0
    json.dump(audit, open(OUT / "VL1_DATA_AUDIT.json", "w"), indent=1)
    for r, v in per_role.items():
        print(r, {k: v[k] for k in ("images", "eligible_images_ge2_referred", "referred_objects", "expressions", "distractors", "eligible_images_with_same_category_referred_pair", "clip_token_overflow_fraction", "images_missing_on_disk")})
    print("images found", audit["images_found"], "missing", audit["images_missing"], "bytes", audit["image_size_bytes"], "load", audit["load_stats"], "flickr leak", fl_leak, "dup flickr", fl_dup, "bad laws", bad_law, f"{audit['seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
