"""Audit (2026-10-13): which pixel bounds did the R7-40a teacher actually run with?  For every teacher record, recompute the merged grid
from the image header with smart_resize (verbatim transformers 4.57.6 logic, scripts/r7_d1_40c_run.py) under three settings and count how
many recorded grids each reproduces:
  frozen   min 256·28², max 1280·28²   (R7-20 / R7-40 protocol constants, scripts/r7_d1_40_host.py MIN_PIXELS / MAX_PIXELS)
  ckpt     min 3136,    max 12845056   (the checkpoint's preprocessor_config.json values)
  classdef min 3136,    max 1280·28²   (Qwen2VLImageProcessorFast class default)
Reads only GQA train image headers (no pixels, no labels)."""
import glob, json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, "/home/infres/yinwang/CS_QMI/ssl_pilot/scripts")
from PIL import Image
import r7_d1_40c_run as RUN
IMG = Path("/projects/EEG-foundation-model/yinghao/data/gqa/images")
S = {"frozen": (256 * 28 * 28, 1280 * 28 * 28), "ckpt": (3136, 12845056), "classdef": (3136, 1280 * 28 * 28)}
size_cache, match, n, differ_frozen_ckpt, ex = {}, Counter(), 0, 0, []
for f in sorted(glob.glob("/home/infres/yinwang/CS_QMI/outputs/R7_D1/r7_40/teacher/*.jsonl")):
    for line in open(f):
        r = json.loads(line); iid = r["image_id"]; g = tuple(r["grid"])
        if iid not in size_cache:
            with Image.open(IMG / f"{iid}.jpg") as im:
                size_cache[iid] = im.size
        w, h = size_cache[iid]; n += 1
        pred = {k: tuple(x // RUN.FACTOR for x in RUN.smart_resize(h, w, min_pixels=a, max_pixels=b)) for k, (a, b) in S.items()}
        for k, p in pred.items():
            match[k] += (p == g)
        if pred["frozen"] != pred["ckpt"]:
            differ_frozen_ckpt += 1
            if len(ex) < 5: ex.append({"image_id": iid, "size_wh": [w, h], "recorded": g, **{k: v for k, v in pred.items()}})
print(json.dumps({"records": n, "images": len(size_cache), "matches": match, "records_where_frozen_and_ckpt_differ": differ_frozen_ckpt,
                  "examples": ex}, default=list, indent=1))
