"""P72 diagnostic (not a pre-registered unit): own-caption retrieval of raw CLIP (identity adapters, no training) on the R2 evaluation splits,
to locate the m = 0 R@1 of the topic-pairing adapters relative to the untrained starting point.  CPU only."""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from precheck_a_adapters import load_split, retrieval  # noqa: E402

DIRS = {"animal": "/home/infres/yinwang/CS_QMI/outputs/P49_precheck_A",
        "indoor_outdoor": "/home/infres/yinwang/CS_QMI/outputs/P57_precheck_A_wave2/features_indoor_outdoor"}
for shift, d in DIRS.items():
    for split in ("SRC-EVAL", "TGT-EVAL"):
        img, txt, _ = load_split(Path(d), split)
        r1, r5 = retrieval(F.normalize(img, dim=-1), F.normalize(txt[:, 4], dim=-1))
        print(f"{shift} {split}: raw CLIP R@1 {r1:.3f} R@5 {r5:.3f} (n = {len(img)})")
