# VL1-12 addendum 3 — Table A on FG-CLIP (v1) Base region features — FROZEN 2026-10-09

Owner 2026-10-09: VL baselines first.  FG-CLIP v1 Base (`qihoo360/fg-clip-base` @ 454d763, Apache-2.0) is initialised from OpenAI CLIP ViT-B/16, so
**FG-CLIP v1 − CLIP crops** is the same-initialisation contrast of the fine-grained training + region interface, parallel to FG-CLIP 2 − SigLIP 2
(addendum 2).  Same protocol as `VL1_FROZEN_PROTOCOL.md` (roles, region-uniform law, F2r with width 512, routes, lr grid, selection, readings).

## Features
`scripts/vl1_12_fgclip1_features.py`: region interface `get_image_box_roi_features` (RoIAlign 1×1 on the projected second-to-last-layer map),
224 squash + CLIP image processor, clipped boxes in 14 × 14 grid units (branch v1.0 `coco_box_cls.py` RoIAlign function; that script's
`evaluate()` calls a crop variant instead — disclosed); text as written, max_length 77, `walk_short_pos=True`.

## Decisions at the freeze
- **Gate (job 1031246):** loads under transformers 4.57 (`einops` added to the baseline venv); COCO val2017 GT-box top-1 **52.28** on all
  36 781 boxes (ImageNet templates) > CLIP B/16 44.2 → PASS.  A published v1 number on this protocol could not be verified [U].
  Cache: 103 780 regions / 64 234 expressions / 13 429 images (172 s).
- Units (CPU): raw, VCS, JS, softmax, RFF at N all × 3 seeds → outputs/VL1_12_fgclip1 (`slurm/vl1_12f1_feed.txt`).

## Reading (descriptive)
1. FG-CLIP v1 − CLIP crops per route, paired by seed: DEV Top-1 and CAL common J (same init; training + region interface, not separated).
2. VCS − JS on FG-CLIP v1 (same posterior).  3. Table B row: raw FG-CLIP v1 cosine.  W11 caveat as in addendum 2.
