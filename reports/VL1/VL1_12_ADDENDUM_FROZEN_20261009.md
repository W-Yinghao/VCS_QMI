# VL1-12 addendum — Table A on FG-CLIP 2 Base region features (estimator as a probe) — FROZEN 2026-10-09

Same protocol as `VL1_FROZEN_PROTOCOL.md` (roles, laws, scorer F2r with the feature width 768, routes, selection, readings); only the feature cache
changes: FG-CLIP 2 Base (released checkpoint @ 430fbc8), official `get_image_region_features` (RoIAlign 1×1 on the dense-feature head, default
256-patch NaFlex processor), expressions lower-cased, max_length 64, walk_type box (`features_fgclip2_base/train_side_features.pt`).  Region-path
check: COCO val2017 GT-box top-1 64.77 (paper 74.9 — discrepancy documented in the baseline plan).
Units: raw, VCS, JS, softmax, RFF at N = all × 3 seeds (lr grid as VL1-10).  Reading: per route, FG-CLIP 2 − CLIP on CAL common J and DEV Top-1
(paired by seed) — "how much region–phrase relation each frozen feature set carries, measured by the same estimator"; VCS − JS on FG-CLIP 2
features (same-posterior check).
