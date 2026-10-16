# V9-N2 overlap audit of the "independent batch" rho interval (review item U07) — f30k precheck cells

Generated 2026-10-16T01:17:18Z by `scripts/v9/n2_overlap_audit.py` (python 3.10.20, numpy 2.2.6); numbers below are copied from `overlap_audit.json` in this folder.  Read-only inputs per cell: `/projects/EEG-foundation-model/yinghao/outputs_v9/n2/f30k/rho<rho>/s<seed>/` `roles.json`, `pairs.observed_pairs.json`, `pairs.noise_manifest.json`; checks also read `weights/*.csv.lock.json`, the VCS / BCE weight CSVs and the precheck JSON.  The eval-only label file was not opened.  Facts and counts only.

Definitions (code references as in `meta.code_facts`; the sha256 of each cited file is in `meta.code_sha256`):

- **H** = held-out "independent batch", **E** = estimation batch: pipeline.py:72-73: held = rng([roles seed, 9031]).choice(sorted distinct observed U image ids, round(20 %), no replacement); U_est_mask = observed image not in held.  A U pair belongs to the group of its observed image.
- Caption-source image of caption c: c // 5 (data.py:45: caption_image = {i: i // 5}; noise.py:33 row image_id = the caption's own image; noise.py:41-52 replaces the observed image of the selected U pairs by a permutation of those pairs' own images).  `X<-Y` counts pairs observed with an image of group X whose caption comes from an image of group Y.
- An image "appears in both groups" when it is the observed or the caption-source image of at least one H pair and of at least one E pair.
- Q rows: weights.py:105-119 repair_within_role (donor = observed image of another row of the same role); pipeline.py:63-69 one rng for critic_fit, critic_select, reference_moments in that order.
- Image graph: one node per train image (29000 images, all roles); an edge joins the observed and the caption-source image of every mismatched U pair, and the caption-source and donor image of every Q row of the three roles (the pair <-> image incidence graph read on its image nodes).
- Both rho calls: weights.py:174 (estimation batch) and :177 (independent batch) pass the same T_P_ref, T_Q_ref, ref_P_img, ref_Q_img; the critic T is fitted once (critic_fit / critic_select) for both.

## 1. Reproduction and input checks

No cell stores the held-out list (`U_independent_images` exists only in the in-memory scores dict).  The split was reproduced with the exact rule and checked (a) against the pipeline's own `compute_scores` run with the teacher stubbed (held-out set, estimation mask, Q rows of the three roles), (b) against the counts every interval lock saved, and (c) by recomputing c = mean T over each batch from the locked VCS / BCE CSV scores with the saved critic (a, b); (c) ties the reproduced split to what the precheck runs used, whatever the code version.

| cell | input checks | held-out = pipeline | est. mask = pipeline | Q rows = pipeline | saved counts (4 locks) | max abs diff of c (est / ind), VCS + BCE | held-out ids sha256 |
|---|---|---|---|---|---|---|---|
| rho0.4/s901 | 14/14 pass | True | True | True | match (4) | 9.60e-14 / 1.43e-14 | bea34a4d02c9 |
| rho0.4/s902 | 14/14 pass | True | True | True | match (4) | 1.27e-13 / 2.54e-14 | 30e06a12125e |
| rho0.4/s903 | 14/14 pass | True | True | True | match (4) | 6.99e-14 / 1.09e-14 | 9bb263992275 |
| rho0.4/s917 | 14/14 pass | True | True | True | match (4) | 9.79e-14 / 2.15e-13 | 0a3dd6c27be8 |
| rho0/s901 | 14/14 pass | True | True | True | match (4) | 1.12e-14 / 3.98e-13 | bea34a4d02c9 |
| rho0/s902 | 14/14 pass | True | True | True | match (4) | 1.57e-13 / 3.52e-13 | 30e06a12125e |
| rho0/s903 | 14/14 pass | True | True | True | match (4) | 2.69e-13 / 2.90e-13 | 9bb263992275 |
| rho0/s917 | 14/14 pass | True | True | True | match (4) | 7.19e-14 / 3.10e-13 | 0a3dd6c27be8 |

Held-out sets per seed are identical at rho 0.4 and rho 0: s901: True, s902: True, s903: True, s917: True.

## 2. Groups and caption-source links (U pairs)

| cell | E pairs / images | H pairs / images | mismatched pairs E / H | H<-E | E<-H | H<-H | E<-E | cross-group share of mismatched | images in both groups (H images with a caption in E + E images with a caption in H) | clusters holding an other-group caption (H / E) |
|---|---|---|---|---|---|---|---|---|---|---|
| rho0.4/s901 | 104400 / 20880 | 26100 / 5220 | 41692 / 10507 | 8432 | 8432 | 2075 | 33260 | 16864 / 52199 = 32.3 % | 11638 (4475 + 7163) | 4484 (85.9 %) / 7206 (34.5 %) |
| rho0.4/s902 | 104400 / 20880 | 26100 / 5220 | 41804 / 10392 | 8332 | 8332 | 2060 | 33472 | 16664 / 52196 = 31.9 % | 11513 (4432 + 7081) | 4433 (84.9 %) / 7074 (33.9 %) |
| rho0.4/s903 | 104400 / 20880 | 26100 / 5220 | 41747 / 10450 | 8371 | 8371 | 2079 | 33376 | 16742 / 52197 = 32.1 % | 11592 (4470 + 7122) | 4484 (85.9 %) / 7113 (34.1 %) |
| rho0.4/s917 | 104400 / 20880 | 26100 / 5220 | 41797 / 10401 | 8341 | 8341 | 2060 | 33456 | 16682 / 52198 = 32.0 % | 11541 (4450 + 7091) | 4421 (84.7 %) / 7149 (34.2 %) |
| rho0/s901 | 104400 / 20880 | 26100 / 5220 | 0 / 0 | 0 | 0 | 0 | 0 | 0 / 0 = n/a | 0 (0 + 0) | 0 (0.0 %) / 0 (0.0 %) |
| rho0/s902 | 104400 / 20880 | 26100 / 5220 | 0 / 0 | 0 | 0 | 0 | 0 | 0 / 0 = n/a | 0 (0 + 0) | 0 (0.0 %) / 0 (0.0 %) |
| rho0/s903 | 104400 / 20880 | 26100 / 5220 | 0 / 0 | 0 | 0 | 0 | 0 | 0 / 0 = n/a | 0 (0 + 0) | 0 (0.0 %) / 0 (0.0 %) |
| rho0/s917 | 104400 / 20880 | 26100 / 5220 | 0 / 0 | 0 | 0 | 0 | 0 | 0 / 0 = n/a | 0 (0 + 0) | 0 (0.0 %) / 0 (0.0 %) |

H<-E equals E<-H in every cell: `inject_mismatch` gives the selected pairs a permutation of their own images (noise.py:46-47), so the selected pairs observed with an H image are as many as the selected pairs whose caption comes from an H image.

## 3. Image graph (caption-source links + Q donor links)

| cell | link edges | components | largest component: images (H / E / anchor), U pairs | largest component with U images: images (H / E), U pairs | components spanning H and E | images in spanning components H / E (share of H) | U pairs in spanning components H / E | H images sharing a component with another H image |
|---|---|---|---|---|---|---|---|---|
| rho0.4/s901 | 60899 | 3178 | 24086 (4849 / 19237 / 0), 120430 | 24086 (4849 / 19237), 120430 | 1 | 4849 / 19237 (92.9 %) | 24245 / 96185 | 4849 |
| rho0.4/s902 | 60896 | 3212 | 24052 (4775 / 19277 / 0), 120260 | 24052 (4775 / 19277), 120260 | 1 | 4775 / 19277 (91.5 %) | 23875 / 96385 | 4775 |
| rho0.4/s903 | 60897 | 3175 | 24089 (4838 / 19251 / 0), 120445 | 24089 (4838 / 19251), 120445 | 1 | 4838 / 19251 (92.7 %) | 24190 / 96255 | 4838 |
| rho0.4/s917 | 60898 | 3249 | 24015 (4785 / 19230 / 0), 120075 | 24015 (4785 / 19230), 120075 | 1 | 4785 / 19230 (91.7 %) | 23925 / 96150 | 4785 |
| rho0/s901 | 8700 | 27263 | 870 (0 / 0 / 870), 0 | 1 (1 / 0), 5 | 0 | 0 / 0 (0.0 %) | 0 / 0 | 0 |
| rho0/s902 | 8700 | 27263 | 870 (0 / 0 / 870), 0 | 1 (0 / 1), 5 | 0 | 0 / 0 (0.0 %) | 0 / 0 | 0 |
| rho0/s903 | 8700 | 27263 | 870 (0 / 0 / 870), 0 | 1 (0 / 1), 5 | 0 | 0 / 0 (0.0 %) | 0 / 0 | 0 |
| rho0/s917 | 8700 | 27263 | 870 (0 / 0 / 870), 0 | 1 (1 / 0), 5 | 0 | 0 / 0 (0.0 %) | 0 / 0 | 0 |

## 4. Q donors and the reference rows shared by both rho calls

| cell | Q rows fit / select / reference | donors in the same role | donors in another anchor role / in U | Q links crossing H and E | self donors (fit / select / reference) | reference rows used by both calls: P rows / P clusters (images) / Q rows (clusters) / distinct Q donor images |
|---|---|---|---|---|---|---|
| rho0.4/s901 | 4350 / 2175 / 2175 | 8700 | 0 / 0 | 0 | 0 / 0 / 0 | 2175 / 435 / 2175 (2175) / 435 |
| rho0.4/s902 | 4350 / 2175 / 2175 | 8700 | 0 / 0 | 0 | 0 / 0 / 0 | 2175 / 435 / 2175 (2175) / 435 |
| rho0.4/s903 | 4350 / 2175 / 2175 | 8700 | 0 / 0 | 0 | 0 / 0 / 0 | 2175 / 435 / 2175 (2175) / 435 |
| rho0.4/s917 | 4350 / 2175 / 2175 | 8700 | 0 / 0 | 0 | 0 / 0 / 0 | 2175 / 435 / 2175 (2175) / 435 |
| rho0/s901 | 4350 / 2175 / 2175 | 8700 | 0 / 0 | 0 | 0 / 0 / 0 | 2175 / 435 / 2175 (2175) / 435 |
| rho0/s902 | 4350 / 2175 / 2175 | 8700 | 0 / 0 | 0 | 0 / 0 / 0 | 2175 / 435 / 2175 (2175) / 435 |
| rho0/s903 | 4350 / 2175 / 2175 | 8700 | 0 / 0 | 0 | 0 / 0 / 0 | 2175 / 435 / 2175 (2175) / 435 |
| rho0/s917 | 4350 / 2175 / 2175 | 8700 | 0 / 0 | 0 | 0 / 0 / 0 | 2175 / 435 / 2175 (2175) / 435 |

## 5. The package's one-unit-one-cluster contract with image ids as units

weights.py:135 builds units [f'P:{u}'], [f'Q:{u}'], [f'R:{u}'] from the cluster ids themselves, so REF.role_disjointness cannot see an image shared by two clusters.  With image ids as units (P cluster = its reference image; Q cluster = its caption-source and donor image; R cluster = its observed image plus the caption-source images of its pairs):

| cell | call | clusters P / Q / R | units in more than one cluster | clusters sharing a unit P / Q / R | shared between the calls: identical P / Q clusters, images in R units of both |
|---|---|---|---|---|---|
| rho0.4/s901 | independent (H) | 435 / 2175 / 5220 | 3394 | 435 / 2175 / 3731 | 435 / 2175, 11638 |
| rho0.4/s901 | estimation (E) | 435 / 2175 / 20880 | 21052 | 435 / 2175 / 19197 | 435 / 2175, 11638 |
| rho0.4/s902 | independent (H) | 435 / 2175 / 5220 | 3326 | 435 / 2175 / 3632 | 435 / 2175, 11513 |
| rho0.4/s902 | estimation (E) | 435 / 2175 / 20880 | 20988 | 435 / 2175 / 19230 | 435 / 2175, 11513 |
| rho0.4/s903 | independent (H) | 435 / 2175 / 5220 | 3380 | 435 / 2175 / 3719 | 435 / 2175, 11592 |
| rho0.4/s903 | estimation (E) | 435 / 2175 / 20880 | 21017 | 435 / 2175 / 19193 | 435 / 2175, 11592 |
| rho0.4/s917 | independent (H) | 435 / 2175 / 5220 | 3349 | 435 / 2175 / 3711 | 435 / 2175, 11541 |
| rho0.4/s917 | estimation (E) | 435 / 2175 / 20880 | 20993 | 435 / 2175 / 19186 | 435 / 2175, 11541 |
| rho0/s901 | independent (H) | 435 / 2175 / 5220 | 435 | 435 / 2175 / 0 | 435 / 2175, 0 |
| rho0/s901 | estimation (E) | 435 / 2175 / 20880 | 435 | 435 / 2175 / 0 | 435 / 2175, 0 |
| rho0/s902 | independent (H) | 435 / 2175 / 5220 | 435 | 435 / 2175 / 0 | 435 / 2175, 0 |
| rho0/s902 | estimation (E) | 435 / 2175 / 20880 | 435 | 435 / 2175 / 0 | 435 / 2175, 0 |
| rho0/s903 | independent (H) | 435 / 2175 / 5220 | 435 | 435 / 2175 / 0 | 435 / 2175, 0 |
| rho0/s903 | estimation (E) | 435 / 2175 / 20880 | 435 | 435 / 2175 / 0 | 435 / 2175, 0 |
| rho0/s917 | independent (H) | 435 / 2175 / 5220 | 435 | 435 / 2175 / 0 | 435 / 2175, 0 |
| rho0/s917 | estimation (E) | 435 / 2175 / 20880 | 435 | 435 / 2175 / 0 | 435 / 2175, 0 |

Implication: at rho 0.4 the held-out batch is separated from the estimation batch by receiving-image id only — 16664–16864 mismatched pairs (31.9–32.3 % of the mismatched) link the two groups, 4421–4484 of the 5220 held-out clusters hold a caption from an estimation image, 1 graph component spans both groups and holds 91.5–92.9 % of the held-out images, and both rho calls use the same reference P / Q rows — so these counts do not support calling the interval "independent" of the point estimate; at rho 0 no caption-source link crosses the split (0 in every cell), and the two calls still use the same reference P / Q rows and the same fitted critic.
