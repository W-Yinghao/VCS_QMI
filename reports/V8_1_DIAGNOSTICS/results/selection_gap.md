# D05 log level — training curves and selection alignment (generated 2026-10-15T18:51:31Z from `d05_log_level.json`; 48 units; no checkpoint read, no new inference)

Rule: log level only (docs/04 D05 level 1): selected = CAL_SELECT prompt-macro mean-rating MSE argmin (ties -> earlier); pairacc = within-prompt pairwise accuracy argmax (ties -> earlier); rps / ce argmin; curves from log.json epochs_log, 48 units.  Selected epoch = the frozen V8-03 rule (CAL_SELECT prompt-macro mean-rating MSE); the other columns are the epoch the same CAL_SELECT log would pick under a different criterion.  Log level only: a difference between columns flags a possible misalignment, it does not establish an external gain (docs/04 D05 level 1).

## 1. Formal arms (seeds 0-2): mean curves over the three seeds

| arm | CAL mean-rating MSE e0 / e1 / e2 / e3 / e4 | CAL pair acc e0..e4 | native train loss e0..e4 | selected (MSE) s0/s1/s2 | pair-acc best s0/s1/s2 | RPS best s0/s1/s2 |
|---|---|---|---|---|---|---|
| U-CS | 0.419 / 0.413 / 0.407 / 0.415 / 0.422 | 0.720 / 0.721 / 0.725 / 0.720 / 0.719 | 0.1690 / 0.1400 / 0.1280 / 0.1184 / 0.1134 | 2/0/2 | 2/2/1 | 1/0/2 |
| U-BCE | 0.427 / 0.408 / 0.406 / 0.414 / 0.419 | 0.723 / 0.725 / 0.724 / 0.721 / 0.720 | 0.5059 / 0.4296 / 0.3949 / 0.3681 / 0.3539 | 2/1/2 | 1/0/1 | 1/1/2 |
| U-RPS | 0.410 / 0.406 / 0.404 / 0.407 / 0.408 | 0.712 / 0.718 / 0.719 / 0.716 / 0.715 | 0.1208 / 0.1038 / 0.0976 / 0.0934 / 0.0914 | 2/0/4 | 2/2/1 | 1/1/0 |
| U-CE | 0.410 / 0.402 / 0.401 / 0.405 / 0.408 | 0.714 / 0.720 / 0.721 / 0.718 / 0.719 | 1.2540 / 1.1240 / 1.0696 / 1.0308 / 1.0134 | 2/1/0 | 2/2/1 | 1/1/0 |
| U-MSE | 0.417 / 0.409 / 0.411 / 0.415 / 0.418 | 0.713 / 0.718 / 0.718 / 0.717 / 0.716 | 0.0307 / 0.0191 / 0.0153 / 0.0128 / 0.0117 | 2/0/2 | 2/0/1 | 2/1/2 |
| U-L1 | 0.422 / 0.411 / 0.405 / 0.402 / 0.404 | 0.718 / 0.721 / 0.721 / 0.720 / 0.721 | 0.1365 / 0.1076 / 0.0938 / 0.0829 / 0.0764 | 3/0/3 | 3/0/2 | 4/3/1 |
| U-RANK | 0.567 / 0.656 / 0.780 / 0.865 / 0.915 | 0.720 / 0.721 / 0.719 / 0.714 / 0.713 | 0.6367 / 0.5694 / 0.5237 / 0.4918 / 0.4743 | 0/0/0 | 1/0/1 | 0/0/0 |
| U-LIST | 0.441 / 0.427 / 0.428 / 0.437 / 0.441 | 0.718 / 0.719 / 0.718 / 0.713 / 0.713 | 1.7073 / 1.6627 / 1.6383 / 1.6227 / 1.6142 | 2/1/2 | 3/1/1 | 0/1/0 |

## 2. Tuning units (seed 17, lr 5e-6 / 1e-5 / 2e-5): mean curves over the three lr

| arm | CAL MSE e0..e4 | CAL pair acc e0..e4 | train loss e0..e4 | selected (MSE) per lr | pair-acc best per lr |
|---|---|---|---|---|---|
| U-CS | 0.453 / 0.404 / 0.410 / 0.420 / 0.424 | 0.716 / 0.721 / 0.719 / 0.719 / 0.718 | 0.1699 / 0.1414 / 0.1292 / 0.1197 / 0.1143 | 1/1/1 | 1/3/1 |
| U-BCE | 0.465 / 0.403 / 0.406 / 0.417 / 0.421 | 0.715 / 0.723 / 0.719 / 0.718 / 0.717 | 0.5091 / 0.4331 / 0.3978 / 0.3715 / 0.3565 | 1/2/1 | 1/1/2 |
| U-RPS | 0.446 / 0.400 / 0.400 / 0.409 / 0.412 | 0.713 / 0.719 / 0.717 / 0.716 / 0.715 | 0.1202 / 0.1035 / 0.0962 / 0.0904 / 0.0871 | 2/1/1 | 2/1/0 |
| U-CE | 0.439 / 0.392 / 0.402 / 0.408 / 0.413 | 0.715 / 0.721 / 0.722 / 0.720 / 0.720 | 1.2484 / 1.1178 / 1.0515 / 0.9942 / 0.9615 | 1/1/1 | 2/4/1 |
| U-MSE | 0.448 / 0.407 / 0.404 / 0.421 / 0.422 | 0.715 / 0.717 / 0.716 / 0.714 / 0.714 | 0.0302 / 0.0190 / 0.0145 / 0.0113 / 0.0095 | 2/2/1 | 0/2/0 |
| U-L1 | 0.456 / 0.414 / 0.413 / 0.414 / 0.413 | 0.713 / 0.715 / 0.712 / 0.716 / 0.716 | 0.1376 / 0.1087 / 0.0948 / 0.0841 / 0.0772 | 2/4/1 | 3/2/1 |
| U-RANK | 0.635 / 0.665 / 0.807 / 0.907 / 0.972 | 0.726 / 0.715 / 0.718 / 0.713 / 0.711 | 0.6347 / 0.5622 / 0.5040 / 0.4588 / 0.4366 | 0/1/0 | 0/0/0 |
| U-LIST | 0.447 / 0.434 / 0.436 / 0.454 / 0.461 | 0.725 / 0.716 / 0.712 / 0.714 / 0.711 | 1.7043 / 1.6575 / 1.6289 / 1.6064 / 1.5980 | 1/2/1 | 0/0/0 |

## 3. Per unit

| unit | selected (MSE) | pair-acc best | RPS best | CE best | pair acc at selected − best | MSE at pair-acc-best epoch − selected | MSE e0 − selected | MSE e4 − selected | train loss e0 → e4 |
|---|---|---|---|---|---|---|---|---|---|
| V8-FIT-U-BCE-s0 | 2 | 1 | 1 | 1 | -0.0004 | +0.0109 | +0.0447 | +0.0170 | 0.5076 → 0.3537 |
| V8-FIT-U-BCE-s1 | 1 | 0 | 1 | 1 | -0.0065 | +0.0017 | +0.0017 | +0.0144 | 0.5045 → 0.3534 |
| V8-FIT-U-BCE-s2 | 2 | 1 | 2 | 0 | -0.0045 | +0.0038 | +0.0239 | +0.0155 | 0.5056 → 0.3547 |
| V8-FIT-U-CE-s0 | 2 | 2 | 1 | 1 | +0.0000 | +0.0000 | +0.0301 | +0.0140 | 1.2558 → 1.0112 |
| V8-FIT-U-CE-s1 | 1 | 2 | 1 | 1 | -0.0016 | +0.0003 | +0.0005 | +0.0053 | 1.2519 → 1.0149 |
| V8-FIT-U-CE-s2 | 0 | 1 | 0 | 0 | -0.0066 | +0.0003 | +0.0000 | +0.0054 | 1.2544 → 1.0143 |
| V8-FIT-U-CS-s0 | 2 | 2 | 1 | 1 | +0.0000 | +0.0000 | +0.0151 | +0.0120 | 0.1699 → 0.1132 |
| V8-FIT-U-CS-s1 | 0 | 2 | 0 | 0 | -0.0010 | +0.0037 | +0.0000 | +0.0182 | 0.1685 → 0.1135 |
| V8-FIT-U-CS-s2 | 2 | 1 | 2 | 2 | -0.0027 | +0.0079 | +0.0265 | +0.0188 | 0.1688 → 0.1134 |
| V8-FIT-U-L1-s0 | 3 | 3 | 4 | 2 | +0.0000 | +0.0000 | +0.0271 | +0.0005 | 0.1370 → 0.0763 |
| V8-FIT-U-L1-s1 | 0 | 0 | 3 | 1 | +0.0000 | +0.0000 | +0.0000 | +0.0055 | 0.1364 → 0.0763 |
| V8-FIT-U-L1-s2 | 3 | 2 | 1 | 1 | -0.0066 | +0.0040 | +0.0356 | +0.0018 | 0.1360 → 0.0766 |
| V8-FIT-U-LIST-s0 | 2 | 3 | 0 | 0 | -0.0009 | +0.0144 | +0.0213 | +0.0175 | 1.7074 → 1.6119 |
| V8-FIT-U-LIST-s1 | 1 | 1 | 1 | 0 | +0.0000 | +0.0000 | +0.0159 | +0.0256 | 1.7064 → 1.6159 |
| V8-FIT-U-LIST-s2 | 2 | 1 | 0 | 0 | -0.0009 | +0.0005 | +0.0108 | +0.0042 | 1.7082 → 1.6149 |
| V8-FIT-U-MSE-s0 | 2 | 2 | 2 | 2 | +0.0000 | +0.0000 | +0.0193 | +0.0147 | 0.0309 → 0.0116 |
| V8-FIT-U-MSE-s1 | 0 | 0 | 1 | 1 | +0.0000 | +0.0000 | +0.0000 | +0.0140 | 0.0305 → 0.0117 |
| V8-FIT-U-MSE-s2 | 2 | 1 | 2 | 2 | -0.0064 | +0.0002 | +0.0112 | +0.0053 | 0.0307 → 0.0117 |
| V8-FIT-U-RANK-s0 | 0 | 1 | 0 | 0 | -0.0001 | +0.1302 | +0.0000 | +0.3834 | 0.6363 → 0.4727 |
| V8-FIT-U-RANK-s1 | 0 | 0 | 0 | 0 | +0.0000 | +0.0000 | +0.0000 | +0.3277 | 0.6363 → 0.4760 |
| V8-FIT-U-RANK-s2 | 0 | 1 | 0 | 0 | -0.0064 | +0.0349 | +0.0000 | +0.3330 | 0.6374 → 0.4743 |
| V8-FIT-U-RPS-s0 | 2 | 2 | 1 | 1 | +0.0000 | +0.0000 | +0.0258 | +0.0122 | 0.1210 → 0.0912 |
| V8-FIT-U-RPS-s1 | 0 | 2 | 1 | 1 | -0.0009 | +0.0055 | +0.0000 | +0.0070 | 0.1205 → 0.0915 |
| V8-FIT-U-RPS-s2 | 4 | 1 | 0 | 0 | -0.0075 | +0.0030 | +0.0006 | +0.0000 | 0.1209 → 0.0917 |
| V8-TUNE-U-BCE-lr1e-05 | 1 | 1 | 1 | 1 | +0.0000 | +0.0000 | +0.0648 | +0.0181 | 0.5063 → 0.3540 |
| V8-TUNE-U-BCE-lr2e-05 | 2 | 1 | 1 | 1 | -0.0084 | +0.0015 | +0.0857 | +0.0243 | 0.5095 → 0.3391 |
| V8-TUNE-U-BCE-lr5e-06 | 1 | 2 | 1 | 1 | -0.0000 | +0.0042 | +0.0369 | +0.0130 | 0.5116 → 0.3766 |
| V8-TUNE-U-CE-lr1e-05 | 1 | 2 | 1 | 1 | -0.0040 | +0.0051 | +0.0380 | +0.0184 | 1.2431 → 0.9574 |
| V8-TUNE-U-CE-lr2e-05 | 1 | 4 | 1 | 1 | -0.0028 | +0.0282 | +0.0623 | +0.0282 | 1.2487 → 0.9161 |
| V8-TUNE-U-CE-lr5e-06 | 1 | 1 | 1 | 1 | +0.0000 | +0.0000 | +0.0411 | +0.0152 | 1.2535 → 1.0110 |
| V8-TUNE-U-CS-lr1e-05 | 1 | 1 | 1 | 1 | +0.0000 | +0.0000 | +0.0487 | +0.0193 | 0.1687 → 0.1132 |
| V8-TUNE-U-CS-lr2e-05 | 1 | 3 | 1 | 1 | -0.0035 | +0.0186 | +0.0621 | +0.0238 | 0.1702 → 0.1084 |
| V8-TUNE-U-CS-lr5e-06 | 1 | 1 | 1 | 1 | +0.0000 | +0.0000 | +0.0362 | +0.0165 | 0.1708 → 0.1213 |
| V8-TUNE-U-L1-lr1e-05 | 2 | 3 | 1 | 2 | -0.0091 | +0.0044 | +0.0581 | +0.0019 | 0.1362 → 0.0759 |
| V8-TUNE-U-L1-lr2e-05 | 4 | 2 | 2 | 2 | -0.0022 | +0.0018 | +0.0561 | +0.0000 | 0.1388 → 0.0724 |
| V8-TUNE-U-L1-lr5e-06 | 1 | 1 | 1 | 1 | +0.0000 | +0.0000 | +0.0244 | +0.0072 | 0.1378 → 0.0833 |
| V8-TUNE-U-LIST-lr1e-05 | 1 | 0 | 1 | 1 | -0.0144 | +0.0097 | +0.0097 | +0.0317 | 1.7031 → 1.5948 |
| V8-TUNE-U-LIST-lr2e-05 | 2 | 0 | 0 | 0 | -0.0110 | +0.0120 | +0.0120 | +0.0251 | 1.7033 → 1.5829 |
| V8-TUNE-U-LIST-lr5e-06 | 1 | 0 | 1 | 1 | -0.0049 | +0.0207 | +0.0207 | +0.0274 | 1.7064 → 1.6164 |
| V8-TUNE-U-MSE-lr1e-05 | 2 | 0 | 2 | 2 | -0.0031 | +0.0431 | +0.0431 | +0.0201 | 0.0297 → 0.0092 |
| V8-TUNE-U-MSE-lr2e-05 | 2 | 2 | 2 | 2 | +0.0000 | +0.0000 | +0.0627 | +0.0218 | 0.0304 → 0.0079 |
| V8-TUNE-U-MSE-lr5e-06 | 1 | 0 | 1 | 1 | -0.0007 | +0.0358 | +0.0358 | +0.0213 | 0.0306 → 0.0116 |
| V8-TUNE-U-RANK-lr1e-05 | 0 | 0 | 0 | 0 | +0.0000 | +0.0000 | +0.0000 | +0.3795 | 0.6319 → 0.4298 |
| V8-TUNE-U-RANK-lr2e-05 | 1 | 0 | 1 | 0 | -0.0187 | +0.0220 | +0.0220 | +0.3039 | 0.6352 → 0.4028 |
| V8-TUNE-U-RANK-lr5e-06 | 0 | 0 | 0 | 0 | +0.0000 | +0.0000 | +0.0000 | +0.3483 | 0.6370 → 0.4771 |
| V8-TUNE-U-RPS-lr1e-05 | 2 | 2 | 1 | 1 | +0.0000 | +0.0000 | +0.0460 | +0.0115 | 0.1196 → 0.0866 |
| V8-TUNE-U-RPS-lr2e-05 | 1 | 1 | 2 | 1 | +0.0000 | +0.0000 | +0.0586 | +0.0138 | 0.1204 → 0.0837 |
| V8-TUNE-U-RPS-lr5e-06 | 1 | 0 | 1 | 1 | -0.0010 | +0.0368 | +0.0368 | +0.0130 | 0.1207 → 0.0912 |

## 4. Findings (counts and ranges computed from the same file)

- Selected-epoch distribution over the 48 units (0-based index = completed epochs − 1): {0: 10, 1: 18, 2: 16, 3: 2, 4: 2}; formal units only: {0: 8, 1: 3, 2: 10, 3: 2, 4: 1}.  Epoch index 0 is the checkpoint after the first full pass (2,379 steps, 26,177 images), never an untrained head (V81_01).
- The native training loss decreases monotonically over the five epochs in 48 / 48 units; the CAL_SELECT mean-rating MSE rises again after the selected epoch in 46 / 48 units (last − selected from +0.000 to +0.383; trained arms' seed means +0.003 to +0.016, U-RANK +0.348).  FIT keeps improving while the CAL mean-rating error has passed its minimum: the frozen rule stops early by design.
- Pair-accuracy criterion vs the frozen MSE criterion: the argmax epoch differs in 29 / 48 units, but the pair accuracy forgone at the selected epoch is at most 0.019 (range -0.019 to +0.000); choosing the pair-accuracy epoch would cost 0.000 to 0.130 in CAL mean-rating MSE.  RPS argmin differs in 18 / 48, CE argmin in 19 / 48.  Across the formal trained arms the seed-mean pair accuracy moves within 0.712-0.725 over all epochs and arms.  Reading: the log shows a small possible misalignment (≤ 0.02 pair accuracy) between the MSE rule and a ranking criterion on CAL_SELECT; it is not evidence of an external gain and no checkpoint other than best / resume exists to test it (level 2 stops here).
- U-RANK (formal): CAL mean-rating MSE 0.518 / 0.579 / 0.604 at e0 rising to 0.901 / 0.906 / 0.937 at e4, while its CAL pair accuracy stays 0.712-0.723 (level with the other arms at every epoch) and its native loss falls 0.636 → 0.473 (point part 0.037 → 0.029, ranking part 0.599 → 0.444; 25.5 scored pairs and 2.0 subgroups per step).  Log-level reading: the ranking arm's within-prompt ordering is intact; what the MSE rule penalises is the scale / level of mu, which drifts as the RankNet term spreads the mu differences (tau = 1).  'U-RANK failed' in V8-04 is a statement about the mean-rating MSE rule, not about its native ordering; the GenAI-Rank N9 result (rank-based) was level with the other arms for the same reason.  Whether this is a scale-only difference is D03's question (V81_03).
- Tuning units: with lr 5e-6 / 1e-5 / 2e-5 the selected epoch is 1 or 2 for every trained arm (table 2) and the epoch-0 → selected MSE drop is 0.014-0.062 (seed 17), larger than in the formal seeds (0.009-0.023): seed 17's first epoch starts from a worse CAL MSE (0.44-0.46 vs 0.40-0.44), so the single tuning seed is a noisier basis for the lr rule than the three formal seeds.  No lr-grid extension is proposed (owner decision 2026-10-15).
- Not claimed: any DEV / GenAI number, any 'overfitting' beyond the CAL mean-rating MSE curve, any per-epoch per-image curve (weights not saved: only best.pt and the final resume.pt exist per unit).
