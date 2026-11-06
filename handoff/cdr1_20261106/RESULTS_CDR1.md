Publication copy (sample names aliased; numbers and conclusions unchanged).

# RESULTS_CDR1

Protocol `CDR1_CONTAMINATION_CALIBRATION_SELECTIVE_RETRIEVAL_V1`; evidence base a55332645aed5b23b7d90f92e662e9242652efd4. Frozen unadapted SigLIP2 (google/siglip2-base-patch16-224 @75de2d55) per-item-L2 global cosine; two scalars per fit; ranking never changed. Development assay built from canonical FIT only; TEST not read; old Flickr DEV not used. All numbers below are generated from the ledgers in this directory. Zero coverage is reported as risk=NA, never 0.

## 0. Execution status

```
{
 "protocol_id": "CDR1_CONTAMINATION_CALIBRATION_SELECTIVE_RETRIEVAL_V1",
 "cases_curated": 24,
 "core_fits_present": 84,
 "core_fits_registered": 84,
 "profile_new_fits": 96,
 "anchor_fits": 24,
 "prediction_metric_rows": 7488,
 "d1_rows": 384,
 "groups_done": [
  "Flickr30K_d1001",
  "Flickr30K_d2002",
  "Flickr30K_d3003",
  "COCO_d1001",
  "COCO_d2002",
  "COCO_d3003"
 ],
 "transfers_done": [
  1001,
  2002,
  3003
 ],
 "ranking_integrity_all_identical": true,
 "DEDPUL": "PENDING (SITE_ADAPTER_REQUIRED; not integrated this round; no comparison claimed)",
 "MA_PaPSP": "NOT_IMPLEMENTED (literature only)",
 "NACP": "NOT_IMPLEMENTED (literature only)",
 "D3_OT": "NOT_RUN (out of scope)",
 "D4": "NA (no saved artifacts of the required quantities read this round)",
 "TEST_read": false,
 "encoder_updates": 0
}
```

## 1. Data facts

| case (seed 1001 shown; all in CASE_LEDGER) | status | beta_fit | mixture valid | FIT_CAL groups | EVAL groups |
|---|---|---|---|---|---|
| Flickr30K_independent_r00_d1001 | CURATED | 0.0001 | True | 17399 | 2900 |
| Flickr30K_independent_r60_d1001 | CURATED | 0.0001 | True | 17399 | 2900 |
| Flickr30K_independent_r80_d1001 | CURATED | 0.0001 | True | 17399 | 2900 |
| Flickr30K_hard_r60_d1001 | CURATED | 0.0001 | False | 17399 | 2900 |
| COCO_independent_r00_d1001 | CURATED | 0.0000 | True | 67964 | 11327 |
| COCO_independent_r60_d1001 | CURATED | 0.0000 | True | 67964 | 11327 |
| COCO_independent_r80_d1001 | CURATED | 0.0000 | True | 67964 | 11327 |
| COCO_hard_r60_d1001 | CURATED | 0.0000 | False | 67964 | 11327 |

Every query stays in the denominator; present/absent counts per gallery are in CASE_LEDGER.csv. CC152K_RECOVERED is a secondary transfer target only (693 DEV pairs; no true rho, no clean oracle).

## 2. Model fits (theory condition: two-parameter monotone family; J and BCE fitted on identical inputs/maps)

Core fits present 84/84; selected converged 84; at bound 0. log E_Q r (should be ~0 if the family fits) per objective/regime:

| objective/regime | n | mean logE_Q r | min | max |
|---|---|---|---|---|
| BCE/known-injection | 18 | 1.1725 | -0.5734 | 5.1824 |
| BCE/naive | 24 | 0.3695 | -0.7689 | 2.6807 |
| J/known-injection | 18 | 3.2096 | -0.6025 | 14.5987 |
| J/naive | 24 | 0.6630 | -0.6977 | 4.2426 |

## 3. D1 mechanism-source probability (EVAL_OPEN clean-P vs independent-Q; not semantic correctness)

| case type | method | n seeds | Brier | NLL | ECE15 | post-proj frac |
|---|---|---|---|---|---|---|
| COCO_hard_r60 | CLEAN_BCE_ORACLE | 3 | 0.0102 | 0.0401 | 0.0024 | 0.0000 |
| COCO_hard_r60 | CLEAN_J_ORACLE | 3 | 0.0102 | 0.0403 | 0.0023 | 0.0000 |
| COCO_hard_r60 | FORWARD_BCE | 3 | 0.0102 | 0.0417 | 0.0036 | 0.0000 |
| COCO_hard_r60 | FORWARD_J | 3 | 0.0109 | 0.0525 | 0.0079 | 0.0000 |
| COCO_hard_r60 | NAIVE_BCE | 3 | 0.0104 | 0.0407 | 0.0034 | 0.0000 |
| COCO_hard_r60 | NAIVE_J | 3 | 0.0103 | 0.0413 | 0.0038 | 0.0000 |
| COCO_hard_r60 | POST_PROJECTED_BCE | 3 | 0.0116 | 0.1493 | 0.0082 | 0.4955 |
| COCO_hard_r60 | POST_PROJECTED_J | 3 | 0.0116 | 0.1492 | 0.0089 | 0.4954 |
| COCO_independent_r00 | CLEAN_BCE_ORACLE | 3 | 0.0102 | 0.0401 | 0.0024 | 0.0000 |
| COCO_independent_r00 | CLEAN_J_ORACLE | 3 | 0.0102 | 0.0403 | 0.0023 | 0.0000 |
| COCO_independent_r00 | FORWARD_BCE | 3 | 0.0102 | 0.0401 | 0.0024 | 0.0000 |
| COCO_independent_r00 | FORWARD_J | 3 | 0.0102 | 0.0403 | 0.0023 | 0.0000 |
| COCO_independent_r00 | NAIVE_BCE | 3 | 0.0102 | 0.0401 | 0.0024 | 0.0000 |
| COCO_independent_r00 | NAIVE_J | 3 | 0.0102 | 0.0403 | 0.0023 | 0.0000 |
| COCO_independent_r00 | POST_PROJECTED_BCE | 3 | 0.0102 | 0.0401 | 0.0024 | 0.0000 |
| COCO_independent_r00 | POST_PROJECTED_J | 3 | 0.0102 | 0.0403 | 0.0023 | 0.0000 |
| COCO_independent_r60 | CLEAN_BCE_ORACLE | 3 | 0.0102 | 0.0401 | 0.0024 | 0.0000 |
| COCO_independent_r60 | CLEAN_J_ORACLE | 3 | 0.0102 | 0.0403 | 0.0023 | 0.0000 |
| COCO_independent_r60 | FORWARD_BCE | 3 | 0.0102 | 0.0407 | 0.0035 | 0.0000 |
| COCO_independent_r60 | FORWARD_J | 3 | 0.0102 | 0.0403 | 0.0026 | 0.0000 |
| COCO_independent_r60 | NAIVE_BCE | 3 | 0.1026 | 0.3517 | 0.2577 | 0.0000 |
| COCO_independent_r60 | NAIVE_J | 3 | 0.1017 | 0.3598 | 0.2682 | 0.0000 |
| COCO_independent_r60 | POST_PROJECTED_BCE | 3 | 0.0706 | 0.2262 | 0.1439 | 0.2149 |
| COCO_independent_r60 | POST_PROJECTED_J | 3 | 0.0590 | 0.2051 | 0.1352 | 0.2313 |
| COCO_independent_r80 | CLEAN_BCE_ORACLE | 3 | 0.0102 | 0.0401 | 0.0024 | 0.0000 |
| COCO_independent_r80 | CLEAN_J_ORACLE | 3 | 0.0102 | 0.0403 | 0.0023 | 0.0000 |
| COCO_independent_r80 | FORWARD_BCE | 3 | 0.0103 | 0.0411 | 0.0041 | 0.0000 |
| COCO_independent_r80 | FORWARD_J | 3 | 0.0102 | 0.0406 | 0.0032 | 0.0000 |
| COCO_independent_r80 | NAIVE_BCE | 3 | 0.1358 | 0.4400 | 0.3201 | 0.0000 |
| COCO_independent_r80 | NAIVE_J | 3 | 0.1397 | 0.4537 | 0.3359 | 0.0000 |
| COCO_independent_r80 | POST_PROJECTED_BCE | 3 | 0.0976 | 0.2893 | 0.1677 | 0.2084 |
| COCO_independent_r80 | POST_PROJECTED_J | 3 | 0.0882 | 0.2712 | 0.1692 | 0.2127 |
| Flickr30K_hard_r60 | CLEAN_BCE_ORACLE | 3 | 0.0115 | 0.0438 | 0.0020 | 0.0000 |
| Flickr30K_hard_r60 | CLEAN_J_ORACLE | 3 | 0.0115 | 0.0440 | 0.0024 | 0.0000 |
| Flickr30K_hard_r60 | FORWARD_BCE | 3 | 0.0132 | 0.0538 | 0.0103 | 0.0000 |
| Flickr30K_hard_r60 | FORWARD_J | 3 | 0.0131 | 0.0676 | 0.0112 | 0.0000 |
| Flickr30K_hard_r60 | NAIVE_BCE | 3 | 0.0124 | 0.0478 | 0.0077 | 0.0000 |
| Flickr30K_hard_r60 | NAIVE_J | 3 | 0.0125 | 0.0495 | 0.0084 | 0.0000 |
| Flickr30K_hard_r60 | POST_PROJECTED_BCE | 3 | 0.0135 | 0.1367 | 0.0124 | 0.4893 |
| Flickr30K_hard_r60 | POST_PROJECTED_J | 3 | 0.0137 | 0.1403 | 0.0129 | 0.4895 |
| Flickr30K_independent_r00 | CLEAN_BCE_ORACLE | 3 | 0.0115 | 0.0438 | 0.0020 | 0.0000 |
| Flickr30K_independent_r00 | CLEAN_J_ORACLE | 3 | 0.0115 | 0.0440 | 0.0024 | 0.0000 |
| Flickr30K_independent_r00 | FORWARD_BCE | 3 | 0.0115 | 0.0438 | 0.0020 | 0.0000 |
| Flickr30K_independent_r00 | FORWARD_J | 3 | 0.0115 | 0.0440 | 0.0024 | 0.0000 |
| Flickr30K_independent_r00 | NAIVE_BCE | 3 | 0.0115 | 0.0438 | 0.0020 | 0.0000 |
| Flickr30K_independent_r00 | NAIVE_J | 3 | 0.0115 | 0.0440 | 0.0024 | 0.0000 |
| Flickr30K_independent_r00 | POST_PROJECTED_BCE | 3 | 0.0115 | 0.0438 | 0.0020 | 0.0000 |
| Flickr30K_independent_r00 | POST_PROJECTED_J | 3 | 0.0115 | 0.0440 | 0.0024 | 0.0000 |
| Flickr30K_independent_r60 | CLEAN_BCE_ORACLE | 3 | 0.0115 | 0.0438 | 0.0020 | 0.0000 |
| Flickr30K_independent_r60 | CLEAN_J_ORACLE | 3 | 0.0115 | 0.0440 | 0.0024 | 0.0000 |
| Flickr30K_independent_r60 | FORWARD_BCE | 3 | 0.0115 | 0.0440 | 0.0021 | 0.0000 |
| Flickr30K_independent_r60 | FORWARD_J | 3 | 0.0115 | 0.0440 | 0.0023 | 0.0000 |
| Flickr30K_independent_r60 | NAIVE_BCE | 3 | 0.1065 | 0.3638 | 0.2675 | 0.0000 |
| Flickr30K_independent_r60 | NAIVE_J | 3 | 0.1047 | 0.3685 | 0.2759 | 0.0000 |
| Flickr30K_independent_r60 | POST_PROJECTED_BCE | 3 | 0.0755 | 0.2401 | 0.1546 | 0.2090 |
| Flickr30K_independent_r60 | POST_PROJECTED_J | 3 | 0.0631 | 0.2167 | 0.1460 | 0.2289 |
| Flickr30K_independent_r80 | CLEAN_BCE_ORACLE | 3 | 0.0115 | 0.0438 | 0.0020 | 0.0000 |
| Flickr30K_independent_r80 | CLEAN_J_ORACLE | 3 | 0.0115 | 0.0440 | 0.0024 | 0.0000 |
| Flickr30K_independent_r80 | FORWARD_BCE | 3 | 0.0116 | 0.0442 | 0.0032 | 0.0000 |
| Flickr30K_independent_r80 | FORWARD_J | 3 | 0.0116 | 0.0444 | 0.0031 | 0.0000 |
| Flickr30K_independent_r80 | NAIVE_BCE | 3 | 0.1412 | 0.4551 | 0.3341 | 0.0000 |
| Flickr30K_independent_r80 | NAIVE_J | 3 | 0.1447 | 0.4668 | 0.3423 | 0.0000 |
| Flickr30K_independent_r80 | POST_PROJECTED_BCE | 3 | 0.1022 | 0.3025 | 0.1836 | 0.2018 |
| Flickr30K_independent_r80 | POST_PROJECTED_J | 3 | 0.0930 | 0.2846 | 0.1796 | 0.2070 |

lambda_pair=0.1 rows are in D1_TABLE.csv. CLEAN_*_ORACLE rows use extra clean information (rho=0 fit of the same seed).

## 4. D2 selective text->image retrieval, same domain (known-rho regime K; oracle rows = information regime O)

Mean over 3 data seeds at M=128, pi=0.5 (full grid M x pi in DECISION_LEDGER.csv). Selected = model-based p_best_correct; risk = accepted error rate.

| case | method | n | cov@.90 | risk@.90 | NA seeds | FAR@.90 | cov@.95 | risk@.95 | FAR@.95 | cost@.90 | Brier(C) | MAP acc | naiveTop1 cov@.90 | naiveTop1 risk@.90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| COCO_hard_r60 | CLEAN_BCE_ORACLE | 3 | 0.2166 | 0.0391 | 0 | 0.0087 | 0.1719 | 0.0302 | 0.0052 | 0.0866 | 0.1171 | 0.7904 | 0.7096 | 0.3902 |
| COCO_hard_r60 | CLEAN_J_ORACLE | 3 | 0.2487 | 0.0483 | 0 | 0.0156 | 0.1997 | 0.0350 | 0.0061 | 0.0868 | 0.1190 | 0.7921 | 0.7244 | 0.4008 |
| COCO_hard_r60 | FORWARD_BCE | 3 | 0.3151 | 0.0936 | 0 | 0.0469 | 0.2726 | 0.0614 | 0.0217 | 0.0971 | 0.1319 | 0.7739 | 0.7391 | 0.4131 |
| COCO_hard_r60 | FORWARD_J | 3 | 0.4696 | 0.2386 | 0 | 0.1762 | 0.4401 | 0.2055 | 0.1354 | 0.1646 | 0.1925 | 0.7409 | 0.7578 | 0.4260 |
| COCO_hard_r60 | NAIVE_BCE | 3 | 0.2543 | 0.0446 | 0 | 0.0139 | 0.2057 | 0.0315 | 0.0078 | 0.0854 | 0.1192 | 0.7947 | 0.7053 | 0.3864 |
| COCO_hard_r60 | NAIVE_J | 3 | 0.2925 | 0.0769 | 0 | 0.0330 | 0.2483 | 0.0473 | 0.0148 | 0.0924 | 0.1256 | 0.7830 | 0.7231 | 0.4000 |
| COCO_hard_r60 | POST_PROJECTED_BCE | 3 | 0.2943 | 0.0730 | 0 | 0.0312 | 0.2504 | 0.0505 | 0.0165 | 0.0914 | 0.1280 | 0.7739 | 0.7465 | 0.4173 |
| COCO_hard_r60 | POST_PROJECTED_J | 3 | 0.3433 | 0.1087 | 0 | 0.0590 | 0.2921 | 0.0653 | 0.0269 | 0.1017 | 0.1398 | 0.7574 | 0.7591 | 0.4269 |
| COCO_independent_r00 | CLEAN_BCE_ORACLE | 3 | 0.2166 | 0.0391 | 0 | 0.0087 | 0.1719 | 0.0302 | 0.0052 | 0.0866 | 0.1171 | 0.7904 | 0.7096 | 0.3902 |
| COCO_independent_r00 | CLEAN_J_ORACLE | 3 | 0.2487 | 0.0483 | 0 | 0.0156 | 0.1997 | 0.0350 | 0.0061 | 0.0868 | 0.1190 | 0.7921 | 0.7244 | 0.4008 |
| COCO_independent_r00 | FORWARD_BCE | 3 | 0.2166 | 0.0391 | 0 | 0.0087 | 0.1719 | 0.0302 | 0.0052 | 0.0866 | 0.1171 | 0.7904 | 0.7096 | 0.3902 |
| COCO_independent_r00 | FORWARD_J | 3 | 0.2487 | 0.0483 | 0 | 0.0156 | 0.1997 | 0.0350 | 0.0061 | 0.0868 | 0.1190 | 0.7921 | 0.7244 | 0.4008 |
| COCO_independent_r00 | NAIVE_BCE | 3 | 0.2166 | 0.0391 | 0 | 0.0087 | 0.1719 | 0.0302 | 0.0052 | 0.0866 | 0.1171 | 0.7904 | 0.7096 | 0.3902 |
| COCO_independent_r00 | NAIVE_J | 3 | 0.2487 | 0.0483 | 0 | 0.0156 | 0.1997 | 0.0350 | 0.0061 | 0.0868 | 0.1190 | 0.7921 | 0.7244 | 0.4008 |
| COCO_independent_r00 | POST_PROJECTED_BCE | 3 | 0.2166 | 0.0391 | 0 | 0.0087 | 0.1719 | 0.0302 | 0.0052 | 0.0866 | 0.1171 | 0.7904 | 0.7096 | 0.3902 |
| COCO_independent_r00 | POST_PROJECTED_J | 3 | 0.2487 | 0.0483 | 0 | 0.0156 | 0.1997 | 0.0350 | 0.0061 | 0.0868 | 0.1190 | 0.7921 | 0.7244 | 0.4008 |
| COCO_independent_r60 | CLEAN_BCE_ORACLE | 3 | 0.2166 | 0.0391 | 0 | 0.0087 | 0.1719 | 0.0302 | 0.0052 | 0.0866 | 0.1171 | 0.7904 | 0.7096 | 0.3902 |
| COCO_independent_r60 | CLEAN_J_ORACLE | 3 | 0.2487 | 0.0483 | 0 | 0.0156 | 0.1997 | 0.0350 | 0.0061 | 0.0868 | 0.1190 | 0.7921 | 0.7244 | 0.4008 |
| COCO_independent_r60 | FORWARD_BCE | 3 | 0.1905 | 0.0296 | 0 | 0.0061 | 0.1380 | 0.0096 | 0.0026 | 0.0866 | 0.1168 | 0.7921 | 0.7118 | 0.3920 |
| COCO_independent_r60 | FORWARD_J | 3 | 0.2083 | 0.0386 | 0 | 0.0078 | 0.1558 | 0.0224 | 0.0043 | 0.0870 | 0.1165 | 0.7925 | 0.7135 | 0.3934 |
| COCO_independent_r60 | NAIVE_BCE | 3 | 0.0000 | NA | 3 | 0.0000 | 0.0000 | NA | 0.0000 | 0.1000 | 0.4113 | 0.5000 | 0.1133 | 0.0000 |
| COCO_independent_r60 | NAIVE_J | 3 | 0.0000 | NA | 3 | 0.0000 | 0.0000 | NA | 0.0000 | 0.1000 | 0.4185 | 0.5000 | 0.0195 | 0.0000 |
| COCO_independent_r60 | POST_PROJECTED_BCE | 3 | 0.0000 | NA | 3 | 0.0000 | 0.0000 | NA | 0.0000 | 0.1000 | 0.3724 | 0.5000 | 0.6146 | 0.3322 |
| COCO_independent_r60 | POST_PROJECTED_J | 3 | 0.0000 | NA | 3 | 0.0000 | 0.0000 | NA | 0.0000 | 0.1000 | 0.3870 | 0.5000 | 0.4132 | 0.1945 |
| COCO_independent_r80 | CLEAN_BCE_ORACLE | 3 | 0.2166 | 0.0391 | 0 | 0.0087 | 0.1719 | 0.0302 | 0.0052 | 0.0866 | 0.1171 | 0.7904 | 0.7096 | 0.3902 |
| COCO_independent_r80 | CLEAN_J_ORACLE | 3 | 0.2487 | 0.0483 | 0 | 0.0156 | 0.1997 | 0.0350 | 0.0061 | 0.0868 | 0.1190 | 0.7921 | 0.7244 | 0.4008 |
| COCO_independent_r80 | FORWARD_BCE | 3 | 0.1849 | 0.0280 | 0 | 0.0052 | 0.1272 | 0.0072 | 0.0017 | 0.0867 | 0.1168 | 0.7925 | 0.7153 | 0.3951 |
| COCO_independent_r80 | FORWARD_J | 3 | 0.1988 | 0.0301 | 0 | 0.0069 | 0.1458 | 0.0090 | 0.0026 | 0.0862 | 0.1165 | 0.7943 | 0.7127 | 0.3927 |
| COCO_independent_r80 | NAIVE_BCE | 3 | 0.0000 | NA | 3 | 0.0000 | 0.0000 | NA | 0.0000 | 0.1000 | 0.4253 | 0.5000 | 0.0000 | NA |
| COCO_independent_r80 | NAIVE_J | 3 | 0.0000 | NA | 3 | 0.0000 | 0.0000 | NA | 0.0000 | 0.1000 | 0.4277 | 0.5000 | 0.0000 | NA |
| COCO_independent_r80 | POST_PROJECTED_BCE | 3 | 0.0000 | NA | 3 | 0.0000 | 0.0000 | NA | 0.0000 | 0.1000 | 0.3829 | 0.5000 | 0.7092 | 0.3901 |
| COCO_independent_r80 | POST_PROJECTED_J | 3 | 0.0000 | NA | 3 | 0.0000 | 0.0000 | NA | 0.0000 | 0.1000 | 0.3909 | 0.5000 | 0.5642 | 0.2962 |
| Flickr30K_hard_r60 | CLEAN_BCE_ORACLE | 3 | 0.2682 | 0.0143 | 0 | 0.0078 | 0.2057 | 0.0057 | 0.0026 | 0.0771 | 0.0740 | 0.8624 | 0.6667 | 0.3105 |
| Flickr30K_hard_r60 | CLEAN_J_ORACLE | 3 | 0.2756 | 0.0167 | 0 | 0.0095 | 0.2309 | 0.0071 | 0.0035 | 0.0772 | 0.0730 | 0.8624 | 0.6706 | 0.3126 |
| Flickr30K_hard_r60 | FORWARD_BCE | 3 | 0.4484 | 0.0909 | 0 | 0.0738 | 0.4162 | 0.0708 | 0.0512 | 0.0960 | 0.1111 | 0.8134 | 0.7726 | 0.3983 |
| Flickr30K_hard_r60 | FORWARD_J | 3 | 0.5082 | 0.2134 | 0 | 0.2014 | 0.4874 | 0.1878 | 0.1675 | 0.1577 | 0.1811 | 0.7509 | 0.7860 | 0.4086 |
| Flickr30K_hard_r60 | NAIVE_BCE | 3 | 0.3720 | 0.0302 | 0 | 0.0200 | 0.3303 | 0.0182 | 0.0122 | 0.0741 | 0.0801 | 0.8503 | 0.7348 | 0.3691 |
| Flickr30K_hard_r60 | NAIVE_J | 3 | 0.4128 | 0.0564 | 0 | 0.0417 | 0.3767 | 0.0318 | 0.0217 | 0.0822 | 0.0909 | 0.8424 | 0.7474 | 0.3780 |
| Flickr30K_hard_r60 | POST_PROJECTED_BCE | 3 | 0.4227 | 0.0666 | 0 | 0.0512 | 0.3841 | 0.0405 | 0.0260 | 0.0859 | 0.1008 | 0.8173 | 0.7773 | 0.4020 |
| Flickr30K_hard_r60 | POST_PROJECTED_J | 3 | 0.4540 | 0.0878 | 0 | 0.0720 | 0.4149 | 0.0708 | 0.0512 | 0.0945 | 0.1158 | 0.7986 | 0.7852 | 0.4079 |
| Flickr30K_independent_r00 | CLEAN_BCE_ORACLE | 3 | 0.2682 | 0.0143 | 0 | 0.0078 | 0.2057 | 0.0057 | 0.0026 | 0.0771 | 0.0740 | 0.8624 | 0.6667 | 0.3105 |
| Flickr30K_independent_r00 | CLEAN_J_ORACLE | 3 | 0.2756 | 0.0167 | 0 | 0.0095 | 0.2309 | 0.0071 | 0.0035 | 0.0772 | 0.0730 | 0.8624 | 0.6706 | 0.3126 |
| Flickr30K_independent_r00 | FORWARD_BCE | 3 | 0.2682 | 0.0143 | 0 | 0.0078 | 0.2057 | 0.0057 | 0.0026 | 0.0771 | 0.0740 | 0.8624 | 0.6667 | 0.3105 |
| Flickr30K_independent_r00 | FORWARD_J | 3 | 0.2756 | 0.0167 | 0 | 0.0095 | 0.2309 | 0.0071 | 0.0035 | 0.0772 | 0.0730 | 0.8624 | 0.6706 | 0.3126 |
| Flickr30K_independent_r00 | NAIVE_BCE | 3 | 0.2682 | 0.0143 | 0 | 0.0078 | 0.2057 | 0.0057 | 0.0026 | 0.0771 | 0.0740 | 0.8624 | 0.6667 | 0.3105 |
| Flickr30K_independent_r00 | NAIVE_J | 3 | 0.2756 | 0.0167 | 0 | 0.0095 | 0.2309 | 0.0071 | 0.0035 | 0.0772 | 0.0730 | 0.8624 | 0.6706 | 0.3126 |
| Flickr30K_independent_r00 | POST_PROJECTED_BCE | 3 | 0.2682 | 0.0143 | 0 | 0.0078 | 0.2057 | 0.0057 | 0.0026 | 0.0771 | 0.0740 | 0.8624 | 0.6667 | 0.3105 |
| Flickr30K_independent_r00 | POST_PROJECTED_J | 3 | 0.2756 | 0.0167 | 0 | 0.0095 | 0.2309 | 0.0071 | 0.0035 | 0.0772 | 0.0730 | 0.8624 | 0.6706 | 0.3126 |
| Flickr30K_independent_r60 | CLEAN_BCE_ORACLE | 3 | 0.2682 | 0.0143 | 0 | 0.0078 | 0.2057 | 0.0057 | 0.0026 | 0.0771 | 0.0740 | 0.8624 | 0.6667 | 0.3105 |
| Flickr30K_independent_r60 | CLEAN_J_ORACLE | 3 | 0.2756 | 0.0167 | 0 | 0.0095 | 0.2309 | 0.0071 | 0.0035 | 0.0772 | 0.0730 | 0.8624 | 0.6706 | 0.3126 |
| Flickr30K_independent_r60 | FORWARD_BCE | 3 | 0.2760 | 0.0140 | 0 | 0.0078 | 0.2244 | 0.0075 | 0.0035 | 0.0763 | 0.0729 | 0.8646 | 0.6697 | 0.3117 |
| Flickr30K_independent_r60 | FORWARD_J | 3 | 0.2808 | 0.0167 | 0 | 0.0095 | 0.2361 | 0.0074 | 0.0035 | 0.0767 | 0.0727 | 0.8655 | 0.6719 | 0.3120 |
| Flickr30K_independent_r60 | NAIVE_BCE | 3 | 0.0000 | NA | 3 | 0.0000 | 0.0000 | NA | 0.0000 | 0.1000 | 0.4417 | 0.5000 | 0.0690 | 0.0208 |
| Flickr30K_independent_r60 | NAIVE_J | 3 | 0.0000 | NA | 3 | 0.0000 | 0.0000 | NA | 0.0000 | 0.1000 | 0.4470 | 0.5000 | 0.0156 | 0.0000 |
| Flickr30K_independent_r60 | POST_PROJECTED_BCE | 3 | 0.0000 | NA | 3 | 0.0000 | 0.0000 | NA | 0.0000 | 0.1000 | 0.4066 | 0.5000 | 0.4727 | 0.1320 |
| Flickr30K_independent_r60 | POST_PROJECTED_J | 3 | 0.0000 | NA | 3 | 0.0000 | 0.0000 | NA | 0.0000 | 0.1000 | 0.4175 | 0.5000 | 0.2912 | 0.0367 |
| Flickr30K_independent_r80 | CLEAN_BCE_ORACLE | 3 | 0.2682 | 0.0143 | 0 | 0.0078 | 0.2057 | 0.0057 | 0.0026 | 0.0771 | 0.0740 | 0.8624 | 0.6667 | 0.3105 |
| Flickr30K_independent_r80 | CLEAN_J_ORACLE | 3 | 0.2756 | 0.0167 | 0 | 0.0095 | 0.2309 | 0.0071 | 0.0035 | 0.0772 | 0.0730 | 0.8624 | 0.6706 | 0.3126 |
| Flickr30K_independent_r80 | FORWARD_BCE | 3 | 0.2734 | 0.0142 | 0 | 0.0078 | 0.2179 | 0.0078 | 0.0035 | 0.0766 | 0.0731 | 0.8655 | 0.6723 | 0.3124 |
| Flickr30K_independent_r80 | FORWARD_J | 3 | 0.2726 | 0.0112 | 0 | 0.0061 | 0.2201 | 0.0061 | 0.0026 | 0.0758 | 0.0731 | 0.8659 | 0.6732 | 0.3133 |
| Flickr30K_independent_r80 | NAIVE_BCE | 3 | 0.0000 | NA | 3 | 0.0000 | 0.0000 | NA | 0.0000 | 0.1000 | 0.4539 | 0.5000 | 0.0000 | NA |
| Flickr30K_independent_r80 | NAIVE_J | 3 | 0.0000 | NA | 3 | 0.0000 | 0.0000 | NA | 0.0000 | 0.1000 | 0.4557 | 0.5000 | 0.0000 | NA |
| Flickr30K_independent_r80 | POST_PROJECTED_BCE | 3 | 0.0000 | NA | 3 | 0.0000 | 0.0000 | NA | 0.0000 | 0.1000 | 0.4161 | 0.5000 | 0.5430 | 0.1833 |
| Flickr30K_independent_r80 | POST_PROJECTED_J | 3 | 0.0000 | NA | 3 | 0.0000 | 0.0000 | NA | 0.0000 | 0.1000 | 0.4222 | 0.5000 | 0.3980 | 0.0697 |

## 5. J vs matched forward-BCE (same contamination map) and correction effects

Per-seed differences A−B (same case/target/M/pi), averaged over 3 seeds x 3 M x 3 pi = 27 cells; dRisk averages only cells where both risks are defined (non-zero coverage).

| scope / case / target / A / B | cells | dBrier(C) | dNLL(C) | dCost@.90 | dCov@.90 | dRisk@.90 (defined cells) | dMAP acc |
|---|---|---|---|---|---|---|---|
| same_domain / COCO_hard_r60 / COCO / FORWARD_BCE / NAIVE_BCE | 27 | 0.0086 | 0.0736 | 0.0077 | 0.0652 | 0.0477 | -0.0149 |
| same_domain / COCO_hard_r60 / COCO / FORWARD_BCE / POST_PROJECTED_BCE | 27 | 0.0043 | 0.0434 | 0.0035 | 0.0201 | 0.0195 | -0.0021 |
| same_domain / COCO_hard_r60 / COCO / FORWARD_J / FORWARD_BCE | 27 | 0.0601 | 1.2298 | 0.0585 | 0.1382 | 0.1473 | -0.0477 |
| same_domain / COCO_hard_r60 / COCO / FORWARD_J / NAIVE_J | 27 | 0.0647 | 1.2627 | 0.0624 | 0.1646 | 0.1662 | -0.0566 |
| same_domain / COCO_hard_r60 / COCO / FORWARD_J / POST_PROJECTED_J | 27 | 0.0538 | 1.1628 | 0.0549 | 0.1210 | 0.1331 | -0.0365 |
| same_domain / COCO_hard_r60 / COCO / NAIVE_J / NAIVE_BCE | 27 | 0.0041 | 0.0407 | 0.0038 | 0.0388 | 0.0288 | -0.0060 |
| same_domain / COCO_hard_r60 / COCO / POST_PROJECTED_J / POST_PROJECTED_BCE | 27 | 0.0106 | 0.1104 | 0.0071 | 0.0373 | 0.0337 | -0.0133 |
| same_domain / COCO_independent_r00 / COCO / FORWARD_BCE / NAIVE_BCE | 27 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| same_domain / COCO_independent_r00 / COCO / FORWARD_BCE / POST_PROJECTED_BCE | 27 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| same_domain / COCO_independent_r00 / COCO / FORWARD_J / FORWARD_BCE | 27 | 0.0001 | 0.0115 | -0.0001 | 0.0260 | 0.0105 | -0.0015 |
| same_domain / COCO_independent_r00 / COCO / FORWARD_J / NAIVE_J | 27 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| same_domain / COCO_independent_r00 / COCO / FORWARD_J / POST_PROJECTED_J | 27 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| same_domain / COCO_independent_r00 / COCO / NAIVE_J / NAIVE_BCE | 27 | 0.0001 | 0.0115 | -0.0001 | 0.0260 | 0.0105 | -0.0015 |
| same_domain / COCO_independent_r00 / COCO / POST_PROJECTED_J / POST_PROJECTED_BCE | 27 | 0.0001 | 0.0115 | -0.0001 | 0.0260 | 0.0105 | -0.0015 |
| same_domain / COCO_independent_r60 / COCO / FORWARD_BCE / NAIVE_BCE | 27 | -0.2812 | -1.1318 | -0.0157 | 0.2076 | NA | 0.3023 |
| same_domain / COCO_independent_r60 / COCO / FORWARD_BCE / POST_PROJECTED_BCE | 27 | -0.2238 | -0.7662 | -0.0157 | 0.2076 | NA | 0.2435 |
| same_domain / COCO_independent_r60 / COCO / FORWARD_J / FORWARD_BCE | 27 | -0.0016 | -0.0001 | 0.0007 | 0.0163 | 0.0135 | 0.0010 |
| same_domain / COCO_independent_r60 / COCO / FORWARD_J / NAIVE_J | 27 | -0.2940 | -1.2441 | -0.0150 | 0.2238 | NA | 0.3110 |
| same_domain / COCO_independent_r60 / COCO / FORWARD_J / POST_PROJECTED_J | 27 | -0.2407 | -0.8519 | -0.0150 | 0.2238 | NA | 0.2706 |
| same_domain / COCO_independent_r60 / COCO / NAIVE_J / NAIVE_BCE | 27 | 0.0113 | 0.1123 | 0.0000 | 0.0000 | NA | -0.0077 |
| same_domain / COCO_independent_r60 / COCO / POST_PROJECTED_J / POST_PROJECTED_BCE | 27 | 0.0154 | 0.0857 | 0.0000 | 0.0000 | NA | -0.0261 |
| same_domain / COCO_independent_r80 / COCO / FORWARD_BCE / NAIVE_BCE | 27 | -0.3051 | -1.4034 | -0.0158 | 0.2042 | NA | 0.3096 |
| same_domain / COCO_independent_r80 / COCO / FORWARD_BCE / POST_PROJECTED_BCE | 27 | -0.2379 | -0.8410 | -0.0158 | 0.2042 | NA | 0.2397 |
| same_domain / COCO_independent_r80 / COCO / FORWARD_J / FORWARD_BCE | 27 | -0.0011 | -0.0021 | 0.0002 | 0.0093 | 0.0063 | 0.0010 |
| same_domain / COCO_independent_r80 / COCO / FORWARD_J / NAIVE_J | 27 | -0.3105 | -1.4761 | -0.0156 | 0.2136 | NA | 0.3106 |
| same_domain / COCO_independent_r80 / COCO / FORWARD_J / POST_PROJECTED_J | 27 | -0.2480 | -0.8967 | -0.0156 | 0.2136 | NA | 0.2550 |
| same_domain / COCO_independent_r80 / COCO / NAIVE_J / NAIVE_BCE | 27 | 0.0043 | 0.0706 | 0.0000 | 0.0000 | NA | 0.0000 |
| same_domain / COCO_independent_r80 / COCO / POST_PROJECTED_J / POST_PROJECTED_BCE | 27 | 0.0089 | 0.0536 | 0.0000 | 0.0000 | NA | -0.0143 |
| same_domain / Flickr30K_hard_r60 / Flickr30K / FORWARD_BCE / NAIVE_BCE | 27 | 0.0271 | 0.4772 | 0.0196 | 0.0717 | 0.0645 | -0.0333 |
| same_domain / Flickr30K_hard_r60 / Flickr30K / FORWARD_BCE / POST_PROJECTED_BCE | 27 | 0.0113 | 0.2282 | 0.0118 | 0.0258 | 0.0341 | -0.0070 |
| same_domain / Flickr30K_hard_r60 / Flickr30K / FORWARD_J / FORWARD_BCE | 27 | 0.0639 | 1.2450 | 0.0479 | 0.0306 | 0.1083 | -0.0615 |
| same_domain / Flickr30K_hard_r60 / Flickr30K / FORWARD_J / NAIVE_J | 27 | 0.0819 | 1.5810 | 0.0611 | 0.0660 | 0.1480 | -0.0843 |
| same_domain / Flickr30K_hard_r60 / Flickr30K / FORWARD_J / POST_PROJECTED_J | 27 | 0.0614 | 1.1814 | 0.0487 | 0.0313 | 0.1106 | -0.0521 |
| same_domain / Flickr30K_hard_r60 / Flickr30K / NAIVE_J / NAIVE_BCE | 27 | 0.0091 | 0.1413 | 0.0065 | 0.0364 | 0.0247 | -0.0105 |
| same_domain / Flickr30K_hard_r60 / Flickr30K / POST_PROJECTED_J / POST_PROJECTED_BCE | 27 | 0.0138 | 0.2917 | 0.0110 | 0.0251 | 0.0318 | -0.0164 |
| same_domain / Flickr30K_independent_r00 / Flickr30K / FORWARD_BCE / NAIVE_BCE | 27 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| same_domain / Flickr30K_independent_r00 / Flickr30K / FORWARD_BCE / POST_PROJECTED_BCE | 27 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| same_domain / Flickr30K_independent_r00 / Flickr30K / FORWARD_J / FORWARD_BCE | 27 | -0.0011 | 0.0151 | 0.0001 | 0.0144 | 0.0052 | 0.0011 |
| same_domain / Flickr30K_independent_r00 / Flickr30K / FORWARD_J / NAIVE_J | 27 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| same_domain / Flickr30K_independent_r00 / Flickr30K / FORWARD_J / POST_PROJECTED_J | 27 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| same_domain / Flickr30K_independent_r00 / Flickr30K / NAIVE_J / NAIVE_BCE | 27 | -0.0011 | 0.0151 | 0.0001 | 0.0144 | 0.0052 | 0.0011 |
| same_domain / Flickr30K_independent_r00 / Flickr30K / POST_PROJECTED_J / POST_PROJECTED_BCE | 27 | -0.0011 | 0.0151 | 0.0001 | 0.0144 | 0.0052 | 0.0011 |
| same_domain / Flickr30K_independent_r60 / Flickr30K / FORWARD_BCE / NAIVE_BCE | 27 | -0.3395 | -1.3656 | -0.0238 | 0.2925 | NA | 0.3620 |
| same_domain / Flickr30K_independent_r60 / Flickr30K / FORWARD_BCE / POST_PROJECTED_BCE | 27 | -0.2895 | -0.9854 | -0.0238 | 0.2925 | NA | 0.3059 |
| same_domain / Flickr30K_independent_r60 / Flickr30K / FORWARD_J / FORWARD_BCE | 27 | -0.0004 | 0.0047 | 0.0006 | 0.0055 | 0.0052 | 0.0008 |
| same_domain / Flickr30K_independent_r60 / Flickr30K / FORWARD_J / NAIVE_J | 27 | -0.3479 | -1.4556 | -0.0231 | 0.2980 | NA | 0.3654 |
| same_domain / Flickr30K_independent_r60 / Flickr30K / FORWARD_J / POST_PROJECTED_J | 27 | -0.3009 | -1.0521 | -0.0231 | 0.2980 | NA | 0.3358 |
| same_domain / Flickr30K_independent_r60 / Flickr30K / NAIVE_J / NAIVE_BCE | 27 | 0.0080 | 0.0947 | 0.0000 | 0.0000 | NA | -0.0026 |
| same_domain / Flickr30K_independent_r60 / Flickr30K / POST_PROJECTED_J / POST_PROJECTED_BCE | 27 | 0.0109 | 0.0713 | 0.0000 | 0.0000 | NA | -0.0292 |
| same_domain / Flickr30K_independent_r80 / Flickr30K / FORWARD_BCE / NAIVE_BCE | 27 | -0.3602 | -1.6490 | -0.0238 | 0.2898 | NA | 0.3646 |
| same_domain / Flickr30K_independent_r80 / Flickr30K / FORWARD_BCE / POST_PROJECTED_BCE | 27 | -0.3039 | -1.0732 | -0.0238 | 0.2898 | NA | 0.3016 |
| same_domain / Flickr30K_independent_r80 / Flickr30K / FORWARD_J / FORWARD_BCE | 27 | -0.0000 | -0.0020 | 0.0001 | 0.0005 | 0.0004 | 0.0003 |
| same_domain / Flickr30K_independent_r80 / Flickr30K / FORWARD_J / NAIVE_J | 27 | -0.3633 | -1.7108 | -0.0238 | 0.2904 | NA | 0.3649 |
| same_domain / Flickr30K_independent_r80 / Flickr30K / FORWARD_J / POST_PROJECTED_J | 27 | -0.3103 | -1.1214 | -0.0238 | 0.2904 | NA | 0.3242 |
| same_domain / Flickr30K_independent_r80 / Flickr30K / NAIVE_J / NAIVE_BCE | 27 | 0.0030 | 0.0598 | 0.0000 | 0.0000 | NA | 0.0000 |
| same_domain / Flickr30K_independent_r80 / Flickr30K / POST_PROJECTED_J / POST_PROJECTED_BCE | 27 | 0.0064 | 0.0463 | 0.0000 | 0.0000 | NA | -0.0223 |
| transfer / COCO_hard_r60 / CC / FORWARD_BCE / NAIVE_BCE | 27 | -0.0177 | -0.0466 | -0.0028 | 0.0629 | 0.0166 | 0.0304 |
| transfer / COCO_hard_r60 / CC / FORWARD_BCE / POST_PROJECTED_BCE | 27 | -0.0001 | -0.0414 | -0.0000 | 0.0126 | 0.0084 | 0.0012 |
| transfer / COCO_hard_r60 / CC / FORWARD_J / FORWARD_BCE | 27 | 0.0113 | 0.2126 | 0.0128 | 0.1150 | 0.0705 | -0.0078 |
| transfer / COCO_hard_r60 / CC / FORWARD_J / NAIVE_J | 27 | 0.0049 | 0.1904 | 0.0126 | 0.1393 | 0.0845 | 0.0057 |
| transfer / COCO_hard_r60 / CC / FORWARD_J / POST_PROJECTED_J | 27 | 0.0144 | 0.1834 | 0.0133 | 0.0923 | 0.0686 | -0.0101 |
| transfer / COCO_hard_r60 / CC / NAIVE_J / NAIVE_BCE | 27 | -0.0113 | -0.0243 | -0.0026 | 0.0386 | 0.0025 | 0.0169 |
| transfer / COCO_hard_r60 / CC / POST_PROJECTED_J / POST_PROJECTED_BCE | 27 | -0.0032 | -0.0121 | -0.0004 | 0.0353 | 0.0103 | 0.0036 |
| transfer / COCO_hard_r60 / Flickr30K / FORWARD_BCE / NAIVE_BCE | 27 | 0.0066 | 0.0481 | 0.0023 | 0.0654 | 0.0255 | -0.0155 |
| transfer / COCO_hard_r60 / Flickr30K / FORWARD_BCE / POST_PROJECTED_BCE | 27 | 0.0017 | -0.0583 | 0.0025 | 0.0190 | 0.0153 | -0.0012 |
| transfer / COCO_hard_r60 / Flickr30K / FORWARD_J / FORWARD_BCE | 27 | 0.0578 | 0.8460 | 0.0418 | 0.1138 | 0.1203 | -0.0652 |
| transfer / COCO_hard_r60 / Flickr30K / FORWARD_J / NAIVE_J | 27 | 0.0624 | 0.8761 | 0.0448 | 0.1359 | 0.1365 | -0.0758 |
| transfer / COCO_hard_r60 / Flickr30K / FORWARD_J / POST_PROJECTED_J | 27 | 0.0505 | 0.7203 | 0.0393 | 0.0950 | 0.1087 | -0.0493 |
| transfer / COCO_hard_r60 / Flickr30K / NAIVE_J / NAIVE_BCE | 27 | 0.0020 | 0.0180 | -0.0007 | 0.0434 | 0.0092 | -0.0049 |
| transfer / COCO_hard_r60 / Flickr30K / POST_PROJECTED_J / POST_PROJECTED_BCE | 27 | 0.0090 | 0.0673 | 0.0051 | 0.0377 | 0.0269 | -0.0171 |
| transfer / COCO_independent_r00 / CC / FORWARD_BCE / NAIVE_BCE | 27 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| transfer / COCO_independent_r00 / CC / FORWARD_BCE / POST_PROJECTED_BCE | 27 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| transfer / COCO_independent_r00 / CC / FORWARD_J / FORWARD_BCE | 27 | -0.0094 | -0.0198 | -0.0015 | 0.0191 | -0.0005 | 0.0133 |
| transfer / COCO_independent_r00 / CC / FORWARD_J / NAIVE_J | 27 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| transfer / COCO_independent_r00 / CC / FORWARD_J / POST_PROJECTED_J | 27 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| transfer / COCO_independent_r00 / CC / NAIVE_J / NAIVE_BCE | 27 | -0.0094 | -0.0198 | -0.0015 | 0.0191 | -0.0005 | 0.0133 |
| transfer / COCO_independent_r00 / CC / POST_PROJECTED_J / POST_PROJECTED_BCE | 27 | -0.0094 | -0.0198 | -0.0015 | 0.0191 | -0.0005 | 0.0133 |
| transfer / COCO_independent_r00 / Flickr30K / FORWARD_BCE / NAIVE_BCE | 27 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| transfer / COCO_independent_r00 / Flickr30K / FORWARD_BCE / POST_PROJECTED_BCE | 27 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| transfer / COCO_independent_r00 / Flickr30K / FORWARD_J / FORWARD_BCE | 27 | -0.0016 | -0.0036 | -0.0013 | 0.0301 | 0.0060 | -0.0003 |
| transfer / COCO_independent_r00 / Flickr30K / FORWARD_J / NAIVE_J | 27 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| transfer / COCO_independent_r00 / Flickr30K / FORWARD_J / POST_PROJECTED_J | 27 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| transfer / COCO_independent_r00 / Flickr30K / NAIVE_J / NAIVE_BCE | 27 | -0.0016 | -0.0036 | -0.0013 | 0.0301 | 0.0060 | -0.0003 |
| transfer / COCO_independent_r00 / Flickr30K / POST_PROJECTED_J / POST_PROJECTED_BCE | 27 | -0.0016 | -0.0036 | -0.0013 | 0.0301 | 0.0060 | -0.0003 |
| transfer / COCO_independent_r60 / CC / FORWARD_BCE / NAIVE_BCE | 27 | -0.2578 | -1.1206 | -0.0118 | 0.1274 | NA | 0.2609 |
| transfer / COCO_independent_r60 / CC / FORWARD_BCE / POST_PROJECTED_BCE | 27 | -0.2153 | -0.7970 | -0.0118 | 0.1274 | NA | 0.2093 |
| transfer / COCO_independent_r60 / CC / FORWARD_J / FORWARD_BCE | 27 | -0.0063 | -0.0179 | -0.0012 | 0.0174 | 0.0030 | 0.0090 |
| transfer / COCO_independent_r60 / CC / FORWARD_J / NAIVE_J | 27 | -0.2724 | -1.2360 | -0.0130 | 0.1448 | NA | 0.2725 |
| transfer / COCO_independent_r60 / CC / FORWARD_J / POST_PROJECTED_J | 27 | -0.2332 | -0.8914 | -0.0130 | 0.1448 | NA | 0.2482 |
| transfer / COCO_independent_r60 / CC / NAIVE_J / NAIVE_BCE | 27 | 0.0082 | 0.0975 | 0.0000 | 0.0000 | NA | -0.0026 |
| transfer / COCO_independent_r60 / CC / POST_PROJECTED_J / POST_PROJECTED_BCE | 27 | 0.0116 | 0.0765 | 0.0000 | 0.0000 | NA | -0.0299 |
| transfer / COCO_independent_r60 / Flickr30K / FORWARD_BCE / NAIVE_BCE | 27 | -0.3318 | -1.3293 | -0.0215 | 0.2371 | NA | 0.3490 |
| transfer / COCO_independent_r60 / Flickr30K / FORWARD_BCE / POST_PROJECTED_BCE | 27 | -0.2873 | -0.9981 | -0.0215 | 0.2371 | NA | 0.2894 |
| transfer / COCO_independent_r60 / Flickr30K / FORWARD_J / FORWARD_BCE | 27 | -0.0031 | -0.0090 | -0.0010 | 0.0206 | 0.0038 | 0.0038 |
| transfer / COCO_independent_r60 / Flickr30K / FORWARD_J / NAIVE_J | 27 | -0.3452 | -1.4523 | -0.0226 | 0.2578 | NA | 0.3649 |
| transfer / COCO_independent_r60 / Flickr30K / FORWARD_J / POST_PROJECTED_J | 27 | -0.3034 | -1.0917 | -0.0226 | 0.2578 | NA | 0.3165 |
| transfer / COCO_independent_r60 / Flickr30K / NAIVE_J / NAIVE_BCE | 27 | 0.0103 | 0.1140 | 0.0000 | 0.0000 | NA | -0.0122 |
| transfer / COCO_independent_r60 / Flickr30K / POST_PROJECTED_J / POST_PROJECTED_BCE | 27 | 0.0130 | 0.0846 | 0.0000 | 0.0000 | NA | -0.0233 |
| transfer / COCO_independent_r80 / CC / FORWARD_BCE / NAIVE_BCE | 27 | -0.2745 | -1.3492 | -0.0115 | 0.1231 | NA | 0.2624 |
| transfer / COCO_independent_r80 / CC / FORWARD_BCE / POST_PROJECTED_BCE | 27 | -0.2250 | -0.8616 | -0.0115 | 0.1231 | NA | 0.2008 |
| transfer / COCO_independent_r80 / CC / FORWARD_J / FORWARD_BCE | 27 | -0.0036 | -0.0096 | -0.0007 | 0.0098 | 0.0002 | 0.0052 |
| transfer / COCO_independent_r80 / CC / FORWARD_J / NAIVE_J | 27 | -0.2813 | -1.4191 | -0.0122 | 0.1329 | NA | 0.2677 |
| transfer / COCO_independent_r80 / CC / FORWARD_J / POST_PROJECTED_J | 27 | -0.2353 | -0.9188 | -0.0122 | 0.1329 | NA | 0.2323 |
| transfer / COCO_independent_r80 / CC / NAIVE_J / NAIVE_BCE | 27 | 0.0032 | 0.0603 | 0.0000 | 0.0000 | NA | 0.0000 |
| transfer / COCO_independent_r80 / CC / POST_PROJECTED_J / POST_PROJECTED_BCE | 27 | 0.0068 | 0.0476 | 0.0000 | 0.0000 | NA | -0.0262 |
| transfer / COCO_independent_r80 / Flickr30K / FORWARD_BCE / NAIVE_BCE | 27 | -0.3533 | -1.6079 | -0.0208 | 0.2284 | NA | 0.3614 |
| transfer / COCO_independent_r80 / Flickr30K / FORWARD_BCE / POST_PROJECTED_BCE | 27 | -0.3037 | -1.1029 | -0.0208 | 0.2284 | NA | 0.2871 |
| transfer / COCO_independent_r80 / Flickr30K / FORWARD_J / FORWARD_BCE | 27 | -0.0023 | -0.0059 | -0.0011 | 0.0154 | 0.0022 | 0.0012 |
| transfer / COCO_independent_r80 / Flickr30K / FORWARD_J / NAIVE_J | 27 | -0.3595 | -1.6854 | -0.0219 | 0.2438 | NA | 0.3626 |
| transfer / COCO_independent_r80 / Flickr30K / FORWARD_J / POST_PROJECTED_J | 27 | -0.3131 | -1.1611 | -0.0219 | 0.2438 | NA | 0.2980 |
| transfer / COCO_independent_r80 / Flickr30K / NAIVE_J / NAIVE_BCE | 27 | 0.0039 | 0.0716 | 0.0000 | 0.0000 | NA | 0.0000 |
| transfer / COCO_independent_r80 / Flickr30K / POST_PROJECTED_J / POST_PROJECTED_BCE | 27 | 0.0071 | 0.0522 | 0.0000 | 0.0000 | NA | -0.0097 |
| transfer / Flickr30K_hard_r60 / CC / FORWARD_BCE / NAIVE_BCE | 27 | -0.0041 | 0.0835 | 0.0026 | 0.0689 | 0.0322 | 0.0110 |
| transfer / Flickr30K_hard_r60 / CC / FORWARD_BCE / POST_PROJECTED_BCE | 27 | 0.0024 | 0.0314 | 0.0028 | 0.0230 | 0.0207 | -0.0008 |
| transfer / Flickr30K_hard_r60 / CC / FORWARD_J / FORWARD_BCE | 27 | 0.0340 | 0.7667 | 0.0322 | 0.1037 | 0.1005 | -0.0209 |
| transfer / Flickr30K_hard_r60 / CC / FORWARD_J / NAIVE_J | 27 | 0.0335 | 0.8288 | 0.0350 | 0.1415 | 0.1245 | -0.0180 |
| transfer / Flickr30K_hard_r60 / CC / FORWARD_J / POST_PROJECTED_J | 27 | 0.0340 | 0.7269 | 0.0329 | 0.1012 | 0.1044 | -0.0208 |
| transfer / Flickr30K_hard_r60 / CC / NAIVE_J / NAIVE_BCE | 27 | -0.0036 | 0.0214 | -0.0002 | 0.0312 | 0.0082 | 0.0081 |
| transfer / Flickr30K_hard_r60 / CC / POST_PROJECTED_J / POST_PROJECTED_BCE | 27 | 0.0025 | 0.0712 | 0.0021 | 0.0255 | 0.0167 | -0.0009 |
| transfer / Flickr30K_hard_r60 / COCO / FORWARD_BCE / NAIVE_BCE | 27 | 0.0293 | 0.6810 | 0.0270 | 0.0694 | 0.0731 | -0.0262 |
| transfer / Flickr30K_hard_r60 / COCO / FORWARD_BCE / POST_PROJECTED_BCE | 27 | 0.0142 | 0.3757 | 0.0152 | 0.0260 | 0.0386 | -0.0064 |
| transfer / Flickr30K_hard_r60 / COCO / FORWARD_J / FORWARD_BCE | 27 | 0.0415 | 1.1319 | 0.0417 | 0.0574 | 0.0933 | -0.0445 |
| transfer / Flickr30K_hard_r60 / COCO / FORWARD_J / NAIVE_J | 27 | 0.0602 | 1.5747 | 0.0596 | 0.0961 | 0.1388 | -0.0588 |
| transfer / Flickr30K_hard_r60 / COCO / FORWARD_J / POST_PROJECTED_J | 27 | 0.0407 | 1.1647 | 0.0423 | 0.0550 | 0.0947 | -0.0389 |
| transfer / Flickr30K_hard_r60 / COCO / NAIVE_J / NAIVE_BCE | 27 | 0.0106 | 0.2382 | 0.0091 | 0.0308 | 0.0276 | -0.0119 |
| transfer / Flickr30K_hard_r60 / COCO / POST_PROJECTED_J / POST_PROJECTED_BCE | 27 | 0.0150 | 0.3428 | 0.0146 | 0.0284 | 0.0372 | -0.0120 |
| transfer / Flickr30K_independent_r00 / CC / FORWARD_BCE / NAIVE_BCE | 27 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| transfer / Flickr30K_independent_r00 / CC / FORWARD_BCE / POST_PROJECTED_BCE | 27 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| transfer / Flickr30K_independent_r00 / CC / FORWARD_J / FORWARD_BCE | 27 | -0.0047 | -0.0139 | -0.0007 | 0.0098 | 0.0003 | 0.0088 |
| transfer / Flickr30K_independent_r00 / CC / FORWARD_J / NAIVE_J | 27 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| transfer / Flickr30K_independent_r00 / CC / FORWARD_J / POST_PROJECTED_J | 27 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| transfer / Flickr30K_independent_r00 / CC / NAIVE_J / NAIVE_BCE | 27 | -0.0047 | -0.0139 | -0.0007 | 0.0098 | 0.0003 | 0.0088 |
| transfer / Flickr30K_independent_r00 / CC / POST_PROJECTED_J / POST_PROJECTED_BCE | 27 | -0.0047 | -0.0139 | -0.0007 | 0.0098 | 0.0003 | 0.0088 |
| transfer / Flickr30K_independent_r00 / COCO / FORWARD_BCE / NAIVE_BCE | 27 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| transfer / Flickr30K_independent_r00 / COCO / FORWARD_BCE / POST_PROJECTED_BCE | 27 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| transfer / Flickr30K_independent_r00 / COCO / FORWARD_J / FORWARD_BCE | 27 | 0.0001 | 0.0202 | 0.0008 | 0.0160 | 0.0091 | 0.0007 |
| transfer / Flickr30K_independent_r00 / COCO / FORWARD_J / NAIVE_J | 27 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| transfer / Flickr30K_independent_r00 / COCO / FORWARD_J / POST_PROJECTED_J | 27 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| transfer / Flickr30K_independent_r00 / COCO / NAIVE_J / NAIVE_BCE | 27 | 0.0001 | 0.0202 | 0.0008 | 0.0160 | 0.0091 | 0.0007 |
| transfer / Flickr30K_independent_r00 / COCO / POST_PROJECTED_J / POST_PROJECTED_BCE | 27 | 0.0001 | 0.0202 | 0.0008 | 0.0160 | 0.0091 | 0.0007 |
| transfer / Flickr30K_independent_r60 / CC / FORWARD_BCE / NAIVE_BCE | 27 | -0.2613 | -1.1347 | -0.0137 | 0.1587 | NA | 0.2670 |
| transfer / Flickr30K_independent_r60 / CC / FORWARD_BCE / POST_PROJECTED_BCE | 27 | -0.2156 | -0.7725 | -0.0137 | 0.1587 | NA | 0.2384 |
| transfer / Flickr30K_independent_r60 / CC / FORWARD_J / FORWARD_BCE | 27 | -0.0021 | -0.0059 | -0.0002 | 0.0038 | -0.0005 | 0.0043 |
| transfer / Flickr30K_independent_r60 / CC / FORWARD_J / NAIVE_J | 27 | -0.2697 | -1.2213 | -0.0139 | 0.1625 | NA | 0.2721 |
| transfer / Flickr30K_independent_r60 / CC / FORWARD_J / POST_PROJECTED_J | 27 | -0.2275 | -0.8445 | -0.0139 | 0.1625 | NA | 0.2565 |
| transfer / Flickr30K_independent_r60 / CC / NAIVE_J / NAIVE_BCE | 27 | 0.0064 | 0.0808 | 0.0000 | 0.0000 | NA | -0.0009 |
| transfer / Flickr30K_independent_r60 / CC / POST_PROJECTED_J / POST_PROJECTED_BCE | 27 | 0.0099 | 0.0661 | 0.0000 | 0.0000 | NA | -0.0139 |
| transfer / Flickr30K_independent_r60 / COCO / FORWARD_BCE / NAIVE_BCE | 27 | -0.2830 | -1.1223 | -0.0136 | 0.2521 | NA | 0.3076 |
| transfer / Flickr30K_independent_r60 / COCO / FORWARD_BCE / POST_PROJECTED_BCE | 27 | -0.2224 | -0.7202 | -0.0136 | 0.2521 | NA | 0.2593 |
| transfer / Flickr30K_independent_r60 / COCO / FORWARD_J / FORWARD_BCE | 27 | 0.0001 | 0.0131 | 0.0002 | 0.0084 | 0.0034 | -0.0013 |
| transfer / Flickr30K_independent_r60 / COCO / FORWARD_J / NAIVE_J | 27 | -0.2916 | -1.2021 | -0.0135 | 0.2605 | NA | 0.3081 |
| transfer / Flickr30K_independent_r60 / COCO / FORWARD_J / POST_PROJECTED_J | 27 | -0.2356 | -0.7826 | -0.0135 | 0.2605 | NA | 0.2856 |
| transfer / Flickr30K_independent_r60 / COCO / NAIVE_J / NAIVE_BCE | 27 | 0.0087 | 0.0929 | 0.0000 | 0.0000 | NA | -0.0017 |
| transfer / Flickr30K_independent_r60 / COCO / POST_PROJECTED_J / POST_PROJECTED_BCE | 27 | 0.0133 | 0.0755 | 0.0000 | 0.0000 | NA | -0.0276 |
| transfer / Flickr30K_independent_r80 / CC / FORWARD_BCE / NAIVE_BCE | 27 | -0.2771 | -1.3622 | -0.0135 | 0.1562 | NA | 0.2681 |
| transfer / Flickr30K_independent_r80 / CC / FORWARD_BCE / POST_PROJECTED_BCE | 27 | -0.2227 | -0.8167 | -0.0135 | 0.1562 | NA | 0.2353 |
| transfer / Flickr30K_independent_r80 / CC / FORWARD_J / FORWARD_BCE | 27 | -0.0003 | -0.0004 | -0.0000 | -0.0009 | -0.0012 | 0.0000 |
| transfer / Flickr30K_independent_r80 / CC / FORWARD_J / NAIVE_J | 27 | -0.2799 | -1.4127 | -0.0136 | 0.1554 | NA | 0.2682 |
| transfer / Flickr30K_independent_r80 / CC / FORWARD_J / POST_PROJECTED_J | 27 | -0.2292 | -0.8602 | -0.0136 | 0.1554 | NA | 0.2456 |
| transfer / Flickr30K_independent_r80 / CC / NAIVE_J / NAIVE_BCE | 27 | 0.0025 | 0.0501 | 0.0000 | 0.0000 | NA | 0.0000 |
| transfer / Flickr30K_independent_r80 / CC / POST_PROJECTED_J / POST_PROJECTED_BCE | 27 | 0.0062 | 0.0430 | 0.0000 | 0.0000 | NA | -0.0103 |
| transfer / Flickr30K_independent_r80 / COCO / FORWARD_BCE / NAIVE_BCE | 27 | -0.3055 | -1.3957 | -0.0138 | 0.2487 | NA | 0.3096 |
| transfer / Flickr30K_independent_r80 / COCO / FORWARD_BCE / POST_PROJECTED_BCE | 27 | -0.2326 | -0.7744 | -0.0138 | 0.2487 | NA | 0.2548 |
| transfer / Flickr30K_independent_r80 / COCO / FORWARD_J / FORWARD_BCE | 27 | 0.0004 | 0.0103 | 0.0006 | 0.0024 | 0.0048 | 0.0008 |
| transfer / Flickr30K_independent_r80 / COCO / FORWARD_J / NAIVE_J | 27 | -0.3085 | -1.4441 | -0.0132 | 0.2511 | NA | 0.3104 |
| transfer / Flickr30K_independent_r80 / COCO / FORWARD_J / POST_PROJECTED_J | 27 | -0.2404 | -0.8128 | -0.0132 | 0.2511 | NA | 0.2776 |
| transfer / Flickr30K_independent_r80 / COCO / NAIVE_J / NAIVE_BCE | 27 | 0.0033 | 0.0587 | 0.0000 | 0.0000 | NA | 0.0000 |
| transfer / Flickr30K_independent_r80 / COCO / POST_PROJECTED_J / POST_PROJECTED_BCE | 27 | 0.0082 | 0.0488 | 0.0000 | 0.0000 | NA | -0.0221 |

## 6. Model mismatch: hard contamination (noise_model_valid_by_construction=false) and transfer

| source case | target | method | n | cov@.90 | risk@.90 | NA seeds | FAR@.90 | cov@.95 | risk@.95 | Brier(C) |
|---|---|---|---|---|---|---|---|---|---|---|
| COCO_hard_r60 | CC | FORWARD_BCE | 3 | 0.1940 | 0.0271 | 0 | 0.0052 | 0.1645 | 0.0177 | 0.1096 |
| COCO_hard_r60 | CC | FORWARD_J | 3 | 0.3277 | 0.0818 | 0 | 0.0408 | 0.3021 | 0.0680 | 0.1105 |
| COCO_hard_r60 | CC | NAIVE_BCE | 3 | 0.1302 | 0.0186 | 0 | 0.0026 | 0.1046 | 0.0165 | 0.1328 |
| COCO_hard_r60 | CC | NAIVE_J | 3 | 0.1793 | 0.0268 | 0 | 0.0043 | 0.1328 | 0.0184 | 0.1187 |
| COCO_hard_r60 | Flickr30K | FORWARD_BCE | 3 | 0.3672 | 0.0283 | 0 | 0.0208 | 0.3155 | 0.0178 | 0.0829 |
| COCO_hard_r60 | Flickr30K | FORWARD_J | 3 | 0.4891 | 0.1480 | 0 | 0.1372 | 0.4588 | 0.1114 | 0.1464 |
| COCO_hard_r60 | Flickr30K | NAIVE_BCE | 3 | 0.2869 | 0.0151 | 0 | 0.0087 | 0.2231 | 0.0077 | 0.0719 |
| COCO_hard_r60 | Flickr30K | NAIVE_J | 3 | 0.3438 | 0.0225 | 0 | 0.0156 | 0.2886 | 0.0164 | 0.0764 |
| COCO_independent_r00 | CC | FORWARD_BCE | 3 | 0.1102 | 0.0195 | 0 | 0.0017 | 0.0903 | 0.0057 | 0.1391 |
| COCO_independent_r00 | CC | FORWARD_J | 3 | 0.1280 | 0.0239 | 0 | 0.0035 | 0.1046 | 0.0041 | 0.1276 |
| COCO_independent_r00 | CC | NAIVE_BCE | 3 | 0.1102 | 0.0195 | 0 | 0.0017 | 0.0903 | 0.0057 | 0.1391 |
| COCO_independent_r00 | CC | NAIVE_J | 3 | 0.1280 | 0.0239 | 0 | 0.0035 | 0.1046 | 0.0041 | 0.1276 |
| COCO_independent_r00 | Flickr30K | FORWARD_BCE | 3 | 0.2517 | 0.0121 | 0 | 0.0061 | 0.1884 | 0.0046 | 0.0731 |
| COCO_independent_r00 | Flickr30K | FORWARD_J | 3 | 0.2908 | 0.0150 | 0 | 0.0087 | 0.2261 | 0.0097 | 0.0728 |
| COCO_independent_r00 | Flickr30K | NAIVE_BCE | 3 | 0.2517 | 0.0121 | 0 | 0.0061 | 0.1884 | 0.0046 | 0.0731 |
| COCO_independent_r00 | Flickr30K | NAIVE_J | 3 | 0.2908 | 0.0150 | 0 | 0.0087 | 0.2261 | 0.0097 | 0.0728 |
| COCO_independent_r60 | CC | FORWARD_BCE | 3 | 0.0981 | 0.0050 | 0 | 0.0009 | 0.0755 | 0.0000 | 0.1475 |
| COCO_independent_r60 | CC | FORWARD_J | 3 | 0.1063 | 0.0202 | 0 | 0.0017 | 0.0846 | 0.0000 | 0.1401 |
| COCO_independent_r60 | CC | NAIVE_BCE | 3 | 0.0000 | NA | 3 | 0.0000 | 0.0000 | NA | 0.4031 |
| COCO_independent_r60 | CC | NAIVE_J | 3 | 0.0000 | NA | 3 | 0.0000 | 0.0000 | NA | 0.4080 |
| COCO_independent_r60 | Flickr30K | FORWARD_BCE | 3 | 0.2018 | 0.0064 | 0 | 0.0026 | 0.1619 | 0.0025 | 0.0760 |
| COCO_independent_r60 | Flickr30K | FORWARD_J | 3 | 0.2352 | 0.0091 | 0 | 0.0043 | 0.1788 | 0.0023 | 0.0734 |
| COCO_independent_r60 | Flickr30K | NAIVE_BCE | 3 | 0.0000 | NA | 3 | 0.0000 | 0.0000 | NA | 0.4382 |
| COCO_independent_r60 | Flickr30K | NAIVE_J | 3 | 0.0000 | NA | 3 | 0.0000 | 0.0000 | NA | 0.4453 |
| COCO_independent_r80 | CC | FORWARD_BCE | 3 | 0.0968 | 0.0050 | 0 | 0.0009 | 0.0690 | 0.0000 | 0.1485 |
| COCO_independent_r80 | CC | FORWARD_J | 3 | 0.1020 | 0.0174 | 0 | 0.0009 | 0.0768 | 0.0000 | 0.1444 |
| COCO_independent_r80 | CC | NAIVE_BCE | 3 | 0.0000 | NA | 3 | 0.0000 | 0.0000 | NA | 0.4126 |
| COCO_independent_r80 | CC | NAIVE_J | 3 | 0.0000 | NA | 3 | 0.0000 | 0.0000 | NA | 0.4143 |
| COCO_independent_r80 | Flickr30K | FORWARD_BCE | 3 | 0.1936 | 0.0044 | 0 | 0.0017 | 0.1489 | 0.0029 | 0.0769 |
| COCO_independent_r80 | Flickr30K | FORWARD_J | 3 | 0.2135 | 0.0063 | 0 | 0.0026 | 0.1684 | 0.0025 | 0.0749 |
| COCO_independent_r80 | Flickr30K | NAIVE_BCE | 3 | 0.0000 | NA | 3 | 0.0000 | 0.0000 | NA | 0.4522 |
| COCO_independent_r80 | Flickr30K | NAIVE_J | 3 | 0.0000 | NA | 3 | 0.0000 | 0.0000 | NA | 0.4546 |
| Flickr30K_hard_r60 | CC | FORWARD_BCE | 3 | 0.2721 | 0.0572 | 0 | 0.0156 | 0.2361 | 0.0344 | 0.1049 |
| Flickr30K_hard_r60 | CC | FORWARD_J | 3 | 0.3746 | 0.1202 | 0 | 0.0747 | 0.3494 | 0.0970 | 0.1258 |
| Flickr30K_hard_r60 | CC | NAIVE_BCE | 3 | 0.1871 | 0.0256 | 0 | 0.0043 | 0.1645 | 0.0258 | 0.1154 |
| Flickr30K_hard_r60 | CC | NAIVE_J | 3 | 0.2214 | 0.0349 | 0 | 0.0078 | 0.1914 | 0.0270 | 0.1090 |
| Flickr30K_hard_r60 | COCO | FORWARD_BCE | 3 | 0.4167 | 0.1829 | 0 | 0.1172 | 0.3733 | 0.1468 | 0.1629 |
| Flickr30K_hard_r60 | COCO | FORWARD_J | 3 | 0.4766 | 0.2854 | 0 | 0.2292 | 0.4527 | 0.2646 | 0.2071 |
| Flickr30K_hard_r60 | COCO | NAIVE_BCE | 3 | 0.3251 | 0.0988 | 0 | 0.0486 | 0.2782 | 0.0712 | 0.1321 |
| Flickr30K_hard_r60 | COCO | NAIVE_J | 3 | 0.3585 | 0.1268 | 0 | 0.0738 | 0.3229 | 0.1038 | 0.1432 |
| Flickr30K_independent_r00 | CC | FORWARD_BCE | 3 | 0.1211 | 0.0199 | 0 | 0.0026 | 0.0955 | 0.0181 | 0.1539 |
| Flickr30K_independent_r00 | CC | FORWARD_J | 3 | 0.1341 | 0.0177 | 0 | 0.0026 | 0.1050 | 0.0199 | 0.1481 |
| Flickr30K_independent_r00 | CC | NAIVE_BCE | 3 | 0.1211 | 0.0199 | 0 | 0.0026 | 0.0955 | 0.0181 | 0.1539 |
| Flickr30K_independent_r00 | CC | NAIVE_J | 3 | 0.1341 | 0.0177 | 0 | 0.0026 | 0.1050 | 0.0199 | 0.1481 |
| Flickr30K_independent_r00 | COCO | FORWARD_BCE | 3 | 0.2305 | 0.0411 | 0 | 0.0104 | 0.1853 | 0.0308 | 0.1207 |
| Flickr30K_independent_r00 | COCO | FORWARD_J | 3 | 0.2470 | 0.0469 | 0 | 0.0148 | 0.2148 | 0.0305 | 0.1211 |
| Flickr30K_independent_r00 | COCO | NAIVE_BCE | 3 | 0.2305 | 0.0411 | 0 | 0.0104 | 0.1853 | 0.0308 | 0.1207 |
| Flickr30K_independent_r00 | COCO | NAIVE_J | 3 | 0.2470 | 0.0469 | 0 | 0.0148 | 0.2148 | 0.0305 | 0.1211 |
| Flickr30K_independent_r60 | CC | FORWARD_BCE | 3 | 0.1276 | 0.0190 | 0 | 0.0026 | 0.1007 | 0.0173 | 0.1492 |
| Flickr30K_independent_r60 | CC | FORWARD_J | 3 | 0.1302 | 0.0185 | 0 | 0.0026 | 0.1072 | 0.0159 | 0.1466 |
| Flickr30K_independent_r60 | CC | NAIVE_BCE | 3 | 0.0000 | NA | 3 | 0.0000 | 0.0000 | NA | 0.4058 |
| Flickr30K_independent_r60 | CC | NAIVE_J | 3 | 0.0000 | NA | 3 | 0.0000 | 0.0000 | NA | 0.4094 |
| Flickr30K_independent_r60 | COCO | FORWARD_BCE | 3 | 0.2513 | 0.0490 | 0 | 0.0130 | 0.1988 | 0.0381 | 0.1210 |
| Flickr30K_independent_r60 | COCO | FORWARD_J | 3 | 0.2569 | 0.0437 | 0 | 0.0139 | 0.2122 | 0.0306 | 0.1213 |
| Flickr30K_independent_r60 | COCO | NAIVE_BCE | 3 | 0.0000 | NA | 3 | 0.0000 | 0.0000 | NA | 0.4151 |
| Flickr30K_independent_r60 | COCO | NAIVE_J | 3 | 0.0000 | NA | 3 | 0.0000 | 0.0000 | NA | 0.4203 |
| Flickr30K_independent_r80 | CC | FORWARD_BCE | 3 | 0.1237 | 0.0196 | 0 | 0.0026 | 0.0981 | 0.0178 | 0.1496 |
| Flickr30K_independent_r80 | CC | FORWARD_J | 3 | 0.1159 | 0.0226 | 0 | 0.0026 | 0.1007 | 0.0046 | 0.1491 |
| Flickr30K_independent_r80 | CC | NAIVE_BCE | 3 | 0.0000 | NA | 3 | 0.0000 | 0.0000 | NA | 0.4139 |
| Flickr30K_independent_r80 | CC | NAIVE_J | 3 | 0.0000 | NA | 3 | 0.0000 | 0.0000 | NA | 0.4152 |
| Flickr30K_independent_r80 | COCO | FORWARD_BCE | 3 | 0.2452 | 0.0518 | 0 | 0.0139 | 0.1905 | 0.0373 | 0.1209 |
| Flickr30K_independent_r80 | COCO | FORWARD_J | 3 | 0.2457 | 0.0534 | 0 | 0.0148 | 0.1962 | 0.0379 | 0.1210 |
| Flickr30K_independent_r80 | COCO | NAIVE_BCE | 3 | 0.0000 | NA | 3 | 0.0000 | 0.0000 | NA | 0.4270 |
| Flickr30K_independent_r80 | COCO | NAIVE_J | 3 | 0.0000 | NA | 3 | 0.0000 | 0.0000 | NA | 0.4288 |

Wrong-pi (predictor pi=0.5, target pi=0.25/0.75) rows are in PREDICTION_LEDGER.csv (wrong_pi=True).

## 7. Unknown rho (regime U): partial interval first, then explicit anchor assumptions

| case | epsilon | status | lower | upper | anchor point | true rho (eval only) | contains |
|---|---|---|---|---|---|---|---|
| COCO_independent_r60_d1001 | 0.0500 | CONDITIONAL_INTERVAL | 0.4908 | 0.6559 | NA | 0.6000 | True |
| COCO_independent_r60_d1001 | 0.1000 | CONDITIONAL_INTERVAL | 0.4625 | 0.6559 | NA | 0.6000 | True |
| COCO_independent_r60_d1001 | 0.0000 | CONDITIONAL_INTERVAL | 0.5163 | 0.6559 | 0.4286 | 0.6000 | True |
| COCO_independent_r60_d1001 | none | CONDITIONAL_INTERVAL | 0.0000 | 0.6559 | NA | 0.6000 | True |
| COCO_independent_r80_d1001 | 0.0500 | CONDITIONAL_INTERVAL | 0.6884 | 0.8714 | NA | 0.8000 | True |
| COCO_independent_r80_d1001 | 0.1000 | CONDITIONAL_INTERVAL | 0.6711 | 0.8714 | NA | 0.8000 | True |
| COCO_independent_r80_d1001 | 0.0000 | CONDITIONAL_INTERVAL | 0.7040 | 0.8714 | 0.6245 | 0.8000 | True |
| COCO_independent_r80_d1001 | none | CONDITIONAL_INTERVAL | 0.0000 | 0.8714 | NA | 0.8000 | True |
| COCO_independent_r60_d2002 | 0.0500 | CONDITIONAL_INTERVAL | 0.5238 | 0.7118 | NA | 0.6000 | True |
| COCO_independent_r60_d2002 | 0.1000 | CONDITIONAL_INTERVAL | 0.4974 | 0.7118 | NA | 0.6000 | True |
| COCO_independent_r60_d2002 | 0.0000 | CONDITIONAL_INTERVAL | 0.5476 | 0.7118 | 0.5721 | 0.6000 | True |
| COCO_independent_r60_d2002 | none | CONDITIONAL_INTERVAL | 0.0000 | 0.7118 | NA | 0.6000 | True |
| COCO_independent_r80_d2002 | 0.0500 | CONDITIONAL_INTERVAL | 0.7171 | 0.9310 | NA | 0.8000 | True |
| COCO_independent_r80_d2002 | 0.1000 | CONDITIONAL_INTERVAL | 0.7013 | 0.9310 | NA | 0.8000 | True |
| COCO_independent_r80_d2002 | 0.0000 | CONDITIONAL_INTERVAL | 0.7312 | 0.9310 | 0.7644 | 0.8000 | True |
| COCO_independent_r80_d2002 | none | CONDITIONAL_INTERVAL | 0.0000 | 0.9310 | NA | 0.8000 | True |
| COCO_independent_r60_d3003 | 0.0500 | CONDITIONAL_INTERVAL | 0.4965 | 0.6764 | NA | 0.6000 | True |
| COCO_independent_r60_d3003 | 0.1000 | CONDITIONAL_INTERVAL | 0.4685 | 0.6764 | NA | 0.6000 | True |
| COCO_independent_r60_d3003 | 0.0000 | CONDITIONAL_INTERVAL | 0.5216 | 0.6764 | 0.5942 | 0.6000 | True |
| COCO_independent_r60_d3003 | none | CONDITIONAL_INTERVAL | 0.0000 | 0.6764 | NA | 0.6000 | True |
| COCO_independent_r80_d3003 | 0.0500 | CONDITIONAL_INTERVAL | 0.6996 | 0.9060 | NA | 0.8000 | True |
| COCO_independent_r80_d3003 | 0.1000 | CONDITIONAL_INTERVAL | 0.6829 | 0.9060 | NA | 0.8000 | True |
| COCO_independent_r80_d3003 | 0.0000 | CONDITIONAL_INTERVAL | 0.7146 | 0.9060 | 0.7982 | 0.8000 | True |
| COCO_independent_r80_d3003 | none | CONDITIONAL_INTERVAL | 0.0000 | 0.9060 | NA | 0.8000 | True |
| Flickr30K_independent_r60_d1001 | 0.0500 | CONDITIONAL_INTERVAL | 0.4792 | 0.6569 | NA | 0.6000 | True |
| Flickr30K_independent_r60_d1001 | 0.1000 | CONDITIONAL_INTERVAL | 0.4503 | 0.6569 | NA | 0.6000 | True |
| Flickr30K_independent_r60_d1001 | 0.0000 | CONDITIONAL_INTERVAL | 0.5052 | 0.6569 | 0.5405 | 0.6000 | True |
| Flickr30K_independent_r60_d1001 | none | CONDITIONAL_INTERVAL | 0.0000 | 0.6569 | NA | 0.6000 | True |
| Flickr30K_independent_r80_d1001 | 0.0500 | CONDITIONAL_INTERVAL | 0.6873 | 0.8904 | NA | 0.8000 | True |
| Flickr30K_independent_r80_d1001 | 0.1000 | CONDITIONAL_INTERVAL | 0.6699 | 0.8904 | NA | 0.8000 | True |
| Flickr30K_independent_r80_d1001 | 0.0000 | CONDITIONAL_INTERVAL | 0.7029 | 0.8904 | 0.7502 | 0.8000 | True |
| Flickr30K_independent_r80_d1001 | none | CONDITIONAL_INTERVAL | 0.0000 | 0.8904 | NA | 0.8000 | True |
| Flickr30K_independent_r60_d2002 | 0.0500 | CONDITIONAL_INTERVAL | 0.4787 | 0.6532 | NA | 0.6000 | True |
| Flickr30K_independent_r60_d2002 | 0.1000 | CONDITIONAL_INTERVAL | 0.4498 | 0.6532 | NA | 0.6000 | True |
| Flickr30K_independent_r60_d2002 | 0.0000 | CONDITIONAL_INTERVAL | 0.5048 | 0.6532 | 0.5479 | 0.6000 | True |
| Flickr30K_independent_r60_d2002 | none | CONDITIONAL_INTERVAL | 0.0000 | 0.6532 | NA | 0.6000 | True |
| Flickr30K_independent_r80_d2002 | 0.0500 | CONDITIONAL_INTERVAL | 0.6546 | 0.8519 | NA | 0.8000 | True |
| Flickr30K_independent_r80_d2002 | 0.1000 | CONDITIONAL_INTERVAL | 0.6355 | 0.8519 | NA | 0.8000 | True |
| Flickr30K_independent_r80_d2002 | 0.0000 | CONDITIONAL_INTERVAL | 0.6719 | 0.8519 | 0.6815 | 0.8000 | True |
| Flickr30K_independent_r80_d2002 | none | CONDITIONAL_INTERVAL | 0.0000 | 0.8519 | NA | 0.8000 | True |
| Flickr30K_independent_r60_d3003 | 0.0500 | CONDITIONAL_INTERVAL | 0.5073 | 0.6915 | NA | 0.6000 | True |
| Flickr30K_independent_r60_d3003 | 0.1000 | CONDITIONAL_INTERVAL | 0.4799 | 0.6915 | NA | 0.6000 | True |
| Flickr30K_independent_r60_d3003 | 0.0000 | CONDITIONAL_INTERVAL | 0.5319 | 0.6915 | 0.6067 | 0.6000 | True |
| Flickr30K_independent_r60_d3003 | none | CONDITIONAL_INTERVAL | 0.0000 | 0.6915 | NA | 0.6000 | True |
| Flickr30K_independent_r80_d3003 | 0.0500 | CONDITIONAL_INTERVAL | 0.7183 | 0.9307 | NA | 0.8000 | True |
| Flickr30K_independent_r80_d3003 | 0.1000 | CONDITIONAL_INTERVAL | 0.7027 | 0.9307 | NA | 0.8000 | True |
| Flickr30K_independent_r80_d3003 | 0.0000 | CONDITIONAL_INTERVAL | 0.7324 | 0.9307 | 0.8257 | 0.8000 | True |
| Flickr30K_independent_r80_d3003 | none | CONDITIONAL_INTERVAL | 0.0000 | 0.9307 | NA | 0.8000 | True |
| CC152K_RECOVERED_20261104_observedFIT_d1001 | none | CONDITIONAL_INTERVAL | 0.0000 | 0.0031 | NA | UNKNOWN | NA |
| CC152K_RECOVERED_20261104_observedFIT_d1001 | 0 | CONDITIONAL_INTERVAL | 0.0000 | 0.0031 | 0.0000 | UNKNOWN | NA |
| CC152K_RECOVERED_20261104_observedFIT_d1001 | 0.05 | CONDITIONAL_INTERVAL | 0.0000 | 0.0031 | NA | UNKNOWN | NA |
| CC152K_RECOVERED_20261104_observedFIT_d1001 | 0.1 | CONDITIONAL_INTERVAL | 0.0000 | 0.0031 | NA | UNKNOWN | NA |
| CC152K_RECOVERED_20261104_observedFIT_d2002 | none | CONDITIONAL_INTERVAL | 0.0000 | 0.0029 | NA | UNKNOWN | NA |
| CC152K_RECOVERED_20261104_observedFIT_d2002 | 0 | CONDITIONAL_INTERVAL | 0.0000 | 0.0029 | 0.0000 | UNKNOWN | NA |
| CC152K_RECOVERED_20261104_observedFIT_d2002 | 0.05 | CONDITIONAL_INTERVAL | 0.0000 | 0.0029 | NA | UNKNOWN | NA |
| CC152K_RECOVERED_20261104_observedFIT_d2002 | 0.1 | CONDITIONAL_INTERVAL | 0.0000 | 0.0029 | NA | UNKNOWN | NA |
| CC152K_RECOVERED_20261104_observedFIT_d3003 | none | CONDITIONAL_INTERVAL | 0.0000 | 0.0029 | NA | UNKNOWN | NA |
| CC152K_RECOVERED_20261104_observedFIT_d3003 | 0 | CONDITIONAL_INTERVAL | 0.0000 | 0.0029 | 0.0000 | UNKNOWN | NA |
| CC152K_RECOVERED_20261104_observedFIT_d3003 | 0.05 | CONDITIONAL_INTERVAL | 0.0000 | 0.0029 | NA | UNKNOWN | NA |
| CC152K_RECOVERED_20261104_observedFIT_d3003 | 0.1 | CONDITIONAL_INTERVAL | 0.0000 | 0.0029 | NA | UNKNOWN | NA |

Profile grid {0,.2,.4,.6,.8,.9} fits are a finite-grid sensitivity, not a confidence band and not used to choose rho. Envelope widths:

| case | objective | kind | mean width | max width |
|---|---|---|---|---|
| COCO_independent_r60_d1001 | J | D1_mechanism_prob | 0.2746 | 0.5605 |
| COCO_independent_r60_d1001 | BCE | D1_mechanism_prob | 0.2601 | 0.5736 |
| COCO_independent_r80_d1001 | J | D1_mechanism_prob | 0.3379 | 0.5810 |
| COCO_independent_r80_d1001 | BCE | D1_mechanism_prob | 0.3236 | 0.5912 |
| COCO_independent_r60_d2002 | J | D1_mechanism_prob | 0.2757 | 0.5603 |
| COCO_independent_r60_d2002 | BCE | D1_mechanism_prob | 0.2615 | 0.5711 |
| COCO_independent_r80_d2002 | J | D1_mechanism_prob | 0.3411 | 0.5673 |
| COCO_independent_r80_d2002 | BCE | D1_mechanism_prob | 0.3274 | 0.5831 |
| COCO_independent_r60_d3003 | J | D1_mechanism_prob | 0.2765 | 0.5781 |
| COCO_independent_r60_d3003 | BCE | D1_mechanism_prob | 0.2617 | 0.5785 |
| COCO_independent_r80_d3003 | J | D1_mechanism_prob | 0.3397 | 0.5850 |
| COCO_independent_r80_d3003 | BCE | D1_mechanism_prob | 0.3254 | 0.5884 |
| Flickr30K_independent_r60_d1001 | J | D1_mechanism_prob | 0.2835 | 0.5757 |
| Flickr30K_independent_r60_d1001 | BCE | D1_mechanism_prob | 0.2721 | 0.5872 |
| Flickr30K_independent_r80_d1001 | J | D1_mechanism_prob | 0.3487 | 0.5849 |
| Flickr30K_independent_r80_d1001 | BCE | D1_mechanism_prob | 0.3372 | 0.5972 |
| Flickr30K_independent_r60_d2002 | J | D1_mechanism_prob | 0.2830 | 0.5753 |
| Flickr30K_independent_r60_d2002 | BCE | D1_mechanism_prob | 0.2728 | 0.5996 |
| Flickr30K_independent_r80_d2002 | J | D1_mechanism_prob | 0.3526 | 0.6042 |
| Flickr30K_independent_r80_d2002 | BCE | D1_mechanism_prob | 0.3417 | 0.6209 |
| Flickr30K_independent_r60_d3003 | J | D1_mechanism_prob | 0.2834 | 0.5612 |
| Flickr30K_independent_r60_d3003 | BCE | D1_mechanism_prob | 0.2730 | 0.5817 |
| Flickr30K_independent_r80_d3003 | J | D1_mechanism_prob | 0.3497 | 0.5592 |
| Flickr30K_independent_r80_d3003 | BCE | D1_mechanism_prob | 0.3402 | 0.5864 |

| case | objective | rule | tau | coverage | risk | FAR | cost |
|---|---|---|---|---|---|---|---|
| COCO_independent_r60_d1001 | J | LOWER_ENVELOPE_GRID | 0.9000 | 0.0000 | NA | 0.0000 | 0.1000 |
| COCO_independent_r60_d1001 | J | LOWER_ENVELOPE_GRID | 0.9500 | 0.0000 | NA | 0.0000 | 0.0500 |
| COCO_independent_r60_d1001 | BCE | LOWER_ENVELOPE_GRID | 0.9000 | 0.0000 | NA | 0.0000 | 0.1000 |
| COCO_independent_r60_d1001 | BCE | LOWER_ENVELOPE_GRID | 0.9500 | 0.0000 | NA | 0.0000 | 0.0500 |
| COCO_independent_r60_d1001 | J | ANCHOR_EPS0_rho0.429 | 0.9000 | 0.0000 | NA | 0.0000 | 0.1000 |
| COCO_independent_r60_d1001 | J | ANCHOR_EPS0_rho0.429 | 0.9500 | 0.0000 | NA | 0.0000 | 0.0500 |
| COCO_independent_r60_d1001 | BCE | ANCHOR_EPS0_rho0.429 | 0.9000 | 0.0664 | 0.0000 | 0.0000 | 0.0934 |
| COCO_independent_r60_d1001 | BCE | ANCHOR_EPS0_rho0.429 | 0.9500 | 0.0234 | 0.0000 | 0.0000 | 0.0488 |
| COCO_independent_r80_d1001 | J | LOWER_ENVELOPE_GRID | 0.9000 | 0.0000 | NA | 0.0000 | 0.1000 |
| COCO_independent_r80_d1001 | J | LOWER_ENVELOPE_GRID | 0.9500 | 0.0000 | NA | 0.0000 | 0.0500 |
| COCO_independent_r80_d1001 | BCE | LOWER_ENVELOPE_GRID | 0.9000 | 0.0000 | NA | 0.0000 | 0.1000 |
| COCO_independent_r80_d1001 | BCE | LOWER_ENVELOPE_GRID | 0.9500 | 0.0000 | NA | 0.0000 | 0.0500 |
| COCO_independent_r80_d1001 | J | ANCHOR_EPS0_rho0.624 | 0.9000 | 0.0000 | NA | 0.0000 | 0.1000 |
| COCO_independent_r80_d1001 | J | ANCHOR_EPS0_rho0.624 | 0.9500 | 0.0000 | NA | 0.0000 | 0.0500 |
| COCO_independent_r80_d1001 | BCE | ANCHOR_EPS0_rho0.624 | 0.9000 | 0.0664 | 0.0000 | 0.0000 | 0.0934 |
| COCO_independent_r80_d1001 | BCE | ANCHOR_EPS0_rho0.624 | 0.9500 | 0.0273 | 0.0000 | 0.0000 | 0.0486 |
| COCO_independent_r60_d2002 | J | LOWER_ENVELOPE_GRID | 0.9000 | 0.0000 | NA | 0.0000 | 0.1000 |
| COCO_independent_r60_d2002 | J | LOWER_ENVELOPE_GRID | 0.9500 | 0.0000 | NA | 0.0000 | 0.0500 |
| COCO_independent_r60_d2002 | BCE | LOWER_ENVELOPE_GRID | 0.9000 | 0.0000 | NA | 0.0000 | 0.1000 |
| COCO_independent_r60_d2002 | BCE | LOWER_ENVELOPE_GRID | 0.9500 | 0.0000 | NA | 0.0000 | 0.0500 |
| COCO_independent_r60_d2002 | J | ANCHOR_EPS0_rho0.572 | 0.9000 | 0.1693 | 0.0308 | 0.0026 | 0.0883 |
| COCO_independent_r60_d2002 | J | ANCHOR_EPS0_rho0.572 | 0.9500 | 0.1419 | 0.0092 | 0.0026 | 0.0442 |
| COCO_independent_r60_d2002 | BCE | ANCHOR_EPS0_rho0.572 | 0.9000 | 0.1654 | 0.0315 | 0.0026 | 0.0887 |
| COCO_independent_r60_d2002 | BCE | ANCHOR_EPS0_rho0.572 | 0.9500 | 0.1224 | 0.0106 | 0.0026 | 0.0452 |
| COCO_independent_r80_d2002 | J | LOWER_ENVELOPE_GRID | 0.9000 | 0.0000 | NA | 0.0000 | 0.1000 |
| COCO_independent_r80_d2002 | J | LOWER_ENVELOPE_GRID | 0.9500 | 0.0000 | NA | 0.0000 | 0.0500 |
| COCO_independent_r80_d2002 | BCE | LOWER_ENVELOPE_GRID | 0.9000 | 0.0000 | NA | 0.0000 | 0.1000 |
| COCO_independent_r80_d2002 | BCE | LOWER_ENVELOPE_GRID | 0.9500 | 0.0000 | NA | 0.0000 | 0.0500 |
| COCO_independent_r80_d2002 | J | ANCHOR_EPS0_rho0.764 | 0.9000 | 0.1536 | 0.0339 | 0.0026 | 0.0898 |
| COCO_independent_r80_d2002 | J | ANCHOR_EPS0_rho0.764 | 0.9500 | 0.1172 | 0.0000 | 0.0000 | 0.0441 |
| COCO_independent_r80_d2002 | BCE | ANCHOR_EPS0_rho0.764 | 0.9000 | 0.1497 | 0.0348 | 0.0026 | 0.0902 |
| COCO_independent_r80_d2002 | BCE | ANCHOR_EPS0_rho0.764 | 0.9500 | 0.1094 | 0.0000 | 0.0000 | 0.0445 |
| COCO_independent_r60_d3003 | J | LOWER_ENVELOPE_GRID | 0.9000 | 0.0000 | NA | 0.0000 | 0.1000 |
| COCO_independent_r60_d3003 | J | LOWER_ENVELOPE_GRID | 0.9500 | 0.0000 | NA | 0.0000 | 0.0500 |
| COCO_independent_r60_d3003 | BCE | LOWER_ENVELOPE_GRID | 0.9000 | 0.0000 | NA | 0.0000 | 0.1000 |
| COCO_independent_r60_d3003 | BCE | LOWER_ENVELOPE_GRID | 0.9500 | 0.0000 | NA | 0.0000 | 0.0500 |
| COCO_independent_r60_d3003 | J | ANCHOR_EPS0_rho0.594 | 0.9000 | 0.2201 | 0.0237 | 0.0104 | 0.0832 |
| COCO_independent_r60_d3003 | J | ANCHOR_EPS0_rho0.594 | 0.9500 | 0.1706 | 0.0153 | 0.0052 | 0.0441 |
| COCO_independent_r60_d3003 | BCE | ANCHOR_EPS0_rho0.594 | 0.9000 | 0.1953 | 0.0200 | 0.0078 | 0.0844 |
| COCO_independent_r60_d3003 | BCE | ANCHOR_EPS0_rho0.594 | 0.9500 | 0.1602 | 0.0000 | 0.0000 | 0.0420 |
| COCO_independent_r80_d3003 | J | LOWER_ENVELOPE_GRID | 0.9000 | 0.0000 | NA | 0.0000 | 0.1000 |
| COCO_independent_r80_d3003 | J | LOWER_ENVELOPE_GRID | 0.9500 | 0.0000 | NA | 0.0000 | 0.0500 |
| COCO_independent_r80_d3003 | BCE | LOWER_ENVELOPE_GRID | 0.9000 | 0.0000 | NA | 0.0000 | 0.1000 |
| COCO_independent_r80_d3003 | BCE | LOWER_ENVELOPE_GRID | 0.9500 | 0.0000 | NA | 0.0000 | 0.0500 |
| COCO_independent_r80_d3003 | J | ANCHOR_EPS0_rho0.798 | 0.9000 | 0.1992 | 0.0196 | 0.0078 | 0.0840 |
| COCO_independent_r80_d3003 | J | ANCHOR_EPS0_rho0.798 | 0.9500 | 0.1615 | 0.0081 | 0.0026 | 0.0432 |
| COCO_independent_r80_d3003 | BCE | ANCHOR_EPS0_rho0.798 | 0.9000 | 0.1862 | 0.0140 | 0.0052 | 0.0840 |
| COCO_independent_r80_d3003 | BCE | ANCHOR_EPS0_rho0.798 | 0.9500 | 0.1406 | 0.0000 | 0.0000 | 0.0430 |
| Flickr30K_independent_r60_d1001 | J | LOWER_ENVELOPE_GRID | 0.9000 | 0.0000 | NA | 0.0000 | 0.1000 |
| Flickr30K_independent_r60_d1001 | J | LOWER_ENVELOPE_GRID | 0.9500 | 0.0000 | NA | 0.0000 | 0.0500 |
| Flickr30K_independent_r60_d1001 | BCE | LOWER_ENVELOPE_GRID | 0.9000 | 0.0000 | NA | 0.0000 | 0.1000 |
| Flickr30K_independent_r60_d1001 | BCE | LOWER_ENVELOPE_GRID | 0.9500 | 0.0000 | NA | 0.0000 | 0.0500 |
| Flickr30K_independent_r60_d1001 | J | ANCHOR_EPS0_rho0.541 | 0.9000 | 0.2305 | 0.0000 | 0.0000 | 0.0770 |
| Flickr30K_independent_r60_d1001 | J | ANCHOR_EPS0_rho0.541 | 0.9500 | 0.1719 | 0.0000 | 0.0000 | 0.0414 |
| Flickr30K_independent_r60_d1001 | BCE | ANCHOR_EPS0_rho0.541 | 0.9000 | 0.2539 | 0.0000 | 0.0000 | 0.0746 |
| Flickr30K_independent_r60_d1001 | BCE | ANCHOR_EPS0_rho0.541 | 0.9500 | 0.1953 | 0.0000 | 0.0000 | 0.0402 |
| Flickr30K_independent_r80_d1001 | J | LOWER_ENVELOPE_GRID | 0.9000 | 0.0000 | NA | 0.0000 | 0.1000 |
| Flickr30K_independent_r80_d1001 | J | LOWER_ENVELOPE_GRID | 0.9500 | 0.0000 | NA | 0.0000 | 0.0500 |
| Flickr30K_independent_r80_d1001 | BCE | LOWER_ENVELOPE_GRID | 0.9000 | 0.0000 | NA | 0.0000 | 0.1000 |
| Flickr30K_independent_r80_d1001 | BCE | LOWER_ENVELOPE_GRID | 0.9500 | 0.0000 | NA | 0.0000 | 0.0500 |
| Flickr30K_independent_r80_d1001 | J | ANCHOR_EPS0_rho0.750 | 0.9000 | 0.2305 | 0.0000 | 0.0000 | 0.0770 |
| Flickr30K_independent_r80_d1001 | J | ANCHOR_EPS0_rho0.750 | 0.9500 | 0.1719 | 0.0000 | 0.0000 | 0.0414 |
| Flickr30K_independent_r80_d1001 | BCE | ANCHOR_EPS0_rho0.750 | 0.9000 | 0.2422 | 0.0000 | 0.0000 | 0.0758 |
| Flickr30K_independent_r80_d1001 | BCE | ANCHOR_EPS0_rho0.750 | 0.9500 | 0.1797 | 0.0000 | 0.0000 | 0.0410 |
| Flickr30K_independent_r60_d2002 | J | LOWER_ENVELOPE_GRID | 0.9000 | 0.0000 | NA | 0.0000 | 0.1000 |
| Flickr30K_independent_r60_d2002 | J | LOWER_ENVELOPE_GRID | 0.9500 | 0.0000 | NA | 0.0000 | 0.0500 |
| Flickr30K_independent_r60_d2002 | BCE | LOWER_ENVELOPE_GRID | 0.9000 | 0.0000 | NA | 0.0000 | 0.1000 |
| Flickr30K_independent_r60_d2002 | BCE | LOWER_ENVELOPE_GRID | 0.9500 | 0.0000 | NA | 0.0000 | 0.0500 |
| Flickr30K_independent_r60_d2002 | J | ANCHOR_EPS0_rho0.548 | 0.9000 | 0.2044 | 0.0064 | 0.0026 | 0.0809 |
| Flickr30K_independent_r60_d2002 | J | ANCHOR_EPS0_rho0.548 | 0.9500 | 0.1719 | 0.0000 | 0.0000 | 0.0414 |
| Flickr30K_independent_r60_d2002 | BCE | ANCHOR_EPS0_rho0.548 | 0.9000 | 0.2370 | 0.0110 | 0.0052 | 0.0789 |
| Flickr30K_independent_r60_d2002 | BCE | ANCHOR_EPS0_rho0.548 | 0.9500 | 0.1888 | 0.0069 | 0.0026 | 0.0419 |
| Flickr30K_independent_r80_d2002 | J | LOWER_ENVELOPE_GRID | 0.9000 | 0.0000 | NA | 0.0000 | 0.1000 |
| Flickr30K_independent_r80_d2002 | J | LOWER_ENVELOPE_GRID | 0.9500 | 0.0000 | NA | 0.0000 | 0.0500 |
| Flickr30K_independent_r80_d2002 | BCE | LOWER_ENVELOPE_GRID | 0.9000 | 0.0000 | NA | 0.0000 | 0.1000 |
| Flickr30K_independent_r80_d2002 | BCE | LOWER_ENVELOPE_GRID | 0.9500 | 0.0000 | NA | 0.0000 | 0.0500 |
| Flickr30K_independent_r80_d2002 | J | ANCHOR_EPS0_rho0.682 | 0.9000 | 0.1328 | 0.0000 | 0.0000 | 0.0867 |
| Flickr30K_independent_r80_d2002 | J | ANCHOR_EPS0_rho0.682 | 0.9500 | 0.0938 | 0.0000 | 0.0000 | 0.0453 |
| Flickr30K_independent_r80_d2002 | BCE | ANCHOR_EPS0_rho0.682 | 0.9000 | 0.1927 | 0.0068 | 0.0026 | 0.0820 |
| Flickr30K_independent_r80_d2002 | BCE | ANCHOR_EPS0_rho0.682 | 0.9500 | 0.1641 | 0.0000 | 0.0000 | 0.0418 |
| Flickr30K_independent_r60_d3003 | J | LOWER_ENVELOPE_GRID | 0.9000 | 0.0000 | NA | 0.0000 | 0.1000 |
| Flickr30K_independent_r60_d3003 | J | LOWER_ENVELOPE_GRID | 0.9500 | 0.0000 | NA | 0.0000 | 0.0500 |
| Flickr30K_independent_r60_d3003 | BCE | LOWER_ENVELOPE_GRID | 0.9000 | 0.0000 | NA | 0.0000 | 0.1000 |
| Flickr30K_independent_r60_d3003 | BCE | LOWER_ENVELOPE_GRID | 0.9500 | 0.0000 | NA | 0.0000 | 0.0500 |
| Flickr30K_independent_r60_d3003 | J | ANCHOR_EPS0_rho0.607 | 0.9000 | 0.2956 | 0.0220 | 0.0130 | 0.0770 |
| Flickr30K_independent_r60_d3003 | J | ANCHOR_EPS0_rho0.607 | 0.9500 | 0.2578 | 0.0152 | 0.0078 | 0.0410 |
| Flickr30K_independent_r60_d3003 | BCE | ANCHOR_EPS0_rho0.607 | 0.9000 | 0.2878 | 0.0226 | 0.0130 | 0.0777 |
| Flickr30K_independent_r60_d3003 | BCE | ANCHOR_EPS0_rho0.607 | 0.9500 | 0.2383 | 0.0164 | 0.0078 | 0.0420 |
| Flickr30K_independent_r80_d3003 | J | LOWER_ENVELOPE_GRID | 0.9000 | 0.0000 | NA | 0.0000 | 0.1000 |
| Flickr30K_independent_r80_d3003 | J | LOWER_ENVELOPE_GRID | 0.9500 | 0.0000 | NA | 0.0000 | 0.0500 |
| Flickr30K_independent_r80_d3003 | BCE | LOWER_ENVELOPE_GRID | 0.9000 | 0.0000 | NA | 0.0000 | 0.1000 |
| Flickr30K_independent_r80_d3003 | BCE | LOWER_ENVELOPE_GRID | 0.9500 | 0.0000 | NA | 0.0000 | 0.0500 |
| Flickr30K_independent_r80_d3003 | J | ANCHOR_EPS0_rho0.826 | 0.9000 | 0.2878 | 0.0226 | 0.0130 | 0.0777 |
| Flickr30K_independent_r80_d3003 | J | ANCHOR_EPS0_rho0.826 | 0.9500 | 0.2305 | 0.0169 | 0.0078 | 0.0424 |
| Flickr30K_independent_r80_d3003 | BCE | ANCHOR_EPS0_rho0.826 | 0.9000 | 0.2878 | 0.0226 | 0.0130 | 0.0777 |
| Flickr30K_independent_r80_d3003 | BCE | ANCHOR_EPS0_rho0.826 | 0.9500 | 0.2383 | 0.0164 | 0.0078 | 0.0420 |

## 8. Extra-information references (oracle, regime O)

| domain / method / target FAR | n | Platt cov@.90 | Platt risk@.90 | null cov | null risk | null FAR |
|---|---|---|---|---|---|---|
| COCO / CLEAN_TOP1_PLATT_TARGET / None | 3 | 0.1710 | 0.0222 | NA | NA | NA |
| COCO / MATCHED_NULL_MAX_TARGET / 0.05 | 3 | NA | NA | 0.3524 | 0.1524 | 0.0616 |
| COCO / MATCHED_NULL_MAX_TARGET / 0.1 | 3 | NA | NA | 0.4197 | 0.2010 | 0.1050 |
| COCO_to_CC / CLEAN_TOP1_PLATT_SOURCE / None | 3 | 0.0786 | 0.0057 | NA | NA | NA |
| COCO_to_CC / MATCHED_NULL_MAX_SOURCE / 0.05 | 3 | NA | NA | 0.1832 | 0.0335 | 0.0043 |
| COCO_to_CC / MATCHED_NULL_MAX_SOURCE / 0.1 | 3 | NA | NA | 0.2196 | 0.0454 | 0.0095 |
| COCO_to_Flickr30K / CLEAN_TOP1_PLATT_SOURCE / None | 3 | 0.1966 | 0.0118 | NA | NA | NA |
| COCO_to_Flickr30K / MATCHED_NULL_MAX_SOURCE / 0.05 | 3 | NA | NA | 0.3945 | 0.0653 | 0.0339 |
| COCO_to_Flickr30K / MATCHED_NULL_MAX_SOURCE / 0.1 | 3 | NA | NA | 0.4488 | 0.1093 | 0.0642 |
| Flickr30K / CLEAN_TOP1_PLATT_TARGET / None | 3 | 0.2539 | 0.0202 | NA | NA | NA |
| Flickr30K / MATCHED_NULL_MAX_TARGET / 0.05 | 3 | NA | NA | 0.3911 | 0.0712 | 0.0347 |
| Flickr30K / MATCHED_NULL_MAX_TARGET / 0.1 | 3 | NA | NA | 0.4727 | 0.1318 | 0.0885 |
| Flickr30K_to_CC / CLEAN_TOP1_PLATT_SOURCE / None | 3 | 0.1024 | 0.0084 | NA | NA | NA |
| Flickr30K_to_CC / MATCHED_NULL_MAX_SOURCE / 0.05 | 3 | NA | NA | 0.1749 | 0.0328 | 0.0035 |
| Flickr30K_to_CC / MATCHED_NULL_MAX_SOURCE / 0.1 | 3 | NA | NA | 0.2552 | 0.0512 | 0.0130 |
| Flickr30K_to_COCO / CLEAN_TOP1_PLATT_SOURCE / None | 3 | 0.2218 | 0.0715 | NA | NA | NA |
| Flickr30K_to_COCO / MATCHED_NULL_MAX_SOURCE / 0.05 | 3 | NA | NA | 0.3602 | 0.1476 | 0.0642 |
| Flickr30K_to_COCO / MATCHED_NULL_MAX_SOURCE / 0.1 | 3 | NA | NA | 0.4605 | 0.2209 | 0.1267 |

## 9. Integrity, projections, cost

Ranking integrity (every prediction top_index == base argmax): True. Base R@K per gallery is in receipts/*/INTEGRITY.json and is identical for every method by construction.

| case | method | target | post-proj mean | post-proj max | bg-proj mean | bg-proj max |
|---|---|---|---|---|---|---|
| COCO_hard_r60_d1001 | CLEAN_BCE_ORACLE | CC | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| COCO_hard_r60_d1001 | CLEAN_BCE_ORACLE | COCO | 0.0000 | 0.0000 | 0.0002 | 0.0005 |
| COCO_hard_r60_d1001 | CLEAN_BCE_ORACLE | Flickr30K | 0.0000 | 0.0000 | 0.0001 | 0.0002 |
| COCO_hard_r60_d1001 | CLEAN_J_ORACLE | CC | 0.0000 | 0.0000 | 0.0001 | 0.0002 |
| COCO_hard_r60_d1001 | CLEAN_J_ORACLE | COCO | 0.0000 | 0.0000 | 0.0003 | 0.0007 |
| COCO_hard_r60_d1001 | CLEAN_J_ORACLE | Flickr30K | 0.0000 | 0.0000 | 0.0004 | 0.0007 |
| COCO_hard_r60_d1001 | FORWARD_BCE | CC | 0.0000 | 0.0000 | 0.0005 | 0.0012 |
| COCO_hard_r60_d1001 | FORWARD_BCE | COCO | 0.0000 | 0.0000 | 0.0006 | 0.0012 |
| COCO_hard_r60_d1001 | FORWARD_BCE | Flickr30K | 0.0000 | 0.0000 | 0.0009 | 0.0020 |
| COCO_hard_r60_d1001 | FORWARD_J | CC | 0.0000 | 0.0000 | 0.0015 | 0.0032 |
| COCO_hard_r60_d1001 | FORWARD_J | COCO | 0.0000 | 0.0000 | 0.0023 | 0.0044 |
| COCO_hard_r60_d1001 | FORWARD_J | Flickr30K | 0.0000 | 0.0000 | 0.0027 | 0.0056 |
| COCO_hard_r60_d1001 | NAIVE_BCE | CC | 0.0000 | 0.0000 | 0.0001 | 0.0002 |
| COCO_hard_r60_d1001 | NAIVE_BCE | COCO | 0.0000 | 0.0000 | 0.0003 | 0.0007 |
| COCO_hard_r60_d1001 | NAIVE_BCE | Flickr30K | 0.0000 | 0.0000 | 0.0004 | 0.0007 |
| COCO_hard_r60_d1001 | NAIVE_J | CC | 0.0000 | 0.0000 | 0.0003 | 0.0007 |
| COCO_hard_r60_d1001 | NAIVE_J | COCO | 0.0000 | 0.0000 | 0.0006 | 0.0012 |
| COCO_hard_r60_d1001 | NAIVE_J | Flickr30K | 0.0000 | 0.0000 | 0.0008 | 0.0017 |
| COCO_hard_r60_d1001 | POST_PROJECTED_BCE | CC | 0.9811 | 0.9824 | 0.0002 | 0.0005 |
| COCO_hard_r60_d1001 | POST_PROJECTED_BCE | COCO | 0.9799 | 0.9824 | 0.0005 | 0.0012 |
| COCO_hard_r60_d1001 | POST_PROJECTED_BCE | Flickr30K | 0.9650 | 0.9693 | 0.0006 | 0.0012 |
| COCO_hard_r60_d1001 | POST_PROJECTED_J | CC | 0.9809 | 0.9824 | 0.0007 | 0.0017 |
| COCO_hard_r60_d1001 | POST_PROJECTED_J | COCO | 0.9797 | 0.9822 | 0.0007 | 0.0015 |
| COCO_hard_r60_d1001 | POST_PROJECTED_J | Flickr30K | 0.9646 | 0.9690 | 0.0011 | 0.0024 |
| COCO_hard_r60_d2002 | CLEAN_BCE_ORACLE | CC | 0.0000 | 0.0000 | 0.0000 | 0.0001 |
| COCO_hard_r60_d2002 | CLEAN_BCE_ORACLE | COCO | 0.0000 | 0.0000 | 0.0002 | 0.0005 |
| COCO_hard_r60_d2002 | CLEAN_BCE_ORACLE | Flickr30K | 0.0000 | 0.0000 | 0.0003 | 0.0007 |
| COCO_hard_r60_d2002 | CLEAN_J_ORACLE | CC | 0.0000 | 0.0000 | 0.0001 | 0.0002 |
| COCO_hard_r60_d2002 | CLEAN_J_ORACLE | COCO | 0.0000 | 0.0000 | 0.0003 | 0.0007 |
| COCO_hard_r60_d2002 | CLEAN_J_ORACLE | Flickr30K | 0.0000 | 0.0000 | 0.0004 | 0.0007 |
| COCO_hard_r60_d2002 | FORWARD_BCE | CC | 0.0000 | 0.0000 | 0.0005 | 0.0010 |
| COCO_hard_r60_d2002 | FORWARD_BCE | COCO | 0.0000 | 0.0000 | 0.0007 | 0.0015 |
| COCO_hard_r60_d2002 | FORWARD_BCE | Flickr30K | 0.0000 | 0.0000 | 0.0010 | 0.0024 |
| COCO_hard_r60_d2002 | FORWARD_J | CC | 0.0000 | 0.0000 | 0.0015 | 0.0032 |
| COCO_hard_r60_d2002 | FORWARD_J | COCO | 0.0000 | 0.0000 | 0.0028 | 0.0061 |
| COCO_hard_r60_d2002 | FORWARD_J | Flickr30K | 0.0000 | 0.0000 | 0.0027 | 0.0059 |
| COCO_hard_r60_d2002 | NAIVE_BCE | CC | 0.0000 | 0.0000 | 0.0001 | 0.0002 |
| COCO_hard_r60_d2002 | NAIVE_BCE | COCO | 0.0000 | 0.0000 | 0.0003 | 0.0007 |
| COCO_hard_r60_d2002 | NAIVE_BCE | Flickr30K | 0.0000 | 0.0000 | 0.0004 | 0.0007 |
| COCO_hard_r60_d2002 | NAIVE_J | CC | 0.0000 | 0.0000 | 0.0003 | 0.0005 |
| COCO_hard_r60_d2002 | NAIVE_J | COCO | 0.0000 | 0.0000 | 0.0005 | 0.0012 |
| COCO_hard_r60_d2002 | NAIVE_J | Flickr30K | 0.0000 | 0.0000 | 0.0007 | 0.0015 |
| COCO_hard_r60_d2002 | POST_PROJECTED_BCE | CC | 0.9753 | 0.9798 | 0.0003 | 0.0005 |
| COCO_hard_r60_d2002 | POST_PROJECTED_BCE | COCO | 0.9801 | 0.9821 | 0.0005 | 0.0012 |
| COCO_hard_r60_d2002 | POST_PROJECTED_BCE | Flickr30K | 0.9657 | 0.9675 | 0.0007 | 0.0015 |
| COCO_hard_r60_d2002 | POST_PROJECTED_J | CC | 0.9750 | 0.9794 | 0.0006 | 0.0012 |
| COCO_hard_r60_d2002 | POST_PROJECTED_J | COCO | 0.9798 | 0.9819 | 0.0008 | 0.0017 |
| COCO_hard_r60_d2002 | POST_PROJECTED_J | Flickr30K | 0.9651 | 0.9668 | 0.0011 | 0.0024 |
| COCO_hard_r60_d3003 | CLEAN_BCE_ORACLE | CC | 0.0000 | 0.0000 | 0.0001 | 0.0002 |
| COCO_hard_r60_d3003 | CLEAN_BCE_ORACLE | COCO | 0.0000 | 0.0000 | 0.0001 | 0.0002 |
| COCO_hard_r60_d3003 | CLEAN_BCE_ORACLE | Flickr30K | 0.0000 | 0.0000 | 0.0005 | 0.0012 |
| COCO_hard_r60_d3003 | CLEAN_J_ORACLE | CC | 0.0000 | 0.0000 | 0.0002 | 0.0005 |
| COCO_hard_r60_d3003 | CLEAN_J_ORACLE | COCO | 0.0000 | 0.0000 | 0.0002 | 0.0002 |
| COCO_hard_r60_d3003 | CLEAN_J_ORACLE | Flickr30K | 0.0000 | 0.0000 | 0.0006 | 0.0015 |
| COCO_hard_r60_d3003 | FORWARD_BCE | CC | 0.0000 | 0.0000 | 0.0004 | 0.0007 |
| COCO_hard_r60_d3003 | FORWARD_BCE | COCO | 0.0000 | 0.0000 | 0.0010 | 0.0022 |
| COCO_hard_r60_d3003 | FORWARD_BCE | Flickr30K | 0.0000 | 0.0000 | 0.0010 | 0.0022 |
| COCO_hard_r60_d3003 | FORWARD_J | CC | 0.0000 | 0.0000 | 0.0013 | 0.0024 |
| COCO_hard_r60_d3003 | FORWARD_J | COCO | 0.0000 | 0.0000 | 0.0027 | 0.0051 |
| COCO_hard_r60_d3003 | FORWARD_J | Flickr30K | 0.0000 | 0.0000 | 0.0029 | 0.0061 |

| stage | CPU wall seconds (sum) |
|---|---|
| bank_export | 18.3178 |
| cc_build | 15.4011 |
| curate | 791.4793 |
| d1 | 94.3982 |
| evaluate | 310.9381 |
| fit | 40.4034 |
| integrity | 5.0668 |
| oracle_eval | 3.3824 |
| oracle_fit | 5.8539 |
| predict | 539.6876 |
| rho_eval | 60.4056 |
| rho_fit | 44.1818 |
| rho_predict | 195.7083 |

No GPU was used. External baselines: DEDPUL PENDING (not integrated, no comparison claimed); MA-PaPSP/NACP not implemented. D3/OT not run; D4 NA.
