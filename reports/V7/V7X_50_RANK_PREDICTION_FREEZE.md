> **SUPERSEDED (2026-10-14, before any GenAI-Rank label was read) by `V7X_50_RANK_PREDICTION_FREEZE_v2.md`.**  v1 tie-filled the M-arm E readouts on the 16 decomposition-failed pools; the frozen deployment rule (`scripts/v7/v7_parser.py`, `scripts/v7/v7_train.py`) falls back to O there, as the V7 supplement v3 closeout package (docs/03 §4) pointed out.  v2 applies the frozen rule; X21 rows follow their own 'no score' rule with a separately named neutral-fill sensitivity.  v1 files are kept for the record and are never scored.

# V7X-50 — GenAI-Rank prediction freeze (2026-10-14T20:25:55Z, label-free)

Universe: `genai_rank_inputs.jsonl` (n = 14400, sha256 `514b9368fd5dffb5…`).  `human_rating.json` was not opened.  All files valid: **True**.  Converted files and `methods.json` (sha256 `d1f598d560a51c6d…`) in `/projects/EEG-foundation-model/yinghao/outputs_v7/x50_rank_freeze_v1`.

| method | family | seed files | rows | valid | converted sha256 (first 16) |
|---|---|---|---|---|---|
| M2_E | v2_external_primary (ours); v2_attribution; v1_registered | 3 | 14400, 14400, 14400 | True | 93dd3b2fe3903fd1 bb4dd5ce0e720801 2642bca06bbbed91 |
| B0_FGA_released_O | v2_external_primary; v1_registered | 1 | 14400 | True | 582911e16e13923c |
| B1_VQAScore | v2_external_primary; v1_registered | 1 | 14400 | True | ea0c82e3f957473e |
| B4T_TIFA | v2_external_primary | 1 | 14400 | True | 225483371ea33a2d |
| B4D_DSG | v2_external_primary | 1 | 14400 | True | cf5105484d28b1d6 |
| M1_E | v2_attribution; v1_registered | 3 | 14400, 14400, 14400 | True | 7172335ec6dc7fd6 5272818f2eded8dd d37c5bc77b516cf0 |
| M2_O | secondary | 3 | 14400, 14400, 14400 | True | 4f4e22ddf5f4bd44 44aadd109e1940bf 5fc80a2763507097 |
| M1_O | secondary | 3 | 14400, 14400, 14400 | True | 8855620b9b3e6dde 3373d8add9068021 890850d4d7b63acd |
| M0_E | secondary | 3 | 14400, 14400, 14400 | True | 20a7b0a3626dbc49 96a821257bf112e6 5fa1dba24c17ae99 |
| M0_O | secondary | 3 | 14400, 14400, 14400 | True | 5f8b64f43ac45482 1304680a0f9076b3 f3856d8b775dad05 |
| N0_O | secondary (native FGA trainer reproduction) | 3 | 14400, 14400, 14400 | True | f7cc35f5e3fe90c6 223e3018949a97c3 99d2326c294f1868 |
| B2_VQAScore_Qwen25VL7B | secondary | 1 | 14400 | True | 2670f124b708053d |
| B3_PNVQA_own | secondary | 1 | 14400 | True | c085875d275b8abf |
| B3P_PNVQA_parser | secondary | 1 | 14400 | True | bf02519aa6a626f6 |
| B5_HPSv3 | secondary | 1 | 14400 | True | b2643501ace20916 |
| CLIPScore_sanity | secondary (sanity row) | 1 | 14400 | True | fd58f5f2f3c10291 |
| X21_N-VCS-F_E | secondary (V7X-20 common features) | 3 | 14400, 14400, 14400 | True | 5003021b7fa96799 f3f3b818a2580be6 a0984563ca5178f4 |
| X21_N-BCE-F_E | secondary (V7X-20 common features) | 3 | 14400, 14400, 14400 | True | 8ebd8640978cd3e6 af63ff6217a34c58 eeff3c58ac397761 |
| X21_N-DV_E | secondary (V7X-20 common features) | 3 | 14400, 14400, 14400 | True | 2b69552bdd079fb4 b4852d336504ae13 98d5402c97555fa4 |
| X21_N-NWJ_E | secondary (V7X-20 common features) | 3 | 14400, 14400, 14400 | True | ee97d10778aececf f04b609c1836a501 611f73f2270e8290 |
| X21_K-KDE-S_E | secondary (V7X-20 common features) | 1 | 14400 | True | e73c939238dd9cda |
| X21_K-NYSTROM_E | secondary (V7X-20 common features) | 1 | 14400 | True | 9fe7575100eb334a |
| X21_K-RULSIF_E | secondary (V7X-20 common features) | 1 | 14400 | True | 7a4c944fb6b463be |

Undefined-score rule: a method's undefined score (flagged failure: E with decomposition_failed, or no questions) is filled with one constant for all 9 candidates of that pool, so the official first-index tie rule selects candidate 0 - the same convention TIFA / DSG / PN-VQA already use for their own failures; fixed before any GenAI-Rank label is read.

Pre-declared sensitivity: the primary and attribution families rescored without the 82 pools where any of their methods has a flagged failure (list in the JSON).  all-tied pools (a method's 9 scores equal) are counted per file; coarse scores (TIFA / DSG answer fractions) tie naturally; the official first-index rule applies, and the frozen scorer's 'official/mean' tie-expectation sensitivity covers them.

| method | tie-filled pools | flagged-failure pools | all-tied pools (natural + filled; per seed file) |
|---|---|---|---|
| M2_E | 16, 16, 16 | 16, 16, 16 | 16, 16, 16 |
| B0_FGA_released_O | 0 | 0 | 0 |
| B1_VQAScore | 0 | 0 | 0 |
| B4T_TIFA | 0 | 66 | 344 |
| B4D_DSG | 0 | 2 | 523 |
| M1_E | 16, 16, 16 | 16, 16, 16 | 16, 16, 16 |
| M2_O | 0, 0, 0 | 16, 16, 16 | 0, 0, 0 |
| M1_O | 0, 0, 0 | 16, 16, 16 | 0, 0, 0 |
| M0_E | 16, 16, 16 | 16, 16, 16 | 16, 16, 16 |
| M0_O | 0, 0, 0 | 16, 16, 16 | 0, 0, 0 |
| N0_O | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 |
| B2_VQAScore_Qwen25VL7B | 0 | 0 | 0 |
| B3_PNVQA_own | 0 | 0 | 0 |
| B3P_PNVQA_parser | 0 | 16 | 16 |
| B5_HPSv3 | 0 | 0 | 0 |
| CLIPScore_sanity | 0 | 0 | 0 |
| X21_N-VCS-F_E | 16, 16, 16 | 16, 16, 16 | 16, 16, 16 |
| X21_N-BCE-F_E | 16, 16, 16 | 16, 16, 16 | 16, 16, 16 |
| X21_N-DV_E | 16, 16, 16 | 16, 16, 16 | 16, 16, 16 |
| X21_N-NWJ_E | 16, 16, 16 | 16, 16, 16 | 16, 16, 16 |
| X21_K-KDE-S_E | 16 | 16 | 16 |
| X21_K-NYSTROM_E | 16 | 16 | 16 |
| X21_K-RULSIF_E | 16 | 16 | 16 |

Scoring (only after the owner's separate approval; frozen scorer, new output path, `--allow-scoring`):

- v2_external_primary: `python -I scripts/v7/v7_score.py rank --methods <methods.json> --labels <sealed human_rating> --out <new path> --ours M2_E --comparators B0_FGA_released_O B1_VQAScore B4T_TIFA B4D_DSG --allow-scoring`
- v2_attribution: `... --ours M2_E --comparators M1_E --allow-scoring`
- v1_registered: `... --ours M2_E --comparators M1_E B0_FGA_released_O B1_VQAScore --allow-scoring`
