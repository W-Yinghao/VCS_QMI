# V7C-03 — external-baseline readiness (label-free; 2026-10-15)

No human rating of any kind is read here.  The comparisons use the authors' released question files (method outputs) and the frozen local runs.

## 1. Completeness of the primary comparators on GenAI-Rank (freeze v3)
| row | interface | candidates | validation | GPU cost (wall clock, this run) |
|---|---|---|---|---|
| B0 FGA-BLIP2 (released) | native overall | 14,400 / 14,400 | exact universe, finite | native eval (V7 N0 harness) |
| B1 VQAScore CLIP-FlanT5-XXL | t2v-metrics 3.0 | 14,400 / 14,400 | exact, finite | 39 min (L40S) |
| B4T TIFA | released local LLaMA-2 generator + UnifiedQA filter + mPLUG-large | 14,400 / 14,400 | exact, finite | questions 3.5 h + VQA 2.3 h (L40S) |
| B4D DSG | DSG prompts with Qwen2.5-32B-Instruct (labelled adaptation) + mPLUG-large | 14,400 / 14,400 | exact, finite | generation ≈ 2.5 GPU-h (H100 NVL / RTX) + VQA ≈ 2.4 GPU-h (3 × L40S) |

Secondary rows (B2, B3, B3-P, B5 HPSv3, CLIPScore, X21, the M-arm O / H / E_cal) are complete in the same freeze.

## 2. Failure sets and their frozen handling
- **TIFA.**
  - 31 of 800 prompts failed to parse.
  - 33 prompts have no question left after the filter; the 31 parse failures are a strict subset of these 33.
  - All 66 corresponding pools score 0.0 for every candidate. They therefore tie, the official first-index rule applies, and they stay in the denominator.
- **DSG (GenAI-Rank).**
  - 7 of 800 prompts have a parse error; 2 pools, from 1 prompt, have no question.
  - The dependency graphs are valid: no missing parent and no cycle.
  - 282 of 2,064 entity tuples have a parent. These are part-of-whole tuples; DSG-1k's own rate is 19 %.
  - The local generator writes 325 "action - (…)" tuples where DSG's taxonomy says "relation - action", and 365 tuples have an empty detailed category. DSG's score uses the questions and dependencies, not the category label.
- **M arms.** 16 pools fall back to O (the frozen rule).

## 3. Source fidelity (fixed sets, seed 2718; `scripts/v7x/v7c_fidelity_prep.py`, `v7c_fidelity_compare.py`; jobs 1040534, 1040718)
| | local | released | agreement |
|---|---|---|---|
| **DSG** (50 DSG-1k items outside tifa160; released = PaLM 2 annotations) | 6.50 tuples per prompt (median 6) | 6.88 (median 6) | tuple content (argument-token Jaccard ≥ 0.5, one-to-one): precision 0.86, recall 0.81; with the category also required: 0.71 / 0.67 |
| | parse errors 2 / 50; dependencies valid; entity-with-parent 13 / 129 | 14 / 128 | broad categories local / released: entity 129 / 128, attribute 65 / 88, relation 40 / 72, global 56 / 38, other 20 / 18, outside the taxonomy 15 / 0 |
| **TIFA** (50 TIFA v1.0 captions; released = GPT-3.5 questions after TIFA's filter) | 7.86 generated, **5.92 kept** per prompt (median 6) | 7.02 (median 7) | element precision 0.91, element recall 0.78 |
| | parse failure 1 / 50; no kept question 1 / 50 | — | yes / no share 0.69 local vs 0.74 released |

**Reading** (descriptive; no pre-fixed threshold existed for this check):
- **DSG.** The local adaptation reproduces the content of the released graphs closely: about 81–86 % of tuple contents match, with near-identical counts and the same dependency structure. Its main departure is how tuples are labelled into DSG's categories.
- **TIFA.** The released local generator keeps about 16 % fewer questions than the released GPT-3.5 set and covers about 78 % of its elements. Nearly all the elements it does ask about are released elements (91 %).

Neither departure is a failure of the run. Both are recorded as properties of the local routes, labelled as such in every table: TIFA's route is official, while DSG's is an adaptation.

**Blind review table.** For the owner's manual check, local vs released question lists sit in random A / B order per prompt in `outputs/V7_CLOSEOUT/fidelity/blind_review_table.csv`; the key is kept separately in `blind_review_key.json`. The table holds benchmark prompt text, so it is not in the public repository.

## 4. Consequence for V7C-04
- All four primary comparators are runnable, complete and frozen, so no family revision is needed.
- The paid-API routes stay disabled (owner, 2026-10-14). These fidelity numbers are the evidence for the API necessity review in V7X-30.
- The frozen local rows are scored as registered.
