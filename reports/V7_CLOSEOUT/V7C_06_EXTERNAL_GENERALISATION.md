# V7C-06 — external generalisation: inventory, overlap and label exposure (2026-10-15; no label read)

## 1. Sources and status
| source | content | V7 predictions | human labels | overlap (by identical prompt text) |
|---|---|---|---|---|
| GenAI-Rank | 800 prompts × DALLE_3 / SDXL_Base × 9 candidates | frozen (V7X-50 v3) | **read 2026-10-15 (V7C-04)** | EvalMuse TRAIN 0 |
| GenAI-Bench | 1,600 prompts × 6 generators (DALLE_3, DeepFloyd-I-XL, Midjourney-6, SDXL-2.1, SDXL-Base, SDXL-Turbo) = 9,600 images | complete for every V7 method except DSG (group `v7x_grp_bench_dsg`, waiting for an H100) | **sealed, not approved** | 750 prompts identical to GenAI-Rank prompts; **850 not in GenAI-Rank**; EvalMuse TRAIN 0 |
| TIFA160 | 160 prompts; Likert ratings and human annotations | not run | **sealed in `third_party/closed/`** (moved unopened, hashed) | — |
| RichHF (alignment part) | — | not run | not downloaded | — |

## 2. What each could support, given current exposure
- **GenAI-Bench, 750 Rank-overlap prompts.** The images and generators differ, but the prompts were seen in V7's GenAI-Rank scoring. Under package docs/04 §6 this is not independent validation of V7.
- **GenAI-Bench, 850 non-Rank prompts.** These are the only large unexposed external pool. **They are registered as V8's untouched confirmation source** (V8-03, frozen before the V7 read). Scoring V7 on them now would expose them, and V8 would then need TIFA160 or RichHF instead (V8-03 fallback).
- **TIFA160.** Small (160 prompts), with a different deployment format (one image per prompt; correlation, not best-of-N). It needs its own approval and an overlap count against GenAI and EvalMuse before any read.
- **RichHF alignment.** It would need a download, data card and terms check first (data-download rules). Only its official alignment dimension is relevant.

## 3. Owner decision requested (nothing is read until then)
1. Keep GenAI-Bench sealed for V8's confirmation (recommended): V7's external result stays the GenAI-Rank one.
2. Or score V7 on the GenAI-Bench 850 non-Rank prompts now: V7 gains a generalisation table, and V8's confirmation moves to TIFA160 or RichHF.
3. Independently, whether TIFA160 may be read, and whether RichHF should be downloaded.
