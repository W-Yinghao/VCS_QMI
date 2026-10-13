# V7-01 data audit: EvalMuse public TRAIN (2026-10-13)

Source: `train_list.json` sha256 `a3ea6941fd5f727a…`, 32,717 records; archive 43,534 members, 0 TRAIN members missing, 10,796 non-TRAIN files never read.  Rules: `scripts/v7/v7_data.py` (fixed before any model output).  Manifests: `outputs/V7/data/manifest_{images,elements}.jsonl` (sha256 `53584c8d021d…`, `631799c31880…`).

## 1. Raw fields and codes

| field | present | types | codes (top) |
|---|---|---|---|
| `prompt_id` | 32,717 | str 32,717 | — |
| `prompt` | 32,717 | str 32,717 | — |
| `type` | 32,717 | str 32,717 | real: 22,323, synth: 10,394 |
| `img_path` | 32,717 | str 32,717 | — |
| `total_score` | 32,717 | list 32,717 | 2: 35,353, 3: 34,800, 4: 21,332, 5: 16,456, 1: 13,148 |
| `element_score` | 32,717 | dict 32,717 | 1: 364,397, 0: 248,104 |
| `promt_meaningless` | 32,717 | list 32,717 | 0: 109,413, 1: 6,390, -1: 5,286 |
| `split_confidence` | 32,717 | list 32,717 | 0: 87,195, 1: 28,608, -1: 5,286 |
| `attribute_confidence` | 32,717 | list 32,717 | 1: 78,255, 0: 37,548, -1: 5,286 |
| `fidelity_label` | 32,717 | list 32,717 | — |

Element votes use only 0 / 1; the three image flags use 0 / 1 with −1 = no answer (kept as missing, never 0).

## 2. Images, elements and label status

32,717 images ({'3': 25071, '6': 7646} annotators per image), 165,224 element events; 815 elements carry a negation cue (heuristic flag), 7,005 match more than once in their prompt.

Image status:

| status | count |
|---|---|
| ok | 25,707 |
| bad_decomposition | 5,662 |
| meaningless | 1,348 |

Element status:

| status | count |
|---|---|
| valid | 128,779 |
| bad_decomposition | 36,445 |

Element scope (main = valid label, meaningful prompt):

| scope | count |
|---|---|
| main | 127,663 |
| excluded_bad_decomposition | 33,537 |
| excluded_meaningless | 4,024 |

Span rule that located each element in its prompt:

| rule | count |
|---|---|
| exact_word | 165,001 |
| space_hyphen | 165 |
| unmatched | 58 |

Generators:

| generator | count |
|---|---|
| SDXL | 3,179 |
| SD_v2.1 | 3,175 |
| SSD1B | 1,773 |
| Playground_v2.5 | 1,767 |
| Kandinsky3 | 1,751 |
| SDXL-Lightning | 1,707 |
| IF | 1,705 |
| SDXL-Turbo | 1,703 |
| SD_v1.2 | 1,691 |
| LCM-SSD1B | 1,689 |
| PixArt-alpha | 1,688 |
| SD_v1.5 | 1,687 |
| LCM-PixArt | 1,681 |
| LCM-SDXL | 1,674 |
| PixArt-Sigma | 1,644 |
| Dreamina_v2.0Pro | 1,496 |
| Midjourney_v6.1 | 1,481 |
| Kolors | 425 |
| HunyuanDiT | 414 |
| SD3 | 387 |

Element categories:

| category | count |
|---|---|
| attribute | 40,131 |
| object | 28,891 |
| activity | 17,802 |
| other | 14,908 |
| location | 10,211 |
| human | 9,993 |
| animal | 5,600 |
| color | 4,086 |
| food | 4,028 |
| spatial | 3,768 |
| other-perspective | 2,816 |
| material | 2,176 |
| counting | 2,118 |
| attribute-style-artist/works | 1,873 |
| Other-time | 1,873 |
| Other-weather | 1,866 |
| Counting | 1,582 |
| Spatial | 1,581 |
| Material) | 1,569 |
| Color) | 1,568 |
| Other-writing&symbol | 1,507 |
| object-other | 1,377 |
| object-cloth&accessories | 1,107 |
| animal/human | 727 |
| object-vehicle | 683 |
| shape | 660 |
| object-electronic | 347 |
| object-household goods | 189 |
| time | 140 |
| comparison | 20 |
| environment | 7 |
| event | 7 |
| attrbiute | 7 |
| comparative | 6 |

## 3. Prompt and image identity

- 2,991 prompt ids; 0 ids carry more than one text; 2,990 normalized texts, 1 shared by several ids.
- Near-duplicate prompt-id pairs (token Jaccard ≥ 0.9): 815.
- Prompt ids overlapping GenAI-Bench prompts (exact or near-duplicate): 0 → their components are excluded from every role.
- Images: 1 groups of byte-identical images, 1287 near-duplicate pairs (dHash ≤ 4), 0 decode errors.
- Components after merging all links: 2,689 (largest 2795 images).

## 4. Roles (split seed 17071; components, so the fractions are not exact)

| role | images | prompt ids | components | main elements |
|---|---|---|---|---|
| FIT | 26,177 | 2,385 | 2,093 | 101,987 |
| CAL_SELECT | 1,653 | 145 | 145 | 6,411 |
| CAL_PROB | 1,630 | 154 | 147 | 7,249 |
| DEV | 3,257 | 307 | 304 | 12,016 |
| EXCLUDED_OVERLAP | 0 | 0 | 0 | 0 |

FIT prior (prompt → image → element uniform): π = 0.5712 over all main elements, π = 0.5713 over main elements with a span (the M training population; unweighted mean 0.6160); image weights 0.366–1.829; 101,935 of 101,987 FIT main elements have a span.

## 5. Prompt length (FIT, uncased BERT with [CLS] / [SEP])

max 85, p99 51; over 32 tokens: 1,428 of 26,177 (the native 32-token cut); over 128: 0; over 512: 0.  Fast vs model (slow) tokenizer id mismatches: 0.

