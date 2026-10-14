# V7X-30 — public scorers on the GenAI-Rank pool (and EvalMuse DEV): plan, settings, status — 2026-10-14

**Status:** plan; nothing in this file is a result.  B0 / B1 jobs are queued (V7 v1); B4T / B4D / B3 / B2 / B5 / B6 need downloads, an
environment and a smoke first.  Every method scores the same 14,400 GenAI-Rank candidates (800 prompts x 2 generators x 9 candidates, official
file order); GenAI-Rank labels stay sealed (0400) and are read only by `v7_score.py rank` after the owner's scoring approval.  Sources: v2
package configs/BASELINES.csv, docs/13; checked on 2026-10-14 against the official repositories / papers named below.

## Primary family (v2; protocol amendment V7_03_AMENDMENT_V2)
M2-E against B0, B1, B4T, B4D on DALLE_3 and SDXL_Base (N = 9): 8 contrasts, Holm.  M2-E vs M1-E (2 contrasts) is a separate attribution family.

| ID | method | source (code / weights) | setting used here | variant | exposure / caveat | status |
|---|---|---|---|---|---|---|
| B0 | FGA-BLIP2 released | EvalMuse@45c8bb4 `eval.py`; HF hanshuhao/FGA-BLIP2 @ a14ad16 | native overall score `score_result` (os), BLIP eval 364, `blip_caption` text processor | native | trained on all of EvalMuse TRAIN, which contains our CAL / DEV images: its EvalMuse DEV numbers are exposed; GenAI-Rank is not in EvalMuse (overlap audit 0) | queued (1038708 EvalMuse roles, 1038709 Rank) |
| B1 | VQAScore, CLIP-FlanT5-XXL | t2v-metrics 3.0 (scoped import of the release code; package `__init__` files skipped); HF zhiqiulin/clip-flant5-xxl @ 89bad6f | template 'Does this figure show "{}"? Please answer yes or no.' / answer 'Yes'; torch 2.5.1 env | native | scoped import is an import-path change only (V7-00) | queued (1038710) |
| B4T | TIFA | github.com/Yushi-Hu/tifa (Apache-2.0), `tifascore` | questions: the released local generator `tifa-benchmark/llama2_tifa_question_generation` (`get_llama2_pipeline`, `get_llama2_question_and_answers`); filter: `allenai/unifiedqa-v2-t5-large-1363200` (`filter_question_and_answers`); VQA: `mplug-large` (README: "We recommend using mplug-large"); score `tifa_score_single` | native (released local route) | the GPT-3.5 route is the paper's; using it needs a paid API (owner decision); if mplug-large cannot be installed, the pre-declared fallback is `blip2-flant5xl` from TIFA's own list (recorded) | to download + env + smoke |
| B4D | DSG | github.com/j-min/DSG (Apache-2.0), `dsg/` (query_utils, parse_utils, vqa_utils) | DSG's own three prompts (tuples -> dependencies -> questions) with a local LLM, Qwen2.5-32B-Instruct (bf16, greedy); VQA mPLUG-large (the example notebook's model; InstructBLIP is the other open option in `vqa_utils.py`); dependency rule as released: a child is zeroed if a parent is answered "no"; score = mean over questions | reproduction adaptation (LLM substituted) | paper: PaLM 2 generation, PaLI-17B VQA (not public); example: gpt-3.5-turbo-16k (paid API, owner decision); fallback VQA InstructBLIP | to download + env + smoke |

## Secondary and extended rows (not in the primary family)
| ID | method | source | setting used here | variant | status |
|---|---|---|---|---|---|
| B3 | PN-VQA | EvalMuse paper §4.2, §8.4-8.5 (no released code) | elements by the paper's Listing 1 template and one yes / no question per element by Listing 2 (both with the local LLM above; the paper used GPT-4); VQA Qwen2-VL-7B-Instruct (the paper's best MLLM family); "This image is generated from {prompt}. Is the answer to {question} in this image {a}?" with a = the question's answer (P_T) and the other answer (P_F); P = softmax over the Yes / No logits; S_e = (P_T + 1 - P_F) / 2; image score = mean S_e | reimplementation from the paper + LLM substitution | to implement + smoke |
| B3-P | PN-VQA on the V7 parser's elements | as B3, elements from the pinned V7 parser (shared component) | component comparison only | task adaptation | after B3 |
| B2 | VQAScore, same modern host | t2v-metrics 3.0 `qwen2.5-vl-7b` (Qwen2.5-VL-7B-Instruct @ cc594898, the parser's host) | the same template as B1 | native code, host swapped | host-upgrade effect only, not a VQAScore reproduction | to smoke |
| B5 | HPSv3 | github.com/MizzenAI/HPSv3 (MIT), HF MizzenAI/HPSv3 (Qwen2-VL-7B backbone), `HPSv3RewardInferencer.reward` -> mu | native mu per (prompt, image) | native | preference score, not a Bernoulli probability | runnability check, then score |
| B6 | VisionReward-Image | github.com/THUDM/VisionReward (Apache-2.0), HF THUDM/VisionReward-Image (CogVLM2-Llama3-19B), `inference-image.py --score` | native weighted checklist score (weights in `VisionReward_Image/weight.json`) | native | 19B bf16 (~40 GB); licence of the weights to be read before use | runnability check |
| B7 | DynEval | arXiv 2607.11199 (ECCV 2026); project page vcl-iisc.github.io/dyneval | — | — | no code or weights linked from the paper page on 2026-10-14; project page to be checked; unavailable rows record the evidence | verify |
| B8 | DEPICT | arXiv 2610.03617 (training-free answer-agreement metric) | — | — | no code linked on 2026-10-14; a reimplementation would need its own protocol (not this round) | verify |
| SANITY | constant prior / seeded random / CLIPScore (openai ViT-L/14-336) / BLIP-2 COCO ITM at init (z0 from X20) | — | floors only; cannot support "externally superior" | native | with X20 |

## Common rules (docs/13 §3)
- Every method gets its strong public setting; a mismatch is first checked against weights, resolution, text truncation, question template,
  probability readout and parsing, and both the pre- and post-fix rows are kept.
- Each method writes one prediction file per (method, seed, score readout) with `prompt_id, generator_id, image_id, candidate_index, score` and its
  runtime / GPU; files enter the GenAI-Rank prediction manifest before any scoring.  Cost = all LLM generation + all VQA calls + model loads.
- An unrunnable primary method is registered with its reason and a new frozen family before any label is opened; a stronger method is never
  dropped to make a lead easier.
- Paid APIs (OpenAI GPT-3.5 / GPT-4 for the native TIFA / DSG / PN-VQA question generation) are not used without the owner's approval.

## Budget (estimates, to be replaced by smoke timings)
TIFA: ~800 prompts of question generation (LLaMA-2-7B) + ~115k mPLUG-large VQA calls; DSG: 800 x 3 LLM calls (Qwen2.5-32B) + ~115k VQA calls;
PN-VQA: 800 x (1 + n_elements) LLM calls + ~144k Qwen2-VL-7B forward passes (positive + negative); B2 / B5: 14,400 forward passes each.  Roughly
2-5 GPU-h per method on L40S / H100; downloads ~120 GB (within the 1.2 TB guard: 1.9 TB free on 2026-10-14).

## Implementation record (2026-10-14; deviations from each method's native setup, written before any score exists)
- **Code trees:** github.com/Yushi-Hu/tifa @ 8e37050f and github.com/j-min/DSG @ 7330119a, read-only under `third_party/`, imported by path; no file
  edited.  The human-rating files those repos ship (TIFA `human_annotations/`, DSG `dsg/data/tifa160-likert-anns.csv`) were moved unopened to
  `third_party/closed/` (sha256 in `SHA256SUMS_v7x_20261014.txt`); DSG's in-context annotations (`tifa160-dev-anns.csv`, `dsg-1k-anns.csv`) stay
  (method inputs, not ratings).
- **Weights** (data_fetch group v7x, published sha256 verified): TIFA generator `tifa-benchmark/llama2_tifa_question_generation` @ 353b3dd, UnifiedQA
  v2 T5-large @ 1d3b8e1, SBERT all-mpnet-base-v2 @ e8c3b32 (served offline through `hf_cache_v7x` for TIFA's hard-coded id), mPLUG-large
  (ModelScope `damo/mplug_visual-question-answering_coco_large_en`, tag v1.0.0; weights identical to master), Qwen2.5-32B-Instruct, Qwen2-VL-7B-Instruct,
  HPSv3.
- **Environments:** `envs/v7_tifa` (torch 2.5.1 cu124, transformers 4.40.2, ModelScope 1.20.1, setuptools < 70 for ModelScope's pkg_resources,
  decord / librosa / soundfile / torchaudio 2.5.1 / zhconv / unicodedata2 for ModelScope's multi_modal imports) and `envs/v7x_llm` (torch 2.9.0
  cu128, transformers 4.57.6; t2v-metrics 3.0 without its pins).
- **Scoped imports** (`scripts/v7x/x_stubs.py`; every stub raises if called): promptcap and lavis.models (TIFA's PromptCap / BLIP-2 routes);
  fairseq: TIFA's requirements list it, but fairseq 0.12.2 does not install (its `omegaconf<2.1` pin resolves only to wheels whose metadata current
  pip rejects), so every fairseq.* import is served by a permissive stub.  It is reached only through ModelScope's OFA package, which its
  multi_modal module loads next to mPLUG's preprocessor; the mPLUG model code imports nothing from fairseq.
- **TIFA:** the registry entry 'mplug-large' points to the local copy of the same hub id; a prompt whose UnifiedQA filter keeps no question scores
  0 for every candidate (all tie -> first index), counted.
- **DSG:** generate_fn = Qwen2.5-32B-Instruct, the prompt as one user message (as DSG's openai_completion), greedy, 500 new tokens; parse errors are
  recorded; a dependency on a question id outside the parsed question set imposes no constraint (DSG's own code would raise there), counted.
- **PN-VQA:** Listings 1 and 2 reconstructed from the arXiv HTML (whitespace normalised; only the two examples each listing shows); answers parsed
  from "A: yes|no"; an element without a parsed question is dropped, counted.
- **B2:** t2v-metrics' Qwen2VLModel with attention 'sdpa' (flash-attn not installed) and B1's templates.
