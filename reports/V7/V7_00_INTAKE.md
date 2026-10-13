# V7-00 intake (2026-10-13 / 14)

V7 = fine-grained text–image alignment on EvalMuse + fixed-pool best-of-N selection on GenAI-Rank (package `VCS_CVPR_V7_T2I_Alignment_Server_20261013`,
handoff v1).  Work lives on branch `v7/t2i-alignment` (git worktree `ssl_pilot_v7`, as the package recommends), reports in `reports/V7/`,
small outputs in `outputs/V7/`, checkpoints and staged artefacts in `/projects/EEG-foundation-model/yinghao/outputs_v7/`.

## HEAD and older work

- Package snapshot `3e2fe59` (2026-10-13 18:51 UTC).  Since then on `main`: E1 results + report (`7d98a17`, `db3bb6c`), REV8 diagnostics frozen
  (`a977b21`), the V7 data items in `scripts/data_fetch.py` (`aeb56b5`), D1-FULL results + report (`7a89736`, `4066b51`).  V7 branch starts at `db3bb6c`.
- Older authorised runs continue unchanged: R7-50 phase F (official GQA testdev / ChartQA test generation under its frozen protocol), E2 (last
  teacher group), the REV8 critic diagnostics (3 GPU jobs).  Nothing was cancelled or resubmitted for V7.

## Data and weights (all under `/projects/EEG-foundation-model/yinghao/`, registered in `data/DATA_MANIFEST.json`)

| item | source @ revision | licence | size | role |
|---|---|---|---|---|
| EvalMuse TRAIN | HF DY-Evalab/EvalMuse @ fd2087b | BSD-3-Clause | 54.6 GB (6-part zip, 43,534 members) + `train_list.json` (32,717 images) | FIT / CAL / DEV derived from TRAIN only |
| EvalMuse `test.json` | same | BSD-3-Clause | 7 MB | **closed** (hashed only); the card says test annotations "will be available later" → no official EvalMuse test in V7 |
| GenAI-Rank candidates | HF zhiqiulin/GenAI-Image-Ranking-800 @ 1c7afac | MIT | DALLE_3 7,200 files; SDXL_Base 7,727 files (527 prompts carry a 10th image, not a candidate) | primary application inputs |
| GenAI-Rank human ratings | same, `human_rating.json` | MIT | 1.8 MB | **closed** (hashed only) |
| GenAI-Bench | HF BaiqiL/GenAI-Bench @ 94e543b (already local) | Apache-2.0 | 1,600 prompts | external terminal; columns read so far: Index, Prompt (HumanRatings never read) |
| released FGA-BLIP2 | HF hanshuhao/FGA-BLIP2 @ a14ad16 | not stated (code Apache-2.0) | 6.2 GB | baseline B0 inference only (may have seen all public TRAIN) |
| BLIP-2 init | LAVIS storage: `eva_vit_g.pth`, `blip2_finetune_coco.pth` (published MD5 verified) | BSD-3-Clause | 2.0 + 4.7 GB | the common training init |
| bert-base-uncased | HF google-bert @ 86b5e09 | Apache-2.0 | 0.9 GB | Q-Former init + tokenizer |
| CLIP-FlanT5-XXL | HF zhiqiulin/clip-flant5-xxl @ 89bad6f (already local) | Apache-2.0 | 23 GB | baseline B1 (VQAScore) |
| Qwen2.5-VL-7B-Instruct | @ cc594898 (already local) | Apache-2.0 | 16 GB | prompt-only element parser (text only); B2 / PN-VQA host |

Not yet fetched: TIFA160, RichHF-18K, NegBench image task, Winoground, HPSv3, VisionReward, `google/flan-t5-xxl` tokenizer (B1); DynEval /
DEPICT pending a code / weight check.

## Code versions

- EvalMuse / FGA-BLIP2 @ `45c8bb4af5d8c0d1d9f3733156177cc67ad9b7e5`, read-only clone (`third_party/EvalMuse@45c8bb4`, Apache-2.0); executed only through
  `scripts/v7/v7_native_launch.py` (tree on `sys.path`, nothing written into it).
- t2v_metrics: `genai_image_ranking.py` blob `0e0fdb3` is unchanged from `c1543c7` (2024-06-13) to HEAD `6ecb74f`; the paper-reproduction
  package is `t2v-metrics==3.0` (wheel sha256 448398dc…, pins torch 2.5.1 / transformers 4.49.0).

## Environment

`envs/v7_fga`: python 3.10, torch 2.9.0+cu128 (L40S / H100 / RTX PRO 6000), torchvision 0.24.0, transformers 4.33.2, timm 0.4.12, fairscale
0.4.4, numpy 1.26.4 — the EvalMuse pins except torch (repo: ">=1.10").  Packages the trimmed LAVIS never imports are left out; the full import
chain (lavis, FGA_Blip2, processors, AlignmentDataset, RunnerBase) is tested.  Tests: 34 V7 tests + the package's 56 reference tests pass in
this env (jobs 1038591, 1038602, 1038609).

## The six source facts (literature/CODE_PROTOCOL_AUDIT_CN.md)

| item | verdict | evidence (EvalMuse@45c8bb4 unless noted) |
|---|---|---|
| C1 native FGA is L1, not Logistic | fact_confirmed | `fga_blip2.py:204–207`: overall L1 + 0.1 × token L1 (masked) + 0.1 × mask L1; no BCE anywhere in the active `itm` branch |
| C2 variance weighting paper ≠ code | fact_confirmed | `var` is read (`fga_blip2.py:136`) but unused; `exp(var)` appears only in the commented stage-2 block after the return (`:237`).  Which path produced the released checkpoint cannot be read from code |
| C3 element labels are vote dictionaries | fact_confirmed, with one addition | `process_train.py:25–37` averages votes and flags; `element2mask.py:68` strips "(category)", `get_index` (`:35–41`) takes the FIRST exact token match (0 = not found), overlapping tokens average (`:90`), a record with any unmatched element is dropped whole (`:81`, `:94`).  Addition: training text goes through `blip_caption` (lowercased, `.!"()*#:;~` removed, 50 words max) while `element2mask.py` maps elements on the RAW prompt's tokens, so token labels can sit one position off after removed punctuation (counted in V7-01) |
| C4 truncation at 32 + index 0 = overall | fact_confirmed | `align_datasets.py:67–73`: cut to 32, zero-padded, `mask[0] = 1`, `token_score[0] = (score − 1) / 4`.  Inference (`element_score`, `eval.py`) does not truncate |
| C5 official N = 3 is three fixed groups | fact_confirmed | t2v_metrics `genai_image_ranking.py:259` (N = 9: `torch.argmax`, first maximum on ties), `:264–270` (N = 3: argmax inside [0:3], [3:6], [6:9]); candidate order = file index 01…09 (the dataset class reads 1…9 only, so the extra SDXL_Base 10th images are never candidates) |
| C6 tool versions are not interchangeable | fact_confirmed | t2v_metrics README "Reproducing Paper Results (Legacy Version)": `pip install t2v-metrics==3.0`; v3.1 (2026-06-04) is a model refresh |

Further native facts read from the code (they fix the N0 recipe): `fga_blip2.yaml` model_type coco, `load_finetuned` True → EVA ViT-g
(`eva_vit_g.pth`, positional embedding interpolated to 364) then `blip2_finetune_coco.pth` over every matching tensor, the mask head the only
random module; ViT frozen; training transform RandomResizedCrop(364, scale 0.5–1) + horizontal flip; AdamW (decay / no-decay groups, wd 0.05),
`linear_warmup_cosine_lr` = 100 warm-up steps from 1e-8 then one cosine value per epoch (epochs 1–4: 0.905 / 0.655 / 0.345 / 0.095 × lr); AMP
fp16 with GradScaler; no gradient clipping; batch 14; 5 epochs; seed 42 (V7 units use their own seeds).  The `alignment` task's evaluation is a
no-op, so no "best" checkpoint exists: every epoch is saved (full state incl. the frozen ViT and the optimizer, ~6 GB) and **N0's native
selection is the last epoch**.  Native evaluation scores an element that does not match as **0** and computes PLCC after a 4-parameter logistic
fitted on the evaluation data itself.

## GenAI-Rank prompts without labels

The Rank release carries its prompts only inside `human_rating.json`.  Its prompt ids (1–1611) are not GenAI-Bench's Index (0–1599).  A
label-free mapping is running: Rank candidates are matched to GenAI-Bench's per-prompt DALLE_3 / SDXL_Base images (sha256, else dHash ≤ 4), with
skill tags as an independent check (`scripts/v7/v7_genai.py`).  If it is incomplete, the only remaining source is a sealed input / label split
of `human_rating.json`, which needs an owner decision.

## Resources (measured values follow in V7_03)

Downloads 82 GB (free space 2.1 TB, floor 1.2 TB).  GPU timing (epoch, scorer per image / per 9-image pool, peak memory) comes from the V7
smokes; nothing is scheduled from estimates.

## For the owner

1. GenAI-Rank prompts: if the image mapping is incomplete, may a sealed split of `human_rating.json` write the prompt text to an input file
   (ratings moved unread into a sealed file, hashed)?
2. Branch: V7 stays on `v7/t2i-alignment` until you decide to merge.
