# V7C-04 — GenAI-Rank scoring request (written before any GenAI-Rank human label is read)

## 1. What is requested
A read of the sealed GenAI-Rank human ratings to score the frozen V7 families, through the frozen scorer only.

**Label file:** `/projects/EEG-foundation-model/yinghao/data/genai_rank/closed/genai_rank_labels_sealed.jsonl`.
- Mode 0400; sha256 `0eb955b45429f5c240c8aed5ad9686db905695c2a74c40c67682c87b506a20dc`.
- This equals the hash recorded at the owner-approved sealing (`outputs/V7/data/genai_rank_seal_report.json`); the file is unchanged.
- It has never been opened since the sealing.

## 2. Approval
- **Owner, 2026-10-14**, replying to the scoring request (the freeze record, its hashes and the frozen commands): "上面那个我也approve".
- **V7 supplement v3 closeout package** (`docs/04 §3`): the read happens only after the prediction freeze and this written request. The request and its ledger row are committed before the scorer runs ("不能先评分后补签").

## 3. Frozen predictions (V7X-50 v3; `reports/V7/V7X_50_RANK_PREDICTION_FREEZE_v3.{json,md}`)
- **Manifest:** 36 methods, each file validated against the label-free universe `genai_rank_inputs.jsonl` (14,400 candidates, sha256 `514b9368…`).
- **methods.json:** sha256 `548c5293b347e367ccc8f4f5f0ac4ca344df0cfa817487db064319e6728a1b00`.
- **Primary-family files (first 16 hex of each sha256):**
  - M2-E: `06dde612c267b951`, `3c1e0425efe13589`, `fb1efa424f9cd669`
  - B0: `582911e16e13923c`
  - B1: `ea0c82e3f957473e`
  - B4T: `225483371ea33a2d`
  - B4D: `cf5105484d28b1d6`
- **Attribution file:** M1-E `1d9cce457c9c5fbc`, `7a97a5d181205fc5`, `598192645ed56d4a`.
- **Rules fixed before the read:**
  - M-arm E falls back to O where the parser failed (16 pools; the frozen `v7_parser` / `v7_train` rule).
  - External systems keep their own outputs: TIFA's 66 no-question pools tie, and the official first-index rule applies.
  - X21 rows are scored on their parsed prompts, with separately named neutral-fill sensitivities.
  - The V7-03 secondary readouts H_α and calibrated E are fixed on CAL_SELECT / CAL_PROB (v3).
- **Freeze history:** v1 is superseded (it tie-filled the M arms, contrary to the frozen rule); v2 → v3 is additive.

## 4. Reproduction and readiness review (V7C-03)
See `V7C_03_EXTERNAL_READINESS.md` (TIFA fidelity: local 5.92 kept questions per prompt vs 7.02 released, element precision 0.91 / recall 0.78).  In brief:
- **Completeness:** B0 / B1 / B4T / B4D are complete for all 14,400 candidates.
- **TIFA failures:** TIFA's 31 parse failures lie inside its 33 no-question prompts; those score 0 for every candidate.
- **DSG structure:** the local DSG has valid dependency graphs. On the 50 DSG-1k fidelity prompts, local tuple content matches the released PaLM 2 annotations at 86 % precision / 81 % recall; its category labels differ (relabelled taxonomy).
- **TIFA fidelity:** see the readiness report.
- **Scorer test:** the scoring wrapper ran end to end on synthetic random labels; the primary contrasts were null after Holm.

## 5. Exact commands (CPU job; the frozen scorer computes every statistic)
```
python -I scripts/v7x/v7c_score_rank.py --freeze /projects/EEG-foundation-model/yinghao/outputs_v7/x50_rank_freeze_v3 \
  --manifest reports/V7/V7X_50_RANK_PREDICTION_FREEZE_v3.json \
  --labels /projects/EEG-foundation-model/yinghao/data/genai_rank/closed/genai_rank_labels_sealed.jsonl \
  --out /home/infres/yinwang/CS_QMI/outputs/V7_CLOSEOUT/scoring_v1 \
  --request reports/V7_CLOSEOUT/V7C_04_SCORING_REQUEST.md --ledger reports/V7_CLOSEOUT/LABEL_ACCESS_LEDGER.csv
```
**Wrapper.** `scripts/v7x/v7c_score_rank.py`, sha256 `ab4b7c35…`. It calls `scripts/v7/v7_score.py rank --allow-scoring` (sha256 `e86af47e…`) once per family:
- **primary:** M2-E vs B0 / B1 / B4T / B4D × DALLE_3 / SDXL_Base, N = 9; 8 contrasts, prompt-cluster bootstrap 5,000 (seed 2718), Holm.
- **attribution:** M2-E vs M1-E (2).
- **v1-registered:** M2-E vs M1-E / B0 / B1 (6).
- **X21:** on the parsed prompts.
- **common-defined subset:** without the 41 prompts with a flagged failure in the primary or attribution methods.

The wrapper refuses to run unless the label hash equals the seal, every frozen file hash matches and the ledger row exists. It deletes its filtered label copies afterwards.

## 6. Exposure record
- **V8.** The V8 package (sha256 `54477a38…`) was unpacked before this read. V8-03 was FROZEN at 2026-10-14T23:21:07Z (`v8/utility-aligned-cs` `0232ff1`), also before this read. V8's GenAI-Rank comparison is therefore a pre-specified follow-up on the same benchmark (package docs/06 §1 case 1).
- **Not read by this request:** GenAI-Bench, TIFA160, RichHF and EvalMuse test labels.
- **After the read:** results-only commit to `reports/V7_CLOSEOUT/V7C_04_results/`, then the report. No prediction, readout, α, parser, calibrator or denominator changes after the read.
