# R7-50 / R7-60 outline (proposal draft, 2026-10-12)

**Status.** Owner decision 2026-10-12: proceed with A after the three corrections below (now applied); the actual test read still needs the
owner's final go at execution time.  Nothing in this outline has run. The official test splits stay closed. Any read of them needs the owner's one-shot
approval of the exact list in §A.1. This outline was drafted while D1-20 and D1-30 run, before any D1-30 result was read.

**Package tasks.**
- R7-50 is "external validation and method attribution". Its deliverables are `external_results.csv` and `claim_ledger.csv`.
- R7-60 is a manuscript-update proposal; V6 is never overwritten. Its deliverables are `summary_to_collaborator.md` and
  `manuscript_delta_proposal.md`.

## R7-50 — external validation and method attribution

### A. Official-test one-shot (requires the owner's approval)

**A.1 What would be read, once.** Only splits whose labels are available locally are proposed. All of them are already on disk under
`closed/`, with sha256 recorded in `reports/R7/R7_DATA_CLOSED_SHA256.csv`.

| split | file (closed) | size | labels | evaluator |
|---|---|---|---|---|
| GQA testdev-balanced | `gqa/closed/testdev_balanced_questions.json` (sha256 14039069…) | 12,578 q | in the file | exact match after LLaVA `convert_gqa_for_eval` (SparseVLM a9e7142 copy) |
| ChartQA test | `chartqa/closed/data/test-00000-of-00001.parquet` (sha256 84889bae…) | 2,500 q (human / augmented, counted at read) | in the file | pinned pix2struct 6fe25c1 relaxed accuracy |
| TextVQA test | `textvqa/closed/TextVQA_0.5.1_test.json` | 5,734 q | **not public** (EvalAI server) | — |

TextVQA test is **not proposed**. Scoring it needs a third-party upload, which the r7 rules forbid. TextVQA val stays the final TextVQA
number, and this is disclosed.

**A.2 Rows, fixed before the read.** Nothing is selected on test.
- **LLaVA-1.5-7B:**
  - full and the uniform grid at 12.5 / 25 / 50 / 75 %;
  - SparseVLM official anchors 192 / 128 / 96 / 64 and adapted rows 72 / 144 / 288 / 432;
  - S1, S2 and S3 × seeds 0–2 at the four budgets (S2 and S3 are the D1-30 primary controls);
  - S1 dynamic × seeds 0–2 at the three CAL operating points.

  That is about 60 configurations.
- **Qwen2.5-VL-7B**, only if R7-40 is complete: full, the uniform grid, and S1, S2 and S3 × seeds 0–2 (owner decision 2026-10-12: R7-40
  includes S2).
- All pre-registered primary rows are reported, whatever D1-30 shows; D1-30 results do not decide which test rows are kept.

**A.3 Endpoints.**
- **Latency** is not re-measured on test.  It is reported as **latency measured on the pre-fixed profiling workload** (the frozen D1-30 / R7-40
  timing subsets, B* = 4), not as each test set's actual mean end-to-end latency: latency depends on input shapes, generation lengths and batch
  composition, and Qwen uses dynamic resolution.  Dynamic selector rows additionally report the kept-token counts and budget distribution
  observed on the test set (correction 2026-10-12, owner-adopted review).
- **Contrasts:** the D1-30 primary contrasts at the D1-30 latency points:
  - S1 − S2;
  - S1 − S3;
  - S1 − SparseVLM, budget-matched (D30-1).
- **Statistics:** the same as D1-30 — image / chart-grouped paired bootstrap with 2,000 replicates, seeds, and Holm per point.
- **Secondary:** fixed-budget rows and retention.

**A.3b Exposure statement (graded; correction 2026-10-12).**
- Relative to this project: the test splits were never used for selector training, hyperparameters or any selection.
- Relative to the hosts' public training data: known overlaps are checked and reported (LLaVA-1.5's mix includes GQA, Visual Genome, COCO and
  TextCaps; all GQA images, test-dev included, come from Visual Genome, so GQA test-dev is not "clean" with respect to upstream image exposure).
- Undisclosed upstream pre-training exposure is not guaranteed either way.

**A.4 Procedure.**
1. The owner approves this list in writing. Approval is one-shot.
2. Answers are generated with the frozen code into a sealed directory, using single-writer files (E1). Their sha256 are committed before any
   scoring.
3. One CPU job reads the test labels and scores every row.
4. Results are committed as they come out.
5. What protects the evaluation is the frozen analysis — test results take no part in choosing models, thresholds, methods or reported rows —
   not the number of physical label reads.  Three cases are kept separate and reported: (a) fixing an implementation error (the error, the fix
   and both outputs are published; never tuning); (b) re-verifying frozen predictions with the deterministic evaluator (allowed, recorded);
   (c) re-selecting methods or rows (not allowed).  A rerun after an infrastructure failure uses the same code and seeds and is reported.
6. Multiplicity is handled by the pre-declared Holm correction, not by the one-read procedure.

**A.5 Cost.** About 15,000 items × about 60 configurations is about 0.9 M answers on LLaVA: 6–10 GPU-h on H100 / L40S. The Qwen rows add about
15,000 × 28, about 0.42 M answers: 4–6 GPU-h.

### B. Method attribution (no test read)

**B.1 `claim_ledger.csv`.** Columns: claim | evidence (stage, file, row) | status (supported / not supported / not tested) | caveat.
Candidate claims:
1. The squared disagreement d is a usable proxy for the realised risk change h. Evidence: the D1-10 CAL readout, the D1-L1 ceiling, and the
   D1-30 DEV risk calibration.
2. A VCS-trained selector ranks masks by realised risk. Evidence: D1-20 CAL Spearman and the D1-30 DEV calibration.
3. Quality at fixed latency: S1 against S2 and S3 at B* = 4 (D1-30 primary).
4. S1 against SparseVLM at matched budget. No latency claim against SparseVLM at B* = 4.
5. LLM-side acceleration at B* = 4 against uniform grid and random (D1-30 secondary).
6. Ablations:
   - question conditioning (A1);
   - the label-based h − d correction (A2);
   - dynamic against fixed budgets (A3).
7. Cross-host consistency on Qwen2.5-VL (R7-40; descriptive).
8. Test confirmation (§A), if approved.

**B.2 `external_results.csv`.**
- **IF-Prune (T05).** Official code: `snap-research/EVLM-IF-Prune` (README says Apache 2.0). It supports Qwen2-VL and InternVL2.5 only, and
  needs its own pinned environment (CUDA 12.4, torch 2.6, flash-attn 2.7.4).
  - Proposal: run its official inference on Qwen2-VL-7B-Instruct on our val sets at matched kept fractions. Report it in its own host column,
    next to full and uniform grid on the same host (relative retention).
  - There is no head-to-head with S1 across hosts. A same-host S1 on Qwen2-VL would be another R7-40-like stage (decision).
- **DiffPrune (T06) and ET-Prune (T07).** No released code, so they cannot be reproduced. They appear as literature rows only (disclosed).
- **FastV.** No licence file, and it pins transformers 4.31.
- **PyramidDrop and FEATHER.** MIT licence, LLaVA support. Optional: official-code rows on LLaVA-1.5 in their own environments, labelled.

**B.3 Also delivered:** `failure_log.csv` and the integrity counts per prediction file (E1).

## R7-60 — manuscript delta proposal (report only; V6 untouched)

- **`summary_to_collaborator.md`** — in plain terms:
  - what D1 adds and what it does not;
  - the tables and figures to cite;
  - the limits: LLM-side only; one primary host plus one confirmation host; val-based except for any approved test read.
- **`manuscript_delta_proposal.md`** — proposed paragraph-level deltas, not edits:
  - where the VCS risk object enters;
  - how h against d is presented, with the D1-L1 calibrated-teacher ceiling;
  - the evaluation regime (B* = 4, measured latency);
  - positioning against IF-Prune, DiffPrune and ET-Prune;
  - what is explicitly not claimed: no new "information-theoretic pruning paradigm"; no vision-side saving.
- **Inputs:** frozen protocols, results files and the claim ledger. No new experiments.

## Draft text for the owner's one-shot approval (to be sent only by the coordinator)

> R7 D1 requests one read of two official test splits — GQA testdev-balanced (12,578 q) and ChartQA test (2,500 q) — scored once with the
> frozen code and the configurations listed in R7-50 §A.2. Predictions are hashed and committed before scoring, and there is no tuning,
> selection or rerun after the read. TextVQA test is not read: it would need an upload to the evaluation server. Approve / decline.
