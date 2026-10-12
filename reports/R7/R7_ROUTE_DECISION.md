# R7 route decision — draft for coordinator review (2026-10-12)

**Status.** This is a draft, not frozen. It follows the package template `templates/ROUTE_DECISION_CN.md`, and the r7 rule of one main route plus at most one light check per round.
- **Evidence:** `reports/R7/R7_BASELINE_REPRODUCTION.md` (R7-20, committed 23e840b).
- **Proposed protocol:** `reports/R7/R7_D1_PROTOCOL_DRAFT_20261012.yaml`.

## Decision

**Advance D1 (question-conditioned visual evidence compression) as the main route.** D2 and D3 are held as alternatives. The one light check is D1-L1, a CPU-only finite check of the nested-mask identity, or none if the coordinator prefers.

Already-approved r5 / r6 jobs are not affected. The route uses only GQA *train* for FIT / CAL / DEV. TextVQA val and ChartQA val are evaluation sets that the protocol pre-registers before they are read. No official test split is opened.

## 1. Candidate direction and closest public work

**The question.** For a fixed VLM host and the current question, how much does the *actual task risk* increase when visual tokens are dropped? And does a selector that predicts this increase choose better token sets and budgets than attention or similarity heuristics?

**Closest work** (all must be cited; none may be claimed as new):

| Work | What it already does |
|---|---|
| FastV [T01] | attention-based pruning |
| SparseVLM [T02] | text raters plus recycling; reproduced in R7-20 |
| FEATHER [T03] | shows text-rich failure modes |
| PyramidDrop [T04] | staged token reduction |
| IF-Prune [T05] | variational information-bottleneck signal from a small VLM |
| DiffPrune [T06] | selector trained on task loss |
| ET-Prune [T07] | evidence-aware dynamic budget |

**Smallest statement of what would be new** (not a divergence or data swap):
- A *risk object* for compression on verifiable binary questions, h = (C − a)² − (C − b)² = d + 2(C − b)(b − a), measured with an independent label.
- Using it to test whether disagreement-based importance d is a valid proxy for the actual risk change h.
- Using it to train budget selectors that are compared at equal information against a logistic-posterior selector and a direct risk-regression selector of the same architecture.

If D1-20 / D1-30 show no advantage over the direct risk regressor, the contribution narrows to an analysis of the posterior-increment proxy (package §F).

## 2. Statistical object and complete task

- **Binary task.** GQA yes/no questions, C ∈ {−1, +1}, balanced by class weights.
  - B is the full observation (question + 576 LLaVA visual tokens).
  - A_k is the observation after a nested mask at k ∈ {72, 144, 288, 432} tokens.
- **Finite teacher.** The host's own first-token posterior gives b and a. D1-10 records d, h and the residual h − d per question, mask and budget.
- **Complete task.** Full free-form answers on GQA DEV, TextVQA val and ChartQA val, with the official evaluators, at a fixed budget grid. The primary endpoint is quality at fixed measured latency.

## 3. Theory, extra assumptions, open questions — kept separate

- **Theory, inherited.** Under a balanced binary law, S_B − S_A = E_M(η_B − η_A)² equals the gap in optimal squared risk, by the L² projection (theory doc §3; F01–F03; U01). The finite identity h − d = 2(C − b)(b − a) is algebra.
- **Extra assumptions.**
  - The host's binary posterior is a reasonable finite teacher.
  - Masks at the LLM input preserve the host's interface, with positions re-indexed as SparseVLM v1 / FastV do.
  - Binary GQA behaviour carries over to free-form answers. This is only checked at D1-30.
- **To be tested.**
  - Whether d tracks h (the D1-10 readout).
  - Whether VCS or logistic selectors beat direct risk regression at equal information (D1-20 / 30).
  - Whether any selector beats SparseVLM / uniform at equal measured latency (D1-30).

## 4. Reproduction, our matched control, and system reference

**Original reproduction (R7-20, fixtures, official code).**
- **LLaVA-1.5-7B.** TextVQA 58.7 against the paper's 58.2; POPE-adversarial F1 81.9 against 85.9 averaged over three splits.
- **SparseVLM.** At 192 tokens the TextVQA change is −0.4, against −0.5 in v4. At 64 tokens TextVQA is 50.9, against 51.8 in v1.
- **Qwen2.5-VL-7B.** TextVQA 83.0 against 84.9.

**Matched control.** The same selector architecture under three objectives: VCS J, balanced logistic, and direct risk regression. All three see the same FIT tuples and masks, with the same steps and selection rule.

**System references.** These are reported separately and are not causal comparisons:
- SparseVLM (official code), uniform-grid and random.
- FastV is not run: no licence file, and its code pins transformers 4.31.
- IF-Prune needs its own pinned env and its own hosts. It is required before any paper claim (D1-40).

## 5. Data independence, licences, final evaluation

**FIT / CAL / DEV split.** All three come from GQA train-balanced (CC BY 4.0). The split is by image hash, image-disjoint, 10,000 / 2,000 / 2,000 images, with at most 4 binary questions per image.

**Host exposure, disclosed.**
- LLaVA-1.5 lists GQA 72K in its instruction mix (arXiv 2310.03744, Table 7), and footnotes that GQA training images were observed in training. GQA train-balanced has 72,140 images and 943,000 questions. The mix also contains Visual Genome, which is where all GQA images come from.
- So every FIT / CAL / DEV image was seen by LLaVA-1.5.
- **Recommended:** exclude the exact QA pairs from `llava_v1_5_mix665k.json` (a 1 GB annotation download, requested). A text-only pass (0 % of tokens) measures question-level memorisation.
- Qwen2.5-VL's training data is undisclosed, so exposure is assumed.

**Evaluation (D1-30).**
- GQA DEV.
- TextVQA val, 5,000 questions. 200 items were read in R7-20 for host reproduction only; this is disclosed.
- ChartQA val, human and augmented reported separately.

**Final evaluation (D1-40, separate pre-registration).** GQA testdev, TextVQA test and ChartQA test stay closed until then.

## 6. Measured resources and complete pre-registered units

**Measured in R7-20** (H100 NVL, batch 1, short answers):
- Full LLaVA 0.099 s/item; SparseVLM 0.103–0.108 s/item.
- Weight load after node-local staging: about 6 min.
- Peak memory 14.7 GiB for LLaVA, 18.4 GiB for SparseVLM.

**D1-10 smoke** (job 1034889, H100 NVL). All six probe sentinels pass. The teacher runs at 0.34 s per question over 14 prefill passes. Uniform-grid 25 % saves 9.4 / 27.2 / 41.4 / 43.6 % of per-item latency at batch 1 / 4 / 16 / 32, so the pre-registered rule gives B\* = 4 (`reports/R7/R7_D1_10_SMOKE.md`).

**Pre-registered units.**

| Stage | Units |
|---|---|
| D1-10 | 56 teacher shards in 14 bundles; the provisional manifest has 43,479 questions × 14 prefill passes, about 6–10 GPU-h from the smoke throughput |
| D1-20 | 9 training units (3 selectors × 3 seeds) + 6 ablation units (A1, A2 × 3 seeds) |
| D1-30 | the fixed budget grid × methods × 3 benchmarks, plus latency runs on the pinned reference GPU |

GPU-hours per stage are in the protocol.

## 7. Expected advantage (qualitative hypothesis) and failure interpretation

**Hypothesis.** A task-conditioned risk estimate allocates kept tokens and budgets better than attention or similarity, at equal measured latency, in a regime where the LLM-side cost matters.

**Failure readings, fixed now:**
- **h − d large and unpredictable (D1-10):** use direct risk as the practical reference. Do not claim that a squared gap estimates information.
- **Direct risk regression ≥ VCS (D1-30):** the contribution moves to an analysis of the limits of the posterior increment (package §F).
- **No quality-preserving room on the budget grid:** check the host's token interface and positions, and the baselines, before touching critics.
- **No LLM-side latency saving in any tested regime:** report quality at fixed kept-token budget only. Do not claim acceleration.

## 8. Why D1 now, and not D2 or D3 now

**Why D1** (all R7-20 evidence):
1. The full chain runs offline today: host, an official pruning baseline, official evaluators, the timing split and the finite h / d check. TextVQA and POPE reproduce on the fixtures.
2. The R7-20 finite check already shows the scientific question is live. On binary items the residual h − d is significantly positive on GQA at 64 tokens (+0.027 [+0.010, +0.044]). On POPE at 64 tokens it is significantly negative (−0.026 [−0.046, −0.006]). Fidelity to the full model is therefore not a surrogate for risk, which is exactly what D1-10 / D1-20 test at scale.
3. Data is in hand or arriving: GQA train images (`images.zip` complete; questions downloading), TextVQA, ChartQA. Nothing is licence-blocked.
4. The cost risk is identified and handled. At batch 1, pruning saves no latency (R7-20). The protocol therefore fixes the latency regime from a timing-only measurement before any quality result, and keeps a pre-declared fallback.

**Why not D2 now:**
- POPE behaves well: there is no text shortcut, with text-only accuracy 49.3.
- But the M-HalDetect training labels carry a strong text-only shortcut: AUROC 0.859 for INACCURATE vs ACCURATE, and 0.966 for ANALYSIS. Its official train and val splits share 27 images, and 2 of 50 POPE-fixture images overlap with it.
- A credible D2 first needs a text-only branch baseline, image-regrouped splits, COCO val2014 (only now extracting) and a verifier training stage. That is a second full system, which the round rule forbids alongside D1.
- D2 stays the first alternative.

**Why not D3 now:**
- The finite check (R7-20) is complete. It shows that n ≥ 32, better 128, per condition is needed, and that unconditional statistics miss marginal-preserving mismatch.
- The natural-image D3 has no reference distribution yet. The package says to settle the reference first and not to launch generation, and this round has no generator / encoder setup.
- D3 stays an alternative. Its finite results are reused as theory support.

## 9. The one light check alongside

**Recommended: D1-L1, a CPU-only finite nested-mask check, under 1 CPU-hour, no GPU.**
- **Setup.** Exact finite laws with nested observations (masks as coarsenings), plus finite teachers with controlled miscalibration.
- **Purpose.** Tabulate the expected size and sign of the residual h − d as a function of teacher calibration error and budget.
- **What it buys.** D1-10's CAL residuals can then be read against a known reference rather than intuition.
- **Cost.** It uses the package reference functions and adds nothing to the GPU queue.

**Alternative: none.** In that case D1-10's interpretation grid stands alone.

The D2 / D3 GPU work explicitly does **not** run this round.
