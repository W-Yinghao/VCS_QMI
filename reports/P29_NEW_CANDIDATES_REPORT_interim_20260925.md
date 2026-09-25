# P29 (interim) — wave H: new critic forms, projector forms and 4 views on the neg-detach cosine base (4 of 5 units final)

Pre-registration: `P28_NEW_CANDIDATES_PREREG_FROZEN_20260925.md`.  Table: `P29_new_candidates_results_table.md` (summarize job 1009123).
`proj_bnonly` (rerun after the first attempt failed the parameter-free-projector gradient check, see `job_ids.json`) is at epoch ≈ 85/200 and is
reported when final.  Base: cosine critic, K = 8, negative detach, a0 = 1: 80.48 (seed 0) / 80.59 ± 0.12 (3 seeds), kNN 74.41, h-rank 30.4.
Rule: HELPS > +1.0 / HURTS < −1.0 vs 80.59.  Seed 0, 200 epochs.

| unit | linear | Δ | kNN | h-rank | z-rank | learned score at ep 200 | verdict |
|---|---|---|---|---|---|---|---|
| **views4** (4 views, J over 6 pairs) | **84.48** | **+3.89** | **81.10** | 57.4 | 21.3 | a = 10.65, b = −8.62 (thr 0.81) | HELPS — largest single factor of the project; 2× encoder compute per epoch |
| diag_metric | 80.56 | −0.03 | 74.02 | 27.5 | 7.0 | a = 9.54, b = −7.85; 89 % of w < 0.1, top-10 dims carry 97 % | neutral |
| mono_spline | 80.46 | −0.13 | 73.90 | 25.0 | 10.3 | flat at −8.4 for s < 0.5, then −4.7 / −3.0 / −0.7 / +1.6 at s = 0.7 / 0.8 / 0.9 / 1.0 | neutral |
| proj_linear (1 Linear, no BN/ReLU) | 80.02 | −0.57 | 75.74 | 30.5 | 14.7 | a = 9.78, b = −8.02 | neutral (kNN +1.3) |
| proj_bnonly (no projector; critic reads L2(BN(h))) | running | | 69.6 @ ep 50 (+1.4) | | | | pending |

## Reading
1. **Four views: +3.9 linear, +6.7 kNN, h-rank 57 (from 30).**  kNN leads the base at every epoch (61.2 vs 53.9 at 10; 78.6 vs 71.8 at 100;
   81.1 vs 74.6 at 200) and the learned (a, b) end where the base ends (thr 0.81), so this is not a critic effect.  What changes is the
   number of positive pairs per image per step (6 instead of 1) and the number of negative pairs (12 288 vs 2 048 per step).  Caveat, as
   pre-registered: a 4-view step costs two encoder forwards per image; train time 12 238 s vs ≈ 6 200 s.  The equal-compute unit
   (`P33_vcs_views4_100ep`, 4 views for 100 epochs) is at epoch 58 with kNN +6.1 over the 2-view base *at the same epoch*; its final value
   against 80.59 decides whether the gain survives at equal compute.  Seeds and the a0 = 5 combination are already queued (P35).
2. **The critic's *form* is exhausted.**  A free monotone score (spline) re-learns a hard threshold near cos 0.9 — sharper than the affine
   cosine, not smoother — and is neutral; a diagonal metric selects ≈ 10 of 128 dimensions and is neutral; earlier the shared metric collapsed
   the code (P27).  Together with P19/P25 (MLP, interaction, bilinear, cosine): **among similarity-type critics the exact functional form
   does not matter; what matters is what gradient reaches the encoder and how many pairs carry it.**
3. **The projector's non-linearity is not the lever** (proj_linear −0.6 linear, +1.3 kNN, rank unchanged): the "projector absorbs the
   spreading" reading of the geometry diagnostic (synthesis §4b) does not translate into a gain by removing the hidden layer.
4. **Operating point under negative detach** (epoch-200 means; new observation from the step logs): with detach the *positive* pairs are
   never saturated (sat_pos 0.0 %, mean T⁺ = 0.82; a0 = 5: 0.2 %, 0.84) while 67 % of negatives are; without detach 83 % of positives are
   saturated (plain cosine, thr 0.47; MLP K = 8 the same).  Detach moves the learned threshold from 0.47 to 0.82, which keeps every positive
   pair in the gradient-active region for the whole run.  Four views raise positive saturation only to 15 % (T⁺ 0.88) while J rises to 0.965.
   This is a candidate mechanism for the +2.4 of detach (synthesis §3.2 said "mechanism unproven"): **detach converts the objective from
   "push negatives below a low threshold" (positives saturated, weak signal, §4.4) into "pull positives above a high threshold"**.

## Consequences
New base candidate: cosine, K = 8, detach, a0 = 5, **4 views** (`P35_vcs_a5_views4`, seeds 0/1/2 queued; equal-compute reading from P33).
Dropped: spline and diagonal-metric scores, linear-only projector.  Open: proj_bnonly (running), a0 sweep (P31), 800-epoch runs.

## Not claimed
Single seed for every unit; selection split; the 4-view gain is at 2× compute until P33 reports; the saturation reading is observational
(no intervention on the threshold yet); no control comparison at equal tuning budget.
