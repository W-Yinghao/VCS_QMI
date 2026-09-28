# CVPR evidence ledger — every claim-able number, its source and its status (2026-09-28, HEAD 74fbcdb)

Purpose: one place the paper can be written from.  Each row names the frozen pre-registration, the results-only commit / table and the report; the status column
says whether the cell is a 3-seed result, a single seed, still running, or only proposed.  Selection = the 5 000-image selection split of the official CIFAR-10
training set (used for every decision); Test = the official 10 000-image test set, opened once (P68).  Nothing in this ledger is a prediction.  Companion documents:
`V2_RECONCILIATION_20260928.md` (branch state), `SSL_TECHNICAL_NOTE_20260926.md` (method + recipe), `SECOND_APP_FINAL_SUMMARY_20260928.md` (second application),
`P41_CONTROL_TUNING_REPORT_20260928.md`, `P68_FINAL_OFFICIAL_TEST_REPORT_20260928.md`, `P82`/`P84`/`P72`/`P80` reports.

## A. First application — CIFAR-10 SSL (ResNet-18 CIFAR stem, frozen-h linear probe / kNN)

### A1. Final VCS recipe by compute budget (1× = 2 views × 200 epochs at B 256)
Recipe: cosine critic tanh(a⟨z₁,z₂⟩+b), a₀ = 5, K = 8 cyclic negatives with the shifted partner detached, 4 views (J averaged over the 6 pairs), AdamW 1e-3,
wd 1e-4, warm-up 10, projector 512→512(BN,ReLU)→128, pixel normalisation (0.5, 0.5, 0.5) / (0.5, 0.5, 0.5).

| compute | configuration | Selection linear / kNN | Test linear / kNN | seeds | prereg → table → report | status |
|---|---|---|---|---|---|---|
| 1× | 4v, B 128, 100 ep | 83.19 ± 0.40 / 78.43 ± 0.11 | 82.98 ± 0.45 / 78.57 ± 0.09 | 3 | P39 → `P40_recipe_ablation_results_table` → P40 (8d8c046); P68 | final |
| 2× | 4v, B 256, 200 ep | 84.54 ± 0.16 / 81.40 ± 0.22 | 84.52 ± 0.18 / 81.31 ± 0.29 | 3 | P35 → `P36_a5_base_results_table` (b7fbc2b) → P36 final report; P68 | final |
| 2× | same, a₀ = 1 | 84.41 ± 0.06 / 80.93 ± 0.16 | — | 3 | P28 → P29 (P29 report) | final (selection only) |
| 4× | 8v, 200 ep | 85.95 ± 0.35 / 83.47 ± 0.38 | 85.87 ± 0.07 / 83.69 ± 0.15 | 3 | P37 → `P38_views_curve_results_table` → P38 (69d74fa); P68 | final |
| 4× | 2v, 800 ep | 85.30 ± 0.21 / 82.33 ± 0.25 | 85.28 ± 0.40 / 82.40 | 3 | P35 → P36 table; P68 | final |
| 4× | 4v, 400 ep | 85.90 / 83.58 | 85.63 | 1 | P35 → P36 table (row COMPLETED 400/400) | single seed |
| 8× | **4v, 800 ep** | **87.01 ± 0.53** (86.42 / 87.16 / 87.44) / 85.46 ± 0.12 | **86.65 ± 0.26** / 85.30 ± 0.10 | 3 | P35 → P36 table; P68 | **final; the paper's VCS number** |
| 8× | 4v, 800 ep, strong aug (crop 0.08, jitter 0.8/0.8/0.8/0.2) | 87.78 / 85.58 | 88.26 / 85.79 | 1 | P43 → `P44_ceiling_results_table` (b3f1d55) | single seed; seeds 1–2 = S4 unit P89 (in preparation) |
| 8× | 8v, 400 ep | 86.76 / 84.94 | 86.86 | 1 | P43 → P44 table | single seed |
| 8× | 16v, 200 ep | 86.74 / 84.48 | 86.08 | 1 | P43 → P44 table | single seed |
| 8× | 4v, 800 ep, B 128 | 86.70 / 84.54 | 87.09 | 1 | P43 → P44 table | single seed |
| 16× | 4v, 1600 ep | 87.50 / 85.84 | 87.60 / 86.43 | 1 | P43 → P44 table | single seed |
| 16× | 8v, 800 ep | 87.14 / 86.32 | 87.42 / 86.58 | 1 | P43 → P44 table | single seed |
| 1× (2v) | original recipe: concat-MLP critic, K = 1, 2v, 200 ep | 74.34 ± 0.46 / 63.93 ± 0.62 | — | 3 | P5 → P7 | final (selection only) |

### A2. Equal-budget controls (P41, frozen 5813832; 3 seeds per cell; tuned with the same knobs VCS had: 4 views, B 128 / 100 ep, 800 ep, own τ / weights)
| compute | VCS (recipe) Sel / Test | SimCLR tuned Sel / Test | VICReg tuned Sel / Test | gap to SimCLR (Test) | gap to VICReg (Test) |
|---|---|---|---|---|---|
| 1× | 83.19 / 82.98 | 86.64 ± 0.33 / 86.94 ± 0.12 | 87.12 ± 0.10 / 87.00 ± 0.25 | +3.96 | +4.02 |
| 2× | 84.54 / 84.52 | 87.84 ± 0.42 / 87.99 ± 0.07 | 86.73 ± 0.30 / 87.11 ± 0.31 | +3.47 | +2.59 |
| 4× | 85.95 / 85.87 | 88.35 ± 0.06 / 88.21 ± 0.10 (2v/800ep) | 87.28 ± 0.62 / 86.91 ± 0.15 | +2.34 | +1.04 |
| 8× | 87.01 / 86.65 | 88.32 ± 0.30 / 88.13 ± 0.07 (4v/800ep) | 87.11 ± 0.51 / 86.87 ± 0.06 | +1.48 | +0.22 |
Untuned frozen P5 controls (2v, 200 ep): SimCLR 86.09 ± 0.38 (Test 86.29 ± 0.12), VICReg 85.46 ± 0.26 (Test 85.07 ± 0.27).  Reading (pre-committed, P41 §4, P68 §3):
under equal tuning both controls exceed the VCS recipe at every budget; the gap narrows with compute; VCS ≈ +1.2 / doubling, SimCLR flat above 2×.  kNN: VCS ahead of
VICReg at 8× (Test 85.30 vs 84.78).  Status: **final**.  Not run: 16× SimCLR control (proposed only).

### A3. Factor history (selection, 200 ep, 2 views unless stated; 3 seeds each) — `SSL_TECHNICAL_NOTE` §5.2
MLP K=1 74.34 → K=8 76.74 (+2.4) → cosine critic 78.21 (+1.5) → negative detach 80.59 (+2.4) → a₀=5 81.56 (+1.0) → 4 views 84.54 (+3.0 at 2× compute; +1.6 at 1×
via B 128) → 800 ep 87.01 (4v) / 85.30 (2v).  Ablations on the final recipe (single seed, P40): detach off −2.2 (4v/B256/100ep: 79.66 vs 81.88); K=1 −0.7; K=127 −0.3;
lr, wd, projector dims, a₀=1 (−1.1 at 1×, −0.1 at 2×), warm-up 5 (−1.2), B 64 (−0.4).  Neutral/worse (earlier bases): critic capacity, critic lr/wd, projector
depth, BN-only projector (−2.8), output BN (−2.1), critic on h (−5.6), EMA/stop-grad/predictor, extra critic steps, symmetric pairing, spline/diag/shared-metric/
bilinear critics, aug variants at 200 ep, schedule shapes.  Status: **final**; owner closed objective-side tuning (S3).

### A4. Mechanism (observational unless an intervention exists; `GEOMETRY_DIAG*.md`, `PROBE_LAYERS*.md`, P84)
Critic form fixes the geometry (MLP critics: h rank 13–21; similarity critics: 58+); learned critics converge to a threshold cos* ≈ 0.8–0.9; under detach the positives
stay unsaturated at 200 ep / 2 views (0 % |T| > 0.95) but **at 4v/800 ep 61 % of positives are above 0.95** (P84: median T 0.96) — saturation statements must name the
horizon; h-uniformity orders VCS runs and separates them from controls (−1.5…−2.3 vs SimCLR −2.8, VICReg −2.5); the last ResNet stage is productive under the recipe;
a₀ acts in the first ≈ 50 epochs.  P84: the training-time cosine critic lags its own 2-parameter class fitted to convergence (+0.025/+0.010/+0.007 J at ep 100/400/800);
matched-JS refits equal VCS refits; representation J does not order downstream accuracy (VCS converged-C0 0.993 vs SimCLR 0.973 at ep 800, accuracy the other way).
Status: **final** for the listed checkpoints.

### A5. Compute (technical note §7)
4v/800ep ≈ 48 508 s (A100, seed 0); 2v ≈ 31 s/epoch, 4v ≈ 65–68 s (A100) / 33 s (H100), 8v ≈ 120–140 s, 16v ≈ 257 s; peak memory 3.7 / 7.6 / 14.9 / 29.6 GB.  Search
cost: ≈ 155 VCS run directories over P3–P44 vs 30 control runs in P41 (disclose both).

## B. Second application (property pre-checks + task units; `SECOND_APP_FINAL_SUMMARY_20260928.md` §1–2) — **complete, no task selected**
| row | verdict | evidence | report |
|---|---|---|---|
| A calibration / threshold transfer | conditional; A-S1 **does not hold**; A-T conditional; external target conditional | native (1+T)/2 as calibrated as Platt only for the trained relation; calibration is a property of balanced proper scoring rules (JS shares σ(2f) = (1+T)/2), not VCS-specific | P50, P58 |
| B closed-form critic | B1 / B-S1 **hold** (gap ≤ 0.012; cross-modal +0.004 / −0.075); B2, B-S2, B-T2 registration **do not hold** (constructed pairs, RGB–NIR, IXI, fMRI EPI/T1w; combination reaches parity on IXI only); B-T1 conditional (J* saturates where CKA spans) | P48, P52, P59 |
| C batch decoupling | **does not hold** (frozen towers flat; from-scratch retention at B 32: VCS 0.979, SimCLR 0.977, VICReg 1.001; queue collapses for both) | P54, P61 |
| D dependence / leakage tests | conditional (revised): permutation-calibrated VCS test on par with HSIC (never ≥ 0.10 below), exact conditional test, fresh-N nulls 0.05–0.06 at R = 1000; conditional within-class test 0.98 vs HSIC 0.18 on one encoder (blur); Hoeffding bound 20–50× more sample-hungry; fMRI motion leakage **does not hold** (VCS = HSIC = QC-FC) | P46, P46-W2, P66 |
Methodological lessons for the methods section: summary §4 (fixed-nuisance null, circular-shift nulls, letter vs intent, sensitivity ≠ reliance, parity ≠ advantage, equal
budgets change the headline).

## C. Estimator-centred round (next-round plan v2 + package v1/v2)
| unit | result | seeds | status | source |
|---|---|---|---|---|
| Oracle resolution table (owner) | NWJ/VCS 1.6 → 37.7, JS/VCS 0.97 → 1.06 (single shuffled negative); in-batch InfoNCE/VCS 0.92 → 1.32 at N = 64, 0.91 → 0.41 at N = 1024 | analytic/MC | owner's file `CS_QMI/RESULTS_20260928.md`; server re-derivation = P85 first item | pending re-derivation |
| P72 R3 mismatch identification | **refuted**: JS AUROC highest in all 4 cells (e.g. animal m 0.4: VCS 0.856, InfoNCE 0.862, logistic 0.843, JS 0.887) | 3 | final | af7172c |
| P72 R2 (topic pairing) | not interpretable (training relation ≠ evaluation relation; design error disclosed) | 3 | final | af7172c |
| P80 R2 exact pairing / cross-fitted R3 | R2 **refuted** (VCS loses the most clean R@1: 4.6–13.1 pts vs ≤ 2 for InfoNCE / logistic); R3 saturated (raw CLIP AUROC ≥ 0.99, uninformative) | 3 | final | 539714e |
| P82 Gaussian probe (d = 20) | residual −25 % / −22 % posterior MSE at I = 4 / 8 (capacity not controlled); mix gain matched by JS; C0 / CQ budget-limited; learned-class gradients biased ±26–41 % | 1 | probe only | 9bfb8a2 |
| P84 frozen checkpoints | see A4 | — | final | d3b4d5f |
| P73/P74 T1 conditional test with fair controls | partial: VCS-perm power 0.13 / 0.25 / 0.61 / 0.96 at n = 200 / 500 / 1000 / 2000 (colour s 0.05, VCS encoder) vs JS 0.07 / 0.23 / 0.43 / 0.80, C2ST 0.06 / 0.18 / 0.46 / 0.82, HSIC 0.11 / 0.23 / 0.25 / 0.54; blur s 0.25: 0.44 vs 0.29 / 0.29 / 0.08 at n = 2000; SimCLR encoder n = 200 only so far | R = 100 | **running** (null / level cells in split jobs; verdict only after P74) | c9b69b4, 3e99437 |
| P75 S2 solo-learn (1000 ep, LARS, official test once per run) | — | 3 + 3 | **running** (VCS s0/s1 ≈ ep 400; SimCLR s0 ≈ 260, s1 ≈ 100; two seeds pending quota) | 5716216 |
| P85 E benchmark (six estimators + kernel controls + E4) | — | — | **in preparation** | — |
| P87 S-CS (VCS-N vs S-Kernel; CS-K-native) | — | — | in preparation | — |
| P89 S4 augmentation × method | — | — | in preparation (VCS strong-aug seed 0 exists: 87.78 / 88.26) | — |
| P91 CIFAR-100 confirmation | — | — | in preparation | — |
| P93 D1 / D2, P83 addendum 1 | — | — | in preparation | — |

## D. What changed since the owner's draft v3 (snapshot 236931d, 2026-09-26 22:15 UTC) — items the paper text must update
1. Abstract / §6.4 / §8 / Table 3 / Table 4: "one four-view 800-epoch run reaches 86.42 %" → three seeds **87.01 ± 0.53** selection (86.42 / 87.16 / 87.44), **86.65 ± 0.26** test.
2. Table 3 pending rows are all complete: 4v/1600ep 87.50, 8v/800ep 87.14, 8v/400ep 86.76, 16v/200ep 86.74, 4v/B128/800ep 86.70, strong-aug 87.78 (all single seed; test values in A1).
3. §6.3 / §6.6 / F.4: "controls have not received the recipe search; P41 is a draft" → P41 frozen (5813832) and complete; tuned controls exceed VCS at every budget (A2).  The
   draft's sentence "we do not infer a general SSL advantage" stays, now supported by an executed equal-budget comparison.
4. §6.1 / E.1: "official test set disabled / final unseen-test confirmation still necessary" → done once (P67/P68), test ≈ selection (mean Δ −0.03); numbers in A1–A2.
5. §6.6: "controlled Gaussian / mixture / nonlinear-channel studies not supplied" → P82 (Gaussian d = 20 probe, single seed) and P84 (frozen checkpoints) exist; the
   six-estimator staircase and the same-target kernel controls are P85 (in preparation); RuLSIF agreement (E4) is part of P85.
6. Table 6 "pixel normalisation": the technical note now states (0.5, 0.5, 0.5) / (0.5, 0.5, 0.5) as the YAMLs do (corrected 2026-09-28).
7. Table 6 "four-view 400 ep report-only": the P36 table row is COMPLETED 400/400 with 85.90 / 83.58 (b7fbc2b); dagger can be removed.
8. Table 6 "saturation claims": restrict to horizon — 0 % positives above 0.95 at 200 ep / 2 views under detach; 61 % at 4v/800ep (P84).
9. §1 / §2: the second application is compared (B above), no task selected; the R-line "gating explains noise tolerance" hypothesis is refuted (P72, P80); native
   calibration is shared with balanced JS (σ(2f) = (1+T)/2) and is not VCS-specific.
10. Related work to add (package v2 §12, supplement §6, plan v2 appendix C): Breiman 1996 / Wolpert 1992 (stacking), Friedman 2001 and Bühlmann–Yu 2003 (L2 boosting —
    the residual candidate is one L2Boost step), Gneiting–Raftery 2007 and Reid–Williamson 2011 (proper scoring rules / Bregman increments), Goldfeld et al. 2020
    (smoothed empirical measures; background only), Chuang et al. 2020 and Huynh et al. 2022 (false negatives; question only, no debiasing used), Wang–Isola 2020,
    Song–Ermon 2020 and McAllester–Stratos 2020 (no claim of beating the exponential sample requirement on the MI scale), Tsai et al. 2021 (RPC balanced case = J),
    arXiv 2409.04747 / CorInfoMax (log-det SSL objectives, 93 % CIFAR-10; the plan's note on the Φ(β, 2n) identity), Poly-View (multi-crop), Deep InfoMax JS / f-GAN.
11. Manuscript positioning (plan v2 §1, §10): contribution = the estimator (mixture reference, exact quadratic variational form, bounded optimal critic, closed-form/neural
    duality); SSL is the main application reported with equal-budget controls; no "superior" wording unless a claim survives against JS (none has so far).

## E. Wording checklist (required / forbidden)
Required: "under equal tuning budgets both controls exceed the frozen VCS recipe at every compute point; the gap narrows with compute (SimCLR +4.0 → +1.5 on the test set)";
"the official test set was evaluated once for the frozen cells; selection and test agree within seed noise"; single-seed cells are labelled; strong-aug 88.26 is one seed and
not part of the frozen recipe; every control set contains JS; mechanism statements are observational at named checkpoints; "parity, not advantage" for B-S2/B-T2;
"conditional" verdicts keep their conditions; estimator comparisons across targets only on target-free axes (separability, power, tail risk, learned-vs-oracle gap);
the negative construction is named in every table; J ∈ [−3, 1] (no "nats" for VCS); "sensitivity, not reliance" for D-T; equal budgets and the search cost disclosed.
Forbidden: "beats / matches SimCLR" as a VCS number; "stable alternative to MINE" before P85 reports; "VCS is uniquely calibrated"; "noise-robust" (refuted); "converges
to S" without the metric; "higher power than HSIC"; the certified Hoeffding bound as practical; "combination / residual improves the estimator" (no gain on frozen features,
capacity-uncontrolled on Gaussian); any CIFAR-100 / ImageNet number before P91 reports; test-set numbers for any cell outside P68 / P75.
