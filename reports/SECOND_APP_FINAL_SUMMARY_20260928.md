# Second application — final technical summary after wave 2 (2026-09-28)

Scope: everything run for the second-application brief (`CS_QMI/VCS_QMI_Second_Application_Server_Brief_v1.md`) — the four property pre-checks
(wave 1, `SECOND_APP_PRECHECKS_SUMMARY_20260927.md`), the supplementary and task-level units of wave 2 (`SECOND_APP_WAVE2_PLAN_20260927.md`),
the two fMRI units, and the equal-budget control tuning of the first application (P41), which the owner asked to run in the same pass.
Every unit was pre-registered before its compute (frozen files and dated addenda), uses identity splits with recorded hashes, touched no
official test set, gave competitors the same tuning grid, kept failed configurations, and stayed inside the movable set.  Per the brief §4 no
task is selected here; the surviving families are handed back.  Pending at the time of writing: the one-shot official test-set evaluation (P67, frozen and running).

## 1. Verdict table (property level and task level)
| row | unit | verdict | one-line evidence | report |
|---|---|---|---|---|
| A | A (wave 1) calibration & threshold transfer | conditional | native (1+T)/2 as calibrated as Platt only in the mid-dependence topic pairing; no transfer advantage (monotone critic) | P50 |
| A | A-S1 calibration-vs-J curve (2 shifts × 4 pairings) | **does not hold** | calibration tracks the *training relation*, not the J interval: exact-pair cells are 0.04–0.05 worse than Platt at J 0.63–0.92; topic/coarse cells within 0.005 | P58 |
| A | A-T mismatch detection without a target calibration set | conditional | native rule closest to nominal among deployable rules (0.071 vs 0.111 off at nominal 0.8) but never within 0.05; all rules over-shoot; task undefined on the indoor→outdoor shift (AUROC 0.53) | P58 |
| A | external target (Localized Narratives / Open Images) | conditional | native within +0.004 / +0.010 of Platt at J 0.56 / 0.32; both ECE ≈ 0.10 out of domain; COCO adapters lower AUROC vs raw CLIP (0.95 / 0.86 vs 0.985) | P58 §4 |
| B | B1 (wave 1) closed form vs neural critic | holds | gap ≤ 0.012 on CIFAR checkpoints | P48 |
| B | B-S1 closed form on cross-modal CLIP features | holds | +0.004 on the source, −0.075 (closed form better) on the shifted target | P59 |
| B | B2 (wave 1) registration energy, constructed pairs | does not hold | 10.2 vs 6.5 maxima; success 0.77 / 0.54 vs 0.88 / 0.70 | P52 |
| B | B-S2 variants (+ combination) | not re-opened | patch features remove the extra maxima, coarse-to-fine restores success; the combination reaches parity (6.6 vs 6.5; 0.94 / 0.79 vs 0.93 / 0.79) — parity, not advantage | P59 |
| B | B-T1 layer-correspondence probe vs CKA | conditional | 5/6 vs 6/6 (same method), 5/6 vs 5/6 (cross method); J* saturates 0.89–0.98 where CKA spans 0.2–0.98 | P59 |
| B | B-T2 real registration: RGB–NIR / IXI PD-T2 / fMRI EPI-T1w | **does not hold** ×3 | J* success 0.50/0.32, 0.90/0.83, 0.88/0.84 vs MI 0.70/0.52, 1.00/0.99, 1.00/0.99; the combination reaches parity on IXI only, half the gap on RGB–NIR, and harms EPI/T1w | P59 |
| C | C (wave 1) batch decoupling on frozen towers | does not hold | all methods flat within 1 % from B 512 to 16 | P54 |
| C | C-S1 from-scratch batch sweep (3 seeds) | conditional | retention at B 32: VCS 0.979, SimCLR 0.977, VICReg 1.001 — no VCS-specific batch effect | P61 |
| C | C-T queue / streaming negatives | **does not hold** | both objectives collapse without a momentum encoder (VCS 0.35, SimCLR 0.12 of reference) | P61 |
| D | D (wave 1) leakage-detection power | conditional (revised) | permutation-calibrated VCS test on par with HSIC (never ≥ 0.10 below), exact conditional test, monotone Ĵ; guaranteed bound 20–50× more sample-hungry; the false-alarm excess that closed it by the letter was the single-N-draw null (addendum 6) | P46 §3.4 |
| D | D-S1 / S2 / S3 (level pin, second seed, blur family) | confirmed / conditional / holds | fresh-N nulls 0.05–0.06 at R = 1000; seed 1 replicates; blur: parity 24/24, and the conditional VCS test beats HSIC 0.98 vs 0.18 on one encoder | P46-W2 |
| D | D-T shortcut-reliance detection (colour, blur) | conditional ×2 | Ĵ tracks reliance (Spearman 0.95 / 0.94), power 1.00 at n = 500 for ρ ≥ 0.8 / 0.95; null cells scatter for the same design reason; detects *sensitivity*, not reliance; magnitude saturates | P46-W2 §4, §6 |
| D | P65 / P66b fMRI motion leakage (circular-shift nulls) | **does not hold** | low-motion cohort: VCS 0.43, HSIC 0.47, QC-FC 0.30 of subjects at full length; monotone in cleaning for 43 %; finding: lagged (1–3 s) motion dependence survives 24-parameter cleaning in a fifth of subjects | P66 |

## 2. Family table (brief appendix A), final
| row | candidate / conditional | closed |
|---|---|---|
| A | mismatch detection, open-set rejection, pointwise PMI maps — *conditional*: only when the critic was trained on the relation it will score and the pair problem is solvable under the shift; the value over a source-fitted Platt is 0.02–0.04 in precision-to-nominal and 2× acceptance | data re-weighting; threshold transfer; "calibrated in a J regime" as a general statement |
| B | unlabelled embedding probes / dependence measurement on frozen features (B1, B-S1) — closed form = neural critic, cheaper and deterministic, not more discriminative than CKA in the strong-dependence regime | registration, extrinsic calibration, cross-spectral stereo cost (constructed pairs and three real sources; engineering brings J* to MI's level on the easy case and nowhere past it) |
| C | — | micro-batch, memory bank / streaming, k-way, asymmetric sample sizes |
| D | leakage & shortcut detection, dependence profiles, domain-shift magnitude — *conditional*: an O(n) permutation-calibrated test with an exact conditional version for discrete Y and a monotone magnitude, on par with HSIC (stronger in the within-class conditional test on one encoder); the certified bound only for S ≳ τ_{n,n}; on real fMRI it added nothing in a low-motion cohort | invariance penalties (not tested); any claim of *higher* power than HSIC |

## 3. First application, comparison state (selection split; `P41_CONTROL_TUNING_REPORT_20260928.md`)
VCS frozen recipe: 83.19 ± 0.40 / 84.54 ± 0.16 / 85.95 ± 0.35 / 87.01 ± 0.53 linear at 1× / 2× / 4× / 8× (16× singles 87.50, 87.14).  Tuned SimCLR:
86.64 / 87.84 / 88.35 / 88.32; tuned VICReg: 87.12 / 86.73 / 87.28 / 87.11.  Under equal tuning both controls exceed VCS at every
budget; the gap narrows with compute (SimCLR +3.5 → +1.3).  The mechanism results (bounded calibrated critic, h-uniformity ordering, no batch
statistics) stand and are not accuracy claims.

## 4. Methodological lessons recorded for the paper's methods section
1. **Null construction with a fixed nuisance draw** (D): repeats that are subsets of one realised assignment share its chance association;
   the permutation test is exact per repeat but the *rate over repeats* is not 0.05 — re-draw N per repeat (or say the rate is per realisation).
2. **Circular-shift nulls on short blocks** (P65): restricting admissible shifts can leave a dozen distinct shifts and quantise the p-value;
   use the full orbit.
3. **Letter vs intent** of a frozen clause (D §3.1, C-S1): when the data land on a threshold within seed noise, report both and let a pre-registered
   diagnosis, not a re-reading, move the verdict.
4. **Sensitivity is not reliance** (D-T): a conditional dependence test flags features that carry the nuisance whether or not the decision uses
   them; the magnitude, not the rejection, tracks reliance — and it saturates near determinism.
5. **Parity is not advantage** (B-S2): a closed form that can be engineered to match a 30-year-old baseline is not a reason to replace it.
6. **Equal budgets change the headline** (P41): the untuned controls already matched VCS's 2× cell; tuning added 0.5–2.3 points.

## 5. Data, provenance, compute
CIFAR-10 (local copy; official test untouched); COCO 2017 (local); CLIP ViT-B/32 laion2b (owner-approved download, sha256 recorded); EPFL RGB–NIR
(owner accepted the unlicensed page; sha256 7c465a23…); IXI PD/T2 (CC BY-SA 3.0; mirror hashes declared before download; owner-run);
AOMIC-PIOP1 ds002785 (CC0; anatomical + functional derivatives; 12 GB); Localized Narratives (CC BY 4.0) on Open Images validation (CC BY 2.0,
per-image attribution).  Large datasets moved to `/projects/EEG-foundation-model/yinghao/CS_QMI_data/` with symlinks.  Wave 2 + P41 used ≈ 250
GPU-hours over two days; every job id is in `reports/job_ids.json`; every unit has a results-only commit before its interpretation commit
(two exceptions where a glob hit ignored files are disclosed in their commit messages).

## 6. Pending and proposed
Pending: P67 official test-set evaluation (52 units, once; running).  Proposed, not run: 16× control
(SimCLR 4v/1600ep); momentum-encoder queue for both objectives; high-motion or task-fMRI cohort and aCompCor reference for P65; the D
conditional cases with N re-drawn; a within-block exact conditional test for time series.

## 7. Official test set (P67 / P68, once, 2026-09-28)
Test and selection agree to within seed noise across all 22 cells (mean Δ −0.03 linear, +0.35 kNN).  Test linear, 3 seeds: VCS 82.98 / 84.52 / 85.87 /
86.65 at 1× / 2× / 4× / 8×; tuned SimCLR 86.94 / 87.99 / 88.21 / 88.13; tuned VICReg 87.00 / 87.11 / 86.91 / 86.87; VCS singles 87.60 (16×) and 88.26
(8× strong-aug).  The equal-budget picture of §3 holds on the test set: SimCLR leads at every budget (+4.0 → +1.5), VICReg leads until it is level with
VCS at 8× on the linear probe (VCS ahead on kNN there).  Report: `P68_FINAL_OFFICIAL_TEST_REPORT_20260928.md`.  Nothing is pending.
