# Row B, wave 2 — report (P59): B-S1 closed form on cross-modal features, B-T1 layer-correspondence probe; B-S2 and B-T2 pending

Prereg: `P59_PRECHECK_B_WAVE2_PREREG_FROZEN_20260927.md` (probes disclosed there).  Tables: `P59_precheck_B_S1_clip.md`, `P59_task_B_T1_layerprobe.md`
(results-only commit d322a00).  Evidence category: completed for B-S1 and B-T1 (CPU jobs 1011126, 1011140); B-S2 (five GPU variants) and
B-T2 (IXI; data blocked, see §3) pending.

## 1. B-S1 — closed-form linear-class critic vs the trained cosine critic on CLIP adapter features (topic pairing, 3 seeds)
| split | neural critic J (mean of 3 seeds) | best tanh-closed-form J (class P1, 513-d) | gap neural − closed |
|---|---|---|---|
| SRC-EVAL | 0.571 | 0.567 | +0.004 |
| TGT-EVAL (animal shift) | 0.165 | 0.240 | −0.075 |

Per seed the in-domain gap is +0.004 / +0.008 / +0.028 (P1 / P2 / P3) on the source and −0.04 to −0.08 on the target for every class.
The transfer table (closed form fitted on SRC-CAL only, evaluated on the target) gives 0.19 (P1) vs the neural 0.17 — still no gap.
**Reading (frozen: gap ≤ 0.02 → holds): holds.**  On cross-modal frozen features the closed-form critic in the smallest linear class
matches the trained critic on the source and exceeds it on the shifted target; the larger classes (P2, P3) over-fit slightly in-domain, as
in P48.  Same caveat as B1: this is a statement about *measurement* on frozen features, not about training.

## 2. B-T1 — closed-form J* (bilinear class on PCA-64-whitened features) as an unlabelled embedding probe: layer correspondence
| pair | vcs_tanh (J*) | vcs_closed (no tanh) | CKA linear | CKA RBF |
|---|---|---|---|---|
| VCS seed 0 vs VCS seed 1 (row-wise / column-wise of 6 stages) | 5 / 5 | 1 / 1 | 6 / 6 | 6 / 6 |
| VCS seed 0 vs SimCLR seed 0 | 5 / 5 | 4 / 1 | 5 / 5 | 5 / 5 |

The single miss of the tanh probe is the h row in both pairs: it prefers the other network's projector output z (0.965 vs 0.940; 0.956 vs 0.866),
i.e. it confuses two stages that carry the same information.  Invariances: orthogonal rotation and isotropic scaling change all measures by
< 10⁻⁷; per-feature scaling moves J* by −0.044 and CKA by −0.16 / −0.17 (J* is the *less* sensitive of the two).
**Reading (frozen: ≥ CKA's identifications → holds; loses one stage → conditional): holds conditionally** — equal to CKA on the
cross-method pair, one stage short on the same-method pair.  What the matrices show beyond the count: the J* probe saturates — 0.89–0.98
on the diagonal with off-diagonal values 0.62–0.93 — whereas CKA spans 0.2–0.98.  This is the near-deterministic regime the brief calls out
(strong dependence pushes S toward its bound and the statistic loses contrast); as a probe it therefore separates *matching from
non-matching* stages by a small margin only, and it is not a better probe than CKA on this evidence.  The untransformed closed form
(`vcs_closed`) is not usable as a probe (1/6): the tanh scalar chosen on validation is what makes the bilinear J* readable.

## 3. Pending
- **B-S2** registration-energy variants (Fourier 121, patch features, smoothing σ = 2 / 4 px, coarse-to-fine): jobs 1011141–1011145 class,
  ≈ 1.7 GPU-h each; appended here when they land.
- **B-T2, source 1 — real cross-spectral pairs (EPFL RGB–NIR Scene, 60 of 477 pairs, stratified over 9 categories; owner accepted the data
  2026-09-27; provenance sha256 7c465a23…, pair-list hash 6c262c3e… in the frozen P60 prereg; GPU job 1011199, 55 min).**

  | measure | peak at truth (transl. / rot.) | mean # local maxima | median basin half-width px / deg | success R = 5 / 10 / 20 / 30 | divergence |
  |---|---|---|---|---|---|
  | closed-form J* (Fourier 49) | 0.93 / 0.98 | 16.4 | 15 / 12 | 0.78 / 0.68 / 0.50 / 0.32 | 0.00 |
  | histogram MI | 0.98 / 0.98 | 8.1 | 20 / 21 | 0.95 / 0.89 / 0.71 / 0.50 | 0.00 |
  | NMI | 0.98 / 0.98 | 8.3 | 20 / 21 | 0.95 / 0.89 / 0.70 / 0.52 | 0.00 |

  **Reading (frozen: does not hold if J* has more local maxima than NMI or success lower by ≥ 0.05): does not hold** — twice the local
  maxima (16.4 vs 8.3), success 0.20–0.21 below NMI at R = 20 / 30, peak-at-truth 0.93 vs 0.98 on translation, no divergence for any measure.
  This is P52's constructed-modality result reproduced on real data, and it is *worse* for J* than on the tent-map pairs (P52: 10.2 vs 6.5
  maxima, success 0.77 / 0.54 vs 0.88 / 0.70): real RGB–NIR intensities are largely monotonically related, which is MI's easy case, while the
  49-feature linear class still has to represent the relation piecewise.  The registration family stays closed on real cross-spectral data.
- **B-T2, source 2 — real multi-contrast MRI (IXI PD/T2, CC BY-SA 3.0; prereg frozen):** the official server refuses this cluster (HTTP 403)
  and the mirror fetch is not permitted to the assistant; the owner runs the download job, after which a watcher verifies the declared sha256
  and chains the smoke and the full run.  Status: *pending (data)*.

## 4. Family table (brief appendix A, row B) after B-S1 / B-T1
Unlabelled embedding probes / measurement on frozen features: *candidate*, with the qualification that the closed-form J* offers
cost and determinism, not better discrimination than CKA in the strong-dependence regime.  Registration / calibration / stereo energies: **closed** — P52 (constructed pairs) is now confirmed on real cross-spectral pairs (B-T2
source 1); B-S2's variants and the MRI source can only qualify, not reverse, this unless a variant matches MI on both maxima and success.
