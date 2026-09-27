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
- **B-T2** real multimodal registration (IXI PD/T2, CC BY-SA 3.0): the official server refuses this cluster (HTTP 403 on every path tried), and
  fetching the archives from a third-party mirror was not permitted in this session.  Status: *pending (data)* until the owner supplies the
  archives or decides otherwise (`P60_TASK_B_IXI_PREREG_DRAFT_20260927.md` holds the licence text, the access evidence and the frozen design).

## 4. Family table (brief appendix A, row B) after B-S1 / B-T1
Unlabelled embedding probes / measurement on frozen features: *candidate*, with the qualification that the closed-form J* offers
cost and determinism, not better discrimination than CKA in the strong-dependence regime.  Registration / calibration / stereo energies:
unchanged (closed by P52) until B-S2 and B-T2 report.
