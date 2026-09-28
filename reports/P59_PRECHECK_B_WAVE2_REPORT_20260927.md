# Row B, wave 2 — report (P59): B-S1 closed form on cross-modal features, B-T1 layer-correspondence probe; B-S2 and B-T2 pending

Prereg: `P59_PRECHECK_B_WAVE2_PREREG_FROZEN_20260927.md` (probes disclosed there).  Tables: `P59_precheck_B_S1_clip.md`, `P59_task_B_T1_layerprobe.md`,
`P59_precheck_B_S2_{fourier121,patch,smooth2,smooth4,ctf}.md`, `P60_task_B_T2_rgbnir.md` (results-only commits d322a00, d330a38 and the B-S2 commit).
Evidence category: completed for B-S1, B-T1, B-S2 and B-T2 source 1; B-T2 source 2 (IXI) pending data.

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

## 3. B-S2 and B-T2
- **B-S2 — registration-energy variants on the P52 constructed pairs (60 images, same seed; GPU jobs 1011141 / 1011191 / 1011192 / 1011204 / 1011205).**
  Baseline P52: J* 10.2 local maxima, success 0.77 / 0.54 at R = 20 / 30; MI / NMI 6.5, 0.88 / 0.70.

  | variant (applied to all measures where it is a preprocessing) | J* maxima | MI / NMI maxima | J* success R = 20 / 30 | MI / NMI success | J* peak at truth |
  |---|---|---|---|---|---|
  | Fourier features 49 → 121 | 10.4 | 6.5 | 0.78 / 0.54 | 0.88 / 0.70 | 1.00 |
  | + 8 × 8 block-mean channel (patch features) | **6.6** | 6.5 | 0.85 / 0.64 | 0.88 / 0.70 | 1.00 |
  | pre-smoothing σ = 2 px | 6.7 | 4.5 / 4.6 | 0.83 / 0.64 | 0.91 / 0.73 | 1.00 |
  | pre-smoothing σ = 4 px | 5.0 | 3.1 / 3.2 | 0.61 / 0.47 | 0.91 / 0.73 | 0.83 |
  | coarse-to-fine optimiser (64 → 128 → 256 px) | 10.2 (surface unchanged) | 6.5 | **0.94 / 0.76** | 0.93 / 0.79 | 1.00 |

  **Reading (frozen: re-opened only if one variant matches MI/NMI on local maxima *and* success within 0.05): not re-opened.**  No single
  variant does both: the block-mean feature class removes the extra maxima (6.6 vs 6.5) but leaves success 0.03 / 0.06 short; the
  coarse-to-fine optimiser brings success to parity (+0.01 / −0.03) but the fine-scale surface keeps its 10.2 maxima; more Fourier features
  change nothing; smoothing helps MI more than J*, and at σ = 4 px J* loses its peak at the truth in 17 % of images.  Substance: the roughness
  of J* is not intrinsic to the objective — it is a feature-class effect that a low-frequency channel or a multi-resolution optimiser
  compensates — but even then J* only *equals* MI, and nothing in these five variants gives it an advantage.  The family stays closed:
  matching a 30-year-old baseline at best is not a reason to use J* as a registration energy.  **Addendum 1 (owner: "都补充上") — the combination patch features + coarse-to-fine (GPU job 1011782):** J* 6.6 local maxima vs 6.5, basin
  22 / 16 vs 24 / 21 px / deg, success 1.00 / 1.00 / 0.94 / 0.79 vs 1.00 / 1.00 / 0.93 / 0.79, peak at truth 1.00, no divergence (MI / NMI diverge
  in 1 % of the R = 30 starts).  Reading (addendum 1: re-opened only if maxima ≤ NMI's *and* success within 0.05): success is at parity
  (+0.01 / 0.00) and the maxima differ by 0.1 per image — by the letter (6.6 > 6.5) the clause is not met, by any practical reading J* now
  *equals* MI on every metric.  Either way the addendum's own sentence applies: parity is not advantage, and the family stays closed.  What
  changes is the diagnosis — the roughness that P52 measured is entirely a feature-class and optimiser effect, not a property of the
  objective; the real-data runs (RGB–NIR, and IXI / fMRI when they land) used the Fourier-49 class without coarse-to-fine, so their J* deficit
  is an upper bound on what the plain closed form loses, not the floor.  Re-running the combination on the real sources is a natural next
  step; it was not pre-registered and is listed as *proposed*.
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
- **B-T2, source 2 — real multi-contrast MRI (IXI PD/T2, CC BY-SA 3.0; 60 subjects, mid-axial slices; archives fetched by the owner from the
  declared-hash mirror after the official server refused the cluster; GPU job 1011907).**

  | measure | peak at truth (transl. / rot.) | mean # local maxima | median basin half-width px / deg | success R = 5 / 10 / 20 / 30 | divergence |
  |---|---|---|---|---|---|
  | closed-form J* (Fourier 49) | 1.00 / 1.00 | 1.8 | 14 / 10 | 1.00 / 0.98 / 0.90 / 0.83 | 0.00 |
  | histogram MI | 1.00 / 1.00 | 1.0 | 24 / 30 | 1.00 / 1.00 / 1.00 / 0.99 | 0.00 |
  | NMI | 1.00 / 1.00 | 1.0 | 24 / 30 | 1.00 / 1.00 / 1.00 / 1.00 | 0.00 |

  **Reading (frozen, as RGB–NIR): does not hold** — success 0.10 / 0.17 below NMI at R = 20 / 30, 1.8 vs 1.0 local maxima, a basin half as wide.
  On dual-echo PD/T2 the intensity relation is almost one-to-one, so histogram MI is unimodal over the whole ± 24 px / ± 30° range and succeeds
  from every start; J* also peaks at the truth in every subject and never diverges, but its narrower basin loses one start in six at R = 30.
  Third real source (fMRI EPI-boldref vs T1w) pending.
- **B-T2, source 3 — fMRI EPI-boldref vs T1w (AOMIC-PIOP1, CC0; 60 subjects; brain-masked mid-axial slices, EPI resampled onto the 1 mm T1w grid;
  GPU job 1011928; slice previews in `reports/slices_fmri/`).**

  | measure | peak at truth (transl. / rot.) | mean # local maxima | median basin half-width px / deg | success R = 5 / 10 / 20 / 30 | divergence |
  |---|---|---|---|---|---|
  | closed-form J* (Fourier 49) | 1.00 / 0.98 | 1.2 | 22 / 15 | 0.97 / 0.95 / 0.88 / 0.84 | 0.00 |
  | histogram MI | 1.00 / 1.00 | 1.0 | 24 / 30 | 1.00 / 1.00 / 1.00 / 0.99 | 0.00 |
  | NMI | 1.00 / 1.00 | 1.0 | 24 / 30 | 1.00 / 1.00 / 1.00 / 0.99 | 0.00 |

  **Reading (frozen, as the other real sources): does not hold** — success 0.12 / 0.15 below NMI at R = 20 / 30, 1.2 vs 1.0 local maxima, a
  rotation basin half as wide (15° vs 30°).  Functional-to-structural registration is MI's textbook case and MI is again near-perfect; J*
  always finds the truth from nearby but loses one start in six from 30 px / 30°.  Three real sources (cross-spectral RGB–NIR, dual-echo PD/T2,
  EPI/T1w) and the constructed pairs now agree, with the same signature each time: correct global maximum, no divergence, narrower basin.


- **Addendum 2 — the combination variant (patch features + coarse-to-fine) on the three real sources (GPU jobs 1012288–1012290; the P52-path
  numbers of each source are the rows above).**

  | source | J* maxima: plain → combo (MI) | J* success R = 20 / 30: plain → combo (MI with the same coarse-to-fine) | J* peak at truth (transl. / rot.) | reading |
  |---|---|---|---|---|
  | RGB–NIR | 16.4 → 11.6 (8.1) | 0.50 / 0.32 → 0.67 / 0.48 (0.82 / 0.65) | 0.93 / 0.95 | closes about half the gap; still 0.15 short |
  | IXI PD/T2 | 1.8 → 1.1 (1.0) | 0.90 / 0.83 → 1.00 / 1.00 (1.00 / 1.00) | 1.00 / 1.00 | parity in success; maxima 1.1 vs 1.0 (letter: not ≤); rotation basin still 14° vs 30° |
  | fMRI EPI/T1w | 1.2 → 1.1 (1.0) | 0.88 / 0.84 → 0.81 / 0.81 (1.00 / 0.99) | **0.50 / 0.80** | the block-mean channel *hurts*: the translation surface peaks off the truth in half the subjects |

  **Reading (addendum 2; family unchanged either way): not re-opened.**  The removability of the plain closed form's deficit is
  source-dependent: complete on dual-echo PD/T2 (where MI was already perfect), partial on cross-spectral pairs, and negative on EPI/T1w, where
  the 8 × 8 block-mean channel — a low-frequency intensity feature — biases the surface for a modality pair whose intensities are not even
  monotonically related.  Coarse-to-fine helps every measure (MI's own success on RGB–NIR rises 0.71 / 0.50 → 0.82 / 0.65), so the optimiser
  is not where J* gains on MI.  Conclusion for row B: as an alignment energy the closed form can be engineered to MI's level on the easy
  case and nowhere past it; the family stays closed on constructed pairs and on all three real sources.

## 4. Family table (brief appendix A, row B) after B-S1 / B-T1
Unlabelled embedding probes / measurement on frozen features: *candidate*, with the qualification that the closed-form J* offers
cost and determinism, not better discrimination than CKA in the strong-dependence regime.  Registration / calibration / stereo energies: **closed** — P52 (constructed pairs) is confirmed on real cross-spectral pairs (RGB–NIR) and on
real multi-contrast MRI (IXI PD/T2, where MI is essentially perfect); the B-S2 combination shows the plain closed form's deficit is removable
down to parity, never to an advantage.
