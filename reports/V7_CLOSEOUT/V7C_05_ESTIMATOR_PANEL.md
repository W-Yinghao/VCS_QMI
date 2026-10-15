# V7C-05 — estimator panel closeout: native targets, same target, learning systems and selection (2026-10-15)

Sources: `reports/V7/V7X_X11_results/` (synthetic anchors), `reports/V7/V7X_X21_results/` (common-feature fits on EvalMuse DEV; main 768-d and the three sensitivities), `reports/V7/V7_TABLE_A_results/` (full models), `reports/V7_CLOSEOUT/V7C_04_results/` (GenAI-Rank selection).  No new unit was trained for this closeout.

## 1. The four panels
| panel | what is compared | result |
|---|---|---|
| **A0 native targets** (X11, synthetic, own truths) | classical CS-KDE (fixed bandwidth multiples, Scott, Silverman), MINE / DV, NWJ on their own targets | Classical CS-KDE misses its truth at every fixed bandwidth in d = 20 (e.g. δ = 2: multiple 1.0 gives 0.048 against a truth of 2; Scott gives 7.06). The KL-bridge critics diverge at strong dependence (δ = 2: DV −0.19 and NWJ −5.6 against a truth of 2). These are native-scale readings, not natural-data errors. |
| **A1 same target S** (X11 synthetic; X21 EvalMuse DEV natural Brier) | KDE posterior, RuLSIF (α = ½), Nyström squared regression, neural VCS / BCE / DV / NWJ on the same P / Q | **X11:** learned VCS ≈ BCE in posterior MSE (δ = 2: 0.0179 vs 0.0174). **X21 DEV:** see §2; VCS ≈ BCE ≈ DV ≈ NWJ, KDE clearly worse, Nyström / RuLSIF equal or slightly better depending on the representation. |
| **A2 learning systems** (Table A; full Q-former fine-tuning) | M0 L1 / M1 logistic / M2 VCS | M2 − M1 DEV natural Brier +0.0005 / +0.0005 / +0.0003 (every interval contains 0); M0 worse (0.181–0.184 vs ≈ 0.173). |
| **C selection** (GenAI-Rank, N = 9, frozen scorer) | full systems and common-feature fits with a single-sample interface | M2-E ≈ B0 FGA; behind VQAScore; ahead of TIFA / DSG on SDXL_Base (V7C-04).  X21 rows: VCS-F ≈ BCE-F ≈ Nyström ≈ RuLSIF; VCS-F ahead of KDE (SDXL_Base +0.074, Holm p 0.005) and of DV (DALLE_3 +0.036, Holm p 0.022). |

The MMD / HSIC rows stay in their test panel (X11: null rejection 0.035–0.060; on the A3 anchor with k = 1, ρ = 0.8, HSIC rejects at 1.00 where MMD rejects at 0.11).  Balanced RPC equals J and is not added as a separate competitor (`r4/package_r4_20261010/docs/03` §2).

## 2. X21 on EvalMuse DEV: natural Brier (prior-corrected; mean of the formal fits; lower is better)
| features | N-VCS-F | N-BCE-F | N-DV | N-NWJ | K-KDE-S | K-RULSIF | K-NYSTROM |
|---|---|---|---|---|---|---|---|
| main 768-d (untrained BLIP-2 Q-former) | 0.1805 | 0.1802 | 0.1810 | 0.1810 | 0.1964 | 0.1802 | **0.1789** |
| PCA-64 (sensitivity) | **0.1850** | 0.1855 | 0.1869 | 0.1862 | 0.1981 | 0.1853 | 0.1856 |
| PCA-256 (sensitivity) | 0.1847 | 0.1852 | 0.1870 | 0.1848 | 0.2172 | 0.1829 | **0.1814** |
| N0-s0 trained features (sensitivity) | 0.1753 | 0.1756 | 0.1766 | 0.1761 | 0.1826 | 0.1756 | **0.1743** |

The intervals for VCS-F minus each comparator are in `V7X_X21_results/*` (`contrasts_natural_brier_dev`).

**Reading.** Relative to a learned critic of this class, the density-estimation route (KDE) is clearly worse on every representation.  The squared-regression kernel route (Nyström) is the best single method on three of the four feature sets.  Learned critics are not uniformly better than kernels; the ordering depends on the dimension of the shared representation (PCA-64 favours the learned critics slightly, PCA-256 favours the kernels).  The CS-type objective has no distinct advantage over BCE, DV or NWJ on the same features.

## 3. Audit of the X21 fits (checklist of package docs/05 §2)
| item | finding |
|---|---|
| common features | one X20 extraction (untrained BLIP-2 COCO), FIT-only D-weighted standardisation; PCA bases fitted on FIT only; the N0-s0 sensitivity loads the N0 seed-0 checkpoint (counts recorded) |
| seeds / tuning | neural fits: tuning seed 17 over lr {1e-4, 3e-4, 1e-3}, formal seeds 0–2 at the chosen lr; kernels deterministic |
| bandwidth (KDE) | **selected at the extended edge** (0.125 × median distance) on all four feature sets, after its one pre-declared extension: KDE might do better at a smaller bandwidth than the protocol allowed |
| centres (RuLSIF / Nyström) | **selected at the extended maximum (8,192)** on all four feature sets: more centres might help |
| ridge | 1e-6 after the extension to 1e-8 (interior) |
| solve residuals | ≤ 7e-14 (relative); Nyström cond(K_CC) 1e9–7e9 with the 1e-6 jitter, harmless at these residuals |
| clip fractions | RuLSIF 1.6–3.1 %, Nyström 1.7–2.7 % of CAL_SELECT outputs outside the bounded range (raw outputs kept) |
| native vs common selection | identical configurations for RuLSIF / Nyström; KDE's native CV choice equals its common choice |
| DV normalisation / CAL_PROB | DV: ell = f − log E_Q[e^f] with the normaliser estimated once per selected model on CAL_PROB; NWJ: the raw bridge f − 1 without offset re-estimation (V7X-20); otherwise CAL_PROB serves only the secondary Platt row |

**Consequence.** The two edge findings mean the kernel rows are, if anything, under-tuned, and the kernels already match or beat the learned critics.  So the edges do not overturn the reading; they strengthen "kernels are not weak here".  A further extension (KDE multiple 0.0625; 16,384 centres) would be a post-hoc DEV sensitivity.  The GenAI-Rank labels have now been read, so it would not be pre-registered for that benchmark, and it is not run without an owner decision.

## 4. Statements the evidence supports, and limits
- **Supported.**
  - On this task and these representations, posterior estimation by a learned critic and by kernel squared regression (Nyström / RuLSIF) reach the same accuracy; Nyström is often the best single method.
  - Classical density estimation (KDE) is worse.
  - The quadratic (VCS) objective neither helps nor hurts relative to BCE, DV and NWJ.
- **Not supported.** "Learned critics always beat kernel methods"; "traditional CS methods fail"; any accuracy advantage specific to the CS objective.
- **Limits.**
  - One element-level task (EvalMuse).
  - Fixed untrained features for X21; the full models use a different (fine-tuned) path.
  - The KDE / centre edges noted above.
