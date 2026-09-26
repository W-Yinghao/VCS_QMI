# Geometry diagnostic (alignment / uniformity, Wang & Isola) — 2026-09-26T02:13:59Z

Selection set, two train-distribution views (fixed RNG), frozen final checkpoints. Lower alignment = views closer; lower (more negative) uniformity = points spread more evenly on the sphere.

| run | method | ep | critic | K | negdet | z: align | z: unif | z cos+ median | z cos− median | h: align | h: unif | h cos+ med | h cos− med | a / b / thr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P5_simclr_seed0 | simclr_matched | 200 | None | 1 | False | 0.262 | -3.842 | 0.923 | -0.010 | 0.236 | -2.807 | 0.917 | 0.239 | — |
| P5_vicreg_seed0 | vicreg_matched_128 | 200 | None | 1 | False | 0.274 | -3.678 | 0.937 | -0.015 | 0.208 | -2.523 | 0.927 | 0.270 | — |
| P24_vcs_cos_negdetach_seed0 | vcs_qmi | 200 | cosine | 8 | True | 0.090 | -1.563 | 0.965 | 0.575 | 0.192 | -1.792 | 0.922 | 0.516 | 9.81 / -8.09 / 0.825 |
| P26_vcs_a5_learn_seed0 | vcs_qmi | 200 | cosine | 8 | True | 0.101 | -1.560 | 0.959 | 0.584 | 0.208 | -1.793 | 0.913 | 0.525 | 10.08 / -8.22 / 0.816 |
| P28_vcs_views4_seed0 | vcs_qmi | 200 | cosine | 8 | True | 0.101 | -1.675 | 0.958 | 0.553 | 0.225 | -2.025 | 0.906 | 0.463 | 10.65 / -8.62 / 0.809 |
| P35_vcs_a5_views4_seed0 | vcs_qmi | 200 | cosine | 8 | True | 0.109 | -1.651 | 0.954 | 0.565 | 0.238 | -2.017 | 0.899 | 0.472 | 10.83 / -8.70 / 0.803 |
| P37_vcs_a5_views8_100ep_seed0 | vcs_qmi | 100 | cosine | 8 | True | 0.199 | -2.558 | 0.914 | 0.317 | 0.214 | -1.863 | 0.909 | 0.513 | 6.70 / -4.45 / 0.664 |
| P26_vcs_base_800ep_seed0 | vcs_qmi | 800 | cosine | 8 | True | 0.045 | -0.927 | 0.982 | 0.761 | 0.265 | -2.283 | 0.889 | 0.392 | 21.12 / -19.18 / 0.908 |
| P24_vcs_cos_800ep_seed0 | vcs_qmi | 800 | cosine | 8 | False | 0.381 | -3.199 | 0.822 | 0.155 | 0.286 | -1.789 | 0.873 | 0.539 | 13.03 / -6.95 / 0.533 |

## Reading (interpretation commit)
| run | linear | h: unif | h cos− med | z: unif | z cos− med | thr |
|---|---|---|---|---|---|---|
| SimCLR / VICReg | 86.4 / 85.6 | **−2.81 / −2.52** | 0.24 / 0.27 | −3.84 / −3.68 | −0.01 / −0.02 | — |
| detach base (a0 = 1, 2 v, 200 ep) | 80.5 | −1.79 | 0.52 | −1.56 | 0.58 | 0.83 |
| a0 = 5 (2 v, 200 ep) | 81.9 | −1.79 | 0.53 | −1.56 | 0.58 | 0.82 |
| 4 views (a0 = 1 / a0 = 5, 200 ep) | 84.5 / 84.5 | **−2.03 / −2.02** | 0.46 / 0.47 | −1.68 / −1.65 | 0.55 / 0.57 | 0.81 / 0.80 |
| 8 views (a0 = 5, 100 ep) | 83.4 | −1.86 | 0.51 | −2.56 | 0.32 | 0.66 |
| detach base, 800 ep | 84.6 | **−2.28** | 0.39 | −0.93 | **0.76** | 0.91 |
| plain cosine, 800 ep (no detach) | 80.7 | −1.79 | 0.54 | −3.20 | 0.16 | 0.53 |

1. **h-uniformity still orders the VCS runs by linear accuracy and still separates them from the controls**: −1.79 (80.5–81.9) → −2.02
   (84.5) → −2.28 (84.6) vs −2.5 / −2.8 for the controls.  Negative-pair cosine in h falls 0.52 → 0.47 → 0.39 vs 0.24–0.27.  The two
   recipe factors that helped (4 views, long schedule) are exactly the two that moved h-uniformity; a0 = 5 moved neither (−1.79 both) and
   gave the smaller, non-additive gain.
2. **z-uniformity is anti-correlated with linear under detach.**  The 800-epoch detach base has the *least* uniform z of all runs (−0.93;
   negatives at cosine 0.76, positives 0.98; the whole code lives in a cone of half-angle ≈ 40°) and the best h.  The critic threshold moves
   up with training (0.83 → 0.91) and the negatives are pushed just below it, never further — the objective is satisfied as soon as the
   pairs are separable (synthesis §4.1), and detach lets the encoder satisfy it with a narrow z cone while h spreads.  Plain cosine without
   detach does the opposite (z −3.20, h −1.79) and is 4 points worse.  **Where the spreading happens (z vs h) is the mechanism, not how much.**
3. 8 views at 100 epochs (threshold 0.66, z −2.56) sits between the two regimes and is 1.1 below 4 views/200 ep: with more pairs per step
   and fewer steps the critic does not sharpen enough to push the spreading into h.

Not claimed: single seed per run; correlational (no intervention on the threshold); selection set.
