# P84 report — estimator package v1 §8: estimator diagnostics on frozen SSL checkpoints (2026-09-28)

Pre-registration `P83_FROZEN_DIAG_PREREG_FROZEN_20260928.md`; GPU job 1013118 (exit 0, ≈ 55 s per checkpoint, 10 checkpoints); table
`P84_frozen_diag.md`, per-checkpoint JSON `P84_frozen_diag/` (results-only commit 2ffd1bb).  EVAL = 3 000 independent (anchor, partner) blocks
of 6 000 held-out identities; SE by block (0.001–0.008).  No oracle on images: J only, S and posterior MSE absent by design.

## Key numbers (EVAL J; higher is better)
| run / checkpoint | training critic | refit C0 (2 000 Adam steps) | refit C2 | **C0 converged (diag)** | VCS mix | VCS residual | JS C2 / JS mix |
|---|---|---|---|---|---|---|---|
| VCS 4v recipe, ep 100 | 0.948 | 0.815 | 0.969 | **0.973** | 0.969 | 0.970 | 0.972 / 0.972 |
| VCS 4v recipe, ep 400 | 0.981 | 0.837 | 0.982 | **0.991** | 0.982 | 0.983 | 0.984 / 0.984 |
| VCS 4v recipe, ep 800 | 0.986 | 0.843 | 0.986 | **0.993** | 0.986 | 0.987 | 0.987 / 0.987 |
| SimCLR 4v, ep 100 / 400 / 800 | — | 0.876 / 0.897 / 0.903 | 0.947 / 0.964 / 0.969 | **0.951 / 0.969 / 0.973** | = C2 | +0.001–0.002 | ≈ C2 |
| original VCS recipe (P5), ep 200 | **0.901** (MLP critic) | 0.644 | 0.893 | 0.814 | 0.893 | 0.894 | 0.898 / 0.898 |
| any run, initial (untrained) | −1.000 (P35) / 0.000 (P5) | 0.086 | 0.144 | 0.086 | 0.147 | 0.146 | 0.142 / 0.143 |

The three initial checkpoints give identical refit numbers: the runs share seed 0 and the same encoder/projector initialisation (their training
critics differ: P35's cosine critic with a₀ = 5, b = 0 starts saturated at T ≈ 1 on every pair, J = −1.000; P5's MLP critic starts at T ≈ 0, J = 0).

## 1. Where the estimator error sits on real representations (spec §16 Q1)
- **The training critic lags the best critic of its own class.**  On the VCS recipe's frozen z, the same 2-parameter cosine class, fitted to
  convergence, beats the training-time critic by **+0.025 / +0.010 / +0.007** at epochs 100 / 400 / 800 (5–10 block SE).  The converged critic is much
  steeper (at ep 800: a ≈ 81 on the raw cosine, threshold 0.91) than the training critic, whose positives stay unsaturated (median T 0.96, 61 % above
  0.95).  So the online estimator is fitting-limited (few critic steps per encoder step, detached negatives), and the lag shrinks as training
  settles — an optimisation, not a function-class, error.  This matches the Gaussian probe, where C0 / CQ were budget-limited.
- **Flexible refits are sample-limited here too.**  C2 (76 k parameters) reaches only the training critic's level at ep 400 / 800 and stays below
  the converged 2-parameter C0 on both VCS and SimCLR features — the same over-fitting pattern as C2 in the Gaussian probe.
- **The fixed-budget C0 row is not a class measurement** (0.84 vs 0.99 converged); it is recorded, not read.

## 2. What reduced the error (Q2)
- **Mix and residual: no measurable gain on frozen features.**  The VCS mix puts ≥ 0.98 weight on C2 after epoch 100 (the budget-limited C0 is a
  poor dictionary member) and equals C2; the residual adds +0.001–0.002 (≤ 1 SE) with λ = 0.28–0.57.  The converged C0 was a diagnostic outside the
  dictionary by pre-registration, so the one combination that could have helped (C0-converged + C2) was not tested — a next-batch item.
- **The largest available gain is simply fitting the training critic's own class to convergence** (+0.007 to +0.025 J).

## 3. JS control (Q3)
Matched-JS refits equal or slightly exceed VCS refits on every checkpoint (C2: +0.001–0.005; converged C0 identical to 3 decimals; JS mix = JS C2).
No VCS-specific estimator advantage on frozen representations.

## 4. Across training and methods
- VCS representations are more separable for the cosine class than SimCLR's at the same budget (converged C0 0.993 vs 0.973 at ep 800), although
  SimCLR has the higher linear-probe accuracy (P68: 88.1 vs 86.7 test) — the J of the representation does not order downstream accuracy.
- The original recipe's MLP training critic (P5, no detach) is *ahead* of every refit (0.901 vs 0.898 best): with a flexible critic trained
  alongside the encoder, the online fit is not the bottleneck; with the 2-parameter cosine critic it is.
- Train-mode gradients (VCS recipe): |grad_left| 0.0033 → 0.0026 and |grad_right| 0.0025 → 0.0018 from ep 100 to 800 (right = positives only,
  negatives detached as designed); at initialisation the saturated critic passes almost nothing to the right side (0.0007).

## 5. Proposed next items (not launched; need the owner's go per spec §14)
1. Dictionary / residual with a **converged** C0 member (and the Gaussian probe re-run with converged C0/CQ, the lr grid of the spec).
2. **Online consequence**: the training critic's lag suggests an SSL variant with more critic steps or a closed-form refresh of (a, b) per epoch —
   an online change, which the spec defers until the estimator results are reviewed.
3. Seeds 0–2 for P82 plus the capacity / equal-cost controls of the residual (P82 §5).
