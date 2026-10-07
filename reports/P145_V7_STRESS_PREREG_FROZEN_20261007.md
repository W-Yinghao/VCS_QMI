# Pre-registration — P145: v7 V7-STRESS — contaminated training positives (ε = 0.10) for VCS (A-P3), matched JS-AP3 and SimCLR, CIFAR-10 / CIFAR-100, seed 0 — FROZEN 2026-10-07T00:58:29Z

Owner 2026-10-05: "VCS_SSL_Server_Plan_v7 … 把这批次的分析也开始做" — plan v7 §6 (STRESS).  Order: plan §12 submits the four PAIR trainings and the six STRESS units in
batches after the HEAD units (P137 reported); frozen after the code gate.  VCS is **not** presumed more noise-tolerant (plan §6.1).

## 1. Intervention (plan §6.2; training positives only — downstream labels and evaluation unchanged)
For every ordered positive (z[a, i], z[b, i]), a ≠ b, with probability ε the right end is replaced by z[b, j], j ≠ i uniform (same view b); otherwise
kept.  The number of positives, the separate P / Q means and the negative set Q (every different-image pair; nothing removed, also when it equals
a contaminated positive) are unchanged, so positives follow P_ε = (1 − ε) P + ε Q and negatives Q.  For the controlled pair
η_ε = (1 − ε) η / (1 − ε η) and S(P_ε, Q) = (1 − ε)² E_M[η² / (1 − ε η)] (exact-summation unit test on a finite model); the trained objective is
still the original J — no claim of estimating the clean S, no post-hoc correction with the known ε.
SimCLR keeps its six-view-pair NT-Xent, logits, self-mask and denominators; for a replaced pair only the target index moves to (b, j); the original
positive stays in the denominator.  One replacement draw per step from a generator seeded by (run seed, step): identical replacement indices for
the three methods with the same seed, and identical after a resume.

## 2. Cells (6) and baselines
{VCS A-P3, JS-AP3, SimCLR} × {CIFAR-10, CIFAR-100}, ε = 0.10, seed 0, 800 epochs; config = the method's seed-0 parent (the P137 parents) with only
`pairing.stress_epsilon = 0.10` (+ run.stage) — `configs/make_p145_configs.py`, `configs/P145_SHA256.json`, `slurm/p145_lines.txt`.  VCS / JS keep the
common scorer (2, 0.5) and lr 1e-3; SimCLR its recipe; no change of augmentation, projector, K or momentum.  Clean baselines (ε = 0) = the parents:
A-P3 89.06 / 60.20, JS-AP3 88.72 / 58.76, SimCLR 88.20 / 58.66 (seed 0, development split).

## 3. Code
`src/vcs_ssl/objectives.py` (P145 block: `stress_replacements`, `stress_all_view_tokens_loss`, `stress_simclr_views`), `src/vcs_ssl/config.py`
(optional `pairing.stress_epsilon`, not filled when absent; `_p145_stress_policy`: 0 < ε < 1, 4 views, shared branch, joint step, no queue /
momentum; VCS / JS only with the fixed A-P3 scorer on `pair_scope: all_view_tokens`; SimCLR), `src/vcs_ssl/train.py` (per-step replacement draw;
diagnostics on logged steps).  Tests `tests/test_vcs_ssl_p145.py`: sampler; ε = 0 equals the frozen all-view loss (VCS, JS) and the frozen
multi-view SimCLR loss (values, gradients); with replacements equal to a pair-by-pair brute force; part gradients; the finite-model identity;
config acceptance / refusals; CPU trainer smokes for the three methods.  Gate `slurm/p145_gate.sbatch` (+ regression suites and config-hash
invariance of every existing config against a HEAD worktree).

## 4. Pre-stated reading (plan §6.3)
- Primary: Δ_m = A_m(ε) − A_m(0), clean frozen-h linear on the same development images; kNN and absolute accuracy alongside.  A smaller drop from a
  worse clean start is not called stronger.
- Process (step logs at the logging interval): T / f and cosine of kept vs replaced positives, the norms of the kept-positive, replaced-positive and
  negative gradient parts w.r.t. the projector outputs and the cosine of the replaced part with the total, encoder / projector gradient norms;
  failures / NaN / loss tails listed separately.
- **Trigger:** if |Δ_VCS − Δ_m| ≥ 0.50 (CIFAR-10) or ≥ 1.00 (CIFAR-100) for m ∈ {JS, SimCLR}, in either direction → freeze and add seeds 1–2 for
  **all three methods** on that dataset (addendum, submitted when the rule triggers).  At most one later ε = 0.20 condition, with separate budget
  confirmation.  If all methods are close or VCS is worse, the result stands; the contamination scheme is not changed to look for an advantage.

## 5. Not claimed
Tolerance at other ε, other contamination types (label noise, P72-style image–text mismatch), other backbones; clean-S estimation.

## Decisions at the freeze (main session)
- Gate job 1025667 (CPU): P145 tests 24 / 24, regression suites (P114, v2, P126, P133, P138, P104, core) 153 / 153, config-hash invariance:
  458 existing configs loaded under HEAD, 0 changed (`reports/P145_GATE_1025667/`).
- Timing: frozen when the experiment queue fell below the owner's floor of 15, per plan §12's batch order (PAIR + STRESS after HEAD).  The P138
  outcome does not enter STRESS (STRESS uses the full all-view-token parents, not the sampled-shift pairing).
- Replacement RNG: `torch.Generator` seeded with seed · 1 000 003 + 145 + step — the same draw for the three methods of a dataset at a given step
  (same seed, same batch size and views); it does not depend on the loader or pair generators, so evaluation and resume cannot shift it.
- Diagnostics are computed on logged steps only (logging interval), so they add no cost to the other steps.
- Launch: `slurm/p145_lines.txt` (normal QOS; RTX6000PRO / H100 / L40S; node51 / 52 / 60 excluded).
