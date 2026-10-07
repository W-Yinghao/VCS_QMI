# Pre-registration — P152 (r3 R3-E1): the missing stability fields on the registered Gaussian ladder — DRAFT 2026-10-08

Trigger: the P151 field matrix — the P85 / P86 / P108 records hold oracle truths, bootstraps, SELECT curves and costs, but **no** critic weights,
per-sample T, per-update gradient norms, output histograms or repeated independent EVAL batches; [O1] §12.2 asks for gradient-norm variance,
critic-output histograms and failure rates against MINE / DV, NWJ, InfoNCE and SMILE.  r3 §6: at most 54 missing condition × method × seed units.

## 1. Units (54) — `scripts/p152_e1_stability.py`
Gaussian d 20 / 20, N_fit 4 096, B 256, dependence I ∈ {2, 6, 10} (S ≈ 0.567 / 0.913 / 0.985), seeds 0–2; methods VCS, matched JS, InfoNCE, NWJ, DV
(MINE, EMA denominator as registered), corrected SMILE (clipped-DV value, JS gradient), all with **in-batch** negatives (the registered multi-negative
implementations; the single-negative / product DV rows stay in P85 as the historical stress rows and are not re-run).  Learning rate: the one P85
selected for that (cell, method) — no new tuning.  Data streams (`build_roles`), torch seeds, batch draws and losses mirror `benchmark.train_neural`
step for step; the frozen benchmark module is not modified.

## 2. Recorded per unit
Per update: loss, critic-parameter gradient L2 norm, non-finite flag.  Checkpoints: updates 250 / 1 000 / 2 000 and the SELECT-best state (P85 rule);
state_dicts saved.  Per checkpoint: native value on the full EVAL (as P85); 64 independent EVAL blocks of 512 anchors → `eval_var_conditional_fit`;
critic-output histograms (f: 201 bins on [−10, 10] + overflow counts; T: 101 bins) and |T| > 0.95 / 0.99 fractions; for VCS / JS the posterior readouts
(common T, oracle η) and per-sample T on EVAL (float16, saved).  Reproduction: the U = 2 000 SELECT-best value vs the P85 stored value (descriptive;
CPU vs GPU arithmetic).

## 3. Pre-stated reading
E-Stability panel completed with: gradient-norm mean / sd / CV / quantiles (last half of training), eval variance conditional on the fit, loss
tails, non-finite steps, histogram saturation — per method and I.  Cross-target comparison only in normalised form (CV, ordering), with the absolute
quantities shown; near-zero mean gradients are labelled uninformative.  Same-target precision stays the P151 panel.  **No success criterion for VCS**: if the
in-batch MI methods show equal or smaller gradient variance, that is the result.

## 4. Cost
Per unit ≈ 2 000 in-batch updates (256 × 256 pair matrix through a 40 → 256 → 256 → 1 MLP) + 4 checkpoint evaluations; measured in the gate
(`slurm/p152_gate.sbatch`, one full unit for VCS + JS).  Expected 5–10 min per method-unit on 16 CPU cores → ≈ 5–9 h over three CPU jobs (one per I).
