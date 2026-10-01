# Pre-registration — P105: T1 critic-class ablation (does VCS's conditional-test advantage come from the objective or from the closed-form critic?), 2026-10-01 — FROZEN 2026-10-01T13:25:26Z before GPU compute (CPU smoke 1016884 passed)

Owner go 2026-10-01 ("好的，第一项也做，写预注册后提交").  Source: the caveat in `P74_CONDITIONAL_TEST_T1_REPORT_20261001.md` §6.  Code:
`scripts/cond_test_t1_ablation.py` (imports T1's sampler, fitters and closed-form critic unchanged), `slurm/cond_test_t1_ablation.sbatch`.

## Question
In T1 the VCS permutation test VAL-picked among three critics — linear and MLP (300 AdamW steps) and the **closed-form** linear critic (exact ridge
solution of the VCS objective in the class φ(z, n) = [z(2n−1), 1], picked in 73–82 % of repeats) — while the JS control had only the two SGD critics.
Is VCS's lead over JS (0.12–0.25 in 7 of 9 unsaturated informative cells) due to the **objective**, or to the **exactly solved linear critic**?

## Design
- Construction identical to T1 (fresh N | Y per repeat, disjoint FIT / EVAL / POOL of n, within-class POOL negatives, B = 200 shared within-class
  permutations, δ = 0.05, 300 steps, 80 / 20 split).  P45 feature files as T1.  New seeds (201–207); draws are independent of T1's, so every contrast
  below is **paired within this run** (all tests on identical draws), not against T1's numbers.
- Critics per repeat: vcs_lin, vcs_mlp, vcs_closed (T1's code); js_lin, js_mlp (T1's code); **js_exact (new)** — the JS / balanced-logistic objective
  solved exactly (convex; L-BFGS to convergence) in the same class φ, ridge on the closed form's grid {1e-3, 1e-2, 1e-1, 1}·sc chosen on VAL JS.
  Each critic gets its own permutation test (forced); plus VAL-picked combinations: **vcs3** = pick{lin, mlp, closed} (T1's vcs_perm), **vcs2** =
  pick{lin, mlp}, **js2** = pick{lin, mlp} (T1's js_perm), **js3** = pick{lin, mlp, exact}.
- Cells: the 9 **unsaturated informative** cells of P74 (fixed list), R = 100 each — VCS-encoder colour s 0.05 @ n 1000 / 2000, s 0.1 @ 500, s 0.2 @ 200;
  SimCLR colour s 0.2 @ 2000; VCS-encoder blur σ 0.5 @ 200 / 500; SimCLR blur σ 0.5 @ 1000 / 2000.  The 7 saturated cells (every T1 test ≥ 0.97) cannot
  discriminate and are not run.  Level: both colour nulls (`null_label_only`, `null_all_planted0.2`) at n = 2000, R = 200, both encoders.

## Pre-stated reading (per-cell power differences, paired; threshold 0.05 as in T1)
Contrasts: d2 = vcs2 − js2 (objective, SGD-fitted classes); d3 = vcs3 − js3 (objective, each with its exact linear solution available);
dc = vcs3 − vcs2 (what the closed form adds to VCS); forced pairs vcs_closed − js_exact, vcs_lin − js_lin, vcs_mlp − js_mlp (reported).
- **"VCS objective"** if d2 ≥ +0.05 in ≥ 6 of 9 cells **and** d3 ≥ +0.05 in ≥ 6 of 9.
- **"Critic class / exact solver"** if |d2| ≤ 0.05 in ≥ 6 of 9 **and** |d3| ≤ 0.05 in ≥ 6 of 9 **and** mean dc ≥ +0.05.
- **"Mixed"** otherwise, reported with which contrast carries the difference (e.g. d3 > 0 but d2 ≈ 0 → the advantage needs the exact solution and
  survives a matched exact JS; d2 > 0 but d3 ≈ 0 → an exact JS solution closes the gap).
- Replication check (reported): vcs3 − js2 per cell, the T1 comparison, on the new draws.
- Level: every test ≤ 0.09 at R = 200 (nominal 0.05, binomial 95 % half-width ≈ 0.03); a test above 0.09 is flagged and its power cells are not read.
- Not claimed: anything beyond these 9 cells and two encoders; nothing about the guaranteed (Hoeffding) test.

## Smoke (disclosed)
CPU job 1016884 (VCS encoder, colour, s 0.2 and `null_label_only` at n 200, R 3, 20 permutations, 60 steps): all 6 critics and 4 picked tests ran;
s 0.2 rejections vcs_lin / mlp / closed 0.67 / 0.67 / 1.00, js_lin / mlp / exact 0.00 / 0.67 / 1.00; null 0 except js_mlp 1 of 3.  ≈ 17 s per repeat
on 8 CPU threads.  Code-path evidence only; nothing changed after it.

## Compute
1 700 repeats (900 power + 800 level), ≈ 15–20 s each on a GPU node (CPU-bound like T1) → ≈ 8 GPU-h in 7 jobs (≤ 3.5 h each), normal QOS,
`slurm/p105_lines.txt`.  Results `reports/P105_t1_ablation_<job>.json` → merged table + report P106.
