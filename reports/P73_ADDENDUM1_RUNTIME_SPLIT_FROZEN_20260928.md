# P73 addendum 1 — runtime split of the T1 grid (frozen 2026-09-28, before any split job runs; no design change)

**Why.**  The pre-registration estimated ~16 GPU-h in total (5–8 s per repeat).  Measured cost is ~19 s per repeat on RTX6000PRO and
~48 s on the (shared, CPU-contended) A100 nodes, roughly independent of n: the runner is single-core CPU-bound (GPU utilisation 32–36 %,
one python process at 100 % CPU).  The colour jobs hold 5 600 repeats each (3 strengths × 4 n × 100 + 2 nulls × (2 × 100 + 2 × 1000)),
i.e. ~30 h (VCS, RTX6000PRO) and ~75 h (SimCLR, A100), past the 23 h wall; the runner writes a partial JSON after every (case, n) cell
but cannot resume.  The SimCLR blur job (1 600 repeats, ~21 h) is at risk as well.

**What changes (execution only).**  The grid, tests, R, permutations, steps, δ, sizes and read-out rules of P73 are unchanged.
1. The four running jobs (1013022–25) keep the cells they finish.  The colour jobs and the SimCLR blur job are cancelled once all their
   *planted* (power) cells are in their `.partial.json`; the VCS blur job (~8.5 h) runs to completion.
2. The colour null cases (`colour_null_label_only`, `colour_null_all_planted0.2`) for both encoders run as separate jobs through the
   existing `--only` / `--sizes` flags: per (encoder, null) one job at n = 500 (R = 1000), one at n = 2000 (R = 1000), one at
   n ∈ {200, 1000} (R = 100) — 12 jobs.  The SimCLR blur nulls (`blur_null_label_only`, `blur_null_all_planted0.5`, R = 100 at all n)
   run as 2 separate jobs.
3. Each split job gets its own `--seed` (101–114, listed in `reports/job_ids.json`) so the null draws of different jobs are not
   identical streams; per-repeat draws are therefore not the draws the monolithic job would have made (statistically equivalent: every
   repeat is an independent draw of N | Y and the data subsample).  Null cells that the monolithic jobs happen to finish before
   cancellation are **not** read (the split jobs are the pre-declared source for every null cell), so no cell is chosen after seeing it.
4. The P74 report merges: power cells from the monolithic partial/final JSONs, null cells from the split-job JSONs.
