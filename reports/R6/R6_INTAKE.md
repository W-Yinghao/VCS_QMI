# R6-00 — intake and closing the old lines — 2026-10-11 (queue queried 18:49 UTC)

- **Package:** `VCS_CVPR_V6_Server_r6_New_Applications_20261011.zip`, sha256 `1bd80788…5e87a`.  Copied without its PDF to `r6/package_r6_20261011/`.
- **Package base HEAD:** `ba3bc3e`; its delivery note checked `e6eee63`.  **Actual HEAD:** `e6eee63`, the same, so nothing new since the
  package was built.
- **Dirty tree:** only the r6 copy, a CPU wrapper, and untracked smoke / partial outputs (never committed).
- **Owner instruction:** "看看这个包，这是新的分析" (look at this package; it is the new analysis).

| item | actual state | source | action |
|---|---|---|---|
| R5-QN (VCS / JS with a query-normalized critic, 12 runs) | 8 training, 4 queued.  Every started run passed its step-0 gate.  Aggregation, rescoring and diagnostics are chained after the runs (1034387–1034390). | protocol `ba3bc3e`; owner chose the 12-run unit | run to completion under its frozen protocol; report as an r5 follow-up, separately from r6; **no further VL repair is queued after it** |
| R5-X1 (R4-VL1 replicated on RefCOCO, 9 runs) | 8 runs queued, gated on smoke 1034404 (waiting for a GPU); the last line is with the capped feeder | protocol `e6eee63`; owner standing authorization for free GPUs | keep (the package does not cancel it); complete and report; **no further RefCOCO round follows from it** |
| SSL / old CLIP-adapter R line | closed (owner 2026-10-09: no new SSL); nothing running | memory, R4-02 ledger | stays closed |
| VL official evaluation | not authorized.  Proposal draft `be585fc` (`reports/R5/final_evaluation_proposal.yaml`) waits for the owner's approval and non-inferiority margin. | r5 R5-40 | unchanged |
| new-task authorization and resources | Owner standing authorization (2026-10-11): useful experiments on free GPUs without asking.  The package reads this as one natural-data route at a time, not a five-route sweep (docs/08 §1); large downloads, account licences and new budgets still need the owner. | memory `autonomous-gpu-experiments`; package docs/08 | one route only, after R6-04; licences needing agreement go to the owner |
| existing data / caches | VL: proposals, score files and checkpoints for R4 / R5.  r6 candidate datasets: being audited (R6-03); none downloaded yet. | R6-03 | see `R6_DATA_BASELINE_MATRIX.csv` |
| package CPU tests | 41 / 41 pass (job 1034434, isolated `python -I`).  The reference-example script failed only because isolated mode keeps its own directory off `sys.path`; re-run with the reviewed `reference/` directory added explicitly (job 1034435). | `slurm/r6_package_check_cpu.sbatch` | done |

## Update conflicts
- **The package's VL figures are superseded.**  It quotes R4-VL1 / R5-30 (VCS 71.73 / 73.03 vs set softmax 73.58 / 76.25; D − E +0.27 /
  −0.22) and says "R5-QN only has a protocol and tests".  R5-QN is now training and R5-X1 is queued.  The figures stand; the newer runs will be
  reported as they finish.
- **Authorization conflict (precedence: owner's latest requirement > approved protocols > package).**  The owner's standing authorization
  allows autonomous submissions on free GPUs.  The package limits that to one natural route.  Followed: both are satisfied by running at most
  one natural route, chosen in R6-04 with its necessity written down.

## First-round status
| state | items |
|---|---|
| done | intake; package tests |
| running | R6-01 cards, R6-02 controlled checks, R6-03 data audit (CPU / literature); R5-QN and R5-X1 (VL wrap-up) |
| not run | R6-04 route choice and R6-05 pilot (depend on 01–03) |
| needs owner | any dataset licence requiring agreement; large downloads; official evaluation |
