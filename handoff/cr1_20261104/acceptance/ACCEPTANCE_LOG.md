Publication copy (sample names aliased; numbers and conclusions unchanged).

# CR1 acceptance attempts
- 1087762 (CPU): package `scripts/run_checks.sh` 31/31 unit tests OK + check_package PASS; `synthetic_smoke.py` all_done (toy only).
- 1087774 (GPU H100 node52): cancelled by me ~1 min after start (output path moved from home to /projects). Its log shows
  check A FAIL caused by the acceptance script comparing CR1 G (pair_chunk 4096) to NR1 G at NR1's default pair_chunk 256
  (max |Δ| 1.19e-7); B, C PASS. Test-design bug, not a code defect. Log kept: logs/cr1_accept_1087774.out.
- 1087776: cancelled by me before any check (same test-design bug).
- 1087778 (GPU H100 nodesumo01): ALL_PASS 9/9 (A–I) -> ACCEPTANCE_1087778.json; outputs in
  /projects/.../outputs_cr1/CR1_20261104/acceptance/acceptance_1087778.
