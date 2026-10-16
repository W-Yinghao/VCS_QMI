# V9-N2 precheck: Spearman tie fix (review U05), 2026-10-16

**Defect.** `spearman` in `src/v9/n2/precheck.py` (lines 49-53 before the fix) ranked with `argsort(argsort(x))`: ordinal ranks, ties broken by row position, NaN only when the ordinal ranks are constant, which never happens for n > 1.  Constant weight vectors (Unweighted_A; every posterior / mass weight when rho-hat is 0) and heavily tied ones (PlainFilter 0/1, RankMass 0/1 with one fractional row) therefore received arbitrary values.

**Fix.** Tie-averaged ranks (`scipy.stats.rankdata(method='average')`, i.e. `scipy.stats.spearmanr`); when either vector is constant the value is `null` with `spearman_reason: "constant_input"` in the JSON pair table (never NaN) and `NA (constant)` in the markdown table.  The precheck pair table is the only caller; the collector (`scripts/v9/n2_precheck_collect.py`) now skips null values in its Spearman > 0.99 list and lists them separately (with the old collector a null value would raise a TypeError at `None > 0.99`).  NA below = null in the JSON (constant input).

**Reproduction.** For every cell, the weight vectors were rebuilt from the saved weight files in the precheck's own order: the role-U rows of `pairs.observed_pairs.json` in file order (`scores["U_pair_ids"]`, `pipeline.compute_scores`), each method's weights read by `weights.read_weight_lock(<method>.csv)` and stacked as `np.array([w[p] for p in U_pair_ids])` (`src/v9/n2/pipeline.py:110`).  The original function reproduces **168 / 168** recorded values bit-exactly (8 cells × 21 method pairs).  The other pair columns are bit-exact too.  The per-method table is bit-exact except ESS, the only field computed with `np.dot` (`sum(w)^2 / np.dot(w, w)`, a BLAS call), which differs by at most 7.41e-16 relative; the precheck ran on node58, this check on a CPU node (Intel(R) Xeon(R) CPU E5-2643 v4 @ 3.40GHz).  The new values agree with `scipy.stats.spearmanr` to within 1.1e-16.  Environment: Python 3.10.20, numpy 2.2.6, scipy 1.15.3 (the precheck's own v9_sgr env).

## Per cell

| cell | seed role | old values reproduced | now NA (constant) | changed by > 0.001 | largest change | near-monotone pairs (> 0.99): old → new |
|---|---|---|---|---|---|---|
| rho 0.4, seed 901 | formal | 21 / 21 | 6 | 9 | PlainFilter\|RuLSIF_Weight: 0.537 → 0.742 | 6 → 6 |
| rho 0, seed 901 | formal | 21 / 21 | 21 | 0 | none (every pair now NA) | 15 → 0 |
| rho 0.4, seed 902 | formal | 21 / 21 | 6 | 9 | BCE_Weight\|PlainFilter: 0.605 → 0.786 | 6 → 6 |
| rho 0, seed 902 | formal | 21 / 21 | 18 | 2 | Nystrom_Weight\|PlainFilter: 0.123 → 0.401 | 6 → 0 |
| rho 0.4, seed 903 | formal | 21 / 21 | 6 | 9 | PlainFilter\|VCS_Weight: 0.586 → 0.774 | 3 → 3 |
| rho 0, seed 903 | formal | 21 / 21 | 21 | 0 | none (every pair now NA) | 15 → 0 |
| rho 0.4, seed 917 | tuning | 21 / 21 | 6 | 9 | Nystrom_Weight\|PlainFilter: 0.595 → 0.779 | 3 → 3 |
| rho 0, seed 917 | tuning | 21 / 21 | 6 | 9 | PlainFilter\|RankMass: 0.914 → 0.390 | 3 → 3 |

Totals: 90 of 168 pair values are now NA (constant input); of the 78 defined values, 47 moved by more than 0.001 (every one of them involves PlainFilter or RankMass) and 31 by at most 1e-5.  Largest change: PlainFilter|RankMass at rho 0, seed 917, from 0.914 to 0.390.

## Old → new, rho 0.4 (3 decimals as in the precheck tables; full precision in `spearman_fix.json`)

| pair | seed 901 | seed 902 | seed 903 | seed 917 |
|---|---|---|---|---|
| BCE_Weight\|Nystrom_Weight | 1.000 → 1.000 | 1.000 → 1.000 | 1.000 → 1.000 | 1.000 → 1.000 |
| BCE_Weight\|PlainFilter | 0.537 → 0.742 | 0.605 → 0.786 | 0.586 → 0.774 | 0.595 → 0.779 |
| BCE_Weight\|RankMass | 0.709 → 0.844 | 0.710 → 0.846 | 0.698 → 0.839 | 0.718 → 0.850 |
| BCE_Weight\|RuLSIF_Weight | 0.998 → 0.998 | 0.996 → 0.996 | 0.987 → 0.987 | 0.988 → 0.988 |
| BCE_Weight\|Unweighted_A | -0.008 → NA | -0.009 → NA | -0.011 → NA | -0.008 → NA |
| BCE_Weight\|VCS_Weight | 1.000 → 1.000 | 1.000 → 1.000 | 1.000 → 1.000 | 1.000 → 1.000 |
| Nystrom_Weight\|PlainFilter | 0.537 → 0.742 | 0.605 → 0.786 | 0.586 → 0.774 | 0.595 → 0.779 |
| Nystrom_Weight\|RankMass | 0.709 → 0.844 | 0.710 → 0.846 | 0.698 → 0.839 | 0.718 → 0.850 |
| Nystrom_Weight\|RuLSIF_Weight | 0.998 → 0.998 | 0.996 → 0.996 | 0.987 → 0.987 | 0.988 → 0.988 |
| Nystrom_Weight\|Unweighted_A | -0.008 → NA | -0.008 → NA | -0.011 → NA | -0.008 → NA |
| Nystrom_Weight\|VCS_Weight | 1.000 → 1.000 | 1.000 → 1.000 | 1.000 → 1.000 | 1.000 → 1.000 |
| PlainFilter\|RankMass | 0.790 → 0.710 | 0.849 → 0.794 | 0.850 → 0.794 | 0.824 → 0.761 |
| PlainFilter\|RuLSIF_Weight | 0.537 → 0.742 | 0.605 → 0.786 | 0.586 → 0.774 | 0.595 → 0.779 |
| PlainFilter\|Unweighted_A | 0.641 → NA | 0.595 → NA | 0.606 → NA | 0.602 → NA |
| PlainFilter\|VCS_Weight | 0.537 → 0.742 | 0.605 → 0.786 | 0.586 → 0.774 | 0.595 → 0.779 |
| RankMass\|RuLSIF_Weight | 0.709 → 0.844 | 0.710 → 0.846 | 0.698 → 0.839 | 0.718 → 0.850 |
| RankMass\|Unweighted_A | 0.520 → NA | 0.520 → NA | 0.527 → NA | 0.515 → NA |
| RankMass\|VCS_Weight | 0.709 → 0.844 | 0.710 → 0.846 | 0.698 → 0.839 | 0.718 → 0.850 |
| RuLSIF_Weight\|Unweighted_A | -0.008 → NA | -0.008 → NA | -0.011 → NA | -0.007 → NA |
| RuLSIF_Weight\|VCS_Weight | 0.998 → 0.998 | 0.996 → 0.996 | 0.987 → 0.987 | 0.988 → 0.988 |
| Unweighted_A\|VCS_Weight | -0.008 → NA | -0.009 → NA | -0.011 → NA | -0.008 → NA |

## Old → new, rho 0 (3 decimals as in the precheck tables; full precision in `spearman_fix.json`)

| pair | seed 901 | seed 902 | seed 903 | seed 917 |
|---|---|---|---|---|
| BCE_Weight\|Nystrom_Weight | 1.000 → NA | -0.049 → NA | 1.000 → NA | 1.000 → 1.000 |
| BCE_Weight\|PlainFilter | 0.910 → NA | 0.885 → NA | 0.908 → NA | 0.119 → 0.383 |
| BCE_Weight\|RankMass | 1.000 → NA | 1.000 → NA | 1.000 → NA | -0.008 → 0.156 |
| BCE_Weight\|RuLSIF_Weight | 1.000 → NA | -0.048 → NA | 1.000 → NA | 0.948 → 0.948 |
| BCE_Weight\|Unweighted_A | 1.000 → NA | 1.000 → NA | 1.000 → NA | -0.034 → NA |
| BCE_Weight\|VCS_Weight | 1.000 → NA | 1.000 → NA | 1.000 → NA | 1.000 → 1.000 |
| Nystrom_Weight\|PlainFilter | 0.910 → NA | 0.123 → 0.401 | 0.908 → NA | 0.119 → 0.383 |
| Nystrom_Weight\|RankMass | 1.000 → NA | -0.049 → NA | 1.000 → NA | -0.008 → 0.156 |
| Nystrom_Weight\|RuLSIF_Weight | 1.000 → NA | 0.984 → 0.984 | 1.000 → NA | 0.948 → 0.948 |
| Nystrom_Weight\|Unweighted_A | 1.000 → NA | -0.049 → NA | 1.000 → NA | -0.034 → NA |
| Nystrom_Weight\|VCS_Weight | 1.000 → NA | -0.049 → NA | 1.000 → NA | 1.000 → 1.000 |
| PlainFilter\|RankMass | 0.910 → NA | 0.885 → NA | 0.908 → NA | 0.914 → 0.390 |
| PlainFilter\|RuLSIF_Weight | 0.910 → NA | 0.124 → 0.401 | 0.908 → NA | 0.119 → 0.383 |
| PlainFilter\|Unweighted_A | 0.910 → NA | 0.885 → NA | 0.908 → NA | 0.898 → NA |
| PlainFilter\|VCS_Weight | 0.910 → NA | 0.885 → NA | 0.908 → NA | 0.119 → 0.383 |
| RankMass\|RuLSIF_Weight | 1.000 → NA | -0.048 → NA | 1.000 → NA | -0.008 → 0.156 |
| RankMass\|Unweighted_A | 1.000 → NA | 1.000 → NA | 1.000 → NA | 0.983 → NA |
| RankMass\|VCS_Weight | 1.000 → NA | 1.000 → NA | 1.000 → NA | -0.008 → 0.156 |
| RuLSIF_Weight\|Unweighted_A | 1.000 → NA | -0.048 → NA | 1.000 → NA | -0.033 → NA |
| RuLSIF_Weight\|VCS_Weight | 1.000 → NA | -0.048 → NA | 1.000 → NA | 0.948 → 0.948 |
| Unweighted_A\|VCS_Weight | 1.000 → NA | 1.000 → NA | 1.000 → NA | -0.034 → NA |

## Which decisions could be affected

- **Gate: none.** The formal block is refused only on teacher AUROC: `degenerate = AUROC > 0.999 or < 0.55` (`src/v9/n2/precheck.py:65`, thresholds `src/v9/n2/precheck.py:24`), `formal_block_allowed = not degenerate` (`src/v9/n2/precheck.py:136`), read by `precheck_allows` (`src/v9/n2/pipeline.py:120`) and the unit planner (`scripts/v9/n2_plan_units.py:28`).  No Spearman is involved.
- **Merge rule: none.** Two methods are one run only if their loss coefficients agree to 1e-12 on every pool row (`src/v9/n2/precheck.py:92`, tolerance `src/v9/n2/precheck.py:25`); the groups are read by `unit_members` (`src/v9/n2/pipeline.py:130`) and the planner (`scripts/v9/n2_plan_units.py:30`).  No Spearman is involved.
- **Callers.** `spearman(` is called at one place only, the pair table of `weight_tables` (`src/v9/n2/precheck.py:78`).  `src/v9/n2/pipeline.py`, `src/v9/n2/weights.py`, `scripts/v9/n2_run.py`, `scripts/v9/n2_release.py` contain 0 matches of 'spearman'.  The value is read only by the precheck markdown table, the JSON dump, the collector's near-monotone list and one test assertion (`tests/v9/test_n2_ext.py:38`, continuous data without ties).
- **Records.** In 8 of 8 cells, the regenerated precheck JSON equals the record except for the Spearman fields.  So `formal_block_allowed`, `refusal_reason` and `loss_coefficients` (groups, merged) are unchanged, and no gate verdict, merge group, unit plan or run changes (the tuning seed 917 included).
- **Descriptive outputs that change.** (1) The Spearman column of the pair table in all 16 `precheck_f30k_rho*_s*.{json,md}`.  (2) The near-monotone (Spearman > 0.99) lines of `PRECHECK_SUMMARY.md` for rho 0 seed 901, rho 0 seed 902, rho 0 seed 903: the constant pairs listed there as `rho_s 1.0 -> MERGED` now appear as `Spearman undefined (NA)`; the merges themselves are the same.  3 listed values change after the collector's rounding to 6 decimals at rho 0.4 seed 902 (BCE_Weight\|Nystrom_Weight 0.99999 → 0.999995; Nystrom_Weight\|RuLSIF_Weight 0.996414 → 0.996419; Nystrom_Weight\|VCS_Weight 0.99999 → 0.999995).  Every cell gains a `Spearman undefined (NA)` line (6, 21, 6, 18, 6, 21 pairs in the 6 cells).  All other gate-summary fields are unchanged.
- **Text claims re-checked.** "VCS vs BCE rank-identical" (`reports/V9/V9_N2_PROTOCOL_FROZEN_20261015.yaml:107`, `reports/V9/V9_N2_ADAPTER_NOTES.md:123`, `reports/V9/V9_N2_precheck/PRECHECK_SUMMARY.md:105`) still holds: their tie-averaged rank vectors are identical in every cell where they are not constant (rho 0.4 seed 901, rho 0.4 seed 902, rho 0.4 seed 903, rho 0.4 seed 917, rho 0 seed 917); at rho 0 seeds 901–903 both are constant, identical and merged.  PlainFilter is reported from the precheck weight tables only (`reports/V9/V9_N2_PROTOCOL_FROZEN_20261015.yaml:117`), so any later description of PlainFilter against other methods should use the regenerated values.

## Tests

- fixed code, `tests/v9/test_n2.py test_n2_ext.py test_n2_release.py test_precheck_spearman.py`: 1 failed, 43 passed, 1 warning in 130.51s (0:02:10); failed: `test_host_lock_pending_markers_block_only_recipe_fields`
- original code, `tests/v9/test_n2.py test_n2_ext.py test_n2_release.py`: 1 failed, 37 passed, 1 warning in 106.79s (0:01:46); failed: `test_host_lock_pending_markers_block_only_recipe_fields`
- fixed code, `tests/v9/test_n2_ext.py test_n2_release.py test_precheck_spearman.py`: 1 failed, 28 passed, 1 warning in 24.23s; failed: `test_host_lock_pending_markers_block_only_recipe_fields`; the files that touch precheck
- original code, `tests/v9/test_n2_ext.py test_n2_release.py`: 1 failed, 22 passed, 1 warning in 12.57s; failed: `test_host_lock_pending_markers_block_only_recipe_fields`
- fixed code, `tests/v9/test_precheck_spearman.py`: 6 passed in 2.34s
- fixed code, `tests/v9/test_precheck_spearman.py` (CS_QMI env: Python 3.12.12, numpy 2.5.2, scipy 1.18.1): 6 passed in 2.80s
- original code, `tests/v9/test_precheck_spearman.py`: 5 failed, 1 passed in 2.13s; does the new test catch the defect? yes; the passing one is the no-tie case, where old and new agree

The one failure is pre-existing and unrelated: test_host_lock_pending_markers_block_only_recipe_fields asserts a PENDING note in reports/V9/HOST_LOCK.yaml; HOST_LOCK no longer has one since the measured costs were filled in (commits 84fb6eb, 103d8f8), so cfg.lock_notes == []; it fails identically on the original code and does not involve precheck.  Main environment: v9_sgr: Python 3.10.20, numpy 2.2.6, scipy 1.15.3, torch 2.9.0+cu128, pytest 9.1.1 (CPU, nodecpu10).

## Files

- `spearman_fix.json`: per cell and per method pair, the recorded value, the value reproduced by the original function (with an exact-match flag), the new value or null with its reason, the change, scipy's value, tie statistics per method and weight-file hashes, plus the decision evidence above.
- `regenerated/precheck_f30k_rho*_s*.{json,md}`: the 8 precheck tables rebuilt with the fixed code from the saved weights (every non-Spearman block is copied from the record).  `regenerated/PRECHECK_SUMMARY.{json,md}`: the gate summary rebuilt from them by the fixed collector; its job / commit / host fields say so.
- The records in `reports/V9/V9_N2_precheck/` (precheck tables and `PRECHECK_SUMMARY.*`, the gate record of job 1042133) are left unchanged.
