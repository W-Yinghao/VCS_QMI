# Supersession note (R4-01, 2026-10-11)

The files in this directory are kept unchanged.  Parts of them read native fields where the common bounded J was meant:
- `e_precision.json` `same_target.*.J_eval` and `J_minus_S`: filled from `native.value` / `native.signed_error`.  Wrong for the JS families (a JS-type native
  target, off by up to 0.36 / 0.28) and for `s_kde:common_risk` (the plug-in S, off by up to 0.13).  The VCS cyclic / in-batch shift is ≤ 0.002.
  `posterior_mse` is correct (read from `posterior.*`).
- `e_stability.json` (seed sd / evaluation bootstrap SE of "the value") and `e_resolution.json` (ordering probabilities from `native.boot_values`): native
  readouts.  The report's §2 numbers for JS are on the JS native scale.
- `axes_N_batch.json` `abs_error`: native.

Corrected, schema-versioned aggregation: `reports/R4/P151_r4/` (`scripts/p151_e0_aggregate_r4.py`, regression test `tests/test_p151_r4.py`; old-vs-new diff in
`diff_r3_r4.json`).  Common-J bootstrap draws were never saved, so the common-readout ordering probabilities are null with a reason.  The report's §1
precision table (posterior MSE) and its §6 reading "JS ≥ VCS on S" are unaffected.
