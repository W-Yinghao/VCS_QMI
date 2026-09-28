# Pre-registration DRAFT — P67: the one-shot official CIFAR-10 test-set evaluation (to be frozen by the main session; NOT run) — FROZEN 2026-09-28T13:36:24Z; all 52 checkpoints present; no tuning or selection follows this evaluation; the unit list is slurm/final_official_test_units.txt at this commit

**Owner's rule.** The official test partition is never used during development; it is evaluated **once, at the very end**, for the frozen best of
each method, and nothing is tuned or selected afterwards.  Every decision in this project — recipe, checkpoints, control winners — was made on the
5 000-image *selection* split of the frozen `dev45k_val5k` manifest; those selection numbers are the ones reported next to the test numbers here.

**Precondition.** All training that feeds the table below is finished: the VCS runs (P35 / P37 / P39 / P43), the frozen P5 controls and the P41 control
tuning including the winners' seeds 1–2 (`P41_vicreg_800ep_seed2` was still at epoch 200 / 800 when this draft was written; the evaluation waits for it).

## Protocol (identical to `pilot` except for the scored images)
- Code: `vcs_ssl.evaluate --protocol final_official_test` (`evaluate_final_official_test`): the run's frozen resolved config and manifest, the named
  checkpoint (sha256 recorded), a fresh eval-mode model copy; **linear head** = the frozen pilot probe (same hyper-parameters, trained on the 45 000-image
  fit split's clean-transform `h`); **kNN** = pilot settings with the 45k fit split as bank; **clean transform** = the run's own.  Scored images: the
  10 000 official test images (`test_batch`, md5 `40351d587109b95175f43aff81a1287e`; sha256 recorded at load time).  Nothing is trained; the encoder
  state is hash-checked before and after.
- Access control: the partition opens only through `load_cifar10_train(root, train=False, allow_official_test=True)` **and** `VCS_FINAL_ROUND=1`
  in the environment; only `slurm/final_official_test.sbatch` (non-stand-in mode) exports that variable.  A result file that already exists is never
  overwritten (the protocol raises unless `--force`, which this prereg forbids).
- Output per (run, checkpoint): `<run>/evaluations/final_official_test_<tag>.json`; aggregate `reports/P68_FINAL_OFFICIAL_TEST.{json,md}` by
  `scripts/final_official_test_eval.py --units slurm/final_official_test_units.txt`.
- Stand-in smoke (allowed before freezing, disclosed below): the same code path on the selection split (`--standin`), which never opens the test batch.

## Units (46 checkpoints; `slurm/final_official_test_units.txt`; all present on disk except where noted)
| method | compute point | cell | runs (checkpoint) | seeds | selection linear % (mean ± sd) |
|---|---|---|---|---|---|
| VCS | 1× | 4 views, B 128, 100 ep | `P39_vcs_a5_views4_b128_100ep_seed{0,1,2}` (epoch_100) | 3 | 83.19 ± 0.40 |
| VCS | 2× | 4 views, B 256, 200 ep | `P35_vcs_a5_views4_seed{0,1,2}` (epoch_200) | 3 | 84.54 ± 0.16 |
| VCS | 4× | 8 views, 200 ep (best 3-seed 4× cell) | `P37_vcs_a5_views8_200ep_seed{0,1,2}` (epoch_200) | 3 | 85.95 ± 0.35 |
| VCS | 4× | 2 views, 800 ep | `P35_vcs_a5_800ep_seed{0,1,2}` (epoch_800) | 3 | 85.30 ± 0.21 |
| VCS | 4× | 4 views, 400 ep — single seed | `P35_vcs_a5_views4_400ep_seed0` (epoch_400) | 1 | 85.90 |
| VCS | 4× | 16 views, 200 ep — single seed | `P43_vcs_a5_views16_200ep_seed0` (epoch_200) | 1 | 86.74 |
| VCS | 8× | **4 views, 800 ep — the frozen best recipe** | `P35_vcs_a5_views4_800ep_seed{0,1,2}` (epoch_800) | 3 | 87.01 ± 0.53 |
| VCS | 8× | 8 views, 400 ep — single | `P43_vcs_a5_views8_400ep_seed0` (epoch_400) | 1 | 86.76 |
| VCS | 8× | 4 views, B 128, 800 ep — single | `P43_vcs_a5_views4_b128_800ep_seed0` (epoch_800) | 1 | 86.70 |
| VCS | 8× | 4 views, 800 ep, strong aug — single | `P43_vcs_a5_views4_800ep_augstrong_seed0` (epoch_800) | 1 | 87.78 |
| VCS | 16× | 4 views, 1600 ep — single | `P43_vcs_a5_views4_1600ep_seed0` (epoch_1600) | 1 | 87.50 |
| VCS | 16× | 8 views, 800 ep — single | `P43_vcs_a5_views8_800ep_seed0` (epoch_800) | 1 | 87.14 |
| SimCLR | 1× | frozen P5 recipe (2 views, 200 ep) | `P5_simclr_seed{0,1,2}` (epoch_200) | 3 | 86.09 ± 0.38 |
| VICReg | 1× | frozen P5 recipe | `P5_vicreg_seed{0,1,2}` (epoch_200) | 3 | 85.46 ± 0.26 |
| SimCLR | 1× | tuned winner: 4 views, B 128, 100 ep | `P41_simclr_views4_b128_100ep_seed{0,1,2}` (epoch_100) | 3 | 86.42 (seed 0; 3-seed mean from `P41_control_tuning_table.md`) |
| SimCLR | 2× | tuned winner: 4 views, 200 ep | `P41_simclr_views4_seed{0,1,2}` (epoch_200) | 3 | 87.48 (seed 0) |
| SimCLR | 4× | tuned winner: 2 views, 800 ep | `P41_simclr_800ep_seed{0,1,2}` (epoch_800) | 3 | 88.38 (seed 0) |
| VICReg | 1× | tuned winner: 4 views, B 128, 100 ep | `P41_vicreg_views4_b128_100ep_seed{0,1,2}` (epoch_100) | 3 | 87.24 (seed 0) |
| VICReg | 2× | tuned winner: 4 views, 200 ep | `P41_vicreg_views4_seed{0,1,2}` (epoch_200) | 3 | 86.64 (seed 0) |
| VICReg | 4× | tuned winner: 2 views, 800 ep | `P41_vicreg_800ep_seed{0,1,2}` (epoch_800) | 3 (seed 2 pending at draft time) | 86.70 (seed 0) |
No equal-compute tuned control exists for the 8× and 16× VCS points (a SimCLR 4v/800ep unit is *proposed*, not run); those VCS rows are reported
against the 4× controls with the compute difference stated.

## What is reported (pre-committed)
1. For each cell: official-test linear-probe top-1 and kNN top-1, mean ± sd over seeds, next to the selection-split numbers; single-seed rows marked.
2. The paper's comparison rows: at 1×, 2× and 4×, VCS (its 3-seed cell) vs the tuned SimCLR and VICReg winners (3 seeds each) and vs the frozen P5
   controls; at 8× and 16× the VCS numbers alone, with the note above.  The selection → test shift of every cell is reported (a sanity check on the
   5 000-image split, not a result to optimise).
3. Rule: **no tuning, no re-selection, no re-run follows this evaluation**; a run's official number is whatever its frozen checkpoint gives.  If a
   checkpoint were missing or corrupt the cell is reported as missing, not replaced.
4. Not claimed: anything about other datasets or backbones; the test numbers are used only in the final tables of the paper.

## Provenance / QC
Test batch md5 verified against the official value; sha256 recorded per evaluation; checkpoint sha256 recorded; encoder-state hash unchanged;
split hash `dev45k_val5k` recorded; `VCS_FINAL_ROUND` value recorded in the aggregate JSON; `git` commit recorded by the job banner.

## Compute
46 evaluations on one GPU: fit-feature extraction ≈ 20–40 s each on H100 (cache hits where the pilot cache survived cleanup), test extraction ≈ 5 s,
probe ≈ 30 s → ≈ 1 h total.  Launch (only after freezing this document and after `P41_vicreg_800ep_seed2` has finished):
`UNITS=slurm/final_official_test_units.txt REPORT=reports/P68_FINAL_OFFICIAL_TEST sbatch -p RTX6000PRO,H100,A100 -J final_official_test slurm/final_official_test.sbatch`

## Smoke (stand-in, disclosed) — filled in after the CPU job
CPU job 1012292 (`slurm_logs/final_official_test_p68_standin_smoke_1012292.out`, exit 0, 58 s with a fit-feature cache hit): the final protocol's stand-in
mode on `P39_vcs_a5_views4_b128_100ep_seed0` / epoch_100 reproduces the pilot selection numbers exactly — stand-in linear 83.52 % / kNN 78.30 % vs the
run's `evaluation_epoch_100.json` 83.52 % / 78.30 % — i.e. the new code path is the pilot protocol with only the scored images swapped.  Output
`reports/P68_FINAL_OFFICIAL_TEST_standin.{json,md}` (stand-in; not a test number).  The test partition was not opened (the job did not export
`VCS_FINAL_ROUND`; `final_official_test_*.json` files exist nowhere).  Unit tests: `test_11_official_test_double_unlock` (flag alone and environment
alone both refused; both together open the gate) and `test_12_final_protocol_standin_and_refusals` (stand-in runs on a synthetic smoke run; the
official path refuses without the unlock; an existing official result is never overwritten without `--force`).

**Freezing note.** The 8× controls (SimCLR / VICReg 4v × 800ep, 3 seeds each) are included; VICReg 8× on the selection split = 87.11 ± 0.51 / 84.46 ± 0.08.  Launch = one job, 52 evaluations, then `P68_FINAL_OFFICIAL_TEST.{json,md}` and a results-only commit before any interpretation.
