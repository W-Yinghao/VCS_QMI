# Pre-registration — wave-2 C-S1: from-scratch batch sweep (P61_batchsweep), FROZEN 2026-09-27T14:58:13Z before launch (CPU test gate 1011093 passed: 29 + 38 tests)

**Flag.** These are from-scratch SSL-type runs on CIFAR-10.  The SSL stop order of 2026-09-26 is read as superseded for this unit only by the
owner's instruction of 2026-09-27 ("全部都补充实验，都有进行任务级precheck"), because the frozen-tower evidence of pre-check C (P54) left
from-scratch training as the only place a batch effect can exist.  Nothing else in the SSL programme is re-opened.

**Question (brief §2 C).** Does the VCS objective — positives and negatives averaged separately, negatives from K cyclic shifts — degrade
less than SimCLR (in-batch NT-Xent, 2B−2 negatives) and VICReg (batch-statistics terms) when the training batch shrinks from 256 to 32?

**Design.** CIFAR-10 frozen split `dev45k_val5k` (hash c35d7cd3…; no official test set), 2 views, 200 epochs, seed 0, B ∈ {32, 64, 128, 256}.
VCS = final 2-view recipe (`configs/cifar10_hpG_a5_learn_vcs_seed0.yaml`: cosine critic tanh(a⟨z1,z2⟩+b), a0 = 5, negative detach,
K = min(8, B − 1), AdamW, warm-up 10, cosine → 1 %, wd 1e-4, projector 512/128).  SimCLR / VICReg = the frozen P5 recipes
(`cifar10_confirm200_{simclr,vicreg}_seed0.yaml`; τ = 0.2; VICReg 25/25/1) — their batch-statistics / in-batch terms are their native form and
are not modified.  Learning rate: linear rule lr = 1e-3 × B/256 declared identically for all three methods (all share lr 1e-3 at B = 256 and
warm-up 10 epochs).  Equal-tuning check: lr × 2 at B = 32 for every method (the direction in which a too-small scaled lr would hurt).
Evaluation: frozen-h linear probe and kNN on the 5 000 selection images at epoch 200 (`evaluate --protocol pilot`), as every earlier run.
Retention(B) = accuracy(B) / accuracy(256) per method (linear primary, kNN secondary); at B = 32 the better of the two lr variants counts
for *every* method (same rule).

| run_id | method | B | lr | K | steps/epoch | est. h (H100/PRO6000) |
|---|---|---|---|---|---|---|
| P61_vcs_B32_seed0 | vcs_qmi | 32 | 1.25e-4 | 8 | 1406 | 3.5 |
| P61_vcs_B32_lr2x_seed0 | vcs_qmi | 32 | 2.5e-4 | 8 | 1406 | 3.5 |
| P61_vcs_B64_seed0 | vcs_qmi | 64 | 2.5e-4 | 8 | 703 | 2.0 |
| P61_vcs_B128_seed0 | vcs_qmi | 128 | 5e-4 | 8 | 351 | 1.3 |
| P61_vcs_B256_seed0 | vcs_qmi | 256 | 1e-3 | 8 | 175 | 1.0 |
| P61_simclr_B32_seed0 / _lr2x | simclr_matched | 32 | 1.25e-4 / 2.5e-4 | — | 1406 | 3.5 each |
| P61_simclr_B64 / B128 / B256_seed0 | simclr_matched | 64 / 128 / 256 | 2.5e-4 / 5e-4 / 1e-3 | — | 703 / 351 / 175 | 2.0 / 1.3 / 1.0 |
| P61_vicreg_B32_seed0 / _lr2x | vicreg_matched_128 | 32 | 1.25e-4 / 2.5e-4 | — | 1406 | 3.5 each |
| P61_vicreg_B64 / B128 / B256_seed0 | vicreg_matched_128 | 64 / 128 / 256 | 2.5e-4 / 5e-4 / 1e-3 | — | 703 / 351 / 175 | 2.0 / 1.3 / 1.0 |

15 runs, ≈ 33 GPU-h on H100/PRO6000 (≈ 2× on A100); no run exceeds the 23 h walltime (single `run_unit` jobs, no chains).  Budget table:
each method gets exactly 5 runs (4 batch sizes + 1 lr check); no method gets any other tuning.  Configs: `configs/cifar10_hpP_*_seed0.yaml`,
sha256 in `configs/HPARAM_P_SHA256.json` (generator `make_p_batch_sweep_configs.py`).  Single seed — marked as such in every table.

**Pre-committed reading (plan row C-S1, verbatim).** holds if VCS retention at B = 32 ≥ 0.98 while at least one control ≤ 0.95;
conditional if VCS ≥ 0.98 and all controls > 0.95 (no batch effect to be immune to, as on frozen towers) or VCS ∈ [0.95, 0.98) with a control
below it; does not hold if VCS < 0.95 or VCS degrades at least as much as every control.
QC sentinels: every run COMPLETED with finite loss; first-step gradient check passed; the B = 256 cells reproduce the known recipe numbers
within 1 point (VCS 2-view a0 = 5 ≈ the P26 value; SimCLR 86.1 ± 0.4; VICReg 85.5 ± 0.3) — otherwise the sweep is not read.
Not claimed: anything beyond B ∈ [32, 256], other datasets, more seeds, or the queue setting (C-T, separate prereg).
