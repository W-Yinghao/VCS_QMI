# Pre-registration — wave-2 C-T: streaming / asymmetric negatives with a feature queue (P63_queue), FROZEN 2026-09-27T14:58:13Z before launch (CPU test gate 1011093 passed; exact queue-vs-cyclic equality and exact resume tested)

**Flag.** From-scratch SSL-type runs; authorised for this unit only by the owner's instruction of 2026-09-27 (see the P61 draft).

**Task family (brief appendix A, row C).** Memory bank & streaming / asymmetric sample sizes: the positives come from a small current batch,
the negatives (the product-of-marginals sample) from a queue of features of previous steps.  The VCS objective averages positives and negatives
separately, so a 16-image batch can be paired with 4096 stale negatives without changing the objective's form; the InfoNCE competitor gets
the same queue as MoCo-style negatives.

**Implementation (named variant, `pairing.negative_source: queue`, `pairing.queue: true`, `pairing.queue_size: 4096`).**  A FIFO ring of
detached critic-input features (VCS: the critic's input view features; SimCLR: the L2-normalised projector outputs) of *both* views of every
previous batch, updated after each optimizer step, kept in the checkpoint (exact resume tested).  VCS: positives (z1_i, z2_i) as in the
reference; for each anchor z1_i, K = 8 partners drawn uniformly without replacement from the queue with the dedicated CPU pairing generator;
partners are detached (the negative-detach semantics of the final recipe); the separate averaging of the reference is unchanged.  SimCLR:
for each of the 2B anchors the logits are [sim(anchor, other view of the same image), sim(anchor, queue_1..4096)] / τ, target 0 (MoCo-style,
no momentum encoder — identical encoder treatment for both methods).  While the queue holds fewer than K entries (the first step only) the
frozen in-batch negatives are used and the step is counted (`queue_fallback_steps`).  VICReg has no negatives and is not part of this unit.
Code: `src/vcs_ssl/objectives.py` (`NegativeQueue`, `vcs_pair_loss_queue`, `simclr_nt_xent_queue`), `config.py`, `train.py`; tests
`test_queue_negatives_*` in `tests/test_integration.py` (config locks, FIFO ring, exact equality with the cyclic-shift loss when the queue holds
the other view's batch, MoCo-style logits against a manual computation, trainer smoke + exact resume for both methods).

**Design.** CIFAR-10 frozen split, 2 views, 200 epochs, seed 0, B ∈ {16, 32}, queue 4096, lr = 1e-3 × B/256 (the P61 rule), everything else
as the P61 bases.  Reference cells: the cyclic / in-batch B = 256 runs of P61 (`P61_vcs_B256_seed0`, `P61_simclr_B256_seed0`).

| run_id | method | B | lr | K | queue | steps/epoch | est. h (H100/PRO6000) |
|---|---|---|---|---|---|---|---|
| P63_vcs_B16_queue4096_seed0 | vcs_qmi | 16 | 6.25e-5 | 8 | 4096 | 2812 | 6.5 |
| P63_vcs_B32_queue4096_seed0 | vcs_qmi | 32 | 1.25e-4 | 8 | 4096 | 1406 | 3.5 |
| P63_simclr_B16_queue4096_seed0 | simclr_matched | 16 | 6.25e-5 | — | 4096 | 2812 | 6.5 |
| P63_simclr_B32_queue4096_seed0 | simclr_matched | 32 | 1.25e-4 | — | 4096 | 1406 | 3.5 |

4 runs, ≈ 20 GPU-h; none exceeds 23 h (single `run_unit` jobs).  Budget table: both methods get exactly the same two runs and the same lr rule;
no other tuning.  Configs `configs/cifar10_hpQ_*_seed0.yaml`, sha256 in `configs/HPARAM_Q_SHA256.json` (generator `make_q_queue_configs.py`).

**Pre-committed reading (plan row C-T, verbatim).** holds if VCS-with-queue reaches ≥ 0.95 of the VCS B = 256 result at B = 16 while
InfoNCE-with-queue reaches < 0.90 of its own B = 256; does not hold if VCS < 0.90 or degrades as much as InfoNCE.  (Between the two:
conditional, stated with the numbers.)  Primary metric: frozen-h linear probe on the selection images at epoch 200; kNN secondary.
QC sentinels: COMPLETED runs, finite losses, `queue_fallback_steps` = 1, no collapse flag; the queue configs' B = 32 cells are also compared
with the cyclic B = 32 cells of P61 (same lr) to separate the queue effect from the small-batch effect.
Not claimed: momentum encoders (neither method has one here), queue sizes other than 4096, more seeds, other datasets.
