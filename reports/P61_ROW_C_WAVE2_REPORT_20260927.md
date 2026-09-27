# Row C, wave 2 — report (P61 / P63): from-scratch batch sweep (C-S1) and queue negatives (C-T)

Preregs: `P61_BATCH_SWEEP_PREREG_FROZEN_20260927.md` (+ addendum 1: seeds 1–2, running), `P63_QUEUE_PREREG_FROZEN_20260927.md`.  Tables (results-only
commit c4db5dc): `P61_batchsweep_table.md`, `P63_queue_table.md` (repository summarizer; linear probe = frozen-h linear head on the 5 000-image
selection split, kNN k = 200).  Evidence category: completed for seed 0 (15 + 4 runs, all `COMPLETED`, 200 / 200 epochs); the 3-seed reading
of C-S1 is pending its 24 seed runs.  These are from-scratch SSL-type runs authorised for these two units only.

## 1. C-S1 — batch sweep, 2 views, 200 epochs, lr = frozen lr × B/256, K = 8, seed 0 (linear-val % / kNN-val %)
| method | B = 32 | B = 32, lr × 2 | B = 64 | B = 128 | B = 256 (reference) | retention at B = 32 (best lr) — linear | kNN |
|---|---|---|---|---|---|---|---|
| VCS (cosine critic, detach, a0 = 5) | 79.42 / 74.60 | 80.26 / 74.36 | 80.96 / 74.96 | 81.06 / 76.78 | 81.82 / 76.84 | **0.981** | 0.971 |
| SimCLR (τ = 0.2) | 84.64 / 79.64 | 84.48 / 80.70 | 85.12 / 80.82 | 85.24 / 82.98 | 86.32 / 83.80 | **0.981** | 0.963 |
| VICReg (25 / 25 / 1) | 85.06 / 80.86 | 85.68 / 81.70 | 85.90 / 81.86 | 86.14 / 82.60 | 85.12 / 81.86 | **1.007** | 0.998 |

Held-out J of the VCS runs: 0.931 (B 32) → 0.941 (B 256), the degenerate-edge regime of the 2-view recipe.  Wall-clock on PRO6000 / H100:
B 32 ≈ 45–105 min, B 256 ≈ 35–75 min.  The equal-tuning check (lr × 2 at B = 32) moves VCS +0.84, VICReg +0.62, SimCLR −0.16 — within the
single-seed noise of this table (≈ ±0.4 from the earlier 3-seed runs).

**Reading (frozen rule, seed 0 only — final reading on the 3-seed means).**  VCS retention at B = 32 is 0.981 (≥ 0.98), but no control is at or
below 0.95: SimCLR retains 0.981 and VICReg 1.007 (VICReg is *better* at B ≤ 128 than at 256 under the linear lr rule).  The *holds* clause
needs a control ≤ 0.95 and fails; the *does-not-hold* clause ("VCS degrades at least as much as every control") is met on seed 0 — VCS loses
1.9 % like SimCLR and more than VICReg's 0 %.  **Interim: does not hold** on seed 0; the seed replicates decide whether the 1.9 vs 1.9 vs 0 %
picture is stable (the differences at stake are of the size of one seed's spread).  On kNN the order is SimCLR (0.963) < VCS (0.971) < VICReg
(0.998), also without a batch effect specific to VCS.  What the sweep shows regardless of the verdict: with a critic that averages positives
and product negatives separately, VCS trains to within 2 % of its B = 256 result at B = 32 — but so do both controls, so the property's
promised *advantage* does not materialise at 2 views / 200 epochs on CIFAR-10 from scratch, any more than it did on frozen towers (P54).

## 2. C-T — streaming / asymmetric negatives: 4096-entry queue of detached features, no momentum encoder, B ∈ {16, 32}
| run | linear-val % | kNN-val % | h-rank (final) | held-out J | vs its B = 256 in-batch reference |
|---|---|---|---|---|---|
| VCS + queue, B = 16 | 28.72 | 34.00 | 1.4 | −0.52 | 0.35 of 81.82 |
| VCS + queue, B = 32 | 22.12 | 22.26 | 2.6 | −0.55 | 0.27 |
| SimCLR + queue, B = 16 | 10.58 | 10.66 | 1.3 | — | 0.12 of 86.32 (chance) |
| SimCLR + queue, B = 32 | 10.74 | 10.60 | 2.2 | — | 0.12 (chance) |

Both methods collapse (the summarizer flags `COLLAPSE_SUSPECTED`; h-rank 1–3; the random-init probe already scores 42 % linear, so these runs
end far *below* their own initialisation).  For VCS the held-out J is −0.5: the critic outputs are saturated negative — the queue's stale
features are not a sample of the *current* marginal, so the Q-term is computed against an outdated distribution and the objective is
minimised by making everything look "negative".  SimCLR with the same queue and no momentum encoder is the known-unstable configuration
that MoCo's momentum encoder exists to fix; it collapses to chance.

**Reading (frozen: holds if VCS-with-queue ≥ 0.95 of its B = 256 result at B = 16 while InfoNCE < 0.90 of its own; does not hold if VCS < 0.90
or degrades as much as InfoNCE): does not hold.**  VCS at 0.35 of reference fails the 0.90 floor outright; that it degrades *less* than
InfoNCE (0.35 vs 0.12) is recorded but changes nothing.  The "asymmetric sample sizes" reading of the family (16 positives vs 4096 negatives)
is therefore negative in this form.  A queue fed by a momentum (EMA) encoder for *both* methods — the standard remedy — is a different,
un-pre-registered unit; it is listed as *proposed* and needs the owner's go, because it changes the training recipe.

## 3. Row C after wave 2 (family table, brief appendix A row C)
Micro-batch / large-voxel, memory-bank / streaming, k-way and asymmetric-sample-size families: **closed** on this evidence — no batch effect
specific to VCS from scratch (seed 0; seeds pending), and the naive queue collapses for both objectives.  The one open question is technical,
not property-level: whether a momentum-encoder queue behaves differently for the two objectives.
