# P100 — momentum-encoder key queue (VCS and tuned SimCLR, 8×, 3 seeds) — report — 2026-10-01

Pre-registration: `P100_MOMENTUM_QUEUE_PREREG_FROZEN_20260930.md`. Results-only commit: `f96648b` (`P100_momentum_queue_table.md`).
Evaluation: selection split, frozen-h linear probe and kNN at epoch 800.

## 1. Results and verdict under the frozen rule: **hurts** for both methods

| method | seed 0 | seed 1 | seed 2 | mean ± sd | baseline, no queue (P35 / P41) | Δ | 2 × pooled SE | rule | kNN mean (baseline) |
|---|---|---|---|---|---|---|---|---|---|
| VCS + queue | 11.30 | 13.12 | 13.32 | **12.58 ± 1.11** | 87.01 ± 0.53 | −74.43 | 1.42 | **hurts** | 28.14 (85.46) |
| SimCLR + queue | 78.08 | 78.98 | 76.68 | **77.91 ± 1.16** | 88.32 ± 0.30 | −10.41 | 1.38 | **hurts** | 80.18 (87.65) |

- The VCS − SimCLR gap is −65.3 with the queue, against −1.31 without it. This is not meaningful, given the collapse described next.
- All 6 runs show the collapse flag at two or more consecutive kNN epochs, so under the prereg they are **reported as collapses** and not re-run with other settings.

## 2. Diagnosis: a design flaw in the pre-registered variant
Both methods show **dimensional collapse**:

| run | effective rank of h at epoch 800 | effective rank at epoch 20 | effective rank of z |
|---|---|---|---|
| VCS + queue | 1.0 | ≈ 1.0 | ≈ 2 |
| SimCLR + queue | 1.0 | — | 9–13 |
| recipe VCS (P35) | 136 | — | — |
| recipe SimCLR (P41) | 158 | — | — |

The VCS logs show the mechanism:
- **The training J gets close to its maximum almost immediately.** J_raw is 0.97 at epoch 2 and 0.99 from epoch 5 to epoch 400.
- **The held-out J with *current-network* product negatives is strongly negative.** It is −0.95 at epoch 20 and −0.64 at epoch 800.
  So the critic scores true product-of-marginals pairs as positives: every image maps to nearly the same point.

**Cause.** In the pre-registered variant, the two sides of the comparison come from different encoders:
- The positive pairs are **online–online**: two views through the online network.
- The negatives are **online vs momentum-encoder keys** from earlier steps.

So the "Q" distribution is **not the product of P's marginals**. The second side of every negative comes from a different, lagging network, with its own BatchNorm statistics.
The objective can then be maximised by a constant online representation that is merely far from the momentum encoder's keys. In other words, it discriminates *which encoder and step produced the key*, not *which image*. The near-maximal training J alongside a collapsed representation is exactly that shortcut.

MoCo avoids it because its positive key also comes from the momentum encoder. Positives and negatives are then both "online query vs momentum key" and differ only in image identity.
- **SimCLR** partly resists the shortcut, because its in-batch online–online negatives remain in the denominator. Its rank still collapses, and its accuracy drops by 10.4.
- **VCS** has no in-batch negatives left: the queue *replaced* its K = 8 cyclic partners. It collapses fully.

**Reading.** P100 does **not** test a MoCo-style queue. It tests an asymmetric construction that breaks the product-of-marginals premise of both objectives, and its "hurts" verdict should be read that way.
The prereg's "Disclosed limits" named related asymmetries (one key view, no shuffle-BN, additive vs replacing queue) but not this one. It is a design error in the frozen protocol, caught here by the collapse sentinel.

## 3. Not claimed and not run
- **Not claimed:** anything about momentum queues in general, or the VCS–SimCLR gap under a correct queue design.
- **Possible follow-up, not submitted, needs the owner's decision.** A MoCo-consistent variant would use:
  - positives: online query of view i vs momentum key of view j;
  - negatives: the same online query vs queued momentum keys, so P and Q differ only in image identity;
  - optionally shuffle-BN, or a cross-view fixed BN.

  It needs a new prereg, a CPU gate that checks a constant online map cannot reach J > 0, and 6 × 8× runs (about 30 GPU-h).

## Delivery
```yaml
experiment_family: full_ssl (method extension)
protocol_id: P100_momentum_queue
source_commit: 4f0b576 (frozen); results f96648b
estimator: vcs (J, cosine-tanh a0 5, K 8 from the queue) | simclr tuned (+ queue negatives)
evaluation_readout: frozen-h linear + kNN, selection split, epoch 800
n_independent_units: 45000 fit images; 3 seeds per method
status: complete — hurts (both); all runs collapsed (h effective rank 1.0); cause = negatives from a different encoder than positives (protocol design flaw)
```
