# Pre-registration — P133: MoCo-consistent momentum-encoder keys WITHOUT a queue, for the A-P3 scorer and its InfoNCE counterpart (CIFAR-10, 4 views, B 256, 800 epochs, seed 0) — FROZEN 2026-10-05T12:11:52Z 

Status: DRAFT (fork C), gate passed for the launch variant (§10).  The main session freezes it (renamed `*_FROZEN_*`, freeze time in the
title) before any full run.  File name kept from the first draft (`..._QUEUE_...`) for traceability; **the unit no longer tests a queue**.
Owner go 2026-10-04: "全部提交" (the list included the corrected MoCo-style unit).  Source: `P100_MOMENTUM_QUEUE_REPORT_20261001.md` §2–3 — P100
paired online–online positives with online-vs-momentum-key negatives, so its Q was not the product of P's marginals; both methods collapsed
(VCS 12.58 ± 1.11, SimCLR 77.91 ± 1.16; h effective rank 1.0); the report proposed a MoCo-consistent variant.
Main-session decisions 2026-10-05: after the queue variant failed the constant-online-map gate (job 1021175), gate a lower-staleness queue (b)
and a no-queue variant (a) together; (b) primary if it passes; **only (a) passed (job 1021714)**, so per that decision the unit becomes
"momentum-encoder keys without a queue".  Seeds 1–2 rule kept as drafted.

## 1. Question
With every pair built as (online query, momentum-encoder key) — positives and negatives differing only in image identity, both from the same
momentum network at the same step — does replacing the online keys of A-P3-style pairs by momentum-encoder keys change the 4-view CIFAR-10
result of (a) VCS with the A-P3 scorer (fixed T = tanh(2s − 1), original J) and (b) its InfoNCE counterpart, relative to their parents without a
momentum encoder (A-P3 89.06 / 87.30; tuned SimCLR 88.20 / 87.20, seed 0)?

## 2. Design (named variant; markers `pairing.moco_consistent: true`, `pairing.moco_use_queue: false`; neither filled when absent)
- **Online network** (encoder + projector, trained): queries q_v(x) = L2-normalised z of each of the V = 4 views (train-mode BN over the 4B batch,
  as in A-P3).
- **Momentum encoder** (copy of encoder + projector, no gradient): θ_k ← m θ_k + (1 − m) θ_q after every optimiser step, m = 0.99.  **BN: eval mode**,
  running buffers = EMA of the online buffers with the same m — every key is a per-sample function of the momentum network (no batch-statistics
  channel; single-GPU substitute for shuffle-BN).  Keys k_v(x) of all 4 views every step.
- **Positives P**: (q_i(x), k_j(x)) for all ordered view pairs i ≠ j — 12 per image (same-view pair excluded: identical input to both encoders).
- **Negatives Q**: (q_i(x), k_v(x′)) for every other image x′ of the current batch and every view v — 4(B − 1) per query.  P and Q keys come from
  the same momentum network at the same step and the same batch, so for a constant online map P and Q are exchangeable (J ≤ 0 in expectation).
- **VCS cell**: the original J with the A-P3 scorer, P and Q averaged separately over their global counts; queries carry the full gradient, keys
  none.  Everything else = A-P3 seed 0 (`cifar10_hpY_AP3_views4_800ep_seed0.yaml`; only pair_scope / all_view_chunk dropped).
- **InfoNCE cell**: for every (x, i, j ≠ i): −log softmax over [⟨q_i(x), k_j(x)⟩, ⟨q_i(x), k⟩ for the same negatives] / τ, τ = 0.2 (P41), mean over the
  12B terms.  Everything else = P41 SimCLR seed 0 (`cifar10_hpN_simclr_views4_800ep_seed0.yaml`).  No online–online negatives.
- (queue_size stays in the config only because the momentum-encoder policy requires the queue fields; it is unused with moco_use_queue false.)

## 3. Cells (stage `P133_moco_consistent`; `configs/make_p133_configs.py --write`; hashes `configs/P133_SHA256.json`)
| variant | VCS run / config (sha256[:16]) | InfoNCE run / config (sha256[:16]) | gate |
|---|---|---|---|
| **noqueue (launch)** | `P133_vcs_moco_noqueue_views4_800ep_seed0` / `cifar10_hpMC_vcs_noqueue_views4_800ep_seed0.yaml` (1448eb020658b337) | `P133_simclr_moco_noqueue_views4_800ep_seed0` / `cifar10_hpMC_simclr_noqueue_views4_800ep_seed0.yaml` (f2c48d775f27d83b) | PASS (1021714) |
| q1024m999 (queue 1 024, m 0.999) | `P133_vcs_moco_q1024m999_…` (466f4df8dba0ade8) | `P133_simclr_moco_q1024m999_…` (b833bf33aa241919) | FAIL (1021714) — never launched |
| q4096m99 (queue 4 096, m 0.99) | `P133_vcs_moco_views4_800ep_seed0` (885468ae7296475b) | `P133_simclr_moco_views4_800ep_seed0` (c54a3ceb17cbbdbd) | FAIL (1021175) — never launched |
Parents: VCS = A-P3 seed 0 (89.06 / 87.30); InfoNCE = P41 SimCLR seed 0 (88.20 / 87.20).  Config diff vs parent (verified): pairing.{negative_source
queue, queue true, queue_size 1024 (unused), momentum_encoder true, momentum_m 0.99, moco_consistent true, moco_use_queue false}, run.stage;
VCS additionally drops pairing.pair_scope / all_view_chunk.

## 4. Code (this unit; old paths unchanged)
`src/vcs_ssl/objectives.py` (P133 block: `KeyUidQueue`, `moco_consistent_scores`, `moco_consistent_vcs`, `moco_consistent_infonce`,
`compute_objective_moco_consistent`; queue = None → batch keys only), `src/vcs_ssl/config.py` (optional `pairing.moco_consistent`,
`pairing.moco_use_queue`, `_p133_moco_policy`), `src/vcs_ssl/train.py` (P133 blocks: eval-mode keys of all views, EMA of parameters AND BN
buffers, optional key/uid ring with checkpoint/resume, run-manifest record `p133_moco_consistent` incl. `use_queue`).  Tests
`tests/test_vcs_ssl_p133.py` (13).  Gates `slurm/p133_gate.sbatch` (1021175), `slurm/p133_gate2.sbatch` (1021714),
`scripts/p133_gate_constant_map.py`; GPU smoke `slurm/p133_gpu_smoke.sbatch` (1021176).

## 5. Gate requirements
(i) no constant online map q(x) = c reaches J > 0 under the P133 pairs (pass: max J ≤ 2 × half-batch SE) on real CIFAR-10 after 20 real P133 steps
entered at peak lr — candidates: 256 random directions, the mean-key direction (and, with a queue, current-minus-queue), c optimised by gradient
ascent on J — while the same constant maps reach J > 0 under the P100 construction (positive control); (ii) queue / uid / EMA (parameters and
buffers) / eval-mode keys / stop–resume unit tests; (iii) P = "same image, different view", Q = "different image" (test); (iv) old configs' hashes
unchanged; (v) 6-step smokes and stop/resume of both configs.

## 6. Pre-stated reading (seed 0 first; owner rule: one full seed, 3 seeds only if promising)
- Primary: final frozen-h linear (selection split, epoch 800); kNN alongside; official test closed.  Δ = P133 cell − its parent (seed 0).
- **Collapse sentinel** (as P100): collapse flag at two consecutive kNN epochs, or h effective rank at epoch 800 < 10 % of the parent's → reported
  as a collapse of this design, not re-run with other settings.  Per run also reported: training J vs held-out J with current-network product
  negatives (critic_holdout), h effective rank trajectory.
- **Seeds 1–2 rule (addendum, submitted immediately if triggered):** a method gets seeds 1–2 if its seed-0 P133 cell is not collapsed and its
  linear ≥ parent + 0.30 (the P127 CIFAR-10 threshold); if one method triggers, the other method's seeds 1–2 are added too (paired VCS–InfoNCE
  contrast).  Then per method vs parent and VCS − InfoNCE, paired by seed, 95 % t interval, labels as P114: **close** |mean| < 0.3; **clear**
  |mean| ≥ 0.3 and the interval excludes 0; otherwise **inconclusive at 3 seeds**.
- No trigger → "no gain from momentum-encoder keys at seed 0" per method, with Δ and the VCS − InfoNCE gap at seed 0 (descriptive).
- No early stopping on J / kNN / rank; non-finite values or infrastructure failures only.

## 7. Not claimed
**Anything about momentum queues**: both queue variants failed the gate (a constant online map reaches J > 0, i.e. the queued keys' staleness is a
usable shortcut on one GPU) and were never trained — the P133 result says nothing about MoCo-style queues; anything beyond this construction
(one GPU, eval-mode key BN, m 0.99, 4 views, B 256); a tuned m or τ; anything beyond CIFAR-10 standard augmentation.

## 8. Disclosed limits
Keys carry no gradient (MoCo) whereas A-P3 routes the full gradient through both sides; negatives are the batch's 4(B − 1) = 1 020 momentum keys
per query — the same image set as A-P3's negatives, with the key side from the momentum network instead of the online one; the
InfoNCE cell has no online–online in-batch negatives (P41 has 2B − 2); eval-mode key BN is the single-GPU substitute for shuffle-BN.

## 9. Cost
2 × 800 epochs.  GPU smoke (job 1021176, node59, healthy RTX PRO 6000, measured on the q4096 variant = an upper bound for no-queue, which drops
the 1 024 × 4 096 queue matrix): 0.147 s/step vs A-P3 0.116 (+27 %), peak 7.0 GB vs 5.3 GB → ≤ ≈ 5.7 h per run on a healthy RTX6000PRO, longer on
H100 / L40S.  Normal QOS; partitions RTX6000PRO, H100, L40S; **--exclude=node51,node52,node60**.  Launch lines: `slurm/p133_lines.txt`.

## 10. Gate and smoke results
**Round 1 — CPU gate job 1021175 (variant q4096m99: queue 4 096 view-0 keys, m 0.99): gate_rc 1, FAIL.**
- Suites green (ssl_core 29, integration 40, v2 14, P100 8, P104 30, P114 15, P126 36, P133 11); config hashes: 416 configs, 0 changed; smokes and
  stop/resume COMPLETED (CPU s/step P133-VCS 14.1 vs A-P3 11.6; P133-InfoNCE 15.5 vs SimCLR 11.0).
- Constant map: random J ∈ [−0.78, −0.28]; mean-key −0.39; current-minus-queue −0.24; **gradient ascent +0.320** (half-batch SE 0.002) → FAIL.
  P100 positive control +0.97.  InfoNCE best constant 7.78 nats vs chance 8.54.
**Round 2 — CPU gate job 1021714 (variants q1024m999 and noqueue; pass rule max J ≤ 2 SE): gate_rc 0 (infrastructure); per-variant verdicts:**
| variant | max J (argmax) | half-batch SE | 2 SE | random J range | mean-key J | positive control (P100) | InfoNCE best const / chance | verdict |
|---|---|---|---|---|---|---|---|---|
| q1024m999 | **+0.0250** (gradient ascent) | 0.0013 | 0.0025 | [−0.79, −0.22] | −0.56 (current−queue −0.59) | +0.966 | 7.563 / 7.623 | **FAIL** (≈ 20 SE above 0) |
| noqueue | **−0.0003** (gradient ascent) | 0.00001 | 0.00001 | [−0.78, −0.26] | −0.48 | +0.964 | 6.929 / 6.929 | **PASS** |
- Also in 1021714: P133 tests 13 passed; config hashes 430 configs, 0 changed; 6-step smokes and stop/resume of all four variant configs COMPLETED.
- Reading: lowering staleness (16 → 4 steps, m 0.99 → 0.999) shrank the constant-map shortcut from +0.32 to +0.025 but did not remove it;
  without a queue the best constant map sits at J ≈ 0 and the InfoNCE constant at chance, as exchangeability predicts.
**GPU smoke job 1021176** (node59, RTX PRO 6000, 60 steps, q4096 variant): see §9.

## Decisions at the freeze (main session)
Owner 2026-10-04 "全部提交" (corrected MoCo-style unit approved).  Gate round 1 (1021175): the original queue (Q 4096, m 0.99) FAILED — a constant online
map reaches J = +0.32 (queue-key staleness separates P from Q).  Gate round 2 (1021714, rule set by the main session before it ran: (b) primary if it
passes, else (a), else stop): (b) Q 1024 / m 0.999 FAILED (+0.025, ≈ 20 SE); **(a) no queue PASSED** (−0.0003 ≤ 2 SE); P100 positive control +0.96
in both.  Tests 13 / 13; 430 config hashes unchanged.
1. The unit is **"MoCo-consistent momentum-encoder keys without a queue"**: positives (q_i(x), k_j(x)), i ≠ j; negatives = momentum keys of the other
   B − 1 images of the batch (all views); key encoder in eval mode with EMA'd BN buffers; m = 0.99 (P100's value).  Nothing is claimed about
   momentum queues — both queue variants failed the gate and are not trained.
2. Cost not measured for the no-queue variant; the q4096 smoke (0.147 s/step, 7.0 GB, ≈ 5.7 h per run on a healthy RTX) is an upper bound.
3. Seeds 1–2 rule as drafted.  Submit the two seed-0 units now (normal QOS, RTX6000PRO / H100 / L40S, node51 + node52 + node60 excluded).
