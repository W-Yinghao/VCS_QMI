# Pre-registration — P93: D-line mechanism diagnostics on completed models — D1 (P72 adapter drift decomposition, disclosed retraining) and D2 (same-class negative-pair gradient share) — 2026-09-28 — FROZEN 2026-09-28T18:15:38Z before GPU compute (D-line CPU gate 1013249 passed; retraining disclosure of D1 accepted by the main session: no P72 adapter was ever saved)

Status: DRAFT (D fork).  The main session freezes it (renamed `*_FROZEN_*`, timestamp in the title) before any GPU job.
Sources: package v2 Server Spec §8.1–8.2, §2.3, §4, §12–13; Research Plan v2 §7.1–7.2, §2.1; supplement §§1.1, 3.4–3.5; owner go 2026-09-28
(full v2 programme).  Results → `reports/P94_d1_p72_adapters.{json,md}`, `reports/P94_d2_same_class_negatives.{json,md}`, checkpoints and
trajectories under `outputs/P93_D_diag/`; report `P94_D_DIAGNOSTICS_REPORT_<date>.md` (results-only commit first, interpretation second).

## Questions
D1. Why did the P72 adapters trained with VCS move furthest from raw CLIP (clean SRC R@1 0.37 → 0.08 under topic pairing, P72/P80) while the
JS adapters changed least?  Which of (a) the selected optimisation budget (lr × steps, realised AdamW updates), (b) the actual gate
1 − T² on injected vs clean pairs, (c) the per-pair posterior-logit gradient at a unified scale, accounts for the accumulated drift?
D2. On the same frozen VCS and SimCLR checkpoints, with identical base images and negative candidates, what share of the negative-pair
gradient per anchor falls on semantically same-class negatives, before and after normalising by their count share, and do the same-class
and different-class negative sums cancel?  (Labels split the report; nothing is filtered or re-weighted — Spec §2.3.)

## Inputs (all existing; nothing downloaded, no new training runs)
- D1 features (frozen open_clip ViT-B/32 laion2b_s34b_b79k; `models/open_clip/PROVENANCE_*.json`): animal shift `outputs/P49_precheck_A/features/{SRC-FIT,SRC-CAL,SRC-EVAL,TGT-EVAL}.pt`
  (splits sha256 587163f0…, 00e25ade…, fa12a552…, cc5b8408…; n_fit 20 000); indoor→outdoor `outputs/P57_precheck_A_wave2/features_indoor_outdoor/features/*.pt`
  (f26372c6…, d1ce7582…, d1397b2b…, 5a194efe…; n_fit 13 327).  COCO topic index `data/coco_index/train2017_index.json` (topic partners seed 20260927).
  P72 record `reports/P72_robust_r2r3.json` (job 1013019): selected (lr, epochs) per (shift, m, method) and the injected indices
  (m = 0.4: animal 8 000 injected, sha 174d78d4…; indoor_outdoor 5 331, sha 2a2a1330…; m = 0: none).  The script asserts the fit-id and injection hashes.
- D2 checkpoints (the P84 inventory): `outputs/P35_vcs_a5_views4_800ep_seed0/checkpoints/{initial,epoch_100,epoch_400,epoch_800}.pt` (VCS recipe, cosine
  critic a₀ = 5, K = 8, detach, 4 views) and `outputs/P41_simclr_views4_800ep_seed0/checkpoints/{same}` (tuned SimCLR, τ = 0.2, 4 views); identical
  `views` block and identical encoder/projector initialisation (same seed → the two `initial` checkpoints share the encoder).  Manifest
  `cifar10_dev45k_val5k.json` (sha c35d7cd3…); roles = `vcs_estim.pairing.role_split` seed 20260928 (P84 roles, hash 2a2c0598…); base images = a seeded
  permutation (seed 20260931) of the FIT role: 16 batches × 256 = 4 096 identities (uids sha256 recorded in the JSON).  CIFAR-10 labels via the frozen loader.

## Code (this commit; CPU gate below)
`scripts/diag_p72_adapters.py` (D1), `scripts/diag_same_class_negatives.py` (D2), `tests/test_diag_v2.py`, `slurm/diag_cpu_gate.sbatch`,
`slurm/diag_p93_d1.sbatch`, `slurm/diag_p93_d2.sbatch`.  Imports (read-only) from `scripts/robust_r2r3.py`, `scripts/precheck_a_adapters.py`,
`vcs_estim.frozen.load_models`, `reference.ssl_core.cyclic_negative_indices`.  Every model is a deep copy in `eval()`; BN buffers are never updated.

## D1 protocol
**Disclosed retraining.**  Spec §8.1 asks to read the existing P72 checkpoints ("有哪个用哪个，不补训").  `robust_r2r3.py` never saved its adapters
(only scores and metrics were written), so no P72 checkpoint exists.  D1 therefore retrains the minimal set — 2 shifts × m ∈ {0, 0.4} × {vcs, js,
infonce, logistic}, seed 0 only — through the *unchanged* `robust_r2r3.train_r` with the (lr, epochs) that P72's clean-CAL selection recorded, the
same topic partners and the same seeded injection.  The retraining is identical in code path, data, pairing, RNG stream, budget and seed; its
seed-0 CAL loss, SRC/TGT R@1 and R3 AUROC are compared with P72's stored seed-0 values as a reproduction record (GPU non-determinism may move
the fourth digit; a deviation beyond 0.02 in R@1 or AUROC is reported as a failed reproduction and the cell is not interpreted).  The models,
an early checkpoint (end of epoch 1) and the full trajectories are saved this time (`outputs/P93_D_diag/p72_adapters/`).
Per (shift, m, method):
(A) trajectory from an instrumented copy of the same loop (identical RNG consumption; QC sentinel: max |Δθ| between the instrumented and the
    original model must be ≤ 1e-6, otherwise the cell is flagged): per step the lr, |∇| of the injected-anchor part and of the clean part of the
    loss (anchor-row attribution, parts sum to the loss), their cosine, the realised AdamW update |dθ| (Adam-preconditioned; no accumulation in
    P72), the projection of each part onto the update direction, mean posterior-logit gradient and |∂ℓ/∂u| by split; per-epoch means; path
    length Σ|dθ| vs net displacement.
(B) early and final checkpoints on the fixed training pairing (the pairs R3 scored; caption 0): score, cosine, residual, actual gate 1 − T²
    (VCS; q(1 − q) for JS / logistic; p(1 − p) for InfoNCE, labelled), posterior q, posterior-logit gradient, |∂ℓ⁺/∂u|, |∂ℓ/∂u| including the anchor's
    K negative terms, |∂ℓ⁺/∂v|, split injected / clean, with a bootstrap-by-image 95 % CI of the injected − clean difference; the K-pool negatives'
    score, gate and gradient by anchor split.
(C) drift: ‖W_img − I‖_F, ‖W_txt − I‖_F, biases, scalars (a, b, τ); feature displacement |u − u₀| vs the identity adapter on FIT (by split),
    SRC-EVAL, TGT-EVAL; early vs final.
(D) local loss swap: on the same frozen scores and pairs, the VCS-form and JS-form posterior-logit gradients at the SAME q of each checkpoint
    (their ratio is the gate 1 − T² — an identity, verified in the tests).  Kept in its own table: it is a local relation on one state, not a
    comparison of two trajectories (Plan §7.1).  InfoNCE has no pairwise posterior and is absent from (D).
Unified scale: posterior logit l = logit q; VCS q = σ(2f), JS / logistic q = σ(f).  Class ratio / intercept recorded per method (VCS, JS 1:1;
logistic 1:255 in-batch with log 255 as the probability-reading correction; InfoNCE none); no correction is applied to any training record.

## D2 protocol
For every checkpoint and every batch: the same 4 seeded views of the same 256 identities (seed 20260931 + 1009·batch), all 6 view pairs (a < b)
with view a as anchors, the same K = 8 cyclic partners from `cyclic_negative_indices` (seed per batch × pair).  Gradients w.r.t. the anchor's
z (L2 projector output = critic input) and h (pre-projector), critic / temperature fixed.
- VCS (`vcs_K8`): ℓ⁻_ij = (T_ij + T_ij²/2)/K with the partner detached as trained, ℓ⁺_i = −(T_ii − T_ii²/2); per-pair z- and h-gradients by
  autograd (K + 1 calls per view pair); 'score' = T_ij, 'gate' = 1 − T_ij², confidence q_i = (1 + T_ii)/2.
- SimCLR native (`simclr_native`): all 2B − 2 other items, weights w_ij from the anchor's own row of the full NT-Xent (positive kept in the
  denominator, self masked): g_ij = w_ij z_j / τ, g⁺ = −(1 − w_pos) z_pos / τ; the anchor's candidate role in other rows is reported as the ratio
  |total gradient| / |own-row gradient| (full 2B-anchor NT-Xent by autograd); 'score' = 'gate' = w_ij (SimCLR's own easy-negative decay);
  h-space only for the three sums (same, different, positive).
- SimCLR K-matched (`simclr_Kmatched`, diagnostic only, not its training loss): softmax over {positive, the same K partners}.
Per anchor: n_same, count share, similarity (cosine) same / different, score and gate same / different, confidence, numerator Σ_same‖g‖,
denominator Σ_{j≠i}‖g‖, F_same = num / (den + 1e-12) (z; h for the K-candidate variants), the vector sums G_same, G_diff, G_pos with their
cosines and the cancellation index ‖G_same + G_diff‖ / (‖G_same‖ + ‖G_diff‖), mass ratio Σnum / Σden, relative weighting = mass ratio / count share.
Bootstrap by image (500 replicates; all instances of an image move together) for every mean; conditional means on anchors with ≥ 1 same-class
negative reported next to the unconditional ones (with K = 8 and 10 classes ≈ 57 % of anchors have at least one).

## Reading rules (pre-stated; descriptive, bootstrap CIs; no significance language beyond the CI)
D1 — three pre-stated attributions per (shift, m = 0.4), read on the final checkpoint and the trajectory:
- *budget*: the ordering of net parameter displacement across the four methods coincides with the ordering of Σ|dθ| (path length), and the
  per-step |dθ| of the method with the largest displacement is within 1.5× of the others → drift is attributable to the selected lr × steps;
- *injected pairs*: Σ proj_inj / (Σ proj_inj + Σ proj_cln) exceeds the injected sample share (0.4) by ≥ 1.5× for the largest-drift method and not
  for the others → the injected pairs drive that method's update;
- *gate*: the injected / clean gate ratio at the final checkpoint differs between VCS and JS by ≥ 2× → the two losses weight the injected pairs
  differently at their own operating points.
  More than one may hold; none holding is reported as "not separated by these measurements".  The local swap table (D) is never read as a cause.
  m = 0 cells give the drift without injection (the topic-pairing drift of P80 §1).  A failed reproduction or QC sentinel voids the cell.
D2 — *VCS reduces the semantic-conflict gradient share* if, at epochs 400 and 800, VCS's relative weighting (mass ratio / count share, z-space) is
< 1 with the bootstrap CI excluding 1 while SimCLR-native's is ≥ 1 with its CI excluding values < 1; *increases* if the reverse; *no method
difference* if both CIs contain 1 or the two CIs overlap.  The initial and epoch-100 checkpoints are trajectory context.  Similarity, gate and
confidence distributions are reported before the gradient shares (same-class negatives need not be the saturated ones).  The cancellation index
and the cosines are reported without thresholds; a cancellation index < 0.5 for one method only is flagged for the interpretation commit.
The K-matched SimCLR row never replaces the native row in a verdict.

## Cost (GPU minutes; from P72's timings: ≈ 2 s per adapter training on the P72 GPU, ≈ 995 s for 480 trainings)
D1: 16 cells × (original training + instrumented copy ≈ 5× + pair metrics on early and final + drift) ≈ 30–60 s per cell → ≈ 10–20 min on one GPU
(walltime 3 h requested).  D2: 8 checkpoints × (16 384 encodes + 96 (batch, pair) analyses + bootstrap) ≈ 1–2 min each → ≈ 15 min (walltime 2 h).
Both fit the spare quota without touching the S-line jobs.

## CPU gate (partition CPU, no GPU): `slurm/diag_cpu_gate.sbatch`, job **1013249** — tests + `--smoke --cpu` of D1 (animal, n = 1 024, 2 epochs),
D2 (two checkpoints, one 64-image batch) and the addendum.  Results: see the section below.

## Launch lines (main session, after freezing; GPU, spare quota)
```
cd /home/infres/yinwang/CS_QMI/ssl_pilot && sbatch --parsable slurm/diag_p93_d1.sbatch
cd /home/infres/yinwang/CS_QMI/ssl_pilot && sbatch --parsable slurm/diag_p93_d2.sbatch
```

## Delivery block (Spec §13.2; one per results file)
```yaml
experiment_family: diagnostic
protocol_id: P93_D1 | P93_D2
source_commit: <commit of this prereg's code>
source_basis: supplement_af7172c_plus_v2
estimator: vcs_neural (+ js / infonce / logistic adapters for D1; simclr for D2)
estimand: none (mechanism diagnostic on fixed models)
evaluation_readout: native scores, gates, per-pair gradients, displacements
loss_scale: per-anchor terms (D2: positive weight 1, each negative 1/K for VCS; own-row NT-Xent for SimCLR)
reference_measure: mixture_equal (VCS, JS) / native_base (InfoNCE, logistic, SimCLR)
critic_class: D1 tanh(a cos + b) adapters (a0 = 5), JS / logistic sigma(a cos + b), InfoNCE cos / tau; D2 CosineCritic (VCS) / temperature 0.2 (SimCLR)
gradient_routing: D1 full (adapters); D2 negative_right_detach (VCS as trained) / native (SimCLR)
n_independent_units: D1 20 000 + 13 327 FIT images per shift; D2 4 096 identities
n_positive_pairs: D1 n_fit per cell; D2 4 096 x 6 view pairs per checkpoint
n_negative_pairs: D1 K = 8 per anchor; D2 K = 8 (VCS, K-matched) / 2B - 2 = 510 (SimCLR native) per anchor
split_manifest_hash: c35d7cd3… (CIFAR-10 dev45k) / P49–P57 split hashes above
noise_target_kind: none
fit_seconds / evaluation_seconds: measured, in the JSON
status: <actual>
```

## Disclosures
- D1 is a retraining (above); the P72 numbers it reproduces are seed 0 only.  The instrumented copy adds `autograd.grad` calls that do not consume RNG;
  its final parameters are compared with the original path's (sentinel).
- The unified posterior-logit scale is a reading convention; JS / logistic native logits, class ratios and intercepts are recorded as trained.
- D2 defines the anchors as view a of each unordered view pair, matching `vcs_pair_loss_negdetach`; SimCLR's symmetric rows for view b are covered
  by the other ordered pair only through its own-row total, reported as the total / own ratio.  Per-pair h-space gradients are formed only for
  the K-candidate variants (cost); the native SimCLR h-space figures are the three sums.
- No EVAL / test data: D1 reads SRC-EVAL / TGT-EVAL only for the reproduction row and the drift magnitude (as P72 did); D2 reads FIT-role images only.

## CPU gate results (job 1013249, nodecpu12, 2026-09-28T18:08–18:09 UTC, 100 s wall; `slurm_logs/diag_cpu_gate_1013249.out`)
- `tests/test_diag_v2.py`: 9 passed (gate identities, posterior-logit gradients and swap ratio = gate, F_same bookkeeping, SimCLR weights / own-row
  gradient, bootstrap-by-image integrity, dictionary identity and bounds, displacement metrics, ridge stationarity, cancellation index).
- D1 smoke (animal, n = 1 024, 2 epochs, m ∈ {0, 0.4}, four methods, 19 s CPU): QC sentinel max |Δθ| instrumented vs original = 0.00e+00 in all 8 cells;
  early / final checkpoints, trajectories, JSON and markdown written (`outputs/P93_D_diag/smoke/`).  Reproduction columns differ from P72 as expected at
  1 024 images / 2 epochs (e.g. SRC R@1 0.49–0.54 vs P72's 0.08–0.19) — code path evidence only.
- D2 smoke (one 64-image batch, 6 view pairs, VCS ep 800 and SimCLR ep 800, 2 s each): VCS F_same_z 0.275 (count share 0.105, mass ratio 0.60),
  SimCLR-native F_same_z 0.224 (count share 0.108, mass ratio 0.23); cancellation index 0.99 / 0.77; gate same / diff 0.185 / 0.032 (VCS) — R = 64 images,
  not read for any verdict.
- Addendum smoke (P35 ep 800, roles 1 024 / 256 / 256 / 256, 100 updates, 31 s CPU): identity residual J(T_w) − Σ w_j J_j − D_EVAL(w) = 0 to float
  precision in all three dictionaries; bound checks D_fit ≤ max D ≤ U_dict true; D(w_P84) = 0 (w = [0, 0, 1]); the 100-update members are budget-limited
  as in P83's smoke, so the smoke's extended-dictionary numbers are not indicative.
Nothing in the design was changed after the gate.
