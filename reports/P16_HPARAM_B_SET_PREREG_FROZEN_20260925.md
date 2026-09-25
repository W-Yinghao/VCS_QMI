# Pre-registration — VCS-QMI B-group "set" batch (P16–P17), frozen 2026-09-25 before launch

Owner decision ("再提交一批") after the bottleneck assessment: the evidence (P11 K, P13 critic/projector screen, P9 800 epochs) points to
early saturation of the bounded quadratic objective (85–95 % of pairs with |t| > 0.95, J ≈ 0.90–0.95) and hence weak encoder signal, not to
model capacity or negative count.  This batch tests the remaining levers that do not change the objective's mathematics, plus first
combinations of the factors that helped.  Single seed 0; baseline `P5_vcs_seed0` (74.84 % / kNN 64.64 / h-rank 12.9) for 200-epoch units,
`P8_vcs800_seed0` (79.24 % / 71.36 / 20.1) for the 800-epoch unit.  Configs and sha256: `configs/HPARAM_B_SET_SHA256.json`.

| unit | change(s) | hypothesis tested |
|---|---|---|
| aug_crop008 | crop scale [0.2,1] → [0.08,1] | harder positives → fewer saturated positive pairs → more signal |
| aug_cj08 | color jitter 0.4/0.4/0.4/0.1 → 0.8/0.8/0.8/0.2 | same |
| aug_blur05 | Gaussian blur p=0.5 (3×3, σ∈[0.1,2]); **new code**, named variant, VCS only | same |
| aug_strong | all three above | same, combined |
| aug_weak | crop [0.5,1], jitter 0.2/0.2/0.2/0.05 | opposite direction (control for the mechanism) |
| b128 / b512 / b1024 | batch size (LR not scaled, per spec) | more/fewer pairs per step |
| wd1e-5 / wd5e-4 | encoder/projector matrix weight decay | B-group factor not yet screened |
| k8_clr3 / k8_clr10 | K=8 + critic LR ×3 / ×10 | additivity of the two helpful factors |
| k8_aug_strong | K=8 + strong augmentation | combination with the strongest hypothesis-driven lever |
| k8_clr10_800ep | K=8 + critic LR ×10, 800 epochs (checkpoints 100/200/400/600/800) | best-achievable estimate with the current objective; compared with P8 seed 0 |

Everything else is the frozen 200-epoch VCS recipe.  Code changes: blur implemented in `build_two_view_transform` (policy: blur allowed
for VCS variants only; controls stay pinned; solarize still refused); augmentation / weight-decay fields recorded in `run_manifest.hparams`.

## Endpoints and reading (same rule as P12/P14)
Final linear-val vs the matching baseline (HELPS > +1.0 / HURTS < −1.0 / neutral); kNN trajectory vs baseline at the same epochs; h-rank and
dominant `p_raw` directions; saturation fractions and heldout-J (for augmentation units a *lower* saturation with equal or better linear-val
is the predicted signature of the hypothesis; lower J with worse linear-val is not).  Batch-size units change steps/epoch (T = 200·⌊45000/B⌋),
so they are compared at equal epochs, not equal steps.  Any non-finite / OOM event is recorded, not patched.

## Not claimed
Single seed; selection set; no control comparison (controls would need the same augmentation / batch changes before any comparison);
nothing about the objective's population properties.
