# Pre-registration — target-branch / predictor / projector-depth variants (P22–P23), frozen 2026-09-25 before launch

Owner's guiding principle: keep the collaborator's plan and the ResNet-18 two-view trunk fixed, explore everything else for the
best-performing VCS SSL implementation.  These variants change only the *SSL wiring* (which representations meet the critic and how
gradients flow); the objective J, the mixture reference, product-of-marginals negatives (cyclic shifts) and the tanh critic are untouched.

## Units (seed 0, 200 epochs; base = `cifar10_hpB_k8_clr10_vcs_seed0.yaml` (K=8, critic lr ×10); baseline `P16_vcs_k8_clr10_seed0`
77.44 % / kNN 68.40 / h-rank 21.4)

| unit | change | rule |
|---|---|---|
| ema0.99 / ema0.996 | `train.target_branch=ema_<τ>` | target view features come from an EMA copy of encoder+projector (τ constant; params EMA, BN buffers copied), no gradient; pairs (s1,t2),(s2,t1) symmetric; negatives with the same shifts against target features |
| stopgrad | `train.target_branch=stopgrad` | target = detached student features of the other view (SimSiam-style, no EMA) |
| sg_pred | stopgrad + `model.projector.predictor=true` | student side passes through a predictor MLP (128→512-BN-ReLU→128) before the critic |
| ema0.99_pred | ema_0.99 + predictor | BYOL-style asymmetry with the VCS objective |
| proj_depth3 | `model.projector.depth=3` | Linear-BN-ReLU-Linear-BN-ReLU-Linear (512-512-128) |

Frozen evaluation is unchanged (student encoder `h`).  The critic hold-out diagnostic uses student features on both sides for these
units (disclosed: its J is not the training J when a target branch is used).  Configs and sha256: `configs/HPARAM_E_TARGET_SHA256.json`.

## Reading
Primary: final linear-val vs 77.44 (HELPS > +1.0 / HURTS < −1.0 / neutral); secondary: h-rank, kNN, heldout-J, saturation, collapse flag
(stop-grad/EMA variants can collapse — the negatives in J should prevent it; a collapse is a result).  Failures recorded, not patched.

## Not claimed
Single seed; selection set; controls untouched; an EMA/predictor variant that helps is "VCS objective with BYOL-style wiring", to be named so.
