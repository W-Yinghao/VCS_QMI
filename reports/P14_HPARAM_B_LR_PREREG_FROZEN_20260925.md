# Pre-registration — VCS-QMI learning-rate / schedule single factors (P14), frozen 2026-09-25 before launch

Owner decision: single-seed exploration of hyper-parameter sets (no 3-seed confirmation; seed SD at K=1 was 0.46 linear points, P5);
immediate question: **is the slow convergence caused by a too-small learning rate?**

## Units (seed 0, 200 epochs, K = 1, everything else = `cifar10_confirm200_vcs_seed0.yaml`; baseline `P5_vcs_seed0` 74.84 % / kNN 64.64)
| run | field | value | question |
|---|---|---|---|
| P14_vcs_lr3e-3_seed0 | `optimizer.lr` | 3e-3 | LR too small? |
| P14_vcs_lr1e-2_seed0 | `optimizer.lr` | 1e-2 | upper bound (divergence is an admissible outcome, recorded as FAILED_NUMERICAL) |
| P14_vcs_lr3e-4_seed0 | `optimizer.lr` | 3e-4 | opposite direction |
| P14_vcs_floor0.1_seed0 | `schedule.min_lr_ratio` | 0.1 | is the late plateau schedule-driven? |
| P14_vcs_const_seed0 | `schedule.min_lr_ratio` | 1.0 (constant LR after the 10-epoch warm-up) | strongest version of the same question |

The critic uses the same multiplier (×1) as encoder/projector in all runs, so `optimizer.lr` scales all three groups together.
Configs + sha256: `configs/HPARAM_B_LR_SHA256.json`.  No code change except recording `min_lr_ratio` in `run_manifest.hparams`.

## Endpoints and reading (same rule as P12)
Primary: final linear-val vs 74.84 (HELPS > +1.0 / HURTS < −1.0 / neutral); secondary: kNN trajectory at 0/10/20/50/100/150/200 versus the
baseline trajectory at the same epochs (speed of convergence, not only the endpoint), heldout-J, h-rank, saturation, gradient norms, any
non-finite event.  A faster early trajectory with the same endpoint is reported as "faster, not better"; a higher endpoint under
`const` or `floor0.1` means the 1 % cosine floor was limiting; divergence at 1e-2 bounds the usable LR.

## Not claimed
Single seed; selection set; no control comparison; nothing about the objective's population properties.
