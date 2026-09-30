# P101 — O line report: calibrated observation noise on frozen SSL representations — 2026-10-01

Pre-registration `P101_O_LINE_FROZEN_REPR_PREREG_FROZEN_20260930.md`; results-only commit (`P101_o_line.md`, job 1015880, exit 0).  Six P84 checkpoints
(VCS 4-view recipe and tuned SimCLR 4-view, epochs 100 / 400 / 800); noise u = z + τ ε/√d on unit-norm z (no re-normalisation), isotropic and in a fixed FIT-PCA
subspace (r = 28, 91 % variance); τ per J_proxy target calibrated on FIT-CAL with the converged-cosine calibration critic and applied as one common τ per target
(reference checkpoint VCS epoch 800); measurement critics refitted on FIT-CRITIC; EVAL read once.  J_proxy is a calibration scale, never called S.

## 1. How much noise each representation tolerates (τ* per checkpoint, descriptive)
| checkpoint | iso τ* for J_proxy 0.95 / 0.85 / 0.70 / 0.50 | PCA τ* |
|---|---|---|
| VCS ep 100 / 400 / 800 | 0.30 / 0.60 / 0.80 / 1.13 ; 0.30 / 0.38 / 0.56 / 0.80 ; 0.30 / 0.41 / 0.56 / 0.80 | 0.10–0.20 / 0.38–0.45 / 0.49–0.60 / 0.65–0.75 |
| SimCLR ep 100 / 400 / 800 | 0.60 / 1.38 / 1.69 / — ; 1.00 / 1.50 / 1.75 / — ; 1.00 / 1.50 / 1.81 / — | 0.60–1.00 / 1.13–1.25 / 1.38–1.56 / 1.75–1.88 |

The SimCLR representations need 2–3× more noise to lose the same dependence (J_proxy 0.5 is unreachable isotropically within τ ≤ 2), consistent with their
wider positive / negative cosine separation (P90 §2: median negative cosine −0.004 SimCLR vs 0.77 VCS).  A small noise level already erases most of the
dependence VCS's narrow-cone z carries.

## 2. Pre-stated reading
Rule: "noise improves resolution" only if some calibrated target and critic class raise an adjacent-checkpoint resolution ≥ 1.5× over τ = 0 **and** every
checkpoint of that pair loses < 2.0 kNN points; isotropic and PCA read separately.
| method | isotropic | PCA subspace |
|---|---|---|
| **VCS** | **no resolution gain.**  The largest ratios (×1.8–3.4 for ep 100 → 400 at J_proxy ≤ 0.85) come with kNN losses of 2.3–14.6 points; the only ratio ≥ 1.5 at 0.85 without ep 100 fails at ep 400 (−2.3). | **no resolution gain.**  Ratios ≥ 1.5 appear only at J_proxy ≤ 0.85, where kNN falls by 8–29 points. |
| **SimCLR** | **no resolution gain** (largest ratio ×1.19). | **gain by the letter, marginal:** at J_proxy 0.5 the two C2 critics raise ep 100 → 400 resolution ×1.53 (VCS objective) and ×1.56 (JS objective) with kNN losses of 0.8 / 0.7 points; C0 reaches ×1.36–1.46. |

Observation, not read as a verdict: under the common τ (calibrated on the VCS reference) the SimCLR checkpoints receive a mild perturbation — their kNN moves by
≤ 1.2 points at every target — while the VCS checkpoints lose 1.4–29 kNN points.  The ep 400 → 800 VCS pair becomes unresolvable (ratio ≤ 0.3) at every isotropic
target ≤ 0.95 for the cosine critics: noise collapses the small late-training differences of the narrow-cone representation first.

## 3. What can be written
- Observation noise at a calibrated J_proxy does **not** improve the resolution of VCS checkpoints on frozen representations; it trades resolution for class
  information.  The v2 O line's hope of a de-saturating, resolution-improving observation scale is not supported for VCS representations.
- SimCLR representations tolerate 2–3× more observation noise for the same J_proxy; a marginal, single-critic-class resolution gain appears only in the PCA
  subspace at the most aggressive target.
- The in-training noise of P95 (τ 0.3, selected in stage A at 87.32 on seed 0) is a different object (it changes the training estimand S_σ); its verdict comes
  from P96 / P104-N1, not from this frozen-representation analysis.

## Delivery (Spec v2 §13.2)
```yaml
experiment_family: diagnostic (O line)
protocol_id: P101_O_line_frozen
source_commit: e466fd5 (frozen); results this commit's parent
estimator: vcs_neural | js_matched (C0, C0conv, C2)
estimand: S_sigma on frozen features, calibrated to J_proxy targets {0.95, 0.85, 0.70, 0.50}
noise_target_kind: J_proxy
noise_tau: common per target (reference VCS epoch 800), per-checkpoint tau* reported
evaluation_readout: J on EVAL blocks, adjacent-checkpoint resolution, kNN of noisy z
n_independent_units: P84 roles (27k/6k/6k/6k identities, seed 20260928), 6 checkpoints
status: complete
```
