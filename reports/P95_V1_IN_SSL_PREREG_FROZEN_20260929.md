# Pre-registration — P95: the package-v1 estimator improvements as online critic variants inside full CIFAR-10 SSL at 8× (residual, simplex dictionary, observation noise, critic refresh; matched JS control), 2026-09-29 — FROZEN 2026-09-29T20:58:24Z before GPU compute (CPU gate 1014585 passed: ssl_core 29, integration 40, v2 14, p95 15; recipe first-step gradients unchanged; config hashes of running units unchanged)

Status: FROZEN 2026-09-29T20:58:24Z (main session; owner go 2026-09-29 "我建议都放进ssl里试试…就在cifar10尝试").  The main session freezes it (renamed `*_FROZEN_*`, freeze time in the title) before any GPU job.
Owner decision (2026-09-29, verbatim): "我建议都放进ssl里试试，这种probe还是不能代替全量实验。就在cifar10尝试".
Sources: `VCS_QMI_Estimator_Research_and_Server_v1.zip` — Research Plan v1 §4 (gate mechanism), §5 (bounded residual, small critic dictionary), §6
(observation scale); Server Spec v1 §7 (convex combination, residual candidate, controls), §9 (noise definition); Server Spec v2 §2.2 (matched JS),
§8.3 / §10 (refresh caution, noise scale); the offline evidence this unit replaces as the deciding test: P82 (Gaussian probe), P84 + P84 addendum 1
(`reports/P84_ADDENDUM1_REPORT_20260928.md`: realised mix / residual gain ≤ 0.001 J on frozen features; converged cosine 3× steeper with 3–7× smaller
representation gradient).  The owner's point is that frozen-feature probes cannot stand in for training: every variant is therefore trained end to end.

## Question
Does any of the v1 estimator improvements, used as the *training* critic of the frozen 8× recipe, give a better frozen representation than the recipe
itself (87.01 ± 0.53 selection-split linear / 85.46 kNN; P35 seeds 0–2: 86.42 / 87.16 / 87.44)?  And, as a control, how does the same critic trained
with the matched balanced-logistic loss compare?

## Fixed (collaborator's plan; unchanged in every cell)
J with separate P and Q means (loss −J; the JS cell is the labelled control), M = (P + Q)/2, product-of-marginals negatives by K = 8 nonzero cyclic shifts,
tanh-bounded critic output, no log transform, no extra loss term.  Everything else is the recipe YAML `configs/cifar10_hpK_a5_views4_800ep_vcs_seed{s}.yaml`:
ResNet-18 CIFAR, 4 views (J averaged over the 6 view pairs), B 256, 800 epochs, AdamW 1e-3 (critic group lr multiplier 1, wd 0), warm-up 10,
cosine critic a0 = 5, negative-partner detach, 512/128 projector, selection-split evaluation (frozen h, linear probe + kNN as in every unit).

## Variants (code: `src/vcs_ssl/models/critic.py`, `src/vcs_ssl/objectives.py`, `src/vcs_ssl/config.py`, `src/vcs_ssl/train.py`; each a config switch)
| family | config switch | definition |
|---|---|---|
| residual | `critic.input: residual_cosine_mlp`, `critic.residual_lambda: λ` | T = (1 − λ)·tanh(a⟨z1, z2⟩ + b) + λ·tanh(g([z1; z2])); g = the recipe's ordered-concat MLP class (hidden [512, 512], ReLU, last layer Xavier gain 0.1, bias 0); λ fixed; |T| ≤ 1 (convex combination) |
| dictionary | `critic.input: dictionary_simplex` | T = Σ_j w_j T_j, w = softmax(θ), θ learnable (init 0 = equal), members {cosine a0 = 5, ordered-concat MLP [512, 512], bilinear_concat [512, 512]}, all trained jointly by J with the critic optimizer; logs w and D(w) = Σ w_j E_M T_j² − E_M T_w² per step |
| noise | `critic.observation_noise_tau: τ` (cosine critic) | the critic reads u = z + τ·ε/√d on both sides of every pair (positive and negative blocks each draw fresh ε; z not re-normalised); training mode only (evaluation and critic hold-out read clean z); noise_total_rms = τ, noise_coordinate_sd = τ/√128; counter-based noise stream (seed from the critic's own seed stream, call counter in the checkpoint) |
| refresh | `critic.refresh_every_epochs: R`, `refresh_batches: 16` (cosine critic) | at the end of every epoch e with e mod R = 0 and e < 800, (a, b) are refit to convergence (full-batch L-BFGS on −J, float64) on 16 × 256 FIT images (two training-augmentation views, K = 8 shifts, encoder/projector in eval(), dedicated RNG seeds, training RNG untouched), applied only if J does not decrease, logged to `logs/critic_refresh.jsonl`; AdamW moments of a, b are kept (disclosed) |
| js (control) | `objective.loss: js_matched_logistic` | same cosine critic logit f = a⟨z1, z2⟩ + b, same pairs / shifts / detach; L = mean_P softplus(−2f) + mean_Q softplus(2f) (Spec v2 §2.2: f-gradient at f = 0 equals that of −J); J and the VCS statistics logged from T = tanh(f) |

Config fields are optional and are *not inserted* into configs that do not carry them: the resolved dict and `config_hash` of every existing config are
unchanged (verified for the recipe, P87 8×, P91 8×, the SimCLR 8× and the pilot configs: identical hashes under the old and the new code), so running
jobs resume and evaluate as before.

## Cells
**Stage A (stage `P95_v1_in_ssl`, seed 0, 8×; `slurm/p95_units_stageA.txt`; `configs/P95_SHA256.json`):**
| run id | config | sha256 (16) |
|---|---|---|
| P95_residual_lam0.25_views4_800ep_seed0 | cifar10_hpV_residual_lam0.25_views4_800ep_seed0.yaml | b669c40990956b0c |
| P95_residual_lam0.5_views4_800ep_seed0 | cifar10_hpV_residual_lam0.5_views4_800ep_seed0.yaml | cc9201bfa9eabab0 |
| P95_dictionary_views4_800ep_seed0 | cifar10_hpV_dictionary_views4_800ep_seed0.yaml | baeb86b490b1520c |
| P95_noise_tau0.1_views4_800ep_seed0 | cifar10_hpV_noise_tau0.1_views4_800ep_seed0.yaml | 6cb22974d1aa1a18 |
| P95_noise_tau0.3_views4_800ep_seed0 | cifar10_hpV_noise_tau0.3_views4_800ep_seed0.yaml | a63404a0aa874119 |
| P95_refresh_R50_views4_800ep_seed0 | cifar10_hpV_refresh_R50_views4_800ep_seed0.yaml | a6d633cc593facfe |
| P95_refresh_R200_views4_800ep_seed0 | cifar10_hpV_refresh_R200_views4_800ep_seed0.yaml | e15316e50c683eae |
| P95_js_matched_views4_800ep_seed0 | cifar10_hpV_js_matched_views4_800ep_seed0.yaml | 90b267d1168bfb05 |
Recipe seed 0 (P35_vcs_a5_views4_800ep_seed0, 86.42 / 85.60) is the reused same-seed reference; it is not re-run.

**Stage B (after a frozen selection addendum):** seeds 1–2 of the selected cell per family (residual, dictionary, noise, refresh) plus js seeds 1–2 —
up to 10 runs, generated by `python configs/make_p95_configs.py --stageB residual=<λ*> dictionary noise=<τ*> refresh=<R*> js --write`
(`slurm/p95_units_stageB.txt`).  Nothing is re-selected after stage B.

## Selection rule (pre-stated; stage A → stage B)
Per family, the stage-A cell with the highest selection-split linear top-1 at epoch 800; ties within 0.1 points → the grid value closer to the recipe
(smaller λ, smaller τ, larger R).  A family whose best seed-0 cell is below the recipe's worst seed (86.42) by more than 1.0 (i.e. < 85.42) is reported and
not seeded; the js control is always seeded.  Numerically failing cells (non-finite loss / gradient, collapse flag at two consecutive kNN epochs) are
recorded and excluded; a family with no valid cell is not seeded and the failure is the result.

## Reading rule (pre-stated; stage B, 3 seeds per selected family; descriptive, no significance language)
With the recipe's 3-seed mean 87.01, seed values 86.42 / 87.16 / 87.44 and pooled SE = sqrt(sd_family²/3 + 0.53²/3):
- **improves the recipe** if the family's 3-seed mean ≥ 87.01 + 2 × pooled SE **and** every seed ≥ 86.42;
- **hurts** if the mean ≤ 87.01 − 2 × pooled SE;
- **on par** otherwise.
kNN (recipe 85.46 ± 0.12) is reported with the same arithmetic next to it, not read as a second verdict.  The js control is read the same way against the
recipe (it answers "is the quadratic J better or worse than the matched logistic loss on the same critic in full training"), not as a VCS variant.
Mechanism records reported descriptively for every stage-A cell from the trainer logs: J, R, t± means and saturation, a / b trajectories, the dictionary
weights and D(w), residual parts, empirical noise scale, refresh ΔJ and (a, b) jumps, h effective rank, held-out J.

## Cost (measured per-step times: see gate)
Recipe 8× ≈ 4.5 h on RTX6000PRO / 7.5 h on H100 (20 / 33 s per epoch).  Variant step-time ratios vs the recipe on the gate CPU: see "Gate" below;
dictionary and residual add MLP / bilinear critic evaluations on 6 view pairs × (B + 8B) pairs and are the heavy cells.  Stage A: 8 runs.  Stage B: ≤ 10 runs.
All ≤ 23 h on RTX6000PRO / H100 (owner 2026-09-29: long runs on the fast GPUs; `--partition=RTX6000PRO,H100 --exclude=node60`).

## Launch (main session, after the freeze)
```
cd /home/infres/yinwang/CS_QMI/ssl_pilot
awk '{printf "sbatch --parsable --partition=RTX6000PRO,H100 --exclude=node60 --job-name=%s --export=ALL,CFG=%s,RUN_ID=%s,FINAL_CKPT=%s slurm/run_unit.sbatch\n", substr($1,5), $2, $1, $3}' slurm/p95_units_stageA.txt > slurm/p95_stageA_lines.txt
LINES_FILE=slurm/p95_stageA_lines.txt STAGE=P95_v1_in_ssl_stageA CAP=29 nohup setsid bash slurm/feed_lines.sbatch > /home/infres/yinwang/CS_QMI/slurm_logs/feed_local_P95_A.out 2>&1 &
# results: source slurm/common.sh; python -m vcs_ssl.summarize --stage P95_v1_in_ssl --output reports/P95_stageA_table.md
```

## Gate
CPU job **1014585** (`slurm/p95_cpu_gate.sbatch`, nodecpu10, 2026-09-29T20:16–20:57 UTC, 41 min) — **PASS** (gate_rc = 0; results `reports/P95_GATE_1014585/`, run dirs deleted).
- Test suites: `test_ssl_core` 29 passed, `test_integration` 40 passed, `test_vcs_ssl_v2` 14 passed, `test_vcs_ssl_p95` 15 passed (bounds, simplex weights and D(w)
  identity, noise scale and dot-product variance (2τ² + τ⁴)/d, counter-based noise reproducibility, JS / −J gradient equality at f = 0 with identical shift draws,
  state-dict round trips, config policy incl. "old configs keep their hash", 5 trainer smokes, refresh leaves encoder / projector parameters and BN buffers
  bit-identical and training RNG untouched while raising J, stop/resume equals the uninterrupted run for dictionary / noise / refresh).
- Frozen recipe (hpK 4v/B256 seed 0) first-step gradient norms, CPU gate vs the P35 GPU run: relative differences 9.5e-5 (encoder), 5.3e-6 (projector),
  5.3e-9 (critic) — the recipe path is unchanged.
- Variant smokes (real CIFAR-10, 4 views, B 256, 6 steps, CPU): all COMPLETED; median step time relative to the recipe smoke 0.98–1.05 (residual λ 0.5: 1.05;
  dictionary 1.00; js 1.02) — on CPU the ResNet forward dominates; the GPU ratio for the MLP / bilinear members will be measured by the first stage-A runs.
  J after 6 steps: recipe −0.29, residual 0.25/0.5 −0.003/0.109, dictionary 0.103 (w = 0.332/0.334/0.334, D(w) 0.21), noise 0.1/0.3 −0.31/−0.22 (empirical
  total RMS 0.3000 for τ = 0.3), js −0.087 (JS loss 2.008), refresh R = 1 0.046.  Refresh at the end of pseudo-epoch 1: (a, b) (5.00, 0.00) → (0.56, …),
  J on the refit sample −0.972 → 0.010, applied.  Stop/resume COMPLETED for dictionary and refresh; pilot evaluation of the dictionary smoke (critic hold-out
  with the dictionary critic in eval mode) ran: linear 43.84 / kNN 33.22 after 6 steps (code-path evidence only).

## Delivery block (Spec v2 §13.2, one per run)
```yaml
experiment_family: full_ssl
protocol_id: P95_v1_in_ssl
source_commit: <commit of the freeze>
source_basis: package_v1 (Research Plan §§4-6, Server Spec §§7, 9) + owner 2026-09-29
estimator: vcs_neural_residual | vcs_neural_dictionary | vcs_neural_noise | vcs_neural_refresh | js_matched_control
estimand: S (training objective J), S_sigma for the noise cells (noisy observation of z)
evaluation_readout: frozen-h linear probe + kNN (selection split); J, held-out J
loss_scale: 1 (J); JS control: balanced logistic, f-gradient matched at f = 0
reference_measure: mixture_equal
critic_class: <per cell>
gradient_routing: negative_right_detach
n_independent_units: 45000 fit images (selection 5000)
n_positive_pairs: B x 6 view pairs per step
n_negative_pairs: 8 B x 6 view pairs per step
split_manifest_hash: cifar10_dev45k_val5k (f819026a…)
noise_target_kind: none | observation_noise (tau in {0.1, 0.3})
noise_tau: <cell>
noise_sigma_coordinate: tau / sqrt(128)
status: <status>
```

## Disclosed limits
1. Stage A is single-seed; only stage B carries a verdict.  With 3 seeds and the recipe's seed sd 0.53, an improvement smaller than ≈ 1 point cannot pass
   the "improves" rule.
2. λ is fixed per cell (grid {0.25, 0.5}) rather than line-searched online (v1 §5.1's exact step uses a held-out TUNE set, which has no online analogue
   without a new data role).
3. The dictionary members are trained jointly with their weights inside SSL (the v1 dictionary was a fixed set of fitted members); D(w) is logged on
   training batches, not on a held-out TUNE set.
4. Refresh refits (a, b) on FIT images only (no selection data), keeps AdamW moments of a and b, and is applied only when J on the refit sample does not
   decrease; with R = 50 there are 15 refreshes, with R = 200 three.
5. The noise cells change the observed variable (critic reads a noisy z): they estimate S_σ during training, not S; evaluation reads clean h.
6. The JS cell is the matched logistic control on the same critic; its tuned optimum may differ (no separate lr / scale search).
