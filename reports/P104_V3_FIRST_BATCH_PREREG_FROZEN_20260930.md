# P104: package v3 first batch (G / U / N lines). Pre-registration **DRAFT** — FROZEN 2026-10-01T02:52:10Z before GPU compute (CPU gate 1016122 all suites green incl. validate_core 25/25; GPU smokes 1015979/1015980/1016095)

**Status: DRAFT.** Nothing is submitted. The eight 800-epoch runs start only after the owner approves; the launch lines are in `slurm/p104_lines.txt`.
**Owner instruction (2026-09-30):** "你来解压并看看这个文件夹中的计划，并执行" (unpack the folder, read its plan and execute it).
The package spec §1 adds these constraints:
- do not cancel, overwrite or restart P95 or P97–P102;
- ImageNet is not submitted;
- all new work uses the fit/selection split only;
- normal QOS only, never runfill.

## 1. Sources

| file | sha256 |
|---|---|
| `VCS_SSL_Performance_Research_and_Server_v3.zip` | `330c72eeebd48e0d7118c03f51410c4208459a709b2d5221de10d76e30089675` |
| `VCS_SSL_Server_Execution_v3.md` | `03184e876e5fa38fe2bb165c5a05def0c36161249abe9a63201757b9b78f7ec6` |
| `VCS_SSL_Optimization_Analysis_v3_CN.md` | `c87ba4a05f8e3b880a00e8a242f0dd1fb1bef406e4949fb72cfcf9628bb5d8a7` |
| `configs/experiments_v3.yaml` (logical list) | `c4c40d2d3cb2f0a1baba8a72f04cc8f8bcfdab5034cbbd062bc27b644cd40043` |
| `support/vcs_perf_core.py` | `7c5e5911ec729a116427b04c06f77c13880711c59a401c58acc6399df2472c84` |
| `support/validate_core.py` | `8f67c65db7958672024857a148f1d81fd5bbac117dad0aafb0a86b3b6a560bd3` |

- The package's evidence cut is `e466fd5`. The repo HEAD at build time was `6e709ba` (the P96 report).
- P104 code is uncommitted working-tree changes on top of HEAD, in `src/vcs_ssl/{config,critic,objectives,optim,diagnostics,train}.py`.
- Per the directive there is no commit and no push. The actual commit must be recorded before launch.

## 2. Base recipe and references

- **Base config:** `configs/cifar10_hpK_a5_views4_800ep_vcs_seed0.yaml`
  - 4 views, B 256, 800 epochs, AdamW 1e-3;
  - cosine critic, learned (a, b), initialised at (5, 0);
  - K = 8 cyclic shifts per view pair over the 6 view pairs;
  - negative right detach.
- **References, fit/selection split, final h, linear-val (%):**
  - **B0, P35 VCS:** 87.01 ± 0.53 (seeds 86.42 / 87.16 / 87.44); kNN 85.46.
  - **P41 SimCLR, 4 views, 800 epochs, standard augmentation:** 88.32 ± 0.30 (seeds 88.20 / 88.10 / 88.66). This is the §10.2 layer-1 comparator: same views and the same protocol.
  - **P90 strong-augmentation SimCLR:** 89.49 ± 0.36. This is only the comparison block for §10.2 layer 2.

## 3. The eight cells (seed 0, 800 epochs)

| ID | a, b | pairs | negative gradient | noise | critic class | config | sha256 (first 16) |
|---|---|---|---|---|---|---|---|
| G1 | fixed 1, 0 (buffers) | K8 cross-view | right detach | none | FixedCosineCritic | `cifar10_hpX_G1_views4_800ep_seed0.yaml` | `623c3dc0f819be83` |
| G3 | fixed 1, 0 | K8 cross-view | full | none | FixedCosineCritic | `cifar10_hpX_G3_views4_800ep_seed0.yaml` | `50621abbf944ba98` |
| U1 | learned, init 5, 0 | all view tokens | right detach | none | CosineCritic | `cifar10_hpX_U1_views4_800ep_seed0.yaml` | `85d9ec462355dd11` |
| N1 | learned, init 5, 0 | K8 cross-view | right detach | τ 0.3, R 4 (mean of per-draw −J) | NoisyCosineCritic | `cifar10_hpX_N1_views4_800ep_seed0.yaml` | `e482266f4463e5c1` |
| N2 | learned, init 5, 0 | K8 cross-view | right detach | τ 0.3, R 1, matched JS on the noisy logit | NoisyCosineCritic | `cifar10_hpX_N2_views4_800ep_seed0.yaml` | `879d35436280c698` |
| G2 | fixed 2, −1 | K8 cross-view | right detach | none | FixedCosineCritic | `cifar10_hpX_G2_views4_800ep_seed0.yaml` | `2ddb42ab65138eb3` |
| G4 | learned, init 5, 0 | K8 cross-view | full | none | CosineCritic | `cifar10_hpX_G4_views4_800ep_seed0.yaml` | `970ece56562b7451` |
| U2 | fixed 1, 0 | all view tokens | right detach | none | FixedCosineCritic | `cifar10_hpX_U2_views4_800ep_seed0.yaml` | `6b8c9a6a81cbfd02` |

- `configs/P104_SHA256.json` holds the full hashes and a per-config diff against the base.
- These are the only differences from the base:
  - the fields named in each row;
  - `run.stage: P104_v3_first_batch`;
  - `logging.checkpoint_epochs` gains epoch 20 (spec §9 asks for epochs 0/20/100/400/800);
  - the learned cells carry an explicit `cosine_bias_init: 0.0`, which is bit-identical to the recipe init (tested).
- The queue order G1, G3, U1, N1, N2, G2, G4, U2 follows spec §4. The order only allocates resources; no cell is dropped on early scores.

### Implementation notes

| topic | behaviour | source |
|---|---|---|
| **G** | a and b are buffers, so the critic has 0 trainable parameters and there is no critic optimizer group. `trainable_affine=false` is recorded explicitly. G3 and G4 use the reference `vcs_pair_loss`: both ends take gradients, with no ×2 compensation and no custom backward. | spec §5 |
| **U: tokens and pairs** | Tokens are view-major, [V·B, D]. P holds ordered distinct-token pairs of the same image, N_P = VB(V−1) = 3,072. Q holds ordered pairs of different images, including the same view index: N_Q = V·B·V(B−1) = 1,044,480. | spec §6 |
| **U: computation** | Row chunks of 256 accumulate sums, divided once by the global counts. Detach is implemented as `flat @ flat.detach().T`. | spec §6 |
| **U: noise refused** | The noisy critic refuses `score_matrix`/`embed`, and the config policy refuses U with τ > 0. | spec §6.4 |
| **N: sampling** | One shift draw per view pair, then R noise draws on the same features and pairs. Each draw uses fresh noise per pair occurrence and side, from the critic's counter-based, checkpointed stream. The loss is the mean of the per-draw losses. | spec §7.1–7.2 |
| **N: R = 1** | R = 1 equals the P95 definition, tested exactly (value and gradients). | spec §7.1 |
| **N: evaluation** | `heldout_J_noisy` uses R_eval 16 from a separate seeded evaluation generator. Encoder, projector and BN stay in eval mode. The clean `heldout_J` keeps its name. Clean and noisy gates are reported as 1 − ½(E_P T² + E_Q T²). | spec §7.3, §12 |
| **Cost counters** | positive pairs, negative pairs, critic pair evaluations (including noise repeats), encoder updates, critic updates, refresh seconds (counted separately). They are checkpointed and restored on resume. New field `steady_state_views_per_s_actual` counts V·B. | spec §9.3, §12 |

## 4. Gate and smoke

### CPU gate `slurm/p104_cpu_gate.sbatch`: job 1016122, `reports/P104_GATE_1016122/` (gate_rc = 0)

**Outcome:** every check passes.

| check | result |
|---|---|
| `test_ssl_core` | 29 passed |
| `test_integration` | 40 passed |
| `test_vcs_ssl_v2` | 14 passed |
| `test_vcs_ssl_p95` | 15 passed |
| `test_vcs_ssl_p100` | 8 passed |
| `test_vcs_ssl_p104` | 30 passed; covers every row of the spec §8 table |
| package `validate_core.py` | passed, 25 checks |
| frozen-recipe first-step gradients, CPU vs the P35 GPU run | max relative difference 8.5e-5 (encoder 8.5e-5, critic 6.0e-6) |
| config-hash invariance | 300 existing configs; 0 hashes changed by the P104 code (HEAD code vs P104 code) |
| 8 cell smokes (6 steps at B 256 × 4 views) | all COMPLETED |
| stop/resume (G1, U1, N1) | all COMPLETED |
| diagnostics-script CPU smoke (G1, U1, N1) | passed; N1 R1/R4 conditional gradient-variance ratio 3.7 |

**Earlier gate runs:**
- **Job 1015978**, three failures:
  - The P95 test caught that my relaxed JS rule had stopped rejecting JS with noise. Fixed: the P95 rule is restored, and the N2 marker `noise_repeats` is now required.
  - The CPU-node init hash differs from the GPU-run hash. This is environmental: on every GPU node, the init hashes equal P35. The CPU check is now informational, and the GPU smoke checks the hashes.
  - The old invariance check compared recorded run hashes. 229 of them differ already at HEAD through earlier schema growth, so the check now compares HEAD code against P104 code.
- **Job 1016097:** one test needed the N2 marker. Fixed.

### GPU smoke `slurm/p104_gpu_smoke.sbatch`

- Jobs 1015979 and 1015980: all 8 cells.
- Job 1016095: G4, N1 and U1 again, with a second uninterrupted run as the GPU run-to-run baseline.
- All three jobs ran on an A100-SXM4-40GB node, with 100 real-CIFAR steps at B 256 × 4 views, in pseudo-epochs of 50.
- Each cell was also run as stop at 50, then resume to 100.

| cell | s/step (steady) | vs recipe (same node) | peak alloc / reserved MB | J at step 99 | grads finite | critic pair evals / step | projected 800 ep, A100 | projected 800 ep, fast node ¹ |
|---|---|---|---|---|---|---|---|---|
| recipe | 0.3438 / 0.3440 | 1.00 | 7569 / 12268 | 0.257 / 0.256 | yes | 13,824 | 13.4 h | 4.5 h |
| G1 | 0.3430 | 1.00 | 7569 / 12268 | 0.315 | yes | 13,824 | 13.3 h | 4.5 h |
| G3 | 0.3440 | 1.00 | 7569 / 12274 | 0.336 | yes | 13,824 | 13.4 h | 4.5 h |
| U1 | 0.3420 | 0.99 | 7569 / 12286 | 0.276 | yes | 1,047,552 (75.8×) | 13.3 h | 4.5 h |
| U2 | 0.3422 | 1.00 | 7569 / 12278 | 0.320 | yes | 1,047,552 | 13.3 h | 4.5 h |
| N1 | 0.3816 | 1.11 | 7569 / 12296 | 0.255 | yes | 55,296 (4×) | 14.8 h | 5.0 h |
| N2 | 0.3484 | 1.01 | 7569 / 12272 | 0.359 ² | yes | 13,824 | 13.5 h | 4.6 h |
| G2 | 0.3414 | 0.99 | 7569 / 12268 | 0.456 | yes | 13,824 | 13.3 h | 4.5 h |
| G4 | 0.3447 | 1.00 | 7569 / 12274 | 0.310 | yes | 13,824 | 13.4 h | 4.5 h |

**Table notes:**
- ¹ Scaled by the step-time ratio from the P95 τ0.3 seed-0 run on node61, which measured 0.1151 s/step. This is an estimate, not a measurement.
- ² N2's logged J is computed from T = tanh of the noisy logit; its training loss is the JS loss.

**Findings:**
- Every cell encodes 1,024 views per step. With 175 steps per epoch, 800 epochs is 140,000 steps and 143.4 M encoded views, the same in all cells.
- On the A100, step time does not follow the scorer count: U scores 76× more pairs than the recipe, but the step is encoder-bound. N1 costs +11% per step from its R = 4 scorer passes and noise generation.
- **Stop/resume:**
  - Cost counters are identical in all 8 cells, and the N-cell noise counters are identical.
  - The resumed J matches the uninterrupted J to 4e-5 to 2e-2 at step 50 and to 2e-4 to 3e-3 at step 99.
  - On CPU, resume is exact to 1e-5 (unit tests).
  - On GPU, parameter states are not bit-exact after 50 steps. The run-to-run baseline from job 1016095 shows how much of that difference comes from GPU kernel nondeterminism alone: two uninterrupted runs differ by about as much as stop/resume does, so the resume difference is within GPU noise.
  - J differences, run vs re-run, at steps 50 / 99: G4 5.0e-3 / 1.1e-2, N1 1.0e-3 / 9.9e-3, U1 1.2e-2 / 3.8e-3.
  - J differences, resume vs run, at steps 50 / 99: G4 4.6e-4 / 7.2e-3, N1 3.8e-3 / 2.4e-3, U1 1.7e-2 / 1.5e-3.
  - That job ran on node59, RTX PRO 6000 Blackwell, and measured the fast-node step times directly: recipe 0.1140 s, G4 0.1147 s (1.01×), U1 0.1129 s (0.99×), N1 0.1282 s (1.12×). So 800 epochs take about 4.4 h for the recipe and about 5.0 h for N1..
- **Init hashes:** on the GPU nodes, the recipe's encoder, projector and critic init hashes equal P35 seed 0. The fixed-critic and noisy cells share the encoder and projector hashes; their critic hash differs by design (buffers and noise seed).

## 5. Readings, fixed before any result (spec §10)

1. **Selection.** When all eight cells are done, rank the VCS cells (not N2) by final-h linear-val on the selection split. Pick up to two candidates.
   - kNN is shown next to linear but does not rank.
   - If two cells are within 0.1 points on linear, prefer the one with lower measured cost (training seconds, then critic pair evaluations), then the simpler one.
2. **Freeze.** After selection, all hyperparameters are frozen. Seeds 1 and 2 never change a configuration.
3. **Required controls:**
   - A fixed-scale winner (G1, G2 or U2) needs a same-initialisation learned-(a, b) control before any mechanism claim. For G1, that is learned (a0, b0) = (1, 0) with the same routing and pairing (spec §4.1).
   - A noise winner (N1) needs N2 on at least 3 seeds.
   - N1 and the P95 τ0.3 R1 runs must be compared on the same R_eval = 16. See open decision O2.
4. **Confirmation.**
   - Layer 1: candidate seeds 0/1/2 against P41 4-view SimCLR seeds 0/1/2, with the same standard augmentation and protocol.
   - Layer 2: seeds 3/4 on both sides, then both sides on the identical strong-augmentation block.
   - Report per-seed values, paired differences, mean, SD and interval.
   - Do not drop a reproducible small gain for missing an arbitrary 1-point threshold. Do not call a selected seed-0 extreme a win.
   - Layer 3, CIFAR-100, uses the P98 probe rule.
5. **Early stopping.** None on J, rank, kNN or a learning threshold. A run pauses only for non-finite values, a data or implementation error, or an infrastructure fault, and resumes from a legal checkpoint with the redone steps recorded.
6. **Test set.** The official test set stays closed.

## 6. Diagnostics (spec §9; `scripts/p104_diagnostics.py`, post hoc on epochs 0/20/100/400/800)

- **§9.1 critic:**
  - a, b and −b/a;
  - cosine mean and quantiles for positives and negatives;
  - T distribution and saturation;
  - clean gate (from T²);
  - regression residuals 1 − T on P and −1 − T on Q.
- **§9.1 geometry:** h, p_raw and z separately, each with mean-vector norm, centred covariance spectrum, effective rank, alignment and uniformity. Computed on 5,000 selection images, eval mode, fixed RNG.
- **§9.2 gradients:**
  - G_pos and G_neg: norms, inner product and cosine, plus the total norm. These are gradients of the minimised loss with respect to the encoder and projector, taken on one fixed batch of fit images with one shift draw.
  - Input-gradient norms per view, for the images and for z.
  - The ratio of the aggregate gradient norm to the sum of per-anchor norms, over 32 anchors.
  - BN runs in train mode on a throw-away copy.
- **§7.4 (N cells):** conditional variance of dL/dz under noise only, R = 1 vs R = 4, with a Welford estimate over 32 draws. In the CPU smoke the R1/R4 variance ratio was 3.7, against 4 in theory.
- None of these select checkpoints or stop runs.

## 7. Result row (spec §13), one per run

```yaml
run_id: P104_<ID>_views4_800ep_seed0
logical_variant: G1|G2|G3|G4|U1|U2|N1|N2
code_commit: <actual at launch>
objective: original_vcs_J            # N2: matched_js
estimand: S                          # N1, N2: S_sigma
critic_affine: {mode: fixed|learned, a_init: <a>, b_init: <b>}
negative_routing: negative_right_detach|full
pair_scope: cross_view_K8|all_view_tokens
noise: {tau: 0.3|0, draws_train: 4|1|null, draws_eval: 16|null}
pretrain_seed: 0
epochs: {planned: 800, completed: <actual>}
linear_feature: h_before_projector
selection_linear_top1: <result or null with reason>
selection_knn_top1: <result or null with reason>
clean_J: <heldout_J at 800>
noisy_J: <heldout_J_noisy at 800, N cells; null otherwise>
encoded_views: <summary.cost.encoded_views>
critic_pair_evaluations: <summary.cost.critic_pair_evaluations>
training_seconds_total: <summary.cost.training_seconds_total>
peak_memory_mb: <summary.peak_allocated_mb>
status: completed|failed|interrupted
```

## 8. Disclosures

- **U held-out J:** for U1 and U2 it is still the existing cross-view K diagnostic, the same as for every other cell. It is not the all-token J used in training, and the two are not interchangeable.
- **Checkpoint schedule:** adding epoch 20 changes the config hash relative to a base-plus-cell-fields-only config. Training does not change.
- **Bias init on learned cells:** the explicit `cosine_bias_init: 0.0` is bit-identical to the recipe init (unit test).
- **N2 policy:** N2 is JS on the noisy logit. The P95 rule that rejects JS with noise still applies unless `critic.noise_repeats` is present, which marks the P104 N2 control.
- **Historical throughput (§12):** the old field `steady_state_views_per_s` assumed 2 views, and `views_per_image` in the historical run manifests reads 2 for the 4-view runs. The real encoded-view throughput of those runs is 2× the logged value; for example, P35 seed 0 logged 1,494.8 views/s, which is 2,989.6 in reality. Historical summaries are not rewritten; the new field `steady_state_views_per_s_actual` carries V·B.
- **Config-hash drift:** 229 of 292 finished runs have recorded config hashes that no longer match re-resolution, at HEAD as well. That drift comes from earlier schema growth. The P104 code changes no config hash of any existing config; the gate checks this against the HEAD code.

## 9. Open decisions for the owner

- **O1: launch approval.** Approve the eight lines in `slurm/p104_lines.txt`: single 23 h jobs, RTX6000PRO/H100, normal QOS, no chain needed.
- **O2: matched N1 comparison.** A post-hoc noisy-J at R_eval = 16 is needed for the existing P95 τ0.3 R1 runs so N1 can be compared on the same R_eval. It is an evaluation-only job; approval needed.
- **O3: fixed-scale control timing.** If a G or U2 cell wins, schedule the same-initialisation learned-(a, b) control (spec §4.1) as soon as selection is done, or after the confirmation seeds.

## Decisions taken at the freeze (main session, 2026-10-01T02:52:10Z)
1. Launch the eight seed-0 units of `slurm/p104_lines.txt` (normal QOS, RTX6000PRO/H100, single run_unit jobs; projected ≤ 15 h on A100, ≈ 4.4–5 h on RTX6000PRO).
2. Code committed before launch (this commit); every run records its commit.
3. Fixed-scale control: if a fixed-affine cell (G1, G2, G3, U2) is among the ≤ 2 selected candidates, its same-initialisation learned-a,b control is submitted
   together with the seed-1/2 confirmation runs (spec §4.1, §10.1), before any mechanism statement.
4. N1 vs P95 τ 0.3: an evaluation-only addendum computes noisy-J with R_eval = 16 on the P95 τ 0.3 checkpoints after the first batch, so the two are compared
   with the same evaluation repeats (spec §7.3).  It does not gate the launch.
