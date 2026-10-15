# V9-N2 adapter notes — what was built, every deviation from the official code, and what is still stubbed (fork D, 2026-10-16)

Scope: the trust-assisted weighted-training adapter of package docs/05 on the SGR host, developed on synthetic fixtures (no NCR data, no official
environment yet — those are fork A's deliverables).  Code: `src/v9/n2/` (`roles.py`, `noise.py`, `sgr_host.py`, `teacher.py`, `weights.py`, `train.py`,
`eval.py`, `strict.py`, `pipeline.py`, `data.py`, `fixtures.py`), driver `scripts/v9/n2_run.py`, tests `tests/v9/test_n2.py`, SLURM `slurm/v9/n2_unit.sbatch`,
patches `v9/patches/{ncr,l2rm}_anchor_assisted.diff`, draft protocol `reports/V9/V9_N2_PROTOCOL_DRAFT.yaml`.

## 1. Host (`sgr_host.py`) — vendored SGR path, two disclosed deviations
Source: NCR's `NCR/model.py` @ 60256ba (= SGRAF `model.py` @ 50d0c6f with the Python-3 fix), Apache 2.0, attribution in the file header.  Kept: EncoderImage,
EncoderText, VisualSA, TextSA, GraphReasoning, SCAN_attention, EncoderSimilarity (SGR only, 3 steps, sim_dim 256, sigmoid output), the hinge loss.  Deviations:
1. `pack_padded_sequence(..., enforce_sorted=False)` — the official collate sorts the batch by caption length; outputs per sample are identical (test).
2. Bi-GRU halving by integer division (the upstream `/ 2` is a Python-2 remnant; NCR already fixed it).
3. Dimensions are constructor parameters (num_region, embed, sim_dim) so the CPU gate can run at reduced size; the official values are the defaults.
`tests/v9/test_n2.py::test_vendored_host_matches_official` loads the official `model.py` (the read-only clone) with the same weights and checks the
similarity matrix to 1e-6.  The per-anchor hinge (`hinge_per_anchor`) reproduces `ContrastiveLoss(max_violation=True)` exactly as a sum (test).

## 2. Roles and noise (`roles.py`, `noise.py`)
- Anchor set A = 10 % of training images, drawn before noise from `np.random.default_rng([seed, 9002])`; 40/30/15/15 by largest remainder; integer counts
  and sha256 of every id list in `roles.json`; captions follow their image; minimum role sizes raise "insufficient data" (never filled from val / test).
- Mismatch = the NCR / L2RM construction (`PrecompDataset`: a random subset of captions gets its image indices permuted among the subset) restricted to U,
  so donors are U images only and A stays clean.  **Deviation from the official protocol, intended by docs/05 §2:** the mapping differs from the
  official noise index (`noise_index/f30k_precomp_0.4.npy`, which permutes across the whole train split); the official index is used only for the A-table
  native reproduction.  `realized_rho` ≤ design (a permutation can map a pair to its own image) and is recorded.
- The trainer-facing table (`*.observed_pairs.json`) carries `observed_image_id` only; the labels live in `eval_only/*.eval_only.json`; `load_observed`
  refuses eval-only files and label-bearing rows; `WeightedStudent` refuses rows with label keys (tests).

## 3. Teacher and scores (`teacher.py`)
The common teacher is the host trained for 5 full epochs on the A-teacher-warmup pairs only with the warm-up loss convention of NCR / L2RM
(`mode='warmup'`: hinge with the MEAN over negatives), then frozen; it is one per (dataset, rho, seed) and shared by all methods (also by NCR+A / L2RM+A as
their warm-up initialisation is a separate, official step — see §8).  Pair score s = the SGR similarity of (observed image, caption) in eval mode
(BatchNorm in VisualSA uses running statistics, so s is batch-independent; test `test_similarity_matrix_equals_pairwise_scores`).

## 4. Critic, rho and weights (`weights.py`)
- Critic class cos-2 on the single teacher score: natural logit f = 2 (a s + b), a = softplus(a_raw) > 0, T = tanh(f / 2); VCS = 1 − J, matched BCE = balanced
  log loss on the same f (package `reference/torch_losses.balanced_fit_loss`).  Fit: full-batch Adam over lr {1e-2, 3e-2, 1e-1} × 4 deterministic starts,
  selection on A-critic-select by the common J (native loss recorded).  N1's fitters (`src/v9/quant.py`: KDE, RuLSIF α = ½, Nyström) are called through
  the same interface when present; otherwise the method raises (no silent fallback).
- Q for critic fitting / selection / reference moments: within-role re-pairing by a random permutation + cyclic shift, same-image pairs rejected
  (`repair_within_role`).  This is a **conditional Q** (docs/03 §A: "拒绝同图donor的Q是条件化抽样分布"), stated in every table.
- rho: `reference/core.rho_from_means` on (a, b, c); status `point_estimate` / `outside_parameter_space` / `unidentified_at_numerical_tolerance`; the
  projected value is used for weights, the raw value is kept; an unidentified cell gets w = 1 and a flag.  Interval: the package's Hoeffding inversion
  on image-level cluster means after `role_disjointness` passes — note that the P and Q clusters of the reference role are the same images in two
  roles (Q is their re-pairing), so the interval is reported as a diagnostic, not as an independent-cluster guarantee.  rho is estimated on the
  unlabeled observed U pairs (transductive); the protocol asks for a second interval on an independent batch — not implemented yet (TODO for the freeze).
- w = `reference/core.clean_posterior(T, rho)`, computed on the clean-P-vs-Q critic as docs/03 §D requires; w ∈ [0, 1], total mass, ESS and clipping
  fraction recorded.  RankMass = `rank_mass_weights(raw score, N3's total mass)`; PlainFilter = raw score ≥ the 5th percentile of SELECT true pairs
  (non-main row).  The weight file is CSV `pair_id,w,score,rho_hat,fit_hash,reference_hash` + `.lock.json` (sha256); `read_weight_lock` refuses a
  tampered file (test).

## 5. Weighted training (`train.py`)
`L = (Σ_A ℓ_i + Σ_U w_i ℓ_i) / (n_A + Σ_U w_i)`, minibatch `mean(v_i ℓ_i) / mean_v_full_pool` with the normaliser computed once over the pool and written
to `run_manifest.json`; ℓ_i = row + column hinge of pair i — **HOST_LOCK 57366c5 (fork A, timing job 1042033): SGRAF with hardest in-batch negatives from step 0 collapses under torch 2.9 (loss → 2×128×0.2, dev R@1 0.1), so every SGR-based row runs NCR's warm-up loss (mean over in-batch negatives, both directions) for `warmup_epochs` = 5 (parsed from HOST_LOCK) and only then the max-violation hinge; the reliability weights apply in both phases and the phase is logged per epoch** (`tests/v9/test_n2_ext.py::test_student_warmup_phase_then_hard_negatives`); v detached (test: no gradient path through v); w never
touches logits; low-w pairs remain batch members (negative pool unchanged); unnormalised U risk, batch mass and batch ESS logged.  Adam lr × multiplier,
×0.1 at lr_update epochs, grad clip 2; per-epoch checkpoints with model / optimiser / RNG states; resume at the epoch boundary (test: the resumed
run reproduces the uninterrupted run's epoch-1 loss to 1e-5); dev selection by official rSum, ties → earlier epoch; `done.json`.  `formal=True` refuses
to start while HOST_LOCK has TO_VERIFY fields.  Single-sample batches are dropped (BatchNorm), as NCR does.  **Not an unbiased estimator** of the
population objective (self-normalised, in-batch negatives): the strict-interface check (§6) is reported beside the full-task results.

## 6. Strict-interface check (`strict.py`, docs/05 §5 last paragraph)
Fixed 1-D teacher representation with Gaussian P/Q score laws (the cos-2 class contains the true logit), loss ℓ(s) = (1 − s)²: oracle weights recover
(1 − ρ) E_P ℓ to < 5 %, the fitted VCS posterior with the moment ρ̂ to < 12 % (test); the richer-student counter-example (loss reads a variable
correlated with C given s) leaves a > 5 % gap — the covariance term of docs/03 §E does not vanish.

## 7. Evaluation (`eval.py`)
Official semantics (captions 5i..5i+4 ↔ image i; I2T any-caption hit; T2I image hit), R@1/5/10, rSum, mR; COCO 5K main + 5×1K folds, Flickr 1K;
ties: fixed id order (stable argsort) primary + seeded random-tie sensitivity (the official code's `np.argsort` is unstable quicksort — a deviation we
make explicit rather than inherit); cross-checked against the package `retrieval_recall` (test).  Dev = NCR's first 1 000 images of the dev file.

## 8. N1 NCR+A and N2 L2RM+A — patches, driver and degradation check (updated 2026-10-16 after the owner's decisions)
The official training loops are scripts (module-level option parsing; NCR co-trains two SGRAF nets and evaluates their averaged similarities; L2RM
adds a cost network, an image queue and OT rematching), so they are driven as patched scripts, not imported.  Delivered:
- **Patches** `v9/patches/{ncr,l2rm}_anchor_assisted.diff`, generated reproducibly by `scripts/v9/n2_make_patches.py` from explicit string edits on
  pristine copies (NCR @ 60256ba: run.py + data.py + co_train.py, 11 hunks; L2RM @ a766c22: opt.py + data.py + main_L2RM.py, 16 hunks, CRLF preserved).
  Two new options `--observed_index_file` / `--anchor_mask_file`; loaders take the V9 observed mapping (clean labels set to −1, never exposed);
  after the GMM (NCR) / BMM (L2RM) split every anchor pair is labeled clean with probability 1 (NCR, both nets) / forced into the correct set
  (L2RM).  Everything else is official.  `git apply --check` passes on fork A's clones (`code_v9/2021-NeurIPS-NCR`, `code_v9/L2RM`) and on the
  read-only reference clones (test `test_patches_apply_to_clones`).
- **Driver** `scripts/v9/n2_official.py`: `export` writes `v9_observed_t2i.npy` (int64 per caption, official order) and `v9_anchor_mask.npy` (bool)
  from the shared stage (asserts anchors clean), `launch` copies the official clone into `<unit>/code/<baseline>`, applies the patch (hash recorded),
  and runs the official script in fork A's env (`envs/v9_sgr` for NCR, `envs/v9_l2rm` for L2RM) with the HOST_LOCK schedule (NCR f30k: 40 epochs,
  lr_update 30, warm-up 5; L2RM: its train script, 35 / 15 / reg 0.01 / warm-up 5), lr × multiplier, `--noise_ratio 0` (the V9 mapping carries the
  noise) — or `--anchor-off` for the native path with the official DECL/RCL index (A-table reproduction, degradation check on real data);
  `collect` runs the official `evaluation.evalrank` on the official test split once and writes `result.json` / `done.json` in the V9 unit-dir layout
  (`test.primary` with the same keys as the V9 evaluator; evaluator and tie rule named).  Dry runs verified on the fixture export.
- **Degradation check** `scripts/v9/n2_degradation.py` (owner 2026-10-15): patched vs native copies, anchor intervention off, official noise index →
  same t2i index, same clean labels, same items, same eval-mode loss on a fixed mini-batch through the untouched model, same selected indices with
  the anchor block skipped.  Passes on the fixture (see the report / `tests/v9/test_n2_ext.py::test_degradation_check_fixture`); on real data it must run
  under `v9_sgr` (nltk tokeniser) and `v9_l2rm` (pomegranate for the BMM) before the first N1 / N2 unit — a launch condition in the protocol.
- Still needed before a formal N1 / N2 unit: fork A's one-epoch timing, the real-data degradation run, and the owner's budget confirmation.
- NCR's final model is two SGR nets averaged at evaluation: its row is labelled "SGR ×2 (NCR co-training)" in every table.

## 9. Real-data loader (`data.py`) — unverified until fork A's data exists
Reads the SCAN / NCR precomp layout (`<split>_ims.npy`, `<split>_caps.txt`, `<name>_vocab.json`), tokenises with nltk exactly as the official code, caches
token ids with the vocab hash.  COCO 2014/2017 mapping, the `testall` 5K file and `dev` size are TO_VERIFY on the real files (DATA_AUDIT).

## 10. Precheck gate, independent-batch interval and the Flickr-only block (owner 2026-10-15)
- `src/v9/n2/precheck.py` + `n2_run.py precheck`: one table per (dataset, rho, seed) — teacher clean-vs-re-paired score distributions on A-critic-select
  (quantiles, AUROC, overlap; refuse the block if AUROC > 0.999 or < 0.55, thresholds in the protocol), every method's U-weight distribution (ESS, mass,
  shares at 0 / 1), pairwise method differences (mean |Δw|, Spearman, share identical to 1e-6), the per-sample loss coefficients after the fixed-pool
  normaliser with the merge rule (implementation-identical only if coefficients equal to 1e-12 on every row AND order / RNG derive from the same seed —
  the trainer makes this provable by deriving both from the seed; otherwise both run), the anchor roles and their isolation from dev / test.
  `run_unit(formal=True)` refuses without an allowing precheck file.
- ρ̂ interval (§05.3 step 3): U images are split 80 / 20 by image (seeded); the point estimate uses the 80 % part (transductive), the interval is reported
  on the held-out 20 % (`record.rho.independent_batch`), weights are computed for all U pairs.
- Block: Flickr30K only (48 formal + ≤ 48 tuning); COCO is a later decision; protocol draft 2.

## 11. Cost and units
Not measured on real data.  Fixture smoke on CPU: see the gate log.  Formal plan: 96 units grouped by (dataset, rho) cell per GPU job, resumable; tuning
seed 917 with lr multipliers {0.5, 1, 2}; one teacher per (dataset, rho, seed) shared by all eight methods.  The planning figure "NCR f30k 40 epochs" is
from the official README (2020 hardware) and is not a budget until fork A's one-epoch timing exists.

## 12. Owner items (not decided here)
- Data download of the NCR release (precomp features + vocab, ~ tens of GB) and the Flickr30K / COCO licence basis for research use.
- Whether NCR+A / L2RM+A run from the patched official scripts (recommended) or are re-implemented inside this adapter (not recommended: fidelity).
- GPU share for V9 next to V8 (package: V8 keeps its reservation).

## 13. Execution notes (real-data runs, 2026-10-15/16)
1. **Gate 2** (CPU job 1042124, commit 8a994f3): 13 passed / 2 skipped (the official-clone comparisons skip on compute nodes; both pass on the login node), fixture smoke ok, exit 0.
2. **Degradation check on real data** (CPU job 1042126): NCR under `envs/v9_sgr` and L2RM under `envs/v9_l2rm`, Flickr30K precomp, official `noise_index/f30k_precomp_0.4.npy`, anchor intervention off: every flag true for both (`reports/V9/V9_N2_degradation/degradation_f30k_{ncr,l2rm}.json`, commit fe1ecc2) — the patched scripts reproduce the native path.
3. **Precheck 1042125 failed before any computation**: `yaml.safe_load(reports/V9/HOST_LOCK.yaml)` raised a ScannerError (unquoted prose scalars with ': ' in fork A's file); no GPU work, no results.  Two fixes landed in parallel: the coordinator quoted the HOST_LOCK scalars (commit c94196d) and resubmitted with the old code (job 1042132); this fork added a tolerant HOST_LOCK loader (`teacher.load_host_lock`: YAML or regex, per-data-name schedule, pending fields → `to_verify`; commit 127dd18) and resubmitted (job 1042133).  Job 1042132 was cancelled by this fork two minutes in: its code snapshot (hash 146ef88) still routed the dict-valued `native_epochs` / `lr_update` into `HostConfig.from_host_lock` and would have failed at the teacher's first epoch, and the two jobs were writing the same shared-stage directories.  1042133 is the precheck of record; its results are the gate record in the protocol draft.
4. `V9_N2_PROTOCOL_DRAFT.yaml` did not parse either (prose scalars); quoted (content unchanged) and covered by `tests/v9/test_n2_ext.py::test_n2_yaml_files_parse` together with HOST_LOCK and the frozen N2 protocol once it exists.
5. **HOST_LOCK 57366c5 adopted** (fork A's finding from timing job 1042033): the weighted student now runs the 5-epoch warm-up (mean) hinge before hard negatives (§5), `HostConfig.warmup_epochs` is read from HOST_LOCK, the protocol draft cites 57366c5 in the host section, and `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` is set for the official NCR / L2RM trainer and evaluator subprocesses (`n2_official.py`) and in `n2_unit.sbatch`.  The precheck of record 1042133 is unaffected: it trains only the teacher (already the warm-up loss, 5 epochs) and no student.  Cost note: the warm-up epochs are part of the 40-epoch schedule (epochs 0–4), not extra epochs.
