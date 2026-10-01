# Pre-registration — P110: package v4 modules V1 (label efficiency + transfer), V2 (corruption / compression tolerance, development phase), V3 (audit-assisted selection on a controlled spurious-cue task), 2026-10-01 — FROZEN 2026-10-01T21:26:20Z before the V jobs (CPU gate 1017571: 10 tests, smokes, full-size V3 check)

Status: FROZEN 2026-10-01T21:26:20Z (main session; owner 2026-10-01 "V1 / V2 / V3 现在准备这三个，当现在提交的实验结束后提交").  The main session freezes it (rename to `*_FROZEN_*`) and registers `slurm/p110_lines.txt` with the orchestrator so
that it is submitted only after every currently submitted P104 / P107 / P108 / P109 job has left the queue (owner 2026-10-01: "V1/V2/V3
现在准备这三个，当现在提交的实验结束后提交").  Source: `VCS_Next_Experiment_Plan_v4_CN.md` §V1–V3.  No pretraining; frozen encoders only.
Code: `src/vcs_vtask/{common,corruptions,v1,v2,v3}.py`, `scripts/p110_v.py` (runner, bundle pattern, resumable), `scripts/p110_aggregate.py`,
`slurm/p110_unit.sbatch` (one GPU job runs a list of encoders × modules), `slurm/p110_gate.sbatch`, `tests/test_p110.py` (10 tests).

## Encoders (read-only; only COMPLETED runs with `epoch_800.pt`)
Recipe VCS `P35_vcs_a5_views4_800ep_seed{0,1,2}`, tuned SimCLR `P41_simclr_views4_800ep_seed{0,1,2}`, `P104_G2_views4_800ep_seed{0,1,2}`,
`P104_U2_views4_800ep_seed{0,1,2}` (all share the CIFAR-10 dev45k / val5k manifest — checked).  The P107 winners and confirmation seeds are appended
by the main session as a new run list (same code, same outputs directory, skips finished files).

## V1 — label efficiency (CIFAR-10) and frozen transfer (CIFAR-100)
- Frozen h (512-d encoder output, the run's own clean normalisation).  Label subsets of the 45k FIT images: 1 % and 10 % (three class-stratified
  draws each) and 100 %; index manifests are seeded by (dataset, fraction, draw) only, hence **identical for every encoder** (their sha is stored).
- Readout: the recipe linear probe (SGD momentum 0.9, batch 256, cosine to 1e-3, 100 epochs, final epoch, PyTorch default init) with (lr, wd) chosen
  from {0.03, 0.1, 0.3} × {0, 5e-4} on an **inner 80 / 20 split of the labelled subset only**, then refit on the subset.  The recipe's fixed
  (lr 0.1, wd 0) probe is reported alongside.  Evaluation: the 5k selection split.  This is **label efficiency** of the frozen representation, not the
  estimator's unlabelled sample efficiency.
- Transfer: the same CIFAR-10-pretrained encoder on CIFAR-100 (local training partition, P91 dev45k / val5k manifest; test partition not opened),
  10 % (three draws) and 100 % labels, same probe protocol.  Reported as **transfer**, never mixed with P91's from-scratch CIFAR-100 SSL.
- Reading (descriptive): per encoder family, mean ± sd over seeds × draws at each label fraction; the gap of each family to recipe VCS and to SimCLR
  per fraction, with per-seed values.  No threshold; "not different" is not claimed without an equivalence margin.

## V2 — corruption / compression tolerance, development phase
- Head: recipe probe (fixed hyper-parameters) on clean h of all 45k FIT images.  Evaluation images: the 5k selection split, clean and under 6 fixed
  development corruptions × 5 severities (Gaussian noise sd 0.04–0.26; Gaussian blur radius 0.5–2; JPEG quality 80–10; contrast 0.75–0.2;
  brightness +0.1–0.5; pixelate 28–12 px).  Every random draw is seeded by (family, severity, image uid): a corrupted image is a function of its uid.
- Per cell: accuracy, relative drop 1 − acc / acc_clean, prediction consistency (same prediction as on the clean image); mean corruption accuracy
  (mCA) over the 30 cells; 95 % CIs by bootstrap over base images (all 31 versions of an image resampled together = clustered by image).
  Models at different clean accuracy are compared on both absolute accuracy and relative drop; neither is picked post hoc.
- **Standard CIFAR-10-C** is present locally (`/projects/EEG-foundation-model/yinghao/FMCA-AV/robustness/cifar10-c`, 19 corruptions × 5 severities
  × 10k).  It is built from the **official CIFAR-10 test images**: the runner refuses it without `--authorised-official-test-c10c`, its lines are in
  a separate file `slurm/p110_lines_cifar10c.txt` marked "needs owner authorisation", it is **not** registered with the orchestrator, and it is never
  used for any choice (reporting only, after the configuration is frozen).  The official test set has been read by earlier protocols (P68, S2);
  this is disclosed.

## V3 — does a conditional-dependence audit improve model selection? (controlled)
- Frozen h; a class-specific cue planted on the image **before** the encoder, colour palette[k], k ∈ 0..9: `tag` (8 × 8 corner patch) and
  `colour` (25 % global blend).  Base images (FIT split, fixed permutation, disjoint): POOL-TRAIN 20 000, CLEAN-VAL 5 000, AUDIT 5 000,
  REVERSAL-TEST 10 000.  POOL-TRAIN: class c shows its own colour with probability ρ ∈ {0.8, 0.95}, otherwise a fixed random other class's colour.
  CLEAN-VAL: no cue.  REVERSAL-TEST: every image shows the colour of class (c + 1) mod 10.  AUDIT: binary N, P(N = 1 | Y) = ½ ± 0.3 by class parity;
  N = 1 → shifted colour (c + 1), N = 0 → own colour.
- Model pool (fixed, 20 members): head ∈ {linear, MLP 512→256→10} × weight decay ∈ {0, 5e-4} × cue-randomisation augmentation a ∈ {0, .25, .5,
  .75, 1} (Adam 1e-3, 30 epochs, batch 512, inputs standardised on POOL-TRAIN).  5 replicates per (encoder, cue, ρ) with fresh cue draws and seeds.
- Audit statistic of a member on its logits (lower = less dependence on N given Y), fitted on AUDIT-FIT 2 000, read on AUDIT-EVAL 2 000,
  within-class negatives from AUDIT-POOL 1 000: VCS closed-form critic J and exact JS value (P105 exact solvers) on the class-conditional
  linear class z ⊗ onehot(Y); class-conditional HSIC with per-class bandwidth (comparator).
- Rules, fixed now: **A** = highest CLEAN-VAL accuracy; **B_k** = among members within δ = 1.0 point of the best CLEAN-VAL accuracy, the smallest
  audit statistic k ∈ {VCS, JS, HSIC}; **C** = expected outcome of a uniformly random eligible member (does the audit add anything beyond
  eligibility?).  Pool training, CLEAN-VAL selection, AUDIT and REVERSAL-TEST use disjoint base images; δ is not tuned.
- Outcome: REVERSAL-test accuracy and worst-class accuracy of the selected member; pool best reported as a ceiling.
- Pre-stated reading: Δ_k = acc(B_k) − acc(A), pooled over encoders × cues × ρ × replicates, 95 % bootstrap CI resampling encoder runs.
  "Audit-assisted selection improves the external outcome" only if the CI of Δ_k excludes 0 on the positive side **and** Δ_k exceeds the
  random-eligible baseline (C − A).  Otherwise: "no improvement shown"; the null result is reported.  VCS vs JS vs HSIC audits are described,
  not ranked by a threshold.  Not claimed: anything beyond these two synthetic cues, this pool, CIFAR-10.

## Label use and test-set disclosure
Labels are used only by the downstream probes / pool classifiers (V1–V3) and the audit's class conditioning; never in SSL pretraining (none here).
All V1–V3 numbers use the CIFAR-10 FIT / selection splits (and CIFAR-100's training partition); the official CIFAR-10 test set is not read,
except by the separately authorised CIFAR-10-C path.

## Gate and smoke (filled in from the jobs below)
Three CPU gate runs (all evidence in `reports/P110_GATE_<job>/`); the first (1017473) failed only on a missing `/usr/bin/time` (harness).
- **1017491:** tests 9/9; smoke of V1 / V2 / V3 on `P35_vcs_a5_views4_800ep_seed0` and `P104_G2_views4_800ep_seed0` (reduced sizes) + aggregate ran
  end to end (340 s).  Full-size V3 design check on G2 seed 0 (1 replicate) revealed two design problems, fixed **before the freeze, on gate data only**:
  (1) the binary class-parity cue of the first design produced no shortcut use at all (reversal ≈ clean accuracy for every member) → replaced by the
  class-specific cue now specified (own colour with probability ρ; reversal = colour of c + 1; CLEAN-VAL without cue); (2) with the class-specific
  cue the colour family produced graded shortcut use (reversal 84.2 → 89.3 as a goes 0 → 1, clean-val within ≈ 1.2 points) but the audit J was ≈ 0
  for every member: the P105 closed-form / exact-JS critics are class-agnostic (z·(2N − 1)), and a class-specific logit shift averages out over
  classes → the audit critics now use the class-conditional linear class z ⊗ onehot(Y) (unit test: J rises 0.0001 / 0.011 / 0.045 / 0.084 for a
  synthetic class-specific shift of 0 / 1 / 2 / 4).  The 4 × 4 tag moved h by ≈ 8 % and was not used by any member → enlarged to 8 × 8.
- **1017571 (final code):** tests 10/10; smoke (265 s) and the full-size V3 check (467 s on CPU, features partly cached) on G2 seed 0:
  reversal accuracy across the pool tag ρ 0.8 86.8–90.0, ρ 0.95 85.5–90.0, colour ρ 0.8 85.7–89.3, ρ 0.95 84.2–89.3; clean-val 87.8–90.4;
  14–18 of 20 members eligible (δ = 1.0); Spearman(reversal acc, audit statistic) VCS −0.29 / −0.65 / −0.56 / −0.57, JS −0.35 / −0.58 / −0.05 / −0.31,
  HSIC −0.41 / −0.77 / −0.58 / −0.70 (one replicate; audit EVAL 2 000; development numbers only, not results).  In this single replicate rule A
  already picks a high-reversal member (clean-val and reversal accuracy are positively related here), so the room for B is small — a legitimate
  outcome that the frozen rules will report as it falls.
- Smoke V1 / V2 values (reduced sizes, meaningless as results): V1 1 % / 10 % / 100 % ≈ 40 / 73 / 81 (P35 s0) and 56 / 81 / 86 (G2 s0) on 500
  selection images; V2 clean / mCA 81.2 / 56.2 (P35) and 85.8 / 61.1 (G2).

## Cost
Measured on CPU in the gate (G2 seed 0): V3 full size 467 s with clean + colour features cached (feature extraction ≈ 100 s per 40 000-image
version on 16 CPUs; each 20-member pool ≈ 60 s on CPU).  Projected on one GPU per encoder: features ≈ 450 k forward passes (CIFAR-10 50 k,
CIFAR-100 50 k, V2 150 k corrupted, V3 200 k) ≈ 2–4 min; V1 probes (11 rows × 7–8 probe fits, the 45 k-row fits dominate) ≈ 8–12 min; V2 ≈ 3 min
(PIL corruptions on CPU); V3 20 pools ≈ 6–10 min → ≈ 20–30 min per encoder.  **3 GPU bundle jobs × 4 encoders ≈ 1.5–2 h each, ≈ 5–6 GPU-h
total**, normal QOS (`slurm/p110_lines.txt`).  Feature cache `outputs/P110_features/` ≈ 300 MB per encoder (float16) ≈ 3.7 GB for 12 encoders.
CIFAR-10-C (not registered; needs authorisation): 950 k test-derived images per encoder ≈ 5 min per encoder on a GPU, one job for 12 encoders ≈ 1 h.

## Delivery (v4 §5 fields)
```yaml
experiment_family: frozen-representation evaluation (V1 label efficiency / transfer, V2 corruption tolerance dev, V3 audit-assisted selection)
protocol_id: P110_v4_V
estimator: none (V1 / V2); audit statistics VCS closed-form J, exact JS, class-conditional HSIC (V3)
evaluation_readout: selection-split top-1 (V1, V2); reversal-test accuracy / worst-class (V3)
n_independent_units: encoder seeds (3 per family); subset draws (V1); base images (V2 bootstrap); replicates (V3)
status: draft
```

## Decisions at the freeze (main session)
1. **Launch:** the orchestrator submits `slurm/p110_lines.txt` (3 GPU bundle jobs) only when no p104_ / p107_ / p108_ / p109_ job remains in squeue
   (owner: after the currently submitted experiments).  Encoders = the frozen default list (P35, P41, P104 G2, P104 U2; seeds 0–2), all COMPLETED by then.
2. **V3 design calibration disclosed:** the cue construction (class-specific colour cue, 8×8 tag) and the class-aware audit features (z ⊗ onehot(Y))
   were fixed on gate runs that used the P35 seed-0 and G2 seed-0 encoders, which are also evaluation encoders.  The calibration only ensured that
   pool members differ in shortcut use and that the audit can see a class-specific shift; it did not compare selection rules A and B on any
   reversal outcome used for the claim.  The V3 reading uses fresh replicate seeds and the frozen rules.
3. **CIFAR-10-C** (`slurm/p110_lines_cifar10c.txt`, official test images) is NOT registered: it needs the owner's separate authorisation and, per v4,
   a frozen main recipe; reporting only, never selection.
4. P107 winners / later encoders are added by an addendum with new lines (finished outputs are skipped).
