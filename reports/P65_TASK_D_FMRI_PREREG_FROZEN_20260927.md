# Pre-registration DRAFT — task-level pre-check D-fMRI: head-motion leakage in "cleaned" resting-state BOLD, detected with autocorrelation-respecting dependence tests (P65), 2026-09-27 — FROZEN 2026-09-27T21:07:05Z before GPU compute (download 1011925, CPU probes 1011932 / 1011933 disclosed inside; owner: "fMRI 的 D 族依赖检测也加上，另开 prereg")

Status: DRAFT (implementation fork); the main session freezes it.  Owner's instruction: "fMRI 的 D 族依赖检测也加上，另开 prereg".  Family: brief
appendix A row D (leakage / shortcut detection, dependence profiles).  Nothing in this unit touches the frozen items; the critic output is tanh;
no hard-negative selection, no auxiliary losses; the reference measure is the product of marginals under stationarity (circular shifts).

## 0. Question
After standard confound regression, how much of the head-motion signal is still present in the BOLD data?  Does the bounded VCS statistic,
calibrated by circular shifts rather than i.i.d. permutations, detect the dependence between BOLD-derived features z_t and framewise
displacement N_t with the same power as HSIC and better than the field's default (QC–FC correlation with the global signal); does its
magnitude decrease monotonically as the cleaning strength increases; and what does the dependence-vs-lag profile look like?

## 1. Data and provenance (written before the download)
- **Dataset.** AOMIC-PIOP1, OpenNeuro **ds002785**, DOI 10.18112/openneuro.ds002785.v2.0.0; raw dataset and the shipped fMRIPrep 1.4.1 derivatives
  (`derivatives/fmriprep/`).  Licence, verbatim from the two `dataset_description.json` files (sha256 45b7ee37… and bea69b97…, recorded by the
  anatomical download of the same day): `"License": "CC0"` (dataset) and `"License": "CC0"` (derivatives).  Acknowledgement text as recorded in
  `data/aomic_piop1/PROVENANCE.json` (Snoek et al., AOMIC; fMRIPrep, Esteban et al. 2019).
- **Bucket.** `https://s3.amazonaws.com/openneuro.org/ds002785/` (public; HTTP 200 from this cluster, verified by HEAD on 2026-09-27).
- **Files per subject** (resting-state run, multiband 3, TR 0.75 s, 480 volumes, EPI 51 × 61 × 45 at 3 × 3 × 3.3 mm, in the subject's T1w frame):
  `derivatives/fmriprep/sub-XXXX/func/sub-XXXX_task-restingstate_acq-mb3_space-T1w_desc-preproc_bold.nii.gz` (≈ 199 MB),
  `…_space-T1w_desc-brain_mask.nii.gz` (4 KB), `…_desc-confounds_regressors.tsv` (≈ 2.9 MB, 392 columns incl. `trans_*`, `rot_*`, their
  `_derivative1`, `_power2`, `_derivative1_power2`, `cosine00–03`, `framewise_displacement`, `global_signal`, `csf`, `white_matter`), and the raw
  `sub-XXXX/func/sub-XXXX_task-restingstate_acq-mb3_bold.json` (TR).  **Deviation from the brief given to the fork:** the MNI-space run is
  389 MB per subject (23 GB for 60); the T1w-space run of the same data is used instead (199 MB, ≈ 12 GB for 60) — same volumes, native EPI
  grid, no MNI resampling.  Downloaded with retries; per-file sha256, bytes and the id-list sha256 go to `data/aomic_piop1/func/PROVENANCE_func.json`.
- **Subjects.** The 60 subject ids already selected for B-T2's fMRI source (`data/aomic_piop1/ids.txt`, sha256 a3dd39dd…; rule: seed 20260927 over
  the 210 subjects whose T1w preproc, brain mask and resting-state `space-T1w` boldref all answered HTTP 200).  Identity = subject id; one run
  per subject; no official test split exists for this dataset and none is created.

## 2. Preprocessing (`scripts/task_d_fmri_prep.py`, CPU, one `.pt` per subject)
Drop the first 5 volumes (and the matching confound rows).  Spatial block means over **2 × 2 × 2 voxel cubes** (≈ 6 × 6 × 6.6 mm) whose mask
fraction ≥ 0.5 (≈ 2–4 k features; the fork's brief said 4 × 4 × 4 at 2 mm MNI voxels — same physical scale at the 3 mm EPI grid).  In every
setting: remove the mean and the fMRIPrep cosine drift regressors (`cosine00–03`) by OLS, from the features and from the motion regressors.
Motion-cleaning knob α ∈ {0, 0.5, 1}: X = the 24-parameter model (6 rigid-body parameters, their temporal derivatives, and the squares of both;
NaN in the first derivative row → 0), β̂ = OLS(X → Y_drift-removed), residual_α = Y − α · X β̂ (α = 1 is standard full regression, α = 0 no
motion cleaning, α = 0.5 half-strength).  z-score each feature; PCA to d = 32 fitted on the run itself (unsupervised; explained variance
recorded) → z_t ∈ R³² per TR, one matrix per α.  Also stored per α: the global signal g_t (mean over features of residual_α before z-scoring).
Nuisance N_t = `framewise_displacement` (first NaN → 0), standardised.  Stored: z (3 α), g (3 α), N, TR, n_volumes, the six raw motion
parameters, FD mean, PCA explained variance.

## 3. Tests (`scripts/task_d_fmri_tests.py`; GPU if available)
Per subject, α, sample size n ∈ {120, 240, all} (leading n TRs after alignment) and lag τ ∈ {0, 1, 2, 4, 8} TRs (pair z_t with N_{t−τ}):
the used TRs are split **contiguously** — first 70 % = training region (its first 80 % fits the critic, its last 20 % selects ridge / tanh
scalar / early stopping), last 30 % = EVAL block; nothing crosses the boundary.  P-term: time-aligned pairs; Q-term: pairs (z_t, N_{t+s}) with
K = 4 circular shifts s ≥ 30 TRs inside the same region (product-of-marginals sample under stationarity).  Critics: closed-form linear class
on φ(z, n) = [z·n, z, n, 1] (66-d; ridge λ ∈ {1e-4, 1e-2, 1} × mean diag, tanh scalar c ∈ {0.25, 0.5, 1, 2, 4}, chosen on the validation part) and a
small MLP on concat(z, n) (33 → 128 → 128 → 1, tanh output, Adam 300 full-batch steps, early-stopped on the validation J); the reported
critic is the one with the better validation J.  **Null distribution: 200 random circular shifts of N (≥ 30 TRs) — never i.i.d. permutations**
(autocorrelation).  Tests at δ = 0.05: `vcs_shift` (Ĵ_eval on the EVAL block vs its circular-shift null), `vcs_hoeff` (Ĵ_eval − τ_{n_eval, n_eval}(δ) > 0),
`hsic_shift` (Gaussian kernels on z and on N, median heuristic, same shift null), `c2st` (same MLP budget, accuracy on EVAL vs shift null), and the
field's direct baseline `qcfc_corr` (Pearson correlation between N and the global signal g on EVAL, same shift null).
**Exact-null calibration case:** each subject's z paired with the FD series of a different subject (the fixed derangement i → i+1 mod 60 in
sorted-id order, series truncated to the shorter length) — independent by construction, with realistic autocorrelation on both sides; run
at τ = 0 for every (α, n).  Output per (α, n, τ): rejection rate over the 60 subjects for every test, mean ± sd Ĵ_eval, mean QC–FC r; JSON with
every per-subject decision; markdown tables.

## 4. Pre-committed reading (verbatim from the main session's directive)
(i) *false alarm*: on the subject-swapped null every test rejects ≤ 0.09 of the 60 subjects at every n and τ = 0 (R = 60 → ≤ 5/60); a test above
that at some n is invalid there and reported as such. (ii) *power*: at α = 0 (no motion cleaning), n = 240, τ = 0, `vcs_shift` rejects in ≥ 80 %
of subjects and `hsic_shift` is not more than 0.05 above it; `qcfc_corr` is reported alongside as the field baseline. (iii) *monotonicity in
cleaning*: mean Ĵ (τ = 0, n = all) decreases strictly from α = 0 to 0.5 to 1 in ≥ 90 % of subjects. (iv) *lag profile* (reported, not read): Ĵ vs τ
at α = 0, expected to peak at τ ∈ {0, 1}. (v) *residual leakage after full cleaning* (finding, not read): the rejection rates at α = 1 for
`vcs_shift` and `hsic_shift` are the estimate of nonlinear motion leakage that the 24-parameter linear model leaves; they are reported with the
false-alarm rate next to them. Holds if (i)–(iii) hold; holds conditionally if one fails; does not hold if (ii) is below 0.5 or Ĵ is not
monotone in α for a majority of subjects.

## 5. Not claimed / limits
Anything clinical; other cleaning models (aCompCor, ICA-AROMA, scrubbing); other datasets or task runs; the conditional (within-block) version
of the test is **not** run here, because block-wise circular shifts are not an exact null for a within-block conditional statistic; the PCA-32
features are one representation of the BOLD signal, not "the" signal; one run per subject; a single TR (0.75 s) and scanner.

## 6. QC sentinels
TR read from the raw JSON (expected 0.75 s); n_volumes ≥ 200 after dropping (expected 475); FD mean per subject reported; PCA explained variance
of the 32 components ≥ 30 % (else the subject is flagged, not dropped); the swapped-null case must use two different subjects of equal TR.

## 7. Probe and compute
CPU smoke: prep + tests on 2 subjects (`--smoke`: n = 120 only, τ ∈ {0, 1}, 20 shifts, 50 critic steps), disclosed below before the full run.
Full: prep = one CPU job (60 subjects, ≈ 1–2 min each); tests = one GPU job (60 × 3 α × 3 n × 5 τ instances + 540 null instances, ≈ 0.5–1 h).

## 8. Smoke probe (disclosed; CPU jobs 1011932 prep, 1011933 tests; 2 subjects sub-0002 / sub-0014, n = 120 only, τ ∈ {0, 1}, 20 shifts, 50 critic steps)
- Prep: T = 475 volumes after dropping 5, TR 0.75 s (from the raw JSON), 7 163 / 6 500 in-mask 2 × 2 × 2 blocks (of 18 538 / 16 800), FD mean
  0.075 / 0.079 mm (a low-motion cohort), PCA-32 explained variance 0.41 / 0.47 (α = 0), 0.36 / 0.44 (α = 1); 16–22 s per subject on CPU.
- Tests: all cells ran end to end (closed-form + MLP critics, shift null, HSIC, C2ST, QC–FC), 2–6 s per cell on CPU; with 2 subjects, 36 evaluation
  TRs (30 % of 120) and 50 steps nothing rejects and Ĵ ≈ 0 ± 0.02 — code-path evidence only, not a result.
- Two things the probe makes visible for the freezing decision (the reading rules above are not changed by the fork): (a) the QC sentinel
  "PCA explained variance ≥ 30 %" will flag most subjects at 7 k features / 32 components (0.36–0.47 here); it should be read as a recorded
  number or lowered to ≥ 30 % — the main session decides before launch; (b) n = 120 leaves only 36 TRs in the evaluation block (n = 240 → 72,
  all → 142), so n = 120 is a small-sample stress cell and the power reading (ii) is at n = 240 by design.
- Download: job 1011925 (CPU), T1w-space run; ≈ 12 GB for 60 subjects; `func/PROVENANCE_func.json` written at the end with per-file sha256.


**Freezing note (main session).**  The PCA-explained-variance QC sentinel is lowered from 50 % to 30 % before compute (the probe gave
0.36–0.47 at 7 k block features / 32 components; the sentinel guards against a degenerate decomposition, not against a realistic one).  The
deviations from the directive (T1w-space run instead of MNI, 2 × 2 × 2 block means at the native 3 mm grid) are accepted as recorded in §1–2.
The power reading of rule (ii) is taken at n = 240 as written; n = 120 (36 evaluation TRs) is reported for completeness.  Nothing else changes.
