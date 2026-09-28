# Pre-registration — P85: direct CS-estimator comparison on shared pair data (package v2, E line) (2026-09-28) — FROZEN 2026-09-28T18:19:49Z before GPU compute (E-line CPU gate 1013255 passed: v2 14/14, stage-0 13/13, reference 29/29; oracle table within MC tolerance of RESULTS_20260928.md).  Design frozen here; the pilot (§6) runs first and its measured GPU cost is appended below as a results-only addendum — a change of grid or method after the pilot would be a numbered addendum, not an edit.

Status: FROZEN 2026-09-28T18:19:49Z (main session).  The main session freezes it (renamed `*_FROZEN_*`, timestamp in the title) before any GPU job; the pilot (§6) runs
first and its measured cost is written into the freeze.  Sources: `VCS_QMI_Estimator_Research_and_Server_v2.zip` — server spec §3 (controls),
§4 (roles), §5 (E line), §12 (tests), §13.2 (delivery block); research plan §4–§5; supplement §3.7 (keep the neural references).  Owner go
2026-09-28 ("整理…然后开始工作…完成 CVPR 的所有准备"; package first).  P69 was never frozen; its generators, six estimators and negative
constructions (`src/vcs_estim/{data,critics,estimators}.py`) are re-used here; the P81/P82 Gaussian probe (roles, oracle, C0–CQ) is not
repeated — P85 covers the spec's new content: same-target kernel controls, the irrelevant-dimension / sample-size / batch axes and the
dependence staircase on fixed finite samples.  Results → `P86_*` (per-cell JSON in `outputs/P85_estim_benchmark/`, aggregate `reports/P86_p85_aggregate.{md,json}`).

## Question
On the *same* finite pair sample, how does the neural VCS critic (VCS-N, original J) compare with the estimators that share its target S —
S-KDE (kernel plug-in of η), S-Kernel (RFF / Nyström features + tanh read-out trained on J), rLS / rLS-tanh (RuLSIF α = ½ ridge in the same
feature class) — in same-target error, posterior fit, sample efficiency, robustness to irrelevant dimensions and cost; and how do all of them
and the neural references (matched JS, InfoNCE, NWJ, DV/MINE, SMILE) and the classical fixed-measure kernel CS (CS-K-native, own truth) resolve
a dependence staircase?  Cross-target quantities are compared by ordering probability and cost only (spec §5.3).

## Code (this unit; all untracked at HEAD 40f2c20 until the main session commits)
- `src/vcs_estim/benchmark.py` — cell runner (roles, methods, readouts, bootstrap, channel derivative, §13.2 block); `src/vcs_estim/p85_grid.py` — the grid;
  `src/vcs_estim/data.py` — reparameterised samplers (`sample_base` / `from_base`), `setting_from_mi`, `pad_sides`; `src/vcs_estim/kernel_cs.py` —
  `rls_moments` / `rls_from_moments` (refactor of `rls_fit`, identical output); `scripts/estim_p85.py` (units → JSON, resumable), `scripts/estim_make_units.py`,
  `scripts/estim_p85_aggregate.py`; `slurm/estim_p85_cells.sbatch` (GPU), `slurm/estim_v2_gate.sbatch` (CPU gate), `slurm/p85_units/*.txt`.
- Tests `tests/test_vcs_estim_v2.py` (spec §12 table, one test per row + the package's own `checks_v2.py` identities + P85 regression checks); gate job(s) listed in §8.
- Untouched: `objectives`, `candidates`, `fitting` (used as is), `frozen*` (D fork), `src/vcs_ssl/**` (S fork).

## Design
**Generators** (P69 code): `gaussian` (d independent bivariate normals, ρ from the MI), `cubic` (y ↦ y³ coordinate-wise), `xor_mixture` (10 sign-mixture
pairs; x, y ∈ ℝ¹⁰).  Truths from the analytic PMI on a 200 000-pair TRUTH role per side: S = E_M tanh(pmi/2)² (± MC se), J at the oracle, 2·JSD, MI;
classical CS truth D_CS^Leb = d log(1 − ρ²/4) − (d/2) log(1 − ρ²) for `gaussian` only (cubic: reference density not square-integrable; xor: no closed
form — CS-K-native gets no error column there).  Irrelevant dimensions (spec §5.1): X = (X_s, N_x), Y = (Y_s, N_y), independent N(0, I) padding from
its own stream per role and side; every PMI is evaluated on the signal coordinates (tests: S invariant, η identical per sample).
**Roles** per cell (spec §4): FIT = N per distribution (P joint pairs and an independent Q product sample), TUNE = SELECT = max(64, N // 4)
(the selection budget scales with N and is the same for every method), EVAL = 32 768 per side (fixed; never used for any choice), GRAD = 20 000
(channel derivative), TRUTH = 200 000.  Every role / side / padding block from its own seeded stream (`roles_hash` in the JSON).  Statistical
unit = the anchor (product negatives: P and Q units resampled separately; cyclic / in-batch negatives: the EVAL block of 1 024 anchors); 200
bootstrap replicates of every native value are stored for the ordering probabilities; K negatives per anchor are never counted as K units.
**Methods on the same FIT / TUNE / SELECT / EVAL** (one JSON row each; `selected` marks the row chosen on SELECT / TUNE, all rows kept):
| id | class / fit | selection (never on EVAL) | native readout + own truth | common readouts |
|---|---|---|---|---|
| VCS-N | JointMLP concat→256→256→1, −J, Adam, batch B, `updates` steps | lr ∈ {1e-4, 5e-4, 2e-3} and the update by SELECT −J every 100 updates | J_eval vs S | J_eval, posterior MSE, Bayes-risk ratio, tails |
| neural refs | same class; js (Deep-InfoMax form, T = tanh(f/2)), infonce, nwj, dv (EMA denominator), smile (τ = 5) | as VCS-N, own native risk | value vs own truth (JS2 / MI) | js: posterior readouts; infonce / nwj / dv / smile: none (no balanced posterior without a gauge step — recorded) |
| negatives | vcs, js: product ∣ cyclic8 ∣ inbatch; infonce: inbatch ∣ cyclic8; nwj, dv, smile: product ∣ inbatch | — | — | product = the same-target comparison partner of the kernel rows |
| CS-K-native | Gaussian-kernel plug-in D_CS on the FIT joint pairs, effective kernel √2 h (= KDE bandwidth h; convention pinned by a test against the smoothed-Gaussian closed form), chunked, float64 | pre-declared default multiple 1.0 × FIT median distance per side (median heuristic); multiples 0.25 / 0.5 / 2 reported as sensitivity — the median grows with padded dimensions, so the default is expected to over-smooth on the d_total axis (a property of the method, reported as such) | D_CS vs D_CS^Leb (gaussian only) | none (different target; never in an S column) |
| S-KDE | product Gaussian KDE on FIT joint pairs, η̂ = tanh(½(log p̂ − log p̂_x − log p̂_y)) on EVAL queries | (h, b) ∈ {¼, ½, 1, 2}² × FIT medians by max J_kernel on TUNE; Scott's rule as the method-native sensitivity row | S_plug = E_M η̂² vs S | J_kernel, posterior MSE (two fields kept apart) |
| S-Kernel | RFF φ(w) = √(2/m) cos(Ωw + b), w = [x; y], Ω, b fixed (seed recorded), θ ∈ ℝ^{m+1} trained on −J with the package trainer (batch B, `updates`, SELECT every 100) | (m, bandwidth multiple, lr) ∈ {256, 1024, 4096} × {0.5, 1, 2} × lr grid by SELECT | J_eval vs S | as VCS-N |
| S-Kernel-Nyström | exact-kernel features from min(512, N) FIT centres, multiple 1 | lr by SELECT | J_eval vs S | as VCS-N |
| rLS | closed-form ridge θ = ½(G + λP)⁻¹d on the same RFF features (Cholesky, intercept unpenalised, flag recorded) | (m, multiple, λ ∈ {1e-4 … 1e-1}) by the RuLSIF LS objective on TUNE (method-native) | J_hat of the raw T (unbounded; diagnostic) | none |
| rLS-tanh | tanh(c T_raw + b0), (c, b0) by Nelder–Mead on J_hat over SELECT scores (numerical, not closed form) | inherits rLS's features | J_eval vs S | as VCS-N |
Per row also: fit / evaluation seconds, peak memory, n_independent_units, n_positive_pairs, n_negative_pairs, numerical failures (non-finite
steps, log-guard activations), tail quantiles of |T − η|, gate mean 1 − E_M T²; the spec §13.2 block fields.
**Channel derivative** (spec §5.3): d/dθ of E_{P_θ}[T − T²/2] with the critic fixed, gradient through the generator from the GRAD bases (θ = ρ or c;
Q is θ-free in every setting), for the selected VCS-N (product), JS (product), S-Kernel, S-KDE and rLS-tanh rows, against the oracle's central
differences (δ ∈ {1e-2, 3e-3, 1e-3}, common random numbers) and the envelope form.

## Grid (`slurm/p85_units/`, 81 distinct cells, 3 seeds; shared cells run once)
| axis | varied | fixed |
|---|---|---|
| irrelevant dimensions (15) | d_total ∈ {2, 10, 20, 50, 100} per side | gaussian, signal d = 2, I_PAD = 1.5 nats → ρ = 0.8814, **S = 0.540 ± 0.0005** (gate truth table; mid dependence), N = 4096, B = 256, 2000 updates |
| independent sample size (12) | N ∈ {256, 1024, 4096, 16384} | gaussian d = 20, I = 4 nats (ρ = 0.5742, S = 0.8024 ± 0.0005, the P82 mid condition), B = min(256, N), 2000 updates |
| batch, equal updates (9) | B ∈ {64, 256, 1024} | gaussian d = 20, I = 4, N = 4096, 2000 updates |
| batch, equal exposure (9) | B ∈ {64, 256, 1024} with updates = 512 000 / B | same pool; exposure = 2000 × 256 positive pairs |
| dependence staircase (45) | I ∈ {2, 4, 6, 8, 10} nats × {gaussian, cubic, xor_mixture} | d = 20 (xor: 10 pairs), N = 4096, B = 256, 2000 updates |
The fresh-stream staircase of the P69 draft (tracking, not fixed-sample estimation; spec §5.2) is kept as code (`scripts/estim_staircase.py`) and
smoke-tested, but is **not** part of this unit.

## Reading rules (pre-stated; everything else is descriptive)
Comparisons are made on the selected rows, per axis value, mean ± sd over the 3 seeds, with the EVAL unit-bootstrap se per cell as the noise scale.
1. **Same-target error (S)** — VCS-N vs each of S-KDE, S-Kernel (RFF and Nyström), rLS-tanh on posterior MSE E_M(T − η)² (primary) and |J_eval − S|:
   "better" if VCS-N is lower on ≥ ¾ of the axis values by more than 2 pooled seed sd, "worse" if the reverse, "on par" otherwise.  Excess-to-Bayes
   risk is read only where 1 − S exceeds 3 oracle se (flag `indistinguishable` otherwise; absolute error kept).
2. **Sample efficiency** — the N axis: the slope of log posterior MSE vs log N per method and the smallest N at which each method's error is within
   2 sd of its N = 16384 value; reported, not thresholded.
3. **Irrelevant dimensions** — the d_total axis at fixed S: the ratio error(d_total = 100) / error(d_total = 2) per method; a method is "robust" if
   the ratio ≤ 2, "degrades" if ≥ 4 (in between: reported as is).
4. **Batch** — equal-updates vs equal-exposure: whether the VCS-N error at B = 64 / 1024 differs from B = 256 by more than 2 sd, separately for the
   two accounting conventions (the spec's optimisation-organisation vs sample-exposure distinction); same for S-Kernel (only trained method with the
   same trainer); the closed-form methods do not depend on B (their rows repeat by construction — a consistency check).
5. **Dependence resolution (cross-target)** — per estimator the ordering probability P(V_{l+1} > V_l) of adjacent staircase levels from the unit
   bootstrap and |Δmean| / pooled sd; MI estimators are compared with VCS / JS on these two quantities only, never on raw error; InfoNCE's bound
   log(K + 1) is stated next to its values.
6. **Cost** — fit seconds and peak memory per method and axis value; CS-K-native and S-KDE scale as O(N²) by construction (recorded, not "found").
7. **Not compared**: raw rLS (unbounded) against anything but its own tanh wrap; CS-K-native against S; JS's native value against S; nothing is
   converted between targets.  Any post-hoc grid change, seed addition or selection after seeing EVAL is a new pre-registration.

## QC (per cell, in the aggregate)
Oracle J on TRUTH within 3 se of S; every selected row finite; non-finite training steps counted; the RuLSIF identity residual on TUNE < 1e-9;
CS-K-native log guard never active on the selected multiple; missing cells listed; per-cell wall-clock and device recorded.

## 6. Pilot before the full submission (spec §5.1) and cost
`slurm/p85_units/pilot.txt` — 4 cells: d_total ∈ {2, 100} × N ∈ {256, 16384}, VCS-N (all three negative constructions) + every kernel control, seed 0.
It measures the O(N²) kernel memory / time at N = 16384 and d_total = 100 on one GPU before the 81-cell grid is queued.
CPU-smoke-based guess (to be replaced by the pilot's GPU measurement in the freeze): see §8.  Full grid: 8 unit files (`full_00..07.txt`, cost-balanced),
one GPU job each, `--time=12:00:00`, partitions A100,H100,L40S,RTX6000PRO (excl. node51), resumable per cell.
```
sbatch --job-name=p85_pilot --export=ALL,UNITS=slurm/p85_units/pilot.txt,OUT=/home/infres/yinwang/CS_QMI/outputs/P85_estim_benchmark slurm/estim_p85_cells.sbatch
for i in 00 01 02 03 04 05 06 07; do sbatch --job-name=p85_full_$i --export=ALL,UNITS=slurm/p85_units/full_$i.txt,OUT=/home/infres/yinwang/CS_QMI/outputs/P85_estim_benchmark slurm/estim_p85_cells.sbatch; done
python scripts/estim_p85_aggregate.py --inputs /home/infres/yinwang/CS_QMI/outputs/P85_estim_benchmark --out reports/P86_p85_aggregate
```

## 7. Delivery (spec §13.2)
Per cell JSON: `cell` (parameters, sizes, roles hash, code commit, device, torch), `truth`, `rows` (each with the YAML block fields), `channel_derivative`,
`summary`.  Aggregate: per-axis tables (same-target, native, cost), ordering probabilities, QC, missing cells, `*_yaml.md` with one block per method.
Template of the block (values filled per row):
```yaml
experiment_family: direct_cs
protocol_id: P85
source_commit: <commit of the cell run>
source_basis: supplement_af7172c_plus_v2
estimator: neural_vcs | neural_js | neural_infonce | neural_nwj | neural_dv | neural_smile | cs_kernel_native | s_kde | s_kernel_rff | s_kernel_nystrom | rls_raw | rls_tanh
estimand: S | JS | MI | native_CS (Lebesgue reference)
evaluation_readout: native | J_common | S_plugin_and_J_common
loss_scale: <as recorded>
reference_measure: mixture_equal | native_base
critic_class: <as recorded>
gradient_routing: full (data fixed; critic only) | none | native autodiff
n_independent_units: <N>
n_positive_pairs: <count>
n_negative_pairs: <count or null>
split_manifest_hash: <roles_hash>
noise_target_kind: none
noise_tau: null
noise_sigma_coordinate: null
fit_seconds: <measured>
evaluation_seconds: <measured>
status: completed | failed
```

## 8. Gate (CPU, before the freeze) — filled by the fork after the job
Gate jobs (CPU partition, `slurm/estim_v2_gate.sbatch`; artefacts in `outputs/P85_gate_<jobid>/`):
- **1013250** (first run of the fork's code): v2 suite 12 / 14, stage-0 13 / 13, reference 29 / 29.  Two failures, both real: (i) the P69-draft
  `oracle_transform("vcs")` returned tanh(pmi/2) while `estimate("vcs")` applies tanh again (double tanh; the staircase's "oracle" was biased low,
  0.522 vs truth 0.569 in the smoke) → fixed to f* = pmi/2; (ii) the kernel-CS plug-in was compared with the *unsmoothed* closed form; a Gaussian KDE with
  bandwidth h estimates D_CS of the smoothed pair, i.e. the same family at ρ/(1 + h²) (affine invariance) → the test now checks both kernel conventions
  against that limit (effective √2h ↔ KDE bandwidth h; raw h ↔ h/√2), within 0.02 at n = 3000 (measured deviations ≤ 0.011).
- **1013252** (after the fixes): v2 suite **14 / 14**, stage-0 13 / 13, reference 29 / 29; truth table, oracle table, unit files, P85 smoke (2 cells, all
  methods, 9 s + 12 s on 8 CPU threads), both aggregators, staircase smoke (oracle now 0.538 / 0.918 vs truth 0.569 / 0.912): all rc = 0.
- **1013255** (final code: per-unit standard errors of the channel derivative, CS-K multiples {0.25, 0.5, 1, 2}, aggregate 'cells outside the axes' section): v2 suite 14 / 14, stage-0 13 / 13, reference 29 / 29, all ten gate steps rc = 0 (log `slurm_logs/estim_v2_gate_1013255.out`, artefacts `outputs/P85_gate_1013255/`).
- Oracle-table re-derivation vs `CS_QMI/RESULTS_20260928.md` (single shuffled negative, n = 10 000, 4 M common draws): S 0.351/0.692/0.953/0.994 (owner
  0.352/0.692/0.953/0.994); NWJ/VCS 1.62/3.01/10.73/37.70 (owner 1.62/3.01/10.78/37.73); JS/VCS 0.968/0.995/1.039/1.048 (owner 0.97/0.99/1.04/1.06);
  VCS/CRB 2.08/3.34/9.32/26.53 (owner 2.08/3.35/9.28/26.51) — all within the MC tolerance of a 4 M-draw ratio (the owner's draws differ); the
  ρ = 0.999999 JS/VCS entry (1.048 vs 1.06) is the rare-event-dominated cell.
- Truth table (gate): gaussian d = 2: I = 1 → S 0.391, 1.5 → 0.540, 2 → 0.661, 3 → 0.828; d = 20: I = 2 → 0.567, 4 → 0.802, 6 → 0.912, 8 → 0.963, 10 → 0.985
  (the staircase levels span S 0.57–0.99; the P69 levels are kept as frozen).
- Smoke sanity (N = 128, 60 updates, one lr — code-path evidence only, no reading): every row finite; oracle J on TRUTH within 3 se of S in both cells;
  RuLSIF identity residual 1e-16; CS-K-native at the median heuristic ≈ 0.003 vs its truth 1.068 with 4 padded dimensions (over-smoothing, see the
  method note above); S-Kernel's SELECT kept update 0 (θ = 0) at this budget — `selected_update` is recorded per row so such trivial fits are visible.

**Cost (CPU-smoke-based guess; the pilot's GPU measurement replaces it at the freeze).**  A tiny cell takes ~10 s on 8 CPU threads.  A full cell has 42
neural fits (14 estimator × negative-construction pairs × 3 lr) + 27 RFF fits + 3 Nyström fits of 2000 updates, 36 closed-form ridges, 16 KDE bandwidth
evaluations, 4 kernel-CS plug-ins, 2 000 000 EVAL scorings and the channel derivative; on an A100 I expect ≈ 5–10 min for B = 256 cells, ≈ 40–60 min for
the six B = 1024 in-batch cells (1 M pairs per step) and ≈ 10–15 min for N = 16384 cells (O(N²) kernels are seconds on a GPU) → ≈ 15–20 GPU-h for the
81-cell grid, i.e. 8 jobs of ≈ 2–3 h.  Peak memory is dominated by in-batch B = 1024 training (≈ 4–6 GB) and the N = 16384 × 4097 RFF feature matrices
(≈ 0.5 GB each, float64).  These numbers are guesses until `pilot.txt` has run.

## 9. Disclosed changes to previously committed modules
- `src/vcs_estim/data.py` (untracked E-line file at HEAD, now restructured): settings expose `sample_base` / `from_base` (reparameterised, differentiable in
  the dependence parameter) — `sample` output distribution unchanged; `setting_from_mi` for arbitrary MI; `pad_sides`.  `make_setting` unchanged.
- `src/vcs_estim/kernel_cs.py` (untracked at HEAD): `rls_fit` now delegates to `rls_moments` + `rls_from_moments` (same numbers; tested equal to 1e-12).
- Committed package modules (`objectives`, `candidates`, `fitting`, `convex_mix`, `bounded_core`, `pairing`, `synthetic`, `evaluation`, `run`, `frozen`) are
  not modified by this fork (`synthetic.py`'s padded generator was already in the working tree; `_gen` is imported for the P85 streams).

## Addendum (execution only, 2026-09-28T19:16:38Z) — pilot job 1013286 failed on a GPU-only bookkeeping call
All four pilot cells raised `RuntimeError: Invalid device argument` in `_peak_mem` (`torch.cuda.reset_peak_memory_stats` before the caching
allocator existed; unreachable on the CPU gate).  Fix in `src/vcs_estim/benchmark.py`: a zero-size allocation initialises the allocator and the
statistics call is guarded.  No estimator, grid, role or read-out code changed.  Pilot resubmitted after this line.

## Pilot measurement (results-only addendum, 2026-09-28T19:42:59Z; job 1013331, H100, )
Wall time per cell 162–176 s (47 rows each: 9 VCS-N rows 4–14 s fit, kernel rows ≤ 3 s), CUDA peak 3.3–3.8 GB, CPU RSS 2.1 GB.  Full grid (81 distinct
cells, 8 cost-balanced chunks; the six B = 1024 in-batch cells and the N = 16384 cells are the heavier ones): ≈ 4–14 GPU-h in total, ≈ 0.5–2 h per chunk on
H100; `slurm/estim_p85_cells.sbatch` keeps its 12 h limit.  Derivative self-check: oracle envelope − finite difference = 1.07 combined SE (threshold 3; no
inspection triggered).  Values are not read here; two observations are logged for the P86 reading: at d_total = 100, N = 256 all nine VCS-N rows share one
J_eval (−0.0088), i.e. the SELECT rule kept the initial critic (recorded per row), and the kernel same-target rows are ≈ 0 at d_total = 100 for both N.
Full grid submitted after this line, unchanged.
