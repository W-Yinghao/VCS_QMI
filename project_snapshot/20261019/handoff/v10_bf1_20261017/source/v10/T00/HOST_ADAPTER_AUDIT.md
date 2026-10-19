# V10 T00 — host-adapter audit of the V9 registration host (read-only)

Date 2026-10-16 (snapshot 20:14:25Z). Stage T00 of phase 1 (package `v10/package_v10_server_v1_20261016`, `docs/HOST_ADAPTER_CN.md`).
Machine-readable mirror: `v10/T00/host_map.json` (same facts). Evidence (command log, two helper utilities and their outputs, sha256 in `host_map.json`):
`/projects/EEG-foundation-model/yinghao/outputs_v10/T00/host_audit_evidence/`.

| item | status |
|---|---|
| host source map against every row of `HOST_ADAPTER_CN.md` | WRITTEN — requires review |
| `host_adapter_verified` | **false** (values, A/B/W gradients and continuous-warp gradients on the real host are T01: NOT_RUN) |
| `scripts/preflight.py` | NOT_RUN by this task (separate T00 item) |
| jobs submitted by this task | 0 |
| V9 files / V9 outputs modified | none |
| image or label data read | none (read: source, sbatch files, log header lines, manifests = roles / ids / hashes / shapes, perturbation lists, one checkpoint's key/shape metadata via a pickle stub, cost fields of two DEV result files) |
| BF-SHG sealed TEST (`closed/test.zip`, `outputs_v9/reg_test`) | not opened |

Location convention: `file:Lstart-Lend` on branch `v10/mf-vcs` at `f652f1e`. The branch head moved during this audit through parallel V10 T00 commits (97119bd, d8364a5, ac7eb7d, 78e019f, … — `scripts/v10`, `tests/v10`, `v10/T00` only); `git diff --quiet f652f1e HEAD -- src/v9 scripts/v9 slurm/v9` was run immediately before this commit and was clean, so every location below holds unchanged at this commit. Shorthands: **P** = `src/v9/reg/r6_reg_pilot.py` (REG-P1 copy), **Q** = `src/v9/reg/r6_reg_p2.py` (REG-P2 copy), **P3** = `src/v9/reg/r6_reg_p3.py` (REG-P3 copy), **G** = `src/v9/reg/g1.py`, **G2** = `src/v9/reg/g2.py`.

## 0. Answers first

1. **Host identity.** Every `src/v9/reg/*.py` file is byte-identical between the package's historical commit `27117c9`, the V10 branch head `f652f1e` and the V9 worktree (`5f08b2f`). Between `27117c9` and `f652f1e` the only change under `src/v9/reg`, `scripts/v9`, `slurm/v9` is `scripts/v9/reg_g1_test_report.py` (+37/−8, the sealed-test report writer; not on any training or registration path). The sbatch code hashes recomputed now — G1 `6c6f094684fbc50a`, G2 `99c942022c5164f3` — equal the hashes printed by the formal G1 jobs (1041930, 1041932, 1041935, 1041937, 1041939) and G2 jobs (1042139, 1042143, 1042154).
2. **Is the current G1 negative term the package's V mode?** The *reference* is V: Q is within-scene, built from an ordinary uniform permutation (`numpy.random.Generator.permutation`), **fixed points allowed**, no derangement. Its conditional expectation over the permutation is exactly the package's V mode (all n² within-scene pairs incl. the diagonal, denominator n²). The *estimator* is a single draw (**perm1**), not the V moments: training draws a fresh permutation per scene per step (G L157, L171); registration keeps one permutation of the 4096 positions for the whole search of one (image, perturbation) (Q L159). Replicated draws (same seeds, numpy only): 400 registration permutations → mean 1.05 fixed points (max 6), 66.3 % with ≥ 1; 3200 training permutations (seed 901, fraction 1, first 200 steps) → mean 1.053, 65.3 % with ≥ 1 (theory 1 and 63.2 %).
3. **Forbidden edges: none** in any Q (critic training, registration, kernel fit). The only spatial exclusion in the host is the ε > 0 mismatch generator (displaced positive at ≥ 2 patch radii), which corrupts P and removes no Q edge. V10 primary uses ε = 0.
4. **Target vs package assumption.** No change of reference is needed (V = the host's conditional target). Differences recorded in §5 (nothing changed): D1 tanh-MLP critic vs bounded bilinear; D2 perm1 vs exact V moments (no perm8 / V option in the host); D3 logistic- and InfoNCE-trained critics are registered with the same J(tanh f) perm1 energy; D4 training scene = 160-px crop, equal scene weights; D5 N = 128 (training) / 4096 (registration); D6 ε axis; D7 weight decay 1e-4 on everything incl. the would-be W; D8 approximate grid stage; D9 float32 hard-coding.
5. **Adapter boundary.** Training descriptors come from `P.build_models` + `P.sample_map`, but the batch assembly is inline in `G.cmd_train` L147-166 (no callable): a V10 trainer must reproduce it with the same RNG call order. Registration descriptors meet the critic in exactly one method, `Q.RigidEnergy.j_critic(self, dv)` (dv `[K, N, 16]` → `[K]`), used by both the exact path (`batch`) and the grid (`grid_values`); a V10 subclass overriding `j_critic` gets the warp chain unchanged, and the geometric gradient reaches both the P and the Q term through the same `dv`. `G.cmd_*`, `G.ensure_layout`, `G2.*cmd_*`, `Q.cmd_register` and every `slurm/v9/*.sbatch` write into V9 roots or run the V9 worktree and must not be called by V10.

## 1. Repository, commit and job state (read-only)

| worktree | branch | HEAD | dirty |
|---|---|---|---|
| `/home/infres/yinwang/CS_QMI/ssl_pilot_v10` | `v10/mf-vcs` | `f652f1eb734522504085e19585b7084b02b9f6f9` (parent `5f08b2f`) at the audit snapshot; later heads = parallel T00 commits on V10 paths only (this commit's parent is in git) | clean at the snapshot; before committing, files of the parallel T00 task were modified / untracked (`v10/T00/EXPOSURE_NOTES.md`, `MANIFEST_BUILD_SUMMARY.json`, `MANIFEST_SHA256.txt`, `PREFLIGHT.json`, `PREFLIGHT_RUN.log`); they are not part of this commit |
| `/home/infres/yinwang/CS_QMI/ssl_pilot_v9` | `v9/reliable-correspondence` | `5f08b2fb378b739367191b6968006c526d4a3741` | ` M reports/V9/V9_N2_units/launch_ledger.jsonl`, `?? data/DATA_MANIFEST.json.lock`, `?? reports/V9/V9_N2_units/release/` (not touched) |

Package historical host commit `27117c9ae78d64d39a6548c40b00ff58f70cc4d4` exists in the repository (2026-10-16 03:49:26 +0200, a V9-N2 corrections commit); see §0.1 for the diff.

Queue snapshot (`squeue -u yinwang`, 2026-10-16T20:14:25Z; identical set at 20:02:59Z; normal-QOS count 27):

| V9 line | jobs |
|---|---|
| V9-N2 running (8 = per-user GPU cap) | 1042343 t06 (L40S node52), 1042345 t08 (L40S node50), 1042346 t09 (L40S node50), 1042347 t10 (RTX6000PRO node58), 1042348 t11 (RTX6000PRO node59), 1042349 t12 (L40S node57), 1042350 t13 (RTX6000PRO node58), 1042351 t14 (H100 node53) |
| V9-N2 pending | 1042352 t15, 1042355 t16, 1042382 t17, 1042417 t18, 1042474 f02, 1042475 f03, 1042550 f04, 1042630 f05, 1042729 f06, 1043044 f07, 1043080 f08, 1043126 f09, 1043258 f10, 1043335 f11, 1043495 f12, 1043575 f13 |
| V9-G2 pending | 1042476 `v9g2_infonce_rerun` (registration-only rerun of V9-G2-infonce-frac0.25-eps0.2-s903 and -frac1-eps0.2-s903, cut by the 6 h limit; 3 h limit, QOSMaxGRESPerUser) |
| V9-G1 | nothing queued: 52 / 52 formal unit results present (`outputs_v9/reg/units/*/p2/results`), aggregate done, sealed-test read done (ledger `reports/V9/V9_TEST_READ_LEDGER.csv`, read_utc 2026-10-15T23:36:09Z, results commit 411248e) |
| V9-G2 | 50 / 52 unit results present (the two infonce units above missing) |

Also in the normal QOS (not V9): 1040276 `c3_ckptoff`, 1039458 `no-shell` (CPU). Nothing was cancelled, held or updated.

## 2. Host source inventory and call graph

| file | lines | sha256[:16] | role on the G1 / G2 path |
|---|---:|---|---|
| `src/v9/reg/g1.py` | 305 | b6122bc731d3d6b5 | FIT-fraction sampler, mismatch generator, critic / CoMIR / kernel training, registration wrapper |
| `src/v9/reg/r6_reg_pilot.py` (P) | 465 | c044adca00034853 | data / cache, encoders + critic, J, losses, `sample_map`, kernel fit, CoMIR model. Its own translation-only registration (`Energy.__call__` L241-255, `optimise` L268-285, `register_images` L377-393, `cmd_register` L400-450) is **not** on the G1/G2 path; `P.cmd_train` / `P.cmd_comir` are superseded by G's copies |
| `src/v9/reg/r6_reg_p2.py` (Q) | 373 | 0e248cac8cd5c72b | rigid geometry, perturbations, `Inputs`, `RigidEnergy`, shared optimiser, `register_images`, CAL selection, summaries |
| `src/v9/reg/g2.py` | 242 | 08231e880130a00b | G2: G's training reused in-process (`activate`), REG-P3 registration |
| `src/v9/reg/r6_reg_p3.py` (P3) | 551 | 5bbc3ebaf11b0cdc | EPFL checks / units / roles / windows, `Inputs3`, unit-pooled `register_one` |
| `src/v9/reg/g1_test.py` | 381 | 6b2dd6dc195c51ab | sealed-test read (done); same `Q.register_images` path; not for V10 phase 1 |
| `src/v9/reg/g1_aggregate.py`, `g2_aggregate.py` | 123, 128 | 21ff827d58f1c9aa, 0c0da8a444af19c3 | readings |
| `scripts/v9/reg_g1.py`, `reg_g2.py` | 92, 91 | 76c0fbbcef65f434, 7601469b42bf02e2 | CLIs: `run --unit` = train if needed, then register (`scripts/v9/reg_g1.py` L28-63, `scripts/v9/reg_g2.py` L32-60) |
| `scripts/v9/reg_g1_test.py`, `reg_g1_test_fixture_diag.py`, `reg_g1_test_report.py`, `reg_g2_report.py` | 98, 46, 137, 104 | 6af24d12f6815072, 949aaa4cf85cd9e1, 681ddb0b2e8aea16, cd5653d4a320adbb | test CLI, repeatability diagnostic, report writers |
| `slurm/v9/reg_g1.sbatch`, `reg_g2.sbatch`, `reg_g1_test.sbatch` (+ gate / timing / aggregate / fixture sbatch and feed files) | 19, 19, 21 | f3e688bb6e5e91da, 4c78a2481483217f, 54332edba1f0a3e2 | every one does `cd /home/infres/yinwang/CS_QMI/ssl_pilot_v9` and writes under `outputs_v9` / `reports/V9` |

Call graph.
- **G1 train**: `slurm/v9/reg_g1.sbatch` → `scripts/v9/reg_g1.py cmd_run` → `G.cmd_train` L140-182 | `G.cmd_comir` L185-234 | kernel fit inside `G.cmd_register` L268-277 → `G.fit_kernel_g1` L237-253.
- **G1 register**: `G.cmd_register` L262-278 → `G.ensure_layout` L129-136 (symlinks R6_REG cache/manifest and R6_REG_P2 perturbations into `outputs_v9/reg/units/<uid>`) → `Q.cmd_register` L289-357 → `Q.Inputs` L113-130, `Q.register_images` L250-265 → `Q.RigidEnergy` L150-219 + `Q.optimise_rigid` L228-246 → `Q.corner_error_full` L93-97; selection `Q.sel_key` L268-271; `Q.summarise` L279-286.
- **G2 train**: `G2.cmd_train` L179-183 = `G.cmd_train` inside `G2.activate` L132-143 (rebinds `G.unit_root`, `G.V9_ROOT`, `G.P1_ROOT`, `G.P2_ROOT`, `G.sample_fit` → `G2.sample_fit_units`, `G.state_hash` → `G2.state_hash_epfl`).
- **G2 register**: `G2.cmd_register` L204-226 → `P3.register_one` L440-517 → `P3.Inputs3` L408-418, `Q.register_images`, `Q.RigidEnergy`, `Q.optimise_rigid`; selection on unit-pooled CAL records (`P3.pool_units` L421-427, P3 L448).

sbatch environment (V9 jobs): `slurm/v9/reg_g1.sbatch` / `reg_g2.sbatch` source `/home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh` (sets `PYTHON=/home/infres/yinwang/CS_QMI/env/bin/python`, `PYTHONUNBUFFERED=1`, OMP / MKL threads 4, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `cd` into `ssl_pilot` and `mkdir -p` of its output roots), then override `PYTHONPATH=$R/src:$R` and `cd $R` with `R=/home/infres/yinwang/CS_QMI/ssl_pilot_v9` (L12-13). Resources: `reg_g1.sbatch` partitions RTX6000PRO,H100,L40S, `--exclude=node51` only, 1 GPU, 8 CPUs, 64G, 10 h default (feed: 8 h critic groups, 2 h CoMIR / kernel); `reg_g2.sbatch` `--exclude=node51,node60`, 8 CPUs, 64G, 8 h default (feed: 6 h critics, 2.5 h CoMIR, 3 h kernel). A V10 sbatch must set its own `R`, PYTHONPATH and exclude list.

Module identity: G inserts `src/v9/reg` into `sys.path` (G L35-36) and imports `r6_reg_pilot` / `r6_reg_p2` as **top-level** modules (G L37-38); P3 does the same (P3 L60-62) and G2 asserts `P3.P is P and P3.Q is Q` (G2 L38-39). A V10 adapter must use `G.P` / `G.Q` (or the top-level names after importing g1); `import v9.reg.r6_reg_p2` would create a second module object.

## 3. Contract table — `HOST_ADAPTER_CN.md` 「必须在当前源码记录的映射」

### 3.1 数据加载 / data loading

| aspect | BF-SHG (G1) | EPFL RGB-NIR (G2) |
|---|---|---|
| raw files | `/projects/EEG-foundation-model/yinghao/data/bf_shg_4550300/{train,validation1,validation2}/<core>_WB.tif` (BF RGB) + `<core>_MI.tif` (SHG): 40 / 25 / 7 pairs; `closed/test.zip` = sealed TEST (not used) | `/projects/EEG-foundation-model/yinghao/data/epfl_rgb_nir/nirscene1/<category>/<id>_rgb.tiff` + `_nir.tiff` (uint8) |
| reader / discovery | `P.core_files` L67-75, `P.read_tiff` L43-64 (uncompressed little-endian TIFF only) | `P3.epfl_pairs` L109-118, `P.read_tiff` |
| native size | 2048 × 2048; BF RGB uint8 ×3, SHG uint8 (manifest `full_shape`, `shg_dtype`, 72 / 72) | e.g. 682 × 1024 (H × W, most frequent shape) |
| working image | 512 × 512 = ×4 block mean (`P.area_down` L84-86; `SIZE, DOWN = 512, 4` P L36) | one centred 512 × 512 native window per kept pair (`P3.centred_window` L203-206), `down = 1` |
| cache | `/home/infres/yinwang/CS_QMI/outputs/R6_REG/cache/<core>.npz` (72), by `P.cmd_prepare` L93-111: `bf` (robust-normalised luminance), `shg`, `bf_rgb`, `shg_log` (= log1p SHG), `bf_rgb_n` (robust-normalised RGB), `shg_log_n` (robust-normalised log1p SHG) | `/home/infres/yinwang/CS_QMI/outputs/R6_REG_P3/cache/<window>.npz` (420), `P3.cache_arrays` L85-90: same keys; `shg = shg_log_n` = robust-normalised **linear** NIR (no log); `nir` = raw NIR / 255 |
| normalisation | `P.robust_norm` L89-90: (a − median) / max(IQR, 1e-6) over the whole array — for RGB one median / IQR jointly over pixels and the 3 channels | same function per window |
| learned-method inputs | `P.learned_inputs` L156-159 → U = `bf_rgb_n` `[3,512,512]`, V = `shg_log_n` `[1,512,512]`, float32 on device | same keys (V = linear NIR) |
| manifest | `outputs/R6_REG/manifest.json` (sha256[:16] b8a9e3e7bce6abca): roles, per-core split / role / sha256 of both files / shapes; `omega [117,394]`, `size 512`, `down 4` | `outputs/R6_REG_P3/manifest.json` (3ecf9fe8e7652413): roles, windows, pairs, units, exclusions, rules |
| split source | `P.roles` L78-81: official `train` sorted by sha256(core id) → first 8 CAL, rest FIT; `validation1` + `validation2` → DEV. Counts FIT 32 / CAL 8 / DEV 32 (25 + 7) | no official split; `P3.roles_epfl` L192-200: per category, units by sha256('R6-REG-P3\|EPFL\|unit\|' + unit): 8 DEV, 1 CAL, rest FIT. Units FIT 330 / CAL 9 / DEV 72; windows FIT 336 / CAL 10 / DEV 74 |
| exclusions | none | 477 pairs: 33 alignment (`P3.alignment_check` L133-146, `P3.misaligned` L148-150), 24 size (< 512 px); 420 kept; 13 near-duplicate links merged (`P3.near_duplicate_links` L153-167, `P3.units_epfl` L170-189) → 411 units (8 multi-pair) |
| scene / core / parent ID | core id = file stem before `_WB.tif` (format `<slide>_<position>`, e.g. `1B_A2`) = TMA core = one image pair; stratum = slide prefix 1B / 2B (`G.specimen_stratum` L62-63; 38 / 34 of the 72 cores); no patient relation in the host | window id `<category>_<id>` → unit (`+`-joined pair ids) → category |
| FIT-fraction sampler | `G.sample_fit` L66-76 (by core, slide strata, largest remainder, rng tag `V9-G1\|fit\|<seed>\|<frac>`) | `G2.sample_fit_units` L84-98 (by unit, category strata, tag `V9-G2\|fit\|...`) |
| training batch fields | no Dataset / DataLoader: all sampled FIT images resident on device (G L143); per step (G L147-166): crops `[n_sc,3,160,160]` / `[n_sc,1,160,160]`; `DU, DV` `[n_sc,16,160,160]`; `yx` `[n_sc,128,2]` float32 crop coordinates (y, x); `yx_v` moving-side positions (= `yx` at ε 0); `perm` `[n_sc,128]` int64; `du, dv` `[n_sc,128,16]`; `dvq = dv[s, perm[s]]` | identical code (G under `activate`) |
| registration inputs | `Q.Inputs` L113-130: `reference(c)` = (`bf` `[1,512,512]`, `bf_rgb_n` `[3,512,512]`); `moving(c,i)` = (robust_norm(V_p), robust_norm(log1p V_p)), V_p = `Q.perturb_map`(expm1(`shg_log`), θ*, t*) (bilinear, reflection); normalisation recomputed on the perturbed image | `P3.Inputs3` L408-418: moving = robust_norm(`Q.perturb_map`(`nir`)) on both paths |
| perturbations | `outputs/R6_REG_P2/perturbations.json` (ca59e51ef36f60e1): 40 images (8 CAL + 32 DEV) × 10 `[θ°, ty, tx]`; `Q.perturbations_p2` L101-104 (t* = `P.perturbations` L127-132, strata 0-17 / 17-34 / 34-51 px at 512, tag `R6-REG-P1\|<core>`; θ* ~ U(−30°, 30°), tag `R6-REG-P2\|<core>`) | `outputs/R6_REG_P3/perturbations.json`: 84 windows (10 CAL + 74 DEV) × 10 (`P3.cmd_perturb` L384-390) |

### 3.2 图像编码 / image encoders

| aspect | learned critics (vcs / logistic / infonce) | CoMIR |
|---|---|---|
| builder | `P.build_models` L148-153 | `P.comir_models` L305-317: official DenseUNet (`/projects/EEG-foundation-model/yinghao/third_party/CoMIR`, commit 1750b11e29532cd4096bd3060d086ecac9b9291c, `models/tiramisu.py`); init_conv_filters 32, down / up blocks (4,)×6, bottleneck 4, upsample, max pooling, dropout 0.2, compression 0.75, `efficient=True`, final 1×1 conv to 1 channel (no bias) |
| reference side | `eU = enc(3)`: BF RGB `bf_rgb_n` | `mB` (3 channels, BF RGB) |
| moving side | `eV = enc(1)`: SHG log1p `shg_log_n` (EPFL: linear NIR) | `mA` (1 channel, SHG log1p / NIR) |
| layers | Conv2d(cin,32,7,pad 3) GELU Conv2d(32,32,9,pad 4) GELU Conv2d(32,32,9,pad 4) GELU Conv2d(32,16,9,pad 4); stride 1, zero padding; no coordinates / ids | DenseLayer = BN-ReLU-Conv3×3-dropout; TransitionDown = BN-ReLU-Conv1×1-Dropout2d-MaxPool |
| receptive field | 31 px (1 + 6 + 8 + 8 + 8; half-width `RF_HALF = 15`, P L36, L149) | not local (whole-frame U-Net) |
| output shape | `[B,16,H,W]` (same size) | latent `[B,1,H,W]` |
| critic head | Linear(32,64) GELU Linear(64,64) GELU Linear(64,1) on cat(du, dv) (P L152); vcs uses T = tanh f | −MSE of latents (registration) |
| BatchNorm / dropout | none | BatchNorm2d (tiramisu.py L41, L120); `F.dropout(p=0.2, training=self.training)` (L61-62); `Dropout2d(0.2)` (L133-134) |
| train / eval mode | training never calls `train()`/`eval()` (no BN / dropout, so no effect); registration: `.eval()` + `requires_grad_(False)` on every parameter (Q L333-337, P3 L489-494) | `mA.train(); mB.train()` per epoch (G L198); `eval()` before latent export (G L223) and at registration (Q L313); `efficient=False` on `mA` modules at registration (Q L316-318) |
| initialisation | `torch.manual_seed(seed)` (G L141) then construction order enc(3), enc(1), critic (P L153): encoder init does not depend on the critic | `torch.manual_seed(seed)` inside `P.comir_models` (P L316) |
| checkpoint | `run/ckpt_<step:05d>.pt` = `{'eU', 'eV', 'critic'` state_dicts, `'step'}` (G L175), every 250 steps → 12 (Q L331-332 / P3 L487-488 refuse ≠ 12 outside smoke); no optimiser or RNG state. Verified on `outputs_v9/reg/units/V9-G1-vcs-frac1-eps0-s901/run/ckpt_03000.pt` (1,718,005 B): `eU.0.weight [32,3,7,7]`, `eU.{2,4}.weight [32,32,9,9]`, `eU.6.weight [16,32,9,9]`, biases `[32]`/`[16]`; `eV.0.weight [32,1,7,7]`, rest as eU; `critic.0.weight [64,32]`, `critic.2.weight [64,64]`, `critic.4.weight [1,64]`; `step 3000` | `run/comir.pt` = `{'A', 'B', 'args'}` (G L223); latents `run/latents/<core>.npz {shg, bf}` for CAL / DEV (G L224-229) |

Kernel row: no encoder; raw 9 × 9 patches of `bf` and `shg`, random Fourier features (`G.fit_kernel_g1` L237-253). MI / NMI: `bf` and `shg` intensities, Parzen 64-bin soft histogram (`Q.soft_mi_batch` L139-147).

### 3.3 配对采样 / pair sampling

| aspect | training (critics) | registration (all shared-searcher energies) |
|---|---|---|
| scene | one aligned random 160 × 160 crop of one core, y0, x0 ~ integers[0, 352) (G L150), identical rot90 k ∈ {0,1,2,3} and horizontal flip for both modalities (G L152-154); training at φ = 0 (no geometric transform) | one (image, perturbation index) |
| scenes per step | `n_sc = min(16, len(fit))` (G L146): 16 at fraction 1, 8 at fraction 0.25. `replace = len(fit) < n_sc` is never true → the crops of one step come from **distinct cores** (G L149; confirmed on 200 replicated steps) | — |
| positions per scene | 128, continuous uniform on [16, 143)² in the augmented crop frame (y, x), pixel centres (G L146, L157; `inner = RF_HALF + 1`) | `N_POS = 4096`, continuous uniform on the disk Ω of radius 129 about c = (255.5, 255.5): r = 129 √U(0,1), angle U(0, 2π) (Q L157-158) |
| positive pairs | (du_i, dv_i) at the same position (ε 0) (G L165-166) | (u_i, dv_i) with dv the warped-then-encoded moving descriptors at the same x_i |
| negative pairs | (du_i, dv_π_s(i)), π_s = `Generator.permutation(128)`, fresh per scene per step (G L157, L171) | (u_i, dv_π(i)), π = `rng.permutation(4096)` (Q L159), fixed for grid, Adam, F_true and profiles |
| permutation RNG | data rng `default_rng(1000 + seed)` (G L141), consumed identically by vcs / logistic / infonce (π drawn every step before the objective branch, also for infonce) | `default_rng(seed_of('R6-REG-P2\|pos\|<core>\|<i>'))` (Q L255, `seed_of` L64-65) → identical x and π for every method and checkpoint |
| fixed points | allowed (replicated: mean 1.053 per permutation, 65.3 % with ≥ 1) | allowed (replicated: mean 1.05, max 6, 66.3 % with ≥ 1) |
| forbidden edges / spatial exclusions in Q | none | none |
| coordinate system | crop pixel-centre (y, x) after augmentation | reference-frame pixel-centre (y, x) of the 512 image |

Other objectives: InfoNCE training scores the full within-scene n × n matrix f(du_i, dv_j) with target j = i, U anchors only (G L167-169, `P.loss_fn` L178-179) — no permutation. Kernel fit: 2048 positions per FIT core, uniform on the REG-P1 square `OMEGA` [117, 394)², one permutation per core (G L242-247). CoMIR: 32 crops per step (182 px, flip, rotation U(−180°, 180°) with reflection, central 128), NT-Xent over 2N latents on MSE similarity / τ 0.5 with only the self-similarity diagonal masked (G L200-219; native).
ε > 0 (V9 axis only): exactly round(ε n) rows per scene get a displaced moving position (`G.mismatch_positions` L88-96, `G.displaced` L79-85), uniform on the same crop interior, rejected when closer than 2 × 15 = 30 px to the true one; Q is then built from these observed moving descriptors. The exclusion corrupts P and never removes a Q edge.

### 3.4 训练目标 / training objective

| aspect | host |
|---|---|
| J | `P.j_value` L169-170: J = mean_P(T − T²/2) + mean_Q(−T − T²/2) = E_P T − E_Q T − ½E_P T² − ½E_Q T² (the collaborator's J) |
| vcs | loss = −J(tanh f_P, tanh f_Q) (`P.loss_fn` L174-175) |
| logistic | softplus(−2 f_P).mean() + softplus(2 f_Q).mean() (L176-177) |
| InfoNCE | cross_entropy(f_mat.reshape(S·n, n), arange(n).repeat(S)) (L178-179) |
| P / Q coefficients | P: +1·T, −½·T²; Q: −1·T, −½·T² |
| scene weights | implicit equal 1/n_sc: `.mean()` over the flattened `[n_sc, n_pos]` tensors with equal n_pos = 128; positions 1/n_pos |
| sign | training minimises −J; registration maximises F (J for critics and kernel; MI; NMI; −MSE for CoMIR) |
| optimiser | AdamW(lr 1e-3, weight_decay 1e-4, default betas / eps) over eU + eV + critic jointly (G L144-145); `clip_grad_norm_(all params, 1.0)` (G L173); 3000 steps (smoke 200); no LR schedule; checkpoints every 250 steps |
| AMP | none: no autocast / GradScaler / `.half()` / bfloat16 / float16 in `src/v9/reg` or `scripts/v9/reg_*`; float32 tensors |
| regularisation | decoupled weight decay 1e-4 on all parameters incl. biases and critic; grad-norm clipping 1.0; crop / rot90 / flip augmentation; no dropout, BatchNorm or ridge on the critic path |
| CoMIR | SGD lr 1e-2, momentum 0.9, weight_decay 1e-5 per network; L1 + L2 activation decay 1e-4; clip 1.0 per network; 23 epochs × 32 steps × 32 crops; p4 rot90 equivariance (G L192-221) |
| kernel | closed form w = (A + 1e-3·mean(diag A)·I)⁻¹ d, A = E_P ψψᵀ + E_Q ψψᵀ, d = E_P ψ − E_Q ψ, ψ = √(2/1024) cos(zW + b), T = w·ψ (no tanh); h = mult × median pairwise distance, mult ∈ {0.5, 1, 2} chosen on CAL (`G.fit_kernel_g1` L237-253; Q L304-310) |

### 3.5 几何 / geometry

| aspect | host |
|---|---|
| transform and direction | `Q.apply_T` L69-72: T_{θ,t}(z) = R(θ)(z − c) + c + t maps **reference → moving** coordinates; z = (y, x); R(θ) = [[cos θ, −sin θ], [sin θ, cos θ]] acting on (y − c, x − c); c = 255.5 (Q L52); rotation about c, then translation |
| perturbation | `Q.perturb_map` L86-90: V_p = V ∘ T*⁻¹, V_p(y) = V(R(−θ*)(y − c − t*) + c), bilinear, reflection padding; every method sees (U, V_p) only |
| units | θ in radians inside `apply_T`; degrees in `perturbations.json` and result records; t in working px |
| pixel centres / align_corners | integer coordinates are pixel centres; grid = (coord + 0.5)/size·2 − 1 with `align_corners=False` (`P.sample_map` L162-166, `Q.sample_pts` L75-78) |
| interpolation | bilinear everywhere |
| padding | zeros in `Q.sample_pts` (default) and `P.sample_map`; reflection in `Q.perturb_map` and in the `comir_wte` full-frame warp. By the support bound below neither padding reaches a sampled value of the critic / MI / kernel energies |
| warp order, critics | **warp-then-encode** (`Q.RigidEnergy.batch` L194-198): per candidate the moving image is resampled onto BOX1 = [110, 401]² (292 × 292) of the reference frame, encoded by eV, descriptors sampled at x − 110 (= `V10_PROTOCOL` `geometry_order_shared_energy`) |
| grid stage, critics | `Q.RigidEnergy.grid_values` L200-211: rotation-only warp of BOX2 = [59, 452]² (394 × 394) encoded once per angle; translations applied as descriptor shifts R(−θ)t (translation-equivariance approximation; exact energies only in the refinement) |
| CoMIR / kernel / MI | CoMIR encode-once: latents of V_p as given sampled at T(x) (Q L319-325, P3 L475-481); `comir_wte` = DEV-only sensitivity row. Kernel: 9 × 9 offsets added in the reference frame then mapped by T (Q L164-167, L185-187). MI / NMI: intensities at T(x) (Q L180-182) |
| support / visibility | one fixed interior disk Ω (radius `R_OMEGA = 129` about c, Q L54) for every method; 129 + 16√2 + 51 + 51 + √2 = 255.04 ≤ 255.5, so every sample (critic receptive field and bilinear neighbours included) stays inside V for all allowed (θ, t) and (θ*, t*). Receptive-field reach inside BOX1 = [0.5, 290.5] of [0, 291] and inside BOX2 = [0.5, 392.5] of [0, 393] → conv zero padding never touches a sampled descriptor. No visibility mask, no per-candidate support |
| search | `Q.optimise_rigid` L228-246: grid θ ∈ {−30°, −27°, …, 30°} (21) × 4-px translation disk \|t\| ≤ 51 — axis values −51, −47, …, 49 (26; **asymmetric**, +51 not on the grid), 508 disk points → 10,668 candidates (= `n_grid` in the G1 / G2 results), under `no_grad`, NaN → −inf (L234); the best 3 refined by 100 Adam steps (lr 0.5) on p = (u, ty, tx), u = θρ, ρ = 129/√2 = 91.2168 (Q L55); after each step u is clamped to ±UMAX = 47.761 and t projected onto \|t\| ≤ 51; result = best exact energy among the 3 |
| initialisation / multi-start | grid top-3 only (`N_STARTS = 3`, P L38) |

### 3.6 评估 / evaluation

| aspect | host |
|---|---|
| error | `Q.corner_error_full` L93-97: mean over the 4 corners at (±104.125, ±104.125) working px about c (`HALF834 = 833/2/4`, Q L59) of \|T̂(corner) − T*(corner)\|, × `P.DOWN` = 4 |
| native px | BF-SHG: err_full = full-resolution (2048-px) px; corners = CoMIR's central 834 × 834 full-resolution patch. EPFL: native px = err_full / 4 (P3 L74 `ERR_UNIT`; `G2.THRESHOLDS_NATIVE` L46); 208.25-px corner patch about the window centre |
| thresholds | success = err_full < 42 and < 9 (`P.THR_FULL` L39; strict) = BF-SHG 42 / 9 native px; EPFL 10.5 / 2.25 native px. Scale asserted by `tests/v9/test_reg.py` L81-85 (10.5 working px → 42; 2.25 → 9; 4.09° → ≈ 42) |
| per-registration record | θ, t, θ̂, t̂, err_full, rot_err_deg = \|θ̂ − θ*\|, trans_err_full = \|t̂ − t*\|·4, F_hat, F_true (exact energy at the truth), n_grid, seconds (Q L257-259); exact F profiles for the first 5 DEV images (Q L261-264) |
| success aggregation | BF-SHG: `P.success` L396-397 = mean over images (cores) of the per-image success fraction; EPFL additionally unit-pooled (`P3.pool_units`, P3 L507-510) |
| failure states | `Q.summarise` L279-286: failure = err_full ≥ 42; energy failure if F_hat ≥ F_true, else search failure. No per-registration numerical-failure, timeout or non-convergence state exists; a missing unit = absent result file |
| fixed model | chosen checkpoint loaded once, `eval()` + `requires_grad_(False)` (Q L334-337); reference descriptors cached per core per checkpoint (Q L338-342); nothing refitted per candidate |
| fixed sampling | x and π fixed per (image, perturbation index) for every method and checkpoint (Q L255); perturbations frozen in `perturbations.json` |
| selection | CAL, lexicographic max of (success < 42, success < 9, −median error, −checkpoint step) over the 12 checkpoints (`Q.sel_key` L268-271; EPFL on unit-pooled CAL, P3 L448); then DEV with the chosen checkpoint |
| statistics (V9) | G1: unit = core, seed t + core bootstrap 2000 (`g1_aggregate.contrast` L51-68, rng 20261016 at L92); G2: unit = EPFL unit, bootstrap 2000 (`g2_aggregate` L50-66) |

## 4. The negative term: V mode, fixed points, forbidden edges

Statement. In training (`G.cmd_train` L157, L171), in registration (`Q.RigidEnergy` L159, `j_critic` L173-176) and in the kernel fit (G L246-247), the negatives are the pairs (a_i, b_π(i)) of **one** ordinary uniform permutation π of the positions of the same scene. Fixed points are allowed — π(i) = i makes the positive pair also a negative pair, exactly as in the package's V mode. For any pair function g and π uniform over all permutations, E_π[(1/n) Σ_i g(a_i, b_π(i))] = (1/n²) Σ_{i,j} g(a_i, b_j), because π(i) is uniform on {1,…,n} for every i. Hence the host's Q is a one-draw Monte-Carlo estimate of the package's V-mode Q with the same within-scene reference and the same normalisation; it is not U (no diagonal deletion) and not an independent-pool Q.

- Estimator: perm1. The host has no perm8, no exact-V and no dense option; these are V10 implementations (`mf_vcs.sampled_permutation_terms`, `product_moments` / `objective_terms`, `dense_objective_terms`).
- Forbidden edges: none in any Q. The package's `SceneBatch(forbidden_edges=...)` limitation is therefore not triggered by the current host.
- Not V-mode objects: InfoNCE (n × n matrix with diagonal targets) and CoMIR's NT-Xent (self-similarity masked) are native objectives, not a VCS Q.
- No target change is needed; nothing was changed.

## 5. Differences between the current host and the package assumptions (recorded, not changed)

| id | item | current host | package | consequence for V10 |
|---|---|---|---|---|
| D1 | critic class | T = tanh(MLP(cat(du, dv))) (P L152, L174-175) | T = aᵀWb, simplex a, b, W ∈ [−1,1]^{r×s}, no output tanh | intended MLP → bilinear axis |
| D2 | Q estimator | perm1 (fresh per scene per step; fixed per registration) | V moments (M0 / M1); perm1 / perm8 as comparators | same reference; perm8 / V / dense must be implemented on the V10 side |
| D3 | registration energy of logistic / InfoNCE critics | the same J(tanh f) perm1 energy as vcs: `Q.cmd_register` L329-348 passes kind `critic` for vcs, logistic and infonce | `InfoNCE_host` "原生目标/候选归一化" | a V10 InfoNCE row reusing the host scores InfoNCE critics with J(tanh f); a native InfoNCE energy would be a new definition (decision not taken here) |
| D4 | training scene | 160-px crop of one core; cores distinct within a step; 16 scenes (8 at fraction 0.25); equal weights | per-scene J, fixed locked scene weights | for exact parity: scene id = crop index s (equivalently the core), weights = 1/n_sc |
| D5 | N | 128 per training scene; 4096 per registration | T02 N = 128 / 512 / 2048 / 8192 | both inside the T02 grid |
| D6 | mismatch | G1 / G2 run ε ∈ {0, 0.2}; at ε > 0 Q uses the displaced (observed) moving positions | `train_mismatch = 0.0` | only the ε = 0 path is the V10 host |
| D7 | W regularisation | AdamW weight decay 1e-4 on every parameter; clip norm 1.0 over all parameters | W weight decay 0 and box projection after each step | separate parameter group for W; whether W enters the clip norm is a V10 choice to record |
| D8 | grid stage | translations as descriptor shifts of a rotation-only encoding; exact refinement | shared searcher inherited | the same approximation applies to every shared-searcher method |
| D9 | dtype | float32 hard-coded: positions (Q L158), `grid_coords` (Q L82), p (Q L237) | fp64 small checks, fp32 primary | an fp64 check must cast positions, boxes, maps and modules (grid_sample needs input and grid of one dtype) |

## 6. Adapter boundary — what a V10 adapter can reuse

### 6.1 Import
`PYTHONPATH=/home/infres/yinwang/CS_QMI/ssl_pilot_v10/src:/home/infres/yinwang/CS_QMI/ssl_pilot_v10`; `from v9.reg import g1 as G; P, Q = G.P, G.Q`; for EPFL `from v9.reg import g2 as G2; P3 = G2.P3`. Read-only reuse of: `P.torch_mods`, `P.load_manifest`, `P.load_img`, `P.learned_inputs`, `P.build_models`, `P.sample_map`, `P.j_value`, `P.loss_fn`, `P.success`, `P.robust_norm`; `Q.seed_of`, `Q.apply_T`, `Q.sample_pts`, `Q.perturb_map`, `Q.corner_error_full`, `Q.Inputs`, `Q.RigidEnergy`, `Q.grid_points`, `Q.optimise_rigid`, `Q.register_images`, `Q.sel_key`, `Q.sel_row`, `Q.summarise`; `G.sample_fit`, `G.specimen_stratum`; `P3.Inputs3`, `P3.pool_units`, `P3.profile_windows`, `G2.sample_fit_units`, `G2.window_unit`.

### 6.2 Training side: per-scene dense descriptors before the critic
No callable returns them; the batch is assembled inline in `G.cmd_train` L147-166. A V10 trainer reproduces that block with the same RNG call order — `choice(fit, n_sc, replace=False)`; per crop `integers(0, 352, 2)`, `integers(4)`, `integers(2)`; `uniform(16, 143, (n_sc, 128, 2))`; `n_sc × permutation(128)`; (ε > 0: mismatch draws) — then `DU, DV = eU(stack crops), eV(stack crops)`, `du[s] = P.sample_map(torch, F, DU[s], yx[s])`, `dv[s] = P.sample_map(torch, F, DV[s], yx_v[s])` → `[n_sc, 128, 16]` each. MF features: A = softmax(left(du)/τ), B = softmax(right(dv)/τ); `mf_vcs.partition_scenes(A.reshape(-1, r), B.reshape(-1, s), torch.arange(n_sc).repeat_interleave(128), q_mode="v")` with weights `[1/n_sc]*n_sc` reproduces the host's scene weighting, and `mf_vcs.sampled_permutation_terms(..., permutations=[perm[s][None] for s in range(n_sc)])` reproduces the host's perm1 pairing (same (a_i, b_π(i)) convention; the host's `perm` is int64 on the feature device). Draw the permutations also for M0 so that crops and positions stay identical across MLP_VCS_perm1 / MF_VCS_perm1 / MF_VCS_M0. Encoders built by `P.build_models` after `torch.manual_seed(seed)` get the host's encoder initialisation (the MLP critic is constructed after them and can be discarded).

### 6.3 Registration side: descriptors and the single override point
Reference descriptors `DU[c] = eU(Q.Inputs.reference(c)[1][None])[0]` (`[16,512,512]`, as Q L338-342) are sampled once at the fixed x into `self.u` `[4096,16]` by `RigidEnergy.__init__` (Q L168-169). Moving descriptors are produced per candidate in `RigidEnergy.batch` (exact, Q L194-198) and `grid_values` (shifted, Q L209-211); both end in `self.j_critic(dv)` with dv `[K, 4096, 16]` → `[K]`. A V10 subclass overrides only `j_critic`: A = feat_a(self.u) `[N, r]` (constant per registration), B = feat_b(dv) `[K, N, s]`, fixed W; V mode t_ki = a_iᵀ W b_ki, q1_k = (Σ_i a_i)ᵀ W (Σ_i b_ki)/N², q2_k = ⟨W, G_A W G_Bk⟩/N², J_k = mean_i(t_ki − t_ki²/2) − q1_k − q2_k/2; MF_perm1 uses `self.perm`. Keeping the constructor keeps the RNG consumption, hence the same x and π for every method. `Q.register_images` (L255) instantiates `RigidEnergy` by module-global name, so without editing V9 files the subclass is used either through a V10 copy of `register_images` (16 lines) or through a context manager that temporarily rebinds `Q.RigidEnergy` (the `G2.activate` pattern). `Q.optimise_rigid`, `Q.sel_key`, `Q.summarise`, `Q.corner_error_full` are reusable unchanged. `RigidEnergy(**kw)` receives `{'eV': eV, 'critic': cr}` (Q L344, L348); the subclass can carry the MF head in `kw`.

### 6.4 Warp chain and gradient flow (P and Q)
`Q.optimise_rigid` L237-239 creates p = (θ₀ρ, ty₀, tx₀) float32 with `requires_grad=True`, `Adam([p], lr=0.5)`, `(-energy(p)).backward()`:
1. `RigidEnergy.__call__` L213-214 → `batch(p[0:1]/RHO, p[1:3][None])`;
2. `Q.apply_T(th, t, box1)` L69-72 (differentiable in θ, t);
3. `Q.sample_pts(mov, pts)` L75-78 → Wk `[1,1,292,292]` (bilinear; gradient w.r.t. the grid; V_p has no grad);
4. `eV(Wk)[0]` → D `[16,292,292]` (weights frozen, gradient w.r.t. the input);
5. `P.sample_map(D, x − 110)` L162-166 → dv `[4096,16]` (fixed positions; gradient w.r.t. D);
6. `j_critic(dv[None])` L173-176: P term uses dv, Q term uses `dv[:, perm]` → both carry ∂/∂(θ, t).

`no_grad` covers only the grid stage (Q L232-233), the final energy per start (L244-245), `RigidEnergy.value` (L216-219) and the reference encoding (Q L340-341) — never the refinement. So the package rules "P/Q moving side under the same transform, gradient through first and second moments" and "no_grad must not wrap the registration energy" are met by a `j_critic` override on the same dv; this is a source reading, the numerical check is T01.

### 6.5 Must not be called from V10
`G.ensure_layout`, `G.cmd_train`, `G.cmd_comir`, `G.cmd_register`, `G.unit_table`, `G2.ensure_layout`, `G2.cmd_train`, `G2.cmd_comir`, `G2.cmd_register`, `G2.make_smoke_root` create directories / symlinks / results under `outputs_v9/reg` or `outputs_v9/reg_epfl` (G L42, L125-136; G2 L41, L62-63, L147-171). `G2.activate` rebinds globals of the shared g1 module. `Q.cmd_register`, `P3.register_one` and `P.cmd_*` write under their root and hard-code the host checkpoint keys `eU / eV / critic` and the 12-checkpoint rule; V10 needs its own register command (encoder + feature maps + W as one pair). Every `slurm/v9/*.sbatch` runs the V9 worktree.

### 6.6 Numerics to carry into T01
- No TF32 / precision / determinism switch in any host file. Installed torch 2.9.0+cu128 defaults (`torch/include/ATen/Context.h`): `allow_tf32_cudnn = true` (L459) — fp32 convolutions may use TF32 on RTX PRO 6000 / H100 / L40S; `float32_matmul_precision = HIGHEST` unless `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE` (L454-457); `benchmark_cudnn = false` (L452). `common.sh` sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` and threads only. fp32 tolerance checks through the encoders should account for TF32 or use fp64.
- V9 measured repeatability (`reports/V9/V9_G1_TEST_READ_PROTOCOL_FROZEN_20261015.yaml` L101-128): MI / NMI / kernel / CoMIR registrations repeat bitwise on the same GPU type; the learned critics do not (largest same-device repeat difference 1.7608 px on RTX PRO 6000, 0.9817 px on L40S); differences also exist across GPU types. Cause not localised in V9. A V10 "restore checkpoint and repeat" check should report its spread instead of expecting bitwise equality.
- The host has equal n per scene everywhere; the package's "different-n fixture" check against position-weighting misuse needs a synthetic fixture.

## 7. Host cost facts (inputs for T02 / BUDGET)

| unit | job / GPU | training | registration |
|---|---|---|---|
| V9-G1-vcs-frac1-eps0-s901 | 1041939, node59, RTX PRO 6000 Blackwell | 3000 steps in 82.96 s, peak 0.967 GiB allocated | registration phase 955.4 s (12 × 8 × 10 CAL + 32 × 10 DEV = 1280 registrations, plus 5 DEV energy profiles); 0.767 s (CAL) / 0.734 s (DEV) per registration; peak 1.698 GiB |
| V9-G2-vcs-frac1-eps0-s901 | 1042139, node39, L40S | — | registration phase 2886.2 s (12 × 10 × 10 CAL + 74 × 10 DEV = 1940 registrations, plus 5 DEV energy profiles); 1.477 / 1.479 s per registration; peak 2.615 GiB |

Grid candidates per registration: 10,668. Source: `train.json` and result-JSON cost fields only; GPU from the job log header.

## 8. Not certified by T00 / open items

- T01 NOT_RUN: value and A / B / W gradient agreement of an MF energy on a real host batch (dense vs moments, fp64 then fp32).
- T01 NOT_RUN: continuous-warp geometric gradient through the subclassed `RigidEnergy` for both P and Q (central differences, ≥ 2 non-zero transforms, two step sizes, away from bilinear kinks).
- T01 NOT_RUN: fixed-W test check (model / W hash unchanged across candidates) and the checkpoint round trip of the V10 (encoder, feature maps, W) pair; the host checkpoint has no W and no optimiser state.
- BF-SHG patient / specimen relations are not in the host; the core is the finest verifiable unit.
- D3 (InfoNCE registration energy) is a protocol decision not taken here.
- V10 seeds 1001-1003 / 1017 differ from G1 (901-903) and R6 (0-2): under the V9 reuse rule (`G.r6_reusable` L111-121) no V9 unit is reusable as a V10 unit.

## 9. Evidence (all read-only; outputs in the evidence directory)

- `git rev-parse` / `git status --porcelain` on both worktrees; `git diff --stat 27117c9 HEAD -- src/v9/reg scripts/v9 slurm/v9` (only `scripts/v9/reg_g1_test_report.py`); sha256 of every mapped file in both worktrees (identical).
- `cat g1.py r6_reg_pilot.py r6_reg_p2.py | sha256sum` = 6c6f094684fbc50a and the G2 five-file hash 99c942022c5164f3, compared with the first line of the formal job logs in `/home/infres/yinwang/CS_QMI/slurm_logs/`.
- `squeue -u yinwang` (20:02:59Z and 20:14:25Z); result-file listing counts for `outputs_v9/reg` and `outputs_v9/reg_epfl`.
- `perm_fixed_points.py` (numpy only): replays `RigidEnergy`'s rng consumption (uniform N, uniform N, permutation N) for the 40 CAL / DEV images × 10 and the first 200 steps of `G.cmd_train`'s data rng (seed 901, fraction 1, ε 0) → `perm_fixed_points.json`.
- `ckpt_keys.py` (zipfile + restricted unpickler, no torch import, no tensor bytes) on one G1 checkpoint → `ckpt_keys_V9-G1-vcs-frac1-eps0-s901_ckpt_03000.json`.
- Geometry constants recomputed with numpy from the host formulas (R_OMEGA, ρ, UMAX, BOX1, BOX2, grid size, support bound); TF32 defaults read from the installed torch headers; CoMIR layer composition read from `third_party/CoMIR/models/tiramisu.py`.
