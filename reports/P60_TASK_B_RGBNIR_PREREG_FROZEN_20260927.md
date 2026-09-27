# Pre-registration — task pre-check B-T2: closed-form J* as a rigid-registration energy on REAL cross-spectral pairs (EPFL RGB–NIR Scene) — FROZEN 2026-09-27T15:14:21Z (owner accepted the data source)

Status: DRAFT (implementation fork); the main session freezes it.  **The dataset has NOT been downloaded**: the dataset page states no licence or
terms-of-use text, and the fork's directive was to stop unless the terms are clearly permissive for research use.  Everything else is ready
(`scripts/task_b_rgbnir.py`, `slurm/precheck_b_wave2_rgbnir.sbatch`, `slurm/download_rgbnir.sbatch`, synthetic code probe passed).

## Provenance (recorded before any download, brief §3)
- Dataset: RGB–NIR Scene Dataset, IVRL, EPFL — M. Brown and S. Süsstrunk, "Multispectral SIFT for Scene Category Recognition", CVPR 2011.
- Page (checked 2026-09-27 from the login node, metadata only): https://www.epfl.ch/labs/ivrl/research/downloads/rgb-nir-scene-dataset/
  (the old page https://ivrlwww.epfl.ch/supplementary_material/cvpr11/ has the same text).  Page text, verbatim: "This dataset consists of 477
  images in 9 categories captured in RGB and Near-infrared (NIR). The images were captured using separate exposures from modified SLR cameras,
  using visible and NIR filters. … The scene categories are: country, field, forest, indoor, mountain, oldbuilding, street, urban, water. …
  Download: Before downloading the full dataset, you might want to look at the smaller dataset browser (36Mb) … RGB-NIR Scene Data (1Gb). TIFF
  images at 1024×768 resolution. Processed and aligned as described in the Technical Details section below. Please contact us if you require
  higher resolution or RAW images."  Registration by the authors: SIFT + RANSAC similarity transform, resampled to a common frame.
- Archive: https://ivrlwww.epfl.ch/supplementary_material/cvpr11/nirscene1.zip — HTTP 200, content-length 952 431 486 bytes, last-modified
  2012-09-18 (HEAD request only).  Dataset browser: https://www.epfl.ch/labs/ivrl/wp-content/uploads/2018/08/nirscene0.zip (36 757 858 bytes).
- **Licence / terms: none stated** on either page (no "licence", "terms", "copyright", "non-commercial" or "research purposes" wording).  The
  dataset is offered for public download from an academic lab page and is widely used in published research with citation of BS11, but that is
  not an explicit permission.  **Decision needed by the owner:** (a) accept the public academic offering with citation as sufficient for a
  research pre-check; (b) e-mail the authors for written terms first; (c) use another cross-spectral source with an explicit licence.  Only after
  (a) or (b): `sbatch slurm/download_rgbnir.sbatch` (CPU partition; records sha256, bytes, url, utc and the terms text above in
  `data/rgb_nir/PROVENANCE.json`, extracts, counts pairs per category) and then this pre-registration is completed with the sha256, the pair
  counts and the 60-pair id list + its sha256.
- Identity: one RGB/NIR pair per scene (file stem); no other split is needed (nothing is trained).  60 pairs chosen by seed 20260927, stratified
  over the 9 categories (round-robin over per-category shuffles: 6–7 per category).

## Design (P52 protocol on real pairs; `scripts/task_b_rgbnir.py` imports `run_pair` / `aggregate` / `write_outputs` from the B2 script)
Modality A = grey(RGB), modality B = NIR (8- or 16-bit TIFF read to [0, 1]; centre square crop; 256 px); truth = identity (the authors'
registration; its residual error is a property of the data, shared by all measures).  J* (49-d Fourier class, ridge 1e-3, no tanh), MI, NMI
(32×32 histogram) on identical pixel samples (20 000) and overlap masks; translation surfaces ±24 px step 2 (θ = 0), rotation profiles ±30°
step 1 (t = 0); Nelder–Mead ≤ 150 evaluations from 10 random offsets per radius R ∈ {5, 10, 20, 30}; success = < 1 px and < 1°; divergence = |t| > 64 px.
Per-category results are kept in the JSON (`category` per pair).
**Pre-committed reading** (as P51/P52, with the plan's re-opening clause): the registration family is re-opened on real data only if J* matches
NMI on both the mean number of local maxima (≤ NMI's) and the success rates at R = 20 and R = 30 (within 0.05) — and **holds** in the P51 sense
if J*'s success at R = 20 and 30 is ≥ NMI + 0.10 with ≤ NMI's local maxima, zero divergence and peak-at-truth in ≥ 95 % of pairs; **does not hold**
if J* has more local maxima than NMI or lower success by ≥ 0.05.  Real RGB–NIR pairs are strongly (mostly monotonically) related in intensity, so
MI/NMI are expected to be strong here; a "holds" would be informative, a "does not hold" confirms P52 on real data.  Not claimed: non-rigid
transforms, other modalities (MRI), or the effect of the authors' registration residual.
**Probe (code only, disclosed).** `--synthetic-smoke`: 3 constructed pairs (COCO val2017 grey image + tent-map "NIR", saved as 8-bit TIFF in the
dataset's `<category>/<name>_rgb|_nir.tiff` layout) exercise the loader, the stratified selection and the full protocol on CPU (job 1011105).
**Compute.** One GPU job ≈ 1.7 h (A100; as P52), after the download (≈ 1 GB, CPU partition).

**Owner decision (2026-09-27, verbatim): "接受 EPFL 那份没有许可文本的数据，那个是自由下载的".**  The dataset is accepted as a publicly offered
academic dataset (Brown & Süsstrunk, CVPR 2011; cited as such); the page's lack of a licence text is recorded as found, and the archive's
sha256 / bytes / url / utc are written by the download job into `data/rgb_nir/PROVENANCE.json`.  The IXI alternative (P60 IXI draft) is kept as
the record of the detour and is not run.  Frozen now; chain: download (CPU) → 3-pair real smoke (GPU) → full 60-pair run (GPU); the 60-pair
id-list sha256 from the smoke is appended below before the full run is read.
