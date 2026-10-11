# R6-03 — data, licence, split and baseline audit — 2026-10-11

Companion of `R6_DATA_BASELINE_MATRIX.csv` (template columns).  Entry points checked on 2026-10-11 (metadata pages, READMEs and HTTP headers only).
**Nothing was downloaded.**  The one exception is a 64 KB read of the Google Drive confirmation page, to obtain the AV-MNIST size.  Local
search covered `/home/infres/yinwang/CS_QMI`, `/projects/EEG-foundation-model` and `/projects/common`.

- **An entry that exists is not a download.**
- **A licence that exists is not an approval.**  Every dataset that needs an agreement is marked "needs owner acceptance".

## 1. Usability at a glance
| dataset | route | status | why |
|---|---|---|---|
| D3 EPFL RGB–NIR Scene | REG | **usable now (local)** | Complete local copy (477 pairs, sha256 recorded).  The page has no licence text; the owner accepted the source on 2026-09-27 (P60). |
| D2 BF / SHG (TMA cores) | REG | **usable now (download needed: 577 MB crops or 1.8 GB full size)** | Open, CC BY 4.0, with official splits and a defined success criterion. |
| D1 Zurich Summer | REG | **usable now (download needed: 138 MB)** | Open, CC BY 4.0.  Only 20 tiles, and no published registration baseline. |
| D4 AV-MNIST (MultiBench) | GAIN | **after owner acceptance** | The ESC-50 noise component is CC BY-NC 3.0, so the derivative is non-commercial.  Audio independence cannot be verified (see §2). |
| D5 CREMA-D | GAIN | **after owner acceptance + installs** | ODbL / DbCL, with a registration form; media via Git LFS (~7.55 GB).  FLV decoding needs ffmpeg / PyAV, missing in both environments. |

None of the five was found locally except D3.  (The IXI PD / T2 data used in P60 is not an r6 candidate; see §4.)

## 2. Per dataset
**D1 Zurich Summer** — https://zenodo.org/records/5914759
- **Release and files:** v1.0, DOI 10.5281/zenodo.5914759, open; `Zurich_dataset_v1.0.zip`, 138.4 MB (md5 c79a08ec…).
- **Content:**
  - 20 GeoTIFF tiles of about 930 × 940 px (QuickBird, August 2002), with NIR / R / G / B pansharpened bands as raw DN;
  - sparse 8-class annotations.
- **Licence:** "Creative Commons Attribution 4.0 International".  Cite Volpi & Ferrari, CVPR-W EARTHVISION 2015, and keep the licence when
  redistributing.
- **Split:** no official split.  The page's example is zh1–7 / zh8–12 / zh13–20, and it asks users to report tile IDs.
- **Use for REG:**
  - The bands come from one sensor, so cross-band alignment is exact.  Known synthetic transforms give perfect ground truth.
  - The independent unit is the tile: 20 tiles from one acquisition, with adjacent tiles correlated.
  - CoMIR used Zurich only to test rotational equivariance and reproducibility of its representations (128 px patches from zh10 / zh12), not
    for registration.  So there is no published registration baseline here.

**D2 BF / SHG** — full size: https://zenodo.org/records/4550300 (v1, 2021-02-19); crops: https://zenodo.org/record/3874362 (2020-06-03)
- **Content:**
  - 206 aligned BF / SHG pairs of tissue-microarray cores, 2048 × 2048 TIFF.
  - The crop version has 834 × 834 crops, a transformed test set, and `metadata.csv` with corner coordinates.
  - The full-size files use suffixes `_BF` / `_SHG` (test) and `_WB` / `_MI` (train / validation).
- **Licence:** "Creative Commons Attribution 4.0 International", with credit to the creators.  The data come from the authors of DOI
  10.1038/s42003-020-01151-5.
- **Sizes:** full size 1.8 GB (test 1.2 GB, train 339 MB, val1 208 MB, val2 57 MB); crops 577 MB.
- **Splits (official):** 40 train, 25 val1 (CNN hyper-parameters), 7 val2 (registration tuning), 134 test.
  - **The test set stays held out:** development uses train / val only.
  - The two record versions must not be mixed.
- **Ground truth:** the pairs are aligned by the dataset authors; the method is not described on the record.
  - Test transforms (CoMIR): rotation up to ±30°, translation up to ±100 px, stratified small / medium / large (44 / 45 / 45).
  - **Success criterion (CoMIR):** failure if the mean corner error is above 100 px.  Successes are reported under <1 % (<9 px) and <5 %
    (<42 px).
- **Published numbers (CoMIR, NeurIPS 2020; out of 134):**

  | method | < 5 % | < 1 % |
  |---|---|---|
  | CoMIR(MSE) + SIFT | 98 | 75 |
  | CoMIR + multistart α-AMD | 90 | — |
  | CurveAlign | 33 | — |
  | MI on raw images | 30 | — |
  | GAN translation | ≤ 31 | — |

- **To check:** whether several cores come from the same patient or TMA slide; this is not documented.

**D3 EPFL RGB–NIR Scene** — page: https://www.epfl.ch/labs/ivrl/research/downloads/rgb-nir-scene-dataset/ (DNS failed from the fetch tool
today); old page: https://ivrlwww.epfl.ch/supplementary_material/cvpr11/index.html (resolves).
- **Content:**
  - 477 images in 9 categories, captured as separate RGB and NIR exposures with modified SLR cameras;
  - TIFF at 1024 × 768; `nirscene1.zip` about 1 GB; a 36 MB browser version; higher-resolution or RAW files on request.
- **Registration:** "The images were registered by extracting SIFT features at approximately 1500x2000 resolution", with the transform
  "recomputed via least squares from the inliers".  **The ground truth is the authors' registration**, so residual error is possible.
- **Licence:** **no licence or terms text** (re-checked 2026-10-11).  Citation: Brown & Süsstrunk, "Multispectral SIFT for Scene Category
  Recognition", CVPR 2011.
- **Local copy:** `/home/infres/yinwang/CS_QMI/data/rgb_nir/` — `nirscene1.zip` (952 431 486 B, sha256 7c465a23…f746bc) plus the extracted
  `nirscene1/` (9 folders, 954 TIFFs = 477 pairs).
  - `PROVENANCE.json` dates the download to 2026-09-27.
  - P60's frozen pre-registration records that the **owner accepted the data source**.
- **Independent unit:** the scene.  Consecutive captures within a category may be near-duplicates; check before splitting.

**D4 AV-MNIST (MultiBench)** — https://multibench.readthedocs.io/en/latest/start/datadownload.html
- **Download:** Google Drive `avmnist.tar.gz`, about 1.5 GB (size from Drive's confirmation page; public link-sharing).
- **Loader:** `datasets/avmnist/get_data.py` reads `image/{train,test}_data.npy`, `audio/{train,test}_data.npy` and `{train,test}_labels.npy`.
  - Images: 60 000 / 10 000 × 28 × 28.  Audio: 112 × 112.
  - Split: train[0:55000] / val[55000:60000] / test 10 000.
- **Construction (CentralNet, arXiv 1808.07275):**
  - MNIST images PCA-degraded to about 25 % of the energy;
  - Free Spoken Digit Dataset (FSDD) clips as 112 × 112 spectrograms, with ESC-50 noise added so the audio count matches MNIST.
- **Licences:**
  - MNIST: CC BY-SA 3.0 (Keras / TensorFlow docs; LeCun's page not re-checked).
  - FSDD: CC BY-SA 4.0 (3000 recordings: 6 speakers × 10 digits × 50; names `{digit}_{speaker}_{index}.wav`; FSDD's own test split is
    indices 0–4).
  - ESC-50: "Creative Commons Attribution Non-Commercial" (CC BY-NC 3.0; only ESC-10 is CC BY).
  - MultiBench's data page states no licence; the repository code is MIT.
  - → **needs owner acceptance** (non-commercial component).
- **Independence:** 70 000 audio samples are built from 3000 FSDD clips of 6 speakers, so clips and speakers repeat between train and test,
  and the released arrays carry no source IDs.  **Audio-side independence cannot be verified.**
  - A learned "audio gain" could partly be clip memorisation.
  - After download, near-duplicates must be checked, e.g. by correlating the spectrograms with each other.
- Inputs are spectrograms only: no raw-modality decoding cost can be measured.

**D5 CREMA-D** — https://cheyneycomputerscience.github.io/CREMA-D/; repo https://github.com/CheyneyComputerScience/CREMA-D; mirror
gitlab.com/cs-cooper-lab/crema-d-mirror.
- **Content:**
  - 7442 clips, 91 actors (48 M / 43 F, aged 20–74), 12 sentences;
  - `VideoFlash/*.flv`, `AudioWAV`, `AudioMP3`;
  - 6 emotions (ANG / DIS / FEA / HAP / NEU / SAD) × levels LO / MD / HI / XX;
  - 2443 crowd raters, each rating 30 audio-only, 30 video-only and 30 audio-visual clips; ratings in `finishedResponses.csv`,
    `processedResults/tabulatedVotes.csv`, `VideoDemographics.csv`.
- **File names:** `ActorID_Sentence_Emotion_Level`, e.g. `1058_IEO_SAD_HI.flv`.
- **Licence:** "made available under the Open Database License" (ODbL 1.0); "Any rights in individual contents of the database are licensed
  under the Database Contents License" (DbCL 1.0).  GitHub users are asked to fill a Google registration form "so we can keep a record of the
  community of CREMA-D users".  → **needs owner acceptance**.
- **Access:**
  - A full clone needs Git LFS, about 7.55 GB.
  - The ZIP (about 24 MB) contains **only LFS pointer stubs**.
  - If the GitHub LFS quota fails, use the GitLab mirror.
  - TF-datasets and some mirrors are audio-only and cannot substitute for the audio-visual data.
- **Split:** none official.  **Split by actor.**  Common audio-visual baselines (e.g. OGM-GE) use a random clip split, so their numbers are not
  comparable to an actor-disjoint split.

## 3. Baselines with public code
| baseline | route | repository / licence | status | offline? | notes |
|---|---|---|---|---|---|
| CoMIR | REG | github.com/MIDA-group/CoMIR — MIT | 51 commits; dates not shown | training: yes (PyTorch + tiramisu).  Registration: Fiji v1.52p (SIFT) and MATLAB 2019b (MI) | Release lists `model_biodata_{mse,cosine}.pt`, trained on the BF / SHG training set (Releases not rendered, so unconfirmed).  Notebooks `train-biodata.ipynb`, `train-zurich.ipynb`.  Porting it to a shared optimiser is not a verbatim reproduction. |
| MI / NMI (classical) | REG | SimpleITK and itk-elastix on PyPI (Apache-2.0) | not installed in either environment | yes, after pip install | MATLAB is not confirmed on the cluster.  CoMIR's MI baseline uses Mattes MI with a (1+1) evolutionary strategy. |
| In-house histogram MI / NMI and closed-form J* | REG | `scripts/task_b_rgbnir.py` (P60) | ran 2026-09-27 / 28 | yes | Same masks and samples for every measure.  Ready as a sanity baseline on D3. |
| MultiBench | GAIN | github.com/pliang279/MultiBench — MIT | active repository (1258 commits) | yes | AV-MNIST loaders and fusion baselines.  No CREMA-D. |
| AdaMML | GAIN | github.com/IBM/AdaMML — Apache-2.0 | **archived 2024-07-22** | yes (PyTorch, librosa) | RGB / audio / flow; Kinetics-Sounds weights released; adapting it to CREMA-D is our own port. |
| EDDI | GAIN | github.com/microsoft/EDDI — licence file not rendered | **archived 2024-06-13** | TensorFlow 1.4 | Tabular continuous data only (UCI Boston example, data not included).  Not runnable on images or audio without a rewrite; useful as a method reference. |
| OGM-GE (CVPR 2022) | GAIN | github.com/GeWu-Lab/OGM-GE_CVPR2022 — **conflicting: README says GPL-3.0, sidebar says MIT** | no archive banner | PyTorch 1.8.1 | Standard CREMA-D audio-visual baseline (frames at 1 fps, audio spectrograms; random clip split); CREMA-D checkpoint on Zenodo 6778788. |

The 2025–2026 active-acquisition papers (L6, L7, L10) were not code-checked in this pass.  Check them before freezing a GAIN protocol.

**Environment gaps** (both `CS_QMI/env` and `vl_baselines`):
- missing: SimpleITK, itk, scikit-image, OpenCV, librosa, torchaudio, PyAV, soundfile, h5py;
- `ffmpeg` is not on PATH;
- present: torch / torchvision 0.24, scipy 1.18.

## 4. Prior internal evidence (already in the repository; relevant to REG)
**P60 B-T2** (2026-09-27 / 28) tested the closed-form VCS J* against histogram MI / NMI as a rigid-registration energy.  Setup: 60 real pairs,
20 000 pixel pairs, identical masks and samples; truth = the dataset registration.

| data | measure | local maxima, translation grid | success R = 5 / 10 / 20 / 30 |
|---|---|---|---|
| D3 EPFL RGB–NIR | VCS | 16.4 | 0.78 / 0.68 / 0.50 / 0.32 |
| D3 EPFL RGB–NIR | MI | 8.1 | 0.95 / 0.89 / 0.71 / 0.50 |
| D3 EPFL RGB–NIR | NMI | 8.3 | 0.95 / 0.89 / 0.70 / 0.52 |
| D3, patch coarse-to-fine | VCS | 11.6 | R = 30: 0.48 |
| D3, patch coarse-to-fine | MI | 8.1 | R = 30: 0.65 |
| IXI PD / T2 (local, CC BY-SA 3.0, not an r6 candidate) | VCS | — | R = 30: 0.83 |
| IXI PD / T2 | MI / NMI | — | R = 30: 0.99 / 1.00 |

So on raw intensities the closed-form J* is a worse registration energy than MI / NMI on both real datasets, and IXI is too easy to
separate methods.  Any r6 REG claim must come from a **learned** local relation critic, and it must beat MI / NMI and CoMIR on the same
geometric metric.

## 5. Recommendation (data feasibility only; the route choice is R6-04's)
**REG — one feasible natural-data pilot: D2 BF / SHG.**
- **Why D2:**
  - open CC BY 4.0, no agreement;
  - official train / val / test splits with a held-out test;
  - a defined geometric success criterion;
  - a strong published learned baseline (CoMIR 98 / 134 at < 5 %) and a weak classical one (MI 30 / 134), so there is room between them;
  - small: 577 MB crops for development (train + val1 + val2 only), or 1.8 GB full size.
- **D3 (local) as the correctness and sanity set:** owner-accepted, but its ground truth is the authors' SIFT registration, and MI / NMI already
  do well at small radii (P60).
- **D1 optional:** exact co-registered bands with synthetic transforms, but only 20 tiles.

**GAIN — no dataset is usable without owner action.**
- **AV-MNIST:** the fastest controlled-scale pipeline (1.5 GB, MultiBench loaders).  Needs owner acceptance of the CC BY-NC (ESC-50)
  component, and its audio-clip reuse limits any natural-data claim.
- **CREMA-D:** the credible natural task (actor-disjoint split, per-modality human ratings).  Needs owner acceptance of the ODbL terms and
  registration form, an LFS clone of about 7.55 GB, and ffmpeg / PyAV installed.

**If exactly one natural-data pilot must start without further owner action, only REG on D2 is feasible today.**

## Sources
- [Zurich Summer (Zenodo 5914759)](https://zenodo.org/records/5914759)
- [BF/SHG full-size (Zenodo 4550300)](https://zenodo.org/records/4550300) · [BF/SHG crops (Zenodo 3874362)](https://zenodo.org/record/3874362)
- [EPFL RGB–NIR (old page)](https://ivrlwww.epfl.ch/supplementary_material/cvpr11/index.html) · [EPFL RGB–NIR (new page)](https://www.epfl.ch/labs/ivrl/research/downloads/rgb-nir-scene-dataset/)
- [MultiBench data download](https://multibench.readthedocs.io/en/latest/start/datadownload.html) · [MultiBench repo](https://github.com/pliang279/MultiBench) · [AV-MNIST loader](https://github.com/pliang279/MultiBench/blob/main/datasets/avmnist/get_data.py)
- [CentralNet / AV-MNIST (arXiv 1808.07275)](https://arxiv.org/pdf/1808.07275) · [FSDD](https://github.com/Jakobovski/free-spoken-digit-dataset) · [ESC-50](https://github.com/karolpiczak/ESC-50) · [MNIST licence (Keras docs)](https://keras.io/api/datasets/mnist/)
- [CREMA-D page](https://cheyneycomputerscience.github.io/CREMA-D/) · [CREMA-D repo](https://github.com/CheyneyComputerScience/CREMA-D)
- [CoMIR repo](https://github.com/MIDA-group/CoMIR) · [CoMIR paper (ar5iv)](https://ar5iv.labs.arxiv.org/html/2006.06325) · [AdaMML](https://github.com/IBM/AdaMML) · [EDDI](https://github.com/microsoft/EDDI) · [OGM-GE](https://github.com/GeWu-Lab/OGM-GE_CVPR2022)
