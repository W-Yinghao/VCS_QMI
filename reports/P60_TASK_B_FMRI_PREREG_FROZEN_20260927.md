# Pre-registration DRAFT — task pre-check B-T2, third real source: functional-to-structural (BOLD EPI → T1w) rigid registration on AOMIC-PIOP1 (OpenNeuro ds002785, CC0) — 2026-09-27 — FROZEN 2026-09-27T21:02:18Z before GPU compute (download 1011914 and CPU smoke 1011916 disclosed inside; owner: add fMRI as a third real source)

Status: DRAFT (implementation fork); the main session freezes it.  Owner's decision: "那个MRI，能不能换到fmri上？" → the coordinator chose AOMIC-PIOP1
as B-T2's *third* real source (next to EPFL RGB–NIR, done, and IXI PD/T2, pending data).  Scout record: `P60_TASK_B_FMRI_SCOUT_20260927.md`.
This provenance section was written BEFORE the download job was submitted (brief §3).

## Provenance
- Dataset: **AOMIC-PIOP1** (Amsterdam Open MRI Collection, Population Imaging of Psychology 1), OpenNeuro **ds002785**, DOI 10.18112/openneuro.ds002785.v2.0.0
  (Snoek, van der Miesen, Beemsterboer, van der Leij, Eigenhuis, Scholte — "The Amsterdam Open MRI Collection, a set of multimodal MRI datasets for
  individual difference analyses"; https://nilab-uva.github.io/AOMIC.github.io/).
- Licence, verbatim from `ds002785/dataset_description.json`: `"License": "CC0"`.  Licence of the shipped derivatives, verbatim from
  `ds002785/derivatives/fmriprep/dataset_description.json`: `"License": "CC0"` (pipeline `"Name": "fMRIPrep", "Version": "1.4.1"`; `"HowToAcknowledge": "Please cite our paper (https://doi.org/10.1038/s41592-018-0235-4), and include the generated citation boilerplate within the Methods section of the text."`).
  Both JSON files are downloaded into the data directory and hashed by the download job.
- Access: public bucket, no credentials — `https://s3.amazonaws.com/openneuro.org/ds002785/…` (HTTP 200 for files and listings from this cluster, checked 2026-09-27).
- File-name patterns per subject `sub-XXXX` (fMRIPrep 1.4.1 layout):
  `derivatives/fmriprep/sub-XXXX/anat/sub-XXXX_desc-preproc_T1w.nii.gz` (T1w in native space, 1 mm isotropic; sub-0001: 240 × 240 × 220 voxels, float32, 13 965 781 bytes),
  `derivatives/fmriprep/sub-XXXX/anat/sub-XXXX_desc-brain_mask.nii.gz` (T1w brain mask; sub-0001: 163 259 bytes),
  `derivatives/fmriprep/sub-XXXX/func/sub-XXXX_task-restingstate_acq-mb3_space-T1w_boldref.nii.gz` (EPI reference resampled into the T1w frame, kept at
  EPI resolution; sub-0001: 46 × 57 × 43 voxels, 3.0 × 3.0 × 3.3 mm, float32, 338 290 bytes).  Sizes and headers from HEAD requests and 4 KB ranged reads.
- Availability (HEAD on all 216 fMRIPrep subject folders, 2026-09-27, `data/aomic_piop1/availability_head_20260927.txt`): **210 subjects** have all
  three files; six lack the resting-state boldref (sub-0022, sub-0140, sub-0144, sub-0147, sub-0162, sub-0178) and are excluded before selection.
- Identity = subject id; `participants.tsv` is public but not used.  No official test set exists for this dataset; nothing here is a benchmark split.

## Subjects
60 subjects drawn by `numpy.random.default_rng(20260927).permutation(sorted(available))[:60]`, sorted; the id list and its sha256 are written by
the download job into `PROVENANCE.json` and copied here after the download (before the full run is read).  Only these 60 subjects' files are
downloaded (≈ 0.9 GB).

## Slice-pair recipe (decisions fixed now)
1. Load the T1w, its brain mask and the boldref (all three carry world coordinates in the same T1w space).  Brain box = bounding box of the mask
   (> 0.5) in T1w voxel coordinates; the axial slice k at the box's mid-height along the third (superior–inferior) axis.
2. Resample the boldref onto the 1 mm T1w voxel grid with the two stored affines (`nibabel.processing.resample_from_to`, linear interpolation) —
   a known transform between two images already in the same space, so the ground-truth alignment (identity) is preserved; the 3 mm EPI appears
   as a smooth ≈ 3×-upsampled image.
3. **Apply the fMRIPrep brain mask to BOTH images for ALL measures** (the EPI has no skull; an unmasked T1w would give MI a skull edge that J*
   cannot use).  T1w intensities clipped at the 99.5th percentile of the masked slice.  Crop both slices to the box's in-plane extent, zero-pad to a
   square, resize to 256 px, per-slice min–max to [0, 1].  **A = T1w, B = boldref.**
4. Then the P52 protocol unchanged (`precheck_b2_registration.run_pair`): J* (49-d Fourier class, ridge 1e-3, no tanh) vs histogram MI vs NMI on
   identical pixel samples (20 000 per evaluation); translation surfaces ± 24 px step 2 (θ = 0), rotation profiles ± 30° step 1 (t = 0);
   Nelder–Mead ≤ 150 evaluations from 10 random offsets per radius R ∈ {5, 10, 20, 30}; success = < 1 px and < 1°; divergence = |t| > 64 px.
   At 256 px over a ≈ 180–200 mm brain box, 24 px ≈ 18 mm ≈ 6 EPI voxels — a realistic EPI → T1w misregistration range.

## Pre-committed reading (copied verbatim from the RGB–NIR P60 prereg)
**Pre-committed reading** (as P51/P52, with the plan's re-opening clause): the registration family is re-opened on real data only if J* matches
NMI on both the mean number of local maxima (≤ NMI's) and the success rates at R = 20 and R = 30 (within 0.05) — and **holds** in the P51 sense
if J*'s success at R = 20 and 30 is ≥ NMI + 0.10 with ≤ NMI's local maxima, zero divergence and peak-at-truth in ≥ 95 % of pairs; **does not hold**
if J* has more local maxima than NMI or lower success by ≥ 0.05.
Expectation specific to this source (stated before running): EPI → T1w is the classic use of mutual information, and the EPI slice is heavily
smoothed with susceptibility dropout, so MI/NMI are expected to be strong; a "holds" would be informative, a "does not hold" confirms P52 / B-T2
on a third real modality pair.  The three real sources are read separately; the family row is re-opened only if the rule is met on at least one
real source and not contradicted on the others.

## Not claimed
EPI geometric distortion / non-rigid registration (the pairs are already distortion-corrected by fMRIPrep); other tasks of PIOP1 (a second EPI per
subject exists but is not used); other datasets (ID1000, UCLA CNP); the effect of fMRIPrep's own coregistration residual (truth = fMRIPrep's
alignment, not a gold standard); anything about fMRI signal analysis (the D-family use in the scout report §4 is a separate proposal).

## Code
`scripts/task_b_fmri.py` (loader above + `run_pair`, `aggregate`, `write_outputs` imported unchanged from `precheck_b2_registration.py`),
`slurm/download_aomic_piop1.sbatch` (CPU; 60 subjects × 3 files; retries; per-file sha256; `PROVENANCE.json`; `pairs_index.json`),
`slurm/precheck_b_wave2_fmri.sbatch` (GPU header as the IXI one; `SMOKE=1` = 3 subjects, 2 inits per radius).

## Probe / smoke (filled after the download)
(to be appended: download job id, id-list sha256, per-file counts and bytes, 3-subject CPU smoke numbers)

## Compute
Download ≈ 0.9 GB (CPU partition).  Smoke: 3 subjects on CPU.  Full run: one GPU job ≈ 1.7 h (as P52 / RGB–NIR).

**Download (CPU job 1011914, `slurm/download_aomic_piop1.sbatch`, finished 2026-09-27T20:55:13Z).** available 210 / 216 subjects; selected 60 by seed 20260927; **id-list sha256 a3dd39dda61cb18c24a47c26f706beaabe98effb593a8c2bb1c3bcc9c3e73845**; 180 files, 931,608,328 bytes, 0 failures; licences read from the downloaded JSONs: dataset `CC0`, derivatives `CC0` (dataset_description sha256 45b7ee374f116c67… / bea69b9766d22e1c…); pipeline {'Name': 'fMRIPrep', 'Version': '1.4.1', 'CodeURL': 'https://github.com/poldracklab/fmriprep/archive/1.4.1.tar.gz'}; per-file sha256 / bytes in `data/aomic_piop1/PROVENANCE.json`.  First / last selected: sub-0002 … sub-0216.

**Smoke probe (CPU job 1011916, `SMOKE=1 --save-slices`, 3 subjects sub-0137 / sub-0148 / sub-0160 = first 3 of the seed-20260927 permutation of the 60,
id-list sha256 da85d6f53e306a9b…, 2 inits per radius, 4 min 12 s total, exit 0; disclosed, not evidence).**  Loader: T1w 1 mm, boldref 3.0 × 3.0 × 3.3 mm
resampled onto the T1w grid, mask applied to both, slice PNGs checked by eye (skull-free T1w next to a smooth EPI with the same brain outline).
Numbers: peak-at-truth 1.00 / 1.00 / 1.00 (J* / MI / NMI), local maxima 1.0 for all three, basin 24 px / 22° (J*) vs 24 / 30 (MI, NMI),
success R = 5 / 10 / 20 / 30: J* 1.00 / 1.00 / 0.83 / 0.83 vs MI and NMI 1.00 throughout, no divergence — with 2 inits per radius on 3 subjects
this only shows that the pipeline runs and that the pair is a genuine cross-modal registration problem.  Full run: 60 subjects, 10 inits per radius,
one GPU job ≈ 1.7 h (P52 cost; CPU smoke ≈ 80 s per subject at 2 inits).
