# Scout — an fMRI variant of B-T2 (functional-to-structural registration): data, licences, access, recipe (2026-09-27, metadata only)

Question from the owner: can the real-data registration test use fMRI instead of (or next to) IXI structural MRI?  The sound form is
*BOLD EPI → T1w* registration of the same subject, mutual information's classic use case, with the P52 protocol (known synthetic rigid offsets
applied to a pair that is already co-registered).  That requires published fMRIPrep-style derivatives in T1w space.  Nothing was downloaded;
every number below comes from HTTP HEAD / listing requests and 4 KB ranged reads of NIfTI headers, all from the login node.

## 1. Sources checked (from this cluster)
| source | licence (verbatim) | access from here | note |
|---|---|---|---|
| **AOMIC-PIOP1, OpenNeuro ds002785** (raw + `derivatives/fmriprep/`, fMRIPrep 1.4.1) | `"License": "CC0"` in `ds002785/dataset_description.json` (DOI 10.18112/openneuro.ds002785.v2.0.0) and `"License": "CC0"` in `derivatives/fmriprep/dataset_description.json` | `https://s3.amazonaws.com/openneuro.org/ds002785/...` → HTTP 200 (files and bucket listings) | 216 subjects with fMRIPrep output; resting-state `space-T1w_boldref` present for every spot-checked subject (0001, 0002, 0050, 0100, 0150, 0216) |
| AOMIC-ID1000, OpenNeuro ds003097 | `"License": "CC0"` (DOI 10.18112/openneuro.ds003097.v1.2.1) | HTTP 200 | 498 subject folders under `derivatives/fmriprep/`; task `moviewatching` only; T1w preproc 31 MB per subject |
| UCLA CNP ds000030 | `"License": "CC0"` (raw dataset) | raw: HTTP 200; its fMRIPrep derivatives are **not** in the OpenNeuroDerivatives bucket (`fmriprep/ds000030-fmriprep/` → 404) | not usable without running fMRIPrep ourselves |
| ds000228 (children / adults, film) | `"License": "PDDL"` | HTTP 200 | PDDL, not CC0; no derivatives checked further |
| OpenNeuroDerivatives collection (bucket `openneuro-derivatives`, region us-west-2; GitHub org `OpenNeuroDerivatives`) | repos carry no licence field on GitHub; the README says only that the methods boilerplate is CC0; the source datasets are CC0 by OpenNeuro policy | listing `https://openneuro-derivatives.s3.us-west-2.amazonaws.com/?prefix=fmriprep/` → 200 (ds000001, ds000002, ds000005, …) | licence of the *derivatives themselves* not stated in a machine-readable place I could find → not "clearly permissive" by the brief's rule; AOMIC avoids the question because its derivatives ship inside the CC0 dataset |

## 2. Recommended dataset: AOMIC-PIOP1 (ds002785)
Per subject (verified by HEAD, sub-0001):
- `derivatives/fmriprep/sub-0001/anat/sub-0001_desc-preproc_T1w.nii.gz` — 13 965 781 bytes; header dim 240 × 240 × 220, 1.0 mm isotropic, float32.
- `derivatives/fmriprep/sub-0001/func/sub-0001_task-restingstate_acq-mb3_space-T1w_boldref.nii.gz` — 338 290 bytes; dim 46 × 57 × 43, 3.0 × 3.0 × 3.3 mm, float32 (the EPI reference resampled *into the T1w frame* but kept at EPI resolution — fMRIPrep 1.4 behaviour).
- `derivatives/fmriprep/sub-0001/anat/sub-0001_desc-brain_mask.nii.gz` — 163 259 bytes (T1w brain mask, for the brain box).
- Other tasks with `space-T1w_boldref` per subject: anticipation, emomatching, faces, gstroop, workingmemory (0.33–0.41 MB each) — a second EPI per subject is available for free if wanted.
Identity = subject id; `participants.tsv` (age, sex, …) is public (HTTP 200).
**Download for 60 subjects** (seed 20260927 over the 216): 60 × (14.0 + 0.34 + 0.16) MB ≈ **0.9 GB**, ~180 HTTPS GETs from the public bucket
(no credentials); a CPU sbatch job with `xargs -P 8` as for RGB–NIR, PROVENANCE.json with the two licence strings, per-file sha256 and the
subject-id list hash.  ID1000 would be ≈ 1.9 GB for 60 subjects (31 MB T1w) and offers 498 subjects if more are ever needed.

## 3. Slice-pair recipe (adaptation of `scripts/task_b_ixi.py`)
Both files share the T1w frame (same affine up to voxel size), so no registration step is needed — that is the point.
1. Load T1w (1 mm) and its brain mask; brain box from the mask; pick the axial slice at the box's mid-height (same rule as IXI).
2. Resample the boldref onto the T1w voxel grid for that slice (nibabel `resample_from_to` or `scipy.ndimage.affine_transform` with the two
   affines; linear interpolation) — this is a fixed, known transform (both images already carry world coordinates in the same space), not an
   estimated one, so the ground-truth alignment is preserved; the EPI's 3 mm voxels simply appear as a smooth 1 mm image (≈ 3× upsampled).
3. Crop both slices to the T1w brain box, square-pad, resize to 256 px, per-slice min–max (after clipping the T1w at its 99.5th percentile);
   A = T1w, B = boldref.  Then the unchanged P52 protocol: surfaces ± 24 px / ± 30°, Nelder–Mead ≤ 150 evaluations from R ∈ {5, 10, 20, 30},
   10 inits per radius; J* (Fourier 49) vs MI vs NMI.  Note that 24 px at 256 px over a ~200 mm box is ≈ 19 mm, i.e. ≈ 6 EPI voxels — a
   realistic misregistration range for EPI→T1w.
4. Caveats to state in the prereg: the EPI is heavily smoother than the T1w (3 mm voxels, susceptibility dropout in orbitofrontal / temporal
   regions), which is exactly the regime where histogram MI is strong; fMRIPrep's boldref is already bias-corrected and skull-stripped by its
   own pipeline (`desc-brain_mask` in func/ exists), while the T1w preproc is not skull-stripped (use the anat mask if a skull-stripped pair is
   wanted — decide before running, apply identically to all measures).
Code delta relative to the IXI script: one loader function (two files, two affines, one resampling call) and the file-pattern for the download;
the protocol module is reused as is.  Estimated GPU time as for IXI / RGB–NIR: ≈ 1.7 GPU-h for 60 pairs after a 3-subject CPU smoke.

## 4. Where fMRI fits better than registration (for the owner's decision; not designed here)
Registration is MI's home turf and, on the RGB–NIR and constructed pairs, J* has only matched MI at best (P52, P59, B-T2), so an EPI→T1w
test mostly asks the same question a third time.  The fMRI use that matches a *property VCS actually showed* (row D: an exact conditional
dependence test with a monotone magnitude, and conditional power where HSIC loses it) is leakage / dependence profiling on the BOLD signal:
S(BOLD-derived features ; nuisance regressors such as framewise displacement, the six motion parameters, CSF / white-matter signals | task or
condition), i.e. "how much motion is still in the cleaned signal", per parcel or per component, with the within-class shuffle giving an exact
test conditional on task blocks.  fMRIPrep publishes exactly these regressors (`*_desc-confounds_regressors.tsv`) next to the preprocessed
BOLD for every AOMIC subject, so the same CC0 download serves both uses; the preprocessed BOLD files are large (45–160 MB per run), so that
unit would download 60 runs ≈ 6–9 GB.  It would be a new D-family task pre-check with its own prereg, competitors (partial correlation /
HSIC per parcel) and reading rules.

## 5. Could not be verified
- A machine-readable licence for the OpenNeuroDerivatives collection (only the boilerplate is declared CC0); AOMIC's own derivatives are CC0.
- Exact count of the 216 PIOP1 subjects that have the resting-state `space-T1w_boldref` (6 of 6 spot checks present; the download job should
  list per subject and record the count).
- Whether `nibabel.processing.resample_from_to` is available in the env (nibabel 5.4.2 is installed; scipy 1.18.1 covers the fallback).
