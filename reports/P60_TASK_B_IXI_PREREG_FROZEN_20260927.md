# Pre-registration — task pre-check B-T2 (second source): closed-form J* as a rigid-registration energy on real multi-contrast MRI (IXI PD/T2) — FROZEN 2026-09-27T15:19:39Z (owner: run both sources)

Status: DRAFT (implementation fork); the main session freezes it.  Replaces the RGB–NIR source of `P60_TASK_B_RGBNIR_PREREG_DRAFT_20260927.md`
(kept as the record of why that source was dropped: no terms stated on its page).  Coordinator's reasoning for IXI: the PD and T2 volumes of a
subject come from the same dual-echo acquisition, so they are inherently aligned real multimodal pairs, and multi-contrast MRI is MI's home
ground — the fairest real-data registration test for J*.

## Provenance (recorded before download, brief §3)
- Dataset page: https://brain-development.org/ixi-dataset/ (fetched 2026-09-27 from the login node).  Verbatim: "In this project we have
  collected nearly 600 MR images from normal, healthy subjects. The MR image acquisition protocol for each subject includes: T1, T2 and
  PD-weighted images, MRA images, Diffusion-weighted images (15 directions). The data has been collected at three different hospitals in London
  … The images in NIFTI format can be downloaded from here: T1 images, T2 images, PD images, MRA images, DTI images … **This data is made
  available under the Creative Commons CC BY-SA 3.0 license. If you use the IXI data please acknowledge the source of the IXI data, e.g. this
  website.**" (licence link on the page: https://creativecommons.org/licenses/by-sa/3.0/legalcode).
- Official archives (links on the page): http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T2.tar and …/IXI-PD.tar
  (also IXI-T1.tar, IXI-MRA.tar, IXI-DTI.tar, IXI.xls).
- **Access from this cluster, 2026-09-27: HTTP 403 Forbidden** for everything under `/brain-development/downloads/IXI/` (T2, PD tars and the
  small IXI.xls), from the login node over IPv4 (egress 137.194.132.127) and IPv6, with curl and wget, with and without a browser
  user-agent / referer, and from a CPU compute node (download job 1011145, log `slurm_logs/download_ixi_1011145.out`: "curl: (22) The requested
  URL returned error: 403").  The parent directory `/brain-development/downloads/` answers 200 with an empty listing and the host root 301 —
  the block is specific to the IXI directory (or to this network for it).
- Mirror found (metadata only, nothing downloaded): Hugging Face dataset `Santhosh1884/IXI-Datasets` (repo revision
  b6fc53887a902c8ae24ae1f262ed11a201b21ddb, created 2025-11-30, no licence field on the card, no README) holding verbatim-named tars with LFS
  metadata IXI-PD.tar 4 068 966 400 bytes sha256 c00124d2409b68f9800938ede547cff71cafb0d51e19bd5138d85a4c4e144130, IXI-T2.tar 3 853 445 120 bytes
  sha256 39af50f87be8f7d0be2f166a64f5f59b5e43577255eb5613553dae37c348c2e4, IXI-T1.tar 4 840 816 640 bytes; the resolve endpoint answers 206 to a
  ranged request.  CC BY-SA 3.0 permits redistribution with attribution / share-alike, so a mirror is licence-compatible, but it is a change of
  source that the owner has to accept: the official checksums cannot be fetched (403), so integrity could only be recorded as the mirror's sha256
  (verified after download) — the fork's attempt to point the download job at the mirror was stopped by the session's permission classifier,
  so the decision and the download go back to the main session.  `slurm/download_ixi.sbatch` (official URLs, provenance JSON with url / licence /
  sha256 / bytes / utc, extraction, `pairs_index.json` of subjects with both contrasts) is ready; it takes `URL_BASE` as an override.
- Identity = IXI subject id (`IXI<id>-<site>-<n>-{PD,T2}.nii.gz`); 60 subjects with both contrasts chosen by seed 20260927; the id list and its
  sha256 go into the run's JSON and here once the data are present.  Environment: `nibabel` 5.4.2 installed with uv (numpy 2.5.2).

## Design (`scripts/task_b_ixi.py`, imports the P52 protocol from `precheck_b2_registration`)
Slice rule (fixed before seeing any data): T2 brain bounding box = voxels above 10 % of the volume's 99th-percentile intensity; the axial slice
in the middle of that box (last NIfTI axis) taken from both contrasts, cropped to the box's in-plane extent, zero-padded to a square, resized to
256 px, per-slice min–max normalised to [0, 1]; modality A = PD, modality B = T2; truth = identity (same acquisition; any residual is a property
of the data shared by all measures).  J* (49-d Fourier class, ridge 1e-3, no tanh), MI, NMI (32×32 histogram) on identical pixel samples (20 000)
and overlap masks; translation surfaces ±24 px step 2 (θ = 0), rotation profiles ±30° step 1 (t = 0); Nelder–Mead ≤ 150 evaluations from 10
random offsets per radius R ∈ {5, 10, 20, 30}; success = < 1 px and < 1°; divergence = |t| > 64 px.  Slice index and box per subject kept in the JSON.
**Pre-committed reading** (P51 rule + the wave-2 re-opening clause): **holds** if J*'s success at R = 20 and R = 30 is ≥ NMI + 0.10 with ≤ NMI's
mean local maxima, zero divergence and peak-at-truth in ≥ 95 % of subjects; **re-opened / holds conditionally** if J* matches NMI within 0.05 on
success at R = 20 and 30 and has ≤ NMI's local maxima (parity on MI's home ground); **does not hold** if J* has more local maxima than NMI or
success lower by ≥ 0.05 at R = 20 or 30.  Not claimed: 3-D or non-rigid registration, other contrast pairs (T1 is acquired separately and is not
inherently aligned), sites pooled (site recorded per subject id).
**Probe (code only, disclosed).** `--synthetic-smoke`: 3 constructed subjects (grey COCO slice under a disc mask as "T2", its tent-map modality as
"PD", 5 axial slices, saved as NIfTI in the download layout) exercise the loader, slice rule, subject selection, id hash and the full protocol on
CPU (job 1011150).  A real 3-subject smoke (`SMOKE=1`) runs once the data are present.
**Compute.** Download ≈ 7.9 GB (CPU partition, `--time=06:00:00`; disk: 28 TB free on /home/infres); one GPU job ≈ 1.7 h (A100, as P52).

**Status (main session, 2026-09-27).** DRAFT, pending the owner's decision on the data source: the official IXI server refuses this cluster
(HTTP 403), and fetching the archives from the third-party mirror was not permitted in this session.  Until the owner provides the data
(official download from another machine, or an explicit go for the mirror) B-T2 is reported as *pending (data)*; no compute is run.

**Owner decision (2026-09-27, verbatim): "我觉得B-T2两个都执行就好了，NIR数据和MRI数据都有价值".**  B-T2 therefore runs on both sources: the EPFL
RGB–NIR pairs (P60 RGB–NIR prereg, running) and the IXI PD/T2 slices (this prereg).  The IXI archives cannot be fetched by this session
(official server 403; the mirror fetch is not permitted to the assistant), so the owner runs the download job themselves; the licence of record
is the official page's CC BY-SA 3.0, the source URL actually used and the archive sha256 / bytes / utc are written by the job into
`data/ixi/PROVENANCE.json`, and the mirror's LFS hashes quoted above are the values the downloaded tars must match (a mismatch stops the unit).
Frozen now with the design of §"IXI design"; the 60-subject id-list sha256 is appended after the 3-subject smoke, before the full run is read.
Reading rules unchanged: holds (≥ NMI + 0.10 success at R = 20 / 30, ≤ NMI local maxima, no divergence, peak-at-truth ≥ 95 %); re-opened /
conditional (success within 0.05 of NMI and ≤ NMI maxima); does not hold otherwise.  The two sources are read separately; the family row is
re-opened only if the rule is met on at least one real source and not contradicted on the other.
