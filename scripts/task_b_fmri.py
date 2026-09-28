"""Second-application task-level pre-check B-T2 (wave 2, third real source): closed-form J* vs MI / NMI as a rigid-registration energy on REAL
functional-to-structural pairs — a subject's resting-state BOLD EPI reference (fMRIPrep `space-T1w_boldref`) vs the same subject's T1w
(fMRIPrep `desc-preproc_T1w`), AOMIC-PIOP1 (OpenNeuro ds002785, CC0).

    python scripts/task_b_fmri.py --data /home/infres/yinwang/CS_QMI/data/aomic_piop1 --out <prefix> [--n-subjects 60] [--smoke]

Data layout (written by slurm/download_aomic_piop1.sbatch): `files/<sub>_desc-preproc_T1w.nii.gz`, `files/<sub>_desc-brain_mask.nii.gz`,
`files/<sub>_task-restingstate_acq-mb3_space-T1w_boldref.nii.gz`, `pairs_index.json` ({"pairs": {sub: {T1w, mask, boldref}}}) and
`selection.json` (the 60 subjects chosen by seed 20260927 among the 210 with all three files; identity = subject id).
Slice rule (fixed in the P60 fMRI prereg): brain box = bounding box of the fMRIPrep T1w brain mask (> 0.5); axial slice at the box's mid-height
along the third axis; the boldref volume is resampled onto the 1 mm T1w grid with the two stored affines (nibabel resample_from_to, linear) —
a known transform, so the fMRIPrep alignment (truth = identity) is preserved; the brain mask is applied to BOTH slices; the T1w is clipped at the
99.5th percentile of its masked slice; both are cropped to the box's in-plane extent, zero-padded to a square, resized to 256 px, per-slice
min–max normalised; modality A = T1w, modality B = boldref.  Then exactly the P52 protocol (`precheck_b2_registration.run_pair`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from PIL import Image
from nibabel.processing import resample_from_to

sys.path.insert(0, str(Path(__file__).resolve().parent))
from precheck_b2_registration import S, aggregate, progress_line, run_pair, write_outputs  # noqa: E402, Variant


def mask_box(mask):
    m = mask > 0.5
    if not m.any():
        raise ValueError("empty brain mask")
    return tuple(slice(int(np.where(m.any(axis=tuple(j for j in range(3) if j != i)))[0][0]), int(np.where(m.any(axis=tuple(j for j in range(3) if j != i)))[0][-1]) + 1) for i in range(3))


def to_square_256(sl):
    h, w = sl.shape; s = max(h, w); pad = np.zeros((s, s), np.float32); pad[(s - h) // 2: (s - h) // 2 + h, (s - w) // 2: (s - w) // 2 + w] = sl
    arr = np.asarray(Image.fromarray(pad, mode="F").resize((S, S), Image.BILINEAR), dtype=np.float32)
    lo, hi = float(arr.min()), float(arr.max()); return torch.tensor((arr - lo) / max(hi - lo, 1e-6)).clamp(0, 1)


def slice_pair(t1_path, mask_path, bold_path, clip_pct=99.5):
    """(A = T1w, B = boldref) 256-px masked mid-axial slices + bookkeeping."""
    t1_img = nib.load(str(t1_path)); mk_img = nib.load(str(mask_path)); bd_img = nib.load(str(bold_path))
    t1 = np.asarray(t1_img.dataobj, dtype=np.float32); mk = np.asarray(mk_img.dataobj, dtype=np.float32)
    if t1.shape != mk.shape:
        raise ValueError(f"T1w / mask shape mismatch {t1.shape} vs {mk.shape}")
    bd_rs = resample_from_to(bd_img, (t1_img.shape, t1_img.affine), order=1)                       # known transform: both already in T1w space
    bd = np.asarray(bd_rs.dataobj, dtype=np.float32)
    box = mask_box(mk); k = (box[2].start + box[2].stop) // 2
    m2 = (mk[box[0], box[1], k] > 0.5).astype(np.float32)
    a = t1[box[0], box[1], k] * m2; b = np.nan_to_num(bd[box[0], box[1], k]) * m2
    if m2.sum() > 0:
        hi = float(np.percentile(a[m2 > 0], clip_pct)); a = np.minimum(a, hi)
        b = np.maximum(b, 0.0)
    info = {"slice": int(k), "box_lo": [int(s.start) for s in box], "box_hi": [int(s.stop) for s in box], "mask_frac_slice": float(m2.mean()),
            "t1_voxel_mm": [float(v) for v in t1_img.header.get_zooms()[:3]], "bold_voxel_mm": [float(v) for v in bd_img.header.get_zooms()[:3]],
            "bold_shape": list(bd_img.shape[:3]), "resampling": "nibabel.processing.resample_from_to(boldref -> T1w grid, order=1)"}
    return to_square_256(a), to_square_256(b), info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/infres/yinwang/CS_QMI/data/aomic_piop1"); ap.add_argument("--out", required=True)
    ap.add_argument("--n-subjects", type=int, default=60); ap.add_argument("--n-pairs", type=int, default=20000); ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--inits-per-radius", type=int, default=10); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--save-slices", action="store_true")
    ap.add_argument("--patch-features", action="store_true", help="B-S2 variant: add the 8x8 block-mean channel to the J* features"); ap.add_argument("--coarse-to-fine", action="store_true", help="B-S2 variant: Nelder-Mead 64 -> 128 -> 256 px")
    a = ap.parse_args()
    VAR = Variant(patch=a.patch_features, ctf=a.coarse_to_fine)
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    if a.smoke:
        a.n_subjects, a.inits_per_radius = 3, 2
    root = Path(a.data); pairs = json.load(open(root / "pairs_index.json"))["pairs"]; subjects = sorted(pairs)
    chosen = subjects if a.n_subjects >= len(subjects) else sorted(np.random.default_rng(a.seed).permutation(subjects)[: a.n_subjects].tolist())
    list_sha = hashlib.sha256(",".join(chosen).encode()).hexdigest(); all_sha = hashlib.sha256(",".join(subjects).encode()).hexdigest()
    print(f"{len(subjects)} downloaded subjects (sha256 {all_sha[:16]}); using {len(chosen)} (sha256 of id list {list_sha[:16]})", flush=True)
    rng = np.random.default_rng(a.seed); per_image = []; slices = {}; t0 = time.time()
    if a.save_slices:
        (Path(a.out).parent / "slices_fmri").mkdir(parents=True, exist_ok=True)
    for n_i, sid in enumerate(chosen):
        A_, B_, info = slice_pair(root / pairs[sid]["T1w"], root / pairs[sid]["mask"], root / pairs[sid]["boldref"]); slices[sid] = info
        if a.save_slices:
            Image.fromarray((torch.cat([A_, B_], 1).numpy() * 255).astype(np.uint8)).save(Path(a.out).parent / "slices_fmri" / f"{sid}.png")
        rec = run_pair(A_, B_, sid, a.seed, a.n_pairs, a.inits_per_radius, rng, device, var=VAR); rec["slice"] = info; per_image.append(rec)
        print(progress_line(n_i, len(chosen), sid, rec, t0), flush=True)
    agg = aggregate(per_image)
    settings = dict(vars(a)); settings.update({"id_list_sha256": list_sha, "downloaded_ids_sha256": all_sha, "n_subjects_downloaded": len(subjects), "subjects_used": chosen,
                                               "modalities": "A = T1w (fMRIPrep desc-preproc, brain-masked, clipped at p99.5), B = resting-state EPI boldref (fMRIPrep space-T1w, resampled to the T1w grid, brain-masked); truth = fMRIPrep alignment (identity)"})
    write_outputs(a.out, f"Task pre-check B-T2 [{VAR.tag or 'P52'}] — closed-form VCS J* vs histogram MI/NMI as a rigid-registration energy on {len(chosen)} real AOMIC-PIOP1 EPI-boldref / T1w mid-axial slice pairs",
                  f"{a.n_pairs} pixel pairs per evaluation, identical overlap mask and samples for all measures; translation grid ±24 px step 2 (θ = 0); rotation ±30° step 1 (t = 0); "
                  f"Nelder–Mead ≤ 150 evaluations from {a.inits_per_radius} random initial offsets per radius; success = < 1 px and < 1°; truth = identity (fMRIPrep coregistration); brain mask applied to both.", settings, device, chosen, agg, per_image)
    return 0


if __name__ == "__main__":
    sys.exit(main())
