"""Second-application task-level pre-check B-T2 (wave 2, IXI): closed-form J* vs MI / NMI as a rigid-registration energy on REAL multi-contrast MRI.

    python scripts/task_b_ixi.py --data /home/infres/yinwang/CS_QMI/data/ixi --out <prefix> [--n-subjects 60] [--smoke]
    python scripts/task_b_ixi.py --synthetic-smoke <dir> --out <prefix>      (code probe without the dataset: 3 constructed NIfTI pairs)

Data: IXI (brain-development.org/ixi-dataset/, CC BY-SA 3.0): the PD and T2 volumes of a subject come from the same dual-echo acquisition, so
they are inherently aligned real multimodal pairs (truth = identity).  `--data` holds `T2/` and `PD/` NIfTI files (IXI<id>-<site>-<n>-T2.nii.gz /
-PD.nii.gz) and `pairs_index.json` written by `slurm/download_ixi.sbatch` (subjects with both contrasts).  Subjects: `--n-subjects` (60) chosen by
seed 20260927 among the subjects with both contrasts (identity = IXI subject id; list + sha256 recorded in the output).
Slice rule (fixed): the volume's brain bounding box = voxels above 10 % of the 99th-percentile intensity (computed on T2, applied to both);
the axial slice in the middle of that box (along the last axis of the NIfTI array) is taken from both contrasts, cropped to the box's in-plane
extent, zero-padded to a square, resized to 256 px, per-slice min–max normalised to [0, 1]; modality A = PD, modality B = T2.
Then exactly the P52 protocol (`precheck_b2_registration.run_pair`): J* (49-d Fourier class, ridge 1e-3, no tanh), MI, NMI on identical pixel
samples; translation surfaces ±24 px / step 2, rotation profiles ±30° / step 1; Nelder–Mead ≤ 150 evaluations from `--inits-per-radius` random
offsets per radius R ∈ {5, 10, 20, 30}; success = < 1 px and < 1°; divergence = |t| > 64 px.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from precheck_b2_registration import COCO, S, aggregate, load_grey, make_modality_b, progress_line, run_pair, write_outputs  # noqa: E402


def brain_box(vol, frac=0.10):
    thr = frac * np.percentile(vol, 99); m = vol > thr
    if not m.any():
        return tuple(slice(0, n) for n in vol.shape)
    return tuple(slice(int(np.where(m.any(axis=tuple(j for j in range(3) if j != i)))[0][0]), int(np.where(m.any(axis=tuple(j for j in range(3) if j != i)))[0][-1]) + 1) for i in range(3))


def mid_slices(pd_path, t2_path):
    """PD and T2 mid-axial slices (after the T2 brain-box crop), square-padded, 256 px, min-max normalised; also returns the slice index."""
    t2 = np.asarray(nib.load(str(t2_path)).dataobj, dtype=np.float32); pd = np.asarray(nib.load(str(pd_path)).dataobj, dtype=np.float32)
    if t2.shape != pd.shape:
        raise ValueError(f"shape mismatch {t2.shape} vs {pd.shape}")
    box = brain_box(t2); k = (box[2].start + box[2].stop) // 2
    out = []
    for vol in (pd, t2):
        sl = vol[box[0], box[1], k]; h, w = sl.shape; s = max(h, w); pad = np.zeros((s, s), np.float32); pad[(s - h) // 2: (s - h) // 2 + h, (s - w) // 2: (s - w) // 2 + w] = sl
        im = Image.fromarray(pad, mode="F").resize((S, S), Image.BILINEAR); arr = np.asarray(im, dtype=np.float32)
        lo, hi = float(arr.min()), float(arr.max()); out.append(torch.tensor((arr - lo) / max(hi - lo, 1e-6)).clamp(0, 1))
    return out[0], out[1], int(k), [int(b.start) for b in box], [int(b.stop) for b in box]


def build_synthetic(dir_: Path, n=3, seed=20260927):
    """Code probe without IXI: 3 fake subjects whose 'T2' volume stacks a grey COCO slice (with a soft brain-like mask) and whose 'PD' is the
    tent-map modality of it, saved as NIfTI in the download layout (T2/, PD/, pairs_index.json)."""
    idx = json.load(open("/home/infres/yinwang/CS_QMI/data/coco_index/val2017_index.json"))["images"]
    ids = sorted(int(i) for i in idx); rng = np.random.default_rng(seed); rng.shuffle(ids); gen = torch.Generator().manual_seed(seed)
    (dir_ / "T2").mkdir(parents=True, exist_ok=True); (dir_ / "PD").mkdir(parents=True, exist_ok=True); pairs = {}
    yy, xx = np.meshgrid(np.linspace(-1, 1, S), np.linspace(-1, 1, S), indexing="ij"); mask = ((xx ** 2 + yy ** 2) < 0.8).astype(np.float32)
    for k, iid in enumerate(ids[:n]):
        a = load_grey(COCO / "val2017" / idx[str(iid)]["file"]).numpy() * mask + 0.05 * mask; b = make_modality_b(torch.tensor(a), gen).numpy() * mask
        vt2 = np.stack([a * (0.5 + 0.5 * z / 4) for z in range(5)], -1); vpd = np.stack([b * (0.5 + 0.5 * z / 4) for z in range(5)], -1)   # 5 'axial' slices
        sid = f"IXI{900 + k:03d}"; ft2, fpd = dir_ / "T2" / f"{sid}-SYN-0000-T2.nii.gz", dir_ / "PD" / f"{sid}-SYN-0000-PD.nii.gz"
        nib.save(nib.Nifti1Image(vt2, np.eye(4)), str(ft2)); nib.save(nib.Nifti1Image(vpd, np.eye(4)), str(fpd)); pairs[sid] = {"T2": str(ft2.relative_to(dir_)), "PD": str(fpd.relative_to(dir_))}
    json.dump({"both": pairs}, open(dir_ / "pairs_index.json", "w"), indent=1)
    return dir_


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/infres/yinwang/CS_QMI/data/ixi"); ap.add_argument("--out", required=True)
    ap.add_argument("--n-subjects", type=int, default=60); ap.add_argument("--n-pairs", type=int, default=20000); ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--inits-per-radius", type=int, default=10); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--synthetic-smoke", default=None)
    a = ap.parse_args()
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    if a.synthetic_smoke:
        a.data = str(build_synthetic(Path(a.synthetic_smoke))); a.smoke = True
    if a.smoke:
        a.n_subjects, a.inits_per_radius = 3, 2
    root = Path(a.data); both = json.load(open(root / "pairs_index.json"))["both"]; subjects = sorted(both)
    rng = np.random.default_rng(a.seed); chosen = sorted(rng.permutation(subjects)[: a.n_subjects].tolist()); list_sha = hashlib.sha256(",".join(chosen).encode()).hexdigest()
    print(f"{len(subjects)} subjects with PD and T2; using {len(chosen)} (sha256 of id list {list_sha[:16]})", flush=True)
    rng = np.random.default_rng(a.seed); per_image = []; slices = {}; t0 = time.time()
    for n_i, sid in enumerate(chosen):
        A_, B_, k, lo, hi = mid_slices(root / both[sid]["PD"], root / both[sid]["T2"]); slices[sid] = {"slice": k, "box_lo": lo, "box_hi": hi}
        rec = run_pair(A_, B_, sid, a.seed, a.n_pairs, a.inits_per_radius, rng, device); rec["slice"] = slices[sid]; per_image.append(rec)
        print(progress_line(n_i, len(chosen), sid, rec, t0), flush=True)
    agg = aggregate(per_image)
    settings = dict(vars(a)); settings.update({"id_list_sha256": list_sha, "n_subjects_available": len(subjects), "subjects_used": chosen, "modalities": "A = PD, B = T2 (same dual-echo acquisition; truth = identity)"})
    write_outputs(a.out, f"Task pre-check B-T2 — closed-form VCS J* vs histogram MI/NMI as a rigid-registration energy on {len(chosen)} real IXI PD/T2 mid-axial slice pairs",
                  f"{a.n_pairs} pixel pairs per evaluation, identical overlap mask and samples for all measures; translation grid ±24 px step 2 (θ = 0); rotation ±30° step 1 (t = 0); "
                  f"Nelder–Mead ≤ 150 evaluations from {a.inits_per_radius} random initial offsets per radius; success = < 1 px and < 1°; truth = identity (same acquisition).", settings, device, chosen, agg, per_image)
    return 0


if __name__ == "__main__":
    sys.exit(main())
