"""Task-level pre-check D-fMRI — step 1: per-subject BOLD features at three motion-cleaning strengths (P65).

    python scripts/task_d_fmri_prep.py --data <aomic root>/func --ids <ids.txt> --out <dir> [--subjects sub-0002,sub-0014] [--smoke]

Per subject (AOMIC-PIOP1 resting-state run in T1w space, fMRIPrep 1.4.1 derivatives): drop the first 5 volumes; block means over b x b x b voxel
cubes (default 2, ~6 mm) with mask fraction >= 0.5; remove mean + cosine drift regressors by OLS (from the features and from the motion
regressors); motion knob alpha in {0, 0.5, 1}: residual_alpha = Y - alpha * X beta_hat with X the 24-parameter model; z-score; PCA to d = 32
fitted on the run (unsupervised) -> z (T x 32) per alpha; global signal g per alpha (mean over features before z-scoring); nuisance N =
framewise_displacement (first NaN -> 0), standardised.  Output <out>/<sub>.pt + <out>/manifest.json (QC per subject).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

MOTION6 = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
MOTION24 = [f"{m}{sfx}" for m in MOTION6 for sfx in ("", "_derivative1", "_power2", "_derivative1_power2")]


def read_tsv(path):
    import csv
    with open(path) as f:
        rd = csv.DictReader(f, delimiter="\t"); rows = list(rd)
    cols = rd.fieldnames

    def col(name):
        v = np.array([r[name] for r in rows], dtype=object)
        out = np.full(len(v), np.nan)
        for i, x in enumerate(v):
            try:
                out[i] = float(x)
            except (TypeError, ValueError):
                out[i] = np.nan
        return out
    return cols, col


def block_means(vol4d, mask, b, min_frac=0.5):
    """vol4d [X,Y,Z,T] float32, mask [X,Y,Z] bool -> [T, F] block means over in-mask voxels of b^3 cubes with mask fraction >= min_frac."""
    X, Y, Z, T = vol4d.shape; px, py, pz = (-X) % b, (-Y) % b, (-Z) % b
    v = np.pad(vol4d, ((0, px), (0, py), (0, pz), (0, 0))); m = np.pad(mask.astype(np.float32), ((0, px), (0, py), (0, pz)))
    Xb, Yb, Zb = v.shape[0] // b, v.shape[1] // b, v.shape[2] // b
    vb = v.reshape(Xb, b, Yb, b, Zb, b, T); mb = m.reshape(Xb, b, Yb, b, Zb, b)
    num = (vb * mb[..., None]).sum(axis=(1, 3, 5)); cnt = mb.sum(axis=(1, 3, 5))
    keep = cnt / float(b ** 3) >= min_frac
    feats = (num[keep] / cnt[keep][:, None]).T   # [T, F]
    return feats.astype(np.float32), int(keep.sum()), int(Xb * Yb * Zb)


def residualise(Y, C):
    """Remove the column space of C from Y (OLS)."""
    beta, *_ = np.linalg.lstsq(C, Y, rcond=None)
    return Y - C @ beta


def process(sub, data_root, out_dir, *, block, pca_d, drop, alphas):
    import nibabel as nib
    d = data_root / sub; pre = f"{sub}_task-restingstate_acq-mb3"
    img = nib.load(d / f"{pre}_space-T1w_desc-preproc_bold.nii.gz"); mask = nib.load(d / f"{pre}_space-T1w_desc-brain_mask.nii.gz").get_fdata() > 0.5
    tr = float(json.load(open(d / f"{pre}_bold.json"))["RepetitionTime"])
    vol = np.asarray(img.dataobj, dtype=np.float32)[..., drop:]              # [X,Y,Z,T]
    cols, col = read_tsv(d / f"{pre}_desc-confounds_regressors.tsv")
    T = vol.shape[-1]
    feats, n_blocks, n_total = block_means(vol, mask, block); del vol
    assert feats.shape[0] == T
    cosines = [c for c in cols if c.startswith("cosine")]
    C = np.column_stack([np.ones(T)] + [col(c)[drop:] for c in cosines])
    Y = residualise(feats.astype(np.float64), C)
    X = np.column_stack([np.nan_to_num(col(c)[drop:], nan=0.0) for c in MOTION24]); X = residualise(X, C)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None); fit = X @ beta
    fd = np.nan_to_num(col("framewise_displacement")[drop:], nan=0.0); N = (fd - fd.mean()) / (fd.std() + 1e-12)
    motion6 = np.column_stack([col(c)[drop:] for c in MOTION6])
    out = {"subject": sub, "tr": tr, "n_volumes": int(T), "dropped": drop, "block": block, "n_features": n_blocks, "n_blocks_total": n_total,
           "fd_mean": float(fd.mean()), "fd_max": float(fd.max()), "n": torch.tensor(N, dtype=torch.float32), "fd_raw": torch.tensor(fd, dtype=torch.float32),
           "motion6": torch.tensor(motion6, dtype=torch.float32), "z": {}, "g": {}, "pca_explained": {}, "alphas": list(alphas)}
    for a in alphas:
        R = Y - a * fit
        g = R.mean(axis=1)
        Zs = (R - R.mean(axis=0)) / (R.std(axis=0) + 1e-8)
        # PCA on the run itself: SVD of the T x F matrix
        U, S, Vt = np.linalg.svd(Zs, full_matrices=False)
        k = min(pca_d, S.size); scores = (U[:, :k] * S[:k]); var = (S ** 2); expl = float(var[:k].sum() / var.sum())
        out["z"][str(a)] = torch.tensor(scores / (scores.std(axis=0, keepdims=True) + 1e-8), dtype=torch.float32)
        out["g"][str(a)] = torch.tensor((g - g.mean()) / (g.std() + 1e-12), dtype=torch.float32)
        out["pca_explained"][str(a)] = expl
    torch.save(out, out_dir / f"{sub}.pt")
    return {k: out[k] for k in ("tr", "n_volumes", "n_features", "n_blocks_total", "fd_mean", "fd_max", "pca_explained")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="AOMIC func root (…/aomic_piop1/func)"); ap.add_argument("--ids", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--subjects", default=None, help="comma list (default: all ids)"); ap.add_argument("--block", type=int, default=2); ap.add_argument("--pca", type=int, default=32)
    ap.add_argument("--drop", type=int, default=5); ap.add_argument("--alphas", default="0,0.5,1"); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    ids = [l.strip() for l in open(a.ids) if l.strip()]
    subs = a.subjects.split(",") if a.subjects else ids
    if a.smoke:
        subs = subs[:2]
    alphas = [float(x) for x in a.alphas.split(",")]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    man_path = out / "manifest.json"
    man = json.load(open(man_path)) if man_path.exists() else {"subjects": {}, "settings": vars(a), "ids_sha256": hashlib.sha256(",".join(ids).encode()).hexdigest()}
    t0 = time.time()
    for s in subs:
        if s in man["subjects"] and (out / f"{s}.pt").exists():
            print(f"{s}: exists, skipping"); continue
        try:
            qc = process(s, Path(a.data), out, block=a.block, pca_d=a.pca, drop=a.drop, alphas=alphas)
        except FileNotFoundError as e:
            print(f"{s}: MISSING FILE {e}", flush=True); man["subjects"][s] = {"error": str(e)}; continue
        man["subjects"][s] = qc; man["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        json.dump(man, open(man_path, "w"), indent=1)
        print(f"{s}: T={qc['n_volumes']} TR={qc['tr']} features={qc['n_features']}/{qc['n_blocks_total']} FDmean={qc['fd_mean']:.3f} PCAexpl={ {k: round(v, 3) for k, v in qc['pca_explained'].items()} } ({time.time() - t0:.0f}s)", flush=True)
    ok = [s for s, q in man["subjects"].items() if "error" not in q]
    print(f"done: {len(ok)} subjects prepared -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
