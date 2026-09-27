"""Second-application pre-check B, question 2: the closed-form VCS J* as a registration similarity — energy surfaces vs histogram MI / NMI.

    python scripts/precheck_b2_registration.py --index-dir <coco index> --out <prefix> [--n-images 60] [--smoke]

Constructed modality pairs with truth known by construction (brief P3): modality A = grey COCO image (256×256 centre crop, [0,1]);
modality B = tent-map remap  b = 1 − |2·g(a) − 1|  of a gamma-adjusted copy (g(a) = a^0.7), plus 3 % Gaussian noise and a smooth
multiplicative gain field — a non-monotone intensity relation (correlation ≈ 0, dependence high), aligned at ω* = identity.
Similarity of A and B under a rigid transform ω = (tx, ty, θ): sample N pixel locations x in the overlap of the two supports (same mask rule for
all measures), pairs (a(x), b(ω(x))) are the joint sample P, pairs (a(x), b(ω(x'))) with x' a random permutation are the product sample Q;
  VCS closed form:  φ(a, b) = [1, Fourier features cos/sin(2π k a) ⊗ cos/sin(2π l b), k,l = 1..3] (49-d), w* = ½(A_M + λI)⁻¹d, J*(ω) = w*ᵀd − w*ᵀA_M w*
  (ridge λ = 1e-3 · mean diag A_M; no tanh — measurement only, per the brief);
  MI / NMI: 32×32 joint histogram of the same pixel pairs, Shannon MI and Studholme NMI = (H(a)+H(b))/H(a,b).
No spatial coordinates enter φ.  Outputs: (1) 2-D translation energy surfaces (θ = 0) on a ±24 px grid, step 2 px, and 1-D rotation profiles
(±30°, step 1°) for every image; capture range (half-width of the basin containing the truth), number of local maxima; (2) success rate of a
derivative-free optimiser (Nelder–Mead on (tx, ty, θ), same budget for all measures) from random initial offsets |t| ≤ R px, |θ| ≤ R°,
R ∈ {5, 10, 20, 30}; success = final error < 1 px and < 1°; divergence = optimiser leaving |t| > 64 px.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.optimize import minimize

COCO = Path("/projects/EEG-foundation-model/yinghao/FMCA-AV/coco")
S = 256


def load_grey(path):
    im = Image.open(path).convert("L"); w, h = im.size; s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((S, S), Image.BILINEAR)
    return torch.tensor(np.asarray(im, dtype=np.float32) / 255.0)


def make_modality_b(a, gen):
    g = a.clamp(0, 1) ** 0.7; b = 1 - (2 * g - 1).abs()
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, S), torch.linspace(-1, 1, S), indexing="ij")
    gain = 1 + 0.15 * torch.sin(2 * math.pi * (0.7 * xx + 0.4 * yy) + float(torch.rand(1, generator=gen)) * 6.28)
    return (b * gain + 0.03 * torch.randn(S, S, generator=gen)).clamp(0, 1)


def warp_coords(x, y, tx, ty, th):
    """Map image-A coordinates (pixels, centred) to image-B coordinates under a rigid transform."""
    c, s = math.cos(math.radians(th)), math.sin(math.radians(th))
    return c * x - s * y + tx, s * x + c * y + ty


def sample_pairs(a, b, tx, ty, th, n, gen, device):
    xs = torch.rand(n, generator=gen) * (S - 1); ys = torch.rand(n, generator=gen) * (S - 1)
    xb, yb = warp_coords(xs - S / 2, ys - S / 2, tx, ty, th); xb, yb = xb + S / 2, yb + S / 2
    ok = (xb >= 0) & (xb <= S - 1) & (yb >= 0) & (yb <= S - 1)        # overlap mask, identical for all measures
    xs, ys, xb, yb = xs[ok], ys[ok], xb[ok], yb[ok]
    av = a[ys.round().long(), xs.round().long()]
    # bilinear sample of b
    x0, y0 = xb.floor().long().clamp(0, S - 2), yb.floor().long().clamp(0, S - 2); fx, fy = xb - x0, yb - y0
    bv = (b[y0, x0] * (1 - fx) * (1 - fy) + b[y0, x0 + 1] * fx * (1 - fy) + b[y0 + 1, x0] * (1 - fx) * fy + b[y0 + 1, x0 + 1] * fx * fy)
    return av.to(device), bv.to(device)


def fourier_feats(a, b, kmax=3):
    fa = [torch.ones_like(a)] + [f(2 * math.pi * k * a) for k in range(1, kmax + 1) for f in (torch.cos, torch.sin)]
    fb = [torch.ones_like(b)] + [f(2 * math.pi * l * b) for l in range(1, kmax + 1) for f in (torch.cos, torch.sin)]
    FA, FB = torch.stack(fa, 1), torch.stack(fb, 1)               # [n, 7] each
    return (FA[:, :, None] * FB[:, None, :]).reshape(len(a), -1)   # [n, 49]


def vcs_closed(av, bv, gen, lam=1e-3):
    if len(av) < 200:
        return float("nan")
    perm = torch.randperm(len(av), generator=gen).to(av.device)
    pp, pq = fourier_feats(av, bv).double(), fourier_feats(av, bv[perm]).double()
    d = pp.mean(0) - pq.mean(0); A = 0.5 * (pp.T @ pp / len(pp) + pq.T @ pq / len(pq)); A = A + lam * float(torch.diag(A).mean()) * torch.eye(len(A), dtype=torch.float64, device=A.device)
    w = 0.5 * torch.linalg.solve(A, d)
    return float(w @ d - w @ A @ w)


def hist_mi(av, bv, bins=32):
    if len(av) < 200:
        return float("nan"), float("nan")
    ia, ib = (av.clamp(0, 1) * (bins - 1)).round().long(), (bv.clamp(0, 1) * (bins - 1)).round().long()
    H = torch.zeros(bins, bins, device=av.device); H.index_put_((ia, ib), torch.ones_like(av), accumulate=True); P = H / H.sum()
    pa, pb = P.sum(1), P.sum(0); ent = lambda p: float(-(p[p > 0] * p[p > 0].log()).sum())
    ha, hb, hab = ent(pa), ent(pb), ent(P.reshape(-1))
    return ha + hb - hab, (ha + hb) / max(hab, 1e-9)


def local_maxima_count(Z):
    Z = np.nan_to_num(Z, nan=-np.inf); c = 0
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            nb = Z[max(0, i - 1): i + 2, max(0, j - 1): j + 2]
            if Z[i, j] == nb.max() and (nb < Z[i, j]).sum() >= nb.size - 1:
                c += 1
    return c


def basin_halfwidth(prof, centre_idx, step):
    """Half-width (in grid units × step) of the monotone basin around the truth in a 1-D profile."""
    r = 0
    while centre_idx + r + 1 < len(prof) and prof[centre_idx + r + 1] <= prof[centre_idx + r]:
        r += 1
    l = 0
    while centre_idx - l - 1 >= 0 and prof[centre_idx - l - 1] <= prof[centre_idx - l]:
        l += 1
    return min(r, l) * step


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", default="/home/infres/yinwang/CS_QMI/data/coco_index"); ap.add_argument("--out", required=True)
    ap.add_argument("--n-images", type=int, default=60); ap.add_argument("--n-pairs", type=int, default=20000); ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--inits-per-radius", type=int, default=10); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    if a.smoke:
        a.n_images, a.inits_per_radius = 3, 2
    idx = json.load(open(Path(a.index_dir) / "val2017_index.json"))["images"]   # val2017 images: not used by A/C (which use train2017)
    ids = sorted(int(i) for i in idx); rng = np.random.default_rng(a.seed); rng.shuffle(ids); ids = ids[: a.n_images]
    gen = torch.Generator().manual_seed(a.seed)
    tgrid = np.arange(-24, 25, 2); rgrid = np.arange(-30, 31, 1); radii = [5, 10, 20, 30]
    measures = ("vcs", "mi", "nmi"); per_image = []; t0 = time.time()
    for n_i, iid in enumerate(ids):
        A_ = load_grey(COCO / "val2017" / idx[str(iid)]["file"]); B_ = make_modality_b(A_, gen); A_, B_ = A_.to(device), B_.to(device)
        def evaluate(tx, ty, th, g=None):
            g = g or torch.Generator().manual_seed(a.seed + 1)   # same pixel sample for all measures at a given ω
            av, bv = sample_pairs(A_.cpu(), B_.cpu(), tx, ty, th, a.n_pairs, g, device)
            v = vcs_closed(av, bv, torch.Generator().manual_seed(3)); mi, nmi = hist_mi(av, bv)
            return {"vcs": v, "mi": mi, "nmi": nmi}
        # translation surfaces (theta = 0) and rotation profiles (t = 0)
        surf = {m: np.full((len(tgrid), len(tgrid)), np.nan) for m in measures}
        for i, tx in enumerate(tgrid):
            for j, ty in enumerate(tgrid):
                e = evaluate(float(tx), float(ty), 0.0)
                for m in measures:
                    surf[m][i, j] = e[m]
        prof = {m: np.array([evaluate(0.0, 0.0, float(th))[m] for th in rgrid]) for m in measures}
        c = len(tgrid) // 2; rc = len(rgrid) // 2
        rec = {"image_id": iid, "surfaces": {m: surf[m].tolist() for m in measures}, "rot_profiles": {m: prof[m].tolist() for m in measures}, "metrics": {}}
        for m in measures:
            rec["metrics"][m] = {"peak_at_truth_translation": bool(np.nanargmax(surf[m]) == c * len(tgrid) + c), "n_local_maxima_translation": local_maxima_count(surf[m]),
                                 "basin_halfwidth_px_x": basin_halfwidth(surf[m][:, c], c, 2), "basin_halfwidth_px_y": basin_halfwidth(surf[m][c, :], c, 2),
                                 "basin_halfwidth_deg": basin_halfwidth(prof[m], rc, 1), "peak_at_truth_rotation": bool(np.nanargmax(prof[m]) == rc),
                                 "value_at_truth": float(surf[m][c, c]), "ratio_truth_to_max_offset": float(surf[m][c, c] / np.nanmax([surf[m][0, 0], surf[m][-1, -1], surf[m][0, -1], surf[m][-1, 0]]))}
        # optimisation from random initial offsets (Nelder-Mead, same budget), success = < 1 px & < 1 deg
        opt = {m: {} for m in measures}
        for R in radii:
            for k in range(a.inits_per_radius):
                init = np.array([rng.uniform(-R, R), rng.uniform(-R, R), rng.uniform(-R, R)])
                for m in measures:
                    def f(p, m=m):
                        if abs(p[0]) > 64 or abs(p[1]) > 64:
                            return 1e3
                        v = evaluate(float(p[0]), float(p[1]), float(p[2]), torch.Generator().manual_seed(a.seed + 1))[m]
                        return 1e3 if not np.isfinite(v) else -v
                    res = minimize(f, init, method="Nelder-Mead", options={"maxfev": 150, "xatol": 0.2, "fatol": 1e-6, "initial_simplex": init + np.array([[0, 0, 0], [4, 0, 0], [0, 4, 0], [0, 0, 4]])})
                    err_t = float(np.hypot(res.x[0], res.x[1])); err_r = float(abs(res.x[2]))
                    opt[m].setdefault(str(R), []).append({"init": init.tolist(), "final": res.x.tolist(), "err_px": err_t, "err_deg": err_r, "success": err_t < 1 and err_r < 1, "diverged": abs(res.x[0]) > 64 or abs(res.x[1]) > 64, "nfev": int(res.nfev)})
        rec["optimisation"] = opt; per_image.append(rec)
        print(f"[{n_i + 1}/{len(ids)}] image {iid}: " + " ".join(f"{m}: truth-peak {rec['metrics'][m]['peak_at_truth_translation']} maxima {rec['metrics'][m]['n_local_maxima_translation']} basin {rec['metrics'][m]['basin_halfwidth_px_x']}px/{rec['metrics'][m]['basin_halfwidth_deg']}deg" for m in measures)
              + " | success@30: " + " ".join(f"{m}={np.mean([o['success'] for o in opt[m]['30']]):.2f}" for m in measures) + f" ({time.time() - t0:.0f}s)", flush=True)
    # aggregate
    agg = {}
    for m in measures:
        agg[m] = {"peak_at_truth_translation_rate": float(np.mean([r["metrics"][m]["peak_at_truth_translation"] for r in per_image])),
                  "peak_at_truth_rotation_rate": float(np.mean([r["metrics"][m]["peak_at_truth_rotation"] for r in per_image])),
                  "mean_local_maxima_translation": float(np.mean([r["metrics"][m]["n_local_maxima_translation"] for r in per_image])),
                  "median_basin_halfwidth_px": float(np.median([min(r["metrics"][m]["basin_halfwidth_px_x"], r["metrics"][m]["basin_halfwidth_px_y"]) for r in per_image])),
                  "median_basin_halfwidth_deg": float(np.median([r["metrics"][m]["basin_halfwidth_deg"] for r in per_image])),
                  "success_rate_by_radius": {str(R): float(np.mean([o["success"] for r in per_image for o in r["optimisation"][m][str(R)]])) for R in radii},
                  "divergence_rate_by_radius": {str(R): float(np.mean([o["diverged"] for r in per_image for o in r["optimisation"][m][str(R)]])) for R in radii}}
    out = {"settings": vars(a), "device": str(device), "image_ids": ids, "aggregate": agg, "per_image": per_image, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(out, open(a.out + ".json", "w"), default=float)
    L = [f"# Pre-check B2 — closed-form VCS J* vs histogram MI/NMI as a rigid-registration energy ({len(ids)} constructed modality pairs) — {out['utc']}", "",
         f"{a.n_pairs} pixel pairs per evaluation, identical overlap mask and samples for all measures; translation grid ±24 px step 2 (θ = 0); rotation ±30° step 1 (t = 0); "
         f"Nelder–Mead ≤ 150 evaluations from {a.inits_per_radius} random initial offsets per radius; success = < 1 px and < 1°.", "",
         "| measure | peak at truth (transl.) | peak at truth (rot.) | mean # local maxima (transl. grid) | median basin half-width px / deg | success R=5 / 10 / 20 / 30 | divergence R=30 |", "|---|---|---|---|---|---|---|"]
    for m in measures:
        g = agg[m]; L.append(f"| {m} | {g['peak_at_truth_translation_rate']:.2f} | {g['peak_at_truth_rotation_rate']:.2f} | {g['mean_local_maxima_translation']:.1f} | {g['median_basin_halfwidth_px']:.0f} / {g['median_basin_halfwidth_deg']:.0f} | "
                 + " / ".join(f"{g['success_rate_by_radius'][str(R)]:.2f}" for R in radii) + f" | {g['divergence_rate_by_radius']['30']:.2f} |")
    Path(a.out + ".md").write_text("\n".join(L) + "\n"); print("->", a.out + ".md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
