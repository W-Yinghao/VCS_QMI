"""Second-application pre-check B, question 2: the closed-form VCS J* as a registration similarity — energy surfaces vs histogram MI / NMI.

    python scripts/precheck_b2_registration.py --index-dir <coco index> --out <prefix> [--n-images 60] [--smoke]
        [--n-fourier 49|121] [--patch-features] [--smooth 0,2,4] [--coarse-to-fine]        (wave-2 variants, B-S2; defaults = P52 protocol)

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

Wave-2 variants (B-S2; each changes ONE thing relative to the P52 protocol, the default path is unchanged):
  --n-fourier 121     Fourier features k, l ≤ 5 per side (11 × 11 = 121-d) instead of k, l ≤ 3 (49-d).
  --patch-features    J* features add a local-context channel: per side [1, cos/sin 2πk·a (k ≤ 3), cos/sin 2πk·ā (k ≤ 2)] with ā the 8×8
                      block-mean intensity at the sampled location (11 per side → 121-d, same dimension as --n-fourier 121); MI / NMI unchanged.
  --smooth 0,2,4      Gaussian pre-smoothing σ px of BOTH images, applied identically before every measure (J*, MI, NMI); one output per σ
                      (suffix _smooth{σ}); σ = 0 is the P52 protocol.
  --coarse-to-fine    optimisation only: Nelder–Mead at 64 px (≤ 50 evaluations), restarted at 128 px (≤ 50) and 256 px (≤ 50) — same total
                      budget of 150; surfaces and profiles stay at full resolution.
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
from torch.nn import functional as F

COCO = Path("/projects/EEG-foundation-model/yinghao/FMCA-AV/coco")
S = 256
TGRID = np.arange(-24, 25, 2); RGRID = np.arange(-30, 31, 1); RADII = [5, 10, 20, 30]; MEASURES = ("vcs", "mi", "nmi")


def load_grey(path):
    im = Image.open(path).convert("L"); w, h = im.size; s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((S, S), Image.BILINEAR)
    return torch.tensor(np.asarray(im, dtype=np.float32) / 255.0)


def make_modality_b(a, gen):
    g = a.clamp(0, 1) ** 0.7; b = 1 - (2 * g - 1).abs()
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, S), torch.linspace(-1, 1, S), indexing="ij")
    gain = 1 + 0.15 * torch.sin(2 * math.pi * (0.7 * xx + 0.4 * yy) + float(torch.rand(1, generator=gen)) * 6.28)
    return (b * gain + 0.03 * torch.randn(S, S, generator=gen)).clamp(0, 1)


def gaussian_smooth(img, sigma):
    """Separable Gaussian blur of a [H,W] image (reflect padding); sigma <= 0 returns the input unchanged."""
    if sigma <= 0:
        return img
    r = max(1, int(math.ceil(3 * sigma))); x = torch.arange(-r, r + 1, dtype=torch.float32); k = torch.exp(-x ** 2 / (2 * sigma ** 2)); k = k / k.sum()
    im = img[None, None]
    im = F.conv2d(F.pad(im, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1)); im = F.conv2d(F.pad(im, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
    return im[0, 0]


def block_mean(img, k=8):
    """8×8 block-mean image of the same size (box filter, reflect padding) — the local-context channel of --patch-features."""
    im = img[None, None]; p = k // 2
    return F.avg_pool2d(F.pad(im, (p, k - 1 - p, p, k - 1 - p), mode="reflect"), k, stride=1)[0, 0]


def downsample(img, f):
    return img if f == 1 else F.avg_pool2d(img[None, None], f)[0, 0]


def warp_coords(x, y, tx, ty, th):
    """Map image-A coordinates (pixels, centred) to image-B coordinates under a rigid transform."""
    c, s = math.cos(math.radians(th)), math.sin(math.radians(th))
    return c * x - s * y + tx, s * x + c * y + ty


def sample_pairs(a, b, tx, ty, th, n, gen, device, size=S, extra=()):
    """Joint pixel pairs (a(x), b(ω(x))) at n random locations of an image of side `size`; `extra` = further (a', b') image pairs sampled at the
    SAME locations (returned after the main pair).  size = S and extra = () is the P52 path."""
    xs = torch.rand(n, generator=gen) * (size - 1); ys = torch.rand(n, generator=gen) * (size - 1)
    xb, yb = warp_coords(xs - size / 2, ys - size / 2, tx, ty, th); xb, yb = xb + size / 2, yb + size / 2
    ok = (xb >= 0) & (xb <= size - 1) & (yb >= 0) & (yb <= size - 1)        # overlap mask, identical for all measures
    xs, ys, xb, yb = xs[ok], ys[ok], xb[ok], yb[ok]
    x0, y0 = xb.floor().long().clamp(0, size - 2), yb.floor().long().clamp(0, size - 2); fx, fy = xb - x0, yb - y0
    def grab(a_, b_):
        av = a_[ys.round().long(), xs.round().long()]
        bv = (b_[y0, x0] * (1 - fx) * (1 - fy) + b_[y0, x0 + 1] * fx * (1 - fy) + b_[y0 + 1, x0] * (1 - fx) * fy + b_[y0 + 1, x0 + 1] * fx * fy)  # bilinear sample of b
        return av.to(device), bv.to(device)
    av, bv = grab(a, b)
    if not extra:
        return av, bv
    return av, bv, [grab(a_, b_) for a_, b_ in extra]


def fourier_side(v, kmax):
    return [torch.ones_like(v)] + [f(2 * math.pi * k * v) for k in range(1, kmax + 1) for f in (torch.cos, torch.sin)]


def fourier_feats(a, b, kmax=3, a_ctx=None, b_ctx=None):
    fa, fb = fourier_side(a, kmax), fourier_side(b, kmax)
    if a_ctx is not None:                                             # --patch-features: block-mean channel, k <= 2
        fa += fourier_side(a_ctx, 2)[1:]; fb += fourier_side(b_ctx, 2)[1:]
    FA, FB = torch.stack(fa, 1), torch.stack(fb, 1)               # [n, 7] each in the P52 path
    return (FA[:, :, None] * FB[:, None, :]).reshape(len(a), -1)   # [n, 49]


def vcs_closed(av, bv, gen, lam=1e-3, kmax=3, ctx=None):
    if len(av) < 200:
        return float("nan")
    perm = torch.randperm(len(av), generator=gen).to(av.device)
    if ctx is None:
        pp, pq = fourier_feats(av, bv, kmax).double(), fourier_feats(av, bv[perm], kmax).double()
    else:
        ac, bc = ctx
        pp, pq = fourier_feats(av, bv, kmax, ac, bc).double(), fourier_feats(av, bv[perm], kmax, ac, bc[perm]).double()
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


class Variant:
    """The measure settings of one run: P52 defaults unless a wave-2 flag is set."""

    def __init__(self, kmax=3, patch=False, smooth=0.0, ctf=False):
        self.kmax, self.patch, self.smooth, self.ctf = kmax, patch, float(smooth), ctf

    @property
    def tag(self):
        parts = []
        if self.kmax != 3:
            parts.append(f"fourier{(2 * self.kmax + 1) ** 2}")
        if self.patch:
            parts.append("patch")
        if self.smooth > 0:
            parts.append(f"smooth{self.smooth:g}")
        if self.ctf:
            parts.append("ctf")
        return "_".join(parts) or "p52"


def run_pair(A_, B_, iid, seed, n_pairs, inits_per_radius, rng, device, var=None):
    """The P52 protocol for one aligned pair (A_, B_) of side S (truth = identity): surfaces, profiles, metrics, optimisation.  `rng` (numpy)
    draws the initial offsets in the same order as P52; `var` = None is the P52 path."""
    var = var or Variant()
    A_, B_ = gaussian_smooth(A_, var.smooth), gaussian_smooth(B_, var.smooth)
    ctx_imgs = [(block_mean(A_), block_mean(B_))] if var.patch else []
    scales = {1: (A_.to(device), B_.to(device), [(x.to(device), y.to(device)) for x, y in ctx_imgs])}
    if var.ctf:
        for f in (4, 2):
            scales[f] = (downsample(A_, f).to(device), downsample(B_, f).to(device), [(downsample(x, f).to(device), downsample(y, f).to(device)) for x, y in ctx_imgs])

    def evaluate(tx, ty, th, g=None, f=1):
        g = g or torch.Generator().manual_seed(seed + 1)   # same pixel sample for all measures at a given ω
        a_, b_, cx = scales[f]; size = S // f
        if not cx:
            av, bv = sample_pairs(a_.cpu(), b_.cpu(), tx, ty, th, n_pairs, g, device, size=size)
            v = vcs_closed(av, bv, torch.Generator().manual_seed(3), kmax=var.kmax)
        else:
            av, bv, ex = sample_pairs(a_.cpu(), b_.cpu(), tx, ty, th, n_pairs, g, device, size=size, extra=[(x.cpu(), y.cpu()) for x, y in cx])
            v = vcs_closed(av, bv, torch.Generator().manual_seed(3), kmax=var.kmax, ctx=ex[0])
        mi, nmi = hist_mi(av, bv)
        return {"vcs": v, "mi": mi, "nmi": nmi}
    # translation surfaces (theta = 0) and rotation profiles (t = 0)
    surf = {m: np.full((len(TGRID), len(TGRID)), np.nan) for m in MEASURES}
    for i, tx in enumerate(TGRID):
        for j, ty in enumerate(TGRID):
            e = evaluate(float(tx), float(ty), 0.0)
            for m in MEASURES:
                surf[m][i, j] = e[m]
    prof = {m: np.array([evaluate(0.0, 0.0, float(th))[m] for th in RGRID]) for m in MEASURES}
    c = len(TGRID) // 2; rc = len(RGRID) // 2
    rec = {"image_id": iid, "surfaces": {m: surf[m].tolist() for m in MEASURES}, "rot_profiles": {m: prof[m].tolist() for m in MEASURES}, "metrics": {}}
    for m in MEASURES:
        rec["metrics"][m] = {"peak_at_truth_translation": bool(np.nanargmax(surf[m]) == c * len(TGRID) + c), "n_local_maxima_translation": local_maxima_count(surf[m]),
                             "basin_halfwidth_px_x": basin_halfwidth(surf[m][:, c], c, 2), "basin_halfwidth_px_y": basin_halfwidth(surf[m][c, :], c, 2),
                             "basin_halfwidth_deg": basin_halfwidth(prof[m], rc, 1), "peak_at_truth_rotation": bool(np.nanargmax(prof[m]) == rc),
                             "value_at_truth": float(surf[m][c, c]), "ratio_truth_to_max_offset": float(surf[m][c, c] / np.nanmax([surf[m][0, 0], surf[m][-1, -1], surf[m][0, -1], surf[m][-1, 0]]))}
    # optimisation from random initial offsets (Nelder-Mead, same budget), success = < 1 px & < 1 deg
    opt = {m: {} for m in MEASURES}
    for R in RADII:
        for k in range(inits_per_radius):
            init = np.array([rng.uniform(-R, R), rng.uniform(-R, R), rng.uniform(-R, R)])
            for m in MEASURES:
                def f_at(f):
                    def f(p, m=m, f=f):
                        if abs(p[0]) > 64 or abs(p[1]) > 64:
                            return 1e3
                        v = evaluate(float(p[0]), float(p[1]), float(p[2]), torch.Generator().manual_seed(seed + 1), f=f)[m]
                        return 1e3 if not np.isfinite(v) else -v
                    return f
                simplex = lambda x0: x0 + np.array([[0, 0, 0], [4, 0, 0], [0, 4, 0], [0, 0, 4]])
                if not var.ctf:
                    res = minimize(f_at(1), init, method="Nelder-Mead", options={"maxfev": 150, "xatol": 0.2, "fatol": 1e-6, "initial_simplex": simplex(init)})
                    x, nfev = res.x, int(res.nfev)
                else:   # coarse-to-fine: translations in full-resolution px throughout; the coarse evaluations use p / f
                    x, nfev = init.copy(), 0
                    for f in (4, 2, 1):
                        fun = f_at(f); g = lambda p, fun=fun, f=f: fun(np.array([p[0] / f, p[1] / f, p[2]]))
                        res = minimize(g, x, method="Nelder-Mead", options={"maxfev": 50, "xatol": 0.2, "fatol": 1e-6, "initial_simplex": simplex(x)}); x, nfev = res.x, nfev + int(res.nfev)
                err_t = float(np.hypot(x[0], x[1])); err_r = float(abs(x[2]))
                opt[m].setdefault(str(R), []).append({"init": init.tolist(), "final": [float(v) for v in x], "err_px": err_t, "err_deg": err_r, "success": err_t < 1 and err_r < 1, "diverged": abs(x[0]) > 64 or abs(x[1]) > 64, "nfev": nfev})
    rec["optimisation"] = opt
    return rec


def progress_line(n_i, n_total, iid, rec, t0):
    return (f"[{n_i + 1}/{n_total}] image {iid}: " + " ".join(f"{m}: truth-peak {rec['metrics'][m]['peak_at_truth_translation']} maxima {rec['metrics'][m]['n_local_maxima_translation']} basin {rec['metrics'][m]['basin_halfwidth_px_x']}px/{rec['metrics'][m]['basin_halfwidth_deg']}deg" for m in MEASURES)
            + " | success@30: " + " ".join(f"{m}={np.mean([o['success'] for o in rec['optimisation'][m]['30']]):.2f}" for m in MEASURES) + f" ({time.time() - t0:.0f}s)")


def aggregate(per_image):
    agg = {}
    for m in MEASURES:
        agg[m] = {"peak_at_truth_translation_rate": float(np.mean([r["metrics"][m]["peak_at_truth_translation"] for r in per_image])),
                  "peak_at_truth_rotation_rate": float(np.mean([r["metrics"][m]["peak_at_truth_rotation"] for r in per_image])),
                  "mean_local_maxima_translation": float(np.mean([r["metrics"][m]["n_local_maxima_translation"] for r in per_image])),
                  "median_basin_halfwidth_px": float(np.median([min(r["metrics"][m]["basin_halfwidth_px_x"], r["metrics"][m]["basin_halfwidth_px_y"]) for r in per_image])),
                  "median_basin_halfwidth_deg": float(np.median([r["metrics"][m]["basin_halfwidth_deg"] for r in per_image])),
                  "success_rate_by_radius": {str(R): float(np.mean([o["success"] for r in per_image for o in r["optimisation"][m][str(R)]])) for R in RADII},
                  "divergence_rate_by_radius": {str(R): float(np.mean([o["diverged"] for r in per_image for o in r["optimisation"][m][str(R)]])) for R in RADII}}
    return agg


def write_outputs(out_prefix, title, subtitle, settings, device, ids, agg, per_image):
    out = {"settings": settings, "device": str(device), "image_ids": ids, "aggregate": agg, "per_image": per_image, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(out, open(out_prefix + ".json", "w"), default=float)
    L = [f"# {title} — {out['utc']}", "", subtitle, "",
         "| measure | peak at truth (transl.) | peak at truth (rot.) | mean # local maxima (transl. grid) | median basin half-width px / deg | success R=5 / 10 / 20 / 30 | divergence R=30 |", "|---|---|---|---|---|---|---|"]
    for m in MEASURES:
        g = agg[m]; L.append(f"| {m} | {g['peak_at_truth_translation_rate']:.2f} | {g['peak_at_truth_rotation_rate']:.2f} | {g['mean_local_maxima_translation']:.1f} | {g['median_basin_halfwidth_px']:.0f} / {g['median_basin_halfwidth_deg']:.0f} | "
                 + " / ".join(f"{g['success_rate_by_radius'][str(R)]:.2f}" for R in RADII) + f" | {g['divergence_rate_by_radius']['30']:.2f} |")
    Path(out_prefix + ".md").write_text("\n".join(L) + "\n"); print("->", out_prefix + ".md")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", default="/home/infres/yinwang/CS_QMI/data/coco_index"); ap.add_argument("--out", required=True)
    ap.add_argument("--n-images", type=int, default=60); ap.add_argument("--n-pairs", type=int, default=20000); ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--inits-per-radius", type=int, default=10); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-fourier", type=int, default=49, choices=[49, 121], help="B-S2: 49 = k,l <= 3 (P52); 121 = k,l <= 5")
    ap.add_argument("--patch-features", action="store_true", help="B-S2: add the 8x8 block-mean channel to the J* features (121-d); MI/NMI unchanged")
    ap.add_argument("--smooth", default="0", help="B-S2: Gaussian pre-smoothing sigma (px) applied to both images for all measures; ',' or ';' list -> one output per sigma")
    ap.add_argument("--coarse-to-fine", action="store_true", help="B-S2: Nelder-Mead 64 -> 128 -> 256 px with 50 evaluations each")
    a = ap.parse_args()
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    if a.smoke:
        a.n_images, a.inits_per_radius = 3, 2
    idx = json.load(open(Path(a.index_dir) / "val2017_index.json"))["images"]   # val2017 images: not used by A/C (which use train2017)
    ids = sorted(int(i) for i in idx); rng0 = np.random.default_rng(a.seed); rng0.shuffle(ids); ids = ids[: a.n_images]
    sigmas = [float(v) for v in a.smooth.replace(";", ",").split(",") if v.strip() != ""]
    kmax = 3 if a.n_fourier == 49 else 5
    variants = [Variant(kmax=kmax, patch=a.patch_features, smooth=s_, ctf=a.coarse_to_fine) for s_ in sigmas]
    for var in variants:
        rng = np.random.default_rng(a.seed); rng.shuffle(sorted(int(i) for i in idx))   # same rng state as P52 when the loop starts
        gen = torch.Generator().manual_seed(a.seed); per_image = []; t0 = time.time()
        for n_i, iid in enumerate(ids):
            A_ = load_grey(COCO / "val2017" / idx[str(iid)]["file"]); B_ = make_modality_b(A_, gen)
            rec = run_pair(A_, B_, iid, a.seed, a.n_pairs, a.inits_per_radius, rng, device, var if var.tag != "p52" else None)
            per_image.append(rec); print(progress_line(n_i, len(ids), iid, rec, t0), flush=True)
        agg = aggregate(per_image)
        settings = dict(vars(a)); settings["variant"] = var.tag
        prefix = a.out if (len(variants) == 1 and var.tag == "p52") else a.out + ("" if len(variants) == 1 else f"_smooth{var.smooth:g}")
        write_outputs(prefix, f"Pre-check B2 — closed-form VCS J* vs histogram MI/NMI as a rigid-registration energy ({len(ids)} constructed modality pairs)" + ("" if var.tag == "p52" else f" — variant {var.tag}"),
                      f"{a.n_pairs} pixel pairs per evaluation, identical overlap mask and samples for all measures; translation grid ±24 px step 2 (θ = 0); rotation ±30° step 1 (t = 0); "
                      f"Nelder–Mead ≤ 150 evaluations from {a.inits_per_radius} random initial offsets per radius; success = < 1 px and < 1°.", settings, device, ids, agg, per_image)
    return 0


if __name__ == "__main__":
    sys.exit(main())
