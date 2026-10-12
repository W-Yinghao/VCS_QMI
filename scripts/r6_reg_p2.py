"""R6 REG-P2 — rigid (rotation + translation) follow-up of REG-P1 on BF/SHG: the same frozen critics, references, selection rule and optimiser
family on a harder transform class (reports/R6/R6_REG_P2_PROTOCOL_*_20261011.yaml; pre-registered in REG-P1's ceiling readout).

REG-P1 is reused read-only (scripts/r6_reg_pilot.py is imported, never modified): data cache, roles (FIT 32 / CAL 8 / DEV 32), the two input
paths, every checkpoint of every trained critic run, the trained CoMIR models, the kernel-critic fit, soft MI / NMI and J.  No retraining: each
REG-P1 model is applied in the frame it was trained for —
  learned critics  warp-then-encode: for every candidate (θ, t) the moving image is resampled into the reference frame, then encoded (the encoders
                   are not rotation-equivariant, so descriptors are never rotated); the coarse grid encodes the rotation-only warp once per angle
                   and applies translations as descriptor shifts (translation equivariance, as REG-P1); the refinement is exact
  CoMIR            encode-once (primary row): the perturbed moving image as given, latents sampled at T(x) — CoMIR's own use (its
                   representations are trained for rotation equivariance: p4 + continuous rotation augmentation)
  CoMIR wte        warp-then-encode (secondary sensitivity row, DEV only): the whole moving image is resampled into the reference frame for each
                   candidate and encoded by CoMIR's SHG network, with the critics' grid machinery (rotation-only warp encoded once per angle,
                   translations as latent shifts) and an exact refinement; F = −MSE of latents
  MI / NMI         moving intensities sampled at T(x);  kernel: moving 9 × 9 patches sampled at T(x + δ) (warp-then-extract)

Geometry (512 px, pixel centres, align_corners = False, centre c = 255.5): T_{θ,t}(z) = R(θ)(z − c) + c + t maps reference to moving
coordinates.  A perturbation T* = (θ*, t*) creates the moving image V_p = V ∘ T*⁻¹ from the ×4-downsampled raw SHG intensity (bilinear,
reflection outside the image), then each input path's REG-P1 preprocessing (median / IQR; log1p for learned methods).  Every method sees
(U, V_p) only.  Support Ω = disk of radius 129 about c: every sample any method takes (critic receptive field around the bilinear neighbours
included) maps inside V for every allowed (θ, t) and (θ*, t*): 129 + 16√2 + 51 + 51 + √2 <= 255.5.

Subcommands (outputs under --root, default outputs/R6_REG_P2; REG-P1 artefacts read from --p1-root, default outputs/R6_REG):
  perturb   10 rigid perturbations per CAL / DEV image: t* = REG-P1's translation for the same image and index (strata 0–17 / 17–34 / 34–51 px
            at 512), θ* uniform on [−30°, 30°] (CoMIR's test range), seeded by sha256 of the core id
  register  one shared rigid optimiser for every energy: grid θ ∈ {−30°, −27°, …, 30°} × REG-P1's 4-px translation grid over |t| <= 51, best 3
            starts refined by 100 Adam steps (lr 0.5) on (u, t) with u = θ·ρ (ρ = RMS radius of Ω, so u is a mean displacement in px), projected
            onto |θ| <= 30°, |t| <= 51 after every step; final = best exact energy.  Selection on CAL (critics over REG-P1's 12 checkpoints,
            kernel multiplier), lexicographic: success < 42, then success < 9, then lower median corner error, then the earliest checkpoint /
            smaller multiplier; DEV evaluation; F at the truth and at the estimate; energy profiles for 5 DEV images
Error: CoMIR's mean corner error over the four corners of the central 834 × 834 full-resolution patch (±104.125 px at 512 about c), × 4 to full
resolution; success < 42 / < 9 px; also rotation error (degrees) and translation error at the centre (full-resolution px).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r6_reg_pilot as P  # noqa: E402  (REG-P1, frozen; read-only)

ROOT2 = Path("/home/infres/yinwang/CS_QMI/outputs/R6_REG_P2")
S, CEN, TMAX, THMAX, TH_STEP = P.SIZE, (P.SIZE - 1) / 2.0, P.TMAX, 30.0, 3.0
RF_BOX = P.RF_HALF + 1                                                            # descriptor receptive-field half-width + bilinear neighbour (L∞)
R_OMEGA = int(math.floor(CEN - 2 * TMAX - math.sqrt(2) - RF_BOX * math.sqrt(2)))  # 129: support disk radius about c
RHO = R_OMEGA / math.sqrt(2)                                                      # RMS radius of the disk Ω: u = θ·ρ
UMAX = math.radians(THMAX) * RHO
BOX1 = (int(math.floor(CEN - R_OMEGA - RF_BOX)), int(math.ceil(CEN + R_OMEGA + RF_BOX)))                  # [110, 401]: exact warp-then-encode box
BOX2 = (int(math.floor(CEN - R_OMEGA - TMAX - RF_BOX)), int(math.ceil(CEN + R_OMEGA + TMAX + RF_BOX)))    # [59, 452]: grid box (shifts up to 51 px)
HALF834 = 833 / 2 / P.DOWN                                                        # CoMIR's 834-px patch: corner pixel centres at ±416.5 full-res px
THR_FULL = P.THR_FULL
CHUNK = {"mi": 128, "nmi": 128, "kernel": 16, "comir": 512, "critic": 64}        # grid candidates per batch (memory only)


def seed_of(tag: str) -> int:
    return int(hashlib.sha256(tag.encode()).hexdigest()[:16], 16)


# ------------------------------------------------------------------------------------------------------------ geometry
def apply_T(torch, th, t, z):
    """Moving coordinates T_{θ_k, t_k}(z) = R(θ_k)(z − c) + c + t_k for K candidates: th [K] (rad), t [K, 2], z [N, 2] (y, x) -> [K, N, 2]."""
    c, s = torch.cos(th)[:, None], torch.sin(th)[:, None]; zy, zx = z[:, 0] - CEN, z[:, 1] - CEN
    return torch.stack([c * zy - s * zx + CEN + t[:, 0:1], s * zy + c * zx + CEN + t[:, 1:2]], -1)


def sample_pts(torch, F, M, pts, padding="zeros"):
    """Bilinear samples of a [C, H, W] map at continuous (y, x) points [K, N, 2] (pixel centres, align_corners=False, as REG-P1) -> [K, N, C]."""
    H, W = M.shape[-2:]; g = torch.stack([(pts[..., 1] + 0.5) / W * 2 - 1, (pts[..., 0] + 0.5) / H * 2 - 1], -1)
    return F.grid_sample(M[None], g[None], mode="bilinear", padding_mode=padding, align_corners=False)[0].permute(1, 2, 0)


def grid_coords(torch, lo, hi, dev):
    ii = torch.arange(lo, hi + 1, dtype=torch.float32, device=dev); yy, xx = torch.meshgrid(ii, ii, indexing="ij")
    return torch.stack([yy.reshape(-1), xx.reshape(-1)], -1)


def perturb_map(torch, F, M, th_deg, t, padding="reflection"):
    """The perturbed moving image V_p = V ∘ T*⁻¹ on the full grid: V_p(y) = V(R(−θ*)(y − c − t*) + c); M [C, S, S] -> [C, S, S]."""
    y = grid_coords(torch, 0, S - 1, M.device) - torch.tensor(t, dtype=torch.float32, device=M.device)
    src = apply_T(torch, torch.tensor([-math.radians(th_deg)], device=M.device), torch.zeros(1, 2, device=M.device), y)
    return sample_pts(torch, F, M, src, padding)[0].T.reshape(M.shape[0], S, S)


def corner_error_full(th_hat, t_hat, th_star, t_star):
    """CoMIR's mean corner error (four corners of the central 834-px patch) between the estimated and true transforms, full-resolution px."""
    a = HALF834; cs = np.array([[-a, -a], [-a, a], [a, -a], [a, a]])
    R = lambda d: np.array([[math.cos(math.radians(d)), -math.sin(math.radians(d))], [math.sin(math.radians(d)), math.cos(math.radians(d))]])
    e = (cs @ R(th_hat).T + np.asarray(t_hat)) - (cs @ R(th_star).T + np.asarray(t_star)); return float(np.linalg.norm(e, axis=1).mean() * P.DOWN)


# ------------------------------------------------------------------------------------------------------------ perturbations
def perturbations_p2(core: str) -> list:
    """10 rigid perturbations (θ* degrees, ty, tx at 512 px): REG-P1's translations for the same image and index, θ* uniform on [−30°, 30°]."""
    t = P.perturbations(core); g = np.random.default_rng(seed_of(f"R6-REG-P2|{core}")); th = g.uniform(-THMAX, THMAX, P.N_PERT)
    return [[float(th[i]), float(t[i, 0]), float(t[i, 1])] for i in range(P.N_PERT)]


def cmd_perturb(a):
    m = P.load_manifest(a.p1_root, a.smoke)
    pert = {c: perturbations_p2(c) for c, r in sorted(m["roles"].items()) if r in ("CAL", "DEV") and (a.p1_root / "cache" / f"{c}.npz").exists()}
    json.dump(pert, open(a.root / ("perturbations_smoke.json" if a.smoke else "perturbations.json"), "w"), indent=1); print(f"[perturb] {len(pert)} images × {P.N_PERT}"); return 0


class Inputs:
    """REG-P1 cache (read-only) -> per-core reference maps and, per perturbation, the moving image as given (kept in memory for reuse)."""

    def __init__(self, torch, F, p1_root, pert, dev):
        self.torch, self.F, self.root, self.pert, self.dev, self.ref, self.mov = torch, F, p1_root, pert, dev, {}, {}

    def reference(self, c):                                                      # (BF luminance [1, S, S], BF RGB [3, S, S]): REG-P1's two paths
        if c not in self.ref:
            bf, rgb = P.load_img(self.root, c, ("bf", "bf_rgb_n"))
            self.ref[c] = (self.torch.tensor(bf, device=self.dev)[None], self.torch.tensor(rgb, device=self.dev).permute(2, 0, 1).contiguous())
        return self.ref[c]

    def moving(self, c, i):                                                      # (SHG intensity path, SHG log1p path) of V ∘ T*⁻¹, each [1, S, S]
        if (c, i) not in self.mov:
            torch = self.torch; sh = np.expm1(P.load_img(self.root, c, ("shg_log",))[0].astype(np.float64)); th, ty, tx = self.pert[c][i]
            shp = perturb_map(torch, self.F, torch.tensor(sh, dtype=torch.float32, device=self.dev)[None], th, (ty, tx))[0].cpu().numpy().astype(np.float64)
            self.mov[(c, i)] = tuple(torch.tensor(x, device=self.dev)[None] for x in (P.robust_norm(shp), P.robust_norm(np.log1p(shp))))
        return self.mov[(c, i)]


# ------------------------------------------------------------------------------------------------------------ energies
def j_rows(torch, tp, tq):
    """REG-P1's J = E_P(T − T²/2) + E_Q(−T − T²/2) along the last axis."""
    return (tp - 0.5 * tp ** 2).mean(-1) + (-tq - 0.5 * tq ** 2).mean(-1)


def soft_mi_batch(torch, u, v, ranges, bins=64, sigma=1.0, nmi=False):
    """REG-P1's soft_mi (Parzen-smoothed 64-bin joint histogram, σ = 1 bin; Shannon MI or Studholme NMI) for v [K, N] against u [N] -> [K]."""
    (ul, uh), (vl, vh) = ranges; c = torch.arange(bins, device=u.device, dtype=u.dtype)
    cu = (u - ul) / (uh - ul) * (bins - 1); cv = (v - vl) / (vh - vl) * (bins - 1)
    wu = torch.exp(-0.5 * ((cu[:, None] - c) / sigma) ** 2); wu = wu / wu.sum(-1, keepdim=True).clamp_min(1e-12)
    wv = torch.exp(-0.5 * ((cv[..., None] - c) / sigma) ** 2); wv = wv / wv.sum(-1, keepdim=True).clamp_min(1e-12)
    Pj = torch.einsum("nb,knc->kbc", wu, wv) / u.shape[0]; ent = lambda p: -(p * torch.log(p.clamp_min(1e-12))).sum(-1)
    ha, hb, hab = ent(Pj.sum(2)), ent(Pj.sum(1)), ent(Pj.reshape(Pj.shape[0], -1))
    return (ha + hb) / hab if nmi else ha + hb - hab


class RigidEnergy:
    """F(θ, t) for one image and one perturbation (maximised); the method sees the reference maps and the moving image V_p only.  Positions x
    (uniform on the disk Ω) and the within-scene permutation are fixed per (image, perturbation) and shared by every method.
    batch(th, t): exact energies [K] (differentiable);  grid_values(θ°, ts): [Kt] at one grid angle (critic: one encode per angle);  p = (u, ty, tx)."""

    def __init__(self, torch, F, kind, ref, mov, rng, **kw):
        self.torch, self.F, self.kind, self.kw, self.mov = torch, F, kind, kw, mov; dev = mov.device
        r = R_OMEGA * np.sqrt(rng.uniform(0.0, 1.0, P.N_POS)); ang = rng.uniform(0.0, 2 * math.pi, P.N_POS)
        self.x = torch.tensor(np.stack([CEN + r * np.sin(ang), CEN + r * np.cos(ang)], -1), dtype=torch.float32, device=dev)
        self.perm = torch.tensor(rng.permutation(P.N_POS), device=dev)
        if kind in ("mi", "nmi", "comir", "comir_wte"):                          # ref: BF luminance (MI / NMI) or the BF CoMIR latent
            self.u = sample_pts(torch, F, ref, self.x[None])[0, :, 0]
            if kind == "comir_wte":
                self.full = grid_coords(torch, 0, S - 1, dev)
        elif kind == "kernel":                                                   # 9 × 9 patches in REG-P1's order; the BF half of W applied once
            o = torch.arange(-4, 5, device=dev, dtype=torch.float32); dy, dx = torch.meshgrid(o, o, indexing="ij")
            self.xp = (self.x[:, None, :] + torch.stack([dy.reshape(-1), dx.reshape(-1)], -1)[None]).reshape(-1, 2)
            pu = sample_pts(torch, F, ref, self.xp[None])[0, :, 0].reshape(P.N_POS, 81); self.au = pu @ kw["W"][:81] + kw["b"]; self.Wv = kw["W"][81:]
        elif kind == "critic":                                                   # ref: BF descriptors (encoded once on the full image)
            self.u = P.sample_map(torch, F, ref, self.x); self.box1, self.box2 = grid_coords(torch, *BOX1, dev), grid_coords(torch, *BOX2, dev)
        else:
            raise ValueError(kind)

    def j_critic(self, dv):                                                      # dv [K, N, 16] -> [K]
        torch, cr = self.torch, self.kw["critic"]; du = self.u[None].expand(dv.shape[0], -1, -1)
        fp = cr(torch.cat([du, dv], -1))[..., 0]; fq = cr(torch.cat([du, dv[:, self.perm]], -1))[..., 0]
        return j_rows(torch, torch.tanh(fp), torch.tanh(fq))

    def batch(self, th, t):
        torch, F, K = self.torch, self.F, len(th)
        if self.kind in ("mi", "nmi"):
            v = sample_pts(torch, F, self.mov, apply_T(torch, th, t, self.x))[..., 0]
            return soft_mi_batch(torch, self.u, v, self.kw["ranges"], nmi=self.kind == "nmi")
        if self.kind == "comir":
            v = sample_pts(torch, F, self.mov, apply_T(torch, th, t, self.x))[..., 0]; return -((self.u[None] - v) ** 2).mean(1)
        if self.kind == "kernel":
            pv = sample_pts(torch, F, self.mov, apply_T(torch, th, t, self.xp))[..., 0].reshape(K, P.N_POS, 81); zv = pv @ self.Wv; w = self.kw["w"]
            return j_rows(torch, torch.cos(self.au + zv) @ w, torch.cos(self.au + zv[:, self.perm]) @ w)
        if self.kind == "comir_wte":                                             # the whole moving image warped into the reference frame, then encoded
            out = []
            for k in range(K):
                Wk = sample_pts(torch, F, self.mov, apply_T(torch, th[k:k + 1], t[k:k + 1], self.full), "reflection")[0, :, 0].reshape(1, 1, S, S)
                out.append(-((self.u - sample_pts(torch, F, self.kw["mA"](Wk)[0], self.x[None])[0, :, 0]) ** 2).mean())
            return torch.stack(out)
        n = BOX1[1] - BOX1[0] + 1; out = []                                     # critic: warp the moving image into the reference box, then encode
        for k in range(K):
            Wk = sample_pts(torch, F, self.mov, apply_T(torch, th[k:k + 1], t[k:k + 1], self.box1))[0, :, 0].reshape(1, 1, n, n)
            out.append(self.j_critic(P.sample_map(torch, F, self.kw["eV"](Wk)[0], self.x - BOX1[0])[None])[0])
        return torch.stack(out)

    def grid_values(self, th_deg, ts):
        torch, F = self.torch, self.F; dev = ts.device; th = torch.full((1,), math.radians(th_deg), device=dev)
        if self.kind not in ("critic", "comir_wte"):
            return torch.cat([self.batch(th.expand(len(b)), b) for b in ts.split(CHUNK[self.kind])])
        zero = torch.zeros(1, 2, device=dev)                                     # rotation-only warp encoded once; W_{θ,t}(z) = W_θ(z + R(−θ) t)
        c, s = math.cos(math.radians(th_deg)), math.sin(math.radians(th_deg)); sh = torch.stack([c * ts[:, 0] + s * ts[:, 1], -s * ts[:, 0] + c * ts[:, 1]], -1)
        if self.kind == "comir_wte":                                             # whole frame (CoMIR's receptive field is not local); latent shifts
            W = sample_pts(torch, F, self.mov, apply_T(torch, th, zero, self.full), "reflection")[0, :, 0].reshape(1, 1, S, S); L = self.kw["mA"](W)[0]
            return torch.cat([-((self.u[None] - sample_pts(torch, F, L, self.x[None] + b[:, None])[..., 0]) ** 2).mean(1) for b in sh.split(CHUNK["comir"])])
        n = BOX2[1] - BOX2[0] + 1
        W = sample_pts(torch, F, self.mov, apply_T(torch, th, zero, self.box2))[0, :, 0].reshape(1, 1, n, n); D = self.kw["eV"](W)[0]
        return torch.cat([self.j_critic(sample_pts(torch, F, D, self.x[None] + b[:, None] - BOX2[0])) for b in sh.split(CHUNK["critic"])])

    def __call__(self, p):                                                       # p = (u, ty, tx), θ = u / ρ; differentiable
        return self.batch((p[0:1] / RHO), p[1:3][None])[0]

    def value(self, th_deg, t):
        torch = self.torch
        with torch.no_grad():
            return float(self.batch(torch.tensor([math.radians(th_deg)], device=self.x.device), torch.tensor([list(t)], dtype=torch.float32, device=self.x.device))[0])


# ------------------------------------------------------------------------------------------------------------ shared rigid optimiser
def grid_points():
    r = np.arange(-TMAX, TMAX + 1e-9, P.GRID_STEP)
    return np.arange(-THMAX, THMAX + 1e-9, TH_STEP), np.array([(y, x) for y in r for x in r if y * y + x * x <= TMAX * TMAX], dtype=np.float64)


def optimise_rigid(torch, energy, dev):
    """Grid θ (3°) × REG-P1's 4-px translation disk; best N_STARTS refined by N_ADAM Adam steps on (u, t) (lr ADAM_LR), projected onto
    |θ| <= 30°, |t| <= 51 after every step; returns (θ̂ degrees, t̂ [2], grid size, F at θ̂, t̂ — the exact energy)."""
    ths, ts = grid_points(); tt = torch.tensor(ts, dtype=torch.float32, device=dev)
    with torch.no_grad():
        vals = np.concatenate([energy.grid_values(float(th), tt).float().cpu().numpy() for th in ths])
    vals = np.nan_to_num(vals, nan=-np.inf); best = []
    for i in np.argsort(vals)[::-1][:P.N_STARTS]:
        th0, t0 = ths[i // len(ts)], ts[i % len(ts)]
        p = torch.tensor([math.radians(th0) * RHO, t0[0], t0[1]], dtype=torch.float32, device=dev, requires_grad=True); opt = torch.optim.Adam([p], lr=P.ADAM_LR)
        for _ in range(P.N_ADAM):
            opt.zero_grad(); (-energy(p)).backward(); opt.step()
            with torch.no_grad():
                p[0].clamp_(-UMAX, UMAX); nrm = p[1:].norm()
                if nrm > TMAX:
                    p[1:].mul_(TMAX / nrm)
        with torch.no_grad():
            best.append((float(energy(p)), p.detach().cpu().numpy().astype(np.float64)))
    f, p = max(best, key=lambda z: z[0]); return math.degrees(p[0] / RHO), p[1:], len(vals), f


# ------------------------------------------------------------------------------------------------------------ registration
def register_images(torch, F, cores, pert, maps_fn, kind, dev, kw=None, n_pert=P.N_PERT, profile_cores=()):
    recs = {}
    for c in cores:
        rec = []
        for i, (thd, ty, tx) in enumerate(pert[c][:n_pert]):
            ref, mov, extra = maps_fn(c, i); e = RigidEnergy(torch, F, kind, ref, mov, np.random.default_rng(seed_of(f"R6-REG-P2|pos|{c}|{i}")), **(kw or {}), **extra)
            t0 = time.time(); th, t, n_grid, f_hat = optimise_rigid(torch, e, dev); sec = time.time() - t0
            rec.append({"theta": thd, "t": [ty, tx], "theta_hat": th, "t_hat": t.tolist(), "err_full": corner_error_full(th, t, thd, (ty, tx)),
                        "rot_err_deg": abs(th - thd), "trans_err_full": float(np.hypot(*(t - np.array([ty, tx])))) * P.DOWN, "F_hat": f_hat,
                        "F_true": e.value(thd, (ty, tx)), "n_grid": n_grid, "seconds": sec})
        recs[c] = rec
        if c in profile_cores:                                                   # exact F around the truth of perturbation 0, inside the allowed set
            thd, ty, tx = pert[c][0]; ref, mov, extra = maps_fn(c, 0); e = RigidEnergy(torch, F, kind, ref, mov, np.random.default_rng(seed_of(f"R6-REG-P2|pos|{c}|0")), **(kw or {}), **extra)
            dth = [d for d in range(-10, 11) if abs(thd + d) <= THMAX]; dx = [s for s in range(-40, 41, 2) if math.hypot(ty, tx + s) <= TMAX]
            recs[c + "#profile"] = {"theta_offsets_deg": dth, "F_theta": [e.value(thd + d, (ty, tx)) for d in dth], "x_offsets_px512": dx, "F_x": [e.value(thd, (ty, tx + s)) for s in dx]}
    return recs


def sel_key(recs, order):
    """Lexicographic CAL selection key (maximised): success < 42, success < 9, lower median corner error, then earlier / smaller (order)."""
    errs = [r["err_full"] for k, v in recs.items() if "#" not in k for r in v]
    return (P.success(recs, THR_FULL[0]), P.success(recs, THR_FULL[1]), -float(np.median(errs)), -order)


def sel_row(recs):
    errs = [r["err_full"] for k, v in recs.items() if "#" not in k for r in v]
    return {"cal_success42": P.success(recs, THR_FULL[0]), "cal_success9": P.success(recs, THR_FULL[1]), "cal_median_err": float(np.median(errs))}


def summarise(res, part):
    recs = [r for k, v in res[part].items() if "#" not in k for r in v]
    res[f"{part}_success42"] = P.success(res[part], THR_FULL[0]); res[f"{part}_success9"] = P.success(res[part], THR_FULL[1])
    for f, name in (("err_full", "err"), ("rot_err_deg", "rot_err_deg"), ("trans_err_full", "trans_err")):
        x = [r[f] for r in recs]; res[f"{part}_median_{name}"] = float(np.median(x)) if x else None; res[f"{part}_p90_{name}"] = float(np.percentile(x, 90)) if x else None
    fail = [r for r in recs if r["err_full"] >= THR_FULL[0]]
    res[f"{part}_failures42"] = len(fail); res[f"{part}_failures42_energy"] = sum(r["F_hat"] >= r["F_true"] for r in fail)   # else: the search missed a higher F
    res[f"{part}_sec_per_registration"] = float(np.mean([r["seconds"] for r in recs])) if recs else None


def cmd_register(a):
    torch, nn, F = P.torch_mods(); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = P.load_manifest(a.p1_root, a.smoke); pert = json.load(open(a.root / ("perturbations_smoke.json" if a.smoke else "perturbations.json")))
    cal = sorted(c for c in pert if m["roles"][c] == "CAL"); dv = sorted(c for c in pert if m["roles"][c] == "DEV")
    fit = sorted(c for c, r in m["roles"].items() if r == "FIT" and (a.p1_root / "cache" / f"{c}.npz").exists()); prof = dv[:5]
    I = Inputs(torch, F, a.p1_root, pert, dev); out = a.root / "results"; out.mkdir(parents=True, exist_ok=True)
    reg = lambda cores, maps, kind, kw=None, profile=(): register_images(torch, F, cores, pert, maps, kind, dev, kw, a.n_pert, profile)
    res = {"method": a.method, "seed": a.seed, "p1_root": str(a.p1_root), "selection_rule": "lexicographic: CAL success<42, success<9, lower median corner error, earliest checkpoint / smaller multiplier", "r_omega": R_OMEGA, "rho": RHO, "box_exact": BOX1, "box_grid": BOX2, "n_pos": P.N_POS,
           "grid": {"theta_step_deg": TH_STEP, "theta_max_deg": THMAX, "t_step": P.GRID_STEP, "t_max": TMAX}, "starts": P.N_STARTS, "adam": [P.N_ADAM, P.ADAM_LR],
           "n_pert": a.n_pert, "corner_half_px512": HALF834}; t0 = time.time()
    if a.method in ("mi", "nmi"):
        def maps(c, i):
            U, V = I.reference(c)[0], I.moving(c, i)[0]; rg = lambda M: tuple(float(x) for x in np.percentile(M.cpu().numpy(), [0.5, 99.5]))
            return U, V, {"ranges": (rg(U), rg(V))}
        res["cal"] = reg(cal, maps, a.method); res["dev"] = reg(dv, maps, a.method, profile=prof)
    elif a.method == "kernel":
        sel = []; maps = lambda c, i: (I.reference(c)[0], I.moving(c, i)[0], {})
        for mult in ((1.0,) if a.smoke else (0.5, 1.0, 2.0)):
            kw = P.fit_kernel(torch, F, a.p1_root, fit, mult, a.seed, dev); kw_e = {k: kw[k] for k in ("W", "b", "w")}
            r = reg(cal, maps, "kernel", kw_e); sel.append((sel_key(r, mult), mult, kw_e, r, kw["h"]))
        best = max(sel, key=lambda z: z[0]); res["selection"] = [{"mult": s[1], "h": s[4], **sel_row(s[3])} for s in sel]; res["chosen_mult"] = best[1]
        res["cal"] = best[3]; res["dev"] = reg(dv, maps, "kernel", best[2], prof)
    elif a.method in ("comir", "comir_wte"):
        run = a.p1_root / "runs" / f"comir_s{a.seed}"; st = torch.load(run / "comir.pt", map_location=dev); mA, mB, _ = P.comir_models(torch, a.seed, dev)
        mA.load_state_dict(st["A"]); mB.load_state_dict(st["B"]); mA.eval(); mB.eval(); LU = {}; res["comir_run"] = json.load(open(run / "comir.json"))
        for p in list(mA.parameters()) + list(mB.parameters()):
            p.requires_grad_(False)
        for mod in mA.modules():                                                 # no activation checkpointing in the refinement (same values)
            if hasattr(mod, "efficient"):
                mod.efficient = False
        def maps(c, i):                                                          # A = SHG (log1p path), B = BF RGB, as REG-P1
            with torch.no_grad():
                if c not in LU:
                    LU[c] = mB(I.reference(c)[1][None])[0]
                if a.method == "comir":                                          # encode-once: the moving image as given
                    return LU[c], mA(I.moving(c, i)[1][None])[0], {}
            return LU[c], I.moving(c, i)[1], {}                                  # warp-then-encode: re-encoded for every warp
        kw = {"mA": mA} if a.method == "comir_wte" else None
        res["cal"] = reg(cal, maps, a.method, kw) if a.method == "comir" else {}   # the sensitivity row selects nothing: DEV only
        res["dev"] = reg(dv, maps, a.method, kw, prof)
    else:                                                                        # learned critic: vcs | logistic | infonce — REG-P1 checkpoints
        run = a.p1_root / "runs" / f"{a.method}_s{a.seed}"; ckpts = sorted(run.glob("ckpt_*.pt")); res["train"] = json.load(open(run / "train.json"))
        if not a.smoke and len(ckpts) != 12:
            raise RuntimeError(f"{run}: {len(ckpts)} checkpoints, REG-P1 saves 12")
        eU, eV, cr = (x.to(dev).eval() for x in P.build_models(nn)); DU = {}; sel = []
        def load(ck):
            st = torch.load(ck, map_location=dev); eU.load_state_dict(st["eU"]); eV.load_state_dict(st["eV"]); cr.load_state_dict(st["critic"]); DU.clear()
            for p in list(eU.parameters()) + list(eV.parameters()) + list(cr.parameters()):
                p.requires_grad_(False)
        def maps(c, i):
            if c not in DU:
                with torch.no_grad():
                    DU[c] = eU(I.reference(c)[1][None])[0]
            return DU[c], I.moving(c, i)[1], {}
        for ck in ckpts:
            load(ck); r = reg(cal, maps, "critic", {"eV": eV, "critic": cr}); sel.append((sel_key(r, int(ck.stem.split("_")[1])), ck, r))
            print(f"[select {a.method} s{a.seed}] {ck.name} CAL success<42 {sel[-1][0][0]:.3f} <9 {sel[-1][0][1]:.3f} median {-sel[-1][0][2]:.1f} px ({time.time() - t0:.0f}s)", flush=True)
        best = max(sel, key=lambda z: z[0]); res["chosen_ckpt"] = best[1].name; res["cal"] = best[2]
        res["selection"] = [{"ckpt": s[1].name, **sel_row(s[2])} for s in sel]
        load(best[1]); res["dev"] = reg(dv, maps, "critic", {"eV": eV, "critic": cr}, prof)
    for part in ("cal", "dev"):
        summarise(res, part)
    res["seconds_total"] = time.time() - t0; res["peak_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else None
    res["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
    name = f"{a.method}" + ("" if a.method in ("mi", "nmi") else f"_s{a.seed}") + ("_smoke" if a.smoke else ""); json.dump(res, open(out / f"{name}.json", "w"), indent=1)
    print(f"[register-p2 {name}] CAL success<42 {res['cal_success42']:.3f}  DEV success<42 {res['dev_success42']:.3f} <9 {res['dev_success9']:.3f} "
          f"median corner {res['dev_median_err']:.1f} px, rot {res['dev_median_rot_err_deg']:.2f}°, failures {res['dev_failures42']} "
          f"(energy {res['dev_failures42_energy']}) ({res['seconds_total']:.0f}s, {res['dev_sec_per_registration']:.2f}s/reg)", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["perturb", "register"]); ap.add_argument("--root", type=Path, default=ROOT2)
    ap.add_argument("--p1-root", type=Path, default=P.ROOT, help="REG-P1 outputs (cache, manifest, runs), read only")
    ap.add_argument("--method", default="vcs", choices=["vcs", "logistic", "infonce", "mi", "nmi", "kernel", "comir", "comir_wte"]); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--n-pert", type=int, default=P.N_PERT, help="perturbations registered per image (the pilot: all 10)")
    a = ap.parse_args()
    if a.smoke:
        a.root, a.p1_root = a.root / "smoke", a.p1_root / "smoke"
    a.root.mkdir(parents=True, exist_ok=True)
    return {"perturb": cmd_perturb, "register": cmd_register}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
