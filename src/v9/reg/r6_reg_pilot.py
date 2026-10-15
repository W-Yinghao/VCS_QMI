# PROVENANCE (V9-G1): verbatim copy of ssl_pilot/scripts/r6_reg_pilot.py at main commit 2d171a6 (sha256 a94a6e6ff3ab1993); read-only reference for the V9 REG extension. Do not edit; the original stays unmodified in the main repo.
"""R6 REG-P1 — learned local-relation energy for cross-modal registration on BF/SHG, translation only
(reports/R6/R6_REG_P1_PROTOCOL_*_20261011.yaml; package r6 docs/04; controlled conventions as scripts/r6_reg_controlled.py).

Subcommands (each writes under --root, default /home/infres/yinwang/CS_QMI/outputs/R6_REG):
  prepare    roles (FIT 32 / CAL 8 of train by sorted sha256 of the core id; DEV = validation1 + validation2), ×4 area downsample 2048 -> 512 and
             per-image median / IQR normalisation of two input paths: learned methods (critics and CoMIR) see BF RGB (3 channels) and log1p SHG;
             MI / NMI and the kernel see BF luminance and SHG intensity; cache + manifest with hashes
  perturb    10 seeded translations per CAL / DEV image, stratified by magnitude (0–17 / 17–34 / 34–51 px at 512 px); SHG is the moving image
  train      dense encoders (BF, SHG; 4 conv layers, 32 channels, receptive field 31 px, no coordinates / ids) -> 16-d descriptors, joint MLP
             critic f, T = tanh f; objective vcs | logistic | infonce; within-scene P / Q on random aligned crops at φ = 0; checkpoints every 250 steps
  kernel     closed-form same-target kernel critic (RFF of raw 9 × 9 patch pairs, ridge) fitted on FIT at φ = 0; bandwidth multiplier chosen on CAL
  comir      CoMIR (official DenseUNet, MIT; default recipe of train-biodata.ipynb) trained on FIT; latents for CAL / DEV
  register   one shared optimiser for every energy: coarse 4-px grid over the allowed disk, best 3 starts refined by 100 Adam steps on φ (lr 0.5 px);
             energies — learned critics (J of T with P_φ, Q_φ, both V positions warped), MI / NMI (64-bin Parzen-smoothed joint histogram),
             kernel J, CoMIR −MSE; checkpoint selection on CAL (success < 42 px full resolution), DEV evaluation, F_T profiles
Laws on the fixed interior support Ω (every sample stays inside the image for every allowed φ and perturbation t*): x uniform (continuous) on Ω;
P_φ = (U(x), V(x + φ − t*)); Q_φ = (U(x), V(x_π + φ − t*)) with a fixed within-scene permutation π — bilinear sampling on both sides, identity
included.  Error = |φ̂ − t*| at 512 px × 4 (full resolution).  Images are the independent units.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np

DATA = Path("/projects/EEG-foundation-model/yinghao/data/bf_shg_4550300"); ROOT = Path("/home/infres/yinwang/CS_QMI/outputs/R6_REG")
COMIR = Path("/projects/EEG-foundation-model/yinghao/third_party/CoMIR")
SIZE, DOWN, TMAX, RF_HALF = 512, 4, 51.0, 15
OMEGA = (int(math.ceil(RF_HALF + 2 * TMAX)), int(SIZE - RF_HALF - 1 - 2 * TMAX))   # [117, 394): x + φ − t* and its bilinear neighbours stay in [15, 496] for |φ|, |t*| <= 51
STRATA = ((0.0, 17.0), (17.0, 34.0), (34.0, 51.0)); N_PERT, N_POS, GRID_STEP, N_STARTS, N_ADAM, ADAM_LR = 10, 4096, 4.0, 3, 100, 0.5
THR_FULL = (42.0, 9.0)                                                          # CoMIR's 5 % / 1 % of 834 px, full-resolution px


# ------------------------------------------------------------------------------------------------------------ data
def read_tiff(path: Path) -> np.ndarray:
    """Uncompressed little-endian TIFF reader (strips, 8 / 16-bit, chunky or planar) — the BF/SHG files; anything else raises."""
    b = path.read_bytes()
    if b[:4] != b"II*\x00":
        raise ValueError(f"{path.name}: not a little-endian classic TIFF")
    off = struct.unpack_from("<I", b, 4)[0]; n = struct.unpack_from("<H", b, off)[0]; tags = {}
    sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 16: 8}; fmt = {1: "B", 3: "H", 4: "I"}
    for i in range(n):
        tag, typ, cnt, val = struct.unpack_from("<HHII", b, off + 2 + 12 * i)
        if typ not in fmt:
            continue
        nbytes = sizes[typ] * cnt; pos = off + 2 + 12 * i + 8 if nbytes <= 4 else val
        tags[tag] = list(struct.unpack_from("<" + fmt[typ] * cnt, b, pos))
    w, h = tags[256][0], tags[257][0]; bps = tags.get(258, [8]); spp = tags.get(277, [1])[0]; comp = tags.get(259, [1])[0]; planar = tags.get(284, [1])[0]
    if comp != 1 or len(set(bps)) != 1 or bps[0] not in (8, 16):
        raise ValueError(f"{path.name}: unsupported TIFF (compression {comp}, bits {bps})")
    dt = np.dtype(np.uint8) if bps[0] == 8 else np.dtype("<u2"); offs, cnts = tags[273], tags[279]
    raw = np.concatenate([np.frombuffer(b, dt, cnt // dt.itemsize, o) for o, cnt in zip(offs, cnts)])
    if raw.size != w * h * spp:
        raise ValueError(f"{path.name}: {raw.size} samples for {w}×{h}×{spp}")
    a = raw.reshape(h, w, spp) if planar == 1 else raw.reshape(spp, h, w).transpose(1, 2, 0)
    return a[..., 0] if spp == 1 else a


def core_files():
    out = {}
    for split in ("train", "validation1", "validation2"):
        for p in sorted((DATA / split).glob("*_WB.tif")):
            core = p.name[: -len("_WB.tif")]; q = p.with_name(core + "_MI.tif")
            if not q.exists():
                raise FileNotFoundError(q)
            out[core] = (split, p, q)
    return out


def roles(cores: dict) -> dict:
    train = sorted([c for c, v in cores.items() if v[0] == "train"], key=lambda c: hashlib.sha256(c.encode()).hexdigest())
    r = {c: ("CAL" if i < 8 else "FIT") for i, c in enumerate(train)}
    r.update({c: "DEV" for c, v in cores.items() if v[0] != "train"}); return r


def area_down(a: np.ndarray, f: int = DOWN) -> np.ndarray:
    h, w = a.shape[:2]; a = a[: h - h % f, : w - w % f].astype(np.float64)
    return a.reshape(h // f, f, w // f, f, *a.shape[2:]).mean((1, 3))


def robust_norm(a: np.ndarray) -> np.ndarray:
    q1, med, q3 = np.percentile(a, [25, 50, 75]); return ((a - med) / max(q3 - q1, 1e-6)).astype(np.float32)


def cmd_prepare(a):
    cores = core_files(); rl = roles(cores); cache = a.root / "cache"; cache.mkdir(parents=True, exist_ok=True); man = {}
    for core, (split, pw, pm) in sorted(cores.items()):
        if a.smoke and core not in smoke_cores(rl):
            continue
        bf = read_tiff(pw); shg = read_tiff(pm)
        if bf.ndim != 3 or bf.shape[2] != 3 or shg.ndim != 2 or bf.shape[:2] != shg.shape:
            raise ValueError(f"{core}: BF {bf.shape} SHG {shg.shape}")
        bfr = area_down(bf.astype(np.float64) / 255.0); lum = 0.299 * bfr[..., 0] + 0.587 * bfr[..., 1] + 0.114 * bfr[..., 2]
        sh = area_down(shg.astype(np.float64) / (255.0 if shg.dtype == np.uint8 else 65535.0))
        if lum.shape != (SIZE, SIZE):
            raise ValueError(f"{core}: downsampled to {lum.shape}, expected {SIZE}²")
        np.savez_compressed(cache / f"{core}.npz", bf=robust_norm(lum), shg=robust_norm(sh), bf_rgb=bfr.astype(np.float32), shg_log=np.log1p(sh).astype(np.float32),
                            bf_rgb_n=robust_norm(bfr), shg_log_n=robust_norm(np.log1p(sh)))      # learned-method inputs (critics and CoMIR): RGB / log1p, robust-normalised
        man[core] = {"split": split, "role": rl[core], "sha256_bf": hashlib.sha256(pw.read_bytes()).hexdigest(), "sha256_shg": hashlib.sha256(pm.read_bytes()).hexdigest(),
                     "full_shape": list(bf.shape), "shg_dtype": str(shg.dtype)}
        print(f"[prepare] {core} {split} -> {rl[core]}", flush=True)
    json.dump({"roles": rl, "files": man, "omega": OMEGA, "size": SIZE, "down": DOWN}, open(a.root / ("manifest_smoke.json" if a.smoke else "manifest.json"), "w"), indent=1)
    return 0


def smoke_cores(rl):
    by = {r: sorted(c for c, x in rl.items() if x == r) for r in ("FIT", "CAL", "DEV")}; return set(by["FIT"][:4] + by["CAL"][:2] + by["DEV"][:2])


def load_manifest(root, smoke):
    return json.load(open(root / ("manifest_smoke.json" if smoke else "manifest.json")))


def load_img(root, core, keys=("bf", "shg")):
    z = np.load(root / "cache" / f"{core}.npz"); return [z[k] for k in keys]


# ------------------------------------------------------------------------------------------------------------ perturbations
def perturbations(core: str) -> np.ndarray:
    """10 translations (dy, dx) at 512 px for an image, stratified by magnitude; seeded by the core id only."""
    g = np.random.default_rng(int(hashlib.sha256(f"R6-REG-P1|{core}".encode()).hexdigest()[:16], 16)); out = []
    for i in range(N_PERT):
        lo, hi = STRATA[i % 3]; r = math.sqrt(g.uniform(lo * lo, hi * hi)); th = g.uniform(0, 2 * math.pi); out.append((r * math.sin(th), r * math.cos(th)))
    return np.array(out, dtype=np.float64)


def cmd_perturb(a):
    m = load_manifest(a.root, a.smoke); pert = {c: perturbations(c).tolist() for c, r in m["roles"].items() if r in ("CAL", "DEV") and (not a.smoke or (a.root / "cache" / f"{c}.npz").exists())}
    json.dump(pert, open(a.root / ("perturbations_smoke.json" if a.smoke else "perturbations.json"), "w"), indent=1); print(f"[perturb] {len(pert)} images × {N_PERT}"); return 0


# ------------------------------------------------------------------------------------------------------------ torch parts
def torch_mods():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


def build_models(nn, desc=16):
    def enc(cin):                                                                # receptive field 1 + 6 + 8 + 8 + 8 = 31 px; no coordinates, no ids
        return nn.Sequential(nn.Conv2d(cin, 32, 7, padding=3), nn.GELU(), nn.Conv2d(32, 32, 9, padding=4), nn.GELU(),
                             nn.Conv2d(32, 32, 9, padding=4), nn.GELU(), nn.Conv2d(32, desc, 9, padding=4))
    critic = nn.Sequential(nn.Linear(2 * desc, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
    return enc(3), enc(1), critic                                                # BF RGB (3), SHG log1p (1): the learned-method input path


def learned_inputs(torch, root, core, dev):
    """[3, H, W] BF RGB and [1, H, W] log1p SHG, both robust-normalised — the shared input path of every learned method."""
    bf, sh = load_img(root, core, ("bf_rgb_n", "shg_log_n"))
    return torch.tensor(bf, device=dev).permute(2, 0, 1).contiguous(), torch.tensor(sh, device=dev)[None]


def sample_map(torch, F, m, yx):
    """Bilinear samples of a [C, H, W] map at continuous (y, x) positions [N, 2] (pixel centres, align_corners=False) -> [N, C]."""
    H, W = m.shape[-2:]; gx = (yx[:, 1] + 0.5) / W * 2 - 1; gy = (yx[:, 0] + 0.5) / H * 2 - 1
    g = torch.stack([gx, gy], -1)[None, None]                                   # [1, 1, N, 2]
    return F.grid_sample(m[None], g, mode="bilinear", align_corners=False)[0, :, 0].T


def j_value(torch, T_p, T_q):
    return (T_p - 0.5 * T_p ** 2).mean() + (-T_q - 0.5 * T_q ** 2).mean()


def loss_fn(torch, F, kind, f_p, f_q, f_mat=None):
    if kind == "vcs":
        return -j_value(torch, torch.tanh(f_p), torch.tanh(f_q))
    if kind == "logistic":
        return F.softplus(-2 * f_p).mean() + F.softplus(2 * f_q).mean()
    if kind == "infonce":                                                        # [S, n, n]: row i = U(x_i) against V(x_j) of the same scene
        S, n, _ = f_mat.shape; return F.cross_entropy(f_mat.reshape(S * n, n), torch.arange(n, device=f_mat.device).repeat(S))
    raise ValueError(kind)


def cmd_train(a):
    torch, nn, F = torch_mods(); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); torch.manual_seed(a.seed); g = np.random.default_rng(1000 + a.seed)
    m = load_manifest(a.root, a.smoke); fit = sorted(c for c, r in m["roles"].items() if r == "FIT" and (a.root / "cache" / f"{c}.npz").exists())
    imgs = {c: learned_inputs(torch, a.root, c, dev) for c in fit}
    eU, eV, cr = (x.to(dev) for x in build_models(nn)); params = list(eU.parameters()) + list(eV.parameters()) + list(cr.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4); steps = 200 if a.smoke else a.steps; out = a.root / "runs" / f"{a.objective}_s{a.seed}"
    out.mkdir(parents=True, exist_ok=True); crop, inner, n_sc, n_pos = 160, RF_HALF + 1, min(16, len(fit)), 128; t0 = time.time(); log = []
    for step in range(1, steps + 1):
        bu, bv = [], []
        for c in g.choice(fit, n_sc, replace=len(fit) < n_sc):
            U, V = imgs[c]; y0, x0 = g.integers(0, SIZE - crop, 2); k = int(g.integers(4)); fl = bool(g.integers(2))
            u, v = U[:, y0:y0 + crop, x0:x0 + crop], V[:, y0:y0 + crop, x0:x0 + crop]   # aligned crop; identical flip / rot90 for both modalities
            u, v = torch.rot90(u, k, (1, 2)), torch.rot90(v, k, (1, 2))
            if fl:
                u, v = u.flip(2), v.flip(2)
            bu.append(u); bv.append(v)
        DU, DV = eU(torch.stack(bu)), eV(torch.stack(bv))                            # [S, 16, crop, crop]
        yx = torch.tensor(g.uniform(inner, crop - inner - 1, (n_sc, n_pos, 2)), dtype=torch.float32, device=dev)
        perm = torch.tensor(np.stack([g.permutation(n_pos) for _ in range(n_sc)]), device=dev)
        du = torch.stack([sample_map(torch, F, DU[s], yx[s]) for s in range(n_sc)]); dv = torch.stack([sample_map(torch, F, DV[s], yx[s]) for s in range(n_sc)])
        if a.objective == "infonce":
            fm = cr(torch.cat([du[:, :, None].expand(-1, -1, n_pos, -1), dv[:, None].expand(-1, n_pos, -1, -1)], -1)).squeeze(-1)
            loss = loss_fn(torch, F, "infonce", None, None, fm)
        else:
            dvq = torch.stack([dv[s, perm[s]] for s in range(n_sc)])                  # within-scene independent positions
            fp = cr(torch.cat([du, dv], -1)).squeeze(-1); fq = cr(torch.cat([du, dvq], -1)).squeeze(-1); loss = loss_fn(torch, F, a.objective, fp, fq)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if step % (50 if a.smoke else 250) == 0 or step == steps:
            torch.save({"eU": eU.state_dict(), "eV": eV.state_dict(), "critic": cr.state_dict(), "step": step}, out / f"ckpt_{step:05d}.pt")
            log.append({"step": step, "loss": float(loss.detach()), "seconds": time.time() - t0}); print(f"[train {a.objective} s{a.seed}] step {step} loss {float(loss.detach()):.4f} ({time.time() - t0:.0f}s)", flush=True)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else None
    json.dump({"objective": a.objective, "seed": a.seed, "steps": steps, "log": log, "seconds": time.time() - t0, "peak_gib": peak, "fit": fit}, open(out / "train.json", "w"), indent=1)
    return 0


# ------------------------------------------------------------------------------------------------------------ energies
class Energy:
    """F(φ) for one image and one perturbation t*: maximised.  Positions and the within-scene permutation are fixed per (image, perturbation)."""

    def __init__(self, torch, F, kind, U, V, t_star, rng, **kw):
        self.torch, self.F, self.kind, self.kw = torch, F, kind, kw; dev = U.device
        lo, hi = OMEGA; self.x = torch.tensor(rng.uniform(lo, hi, (N_POS, 2)), dtype=torch.float32, device=dev)
        self.perm = torch.tensor(rng.permutation(N_POS), device=dev); self.t = torch.tensor(t_star, dtype=torch.float32, device=dev)
        self.U, self.V = U, V                                                    # maps [C, H, W] (descriptors, intensities, patches source or latents)
        if kind in ("critic", "kernel", "comir"):
            self.u = self.prep_u()

    def prep_u(self):
        torch, F = self.torch, self.F
        if self.kind == "kernel":
            return self.patches(self.U, self.x)
        return sample_map(torch, F, self.U, self.x)

    def patches(self, M, yx):                                                    # raw 9 × 9 patches at continuous positions -> [N, 81]
        torch = self.torch; o = torch.arange(-4, 5, device=M.device, dtype=torch.float32); dy, dx = torch.meshgrid(o, o, indexing="ij")
        pts = (yx[:, None, :] + torch.stack([dy.reshape(-1), dx.reshape(-1)], -1)[None]).reshape(-1, 2)
        return sample_map(torch, self.F, M, pts).reshape(len(yx), 81 * M.shape[0])

    def __call__(self, phi):                                                     # phi [2] (dy, dx), differentiable
        torch, F = self.torch, self.F; yv = self.x + phi - self.t
        if self.kind == "critic":
            v = sample_map(torch, F, self.V, yv); cr = self.kw["critic"]
            fp = cr(torch.cat([self.u, v], -1)).squeeze(-1); fq = cr(torch.cat([self.u, v[self.perm]], -1)).squeeze(-1)
            return j_value(torch, torch.tanh(fp), torch.tanh(fq))
        if self.kind == "kernel":
            v = self.patches(self.V, yv); W_, b_, w_ = self.kw["W"], self.kw["b"], self.kw["w"]
            tp = torch.cos(torch.cat([self.u, v], -1) @ W_ + b_) @ w_; tq = torch.cos(torch.cat([self.u, v[self.perm]], -1) @ W_ + b_) @ w_
            return j_value(torch, tp, tq)
        if self.kind == "comir":
            v = sample_map(torch, F, self.V, yv); return -((self.u - v) ** 2).mean()
        if self.kind in ("mi", "nmi"):
            u = sample_map(torch, F, self.U, self.x)[:, 0]; v = sample_map(torch, F, self.V, yv)[:, 0]; return soft_mi(torch, u, v, self.kw["ranges"], nmi=self.kind == "nmi")
        raise ValueError(self.kind)


def soft_mi(torch, u, v, ranges, bins=64, sigma=1.0, nmi=False):
    """Parzen-smoothed 64-bin joint histogram (Gaussian soft binning, σ = 1 bin) -> Shannon MI or Studholme NMI (differentiable in the samples)."""
    (ul, uh), (vl, vh) = ranges; cu = (u - ul) / (uh - ul) * (bins - 1); cv = (v - vl) / (vh - vl) * (bins - 1)
    c = torch.arange(bins, device=u.device, dtype=u.dtype); wu = torch.exp(-0.5 * ((cu[:, None] - c) / sigma) ** 2); wv = torch.exp(-0.5 * ((cv[:, None] - c) / sigma) ** 2)
    wu = wu / wu.sum(1, keepdim=True).clamp_min(1e-12); wv = wv / wv.sum(1, keepdim=True).clamp_min(1e-12)
    P = (wu.T @ wv) / len(u); pa, pb = P.sum(1), P.sum(0); ent = lambda p: -(p * torch.log(p.clamp_min(1e-12))).sum()
    ha, hb, hab = ent(pa), ent(pb), ent(P.reshape(-1))
    return (ha + hb) / hab if nmi else ha + hb - hab


def optimise(torch, energy, dev):
    """The shared optimiser: coarse 4-px grid over the disk |φ| <= TMAX, best N_STARTS refined by N_ADAM Adam steps (lr ADAM_LR px); maximises."""
    r = np.arange(-TMAX, TMAX + 1e-9, GRID_STEP); grid = [(y, x) for y in r for x in r if y * y + x * x <= TMAX * TMAX]; vals = []
    with torch.no_grad():
        for p in grid:
            vals.append(float(energy(torch.tensor(p, dtype=torch.float32, device=dev))))
    best = []
    for i in np.argsort(vals)[::-1][:N_STARTS]:
        phi = torch.tensor(grid[i], dtype=torch.float32, device=dev, requires_grad=True); opt = torch.optim.Adam([phi], lr=ADAM_LR)
        for _ in range(N_ADAM):
            opt.zero_grad(); (-energy(phi)).backward(); opt.step()
            with torch.no_grad():
                nrm = phi.norm()
                if nrm > TMAX:
                    phi.mul_(TMAX / nrm)                                          # stay inside the allowed disk (and the fixed support)
        with torch.no_grad():
            best.append((float(energy(phi)), phi.detach().cpu().numpy().copy()))
    return max(best, key=lambda z: z[0])[1], len(grid)


# ------------------------------------------------------------------------------------------------------------ kernel critic
def fit_kernel(torch, F, root, fit, mult, seed, dev, m_feat=1024, n_per=2048):
    """Closed-form same-target kernel critic T = w · ψ, ψ = √(2/m) cos(W z + b), z = [9×9 BF patch, 9×9 SHG patch]; w = A⁻¹ d maximises J."""
    g = np.random.default_rng(7000 + seed); zp, zq = [], []
    for c in fit:
        U, V = (torch.tensor(x, device=dev)[None] for x in load_img(root, c)); yx = torch.tensor(g.uniform(OMEGA[0], OMEGA[1], (n_per, 2)), dtype=torch.float32, device=dev)
        e = Energy(torch, F, "kernel", U, V, (0.0, 0.0), g, W=None, b=None, w=None); pu = e.patches(U, yx); pv = e.patches(V, yx); pm = torch.tensor(g.permutation(n_per), device=dev)
        zp.append(torch.cat([pu, pv], -1)); zq.append(torch.cat([pu, pv[pm]], -1))
    zp, zq = torch.cat(zp), torch.cat(zq); sub = zp[torch.tensor(g.choice(len(zp), 2000, replace=False), device=dev)]
    med = float(torch.cdist(sub, sub).median()); h = mult * med; gt = torch.Generator(device="cpu").manual_seed(seed)
    W_ = (torch.randn(zp.shape[1], m_feat, generator=gt) / h).to(dev); b_ = (torch.rand(m_feat, generator=gt) * 2 * math.pi).to(dev); s_ = math.sqrt(2.0 / m_feat)
    fp, fq = s_ * torch.cos(zp @ W_ + b_), s_ * torch.cos(zq @ W_ + b_); A = (fp.T @ fp) / len(fp) + (fq.T @ fq) / len(fq); d = fp.mean(0) - fq.mean(0)
    A = A + 1e-3 * A.diagonal().mean() * torch.eye(m_feat, device=dev); w_ = torch.linalg.solve(A.double(), d.double()).float()
    return {"W": W_, "b": b_, "w": w_ * s_, "median": med, "h": h}


# ------------------------------------------------------------------------------------------------------------ CoMIR (official DenseUNet, default recipe)
def comir_models(torch, seed, dev):
    sys.path.insert(0, str(COMIR)); from models.tiramisu import DenseUNet  # noqa: E402  (official code, MIT; reviewed)
    args = {"init_conv_filters": 32, "down_blocks": (4, 4, 4, 4, 4, 4), "up_blocks": (4, 4, 4, 4, 4, 4), "bottleneck_layers": 4, "upsampling_type": "upsample",
            "transition_pooling": "max", "dropout_rate": 0.2, "early_transition": False, "activation_func": None, "compression": 0.75, "efficient": True}

    class ModNet(DenseUNet):
        def __init__(self, **kw):
            super().__init__(**kw, include_top=False); self.final_conv = torch.nn.Conv2d(self.get_channels_count()[-1], 1, 1, bias=False)

        def forward(self, x):
            return self.final_conv(super().forward(x))
    torch.manual_seed(seed)
    return ModNet(in_channels=1, nb_classes=1, **args).to(dev), ModNet(in_channels=3, nb_classes=1, **args).to(dev), args


def cmd_comir(a):
    """Default recipe of CoMIR's train-biodata.ipynb on our FIT images, on the shared learned-method input path (robust-normalised log1p SHG,
    1 channel = modality A; robust-normalised BF RGB = modality B): random 128 crops,
    horizontal flip, rotation U(−180°, 180°) with reflection padding, p4 equivariance, pairwise InfoNCE on MSE similarity (τ 0.5) over 2N latents,
    activation decay L1 / L2 1e-4, SGD lr 1e-2 momentum 0.9 wd 1e-5, gradient clip 1, 23 epochs × 32 steps × 32 patches.  Then latents for CAL / DEV."""
    torch, nn, F = torch_mods(); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); g = np.random.default_rng(3000 + a.seed)
    m = load_manifest(a.root, a.smoke); have = lambda c: (a.root / "cache" / f"{c}.npz").exists()
    fit = sorted(c for c, r in m["roles"].items() if r == "FIT" and have(c)); imgs = {c: load_img(a.root, c, ("shg_log_n", "bf_rgb_n")) for c in fit}   # shared learned path
    mA, mB, args = comir_models(torch, a.seed, dev); oA = torch.optim.SGD(mA.parameters(), lr=1e-2, weight_decay=1e-5, momentum=0.9); oB = torch.optim.SGD(mB.parameters(), lr=1e-2, weight_decay=1e-5, momentum=0.9)
    epochs, spe, bs = (2, 4, 8) if a.smoke else (23, 32, 32); tau = 0.5
    if a.comir_epochs:                                                           # pipeline checks only; the pilot uses the recipe (23 × 32 × 32)
        epochs, spe, bs = a.comir_epochs, 32, 32
    out = a.root / "runs" / f"comir_s{a.seed}"; (out / "latents").mkdir(parents=True, exist_ok=True); t0 = time.time(); log = []   # every path (was only under --comir-epochs)
    for ep in range(1, epochs + 1):
        mA.train(); mB.train()
        for _ in range(spe):
            xa, xb = [], []
            for c in g.choice(fit, bs, replace=True):
                A_, B_ = imgs[c]; big = 182; y0, x0 = g.integers(0, SIZE - big, 2)
                st = torch.tensor(np.concatenate([A_[y0:y0 + big, x0:x0 + big, None], B_[y0:y0 + big, x0:x0 + big]], -1), device=dev).permute(2, 0, 1)
                if g.random() < 0.5:
                    st = st.flip(-1)
                ang = math.radians(g.uniform(-180, 180)); ca, sa = math.cos(ang), math.sin(ang)
                theta = torch.tensor([[ca, -sa, 0.0], [sa, ca, 0.0]], dtype=torch.float32, device=dev)[None]
                grid = F.affine_grid(theta, (1, 4, big, big), align_corners=False); st = F.grid_sample(st[None], grid, mode="bilinear", padding_mode="reflection", align_corners=False)[0]
                c0 = (big - 128) // 2; st = st[:, c0:c0 + 128, c0:c0 + 128]; xa.append(st[:1]); xb.append(st[1:])
            xa, xb = torch.stack(xa), torch.stack(xb); ka, kb = g.integers(4, size=bs), g.integers(4, size=bs)
            rot = lambda t, k: torch.stack([torch.rot90(t[i], int(k[i]), (1, 2)) for i in range(len(t))])
            L1 = rot(mA(rot(xa, ka)), -ka); L2 = rot(mB(rot(xb, kb)), -kb)
            pts = torch.cat([L1, L2]).reshape(2 * bs, -1); sim = -torch.cdist(pts, pts) ** 2 / pts.shape[1] / tau   # MSE similarity / τ
            n2 = 2 * bs; idx = torch.arange(n2, device=dev); pos = sim[idx, (idx + bs) % n2]; neg = sim.masked_fill(torch.eye(n2, dtype=torch.bool, device=dev), -float("inf"))
            loss = (-pos + torch.logsumexp(neg, 1)).mean()
            Z = L1.numel() + L2.numel(); loss = loss + 1e-4 * (1 + (L1.abs().sum() + L2.abs().sum())) / Z + 1e-4 * (1 + (L1.pow(2).sum() + L2.pow(2).sum())) / Z
            oA.zero_grad(); oB.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(mA.parameters(), 1.0); torch.nn.utils.clip_grad_norm_(mB.parameters(), 1.0); oA.step(); oB.step()
        log.append({"epoch": ep, "loss": float(loss.detach()), "seconds": time.time() - t0}); print(f"[comir s{a.seed}] epoch {ep} loss {float(loss.detach()):.4f} ({time.time() - t0:.0f}s)", flush=True)
    mA.eval(); mB.eval(); torch.save({"A": mA.state_dict(), "B": mB.state_dict(), "args": args}, out / "comir.pt")
    with torch.no_grad():
        for c, r in m["roles"].items():
            if r in ("CAL", "DEV") and have(c):
                A_, B_ = load_img(a.root, c, ("shg_log_n", "bf_rgb_n"))
                la = mA(torch.tensor(A_, device=dev)[None, None]); lb = mB(torch.tensor(B_, device=dev).permute(2, 0, 1)[None])
                np.savez_compressed(out / "latents" / f"{c}.npz", shg=la[0].cpu().numpy(), bf=lb[0].cpu().numpy())
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else None
    json.dump({"seed": a.seed, "epochs": epochs, "log": log, "seconds": time.time() - t0, "peak_gib": peak, "fit": fit, "comir_commit": comir_commit()}, open(out / "comir.json", "w"), indent=1)
    return 0


def comir_commit():
    head = COMIR / ".git" / "HEAD"
    try:
        ref = head.read_text().strip()
        return (COMIR / ".git" / ref[5:]).read_text().strip() if ref.startswith("ref: ") else ref
    except OSError:
        return None


# ------------------------------------------------------------------------------------------------------------ registration
def register_images(torch, F, a, cores, pert, maps_fn, kind, dev, kw=None, profile_cores=()):
    """Registers every perturbation of every image with the shared optimiser; returns per-image records (errors at full resolution)."""
    recs = {}
    for c in cores:
        U, V, extra = maps_fn(c); rec = []
        for i, t in enumerate(pert[c]):
            rng = np.random.default_rng(int(hashlib.sha256(f"pos|{c}|{i}".encode()).hexdigest()[:16], 16))   # same positions for every method
            e = Energy(torch, F, kind, U, V, t, rng, **(kw or {}), **extra); t0 = time.time(); phi, n_grid = optimise(torch, e, dev)
            err = float(np.hypot(*(phi - np.array(t)))) * DOWN; rec.append({"t": list(t), "phi": phi.tolist(), "err_full": err, "seconds": time.time() - t0})
        recs[c] = rec
        if c in profile_cores:
            rng = np.random.default_rng(int(hashlib.sha256(f"pos|{c}|0".encode()).hexdigest()[:16], 16)); t = pert[c][0]
            e = Energy(torch, F, kind, U, V, t, rng, **(kw or {}), **extra)
            with torch.no_grad():
                prof = [float(e(torch.tensor((t[0], t[1] + s), dtype=torch.float32, device=dev))) for s in np.arange(-40, 41, 2)]
            recs[c + "#profile"] = {"offsets_px512": list(range(-40, 41, 2)), "F": prof}
    return recs


def success(recs, thr):
    per = [np.mean([r["err_full"] < thr for r in v]) for k, v in recs.items() if "#" not in k]; return float(np.mean(per)) if per else float("nan")


def cmd_register(a):
    torch, nn, F = torch_mods(); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = load_manifest(a.root, a.smoke); pert = json.load(open(a.root / ("perturbations_smoke.json" if a.smoke else "perturbations.json")))
    have = lambda c: (a.root / "cache" / f"{c}.npz").exists()
    cal = sorted(c for c, r in m["roles"].items() if r == "CAL" and have(c)); dv = sorted(c for c, r in m["roles"].items() if r == "DEV" and have(c))
    fit = sorted(c for c, r in m["roles"].items() if r == "FIT" and have(c)); prof = dv[:5]; out = a.root / "results"; out.mkdir(parents=True, exist_ok=True)
    img = lambda c: [torch.tensor(x, device=dev)[None] for x in load_img(a.root, c)]
    res = {"method": a.method, "seed": a.seed, "omega": OMEGA, "n_pos": N_POS, "grid_step": GRID_STEP, "starts": N_STARTS, "adam": [N_ADAM, ADAM_LR]}; t0 = time.time()
    if a.method in ("mi", "nmi"):
        def maps(c):
            U, V = img(c); rg = lambda M: tuple(float(x) for x in np.percentile(M.cpu().numpy(), [0.5, 99.5]))
            return U, V, {"ranges": (rg(U), rg(V))}
        res["cal"] = register_images(torch, F, a, cal, pert, maps, a.method, dev); res["dev"] = register_images(torch, F, a, dv, pert, maps, a.method, dev, profile_cores=prof)
    elif a.method == "kernel":
        sel = []
        for mult in ((1.0,) if a.smoke else (0.5, 1.0, 2.0)):
            kw = fit_kernel(torch, F, a.root, fit, mult, a.seed, dev); kw_e = {k: kw[k] for k in ("W", "b", "w")}
            maps = lambda c: (*img(c), {}); r = register_images(torch, F, a, cal, pert, maps, "kernel", dev, kw_e); sel.append((success(r, THR_FULL[0]), -mult, mult, kw_e, r, kw["h"]))
        best = max(sel, key=lambda z: (z[0], z[1])); res["selection"] = [{"mult": s[2], "cal_success42": s[0], "h": s[5]} for s in sel]; res["chosen_mult"] = best[2]; res["cal"] = best[4]
        res["dev"] = register_images(torch, F, a, dv, pert, lambda c: (*img(c), {}), "kernel", dev, best[3], profile_cores=prof)
    elif a.method == "comir":
        run = a.root / "runs" / f"comir_s{a.seed}"
        def maps(c):
            z = np.load(run / "latents" / f"{c}.npz"); return torch.tensor(z["bf"], device=dev), torch.tensor(z["shg"], device=dev), {}
        res["cal"] = register_images(torch, F, a, cal, pert, maps, "comir", dev); res["dev"] = register_images(torch, F, a, dv, pert, maps, "comir", dev, profile_cores=prof)
    else:                                                                        # learned critic: vcs | logistic | infonce
        run = a.root / "runs" / f"{a.method}_s{a.seed}"; ckpts = sorted(run.glob("ckpt_*.pt")); sel = []
        eU, eV, cr = (x.to(dev).eval() for x in build_models(nn))
        def load(ck):
            st = torch.load(ck, map_location=dev); eU.load_state_dict(st["eU"]); eV.load_state_dict(st["eV"]); cr.load_state_dict(st["critic"])
            for p in list(eU.parameters()) + list(eV.parameters()) + list(cr.parameters()):
                p.requires_grad_(False)
        def maps(c):
            U, V = learned_inputs(torch, a.root, c, dev)
            with torch.no_grad():
                return eU(U[None])[0], eV(V[None])[0], {}
        for ck in ckpts:
            load(ck); r = register_images(torch, F, a, cal, pert, maps, "critic", dev, {"critic": cr}); sel.append((success(r, THR_FULL[0]), -int(ck.stem.split("_")[1]), ck, r))
        best = max(sel, key=lambda z: (z[0], z[1])); res["selection"] = [{"ckpt": s[2].name, "cal_success42": s[0]} for s in sel]; res["chosen_ckpt"] = best[2].name; res["cal"] = best[3]
        load(best[2]); res["dev"] = register_images(torch, F, a, dv, pert, maps, "critic", dev, {"critic": cr}, profile_cores=prof)
    for part in ("cal", "dev"):
        res[f"{part}_success42"] = success(res[part], THR_FULL[0]); res[f"{part}_success9"] = success(res[part], THR_FULL[1])
        errs = [r["err_full"] for k, v in res[part].items() if "#" not in k for r in v]; res[f"{part}_median_err"] = float(np.median(errs)) if errs else None
        res[f"{part}_p90_err"] = float(np.percentile(errs, 90)) if errs else None
        secs = [r["seconds"] for k, v in res[part].items() if "#" not in k for r in v]; res[f"{part}_sec_per_registration"] = float(np.mean(secs)) if secs else None
    res["seconds_total"] = time.time() - t0; res["peak_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else None
    name = f"{a.method}" + ("" if a.method in ("mi", "nmi") else f"_s{a.seed}") + ("_smoke" if a.smoke else "")
    json.dump(res, open(out / f"{name}.json", "w"), indent=1)
    print(f"[register {name}] CAL success<42 {res['cal_success42']:.3f}  DEV success<42 {res['dev_success42']:.3f} <9 {res['dev_success9']:.3f} "
          f"median {res['dev_median_err']:.1f} px ({res['seconds_total']:.0f}s, {res['dev_sec_per_registration']:.2f}s/reg)", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["prepare", "perturb", "train", "comir", "register"]); ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--objective", default="vcs", choices=["vcs", "logistic", "infonce"]); ap.add_argument("--method", default="vcs", choices=["vcs", "logistic", "infonce", "mi", "nmi", "kernel", "comir"])
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--comir-epochs", type=int, default=0, help="check runs only: CoMIR epochs at the recipe's 32 steps × 32 patches (0 = recipe)"); a = ap.parse_args()
    if a.smoke:
        a.root = a.root / "smoke"
    a.root.mkdir(parents=True, exist_ok=True)
    return {"prepare": cmd_prepare, "perturb": cmd_perturb, "train": cmd_train, "comir": cmd_comir, "register": cmd_register}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
