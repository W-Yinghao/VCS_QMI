"""Registration closing study: pair-calibrated VCS structure matcher, inference-only diagnostics A/B/C (DESIGN_CN.md).

  close_reg.py cpucheck              CPU: Torch/MF calibration vs the package NumPy reference, Q diagonal, box optimum, BCE, geometry
  close_reg.py fit                   GPU: FIT pass (host MINIMA-RoMa, descriptor samples, real seeds) -> fixed basis + global W (C4)
  close_reg.py run --role R [--shard i --nshard n] [--limit k]   GPU: CAL (A+B+C) or DEV (B+C) rows
  close_reg.py analyze               ANALYSIS.json / ANALYSIS_CN.txt
Host, problems, error metric and RANSAC are imported read-only from handoff/r23_sota_matchers_20261026 (harness/methods, LOCK
minima_roma log_r15); the bounded quadratic solver and exact moments are the V10 package backend v10.mf_vcs (imported read-only).
Roles FIT / CAL / DEV only (host DataGuard refuses TEST cores).  No training of any network; one fixed basis is fit on FIT."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
SOTA = BASE / 'handoff/r23_sota_matchers_20261026'
PKG = BASE / 'reg_arch_review/VCS_Registration_Architecture_Review_20261027'
sys.path.insert(0, str(SOTA)); sys.path.insert(0, str(PKG / 'reference'))
import harness as HN  # noqa: E402  (inserts ssl_pilot_v10_r22c/src and imports v10.adapter, methods)
import methods as MM  # noqa: E402
from v10 import mf_vcs as MF  # noqa: E402

PROT = json.loads((HERE / 'PROTOCOL.json').read_text()) if (HERE / 'PROTOCOL.json').exists() else None
SMOKE = os.environ.get('R27_SMOKE') == '1'
WORK = HERE / 'smoke' if SMOKE else HERE
OUT = WORK / 'rows'
SIZE = 512
C = np.array([(SIZE - 1) / 2, (SIZE - 1) / 2])


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, d):
    HN.write(p, d)


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


# ------------------------------------------------------------------------------------------------------------ geometry
def rot(deg):
    a = math.radians(deg); return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def true_target(z, truth):
    """Host convention mov(R(theta)(z - c) + c + t) ~ ref(z), (y, x) at the 512 grid."""
    th, ty, tx = truth
    return (z - C) @ rot(th).T + C + np.array([ty, tx])


def to_rotated(q, alpha):
    """Inverse of methods.rot_pts: point of the original moving image -> point of the image rotated by alpha."""
    return (q - C) @ rot(-alpha).T + C


def blocks(z):
    b = PROT['blocks']; n = SIZE // b['size']
    bi, bj = (z[:, 0] // b['size']).astype(int).clip(0, n - 1), (z[:, 1] // b['size']).astype(int).clip(0, n - 1)
    support = (bi + bj) % 2 == 0
    edges = np.arange(b['size'], SIZE, b['size'])
    d = np.min(np.abs(z[:, :, None] - edges[None, None, :]), axis=(1, 2)) if len(edges) else np.full(len(z), 1e9)
    return bi * n + bj, support, d >= b['guard_px']


def query_grid():
    s = PROT['queries']['stride_px']; g = np.arange(s / 2, SIZE, s)
    yy, xx = np.meshgrid(g, g, indexing='ij'); return np.stack([yy.ravel(), xx.ravel()], 1)


def window_radius(z):
    w = PROT['queries']['window']; return w['base_px'] + w['slope'] * np.linalg.norm(z - C, axis=1)


def ransac(p, q, seed):
    r = PROT['ransac']
    return MM.ransac_rigid(p, q, C, seed=seed, tau=r['tau_px'], iters=r['iters'], theta_max=float(HN.Q.THMAX), t_max=float(HN.P.TMAX))


def hypotheses_oracle(p, q, seed, truth, m):
    """Same-budget transform-candidate oracle: the top-m minimal-set hypotheses (by inlier count) of the same RANSAC draw, each
    least-squares refit once on its inliers; returns the best corner error among them (diagnostic, never used for selection)."""
    r = PROT['ransac']; n = len(p)
    if n < 2:
        return None
    rng = np.random.default_rng(seed); P_, Q_ = p - C, q - C
    i = rng.integers(0, n, size=(r['iters'], 2)); i = i[i[:, 0] != i[:, 1]]
    zp = (P_[i[:, 1], 0] - P_[i[:, 0], 0]) + 1j * (P_[i[:, 1], 1] - P_[i[:, 0], 1])
    zq = (Q_[i[:, 1], 0] - Q_[i[:, 0], 0]) + 1j * (Q_[i[:, 1], 1] - Q_[i[:, 0], 1])
    ok = (np.abs(zp) > 1e-6) & (np.abs(zq) > 1e-6)
    th = np.angle(zq / np.where(ok, zp, 1)); ct, st = np.cos(th), np.sin(th); p0, q0 = P_[i[:, 0]], Q_[i[:, 0]]
    ty = q0[:, 0] - (ct * p0[:, 0] - st * p0[:, 1]); tx = q0[:, 1] - (st * p0[:, 0] + ct * p0[:, 1])
    ok &= (np.abs(np.degrees(th)) <= HN.Q.THMAX * 1.05) & (np.hypot(ty, tx) <= HN.P.TMAX * 1.05)
    if not ok.any():
        return None
    th, ct, st, ty, tx = th[ok], ct[ok], st[ok], ty[ok], tx[ok]
    ry = ct[:, None] * P_[None, :, 0] - st[:, None] * P_[None, :, 1] + ty[:, None] - Q_[None, :, 0]
    rx = st[:, None] * P_[None, :, 0] + ct[:, None] * P_[None, :, 1] + tx[:, None] - Q_[None, :, 1]
    inl = (ry * ry + rx * rx) < r['tau_px'] ** 2; cnt = inl.sum(1); order = np.argsort(-cnt, kind='stable')[:m]
    best = None
    for k in order:
        th2, t2 = (math.degrees(th[k]), np.array([ty[k], tx[k]]))
        if inl[k].sum() >= 3:
            th2, t2 = MM.rigid_lsq(p[inl[k]], q[inl[k]], C)
        e = HN.Q.corner_error_full(th2, t2, truth[0], (truth[1], truth[2]))
        best = e if best is None else min(best, e)
    return float(best)


def err_of(res, truth):
    if res is None or res[2] < 3:
        return None
    return float(HN.Q.corner_error_full(res[0], np.array(res[1]), truth[0], (truth[1], truth[2])))


# ------------------------------------------------------------------------------------------------------------ host + features
class Host:
    """MINIMA-RoMa log_r15 exactly as the locked A method (same calls in the same order -> same RNG consumption), keeping the dense
    symmetric warp and certainty of every sweep angle, plus a 560-px encoder pass for frozen coarse/fine descriptors."""

    def __init__(self, device):
        lock = json.loads((SOTA / 'LOCK.json').read_text())['methods']['minima_roma']
        self.m = MM.build('minima_roma', lock['config'], device); self.cfg = self.m.cfg; self.net = self.m.net
        self.device = torch.device(device)
        from romatch.utils import get_tuple_transform_ops
        self.tf = get_tuple_transform_ops(resize=(560, 560), normalize=True, clahe=False)

    def run(self, ref, mov, ctx):
        from PIL import Image
        t0 = time.time(); a, b = MM.prep(ref, mov, self.cfg['preproc']); per = []; best = None
        for k, al in enumerate(self.m.angles(ctx)):
            br = b if al == 0.0 else MM.rot_img(b, al)
            A_ = Image.fromarray(a).convert('RGB'); B_ = Image.fromarray(br).convert('RGB')
            with torch.inference_mode():
                warp, cert = self.net.match(A_, B_, batched=False, device=self.device)
                mt, _ = self.net.sample(warp, cert, num=self.m.num)
                ka, kb = self.net.to_pixel_coordinates(mt, a.shape[0], a.shape[1], br.shape[0], br.shape[1])
            p = MM._xy_to_yx(ka.cpu().numpy() - 0.5); qr = MM._xy_to_yx(kb.cpu().numpy() - 0.5)
            q = MM.rot_pts(qr, al, C) if al != 0.0 else qr
            r = MM.ransac_rigid(p, q, C, seed=int(ctx['seed']) * 1009 + k, tau=self.cfg.get('tau', 3.0),
                                theta_max=ctx['theta_max'], t_max=ctx['t_max']) if len(p) else None
            per.append(dict(alpha=al, k=k, br=br, warp=warp, cert=cert, p=p, qr=qr, q=q, ransac=r))
            if r is not None and (best is None or r[2] > best[2]):
                best = (r[0], r[1], r[2], al, k)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        return a, per, best, time.time() - t0

    @torch.no_grad()
    def features(self, a, br):
        """Frozen descriptor maps: DINOv2 stride-14 (1024) bilinearly upsampled to the VGG stride-4 grid (140x140 at 560), each part
        L2-normalised, concatenated / sqrt(2).  Returns [2, D, 140, 140] float32 (ref, rotated moving)."""
        from PIL import Image
        A_, B_ = self.tf((Image.fromarray(a).convert('RGB'), Image.fromarray(br).convert('RGB')))
        X = torch.cat([A_[None], B_[None]]).to(self.device)
        fp = self.net.encoder(X, upsample=False)
        d16 = F.interpolate(fp[16].float(), size=fp[4].shape[-2:], mode='bilinear', align_corners=False)
        v4 = fp[4].float()
        return torch.cat([F.normalize(d16, dim=1), F.normalize(v4, dim=1)], 1) / math.sqrt(2.0)


def grid_norm(z):
    """(y, x) pixel centres at the 512 grid -> grid_sample normalised (x, y), align_corners=False."""
    return np.stack([(z[:, 1] + 0.5) / SIZE * 2 - 1, (z[:, 0] + 0.5) / SIZE * 2 - 1], 1)


def sample_map(fmap, z):
    """fmap [D, h, w] (torch, device), z (n, 2) (y, x) at 512 -> [n, D] float64 numpy."""
    g = torch.from_numpy(grid_norm(z)).float().to(fmap.device)[None, :, None, :]
    return F.grid_sample(fmap[None], g, mode='bilinear', align_corners=False)[0, :, :, 0].T.double().cpu().numpy()


def warp_lookup(warp, z, half):
    """Bilinear lookup of the symmetric RoMa warp (hs, 2ws, 4).  half='AB': A pixel z -> B pixel; 'BA': B pixel z -> A pixel."""
    hs, w2, _ = warp.shape; ws = w2 // 2
    part = warp[:, :ws, 2:] if half == 'AB' else warp[:, ws:, :2]
    g = torch.from_numpy(grid_norm(z)).float().to(warp.device)[None, :, None, :]
    out = F.grid_sample(part.permute(2, 0, 1)[None].float(), g, mode='bilinear', align_corners=False)[0, :, :, 0].T.cpu().numpy()
    return np.stack([SIZE / 2 * (out[:, 1] + 1) - 0.5, SIZE / 2 * (out[:, 0] + 1) - 0.5], 1)


def cert_lookup(cert, z):
    hs, w2 = cert.shape; ws = w2 // 2
    g = torch.from_numpy(grid_norm(z)).float().to(cert.device)[None, :, None, :]
    return F.grid_sample(cert[:, :ws][None, None].float(), g, mode='bilinear', align_corners=False)[0, 0, :, 0].cpu().numpy()


# ------------------------------------------------------------------------------------------------------------ basis + critics
class Basis:
    def __init__(self, path):
        d = np.load(path); self.mean = {m: d[f'mean_{m}'] for m in 'rm'}; self.comp = {m: d[f'comp_{m}'] for m in 'rm'}
        self.cent = {m: d[f'cent_{m}'] for m in 'rm'}; self.T = {m: float(d[f'T_{m}']) for m in 'rm'}

    def simplex(self, x, m):
        y = (x - self.mean[m]) @ self.comp[m].T
        d2 = ((y[:, None, :] - self.cent[m][None]) ** 2).sum(-1)
        z = -(d2 - d2.min(1, keepdims=True)) / self.T[m]; e = np.exp(z); e /= e.sum(1, keepdims=True)
        return e / e.sum(1, keepdims=True)


def fit_basis(samples_r, samples_m, prot):
    from sklearn.cluster import KMeans
    out = {}
    for m, X in (('r', samples_r), ('m', samples_m)):
        mean = X.mean(0); U, S, Vt = np.linalg.svd(X - mean, full_matrices=False); comp = Vt[:prot['pca_dim']]
        Y = (X - mean) @ comp.T
        km = KMeans(n_clusters=prot['r'], n_init=4, random_state=prot['kmeans_seed']).fit(Y)
        d2 = ((Y[:, None, :] - km.cluster_centers_[None]) ** 2).sum(-1)
        out.update({f'mean_{m}': mean, f'comp_{m}': comp, f'cent_{m}': km.cluster_centers_, f'T_{m}': np.median(d2.min(1))})
    return out


def vcs_fit(a, b, prot):
    """Pair calibration: max_W J(W) on the uniform support joint and its complete product (V mode, diagonal included)."""
    s = MF.SceneBatch(torch.from_numpy(a), torch.from_numpy(b), q_mode='v')
    return vcs_fit_scenes([s], [1.0], prot)


def vcs_fit_scenes(scenes, weights, prot):
    head = MF.QuadraticHead(scenes, weights=weights, ridge=0.0); g_ref = float(head.delta.abs().sum())
    if g_ref == 0.0:
        return np.zeros(tuple(head.delta.shape)), dict(converged=True, iterations=0, gap=0.0, g_ref=0.0, j=0.0)
    sol = MF.solve_box_head(head, initial=None, tolerance=prot['vcs']['rel_tol'] * g_ref, max_iterations=prot['vcs']['max_iter'])
    return sol.w.numpy(), dict(converged=bool(sol.converged), iterations=int(sol.iterations), gap=float(sol.fw_gap), g_ref=g_ref,
                               j=float(sol.j))


def vcs_J(W, a, b):
    s = MF.SceneBatch(torch.from_numpy(a), torch.from_numpy(b), q_mode='v')
    return float(MF.objective_terms([s], torch.from_numpy(W), weights=[1.0]).j)


def bce_value_grad(W, a, b):
    n = len(a); fP = np.einsum('ir,rs,is->i', a, W, b); FQ = a @ W @ b.T
    val = 0.5 * np.mean(np.log((1 + fP) / 2)) + 0.5 * np.mean(np.log((1 - FQ) / 2))
    g = 0.5 * (a / (1 + fP)[:, None]).T @ b / n - 0.5 * a.T @ (1.0 / (1 - FQ)) @ b / (n * n)
    return float(val), g


def bce_fit(a, b, prot, iterations=None):
    """Balanced BCE (P and Q half mass each, sigma=(1+f)/2) on the same support joint/product and the same bounded simplex-bilinear
    class (box shrunk by eps to keep the log finite).  Box-constrained L-BFGS-B; C6b runs to its own convergence (box gap <=
    rel_tol x gap(0), or max_iter), C6a is capped at `iterations` (the VCS solver's iteration count for the pair)."""
    from scipy.optimize import minimize
    c = prot['bce']; lo, hi = -1 + c['eps'], 1 - c['eps']; shp = (a.shape[1], b.shape[1])
    gap = lambda W, g: float(np.where(g >= 0, g * (hi - W), g * (lo - W)).sum())
    W0 = np.zeros(shp); v0, g0_ = bce_value_grad(W0, a, b); g0 = gap(W0, g0_); tol = c['rel_tol'] * g0
    cap = c['max_iter'] if iterations is None else max(1, int(iterations))
    fun = lambda x: tuple(-t for t in (lambda vg: (vg[0], vg[1].reshape(-1)))(bce_value_grad(x.reshape(shp), a, b)))
    res = minimize(fun, W0.reshape(-1), jac=True, method='L-BFGS-B', bounds=[(lo, hi)] * (shp[0] * shp[1]),
                   options=dict(maxiter=cap, maxfun=4 * cap + 20, ftol=0.0, gtol=1e-14))
    W = res.x.reshape(shp); val, g = bce_value_grad(W, a, b)
    return W, dict(iterations=int(res.nit), gap=gap(W, g), gap0=g0, converged=bool(gap(W, g) <= tol), value=val, message=str(res.message))


# ------------------------------------------------------------------------------------------------------------ correspondence proposal
def propose(W, aq, bmap_simplex, bpos, zq, k2_sep):
    """Fixed critic f = a_q^T W b_t; for each query the top-1 target inside its search window (rotated moving coordinates) with a
    parabolic sub-cell refinement on the 140x140 grid, plus the 2nd spatially separated peak (>= k2_sep px).  Returns
    pred (n, 2), pred2 (n, 2), score (n,), valid (n,)."""
    side = int(round(math.sqrt(len(bpos)))); S = (aq @ W) @ bmap_simplex.T          # [nq, side*side]
    rad = window_radius(zq); d = np.linalg.norm(bpos[None] - zq[:, None], axis=2); S = np.where(d <= rad[:, None], S, -np.inf)
    pred = np.full((len(zq), 2), np.nan); pred2 = pred.copy(); sc = np.full(len(zq), np.nan); valid = np.isfinite(S).any(1)
    step = SIZE / side
    for i in np.nonzero(valid)[0]:
        j = int(np.argmax(S[i])); sc[i] = S[i, j]; yi, xi = divmod(j, side); dy = dx = 0.0
        Si = S[i].reshape(side, side)
        if 0 < yi < side - 1 and np.isfinite(Si[yi - 1, xi]) and np.isfinite(Si[yi + 1, xi]):
            den = Si[yi - 1, xi] - 2 * Si[yi, xi] + Si[yi + 1, xi]; dy = 0.0 if den >= 0 else 0.5 * (Si[yi - 1, xi] - Si[yi + 1, xi]) / den
        if 0 < xi < side - 1 and np.isfinite(Si[yi, xi - 1]) and np.isfinite(Si[yi, xi + 1]):
            den = Si[yi, xi - 1] - 2 * Si[yi, xi] + Si[yi, xi + 1]; dx = 0.0 if den >= 0 else 0.5 * (Si[yi, xi - 1] - Si[yi, xi + 1]) / den
        pred[i] = bpos[j] + step * np.array([np.clip(dy, -0.5, 0.5), np.clip(dx, -0.5, 0.5)])
        far = np.linalg.norm(bpos - bpos[j], axis=1) >= k2_sep; s2 = np.where(far, S[i], -np.inf)
        if np.isfinite(s2).any():
            pred2[i] = bpos[int(np.argmax(s2))]
    return pred, pred2, sc, valid


def corr_stats(pred_orig, true_orig, valid, pred2_orig=None):
    out = {}
    e = np.linalg.norm(pred_orig - true_orig, axis=1); e = np.where(valid, e, np.inf)
    for t in PROT['correct_px']:
        out[f'correct_{t}'] = int((e <= t).sum())
    out['n'] = int(valid.sum())
    if pred2_orig is not None:
        e2 = np.where(valid, np.linalg.norm(pred2_orig - true_orig, axis=1), np.inf)
        out['top2_cover_3.0'] = int((np.minimum(e, np.nan_to_num(e2, nan=np.inf)) <= 3.0).sum())
    return out


# ------------------------------------------------------------------------------------------------------------ one problem
def select_support(host_rec, prot, cert_thr=None):
    """Observable seeds: host matches at the given angle that are host-RANSAC inliers, bidirectionally consistent (B->A warp within
    tau), with RoMa certainty >= thr, inside support blocks outside the guard band; stratified cap per block (highest certainty)."""
    s = prot['support']; p, qr, r = host_rec['p'], host_rec['qr'], host_rec['ransac']
    if r is None or len(p) == 0:
        return np.zeros(0, int), {}
    inl = MM.residuals(r[0], r[1], p, host_rec['q'], C) < prot['ransac']['tau_px']
    back = warp_lookup(host_rec['warp'], qr, 'BA'); bidir = np.linalg.norm(back - p, axis=1) <= s['bidir_px']
    cert = cert_lookup(host_rec['cert'], p); blk, sup, guard = blocks(p)
    ok = inl & bidir & (cert >= s['cert_min']) & sup & guard
    idx = []
    for bk in np.unique(blk[ok]):
        cand = np.nonzero(ok & (blk == bk))[0]; cand = cand[np.argsort(-cert[cand], kind='stable')][:s['per_block']]; idx.extend(cand)
    idx = np.array(sorted(idx), int)[:s['max_total']]
    return idx, dict(n_inlier=int(inl.sum()), n_bidir=int((inl & bidir).sum()), n_ok=int(ok.sum()), cert=cert, ok=ok, blk=blk)


def hull_area(z):
    if len(z) < 3:
        return 0.0
    import cv2
    return float(cv2.contourArea(cv2.convexHull(z[:, ::-1].astype(np.float32))))


def h_stats(a, b):
    s = MF.SceneBatch(torch.from_numpy(a), torch.from_numpy(b), q_mode='v'); head = MF.QuadraticHead([s], weights=[1.0])
    n = head.delta.numel(); E = torch.eye(n, dtype=torch.float64)
    Hm = torch.stack([head.hvp(E[i].reshape(head.delta.shape)).reshape(-1) for i in range(n)]).numpy() if n <= 1100 else None
    if Hm is None:
        return {}
    ev = np.clip(np.linalg.eigvalsh((Hm + Hm.T) / 2), 0, None); p = ev / ev.sum() if ev.sum() > 0 else ev
    erank = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum())) if ev.sum() > 0 else 0.0
    pos = ev[ev > 1e-12 * ev.max()] if ev.max() > 0 else ev
    return dict(eff_rank=erank, cond=float(pos.max() / pos.min()) if len(pos) else None, n_params=n)


def problem(host, basis, Wg, ref, mov, truth, c, i, role, prot):
    ctx = HN.ctx_of(c, i); torch.manual_seed(ctx['seed']); np.random.seed(ctx['seed'] % (2 ** 32))
    t0 = time.time(); a_img, per, best, host_sec = host.run(ref, mov, ctx)
    row = dict(core=c, perturbation=i, role=role, truth=[float(x) for x in truth], host_seconds=host_sec)
    if best is None or best[2] < 3:
        row['host'] = dict(err=None, ok=False); row['fallback'] = 'host_failed'; return row
    row['host'] = dict(err=err_of(best[:3], truth), ok=True, theta=best[0], t=list(map(float, best[1])), n_inl=int(best[2]),
                       alpha=best[3])
    zq_all = query_grid(); blk, sup, guard = blocks(zq_all); zq = zq_all[(~sup) & guard]
    rad = window_radius(zq)
    side = 140; g = (np.arange(side) + 0.5) * SIZE / side - 0.5; yy, xx = np.meshgrid(g, g, indexing='ij'); bpos = np.stack([yy.ravel(), xx.ravel()], 1)

    def pack(alpha_rec):
        fm = host.features(a_img, alpha_rec['br'])
        bsim = basis.simplex(fm[1].permute(1, 2, 0).reshape(-1, fm.shape[1]).double().cpu().numpy(), 'm')
        aq = basis.simplex(sample_map(fm[0], zq), 'r'); return fm, bsim, aq

    def expand(W, aq, bsim, alpha):
        pred, pred2, sc, valid = propose(W, aq, bsim, bpos, zq, prot['queries']['peak2_sep_px'])
        po = np.where(valid[:, None], MM.rot_pts(np.nan_to_num(pred), alpha, C), np.nan)
        p2o = np.where(np.isfinite(pred2).all(1)[:, None], MM.rot_pts(np.nan_to_num(pred2), alpha, C), np.nan)
        return po, p2o, valid

    tq = true_target(zq, truth)
    # ---------------- A: oracle support at the sweep angle nearest the true rotation (CAL/FIT only)
    if role in ('FIT', 'CAL'):
        ko = int(np.argmin([abs(r['alpha'] - truth[0]) for r in per])); rec = per[ko]; fm, bsim, aq = pack(rec)
        zs_all = zq_all[sup & guard]; ts = to_rotated(true_target(zs_all, truth), rec['alpha'])
        inside = ((ts >= 8) & (ts <= SIZE - 9)).all(1); zs, ts = zs_all[inside], ts[inside]
        bs, _, _ = blocks(zs); keep = []
        for bk in np.unique(bs):
            keep.extend(np.nonzero(bs == bk)[0][:prot['support']['per_block']])
        keep = np.array(sorted(keep), int)[:prot['support']['max_total']]; zs, ts = zs[keep], ts[keep]
        a_s = basis.simplex(sample_map(fm[0], zs), 'r'); b_s = basis.simplex(sample_map(fm[1], ts), 'm')
        Wo, info_o = vcs_fit(a_s, b_s, prot); po, p2o, valid = expand(Wo, aq, bsim, rec['alpha'])
        Arow = dict(alpha=rec['alpha'], n_support=len(zs), solver=info_o, oracle=corr_stats(po, tq, valid, p2o))
        nr = ransac(zq[valid], po[valid], int(ctx['seed']) * 7919 + 1); Arow['oracle']['new_only_err'] = err_of(nr, truth)
        hr = ransac(np.concatenate([rec['p'], zq[valid]]), np.concatenate([rec['q'], po[valid]]), int(ctx['seed']) * 7919 + 2)
        Arow['oracle']['host_plus_new_err'] = err_of(hr, truth); Arow['host_angle_err'] = err_of(rec['ransac'], truth)
        idx, sinfo = select_support(rec, prot)                       # real seeds at the same angle, same expansion
        if len(idx) >= prot['support']['min_total']:
            a_r = basis.simplex(sample_map(fm[0], rec['p'][idx]), 'r'); b_r = basis.simplex(sample_map(fm[1], rec['qr'][idx]), 'm')
            Wr, info_r = vcs_fit(a_r, b_r, prot); po, p2o, valid = expand(Wr, aq, bsim, rec['alpha'])
            Arow['real'] = dict(n_support=len(idx), solver=info_r, **corr_stats(po, tq, valid, p2o))
            nr = ransac(zq[valid], po[valid], int(ctx['seed']) * 7919 + 1); Arow['real']['new_only_err'] = err_of(nr, truth)
        row['A'] = Arow
    # ---------------- B + C at the host-selected angle
    rec = per[best[4]]; fm, bsim, aq = pack(rec); seed0 = int(ctx['seed']) * 1009 + best[4]
    idx, sinfo = select_support(rec, prot); zs = rec['p'][idx]; qs = rec['q'][idx]
    true_s = true_target(zs, truth); s_err = np.linalg.norm(qs - true_s, axis=1)
    nblk = int(len(np.unique(blocks(zs)[0]))) if len(zs) else 0
    B = dict(n_support=int(len(idx)), n_blocks=nblk, hull_area=hull_area(zs), support_err_rate_3=float(np.mean(s_err > 3.0)) if len(zs) else None,
             n_host_inlier=sinfo.get('n_inlier'), n_bidir=sinfo.get('n_bidir'), n_ok=sinfo.get('n_ok'), n_queries=int(len(zq)))
    fallback = len(idx) < prot['support']['min_total'] or nblk < prot['support']['min_blocks']
    arms = {'C1_host': dict(err=row['host']['err'])}
    hp, hq = rec['p'], rec['q']
    # C2: same queries, RoMa's own dense warp target (equal-budget extra search)
    q2 = MM.rot_pts(warp_lookup(rec['warp'], zq, 'AB'), rec['alpha'], C); v2 = np.ones(len(zq), bool)
    arms['C2_extra_search'] = dict(err=err_of(ransac(np.concatenate([hp, zq]), np.concatenate([hq, q2]), seed0), truth),
                                   new=corr_stats(q2, tq, v2), new_only_err=err_of(ransac(zq, q2, seed0 + 1), truth),
                                   oracle20=hypotheses_oracle(np.concatenate([hp, zq]), np.concatenate([hq, q2]), seed0, truth, prot['oracle_m']))
    arms['C1_host']['oracle20'] = hypotheses_oracle(hp, hq, seed0, truth, prot['oracle_m'])
    if fallback:
        row['fallback'] = 'support'
        arms['C3_seed_only'] = dict(err=row['host']['err'], fallback=True)
        for k in ('C4_global_vcs', 'C5_pair_vcs', 'C6a_pair_bce_samecompute', 'C6b_pair_bce_converged'):
            arms[k] = dict(err=row['host']['err'], fallback=True)
        row['B'] = B; row['C'] = arms; row['seconds'] = time.time() - t0; return row
    arms['C3_seed_only'] = dict(err=err_of(ransac(zs, qs, seed0 + 2), truth))
    a_s = basis.simplex(sample_map(fm[0], zs), 'r'); b_s = basis.simplex(sample_map(fm[1], rec['qr'][idx]), 'm')
    Wp, info = vcs_fit(a_s, b_s, prot); B['solver'] = info; B.update(h_stats(a_s, b_s))
    # fit vs holdout J: held-out host matches passing the same criteria inside check blocks
    _, hsup, hguard = blocks(hp)
    inl = MM.residuals(rec['ransac'][0], rec['ransac'][1], hp, hq, C) < prot['ransac']['tau_px']
    back = warp_lookup(rec['warp'], rec['qr'], 'BA'); bid = np.linalg.norm(back - hp, axis=1) <= prot['support']['bidir_px']
    cert = cert_lookup(rec['cert'], hp); hold = inl & bid & (cert >= prot['support']['cert_min']) & (~hsup) & hguard
    hidx = np.nonzero(hold)[0][:prot['support']['max_total']]
    B['J_fit'] = vcs_J(Wp, a_s, b_s)
    if len(hidx) >= 2:
        a_h = basis.simplex(sample_map(fm[0], hp[hidx]), 'r'); b_h = basis.simplex(sample_map(fm[1], rec['qr'][hidx]), 'm')
        B['J_holdout'] = vcs_J(Wp, a_h, b_h); B['n_holdout'] = int(len(hidx))
    if not info['converged']:
        row['fallback'] = 'solver'
    critics = {'C4_global_vcs': Wg, 'C5_pair_vcs': Wp}
    Wb, ib = bce_fit(a_s, b_s, prot, iterations=max(1, info['iterations'])); critics['C6a_pair_bce_samecompute'] = Wb; B['bce_a'] = ib
    Wb2, ib2 = bce_fit(a_s, b_s, prot); critics['C6b_pair_bce_converged'] = Wb2; B['bce_b'] = ib2
    for k, W in critics.items():
        if k == 'C5_pair_vcs' and not info['converged']:
            arms[k] = dict(err=row['host']['err'], fallback=True); continue
        po, p2o, valid = expand(W, aq, bsim, rec['alpha'])
        e_main = err_of(ransac(np.concatenate([hp, zq[valid]]), np.concatenate([hq, po[valid]]), seed0), truth)
        arms[k] = dict(err=e_main, new=corr_stats(po, tq, valid, p2o), new_only_err=err_of(ransac(zq[valid], po[valid], seed0 + 1), truth),
                       seed_plus_new_err=err_of(ransac(np.concatenate([zs, zq[valid]]), np.concatenate([qs, po[valid]]), seed0 + 3), truth))
        if k == 'C5_pair_vcs':
            arms[k]['oracle20'] = hypotheses_oracle(np.concatenate([hp, zq[valid]]), np.concatenate([hq, po[valid]]), seed0, truth, prot['oracle_m'])
    row['B'] = B; row['C'] = arms; row['seconds'] = time.time() - t0
    return row


# ------------------------------------------------------------------------------------------------------------ modes
def protocol_check():
    assert PROT is not None, 'PROTOCOL.json must exist before any run'
    assert sha(HERE / 'close_reg.py') == PROT['frozen']['close_reg.py'], 'close_reg.py changed since PROTOCOL freeze'


def cmd_fit(limit=None):
    WORK.mkdir(exist_ok=True); protocol_check(); dev = 'cuda'; g = HN.make_guard(); host = Host(dev); probs = HN.Problems(g, 'FIT')
    pts = [(c, i) for c in probs.cores for i in range(PROT['fit']['n_pert'])][:limit]
    rng = np.random.default_rng(PROT['fit']['sample_seed']); Sr, Sm, seeds = [], [], []
    for c, i in pts:
        ctx = HN.ctx_of(c, i); torch.manual_seed(ctx['seed']); np.random.seed(ctx['seed'] % (2 ** 32))
        a_img, per, best, _ = host.run(probs.ref(c), probs.mov(c, i), ctx)
        if best is None:
            continue
        rec = per[best[4]]; fm = host.features(a_img, rec['br'])
        z = rng.uniform(8, SIZE - 9, size=(PROT['fit']['samples_per_image'], 2))
        Sr.append(sample_map(fm[0], z)); Sm.append(sample_map(fm[1], z))
        idx, _ = select_support(rec, PROT)
        if len(idx) >= PROT['support']['min_total']:
            seeds.append((sample_map(fm[0], rec['p'][idx]), sample_map(fm[1], rec['qr'][idx])))
        print('FIT', c, i, len(idx), flush=True)
    probs.clear()
    bas = fit_basis(np.concatenate(Sr), np.concatenate(Sm), PROT['basis']); np.savez(WORK / 'BASIS.npz', **bas); basis = Basis(WORK / 'BASIS.npz')
    scenes = [MF.SceneBatch(torch.from_numpy(basis.simplex(a, 'r')), torch.from_numpy(basis.simplex(b, 'm')), q_mode='v') for a, b in seeds]
    Wg, info = vcs_fit_scenes(scenes, [1.0 / len(scenes)] * len(scenes), PROT)
    np.save(WORK / 'W_GLOBAL.npy', Wg)
    write(WORK / 'FIT.json', dict(n_problems=len(pts), n_scenes=len(scenes), solver=info, T=basis.T, basis_sha256=sha(WORK / 'BASIS.npz'),
                                  W_sha256=sha(WORK / 'W_GLOBAL.npy'), job=os.environ.get('SLURM_JOB_ID'),
                                  roles_read=sorted({x['role'] for x in g.reads}), gpu=torch.cuda.get_device_name()))
    print('FIT done', info, flush=True)


def cmd_run(role, shard, nshard, limit):
    protocol_check(); fit = json.loads((WORK / 'FIT.json').read_text())
    assert sha(WORK / 'BASIS.npz') == fit['basis_sha256'] and sha(WORK / 'W_GLOBAL.npy') == fit['W_sha256']
    basis = Basis(WORK / 'BASIS.npz'); Wg = np.load(WORK / 'W_GLOBAL.npy')
    g = HN.make_guard(); host = Host('cuda'); probs = HN.Problems(g, role)
    allp = probs.all()[shard::nshard][:limit]; path = OUT / f'{role}_s{shard}of{nshard}.jsonl'; path.parent.mkdir(exist_ok=True)
    done = {(r['core'], r['perturbation']) for r in HN.rows(path)}; last = None
    with path.open('a') as f:
        for c, i in allp:
            if (c, i) in done:
                continue
            if c != last and last is not None:
                probs.clear()
            last = c
            r = problem(host, basis, Wg, probs.ref(c), probs.mov(c, i), probs.pert[c][i], c, i, role, PROT)
            r.update(job=os.environ.get('SLURM_JOB_ID'), gpu=torch.cuda.get_device_name()); f.write(json.dumps(jsonable(r)) + '\n'); f.flush()
            print(role, c, i, {k: v.get('err') for k, v in r.get('C', {}).items()}, f"{r.get('seconds', 0):.1f}s", flush=True)
    assert {x['role'] for x in g.reads} <= {'FIT', 'CAL', 'DEV'}
    print('ROLES', sorted({x['role'] for x in g.reads}), flush=True)


def cmd_cpucheck():
    """Torch/MF calibration against the package NumPy reference; Q includes the diagonal; box optimum; BCE; geometry conventions."""
    import quadratic_pair_adapter as R
    from scipy.optimize import minimize
    rng = np.random.default_rng(0); res = {}
    def simplex(n, k):
        x = rng.gamma(0.7, size=(n, k)); return x / x.sum(1, keepdims=True)
    n, r, s = 40, 5, 4; a, b = simplex(n, r), simplex(n, s); W = rng.uniform(-1, 1, (r, s))
    sc = MF.SceneBatch(torch.from_numpy(a), torch.from_numpy(b), q_mode='v')
    j_mf = float(MF.objective_terms([sc], torch.from_numpy(W), weights=[1.0]).j)
    Wjoint = np.eye(n) / n; d, H = R.vcs_problem(a, b, Wjoint); aref = np.concatenate([[0.0], math.sqrt(2) * W.reshape(-1)])
    j_ref = R.objective(aref, d, H)
    F_ = a @ W @ b.T; diag = np.diag(F_); j_dense = diag.mean() - F_.mean() - 0.5 * (np.mean(diag ** 2) + np.mean(F_ ** 2))
    offmask = ~np.eye(n, dtype=bool); j_off = diag.mean() - F_[offmask].mean() - 0.5 * (np.mean(diag ** 2) + np.mean(F_[offmask] ** 2))
    res['J'] = dict(mf=j_mf, reference=j_ref, dense_with_diag=j_dense, dense_offdiag_only=j_off,
                    pass_=abs(j_mf - j_ref) < 1e-12 and abs(j_mf - j_dense) < 1e-12 and abs(j_mf - j_off) > 1e-6)
    Wt = torch.from_numpy(W).requires_grad_(True); MF.objective_terms([sc], Wt, weights=[1.0]).j.backward()
    g_ref = (math.sqrt(2) * (d - H @ aref)[1:]).reshape(r, s)
    res['grad'] = dict(max_abs_diff=float(np.abs(Wt.grad.numpy() - g_ref).max())); res['grad']['pass_'] = res['grad']['max_abs_diff'] < 1e-12
    head = MF.QuadraticHead([sc], weights=[1.0]); sol = MF.solve_box_head(head, tolerance=1e-12, max_iterations=200000)
    Hw = (H[1:, 1:] * 2.0); dw = math.sqrt(2) * d[1:]
    opt = minimize(lambda x: -(x @ dw - 0.5 * x @ Hw @ x), np.zeros(r * s), jac=lambda x: -(dw - Hw @ x), method='L-BFGS-B',
                   bounds=[(-1, 1)] * (r * s), options=dict(ftol=1e-15, gtol=1e-12, maxiter=100000))
    j_box = float(sol.j); j_lb = float(-opt.fun)
    res['box_optimum'] = dict(mf_solver_J=j_box, scipy_lbfgsb_J=j_lb, fw_gap=float(sol.fw_gap), converged=bool(sol.converged),
                              W_max_abs_diff=float(np.abs(sol.w.numpy().reshape(-1) - opt.x).max()))
    res['box_optimum']['pass_'] = abs(j_box - j_lb) < 1e-9 and sol.converged
    v = subprocess.run([sys.executable, '-B', str(PKG / 'reference/validate_quadratic_pair_adapter.py')], capture_output=True, text=True, cwd=str(PKG / 'reference'))
    res['package_reference_validation'] = dict(returncode=v.returncode, tail=v.stdout[-600:]); res['package_reference_validation']['pass_'] = v.returncode == 0
    # BCE gradient and value
    Wb = rng.uniform(-0.9, 0.9, (r, s)); val, gb = bce_value_grad(Wb, a, b)
    eps = 1e-6; num = np.zeros_like(Wb)
    for u in range(r):
        for w in range(s):
            Wp_, Wm_ = Wb.copy(), Wb.copy(); Wp_[u, w] += eps; Wm_[u, w] -= eps
            num[u, w] = (bce_value_grad(Wp_, a, b)[0] - bce_value_grad(Wm_, a, b)[0]) / (2 * eps)
    res['bce_grad'] = dict(max_abs_diff=float(np.abs(num - gb).max())); res['bce_grad']['pass_'] = res['bce_grad']['max_abs_diff'] < 1e-7
    # table critic: VCS and balanced BCE share T*=(p-q)/(p+q)
    p_ = rng.dirichlet(np.ones(12)); q_ = rng.dirichlet(np.ones(12)); T = (p_ - q_) / (p_ + q_)
    gv = (p_ - q_) - (p_ + q_) * T; gbce = 0.5 * p_ / (1 + T) - 0.5 * q_ / (1 - T)
    res['table_optimum'] = dict(vcs_grad=float(np.abs(gv).max()), bce_grad=float(np.abs(gbce).max()))
    res['table_optimum']['pass_'] = max(res['table_optimum'].values()) < 1e-12
    # geometry: truth mapping consistent with methods.residuals; window covers the true target at the nearest sweep angle
    cov = []; resid = []
    for _ in range(2000):
        th = rng.uniform(-30, 30); t = rng.uniform(-1, 1, 2); t = t / max(1.0, np.linalg.norm(t)) * rng.uniform(0, 51)
        z = rng.uniform(0, 511, (8, 2)); q = true_target(z, (th, t[0], t[1])); resid.append(MM.residuals(th, t, z, q, C).max())
        al = min((-30, -15, 0, 15, 30), key=lambda x: abs(x - th)); zr = to_rotated(q, al)
        cov.append(bool((np.linalg.norm(zr - z, axis=1) <= window_radius(z)).all()))
        assert np.allclose(MM.rot_pts(zr, al, C), q)
    res['geometry'] = dict(max_residual=float(max(resid)), window_coverage=float(np.mean(cov)))
    res['geometry']['pass_'] = res['geometry']['max_residual'] < 1e-9 and res['geometry']['window_coverage'] == 1.0
    # simplex basis + BCE solver run end-to-end on the toy
    Wbce, ib = bce_fit(a, b, PROT); res['bce_solver'] = dict(ib, pass_=bool(ib['converged']))
    res['all_pass'] = all(v['pass_'] for v in res.values() if isinstance(v, dict) and 'pass_' in v)
    write(HERE / 'CPU_CHECK.json', jsonable(res)); print(json.dumps(jsonable(res), indent=1)); assert res['all_pass']


def cmd_analyze():
    rs = []
    for p in sorted(OUT.glob('*.jsonl')):
        rs += HN.rows(p)
    out = {}
    for role in ('CAL', 'DEV'):
        R_ = [r for r in rs if r['role'] == role]
        if not R_:
            continue
        cores = sorted({r['core'] for r in R_}); arms = sorted({k for r in R_ for k in r.get('C', {})})
        def errs(k):
            return np.array([np.inf if (r.get('C', {}).get(k, {}).get('err') is None) else r['C'][k]['err'] for r in R_])
        task = {}
        for k in arms:
            e = errs(k)
            task[k] = dict(n=len(e), strict=float(np.mean(e < 9)), loose=float(np.mean(e < 42)), loose_fail=int(np.sum(e >= 42)),
                           median=float(np.median(e)), p95=float(np.percentile(e, 95)),
                           worst_core=max(cores, key=lambda c: np.mean([x for x, r in zip(e, R_) if r['core'] == c])),
                           fallback=int(sum(bool(r['C'].get(k, {}).get('fallback')) for r in R_ if 'C' in r)))
            o = [r['C'][k].get('oracle20') for r in R_ if 'C' in r and r['C'].get(k, {}).get('oracle20') is not None]
            if o:
                task[k]['oracle20_strict'] = float(np.mean(np.array(o) < 9))
            nw = [r['C'][k]['new'] for r in R_ if 'C' in r and 'new' in r['C'].get(k, {})]
            if nw:
                tot = sum(x['n'] for x in nw)
                task[k]['new_correct_rate'] = {t: sum(x[f'correct_{t}'] for x in nw) / max(tot, 1) for t in PROT['correct_px']}
                if 'top2_cover_3.0' in nw[0]:
                    task[k]['top2_cover_3.0'] = sum(x['top2_cover_3.0'] for x in nw) / max(tot, 1)
                no = np.array([np.inf if r['C'][k].get('new_only_err') is None else r['C'][k]['new_only_err'] for r in R_ if 'C' in r and 'new' in r['C'].get(k, {})])
                task[k]['new_only_strict'] = float(np.mean(no < 9)); task[k]['new_only_loose'] = float(np.mean(no < 42))
            sp = [r['C'][k].get('seed_plus_new_err') for r in R_ if 'C' in r and k in r['C'] and 'seed_plus_new_err' in r['C'][k]]
            if sp:
                sp = np.array([np.inf if x is None else x for x in sp]); task[k]['seed_plus_new_strict'] = float(np.mean(sp < 9))
        rng = np.random.default_rng(20261027); pairs = {}
        for k in ('C1_host', 'C2_extra_search', 'C3_seed_only', 'C6a_pair_bce_samecompute', 'C6b_pair_bce_converged', 'C4_global_vcs'):
            if k not in arms or 'C5_pair_vcs' not in arms:
                continue
            d = (errs('C5_pair_vcs') < 9).astype(float) - (errs(k) < 9).astype(float)
            dl = (errs('C5_pair_vcs') < 42).astype(float) - (errs(k) < 42).astype(float)
            per = np.array([d[[r['core'] == c for r in R_]].mean() for c in cores]); perl = np.array([dl[[r['core'] == c for r in R_]].mean() for c in cores])
            bs = np.array([per[rng.integers(0, len(cores), len(cores))].mean() for _ in range(10000)])
            pairs[f'C5-{k}'] = dict(strict_pp=100 * float(d.mean()), ci95_pp=[100 * float(np.percentile(bs, 2.5)), 100 * float(np.percentile(bs, 97.5))],
                                    loose_pp=100 * float(dl.mean()), cores_pos=int((per > 0).sum()), cores_neg=int((per < 0).sum()), n_cores=len(cores))
        Bs = [r['B'] for r in R_ if 'B' in r]
        mech = dict(n=len(Bs), fallback=dict((f, sum(r.get('fallback') == f for r in R_)) for f in ('host_failed', 'support', 'solver')))
        for key in ('n_support', 'n_blocks', 'hull_area', 'support_err_rate_3', 'eff_rank', 'cond', 'J_fit', 'J_holdout'):
            v = [b[key] for b in Bs if b.get(key) is not None]
            if v:
                mech[key] = dict(median=float(np.median(v)), p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90)))
        it = [b['solver']['iterations'] for b in Bs if 'solver' in b]
        if it:
            mech['vcs_iterations_median'] = float(np.median(it)); mech['vcs_converged'] = float(np.mean([b['solver']['converged'] for b in Bs if 'solver' in b]))
            mech['bce_b_converged'] = float(np.mean([b['bce_b']['converged'] for b in Bs if 'bce_b' in b]))
        out[role] = dict(task=task, paired=pairs, mechanism=mech, n_problems=len(R_), n_cores=len(cores))
        if role in ('CAL',):
            A_ = [r['A'] for r in R_ if 'A' in r]
            agg = {}
            for which in ('oracle', 'real'):
                xs = [a_[which] for a_ in A_ if which in a_]
                if not xs:
                    continue
                tot = sum(x['n'] for x in xs)
                agg[which] = dict(n_problems=len(xs), correct_rate={t: sum(x[f'correct_{t}'] for x in xs) / max(tot, 1) for t in PROT['correct_px']},
                                  top2_cover_3=sum(x['top2_cover_3.0'] for x in xs) / max(tot, 1),
                                  new_only_strict=float(np.mean([(x['new_only_err'] or np.inf) < 9 for x in xs])),
                                  new_only_loose=float(np.mean([(x['new_only_err'] or np.inf) < 42 for x in xs])))
                if which == 'oracle':
                    agg[which]['host_plus_new_strict'] = float(np.mean([(x['host_plus_new_err'] or np.inf) < 9 for x in xs]))
            agg['host_same_angle_strict'] = float(np.mean([(a_['host_angle_err'] or np.inf) < 9 for a_ in A_]))
            out[role]['A'] = agg
    host_match = []
    a_rows = {(r['core'], r['perturbation']): r for r in HN.rows(SOTA / 'dev/minima_roma/log_r15.jsonl')}
    for r in rs:
        if r['role'] == 'DEV' and (r['core'], r['perturbation']) in a_rows and r['host'].get('err') is not None:
            host_match.append(abs(a_rows[(r['core'], r['perturbation'])]['err_full'] - r['host']['err']))
    if host_match:
        out['host_vs_A_rows'] = dict(n=len(host_match), identical=int(sum(x < 1e-9 for x in host_match)), max_abs_err_diff=float(max(host_match)),
                                     median_abs_err_diff=float(np.median(host_match)))
    out['gpu_seconds_problem_sum'] = float(sum(r.get('seconds', r.get('host_seconds', 0)) for r in rs))
    write(HERE / 'ANALYSIS.json', jsonable(out)); print(json.dumps(jsonable(out), indent=1)[:6000])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['cpucheck', 'fit', 'run', 'analyze'])
    ap.add_argument('--role', choices=['CAL', 'DEV']); ap.add_argument('--shard', type=int, default=0); ap.add_argument('--nshard', type=int, default=1)
    ap.add_argument('--limit', type=int, default=None); a = ap.parse_args()
    if a.mode == 'cpucheck':
        cmd_cpucheck()
    elif a.mode == 'fit':
        cmd_fit(a.limit)
    elif a.mode == 'run':
        cmd_run(a.role, a.shard, a.nshard, a.limit)
    else:
        cmd_analyze()
