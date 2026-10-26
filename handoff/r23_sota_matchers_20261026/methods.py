"""Shared registration-method interface for the R23 SOTA baselines (task A) and the CoMIR-protocol evaluation (task B).

    m = build(name, cfg, device);  out = m.register(ref, mov, ctx)  ->  dict(theta_deg, ty, tx, ok, n_matches, n_inliers, alpha, info)

ref: dict 'lum' (H,W) float (BF luminance, any scale), 'rgb' (H,W,3) float in [0,1] (raw BF RGB)
mov: dict 'int' (H,W) float (SHG intensity), 'log' (H,W) float (log1p SHG)
Convention (host Q.apply_T): mov(R(theta)(z - c) + c + t) ~ ref(z), z = (y, x), c = ((H-1)/2, (W-1)/2), R = [[cos, -sin], [sin, cos]]
acting on (y, x); units = pixels of the given grid; pixel centres at integer coordinates (host sample_pts, cv2).
ctx: dict(seed=int, theta_max=deg, t_max=px) - the search domain every method shares (host: 30 deg, 51 px at 512).
Matchers that are not rotation invariant get the same pre-registered rotation sweep of the moving image (IWM, Learn2Reg 2024);
a rigid fit (2-point RANSAC + least squares, no scale) inside the shared domain picks the hypothesis with most inliers.
"""
from __future__ import annotations

import math
import sys
import time
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

SRC = Path('/projects/EEG-foundation-model/yinghao/models/sota_src')
MODELS = Path('/projects/EEG-foundation-model/yinghao/models')
PREPROC = ('gray', 'log', 'inv', 'iwm')


# ------------------------------------------------------------------------------------------------------------ images
def u8(a, lo=0.5, hi=99.5):
    a = np.asarray(a, dtype=np.float64); p, q = np.percentile(a, [lo, hi])
    return np.clip((a - p) / max(q - p, 1e-9) * 255.0, 0, 255).astype(np.uint8)


def prep(ref, mov, mode):
    """(ref_u8, mov_u8) single-channel uint8 images for a preprocessing mode."""
    if mode == 'gray':
        return u8(ref['lum']), u8(mov['int'])
    if mode == 'log':
        return u8(ref['lum']), u8(mov['log'])
    if mode == 'inv':
        return 255 - u8(ref['lum']), u8(mov['int'])
    if mode == 'iwm':                        # Learn2Reg 2024 IWM: H&E hue channel + histogram equalisation + 5x5 median
        rgb = np.clip(np.asarray(ref['rgb']) * 255.0, 0, 255).astype(np.uint8)
        hue = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 0].astype(np.float64) * (255.0 / 179.0)
        a = cv2.medianBlur(cv2.equalizeHist(hue.astype(np.uint8)), 5)
        b = cv2.medianBlur(cv2.equalizeHist(u8(mov['int'])), 5)
        return a, b
    raise ValueError(mode)


def rot_img(img, alpha_deg):
    """img_rot(z) = img(R(alpha)(z - c) + c), bilinear, zero padding (same mapping as the host warp with t = 0)."""
    H, W = img.shape[:2]; cy, cx = (H - 1) / 2, (W - 1) / 2; a = math.radians(alpha_deg); c, s = math.cos(a), math.sin(a)
    # dst(x, y) = src(M [x, y, 1]); src_y = c (y-cy) - s (x-cx) + cy, src_x = s (y-cy) + c (x-cx) + cx
    M = np.array([[c, s, cx - c * cx - s * cy], [-s, c, cy + s * cx - c * cy]], dtype=np.float64)
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def rot_pts(q, alpha_deg, c):
    """Map points of the rotated image back to the original moving image: q_orig = R(alpha)(q - c) + c, (y, x)."""
    a = math.radians(alpha_deg); R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return (q - c) @ R.T + c


# ------------------------------------------------------------------------------------------------------------ rigid fit
def rigid_lsq(p, q, c):
    """Least-squares rigid (no scale) q - c = R (p - c) + t, (y, x).  Returns theta_deg, t (2,)."""
    P, Qq = p - c, q - c; mp, mq = P.mean(0), Qq.mean(0)
    Hm = (P - mp).T @ (Qq - mq); U, _, Vt = np.linalg.svd(Hm)
    D = np.diag([1.0, np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0]); R = Vt.T @ D @ U.T
    return math.degrees(math.atan2(R[1, 0], R[0, 0])), mq - R @ mp


def residuals(theta_deg, t, p, q, c):
    a = math.radians(theta_deg); R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return np.linalg.norm((p - c) @ R.T + t - (q - c), axis=1)


def ransac_rigid(p, q, c, seed, tau=3.0, iters=2000, theta_max=30.0, t_max=51.0, margin=1.05):
    """2-point RANSAC for a rigid map inside the shared domain (|theta| <= theta_max, |t| <= t_max, both x margin), then
    least-squares refits on the inliers.  Returns (theta_deg, t, n_inliers) or None."""
    n = len(p)
    if n < 2:
        return None
    rng = np.random.default_rng(seed); P, Qq = p - c, q - c
    i = rng.integers(0, n, size=(iters, 2)); i = i[i[:, 0] != i[:, 1]]
    if len(i) == 0:
        return None
    zp = (P[i[:, 1], 0] - P[i[:, 0], 0]) + 1j * (P[i[:, 1], 1] - P[i[:, 0], 1])          # y + i x: rotation multiplies by e^{i theta}
    zq = (Qq[i[:, 1], 0] - Qq[i[:, 0], 0]) + 1j * (Qq[i[:, 1], 1] - Qq[i[:, 0], 1])
    ok = (np.abs(zp) > 1e-6) & (np.abs(zq) > 1e-6)
    th = np.angle(zq / np.where(ok, zp, 1)); ct, st = np.cos(th), np.sin(th)
    p0, q0 = P[i[:, 0]], Qq[i[:, 0]]
    ty = q0[:, 0] - (ct * p0[:, 0] - st * p0[:, 1]); tx = q0[:, 1] - (st * p0[:, 0] + ct * p0[:, 1])
    ok &= (np.abs(np.degrees(th)) <= theta_max * margin) & (np.hypot(ty, tx) <= t_max * margin)
    if not ok.any():
        return None
    th, ct, st, ty, tx = th[ok], ct[ok], st[ok], ty[ok], tx[ok]
    ry = ct[:, None] * P[None, :, 0] - st[:, None] * P[None, :, 1] + ty[:, None] - Qq[None, :, 0]
    rx = st[:, None] * P[None, :, 0] + ct[:, None] * P[None, :, 1] + tx[:, None] - Qq[None, :, 1]
    inl = (ry * ry + rx * rx) < tau * tau; k = int(np.argmax(inl.sum(1))); mask = inl[k]
    best = (math.degrees(th[k]), np.array([ty[k], tx[k]]))
    for _ in range(3):
        if mask.sum() < 3:
            break
        th2, t2 = rigid_lsq(p[mask], q[mask], c)
        if abs(th2) > theta_max * margin or np.hypot(*t2) > t_max * margin:
            break
        best = (th2, t2); mask = residuals(th2, t2, p, q, c) < tau
    return best[0], best[1], int(mask.sum())


# ------------------------------------------------------------------------------------------------------------ base
class Method:
    rotation_invariant = False

    def __init__(self, cfg, device):
        self.cfg, self.device = dict(cfg), torch.device(device)

    def angles(self, ctx):
        if self.rotation_invariant or self.cfg.get('rot_step', 0) == 0:
            return [0.0]
        m, s = ctx.get('theta_max', 30.0), self.cfg['rot_step']
        return [float(a) for a in np.arange(-m, m + 1e-9, s)]

    def match(self, a, b):                    # uint8 images -> p (ref, y x), q (mov, y x), weights
        raise NotImplementedError

    def register(self, ref, mov, ctx):
        t0 = time.time(); a, b = prep(ref, mov, self.cfg.get('preproc', 'gray'))
        H, W = a.shape; c = np.array([(H - 1) / 2, (W - 1) / 2]); best = None; nm_best = 0
        for k, al in enumerate(self.angles(ctx)):
            br = b if al == 0.0 else rot_img(b, al)
            p, q = self.match(a, br)
            if len(p) == 0:
                continue
            q = rot_pts(q, al, c) if al != 0.0 else q
            r = ransac_rigid(p, q, c, seed=int(ctx.get('seed', 0)) * 1009 + k, tau=self.cfg.get('tau', 3.0),
                             theta_max=ctx.get('theta_max', 30.0), t_max=ctx.get('t_max', 51.0))
            if r is not None and (best is None or r[2] > best[2]):
                best = (r[0], r[1], r[2], al); nm_best = len(p)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        if best is None or best[2] < 3:
            return dict(theta_deg=0.0, ty=0.0, tx=0.0, ok=False, n_matches=nm_best, n_inliers=0 if best is None else best[2],
                        alpha=None, seconds=time.time() - t0)
        return dict(theta_deg=float(best[0]), ty=float(best[1][0]), tx=float(best[1][1]), ok=True, n_matches=nm_best,
                    n_inliers=best[2], alpha=best[3], seconds=time.time() - t0)


def _gray3(img, device):
    t = torch.from_numpy(np.ascontiguousarray(img)).float().div(255.0).to(device)
    return t[None].expand(3, -1, -1) if t.ndim == 2 else t.permute(2, 0, 1)


def _xy_to_yx(k):
    k = np.asarray(k, dtype=np.float64).reshape(-1, 2); return k[:, ::-1].copy()


# ------------------------------------------------------------------------------------------------------------ LightGlue family
class LightGlueMethod(Method):
    """SuperPoint / DISK / ALIKED + LightGlue (cvg/LightGlue); weights from the offline torch hub cache; optional MINIMA weights."""

    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        sys.path.insert(0, str(SRC / 'LightGlue'))
        from lightglue import ALIKED, DISK, LightGlue, SuperPoint
        feats = cfg['features']; ext = {'superpoint': SuperPoint, 'disk': DISK, 'aliked': ALIKED}[feats]
        self.ext = ext(max_num_keypoints=cfg.get('max_kp', 2048)).eval().to(self.device)
        if cfg.get('minima'):
            self.lg = LightGlue(features=None, input_dim=256, descriptor_dim=256, n_layers=9, num_heads=4,
                                filter_threshold=0.1).eval().to(self.device)
            sd = torch.load(MODELS / 'minima/minima_lightglue.pth', map_location='cpu')
            for i in range(9):
                sd = {k.replace(f'self_attn.{i}', f'transformers.{i}.self_attn').replace(f'cross_attn.{i}', f'transformers.{i}.cross_attn'): v
                      for k, v in sd.items()}
            missing, unexpected = self.lg.load_state_dict(sd, strict=False)
            # confidence_thresholds is a non-learned buffer of newer LightGlue code (computed from the config), absent from MINIMA's file
            assert set(missing) <= {'confidence_thresholds'}, ('MINIMA LightGlue weights do not cover the model', missing[:5])
            self.load_note = dict(unexpected=list(unexpected)[:10])
        else:
            self.lg = LightGlue(features=feats).eval().to(self.device)

    @torch.inference_mode()
    def match(self, a, b):
        from lightglue.utils import rbd
        fa = self.ext.extract(_gray3(a, self.device), resize=None); fb = self.ext.extract(_gray3(b, self.device), resize=None)
        m = self.lg({'image0': fa, 'image1': fb}); fa, fb, m = [rbd(x) for x in (fa, fb, m)]
        mm = m['matches']
        return _xy_to_yx(fa['keypoints'][mm[:, 0]].cpu().numpy()), _xy_to_yx(fb['keypoints'][mm[:, 1]].cpu().numpy())


# ------------------------------------------------------------------------------------------------------------ XFeat
class XFeatMethod(Method):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        sys.path.insert(0, str(SRC / 'accelerated_features'))
        from modules.xfeat import XFeat
        self.xf = XFeat(weights=str(MODELS / 'xfeat/xfeat.pt'), top_k=cfg.get('top_k', 4096))
        self.lighter = cfg.get('lighterglue', False)

    @torch.inference_mode()
    def match(self, a, b):
        ta = _gray3(a, self.xf.dev)[None]; tb = _gray3(b, self.xf.dev)[None]
        if not self.lighter:
            pa, pb = self.xf.match_xfeat(ta, tb)
        else:
            da = self.xf.detectAndCompute(ta, top_k=self.xf.top_k)[0]; db = self.xf.detectAndCompute(tb, top_k=self.xf.top_k)[0]
            da['image_size'] = (a.shape[1], a.shape[0]); db['image_size'] = (b.shape[1], b.shape[0])
            pa, pb, _ = self.xf.match_lighterglue(da, db)
        return _xy_to_yx(pa), _xy_to_yx(pb)


# ------------------------------------------------------------------------------------------------------------ LoFTR (MINIMA)
class LoFTRMethod(Method):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        sys.path.insert(0, str(SRC / 'MINIMA'))
        import types
        import kornia.utils
        try:
            import kornia.utils.grid  # noqa: F401  (removed in kornia 0.9; LoFTR_minima imports create_meshgrid from it)
        except ImportError:
            shim = types.ModuleType('kornia.utils.grid'); shim.create_meshgrid = kornia.utils.create_meshgrid; sys.modules['kornia.utils.grid'] = shim
        from third_party.LoFTR_minima.src.loftr import LoFTR, default_cfg
        c = deepcopy(default_cfg); c['coarse']['temp_bug_fix'] = True; c['match_coarse']['thr'] = cfg.get('thr', 0.2)
        self.net = LoFTR(config=c); self.net.load_state_dict(torch.load(MODELS / 'minima/minima_loftr.ckpt', map_location='cpu')['state_dict'], strict=True)
        self.net = self.net.eval().to(self.device)

    @torch.inference_mode()
    def match(self, a, b):
        H, W = a.shape; h8, w8 = H // 8 * 8, W // 8 * 8
        ta = torch.from_numpy(a[:h8, :w8]).float().div(255)[None, None].to(self.device)
        tb = torch.from_numpy(b[:h8, :w8]).float().div(255)[None, None].to(self.device)
        batch = {'image0': ta, 'image1': tb}; self.net(batch)
        return _xy_to_yx(batch['mkpts0_f'].cpu().numpy()), _xy_to_yx(batch['mkpts1_f'].cpu().numpy())


# ------------------------------------------------------------------------------------------------------------ RoMa / MINIMA-RoMa
class RoMaMethod(Method):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        sys.path.insert(0, str(SRC / 'MINIMA/third_party/RoMa_minima'))
        from romatch import roma_outdoor
        w = torch.load(MODELS / ('minima/minima_roma.pth' if cfg.get('minima') else 'roma/roma_outdoor.pth'), map_location=self.device)
        dw = torch.load(MODELS / 'roma/dinov2_vitl14_pretrain.pth', map_location=self.device)
        self.net = roma_outdoor(device=self.device, weights=w, dinov2_weights=dw)
        self.num = cfg.get('num', 5000)

    def match(self, a, b):
        from PIL import Image
        A = Image.fromarray(a).convert('RGB'); B = Image.fromarray(b).convert('RGB')
        with torch.inference_mode():
            warp, cert = self.net.match(A, B, batched=False, device=self.device)
            m, _ = self.net.sample(warp, cert, num=self.num)
            ka, kb = self.net.to_pixel_coordinates(m, a.shape[0], a.shape[1], b.shape[0], b.shape[1])
        # to_pixel_coordinates uses the pixel-corner convention (x in [0, W]); host/cv2 pixel centres are at integers
        return _xy_to_yx(ka.cpu().numpy() - 0.5), _xy_to_yx(kb.cpu().numpy() - 0.5)


# ------------------------------------------------------------------------------------------------------------ globalign (CMIF)
class GlobalignMethod(Method):
    """MIDA-group globalign: exhaustive MI over all whole-pixel translations (FFT, CMIF) x an angle grid, plus random refinement.
    Uniform quantisation of the preprocessed uint8 images into Q levels; the returned parameters are converted to the shared
    convention by mapping reference points with globalign.warp_points_rigid and a rigid least-squares fit."""
    rotation_invariant = True

    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        sys.path.insert(0, str(SRC / 'globalign')); import globalign
        self.G = globalign

    def register(self, ref, mov, ctx):
        t0 = time.time(); a, b = prep(ref, mov, self.cfg.get('preproc', 'gray')); Qn = self.cfg.get('Q', 16)
        A = np.minimum((a.astype(np.int64) * Qn) // 256, Qn - 1); B = np.minimum((b.astype(np.int64) * Qn) // 256, Qn - 1)
        M = np.ones(A.shape, dtype=bool); tm = ctx.get('theta_max', 30.0)
        param, _ = self.G.align_rigid_and_refine(A, B, M, M, Qn, Qn, self.cfg.get('angles_n', 61), tm,
                                                 refinement_param={'n': self.cfg.get('refine_n', 32), 'max_angle': 1.0},
                                                 overlap=self.cfg.get('overlap', 0.5), enable_partial_overlap=True, normalize_mi=False,
                                                 on_gpu=self.device.type == 'cuda', save_maps=False,
                                                 rng=np.random.default_rng(int(ctx.get('seed', 0))))
        H, W = a.shape; c = np.array([(H - 1) / 2, (W - 1) / 2])
        g = np.stack(np.meshgrid(np.linspace(0, H - 1, 9), np.linspace(0, W - 1, 9), indexing='ij'), -1).reshape(-1, 2)
        q = np.asarray(self.G.warp_points_rigid(g, param), dtype=np.float64)
        th, t = rigid_lsq(g, q, c)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        return dict(theta_deg=float(th), ty=float(t[0]), tx=float(t[1]), ok=True, n_matches=0, n_inliers=0, alpha=None,
                    seconds=time.time() - t0, info=dict(mi=float(param[0]), raw=[float(x) for x in param]))


# ------------------------------------------------------------------------------------------------------------ DINOv2 + host searcher
def dinov2(arch, device):
    sys.path.insert(0, str(SRC / 'dinov2'))
    from dinov2.hub.backbones import dinov2_vitb14, dinov2_vits14
    m = {'vits14': dinov2_vits14, 'vitb14': dinov2_vitb14}[arch](pretrained=False)
    m.load_state_dict(torch.load(MODELS / f'dinov2/dinov2_{arch}_pretrain.pth', map_location='cpu'), strict=True)
    for p in m.parameters():
        p.requires_grad_(False)
    return m.eval().to(device)


_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); _STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class DinoEnergy:
    """Host-compatible energy (Q.optimise_rigid: grid_values per grid angle, differentiable __call__(p), value) with frozen
    DINOv2 patch tokens as descriptors and the cosine readout: same 4096 disk positions (host rng), BOX1 warp-then-encode for
    the local path and BOX2 rotation-only encode + shifts for the grid path, as Q.RigidEnergy 'critic'."""

    def __init__(self, Q, vit, ref3, mov1, rng, scale):
        self.Q, self.vit, self.scale, torch_ = Q, vit, scale, torch
        dev = mov1.device; self.mov = mov1; P = Q.P
        r = Q.R_OMEGA * np.sqrt(rng.uniform(0.0, 1.0, P.N_POS)); ang = rng.uniform(0.0, 2 * math.pi, P.N_POS)
        self.x = torch.tensor(np.stack([Q.CEN + r * np.sin(ang), Q.CEN + r * np.cos(ang)], -1), dtype=torch.float32, device=dev)
        self.box1, self.box2 = Q.grid_coords(torch, *Q.BOX1, dev), Q.grid_coords(torch, *Q.BOX2, dev)
        with torch.no_grad():
            self.u = F.normalize(self.sample(self.encode(ref3[None]), self.x, ref3.shape[-1])[0], dim=-1)

    def encode(self, img3):                               # [B,3,n,n] in [0,1] -> tokens [B,C,h,w]
        n = img3.shape[-1]; m = max(14, int(round(n * self.scale / 14)) * 14)
        z = F.interpolate(img3, size=(m, m), mode='bilinear', align_corners=False)
        z = (z - _MEAN.to(z.device)) / _STD.to(z.device)
        t = self.vit.forward_features(z)['x_norm_patchtokens']; h = m // 14
        return t.reshape(img3.shape[0], h, h, -1).permute(0, 3, 1, 2)

    @staticmethod
    def sample(fmap, pts, n):                             # fmap [B,C,h,w] covering an n x n pixel extent; pts [B?,N,2] (y,x)
        pts = pts if pts.ndim == 3 else pts[None].expand(fmap.shape[0], -1, -1)
        g = torch.stack([(pts[..., 1] + 0.5) / n * 2 - 1, (pts[..., 0] + 0.5) / n * 2 - 1], -1)
        return F.grid_sample(fmap, g[:, None], mode='bilinear', align_corners=False)[:, :, 0].permute(0, 2, 1)

    def _cos(self, dv):
        return (self.u[None] * F.normalize(dv, dim=-1)).sum(-1).mean(-1)

    def batch(self, th, t):
        Q = self.Q; n = Q.BOX1[1] - Q.BOX1[0] + 1; out = []
        for k in range(len(th)):
            Wk = Q.sample_pts(torch, F, self.mov, Q.apply_T(torch, th[k:k + 1], t[k:k + 1], self.box1))[0, :, 0].reshape(1, 1, n, n)
            out.append(self._cos(self.sample(self.encode(Wk.expand(-1, 3, -1, -1)), self.x - Q.BOX1[0], n))[0])
        return torch.stack(out)

    def grid_values(self, th_deg, ts):
        Q = self.Q; dev = ts.device; th = torch.full((1,), math.radians(th_deg), device=dev); zero = torch.zeros(1, 2, device=dev)
        c, s = math.cos(math.radians(th_deg)), math.sin(math.radians(th_deg))
        sh = torch.stack([c * ts[:, 0] + s * ts[:, 1], -s * ts[:, 0] + c * ts[:, 1]], -1); n2 = Q.BOX2[1] - Q.BOX2[0] + 1
        W = Q.sample_pts(torch, F, self.mov, Q.apply_T(torch, th, zero, self.box2))[0, :, 0].reshape(1, 1, n2, n2)
        D = self.encode(W.expand(-1, 3, -1, -1)); out = []
        for b in sh.split(32):
            pts = self.x[None] + b[:, None] - Q.BOX2[0]
            out.append(self._cos(self.sample(D.expand(len(b), -1, -1, -1), pts, n2)))
        return torch.cat(out)

    def __call__(self, p):
        return self.batch(p[0:1] / self.Q.RHO, p[1:3][None])[0]

    def value(self, th_deg, t):
        with torch.no_grad():
            return float(self.batch(torch.tensor([math.radians(th_deg)], device=self.x.device),
                                    torch.tensor([list(t)], dtype=torch.float32, device=self.x.device))[0])


class DinoHostMethod(Method):
    """DINO-Reg-style training-free baseline: frozen DINOv2 tokens in the host energy with the unchanged host searcher
    (grid 21 x 508, top-3 starts, 100 Adam steps).  Defined only on the host 512 grid (host domain constants)."""
    rotation_invariant = True

    def __init__(self, cfg, device):
        super().__init__(cfg, device); self.vit = dinov2(cfg['arch'], self.device)

    def register(self, ref, mov, ctx):
        Q = ctx['Q']; t0 = time.time()
        a = torch.from_numpy(np.clip(np.asarray(ref['rgb'], dtype=np.float32), 0, 1)).permute(2, 0, 1).to(self.device) \
            if self.cfg.get('ref', 'rgb') == 'rgb' else torch.from_numpy(u8(ref['lum']).astype(np.float32) / 255).to(self.device)[None].expand(3, -1, -1)
        m = torch.from_numpy(u8(mov[self.cfg.get('mov', 'int')]).astype(np.float32) / 255).to(self.device)[None]
        e = DinoEnergy(Q, self.vit, a, m, ctx['rng'], self.cfg.get('scale', 1.0))
        th, t, n_grid, f = Q.optimise_rigid(torch, e, self.device)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        return dict(theta_deg=float(th), ty=float(t[0]), tx=float(t[1]), ok=True, n_matches=0, n_inliers=0, alpha=None,
                    seconds=time.time() - t0, info=dict(F_hat=float(f), n_grid=int(n_grid)))


# ------------------------------------------------------------------------------------------------------------ registry
FAMILIES = {
    'sp_lg': (LightGlueMethod, dict(features='superpoint')),
    'disk_lg': (LightGlueMethod, dict(features='disk')),
    'aliked_lg': (LightGlueMethod, dict(features='aliked')),
    'minima_sp_lg': (LightGlueMethod, dict(features='superpoint', minima=True)),
    'xfeat': (XFeatMethod, dict(lighterglue=False)),
    'xfeat_lg': (XFeatMethod, dict(lighterglue=True)),
    'minima_loftr': (LoFTRMethod, dict()),
    'roma': (RoMaMethod, dict(minima=False)),
    'minima_roma': (RoMaMethod, dict(minima=True)),
    'globalign': (GlobalignMethod, dict()),
    'dinov2_host': (DinoHostMethod, dict()),
}


def build(name, cfg, device):
    cls, base = FAMILIES[name]; c = dict(base); c.update(cfg); return cls(c, device)
