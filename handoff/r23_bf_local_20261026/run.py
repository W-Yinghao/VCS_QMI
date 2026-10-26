"""R23 BF-SHG: cross-modal registration under local structural inconsistency (2026-10-26, development, no TEST).

Same unit-cosine critic and within-crop product reference as handoff/r22_geometry_train_20261023/train.py; objectives vcs_exact /
nce / soft_nce / bce (learnable logit scale+bias) / rince on a shared descriptor that is either the existing single-scale encoder or
a dual-scale descriptor (same encoder on the crop and on a 2x average-pooled crop, sampled at the same physical point, concat ->
learnable Linear(32->16, no bias, init [I|I]) -> L2 normalize). Training: crop 256, 512 positions valid for both scales, n_sc 16,
10000 steps; 'aug' deforms the SHG crop locally with prob 0.5 per crop (separate RNG). Evaluation: host searcher (Q.optimise_rigid:
3 deg x 4 px grid, top-3, 100 Adam) on a common valid domain recomputed from the dual-scale receptive field, COS readout, conditions
clean / occlude / deform / sparse_only (+ auxiliary strengths) with instances fixed per (core, perturbation).
Frozen sources are imported, never edited: long-v2 run.py (context/config), geometry_train train.py (normalized/exact_j), host.
"""
import argparse, fcntl, hashlib, importlib.util, json, math, os, statistics, sys, time, types
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
_spec = importlib.util.spec_from_file_location('long_v2', BASE / 'handoff/r22_geometry_long_v2_20261024/run.py')
L = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(L)
T, R, A, torch = L.T, L.R, L.A, L.torch
F, Q, P = T.F, A.Q, A.P
nn = torch.nn
DS, SEED = 'bf', 1001
CROP, NPOS, NSC, STEPS = 256, 512, 16, 10000
INNER = 2 * (P.RF_HALF + 1) + 1            # 33: coarse coordinate (y-0.5)/2 stays in the host-valid [16, 111] of the pooled 128 crop
RF_BOX_DUAL = 2 * Q.RF_BOX + 2             # 34: pooled RF (2 x 31 + 1 px) plus the bilinear neighbour at the coarse scale (2 px)
LAMBDA_RINCE = 0.01
WRONG = (0.0, 8 / math.sqrt(2), 8 / math.sqrt(2))   # fixed wrong candidate for the region-wise support (v1 FAR ring, ~32 err_full)
OBJECTIVES = ('vcs_exact', 'nce', 'soft_nce', 'bce', 'rince')
CONDITIONS = {'clean': None, 'occlude': (8.0, .15), 'deform': (8.0, .15), 'sparse_only': (8.0, .15),
              'deform_4px_15': (4.0, .15), 'deform_8px_30': (8.0, .30)}
MAIN_CONDITIONS = ('clean', 'occlude', 'deform', 'sparse_only')
AUX_CONDITIONS = ('deform_4px_15', 'deform_8px_30')


def domain(rf_box):
    r = int(math.floor(Q.CEN - 2 * Q.TMAX - math.sqrt(2) - rf_box * math.sqrt(2)))
    b1 = (int(math.floor(Q.CEN - r - rf_box)), int(math.ceil(Q.CEN + r + rf_box)))
    b2 = (int(math.floor(Q.CEN - r - Q.TMAX - rf_box)), int(math.ceil(Q.CEN + r + Q.TMAX + rf_box)))
    return dict(R=r, box1=b1, box2=b2, rf_box=rf_box)


HOST_DOMAIN = domain(Q.RF_BOX)
NEW_DOMAIN = domain(RF_BOX_DUAL)
assert (HOST_DOMAIN['R'], HOST_DOMAIN['box1'], HOST_DOMAIN['box2']) == (Q.R_OMEGA, Q.BOX1, Q.BOX2)
assert all((d[b][1] - d[b][0] + 1) % 2 == 0 for d in (HOST_DOMAIN, NEW_DOMAIN) for b in ('box1', 'box2'))


# ------------------------------------------------------------------------------------------------------------ configurations
def cfg_name(c):
    parts = [c['scale'], c['train'], c['obj']] + [f"{k[0]}{c[k]:g}" for k in ('tau', 'sigma', 'q') if k in c] + [f"lr{c['lr']:g}"]
    return '_'.join(parts)


def cal_grid():
    g = [dict(obj='vcs_exact', lr=lr) for lr in (1e-4, 3e-4, 1e-3)]
    g += [dict(obj='nce', tau=t, lr=lr) for t in (.03, .07, .15, .30) for lr in (3e-4, 1e-3)]
    g += [dict(obj='soft_nce', sigma=s, tau=t, lr=3e-4) for s in (4., 8.) for t in (.07, .15)]
    g += [dict(obj='bce', lr=lr) for lr in (3e-4, 1e-3)]
    g += [dict(obj='rince', q=q, tau=t, lr=3e-4) for q in (.1, .5) for t in (.07, .15)]
    for c in g: c.update(scale='dual', train='aug')
    assert len(g) == 21 and len({cfg_name(c) for c in g}) == 21
    return g


FAMILY = {'vcs_exact': 'vcs', 'nce': 'nce_control', 'soft_nce': 'nce_control', 'bce': 'bce', 'rince': 'rince'}


def lr_mult(step):          # 1-based update index
    return 1.0 if step <= 5000 else (0.1 if step <= 8000 else 0.01)


def verify():
    for rel, h in json.loads((HERE / 'FREEZE.json').read_text())['files'].items():
        assert R.sha(BASE / rel) == h, rel
    return R.sha(HERE / 'FREEZE.json')


# ------------------------------------------------------------------------------------------------------------ objectives
def objective(u, v, c, yx=None, ab=None):
    """u, v [S, N, d] raw descriptors (normalized here), yx [S, N, 2] positions (512-scale px) -> loss, diagnostics."""
    u, v = T.normalized(u, v)
    kind = c['obj']
    if kind == 'vcs_exact':
        j = T.exact_j(u, v).mean(); return -j, dict(J=float(j.detach()))
    s = u @ v.transpose(-1, -2)
    n = s.shape[-1]
    if kind == 'nce':
        z = s / c['tau']; lab = torch.arange(n, device=s.device).expand(s.shape[:2]).reshape(-1)
        return .5 * (F.cross_entropy(z.reshape(-1, n), lab) + F.cross_entropy(z.transpose(1, 2).reshape(-1, n), lab)), {}
    if kind == 'soft_nce':
        d2 = (yx[:, :, None, :] - yx[:, None, :, :]).square().sum(-1)
        qt = torch.softmax(-d2 / (2 * c['sigma'] ** 2), -1)                       # symmetric distance kernel, row-normalized
        z = s / c['tau']
        rows = -(qt * torch.log_softmax(z, -1)).sum(-1).mean()
        cols = -(qt.transpose(1, 2) * torch.log_softmax(z.transpose(1, 2), -1)).sum(-1).mean()
        return .5 * (rows + cols), {}
    eye = torch.eye(n, dtype=torch.bool, device=s.device)
    if kind == 'bce':
        logit = ab[0] * s + ab[1]
        loss = F.softplus(-logit.diagonal(dim1=-2, dim2=-1)).mean() + F.softplus(logit[:, ~eye]).mean()
        return loss, dict(a=float(ab[0].detach()), b=float(ab[1].detach()))
    if kind == 'rince':
        z = s / c['tau']; q = c['q']; pos = z.diagonal(dim1=-2, dim2=-1)

        def side(zz):           # RINCE (Chuang et al. 2022): -e^{q s+}/q + (lambda * sum_j e^{s_j})^q / q, sum over positive + negatives
            return (-torch.exp(q * pos) / q + torch.exp(q * (math.log(LAMBDA_RINCE) + torch.logsumexp(zz, -1))) / q).mean()
        return .5 * (side(z) + side(z.transpose(1, 2))), {}
    raise ValueError(kind)


# ------------------------------------------------------------------------------------------------------------ descriptors
class Desc(nn.Module):
    """Existing encoder (single scale) or the shared dual-scale descriptor; sample() returns raw (unnormalized) descriptors."""
    def __init__(self, enc, dual, d=16):
        super().__init__()
        self.enc, self.dual = enc, bool(dual)
        if self.dual:
            self.fuse = nn.Linear(2 * d, d, bias=False)
            with torch.no_grad():
                self.fuse.weight.copy_(torch.cat([torch.eye(d), torch.eye(d)], 1).to(self.fuse.weight))

    def maps(self, img):                               # img [B, C, H, W] -> (fine [B, d, H, W], coarse [B, d, H/2, W/2] | None)
        return self.enc(img), (self.enc(F.avg_pool2d(img, 2)) if self.dual else None)

    def sample(self, fine, coarse, yx):                # fine [d, H, W], yx [N, 2] or [K, N, 2] (y, x pixel centres)
        flat = yx.reshape(-1, 2)
        a = P.sample_map(torch, F, fine, flat)
        if not self.dual:
            return a.reshape(*yx.shape[:-1], -1)
        b = P.sample_map(torch, F, coarse, (flat - .5) / 2)  # pooled pixel j covers fine 2j, 2j+1 (centre 2j + 0.5)
        return self.fuse(torch.cat([a, b], -1)).reshape(*yx.shape[:-1], -1)

    def batch(self, crops, yx):                        # host batch_descriptors order: encode all crops, sample per crop
        fine, coarse = self.maps(crops)
        yx = torch.tensor(yx, dtype=torch.float32, device=fine.device).to(fine.dtype)
        return torch.stack([self.sample(fine[s], None if coarse is None else coarse[s], yx[s]) for s in range(len(yx))])


class Model:
    """Encoders from the host init (GeometryTrainer, seed 1001 = long-v2 init) wrapped as descriptors; optional BCE logit (a, b)."""
    def __init__(self, c, fit, imgs, device, seed=SEED):
        cfg = A.TrainConfig(method='MLP_perm1', seed=seed, encoder_lr=c['lr'], head_lr=c['lr'], n_sc=NSC, n_pos=NPOS, crop=CROP, grad_spot_every=0)
        self.t = T.GeometryTrainer(cfg, imgs, fit, device, 'vcs_exact')
        self.initial_encoder_sha256 = A.state_hash(self.t.eU, self.t.eV)
        self.t.sampler.inner = INNER
        dual = c['scale'] == 'dual'
        self.dU, self.dV = Desc(self.t.eU, dual).to(device), Desc(self.t.eV, dual).to(device)
        self.ab = nn.Parameter(torch.tensor([2.0, 0.0], device=device)) if c['obj'] == 'bce' else None
        main = list(self.dU.parameters()) + list(self.dV.parameters())
        groups = [dict(params=main, lr=c['lr'], weight_decay=cfg.weight_decay)]
        if self.ab is not None: groups.append(dict(params=[self.ab], lr=c['lr'], weight_decay=0.0))
        self.opt = torch.optim.AdamW(groups)
        self.params = main + ([self.ab] if self.ab is not None else [])
        self.c, self.device, self.n_steps = c, torch.device(device), 0
        self.aug = np.random.default_rng(Q.seed_of(f'R23-AUG|{DS}|{seed}'))

    def state(self):
        return dict(dU=self.dU.state_dict(), dV=self.dV.state_dict(), ab=None if self.ab is None else self.ab.detach().cpu())

    def load(self, st):
        self.dU.load_state_dict(st['dU']); self.dV.load_state_dict(st['dV'])
        if self.ab is not None: self.ab.data.copy_(st['ab'].to(self.ab))

    def digest(self):
        h = hashlib.sha256()
        for m in (self.dU, self.dV):
            for k, v in sorted(m.state_dict().items()): h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
        if self.ab is not None: h.update(self.ab.detach().cpu().numpy().tobytes())
        return h.hexdigest()


# ------------------------------------------------------------------------------------------------------------ local deformation
def make_regions(rng, area, frame, size=None, radius=None):
    """1 or 2 (prob .5) non-overlapping disks of total area `area`; frame 'disk': inside the valid disk (centre CEN, radius `radius`);
    frame 'crop': inside a size x size crop. Returns dict(r, centers [n,2] (y,x), dirs [n,2] unit)."""
    n = 1 if rng.uniform() < .5 else 2
    r = math.sqrt(area / (n * math.pi))
    for _ in range(10000):
        if frame == 'disk':
            rr = (radius - r) * np.sqrt(rng.uniform(0, 1, n)); aa = rng.uniform(0, 2 * math.pi, n)
            cs = np.stack([Q.CEN + rr * np.sin(aa), Q.CEN + rr * np.cos(aa)], -1)
        else:
            cs = rng.uniform(r, size - 1 - r, (n, 2))
        if n == 1 or np.hypot(*(cs[0] - cs[1])) >= 2 * r: break
    else:
        raise RuntimeError('region placement failed')
    ang = rng.uniform(0, 2 * math.pi, n)
    return dict(r=float(r), centers=cs.tolist(), dirs=np.stack([np.sin(ang), np.cos(ang)], -1).tolist())


def displacement(pts, reg, amp):
    d = torch.zeros_like(pts)
    for c, e in zip(reg['centers'], reg['dirs']):
        rho = (pts - torch.tensor(c, dtype=pts.dtype, device=pts.device)).norm(dim=-1) / reg['r']
        w = torch.where(rho < 1, .5 * (1 + torch.cos(math.pi * rho.clamp(max=1))), torch.zeros_like(rho))
        d = d + amp * w[..., None] * torch.tensor(e, dtype=pts.dtype, device=pts.device)
    return d


def deform_image(M, reg, amp, iters=20):
    """V o phi^-1 with phi(x) = x + d(x): fixed-point inverse (contraction: amp*pi/(2r) < 1), bilinear, border padding. M [C, H, W]."""
    assert amp * math.pi / (2 * reg['r']) < 1
    H = M.shape[-1]; assert M.shape[-2] == H
    y = Q.grid_coords(torch, 0, H - 1, M.device).to(M.dtype); x = y.clone()
    for _ in range(iters): x = y - displacement(x, reg, amp)
    return Q.sample_pts(torch, F, M, x[None], 'border')[0].T.reshape(M.shape)


def region_mask(pts, reg):
    m = torch.zeros(pts.shape[:-1], dtype=torch.bool, device=pts.device)
    for c in reg['centers']:
        m |= (pts - torch.tensor(c, dtype=pts.dtype, device=pts.device)).norm(dim=-1) < reg['r']
    return m


def min_jacobian(reg, amp, size, device='cpu'):
    y = Q.grid_coords(torch, 0, size - 1, device).double(); d = displacement(y, reg, amp).reshape(size, size, 2)
    dyy, dyx = torch.gradient(d[..., 0]); dxy, dxx = torch.gradient(d[..., 1])
    return float(((1 + dyy) * (1 + dxx) - dyx * dxy).min())


def eval_regions(core, i, cond):
    amp, frac = CONDITIONS[cond]
    rng = np.random.default_rng(Q.seed_of(f'R23-DEF|{core}|{i}'))
    return make_regions(rng, frac * math.pi * NEW_DOMAIN['R'] ** 2, 'disk', radius=NEW_DOMAIN['R']), amp


class CondInputs(A.GuardedInputs):
    """Host GuardedInputs (moving V o T*^-1, learned log1p path) with the evaluation condition applied to V in its native frame.
    clean: the host path. deform: V o phi^-1 then the host perturbation and robust norm. occlude / sparse_only: the clean normalized
    moving image with the instance regions (occlude) or their complement (sparse_only) set to 0 = the robust median, the region
    mask carried through the same perturbation (zeros padding); normalization statistics are the clean ones (a filled image has
    a degenerate IQR)."""
    def __init__(self, guard, roles, purpose, pert, device, cond):
        super().__init__(guard, roles, purpose, pert, device); self.cond = cond; self.regions = {}

    def moving(self, c, i):
        if self.cond == 'clean': return super().moving(c, i)
        if (c, i) not in self.mov:
            self.guard.check(c, self.allowed, self.purpose + f':moving#{i}:{self.cond}')
            reg, amp = eval_regions(c, i, self.cond); self.regions[(c, i)] = reg
            sh = torch.tensor(np.expm1(P.load_img(self.root, c, ('shg_log',))[0].astype(np.float64)), dtype=torch.float32, device=self.dev)[None]
            th, ty, tx = self.pert[c][i]
            if self.cond.startswith('deform'):
                shp = Q.perturb_map(torch, self.F, deform_image(sh, reg, amp), th, (ty, tx))[0].cpu().numpy().astype(np.float64)
                out = torch.tensor(P.robust_norm(np.log1p(shp)), device=self.dev)[None]
            else:
                shp = Q.perturb_map(torch, self.F, sh, th, (ty, tx))[0].cpu().numpy().astype(np.float64)
                clean = torch.tensor(P.robust_norm(np.log1p(shp)), device=self.dev)[None]
                pts = Q.grid_coords(torch, 0, Q.S - 1, self.dev)
                mask = region_mask(pts, reg).float().reshape(1, Q.S, Q.S)
                mp = Q.perturb_map(torch, self.F, mask, th, (ty, tx), padding='zeros').clamp(0, 1)
                out = clean * (1 - mp) if self.cond == 'occlude' else clean * mp
            self.mov[(c, i)] = (out, out)
        return self.mov[(c, i)]


# ------------------------------------------------------------------------------------------------------------ energy / search
class Energy:
    """Host Q.RigidEnergy('critic') + COS readout on a parameterized domain and descriptor. Positions use the host rng calls."""
    def __init__(self, dU, dV, ref_maps, mov, rng, dom):
        self.dV, self.mov, self.dom = dV, mov, dom; dev = mov.device
        r = dom['R'] * np.sqrt(rng.uniform(0.0, 1.0, P.N_POS)); ang = rng.uniform(0.0, 2 * math.pi, P.N_POS)
        self.x = torch.tensor(np.stack([Q.CEN + r * np.sin(ang), Q.CEN + r * np.cos(ang)], -1), dtype=torch.float32, device=dev)
        rng.permutation(P.N_POS)                     # host stream (unused by COS)
        self.u = dU.sample(ref_maps[0], ref_maps[1], self.x)
        self.box1, self.box2 = Q.grid_coords(torch, *dom['box1'], dev), Q.grid_coords(torch, *dom['box2'], dev)

    def cos(self, dv):
        u, v = T.normalized(self.u[None], dv); return (u * v).sum(-1).mean(-1)

    def local_desc(self, th, t):                     # one candidate: warp into box1, encode, sample at x
        n = self.dom['box1'][1] - self.dom['box1'][0] + 1
        W = Q.sample_pts(torch, F, self.mov, Q.apply_T(torch, th, t, self.box1))[0, :, 0].reshape(1, 1, n, n)
        f, c = self.dV.maps(W)
        return self.dV.sample(f[0], None if c is None else c[0], self.x - self.dom['box1'][0])

    def batch(self, th, t):
        return torch.stack([self.cos(self.local_desc(th[k:k + 1], t[k:k + 1])[None])[0] for k in range(len(th))])

    def grid_values(self, th_deg, ts):
        dev = ts.device; th = torch.full((1,), math.radians(th_deg), device=dev); zero = torch.zeros(1, 2, device=dev)
        cc, ss = math.cos(math.radians(th_deg)), math.sin(math.radians(th_deg))
        sh = torch.stack([cc * ts[:, 0] + ss * ts[:, 1], -ss * ts[:, 0] + cc * ts[:, 1]], -1)
        n = self.dom['box2'][1] - self.dom['box2'][0] + 1
        W = Q.sample_pts(torch, F, self.mov, Q.apply_T(torch, th, zero, self.box2))[0, :, 0].reshape(1, 1, n, n)
        f, c = self.dV.maps(W); f, c = f[0], None if c is None else c[0]
        return torch.cat([self.cos(self.dV.sample(f, c, self.x[None] + b[:, None] - self.dom['box2'][0])) for b in sh.split(Q.CHUNK['critic'])])

    def __call__(self, p):
        return self.batch(p[0:1] / Q.RHO, p[1:3][None])[0]

    def value(self, th_deg, t):
        with torch.no_grad():
            return float(self.batch(torch.tensor([math.radians(th_deg)], device=self.x.device), torch.tensor([list(t)], dtype=torch.float32, device=self.x.device))[0])

    def position_scores(self, pose):                 # per-position cos at pose (deg, ty, tx) [N]
        with torch.no_grad():
            dv = self.local_desc(torch.tensor([math.radians(pose[0])], device=self.x.device), torch.tensor([[pose[1], pose[2]]], dtype=torch.float32, device=self.x.device))
            u, v = T.normalized(self.u, dv); return (u * v).sum(-1)


class Session:
    def __init__(self, dU, dV, guard, role, cores, device, cond, dom):
        self.dU, self.dV, self.dom, self.device, self.core = dU, dV, dom, torch.device(device), None
        self.perturbations = {}
        for c in cores:
            guard.log_rule(c, {role}, f'R23 BF {role} frozen perturbation rule'); self.perturbations[c] = Q.perturbations_p2(c)
        self.inputs = CondInputs(guard, {role}, f'R23 BF {role} registration', self.perturbations, self.device, cond)

    def energy(self, core, i):
        if self.core != core:
            self.inputs.ref.clear(); self.inputs.mov.clear()
            with torch.no_grad():
                f, c = self.dU.maps(self.inputs.reference(core)[1][None].float())
            self.ref = (f[0], None if c is None else c[0]); self.core = core
        rng = np.random.default_rng(Q.seed_of(f'R6-REG-P2|pos|{core}|{i}'))
        e = Energy(self.dU, self.dV, self.ref, self.inputs.moving(core, i)[1], rng, self.dom)
        self.inputs.mov.pop((core, i), None)
        return e


class frozen:
    def __init__(self, *mods): self.mods = mods
    def __enter__(self):
        self.params = [p for m in self.mods for p in m.parameters()]; self.flags = [p.requires_grad for p in self.params]
        self.before = [p.detach().clone() for p in self.params]
        for p in self.params: p.requires_grad_(False)
    def __exit__(self, *a):
        for p, f in zip(self.params, self.flags): p.requires_grad_(f)
        assert all(torch.equal(p.detach(), b) for p, b in zip(self.params, self.before)), 'evaluation changed the model'


def evaluate(dU, dV, guard, role, cond, out, dom=NEW_DOMAIN, cores=None, n_pert=10):
    """Full registration of every (core, perturbation) of `role` under `cond`; rows appended (finished rows kept)."""
    path = out / f'{role}_{cond}.jsonl'; cores = sorted(guard.included(role)) if cores is None else cores
    expected = [(c, i) for c in cores for i in range(n_pert)]
    done = {(r['core'], r['perturbation']): r for r in R.rows(path)}
    assert all(k in expected for k in done)
    with frozen(dU, dV):
        s = Session(dU, dV, guard, role, cores, 'cuda' if torch.cuda.is_available() else 'cpu', cond, dom)
        for core, i in expected:
            if (core, i) in done: continue
            e = s.energy(core, i); truth = s.perturbations[core][i]
            rec = R.plain(A.registration_record(e, str(e.x.device), truth))
            assert all(math.isfinite(rec[k]) for k in ('err_full', 'F_true', 'F_hat'))
            reg = s.inputs.regions.get((core, i)) or eval_regions(core, i, 'deform')[0]      # clean: the main instance's regions
            inside = region_mask(e.x, reg)
            st, sw = e.position_scores(truth), e.position_scores((truth[0] + WRONG[0], truth[1] + WRONG[1], truth[2] + WRONG[2]))
            dlt = st - sw
            row = dict(core=core, scene=core, perturbation=i, role=role, condition=cond, E=rec, n_region_positions=int(inside.sum()),
                       delta_region=float(dlt[inside].mean()) if inside.any() else None, delta_outside=float(dlt[~inside].mean()),
                       delta_all=float(dlt.mean()), regions=reg)
            R.append(path, row); done[(core, i)] = row; del e
    return summary([done[k] for k in expected])


def summary(rows):
    by = defaultdict(list)
    for r in rows: by[r['scene']].append(r['E']['err_full'])
    errs = [r['E']['err_full'] for r in rows]
    return dict(n=len(rows), success9=statistics.mean(sum(e < 9 for e in v) / len(v) for v in by.values()),
                success42=statistics.mean(sum(e < 42 for e in v) / len(v) for v in by.values()),
                median_err=float(np.median(errs)), p90_err=float(np.percentile(errs, 90)), p95_err=float(np.percentile(errs, 95)),
                energy_prefers_wrong=sum(r['E']['err_full'] >= 9 and r['E']['F_hat'] >= r['E']['F_true'] for r in rows),
                fail9=sum(e >= 9 for e in errs),
                delta_region=float(np.mean([r['delta_region'] for r in rows if r['delta_region'] is not None])),
                delta_outside=float(np.mean([r['delta_outside'] for r in rows])))


# ------------------------------------------------------------------------------------------------------------ training
def train_step(m):
    lr = m.c['lr'] * lr_mult(m.n_steps + 1)
    for g in m.opt.param_groups: g['lr'] = lr
    t = m.t; hb = t.sampler.draw(); bu, bv = A.assemble_crops(t.imgs, hb, CROP); bu, bv = bu.float(), bv.float()
    start = time.perf_counter(); nd = 0
    if m.c['train'] == 'aug':
        for s in range(len(bv)):
            if m.aug.uniform() < .5:
                reg = make_regions(m.aug, .15 * CROP * CROP, 'crop', size=CROP); bv[s] = deform_image(bv[s], reg, 8.0); nd += 1
    du, dv = m.dU.batch(bu, hb.yx), m.dV.batch(bv, hb.yx)
    yx = torch.tensor(hb.yx, dtype=torch.float32, device=du.device)
    loss, extra = objective(du, dv, m.c, yx, m.ab)
    m.opt.zero_grad(set_to_none=True); loss.backward()
    gn = float(torch.nn.utils.clip_grad_norm_(m.params, 1.0)); m.opt.step(); m.n_steps += 1
    if m.device.type == 'cuda': torch.cuda.synchronize()
    row = dict(step=m.n_steps, lr=lr, loss=float(loss.detach()), grad_norm_preclip=gn, n_deformed=nd, batch_sha256=L.batch_sha(hb),
               seconds=time.perf_counter() - start, **extra)
    assert hb.step == m.n_steps and math.isfinite(row['loss']) and math.isfinite(gn)
    return row


def lock():
    p = HERE / 'LOCK.json'; return json.loads(p.read_text()) if p.exists() else None


def pilot_configs(lk):
    sel = {f: dict(lk['selected'][f]['config']) for f in ('vcs', 'nce_control', 'bce', 'rince')}
    new = {'P1': dict(sel['vcs'], scale='fine'), 'P2': dict(sel['nce_control'], scale='fine'),
           'P7': dict(sel['vcs'], train='clean'), 'P8': dict(sel['nce_control'], train='clean')}
    return dict(P3=sel['vcs'], P4=sel['nce_control'], P5=sel['bce'], P6=sel['rince'], **new)


def find_config(name):
    for c in cal_grid():
        if cfg_name(c) == name: return c, 'cal'
    lk = lock()
    if lk:
        for k, c in pilot_configs(lk).items():
            if cfg_name(c) == name and k in ('P1', 'P2', 'P7', 'P8'): return c, k
    raise SystemExit(f'Unknown or not-yet-locked configuration {name}')


def run_train(name, dev=False, smoke_steps=0):
    freeze = verify(); cpu = json.loads((HERE / 'CPU_CHECK.json').read_text()); assert cpu['status'] == 'PASS' and cpu['freeze_sha256'] == freeze
    assert torch.cuda.is_available() and os.environ.get('SLURM_JOB_ID'); torch.set_num_threads(4)
    c, kind = find_config(name)
    if dev: assert kind in ('P1', 'P2', 'P7', 'P8') and lock() is not None, 'DEV only for locked new pilots'
    out = HERE / ('smoke' if smoke_steps else 'runs') / name; out.mkdir(parents=True, exist_ok=True)
    job = os.environ['SLURM_JOB_ID']; started = time.time(); ident = dict(name=name, config=c, kind=kind, seed=SEED, freeze_sha256=freeze)
    with (out / 'run.lock').open('a') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (out / 'COMPLETE.json').exists() and not list(out.glob('ATTEMPT_*.json')), 'No silent rerun or retry'
        R.write(out / f'ATTEMPT_{job}.json', dict(status='RUNNING', identity=ident, start=started, environment=R.environment()))
        u, guard, H, fit, imgs = L.context(DS, SEED, 'cuda', images=True)
        m = Model(c, fit, imgs, 'cuda')
        orig = json.loads((BASE / f'handoff/r22_geometry_train_20261023/runs/{DS}_vcs_exact_{SEED}/COMPLETE.json').read_text())
        assert m.initial_encoder_sha256 == orig['initial_encoder_sha256'], 'encoder init differs from the long-v2 init'
        steps = smoke_steps or STEPS
        with (out / 'TRAIN.jsonl').open('a') as tr:
            while m.n_steps < steps:
                row = train_step(m); tr.write(json.dumps(R.plain(row), allow_nan=False) + '\n')
                if m.n_steps % 500 == 0: tr.flush(); print(name, m.n_steps, row['loss'], flush=True)
        torch.save(dict(identity=ident, step=m.n_steps, state=m.state(), digest=m.digest()), out / f'step_{m.n_steps}.pt')
        rows = R.rows(out / 'TRAIN.jsonl'); assert [r['step'] for r in rows] == list(range(1, steps + 1))
        result = dict(status='COMPLETE', identity=ident, job=job, steps=m.n_steps, gpu=torch.cuda.get_device_name(), environment=R.environment(),
                      initial_encoder_sha256=m.initial_encoder_sha256, model_digest=m.digest(), checkpoint_sha256=R.sha(out / f'step_{m.n_steps}.pt'),
                      train_seconds=sum(r['seconds'] for r in rows), mean_step_ms=1000 * statistics.mean(r['seconds'] for r in rows[10:]),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(), train_sha256=R.sha(out / 'TRAIN.jsonl'))
        if smoke_steps:
            cal = sorted(guard.included('CAL'))[:1]; ev = {}
            for cond in ('clean', 'deform'):
                t0 = time.time(); ev[cond] = evaluate(m.dU, m.dV, guard, 'CAL', cond, out, cores=cal, n_pert=1); ev[cond]['seconds'] = time.time() - t0
            result['smoke_eval'] = ev
        else:
            result['cal'] = {cond: evaluate(m.dU, m.dV, guard, 'CAL', cond, out) for cond in ('deform', 'clean')}
            if dev:
                result['dev'] = {cond: evaluate(m.dU, m.dV, guard, 'DEV', cond, out) for cond in MAIN_CONDITIONS}
        assert {r['role'] for r in guard.reads} <= ({'FIT', 'CAL', 'DEV'} if dev else {'FIT', 'CAL'})
        result.update(roles_read=sorted({r['role'] for r in guard.reads}), seconds=time.time() - started)
        R.write(out / 'COMPLETE.json', result); R.write(out / f'ATTEMPT_{job}.json', dict(status='COMPLETE', identity=ident, seconds=result['seconds']))
        print('R23_COMPLETE', name, json.dumps({k: v for k, v in result.items() if k in ('cal', 'dev', 'smoke_eval')}), flush=True)


def load_model(target, device='cuda'):
    """target: a run name (this package, step_10000.pt) or ref_vcs_exact / ref_nce (long-v2 30k seed-1001)."""
    if target.startswith('ref_'):
        arm = target[4:]; d = BASE / f'handoff/r22_geometry_long_v2_20261024/runs/{DS}_{arm}_{SEED}'; done = json.loads((d / 'COMPLETE.json').read_text())
        assert done['status'] == 'COMPLETE' and done['final_steps'] == 30000
        u, guard, H, fit, _ = L.context(DS, SEED, device, images=False)
        t = T.GeometryTrainer(L.config(SEED), {}, fit, device, arm); R.CP.load(t, d / 'checkpoints/step_30000.pt', done['identity'])
        assert R.sha(d / 'checkpoints/step_30000.pt') == done['final_checkpoint_sha256']
        return Desc(t.eU, False), Desc(t.eV, False), guard, dict(reference=target, checkpoint_sha256=done['final_checkpoint_sha256'])
    c, _ = find_config(target); d = HERE / 'runs' / target; done = json.loads((d / 'COMPLETE.json').read_text()); assert done['status'] == 'COMPLETE'
    u, guard, H, fit, _ = L.context(DS, SEED, device, images=False)
    m = Model(c, fit, {}, device); st = torch.load(d / 'step_10000.pt', map_location='cpu', weights_only=True); m.load(st['state'])
    assert m.digest() == done['model_digest'] == st['digest']
    return m.dU, m.dV, guard, dict(run=target, model_digest=done['model_digest'])


def run_eval(target, aux=False):
    freeze = verify(); assert lock() is not None, 'DEV evaluation only after LOCK.json'
    assert torch.cuda.is_available() and os.environ.get('SLURM_JOB_ID'); torch.set_num_threads(4)
    out = HERE / 'evals' / target; out.mkdir(parents=True, exist_ok=True); job = os.environ['SLURM_JOB_ID']; started = time.time()
    with (out / 'eval.lock').open('a') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
        R.write(out / f'ATTEMPT_{job}.json', dict(status='RUNNING', target=target, aux=aux, start=started))
        dU, dV, guard, src = load_model(target)
        conds = MAIN_CONDITIONS + (AUX_CONDITIONS if aux else ())
        res = {cond: evaluate(dU, dV, guard, 'DEV', cond, out) for cond in conds}
        assert {r['role'] for r in guard.reads} <= {'FIT', 'CAL', 'DEV'}
        R.write(out / f'COMPLETE_{"aux" if aux else "main"}.json', dict(status='COMPLETE', target=target, source=src, freeze_sha256=freeze, job=job,
                gpu=torch.cuda.get_device_name(), dev=res, seconds=time.time() - started, domain=NEW_DOMAIN))
        R.write(out / f'ATTEMPT_{job}.json', dict(status='COMPLETE', seconds=time.time() - started))
        print('R23_EVAL_COMPLETE', target, json.dumps(res), flush=True)


def select():
    """Pre-registered per-family selection on CAL (deform strict, deform median, clean strict, smaller LR). No DEV read."""
    assert lock() is None, 'LOCK.json already written'
    res, missing = {}, []
    for c in cal_grid():
        p = HERE / 'runs' / cfg_name(c) / 'COMPLETE.json'
        if not p.exists(): missing.append(cfg_name(c)); continue
        d = json.loads(p.read_text()); assert d['status'] == 'COMPLETE' and 'dev' not in d; res[cfg_name(c)] = (c, d['cal'])
    if missing: print('WAITING', len(res), '/ 21; missing', missing); return
    key = lambda kv: (kv[1][1]['deform']['success9'], -kv[1][1]['deform']['median_err'], kv[1][1]['clean']['success9'], -kv[1][0]['lr'])
    sel = {}
    for fam in ('vcs', 'nce_control', 'bce', 'rince'):
        name, (c, cal) = max(((n, v) for n, v in res.items() if FAMILY[v[0]['obj']] == fam), key=key)
        sel[fam] = dict(name=name, config=c, cal=cal)
    lk = dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), freeze_sha256=verify(), rule='per family max (CAL deform success9, -CAL deform median, CAL clean success9, -lr); nce_control over nce and soft_nce',
              selected=sel, candidates={n: v[1] for n, v in res.items()})
    R.write(HERE / 'LOCK.json', lk)
    pc = pilot_configs(lk); sb = str(HERE / 'gpu.sbatch')
    lines = [f"sbatch --job-name=r23_pilot_{k} --time=08:00:00 {sb} train --name {cfg_name(pc[k])} --dev" for k in ('P1', 'P2', 'P7', 'P8')]
    (HERE / 'LINES_PILOT.txt').write_text('\n'.join(lines) + '\n')
    ev = [f"sbatch --job-name=r23_eval_{k} --time=10:00:00 {sb} eval --target {cfg_name(pc[k])}" + (' --aux' if k in ('P3', 'P4') else '') for k in ('P3', 'P4', 'P5', 'P6')]
    ev += [f"sbatch --job-name=r23_eval_{t} --time=08:00:00 {sb} eval --target {t}" for t in ('ref_vcs_exact', 'ref_nce')]
    (HERE / 'LINES_EVAL.txt').write_text('\n'.join(ev) + '\n')
    print('LOCKED', json.dumps({k: v['name'] for k, v in sel.items()}))


def analyze():
    lk = lock(); assert lk, 'no LOCK.json'; pc = pilot_configs(lk); res, missing = {}, []
    for k, c in pc.items():
        n = cfg_name(c)
        p = (HERE / 'runs' / n / 'COMPLETE.json') if k in ('P1', 'P2', 'P7', 'P8') else (HERE / 'evals' / n / 'COMPLETE_main.json')
        if not p.exists(): missing.append(k); continue
        res[k] = json.loads(p.read_text())['dev']
    for t in ('ref_vcs_exact', 'ref_nce'):
        p = HERE / 'evals' / t / 'COMPLETE_main.json'
        if p.exists(): res[t] = json.loads(p.read_text())['dev']
        else: missing.append(t)
    if missing: print('WAITING', missing); return
    d = lambda a, b, cond, k='success9': 100 * (res[a][cond][k] - res[b][cond][k])
    out = dict(status='COMPLETE', pilots={k: cfg_name(c) for k, c in pc.items()}, dev=res,
               contrasts={cond: dict(P3_minus_P4=d('P3', 'P4', cond), P3_minus_P5=d('P3', 'P5', cond), P3_minus_P6=d('P3', 'P6', cond),
                                     interaction_dual_minus_fine=(d('P3', 'P1', cond) - d('P4', 'P2', cond)),
                                     aug_vs_clean_vcs=d('P3', 'P7', cond), aug_vs_clean_nce=d('P4', 'P8', cond), clean_P7_minus_P8=d('P7', 'P8', cond),
                                     P3_minus_P4_p95=res['P3'][cond]['p95_err'] - res['P4'][cond]['p95_err'])
                          for cond in MAIN_CONDITIONS})
    c = out['contrasts']['deform']
    out['expansion_rule'] = dict(rule='P3 - P4 >= +3 pp strict on DEV deform with p95 not worse', P3_minus_P4=c['P3_minus_P4'],
                                 p95_delta=c['P3_minus_P4_p95'], expand=c['P3_minus_P4'] >= 3 and c['P3_minus_P4_p95'] <= 0)
    R.write(HERE / 'ANALYSIS.json', out); print(json.dumps(out['contrasts'], indent=1)); print(out['expansion_rule'])


# ------------------------------------------------------------------------------------------------------------ CPU check
def cpu_check():
    freeze = verify(); torch.set_num_threads(4); torch.manual_seed(7); checks = {}
    # objectives: gradients, equivalences
    u0 = torch.randn(2, 9, 16, dtype=torch.float64, requires_grad=True); v0 = torch.randn(2, 9, 16, dtype=torch.float64, requires_grad=True)
    yx = torch.rand(2, 9, 2, dtype=torch.float64) * 200; ab = torch.tensor([2.0, 0.0], dtype=torch.float64, requires_grad=True)
    for c in [dict(obj='vcs_exact'), dict(obj='nce', tau=.07), dict(obj='soft_nce', sigma=4., tau=.15), dict(obj='bce'), dict(obj='rince', q=.5, tau=.15)]:
        f = lambda a, b, w: objective(a, b, c, yx, w)[0]
        assert torch.autograd.gradcheck(f, (u0, v0, ab), check_undefined_grad=False) if c['obj'] == 'bce' else torch.autograd.gradcheck(lambda a, b: f(a, b, None), (u0, v0))
    perm = torch.arange(9).repeat(2, 1)
    assert torch.equal(objective(u0, v0, dict(obj='nce', tau=.07))[0], T.objective(u0, v0, 'nce', perm)[0])
    assert torch.equal(objective(u0, v0, dict(obj='vcs_exact'))[0], T.objective(u0, v0, 'vcs_exact', perm)[0])
    a = objective(u0, v0, dict(obj='soft_nce', sigma=1e-3, tau=.15), yx)[0]; b = objective(u0, v0, dict(obj='nce', tau=.15))[0]
    assert torch.allclose(a, b, atol=1e-10), (a, b)
    a = objective(u0, v0, dict(obj='rince', q=1e-7, tau=.15))[0]; b = objective(u0, v0, dict(obj='nce', tau=.15))[0] + math.log(LAMBDA_RINCE)
    assert torch.allclose(a, b, atol=1e-4), (a, b)      # q -> 0: RINCE -> InfoNCE + log(lambda)
    checks['objectives'] = 'gradcheck 5 objectives; nce(0.07)/vcs_exact equal train.py; soft_nce(sigma->0)=nce; rince(q->0)=nce+log(lambda)'
    # domains
    checks['domains'] = dict(host=HOST_DOMAIN, new=NEW_DOMAIN)
    # data, init, sampler, descriptors, one step per objective and scale (2 FIT cores, n_sc 2 to stay CPU-sized)
    u, guard, H, fit, _ = L.context(DS, SEED, 'cpu', images=False)
    imgs = A.load_learned_inputs(guard, fit[:2], {'FIT'}, 'R23 CPU check FIT', 'cpu')
    orig = json.loads((BASE / f'handoff/r22_geometry_train_20261023/runs/{DS}_vcs_exact_{SEED}/COMPLETE.json').read_text())
    steps = {}
    for cfg in [dict(obj=o, lr=3e-4, tau=.07, sigma=4., q=.5, scale=s, train='aug') for o in OBJECTIVES for s in ('fine', 'dual')]:
        m = Model(cfg, fit[:2], imgs, 'cpu'); assert m.initial_encoder_sha256 == orig['initial_encoder_sha256']
        m.t.sampler.n_sc = 2
        hb_probe = A.HostBatchSampler(fit[:2], SEED, n_sc=2, n_pos=NPOS, crop=CROP); hb_probe.inner = INNER; probe = hb_probe.draw()
        assert probe.yx.min() >= INNER and probe.yx.max() <= CROP - INNER - 1 and ((probe.yx - .5) / 2).min() >= 16 and ((probe.yx - .5) / 2).max() <= 111
        if cfg['scale'] == 'fine' and cfg['obj'] == 'vcs_exact':
            bu, bv = A.assemble_crops(imgs, probe, CROP)
            du0, dv0 = A.batch_descriptors(m.t.eU, m.t.eV, bu, bv, probe.yx)
            assert torch.equal(du0, m.dU.batch(bu, probe.yx)) and torch.equal(dv0, m.dV.batch(bv, probe.yx))
        row = train_step(m); assert row['batch_sha256'] == L.batch_sha(probe)
        g = [p.grad for p in m.params]; assert all(x is not None and torch.isfinite(x).all() for x in g)
        if cfg['scale'] == 'dual': assert m.dU.fuse.weight.grad.abs().sum() > 0 and m.dV.fuse.weight.grad.abs().sum() > 0
        if cfg['obj'] == 'bce': assert m.ab.grad.abs().sum() > 0
        steps[cfg_name(cfg)] = dict(loss=row['loss'], grad_norm=row['grad_norm_preclip'], n_deformed=row['n_deformed'])
    checks['train_steps'] = steps
    checks['paired_streams'] = 'every objective/scale draws the same host batch (batch hash) and the same augmentation stream at step 1'
    # host-equivalence of the parameterized energy (single scale, host domain) on a real CAL problem and the long-v2 30k model
    d = BASE / f'handoff/r22_geometry_long_v2_20261024/runs/{DS}_vcs_exact_{SEED}'; done = json.loads((d / 'COMPLETE.json').read_text())
    t = T.GeometryTrainer(L.config(SEED), {}, fit, 'cpu', 'vcs_exact'); R.CP.load(t, d / 'checkpoints/step_30000.pt', done['identity'])
    core = sorted(guard.included('CAL'))[0]; pert = {core: Q.perturbations_p2(core)}; guard.log_rule(core, {'CAL'}, 'R23 CPU check')
    host_in = A.GuardedInputs(guard, {'CAL'}, 'R23 CPU check host', pert, 'cpu'); mine_in = CondInputs(guard, {'CAL'}, 'R23 CPU check mine', pert, 'cpu', 'clean')
    assert torch.equal(host_in.moving(core, 0)[1], mine_in.moving(core, 0)[1])
    with torch.no_grad():
        ref = A.reference_descriptors(t.eU, host_in, core); mov = host_in.moving(core, 0)[1]
        he = Q.RigidEnergy(torch, F, 'critic', ref, mov, np.random.default_rng(Q.seed_of(f'R6-REG-P2|pos|{core}|0')), eV=t.eV, critic=t.critic)
        he.j_critic = types.MethodType(T.cosine_energy, he); he.geometry_readout = 'COS'
        dU, dV = Desc(t.eU, False), Desc(t.eV, False)
        me = Energy(dU, dV, (ref, None), mov, np.random.default_rng(Q.seed_of(f'R6-REG-P2|pos|{core}|0')), HOST_DOMAIN)
        assert torch.equal(he.x, me.x)
        th = torch.tensor([0.1, -0.2, 0.05]); tt = torch.tensor([[1.0, -2.0], [3.5, 0.5], [-4.0, 2.0]])
        hb_, mb_ = he.batch(th, tt), me.batch(th, tt); ts = torch.tensor(Q.grid_points()[1][:70], dtype=torch.float32)
        hg, mg = he.grid_values(6.0, ts), me.grid_values(6.0, ts)
    eq = dict(batch_max_abs=float((hb_ - mb_).abs().max()), grid_max_abs=float((hg - mg).abs().max()))
    assert eq['batch_max_abs'] <= 1e-6 and eq['grid_max_abs'] <= 1e-6, eq
    # dual-scale energy on the new domain: same descriptor function in grid and local paths; report grid-vs-local gap at grid nodes
    m = Model(dict(obj='vcs_exact', lr=3e-4, scale='dual', train='aug'), fit[:2], {}, 'cpu')
    gaps = {}
    with torch.no_grad():
        for label, (du_, dv_, dom) in {'single_host': (dU, dV, HOST_DOMAIN), 'single_new': (dU, dV, NEW_DOMAIN), 'dual_new': (m.dU, m.dV, NEW_DOMAIN)}.items():
            f, c = du_.maps(host_in.reference(core)[1][None]); e = Energy(du_, dv_, (f[0], None if c is None else c[0]), mov, np.random.default_rng(1), dom)
            ts6 = torch.tensor(Q.grid_points()[1][::50], dtype=torch.float32); gv = e.grid_values(3.0, ts6)
            lv = e.batch(torch.full((len(ts6),), math.radians(3.0)), ts6); gaps[label] = float((gv - lv).abs().max())
    checks['energy'] = dict(host_equivalence=eq, grid_vs_local_max_abs=gaps)
    # condition inputs: deform with amplitude 0 equals clean; occlude + sparse partition the clean image
    reg, _ = eval_regions(core, 0, 'deform')
    sh = torch.tensor(np.expm1(P.load_img(host_in.root, core, ('shg_log',))[0].astype(np.float64)), dtype=torch.float32)[None]
    assert torch.allclose(deform_image(sh, reg, 0.0), sh, atol=1e-5)
    oc, sp = CondInputs(guard, {'CAL'}, 'c', pert, 'cpu', 'occlude').moving(core, 0)[1], CondInputs(guard, {'CAL'}, 'c', pert, 'cpu', 'sparse_only').moving(core, 0)[1]
    assert torch.allclose(oc + sp, host_in.moving(core, 0)[1], atol=1e-5)
    checks['conditions'] = 'clean == host moving (bitwise); deform(A=0) == identity; occlude + sparse_only == clean'
    # generator sanity on every CAL/DEV instance and 2000 training draws
    gen = {}
    disk = Q.grid_coords(torch, 0, Q.S - 1, 'cpu'); disk = disk[(disk - Q.CEN).norm(dim=-1) < NEW_DOMAIN['R']]
    for cond in ('deform', 'deform_4px_15', 'deform_8px_30'):
        fr, jac, bound, nreg = [], [], [], []
        for role in ('CAL', 'DEV'):
            for core_ in guard.included(role):
                for i in range(10):
                    reg, amp = eval_regions(core_, i, cond)
                    cs = np.array(reg['centers']); assert all(np.hypot(*(c_ - Q.CEN)) + reg['r'] <= NEW_DOMAIN['R'] + 1e-9 for c_ in cs)
                    if len(cs) == 2: assert np.hypot(*(cs[0] - cs[1])) >= 2 * reg['r'] - 1e-9
                    fr.append(float(region_mask(disk, reg).float().mean())); bound.append(amp * math.pi / (2 * reg['r'])); nreg.append(len(cs))
                    if i < 2: jac.append(min_jacobian(reg, amp, Q.S))
        gen[cond] = dict(n=len(fr), area_fraction_mean=float(np.mean(fr)), area_fraction_min=float(np.min(fr)), area_fraction_max=float(np.max(fr)),
                         amplitude=CONDITIONS[cond][0], lipschitz_bound_max=float(np.max(bound)), min_jacobian_det=float(np.min(jac)), two_region_share=float(np.mean([n == 2 for n in nreg])))
        assert max(bound) < 1 and min(jac) > 0
    rng = np.random.default_rng(0); fr, jac, bound = [], [], []
    cpts = Q.grid_coords(torch, 0, CROP - 1, 'cpu')
    for k in range(2000):
        reg = make_regions(rng, .15 * CROP * CROP, 'crop', size=CROP); fr.append(float(region_mask(cpts, reg).float().mean())); bound.append(8 * math.pi / (2 * reg['r']))
        if k < 50: jac.append(min_jacobian(reg, 8.0, CROP))
    gen['train_crop'] = dict(n=2000, area_fraction_mean=float(np.mean(fr)), lipschitz_bound_max=float(np.max(bound)), min_jacobian_det=float(np.min(jac)), amplitude=8.0)
    R.write(HERE / 'GENERATOR.json', gen)
    assert {r['role'] for r in guard.reads} <= {'FIT', 'CAL'}
    checks['roles_read'] = sorted({r['role'] for r in guard.reads})
    R.write(HERE / 'CPU_CHECK.json', dict(status='PASS', freeze_sha256=freeze, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), checks=checks, generator=gen))
    print('R23_CPU PASS', json.dumps(checks['energy']), flush=True)


def smoke(steps):
    """GPU: ~steps dual-scale updates per objective + single-scale VCS, one small CAL eval per run (clean, deform)."""
    cfgs = [dict(obj='vcs_exact', lr=3e-4), dict(obj='nce', tau=.07, lr=3e-4), dict(obj='soft_nce', sigma=4., tau=.07, lr=3e-4),
            dict(obj='bce', lr=3e-4), dict(obj='rince', q=.5, tau=.07, lr=3e-4)]
    out = {}
    for c in [dict(x, scale='dual', train='aug') for x in cfgs] + [dict(cfgs[0], scale='fine', train='aug')]:
        name = cfg_name(c)
        globals()['find_config'] = (lambda n, _c=c: (_c, 'smoke'))
        run_train(name, smoke_steps=steps)
        out[name] = json.loads((HERE / 'smoke' / name / 'COMPLETE.json').read_text())
    R.write(HERE / 'smoke' / f'SMOKE_{os.environ["SLURM_JOB_ID"]}.json', dict(status='PASS', runs={k: dict(mean_step_ms=v['mean_step_ms'], peak_gib=v['peak_allocated_bytes'] / 2 ** 30,
            gpu=v['gpu'], smoke_eval=v['smoke_eval']) for k, v in out.items()}))
    print('R23_SMOKE PASS', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['check', 'smoke', 'train', 'select', 'eval', 'analyze'])
    p.add_argument('--name'); p.add_argument('--target'); p.add_argument('--dev', action='store_true'); p.add_argument('--aux', action='store_true')
    p.add_argument('--steps', type=int, default=200); a = p.parse_args()
    if a.mode == 'check': cpu_check()
    elif a.mode == 'smoke': verify(); smoke(a.steps)
    elif a.mode == 'train': run_train(a.name, a.dev)
    elif a.mode == 'select': select()
    elif a.mode == 'eval': run_eval(a.target, a.aux)
    else: analyze()
