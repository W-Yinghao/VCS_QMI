"""R24 stage 0/1: the released Neon_ShgBf test-time pipeline (Learn2Reg 2024 COMULIS SHG-BF, next-gen-nn) as a fixed strong host on
our CoMIR-protocol BF-SHG problems, with every candidate deformation of the host's own search recorded and scored by several
selection rules.

Problem (per core, draw k = the R23-B CoMIR-protocol draw): fixed = full-res 834 x 834 SHG centre patch (607..1440) of the core;
moving = BF of the same window after the rigid transform T(z) = R(theta)(z - c) + c + t was applied to the full BF core
(warp then crop, rotation +-30 deg, |t| <= 100 px).  Truth: fixed pixel z <-> moving pixel T(z).  Neon crops 834 -> 832 like its
test script and predicts a displacement field u (dy, dx) with  warped(q) = moving(q + u(q)).
Errors (full-res px): grid TRE = mean over a 16 x 16 grid of interior patch points of |q + u(q) - T(q)|; corner = the same at the
4 patch corners (CoMIR definition, extrapolated flow for a non-rigid method).

Host loop: a line-by-line copy of test_ShgBfReg_testset_io.run() for one image (3 restarts x 60 iterations, NCC half then MI
half, MI on 8-cluster k-means labels selects the round best, the best round is the output), with the per-problem RNG reset to
the script's seed 0.  Candidate pool = per round the XFeat affine initialisation + the 60 iterates (183 candidates).
Scores per candidate (all on the submitted flow, fixed descriptors at the same 4096 positions, valid = receptive field inside the
warped moving patch): host MI (native, round-specific labels), our soft MI, and for each of 9 long-v2 30k encoders (BF exact VCS /
BCE / InfoNCE x seeds 1001-1003, the same shared 16-d cosine geometry) its own objective (exact J, BCE, InfoNCE) and COS.
"""
import argparse, json, math, os, sys, time, hashlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
NEON = Path('/projects/EEG-foundation-model/yinghao/models/neon_shgbf/src')
CKPT = NEON / 'best_score_-6.6130_net_epoch_22.pth'
LONG = HERE.parents[0] / 'r22_geometry_long_v2_20261024/runs'
sys.path.insert(0, str(NEON))
import torch, torch.nn as nn, torch.nn.functional as F                                   # noqa: E402
from utils import setters                                                                 # noqa: E402  (Neon)
from utils.functions import registerSTModel, normalize_common_torch, image2cat_kmeans, to_tensor, get_downsampled_images, convert_state_dict  # noqa: E402
from utils.loss import NccLoss                                                            # noqa: E402
from models import getModel                                                               # noqa: E402
from models.xfeat import XFeat                                                            # noqa: E402
from ff_mi import mi_loss                                                                 # noqa: E402
import test_ShgBfReg_testset_io as NT                                                     # noqa: E402  (get_flow, smallDispModel)
sys.path.insert(0, str(HERE))
from problems import C, area, build_problem, truth_map, GRID, CORNERS, S, W, CF, CEN, DOWN  # noqa: E402
import cv2                                                                                # noqa: E402

N_POS = 4096; RF = 15
ARMS = ('vcs_exact', 'bce', 'nce'); SEEDS = (1001, 1002, 1003)
OPT = dict(n_iters=60, lr=1e-4, sim_w=1., reg_w=1., k=8, rounds=3)                        # the script defaults / README command


def sha(p):
    h = hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()


def flow_at(flow, pts):
    """bilinear sample of flow [1,2,W,W] at patch points (row, col) -> [n,2] (dy, dx)."""
    p = torch.as_tensor(pts, dtype=torch.float32, device=flow.device)
    g = torch.stack([p[:, 1] / (W - 1) * 2 - 1, p[:, 0] / (W - 1) * 2 - 1], -1)[None, None]
    return F.grid_sample(flow, g, align_corners=True)[0, :, 0].T.double().cpu().numpy()


def errors(flow, prob):
    out = {}
    for name, pts in (('grid', GRID), ('corner', CORNERS)):
        out[name] = float(np.linalg.norm(pts + flow_at(flow, pts) - truth_map(prob, pts), axis=1).mean())
    return out


def gt_flow(prob, dev):
    q = np.stack(np.meshgrid(np.arange(W), np.arange(W), indexing='ij'), -1).reshape(-1, 2).astype(np.float64)
    u = truth_map(prob, q) - q; return torch.tensor(u.T.reshape(1, 2, W, W), dtype=torch.float32, device=dev)


# ------------------------------------------------------------------------------------------------------------ host inputs
def host_inputs(pb, dev):
    """Neon ImageTransform(is_training=False): to_tensor, per-image per-channel min-max; SHG mask = fixed >= 1; crop to 832."""
    def mm(t):
        mx = F.adaptive_max_pool2d(t, (1, 1)); mn = -F.adaptive_max_pool2d(-t, (1, 1)); return (t - mn) / (mx - mn)
    x = mm(torch.tensor(pb['mov'], dtype=torch.float32).permute(2, 0, 1) / 255.0)
    y = mm(torch.tensor(pb['fix'], dtype=torch.float32)[None] / 255.0)
    y_msk = torch.tensor(pb['fix'] >= 1).float()[None]
    return x[None, :, :W, :W].to(dev), y[None, :, :W, :W].to(dev), y_msk[None, :, :W, :W].to(dev)


def load_neon(dev, via_getters=False):
    opt = dict(model='tiramisuAndXfeatComplex', nkwargs=dict(ti_pretrained='0', xf_pretrained='0'), load_ckpt='none')
    if via_getters:                    # released code path: getters reloads '../best_score_...pth' relative to the cwd (bug 1)
        from utils import getters
        cwd = os.getcwd(); os.chdir(NEON / 'fields_cwd'); m, _ = getters.getTestModelWithCheckpoints(opt); os.chdir(cwd); return m
    cwd = os.getcwd(); os.chdir(NEON); m = getModel(opt); os.chdir(cwd)    # xfeatComplex always reads './xfeat.pt' (overwritten below)
    m.load_state_dict(convert_state_dict(torch.load(CKPT, map_location='cpu', weights_only=False))); return m


# ------------------------------------------------------------------------------------------------------------ scorers
def enc(cin):
    return nn.Sequential(nn.Conv2d(cin, 32, 7, padding=3), nn.GELU(), nn.Conv2d(32, 32, 9, padding=4), nn.GELU(),
                         nn.Conv2d(32, 32, 9, padding=4), nn.GELU(), nn.Conv2d(32, 16, 9, padding=4))


def load_encoders(dev):
    out = {}
    for arm in ARMS:
        for seed in SEEDS:
            d = LONG / f'bf_{arm}_{seed}/checkpoints'; meta = json.loads((d / 'step_30000.json').read_text())
            assert sha(d / 'step_30000.pt') == meta['sha256'] and meta['identity']['arm'] == arm and meta['step'] == 30000
            ck = torch.load(d / 'step_30000.pt', map_location='cpu', weights_only=False)
            eU, eV = enc(3), enc(1); eU.load_state_dict(ck['modules'][0]); eV.load_state_dict(ck['modules'][1])
            out[f'{arm}_{seed}'] = (eU.to(dev).eval(), eV.to(dev).eval())
    return out


def exact_j(u, v):
    p = (u * v).sum(-1); mu, mv = u.mean(0), v.mean(0)
    return (p - .5 * p.square()).mean() - (mu * mv).sum() - .5 * ((u.T @ u / len(u)) * (v.T @ v / len(v))).sum()


def own_score(arm, u, v):
    if arm == 'vcs_exact': return exact_j(u, v)
    s = u @ v.T
    if arm == 'bce':
        p = s.diagonal(); return torch.log(((1 + p) / 2).clamp(1e-6, 1 - 1e-6)).mean() + torch.log(((1 - s) / 2).clamp(1e-6, 1 - 1e-6)).mean() + 2 * math.log(2)
    lab = torch.arange(len(u), device=u.device)
    return -.5 * (F.cross_entropy(s / .07, lab) + F.cross_entropy(s.T / .07, lab))


def soft_mi(a, b, bins=64, sigma=1.0):
    def w(x):
        lo, hi = torch.quantile(x, .01), torch.quantile(x, .99); c = torch.arange(bins, device=x.device, dtype=x.dtype)
        z = ((x - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1) * (bins - 1); e = torch.exp(-.5 * ((z[:, None] - c) / sigma) ** 2)
        return e / e.sum(-1, keepdim=True)
    Pj = w(a).T @ w(b) / len(a); ent = lambda p: -(p * torch.log(p.clamp_min(1e-12))).sum()
    return ent(Pj.sum(1)) + ent(Pj.sum(0)) - ent(Pj)


class Scorer:
    def __init__(self, pb, prob, encs, dev):
        self.dev, self.encs = dev, encs
        bm, bs = pb['bf_norm']; sm, ss = pb['shg_norm']
        self.bfn = (bm, bs)
        self.raw = torch.tensor(pb['mov'][:W, :W], dtype=torch.float32, device=dev).permute(2, 0, 1)[None] / 255.0
        shd = torch.tensor(area(pb['fix'][:W, :W].astype(np.float64) / 255.0), dtype=torch.float32, device=dev)
        self.shg_in = ((torch.log1p(shd) - sm) / ss)[None, None]
        self.shg_lum = torch.log1p(shd)
        g = np.random.default_rng(C.Q.seed_of(f"R24-NEON-POS|{prob['core']}|{prob['k']}"))
        self.pos = torch.tensor(g.integers(RF + 1, W // DOWN - RF - 1, size=(N_POS, 2)), device=dev)
        with torch.no_grad():
            self.fixd = {k: F.normalize(eV(self.shg_in)[0], dim=0) for k, (eU, eV) in encs.items()}
        self.st = registerSTModel((W, W), 'bilinear').to(dev)

    @torch.no_grad()
    def __call__(self, flow):
        warped = self.st(self.raw, flow); ones = self.st(torch.ones_like(self.raw[:, :1]), flow)
        wd = F.avg_pool2d(warped, DOWN); md = F.avg_pool2d(ones, DOWN)
        valid_map = -F.max_pool2d(-md, 2 * RF + 1, 1, RF) > 0.999                          # receptive field inside the warped moving patch
        ok = valid_map[0, 0][self.pos[:, 0], self.pos[:, 1]]; n = int(ok.sum()); out = dict(n_valid=n)
        if n < 64:
            return out
        py, px = self.pos[ok, 0], self.pos[ok, 1]; bfn = (wd - self.bfn[0]) / self.bfn[1]
        for k, (eU, eV) in self.encs.items():
            u = F.normalize(eU(bfn)[0][:, py, px].T, dim=-1); v = self.fixd[k][:, py, px].T
            out[k] = dict(own=float(own_score(k.rsplit('_', 1)[0], u, v)), cos=float((u * v).sum(-1).mean()))
        lum = (wd[0, 0] * .299 + wd[0, 1] * .587 + wd[0, 2] * .114)[py, px]
        out['soft_mi'] = float(soft_mi(lum.double(), self.shg_lum[py, px].double()))
        return out


# ------------------------------------------------------------------------------------------------------------ host loop
def neon_register(model, x_imgs, y_imgs, y_msk, dev, on_candidate):
    """Line-by-line the per-image body of test_ShgBfReg_testset_io.run(); on_candidate(round, it, flow, mi) for every candidate."""
    reg_model_bi = registerSTModel((W, W), 'bilinear').to(dev)
    ncc = [NccLoss(win=[19, 19]), NccLoss(win=[13, 13]), NccLoss(win=[9, 9]), NccLoss(win=[5, 5]), NccLoss(win=[3, 3])]
    k = OPT['k']; round_best_ncc, round_best_flow = 0, None
    xfeat_wrap = XFeat(device=dev); xfeat_wrap.init_net(model.xfeat); xfeat_wrap.eval()
    for round_ in range(OPT['rounds']):
        flow_model = NT.smallDispModel(N_s=32, lk_size=15).to(dev)
        optimizer = torch.optim.Adam(flow_model.parameters(), lr=OPT['lr'], weight_decay=0, amsgrad=True); flow_model.train()
        y = torch.log(y_imgs + 1)
        with torch.no_grad():
            x_feas, y_feas = model.tiramisu(x_imgs, y)
            x_feas, y_feas = normalize_common_torch(x_feas, y_feas)
        if round_ == 0:                # identical input and fixed random_state in every round: computed once, same labels
            y_lbl0 = to_tensor(image2cat_kmeans(y.cpu().numpy()[0].reshape(W, W, 1), k=k), True)
        y_lbl = y_lbl0
        pred_flow_ = NT.get_flow(xfeat_wrap, x_feas, y_feas)
        x_org = reg_model_bi(x_imgs, pred_flow_)
        x_lbl = to_tensor(image2cat_kmeans(x_org[0].permute(1, 2, 0).cpu().numpy(), k=k), True)
        xx, yy = x_feas.detach(), y_feas.detach(); xx = reg_model_bi(xx, pred_flow_)
        x_lbl, y_lbl = x_lbl.detach(), y_lbl.detach().float()
        x_mask = reg_model_bi(torch.ones_like(x_lbl), pred_flow_)
        best_flow, best_ncc = pred_flow_, 0
        on_candidate(round_, -1, pred_flow_.detach(), float(mi_loss(x_lbl, y_lbl, x_mask, Q_A=k, Q_B=k)))   # affine init (never selected natively)
        for iter_ in range(OPT['n_iters']):
            optimizer.zero_grad()
            pflows, flows, xs, ys = flow_model(xx, yy, y_msk)
            xs = get_downsampled_images(xx, 4, mode='bilinear'); ys = get_downsampled_images(yy, 4, mode='bilinear')
            x_lbl_s = get_downsampled_images(x_lbl, 4, mode='nearest'); y_lbl_s = get_downsampled_images(y_lbl, 4, mode='nearest')
            x_mask_s = get_downsampled_images(x_mask, 4, mode='nearest')
            tr = [flow_model.transformer_0, flow_model.transformer_1, flow_model.transformer_2, flow_model.transformer_3, flow_model.transformer_4]
            if iter_ < int(OPT['n_iters']) / 2:
                sim = sum(ncc[i](tr[i](xs[i], flows[i]), ys[i]) / 2 ** i for i in range(5))
            else:
                sim = sum((1. - mi_loss(y_lbl_s[i], tr[i](x_lbl_s[i], flows[i]), x_mask_s[i], Q_A=k, Q_B=k)) / 2 ** i for i in range(5))
            pred_flow = flows[0] + pred_flow_
            loss = OPT['sim_w'] * sim + OPT['reg_w'] * (sim * 0)
            loss.backward(); optimizer.step()
            with torch.no_grad():
                eval_ncc_ = mi_loss(reg_model_bi(x_lbl, flows[0]), y_lbl, x_mask, Q_A=k, Q_B=k).item()
            on_candidate(round_, iter_, pred_flow.detach(), eval_ncc_)
            if eval_ncc_ > best_ncc:
                best_ncc, best_flow = eval_ncc_, pred_flow.detach()
        if best_ncc > round_best_ncc:
            round_best_ncc, round_best_flow = best_ncc, best_flow
        xfeat_wrap = XFeat(device=dev); xfeat_wrap.init_net(model.xfeat); xfeat_wrap.eval()
    return round_best_flow, round_best_ncc


def seed_all():
    setters.setSeed(0); cv2.setRNGSeed(0)


# ------------------------------------------------------------------------------------------------------------ main
def run(role, shard, nshard, out, smoke=False, variant='fixed', kmax=C.K_DRAWS):
    dev = 'cuda'; assert torch.cuda.is_available(); out = Path(out); out.mkdir(parents=True, exist_ok=True)
    g = C.guard(); probs = [C.draw('comir', c, k) for c in sorted(g.included(role)) for k in range(kmax)]
    probs = probs[shard::nshard][:2 if smoke else None]
    model = load_neon(dev, via_getters=(variant == 'released')); model.eval()
    for p in model.parameters(): p.requires_grad = False
    if smoke:                                  # bug-1 check: both load paths give the same weights for the author checkpoint
        m2 = load_neon(dev, via_getters=True); a, b = model.state_dict(), m2.state_dict()
        assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a); del m2
    encs = load_encoders(dev); rows_path = out / f'{role}_s{shard}.jsonl'
    done = {(r['core'], r['k']) for r in map(json.loads, open(rows_path))} if rows_path.exists() else set()
    for prob in probs:
        if (prob['core'], prob['k']) in done: continue
        t0 = time.time(); pb = build_problem(g, role, prob)
        x, y, ym = host_inputs(pb, dev); sc = Scorer(pb, prob, encs, dev)
        if smoke:                              # truth convention: warping the moving BF by the true flow recovers the fixed-frame BF
            wt = registerSTModel((W, W), 'bilinear').to(dev)(torch.tensor(pb['mov'][:W, :W], dtype=torch.float32, device=dev).permute(2, 0, 1)[None], gt_flow(prob, dev))
            ob = torch.tensor(pb['orig_bf'][:W, :W], dtype=torch.float32, device=dev).permute(2, 0, 1)[None]
            inside = registerSTModel((W, W), 'bilinear').to(dev)(torch.ones_like(wt[:, :1]), gt_flow(prob, dev))[0, 0] > 0.999   # truth inside the moving patch
            cc = float(torch.corrcoef(torch.stack([wt[0, 0][inside], ob[0, 0][inside]]))[0, 1]); assert cc > 0.98, cc
            gs = sc(gt_flow(prob, dev)); assert gs['n_valid'] >= 64
        cands = []
        def on_c(rd, it, flow, mi):
            cands.append(dict(round=rd, iter=it, mi_native=mi, **errors(flow, prob), **sc(flow)))
        seed_all(); best, best_mi = neon_register(model, x, y, ym, dev, on_c)
        row = dict(core=prob['core'], k=prob['k'], role=role, theta=prob['theta'], ty_full=prob['ty_full'], tx_full=prob['tx_full'],
                   native=(dict(mi=best_mi, output='field', **errors(best, prob)) if best is not None else   # host wrote no field: identity
                           dict(mi=None, output='none', **errors(torch.zeros(1, 2, W, W, device=dev), prob))), seconds=time.time() - t0, candidates=cands, variant=variant,
                   gpu=torch.cuda.get_device_name(), job=os.environ.get('SLURM_JOB_ID'))
        if smoke: row['smoke_truth_ncc'] = cc; row['smoke_truth_scores'] = gs
        with open(rows_path, 'a') as f: f.write(json.dumps(row) + '\n')
        print(prob['core'], prob['k'], row['native']['output'], 'native grid %.2f corner %.2f oracle %.2f  %.0fs' % (row['native']['grid'], row['native']['corner'],
              min(c['grid'] for c in cands), row['seconds']), flush=True)
    print('DONE', role, shard, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--role', default='CAL'); ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshard', type=int, default=1); ap.add_argument('--out', required=True); ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--variant', default='fixed', choices=('fixed', 'released')); ap.add_argument('--kmax', type=int, default=C.K_DRAWS)
    a = ap.parse_args(); run(a.role, a.shard, a.nshard, a.out, a.smoke, a.variant, a.kmax)
