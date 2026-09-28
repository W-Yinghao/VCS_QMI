"""Estimator package v1 (§6.2, §7.2): candidate and residual training with SELECT-based checkpoint choice.

Defaults (implementation defaults of the package, not theory): Adam, lr 5e-4, weight decay 0, batch 256 positive + 256 negative pairs,
at most 2000 updates, the native risk evaluated on SELECT every 100 updates (and at update 0); the lowest-SELECT-risk state is kept.
FIT P and FIT Q are sampled by independent epoch permutations.  Oracle quantities are never used here.
"""
from __future__ import annotations

import copy
import time

import torch

from .objectives import j_hat, js_match_loss, vcs_loss


def _batches(n, batch, gen):
    while True:
        perm = torch.randperm(n, generator=gen)
        for s in range(0, n - batch + 1, batch):
            yield perm[s: s + batch]


def train(model, loss_fn, fit, select, *, lr=5e-4, wd=0.0, batch=256, updates=2000, every=100, seed=0, device="cpu"):
    """loss_fn(model, xp, yp, xq, yq) -> scalar native risk.  fit / select: RoleData (float64 tensors, cast to float32 on device)."""
    t0 = time.time(); g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    model = model.to(device); opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    F = [t.float().to(device) for t in (fit.xp, fit.yp, fit.xq, fit.yq)]
    Sl = [t.float().to(device) for t in (select.xp, select.yp, select.xq, select.yq)]
    bp, bq = _batches(len(F[0]), batch, g), _batches(len(F[2]), batch, g)

    def sel_risk():
        model.eval()
        with torch.no_grad():
            v = float(loss_fn(model, *Sl))
        model.train(); return v

    curve = [(0, sel_risk())]; best = (curve[0][1], 0, copy.deepcopy(model.state_dict()))
    for step in range(1, updates + 1):
        ip, iq = next(bp).to(device), next(bq).to(device)
        loss = loss_fn(model, F[0][ip], F[1][ip], F[2][iq], F[3][iq])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % every == 0:
            v = sel_risk(); curve.append((step, v))
            if v < best[0]:
                best = (v, step, copy.deepcopy(model.state_dict()))
    model.load_state_dict(best[2]); model.eval()
    return model, {"select_curve": curve, "selected_update": best[1], "select_risk": best[0], "fit_seconds": time.time() - t0,
                   "updates": updates, "lr": lr, "batch": batch, "weight_decay": wd}


def native_loss(kind):
    def f(model, xp, yp, xq, yq):
        fp, fn = model(xp, yp), model(xq, yq)
        return vcs_loss(fp, fn) if kind == "vcs" else js_match_loss(fp, fn)
    return f


def residual_loss(base, lam0):
    """-J((1 - lam0) T0 + lam0 U), T0 frozen; the trained quantity is the raw J (§7.2)."""
    def f(model, xp, yp, xq, yq):
        with torch.no_grad():
            t0p, t0n = torch.tanh(base(xp, yp)), torch.tanh(base(xq, yq))
        up, un = torch.tanh(model(xp, yp)), torch.tanh(model(xq, yq))
        return -j_hat((1 - lam0) * t0p + lam0 * up, (1 - lam0) * t0n + lam0 * un)
    return f


@torch.no_grad()
def outputs(model, x, y, device="cpu", chunk=65536) -> torch.Tensor:
    """Un-squashed f on (x, y), float64 on CPU."""
    model.eval(); out = []
    for s in range(0, len(x), chunk):
        out.append(model(x[s: s + chunk].float().to(device), y[s: s + chunk].float().to(device)).double().cpu())
    return torch.cat(out)
