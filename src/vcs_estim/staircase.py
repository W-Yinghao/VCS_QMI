"""Staircase runner: one (setting, estimator, negative variant, N, lr, seed) cell trained through the dependence staircase with a fresh
critic-free evaluation batch at every log point; oracle estimates on the same evaluation batches where the PMI is known."""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from .critics import JointMLP, OracleCritic
from .data import LEVELS_NATS, contaminate, make_setting, truths_by_mc
from .estimators import EMA, TARGET, estimate, oracle_transform, scores


def run_cell(*, setting: str, kind: str, variant: str, N: int, lr: float, seed: int, steps_per_level: int = 4000, log_every: int = 100,
             levels=None, d: int = 20, device=None, truth_mc: int = 200_000, contamination=None) -> dict:
    device = device or (torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu"))
    levels = list(levels if levels is not None else range(len(LEVELS_NATS)))
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed * 1000 + 7); egen = torch.Generator().manual_seed(seed * 1000 + 11)
    s0 = make_setting(setting, levels[0], d)
    critic = JointMLP(s0.d, s0.d).to(device); opt = torch.optim.Adam(critic.parameters(), lr=lr); ema = EMA() if kind == "dv" else None
    high = make_setting(setting, len(LEVELS_NATS) - 1, d) if contamination else None
    out = {"setting": setting, "kind": kind, "variant": variant, "N": N, "lr": lr, "seed": seed, "target": TARGET[kind], "steps_per_level": steps_per_level,
           "log_every": log_every, "device": str(device), "contamination": contamination, "levels": []}
    t0 = time.time()
    for li in levels:
        st = make_setting(setting, li, d); truth = truths_by_mc(st, truth_mc, seed=123, device=device)
        oracle = OracleCritic(st.pmi, oracle_transform(kind))
        logs = []
        for step in range(1, steps_per_level + 1):
            x, y = st.sample(N, gen, device)
            if contamination:
                x, y, _ = contaminate(x, y, contamination["eps"], contamination["kind"], gen, high)
            f_pos, f_neg = scores(critic, x, y, variant, gen); loss, _ = estimate(kind, f_pos, f_neg, ema=ema)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if step % log_every == 0:
                with torch.no_grad():
                    xe, ye = st.sample(N, egen, device)
                    if contamination:
                        xe, ye, _ = contaminate(xe, ye, contamination["eps"], contamination["kind"], egen, high)
                    fp, fn = scores(critic, xe, ye, variant, egen); _, ve = estimate(kind, fp, fn)
                    fpt, fnt = scores(critic, x, y, variant, gen); _, vt = estimate(kind, fpt, fnt)
                    op, on = scores(oracle, xe, ye, variant, egen); _, vo = estimate(kind, op, on)
                logs.append((step, float(vt), float(ve), float(vo)))
        arr = np.array(logs); ve = arr[:, 2]; vo = arr[:, 3]; last = ve[-max(1, 1000 // log_every):]
        tv = truth[TARGET[kind]]; finite = np.isfinite(ve)
        div = int((~finite).sum() + (np.abs(ve[finite]) > 3 * max(abs(tv), 1e-6)).sum())
        jumps = np.abs(np.diff(ve[finite])) if finite.sum() > 1 else np.array([0.0])
        out["levels"].append({"level": li, "mi_nats": st.mi, "param": st.param, "truth": truth, "n_logs": len(logs),
                              "mean_last": float(np.nanmean(last)), "sd_last": float(np.nanstd(last, ddof=1)) if len(last) > 1 else 0.0,
                              "median_last": float(np.nanmedian(last)), "iqr_last": float(np.nanpercentile(last, 75) - np.nanpercentile(last, 25)),
                              "q01": float(np.nanpercentile(ve[finite], 1)) if finite.any() else float("nan"), "q99": float(np.nanpercentile(ve[finite], 99)) if finite.any() else float("nan"),
                              "max_jump": float(jumps.max()) if len(jumps) else 0.0, "divergences": div,
                              "oracle_mean_last": float(np.nanmean(vo[-max(1, 1000 // log_every):])), "learned_over_oracle": float(np.nanmean(last) / np.nanmean(vo[-max(1, 1000 // log_every):])) if np.nanmean(vo[-max(1, 1000 // log_every):]) != 0 else float("nan"),
                              "train_mean_last": float(np.nanmean(arr[-max(1, 1000 // log_every):, 1])), "log": [list(map(float, r)) for r in logs]})
    out["wall_s"] = time.time() - t0
    return out
