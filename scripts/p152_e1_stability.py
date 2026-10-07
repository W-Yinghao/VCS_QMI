"""P152 (r3 R3-E1) — the stability fields the P85 / P86 records lack, on the registered 3-level Gaussian ladder, by a RECORDED re-fit that mirrors
`vcs_estim.benchmark.train_neural` step for step (same data streams via build_roles, same torch seeds, same batch draws, same losses via
`estimators.estimate`, EMA for DV, SELECT risk every 100 updates) for the learning rate P85 selected in that cell.  Nothing in the frozen benchmark
module is modified.

Per unit (I in {2, 6, 10}, gaussian d 20 / 20, N 4096, B 256, seed in {0, 1, 2}; kind in {vcs, js, infonce, nwj, dv, smile}, in-batch negatives):
  per update   : native loss value, critic-parameter gradient L2 norm (after backward), non-finite flag           -> gradient-norm variance
  checkpoints  : updates 250 / 1000 / 2000 and the SELECT-best state (P85 rule); state_dict of each saved
  per checkpoint: native value on the full EVAL (as P85), 64 independent EVAL blocks of 512 (eval variance conditional on the fit),
                 critic-output histograms (f: 201 bins on [-10, 10] with overflow counts; T = tanh f: 101 bins), saturation fractions,
                 for vcs / js the posterior readouts (common T) and per-sample T on EVAL (saved, float16) -> paired bootstraps later
  reproduction : the U = 2000 SELECT-best native value vs the P85 stored value of the same (cell, kind, lr) (descriptive; CPU vs GPU arithmetic)
    python scripts/p152_e1_stability.py --I 6 --seed 0 --out outputs/P152_e1 [--kinds vcs,js] [--smoke]
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from vcs_estim import benchmark as BM  # noqa: E402
from vcs_estim.critics import JointMLP  # noqa: E402
from vcs_estim.data import setting_from_mi  # noqa: E402
from vcs_estim.estimators import EMA, estimate, scores  # noqa: E402
from vcs_ssl.utils import atomic_write_json, utc_now  # noqa: E402

KINDS = ("vcs", "js", "infonce", "nwj", "dv", "smile")
CHECKPOINTS = (250, 1000, 2000)
EVAL_BLOCKS, EVAL_BLOCK_N = 64, 512
P85_DIR = Path("/home/infres/yinwang/CS_QMI/outputs/P85_estim_benchmark")
SMILE_DIR = Path("/home/infres/yinwang/CS_QMI/outputs/P85_estim_benchmark_smile_fix")
DEV = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")


def p85_selected(I: float, seed: int, kind: str) -> tuple[float, float | None]:
    """(selected lr, stored native value) of neural:<kind>:inbatch in the matching P85 cell (SMILE from the corrected files)."""
    name = f"P85_gaussian_ds20_dt20_I{I:g}_N4096_B256_U2000_s{seed}.json"
    d = json.load(open((SMILE_DIR if kind == "smile" else P85_DIR) / name))
    rows = [r for r in d["rows"] if r["method"].startswith(f"neural:{kind}:inbatch:lr")]
    sel = [r for r in rows if r.get("selected")] or [min(rows, key=lambda r: r["select"]["risk"])]
    return float(sel[0]["lr"]), float(sel[0]["native"]["value"])


def hist(f: np.ndarray) -> dict:
    edges = np.linspace(-10, 10, 202); c, _ = np.histogram(np.clip(f, -10, 10), bins=edges)
    t = np.tanh(f); ct, _ = np.histogram(t, bins=np.linspace(-1, 1, 102))
    return {"f_edges": [-10, 10, 201], "f_counts": c.tolist(), "f_below": int((f < -10).sum()), "f_above": int((f > 10).sum()),
            "T_counts": ct.tolist(), "f_quantiles": dict(zip(["q01", "q05", "q50", "q95", "q99"], np.percentile(f, [1, 5, 50, 95, 99]).tolist())),
            "sat_T_0.95": float((np.abs(t) > 0.95).mean()), "sat_T_0.99": float((np.abs(t) > 0.99).mean())}


def eval_checkpoint(critic, kind, roles, tr, d_signal, seed, want_T: bool) -> tuple[dict, dict]:
    E = roles["EVAL"]; gen = torch.Generator().manual_seed(seed * 31 + 5)
    a, b, how, units, n_pos, n_neg = BM._native_on(critic, kind, "inbatch", E, DEV, BM.EVAL_BLOCK, gen)
    out = {"native_value": BM.combine(a, b, how), "own_truth": tr[BM.TARGET[kind]], "n_pos": n_pos, "n_neg": n_neg}
    out["signed_error"] = out["native_value"] - out["own_truth"]
    # 64 independent EVAL blocks (512 anchors each; in-batch negatives inside the block) -> eval variance conditional on the fit
    # (full EVAL = 32768 = 64 x 512 exactly; the smoke EVAL of 512 gets 8 blocks of 64)
    nb = EVAL_BLOCKS if E.n >= EVAL_BLOCKS * EVAL_BLOCK_N else 8; bn = EVAL_BLOCK_N if nb == EVAL_BLOCKS else E.n // nb
    vals = []
    with torch.no_grad():
        for j in range(nb):
            s, e = j * bn, (j + 1) * bn
            fp, fn = scores(critic, BM._dev(E.xp[s:e], DEV), BM._dev(E.yp[s:e], DEV), "inbatch", None)
            aa, bb, hh = BM.per_anchor(kind, fp.cpu(), fn.cpu()); vals.append(BM.combine(aa, bb, hh))
    vals = np.array(vals, dtype=float); fin = vals[np.isfinite(vals)]
    out["eval_blocks"] = {"n": int(len(vals)), "block_n": int(bn), "mean": float(fin.mean()) if len(fin) else None, "var_conditional_fit": float(fin.var(ddof=1)) if len(fin) > 1 else None,
                          "n_nonfinite": int((~np.isfinite(vals)).sum()), "values": [round(float(v), 6) for v in vals]}
    with torch.no_grad():
        n_h = min(8192, E.n)
        fp = torch.cat([critic.pairs(BM._dev(E.xp[s:e], DEV), BM._dev(E.yp[s:e], DEV)).double().cpu() for s, e in BM._blocks(E.n, 65536)])
        fn = torch.cat([critic.pairs(BM._dev(E.xq[s:e], DEV), BM._dev(E.yq[s:e], DEV)).double().cpu() for s, e in BM._blocks(E.n, 65536)])
    out["hist_P"] = hist(fp[:n_h].numpy()); out["hist_Q"] = hist(fn[:n_h].numpy())
    arrays = {}
    if kind in ("vcs", "js"):
        half = 1.0 if kind == "vcs" else 0.5
        eta_p, eta_n = BM.eta_of(roles["SETTING"], E.xp, E.yp, d_signal), BM.eta_of(roles["SETTING"], E.xq, E.yq, d_signal)
        out["posterior"] = BM.posterior_readouts(torch.tanh(half * fp), torch.tanh(half * fn), eta_p, eta_n, tr, gen, BM.BOOT)
        out["posterior"].pop("score_diagnostics", None)
        if want_T:
            arrays["T_P"] = torch.tanh(half * fp).numpy().astype(np.float16); arrays["T_Q"] = torch.tanh(half * fn).numpy().astype(np.float16)
    return out, arrays


def recorded_fit(kind: str, lr: float, roles: dict, d_total: int, B: int, updates: int, seed: int, every: int = 100):
    """train_neural with recording; identical RNG roles and update rule."""
    torch.manual_seed(seed * 7919 + 17); critic = JointMLP(d_total, d_total).to(DEV); opt = torch.optim.Adam(critic.parameters(), lr=lr)
    ema = EMA() if kind == "dv" else None; g = torch.Generator().manual_seed(seed * 1000 + 7); sg = torch.Generator().manual_seed(seed * 1000 + 11)
    F, S = roles["FIT"], roles["SELECT"]; Fx, Fy = BM._dev(F.xp, DEV), BM._dev(F.yp, DEV); n = F.n
    params = [p for p in critic.parameters()]

    def sel_risk():
        critic.eval(); a, b, how, _, _, _ = BM._native_on(critic, kind, "inbatch", S, DEV, BM.EVAL_BLOCK, sg); critic.train()
        v = BM.combine(a, b, how); return -v if math.isfinite(v) else float("inf")
    curve = [(0, sel_risk())]; best = (curve[0][1], 0, copy.deepcopy(critic.state_dict()))
    gnorm, lossv, nonfinite, ckpts = np.full(updates, np.nan), np.full(updates, np.nan), 0, {}
    t0 = time.time()
    for step in range(1, updates + 1):
        ip = torch.randint(0, n, (B,), generator=g).to(DEV)
        fp, fn = scores(critic, Fx[ip], Fy[ip], "inbatch", g)
        loss, _ = estimate(kind, fp, fn, ema=ema)
        if not torch.isfinite(loss):
            nonfinite += 1; opt.zero_grad(set_to_none=True)
        else:
            opt.zero_grad(set_to_none=True); loss.backward()
            gnorm[step - 1] = float(torch.sqrt(sum((p.grad.double() ** 2).sum() for p in params if p.grad is not None))); lossv[step - 1] = float(loss)
            opt.step()
        if step % every == 0:
            v = sel_risk(); curve.append((step, v))
            if v < best[0]:
                best = (v, step, copy.deepcopy(critic.state_dict()))
        if step in CHECKPOINTS:
            ckpts[step] = copy.deepcopy(critic.state_dict())
    info = {"select_curve": curve, "selected_update": best[1], "select_risk": best[0], "fit_seconds": time.time() - t0, "nonfinite_steps": nonfinite, "lr": lr}
    return critic, best[2], ckpts, gnorm, lossv, info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--I", type=float, required=True); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--kinds", default=",".join(KINDS)); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    d, N, B, U = 20, 4096, 256, 2000
    setting = setting_from_mi("gaussian", a.I, d)
    roles, sizes = BM.build_roles(setting, d, d, N, a.seed, a.smoke); roles["SETTING"] = setting
    tr = BM.truths(setting, d, roles["TRUTH"])
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"I{a.I:g}_s{a.seed}" + ("_smoke" if a.smoke else "")
    res = {"unit": {"setting": "gaussian", "d": d, "N": N, "B": B, "updates": U, "I": a.I, "seed": a.seed, "roles_hash": BM.roles_hash(roles), "sizes": sizes, "device": str(DEV), "utc": utc_now()},
           "truth": {k: v for k, v in tr.items() if not isinstance(v, (list, dict))}, "kinds": {}}
    arrays = {}
    for kind in a.kinds.split(","):
        lr, p85_value = p85_selected(a.I, a.seed, kind) if not a.smoke else (5e-4, None)
        updates = U if not a.smoke else 60
        critic, best_state, ckpts, gnorm, lossv, info = recorded_fit(kind, lr, roles, d, B, updates, a.seed)
        g = gnorm[np.isfinite(gnorm)]
        rec = {"lr_from_P85_selection": lr, "fit": info, "gradient_norm": {"mean": float(g.mean()), "sd": float(g.std(ddof=1)), "cv": float(g.std(ddof=1) / g.mean()) if g.mean() > 0 else None,
               "quantiles": dict(zip(["q05", "q50", "q95", "q99", "max"], np.percentile(g, [5, 50, 95, 99, 100]).tolist())),
               "last_half_mean": float(g[len(g) // 2:].mean()), "last_half_sd": float(g[len(g) // 2:].std(ddof=1)), "n_recorded": int(len(g))},
               "loss": {"last_half_mean": float(np.nanmean(lossv[len(lossv) // 2:])), "last_half_sd": float(np.nanstd(lossv[len(lossv) // 2:], ddof=1))}, "checkpoints": {}}
        arrays[f"{kind}/grad_norm"] = gnorm.astype(np.float32); arrays[f"{kind}/loss"] = lossv.astype(np.float32)
        states = {**{f"u{s}": st for s, st in ckpts.items()}, "select_best": best_state}
        for name, st in states.items():
            critic.load_state_dict(st); critic.eval()
            ev, arr = eval_checkpoint(critic, kind, roles, tr, d, a.seed, want_T=(name == "select_best"))
            rec["checkpoints"][name] = ev
            for k, v in arr.items():
                arrays[f"{kind}/{name}/{k}"] = v
        if p85_value is not None:
            v = rec["checkpoints"]["select_best"]["native_value"]
            rec["reproduction_vs_P85"] = {"p85_value": p85_value, "this_value": v, "abs_diff": abs(v - p85_value), "note": "P85 ran on GPU; CPU arithmetic differs over 2000 Adam steps (descriptive)"}
        torch.save({k: {kk: vv.cpu() for kk, vv in st.items()} for k, st in states.items()}, out_dir / f"weights_{tag}_{kind}.pt")
        res["kinds"][kind] = rec
        print(f"[{tag} {kind}] lr {lr:g} value {rec['checkpoints']['select_best']['native_value']:.4f} (truth {rec['checkpoints']['select_best']['own_truth']:.4f}) "
              f"gnorm cv {rec['gradient_norm']['cv']} nonfinite {info['nonfinite_steps']} eval-block sd {np.sqrt(rec['checkpoints']['select_best']['eval_blocks']['var_conditional_fit'] or 0):.4f} ({info['fit_seconds']:.0f}s)", flush=True)
        atomic_write_json(out_dir / f"{tag}.partial.json", res)
    np.savez_compressed(out_dir / f"{tag}.npz", **arrays); atomic_write_json(out_dir / f"{tag}.json", res)
    print(f"wrote {out_dir / tag}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
