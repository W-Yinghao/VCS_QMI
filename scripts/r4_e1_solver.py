"""R4-E1 (r4 package docs/02 §R4-E1, docs/05 §3) — solver check on ONE existing oracle condition: from the same JS start, does continuing with
the fit criterion J instead of JS change the common risk?  Protocol: reports/R4/R4_E1_SOLVER_PROTOCOL_FROZEN_20261011.yaml.

Condition = the P85 / P152 cell with an exact oracle: Gaussian, d_signal = d_total = 20, I = 6 nats, N = 4096, B = 256, in-batch negatives,
JointMLP critic (concat -> 256 ReLU -> 256 ReLU -> 1), U = 2000 Adam updates per trajectory, seeds 0-2.  Every trajectory starts from the same
initial critic (torch seed s*7919 + 17) and draws the same minibatch stream (generator s*1000 + 7); roles = P85's build_roles (FIT / SELECT /
EVAL / TRUTH), checked against P152's roles hash.  Trajectories, each on the P85 lr grid {1e-4, 5e-4, 2e-3} (primary lr 1e-4 = P85's selection
for both objectives at this cell and every seed):
  VV   direct VCS: J on T = tanh f, 2000 updates
  LL   direct JS (Deep-InfoMax softplus form on the logit u), 2000 updates, continuous Adam; common coordinate T = tanh(u / 2)
  LV   from LL(lr 1e-4) at update 1000: J on T = tanh(u / 2) for 1000 updates
  LLc  from the same state: JS for 1000 updates (the control with the identical reset)
Optimizer-state convention, the same for both branches: fresh Adam at the branch point; the minibatch stream continues from the branch point's
generator state (so LL, LV and LLc see identical second-half batches).
Every 100 updates: SELECT common J (in-batch, 1024-anchor blocks: the CAL rule; equals P85's native VCS SELECT value when T = tanh f), SELECT
native value (P85's rule, reproduction only), EVAL common J and posterior MSE (product Q pairs, oracle eta), FIT training risk (the arm's own
minibatch loss, mean of the last 100) and common J on the first 1024 FIT anchors (in-block negatives), saturation |T| > 0.95 on EVAL P / Q, and
the critic-gradient norm of every update.  At the key states u1000, u1500, u2000 and each trajectory's CAL-selected state (argmax SELECT common J
over its whole 0-2000 path, earliest on ties): BM.posterior_readouts with BOOT = 200 and the per-pair T arrays (npz, float32).
Gates (non-smoke; abort before the branches): roles hash = P152's; direct VCS / JS at lr 1e-4 posterior MSE at u1000 and u2000 within 0.002 of
P152's records (weight differences recorded).  Outputs: <out>/I6_N4096_s<seed>.{json,npz} + weights_I6_N4096_s<seed>.pt.
    python scripts/r4_e1_solver.py --seed 0 --out /home/infres/yinwang/CS_QMI/outputs/R4_E1 [--N 4096] [--smoke]
"""
from __future__ import annotations

import argparse
import copy
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
from vcs_estim.estimators import estimate, scores  # noqa: E402
from vcs_ssl.utils import atomic_write_json, utc_now  # noqa: E402

PROTOCOL = "reports/R4/R4_E1_SOLVER_PROTOCOL_FROZEN_20261011.yaml"
P152 = Path("/home/infres/yinwang/CS_QMI/outputs/P152_e1")
D, B, I_LEVEL, U, BRANCH, EVERY = 20, 256, 6.0, 2000, 1000, 100
LRS, LR_PRIMARY, GATE_TOL, FIT_ANCHORS = BM.LRS, 1e-4, 0.002, 1024
DEV = torch.device("cpu")
ARMS = {"VV": ("J", 1.0, "V->V direct VCS"), "LL": ("js", 0.5, "L->L direct JS, continuous Adam"),
        "LV": ("J", 0.5, "L->V: J on tanh(u/2) from LL(lr 1e-4)@1000, Adam reset"), "LLc": ("js", 0.5, "L->L control: JS from LL(lr 1e-4)@1000, Adam reset")}


def loss_fn(objective: str, half: float, fp: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
    """'J': the VCS loss -J on T = tanh(half * f) (exactly estimate('vcs') when half = 1); 'js': the Deep-InfoMax loss on the raw logit."""
    if objective == "J":
        return estimate("vcs", fp, fn)[0] if half == 1.0 else estimate("vcs", half * fp, half * fn)[0]
    if objective == "js":
        return estimate("js", fp, fn)[0]
    raise ValueError(objective)


@torch.no_grad()
def inbatch_values(critic, role, half: float, native: str | None, n_anchor: int | None = None, block: int = BM.EVAL_BLOCK) -> tuple[float, float | None]:
    """(common J of T = tanh(half * f), native value of `native`) with in-batch negatives over blocks of `block` anchors (BM._native_on's
    blocking); n_anchor restricts to the first anchors (the FIT training-risk readout)."""
    n = role.n if n_anchor is None else min(n_anchor, role.n); A, Bq, An, Bn = [], [], [], []
    for s, e in BM._blocks(n, block):
        if e - s <= 8:
            continue
        fp, fn = scores(critic, BM._dev(role.xp[s:e], DEV), BM._dev(role.yp[s:e], DEV), "inbatch", None)
        fp, fn = fp.double().cpu(), fn.double().cpu()
        a, b, _ = BM.per_anchor("vcs", half * fp, half * fn); A.append(a); Bq.append(b)
        if native is not None:
            a, b, _ = BM.per_anchor(native, fp, fn); An.append(a); Bn.append(b)
    common = BM.combine(torch.cat(A), torch.cat(Bq), "sum")
    return common, (BM.combine(torch.cat(An), torch.cat(Bn), "sum") if native is not None else None)


@torch.no_grad()
def eval_T(critic, E, half: float) -> tuple[torch.Tensor, torch.Tensor]:
    fp = torch.cat([critic.pairs(BM._dev(E.xp[s:e], DEV), BM._dev(E.yp[s:e], DEV)).double().cpu() for s, e in BM._blocks(E.n, 65536)])
    fn = torch.cat([critic.pairs(BM._dev(E.xq[s:e], DEV), BM._dev(E.yq[s:e], DEV)).double().cpu() for s, e in BM._blocks(E.n, 65536)])
    return torch.tanh(half * fp), torch.tanh(half * fn)


def eval_point(tp, tn, eta_p, eta_n, S: float) -> dict:
    """Point readouts of BM.posterior_readouts (same formulas) + saturation on the EVAL P / Q pools."""
    ap, aq = tp - 0.5 * tp ** 2, -tn - 0.5 * tn ** 2; rp, rq = (tp - eta_p) ** 2, (tn - eta_n) ** 2; J = float(ap.mean() + aq.mean())
    return {"J_eval": J, "J_eval_minus_S": J - S, "posterior_mse": float(0.5 * rp.mean() + 0.5 * rq.mean()),
            "sat_P_0.95": float((tp.abs() > 0.95).double().mean()), "sat_Q_0.95": float((tn.abs() > 0.95).double().mean())}


def select_index(curve: list[dict]) -> int:
    """The CAL rule: argmax of the SELECT common J over the recorded path, earliest on ties (non-finite values never selected)."""
    best, k = -math.inf, 0
    for i, c in enumerate(curve):
        v = c["select_common_J"]
        if math.isfinite(v) and v > best:
            best, k = v, i
    return k


class Ctx:
    def __init__(self, roles, tr, eta, smoke):
        self.roles, self.tr, self.eta, self.smoke = roles, tr, eta, smoke
        self.every, self.u, self.branch = (10, 60, 30) if smoke else (EVERY, U, BRANCH)

    def readout(self, critic, half: float, native: str) -> dict:
        critic.eval()
        sel_c, sel_n = self.inbatch(critic, "SELECT", half, native)
        fit_c, _ = inbatch_values(critic, self.roles["FIT"], half, None, n_anchor=FIT_ANCHORS)
        tp, tn = eval_T(critic, self.roles["EVAL"], half); critic.train()
        return {"select_common_J": sel_c, "select_native_value": sel_n, "fit_common_J_first1024": fit_c, **eval_point(tp, tn, *self.eta, self.tr["S"])}

    def inbatch(self, critic, role, half, native):
        return inbatch_values(critic, self.roles[role], half, native)


def train(ctx: Ctx, critic, opt, g, objective: str, half: float, native: str, start: int, stop: int, curve: list, states: dict, gn: list, lv: list,
          on_step=None) -> int:
    """Advance from update `start` to `stop` (P152's recorded loop: same draws, same loss call, same non-finite rule); readout every ctx.every."""
    F = ctx.roles["FIT"]; Fx, Fy = BM._dev(F.xp, DEV), BM._dev(F.yp, DEV); n = F.n; params = list(critic.parameters()); nonfinite = 0
    for step in range(start + 1, stop + 1):
        ip = torch.randint(0, n, (B,), generator=g).to(DEV)
        fp, fn = scores(critic, Fx[ip], Fy[ip], "inbatch", g)
        loss = loss_fn(objective, half, fp, fn)
        if not torch.isfinite(loss):
            nonfinite += 1; opt.zero_grad(set_to_none=True); gn.append(float("nan")); lv.append(float("nan"))
        else:
            opt.zero_grad(set_to_none=True); loss.backward()
            gn.append(float(torch.sqrt(sum((p.grad.double() ** 2).sum() for p in params if p.grad is not None)))); lv.append(float(loss)); opt.step()
        if step % ctx.every == 0:
            r = ctx.readout(critic, half, native); w = [x for x in lv[-ctx.every:] if math.isfinite(x)]; q = [x for x in gn[-ctx.every:] if math.isfinite(x)]
            curve.append({"update": step, "train_loss_last": float(np.mean(w)) if w else None, "grad_norm_last": float(np.mean(q)) if q else None, **r})
            states[step] = copy.deepcopy(critic.state_dict())
        if on_step is not None:
            on_step(step, critic, opt, g)
    return nonfinite


def checksum(state: dict) -> float:
    return float(sum(v.double().sum() * (i + 1) for i, v in enumerate(state.values())))


def key_states(ctx: Ctx, critic, half: float, curve: list, states: dict, tag: str, seed: int, arrays: dict) -> dict:
    """Bootstrap readouts and per-pair T arrays at u_branch, u_mid, u_end and the CAL-selected state."""
    k = select_index(curve); u_sel = curve[k]["update"]; mid = ctx.branch + (ctx.u - ctx.branch) // 2
    want = {f"u{ctx.branch}": ctx.branch, f"u{mid}": mid, f"u{ctx.u}": ctx.u, "cal_selected": u_sel}; out = {}
    E = ctx.roles["EVAL"]
    for name, u in want.items():
        if u not in states:
            continue
        critic.load_state_dict(states[u]); critic.eval(); tp, tn = eval_T(critic, E, half); gen = torch.Generator().manual_seed(seed * 31 + 5)
        pr = BM.posterior_readouts(tp, tn, *ctx.eta, ctx.tr, gen, BM.BOOT); pr.pop("score_diagnostics", None); pr["update"] = u
        out[name] = pr; arrays[f"{tag}/{name}/T_P"] = tp.numpy().astype(np.float32); arrays[f"{tag}/{name}/T_Q"] = tn.numpy().astype(np.float32)
    critic.train()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--N", type=int, default=4096, help="4096 = the P152 cell (primary); one other level only as the separate N extension")
    ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    setting = setting_from_mi("gaussian", I_LEVEL, D)
    roles, sizes = BM.build_roles(setting, D, D, a.N, a.seed, a.smoke); tr = BM.truths(setting, D, roles["TRUTH"])
    E = roles["EVAL"]; eta = (BM.eta_of(setting, E.xp, E.yp, D), BM.eta_of(setting, E.xq, E.yq, D)); ctx = Ctx(roles, tr, eta, a.smoke)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); tag = f"I{I_LEVEL:g}_N{a.N}_s{a.seed}" + ("_smoke" if a.smoke else "")
    res = {"protocol": PROTOCOL, "unit": {"setting": "gaussian", "d": D, "N": a.N, "B": B, "I": I_LEVEL, "seed": a.seed, "updates": ctx.u, "branch_update": ctx.branch,
           "every": ctx.every, "lr_grid": list(LRS), "lr_primary": LR_PRIMARY, "optimizer_state_convention": "fresh Adam at the branch point for both branches; batch stream continues",
           "roles_hash": BM.roles_hash(roles), "sizes": sizes, "device": str(DEV), "threads": torch.get_num_threads(), "utc": utc_now(), "smoke": a.smoke},
           "truth": {k: v for k, v in tr.items() if not isinstance(v, (list, dict))}, "gates": {}, "trajectories": {}}
    arrays = {"eta_P": eta[0].numpy().astype(np.float32), "eta_Q": eta[1].numpy().astype(np.float32)}; weights = {}
    ref = None if (a.smoke or a.N != 4096) else json.load(open(P152 / f"I{I_LEVEL:g}_s{a.seed}.json"))
    if ref is not None:
        ok = ref["unit"]["roles_hash"] == res["unit"]["roles_hash"]; res["gates"]["roles_hash"] = {"p152": ref["unit"]["roles_hash"], "pass": ok}
        if not ok:
            atomic_write_json(out / f"{tag}.gate_failed.json", res); print(f"[{tag}] ABORT: roles hash differs from P152", flush=True); return 3
    branch = {}

    def run(name: str, lr: float, start_state=None, g_state=None) -> None:
        objective, half, desc = ARMS[name]; native = "vcs" if (objective == "J" and half == 1.0) else "js"; t0 = time.time()
        torch.manual_seed(a.seed * 7919 + 17); critic = JointMLP(D, D).to(DEV)
        g = torch.Generator().manual_seed(a.seed * 1000 + 7); curve, states, gn, lv = [], {}, [], []
        if start_state is None:                                                              # direct: from the initial critic
            opt = torch.optim.Adam(critic.parameters(), lr=lr); curve.append({"update": 0, "train_loss_last": None, "grad_norm_last": None, **ctx.readout(critic, half, native)})
            states[0] = copy.deepcopy(critic.state_dict()); start = 0
            if name == "LL" and lr == LR_PRIMARY:                                             # the shared branch point (weights + generator state)
                def keep(step, c, o, gg):
                    if step == ctx.branch:
                        branch.update(state=copy.deepcopy(c.state_dict()), g=gg.get_state().clone(), curve=None)
            else:
                keep = None
        else:                                                                                # branch: identical start, fresh Adam (the convention)
            critic.load_state_dict(start_state); g.set_state(g_state); opt = torch.optim.Adam(critic.parameters(), lr=lr); start = ctx.branch; keep = None
            curve += copy.deepcopy(branch["curve"]); states.update({u: s for u, s in branch["states"].items()}); gn += branch["gn"]; lv += branch["lv"]
        nonfinite = train(ctx, critic, opt, g, objective, half, native, start, ctx.u, curve, states, gn, lv, keep)
        k = f"{name}_lr{lr:g}"
        rec = {"arm": name, "description": desc, "lr": lr, "fit_criterion": {"J": "J (VCS loss) on T = tanh(%g f)" % half, "js": "JS (Deep-InfoMax)"}[objective],
               "evaluation_coordinate": "T = tanh f" if half == 1.0 else "T = tanh(u/2)", "start": "initial critic" if start_state is None else f"LL_lr{LR_PRIMARY:g}@{ctx.branch}",
               "start_checksum": checksum(start_state) if start_state is not None else checksum(states[0]), "curve": curve, "nonfinite_steps": nonfinite,
               "updates_this_trajectory": ctx.u - start, "seconds": time.time() - t0}
        sel = select_index(curve); rec["cal_selected"] = {"update": curve[sel]["update"], "select_common_J": curve[sel]["select_common_J"]}
        rec["key_states"] = key_states(ctx, critic, half, curve, states, k, a.seed, arrays)
        arrays[f"{k}/grad_norm"] = np.array(gn, dtype=np.float32); arrays[f"{k}/train_loss"] = np.array(lv, dtype=np.float32)
        weights[k] = {"final": {kk: v.clone() for kk, v in states[ctx.u].items()}, "cal_selected": {kk: v.clone() for kk, v in states[curve[sel]["update"]].items()}}
        if start_state is None:
            weights[k]["u_branch"] = {kk: v.clone() for kk, v in states[ctx.branch].items()}
        if name == "LL" and lr == LR_PRIMARY:
            branch.update(curve=[c for c in curve if c["update"] <= ctx.branch], states={u: s for u, s in states.items() if u <= ctx.branch},
                          gn=gn[:ctx.branch], lv=lv[:ctx.branch]); weights["branch_point"] = {kk: v.clone() for kk, v in branch["state"].items()}
        res["trajectories"][k] = rec
        print(f"[{tag} {k}] CAL-selected u{rec['cal_selected']['update']}: EVAL posterior MSE {rec['key_states']['cal_selected']['posterior_mse']:.4f} "
              f"J {rec['key_states']['cal_selected']['J_eval']:.4f}; u{ctx.u} MSE {curve[-1]['posterior_mse']:.4f} ({rec['seconds']:.0f}s)", flush=True)
        atomic_write_json(out / f"{tag}.partial.json", res)

    for name in ("VV", "LL"):
        run(name, LR_PRIMARY)
    if ref is not None:                                                                       # gate 2: reproduction of P152 at the primary lr
        rep = {}
        for name, kind in (("VV", "vcs"), ("LL", "js")):
            tr_ = res["trajectories"][f"{name}_lr{LR_PRIMARY:g}"]; wref = torch.load(P152 / f"weights_I{I_LEVEL:g}_s{a.seed}_{kind}.pt", map_location="cpu", weights_only=False)
            for u in (1000, 2000):
                mine = next(c for c in tr_["curve"] if c["update"] == u)["posterior_mse"]; theirs = ref["kinds"][kind]["checkpoints"][f"u{u}"]["posterior"]["posterior_mse"]
                st = weights[f"{name}_lr{LR_PRIMARY:g}"]["final" if u == 2000 else "u_branch"]
                dw = max(float((st[kk] - wref[f"u{u}"][kk]).abs().max()) for kk in st)
                rep[f"{kind}_u{u}"] = {"this": mine, "p152": theirs, "abs_diff": abs(mine - theirs), "max_abs_weight_diff": dw, "pass": abs(mine - theirs) <= GATE_TOL}
        res["gates"]["p152_reproduction"] = {"tolerance_posterior_mse": GATE_TOL, "checks": rep, "pass": all(v["pass"] for v in rep.values())}
        if not res["gates"]["p152_reproduction"]["pass"]:
            atomic_write_json(out / f"{tag}.gate_failed.json", res); print(f"[{tag}] ABORT: P152 reproduction gate failed {rep}", flush=True); return 3
        print(f"[{tag}] gates pass: roles hash; P152 reproduction {[(k, round(v['abs_diff'], 5), v['max_abs_weight_diff']) for k, v in rep.items()]}", flush=True)
    for lr in LRS:                                                                            # both branches, every lr of the grid, identical start
        for name in ("LV", "LLc"):
            run(name, lr, branch["state"], branch["g"])
    for lr in LRS:
        if lr != LR_PRIMARY:
            for name in ("VV", "LL"):
                run(name, lr)
    res["costs"] = {"updates_total": int(sum(t["updates_this_trajectory"] for t in res["trajectories"].values())),
                    "seconds_total": float(sum(t["seconds"] for t in res["trajectories"].values())), "n_trajectories": len(res["trajectories"])}
    torch.save(weights, out / f"weights_{tag}.pt"); np.savez_compressed(out / f"{tag}.npz", **arrays); atomic_write_json(out / f"{tag}.json", res)
    (out / f"{tag}.partial.json").unlink(missing_ok=True); print(f"[{tag}] wrote {out / tag}.json ({res['costs']})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
