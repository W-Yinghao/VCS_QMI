"""R4-E1 readings (reports/R4/R4_E1_SOLVER_PROTOCOL_FROZEN_20261011.yaml) over outputs/R4_E1/I6_N<N>_s<seed>.{json,npz}.

Primary (lr 1e-4 for every arm; mean ± sd over seeds 0-2 and per seed): EVAL posterior MSE and common J of VV, LL, LV, LLc at u1000 (the shared
JS start of LV / LLc), u1500, u2000 and each trajectory's CAL-selected state (argmax SELECT common J over its whole 0-2000 path).
Contrasts, paired by seed (95 % t over 3 seeds) and per seed by a paired EVAL-pair bootstrap (P and Q pools resampled independently, 1000 reps,
conditional on the fit — not a repeat of training):
  P1  LV − LLc   same start, same reset, same batches: the fit criterion alone (primary)
  P2  LV − VV    JS start + J vs direct VCS at the same total budget (primary)
  S1  LL − LLc   the reset alone
  S2  VV − LL    the direct gap (P151 at the selected state: VCS .050 vs JS .041)
Interpretation only: FIT common J (first 1024 anchors, in-block negatives) and each arm's own training loss; saturation; CAL-selected update;
curves every 100 updates.  Secondary: every arm at its CAL-selected lr (the lr whose CAL-selected state has the highest SELECT common J; branches
choose among their three second-half lrs from the same start), P1 / P2 recomputed.
Fixed wording (posterior MSE; J_eval reads the same with the sign reversed): P1 interval below 0 -> "continuing with J lowers the posterior MSE by
x relative to continuing with JS from the same start (this condition only)"; containing 0 -> "no detectable difference between continuing with J
and with JS"; above 0 -> "continuing with JS is better by x".  P2 below 0 -> "the JS start followed by J beats direct VCS by x at the same budget";
containing 0 -> "no detectable difference from direct VCS"; above 0 -> "direct VCS is better by x".  If LV − LLc raises the FIT common J
(training risk) at the CAL-selected state while P1 there does not lower the EVAL posterior MSE -> "the switch improves the training risk only; no
claim of solving the estimation problem" (also shown at u2000, descriptive).  The warm start is a solver path, not a new estimator, and is not
given a name.
    python scripts/r4_e1_aggregate.py [--N 4096]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vl1_10_aggregate import tint  # noqa: E402

IN = Path("/home/infres/yinwang/CS_QMI/outputs/R4_E1"); OUT = Path(__file__).resolve().parents[1] / "reports" / "R4"
SEEDS, LRS, LR0, ARMS = (0, 1, 2), (1e-4, 5e-4, 2e-3), 1e-4, ("VV", "LL", "LV", "LLc")
CONTRASTS = {"P1_LV_minus_LLc": ("LV", "LLc"), "P2_LV_minus_VV": ("LV", "VV"), "S1_LL_minus_LLc": ("LL", "LLc"), "S2_VV_minus_LL": ("VV", "LL")}
REPS = 1000


def key(arm, lr):
    return f"{arm}_lr{lr:g}"


def at(tr: dict, state: str) -> dict:
    """EVAL readouts at a key state + the curve entry (training risk, saturation) of the same update."""
    ks = tr["key_states"][state]; c = next(x for x in tr["curve"] if x["update"] == ks["update"])
    return {"update": ks["update"], "posterior_mse": ks["posterior_mse"], "J_eval": ks["J_eval"], "J_eval_minus_S": ks["J_eval_minus_S"],
            "fit_common_J": c["fit_common_J_first1024"], "select_common_J": c["select_common_J"], "train_loss_last": c["train_loss_last"],
            "sat_P": c["sat_P_0.95"], "sat_Q": c["sat_Q_0.95"]}


def boot_diff(z, kx: str, ky: str, state: str, seed: int) -> dict:
    """Paired EVAL-pair bootstrap of X − Y in posterior MSE and common J (P and Q pools resampled independently, as BM.boot_units)."""
    ep, eq = z["eta_P"].astype(np.float64), z["eta_Q"].astype(np.float64)
    xp, xq, yp, yq = (z[f"{k}/{state}/{s}"].astype(np.float64) for k, s in ((kx, "T_P"), (kx, "T_Q"), (ky, "T_P"), (ky, "T_Q")))
    mp, mq = 0.5 * ((xp - ep) ** 2 - (yp - ep) ** 2), 0.5 * ((xq - eq) ** 2 - (yq - eq) ** 2)
    jp, jq = (xp - xp ** 2 / 2) - (yp - yp ** 2 / 2), (-xq - xq ** 2 / 2) - (-yq - yq ** 2 / 2)
    rng = np.random.default_rng(1000 + seed); bm, bj = np.empty(REPS), np.empty(REPS)
    for r in range(REPS):
        ip, iq = rng.integers(0, len(ep), len(ep)), rng.integers(0, len(eq), len(eq)); bm[r] = mp[ip].mean() + mq[iq].mean(); bj[r] = jp[ip].mean() + jq[iq].mean()
    return {"posterior_mse": {"diff": float(mp.mean() + mq.mean()), "boot95": [float(np.quantile(bm, 0.025)), float(np.quantile(bm, 0.975))]},
            "J_eval": {"diff": float(jp.mean() + jq.mean()), "boot95": [float(np.quantile(bj, 0.025)), float(np.quantile(bj, 0.975))]}}


def word(name: str, t: dict) -> str:
    ci = t.get("ci95")
    if ci is None:
        return "fewer than 2 seeds: no reading"
    lo, hi = ci; x = abs(t["mean"])
    if name.startswith("P1"):
        return (f"continuing with J lowers the posterior MSE by {x:.4f} relative to continuing with JS from the same start (this condition only)" if hi < 0 else
                "no detectable difference between continuing with J and with JS" if lo <= 0 <= hi else f"continuing with JS is better by {x:.4f}")
    if name.startswith("P2"):
        return (f"the JS start followed by J beats direct VCS by {x:.4f} at the same budget" if hi < 0 else
                "no detectable difference from direct VCS" if lo <= 0 <= hi else f"direct VCS is better by {x:.4f}")
    return "descriptive"


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--N", type=int, default=4096); ap.add_argument("--in-dir", default=str(IN)); ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--smoke", action="store_true", help="read the *_smoke units (code check; write --out-dir outside reports/)"); a = ap.parse_args()
    runs, Z, missing = {}, {}, []; sfx = "_smoke" if a.smoke else ""
    for s in SEEDS:
        f = Path(a.in_dir) / f"I6_N{a.N}_s{s}{sfx}.json"
        if not f.exists():
            missing.append(f.name); continue
        runs[s] = json.load(open(f)); Z[s] = np.load(Path(a.in_dir) / f"I6_N{a.N}_s{s}{sfx}.npz")
    if not runs:
        print("no units", missing); return 2
    un = runs[min(runs)]["unit"]; b, u = un["branch_update"], un["updates"]; mid = b + (u - b) // 2
    STATES = tuple(st for st in (f"u{b}", f"u{mid}", f"u{u}", "cal_selected") if st in runs[min(runs)]["trajectories"][key("LL", LR0)]["key_states"])
    res = {"protocol": "reports/R4/R4_E1_SOLVER_PROTOCOL_FROZEN_20261011.yaml", "N": a.N, "seeds": sorted(runs), "missing": missing,
           "gates": {s: r["gates"] for s, r in runs.items()}, "primary": {}, "contrasts": {}, "secondary_cal_lr": {}, "curves": {}, "costs": {s: r["costs"] for s, r in runs.items()}}
    for arm in ARMS:
        res["primary"][arm] = {st: {"per_seed": {s: at(runs[s]["trajectories"][key(arm, LR0)], st) for s in runs}} for st in STATES}
        for st in STATES:
            ps = res["primary"][arm][st]["per_seed"]
            res["primary"][arm][st]["summary"] = {q: tint([ps[s][q] for s in ps]) for q in ("posterior_mse", "J_eval", "fit_common_J", "sat_P", "sat_Q")}
            res["primary"][arm][st]["summary"]["update"] = [ps[s]["update"] for s in ps]
    for name, (x, y) in CONTRASTS.items():
        res["contrasts"][name] = {}
        for st in STATES:
            d = {q: [res["primary"][x][st]["per_seed"][s][q] - res["primary"][y][st]["per_seed"][s][q] for s in runs] for q in ("posterior_mse", "J_eval", "fit_common_J")}
            res["contrasts"][name][st] = {"seed_t": {q: tint(v) for q, v in d.items()}, "per_seed_boot": {s: boot_diff(Z[s], key(x, LR0), key(y, LR0), st, s) for s in runs}}
            res["contrasts"][name][st]["reading_posterior_mse"] = word(name, res["contrasts"][name][st]["seed_t"]["posterior_mse"])
    res["training_risk_rule"] = {}                                                             # primary at the CAL-selected state; the end of training descriptive
    for st in ("cal_selected", f"u{u}"):
        c = res["contrasts"]["P1_LV_minus_LLc"][st]["seed_t"]; up = c["fit_common_J"]["mean"] > 0; ev = c["posterior_mse"].get("ci95")
        res["training_risk_rule"][st] = {"fit_common_J_LV_minus_LLc": c["fit_common_J"], "eval_posterior_mse_LV_minus_LLc": c["posterior_mse"],
                                         "reading": "the switch improves the training risk only; no claim of solving the estimation problem" if up and (ev is None or ev[1] >= 0) else "rule not triggered"}
    for arm in ARMS:                                                                           # secondary: CAL-selected lr per arm
        pick = {}
        for s in runs:
            cands = [(runs[s]["trajectories"][key(arm, lr)]["cal_selected"]["select_common_J"], -i, lr) for i, lr in enumerate(LRS) if key(arm, lr) in runs[s]["trajectories"]]
            lr = max(cands)[2]; pick[s] = {"lr": lr, **at(runs[s]["trajectories"][key(arm, lr)], "cal_selected")}
        res["secondary_cal_lr"][arm] = {"per_seed": pick, "summary": {q: tint([pick[s][q] for s in pick]) for q in ("posterior_mse", "J_eval")}}
    for name in ("P1_LV_minus_LLc", "P2_LV_minus_VV"):
        x, y = CONTRASTS[name]; px, py = res["secondary_cal_lr"][x]["per_seed"], res["secondary_cal_lr"][y]["per_seed"]
        v = tint([px[s]["posterior_mse"] - py[s]["posterior_mse"] for s in runs])
        res["secondary_cal_lr"][name] = {"posterior_mse": v, "per_seed_boot": {s: boot_diff(Z[s], key(x, px[s]["lr"]), key(y, py[s]["lr"]), "cal_selected", s) for s in runs},
                                         "reading": word(name, v)}
    for arm in ARMS:
        ups = [c["update"] for c in runs[min(runs)]["trajectories"][key(arm, LR0)]["curve"]]
        res["curves"][arm] = {q: [float(np.mean([next(c[q] for c in runs[s]["trajectories"][key(arm, LR0)]["curve"] if c["update"] == u) for s in runs])) for u in ups]
                              for q in ("posterior_mse", "J_eval", "select_common_J", "fit_common_J_first1024", "sat_P_0.95", "sat_Q_0.95")}
        res["curves"][arm]["update"] = ups
    suf = ("" if a.N == 4096 else f"_N{a.N}") + sfx; od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(od / f"R4_E1_results{suf}.json", "w"), indent=1, default=float)
    f4 = lambda t: "—" if t is None else f"{t['mean']:.4f}" + (f" ± {t['sd']:.4f}" if t.get("sd") is not None else "")
    fc = lambda t: f"{t['mean']:+.4f}" + (f" [{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}]" if t.get("ci95") else " (1 seed)")
    L = [f"# R4-E1 — solver check (Gaussian d 20, I 6, N {a.N}, B 256, U {u}; lr 1e-4){' — SMOKE (code check only)' if a.smoke else ''}: results", "",
         f"Seeds {sorted(runs)}; missing {missing or 'none'}.  Gates: " + "; ".join(f"s{s}: roles hash {g.get('roles_hash', {}).get('pass')}, P152 reproduction {g.get('p152_reproduction', {}).get('pass')}" for s, g in res["gates"].items()), "",
         "| arm | state | EVAL posterior MSE | EVAL common J | FIT common J (training risk) | sat P / Q | updates |", "|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        for st in STATES:
            sm = res["primary"][arm][st]["summary"]
            L.append(f"| {arm} | {st} | {f4(sm['posterior_mse'])} | {f4(sm['J_eval'])} | {f4(sm['fit_common_J'])} | {sm['sat_P']['mean']:.3f} / {sm['sat_Q']['mean']:.3f} | {sm['update']} |")
    L += ["", "| contrast | state | posterior MSE (seed t) | common J (seed t) | FIT common J (seed t) | per-seed EVAL-pair bootstrap, posterior MSE | reading |", "|---|---|---|---|---|---|---|"]
    for name in CONTRASTS:
        for st in STATES:
            c = res["contrasts"][name][st]; bs = "; ".join(f"s{s} {v['posterior_mse']['diff']:+.4f} [{v['posterior_mse']['boot95'][0]:+.4f}, {v['posterior_mse']['boot95'][1]:+.4f}]" for s, v in c["per_seed_boot"].items())
            L.append(f"| {name} | {st} | {fc(c['seed_t']['posterior_mse'])} | {fc(c['seed_t']['J_eval'])} | {fc(c['seed_t']['fit_common_J'])} | {bs} | {c['reading_posterior_mse']} |")
    L += ["", "Training-risk rule: " + "; ".join(f"{st}: {v['reading']}" for st, v in res["training_risk_rule"].items()) + ".", "", "Secondary (each arm at its CAL-selected lr):", "",
          "| arm | lr chosen per seed | EVAL posterior MSE | EVAL common J |", "|---|---|---|---|"]
    for arm in ARMS:
        sc = res["secondary_cal_lr"][arm]; L.append(f"| {arm} | {[sc['per_seed'][s]['lr'] for s in sc['per_seed']]} | {f4(sc['summary']['posterior_mse'])} | {f4(sc['summary']['J_eval'])} |")
    for name in ("P1_LV_minus_LLc", "P2_LV_minus_VV"):
        L.append(f"| {name} | — | {fc(res['secondary_cal_lr'][name]['posterior_mse'])} | {res['secondary_cal_lr'][name]['reading']} |")
    ups = res["curves"]["LV"]["update"]; show = [x for x in ups if x in (0, b // 2) or x >= b]
    L += ["", "EVAL posterior MSE curves (mean over seeds; lr 1e-4):", "", "| update | " + " | ".join(ARMS) + " |", "|---|---|---|---|---|"]
    for u in show:
        row = [res["curves"][arm]["posterior_mse"][res["curves"][arm]["update"].index(u)] if u in res["curves"][arm]["update"] else None for arm in ARMS]
        L.append(f"| {u} | " + " | ".join("—" if v is None else f"{v:.4f}" for v in row) + " |")
    (od / f"R4_E1_tables{suf}.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
