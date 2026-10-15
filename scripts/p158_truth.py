"""P158 — nested increments with truth on the X11 A2 joint-vs-product chain (project map 2026-10-14 §11 B.2; the P118 §6.1 design with the
conditional-binary oracle replaced by a joint (P) vs product (Q) Gaussian-pair oracle; scripts/p158_common.py).

    python scripts/p158_truth.py --out reports/P158/truth --conds null,weak,moderate [--repeats 5] [--n 3000] [--steps 300] [--smoke]

Chain B (20 + 20) ⊇ M (10 + 10: only irrelevant coordinates dropped, true increment 0) ⊇ A (5 + 5: half the signal pairs dropped, true increment
S_M − S_A > 0 on the VCS scale, G_M − G_A > 0 on the JS scale).  Per repeat (fresh FIT and EVAL draws of n P units each, one non-zero cyclic shift
per draw for the Q pairs) and objective (VCS −J, matched JS), the fitting variants:
    indep    every set fitted independently (P102 / P118 baseline)
    nested   A independent; M = frozen A∘map + zero-initialised residual; B = frozen M∘map + residual
    zero     constant-zero critic on every set (control: perfect consistency residuals, wrong increments)
No exact product term exists here (continuous inputs): fitting and every readout use the sampled Q pairs of the draw.  Every critic is read out
twice on EVAL — the common squared score (J, ΔJ, D_T, r_BA, R_orth, posterior MSE of T vs the exact eta; truth = S differences) and the JS native
score (G, ΔG, Bregman D_JS, r, R_orth_js; truth = G differences).  Per repeat a paired bootstrap over EVAL units (1 000 resamples) gives CIs of
r_BA (both steps) and R_orth for both readouts.  Nothing on EVAL is clipped, made monotone or orthogonalised; EVAL selects nothing.
Output: <out>_<cond>.json per condition (skip-if-done), partial file while running.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
import p158_common as C  # noqa: E402
from vcs_estim import increments as INC  # noqa: E402
from vcs_measure import nested_joint as NJ  # noqa: E402
from vcs_ssl.utils import atomic_write_json, utc_now  # noqa: E402

CONDS = {"null": 0.0, "weak": 0.13, "moderate": 0.22}      # pre-named from the oracle only (S_M ≈ 0, 0.04, 0.11; see the prereg)
VARIANTS = ("indep", "nested", "zero")
CMAP = {("M", "A"): lambda x, y: (x[:, :C.KEEP_A], y[:, :C.KEEP_A]), ("B", "M"): lambda x, y: (x[:, :C.K_SIG], y[:, :C.K_SIG])}
READOUTS = ("common", "js_native")
BOOT_REPS = 1000
DEVICE = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")


def chain_readouts(f: dict) -> dict:
    """Both readouts of a chain {set: (fp, fq) logits}: common (on T = tanh f) and JS native (on f)."""
    tp, tq = [np.tanh(f[k][0]) for k in C.CHAIN], [np.tanh(f[k][1]) for k in C.CHAIN]
    com = INC.chain_summary(tp, tq, C.CHAIN); js = C.js_chain_summary([f[k][0] for k in C.CHAIN], [f[k][1] for k in C.CHAIN], C.CHAIN)
    tot_c = INC.increment(tp[0], tq[0], tp[-1], tq[-1]); tot_j = js["steps"]  # total B->A for JS below
    inc_c = {s: com["steps"][s]["delta_J"] for s in ("B->M", "M->A")} | {"B->A": tot_c["delta_J"]}
    dt_c = {s: com["steps"][s]["D_T"] for s in ("B->M", "M->A")} | {"B->A": tot_c["D_T"]}
    r_c = {s: com["steps"][s]["r_BA"] for s in ("B->M", "M->A")} | {"B->A": tot_c["r_BA"]}
    gb, ga = js["G"]["B"], js["G"]["A"]; d_ba = C.d_js(f["B"][0], f["B"][1], f["A"][0], f["A"][1])
    inc_j = {s: tot_j[s]["delta_G"] for s in ("B->M", "M->A")} | {"B->A": gb - ga}
    dt_j = {s: tot_j[s]["D_JS"] for s in ("B->M", "M->A")} | {"B->A": d_ba}
    r_j = {s: tot_j[s]["r_BA"] for s in ("B->M", "M->A")} | {"B->A": (gb - ga) - d_ba}
    viol_c = int(sum(com["J"][C.CHAIN[i]] < com["J"][C.CHAIN[i + 1]] for i in range(2)))
    return {"common": {"value": com["J"], "increment": inc_c, "D": dt_c, "r": r_c, "R_orth": com["R_orth"], "ordering_violations": viol_c},
            "js_native": {"value": js["G"], "increment": inc_j, "D": dt_j, "r": r_j, "R_orth": js["R_orth"], "ordering_violations": js["ordering_violations"]}}


def boot_stats(f: dict, seed: int) -> dict:
    """Paired bootstrap over EVAL units of (r_BA M->A, r_BA B->M, R_orth) for both readouts; 95 % percentile CIs and whether 0 is inside."""
    fp, fq = {k: f[k][0] for k in C.CHAIN}, {k: f[k][1] for k in C.CHAIN}

    def stat_common(p, q):
        ch = INC.chain_summary([np.tanh(p[k]) for k in C.CHAIN], [np.tanh(q[k]) for k in C.CHAIN], C.CHAIN)
        return [ch["steps"]["M->A"]["r_BA"], ch["steps"]["B->M"]["r_BA"], ch["R_orth"]]

    def stat_js(p, q):
        ch = C.js_chain_summary([p[k] for k in C.CHAIN], [q[k] for k in C.CHAIN], C.CHAIN)
        return [ch["steps"]["M->A"]["r_BA"], ch["steps"]["B->M"]["r_BA"], ch["R_orth"]]
    out = {}
    for rd, fn in (("common", stat_common), ("js_native", stat_js)):
        v = INC.paired_bootstrap(fp, fq, fn, reps=BOOT_REPS, seed=seed)
        out[rd] = {nm: {"ci": INC.ci(v[:, i]), "zero_in_ci": bool(INC.ci(v[:, i])[0] <= 0.0 <= INC.ci(v[:, i])[1])} for i, nm in enumerate(("r_M->A", "r_B->M", "R_orth"))}
    return out


def run_repeat(rho: float, orc: dict, n: int, rng, *, steps: int, seed: int, boot: bool = True) -> dict:
    xf, yf = C.sample_p(n, rho, rng); xe, ye = C.sample_p(n, rho, rng); nf, ne = C.shift(n, rng), C.shift(n, rng)
    t = lambda a: torch.as_tensor(a, dtype=torch.float32).to(DEVICE)
    Ff = {k: (t(v[0]), t(v[1])) for k, v in C.feats(xf, yf).items()}; Fe = {k: (t(v[0]), t(v[1])) for k, v in C.feats(xe, ye).items()}
    nf_t, ne_t = torch.as_tensor(nf).to(DEVICE), torch.as_tensor(ne).to(DEVICE)
    eta_e = {k: (C.eta(xe, ye, rho, C.KSET[k]), C.eta(xe, ye[ne], rho, C.KSET[k])) for k in C.CHAIN}     # exact eta on the EVAL P and Q pairs
    true = C.true_increments(orc)
    res = {"n": n, "rho": rho, "true_increment": true, "per_objective": {}}
    for obj in ("vcs", "js"):
        res["per_objective"][obj] = {}; indep = None
        for variant in VARIANTS:
            crit, info = {}, {}
            if variant == "zero":
                f = {k: (np.zeros(n), np.zeros(n)) for k in C.CHAIN}
            else:
                if variant == "indep":
                    for j, k in enumerate(C.CHAIN):
                        crit[k], pick, vals, bs = NJ.fit_picked(Ff[k][0], Ff[k][1], nf_t, objective=obj, steps=steps, seed=seed * 100 + j)
                        info[k] = {"picked": pick, "val_J": vals, "best_step": bs}
                    indep = crit
                else:
                    crit["A"] = indep["A"]
                    for j, (fine, coarse) in enumerate((("M", "A"), ("B", "M"))):
                        crit[fine], pick, vals, bs = NJ.fit_picked(Ff[fine][0], Ff[fine][1], nf_t, objective=obj, steps=steps, seed=seed * 100 + 50 + j,
                                                                   coarse=crit[coarse], cmap=CMAP[(fine, coarse)])
                        info[f"{fine}<-{coarse}"] = {"picked": pick, "val_J": vals, "best_step": bs}
                f = {k: NJ.readouts(crit[k], Fe[k][0], Fe[k][1], ne_t) for k in C.CHAIN}
            ro = chain_readouts(f)
            post = {k: C.posterior_mse(np.tanh(f[k][0]), np.tanh(f[k][1]), eta_e[k][0], eta_e[k][1]) for k in C.CHAIN}
            out = {"critics": info, "posterior_mse": post}
            for rd in READOUTS:
                tr = true["vcs" if rd == "common" else "js"]
                out[rd] = ro[rd] | {"increment_error": {s: ro[rd]["increment"][s] - tr[s] for s in tr}}
            if boot:
                out["bootstrap"] = boot_stats(f, seed=seed)
            res["per_objective"][obj][variant] = out
    return res


def summarise(reps: list) -> dict:
    out = {}
    for obj in ("vcs", "js"):
        for v in VARIANTS:
            for rd in READOUTS:
                g = lambda fn: [fn(r["per_objective"][obj][v]) for r in reps]
                vals = {f"inc_err_{s}": g(lambda x, s=s: x[rd]["increment_error"][s]) for s in ("B->M", "M->A", "B->A")}
                vals |= {f"inc_{s}": g(lambda x, s=s: x[rd]["increment"][s]) for s in ("B->M", "M->A", "B->A")}
                vals |= {f"D_{s}": g(lambda x, s=s: x[rd]["D"][s]) for s in ("B->M", "M->A", "B->A")}
                vals |= {f"r_{s}": g(lambda x, s=s: x[rd]["r"][s]) for s in ("B->M", "M->A", "B->A")}
                vals |= {f"post_{k}": g(lambda x, k=k: x["posterior_mse"][k]) for k in C.CHAIN}
                vals |= {"R_orth": g(lambda x: x[rd]["R_orth"]), "ordering_violations": g(lambda x: x[rd]["ordering_violations"])}
                vals |= {f"value_{k}": g(lambda x, k=k: x[rd]["value"][k]) for k in C.CHAIN}
                s = {k: {"mean": float(np.mean(x)), "sd": float(np.std(x, ddof=1)) if len(x) > 1 else None} for k, x in vals.items()}
                for st in ("B->M", "M->A", "B->A"):
                    s[f"inc_rmse_{st}"] = float(np.sqrt(np.mean(np.square(vals[f"inc_err_{st}"]))))
                if all("bootstrap" in r["per_objective"][obj][v] for r in reps):
                    for nm in ("r_M->A", "r_B->M", "R_orth"):
                        s[f"boot_zero_in_ci_{nm}"] = int(sum(r["per_objective"][obj][v]["bootstrap"][rd][nm]["zero_in_ci"] for r in reps))
                        s[f"boot_ci_width_{nm}"] = float(np.mean([np.diff(r["per_objective"][obj][v]["bootstrap"][rd][nm]["ci"])[0] for r in reps]))
                out[f"{obj}/{v}/{rd}"] = s
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--conds", default="null,weak,moderate"); ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--repeats", type=int, default=5); ap.add_argument("--steps", type=int, default=300); ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--no-boot", action="store_true"); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.n, a.repeats, a.steps = 400, 2, 30
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); t0 = time.time()
    for cname in a.conds.split(","):
        outp = Path(f"{a.out}_{cname}.json")
        if outp.is_file():
            print(f"[skip] {outp} exists", flush=True); continue
        rho = CONDS[cname]; tc = time.time()
        orc = C.oracle(rho, n=50_000 if a.smoke else 400_000); rng = np.random.default_rng([a.seed, int(round(rho * 1000))])
        R = {"utc": utc_now(), "device": str(DEVICE), "settings": vars(a) | {"cond": cname}, "generator": {"K_SIG": C.K_SIG, "M_IRR": C.M_IRR, "KEEP_A": C.KEEP_A, "negatives": "cyclic1 (one non-zero shift per draw)"},
             "cond": cname, "rho": rho, "oracle": orc, "repeats": []}
        for r in range(a.repeats):
            R["repeats"].append(run_repeat(rho, orc, a.n, rng, steps=a.steps, seed=a.seed + r, boot=not a.no_boot))
            R["summary"] = summarise(R["repeats"]); atomic_write_json(Path(f"{a.out}_{cname}.partial.json"), R)
            sm = R["summary"]; tr = C.true_increments(orc)
            print(f"[{cname} rho={rho} rep {r + 1}/{a.repeats} true M->A S {tr['vcs']['M->A']:.4f} / G {tr['js']['M->A']:.4f}] " + " | ".join(
                f"{obj}/{v}: errMA {sm[f'{obj}/{v}/common']['inc_err_M->A']['mean']:+.4f} errBM {sm[f'{obj}/{v}/common']['inc_err_B->M']['mean']:+.4f} "
                f"postB {sm[f'{obj}/{v}/common']['post_B']['mean']:.4f} Rorth {sm[f'{obj}/{v}/common']['R_orth']['mean']:+.4f} "
                f"jsErrMA {sm[f'{obj}/{v}/js_native']['inc_err_M->A']['mean']:+.4f}" for obj in ("vcs", "js") for v in VARIANTS) + f" ({time.time() - t0:.0f}s)", flush=True)
        R["seconds"] = time.time() - tc; atomic_write_json(outp, R)
        Path(f"{a.out}_{cname}.partial.json").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
