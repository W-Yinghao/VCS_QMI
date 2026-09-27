"""Second-application pre-check D (leakage detection power) — step 2: independence tests on the planted features.

    python scripts/precheck_d_tests.py --features <dir from step 1> --out <report prefix> [--sizes ...] [--repeats R] [--delta 0.05]

Per case (strength s), sample size n and repeat r, three DISJOINT samples of n items are drawn from the 45k pool: FIT (critic / classifier
fitting), EVAL (test statistic) and POOL (independent nuisance values for the product-of-marginals samples; never in-sample shuffles).
Tests (all at level delta on the EVAL sample; identical architecture and optimisation budget for the learned critics):
  vcs_hoeff  VCS one-sided test with the distribution-free bound: reject iff J_eval > tau(n,m,delta), tau = sqrt(2 log(4/delta)/n) + sqrt(2 log(4/delta)/m)
             (critic T = tanh(MLP([z, onehot N])) fitted on FIT with the J objective; positives (z_i, N_i), negatives (z_i, N_pool_i)).
  vcs_perm   same statistic J_eval, permutation null (N permuted among the EVAL items, B permutations) — exact under independence.
  hsic_perm  HSIC (Gaussian kernel on standardised z, median heuristic; delta kernel on N), permutation null, B permutations.
  c2st       classifier two-sample test: same MLP trained with BCE to separate joint (z_i, N_i) from product (z_i, N_pool_i) pairs on FIT,
             accuracy on EVAL vs 1/2, one-sided binomial (normal approximation).
Conditional cases (N correlated with the class Y): product samples and permutations are drawn WITHIN the class of each item (exact samples of
P_Y P_{Z|Y} P_{N|Y}); reported separately.  Outputs: JSON with every rejection decision and statistic, and a markdown summary.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from vcs_ssl.utils import atomic_write_json, utc_now


# ----------------------------------------------------------------------------------------------------------------------- critics
class PairMLP(nn.Module):
    def __init__(self, d, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d + 2, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, z, n):
        return self.net(torch.cat((z, nn.functional.one_hot(n, 2).float()), 1)).squeeze(1)


def j_stat(t_pos, t_neg):
    return float((t_pos - 0.5 * t_pos ** 2).mean() + (-t_neg - 0.5 * t_neg ** 2).mean())


def fit_vcs_critic(z, n, n_neg, steps, seed, lr=1e-3):
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    m = PairMLP(z.shape[1]); opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        tp = torch.tanh(m(z, n)); tn = torch.tanh(m(z, n_neg))
        loss = -((tp - 0.5 * tp ** 2).mean() + (-tn - 0.5 * tn ** 2).mean())
        loss.backward(); opt.step()
    return m.eval()


def fit_c2st(z, n, n_neg, steps, seed, lr=1e-3):
    torch.manual_seed(seed)
    m = PairMLP(z.shape[1]); opt = torch.optim.Adam(m.parameters(), lr=lr); bce = nn.BCEWithLogitsLoss()
    zz = torch.cat((z, z)); nn_ = torch.cat((n, n_neg)); lab = torch.cat((torch.ones(len(z)), torch.zeros(len(z))))
    for _ in range(steps):
        opt.zero_grad(); loss = bce(m(zz, nn_), lab); loss.backward(); opt.step()
    return m.eval()


# ----------------------------------------------------------------------------------------------------------------------- HSIC
def hsic_stat(K, n_bits):
    """HSIC_b with a Gaussian kernel matrix K on z and the delta kernel on a binary variable (L_ij = 1[n_i == n_j]); O(n^2)."""
    nb = n_bits.float(); L = nb[:, None] * nb[None, :] + (1 - nb)[:, None] * (1 - nb)[None, :]
    N = K.shape[0]; H = torch.eye(N) - 1.0 / N
    return float(torch.trace(K @ H @ L @ H) / N ** 2)


def gaussian_kernel(z):
    d2 = torch.cdist(z, z).pow(2); med = torch.median(d2[d2 > 0]) if (d2 > 0).any() else torch.tensor(1.0)
    return torch.exp(-d2 / med)


# ----------------------------------------------------------------------------------------------------------------------- one test instance
def tau_hoeff(n, m, delta):
    return math.sqrt(2 * math.log(4 / delta) / n) + math.sqrt(2 * math.log(4 / delta) / m)


def within_class_pool(y_eval, y_pool, n_pool, rng):
    """For each eval item pick a pool item of the same class (independent draw within the class)."""
    out = np.empty(len(y_eval), dtype=np.int64)
    for c in np.unique(y_eval):
        src = np.where(y_pool == c)[0]; dst = np.where(y_eval == c)[0]
        out[dst] = n_pool[rng.choice(src, size=len(dst), replace=len(src) < len(dst))]
    return out


def run_instance(H, Y, N, n, rng, *, conditional, delta, perms, steps, seed):
    idx = rng.permutation(len(H))[: 3 * n]; fit, ev, pool = idx[:n], idx[n:2 * n], idx[2 * n:]
    mu, sd = H[fit].mean(0), H[fit].std(0) + 1e-6
    zf, ze = (H[fit] - mu) / sd, (H[ev] - mu) / sd
    nf, ne = torch.as_tensor(N[fit]), torch.as_tensor(N[ev])
    if conditional:
        nf_neg = torch.as_tensor(within_class_pool(Y[fit], Y[pool], N[pool], rng))     # fit negatives from the pool (within class)
        ne_neg = torch.as_tensor(within_class_pool(Y[ev], Y[pool], N[pool], rng))
    else:
        nf_neg = torch.as_tensor(N[pool]); ne_neg = torch.as_tensor(N[pool][rng.permutation(n)])
    res = {}
    # VCS critic
    crit = fit_vcs_critic(zf, nf, nf_neg, steps, seed)
    with torch.no_grad():
        tp, tn = torch.tanh(crit(ze, ne)), torch.tanh(crit(ze, ne_neg))
    J = j_stat(tp, tn); tau = tau_hoeff(n, n, delta)
    res["vcs_hoeff"] = {"J_eval": J, "tau": tau, "reject": J > tau, "sat_pos": float((tp.abs() > 0.95).float().mean())}
    # VCS permutation null: permute N among the eval items (within class if conditional); negatives stay the pool draw
    null = []
    with torch.no_grad():
        for _ in range(perms):
            if conditional:
                pn = np.empty(n, dtype=np.int64)
                for c in np.unique(Y[ev]):
                    ii = np.where(Y[ev] == c)[0]; pn[ii] = N[ev][ii][rng.permutation(len(ii))]
            else:
                pn = N[ev][rng.permutation(n)]
            null.append(j_stat(torch.tanh(crit(ze, torch.as_tensor(pn))), tn))
    null = np.asarray(null); p_perm = float((1 + (null >= J).sum()) / (1 + perms))
    res["vcs_perm"] = {"J_eval": J, "p": p_perm, "reject": p_perm <= delta}
    # HSIC permutation
    K = gaussian_kernel(ze); h0 = hsic_stat(K, ne); hn = []
    for _ in range(perms):
        if conditional:
            pn = np.empty(n, dtype=np.int64)
            for c in np.unique(Y[ev]):
                ii = np.where(Y[ev] == c)[0]; pn[ii] = N[ev][ii][rng.permutation(len(ii))]
        else:
            pn = N[ev][rng.permutation(n)]
        hn.append(hsic_stat(K, torch.as_tensor(pn)))
    hn = np.asarray(hn); p_h = float((1 + (hn >= h0).sum()) / (1 + perms))
    res["hsic_perm"] = {"stat": h0, "p": p_h, "reject": p_h <= delta}
    # C2ST
    clf = fit_c2st(zf, nf, nf_neg, steps, seed + 1)
    with torch.no_grad():
        acc = float(((clf(ze, ne) > 0).float().mean() + (clf(ze, ne_neg) <= 0).float().mean()) / 2)
    m2 = 2 * n; zscore = (acc - 0.5) / math.sqrt(0.25 / m2); p_c = 0.5 * math.erfc(zscore / math.sqrt(2))
    res["c2st"] = {"acc": acc, "p": p_c, "reject": p_c <= delta}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="100,200,500,1000,2000,5000"); ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--repeats-large", type=int, default=30, help="repeats for n >= 2000"); ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--steps", type=int, default=300); ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--cond-sizes", default="500,2000"); ap.add_argument("--cond-repeats", type=int, default=50)
    ap.add_argument("--cases", default=None, help="comma list of case names to run (default: all in the manifest)")
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    fd = Path(a.features); man = json.load(open(fd / "manifest.json"))
    cases = a.cases.split(",") if a.cases else list(man["cases"].keys())
    sizes = [int(v) for v in a.sizes.split(",")]; cond_sizes = [int(v) for v in a.cond_sizes.split(",")]
    if a.smoke:
        sizes, cond_sizes, a.repeats, a.repeats_large, a.cond_repeats, a.perms, a.steps = [100, 200], [200], 3, 3, 3, 20, 50
    rng = np.random.default_rng(a.seed)
    results = {"manifest": man, "settings": vars(a), "utc_start": utc_now(), "cases": {}}
    t0 = time.time()
    for case in cases:
        d = torch.load(fd / man["cases"][case]["file"], map_location="cpu", weights_only=False)
        H, Y, N = d["h"], d["y"].numpy(), d["n"].numpy(); conditional = case.startswith("cond_")
        grid = cond_sizes if conditional else sizes
        results["cases"][case] = {"strength": d["strength"], "conditional": conditional, "by_n": {}}
        for n in grid:
            R = a.cond_repeats if conditional else (a.repeats_large if n >= 2000 else a.repeats)
            inst = [run_instance(H, Y, N, n, rng, conditional=conditional, delta=a.delta, perms=a.perms, steps=a.steps, seed=a.seed * 1000 + r) for r in range(R)]
            summ = {t: {"power": float(np.mean([i[t]["reject"] for i in inst]))} for t in ("vcs_hoeff", "vcs_perm", "hsic_perm", "c2st")}
            summ["vcs_hoeff"]["J_eval_mean"] = float(np.mean([i["vcs_hoeff"]["J_eval"] for i in inst])); summ["vcs_hoeff"]["J_eval_sd"] = float(np.std([i["vcs_hoeff"]["J_eval"] for i in inst]))
            summ["vcs_hoeff"]["tau"] = inst[0]["vcs_hoeff"]["tau"]; summ["c2st"]["acc_mean"] = float(np.mean([i["c2st"]["acc"] for i in inst]))
            # unconditional runs in conditional cases: ignore class (does the *unconditional* test reject, as it should for label_only?)
            if conditional:
                inst_u = [run_instance(H, Y, N, n, rng, conditional=False, delta=a.delta, perms=a.perms, steps=a.steps, seed=a.seed * 7000 + r) for r in range(R)]
                summ["unconditional_power"] = {t: float(np.mean([i[t]["reject"] for i in inst_u])) for t in ("vcs_hoeff", "vcs_perm", "hsic_perm", "c2st")}
            results["cases"][case]["by_n"][str(n)] = {"repeats": R, "summary": summ, "instances": inst}
            print(f"[{case}] n={n} R={R}: " + " ".join(f"{t}={summ[t]['power']:.2f}" for t in ("vcs_hoeff", "vcs_perm", "hsic_perm", "c2st")) + f"  J={summ['vcs_hoeff']['J_eval_mean']:.4f} tau={summ['vcs_hoeff']['tau']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    results["utc_end"] = utc_now(); atomic_write_json(Path(a.out + ".json"), results)
    # markdown summary
    L = [f"# Pre-check D — independence-test power on planted nuisances ({man['run']}, {man['checkpoint']}) — {utc_now()}", "",
         f"Pool: {man['n_fit']} fit images; per instance three disjoint samples of n (fit / eval / independent pool); level δ = {a.delta}; permutations {a.perms}; "
         f"critic/classifier: MLP({H.shape[1]}+2→128→128→1), {a.steps} Adam steps; repeats {a.repeats} (n ≥ 2000: {a.repeats_large}).", "",
         "| case | s | n | R | vcs_hoeff | vcs_perm | hsic_perm | c2st | mean J_eval | τ | c2st acc |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for case, cr in results["cases"].items():
        for n, r in cr["by_n"].items():
            s = r["summary"]
            L.append(f"| {case} | {cr['strength']:g} | {n} | {r['repeats']} | {s['vcs_hoeff']['power']:.2f} | {s['vcs_perm']['power']:.2f} | {s['hsic_perm']['power']:.2f} | {s['c2st']['power']:.2f} | {s['vcs_hoeff']['J_eval_mean']:.4f} ± {s['vcs_hoeff']['J_eval_sd']:.4f} | {s['vcs_hoeff']['tau']:.3f} | {s['c2st']['acc_mean']:.3f} |")
            if "unconditional_power" in s:
                u = s["unconditional_power"]; L.append(f"| {case} (unconditional test) | | {n} | {r['repeats']} | {u['vcs_hoeff']:.2f} | {u['vcs_perm']:.2f} | {u['hsic_perm']:.2f} | {u['c2st']:.2f} | | | |")
    Path(a.out + ".md").write_text("\n".join(L) + "\n"); print("->", a.out + ".md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
