"""Next round, T line — T1: the within-class conditional dependence test with fair controls and N | Y re-drawn for every repeat.

    python scripts/cond_test_t1.py --features-colour <P45 colour dir> [--features-blur <P45 blur dir>] --out <prefix> [--smoke]

Question: Z ⊥ N | Y (discrete Y = CIFAR class), where the nuisance N ∈ {0, 1} is class-correlated (P(N = 1 | Y) = ½ ± 0.3 by class parity) and the
images with N = 1 carry a planted nuisance (colour-temperature shift s or Gaussian blur σ).  Unlike P45/P46 (one N draw per pool), every repeat
draws N | Y afresh: an item that gets N = 1 takes its feature from the planted file (only items whose stored draw was 1 have a planted feature; the
stored draw is i.i.d. Bernoulli(½), so conditioning on it does not bias the images), an item with N = 0 takes the clean (s = 0) feature — the files
share uids and y, verified at load.  Three disjoint samples of n per repeat (FIT / EVAL / POOL); level δ; B within-class permutations of N on EVAL,
**the same permutations for every test**.
Null cases (exact by construction): `null_label_only` — clean features for everyone, N | Y dependent; `null_all_planted` — planted-file features for
everyone (so nuisance-affected features are present) and N | Y drawn independently of the item.
Tests (identical splits, budgets and permutations):
  vcs_perm      VCS J with the VAL-picked critic (linear / MLP / closed-form linear class), within-class permutation null
  vcs_hoeff     the guaranteed test: J_eval − τ_{n,n}(δ) > 0
  hsic_perm     HSIC, Gaussian kernel with ONE median bandwidth (the P46 control)
  hsic_class    HSIC with a per-class median bandwidth, statistic = Σ_c (n_c / n) HSIC_b within class c
  hsic_deep     deep-kernel HSIC: φ(z) = MLP trained on FIT (80 %) to maximise the within-class HSIC minus its permutation baseline, selected on VAL
                (20 %), then the same per-class HSIC test on EVAL with φ fixed (Liu et al. 2020 style)
  c2st_logit    the C2ST classifier (BCE, same MLP and budget), statistic = mean logit of the observed EVAL pairs, permutation null
  js_perm       the Deep-InfoMax JS objective on the same critic classes (linear / MLP), VAL-picked, statistic on EVAL, permutation null
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from precheck_d_tests import (DEVICE, PairLinear, PairMLP, closed_form_critic, fit_c2st, fit_vcs_critic, gaussian_kernel, hsic_stat,  # noqa: E402
                              j_stat, tau_hoeff, within_class_pool)
from vcs_ssl.utils import atomic_write_json, utc_now  # noqa: E402

SOFTPLUS = nn.functional.softplus


# ----------------------------------------------------------------------------------------------------------------------- JS critic
def js_value(f_pos, f_neg):
    """Deep-InfoMax JS estimate: E_P[-softplus(-f)] - E_Q[softplus(f)] + log 4 (bounded by log 2 at the optimum)."""
    return float((-SOFTPLUS(-f_pos)).mean() - SOFTPLUS(f_neg).mean() + math.log(4))


def _js_loss(m, z, n, n_neg):
    return -((-SOFTPLUS(-m(z, n))).mean() - SOFTPLUS(m(z, n_neg)).mean())


def fit_js_critic(z, n, n_neg, steps, seed, lr=1e-3, kind="linear", wd=1e-2, val_frac=0.2):
    """Same classes, optimiser, budget and early stopping as `fit_vcs_critic`, with the JS objective; `val_J` holds the VAL JS value."""
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(z), generator=g); nv = max(8, int(val_frac * len(z))); vi, ti = perm[:nv], perm[nv:]
    m = (PairLinear(z.shape[1]) if kind == "linear" else PairMLP(z.shape[1])).to(z.device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    best, best_state, best_step = float("inf"), {k: v.clone() for k, v in m.state_dict().items()}, 0
    for step in range(1, steps + 1):
        opt.zero_grad(); loss = _js_loss(m, z[ti], n[ti], n_neg[ti]); loss.backward(); opt.step()
        if step % 10 == 0 or step == steps:
            with torch.no_grad():
                lv = float(_js_loss(m, z[vi], n[vi], n_neg[vi]))
            if lv < best:
                best, best_step, best_state = lv, step, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state); m.eval(); m.best_step = best_step; m.val_J = -best + math.log(4)
    return m


# ----------------------------------------------------------------------------------------------------------------------- HSIC variants
def class_blocks(y):
    return [np.where(y == c)[0] for c in np.unique(y)]


def class_kernels(z, blocks):
    """Per-class Gaussian kernel matrices with the median heuristic computed WITHIN the class."""
    return [gaussian_kernel(z[torch.as_tensor(ii, device=z.device)]) for ii in blocks]


def hsic_class_stat(Ks, blocks, n_bits, n_total):
    """Σ_c (n_c / n) HSIC_b(K_c, N_c): the per-class-bandwidth conditional HSIC statistic."""
    s = 0.0
    for K, ii in zip(Ks, blocks):
        if len(ii) < 4:
            continue
        s += len(ii) / n_total * hsic_stat(K, n_bits[torch.as_tensor(ii, device=n_bits.device)])
    return float(s)


def _hsic_class_torch(zf, blocks_t, nb, n_total):
    """Differentiable version (no median detach issue: the bandwidth is detached) used to train the deep kernel."""
    s = 0.0
    for ii in blocks_t:
        if len(ii) < 4:
            continue
        zz = zf[ii]; d2 = torch.cdist(zz, zz).pow(2); med = torch.median(d2[d2 > 0]).detach() if (d2 > 0).any() else torch.tensor(1.0, device=zz.device)
        K = torch.exp(-d2 / med); nbc = nb[ii].float(); L = nbc[:, None] * nbc[None, :] + (1 - nbc)[:, None] * (1 - nbc)[None, :]
        N = K.shape[0]; H = torch.eye(N, device=K.device) - 1.0 / N
        s = s + len(ii) / n_total * torch.trace(K @ H @ L @ H) / N ** 2
    return s


class DeepKernel(nn.Module):
    def __init__(self, d, hidden=128, out=32):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, out))

    def forward(self, z):
        return self.net(z)


def fit_deep_kernel(z, n, y, steps, seed, rng, lr=1e-3, wd=1e-2, val_frac=0.2, n_base=8):
    """Train φ on 80 % of FIT to maximise [within-class HSIC(φ(z), N) − mean over n_base within-class permutations] / (sd over the permutations + 1e-8);
    select the step by the same criterion on the 20 % VAL part.  Same step budget as the critics."""
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(z), generator=g); nv = max(16, int(val_frac * len(z))); vi, ti = perm[:nv].numpy(), perm[nv:].numpy()
    m = DeepKernel(z.shape[1]).to(z.device); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    def prep(ix):
        yy = y[ix]; blocks = [torch.as_tensor(b, device=z.device) for b in class_blocks(yy)]; nb = n[torch.as_tensor(ix, device=z.device)]
        perms = []
        for _ in range(n_base):
            pn = nb.clone().cpu().numpy()
            for b in class_blocks(yy):
                pn[b] = pn[b][rng.permutation(len(b))]
            perms.append(torch.as_tensor(pn, device=z.device))
        return torch.as_tensor(ix, device=z.device), blocks, nb, perms
    (it, bt, nt, pt), (iv, bv, nv_, pv) = prep(ti), prep(vi)
    def crit(ix, blocks, nb, perms, train):
        f = m(z[ix]); s = _hsic_class_torch(f, blocks, nb, len(ix))
        base = torch.stack([_hsic_class_torch(f, blocks, p, len(ix)) for p in perms])
        return (s - base.mean()) / (base.std() + 1e-8)
    best, best_state, best_step = -float("inf"), {k: v.clone() for k, v in m.state_dict().items()}, 0
    for step in range(1, steps + 1):
        opt.zero_grad(); loss = -crit(it, bt, nt, pt, True); loss.backward(); opt.step()
        if step % 10 == 0 or step == steps:
            with torch.no_grad():
                v = float(crit(iv, bv, nv_, pv, False))
            if v > best:
                best, best_step, best_state = v, step, {k: v_.clone() for k, v_ in m.state_dict().items()}
    m.load_state_dict(best_state); m.eval(); m.best_step = best_step; m.val_score = best
    return m


# ----------------------------------------------------------------------------------------------------------------------- data assembly
class Family:
    """Feature files of one nuisance family for one encoder: clean h (s = 0) and planted h per strength, sharing uids / y / the stored N draw."""

    def __init__(self, fdir: Path):
        man = json.load(open(fdir / "manifest.json")); self.name = man.get("nuisance", "colour"); self.dir = fdir
        clean = torch.load(fdir / man["cases"]["s0"]["file"], map_location="cpu", weights_only=False)
        self.h0, self.y, self.uids, self.n_stored = clean["h"].float(), clean["y"].numpy(), clean["uids"].numpy(), clean["n"].numpy()
        self.planted = {}
        for k, v in man["cases"].items():
            if k.startswith("s") and v["strength"] > 0:
                d = torch.load(fdir / v["file"], map_location="cpu", weights_only=False)
                assert (d["uids"].numpy() == self.uids).all() and (d["y"].numpy() == self.y).all(), f"uid/y mismatch in {k}"
                assert torch.allclose(d["h"][d["n"] == 0], self.h0[d["n"] == 0]), f"clean features differ for N = 0 items in {k}"
                self.planted[float(v["strength"])] = d["h"].float()
        self.by_class = {c: np.where(self.y == c)[0] for c in np.unique(self.y)}
        self.by_class_n = {(c, b): np.where((self.y == c) & (self.n_stored == b))[0] for c in np.unique(self.y) for b in (0, 1)}
        self.p_c = {c: 0.5 + 0.3 * (1.0 if c % 2 == 0 else -1.0) for c in np.unique(self.y)}


def draw_sample(fam: Family, n, rng, mode, strength, used: set):
    """One sample of n items with fresh N | Y.  mode: 'planted' (N = 1 -> planted feature, N = 0 -> clean), 'null_label_only' (clean for all),
    'null_all_planted' (planted-file feature for all, N independent of the item).  Returns (H, Y, N, idx); idx are pool positions (kept disjoint via `used`)."""
    classes = np.array(sorted(fam.by_class)); ys = rng.choice(classes, size=n, replace=True)  # class frequencies as in the pool (uniform for CIFAR)
    ns = (rng.random(n) < np.vectorize(fam.p_c.get)(ys)).astype(np.int64)
    idx = np.empty(n, dtype=np.int64)
    for c in classes:
        for b in (0, 1):
            want = np.where((ys == c) & (ns == b))[0]
            if len(want) == 0:
                continue
            src = fam.by_class_n[(c, b)] if mode == "planted" else fam.by_class[c]
            src = src[~np.isin(src, list(used))] if used else src
            pick = rng.choice(src, size=len(want), replace=False); idx[want] = pick; used.update(pick.tolist())
    if mode == "planted":
        H = torch.where(torch.as_tensor(ns == 1)[:, None], fam.planted[strength][idx], fam.h0[idx])
    elif mode == "null_label_only":
        H = fam.h0[idx]
    elif mode == "null_all_planted":
        H = fam.planted[strength][idx]
    else:
        raise ValueError(mode)
    return H, ys, ns, idx


# ----------------------------------------------------------------------------------------------------------------------- one repeat
def run_repeat(fam: Family, n, rng, *, mode, strength, delta, perms, steps, seed):
    used = set()
    Hf, Yf, Nf, _ = draw_sample(fam, n, rng, mode, strength, used)
    He, Ye, Ne, _ = draw_sample(fam, n, rng, mode, strength, used)
    Hp, Yp, Np, _ = draw_sample(fam, n, rng, mode, strength, used)
    mu, sd = Hf.mean(0), Hf.std(0) + 1e-6
    zf, ze = ((Hf - mu) / sd).to(DEVICE), ((He - mu) / sd).to(DEVICE)
    nf, ne = torch.as_tensor(Nf).to(DEVICE), torch.as_tensor(Ne).to(DEVICE)
    nf_neg = torch.as_tensor(within_class_pool(Yf, Yp, Np, rng)).to(DEVICE); ne_neg = torch.as_tensor(within_class_pool(Ye, Yp, Np, rng)).to(DEVICE)
    blocks = class_blocks(Ye)
    # shared within-class permutations of N on EVAL (identical for every permutation test)
    perm_N = []
    for _ in range(perms):
        pn = Ne.copy()
        for ii in blocks:
            pn[ii] = pn[ii][rng.permutation(len(ii))]
        perm_N.append(torch.as_tensor(pn).to(DEVICE))
    pval = lambda null, obs: float((1 + (np.asarray(null) >= obs).sum()) / (1 + len(null)))
    res = {"n": int(n), "n1_frac_eval": float(Ne.mean())}
    tau = tau_hoeff(n, n, delta)
    # ---- VCS: three critics, VAL-picked
    crits = {"lin": fit_vcs_critic(zf, nf, nf_neg, steps, seed, kind="linear"), "mlp": fit_vcs_critic(zf, nf, nf_neg, steps, seed, kind="mlp"),
             "closed": closed_form_critic(zf, nf, nf_neg, seed)}
    with torch.no_grad():
        for name, cr in crits.items():
            tp_, tn_ = torch.tanh(cr(ze, ne)), torch.tanh(cr(ze, ne_neg)); J = j_stat(tp_, tn_)
            res[f"vcs_hoeff_{name}"] = {"J_eval": J, "reject": J > tau, "val_J": float(getattr(cr, "val_J", float("nan")))}
        pick = max(crits, key=lambda k: getattr(crits[k], "val_J", -9)); crit = crits[pick]
        tn = torch.tanh(crit(ze, ne_neg)); J = j_stat(torch.tanh(crit(ze, ne)), tn)
        res["vcs_hoeff"] = dict(res[f"vcs_hoeff_{pick}"], picked=pick, tau=tau)
        null = [j_stat(torch.tanh(crit(ze, pn)), tn) for pn in perm_N]
    res["vcs_perm"] = {"J_eval": J, "p": pval(null, J), "reject": pval(null, J) <= delta, "picked": pick}
    # ---- HSIC, single bandwidth
    K = gaussian_kernel(ze); h0 = hsic_stat(K, ne); null = [hsic_stat(K, pn) for pn in perm_N]
    res["hsic_perm"] = {"stat": h0, "p": pval(null, h0), "reject": pval(null, h0) <= delta}
    # ---- HSIC, per-class bandwidth
    Ks = class_kernels(ze, blocks); h1 = hsic_class_stat(Ks, blocks, ne, n); null = [hsic_class_stat(Ks, blocks, pn, n) for pn in perm_N]
    res["hsic_class"] = {"stat": h1, "p": pval(null, h1), "reject": pval(null, h1) <= delta}
    # ---- deep-kernel HSIC (features learned on FIT, per-class test on EVAL)
    dk = fit_deep_kernel(zf, nf, Yf, steps, seed + 2, rng)
    with torch.no_grad():
        fe = dk(ze)
    Kd = class_kernels(fe, blocks); h2 = hsic_class_stat(Kd, blocks, ne, n); null = [hsic_class_stat(Kd, blocks, pn, n) for pn in perm_N]
    res["hsic_deep"] = {"stat": h2, "p": pval(null, h2), "reject": pval(null, h2) <= delta, "val_score": float(dk.val_score), "best_step": int(dk.best_step)}
    # ---- C2ST with the mean-logit statistic
    clf = fit_c2st(zf, nf, nf_neg, steps, seed + 1)
    with torch.no_grad():
        s0 = float(clf(ze, ne).mean()); null = [float(clf(ze, pn).mean()) for pn in perm_N]
        acc = float(((clf(ze, ne) > 0).float().mean() + (clf(ze, ne_neg) <= 0).float().mean()) / 2)
    res["c2st_logit"] = {"stat": s0, "acc": acc, "p": pval(null, s0), "reject": pval(null, s0) <= delta}
    # ---- JS statistic, same critic classes (linear / MLP), VAL-picked
    jcr = {"lin": fit_js_critic(zf, nf, nf_neg, steps, seed + 3, kind="linear"), "mlp": fit_js_critic(zf, nf, nf_neg, steps, seed + 3, kind="mlp")}
    jpick = max(jcr, key=lambda k: jcr[k].val_J); jc = jcr[jpick]
    with torch.no_grad():
        fneg = jc(ze, ne_neg); jsv = js_value(jc(ze, ne), fneg); null = [js_value(jc(ze, pn), fneg) for pn in perm_N]
    res["js_perm"] = {"stat": jsv, "p": pval(null, jsv), "reject": pval(null, jsv) <= delta, "picked": jpick, "val_JS": float(jc.val_J)}
    return res


TESTS = ("vcs_perm", "vcs_hoeff", "hsic_perm", "hsic_class", "hsic_deep", "c2st_logit", "js_perm")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-colour", default=None); ap.add_argument("--features-blur", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--strengths-colour", default="0.05,0.1,0.2"); ap.add_argument("--strengths-blur", default="0.25,0.5")
    ap.add_argument("--sizes", default="200,500,1000,2000"); ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--level-sizes", default="500,2000"); ap.add_argument("--level-repeats", type=int, default=1000)
    ap.add_argument("--level-family", default="colour", help="family whose null cases get --level-repeats (the other family's nulls run at --repeats)")
    ap.add_argument("--perms", type=int, default=200); ap.add_argument("--steps", type=int, default=300); ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", default=None, help="';'-separated case names to run (default all)")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))); print("device:", DEVICE, flush=True)
    fams = {}
    if a.features_colour: fams["colour"] = Family(Path(a.features_colour))
    if a.features_blur: fams["blur"] = Family(Path(a.features_blur))
    assert fams, "give at least one feature dir"
    sizes = [int(s) for s in a.sizes.split(",")]; level_sizes = [int(s) for s in a.level_sizes.split(",")]
    strengths = {"colour": [float(s) for s in a.strengths_colour.split(",")], "blur": [float(s) for s in a.strengths_blur.split(",")]}
    R, RL, perms, steps = a.repeats, a.level_repeats, a.perms, a.steps
    if a.smoke:
        sizes, level_sizes, R, RL, perms, steps = [200], [200], 5, 5, 20, 60
        strengths = {k: v[1:2] for k, v in strengths.items()}
    # cases: power cells per strength; two nulls per family (null_all_planted at the family's largest strength)
    cases = []
    for fname, fam in fams.items():
        avail = sorted(fam.planted)
        for s in strengths[fname]:
            assert s in fam.planted, f"strength {s} not in {fam.dir} (have {avail})"
            cases.append((fname, f"{fname}_s{s:g}", "planted", s))
        cases.append((fname, f"{fname}_null_label_only", "null_label_only", 0.0))
        cases.append((fname, f"{fname}_null_all_planted{max(strengths[fname]):g}", "null_all_planted", max(strengths[fname])))
    if a.only:
        keep = set(a.only.split(";")); cases = [c for c in cases if c[1] in keep]
    rng = np.random.default_rng(a.seed)
    results = {"settings": vars(a), "device": str(DEVICE), "utc": utc_now(), "families": {k: {"dir": str(v.dir), "nuisance": v.name, "strengths": sorted(v.planted)} for k, v in fams.items()},
               "tests": TESTS, "cases": {}}
    partial = Path(a.out + ".partial.json"); t0 = time.time()
    for fname, cname, mode, s in cases:
        fam = fams[fname]; results["cases"][cname] = {"family": fname, "mode": mode, "strength": s, "by_n": {}}
        for n in sizes:
            reps = R if mode == "planted" else (RL if (fname == a.level_family and n in level_sizes) else R)
            inst = [run_repeat(fam, n, rng, mode=mode, strength=s, delta=a.delta, perms=perms, steps=steps, seed=a.seed * 1000 + r) for r in range(reps)]
            summ = {t: {"power": float(np.mean([i[t]["reject"] for i in inst]))} for t in TESTS}
            for t in ("vcs_hoeff_lin", "vcs_hoeff_mlp", "vcs_hoeff_closed"):
                summ[t] = {"power": float(np.mean([i[t]["reject"] for i in inst]))}
            summ["vcs_perm"]["J_mean"] = float(np.mean([i["vcs_perm"]["J_eval"] for i in inst])); summ["vcs_perm"]["J_sd"] = float(np.std([i["vcs_perm"]["J_eval"] for i in inst]))
            summ["js_perm"]["JS_mean"] = float(np.mean([i["js_perm"]["stat"] for i in inst])); summ["vcs_hoeff"]["tau"] = inst[0]["vcs_hoeff"]["tau"]
            summ["vcs_perm"]["picked"] = dict(zip(*np.unique([i["vcs_perm"]["picked"] for i in inst], return_counts=True)))
            results["cases"][cname]["by_n"][str(n)] = {"repeats": reps, "summary": summ, "instances": inst}
            print(f"[{cname}] n={n} R={reps}: " + " ".join(f"{t}={summ[t]['power']:.2f}" for t in TESTS) + f"  J={summ['vcs_perm']['J_mean']:.4f} tau={summ['vcs_hoeff']['tau']:.3f} ({time.time() - t0:.0f}s)", flush=True)
            atomic_write_json(partial, results)
    atomic_write_json(Path(a.out + ".json"), results)
    # ---- markdown
    L = [f"# T1 — within-class conditional test with fair controls, N | Y re-drawn per repeat — {results['utc']}", "",
         f"Families: {', '.join(f'{k} ({v.dir.name})' for k, v in fams.items())}; δ = {a.delta}; {perms} shared within-class permutations; critics / classifier / deep kernel: {steps} steps, fit on 80 % of FIT, selected on 20 %.", "",
         "| case | mode | s | n | R | " + " | ".join(TESTS) + " | hoeff lin / mlp / closed | mean J (VCS) | mean JS |", "|---|---|---|---|---|" + "---|" * len(TESTS) + "---|---|---|"]
    for cname, c in results["cases"].items():
        for n, r in c["by_n"].items():
            sm = r["summary"]
            L.append(f"| {cname} | {c['mode']} | {c['strength']:g} | {n} | {r['repeats']} | " + " | ".join(f"{sm[t]['power']:.2f}" for t in TESTS)
                     + f" | {sm['vcs_hoeff_lin']['power']:.2f} / {sm['vcs_hoeff_mlp']['power']:.2f} / {sm['vcs_hoeff_closed']['power']:.2f} | {sm['vcs_perm']['J_mean']:.4f} ± {sm['vcs_perm']['J_sd']:.4f} | {sm['js_perm']['JS_mean']:.4f} |")
    L += ["", "## Smallest n with power ≥ 0.8 (planted cases)", "", "| case | " + " | ".join(TESTS) + " |", "|---|" + "---|" * len(TESTS)]
    for cname, c in results["cases"].items():
        if c["mode"] != "planted":
            continue
        row = []
        for t in TESTS:
            ns = [int(n) for n, r in c["by_n"].items() if r["summary"][t]["power"] >= 0.8]; row.append(str(min(ns)) if ns else "—")
        L.append(f"| {cname} | " + " | ".join(row) + " |")
    Path(a.out + ".md").write_text("\n".join(L) + "\n"); print("->", a.out + ".md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
