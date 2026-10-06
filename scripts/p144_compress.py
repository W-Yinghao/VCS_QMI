"""P144 (MV6-C1) — dependence change under the real deterministic maps h -> r -> z and h -> O on the paired colour / blur fixture.

Observations (Y, U, N) with U in {h, r, z, O}: r = projector(h) (eval mode), z = r / max(||r||, eps), O = the fixed clean-h logistic head of the
P109 / P143 feature store.  The maps are applied on CPU in fp32 to the cached fp32 h, so A = g(B) holds exactly by construction; the GPU-cached r
and the stored-logit path are compared with the recomputation and the discrepancy is recorded (map gate).  Each observation is standardised
with mean / sd fixed on a PROBE set of 4 000 FIT-split base images (an affine bijection: no information change), never drawn afterwards.
Per repeat (P118 visual construction): FIT n from the remaining FIT split, EVAL n from EVAL, POOL from VAL; fresh N | Y ~ Bernoulli(½ ± 0.3);
the observed h is the planted version iff N = 1 ('planted') or clean for all ('null').  Variants as P118: indep / nested x sampled / exact Q, plus
the constant-zero control; objectives VCS (primary) and JS.  Chain h ⊇ r ⊇ z (steps h->r, r->z) and the pair h -> O.

    python scripts/p144_compress.py --runs P107_AP3_views4_800ep_seed1 --features outputs/P143_features --out reports/P144/visual [--smoke]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
from precheck_d_tests import DEVICE, within_class_pool  # noqa: E402
from vcs_measure import nested as NS  # noqa: E402
from vcs_measure.audit import p_n1  # noqa: E402
from vcs_measure.common import load_run  # noqa: E402
from vcs_ssl.utils import atomic_write_json, utc_now  # noqa: E402

CELLS = {"colour_s0.1": ("planted", "colour_s0.1"), "blur_s0.25": ("planted", "blur_s0.25"), "null": ("null_label_only", None)}
VARIANTS = ("indep_sampled", "indep_exact", "nested_sampled", "nested_exact", "zero")
CHAIN = ["h", "r", "z"]
PROBE_N, PROBE_SEED = 4000, 144


class Store:
    def __init__(self, run: str, root: Path):
        d = torch.load(root / run / "features.pt", map_location="cpu", weights_only=False)
        assert "maps_fp32" in d, "feature store lacks fp32 maps (extract with --fp32-maps)"
        self.run, self.y, self.part = run, d["y"].numpy(), d["part"].numpy()
        R = load_run(run, device="cpu"); proj, eps = R["projector"].eval(), R["eps"]
        clf = d["classifier"]
        self.U, self.gate = {}, {}
        with torch.no_grad():
            for v, m in d["maps_fp32"].items():
                h = m["h"].float(); r = proj(h); z = r / r.norm(dim=1, keepdim=True).clamp_min(eps)
                O = ((h - clf["mu"]) / clf["sd"]) @ clf["W"] + clf["b"]
                self.U[v] = {"h": h, "r": r, "z": z, "O": O}
                self.gate[v] = {"max_abs_r_cached_vs_cpu": float((m["r"].float() - r).abs().max()),
                                "max_abs_z_fp16_vs_cpu": float((d["features"][v]["z"].float() - z).abs().max()),
                                "max_abs_logits_stored_vs_cpu": float((d["features"][v]["logits"].float() - O).abs().max())}
        self.idx = {k: np.where(self.part == i)[0] for i, k in enumerate(("fit", "val", "eval"))}
        k = min(PROBE_N, len(self.idx["fit"]) // 3)   # 4 000 at the registered sizes (FIT 10 000); smaller only in smokes
        self.probe = np.random.default_rng(PROBE_SEED).choice(self.idx["fit"], size=k, replace=False)
        self.fit_pool = np.setdiff1d(self.idx["fit"], self.probe)
        c = self.U["clean"]; pr = torch.as_tensor(self.probe)
        self.std = {k: (c[k][pr].mean(0), c[k][pr].std(0) + 1e-6) for k in ("h", "r", "z", "O")}
        self.proj, self.eps, self.clf = proj, eps, clf

    def standardise(self, k, x):
        mu, sd = self.std[k]; return (x - mu) / sd

    def maps(self):
        """Fine -> coarse maps on standardised observations (exact compositions of the fp32 CPU maps)."""
        s, un = self.std, lambda k, x: x * self.std[k][1].to(x.device) + self.std[k][0].to(x.device)
        proj = self.proj.to(DEVICE); clf = {k: v.to(DEVICE) for k, v in self.clf.items()}
        st = lambda k, x: (x - s[k][0].to(x.device)) / s[k][1].to(x.device)
        def h2r(x):
            return st("r", proj(un("h", x)))
        def r2z(x):
            r = un("r", x); return st("z", r / r.norm(dim=1, keepdim=True).clamp_min(self.eps))
        def h2O(x):
            h = un("h", x); return st("O", ((h - clf["mu"]) / clf["sd"]) @ clf["W"] + clf["b"])
        return {("r", "z"): r2z, ("h", "r"): h2r, ("h", "O"): h2O}


def draw(S: Store, rows_from, n, rng, mode, version):
    rows = rng.choice(rows_from, size=n, replace=False); y = S.y[rows]; nn_ = (rng.random(n) < p_n1(y)).astype(np.int64)
    rt, pick = torch.as_tensor(rows), torch.as_tensor(nn_ == 1)[:, None]
    obs = {}
    for k in ("h", "r", "z", "O"):
        c = S.U["clean"][k][rt]
        x = torch.where(pick, S.U[version][k][rt], c) if mode == "planted" else c
        obs[k] = S.standardise(k, x)
    return obs, y, nn_, rows


def zero_ro(n, p1):
    return np.zeros(n), (np.zeros((n, 1)), np.ones((n, 1))), (np.zeros((n, 2)), np.stack([1 - p1, p1], 1))


def run_repeat(S, cm, n, rng, mode, version, *, steps, seed, n_classes):
    Of, Yf, Nf, rf = draw(S, S.fit_pool, n, rng, mode, version); Oe, Ye, Ne, re_ = draw(S, S.idx["eval"], n, rng, mode, version)
    _, Yp, Np, rp = draw(S, S.idx["val"], min(n, len(S.idx["val"])), rng, mode, version)
    assert not (set(rf) & set(re_)) and not (set(rf) & set(rp)) and not (set(re_) & set(rp)) and not (set(rf) & set(S.probe))
    Ff = {k: v.to(DEVICE) for k, v in Of.items()}; Fe = {k: v.to(DEVICE) for k, v in Oe.items()}
    t = lambda a: torch.as_tensor(a).to(DEVICE)
    nf, yf, ne, ye = t(Nf), t(Yf), t(Ne), t(Ye)
    negf = {"n_neg": t(within_class_pool(Yf, Yp, Np, rng))}; nege = {"n_neg": t(within_class_pool(Ye, Yp, Np, rng))}
    exf = {"p1": torch.as_tensor(p_n1(Yf), dtype=torch.float32).to(DEVICE)}; p1e_np = p_n1(Ye); p1e = torch.as_tensor(p1e_np, dtype=torch.float32).to(DEVICE)
    res = {"n": n, "mode": mode, "version": version, "n1_frac_eval": float(Ne.mean()), "per_objective": {}}
    names = CHAIN + ["O", "h_for_O"]
    for obj in ("vcs", "js"):
        res["per_objective"][obj] = {}; indep = {}
        for variant in VARIANTS:
            info = {}
            if variant == "zero":
                ro = {nm: zero_ro(n, p1e_np) for nm in names}
            else:
                neg = exf if variant.endswith("exact") else negf
                if variant.startswith("indep"):
                    crit = {}
                    for k, nm in enumerate(["h", "r", "z", "O"]):
                        crit[nm], pick, vals, bs = NS.fit_picked(Ff[nm], nf, yf, neg, objective=obj, n_classes=n_classes, steps=steps, seed=seed * 100 + k)
                        info[nm] = {"picked": pick, "val_J": vals, "best_step": bs}
                    indep[variant] = crit; crit = dict(crit); crit["h_for_O"] = crit["h"]
                else:
                    base = indep["indep_" + variant.split("_")[1]]; crit = {"z": base["z"], "O": base["O"]}
                    for k, (fine, coarse) in enumerate((("r", "z"), ("h", "r"))):
                        crit[fine], pick, vals, bs = NS.fit_picked(Ff[fine], nf, yf, neg, objective=obj, n_classes=n_classes, steps=steps,
                                                                    seed=seed * 100 + 50 + k, coarse=crit[coarse], cmap=cm[(fine, coarse)])
                        info[f"{fine}<-{coarse}"] = {"picked": pick, "val_J": vals, "best_step": bs}
                    crit["h_for_O"], pick, vals, bs = NS.fit_picked(Ff["h"], nf, yf, neg, objective=obj, n_classes=n_classes, steps=steps,
                                                                     seed=seed * 100 + 60, coarse=base["O"], cmap=cm[("h", "O")])
                    info["h<-O"] = {"picked": pick, "val_J": vals, "best_step": bs}
                ro = {nm: NS.readouts(crit[nm], Fe["h" if nm == "h_for_O" else nm], ne, ye, nege, p1e) for nm in names}
            out = {"critics": info}
            for rd, j in (("readout_sampled", 1), ("readout_exact", 2)):
                wq = ro["h"][j][1]
                out[rd] = {"chain_h_r_z": NS.chain_w([(ro[c][0], ro[c][j][0]) for c in CHAIN], CHAIN, wq),
                           "pair_h_z": NS.increment_w((ro["h"][0], ro["h"][j][0]), (ro["z"][0], ro["z"][j][0]), wq),
                           "pair_h_O": NS.increment_w((ro["h_for_O"][0], ro["h_for_O"][j][0]), (ro["O"][0], ro["O"][j][0]), wq)}
            res["per_objective"][obj][variant] = out
    return res


def summarise(reps):
    out = {}
    for obj in ("vcs", "js"):
        for v in VARIANTS:
            for rd in ("readout_sampled", "readout_exact"):
                g = lambda f: [f(r["per_objective"][obj][v][rd]) for r in reps]
                vals = {"J_h": g(lambda x: x["chain_h_r_z"]["J"]["h"]), "J_r": g(lambda x: x["chain_h_r_z"]["J"]["r"]),
                        "J_z": g(lambda x: x["chain_h_r_z"]["J"]["z"]), "R_orth": g(lambda x: x["chain_h_r_z"]["R_orth"]),
                        "nesting_violations": g(lambda x: x["chain_h_r_z"]["nesting_violations"])}
                for st in ("h->r", "r->z"):
                    for k in ("delta_J", "D_T"):
                        vals[f"{st}/{k}"] = g(lambda x, st=st, k=k: x["chain_h_r_z"]["steps"][st][k])
                for pr in ("pair_h_z", "pair_h_O"):
                    for k in ("delta_J", "D_T", "J_B", "J_A"):
                        vals[f"{pr}/{k}"] = g(lambda x, pr=pr, k=k: x[pr][k])
                out[f"{obj}/{v}/{rd}"] = {k: {"mean": float(np.mean(x)), "sd_refit": float(np.std(x, ddof=1)) if len(x) > 1 else None} for k, x in vals.items()}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True); ap.add_argument("--features", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--cells", default="colour_s0.1,blur_s0.25,null"); ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--repeats", type=int, default=5); ap.add_argument("--steps", type=int, default=300); ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--tag", default=""); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.n, a.repeats, a.steps = 400, 2, 30
    t0 = time.time()
    for run in a.runs.split(","):
        dst = Path(f"{a.out}_{run}{a.tag}.json")
        if dst.is_file():
            print(f"skip {run} (exists)", flush=True); continue
        S = Store(run, Path(a.features)); n_classes = int(len(np.unique(S.y))); cm = S.maps()
        RR = {"run": run, "utc": utc_now(), "device": str(DEVICE), "settings": vars(a), "map_gate": S.gate, "cells": {}}
        print(f"[{run}] map gate (max over versions): " + str({k: max(g[k] for g in S.gate.values()) for k in next(iter(S.gate.values()))}), flush=True)
        for cname in a.cells.split(","):
            mode, version = CELLS[cname]; rng = np.random.default_rng([a.seed, list(CELLS).index(cname)])
            reps = [run_repeat(S, cm, a.n, rng, mode, version, steps=a.steps, seed=a.seed + r, n_classes=n_classes) for r in range(a.repeats)]
            RR["cells"][cname] = {"summary": summarise(reps), "repeats": reps}; sm = RR["cells"][cname]["summary"]
            for obj in ("vcs", "js"):
                k = f"{obj}/nested_exact/readout_exact"
                print(f"[{run} {cname} {obj} nested_exact] h->r dJ {sm[k]['h->r/delta_J']['mean']:+.4f} D_T {sm[k]['h->r/D_T']['mean']:.4f} | "
                      f"r->z dJ {sm[k]['r->z/delta_J']['mean']:+.4f} D_T {sm[k]['r->z/D_T']['mean']:.4f} | h->O dJ {sm[k]['pair_h_O/delta_J']['mean']:+.4f} "
                      f"D_T {sm[k]['pair_h_O/D_T']['mean']:.4f} ({time.time() - t0:.0f}s)", flush=True)
            atomic_write_json(Path(f"{a.out}_{run}{a.tag}.partial.json"), RR)
        RR["seconds"] = time.time() - t0; atomic_write_json(dst, RR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
