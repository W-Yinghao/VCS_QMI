"""Pre-check D — synthetic null check for the level of the permutation tests (P45 addendum 6, null diagnosis).

    python scripts/precheck_d_synthetic_null.py --mode fresh|fixed|fixed_redraw --out <prefix> [--sizes 2000,5000,10000] [--repeats 1000]
                                                [--pool-size 45000] [--dim 512] [--perms 200] [--steps 300] [--seed 4]

Runs the UNCHANGED `run_instance` of scripts/precheck_d_tests.py (same critics, permutation nulls, HSIC, C2ST) on Gaussian features
z ~ N(0, I_dim) with an independent binary nuisance N ~ Bernoulli(1/2) — an exact null — in three designs:
  fresh          every repeat draws fresh z (3n × dim) and N: independent repeats, the textbook level check (expected rate = δ);
  fixed          one pool of --pool-size z with N drawn ONCE; repeats are random 3n-subsets of that pool — the P45 design at s = 0;
  fixed_redraw   the same pool and subsets, but N re-drawn for the items of every repeat (what --redraw-null-n does on the real features).
Outputs the rejection rate of every test per n (JSON with per-instance decisions and p-values; markdown table) together with the
two-sided binomial 95 % band for an exact level-δ test at that R.  Runs on the GPU when one is visible (DEVICE of the tests module).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from precheck_d_tests import DEVICE, run_instance  # noqa: E402
from vcs_ssl.utils import atomic_write_json, utc_now  # noqa: E402

TESTS = ("vcs_hoeff", "vcs_hoeff_lin", "vcs_hoeff_mlp", "vcs_hoeff_closed", "vcs_perm", "hsic_perm", "c2st")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["fresh", "fixed", "fixed_redraw"]); ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default=None, help="comma list of n (default: fresh 500,2000; fixed modes 2000,5000,10000)")
    ap.add_argument("--repeats", type=int, default=1000); ap.add_argument("--pool-size", type=int, default=45000); ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--perms", type=int, default=200); ap.add_argument("--steps", type=int, default=300); ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=4)
    a = ap.parse_args()
    sizes = [int(v) for v in a.sizes.split(",")] if a.sizes else ([500, 2000] if a.mode == "fresh" else [2000, 5000, 10000])
    print("device:", DEVICE, "mode:", a.mode, "sizes:", sizes, "R:", a.repeats, flush=True)
    rng = np.random.default_rng(a.seed)
    results = {"settings": vars(a), "device": str(DEVICE), "utc_start": utc_now(), "by_n": {}}
    pool = None
    if a.mode != "fresh":
        assert a.pool_size >= 3 * max(sizes), "pool must hold three disjoint samples of the largest n"
        H_pool = torch.as_tensor(rng.standard_normal((a.pool_size, a.dim), dtype=np.float32)); N_pool = rng.integers(0, 2, size=a.pool_size)
        Y_pool = np.zeros(a.pool_size, dtype=np.int64); pool = (H_pool, Y_pool, N_pool)
        results["pool"] = {"size": a.pool_size, "n1_frac": float(N_pool.mean()), "N_sha256_first16": __import__("hashlib").sha256(N_pool.tobytes()).hexdigest()[:16]}
    t0 = time.time()
    for n in sizes:
        inst = []
        for r in range(a.repeats):
            if a.mode == "fresh":
                H = torch.as_tensor(rng.standard_normal((3 * n, a.dim), dtype=np.float32)); N = rng.integers(0, 2, size=3 * n); Y = np.zeros(3 * n, dtype=np.int64)
                res = run_instance(H, Y, N, n, rng, conditional=False, delta=a.delta, perms=a.perms, steps=a.steps, seed=a.seed * 1000 + r)
            else:
                H, Y, N = pool
                res = run_instance(H, Y, N, n, rng, conditional=False, delta=a.delta, perms=a.perms, steps=a.steps, seed=a.seed * 1000 + r, redraw_n=(a.mode == "fixed_redraw"))
            inst.append({t: {"reject": bool(res[t]["reject"]), "p": res[t].get("p"), "J_eval": res[t].get("J_eval"), "stat": res[t].get("stat")} for t in TESTS})
        rate = {t: float(np.mean([i[t]["reject"] for i in inst])) for t in TESTS}
        half = 1.96 * math.sqrt(a.delta * (1 - a.delta) / a.repeats)
        results["by_n"][str(n)] = {"repeats": a.repeats, "rate": rate, "binomial95": [a.delta - half, a.delta + half], "instances": inst}
        atomic_write_json(Path(a.out + ".partial.json"), results)
        print(f"[{a.mode}] n={n} R={a.repeats}: " + " ".join(f"{t}={rate[t]:.3f}" for t in TESTS) + f"  (95% band {a.delta - half:.3f}–{a.delta + half:.3f}; {time.time() - t0:.0f}s)", flush=True)
    results["utc_end"] = utc_now(); atomic_write_json(Path(a.out + ".json"), results)
    L = [f"# Pre-check D — synthetic null level check, mode `{a.mode}` — {utc_now()}", "",
         f"Gaussian z (dim {a.dim}), N ~ Bernoulli(1/2) independent; δ = {a.delta}; {a.perms} permutations; critics {a.steps} steps; "
         + ("fresh z and N per repeat." if a.mode == "fresh" else f"one pool of {a.pool_size} items, N drawn once" + (", re-drawn per repeat." if a.mode == "fixed_redraw" else ".")), "",
         "| n | R | 95 % band (exact test) | vcs_hoeff | hoeff lin | hoeff mlp | hoeff closed | vcs_perm | hsic_perm | c2st |", "|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in results["by_n"].items():
        lo, hi = r["binomial95"]; rt = r["rate"]
        L.append(f"| {n} | {r['repeats']} | {lo:.3f}–{hi:.3f} | " + " | ".join(f"{rt[t]:.3f}" for t in TESTS) + " |")
    Path(a.out + ".md").write_text("\n".join(L) + "\n"); print("->", a.out + ".md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
