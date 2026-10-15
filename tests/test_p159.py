"""P159 (SEP-3) gate tests: cell lists; the natural-data contamination agrees with P156's synthetic contamination on a synthetic draw (same k,
untouched complement, injected pairs independent / true); AUROC and ECE readouts on cases with known values; the aggregate rule on synthetic effects;
sep.py / p157.py untouched."""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vcs_estim import p159, sep  # noqa: E402
from vcs_estim.benchmark import Role  # noqa: E402


def test_cell_lists():
    cs = p159.cells_for_seed(300); names = {p159.cell_name(c) for c in cs}
    assert len(cs) == 40 == len(names) and all(c["arms"] == list(sep.ARMS_PAIR) and c["lrs"] == list(p159.LRS) and c["d"] == 512 for c in cs)
    assert sum(c["ctype"] == "clean" for c in cs) == 8 and {c["eps"] for c in cs} == {0.0, 0.1, 0.3}
    assert {c["N"] for c in cs if c["source"] == "coco_clip"} == {4096, 16384} and {c["N"] for c in cs if c["source"] == "cifar_ap3"} == {4096, 8192}
    assert p159.LRS == (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)


def _synthetic_roles(n=2000, d=8, rho=0.8, seed=5):
    g = torch.Generator().manual_seed(seed); x = torch.randn(n, d, generator=g, dtype=torch.float64); e = torch.randn(n, d, generator=g, dtype=torch.float64)
    y = rho * x + math.sqrt(1 - rho * rho) * e; s = 7; return {"FIT": Role(x, y, x.clone(), torch.roll(y, s, 0)), "EVAL": Role(x[:100], y[:100], x[:100].clone(), torch.roll(y[:100], 3, 0))}


@pytest.mark.parametrize("ctype", ["independent", "false_negative"])
def test_contamination_matches_synthetic_law(ctype, monkeypatch):
    """On a synthetic Gaussian draw: exactly round(eps n) units changed (as sep.contaminate_fn_roles / r1), the complement untouched, EVAL untouched,
    injected positives are product-like (mean x.y ~ 0) for 'independent' and injected negatives are true-like (mean x.y ~ rho d) for 'false_negative'."""
    roles = _synthetic_roles(); eps = 0.3; n = roles["FIT"].n; d = roles["FIT"].xp.shape[1]
    monkeypatch.setitem(p159._CACHE, "synthetic", {"data": {}, "prov": {}})
    out, rec = p159.contaminate("synthetic", roles, {"FIT": {}}, ctype, eps, 300)
    k = int(round(eps * n)); assert rec["FIT"]["k"] == k and "EVAL" not in rec and out["EVAL"] is roles["EVAL"]
    F, C = roles["FIT"], out["FIT"]
    if ctype == "independent":
        changed = (C.yp != F.yp).any(1); assert int(changed.sum()) == k and torch.equal(C.xp, F.xp) and torch.equal(C.xq, F.xq) and torch.equal(C.yq, F.yq)
        inj = (C.xp[changed] * C.yp[changed]).sum(1).mean() / d; keep = (C.xp[~changed] * C.yp[~changed]).sum(1).mean() / d
        assert abs(float(inj)) < 0.15 and float(keep) > 0.6                           # product-like vs true pairs (rho 0.8)
    else:
        changed = (C.xq != F.xq).any(1); assert int(changed.sum()) == k and torch.equal(C.xp, F.xp) and torch.equal(C.yp, F.yp)
        inj = (C.xq[changed] * C.yq[changed]).sum(1).mean() / d; keep = (C.xq[~changed] * C.yq[~changed]).sum(1).mean() / d
        assert float(inj) > 0.6 and abs(float(keep)) < 0.15                            # true pairs among negatives vs product pairs
    # same accounting as the synthetic contaminator of P156 on the same eps / n
    _, rec_syn = sep.contaminate_fn_roles({"FIT": roles["FIT"]}, eps, 0.8, 300)
    assert rec_syn["FIT"]["k"] == rec["FIT"]["k"] == k


def test_auroc_and_ece_known_values():
    g = torch.Generator().manual_seed(1); n = 200_000
    mu = 1.0; s1 = torch.randn(n, generator=g) + mu; s0 = torch.randn(n, generator=g) - mu; y = torch.cat([torch.ones(n), torch.zeros(n)])
    from math import erf, sqrt
    phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))
    assert abs(p159.auroc(torch.cat([s1, s0]), y) - phi(2 * mu / sqrt(2))) < 3e-3           # two Gaussians, unit variance: AUROC = Phi(d' / sqrt 2)
    assert p159.auroc(torch.tensor([1.0, 1.0, 0.0, 0.0]), torch.tensor([1.0, 0.0, 1.0, 0.0])) == 0.5   # ties: average ranks
    p = torch.rand(n, generator=g); yb = (torch.rand(n, generator=g) < p).double()
    assert p159.ece(p, yb) < 0.01                                                            # calibrated probabilities: ECE ~ 0
    assert abs(p159.ece(torch.full((n,), 0.9), torch.zeros(n)) - 0.9) < 1e-6               # constant 0.9 against all-zero labels: ECE 0.9 (float32 input)


def test_readouts_on_small_synthetic_cell():
    roles = _synthetic_roles(n=600, d=16); dev = torch.device("cpu")
    critic, stateJ, info = sep.train_arm("vcs", "quad3", "product", 0.03, {"FIT": roles["FIT"], "SELECT": roles["FIT"]}, 16, 128, 30, 300, dev, None, 10)
    ro = p159.readouts(critic, "vcs", roles["EVAL"], dev)
    assert set(ro) >= {"J_eval", "native_risk", "auroc", "ece", "gate_mean"} and 0.5 <= ro["auroc"] <= 1.0 and 0 <= ro["ece"] <= 1 and math.isfinite(ro["J_eval"])


def test_frozen_modules_untouched():
    import hashlib
    h = lambda p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()[:16]
    committed = subprocess.run(["git", "show", "HEAD:src/vcs_estim/sep.py"], cwd=ROOT, capture_output=True).stdout
    committed2 = subprocess.run(["git", "show", "HEAD:src/vcs_estim/p157.py"], cwd=ROOT, capture_output=True).stdout
    assert hashlib.sha256(committed).hexdigest()[:16] == h("src/vcs_estim/sep.py") and hashlib.sha256(committed2).hexdigest()[:16] == h("src/vcs_estim/p157.py")


def _fake(tmp: Path, src, cls, ct, e, N, seed, J_js, J_vcs, A_js=0.9, A_vcs=0.9, E_js=0.05, E_vcs=0.05):
    c = p159.cell(src, cls, ct, e, N, seed)
    row = lambda arm, J, Au, Ec: {"arm": arm, "lr": 0.1, "selected": True, "selected_commonJ": True, "lr_edge": None, "select": {},
                                 "readout": {"J_eval": J, "auroc": Au, "ece": Ec, "native_risk": 0.1}, "readout_commonJ_selected": {"J_eval": J, "auroc": Au, "ece": Ec, "native_risk": 0.1}}
    (tmp / (p159.cell_name(c) + ".json")).write_text(json.dumps({"cell": {**c, "name": p159.cell_name(c)}, "rows": [row("js", J_js, A_js, E_js), row("vcs", J_vcs, A_vcs, E_vcs)]}))


@pytest.mark.parametrize("q1,q2,expect", [(0.2, 0.0, "SUPPORTED"), (0.2, 0.2, "FIT-ONLY"), (0.2, -0.2, "CONTAMINATION-SPECIFIC WITH CLEAN DEFICIT"), (0.0, 0.0, "NOT SUPPORTED")])
def test_aggregate_rule(tmp_path, q1, q2, expect):
    import random
    rnd = random.Random(0); v = {}                                     # per-cell noise, negated on the mirrored seed (300 + j <-> 304 + j): zero-mean by construction
    for s in p159.SEEDS:
        for c in p159.cells_for_seed(s):
            key = (c["source"], c["cls"], c["ctype"], c["eps"], c["N"]); v.setdefault(key, rnd.random())
            sign = 1.0 if s - 300 < 4 else -1.0; eff = q2 if c["eps"] == 0 else q1; base = 0.5
            _fake(tmp_path, c["source"], c["cls"], c["ctype"], c["eps"], c["N"], s, base, base * (1 + eff) + sign * 0.002 * v[key])
    out = tmp_path / "agg"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "p159_aggregate.py"), "--dir", str(tmp_path), "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    res = json.loads((tmp_path / "agg.json").read_text())["selected"]
    assert res["outcome"] == expect and res["n_cells"] == 40 and res["primary"]["Q1_dJ_eps_gt_0"]["complete"] and not res["missing_cells"]
    assert abs(res["primary"]["Q1_dJ_eps_gt_0"]["stat"]["mean"] - q1) < 0.02 and len(res["degradation"]) == 32
