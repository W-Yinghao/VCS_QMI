"""P157 (SEP-2) gate tests: cell lists, finite-guarded re-selection, an end-to-end tiny cell of each contamination type in the parametric
classes (type (b) was not run in parametric classes by P156), and the aggregate on synthetic cells with known effects."""
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
from vcs_estim import p157, sep  # noqa: E402

CPU = torch.device("cpu")


def test_cell_lists():
    cs = p157.cells_for_seed(200); names = {sep.cell_name(c) for c in cs}
    assert len(cs) == 64 == len(names) and all(c["block"] == "S2" and c["arms"] == list(sep.ARMS_PAIR) and c["lrs"] == list(p157.LRS) for c in cs)
    assert {c["N"] for c in cs} == {4096, 16384} and {c["ctype"] for c in cs} == {"independent", "false_negative"}
    pop = p157.population_cells(); assert len(pop) == 32 == len({sep.cell_name(c) for c in pop}) and all(c["N"] == 131072 and c["updates"] == 8000 for c in pop)
    assert all(sep.cell_name(c).endswith("_p157") for c in cs + pop)


def test_reselect_never_picks_a_diverged_point():
    mk = lambda lr, risk, j: {"arm": "js", "lr": lr, "select": {"select_risk": risk, "commonJ_select": j}, "selected": True, "selected_commonJ": True}
    R = {"cell": {}, "rows": [mk(0.1, 0.5, 0.2), mk(0.3, 0.4, 0.3), mk(1.0, float("nan"), float("inf"))]}
    R = p157.reselect_finite(R); sel = {r["lr"]: (r["selected"], r["selected_commonJ"], r["lr_edge"]) for r in R["rows"]}
    assert sel == {0.1: (False, False, "low"), 0.3: (True, True, None), 1.0: (False, False, "high")} and R["cell"]["protocol_id"] == "P157"


@pytest.mark.parametrize("ctype", ["independent", "false_negative"])
@pytest.mark.parametrize("cls", ["cos2", "quad3"])
def test_tiny_parametric_cell_end_to_end(cls, ctype):
    c = sep.cell("S2", cls, "product", ctype, 0.3, 20, 4.0, 256, 40, sep.ARMS_PAIR, p157.LRS, 200)
    R1 = p157.reselect_finite(sep.run_cell(c, CPU, smoke=True)); R2 = p157.reselect_finite(sep.run_cell(c, CPU, smoke=True))
    e1 = [r["readout"]["posterior_mse"] for r in R1["rows"]]
    assert e1 == [r["readout"]["posterior_mse"] for r in R2["rows"]] and all(math.isfinite(v) for v in e1)
    assert {r["arm"] for r in R1["rows"] if r["selected"]} == {"js", "vcs"}


def _fake(tmp: Path, cls, ct, e, d, I, N, seed, e_js, e_vcs, block="S2"):
    c = {**sep.cell(block, cls, "product", ct, e, d, I, N, 2000 if block == "S2" else 8000, sep.ARMS_PAIR, p157.LRS, seed), "tag": "p157"}
    row = lambda arm, v: {"arm": arm, "lr": 0.1, "selected": True, "selected_commonJ": True, "lr_edge": None, "select": {},
                          "readout": {"posterior_mse": v}, "readout_commonJ_selected": {"posterior_mse": v}}
    (tmp / (sep.cell_name(c) + ".json")).write_text(json.dumps({"cell": c, "rows": [row("js", e_js), row("vcs", e_vcs)]}))


def test_aggregate_known_effects(tmp_path):
    import random
    rnd = random.Random(0)
    for s in p157.SEEDS:
        for (cls, ct, e, d, I) in p157.configs():
            for N in p157.NS:
                eff = -0.3 if N == 4096 else 0.0                       # VCS 30 % lower at N 4,096, no effect at 16,384
                _fake(tmp_path, cls, ct, e, d, I, N, s, 0.05 + 0.002 * rnd.random(), 0.05 * (1 + eff) + 0.002 * rnd.random())
    for (cls, ct, e, d, I) in p157.configs():
        _fake(tmp_path, cls, ct, e, d, I, 131072, 200, 0.04, 0.03, block="S2P")
    out = tmp_path / "agg"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "p157_aggregate.py"), "--dir", str(tmp_path), "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    res = json.loads((tmp_path / "agg.json").read_text())["selected"]
    assert res["primary"]["Q1_N4096"]["reading"].startswith("< 0") and res["primary"]["Q2_N16384"]["reading"] == "n.s."
    assert res["outcome"] == "FINITE-SAMPLE ONLY" and res["n_cells"] == 64 and res["primary"]["Q1_N4096"]["complete"]
    assert abs(res["primary"]["Q1_N4096"]["stat"]["mean"] + 0.3) < 0.05 and len(res["population_block"]) == 32
