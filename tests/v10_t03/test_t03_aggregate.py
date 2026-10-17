"""The aggregator skeleton (v10/tools/t03_aggregate.py) on tiny stand-ins for the 7 smoke units (4 steps each): smoke checks and the
grid cost projection, and the grid tables with every non-smoke unit reported as NOT_RUN."""
from __future__ import annotations

import importlib.util
import json

import pytest

import t03_common as C
import t03_run as R
import t03_train as TR
import t03_units as U

spec = importlib.util.spec_from_file_location("t03_aggregate", C.REPO / "v10" / "tools" / "t03_aggregate.py")
AG = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AG)


@pytest.fixture()
def smoke_standins(tiny_suite, out_root):
    suite, record = tiny_suite
    ident = C.identity()
    for row in U.load_rows():
        if row["block"] != "smoke":
            continue
        r = U.parse(row)
        cfg = TR.UnitConfig(unit_id=r["unit_id"], block="smoke", method=r["method"], family=r["family"], m=r["m"], rho=r["rho"], K=r["K"],
                            lr=r["lr"], cap=r["cap"], data_root_seed=7, algo_seed=r["algo_seed"], steps=4, ckpt_every=2)
        res = R.execute_unit(cfg, suite=suite, record=record, out=R.unit_dir(cfg.unit_id), ident=ident, env={"note": "test"}, extra={"unit_row": row})
        assert res["status"] == "COMPLETE", res.get("failure")
    return out_root


def test_smoke_summary(smoke_standins):
    s = AG.smoke(smoke_standins)
    assert s["status"] == "PASS", s["checks"]
    assert len(s["units"]) == 7 and all(u["wall_seconds_unit_total"] > 0 for u in s["units"])
    p = s["grid_cost_projection"]
    assert p["status"] == "PROJECTION_NOT_MEASUREMENT" and p["units"] == {"CAL": 252, "main": 600, "algo_extra": 120, "capacity": 240}
    assert 0 < p["wall_hours_sequential_low"] <= p["wall_hours_sequential_high"]
    json.dumps(C.jsonable(s), allow_nan=False)


def test_grid_tables_report_missing_units(smoke_standins):
    g = AG.grid(smoke_standins)
    assert g["status"]["smoke"] == {"COMPLETE": 7} and g["status"]["main"] == {"NOT_RUN": 600}
    assert len(g["main"]) == 60 and all(c["n_complete"] == 0 and c["expected"] == 10 for c in g["main"])
    assert len(g["capacity"]) == 24 and len(g["algo_extra"]) == 120
    assert g["saturated"]["conflicts"] == [] and len(g["saturated"]["rows"]) == 1


def test_stats_helper():
    st = AG._stats([0.1, -0.1, 0.2, 0.0])
    assert st["R"] == 4 and st["mean"] == pytest.approx(0.05) and st["rmse"] == pytest.approx((0.06 / 4) ** 0.5)
    assert st["mean_ci95_t"][0] < st["mean"] < st["mean_ci95_t"][1] and st["var_ci95_chi2"][0] < st["var"] < st["var_ci95_chi2"][1]
