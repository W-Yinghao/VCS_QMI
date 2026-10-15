"""V8-03 lr rule (v8_select_lr.choose): lowest CAL_SELECT MSE, exact ties -> lower lr, refuses while any tuning unit is missing."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "v8"))
import v8_select_lr as S  # noqa: E402


def _units(tmp_path, mse, drop=None):
    rows = list(csv.DictReader(S.TUNING_CSV.read_text(encoding="utf-8-sig").splitlines()))
    for r in rows:
        if r["unit_id"] == drop:
            continue
        d = tmp_path / r["unit_id"]; d.mkdir()
        (d / "done.json").write_text(json.dumps({"unit": r["unit_id"], "selected_epoch": 1, "cal_select_mse": mse(r["arm"], float(r["lr"])), "gpu": "x"}))
    return rows


def test_choice_and_ties(tmp_path):
    table = {5e-6: 0.40, 1e-5: 0.39, 2e-5: 0.41}
    def mse(arm, lr):
        if arm == "U-BCE":
            return 0.38                                            # all tied -> lowest lr
        if arm == "U-L1":
            return {5e-6: 0.45, 1e-5: 0.44, 2e-5: 0.44}[lr]        # tie between 1e-5 and 2e-5 -> 1e-5
        return table[lr]
    rows = _units(tmp_path, mse); res = S.choose(tmp_path)["arms"]
    assert len(rows) == 24 and set(res) == {"U-CS", "U-BCE", "U-RPS", "U-CE", "U-MSE", "U-L1", "U-RANK", "U-LIST"}
    assert res["U-CS"]["lr"] == "1e-05" and res["U-CS"]["lr_edge"] is None
    assert res["U-BCE"]["lr"] == "5e-06" and res["U-BCE"]["lr_edge"] == "low"
    assert res["U-L1"]["lr"] == "1e-05"


def test_refuses_when_a_unit_is_missing(tmp_path):
    _units(tmp_path, lambda a, l: 0.4, drop="V8-TUNE-U-LIST-lr2e-05")
    with pytest.raises(SystemExit):
        S.choose(tmp_path)
