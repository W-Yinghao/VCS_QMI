"""CAL selection rules: max CAL U-J; ties -> earlier checkpoint -> frozen candidate ordinal; non-finite values block."""
from __future__ import annotations

import pytest

import t03_select as S


def tr(vals, steps=(250, 500, 750)):
    return [{"step": s, "CAL_UJ": v} for s, v in zip(steps, vals)]


def test_candidate_order():
    assert [(o, lr, c) for o, lr, c in S.candidates_for("MF_VCS_M1")][:4] == [(0, 1e-4, 10), (1, 1e-4, 30), (2, 1e-4, 100), (3, 3e-4, 10)]
    assert [o for o, _, _ in S.candidates_for("MLP_VCS_perm8")] == [0, 1, 2]
    with pytest.raises(ValueError):
        S.candidate_ordinal("MF_VCS_M0", 1e-4, 10)
    with pytest.raises(ValueError):
        S.candidate_ordinal("MF_VCS_M1", 1e-4, None)


def test_checkpoint_ties_take_the_earlier_step():
    r = S.select_checkpoint(tr([0.1, 0.3, 0.3]))
    assert r["step"] == 500 and r["tied_steps"] == [500, 750]
    assert S.select_checkpoint(tr([-0.2, -0.1, -0.3]))["step"] == 500           # negative values are compared as they are


def test_group_ties_earlier_checkpoint_then_ordinal():
    c = [{"ordinal": 2, "unit_id": "c2", "lr": 1e-3, "cap": None, "trace": tr([0.2, 0.5, 0.1])},
         {"ordinal": 0, "unit_id": "c0", "lr": 1e-4, "cap": None, "trace": tr([0.1, 0.2, 0.5])},
         {"ordinal": 1, "unit_id": "c1", "lr": 3e-4, "cap": None, "trace": tr([0.1, 0.5, 0.4])}]
    sel = S.select_group(c)["selected"]
    assert (sel["unit_id"], sel["step"]) == ("c1", 500)                          # 0.5 at step 500 by c1 and c2 -> ordinal 1
    c[1]["trace"] = tr([0.5, 0.0, 0.0])
    assert S.select_group(c)["selected"]["unit_id"] == "c0"                       # 0.5 at the earlier step 250 wins over ordinals


def test_identical_m1_caps_select_the_smallest_cap():
    same = tr([0.1, 0.2, 0.2])
    c = [{"ordinal": i, "unit_id": f"m{i}", "lr": 1e-4, "cap": cap, "trace": same} for i, cap in enumerate((10, 30, 100))]
    sel = S.select_group(c)["selected"]
    assert (sel["cap"], sel["step"]) == (10, 500)


def test_nonfinite_or_duplicate_blocks():
    with pytest.raises(S.SelectionBlocked):
        S.select_checkpoint(tr([0.1, float("nan"), 0.2]))
    with pytest.raises(S.SelectionBlocked):
        S.select_checkpoint(tr([0.1, None, 0.2]))
    with pytest.raises(S.SelectionBlocked):
        S.select_group([{"ordinal": 0, "unit_id": "a", "lr": 1e-4, "cap": None, "trace": tr([0.1, 0.2, 0.3])},
                        {"ordinal": 0, "unit_id": "b", "lr": 1e-4, "cap": None, "trace": tr([0.1, 0.2, 0.3])}])
