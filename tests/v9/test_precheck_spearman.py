"""V9-N2 precheck pair table (review U05): Spearman on tie-averaged ranks (scipy.stats.rankdata(method='average') / spearmanr semantics); undefined
(constant input, n < 2) -> None + a reason, never NaN; the JSON stays strict-valid and the markdown prints 'NA (constant)'."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from v9.n2 import precheck as PC  # noqa: E402


def _strict_json(text: str):
    def no_constant(tok):
        raise ValueError(f"non-standard JSON token {tok}")
    return json.loads(text, parse_constant=no_constant)


def test_review_examples():
    assert PC.spearman_with_reason(np.array([1, 1, 1, 1.0]), np.array([0, 1, 2, 3.0])) == (None, "constant_input")
    assert PC.spearman_with_reason(np.array([0, 1, 2, 3.0]), np.array([1, 1, 1, 1.0])) == (None, "constant_input")
    assert PC.spearman(np.array([1, 1, 1, 1.0]), np.array([0, 1, 2, 3.0])) is None
    r, why = PC.spearman_with_reason(np.array([0, 0, 1, 1.0]), np.array([0, 1, 0, 1.0])); assert why == "" and abs(r - 0.0) < 1e-15
    r, why = PC.spearman_with_reason(np.array([1, 1, 2, 3.0]), np.array([4, 3, 2, 1.0])); assert why == "" and abs(r - (-0.9486832980505139)) < 1e-12


def test_no_ties_closed_form_and_old_ordinal_agree():
    """Without ties: rho = 1 - 6 sum d^2 / (n (n^2 - 1)), and the old ordinal-rank value is unchanged (ties were the only defect)."""
    rng = np.random.default_rng(5); x = rng.random(2000); y = 0.4 * x + rng.random(2000)
    assert len(np.unique(x)) == len(x) and len(np.unique(y)) == len(y)
    d = np.argsort(np.argsort(x)) - np.argsort(np.argsort(y)); n = len(x)
    closed = 1 - 6 * float(np.sum(d.astype(float) ** 2)) / (n * (n ** 2 - 1))
    old = float(np.corrcoef(np.argsort(np.argsort(x, kind="mergesort")), np.argsort(np.argsort(y, kind="mergesort")))[0, 1])
    r = PC.spearman(x, y); assert abs(r - closed) < 1e-12 and abs(r - old) < 1e-12


def test_matches_scipy_without_and_with_ties():
    st = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(7); x = rng.normal(size=500); y = x + rng.normal(size=500)
    assert len(np.unique(x)) == len(x) and abs(PC.spearman(x, y) - float(st.spearmanr(x, y).statistic)) < 1e-12
    w01 = (rng.random(3000) < 0.7).astype(float); cont = rng.random(3000)                         # PlainFilter-like 0/1 vs a continuous weight
    rm = np.zeros(3000); rm[np.argsort(-cont, kind="stable")[:1800]] = 1.0; rm[np.argsort(-cont, kind="stable")[1800]] = 0.37   # RankMass-like
    q = np.round(cont * 5) / 5                                                                     # heavy ties on both sides
    for a, b in ((w01, cont), (rm, cont), (w01, rm), (q, w01), (q, rm)):
        assert abs(PC.spearman(a, b) - float(st.spearmanr(a, b).statistic)) < 1e-12
        assert np.array_equal(PC._average_ranks(a), st.rankdata(a, method="average"))


def test_tied_value_does_not_depend_on_row_order():
    """The old ordinal ranks broke ties by position, so a joint row permutation changed the value; tie-averaged ranks do not."""
    rng = np.random.default_rng(11); cont = rng.random(4000); w01 = (cont + 0.3 * rng.random(4000) > 0.6).astype(float)
    base = PC.spearman(w01, cont)
    for s in range(3):
        p = np.random.default_rng(100 + s).permutation(4000); assert abs(PC.spearman(w01[p], cont[p]) - base) < 1e-12


def test_small_n_edge_cases():
    assert PC.spearman_with_reason(np.array([0.3]), np.array([0.7])) == (None, "fewer_than_2_values")
    assert PC.spearman_with_reason(np.array([]), np.array([])) == (None, "fewer_than_2_values")
    assert abs(PC.spearman(np.array([1.0, 2.0]), np.array([3.0, 4.0])) - 1.0) < 1e-12
    assert abs(PC.spearman(np.array([1.0, 2.0]), np.array([4.0, 3.0])) + 1.0) < 1e-12
    assert PC.spearman_with_reason(np.array([1.0, 1.0]), np.array([1.0, 2.0])) == (None, "constant_input")
    assert PC.spearman_with_reason(np.array([0.0, np.nan]), np.array([1.0, 2.0])) == (None, "non_finite_input")
    with pytest.raises(ValueError):
        PC.spearman(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


def test_pair_table_json_strict_and_markdown_na(tmp_path):
    """weight_tables + write_precheck on constant (Unweighted_A, rho-hat 0 posteriors), 0/1 (PlainFilter) and continuous weights: null + reason in the
    JSON (strict parse, no NaN), 'NA (constant)' in the markdown pair table, numeric cells unchanged in format."""
    rng = np.random.default_rng(3); n = 60; cont = rng.uniform(0.1, 0.9, n)
    wb = {"Unweighted_A": np.ones(n), "VCS_Weight": np.ones(n), "PlainFilter": (cont > 0.3).astype(float), "BCE_Weight": cont}
    wt = PC.weight_tables(wb); pw = wt["pairwise"]
    assert pw["Unweighted_A|VCS_Weight"]["spearman"] is None and pw["Unweighted_A|VCS_Weight"]["spearman_reason"] == "constant_input"
    assert pw["BCE_Weight|Unweighted_A"]["spearman"] is None and pw["PlainFilter|Unweighted_A"]["spearman_reason"] == "constant_input"
    assert pw["BCE_Weight|PlainFilter"]["spearman_reason"] == "" and 0 < pw["BCE_Weight|PlainFilter"]["spearman"] < 1
    assert pw["Unweighted_A|VCS_Weight"]["share_identical_1e-6"] == 1.0                            # the other columns are untouched
    rows = [{"pair_id": f"a{i}", "role": "critic_fit"} for i in range(5)] + [{"pair_id": f"u{i}", "role": "U"} for i in range(n)]
    coef = PC.loss_coefficients(rows, {m: {f"u{i}": float(w[i]) for i in range(n)} for m, w in wb.items()}, seed=901)
    sep = PC.teacher_separability(rng.normal(0.7, 0.1, 200), rng.normal(0.3, 0.1, 200))
    iso = {"counts": {"U": n}, "n_U": n, "hashes": {}, "anchor_fraction": 0.1, "split": "train", "usage": {"U": "student training with weights"}}
    nm = {"design_rho": 0.0, "realized_rho": 0.0, "mapping_sha256": "0" * 64, "n_A_mismatched": 0}
    PC.write_precheck(tmp_path, "fixture", 0.0, 901, sep, wt, coef, iso, nm)
    rec = _strict_json((tmp_path / "precheck_fixture_rho0_s901.json").read_text())
    assert rec["weights"]["pairwise"]["Unweighted_A|VCS_Weight"]["spearman"] is None and rec["weights"]["pairwise"]["Unweighted_A|VCS_Weight"]["spearman_reason"] == "constant_input"
    md = (tmp_path / "precheck_fixture_rho0_s901.md").read_text()
    assert "| Unweighted_A|VCS_Weight | 0.0000 | NA (constant) | 1.000 | 0.0000 |" in md
    assert f"| BCE_Weight|PlainFilter | {pw['BCE_Weight|PlainFilter']['mean_abs_diff']:.4f} | {pw['BCE_Weight|PlainFilter']['spearman']:.3f} |" in md
    assert PC._fmt_spearman({"spearman": None, "spearman_reason": "fewer_than_2_values"}) == "NA (n < 2)"
