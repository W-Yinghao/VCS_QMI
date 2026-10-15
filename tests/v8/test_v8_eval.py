"""V8 evaluation step (scripts/v8/v8_eval.py): synthetic units with known best-of-9 outcomes through the FROZEN scorer (first-index ties,
bootstrap determinism with seed 2718, Holm), the unit-inventory refusal, and the distribution metrics against hand values.  No real label is
read: every label file here is synthetic and lives in tmp_path."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "v8"))
import v8_eval as E  # noqa: E402

GENS = ("DALLE_3", "SDXL_Base")


def _universe(n_prompts=6):
    return [(f"{p:05d}", g, c) for p in range(1, n_prompts + 1) for g in GENS for c in range(9)]


def _write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _labels(tmp_path, human):
    rows = [{"prompt_id": p, "generator_id": g, "image_id": f"{g}/{p}_{c:02d}.jpeg", "candidate_index": c, "human_score": float(human[(p, g)][c])} for p, g, c in _universe()]
    f = tmp_path / "labels.jsonl"; _write_jsonl(f, rows); return f


def _pred(tmp_path, name, score):
    rows = [{"prompt_id": p, "generator_id": g, "image_id": f"{g}/{p}_{c:02d}.jpeg", "candidate_index": c, "score": float(score[(p, g)][c])} for p, g, c in _universe()]
    f = tmp_path / f"{name}.jsonl"; _write_jsonl(f, rows); return str(f)


def test_frozen_scorer_first_index_ties_and_known_selection(tmp_path):
    """ours ranks candidate 4 highest everywhere (ties at candidates 4 and 7 -> first index 4 wins); comp ranks candidate 0 highest."""
    rng = np.random.default_rng(0); human = {}
    for p, g, _ in _universe():
        if (p, g) not in human:
            human[(p, g)] = rng.integers(1, 6, 9).astype(float)
    labels = _labels(tmp_path, human)
    ours = {k: np.array([0, 0, 0, 0, 1, 0, 0, 1, 0], float) for k in human}; comp = {k: np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], float) for k in human}
    methods = {"ours": [_pred(tmp_path, "ours_s0", ours), _pred(tmp_path, "ours_s1", ours)], "comp": [_pred(tmp_path, "comp", comp)]}
    work = tmp_path / "work"; work.mkdir(); res = E.run_scorer(methods, labels, tmp_path / "fam.json", "ours", ["comp"], work)
    for gen in GENS:
        exp_ours = np.mean([human[(p, gen)][4] for p, g, _ in _universe() if g == gen and _ == 0]); exp_comp = np.mean([human[(p, gen)][0] for p, g, _ in _universe() if g == gen and _ == 0])
        assert np.allclose(res["methods"]["ours"][gen]["N9"]["selected"], exp_ours) and np.isclose(res["methods"]["comp"][gen]["N9"]["selected"][0], exp_comp)
        c = res["primary"][f"ours - comp @ {gen} N9"]; assert np.isclose(c["delta"], exp_ours - exp_comp) and len(c["per_seed_delta"]) == 2
    assert all("p_holm" in c for c in res["primary"].values()) and len(res["primary"]) == 2


def test_bootstrap_is_deterministic_and_refuses_overwrite(tmp_path):
    rng = np.random.default_rng(1); human = {}; sc = {}
    for p, g, _ in _universe():
        if (p, g) not in human:
            human[(p, g)] = rng.integers(1, 6, 9).astype(float); sc[(p, g)] = rng.random(9)
    labels = _labels(tmp_path, human); methods = {"ours": [_pred(tmp_path, "o", sc)], "comp": [_pred(tmp_path, "c", {k: v[::-1] for k, v in sc.items()})]}
    w = tmp_path / "w"; w.mkdir(); r1 = E.run_scorer(methods, labels, tmp_path / "a.json", "ours", ["comp"], w); r2 = E.run_scorer(methods, labels, tmp_path / "b.json", "ours", ["comp"], w)
    assert r1["primary"] == r2["primary"]
    with pytest.raises(SystemExit):
        E.run_scorer(methods, labels, tmp_path / "a.json", "ours", ["comp"], w)


def test_holm_reference_monotone():
    sys.path.insert(0, "/home/infres/yinwang/CS_QMI/outputs/V7/package_ref/reference"); import v7_metrics as RM
    p = np.array([0.01, 0.04, 0.03, 0.5]); h = RM.holm(p)
    assert np.allclose(h, [0.04, 0.09, 0.09, 0.5]) and np.all(h >= p)


def test_inventory_lists_missing_units(tmp_path):
    csvf = tmp_path / "FIT_UNITS.csv"
    with open(csvf, "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["unit_id", "arm", "loss", "seed"])
        for a in E.ARMS:
            for s in E.SEEDS:
                wr.writerow([f"V8-FIT-{a}-s{s}", a, a.lower(), s])
    units = tmp_path / "units"
    for a in E.ARMS:
        for s in E.SEEDS:
            d = units / f"V8-FIT-{a}-s{s}" / "pred"; d.mkdir(parents=True)
            for n in ("export.json", "genai_rank.jsonl", "genai_bench.jsonl", "images_CAL_PROB.jsonl", "images_DEV.jsonl"):
                (d / n).write_text("{}\n")
            if not (a == "U-LIST" and s == 2):
                (d.parent / "done.json").write_text("{}")
    inv = E.check_units(units, csvf)
    assert inv["n_expected"] == 24 and len(inv["present"]) == 23 and inv["missing"] == {"V8-FIT-U-LIST-s2": ["done.json"]}
    with pytest.raises(SystemExit, match="V8-FIT-U-LIST-s2"):
        E.require_complete(units, csvf)
    (units / "V8-FIT-U-LIST-s2" / "done.json").write_text("{}")
    assert len(E.require_complete(units, csvf)) == 24


def test_distribution_metrics_hand_values():
    recs = {"a": {"image_id": "a", "prompt_id": "p1", "hist": [0, 0, 0.5, 0.5, 0], "tail": [1, 1, 0.5, 0], "mean": 3.5, "w": 1.0},
            "b": {"image_id": "b", "prompt_id": "p2", "hist": [0, 0, 0, 0, 1], "tail": [1, 1, 1, 1], "mean": 5.0, "w": 3.0}}
    pred = [{"image_id": "a", "q": [0, 0, 1, 0, 0], "u": [1, 1, 0, 0], "mu": 3.0}, {"image_id": "b", "q": [0, 0, 0, 0.5, 0.5], "u": [1, 1, 1, 0.5], "mu": 4.5}]
    m = E.distribution_metrics(pred, recs)
    # per-threshold Brier: a -> [0, 0, 0.5, 0]; b -> [0, 0, 0, 0.25]; natural weights 0.25 / 0.75
    assert np.allclose(m["brier_k_natural"], [0, 0, 0.125, 0.1875]) and np.isclose(m["rps_natural"], np.mean([0, 0, 0.125, 0.1875]))
    assert np.isclose(m["hist_logloss_natural"], 0.25 * (-0.5 * np.log(1) - 0.5 * np.log(1e-12)) + 0.75 * (-np.log(0.5)))
    assert np.isclose(m["mean_mse_prompt_macro"], np.mean([0.25, 0.25])) and np.isclose(m["mean_mae_prompt_macro"], 0.5) and np.isclose(m["mean_mse_natural"], 0.25)
    with pytest.raises(ValueError):
        E.distribution_metrics(pred[:1], recs)
    u_only = [{"image_id": "a", "u": [1, 1, 0, 0], "mu": 3.0}, {"image_id": "b", "u": [1, 1, 1, 0.5], "mu": 4.5}]
    assert E.distribution_metrics(u_only, recs, use="u")["tails_monotone"] and np.isclose(E.distribution_metrics(u_only, recs, use="u")["hist_logloss_natural"], m["hist_logloss_natural"])
    crossing = [{"image_id": "a", "u": [1, 0.2, 0.6, 0], "mu": 3.0}, {"image_id": "b", "u": [1, 1, 1, 0.5], "mu": 4.5}]
    assert E.distribution_metrics(crossing, recs, use="u")["hist_logloss_natural"] is None


def test_temperature_fit_and_apply():
    recs = {f"i{k}": {"image_id": f"i{k}", "prompt_id": "p", "hist": [0.2] * 5, "tail": [0.8, 0.6, 0.4, 0.2], "mean": 3.0, "w": 1.0} for k in range(4)}
    pred = [{"image_id": f"i{k}", "logits": [4, 0, 0, 0, 0]} for k in range(4)]            # over-confident -> the uniform target wants a high temperature
    T = E.fit_temperature(pred, recs); assert T > 1.0
    cal = E.apply_temperature(pred, T); assert np.allclose(np.sum(cal[0]["q"]), 1) and np.isclose(cal[0]["mu"], 1 + np.sum(cal[0]["u"]))
    assert E.distribution_metrics(cal, recs)["hist_logloss_natural"] < E.distribution_metrics(E.apply_temperature(pred, 1.0), recs)["hist_logloss_natural"]


def test_markdown_renders_minimal():
    res = {"utc": "t", "units": {"n_expected": 24}, "scorer": {"scorer": "x/v7_score.py", "sha256": "0" * 64}, "external": {"methods_json_sha256": "1" * 64, "subset_excluded_prompts": []}, "labels": {"sealed_sha256": "2" * 64},
           "genai_rank": {}, "lr_choice": {"arms": {a: {"lr": "1e-05", "cal_select_mse": 0.4, "lr_edge": None} for a in E.ARMS}},
           "unit_choices": {f"V8-FIT-{a}-s{s}": {"selected_epoch": 1, "cal_select_mse": 0.4} for a in E.ARMS for s in E.SEEDS},
           "distribution": {f"V8-FIT-{a}-s{s}": {r: {"raw": {"brier_k_natural": [0.1] * 4, "rps_natural": 0.1, "hist_logloss_natural": 1.0, "mean_mse_prompt_macro": 0.4, "mean_mae_prompt_macro": 0.5, "mean_mse_natural": 0.4},
                                                       "calibrated_T_cal_prob": {"rps_natural": 0.1, "hist_logloss_natural": 1.0, "mean_mse_prompt_macro": 0.4}} for r in E.ROLES} for a in E.ARMS for s in E.SEEDS}}
    md = E.markdown(res); assert "## lr rule" in md and "distribution readouts on DEV" in md
