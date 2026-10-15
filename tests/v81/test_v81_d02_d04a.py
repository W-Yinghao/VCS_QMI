"""CPU tests for the V8.1 D02 / D03 / D04a helpers (no server data; the package reference is imported by path)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "v81"))
import v81_common as C  # noqa: E402
import d02_normalize as N  # noqa: E402

core = C.load_core()


def test_prenorm_fixes_simplex_and_keeps_score(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(20):
        q = rng.dirichlet(np.ones(5)).astype(np.float32).astype(float)   # float32 rounding: sum != 1 at 1e-8
        q[0] += 1.5e-7
        u = C.tails(q)
        rows.append({"image_id": f"g/{i}.png", "prompt_id": str(i // 4), "q": list(q), "u": list(u), "mu": float(1 + u.sum())})
    src = tmp_path / "pred.jsonl"
    C.write_jsonl(src, rows)
    out = tmp_path / "pre.jsonl"
    meta = N.prenorm(src, out)
    new = C.read_jsonl(out)
    assert meta["deviations"]["max_abs_sum_q_minus_1"] > 1e-8
    for r0, r1 in zip(rows, new):
        core.simplex(r1["q"])
        assert r1["mu"] == r0["mu"]
        assert np.allclose(r1["u"], C.tails(r1["q"]))
    assert N.prenorm(src, out) == meta   # idempotent: existing output is reused


def test_external_rows_follow_adapt_v8_candidate_order():
    recs = {"b/1.png": {"prompt_id": "7", "prompt_group": "g7", "overall_votes": [3, 4, 4]}, "a/1.png": {"prompt_id": "7", "prompt_group": "g7", "overall_votes": [5, 5, 4]},
            "c/2.png": {"prompt_id": "8", "prompt_group": "g8", "overall_votes": [1, 2, 2]}}
    per = {}
    for i, r in recs.items():
        per.setdefault(r["prompt_id"], []).append(i)
    order = {i: k for ids in per.values() for k, i in enumerate(sorted(ids, key=str))}   # adapt_v8.py rule
    rows, meta = N.rows_from_scores("X", {"a/1.png": 0.2, "b/1.png": 0.9, "c/2.png": 0.5}, "native", -1, "released", recs, order)
    assert [r["image_id"] for r in rows] == ["a/1.png", "b/1.png", "c/2.png"]
    assert [r["candidate_index"] for r in rows] == [0, 1, 0]
    assert meta["score_kind"] == "native_score" and meta["n_missing_dev_images"] == 0
    rows2, meta2 = N.rows_from_scores("Y", {"a/1.png": 1.5, "b/1.png": 4.9, "c/2.png": 2.0}, "regressor_1_5", 0, "selected", recs, order)
    assert meta2["score_kind"] == "rating_mean"
    core.validate_rows(rows2, labelled=True)
    rows3, meta3 = N.rows_from_scores("Z", {"a/1.png": 0.5}, "regressor_1_5", 0, "selected", recs, order)
    assert meta3["n_missing_dev_images"] == 2


def test_macro_decomposition_identities():
    rng = np.random.default_rng(1)
    errs = [rng.normal(0.3, 1.0, n) for n in (3, 5, 7, 1)]
    d = C.macro_decomposition(errs)
    assert abs(d["identity_residual"]) < 1e-12 and abs(d["identity_residual_between"]) < 1e-12
    assert d["mse_macro"] == pytest.approx(np.mean([(e * e).mean() for e in errs]))
    pool = core.mse_decomposition(errs[1] + 2.0, np.full(5, 2.0))
    assert pool["bias_squared"] + pool["centered_mse"] == pytest.approx(pool["mse"])


def test_pooled_spearman_cluster_bootstrap_paired():
    rng = np.random.default_rng(2)
    n = 300
    y = rng.normal(size=n)
    clusters = [str(i // 5) for i in range(n)]
    S = np.stack([y, y + 0.1 * rng.normal(size=n), rng.normal(size=n)])
    B = C.pooled_spearman_cluster_bootstrap(S, y, clusters, reps=200, seed=3)
    assert B["level"][0] == pytest.approx(1.0) and B["ci95"][0][0] == pytest.approx(1.0) and B["ci95"][0][1] == pytest.approx(1.0)
    assert B["ci95"][1][0] > 0.9
    assert B["ci95"][2][0] < 0 < B["ci95"][2][1]
    d = B["boot"][:, 1] - B["boot"][:, 0]
    assert np.all(d <= 1e-12)            # paired replicate differences: model 1 never beats the perfect model 0
    assert B["n_clusters"] == 60 and B["boot"].shape == (200, 3)


def test_label_free_rows_reject_labels():
    row = {"dataset": "GenAI-Rank", "role": "GENAI_RANK", "method": "U-CS", "seed": 0, "checkpoint_id": "best_e2", "prompt_id": "p", "pool_id": "p|g",
           "image_id": "g/p_1.jpeg", "candidate_index": 0, "score": 3.0, "score_kind": "rating_mean"}
    core.validate_rows([row], labelled=False)
    with pytest.raises(ValueError):
        core.validate_rows([{**row, "votes": [3, 3, 3]}], labelled=False)
    with pytest.raises(ValueError):
        core.validate_rows([{**row, "human_mean": 3.0}], labelled=True)   # role not allowed for labelled diagnostics


def test_model_name_and_group_pools():
    rows = [{"dataset": "d", "role": "DEV", "method": "m", "seed": 0, "checkpoint_id": "c", "prompt_id": "1", "pool_id": "1", "image_id": "b", "candidate_index": 1, "score": 2.0, "cluster_id": "g"},
            {"dataset": "d", "role": "DEV", "method": "m", "seed": 0, "checkpoint_id": "c", "prompt_id": "1", "pool_id": "1", "image_id": "a", "candidate_index": 0, "score": 3.0, "cluster_id": "g"}]
    pools = C.group_pools(rows)
    assert [r["image_id"] for r in pools["1"]] == ["a", "b"]
    assert C.model_name(rows[0]) == "m|s0|c"
    with pytest.raises(ValueError):
        C.group_pools([rows[0], {**rows[1], "cluster_id": "other"}])


def test_j_readout_equals_s_at_the_true_ratio_and_is_lower_elsewhere():
    import d04a_distribution as D4
    rng = np.random.default_rng(5)
    ids, recs, pred_true, pred_off = [], {}, {}, {}
    hists = rng.dirichlet(np.ones(5), size=40)
    prompts = [str(i // 4) for i in range(40)]
    for n in range(40):
        ids.append(f"g/{n}.png")
        recs[ids[-1]] = {"hist": hists[n], "prompt_id": prompts[n]}
    n_p = {p: prompts.count(p) for p in set(prompts)}
    w = np.array([1.0 / n_p[p] for p in prompts]); w /= w.sum()
    Y = C.tails(hists)
    prior = [0.85, 0.55, 0.3, 0.12]
    T_true = np.empty((40, 4))
    S = []
    for k in range(4):
        p = w * Y[:, k] / (w @ Y[:, k]); q = w * (1 - Y[:, k]) / (w @ (1 - Y[:, k]))
        eta = (p - q) / (p + q)
        T_true[:, k] = eta
        S.append(float(((p + q) / 2) @ eta ** 2))
    def u_from_T(T):
        ell = 2 * np.arctanh(np.clip(T, -1 + 1e-12, 1 - 1e-12))
        lp = np.log(np.array(prior) / (1 - np.array(prior)))
        return 1 / (1 + np.exp(-(ell + lp)))
    U_true = u_from_T(T_true); U_off = u_from_T(np.clip(T_true + 0.2 * rng.normal(size=T_true.shape), -0.99, 0.99))
    for n, i in enumerate(ids):
        pred_true[i] = {"u": list(U_true[n])}
        pred_off[i] = {"u": list(U_off[n])}
    jt = D4.j_readout(ids, pred_true, recs, prior)
    jo = D4.j_readout(ids, pred_off, recs, prior)
    for k in range(4):
        assert jt[f"J_Hge{k+2}"] == pytest.approx(S[k], abs=1e-9)
        assert jo[f"J_Hge{k+2}"] < jt[f"J_Hge{k+2}"]
    assert jt["role_tail_marginal"] == pytest.approx(list(w @ Y))


def test_delta_matrix_is_antisymmetric():
    import d02_agreement as A
    rows = [{"kind": "v8_same_seed", "a_method": "U-CS", "b_method": "U-BCE", "a_seed": s, "b_seed": s, "overall_delta_a_minus_b": 0.1 * (s + 1), "top1_agreement": 0.8} for s in C.SEEDS]
    M = A.matrices(rows, "overall_delta_a_minus_b")["mean"]
    assert M[0, 1] == pytest.approx(0.2) and M[1, 0] == pytest.approx(-0.2)
    T = A.matrices(rows, "top1_agreement")["mean"]
    assert T[0, 1] == T[1, 0] == pytest.approx(0.8)


def test_exact_human_mean_keeps_rational_ties():
    a = {"votes": [3, 3, 5]}; b = {"votes": [3, 4, 4]}; c = {"votes": [4, 4, 3, 3, 4, 4]}
    assert C.human_mean_of(a, core) == C.human_mean_of(b, core)
    assert C.human_mean_of(c, core) == pytest.approx(22 / 6)
    ha, hb = core.histogram(a["votes"]) @ C.LEVELS, core.histogram(b["votes"]) @ C.LEVELS
    assert abs(ha - hb) < 1e-12     # the package value is equal only up to rounding (it may differ by 1 ulp)


def test_j_from_counts_matches_j_readout_at_unit_multiplicity():
    import d04a_distribution as D4
    rng = np.random.default_rng(9)
    ids = [f"g/{n}.png" for n in range(30)]
    recs = {i: {"hist": rng.dirichlet(np.ones(5)), "prompt_id": str(n // 5)} for n, i in enumerate(ids)}
    pred = {i: {"u": sorted(rng.uniform(0.05, 0.95, 4), reverse=True)} for i in ids}
    prior = [0.85, 0.55, 0.3, 0.12]
    w, Y, T = D4.j_terms(ids, pred, recs, prior)
    j1 = D4.j_from_counts(np.ones((1, len(ids))), w, Y, T)[0]
    assert j1 == pytest.approx(D4.j_readout(ids, pred, recs, prior)["J_mean"], abs=1e-12)
    j2 = D4.j_from_counts(2 * np.ones((1, len(ids))), w, Y, T)[0]       # uniform multiplicity cancels
    assert j2 == pytest.approx(j1, abs=1e-12)
