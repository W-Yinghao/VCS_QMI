"""V7X-11 anchors: generators, analytic log ratios, closed-form truths vs Monte-Carlo, and a tiny end-to-end run of both commands."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import multivariate_normal as MVN

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "v7x"))
import x11_anchors as A  # noqa: E402


def logpq_scipy(anchor, c, x):
    d = x.shape[1]; lq = MVN(np.zeros(d), np.eye(d)).logpdf(x)
    if anchor == "A1":
        mu = np.zeros(d); mu[0] = c["delta"]; return MVN(mu, np.eye(d)).logpdf(x) - lq
    k = d // 2; S = lambda r: np.block([[np.eye(k), r * np.eye(k)], [r * np.eye(k), np.eye(k)]])
    if anchor == "A2":
        return MVN(np.zeros(d), S(c["rho"])).logpdf(x) - lq
    u, v = x[:, :k], x[:, k:]; tot = 0
    for j in range(k):
        xy = np.stack([u[:, j], v[:, j]], 1); a = MVN(np.zeros(2), [[1, c["rho"]], [c["rho"], 1]]).logpdf(xy); b = MVN(np.zeros(2), [[1, -c["rho"]], [-c["rho"], 1]]).logpdf(xy)
        tot = tot + np.logaddexp(a, b) - math.log(2) - MVN(np.zeros(2), np.eye(2)).logpdf(xy)
    return tot


@pytest.mark.parametrize("anchor,ci", [("A1", 1), ("A1", 5), ("A2", 1), ("A2", 5), ("A3", 0), ("A3", 1)])
def test_log_ratio_matches_scipy(anchor, ci):
    c = A.ANCHORS[anchor][ci]; x = np.random.default_rng(0).normal(size=(50, A.dim(anchor, c))) * 1.3
    assert np.allclose(A.log_ratio(anchor, c, x), logpq_scipy(anchor, c, x), atol=1e-8)


@pytest.mark.parametrize("anchor,ci", [("A1", 2), ("A2", 1), ("A2", 4)])
def test_closed_truths_match_mc(anchor, ci, monkeypatch):
    monkeypatch.setattr(A, "N_TRUTH", 400000); c = A.ANCHORS[anchor][ci]; rng = np.random.default_rng(1); tr = A.truths(anchor, c, rng)
    xp = A.sample(anchor, c, 400000, rng, "P"); lp = A.log_ratio(anchor, c, xp)
    assert lp.mean() == pytest.approx(tr["KL_PQ"], rel=0.02, abs=0.01)
    if tr["pearson_chi2"] < 5:
        assert np.exp(lp).mean() - 1 == pytest.approx(tr["pearson_chi2"], rel=0.05, abs=0.02)
    assert 0 <= tr["mixture_S"] <= 1 and 0 <= tr["PE_half"] <= 1


def test_null_truths_zero():
    for anchor, ci in (("A1", 0), ("A1", 3), ("A2", 0), ("A2", 3)):
        tr = A.truths(anchor, A.ANCHORS[anchor][ci], np.random.default_rng(2))
        assert tr["cs_lebesgue_sqratio"] == pytest.approx(0, abs=1e-12) and tr["KL_PQ"] == pytest.approx(0, abs=1e-12) and tr["mixture_S"] == pytest.approx(0, abs=1e-12)


def test_samples_have_the_declared_law():
    rng = np.random.default_rng(3); c = A.ANCHORS["A2"][5]; x = A.sample("A2", c, 200000, rng, "P"); k = x.shape[1] // 2
    assert np.allclose(np.cov(x.T)[:k, k:], c["rho"] * np.eye(k), atol=0.01)
    c3 = A.ANCHORS["A3"][1]; x3 = A.sample("A3", c3, 200000, rng, "P"); k3 = x3.shape[1] // 2
    assert np.abs(np.cov(x3.T)[:k3, k3:]).max() < 0.01                                      # xor: zero correlation, nonzero dependence
    assert np.mean(x3[:, 0] ** 2 * x3[:, k3] ** 2) > 1.2


def test_end_to_end_tiny(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "N_EVAL", 512); monkeypatch.setattr(A, "N_TRUTH", 20000); monkeypatch.setattr(A, "PERMS", 20)
    assert A.main(["est", "--anchor", "A1", "--out", str(tmp_path), "--reps", "1", "--sizes", "128", "--cells", "2"]) == 0
    lines = [json.loads(l) for l in (tmp_path / "est_A1_c2.jsonl").read_text().splitlines()]
    assert "truth" in lines[0] and lines[1]["K-KDE-S"]["posterior_mse"] >= 0 and "native_KL" in lines[1]["N-DV"]
    assert {r["variant"] for r in lines[1]["K-CS-NATIVE"]} >= {"mult1.0", "scott", "silverman"}
    monkeypatch.setattr(A, "TEST_N", 64)
    assert A.main(["tests", "--anchor", "A2", "--out", str(tmp_path), "--reps", "3"]) == 0
    t = [json.loads(l) for l in (tmp_path / "tests_A2.jsonl").read_text().splitlines()]
    assert len(t) == 6 and all(r["T-HSIC_rejections"] is not None for r in t)
