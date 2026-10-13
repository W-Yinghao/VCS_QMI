"""V7X-21 fitting code on a synthetic X20 directory (CPU torch): file contract, label guard, neural fits beat the prior, the GPU-block kernels
equal the exact NumPy estimators, the A0 classical CS equals the closed KDE form."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "v7x")); sys.path.insert(0, str(ROOT / "scripts" / "v7"))
torch = pytest.importorskip("torch")
import x21_fit as X  # noqa: E402
import x_estimators as XE  # noqa: E402


def make_x20(tmp: Path, n=(1200, 300, 300, 200), seed=0, dim=768, rank=True, shift=0.15):
    """P: N(+shift, I), Q: N(-shift, I) in every coordinate (true log ratio 4 shift sum(x)); soft votes y on a 0.25 grid from the true posterior.
    shift 0.15 is nearly separable (class means 8.3 sd apart: KL near infinite, where NWJ's E_P f term is unbounded); 0.05 overlaps (2.8 sd)."""
    rng = np.random.default_rng(seed); parts = []
    for role, m in zip(("FIT", "CAL_SELECT", "CAL_PROB", "DEV"), n):
        cls = rng.random(m) < 0.55; x = (rng.normal(size=(m, dim)) + np.where(cls, shift, -shift)[:, None]).astype(np.float32)
        post = 1 / (1 + np.exp(-4 * shift * x.sum(1).astype(np.float64))); y = np.round(post * 4) / 4
        np.save(tmp / f"feats_{role}.npy", x); rows = []
        for k in range(m):
            r = {"element_id": f"{role}{k}", "image_id": f"{role}img{k // 3}", "prompt_id": f"{role}p{k // 6}", "w_img": 1.0, "n_el": 3, "n_tok": 2, "z0": 0.0}
            if role != "DEV":
                r["y"] = float(y[k])
            rows.append(r)
        (tmp / f"index_{role}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows)); parts.append({"tag": role, "sha256": f"h{role}", "n_elements": m})
    (tmp / "manifest.json").write_text(json.dumps({"parts": parts}))
    if rank:
        x = rng.normal(size=(90, dim)).astype(np.float32); np.save(tmp / "feats_GENAI_RANK.npy", x)
        idx = [{"element_id": f"r{k}", "image_id": f"G/{k // 3}", "prompt_id": f"rp{k // 27}", "w_img": 1.0, "n_el": 3, "n_tok": 1, "z0": 0.0,
                "generator_id": "G", "candidate_index": (k // 3) % 9} for k in range(90)]
        (tmp / "index_GENAI_RANK.jsonl").write_text("".join(json.dumps(r) + "\n" for r in idx))
        imgs = [{"image_id": f"G/{i}", "prompt_id": f"rp{i // 9}", "O0": 3.0, "n_el": 3, "generator_id": "G", "candidate_index": i % 9} for i in range(30)]
        imgs.append({"image_id": "G/30", "prompt_id": "rp3", "O0": 3.0, "n_el": 0, "generator_id": "G", "candidate_index": 3})
        (tmp / "images_GENAI_RANK.jsonl").write_text("".join(json.dumps(r) + "\n" for r in imgs))
        (tmp / "manifest_rank.json").write_text(json.dumps({"parts": [{"tag": "GENAI_RANK", "sha256": "hR", "n_elements": 90}]}))
    return tmp


def load(tmp):
    F = X.Feats(tmp, "cpu"); F.x20 = tmp; return F


def test_label_guard(tmp_path):
    make_x20(tmp_path, rank=False); idx = (tmp_path / "index_DEV.jsonl").read_text().splitlines()
    (tmp_path / "index_DEV.jsonl").write_text("\n".join(json.dumps({**json.loads(l), "y": 1.0}) for l in idx) + "\n")
    with pytest.raises(SystemExit):
        X.Feats(tmp_path, "cpu")


@pytest.mark.parametrize("method", X.NEURAL)
def test_neural_fit_beats_prior(tmp_path, monkeypatch, method):
    make_x20(tmp_path, shift=0.05); monkeypatch.setattr(X, "BATCH", 128); F = load(tmp_path)
    rec = X.neural_fit(F, method, 1e-3, 0, tmp_path / "out", epochs=8)
    base = F.brier("CAL_SELECT", np.full(len(F.r["CAL_SELECT"]["idx"]), F.prior))
    assert rec["selected_cal_select_brier"] < base and rec["nonfinite_steps"] == 0
    for role in ("CAL_SELECT", "CAL_PROB", "DEV", "GENAI_RANK"):
        assert (tmp_path / "out" / f"pred_{role}.jsonl").exists()
    rk = [json.loads(l) for l in (tmp_path / "out" / "genai_rank.jsonl").read_text().splitlines()]
    assert len(rk) == 31 and sum(r["decomposition_failed"] for r in rk) == 1
    log = json.loads((tmp_path / "out" / "log.json").read_text())
    assert (log["dv_log_normaliser_frozen"] is not None) == (method == "N-DV")


def test_kde_blocks_equal_exact(tmp_path):
    make_x20(tmp_path, n=(150, 40, 40, 20), dim=768, rank=False); F = load(tmp_path); h0 = 30.0
    res, _ = X.kde_logratio(F, [1.0], h0, ["CAL_SELECT"]); a, b = F.masses("FIT"); C = F.r["FIT"]["z"]; q = F.r["CAL_SELECT"]["z"]
    exact = XE.KDE(C[a > 0], h0, a[a > 0]).logpdf(q) - XE.KDE(C[b > 0], h0, b[b > 0]).logpdf(q)
    assert np.allclose(res["CAL_SELECT"][1.0]["ell"], exact, atol=1e-4)


def test_cs_native_equals_closed_kde(tmp_path, monkeypatch):
    make_x20(tmp_path, n=(120, 30, 30, 10), rank=False); F = load(tmp_path); monkeypatch.setattr(X, "MULTS", (1.0,)); monkeypatch.setattr(X, "h0_of", lambda F: 25.0)
    rec = X.cs_native(F, tmp_path / "a0"); a, b = F.masses("FIT"); C = F.r["FIT"]["z"]
    exact = XE.cs_kde(XE.KDE(C[a > 0], 25.0, a[a > 0]), XE.KDE(C[b > 0], 25.0, b[b > 0]))
    assert rec["rows"][0]["D_cs_sqratio"] == pytest.approx(exact, rel=1e-3, abs=1e-6)


@pytest.mark.parametrize("method", X.KERNEL)
def test_kernel_fit_runs_and_ranks(tmp_path, monkeypatch, method):
    make_x20(tmp_path, n=(600, 200, 200, 100), dim=768, rank=False); F = load(tmp_path)
    monkeypatch.setattr(X, "CENTRES", (64,)); monkeypatch.setattr(X, "MULTS", (0.5, 1.0, 2.0)); monkeypatch.setattr(X, "RIDGES", (1e-4, 1e-2))
    rec = X.kernel_fit(F, method, tmp_path / "k")
    preds = [json.loads(l) for l in (tmp_path / "k" / "pred_DEV.jsonl").read_text().splitlines()]
    ell = np.array([p["ell"] for p in preds]); assert np.all(np.isfinite(ell)) and len(rec["grid"]) >= 1
    true = (F.r["DEV"]["z"] * F.sd + F.mu).sum(1); rho = np.corrcoef(np.argsort(np.argsort(ell)), np.argsort(np.argsort(true)))[0, 1]
    assert rho > 0.5


def test_nystrom_block_equals_numpy():
    rng = np.random.default_rng(3); C = rng.normal(size=(30, 5)); x = rng.normal(size=(12, 5)); h = 1.7
    nm = XE.nystrom_map(C, h); ref = XE.nystrom_features(x, nm)
    Ct, xt = torch.tensor(C, dtype=torch.float32), torch.tensor(x, dtype=torch.float32)
    K = torch.exp(-0.5 * X.sqd(Ct, Ct).double() / h ** 2); eps = 1e-6 * float(K.diagonal().mean()); vals, vecs = torch.linalg.eigh(K + eps * torch.eye(30, dtype=torch.float64))
    ours = (torch.exp(-0.5 * X.sqd(xt, Ct).double() / h ** 2) @ ((vecs / torch.sqrt(vals)) @ vecs.T)).numpy()
    assert np.allclose(ours, ref, atol=1e-4)
