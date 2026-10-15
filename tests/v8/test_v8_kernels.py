"""V8 Table C kernel bridges: PAVA against the package reference, the tail map, and an end-to-end run of each route on synthetic features."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "v8"))
REF = Path("/home/infres/yinwang/CS_QMI/outputs/V8/package_ref_v1/VCS_V8_Utility_Aligned_CS_Server_v1_20261014/reference")
import v8_kernels as K  # noqa: E402
import v8_labels as LB  # noqa: E402


def test_pava_matches_reference():
    sys.path.insert(0, str(REF)); import core as R
    rng = np.random.default_rng(0); V = rng.random((50, 4))
    ours = K.pava_decreasing(V)
    for v, o in zip(V, ours):
        assert np.allclose(o, R.pava_decreasing(v)) and np.all(np.diff(o) <= 1e-12)


def test_tail_map_inverts_the_critic():
    pi = np.array([0.888, 0.563, 0.288, 0.118]); u = np.random.default_rng(1).random((20, 4)) * 0.98 + 0.01
    T = (u - pi) / (pi + (1 - 2 * pi) * u); assert np.allclose(K.to_tail(T, pi), u)


def _synthetic(d: Path, n_fit=400):
    rng = np.random.default_rng(2); d.mkdir(parents=True, exist_ok=True)
    for role, n in (("FIT", n_fit), ("CAL_SELECT", 80), ("CAL_PROB", 60), ("DEV", 60), ("GENAI_RANK", 36)):
        x = rng.normal(size=(n, 8)).astype(np.float32); np.save(d / f"feats_{role}.npy", x); rows = []
        for i in range(n):
            q = 1 / (1 + np.exp(-x[i, 0])); votes = [int(np.clip(round(1 + 4 * q + rng.normal(0, .7)), 1, 5)) for _ in range(3)]; h = LB.histogram(votes)
            r = {"image_id": f"{role}/{i}", "prompt_id": f"p{i // 6}", "w": 1.0}
            if role != "GENAI_RANK":
                r.update({"hist": h.tolist(), "tail": LB.tails(h).tolist(), "mean": float(np.mean(votes))})
            else:
                r.update({"generator_id": "G", "candidate_index": i % 9})
            rows.append(r)
        (d / f"index_{role}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.mark.parametrize("route", ["K-KDE", "K-RULSIF", "K-NYSTROM"])
def test_route_end_to_end(tmp_path, monkeypatch, route):
    _synthetic(tmp_path / "f"); monkeypatch.setattr(K, "CENTRES", (32, 64)); monkeypatch.setattr(K, "EXT", {"mult": (0.125, 8.0), "ridge": (1e-8, 10.0), "centres": 128})
    monkeypatch.setattr(sys, "argv", ["v8_kernels.py", "--features", str(tmp_path / "f"), "--route", route, "--out", str(tmp_path / "o")]); assert K.main() == 0
    rec = json.loads((tmp_path / "o" / "fit.json").read_text()); assert rec["selected"]["cal_select_mse"] == min(r["cal_select_mse"] for r in rec["grid"])
    P = [json.loads(l) for l in open(tmp_path / "o" / "pred_GENAI_RANK.jsonl")]; assert len(P) == 36 and all(1 <= p["mu"] <= 5 and np.all(np.diff(p["u_pava"]) <= 1e-12) for p in P)
    assert rec["metrics"]["DEV"]["mean_mse_prompt_macro"] < 2.0
