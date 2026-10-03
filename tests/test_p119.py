"""P119 tests: readouts are a method-independent map; standardisation uses training statistics only; seed-paired aggregation on a toy table;
the float16 vs float32 comparison path."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vcs_vtask.probe5 import READOUTS, paired_by_seed, prepare, readout_features, standardise  # noqa: E402


def test_readouts_shapes_and_definitions():
    g = torch.Generator().manual_seed(0); h = torch.rand(50, 512, generator=g) * 3
    n = h.norm(dim=1, keepdim=True)
    assert torch.equal(readout_features(h, "raw_unstd"), h) and torch.equal(readout_features(h, "raw_std"), h)
    assert torch.allclose(readout_features(h, "l2_std").norm(dim=1), torch.ones(50), atol=1e-6)
    assert torch.allclose(readout_features(h, "lognorm_std"), n.log()) and readout_features(h, "lognorm_std").shape == (50, 1)
    assert readout_features(h, "l2_lognorm_std").shape == (50, 513)
    with pytest.raises(ValueError):
        readout_features(h, "nope")


def test_readouts_symmetric_across_methods():
    """The readout pipeline has no method argument: two encoders' features go through the identical map; a positive rescaling of h changes only
    the norm feature (l2 part invariant), and standardisation removes a global scale from raw_std."""
    g = torch.Generator().manual_seed(1); h = torch.rand(40, 512, generator=g); h2 = 5.0 * h
    assert torch.allclose(readout_features(h, "l2_std"), readout_features(h2, "l2_std"), atol=1e-6)
    a, _ = prepare("raw_std", h[:30], h[30:]); b, _ = prepare("raw_std", h2[:30], h2[30:])
    assert torch.allclose(a, b, atol=1e-4)
    for k in READOUTS:
        x1, e1 = prepare(k, h[:30], h[30:]); x2, e2 = prepare(k, h[:30].clone(), h[30:].clone())
        assert torch.equal(x1, x2) and torch.equal(e1, e2)


def test_standardise_uses_train_only():
    tr = torch.tensor([[0.0], [2.0]]); ev = torch.tensor([[100.0]])
    t2, e2 = standardise(tr, ev)
    assert torch.allclose(t2.mean(0), torch.zeros(1)) and float(e2) == pytest.approx((100 - 1) / float(tr.std()))


def test_paired_by_seed_toy():
    # family A beats ref by +1, +2, +3 on seeds 0,1,2 (two draws each, draw values ±0.5 around the seed mean)
    ref = {s: {0.01: [80.0 - 0.5, 80.0 + 0.5]} for s in range(3)}
    A = {s: {0.01: [80.0 + (s + 1) - 0.5, 80.0 + (s + 1) + 0.5]} for s in range(3)}
    out = paired_by_seed({"A": A, "R": ref}, "R", ["A", "R"])["A - R @ 0.01"]
    assert out["per_seed"] == pytest.approx([1.0, 2.0, 3.0]) and out["mean"] == pytest.approx(2.0)
    assert out["sd_over_seeds"] == pytest.approx(1.0)
    from scipy import stats
    h = stats.t.ppf(0.975, 2) * 1.0 / np.sqrt(3)
    assert out["ci95"] == pytest.approx([2.0 - h, 2.0 + h])
    assert out["mean_within_seed_draw_sd"] == pytest.approx(np.std([79.5, 80.5], ddof=1))
    assert out["seeds"] == [0, 1, 2]  # draws are averaged within a seed, never counted as seeds


def test_fp16_vs_fp32_comparison_path():
    g = torch.Generator().manual_seed(2); h32 = torch.rand(64, 512, generator=g) * 4; h16 = h32.half()
    rel = float((h16.float() - h32).norm() / h32.norm())
    assert 0 < rel < 1e-3  # float16 rounding of O(1) activations
    a, _ = prepare("raw_std", h16.float()[:48], h16.float()[48:]); b, _ = prepare("raw_std", h32[:48], h32[48:])
    assert float((a - b).abs().max()) < 1e-2
