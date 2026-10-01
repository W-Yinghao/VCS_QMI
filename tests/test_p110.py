"""P110 unit tests (no encoders, no data files)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from vcs_vtask import common, corruptions, v3  # noqa: E402
from vcs_vtask.v2 import run_cifar10c  # noqa: E402


def test_stratified_subset_fixed_and_balanced():
    y = np.repeat(np.arange(10), 4500)
    a = common.stratified_subset(y, 0.01, 7); b = common.stratified_subset(y, 0.01, 7); c = common.stratified_subset(y, 0.01, 8)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert len(a) == 450 and (np.bincount(y[a]) == 45).all()
    assert len(common.stratified_subset(y, 1.0, 0)) == len(y)


def test_inner_split_disjoint_stratified():
    y = np.repeat(np.arange(10), 45)
    tr, va = common.inner_split(y, 3)
    assert not set(tr) & set(va) and len(tr) + len(va) == len(y) and (np.bincount(y[va]) == 9).all()


def test_probe_selected_learns_separable_and_chooses_in_grid():
    g = torch.Generator().manual_seed(0); y = torch.arange(4).repeat_interleave(60)
    x = torch.randn(240, 8, generator=g) * 0.1; x[torch.arange(240), y] += 3.0
    r = common.fit_probe_selected(x, y, 4, torch.device("cpu"), sel_seed=1)
    assert r["chosen"] in common.PROBE_GRID and common.accuracy(r["head"], x, y, torch.device("cpu")) > 95


def test_corruptions_deterministic_and_shaped():
    rng = np.random.default_rng(0); im = rng.integers(0, 256, size=(6, 32, 32, 3), dtype=np.uint8); u = np.arange(100, 106)
    for f, s in corruptions.cells():
        a = corruptions.corrupt(im, u, f, s); b = corruptions.corrupt(im, u, f, s)
        assert a.shape == im.shape and a.dtype == np.uint8 and np.array_equal(a, b), (f, s)
    assert len(corruptions.cells()) == 30
    n1 = corruptions.corrupt(im, u, "gaussian_noise", 1); n5 = corruptions.corrupt(im, u, "gaussian_noise", 5)
    assert np.abs(n5.astype(int) - im).mean() > np.abs(n1.astype(int) - im).mean()


def test_c10c_refuses_without_authorisation():
    with pytest.raises(PermissionError):
        run_cifar10c(None, torch.device("cpu"), authorised=False)


def test_v3_splits_disjoint_and_cues():
    sp = v3.base_splits(np.arange(45000))
    allu = np.concatenate(list(sp.values())); assert len(allu) == len(set(allu)) == 40000
    im = np.full((3, 32, 32, 3), 100, dtype=np.uint8); k = np.array([0, 3, 9])
    t = v3.cue_tag(im, k); assert (t[1, :8, :8] == v3.PALETTE[3]).all() and (t[:, 9:, 9:] == 100).all()
    c = v3.cue_colour(im, k); assert np.allclose(c[2, 10, 10], np.round(0.75 * 100 + 0.25 * v3.PALETTE[9]), atol=1)
    assert len({tuple(p) for p in v3.PALETTE}) == 10


def test_v3_draw_n_rates():
    y = np.repeat(np.arange(10), 20000); n = v3.draw_n(y, 0.9, 0.1, np.random.default_rng(0))
    assert abs(n[y % 2 == 0].mean() - 0.9) < 0.01 and abs(n[y % 2 == 1].mean() - 0.1) < 0.01


def test_v3_selection_rules():
    mem = [{"idval_acc": a, "test_acc": t, "test_worst_group": w, "vcs_closed_J": s1, "js_exact": s2, "hsic_class": s3}
           for a, t, w, s1, s2, s3 in [(90.0, 60, 20, 0.5, 0.5, 0.5), (89.5, 80, 70, 0.1, 0.2, 0.3), (88.0, 85, 80, 0.0, 0.0, 0.0)]]
    s = v3.select(mem)
    assert s["A"]["member"] == 0 and s["eligible"] == [0, 1]           # member 2 is outside δ = 1.0
    assert s["B_vcs_closed_J"]["member"] == 1 and s["B_js_exact"]["member"] == 1
    assert s["C_random_eligible"]["test_acc"] == 70 and s["pool_best_test"] == 85


def test_v3_member_training_and_audit_shapes():
    rng = np.random.default_rng(0); n_img = 600; y = np.repeat(np.arange(10), 60)
    Hc = torch.randn(n_img, 16); Hq = Hc.clone(); Hq[:, 0] += 3.0                     # the cue shifts one coordinate
    n = v3.draw_n(y, 0.9, 0.1, rng)
    m, mu, sd = v3.train_member(Hc, Hq, y, n, {"head": "linear", "wd": 0.0, "aug": 0.0}, seed=0, dev=torch.device("cpu"), epochs=2)
    L = v3.logits(m, mu, sd, Hc, Hq, n, torch.device("cpu")); assert L.shape == (n_img, 10)
    ya = np.repeat(np.arange(10), 500); Ha = torch.randn(5000, 16); Hqa = Ha.clone(); Hqa[:, 0] += 3.0; na = v3.draw_n(ya, 0.8, 0.2, rng)
    st = v3.audit_stats(v3.logits(m, mu, sd, Ha, Hqa, na, torch.device("cpu")), ya, na, np.random.default_rng(1), seed=0)
    assert set(st) == set(v3.STATS) and all(np.isfinite(v) for v in st.values())


def test_v3_class_conditional_audit_detects_class_specific_shift():
    """A cue that moves logit c+1 up for class c is invisible to a class-agnostic critic on average but must give J > 0 class-conditionally."""
    rng = np.random.default_rng(0); y = np.repeat(np.arange(10), 500); n = v3.draw_n(y, 0.8, 0.2, rng)
    L = torch.randn(5000, 10); L[np.arange(5000), (y + 1) % 10] += 2.0 * torch.as_tensor(n).float()
    st = v3.audit_stats(L, y, n, np.random.default_rng(1), seed=0)
    L0 = torch.randn(5000, 10); st0 = v3.audit_stats(L0, y, n, np.random.default_rng(1), seed=0)
    assert st["vcs_closed_J"] > 0.02 and st["vcs_closed_J"] > st0["vcs_closed_J"] + 0.02 and st["js_exact"] > st0["js_exact"] + 0.02
