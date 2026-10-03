"""P124 tests: the coarse/fine map from the local CIFAR-100 files matches the official meta (a function; 20 coarse × 5 fine; known names); the
conditional readouts are restricted to the 5 fine classes of the given coarse class; readouts have no method argument (symmetric); the 100-way head
decomposition identity; the frozen labels and pattern rule of the aggregator."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
from vcs_vtask import granularity as G  # noqa: E402

HAVE_DATA = G.TRAIN_PICKLE.is_file() and G.META_PICKLE.is_file()


@pytest.mark.skipif(not HAVE_DATA, reason="local CIFAR-100 python files absent")
def test_coarse_fine_map_matches_official_meta():
    L = G.load_labels()
    assert len(L["fine_names"]) == 100 and len(L["coarse_names"]) == 20
    assert sorted(c for fs in L["groups"].values() for c in fs) == list(range(100))  # a partition of the 100 fine classes
    assert all(len(fs) == 5 for fs in L["groups"].values())
    name = lambda i: L["fine_names"][i]  # noqa: E731
    f2c = {name(i): L["coarse_names"][L["f2c"][i]] for i in range(100)}
    # official CIFAR-100 superclass table (cs.toronto.edu/~kriz/cifar.html)
    assert f2c["apple"] == "fruit_and_vegetables" and f2c["beaver"] == "aquatic_mammals" and f2c["aquarium_fish"] == "fish"
    assert f2c["bicycle"] == "vehicles_1" and f2c["tractor"] == "vehicles_2" and f2c["oak_tree"] == "trees" and f2c["baby"] == "people"
    assert np.bincount(L["coarse"], minlength=20).tolist() == [2500] * 20


@pytest.mark.skipif(not HAVE_DATA, reason="local CIFAR-100 python files absent")
def test_pickle_fine_labels_equal_torchvision_order():
    from vcs_ssl.data.cifar import load_cifar100_train
    data = load_cifar100_train(G.CIFAR100_ROOT)
    G.load_labels(data.targets)  # asserts position-by-position equality


def toy_groups():
    return {g: [5 * g + i for i in range(5)] for g in range(20)}


def test_conditional_subset_restricted_to_group():
    groups = toy_groups(); rng = np.random.default_rng(0)
    fine = rng.integers(0, 100, 2000); coarse = fine // 5
    for g in (0, 7, 19):
        pos, lab = G.conditional_subset(fine, coarse, groups, g)
        assert (coarse[pos] == g).all() and set(np.unique(lab)) <= set(range(5))
        assert np.array_equal(np.array(groups[g])[lab], fine[pos])  # remap is invertible within the group


def test_masked_predictions_never_leave_the_given_group():
    groups = toy_groups(); n = 300; g = torch.Generator().manual_seed(1)
    logits = torch.randn(n, 100, generator=g); coarse_given = torch.randint(0, 20, (n,), generator=g)
    logits[:, 99] = 100.0  # an out-of-group class dominates the unrestricted argmax
    pred = G.masked_group_predictions(logits, coarse_given, groups)
    assert ((pred // 5) == coarse_given).all()
    rows = coarse_given != 19
    assert (pred[rows] != 99).all()


def test_fine_head_decomposition_identity():
    rng = np.random.default_rng(2); f2c = np.arange(100) // 5
    y = torch.as_tensor(rng.integers(0, 100, 1000)); pred = y.clone(); flip = torch.as_tensor(rng.random(1000) < 0.4)
    pred[flip] = torch.as_tensor(rng.integers(0, 100, int(flip.sum())))
    d = G.fine_head_decomposition(pred, y, f2c)
    assert abs(d["fine_acc_pct"] - d["implied_coarse_acc_pct"] * d["fine_given_coarse_correct_pct"] / 100.0) < 1e-9


def test_readouts_symmetric_and_deterministic():
    """No method argument: identical features give identical results; a positive rescaling of h leaves l2_std and kNN unchanged."""
    gen = torch.Generator().manual_seed(3); hf = torch.rand(400, 32, generator=gen); hs = torch.rand(100, 32, generator=gen)
    yf = torch.randint(0, 5, (400,), generator=gen); ys = torch.randint(0, 5, (100,), generator=gen)
    dev = torch.device("cpu")
    r1 = G._readouts(hf, hs, yf, ys, 5, 11, dev, readouts=("l2_std", "knn")); r2 = G._readouts(hf.clone(), hs.clone(), yf, ys, 5, 11, dev, readouts=("l2_std", "knn"))
    r3 = G._readouts(4.0 * hf, 4.0 * hs, yf, ys, 5, 11, dev, readouts=("l2_std", "knn"))
    for k in ("l2_std", "knn"):
        assert r1[k]["acc_pct"] == r2[k]["acc_pct"]
        assert abs(r1[k]["acc_pct"] - r3[k]["acc_pct"]) < 1e-6


def test_aggregator_labels_and_pattern():
    import p124_aggregate as A
    assert A.label(0.2, -1, 1) == "close" and A.label(0.5, 0.1, 0.9) == "clear +" and A.label(-0.6, -1.0, -0.2) == "clear −"
    assert A.label(0.8, -0.3, 1.9) == "inconclusive"
    assert A.pattern("clear +", "close").startswith("coarse-specific") and A.pattern("close", "clear +").startswith("within-coarse")
    assert A.pattern("clear +", "clear +") == "both up" and A.pattern("close", "close").startswith("neither")
    assert A.pattern("inconclusive", "close").startswith("no pattern")
    p = A.paired({0: 1.0, 1: 1.2, 2: 0.8}, {0: 0.0, 1: 0.1, 2: -0.1})
    assert abs(p["mean"] - 1.0) < 1e-9 and p["label"] == "clear +" and abs(p["per_seed"][1] - 1.1) < 1e-9
