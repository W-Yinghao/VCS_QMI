"""P141 tests: the four sites (layer3-pool / h / r / z) have the expected shapes for ResNet-18 and ResNet-50 encoders; h equals the encoder's own
forward output (the h of every earlier readout); z = L2(r) with the run's eps and is a deterministic function of r; the per-site task loop has no
site- or method-dependent argument and returns the P124 structure on toy data; the aggregator's P114 labels and seed pairing; the runner skips runs
that are not COMPLETED."""
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
from vcs_vtask import layer_readout as LR  # noqa: E402


def built(cfg_name):
    from vcs_ssl.config import load_config
    from vcs_ssl.models import build_models
    cfg = load_config(str(REPO / "configs" / cfg_name))
    b = build_models(cfg, seed=0, device="cpu")
    b["encoder"].eval(); b["projector"].eval()
    return cfg, b


@pytest.mark.parametrize("cfg_name,l3,h", [("cifar100_hpY_AP3_c100_views4_800ep_seed0.yaml", 256, 512),
                                           ("cifar100_hpR50_AP3_views4_800ep_seed0.yaml", 1024, 2048)])
def test_site_shapes_and_identities(cfg_name, l3, h):
    cfg, b = built(cfg_name)
    x = torch.randn(3, 3, 32, 32, generator=torch.Generator().manual_seed(0))
    eps = float(cfg["model"]["normalization"]["eps"])
    f = LR.site_features(b["encoder"], b["projector"], x, eps)
    assert tuple(f["layer3"].shape) == (3, l3) and tuple(f["h"].shape) == (3, h)
    assert tuple(f["r"].shape) == (3, cfg["model"]["projector"]["output_dim"]) == tuple(f["z"].shape)
    with torch.no_grad():
        assert torch.allclose(f["h"], b["encoder"](x), atol=1e-5)          # same h as the encoder forward (earlier readouts)
        assert torch.allclose(f["r"], b["projector"](f["h"]), atol=1e-6)
    assert torch.allclose(f["z"], F.normalize(f["r"], dim=1, eps=eps)) and torch.allclose(f["z"].norm(dim=1), torch.ones(3), atol=1e-5)
    assert torch.equal(F.normalize(f["r"].clone(), dim=1, eps=eps), f["z"])  # r -> z deterministic


def toy_labels(n=1200, seed=0):
    rng = np.random.default_rng(seed); fine = np.tile(np.arange(100), n // 100); rng.shuffle(fine); coarse = fine // 5
    groups = {g: [5 * g + i for i in range(5)] for g in range(20)}; f2c = np.arange(100) // 5
    return {"fine": fine, "coarse": coarse, "f2c": f2c, "groups": groups}


def test_task_loop_structure_and_symmetry():
    params = list(inspect.signature(LR.tasks_from_features).parameters)
    assert "method" not in params and "site" not in params                 # identical rules for every site and method
    L = toy_labels(); n = len(L["fine"]); rng = np.random.default_rng(1)
    x = torch.as_tensor(rng.normal(size=(n, 16)), dtype=torch.float32) + torch.as_tensor(np.eye(100)[L["fine"]][:, :16] * 3.0, dtype=torch.float32)
    fit, sel = np.arange(0, 1000), np.arange(1000, n)
    out = LR.tasks_from_features(x[fit], x[sel], L, fit, sel, torch.device("cpu"), ("recipe_raw", "knn"), "toy")
    assert out["dim"] == 16 and set(out["tasks"]) == {"coarse", "fine", "conditional"}
    for t in ("coarse", "fine"):
        assert 0.0 <= out["tasks"][t]["recipe_raw"]["acc_pct"] <= 100.0 and "head" not in out["tasks"][t]["recipe_raw"]
    assert set(out["tasks"]["conditional"]["knn"]) == {"macro_pct", "overall_pct", "per_group_pct"}
    d = out["fine_head_decomposition"]
    assert abs(d["fine_acc_pct"] - d["implied_coarse_acc_pct"] * d["fine_given_coarse_correct_pct"] / 100.0) < 1e-6


def test_aggregator_labels_and_pairing():
    import p141_aggregate as A
    assert A.label(np.array([0.1, -0.1, 0.2]))[3] == "close"
    assert A.label(np.array([0.8, 0.9, 1.0, 0.85, 0.95]))[3] == "clear"
    assert A.label(np.array([0.9, -0.2, 1.4]))[3] == "inconclusive"
    assert A.label(np.array([0.5]))[3] == "single seed"
    mk = lambda v: {"sites": {"h": {"tasks": {"coarse": {"recipe_raw": {"acc_pct": v}}}}}}  # noqa: E731
    R = {"P107_AP3_c100_views4_800ep_seed0": mk(72.0), "P107_AP3_c100_views4_800ep_seed1": mk(71.0),
         "P91_c100_simclr_views4_800ep_seed0": mk(70.0), "P91_c100_simclr_views4_800ep_seed2": mk(69.0)}
    c = A.paired(R, "A-P3", "SimCLR", "h", "coarse")
    assert c["seeds"] == [0] and c["delta"] == [2.0] and c["label"] == "single seed"   # only matched seeds are paired


def test_runner_skips_runs_not_completed(tmp_path, monkeypatch):
    import p141_layers
    monkeypatch.setattr(sys, "argv", ["p141_layers.py", "--runs", "NO_SUCH_RUN_seed0", "--out", str(tmp_path), "--cpu"])
    assert p141_layers.main() == 0 and not list(tmp_path.glob("layers_*.json"))
