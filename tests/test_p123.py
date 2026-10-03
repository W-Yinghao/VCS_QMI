"""P123 (v6 V6-GRAD): P115 code untouched; counterfactual with the actual scorer = the training loss / gradients; derivative formulas vs
autograd; projection contributions sum to 1; pair counts; peaks; smoke of the batch diagnostics (VCS cross-view, VCS all-view, SimCLR)."""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as T

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import vcs_diag.augdiag as AD  # noqa: E402
from vcs_ssl.models.critic import CosineCritic, FixedCosineCritic  # noqa: E402

P115_FUNCS = ("crop_overlap", "quant", "summary", "eff_rank", "known_box_views", "objective_terms", "positive_pulls", "flat_grad", "_cos",
              "batch_diagnostics", "geometry", "resolve_checkpoints")


def _cfg(method="vcs_qmi", scope="cross_view_k", nd=True, k=3):
    return {"run": {"method": method}, "objective": {"loss": "negative_J", "simclr_temperature": 0.2},
            "pairing": {"k": k, "negative_detach": nd, "pair_scope": scope}, "model": {"normalization": {"eps": 1e-8, "vcs_and_simclr": "l2"}}}


def _z(V=3, B=6, D=8):
    return F.normalize(torch.randn(V * B, D, dtype=torch.float64), dim=1).requires_grad_(True)


def _src_defs(src: str) -> dict:
    tree = ast.parse(src)
    return {n.name: ast.get_source_segment(src, n) for n in tree.body if isinstance(n, ast.FunctionDef)}


# ---------------------------------------------------------------------------------------------- P115 code unchanged
def test_p115_functions_unchanged_vs_frozen_commit():
    try:
        old = subprocess.run(["git", "-C", str(REPO), "show", "c84b504:src/vcs_diag/augdiag.py"], capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("git history not available")
    new = (REPO / "src" / "vcs_diag" / "augdiag.py").read_text()
    o, n = _src_defs(old), _src_defs(new)
    for f in P115_FUNCS:
        assert o[f] == n[f], f"P115 function {f} changed"
    assert new.startswith(old.rstrip("\n")), "P115 section must be a verbatim prefix of the extended module"


# ---------------------------------------------------------------------------------------------- counterfactual(actual scorer) = training loss
@pytest.mark.parametrize("scope,nd,learned", [("cross_view_k", True, False), ("cross_view_k", False, False), ("cross_view_k", True, True),
                                              ("all_view_tokens", False, False), ("all_view_tokens", True, False)])
def test_cf_with_actual_scorer_equals_training_loss(scope, nd, learned):
    V, B, D = (4, 5, 8) if scope == "all_view_tokens" else (3, 6, 8)
    crit = (CosineCritic(D, scale_init=7.0, bias_init=-3.0) if learned else FixedCosineCritic(D, 2.0, -1.0)).double()
    z = _z(V, B, D); cfg = _cfg(scope=scope, nd=nd)
    ref = AD.objective_terms(cfg, crit, list(z.chunk(V)), pair_seed=11)
    sets = AD.pair_sets(cfg, V, B, pair_seed=11, device="cpu")
    t = AD.cf_terms(z, sets, AD.scorer_from_critic(crit))
    assert float(t["Lp"]) == pytest.approx(float(ref["Lp"]), abs=1e-12) and float(t["Lq"]) == pytest.approx(float(ref["Lq"]), abs=1e-12)
    for a, b in ((t["Lp"], ref["Lp"]), (t["Lq"], ref["Lq"])):
        assert torch.allclose(torch.autograd.grad(a, z, retain_graph=True)[0], torch.autograd.grad(b, z, retain_graph=True)[0], atol=1e-12)
    # the Gram-matrix path (all-view) and the elementwise path agree
    sp, sq = AD.pair_similarities(z, sets)
    zr = z.detach() if nd else z
    assert torch.allclose(sp, (z[sets["pl"]] * z[sets["pr"]]).sum(-1), atol=1e-12) and torch.allclose(sq, (z[sets["ql"]] * zr[sets["qr"]]).sum(-1), atol=1e-12)


def test_pair_counts():
    V, B, k = 4, 5, 3
    s = AD.pair_sets(_cfg(scope="all_view_tokens", nd=False), V, B, pair_seed=0, device="cpu")
    assert len(s["pl"]) == V * B * (V - 1) and len(s["ql"]) == V * B * V * (B - 1)
    assert bool((s["p_img"] == s["pl"] % B).all()) and not bool((s["q_img_l"] == s["q_img_r"]).any())
    s2 = AD.pair_sets(_cfg(scope="cross_view_k", nd=True, k=k), V, B, pair_seed=0, device="cpu")
    npairs = V * (V - 1) // 2
    assert len(s2["pl"]) == B * npairs and len(s2["ql"]) == B * k * npairs and not bool((s2["q_img_l"] == s2["q_img_r"]).any())
    s3 = AD.pair_sets(_cfg(method="simclr_matched"), V, B, pair_seed=0, device="cpu")       # SimCLR counterfactual = all-view, full gradient
    assert s3["scope"] == "all_view_tokens" and s3["negative_detach"] is False


# ---------------------------------------------------------------------------------------------- derivative formulas vs autograd
@pytest.mark.parametrize("spec", [{"name": "x", "kind": "affine", "a": 2.0, "kappa": 0.5, "lam": 0.0}, {"name": "x", "kind": "affine", "a": 3.0, "kappa": 0.25, "lam": 0.0},
                                  {"name": "x", "kind": "curv", "a": 2.0, "kappa": 0.5, "lam": 0.25}, {"name": "x", "kind": "curv", "a": 2.0, "kappa": 0.5, "lam": -0.25}])
def test_dA_ds_matches_autograd(spec):
    s = torch.linspace(-0.99, 0.99, 41, dtype=torch.float64).requires_grad_(True)
    for c in (1.0, -1.0):
        t = torch.tanh(AD.scorer_f(spec, s)); A = c * t - 0.5 * t.square()
        g = torch.autograd.grad(A.sum(), s)[0]
        td = t.detach(); ana = (c - td) * (1 - td.square()) * AD.scorer_fprime(spec, s.detach())
        assert torch.allclose(g, ana, atol=1e-12)
    fp = torch.autograd.grad(AD.scorer_f(spec, s).sum(), s)[0]
    assert torch.allclose(fp, AD.scorer_fprime(spec, s.detach()), atol=1e-12)


def test_curvature_endpoints_monotone_and_lambda0_is_affine():
    s = torch.linspace(-1, 1, 201, dtype=torch.float64)
    aff = {"kind": "affine", "a": 2.0, "kappa": 0.5, "lam": 0.0}
    for lam in (-0.25, 0.25):
        cur = {"kind": "curv", "a": 2.0, "kappa": 0.5, "lam": lam}
        assert float(AD.scorer_f(cur, torch.tensor(0.0))) == pytest.approx(-1.0) and float(AD.scorer_f(cur, torch.tensor(1.0))) == pytest.approx(1.0)
        fp = AD.scorer_fprime(cur, s)
        assert float(fp.min()) >= 2.0 / 4 - 1e-12 and float(fp.max()) <= 7 * 2.0 / 4 + 1e-12
        assert AD.scorer_zero(cur) is not None and abs(AD.scorer_zero(cur) - 0.5) > 1e-3       # zero != κ once λ != 0
    assert torch.allclose(AD.scorer_f({**aff, "kind": "curv"}, s), AD.scorer_f(aff, s))
    assert AD.scorer_zero(aff) == pytest.approx(0.5, abs=1e-9)


def test_affine_peaks_closed_form_and_out_of_range():
    pk = AD.scorer_peaks({"kind": "affine", "a": 2.0, "kappa": 0.5, "lam": 0.0})
    assert pk["pos"]["s_peak_in_range"] == pytest.approx(pk["pos"]["s_peak_closed_form"], abs=3e-4)
    assert pk["neg"]["s_peak_in_range"] == pytest.approx(pk["neg"]["s_peak_closed_form"], abs=3e-4)
    assert pk["pos"]["max_abs_dA_ds_in_range"] == pytest.approx(32 * 2.0 / 27, rel=1e-6)
    pk2 = AD.scorer_peaks({"kind": "affine", "a": 1.0, "kappa": 0.75, "lam": 0.0})      # s− = 0.75 + log2/2 = 1.097 > 1 -> boundary
    assert not pk2["neg"]["closed_form_in_range"] and pk2["neg"]["s_peak_in_range"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------------------------- batch diagnostics smoke + shares sum to 1
def _tf():
    return T.Compose([T.RandomResizedCrop(32, scale=(0.08, 1.0)), T.RandomHorizontalFlip(), T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                      T.RandomGrayscale(0.2), T.ToTensor()])


@pytest.mark.parametrize("method,scope,nd", [("vcs_qmi", "cross_view_k", True), ("vcs_qmi", "all_view_tokens", False), ("simclr_matched", "cross_view_k", True)])
def test_grad_batch_diagnostics_smoke(method, scope, nd):
    torch.manual_seed(0)
    enc = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
    proj = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8))
    crit = FixedCosineCritic(8, 2.0, -1.0) if method == "vcs_qmi" else None
    imgs = np.random.default_rng(1).integers(0, 255, size=(12, 32, 32, 3), dtype=np.uint8)
    views, boxes = AD.known_box_views(imgs, np.arange(8), _tf(), 3, seed=4)
    labels = torch.tensor([0, 1, 0, 1, 2, 2, 0, 1])
    out = AD.grad_batch_diagnostics(_cfg(method=method, scope=scope, nd=nd), enc, proj, crit, views, boxes, torch.device("cpu"), pair_seed=2, labels_img=labels)
    assert len(out["counterfactual"]) == 11
    blocks = [out["actual"]] + list(out["counterfactual"].values())
    for b in blocks:
        rows = [r for r in b["iou_groups_pos"] if r["n_pos"]]
        assert sum(r["enc_proj_contribution"] for r in rows) == pytest.approx(1.0, abs=1e-4)
        assert sum(r["h_proj_contribution"] for r in rows) == pytest.approx(1.0, abs=1e-4)
        assert all(r["enc_norm"] >= 0 for r in rows)
    if method == "vcs_qmi":
        chk = out["actual"]["check_vs_training_loss"]
        assert chk["Lp_rel_err"] < 1e-5 and chk["Lq_rel_err"] < 1e-5
        same = out["counterfactual"]["affine_a2_k0.5"]                                          # actual scorer (2, 0.5) = counterfactual cell
        assert same["Lp"] == pytest.approx(out["actual"]["Lp"], rel=1e-6) and same["grad_vector_norms"]["enc"]["P"] == pytest.approx(out["actual"]["grad_vector_norms"]["enc"]["P"], rel=1e-5)
    for b in out["counterfactual"].values():
        assert b["pos"]["scalar_total_abs_action"] >= 0 and 0 <= b["pos"]["mass_near_peak"] <= 1
        assert "semantic_view_isolated" in b and 0 <= b["semantic_view_isolated"]["frac_same_class_neg"] <= 1
