"""D line (package v2 §8, §9.1): bookkeeping and identity checks for the P93 diagnostics and the P83 addendum-1 supplement.
Exact algebra in float64 at floating-point tolerance; no data, no checkpoints."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from vcs_estim import frozen_supplement as FS  # noqa: E402
from vcs_estim.kernel_cs import dictionary_disagreement, dictionary_upper_bound  # noqa: E402
import diag_same_class_negatives as D2  # noqa: E402
import diag_p72_adapters as D1  # noqa: E402
from precheck_a_adapters import Adapters  # noqa: E402

torch.set_default_dtype(torch.float64)
TOL = 1e-10
RNG = np.random.default_rng(20260928)


def test_local_gate_identity_vcs_and_matched_js():
    """d/df of the VCS per-pair loss -(C T - T^2/2) is (T - C)(1 - T^2); of the matched JS softplus(-2 C f) it is T - C (q = sigma(2f))."""
    f = torch.tensor(RNG.standard_normal(200) * 2, requires_grad=True)
    for Cval in (1.0, -1.0):
        T = torch.tanh(f); lv = -(Cval * T - 0.5 * T ** 2); gv = torch.autograd.grad(lv.sum(), f)[0]
        assert torch.allclose(gv, (T - Cval) * (1 - T ** 2), atol=TOL)
        lj = F.softplus(-2 * Cval * f); gj = torch.autograd.grad(lj.sum(), f)[0]
        assert torch.allclose(gj, T - Cval, atol=TOL)


def test_d1_posterior_logit_gradients_and_swap_ratio():
    """positive_terms: g_post is |d loss / d posterior-logit| (VCS l = 2f, JS l = f); the swap ratio VCS-form / JS-form equals the gate 1 - T^2."""
    torch.manual_seed(0); u = F.normalize(torch.randn(64, 512), dim=-1); v = F.normalize(torch.randn(64, 512), dim=-1)
    m = Adapters("vcs").double(); m.a.data.fill_(3.0); m.b.data.fill_(0.2)
    with torch.no_grad():
        pt = D1.positive_terms("vcs", m, u, v)
    l = (2 * (m.a * (u * v).sum(-1) + m.b)).detach().requires_grad_(True); T = torch.tanh(l / 2); loss = -(T - 0.5 * T ** 2)
    g = torch.autograd.grad(loss.sum(), l)[0]
    assert torch.allclose(pt["g_post"], g.abs(), atol=TOL) and torch.allclose(pt["q"], (1 + T.detach()) / 2, atol=TOL)
    mj = Adapters("js").double(); mj.a.data.fill_(3.0); mj.b.data.fill_(0.2)
    with torch.no_grad():
        pj = D1.positive_terms("js", mj, u, v)
    lj = (mj.a * (u * v).sum(-1) + mj.b).detach().requires_grad_(True); gj = torch.autograd.grad(F.softplus(-lj).sum(), lj)[0]
    assert torch.allclose(pj["g_post"], gj.abs(), atol=TOL)
    q = pt["q"]; Tq = 2 * q - 1; swap_v = (1 - Tq) * (1 - Tq ** 2) / 2; swap_j = 1 - q
    assert torch.allclose(swap_v / swap_j, 1 - Tq ** 2, atol=TOL)


def test_f_same_bookkeeping_numerator_denominator_eps():
    norms = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [4.0, 0.0, 1.0]]); same = torch.tensor([[True, False, True], [False, False, False], [False, False, True]])
    num, den, Fs = D2.same_class_share(norms, same, eps=1e-12)
    assert torch.allclose(num, torch.tensor([4.0, 0.0, 1.0])) and torch.allclose(den, torch.tensor([6.0, 0.0, 5.0]))
    assert abs(float(Fs[0]) - 4 / 6) < TOL and float(Fs[1]) == 0.0 and abs(float(Fs[2]) - 0.2) < TOL


def test_simclr_negative_weights_and_own_row_gradient():
    """Negative softmax weights sum to 1 - w_pos; the own-row gradient (candidates detached) is (1/tau)(sum_j w_ij z_j - z_pos)."""
    torch.manual_seed(1); B, D, tau = 6, 16, 0.2; z1 = F.normalize(torch.randn(B, D), dim=1); z2 = F.normalize(torch.randn(B, D), dim=1)
    x_all = torch.cat([z1, z2]); i = 2
    zi = z1[i].clone().requires_grad_(True); logits = (x_all @ zi) / tau; logits = logits.clone(); logits[i] = -math.inf
    ce = -torch.log_softmax(logits, 0)[i + B]; g = torch.autograd.grad(ce, zi)[0]
    p = torch.softmax(logits.detach(), 0); w_pos = p[i + B]; neg = torch.ones(2 * B, dtype=torch.bool); neg[i] = False; neg[i + B] = False
    assert abs(float(p[neg].sum()) - (1 - float(w_pos))) < TOL
    analytic = ((p[neg, None] * x_all[neg]).sum(0) - (1 - w_pos) * z2[i]) / tau
    assert torch.allclose(g, analytic, atol=1e-9)


def test_bootstrap_by_image_keeps_images_intact():
    """Instances of one image carry offsets that cancel within the image: every replicate mean must lie inside the image-level range."""
    uid = np.repeat(np.arange(30), 3); base = RNG.standard_normal(30); off = np.tile([1000.0, -500.0, -500.0], 30)
    vals = base[uid] + off
    out = D2.bootstrap_by_image({"v": vals}, uid, reps=200)["v"]
    assert out["n_images"] == 30 and base.min() - 1e-9 <= out["ci95"][0] <= out["ci95"][1] <= base.max() + 1e-9
    mr = D2.mass_ratio_boot(np.ones_like(vals), np.ones_like(vals), uid, reps=50)
    assert mr["ratio"] == 1.0 and mr["ci95"] == [1.0, 1.0]


def test_dictionary_identity_bounds_and_G_forms():
    """J(T_w) = sum_j w_j J(T_j) + D(w) exactly; D(w) <= max_w D(w) <= U_dict; the G-based forms agree with kernel_cs' row forms."""
    m = 4; TP = np.tanh(RNG.standard_normal((300, m)) + 1.0); TQ = np.tanh(RNG.standard_normal((300, m)) - 1.0)
    w = RNG.random(m); w /= w.sum(); wM = FS.m_weights(300, 300); Tm = np.concatenate([TP, TQ]); G = FS.gram(Tm, wM)
    J = lambda tp, tn: float((tp - 0.5 * tp ** 2).mean() + (-tn - 0.5 * tn ** 2).mean())
    lhs = J(TP @ w, TQ @ w); rhs = sum(w[j] * J(TP[:, j], TQ[:, j]) for j in range(m)) + dictionary_disagreement(Tm, w, wM)
    assert abs(lhs - rhs) < 1e-12
    assert abs(FS.disagreement_from_G(G, w) - dictionary_disagreement(Tm, w, wM)) < 1e-12
    mx = FS.max_disagreement(G)["max_D"]; U = dictionary_upper_bound(Tm, wM); Ug = FS.bounds_from_G(G, float((wM * (Tm.max(1) - Tm.min(1)) ** 2).sum() / wM.sum()))
    assert FS.disagreement_from_G(G, w) <= mx + 1e-9 <= U["U_dict"] + 1e-9
    assert abs(U["range_bound"] - Ug["range_bound"]) < 1e-12 and abs(U["pair_bound"] - Ug["pair_bound"]) < 1e-12
    for j in range(m):   # vertices: D = 0 and max over the simplex dominates every vertex / edge midpoint
        e = np.eye(m)[j]; assert abs(FS.disagreement_from_G(G, e)) < 1e-12
    bs = FS.block_bootstrap(TP, TQ, {"w": w}, reps=20)
    assert bs["n_blocks"] == 300 and bs["D_w"]["ci95"][0] <= bs["D_w"]["ci95"][1]


def test_displacement_metrics_known_perturbation():
    m = Adapters("vcs").double(); delta = torch.zeros(512, 512); delta[0, 1] = 3.0; delta[5, 5] = 4.0
    m.img.weight.data += delta
    I = torch.eye(512); assert abs(float((m.img.weight - I).norm()) - 5.0) < TOL
    x = F.normalize(torch.randn(8, 512), dim=-1); u_id, _ = Adapters("vcs").double().enc(x, x)
    assert torch.allclose(u_id, x, atol=TOL)   # identity adapter: zero feature displacement by construction


def test_ridge_member_closed_form_is_stationary():
    """The closed-form w* maximises the linear J over the ridge-penalised class: random perturbations do not increase the penalised objective."""
    class P: pass
    n, d = 400, 8; P.xp = F.normalize(torch.randn(n, d), dim=1); P.yp = F.normalize(P.xp + 0.3 * torch.randn(n, d), dim=1)
    P.xq = F.normalize(torch.randn(n, d), dim=1); P.yq = F.normalize(torch.randn(n, d), dim=1)
    member, info = FS.fit_tanh_ridge(P, lam_rel=1e-4)
    one = lambda x: torch.cat([x, torch.ones(len(x), 1)], 1); pp, pq = one(P.xp * P.yp), one(P.xq * P.yq)
    dvec = pp.mean(0) - pq.mean(0); A = 0.5 * (pp.T @ pp / n + pq.T @ pq / n); lam = info["lambda_abs"]
    obj = lambda w: float(w @ dvec - w @ A @ w - lam * (w @ w))
    w = member.w
    for _ in range(20):
        assert obj(w + 1e-3 * torch.randn(d + 1)) <= obj(w) + 1e-12
    f = member(P.xp, P.yp); assert torch.allclose(f, info["c"] * (pp @ w), atol=1e-9)
    assert info["frac_abs_T_gt_1_fit_pos"] >= 0 and 0 < info["c"]


def test_vector_relations_cancellation_index():
    a = torch.tensor([[1.0, 0.0], [1.0, 0.0]]); b = torch.tensor([[-1.0, 0.0], [1.0, 0.0]]); c = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    r = D2.vector_relations(a, b, c)
    assert abs(float(r["cancel"][0])) < 1e-9 and abs(float(r["cancel"][1]) - 1.0) < 1e-9 and abs(float(r["cos_same_pos"][0])) < 1e-9
