"""R22-B CPU checks (run on Slurm cpu-high in the NCR env v9_sgr):

    python -I tests/test_r22_posterior.py --out <CPU_CHECKS.json>

 1 posterior: stable form == direct (1-rho)(1+T)/(1+(1-2rho)T) == package reference core.clean_posterior
 2 recovery: synthetic P/Q with a linear log-ratio (critic class contains T*), U = (1-rho)P + rho Q at real Flickr role sizes:
   both arms recover rho, the critic (a*, b*) and the true posterior
 3 fallback: gap ~ 0, rho_raw >= 1, rho_raw <= 0 -> GMM probability returned unchanged; deployed -> anchors 1, U posterior
 4 Q pairs: never the caption's own image, donors from the same role, deterministic per (seed, epoch), redrawn across epochs
 5 N2F cross-check: same selected critic / J / rho as src/v9/n2/weights.fit_scalar_critic + estimate_rho (N2F code, float32 parameters);
   the numpy analytic-gradient Adam equals torch autograd + torch.optim.Adam in float64
 6 official SGRAF on CPU: eval_loss_and_diag == SGRAF.train(mode="eval_loss") bitwise (loss, Eiters); score_pairs == the batch diagonal
 7 hook: replace_probabilities on a toy dataset (official collate_fn) writes the ledger / arrays and returns per-network probabilities
 8 patch layer: the three patches apply to the official clone, the patched files compile, the posterior module is byte-identical
"""
import argparse
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
REPO = SRC.parents[1]
CLONE = Path("/projects/EEG-foundation-model/yinghao/code_v9/2021-NeurIPS-NCR")
sys.path.insert(0, str(SRC))
import r22_posterior as R  # noqa: E402


def load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def gauss_logpdf(x, mu, sd):
    return -0.5 * ((x - mu) / sd) ** 2 - math.log(sd * math.sqrt(2 * math.pi))


def synth(rng, rho, sizes=(4350, 2175, 7975, 130500), mu=(0.65, 0.35), sd=0.15):
    """P: s ~ N(mu1, sd^2), Q: s ~ N(mu0, sd^2) -> f* = log(p/q)/2 = a* s + b* with a* = (mu1-mu0)/(2 sd^2) > 0."""
    n_fit, n_sel, n_ref, n_u = sizes
    P = lambda n: rng.normal(mu[0], sd, n)
    Q = lambda n: rng.normal(mu[1], sd, n)
    noisy = rng.random(n_u) < rho
    sU = np.where(noisy, Q(n_u), P(n_u))
    a_star = (mu[0] - mu[1]) / (2 * sd ** 2)
    b_star = -(mu[0] ** 2 - mu[1] ** 2) / (4 * sd ** 2)
    lp, lq = gauss_logpdf(sU, mu[0], sd), gauss_logpdf(sU, mu[1], sd)
    w_true = 1 / (1 + rho / (1 - rho) * np.exp(lq - lp))
    return {"fit": (P(n_fit), Q(n_fit)), "sel": (P(n_sel), Q(n_sel)), "ref": (P(n_ref), Q(n_ref)), "U": sU, "noisy": noisy,
            "a_star": a_star, "b_star": b_star, "w_true": w_true, "rho_real": float(noisy.mean())}


def arrays_for_decide(d):
    """Lay the synthetic roles out as decide() expects: anchors (FIT critic_fit / SELECT critic_select / REF reference_moments), then U."""
    sP = np.r_[d["fit"][0], d["sel"][0], d["ref"][0]]
    sQ = np.r_[d["fit"][1], d["sel"][1], d["ref"][1]]
    roles_anchor = np.r_[np.full(len(d["fit"][0]), 1), np.full(len(d["sel"][0]), 2), np.full(len(d["ref"][0]), 3)].astype(np.int8)
    n_a = len(sP)
    s_all = np.r_[sP, d["U"]]
    anchor_idx, u_idx = np.arange(n_a), np.arange(n_a, n_a + len(d["U"]))
    gmm = np.random.default_rng(5).random(len(s_all))
    return s_all, sQ, anchor_idx, roles_anchor, u_idx, gmm


def t_posterior():
    ref = load_path("r22_ref_core", REPO / "v9/package_v9_20261016/reference/core.py")
    rng = np.random.default_rng(1)
    f = rng.uniform(-8, 8, 20000)
    out = {}
    for rho in (0.05, 0.4, 0.9):
        st, di, rc = R.posterior_stable(f, rho), R.posterior_direct(np.tanh(f), rho), ref.clean_posterior(np.tanh(f), rho)
        out[str(rho)] = {"stable_vs_direct": float(np.abs(st - di).max()), "stable_vs_reference": float(np.abs(st - rc).max())}
    big = R.posterior_stable(np.array([-400.0, 400.0]), 0.4)
    ok = all(v["stable_vs_direct"] < 1e-12 and v["stable_vs_reference"] < 1e-12 for v in out.values()) and np.isfinite(big).all() \
        and big[0] < 1e-300 and big[1] == 1.0
    return ok, {"max_abs_diff": out, "extreme_f": big.tolist()}


def t_recovery():
    rng = np.random.default_rng(22)
    res, ok = {}, True
    for rho in (0.2, 0.4, 0.6):
        d = synth(rng, rho)
        s_all, sQ, anchor_idx, roles_anchor, u_idx, gmm = arrays_for_decide(d)
        for kind in ("vcs", "bce"):
            prob, rec = R.decide(s_all, sQ, anchor_idx, roles_anchor, u_idx, gmm, kind)
            c = rec["critic"]
            w = prob[u_idx]
            r = {"rho_real": d["rho_real"], "rho_hat": rec["rho"]["raw"], "gap": rec["rho"]["gap"], "status": rec["rho"]["status"],
                 "a_hat": c["a"], "a_star": d["a_star"], "b_hat": c["b"], "b_star": d["b_star"], "selected": [c["lr"], c["init"]],
                 "mean_abs_w_minus_true": float(np.abs(w - d["w_true"]).mean()), "w_mean": float(w.mean()),
                 "true_w_mean": float(d["w_true"].mean()), "anchors_one": bool((prob[anchor_idx] == 1).all()),
                 "fit_seconds": rec["fit_seconds"], "formula_max_abs_diff": rec["formula_max_abs_diff"]}
            good = (rec["deployed"] and abs(r["rho_hat"] - d["rho_real"]) < 0.03 and abs(r["a_hat"] / d["a_star"] - 1) < 0.1
                    and abs(r["b_hat"] - d["b_star"]) < 0.1 * abs(d["b_star"]) and r["mean_abs_w_minus_true"] < 0.02 and r["anchors_one"])
            r["pass"] = bool(good)
            ok &= good
            res[f"rho{rho}_{kind}"] = r
    return ok, res


def t_fallback():
    rng = np.random.default_rng(3)
    res, ok = {}, True
    d = synth(rng, 0.4, sizes=(4350, 2175, 7975, 20000))
    s_all, sQ, anchor_idx, roles_anchor, u_idx, gmm = arrays_for_decide(d)
    cases = {}
    same = s_all.copy()                                            # Q drawn like P: no separation
    sQ_same = rng.normal(0.65, 0.15, len(sQ))
    cases["no_gap"] = (same, sQ_same)
    beyond_q = s_all.copy()
    beyond_q[u_idx] = rng.normal(0.15, 0.1, len(u_idx))            # U below Q: rho_raw > 1
    cases["rho_ge_1"] = (beyond_q, sQ)
    beyond_p = s_all.copy()
    beyond_p[u_idx] = rng.normal(0.9, 0.05, len(u_idx))            # U above P: rho_raw < 0
    cases["rho_le_0"] = (beyond_p, sQ)
    for name, (sa, sq) in cases.items():
        for kind in ("vcs", "bce"):
            prob, rec = R.decide(sa, sq, anchor_idx, roles_anchor, u_idx, gmm, kind)
            r = {"status": rec["rho"]["status"], "rho_raw": rec["rho"].get("raw"), "gap": rec["rho"].get("gap"), "fallback": rec["fallback"],
                 "prob_is_gmm": bool(np.array_equal(prob, gmm)), "prob_source": rec["prob_source"]}
            want = {"no_gap": "weak_or_reversed_empirical_separation", "rho_ge_1": "outside_open_unit_interval",
                    "rho_le_0": "outside_open_unit_interval"}[name]
            r["pass"] = bool(rec["fallback"] and r["prob_is_gmm"] and r["status"] in (want, "unidentified_at_numerical_tolerance"))
            if name == "rho_ge_1":
                r["pass"] &= r["rho_raw"] is not None and r["rho_raw"] >= 1
            if name == "rho_le_0":
                r["pass"] &= r["rho_raw"] is not None and r["rho_raw"] <= 0
            ok &= r["pass"]
            res[f"{name}_{kind}"] = r
    st = {"gap_.009": R.rho_moments(np.full(5, .509), np.full(5, .5), np.full(5, .505))["status"],
          "gap_.011": R.rho_moments(np.full(5, .511), np.full(5, .5), np.full(5, .505))["status"],
          "raw_0": R.rho_moments(np.full(5, .9), np.full(5, .1), np.full(5, .9))["status"],
          "raw_1": R.rho_moments(np.full(5, .9), np.full(5, .1), np.full(5, .1))["status"],
          "nan": R.rho_moments(np.array([np.nan]), np.full(5, .1), np.full(5, .1))["status"]}
    good = st == {"gap_.009": "weak_or_reversed_empirical_separation", "gap_.011": "point_estimate", "raw_0": "outside_open_unit_interval",
                  "raw_1": "outside_open_unit_interval", "nan": "nonfinite_moment"}
    res["status_boundaries"] = {"statuses": st, "pass": bool(good)}
    return ok and good, res


def t_repair():
    n_img = 600
    t2i = np.repeat(np.arange(n_img), 5)
    rng = np.random.default_rng(4)
    img_role = rng.integers(-1, 4, n_img)
    roles = img_role[t2i].astype(np.int8)
    anchor_idx = np.flatnonzero(roles >= 0)
    ra = roles[anchor_idx]
    q1 = R.repair_within_role(anchor_idx, ra, t2i, 901, 0)
    q1b = R.repair_within_role(anchor_idx, ra, t2i, 901, 0)
    q2 = R.repair_within_role(anchor_idx, ra, t2i, 901, 1)
    role_of_img = dict(zip(range(n_img), img_role))
    res = {"own_image_kept": int(np.sum(q1 == t2i[anchor_idx])), "donor_role_mismatch": int(sum(role_of_img[int(i)] != int(r) for i, r in zip(q1, ra))),
           "deterministic": bool(np.array_equal(q1, q1b)), "frac_changed_next_epoch": float(np.mean(q1 != q2)),
           "donor_image_coverage": float(len(set(q1.tolist())) / len(set(t2i[anchor_idx].tolist())))}
    ok = res["own_image_kept"] == 0 and res["donor_role_mismatch"] == 0 and res["deterministic"] and res["frac_changed_next_epoch"] > 0.9
    return ok, res


def torch_adam_reference(sp, sq, kind, lr, init, steps):
    """The same fit with torch autograd (float64 parameters) and torch.optim.Adam."""
    import torch.nn.functional as F
    tp, tq = torch.as_tensor(np.asarray(sp, dtype=np.float64)), torch.as_tensor(np.asarray(sq, dtype=np.float64))
    a_raw = torch.tensor(float(init[0]), dtype=torch.float64, requires_grad=True)
    b = torch.tensor(float(init[1]), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([a_raw, b], lr=lr)
    for _ in range(steps):
        fp, fq = F.softplus(a_raw) * tp + b, F.softplus(a_raw) * tq + b
        if kind == "vcs":
            loss = .5 * (1 - torch.tanh(fp)).square().mean() + .5 * (1 + torch.tanh(fq)).square().mean()
        else:
            loss = .5 * F.softplus(-2 * fp).mean() + .5 * F.softplus(2 * fq).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return np.array([float(a_raw.detach()), float(b.detach())])


def t_n2f():
    sys.path.insert(0, str(REPO / "src"))
    from v9.n2 import weights as W                                 # N2F implementation (float32 parameters, torch Adam)
    rng = np.random.default_rng(9)
    d = synth(rng, 0.4)
    res, ok = {}, True
    small = synth(np.random.default_rng(10), 0.4, sizes=(1500, 10, 10, 10))
    for kind in ("vcs", "bce"):
        for lr, init in ((1e-1, (0.0, 0.0)), (1e-2, (2.0, -1.0))):
            th_np = R.adam_fit(small["fit"][0], small["fit"][1], kind, lr, init, 400)
            th_t = torch_adam_reference(small["fit"][0], small["fit"][1], kind, lr, init, 400)
            res[f"numpy_vs_torch_adam_{kind}_lr{lr}"] = d_ = {"numpy": th_np.tolist(), "torch": th_t.tolist(),
                                                               "max_abs": float(np.abs(th_np - th_t).max())}
            ok &= d_["max_abs"] < 1e-9
    for kind in ("vcs", "bce"):
        mine = R.fit_critic(d["fit"][0], d["fit"][1], d["sel"][0], d["sel"][1], kind)["selected"]
        t0 = time.time()
        n2f = W.fit_scalar_critic(d["fit"][0], d["fit"][1], d["sel"][0], d["sel"][1], kind=kind)
        sec = time.time() - t0
        sel = n2f["selected"]
        T = n2f["critic"].T
        est = W.estimate_rho(T(d["ref"][0]), T(d["ref"][1]), T(d["U"]))
        my_f = lambda s: np.tanh(R.f_np(s, mine["a_raw"], mine["b"]))
        my_rho = R.rho_moments(my_f(d["ref"][0]), my_f(d["ref"][1]), my_f(d["U"]))
        w_n2f = W.posterior_weights(T(d["U"]), est["raw"])
        w_mine = R.posterior_stable(R.f_np(d["U"], mine["a_raw"], mine["b"]), my_rho["raw"])
        r = {"same_grid_point": [sel["lr"], list(sel["init"])] == [mine["lr"], list(mine["init"])], "a": [mine["a"], sel["a"]], "b": [mine["b"], sel["b"]],
             "select_J": [mine["select_common_J"], sel["select_common_J"]], "rho": [my_rho["raw"], est["raw"]],
             "max_abs_w_diff": float(np.abs(w_n2f - w_mine).max()), "n2f_seconds": round(sec, 2)}
        r["pass"] = bool(abs(r["select_J"][0] - r["select_J"][1]) < 1e-5 and abs(r["rho"][0] - r["rho"][1]) < 1e-3 and r["max_abs_w_diff"] < 1e-3)
        ok &= r["pass"]
        res[kind] = r
    return ok, res


def tiny_sgraf():
    sys.path.insert(0, str(CLONE / "NCR"))
    import model as M
    opt = SimpleNamespace(grad_clip=2.0, img_dim=16, embed_size=32, word_dim=8, vocab_size=60, num_layers=1, bi_gru=True,
                          no_imgnorm=False, no_txtnorm=False, sim_dim=16, module_name="SGR", sgr_step=3, margin=0.2, learning_rate=2e-4)
    torch.manual_seed(0)
    return M.SGRAF(opt), opt


def t_sgraf():
    m, _ = tiny_sgraf()
    m.val_start()
    g = torch.Generator().manual_seed(1)
    images = torch.randn(7, 36, 16, generator=g)                # SGR's VisualSA BatchNorm1d expects 36 regions
    lengths = [6, 3, 5, 4, 6, 2, 5]
    caps = torch.zeros(7, 6, dtype=torch.long)
    for i, L in enumerate(lengths):
        caps[i, :L] = torch.randint(1, 60, (L,), generator=g)
    with torch.no_grad():
        e0 = m.Eiters
        l_off = m.train(images, caps, lengths, mode="eval_loss")
        e1 = m.Eiters
        l_new, diag = R.eval_loss_and_diag(m, images, caps, lengths)
        e2 = m.Eiters
        ie, ce, cl = m.forward_emb(images, caps, lengths)
        full = m.forward_sim(ie, ce, cl)
    # score_pairs: the same observed pairs in other batch arrangements (as the Q pass and eval_train batches differ)
    ds = SimpleNamespace(images=images.numpy(), captions=[caps[i, :L].tolist() for i, L in enumerate(lengths)])
    sys.path.insert(0, str(CLONE / "NCR"))
    from data import collate_fn
    sp = R.score_pairs(m, ds, np.arange(7), np.arange(7), collate_fn, batch=3)
    perm = np.array([3, 0, 6, 2, 5, 1, 4])
    sp2 = R.score_pairs(m, ds, perm, perm, collate_fn, batch=4)
    cross = R.score_pairs(m, ds, np.array([0, 1]), np.array([4, 2]), collate_fn, batch=128)
    res = {"loss_bitwise_equal": bool(torch.equal(l_off, l_new)), "eiters_increment": [e1 - e0, e2 - e1],
           "diag_equal_full_diag": bool(torch.equal(diag, full.diag())),
           "score_pairs_vs_diag_max_abs": float(np.abs(sp - diag.double().numpy()).max()),
           "score_pairs_permuted_max_abs": float(np.abs(sp2 - diag.double().numpy()[perm]).max()),
           "cross_pair_vs_matrix_max_abs": float(np.abs(cross - full.double().numpy()[[4, 2], [0, 1]]).max()),
           "scores_in_unit_interval": bool(((sp > 0) & (sp < 1)).all())}
    ok = (res["loss_bitwise_equal"] and res["eiters_increment"] == [1, 1] and res["diag_equal_full_diag"] and res["score_pairs_vs_diag_max_abs"] < 1e-5
          and res["score_pairs_permuted_max_abs"] < 1e-5 and res["cross_pair_vs_matrix_max_abs"] < 1e-5 and res["scores_in_unit_interval"])
    return ok, res


def t_hook():
    sys.path.insert(0, str(CLONE / "NCR"))
    from data import collate_fn

    class Toy:
        """forward_sim = sigmoid(6 cos(image, caption vector)); a caption's vector is its true image's feature + noise."""
        def __init__(self, feats):
            self.feats = torch.tensor(feats, dtype=torch.float32)
            self.Eiters = 0

        def val_start(self):
            pass

        def forward_emb(self, images, captions, lengths):
            return images.mean(1), self.feats[captions[:, 0]] + 0.3 * self.feats[captions[:, 1]], lengths

        def forward_sim(self, ie, ce, cl):
            return torch.sigmoid(6 * torch.nn.functional.normalize(ie, dim=1) @ torch.nn.functional.normalize(ce, dim=1).T)

    rng = np.random.default_rng(7)
    n_img = 6000
    feats = rng.normal(size=(n_img, 8)).astype(np.float32)
    true_img = np.repeat(np.arange(n_img), 5)
    img_role = np.where(rng.random(n_img) < 0.1, rng.integers(0, 4, n_img), -1)
    roles = img_role[true_img].astype(np.int8)
    anchor = roles >= 0
    t2i = true_img.copy()
    u = np.flatnonzero(~anchor)
    noisy = u[rng.random(len(u)) < 0.4]
    t2i[noisy] = rng.permutation(t2i[noisy])                       # donors from U only, as in V9
    captions = [[int(i), int(rng.integers(0, n_img))] for i in true_img]
    images = np.repeat(feats[:, None, :], 3, axis=1) + 0.05 * rng.normal(size=(n_img, 3, 8)).astype(np.float32)
    ds = SimpleNamespace(images=images, captions=captions, t2i_index=t2i, anchor_mask=anchor)
    models = [Toy(feats), Toy(feats)]
    scores = []
    for m in models:
        scores.append(R.score_pairs(m, ds, np.arange(len(t2i)), t2i, collate_fn))
    tmp = Path(tempfile.mkdtemp(prefix="r22hook_"))
    np.save(tmp / "roles.npy", roles)
    gmm = [rng.random(len(t2i)), rng.random(len(t2i))]
    R._STATE.clear()
    opt = SimpleNamespace(seed=901, r22_posterior="vcs", r22_roles_file=str(tmp / "roles.npy"), r22_log=str(tmp / "log"))
    probs = R.replace_probabilities(opt, 0, ds, models, scores, gmm, collate_fn)
    line = json.loads((tmp / "log/posterior.jsonl").read_text().splitlines()[-1])
    arr = np.load(tmp / "log/epoch_00.npz")
    is_noisy = np.zeros(len(t2i), bool)
    is_noisy[noisy] = t2i[noisy] != true_img[noisy]
    w = probs[0][u]
    from sklearn.metrics import roc_auc_score
    res = {"deployed": [line["A"]["deployed"], line["B"]["deployed"]], "rho_hat": line["A"]["rho"]["raw"], "rho_real_U": float(is_noisy[u].mean()),
           "auc_w_clean": float(roc_auc_score(~is_noisy[u], w)), "anchors_one": bool((probs[0][anchor] == 1).all()),
           "npz_keys": sorted(arr.files), "seconds": line["seconds"]}
    shutil.rmtree(tmp)
    ok = all(res["deployed"]) and abs(res["rho_hat"] - res["rho_real_U"]) < 0.05 and res["auc_w_clean"] > 0.9 and res["anchors_one"]
    return ok, res


def t_patch():
    tmp = Path(tempfile.mkdtemp(prefix="r22patch_"))
    code = tmp / "ncr"
    shutil.copytree(CLONE, code, ignore=shutil.ignore_patterns(".git", "__pycache__", "runs", "output", "noise_index", "*.png", "cc152k"))
    rcs = []
    for p in (REPO / "v9/patches/ncr_anchor_assisted.diff", REPO / "v9_n2p/ab/patches/ncr_n2p_devonly.diff", SRC / "ncr_r22_posterior.diff"):
        r = subprocess.run(["patch", "-p1", "-s", "-i", str(p)], cwd=code, capture_output=True, text=True)
        rcs.append(r.returncode)
    comp = subprocess.run([sys.executable, "-m", "py_compile", str(code / "NCR/co_train.py"), str(code / "NCR/run.py"), str(code / "NCR/r22_posterior.py")],
                          capture_output=True, text=True)
    same = (code / "NCR/r22_posterior.py").read_bytes() == (SRC / "r22_posterior.py").read_bytes()
    txt = (code / "NCR/co_train.py").read_text()
    res = {"patch_rc": rcs, "compile_rc": comp.returncode, "compile_err": comp.stderr[-500:], "posterior_module_identical": same,
           "hooks": {"eval_diag": txt.count("r22_eval_loss_and_diag(model_"), "replace": txt.count("r22_replace_probabilities(opt, epoch")}}
    shutil.rmtree(tmp)
    ok = rcs == [0, 0, 0] and comp.returncode == 0 and same and res["hooks"] == {"eval_diag": 2, "replace": 1}
    return ok, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    checks, t_all = {}, time.time()
    for name, fn in (("1_posterior_formula", t_posterior), ("2_recovery", t_recovery), ("3_fallback", t_fallback), ("4_q_pairs", t_repair),
                     ("5_n2f_crosscheck", t_n2f), ("6_official_sgraf", t_sgraf), ("7_hook", t_hook), ("8_patch_layer", t_patch)):
        t0 = time.time()
        try:
            ok, res = fn()
        except Exception:
            ok, res = False, {"exception": traceback.format_exc()}
        checks[name] = {"pass": bool(ok), "seconds": round(time.time() - t0, 1), "detail": res}
        print(f"{name}: {'PASS' if ok else 'FAIL'} ({checks[name]['seconds']} s)", flush=True)
    status = "PASS" if all(c["pass"] for c in checks.values()) else "FAIL"
    out = {"schema": "r22b-cpu-checks-v1", "status": status, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "host": os.uname().nodename, "slurm_job": os.environ.get("SLURM_JOB_ID"), "python": sys.executable, "torch": torch.__version__,
           "numpy": np.__version__, "seconds": round(time.time() - t_all, 1), "checks": checks}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    print(status)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
