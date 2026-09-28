"""Estimator package v1 (§8, with §4.3 / §4.4 roles and §15 fields): estimator diagnostics on fixed SSL checkpoints.

    python -m vcs_estim.frozen --stage frozen_diagnostics --run-dir <outputs/RUN> --ckpt epoch_800 --out <json> [--smoke] [--cpu]

Per checkpoint (encoder / projector frozen, eval() on a deep copy):
  roles    FIT / TUNE / SELECT / EVAL = 27k / 6k / 6k / 6k identities of the dev45k fit split (pairing.role_split, ROLE_SEED); every pair
           stays inside its role.  Each identity gets two augmented views (the run's own training augmentation, seeded per role, so every
           checkpoint sees the same views) -> z (L2-normalised projector output, the critic input of the recipe) and h.
  pairs    P = (view1_i, view2_i).  FIT Q = (view1_i, view2_{pi(i)}) for K = 8 distinct nonzero cyclic shifts of a seeded order.
           TUNE / SELECT / EVAL Q: independent blocks (anchor a, partner b), a != b, no identity reused across blocks; the block contributes
           P = (view1_a, view2_a) and Q = (view1_a, view2_b); standard errors are over blocks.
  1. training critic (VCS runs only): J on EVAL, residuals 1 - T+ / -1 - T-, gates 1 - T^2, (T - C)(1 - T^2) with C = +1 / -1, score
     quantiles and |T| > 0.95 right / wrong end; input gradients grad_left / grad_right from a separate *train-mode diagnostic* copy on a
     reproduced training batch (B = 256 FIT identities, the run's view count and K, the run's negative detach kept as is).
  2. measurement critics refitted on the frozen z: C0 / C1 / C2 with VCS and matched JS (Adam 5e-4, 2000 updates, batch 256, SELECT every
     100), the VCS {C0, C1, C2} simplex mix on TUNE, one VCS residual (C2 structure, lambda0 = 0.5, lambda on TUNE), the JS dictionary.
  3. EVAL once: J of every fixed model, Delta J vs the training critic, fit seconds, parameter counts.
No oracle exists here: S_truth / posterior_mse are absent with a reason (never 0).  For SimCLR checkpoints only measurement critics exist.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from . import candidates as C
from .convex_mix import fit_js_mixture, fit_small_simplex, js_mixture_logq, residual_step
from .fitting import native_loss, outputs, residual_loss, train
from .objectives import j_hat, js_match_loss, js_native, score_diagnostics
from .pairing import ROLE_SEED, check_disjoint, eval_blocks, role_split, split_hash
from .run import SOURCE_REF, git_commit

DICT = ("C0", "C1", "C2")
ABSENT = {"S_truth": None, "S_truth_absent_reason": "no oracle for image pairs (dependence of augmented views is unknown)",
          "posterior_mse": None, "posterior_mse_absent_reason": "eta unknown for image pairs",
          "excess_to_bayes_risk": None, "excess_to_bayes_risk_absent_reason": "requires S and eta"}


@dataclass
class Pairs:
    xp: torch.Tensor; yp: torch.Tensor; xq: torch.Tensor; yq: torch.Tensor


# ---------------------------------------------------------------------------------------------------------------- loading
def load_cfg(run_dir: Path) -> dict:
    return yaml.safe_load((run_dir / "config.resolved.yaml").read_text())


def load_models(run_dir: Path, ckpt: str, device):
    from vcs_ssl.models import build_models
    cfg = load_cfg(run_dir)
    M = build_models(cfg, seed=int(cfg["run"]["seed"]), device="cpu")
    st = torch.load(run_dir / "checkpoints" / f"{ckpt}.pt", map_location="cpu", weights_only=False)
    M["encoder"].load_state_dict(st["encoder_state"]); M["projector"].load_state_dict(st["projector_state"])
    critic = M["critic"]
    if critic is not None:
        critic.load_state_dict(st["critic_state"])
    meta = {k: st.get(k) for k in ("run_id", "code_commit", "config_hash", "manifest_hash", "method", "seed", "stage", "completed_epoch",
                                   "optimizer_step", "K")}
    return cfg, M["encoder"].to(device), M["projector"].to(device), (critic.to(device) if critic is not None else None), meta


class _TwoViews(torch.utils.data.Dataset):
    def __init__(self, images, uids, tf):
        from PIL import Image
        self.images, self.uids, self.tf, self.Image = images, np.asarray(uids), tf, Image

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        img = self.Image.fromarray(self.images[int(self.uids[i])])
        return self.tf(img), self.tf(img)


@torch.no_grad()
def encode_role(encoder, projector, images, uids, tf, seed, device, eps, workers=6, batch=512):
    """Two augmented views per identity; eval() on deep copies; returns z1, z2 (L2) and h1, h2 (float32, CPU)."""
    from vcs_ssl.data.datasets import seed_worker
    enc, proj = copy.deepcopy(encoder).eval(), copy.deepcopy(projector).eval()
    g = torch.Generator().manual_seed(seed)
    dl = torch.utils.data.DataLoader(_TwoViews(images, uids, tf), batch_size=batch, shuffle=False, num_workers=workers,
                                     generator=g, worker_init_fn=seed_worker)
    torch.manual_seed(seed)      # the num_workers = 0 path draws augmentation from the global stream
    Z1, Z2, H1, H2 = [], [], [], []
    for v1, v2 in dl:
        h = enc(torch.cat([v1, v2]).to(device)); z = F.normalize(proj(h), dim=1, eps=eps)
        a, b = z.chunk(2); ha, hb = h.chunk(2)
        Z1.append(a.float().cpu()); Z2.append(b.float().cpu()); H1.append(ha.float().cpu()); H2.append(hb.float().cpu())
    return torch.cat(Z1), torch.cat(Z2), torch.cat(H1), torch.cat(H2)


def fit_pairs(z1, z2, K, seed) -> tuple[Pairs, dict]:
    n = len(z1); g = torch.Generator().manual_seed(seed); order = torch.randperm(n, generator=g)
    shifts = (torch.randperm(n - 1, generator=g)[:K] + 1).tolist()
    left, right = [], []
    for s in shifts:
        partner = order[(torch.arange(n) + s) % n]; left.append(z1[order]); right.append(z2[partner])
    return Pairs(z1, z2, torch.cat(left), torch.cat(right)), {"negative_construction": f"cyclic_K{K}", "shifts": shifts}


def block_pairs(z1, z2, uids, seed) -> tuple[Pairs, dict]:
    pos = {int(u): i for i, u in enumerate(uids)}; bl = eval_blocks(uids, seed)
    a = torch.tensor([pos[int(u)] for u in bl[:, 0]]); b = torch.tensor([pos[int(u)] for u in bl[:, 1]])
    return Pairs(z1[a], z2[a], z1[a], z2[b]), {"negative_construction": "independent_blocks", "n_blocks": len(bl)}


# ---------------------------------------------------------------------------------------------------------------- metrics
def block_J(tp, tn) -> dict:
    """Raw J and its standard error over blocks (P and Q of block i share the anchor, so the block sum is the unit)."""
    tp, tn = tp.double(), tn.double(); c = (tp - 0.5 * tp ** 2) + (-tn - 0.5 * tn ** 2)
    return {"J": float(c.mean()), "J_se_block": float(c.std() / math.sqrt(len(c))), "n_blocks": len(c)}


def critic_diag(tp, tn) -> dict:
    tp64, tn64 = tp.double(), tn.double(); qs = [0.01, 0.1, 0.5, 0.9, 0.99]
    gp, gn = (1 - tp64) * (1 - tp64 ** 2), (-1 - tn64) * (1 - tn64 ** 2)       # (C - T)(1 - T^2): logit gradient of J
    d = score_diagnostics(tp64.numpy(), tn64.numpy())
    d.update({"residual_pos_mean": float((1 - tp64).mean()), "residual_neg_mean": float((-1 - tn64).mean()),
              "T_minus_C_gate_pos": {"mean": float((-gp).mean()), "quantiles": np.quantile((-gp).numpy(), qs).tolist()},
              "T_minus_C_gate_neg": {"mean": float((-gn).mean()), "quantiles": np.quantile((-gn).numpy(), qs).tolist()}})
    return d


def train_mode_grad_diag(cfg, encoder, projector, critic, images, uids, device, seed=20260928) -> dict:
    """Train-mode diagnostic on a separate copy: one reproduced training batch (B images, the run's view count, K cyclic shifts,
    negative detach as trained); gradients of J w.r.t. the critic's left / right inputs, split by positive / negative pairs."""
    from reference.ssl_core import cyclic_negative_indices, vcs_from_scores
    from vcs_ssl.data.transforms import build_two_view_transform
    from PIL import Image
    enc, proj, cri = copy.deepcopy(encoder).train(), copy.deepcopy(projector).train(), copy.deepcopy(critic).train()
    B, nv, K = int(cfg["train"]["batch_size_images"]), int(cfg["views"]["count"]), int(cfg["pairing"]["k"])
    detach = bool(cfg["pairing"].get("negative_detach", False)); eps = float(cfg["model"]["normalization"]["eps"])
    g = torch.Generator().manual_seed(seed); pick = torch.randperm(len(uids), generator=g)[:B]
    tf = build_two_view_transform(cfg["views"]); torch.manual_seed(seed)
    views = [torch.stack([tf(Image.fromarray(images[int(uids[i])])) for i in pick]) for _ in range(nv)]
    h = enc(torch.cat(views).to(device)); z = F.normalize(proj(h), dim=1, eps=eps).chunk(nv)
    out = {"label": "train-mode diagnostic", "B": B, "views": nv, "K": K, "negative_detach": detach, "view_pairs": []}
    pg = torch.Generator().manual_seed(seed + 1)
    for a in range(nv):
        for b in range(a + 1, nv):
            z1, z2 = z[a].detach(), z[b].detach()
            pl, pr = z1.clone().requires_grad_(True), z2.clone().requires_grad_(True)
            idx, _ = cyclic_negative_indices(B, K, generator=pg, device=device)
            nl = z1.unsqueeze(0).expand(K, -1, -1).reshape(-1, z1.shape[1]).clone().requires_grad_(True)
            nr_src = z2[idx].reshape(-1, z2.shape[1]).clone(); nr = nr_src if detach else nr_src.requires_grad_(True)
            tp, tn = cri(pl, pr), cri(nl, nr); J = vcs_from_scores(tp, tn)["J_raw"]
            leaves = [pl, pr, nl] + ([] if detach else [nr]); grads = torch.autograd.grad(J, leaves)
            gpl, gpr, gnl = grads[0], grads[1], grads[2]; gnr = torch.zeros_like(nr_src) if detach else grads[3]
            nrm = lambda t: float(t.norm(dim=1).mean())
            # per-image left gradient = positive part + the K negative rows that share the anchor
            left_total = gpl + gnl.view(K, B, -1).sum(0)
            out["view_pairs"].append({"views": [a, b], "J": float(J.detach()), "grad_left_pos": nrm(gpl), "grad_right_pos": nrm(gpr),
                                      "grad_left_neg": nrm(gnl), "grad_right_neg": nrm(gnr), "grad_left": nrm(left_total),
                                      "grad_right": nrm(gpr), "right_neg_gradient_zero_by_detach": detach})
    for key in ("J", "grad_left_pos", "grad_right_pos", "grad_left_neg", "grad_right_neg", "grad_left", "grad_right"):
        out[f"mean_{key}"] = float(np.mean([v[key] for v in out["view_pairs"]]))
    return out


# ---------------------------------------------------------------------------------------------------------------- main stage
def run_checkpoint(a) -> dict:
    from vcs_ssl.data.cifar import load_cifar10_train
    from vcs_ssl.data.transforms import build_two_view_transform
    dev = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"; t_all = time.time()
    run_dir = Path(a.run_dir); cfg, encoder, projector, critic, meta = load_models(run_dir, a.ckpt, dev)
    eps = float(cfg["model"]["normalization"]["eps"]); tf = build_two_view_transform(cfg["views"])
    manifest = json.load(open(cfg["data"]["manifest"])); roles = role_split(manifest["fit_uids"], ROLE_SEED); check_disjoint(roles)
    if a.smoke:
        roles = {r: v[: {"FIT": 1024, "TUNE": 256, "SELECT": 256, "EVAL": 256}[r]] for r, v in roles.items()}
    images = load_cifar10_train(cfg["data"]["root"]).data
    t0 = time.time(); Z = {}
    for i, (r, ids) in enumerate(roles.items()):
        Z[r] = encode_role(encoder, projector, images, ids, tf, 20260928 + 17 * i, dev, eps, workers=a.workers)
    feat_s = time.time() - t0
    FIT, fit_info = fit_pairs(Z["FIT"][0], Z["FIT"][1], 8, 20260929)
    RP = {"FIT": FIT}; pinfo = {"FIT": fit_info}
    for i, r in enumerate(("TUNE", "SELECT", "EVAL")):
        RP[r], pinfo[r] = block_pairs(Z[r][0], Z[r][1], roles[r], 20260930 + i)
    E, TU = RP["EVAL"], RP["TUNE"]
    common = {"code_commit": git_commit(), "source_reference_commit": SOURCE_REF, "setting": "frozen", "seed": a.seed, "target_kind": "S",
              "roles_hash": split_hash(roles), "gradient_routing": "measurement critic only (encoder / projector frozen, eval mode)",
              "noise_coordinate_sd": 0.0, "noise_total_rms": 0.0, "critic_input": "z_l2 (L2-normalised projector output)",
              "n_independent_units": {r: len(v) for r, v in roles.items()}, **ABSENT}
    R = {"checkpoint": {"run_dir": str(run_dir), "ckpt": a.ckpt, **meta, "method_cfg": cfg["run"]["method"],
                        "negative_detach": cfg["pairing"].get("negative_detach"), "views": cfg["views"]["count"]},
         "roles": {"seed": ROLE_SEED, "sizes": {r: len(v) for r, v in roles.items()}, "hash": split_hash(roles), "pairing": pinfo},
         "device": dev, "feature_seconds": feat_s, "rows": [], "combos": {}}

    # 1. the training-time critic
    base_J = None
    if critic is not None:
        with torch.no_grad():
            c = copy.deepcopy(critic).eval()
            tp, tn = c(E.xp.to(dev), E.yp.to(dev)).cpu(), c(E.xq.to(dev), E.yq.to(dev)).cpu()
        base_J = block_J(tp, tn)["J"]
        R["training_critic"] = {"estimator": "training_critic", "critic_class": type(critic).__name__, **block_J(tp, tn), **critic_diag(tp, tn),
                                "trainable_parameters": sum(p.numel() for p in critic.parameters())}
        if not a.smoke or a.grad_diag:
            R["training_critic"]["train_mode_gradients"] = train_mode_grad_diag(cfg, encoder, projector, critic, images, roles["FIT"], dev)
    else:
        R["training_critic"] = None; R["training_critic_absent_reason"] = "method has no critic (measurement critics only)"

    # 2. measurement critics
    d = FIT.xp.shape[1]; fitted = {}
    for kind in ("vcs", "js"):
        for fam in DICT:
            torch.manual_seed(1000 * a.seed + {"C0": 1, "C1": 2, "C2": 3}[fam] + (0 if kind == "vcs" else 500))
            m, info = train(C.build(fam, d), native_loss(kind), FIT, RP["SELECT"], updates=a.updates, seed=a.seed, device=dev)
            t0 = time.time(); fp, fn = outputs(m, E.xp, E.yp, dev), outputs(m, E.xq, E.yq, dev); ev = time.time() - t0
            tp, tn = torch.tanh(fp), torch.tanh(fn); bj = block_J(tp, tn)
            row = {**common, "run_id": f"{run_dir.name}_{a.ckpt}_{kind}_{fam}", "status": "completed", "estimator": f"{kind}_single",
                   "critic_family": fam, "loss_scale_convention": "minus_J" if kind == "vcs" else "matched_JS",
                   "negative_construction": "FIT cyclic_K8 / EVAL independent_blocks", "trainable_parameters": C.n_params(m),
                   "selected_checkpoint": info["selected_update"], "select_risk": info["select_risk"], "select_curve": info["select_curve"],
                   "fit_seconds": info["fit_seconds"], "eval_seconds": ev, "n_positive_pairs": len(tp), "n_negative_pairs": len(tn),
                   "J_eval": bj["J"], "J_eval_se_block": bj["J_se_block"], "delta_J_vs_training_critic": (bj["J"] - base_J) if base_J is not None else None,
                   "J_fit": float(j_hat(torch.tanh(outputs(m, FIT.xp, FIT.yp, dev)), torch.tanh(outputs(m, FIT.xq, FIT.yq, dev)))),
                   "score_diagnostics": critic_diag(tp, tn)}
            if kind == "js":
                row["JS_native_eval"] = js_native(float(js_match_loss(fp, fn)))
                row["note"] = "J_eval of a JS model = the common-posterior VCS regression evaluation, not JS's native estimate"
            R["rows"].append(row); fitted[(kind, fam)] = (m, row)
    sel = {fam: fitted[("vcs", fam)][1]["select_risk"] for fam in DICT}; best = min(sel, key=sel.get)
    R["combos"]["vcs_best_single_by_select"] = {"family": best, "select_risk": sel, "J_eval": fitted[("vcs", best)][1]["J_eval"]}

    def dscores(kind, P):
        return (torch.stack([outputs(fitted[(kind, f)][0], P.xp, P.yp, dev) for f in DICT], 1),
                torch.stack([outputs(fitted[(kind, f)][0], P.xq, P.yq, dev) for f in DICT], 1))
    fTp, fTn = dscores("vcs", TU); fEp, fEn = dscores("vcs", E)
    fit = fit_small_simplex(torch.tanh(fTp).numpy(), torch.tanh(fTn).numpy()); w = torch.tensor(fit.weights)
    bj = block_J(torch.tanh(fEp) @ w, torch.tanh(fEn) @ w)
    R["combos"]["vcs_mix"] = {**common, "estimator": "vcs_mix", "critic_family": "+".join(DICT), "combination_fit_role": "TUNE",
                              "combination_weights": fit.weights.tolist(), "tune_objective": fit.objective, "kkt_gap": fit.kkt_gap,
                              "J_eval": bj["J"], "J_eval_se_block": bj["J_se_block"],
                              "delta_J_vs_training_critic": (bj["J"] - base_J) if base_J is not None else None,
                              "trainable_parameters": sum(fitted[("vcs", f)][1]["trainable_parameters"] for f in DICT)}
    gTp, gTn = dscores("js", TU); gEp, gEn = dscores("js", E)
    jf = fit_js_mixture(gTp.numpy(), gTn.numpy()); wj = np.array(jf["weights"])
    lqp = js_mixture_logq(gEp.numpy(), wj)[0]; lqn, l1qn = js_mixture_logq(gEn.numpy(), wj)
    bj = block_J(torch.tensor(2 * np.exp(lqp) - 1), torch.tensor(2 * np.exp(lqn) - 1))
    R["combos"]["js_mix"] = {**common, "estimator": "js_mix", "critic_family": "+".join(DICT), "combination_fit_role": "TUNE", **jf,
                             "JS_native_eval": math.log(2) - 0.5 * float(-lqp.mean() - l1qn.mean()), "J_eval": bj["J"], "J_eval_se_block": bj["J_se_block"],
                             "delta_J_vs_training_critic": (bj["J"] - base_J) if base_J is not None else None,
                             "note": "J_eval uses T = 2 q_w - 1: the common-posterior VCS regression evaluation"}
    base = fitted[("vcs", best)][0]
    for p in base.parameters():
        p.requires_grad_(False)
    torch.manual_seed(7919 + a.seed)
    U, info = train(C.build("C2", d), residual_loss(base, a.lam0), FIT, RP["SELECT"], updates=a.updates, seed=a.seed + 100, device=dev)
    sc = lambda m, P: (torch.tanh(outputs(m, P.xp, P.yp, dev)), torch.tanh(outputs(m, P.xq, P.yq, dev)))
    (b0p, b0n), (u0p, u0n) = sc(base, TU), sc(U, TU); st = residual_step(b0p.numpy(), b0n.numpy(), u0p.numpy(), u0n.numpy())
    (bEp, bEn), (uEp, uEn) = sc(base, E), sc(U, E); lam = st.coefficient
    bj = block_J(bEp + lam * (uEp - bEp), bEn + lam * (uEn - bEn))
    R["combos"]["vcs_residual"] = {**common, "estimator": "vcs_residual", "critic_family": f"{best} + C2 residual", "lambda0_train": a.lam0,
                                   "combination_fit_role": "TUNE", "lambda": lam, "A": st.A, "B": st.B, "predicted_tune_gain": st.predicted_gain,
                                   "zero_direction": st.zero_direction, "selected_checkpoint": info["selected_update"], "fit_seconds": info["fit_seconds"],
                                   "trainable_parameters": C.n_params(U), "base_J_eval": block_J(bEp, bEn)["J"], "U_alone_J_eval": block_J(uEp, uEn)["J"],
                                   "J_eval": bj["J"], "J_eval_se_block": bj["J_se_block"],
                                   "delta_J_vs_training_critic": (bj["J"] - base_J) if base_J is not None else None}
    R["wall_seconds"] = time.time() - t_all
    return R


def markdown(results: list[dict]) -> str:
    L = ["# Estimator package v1 §8 — frozen-checkpoint estimator diagnostics", "",
         f"EVAL = {results[0]['roles']['sizes']['EVAL'] // 2 if results else '?'} independent (anchor, partner) blocks per checkpoint; J ± block SE; ΔJ vs the run's own training critic (VCS runs only). "
         "No oracle: S_truth / posterior MSE absent by design.", "",
         "| run | ckpt | training critic J | vcs C0 | vcs C1 | vcs C2 | vcs mix (w) | vcs residual (λ) | js C0 / C1 / C2 (common-posterior J) | js mix J | train-mode |grad_left| / |grad_right| |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for R in results:
        ck = R["checkpoint"]; tc = R.get("training_critic"); rows = {(r["estimator"], r["critic_family"]): r for r in R["rows"]}
        f = lambda r: f"{r['J_eval']:.4f} ± {r['J_eval_se_block']:.4f}"
        g = tc.get("train_mode_gradients") if tc else None
        L.append(f"| {Path(ck['run_dir']).name} | {ck['ckpt']} (ep {ck.get('completed_epoch')}) | "
                 f"{(f'{tc['J']:.4f} ± {tc['J_se_block']:.4f}') if tc else '— (no critic)'} | "
                 + " | ".join(f(rows[("vcs_single", fam)]) for fam in DICT)
                 + f" | {f(R['combos']['vcs_mix'])} ({', '.join(f'{x:.2f}' for x in R['combos']['vcs_mix']['combination_weights'])})"
                 + f" | {f(R['combos']['vcs_residual'])} ({R['combos']['vcs_residual']['lambda']:.2f})"
                 + " | " + " / ".join(f"{rows[('js_single', fam)]['J_eval']:.4f}" for fam in DICT)
                 + f" | {R['combos']['js_mix']['J_eval']:.4f}"
                 + (f" | {g['mean_grad_left']:.3g} / {g['mean_grad_right']:.3g}" if g else " | —") + " |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="frozen_diagnostics", choices=["frozen_diagnostics", "aggregate"])
    ap.add_argument("--run-dir"); ap.add_argument("--ckpt"); ap.add_argument("--out", required=True)
    ap.add_argument("--inputs", nargs="*", help="aggregate: per-checkpoint JSONs")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--updates", type=int, default=2000); ap.add_argument("--lam0", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--grad-diag", action="store_true", help="smoke: also run the train-mode gradient diagnostic")
    a = ap.parse_args()
    if a.stage == "aggregate":
        res = [json.load(open(p)) for p in a.inputs]; Path(a.out).write_text(markdown(res)); print(Path(a.out).read_text()); return
    if a.smoke:
        a.updates = min(a.updates, 100)
    R = run_checkpoint(a); R["args"] = vars(a)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(R, open(a.out, "w"), indent=1, default=float)
    tc = R["training_critic"]
    print(f"{a.run_dir} {a.ckpt}: training critic J {tc['J']:.4f}" if tc else f"{a.run_dir} {a.ckpt}: no training critic")
    for r in R["rows"]:
        print(f"  {r['run_id']:<60} J_eval {r['J_eval']:.4f} ± {r['J_eval_se_block']:.4f}  fit {r['fit_seconds']:.0f}s")
    for k, v in R["combos"].items():
        print(f"  {k:<30} J_eval {v['J_eval']:.4f}")
    if tc and "train_mode_gradients" in tc:
        g = tc["train_mode_gradients"]; print(f"  train-mode diagnostic: |grad_left| {g['mean_grad_left']:.3g}  |grad_right| {g['mean_grad_right']:.3g}  (detach={g['negative_detach']})")
    print(f"  features {R['feature_seconds']:.0f}s, wall {R['wall_seconds']:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
