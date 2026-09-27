"""Second-application pre-check B, question 1: closed-form linear-class critic vs the trained tanh critic, on frozen features.

    python scripts/precheck_b1_closedform.py --runs P5_vcs_seed0:epoch_200.pt,P18_vcs_crit_cosine_seed0:epoch_200.pt,... --out <report prefix>

For each checkpoint: two train-distribution views of the 5k selection images (fixed RNG) through the frozen encoder + projector give
(z1, z2) (L2-normalised projector outputs) and (h1, h2) (L2-normalised encoder outputs).  Pairs are split 2500 fit / 2500 eval.  Negatives =
K = 8 non-zero cyclic shifts of the second view within each split (product-of-marginals samples, as in training).
Linear feature classes φ(x, y):  P1 = z1⊙z2 (D);  P2 = [z1⊙z2, |z1−z2|] (2D);  P3 = [z1⊙z2, |z1−z2|, z1, z2] (4D);  H1 = h1⊙h2 (512); each + intercept.
Closed form (plan appendix B):  d = E_P[φ] − E_Q[φ],  A_M = ½(E_P[φφᵀ] + E_Q[φφᵀ]),  w* = ½ (A_M + λI)⁻¹ d,  J(w) = wᵀd − wᵀA_M w  (no tanh; a
lower bound on S).  Also tanh(c · w*ᵀφ) with the scalar c fitted on the fit split (the frozen output form), and the checkpoint's own trained
critic on the same eval pairs.  Nothing is trained except w* (a linear solve) and c (1-D grid).  Ridge λ is relative to mean diag(A_M).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from vcs_ssl.checkpoint import load_checkpoint
from vcs_ssl.config import load_resolved
from vcs_ssl.data.cifar import load_cifar10_train
from vcs_ssl.data.datasets import TwoViewNoLabelEvalDataset, make_eval_loader
from vcs_ssl.data.splits import load_manifest
from vcs_ssl.data.transforms import build_two_view_transform
from vcs_ssl.models import build_models
from vcs_ssl.utils import atomic_write_json, utc_now

K = 8


@torch.no_grad()
def two_view_features(run_dir: Path, ckpt: str, device):
    cfg = load_resolved(run_dir / "config.resolved.yaml"); man = load_manifest(run_dir / "manifest.json")
    ck = load_checkpoint(run_dir / "checkpoints" / ckpt)
    built = build_models(cfg, seed=int(cfg["run"]["seed"]), device=device)
    enc, proj, crit = built["encoder"], built["projector"], built["critic"]
    enc.load_state_dict(ck["encoder_state"]); proj.load_state_dict(ck["projector_state"])
    if crit is not None:
        crit.load_state_dict(ck["critic_state"]); crit.eval()
    enc.eval(); proj.eval()
    data = load_cifar10_train(cfg["data"]["root"]); sel = np.asarray(man["selection_uids"])
    tf = build_two_view_transform(cfg["views"])
    loader = make_eval_loader(TwoViewNoLabelEvalDataset(data.data, sel, tf), batch_size=500, num_workers=4, generator=torch.Generator().manual_seed(20260927), pin_memory=False)
    Z1, Z2, H1, H2 = [], [], [], []
    for x1, x2, _ in loader:
        h = enc(torch.cat((x1, x2)).to(device)); z = F.normalize(proj(h), dim=1); hl = F.normalize(h, dim=1)
        a, b = z.chunk(2); c, d = hl.chunk(2); Z1.append(a.cpu()); Z2.append(b.cpu()); H1.append(c.cpu()); H2.append(d.cpu())
    return cfg, crit, torch.cat(Z1), torch.cat(Z2), torch.cat(H1), torch.cat(H2)


def shifts_idx(n, gen):
    ks = torch.randperm(n - 1, generator=gen)[:K] + 1
    return torch.stack([(torch.arange(n) + int(k)) % n for k in ks])  # [K, n]


def feats(kind, a, b):
    if kind == "P1" or kind == "H1":
        return a * b
    if kind == "P2":
        return torch.cat((a * b, (a - b).abs()), 1)
    if kind == "P3":
        return torch.cat((a * b, (a - b).abs(), a, b), 1)
    raise ValueError(kind)


def with_intercept(phi):
    return torch.cat((phi, torch.ones(len(phi), 1)), 1)


def moments(phi_pos, phi_neg):
    d = phi_pos.mean(0) - phi_neg.mean(0)
    A = 0.5 * (phi_pos.T @ phi_pos / len(phi_pos) + phi_neg.T @ phi_neg / len(phi_neg))
    return d.double(), A.double()


def J_linear(w, phi_pos, phi_neg):
    tp, tn = phi_pos.double() @ w, phi_neg.double() @ w
    return float((tp - 0.5 * tp ** 2).mean() + (-tn - 0.5 * tn ** 2).mean()), float((tp.abs() > 1).float().mean()), float((tn.abs() > 1).float().mean())


def J_tanh(c, w, phi_pos, phi_neg):
    tp, tn = torch.tanh(c * (phi_pos.double() @ w)), torch.tanh(c * (phi_neg.double() @ w))
    return float((tp - 0.5 * tp ** 2).mean() + (-tn - 0.5 * tn ** 2).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="list of run_id:checkpoint separated by , or ;")
    ap.add_argument("--output-root", default=os.environ.get("OUTPUT_ROOT", "/home/infres/yinwang/CS_QMI/outputs"))
    ap.add_argument("--out", required=True); ap.add_argument("--ridge", default="1e-6,1e-4,1e-2")
    a = ap.parse_args()
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cpu":
        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ridges = [float(v) for v in a.ridge.split(",")]
    rows, out = [], {"utc": utc_now(), "K": K, "runs": {}}
    for item in a.runs.replace(";", ",").split(","):  # ';' accepted because sbatch --export splits values on commas
        item = item.strip()
        if not item:
            continue
        run, ckpt = item.split(":"); rd = Path(a.output_root) / run
        cfg, crit, Z1, Z2, H1, H2 = two_view_features(rd, ckpt, device)
        n = len(Z1); half = n // 2; gen = torch.Generator().manual_seed(7)
        splits = {"fit": torch.arange(0, half), "eval": torch.arange(half, n)}
        idx = {s: shifts_idx(len(ix), gen) for s, ix in splits.items()}
        rec = {"checkpoint": ckpt, "critic": cfg["model"]["critic"]["input"] if cfg["model"]["critic"]["enabled"] else None, "n_pairs": n, "classes": {}}
        # neural critic on the eval pairs
        if crit is not None:
            with torch.no_grad():
                ix = splits["eval"]; z1, z2 = Z1[ix], Z2[ix]
                tp = crit(z1, z2); tn = crit(z1.repeat(K, 1), z2[idx["eval"].reshape(-1)])
                rec["neural_J_eval"] = float((tp - 0.5 * tp ** 2).mean() + (-tn - 0.5 * tn ** 2).mean()); rec["neural_sat_pos"] = float((tp.abs() > 0.95).float().mean())
        for kind, (A_, B_) in (("P1", (Z1, Z2)), ("P2", (Z1, Z2)), ("P3", (Z1, Z2)), ("H1", (H1, H2))):
            phi = {}
            for s, ix in splits.items():
                a1, b1 = A_[ix], B_[ix]
                phi[s] = (with_intercept(feats(kind, a1, b1)), with_intercept(feats(kind, a1.repeat(K, 1), b1[idx[s].reshape(-1)])))
            d, A = moments(*phi["fit"]); scale = float(torch.diag(A).mean())
            rec["classes"][kind] = {"dim": int(phi["fit"][0].shape[1]), "ridge": {}}
            for lam in ridges:
                w = 0.5 * torch.linalg.solve(A + lam * scale * torch.eye(len(A), dtype=torch.float64), d)
                Jfit, satp, satn = J_linear(w, *phi["fit"]); Jev, satpe, satne = J_linear(w, *phi["eval"])
                cs = torch.logspace(-1, 1.5, 40); Jc = [J_tanh(float(c), w, *phi["fit"]) for c in cs]; c_best = float(cs[int(np.argmax(Jc))])
                Jtanh_ev = J_tanh(c_best, w, *phi["eval"])
                rec["classes"][kind]["ridge"][f"{lam:g}"] = {"J_fit_closed": Jfit, "J_eval_closed": Jev, "frac_|T|>1_eval_pos": satpe, "frac_|T|>1_eval_neg": satne,
                                                           "c_tanh": c_best, "J_eval_tanh": Jtanh_ev, "J_star_formula_fit": float(0.25 * d @ torch.linalg.solve(A + lam * scale * torch.eye(len(A), dtype=torch.float64), d))}
                rows.append((run, rec["critic"], kind, rec["classes"][kind]["dim"], lam, Jfit, Jev, Jtanh_ev, c_best, rec.get("neural_J_eval")))
            print(f"[{run}] {kind}: " + " ".join(f"λ={l:g}: J_eval {rec['classes'][kind]['ridge'][f'{l:g}']['J_eval_closed']:.4f}/tanh {rec['classes'][kind]['ridge'][f'{l:g}']['J_eval_tanh']:.4f}" for l in ridges) + f" | neural {rec.get('neural_J_eval')}", flush=True)
        out["runs"][run] = rec
    atomic_write_json(Path(a.out + ".json"), out)
    L = [f"# Pre-check B1 — closed-form linear-class critic vs trained tanh critic on frozen features — {utc_now()}", "",
         f"5 000 selection images, two train-distribution views (fixed RNG), 2 500 fit / 2 500 eval pairs, K = {K} cyclic-shift negatives per pair in each split. "
         "J values are on the eval split; closed-form T = w*ᵀφ has no tanh (|T| > 1 fraction reported in the JSON); tanh(c·w*ᵀφ) fits only the scalar c on the fit split.", "",
         "| run | trained critic | class | dim | ridge λ | J_fit closed | J_eval closed | J_eval tanh(c·w*ᵀφ) | c | neural J_eval (same pairs) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]:g} | {r[5]:.4f} | {r[6]:.4f} | {r[7]:.4f} | {r[8]:.2f} | {('%.4f' % r[9]) if r[9] is not None else '—'} |")
    Path(a.out + ".md").write_text("\n".join(L) + "\n"); print("->", a.out + ".md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
