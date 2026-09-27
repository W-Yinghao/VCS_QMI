"""Second-application pre-check B-S1 (wave 2): closed-form linear-class critic vs the trained cosine critic on CROSS-MODAL frozen features.

    python scripts/precheck_b1_clip.py --features /home/infres/yinwang/CS_QMI/outputs/P49_precheck_A --out <prefix> [--smoke]

Re-trains the P50 setting-2 selected VCS adapters (topic pairing; lr 1e-3, 15 epochs, seeds 0/1/2; identity-initialised 512→512 adapters on
frozen CLIP ViT-B/32 features, `precheck_a_adapters.train`) and compares, on the adapter outputs u (image) and v (caption), the trained critic
T = tanh(a·cos(u, v) + b) with the B1 closed-form linear-class critic (P47 protocol on pairs):
  classes  P1 = u⊙v (512)   P2 = [u⊙v, |u−v|] (1024)   P3 = [u⊙v, |u−v|, u, v] (2048), each + intercept;
  d = E_P[φ] − E_Q[φ], A_M = ½(E_P[φφᵀ] + E_Q[φφᵀ]), w* = ½(A_M + λI)⁻¹d; J(w) = wᵀd − wᵀA_M w (no tanh) and tanh(c·w*ᵀφ) with the scalar c;
  λ ∈ {1e-6, 1e-4, 1e-2} × mean diag(A_M) and c are chosen on a validation part (20 % of the fit pairs), never on the eval pairs.
Pairs: positives (u_i, v of the fixed topic partner's caption 4), K = 8 negatives per image from random other images' caption 4 (the same
pairs for the neural critic and every closed form).  Two readings per split (SRC-EVAL, TGT-EVAL):
  in-domain  — the split's pairs are halved (fit / eval, fixed permutation): the closed form is fitted on the fit half (with its VAL part) and
               evaluated on the eval half, next to the neural J on the same eval half (this mirrors P47 exactly);
  transfer   — the closed form fitted on SRC-CAL pairs (never used to train the adapters) evaluated on the whole SRC-EVAL / TGT-EVAL pair sets.
Nothing is trained beyond the adapters (the P50 configuration), the linear solve and the scalar c.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from precheck_a_adapters import load_split, topic_partners, train  # noqa: E402

K = 8
CLASSES = ("P1", "P2", "P3")


def feats(kind, u, v):
    if kind == "P1":
        return u * v
    if kind == "P2":
        return torch.cat((u * v, (u - v).abs()), 1)
    return torch.cat((u * v, (u - v).abs(), u, v), 1)


def with_intercept(phi):
    return torch.cat((phi, torch.ones(len(phi), 1, dtype=phi.dtype)), 1)


def moments(pp, pq):
    d = pp.mean(0) - pq.mean(0); A = 0.5 * (pp.T @ pp / len(pp) + pq.T @ pq / len(pq))
    return d, A


def J_of(tp, tn):
    return float((tp - 0.5 * tp ** 2).mean() + (-tn - 0.5 * tn ** 2).mean())


def closed_form(pp_fit, pq_fit, pp_val, pq_val, ridges):
    """Fit w* on the fit pairs for every ridge; choose (λ, c) on the validation pairs by the tanh-wrapped J; return the chosen (w, λ, c) and the grid."""
    d, A = moments(pp_fit, pq_fit); scale = float(torch.diag(A).mean()); grid = {}; best = None
    cs = torch.logspace(-1, 1.5, 40, dtype=torch.float64)
    for lam in ridges:
        w = 0.5 * torch.linalg.solve(A + lam * scale * torch.eye(len(A), dtype=torch.float64), d)
        tp, tn = pp_val @ w, pq_val @ w
        Jc = [J_of(torch.tanh(c * tp), torch.tanh(c * tn)) for c in cs]; i = int(np.argmax(Jc)); c_best = float(cs[i])
        grid[f"{lam:g}"] = {"J_val_closed": J_of(tp, tn), "J_val_tanh": Jc[i], "c": c_best}
        if best is None or Jc[i] > best[3]:
            best = (w, lam, c_best, Jc[i])
    return best[0], best[1], best[2], grid


def evaluate(w, c, pp, pq):
    tp, tn = pp @ w, pq @ w
    return {"J_closed": J_of(tp, tn), "J_tanh": J_of(torch.tanh(c * tp), torch.tanh(c * tn)), "frac_|T|>1_pos": float((tp.abs() > 1).float().mean()), "frac_|T|>1_neg": float((tn.abs() > 1).float().mean())}


@torch.no_grad()
def pair_tensors(m, img, txt, fixed, device, seed):
    """u [n,d], v_pos [n,d], v_neg [n,K,d] on caption 4 for the images with a topic partner; pool negatives with a fixed generator (as heldout_J)."""
    keep = torch.where(fixed >= 0)[0]; src = fixed.clamp(min=0); g = torch.Generator().manual_seed(seed)
    u, v = m.enc(img.to(device), txt[:, 4].to(device)); pool = torch.randint(0, len(img), (len(keep), K), generator=g)
    v_neg = F.normalize(m.txt(txt[pool, 4].to(device)), dim=-1)
    return u[keep].cpu(), v[src[keep]].cpu(), v_neg.cpu()


def neural_T(m, u, v):
    return torch.tanh(float(m.a.detach()) * (u * v).sum(-1) + float(m.b.detach()))


def phis(kind, u, vp, vn):
    """φ for the positive pairs [n, D+1] and the K negatives [nK, D+1], double."""
    n = len(u); pp = with_intercept(feats(kind, u, vp)).double(); pq = with_intercept(feats(kind, u.repeat_interleave(K, 0), vn.reshape(n * K, -1))).double()
    return pp, pq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--ridge", default="1e-6,1e-4,1e-2"); ap.add_argument("--index-dir", default="/home/infres/yinwang/CS_QMI/data/coco_index"); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    fd = Path(a.features); raw = {k: load_split(fd, k) for k in ("SRC-FIT", "SRC-CAL", "SRC-EVAL", "TGT-EVAL")}
    seeds = [int(s) for s in a.seeds.split(",")]; ridges = [float(v) for v in a.ridge.split(",")]
    if a.smoke:
        seeds, a.epochs = [0], 1; raw = {k: (v[0][:2000], v[1][:2000], v[2][:2000]) for k, v in raw.items()}
    partners = {k: topic_partners(v[2], a.index_dir, 20260927) for k, v in raw.items()}
    fit_img, fit_txt, _ = raw["SRC-FIT"]; cal_img, cal_txt, _ = raw["SRC-CAL"]
    results = {"settings": vars(a), "device": str(device), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seeds": {}}
    rows_in, rows_tr = [], []; t0 = time.time()
    for seed in seeds:
        m, val = train("vcs", fit_img, fit_txt, cal_img, cal_txt, a.lr, a.epochs, seed, device, fit_partners=partners["SRC-FIT"][0], cal_partners=partners["SRC-CAL"][0])
        rec = {"cal_loss": val, "a": float(m.a.detach()), "b": float(m.b.detach()), "in_domain": {}, "transfer": {}}
        P = {k: pair_tensors(m, raw[k][0], raw[k][1], partners[k][1], device, 17 + i) for i, k in enumerate(("SRC-CAL", "SRC-EVAL", "TGT-EVAL"))}
        # ---- in-domain (P47 mirror): halve the split's pairs; fit (80 % fit / 20 % val) on the first half, evaluate on the second
        for split in ("SRC-EVAL", "TGT-EVAL"):
            u, vp, vn = P[split]; n = len(u); perm = torch.randperm(n, generator=torch.Generator().manual_seed(7)); half = n // 2
            fit_ix, ev_ix = perm[:half], perm[half:]; nv = max(8, int(0.2 * half)); val_ix, fit2_ix = fit_ix[:nv], fit_ix[nv:]
            tp, tn = neural_T(m, u[ev_ix], vp[ev_ix]), neural_T(m, u[ev_ix].repeat_interleave(K, 0), vn[ev_ix].reshape(-1, vn.shape[-1]))
            r = {"n_pairs_eval": int(len(ev_ix)), "neural_J_eval": J_of(tp, tn), "neural_sat_pos": float((tp.abs() > 0.95).float().mean()), "classes": {}}
            for kind in CLASSES:
                pf, qf = phis(kind, u[fit2_ix], vp[fit2_ix], vn[fit2_ix]); pv, qv = phis(kind, u[val_ix], vp[val_ix], vn[val_ix]); pe, qe = phis(kind, u[ev_ix], vp[ev_ix], vn[ev_ix])
                w, lam, c, grid = closed_form(pf, qf, pv, qv, ridges); ev = evaluate(w, c, pe, qe)
                r["classes"][kind] = {"dim": int(pf.shape[1]), "ridge_chosen": lam, "c_chosen": c, "grid": grid, **ev}
                rows_in.append((seed, split, kind, int(pf.shape[1]), lam, c, ev["J_closed"], ev["J_tanh"], r["neural_J_eval"]))
            r["best_tanh_closed"] = max(v["J_tanh"] for v in r["classes"].values()); r["gap"] = r["neural_J_eval"] - r["best_tanh_closed"]
            rec["in_domain"][split] = r
            print(f"[seed {seed}] in-domain {split}: neural {r['neural_J_eval']:.4f} | " + " ".join(f"{k}: closed {v['J_closed']:.4f} tanh {v['J_tanh']:.4f}" for k, v in r["classes"].items()) + f" | gap {r['gap']:+.4f} ({time.time() - t0:.0f}s)", flush=True)
        # ---- transfer: fit on SRC-CAL pairs (80/20), evaluate on the whole SRC-EVAL / TGT-EVAL pair sets
        u, vp, vn = P["SRC-CAL"]; n = len(u); perm = torch.randperm(n, generator=torch.Generator().manual_seed(11)); nv = max(8, int(0.2 * n)); val_ix, fit_ix = perm[:nv], perm[nv:]
        for kind in CLASSES:
            pf, qf = phis(kind, u[fit_ix], vp[fit_ix], vn[fit_ix]); pv, qv = phis(kind, u[val_ix], vp[val_ix], vn[val_ix]); w, lam, c, grid = closed_form(pf, qf, pv, qv, ridges)
            rec["transfer"][kind] = {"ridge_chosen": lam, "c_chosen": c, "grid": grid}
            for split in ("SRC-EVAL", "TGT-EVAL"):
                ue, vpe, vne = P[split]; pe, qe = phis(kind, ue, vpe, vne); ev = evaluate(w, c, pe, qe)
                tp, tn = neural_T(m, ue, vpe), neural_T(m, ue.repeat_interleave(K, 0), vne.reshape(-1, vne.shape[-1])); nj = J_of(tp, tn)
                rec["transfer"][kind][split] = {**ev, "neural_J": nj, "n_pairs": int(len(ue))}
                rows_tr.append((seed, kind, split, lam, c, ev["J_closed"], ev["J_tanh"], nj))
        print(f"[seed {seed}] transfer: " + " ".join(f"{k}/{sp}: tanh {rec['transfer'][k][sp]['J_tanh']:.4f} vs neural {rec['transfer'][k][sp]['neural_J']:.4f}" for k in CLASSES for sp in ("SRC-EVAL", "TGT-EVAL")), flush=True)
        results["seeds"][str(seed)] = rec
    json.dump(results, open(a.out + ".json", "w"), indent=1, default=float)
    L = [f"# Pre-check B-S1 — closed-form linear-class critic vs trained cosine critic on cross-modal CLIP adapter features (topic pairing) — {results['utc']}", "",
         f"Adapters: P50 setting-2 selection (lr {a.lr:g}, {a.epochs} ep), seeds {seeds}; K = {K} pool negatives per image; classes + intercept; λ and c chosen on a validation part of the fit pairs.", "",
         "## In-domain (P47 mirror: fit half / eval half of each split's pairs)", "",
         "| seed | split | class | dim | λ | c | J_eval closed (no tanh) | J_eval tanh(c·w*ᵀφ) | neural J_eval (same pairs) | gap (neural − tanh-closed) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows_in:
        L.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]:g} | {r[5]:.2f} | {r[6]:.4f} | {r[7]:.4f} | {r[8]:.4f} | {r[8] - r[7]:+.4f} |")
    L += ["", "| split | mean over seeds: neural J | best tanh-closed J (class) | gap |", "|---|---|---|---|"]
    for split in ("SRC-EVAL", "TGT-EVAL"):
        recs = [results["seeds"][str(s)]["in_domain"][split] for s in seeds]
        nj = np.mean([r["neural_J_eval"] for r in recs]); bt = np.mean([r["best_tanh_closed"] for r in recs])
        L.append(f"| {split} | {nj:.4f} | {bt:.4f} | {nj - bt:+.4f} |")
    L += ["", "## Transfer (closed form fitted on SRC-CAL pairs, evaluated on the whole eval sets)", "", "| seed | class | split | λ | c | J closed | J tanh | neural J |", "|---|---|---|---|---|---|---|---|"]
    for r in rows_tr:
        L.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:g} | {r[4]:.2f} | {r[5]:.4f} | {r[6]:.4f} | {r[7]:.4f} |")
    Path(a.out + ".md").write_text("\n".join(L) + "\n"); print("->", a.out + ".md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
