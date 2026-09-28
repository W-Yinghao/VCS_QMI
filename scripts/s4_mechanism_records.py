"""S4 mechanism records (Server Spec v2 §7.2) on frozen checkpoints — read-only, minutes on a GPU.

    python scripts/s4_mechanism_records.py --runs P35_vcs_a5_views4_800ep_seed0,P41_simclr_views4_800ep_seed0 --epochs 100,400,800 \
        --n 2048 --aug own --out-dir reports/P89_mechanism [--cpu] [--smoke]
    python scripts/s4_mechanism_records.py --table reports/P89_mechanism --out reports/P89_mechanism.md

For every (run, epoch) the same base images (the first ``n`` sorted selection UIDs = images never used for SSL fitting), the same two-view
draw (one RNG seed for all runs) and the same K = 8 cyclic-shift partners are scored by the run's own frozen model (eval mode, deep copy, BN
running statistics):
  VCS (any critic):  J, R, positive / negative scores, the ACTUAL gate 1 - T^2 (equal-weight over P and Q, not 1 - J), residuals C - T,
                     saturation, cosine medians of positive and shifted pairs, the critic's (a, b) or RFF sigma;
  SimCLR:            native NT-Xent on the same two views (2B - 2 negatives): positive probability, negative mass, max negative weight, effective
                     number of negatives (exp entropy); plus the same cosine statistics on the K cyclic partners (diagnostic only);
  CS-K-native:       D_CS and its three log terms at the run's calibrated sigma;
  every method:      per-image input-gradient norms ||d loss / d x|| (both views) and projector-output gradient norms ||d loss / d p||,
                     alignment / uniformity (Wang & Isola) of z_l2 and h_l2, effective rank of h and z (selection subset).
``--aug own`` scores under the run's own augmentation block; ``--aug standard`` under the frozen standard block (crop 0.2, jitter 0.4/0.1) so
strong-aug and standard-aug models see identical inputs.  Output: one JSON per (run, epoch, aug); ``--table`` renders them.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from reference.ssl_core import cyclic_negative_indices, simclr_nt_xent, vcs_from_scores, vicreg_loss  # noqa: E402
from vcs_ssl.checkpoint import load_checkpoint  # noqa: E402
from vcs_ssl.config import load_resolved  # noqa: E402
from vcs_ssl.data.cifar import load_train_partition  # noqa: E402
from vcs_ssl.data.datasets import TwoViewNoLabelEvalDataset, make_eval_loader  # noqa: E402
from vcs_ssl.data.splits import load_manifest  # noqa: E402
from vcs_ssl.data.transforms import build_two_view_transform, two_view_transform_signature  # noqa: E402
from vcs_ssl.diagnostics import spectrum_stats  # noqa: E402
from vcs_ssl.kernel_cs import kernel_cs_pair_loss  # noqa: E402
from vcs_ssl.models import build_models  # noqa: E402
from vcs_ssl.utils import atomic_write_json, sha256_file, utc_now  # noqa: E402

OUTPUT_ROOT = Path("/home/infres/yinwang/CS_QMI/outputs")
STANDARD_VIEWS_SRC = REPO / "configs" / "cifar10_hpK_a5_views4_800ep_vcs_seed0.yaml"  # frozen standard augmentation block
VIEW_SEED, PAIR_SEED, K = 20260929, 20260931, 8


def q(x: torch.Tensor) -> dict:
    x = x.detach().float().cpu()
    qs = torch.quantile(x, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]))
    return {"mean": float(x.mean()), "q05": float(qs[0]), "q25": float(qs[1]), "median": float(qs[2]), "q75": float(qs[3]), "q95": float(qs[4])}


@torch.no_grad()
def uniformity(z: torch.Tensor, t: float = 2.0, max_n: int = 4096) -> float:
    z = z[:max_n]
    d2 = torch.cdist(z, z).pow(2)
    mask = ~torch.eye(len(z), dtype=torch.bool, device=z.device)
    return float(torch.log(torch.exp(-t * d2[mask]).mean()))


def record(run_id: str, epoch: int, *, n: int, aug: str, device: torch.device, batch: int, workers: int, smoke: bool) -> dict:
    run_dir = OUTPUT_ROOT / run_id
    cfg = load_resolved(run_dir / "config.resolved.yaml")
    man = load_manifest(run_dir / "manifest.json")
    ck_name = f"epoch_{epoch:03d}.pt"
    ck_path = run_dir / "checkpoints" / ck_name
    ck = load_checkpoint(ck_path)
    built = build_models(cfg, seed=int(cfg["run"]["seed"]), device=device)
    enc, proj, crit = built["encoder"], built["projector"], built["critic"]
    enc.load_state_dict(ck["encoder_state"]); proj.load_state_dict(ck["projector_state"])
    if crit is not None:
        crit.load_state_dict(ck["critic_state"])
    for m in (enc, proj, crit):
        if m is not None:
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
    method = cfg["run"]["method"]
    rm = json.load(open(run_dir / "run_manifest.json"))
    kernel_sigma = rm.get("kernel_sigma")
    data = load_train_partition(cfg["data"]["name"], cfg["data"]["root"])
    sel = np.sort(np.asarray(man["selection_uids"], dtype=np.int64))[:n]
    if aug == "own":
        views = cfg["views"]
    else:
        import yaml  # noqa: PLC0415
        views = dict(yaml.safe_load(STANDARD_VIEWS_SRC.read_text())["views"]); views["count"] = cfg["views"]["count"]
    tf = build_two_view_transform(views)
    loader = make_eval_loader(TwoViewNoLabelEvalDataset(data.data, sel, tf), batch_size=batch, num_workers=workers,
                              generator=torch.Generator().manual_seed(VIEW_SEED), pin_memory=device.type == "cuda")
    pg = torch.Generator().manual_seed(PAIR_SEED)
    eps = cfg["model"]["normalization"]["eps"]
    nd = bool(cfg["pairing"]["negative_detach"]) if method == "vcs_qmi" else False
    acc: dict[str, list] = {k: [] for k in ("t_pos", "t_neg", "gate_pos", "gate_neg", "res_pos", "res_neg", "cos_pos", "cos_neg", "hcos_pos", "hcos_neg",
                                            "gx1", "gx2", "gp", "z1", "h1", "hraw", "loss", "p_pos", "neg_mass", "neg_max", "neg_eff", "dcs", "log_A", "log_Bq", "log_C")}
    t0 = time.perf_counter()
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(VIEW_SEED)  # augmentation parameters when num_workers == 0
        for x1, x2, _ in loader:
            x1 = x1.to(device).requires_grad_(True); x2 = x2.to(device).requires_grad_(True)
            h = enc(torch.cat((x1, x2)))
            p = proj(h); p.retain_grad()
            z = F.normalize(p, dim=1, eps=eps); hl = F.normalize(h, dim=1, eps=eps)
            z1, z2 = z.chunk(2); h1, h2 = hl.chunk(2)
            B = len(z1)
            idx, _sh = cyclic_negative_indices(B, min(K, B - 1), generator=pg, device=z.device)
            cos_pos = (z1 * z2).sum(-1); cos_neg = (z1 * z2[idx[0]]).sum(-1)
            acc["cos_pos"].append(cos_pos.detach()); acc["cos_neg"].append(cos_neg.detach())
            acc["hcos_pos"].append((h1 * h2).sum(-1).detach()); acc["hcos_neg"].append((h1 * h2[idx[0]]).sum(-1).detach())
            if method == "vcs_qmi":
                key = {"z": z, "h_l2": hl}[cfg["model"]["critic"].get("feature_source", "z")]
                if cfg["model"]["normalization"]["vcs_and_simclr"] == "none":
                    key = p
                a1, a2 = key.chunk(2)
                t_pos = crit(a1, a2)
                left = a1.unsqueeze(0).expand(idx.shape[0], -1, -1).reshape(-1, a1.shape[1])
                right = (a2.detach() if nd else a2)[idx].reshape(-1, a2.shape[1])
                t_neg = crit(left, right)
                st = vcs_from_scores(t_pos, t_neg)
                loss = st["loss"]
                acc["t_pos"].append(t_pos.detach()); acc["t_neg"].append(t_neg.detach())
                acc["gate_pos"].append((1 - t_pos.detach() ** 2)); acc["gate_neg"].append((1 - t_neg.detach() ** 2))
                acc["res_pos"].append((1 - t_pos.detach())); acc["res_neg"].append((-1 - t_neg.detach()))
            elif method == "simclr_matched":
                tau = float(cfg["objective"]["simclr_temperature"])
                loss = simclr_nt_xent(z1, z2, temperature=tau)
                with torch.no_grad():
                    x = torch.cat((z1, z2)); logits = (x @ x.T) / tau
                    nn_ = len(x); diag = torch.eye(nn_, dtype=torch.bool, device=x.device)
                    logits = logits.masked_fill(diag, -torch.inf); prob = logits.softmax(1)
                    tgt = (torch.arange(nn_, device=x.device) + B) % nn_
                    p_pos = prob[torch.arange(nn_), tgt]
                    negw = prob.clone(); negw[torch.arange(nn_), tgt] = 0.0
                    mass = negw.sum(1); nw = negw / mass.clamp_min(1e-12).unsqueeze(1)
                    ent = -(nw.clamp_min(1e-12).log() * nw).sum(1)
                    acc["p_pos"].append(p_pos); acc["neg_mass"].append(mass); acc["neg_max"].append(negw.max(1).values); acc["neg_eff"].append(ent.exp())
            elif method == "cs_kernel_native":
                out = kernel_cs_pair_loss(z1, z2, sigma=float(kernel_sigma), chunk=int(cfg["objective"].get("kernel_cs_chunk", 0)))
                loss = out["loss"]
                acc["dcs"].append(out["kernel_cs"].detach().reshape(1)); acc["log_A"].append(out["kcs_log_A"].detach().reshape(1))
                acc["log_Bq"].append(out["kcs_log_Bq"].detach().reshape(1)); acc["log_C"].append(out["kcs_log_C"].detach().reshape(1))
            elif method == "vicreg_matched_128":
                w = cfg["objective"]["vicreg_weights"]; p1, p2 = p.chunk(2)
                loss = vicreg_loss(p1, p2, inv_weight=w["invariance"], var_weight=w["variance"], cov_weight=w["covariance"], eps=cfg["objective"]["vicreg_variance_eps"])["loss"]
            else:
                raise ValueError(method)
            loss.backward()
            acc["loss"].append(loss.detach().reshape(1))
            acc["gx1"].append(x1.grad.flatten(1).norm(dim=1)); acc["gx2"].append(x2.grad.flatten(1).norm(dim=1))
            acc["gp"].append(p.grad.norm(dim=1))
            acc["z1"].append(z1.detach()); acc["h1"].append(h1.detach()); acc["hraw"].append(h.detach().chunk(2)[0])
            if smoke:
                break
    cat = {k: torch.cat(v) for k, v in acc.items() if v}
    Zs, Hs, Hraw = cat["z1"], cat["h1"], cat["hraw"]
    out = {"run": run_id, "method": method, "epoch": epoch, "checkpoint": ck_name, "checkpoint_sha256": sha256_file(ck_path), "aug": aug,
           "views_signature": two_view_transform_signature(views), "n_images": int(len(sel)), "batch": batch, "K": int(K), "view_seed": VIEW_SEED,
           "pair_seed": PAIR_SEED, "negative_detach": nd, "critic_impl": rm.get("critic_impl"), "kernel_sigma": kernel_sigma, "smoke": smoke,
           "model_mode": "eval (BN running statistics), parameters frozen; gradients w.r.t. inputs only", "device": str(device), "utc": utc_now(),
           "loss_mean": float(cat["loss"].mean()),
           "input_grad_norm_view1": q(cat["gx1"]), "input_grad_norm_view2": q(cat["gx2"]), "projector_grad_norm": q(cat["gp"]),
           "cos_z_pos": q(cat["cos_pos"]), "cos_z_neg": q(cat["cos_neg"]), "cos_h_pos": q(cat["hcos_pos"]), "cos_h_neg": q(cat["hcos_neg"]),
           "alignment_z": float((cat["cos_pos"] * -2 + 2).mean()), "alignment_h": float((cat["hcos_pos"] * -2 + 2).mean()),
           "uniformity_z": uniformity(Zs), "uniformity_h": uniformity(Hs),
           "effective_rank_h": spectrum_stats(Hraw.cpu())["effective_rank"], "effective_rank_z": spectrum_stats(Zs.cpu())["effective_rank"],  # raw h as in the trainer's spectrum
           "seconds": time.perf_counter() - t0}
    if method == "vcs_qmi":
        tp, tn = cat["t_pos"], cat["t_neg"]
        J = float(tp.mean() - tn.mean() - 0.5 * (tp ** 2).mean() - 0.5 * (tn ** 2).mean())
        out.update({"J": J, "R_binary": 1 - J, "t_pos": q(tp), "t_neg": q(tn),
                    "gate_pos": q(cat["gate_pos"]), "gate_neg": q(cat["gate_neg"]), "gate_M": 0.5 * float(cat["gate_pos"].mean()) + 0.5 * float(cat["gate_neg"].mean()),
                    "one_minus_J": 1 - J, "residual_pos": q(cat["res_pos"]), "residual_neg": q(cat["res_neg"]),
                    "residual_sq_M": 0.5 * float((cat["res_pos"] ** 2).mean()) + 0.5 * float((cat["res_neg"] ** 2).mean()),
                    "sat_pos_frac": float((tp.abs() > 0.95).float().mean()), "sat_neg_frac": float((tn.abs() > 0.95).float().mean())})
        if crit is not None and hasattr(crit, "scale") and hasattr(crit, "bias"):
            out["critic_ab"] = {"a": float(crit.scale), "b": float(crit.bias), "threshold_cos": float(-crit.bias / crit.scale) if abs(float(crit.scale)) > 1e-8 else None}
        if crit is not None and hasattr(crit, "sigma"):
            out["rff"] = {"sigma": float(crit.sigma), "n_features": int(crit.n_features)}
    elif method == "simclr_matched":
        out.update({"nt_xent": out["loss_mean"], "p_pos": q(cat["p_pos"]), "neg_mass": q(cat["neg_mass"]), "neg_max_weight": q(cat["neg_max"]), "neg_effective_number": q(cat["neg_eff"]),
                    "temperature": float(cfg["objective"]["simclr_temperature"])})
    elif method == "cs_kernel_native":
        out.update({"D_CS": float(cat["dcs"].mean()), "log_A": float(cat["log_A"].mean()), "log_Bq": float(cat["log_Bq"].mean()), "log_C": float(cat["log_C"].mean())})
    return out


def table(in_dir: Path) -> str:
    rows = [json.load(open(p)) for p in sorted(in_dir.glob("*.json"))]
    L = [f"# S4 mechanism records — {in_dir} ({utc_now()})", "", "Frozen checkpoints, eval mode, same base images / views / shifts for every run (seeds in the JSONs).  "
         "gate_M = actual mean 1 − T² over the equal mixture (not 1 − J).  Gradients are per image, loss averaged over the batch.", "",
         "| run | ep | aug | method | J / NT-Xent / D_CS | gate_M | 1−J | |res| pos / neg | sat pos / neg | cos_z pos / neg (med) | ‖∂x‖ v1 (mean) | ‖∂p‖ (mean) | align h | unif h | erank h | SimCLR neg mass / eff. n |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        obj = r.get("J", r.get("nt_xent", r.get("D_CS")))
        L.append(f"| {r['run']} | {r['epoch']} | {r['aug']} | {r['method']} | {obj:.4f} | {r.get('gate_M', float('nan')):.4f} | {r.get('one_minus_J', float('nan')):.4f} | "
                 f"{abs(r['residual_pos']['mean']) if 'residual_pos' in r else float('nan'):.3f} / {abs(r['residual_neg']['mean']) if 'residual_neg' in r else float('nan'):.3f} | "
                 f"{r.get('sat_pos_frac', float('nan')):.3f} / {r.get('sat_neg_frac', float('nan')):.3f} | {r['cos_z_pos']['median']:.3f} / {r['cos_z_neg']['median']:.3f} | "
                 f"{r['input_grad_norm_view1']['mean']:.3e} | {r['projector_grad_norm']['mean']:.3e} | {r['alignment_h']:.3f} | {r['uniformity_h']:.3f} | {r['effective_rank_h']:.1f} | "
                 f"{r['neg_mass']['mean'] if 'neg_mass' in r else float('nan'):.3f} / {r['neg_effective_number']['mean'] if 'neg_effective_number' in r else float('nan'):.1f} |")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=None); ap.add_argument("--epochs", default="100,400,800"); ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--aug", choices=("own", "standard"), default="own"); ap.add_argument("--out-dir", default=None); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--cpu", action="store_true"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--table", default=None); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.table:
        Path(a.out).write_text(table(Path(a.table)), encoding="utf-8"); print("wrote", a.out); return 0
    device = torch.device("cpu") if a.cpu or not torch.cuda.is_available() else torch.device("cuda", 0)
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for run in a.runs.split(","):
        for ep in (int(e) for e in a.epochs.split(",")):
            dest = out_dir / f"{run}__epoch_{ep:03d}__{a.aug}.json"
            if dest.is_file() and not a.smoke:
                print("skip (exists)", dest); continue
            r = record(run, ep, n=a.n if not a.smoke else min(a.n, 128), aug=a.aug, device=device, batch=a.batch if not a.smoke else 64, workers=a.workers, smoke=a.smoke)
            atomic_write_json(dest, r)
            print(f"{run} ep{ep} {a.aug}: {r['method']} J/obj={r.get('J', r.get('nt_xent', r.get('D_CS')))} gate_M={r.get('gate_M')} erank_h={r['effective_rank_h']:.1f} ({r['seconds']:.0f}s) -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
