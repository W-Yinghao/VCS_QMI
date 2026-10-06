"""P138 stage 1 (v7 V7-PAIR §5.3): read-only error of the sampled-shift objective against the full all-view-token objective on trained
encoders.  Nothing is trained; run directories are never written; the official test set is never read.

Per (run, checkpoint): 4 fixed base minibatches of B FIT images (seeded permutation of the run's manifest FIT uids; views from the run's own
training transform with a forked, seeded RNG — identical views for every checkpoint of a run).  ONE forward per batch with the encoder /
projector in TRAIN mode (batch-statistics BN, as in training); its graph is reused for every shift draw (BN state therefore fixed across
draws — the "frozen BN" protocol); one draw is also checked against an isolated fresh forward on a model copy.
  z space   : K in {1, 4, 16, 64, B-1}, 64 draws each (pair generator seeded per batch) -> bias / variance of L_K vs L_all, the exact
              finite-population variance (1/K)(1-K/D) s_H^2 from all D = B-1 shift terms H_d, and dL/dz errors vs the full dL/dz.
  encoder   : K in {4, 16}, 16 draws each -> gradient over all encoder + projector parameters vs g_all: MSE, cosine, relative error
              ||g_K - g_all|| / max(||g_all||, FLOOR) with FLOOR = 1e-8 fixed in advance (absolute errors and ||g_all|| also reported).
  context   : across the 4 base batches, the spread of L_all and g_all (image / augmentation sampling noise, not removed by the reference).
Losses are evaluated in float64 on z (the encoder runs in float32 as in training).
    python scripts/p138_pair_readonly.py --runs P107_AP3_views4_800ep_seed0,... --checkpoints 100,800 --out-dir <dir> [--smoke] [--cpu]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO, REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from p104_diagnostics import load  # noqa: E402
from vcs_diag.augdiag import known_box_views  # noqa: E402
from vcs_ssl.config import load_resolved  # noqa: E402
from vcs_ssl.data.cifar import load_train_partition  # noqa: E402
from vcs_ssl.data.splits import load_manifest  # noqa: E402
from vcs_ssl.data.transforms import build_two_view_transform  # noqa: E402
from vcs_ssl.objectives import sample_image_shifts, sampled_shift_all_view_loss  # noqa: E402
from vcs_ssl.utils import atomic_write_json, utc_now  # noqa: E402

OUT = Path("/home/infres/yinwang/CS_QMI/outputs")
FLOOR = 1e-8
K_Z = (1, 4, 16, 64)          # + B-1 added at run time
K_G = (4, 16)


def objective_of(cfg) -> str:
    return "js" if cfg["objective"]["loss"] == "js_matched_logistic" else "vcs"


def forward_z(enc, proj, views, device, eps):
    x = torch.cat(views, 0).to(device)
    z = F.normalize(proj(enc(x)), dim=1, eps=eps)
    return list(z.double().chunk(len(views), 0))


def params_of(enc, proj):
    return [p for m in (enc, proj) for p in m.parameters() if p.requires_grad]


def flat(gs):
    return torch.cat([g.reshape(-1) for g in gs])


def stats_vs(ref: torch.Tensor, xs: list[torch.Tensor]) -> dict:
    d = torch.stack([x - ref for x in xs])
    rn = float(ref.norm())
    err = d.norm(dim=1)
    cos = torch.stack([F.cosine_similarity(x, ref, dim=0) for x in xs])
    return {"ref_norm": rn, "mse": float(d.square().mean()), "abs_err_mean": float(err.mean()), "rel_err_mean": float(err.mean() / max(rn, FLOOR)),
            "rel_err_q90": float(torch.quantile(err, 0.9) / max(rn, FLOOR)), "cos_mean": float(cos.mean()), "cos_min": float(cos.min()),
            "mean_draw_rel_bias": float((d.mean(0)).norm() / max(rn, FLOOR))}


def one_batch(cfg, enc, proj, crit, views, device, *, seed: int, B: int, n_z: int, n_g: int, isolated_check: bool) -> dict:
    obj = objective_of(cfg); eps = cfg["model"]["normalization"]["eps"]
    enc.train(); proj.train()
    t0 = time.time()
    vz = forward_z(enc, proj, views, device, eps)
    prm = params_of(enc, proj)
    D = B - 1
    full = sampled_shift_all_view_loss(vz, crit, shifts=list(range(1, B)), objective=obj)
    L_all = full["loss"]
    gz_all = flat(torch.autograd.grad(L_all, vz, retain_graph=True))
    t1 = time.time()
    g_all = flat(torch.autograd.grad(L_all, prm, retain_graph=True))
    t_full_bwd = time.time() - t1
    with torch.no_grad():  # exact shift terms: L_{d} = L_P + H_d
        Ld = torch.stack([sampled_shift_all_view_loss(vz, crit, shifts=[d], objective=obj)["loss"] for d in range(1, B)])
        s_H2 = float(Ld.var(unbiased=True))
    gen = torch.Generator().manual_seed(seed)
    rz = {}
    for K in tuple(k for k in K_Z if k < D) + (D,):
        Ls, gzs = [], []
        tk = time.time()
        for _ in range(n_z if K < D else 1):
            sh = sample_image_shifts(B, K, gen)
            s = sampled_shift_all_view_loss(vz, crit, shifts=sh, objective=obj)
            Ls.append(float(s["loss"].detach()))
            gzs.append(flat(torch.autograd.grad(s["loss"], vz, retain_graph=True)))
        Ls_t = torch.tensor(Ls, dtype=torch.float64)
        rz[str(K)] = {"draws": len(Ls), "L_mean": float(Ls_t.mean()), "bias": float(Ls_t.mean() - L_all.detach()),
                      "bias_se": float(Ls_t.std(unbiased=True) / np.sqrt(len(Ls))) if len(Ls) > 1 else 0.0,
                      "var_empirical": float(Ls_t.var(unbiased=True)) if len(Ls) > 1 else 0.0,
                      "var_exact_formula": (1.0 / K) * (1.0 - K / D) * s_H2,
                      "score_elements": int(s["score_elements_computed"]), "score_elements_full": int(full["score_elements_computed"]),
                      "seconds_per_draw_z": (time.time() - tk) / len(Ls), "dLdz": stats_vs(gz_all, gzs)}
    rg = {}
    for K in K_G:
        gs = []; tk = time.time()
        for _ in range(n_g):
            sh = sample_image_shifts(B, K, gen)
            s = sampled_shift_all_view_loss(vz, crit, shifts=sh, objective=obj)
            gs.append(flat(torch.autograd.grad(s["loss"], prm, retain_graph=True)))
        rg[str(K)] = {"draws": n_g, "seconds_per_draw_backward": (time.time() - tk) / n_g, **stats_vs(g_all, gs)}
    iso = None
    if isolated_check:  # one K=16 draw: shared-graph gradient vs an isolated fresh train-mode forward on a model copy
        sh = sample_image_shifts(B, 16, torch.Generator().manual_seed(seed + 7))
        s = sampled_shift_all_view_loss(vz, crit, shifts=sh, objective=obj)
        g_shared = flat(torch.autograd.grad(s["loss"], prm, retain_graph=True))
        e2, p2 = copy.deepcopy(enc), copy.deepcopy(proj)
        e2.train(); p2.train()
        vz2 = forward_z(e2, p2, views, device, eps)
        s2 = sampled_shift_all_view_loss(vz2, crit, shifts=sh, objective=obj)
        g_iso = flat(torch.autograd.grad(s2["loss"], params_of(e2, p2)))
        iso = {"rel_diff": float((g_shared - g_iso).norm() / max(float(g_iso.norm()), FLOOR)), "loss_diff": float(s["loss"] - s2["loss"])}
        del e2, p2
    return {"L_all": float(L_all.detach()), "J_all": float(full["J_raw"].detach()), "s_H2": s_H2, "g_all_norm": float(g_all.norm()),
            "full_backward_seconds": t_full_bwd, "forward_and_full_seconds": time.time() - t0, "z_space": rz, "encoder_grad": rg,
            "isolated_train_mode_check": iso, "_g_all": g_all.cpu(), "_L_all": float(L_all.detach())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True); ap.add_argument("--checkpoints", default="100,800")
    ap.add_argument("--batches", type=int, default=4); ap.add_argument("--z-draws", type=int, default=64); ap.add_argument("--g-draws", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20261005); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    device = torch.device("cpu") if a.cpu or not torch.cuda.is_available() else torch.device("cuda", 0)
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    data_cache = {}
    for run in [r for r in a.runs.split(",") if r]:
        dst = od / f"P138_readonly_{run}.json"
        if dst.exists():
            print(f"[skip] {dst} exists", flush=True); continue
        rd = OUT / run
        cfg0 = load_resolved(rd / "config.resolved.yaml"); man = load_manifest(rd / "manifest.json")
        name = cfg0["data"]["name"]
        if name not in data_cache:
            data_cache[name] = load_train_partition(name, cfg0["data"]["root"])  # official TRAIN file only
        data = data_cache[name]
        V = int(cfg0["views"]["count"]); B = int(cfg0["train"]["batch_size_images"]); nb, nz, ng = a.batches, a.z_draws, a.g_draws
        if a.smoke:
            B, nb, nz, ng = 32, 1, 4, 2
        fit = np.random.default_rng(a.seed).permutation(np.asarray(man["fit_uids"], dtype=np.int64))[: B * nb]
        tf = build_two_view_transform(cfg0["views"])
        batches = [known_box_views(data.data, fit[i * B:(i + 1) * B], tf, V, seed=a.seed + 100 + i)[0] for i in range(nb)]
        ckpts = sorted(p.name for p in (rd / "checkpoints").iterdir())
        rec = {"run": run, "created_utc": utc_now(), "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
               "objective": objective_of(cfg0), "views": V, "batch": B, "base_batches": nb, "z_draws": nz, "g_draws": ng, "seed": a.seed,
               "relative_error_floor": FLOOR, "base_uids_sha256": hashlib.sha256(fit.tobytes()).hexdigest(), "smoke": a.smoke,
               "checkpoints": {}, "missing_checkpoints": []}
        for ep in [int(x) for x in a.checkpoints.split(",") if x]:
            nm = f"epoch_{ep:03d}.pt"
            if nm not in ckpts:
                rec["missing_checkpoints"].append(nm); print(f"[{run}] {nm} missing (not substituted)", flush=True); continue
            t0 = time.time()
            if device.type == "cuda":
                torch.zeros(0, device=device); torch.cuda.reset_peak_memory_stats(device)
            cfg, ck, enc, proj, crit = load(rd, nm, device)
            crit = crit.double()
            per = [one_batch(cfg, enc, proj, crit, views, device, seed=a.seed + 1000 * i + ep, B=B, n_z=nz, n_g=ng, isolated_check=(i == 0))
                   for i, views in enumerate(batches)]
            gs = torch.stack([p.pop("_g_all") for p in per]); Ls = [p.pop("_L_all") for p in per]
            gbar = gs.mean(0)
            spread = {"L_all_sd_over_batches": float(np.std(Ls, ddof=1)) if len(Ls) > 1 else 0.0,
                      "g_all_rel_dev_from_batch_mean": [float((g - gbar).norm() / max(float(gbar.norm()), FLOOR)) for g in gs],
                      "g_all_pairwise_cos": [float(F.cosine_similarity(gs[i], gs[j], dim=0)) for i in range(len(gs)) for j in range(i + 1, len(gs))]}
            rec["checkpoints"][nm] = {"epoch": ck.get("epoch"), "batches": per, "batch_spread": spread, "seconds": time.time() - t0,
                                      "peak_mem_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None}
            g16 = np.mean([p["encoder_grad"]["16"]["rel_err_mean"] for p in per]); c16 = np.mean([p["encoder_grad"]["16"]["cos_mean"] for p in per])
            print(f"[{run}] {nm}: {time.time() - t0:.0f}s  K16 grad rel.err {g16:.3f} cos {c16:.3f}  batch-spread rel.dev "
                  f"{np.mean(spread['g_all_rel_dev_from_batch_mean']):.3f}", flush=True)
            del enc, proj, crit
            if device.type == "cuda":
                torch.cuda.empty_cache()
        atomic_write_json(dst, rec)
        print(f"-> {dst}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
