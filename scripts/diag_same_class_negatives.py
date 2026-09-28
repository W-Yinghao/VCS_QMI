"""D2 (package v2, Spec §8.2 / Plan §7.2) — same-class negative-pair gradient share on frozen VCS and SimCLR checkpoints.

    python scripts/diag_same_class_negatives.py --out <prefix> [--ckpts run:ckpt,...] [--n-batches 16] [--B 256] [--smoke] [--cpu]

Identical base images and candidates for both methods: the same seeded identities from the FIT role of the P83/P84 role split (labels
are read only to split the report; nothing is filtered or re-weighted), the same seeded views (the runs share the `views` block, so the
augmented tensors are bit-identical for both encoders), the same K = 8 cyclic-shift partners per (batch, view pair).  Every model is a
deep copy in eval() mode (BN buffers untouched); gradients are taken w.r.t. the anchor's representation z (L2 projector output, the critic
input) and h (pre-projector), with the critic / temperature fixed.  Per anchor i (view a of image i) and negative j:

  VCS (K = 8, partner side detached as trained):   l-_ij = (T_ij + T_ij^2/2) / K,   l+_i = -(T_ii - T_ii^2/2);  T = critic(z_i, z_j)
  SimCLR native (2B - 2 negatives, real softmax weights w_ij of the anchor's own row, temperature tau):  g_ij = w_ij z_j / tau,
        g_pos = -(1 - w_pos) z_pos / tau  (own-row decomposition; the anchor's candidate role in other rows is reported as a total/own ratio)
  SimCLR K-matched (diagnostic only, not its training loss): softmax over {positive, the same K partners}.

  F_same(i) = sum_{j: Y_j = Y_i} |g_ij| / (sum_{j != i} |g_ij| + eps)   with numerator and denominator reported; vector sums G_same, G_diff,
  G_pos and their cosines; cancellation index |G_same + G_diff| / (|G_same| + |G_diff|).  Bootstrap by image (all instances of an image move
  together).  Similarity / confidence distributions are reported before the gradients.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO))
from reference.ssl_core import cyclic_negative_indices  # noqa: E402
from vcs_estim.frozen import load_models  # noqa: E402
from vcs_estim.pairing import ROLE_SEED, role_split  # noqa: E402

DEFAULT_CKPTS = ",".join(f"{r}:{c}" for r in ("P35_vcs_a5_views4_800ep_seed0", "P41_simclr_views4_800ep_seed0") for c in ("initial", "epoch_100", "epoch_400", "epoch_800"))
OUT_ROOT = Path("/home/infres/yinwang/CS_QMI/outputs")
EPS = 1e-12
FIELDS = ("n_neg", "n_same", "count_share", "sim_same", "sim_diff", "score_same", "score_diff", "gate_same", "gate_diff", "conf_pos",
          "num_z", "den_z", "F_same_z", "num_h", "den_h", "F_same_h", "G_same_z", "G_diff_z", "G_pos_z", "cos_same_diff", "cos_same_pos", "cos_diff_pos",
          "cancel_z", "vec_share_same_z", "G_same_h", "G_diff_h", "G_pos_h", "cancel_h", "total_over_own")


# ------------------------------------------------------------------------------------------------------------ bookkeeping helpers
def same_class_share(norms: torch.Tensor, same: torch.Tensor, eps: float = EPS):
    """norms [B, M] per-pair gradient norms, same [B, M] bool.  Returns numerator, denominator, F_same per anchor."""
    num = (norms * same.float()).sum(1); den = norms.sum(1)
    return num, den, num / (den + eps)


def vector_relations(G_same, G_diff, G_pos):
    """Cosines and the cancellation index of the per-anchor gradient sums [B, D]."""
    cos = lambda a, b: (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + EPS)
    ns, nd = G_same.norm(dim=1), G_diff.norm(dim=1)
    return {"cos_same_diff": cos(G_same, G_diff), "cos_same_pos": cos(G_same, G_pos), "cos_diff_pos": cos(G_diff, G_pos),
            "cancel": (G_same + G_diff).norm(dim=1) / (ns + nd + EPS), "vec_share_same": ns / ((G_same + G_diff).norm(dim=1) + EPS)}


def bootstrap_by_image(values: dict, uid: np.ndarray, reps: int = 500, seed: int = 20260931) -> dict:
    """values: name -> array over anchor instances (may contain NaN); uid: image id per instance.  Resamples images; all instances of
    an image move together.  Returns mean and 95 % CI of the nan-mean per name, plus the mass ratios sum(num)/sum(den)."""
    u, inv = np.unique(uid, return_inverse=True); rng = np.random.default_rng(seed); out = {}
    W = np.stack([np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))[inv] for _ in range(reps)]).astype(np.float64)  # [reps, N]
    for k, v in values.items():
        v = np.asarray(v, np.float64); ok = ~np.isnan(v)
        if ok.sum() == 0:
            out[k] = {"mean": None}; continue
        m = (W[:, ok] * v[ok]).sum(1) / np.maximum(W[:, ok].sum(1), 1)
        out[k] = {"mean": float(v[ok].mean()), "ci95": [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))], "n_instances": int(ok.sum()), "n_images": int(len(u))}
    return out


def mass_ratio_boot(num, den, uid, reps=500, seed=20260931):
    num, den = np.asarray(num, np.float64), np.asarray(den, np.float64); u, inv = np.unique(uid, return_inverse=True); rng = np.random.default_rng(seed)
    r = []
    for _ in range(reps):
        w = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))[inv]; r.append((w * num).sum() / max((w * den).sum(), EPS))
    return {"ratio": float(num.sum() / max(den.sum(), EPS)), "ci95": [float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))]}


# ------------------------------------------------------------------------------------------------------------ per-batch analysis
def analyse_vcs(critic, z1, h1, z2, y, idx, K):
    """z1 = normalize(proj(h1)) with graph to h1 (leaf); z2 detached partners; idx [K, B]."""
    B = len(z1); crit = critic
    T_pos = crit(z1, z2.detach()); l_pos = -(T_pos - 0.5 * T_pos ** 2)
    gz_pos, gh_pos = torch.autograd.grad(l_pos.sum(), (z1, h1), retain_graph=True)
    gz, gh, Tn = [], [], []
    for k in range(K):
        zj = z2.detach()[idx[k]]; T = crit(z1, zj); l = (T + 0.5 * T ** 2) / K
        a, b = torch.autograd.grad(l.sum(), (z1, h1), retain_graph=True); gz.append(a); gh.append(b); Tn.append(T.detach())
    gz = torch.stack(gz, 1); gh = torch.stack(gh, 1); Tn = torch.stack(Tn, 1)            # [B, K, D]
    same = (y[idx].T == y[:, None])                                                        # [B, K]
    with torch.no_grad():
        sim = (z1.detach()[:, None, :] * z2.detach()[idx].transpose(0, 1)).sum(-1)      # [B, K] cosine
        gate = 1 - Tn ** 2; conf = (1 + T_pos.detach()) / 2
        return _pack(same, sim, Tn, gate, conf, gz, gh, gz_pos, gh_pos, total_over_own=None)


def analyse_simclr(z1, h1, z2, y, idx, K, tau, kmatched: bool):
    """Anchors = view-a rows.  native: negatives = all 2B - 2 other items (same view and other view), weights from the anchor's own row of
    the full NT-Xent denominator (positive kept in the denominator, self masked).  kmatched: softmax over {positive, K partners} (diagnostic)."""
    B, D = z1.shape
    if kmatched:
        cand = z2.detach()[idx].transpose(0, 1)                                          # [B, K, D]
        logits = torch.cat([(z1 * z2.detach()).sum(-1, keepdim=True), (z1[:, None, :] * cand).sum(-1)], 1) / tau  # [B, 1 + K]
        p = torch.softmax(logits.detach(), 1); w_pos, w = p[:, 0], p[:, 1:]
        same = (y[idx].T == y[:, None]); cand_lab = None
        sim = (z1.detach()[:, None, :] * cand).sum(-1)
        gz_pos, gh_pos = torch.autograd.grad((-(1 - w_pos) * (z1 * z2.detach()).sum(-1) / tau).sum(), (z1, h1), retain_graph=True)
        gz = w[:, :, None] * cand / tau                                                   # own-row per-pair z gradients [B, K, D]
        gh = torch.stack([torch.autograd.grad((z1 * gz[:, k].detach()).sum(), h1, retain_graph=True)[0] for k in range(K)], 1)
        total_over_own = None
    else:
        x_all = torch.cat([z1.detach(), z2.detach()])                                     # [2B, D] candidates (detached: own-row view)
        logits = (z1 @ x_all.T) / tau; logits[torch.arange(B), torch.arange(B)] = -torch.inf
        p = torch.softmax(logits.detach(), 1); pos = torch.arange(B) + B; w_pos = p[torch.arange(B), pos]
        mask = torch.ones(B, 2 * B, dtype=torch.bool, device=z1.device); mask[torch.arange(B), torch.arange(B)] = False; mask[torch.arange(B), pos] = False
        w = p.masked_fill(~mask, 0.0)                                                     # negatives' weights [B, 2B]
        lab_all = torch.cat([y, y]); same = (lab_all[None, :] == y[:, None]) & mask
        sim = (z1.detach() @ x_all.T).masked_fill(~mask, float("nan"))
        gz_pos, gh_pos = torch.autograd.grad((-(1 - w_pos) * (z1 * z2.detach()).sum(-1) / tau).sum(), (z1, h1), retain_graph=True)
        gz = w[:, :, None] * x_all[None, :, :] / tau                                      # [B, 2B, D] own-row per-pair z gradients
        gh = None                                                                          # per-pair h gradients only for the sums (below)
        # total gradient of the anchor incl. its candidate role in other rows (2B x batch NT-Xent, both views as anchors)
        xa = torch.cat([z1, z2]).detach().requires_grad_(True); lg = (xa @ xa.T) / tau; lg = lg.masked_fill(torch.eye(2 * B, dtype=torch.bool, device=xa.device), -torch.inf)
        tgt = (torch.arange(2 * B, device=xa.device) + B) % (2 * B); L = F.cross_entropy(lg, tgt, reduction="sum")
        g_tot = torch.autograd.grad(L, xa)[0][:B]
        own = gz.sum(1) + gz_pos
        total_over_own = g_tot.norm(dim=1) / (own.norm(dim=1) + EPS)
        cand_lab = lab_all
    with torch.no_grad():
        conf = w_pos; score = w if not kmatched else w; gate = w if not kmatched else w  # SimCLR: the softmax weight is both its "score" and its decay
        if kmatched:
            return _pack(same, sim, score, gate, conf, gz, gh, gz_pos, gh_pos, total_over_own)
    # native: build the sums in h space with three vjps
    same_f = same.float()[:, :, None]; G_same_z = (gz * same_f).sum(1); G_diff_z = (gz * (mask.float()[:, :, None] - same_f)).sum(1)
    G_same_h = torch.autograd.grad((z1 * G_same_z.detach()).sum(), h1, retain_graph=True)[0]
    G_diff_h = torch.autograd.grad((z1 * G_diff_z.detach()).sum(), h1, retain_graph=True)[0]
    with torch.no_grad():
        norms = gz.norm(dim=2); num, den, Fz = same_class_share(norms, same)
        n_neg = mask.sum(1).float(); n_same = same.sum(1).float()
        rel = vector_relations(G_same_z, G_diff_z, gz_pos)
        relh = vector_relations(G_same_h, G_diff_h, gh_pos)
        nanmean = lambda t, m: (t.masked_fill(~m, 0).sum(1) / m.sum(1).clamp_min(1)).masked_fill(m.sum(1) == 0, float("nan"))
        diff = mask & ~same
        return {"n_neg": n_neg, "n_same": n_same, "count_share": n_same / n_neg, "sim_same": nanmean(sim.nan_to_num(0), same), "sim_diff": nanmean(sim.nan_to_num(0), diff),
                "score_same": nanmean(w, same), "score_diff": nanmean(w, diff), "gate_same": nanmean(w, same), "gate_diff": nanmean(w, diff), "conf_pos": conf,
                "num_z": num, "den_z": den, "F_same_z": Fz, "num_h": torch.full_like(Fz, float("nan")), "den_h": torch.full_like(Fz, float("nan")), "F_same_h": torch.full_like(Fz, float("nan")),
                "G_same_z": G_same_z.norm(dim=1), "G_diff_z": G_diff_z.norm(dim=1), "G_pos_z": gz_pos.norm(dim=1), **{k: v for k, v in rel.items() if k in ("cos_same_diff", "cos_same_pos", "cos_diff_pos")},
                "cancel_z": rel["cancel"], "vec_share_same_z": rel["vec_share_same"], "G_same_h": G_same_h.norm(dim=1), "G_diff_h": G_diff_h.norm(dim=1), "G_pos_h": gh_pos.norm(dim=1),
                "cancel_h": relh["cancel"], "total_over_own": total_over_own}


def _pack(same, sim, score, gate, conf, gz, gh, gz_pos, gh_pos, total_over_own):
    """K-candidate variants: per-pair z and h gradients available."""
    with torch.no_grad():
        norms_z = gz.norm(dim=2); num_z, den_z, Fz = same_class_share(norms_z, same)
        norms_h = gh.norm(dim=2); num_h, den_h, Fh = same_class_share(norms_h, same)
        sf = same.float()[:, :, None]; df = (~same).float()[:, :, None]
        G_same_z, G_diff_z = (gz * sf).sum(1), (gz * df).sum(1); G_same_h, G_diff_h = (gh * sf).sum(1), (gh * df).sum(1)
        rel, relh = vector_relations(G_same_z, G_diff_z, gz_pos), vector_relations(G_same_h, G_diff_h, gh_pos)
        nanmean = lambda t, m: (t.masked_fill(~m, 0).sum(1) / m.sum(1).clamp_min(1)).masked_fill(m.sum(1) == 0, float("nan"))
        n_neg = torch.full((len(same),), float(same.shape[1]), device=same.device); n_same = same.sum(1).float()
        return {"n_neg": n_neg, "n_same": n_same, "count_share": n_same / n_neg, "sim_same": nanmean(sim, same), "sim_diff": nanmean(sim, ~same),
                "score_same": nanmean(score, same), "score_diff": nanmean(score, ~same), "gate_same": nanmean(gate, same), "gate_diff": nanmean(gate, ~same), "conf_pos": conf,
                "num_z": num_z, "den_z": den_z, "F_same_z": Fz, "num_h": num_h, "den_h": den_h, "F_same_h": Fh,
                "G_same_z": G_same_z.norm(dim=1), "G_diff_z": G_diff_z.norm(dim=1), "G_pos_z": gz_pos.norm(dim=1), "cos_same_diff": rel["cos_same_diff"], "cos_same_pos": rel["cos_same_pos"],
                "cos_diff_pos": rel["cos_diff_pos"], "cancel_z": rel["cancel"], "vec_share_same_z": rel["vec_share_same"], "G_same_h": G_same_h.norm(dim=1), "G_diff_h": G_diff_h.norm(dim=1),
                "G_pos_h": gh_pos.norm(dim=1), "cancel_h": relh["cancel"], "total_over_own": (total_over_own if total_over_own is not None else torch.full_like(Fz, float("nan")))}


# ------------------------------------------------------------------------------------------------------------ driver
def build_views(images, uids_batches, tf, nv, seed):
    """Seeded views per batch (same for every checkpoint): list over batches of [nv] tensors [B, 3, 32, 32]."""
    from PIL import Image
    out = []
    for bi, ub in enumerate(uids_batches):
        torch.manual_seed(seed + 1009 * bi)
        out.append([torch.stack([tf(Image.fromarray(images[int(u)])) for u in ub]) for _ in range(nv)])
    return out


def summarise(inst: dict, uid: np.ndarray) -> dict:
    vals = {k: np.concatenate([np.asarray(x, np.float64) for x in v]) for k, v in inst.items()}
    boot = bootstrap_by_image(vals, uid)
    # conditional means (anchors with >= 1 same-class negative) for the share quantities
    has = vals["n_same"] > 0
    cond = bootstrap_by_image({k: np.where(has, vals[k], np.nan) for k in ("F_same_z", "F_same_h", "cancel_z", "cancel_h", "cos_same_diff", "cos_same_pos", "vec_share_same_z")}, uid)
    out = {"per_instance_means": boot, "conditional_on_same_present": cond, "frac_anchors_with_same_class_negative": float(has.mean()),
           "mass_ratio_z": mass_ratio_boot(vals["num_z"], vals["den_z"], uid), "count_share_pooled": float(vals["n_same"].sum() / max(vals["n_neg"].sum(), 1)),
           "n_instances": int(len(uid)), "n_images": int(len(np.unique(uid)))}
    if not np.isnan(vals["num_h"]).all():
        ok = ~np.isnan(vals["num_h"]); out["mass_ratio_h"] = mass_ratio_boot(vals["num_h"][ok], vals["den_h"][ok], uid[ok])
    out["relative_weighting_same_vs_count_z"] = out["mass_ratio_z"]["ratio"] / max(out["count_share_pooled"], EPS)
    return out


def run(a) -> int:
    dev = torch.device("cuda", 0) if torch.cuda.is_available() and not a.cpu else torch.device("cpu"); t_all = time.time()
    from vcs_ssl.data.cifar import load_cifar10_train
    from vcs_ssl.data.transforms import build_two_view_transform
    ckpts = [c.split(":") for c in a.ckpts.split(",")]
    cfg0, *_ = load_models(OUT_ROOT / ckpts[0][0], ckpts[0][1], "cpu")
    manifest = json.load(open(cfg0["data"]["manifest"])); roles = role_split(manifest["fit_uids"], ROLE_SEED)
    data = load_cifar10_train(cfg0["data"]["root"]); images, labels = data.data, data.targets
    g = torch.Generator().manual_seed(a.seed); fit = np.asarray(roles["FIT"]); pick = torch.randperm(len(fit), generator=g)[: a.n_batches * a.B].numpy()
    uids_batches = [fit[pick[i * a.B:(i + 1) * a.B]] for i in range(a.n_batches)]
    nv = a.n_views or int(cfg0["views"]["count"]); tf = build_two_view_transform(cfg0["views"]); views_sig = json.dumps(cfg0["views"], sort_keys=True)
    t0 = time.time(); views = build_views(images, uids_batches, tf, nv, a.seed); view_s = time.time() - t0
    pairs = [(x, y_) for x in range(nv) for y_ in range(x + 1, nv)]
    idx_all = [[cyclic_negative_indices(a.B, a.K, generator=torch.Generator().manual_seed(a.seed * 7919 + 31 * bi + pi))[0] for pi in range(len(pairs))] for bi in range(a.n_batches)]
    R = {"settings": {**vars(a), "device": str(dev), "n_views": nv, "view_pairs": pairs, "views_block": cfg0["views"], "role_seed": ROLE_SEED, "n_base_images": int(a.n_batches * a.B)},
         "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "labels_use": "split of the report only (Spec §2.3); nothing filtered or re-weighted",
         "base_image_uids_sha256": __import__("hashlib").sha256(",".join(map(str, np.concatenate(uids_batches).tolist())).encode()).hexdigest(), "checkpoints": {}}
    for run, ck in ckpts:
        cfg, enc, proj, critic, meta = load_models(OUT_ROOT / run, ck, dev)
        if json.dumps(cfg["views"], sort_keys=True) != views_sig:
            raise RuntimeError(f"{run}: views block differs from the first checkpoint; the seeded views would not be identical")
        enc, proj = copy.deepcopy(enc).eval(), copy.deepcopy(proj).eval(); crit = copy.deepcopy(critic).eval() if critic is not None else None
        for mod in (enc, proj, crit):
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad_(False)
        eps = float(cfg["model"]["normalization"]["eps"]); method = cfg["run"]["method"]; tau = float(cfg["objective"]["simclr_temperature"])
        variants = {"vcs_K8": None} if method == "vcs_qmi" else {"simclr_native": False, "simclr_Kmatched": True}
        inst = {v: {k: [] for k in FIELDS} for v in variants}; uid_inst = {v: [] for v in variants}; per_pair = {v: {} for v in variants}; t1 = time.time()
        for bi, ub in enumerate(uids_batches):
            y = torch.as_tensor(labels[ub], device=dev)
            with torch.no_grad():
                H = [enc(v.to(dev)) for v in views[bi]]
            for pi, (va, vb) in enumerate(pairs):
                h1 = H[va].clone().requires_grad_(True); z1 = F.normalize(proj(h1), dim=1, eps=eps)
                with torch.no_grad():
                    z2 = F.normalize(proj(H[vb]), dim=1, eps=eps)
                idx = idx_all[bi][pi].to(dev)
                for vname, km in variants.items():
                    res = analyse_vcs(crit, z1, h1, z2, y, idx, a.K) if method == "vcs_qmi" else analyse_simclr(z1, h1, z2, y, idx, a.K, tau, km)
                    for k in FIELDS:
                        inst[vname][k].append(res[k].detach().cpu().numpy())
                    uid_inst[vname].append(ub)
                    pp = per_pair[vname].setdefault(f"{va}{vb}", {"F_same_z": [], "count_share": []})
                    pp["F_same_z"].append(float(np.nanmean(res["F_same_z"].cpu().numpy()))); pp["count_share"].append(float(res["count_share"].mean()))
        out = {"meta": meta, "method": method, "critic_class": type(critic).__name__ if critic is not None else None, "temperature": tau if method != "vcs_qmi" else None,
               "critic_params": ({k: float(v) for k, v in critic.named_parameters()} if critic is not None and sum(p.numel() for p in critic.parameters()) <= 4 else None),
               "seconds": time.time() - t1, "variants": {}}
        for vname in variants:
            uid = np.concatenate(uid_inst[vname]); out["variants"][vname] = summarise(inst[vname], uid)
            out["variants"][vname]["per_view_pair"] = {k: {kk: float(np.mean(vv)) for kk, vv in v.items()} for k, v in per_pair[vname].items()}
            out["variants"][vname]["definition"] = {"vcs_K8": "K = 8 cyclic partners, partner detached, l- = (T + T^2/2)/K, l+ = -(T - T^2/2); 'gate' = 1 - T^2, 'score' = T",
                                                    "simclr_native": "all 2B-2 other items, own-row softmax weights of the full NT-Xent (positive in the denominator); 'score' = 'gate' = w_ij; per-pair h gradients not formed (sums only)",
                                                    "simclr_Kmatched": "diagnostic: softmax over {positive, the same K partners}; not the training loss"}[vname]
        R["checkpoints"][f"{run}:{ck}"] = out
        s = out["variants"]; first = next(iter(s)); v = s[first]
        print(f"{run}:{ck} [{first}] F_same_z {v['per_instance_means']['F_same_z']['mean']:.4f} (count share {v['count_share_pooled']:.4f}, mass ratio {v['mass_ratio_z']['ratio']:.4f}) "
              f"cancel_z {v['per_instance_means']['cancel_z']['mean']:.3f} gate same/diff {v['per_instance_means']['gate_same']['mean']:.4f}/{v['per_instance_means']['gate_diff']['mean']:.4f} ({out['seconds']:.0f}s)", flush=True)
    R["view_seconds"] = view_s; R["wall_seconds"] = time.time() - t_all
    Path(a.out + ".json").parent.mkdir(parents=True, exist_ok=True); json.dump(R, open(a.out + ".json", "w"), indent=1, default=float)
    write_markdown(R, a); print("->", a.out + ".md"); return 0


def write_markdown(R, a):
    st = R["settings"]
    L = [f"# D2 — same-class negative-pair gradient share on frozen checkpoints — {R['utc']}", "",
         f"{st['n_base_images']} FIT-role base images ({st['n_batches']} batches x {st['B']}), {st['n_views']} seeded views, view pairs {st['view_pairs']}, K = {st['K']} cyclic partners shared by both methods; "
         "labels split the report only.  Means over anchor instances with bootstrap-by-image 95 % CI; 'mass ratio' = sum num / sum den; 'count share' = same-class fraction of the candidates; "
         "relative weighting = mass ratio / count share (1 = same-class negatives weighted like any other).", "",
         "| checkpoint | variant | same-class rate (count) | sim same / diff | score same / diff | gate same / diff | conf+ | F_same z mean [CI] | mass ratio z [CI] | rel. weighting | F_same h (K-matched) | cos(G_same,G_diff) | cos(G_diff,G_pos) | cancel z | cancel h | total/own |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, C in R["checkpoints"].items():
        for vname, V in C["variants"].items():
            m = V["per_instance_means"]; f = lambda k, p=4: (f"{m[k]['mean']:.{p}f}" if m.get(k, {}).get("mean") is not None else "—")
            ci = lambda d: (f"{d['mean']:.4f} [{d['ci95'][0]:.4f}, {d['ci95'][1]:.4f}]" if d.get("mean") is not None else "—")
            mr = V["mass_ratio_z"]
            L.append(f"| {name} | {vname} | {V['count_share_pooled']:.4f} | {f('sim_same')} / {f('sim_diff')} | {f('score_same')} / {f('score_diff')} | {f('gate_same')} / {f('gate_diff')} | {f('conf_pos')} | {ci(m['F_same_z'])} | "
                     f"{mr['ratio']:.4f} [{mr['ci95'][0]:.4f}, {mr['ci95'][1]:.4f}] | {V['relative_weighting_same_vs_count_z']:.3f} | {f('F_same_h')} | {f('cos_same_diff', 3)} | {f('cos_diff_pos', 3)} | {f('cancel_z', 3)} | {f('cancel_h', 3)} | {f('total_over_own', 3)} |")
    L.append("")
    Path(a.out + ".md").write_text("\n".join(L) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--ckpts", default=DEFAULT_CKPTS)
    ap.add_argument("--n-batches", type=int, default=16); ap.add_argument("--B", type=int, default=256); ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--n-views", type=int, default=0, help="0 = the run's views.count"); ap.add_argument("--seed", type=int, default=20260931)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.n_batches, a.B = min(a.n_batches, 1), min(a.B, 64)
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
