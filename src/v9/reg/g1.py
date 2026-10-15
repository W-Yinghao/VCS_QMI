"""V9-G1 — limited REG extension on BF/SHG (package v9 docs/06): r6 REG-P1 training path × two new axes, registered with r6 REG-P2's frozen
rigid machinery (both copied verbatim into this package: r6_reg_pilot.py = REG-P1, r6_reg_p2.py = REG-P2; nothing in them is modified).

Axes (per unit: method × fit_fraction × train_mismatch × seed):
  fit_fraction   {0.25, 1.0} of REG-P1's 32 FIT cores, sampled by physical specimen (TMA core = image pair; strata = TMA slide, the id prefix
                 "1B" / "2B") with the largest-remainder rule, seeded by the unit seed; patches are never the sampling unit.  Realised counts are
                 written into train.json.
  train_mismatch {0, 0.2}: exactly round(ε n) of the positive correspondences of every training step get a displaced moving-side position drawn
                 from the SAME training image (uniform over the admissible positions of the same crop), rejecting positions closer than 2 patch
                 radii to the true one.  Patch radius per method: critics 15 px (descriptor receptive-field half-width), kernel 4 px (9 × 9
                 patch), CoMIR 64 px (half of its 128-px training crop; the displaced crop never overlaps the true one).  This Q is conditional
                 on the image and crop — it is not an unconditional pixel product (package docs/06 §3).  The same displaced correspondences
                 are given to every critic objective (one data RNG stream per seed, consumed identically by vcs / logistic / infonce); CoMIR
                 draws its own crop-level mismatches from the same seed stream (its sampling unit is the crop, disclosed).  The trainer sees
                 only the observed pairs — no mismatch label enters any loss or model; only the realised fraction is logged.
Registration: REG-P2's rigid family, perturbations, support, optimiser, CAL selection (lexicographic over the 12 checkpoints) and DEV
evaluation, called unchanged through a per-unit root that mirrors REG-P1's layout with symlinks (cache / manifest / perturbations read-only).
MI / NMI have no training and no FIT dependence: their REG-P2 DEV / CAL results are reused after a hash check (perturbations + cache manifest).
Reuse of r6 units (package: "only if all hashes match"): a unit is reusable only when seed, fit set (with the data hashes), ε = 0, fraction 1,
steps and recipe constants are identical to the r6 run — the training function below consumes the data RNG exactly as REG-P1's when ε = 0.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r6_reg_pilot as P  # noqa: E402  (REG-P1, verbatim copy; read-only)
import r6_reg_p2 as Q  # noqa: E402  (REG-P2, verbatim copy; read-only)

P1_ROOT = Path("/home/infres/yinwang/CS_QMI/outputs/R6_REG")                      # REG-P1 cache / manifest / runs (read-only)
P2_ROOT = Path("/home/infres/yinwang/CS_QMI/outputs/R6_REG_P2")                   # REG-P2 perturbations / results (read-only)
V9_ROOT = Path(os.environ.get("V9_REG_ROOT", "/projects/EEG-foundation-model/yinghao/outputs_v9/reg"))
METHODS = ("vcs", "logistic", "infonce", "comir")                                   # trainable (4); kernel = fitted reference row (seed 901 only)
FRACTIONS, EPSILONS, SEEDS = (0.25, 1.0), (0.0, 0.2), (901, 902, 903)
PATCH_RADIUS = {"critic": float(P.RF_HALF), "kernel": 4.0, "comir": 64.0}          # px at 512; exclusion radius = 2 × patch radius
R6_SEEDS = (0, 1, 2)


# ------------------------------------------------------------------------------------------------------------ units
def unit_id(method, frac, eps, seed):
    return f"V9-G1-{method}-frac{frac:g}-eps{eps:g}-s{seed}"


def all_units():
    return [(unit_id(m, f, e, s), m, f, e, s) for m in METHODS for f in FRACTIONS for e in EPSILONS for s in SEEDS]


def kernel_units():
    return [(unit_id("kernel", f, e, 901), "kernel", f, e, 901) for f in FRACTIONS for e in EPSILONS]


def specimen_stratum(core: str) -> str:
    return core.split("_")[0]                                                       # TMA slide ("1B" / "2B"); the core is the physical specimen


def sample_fit(fit_cores, frac, seed):
    """Stratified (by slide) random sample of round(frac × n) cores, largest-remainder allocation (ties: stratum name order), seeded."""
    fit = sorted(fit_cores); n = int(round(frac * len(fit)))
    strata = sorted(set(specimen_stratum(c) for c in fit)); members = {s: [c for c in fit if specimen_stratum(c) == s] for s in strata}
    quota = {s: frac * len(members[s]) for s in strata}; alloc = {s: int(math.floor(quota[s])) for s in strata}
    for s in sorted(strata, key=lambda s: (-(quota[s] - alloc[s]), s))[: n - sum(alloc.values())]:
        alloc[s] += 1
    g = np.random.default_rng(int(hashlib.sha256(f"V9-G1|fit|{seed}|{frac:g}".encode()).hexdigest()[:16], 16)); out = []
    for s in strata:
        out += [members[s][i] for i in sorted(g.choice(len(members[s]), alloc[s], replace=False))]
    return sorted(out), {s: alloc[s] for s in strata}


def displaced(g, true_yx, lo, hi, radius, max_tries=1000):
    """One admissible position of the same image / crop (uniform on [lo, hi]²) at L2 distance >= radius from the true one; rejection sampling."""
    for _ in range(max_tries):
        y, x = g.uniform(lo, hi), g.uniform(lo, hi)
        if math.hypot(y - true_yx[0], x - true_yx[1]) >= radius:
            return (y, x)
    raise RuntimeError("no admissible displaced position")


def mismatch_positions(g, yx, lo, hi, eps, radius):
    """yx [n, 2] true moving-side positions -> (yx_obs [n, 2], n_mis): exactly round(eps n) rows displaced (first round(eps n) of a seeded permutation)."""
    yx = np.array(yx, dtype=np.float64); n = len(yx); n_mis = int(round(eps * n))
    if n_mis == 0:
        return yx, 0
    idx = g.permutation(n)[:n_mis]
    for i in idx:
        yx[i] = displaced(g, yx[i], lo, hi, radius)
    return yx, n_mis


def code_hash():
    return hashlib.sha256((inspect.getsource(cmd_train) + inspect.getsource(cmd_comir) + inspect.getsource(fit_kernel_g1)).encode()).hexdigest()[:16]


def state_hash(manifest, fit, seed, frac, eps, method, steps):
    h = hashlib.sha256()
    for c in sorted(fit):
        h.update(f"{c}|{manifest['files'][c]['sha256_bf']}|{manifest['files'][c]['sha256_shg']}|".encode())
    h.update(f"seed={seed}|frac={frac:g}|eps={eps:g}|method={method}|steps={steps}|size={manifest['size']}|down={manifest['down']}".encode())
    return h.hexdigest()[:16]


def r6_reusable(manifest, fit_full, method, frac, eps, seed):
    """Package rule: reuse an r6 full-data clean unit only if everything matches (seed, fit set + data hashes, ε 0, fraction 1, recipe)."""
    if method not in ("vcs", "logistic", "infonce", "comir") or frac != 1.0 or eps != 0.0 or seed not in R6_SEEDS:
        return None
    run = P1_ROOT / "runs" / f"{method}_s{seed}"; meta = run / ("comir.json" if method == "comir" else "train.json")
    if not meta.exists():
        return None
    j = json.load(open(meta))
    if sorted(j.get("fit", [])) != sorted(fit_full) or (method != "comir" and j.get("steps") != 3000) or (method == "comir" and j.get("epochs") != 23):
        return None
    return {"run": str(run), "state_hash": state_hash(manifest, fit_full, seed, 1.0, 0.0, method, 3000 if method != "comir" else 23)}


# ------------------------------------------------------------------------------------------------------------ unit roots (REG-P1 layout, symlinked read-only)
def unit_root(uid):
    return V9_ROOT / "units" / uid


def ensure_layout(uid, method, seed, p1_root=P1_ROOT, p2_root=P2_ROOT):
    """<unit>/p1/{cache -> REG-P1 cache, manifest.json -> REG-P1 manifest, runs/<method>_s<seed> -> <unit>/run}; <unit>/p2/perturbations.json -> REG-P2's."""
    u = unit_root(uid); p1, p2 = u / "p1", u / "p2"; (p1 / "runs").mkdir(parents=True, exist_ok=True); p2.mkdir(parents=True, exist_ok=True); (u / "run").mkdir(exist_ok=True)
    for link, target in ((p1 / "cache", p1_root / "cache"), (p1 / "manifest.json", p1_root / "manifest.json"), (p2 / "perturbations.json", p2_root / "perturbations.json"),
                         (p1 / "runs" / f"{method}_s{seed}", u / "run")):
        if not link.is_symlink() and not link.exists():
            link.symlink_to(target)
    return u


# ------------------------------------------------------------------------------------------------------------ training (REG-P1's cmd_train with the two axes)
def cmd_train(uid, objective, frac, eps, seed, steps=3000, smoke=False, p1_root=P1_ROOT):
    torch, nn, F = P.torch_mods(); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); torch.manual_seed(seed); g = np.random.default_rng(1000 + seed)
    m = P.load_manifest(p1_root, smoke); fit_full = sorted(c for c, r in m["roles"].items() if r == "FIT" and (p1_root / "cache" / f"{c}.npz").exists())
    fit, counts = sample_fit(fit_full, frac, seed); imgs = {c: P.learned_inputs(torch, p1_root, c, dev) for c in fit}
    eU, eV, cr = (x.to(dev) for x in P.build_models(nn)); params = list(eU.parameters()) + list(eV.parameters()) + list(cr.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4); steps = 200 if smoke else steps; out = unit_root(uid) / "run"; out.mkdir(parents=True, exist_ok=True)
    crop, inner, n_sc, n_pos = 160, P.RF_HALF + 1, min(16, len(fit)), 128; radius = 2 * PATCH_RADIUS["critic"]; t0 = time.time(); log = []; mis_total = 0
    for step in range(1, steps + 1):
        bu, bv = [], []
        for c in g.choice(fit, n_sc, replace=len(fit) < n_sc):
            U, V = imgs[c]; y0, x0 = g.integers(0, P.SIZE - crop, 2); k = int(g.integers(4)); fl = bool(g.integers(2))
            u, v = U[:, y0:y0 + crop, x0:x0 + crop], V[:, y0:y0 + crop, x0:x0 + crop]
            u, v = torch.rot90(u, k, (1, 2)), torch.rot90(v, k, (1, 2))
            if fl:
                u, v = u.flip(2), v.flip(2)
            bu.append(u); bv.append(v)
        DU, DV = eU(torch.stack(bu)), eV(torch.stack(bv))
        yx_np = g.uniform(inner, crop - inner - 1, (n_sc, n_pos, 2)); perm = torch.tensor(np.stack([g.permutation(n_pos) for _ in range(n_sc)]), device=dev)
        if eps > 0:                                                                 # displaced moving-side positions (same crop; >= 2 patch radii away)
            obs = []
            for s in range(n_sc):
                o, nm = mismatch_positions(g, yx_np[s], inner, crop - inner - 1, eps, radius); obs.append(o); mis_total += nm
            yx_v_np = np.stack(obs)
        else:
            yx_v_np = yx_np
        yx = torch.tensor(yx_np, dtype=torch.float32, device=dev); yx_v = torch.tensor(yx_v_np, dtype=torch.float32, device=dev)
        du = torch.stack([P.sample_map(torch, F, DU[s], yx[s]) for s in range(n_sc)]); dv = torch.stack([P.sample_map(torch, F, DV[s], yx_v[s]) for s in range(n_sc)])
        if objective == "infonce":
            fm = cr(torch.cat([du[:, :, None].expand(-1, -1, n_pos, -1), dv[:, None].expand(-1, n_pos, -1, -1)], -1)).squeeze(-1)
            loss = P.loss_fn(torch, F, "infonce", None, None, fm)
        else:
            dvq = torch.stack([dv[s, perm[s]] for s in range(n_sc)])
            fp = cr(torch.cat([du, dv], -1)).squeeze(-1); fq = cr(torch.cat([du, dvq], -1)).squeeze(-1); loss = P.loss_fn(torch, F, objective, fp, fq)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if step % (50 if smoke else 250) == 0 or step == steps:
            torch.save({"eU": eU.state_dict(), "eV": eV.state_dict(), "critic": cr.state_dict(), "step": step}, out / f"ckpt_{step:05d}.pt")
            log.append({"step": step, "loss": float(loss.detach()), "seconds": time.time() - t0}); print(f"[train {uid}] step {step} loss {float(loss.detach()):.4f} ({time.time() - t0:.0f}s)", flush=True)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else None
    json.dump({"unit": uid, "objective": objective, "seed": seed, "steps": steps, "fit_fraction": frac, "train_mismatch": eps, "fit": fit, "fit_counts_by_slide": counts,
               "n_fit_full": len(fit_full), "realised_mismatch_fraction": mis_total / float(steps * n_sc * n_pos), "exclusion_radius_px": radius, "log": log,
               "seconds": time.time() - t0, "peak_gib": peak, "state_hash": state_hash(m, fit, seed, frac, eps, objective, steps), "code_hash": code_hash()},
              open(out / "train.json", "w"), indent=1)
    return 0


def cmd_comir(uid, frac, eps, seed, smoke=False, p1_root=P1_ROOT, comir_epochs=0):
    """REG-P1's cmd_comir (recipe unchanged) on the sampled FIT cores; crop-level mismatch: exactly round(ε × batch) patches take modality B from
    a displaced crop of the same image (displacement >= 128 px = 2 × the 64-px patch radius), same flip / rotation on both."""
    torch, nn, F = P.torch_mods(); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); g = np.random.default_rng(3000 + seed)
    m = P.load_manifest(p1_root, smoke); have = lambda c: (p1_root / "cache" / f"{c}.npz").exists()
    fit_full = sorted(c for c, r in m["roles"].items() if r == "FIT" and have(c)); fit, counts = sample_fit(fit_full, frac, seed)
    imgs = {c: P.load_img(p1_root, c, ("shg_log_n", "bf_rgb_n")) for c in fit}
    mA, mB, args = P.comir_models(torch, seed, dev); oA = torch.optim.SGD(mA.parameters(), lr=1e-2, weight_decay=1e-5, momentum=0.9); oB = torch.optim.SGD(mB.parameters(), lr=1e-2, weight_decay=1e-5, momentum=0.9)
    epochs, spe, bs = (2, 4, 8) if smoke else (23, 32, 32); tau = 0.5
    if comir_epochs:
        epochs, spe, bs = comir_epochs, 32, 32
    out = unit_root(uid) / "run"; (out / "latents").mkdir(parents=True, exist_ok=True); t0 = time.time(); log = []; big = 182; radius = 2 * PATCH_RADIUS["comir"]; mis_total = 0
    for ep in range(1, epochs + 1):
        mA.train(); mB.train()
        for _ in range(spe):
            xa, xb = [], []; n_mis = int(round(eps * bs)); mis_idx = set(g.permutation(bs)[:n_mis].tolist()) if n_mis else set(); mis_total += n_mis
            for bi, c in enumerate(g.choice(fit, bs, replace=True)):
                A_, B_ = imgs[c]; y0, x0 = g.integers(0, P.SIZE - big, 2)
                if bi in mis_idx:
                    y1, x1 = displaced(g, (float(y0), float(x0)), 0, P.SIZE - big, radius); y1, x1 = int(round(y1)), int(round(x1))
                else:
                    y1, x1 = y0, x0
                st = torch.tensor(np.concatenate([A_[y0:y0 + big, x0:x0 + big, None], B_[y1:y1 + big, x1:x1 + big]], -1), device=dev).permute(2, 0, 1)
                if g.random() < 0.5:
                    st = st.flip(-1)
                ang = math.radians(g.uniform(-180, 180)); ca, sa = math.cos(ang), math.sin(ang)
                theta = torch.tensor([[ca, -sa, 0.0], [sa, ca, 0.0]], dtype=torch.float32, device=dev)[None]
                grid = F.affine_grid(theta, (1, 4, big, big), align_corners=False); st = F.grid_sample(st[None], grid, mode="bilinear", padding_mode="reflection", align_corners=False)[0]
                c0 = (big - 128) // 2; st = st[:, c0:c0 + 128, c0:c0 + 128]; xa.append(st[:1]); xb.append(st[1:])
            xa, xb = torch.stack(xa), torch.stack(xb); ka, kb = g.integers(4, size=bs), g.integers(4, size=bs)
            rot = lambda t, k: torch.stack([torch.rot90(t[i], int(k[i]), (1, 2)) for i in range(len(t))])
            L1 = rot(mA(rot(xa, ka)), -ka); L2 = rot(mB(rot(xb, kb)), -kb)
            pts = torch.cat([L1, L2]).reshape(2 * bs, -1); sim = -torch.cdist(pts, pts) ** 2 / pts.shape[1] / tau
            n2 = 2 * bs; idx = torch.arange(n2, device=dev); pos = sim[idx, (idx + bs) % n2]; neg = sim.masked_fill(torch.eye(n2, dtype=torch.bool, device=dev), -float("inf"))
            loss = (-pos + torch.logsumexp(neg, 1)).mean()
            Z = L1.numel() + L2.numel(); loss = loss + 1e-4 * (1 + (L1.abs().sum() + L2.abs().sum())) / Z + 1e-4 * (1 + (L1.pow(2).sum() + L2.pow(2).sum())) / Z
            oA.zero_grad(); oB.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(mA.parameters(), 1.0); torch.nn.utils.clip_grad_norm_(mB.parameters(), 1.0); oA.step(); oB.step()
        log.append({"epoch": ep, "loss": float(loss.detach()), "seconds": time.time() - t0}); print(f"[comir {uid}] epoch {ep} loss {float(loss.detach()):.4f} ({time.time() - t0:.0f}s)", flush=True)
    mA.eval(); mB.eval(); torch.save({"A": mA.state_dict(), "B": mB.state_dict(), "args": args}, out / "comir.pt")
    with torch.no_grad():
        for c, r in m["roles"].items():
            if r in ("CAL", "DEV") and have(c):
                A_, B_ = P.load_img(p1_root, c, ("shg_log_n", "bf_rgb_n"))
                la = mA(torch.tensor(A_, device=dev)[None, None]); lb = mB(torch.tensor(B_, device=dev).permute(2, 0, 1)[None])
                np.savez_compressed(out / "latents" / f"{c}.npz", shg=la[0].cpu().numpy(), bf=lb[0].cpu().numpy())
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else None
    json.dump({"unit": uid, "seed": seed, "epochs": epochs, "fit_fraction": frac, "train_mismatch": eps, "fit": fit, "fit_counts_by_slide": counts, "n_fit_full": len(fit_full),
               "realised_mismatch_fraction": mis_total / float(epochs * spe * bs), "exclusion_radius_px": radius, "log": log, "seconds": time.time() - t0, "peak_gib": peak,
               "comir_commit": P.comir_commit(), "state_hash": state_hash(m, fit, seed, frac, eps, "comir", epochs), "code_hash": code_hash()}, open(out / "comir.json", "w"), indent=1)
    return 0


def fit_kernel_g1(torch, F, root, fit, mult, seed, dev, eps=0.0, m_feat=1024, n_per=2048):
    """REG-P1's closed-form RFF kernel critic fitted on the sampled FIT cores; with ε > 0, round(ε n) moving patches per core at displaced positions
    (same core, >= 8 px = 2 × the 4-px patch radius).  ε = 0 reproduces P.fit_kernel exactly (same RNG consumption)."""
    g = np.random.default_rng(7000 + seed); zp, zq = [], []; radius = 2 * PATCH_RADIUS["kernel"]
    for c in fit:
        U, V = (torch.tensor(x, device=dev)[None] for x in P.load_img(root, c)); yx_np = g.uniform(P.OMEGA[0], P.OMEGA[1], (n_per, 2))
        yx = torch.tensor(yx_np, dtype=torch.float32, device=dev)
        e = P.Energy(torch, F, "kernel", U, V, (0.0, 0.0), g, W=None, b=None, w=None); pu = e.patches(U, yx)
        yv_np, _ = mismatch_positions(g, yx_np, P.OMEGA[0], P.OMEGA[1], eps, radius) if eps > 0 else (yx_np, 0)
        pv = e.patches(V, torch.tensor(yv_np, dtype=torch.float32, device=dev)); pm = torch.tensor(g.permutation(n_per), device=dev)
        zp.append(torch.cat([pu, pv], -1)); zq.append(torch.cat([pu, pv[pm]], -1))
    zp, zq = torch.cat(zp), torch.cat(zq); sub = zp[torch.tensor(g.choice(len(zp), 2000, replace=False), device=dev)]
    med = float(torch.cdist(sub, sub).median()); h = mult * med; gt = torch.Generator(device="cpu").manual_seed(seed)
    W_ = (torch.randn(zp.shape[1], m_feat, generator=gt) / h).to(dev); b_ = (torch.rand(m_feat, generator=gt) * 2 * math.pi).to(dev); s_ = math.sqrt(2.0 / m_feat)
    fp, fq = s_ * torch.cos(zp @ W_ + b_), s_ * torch.cos(zq @ W_ + b_); A = (fp.T @ fp) / len(fp) + (fq.T @ fq) / len(fq); d = fp.mean(0) - fq.mean(0)
    A = A + 1e-3 * A.diagonal().mean() * torch.eye(m_feat, device=dev); w_ = torch.linalg.solve(A.double(), d.double()).float()
    return {"W": W_, "b": b_, "w": w_ * s_, "median": med, "h": h}


# ------------------------------------------------------------------------------------------------------------ registration (REG-P2, unchanged, through the unit root)
class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def cmd_register(uid, method, seed, smoke=False, n_pert=P.N_PERT, frac=1.0, eps=0.0):
    """REG-P2's cmd_register on the unit's run (critics: CAL selection over the 12 checkpoints, DEV; CoMIR: encode-once row).  For the kernel row the
    REG-P2 path is replayed with this module's fit (sampled FIT, mismatch) instead of P.fit_kernel."""
    u = ensure_layout(uid, method, seed); p1, p2 = u / "p1", u / "p2"
    if smoke:
        p1, p2 = u / "p1" / "smoke", u / "p2" / "smoke"
    if method == "kernel":
        orig = P.fit_kernel
        m = P.load_manifest(p1, smoke); fit_full = sorted(c for c, r in m["roles"].items() if r == "FIT" and (p1 / "cache" / f"{c}.npz").exists()); fit, counts = sample_fit(fit_full, frac, seed)
        P.fit_kernel = lambda torch, F, root, _fit, mult, s, dev: fit_kernel_g1(torch, F, root, fit, mult, s, dev, eps)   # REG-P2 calls P.fit_kernel(torch, F, root, fit, mult, seed, dev)
        try:
            rc = Q.cmd_register(_Args(method="kernel", seed=seed, smoke=smoke, n_pert=n_pert, root=p2, p1_root=p1))
        finally:
            P.fit_kernel = orig
        res_path = p2 / "results" / ("kernel_s%d%s.json" % (seed, "_smoke" if smoke else "")); r = json.load(open(res_path)); r.update({"unit": uid, "fit_fraction": frac, "train_mismatch": eps, "fit": fit, "fit_counts_by_slide": counts})
        json.dump(r, open(res_path, "w"), indent=1); return rc
    return Q.cmd_register(_Args(method=method, seed=seed, smoke=smoke, n_pert=n_pert, root=p2, p1_root=p1))


def result_path(uid, method, seed, smoke=False):
    name = method + ("" if method in ("mi", "nmi") else f"_s{seed}") + ("_smoke" if smoke else "")
    return unit_root(uid) / "p2" / ("smoke/" if smoke else "") / "results" / f"{name}.json"


def mi_nmi_reuse_check(p2_root=P2_ROOT, p1_root=P1_ROOT):
    """MI / NMI depend on no training data: REG-P2's results are reused iff the perturbations and the cache manifest they read are the ones on disk."""
    pert = hashlib.sha256((p2_root / "perturbations.json").read_bytes()).hexdigest()[:16]; man = hashlib.sha256((p1_root / "manifest.json").read_bytes()).hexdigest()[:16]
    out = {"perturbations_sha256": pert, "manifest_sha256": man, "results": {}}
    for mth in ("mi", "nmi"):
        f = p2_root / "results" / f"{mth}.json"; r = json.load(open(f)) if f.exists() else None
        out["results"][mth] = {"file": str(f), "present": r is not None, "n_pert": r.get("n_pert") if r else None, "dev_images": len([k for k in r["dev"] if "#" not in k]) if r else None}
    return out


# ------------------------------------------------------------------------------------------------------------ unit table
def unit_table(p1_root=P1_ROOT):
    m = P.load_manifest(p1_root, False); fit_full = sorted(c for c, r in m["roles"].items() if r == "FIT"); rows = []
    for uid, method, frac, eps, seed in all_units() + kernel_units():
        fit, counts = sample_fit(fit_full, frac, seed); reuse = r6_reusable(m, fit_full, method, frac, eps, seed)
        done = result_path(uid, method, seed).exists()
        rows.append({"unit_id": uid, "dataset": "BF-SHG", "method": method, "seed": seed, "fit_fraction": frac, "train_mismatch": eps, "n_fit": len(fit), "fit_by_slide": json.dumps(counts),
                     "reuse": "r6:" + reuse["run"] if reuse else "none_seed_differs" if (frac == 1.0 and eps == 0.0 and method != "kernel") else "not_applicable",
                     "status": "done" if done else "planned_not_submitted", "state_hash": state_hash(m, fit, seed, frac, eps, method, 3000 if method not in ("comir", "kernel") else 23)})
    return rows
