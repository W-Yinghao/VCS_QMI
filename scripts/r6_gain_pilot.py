"""r6 GAIN-P1 — task-relevant evidence gain and on-demand modality computation on natural data (r6 package docs/05_EVIDENCE_GAIN_CN.md;
protocol draft reports/R6/R6_GAIN_P1_PROTOCOL_DRAFT_20261011.yaml; controlled checks scripts/r6_gain_controlled.py).

Dataset 1, AV-MNIST (MultiBench release): U = image (28 × 28, cheap), V = audio spectrogram (112 × 112, extra branch), Y = digit (10 classes).
Only the official train release (60 000) is read; the official test arrays stay inside the archive (closed).  Images are all unique, but
the audio side holds only 1 601 byte-exact unique spectrograms, each reused ~37 times and paired with images of several digits (mean label
purity 0.44; `audit`, 2026-10-11).  Roles are therefore grouped by audio clip (byte-exact spectrogram hash), so no DEV / CAL / FIT_POLICY
spectrogram occurs in FIT_TASK; the 6 FSDD speakers still overlap across roles (too few to group; disclosed).
Dataset 2, CREMA-D (plugs in through the same interface): roles by actor; U = audio log-mel (cheap), V = video frames (decoding + branch).

System (both families identical apart from the loss): shared U encoder; cheap head a(U); V encoder + fusion head b(U, V); trained end to end
on FIT_TASK with family "brier" (multi-class Brier on softmax probabilities) or "ce" (cross-entropy); epoch chosen on CAL (Brier(a) + Brier(b)).
Policy (U side only: the cheap features and a's probabilities): MLP regressing z = ||b − a||² (GAIN-square) or h = ||e_y − a||² − ||e_y − b||²
(GAIN-utility) on FIT_POLICY out-of-sample predictions; baselines never / always / random / entropy / 1 − max p; oracles h, z (not deployable).
Budgets 10 / 25 / 50 / 75 % (plus 0 / 100 %).  Primary: exact-rate routing on DEV — every policy acquires the top ⌈βn⌉ samples by its score
(labels never used; ties by a seeded random order), so contrasts measure ranking quality.  Secondary: deployable thresholds from CAL score
quantiles with their realised DEV rates.  Costs measured per component (AV-MNIST: recorded, no cost claims).

    python scripts/r6_gain_pilot.py roles --dataset avmnist
    python scripts/r6_gain_pilot.py audit --dataset avmnist
    python scripts/r6_gain_pilot.py run --dataset avmnist --family brier --seed 0 [--smoke]
    python scripts/r6_gain_pilot.py aggregate --dataset avmnist [--smoke]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path("/home/infres/yinwang/CS_QMI/outputs/R6_GAIN")
AVM = Path("/projects/EEG-foundation-model/yinghao/data/av_mnist_multibench/extracted/avmnist")
CREMA = Path("/projects/EEG-foundation-model/yinghao/data/crema_d/media")    # verified LFS media (sha256 + size vs the pointers); repo/ holds stubs
SEED = 20261011
ROLE_FRAC = (("FIT_TASK", 0.6), ("FIT_POLICY", 0.2), ("CAL", 0.1), ("DEV", 0.1))
BUDGETS = (0.10, 0.25, 0.50, 0.75)
FAMILIES, SEEDS = ("brier", "ce"), tuple(range(10))
DEPLOYABLE = ("random", "entropy", "maxprob", "gain_z", "gain_h")
SCORED = ("entropy", "maxprob", "gain_z", "gain_h")
CONTRASTS = {"gain_h_minus_entropy": ("gain_h", "entropy"), "gain_h_minus_random": ("gain_h", "random"), "gain_z_minus_gain_h": ("gain_z", "gain_h"),
             "gain_z_minus_entropy": ("gain_z", "entropy"), "gain_h_minus_oracle_h": ("gain_h", "oracle_h"), "entropy_minus_random": ("entropy", "random")}
ORACLES = ("oracle_h", "oracle_z")
RANDOM_REPEATS, BATCH, EPOCHS, POLICY_EPOCHS = 20, 256, 30, 300


# ---------------------------------------------------------------------------------------------------------------- data
def avmnist_arrays():
    """The official train release only (60 000 samples).  test_data.npy / test_labels.npy are never extracted or read."""
    u = np.load(AVM / "image" / "train_data.npy", mmap_mode="r", allow_pickle=False)        # (60000, 784) float64
    v = np.load(AVM / "audio" / "train_data.npy", mmap_mode="r", allow_pickle=False)        # (60000, 112, 112) uint8
    y = np.load(AVM / "train_labels.npy", allow_pickle=False).astype(np.int64)
    return u, v, y


def crema_clips():
    """CREMA-D clip list from the verified media folder (file names ActorID_Sentence_Emotion_Level); groups = actor.  Emotions are the 6
    official classes; the role split is by actor.  Media decoding (WAV, FLV) is not part of this round."""
    wav = sorted((CREMA / "AudioWAV").glob("*.wav")); emo = ("ANG", "DIS", "FEA", "HAP", "NEU", "SAD"); rows = []
    for p in wav:
        parts = p.stem.split("_")
        if len(parts) == 4 and parts[2] in emo:
            rows.append({"stem": p.stem, "actor": int(parts[0]), "sentence": parts[1], "label": emo.index(parts[2]), "level": parts[3]})
    return rows


def avmnist_clip_groups(v):
    """Audio-clip source groups: byte-exact duplicate spectrograms (sha1 of the uint8 array) -> group index."""
    hs = np.array([hashlib.sha1(np.ascontiguousarray(v[i]).tobytes()).hexdigest() for i in range(len(v))])
    return np.unique(hs, return_inverse=True)[1].astype(np.int64)


def groups_and_labels(dataset):
    if dataset == "avmnist":
        _, v, y = avmnist_arrays(); return avmnist_clip_groups(v), y      # audio clip = independent unit (images are unique)
    if dataset == "cremad":
        rows = crema_clips(); return np.array([r["actor"] for r in rows]), np.array([r["label"] for r in rows])
    raise ValueError(dataset)


def make_roles(groups, seed=SEED):
    """Whole independent units (groups) go to one role: seeded permutation of the unique groups cut at the fixed fractions."""
    ug = np.unique(groups); perm = np.random.default_rng(seed).permutation(len(ug)); n = len(ug)
    sizes = [int(round(f * n)) for _, f in ROLE_FRAC]; sizes[-1] = n - sum(sizes[:-1])
    out, start = {}, 0
    for (name, _), k in zip(ROLE_FRAC, sizes):
        chosen = ug[perm[start:start + k]]; start += k
        out[name] = np.sort(np.flatnonzero(np.isin(groups, chosen)))
    return out


def roles_path(dataset, smoke=False):
    return OUT / ("smoke" if smoke else "") / dataset / "roles.json"


def load_roles(dataset, smoke=False):
    r = json.load(open(roles_path(dataset, False)))
    roles = {k: np.array(v["indices"], dtype=np.int64) for k, v in r["roles"].items()}
    if smoke:
        lim = {"FIT_TASK": 2000, "FIT_POLICY": 800, "CAL": 400, "DEV": 400}; roles = {k: v[:lim[k]] for k, v in roles.items()}
    return roles


# ---------------------------------------------------------------------------------------------------------------- models
def build_system(dataset, n_classes):
    import torch
    import torch.nn as nn

    def mnist_enc():
        return nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                             nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU())

    def spec_enc(ch=1):
        return nn.Sequential(nn.Conv2d(ch, 32, 5, stride=2, padding=2), nn.ReLU(), nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                             nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.ReLU(),
                             nn.AdaptiveAvgPool2d(1), nn.Flatten())

    class FrameEnc(nn.Module):                                                       # CREMA-D video: per-frame CNN, mean over frames
        def __init__(self):
            super().__init__(); self.f = spec_enc(3)

        def forward(self, x):                                                        # x: [B, T, 3, H, W]
            b, t = x.shape[:2]; return self.f(x.flatten(0, 1)).view(b, t, -1).mean(1)

    class System(nn.Module):
        def __init__(self):
            super().__init__()
            self.u_enc = mnist_enc() if dataset == "avmnist" else spec_enc(1)
            self.v_enc = spec_enc(1) if dataset == "avmnist" else FrameEnc()
            self.head_a = nn.Linear(128, n_classes)
            self.head_b = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, n_classes))

        def cheap(self, u):
            h = self.u_enc(u); return h, self.head_a(h)

        def full(self, h, v):
            return self.head_b(torch.cat([h, self.v_enc(v)], 1))

    return System()


def family_loss(logits, y, family):
    import torch.nn.functional as F
    if family == "ce":
        return F.cross_entropy(logits, y)
    if family == "brier":                                                            # multi-class Brier on softmax probabilities
        p = logits.softmax(1); e = F.one_hot(y, p.shape[1]).to(p.dtype); return ((p - e) ** 2).sum(1).mean()
    raise ValueError(family)


def brier_vec(p, y):
    e = np.eye(p.shape[1])[y]; return ((p - e) ** 2).sum(1)


def gain_targets(pa, pb, y):
    """z = ||b − a||² (squared disagreement) and h = ||e − a||² − ||e − b||² (realised Brier improvement); h − z = 2 (b − a)·(e − b)."""
    e = np.eye(pa.shape[1])[y]; z = ((pb - pa) ** 2).sum(1); h = ((e - pa) ** 2).sum(1) - ((e - pb) ** 2).sum(1); return z, h


def policy_features(feat_u, pa):
    """The policy sees only deployment-visible U-side quantities: the cheap features and a's probabilities (never V, b, y or a gain)."""
    return np.concatenate([np.asarray(feat_u, np.float32), np.asarray(pa, np.float32)], 1)


# ---------------------------------------------------------------------------------------------------------------- tensors
class Tensors:
    """Holds the role arrays in memory (normalised with FIT_TASK statistics only) and yields batches on the device."""
    def __init__(self, dataset, roles, dev):
        import torch
        self.dev, self.dataset = dev, dataset
        if dataset != "avmnist":
            raise NotImplementedError("CREMA-D tensors: decode audio log-mel / video frames first (crema_load), not implemented in this round")
        u, v, y = avmnist_arrays(); fit = roles["FIT_TASK"]
        self.u_mu, self.u_sd = float(np.mean(u[fit])), float(np.std(u[fit])) + 1e-6
        s1 = s2 = 0.0
        for i in range(0, len(fit), 4096):                                            # FIT_TASK statistics in chunks (float64 sums)
            c = np.asarray(v[fit[i:i + 4096]], np.float64) / 255.0; s1 += float(c.sum()); s2 += float((c ** 2).sum())
        m = len(fit) * v.shape[1] * v.shape[2]; self.v_mu = s1 / m; self.v_sd = math.sqrt(max(s2 / m - self.v_mu ** 2, 0.0)) + 1e-6
        self.y = torch.as_tensor(y)
        self.u = torch.as_tensor(((np.asarray(u, np.float32) - self.u_mu) / self.u_sd).reshape(-1, 1, 28, 28))
        self.v_raw = torch.from_numpy(np.array(v))                                    # uint8 copy in RAM; decoded / normalised per batch (V cost)

    def u_batch(self, idx):
        return self.u[idx].to(self.dev, non_blocking=True)

    def v_batch(self, idx):
        import torch
        return ((self.v_raw[idx].to(self.dev, non_blocking=True).float() / 255.0 - self.v_mu) / self.v_sd).unsqueeze(1)

    def y_batch(self, idx):
        return self.y[idx].to(self.dev, non_blocking=True)


# ---------------------------------------------------------------------------------------------------------------- training / prediction
def train_system(T, roles, family, seed, epochs, dev, log):
    import copy
    import torch
    torch.manual_seed(seed); np.random.seed(seed)
    model = build_system(T.dataset, int(T.y.max()) + 1).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    fit = torch.as_tensor(roles["FIT_TASK"]); g = torch.Generator().manual_seed(seed); best = (math.inf, 0, None); hist = []
    for ep in range(1, epochs + 1):
        model.train(); perm = fit[torch.randperm(len(fit), generator=g)]; t0 = time.time(); tot = 0.0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]; y = T.y_batch(idx); h, la = model.cheap(T.u_batch(idx)); lb = model.full(h, T.v_batch(idx))
            loss = family_loss(la, y, family) + family_loss(lb, y, family); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tot += float(loss.detach()) * len(idx)
        _, pa, pb = predict(model, T, roles["CAL"]); yc = T.y[roles["CAL"]].numpy()
        ra, rb = float(brier_vec(pa, yc).mean()), float(brier_vec(pb, yc).mean()); hist.append({"epoch": ep, "train_loss": tot / len(perm), "cal_brier_a": ra, "cal_brier_b": rb, "seconds": time.time() - t0})
        log(f"epoch {ep}: loss {tot / len(perm):.4f} CAL Brier a {ra:.4f} b {rb:.4f} ({time.time() - t0:.1f}s)")
        if ra + rb < best[0]:
            best = (ra + rb, ep, copy.deepcopy(model.state_dict()))
    model.load_state_dict(best[2]); return model, hist, best[1]


def predict(model, T, idx, chunk=1024):
    import torch
    model.eval(); feats, pas, pbs = [], [], []
    with torch.no_grad():
        for i in range(0, len(idx), chunk):
            j = torch.as_tensor(idx[i:i + chunk]); h, la = model.cheap(T.u_batch(j)); lb = model.full(h, T.v_batch(j))
            feats.append(h.float().cpu().numpy()); pas.append(la.softmax(1).double().cpu().numpy()); pbs.append(lb.softmax(1).double().cpu().numpy())
    return np.concatenate(feats), np.concatenate(pas), np.concatenate(pbs)


def train_policy(Xf, tf, Xc, tc, seed, dev, epochs=POLICY_EPOCHS, patience=30):
    """Two-layer MLP regressor on U-side features; standardised with FIT_POLICY statistics; MSE; early stopping on CAL MSE."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed); mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
    net = nn.Sequential(nn.Linear(Xf.shape[1], 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3); tmu, tsd = float(tf.mean()), float(tf.std()) + 1e-9
    Xft = torch.as_tensor((Xf - mu) / sd, dtype=torch.float32, device=dev); tft = torch.as_tensor((tf - tmu) / tsd, dtype=torch.float32, device=dev)
    Xct = torch.as_tensor((Xc - mu) / sd, dtype=torch.float32, device=dev); tct = torch.as_tensor((tc - tmu) / tsd, dtype=torch.float32, device=dev)
    g = torch.Generator().manual_seed(seed); best = (math.inf, 0, None); bad = 0
    import copy
    for ep in range(1, epochs + 1):
        net.train(); perm = torch.randperm(len(Xft), generator=g).to(dev)
        for i in range(0, len(perm), 512):
            j = perm[i:i + 512]; loss = ((net(Xft[j]).squeeze(1) - tft[j]) ** 2).mean(); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            cal = float(((net(Xct).squeeze(1) - tct) ** 2).mean())
        if cal < best[0] - 1e-6:
            best, bad = (cal, ep, copy.deepcopy(net.state_dict())), 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best[2]); net.eval()

    def score(X):
        with torch.no_grad():
            return net(torch.as_tensor((X - mu) / sd, dtype=torch.float32, device=dev)).squeeze(1).double().cpu().numpy() * tsd + tmu
    return score, {"cal_mse_std": best[0], "epoch": best[1]}


# ---------------------------------------------------------------------------------------------------------------- routing and evaluation
def entropy(p):
    return -(p * np.log(np.clip(p, 1e-12, 1))).sum(1)


def route(acquire, pa, pb, y):
    p = np.where(acquire[:, None], pb, pa); b = brier_vec(p, y); return b, (p.argmax(1) == y).astype(np.float64)


def budget_curve(score_cal, score_dev, pa, pb, y, budgets=BUDGETS):
    """Threshold = (1 − β) quantile of the CAL scores; DEV acquires score > threshold.  Returns per-budget realised rate and per-sample risk."""
    out = {}
    for beta in budgets:
        tau = np.quantile(score_cal, 1 - beta); acq = score_dev > tau; b, a = route(acq, pa, pb, y)
        out[str(beta)] = {"threshold": float(tau), "rate": float(acq.mean()), "brier": float(b.mean()), "acc": float(a.mean()), "per_sample_brier": b, "acquire": acq}
    return out


def n_acquire(beta, n):
    return int(math.ceil(beta * n - 1e-9))


def topk_curve(score, pa, pb, y, tie_seed, budgets=BUDGETS):
    """Primary: exact-rate routing — acquire the top ⌈βn⌉ samples by score (descending), ties broken by a seeded random order.  The
    selection uses only the score and the tie seed; y enters only the evaluation of the routed prediction."""
    n = len(score); key = np.random.default_rng(tie_seed).permutation(n); order = np.lexsort((key, -np.asarray(score, np.float64))); out = {}
    for beta in budgets:
        acq = np.zeros(n, bool); acq[order[:n_acquire(beta, n)]] = True; b, a = route(acq, pa, pb, y)
        out[str(beta)] = {"rate": float(acq.mean()), "brier": float(b.mean()), "acc": float(a.mean()), "per_sample_brier": b, "acquire": acq}
    return out


def random_curve(n, pa, pb, y, seed, budgets=BUDGETS, repeats=RANDOM_REPEATS):
    rng = np.random.default_rng(seed + 7); out = {}
    for beta in budgets:
        k = n_acquire(beta, n); bs, accs, per = [], [], np.zeros(n)
        for _ in range(repeats):
            acq = np.zeros(n, bool); acq[rng.choice(n, k, replace=False)] = True; b, a = route(acq, pa, pb, y); bs.append(b.mean()); accs.append(a.mean()); per += b / repeats
        out[str(beta)] = {"rate": k / n, "brier": float(np.mean(bs)), "brier_sd_over_repeats": float(np.std(bs)), "acc": float(np.mean(accs)), "per_sample_brier": per}
    return out


def gain_bins(score, h, z, nbins=10):
    q = np.quantile(score, np.linspace(0, 1, nbins + 1)); idx = np.clip(np.searchsorted(q, score, side="right") - 1, 0, nbins - 1); rows = []
    for k in range(nbins):
        m = idx == k
        if m.any():
            rows.append({"bin": k, "n": int(m.sum()), "pred": float(score[m].mean()), "h": float(h[m].mean()), "z": float(z[m].mean()), "h_minus_z": float((h[m] - z[m]).mean())})
    return rows


# ---------------------------------------------------------------------------------------------------------------- costs
def measure_costs(model, T, policy_dim, dev, n=2048):
    """Per-sample milliseconds of each deployed component, at batch 256 and 1; sparse routing at each budget (gather + V branch + fusion)."""
    import torch
    import torch.nn as nn
    cuda = dev.type == "cuda"; idx = torch.arange(min(n, len(T.y))); pol = nn.Sequential(nn.Linear(policy_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)).to(dev)
    model.eval()

    def timed(fn, reps=5):
        fn()
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        if cuda:
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps

    res = {"device": str(dev), "gpu": torch.cuda.get_device_name(0) if cuda else "cpu"}
    with torch.no_grad():
        for bs in (256, 1):
            j = idx[:bs]; u = T.u_batch(j); h, la = model.cheap(u); v = T.v_batch(j); X = torch.cat([h, la.softmax(1)], 1)
            res[f"bs{bs}"] = {"u_load_ms": 1e3 * timed(lambda: T.u_batch(j)) / bs, "u_branch_ms": 1e3 * timed(lambda: model.cheap(u)) / bs,
                              "policy_ms": 1e3 * timed(lambda: pol(X)) / bs, "entropy_ms": 1e3 * timed(lambda: -(la.softmax(1) * la.log_softmax(1)).sum(1)) / bs,
                              "v_decode_ms": 1e3 * timed(lambda: T.v_batch(j)) / bs,
                              "v_branch_ms": 1e3 * timed(lambda: model.v_enc(v)) / bs, "fusion_ms": 1e3 * timed(lambda: model.head_b(torch.cat([h, model.v_enc(v)], 1))) / bs
                              - 1e3 * timed(lambda: model.v_enc(v)) / bs}
        j = idx[:256]; u = T.u_batch(j); h, la = model.cheap(u); sparse = {}
        for beta in (0.10, 0.25, 0.50, 0.75, 1.0):
            k = max(1, int(round(beta * 256))); sel = j[:k]
            sparse[str(beta)] = 1e3 * timed(lambda: model.full(h[:k], T.v_batch(sel))) / 256
        res["sparse_extra_ms_per_sample_bs256"] = sparse
    return res


# ---------------------------------------------------------------------------------------------------------------- subcommands
def cmd_roles(a):
    groups, y = groups_and_labels(a.dataset); roles = make_roles(groups)
    rec = {"dataset": a.dataset, "seed": SEED, "fractions": dict(ROLE_FRAC), "n": int(len(y)), "n_groups": int(len(np.unique(groups))), "groups": groups.tolist(),
           "independent_unit": "audio clip (byte-exact spectrogram; images unique; FSDD speakers overlap)" if a.dataset == "avmnist" else "actor",
           **({"stems": [r["stem"] for r in crema_clips()], "media_root": str(CREMA)} if a.dataset == "cremad" else {}),
           "roles": {k: {"n": int(len(v)), "n_groups": int(len(np.unique(groups[v]))), "sha256": hashlib.sha256(v.tobytes()).hexdigest(), "label_counts": np.bincount(y[v]).tolist(), "indices": v.tolist()}
                     for k, v in roles.items()}, "official_test": "not read (closed)"}
    p = roles_path(a.dataset); p.parent.mkdir(parents=True, exist_ok=True); json.dump(rec, open(p, "w"))
    print(json.dumps({"n_groups": rec["n_groups"], **{k: {kk: vv for kk, vv in v.items() if kk != "indices"} for k, v in rec["roles"].items()}}, indent=1)); return 0


def cmd_audit(a):
    """AV-MNIST audio near-duplicates across roles.  Spectrograms (56 × 56 average-pooled) are standardised per pixel with FIT_TASK
    statistics — the pattern shared by all spectrograms otherwise makes every pair look identical (raw-centred cosine ≈ 1 everywhere) —
    then each query's nearest FIT_TASK neighbour by cosine is found, with reference levels (cosine to a random same-label and a random
    other-label FIT_TASK sample) and the neighbour's label agreement by cosine bin.  High-cosine neighbours with the same label indicate a
    reused FSDD clip (with different ESC-50 noise); same-noise, different-digit pairs would show high cosine with a different label.
    Without source IDs neither can be repaired; the result is disclosed with every GAIN result."""
    if a.dataset != "avmnist":
        print("audit is AV-MNIST specific"); return 0
    _, v, y = avmnist_arrays(); roles = load_roles(a.dataset, a.smoke); rng = np.random.default_rng(SEED)

    def pooled(idx):
        out = np.empty((len(idx), 56 * 56), np.float32)
        for i in range(0, len(idx), 4096):
            j = idx[i:i + 4096]; out[i:i + len(j)] = np.asarray(v[j], np.float32).reshape(len(j), 56, 2, 56, 2).mean((2, 4)).reshape(len(j), -1)
        return out

    u_img = avmnist_arrays()[0]; role_of = np.empty(len(y), object)
    for k, idx in load_roles(a.dataset, False).items():
        role_of[idx] = k
    exact = {}
    for name, arr in (("audio", v), ("image", u_img)):
        hs = np.array([hashlib.sha1(np.ascontiguousarray(arr[i]).tobytes()).hexdigest() for i in range(len(y))]); _, inv, cnt = np.unique(hs, return_inverse=True, return_counts=True)
        mult = cnt[inv]; purity = []; nlab = []; cross = 0
        for g in np.flatnonzero(cnt > 1):
            members = np.flatnonzero(inv == g) if cnt[g] < 1000 else None
            if members is not None:
                bc = np.bincount(y[members]); purity.append(float(bc.max() / len(members))); nlab.append(int((bc > 0).sum()))
                cross += int(len(set(role_of[members])) > 1) * len(members)
        exact[name] = {"n": int(len(y)), "n_unique": int(len(cnt)), "frac_samples_with_exact_duplicate": float((mult > 1).mean()),
                       "group_size_percentiles": {str(q): float(np.percentile(mult, q)) for q in (50, 90, 99, 100)},
                       "label_purity_of_duplicate_groups_mean": float(np.mean(purity)) if purity else None,
                       "label_purity_percentiles": {str(q): float(np.percentile(purity, q)) for q in (5, 50, 95)} if purity else None,
                       "distinct_labels_per_group_percentiles": {str(q): float(np.percentile(nlab, q)) for q in (5, 50, 95)} if nlab else None,
                       "samples_in_groups_spanning_roles": cross}
    ref_idx = roles["FIT_TASK"]; R = pooled(ref_idx); mu = R.mean(0); sdp = R.std(0); sd = sdp + 0.01 * float(R.std()) + 1e-6
    full = np.asarray(v[ref_idx[:2000]])
    raw = {"value_range": [int(full.min()), int(full.max())], "frac_zero": float((full == 0).mean()), "frac_255": float((full == 255).mean()),
           "total_var_pooled": float(R.var()), "per_sample_deviation_var": float(((R - mu) ** 2).mean()),
           "share_of_variance_common_pattern": float(1 - ((R - mu) ** 2).mean() / R.var()), "frac_pixels_near_constant": float((sdp < 1e-3 * float(R.std())).mean()),
           "raw_centred_cos_random_pairs_median": float(np.median((lambda Z: (Z[:1000] * Z[1000:2000]).sum(1))(
               (lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-6))(R[:2000] - R[:2000].mean(1, keepdims=True)))))}

    def emb(X):
        Z = (X - mu) / sd; return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-6)

    groups_all = np.array(json.load(open(roles_path(a.dataset)))["groups"])
    RZ = emb(R); yr = y[ref_idx]; res = {"raw": raw, "exact_byte_duplicates": exact, "nn_excludes": "references in the query's own clip group (exact matches)", "note": "cosines on per-pixel standardised 56x56 pooled spectrograms (FIT_TASK statistics)"}
    same_pool = {c: np.flatnonzero(yr == c) for c in range(10)}; other_pool = {c: np.flatnonzero(yr != c) for c in range(10)}
    for role in ("DEV", "CAL", "FIT_POLICY"):
        q_idx = roles[role]; Q = emb(pooled(q_idx)); yq = y[q_idx]; best = np.full(len(q_idx), -2.0, np.float32); arg = np.zeros(len(q_idx), np.int64)
        n99 = np.zeros(len(q_idx), np.int64); n99_same = np.zeros(len(q_idx), np.int64); lab99 = np.zeros((len(q_idx), 10), np.int64)
        gq = groups_all[q_idx]
        for i in range(0, len(RZ), 4096):
            sim = Q @ RZ[i:i + 4096].T; sim[gq[:, None] == groups_all[ref_idx[i:i + 4096]][None, :]] = -2.0      # exclude exact matches (same clip)
            m = sim.max(1); am = sim.argmax(1) + i; upd = m > best; best[upd] = m[upd]; arg[upd] = am[upd]
            hit = sim > 0.99; n99 += hit.sum(1); n99_same += (hit & (yr[i:i + 4096][None, :] == yq[:, None])).sum(1)
            for c in range(10):
                lab99[:, c] += (hit & (yr[i:i + 4096] == c)[None, :]).sum(1)
        same = yr[arg] == yq; g = n99 > 0
        twin = {"frac_with_twin_cos_gt_0.99": float(g.mean()), "twin_group_size_percentiles": {str(q): float(np.percentile(n99[g], q)) for q in (5, 50, 95)} if g.any() else None,
                "share_of_twins_with_query_label_mean": float((n99_same[g] / n99[g]).mean()) if g.any() else None,
                "distinct_labels_among_twins_median": float(np.median((lab99[g] > 0).sum(1))) if g.any() else None,
                "one_minus_nn_cos_percentiles": {str(q): float(np.percentile(1.0 - best.astype(np.float64), q)) for q in (1, 50, 99)},
                "one_minus_nn_cos_median_same_vs_other_label": [float(np.median(1.0 - best[same])) if same.any() else None, float(np.median(1.0 - best[~same])) if (~same).any() else None]}
        rs = np.array([rng.choice(same_pool[c]) for c in yq]); ro = np.array([rng.choice(other_pool[c]) for c in yq])
        cs, co = (Q * RZ[rs]).sum(1), (Q * RZ[ro]).sum(1); pct = (1, 5, 25, 50, 75, 95, 99)
        bins = [-1.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.01]; by_bin = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mk = (best >= lo) & (best < hi)
            by_bin.append({"cos": [lo, hi], "n": int(mk.sum()), "nn_same_label": float(same[mk].mean()) if mk.any() else None})
        res[role] = {"n": int(len(q_idx)), "nn_cos": {str(p): float(np.percentile(best, p)) for p in pct},
                     "ref_same_label_cos": {str(p): float(np.percentile(cs, p)) for p in pct}, "ref_other_label_cos": {str(p): float(np.percentile(co, p)) for p in pct},
                     "nn_same_label_rate": float(same.mean()), "by_nn_cos": by_bin,
                     "frac_nn_above_same_label_ref_p99": float((best > np.percentile(cs, 99)).mean()), "twins": twin}
    out = OUT / ("smoke" if a.smoke else "") / a.dataset / "audio_duplicate_audit.json"; out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1); print(json.dumps(res, indent=1)); return 0


def cmd_run(a):
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu"); t_start = time.time()
    base = OUT / ("smoke" if a.smoke else "") / a.dataset / f"{a.family}_s{a.seed}"; base.mkdir(parents=True, exist_ok=True)
    log_f = open(base / "log.txt", "a")

    def log(msg):
        line = f"[{a.dataset} {a.family} s{a.seed}] {msg}"; print(line, flush=True); log_f.write(line + "\n"); log_f.flush()

    roles = load_roles(a.dataset, a.smoke); T = Tensors(a.dataset, roles, dev)
    log(f"device {dev}; roles " + ", ".join(f"{k} {len(v)}" for k, v in roles.items()))
    model, hist, ep_sel = train_system(T, roles, a.family, a.seed, 2 if a.smoke else a.epochs, dev, log)
    torch.save({"state": model.state_dict(), "epoch": ep_sel, "family": a.family, "seed": a.seed}, base / "system.pt")
    pred = {r: predict(model, T, roles[r]) for r in ("FIT_POLICY", "CAL", "DEV")}; ys = {r: T.y[roles[r]].numpy() for r in pred}
    tg = {r: gain_targets(pred[r][1], pred[r][2], ys[r]) for r in pred}; X = {r: policy_features(pred[r][0], pred[r][1]) for r in pred}
    scores, pinfo = {}, {}
    for tname, ti in (("gain_z", 0), ("gain_h", 1)):
        fn, info = train_policy(X["FIT_POLICY"], tg["FIT_POLICY"][ti], X["CAL"], tg["CAL"][ti], a.seed, dev, epochs=20 if a.smoke else POLICY_EPOCHS)
        scores[tname] = {"CAL": fn(X["CAL"]), "DEV": fn(X["DEV"])}; pinfo[tname] = info
    for r in ("CAL", "DEV"):
        pa = pred[r][1]
        scores.setdefault("entropy", {})[r] = entropy(pa); scores.setdefault("maxprob", {})[r] = 1 - pa.max(1)
        scores.setdefault("oracle_h", {})[r] = tg[r][1]; scores.setdefault("oracle_z", {})[r] = tg[r][0]
    pa, pb, y = pred["DEV"][1], pred["DEV"][2], ys["DEV"]; z, h = tg["DEV"]
    exact = {k: topk_curve(scores[k]["DEV"], pa, pb, y, tie_seed=SEED + a.seed) for k in SCORED + ORACLES}
    exact["random"] = random_curve(len(y), pa, pb, y, a.seed, repeats=3 if a.smoke else RANDOM_REPEATS)
    thr = {k: budget_curve(scores[k]["CAL"], scores[k]["DEV"], pa, pb, y) for k in SCORED}
    ends = {"never": route(np.zeros(len(y), bool), pa, pb, y), "always": route(np.ones(len(y), bool), pa, pb, y)}
    costs = measure_costs(model, T, X["DEV"].shape[1], dev)
    import resource
    memory = {"cuda_max_allocated_mb": torch.cuda.max_memory_allocated() / 2 ** 20 if dev.type == "cuda" else None,
              "cuda_max_reserved_mb": torch.cuda.max_memory_reserved() / 2 ** 20 if dev.type == "cuda" else None,
              "ru_maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}

    def strip(cv):
        return {b: {kk: vv for kk, vv in c.items() if kk not in ("per_sample_brier", "acquire")} for b, c in cv.items()}

    res = {"dataset": a.dataset, "family": a.family, "seed": a.seed, "smoke": a.smoke, "device": str(dev), "selected_epoch": ep_sel, "history": hist, "policy": pinfo,
           "dev": {"n": int(len(y)), "brier_never": float(ends["never"][0].mean()), "brier_always": float(ends["always"][0].mean()),
                   "acc_never": float(ends["never"][1].mean()), "acc_always": float(ends["always"][1].mean()),
                   "mean_h": float(h.mean()), "frac_h_positive": float((h > 0).mean()), "mean_z": float(z.mean()), "mean_h_minus_z": float((h - z).mean()),
                   "identity_max_abs_err": float(np.abs((h - z) - 2 * ((pb - pa) * (np.eye(pa.shape[1])[y] - pb)).sum(1)).max()),
                   "curves_exact": {k: strip(cv) for k, cv in exact.items()}, "curves_cal_threshold": {k: strip(cv) for k, cv in thr.items()},
                   "gain_bins": {k: gain_bins(scores[k]["DEV"], h, z) for k in ("gain_z", "gain_h", "entropy")},
                   "corr_pred_h": {k: float(np.corrcoef(scores[k]["DEV"], h)[0, 1]) for k in SCORED}},
           "costs": costs, "memory": memory, "seconds_total": time.time() - t_start}
    np.savez_compressed(base / "per_sample.npz", dev_index=roles["DEV"], y=y, pa=pa.astype(np.float32), pb=pb.astype(np.float32), h=h, z=z,
                        **{f"brier_{k}_{b}": c["per_sample_brier"].astype(np.float32) for k, cv in exact.items() for b, c in cv.items()},
                        **{f"thr_brier_{k}_{b}": c["per_sample_brier"].astype(np.float32) for k, cv in thr.items() for b, c in cv.items()},
                        brier_never=ends["never"][0].astype(np.float32), brier_always=ends["always"][0].astype(np.float32))
    json.dump(res, open(base / "result.json", "w"), indent=1)
    log(f"DEV Brier never {res['dev']['brier_never']:.4f} always {res['dev']['brier_always']:.4f}; mean h {res['dev']['mean_h']:.4f} (h>0 {res['dev']['frac_h_positive']:.3f}); "
        f"exact 25 %: gain_h {exact['gain_h']['0.25']['brier']:.4f} gain_z {exact['gain_z']['0.25']['brier']:.4f} entropy {exact['entropy']['0.25']['brier']:.4f} "
        f"random {exact['random']['0.25']['brier']:.4f} oracle_h {exact['oracle_h']['0.25']['brier']:.4f}; {res['seconds_total']:.0f}s; memory {memory}")
    return 0


def tint(x):
    x = np.asarray(x, float); n = len(x); m = float(x.mean())
    if n < 2:
        return {"mean": m, "sd": None, "ci95": None, "n": n}
    from scipy import stats
    sd = float(x.std(ddof=1)); h = float(stats.t.ppf(0.975, n - 1)) * sd / math.sqrt(n); return {"mean": m, "sd": sd, "ci95": [m - h, m + h], "n": n}


def cost_at(costs, pol, rate, bs="bs256"):
    """Measured ms / sample (mean over the systems' own measurements): U load + cheap branch + policy (gain MLP; entropy / 1 − max p from
    a's probabilities; 0 for never / always / random) + extra path at the realised rate (bs 256: measured sparse gather + V load / normalise
    + V branch + fusion, interpolated over the measured rates; bs 1: rate × (V load / normalise + V branch + fusion))."""
    def m(key):
        return float(np.mean([c[bs].get(key, np.nan) for c in costs]))
    pc = {"gain_z": m("policy_ms"), "gain_h": m("policy_ms"), "entropy": m("entropy_ms"), "maxprob": m("entropy_ms")}.get(pol, 0.0)
    if bs == "bs256":
        xs = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
        extra = float(np.interp(rate, xs, [0.0] + [float(np.mean([c["sparse_extra_ms_per_sample_bs256"][str(x)] for c in costs])) for x in xs[1:]]))
    else:
        extra = rate * (m("v_decode_ms") + m("v_branch_ms") + m("fusion_ms"))
    return m("u_load_ms") + m("u_branch_ms") + pc + extra


def two_way_boot(D, gi, rng, nboot=2000):
    """Primary interval: resample seeds and DEV clusters jointly.  D: (n_seeds, n) per-sample values; gi: cluster index 0..G−1 per sample.
    Statistic = mean over the resampled seeds and the samples of the resampled clusters."""
    ns, G = D.shape[0], int(gi.max()) + 1; S = np.stack([np.bincount(gi, weights=D[k], minlength=G) for k in range(ns)]); N = np.bincount(gi, minlength=G).astype(float)
    st = np.empty(nboot)
    for t in range(nboot):
        si = rng.integers(0, ns, ns); ci = rng.integers(0, G, G); st[t] = S[si][:, ci].sum() / (ns * N[ci].sum())
    return {"mean": float(D.mean()), "ci95": [float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))], "n_seeds": int(ns), "n_groups": int(G)}


def cluster_boot(d, gi, rng, nboot=2000):
    """Secondary: clusters only, conditional on the trained systems (d = per-sample values averaged over the seeds)."""
    G = int(gi.max()) + 1; S = np.bincount(gi, weights=d, minlength=G); N = np.bincount(gi, minlength=G).astype(float)
    ii = rng.integers(0, G, (nboot, G)); st = S[ii].sum(1) / N[ii].sum(1)
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))], "n_groups": int(G)}


def cmd_aggregate(a):
    """Per family over the seeds.  Primary: exact-rate routing on DEV; summary = mean DEV Brier over the four budgets (discrete area under
    the risk–budget curve, "auc"); interval = two-way bootstrap over seeds × DEV clusters (audio clip / actor), B = 2000, 95 % percentile.
    Secondary: per-budget values; seed-paired t (df = seeds − 1); cluster bootstrap conditional on the systems; deployable CAL-threshold
    curves with realised rates and measured costs; diagnostics (h > 0 share, mean h / z, corr(score, h)); always − never."""
    root = OUT / ("smoke" if a.smoke else "") / a.dataset; runs = {}
    for f in FAMILIES:
        for s in SEEDS:
            p = root / f"{f}_s{s}" / "result.json"
            if p.exists():
                runs[(f, s)] = (json.load(open(p)), np.load(root / f"{f}_s{s}" / "per_sample.npz"))
    rng = np.random.default_rng(SEED); all_groups = np.array(json.load(open(roles_path(a.dataset)))["groups"])
    pols = ("random",) + SCORED + ORACLES; keys = ["auc"] + [str(b) for b in BUDGETS]
    res = {"dataset": a.dataset, "systems": sorted(f"{f}_s{s}" for f, s in runs), "missing": [f"{f}_s{s}" for f in FAMILIES for s in SEEDS if (f, s) not in runs],
           "cost_devices": sorted({r[0]["costs"]["gpu"] for r in runs.values()}), "families": {}}
    for f in FAMILIES:
        ss = [s for s in SEEDS if (f, s) in runs]
        if not ss:
            continue
        RR = [runs[(f, s)][0] for s in ss]; PS = [runs[(f, s)][1] for s in ss]; costs = [r["costs"] for r in RR]
        assert all(np.array_equal(ps["dev_index"], PS[0]["dev_index"]) for ps in PS), "DEV indices differ across seeds"
        gi = np.unique(all_groups[PS[0]["dev_index"]], return_inverse=True)[1]

        def per(pol, key, prefix="brier"):
            if key == "auc":
                return np.mean([per(pol, str(b), prefix) for b in BUDGETS], 0)
            return np.stack([ps[f"{prefix}_{pol}_{key}"].astype(np.float64) for ps in PS])

        def three(D):
            return {"two_way": two_way_boot(D, gi, rng), "seed_t": tint(D.mean(1)), "cluster_cond": cluster_boot(D.mean(0), gi, rng)}

        ends = {e: np.stack([ps[f"brier_{e}"].astype(np.float64) for ps in PS]) for e in ("never", "always")}
        fam = {"seeds": ss, "n_dev": int(len(gi)), "n_dev_groups": int(gi.max()) + 1,
               "endpoints": {e: {"brier": tint(ends[e].mean(1)), "acc": tint([r["dev"]["acc_" + e] for r in RR]),
                                 "cost_ms": {bs: cost_at(costs, e, float(e == "always"), bs) for bs in ("bs256", "bs1")}} for e in ends},
               "always_minus_never": three(ends["always"] - ends["never"]),
               "diagnostics": {k: tint([r["dev"][k] for r in RR]) for k in ("frac_h_positive", "mean_h", "mean_z", "mean_h_minus_z", "identity_max_abs_err")},
               "corr_pred_h": {k: tint([r["dev"]["corr_pred_h"][k] for r in RR]) for k in SCORED},
               "exact_rate_brier": {pol: {k: tint(per(pol, k).mean(1)) for k in keys} for pol in pols},
               "exact_rate_acc": {pol: {str(b): tint([r["dev"]["curves_exact"][pol][str(b)]["acc"] for r in RR]) for b in BUDGETS} for pol in pols},
               "exact_rate_cost_ms": {pol: {str(b): {bs: cost_at(costs, pol, n_acquire(b, len(gi)) / len(gi), bs) for bs in ("bs256", "bs1")} for b in BUDGETS} for pol in DEPLOYABLE},
               "cal_threshold": {}, "contrasts": {}}
        for pol in SCORED:
            fam["cal_threshold"][pol] = {}
            for b in BUDGETS:
                rate = tint([r["dev"]["curves_cal_threshold"][pol][str(b)]["rate"] for r in RR])
                fam["cal_threshold"][pol][str(b)] = {"rate": rate, "brier": tint(per(pol, str(b), "thr_brier").mean(1)),
                                                     "acc": tint([r["dev"]["curves_cal_threshold"][pol][str(b)]["acc"] for r in RR]),
                                                     "cost_ms": {bs: cost_at(costs, pol, rate["mean"], bs) for bs in ("bs256", "bs1")}}
        for name, (x, yv) in CONTRASTS.items():
            fam["contrasts"][name] = {k: three(per(x, k) - per(yv, k)) for k in keys}
        res["families"][f] = fam
    res["two_by_two_auc"] = {f"{f}|{pol}": res["families"][f]["exact_rate_brier"][pol]["auc"] for f in res["families"] for pol in ("gain_z", "gain_h")}
    json.dump(res, open(root / "aggregate.json", "w"), indent=1)

    def ci(c):
        return f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]" if c.get("ci95") else "n/a"

    L = [f"# r6 GAIN-P1 — {a.dataset}: DEV results (exact-rate routing primary; Brier, lower is better)", "",
         f"systems {res['systems']}; missing {res['missing'] or 'none'}; cost devices {res['cost_devices']}", ""]
    for f, fam in res["families"].items():
        L += [f"## family {f} (seeds {fam['seeds']}; DEV n = {fam['n_dev']}, clusters = {fam['n_dev_groups']})", ""]
        for e in ("never", "always"):
            ep = fam["endpoints"][e]
            L.append(f"- {e}: Brier {ep['brier']['mean']:.4f} ± {ep['brier']['sd'] or 0:.4f}, acc {ep['acc']['mean']:.4f}; {ep['cost_ms']['bs256']:.4f} ms/sample (bs 256), {ep['cost_ms']['bs1']:.3f} ms (bs 1)")
        am = fam["always_minus_never"]
        L += [f"- always − never: {am['two_way']['mean']:+.4f} two-way {ci(am['two_way'])}; seed t {ci(am['seed_t'])}; clusters | systems {ci(am['cluster_cond'])}",
              f"- diagnostics: h > 0 share {fam['diagnostics']['frac_h_positive']['mean']:.3f}; mean h {fam['diagnostics']['mean_h']['mean']:+.4f}; mean z {fam['diagnostics']['mean_z']['mean']:.4f}; "
              f"identity max err {fam['diagnostics']['identity_max_abs_err']['mean']:.1e}; corr(score, h): " + ", ".join(f"{k} {v['mean']:+.3f}" for k, v in fam["corr_pred_h"].items()), "",
              "| policy (exact rate) | mean over budgets (primary) | 10 % | 25 % | 50 % | 75 % |", "|---|---|---|---|---|---|"]
        for pol in pols:
            L.append(f"| {pol} | " + " | ".join(f"{fam['exact_rate_brier'][pol][k]['mean']:.4f}" for k in keys) + " |")
        L += ["", "| contrast (Brier; < 0: first better) | budget | mean | two-way bootstrap 95 % (seeds × clusters) | seed-paired t 95 % | clusters given systems 95 % |", "|---|---|---|---|---|---|"]
        for name, cc in fam["contrasts"].items():
            for k in keys:
                c = cc[k]
                L.append(f"| {name} | {'mean (primary)' if k == 'auc' else str(int(100 * float(k))) + ' %'} | {c['two_way']['mean']:+.4f} | {ci(c['two_way'])} | {ci(c['seed_t'])} | {ci(c['cluster_cond'])} |")
        L += ["", "Secondary — deployable CAL-threshold routing (realised DEV rates) and measured cost (AV-MNIST: recorded, no cost claims):", "",
              "| policy | budget | realised rate | Brier | acc | ms/sample bs 256 | ms/sample bs 1 |", "|---|---|---|---|---|---|---|"]
        for pol in SCORED:
            for b in BUDGETS:
                c = fam["cal_threshold"][pol][str(b)]
                L.append(f"| {pol} | {int(100 * b)} % | {c['rate']['mean']:.3f} | {c['brier']['mean']:.4f} | {c['acc']['mean']:.4f} | {c['cost_ms']['bs256']:.4f} | {c['cost_ms']['bs1']:.3f} |")
        L.append("")
    (root / "aggregate.md").write_text("\n".join(L)); print("\n".join(L)); return 0


def main() -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("roles", "audit", "run", "aggregate"):
        p = sub.add_parser(name); p.add_argument("--dataset", default="avmnist", choices=["avmnist", "cremad"]); p.add_argument("--smoke", action="store_true")
        if name == "run":
            p.add_argument("--family", required=True, choices=FAMILIES); p.add_argument("--seed", type=int, required=True)
            p.add_argument("--epochs", type=int, default=EPOCHS); p.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    return {"roles": cmd_roles, "audit": cmd_audit, "run": cmd_run, "aggregate": cmd_aggregate}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
