"""V3 — does a conditional-dependence audit improve model selection on a controlled spurious-cue task? (package v4 §V3)

Frozen encoder h; small downstream classifiers only (no pretraining).  A CLASS-SPECIFIC cue is planted on the image before the encoder, colour
palette[k] for cue value k ∈ 0..9:   tag   an 8×8 patch of palette[k] in the top-left corner;   colour   a 25 % global blend towards palette[k].
Base images (FIT split only, fixed permutation, disjoint): POOL-TRAIN 20 000 / CLEAN-VAL 5 000 / AUDIT 5 000 / REVERSAL-TEST 10 000.
  POOL-TRAIN: class c shows its own colour (k = c) with probability ρ ∈ {0.8, 0.95}, otherwise a random other class's colour (fixed per image).
  CLEAN-VAL: clean images (no cue) — the selection signal a practitioner sees without the cue.
  REVERSAL-TEST: every image shows the colour of class (c + 1) mod 10.
  AUDIT: binary N with P(N = 1 | Y) = ½ ± 0.3 by class parity (T1 design); N = 1 → the shifted colour (c + 1), N = 0 → the own colour (c).
Model pool (20 members, fixed): head ∈ {linear, MLP 512→256→10} × weight decay ∈ {0, 5e-4} × cue-randomisation augmentation
a ∈ {0, .25, .5, .75, 1} (each epoch a fraction a of training images shows its random-other colour instead of its drawn one).
Audit statistic of a member (lower = less dependence of its logits on N given Y), fitted on AUDIT-FIT 2 000, read on AUDIT-EVAL 2 000,
within-class negatives from AUDIT-POOL 1 000: the VCS closed-form critic J and the exact JS value (P105 exact solvers, on the class-conditional linear class z ⊗ onehot(Y)),
class-conditional HSIC (per-class bandwidth).
Rules (fixed before any run): A = highest CLEAN-VAL accuracy; B_k = among members within δ = 1.0 point of the best CLEAN-VAL accuracy, the
smallest audit statistic k; C = the expected outcome of a uniformly random eligible member (does the audit add anything beyond eligibility?).
External outcome: REVERSAL-test accuracy and worst-class accuracy of the selected member."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from cond_test_t1 import class_blocks, class_kernels, hsic_class_stat, js_value  # noqa: E402
from cond_test_t1_ablation import exact_js_critic  # noqa: E402
from precheck_d_tests import closed_form_critic, j_stat, within_class_pool  # noqa: E402

from vcs_ssl.data.cifar import load_cifar10_train

from .common import Encoder, cached_h

SPLITS = {"train": 20000, "idval": 5000, "audit": 5000, "test": 10000}   # idval = CLEAN-VAL (no cue)
SPLIT_SEED = 110
POOL = [{"head": h, "wd": wd, "aug": a} for h in ("linear", "mlp") for wd in (0.0, 5e-4) for a in (0.0, 0.25, 0.5, 0.75, 1.0)]
DELTA_PT = 1.0
STATS = ("vcs_closed_J", "js_exact", "hsic_class")


PALETTE = np.array([[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48], [145, 30, 180], [70, 240, 240],
                    [240, 50, 230], [210, 245, 60], [128, 128, 128]], dtype=np.float32)


def cue_tag(imgs: np.ndarray, k: np.ndarray) -> np.ndarray:
    x = imgs.copy(); x[:, 0:8, 0:8, :] = PALETTE[k][:, None, None, :].astype(np.uint8)
    return x


def cue_colour(imgs: np.ndarray, k: np.ndarray) -> np.ndarray:
    x = 0.75 * imgs.astype(np.float32) + 0.25 * PALETTE[k][:, None, None, :]
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


CUES = {"tag": cue_tag, "colour": cue_colour}


def base_splits(fit_uids: np.ndarray) -> dict[str, np.ndarray]:
    perm = np.random.default_rng(SPLIT_SEED).permutation(np.asarray(fit_uids)); out, i = {}, 0
    for k, n in SPLITS.items():
        out[k] = perm[i:i + n]; i += n
    return out


def draw_n(y: np.ndarray, p_even: float, p_odd: float, rng) -> np.ndarray:
    p = np.where(y % 2 == 0, p_even, p_odd)
    return (rng.random(len(y)) < p).astype(np.int64)


class MLP(nn.Module):
    def __init__(self, d, c=10):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, c))

    def forward(self, x):
        return self.net(x)


def train_member(Hc, Hq, y, n, spec, seed, dev, epochs=30, bs=512):
    """Hc / Hq: features of the two versions of each training image (V3: Hc = random-other colour, Hq = own colour); the observed version is Hq
    where n = 1.  Augmentation a: each epoch a fraction a of images shows Hc (the class-uninformative version).  Inputs standardised by training stats."""
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    X0 = torch.where(torch.as_tensor(n == 1)[:, None], Hq, Hc)
    mu, sd = X0.mean(0), X0.std(0) + 1e-6
    m = (nn.Linear(Hc.shape[1], 10) if spec["head"] == "linear" else MLP(Hc.shape[1])).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=spec["wd"])
    Hc_d, Hq_d, y_d, n_d = ((Hc - mu) / sd).to(dev), ((Hq - mu) / sd).to(dev), torch.as_tensor(y).to(dev), torch.as_tensor(n).to(dev)
    N = len(y)
    for _ in range(epochs):
        redraw = torch.rand(N, generator=g) < spec["aug"]; nn_ = torch.where(redraw, torch.zeros(N, dtype=torch.long), torch.as_tensor(n)).to(dev)
        perm = torch.randperm(N, generator=g).to(dev)
        for s in range(0, N, bs):
            i = perm[s:s + bs]; x = torch.where((nn_[i] == 1)[:, None], Hq_d[i], Hc_d[i])
            loss = F.cross_entropy(m(x), y_d[i]); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    m.eval()
    return m, mu, sd


@torch.no_grad()
def logits(m, mu, sd, Hc, Hq, n, dev):
    x = torch.where(torch.as_tensor(n == 1)[:, None], Hq, Hc)
    return m(((x - mu) / sd).to(dev)).cpu()


def class_expand(z: torch.Tensor, y: np.ndarray, n_classes: int = 10) -> torch.Tensor:
    """z ⊗ onehot(y): [n, n_classes · d]; a linear critic on it is a separate linear critic per class."""
    oh = torch.nn.functional.one_hot(torch.as_tensor(y), n_classes).float()
    return (oh[:, :, None] * z[:, None, :]).reshape(len(z), -1)


def audit_stats(L: torch.Tensor, y: np.ndarray, n: np.ndarray, rng, seed: int) -> dict:
    """Conditional-dependence statistics of logits L on N given Y (fit on AUDIT-FIT, read on AUDIT-EVAL)."""
    idx = rng.permutation(len(y)); fi, ei, pi = idx[:2000], idx[2000:4000], idx[4000:5000]
    mu, sd = L[fi].mean(0), L[fi].std(0) + 1e-6
    zf, ze = ((L[fi] - mu) / sd).float(), ((L[ei] - mu) / sd).float()
    # class-conditional linear class for the exact critics: z ⊗ onehot(Y) (the cue moves logits in a class-specific direction, which a
    # class-agnostic sign-flip critic averages away — found in the P110 gate's design check, fixed before the freeze)
    zf_c, ze_c = class_expand(zf, y[fi]), class_expand(ze, y[ei])
    nf, ne = torch.as_tensor(n[fi]), torch.as_tensor(n[ei])
    nf_neg = torch.as_tensor(within_class_pool(y[fi], y[pi], n[pi], rng)); ne_neg = torch.as_tensor(within_class_pool(y[ei], y[pi], n[pi], rng))
    cf = closed_form_critic(zf_c, nf, nf_neg, seed); jx = exact_js_critic(zf_c, nf, nf_neg, seed + 1)
    with torch.no_grad():
        J = j_stat(torch.tanh(cf(ze_c, ne)), torch.tanh(cf(ze_c, ne_neg)))
        JS = js_value(jx(ze_c, ne), jx(ze_c, ne_neg))
    blocks = class_blocks(y[ei]); H = hsic_class_stat(class_kernels(ze, blocks), blocks, ne.float(), len(ei))
    return {"vcs_closed_J": float(J), "js_exact": float(JS), "hsic_class": float(H)}


def select(members: list[dict]) -> dict:
    acc = np.array([m["idval_acc"] for m in members]); best = acc.max(); elig = np.where(acc >= best - DELTA_PT)[0]
    A = int(np.argmax(acc)); out = {"A": {"member": A}, "eligible": elig.tolist()}
    for k in STATS:
        s = np.array([members[i][k] for i in elig]); out[f"B_{k}"] = {"member": int(elig[int(np.argmin(s))])}
    for key in list(out):
        if isinstance(out[key], dict) and "member" in out[key]:
            mm = members[out[key]["member"]]; out[key].update(test_acc=mm["test_acc"], test_worst_group=mm["test_worst_group"], idval_acc=mm["idval_acc"])
    out["C_random_eligible"] = {"test_acc": float(np.mean([members[i]["test_acc"] for i in elig])),
                                "test_worst_group": float(np.mean([members[i]["test_worst_group"] for i in elig]))}
    out["pool_best_test"] = float(max(m["test_acc"] for m in members))
    return out


def run_v3(enc: Encoder, dev: torch.device, *, replicates: int = 5, rhos=(0.8, 0.95), smoke: bool = False) -> dict:
    data = load_cifar10_train(enc.R["cfg"]["data"]["root"]); sp = base_splits(np.asarray(enc.manifest["fit_uids"]))
    if smoke:
        sp = {k: v[: max(600, len(v) // 20)] if k != "audit" else v[:5000] for k, v in sp.items()}
    names = list(SPLITS); allu = np.concatenate([sp[k] for k in names]); bounds = np.cumsum([0] + [len(sp[k]) for k in names])
    pos = {k: slice(int(bounds[i]), int(bounds[i + 1])) for i, k in enumerate(names)}
    yall = data.targets[allu].astype(np.int64)
    # alternative cue value per image: a fixed random OTHER class for POOL-TRAIN / CLEAN-VAL, the shifted class (c + 1) for AUDIT / REVERSAL-TEST
    rng0 = np.random.default_rng(SPLIT_SEED + 1); other = (yall + rng0.integers(1, 10, len(yall))) % 10; shifted = (yall + 1) % 10
    k_alt = other.copy(); k_alt[pos["audit"]] = shifted[pos["audit"]]; k_alt[pos["test"]] = shifted[pos["test"]]
    tag = "smoke_" if smoke else ""
    Hclean = cached_h(enc, f"{tag}v3_clean_h", lambda: data.data[allu])
    out = {"run": enc.run, "splits": {k: int(len(v)) for k, v in sp.items()}, "pool": POOL, "delta_pt": DELTA_PT, "rhos": list(rhos),
           "design": "class-specific cue; CLEAN-VAL without cue; REVERSAL-TEST colour of class c+1; AUDIT N=1 shifted / N=0 own", "families": {}}
    epochs = 3 if smoke else 30
    for fam, fn in CUES.items():
        Hown = cached_h(enc, f"{tag}v3_{fam}_own_h", lambda: fn(data.data[allu], yall))
        Halt = cached_h(enc, f"{tag}v3_{fam}_alt_h", lambda: fn(data.data[allu], k_alt))
        out["families"][fam] = {}
        for rho in rhos:
            reps = []
            for r in range(1 if smoke else replicates):
                t0 = time.time(); rng = np.random.default_rng([SPLIT_SEED, int(rho * 1000), r, len(fam)])
                y = {k: yall[pos[k]] for k in names}
                shows_own = (rng.random(len(y["train"])) < rho).astype(np.int64)
                n_audit = draw_n(y["audit"], 0.8, 0.2, rng)
                tr = (Halt[pos["train"]], Hown[pos["train"]])
                members = []
                for mi, spec in enumerate(POOL):
                    m, mu, sd = train_member(*tr, y["train"], shows_own, spec, seed=1000 * r + mi, dev=dev, epochs=epochs)
                    pv = logits(m, mu, sd, Hclean[pos["idval"]], Hclean[pos["idval"]], np.zeros(len(y["idval"]), dtype=np.int64), dev).argmax(1).numpy()
                    pt = logits(m, mu, sd, Halt[pos["test"]], Halt[pos["test"]], np.ones(len(y["test"]), dtype=np.int64), dev).argmax(1).numpy()
                    corr = pt == y["test"]
                    rec = {**spec, "idval_acc": float((pv == y["idval"]).mean()) * 100, "test_acc": float(corr.mean()) * 100,
                           "test_worst_group": float(min(corr[y["test"] == c].mean() for c in range(10))) * 100,
                           "train_cue_acc": float((logits(m, mu, sd, *tr, shows_own, dev).argmax(1).numpy() == y["train"]).mean()) * 100}
                    La = logits(m, mu, sd, Hown[pos["audit"]], Halt[pos["audit"]], n_audit, dev)
                    rec.update(audit_stats(La, y["audit"], n_audit, np.random.default_rng([r, mi, 7]), seed=r * 100 + mi))
                    members.append(rec)
                sel = select(members); reps.append({"replicate": r, "members": members, "selection": sel, "seconds": time.time() - t0})
                print(f"[{enc.run}] V3 {fam} rho {rho} rep {r}: A test {sel['A']['test_acc']:.2f} | B_vcs {sel['B_vcs_closed_J']['test_acc']:.2f} "
                      f"B_js {sel['B_js_exact']['test_acc']:.2f} B_hsic {sel['B_hsic_class']['test_acc']:.2f} | C {sel['C_random_eligible']['test_acc']:.2f} "
                      f"| best {sel['pool_best_test']:.2f} ({time.time() - t0:.0f}s)", flush=True)
            out["families"][fam][str(rho)] = reps
    return out
