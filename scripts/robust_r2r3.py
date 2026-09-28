"""Next round, R line — R2 (training-level noisy pairing) and R3 (mismatch identification) on the frozen-CLIP + identity-adapter setting of
pre-check A (topic pairing = P50 setting 2 / P57), one run per shift set.

    python scripts/robust_r2r3.py --shift-dirs animal=<P49 dir>,indoor_outdoor=<P57 dir> --out <prefix> [--mismatch 0,0.1,0.2,0.4,0.6] [--smoke]

Mismatch injection (R2): for a seeded fraction m of SRC-FIT images the training positive caption comes from a *random other* image (not
topic-matched); the injected indices and their wrong partners are saved.  SRC-CAL (selection) and the evaluation splits stay clean.
Methods, equal budget (P49 grid lr x epochs selected on clean SRC-CAL by each method's own loss; 3 seeds of the selection):
  vcs       J with tanh critic, K pool negatives (as P50);
  infonce   in-batch InfoNCE;
  logistic  pairwise logistic on the in-batch pairs (native; the "+log K intercept" variant is a rank-equivalent score correction and is
            reported as a column of the same model, since R2/R3 are rank-based);
  js        balanced logistic in the Deep-InfoMax form  E_P[softplus(-f)] + E_Q[softplus(f)]  on the SAME positive / K-negative construction
            as vcs (f = a*cos + b), implemented here without touching precheck_a_adapters.py.
R2 metric: image->text R@1 / R@5 on the clean SRC-EVAL and TGT-EVAL (own caption 4) vs m; degradation = R@1(m) - R@1(0).
R3: at m in --r3-mismatch, rank the SRC-FIT training pairs (fixed pairing: injected partner or the fixed topic partner, caption 0) by each
method's native score with no clean calibration; AUROC of "clean" vs "injected"; raw CLIP cosine as reference.

--pairing exact (P79, repairs P72 §1): the clean training positive of image i is its OWN caption (captions 0-3), SRC-CAL selection uses exact
pairs, injected images get one random other image's captions as before; R3 scores the pairing (own image, or injected partner), caption 0.
A raw-CLIP R2 reference row (identity adapters, no training) is added.  --crossfit (exact only): R3 additionally reports a 2-fold
cross-fitted AUROC — SRC-FIT split once into two identity-disjoint halves (seeded), the selected configuration is trained on one half
(with that half's injected pairs; K-pool negatives drawn from that half's captions) and scores the other half's pairs; halves swapped;
AUROC over the pooled out-of-fold scores.  --pairing topic (default) is unchanged.
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
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from precheck_a_adapters import Adapters, infonce_loss, load_split, logistic_loss, pair_sets, retrieval, topic_partners, vcs_loss  # noqa: E402
from precheck_a_wave2 import auroc  # noqa: E402

SPLITS = ("SRC-FIT", "SRC-CAL", "SRC-EVAL", "TGT-EVAL")
METHODS = ("vcs", "infonce", "logistic", "js")


def js_loss(m, u, v, v_pool):
    """Balanced logistic (Deep InfoMax JS form) on the vcs positive / K-negative construction; f = a*cos + b."""
    fp = m.a * (u * v).sum(-1) + m.b
    fn = m.a * (u[:, None, :] * v_pool).sum(-1) + m.b
    loss = F.softplus(-fp).mean() + F.softplus(fn).mean()
    js = math.log(4) - float(loss)  # JS-divergence estimate (nats), bounded by log 2
    return loss, {"JS": js}


def train_r(method, fit_img, fit_txt, cal_img, cal_txt, lr, epochs, seed, device, K=8, batch=256, wd=1e-4, fit_partners=None, cal_partners=None):
    """precheck_a_adapters.train with the js method added (identical loop, generator use and selection loss otherwise)."""
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    m = Adapters(method).to(device); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    n = len(fit_img); steps = epochs * (n // batch); step = 0
    fit_img, fit_txt = fit_img.to(device), fit_txt.to(device)

    def partner(ix_cpu, cands):
        return torch.tensor([c[int(torch.randint(0, len(c), (1,), generator=g))] if c else int(i) for i, c in zip(ix_cpu.tolist(), (cands[i] for i in ix_cpu.tolist()))])

    done = False
    while not done:
        perm = torch.randperm(n, generator=g)
        for s in range(0, n - batch + 1, batch):
            if step >= steps:
                done = True; break
            ix_cpu = perm[s: s + batch]; ix = ix_cpu.to(device); ci = torch.randint(0, 4, (batch,), generator=g).to(device)
            src = ix if fit_partners is None else partner(ix_cpu, fit_partners).to(device)
            xi, xt = fit_img[ix], fit_txt[src, ci]
            for pg in opt.param_groups:
                pg["lr"] = lr * 0.5 * (1 + math.cos(math.pi * step / max(steps, 1)))
            opt.zero_grad(); u, v = m.enc(xi, xt)
            if method in ("vcs", "js"):
                pool = torch.randint(0, n, (batch, K), generator=g).to(device); pc = torch.randint(0, 4, (batch, K), generator=g).to(device)
                v_pool = F.normalize(m.txt(fit_txt[pool, pc]), dim=-1)
                loss, _ = vcs_loss(m, u, v, v_pool) if method == "vcs" else js_loss(m, u, v, v_pool)
            elif method == "infonce":
                loss, _ = infonce_loss(m, u, v)
            else:
                loss, _ = logistic_loss(m, u, v)
            loss.backward(); opt.step(); step += 1
        if step >= steps:
            done = True
    m.eval()
    with torch.no_grad():
        ci = torch.randint(0, 4, (len(cal_img),), generator=g); csrc = torch.arange(len(cal_img)) if cal_partners is None else partner(torch.arange(len(cal_img)), cal_partners)
        u, v = m.enc(cal_img.to(device), cal_txt[csrc, ci].to(device))
        if method in ("vcs", "js"):
            pool = torch.randint(0, len(cal_img), (len(cal_img), K), generator=g); pc = torch.randint(0, 4, (len(cal_img), K), generator=g)
            v_pool = F.normalize(m.txt(cal_txt[pool, pc].to(device)), dim=-1)
            val, _ = vcs_loss(m, u, v, v_pool) if method == "vcs" else js_loss(m, u, v, v_pool)
        else:
            fn = infonce_loss if method == "infonce" else logistic_loss
            val = torch.tensor(np.mean([float(fn(m, u[s: s + 256], v[s: s + 256])[0]) for s in range(0, len(u) - 255, 256)]))
    return m, float(val)


def inject(n, m, topic_cands, seed):
    """Seeded mismatch injection: a fraction m of fit images gets a single random *other* image as its only 'positive' partner."""
    rng = np.random.default_rng(seed); k = int(round(m * n)); inj = np.sort(rng.choice(n, size=k, replace=False)) if k else np.array([], dtype=int)
    wrong = {}
    for i in inj.tolist():
        j = int(rng.integers(0, n - 1)); j = j + 1 if j >= i else j
        wrong[i] = j
    cands = [([wrong[i]] if i in wrong else topic_cands[i]) for i in range(n)]
    return cands, inj, wrong


def native_score(m, u, v):
    return m.score(u, v)[0]


def r3_scores(m, method, fit_img, fit_txt, fixed_topic, wrong, device, raw=False):
    """Scores of the fixed training pairing (injected partner, else the fixed topic partner, else the image itself; caption 0)."""
    n = len(fit_img); part = torch.tensor([wrong.get(i, int(fixed_topic[i]) if int(fixed_topic[i]) >= 0 else i) for i in range(n)])
    with torch.no_grad():
        if raw:
            u, v = F.normalize(fit_img, dim=-1), F.normalize(fit_txt[part, 0], dim=-1); return (u * v).sum(-1)
        u, v = m.enc(fit_img.to(device), fit_txt[part, 0].to(device))
        return native_score(m, u, v).cpu()


def crossfit_scores(method, fit_img, fit_txt, cal_img, cal_txt, lr, ep, seed, device, K, wrong, fold_seed):
    """2-fold cross-fitted native scores of the exact-pairing training pairs (own caption, or the injected partner), caption 0."""
    n = len(fit_img); perm = torch.randperm(n, generator=torch.Generator().manual_seed(fold_seed)); halves = (perm[: n // 2], perm[n // 2:])
    part = torch.tensor([wrong.get(i, i) for i in range(n)]); out = torch.empty(n)
    for tr, te in (halves, halves[::-1]):
        tr_l = tr.tolist(); pos = {i: k for k, i in enumerate(tr_l)}
        extra = sorted({wrong[i] for i in tr_l if i in wrong and wrong[i] not in pos}); epos = {j: len(tr_l) + k for k, j in enumerate(extra)}
        sub_img = fit_img[tr]; sub_txt = torch.cat([fit_txt[tr], fit_txt[torch.tensor(extra, dtype=torch.long)]]) if extra else fit_txt[tr]
        cands = [[pos.get(wrong[i], epos.get(wrong[i]))] if i in wrong else [pos[i]] for i in tr_l]
        # train_r draws K-pool negatives from indices < len(sub_img): the fold's own captions only
        m, _ = train_r(method, sub_img, sub_txt, cal_img, cal_txt, lr, ep, seed, device, K=K, fit_partners=cands, cal_partners=None)
        with torch.no_grad():
            u, v = m.enc(fit_img[te].to(device), fit_txt[part[te], 0].to(device)); out[te] = native_score(m, u, v).cpu()
    return out


def run(a) -> int:
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    shift_dirs = dict(kv.split("=", 1) for kv in a.shift_dirs.split(","))
    lrs = [float(x) for x in a.lrs.split(",")]; epochs = [int(x) for x in a.epochs.split(",")]; seeds = [int(x) for x in a.seeds.split(",")]
    mism = [float(x) for x in a.mismatch.split(",")]; r3m = [float(x) for x in a.r3_mismatch.split(",")]; methods = [x for x in a.methods.split(",") if x in METHODS]
    if a.crossfit and a.pairing != "exact":
        raise SystemExit("--crossfit is implemented for --pairing exact only")
    if a.smoke:
        lrs, epochs, seeds, mism, r3m = [1e-3], [1], [0], [0.0, 0.4], [0.4]
    R = {"settings": {**vars(a), "device": str(device), "lrs": lrs, "epochs": epochs, "seeds": seeds, "mismatch": mism, "r3_mismatch": r3m, "methods": methods, "K": a.K},
         "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "shifts": {}}
    t0 = time.time()
    for shift, fdir in shift_dirs.items():
        fd = Path(fdir); raw = {k: load_split(fd, k) for k in SPLITS}
        if a.smoke:
            raw = {k: (v[0][:2000], v[1][:2000], v[2][:2000]) for k, v in raw.items()}
        fit_img, fit_txt, fit_ids = raw["SRC-FIT"]; cal_img, cal_txt, cal_ids = raw["SRC-CAL"]
        topic_fit, fixed_fit = topic_partners(fit_ids, a.index_dir, 20260927); topic_cal, _ = topic_partners(cal_ids, a.index_dir, 20260927)
        if a.pairing == "exact":  # own caption as the clean positive; exact pairs on CAL; R3 fixed pairing = the image itself
            topic_fit, fixed_fit, topic_cal = [[i] for i in range(len(fit_img))], torch.arange(len(fit_img)), None
        shash = int(hashlib.sha256(shift.encode()).hexdigest()[:8], 16)
        S = {"n_fit": int(len(fit_img)), "fit_ids_sha256": hashlib.sha256(",".join(str(int(i)) for i in fit_ids).encode()).hexdigest(), "levels": {}}
        if a.pairing == "exact":  # raw CLIP R2 reference (identity adapters, no training)
            S["raw_clip"] = {}
            for split in ("SRC-EVAL", "TGT-EVAL"):
                img, txt, _ = raw[split]; r1, r5 = retrieval(F.normalize(img, dim=-1), F.normalize(txt[:, 4], dim=-1)); S["raw_clip"][split] = {"R1": r1, "R5": r5}
        # raw CLIP reference for R3 (independent of m: only the injected labels change)
        for m_frac in mism:
            cands, inj, wrong = inject(len(fit_img), m_frac, topic_fit, [20260928, int(round(m_frac * 1000)), shash])
            lab = torch.zeros(len(fit_img)); lab[torch.as_tensor(inj, dtype=torch.long)] = 1.0 if len(inj) else 0.0
            L = {"m": m_frac, "n_injected": int(len(inj)), "injected_sha256": hashlib.sha256(",".join(map(str, inj.tolist())).encode()).hexdigest(),
                 "injected": inj.tolist(), "wrong_partner": {str(k): v for k, v in wrong.items()}, "methods": {}}
            if m_frac in r3m and len(inj):
                sc = r3_scores(None, None, fit_img, fit_txt, fixed_fit, wrong, device, raw=True)
                L["r3_raw_clip_auroc"] = auroc(sc, 1 - lab)
            for method in methods:
                grid = []
                for lr in lrs:
                    for ep in epochs:
                        _, val = train_r(method, fit_img, fit_txt, cal_img, cal_txt, lr, ep, 0, device, K=a.K, fit_partners=cands, cal_partners=topic_cal)
                        grid.append({"lr": lr, "epochs": ep, "cal_loss": val})
                best = min(grid, key=lambda r: r["cal_loss"]); P = {"grid": grid, "selected": {"lr": best["lr"], "epochs": best["epochs"]}, "seeds": {}}
                for seed in seeds:
                    m, val = train_r(method, fit_img, fit_txt, cal_img, cal_txt, best["lr"], best["epochs"], seed, device, K=a.K, fit_partners=cands, cal_partners=topic_cal)
                    r = {"cal_loss": val}
                    for split in ("SRC-EVAL", "TGT-EVAL"):
                        img, txt, _ = raw[split]; _, _, _, u, v = pair_sets(m, img, txt, device, 13, None); r1, r5 = retrieval(u, v)
                        r[split] = {"R1": r1, "R5": r5}
                    if m_frac in r3m and len(inj):
                        sc = r3_scores(m, method, fit_img, fit_txt, fixed_fit, wrong, device)
                        r["r3_auroc"] = auroc(sc, 1 - lab)
                        if method == "logistic":  # +log K intercept correction: rank-equivalent, reported for completeness
                            r["r3_auroc_logk"] = auroc(sc + math.log(255), 1 - lab)
                        if a.crossfit:
                            cs = crossfit_scores(method, fit_img, fit_txt, cal_img, cal_txt, best["lr"], best["epochs"], seed, device, a.K, wrong, 20260929 + shash)
                            r["r3_auroc_crossfit"] = auroc(cs, 1 - lab)
                    P["seeds"][str(seed)] = r
                L["methods"][method] = P
                mean = lambda sp, f: float(np.mean([P["seeds"][s][sp][f] for s in P["seeds"]]))
                print(f"[{shift}] m={m_frac:g} {method} lr={best['lr']:g} ep={best['epochs']}: SRC R@1 {mean('SRC-EVAL', 'R1'):.3f} TGT R@1 {mean('TGT-EVAL', 'R1'):.3f}"
                      + (f" | R3 AUROC {np.mean([P['seeds'][s]['r3_auroc'] for s in P['seeds']]):.3f}" if m_frac in r3m and len(inj) else "")
                      + (f" cross-fit {np.mean([P['seeds'][s]['r3_auroc_crossfit'] for s in P['seeds']]):.3f}" if a.crossfit and m_frac in r3m and len(inj) else "") + f" ({time.time() - t0:.0f}s)", flush=True)
            S["levels"][f"{m_frac:g}"] = L
        R["shifts"][shift] = S
    Path(a.out + ".json").parent.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(a.out + ".json", "w"), indent=1, default=float)
    write_markdown(R, a); print("->", a.out + ".md")
    return 0


def write_markdown(R, a):
    st = R["settings"]; L = [f"# R2 / R3 — noisy pairing (R2) and mismatch identification (R3) on frozen CLIP + identity adapters, {st.get('pairing', 'topic')} pairing — {R['utc']}",
                            f"Mismatch fractions m = {st['mismatch']}; grid lr {st['lrs']} x epochs {st['epochs']} selected on clean SRC-CAL by each method's own loss; seeds {st['seeds']}; K = {st['K']}. "
                            "R@1 / R@5 = image->text retrieval of the own caption on the clean evaluation splits (mean ± sd over seeds).", ""]
    for shift, S in R["shifts"].items():
        L += [f"## {shift} (SRC-FIT n = {S['n_fit']})", "", "### R2 — retrieval vs mismatch fraction", "",
              "| m | method | selected lr / ep | SRC-EVAL R@1 | Δ vs m=0 | SRC-EVAL R@5 | TGT-EVAL R@1 | Δ vs m=0 | TGT-EVAL R@5 |", "|---|---|---|---|---|---|---|---|---|"]
        base = {}
        if "raw_clip" in S:
            rc = S["raw_clip"]; L.append(f"| – | raw CLIP (no training) | – | {rc['SRC-EVAL']['R1']:.3f} | – | {rc['SRC-EVAL']['R5']:.3f} | {rc['TGT-EVAL']['R1']:.3f} | – | {rc['TGT-EVAL']['R5']:.3f} |")
        for mk, Lv in S["levels"].items():
            for method, P in Lv["methods"].items():
                def ms(sp, f):
                    xs = [P["seeds"][s][sp][f] for s in P["seeds"]]; return float(np.mean(xs)), (float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0)
                r1s, r5s, r1t, r5t = ms("SRC-EVAL", "R1"), ms("SRC-EVAL", "R5"), ms("TGT-EVAL", "R1"), ms("TGT-EVAL", "R5")
                if float(mk) == 0:
                    base[method] = (r1s[0], r1t[0])
                ds = r1s[0] - base.get(method, (r1s[0], r1t[0]))[0]; dt = r1t[0] - base.get(method, (r1s[0], r1t[0]))[1]
                L.append(f"| {mk} | {method} | {P['selected']['lr']:g} / {P['selected']['epochs']} | {r1s[0]:.3f} ± {r1s[1]:.3f} | {ds:+.3f} | {r5s[0]:.3f} | {r1t[0]:.3f} ± {r1t[1]:.3f} | {dt:+.3f} | {r5t[0]:.3f} |")
        L += ["", "### R3 — AUROC of the native training-pair score for clean vs injected pairs (no clean calibration)", "", "| m | method | AUROC (mean ± sd over seeds) |", "|---|---|---|"]
        for mk, Lv in S["levels"].items():
            if "r3_raw_clip_auroc" not in Lv:
                continue
            L.append(f"| {mk} | raw CLIP cosine | {Lv['r3_raw_clip_auroc']:.3f} |")
            for method, P in Lv["methods"].items():
                xs = [P["seeds"][s]["r3_auroc"] for s in P["seeds"] if "r3_auroc" in P["seeds"][s]]
                if xs:
                    L.append(f"| {mk} | {method} | {np.mean(xs):.3f} ± {(np.std(xs, ddof=1) if len(xs) > 1 else 0):.3f} |")
                xs3 = [P["seeds"][s]["r3_auroc_crossfit"] for s in P["seeds"] if "r3_auroc_crossfit" in P["seeds"][s]]
                if xs3:
                    L.append(f"| {mk} | {method} (2-fold cross-fitted) | {np.mean(xs3):.3f} ± {(np.std(xs3, ddof=1) if len(xs3) > 1 else 0):.3f} |")
                if method == "logistic":
                    xs2 = [P["seeds"][s]["r3_auroc_logk"] for s in P["seeds"] if "r3_auroc_logk" in P["seeds"][s]]
                    if xs2:
                        L.append(f"| {mk} | logistic (+log K intercept, rank-equivalent) | {np.mean(xs2):.3f} |")
        L.append("")
    Path(a.out + ".md").write_text("\n".join(L) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shift-dirs", required=True, help="name=dir[,name=dir]"); ap.add_argument("--out", required=True)
    ap.add_argument("--mismatch", default="0,0.1,0.2,0.4,0.6"); ap.add_argument("--r3-mismatch", default="0.2,0.4")
    ap.add_argument("--methods", default="vcs,infonce,logistic,js"); ap.add_argument("--lrs", default="1e-3,3e-4,1e-4"); ap.add_argument("--epochs", default="5,15,40")
    ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--pairing", choices=("topic", "exact"), default="topic"); ap.add_argument("--crossfit", action="store_true")
    ap.add_argument("--index-dir", default="/home/infres/yinwang/CS_QMI/data/coco_index"); ap.add_argument("--smoke", action="store_true")
    return run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
