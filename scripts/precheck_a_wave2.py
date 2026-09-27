"""Second-application pre-check A — wave 2: A-S1 calibration-vs-dependence curve and A-T mismatch detection without a target calibration set.

    python scripts/precheck_a_wave2.py features --out <dir>                                  # indoor -> outdoor shift: identity splits + frozen CLIP features (GPU)
    python scripts/precheck_a_wave2.py run --shift-dirs animal=<P49 dir>,indoor_outdoor=<dir> --out <prefix> [--pairings exact,topic,coarse,random] [--smoke]

Re-uses the frozen P49/P50 protocol unchanged (helpers imported from precheck_a_adapters / precheck_a_features): frozen open CLIP ViT-B/32 towers,
identity-initialised 512x512 linear adapters per tower, methods vcs / infonce / logistic, grid lr {1e-3, 3e-4, 1e-4} x epochs {5, 15, 40} selected on
SRC-CAL by the method's own loss, 3 seeds of the selected configuration, balanced joint/product evaluation pairs on held-out caption 4.
A-S1 — the *pairing difficulty* is the knob that moves the dependence: exact (own caption) / topic (caption of another image with the same
supercategory set, P50 setting 2) / coarse (another image sharing >= 1 supercategory but with a different set) / random (any other image);
two shifts: animal (P49 splits) and indoor -> outdoor (built here).  Per (shift, pairing, method): held-out J (vcs), native ECE ((1+T)/2 for
vcs, sigmoid for logistic), Platt-on-cosine ECE (Platt fitted on SRC-CAL), on SRC-EVAL and TGT-EVAL.
A-T — task: with the adapters trained on the topic pairing, decide "matched" (caption of a same-topic image) vs "mismatched"; hard mismatch =
caption of a coarse partner (shares >= 1 supercategory, different set), easy mismatch = caption of a random image; evaluation only, no
training selection.  Rules fixed on the source: native accept if p_hat >= 0.8 ((1+T)/2 for vcs, sigmoid for logistic); source-Platt-on-cosine
(Platt fitted on SRC-CAL task pairs) accept if >= 0.8, for every method's embedding and for raw CLIP; oracle target Platt (cross-fitted on
the target task pairs; upper reference, not deployable).  Metrics: realized precision among accepted vs nominal 0.8 (primary), acceptance rate,
mean p_hat of accepted, FNR / FPR / balanced error at the rule, AUROC (ranking), ECE.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from precheck_a_adapters import Adapters, brier, ece, heldout_J, load_split, pair_sets, platt, retrieval, topic_partners, train  # noqa: E402
from precheck_a_features import CLIP_CACHE, COCO, ImgDS, sha_ids  # noqa: E402

INDOOR = {"furniture", "appliance", "indoor", "kitchen", "electronic", "food"}
OUTDOOR = {"vehicle", "outdoor", "sports"}
SPLITS = ("SRC-FIT", "SRC-CAL", "SRC-EVAL", "TGT-EVAL")


# ------------------------------------------------------------------------------------------------------------ features: indoor -> outdoor shift
def build_features(a) -> int:
    import open_clip
    from torch.utils.data import DataLoader
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    out = Path(a.out); (out / "features").mkdir(parents=True, exist_ok=True)
    idx = json.load(open(Path(a.index_dir) / "train2017_index.json")); images, caps = idx["images"], idx["captions"]
    src = sorted(int(i) for i, v in images.items() if v["supercats"] and set(v["supercats"]) <= INDOOR and v["n_captions"] >= 5)
    tgt = sorted(int(i) for i, v in images.items() if v["supercats"] and set(v["supercats"]) <= OUTDOOR and v["n_captions"] >= 5)
    n_fit = min(a.n_fit, len(src) - a.n_cal - a.n_eval)
    rng = np.random.default_rng(a.seed); rng.shuffle(src); rng.shuffle(tgt)
    splits = {"SRC-FIT": src[:n_fit], "SRC-CAL": src[n_fit: n_fit + a.n_cal], "SRC-EVAL": src[n_fit + a.n_cal: n_fit + a.n_cal + a.n_eval], "TGT-EVAL": tgt[: a.n_tgt]}
    assert len(set().union(*map(set, splits.values()))) == sum(len(v) for v in splits.values()), "splits overlap"
    rule = ("source = train2017 images whose supercategory set is non-empty and a subset of {furniture, appliance, indoor, kitchen, electronic, food}; "
            "target = non-empty subset of {vehicle, outdoor, sports}; >= 5 captions; images with 'person' or 'animal' or 'accessory' are in neither pool")
    json.dump({"seed": a.seed, "rule": rule, "pool_sizes": {"source": len(src), "target": len(tgt)}, "n_fit": n_fit, "splits": splits,
               "sha256": {k: sha_ids(v) for k, v in splits.items()}}, open(out / "splits.json", "w"))
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", cache_dir=CLIP_CACHE); tok = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval()
    manifest = {"towers": "open_clip ViT-B-32 laion2b_s34b_b79k (see models/open_clip/PROVENANCE_*.json)", "device": str(device), "rule": rule,
                "splits_sha256": {k: sha_ids(v) for k, v in splits.items()}, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": {}}
    with torch.no_grad():
        for name, ids in splits.items():
            files = [str(COCO / "train2017" / images[str(i)]["file"]) for i in ids]
            dl = DataLoader(ImgDS(files, preprocess), batch_size=a.batch, num_workers=a.workers, shuffle=False)
            feats = torch.zeros(len(ids), 512); t0 = time.time()
            for x, ii in dl:
                feats[ii] = model.encode_image(x.to(device)).float().cpu()
            texts = [caps[str(i)][:5] for i in ids]; tf = torch.zeros(len(ids), 5, 512); flat = [t for ts in texts for t in ts]
            for s in range(0, len(flat), 1024):
                tf.view(-1, 512)[s: s + 1024] = model.encode_text(tok(flat[s: s + 1024]).to(device)).float().cpu()
            torch.save({"image_ids": torch.tensor(ids), "img": feats, "txt": tf, "captions": texts}, out / "features" / f"{name}.pt")
            manifest["files"][name] = {"n_images": len(ids), "n_captions": len(flat), "seconds": time.time() - t0}
            print(f"{name}: {len(ids)} images, {len(flat)} captions, {time.time() - t0:.0f}s", flush=True)
    json.dump(manifest, open(out / "manifest.json", "w"), indent=2); print("done ->", out)
    return 0


# ------------------------------------------------------------------------------------------------------------------------ partner samplers
def make_partners(image_ids, index_dir, seed, kind, n_cand=64):
    """Per-image candidate partner lists (training draws one per step) and one fixed evaluation partner (-1 if none), in the format of
    topic_partners.  exact -> (None, None); topic -> same supercategory set (P50 setting 2); coarse -> shares >= 1 supercategory but a
    different set; random -> any other image.  coarse / random keep 64 pre-drawn candidates per image (seeded), the first being the fixed one."""
    if kind == "exact":
        return None, None
    if kind == "topic":
        return topic_partners(image_ids, index_dir, seed)
    idx = json.load(open(Path(index_dir) / "train2017_index.json"))["images"]
    keys = [tuple(idx[str(int(i))]["supercats"]) for i in image_ids]; n = len(keys); rng = np.random.default_rng(seed)
    cands = []
    if kind == "random":
        for pos in range(n):
            c = rng.integers(0, n - 1, size=n_cand); c = np.where(c >= pos, c + 1, c); cands.append(c.tolist())
    elif kind == "coarse":
        kid_of = {k: j for j, k in enumerate(sorted(set(keys)))}; kid = np.asarray([kid_of[k] for k in keys])
        by_cat = {}
        for pos, k in enumerate(keys):
            for s in k:
                by_cat.setdefault(s, []).append(pos)
        by_cat = {s: np.asarray(v) for s, v in by_cat.items()}
        for pos, k in enumerate(keys):
            pool = np.unique(np.concatenate([by_cat[s] for s in k])) if k else np.zeros(0, dtype=int)
            pool = pool[kid[pool] != kid[pos]]
            cands.append(rng.choice(pool, size=n_cand, replace=True).tolist() if len(pool) else [])
    else:
        raise ValueError(kind)
    fixed = torch.tensor([c[0] if c else -1 for c in cands])
    return cands, fixed


# --------------------------------------------------------------------------------------------------------------------------- task helpers
def auroc(score, y):
    o = torch.argsort(score); ranks = torch.empty(len(score)); ranks[o] = torch.arange(1, len(score) + 1, dtype=torch.float)
    n1, n0 = float((y == 1).sum()), float((y == 0).sum())
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1.0))


def task_pairs(m, img, txt, device, pos_fixed, neg_fixed, seed):
    """Matched = caption 4 of the fixed same-topic partner; mismatched = caption 4 of `neg_fixed` (coarse partner) or, when neg_fixed is None,
    of a random other image (fixed derangement).  Images lacking a needed partner are dropped.  Returns score, cosine, label."""
    n = len(img); g = torch.Generator().manual_seed(seed); perm = torch.randperm(n, generator=g); perm = torch.where(perm == torch.arange(n), (perm + 1) % n, perm)
    neg = perm if neg_fixed is None else neg_fixed
    keep = torch.where((pos_fixed >= 0) & (neg >= 0))[0]
    with torch.no_grad():
        u, v = m.enc(img.to(device), txt[:, 4].to(device)); u, v = u.cpu(), v.cpu()
        s_pos, c_pos = m.cpu().score(u[keep], v[pos_fixed[keep]]); s_neg, c_neg = m.score(u[keep], v[neg[keep]]); m.to(device)
    return torch.cat((s_pos, s_neg)), torch.cat((c_pos, c_neg)), torch.cat((torch.ones(len(keep)), torch.zeros(len(keep))))


def rule_metrics(p, y, score, nominal):
    acc = p >= nominal; n_acc = int(acc.sum())
    prec = float(y[acc].mean()) if n_acc else float("nan"); mean_p = float(p[acc].mean()) if n_acc else float("nan")
    fnr = float((p[y == 1] < nominal).float().mean()); fpr = float((p[y == 0] >= nominal).float().mean()); e, mx, _ = ece(p, y)
    return {"accept_rate": n_acc / len(p), "precision": prec, "abs_dev_from_nominal": abs(prec - nominal) if n_acc else float("nan"), "mean_p_accepted": mean_p,
            "FNR": fnr, "FPR": fpr, "balanced_err": 0.5 * (fnr + fpr), "AUROC": auroc(score, y), "ECE": e, "max_dev": mx, "Brier": brier(p, y)}


def oracle_platt(c, y):
    """Cross-fitted target Platt on the cosine: fit on the even-indexed pairs, predict the odd ones, and vice versa (upper reference)."""
    n = len(c); half = n // 2; ia = torch.cat((torch.arange(0, half, 2), torch.arange(half, n, 2))); ib = torch.tensor(sorted(set(range(n)) - set(ia.tolist())))
    p = torch.empty(n); p[ib], _, _ = platt(c[ia], y[ia], c[ib]); p[ia], _, _ = platt(c[ib], y[ib], c[ia]); return p


def task_eval(m, method, splits, partners_topic, partners_coarse, device, nominal):
    """A-T on SRC-EVAL and TGT-EVAL for one trained (topic-pairing) model; source rules fitted on SRC-CAL task pairs."""
    out = {}
    cal_img, cal_txt = splits["SRC-CAL"]
    for neg_kind in ("hard", "easy"):
        neg_cal = partners_coarse["SRC-CAL"][1] if neg_kind == "hard" else None
        cal_s, cal_c, cal_y = task_pairs(m, cal_img, cal_txt, device, partners_topic["SRC-CAL"][1], neg_cal, 21)
        for split in ("SRC-EVAL", "TGT-EVAL"):
            img, txt = splits[split]; neg_ev = partners_coarse[split][1] if neg_kind == "hard" else None
            s_, c_, y_ = task_pairs(m, img, txt, device, partners_topic[split][1], neg_ev, 23)
            rules = {}
            if method == "vcs":
                rules["native_(1+T)/2"] = (1 + s_) / 2
            if method == "logistic":
                rules["native_sigmoid"] = torch.sigmoid(s_)
            rules["cosine+Platt(SRC-CAL)"], _, _ = platt(cal_c, cal_y, c_)
            if method in ("vcs", "logistic"):
                rules["score+Platt(SRC-CAL)"], _, _ = platt(cal_s, cal_y, s_)
            rules["cosine+Platt(TGT oracle, cross-fitted)"] = oracle_platt(c_, y_)
            out[f"{neg_kind}/{split}"] = {"n_pairs": int(len(y_)), "rules": {k: rule_metrics(p, y_, c_, nominal) for k, p in rules.items()}}
    return out


# ------------------------------------------------------------------------------------------------------------------------------------- run
def run(a) -> int:
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    shift_dirs = dict(kv.split("=", 1) for kv in a.shift_dirs.split(","))
    pairings = a.pairings.split(","); methods = a.methods.split(",")
    lrs = [float(x) for x in a.lrs.split(",")]; epochs = [int(x) for x in a.epochs.split(",")]; seeds = [int(x) for x in a.seeds.split(",")]
    if a.smoke:
        lrs, epochs, seeds, pairings = [1e-3], [1], [0], [p for p in ("exact", "topic", "coarse", "random") if p in pairings][:2] or ["exact", "topic"]
    results = {"settings": vars(a), "device": str(device), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "shifts": {}}
    t0 = time.time()
    for shift, fdir in shift_dirs.items():
        fd = Path(fdir); raw = {k: load_split(fd, k) for k in SPLITS}
        if a.smoke:
            raw = {k: (v[0][:2000], v[1][:2000], v[2][:2000]) for k, v in raw.items()}
        splits = {k: (v[0], v[1]) for k, v in raw.items()}
        R = {"features": str(fd), "splits_sha256": json.load(open(fd / "splits.json")).get("sha256"), "n": {k: len(v[0]) for k, v in raw.items()}, "pairings": {}, "task": {}}
        partners_all = {kind: {k: make_partners(raw[k][2], a.index_dir, 20260927, kind) for k in SPLITS} for kind in sorted(set(pairings) | {"topic", "coarse"})}
        for kind, per in partners_all.items():
            if kind != "exact":
                print(f"[{shift}] partners {kind}: " + ", ".join(f"{k} {int((per[k][1] >= 0).sum())}/{len(per[k][1])}" for k in SPLITS), flush=True)
        # raw CLIP reference for the task (identity adapters, no training)
        ref = Adapters("infonce").to(device).eval()
        R["task"]["raw_clip"] = task_eval(ref, "infonce", splits, partners_all["topic"], partners_all["coarse"], device, a.nominal)
        for pairing in pairings:
            partners = partners_all[pairing]; fit_img, fit_txt = splits["SRC-FIT"]; cal_img, cal_txt = splits["SRC-CAL"]
            fit_p, cal_p = partners["SRC-FIT"][0], partners["SRC-CAL"][0]; R["pairings"][pairing] = {}
            for method in methods:
                grid = []
                for lr in lrs:
                    for ep in epochs:
                        m, val = train(method, fit_img, fit_txt, cal_img, cal_txt, lr, ep, 0, device, fit_partners=fit_p, cal_partners=cal_p)
                        grid.append({"lr": lr, "epochs": ep, "cal_loss": val})
                best = min(grid, key=lambda r: r["cal_loss"]); res = {"grid": grid, "selected": best, "seeds": {}}
                for seed in seeds:
                    m, val = train(method, fit_img, fit_txt, cal_img, cal_txt, best["lr"], best["epochs"], seed, device, fit_partners=fit_p, cal_partners=cal_p)
                    r = {"cal_loss": val, "params": {k: float(v) for k, v in m.named_parameters() if v.numel() == 1}}
                    cal_s, cal_c, cal_y, _, _ = pair_sets(m, cal_img, cal_txt, device, 11, partners["SRC-CAL"][1])
                    for split in ("SRC-EVAL", "TGT-EVAL"):
                        img, txt = splits[split]; s_, c_, y_, u, v = pair_sets(m, img, txt, device, 13, partners[split][1])
                        probs = {}
                        if method == "vcs":
                            probs["native"] = (1 + s_) / 2
                        if method == "logistic":
                            probs["native"] = torch.sigmoid(s_)
                        probs["cosine+Platt(CAL)"], _, _ = platt(cal_c, cal_y, c_)
                        row = {"R1_R5": retrieval(u, v), "calibration": {}}
                        for name, p in probs.items():
                            e, mx, _ = ece(p, y_); row["calibration"][name] = {"ECE": e, "max_dev": mx, "Brier": brier(p, y_)}
                        if method == "vcs":
                            row["heldout_J"], row["saturation"] = heldout_J(m, img, txt, device, 17, fixed_partner=partners[split][1])
                        r[split] = row
                    if pairing == "topic":
                        r["task"] = task_eval(m, method, splits, partners_all["topic"], partners_all["coarse"], device, a.nominal)
                    res["seeds"][str(seed)] = r
                R["pairings"][pairing][method] = res
                msg = " | ".join(f"{sp}: " + ", ".join(f"{k} {np.mean([rr[sp]['calibration'][k]['ECE'] for rr in res['seeds'].values()]):.3f}" for k in next(iter(res["seeds"].values()))[sp]["calibration"]) for sp in ("SRC-EVAL", "TGT-EVAL"))
                J = f" | J {np.mean([rr['SRC-EVAL']['heldout_J'] for rr in res['seeds'].values()]):.3f}/{np.mean([rr['TGT-EVAL']['heldout_J'] for rr in res['seeds'].values()]):.3f}" if method == "vcs" else ""
                print(f"[{shift}/{pairing}/{method}] sel lr={best['lr']:g} ep={best['epochs']}: ECE {msg}{J} ({time.time() - t0:.0f}s)", flush=True)
        results["shifts"][shift] = R
    json.dump(results, open(a.out + ".json", "w"), indent=1, default=float)
    write_markdown(results, a); print("->", a.out + ".md")
    return 0


def write_markdown(results, a):
    mean = lambda res, f: float(np.mean([f(r) for r in res["seeds"].values()]))
    L = [f"# Pre-check A wave 2 — calibration vs dependence (A-S1) and mismatch detection without a target calibration set (A-T) — {results['utc']}", "",
         "Frozen CLIP ViT-B/32, identity 512x512 adapters, grid lr x epochs selected on SRC-CAL by the method's own loss, "
         f"{len(a.seeds.split(','))} seeds (1 in smoke).  ECE: 15 equal-mass bins on balanced joint/product pairs (held-out caption 4).  Mean over seeds.", ""]
    L += ["## A-S1 — per (shift, pairing, method)", "", "| shift | pairing | method | selected | split | J (vcs) | native ECE | cosine+Platt(CAL) ECE | native − Platt | R@1 |", "|---|---|---|---|---|---|---|---|---|---|"]
    curve = []
    for shift, R in results["shifts"].items():
        for pairing, per in R["pairings"].items():
            for method, res in per.items():
                for split in ("SRC-EVAL", "TGT-EVAL"):
                    J = mean(res, lambda r: r[split]["heldout_J"]) if method == "vcs" else float("nan")
                    nat = mean(res, lambda r: r[split]["calibration"]["native"]["ECE"]) if "native" in next(iter(res["seeds"].values()))[split]["calibration"] else float("nan")
                    pl = mean(res, lambda r: r[split]["calibration"]["cosine+Platt(CAL)"]["ECE"]); r1 = mean(res, lambda r: r[split]["R1_R5"][0])
                    L.append(f"| {shift} | {pairing} | {method} | {res['selected']['lr']:g}, {res['selected']['epochs']} | {split} | {J:.3f} | {nat:.4f} | {pl:.4f} | {nat - pl:+.4f} | {r1:.3f} |")
                    if method == "vcs":
                        curve.append((J, shift, pairing, split, nat, pl, {mm: mean(per[mm], lambda r: r[split]["calibration"]["cosine+Platt(CAL)"]["ECE"]) for mm in per if mm != "vcs"}))
    L += ["", "## A-S1 — curve: VCS native − Platt gap vs held-out J (sorted by J; competitors' Platt-on-cosine ECE alongside)", "",
          "| J | shift | pairing | split | VCS native ECE | VCS cosine+Platt | gap | " + " | ".join(f"{mm} cosine+Platt" for mm in (curve[0][6] if curve else {})) + " |",
          "|---|---|---|---|---|---|---|" + "---|" * (len(curve[0][6]) if curve else 0)]
    for J, shift, pairing, split, nat, pl, others in sorted(curve, key=lambda t: t[0]):
        L.append(f"| {J:.3f} | {shift} | {pairing} | {split} | {nat:.4f} | {pl:.4f} | {nat - pl:+.4f} | " + " | ".join(f"{v:.4f}" for v in others.values()) + " |")
    L += ["", f"## A-T — mismatch detection (adapters trained on the topic pairing; nominal precision {a.nominal}; hard = coarse-topic caption, easy = random caption)", "",
          "| shift | neg | split | method | rule | accept | precision | \\|prec − nominal\\| | mean p̂ acc. | FNR | FPR | bal.err | AUROC | ECE |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for shift, R in results["shifts"].items():
        rows = {"raw_clip": {"seeds": {"0": {"task": R["task"]["raw_clip"]}}}}
        rows.update({m: res for m, res in R["pairings"].get("topic", {}).items()})
        for method, res in rows.items():
            first = next(iter(res["seeds"].values())).get("task")
            if not first:
                continue
            for key in first:
                neg, split = key.split("/")
                for rule in first[key]["rules"]:
                    v = {f: float(np.mean([r["task"][key]["rules"][rule][f] for r in res["seeds"].values()])) for f in first[key]["rules"][rule]}
                    L.append(f"| {shift} | {neg} | {split} | {method} | {rule} | {v['accept_rate']:.3f} | {v['precision']:.3f} | {v['abs_dev_from_nominal']:.3f} | {v['mean_p_accepted']:.3f} | {v['FNR']:.3f} | {v['FPR']:.3f} | {v['balanced_err']:.3f} | {v['AUROC']:.3f} | {v['ECE']:.4f} |")
    Path(a.out + ".md").write_text("\n".join(L) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features"); f.add_argument("--out", required=True); f.add_argument("--index-dir", default="/home/infres/yinwang/CS_QMI/data/coco_index")
    f.add_argument("--n-fit", type=int, default=20000); f.add_argument("--n-cal", type=int, default=5000); f.add_argument("--n-eval", type=int, default=5000); f.add_argument("--n-tgt", type=int, default=5000)
    f.add_argument("--seed", type=int, default=20260927); f.add_argument("--workers", type=int, default=8); f.add_argument("--batch", type=int, default=256)
    r = sub.add_parser("run"); r.add_argument("--shift-dirs", required=True, help="name=dir[,name=dir]"); r.add_argument("--out", required=True)
    r.add_argument("--pairings", default="exact,topic,coarse,random"); r.add_argument("--methods", default="vcs,infonce,logistic")
    r.add_argument("--lrs", default="1e-3,3e-4,1e-4"); r.add_argument("--epochs", default="5,15,40"); r.add_argument("--seeds", default="0,1,2")
    r.add_argument("--nominal", type=float, default=0.8); r.add_argument("--index-dir", default="/home/infres/yinwang/CS_QMI/data/coco_index"); r.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    return build_features(a) if a.cmd == "features" else run(a)


if __name__ == "__main__":
    sys.exit(main())
