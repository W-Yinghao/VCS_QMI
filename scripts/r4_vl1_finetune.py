"""R4-VL1 — proposal-aware full fine-tuning (reports/R4/R4_VL_TRAINING_PROTOCOL_FROZEN_20261011.yaml; package docs/04 §4–6).

Same recipe as VL3 (AdamW 1e-5 / wd 0.05, 10 epochs, cosine with 200 warm-up steps, 32 images per step, critic a·cos + b init (10, −2.5), both
towers updated), but the candidates are the clean VL1-21 detector's proposals of the image (score ≥ 0.05, top-100, per-class NMS, clipped, < 1 px
dropped; FIT proposals from outputs/VL1_21/proposals/<ds>_FIT.pt) instead of the annotated referred boxes.  Laws per image (proposal-reference
Q_ν, not the original conditional QMI): A(r, w) = 1[IoU(r, box(w)) ≥ 0.5]; π uniform over the image's queries with ≥ 1 valid proposal;
p(r, w) = π(w) A(r, w) / Σ_s A(s, w); q(r, w) = π(w) / n_proposals.  Queries without a valid proposal give no gradient (counted); images without any
valid query are left out of FIT (counted).  Objectives on the same laws: vcs = original J; js = balanced logistic; softmax = multi-positive
set-event CE over the image's proposals (= VL3's conditional softmax when every query has one positive).
Exact chunked accumulation: the batch's images are processed in chunks of ≤ --chunk-crops crops (0 = all at once); loss = mean over the 32 images
in every case (no BatchNorm, so the forward is identical).
Selection: CAL all-proposal image-macro Acc@IoU 0.5 per epoch (same rule for every objective; uncovered queries count as errors).  DEV is read at
step 0 (zero-shot arm + gate: given-box Top-1 within 0.3 of the frozen raw value) and once at the selected epoch.  Also recorded: Acc@0.75,
query-level accuracy, coverage, given-box Top-1, detector-score top-K accuracies (K ∈ {4, 8, 16, 32}; K chosen on CAL), common J on the proposal
laws (J_own; J_recal = affine refit on CAL), margins.  Writes outputs/R4_VL1/<ds>_<backbone>_<objective>_s<seed>[_smoke].json and the CAL-selected
weights to /projects/.../models/r4_vl1/.
    VL_DATASET=refcocog python scripts/r4_vl1_finetune.py --backbone siglip2_b16 --objective vcs --seed 0 [--chunk-crops 1024] [--grad-ckpt] [--smoke]
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
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vl1_22_tableC import TOPK, iou_xyxy  # noqa: E402
from vl3_aggregate import RAW  # noqa: E402
from vl3_finetune import BATCH, CLIP_ARCHS, CRITIC_LR, SIGLIP_REPOS, WARMUP, WD, RG, Backbone, Critic, scores  # noqa: E402
from vcs_vl.pairlaw import padded_batch_loss, padded_j  # noqa: E402

OUT = Path("/home/infres/yinwang/CS_QMI/outputs/R4_VL1"); CKPT = Path("/projects/EEG-foundation-model/yinghao/models/r4_vl1")
PROP = Path("/home/infres/yinwang/CS_QMI/outputs/VL1_21/proposals"); OBJ = {"vcs": "vcs", "js": "balanced_logistic", "softmax": "set_softmax"}
KS, IOU_TRAIN = (4, 8, 16, 32), 0.5


def laws_from_A(A: np.ndarray):
    """p = π(w) A / ΣA, q = π(w) ν(r) (ν uniform over proposals); π uniform over queries with a positive; returns (p, q, n_valid, n_invalid)."""
    A = np.asarray(A, dtype=np.float32); n_p, n_q = A.shape; col = A.sum(0); ok = col > 0; n_valid = int(ok.sum())
    pi = ok.astype(np.float32) / max(1, n_valid)
    p = A / np.maximum(col, 1)[None, :] * pi[None, :]; q = np.full((n_p, n_q), 1.0 / max(1, n_p), np.float32) * pi[None, :]
    return p.astype(np.float32), q.astype(np.float32), n_valid, int(n_q - n_valid)


def xyxy(b):
    return [b[0], b[1], b[0] + b[2], b[1] + b[3]]


def build(scenes, role, props, want, for_training):
    """Eligible images of a role (≥ 2 referred objects, path, ≥ 1 proposal); FIT keeps valid queries only and drops images without any."""
    recs, counts = [], {"images_role": 0, "images_no_proposal_file": 0, "images_no_proposals": 0, "images_no_valid_query": 0, "queries": 0, "queries_valid": 0}
    for s in scenes:
        if role[s.image_id] != want or len(s.referred) < 2 or not s.path:
            continue
        counts["images_role"] += 1
        if s.image_id not in props:
            counts["images_no_proposal_file"] += 1; continue
        P = props[s.image_id][:TOPK].numpy(); boxes = []
        for x1, y1, x2, y2, sc, lb in P.tolist():
            b = RG.clip_box((x1, y1, x2 - x1, y2 - y1), s.width, s.height)
            if b is not None:
                boxes.append(b)
        if not boxes:
            counts["images_no_proposals"] += 1; continue
        B = np.array(boxes, dtype=np.float64); texts, tgt = [], []
        for i, o in enumerate(s.referred):
            for e in o.expressions:
                texts.append(e["raw"]); tgt.append(i)
        refs = np.array([xyxy(o.box_xywh) for o in s.referred], dtype=np.float64)
        iou = np.stack([iou_xyxy(refs[t], B) for t in tgt]).T                          # [n_p, n_q]
        A = (iou >= IOU_TRAIN).astype(np.float32); counts["queries"] += len(texts)
        if for_training:
            keep = A.sum(0) > 0
            if not keep.any():
                counts["images_no_valid_query"] += 1; continue
            texts = [t for t, k in zip(texts, keep) if k]; A = A[:, keep]; tgt = [t for t, k in zip(tgt, keep) if k]
        p, q, nv, _ = laws_from_A(A); counts["queries_valid"] += nv
        recs.append({"image_id": s.image_id, "path": s.path, "boxes": boxes, "ref_boxes_clip": [RG.clip_box(o.box_xywh, s.width, s.height) for o in s.referred],
                     "texts": texts, "target": np.array(tgt, dtype=np.int64), "A": A, "A75": (iou >= 0.75).astype(np.float32) if not for_training else None,
                     "p": p, "q": q, "n_valid": nv})
    return recs, counts


class Train(torch.utils.data.Dataset):
    def __init__(self, recs, bb):
        self.recs, self.bb = recs, bb

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        with Image.open(r["path"]) as im:
            im = im.convert("RGB"); crops = torch.stack([self.bb.preprocess(im.crop(tuple(b))) for b in r["boxes"]])
        return crops, self.bb.tokenize(r["texts"]), torch.from_numpy(r["p"]), torch.from_numpy(r["q"])


def collate(batch):
    crops = torch.cat([b[0] for b in batch]); toks = torch.cat([b[1] for b in batch]); ms = [b[0].shape[0] for b in batch]; ks = [b[1].shape[0] for b in batch]
    n, R, W = len(batch), max(ms), max(ks); P = torch.zeros(n, R, W); Q = torch.zeros(n, R, W)
    for i, b in enumerate(batch):
        P[i, :ms[i], :ks[i]] = b[2]; Q[i, :ms[i], :ks[i]] = b[3]
    return crops, toks, ms, ks, P, Q


class Eval(torch.utils.data.Dataset):
    def __init__(self, recs, bb):
        self.recs, self.bb = recs, bb

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        with Image.open(r["path"]) as im:
            im = im.convert("RGB"); crops = torch.stack([self.bb.preprocess(im.crop(tuple(b))) for b in r["boxes"] + [tuple(b) for b in r["ref_boxes_clip"]]])
        return i, crops, self.bb.tokenize(r["texts"])


def pad_laws(recs):
    """Padded [n, R, W] P / Q / A5 / A75 / query mask / region mask for an evaluation split (P, Q zero on queries without a positive)."""
    n = len(recs); R = max(len(r["boxes"]) for r in recs); W = max(len(r["texts"]) for r in recs)
    d = {k: torch.zeros(n, R, W) for k in ("P", "Q", "A5", "A75")}; d["wmask"] = torch.zeros(n, W, dtype=torch.bool); d["rmask"] = torch.zeros(n, R, dtype=torch.bool)
    for i, r in enumerate(recs):
        m, k = len(r["boxes"]), len(r["texts"]); d["P"][i, :m, :k] = torch.from_numpy(r["p"]); d["Q"][i, :m, :k] = torch.from_numpy(r["q"])
        d["A5"][i, :m, :k] = torch.from_numpy(r["A"]); d["A75"][i, :m, :k] = torch.from_numpy(r["A75"]); d["wmask"][i, :k] = True; d["rmask"][i, :m] = True
    d["has_valid"] = d["P"].sum((1, 2)) > 0
    return d


@torch.no_grad()
def collect(bb, recs, dl, dev):
    """Cosines of every evaluation image: C_prop [n, R, W] (padded) and given-box hits."""
    bb.eval(); n = len(recs); R = max(len(r["boxes"]) for r in recs); W = max(len(r["texts"]) for r in recs); C = torch.zeros(n, R, W); gb = torch.zeros(n)
    for i, crops, toks in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            U = torch.cat([bb.enc_img(crops[k:k + 256].to(dev, non_blocking=True)) for k in range(0, len(crops), 256)]); V = bb.enc_txt(toks.to(dev, non_blocking=True))
        cos = (F.normalize(U.float(), dim=-1) @ F.normalize(V.float(), dim=-1).T).cpu(); m = len(recs[i]["boxes"])
        C[i, :m, :cos.shape[1]] = cos[:m]; gb[i] = float((cos[m:].argmax(0).numpy() == recs[i]["target"]).mean())
    bb.train(); return C, gb


def acc(C, d, A, K=None):
    """Image-macro and query accuracy of the top-1 proposal (restricted to the first K proposals in detector-score order when K is given)."""
    rmask = d["rmask"].clone()
    if K is not None:
        rmask[:, K:] = False
    pred = C.masked_fill(~rmask[:, :, None], -torch.inf).argmax(1)                              # [n, W]
    hit = A.gather(1, pred[:, None, :]).squeeze(1) * d["wmask"]
    return {"image_macro": float((hit.sum(1) / d["wmask"].sum(1)).mean()), "query": float(hit.sum() / d["wmask"].sum())}


def recal(C, d):
    """Affine critic a' cos + b' maximising the mean common J over images with a valid query (Adam, 400 steps)."""
    sel = d["has_valid"]; al = torch.tensor(math.log(math.expm1(10.0)), requires_grad=True); be = torch.tensor(-2.5, requires_grad=True); opt = torch.optim.Adam([al, be], lr=0.05)
    for _ in range(400):
        J = padded_j(F.softplus(al) * C[sel] + be, d["P"][sel], d["Q"][sel]).mean(); opt.zero_grad(); (-J).backward(); opt.step()
    return float(F.softplus(al)), float(be)


def metrics(C, gb, d, a, b, ar=None, br=None):
    out = {"acc50": acc(C, d, d["A5"]), "acc75": acc(C, d, d["A75"]), "given_box_top1_image_macro": float(gb.mean()),
           "coverage_image_macro": float(((d["A5"].sum(1) > 0).float() * d["wmask"]).sum(1).div(d["wmask"].sum(1)).mean()),
           "topK_acc50": {str(K): acc(C, d, d["A5"], K)["image_macro"] for K in KS}, "n_images": int(C.shape[0]), "n_queries": int(d["wmask"].sum())}
    sel = d["has_valid"]; out["J_own"] = float(padded_j(a * C[sel] + b, d["P"][sel], d["Q"][sel]).mean())
    if ar is not None:
        out["J_recal"] = float(padded_j(ar * C[sel] + br, d["P"][sel], d["Q"][sel]).mean())
    valid = d["A5"] > 0; inval = (~valid) & d["rmask"][:, :, None] & d["wmask"][:, None, :]
    bv = C.masked_fill(~valid, -torch.inf).amax(1); bi = C.masked_fill(~inval, -torch.inf).amax(1); ok = torch.isfinite(bv) & torch.isfinite(bi)
    out["margin_mean"] = float((bv - bi)[ok].mean()) if ok.any() else None
    return out


def evaluate(bb, critic, recs, dl, d, dev, ar_br=None):
    C, gb = collect(bb, recs, dl, dev); a, b = float(F.softplus(critic.alpha)), float(critic.b)
    if ar_br is None:
        ar_br = recal(C, d)
    return metrics(C, gb, d, a, b, *ar_br), ar_br


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--backbone", required=True, choices=list(CLIP_ARCHS) + list(SIGLIP_REPOS)); ap.add_argument("--objective", required=True, choices=list(OBJ))
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--chunk-crops", type=int, default=0, help="max crops per forward/backward chunk (0 = whole batch); exact accumulation either way")
    ap.add_argument("--grad-ckpt", action="store_true"); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); dev = torch.device("cuda"); ds = RG.DATASET.replace("+", "plus")
    tag = f"{ds}_{a.backbone}_{a.objective}_s{a.seed}" + ("_smoke" if a.smoke else ""); OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / f"{tag}.json").exists() and not a.smoke:
        print("exists", tag); return 0
    bb = Backbone(a.backbone, a.grad_ckpt).to(dev); critic = Critic().to(dev)
    scenes, _ = RG.load_scenes(); role = RG.dev_roles(scenes); props = {r: torch.load(PROP / f"{ds}_{r}.pt", map_location="cpu", weights_only=False) for r in ("FIT", "CAL", "DEV")}
    fit, c_fit = build(scenes, role, props["FIT"], "FIT", True); cal, c_cal = build(scenes, role, props["CAL"], "CAL", False); dv, c_dev = build(scenes, role, props["DEV"], "DEV", False)
    if a.smoke:
        fit, cal, dv = fit[:64], cal[:48], dv[:48]
    print(f"[{tag}] FIT {len(fit)} images ({c_fit}); CAL {len(cal)}; DEV {len(dv)}; crops/epoch {sum(len(r['boxes']) for r in fit)}", flush=True)
    g = torch.Generator().manual_seed(a.seed)
    dl_fit = torch.utils.data.DataLoader(Train(fit, bb), batch_size=BATCH, shuffle=True, generator=g, num_workers=a.workers, collate_fn=collate, drop_last=True, pin_memory=True, persistent_workers=True)
    mk = lambda r: torch.utils.data.DataLoader(Eval(r, bb), batch_size=1, shuffle=False, num_workers=a.workers, collate_fn=lambda x: x[0])
    dl_cal, dl_dev = mk(cal), mk(dv); d_cal, d_dev = pad_laws(cal), pad_laws(dv)
    decay = [p for n, p in bb.named_parameters() if p.ndim >= 2]; no_decay = [p for n, p in bb.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "lr": a.lr, "weight_decay": WD}, {"params": no_decay, "lr": a.lr, "weight_decay": 0.0}, {"params": list(critic.parameters()), "lr": CRITIC_LR, "weight_decay": 0.0}])
    base = [gp["lr"] for gp in opt.param_groups]; epochs = 1 if a.smoke else a.epochs; total = epochs * len(dl_fit); step = 0
    sched = lambda s: (s + 1) / WARMUP if s < WARMUP else 0.5 * (1 + math.cos(math.pi * (s - WARMUP) / max(1, total - WARMUP)))
    t0 = time.time(); e_cal0, arbr = evaluate(bb, critic, cal, dl_cal, d_cal, dev); e_dev0, _ = evaluate(bb, critic, dv, dl_dev, d_dev, dev, arbr)
    raw = RAW.get((ds, a.backbone)); gate = None if raw is None else abs(e_dev0["given_box_top1_image_macro"] - raw) <= 0.003
    print(f"[{tag}] step 0: DEV given-box {e_dev0['given_box_top1_image_macro']:.4f} vs frozen raw {raw} -> gate {'pass' if gate else 'FAIL' if gate is not None else 'n/a'}; "
          f"DEV acc50 {e_dev0['acc50']['image_macro']:.4f} CAL acc50 {e_cal0['acc50']['image_macro']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    hist = [{"epoch": 0, "step": 0, "cal": e_cal0}]; best = (e_cal0["acc50"]["image_macro"], 0, {k: v.detach().to("cpu", torch.float16).clone() for k, v in bb.state_dict().items()}, (float(F.softplus(critic.alpha)), float(critic.b)), arbr)
    for ep in range(1, epochs + 1):
        run = []; t_ep = time.time()
        for i, (crops, toks, ms, ks, P, Q) in enumerate(dl_fit):
            for gp, b0 in zip(opt.param_groups, base):
                gp["lr"] = b0 * sched(step)
            opt.zero_grad(set_to_none=True); n = len(ms); R, W = P.shape[1], P.shape[2]; P, Q = P.to(dev), Q.to(dev); idx = oi = ot = 0; tot = 0.0
            while idx < n:
                j, c = idx, 0
                while j < n and (j == idx or a.chunk_crops <= 0 or c + ms[j] <= a.chunk_crops):
                    c += ms[j]; j += 1
                kk = sum(ks[idx:j])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    U = bb.enc_img(crops[oi:oi + c].to(dev, non_blocking=True)); V = bb.enc_txt(toks[ot:ot + kk].to(dev, non_blocking=True))
                U = F.normalize(U.float(), dim=-1); V = F.normalize(V.float(), dim=-1)
                f = critic(scores(U, V, ms[idx:j], ks[idx:j], R, W)); loss = padded_batch_loss(f, P[idx:j], Q[idx:j], OBJ[a.objective]) * ((j - idx) / n)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at epoch {ep} iter {i}")
                loss.backward(); tot += float(loss); oi += c; ot += kk; idx = j
            torch.nn.utils.clip_grad_norm_(list(bb.parameters()) + list(critic.parameters()), 1.0); opt.step(); step += 1; run.append(tot)
            if a.smoke and i >= 20:
                break
        e_cal, arbr = evaluate(bb, critic, cal, dl_cal, d_cal, dev); hist.append({"epoch": ep, "step": step, "train_loss": float(np.mean(run)), "epoch_seconds": time.time() - t_ep, "cal": e_cal})
        print(f"[{tag}] epoch {ep}: loss {np.mean(run):.4f} CAL acc50 {e_cal['acc50']['image_macro']:.4f} acc75 {e_cal['acc75']['image_macro']:.4f} given-box {e_cal['given_box_top1_image_macro']:.4f} "
              f"J_own {e_cal['J_own']:.4f} J_recal {e_cal['J_recal']:.4f} a {float(F.softplus(critic.alpha)):.2f} ({time.time() - t_ep:.0f}s epoch, {time.time() - t0:.0f}s total)", flush=True)
        if e_cal["acc50"]["image_macro"] > best[0]:
            best = (e_cal["acc50"]["image_macro"], ep, {k: v.detach().to("cpu", torch.float16).clone() for k, v in bb.state_dict().items()}, (float(F.softplus(critic.alpha)), float(critic.b)), arbr)
    bb.load_state_dict({k: v.float() for k, v in best[2].items()}); critic.alpha.data.fill_(math.log(math.expm1(best[3][0]))); critic.b.data.fill_(best[3][1])
    e_dev, _ = evaluate(bb, critic, dv, dl_dev, d_dev, dev, best[4]); k_cal = max(KS, key=lambda K: hist[best[1]]["cal"]["topK_acc50"][str(K)])
    res = {"dataset": RG.DATASET, "backbone": a.backbone, "objective": a.objective, "loss_impl": OBJ[a.objective], "seed": a.seed, "lr": a.lr, "epochs": epochs, "batch_images": BATCH,
           "chunk_crops": a.chunk_crops, "grad_ckpt": a.grad_ckpt, "iou_train": IOU_TRAIN, "counts": {"fit": c_fit, "cal": c_cal, "dev": c_dev}, "n_fit_images": len(fit),
           "crops_per_epoch": int(sum(len(r["boxes"]) for r in fit)), "steps": step, "seconds": time.time() - t0, "gate_step0_given_box": {"value": e_dev0["given_box_top1_image_macro"], "frozen_raw": raw, "pass": gate},
           "dev_step0_zero_shot": e_dev0, "history": hist, "selected": {"cal_acc50_epoch": best[1], "critic_a_b": best[3], "recal_a_b": best[4], "topK_chosen_on_cal": k_cal},
           "dev_at_selected": e_dev, "dev_topK_at_cal_chosen_K": e_dev["topK_acc50"][str(k_cal)], "smoke": a.smoke}
    json.dump(res, open(OUT / f"{tag}.json", "w"), indent=1)
    if not a.smoke:
        CKPT.mkdir(parents=True, exist_ok=True); torch.save({"backbone": best[2], "epoch": best[1], "critic_a_b": best[3]}, CKPT / f"{tag}_calAcc.pt")
    print(f"[{tag}] selected (CAL acc50) epoch {best[1]}: DEV acc50 macro {e_dev['acc50']['image_macro']:.4f} (query {e_dev['acc50']['query']:.4f}; acc75 {e_dev['acc75']['image_macro']:.4f}; "
          f"given-box {e_dev['given_box_top1_image_macro']:.4f}; top-{k_cal} {res['dev_topK_at_cal_chosen_K']:.4f}; J_recal {e_dev['J_recal']:.4f}); {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
