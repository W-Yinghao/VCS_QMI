"""V8 trainer: the eight utility-aligned recipes on the shared BLIP-2 / Q-former path (package docs/04).

Model: V7's frozen construction (scripts/v7/v7_train.build_model: FGA_Blip2 'coco' config, public BLIP-2 COCO weights, EVA ViT-g frozen and in
eval mode); trained: ln_vision, the Q-former and its query tokens (V7's trained set) + a new 5-logit head on the mean of the 32 query outputs;
V7's ITM / mask / projection heads are unused and frozen.
Head init, the same for every arm at a seed: weight ~ N(0, 0.02) from a seeded generator, bias = log of the FIT natural vote histogram.
Batches: the shared grouped plan of v8_core.batch_manifest (identical across arms at a seed; every FIT image once per epoch).  Optimiser: AdamW
(0.9, 0.999), weight decay 0.05 on matrices (none on biases / norms / query tokens), warm-up 100 steps from 1e-8 then cosine to 0 over the actual
total steps; AMP fp16 for the multimodal forward, all heads / losses in fp32.  Epoch selection: CAL_SELECT prompt-macro mean-rating MSE
(ties -> earlier epoch).  Inputs: full-frame resize 364 (BLIP eval processor), the raw prompt (uncased BERT, <= 128 tokens, never truncated).
    train  --arm U-CS --lr 1e-5 --seed 0 --unit-dir D [--max-steps N --eval-n N]      (the last two: smoke only)
    export --unit-dir D [--roles CAL_SELECT CAL_PROB DEV] [--rank-inputs F] [--bench-inputs F]
No GenAI or EvalMuse-test human label is read here.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "v7"))
import v8_core as C  # noqa: E402
import v8_labels as LB  # noqa: E402
import v7_common as VC  # noqa: E402

EPOCHS, MAX_TXT, WD, WARMUP = 5, 128, 0.05, 100
DATA = Path("/home/infres/yinwang/CS_QMI/outputs/V7/data")


# ------------------------------------------------------------------------------------------------------------------------------------ data
class RatingData:
    """Overall-valid images of one role with their vote histograms, natural weights and token ids."""

    def __init__(self, recs: list[dict], tokenizer, prompts: dict, max_len: int = MAX_TXT, limit: int | None = None):
        recs = recs[:limit] if limit else recs; self.rows = []; cache = {}
        for r in recs:
            p = prompts[r["image_id"]]
            if p not in cache:
                cache[p] = tokenizer(p)["input_ids"]
            ids = cache[p]
            if len(ids) > max_len:
                raise SystemExit(f"prompt longer than {max_len} tokens ({len(ids)}): {r['image_id']} — never truncated")
            self.rows.append({**r, "prompt": p, "ids": ids})
        self.prompt_images = defaultdict(list)
        for k, r in enumerate(self.rows):
            self.prompt_images[r["prompt_id"]].append(k)

    def __len__(self):
        return len(self.rows)


class Collate:
    def __init__(self, data: RatingData, images, processor, pad_id: int = 0):
        self.data, self.images, self.proc, self.pad = data, images, processor, pad_id

    def __call__(self, items):
        """items: [(row index, subgroup id, n subgroups of its prompt)] or [row index]."""
        import torch
        items = [it if isinstance(it, tuple) else (it, -1, 1) for it in items]; rows = [self.data.rows[i] for i, _, _ in items]
        L = max(len(r["ids"]) for r in rows); B = len(rows)
        ids = torch.full((B, L), self.pad, dtype=torch.long); att = torch.zeros((B, L), dtype=torch.long)
        for b, r in enumerate(rows):
            ids[b, :len(r["ids"])] = torch.tensor(r["ids"]); att[b, :len(r["ids"])] = 1
        return {"image": torch.stack([self.proc(self.images(r["image_id"])) for r in rows]), "ids": ids, "att": att,
                "hist": torch.tensor([r["hist"] for r in rows], dtype=torch.float32), "w": torch.tensor([r["w"] for r in rows], dtype=torch.float32),
                "subgroup": [g for _, g, _ in items], "n_sub": [n for _, _, n in items], "image_id": [r["image_id"] for r in rows],
                "prompt_id": [r["prompt_id"] for r in rows]}


# ----------------------------------------------------------------------------------------------------------------------------------- model
def build(seed: int, device, fit_hist):
    """V7's frozen FGA-BLIP2 construction + the shared 5-logit head (seeded weight, FIT log-histogram bias)."""
    import torch
    from v7_train import build_model
    model, info = build_model(MAX_TXT, seed, device)
    for n, p in model.named_parameters():
        p.requires_grad_(n.startswith(("Qformer.", "ln_vision.")) or n == "query_tokens")      # V7's trained set minus its unused heads
    g = torch.Generator().manual_seed(seed * 7919 + 5); head = torch.nn.Linear(model.Qformer.config.hidden_size, 5)
    with torch.no_grad():
        head.weight.copy_(torch.randn(head.weight.shape, generator=g) * 0.02); head.bias.copy_(torch.log(torch.tensor(fit_hist, dtype=torch.float32).clamp_min(1e-6)))
    model.v8_head = head.to(device)
    info.update({"head": "Linear(768 -> 5) on the mean of the 32 query outputs", "head_bias_init": [float(x) for x in head.bias.detach().cpu()],
                 "n_trainable_v8": int(sum(p.numel() for p in model.parameters() if p.requires_grad))})
    return model, info


def forward_logits(model, b, device):
    import torch
    image = b["image"].to(device, non_blocking=True); ids, att = b["ids"].to(device), b["att"].to(device)
    with torch.no_grad():
        vis = model.visual_encoder(image)                                                           # frozen EVA ViT-g (eval mode)
    image_embeds = model.ln_vision(vis)
    image_atts = torch.ones(image_embeds.shape[:-1], dtype=torch.long, device=device)
    q = model.query_tokens.expand(image_embeds.shape[0], -1, -1); q_att = torch.ones(q.shape[:-1], dtype=torch.long, device=device)
    out = model.Qformer.bert(ids, query_embeds=q, attention_mask=torch.cat([q_att, att], 1), encoder_hidden_states=image_embeds,
                             encoder_attention_mask=image_atts, return_dict=True)
    pooled = out.last_hidden_state[:, :q.shape[1]].float().mean(1)
    with torch.cuda.amp.autocast(enabled=False):
        return model.v8_head(pooled)                                                                # the rating head in fp32


def set_train(model):
    model.train(); model.visual_encoder.eval()                                                      # the frozen ViT stays in eval mode


def optimizer(model, lr):
    import torch
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (decay if p.ndim >= 2 and n != "query_tokens" else no_decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": WD}, {"params": no_decay, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.999))


def predict(model, data: RatingData, images, processor, device, prior, batch=32):
    import torch
    model.eval(); col = Collate(data, images, processor); out = []
    with torch.no_grad():
        for s in range(0, len(data), batch):
            b = col(list(range(s, min(s + batch, len(data)))))
            with torch.cuda.amp.autocast(enabled=device == "cuda"):
                logits = forward_logits(model, b, device)
            o = C.outputs(logits.float(), torch.tensor(prior, device=logits.device))
            for j in range(len(b["image_id"])):
                out.append({"image_id": b["image_id"][j], "prompt_id": b["prompt_id"][j], "logits": logits[j].float().cpu().tolist(),
                            "q": o["q"][j].cpu().tolist(), "u": o["u"][j].cpu().tolist(), "mu": float(o["mu"][j])})
    return out


def cal_metrics(pred: list[dict], data: RatingData, prior) -> dict:
    """Prompt-macro mean-rating MSE (the selection criterion) + secondary distribution and within-prompt diagnostics."""
    by = {r["image_id"]: r for r in data.rows}; per_p = defaultdict(list); rps, ce, pair_acc = [], [], []
    for p in pred:
        r = by[p["image_id"]]; per_p[r["prompt_id"]].append((p["mu"], r["mean"]))
        y = np.array(r["tail"]); u = np.array(p["u"]); rps.append(float(np.mean(y * (1 - u) ** 2 + (1 - y) * u ** 2)))
        ce.append(float(-(np.array(r["hist"]) * np.log(np.clip(p["q"], 1e-12, 1))).sum()))
    for v in per_p.values():
        mu, h = np.array([a for a, _ in v]), np.array([b for _, b in v]); i, j = np.triu_indices(len(v), 1); dh = h[i] - h[j]; ok = dh != 0
        if ok.any():
            ds = mu[i] - mu[j]; pair_acc.append(float(np.where(ds[ok] == 0, 0.5, (np.sign(ds[ok]) == np.sign(dh[ok])).astype(float)).mean()))
    return {"mean_mse_prompt_macro": float(np.mean([np.mean([(a - b) ** 2 for a, b in v]) for v in per_p.values()])), "n_prompts": len(per_p),
            "n_images": len(pred), "rps_micro": float(np.mean(rps)), "ce_micro": float(np.mean(ce)),
            "within_prompt_pairwise_acc_macro": float(np.mean(pair_acc)) if pair_acc else None}


def trainable_state(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith("visual_encoder.")}


# ------------------------------------------------------------------------------------------------------------------------------------ train
def load_role_data(tok, limit=None, roles=("FIT", "CAL_SELECT")):
    rows = LB.load(DATA); prompts = {r["image_id"]: r["prompt"] for r in rows}; R = LB.records(rows)
    return {role: RatingData(R[role], tok, prompts, limit=(limit if role != "FIT" else None)) for role in roles}, R


def cmd_train(a):
    import torch
    from torch.utils.data import DataLoader
    from transformers import BertTokenizerFast
    t_start = time.time(); unit = Path(a.unit_dir); unit.mkdir(parents=True, exist_ok=True); kind = C.ARMS[a.arm]
    if (unit / "done.json").exists():
        print(f"[v8_train] {unit.name} already complete", flush=True); return 0
    device = "cuda" if torch.cuda.is_available() else "cpu"; VC.stage_workdir(VC.BIG_OUT / "work"); tok = BertTokenizerFast.from_pretrained(str(VC.BERT_DIR))
    D, R = load_role_data(tok, limit=a.eval_n); fit, cal = D["FIT"], D["CAL_SELECT"]; prior = LB.fit_priors(R["FIT"]).tolist()
    fit_hist = (np.array([r["w"] for r in R["FIT"]])[:, None] * np.array([r["hist"] for r in R["FIT"]])).sum(0) / sum(r["w"] for r in R["FIT"])
    model, info = build(a.seed, device, fit_hist.tolist())
    bad = [r["image_id"] for r in fit.rows[:2000] if model.tokenizer(r["prompt"]).input_ids != r["ids"]]
    if bad:
        raise SystemExit(f"fast / model tokenizer ids differ on {len(bad)} prompts (e.g. {bad[:3]})")
    VC.import_lavis()
    from lavis.processors.blip_processors import BlipImageEvalProcessor
    proc = BlipImageEvalProcessor(image_size=364); images = VC.ArchiveImages(); opt = optimizer(model, a.lr)
    E = a.epochs; scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda"); plans = [C.batch_manifest(fit.prompt_images, a.seed, e) for e in range(E)]
    total = sum(len(p) for p in plans) if not a.max_steps else a.max_steps * E; step, start, best = 0, 0, (math.inf, None)
    log = {"arm": a.arm, "loss": kind, "lr": a.lr, "seed": a.seed, "prior_fit": prior, "fit_hist": fit_hist.tolist(), "model": info, "epochs": E,
           "weight_decay": WD, "warmup": WARMUP, "total_steps": total, "n_fit": len(fit), "n_cal_select": len(cal), "epochs_log": [], "smoke": bool(a.max_steps) or E != EPOCHS,
           "batch_plan_sha256": [__import__("hashlib").sha256(json.dumps(p).encode()).hexdigest() for p in plans]}
    resume = unit / "resume.pt"
    if resume.exists():
        ck = torch.load(resume, map_location="cpu", weights_only=False); model.load_state_dict(ck["model"], strict=False); opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"]); start, best, log, step = ck["epoch"] + 1, tuple(ck["best"]), ck["log"], ck["step"]; log.setdefault("resumed_at", []).append(start)
    col = Collate(fit, images, proc); pt = torch.tensor(prior)
    for epoch in range(start, E):
        loader = DataLoader(plans[epoch], batch_size=None, shuffle=False, collate_fn=col, num_workers=a.workers, persistent_workers=False,
                            prefetch_factor=4 if a.workers else None)
        set_train(model); t0 = time.time(); agg = Counter(); n_steps = 0; seen = 0
        for i, b in enumerate(loader):
            if a.max_steps and i >= a.max_steps:
                break
            for gpar in opt.param_groups:
                gpar["lr"] = C.warmup_cosine(step, total, a.lr, WARMUP)
            with torch.cuda.amp.autocast(enabled=device == "cuda"):
                logits = forward_logits(model, b, device)
            loss, parts, _ = C.batch_loss(logits.float(), b["hist"].to(device), b["w"].to(device), pt.to(device), kind, b["subgroup"], b["n_sub"])
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            agg.update({k: v for k, v in parts.items()}); agg["loss"] += float(loss); n_steps += 1; step += 1; seen += len(b["image_id"])
            if i % 200 == 0:
                print(f"[v8_train] {a.arm} s{a.seed} lr{a.lr:g} epoch {epoch} step {i} loss {float(loss):.4f} lr {opt.param_groups[0]['lr']:.2e} ({time.time() - t0:.0f}s)", flush=True)
        train_s = time.time() - t0; t1 = time.time(); m = cal_metrics(predict(model, cal, images, proc, device, prior), cal, prior); eval_s = time.time() - t1
        rec = {"epoch": epoch, "steps": n_steps, "images_seen": seen, "train_s": train_s, "eval_s": eval_s, "train_means": {k: v / max(n_steps, 1) for k, v in agg.items()},
               "cal_select": m, "peak_mem_gb": (torch.cuda.max_memory_allocated() / 2 ** 30) if device == "cuda" else None}
        log["epochs_log"].append(rec); print(f"[v8_train] epoch {epoch}: CAL_SELECT mean-rating MSE {m['mean_mse_prompt_macro']:.5f} ({train_s:.0f}s + {eval_s:.0f}s)", flush=True)
        state = trainable_state(model)
        if m["mean_mse_prompt_macro"] < best[0]:                                                  # strict: ties keep the earlier epoch
            best = (m["mean_mse_prompt_macro"], epoch); torch.save({"model": state, "epoch": epoch}, unit / "best.tmp"); os.replace(unit / "best.tmp", unit / "best.pt")
        torch.save({"model": state, "opt": opt.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch, "best": list(best), "log": log, "step": step}, unit / "resume.tmp")
        os.replace(unit / "resume.tmp", resume)
    log.update({"selected_epoch": best[1], "selected_cal_select_mse": best[0], "seconds": time.time() - t_start, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"})
    (unit / "log.json").write_text(json.dumps(log, indent=1))
    if not a.max_steps and E == EPOCHS:
        (unit / "done.json").write_text(json.dumps({"unit": unit.name, "selected_epoch": best[1], "cal_select_mse": best[0], "gpu": log["gpu"]}, indent=1))
    return 0


class CandRows(RatingData):
    """External candidates (no labels): rows with a dummy histogram, read by Collate; ids from the same tokenizer and length rule."""

    def __init__(self, cands: list[dict], tokenizer, max_len: int = MAX_TXT):
        self.rows = []; cache = {}
        for c in cands:
            p = c["prompt"]
            if p not in cache:
                cache[p] = tokenizer(p)["input_ids"]
            if len(cache[p]) > max_len:
                raise SystemExit(f"external prompt longer than {max_len} tokens: {c['prompt_id']}")
            self.rows.append({**c, "ids": cache[p], "hist": [0.2] * 5, "w": 1.0})


def cmd_export(a):
    """Selected checkpoint (best.pt), eval mode -> q / u / mu per image for the EvalMuse roles and, optionally, GenAI-Rank / GenAI-Bench candidates."""
    import torch
    from transformers import BertTokenizerFast
    sys.path.insert(0, str(HERE.parent / "v7x"))
    unit = Path(a.unit_dir); log = json.loads((unit / "log.json").read_text()); device = "cuda" if torch.cuda.is_available() else "cpu"; prior = log["prior_fit"]
    VC.stage_workdir(VC.BIG_OUT / "work"); tok = BertTokenizerFast.from_pretrained(str(VC.BERT_DIR))
    model, _ = build(log["seed"], device, log["fit_hist"]); ck = torch.load(unit / "best.pt", map_location="cpu", weights_only=False)
    msg = model.load_state_dict(ck["model"], strict=False)
    if [k for k in msg.missing_keys if not k.startswith("visual_encoder.")] or msg.unexpected_keys:
        raise SystemExit(f"selected checkpoint does not match the model: {msg}")
    VC.import_lavis()
    from lavis.processors.blip_processors import BlipImageEvalProcessor
    proc = BlipImageEvalProcessor(image_size=364); out = unit / "pred"; out.mkdir(exist_ok=True); res = {}
    D, R = load_role_data(tok, roles=tuple(a.roles)); arch = VC.ArchiveImages()
    for role in a.roles:
        pred = predict(model, D[role], arch, proc, device, prior); res[role] = cal_metrics(pred, D[role], prior)
        (out / f"images_{role}.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pred))
    ext = []
    if a.rank_inputs:
        from v7_train import RankImages
        ext.append(("genai_rank", VC.read_jsonl(a.rank_inputs), RankImages()))
    if a.bench_inputs:
        from x30_m_bench import FileImages
        ext.append(("genai_bench", VC.read_jsonl(a.bench_inputs), FileImages(a.bench_images)))
    for tag, cands, imgs in ext:
        rows = CandRows(cands, tok); pred = predict(model, rows, imgs, proc, device, prior); by = {c["image_id"]: c for c in cands}
        (out / f"{tag}.jsonl").write_text("".join(json.dumps({"prompt_id": by[p["image_id"]]["prompt_id"], "generator_id": by[p["image_id"]].get("generator_id"),
                                                              "image_id": p["image_id"], "candidate_index": by[p["image_id"]].get("candidate_index"),
                                                              "score": p["mu"], "q": p["q"], "u": p["u"]}) + "\n" for p in pred))
        res[tag] = {"n": len(pred)}
    (out / "export.json").write_text(json.dumps({"unit": unit.name, "selected_epoch": ck["epoch"], "prior": prior, "roles": res}, indent=1)); return 0


def cmd_features(a):
    """Table C inputs: the UNTRAINED model (V7's frozen BLIP-2 COCO construction, no V8 training) -> the 768-d mean of the 32 query outputs per
    image (exactly the V8 head's input) for FIT / CAL_SELECT / CAL_PROB / DEV and the GenAI-Rank candidates; float32 npy + an index jsonl per role."""
    import hashlib
    import torch
    from transformers import BertTokenizerFast
    device = "cuda" if torch.cuda.is_available() else "cpu"; VC.stage_workdir(VC.BIG_OUT / "work"); tok = BertTokenizerFast.from_pretrained(str(VC.BERT_DIR))
    from v7_train import build_model
    model, info = build_model(MAX_TXT, 0, device); model.eval(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    VC.import_lavis()
    from lavis.processors.blip_processors import BlipImageEvalProcessor
    proc = BlipImageEvalProcessor(image_size=364); D, R = load_role_data(tok, roles=("FIT", "CAL_SELECT", "CAL_PROB", "DEV")); arch = VC.ArchiveImages(); jobs = [(r, D[r], arch) for r in D]
    if a.rank_inputs:
        from v7_train import RankImages
        jobs.append(("GENAI_RANK", CandRows(VC.read_jsonl(a.rank_inputs), tok), RankImages()))
    man = {"model": info, "pooling": "mean of the 32 query outputs (V8 head input)", "parts": []}
    for role, data, imgs in jobs:
        col = Collate(data, imgs, proc); feats = []
        with torch.no_grad():
            for s0 in range(0, len(data), 32):
                b = col(list(range(s0, min(s0 + 32, len(data))))); image = b["image"].to(device); ids, att = b["ids"].to(device), b["att"].to(device)
                with torch.cuda.amp.autocast(enabled=device == "cuda"):
                    emb = model.ln_vision(model.visual_encoder(image)); ia = torch.ones(emb.shape[:-1], dtype=torch.long, device=device)
                    q = model.query_tokens.expand(emb.shape[0], -1, -1); qa = torch.ones(q.shape[:-1], dtype=torch.long, device=device)
                    o = model.Qformer.bert(ids, query_embeds=q, attention_mask=torch.cat([qa, att], 1), encoder_hidden_states=emb, encoder_attention_mask=ia, return_dict=True)
                feats.append(o.last_hidden_state[:, :q.shape[1]].float().mean(1).cpu().numpy())
        X = np.concatenate(feats).astype(np.float32); np.save(out / f"feats_{role}.npy", X)
        keep = ("image_id", "prompt_id", "generator_id", "candidate_index", "hist", "tail", "mean", "w", "n")
        (out / f"index_{role}.jsonl").write_text("".join(json.dumps({k: r[k] for k in keep if k in r and not (role == "GENAI_RANK" and k in ("hist", "tail", "mean"))}) + "\n" for r in data.rows))
        man["parts"].append({"tag": role, "n": len(X), "sha256": hashlib.sha256(X.tobytes()).hexdigest()}); print(f"[v8 features] {role}: {X.shape}", flush=True)
    (out / "manifest.json").write_text(json.dumps(man, indent=1)); return 0


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--arm", required=True, choices=sorted(C.ARMS)); t.add_argument("--lr", type=float, required=True)
    t.add_argument("--seed", type=int, required=True); t.add_argument("--unit-dir", required=True); t.add_argument("--workers", type=int, default=6)
    t.add_argument("--max-steps", type=int, default=0); t.add_argument("--eval-n", type=int, default=None)
    t.add_argument("--epochs", type=int, default=EPOCHS, help="smoke / timing only; a formal or tuning unit always runs 5")
    x = sub.add_parser("export"); x.add_argument("--unit-dir", required=True); x.add_argument("--roles", nargs="+", default=["CAL_SELECT", "CAL_PROB", "DEV"])
    x.add_argument("--rank-inputs", default=None); x.add_argument("--bench-inputs", default=None)
    x.add_argument("--bench-images", default="/projects/EEG-foundation-model/yinghao/outputs_v7/genai_bench_images")
    f = sub.add_parser("features"); f.add_argument("--out", required=True); f.add_argument("--rank-inputs", default=None)
    a = ap.parse_args(argv); return {"train": cmd_train, "export": cmd_export, "features": cmd_features}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
