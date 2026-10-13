"""V7 M arms on the FGA-BLIP2 training path (V7 package docs/06): one shared semantic-element interface, three element objectives.
  M0_semantic_l1 (|sigmoid(z) - y|), M1_semantic_logistic (balanced soft BCE), M2_semantic_vcs ((1 - J) / 4); every arm: the same init (FGA config
  'coco': EVA ViT-g + blip2_finetune_coco.pth, ViT frozen, mask head seeded), the same data order per seed, the same auxiliary terms
  (overall L1 on the 1-5 vote mean, validity-mask L1 on content tokens; FGA coefficients 1 / 0.1 / 0.1), the native optimizer groups, LAVIS's
  LinearWarmupCosineLRScheduler (warm-up 100 steps from 1e-8, per-epoch cosine to 0), AMP fp16 with GradScaler, global batch 14, 5 epochs,
  drop_last.  Input: deterministic full-frame resize to 364 (BLIP eval processor: no crop, no flip); the raw prompt with uncased BERT, length
  <= max_txt_len (a longer prompt is an error, never truncated).  Element logit = mean of the span tokens' Bernoulli logits (itm class-1 minus
  class-0); an element counts once.  Epoch selection: natural element Brier on CAL_SELECT with the FIT prior (prompt -> image -> element macro).
    train --arm A --lr LR --seed S --unit-dir D [--max-txt-len L] [--max-steps N --eval-n N]   (the last two: smoke only)
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v7_common as VC  # noqa: E402

EPOCHS, BATCH, WARMUP, WD, LR_LAYER_DECAY, WARMUP_START = 5, 14, 100, 0.05, 0.95, 1e-8
MAX_SPAN_TOKENS = 64


# ------------------------------------------------------------------------------------------------------------------------------------ data
def load_roles(data_dir: Path):
    imgs = VC.read_jsonl(data_dir / "manifest_images.jsonl"); els = VC.read_jsonl(data_dir / "manifest_elements.jsonl")
    by_img = defaultdict(list)
    for e in els:
        if e["scope"] == "main" and e["spans"]:
            by_img[e["image_id"]].append(e)
    return imgs, by_img


def fit_prior_spanned(imgs, by_img):
    """pi over FIT main elements that have a span (the M training population), prompt -> image -> element uniform."""
    fit = [i for i in imgs if i["role"] == "FIT"]; per = Counter(i["prompt_id"] for i in fit); n_i, n_p = len(fit), len(per)
    num = den = 0.0
    for i in fit:
        es = by_img.get(i["image_id"], [])
        if not es:
            continue
        w_img = n_i / (n_p * per[i["prompt_id"]])
        for e in es:
            num += w_img / len(es) * e["y"]; den += w_img / len(es)
    return num / den


class ElementData:
    """Rows of one role: image id, prompt, overall mean, FIT weight (1 outside FIT), and the elements' token positions from offsets."""

    def __init__(self, imgs, by_img, role, tokenizer_fast, max_len, limit=None):
        rows = [i for i in imgs if i["role"] == role and i.get("overall_mean") is not None]
        rows = rows[:limit] if limit else rows
        per = Counter(i["prompt_id"] for i in rows); n_i, n_p = len(rows), len(per); self.rows = []; cache = {}
        for i in rows:
            p = i["prompt"]
            if p not in cache:
                enc = tokenizer_fast(p, return_offsets_mapping=True); cache[p] = (enc["input_ids"], enc["offset_mapping"])
            ids, offs = cache[p]
            if len(ids) > max_len:
                raise SystemExit(f"prompt longer than max_txt_len ({len(ids)} > {max_len}): {i['image_id']} — never truncated silently")
            els = []
            for e in by_img.get(i["image_id"], []):
                t = VC.token_spans(offs, e["spans"])
                if t:
                    els.append({"element_id": e["element_id"], "y": e["y"], "tok": t[:MAX_SPAN_TOKENS], "n_tok": len(t)})
            self.rows.append({"image_id": i["image_id"], "prompt_id": i["prompt_id"], "prompt": p, "ids": ids, "overall": i["overall_mean"],
                              "w": n_i / (n_p * per[i["prompt_id"]]), "elements": els})

    def __len__(self):
        return len(self.rows)


class Batcher:
    """Collate: images (BLIP eval processor 364), padded ids / masks, element pooling indices, weights and targets."""

    def __init__(self, data: ElementData, images, processor, pad_id=0):
        self.data, self.images, self.proc, self.pad = data, images, processor, pad_id

    def __call__(self, idx):
        import torch
        rows = [self.data.rows[k] for k in idx]; L = max(len(r["ids"]) for r in rows); B = len(rows)
        ids = torch.full((B, L), self.pad, dtype=torch.long); att = torch.zeros((B, L), dtype=torch.long); content = torch.zeros((B, L))
        owner, toks, ys, wes, eids = [], [], [], [], []; gt = torch.zeros((B, L)); has_el = torch.zeros(B)
        for b, r in enumerate(rows):
            n = len(r["ids"]); ids[b, :n] = torch.tensor(r["ids"]); att[b, :n] = 1; content[b, 1:n - 1] = 1
            es = r["elements"]
            for e in es:
                owner.append(b); toks.append(e["tok"]); ys.append(e["y"]); wes.append(r["w"] / len(es)); eids.append(e["element_id"]); gt[b, e["tok"]] = 1
            has_el[b] = 1.0 if es else 0.0
        K = max([len(t) for t in toks] + [1]); tok_idx = torch.zeros((len(toks), K), dtype=torch.long); tok_mask = torch.zeros((len(toks), K), dtype=torch.bool)
        for k, t in enumerate(toks):
            tok_idx[k, :len(t)] = torch.tensor(t); tok_mask[k, :len(t)] = True
        img = torch.stack([self.proc(self.images(r["image_id"])) for r in rows])
        return {"image": img, "ids": ids, "att": att, "content": content, "mask_gt": gt, "has_el": has_el, "owner": torch.tensor(owner, dtype=torch.long),
                "tok_idx": tok_idx, "tok_mask": tok_mask, "y": torch.tensor(ys, dtype=torch.float32), "w_e": torch.tensor(wes, dtype=torch.float32),
                "w_img": torch.tensor([r["w"] for r in rows], dtype=torch.float32), "overall": torch.tensor([r["overall"] for r in rows], dtype=torch.float32),
                "image_id": [r["image_id"] for r in rows], "prompt_id": [r["prompt_id"] for r in rows], "element_id": eids}


# ----------------------------------------------------------------------------------------------------------------------------------- model
def build_model(max_txt_len: int, seed: int, device):
    """FGA_Blip2 through the native from_config path (coco defaults + the FGA overrides), weights loaded explicitly so that the missing /
    unexpected keys are recorded; the mask head is the only random module (seeded)."""
    import torch
    VC.import_lavis()
    from lavis.common.registry import registry
    from omegaconf import OmegaConf
    cls = registry.get_model_class("fga_blip2"); cfg = OmegaConf.load(cls.default_config_path("coco")).model
    cfg.freeze_vit = True; cfg.max_txt_len = max_txt_len; finetuned = cfg.finetuned; cfg.load_finetuned = False; cfg.load_pretrained = False
    torch.manual_seed(seed); model = cls.from_config(cfg)
    msg = model.load_checkpoint(finetuned)
    info = {"init": finetuned, "missing_keys": list(msg.missing_keys), "unexpected_keys": list(msg.unexpected_keys),
            "trainable": sorted({n.split(".")[0] for n, p in model.named_parameters() if p.requires_grad}),
            "frozen": sorted({n.split(".")[0] for n, p in model.named_parameters() if not p.requires_grad}),
            "n_trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "n_total": int(sum(p.numel() for p in model.parameters())), "max_txt_len": max_txt_len, "vit_precision": cfg.vit_precision}
    return model.to(device), info


def forward(model, b, device):
    """-> overall O [B] (native: query-token class-1 probability mean * 4 + 1), token Bernoulli logits [B, L], mask head output [B, L]."""
    import torch
    image = b["image"].to(device, non_blocking=True); ids, att = b["ids"].to(device), b["att"].to(device)
    image_embeds = model.ln_vision(model.visual_encoder(image)); image_atts = torch.ones(image_embeds.shape[:-1], dtype=torch.long, device=device)
    q = model.query_tokens.expand(image_embeds.shape[0], -1, -1); q_att = torch.ones(q.shape[:-1], dtype=torch.long, device=device)
    out = model.Qformer.bert(ids, query_embeds=q, attention_mask=torch.cat([q_att, att], 1), encoder_hidden_states=image_embeds,
                             encoder_attention_mask=image_atts, return_dict=True)
    logit = model.itm_head(out.last_hidden_state).float(); nq = q.shape[1]
    O = torch.softmax(logit[:, :nq], -1)[..., 1].mean(1) * 4 + 1; z_tok = logit[:, nq:, 1] - logit[:, nq:, 0]
    text_out = model.Qformer.bert(ids, attention_mask=att, return_dict=True); mask = model.mask_proj(text_out.last_hidden_state).squeeze(-1).float()
    return O, z_tok, mask


def losses(model, b, device, objective, prior):
    import torch
    O, z_tok, mask = forward(model, b, device); w_img = b["w_img"].to(device)
    l_o = (w_img * (O - b["overall"].to(device)).abs()).sum() / w_img.sum()
    z = VC.pool_logits(z_tok, b["owner"].to(device), b["tok_idx"].to(device), b["tok_mask"].to(device)); w_e = b["w_e"].to(device)
    l_e = VC.element_loss(z, b["y"].to(device), w_e, prior, float(w_e.sum()), objective) if len(z) else z_tok.sum() * 0
    content, gt, has = b["content"].to(device), b["mask_gt"].to(device), b["has_el"].to(device)
    per_img = ((mask - gt).abs() * content).sum(1) / content.sum(1).clamp(min=1); wm = w_img * has
    l_m = (wm * per_img).sum() / wm.sum() if wm.sum() > 0 else per_img.sum() * 0
    total = VC.COEF["overall"] * l_o + VC.COEF["element"] * l_e + VC.COEF["mask"] * l_m
    return total, {"overall_l1": float(l_o), "element": float(l_e), "mask_l1": float(l_m)}


def evaluate(model, data: ElementData, images, processor, device, prior, batch=32):
    """Eval mode (dropout off, no gradient): element logits / natural probabilities, overall O, CAL metrics."""
    import torch
    model.eval(); bt = Batcher(data, images, processor); el, ov = [], []
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device == "cuda"):
        for s in range(0, len(data), batch):
            b = bt(list(range(s, min(s + batch, len(data))))); O, z_tok, _ = forward(model, b, device)
            z = VC.pool_logits(z_tok, b["owner"].to(device), b["tok_idx"].to(device), b["tok_mask"].to(device)).float().cpu().numpy()
            for k, eid in enumerate(b["element_id"]):
                r = b["owner"][k].item(); el.append({"element_id": eid, "image_id": b["image_id"][r], "prompt_id": b["prompt_id"][r], "z": float(z[k]), "y": float(b["y"][k])})
            for r in range(len(b["image_id"])):
                ov.append({"image_id": b["image_id"][r], "prompt_id": b["prompt_id"][r], "O": float(O[r]), "overall": float(b["overall"][r])})
    model.train()
    z = np.array([e["z"] for e in el]); y = np.array([e["y"] for e in el]); p = VC.natural_probability(z, prior); pids = [e["prompt_id"] for e in el]; iids = [e["image_id"] for e in el]
    brier, n_p = VC.hierarchical_mean(pids, iids, VC.expected_brier(p, y)); t = np.tanh(z / 2)
    m = {"natural_brier_macro": brier, "n_prompts": n_p, "n_elements": len(el), "natural_brier_micro": float(VC.expected_brier(p, y).mean()),
         "nll_macro": VC.hierarchical_mean(pids, iids, VC.soft_nll(p, y))[0], "l1_macro": VC.hierarchical_mean(pids, iids, np.abs(1 / (1 + np.exp(-z)) - y))[0],
         "common_J": VC.common_j(t, y), "overall_srcc": VC.avg_rank_spearman([o["O"] for o in ov], [o["overall"] for o in ov]),
         "overall_l1": float(np.mean([abs(o["O"] - o["overall"]) for o in ov]))}
    return m, el, ov


# ------------------------------------------------------------------------------------------------------------------------------------ train
def trainable_state(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith("visual_encoder.")}


def state_hash(state) -> str:
    """sha256 over the (sorted) tensors' names, dtypes, shapes and bytes: the epoch / selected weights fingerprint."""
    import hashlib
    h = hashlib.sha256()
    for k in sorted(state):
        t = state[k].contiguous(); h.update(f"{k}|{t.dtype}|{tuple(t.shape)}".encode())
        h.update((t.float() if str(t.dtype) == "torch.bfloat16" else t).numpy().tobytes())
    return h.hexdigest()


def cmd_export(a):
    """The selected checkpoint (best.pt) in eval mode -> element / image predictions for CAL_SELECT, CAL_PROB and DEV (EvalMuse local roles)."""
    import torch
    from transformers import BertTokenizerFast
    unit = Path(a.unit_dir); log = json.loads((unit / "log.json").read_text()); device = "cuda" if torch.cuda.is_available() else "cpu"
    VC.stage_workdir(VC.BIG_OUT / "work"); tok = BertTokenizerFast.from_pretrained(str(VC.BERT_DIR)); imgs, by_img = load_roles(Path(a.data_dir))
    model, _ = build_model(log["model"]["max_txt_len"], log["seed"], device); ck = torch.load(unit / "best.pt", map_location="cpu", weights_only=False)
    msg = model.load_state_dict(ck["model"], strict=False)
    if [k for k in msg.missing_keys if not k.startswith("visual_encoder.")] or msg.unexpected_keys:
        raise SystemExit(f"selected checkpoint does not match the model: {msg}")
    VC.import_lavis()
    from lavis.processors.blip_processors import BlipImageEvalProcessor
    proc = BlipImageEvalProcessor(image_size=364); images = VC.ArchiveImages(); prior = log["prior_fit_spanned"]; out = unit / "pred"; out.mkdir(exist_ok=True); res = {}
    for role in a.roles:
        data = ElementData(imgs, by_img, role, tok, log["model"]["max_txt_len"]); m, el, ov = evaluate(model, data, images, proc, device, prior)
        with open(out / f"elements_{role}.jsonl", "w") as fh:
            for e in el:
                fh.write(json.dumps({**e, "p_prior": float(VC.natural_probability([e["z"]], prior)[0])}) + "\n")
        with open(out / f"images_{role}.jsonl", "w") as fh:
            for o in ov:
                fh.write(json.dumps(o) + "\n")
        res[role] = m; print(f"[v7_train] export {unit.name} {role}: {m['n_elements']} elements, natural Brier {m['natural_brier_macro']:.5f}", flush=True)
    (out / "export.json").write_text(json.dumps({"unit": unit.name, "selected_epoch": ck["epoch"], "state_sha256": ck.get("state_sha256"), "prior": prior, "roles": res}, indent=1))
    return 0


def cmd_train(a):
    import torch
    from torch.utils.data import DataLoader
    t_start = time.time(); unit = Path(a.unit_dir); unit.mkdir(parents=True, exist_ok=True); objective = VC.ARMS[a.arm]
    if (unit / "done.json").exists():
        print(f"[v7_train] {unit.name} already complete", flush=True); return 0
    device = "cuda" if torch.cuda.is_available() else "cpu"; VC.stage_workdir(VC.BIG_OUT / "work")
    from transformers import BertTokenizerFast
    tok = BertTokenizerFast.from_pretrained(str(VC.BERT_DIR))
    imgs, by_img = load_roles(Path(a.data_dir)); prior = fit_prior_spanned(imgs, by_img)
    audit = json.loads((Path(a.data_dir) / "audit.json").read_text())
    fit = ElementData(imgs, by_img, "FIT", tok, a.max_txt_len); cal = ElementData(imgs, by_img, "CAL_SELECT", tok, a.max_txt_len, limit=a.eval_n)
    model, info = build_model(a.max_txt_len, a.seed, device)
    mt = model.tokenizer; bad = [r["image_id"] for r in fit.rows[:2000] if mt(r["prompt"]).input_ids != r["ids"]]
    if bad:
        raise SystemExit(f"fast / model tokenizer ids differ on {len(bad)} prompts (e.g. {bad[:3]})")
    VC.import_lavis()
    from lavis.common.optims import LinearWarmupCosineLRScheduler
    from lavis.processors.blip_processors import BlipImageEvalProcessor
    proc = BlipImageEvalProcessor(image_size=364); images = VC.ArchiveImages()
    opt = torch.optim.AdamW(model.get_optimizer_params(WD, LR_LAYER_DECAY), lr=a.lr, betas=(0.9, 0.999))
    sched = LinearWarmupCosineLRScheduler(opt, max_epoch=EPOCHS, min_lr=0.0, init_lr=a.lr, warmup_steps=WARMUP, warmup_start_lr=WARMUP_START)
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda"); start, best = 0, (math.inf, None)
    log = {"arm": a.arm, "objective": objective, "lr": a.lr, "seed": a.seed, "prior_fit_spanned": prior, "prior_audit": audit["fit"].get("pi"), "model": info,
           "coef": VC.COEF, "batch": BATCH, "epochs": EPOCHS, "warmup": WARMUP, "weight_decay": WD, "n_fit": len(fit), "n_cal_select": len(cal),
           "fit_elements": sum(len(r["elements"]) for r in fit.rows), "epochs_log": [], "smoke": bool(a.max_steps)}
    resume = unit / "resume.pt"
    if resume.exists():
        ck = torch.load(resume, map_location="cpu", weights_only=False); model.load_state_dict(ck["model"], strict=False); opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"]); start, best, log = ck["epoch"] + 1, tuple(ck["best"]), ck["log"]; log.setdefault("resumed_at", []).append(start)
    bt = Batcher(fit, images, proc)
    for epoch in range(start, EPOCHS):
        g = torch.Generator().manual_seed(a.seed * 1000 + epoch)                     # the same order for every arm at this seed
        loader = DataLoader(range(len(fit)), batch_size=BATCH, shuffle=True, drop_last=True, generator=g, collate_fn=bt, num_workers=a.workers,
                            persistent_workers=False, prefetch_factor=4 if a.workers else None)
        model.train(); t0 = time.time(); agg = Counter(); n_steps = 0
        for i, b in enumerate(loader):
            if a.max_steps and i >= a.max_steps:
                break
            sched.step(cur_epoch=epoch, cur_step=i)
            with torch.cuda.amp.autocast(enabled=device == "cuda"):
                loss, parts = losses(model, b, device, objective, prior)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad()
            agg.update(parts); agg["loss"] += float(loss); n_steps += 1
            if i % 200 == 0:
                print(f"[v7_train] {a.arm} s{a.seed} lr{a.lr:g} epoch {epoch} step {i} loss {float(loss):.4f} lr {opt.param_groups[0]['lr']:.2e} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        train_s = time.time() - t0; t1 = time.time(); m, _, _ = evaluate(model, cal, images, proc, device, prior); eval_s = time.time() - t1
        rec = {"epoch": epoch, "steps": n_steps, "images_seen": n_steps * BATCH, "train_s": train_s, "eval_s": eval_s, "lr_end": opt.param_groups[0]["lr"],
               "train_means": {k: v / max(n_steps, 1) for k, v in agg.items()}, "cal_select": m,
               "peak_mem_gb": (torch.cuda.max_memory_allocated() / 2 ** 30) if device == "cuda" else None}
        log["epochs_log"].append(rec); print(f"[v7_train] epoch {epoch}: CAL_SELECT natural Brier {m['natural_brier_macro']:.5f} ({train_s:.0f}s + {eval_s:.0f}s)", flush=True)
        state = trainable_state(model); rec["state_sha256"] = state_hash(state)
        if m["natural_brier_macro"] < best[0]:
            best = (m["natural_brier_macro"], epoch); torch.save({"model": state, "epoch": epoch, "state_sha256": rec["state_sha256"]}, unit / "best.tmp"); os.replace(unit / "best.tmp", unit / "best.pt")
        torch.save({"model": state, "opt": opt.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch, "best": list(best), "log": log}, unit / "resume.tmp")
        os.replace(unit / "resume.tmp", resume)
    log.update({"selected_epoch": best[1], "selected_cal_select_brier": best[0], "native_criteria_per_epoch": [{"epoch": r["epoch"], "l1_macro": r["cal_select"]["l1_macro"],
                "nll_macro": r["cal_select"]["nll_macro"], "common_J": r["cal_select"]["common_J"]} for r in log["epochs_log"]],
                "seconds": time.time() - t_start, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"})
    (unit / "log.json").write_text(json.dumps(log, indent=1))
    if not a.max_steps:
        (unit / "done.json").write_text(json.dumps({"unit": unit.name, "selected_epoch": best[1], "cal_select_brier": best[0], "gpu": log["gpu"]}, indent=1))
    return 0


class RankImages:
    """GenAI-Rank candidates read straight from DALLE_3.zip / SDXL_Base.zip (member names as in genai_rank_inputs.jsonl)."""

    def __init__(self, root="/projects/EEG-foundation-model/yinghao/data/genai_rank"):
        self.root = root; self._z = {}; self._pid = None

    def __call__(self, image_id):
        import io
        import zipfile
        from PIL import Image
        if self._pid != os.getpid():
            self._z, self._pid = {}, os.getpid()
        g = image_id.split("/")[0]; z = self._z.get(g) or self._z.setdefault(g, zipfile.ZipFile(f"{self.root}/{g}.zip"))
        return Image.open(io.BytesIO(z.read(image_id))).convert("RGB")


def cmd_export_rank(a):
    """Selected checkpoint, eval mode -> per GenAI-Rank candidate: O, the parser elements' logits, E = mean prior-corrected probability over the
    prompt's parsed elements (decomposition_failed -> E None; the deployment score then falls back to O, flagged).  No human score is read."""
    import torch
    from transformers import BertTokenizerFast
    unit = Path(a.unit_dir); log = json.loads((unit / "log.json").read_text()); device = "cuda" if torch.cuda.is_available() else "cpu"; prior = log["prior_fit_spanned"]
    VC.stage_workdir(VC.BIG_OUT / "work"); tok = BertTokenizerFast.from_pretrained(str(VC.BERT_DIR)); L = log["model"]["max_txt_len"]
    cands = VC.read_jsonl(a.inputs); cache = {}
    for r in VC.read_jsonl(a.parser_cache):
        if "prompt" in r:
            cache[r["prompt"]] = r
    model, _ = build_model(L, log["seed"], device); ck = torch.load(unit / "best.pt", map_location="cpu", weights_only=False); model.load_state_dict(ck["model"], strict=False)
    VC.import_lavis()
    from lavis.processors.blip_processors import BlipImageEvalProcessor
    proc = BlipImageEvalProcessor(image_size=364); images = RankImages(); rows = []
    for c in cands:
        pr = cache.get(c["prompt"]); enc = tok(c["prompt"], return_offsets_mapping=True); ids = enc["input_ids"]
        if len(ids) > L:
            raise SystemExit(f"external prompt longer than max_txt_len ({len(ids)} > {L}): {c['prompt_id']}")
        els = [] if pr is None or pr["status"] != "ok" else [{"element_id": f"{c['prompt_id']}::{e['id']}", "y": 0.0, "tok": VC.token_spans(enc["offset_mapping"], e["spans"])[:MAX_SPAN_TOKENS],
                                                             "polarity": e["polarity"]} for e in pr["elements"]]
        els = [e for e in els if e["tok"]]
        rows.append({"image_id": c["image_id"], "prompt_id": c["prompt_id"], "prompt": c["prompt"], "ids": ids, "overall": 0.0, "w": 1.0, "elements": els,
                     "generator_id": c["generator_id"], "candidate_index": c["candidate_index"], "parser_status": "missing" if pr is None else pr["status"]})
    data = ElementData.__new__(ElementData); data.rows = rows; bt = Batcher(data, images, proc); out = []
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device == "cuda"):
        for s in range(0, len(rows), 32):
            idx = list(range(s, min(s + 32, len(rows)))); b = bt(idx); O, z_tok, _ = forward(model, b, device)
            z = VC.pool_logits(z_tok, b["owner"].to(device), b["tok_idx"].to(device), b["tok_mask"].to(device)).float().cpu().numpy() if len(b["owner"]) else np.zeros(0)
            per = defaultdict(list)
            for k in range(len(z)):
                per[int(b["owner"][k])].append(float(z[k]))
            for j, ridx in enumerate(idx):
                r = rows[ridx]; zs = per.get(j, []); ps = VC.natural_probability(zs, prior).tolist() if zs else []
                out.append({"prompt_id": r["prompt_id"], "generator_id": r["generator_id"], "image_id": r["image_id"], "candidate_index": r["candidate_index"],
                            "O": float(O[j]), "E": (float(np.mean(ps)) if ps else None), "element_z": zs, "n_elements": len(zs),
                            "decomposition_failed": not ps, "parser_status": r["parser_status"]})
    dst = unit / "pred" / "genai_rank.jsonl"; dst.parent.mkdir(exist_ok=True)
    with open(dst, "w") as fh:
        for o in out:
            fh.write(json.dumps(o) + "\n")
    print(f"[v7_train] export-rank {unit.name}: {len(out)} candidates, decomposition_failed {sum(o['decomposition_failed'] for o in out)}", flush=True); return 0


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--arm", choices=sorted(VC.ARMS), required=True); t.add_argument("--lr", type=float, required=True)
    t.add_argument("--seed", type=int, required=True); t.add_argument("--unit-dir", required=True); t.add_argument("--data-dir", default=str(VC.DATA_OUT))
    t.add_argument("--max-txt-len", type=int, default=128); t.add_argument("--workers", type=int, default=8); t.add_argument("--max-steps", type=int, default=0)
    t.add_argument("--eval-n", type=int, default=None)
    x = sub.add_parser("export"); x.add_argument("--unit-dir", required=True); x.add_argument("--data-dir", default=str(VC.DATA_OUT))
    x.add_argument("--roles", nargs="+", default=["CAL_SELECT", "CAL_PROB", "DEV"])
    xr = sub.add_parser("export-rank"); xr.add_argument("--unit-dir", required=True); xr.add_argument("--inputs", default=str(VC.DATA_OUT / "genai_rank_inputs.jsonl"))
    xr.add_argument("--parser-cache", default=str(VC.BIG_OUT / "parser" / "genai_bench_prompts_cache_v2.jsonl"))
    a = ap.parse_args(argv)
    return {"train": cmd_train, "export": cmd_export, "export-rank": cmd_export_rank}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
