#!/usr/bin/env python
"""V8.1 C1 — partial Q-former adaptation (FULL_PLAN_CN.md 06 "条件实验" C1; owner approval 2026-10-15 「这俩都可以提交」, main df4ced4).

ONE factor changed against V8's formal recipe: the trainable set.  V8 trains the whole Q-former + ln_vision + query tokens + the 5-logit head
(186,314,048 parameters); C1 trains ONLY the last two BertLayer blocks of the Q-former encoder (resolved by enumerating the real
`model.Qformer.bert.encoder.layer` ModuleList, never by name guessing) + ln_vision + query tokens + the head.  Every other Q-former parameter is
frozen AND its modules run in eval mode during training (no dropout there); the two trainable blocks run in train mode; the ViT stays frozen
and in eval mode as in V8.  Everything else is V8's code, unchanged and imported from the read-only copy scripts/v81/v8_src (PROVENANCE.md;
blob 67f9dd5d of scripts/v8/v8_train.py, unchanged since the formal commit cc6299f): build(seed, FIT histogram) initialisation, the per-seed
5-epoch grouped batch plans, AdamW (0.9, 0.999) with weight decay 0.05 on matrices, 100-step warm-up + cosine, AMP fp16 + GradScaler,
the v8_core losses, CAL_SELECT prompt-macro mean-rating MSE selection (ties -> earlier epoch), best.pt / resume.pt, and the export.

`cmd_train` below is V8's `cmd_train` source with exactly the substitutions listed in V8_TO_C1 (tests/v81/test_c1.py re-derives it from the
V8 source and asserts equality).  Extra artifact, never used for selection: the trainable tensors after every epoch (<unit>/epochs/).

    train    --arm U-CS --lr parent --seed 0 --unit-dir D [--max-steps N --eval-n N --epochs E]   (lr 'parent' = the V8 formal unit's lr)
    export   --unit-dir D --roles CAL_SELECT CAL_PROB DEV                                           (V8's export; no GenAI / TIFA input)
    audit    --unit-dir D [--ids reports/V8_1_DIAGNOSTICS/results/fit_audit_ids.json]               (FIT-AUDIT predictions of best.pt and resume.pt)
    selftest --out F [--arm U-CS --seed 0]                                                           (real stack: inventory, modes, one-step mask check, V8 vs C1 s/step)
    inventory --out F                                                                                (parameter inventory only)
No GenAI-Rank / GenAI-Bench / TIFA160 / EvalMuse-test label or file is read here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "v8_src" / "v8")); sys.path.insert(0, str(HERE / "v8_src" / "v7"))
import v8_core as C  # noqa: E402
import v8_labels as LB  # noqa: E402
import v8_train as T  # noqa: E402
import v7_common as VC  # noqa: E402
from v8_train import (EPOCHS, WARMUP, WD, Collate, cal_metrics, forward_logits, load_role_data, optimizer, predict,  # noqa: E402,F401
                      trainable_state)

C1_VARIANT = "V8.1-C1: last 2 Q-former BertLayer blocks + ln_vision + query_tokens + head trainable; rest of the Q-former frozen and eval"
V8_UNITS = Path("/projects/EEG-foundation-model/yinghao/outputs_v8/units")
V8_TRAINABLE_TOTAL = 186314048                       # V8's trainable count (param audit, results/param_audit.json init_meta)
N_BLOCKS = 2

# V8 cmd_train -> C1 cmd_train: (old, new) applied in order to the V8 source; each old occurs exactly once (tests/v81/test_c1.py).
V8_TO_C1 = [
    ('    model, info = build(a.seed, device, fit_hist.tolist())\n',
     '    model, info = build_c1(a.seed, device, fit_hist.tolist())  # [C1] trainable mask: last 2 Q-former blocks + ln_vision + query tokens + head\n'),
    ('"batch_plan_sha256": [__import__("hashlib").sha256(json.dumps(p).encode()).hexdigest() for p in plans]}\n',
     '"batch_plan_sha256": [__import__("hashlib").sha256(json.dumps(p).encode()).hexdigest() for p in plans], "variant": C1_VARIANT}  # [C1] log field\n'),
    ('        set_train(model); t0 = time.time(); agg = Counter(); n_steps = 0; seen = 0\n',
     '        set_train_c1(model); t0 = time.time(); agg = Counter(); n_steps = 0; seen = 0  # [C1] frozen Q-former eval, last 2 blocks train\n'),
    ('        os.replace(unit / "resume.tmp", resume)\n',
     '        os.replace(unit / "resume.tmp", resume)\n        save_epoch_trainable(model, unit, epoch)  # [C1] extra artifact, never used for selection\n'),
    ('json.dumps({"unit": unit.name, "selected_epoch": best[1], "cal_select_mse": best[0], "gpu": log["gpu"]}, indent=1)',
     'json.dumps({"unit": unit.name, "selected_epoch": best[1], "cal_select_mse": best[0], "gpu": log["gpu"], "variant": C1_VARIANT}, indent=1)'),
]


def v8_cmd_train_source() -> str:
    src = (HERE / "v8_src" / "v8" / "v8_train.py").read_text(); i = src.index("def cmd_train(a):"); j = src.index("\n\nclass CandRows")
    return src[i:j].rstrip("\n") + "\n"


def derive_c1_source(v8_src: str) -> str:
    out = v8_src
    for old, new in V8_TO_C1:
        if out.count(old) != 1:
            raise ValueError(f"substitution source must occur exactly once: {old[:60]!r} ({out.count(old)})")
        out = out.replace(old, new)
    return out


# ----------------------------------------------------------------------------------------------------------------------------- the C1 mask
def c1_blocks(model) -> list:
    """The last N_BLOCKS modules of the real Q-former encoder ModuleList (by enumeration, not by name)."""
    import torch
    layers = model.Qformer.bert.encoder.layer
    if not isinstance(layers, torch.nn.ModuleList) or len(layers) < N_BLOCKS:
        raise SystemExit(f"Q-former encoder layer list not found or too short: {type(layers)}")
    return list(layers)[-N_BLOCKS:]


def apply_c1_mask(model) -> dict:
    """Set requires_grad for C1 and return the inventory (names, shapes, counts) — model as built by v8_train.build (V8 mask applied)."""
    blocks = c1_blocks(model); layers = list(model.Qformer.bert.encoder.layer)
    idx = [k for k, m in enumerate(layers) if any(m is b for b in blocks)]
    block_ids = {id(p) for b in blocks for p in b.parameters()}
    v8_mask = {n: bool(p.requires_grad) for n, p in model.named_parameters()}
    keep = lambda n: n.startswith(("ln_vision.", "v8_head.")) or n == "query_tokens"
    for n, p in model.named_parameters():
        p.requires_grad_(id(p) in block_ids or keep(n))
    tensors = []
    for n, p in model.named_parameters():
        grp = ("c1_block" if id(p) in block_ids else "ln_vision" if n.startswith("ln_vision.") else "head" if n.startswith("v8_head.") else "query_tokens"
               if n == "query_tokens" else "qformer_frozen" if n.startswith("Qformer.") else "vit" if n.startswith("visual_encoder.") else "other_frozen")
        tensors.append({"name": n, "shape": list(p.shape), "numel": int(p.numel()), "trainable_c1": bool(p.requires_grad), "trainable_v8": v8_mask[n], "group": grp})
    block_names = sorted({t["name"] for t in tensors if t["group"] == "c1_block"})
    prefixes = sorted({".".join(n.split(".")[:5]) for n in block_names})
    tot = lambda f: int(sum(t["numel"] for t in tensors if f(t)))
    inv = {"variant": C1_VARIANT, "n_encoder_layers": len(layers), "c1_block_indices": idx, "c1_block_name_prefixes": prefixes,
           "c1_blocks_have_cross_attention": [bool(getattr(b, "has_cross_attention", False)) for b in blocks],
           "n_trainable_c1": tot(lambda t: t["trainable_c1"]), "n_trainable_v8": tot(lambda t: t["trainable_v8"]), "v8_trainable_expected": V8_TRAINABLE_TOTAL,
           "n_trainable_tensors_c1": sum(t["trainable_c1"] for t in tensors), "n_trainable_tensors_v8": sum(t["trainable_v8"] for t in tensors),
           "by_group": {g: {"tensors": sum(1 for t in tensors if t["group"] == g), "params": tot(lambda t, g=g: t["group"] == g),
                            "trainable_c1": tot(lambda t, g=g: t["group"] == g and t["trainable_c1"])} for g in sorted({t["group"] for t in tensors})},
           "newly_frozen_vs_v8": tot(lambda t: t["trainable_v8"] and not t["trainable_c1"]), "newly_trainable_vs_v8": tot(lambda t: t["trainable_c1"] and not t["trainable_v8"]),
           "tensors": tensors}
    inv["checks"] = {"block_names_match_indices": prefixes == [f"Qformer.bert.encoder.layer.{k}" for k in idx],
                     "c1_subset_of_v8": inv["newly_trainable_vs_v8"] == 0, "v8_total_matches_audit": inv["n_trainable_v8"] == V8_TRAINABLE_TOTAL,
                     "vit_frozen": tot(lambda t: t["group"] == "vit" and t["trainable_c1"]) == 0}
    return inv


def build_c1(seed: int, device, fit_hist):
    model, info = T.build(seed, device, fit_hist); inv = apply_c1_mask(model)
    if not all(inv["checks"].values()):
        raise SystemExit(f"C1 inventory checks failed: {inv['checks']}")
    info["c1"] = {k: v for k, v in inv.items() if k != "tensors"}
    info["n_trainable_v8"] = int(sum(p.numel() for p in model.parameters() if p.requires_grad))   # key name kept from V8; now the C1 count
    return model, info


def set_train_c1(model):
    """Every module eval (frozen Q-former blocks / embeddings, ViT, unused heads), then the two trainable blocks + ln_vision + head in train mode."""
    model.eval()
    for b in c1_blocks(model):
        b.train()
    model.ln_vision.train(); model.v8_head.train()


def save_epoch_trainable(model, unit: Path, epoch: int):
    d = unit / "epochs"; d.mkdir(exist_ok=True); import torch
    st = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    tmp = d / f"epoch{epoch}.tmp"; torch.save({"trainable": st, "epoch": epoch, "variant": C1_VARIANT}, tmp); os.replace(tmp, d / f"epoch{epoch}.pt")


def mode_report(model) -> dict:
    """Training flags as seen during C1 training (after set_train_c1)."""
    import torch
    blocks = c1_blocks(model); bid = {id(m) for b in blocks for m in b.modules()}
    drop = [(m, id(m) in bid) for m in model.Qformer.modules() if isinstance(m, torch.nn.Dropout)]
    return {"visual_encoder": model.visual_encoder.training, "Qformer_root": model.Qformer.training, "embeddings": model.Qformer.bert.embeddings.training,
            "encoder_layers": [m.training for m in model.Qformer.bert.encoder.layer], "ln_vision": model.ln_vision.training, "v8_head": model.v8_head.training,
            "dropout_in_c1_blocks_training": sorted({m.training for m, inb in drop if inb}), "dropout_elsewhere_training": sorted({m.training for m, inb in drop if not inb}),
            "n_dropout_in_c1_blocks": sum(1 for _, inb in drop if inb), "n_dropout_elsewhere": sum(1 for _, inb in drop if not inb)}


# ------------------------------------------------------------------------------------------------------------- V8 cmd_train with V8_TO_C1
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
    model, info = build_c1(a.seed, device, fit_hist.tolist())  # [C1] trainable mask: last 2 Q-former blocks + ln_vision + query tokens + head
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
           "batch_plan_sha256": [__import__("hashlib").sha256(json.dumps(p).encode()).hexdigest() for p in plans], "variant": C1_VARIANT}  # [C1] log field
    resume = unit / "resume.pt"
    if resume.exists():
        ck = torch.load(resume, map_location="cpu", weights_only=False); model.load_state_dict(ck["model"], strict=False); opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"]); start, best, log, step = ck["epoch"] + 1, tuple(ck["best"]), ck["log"], ck["step"]; log.setdefault("resumed_at", []).append(start)
    col = Collate(fit, images, proc); pt = torch.tensor(prior)
    for epoch in range(start, E):
        loader = DataLoader(plans[epoch], batch_size=None, shuffle=False, collate_fn=col, num_workers=a.workers, persistent_workers=False,
                            prefetch_factor=4 if a.workers else None)
        set_train_c1(model); t0 = time.time(); agg = Counter(); n_steps = 0; seen = 0  # [C1] frozen Q-former eval, last 2 blocks train
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
        save_epoch_trainable(model, unit, epoch)  # [C1] extra artifact, never used for selection
    log.update({"selected_epoch": best[1], "selected_cal_select_mse": best[0], "seconds": time.time() - t_start, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"})
    (unit / "log.json").write_text(json.dumps(log, indent=1))
    if not a.max_steps and E == EPOCHS:
        (unit / "done.json").write_text(json.dumps({"unit": unit.name, "selected_epoch": best[1], "cal_select_mse": best[0], "gpu": log["gpu"], "variant": C1_VARIANT}, indent=1))
    return 0


# --------------------------------------------------------------------------------------------------------------------- other commands
def parent_lr(arm: str, seed: int) -> float:
    return float(json.loads((V8_UNITS / f"V8-FIT-{arm}-s{seed}" / "log.json").read_text())["lr"])


def cmd_export(a):
    """V8's export unchanged (selected best.pt, eval mode); only EvalMuse development roles."""
    bad = [r for r in a.roles if r not in ("CAL_SELECT", "CAL_PROB", "DEV")]
    if bad:
        raise SystemExit(f"C1 exports development roles only, not {bad}")
    return T.main(["export", "--unit-dir", a.unit_dir, "--roles", *a.roles])


def cmd_audit(a):
    """FIT-AUDIT predictions (q / u / mu) of best.pt and resume.pt, eval mode, V8's predict on the full model (no cache)."""
    import torch
    from transformers import BertTokenizerFast
    unit = Path(a.unit_dir); log = json.loads((unit / "log.json").read_text()); device = "cuda" if torch.cuda.is_available() else "cpu"; prior = log["prior_fit"]
    VC.stage_workdir(VC.BIG_OUT / "work"); tok = BertTokenizerFast.from_pretrained(str(VC.BERT_DIR))
    rows = LB.load(T.DATA); prompts = {r["image_id"]: r["prompt"] for r in rows}; R = LB.records(rows)
    ids = json.loads(Path(a.ids).read_text()); order = {i: k for k, i in enumerate(ids["image_ids"])}
    recs = sorted([r for r in R["FIT"] if r["image_id"] in order], key=lambda r: order[r["image_id"]])
    if len(recs) != len(order):
        raise SystemExit("FIT-AUDIT ids missing among valid FIT records")
    data = T.RatingData(recs, tok, prompts); model, _ = T.build(log["seed"], device, log["fit_hist"])
    VC.import_lavis()
    from lavis.processors.blip_processors import BlipImageEvalProcessor
    proc = BlipImageEvalProcessor(image_size=364); images = VC.ArchiveImages(); out = unit / "pred"; out.mkdir(exist_ok=True); res = {"fit_audit_ids_sha256": ids["image_ids_sha256"]}
    for k in ("best", "resume"):
        ck = torch.load(unit / f"{k}.pt", map_location="cpu", weights_only=False); msg = model.load_state_dict(ck["model"], strict=False)
        if [m for m in msg.missing_keys if not m.startswith("visual_encoder.")] or msg.unexpected_keys:
            raise SystemExit(f"{k}.pt does not match the model: {msg}")
        pred = predict(model, data, images, proc, device, prior); res[k] = {"epoch": ck["epoch"], "metrics": cal_metrics(pred, data, prior)}
        (out / f"fit_audit_{k}.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pred))
    (out / "fit_audit.json").write_text(json.dumps(res, indent=1)); print(f"[c1 audit] {unit.name}: best e{res['best']['epoch']} resume e{res['resume']['epoch']}", flush=True); return 0


def cmd_inventory(a):
    VC.stage_workdir(VC.BIG_OUT / "work"); R = LB.records(LB.load(T.DATA))
    fit_hist = ((np.array([r["w"] for r in R["FIT"]])[:, None] * np.array([r["hist"] for r in R["FIT"]])).sum(0) / sum(r["w"] for r in R["FIT"])).tolist()
    model, _ = T.build(0, "cpu", fit_hist); inv = apply_c1_mask(model)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(inv, indent=1))
    print(json.dumps({k: v for k, v in inv.items() if k != "tensors"}, indent=1)); return 0 if all(inv["checks"].values()) else 1


def cmd_selftest(a):
    """Real stack: inventory, training-mode flags, one optimiser step (frozen bitwise unchanged, trainable changed, frozen grads None),
    then V8-mask vs C1-mask s/step + peak memory on the same GPU (epoch-0 plan batches, AMP + GradScaler as in training, isolated model)."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import BertTokenizerFast
    device = "cuda" if torch.cuda.is_available() else "cpu"; VC.stage_workdir(VC.BIG_OUT / "work"); tok = BertTokenizerFast.from_pretrained(str(VC.BERT_DIR))
    D, R = load_role_data(tok, limit=8); fit = D["FIT"]; prior = LB.fit_priors(R["FIT"]).tolist(); kind = C.ARMS[a.arm]
    fit_hist = (np.array([r["w"] for r in R["FIT"]])[:, None] * np.array([r["hist"] for r in R["FIT"]])).sum(0) / sum(r["w"] for r in R["FIT"])
    model, info = build_c1(a.seed, device, fit_hist.tolist()); res = {"gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu", "torch": torch.__version__,
                                                                      "arm": a.arm, "seed": a.seed, "inventory": info["c1"]}
    set_train_c1(model); res["modes_during_training"] = mode_report(model)
    VC.import_lavis()
    from lavis.processors.blip_processors import BlipImageEvalProcessor
    proc = BlipImageEvalProcessor(image_size=364); images = VC.ArchiveImages(); col = Collate(fit, images, proc); plan = C.batch_manifest(fit.prompt_images, a.seed, 0)
    pt = torch.tensor(prior, device=device)
    before = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    opt = optimizer(model, 1e-4); b = col(plan[0])
    with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
        logits = forward_logits(model, b, device)
    loss, _, _ = C.batch_loss(logits.float(), b["hist"].to(device), b["w"].to(device), pt, kind, b["subgroup"], b["n_sub"]); loss.backward()
    frozen_grad_none = all(p.grad is None for p in model.parameters() if not p.requires_grad)
    gnorm = {n: (float(p.grad.detach().float().norm()) if p.grad is not None else None) for n, p in model.named_parameters() if p.requires_grad}
    opt.step(); opt.zero_grad(set_to_none=True)
    after = {n: p.detach().cpu() for n, p in model.named_parameters()}; tr = {n for n, p in model.named_parameters() if p.requires_grad}
    fro_same = [n for n in before if n not in tr and torch.equal(before[n], after[n])]; tr_changed = [n for n in tr if not torch.equal(before[n], after[n])]
    zero_grad = sorted(n for n, g in gnorm.items() if g == 0.0); unchanged = sorted(tr - set(tr_changed))
    res["one_step"] = {"loss": float(loss), "n_frozen": len(before) - len(tr), "n_frozen_bitwise_unchanged": len(fro_same), "n_trainable": len(tr), "n_trainable_changed": len(tr_changed),
                       "trainable_unchanged": unchanged, "trainable_zero_grad": zero_grad, "trainable_grad_none": sorted(n for n, g in gnorm.items() if g is None),
                       "zero_grad_params": int(sum(dict(model.named_parameters())[n].numel() for n in zero_grad)), "frozen_grads_none": frozen_grad_none,
                       "note": "a trainable tensor may stay unchanged only if its gradient is exactly zero (no path to the pooled query outputs; matrices still move by weight decay)"}
    res["one_step"]["pass"] = (res["one_step"]["n_frozen_bitwise_unchanged"] == res["one_step"]["n_frozen"] and frozen_grad_none and not res["one_step"]["trainable_grad_none"]
                               and set(unchanged) <= set(zero_grad))
    del before, after
    timing = {}
    for variant in ("V8", "C1"):
        for n, p in model.named_parameters():                                            # restore the mask of the variant on the same object
            p.requires_grad_((n.startswith(("Qformer.", "ln_vision.", "v8_head.")) or n == "query_tokens") if variant == "V8" else False)
        if variant == "C1":
            apply_c1_mask(model)
        opt = optimizer(model, 1e-6); scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda"); ts = []
        loader = DataLoader(plan[:a.n_warm + a.n_time], batch_size=None, shuffle=False, collate_fn=col, num_workers=6, prefetch_factor=4)
        (T.set_train if variant == "V8" else set_train_c1)(model); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t = time.time()
        for i, b in enumerate(loader):
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = forward_logits(model, b, device)
            loss, _, _ = C.batch_loss(logits.float(), b["hist"].to(device), b["w"].to(device), pt, kind, b["subgroup"], b["n_sub"])
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); torch.cuda.synchronize()
            now = time.time(); ts.append(now - t); t = now
        st = ts[a.n_warm:]
        timing[variant] = {"n_trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)), "s_per_step_mean": float(np.mean(st)), "s_per_step_median": float(np.median(st)),
                           "n_timed": len(st), "peak_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
    res["timing_same_gpu"] = timing; res["steps_per_epoch"] = len(plan)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "inventory"}, indent=1)); return 0 if res["one_step"]["pass"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--arm", required=True, choices=sorted(C.ARMS)); t.add_argument("--lr", required=True, help="a float or 'parent'")
    t.add_argument("--seed", type=int, required=True); t.add_argument("--unit-dir", required=True); t.add_argument("--workers", type=int, default=6)
    t.add_argument("--max-steps", type=int, default=0); t.add_argument("--eval-n", type=int, default=None); t.add_argument("--epochs", type=int, default=EPOCHS)
    x = sub.add_parser("export"); x.add_argument("--unit-dir", required=True); x.add_argument("--roles", nargs="+", default=["CAL_SELECT", "CAL_PROB", "DEV"])
    u = sub.add_parser("audit"); u.add_argument("--unit-dir", required=True)
    u.add_argument("--ids", default=str(HERE.parents[1] / "reports" / "V8_1_DIAGNOSTICS" / "results" / "fit_audit_ids.json"))
    s = sub.add_parser("selftest"); s.add_argument("--out", required=True); s.add_argument("--arm", default="U-CS"); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--n-warm", type=int, default=10); s.add_argument("--n-time", type=int, default=50)
    i = sub.add_parser("inventory"); i.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "train":
        a.lr = parent_lr(a.arm, a.seed) if a.lr == "parent" else float(a.lr)
        print(f"[c1] {a.arm} s{a.seed} lr {a.lr:g} ({C1_VARIANT})", flush=True)
    return {"train": cmd_train, "export": cmd_export, "audit": cmd_audit, "selftest": cmd_selftest, "inventory": cmd_inventory}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
