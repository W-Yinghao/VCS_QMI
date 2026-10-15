#!/usr/bin/env python
"""V8.1 fork 3 — parameter-level audit of the 24 formal V8 checkpoints (CPU only; read-only on every V8 artefact).

Fills the gap fork 1 recorded in epoch_audit.csv ("successful_updates: unknown — AMP GradScaler skip count not recorded; scaler state only inside
resume.pt, not read here") and FULL_PLAN_CN §03 §4 ("本轮检查的是实际freeze mask、模块mode和AMP成功更新") with the checkpoints themselves:

  * RECONSTRUCTED_INIT per seed: V8's build(seed) on CPU (read-only copy scripts/v81/v8_src); the trainable set and the AdamW parameter grouping are
    rebuilt with V8's own rule (v8_train.build / v8_train.optimizer).
  * per checkpoint (best.pt, resume.pt) and parameter group (head, ln_vision, query tokens, Q-former embeddings, per layer self-attention /
    cross-attention / query FFN / text FFN, LM head, frozen modules): ||theta - theta_init||_2, relative change, max |delta|, tensors bit-identical
    to the reconstruction.  Frozen (requires_grad False) tensors and buffers must be bit-identical.
  * resume.pt optimiser / scaler: AdamW per-parameter step counts vs the global iteration counter (steps skipped by the fp16 GradScaler = global
    iterations - AdamW steps), trainable parameters that never received a gradient (no AdamW state), final lr / weight decay per group,
    GradScaler final scale and growth tracker; best vs resume distance; resume's best field and resumed_at vs log.json.
  * The frozen EVA ViT-g is never saved in a V8 checkpoint (trainable_state excludes visual_encoder.*): every V8 load rebuilds it from the public
    BLIP-2 COCO weights, so it cannot have changed; this is recorded, not measured.

    python -I scripts/v81/v81_fork3_param_audit.py [--out reports/V8_1_DIAGNOSTICS/results/param_audit.json] [--units U1 U2 ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import re
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "v8_src" / "v8")); sys.path.insert(0, str(HERE / "v8_src" / "v7"))

UNITS = Path("/projects/EEG-foundation-model/yinghao/outputs_v8/units")
OUT_DEFAULT = HERE.parents[1] / "reports" / "V8_1_DIAGNOSTICS" / "results" / "param_audit.json"
ARMS = ["U-CS", "U-BCE", "U-RPS", "U-CE", "U-MSE", "U-L1", "U-RANK", "U-LIST"]
SEEDS = [0, 1, 2]
LAYER_RE = re.compile(r"^Qformer\.bert\.encoder\.layer\.(\d+)\.(.+)$")
COARSE = ("trainable_all", "head", "ln_vision", "query_tokens", "qformer_all", "qf.embeddings", "qformer_self_attn_all", "qformer_cross_attn_all",
          "qformer_ffn_query_all", "qformer_ffn_text_all", "qf.cls_head", "qf.other", "frozen_all")


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(msg):
    print(f"[param-audit {now()}] {msg}", flush=True)


def group_of(name: str, trainable: bool) -> str:
    if not trainable:
        return "frozen:" + name.split(".")[0]
    if name.startswith("v8_head."):
        return "head"
    if name.startswith("ln_vision."):
        return "ln_vision"
    if name == "query_tokens":
        return "query_tokens"
    if name.startswith("Qformer.bert.embeddings."):
        return "qf.embeddings"
    if name.startswith("Qformer.cls."):
        return "qf.cls_head"
    m = LAYER_RE.match(name)
    if m:
        i, rest = int(m.group(1)), m.group(2)
        if rest.startswith("crossattention."):
            kind = "cross_attn"
        elif rest.startswith("attention."):
            kind = "self_attn"
        elif rest.startswith(("intermediate_query.", "output_query.")):
            kind = "ffn_query"
        elif rest.startswith(("intermediate.", "output.")):
            kind = "ffn_text"
        else:
            kind = "other"
        return f"qf.L{i:02d}.{kind}"
    if name.startswith("Qformer."):
        return "qf.other"
    return "trainable_other"


def keys_of(group: str) -> list:
    """The fine group plus the coarse aggregates it belongs to."""
    if group.startswith("frozen:"):
        return [group, "frozen_all"]
    out = [group, "trainable_all"]
    if group.startswith("qf."):
        out.append("qformer_all"); parts = group.split(".")
        if len(parts) == 3 and parts[1].startswith("L"):
            out.append(f"qformer_{parts[2]}_all")
    return out


def compare(state: dict, ref: dict, trainable: set) -> dict:
    """Group statistics of state - ref over the tensors of ref (float64 accumulation)."""
    import torch
    acc = {}
    for n, t0 in ref.items():
        if n not in state:
            continue
        t1 = state[n]
        if tuple(t1.shape) != tuple(t0.shape):
            raise SystemExit(f"shape mismatch for {n}: {tuple(t1.shape)} vs {tuple(t0.shape)}")
        for key in keys_of(group_of(n, n in trainable)):
            a = acc.setdefault(key, {"n_params": 0, "n_tensors": 0, "n_identical": 0, "ss_ref": 0.0, "ss_delta": 0.0, "max_abs_delta": 0.0, "nonidentical_examples": []})
            same = bool(torch.equal(t0, t1)); a["n_params"] += t0.numel(); a["n_tensors"] += 1; a["n_identical"] += int(same)
            if t0.is_floating_point():
                d = t1.double() - t0.double(); a["ss_ref"] += float((t0.double() ** 2).sum()); a["ss_delta"] += float((d ** 2).sum())
                a["max_abs_delta"] = max(a["max_abs_delta"], float(d.abs().max()) if d.numel() else 0.0)
            elif not same:
                a["max_abs_delta"] = max(a["max_abs_delta"], float("inf"))
            if not same and len(a["nonidentical_examples"]) < 3 and key.startswith("frozen"):
                a["nonidentical_examples"].append(n)
    for a in acc.values():
        a["norm_ref"] = math.sqrt(a["ss_ref"]); a["norm_delta"] = math.sqrt(a["ss_delta"]); a["rel_change"] = a["norm_delta"] / a["norm_ref"] if a["ss_ref"] > 0 else None
        del a["ss_ref"], a["ss_delta"]
    return acc


def optimizer_audit(ck: dict, decay: list, no_decay: list, shapes: dict, trainable_groups: dict) -> dict:
    """AdamW state of resume.pt mapped back to parameter names with V8's grouping rule; step counts vs the global iteration counter."""
    osd = ck["opt"]; pgs = osd["param_groups"]
    if len(pgs) != 2 or len(pgs[0]["params"]) != len(decay) or len(pgs[1]["params"]) != len(no_decay):
        raise SystemExit(f"optimiser groups do not match V8's rule: {[len(g['params']) for g in pgs]} vs {len(decay)} / {len(no_decay)}")
    names = dict(zip(pgs[0]["params"], decay)); names.update(zip(pgs[1]["params"], no_decay)); st = osd["state"]
    steps, no_state, shape_bad = {}, [], []
    for pid, n in names.items():
        s = st.get(pid)
        if s is None:
            no_state.append(n); continue
        if tuple(s["exp_avg"].shape) != shapes[n]:
            shape_bad.append(n)
        steps[n] = float(s["step"])
    if shape_bad:
        raise SystemExit(f"AdamW state shapes do not match the parameters for {shape_bad[:3]} — mapping failed")
    glob = int(ck["step"]); cnt = Counter(steps.values()); top = max(steps.values()) if steps else None
    no_state_groups = Counter(trainable_groups[n] for n in no_state)
    sc = ck.get("scaler") or {}
    return {"global_iterations": glob, "adamw_step_values": {str(int(k)): v for k, v in sorted(cnt.items())}, "adamw_step_max": top,
            "skipped_by_gradscaler": (glob - int(top)) if top is not None else None, "n_trainable_tensors": len(names), "n_tensors_with_state": len(steps),
            "n_tensors_never_stepped": len(no_state), "never_stepped_groups": dict(no_state_groups), "never_stepped_examples": no_state[:5],
            "param_groups": [{"n_tensors": len(g["params"]), "lr_final": g.get("lr"), "weight_decay": g.get("weight_decay"), "betas": list(g.get("betas", [])), "eps": g.get("eps"),
                              "initial_lr": g.get("initial_lr")} for g in pgs],
            "gradscaler": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in sc.items()}}


def main(argv=None):
    import numpy as np
    import torch
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(OUT_DEFAULT)); ap.add_argument("--units", nargs="*", default=None); a = ap.parse_args(argv)
    import v8_labels as LB
    import v8_train as T
    import v7_common as VC
    t_start = time.time(); VC.stage_workdir(VC.BIG_OUT / "work")
    R = LB.records(LB.load(T.DATA)); fit = R["FIT"]
    fit_hist = ((np.array([r["w"] for r in fit])[:, None] * np.array([r["hist"] for r in fit])).sum(0) / sum(r["w"] for r in fit)).tolist()
    units = a.units or [f"V8-FIT-{arm}-s{s}" for arm in ARMS for s in SEEDS]; seeds = sorted({int(u.rsplit("-s", 1)[1]) for u in units})
    inits, meta = {}, {}
    for s in seeds:
        t0 = time.time(); m, info = T.build(s, "cpu", fit_hist)
        trainable = {n for n, p in m.named_parameters() if p.requires_grad}
        decay = [n for n, p in m.named_parameters() if p.requires_grad and p.ndim >= 2 and n != "query_tokens"]
        no_decay = [n for n, p in m.named_parameters() if p.requires_grad and not (p.ndim >= 2 and n != "query_tokens")]
        shapes = {n: tuple(p.shape) for n, p in m.named_parameters()}
        inits[s] = {k: v.detach().clone() for k, v in T.trainable_state(m).items()}
        meta[s] = {"trainable": trainable, "decay": decay, "no_decay": no_decay, "shapes": shapes,
                   "n_trainable_params": int(sum(p.numel() for p in m.parameters() if p.requires_grad)),
                   "n_vit_params": int(sum(p.numel() for n, p in m.named_parameters() if n.startswith("visual_encoder."))),
                   "vit_requires_grad_any": bool(any(p.requires_grad for n, p in m.named_parameters() if n.startswith("visual_encoder."))),
                   "init_missing_keys": info.get("missing_keys"), "init_unexpected_keys": info.get("unexpected_keys"), "seconds": time.time() - t0}
        del m; log(f"RECONSTRUCTED_INIT seed {s}: {len(inits[s])} tensors, {meta[s]['n_trainable_params']} trainable params ({meta[s]['seconds']:.0f}s)")
    tg = {n: group_of(n, True) for n in meta[seeds[0]]["trainable"]}
    out = {"env": {"host": platform.node(), "torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(), "utc_start": now()},
           "note_vit": "visual_encoder.* is never saved in a V8 checkpoint; every V8 load rebuilds it from the public BLIP-2 COCO weights (requires_grad False)",
           "init_meta": {str(s): {k: v for k, v in meta[s].items() if k not in ("trainable", "decay", "no_decay", "shapes")} | {"n_trainable_tensors": len(meta[s]["trainable"]),
                                                                                                                          "n_decay_tensors": len(meta[s]["decay"]), "n_no_decay_tensors": len(meta[s]["no_decay"])}
                         for s in seeds},
           "trainable_groups": dict(sorted(Counter(tg.values()).items())), "units": {}}
    for u in units:
        t0 = time.time(); seed = int(u.rsplit("-s", 1)[1]); d = UNITS / u; L = json.loads((d / "log.json").read_text()); init = inits[seed]; M = meta[seed]; rec = {"seed": seed, "arm": L["arm"], "lr": L["lr"]}
        states = {}
        for k in ("best", "resume"):
            ck = torch.load(d / f"{k}.pt", map_location="cpu", weights_only=False); st = ck["model"]
            missing = sorted(set(init) - set(st)); extra = sorted(set(st) - set(init))
            rec[k] = {"epoch": ck.get("epoch"), "n_tensors": len(st), "missing_vs_init": missing[:5], "n_missing_vs_init": len(missing), "extra_vs_init": extra[:5], "n_extra_vs_init": len(extra),
                      "vs_init": compare(st, init, M["trainable"])}
            if k == "resume":
                rec[k].update({"global_step": ck.get("step"), "best_field": ck.get("best"), "resumed_at": (ck.get("log") or {}).get("resumed_at"),
                               "optimizer": optimizer_audit(ck, M["decay"], M["no_decay"], M["shapes"], tg)})
            states[k] = st; del ck
        rec["resume_vs_best"] = compare(states["resume"], states["best"], M["trainable"]); rec["log_selected_epoch"] = L.get("selected_epoch"); rec["log_resumed_at"] = L.get("resumed_at")
        rec["seconds"] = time.time() - t0; out["units"][u] = rec; del states
        o = rec["resume"]["optimizer"]; vb = rec["best"]["vs_init"]
        log(f"{u}: best e{rec['best']['epoch']} rel(trainable) {vb['trainable_all']['rel_change']:.3e} rel(cross) {vb['qformer_cross_attn_all']['rel_change']:.3e} "
            f"rel(head) {vb['head']['rel_change']:.3e}; frozen identical {vb['frozen_all']['n_identical']}/{vb['frozen_all']['n_tensors']}; AdamW steps {o['adamw_step_values']} of "
            f"{o['global_iterations']} (skipped {o['skipped_by_gradscaler']}), never stepped {o['n_tensors_never_stepped']}; scale {o['gradscaler'].get('scale')} ({rec['seconds']:.0f}s)")
    out["env"]["utc_end"] = now(); out["seconds"] = time.time() - t_start
    p = Path(a.out); p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_suffix(".tmp"); tmp.write_text(json.dumps(out, indent=1)); os.replace(tmp, p)
    rows = []
    for u, rec in out["units"].items():
        for k in ("best", "resume"):
            for g, s in rec[k]["vs_init"].items():
                rows.append({"unit": u, "arm": rec["arm"], "seed": rec["seed"], "lr": rec["lr"], "checkpoint": k, "epoch": rec[k]["epoch"], "reference": "RECONSTRUCTED_INIT", "group": g,
                             **{c: s[c] for c in ("n_params", "n_tensors", "n_identical", "norm_ref", "norm_delta", "rel_change", "max_abs_delta")}})
        for g, s in rec["resume_vs_best"].items():
            rows.append({"unit": u, "arm": rec["arm"], "seed": rec["seed"], "lr": rec["lr"], "checkpoint": "resume", "epoch": rec["resume"]["epoch"], "reference": f"best (e{rec['best']['epoch']})", "group": g,
                         **{c: s[c] for c in ("n_params", "n_tensors", "n_identical", "norm_ref", "norm_delta", "rel_change", "max_abs_delta")}})
    with open(p.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
    log(f"done: {len(out['units'])} units -> {p} (+ .csv, {len(rows)} rows) in {out['seconds']:.0f}s"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
