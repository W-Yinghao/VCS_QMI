"""G0 CUDA preflight on FIT for the AGIQA-3K alignment pilot (server wrapper; package code unchanged).

Package requirements covered (INSTALL_CN.md section 4, SERVER_HANDOFF_CN.md section 7 G0, SERVER_PROTOCOL_CN.md section 5 G0):
real checkpoint + real FIT images; forward/backward; unfrozen layers; finite gradients; output range; one optimizer update;
representation gradients after a nonzero head update; W change; full-epoch time; checkpoint bytes; 1024/768/1024 shapes;
batch 5 fits in memory. Families: the six pilot families of `aigiqa.cli make_grid --stage pilot`, at lambda 0.1 (Task-only 0)
and seed 3101, plus a short finite check at lambda 100 for every regularized family.

Data roles: only FIT rows are used for computation. CAL and TEST are never opened (verify_prepared hashes the FIT/CAL tables
exactly as the package does; the CAL-sized inference timing uses FIT rows). No CAL metric is computed. Nothing is written
into registered run directories; this job writes only to --out and to node-local tmp (checkpoint-size probe, then deleted).
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import re
import statistics
import tempfile
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from aigiqa.cli import make_grid
from aigiqa.data import IQADataset
from aigiqa.engine import make_loader, predict_loader, preflight, seed_all, synchronize, environment
from aigiqa.losses import IQALosses
from aigiqa.manifest import INPUT_FIELDS, read_csv, sha256_file, verify_prepared, write_csv
from aigiqa.models import build_model
from vcs_common.dependence import DependenceRegularizer, parameter_groups
from vcs_common.runtime import runtime_record

FAMILIES = ["task_only", "reference_nce_text", "bce_text", "vcs_exact_text", "vcs_sampled8_text", "corrected_cmim_both"]
SEED = 3101


class Checks:
    def __init__(self):
        self.items = []

    def add(self, name, ok, **detail):
        self.items.append({"check": name, "pass": bool(ok), **detail})
        print(("PASS " if ok else "FAIL ") + name + ("" if ok else f" {json.dumps(detail, default=str)[:600]}"), flush=True)
        return ok

    @property
    def ok(self):
        return all(i["pass"] for i in self.items)


def tensor_stats(t):
    t = t.detach().float()
    return {"min": float(t.min()), "max": float(t.max()), "mean": float(t.mean()), "finite": bool(torch.isfinite(t).all())}


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def device_sanity(c, device):
    p = torch.cuda.get_device_properties(device)
    arch = f"sm_{p.major}{p.minor}"
    compiled = torch.cuda.get_arch_list()
    # SASS runs on the same major with an equal or higher minor (sm_86 cubins serve L40S sm_89); the matmul below is the real gate.
    sass = [s for s in compiled if s.startswith("sm_") and int(s[3:]) // 10 == p.major and int(s[3:]) % 10 <= p.minor]
    c.add("cuda_arch_binary_compatible_build", bool(sass), arch=arch, arch_list=compiled)
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(512, 512, generator=g, dtype=torch.float64), torch.randn(512, 512, generator=g, dtype=torch.float64)
    ref = a @ b
    got = (a.float().to(device) @ b.float().to(device)).double().cpu()
    err = float((got - ref).abs().max() / ref.abs().max())
    c.add("cuda_fp32_matmul_vs_fp64_reference", err < 5e-3, rel_max_err=err)  # loose: TF32 default may apply on some types
    return {"name": p.name, "capability": [p.major, p.minor], "total_memory": p.total_memory, "arch": arch, "matmul_rel_err": err}


def weight_load_audit(c, cfg):
    """ImageReward.load uses torch.load (weights_only default of this torch) and load_state_dict(strict=False)."""
    res = {}
    ckpt, med = cfg["model"]["checkpoint"], cfg["model"]["med_config"]
    t0 = time.time()
    try:
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        res["torch_load_weights_only_true"] = "OK"
    except Exception as exc:  # recorded; ImageReward.load would fail the same way under this torch default
        res["torch_load_weights_only_true"] = f"{type(exc).__name__}: {exc}"[:800]
        c.add("imagereward_checkpoint_loads_with_torch_default_weights_only", False, error=res["torch_load_weights_only_true"])
        return res
    c.add("imagereward_checkpoint_loads_with_torch_default_weights_only", True)
    res["load_seconds"] = time.time() - t0
    res["n_tensors"] = len(sd)
    res["n_parameters_in_checkpoint"] = int(sum(v.numel() for v in sd.values()))
    res["dtypes"] = sorted({str(v.dtype) for v in sd.values()})
    from ImageReward.ImageReward import ImageReward as RewardModel
    model = RewardModel(med_config=med, device="cpu")
    msg = model.load_state_dict(sd, strict=False)
    res["missing_keys"], res["unexpected_keys"] = list(msg.missing_keys), list(msg.unexpected_keys)
    loaded = model.state_dict()
    res["value_mismatch_after_load"] = [k for k, v in loaded.items() if k in sd and not torch.equal(v, sd[k])]
    benign = re.compile(r"(^|\.)position_ids$")
    serious_missing = [k for k in res["missing_keys"] if not benign.search(k)]
    serious_unexpected = [k for k in res["unexpected_keys"] if not benign.search(k)]
    c.add("imagereward_all_checkpoint_weights_load", not serious_missing and not serious_unexpected and not res["value_mismatch_after_load"],
          missing=serious_missing[:20], unexpected=serious_unexpected[:20], mismatch=res["value_mismatch_after_load"][:20])
    del model, sd, loaded
    cleanup()
    return res


def trainable_audit(c, model, losses):
    reward = model.reward
    text_layers, vis_blocks = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    other_reward = defaultdict(lambda: [0, 0])
    for name, p in reward.named_parameters():
        m_t = re.match(r"blip\.text_encoder\..*?layer\.(\d+)\.", name)
        m_v = re.match(r"blip\.visual_encoder\.blocks\.(\d+)\.", name)
        key = 1 if p.requires_grad else 0
        if m_t:
            text_layers[int(m_t.group(1))][key] += p.numel()
        elif m_v:
            vis_blocks[int(m_v.group(1))][key] += p.numel()
        else:
            other_reward[name.split(".")[0] + "." + name.split(".")[1] if "." in name else name][key] += p.numel()
    text_ok = all((v[1] > 0 and v[0] == 0) if i >= 6 else (v[1] == 0 and v[0] > 0) for i, v in text_layers.items()) and len(text_layers) == 12
    vis_ok = all((v[1] > 0 and v[0] == 0) if i >= 12 else (v[1] == 0 and v[0] > 0) for i, v in vis_blocks.items()) and len(vis_blocks) == 24
    norm_trainable = all(p.requires_grad for n, p in reward.blip.visual_encoder.named_parameters() if n.startswith("norm."))
    others_frozen = [n for n, p in reward.named_parameters() if p.requires_grad
                     and not re.match(r"blip\.text_encoder\..*?layer\.(\d+)\.", n) and not re.match(r"blip\.visual_encoder\.blocks\.(\d+)\.", n)
                     and not n.startswith("blip.visual_encoder.norm.")]
    new_modules = {name: (sum(p.numel() for p in mod.parameters()), all(p.requires_grad for p in mod.parameters()))
                   for name, mod in [("image_to_text", model.image_to_text), ("text_cross_attention", model.text_cross_attention), ("fusion", model.fusion)]}
    c.add("unfrozen_text_layers_6_to_11_only", text_ok, layers={k: v for k, v in sorted(text_layers.items())})
    c.add("unfrozen_visual_blocks_12_to_23_only", vis_ok, blocks={k: v for k, v in sorted(vis_blocks.items())})
    c.add("visual_final_norm_trainable", norm_trainable)
    c.add("no_other_imagereward_parameter_trainable", not others_frozen, trainable=others_frozen[:20])
    c.add("new_cmim_modules_fully_trainable", all(ok for _, ok in new_modules.values()), modules=new_modules)
    return {"text_layers_frozen_trainable": {k: v for k, v in sorted(text_layers.items())},
            "visual_blocks_frozen_trainable": {k: v for k, v in sorted(vis_blocks.items())},
            "other_imagereward_groups_frozen_trainable": dict(other_reward), "new_modules_params_trainable": new_modules,
            "model_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "model_frozen": sum(p.numel() for p in model.parameters() if not p.requires_grad),
            "regularizer_trainable": sum(p.numel() for p in losses.parameters() if p.requires_grad)}


def build(cfg):
    """engine._train's construction: model, losses, AdamW with the same parameter groups and learning rates."""
    seed_all(cfg["seed"])
    device = cfg.get("device", "cuda")
    t0 = time.time()
    model = build_model(cfg["model"], device)
    losses = IQALosses(model, cfg["loss"]).to(device)
    synchronize(device)
    build_s = time.time() - t0
    model_groups = parameter_groups(model, weight_decay=cfg.get("weight_decay", .001))
    aux_groups = parameter_groups(losses, weight_decay=cfg.get("weight_decay", .001))
    for group in aux_groups:
        group["lr"] = cfg.get("regularizer_learning_rate", .001)
    opt = torch.optim.AdamW(model_groups + aux_groups, lr=cfg.get("learning_rate", 1e-5), betas=(.9, .999), eps=1e-8)
    return model, losses, opt, build_s


def fit_loader(cfg, info, workers=None):
    data = IQADataset(Path(cfg["data_dir"]) / "fit.csv", info["image_root"], cfg["target"], cfg.get("image_size", 224))
    return data, make_loader(data, cfg.get("batch_size", 5), cfg.get("num_workers", 0) if workers is None else workers, True, cfg["seed"])


def dependence_heads(losses):
    return {b: r for b, r in losses.regularizers.items() if isinstance(r, DependenceRegularizer)}


def one_step(model, losses, opt, batch, device, feature_grads=True):
    opt.zero_grad(set_to_none=True)
    output = model(batch["image"].to(device), batch["prompt"])
    losses.prepare_step(output)
    total, task, dep = losses(output, batch["label"].to(device))
    grads = {}
    if feature_grads:
        tg = torch.autograd.grad(task, output["fusion"], retain_graph=True, allow_unused=True)[0]
        dg = torch.autograd.grad(dep, output["fusion"], retain_graph=True, allow_unused=True)[0] if dep.requires_grad else None
        grads = {"fusion_task_grad_norm": 0. if tg is None else float(tg.norm()),
                 "fusion_dependence_grad_norm": 0. if dg is None else float(dg.norm())}
    total.backward()
    combined = list(model.parameters()) + list(losses.parameters())
    finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in combined) and bool(torch.isfinite(total))
    with_grad = sum(1 for p in combined if p.requires_grad and p.grad is not None)
    nonzero = sum(1 for p in combined if p.requires_grad and p.grad is not None and bool(p.grad.abs().sum() > 0))
    opt.step()
    losses.project_()
    return output, {"total": float(total.detach()), "task_mse": float(task.detach()), "dependence": float(dep.detach()),
                    "finite": finite, "trainable_tensors_with_grad": with_grad, "trainable_tensors_nonzero_grad": nonzero,
                    "prediction": tensor_stats(output["prediction"]), "label_mos_over_5": tensor_stats(batch["label"]),
                    "shapes": {k: list(v.shape) for k, v in output.items()}, **grads,
                    "diagnostics": {k: v for k, v in losses.diagnostics.items() if isinstance(v, (int, float))}}


def timed_epoch(model, losses, opt, loader, device):
    """Replica of engine._train's per-step body (no CAL pass, no scheduler, no checkpoint): FIT epoch wall time."""
    model.train(); losses.train()
    combined = nn.ModuleDict({"model": model, "losses": losses})
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    step_times, sums, counts = [], defaultdict(float), defaultdict(int)
    synchronize(device)
    start = time.perf_counter()
    last = start
    for step, batch in enumerate(loader):
        opt.zero_grad(set_to_none=True)
        output = model(batch["image"].to(device), batch["prompt"])
        losses.prepare_step(output)
        total, task, dependence = losses(output, batch["label"].to(device))
        if step == min(1, len(loader) - 1):
            task_grad = torch.autograd.grad(task, output["fusion"], retain_graph=True, allow_unused=True)[0]
            dep_grad = torch.autograd.grad(dependence, output["fusion"], retain_graph=True, allow_unused=True)[0] if dependence.requires_grad else None
            sums["fusion_task_grad_norm"] += 0. if task_grad is None else float(task_grad.detach().norm())
            sums["fusion_dependence_grad_norm"] += 0. if dep_grad is None else float(dep_grad.detach().norm())
            counts["fusion_task_grad_norm"] += 1; counts["fusion_dependence_grad_norm"] += 1
        total.backward()
        if not torch.isfinite(total) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in combined.parameters()):
            raise FloatingPointError(f"Nonfinite G0 training at step={step}")
        opt.step()
        losses.project_()
        for name, value in losses.diagnostics.items():
            if isinstance(value, (float, int)) and np.isfinite(value):
                sums[name] += float(value); counts[name] += 1
        now = time.perf_counter(); step_times.append(now - last); last = now
    synchronize(device)
    seconds = time.perf_counter() - start
    st = sorted(step_times)
    return {"steps": len(step_times), "seconds": seconds, "step_seconds_median": statistics.median(st), "step_seconds_p90": st[int(.9 * (len(st) - 1))],
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda") else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if str(device).startswith("cuda") else None,
            "mean_diagnostics": {k: sums[k] / counts[k] for k in sums}}


def warmup_epoch(model, losses, cfg, loader, device):
    """Replica of engine._train's Gaussian warm-up (corrected_cmim_both, 1 epoch) for timing."""
    warm_opt = torch.optim.AdamW(losses.forward_regularizer.parameters(), lr=cfg.get("warmup_learning_rate", 5e-6), weight_decay=cfg.get("weight_decay", .001))
    model.eval()
    synchronize(device)
    start = time.perf_counter()
    for batch in loader:
        with torch.no_grad():
            output = model(batch["image"].to(device), batch["prompt"])
        warm_opt.zero_grad(set_to_none=True)
        value = losses.forward_regularizer(output["visual"], output["text"])
        value.backward()
        warm_opt.step()
    synchronize(device)
    return {"seconds": time.perf_counter() - start, "last_value_finite": bool(torch.isfinite(value))}


def inference_timing(model, cfg, info, n_rows, tmp, device):
    """Label-free inference over the first n_rows FIT images (CAL-sized; CAL itself is not opened)."""
    rows = read_csv(Path(cfg["data_dir"]) / "fit.csv")[:n_rows]
    path = Path(tmp) / "fit_subset_inputs.csv"
    write_csv(path, rows, INPUT_FIELDS)
    data = IQADataset(path, info["image_root"], None, cfg.get("image_size", 224))
    loader = make_loader(data, cfg.get("batch_size", 5), cfg.get("num_workers", 0), False, cfg["seed"])
    synchronize(device)
    start = time.perf_counter()
    preds = predict_loader(model, loader, device)
    synchronize(device)
    vals = torch.tensor([r["prediction"] for r in preds])
    return {"n_images": len(preds), "seconds": time.perf_counter() - start, "prediction": tensor_stats(vals)}


def checkpoint_probe(model, losses, cfg, target_dir):
    """Same dict as engine._train's best.pt, saved on the filesystem the runs write to (G0 out dir), timed, then deleted."""
    path = Path(target_dir) / "best_probe.pt"
    start = time.perf_counter()
    torch.save({"model": model.state_dict(), "losses": losses.state_dict(), "config": cfg, "epoch": 1,
                "cal_metrics": {"note": "G0 probe; no CAL metric computed"},
                "prepared_manifest_sha256": sha256_file(Path(cfg["data_dir"]) / "manifest.json")}, path)
    with open(path, "rb+") as f:
        os.fsync(f.fileno())
    seconds = time.perf_counter() - start
    size = path.stat().st_size
    path.unlink()
    return {"bytes": size, "save_seconds_target_fs": seconds, "target_dir": str(target_dir)}


def family_run(c, fam, cfg, cfg_hi, info, n_cal, tmp, device, quick, out):
    res = {"config": cfg}
    t0 = time.time()
    try:
        res["package_preflight"] = preflight(copy.deepcopy(cfg))
        res["package_preflight_seconds"] = time.time() - t0
        c.add(f"{fam}:package_preflight_PASS", res["package_preflight"].get("status") == "PASS")
        shapes = res["package_preflight"]["shapes"]
        c.add(f"{fam}:feature_shapes_1024_768_1024", shapes.get("visual", [0, 0])[1] == 1024 and shapes.get("text", [0, 0])[1] == 768
              and shapes.get("fusion", [0, 0])[1] == 1024, shapes=shapes)
    except Exception as exc:
        c.add(f"{fam}:package_preflight_PASS", False, error=f"{type(exc).__name__}: {exc}"[:800], tb=traceback.format_exc()[-1500:])
        return res
    cleanup()
    model, losses, opt, build_s = build(cfg)
    res["model_build_seconds_incl_sha256"] = build_s
    res["trainable"] = trainable_audit(c, model, losses)
    data, loader = fit_loader(cfg, info)
    if cfg.get("gaussian_warmup_epochs", 0):
        res["gaussian_warmup_epoch"] = warmup_epoch(model, losses, cfg, loader, device)
        c.add(f"{fam}:gaussian_warmup_finite", res["gaussian_warmup_epoch"]["last_value_finite"])
    model.train(); losses.train()
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    trainable = {n: p.detach().clone() for n, p in list(model.named_parameters()) + [("losses." + n, p) for n, p in losses.named_parameters()] if p.requires_grad}
    heads = dependence_heads(losses)
    w_before = {b: r.W.detach().clone() for b, r in heads.items()}
    it = iter(loader)
    _, step1 = one_step(model, losses, opt, next(it), device)
    params_now = dict(list(model.named_parameters()) + [("losses." + n, p) for n, p in losses.named_parameters()])
    changed = sum(1 for n, v in trainable.items() if not torch.equal(params_now[n].detach(), v))
    frozen_changed = [n for n, v in frozen.items() if not torch.equal(params_now[n].detach(), v)]
    res["step1"] = step1 | {"trainable_tensors": len(trainable), "trainable_tensors_changed": changed, "frozen_tensors_changed": frozen_changed[:10]}
    c.add(f"{fam}:step1_finite", step1["finite"] and step1["prediction"]["finite"])
    c.add(f"{fam}:step1_frozen_bitwise_unchanged", not frozen_changed, changed=frozen_changed[:10])
    c.add(f"{fam}:step1_trainable_updated", changed > 0 and step1["trainable_tensors_nonzero_grad"] > 0, changed=changed)
    if heads:
        w = {b: {"changed": not torch.equal(r.W.detach(), w_before[b]), "abs_max": float(r.W.detach().abs().max()),
                 "fro_delta": float((r.W.detach() - w_before[b]).norm())} for b, r in heads.items()}
        res["step1"]["W"] = w
        c.add(f"{fam}:W_changed_and_boxed_after_project", all(v["changed"] and v["abs_max"] <= 1.0 for v in w.values()), W=w)
        c.add(f"{fam}:step1_zero_head_feature_grad_is_zero_for_dependence", step1["fusion_dependence_grad_norm"] == 0.0,
              norm=step1["fusion_dependence_grad_norm"])
    del frozen, trainable
    cleanup()
    _, step2 = one_step(model, losses, opt, next(it), device)
    del it  # release the step iterator's workers before the timed epoch
    res["step2_after_nonzero_head"] = step2
    c.add(f"{fam}:step2_finite", step2["finite"])
    if fam != "task_only":
        c.add(f"{fam}:step2_dependence_feature_grad_nonzero", step2["fusion_dependence_grad_norm"] > 0, norm=step2["fusion_dependence_grad_norm"])
    c.add(f"{fam}:task_feature_grad_nonzero", step2["fusion_task_grad_norm"] > 0)
    if not quick:
        try:
            res["timed_fit_epoch"] = timed_epoch(model, losses, opt, loader, device)
            c.add(f"{fam}:full_fit_epoch_finite", True)
        except FloatingPointError as exc:
            c.add(f"{fam}:full_fit_epoch_finite", False, error=str(exc))
        res["fit_images"], res["fit_batches"] = len(data), len(loader)
        res["cal_sized_inference"] = inference_timing(model, cfg, info, n_cal, tmp, device)
        res["checkpoint_probe"] = checkpoint_probe(model, losses, cfg, out)
    del model, losses, opt, loader, data
    cleanup()
    if cfg_hi is not None:
        model, losses, opt, _ = build(cfg_hi)
        _, loader = fit_loader(cfg_hi, info, workers=0)
        it, hi = iter(loader), []
        for _ in range(3):
            _, s = one_step(model, losses, opt, next(it), device, feature_grads=False)
            hi.append({k: s[k] for k in ("total", "task_mse", "dependence", "finite")})
        res["lambda100_three_steps"] = hi
        c.add(f"{fam}:lambda100_three_steps_finite", all(s["finite"] for s in hi), steps=hi)
        del model, losses, opt, loader
        cleanup()
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-config", required=True, help="server base config (package base_alignment.json with staged paths/hashes)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--families", nargs="+", default=FAMILIES)
    ap.add_argument("--quick", action="store_true", help="skip timed epoch / inference / checkpoint probe")
    ap.add_argument("--synthetic-dry-run", action="store_true",
                    help="CPU wiring test on synthetic weights/images (g0_cpu_dryrun.py); never a G0 result")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"G0 output must be fresh: {out}")
    out.mkdir(parents=True)
    c = Checks()
    started = time.time()
    record = {"schema": "aigiqa_server_g0_v1", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "slurm": {k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURMD_NODENAME", "CUDA_VISIBLE_DEVICES")},
              "runtime": runtime_record(), "package_environment": environment(), "families": {}}
    base = json.loads(Path(a.base_config).read_text())
    device = base.get("device", "cuda:0")
    if a.synthetic_dry_run:
        record["schema"] = "aigiqa_server_g0_SYNTHETIC_DRY_RUN_not_a_G0_result"
    elif not str(device).startswith("cuda"):
        raise SystemExit("G0 must run on CUDA; CPU only with --synthetic-dry-run")
    if str(device).startswith("cuda"):
        c.add("cuda_available", torch.cuda.is_available())
        record["device"] = device_sanity(c, device)
    g0_base = copy.deepcopy(base)
    g0_base["output_dir"] = str(out / "unused_run_root")  # G0 never writes run outputs; keeps registered run dirs untouched
    (out / "g0_base_config.json").write_text(json.dumps(g0_base, indent=2) + "\n")
    record["grid"] = make_grid(out / "g0_base_config.json", out / "grid", "pilot")  # the package's own generator
    configs = {p.stem: json.loads(p.read_text()) for p in (out / "grid").glob("*.json") if p.name != "grid_summary.json"}
    c.add("package_make_grid_pilot_78", record["grid"]["run_count"] == 78, run_count=record["grid"]["run_count"])
    info = verify_prepared(base["data_dir"])
    n_cal = info["image_counts"]["cal"]
    record["prepared"] = {"manifest_sha256": sha256_file(Path(base["data_dir"]) / "manifest.json"), "image_counts": info["image_counts"],
                          "split_group_counts": info["split_group_counts"], "image_root": info["image_root"]}
    record["weight_load_audit"] = weight_load_audit(c, base)
    with tempfile.TemporaryDirectory(prefix="aig_g0_") as tmp:
        for fam in a.families:
            lam = "0" if fam == "task_only" else "0.1"
            cfg = configs[f"{fam}_w{lam}_s{SEED}"]
            cfg_hi = None if fam == "task_only" else configs[f"{fam}_w100_s{SEED}"]
            t0 = time.time()
            try:
                record["families"][fam] = family_run(c, fam, cfg, cfg_hi, info, n_cal, tmp, device, a.quick, out)
            except Exception as exc:  # recorded as a failed check; the remaining families still run and G0_RESULT is written
                c.add(f"{fam}:family_completed_without_exception", False, error=f"{type(exc).__name__}: {exc}"[:800],
                      tb=traceback.format_exc()[-2000:])
                record["families"][fam] = {"config": cfg, "exception": f"{type(exc).__name__}: {exc}"[:800]}
                cleanup()
            record["families"][fam]["family_seconds"] = time.time() - t0
            (out / f"family_{fam}.json").write_text(json.dumps(record["families"][fam], indent=1, default=str) + "\n")
    record["checks"] = c.items
    record["status"] = ("DRYRUN_" if a.synthetic_dry_run else "") + ("PASS" if c.ok else "FAIL")
    record["seconds"] = time.time() - started
    record["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / "G0_RESULT.json").write_text(json.dumps(record, indent=1, default=str) + "\n")
    print(json.dumps({"status": record["status"], "failed": [i["check"] for i in c.items if not i["pass"]], "seconds": record["seconds"]}, indent=1))
    raise SystemExit(0 if c.ok else 1)


if __name__ == "__main__":
    main()
