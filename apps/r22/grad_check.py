"""One real FIT batch on the real models: which modules the r22 auxiliary gradient reaches (GPU smoke, before training).

KD (VCS-CRD): the weighted term must reach the student and both embedding heads, never the teacher; memory rows are
buffers, updated only at the batch's own indices. AIG (bank / MOS-weighted P, alternating_target_sg): the critic fit
changes only the critic; the outer weighted term must reach the fusion input, not the text input (target SG), not the
critic; the bank grows by one batch and carries no gradient. Raises on any failed check; writes a JSON record.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch


def _norms(names_params, grads):
    out = defaultdict(float)
    for (name, _), g in zip(names_params, grads):
        if g is not None:
            out[name.split(".")[0]] += float(g.detach().float().square().sum())
    return {k: v ** .5 for k, v in sorted(out.items())}


def kd(cfg_path, out):
    from kd.runner import read_config, _teacher, _device, _loader
    from kd.data import read_manifest, RoleDataset
    from kd.models import create_model, feature_dim
    from kd.losses import DistillationLoss
    from kd.crd_adapter import OfficialCRD
    cfg = read_config(cfg_path)
    manifest = read_manifest(cfg["data"]["manifest"])
    device = _device(cfg)
    torch.manual_seed(cfg["seed"])
    fit = RoleDataset(cfg["data"]["root"], manifest, "fit", augment=True, seed=cfg["seed"])
    student = create_model(cfg["student"], manifest["num_classes"], cfg["seed"])
    sdim = feature_dim(student)
    teacher, tdim, _ = _teacher(cfg, manifest, device)
    student.to(device).train()
    losses = DistillationLoss(sdim, tdim, cfg["loss"]).to(device)
    losses.crd = OfficialCRD(sdim, tdim, fit.ids, fit.labels[fit.ids].tolist(), cfg["loss"]["crd"], cfg["seed"]).to(device)
    losses.train()
    images, labels, ids = next(iter(_loader(fit, cfg, train=True, epoch=0)))
    images, labels = images.to(device), labels.to(device)
    logits_s, feat_s = student(images)
    with torch.no_grad():
        logits_t, feat_t = teacher(images)
    mem = losses.crd.loss.contrast
    before = (mem.memory_v1.clone(), mem.memory_v2.clone())
    anchor, _ = losses(logits_s, logits_t, feat_s, feat_t, labels, 0)
    aux = cfg["loss"]["crd_weight"] * losses.crd(feat_s, feat_t, ids)
    groups = {"student": list(student.named_parameters()),
              "embed_s": list(losses.crd.loss.embed_s.named_parameters()),
              "embed_t": list(losses.crd.loss.embed_t.named_parameters())}
    flat = [(g + "." + n, p) for g, ps in groups.items() for n, p in ps]
    grads = torch.autograd.grad(aux, [p for _, p in flat], allow_unused=True, retain_graph=True)
    norms = _norms(flat, grads)
    g_anchor = torch.autograd.grad(anchor, feat_s, retain_graph=True)[0]
    g_aux = torch.autograd.grad(aux, feat_s, retain_graph=True)[0]
    pos = torch.as_tensor([losses.crd.lookup[int(i)] for i in ids], device=device)
    changed = ((mem.memory_v1 != before[0]).any(1) | (mem.memory_v2 != before[1]).any(1))
    mask = torch.zeros_like(changed); mask[pos] = True
    checks = {
        "finite": bool(torch.isfinite(aux)) and bool(torch.isfinite(anchor)),
        "student_grad": norms.get("student", 0.) > 0, "embed_s_grad": norms.get("embed_s", 0.) > 0,
        "embed_t_grad": norms.get("embed_t", 0.) > 0,
        "teacher_no_grad": all(not p.requires_grad and p.grad is None for p in teacher.parameters()),
        "teacher_features_no_grad": not feat_t.requires_grad,
        "memory_is_buffer": not mem.memory_v1.requires_grad and not mem.memory_v2.requires_grad,
        "memory_updated_only_at_batch_rows": bool(changed[mask].all()) and not bool(changed[~mask].any()),
        "objective_is_vcs": losses.crd.objective == "vcs"}
    record = {"app": "kd", "config": str(cfg_path), "checks": checks, "grad_norms_weighted_aux": norms,
              "pre_classifier_grad_norm": {"anchor_ce_plus_dkd": float(g_anchor.norm()), "weighted_aux": float(g_aux.norm())},
              "aux_over_anchor": float(g_aux.norm() / g_anchor.norm()), "parts": losses.crd.last_parts,
              "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu"}
    return record


def aig(cfg, out):
    from aigiqa.engine import build_losses, make_loader, validate_config
    from aigiqa.manifest import verify_prepared
    from aigiqa.data import IQADataset
    from aigiqa.models import build_model
    validate_config(cfg)
    info = verify_prepared(cfg["data_dir"])
    device = cfg.get("device", "cuda")
    torch.manual_seed(cfg["seed"])
    model = build_model(cfg["model"], device)
    losses = build_losses(model, cfg, device)
    data = IQADataset(Path(cfg["data_dir"]) / "fit.csv", info["image_root"], cfg["target"], cfg.get("image_size", 224))
    batch = next(iter(make_loader(data, cfg.get("batch_size", 5), 0, True, cfg["seed"])))
    model.train(); losses.train()
    epoch = cfg["loss"]["application_update"]["warmup_epochs"]          # first epoch with a nonzero host weight
    losses.application_update.set_epoch(epoch)
    reg = losses.regularizers["text"]
    critic = list(reg.named_parameters())
    critic_before = [p.detach().clone() for _, p in critic]
    bank_before = int(getattr(reg, "bank_count", torch.tensor(-1)))
    output = model(batch["image"].to(device), batch["prompt"])
    labels = batch["label"].to(device)
    losses.prepare_step(output, labels)
    host = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    critic_changed = any(not torch.equal(p.detach(), b) for (_, p), b in zip(critic, critic_before))
    host_grad_after_fit = any(p.grad is not None for _, p in host)
    total, task, dependence = losses(output, labels)
    weighted = losses.application_update.weighted(dependence)
    g_fusion = torch.autograd.grad(weighted, output["fusion"], allow_unused=True, retain_graph=True)[0]
    g_text = (torch.autograd.grad(weighted, output["text"], allow_unused=True, retain_graph=True)[0]
              if output["text"].requires_grad else None)
    g_critic = torch.autograd.grad(weighted, [p for _, p in critic], allow_unused=True, retain_graph=True) if weighted.requires_grad else []
    g_host = torch.autograd.grad(weighted, [p for _, p in host], allow_unused=True, retain_graph=True)
    host_norms = _norms(host, g_host)
    checks = {
        "finite": bool(torch.isfinite(total)) and bool(torch.isfinite(dependence)),
        "host_weight_positive": losses.application_update.weight > 0,
        "critic_fit_changed_critic": critic_changed, "critic_fit_no_host_grad": not host_grad_after_fit,
        "outer_reaches_fusion": g_fusion is not None and float(g_fusion.norm()) > 0,
        # x = fusion(visual, text), so text still gets aux gradient THROUGH x (shared host path, intended);
        # the direct target edge is cut: with x detached the outer term has no graph to text at all.
        "outer_no_direct_text_edge": not losses.application_update.outer_loss(
            "text", output["fusion"].detach(), output["text"]).requires_grad,
        "outer_no_critic_grad": all(g is None for g in g_critic),
        "outer_reaches_host_parameters": sum(host_norms.values()) > 0}
    if hasattr(reg, "bank"):
        checks["bank_grew_by_batch"] = int(reg.bank_count) == min(bank_before + len(labels), reg.bank_size)
        checks["bank_no_grad"] = not reg.bank.requires_grad
    record = {"app": "aigiqa", "config_output_dir": cfg["output_dir"], "checks": checks,
              "weighted_aux_grad_norm_by_host_module": host_norms,
              "fusion_grad_norm": float(g_fusion.norm()) if g_fusion is not None else 0.,
              "text_grad_norm_via_fusion": float(g_text.norm()) if g_text is not None else 0.,
              "task_grad_norm_fusion": float(torch.autograd.grad(task, output["fusion"], retain_graph=True)[0].norm()),
              "diagnostics": {k: v for k, v in losses.diagnostics.items() if isinstance(v, (int, float))},
              "device": torch.cuda.get_device_name() if str(device).startswith("cuda") else "cpu"}
    return record


def run(app, cfg, cfg_path, out):
    record = kd(cfg_path, out) if app == "kd" else aig(cfg, out)
    record["status"] = "PASS" if all(record["checks"].values()) else "FAIL"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"grad_check": record["status"], "checks": record["checks"]}))
    if record["status"] != "PASS":
        raise SystemExit(f"grad check FAIL: {record['checks']}")
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
