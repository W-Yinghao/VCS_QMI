"""Actual training, CAL selection, immutable freeze and once-only TEST evaluation."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import RoleDataset, canonical_hash, digest_file, read_manifest, write_json
from .losses import DistillationLoss
from .models import PINS, create_model, feature_dim, load_model_checkpoint


def environment():
    return {"python": platform.python_version(), "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "cuda_available": torch.cuda.is_available(),
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "precision_policy": "float32_no_autocast_pytorch_tf32_defaults_unchanged",
            "upstream_expected_pins": PINS}


def read_config(path):
    cfg = json.loads(Path(path).read_text())
    cfg = copy.deepcopy(cfg)
    allowed = {"schema", "name", "seed", "device", "threads", "data", "teacher", "student", "loss", "optimizer", "training", "track", "notes"}
    extra = set(cfg) - allowed
    if extra:
        raise ValueError(f"Unknown top-level config fields: {extra}")
    if cfg.get("schema") != "kd_config_v1":
        raise ValueError("Expected kd_config_v1")
    if cfg.get("track") not in ("strict_fit_teacher", "public_teacher_cal_exposed", "synthetic_smoke"):
        raise ValueError("An explicit teacher exposure track is required")
    for field in ("root", "manifest"):
        cfg["data"][field] = str(Path(cfg["data"][field]).expanduser().resolve())
    for which in ("teacher", "student"):
        model_cfg = cfg[which]
        for field in ("source_root", "checkpoint", "provenance"):
            if model_cfg.get(field):
                model_cfg[field] = str(Path(model_cfg[field]).expanduser().resolve())
    if cfg["loss"].get("crd"):
        cfg["loss"]["crd"]["source_root"] = str(Path(cfg["loss"]["crd"]["source_root"]).expanduser().resolve())
    if cfg["training"]["epochs"] < 1 or cfg["training"]["batch_size"] < 2:
        raise ValueError("At least one epoch and batch size >=2 required")
    if cfg["optimizer"].get("head_weight_decay", 0.) != 0.:
        raise ValueError("Matched dependency heads must have zero weight decay")
    if cfg["loss"].get("dependence", {}).get("q_mode", "u") not in ("u", "v"):
        raise ValueError("Q must be u or v; no label-filtered negatives in the matched comparison")
    return cfg


def source_digest():
    """Freeze code affecting training, including sibling vcs_common if present."""
    h = hashlib.sha256()
    root = Path(__file__).resolve().parent.parent
    for folder in (root / "kd", root / "vcs_common", root / "shared" / "vcs_common"):
        if folder.exists():
            for p in sorted(folder.rglob("*.py")):
                if "references" in p.parts or "tests" in p.parts:
                    continue
                h.update(str(p.relative_to(root)).encode()); h.update(p.read_bytes())
    return h.hexdigest()


def state_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _device(cfg):
    device = torch.device(cfg.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no automatic CPU fallback for real experiments")
    if cfg.get("threads"):
        torch.set_num_threads(int(cfg["threads"]))
    return device


def _loader(dataset, cfg, train=False, epoch=0):
    dataset.set_epoch(epoch)
    g = torch.Generator().manual_seed(cfg["seed"] + 3571 * epoch + (0 if train else 19))
    return DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=train,
                      drop_last=train, num_workers=cfg["training"].get("workers", 4),
                      pin_memory=cfg.get("device", "cpu").startswith("cuda"), persistent_workers=False, generator=g)


@torch.no_grad()
def infer(model, dataset, cfg, device):
    model.eval()
    logits_all, labels_all, ids_all = [], [], []
    start = time.perf_counter()
    for images, labels, ids in _loader(dataset, cfg):
        logits, _ = model(images.to(device))
        logits_all.append(logits.cpu()); labels_all.append(labels); ids_all.append(ids)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    logits, labels, ids = torch.cat(logits_all), torch.cat(labels_all), torch.cat(ids_all)
    probs = logits.softmax(1)
    conf, pred = probs.max(1)
    correct = pred.eq(labels)
    top5 = logits.topk(min(5, logits.shape[1]), 1).indices.eq(labels[:, None]).any(1)
    nll = -probs[torch.arange(len(labels)), labels].clamp_min(torch.finfo(probs.dtype).tiny).log()
    target_onehot = torch.nn.functional.one_hot(labels, logits.shape[1]).float()
    ece = 0.
    for lower in torch.linspace(0., 1., 16)[:-1]:
        selected = (conf >= lower) & (conf < lower + 1/15 if lower < 14/15 else conf <= 1)
        if selected.any():
            ece += float(selected.float().mean() * (correct[selected].float().mean() - conf[selected].mean()).abs())
    metrics = {"n": len(labels), "top1": float(correct.float().mean()), "top5": float(top5.float().mean()),
               "nll": float(nll.mean()), "brier": float((probs - target_onehot).square().sum(1).mean()), "ece_15bins": ece,
               "inference_loader_wall_seconds": elapsed,
               "metric_directions": {"top1": "higher", "top5": "higher", "nll": "lower", "brier": "lower", "ece_15bins": "lower"}}
    rows = [{"id": int(ids[i]), "label": int(labels[i]), "prediction": int(pred[i]), "correct": int(correct[i]),
             "confidence": float(conf[i]), "nll": float(nll[i]), "probabilities": probs[i].tolist()} for i in range(len(labels))]
    return metrics, rows


def write_predictions(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("id", "label", "prediction", "correct", "confidence", "nll", "probabilities"))
        writer.writeheader()
        for row in rows:
            row = dict(row); row["probabilities"] = json.dumps(row["probabilities"], separators=(",", ":"))
            writer.writerow(row)


def _teacher(cfg, manifest, device):
    tc = cfg["teacher"]
    if not tc.get("checkpoint") or not Path(tc["checkpoint"]).is_file():
        raise FileNotFoundError("Teacher checkpoint is required; train a FIT-only teacher first. No random-teacher fallback.")
    if not tc.get("provenance"):
        raise ValueError("Teacher provenance JSON is mandatory")
    provenance = json.loads(Path(tc["provenance"]).read_text())
    sha = digest_file(tc["checkpoint"])
    if provenance.get("checkpoint_sha256") != sha:
        raise ValueError("Teacher checkpoint SHA differs from provenance")
    if provenance.get("architecture") != tc["name"]:
        raise ValueError("Teacher architecture/provenance mismatch")
    if cfg["track"] == "strict_fit_teacher":
        if provenance.get("training_role") != "FIT" or provenance.get("split_sha256") != manifest["split_sha256"]:
            raise ValueError("Primary track requires the exact FIT-only teacher for this manifest")
        if provenance.get("test_used_for_selection", True):
            raise ValueError("Primary teacher provenance must exclude TEST selection")
    elif cfg["track"] == "public_teacher_cal_exposed":
        if provenance.get("training_role") != "official_full_train_CAL_exposed":
            raise ValueError("Published teacher track must explicitly disclose CAL exposure")
    model = create_model(tc, manifest["num_classes"], seed=cfg["seed"] + 999)
    load_model_checkpoint(model, tc["checkpoint"])
    dim = feature_dim(model)
    model.to(device).eval().requires_grad_(False)
    return model, dim, provenance


def _train_impl(config_path, output, teacher_training=False, resume=False):
    cfg = read_config(config_path)
    manifest = read_manifest(cfg["data"]["manifest"])
    if cfg["track"] == "synthetic_smoke" and manifest["dataset"] != "synthetic_smoke":
        raise ValueError("Smoke track cannot consume a real dataset")
    if manifest["dataset"] == "synthetic_smoke" and cfg["track"] != "synthetic_smoke":
        raise ValueError("Synthetic results must be explicitly labeled smoke")
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    config_hash = canonical_hash(cfg)
    if (output / "run.json").exists() and not resume:
        raise FileExistsError("Existing run: choose a new output or --resume an unfinished identical run")
    if resume and not (output / "last.pt").exists():
        raise FileNotFoundError("Epoch-boundary last.pt required for resume")
    device = _device(cfg)
    torch.manual_seed(cfg["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg["seed"])
    fit = RoleDataset(cfg["data"]["root"], manifest, "fit", augment=True, seed=cfg["seed"])
    cal = RoleDataset(cfg["data"]["root"], manifest, "cal")
    model_cfg = cfg["teacher"] if teacher_training else cfg["student"]
    student = create_model(model_cfg, manifest["num_classes"], cfg["seed"])
    student_dim = feature_dim(student)
    initial_sha = state_digest(student)
    teacher, teacher_dim, provenance = None, student_dim, None
    loss_cfg = {"anchor": "none", "feature": "none", "ce_weight": 1.0} if teacher_training else cfg["loss"]
    needs_teacher = (loss_cfg.get("anchor", "none") != "none" or loss_cfg.get("feature", "none") != "none"
                     or loss_cfg.get("dependence", {}).get("method", "none") != "none" or bool(loss_cfg.get("crd")))
    teacher_initial = None
    if needs_teacher:
        teacher, teacher_dim, provenance = _teacher(cfg, manifest, device)
        teacher_initial = state_digest(teacher)
    student.to(device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg["seed"] + 12345)
        losses = DistillationLoss(student_dim, teacher_dim, loss_cfg).to(device)
        if loss_cfg.get("crd"):
            from .crd_adapter import OfficialCRD
            losses.crd = OfficialCRD(student_dim, teacher_dim, fit.ids,
                                    fit.labels[fit.ids].tolist(), loss_cfg["crd"], cfg["seed"]).to(device)
    optcfg = cfg["teacher"].get("optimizer", cfg["optimizer"]) if teacher_training else cfg["optimizer"]
    groups = [{"params": list(student.parameters()), "weight_decay": optcfg["weight_decay"]}]
    head_params = [p for p in losses.parameters() if p.requires_grad]
    head_kind = optcfg.get("head_optimizer", "sgd")
    if head_kind not in ("sgd", "adamw"):
        raise ValueError("head_optimizer must be sgd or adamw")
    if head_params and head_kind == "sgd":
        groups.append({"params": head_params, "weight_decay": 0., "lr": optcfg.get("head_lr", optcfg["lr"])})
    optimizer = torch.optim.SGD(groups, lr=optcfg["lr"], momentum=optcfg.get("momentum", .9), nesterov=False)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=optcfg.get("milestones", [150, 180, 210]), gamma=optcfg.get("gamma", .1))
    head_optimizer = torch.optim.AdamW(head_params, lr=optcfg.get("head_lr", .001), weight_decay=0.) if head_params and head_kind == "adamw" else None
    head_scheduler = torch.optim.lr_scheduler.MultiStepLR(head_optimizer, milestones=optcfg.get("milestones", [150, 180, 210]), gamma=optcfg.get("gamma", .1)) if head_optimizer else None
    run = {"schema": "kd_run_v1", "status": "running", "config": cfg, "config_sha256": config_hash,
           "source_sha256": source_digest(), "split_sha256": manifest["split_sha256"], "environment": environment(),
           "teacher_training": teacher_training, "teacher_provenance": provenance, "teacher_initial_state_sha256": teacher_initial,
           "student_initial_state_sha256": initial_sha, "train_data_role": "FIT", "selection_data_role": "CAL",
           "test_loaded_during_train": False, "parameters_student": sum(p.numel() for p in student.parameters()),
           "parameters_auxiliary": sum(p.numel() for p in losses.parameters()),
           "teacher_selection_rule": "fixed_final_epoch" if teacher_training else None,
           "student_selection_rule": "max_CAL_top1_then_min_CAL_nll_then_earliest_epoch"}
    history, first_epoch, best_key = [], 0, (-float("inf"), -float("inf"))
    if resume:
        old = json.loads((output / "run.json").read_text())
        if old["status"] == "complete":
            raise ValueError("Completed runs are immutable; use a new output")
        if old["config_sha256"] != config_hash or old["source_sha256"] != run["source_sha256"] or old["split_sha256"] != run["split_sha256"]:
            raise ValueError("Resume config/source/split mismatch")
        if old.get("teacher_initial_state_sha256") != teacher_initial:
            raise ValueError("Teacher changed before resume")
        ckpt = torch.load(output / "last.pt", map_location=device, weights_only=True)
        student.load_state_dict(ckpt["student"]); losses.load_state_dict(ckpt["losses"])
        optimizer.load_state_dict(ckpt["optimizer"]); scheduler.load_state_dict(ckpt["scheduler"])
        if head_optimizer is not None:
            head_optimizer.load_state_dict(ckpt["head_optimizer"]); head_scheduler.load_state_dict(ckpt["head_scheduler"])
        first_epoch = ckpt["epoch"] + 1; best_key = tuple(ckpt["best_key"])
        history = json.loads((output / "history.json").read_text())
    write_json(output / "run.json", run)
    write_json(output / "resolved_config.json", cfg)
    wall_start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(first_epoch, cfg["training"]["epochs"]):
        # Epoch-local RNG makes a resumed epoch boundary reproduce sampling.
        torch.manual_seed(cfg["seed"] + 10000019 * (epoch + 1))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(cfg["seed"] + 10000019 * (epoch + 1))
        student.train(); losses.train()
        if teacher is not None:
            teacher.eval()
        train_sum, n, step_losses = 0., 0, {}
        gradient_probe = None
        batch_diagnostics = []
        epoch_start = time.perf_counter()
        if hasattr(losses, "crd"):
            losses.crd.set_epoch(epoch)
        for images, labels, ids in _loader(fit, cfg, train=True, epoch=epoch):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            if head_optimizer is not None:
                head_optimizer.zero_grad(set_to_none=True)
            logits_s, feat_s = student(images)
            logits_t, feat_t = None, None
            if teacher is not None:
                with torch.no_grad():
                    logits_t, feat_t = teacher(images)
                losses.prepare_step(feat_s, feat_t)
            loss, parts = losses(logits_s, logits_t, feat_s, feat_t, labels, epoch)
            if losses.dependence is not None:
                batch_diagnostics.append(dict(losses.dependence.diagnostics))
            if hasattr(losses, "crd"):
                crd_loss = losses.crd(feat_s, feat_t, ids)
                weighted_crd = loss_cfg.get("crd_weight", .8) * crd_loss
                loss = loss + weighted_crd
                losses.last_representation_loss = weighted_crd if losses.last_representation_loss is None else losses.last_representation_loss + weighted_crd
                parts["crd"] = float(crd_loss.detach())
                parts.update(losses.crd.last_parts)  # r22: VCS-CRD J terms (empty for the official NCE)
            if n == 0 and losses.last_representation_loss is not None:
                g_task = torch.autograd.grad(losses.last_task, feat_s, retain_graph=True, allow_unused=True)[0]
                g_aux = torch.autograd.grad(losses.last_representation_loss, feat_s, retain_graph=True, allow_unused=True)[0]
                task_norm = float(g_task.norm()) if g_task is not None else 0.
                aux_norm = float(g_aux.norm()) if g_aux is not None else 0.
                gradient_probe = {"scope": "first_batch_each_epoch_pre_classifier_feature",
                                  "ce_gradient_norm": task_norm, "weighted_representation_gradient_norm": aux_norm,
                                  "aux_over_ce": aux_norm / task_norm if task_norm > 0 else None}
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}: {parts}")
            loss.backward()
            if teacher is not None and any(p.grad is not None for p in teacher.parameters()):
                raise AssertionError("Teacher acquired a gradient")
            for p in list(student.parameters()) + head_params:
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise FloatingPointError("Nonfinite training gradient")
            optimizer.step()
            if head_optimizer is not None:
                head_optimizer.step()
            losses.project_()
            train_sum += float(loss.detach()) * len(labels); n += len(labels)
            for k, v in parts.items():
                step_losses[k] = step_losses.get(k, 0.) + v * len(labels)
        if n == 0:
            raise ValueError("FIT smaller than batch size; no training steps executed")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - epoch_start
        scheduler.step()
        if head_scheduler is not None:
            head_scheduler.step()
        cal_metrics, rows = infer(student, cal, cfg, device)
        key = (cal_metrics["top1"], -cal_metrics["nll"])
        diag = getattr(losses.dependence, "diagnostics", {}) if losses.dependence is not None else {}
        dep_aggregate = {"n_batches": len(batch_diagnostics)}
        js = [d["raw_j"] for d in batch_diagnostics if "raw_j" in d]
        if js:
            dep_aggregate.update(raw_j_mean=statistics.mean(js), raw_j_min=min(js), raw_j_max=max(js))
        solved = [d for d in batch_diagnostics if "solver_converged" in d]
        if solved:
            ratios = [d["solver_gap"] / d["solver_zero_gap"] if d["solver_zero_gap"] > 0 else 0. for d in solved]
            dep_aggregate.update(solver_scope="inner_pre_outer_batch", solver_n=len(solved),
                 solver_converged_count=sum(bool(d["solver_converged"]) for d in solved),
                 solver_cap_hit_count=sum(not d["solver_converged"] and d["solver_iterations"] >= d["solver_cap"] for d in solved),
                 solver_zero_reference_count=sum(d["solver_zero_gap"] == 0. for d in solved),
                 solver_iterations_mean=statistics.mean(d["solver_iterations"] for d in solved),
                 solver_iterations_max=max(d["solver_iterations"] for d in solved),
                 solver_gap_ratio_mean=statistics.mean(ratios), solver_gap_ratio_max=max(ratios))
        # Shared diagnostics are JSON primitives; explicitly reject unexpected tensors.
        record = {"epoch": epoch, "fit_n_seen": n, "fit_loss": train_sum / n,
                  "loss_parts": {k: v / n for k, v in step_losses.items()}, "cal": cal_metrics,
                  "train_epoch_wall_seconds": train_seconds, "next_lr": optimizer.param_groups[0]["lr"], "dependence": diag,
                  "dependence_aggregate": dep_aggregate,
                  "gradient_probe": gradient_probe}
        history.append(record)
        improved = key > best_key
        if improved:
            best_key = key
        checkpoint = {"student": student.state_dict(), "losses": losses.state_dict(), "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(), "epoch": epoch, "config_sha256": config_hash,
                      "split_sha256": manifest["split_sha256"], "best_key": best_key,
                      "head_optimizer": head_optimizer.state_dict() if head_optimizer else None,
                      "head_scheduler": head_scheduler.state_dict() if head_scheduler else None}
        torch.save(checkpoint, output / "last.pt")
        if improved:
            torch.save(checkpoint, output / "best.pt")
            write_json(output / "best_cal_metrics.json", dict(cal_metrics, selected_epoch=epoch))
            write_predictions(output / "best_cal_predictions.csv", rows)
        write_json(output / "history.json", history)
        print(json.dumps({"run": cfg["name"], "epoch": epoch + 1, "cal_top1": cal_metrics["top1"], "fit_loss": train_sum / n}, allow_nan=False), flush=True)
    if teacher is not None and state_digest(teacher) != teacher_initial:
        raise AssertionError("Frozen teacher parameters or BN buffers changed")
    run.update({"status": "complete", "epochs_completed": len(history), "wall_seconds_this_invocation": time.perf_counter() - wall_start,
                "fit_wall_seconds_sum": sum(h["train_epoch_wall_seconds"] for h in history),
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                "teacher_frozen_verified": teacher is not None, "finished_unix_time": time.time()})
    if teacher_training:
        # The primary teacher is the fixed final epoch, never TEST-selected.
        ckpt = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        teacher_path = output / "teacher.pt"
        torch.save({"model": ckpt["student"]}, teacher_path)
        prov = {"schema": "kd_teacher_provenance_v1", "architecture": cfg["teacher"]["name"],
                "backend": cfg["teacher"]["backend"], "training_role": "FIT", "split_sha256": manifest["split_sha256"],
                "checkpoint_sha256": digest_file(teacher_path), "selection_rule": "fixed_final_epoch",
                "epoch": ckpt["epoch"], "test_used_for_selection": False, "fit_n": len(fit),
                "source_sha256": run["source_sha256"], "config_sha256": config_hash}
        write_json(output / "teacher_provenance.json", prov)
    run["best_checkpoint_sha256"] = digest_file(output / "best.pt")
    run["best_cal_predictions_sha256"] = digest_file(output / "best_cal_predictions.csv")
    run["best_cal_metrics_sha256"] = digest_file(output / "best_cal_metrics.json")
    write_json(output / "run.json", run)
    return run


def train(config_path, output, teacher_training=False, resume=False):
    try:
        return _train_impl(config_path, output, teacher_training, resume)
    except BaseException as exc:
        path = Path(output) / "run.json"
        if path.exists():
            run = json.loads(path.read_text())
            if run.get("status") == "running":
                run["status"] = "failed"
                run["failure"] = {"type": type(exc).__name__, "message": str(exc)}
                write_json(path, run)
        raise


def freeze(run_paths, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("A frozen protocol cannot be overwritten")
    entries, split, track, source, benchmark = [], None, None, None, None
    groups = set()
    for run_path in run_paths:
        folder = Path(run_path).resolve()
        r = json.loads((folder / "run.json").read_text())
        if r["status"] != "complete" or r["teacher_training"]:
            raise ValueError("Freeze only completed student runs")
        cfg = r["config"]
        this_benchmark = canonical_hash({"student": cfg["student"], "teacher": cfg["teacher"],
                                         "training": cfg["training"], "data": cfg["data"], "device": cfg["device"]})
        if benchmark is not None and benchmark != this_benchmark:
            raise ValueError("One freeze must use one architecture pair, teacher, dataset and training budget")
        benchmark = this_benchmark
        if canonical_hash(cfg) != r["config_sha256"]:
            raise ValueError("Run config was modified")
        for a, b, name in ((split, r["split_sha256"], "split"), (track, cfg["track"], "track"), (source, r["source_sha256"], "source")):
            if a is not None and a != b:
                raise ValueError(f"Do not combine inconsistent {name}s into one freeze")
        split, track, source = r["split_sha256"], cfg["track"], r["source_sha256"]
        identity = (cfg["name"], cfg["seed"])
        if identity in groups:
            raise ValueError("Duplicate method-name/seed. Select one CAL winner per method and seed before freezing.")
        groups.add(identity)
        if digest_file(folder / "best.pt") != r["best_checkpoint_sha256"]:
            raise ValueError("Checkpoint differs from completed run")
        if digest_file(folder / "best_cal_predictions.csv") != r["best_cal_predictions_sha256"]:
            raise ValueError("CAL evidence differs from completed run")
        if digest_file(folder / "best_cal_metrics.json") != r["best_cal_metrics_sha256"]:
            raise ValueError("CAL metrics differ from completed run")
        entries.append({"run_path": str(folder), "name": cfg["name"], "seed": cfg["seed"],
                        "config_sha256": r["config_sha256"], "checkpoint_sha256": r["best_checkpoint_sha256"],
                        "run_json_sha256": digest_file(folder / "run.json"),
                        "cal_predictions_sha256": r["best_cal_predictions_sha256"],
                        "cal_metrics_sha256": r["best_cal_metrics_sha256"]})
    if not entries:
        raise ValueError("No runs provided")
    body = {"schema": "kd_freeze_v1", "split_sha256": split, "source_sha256": source, "track": track,
            "benchmark_sha256": benchmark,
            "entries": entries, "test_selection_forbidden": True,
            "selection_rule": "user_supplied_CAL_winners_bound_before_TEST", "created_unix_time": time.time()}
    body["freeze_sha256"] = canonical_hash(body)
    write_json(output, body)
    return body


def evaluate(run_path, freeze_path):
    folder = Path(run_path).resolve()
    if (folder / "test_metrics.json").exists():
        raise FileExistsError("TEST already evaluated for this run; reuse the saved predictions without reselecting")
    frozen = json.loads(Path(freeze_path).read_text())
    seal = frozen.pop("freeze_sha256")
    if canonical_hash(frozen) != seal:
        raise ValueError("Freeze was modified")
    matches = [e for e in frozen["entries"] if e["run_path"] == str(folder)]
    if len(matches) != 1:
        raise PermissionError("Run is absent from frozen selection")
    e = matches[0]
    if digest_file(folder / "run.json") != e["run_json_sha256"] or digest_file(folder / "best.pt") != e["checkpoint_sha256"]:
        raise ValueError("Run/checkpoint changed after freeze")
    run = json.loads((folder / "run.json").read_text())
    cfg = run["config"]
    if canonical_hash(cfg) != e["config_sha256"] or source_digest() != frozen["source_sha256"]:
        raise ValueError("Config/source changed after freeze")
    manifest = read_manifest(cfg["data"]["manifest"])
    if manifest["split_sha256"] != frozen["split_sha256"]:
        raise ValueError("Split changed after freeze")
    if digest_file(folder / "best_cal_predictions.csv") != e["cal_predictions_sha256"]:
        raise ValueError("CAL evidence changed after freeze")
    if digest_file(folder / "best_cal_metrics.json") != e["cal_metrics_sha256"]:
        raise ValueError("CAL metrics changed after freeze")
    device = _device(cfg)
    model = create_model(cfg["student"], manifest["num_classes"], cfg["seed"])
    load_model_checkpoint(model, folder / "best.pt")
    model.to(device).eval()
    test = RoleDataset(cfg["data"]["root"], manifest, "test", allow_test=True)
    metrics, rows = infer(model, test, cfg, device)
    metrics.update({"role": "TEST", "freeze_sha256": seal, "name": cfg["name"], "seed": cfg["seed"], "track": cfg["track"]})
    write_predictions(folder / "test_predictions.csv", rows)
    metrics["predictions_sha256"] = digest_file(folder / "test_predictions.csv")
    write_json(folder / "test_metrics.json", metrics)
    return metrics


def compare(run_paths, output):
    """Paired seed differences; a descriptive summary, never an automatic claim."""
    runs = []
    for p in run_paths:
        p = Path(p)
        r = json.loads((p / "run.json").read_text())
        m = json.loads((p / "test_metrics.json").read_text())
        if digest_file(p / "test_predictions.csv") != m["predictions_sha256"]:
            raise ValueError("TEST predictions were modified")
        runs.append((r, m))
    if len({(r["split_sha256"], r["config"]["track"], m["freeze_sha256"]) for r, m in runs}) != 1:
        raise ValueError("Comparison requires one split, exposure track and freeze")
    by_name = {}
    for r, m in runs:
        by_name.setdefault(m["name"], {})[m["seed"]] = {"top1": m["top1"], "nll": m["nll"],
                                                   "fit_wall_seconds_sum": r["fit_wall_seconds_sum"],
                                                   "cuda_peak_allocated_bytes": r["cuda_peak_allocated_bytes"]}
    summaries = {}
    for name, values in by_name.items():
        top1 = [v["top1"] for v in values.values()]
        summaries[name] = {"n_seeds": len(top1), "mean_top1": statistics.mean(top1),
                           "sd_top1": statistics.stdev(top1) if len(top1) > 1 else None, "seeds": values}
    pairs = []
    names = sorted(by_name)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            seeds = sorted(set(by_name[a]) & set(by_name[b]))
            d = [by_name[a][s]["top1"] - by_name[b][s]["top1"] for s in seeds]
            pairs.append({"a": a, "b": b, "paired_seeds": seeds, "delta_top1_per_seed": d,
                          "mean_delta_top1": statistics.mean(d) if d else None,
                          "sd_delta_top1": statistics.stdev(d) if len(d) > 1 else None})
    result = {"methods": summaries, "paired_comparisons": pairs,
              "status": "DESCRIPTIVE_ONLY_until_required_baselines_and_protocol_checks_complete",
              "units": "accuracy_fraction; multiply differences by100 for percentage points"}
    write_json(output, result)
    return result
