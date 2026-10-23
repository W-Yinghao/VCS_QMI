from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import IQADataset
from .losses import IQALosses
from vcs_common.application_update import representation_gradient_diagnostics
from .manifest import (INPUT_FIELDS, read_csv, write_csv, write_json, sha256_file,
                       verify_prepared, cross_dataset_audit, safe_image_path)
from .metrics import metrics, paired_bootstrap
from .models import build_model

PREDICTION_FIELDS = ["sample_id", "dataset", "prompt_group", "prediction"]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def environment():
    versions = {}
    for name in ["torch", "torchvision", "image-reward", "transformers", "timm", "numpy", "Pillow"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "NOT_INSTALLED"
    return {"versions": versions, "torch_runtime": torch.__version__, "cuda": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "tf32_matmul_default_observed": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn_default_observed": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "tf32_policy": "keep torch defaults for every method; settings are logged, never changed here",
            "amp": False, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def build_losses(model, cfg, device):
    # Preserve the host/dropout RNG stream across all five new recipe arms.
    # Previously trained configurations without application_update are unchanged.
    if cfg["loss"].get("application_update") is not None:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(cfg["seed"] + 12345)
            return IQALosses(model, cfg["loss"]).to(device)
    return IQALosses(model, cfg["loss"]).to(device)


def validate_config(cfg):
    required = {"model", "data_dir", "output_dir", "target", "seed", "loss"}
    if not required <= cfg.keys():
        raise ValueError(f"Missing config keys: {sorted(required - cfg.keys())}")
    if cfg["target"] not in {"quality", "alignment"}:
        raise ValueError("Quality and alignment need separate trained models")
    if cfg.get("batch_size", 5) < 2:
        raise ValueError("Training batch size must be >=2")
    if cfg.get("epochs", 20) < 1:
        raise ValueError("epochs must be >=1")
    if cfg.get("selection_metric", "srcc") not in {"srcc", "within_prompt_pairwise_macro"}:
        raise ValueError("Choose a signed CAL selection metric before training")
    if cfg["model"].get("name") == "tiny_synthetic" and cfg.get("evidence_role") != "SYNTHETIC_INTEGRATION_ONLY":
        raise ValueError("Tiny model requires the explicit synthetic-only evidence role")
    if cfg["model"].get("name") != "tiny_synthetic" and cfg.get("evidence_role") == "SYNTHETIC_INTEGRATION_ONLY":
        raise ValueError("Do not mark a real model synthetic")
    for key in ["weight", "forward_weight", "entropy_surrogate_weight"]:
        value = cfg["loss"].get(key, 0.)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"Invalid loss weight {key}")


def make_loader(dataset, batch_size, workers, train, seed):
    return DataLoader(dataset, batch_size=batch_size, shuffle=train,
                      drop_last=train, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    predictions = []
    for batch in loader:
        output = model(batch["image"].to(device), batch["prompt"])
        values = output["prediction"].detach().cpu().numpy().reshape(-1)
        for i, value in enumerate(values):
            row = {"sample_id": batch["sample_id"][i], "dataset": batch["dataset"][i],
                   "prompt_group": batch["prompt_group"][i], "prediction": float(value)}
            if "label" in batch:
                row["mos"] = float(batch["raw_mos"][i])
            predictions.append(row)
    return predictions


def preflight(cfg):
    validate_config(cfg)
    info = verify_prepared(cfg["data_dir"])
    device = cfg.get("device", "cuda")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; do not silently run a real training job on CPU")
    seed_all(cfg["seed"])
    model = build_model(cfg["model"], device)
    losses = build_losses(model, cfg, device)
    dataset = IQADataset(Path(cfg["data_dir"]) / "fit.csv", info["image_root"], cfg["target"], cfg.get("image_size", 224))
    loader = make_loader(dataset, min(cfg.get("batch_size", 5), len(dataset)), 0, True, cfg["seed"])
    batch = next(iter(loader))
    output = model(batch["image"].to(device), batch["prompt"])
    losses.prepare_step(output, batch["label"].to(device))
    total, _, _ = losses(output, batch["label"].to(device))
    total.backward()
    bad = [name for name, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    if bad or not torch.isfinite(total):
        raise FloatingPointError(f"Nonfinite real preflight: {bad[:5]}")
    return {"status": "PASS", "evidence_role": cfg.get("evidence_role", "DEV"),
            "model": cfg["model"]["name"], "shapes": {k: list(v.shape) for k, v in output.items()},
            "loss": float(total.detach()), "diagnostics": losses.diagnostics,
            "model_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "regularizer_trainable_parameters": sum(p.numel() for p in losses.parameters() if p.requires_grad),
            "environment": environment()}


def train(cfg):
    validate_config(cfg)
    out = Path(cfg["output_dir"])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refuse to overwrite run: {out}")
    info = verify_prepared(cfg["data_dir"])
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", cfg)
    write_json(out / "environment.json", environment())
    try:
        return _train(cfg, info, out)
    except Exception as exc:
        write_json(out / "FAILED.json", {"exception": type(exc).__name__, "message": str(exc)})
        raise


def _train(cfg, info, out):
    from vcs_common.dependence import parameter_groups
    seed_all(cfg["seed"])
    device = cfg.get("device", "cuda")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = build_model(cfg["model"], device)
    losses = build_losses(model, cfg, device)
    combined = nn.ModuleDict({"model": model, "losses": losses})
    model_groups = parameter_groups(model, weight_decay=cfg.get("weight_decay", .001))
    auxiliary_groups = [] if losses.application_update.alternating else parameter_groups(losses, weight_decay=0. if losses.application_update.cfg else cfg.get("weight_decay", .001))
    for group in auxiliary_groups:
        group["lr"] = cfg.get("regularizer_learning_rate", .001)
    opt = torch.optim.AdamW(model_groups + auxiliary_groups,
                            lr=cfg.get("learning_rate", 1e-5), betas=(.9, .999), eps=1e-8)
    losses.application_update.assert_outer_optimizer(opt)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.get("lr_step_epochs", 4), gamma=.5)
    data_dir = Path(cfg["data_dir"])
    train_data = IQADataset(data_dir / "fit.csv", info["image_root"], cfg["target"], cfg.get("image_size", 224))
    cal_data = IQADataset(data_dir / "cal.csv", info["image_root"], cfg["target"], cfg.get("image_size", 224))
    train_loader = make_loader(train_data, cfg.get("batch_size", 5), cfg.get("num_workers", 0), True, cfg["seed"])
    cal_loader = make_loader(cal_data, cfg.get("batch_size", 5), cfg.get("num_workers", 0), False, cfg["seed"])
    if not len(train_loader):
        raise ValueError("No complete training batch")
    warmup = cfg.get("gaussian_warmup_epochs", 0)
    synchronize(device)
    warmup_start = time.perf_counter()
    if warmup:
        if losses.forward_regularizer is None:
            raise ValueError("Gaussian warm-up only exists for corrected_gaussian baseline")
        warm_opt = torch.optim.AdamW(losses.forward_regularizer.parameters(), lr=cfg.get("warmup_learning_rate", 5e-6), weight_decay=cfg.get("weight_decay", .001))
        model.eval()
        for _ in range(warmup):
            for batch in train_loader:
                with torch.no_grad():
                    output = model(batch["image"].to(device), batch["prompt"])
                warm_opt.zero_grad(set_to_none=True)
                value = losses.forward_regularizer(output["visual"], output["text"])
                value.backward()
                warm_opt.step()
    synchronize(device)
    warmup_seconds = time.perf_counter() - warmup_start if warmup else 0.0
    selection = cfg.get("selection_metric", "srcc")
    best, best_epoch, best_metrics = -float("inf"), None, None
    history, total_fit_time = [], 0.0
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(cfg.get("epochs", 20)):
        model.train()
        losses.train()
        losses.application_update.set_epoch(epoch)
        sums, counts = defaultdict(float), defaultdict(int)
        synchronize(device)
        started = time.perf_counter()
        for step, batch in enumerate(train_loader):
            opt.zero_grad(set_to_none=True)
            output = model(batch["image"].to(device), batch["prompt"])
            labels = batch["label"].to(device)
            losses.prepare_step(output, labels)
            total, task, dependence = losses(output, labels)
            if step == min(1, len(train_loader) - 1):
                probe = representation_gradient_diagnostics(task,
                    losses.application_update.weighted(dependence), output["fusion"])
                for key, value in probe.items():
                    if value is not None:
                        sums["fusion_" + key] += value
                        counts["fusion_" + key] += 1
            total.backward()
            if not torch.isfinite(total) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in combined.parameters()):
                raise FloatingPointError(f"Nonfinite training at epoch={epoch}, step={step}")
            opt.step()
            losses.project_()
            for name, value in losses.diagnostics.items():
                if isinstance(value, (float, int)) and np.isfinite(value):
                    sums[name] += float(value)
                    counts[name] += 1
            for name in ["visual", "text", "fusion"]:
                key = name + "_batch_std_mean"
                sums[key] += float(output[name].detach().std(0, unbiased=False).mean())
                counts[key] += 1
        synchronize(device)
        fit_seconds = time.perf_counter() - started
        total_fit_time += fit_seconds
        started = time.perf_counter()
        predictions = predict_loader(model, cal_loader, device)
        cal_metrics = metrics(predictions, cfg.get("tie_epsilon_mos", .05))
        synchronize(device)
        row = {"epoch": epoch + 1, "fit_seconds": fit_seconds, "cal_seconds": time.perf_counter() - started,
               "learning_rate": opt.param_groups[0]["lr"], "train_diagnostics": {k: sums[k] / counts[k] for k in sums},
               "cal_metrics": cal_metrics}
        history.append(row)
        write_json(out / "history.json", history)
        score = cal_metrics[selection]
        if score is not None and score > best:
            best, best_epoch, best_metrics = score, epoch + 1, cal_metrics
            torch.save({"model": model.state_dict(), "losses": losses.state_dict(), "config": cfg,
                        "epoch": best_epoch, "cal_metrics": cal_metrics,
                        "application_update": losses.application_update.state_dict(),
                        "prepared_manifest_sha256": sha256_file(data_dir / "manifest.json")}, out / "best.pt")
            write_csv(out / "cal_predictions.csv", predictions, PREDICTION_FIELDS + ["mos"])
        scheduler.step()
        losses.application_update.epoch_end()
    if best_epoch is None:
        raise RuntimeError("CAL selection metric undefined at every epoch; preserve failure and repair protocol before TEST")
    result = {"status": "COMPLETED_FIT_CAL_ONLY", "evidence_role": cfg.get("evidence_role", "DEV"),
              "seed": cfg["seed"], "target": cfg["target"], "best_epoch": best_epoch, "selection_metric": selection,
              "cal_metrics": best_metrics, "fit_seconds_excludes_warmup": total_fit_time,
              "gaussian_warmup_seconds": warmup_seconds,
              "training_seconds_including_warmup": total_fit_time + warmup_seconds,
              "fit_images_per_epoch": len(train_loader) * cfg.get("batch_size", 5),
              "random_tail_dropped_per_epoch": len(train_data) % cfg.get("batch_size", 5),
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda") else None,
              "checkpoint_sha256": sha256_file(out / "best.pt"),
              "config_sha256": sha256_file(out / "config.json"),
              "prepared_manifest_sha256": sha256_file(data_dir / "manifest.json"),
              "test_labels_read": False,
              "application_update_policy": losses.application_update.cfg,
              "critic_updates_total": losses.application_update.updates,
              "parameters_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "regularizer_parameters_trainable": sum(p.numel() for p in losses.parameters() if p.requires_grad)}
    write_json(out / "result.json", result)
    return result


def _config_family(cfg):
    loss = cfg["loss"]
    return cfg.get("method_label", loss["method"] + "_" + loss.get("forward", "none") + "_" + "-".join(loss.get("branches", ["text"])))


def freeze(run_dirs, inputs, output):
    """Select hyperparameter candidates using equal seed sets and CAL only.

    All runs from each family must be supplied, including rejected candidates.
    TEST labels are neither required nor opened. The output pins the exact list
    of selected checkpoints and label-free evaluation inputs.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError("Do not overwrite a frozen protocol")
    grouped = defaultdict(lambda: defaultdict(list))
    all_runs = []
    source_dir = target = selection_metric = None
    for path in map(Path, run_dirs):
        cfg = json.loads((path / "config.json").read_text())
        result = json.loads((path / "result.json").read_text())
        if result["status"] != "COMPLETED_FIT_CAL_ONLY" or (path / "FAILED.json").exists():
            raise ValueError(f"Incomplete/failed candidate: {path}")
        if sha256_file(path / "best.pt") != result["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed after CAL selection")
        if sha256_file(path / "config.json") != result["config_sha256"]:
            raise ValueError("Config changed after training")
        if sha256_file(Path(cfg["data_dir"]) / "manifest.json") != result["prepared_manifest_sha256"]:
            raise ValueError("Prepared data manifest changed after training")
        if source_dir is None:
            source_dir, target, selection_metric = cfg["data_dir"], cfg["target"], result["selection_metric"]
        if Path(source_dir).resolve() != Path(cfg["data_dir"]).resolve() or cfg["target"] != target:
            raise ValueError("Freeze one source dataset / target dimension at a time")
        if result["selection_metric"] != selection_metric:
            raise ValueError("All candidate/method families need the same preregistered CAL selection metric")
        candidate = copy.deepcopy(cfg)
        for key in ["seed", "output_dir", "device", "num_workers"]:
            candidate.pop(key, None)
        signature = hashlib.sha256(json.dumps(candidate, sort_keys=True).encode()).hexdigest()[:16]
        rec = {"run_dir": str(path.resolve()), "config": cfg, "result": result, "candidate": signature}
        grouped[_config_family(cfg)][signature].append(rec)
        all_runs.append(str(path.resolve()))
    if not all_runs:
        raise ValueError("No completed runs supplied")
    selections, candidate_table = [], []
    family_seed_sets = []
    for family, candidates in grouped.items():
        seed_sets = [{r["config"]["seed"] for r in rr} for rr in candidates.values()]
        if any(s != seed_sets[0] for s in seed_sets):
            raise ValueError(f"Unequal CAL tuning seed sets in family {family}")
        family_seed_sets.append(seed_sets[0])
        ranking = []
        for sig, rr in candidates.items():
            if len(rr) != len({r["config"]["seed"] for r in rr}):
                raise ValueError("Duplicate candidate seed")
            selection = rr[0]["result"]["selection_metric"]
            vals = [r["result"]["cal_metrics"][selection] for r in rr]
            score = float(np.mean(vals))
            ranking.append((score, sig, rr))
            candidate_table.append({"family": family, "candidate": sig, "mean_cal_selection": score, "seeds": sorted(seed_sets[0])})
        # deterministic hash tie-break; all candidates used identical selection.
        chosen = sorted(ranking, key=lambda x: (-x[0], x[1]))[0]
        for rec in chosen[2]:
            path = Path(rec["run_dir"])
            key = f"{family}__seed{rec['config']['seed']}"
            selections.append({"run_key": key, "family": family, "candidate": chosen[1],
                               "checkpoint": str(path / "best.pt"), "checkpoint_sha256": rec["result"]["checkpoint_sha256"],
                               "config_sha256": sha256_file(path / "config.json"), "config": rec["config"],
                               "selection_mean_cal": chosen[0]})
    if any(s != family_seed_sets[0] for s in family_seed_sets):
        raise ValueError("Frozen causal comparison requires identical training seed sets across method families")
    inputs_records = []
    for input_path in map(Path, inputs):
        rows = read_csv(input_path)
        if not rows or any("quality_mos" in r or "alignment_mos" in r for r in rows):
            raise ValueError("Freeze requires label-free inputs")
        audit = cross_dataset_audit(source_dir, input_path)
        # Within-source TEST should be disjoint, cross-dataset exact/near overlap
        # should be removed using metadata before this freeze, not target scores.
        if audit["overlap_count"]:
            raise ValueError(f"Source/FIT-CAL overlap in target inputs: {audit['overlap_count']}; audit/exclude by metadata before freezing")
        inputs_records.append({"path": str(input_path.resolve()), "sha256": sha256_file(input_path),
                               "dataset": rows[0]["dataset"], "n_images": len(rows), "audit": audit})
    value = {"schema": "vcs_aigiqa_frozen_protocol_v1", "target": target,
             "source_manifest_sha256": sha256_file(Path(source_dir) / "manifest.json"),
             "candidate_runs": all_runs, "candidate_table": candidate_table,
             "selected_runs": selections, "evaluation_inputs": inputs_records,
             "test_labels_read_during_selection": False,
             "limitation": "This records a workflow boundary, not a cryptographic proof that a human never saw labels."}
    write_json(output, value)
    return value


def predict(freeze_path, run_key, inputs, image_root, output, device=None):
    frozen = json.loads(Path(freeze_path).read_text())
    records = [r for r in frozen["selected_runs"] if r["run_key"] == run_key]
    if len(records) != 1:
        raise ValueError(f"run_key must identify one frozen selected run: {run_key}")
    record = records[0]
    input_hash = sha256_file(inputs)
    if input_hash not in {r["sha256"] for r in frozen["evaluation_inputs"]}:
        raise ValueError("Inputs were not listed in the frozen protocol")
    if sha256_file(record["checkpoint"]) != record["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint changed")
    cfg = record["config"]
    actual_device = device or cfg.get("device", "cuda")
    model = build_model(cfg["model"], actual_device)
    checkpoint = torch.load(record["checkpoint"], map_location=actual_device, weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    data = IQADataset(inputs, image_root, None, cfg.get("image_size", 224))
    loader = make_loader(data, cfg.get("batch_size", 5), cfg.get("num_workers", 0), False, cfg["seed"])
    output = Path(output)
    if output.exists():
        raise FileExistsError("Do not overwrite frozen predictions")
    synchronize(actual_device)
    start = time.perf_counter()
    rows = predict_loader(model, loader, actual_device)
    synchronize(actual_device)
    write_csv(output, rows, PREDICTION_FIELDS)
    meta = {"run_key": run_key, "target": cfg["target"], "n_images": len(rows),
            "prediction_sha256": sha256_file(output), "input_sha256": input_hash,
            "freeze_sha256": sha256_file(freeze_path), "checkpoint_sha256": record["checkpoint_sha256"],
            "seconds_including_image_io": time.perf_counter() - start, "test_labels_read": False}
    write_json(str(output) + ".meta.json", meta)
    return meta


def merge_labels(predictions, labels, target):
    pred = read_csv(predictions)
    label = read_csv(labels)
    ids = {r["sample_id"] for r in pred}
    if len(ids) != len(pred):
        raise ValueError("Duplicate prediction id")
    lm = {r["sample_id"]: r for r in label}
    if len(lm) != len(label) or not ids <= lm.keys():
        raise ValueError("Missing or duplicate evaluation labels")
    return [{**r, "mos": float(lm[r["sample_id"]][target + "_mos"])} for r in pred]


def evaluate(predictions, labels, output, target, tie_epsilon_mos=.05):
    meta_path = Path(str(predictions) + ".meta.json")
    if not meta_path.is_file():
        raise ValueError("Prediction metadata required: use predict or import_scores")
    meta = json.loads(meta_path.read_text())
    if meta["prediction_sha256"] != sha256_file(predictions) or meta["target"] != target:
        raise ValueError("Prediction checksum/target mismatch")
    rows = merge_labels(predictions, labels, target)
    result = metrics(rows, tie_epsilon_mos)
    if "external_provenance" in meta:
        result["rmse_normalized"] = None
        result["rmse_omission_reason"] = "External raw score is not on MOS/5 scale; no TEST calibration allowed"
    result.update(prediction_sha256=sha256_file(predictions), labels_sha256=sha256_file(labels),
                  target=target, prediction_provenance=meta)
    write_json(output, result)
    return result


def compare(a, b, labels, output, target, repetitions=1000):
    provenance = []
    for path in [a, b]:
        meta_path = Path(str(path) + ".meta.json")
        if not meta_path.is_file():
            raise ValueError("Paired comparison needs prediction metadata")
        meta = json.loads(meta_path.read_text())
        if meta["prediction_sha256"] != sha256_file(path) or meta["target"] != target:
            raise ValueError("Paired prediction checksum/target mismatch")
        provenance.append(meta)
    for field in ["input_sha256", "freeze_sha256"]:
        if provenance[0][field] != provenance[1][field]:
            raise ValueError("Compare methods on the same frozen inputs and protocol")
    rows_a, rows_b = merge_labels(a, labels, target), merge_labels(b, labels, target)
    result = paired_bootstrap(rows_a, rows_b, repetitions=repetitions)
    result.update(a_sha256=sha256_file(a), b_sha256=sha256_file(b), target=target)
    write_json(output, result)
    return result


def import_scores(scores, inputs, freeze_path, output, target, provenance_json):
    """Register external baselines using raw scores and explicit version/budget."""
    frozen = json.loads(Path(freeze_path).read_text())
    if target != frozen["target"] or sha256_file(inputs) not in {r["sha256"] for r in frozen["evaluation_inputs"]}:
        raise ValueError("Inputs / target not frozen")
    provenance = json.loads(Path(provenance_json).read_text())
    required = {"method", "source_url", "source_revision", "checkpoint_sha256", "extra_inputs", "score_direction"}
    if not required <= provenance.keys() or provenance["score_direction"] not in {"higher_is_better", "lower_is_better"}:
        raise ValueError("External provenance must include version, checkpoint, extra-input budget, direction")
    source = read_csv(scores)
    sm = {r["sample_id"]: float(r["prediction"]) for r in source}
    rr = read_csv(inputs)
    if len(sm) != len(source) or set(sm) != {r["sample_id"] for r in rr}:
        raise ValueError("External scores must cover every frozen input exactly once")
    if Path(output).exists():
        raise FileExistsError("Do not overwrite predictions")
    sign = 1 if provenance["score_direction"] == "higher_is_better" else -1
    rows = [{**r, "prediction": sign * sm[r["sample_id"]]} for r in rr]
    if any(not np.isfinite(r["prediction"]) for r in rows):
        raise ValueError("Nonfinite external score")
    write_csv(output, rows, PREDICTION_FIELDS)
    meta = {"target": target, "prediction_sha256": sha256_file(output), "input_sha256": sha256_file(inputs),
            "freeze_sha256": sha256_file(freeze_path), "external_provenance": provenance,
            "test_labels_read_by_adapter": False,
            "score_scale": "external raw oriented score; normalized RMSE is not comparable without SOURCE-CAL calibration"}
    write_json(str(output) + ".meta.json", meta)
    return meta


@torch.no_grad()
def native_imagereward(cfg, inputs, image_root, output_scores):
    """Official native scoring/preprocess. Import into a freeze for evaluation."""
    from PIL import Image
    device = cfg.get("device", "cuda")
    model = build_model(cfg["model"], device)
    if not hasattr(model, "reward"):
        raise ValueError("Native ImageReward requires the real checkpoint")
    reward = model.reward.eval()
    rows = read_csv(inputs)
    if any("quality_mos" in r or "alignment_mos" in r for r in rows):
        raise ValueError("Native scoring uses label-free inputs")
    for row in rows:
        path = safe_image_path(image_root, row["image"])
        if sha256_file(path) != row["image_sha256"]:
            raise ValueError("Image bytes changed")
        with Image.open(path) as image:
            row["prediction"] = float(reward.score(row["prompt"], image.convert("RGB")))
    if Path(output_scores).exists():
        raise FileExistsError("Do not overwrite raw scores")
    write_csv(output_scores, rows, ["sample_id", "prediction"])
    return {"n_images": len(rows), "method": "ImageReward_native_frozen", "checkpoint_sha256": cfg["model"]["checkpoint_sha256"]}
