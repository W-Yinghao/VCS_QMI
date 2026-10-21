#!/usr/bin/env python
"""Pre-flight for ONE globally locked KD candidate, run BEFORE `vcs_common.experiment train`.

Why: the global `train` writes a RUNNING receipt first; any later crash (no CUDA device, bad node, unreadable
data, DataLoader worker failure) turns the job into FAILED/INTERRUPTED and, per EXPERIMENT_LOCK §6, can only be
repaired by an amendment that reruns every candidate. This script exercises the same native path on the
allocated node/GPU first: lock check, receipt absent, fresh run dir, role datasets (train hash/MD5), strict FIT
teacher (provenance/hash/pins), student, the configured loss (incl. official CRD) and one optimizer-free
forward/backward on one real FIT batch with the configured DataLoader workers.
It writes NOTHING into the lock state or the run directory. Exit 0 = PASS, 1 = FAIL.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lock", required=True)
    p.add_argument("--job-id", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min-free-gb", type=float, default=5.0)
    args = p.parse_args()
    t0 = time.time()
    result = {"schema": "kd_server_preflight_v1", "lock": str(Path(args.lock).resolve()), "job_id": args.job_id, "checks": {}}
    ok = True
    try:
        from vcs_common import experiment as ex
        lock = ex.check(args.lock)  # grid/source/manifest/assets/config hashes; rejects superseded or released locks
        result["lock_sha256"] = lock["lock_sha256"]
        job = ex._job(lock, args.job_id)
        receipt = ex._receipt_path(args.lock, job)
        run_dir = Path(job["run_dir"])
        result["run_dir"] = str(run_dir)
        checks = result["checks"]
        checks["receipt_absent"] = not receipt.exists()
        checks["run_dir_fresh"] = not (run_dir.exists() and any(run_dir.iterdir()))
        free_gb = shutil.disk_usage(lock["launch_cwd"]).free / 1e9
        checks["disk_free_gb"] = round(free_gb, 1)
        ok &= checks["receipt_absent"] and checks["run_dir_fresh"] and free_gb >= args.min_free_gb
        if ok:
            os.chdir(lock["launch_cwd"])
            import torch
            from kd.data import RoleDataset, read_manifest
            from kd.losses import DistillationLoss
            from kd.models import create_model, feature_dim
            from kd.runner import _device, _loader, _teacher, read_config
            cfg = read_config(job["config_path"])
            manifest = read_manifest(cfg["data"]["manifest"])
            checks["split_matches_lock"] = manifest["split_sha256"] == job["expected_split_sha256"]
            device = _device(cfg)
            fit = RoleDataset(cfg["data"]["root"], manifest, "fit", augment=True, seed=cfg["seed"])
            cal = RoleDataset(cfg["data"]["root"], manifest, "cal")
            checks["role_sizes"] = [len(fit), len(cal)]
            teacher, teacher_dim, provenance = _teacher(cfg, manifest, device)
            checks["teacher_provenance_role"] = provenance.get("training_role")
            student = create_model(cfg["student"], manifest["num_classes"], cfg["seed"])
            student_dim = feature_dim(student)
            student.to(device).train()
            losses = DistillationLoss(student_dim, teacher_dim, cfg["loss"]).to(device)
            crd = None
            if cfg["loss"].get("crd"):
                from kd.crd_adapter import OfficialCRD
                crd = OfficialCRD(student_dim, teacher_dim, fit.ids, fit.labels[fit.ids].tolist(), cfg["loss"]["crd"], cfg["seed"]).to(device)
            images, labels, ids = next(iter(_loader(fit, cfg, train=True, epoch=0)))
            images, labels = images.to(device), labels.to(device)
            logits_s, feat_s = student(images)
            with torch.no_grad():
                logits_t, feat_t = teacher(images)
            losses.train()
            losses.prepare_step(feat_s, feat_t)
            loss, parts = losses(logits_s, logits_t, feat_s, feat_t, labels, 0)
            if crd is not None:
                loss = loss + cfg["loss"].get("crd_weight", .8) * crd(feat_s, feat_t, ids)
            loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            grads_finite = all(torch.isfinite(q.grad).all() for q in student.parameters() if q.grad is not None)
            checks["one_batch_loss_finite"] = bool(torch.isfinite(loss))
            checks["one_batch_grads_finite"] = bool(grads_finite)
            checks["teacher_grad_free"] = all(q.grad is None for q in teacher.parameters())
            checks["device"] = str(device)
            if device.type == "cuda":
                checks["gpu_name"] = torch.cuda.get_device_name(device)
            ok &= all(checks[k] is True for k in ("split_matches_lock", "one_batch_loss_finite", "one_batch_grads_finite", "teacher_grad_free"))
            ok &= checks["role_sizes"] == [45000, 5000] and checks["teacher_provenance_role"] == "FIT"
    except Exception as exc:
        ok = False
        result["exception"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    result["status"] = "PASS" if ok else "FAIL"
    result["seconds"] = round(time.time() - t0, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "checks", "seconds")}, default=str), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
