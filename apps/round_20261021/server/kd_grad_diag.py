"""KD checkpoint gradient diagnostic (server-side addition, round 20261021; the package has no entry for it).

APPS_SERVER_CN §5 / 02_EXPERIMENT_PLAN §5: before the 9 probe runs, compare on the previous round's selected
checkpoints the CE + actual-epoch DKD anchor gradient with the lambda-weighted auxiliary gradient (norm ratio,
cosine), on a fixed, multi-batch, reproducible FIT set (03_RESULT_ANALYSIS §7). Nothing is trained or saved.

Reuses the installed stage's own code: kd.runner (config/teacher checks), kd.losses.DistillationLoss (its
`last_anchor` = CE + actual DKD, `last_representation_loss` = lambda * aux) and
vcs_common.application_update.representation_gradient_diagnostics. The selected checkpoints were trained jointly
(no application_update), so the stage's DistillationLoss reproduces their original loss exactly (weight = lambda).

Data: FIT only (RoleDataset role "fit", reads the CIFAR train file). TEST is never constructed; the run
directories' test_* files are not opened (only run.json and best.pt). CPU float32, deterministic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def summary(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"n": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--freeze", required=True, help="previous round kd_final_release.native_freeze.json")
    ap.add_argument("--families", default="dkd_vcs_exact,dkd_bce")
    ap.add_argument("--batches", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20261022, help="fixes the FIT subset and its augmentation")
    ap.add_argument("--probe-lambda", type=float, default=0.01, help="lambda of the new probe, for the rescaled ratio")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    stage = Path(args.stage).resolve()
    sys.path[:0] = [str(stage), str(stage / "shared")]
    import torch
    from kd.data import RoleDataset, canonical_hash, read_manifest
    from kd.losses import DistillationLoss
    from kd.models import create_model, feature_dim, load_model_checkpoint
    from kd.runner import _teacher, source_digest, state_digest
    from vcs_common.application_update import representation_gradient_diagnostics
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    t_start = time.perf_counter()

    frozen = json.loads(Path(args.freeze).read_text())
    seal = frozen.pop("freeze_sha256")
    if canonical_hash(frozen) != seal:
        raise ValueError("Previous-round freeze was modified")
    families = args.families.split(",")
    entries = sorted((e for e in frozen["entries"] if e["name"] in families), key=lambda e: (e["name"], e["seed"]))
    if len(entries) != 3 * len(families):
        raise ValueError(f"Expected 3 selected seeds per family, got {len(entries)}")

    fixed_batches, fit_ids, manifest0 = None, None, None
    teachers, results = {}, []
    for e in entries:
        folder = Path(e["run_path"])
        if sha256(folder / "run.json") != e["run_json_sha256"] or sha256(folder / "best.pt") != e["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint/run differs from the sealed freeze: {folder}")
        run = json.loads((folder / "run.json").read_text())
        cfg = run["config"]
        if canonical_hash(cfg) != e["config_sha256"]:
            raise ValueError("Config differs from the sealed freeze")
        manifest = read_manifest(cfg["data"]["manifest"])
        if manifest["split_sha256"] != frozen["split_sha256"]:
            raise ValueError("Split differs from the sealed freeze")
        if fixed_batches is None:
            manifest0 = manifest
            fit = RoleDataset(cfg["data"]["root"], manifest, "fit", augment=True, seed=args.seed)
            fit.set_epoch(0)
            g = torch.Generator().manual_seed(args.seed)
            order = torch.randperm(len(fit), generator=g)[: args.batches * args.batch_size].tolist()
            fixed_batches = []
            for b in range(args.batches):
                items = [fit[i] for i in order[b * args.batch_size:(b + 1) * args.batch_size]]
                fixed_batches.append((torch.stack([x for x, _, _ in items]), torch.stack([y for _, y, _ in items]),
                                      [int(i) for _, _, i in items]))
            fit_ids = [i for _, _, ids in fixed_batches for i in ids]
        elif manifest["split_sha256"] != manifest0["split_sha256"]:
            raise ValueError("All checkpoints must share one split")
        key = (cfg["teacher"]["checkpoint"], cfg["teacher"]["provenance"])
        if key not in teachers:
            teachers[key] = _teacher(cfg, manifest, device)
        teacher, teacher_dim, _ = teachers[key]
        teacher_sha0 = state_digest(teacher)
        student = create_model(cfg["student"], manifest["num_classes"], cfg["seed"])
        student_dim = feature_dim(student)
        load_model_checkpoint(student, folder / "best.pt")
        blob = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
        epoch = int(blob["epoch"])
        losses = DistillationLoss(student_dim, teacher_dim, cfg["loss"])
        losses.load_state_dict(blob["losses"])
        if losses.application_update.cfg is not None or losses.dependence is None:
            raise ValueError("Expected a jointly trained checkpoint with one dependence branch")
        lam = float(cfg["loss"]["dependence_weight"])
        student.to(device).train()   # train-mode BN as in the training step; no optimizer step, nothing saved
        losses.to(device).train()
        teacher.eval()
        params = [p for p in student.parameters() if p.requires_grad]
        rows = []
        for b, (images, labels, _) in enumerate(fixed_batches):
            torch.manual_seed(args.seed + b)
            logits_s, feat_s = student(images)
            with torch.no_grad():
                logits_t, feat_t = teacher(images)
            losses.prepare_step(feat_s, feat_t)
            total, parts = losses(logits_s, logits_t, feat_s, feat_t, labels, epoch)
            anchor, aux, ce = losses.last_anchor, losses.last_representation_loss, losses.last_task
            rep = representation_gradient_diagnostics(anchor, aux, feat_s)
            rep_ce = representation_gradient_diagnostics(ce, aux, feat_s)
            ga = torch.autograd.grad(anchor, params, retain_graph=True, allow_unused=True)
            gu = torch.autograd.grad(aux, params, retain_graph=False, allow_unused=True)
            fa = torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g, p in zip(ga, params)])
            fu = torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g, p in zip(gu, params)])
            na, nu = float(fa.norm()), float(fu.norm())
            ratio = rep["weighted_aux_over_anchor"]
            rows.append({
                "batch": b, "ce": parts["task"], "dkd_actual": parts.get("dkd"), "aux_unweighted": parts["dependence"],
                "raw_j": losses.dependence.diagnostics.get("raw_j"),
                "head_abs_max": losses.dependence.diagnostics.get("head_abs_max"),
                "feat_anchor_grad_norm": rep["anchor_gradient_norm"],
                "feat_weighted_aux_grad_norm": rep["weighted_aux_gradient_norm"],
                "feat_weighted_aux_over_anchor": ratio,
                "feat_aux_over_anchor_at_probe_lambda": None if ratio is None else ratio * args.probe_lambda / lam,
                "feat_anchor_aux_cosine": rep["anchor_aux_cosine"],
                "feat_ce_only_aux_cosine": rep_ce["anchor_aux_cosine"],
                "feat_weighted_aux_over_ce_only": rep_ce["weighted_aux_over_anchor"],
                "param_anchor_grad_norm": na, "param_weighted_aux_grad_norm": nu,
                "param_weighted_aux_over_anchor": nu / na if na > 0 else None,
                "param_anchor_aux_cosine": float(fa @ fu / (na * nu)) if na > 0 and nu > 0 else None,
            })
        if state_digest(teacher) != teacher_sha0:
            raise AssertionError("Teacher state changed during the diagnostic")
        keys = [k for k in rows[0] if k != "batch"]
        agg = {k: summary([r[k] for r in rows]) for k in keys}
        for k in ("feat_anchor_aux_cosine", "param_anchor_aux_cosine", "feat_ce_only_aux_cosine"):
            vals = [r[k] for r in rows if r[k] is not None]
            agg[k]["fraction_negative"] = sum(v < 0 for v in vals) / len(vals) if vals else None
        results.append({"family": e["name"], "seed": e["seed"], "lambda": lam, "method": cfg["loss"]["dependence"]["method"],
                        "checkpoint_epoch": epoch, "dkd_warmup_factor": min((epoch + 1) / max(cfg["loss"].get("dkd_warmup", 20), 1), 1.),
                        "checkpoint_sha256": e["checkpoint_sha256"], "config_sha256": e["config_sha256"],
                        "aggregate_over_batches": agg, "per_batch": rows})
        print(json.dumps({"family": e["name"], "seed": e["seed"], "lambda": lam,
                          "feat_ratio_mean": agg["feat_weighted_aux_over_anchor"]["mean"],
                          "feat_cos_mean": agg["feat_anchor_aux_cosine"]["mean"],
                          "param_ratio_mean": agg["param_weighted_aux_over_anchor"]["mean"],
                          "param_cos_mean": agg["param_anchor_aux_cosine"]["mean"]}), flush=True)

    by_family = {}
    for fam in families:
        rr = [r for r in results if r["family"] == fam]
        by_family[fam] = {k: {"per_seed_means": [r["aggregate_over_batches"][k]["mean"] for r in rr],
                              "mean_of_seed_means": statistics.mean(r["aggregate_over_batches"][k]["mean"] for r in rr)}
                          for k in ("feat_weighted_aux_over_anchor", "feat_aux_over_anchor_at_probe_lambda",
                                    "feat_anchor_aux_cosine", "feat_ce_only_aux_cosine", "param_weighted_aux_over_anchor",
                                    "param_anchor_aux_cosine", "raw_j", "aux_unweighted", "dkd_actual", "ce")}
        by_family[fam]["seeds"] = [r["seed"] for r in rr]
        by_family[fam]["lambda"] = rr[0]["lambda"]
    out = {"schema": "r21_kd_checkpoint_gradient_diagnostic_v1", "evidence_role": "PILOT/DEVELOPMENT mechanism diagnostic",
           "added_server_side": True, "package_entry": None,
           "checkpoints": "previous round (kd_20261020) sealed final release, CAL-selected best.pt per seed",
           "freeze_path": str(Path(args.freeze).resolve()), "freeze_sha256": seal,
           "anchor": "CE + actual-epoch DKD (losses.last_anchor); CE-only cosine reported separately for contrast with the old last_task log",
           "aux": "lambda * dependence loss with the checkpoint's own critic and lambda (losses.last_representation_loss)",
           "gradient_targets": {"feat": "pre-classifier student feature (same target as the runner's in-run gradient_probe)",
                                "param": "all student parameters, flattened"},
           "probe_lambda": args.probe_lambda,
           "fit_batches": {"n_batches": args.batches, "batch_size": args.batch_size, "seed": args.seed,
                           "augmentation": "training crop/flip, RoleDataset(seed=args.seed, epoch=0)",
                           "fit_ids_sha256": hashlib.sha256(json.dumps(fit_ids).encode()).hexdigest()},
           "mode": "student train-mode BN, teacher eval+no_grad, no optimizer step, CPU float32",
           "test_constructed": False, "split_sha256": manifest0["split_sha256"],
           "stage": str(stage), "stage_source_sha256": source_digest(),
           "environment": {"python": platform.python_version(), "torch": torch.__version__, "threads": args.threads,
                           "host": platform.node()},
           "wall_seconds": time.perf_counter() - t_start,
           "by_family": by_family, "per_checkpoint": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
