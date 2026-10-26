"""Run from the bundle root with the shared folder on PYTHONPATH."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from .data import prepare, write_json
from .runner import compare, evaluate, freeze, read_config, train


ARMS = ("student", "kd", "dkd", "dkd_mse", "dkd_cosine", "dkd_vid_pooled", "dkd_bce",
        "dkd_vcs_exact", "dkd_vcs_sampled", "dkd_vcs_m1", "dkd_infonce", "crd_original", "dkd_crd_uniform",
        "itrd_reference", "dkd_itrd")


def arm_config(base, arm, coefficient=1.0):
    cfg = copy.deepcopy(base)
    cfg["name"] = arm
    cfg["loss"] = {"anchor": "none", "feature": "none", "ce_weight": 1.0}
    # Task architecture keeps the source SGD recipe. Only dependence heads use
    # the shared AdamW recipe; original feature/CRD controls retain SGD heads.
    cfg["optimizer"]["head_optimizer"] = "sgd"
    cfg["optimizer"]["head_lr"] = cfg["optimizer"]["lr"]
    if arm == "student":
        return cfg
    if arm == "kd":
        cfg["loss"].update(anchor="kd", temperature=4., ce_weight=.1, anchor_weight=.9)
        return cfg
    if arm == "itrd_reference":
        cfg["loss"].update(feature="itrd", feature_weight=1., itrd={"alpha": 1.01, "correlation_weight": 2., "mutual_weight": 1., "std_floor": 0.})
        return cfg
    if arm.startswith("crd_") or arm == "dkd_crd_uniform":
        if arm.startswith("dkd_"):
            cfg["loss"].update(anchor="dkd", temperature=4., dkd_alpha=1., dkd_beta=base.get("loss", {}).get("dkd_beta", 8.), dkd_warmup=20, anchor_weight=1.)
        cfg["loss"].update(crd_weight=.8, crd={"source_root": "external/RepDistiller", "feat_dim": 128,
                         "negatives": 16384, "temperature": .07, "momentum": .5,
                         "negative_mode": "class_excluding_original" if arm == "crd_original" else "uniform_other_instances"})
        return cfg
    cfg["loss"].update(anchor="dkd", temperature=4., dkd_alpha=1., dkd_beta=base.get("loss", {}).get("dkd_beta", 8.), dkd_warmup=20, anchor_weight=1.)
    if arm == "dkd":
        return cfg
    suffix = arm.removeprefix("dkd_")
    if suffix in ("mse", "cosine", "vid_pooled", "itrd"):
        cfg["loss"].update(feature=suffix, feature_weight=coefficient)
    elif suffix in ("bce", "vcs_exact", "vcs_sampled", "vcs_m1", "infonce"):
        cfg["optimizer"].update(head_optimizer="adamw", head_lr=.001)
        cfg["loss"].update(dependence_weight=coefficient, dependence={"method": suffix,
                          "rank": 32, "q_mode": "u", "sampled_draws": 8, "temperature": 1.0,
                          "solver_max_iter": 30, "solver_relative_tol": .01, "nce_temperature": .1})
    else:
        raise ValueError(f"Unknown arm {arm}")
    return cfg


def make_grid(base_path, output, arms, seeds, coefficients):
    base = json.loads(Path(base_path).read_text())
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Grid output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for arm in arms:
        coefficients_arm = coefficients if arm.startswith("dkd_") and "crd" not in arm else [1.]
        for coeff in coefficients_arm:
            for seed in seeds:
                cfg = arm_config(base, arm, coeff); cfg["seed"] = seed
                name = f"{arm}_w{coeff:g}_seed{seed}.json"
                write_json(output / name, cfg)
                rows.append({"config": name, "method": arm, "coefficient": coeff, "seed": seed,
                             "role": "CAL_candidate", "status": "NOT_RUN"})
    write_json(output / "grid_manifest.json", {"rows": rows, "no_jobs_submitted": True})
    return {"config_count": len(rows), "output": str(output)}


def select_cal(run_paths, output):
    """Choose one candidate per method by paired-seed mean CAL top1 then NLL.

    Requires the same seed set across candidates within a method. This command
    never reads TEST metrics and produces configs to reuse for final seed runs.
    """
    from .data import canonical_hash, digest_file
    groups, split, source = {}, None, None
    for path in run_paths:
        p = Path(path).resolve()
        run = json.loads((p / "run.json").read_text())
        if run["status"] != "complete" or run["teacher_training"]:
            raise ValueError("CAL selection requires completed student runs")
        if split is not None and split != run["split_sha256"]:
            raise ValueError("CAL runs use different manifests")
        if source is not None and source != run["source_sha256"]:
            raise ValueError("CAL runs use different source versions")
        split = run["split_sha256"]
        source = run["source_sha256"]
        if canonical_hash(run["config"]) != run["config_sha256"]:
            raise ValueError("CAL run config was modified")
        if digest_file(p / "best_cal_metrics.json") != run["best_cal_metrics_sha256"] or digest_file(p / "best_cal_predictions.csv") != run["best_cal_predictions_sha256"]:
            raise ValueError("CAL evidence no longer matches completed run")
        cfg = copy.deepcopy(run["config"])
        seed = cfg.pop("seed")
        signature = canonical_hash(cfg)
        m = json.loads((p / "best_cal_metrics.json").read_text())
        group = groups.setdefault(cfg["name"], {}).setdefault(signature, {"config": cfg, "runs": {}})
        if seed in group["runs"]:
            raise ValueError("Duplicate candidate/seed")
        group["runs"][seed] = {"run_path": str(p), "top1": m["top1"], "nll": m["nll"]}
    results = {}
    for name, candidates in groups.items():
        seed_sets = {tuple(sorted(v["runs"])) for v in candidates.values()}
        if len(seed_sets) != 1:
            raise ValueError(f"Unequal CAL seed budget for {name}")
        for signature, v in candidates.items():
            n = len(v["runs"])
            v["mean_cal_top1"] = sum(x["top1"] for x in v["runs"].values()) / n
            v["mean_cal_nll"] = sum(x["nll"] for x in v["runs"].values()) / n
        winner = min(candidates, key=lambda s: (-candidates[s]["mean_cal_top1"], candidates[s]["mean_cal_nll"], s))
        results[name] = {"selected_config_signature": winner, "winner": candidates[winner], "all_candidates": candidates}
    result = {"schema": "kd_cal_selection_v1", "split_sha256": split, "test_read": False,
              "rule": "mean_CAL_top1_then_mean_CAL_nll_then_lexical_config_hash", "methods": results}
    write_json(output, result)
    return {"output": str(output), "methods": sorted(results)}


def make_final_grid(selection_path, output, seeds):
    from .data import digest_file, canonical_hash, read_manifest
    selection = json.loads(Path(selection_path).read_text())
    if selection.get("schema") == "vcs_global_cal_selection_v1":
        checked=dict(selection); seal=checked.pop("selection_sha256",None)
        if canonical_hash(checked)!=seal or selection.get("app")!="kd" or selection.get("test_released") is not False:
            raise ValueError("Invalid global CAL selection seal/app/role")
        lock=json.loads(Path(selection["lock_path"]).read_text())
        checked=dict(lock);lock_seal=checked.pop("lock_sha256",None)
        if canonical_hash(checked)!=lock_seal or lock_seal!=selection["lock_sha256"]:
            raise ValueError("Global CAL source lock hash mismatch")
        jobs={j["job_id"]:j for j in lock["jobs"]};methods={};split=None
        for job_id in selection["selected_job_ids"]:
            job=jobs[job_id];cfg=copy.deepcopy(job["effective_config"])
            seed=cfg.pop("seed");name=cfg["name"];sig=canonical_hash(cfg)
            manifest=read_manifest(cfg["data"]["manifest"])
            if split is not None and split!=manifest["split_sha256"]:
                raise ValueError("Selected global runs use different role manifests")
            split=manifest["split_sha256"]
            v=methods.setdefault(name,{"selected_config_signature":sig,"winner":{"config":cfg,"runs":{}}})
            if v["selected_config_signature"]!=sig:
                raise ValueError("Global selection contains multiple candidates for one method")
            v["winner"]["runs"][str(seed)]={"job_id":job_id,"run_path":job["run_dir"]}
        selection={"schema":"kd_cal_selection_v1","test_read":False,"methods":methods,"split_sha256":split}
    elif selection.get("schema") != "kd_cal_selection_v1" or selection.get("test_read") is not False:
        raise ValueError("Expected a CAL-only selection result")
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("Final seed list must be nonempty and unique")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Final config directory must be new/empty")
    output.mkdir(parents=True, exist_ok=True)
    entries=[]
    for name, value in selection["methods"].items():
        winner = value["winner"]
        cal_seeds={int(s) for s in winner["runs"]}
        if cal_seeds & set(seeds):
            raise ValueError("This final-stage recipe uses new student seeds; CAL search runs are not TEST candidates")
        for seed in seeds:
            cfg=copy.deepcopy(winner["config"]);cfg["seed"]=seed
            filename=f"{name}_seed{seed}.json"
            write_json(output/filename,cfg)
            entries.append({"config":filename,"method":name,"seed":seed,"status":"NOT_RUN",
                            "selected_config_signature":value["selected_config_signature"]})
    write_json(output/"final_grid_manifest.json",{"schema":"kd_final_grid_v1","selection_sha256":digest_file(selection_path),
              "split_sha256":selection["split_sha256"],"seeds":seeds,"entries":entries,
              "one_fixed_hyperparameter_candidate_per_method":True,"test_not_read":True,
              "note":"New initialization seeds do not create a new independent dataset."})
    return {"output":str(output),"config_count":len(entries),"seeds":seeds}


def smoke(output, arms=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Smoke output must be a new directory")
    output.mkdir(parents=True)
    prepare(output / "data", output / "roles.json", synthetic=True)
    base = {"schema": "kd_config_v1", "name": "smoke", "seed": 27182, "device": "cpu", "threads": 1,
            "track": "synthetic_smoke", "data": {"root": str(output / "data"), "manifest": str(output / "roles.json")},
            "teacher": {"backend": "tiny", "name": "tiny_teacher"}, "student": {"backend": "tiny", "name": "tiny_student"},
            "loss": {"anchor": "none", "feature": "none"},
            "optimizer": {"lr": .03, "momentum": .9, "weight_decay": .0005, "head_weight_decay": 0., "milestones": [2], "gamma": .1},
            "training": {"epochs": 2, "batch_size": 16, "workers": 0}}
    config_path = output / "teacher_config.json"; write_json(config_path, base)
    train(config_path, output / "teacher", teacher_training=True)
    base["teacher"]["checkpoint"] = str(output / "teacher" / "teacher.pt")
    base["teacher"]["provenance"] = str(output / "teacher" / "teacher_provenance.json")
    arms = arms or [a for a in ARMS if "crd" not in a]
    paths = []
    for arm in arms:
        cfg = arm_config(base, arm, .1)
        if "dependence" in cfg["loss"]:
            cfg["loss"]["dependence"].update(rank=4, solver_max_iter=10)
        cfg_path = output / (arm + ".json"); write_json(cfg_path, cfg)
        run_dir = output / arm
        train(cfg_path, run_dir)
        paths.append(run_dir)
    freeze(paths, output / "freeze.json")
    metrics = [evaluate(p, output / "freeze.json") for p in paths]
    compare(paths, output / "comparison.json")
    result = {"status": "CPU_SYNTHETIC_EXECUTION_ONLY", "arms_completed": arms, "n_arms": len(arms),
              "teacher_freeze_checked": True, "freeze_and_test_path_checked": True,
              "no_benchmark_accuracy_claim": True, "output": str(output)}
    write_json(output / "smoke_result.json", result)
    return result


def register_public_teacher(checkpoint, architecture, output, source_url):
    from .data import digest_file
    if Path(output).exists():
        raise FileExistsError("Teacher provenance cannot be overwritten")
    result = {"schema": "kd_teacher_provenance_v1", "architecture": architecture,
              "training_role": "official_full_train_CAL_exposed", "checkpoint_sha256": digest_file(checkpoint),
              "source_url": source_url, "selection_rule": "published_checkpoint_as_distributed",
              "test_used_for_selection": None, "training_exposure_basis": "upstream_report_full_CIFAR100_training_set",
              "confirmed_independent_CAL": False, "use_scope": "sensitivity_only"}
    write_json(output, result)
    return result


def profile_run(run_path, output, batch_sizes, steps, warmup):
    import time
    import torch
    from .data import digest_file, read_manifest
    from .models import create_model, load_model_checkpoint
    from .runner import _device, _teacher, environment
    if steps < 1 or warmup < 1 or any(b < 1 for b in batch_sizes):
        raise ValueError("Positive profiling steps, warmup and batches required")
    run = json.loads((Path(run_path) / "run.json").read_text()); cfg = run["config"]
    manifest = read_manifest(cfg["data"]["manifest"]); device = _device(cfg)
    student = create_model(cfg["student"], manifest["num_classes"], cfg["seed"])
    load_model_checkpoint(student, Path(run_path) / "best.pt"); student.to(device).eval()
    teacher, _, _ = _teacher(cfg, manifest, device)
    rows = []
    with torch.no_grad():
        for name, model in (("student", student), ("teacher", teacher)):
            for batch in batch_sizes:
                images = torch.randn(batch, 3, 32, 32, device=device)
                for _ in range(warmup):
                    model(images)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    base_memory = torch.cuda.memory_allocated(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(steps):
                        model(images)
                    end.record(); torch.cuda.synchronize(device)
                    seconds = start.elapsed_time(end) / 1000.
                    peak_extra = torch.cuda.max_memory_allocated(device) - base_memory
                else:
                    start = time.perf_counter()
                    for _ in range(steps):
                        model(images)
                    seconds = time.perf_counter() - start
                    peak_extra = None
                rows.append({"model": name, "batch_size": batch, "steps": steps, "warmup": warmup,
                             "milliseconds_per_batch": 1000. * seconds / steps,
                             "milliseconds_per_image": 1000. * seconds / (steps * batch),
                             "cuda_peak_activation_increment_bytes": peak_extra,
                             "parameters": sum(p.numel() for p in model.parameters())})
    result = {"schema": "kd_profile_v1", "scope": "eval_forward_random_32x32_inputs_no_dataloader_no_TEST",
              "student_checkpoint_sha256": digest_file(Path(run_path) / "best.pt"), "environment": environment(), "rows": rows}
    write_json(output, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("--data-root", required=True); q.add_argument("--manifest", required=True)
    q.add_argument("--seed", type=int, default=20261019); q.add_argument("--cal-per-class", type=int, default=50)
    q.add_argument("--download", action="store_true"); q.add_argument("--synthetic", action="store_true")
    for name in ("train", "train-teacher"):
        q = sub.add_parser(name); q.add_argument("--config", required=True); q.add_argument("--out", required=True); q.add_argument("--resume", action="store_true")
    q = sub.add_parser("freeze"); q.add_argument("--runs", nargs="+", required=True); q.add_argument("--out", required=True)
    q = sub.add_parser("evaluate"); q.add_argument("--run", required=True); q.add_argument("--freeze", required=True)
    q = sub.add_parser("compare"); q.add_argument("--runs", nargs="+", required=True); q.add_argument("--out", required=True)
    q = sub.add_parser("select-cal"); q.add_argument("--runs", nargs="+", required=True); q.add_argument("--out", required=True)
    q = sub.add_parser("make-final-grid"); q.add_argument("--selection",required=True);q.add_argument("--out",required=True)
    q.add_argument("--seeds",nargs="+",type=int,default=[31415,16180,57721])
    q = sub.add_parser("make-grid"); q.add_argument("--base", required=True); q.add_argument("--out", required=True)
    q.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS[:9]))
    q.add_argument("--seeds", nargs="+", type=int, default=[27182]); q.add_argument("--coefficients", nargs="+", type=float, default=[.01, .1, 1., 10., 100.])
    q = sub.add_parser("smoke"); q.add_argument("--out", required=True); q.add_argument("--arms", nargs="+", choices=ARMS)
    q = sub.add_parser("register-public-teacher"); q.add_argument("--checkpoint", required=True)
    q.add_argument("--architecture", required=True, choices=["wrn_40_2", "resnet32x4"]); q.add_argument("--out", required=True)
    q.add_argument("--source-url", default="https://github.com/megvii-research/mdistiller/releases/tag/checkpoints")
    q = sub.add_parser("profile"); q.add_argument("--run", required=True); q.add_argument("--out", required=True)
    q.add_argument("--batch-sizes", nargs="+", type=int, default=[1,64]); q.add_argument("--steps", type=int, default=100); q.add_argument("--warmup", type=int, default=20)
    args = p.parse_args()
    if args.command == "prepare":
        result = prepare(args.data_root, args.manifest, args.seed, args.cal_per_class, args.download, args.synthetic)
        result = {k: v for k, v in result.items() if not k.endswith("_ids")}
    elif args.command in ("train", "train-teacher"):
        run = train(args.config, args.out, args.command == "train-teacher", args.resume)
        result = {"status": run["status"], "output": args.out, "epochs": run["epochs_completed"]}
    elif args.command == "freeze":
        result = freeze(args.runs, args.out)
    elif args.command == "evaluate":
        result = evaluate(args.run, args.freeze)
    elif args.command == "compare":
        result = compare(args.runs, args.out)
    elif args.command == "select-cal":
        result = select_cal(args.runs, args.out)
    elif args.command == "make-final-grid":
        result = make_final_grid(args.selection,args.out,args.seeds)
    elif args.command == "make-grid":
        result = make_grid(args.base, args.out, args.arms, args.seeds, args.coefficients)
    elif args.command == "register-public-teacher":
        result = register_public_teacher(args.checkpoint, args.architecture, args.out, args.source_url)
    elif args.command == "profile":
        result = profile_run(args.run, args.out, args.batch_sizes, args.steps, args.warmup)
    else:
        result = smoke(args.out, args.arms)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
