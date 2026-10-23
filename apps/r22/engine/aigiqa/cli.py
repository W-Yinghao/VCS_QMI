"""Run with python -m aigiqa.cli from the extracted package root."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
from pathlib import Path

from .manifest import INPUT_FIELDS, prepare, write_json, write_csv, read_csv, sha256_file, cross_dataset_audit


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_grid(base_config, output, stage="pilot"):
    """Write reviewable commands; never launch a GPU job from this command."""
    base = load(base_config)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refuse to overwrite a grid")
    output.mkdir(parents=True, exist_ok=True)
    methods = [
        ("task_only", {"method": "none", "forward": "none", "weight": 0.}),
        ("reference_nce_text", {"method": "reference_infonce", "forward": "none", "branches": ["text"]}),
        ("bce_text", {"method": "bce", "forward": "none", "branches": ["text"]}),
        ("vcs_exact_text", {"method": "vcs_exact", "forward": "none", "branches": ["text"]}),
        ("vcs_sampled8_text", {"method": "vcs_sampled", "forward": "none", "branches": ["text"], "sampled_draws": 8}),
        ("corrected_cmim_both", {"method": "reference_infonce", "forward": "corrected_gaussian", "branches": ["text", "visual"], "entropy_surrogate_weight": 0., "forward_weight": .1}),
    ]
    if stage == "pilot":
        seeds, weights = [3101, 3102, 3103], [.01, .1, 1., 10., 100.]
    elif stage == "ablation":
        seeds, weights = [3101, 3102, 3103], [.1]
        methods = [
            ("vcs_dense_text", {"method": "vcs_dense", "forward": "none", "branches": ["text"]}),
            ("vcs_exact_both", {"method": "vcs_exact", "forward": "none", "branches": ["text", "visual"]}),
            ("vcs_exact_visual", {"method": "vcs_exact", "forward": "none", "branches": ["visual"]}),
            ("bce_logits_text", {"method": "bce_logits", "forward": "none", "branches": ["text"]}),
            ("forward_nll_only", {"method": "none", "forward": "corrected_gaussian", "weight": 0., "forward_weight": .1}),
            ("forward_nll_entropy_surrogate", {"method": "reference_infonce", "forward": "corrected_gaussian", "branches": ["text", "visual"], "entropy_surrogate_weight": 1., "forward_weight": .1}),
        ]
        for cap in [10, 30, 100]:
            methods.append((f"vcs_m1_cap{cap}_text", {"method": "vcs_m1", "forward": "none", "branches": ["text"], "solver_max_iter": cap, "solver_relative_tol": .01}))
    else:
        raise ValueError("stage must be pilot or ablation")
    rows = []
    for label, loss in methods:
        grid_weights = [loss.get("weight", 0.)] if loss["method"] == "none" else weights
        for weight, seed in itertools.product(grid_weights, seeds):
            cfg = copy.deepcopy(base)
            cfg["method_label"], cfg["seed"] = label, seed
            cfg["loss"].update(loss)
            cfg["loss"]["weight"] = weight
            cfg["gaussian_warmup_epochs"] = 1 if loss.get("forward") == "corrected_gaussian" else 0
            name = f"{label}_w{weight:g}_s{seed}"
            cfg["output_dir"] = str(Path(base["output_dir"]) / stage / name)
            path = output / f"{name}.json"
            write_json(path, cfg)
            rows.append({"method": label, "seed": seed, "weight": weight, "config": str(path),
                         "command": f"python -m aigiqa.cli train --config {path}"})
    write_csv(output / "run_matrix.csv", rows, ["method", "seed", "weight", "config", "command"])
    write_json(output / "grid_summary.json", {"stage": stage, "run_count": len(rows), "targets": [base["target"]],
               "data_dir": base["data_dir"], "selection": "SOURCE CAL only", "no_jobs_launched": True,
               "note": "Pilot fixed rank32/batch5/temperature1. Do not combine rank/batch/temperature grids with this grid blindly."})
    return {"run_count": len(rows), "matrix": str(output / "run_matrix.csv")}


def smoke(output):
    """Exercise data prep, actual optimization, freeze, TEST predict/evaluate."""
    import numpy as np
    from PIL import Image
    from .engine import train, freeze, predict, evaluate, preflight
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Smoke output must be fresh")
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(442)
    rows = []
    # Distinct random words avoid accidental near-duplicate group merging.
    words = ["amber", "beacon", "cactus", "dahlia", "ember", "fjord", "galaxy", "harbor", "indigo", "jungle", "koala", "lotus", "maple", "nebula", "opal", "prairie", "quartz", "raven", "saffron", "tundra"]
    for i, word in enumerate(words):
        for j in range(3):
            image = f"{i:03}_{j}.png"
            arr = rng.integers(0, 90, (24, 24, 3), dtype=np.uint8) + j * 50
            Image.fromarray(arr).save(images / image)
            rows.append({"image": image, "prompt": word, "quality_mos": 1 + j + .02 * i,
                         "alignment_mos": 1 + .8 * j + .025 * i, "generator": f"g{j}"})
    annotations = root / "annotations.csv"
    write_csv(annotations, rows, ["image", "prompt", "quality_mos", "alignment_mos", "generator"])
    data = root / "prepared"
    prepare(annotations, images, data, "canonical")
    runs = []
    all_results = []
    for method_label in ["none", "reference_infonce", "bce", "vcs_exact", "vcs_sampled", "vcs_m1", "corrected_cmim"]:
        method = "reference_infonce" if method_label == "corrected_cmim" else method_label
        run = root / "runs" / method_label
        cfg = {"model": {"name": "tiny_synthetic"}, "data_dir": str(data.resolve()), "output_dir": str(run.resolve()),
               "target": "alignment", "seed": 9, "device": "cpu", "batch_size": 5, "num_workers": 0,
               "image_size": 24, "epochs": 2, "learning_rate": .003, "selection_metric": "srcc",
               "evidence_role": "SYNTHETIC_INTEGRATION_ONLY", "method_label": method_label,
               "loss": {"method": method, "forward": "none", "branches": ["text"], "rank": 4, "weight": .1,
                        "q_mode": "u", "sampled_draws": 8, "solver_max_iter": 10, "solver_relative_tol": .01}}
        if method_label == "corrected_cmim":
            cfg["loss"].update(forward="corrected_gaussian", branches=["text", "visual"], forward_weight=.1)
            cfg["gaussian_warmup_epochs"] = 1
        if method == "none":
            all_results.append({"preflight": preflight(cfg)})
        all_results.append({"method": method_label, "train": train(cfg)})
        runs.append(run)
    frozen_path = root / "frozen.json"
    frozen = freeze(runs, [data / "test_inputs.csv"], frozen_path)
    for row in frozen["selected_runs"]:
        output_pred = root / "predictions" / f"{row['run_key']}.csv"
        pred_info = predict(frozen_path, row["run_key"], data / "test_inputs.csv", images, output_pred, device="cpu")
        metric_info = evaluate(output_pred, data / "sealed" / "test_labels.csv",
                               root / "metrics" / f"{row['run_key']}.json", "alignment")
        all_results.append({"run_key": row["run_key"], "prediction": pred_info, "metrics": metric_info})
    result = {"status": "PASS", "evidence_role": "SYNTHETIC_INTEGRATION_ONLY", "n_methods": len(runs),
              "scope": "Real CPU train/CAL/freeze/predict/evaluate pipeline; no real ImageReward weights or real benchmark images tested", "results": all_results}
    write_json(root / "SMOKE_RESULT.json", result)
    return {"status": "PASS", "n_methods": len(runs), "result": str(root / "SMOKE_RESULT.json")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    x = sub.add_parser("prepare")
    x.add_argument("--source-csv", required=True); x.add_argument("--image-root", required=True)
    x.add_argument("--output", required=True); x.add_argument("--dataset", required=True,
        choices=["AGIQA-3k", "AIGCIQA2023", "PKU-AIGIQA-T2I", "canonical"])
    x.add_argument("--seed", type=int, default=20261019)
    x.add_argument("--metadata-only", action="store_true", help="Prepare an unverified draft; formal training will refuse it")
    for name in ["train", "preflight"]:
        x = sub.add_parser(name); x.add_argument("--config", required=True)
    x = sub.add_parser("freeze")
    x.add_argument("--runs", nargs="+", required=True); x.add_argument("--inputs", nargs="+", required=True); x.add_argument("--output", required=True)
    x = sub.add_parser("predict")
    x.add_argument("--freeze", required=True); x.add_argument("--run-key", required=True)
    x.add_argument("--inputs", required=True); x.add_argument("--image-root", required=True); x.add_argument("--output", required=True); x.add_argument("--device")
    x = sub.add_parser("evaluate")
    x.add_argument("--predictions", required=True); x.add_argument("--labels", required=True); x.add_argument("--output", required=True)
    x.add_argument("--target", choices=["alignment", "quality"], required=True); x.add_argument("--tie-epsilon-mos", type=float, default=.05)
    x = sub.add_parser("compare")
    x.add_argument("--a", required=True); x.add_argument("--b", required=True); x.add_argument("--labels", required=True); x.add_argument("--output", required=True)
    x.add_argument("--target", choices=["alignment", "quality"], required=True); x.add_argument("--repetitions", type=int, default=1000)
    x = sub.add_parser("import_scores")
    for name in ["scores", "inputs", "freeze", "output", "provenance"]:
        x.add_argument("--" + name, required=True)
    x.add_argument("--target", choices=["alignment", "quality"], required=True)
    x = sub.add_parser("native_imagereward")
    for name in ["config", "inputs", "image-root", "output-scores"]:
        x.add_argument("--" + name, required=True)
    x = sub.add_parser("cross_audit")
    x.add_argument("--source-dir", required=True); x.add_argument("--target-inputs", required=True); x.add_argument("--output", required=True)
    x.add_argument("--exclude-overlaps-output", help="Write target metadata subset without source prompt/image overlaps; uses no labels")
    x = sub.add_parser("make_grid")
    x.add_argument("--base-config", required=True); x.add_argument("--output", required=True)
    x.add_argument("--stage", choices=["pilot", "ablation"], default="pilot")
    x = sub.add_parser("smoke"); x.add_argument("--output", required=True)
    x = sub.add_parser("hash_assets"); x.add_argument("paths", nargs="+")
    a = p.parse_args()
    if a.command == "prepare":
        result = prepare(a.source_csv, a.image_root, a.output, a.dataset, seed=a.seed, verify_images=not a.metadata_only)
    elif a.command in {"train", "preflight"}:
        from . import engine
        result = getattr(engine, a.command)(load(a.config))
    elif a.command == "freeze":
        from .engine import freeze
        result = freeze(a.runs, a.inputs, a.output)
    elif a.command == "predict":
        from .engine import predict
        result = predict(a.freeze, a.run_key, a.inputs, a.image_root, a.output, a.device)
    elif a.command == "evaluate":
        from .engine import evaluate
        result = evaluate(a.predictions, a.labels, a.output, a.target, a.tie_epsilon_mos)
    elif a.command == "compare":
        from .engine import compare
        result = compare(a.a, a.b, a.labels, a.output, a.target, a.repetitions)
    elif a.command == "import_scores":
        from .engine import import_scores
        result = import_scores(a.scores, a.inputs, a.freeze, a.output, a.target, a.provenance)
    elif a.command == "native_imagereward":
        from .engine import native_imagereward
        result = native_imagereward(load(a.config), a.inputs, a.image_root, a.output_scores)
    elif a.command == "cross_audit":
        result = cross_dataset_audit(a.source_dir, a.target_inputs); write_json(a.output, result)
        if a.exclude_overlaps_output:
            if Path(a.exclude_overlaps_output).exists():
                raise FileExistsError("Do not overwrite cross-dataset inputs")
            excluded = {r["sample_id"] for r in result["overlaps"]}
            kept = [r for r in read_csv(a.target_inputs) if r["sample_id"] not in excluded]
            if not kept:
                raise ValueError("No target images remain after source-overlap exclusion")
            write_csv(a.exclude_overlaps_output, kept, INPUT_FIELDS)
            result["filtered_inputs"] = a.exclude_overlaps_output
            result["filtered_images"] = len(kept)
            write_json(a.output, result)
    elif a.command == "make_grid":
        result = make_grid(a.base_config, a.output, a.stage)
    elif a.command == "smoke":
        result = smoke(a.output)
    else:
        result = {str(Path(path)): sha256_file(path) for path in a.paths}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
