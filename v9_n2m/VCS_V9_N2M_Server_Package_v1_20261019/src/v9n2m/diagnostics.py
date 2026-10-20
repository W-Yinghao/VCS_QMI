"""Inference-only, fixed-student loss export; evaluator labels are not read here."""
from __future__ import annotations

import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .common import read_json, sha256_file, utc_now, write_csv, write_json
from .identity import validate_complete_unit, validate_vcs_reference, verify_training_inputs
from .prepare import verify_experiment
from .runner import load_training_data
from .runtime import collect_environment, configure_precision, environment_differences


def export_fixed_losses(experiment_path, seed, out_csv):
    """Eight prespecified full batches at the inherited VCS best-DEV checkpoint.

    This uses the same observed batch and frozen model for every counterfactual
    weighting. It is descriptive; it is not a clean-negative-pool oracle risk.
    """
    from .controls import load_observed
    exp = verify_experiment(experiment_path)
    unit = next(u for u in exp["units"] if u["seed"] == int(seed) and u["arm"] == "VCS")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run inference in an allocated server Slurm GPU job")
    out_csv = Path(out_csv).resolve()
    meta_path = Path(str(out_csv) + ".meta.json")
    if out_csv.exists() or meta_path.exists():
        raise ValueError("Fixed-loss export already exists; preserve it and choose a new diagnostic output path")
    start = time.monotonic()
    configure_precision()
    import numpy as np
    import torch
    hardware = collect_environment()
    if not torch.cuda.is_available() or environment_differences(exp["target_environment"], hardware):
        raise ValueError("Fixed-loss diagnostic must use the frozen CUDA environment")
    verify_training_inputs(exp["base_handoff"], exp["inputs"], seeds=(int(seed),))
    reference = Path(unit["historical_reference_dir"])
    validate_complete_unit(reference, expected_identity={"method": "VCS_Weight", "seed": int(seed), "scale": "sum",
                                                       "freeze_sha256": exp["base_freeze_sha256"]}, check_binary=True)
    validate_vcs_reference(reference, exp["base_handoff"], int(seed))
    rows = load_observed(unit["observed_path"])
    cfg, train, dev_images, dev_captions, collate = load_training_data(exp)
    from v9.n2.teacher import build_host, iterate_batches
    from v9.n2.sgr_host import hinge_per_anchor
    checkpoint = reference / "best.pt"
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    host = build_host(cfg).to("cuda")
    host.load_state_dict(ck["model"])
    host.eval()
    rng = np.random.default_rng([int(seed), 9020, 5])
    records, batches = [], []
    with torch.no_grad():
        for batch_id, idx in enumerate(iterate_batches(len(rows), 128, rng)):
            if batch_id >= 8:
                break
            selected = [rows[i] for i in idx]
            ims, caps, lens = collate(selected)
            sims = host(ims.to("cuda"), caps.to("cuda"), lens)
            er, ec = hinge_per_anchor(sims, cfg.margin, hard_negative=True)
            ell = (er + ec).detach().cpu().numpy().astype(np.float64)
            if len(selected) != 128 or not np.isfinite(ell).all() or np.any(ell < 0):
                raise ValueError("Invalid exported per-anchor hard-hinge losses")
            for row, loss in zip(selected, ell):
                records.append({"batch_id": batch_id, "pair_id": row["pair_id"], "ell": format(float(loss), ".17g")})
            batches.append({"batch_id": batch_id, "n": len(idx), "pool_indices": idx.tolist()})
    write_csv(out_csv, records)
    metadata = {"schema_version": 1, "purpose": "EVAL_ONLY fixed-student counterfactual weighting diagnostic",
                "created_utc": utc_now(), "seed": int(seed), "experiment_sha256": sha256_file(experiment_path),
                "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
                "checkpoint_selection": "Original complete N2F VCS SUM best full DEV; same checkpoint for every counterfactual arm",
                "checkpoint_epoch": int(ck["epoch"]), "observed_sha256": unit["observed_sha256"],
                "loss_csv_sha256": sha256_file(out_csv),
                "batch_definition": "First eight full batches from default_rng([seed,9020,5]) permutation of the unchanged full observed pool; fixed before reading any mechanism label",
                "ell_definition": "row+column per-anchor hinge, margin=.2, hard_negative=True; inherited SGR in eval mode",
                "same_negative_pool": True, "weights_applied_during_export": False,
                "evaluator_labels_read": False, "test_read": False, "batches": batches,
                "host_config": asdict(cfg), "hardware": hardware, "elapsed_s": time.monotonic()-start,
                "scope": "Frozen-model inference/load walltime, not new full student training",
                "limitation": "Observed noisy negatives and DEV-selected VCS-trained checkpoint are retained; values do not estimate retrained endpoints or all-clean negatives."}
    write_json(meta_path, metadata)
    return metadata
