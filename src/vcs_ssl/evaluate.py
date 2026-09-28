"""Frozen-feature evaluation CLI (spec §10):

    python -m vcs_ssl.evaluate --run-dir "$RUN_DIR" --checkpoint epoch_020.pt --protocol pilot

Loads the run's *frozen* resolved config + manifest, rebuilds a fresh model copy from the checkpoint, extracts clean
512-d ``h`` for fit (45k) and selection (5k) into a keyed float32 cache, then runs the fixed-budget linear probe, the kNN
monitor, the spectrum diagnostics and (VCS only) the critic hold-out diagnostic.  Nothing here trains encoder/projector/critic.

Final round (P67): ``--protocol final_official_test`` keeps the head, the kNN bank (the 45k fit set), the probe hyper-parameters
and the clean transform of ``pilot`` and scores the 10,000 official test images once per (run, checkpoint); it needs the
double unlock (``VCS_FINAL_ROUND=1``) and refuses to overwrite an existing result unless ``--force``.  ``--standin-selection``
runs the same code path on the 5k selection split (smoke only) and never touches the test partition.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import load_checkpoint
from .config import load_resolved
from .data.cifar import load_cifar10_train
from .data.splits import load_manifest
from .data.transforms import build_clean_transform, build_two_view_transform, clean_transform_signature
from .diagnostics import critic_holdout, extract_features, knn_eval, linear_probe, spectrum_report
from .models import build_models
from .utils import Timer, apply_precision_policy, atomic_write_json, environment_info, precision_flags, sha256_file, state_dict_sha256, utc_now


def feature_cache_key(*, ckpt_sha: str, split_sha: str, transform_sha: str, dtype: str, uids: np.ndarray, which: str) -> str:
    h = hashlib.sha256()
    for part in (ckpt_sha, split_sha, transform_sha, dtype, which):
        h.update(part.encode())
    h.update(np.ascontiguousarray(uids, dtype=np.int64).tobytes())
    return h.hexdigest()


def cached_features(cache_dir: Path, key: str, compute) -> dict[str, Any]:
    p = cache_dir / f"{key}.pt"
    if p.is_file():
        out = torch.load(p, weights_only=True)
        out["_cache"] = "hit"
        return out
    out = compute()
    out = {k: v for k, v in out.items() if isinstance(v, torch.Tensor)} | {"seconds": out["seconds"]}
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    torch.save(out, tmp)
    tmp.replace(p)
    out["_cache"] = "miss"
    return out


PROTOCOLS = ("pilot", "final_official_test")


def evaluate_run(run_dir: Path, checkpoint: str, *, protocol: str = "pilot", device: torch.device | None = None,
                 num_workers: int = 4, standin_selection: bool = False, force: bool = False, data=None, test_data=None) -> dict[str, Any]:
    """``pilot``: the frozen first-round protocol (unchanged).  ``final_official_test``: see :func:`evaluate_final_official_test`.

    ``data`` / ``test_data`` inject already-loaded partitions (tests); by default they are loaded from the run's data root.
    """
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}; defined: {PROTOCOLS}")
    if protocol == "final_official_test":
        return evaluate_final_official_test(run_dir, checkpoint, device=device, num_workers=num_workers,
                                            standin_selection=standin_selection, force=force, data=data, test_data=test_data)
    device = device or (torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu"))
    cfg = load_resolved(run_dir / "config.resolved.yaml")
    apply_precision_policy(cfg["train"])
    manifest = load_manifest(run_dir / "manifest.json", expected_seed=cfg["data"]["split_seed"], expected_val_per_class=cfg["data"]["val_per_class"])
    ckpt_path = run_dir / "checkpoints" / checkpoint
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    ckpt_sha = sha256_file(ckpt_path)
    ck = load_checkpoint(ckpt_path)
    if ck["config_hash"] != cfg["_meta"]["config_hash_recomputed"] and ck["config_hash"] != cfg["_meta"].get("config_hash"):
        raise ValueError("checkpoint config_hash does not match the run's frozen resolved config")
    if ck["manifest_hash"] != manifest["manifest_sha256"]:
        raise ValueError("checkpoint manifest_hash does not match the run's manifest")

    data = data if data is not None else load_cifar10_train(cfg["data"]["root"])
    if manifest["raw_file_hashes"] != data.file_hashes:
        raise ValueError("manifest raw file hashes differ from the data on disk")
    fit_uids = np.asarray(manifest["fit_uids"], dtype=np.int64)
    sel_uids = np.asarray(manifest["selection_uids"], dtype=np.int64)
    clean = build_clean_transform(cfg["views"])
    tsig = clean_transform_signature(cfg["views"])
    ecfg = cfg["evaluation"]
    dtype = ecfg["linear"]["feature_cache_dtype"]
    if dtype != "float32":
        raise ValueError("feature cache dtype must be float32 in the pilot")

    devices = [device.index or 0] if device.type == "cuda" else []
    result: dict[str, Any] = {"run_id": run_dir.name, "method": cfg["run"]["method"], "seed": cfg["run"]["seed"], "checkpoint": checkpoint,
                              "checkpoint_sha256": ckpt_sha, "checkpoint_completed_epoch": ck["completed_epoch"], "checkpoint_step": ck["optimizer_step"],
                              "split_hash": manifest["manifest_sha256"], "clean_transform_sha256": tsig, "protocol": protocol,
                              "device": str(device), "gpu": environment_info().get("gpus"), "precision_flags": precision_flags(), "utc": utc_now()}
    with Timer(device) as total_t, torch.random.fork_rng(devices=devices):
        torch.manual_seed(0)
        built = build_models(cfg, seed=int(cfg["run"]["seed"]), device=device)
        enc, proj, crit = built["encoder"], built["projector"], built["critic"]
        enc.load_state_dict(ck["encoder_state"])
        proj.load_state_dict(ck["projector_state"])
        if crit is not None:
            crit.load_state_dict(ck["critic_state"])
        ev_pred, ev_teacher = built.get("predictor"), built.get("teacher")
        if ev_pred is not None and ck.get("predictor_state") is not None:
            ev_pred.load_state_dict(ck["predictor_state"])
        if ev_teacher is not None and ck.get("teacher_encoder_state") is not None:
            ev_teacher["encoder"].load_state_dict(ck["teacher_encoder_state"])
            ev_teacher["projector"].load_state_dict(ck["teacher_projector_state"])
        for m in (enc, proj, crit, ev_pred, *((ev_teacher["encoder"], ev_teacher["projector"]) if ev_teacher else ())):
            if m is not None:
                m.eval()
                for p in m.parameters():
                    p.requires_grad_(False)
        enc_hash_before = state_dict_sha256(enc)

        cache_dir = run_dir / "features"
        k_fit = feature_cache_key(ckpt_sha=ckpt_sha, split_sha=manifest["manifest_sha256"], transform_sha=tsig, dtype=dtype, uids=fit_uids, which="fit_h")
        k_sel = feature_cache_key(ckpt_sha=ckpt_sha, split_sha=manifest["manifest_sha256"], transform_sha=tsig, dtype=dtype, uids=sel_uids, which="sel_hpz")
        fit = cached_features(cache_dir, k_fit, lambda: extract_features(enc, None, data.data, data.targets, fit_uids, clean, device=device,
                                                                          num_workers=num_workers, seed=11))
        sel = cached_features(cache_dir, k_sel, lambda: extract_features(enc, proj, data.data, data.targets, sel_uids, clean, device=device,
                                                                          num_workers=num_workers, seed=12, l2_eps=cfg["model"]["normalization"]["eps"]))
        result["feature_cache"] = {"fit_key": k_fit, "fit": fit["_cache"], "sel_key": k_sel, "sel": sel["_cache"],
                                   "fit_extract_seconds": fit["seconds"], "sel_extract_seconds": sel["seconds"], "dtype": dtype}

        # main endpoint: frozen linear probe on h (fit labels train the head, selection labels score it)
        lp = linear_probe(fit["h"], fit["labels"], sel["h"], sel["labels"], ecfg["linear"], device=device)
        result["linear"] = lp
        result["linear_val_top1_pct"] = lp["linear_val_top1_pct"]
        result["linear_val_ce"] = lp["linear_val_ce"]
        # kNN monitor
        kc = ecfg["knn"]
        knn = knn_eval(fit["h"], fit["labels"], sel["h"], sel["labels"], k=kc["k"], temperature=kc["temperature"], chunk=kc["query_chunk"], device=device)
        result["knn"] = knn
        result["knn_val_top1_pct"] = knn["knn_val_top1_pct"]
        # spectrum
        sc = ecfg["spectrum"]
        if sc["enabled"]:
            n = int(sc["selection_first_sorted_ids"])
            spec = spectrum_report({k: sel[k][:n] for k in ("h", "p_raw", "z_l2")}, names=tuple(sc["features"]), center=sc["center"], ddof=sc["covariance_ddof"])
            spec["_uids_sha256"] = hashlib.sha256(sel_uids[:n].tobytes()).hexdigest()
            result["spectrum"] = spec
            result["h_effective_rank"] = spec["h"]["effective_rank"]
            result["z_effective_rank"] = spec["z_l2"]["effective_rank"]
        # critic hold-out (VCS only)
        cv = ecfg["critic_validation"]
        if cv["enabled"] and crit is not None:
            ch = critic_holdout(enc, proj, crit, data.data, sel_uids, build_two_view_transform(cfg["views"]), device=device, batch_size=cv["batch_size"],
                                repeats=cv["repeats"], rng_seed=cv["rng_seed"], k=int(cfg["pairing"]["k"]), num_workers=num_workers,
                                l2_eps=cfg["model"]["normalization"]["eps"], normalize_input=cfg["model"]["normalization"]["vcs_and_simclr"] != "none",
                                symmetric=cfg["pairing"]["sampler"] == "random_nonzero_cyclic_shift_symmetric",
                                feature_source=cfg["model"]["critic"].get("feature_source", "z"),
                                target_branch=cfg["train"].get("target_branch", "shared"), teacher=ev_teacher, predictor=ev_pred)
            result["critic_holdout"] = ch
            result["heldout_J"] = ch["heldout_J_mean"]
            result["heldout_J_sd"] = ch["heldout_J_sd"]
        enc_hash_after = state_dict_sha256(enc)
        result["encoder_state_unchanged"] = enc_hash_before == enc_hash_after
        if not result["encoder_state_unchanged"]:
            raise RuntimeError("encoder state changed during evaluation")
    result["eval_seconds"] = total_t.elapsed
    tag = Path(checkpoint).stem
    ed = run_dir / "evaluations"
    ed.mkdir(exist_ok=True)
    atomic_write_json(ed / f"linear_{tag}.json", {k: v for k, v in result.items() if k not in ("spectrum", "critic_holdout", "knn")} | {"linear": lp})
    atomic_write_json(ed / f"knn_full_{tag}.json", knn | {"checkpoint": checkpoint, "checkpoint_sha256": ckpt_sha})
    if "spectrum" in result:
        atomic_write_json(ed / f"spectrum_full_{tag}.json", result["spectrum"])
    if "critic_holdout" in result:
        atomic_write_json(ed / f"critic_holdout_full_{tag}.json", result["critic_holdout"])
    atomic_write_json(ed / f"evaluation_{tag}.json", result)
    return result


def _load_run_and_models(run_dir: Path, checkpoint: str, device: torch.device):
    """Shared with the final protocol: frozen config, manifest, checkpoint checks and a fresh eval-mode model copy."""
    cfg = load_resolved(run_dir / "config.resolved.yaml")
    apply_precision_policy(cfg["train"])
    manifest = load_manifest(run_dir / "manifest.json", expected_seed=cfg["data"]["split_seed"], expected_val_per_class=cfg["data"]["val_per_class"])
    ckpt_path = run_dir / "checkpoints" / checkpoint
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    ckpt_sha = sha256_file(ckpt_path)
    ck = load_checkpoint(ckpt_path)
    if ck["config_hash"] != cfg["_meta"]["config_hash_recomputed"] and ck["config_hash"] != cfg["_meta"].get("config_hash"):
        raise ValueError("checkpoint config_hash does not match the run's frozen resolved config")
    if ck["manifest_hash"] != manifest["manifest_sha256"]:
        raise ValueError("checkpoint manifest_hash does not match the run's manifest")
    built = build_models(cfg, seed=int(cfg["run"]["seed"]), device=device)
    enc, proj = built["encoder"], built["projector"]
    enc.load_state_dict(ck["encoder_state"])
    proj.load_state_dict(ck["projector_state"])
    for m in (enc, proj):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
    return cfg, manifest, ck, ckpt_sha, enc, proj


def evaluate_final_official_test(run_dir: Path, checkpoint: str, *, device: torch.device | None = None, num_workers: int = 4,
                                 standin_selection: bool = False, force: bool = False, data=None, test_data=None) -> dict[str, Any]:
    """P67 — one-shot official test evaluation.

    Identical to ``pilot`` in everything that is a choice: the linear head (same hyper-parameters, trained on the 45k fit
    ``h``), the kNN monitor (fit set as bank), the clean transform and the checkpoint.  Only the scored images change:
    the 10,000 official test images (``standin_selection=False``) or, for smoke tests only, the 5k selection split
    (``standin_selection=True``; the test partition is then never loaded).  Results go to
    ``<run-dir>/evaluations/final_official_test_<tag>.json`` (or ``final_standin_<tag>.json``); an existing official
    result is never overwritten unless ``force``.
    """
    device = device or (torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu"))
    tag = Path(checkpoint).stem
    ed = run_dir / "evaluations"
    out_path = ed / (f"final_standin_{tag}.json" if standin_selection else f"final_official_test_{tag}.json")
    if out_path.is_file() and not standin_selection and not force:
        raise FileExistsError(f"{out_path} exists: the official test evaluation of this checkpoint has already been run (use --force only to redo it knowingly)")
    if not standin_selection and test_data is None:
        from .data.cifar import official_test_unlocked  # noqa: PLC0415
        if not official_test_unlocked(True):
            raise PermissionError("final_official_test needs the final-round unlock: export VCS_FINAL_ROUND=1 (owner's rule: once, at the end)")
    cfg, manifest, ck, ckpt_sha, enc, proj = _load_run_and_models(run_dir, checkpoint, device)
    data = data if data is not None else load_cifar10_train(cfg["data"]["root"])
    if manifest["raw_file_hashes"] != data.file_hashes:
        raise ValueError("manifest raw file hashes differ from the data on disk")
    fit_uids = np.asarray(manifest["fit_uids"], dtype=np.int64)
    clean = build_clean_transform(cfg["views"])
    tsig = clean_transform_signature(cfg["views"])
    ecfg = cfg["evaluation"]
    dtype = ecfg["linear"]["feature_cache_dtype"]
    if standin_selection:
        eval_images, eval_targets = data.data, data.targets
        eval_uids = np.asarray(manifest["selection_uids"], dtype=np.int64)
        eval_split = {"name": "selection (stand-in; NOT the official test)", "n": int(len(eval_uids)), "sha256": manifest["manifest_sha256"]}
        which = "sel_h_standin"
    else:
        test = test_data if test_data is not None else load_cifar10_train(cfg["data"]["root"], train=False, allow_official_test=True)
        eval_images, eval_targets = test.data, test.targets
        eval_uids = np.arange(len(eval_targets), dtype=np.int64)
        eval_split = {"name": "official CIFAR-10 test partition", "n": int(len(eval_uids)), "sha256": test.source.get("test_batch_sha256"),
                      "file_hashes": test.file_hashes.get("test_batch")}
        which = "test_h_official"
    devices = [device.index or 0] if device.type == "cuda" else []
    result: dict[str, Any] = {"run_id": run_dir.name, "method": cfg["run"]["method"], "seed": cfg["run"]["seed"], "checkpoint": checkpoint,
                              "checkpoint_sha256": ckpt_sha, "checkpoint_completed_epoch": ck["completed_epoch"], "checkpoint_step": ck["optimizer_step"],
                              "split_hash": manifest["manifest_sha256"], "clean_transform_sha256": tsig, "protocol": "final_official_test",
                              "standin_selection": bool(standin_selection), "eval_split": eval_split, "head_and_bank": "45k fit split (as pilot)",
                              "probe_config": ecfg["linear"], "knn_config": ecfg["knn"], "device": str(device), "gpu": environment_info().get("gpus"),
                              "precision_flags": precision_flags(), "utc": utc_now()}
    with Timer(device) as total_t, torch.random.fork_rng(devices=devices):
        torch.manual_seed(0)
        enc_hash_before = state_dict_sha256(enc)
        cache_dir = run_dir / "features"
        k_fit = feature_cache_key(ckpt_sha=ckpt_sha, split_sha=manifest["manifest_sha256"], transform_sha=tsig, dtype=dtype, uids=fit_uids, which="fit_h")
        k_eval = feature_cache_key(ckpt_sha=ckpt_sha, split_sha=str(eval_split["sha256"]), transform_sha=tsig, dtype=dtype, uids=eval_uids, which=which)
        fit = cached_features(cache_dir, k_fit, lambda: extract_features(enc, None, data.data, data.targets, fit_uids, clean, device=device,
                                                                          num_workers=num_workers, seed=11))
        ev = cached_features(cache_dir, k_eval, lambda: extract_features(enc, None, eval_images, eval_targets, eval_uids, clean, device=device,
                                                                          num_workers=num_workers, seed=13))
        result["feature_cache"] = {"fit_key": k_fit, "fit": fit["_cache"], "eval_key": k_eval, "eval": ev["_cache"], "dtype": dtype}
        lp = linear_probe(fit["h"], fit["labels"], ev["h"], ev["labels"], ecfg["linear"], device=device)
        kc = ecfg["knn"]
        knn = knn_eval(fit["h"], fit["labels"], ev["h"], ev["labels"], k=kc["k"], temperature=kc["temperature"], chunk=kc["query_chunk"], device=device)
        result["linear"] = lp
        result["linear_test_top1_pct" if not standin_selection else "linear_val_top1_pct"] = lp["linear_val_top1_pct"]
        result["knn"] = knn
        result["knn_test_top1_pct" if not standin_selection else "knn_val_top1_pct"] = knn["knn_val_top1_pct"]
        result["encoder_state_unchanged"] = state_dict_sha256(enc) == enc_hash_before
        if not result["encoder_state_unchanged"]:
            raise RuntimeError("encoder state changed during evaluation")
    result["eval_seconds"] = total_t.elapsed
    ed.mkdir(exist_ok=True)
    atomic_write_json(out_path, result)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", required=True, help="file name inside <run-dir>/checkpoints, e.g. epoch_020.pt or initial.pt")
    ap.add_argument("--protocol", default="pilot", choices=list(PROTOCOLS))
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--standin-selection", action="store_true", help="final protocol, smoke only: score the selection split instead of the test partition")
    ap.add_argument("--force", action="store_true", help="final protocol: overwrite an existing official-test result (never by default)")
    args = ap.parse_args(argv)
    res = evaluate_run(Path(args.run_dir), args.checkpoint, protocol=args.protocol, num_workers=args.num_workers,
                       standin_selection=args.standin_selection, force=args.force)
    if args.protocol == "final_official_test" and not args.standin_selection:
        print(f"[{res['run_id']}] {args.checkpoint}: OFFICIAL TEST linear_top1 {res['linear_test_top1_pct']:.2f}%  knn_top1 {res['knn_test_top1_pct']:.2f}%  ({res['eval_seconds']:.0f}s)", flush=True)
    else:
        print(f"[{res['run_id']}] {args.checkpoint}: linear_val_top1 {res['linear_val_top1_pct']:.2f}%  knn_val_top1 {res['knn_val_top1_pct']:.2f}%  "
              f"h_rank {res.get('h_effective_rank')}  heldout_J {res.get('heldout_J')}  ({res['eval_seconds']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
