"""Frozen-feature evaluation CLI (spec §10):

    python -m vcs_ssl.evaluate --run-dir "$RUN_DIR" --checkpoint epoch_020.pt --protocol pilot

Loads the run's *frozen* resolved config + manifest, rebuilds a fresh model copy from the checkpoint, extracts clean
512-d ``h`` for fit (45k) and selection (5k) into a keyed float32 cache, then runs the fixed-budget linear probe, the kNN
monitor, the spectrum diagnostics and (VCS only) the critic hold-out diagnostic.  Nothing here trains encoder/projector/critic.
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


def evaluate_run(run_dir: Path, checkpoint: str, *, protocol: str = "pilot", device: torch.device | None = None,
                 num_workers: int = 4) -> dict[str, Any]:
    if protocol != "pilot":
        raise ValueError("only the 'pilot' protocol is defined in the first round")
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

    data = load_cifar10_train(cfg["data"]["root"])
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
        for m in (enc, proj, crit):
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
                                feature_source=cfg["model"]["critic"].get("feature_source", "z"))
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", required=True, help="file name inside <run-dir>/checkpoints, e.g. epoch_020.pt or initial.pt")
    ap.add_argument("--protocol", default="pilot")
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args(argv)
    res = evaluate_run(Path(args.run_dir), args.checkpoint, protocol=args.protocol, num_workers=args.num_workers)
    print(f"[{res['run_id']}] {args.checkpoint}: linear_val_top1 {res['linear_val_top1_pct']:.2f}%  knn_val_top1 {res['knn_val_top1_pct']:.2f}%  "
          f"h_rank {res.get('h_effective_rank')}  heldout_J {res.get('heldout_J')}  ({res['eval_seconds']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
