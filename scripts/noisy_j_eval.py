"""P104 addendum 2 (package v3 §7.3) — evaluation-only clean and noisy held-out J on saved checkpoints, so that N1 (noise τ 0.3, R = 4 loss averaging)
and the P95 τ 0.3 runs (R = 1) are compared with the SAME evaluation: `vcs_ssl.diagnostics.critic_holdout` exactly as the training evaluation calls it
(same selection images, two-view transform, repeats, rng_seed, K, eval mode), with noise_tau from the run's config and noise_repeats = R_eval (16).
Read-only: builds a fresh model per checkpoint; the run directory is never written; the official test set is never touched.

    python scripts/noisy_j_eval.py --run-dir outputs/P95_noise_tau0.3_views4_800ep_seed0 --checkpoints epoch_100.pt,epoch_800.pt --r-eval 16 --out X.json
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from vcs_ssl.checkpoint import load_checkpoint  # noqa: E402
from vcs_ssl.config import load_resolved  # noqa: E402
from vcs_ssl.data.cifar import load_cifar10_train  # noqa: E402
from vcs_ssl.data.splits import load_manifest  # noqa: E402
from vcs_ssl.data.transforms import build_two_view_transform  # noqa: E402
from vcs_ssl.diagnostics import critic_holdout  # noqa: E402
from vcs_ssl.models import build_models  # noqa: E402
from vcs_ssl.utils import atomic_write_json, utc_now  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True); ap.add_argument("--checkpoints", default="epoch_100.pt,epoch_200.pt,epoch_400.pt,epoch_600.pt,epoch_800.pt")
    ap.add_argument("--r-eval", type=int, default=16); ap.add_argument("--out", required=True); ap.add_argument("--num-workers", type=int, default=6)
    a = ap.parse_args()
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    run_dir = Path(a.run_dir); cfg = load_resolved(run_dir / "config.resolved.yaml"); man = load_manifest(run_dir / "manifest.json")
    tau = float(cfg["model"]["critic"].get("observation_noise_tau", 0.0)); assert tau > 0, "not a noisy-critic run"
    cv = cfg["evaluation"]["critic_validation"]; data = load_cifar10_train(cfg["data"]["root"])  # official TRAIN file only
    sel = np.asarray(man["selection_uids"], dtype=np.int64); two_view = build_two_view_transform(cfg["views"])
    out = {"run": run_dir.name, "created_utc": utc_now(), "device": str(device), "noise_tau": tau, "r_eval": a.r_eval,
           "critic_validation": {k: cv[k] for k in ("repeats", "batch_size", "rng_seed")}, "k": int(cfg["pairing"]["k"]), "checkpoints": {}}
    for name in [c for c in a.checkpoints.split(",") if c]:
        p = run_dir / "checkpoints" / name
        if not p.is_file():
            out["checkpoints"][name] = {"missing": True}; continue
        t0 = time.time()
        devices = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(0)  # as the training evaluation
            built = build_models(cfg, seed=int(cfg["run"]["seed"]), device=device); ck = load_checkpoint(p)
            enc, proj, crit = built["encoder"], built["projector"], built["critic"]
            enc.load_state_dict(ck["encoder_state"]); proj.load_state_dict(ck["projector_state"]); crit.load_state_dict(ck["critic_state"])
            assert getattr(crit, "is_noisy", False)
            ch = critic_holdout(enc, proj, crit, data.data, sel, two_view, device=device, feature_source=cfg["model"]["critic"].get("feature_source", "z"),
                                batch_size=cv["batch_size"], repeats=cv["repeats"], rng_seed=cv["rng_seed"], k=int(cfg["pairing"]["k"]),
                                num_workers=a.num_workers, l2_eps=cfg["model"]["normalization"]["eps"],
                                normalize_input=cfg["model"]["normalization"]["vcs_and_simclr"] != "none", noise_tau=tau, noise_repeats=a.r_eval)
        keep = ("heldout_J_mean", "heldout_J_sd", "heldout_J_noisy_mean", "heldout_gate_clean_mean", "heldout_gate_noisy_mean")
        out["checkpoints"][name] = {"epoch": ck.get("epoch"), **{k: ch.get(k) for k in keep},
                                    "a": float(crit.scale), "b": float(crit.bias), "seconds": time.time() - t0}
        print(f"{run_dir.name} {name}: J clean {ch['heldout_J_mean']:.4f} noisy {ch['heldout_J_noisy_mean']:.4f} "
              f"gate clean {ch['heldout_gate_clean_mean']:.4f} noisy {ch['heldout_gate_noisy_mean']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        del enc, proj, crit, built, ck
    atomic_write_json(Path(a.out), out)
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
