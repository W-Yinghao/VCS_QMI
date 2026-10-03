"""P115 diagnostics runner (v5 NEXT-A-AUG §3 "对应诊断"): read-only, per run and checkpoint; see src/vcs_diag/augdiag.py for what is recorded.

    python scripts/p115_diag.py --runs P107_AP3_views4_800ep_seed0,P41_simclr_views4_800ep_seed0 [--checkpoints 20,100,400,800] \
        [--grad-batches 2] [--geo-images 2000] [--out-dir /home/infres/yinwang/CS_QMI/outputs/P115_diag_results] [--smoke] [--cpu]

Fixed across runs (all CIFAR-10 dev45k/val5k runs share the manifest): base images for gradients / distributions = the first
grad_batches × B FIT uids after a seeded permutation; geometry images = the first geo_images SELECTION uids after a seeded permutation;
augmentation draws come from a forked, seeded torch RNG (identical views for every run and checkpoint of the same view count).
Requested epochs that were not saved map to the nearest trained checkpoint (recorded; initial.pt is never used).  One JSON per run;
an existing JSON is skipped (resumable).  The run directory is never written; the official test set is never read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO, REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from p104_diagnostics import load  # noqa: E402  (imported, not edited)
from vcs_diag.augdiag import batch_diagnostics, geometry, known_box_views, resolve_checkpoints  # noqa: E402
from vcs_ssl.config import load_resolved  # noqa: E402
from vcs_ssl.data.cifar import load_cifar10_train  # noqa: E402
from vcs_ssl.data.splits import load_manifest  # noqa: E402
from vcs_ssl.data.transforms import build_clean_transform, build_two_view_transform  # noqa: E402
from vcs_ssl.utils import atomic_write_json, utc_now  # noqa: E402

OUT = Path("/home/infres/yinwang/CS_QMI/outputs")


def mean_tree(items: list):
    """Mean of numeric leaves across batches (lists of equal-length numbers averaged elementwise; other leaves from the first item)."""
    items = [x for x in items if x is not None]
    if not items:
        return None
    a = items[0]
    if isinstance(a, dict):
        keys = list(dict.fromkeys(k for it in items for k in it))
        return {k: mean_tree([it.get(k) for it in items]) for k in keys}
    if isinstance(a, (int, float)) and not isinstance(a, bool) and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in items):
        return float(np.mean(items))
    if isinstance(a, list) and a and all(isinstance(x, list) and len(x) == len(a) for x in items):
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for x in items for v in x):
            return [float(v) for v in np.mean(np.array(items, dtype=float), axis=0)]
        if all(isinstance(v, dict) for x in items for v in x):
            return [mean_tree([x[j] for x in items]) for j in range(len(a))]
        try:  # nested numeric lists of equal shape (e.g. cosine matrices)
            return np.mean(np.array(items, dtype=float), axis=0).round(5).tolist()
        except (ValueError, TypeError):
            pass
    return a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True); ap.add_argument("--checkpoints", default="20,100,400,800")
    ap.add_argument("--grad-batches", type=int, default=2); ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--geo-images", type=int, default=2000); ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--pair-seed", type=int, default=99); ap.add_argument("--out-dir", default=str(OUT / "P115_diag_results"))
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    device = torch.device("cpu") if a.cpu or not torch.cuda.is_available() else torch.device("cuda", 0)
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    req = [int(x) for x in a.checkpoints.split(",") if x]
    data = None
    for run in [r for r in a.runs.split(",") if r]:
        dst = od / f"{run}.json"
        if dst.exists():
            print(f"[skip] {run}: {dst} exists", flush=True); continue
        rd = OUT / run
        st = json.load(open(rd / "status.json"))
        if st.get("status") != "COMPLETED":
            print(f"[skip] {run}: status {st.get('status')}", flush=True); continue
        cfg0 = load_resolved(rd / "config.resolved.yaml"); man = load_manifest(rd / "manifest.json")
        if data is None:
            data = load_cifar10_train(cfg0["data"]["root"])  # official TRAIN file only
        V = int(cfg0["views"].get("count", 2)); B = int(a.batch or cfg0["train"]["batch_size_images"]); nb = a.grad_batches; ng = a.geo_images
        if a.smoke:
            B, nb, ng = 16, 1, 64
        fit = np.random.default_rng(a.seed).permutation(np.asarray(man["fit_uids"], dtype=np.int64))[: B * nb]
        sel = np.random.default_rng(a.seed + 1).permutation(np.asarray(man["selection_uids"], dtype=np.int64))[:ng]
        tf, clean = build_two_view_transform(cfg0["views"]), build_clean_transform(cfg0["views"])
        t_aug = time.time()
        batches = [known_box_views(data.data, fit[i * B:(i + 1) * B], tf, V, seed=a.seed + 100 + i) for i in range(nb)]
        rec = {"run": run, "created_utc": utc_now(), "device": str(device), "method": cfg0["run"]["method"], "views": V, "batch": B,
               "grad_batches": nb, "geo_images": ng, "diag_seed": a.seed, "pair_seed": a.pair_seed,
               "pair_scope": cfg0["pairing"].get("pair_scope", "cross_view_k"), "negative_detach": cfg0["pairing"].get("negative_detach"),
               "augmentation": {"random_resized_crop": cfg0["views"]["random_resized_crop"], "color_jitter": cfg0["views"]["color_jitter"]},
               "base_uids_sha256": hashlib.sha256(fit.tobytes()).hexdigest(), "geo_uids_sha256": hashlib.sha256(sel.tobytes()).hexdigest(),
               "augment_seconds": time.time() - t_aug, "smoke": a.smoke,
               "checkpoint_map": resolve_checkpoints(sorted(p.name for p in (rd / "checkpoints").iterdir()), req), "checkpoints": {}}
        for cm in rec["checkpoint_map"]:
            if cm.get("checkpoint") is None or cm.get("duplicate"):
                continue
            name = cm["checkpoint"]; t0 = time.time()
            if device.type == "cuda":
                torch.zeros(0, device=device)  # initialise the CUDA context first (reset before init raises "Invalid device argument", cf. P85)
                try:
                    torch.cuda.reset_peak_memory_stats(device)
                except RuntimeError:
                    pass
            cfg, ck, enc, proj, crit = load(rd, name, device)
            r = {"epoch": ck.get("epoch"), "step": ck.get("step")}
            r["geometry"] = geometry(enc, proj, data.data, sel, clean, device, cfg["model"]["normalization"]["eps"])
            per = [batch_diagnostics(cfg, enc, proj, crit, views, boxes, device, pair_seed=a.pair_seed + i) for i, (views, boxes) in enumerate(batches)]
            r["batches"] = per; r["mean_over_batches"] = mean_tree(per)
            r["seconds"] = time.time() - t0
            r["peak_mem_mb"] = (torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else None
            rec["checkpoints"][name] = r
            print(f"[{run}] {name}: {r['seconds']:.0f}s  L {r['mean_over_batches']['loss']:.4f}  share(iou<0.1) "
                  f"{r['mean_over_batches']['crop_groups']['iou'][0].get('share_of_P_grad')}  cancel {r['mean_over_batches']['view_pairs']['cancellation_ratio']:.3f}", flush=True)
            del enc, proj, crit
            if device.type == "cuda":
                torch.cuda.empty_cache()
        atomic_write_json(dst, rec)
        print(f"-> {dst}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
