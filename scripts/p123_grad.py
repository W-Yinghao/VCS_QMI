"""P123 runner (v6 V6-GRAD): per run and checkpoint, on fixed base images, the ACTUAL table (the checkpoint's own loss / pairing / scale) and
the COUNTERFACTUAL-READONLY table (same checkpoint, images, pairs and routing; only the scorer f replaced: affine a ∈ {1,2,3} × κ ∈ {0.25,0.5,0.75}
and the two v6 §7 curvatures λ = ±0.25 at (2, 0.5)); see src/vcs_diag/augdiag.py (P123 section).  Read-only; never a training result.

    python scripts/p123_grad.py --runs P107_AP3_views4_800ep_seed0,... [--checkpoints 100,400,800] [--grad-batches 4] \
        [--out-dir /home/infres/yinwang/CS_QMI/outputs/P123_grad_results] [--smoke] [--cpu] [--only-actual] [--repro-p115]

Base images: the first grad_batches × B FIT uids after the P115 seeded permutation of the run's manifest (CIFAR-10 runs share one manifest;
CIFAR-100 runs share theirs); augmentation views from the P115 forked seeded RNG (seed + 100 + batch); pair RNG = pair_seed + batch.  Batches 0–1
of a CIFAR-10 run are therefore the P115 job-1 batches.  Requested epochs not saved map to the nearest trained checkpoint (true epoch recorded,
never relabelled).  Labels are read only for the isolated semantic view of negatives (never filter or weight).  One JSON per run (resumable).
--repro-p115: compare the actual block of batch 0 at the P115-measured checkpoints with the stored P115 job-1 JSON (encoder P / Q norms).
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
from p115_diag import mean_tree  # noqa: E402  (imported, not edited)
from vcs_diag.augdiag import COUNTERFACTUAL_SCORERS, geometry, grad_batch_diagnostics, known_box_views, resolve_checkpoints  # noqa: E402
from vcs_ssl.config import load_resolved  # noqa: E402
from vcs_ssl.data.cifar import load_train_partition  # noqa: E402
from vcs_ssl.data.splits import load_manifest  # noqa: E402
from vcs_ssl.data.transforms import build_clean_transform, build_two_view_transform  # noqa: E402
from vcs_ssl.utils import atomic_write_json, utc_now  # noqa: E402

OUT = Path("/home/infres/yinwang/CS_QMI/outputs")
P115_DIR = OUT / "P115_diag_results"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True); ap.add_argument("--checkpoints", default="100,400,800")
    ap.add_argument("--grad-batches", type=int, default=4); ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--geo-images", type=int, default=2000); ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--pair-seed", type=int, default=99); ap.add_argument("--out-dir", default=str(OUT / "P123_grad_results"))
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--cpu", action="store_true"); ap.add_argument("--only-actual", action="store_true")
    ap.add_argument("--repro-p115", action="store_true", help="batch 0 only, actual only, compare with the stored P115 job-1 JSON")
    ap.add_argument("--scorers", default="", help="comma list of counterfactual scorer names (default: all 11)")
    a = ap.parse_args()
    device = torch.device("cpu") if a.cpu or not torch.cuda.is_available() else torch.device("cuda", 0)
    if device.type == "cpu":
        torch.set_num_threads(int(__import__("os").environ.get("SLURM_CPUS_PER_TASK", "8")))
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    req = [int(x) for x in a.checkpoints.split(",") if x]
    scorers = [s for s in COUNTERFACTUAL_SCORERS if not a.scorers or s["name"] in a.scorers.split(",")]
    cache: dict[str, object] = {}
    for run in [r for r in a.runs.split(",") if r]:
        dst = od / f"{run}.json"
        if dst.exists():
            print(f"[skip] {run}: {dst} exists", flush=True); continue
        rd = OUT / run
        st = json.load(open(rd / "status.json"))
        if st.get("status") != "COMPLETED":
            print(f"[skip] {run}: status {st.get('status')}", flush=True); continue
        cfg0 = load_resolved(rd / "config.resolved.yaml"); man = load_manifest(rd / "manifest.json")
        dname = cfg0["data"]["name"]
        if dname not in cache:
            cache[dname] = load_train_partition(dname, cfg0["data"]["root"])  # TRAIN partition only; official test never read
        data = cache[dname]
        V = int(cfg0["views"].get("count", 2)); B = int(a.batch or cfg0["train"]["batch_size_images"]); nb = a.grad_batches; ng = a.geo_images
        if a.smoke:
            B, nb, ng = 16, 1, 64
        if a.repro_p115:
            nb = 1
        fit = np.random.default_rng(a.seed).permutation(np.asarray(man["fit_uids"], dtype=np.int64))[: B * nb]
        sel = np.random.default_rng(a.seed + 1).permutation(np.asarray(man["selection_uids"], dtype=np.int64))[:ng]
        tf, clean = build_two_view_transform(cfg0["views"]), build_clean_transform(cfg0["views"])
        t_aug = time.time()
        batches = [known_box_views(data.data, fit[i * B:(i + 1) * B], tf, V, seed=a.seed + 100 + i) for i in range(nb)]
        labels = [torch.as_tensor(np.asarray(data.targets)[fit[i * B:(i + 1) * B]]) for i in range(nb)]
        req_run = req
        if a.repro_p115:
            p115 = json.load(open(P115_DIR / f"{run}.json")); req_run = [cm["epoch"] for cm in p115["checkpoint_map"] if cm.get("checkpoint") and not cm.get("duplicate")][-1:]
        rec = {"run": run, "created_utc": utc_now(), "device": str(device), "dataset": dname, "method": cfg0["run"]["method"], "views": V, "batch": B,
               "grad_batches": nb, "geo_images": ng, "diag_seed": a.seed, "pair_seed": a.pair_seed,
               "pair_scope": cfg0["pairing"].get("pair_scope", "cross_view_k"), "negative_detach": cfg0["pairing"].get("negative_detach"),
               "augmentation": {"random_resized_crop": cfg0["views"]["random_resized_crop"], "color_jitter": cfg0["views"]["color_jitter"]},
               "base_uids_sha256": hashlib.sha256(fit.tobytes()).hexdigest(), "geo_uids_sha256": hashlib.sha256(sel.tobytes()).hexdigest(),
               "augment_seconds": time.time() - t_aug, "smoke": a.smoke, "only_actual": a.only_actual or a.repro_p115,
               "counterfactual_scorers": [s["name"] for s in scorers] if not (a.only_actual or a.repro_p115) else [],
               "checkpoint_map": resolve_checkpoints(sorted(p.name for p in (rd / "checkpoints").iterdir()), req_run), "checkpoints": {}}
        for cm in rec["checkpoint_map"]:
            if cm.get("checkpoint") is None or cm.get("duplicate"):
                continue
            name = cm["checkpoint"]; t0 = time.time()
            if device.type == "cuda":
                torch.zeros(0, device=device)  # CUDA context first (cf. P85 / P115)
                try:
                    torch.cuda.reset_peak_memory_stats(device)
                except RuntimeError:
                    pass
            cfg, ck, enc, proj, crit = load(rd, name, device)
            r = {"true_epoch": int(name[6:-3]), "ckpt_epoch_field": ck.get("epoch"), "step": ck.get("step")}
            if not (a.repro_p115 or a.only_actual):
                r["geometry_eval"] = geometry(enc, proj, data.data, sel, clean, device, cfg["model"]["normalization"]["eps"])
            per = []
            for i, (views, boxes) in enumerate(batches):
                tb = time.time()
                per.append(grad_batch_diagnostics(cfg, enc, proj, crit, views, boxes, device, pair_seed=a.pair_seed + i, labels_img=labels[i],
                                                  scorers=scorers, only_actual=a.only_actual or a.repro_p115))
                per[-1]["seconds"] = time.time() - tb
            r["batches"] = per; r["mean_over_batches"] = mean_tree(per)
            if a.repro_p115:
                ref = p115["checkpoints"][name]["batches"][0]["grad"]["encoder"]; mine = per[0]["actual"]["grad_vector_norms"]["enc"]
                r["repro_p115"] = {"p115_P": ref["P"], "p123_P": mine["P"], "p115_Q": ref["Q"], "p123_Q": mine["Q"],
                                   "rel_err_P": abs(ref["P"] - mine["P"]) / ref["P"], "rel_err_Q": abs(ref["Q"] - mine["Q"]) / max(ref["Q"], 1e-30)}
                print(f"[{run}] repro P115 {name}: {r['repro_p115']}", flush=True)
            r["seconds"] = time.time() - t0
            r["peak_mem_mb"] = (torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else None
            rec["checkpoints"][name] = r
            act = r["mean_over_batches"]["actual"]
            print(f"[{run}] {name}: {r['seconds']:.0f}s  actual enc |gP| {act['grad_vector_norms']['enc']['P']:.4g} |gQ| {act['grad_vector_norms']['enc']['Q']:.4g}", flush=True)
            del enc, proj, crit
            if device.type == "cuda":
                torch.cuda.empty_cache()
        atomic_write_json(dst, rec)
        print(f"-> {dst}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
