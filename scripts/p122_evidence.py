"""P122 — v6 V6-EVIDENCE runner: hierarchical s -> Z -> H measurement critics on frozen encoders under COMMON measurement augmentation laws.

    python scripts/p122_evidence.py --dataset cifar10 --runs P35_vcs_a5_views4_800ep_seed0,P107_AP3_views4_800ep_seed0 --laws standard,strong \
        --out outputs/P122_evidence [--smoke]

Per dataset: one pool of disjoint base images from the manifest FIT uids (FIT 8192 / TUNE 2048 / EVAL 8192; half anchors, half partners), one set of
augmented images per measurement law (fixed per-image seeds, cached, shared by every encoder), float32 raw h and z per encoder (eval mode, frozen
BN, cached), then for each loss (vcs, js) the level candidates of `vcs_measure.evidence` selected on TUNE and read on EVAL.  One JSON per
(encoder, law).  The official test partition is never opened (training-partition loaders only).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import vcs_measure.evidence as ev  # noqa: E402
from vcs_measure.common import load_run  # noqa: E402
from vcs_ssl.data.cifar import load_train_partition  # noqa: E402
from vcs_ssl.data.splits import load_manifest  # noqa: E402
from vcs_ssl.utils import atomic_write_json, utc_now  # noqa: E402

POOL_SEED = 20261003
AUG_SEED = 122
LAW_SOURCES = {  # the stochastic block of these frozen configs IS the measurement law (common to every encoder)
    ("cifar10", "standard"): "configs/cifar10_hpK_a5_views4_800ep_vcs_seed0.yaml",
    ("cifar10", "strong"): "configs/cifar10_hpO_a5_views4_800ep_augstrong_vcs_seed0.yaml",
    ("cifar100", "standard"): "configs/cifar100_hpS_vcs_a5_views4_800ep_seed0.yaml",
}
MANIFESTS = {"cifar10": "/home/infres/yinwang/CS_QMI/manifests/cifar10_dev45k_val5k.json",
             "cifar100": "/home/infres/yinwang/CS_QMI/manifests/cifar100_dev45k_val5k.json"}
ROOTS = {"cifar10": "/home/infres/yinwang/CS_QMI/data/cifar10", "cifar100": "/home/infres/yinwang/CS_QMI/data/cifar100"}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def key(*parts) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:20]


def peak_mem(device) -> dict:
    if device.type != "cuda":
        return {}
    return {"peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2 ** 20, "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 2 ** 20}


@torch.no_grad()
def rows_of(cands, Pe):
    out = []
    for c in cands:
        fp, fq = c["fn"](Pe)
        out.append({"name": getattr(c["fn"], "name", type(c["fn"]).__name__), "params": c.get("params") or getattr(c["fn"], "params", {}),
                    "tune_risk": c["tune_risk"], "eval": ev.readouts(fp, fq), "seconds": c.get("seconds", 0.0)})
    return out


def run_one(R, feats: dict, loss: str, device, diag_n: int) -> dict:
    Pf, Pt, Pe = feats["fit"], feats["tune"], feats["eval"]
    out = {"loss": loss, "levels": {}}
    crit = R["critic"] if R["cfg"]["run"]["method"] == "vcs_qmi" else None
    train = ev.TrainScorer(crit.to(device)) if crit is not None and hasattr(crit, "logits") else None
    # s level (train scorer is a candidate = the parent)
    t0 = time.time()
    cands = ev.fit_affine(Pf, Pt, loss) + [ev.fit_bins(Pf, 16), ev.fit_bins(Pf, 32)] + ev.fit_mlp1d(Pf, Pt, loss, device)
    if train is not None:
        cands = [{"fn": train, "seconds": 0.0, "params": {"parent": "train scorer"}}] + cands
    s_best, s_rows = ev.select(cands, Pt, loss)
    out["levels"]["s"] = {"candidates": rows_of(s_rows, Pe), "selected": getattr(s_best["fn"], "name", "train"), "seconds": time.time() - t0}
    # Z level on the selected s branch
    t0 = time.time()
    z_cands = ev.fit_nested(s_best["fn"], Pf, Pt, loss, "z", device)
    z_best, z_rows = ev.select(z_cands, Pt, loss)
    out["levels"]["Z"] = {"candidates": rows_of(z_rows, Pe), "selected": z_best["params"], "seconds": time.time() - t0}
    # H level on the selected Z model (h standardised with FIT statistics)
    t0 = time.time()
    hall = torch.cat([Pf.h1, Pf.h2, Pf.hb]); mu, sd = hall.mean(0, keepdim=True), hall.std(0, keepdim=True).clamp_min(1e-6)
    h_cands = ev.fit_nested(z_best["fn"], Pf, Pt, loss, "h", device, mu=mu, sd=sd)
    h_best, h_rows = ev.select(h_cands, Pt, loss)
    out["levels"]["H"] = {"candidates": rows_of(h_rows, Pe), "selected": h_best["params"], "seconds": time.time() - t0}
    # EVAL readouts and paired increments (unit = pair index)
    J = {}
    for lv, fn in (("train", train), ("s", s_best["fn"]), ("Z", z_best["fn"]), ("H", h_best["fn"])):
        if fn is None:
            continue
        j, fp, fq = ev.unit_j(fn, Pe); J[lv] = j
        out.setdefault("eval", {})[lv] = ev.readouts(fp, fq)
        if diag_n:
            out.setdefault("density_diag", {})[lv] = ev.density_diag(fn, Pe, diag_n, diag_n)
    inc = {}
    if "train" in J:
        inc["s_minus_train"] = ev.boot_increment(J["s"], J["train"])
    inc["Z_minus_s"] = ev.boot_increment(J["Z"], J["s"]); inc["H_minus_Z"] = ev.boot_increment(J["H"], J["Z"])
    inc["H_minus_s"] = ev.boot_increment(J["H"], J["s"])
    out["increments"] = inc
    out["wording"] = "additional score recovered in these function classes and budgets (not true information terms)"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=("cifar10", "cifar100")); ap.add_argument("--runs", required=True)
    ap.add_argument("--laws", default="standard"); ap.add_argument("--out", required=True)
    ap.add_argument("--fit", type=int, default=8192); ap.add_argument("--tune", type=int, default=2048); ap.add_argument("--eval", type=int, default=8192)
    ap.add_argument("--cache", default="/home/infres/yinwang/CS_QMI/outputs/P122_cache"); ap.add_argument("--diag-n", type=int, default=512)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.fit, a.tune, a.eval, a.diag_n = 512, 256, 512, 64
        ev.STEPS, ev.LRS, ev.INITS = 60, (5e-4,), (0,)
    device = torch.device("cpu") if a.cpu or not torch.cuda.is_available() else torch.device("cuda", 0)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    out_dir, cache = Path(a.out), Path(a.cache); out_dir.mkdir(parents=True, exist_ok=True); cache.mkdir(parents=True, exist_ok=True)
    data = load_train_partition(a.dataset, ROOTS[a.dataset])            # official TRAIN partition only
    man = load_manifest(MANIFESTS[a.dataset])
    sizes = {"fit": a.fit, "tune": a.tune, "eval": a.eval}
    pools = ev.make_pools(np.asarray(man["fit_uids"], dtype=np.int64), sizes, POOL_SEED); ph = ev.pools_hash(pools)
    laws = {}
    for law in a.laws.split(","):
        views = yaml.safe_load(open(REPO / LAW_SOURCES[(a.dataset, law)]))["views"]
        ak = key(a.dataset, ev.law_signature(views), ph, AUG_SEED, ev.CODE_VERSION); ap_ = cache / f"aug_{a.dataset}_{law}_{ak}.pt"
        t0 = time.time()
        if ap_.is_file():
            imgs = torch.load(ap_)
        else:
            imgs = ev.augment_pools(data.data, pools, views, AUG_SEED); torch.save(imgs, ap_)
        laws[law] = {"views": views, "imgs": imgs, "aug_key": ak, "seconds": time.time() - t0}
        print(f"[{a.dataset}/{law}] augmented images {ak} ({time.time() - t0:.0f}s)", flush=True)
    for run in a.runs.split(","):
        R = load_run(run, device=device)
        assert R["cfg"]["data"]["name"] == a.dataset, f"{run} is a {R['cfg']['data']['name']} run"
        ck_sha = sha256_file(Path("/home/infres/yinwang/CS_QMI/outputs") / run / "checkpoints" / "epoch_800.pt")
        vw = R["cfg"]["views"]; mean, std = vw["normalize_mean"], vw["normalize_std"]
        for law, L in laws.items():
            outp = out_dir / f"{run}__{law}.json"
            if outp.is_file() and not a.smoke:
                print(f"[skip] {outp.name}"); continue
            t0 = time.time()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            fk = key(a.dataset, ph, L["aug_key"], ck_sha, "float32", mean, std, ev.CODE_VERSION); fp_ = cache / f"feat_{run}_{law}_{fk}.pt"
            if fp_.is_file():
                F_ = torch.load(fp_)
            else:
                F_ = {k: {v: ev.encode(R, L["imgs"][k][v], mean, std, device) for v in ("A1", "A2", "B")} for k in sizes}
                torch.save(F_, fp_)
            t_feat = time.time() - t0
            feats = {k: ev.Pairs(F_[k]["A1"]["z"], F_[k]["A2"]["z"], F_[k]["B"]["z"], F_[k]["A1"]["h"], F_[k]["A2"]["h"], F_[k]["B"]["h"]).to(device) for k in sizes}
            rec = {"protocol": "P122_v6_evidence", "utc": utc_now(), "run": run, "dataset": a.dataset, "measurement_law": law,
                   "train_augmentation": {"crop_scale": vw["random_resized_crop"]["scale"], "color_jitter": vw["color_jitter"]},
                   "measurement_augmentation": {"source": LAW_SOURCES[(a.dataset, law)], "signature": ev.law_signature(L["views"])},
                   "pools": {"sizes": sizes, "hash": ph, "seed": POOL_SEED, "units": {k: int(len(pools[k]["anchors"])) for k in sizes}},
                   "aug_seed": AUG_SEED, "cache_keys": {"aug": L["aug_key"], "feat": fk}, "checkpoint_sha256": ck_sha, "dtype": "float32",
                   "method": R["cfg"]["run"]["method"], "device": str(device), "budget": {"steps": ev.STEPS, "lrs": list(ev.LRS), "inits": list(ev.INITS)},
                   "smoke": a.smoke, "losses": {}}
            for loss in ("vcs", "js"):
                tl = time.time(); rec["losses"][loss] = run_one(R, feats, loss, device, a.diag_n); rec["losses"][loss]["seconds"] = time.time() - tl
                e = rec["losses"][loss]["eval"]; inc = rec["losses"][loss]["increments"]
                print(f"[{run}/{law}/{loss}] J " + " ".join(f"{k}={v['J']:.4f}" for k, v in e.items()) + " | " +
                      " ".join(f"{k}={v['mean']:+.4f}[{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]" for k, v in inc.items()) + f" ({time.time() - tl:.0f}s)", flush=True)
            rec["seconds"] = {"features": t_feat, "total": time.time() - t0}; rec["memory"] = peak_mem(device)
            atomic_write_json(outp, rec)
        del R
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
