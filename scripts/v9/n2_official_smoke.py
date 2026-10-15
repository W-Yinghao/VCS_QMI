"""V9-N2 wave-2 smoke (GPU, <= 30 min; run by `slurm/v9/n2_official.sbatch smoke`): the PATCHED official NCR+A and L2RM+A with the anchor term ON
on a real-data subset of Flickr30K precomp (first --n-train training images, first --n-eval dev / test images), 1 warm-up + 1 main epoch each,
through the same driver calls as a wave-2 unit (export -> launch -> collect-dev -> collect).  Checks that the V9 observed index is loaded, that the
anchor block runs (its `[V9] anchor-supervised term` line), that both trainers exit 0, and that the dev best rSum and the official test recalls
are collected.  A code-path check only — never evidence (subset, 1+1 epochs)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from v9.n2 import noise as NZ, roles as RL  # noqa: E402

SRC = Path("/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data/f30k_precomp")


def subset(dst: Path, n_train: int, n_eval: int) -> Path:
    d = dst / "f30k_precomp"
    d.mkdir(parents=True, exist_ok=True)
    for split, n in (("train", n_train), ("dev", n_eval), ("test", n_eval)):
        ims = np.load(SRC / f"{split}_ims.npy", mmap_mode="r")
        np.save(d / f"{split}_ims.npy", np.ascontiguousarray(ims[:n]))
        with open(SRC / f"{split}_caps.txt") as f:
            lines = [next(f) for _ in range(5 * n)]
        (d / f"{split}_caps.txt").write_text("".join(lines))
    return d


def shared_stage(base: Path, n_train: int, rho: float, seed: int) -> None:
    base.mkdir(parents=True, exist_ok=True)
    roles = RL.assign_roles([str(i) for i in range(n_train)], seed, 0.1, min_sizes={k: 1 for k in RL.ANCHOR_ROLES})
    RL.write_roles(roles, base / "roles.json")
    roles = json.loads((base / "roles.json").read_text())
    pairs = NZ.make_pair_table({i: i // 5 for i in range(5 * n_train)}, RL.role_of_image(roles))
    observed, labels, manifest = NZ.inject_mismatch(pairs, rho, seed)
    NZ.write_noise(observed, labels, manifest, base, "pairs")


def run(cmd: list, log: Path) -> int:
    with open(log, "a") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n"); fh.flush()
        return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT).returncode


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--n-train", type=int, default=1000); ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--rho", type=float, default=0.4); ap.add_argument("--seed", type=int, default=901); ap.add_argument("--baselines", nargs="+", default=["ncr", "l2rm"])
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); py = sys.executable; drv = str(ROOT / "scripts/v9/n2_official.py")
    t0 = time.time(); data = subset(out / "data", a.n_train, a.n_eval); shared = out / "shared"; shared_stage(shared, a.n_train, a.rho, a.seed)
    report = {"n_train_images": a.n_train, "n_eval_images": a.n_eval, "rho": a.rho, "seed": a.seed, "subset_seconds": round(time.time() - t0, 1), "baselines": {}}
    for b in a.baselines:
        unit = out / f"unit_{b}"; unit.mkdir(exist_ok=True); log = out / f"smoke_{b}.log"; t1 = time.time(); r = {}
        r["export_rc"] = run([py, drv, "export", "--shared", str(shared), "--data-root", str(data), "--out", str(unit)], log)
        r["launch_rc"] = run([py, drv, "launch", "--baseline", b, "--unit", str(unit), "--dataset", "f30k", "--seed", str(a.seed), "--rho", str(a.rho),
                              "--data-root", str(out / "data"), "--num-epochs", "1", "--warmup-epoch", "1", "--lr-update", "1"], log)
        r["train_seconds"] = round(time.time() - t1, 1)
        tl = (unit / "official_train.log").read_text(errors="replace") if (unit / "official_train.log").exists() else ""
        r["observed_index_loaded"] = "load V9 observed index" in tl
        r["anchor_term_lines"] = [ln.strip() for ln in tl.splitlines() if "[V9] anchor-supervised term" in ln][:4]
        r["traceback"] = "Traceback" in tl
        r["collect_dev_rc"] = run([py, drv, "collect-dev", "--baseline", b, "--unit", str(unit)], log)
        r["collect_rc"] = run([py, drv, "collect", "--baseline", b, "--unit", str(unit), "--dataset", "f30k", "--data-root", str(out / "data")], log)
        r["done"] = json.loads((unit / "done.json").read_text()) if (unit / "done.json").exists() else None
        r["test_primary"] = json.loads((unit / "result.json").read_text())["test"]["primary"] if (unit / "result.json").exists() else None
        r["ok"] = (r["export_rc"] == 0 and r["launch_rc"] == 0 and r["collect_dev_rc"] == 0 and r["collect_rc"] == 0 and r["observed_index_loaded"]
                   and bool(r["anchor_term_lines"]) and not r["traceback"] and r["done"] is not None and r["test_primary"] is not None)
        report["baselines"][b] = r
    report["ok"] = all(r["ok"] for r in report["baselines"].values()); report["seconds"] = round(time.time() - t0, 1)
    (out / "smoke_report.json").write_text(json.dumps(report, indent=1)); print(json.dumps(report, indent=1))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
