"""V9-N2 wave-2 smoke (GPU, <= 30 min; run by `slurm/v9/n2_official.sbatch smoke`): the PATCHED official NCR+A and L2RM+A with the anchor term ON
on a real-data subset of Flickr30K precomp (first --n-train training images, first --n-eval test images, the official dev split), 1 warm-up + 1 main epoch each,
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
    """train / test: the first n images (5 captions each).  dev: the OFFICIAL files, linked — NCR's and L2RM's PrecompDataset hard-code the dev
    length to 1000 images x 5 captions (smoke 1042220 overran a 100-image dev subset at the first validation), and the real runs validate on
    exactly these files."""
    d = dst / "f30k_precomp"
    d.mkdir(parents=True, exist_ok=True)
    for f in ("dev_ims.npy", "dev_caps.txt"):
        if not (d / f).exists():
            (d / f).symlink_to(SRC / f)
    for split, n in (("train", n_train), ("test", n_eval)):
        ims = np.load(SRC / f"{split}_ims.npy", mmap_mode="r")
        n_caps = sum(1 for _ in open(SRC / f"{split}_caps.txt"))
        rows = 5 * n if ims.shape[0] == n_caps else n            # f30k dev / test store one image row per caption (rows 5i..5i+4 identical); train unique rows
        np.save(d / f"{split}_ims.npy", np.ascontiguousarray(ims[:rows]))
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


def dev_trajectory(train_log: str) -> list:
    """every validation the official trainer printed: 'Image to text: r1, r5, r10, medr, meanr' then 'Text to image: ...' -> recalls + rSum."""
    import re
    i2t = re.findall(r"Image to text:\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)", train_log)
    t2i = re.findall(r"Text to image:\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)", train_log)
    out = []
    for a, b in zip(i2t, t2i):
        a, b = [float(x) for x in a], [float(x) for x in b]
        out.append({"i2t_R1": a[0], "t2i_R1": b[0], "rsum": round(sum(a) + sum(b), 1)})
    return out


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
        # main session 2026-10-17: a dev-rSum drop after the first co-training / main epoch (as in timing job 1042195: dev R@1 48.5 / 43.2 ->
        # 13.0 / 0.8) is a FAILURE, not a pass.  The last validation before the main phase is the warm-up reference (NCR: after the warm-up
        # epochs; L2RM: after its warm-up epochs); every later validation must keep >= 50 % of its rSum and t2i R@1 above chance x 4.
        traj = dev_trajectory(tl.split("-------- Testing --------")[0]); r["dev_trajectory"] = traj   # the trainers' own end-of-training test read is not a dev validation
        n_warm = 1                                                  # smoke schedule: 1 warm-up epoch -> its validation is entry 0 (NCR also validates once after warm-up)
        ref = traj[n_warm - 1]["rsum"] if len(traj) >= n_warm else None
        later = traj[n_warm:]
        chance_r1 = 100.0 / 200                                     # official dev validation: first 1,000 captions = 200 images
        r["dev_drop"] = (ref is None or not later or any(x["rsum"] < 0.5 * ref or x["t2i_R1"] < 4 * chance_r1 for x in later))
        r["collect_dev_rc"] = run([py, drv, "collect-dev", "--baseline", b, "--unit", str(unit)], log)
        r["collect_rc"] = run([py, drv, "collect", "--baseline", b, "--unit", str(unit), "--dataset", "f30k", "--data-root", str(out / "data")], log)
        r["done"] = json.loads((unit / "done.json").read_text()) if (unit / "done.json").exists() else None
        r["test_primary"] = json.loads((unit / "result.json").read_text())["test"]["primary"] if (unit / "result.json").exists() else None
        r["ok"] = (r["export_rc"] == 0 and r["launch_rc"] == 0 and r["collect_dev_rc"] == 0 and r["collect_rc"] == 0 and r["observed_index_loaded"]
                   and bool(r["anchor_term_lines"]) and not r["traceback"] and r["done"] is not None and r["test_primary"] is not None and not r["dev_drop"])
        report["baselines"][b] = r
    report["ok"] = all(r["ok"] for r in report["baselines"].values()); report["seconds"] = round(time.time() - t0, 1)
    (out / "smoke_report.json").write_text(json.dumps(report, indent=1)); print(json.dumps(report, indent=1))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
