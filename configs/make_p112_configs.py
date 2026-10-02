"""P112 — fixed-scale sensitivity (package v4 §A3 item 1; owner 2026-10-02 "再排十个任务到队列中"): ten seed-0 800-epoch units.
Each config is the corresponding G2 config with ONLY the fixed (a, b) changed; kappa = -b/a is the zero-score cosine threshold (G2: a 2, kappa 0.5).
    CIFAR-10 standard : (a, kappa) = (1.5, 0.5), (3, 0.5), (2, 0.25), (2, 0.75)   parent configs/cifar10_hpX_G2_views4_800ep_seed0.yaml
    CIFAR-10 strong   : (2, 0.25), (2, 0.75), (3, 0.5)                           parent configs/cifar10_hpZ_G2_augstrong_views4_800ep_seed0.yaml
    CIFAR-100         : (2, 0.25), (2, 0.75), (3, 0.5)                           parent configs/cifar100_hpZ_G2_c100_views4_800ep_seed0.yaml
    python configs/make_p112_configs.py [--write]
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
STAGE = "P112_fixed_scale_sensitivity"
PARENTS = {"std": "cifar10_hpX_G2_views4_800ep_seed0.yaml", "strong": "cifar10_hpZ_G2_augstrong_views4_800ep_seed0.yaml",
           "c100": "cifar100_hpZ_G2_c100_views4_800ep_seed0.yaml"}
CELLS = [("std", 1.5, 0.5), ("std", 3.0, 0.5), ("std", 2.0, 0.25), ("std", 2.0, 0.75),
         ("strong", 2.0, 0.25), ("strong", 2.0, 0.75), ("strong", 3.0, 0.5),
         ("c100", 2.0, 0.25), ("c100", 2.0, 0.75), ("c100", 3.0, 0.5)]

def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    shas, lines = {}, []
    for setting, sc, kap in CELLS:
        c = yaml.safe_load(open(ROOT / "configs" / PARENTS[setting]))
        crit = c["model"]["critic"]; assert crit["affine_mode"] == "fixed" and crit["cosine_scale_init"] == 2.0 and crit["cosine_bias_init"] == -1.0
        crit["cosine_scale_init"] = float(sc); crit["cosine_bias_init"] = float(-sc * kap); c["run"]["stage"] = STAGE
        tag = f"{setting}_a{sc:g}_k{kap:g}"; run_id = f"P112_{tag}_seed0"
        name = ("cifar100" if setting == "c100" else "cifar10") + f"_hpF_{tag}_seed0.yaml"
        txt = yaml.safe_dump(c, sort_keys=False); sha = hashlib.sha256(txt.encode()).hexdigest()
        shas[name] = {"sha256": sha, "run_id": run_id, "parent": PARENTS[setting], "a": sc, "b": -sc * kap, "kappa": kap}
        print(f"{run_id:34s} {sha[:16]} a={sc} b={-sc*kap} parent={PARENTS[setting]}")
        lines.append(f"sbatch --parsable --partition=RTX6000PRO,H100 --exclude=node60 --job-name=p112_{tag} "
                     f"--export=ALL,CFG=configs/{name},RUN_ID={run_id},FINAL_CKPT=epoch_800.pt slurm/run_unit.sbatch")
        if a.write: (ROOT / "configs" / name).write_text(txt)
    if a.write:
        (ROOT / "configs" / "P112_SHA256.json").write_text(json.dumps({"stage": STAGE, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "configs": shas}, indent=1))
        (ROOT / "slurm" / "p112_lines.txt").write_text("# P112 fixed-scale sensitivity — normal QOS (never runfill)\n" + "\n".join(lines) + "\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
