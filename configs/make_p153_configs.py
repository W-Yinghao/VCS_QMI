"""P153 — exploration (our method): VCS A-P3 with weaker crops, RandomResizedCrop scale min 0.20 -> 0.35, CIFAR-10 and CIFAR-100, seed 0.  Each
config = the A-P3 seed-0 parent with ONLY views.random_resized_crop.scale changed (and run.stage).
    python configs/make_p153_configs.py [--write]   (with slurm/common.sh exported)
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
STAGE = "P153_ap3_weak_crop"
PARENTS = {"c10": "cifar10_hpY_AP3_views4_800ep_seed0.yaml", "c100": "cifar100_hpY_AP3_c100_views4_800ep_seed0.yaml"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "src")); from vcs_ssl.config import load_config
    shas, lines = {}, []
    for ds, base in PARENTS.items():
        c = yaml.safe_load(open(ROOT / "configs" / base)); assert c["views"]["random_resized_crop"]["scale"] == [0.2, 1.0]
        c["views"]["random_resized_crop"]["scale"] = [0.35, 1.0]; c["run"]["stage"] = STAGE
        name = ("cifar100" if ds == "c100" else "cifar10") + f"_hpWC_AP3_{ds}_crop035_seed0.yaml"; run_id = f"P153_AP3_{ds}_crop035_seed0"
        txt = yaml.safe_dump(c, sort_keys=False); sha = hashlib.sha256(txt.encode()).hexdigest()
        if a.write:
            (ROOT / "configs" / name).write_text(txt); load_config(ROOT / "configs" / name, require_dirs=False)
        shas[name] = {"sha256": sha, "run_id": run_id, "parent": base}; print(run_id, sha[:16])
        lines.append(f"sbatch --parsable --partition=RTX6000PRO,H100,L40S --exclude=node51,node52,node60 --job-name=p153_{ds}_crop035 --export=ALL,CFG=configs/{name},RUN_ID={run_id},FINAL_CKPT=epoch_800.pt slurm/run_unit.sbatch")
    if a.write:
        (ROOT / "configs" / "P153_SHA256.json").write_text(json.dumps({"stage": STAGE, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "configs": shas}, indent=1))
        (ROOT / "slurm" / "p153_lines.txt").write_text("# P153 A-P3 weak crop (scale min 0.35) — normal QOS; node51 / 52 / 60 excluded\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
