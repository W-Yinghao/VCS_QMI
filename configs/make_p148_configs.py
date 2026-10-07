"""P148 — exploration (our method): VCS A-P3 matrix weight decay 1e-4 -> {5e-4, 1e-5}, CIFAR-10 and CIFAR-100, seed 0.  Each config = the A-P3
seed-0 parent of that dataset with ONLY optimizer.matrix_weight_decay_encoder_projector changed (and run.stage).
    python configs/make_p148_configs.py [--write]   (run with slurm/common.sh exported: the policy load needs OUTPUT_ROOT)
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
STAGE = "P148_ap3_weight_decay"
PARENTS = {"c10": "cifar10_hpY_AP3_views4_800ep_seed0.yaml", "c100": "cifar100_hpY_AP3_c100_views4_800ep_seed0.yaml"}
WD = {"wd5e-4": 5e-4, "wd1e-5": 1e-5}


def build(ds, tag):
    c = yaml.safe_load(open(ROOT / "configs" / PARENTS[ds]))
    assert c["optimizer"]["matrix_weight_decay_encoder_projector"] == 1e-4
    c["optimizer"]["matrix_weight_decay_encoder_projector"] = WD[tag]; c["run"]["stage"] = STAGE
    return c


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "src")); from vcs_ssl.config import load_config
    shas, lines = {}, []
    for ds in PARENTS:
        for tag in WD:
            c = build(ds, tag); run_id = f"P148_AP3_{ds}_{tag}_seed0"
            name = ("cifar100" if ds == "c100" else "cifar10") + f"_hpWD_AP3_{ds}_{tag}_seed0.yaml"
            txt = yaml.safe_dump(c, sort_keys=False); sha = hashlib.sha256(txt.encode()).hexdigest()
            if a.write:
                (ROOT / "configs" / name).write_text(txt); load_config(ROOT / "configs" / name, require_dirs=False)
            shas[name] = {"sha256": sha, "run_id": run_id, "parent": PARENTS[ds]}; print(run_id, sha[:16])
            lines.append(f"sbatch --parsable --partition=RTX6000PRO,H100,L40S --exclude=node51,node52,node60 --job-name=p148_{ds}_{tag} --export=ALL,CFG=configs/{name},RUN_ID={run_id},FINAL_CKPT=epoch_800.pt slurm/run_unit.sbatch")
    if a.write:
        (ROOT / "configs" / "P148_SHA256.json").write_text(json.dumps({"stage": STAGE, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "configs": shas}, indent=1))
        (ROOT / "slurm" / "p148_lines.txt").write_text("# P148 A-P3 weight decay — normal QOS; RTX6000PRO / H100 / L40S; node51, node52, node60 excluded\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
