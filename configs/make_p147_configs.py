"""P147 — exploration (our method): VCS A-P3 at batch 512 with the learning rate scaled linearly with the batch (1e-3 -> 2e-3), CIFAR-10 and
CIFAR-100, seed 0.  Each config = the P136 b512 config of that dataset (batch 512, lr 1e-3, scale_lr_with_batch false — P136's disclosed
confound) with ONLY optimizer.lr = 0.002 (and run.stage).
    python configs/make_p147_configs.py [--write]
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
STAGE = "P147_ap3_b512_lr2x"
PARENTS = {"c10": "cifar10_hpX_AP3_c10_b512_seed0.yaml", "c100": "cifar100_hpX_AP3_c100_b512_seed0.yaml"}


def build(ds):
    c = yaml.safe_load(open(ROOT / "configs" / PARENTS[ds]))
    assert c["train"]["batch_size_images"] == 512 and c["optimizer"]["lr"] == 0.001 and c["schedule"]["scale_lr_with_batch"] is False
    c["optimizer"]["lr"] = 0.002; c["run"]["stage"] = STAGE
    return c


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "src")); from vcs_ssl.config import load_config
    shas, lines = {}, []
    for ds in PARENTS:
        c = build(ds); run_id = f"P147_AP3_{ds}_b512_lr2x_seed0"
        name = ("cifar100" if ds == "c100" else "cifar10") + f"_hpBL_AP3_{ds}_b512_lr2x_seed0.yaml"
        txt = yaml.safe_dump(c, sort_keys=False); sha = hashlib.sha256(txt.encode()).hexdigest()
        if a.write:
            (ROOT / "configs" / name).write_text(txt); load_config(ROOT / "configs" / name, require_dirs=False)  # policy acceptance
        shas[name] = {"sha256": sha, "run_id": run_id, "parent": PARENTS[ds]}; print(run_id, sha[:16])
        lines.append(f"sbatch --parsable --partition=RTX6000PRO,H100,L40S --exclude=node51,node52,node60 --job-name=p147_{ds} --export=ALL,CFG=configs/{name},RUN_ID={run_id},FINAL_CKPT=epoch_800.pt slurm/run_unit.sbatch")
    if a.write:
        (ROOT / "configs" / "P147_SHA256.json").write_text(json.dumps({"stage": STAGE, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "configs": shas}, indent=1))
        (ROOT / "slurm" / "p147_lines.txt").write_text("# P147 A-P3 b512 lr 2e-3 — normal QOS; RTX6000PRO / H100 / L40S; node51, node52, node60 excluded\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
