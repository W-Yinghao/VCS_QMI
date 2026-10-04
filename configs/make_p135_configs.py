"""P135 — optimizer learning-rate check for our method family on the A-P3 structure (VCS A-P3 and matched JS-AP3), CIFAR-10 and CIFAR-100, seed 0.
The AdamW lr (1e-3) was never searched for either loss; owner 2026-10-04: our method is new, its optimum is unknown — tune our side, not SimCLR.
Each config = that dataset's A-P3 / JS-AP3 seed-0 config with only optimizer.lr (and run.stage) changed: lr in {5e-4, 2e-3}; 1e-3 = existing runs.
    python configs/make_p135_configs.py [--write]
"""
import argparse, hashlib, json, time
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
STAGE = "P135_lr_check_ap3_family"
PARENTS = {("vcs", "c10"): "cifar10_hpY_AP3_views4_800ep_seed0.yaml", ("vcs", "c100"): "cifar100_hpY_AP3_c100_views4_800ep_seed0.yaml",
           ("js", "c10"): "cifar10_hpJS_AP3_views4_800ep_seed0.yaml", ("js", "c100"): "cifar100_hpV6_JS_AP3_c100_views4_800ep_seed0.yaml"}
LRS = {"lr0.5x": 5e-4, "lr2x": 2e-3}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    shas, lines = {}, []
    for (loss, ds), parent in PARENTS.items():
        for tag_lr, lr in LRS.items():
            c = yaml.safe_load(open(ROOT / "configs" / parent)); assert float(c["optimizer"]["lr"]) == 1e-3
            c["optimizer"]["lr"] = float(lr); c["run"]["stage"] = STAGE
            tag = f"{loss}_{ds}_{tag_lr}"; run_id = f"P135_{tag}_seed0"
            name = ("cifar100" if ds == "c100" else "cifar10") + f"_hpLR_{tag}_seed0.yaml"
            txt = yaml.safe_dump(c, sort_keys=False); sha = hashlib.sha256(txt.encode()).hexdigest()
            shas[name] = {"sha256": sha, "run_id": run_id, "parent": parent, "lr": lr}; print(run_id, sha[:16])
            lines.append(f"sbatch --parsable --partition=RTX6000PRO,H100,L40S --exclude=node51,node60 --job-name=p135_{tag} --export=ALL,CFG=configs/{name},RUN_ID={run_id},FINAL_CKPT=epoch_800.pt slurm/run_unit.sbatch")
            if a.write: (ROOT / "configs" / name).write_text(txt)
    if a.write:
        (ROOT / "configs" / "P135_SHA256.json").write_text(json.dumps({"stage": STAGE, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "configs": shas}, indent=1))
        (ROOT / "slurm" / "p135_lines.txt").write_text("# P135 lr check (VCS A-P3 / JS-AP3, C10 / C100) — normal QOS; RTX6000PRO / H100 / L40S; node51 + node60 excluded\n" + "\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
