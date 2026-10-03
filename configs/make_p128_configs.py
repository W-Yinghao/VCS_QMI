"""P128 — v6 §8.2 SimCLR tuning budget at the 4-view / 800-epoch protocol (counterpart of P127), seed 0.
Grid fixed by v6: temperature in {0.1, 0.2, 0.5} x two lr multipliers {1x, 2x} around the existing recipe (tau 0.2, AdamW lr 1e-3 = P41 C10 / P91 C100
seed 0, reused, not re-run).  Each config = that dataset's SimCLR 4v/800ep seed-0 config with only objective.simclr_temperature, optimizer.lr, run.stage.
    python configs/make_p128_configs.py [--write]
"""
import argparse, hashlib, json, time
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
STAGE = "P128_v6_simclr_grid"
PARENTS = {"c10": "cifar10_hpN_simclr_views4_800ep_seed0.yaml", "c100": "cifar100_hpS_simclr_views4_800ep_seed0.yaml"}
GRID = [(0.1, 1), (0.5, 1), (0.1, 2), (0.2, 2), (0.5, 2)]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    shas, lines = {}, []
    for ds in ("c100", "c10"):
        parent = PARENTS[ds]
        for tau, m in GRID:
            c = yaml.safe_load(open(ROOT / "configs" / parent))
            assert c["objective"]["loss"] == "nt_xent", c["objective"]["loss"]
            assert float(c["objective"]["simclr_temperature"]) == 0.2 and float(c["optimizer"]["lr"]) == 1e-3
            c["objective"]["simclr_temperature"] = float(tau); c["optimizer"]["lr"] = float(1e-3 * m); c["run"]["stage"] = STAGE
            tag = f"{ds}_t{tau:g}_lr{m}x"; run_id = f"P128_simclr_{tag}_seed0"
            name = ("cifar100" if ds == "c100" else "cifar10") + f"_hpG6_simclr_{tag}_seed0.yaml"
            txt = yaml.safe_dump(c, sort_keys=False); sha = hashlib.sha256(txt.encode()).hexdigest()
            shas[name] = {"sha256": sha, "run_id": run_id, "parent": parent, "temperature": tau, "lr": 1e-3 * m}
            print(run_id, sha[:16]); lines.append(f"sbatch --parsable --partition=RTX6000PRO,H100,L40S --job-name=p128_{tag} --export=ALL,CFG=configs/{name},RUN_ID={run_id},FINAL_CKPT=epoch_800.pt slurm/run_unit.sbatch")
            if a.write: (ROOT / "configs" / name).write_text(txt)
    if a.write:
        (ROOT / "configs" / "P128_SHA256.json").write_text(json.dumps({"stage": STAGE, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "configs": shas}, indent=1))
        (ROOT / "slurm" / "p128_lines.txt").write_text("# P128 v6 §8.2 SimCLR grid — normal QOS (never runfill); RTX6000PRO / H100 / L40S; node60 allowed\n" + "\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
