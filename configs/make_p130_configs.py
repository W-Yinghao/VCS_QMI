"""P130 — v6 §8.3 CIFAR-100 architecture check: CIFAR-stem ResNet-50 (h 2048) for A-P3 (VCS), JS-AP3 (matched JS) and SimCLR, seed 0.
Each config = that method's CIFAR-100 seed-0 ResNet-18 config (A-P3: P107 addendum 2; JS-AP3: P120; SimCLR: P91) with ONLY
model.backbone resnet18_cifar -> resnet50_cifar, model.h_dim 512 -> 2048 (the projector's input follows h_dim; hidden 512 / output 128 unchanged)
and run.stage.  Everything else (pairing, scorer, objective, optimizer, schedule, views, evaluation, seeds / RNG roles) is the parent's.
    python configs/make_p130_configs.py [--write]
"""
import argparse, hashlib, json, time
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
STAGE = "P130_v6_c100_r50"
CELLS = {"AP3": ("cifar100_hpY_AP3_c100_views4_800ep_seed0.yaml", "P130_AP3_c100_r50_views4_800ep_seed0", "cifar100_hpR50_AP3_views4_800ep_seed0.yaml"),
         "JS": ("cifar100_hpV6_JS_AP3_c100_views4_800ep_seed0.yaml", "P130_JS_AP3_c100_r50_views4_800ep_seed0", "cifar100_hpR50_JS_AP3_views4_800ep_seed0.yaml"),
         "simclr": ("cifar100_hpS_simclr_views4_800ep_seed0.yaml", "P130_simclr_c100_r50_views4_800ep_seed0", "cifar100_hpR50_simclr_views4_800ep_seed0.yaml")}
EXCLUDE = "node51,node60"            # current GPU policy (node60 throttled, node51 owner-excluded)
PARTITION = "RTX6000PRO,H100,L40S"   # SLURM fills partitions in this order (fastest first)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    shas, lines = {}, []
    for key, (parent, run_id, name) in CELLS.items():
        c = yaml.safe_load(open(ROOT / "configs" / parent)); m = c["model"]
        assert m["backbone"] == "resnet18_cifar" and int(m["h_dim"]) == 512 and c["run"]["seed"] == 0
        m["backbone"] = "resnet50_cifar"; m["h_dim"] = 2048; c["run"]["stage"] = STAGE
        txt = yaml.safe_dump(c, sort_keys=False); sha = hashlib.sha256(txt.encode()).hexdigest()
        shas[name] = {"sha256": sha, "run_id": run_id, "parent": parent}
        print(run_id, name, sha[:16])
        # two-link chain (afterany): link 2 resumes from last.pt only if link 1 stopped at the walltime, else it only re-evaluates
        lines.append(f'j1=$(sbatch --parsable --partition={PARTITION} --exclude={EXCLUDE} --job-name=p130_{key.lower()} '
                     f'--export=ALL,CFG=configs/{name},RUN_ID={run_id},FINAL_CKPT=epoch_800.pt slurm/run_unit.sbatch); '
                     f'j2=$(sbatch --parsable --partition={PARTITION} --exclude={EXCLUDE} --dependency=afterany:$j1 --job-name=p130_{key.lower()}_c2 '
                     f'--export=ALL,CFG=configs/{name},RUN_ID={run_id},FINAL_CKPT=epoch_800.pt slurm/run_unit.sbatch); echo "{run_id} $j1 $j2"')
        if a.write: (ROOT / "configs" / name).write_text(txt)
    if a.write:
        (ROOT / "configs" / "P130_SHA256.json").write_text(json.dumps({"stage": STAGE, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "configs": shas}, indent=1))
        (ROOT / "slurm" / "p130_lines.txt").write_text("# P130 CIFAR-100 ResNet-50 (seed 0): one 2-link chain per method; normal QOS (never runfill)\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
