"""P130 addendum 1 — ResNet-50 CIFAR-100 seeds 1-2 for A-P3 / JS-AP3 / SimCLR (seed rule fixed at the P130 freeze: |A-P3 − SimCLR| = 6.12 >= 0.5 at
seed 0).  Each config = the P130 seed-0 config with ONLY run.seed / run.stage changed; each unit is a 2-link chain (afterany; link 2 resumes from
last.pt), as for seed 0.
    python configs/make_p130a1_configs.py [--write]
"""
import argparse, hashlib, json, time
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
STAGE = "P130A1_v6_c100_r50_seeds"
BASES = {"ap3": ("cifar100_hpR50_AP3_views4_800ep", "P130_AP3_c100_r50_views4_800ep"), "js": ("cifar100_hpR50_JS_AP3_views4_800ep", "P130_JS_AP3_c100_r50_views4_800ep"),
         "simclr": ("cifar100_hpR50_simclr_views4_800ep", "P130_simclr_c100_r50_views4_800ep")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    shas, lines = {}, []
    for m, (cb, rb) in BASES.items():
        for s in (1, 2):
            c = yaml.safe_load(open(ROOT / "configs" / f"{cb}_seed0.yaml")); assert c["run"]["seed"] == 0
            c["run"]["seed"] = s; c["run"]["stage"] = STAGE
            name = f"{cb}_seed{s}.yaml"; rid = f"{rb}_seed{s}"
            txt = yaml.safe_dump(c, sort_keys=False); sha = hashlib.sha256(txt.encode()).hexdigest()
            shas[name] = {"sha256": sha, "run_id": rid, "base": f"{cb}_seed0.yaml"}; print(rid, sha[:16])
            P = "--partition=RTX6000PRO,H100,L40S --exclude=node51,node52,node60"
            lines.append(f"j1=$(sbatch --parsable {P} --job-name=p130a1_{m}_s{s} --export=ALL,CFG=configs/{name},RUN_ID={rid},FINAL_CKPT=epoch_800.pt slurm/run_unit.sbatch); "
                         f"j2=$(sbatch --parsable {P} --dependency=afterany:$j1 --job-name=p130a1_{m}_s{s}_c2 --export=ALL,CFG=configs/{name},RUN_ID={rid},FINAL_CKPT=epoch_800.pt slurm/run_unit.sbatch); echo \"{rid} $j1 $j2\"")
            if a.write: (ROOT / "configs" / name).write_text(txt)
    if a.write:
        (ROOT / "configs" / "P130A1_SHA256.json").write_text(json.dumps({"stage": STAGE, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "configs": shas}, indent=1))
        (ROOT / "slurm" / "p130a1_lines.txt").write_text("# P130 addendum 1 — ResNet-50 CIFAR-100 seeds 1-2 x 3 methods, 2-link chains; normal QOS; node51 / 52 / 60 excluded\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
