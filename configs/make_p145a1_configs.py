"""P145 addendum 1 — CIFAR-100 seeds 1-2 for VCS / JS / SimCLR (trigger: |Δ_VCS − Δ_JS| = 1.08 >= 1.00 at seed 0).  Each config = the P145 CIFAR-100
seed-0 config with ONLY run.seed / run.stage changed (the clean parents' seeds 1-2 differ from seed 0 only in run.seed: P107 A-P3, P120 JS-AP3,
P91 SimCLR).
    python configs/make_p145a1_configs.py [--write]
"""
import argparse, hashlib, json, time
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
STAGE = "P145A1_v7_stress_c100_seeds"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    shas, lines = {}, []
    for m in ("vcs", "js", "simclr"):
        base = f"cifar100_hpST10_{m}_c100_seed0.yaml"
        for s in (1, 2):
            c = yaml.safe_load(open(ROOT / "configs" / base)); assert c["run"]["seed"] == 0 and c["pairing"]["stress_epsilon"] == 0.10
            c["run"]["seed"] = s; c["run"]["stage"] = STAGE
            name = f"cifar100_hpST10_{m}_c100_seed{s}.yaml"; run_id = f"P145_STRESS10_{m}_c100_seed{s}"
            txt = yaml.safe_dump(c, sort_keys=False); sha = hashlib.sha256(txt.encode()).hexdigest()
            shas[name] = {"sha256": sha, "run_id": run_id, "base": base}; print(run_id, sha[:16])
            lines.append(f"sbatch --parsable --partition=RTX6000PRO,H100,L40S --exclude=node51,node52,node60 --job-name=p145a1_{m}_s{s} --export=ALL,CFG=configs/{name},RUN_ID={run_id},FINAL_CKPT=epoch_800.pt slurm/run_unit.sbatch")
            if a.write: (ROOT / "configs" / name).write_text(txt)
    if a.write:
        (ROOT / "configs" / "P145A1_SHA256.json").write_text(json.dumps({"stage": STAGE, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "configs": shas}, indent=1))
        (ROOT / "slurm" / "p145a1_lines.txt").write_text("# P145 addendum 1 — CIFAR-100 seeds 1-2, VCS / JS / SimCLR; normal QOS; node51 / 52 / 60 excluded\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
