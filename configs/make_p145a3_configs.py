"""P145 addendum 3 — CIFAR-100 seeds 1-2 for VCS / JS / SimCLR at STRESS eps = 0.20 (triggered by addendum 2: seed-0 |Δ_VCS − Δ_JS| = 1.12 >= 1.00;
owner confirmed running it 2026-10-09).  Each config = the eps-0.20 seed-0 config with ONLY run.seed and run.stage changed; checked to equal the
eps-0.10 seed-s config except pairing.stress_epsilon and run.stage.
    python configs/make_p145a3_configs.py [--write]   (with slurm/common.sh exported: the policy load needs OUTPUT_ROOT)
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
STAGE = "P145A3_v7_stress_eps020_c100_seeds"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "src")); from vcs_ssl.config import load_config
    shas, lines = {}, []
    for m in ("vcs", "js", "simclr"):
        for s in (1, 2):
            base = f"cifar100_hpST20_{m}_c100_seed0.yaml"
            c = yaml.safe_load(open(ROOT / "configs" / base)); assert c["run"]["seed"] == 0 and c["pairing"]["stress_epsilon"] == 0.20
            c["run"]["seed"] = s; c["run"]["stage"] = STAGE
            ref = yaml.safe_load(open(ROOT / "configs" / f"cifar100_hpST10_{m}_c100_seed{s}.yaml"))
            ref["pairing"]["stress_epsilon"] = 0.20; ref["run"]["stage"] = STAGE; assert ref == c, f"{m} seed {s}: differs from the eps-0.10 seed config beyond eps / stage"
            name = f"cifar100_hpST20_{m}_c100_seed{s}.yaml"; run_id = f"P145_STRESS20_{m}_c100_seed{s}"
            txt = yaml.safe_dump(c, sort_keys=False); sha = hashlib.sha256(txt.encode()).hexdigest()
            if a.write:
                (ROOT / "configs" / name).write_text(txt); load_config(ROOT / "configs" / name, require_dirs=False)
            shas[name] = {"sha256": sha, "run_id": run_id, "base": base}; print(run_id, sha[:16])
            lines.append(f"sbatch --parsable --nice=1000 --partition=RTX6000PRO,H100,L40S --exclude=node51,node52,node60 --job-name=p145a3_{m}_s{s} --export=ALL,CFG=configs/{name},RUN_ID={run_id},FINAL_CKPT=epoch_800.pt slurm/run_unit.sbatch")
    if a.write:
        (ROOT / "configs" / "P145A3_SHA256.json").write_text(json.dumps({"stage": STAGE, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "configs": shas}, indent=1))
        (ROOT / "slurm" / "p145a3_lines.txt").write_text("# P145 addendum 3 — STRESS eps 0.20, CIFAR-100 seeds 1-2; normal QOS, nice 1000; node51 / 52 / 60 excluded\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
