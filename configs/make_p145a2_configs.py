"""P145 addendum 2 — STRESS eps = 0.20 (owner budget confirmation 2026-10-08), seed 0, VCS / JS / SimCLR x CIFAR-10 / CIFAR-100.  Each config = the
P145 eps-0.10 seed-0 config with ONLY pairing.stress_epsilon 0.10 -> 0.20 and run.stage changed.
    python configs/make_p145a2_configs.py [--write]   (with slurm/common.sh exported: the policy load needs OUTPUT_ROOT)
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
STAGE = "P145A2_v7_stress_eps020"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "src")); from vcs_ssl.config import load_config
    shas, lines = {}, []
    for ds, dsn in (("c10", "cifar10"), ("c100", "cifar100")):
        for m in ("vcs", "js", "simclr"):
            base = f"{dsn}_hpST10_{m}_{ds}_seed0.yaml"
            c = yaml.safe_load(open(ROOT / "configs" / base)); assert c["run"]["seed"] == 0 and c["pairing"]["stress_epsilon"] == 0.10
            c["pairing"]["stress_epsilon"] = 0.20; c["run"]["stage"] = STAGE
            name = f"{dsn}_hpST20_{m}_{ds}_seed0.yaml"; run_id = f"P145_STRESS20_{m}_{ds}_seed0"
            txt = yaml.safe_dump(c, sort_keys=False); sha = hashlib.sha256(txt.encode()).hexdigest()
            if a.write:
                (ROOT / "configs" / name).write_text(txt); load_config(ROOT / "configs" / name, require_dirs=False)
            shas[name] = {"sha256": sha, "run_id": run_id, "base": base}; print(run_id, sha[:16])
            lines.append(f"sbatch --parsable --partition=RTX6000PRO,H100,L40S --exclude=node51,node52,node60 --job-name=p145a2_{m}_{ds} --export=ALL,CFG=configs/{name},RUN_ID={run_id},FINAL_CKPT=epoch_800.pt slurm/run_unit.sbatch")
    if a.write:
        (ROOT / "configs" / "P145A2_SHA256.json").write_text(json.dumps({"stage": STAGE, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "configs": shas}, indent=1))
        (ROOT / "slurm" / "p145a2_lines.txt").write_text("# P145 addendum 2 — STRESS eps 0.20, seed 0; normal QOS; node51 / 52 / 60 excluded\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
