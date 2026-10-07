"""P138 addendum 1 — seeds 1-2 of the four K = 16 cells (retention rule triggered for all four).  Each config = the P138 seed-0 config with ONLY
run.seed and run.stage changed (the K = all parents' seeds 1-2 exist: P107 A-P3 C10 / C100, P114 JS-AP3 C10, P120 JS-AP3 C100).
    python configs/make_p138a1_configs.py [--write]
"""
import argparse, hashlib, json, time
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
STAGE = "P138A1_v7_pair_k16_seeds"
CELLS = ["vcs_c10", "js_c10", "vcs_c100", "js_c100"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    shas, lines = {}, []
    for tag in CELLS:
        ds = "cifar100" if tag.endswith("c100") else "cifar10"
        base = f"{ds}_hpPK16_{tag}_seed0.yaml"
        for s in (1, 2):
            c = yaml.safe_load(open(ROOT / "configs" / base)); assert c["run"]["seed"] == 0
            c["run"]["seed"] = s; c["run"]["stage"] = STAGE
            name = f"{ds}_hpPK16_{tag}_seed{s}.yaml"; run_id = f"P138_PAIR_K16_{tag}_seed{s}"
            txt = yaml.safe_dump(c, sort_keys=False); sha = hashlib.sha256(txt.encode()).hexdigest()
            shas[name] = {"sha256": sha, "run_id": run_id, "base": base}; print(run_id, sha[:16])
            lines.append(f"sbatch --parsable --partition=RTX6000PRO,H100,L40S --exclude=node51,node52,node60 --job-name=p138a1_{tag}_s{s} --export=ALL,CFG=configs/{name},RUN_ID={run_id},FINAL_CKPT=epoch_800.pt slurm/run_unit.sbatch")
            if a.write: (ROOT / "configs" / name).write_text(txt)
    if a.write:
        (ROOT / "configs" / "P138A1_SHA256.json").write_text(json.dumps({"stage": STAGE, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "configs": shas}, indent=1))
        (ROOT / "slurm" / "p138a1_lines.txt").write_text("# P138 addendum 1 — seeds 1-2 of the four K = 16 cells; normal QOS; node51 / 52 / 60 excluded\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
