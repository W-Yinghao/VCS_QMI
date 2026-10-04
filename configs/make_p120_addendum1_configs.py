"""P120 addendum 1 — CIFAR-100 A-P3 vs matched JS, fresh seeds 3, 4 (4 units): the seed-0 configs with only run.seed (and run.stage) changed,
exactly as P107 addendum 2 made A-P3 CIFAR-10 seeds 3, 4.
    python configs/make_p120_addendum1_configs.py [--write]
"""
import argparse, hashlib, json, time
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
STAGE = "P120_addendum1_c100_seeds34"
CELLS = {"AP3": ("cifar100_hpY_AP3_c100_views4_800ep_seed0.yaml", "cifar100_hpY_AP3_c100_views4_800ep_seed{s}.yaml", "P120A1_AP3_c100_views4_800ep_seed{s}"),
         "JS": ("cifar100_hpV6_JS_AP3_c100_views4_800ep_seed0.yaml", "cifar100_hpV6_JS_AP3_c100_views4_800ep_seed{s}.yaml", "P120_JS_AP3_c100_views4_800ep_seed{s}")}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    shas, lines = {}, []
    for s in (3, 4):
        for key, (parent, name_t, run_t) in CELLS.items():
            c = yaml.safe_load(open(ROOT / "configs" / parent)); assert c["run"]["seed"] == 0
            c["run"]["seed"] = s; c["run"]["stage"] = STAGE
            name, run_id = name_t.format(s=s), run_t.format(s=s)
            assert not (ROOT / "configs" / name).exists(), name
            txt = yaml.safe_dump(c, sort_keys=False); sha = hashlib.sha256(txt.encode()).hexdigest()
            shas[name] = {"sha256": sha, "run_id": run_id, "parent": parent, "seed": s}; print(run_id, sha[:16])
            lines.append(f"sbatch --parsable --partition=RTX6000PRO,H100,L40S --job-name=p120a1_{key.lower()}_s{s} --export=ALL,CFG=configs/{name},RUN_ID={run_id},FINAL_CKPT=epoch_800.pt slurm/run_unit.sbatch")
            if a.write: (ROOT / "configs" / name).write_text(txt)
    if a.write:
        (ROOT / "configs" / "P120_ADDENDUM1_SHA256.json").write_text(json.dumps({"stage": STAGE, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "configs": shas}, indent=1))
        (ROOT / "slurm" / "p120a1_lines.txt").write_text("# P120 addendum 1 — C100 A-P3 / JS seeds 3, 4 — normal QOS; RTX6000PRO / H100 / L40S\n" + "\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
