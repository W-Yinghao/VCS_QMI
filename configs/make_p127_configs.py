"""P127 — v6 §8.2 per-dataset affine grid on the A-P3 structure (all-view tokens, full gradient), seed 0 (owner 2026-10-03: "再提交几个，占满30个的slot限制").
Grid fixed by v6: (a, kappa) in {(1.5,.5),(2,.25),(2,.5),(2,.75),(3,.5),(3,.25)}; (2, .5) = existing A-P3 seed 0 (P107) — not re-run.
Each config = the A-P3 config of that dataset (seed 0) with only cosine_scale_init = a and cosine_bias_init = -a*kappa (fixed scorer), stage P127.
    python configs/make_p127_configs.py [--write]
"""
import argparse, hashlib, json, time
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
STAGE = "P127_v6_affine_grid_ap3"
PARENTS = {"c10": "cifar10_hpY_AP3_views4_800ep_seed0.yaml", "c100": "cifar100_hpY_AP3_c100_views4_800ep_seed0.yaml"}
GRID = [(1.5, .5), (2, .25), (2, .75), (3, .5), (3, .25)]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    shas, lines = {}, []
    for ds, parent in PARENTS.items():
        for sc, kap in GRID:
            c = yaml.safe_load(open(ROOT / "configs" / parent)); cr = c["model"]["critic"]
            assert cr["affine_mode"] == "fixed" and float(cr["cosine_scale_init"]) == 2.0 and float(cr["cosine_bias_init"]) == -1.0
            cr["cosine_scale_init"] = float(sc); cr["cosine_bias_init"] = float(-sc * kap); c["run"]["stage"] = STAGE
            tag = f"{ds}_a{sc:g}_k{kap:g}"; run_id = f"P127_AP3_{tag}_seed0"
            name = ("cifar100" if ds == "c100" else "cifar10") + f"_hpG6_AP3_{tag}_seed0.yaml"
            txt = yaml.safe_dump(c, sort_keys=False); sha = hashlib.sha256(txt.encode()).hexdigest()
            shas[name] = {"sha256": sha, "run_id": run_id, "parent": parent, "a": sc, "kappa": kap, "b": -sc * kap}
            print(run_id, sha[:16]); lines.append(f"sbatch --parsable --partition=RTX6000PRO,H100,L40S --job-name=p127_{tag} --export=ALL,CFG=configs/{name},RUN_ID={run_id},FINAL_CKPT=epoch_800.pt slurm/run_unit.sbatch")
            if a.write: (ROOT / "configs" / name).write_text(txt)
    if a.write:
        (ROOT / "configs" / "P127_SHA256.json").write_text(json.dumps({"stage": STAGE, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "configs": shas}, indent=1))
        (ROOT / "slurm" / "p127_lines.txt").write_text("# P127 v6 §8.2 A-P3 affine grid — normal QOS (never runfill); RTX6000PRO / H100 / L40S; node60 allowed\n" + "\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
