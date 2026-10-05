"""P137 — v7 V7-HEAD: projector output dimension 128 -> 512 for VCS (A-P3), matched JS-AP3 and SimCLR, CIFAR-10 and CIFAR-100, seed 0.
Each config = that method's / dataset's seed-0 parent config with ONLY model.projector.output_dim = 512 (and run.stage).  VCS / JS keep the common
scorer (a, kappa) = (2, 0.5) and lr 1e-3; SimCLR keeps its recipe.  The 128-dim parents are the reused controls.
    python configs/make_p137_configs.py [--write]
"""
import argparse, hashlib, json, time
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
STAGE = "P137_v7_head_projector512"
PARENTS = {("vcs", "c10"): "cifar10_hpY_AP3_views4_800ep_seed0.yaml", ("js", "c10"): "cifar10_hpJS_AP3_views4_800ep_seed0.yaml",
           ("simclr", "c10"): "cifar10_hpN_simclr_views4_800ep_seed0.yaml", ("vcs", "c100"): "cifar100_hpY_AP3_c100_views4_800ep_seed0.yaml",
           ("js", "c100"): "cifar100_hpV6_JS_AP3_c100_views4_800ep_seed0.yaml", ("simclr", "c100"): "cifar100_hpS_simclr_views4_800ep_seed0.yaml"}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    shas, lines = {}, []
    for (m, ds), parent in PARENTS.items():
        c = yaml.safe_load(open(ROOT / "configs" / parent)); assert c["model"]["projector"]["output_dim"] == 128
        c["model"]["projector"]["output_dim"] = 512; c["model"]["projector"]["v7_head_width"] = 512; c["run"]["stage"] = STAGE
        tag = f"{m}_{ds}"; run_id = f"P137_HEAD512_{tag}_seed0"
        name = ("cifar100" if ds == "c100" else "cifar10") + f"_hpH512_{tag}_seed0.yaml"
        txt = yaml.safe_dump(c, sort_keys=False); sha = hashlib.sha256(txt.encode()).hexdigest()
        shas[name] = {"sha256": sha, "run_id": run_id, "parent": parent, "method": m}; print(run_id, sha[:16])
        lines.append(f"sbatch --parsable --partition=RTX6000PRO,H100,L40S --exclude=node51,node52,node60 --job-name=p137_{tag} --export=ALL,CFG=configs/{name},RUN_ID={run_id},FINAL_CKPT=epoch_800.pt slurm/run_unit.sbatch")
        if a.write: (ROOT / "configs" / name).write_text(txt)
    if a.write:
        (ROOT / "configs" / "P137_SHA256.json").write_text(json.dumps({"stage": STAGE, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "configs": shas}, indent=1))
        (ROOT / "slurm" / "p137_lines.txt").write_text("# P137 V7-HEAD projector 512 — normal QOS; RTX6000PRO / H100 / L40S; node51, node52, node60 excluded\n" + "\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
