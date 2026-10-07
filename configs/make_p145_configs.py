"""P145 — v7 V7-STRESS: training positive pairs contaminated at eps = 0.10 for VCS (A-P3), matched JS-AP3 and SimCLR, CIFAR-10 and CIFAR-100,
seed 0 (plan §6.3).  Each config = that method's / dataset's seed-0 parent (the P137 parents) with ONLY pairing.stress_epsilon = 0.10 (and
run.stage).  The clean (eps = 0) parents are the reused baselines.  VCS / JS keep the common scorer (a, kappa) = (2, 0.5) and lr 1e-3; SimCLR
keeps its recipe.
    python configs/make_p145_configs.py [--write]
"""
import argparse, hashlib, json, time
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
STAGE = "P145_v7_stress_eps0.10"
EPS = 0.10
PARENTS = {("vcs", "c10"): "cifar10_hpY_AP3_views4_800ep_seed0.yaml", ("js", "c10"): "cifar10_hpJS_AP3_views4_800ep_seed0.yaml",
           ("simclr", "c10"): "cifar10_hpN_simclr_views4_800ep_seed0.yaml", ("vcs", "c100"): "cifar100_hpY_AP3_c100_views4_800ep_seed0.yaml",
           ("js", "c100"): "cifar100_hpV6_JS_AP3_c100_views4_800ep_seed0.yaml", ("simclr", "c100"): "cifar100_hpS_simclr_views4_800ep_seed0.yaml"}


def build(m: str, ds: str) -> dict:
    c = yaml.safe_load(open(ROOT / "configs" / PARENTS[(m, ds)]))
    assert "stress_epsilon" not in c["pairing"]
    c["pairing"]["stress_epsilon"] = EPS; c["run"]["stage"] = STAGE
    return c


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    shas, lines = {}, []
    for (m, ds), parent in PARENTS.items():
        c = build(m, ds)
        tag = f"{m}_{ds}"; run_id = f"P145_STRESS10_{tag}_seed0"
        name = ("cifar100" if ds == "c100" else "cifar10") + f"_hpST10_{tag}_seed0.yaml"
        txt = yaml.safe_dump(c, sort_keys=False); sha = hashlib.sha256(txt.encode()).hexdigest()
        shas[name] = {"sha256": sha, "run_id": run_id, "parent": parent, "method": m}; print(run_id, sha[:16])
        lines.append(f"sbatch --parsable --partition=RTX6000PRO,H100,L40S --exclude=node51,node52,node60 --job-name=p145_{tag} --export=ALL,CFG=configs/{name},RUN_ID={run_id},FINAL_CKPT=epoch_800.pt slurm/run_unit.sbatch")
        if a.write: (ROOT / "configs" / name).write_text(txt)
    if a.write:
        (ROOT / "configs" / "P145_SHA256.json").write_text(json.dumps({"stage": STAGE, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "configs": shas}, indent=1))
        (ROOT / "slurm" / "p145_lines.txt").write_text("# P145 V7-STRESS eps 0.10 — normal QOS; RTX6000PRO / H100 / L40S; node51, node52, node60 excluded\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
