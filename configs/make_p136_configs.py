"""P136 — exploration for our method (VCS, A-P3 structure): longer schedule, more views, larger batch; CIFAR-10 and CIFAR-100, seed 0.
Owner 2026-10-04/05: keep >= 15 experiment jobs queued; explore everything except the plan and backbone for the best VCS performance; one full
seed per new idea first.  Each config = that dataset's A-P3 seed-0 config with ONE change:
  ep1600 : train.epochs 1600 (warm-up 10, cosine over 1600; knn / checkpoint epochs + 1200, 1600)
  v8     : views.count 8 (allowed named variant; all-view tokens: 56 ordered positive view pairs per image instead of 12)
  b512   : train.batch_size_images 512 (lr kept at 1e-3, scale_lr_with_batch false — disclosed)
    python configs/make_p136_configs.py [--write]
"""
import argparse, hashlib, json, time
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
STAGE = "P136_ap3_schedule_views_batch"
PARENTS = {"c10": "cifar10_hpY_AP3_views4_800ep_seed0.yaml", "c100": "cifar100_hpY_AP3_c100_views4_800ep_seed0.yaml"}

def apply(c, cell):
    if cell == "ep1600":
        c["train"]["epochs"] = 1600
        for key in ("knn_epochs",):
            c["evaluation"][key] = sorted(set(c["evaluation"][key]) | {1200, 1600})
        c["logging"]["checkpoint_epochs"] = sorted(set(c["logging"]["checkpoint_epochs"]) | {1200, 1600})
        return "epoch_1600.pt"
    if cell == "v8":
        c["views"]["count"] = 8; return "epoch_800.pt"
    if cell == "b512":
        c["train"]["batch_size_images"] = 512; return "epoch_800.pt"
    raise ValueError(cell)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    shas, lines = {}, []
    for ds, parent in PARENTS.items():
        for cell in ("ep1600", "v8", "b512"):
            c = yaml.safe_load(open(ROOT / "configs" / parent)); final = apply(c, cell); c["run"]["stage"] = STAGE
            tag = f"{ds}_{cell}"; run_id = f"P136_AP3_{tag}_seed0"
            name = ("cifar100" if ds == "c100" else "cifar10") + f"_hpX_AP3_{tag}_seed0.yaml"
            txt = yaml.safe_dump(c, sort_keys=False); sha = hashlib.sha256(txt.encode()).hexdigest()
            shas[name] = {"sha256": sha, "run_id": run_id, "parent": parent, "cell": cell, "final": final}; print(run_id, sha[:16])
            lines.append(f"sbatch --parsable --partition=RTX6000PRO,H100,L40S --exclude=node51,node60 --job-name=p136_{tag} --export=ALL,CFG=configs/{name},RUN_ID={run_id},FINAL_CKPT={final} slurm/run_unit.sbatch")
            if a.write: (ROOT / "configs" / name).write_text(txt)
    if a.write:
        (ROOT / "configs" / "P136_SHA256.json").write_text(json.dumps({"stage": STAGE, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "configs": shas}, indent=1))
        (ROOT / "slurm" / "p136_lines.txt").write_text("# P136 A-P3 schedule / views / batch exploration — normal QOS; RTX6000PRO / H100 / L40S; node51 + node60 excluded\n" + "\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
