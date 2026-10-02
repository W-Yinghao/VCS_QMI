"""P107 addendum 2 — A-P3 confirmation layer 2 (A-P3 stably positive vs SimCLR on seeds 0-2 by the frozen addendum-1 rule): eight 800-epoch units.
  fresh seeds 3, 4 : A-P3 on the recipe base with run.seed = s (SimCLR seeds 3, 4 exist from P111)
  strong aug       : A-P3 seeds 0, 1, 2 with the P89 strong views block (vs P89 SimCLR strong, P111 G2 strong)
  CIFAR-100        : A-P3 seeds 0, 1, 2 on the P91 CIFAR-100 8x base (vs P91 VCS / SimCLR, P111 G2 CIFAR-100)
A-P3 settings come from make_p107_configs.apply(c, "A-P3"); the strong block from make_p111_configs.strong; nothing is re-typed.
    python configs/make_p107_confirm2_configs.py [--write]
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "configs"))
from make_p107_configs import apply  # noqa: E402
from make_p111_configs import strong, load  # noqa: E402
STAGE = "P107_confirm2"

def cells():
    out = []
    for s in (3, 4):
        b = load("cifar10_hpK_a5_views4_800ep_vcs_seed0.yaml"); b["run"]["seed"] = s
        out.append((f"P107_AP3_views4_800ep_seed{s}", "cifar10_hpY_AP3_views4_800ep_seed%d.yaml" % s, apply(b, "A-P3")))
    for s in (0, 1, 2):
        ref = load("cifar10_hpO_a5_views4_800ep_augstrong_vcs_seed0.yaml" if s == 0 else f"cifar10_hpS_a5_views4_800ep_augstrong_vcs_seed{s}.yaml")
        out.append((f"P107_AP3_augstrong_views4_800ep_seed{s}", f"cifar10_hpY_AP3_augstrong_views4_800ep_seed{s}.yaml",
                    strong(apply(load(f"cifar10_hpK_a5_views4_800ep_vcs_seed{s}.yaml"), "A-P3"), ref)))
    for s in (0, 1, 2):
        out.append((f"P107_AP3_c100_views4_800ep_seed{s}", f"cifar100_hpY_AP3_c100_views4_800ep_seed{s}.yaml",
                    apply(load(f"cifar100_hpS_vcs_a5_views4_800ep_seed{s}.yaml"), "A-P3")))
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    shas, lines = {}, []
    for run_id, name, c in cells():
        c["run"]["stage"] = STAGE
        ce = c["logging"]["checkpoint_epochs"]
        if 20 not in ce: c["logging"]["checkpoint_epochs"] = sorted(ce + [20])
        txt = yaml.safe_dump(c, sort_keys=False); sha = hashlib.sha256(txt.encode()).hexdigest()
        shas[name] = {"sha256": sha, "run_id": run_id}
        print(f"{run_id:40s} {sha[:16]} seed={c['run']['seed']} data={c['data']['name']}")
        lines.append(f"sbatch --parsable --partition=RTX6000PRO,H100 --job-name={run_id.lower().replace('_views4_800ep','')[:28]} "
                     f"--export=ALL,CFG=configs/{name},RUN_ID={run_id},FINAL_CKPT=epoch_800.pt slurm/run_unit.sbatch")
        if a.write: (ROOT / "configs" / name).write_text(txt)
    if a.write:
        (ROOT / "configs" / "P107_confirm2_SHA256.json").write_text(json.dumps({"stage": STAGE, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "configs": shas}, indent=1))
        (ROOT / "slurm" / "p107_confirm2_lines.txt").write_text("# P107 addendum 2 — A-P3 confirmation layer 2 (normal QOS, never runfill; node60 allowed)\n" + "\n".join(lines) + "\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
