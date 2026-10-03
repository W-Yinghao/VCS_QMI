"""P120 — v6 V6-C100-CORE (owner 2026-10-03: "这是新的一版分析和设计，你来理解，并组织实验提交"): six CIFAR-100 800-epoch units.
Each YAML = the completed CIFAR-100 A-P3 config `configs/cifar100_hpY_AP3_c100_views4_800ep_seed{s}.yaml` (P107 addendum 2) with only:
  V6-C100-JS   : make_p114_configs.apply  (objective.loss -> js_matched_logistic, js_fixed_scorer, js_all_view_tokens; fixed f = 2s - 1, all-view, full gradient)
  V6-C100-FREE : critic.affine_mode fixed -> learned (a, b start at (2, -1), as P107 A-P3F); pairing / routing unchanged
and run.stage = P120_v6_c100_core.   python configs/make_p120_configs.py [--write]
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "configs"))
from make_p114_configs import apply as js_apply  # noqa: E402
STAGE = "P120_v6_c100_core"
PARENT = "cifar100_hpY_AP3_c100_views4_800ep_seed{s}.yaml"

def free_apply(c):
    cr = c["model"]["critic"]; assert cr["affine_mode"] == "fixed" and float(cr["cosine_scale_init"]) == 2.0 and float(cr["cosine_bias_init"]) == -1.0
    cr["affine_mode"] = "learned"; return c

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    shas, lines = {}, []
    for tag, fn in (("JS", js_apply), ("FREE", free_apply)):
        for s in (0, 1, 2):
            c = fn(json.loads(json.dumps(yaml.safe_load(open(ROOT / "configs" / PARENT.format(s=s))))))
            c["run"]["stage"] = STAGE
            name = f"cifar100_hpV6_{tag}_AP3_c100_views4_800ep_seed{s}.yaml"; run_id = f"P120_{tag}_AP3_c100_views4_800ep_seed{s}"
            txt = yaml.safe_dump(c, sort_keys=False); sha = hashlib.sha256(txt.encode()).hexdigest()
            shas[name] = {"sha256": sha, "run_id": run_id, "parent": PARENT.format(s=s), "variant": tag}
            print(run_id, sha[:16])
            lines.append(f"sbatch --parsable --partition=RTX6000PRO,H100 --job-name=p120_{tag.lower()}_s{s} --export=ALL,CFG=configs/{name},RUN_ID={run_id},FINAL_CKPT=epoch_800.pt slurm/run_unit.sbatch")
            if a.write: (ROOT / "configs" / name).write_text(txt)
    if a.write:
        (ROOT / "configs" / "P120_SHA256.json").write_text(json.dumps({"stage": STAGE, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "configs": shas}, indent=1))
        (ROOT / "slurm" / "p120_lines.txt").write_text("# P120 v6 V6-C100-CORE — normal QOS (never runfill); node60 allowed\n" + "\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
