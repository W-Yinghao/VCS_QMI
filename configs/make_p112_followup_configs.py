"""P112 addendum 1 — follow-up seeds 1, 2 for screen cells beating their setting's G2 seed-0 reference by >= 0.5 linear (frozen P112 rule):
strong aug (a 2, kappa 0.75) 88.22 vs 86.86 (+1.36) and strong aug (a 2, kappa 0.25) 87.88 (+1.02).  Each config = the P111 G2-strong config of that
seed with only (a, b) changed (same construction as make_p112_configs.py).
"""
import argparse, hashlib, json, time
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
STAGE = "P112_fixed_scale_sensitivity"
CELLS = [("strong", 2.0, 0.75), ("strong", 2.0, 0.25)]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    man = json.load(open(ROOT / "configs" / "P112_SHA256.json")); lines = []
    for setting, sc, kap in CELLS:
        for s in (1, 2):
            parent = f"cifar10_hpZ_G2_augstrong_views4_800ep_seed{s}.yaml"
            c = yaml.safe_load(open(ROOT / "configs" / parent)); crit = c["model"]["critic"]
            assert crit["affine_mode"] == "fixed" and crit["cosine_scale_init"] == 2.0 and crit["cosine_bias_init"] == -1.0 and c["run"]["seed"] == s
            crit["cosine_scale_init"] = float(sc); crit["cosine_bias_init"] = float(-sc * kap); c["run"]["stage"] = STAGE
            tag = f"{setting}_a{sc:g}_k{kap:g}"; run_id = f"P112_{tag}_seed{s}"; name = f"cifar10_hpF_{tag}_seed{s}.yaml"
            txt = yaml.safe_dump(c, sort_keys=False); sha = hashlib.sha256(txt.encode()).hexdigest()
            man["configs"][name] = {"sha256": sha, "run_id": run_id, "parent": parent, "a": sc, "b": -sc * kap, "kappa": kap, "addendum": 1}
            print(run_id, sha[:16]); lines.append(f"sbatch --parsable --partition=RTX6000PRO,H100 --job-name=p112_{tag}_s{s} --export=ALL,CFG=configs/{name},RUN_ID={run_id},FINAL_CKPT=epoch_800.pt slurm/run_unit.sbatch")
            if a.write: (ROOT / "configs" / name).write_text(txt)
    if a.write:
        man.setdefault("addendum1_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())); (ROOT / "configs" / "P112_SHA256.json").write_text(json.dumps(man, indent=1))
        (ROOT / "slurm" / "p112_followup_lines.txt").write_text("# P112 addendum 1 — follow-up seeds (normal QOS, never runfill; node60 allowed)\n" + "\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
