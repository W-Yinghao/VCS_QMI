"""P138 addendum 2 — seeds 3-4 for the three K = 16 cells whose non-inferiority was not shown at 3 seeds (JS C10, VCS C100, JS C100), plus the
missing K = all parent JS-AP3 C10 seeds 3-4 (P114 family).  Each config = the seed-0 config with ONLY run.seed / run.stage changed.  Existing
parents seeds 3-4: P120A1 A-P3 C100, P120 JS-AP3 C100.
    python configs/make_p138a2_configs.py [--write]   (with slurm/common.sh exported)
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
STAGE = "P138A2_v7_pair_k16_seeds34"
UNITS = [(f"cifar10_hpPK16_js_c10_seed0.yaml", "cifar10_hpPK16_js_c10_seed{s}.yaml", "P138_PAIR_K16_js_c10_seed{s}", "p138a2_js_c10_s{s}"),
         (f"cifar100_hpPK16_vcs_c100_seed0.yaml", "cifar100_hpPK16_vcs_c100_seed{s}.yaml", "P138_PAIR_K16_vcs_c100_seed{s}", "p138a2_vcs_c100_s{s}"),
         (f"cifar100_hpPK16_js_c100_seed0.yaml", "cifar100_hpPK16_js_c100_seed{s}.yaml", "P138_PAIR_K16_js_c100_seed{s}", "p138a2_js_c100_s{s}"),
         ("cifar10_hpJS_AP3_views4_800ep_seed0.yaml", "cifar10_hpJS_AP3_views4_800ep_seed{s}.yaml", "P114_JSAP3_views4_800ep_seed{s}", "p138a2_parent_js_c10_s{s}")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "src")); from vcs_ssl.config import load_config
    shas, lines = {}, []
    for base, pat, rid, job in UNITS:
        for s in (3, 4):
            c = yaml.safe_load(open(ROOT / "configs" / base)); assert c["run"]["seed"] == 0
            c["run"]["seed"] = s; c["run"]["stage"] = STAGE
            name, run_id = pat.format(s=s), rid.format(s=s)
            assert not (ROOT / "configs" / name).exists() or a.write is False or yaml.safe_load(open(ROOT / "configs" / name))["run"]["seed"] == s
            txt = yaml.safe_dump(c, sort_keys=False); sha = hashlib.sha256(txt.encode()).hexdigest()
            if a.write:
                (ROOT / "configs" / name).write_text(txt); load_config(ROOT / "configs" / name, require_dirs=False)
            shas[name] = {"sha256": sha, "run_id": run_id, "base": base}; print(run_id, sha[:16])
            lines.append(f"sbatch --parsable --partition=RTX6000PRO,H100,L40S --exclude=node51,node52,node60 --job-name={job.format(s=s)} --export=ALL,CFG=configs/{name},RUN_ID={run_id},FINAL_CKPT=epoch_800.pt slurm/run_unit.sbatch")
    if a.write:
        (ROOT / "configs" / "P138A2_SHA256.json").write_text(json.dumps({"stage": STAGE, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "configs": shas}, indent=1))
        (ROOT / "slurm" / "p138a2_lines.txt").write_text("# P138 addendum 2 — seeds 3-4 (3 K16 cells + JS-AP3 C10 parent); normal QOS; node51 / 52 / 60 excluded\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
