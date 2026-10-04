"""P129 — v6 §8.2 per-dataset affine grid for matched JS on the A-P3 structure (counterpart of P127), seed 0 (owner 2026-10-04: "全部提交，但是JS tuning grid还是要的").
Grid fixed by v6: (a, kappa) in {(1.5,.5),(2,.25),(2,.5),(2,.75),(3,.5),(3,.25)}; (2, .5) = existing JS-AP3 seed 0 (P114 C10 / P120 C100) — not re-run.
Each config = the JS-AP3 config of that dataset (seed 0) with only cosine_scale_init = a and cosine_bias_init = -a*kappa (fixed scorer), stage P129.
    python configs/make_p129_configs.py [--write]
"""
import argparse, hashlib, json, time
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
STAGE = "P129_v6_affine_grid_js"
PARENTS = {"c10": "cifar10_hpJS_AP3_views4_800ep_seed0.yaml", "c100": "cifar100_hpV6_JS_AP3_c100_views4_800ep_seed0.yaml"}
GRID = [(1.5, .5), (2, .25), (2, .75), (3, .5), (3, .25)]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    shas, lines = {}, []
    for ds, parent in PARENTS.items():
        for sc, kap in GRID:
            c = yaml.safe_load(open(ROOT / "configs" / parent)); cr = c["model"]["critic"]
            assert cr["affine_mode"] == "fixed" and float(cr["cosine_scale_init"]) == 2.0 and float(cr["cosine_bias_init"]) == -1.0
            assert c["objective"]["loss"] == "js_matched_logistic" and c["objective"]["js_fixed_scorer"] and c["objective"]["js_all_view_tokens"]
            cr["cosine_scale_init"] = float(sc); cr["cosine_bias_init"] = float(-sc * kap); c["run"]["stage"] = STAGE
            tag = f"{ds}_a{sc:g}_k{kap:g}"; run_id = f"P129_JS_{tag}_seed0"
            name = ("cifar100" if ds == "c100" else "cifar10") + f"_hpG6_JS_{tag}_seed0.yaml"
            txt = yaml.safe_dump(c, sort_keys=False); sha = hashlib.sha256(txt.encode()).hexdigest()
            shas[name] = {"sha256": sha, "run_id": run_id, "parent": parent, "a": sc, "kappa": kap, "b": -sc * kap}
            print(run_id, sha[:16]); lines.append(f"sbatch --parsable --partition=RTX6000PRO,H100,L40S --exclude=node51,node60 --job-name=p129_{tag} --export=ALL,CFG=configs/{name},RUN_ID={run_id},FINAL_CKPT=epoch_800.pt slurm/run_unit.sbatch")
            if a.write: (ROOT / "configs" / name).write_text(txt)
    if a.write:
        (ROOT / "configs" / "P129_SHA256.json").write_text(json.dumps({"stage": STAGE, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "configs": shas}, indent=1))
        (ROOT / "slurm" / "p129_lines.txt").write_text("# P129 v6 §8.2 matched-JS affine grid — normal QOS (never runfill); RTX6000PRO / H100 / L40S; node51 + node60 (throttled) excluded\n" + "\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
