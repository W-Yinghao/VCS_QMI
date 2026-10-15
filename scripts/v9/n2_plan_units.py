"""Plan the V9-N2 Flickr block runs after the pre-registered merge rule (frozen protocol, owner / coordinator 2026-10-15).

    python scripts/v9/n2_plan_units.py --precheck-dir reports/V9/V9_N2_precheck --seeds 901 902 903 --kind formal --dest reports/V9/V9_N2_units
    python scripts/v9/n2_plan_units.py --precheck-dir <out_root>/f30k --seeds 917 --kind tuning --dest reports/V9/V9_N2_units   # after the seed-917 precheck

Methods of the block (NR_FORMAL_UNITS.csv, Flickr rows, SGR-student arms): Unweighted_A, VCS_Weight, BCE_Weight, RuLSIF_Weight, Nystrom_Weight, RankMass;
NCR_A / L2RM_A are wave 2 (official clones).  In a cell (rho, seed) the precheck's coefficient groups say which arms are ONE training: the group is run
once under its first member (sorted) and the result is attributed to every member.  Output: <kind>_units.json (run list with members), group files
<kind>_g<k>.txt (<= `--per-job` runs per GPU job, 24 h partition limit), and <kind>_feed.txt (one sbatch line per group for slurm/v9/n2_feed.sh)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ARMS = ("Unweighted_A", "VCS_Weight", "BCE_Weight", "RuLSIF_Weight", "Nystrom_Weight", "RankMass")
SB = ("sbatch --parsable --job-name={name} --export=ALL,V9_SGR_PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python,"
      "DATA_ROOT=/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data,VOCAB=/projects/EEG-foundation-model/yinghao/code_v9/SGRAF_py3/vocab,"
      "OUT_ROOT=/projects/EEG-foundation-model/yinghao/outputs_v9/n2,HOST_LOCK=reports/V9/HOST_LOCK.yaml{formal} slurm/v9/n2_unit.sbatch group {gfile}")


def groups_of(precheck_dir: Path, ds: str, rho: float, seed: int) -> list:
    cands = [precheck_dir / f"precheck_{ds}_rho{rho:g}_s{seed}.json", precheck_dir / f"rho{rho:g}" / f"s{seed}" / "precheck" / f"precheck_{ds}_rho{rho:g}_s{seed}.json"]
    p = next((c for c in cands if c.exists()), None)
    if p is None:
        raise SystemExit(f"no precheck for rho {rho} seed {seed} under {precheck_dir}")
    rec = json.loads(p.read_text())
    if not rec["formal_block_allowed"]:
        raise SystemExit(f"precheck refused rho {rho} seed {seed}: {rec['refusal_reason']}")
    return [sorted(g) for g in rec["loss_coefficients"]["groups"]], p


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--precheck-dir", required=True); ap.add_argument("--dataset", default="f30k"); ap.add_argument("--rhos", nargs="+", type=float, default=[0.4, 0])
    ap.add_argument("--seeds", nargs="+", type=int, required=True); ap.add_argument("--kind", choices=["formal", "tuning"], required=True); ap.add_argument("--lr-mults", nargs="+", type=float, default=None)
    ap.add_argument("--per-job", type=int, default=2); ap.add_argument("--dest", required=True); a = ap.parse_args()
    mults = a.lr_mults or ([1.0] if a.kind == "formal" else [0.5, 1.0, 2.0]); dest = Path(a.dest); dest.mkdir(parents=True, exist_ok=True); runs = []
    for seed in a.seeds:
        for rho in a.rhos:
            groups, src = groups_of(Path(a.precheck_dir), a.dataset, rho, seed); seen = set()
            for m in ARMS:
                if m in seen:
                    continue
                g = next((g for g in groups if m in g), [m]); members = [x for x in g if x in ARMS]; seen.update(members); canon = members[0]
                for lm in mults:
                    runs.append({"dataset": a.dataset, "rho": rho, "seed": seed, "method": canon, "lr_mult": lm, "members": members, "kind": a.kind,
                                 "unit_dir": f"{a.dataset}/rho{rho:g}/s{seed}/unit_{canon}_lr{lm:g}", "precheck": str(src),
                                 "note": "one training reported for every member (identical loss coefficients; not independent replicates)" if len(members) > 1 else ""})
    # order: rho 0.4 first (the primary contrasts), then rho 0; within rho by seed then method; multipliers interleaved so every (method, rho) gets its x1 early
    runs.sort(key=lambda r: (-r["rho"], r["lr_mult"] != 1.0, r["seed"], ARMS.index(r["method"]), r["lr_mult"]))
    (dest / f"{a.kind}_units.json").write_text(json.dumps({"kind": a.kind, "dataset": a.dataset, "rhos": a.rhos, "seeds": a.seeds, "lr_mults": mults, "n_runs": len(runs),
                                                            "n_arm_rows_covered": sum(len(r["members"]) for r in runs), "runs": runs}, indent=1))
    feed = []
    for k in range(0, len(runs), a.per_job):
        chunk = runs[k:k + a.per_job]; gfile = dest / f"{a.kind}_g{k // a.per_job + 1:02d}.txt"
        gfile.write_text("".join(f"{r['dataset']} {r['rho']:g} {r['method']} {r['seed']} {r['lr_mult']:g}\n" for r in chunk))
        feed.append(SB.format(name=f"v9n2_{a.kind[0]}{k // a.per_job + 1:02d}", formal=",FORMAL=1" if a.kind == "formal" else "", gfile=gfile))
    (dest / f"{a.kind}_feed.txt").write_text("\n".join(feed) + "\n")
    print(f"{a.kind}: {len(runs)} runs covering {sum(len(r['members']) for r in runs)} arm rows -> {len(feed)} jobs ({a.per_job}/job); merged runs: "
          + "; ".join(f"rho{r['rho']:g} s{r['seed']} lr{r['lr_mult']:g}: {'+'.join(r['members'])}" for r in runs if len(r["members"]) > 1)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
