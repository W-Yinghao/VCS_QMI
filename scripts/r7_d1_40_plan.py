"""r7 R7-40 unit plan (after the coordinator's staged freezes).  Every line carries its completion marker as a trailing comment
(<sbatch line> #MARK <path>); the drivers (slurm/r7_d1_40a_submit.sh, slurm/r7_d1_40b_submit.sh) resubmit a unit only after its job has left
the queue without the marker (one writer per output file), at most RETRIES times.  Positional wrapper arguments only (no --export).
  probe   [--s4-rule R] --lines L : 40a stage 0, the launch probe (sentinels S0–S7 with the frozen S4 rule) — marker probe/probe_report.json
  teacher --manifest-dir D --lines L : 40a, the CAL / FIT bundles of the frozen D1 shard plan (DEV bundles excluded: sealed until R7-40c) —
                                    marker teacher/bundle_<b>.done.json
  cache   --lines L               : 40b stage 0, the feature cache — marker cache/cache_d1_40.json
  stage1  --lines L               : 40b, S1 vcs / S2 logistic / S3 direct × lr {1e-4, 3e-4} × seed 0 (frozen OPEN-6) — marker selectors/<tag>.done.json
  choose  --selectors-dir S       : the frozen OPEN-6 rule (scripts/r7_d1_plan20.choose, imported unchanged): per objective the rate with the
                                    higher selected CAL Spearman, a tie within 1e-4 -> 3e-4 -> S/lr_choice.json
  stage2  --selectors-dir S --lines L : seeds 1, 2 × 3 objectives at the chosen rates (no ablations on Qwen: scope rule of the R7-40 plan)
Partitions (decided at the freeze; --partition, a comma list of RTX6000PRO / H100 / L40S): defaults — probe, teacher and cache units any of the
three (peak 18.9 GB measured); selector units RTX6000PRO / H100 (D1-20 execution note E5: one hardware class); node51 / node52 / node60 always
excluded.  Probe rules (decided at the freeze): --s4-rule min | rate:<r>, --s1-rule r720 | same_gpu.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7_d1_plan20 as P20  # noqa: E402  (frozen D1-20 planner: tags and the OPEN-6 choice)

R40 = Path("/home/infres/yinwang/CS_QMI/outputs/R7_D1/r7_40")
EXCL = "--exclude=node51,node52,node60"
LRS, OBJECTIVES = P20.LRS, P20.OBJECTIVES


def gpu(partition: str) -> str:
    """--partition / --exclude options of a unit line (partition list from the freeze; node51 / node52 / node60 always excluded)."""
    if not partition or any(c in partition for c in " =;'\"") or set(partition.split(",")) - {"RTX6000PRO", "H100", "L40S"}:
        raise SystemExit(f"bad partition list {partition!r} (allowed: RTX6000PRO, H100, L40S)")
    return f"--partition={partition} {EXCL}"


def probe_lines(root: Path, s4_rule="min", s1_rule="r720", partition="RTX6000PRO,H100,L40S"):
    return [(f"sbatch --parsable --job-name=r7d40_probe {gpu(partition)} slurm/r7_d1_40_gpu.sbatch probe {s4_rule} {s1_rule}", root / "probe" / "probe_report.json")]


def teacher_bundles(plan: dict):
    """CAL / FIT bundles of the frozen shard plan, in plan order (CAL first); DEV bundles are not part of R7-40a."""
    return [b for b, shards in plan["bundles"].items() if all(s.split("-")[0] in ("CAL", "FIT") for s in shards)]


def teacher_lines(root: Path, plan: dict, partition="RTX6000PRO,H100,L40S"):
    return [(f"sbatch --parsable --job-name=r7d40t_{b} {gpu(partition)} slurm/r7_d1_40_gpu.sbatch bundle {b}", root / "teacher" / f"bundle_{b}.done.json")
            for b in teacher_bundles(plan)]


def cache_lines(root: Path, partition="RTX6000PRO,H100,L40S"):
    return [(f"sbatch --parsable --job-name=r7d40c_cache {gpu(partition)} slurm/r7_d1_40_gpu.sbatch cache", root / "cache" / "cache_d1_40.json")]


def unit_line(root: Path, obj, lr, seed, partition="RTX6000PRO,H100"):
    t = P20.tag(obj, lr, seed)
    return (f"sbatch --parsable --job-name=r7d40s_{t} {gpu(partition)} slurm/r7_d1_40_sel.sbatch train {obj} {lr:g} {seed}", root / "selectors" / f"{t}.done.json")


def stage1(root: Path, partition="RTX6000PRO,H100"):
    return [unit_line(root, o, lr, 0, partition) for o in OBJECTIVES for lr in LRS]


def stage2(root: Path, choice: dict, partition="RTX6000PRO,H100"):
    return [unit_line(root, o, choice[o]["chosen_lr"], s, partition) for o in OBJECTIVES for s in (1, 2)]


def write(lines, path, title):
    with open(path, "w") as fh:
        fh.write(f"# {title}: <sbatch line> #MARK <completion marker>\n")
        for line, mark in lines:
            fh.write(f"{line} #MARK {mark}\n")
    print(f"[r7_d1_40_plan] {len(lines)} lines -> {path} (not submitted)")


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=str(R40)); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe"); p.add_argument("--lines", required=True); p.add_argument("--s4-rule", default="min")
    p.add_argument("--s1-rule", choices=["r720", "same_gpu"], default="r720"); p.add_argument("--partition", default="RTX6000PRO,H100,L40S")
    t = sub.add_parser("teacher"); t.add_argument("--manifest-dir", required=True); t.add_argument("--lines", required=True); t.add_argument("--partition", default="RTX6000PRO,H100,L40S")
    c = sub.add_parser("cache"); c.add_argument("--lines", required=True); c.add_argument("--partition", default="RTX6000PRO,H100,L40S")
    s1 = sub.add_parser("stage1"); s1.add_argument("--lines", required=True); s1.add_argument("--partition", default="RTX6000PRO,H100")
    ch = sub.add_parser("choose"); ch.add_argument("--selectors-dir", required=True)
    s2 = sub.add_parser("stage2"); s2.add_argument("--selectors-dir", required=True); s2.add_argument("--lines", required=True); s2.add_argument("--partition", default="RTX6000PRO,H100")
    a = ap.parse_args(); root = Path(a.root)
    if a.cmd == "probe":
        write(probe_lines(root, a.s4_rule, a.s1_rule, a.partition), a.lines, "R7-40a stage 0: launch probe")
    elif a.cmd == "teacher":
        write(teacher_lines(root, json.loads((Path(a.manifest_dir) / "d1_shards.json").read_text()), a.partition), a.lines, "R7-40a teacher bundles (CAL, then FIT)")
    elif a.cmd == "cache":
        write(cache_lines(root, a.partition), a.lines, "R7-40b stage 0: feature cache")
    elif a.cmd == "stage1":
        write(stage1(root, a.partition), a.lines, "R7-40b stage 1: learning-rate search, seed 0")
    elif a.cmd == "choose":
        ch = P20.choose(Path(a.selectors_dir)); (Path(a.selectors_dir) / "lr_choice.json").write_text(json.dumps(ch, indent=1)); print(json.dumps(ch, indent=1))
    else:
        write(stage2(root, json.loads((Path(a.selectors_dir) / "lr_choice.json").read_text()), a.partition), a.lines, "R7-40b stage 2: seeds 1-2 at the chosen rates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
