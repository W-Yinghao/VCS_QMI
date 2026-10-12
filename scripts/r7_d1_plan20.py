"""r7 R7-30 D1-20 unit plan and two-stage feed (frozen OPEN-6: learning rate {1e-4, 3e-4} on seed 0 per objective, CAL-chosen; seeds 1-2
and the ablations use the chosen rate).
  stage1 [--cache-job ID] --lines L : the 6 learning-rate-search units (S1 vcs, S2 logistic, S3 direct × {1e-4, 3e-4} × seed 0); the driver
                                    (slurm/r7_d1_20_submit.sh) file-gates them on the cache instead of afterok (purged-dependency lesson)
  choose --selectors-dir S        : per objective, the rate with the higher selected CAL Spearman (tie within 1e-4 -> 3e-4); refuses until
                                    both seed-0 runs of every objective are complete; writes S/lr_choice.json
  stage2 --selectors-dir S --lines L : the 12 dependent units: seeds 1, 2 × 3 objectives at the chosen rate; A1 (vcs, no question) and A2
                                    (direct, target d) × seeds 0, 1, 2 at the modified objective's chosen rate
Lines: main env (slurm/r7_d1_sel.sbatch), any of RTX6000PRO / H100 / L40S, node51 / 52 / 60 excluded, positional arguments, no --export;
fed by slurm/feed_normal.sh with CAP=27.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LRS = (1e-4, 3e-4)
OBJECTIVES = ("vcs", "logistic", "direct")


def tag(obj, lr, seed, noq=False, dtarget=False):
    return f"{obj}_lr{lr:g}_s{seed}" + ("_noq" if noq else "") + ("_dtarget" if dtarget else "")


def line(obj, lr, seed, extra="", dep=None):
    name = f"r7d1s_{tag(obj, lr, seed, '--no-question' in extra, '--target d' in extra)}"
    return (f"sbatch --parsable --job-name={name} --partition=RTX6000PRO,H100,L40S --exclude=node51,node52,node60"
            + (f" --dependency=afterok:{dep}" if dep else "") + f" slurm/r7_d1_sel.sbatch train {obj} {lr:g} {seed}" + (f" {extra}" if extra else ""))


def stage1(cache_job=None):
    return [line(o, lr, 0, dep=cache_job) for o in OBJECTIVES for lr in LRS]


def choose(sel_dir: Path):
    out = {}
    for o in OBJECTIVES:
        logs = {}
        for lr in LRS:
            f = sel_dir / f"{tag(o, lr, 0)}.json"; pt = sel_dir / f"{tag(o, lr, 0)}.pt"
            if not (f.exists() and pt.exists()):
                raise SystemExit(f"seed-0 unit {tag(o, lr, 0)} not complete")
            logs[lr] = json.loads(f.read_text())["selected_cal_spearman"]
        best = 3e-4 if abs(logs[1e-4] - logs[3e-4]) <= 1e-4 else max(LRS, key=lambda lr: logs[lr])
        out[o] = {"chosen_lr": best, "cal_spearman": {f"{lr:g}": v for lr, v in logs.items()}}
    return out


def stage2(choice):
    L = [line(o, choice[o]["chosen_lr"], s) for o in OBJECTIVES for s in (1, 2)]
    L += [line("vcs", choice["vcs"]["chosen_lr"], s, "--no-question") for s in (0, 1, 2)]
    L += [line("direct", choice["direct"]["chosen_lr"], s, "--target d") for s in (0, 1, 2)]
    return L


def write(lines, path, title):
    Path(path).write_text(f"# {title}; submit with: CAP=27 nohup setsid bash slurm/feed_normal.sh {path} >/dev/null 2>&1 &\n" + "\n".join(lines) + "\n")
    print(f"[r7_d1_plan20] {len(lines)} lines -> {path} (not submitted)")


def main() -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("stage1"); s1.add_argument("--cache-job", default=None); s1.add_argument("--lines", required=True)
    c = sub.add_parser("choose"); c.add_argument("--selectors-dir", required=True)
    s2 = sub.add_parser("stage2"); s2.add_argument("--selectors-dir", required=True); s2.add_argument("--lines", required=True)
    a = ap.parse_args()
    if a.cmd == "stage1":
        write(stage1(a.cache_job), a.lines, "R7-30 D1-20 stage 1: learning-rate search, seed 0 (the driver submits these after the cache file gate)")
    elif a.cmd == "choose":
        ch = choose(Path(a.selectors_dir)); (Path(a.selectors_dir) / "lr_choice.json").write_text(json.dumps(ch, indent=1)); print(json.dumps(ch, indent=1))
    else:
        ch = json.loads((Path(a.selectors_dir) / "lr_choice.json").read_text())
        write(stage2(ch), a.lines, "R7-30 D1-20 stage 2: seeds 1-2 at the chosen rate + ablations A1 / A2 × seeds 0-2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
