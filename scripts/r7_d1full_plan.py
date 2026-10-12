"""r7 R7-30 D1-FULL unit plan (frozen D1-20 recipe on FIT_FULL; frozen D1-30 runner for the selector rows).  Every line ends with its completion
marker ('<sbatch line> #MARK <path>'); the driver (slurm/r7_d1full_submit.sh) submits a line only when its marker is absent and resubmits a unit
that left the queue without it.  Positional wrapper arguments only (no --export); node51 / node52 / node60 always excluded.
  stage1 --lines L                 : the 6 learning-rate-search units (S1 vcs, S2 logistic, S3 direct × {1e-4, 3e-4} × seed 0), RTX6000PRO,H100 only
                                     (D1-20 execution note E5: real-data batches overflow an L40S; FIT_FULL keeps every feature on the GPU)
  choose --selectors-dir S         : the frozen D1-20 choice (r7_d1_plan20.choose: higher selected CAL Spearman, tie within 1e-4 -> 3e-4) -> lr_choice.json
  stage2 --selectors-dir S --lines L : the 12 dependent units of r7_d1_plan20.stage2 (seeds 1-2 × 3 objectives; A1 vcs --no-question and A2 direct
                                     --target d × seeds 0-2) at the chosen rates
  units --lr-choice L              : the 15 unit tags the evaluation uses
  eval-configs --lr-choice L --out C : configs_<bench>.txt with the SELECTOR rows only (sel:<unit>:<k> for the 15 units × 4 budgets, dyn:<unit>:<f> for
                                     the 9 S1-S3 units × 3 operating points = 87); full / static / SparseVLM / OCR rows are shared with D1-30 and not
                                     rerun.  timing_<bench>.txt = the 27 dynamic rows (their own latency; the fixed rows share D1-30's latency)
  eval-lines --root R --lines L [--no-devrisk] [--no-timing] [--devrisk-only] : quality shards with the D1-30 sharding (GQA 6, TextVQA 4, ChartQA 2; H100,L40S — no
                                     OCR-anchor rows, so the E2 reason for H100-only TextVQA does not apply), the D1-30 DEV risk secondary for the D1-FULL
                                     units (default on), and the timing of the D1-FULL dynamic rows with the frozen D1-30 timing procedure on the reference
                                     node, one unit per benchmark (coordinator 2026-10-12: required; the fixed-budget rows share D1-30's latency)
Output root (coordinator 2026-10-12): DF/d1_30 mirrors outputs/R7_D1/d1_30 — configs/, pred/<bench>/<cfg>@s<i>of<n>.jsonl rows {qid, image_id, pred,
kept} with done_shard<i>of<n>.json markers, timing/<bench>/timing_<bench>.json, devrisk/d1_30_devrisk.json — the frozen D1-30 format, kept as
the full-scale reference for the later official-test phase.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r7_d1_30_run as RUN  # noqa: E402   (frozen D1-30 configuration grammar)
import r7_d1_plan20 as P20  # noqa: E402   (frozen D1-20 unit set and choice rule)

DF = Path("/home/infres/yinwang/CS_QMI/outputs/R7_D1_FULL")
EXCL = "--exclude=node51,node52,node60"
SEL_GPU = f"--partition=RTX6000PRO,H100 {EXCL}"
LLAVA_GPU = f"--partition=H100,L40S {EXCL}"
REF_GPU = f"--partition=H100 --nodelist=nodesumo01 {EXCL}"
BENCHES = ("gqa_dev", "textvqa_val", "chartqa_val")
# time limits ≈ 3 × the expected runtime, for backfill (the coordinator's R7-40b practice); every unit resumes after a timeout
TIME = {"sel": "--time=08:00:00", "eval": "--time=05:00:00", "devrisk": "--time=02:00:00", "timing": "--time=05:00:00"}
SHARDS = {"gqa_dev": 6, "textvqa_val": 4, "chartqa_val": 2}                            # = r7_d1_30_plan.SHARDS


def sel_line(obj, lr, seed, extra="", sel_dir=DF / "selectors"):
    tag = P20.tag(obj, lr, seed, "--no-question" in extra, "--target d" in extra)
    cmd = f"sbatch --parsable --job-name=r7dFs_{tag} {SEL_GPU} {TIME['sel']} slurm/r7_d1full_sel.sbatch train {obj} {lr:g} {seed}" + (f" {extra}" if extra else "")
    return cmd, Path(sel_dir) / f"{tag}.complete.json"


def stage1(sel_dir=DF / "selectors"):
    return [sel_line(o, lr, 0, sel_dir=sel_dir) for o in P20.OBJECTIVES for lr in P20.LRS]


def stage2(choice, sel_dir=DF / "selectors"):
    L = [sel_line(o, choice[o]["chosen_lr"], s, sel_dir=sel_dir) for o in P20.OBJECTIVES for s in (1, 2)]
    L += [sel_line("vcs", choice["vcs"]["chosen_lr"], s, "--no-question", sel_dir) for s in (0, 1, 2)]
    L += [sel_line("direct", choice["direct"]["chosen_lr"], s, "--target d", sel_dir) for s in (0, 1, 2)]
    return L


def units(choice):
    main, abl = RUN.unit_names(choice)
    return [u for o in RUN.OBJECTIVES for u in main[o]] + [u for k in ("A1", "A2") for u in abl[k]]


def selector_configs(choice):
    main, abl = RUN.unit_names(choice)
    return [c for c in RUN.config_list([u for o in RUN.OBJECTIVES for u in main[o]], abl["A1"] + abl["A2"]) if c.startswith(("sel:", "dyn:"))]


def write_eval_configs(choice, out: Path):
    out.mkdir(parents=True, exist_ok=True); cfg = selector_configs(choice); dyn = [c for c in cfg if c.startswith("dyn:")]
    for b in BENCHES:
        (out / f"configs_{b}.txt").write_text("\n".join(cfg) + "\n"); (out / f"timing_{b}.txt").write_text("\n".join(dyn) + "\n")
    return {"configs_per_bench": len(cfg), "timing_configs_per_bench": len(dyn)}


def eval_lines(root: Path, devrisk=True, timing=True, devrisk_only=False):
    L = []
    if devrisk_only:
        return [(f"sbatch --parsable --job-name=r7dFr_devrisk {LLAVA_GPU} {TIME['devrisk']} slurm/r7_d1full_gpu.sbatch devrisk", root / "devrisk" / "d1_30_devrisk.json")]
    for b in BENCHES:
        for i in range(SHARDS[b]):
            L.append((f"sbatch --parsable --job-name=r7dFq_{b}_{i} {LLAVA_GPU} {TIME['eval']} slurm/r7_d1full_gpu.sbatch eval {b} {i}/{SHARDS[b]}",
                      root / "pred" / b / f"done_shard{i}of{SHARDS[b]}.json"))
    if devrisk:
        L.append((f"sbatch --parsable --job-name=r7dFr_devrisk {LLAVA_GPU} {TIME['devrisk']} slurm/r7_d1full_gpu.sbatch devrisk", root / "devrisk" / "d1_30_devrisk.json"))
    if timing:   # owner rule 2026-10-13: units pinned to one node run as ONE job (back-to-back, no other job can slip in between)
        L.append((f"sbatch --parsable --job-name=r7dFt_timing_all {REF_GPU} --time=10:00:00 slurm/r7_d1full_gpu.sbatch timing_all",
                  root / "timing" / "timing_bundle.done.json"))
    return L


def write_lines(path, pairs, title):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}: <sbatch line> #MARK <completion marker>\n" + "".join(f"{c} #MARK {m}\n" for c, m in pairs))
    print(f"[r7_d1full_plan] {len(pairs)} lines -> {path} (not submitted)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("stage1"); s1.add_argument("--lines", required=True); s1.add_argument("--selectors-dir", default=str(DF / "selectors"))
    c = sub.add_parser("choose"); c.add_argument("--selectors-dir", default=str(DF / "selectors"))
    s2 = sub.add_parser("stage2"); s2.add_argument("--selectors-dir", default=str(DF / "selectors")); s2.add_argument("--lines", required=True)
    u = sub.add_parser("units"); u.add_argument("--lr-choice", required=True)
    ec = sub.add_parser("eval-configs"); ec.add_argument("--lr-choice", required=True); ec.add_argument("--out", required=True)
    el = sub.add_parser("eval-lines"); el.add_argument("--root", default=str(DF / "d1_30")); el.add_argument("--lines", required=True)
    el.add_argument("--no-devrisk", action="store_true"); el.add_argument("--no-timing", action="store_true"); el.add_argument("--devrisk-only", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "stage1":
        write_lines(a.lines, stage1(Path(a.selectors_dir)), "R7-30 D1-FULL stage 1: learning-rate search, seed 0")
    elif a.cmd == "choose":
        ch = P20.choose(Path(a.selectors_dir)); ch = {o: dict(v, rule="frozen r7_d1_plan20.choose (D1-20 OPEN-6)") for o, v in ch.items()}
        p = Path(a.selectors_dir) / "lr_choice.json"; tmp = p.with_name(p.name + ".tmp"); tmp.write_text(json.dumps(ch, indent=1)); os.replace(tmp, p)
        print(json.dumps(ch, indent=1))
    elif a.cmd == "stage2":
        ch = json.loads((Path(a.selectors_dir) / "lr_choice.json").read_text())
        write_lines(a.lines, stage2(ch, Path(a.selectors_dir)), "R7-30 D1-FULL stage 2: seeds 1-2 at the chosen rate + ablations A1 / A2 × seeds 0-2")
    elif a.cmd == "units":
        print(" ".join(units(json.loads(Path(a.lr_choice).read_text()))))
    elif a.cmd == "eval-configs":
        print(json.dumps(write_eval_configs(json.loads(Path(a.lr_choice).read_text()), Path(a.out))))
    else:
        write_lines(a.lines, eval_lines(Path(a.root), not a.no_devrisk, not a.no_timing, a.devrisk_only),
                    "R7-30 D1-FULL evaluation (frozen D1-30 runner / timing / DEV risk, D1-FULL checkpoints)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
