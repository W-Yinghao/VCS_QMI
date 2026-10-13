"""r7 R7-40c — Qwen2.5-VL-7B evaluation stage of the R7-40 cross-host confirmation (the frozen D1-30 design on the second host): the
configuration grammar, the configuration lists and the unit plan.  No torch import (the scorer and the driver's planner calls use it).
Budgets are fractions f of each image's OWN merged-token count N (r7_d1_40_masks.budget: k = max(1, floor(f·N + 1/2)); f = 1 is the full pass).
Configurations (one prediction file per configuration and shard):
  full                    every merged token (the official path's input)
  static:grid:<f>         the D1-10 uniform-grid mechanism on the image's merged grid (r7_d1_40_masks.order_grid, qid-seeded), f in {0.125, 0.25,
                          0.5, 0.75} (0.125 = stress point)
  static:random:<f>, static:qsim:<f>      extended rows (OPEN-R740C-1; the R7-40 plan's training-free controls)
  sel:<unit>:<f>          hard top-k of a frozen R7-40b unit's gate logits over the image's valid tokens (S1 vcs / S2 logistic / S3 direct ×
                          seeds 0-2 at the CAL-chosen rates of selectors/lr_choice.json)
  dyn:<unit>:<op>         dynamic budget: per item the smallest f in {0.125, 0.25, 0.5, 0.75, 1} whose top-k mask has predicted risk r <= tau_op,
                          tau_op fixed on CAL by R7-40b for the mean kept fraction op in {0.25, 0.5, 0.75} (r7_d1_40_selectors.dynamic_tau40,
                          nominal fractions, D40b-5); S1 units (core), S2 / S3 units extended (OPEN-R740C-1)
Row sets: core = 1 + 4 + 36 + 9 = 50 per benchmark (the coordinator's 40c task list); +static = 58 (adds random / qsim); +dynamic = 68 (adds
S2 / S3 dynamic); all = 76 (the R7-40 protocol outline).  D1-30 has no text-only row, so none here; no OCR-anchor row (Qwen's published
TextVQA number uses no OCR tokens; OPEN-R740-7).
Timing configurations (D1-30 rule): full, the static rows, the representative unit (S1 seed 0) at the four fractions, every dynamic row.
Unit lines carry '#MARK <completion marker>' and optionally '#GATE <file> [<file> ...]' (submitted only once every gate file exists); the
driver slurm/r7_d1_40c_submit.sh resubmits a unit only after its job left the queue without the marker.  Positional wrapper arguments only.
    python -I scripts/r7_d1_40c_plan.py configs --lr-choice L --out C [--rows core|static|dynamic|all]
    python -I scripts/r7_d1_40c_plan.py units --lr-choice L            (the 12 R7-40b markers the driver's gate 1 waits for)
    python -I scripts/r7_d1_40c_plan.py lines --root R --lines F [--quality-partition P] [--bstar report|in_bundle] [--no-dev] [--no-timing] [--no-official]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7_d1_40_masks as M4  # noqa: E402  (frozen R7-40a: fractions, budget rule)
import r7_d1_plan20 as P20  # noqa: E402  (frozen D1-20: unit tags, learning-rate grid)

D1 = Path("/home/infres/yinwang/CS_QMI/outputs/R7_D1")
R40 = D1 / "r7_40"
R40C = R40 / "eval_40c"
D30_SETS = D1 / "d1_30" / "sets"
BENCHES = ("gqa_dev", "textvqa_val", "chartqa_val")
FRACTIONS = M4.FRACTIONS                                    # (0.125, 0.25, 0.5, 0.75)
FKEYS = M4.FKEYS                                            # ("0.125", "0.25", "0.5", "0.75")
DYN_GRID = tuple(FRACTIONS) + (1.0,)                        # the dynamic rule's fraction grid (r7_d1_40_selectors.risk_grid)
OPS = ("0.25", "0.5", "0.75")                               # CAL operating points (mean kept fraction)
OBJECTIVES = P20.OBJECTIVES                                 # ("vcs", "logistic", "direct")
NAMES = {"vcs": "S1", "logistic": "S2", "direct": "S3"}
SEEDS = (0, 1, 2)
ROW_SETS = ("core", "static", "dynamic", "all")
MAX_NEW = {"gqa_dev": 128, "textvqa_val": 128, "chartqa_val": 16,                 # D30-7 (frozen D1-30 generation lengths)
           "gqa_testdev": 128, "chartqa_test": 16,                                # R7-50 phase Q (same values as the mapped validation sets)
           "gqa_fixture": 128, "textvqa_fixture": 128, "chartqa_proxy": 16, "chartqa_train_smoke": 16}
EXCL = "--exclude=node51,node60"                            # node52 re-admitted after a passing health probe (commit 5a35e67)
ALLOWED_PARTITIONS = {"RTX6000PRO", "H100", "L40S"}
REF = "--partition=H100 --nodelist=nodesumo01"              # reference GPU (NVIDIA H100 NVL) for every latency number
QUALITY_SHARDS = {"gqa_dev": 6, "textvqa_val": 4, "chartqa_val": 2}             # as D1-30
OFFICIAL_SHARDS = {"gqa_dev": 1, "textvqa_val": 1, "chartqa_val": 1}            # the batch-1 official full-row anchor (decision 15): one unit per benchmark
TIME = {"gqa_dev": "01:30:00", "textvqa_val": "03:00:00", "chartqa_val": "02:30:00", "timing_bundle": "08:00:00",   # ≈ 2 × the runtime at the
        "official_gqa_dev": "01:00:00", "official_textvqa_val": "01:15:00", "official_chartqa_val": "00:45:00",       # L40S rates measured in
        "devteacher": "01:00:00", "devrisk": "01:00:00"}                                                             # job 1036576 (40c draft)
DEV_BUNDLES = ("b012", "b013")                              # the frozen D1 shard plan's DEV bundles (sealed until R7-40c, D40a-5)
TIMING_REPORT = R40 / "timing" / "timing_report.json"       # R7-40a reference timing report (B*, D40a-7): job 1035608 or the 40c timing bundle


def fkey(f) -> str:
    return f"{float(f):g}"


def unit_tag(obj, lr, seed) -> str:
    return P20.tag(obj, float(lr), int(seed))


def unit_names(lr_choice: dict):
    """{objective: [unit tag of seed 0, 1, 2]} at each objective's CAL-chosen rate (R7-40b lr_choice.json)."""
    return {o: [unit_tag(o, lr_choice[o]["chosen_lr"], s) for s in SEEDS] for o in OBJECTIVES}


def r7_40b_tags(lr_choice: dict):
    """The 12 R7-40b training units: 6 seed-0 learning-rate-search units + 6 seed-1 / seed-2 units at the chosen rates."""
    s1 = [unit_tag(o, lr, 0) for o in OBJECTIVES for lr in P20.LRS]
    s2 = [unit_tag(o, lr_choice[o]["chosen_lr"], s) for o in OBJECTIVES for s in (1, 2)]
    return s1 + s2


def config_list(main: dict, rows: str = "core"):
    """Configurations of one benchmark.  main: {objective: [3 unit tags]}."""
    if rows not in ROW_SETS:
        raise SystemExit(f"unknown row set {rows!r} ({', '.join(ROW_SETS)})")
    mechs = ("grid", "random", "qsim") if rows in ("static", "all") else ("grid",)
    dyn_obj = OBJECTIVES if rows in ("dynamic", "all") else ("vcs",)
    cfg = ["full"] + [f"static:{m}:{f}" for m in mechs for f in FKEYS]
    cfg += [f"sel:{u}:{f}" for o in OBJECTIVES for u in main[o] for f in FKEYS]
    cfg += [f"dyn:{u}:{op}" for o in dyn_obj for u in main[o] for op in OPS]
    return cfg


def timing_list(cfg, rep_unit: str):
    return ["full"] + [c for c in cfg if c.startswith("static:")] + [f"sel:{rep_unit}:{f}" for f in FKEYS] + [c for c in cfg if c.startswith("dyn:")]


def parse(cfg: str):
    """'full' -> ('full', None, None); 'static:grid:0.25' -> ('static', 'grid', 0.25); 'sel:<unit>:0.5' / 'dyn:<unit>:0.75' likewise."""
    if cfg == "full":
        return "full", None, None
    kind, a, b = cfg.split(":")
    if kind not in ("static", "sel", "dyn"):
        raise ValueError(f"unknown configuration {cfg!r}")
    f = float(b)
    if (kind in ("static", "sel") and fkey(f) not in FKEYS) or (kind == "dyn" and fkey(f) not in OPS):
        raise ValueError(f"fraction {b} not on the frozen grid for {cfg!r}")
    if kind == "static" and a not in ("grid", "random", "qsim"):
        raise ValueError(f"unknown mechanism in {cfg!r}")
    return kind, a, f


def objective_of(unit: str) -> str:
    return unit.split("_lr")[0]


def write_configs(lr_choice: dict, out: Path, rows: str = "core", benches=BENCHES):
    main = unit_names(lr_choice); rep = main["vcs"][0]; out.mkdir(parents=True, exist_ok=True); res = {"rows": rows, "representative_unit": rep}
    for b in benches:
        cfg = config_list(main, rows); tim = timing_list(cfg, rep)
        (out / f"configs_{b}.txt").write_text("\n".join(cfg) + "\n"); (out / f"timing_{b}.txt").write_text("\n".join(tim) + "\n")
        res[b] = {"configs": len(cfg), "timing_configs": len(tim)}
    return res


def read_configs(path) -> list:
    return [c.strip() for c in Path(path).read_text().splitlines() if c.strip() and not c.startswith("#")]


# ------------------------------------------------------------------------------------------------------------------------------ unit lines
def gpu(partition: str) -> str:
    if not partition or any(c in partition for c in " =;'\"$`") or set(partition.split(",")) - ALLOWED_PARTITIONS:
        raise SystemExit(f"bad partition list {partition!r} (allowed: {', '.join(sorted(ALLOWED_PARTITIONS))})")
    return f"--partition={partition} {EXCL}"


def lines(root: Path, quality_partition="L40S", dev=True, timing=True, r40: Path = R40, timing_report: Path = TIMING_REPORT,
          dev_partition="L40S", shards=None, bstar="report", official=True):
    """(sbatch line, marker, gate files) of every R7-40c GPU unit:
      quality  per benchmark i/n shards of the frozen D1-30 sets (all configurations of the benchmark in one job: paired rows share the GPU)
      timing   ONE bundle job on the reference node (owner rule 2026-10-13): Qwen's B* (bstar 'in_bundle': the frozen R7-40a procedure runs
               first unless its report exists; 'report': the unit waits for the R7-40a report), then gqa_dev, textvqa_val, chartqa_val
               back-to-back; one marker; retried as a whole
      official the batch-1 official full-row anchor (decision 15): the full row through the official HF generate at batch 1, one unit per
               benchmark on the quality partition (secondary diagnostic next to the batched full row)
      DEV      the two sealed DEV teacher bundles with the frozen R7-40a code (--allow-dev; the 40a GPU class, D40a-4), then the DEV risk
               calibration secondary, gated on both bundle markers."""
    if bstar not in ("in_bundle", "report"):
        raise SystemExit(f"bad B* mode {bstar!r} (in_bundle | report)")
    shards = shards or QUALITY_SHARDS; L = []
    for b in BENCHES:
        for i in range(shards[b]):
            L.append((f"sbatch --parsable --job-name=r7d40cq_{b}_{i} {gpu(quality_partition)} --time={TIME[b]} slurm/r7_d1_40c_gpu.sbatch run {b} {i}/{shards[b]}",
                      root / "pred" / b / f"done_shard{i}of{shards[b]}.json", []))
    if official:
        for b in BENCHES:
            n = OFFICIAL_SHARDS[b]
            for i in range(n):
                L.append((f"sbatch --parsable --job-name=r7d40co_{b}_{i} {gpu(quality_partition)} --time={TIME['official_' + b]} slurm/r7_d1_40c_gpu.sbatch official {b} {i}/{n}",
                          root / "pred" / b / f"done_official_shard{i}of{n}.json", []))
    if timing:
        L.append((f"sbatch --parsable --job-name=r7d40ct_bundle {REF} {EXCL} --time={TIME['timing_bundle']} slurm/r7_d1_40c_gpu.sbatch timing_bundle {bstar}",
                  root / "timing" / "timing_bundle.done.json", [] if bstar == "in_bundle" else [timing_report]))
    if dev:
        marks = []
        for bd in DEV_BUNDLES:
            m = r40 / "teacher" / f"bundle_{bd}.done.json"; marks.append(m)
            L.append((f"sbatch --parsable --job-name=r7d40cd_{bd} {gpu(dev_partition)} --time={TIME['devteacher']} slurm/r7_d1_40c_gpu.sbatch devteacher {bd}", m, []))
        L.append((f"sbatch --parsable --job-name=r7d40c_devrisk {gpu(dev_partition)} --time={TIME['devrisk']} slurm/r7_d1_40c_gpu.sbatch devrisk",
                  root / "devrisk" / "r7_40c_devrisk.json", marks))
    return L


def write_lines(units, path: Path, title: str):
    with open(path, "w") as fh:
        fh.write(f"# {title}: <sbatch line> #MARK <completion marker> [#GATE <file> ...]\n")
        for line, mark, gates in units:
            fh.write(f"{line} #MARK {mark}" + (f" #GATE {' '.join(str(g) for g in gates)}" if gates else "") + "\n")
    print(f"[r7_d1_40c_plan] {len(units)} lines -> {path} (not submitted)")


def main() -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("configs"); c.add_argument("--lr-choice", required=True); c.add_argument("--out", required=True)
    c.add_argument("--rows", choices=ROW_SETS, default="core")
    u = sub.add_parser("units"); u.add_argument("--lr-choice", required=True); u.add_argument("--used", action="store_true")
    s = sub.add_parser("lines"); s.add_argument("--root", default=str(R40C)); s.add_argument("--lines", required=True)
    s.add_argument("--quality-partition", default="L40S"); s.add_argument("--no-dev", action="store_true"); s.add_argument("--no-timing", action="store_true")
    s.add_argument("--r40", default=str(R40)); s.add_argument("--timing-report", default=str(TIMING_REPORT)); s.add_argument("--bstar", choices=["in_bundle", "report"], default="report")
    s.add_argument("--no-official", action="store_true")
    a = ap.parse_args()
    if a.cmd == "configs":
        print(json.dumps(write_configs(json.loads(Path(a.lr_choice).read_text()), Path(a.out), a.rows))); return 0
    if a.cmd == "units":
        ch = json.loads(Path(a.lr_choice).read_text())
        print(" ".join([u for o in OBJECTIVES for u in unit_names(ch)[o]] if a.used else r7_40b_tags(ch))); return 0
    write_lines(lines(Path(a.root), a.quality_partition, not a.no_dev, not a.no_timing, Path(a.r40), Path(a.timing_report), bstar=a.bstar, official=not a.no_official),
                Path(a.lines), "R7-40c units (quality shards, batch-1 official anchors, ONE timing bundle on nodesumo01, DEV teacher + DEV risk)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
