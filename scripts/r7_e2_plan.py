"""r7 E2 — the Qwen2.5-VL-7B analogue of D1-FULL (reports/R7/R7_E2_QWEN_FULL_PROTOCOL_DRAFT_20261013.yaml): the frozen R7-40 pipeline — R7-40a
teacher (scripts/r7_d1_40_teacher.py), R7-40b feature cache and selectors with the frozen CAL learning-rate rule (scripts/r7_d1_40_cache.py,
r7_d1_40_train.py, r7_d1_plan20.choose), R7-40c evaluation (scripts/r7_d1_40c_run.py) — on D1-FULL's frozen FIT_FULL (70,256 images, 280,787
questions).  The frozen scripts run unchanged through the E2 wrappers (slurm/r7_e2_{gpu,sel,cpu}.sbatch); this module only writes unit lines.
Every line ends with its completion marker ('<sbatch line> #MARK <path>'); the driver (slurm/r7_e2_submit.sh) submits a line only when its marker
is absent and resubmits a unit only after its job has left the queue without it.  Positional wrapper arguments only (no --export); node51 /
node60 always excluded.  No torch import.
  teacher-lines --data-dir D --lines L            : one line per bundle of D/d1_shards.json (FIT shards only; CAL is reused, DEV never computed);
                                                    L40S only (D40a-4: one GPU class for every Qwen teacher record)
  cache-lines --data-dir D --lines L [--basis B]  : the frozen R7-40b cache job (r7_d1_40_cache.py d1_40) on E2's manifest dir; B = fitfull (PCA on
                                                    every FIT_FULL image, the frozen 'fit on FIT' rule with FIT = FIT_FULL) | frozen10k (PCA on the
                                                    first 10,000 FIT images = the frozen FIT, i.e. R7-40b's image set)
  stage1 --lines L                                : 6 seed-0 units (S1 vcs / S2 logistic / S3 direct × lr {1e-4, 3e-4}), RTX6000PRO,H100 only (40b_2 / 40b_3)
  choose --selectors-dir S                        : the frozen choice r7_d1_plan20.choose (higher selected CAL Spearman; tie within 1e-4 -> 3e-4)
  stage2 --selectors-dir S --lines L              : seeds 1, 2 × 3 objectives at the chosen rates (no ablations on Qwen: R7-40 scope rule)
  units --lr-choice L                             : the 12 unit tags (the driver's gate before the evaluation)
  run-configs --configs-dir C --out O [--rows R]  : the rows E2's quality units generate, from the R7-40c core list C/configs_<bench>.txt (written by
                                                    the frozen R7-40c stage-0 check with E2's lr_choice): R = selectors -> 'full' (in-job anchor) + the
                                                    45 selector rows; core -> all 50 core rows
  eval-lines --lines L                            : 12 quality shards with the D1-30 / R7-40c sharding (GQA 6, TextVQA 4, ChartQA 2), L40S only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r7_d1_40c_plan as PL  # noqa: E402  (frozen R7-40c: configuration grammar, benchmarks, sharding)
import r7_d1_plan20 as P20  # noqa: E402  (frozen D1-20: unit tags, learning-rate grid, choice rule)

E2 = Path("/home/infres/yinwang/CS_QMI/outputs/R7_E2")
EXCL = "--exclude=node51,node60"
ALLOWED_PARTITIONS = {"RTX6000PRO", "H100", "L40S"}
TEACHER_PARTITION = "L40S"                       # D40a-4: every R7-40a teacher record (FIT, CAL, DEV) came from an L40S
CACHE_PARTITION = "RTX6000PRO,H100,L40S"         # 40b_1 (the cache's GPU footprint is ~17 GB; host RAM ~50 GB at FIT_FULL scale)
SEL_PARTITION = "RTX6000PRO,H100"                # 40b_2 / 40b_3: the direct objective overflows an L40S already at 10k
QUALITY_PARTITION = "L40S"                       # R7-40c decision 7 (and the condition for reusing R7-40c's selector-independent rows)
# time limits ≈ 3 × the expected runtime (backfill; every unit resumes, except the one-pass cache job, whose limit is ≈ 3.3 ×)
TIME = {"bundle": "02:00:00", "cache": "10:00:00", "sel": "04:30:00", "gqa_dev": "01:40:00", "textvqa_val": "03:45:00", "chartqa_val": "02:10:00"}
BASES = ("fitfull", "frozen10k")
ROWS = ("selectors", "core")
FROZEN_FIT_IMAGES = 10000                         # the frozen FIT = the first 10,000 FIT_FULL images (byte-prefix manifest)
CORE_ROWS = 50                                    # R7-40c core: full + 4 grid + 36 sel + 9 dyn


def gpu(partition: str) -> str:
    if not partition or any(c in partition for c in " =;'\"$`") or set(partition.split(",")) - ALLOWED_PARTITIONS:
        raise SystemExit(f"bad partition list {partition!r} (allowed: {', '.join(sorted(ALLOWED_PARTITIONS))})")
    return f"--partition={partition} {EXCL}"


def teacher_lines(plan: dict, root: Path = E2, partition: str = TEACHER_PARTITION):
    bad = [s for v in plan["bundles"].values() for s in v if not s.startswith("FIT-")]
    if bad:
        raise SystemExit(f"E2 teacher bundles may hold FIT shards only (CAL is reused, DEV is never computed): {bad[:3]}")
    return [(f"sbatch --parsable --job-name=r7e2t_{b} {gpu(partition)} --time={TIME['bundle']} slurm/r7_e2_gpu.sbatch bundle {b}",
             Path(root) / "teacher" / f"bundle_{b}.done.json") for b in plan["bundles"]]


def pca_images(meta: dict, basis: str) -> int:
    """Images the frozen cache fits its PCA on (--max-images): every FIT_FULL image (fitfull) or the frozen FIT (frozen10k)."""
    if basis not in BASES:
        raise SystemExit(f"PCA basis must be one of {BASES}, got {basis!r}")
    return int(meta["counts"]["FIT"]["images"]) if basis == "fitfull" else FROZEN_FIT_IMAGES


def cache_lines(meta: dict, root: Path = E2, basis: str = "fitfull", partition: str = CACHE_PARTITION):
    return [(f"sbatch --parsable --job-name=r7e2c_cache {gpu(partition)} --time={TIME['cache']} slurm/r7_e2_gpu.sbatch cache {pca_images(meta, basis)}",
             Path(root) / "cache" / "cache_d1_40.json")]


def sel_line(obj, lr, seed, root: Path = E2, partition: str = SEL_PARTITION):
    t = P20.tag(obj, float(lr), int(seed))
    return (f"sbatch --parsable --job-name=r7e2s_{t} {gpu(partition)} --time={TIME['sel']} slurm/r7_e2_sel.sbatch train {obj} {float(lr):g} {int(seed)}",
            Path(root) / "selectors" / f"{t}.done.json")


def stage1(root: Path = E2, partition: str = SEL_PARTITION):
    return [sel_line(o, lr, 0, root, partition) for o in P20.OBJECTIVES for lr in P20.LRS]


def stage2(choice: dict, root: Path = E2, partition: str = SEL_PARTITION):
    return [sel_line(o, choice[o]["chosen_lr"], s, root, partition) for o in P20.OBJECTIVES for s in (1, 2)]


def unit_tags(choice: dict):
    return [P20.tag(o, lr, 0) for o in P20.OBJECTIVES for lr in P20.LRS] + [P20.tag(o, float(choice[o]["chosen_lr"]), s) for o in P20.OBJECTIVES for s in (1, 2)]


def choose(sel_dir: Path):
    """The frozen D1-20 / R7-40b choice, re-run on E2's seed-0 units; atomic write of lr_choice.json."""
    ch = P20.choose(Path(sel_dir)); ch = {o: dict(v, rule="frozen r7_d1_plan20.choose (D1-20 OPEN-6; R7-40b OPEN-R740-3)") for o, v in ch.items()}
    p = Path(sel_dir) / "lr_choice.json"; tmp = p.with_name(p.name + ".tmp"); tmp.write_text(json.dumps(ch, indent=1)); os.replace(tmp, p)
    return ch


def run_configs(core: list, rows: str) -> list:
    """Rows E2's quality units generate.  core: the R7-40c core list with E2's unit tags (50 rows).  'selectors': the in-job full row (anchor
    for the reuse check of R7-40c's selector-independent rows) + the 36 sel and 9 dyn rows; 'core': all 50 rows."""
    if rows not in ROWS:
        raise SystemExit(f"rows must be one of {ROWS}, got {rows!r}")
    kinds = [PL.parse(c)[0] for c in core]
    if len(core) != CORE_ROWS or len(set(core)) != CORE_ROWS or core[0] != "full" or kinds.count("static") != 4 or kinds.count("sel") != 36 or kinds.count("dyn") != 9:
        raise SystemExit(f"not the R7-40c core list ({len(core)} rows: {dict((k, kinds.count(k)) for k in set(kinds))})")
    return list(core) if rows == "core" else ["full"] + [c for c in core if c.startswith(("sel:", "dyn:"))]


def write_run_configs(configs_dir: Path, out: Path, rows: str, benches=PL.BENCHES):
    """Run lists per benchmark; refuses to change an existing plan (other ROWS or other rows: predictions already on disk would not match)."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True); res = {"rows": rows, "source": str(configs_dir)}; prev = out / "run_rows.json"
    if prev.exists() and json.loads(prev.read_text()).get("rows") != rows:
        raise SystemExit(f"{out} was planned with rows={json.loads(prev.read_text()).get('rows')}: another ROWS needs a fresh eval dir (coordinator)")
    for b in benches:
        run = run_configs(PL.read_configs(Path(configs_dir) / f"configs_{b}.txt"), rows); f = out / f"configs_{b}.txt"; text = "\n".join(run) + "\n"
        if f.exists() and f.read_text() != text:
            raise SystemExit(f"{f} exists with other rows (lr_choice changed?): STOP (coordinator)")
        f.write_text(text); res[b] = len(run)
    prev.write_text(json.dumps(res, indent=1))
    return res


def eval_lines(root: Path = E2, partition: str = QUALITY_PARTITION):
    L = []
    for b in PL.BENCHES:
        n = PL.QUALITY_SHARDS[b]
        for i in range(n):
            L.append((f"sbatch --parsable --job-name=r7e2q_{b}_{i} {gpu(partition)} --time={TIME[b]} slurm/r7_e2_gpu.sbatch run {b} {i}/{n}",
                      Path(root) / "eval" / "pred" / b / f"done_shard{i}of{n}.json"))
    return L


def write_lines(path, pairs, title):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}: <sbatch line> #MARK <completion marker>\n" + "".join(f"{c} #MARK {m}\n" for c, m in pairs))
    print(f"[r7_e2_plan] {len(pairs)} lines -> {path} (not submitted)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=str(E2)); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("teacher-lines"); t.add_argument("--data-dir", required=True); t.add_argument("--lines", required=True)
    c = sub.add_parser("cache-lines"); c.add_argument("--data-dir", required=True); c.add_argument("--lines", required=True)
    c.add_argument("--basis", choices=BASES, default="fitfull"); c.add_argument("--partition", default=CACHE_PARTITION)
    s1 = sub.add_parser("stage1"); s1.add_argument("--lines", required=True)
    ch = sub.add_parser("choose"); ch.add_argument("--selectors-dir", required=True)
    s2 = sub.add_parser("stage2"); s2.add_argument("--selectors-dir", required=True); s2.add_argument("--lines", required=True)
    u = sub.add_parser("units"); u.add_argument("--lr-choice", required=True)
    rc = sub.add_parser("run-configs"); rc.add_argument("--configs-dir", required=True); rc.add_argument("--out", required=True)
    rc.add_argument("--rows", choices=ROWS, default="selectors")
    el = sub.add_parser("eval-lines"); el.add_argument("--lines", required=True)
    a = ap.parse_args(argv); root = Path(a.root)
    if a.cmd == "teacher-lines":
        write_lines(a.lines, teacher_lines(json.loads((Path(a.data_dir) / "d1_shards.json").read_text()), root), "r7 E2 teacher bundles (frozen R7-40a teacher, L40S)")
    elif a.cmd == "cache-lines":
        meta = json.loads((Path(a.data_dir) / "d1_manifest_meta.json").read_text())
        write_lines(a.lines, cache_lines(meta, root, a.basis, a.partition), f"r7 E2 feature cache (frozen R7-40b cache script, PCA basis {a.basis})")
    elif a.cmd == "stage1":
        write_lines(a.lines, stage1(root), "r7 E2 stage 1: learning-rate search, seed 0 (frozen R7-40b trainer)")
    elif a.cmd == "choose":
        print(json.dumps(choose(Path(a.selectors_dir)), indent=1))
    elif a.cmd == "stage2":
        write_lines(a.lines, stage2(json.loads((Path(a.selectors_dir) / "lr_choice.json").read_text()), root), "r7 E2 stage 2: seeds 1-2 at the chosen rates")
    elif a.cmd == "units":
        print(" ".join(unit_tags(json.loads(Path(a.lr_choice).read_text()))))
    elif a.cmd == "run-configs":
        print(json.dumps(write_run_configs(Path(a.configs_dir), Path(a.out), a.rows)))
    else:
        write_lines(a.lines, eval_lines(root), "r7 E2 evaluation: quality shards (frozen R7-40c runner, E2 selectors / PCA, L40S)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
