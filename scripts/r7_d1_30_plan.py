"""r7 R7-30 D1-30 unit plan (after the D1-30 staged freeze and the D1-20 completion gate).
  configs --lr-choice L --out C : configs_<bench>.txt (the frozen configuration grid; TextVQA adds the OCR anchor) and timing_<bench>.txt
                                  (full [+ full_ocr], 12 static rows, the representative S1 seed-0 unit at 4 budgets, every dynamic row)
  units --lr-choice L           : the 15 D1-20 unit tags (S1-S3 × seeds 0-2 at the chosen rates; A1 / A2 × seeds 0-2)
  lines --root R --lines F      : sbatch lines — quality shards (GQA 6, TextVQA 4, ChartQA 2), SparseVLM shards (GQA 3, TextVQA 3, ChartQA 1; 8 rows, + 4 OCR anchors on TextVQA),
                                  timing per benchmark on nodesumo01 (our pipeline + the SparseVLM batch-1 rows in one job), the DEV risk
                                  calibration; each line carries its completion marker as a trailing comment (#MARK <path>)
LLaVA-env units: --partition=H100,L40S --exclude=node51,node52,node60 (no Blackwell kernels in that env), except the TextVQA quality units:
--partition=H100 (execution note E2: OCR-anchor prompts at batch 32 exceed L40S memory); timing: --partition=H100 --nodelist=nodesumo01.
Positional wrapper arguments only (no --export).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7_d1_30_run as RUN  # noqa: E402

BENCHES = ("gqa_dev", "textvqa_val", "chartqa_val")
SHARDS = {"gqa_dev": 6, "textvqa_val": 4, "chartqa_val": 2}
SV_SHARDS = {"gqa_dev": 3, "textvqa_val": 3, "chartqa_val": 1}
GPU = "--partition=H100,L40S --exclude=node51,node52,node60"
REF = "--partition=H100 --nodelist=nodesumo01"
# Execution note D1-30-E2: TextVQA's OCR-anchor prompts (912–1,055 tokens) at the frozen generation batch of 32 exceed an L40S (44.4 GB):
# OOM in the prefill (job 1035243; probe 1035261).  GQA / ChartQA / TextVQA without OCR fit in the forced worst case (32 longest prompts,
# full 576 tokens, every answer forced to max_new_tokens: peak allocated 38.6–39.2 GB of 44.4).  TextVQA quality units therefore run on H100
# only; batch size, batch composition and every other unit are unchanged.
QUALITY_GPU = {"gqa_dev": GPU, "textvqa_val": "--partition=H100 --exclude=node51,node52,node60", "chartqa_val": GPU}


def write_configs(lr_choice: dict, out: Path):
    main, abl = RUN.unit_names(lr_choice); mu = [u for o in RUN.OBJECTIVES for u in main[o]]; au = [u for k in ("A1", "A2") for u in abl[k]]; rep = main["vcs"][0]
    out.mkdir(parents=True, exist_ok=True); res = {}
    for b in BENCHES:
        cfg = RUN.config_list(mu, au, ocr_anchor=b.startswith("textvqa"))
        tim = ["full"] + (["full_ocr"] if b.startswith("textvqa") else []) + [c for c in cfg if c.startswith("static:")]
        tim += [f"sel:{rep}:{k}" for k in RUN.MK.BUDGETS] + [c for c in cfg if c.startswith("dyn:")]
        (out / f"configs_{b}.txt").write_text("\n".join(cfg) + "\n"); (out / f"timing_{b}.txt").write_text("\n".join(tim) + "\n")
        res[b] = {"configs": len(cfg), "timing_configs": len(tim)}
    return res


def lines(root: Path):
    L = []
    for b in BENCHES:
        for i in range(SHARDS[b]):
            L.append((f"sbatch --parsable --job-name=r7d30q_{b}_{i} {QUALITY_GPU[b]} slurm/r7_d1_30_gpu.sbatch run {b} {i}/{SHARDS[b]}",
                      root / "pred" / b / f"done_shard{i}of{SHARDS[b]}.json"))
        for i in range(SV_SHARDS[b]):
            L.append((f"sbatch --parsable --job-name=r7d30v_{b}_{i} {GPU} slurm/r7_d1_30_gpu.sbatch sparsevlm {b} {i}/{SV_SHARDS[b]}",
                      root / "pred" / b / f"sparsevlm_run_{b}_shard{i}of{SV_SHARDS[b]}.json"))
    for b in BENCHES:
        L.append((f"sbatch --parsable --job-name=r7d30t_{b} {REF} --exclude=node51,node52,node60 slurm/r7_d1_30_gpu.sbatch timing {b}",
                  root / "timing" / b / f"sparsevlm_timing_{b}.json"))
    L.append((f"sbatch --parsable --job-name=r7d30r_devrisk {GPU} slurm/r7_d1_30_gpu.sbatch devrisk", root / "devrisk" / "d1_30_devrisk.json"))
    return L


def main() -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("configs"); c.add_argument("--lr-choice", required=True); c.add_argument("--out", required=True)
    s = sub.add_parser("lines"); s.add_argument("--root", required=True); s.add_argument("--lines", required=True)
    u = sub.add_parser("units"); u.add_argument("--lr-choice", required=True)
    a = ap.parse_args()
    if a.cmd == "units":                                                               # the 15 D1-20 unit tags D1-30 uses (the gate waits for them)
        main_u, abl_u = RUN.unit_names(json.loads(Path(a.lr_choice).read_text()))
        print(" ".join(u for v in list(main_u.values()) + list(abl_u.values()) for u in v)); return 0
    if a.cmd == "configs":
        print(json.dumps(write_configs(json.loads(Path(a.lr_choice).read_text()), Path(a.out)))); return 0
    with open(a.lines, "w") as fh:
        fh.write("# R7-30 D1-30 units: <sbatch line> #MARK <completion marker>\n")
        for line, mark in lines(Path(a.root)):
            fh.write(f"{line} #MARK {mark}\n")
    print(f"{len(lines(Path(a.root)))} units -> {a.lines}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
