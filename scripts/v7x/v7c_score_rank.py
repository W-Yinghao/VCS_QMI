"""V7C-04: run the FROZEN GenAI-Rank scorer (scripts/v7/v7_score.py rank --allow-scoring) on the frozen predictions of V7X-50 v2, one call per
pre-declared family.  This wrapper only selects files and filters prompt sets; every statistic is computed by the frozen scorer.

    python -I scripts/v7x/v7c_score_rank.py --freeze /projects/EEG-foundation-model/yinghao/outputs_v7/x50_rank_freeze_v2 \
        --manifest reports/V7/V7X_50_RANK_PREDICTION_FREEZE_v2.json --labels <sealed labels> --out <new dir> \
        --request reports/V7_CLOSEOUT/V7C_04_SCORING_REQUEST.md --ledger reports/V7_CLOSEOUT/LABEL_ACCESS_LEDGER.csv

Families (amendment v2 + V7X-50 v2):
  primary      --ours M2_E --comparators B0_FGA_released_O B1_VQAScore B4T_TIFA B4D_DSG          (8 contrasts, Holm)
  attribution  --ours M2_E --comparators M1_E                                                     (2)
  v1           --ours M2_E --comparators M1_E B0_FGA_released_O B1_VQAScore                       (6, 'v1-registered')
  x21          the X21 rows on their parsed prompts (labels and predictions restricted to the same prompts), --ours X21_N-VCS-F_E vs the other X21 rows
  subset       primary + attribution without the prompts where any of their methods has a flagged failure (V7X-50 v2 sensitivity)
Every call gets the methods it needs (all full-universe methods for the per-method tables); outputs are never overwritten (the scorer refuses).
The wrapper refuses to start unless the request file and a ledger row exist and the label file's sha256 equals the sealed hash in the manifest
record (it never prints a label).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

SCORER = Path(__file__).resolve().parents[1] / "v7" / "v7_score.py"
SEALED_SHA256 = "0eb955b45429f5c240c8aed5ad9686db905695c2a74c40c67682c87b506a20dc"       # outputs/V7/data/genai_rank_seal_report.json
FAMILIES = {"primary": ("M2_E", ["B0_FGA_released_O", "B1_VQAScore", "B4T_TIFA", "B4D_DSG"]), "attribution": ("M2_E", ["M1_E"]),
            "v1": ("M2_E", ["M1_E", "B0_FGA_released_O", "B1_VQAScore"])}
X21_MAIN = ["X21_N-VCS-F_E", "X21_N-BCE-F_E", "X21_N-DV_E", "X21_N-NWJ_E", "X21_K-KDE-S_E", "X21_K-NYSTROM_E", "X21_K-RULSIF_E"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def filter_jsonl(src, dst, drop_prompts: set):
    with open(src) as f, open(dst, "w") as g:
        for line in f:
            if json.loads(line)["prompt_id"] not in drop_prompts:
                g.write(line)


def run(methods: dict, labels, out_json: Path, ours, comps, workdir: Path):
    mj = workdir / f"methods_{out_json.stem}.json"; mj.write_text(json.dumps(methods, indent=1))
    cmd = [sys.executable, "-I", str(SCORER), "rank", "--methods", str(mj), "--labels", str(labels), "--out", str(out_json), "--ours", ours, "--comparators", *comps, "--allow-scoring"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    (workdir / f"{out_json.stem}.log").write_text(r.stdout + "\n" + r.stderr)
    if r.returncode != 0:
        raise SystemExit(f"frozen scorer failed for {out_json.name}: {r.stderr[-800:]}")
    return out_json


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--freeze", required=True); ap.add_argument("--manifest", required=True); ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--request", required=True); ap.add_argument("--ledger", required=True)
    ap.add_argument("--synthetic-test", action="store_true", help="code check on synthetic labels: skips the sealed-hash check, never on real labels")
    a = ap.parse_args()
    if not Path(a.request).is_file():
        raise SystemExit("refusing: the V7C-04 scoring request must be written (and committed) before any label is read")
    rows = list(csv.DictReader(open(a.ledger, encoding="utf-8-sig"))) if Path(a.ledger).is_file() else []
    if not a.synthetic_test and not any(r.get("dataset", "").startswith("GenAI-Rank") for r in rows):
        raise SystemExit("refusing: no GenAI-Rank row in the label-access ledger")
    if not a.synthetic_test and sha(a.labels) != SEALED_SHA256:
        raise SystemExit("refusing: the label file is not the sealed GenAI-Rank label file")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False); work = out / "work"; work.mkdir()
    man = json.loads(Path(a.manifest).read_text()); methods = json.loads((Path(a.freeze) / "methods.json").read_text())
    if sha(Path(a.freeze) / "methods.json") != man["methods_json"]["sha256"]:
        raise SystemExit("refusing: methods.json differs from the frozen manifest")
    for name, m in man["methods"].items():                                       # every converted file is the frozen one
        for f in m["files"]:
            if sha(f["file"]) != f["sha256"]:
                raise SystemExit(f"refusing: {f['file']} differs from the frozen manifest")
    full = {k: v for k, v in methods.items() if k not in X21_MAIN}; done = {}
    for fam, (ours, comps) in FAMILIES.items():
        done[fam] = str(run(full, a.labels, out / f"{fam}.json", ours, comps, work))
    # X21 rows on their parsed prompts
    failed = {p for k in X21_MAIN for f in man["methods"][k]["files"] for _, p in f["validation"]["flagged_failure_pools"]}
    lab_x = work / "labels_x21_parsed.jsonl"; filter_jsonl(a.labels, lab_x, failed); lab_x.chmod(0o400)
    x21 = {k: v for k, v in methods.items() if k in X21_MAIN}
    done["x21"] = str(run(x21, lab_x, out / "x21.json", "X21_N-VCS-F_E", [k for k in X21_MAIN if k != "X21_N-VCS-F_E"], work))
    # common-defined subset of the primary and attribution families
    drop = {p for _, p in man["sensitivity_common_defined_subset"]["excluded_pools"]}; sub = {}
    for k in ["M2_E", "M1_E", "B0_FGA_released_O", "B1_VQAScore", "B4T_TIFA", "B4D_DSG"]:
        sub[k] = []
        for f in methods[k]:
            dst = work / f"subset_{Path(f).name}"; filter_jsonl(f, dst, drop); sub[k].append(str(dst))
    lab_s = work / "labels_subset.jsonl"; filter_jsonl(a.labels, lab_s, drop); lab_s.chmod(0o400)
    done["subset_primary"] = str(run(sub, lab_s, out / "subset_primary.json", "M2_E", FAMILIES["primary"][1], work))
    done["subset_attribution"] = str(run(sub, lab_s, out / "subset_attribution.json", "M2_E", ["M1_E"], work))
    for p in (lab_x, lab_s):                                                       # filtered label copies are not kept
        p.unlink()
    (out / "runs.json").write_text(json.dumps({"families": done, "n_x21_excluded_prompts": len(failed), "n_subset_excluded_prompts": len(drop),
                                               "freeze_methods_json_sha256": man["methods_json"]["sha256"], "scorer": str(SCORER), "scorer_sha256": sha(SCORER),
                                               "synthetic_test": a.synthetic_test}, indent=1))
    print(json.dumps(done, indent=1)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
