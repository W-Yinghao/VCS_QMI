"""V7C-03 source fidelity, step 1 (label-free): fixed prompt sets for comparing the local question generators with the authors' released files.

    python -I scripts/v7x/v7c_fidelity_prep.py --out /home/infres/yinwang/CS_QMI/outputs/V7_CLOSEOUT/fidelity

DSG: 50 DSG-1k items, uniformly sampled (seed 2718) from the 900 items outside tifa160 (TIFA160's prompts are kept untouched for a possible
external evaluation); released = the PaLM 2 tuples / dependencies / questions in dsg-1k-anns.csv.  TIFA: 50 TIFA v1.0 captions (seed 2718);
released = tifa_v1.0_question_answers.json (GPT-3.5 questions after TIFA's UnifiedQA filter).  Neither file holds human image ratings.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import random

TP = Path("/projects/EEG-foundation-model/yinghao/third_party")
DSG_ANN = TP / "DSG@7330119" / "dsg" / "data" / "dsg-1k-anns.csv"
TIFA_QA = TP / "tifa@8e37050" / "tifa_v1.0" / "tifa_v1.0_question_answers.json"
SEED, N = 2718, 50


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(DSG_ANN))); by = defaultdict(list)
    for r in rows:
        by[r["item_id"]].append(r)
    items = sorted(i for i in by if not i.startswith("tifa160"))
    pick = sorted(random.Random(SEED).sample(items, N))
    with open(out / "dsg1k_50_prompts.jsonl", "w") as f:
        for i in pick:
            f.write(json.dumps({"prompt_id": i, "prompt": by[i][0]["text"]}) + "\n")
    with open(out / "dsg1k_50_released.jsonl", "w") as f:
        for i in pick:
            f.write(json.dumps({"prompt_id": i, "prompt": by[i][0]["text"], "tuples": [
                {"id": int(r["proposition_id"]), "dependency": [int(x) for x in str(r["dependency"]).split(",") if x.strip() != ""],
                 "category_broad": r["category_broad"], "category_detailed": r["category_detailed"], "tuple": r["tuple"],
                 "question": r["question_natural_language"]} for r in by[i]]}) + "\n")
    T = json.load(open(TIFA_QA)); bt = defaultdict(list)
    for q in T:
        bt[q["id"]].append(q)
    caps = sorted(bt); pick_t = sorted(random.Random(SEED).sample(caps, N))
    with open(out / "tifa_v1_50_prompts.jsonl", "w") as f:
        for i in pick_t:
            f.write(json.dumps({"prompt_id": i, "prompt": bt[i][0]["caption"]}) + "\n")
    with open(out / "tifa_v1_50_released.jsonl", "w") as f:
        for i in pick_t:
            f.write(json.dumps({"prompt_id": i, "prompt": bt[i][0]["caption"], "qas": [
                {k: q.get(k) for k in ("question", "choices", "answer", "element", "element_type")} for q in bt[i]]}) + "\n")
    meta = {"seed": SEED, "n": N, "dsg_source": str(DSG_ANN), "dsg_sha256": sha(DSG_ANN), "dsg_pool": len(items), "dsg_items": pick,
            "tifa_source": str(TIFA_QA), "tifa_sha256": sha(TIFA_QA), "tifa_pool": len(caps), "tifa_ids": pick_t,
            "note": "label-free: method outputs only (questions / tuples), no human image ratings; tifa160 items excluded"}
    (out / "fidelity_sets.json").write_text(json.dumps(meta, indent=1)); print(json.dumps({k: v for k, v in meta.items() if not isinstance(v, list)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
