"""V7C-01 / 02: E_auto (the frozen prompt-only parser's elements) and O for the M units on EvalMuse CAL_SELECT / CAL_PROB / DEV images.

    prep   --out DIR                         prompts (parser input) + per-role candidate lists from the V7 data manifest (label-free fields only)
    export --units U.. --cands F --cache C --tag T    the frozen x30_m_bench.export readout (O; E over the parser elements; decomposition_failed)
                                                       with the EvalMuse archive reader (VC.ArchiveImages) instead of image files
The candidate index is the image's position within its prompt by image_id (diagnostic pools only; no official grouping is implied).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v7"))
ROLES = ("CAL_SELECT", "CAL_PROB", "DEV")


def cmd_prep(a):
    import v7_common as VC
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); man = VC.read_jsonl(VC.DATA_OUT / "manifest_images.jsonl"); prompts = {}
    for role in ROLES:
        rows = sorted((r for r in man if r["role"] == role), key=lambda r: (r["prompt_id"], r["image_id"])); by = defaultdict(list)
        for r in rows:
            by[r["prompt_id"]].append(r)
        with open(out / f"cands_{role}.jsonl", "w") as fh:
            for pid, rr in by.items():
                for k, r in enumerate(rr):
                    fh.write(json.dumps({"prompt_id": pid, "generator_id": r["generator_id"], "image_id": r["image_id"], "candidate_index": k, "prompt": r["prompt"]}) + "\n")
                prompts[(role, pid)] = rr[0]["prompt"]
    with open(out / "evalmuse_cal_dev_prompts.jsonl", "w") as fh:
        for (role, pid), p in sorted(prompts.items()):
            fh.write(json.dumps({"source": f"evalmuse_{role}", "id": pid, "prompt": p}, ensure_ascii=False) + "\n")
    print(json.dumps({"prompts": len(prompts), **{role: sum(1 for k in prompts if k[0] == role) for role in ROLES}})); return 0


def cmd_export(a):
    import torch
    from transformers import BertTokenizerFast
    import v7_common as VC
    import x30_m_bench as XB
    device = "cuda" if torch.cuda.is_available() else "cpu"; VC.stage_workdir(VC.BIG_OUT / "work"); tok = BertTokenizerFast.from_pretrained(str(VC.BERT_DIR))
    cache = {r["prompt"]: r for r in VC.read_jsonl(a.cache) if "prompt" in r}; cands = VC.read_jsonl(a.cands)
    VC.import_lavis()
    from lavis.processors.blip_processors import BlipImageEvalProcessor
    proc = BlipImageEvalProcessor(image_size=364); images = VC.ArchiveImages()
    for u in a.units:
        XB.export(Path(u), cands, cache, images, a.tag, device, tok, proc)
    return 0


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep"); p.add_argument("--out", required=True)
    e = sub.add_parser("export"); e.add_argument("--units", nargs="+", required=True); e.add_argument("--cands", required=True); e.add_argument("--cache", required=True); e.add_argument("--tag", required=True)
    a = ap.parse_args(); return {"prep": cmd_prep, "export": cmd_export}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
