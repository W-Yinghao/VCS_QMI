"""V7X-30 B4T: TIFA on the GenAI-Rank pool through its released local route (github.com/Yushi-Hu/tifa @ 8e37050, read-only, imported by path).
Questions: tifa-benchmark/llama2_tifa_question_generation via get_llama2_question_and_answers (TIFA's settings: greedy beam 5, max_length 512);
filter: filter_question_and_answers with UnifiedQA v2 T5-large; VQA: VQAModel('mplug-large') (ModelScope mPLUG, local copy of the hub id at v1.0.0)
with TIFA's SBERT free-form -> choice mapping; score: tifa_score_single.  Scoped import (scripts/v7x/x_stubs.py): promptcap and lavis.models
(top-level imports of tifascore/vqa_models.py for the PromptCap / BLIP-2 routes) and fairseq (ModelScope's OFA speech preprocessor) are stubbed;
the mplug-large registry entry points to the local copy; no TIFA file is edited.  A generation TIFA's own parser rejects is stored raw with its error and keeps no question.  A prompt whose filter keeps no question gets score 0 for every candidate (all tie -> first index), flagged.
    qg --prompts P --out Q.jsonl [--limit N]                             questions per prompt (resumable cache)
    score --qa Q.jsonl --inputs I --out S.jsonl [--shard k/n] [--limit N] TIFA score per candidate (images staged node-local; no human score read)"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import types
import zipfile
from pathlib import Path

TIFA_ROOT = Path("/projects/EEG-foundation-model/yinghao/third_party/tifa@8e37050"); MODELS = Path("/projects/EEG-foundation-model/yinghao/models")
QG, UQA, MPLUG = MODELS / "tifa_llama2_qg", MODELS / "unifiedqa-v2-t5-large-1363200", MODELS / "mplug_vqa_coco_large_en"
RANK = Path("/projects/EEG-foundation-model/yinghao/data/genai_rank")


def import_tifa():
    sys.path.insert(0, str(Path(__file__).resolve().parent)); import x_stubs; x_stubs.install(tifa=True)     # fairseq / promptcap / lavis stubs
    sys.path.insert(0, str(TIFA_ROOT))
    import tifascore
    from tifascore import vqa_models as VM
    VM.vqa_models["mplug-large"] = ("MPLUG", str(MPLUG))
    return tifascore, VM


def read_jsonl(p):
    with open(p) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def cmd_qg(a):
    T, _ = import_tifa(); out = Path(a.out); done = {r["prompt_id"] for r in read_jsonl(out)} if out.exists() else set()
    prompts = [p for p in read_jsonl(a.prompts) if p["prompt_id"] not in done][: a.limit or None]
    pipe = T.get_llama2_pipeline(str(QG)); qa_model = T.UnifiedQAModel(str(UQA)); t0 = time.time()
    with open(out, "a") as fh:
        for k, p in enumerate(prompts):
            try:
                raw = T.get_llama2_question_and_answers(pipe, p["prompt"]); kept = T.filter_question_and_answers(qa_model, raw); err = None
            except (IndexError, ValueError) as e:                              # TIFA's parse_resp on a malformed generation (e.g. 'About x' without '(type)')
                from tifascore import question_gen_llama2 as QGM
                raw, kept, err = {"raw_completion": QGM.llama2_completion(pipe, p["prompt"])}, [], f"{type(e).__name__}: {e}"
            fh.write(json.dumps({"prompt_id": p["prompt_id"], "prompt": p["prompt"], "qas_generated": raw, "qas": kept, "parse_error": err}) + "\n"); fh.flush()
            if k % 25 == 0:
                print(f"[tifa] qg {k + 1}/{len(prompts)} ({time.time() - t0:.0f}s): {len(raw)} generated, {len(kept)} kept", flush=True)
    return 0


def stage(cands, dest: Path):
    zs = {}; dest.mkdir(parents=True, exist_ok=True)
    for c in cands:
        g = c["image_id"].split("/")[0]; z = zs.get(g) or zs.setdefault(g, zipfile.ZipFile(RANK / f"{g}.zip")); p = dest / c["image_id"]
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(z.read(c["image_id"]))
    return {c["image_id"]: str(dest / c["image_id"]) for c in cands}


def cmd_score(a):
    T, VM = import_tifa(); qa = {r["prompt_id"]: r for r in read_jsonl(a.qa)}; cands = read_jsonl(a.inputs)
    k, n = (int(x) for x in a.shard.split("/")); cands = [c for i, c in enumerate(cands) if i % n == k][: a.limit or None]
    out = Path(a.out); done = {r["image_id"] for r in read_jsonl(out)} if out.exists() else set(); cands = [c for c in cands if c["image_id"] not in done]
    paths = ({c["image_id"]: str(Path(a.image_root) / c["image_id"]) for c in cands} if a.image_root else stage(cands, Path(os.environ.get("V7X_LOCAL", "/tmp")) / "v7x_tifa_images"))
    vqa = T.VQAModel("mplug-large"); t0 = time.time()
    with open(out, "a") as fh:
        for i, c in enumerate(cands):
            q = qa.get(c["prompt_id"])
            if q is None:
                raise SystemExit(f"no questions cached for prompt {c['prompt_id']}")
            if q["qas"]:
                r = T.tifa_score_single(vqa, q["qas"], paths[c["image_id"]]); score, det = float(r["tifa_score"]), r["question_details"]
            else:
                score, det = 0.0, {}
            fh.write(json.dumps({"prompt_id": c["prompt_id"], "generator_id": c["generator_id"], "image_id": c["image_id"], "candidate_index": c["candidate_index"],
                                 "score": score, "n_questions": len(q["qas"]), "no_questions": not q["qas"], "question_details": det}) + "\n"); fh.flush()
            if i % 200 == 0:
                print(f"[tifa] score {i + 1}/{len(cands)} ({time.time() - t0:.0f}s)", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("qg"); g.add_argument("--prompts", required=True); g.add_argument("--out", required=True); g.add_argument("--limit", type=int, default=0)
    s = sub.add_parser("score"); s.add_argument("--qa", required=True); s.add_argument("--inputs", required=True); s.add_argument("--out", required=True)
    s.add_argument("--shard", default="0/1"); s.add_argument("--limit", type=int, default=0); s.add_argument("--image-root", default=None)
    a = ap.parse_args(argv)
    return {"qg": cmd_qg, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
