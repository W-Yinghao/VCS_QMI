"""V7 baseline B1: VQAScore with CLIP-FlanT5-XXL through t2v-metrics 3.0 (the paper-reproduction release), unmodified: its CLIPT5Model with the
default question / answer templates ('Does this figure show "{}"? Please answer yes or no.' / 'Yes'), bf16 autocast, 'pad' aspect handling.
Offline: HF_HOME = models/hf_cache_v7 (hub layout; see v7_hf_cache.py) and ./hf_cache -> its hub dir in the working directory.  The release
loads image PATHS, so the GenAI-Rank candidates are staged as files first (v7_native.py stage-rank).  One score per (candidate, its prompt).
    rank --inputs genai_rank_inputs.jsonl --images DIR --out F [--batch 16]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rank"); r.add_argument("--inputs", required=True); r.add_argument("--images", required=True); r.add_argument("--out", required=True)
    r.add_argument("--batch", type=int, default=16); a = ap.parse_args()
    import torch
    import t2v_metrics
    from t2v_metrics.models.vqascore_models.clip_t5_model import CLIPT5Model, default_answer_template, default_question_template
    rows = [json.loads(l) for l in open(a.inputs) if l.strip()]
    missing = [r["prompt_id"] for r in rows if not r.get("prompt")]
    if missing:
        raise SystemExit(f"{len(missing)} candidates without a prompt (the Rank prompt mapping is incomplete)")
    done = set(); out = Path(a.out)
    if out.exists():
        done = {json.loads(l)["image_id"] for l in open(out) if l.strip() and "image_id" in l}
    todo = [r for r in rows if r["image_id"] not in done]; t0 = time.time()
    model = CLIPT5Model(model_name="clip-flant5-xxl", device="cuda", cache_dir="./hf_cache")
    head = {"header": {"method": "B1 VQAScore CLIP-FlanT5-XXL", "t2v_metrics": getattr(t2v_metrics, "__version__", "3.0"), "torch": torch.__version__,
                       "gpu": torch.cuda.get_device_name(0), "question_template": default_question_template, "answer_template": default_answer_template,
                       "batch": a.batch, "load_s": time.time() - t0}}
    with open(out, "a") as fh:
        if not done:
            fh.write(json.dumps(head) + "\n")
        for s in range(0, len(todo), a.batch):
            chunk = todo[s:s + a.batch]; paths = [str(Path(a.images) / r["image_id"]) for r in chunk]
            sc = model.forward(images=paths, texts=[r["prompt"] for r in chunk]).float().cpu().tolist()
            for r, v in zip(chunk, sc):
                fh.write(json.dumps({"prompt_id": r["prompt_id"], "generator_id": r["generator_id"], "image_id": r["image_id"], "candidate_index": r["candidate_index"],
                                     "score": float(v)}) + "\n")
            fh.flush()
            if (s // a.batch) % 50 == 0:
                print(f"[v7_vqascore] {s + len(chunk)}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
