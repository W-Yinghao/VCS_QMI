"""V7X-30 B4D: DSG on the GenAI-Rank pool (github.com/j-min/DSG @ 7330119, read-only, imported by path).
gen  (env v7x_llm): DSG's generate_dsg (its tuple / dependency / question preambles and in-context examples) with a local LLM in place of the paper's
     PaLM 2 / the example's gpt-3.5-turbo-16k: Qwen2.5-32B-Instruct, the prompt as one user message (as dsg/openai_utils.py), greedy, max 500 new
     tokens (DSG's max_tokens); outputs parsed with dsg.parse_utils.  Reproduction adaptation (LLM substituted), recorded.
score (env v7_tifa): DSG's MPLUG class (mPLUG-large, local copy of the hub id at v1.0.0) answers every question; dsg.vqa_utils.calc_vqa_score
     (answer == 'yes'; children of a 'no' parent zeroed); score = average_score_with_dependency (without-dependency kept).  A prompt whose parse
     yields no question gets score 0 for every candidate (all tie -> first index), flagged.  No human score is read.
    gen --prompts P --out G.jsonl [--shard k/n] [--limit N]
    score --dsg G.jsonl [G2.jsonl ...] --inputs I --out S.jsonl [--shard k/n] [--limit N]"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path

DSG_ROOT = Path("/projects/EEG-foundation-model/yinghao/third_party/DSG@7330119"); MODELS = Path("/projects/EEG-foundation-model/yinghao/models")
LLM, MPLUG = MODELS / "Qwen2.5-32B-Instruct", MODELS / "mplug_vqa_coco_large_en"; RANK = Path("/projects/EEG-foundation-model/yinghao/data/genai_rank")
MAX_NEW = 500


def read_jsonl(p):
    with open(p) as fh:
        return [json.loads(l) for l in fh if l.strip()]


class LocalLLM:
    """generate_fn for DSG: the prompt as one user message, greedy decoding, at most MAX_NEW new tokens."""

    def __init__(self, path=LLM):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch; self.tok = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=torch.bfloat16, device_map="cuda").eval(); self.calls = 0; self.new_tokens = 0

    def __call__(self, prompt: str) -> str:
        ids = self.tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt").to("cuda")
        with self.torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False, temperature=None, top_p=None, top_k=None)
        gen = out[0, ids.shape[1]:]; self.calls += 1; self.new_tokens += int(gen.shape[0])
        return self.tok.decode(gen, skip_special_tokens=True)


def cmd_gen(a):
    sys.path.insert(0, str(DSG_ROOT))                                        # dsg.query_utils reads its in-context csv next to its own file
    from dsg.query_utils import generate_dsg
    from dsg.parse_utils import parse_dependency_output, parse_question_output, parse_tuple_output
    k, n = (int(x) for x in a.shard.split("/")); prompts = [p for i, p in enumerate(read_jsonl(a.prompts)) if i % n == k][: a.limit or None]
    out = Path(a.out); done = {r["prompt_id"] for r in read_jsonl(out)} if out.exists() else set(); prompts = [p for p in prompts if p["prompt_id"] not in done]
    llm = LocalLLM(); t0 = time.time()
    with open(out, "a") as fh:
        for i, p in enumerate(prompts):
            t_out, q_out, d_out = generate_dsg({p["prompt_id"]: {"input": p["prompt"]}}, generate_fn=llm, verbose=False)
            raw = {"tuple": t_out[p["prompt_id"]]["output"], "question": q_out[p["prompt_id"]]["output"], "dependency": d_out[p["prompt_id"]]["output"]}
            rec = {"prompt_id": p["prompt_id"], "prompt": p["prompt"], "raw": raw, "parse_error": None}
            try:
                rec["qid2tuple"] = parse_tuple_output(raw["tuple"]); rec["qid2question"] = parse_question_output(raw["question"]); rec["qid2dependency"] = parse_dependency_output(raw["dependency"])
            except Exception as e:                                               # recorded, never silently repaired
                rec["parse_error"] = f"{type(e).__name__}: {e}"
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            if i % 20 == 0:
                print(f"[dsg] gen {i + 1}/{len(prompts)} ({time.time() - t0:.0f}s, {llm.calls} LLM calls, {llm.new_tokens} new tokens)", flush=True)
    return 0


def cmd_score(a):
    sys.path.insert(0, str(Path(__file__).resolve().parent)); import x_stubs; x_stubs.install()        # fairseq stub (ModelScope's OFA speech path)
    sys.path.insert(0, str(DSG_ROOT))
    from dsg.vqa_utils import MPLUG as DSG_MPLUG, calc_vqa_score
    gen = {}
    for g in a.dsg:
        gen.update({r["prompt_id"]: r for r in read_jsonl(g)})
    k, n = (int(x) for x in a.shard.split("/")); cands = [c for i, c in enumerate(read_jsonl(a.inputs)) if i % n == k][: a.limit or None]
    out = Path(a.out); done = {r["image_id"] for r in read_jsonl(out)} if out.exists() else set(); cands = [c for c in cands if c["image_id"] not in done]
    local = Path(os.environ.get("V7X_LOCAL", "/tmp")) / "v7x_dsg_images"; zs = {}; vqa = DSG_MPLUG(str(MPLUG)); t0 = time.time()
    with open(out, "a") as fh:
        for i, c in enumerate(cands):
            r = gen.get(c["prompt_id"])
            if r is None:
                raise SystemExit(f"no DSG for prompt {c['prompt_id']}")
            qs = {int(q): v for q, v in (r.get("qid2question") or {}).items()}
            deps = {int(q): [int(x) for x in v] for q, v in (r.get("qid2dependency") or {}).items()}
            if not qs:
                rec = {"score": 0.0, "score_without_dependency": 0.0, "no_questions": True}
            else:
                g = c["image_id"].split("/")[0]; z = zs.get(g) or zs.setdefault(g, zipfile.ZipFile(RANK / f"{g}.zip")); p = local / c["image_id"]
                if not p.exists():
                    p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(z.read(c["image_id"]))
                ans = {q: vqa.vqa(str(p), v) for q, v in qs.items()}
                deps = {q: [d for d in deps.get(q, [0]) if d == 0 or d in ans] or [0] for q in ans}     # parents outside the question set: no constraint
                s = calc_vqa_score(ans, deps)
                rec = {"score": float(s["average_score_with_dependency"]), "score_without_dependency": float(s["average_score_without_dependency"]),
                       "no_questions": False, "qid2answer": ans, "qid2validity": s["qid2validity"]}
            fh.write(json.dumps({"prompt_id": c["prompt_id"], "generator_id": c["generator_id"], "image_id": c["image_id"], "candidate_index": c["candidate_index"],
                                 "n_questions": len(qs), **rec}) + "\n"); fh.flush()
            if i % 200 == 0:
                print(f"[dsg] score {i + 1}/{len(cands)} ({time.time() - t0:.0f}s)", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen"); g.add_argument("--prompts", required=True); g.add_argument("--out", required=True); g.add_argument("--shard", default="0/1"); g.add_argument("--limit", type=int, default=0)
    s = sub.add_parser("score"); s.add_argument("--dsg", nargs="+", required=True); s.add_argument("--inputs", required=True); s.add_argument("--out", required=True)
    s.add_argument("--shard", default="0/1"); s.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    return {"gen": cmd_gen, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
