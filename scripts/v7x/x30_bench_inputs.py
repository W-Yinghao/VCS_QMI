"""GenAI-Bench (BaiqiL/GenAI-Bench @ 94e543b, local) as an extra external set, label-free: reads ONLY the Index, Prompt and the six generator image
columns (never HumanRatings / Tags); writes genai_bench_inputs.jsonl (prompt_id 'gb<Index>', generator_id, image_id '<generator>/<Index>.<ext>',
candidate_index = generator order, prompt) and genai_bench_prompts.jsonl, and the images as files.  Scoring needs the owner's read approval.
    inputs --images DIR [--out-dir D]
    reuse --rank-gen G1 [G2 ...] --out-prefix P --shards N    pre-fill a GenAI-Bench generation cache from finished GenAI-Rank generations of
                                                               the SAME prompt text (greedy, deterministic), prompt_id remapped, shard = position % N"""
import argparse
import glob
import io
import json
import sys
from pathlib import Path

GENS = ("DALLE_3", "DeepFloyd_I_XL_v1", "Midjourney_6", "SDXL_2_1", "SDXL_Base", "SDXL_Turbo")
SRC = Path("/projects/EEG-foundation-model/yinghao/data/genai_bench/data"); DATA = Path("/home/infres/yinwang/CS_QMI/outputs/V7/data")


def cmd_inputs(a):
    import pyarrow.parquet as pq
    from PIL import Image
    img = Path(a.images); out = Path(a.out_dir); rows, prompts = [], []
    for f in sorted(SRC.glob("*.parquet")):
        t = pq.read_table(f, columns=["Index", "Prompt", *GENS]).to_pydict()
        for k in range(len(t["Index"])):
            idx = int(t["Index"][k]); pid = f"gb{idx:04d}"; prompts.append({"prompt_id": pid, "prompt": t["Prompt"][k], "bench_index": idx})
            for g, gen in enumerate(GENS):
                cell = t[gen][k]; b = cell["bytes"]; ext = (Image.open(io.BytesIO(b)).format or "png").lower().replace("jpeg", "jpg")
                iid = f"{gen}/{idx:05d}.{ext}"; p = img / iid; p.parent.mkdir(parents=True, exist_ok=True)
                if not p.exists():
                    p.write_bytes(b)
                rows.append({"prompt_id": pid, "generator_id": gen, "image_id": iid, "candidate_index": g, "prompt": t["Prompt"][k]})
    prompts.sort(key=lambda r: r["bench_index"]); rows.sort(key=lambda r: (r["prompt_id"], r["candidate_index"]))
    with open(out / "genai_bench_inputs.jsonl", "w") as fh:
        [fh.write(json.dumps(r) + "\n") for r in rows]
    with open(out / "genai_bench_prompts.jsonl", "w") as fh:
        [fh.write(json.dumps(r) + "\n") for r in prompts]
    print(json.dumps({"prompts": len(prompts), "images": len(rows), "columns_read": ["Index", "Prompt", *GENS]}))


def cmd_reuse(a):
    by_text = {}
    for g in a.rank_gen:
        for l in open(g):
            r = json.loads(l); by_text[r["prompt"]] = r
    prompts = [json.loads(l) for l in open(DATA / "genai_bench_prompts.jsonl")]; files = [open(f"{a.out_prefix}_{k}of{a.shards}.jsonl", "a") for k in range(a.shards)]; n = 0
    for i, p in enumerate(prompts):
        r = by_text.get(p["prompt"])
        if r is not None:
            files[i % a.shards].write(json.dumps({**r, "prompt_id": p["prompt_id"], "reused_from_rank_prompt_id": r["prompt_id"]}) + "\n"); n += 1
    [f.close() for f in files]; print(json.dumps({"bench_prompts": len(prompts), "reused": n}))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("inputs"); i.add_argument("--images", required=True); i.add_argument("--out-dir", default=str(DATA))
    r = sub.add_parser("reuse"); r.add_argument("--rank-gen", nargs="+", required=True); r.add_argument("--out-prefix", required=True); r.add_argument("--shards", type=int, required=True)
    a = ap.parse_args(); return cmd_inputs(a) if a.cmd == "inputs" else cmd_reuse(a)


if __name__ == "__main__":
    sys.exit(main())
