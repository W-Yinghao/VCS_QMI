"""r7 R7-30 D1-30 SparseVLM rows: the official code (Gumpest/SparseVLMs a9e7142, USE_VERSION 1_0) through r7_vl_common.Host, batch 1 (the
official sparse forward assumes one sequence).  Rows: the four native official configurations RETAIN_TOKN 192 / 128 / 96 / 64 as anchors
(coordinator decision D30-2) and 'adapted' rows at the D1-30 grid 72 / 144 / 288 / 432 (frozen OPEN-8).  TextVQA also runs the anchors with
the LLaVA OCR-token prompt (published-number anchors).  No latency claim is made against SparseVLM at B* = 4 (decision D30-1): its rows are
compared at matched equal-token budget (primary basis) and at matched batch-1 latency (secondary).
Adapted per-layer lists (pruning layers 2, 6, 15): the shape is interpolated linearly in the nominal budget between the bracketing official
v1.0 lists {64, 96, 128, 192} and the no-pruning list [576, 576, 576] at 576; one common scale c is then chosen so that the official
'Equal Tokens' counter, computed exactly by official_equal_tokens() (it reproduces the R7-20 measurements 192.03125 and 64.8125), is
closest to the target.  The runtime check compares the code's own counter with that prediction for every row.
The rows run in one process: the official module reads RETAIN_TOKN / sparse_token_dict at call time, so they are set per row
(score.RETAIN_TOKN and score.sparse_token_dict[T]); the environment variable is set as well.
    quality: python scripts/r7_d1_30_sparsevlm.py --sets-dir S --bench gqa_dev [...] --shard i/n --out O [--rows 192 128 96 64 72 144 288 432]
    timing : python scripts/r7_d1_30_sparsevlm.py --sets-dir S --bench gqa_dev --out O --timing 64     (nodesumo01 only; 8 forced tokens;
             the same first 64 timing items as our B = 1 rows; bootstrap CI of the median over items)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7_d1_30_run as RUN  # noqa: E402
import r7_vl_common as V  # noqa: E402

OFFICIAL_V1 = {64: [66, 30, 17], 96: [238, 48, 26], 128: [303, 110, 36], 192: [300, 200, 110]}   # score.py a9e7142, USE_VERSION 1_0
ANCHORS = (192, 128, 96, 64)
ADAPTED = (72, 144, 288, 432)
PRUNE_LAYERS, N_LAYERS, N_VIS = (2, 6, 15), 32, 576


def official_equal_tokens(lst, n=N_VIS):
    """The official prefill counter (num_token_pool / num_forward) for a per-layer list: at each pruning layer k = min(s, v − 1) tokens are
    kept, int(0.3 (v − k)) + 1 pruned tokens are merged into int(that / 10) + 1 cluster tokens; v then holds until the next pruning layer."""
    v, per = n, []
    for s in lst:
        k = min(int(s), v - 1); m1 = int((v - k) * 0.3) + 1; v = k + int(m1 / 10) + 1; per.append(v)
    spans = [PRUNE_LAYERS[0], PRUNE_LAYERS[1] - PRUNE_LAYERS[0], PRUNE_LAYERS[2] - PRUNE_LAYERS[1], N_LAYERS - PRUNE_LAYERS[2]]
    return (spans[0] * n + spans[1] * per[0] + spans[2] * per[1] + spans[3] * per[2]) / N_LAYERS


def adapted_list(T, n=N_VIS):
    if T in OFFICIAL_V1:
        return list(OFFICIAL_V1[T])
    anchors = dict(OFFICIAL_V1); anchors[n] = [n, n, n]; keys = sorted(anchors)
    lo = max(k for k in keys if k < T); hi = min(k for k in keys if k > T); w = (T - lo) / (hi - lo)
    shape = [a + w * (b - a) for a, b in zip(anchors[lo], anchors[hi])]
    best = None
    for c in np.linspace(0.5, 2.0, 3001):
        lst = [int(min(n, max(1, round(c * s)))) for s in shape]; err = abs(official_equal_tokens(lst, n) - T)
        if best is None or err < best[0] - 1e-12:
            best = (err, lst)
    return best[1]


def row_tag(T, ocr=False):
    return f"sparsevlm:{T}:{'official' if T in ANCHORS else 'adapted'}" + (":ocr" if ocr else "")


def set_row(SC, T):
    os.environ["RETAIN_TOKN"] = str(T); SC.RETAIN_TOKN = T; SC.sparse_token_dict[T] = adapted_list(T)


def rows_for(bench, rows):
    """(budget, OCR prompt) rows: every requested row with the primary prompt; on TextVQA also the requested official anchors with OCR."""
    out = [(T, False) for T in rows]
    if bench.startswith("textvqa"):
        out += [(T, True) for T in rows if T in ANCHORS]
    return out


def answer_timed(host, prompt, image_file, n_new):
    """Batch-1 official path with forced n_new tokens; wall time incl. image pre-processing (CUDA-synchronised); the image is read and decoded
    before the clock starts, as in our rows."""
    torch = host.torch
    from PIL import Image
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token
    conv = conv_templates["vicuna_v1"].copy(); conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt); conv.append_message(conv.roles[1], None)
    ids = tokenizer_image_token(conv.get_prompt(), host.tok, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    img = Image.open(image_file).convert("RGB"); torch.cuda.synchronize(); t0 = time.perf_counter()
    px = process_images([img], host.proc, host.model.config)[0].unsqueeze(0).to(dtype=torch.float16, device="cuda")
    with torch.inference_mode():
        host.model.generate(ids, images=px, image_sizes=[img.size], do_sample=False, num_beams=1, max_new_tokens=n_new, min_new_tokens=n_new, use_cache=True)
    torch.cuda.synchronize()
    return 1e3 * (time.perf_counter() - t0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--sets-dir", required=True); ap.add_argument("--bench", nargs="+", required=True)
    ap.add_argument("--shard", default="0/1"); ap.add_argument("--out", required=True); ap.add_argument("--rows", type=int, nargs="+", default=list(ANCHORS + ADAPTED))
    ap.add_argument("--timing", type=int, default=0); ap.add_argument("--allow-any-gpu", action="store_true"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv); work = []
    for bench in a.bench:
        out = Path(a.out) / bench; out.mkdir(parents=True, exist_ok=True); plan = rows_for(bench, a.rows)
        if a.timing:
            items = [json.loads(x) for x in (Path(a.sets_dir) / f"{bench}.jsonl").read_text().splitlines() if x.strip()]
            items = sorted(items, key=lambda it: V._h("d1-30-timing", it["qid"]))[: a.timing]; plan = [(T, False) for T in a.rows]; todo = None
        else:
            items = RUN.load_items(a.sets_dir, bench, a.shard)[: a.limit or None]
            todo = {(T, o): [it for it in items if it["qid"] not in {r["qid"] for r in V.read_jsonl(out / f"{RUN.cfg_file(row_tag(T, o))}.jsonl")}] for T, o in plan}
        work.append((bench, out, plan, items, todo))
    if not a.timing and not any(v for w in work for v in w[4].values()):
        for bench, out, plan, items, todo in work:
            write_marker(out, bench, a.shard, [{"config": row_tag(T, o), "items": 0} for T, o in plan], None, None)
        print("[r7_d1_30_sparsevlm] complete"); return 0
    host = V.Host(Path(a.out), sparse=True, retain=192); torch = host.torch
    import llava.model.language_model.score as SC                                       # imported by the host; the module the official code reads
    gpu = torch.cuda.get_device_name(0)
    if a.timing and gpu != "NVIDIA H100 NVL" and not a.allow_any_gpu:
        raise SystemExit(f"latency runs only on the reference GPU, got {gpu}")
    for bench, out, plan, items, todo in work:
        recs = []
        for T, ocr in plan:
            set_row(SC, T); host.reset_sparse_counters(); tag = row_tag(T, ocr); t0 = time.time(); V.cuda_peak_reset(torch)
            if a.timing:
                for it in items[:3]:                                                   # warm-up
                    answer_timed(host, it["prompt"], it["image_file"], 8)
                ms = [answer_timed(host, it["prompt"], it["image_file"], 8) for it in items]
                rng = np.random.default_rng(0); x = np.asarray(ms); bs = np.median(x[rng.integers(0, len(x), (2000, len(x)))], 1)
                rec = {"bench": bench, "config": tag, "B": 1, "n_items": len(ms), "per_item_ms_median": float(np.median(ms)), "per_item_ms_p95": float(np.percentile(ms, 95)),
                       "per_item_ms_median_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
            else:
                with open(out / f"{RUN.cfg_file(tag)}.jsonl", "a") as fh:
                    for it in todo[(T, ocr)]:
                        r = host.answer(it["prompt_ocr_anchor"] if ocr else it["prompt"], it["image_file"], max_new_tokens=RUN.MAX_NEW[bench])
                        fh.write(V.jdump({"qid": it["qid"], "image_id": it["image_id"], "pred": r["text"], "kept": T, "wall_ms": r["wall_ms"]}) + "\n"); fh.flush()
                rec = {"bench": bench, "config": tag, "shard": a.shard, "items": len(todo[(T, ocr)])}
            cnt = host.sparse_counters(); pred = official_equal_tokens(SC.sparse_token_dict[T])
            rec.update({"retain": T, "layer_list": SC.sparse_token_dict[T], "predicted_equal_tokens": pred, **cnt, "seconds": time.time() - t0,
                        "peak_mem_gb": V.cuda_peak_gb(torch), "counter_check": (None if not cnt else abs(cnt["equal_visual_tokens"] - pred) < 1e-6)})
            recs.append(rec); print(V.jdump(rec), flush=True)
            if cnt and abs(cnt["equal_visual_tokens"] - pred) > 1e-6:
                raise SystemExit(f"official counter {cnt['equal_visual_tokens']} != predicted {pred} for row {tag}: the row setting did not take effect")
        if a.timing:
            (out / f"sparsevlm_timing_{bench}.json").write_text(V.jdump({"rows": recs, "env": V.env_info(), "gpu": gpu, "host": __import__("socket").gethostname()}, indent=1))
        else:
            write_marker(out, bench, a.shard, recs, gpu, V.env_info())
    return 0


def write_marker(out: Path, bench, shard, recs, gpu, env):
    """Completion marker for the driver's file gate (kept when it already exists: a rerun of a complete shard does no work)."""
    p = out / f"sparsevlm_run_{bench}_shard{shard.replace('/', 'of')}.json"
    if gpu is not None or not p.exists():
        p.write_text(V.jdump({"rows": recs, "env": env, "gpu": gpu}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
