"""r7 R7-30 D1-30 latency (frozen latency_regime; coordinator decision D30-9).  Reference GPU NVIDIA H100 NVL on nodesumo01 only (the job
refuses another GPU unless --allow-any-gpu, smoke only).  Timing subset: the first 512 items of a benchmark in
sha256('r7-30/d1-30-timing/<qid>') order — all 512 at the primary B* = 4, the first 256 at B = 16 / 32, the first 64 at B = 1; answers are
not scored; one warm-up batch per (configuration, B) not recorded.  Generation is forced to 8 decode steps (frozen), so latency does not
depend on answer length.
Images are read and decoded before the clock starts (as in the frozen D1-10 timing, which timed host.encode on opened images).
Fixed-budget rows (full, full_ocr, static, sel): consecutive batches of B; per batch, CUDA-synchronised: vision (pre-processing + CLIP +
projector), select (question embedding, qsim scoring or PCA + selector forwards; 0 for full), assemble (splicing), prefill (separate
prefill-only pass, for TTFT), llm (prefill + 8 forced decode steps).  Per-item latency = (vision + select + assemble + llm) / B (the frozen
'(vision + selector + prefill + decode of a batch) / batch size'); median and P95 over batches.
Dynamic rows: the frozen rule buckets batches by kept-token count (equal budget inside a batch).  Stage 1 = consecutive arrival batches of B
(vision + selection, which yields each item's budget); stage 2 = per budget bucket, full batches of B for assemble + llm (a bucket's items
are cycled to fill its last batch: the steady-state cost of serving that budget, without partial-batch artefacts of a finite subset).
Per-item latency = median stage-1 per-item time + Σ_k w_k · median stage-2 per-item time in bucket k (w_k = bucket share); P95 likewise from
the P95s; TTFT = median stage-1 batch time + Σ_k w_k · median(assemble + prefill) of bucket k.
Each row carries a bootstrap CI (2,000 replicates over batches; within buckets for dynamic rows) of its median per-item latency — reported,
not propagated into the quality-at-latency interpolation.
    python scripts/r7_d1_30_timing.py --sets-dir S --bench gqa_dev --configs-file C --out O [--n 512 --n-big 256 --n1 64 --batches 1 4 16 32]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7_d1_30_run as RUN  # noqa: E402
import r7_vl_common as V  # noqa: E402

REFERENCE_GPU = "NVIDIA H100 NVL"
REPS = 2000


def timing_items(sets_dir, bench, n):
    items = [json.loads(x) for x in (Path(sets_dir) / f"{bench}.jsonl").read_text().splitlines() if x.strip()]
    return sorted(items, key=lambda it: V._h("d1-30-timing", it["qid"]))[:n]


def pool_size(B, a):
    return a.n1 if B == 1 else (a.n_big if B >= 16 else a.n)


def median_ci(x, reps=REPS, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    bs = np.median(x[rng.integers(0, len(x), (reps, len(x)))], 1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


class Clock:
    def __init__(self, torch):
        self.cuda = torch.cuda.is_available(); self.torch = torch

    def __call__(self):
        if self.cuda:
            self.torch.cuda.synchronize()
        return time.perf_counter()


def fixed_row(cfg, B, pool, host, bank, opener, clock):
    """Consecutive batches of B; returns per-batch component times (ms) and visual-token counts."""
    batches = [pool[i:i + B] for i in range(0, len(pool) - B + 1, B)]; comp = {k: [] for k in ("vision", "select", "assemble", "prefill", "llm")}; kept_n = []
    for bi, batch in enumerate([batches[0]] + batches):                                # first batch repeated as warm-up (not recorded)
        imgs = [opener(it["image_file"]) for it in batch]; t0 = clock(); feats = host.encode(imgs); t1 = clock()
        st = bank.prepare(feats, RUN.question_embeddings(host, [it["question"] for it in batch])) if cfg.startswith("sel:") else None
        qv = [host.question_vector(it["question"]) for it in batch] if cfg.startswith("static:qsim") else None
        kept = RUN.kept_for(cfg, batch, feats, host, bank, st, qv); t2 = clock()
        seqs = [RUN.sequence(host, it, f, kk, cfg) for it, f, kk in zip(batch, feats, kept)]; t3 = clock()
        host.last_logits(seqs); t4 = clock(); host.generate(seqs, max_new_tokens=8, min_new_tokens=8); t5 = clock()
        if bi:
            for k, v in (("vision", t1 - t0), ("select", t2 - t1), ("assemble", t3 - t2), ("prefill", t4 - t3), ("llm", t5 - t4)):
                comp[k].append(1e3 * v)
            kept_n += [RUN.N_VIS if kk is None else len(kk) for kk in kept]
    c = {k: np.asarray(v) for k, v in comp.items()}; per = (c["vision"] + c["select"] + c["assemble"] + c["llm"]) / B
    ttft = c["vision"] + c["select"] + c["assemble"] + c["prefill"]
    return {"n_batches": len(per), "per_item_ms_median": float(np.median(per)), "per_item_ms_median_ci95": median_ci(per),
            "per_item_ms_p95": float(np.percentile(per, 95)), "ttft_ms_median": float(np.median(ttft)), "ttft_ms_p95": float(np.percentile(ttft, 95)),
            **{f"{k}_ms_batch_median": float(np.median(v)) for k, v in c.items()}, "decode_ms_per_token_median": float(np.median((c["llm"] - c["prefill"]) / 8)),
            "mean_visual_tokens": float(np.mean(kept_n))}


def dynamic_row(cfg, B, pool, host, bank, opener, clock):
    """Stage 1: arrival batches (vision + selection -> budgets); stage 2: budget buckets, batches of B (last partial) for assemble + llm."""
    torch = host.torch; batches = [pool[i:i + B] for i in range(0, len(pool) - B + 1, B)]; s1, vis = [], []
    for bi, batch in enumerate([batches[0]] + batches):
        imgs = [opener(it["image_file"]) for it in batch]; t0 = clock(); feats = host.encode(imgs)
        st = bank.prepare(feats, RUN.question_embeddings(host, [it["question"] for it in batch])); kept = RUN.kept_for(cfg, batch, feats, host, bank, st); t2 = clock()
        if bi:
            s1.append(1e3 * (t2 - t0))
            for it, f, kk in zip(batch, feats, kept):
                vis.append((len(kk), it, f[torch.as_tensor(np.asarray(kk), device=f.device)]))
    buckets = {}
    for k, it, v in vis:
        buckets.setdefault(k, []).append((it, v))
    s2, n_items, cycled = {}, len(vis), {}
    for k in sorted(buckets):
        rows = buckets[k]; nb = -(-len(rows) // B); cyc = (rows * (-(-nb * B // len(rows))))[: nb * B]; cycled[k] = nb * B - len(rows)
        bb = [cyc[i:i + B] for i in range(0, nb * B, B)]; s2[k] = {"per": [], "ttft": [], "dec": []}
        for bi, batch in enumerate([bb[0]] + bb):                                      # warm-up per bucket (not recorded)
            t2 = clock(); seqs = [RUN.sequence(host, it, v, None, cfg) for it, v in batch]; t3 = clock()
            host.last_logits(seqs); t4 = clock(); host.generate(seqs, max_new_tokens=8, min_new_tokens=8); t5 = clock()
            if bi:
                s2[k]["per"].append(1e3 * (t3 - t2 + t5 - t4) / B); s2[k]["ttft"].append(1e3 * (t3 - t2 + t4 - t3))
                s2[k]["dec"].append(1e3 * ((t5 - t4) - (t4 - t3)) / 8)
    w = {k: len(buckets[k]) / n_items for k in buckets}; s1 = np.asarray(s1)

    def comp(stat1, stat2):
        return stat1(s1 / B) + sum(w[k] * stat2(np.asarray(s2[k]["per"])) for k in s2)
    rng = np.random.default_rng(0); bs = []
    for _ in range(REPS):                                                              # bootstrap within stages / buckets
        a = s1[rng.integers(0, len(s1), len(s1))] / B
        bs.append(np.median(a) + sum(w[k] * np.median(np.asarray(s2[k]["per"])[rng.integers(0, len(s2[k]["per"]), len(s2[k]["per"]))]) for k in s2))
    return {"n_batches": int(len(s1)), "stage2_batches": {str(k): len(s2[k]["per"]) for k in s2}, "stage2_cycled_items": {str(k): cycled[k] for k in s2},
            "budget_share": {str(k): w[k] for k in sorted(w)},
            "per_item_ms_median": float(comp(np.median, np.median)), "per_item_ms_median_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "per_item_ms_p95": float(comp(lambda x: np.percentile(x, 95), lambda x: np.percentile(x, 95))),
            "ttft_ms_median": float(np.median(s1) + sum(w[k] * np.median(s2[k]["ttft"]) for k in s2)),
            "select_vision_ms_batch_median": float(np.median(s1)), "decode_ms_per_token_median": float(sum(w[k] * np.median(s2[k]["dec"]) for k in s2)),
            "mean_visual_tokens": float(np.mean([k for k, _, _ in vis])), "bucketed": True}


def main(argv=None, host=None, bank=None, opener=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--sets-dir", required=True); ap.add_argument("--bench", required=True)
    ap.add_argument("--configs-file", required=True); ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--n-big", type=int, default=256); ap.add_argument("--n1", type=int, default=64); ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 32])
    ap.add_argument("--selectors-dir", default=str(RUN.D1 / "selectors")); ap.add_argument("--pca", default=str(RUN.D1 / "cache" / "pca.pt"))
    ap.add_argument("--allow-any-gpu", action="store_true"); a = ap.parse_args(argv)
    from PIL import Image
    items = timing_items(a.sets_dir, a.bench, a.n)
    configs = [c.strip() for c in Path(a.configs_file).read_text().splitlines() if c.strip() and not c.startswith("#")]
    if host is None:
        import r7_d1_teacher as TE
        host = TE.LlavaPrefill(Path(a.out))
    torch = host.torch; cuda = torch.cuda.is_available(); gpu = torch.cuda.get_device_name(0) if cuda else "cpu"
    if cuda and gpu != REFERENCE_GPU and not a.allow_any_gpu:
        raise SystemExit(f"latency runs only on the reference GPU ({REFERENCE_GPU}), got {gpu}")
    units = sorted({c.split(":")[1] for c in configs if c.startswith(("sel:", "dyn:"))})
    if bank is None and units:
        bank = RUN.SelectorBank(Path(a.selectors_dir), Path(a.pca), units)
    if units:
        RUN.check_tau(configs, bank)
    opener = opener or (lambda fp: Image.open(fp).convert("RGB")); clock = Clock(torch)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rows = []
    for B in a.batches:
        pool = items[: pool_size(B, a)]
        for cfg in configs:
            V.cuda_peak_reset(torch)
            r = (dynamic_row if cfg.startswith("dyn:") else fixed_row)(cfg, B, pool, host, bank, opener, clock)
            rows.append({"bench": a.bench, "config": cfg, "B": B, "pool": len(pool), **r, "peak_mem_gb": V.cuda_peak_gb(torch)}); print(V.jdump(rows[-1]), flush=True)
    rep = {"rows": rows, "env": V.env_info(), "gpu": gpu, "host": __import__("socket").gethostname(), "pools": {"B4": a.n, "B16_32": a.n_big, "B1": a.n1},
           "note": "frozen latency regime (B* = 4 primary); 8 forced decode steps; per-item = batch time / B; dynamic rows bucketed by kept-token count"}
    (out / f"timing_{a.bench}.json").write_text(V.jdump(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
