"""r7 R7-40c latency (the frozen D1-30 timing design, D30-9, on Qwen2.5-VL-7B): reference GPU NVIDIA H100 NVL on nodesumo01 only (the job
refuses another GPU unless --allow-any-gpu, smoke only).  Production runs as ONE bundle job on nodesumo01 (owner rule 2026-10-13; 'bundle'
below): Qwen's B* first (the R7-40a report, or the frozen R7-40a procedure run inside the bundle), then the three benchmarks back-to-back.  B* is Qwen's own, from the R7-40a reference timing report (D40a-7; rule and
pool-of-batches procedure pre-declared in reports/R7/R7_40_PROTOCOL_DRAFT_20261012.yaml and hash-checked): the report must come from the
reference GPU on nodesumo01; if it has no B* (OPEN-R740-8) the primary endpoint is quality at fixed fractions and the 512-item pool goes to
B = 4 (the D1-30 primary batch size), with the latency rows reported as secondary (no acceleration claim).
Timing subset (D1-30): the first 512 items of a benchmark in sha256('r7-30/d1-30-timing/<qid>') order (r7_d1_30_timing.timing_items); all 512
at the primary B (B* or 4), the first 256 at the other B >= 4, the first 64 at B = 1; answers are not scored; one warm-up batch per
(configuration, B), not recorded; 8 forced decode steps (min_new_tokens = max_new_tokens = 8, shipped repetition penalty), so latency does not
depend on answer length.  Images are opened and decoded before the clock (as the frozen D1-10 / R7-40a timing).
Fixed-fraction rows (full, static, sel): consecutive batches of B; per batch, CUDA-synchronised: vision (image processor + vision tower, one
batched tower call over the batch's images), select (grid / random ordering, qsim scoring, or question ids + embeddings + PCA + one batched
selector forward + top-k; 0 for full), assemble (chat-template segments, embeddings, full-sequence 3-D rotary ids, pruning, splicing), prefill
(separate prefill-only pass, for TTFT), llm (prefill + 7 decode forwards = 8 forced new tokens).  Per-item latency = (vision + select +
assemble + llm) / B; median and P95 over batches; bootstrap CI of the median over batches (2,000 replicates; reported, not propagated).
Dynamic rows (frozen 'batches bucketed by kept-token count', implemented as D1-30's two stages, with Qwen's variable N the bucket is the
chosen nominal fraction — equal budget fraction inside a batch): stage 1 = arrival batches of B (vision + selection, which yields each item's
fraction), stage 2 = per fraction bucket, full batches of B (a bucket's items cycled to fill its last batch) for assemble + llm.  Per-item
latency = median stage-1 per-item time + Σ_f w_f · median stage-2 per-item time of bucket f (w_f = bucket share); P95 likewise.
Every row reports the vision share of its per-item latency (claims are limited to the LLM side, R7-40 scope rule).
    python -I scripts/r7_d1_40c_timing.py --sets-dir S --bench gqa_dev --configs-file C --out O [--timing-report R] [--b-star N|none]   (one benchmark)
    python -I scripts/r7_d1_40c_timing.py bundle --sets-dir S --configs-dir C --out O --b-star-mode in_bundle|report   (production: ONE job on nodesumo01)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7_d1_30_timing as TM  # noqa: E402  (frozen D1-30: timing subset order, median CI, clock)
import r7_d1_40_masks as M4  # noqa: E402
import r7_d1_40c_plan as PL  # noqa: E402
import r7_d1_40c_run as RN  # noqa: E402
import r7_vl_common as V  # noqa: E402

REFERENCE_GPU = "NVIDIA H100 NVL"
REFERENCE_HOST = "nodesumo01"
BATCHES = (1, 4, 16, 32)
FORCED = 8


def read_b_star(report: Path, allow_any=False):
    """Qwen's B* from the R7-40a reference timing report; refuses a report not measured on the reference GPU / node (unless smoke)."""
    rep = json.loads(Path(report).read_text())
    ok = rep.get("gpu") == REFERENCE_GPU and str(rep.get("host", "")).startswith(REFERENCE_HOST)
    if not ok and not allow_any:
        raise SystemExit(f"timing report {report} was not measured on {REFERENCE_GPU} / {REFERENCE_HOST} (gpu {rep.get('gpu')}, host {rep.get('host')})")
    return rep.get("B_star"), {"file": str(report), "sha256": RN.sha_file(report), "gpu": rep.get("gpu"), "host": rep.get("host"),
                                "B_star": rep.get("B_star"), "B_star_without_select": rep.get("B_star_without_select"), "rule": rep.get("rule")}


def primary_batch(b_star):
    return int(b_star) if b_star else 4


def pool_size(B, primary_B, n=512, n_big=256, n1=64):
    return n if B == primary_B else (n1 if B == 1 else n_big)


def _frame(host, it, n, thw):
    """Chat-prompt segments around the image placeholder, their embeddings and the full sequence's 3-D rotary ids (official get_rope_index)."""
    pre, post = host.segments(it["prompt"], True)
    return pre, post, host.core.embed_ids(pre), host.core.embed_ids(post), host.core.full_positions(pre, n, post, thw)


def assemble_kept(host, it, v_kept, n, thw, kk):
    """The timed 'assemble' step: frame + splice of the kept tokens v_kept (already gathered; kk = their raster indices, None = all n)."""
    import r7_d1_40_host as QH
    torch = host.torch; pre, post, pre_e, post_e, pos = _frame(host, it, n, thw)
    if kk is None:
        return torch.cat([pre_e, v_kept, post_e]), pos, np.concatenate([pre, np.full(n, host.image_token_id), post])
    kk = np.asarray(kk, dtype=np.int64)
    return torch.cat([pre_e, v_kept, post_e]), QH.prune_positions(pos, len(pre), n, kk), np.concatenate([pre, np.full(len(kk), host.image_token_id), post])


def assemble_raw(host, it, feat, thw, kk):
    """'assemble' from the image's full merged tokens: gather the kept rows, then frame + splice."""
    torch = host.torch; n = int(feat.shape[0])
    v = feat if kk is None else feat[torch.as_tensor(np.asarray(kk, dtype=np.int64), device=feat.device)]
    return assemble_kept(host, it, v, n, thw, kk)


def select_batch(cfg, batch, enc, host, bank):
    """Kept indices (None = all) and fractions of one batch for a configuration (the timed 'select' step)."""
    kind, a, f = PL.parse(cfg)
    if kind == "full":
        return [None] * len(batch), [1.0] * len(batch)
    if kind == "static":
        out = []
        for it, (feat, grid, thw) in zip(batch, enc):
            rows, cols = int(grid[0]), int(grid[1]); n = rows * cols
            if a == "grid":
                order = M4.order_grid(it["qid"], rows, cols, with_logp=False)[0]
            elif a == "random":
                order = M4.order_random(it["qid"], rows, cols)[0]
            else:
                fl = feat.float(); qv = host.question_vector(it["question"]).to(fl.device)
                order = M4.order_qsim(it["qid"], ((fl @ qv) / (fl.norm(dim=1) * qv.norm() + 1e-12)).cpu().numpy(), n)[0]
            out.append(M4.kept(order, M4.budget(f, n)))
        return out, [f] * len(batch)
    states = [bank.prepare(feat, grid, host.core.embed_ids(host.tok(it["question"], add_special_tokens=False).input_ids)) for it, (feat, grid, _) in zip(batch, enc)]
    kept, fr = bank.select_batch(a, states, f=f if kind == "sel" else None, op=f if kind == "dyn" else None)
    return [None if x >= 1.0 else k for k, x in zip(kept, fr)], fr


def fixed_row(cfg, B, pool, host, bank, opener, clock):
    batches = [pool[i:i + B] for i in range(0, len(pool) - B + 1, B)]; comp = {k: [] for k in ("vision", "select", "assemble", "prefill", "llm")}
    kept_n, frac = [], []
    for bi, batch in enumerate([batches[0]] + batches):                                 # first batch repeated as warm-up (not recorded)
        imgs = [opener(it["image_file"]) for it in batch]
        t0 = clock(); enc = host.encode(imgs); t1 = clock()
        kept, fr = select_batch(cfg, batch, enc, host, bank); t2 = clock()
        seqs, poss, pids = zip(*[assemble_raw(host, it, e[0], e[2], kk) for it, e, kk in zip(batch, enc, kept)]); t3 = clock()
        host.core.last_logits(list(seqs), list(poss)); t4 = clock()
        host.core.generate(list(seqs), list(poss), list(pids), max_new_tokens=FORCED, min_new_tokens=FORCED); t5 = clock()
        if bi:
            for k, v in (("vision", t1 - t0), ("select", t2 - t1), ("assemble", t3 - t2), ("prefill", t4 - t3), ("llm", t5 - t4)):
                comp[k].append(1e3 * v)
            kept_n += [int(e[0].shape[0]) if kk is None else len(kk) for e, kk in zip(enc, kept)]
            frac += [(1.0 if kk is None else len(kk) / int(e[0].shape[0])) for e, kk in zip(enc, kept)]
    c = {k: np.asarray(v) for k, v in comp.items()}; per = (c["vision"] + c["select"] + c["assemble"] + c["llm"]) / B
    ttft = c["vision"] + c["select"] + c["assemble"] + c["prefill"]
    return {"n_batches": len(per), "per_item_ms_median": float(np.median(per)), "per_item_ms_median_ci95": TM.median_ci(per),
            "per_item_ms_p95": float(np.percentile(per, 95)), "ttft_ms_median": float(np.median(ttft)), "ttft_ms_p95": float(np.percentile(ttft, 95)),
            **{f"{k}_ms_batch_median": float(np.median(v)) for k, v in c.items()},
            "vision_share_of_per_item": float(np.median(c["vision"] / B) / np.median(per)),
            "decode_ms_per_token_median": float(np.median((c["llm"] - c["prefill"]) / (FORCED - 1))) if FORCED > 1 else None,
            "mean_visual_tokens": float(np.mean(kept_n)), "mean_kept_fraction": float(np.mean(frac))}


def dynamic_row(cfg, B, pool, host, bank, opener, clock):
    torch = host.torch; batches = [pool[i:i + B] for i in range(0, len(pool) - B + 1, B)]; s1, s1_vis, items = [], [], []
    for bi, batch in enumerate([batches[0]] + batches):
        imgs = [opener(it["image_file"]) for it in batch]
        t0 = clock(); enc = host.encode(imgs); t1 = clock(); kept, fr = select_batch(cfg, batch, enc, host, bank); t2 = clock()
        if bi:
            s1.append(1e3 * (t2 - t0)); s1_vis.append(1e3 * (t1 - t0))
            for it, (feat, grid, thw), kk, x in zip(batch, enc, kept, fr):
                v = feat if kk is None else feat[torch.as_tensor(np.asarray(kk, dtype=np.int64), device=feat.device)]
                items.append((x, it, v, kk, thw, int(feat.shape[0])))
    buckets = {}
    for x, it, v, kk, thw, n in items:
        buckets.setdefault(x, []).append((it, v, kk, thw, n))
    s2, cycled = {}, {}
    for x in sorted(buckets):
        rows = buckets[x]; nb = -(-len(rows) // B); cyc = (rows * (-(-nb * B // len(rows))))[: nb * B]; cycled[x] = nb * B - len(rows)
        bb = [cyc[i:i + B] for i in range(0, nb * B, B)]; s2[x] = {"per": [], "ttft": [], "dec": []}
        for bi, batch in enumerate([bb[0]] + bb):                                       # warm-up per bucket (not recorded)
            t2 = clock(); seqs, poss, pids = zip(*[assemble_kept(host, it, v, n, thw, kk) for it, v, kk, thw, n in batch]); t3 = clock()
            host.core.last_logits(list(seqs), list(poss)); t4 = clock()
            host.core.generate(list(seqs), list(poss), list(pids), max_new_tokens=FORCED, min_new_tokens=FORCED); t5 = clock()
            if bi:
                s2[x]["per"].append(1e3 * (t3 - t2 + t5 - t4) / B); s2[x]["ttft"].append(1e3 * (t3 - t2 + t4 - t3))
                s2[x]["dec"].append(1e3 * ((t5 - t4) - (t4 - t3)) / (FORCED - 1))
    n_items = len(items); w = {x: len(buckets[x]) / n_items for x in buckets}; s1 = np.asarray(s1)

    def comp(stat1, stat2):
        return stat1(s1 / B) + sum(w[x] * stat2(np.asarray(s2[x]["per"])) for x in s2)
    rng = np.random.default_rng(0); bs = []
    for _ in range(TM.REPS):                                                            # bootstrap within stages / buckets
        a = s1[rng.integers(0, len(s1), len(s1))] / B
        bs.append(np.median(a) + sum(w[x] * np.median(np.asarray(s2[x]["per"])[rng.integers(0, len(s2[x]["per"]), len(s2[x]["per"]))]) for x in s2))
    per_med = float(comp(np.median, np.median))
    return {"n_batches": int(len(s1)), "stage2_batches": {PL.fkey(x): len(s2[x]["per"]) for x in s2}, "stage2_cycled_items": {PL.fkey(x): cycled[x] for x in s2},
            "budget_share": {PL.fkey(x): w[x] for x in sorted(w)}, "per_item_ms_median": per_med,
            "per_item_ms_median_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "per_item_ms_p95": float(comp(lambda x: np.percentile(x, 95), lambda x: np.percentile(x, 95))),
            "ttft_ms_median": float(np.median(s1) + sum(w[x] * np.median(s2[x]["ttft"]) for x in s2)),
            "select_vision_ms_batch_median": float(np.median(s1)), "vision_ms_batch_median": float(np.median(s1_vis)),
            "vision_share_of_per_item": float(np.median(np.asarray(s1_vis) / B) / per_med),
            "decode_ms_per_token_median": float(sum(w[x] * np.median(s2[x]["dec"]) for x in s2)),
            "mean_visual_tokens": float(np.mean([int(v.shape[0]) for _, _, v, _, _, _ in items])),
            "mean_kept_fraction": float(np.mean([int(v.shape[0]) / n for _, _, v, _, _, n in items])), "bucketed": True, "bucket": "chosen fraction"}


def main(argv=None, host=None, bank=None, opener=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--sets-dir", required=True); ap.add_argument("--bench", required=True)
    ap.add_argument("--configs-file", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--timing-report", default=str(PL.TIMING_REPORT)); ap.add_argument("--b-star", default=None, help="smoke only: override (int or 'none')")
    ap.add_argument("--n", type=int, default=512); ap.add_argument("--n-big", type=int, default=256); ap.add_argument("--n1", type=int, default=64)
    ap.add_argument("--batches", type=int, nargs="+", default=list(BATCHES)); ap.add_argument("--selectors-dir", default=str(RN.SELECTORS))
    ap.add_argument("--pca", default=str(RN.PCA)); ap.add_argument("--allow-any-gpu", action="store_true"); a = ap.parse_args(argv)
    if a.b_star is not None:
        if not a.allow_any_gpu:
            raise SystemExit("--b-star overrides are for smokes only (with --allow-any-gpu)")
        bstar, brec = (None if a.b_star == "none" else int(a.b_star)), {"override": a.b_star, "smoke": True}
    else:
        bstar, brec = read_b_star(Path(a.timing_report), a.allow_any_gpu)
    pb = primary_batch(bstar)
    if pb not in a.batches and not a.allow_any_gpu:                  # the scorer reads the latency basis at the primary batch: refuse before measuring
        raise SystemExit(f"the primary batch {pb} (B* {bstar}) is not among the measured batches {a.batches}")
    items = [{k: v for k, v in it.items() if k not in RN.ANSWER_KEYS} for it in TM.timing_items(a.sets_dir, a.bench, a.n)]
    configs = PL.read_configs(a.configs_file); [PL.parse(c) for c in configs]
    if host is None:
        import r7_d1_40_host as QH
        host = QH.QwenPrefill(Path(a.out))
    torch = host.torch; cuda = torch.cuda.is_available(); gpu = torch.cuda.get_device_name(0) if cuda else "cpu"
    if cuda and gpu != REFERENCE_GPU and not a.allow_any_gpu:
        raise SystemExit(f"latency runs only on the reference GPU ({REFERENCE_GPU}), got {gpu}")
    units = sorted({PL.parse(c)[1] for c in configs if c.startswith(("sel:", "dyn:"))})
    if bank is None and units:
        bank = RN.QwenSelectorBank(Path(a.selectors_dir), Path(a.pca), units)
    if units:
        RN.check_tau(configs, bank)
    opener = opener or RN.open_image; clock = TM.Clock(torch); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rows = []; t0 = time.time()
    for B in a.batches:
        pool = items[: pool_size(B, pb, a.n, a.n_big, a.n1)]
        for cfg in configs:
            V.cuda_peak_reset(torch)
            r = (dynamic_row if cfg.startswith("dyn:") else fixed_row)(cfg, B, pool, host, bank, opener, clock)
            rows.append({"bench": a.bench, "config": cfg, "B": B, "pool": len(pool), **r, "peak_mem_gb": V.cuda_peak_gb(torch)}); print(V.jdump(rows[-1]), flush=True)
    rep = {"rows": rows, "env": V.env_info(), "gpu": gpu, "host": __import__("socket").gethostname(), "b_star": bstar, "primary_batch": pb,
           "b_star_source": brec, "pools": {f"B{B}": pool_size(B, pb, a.n, a.n_big, a.n1) for B in a.batches}, "forced_decode_steps": FORCED,
           "seconds": time.time() - t0, "allow_any_gpu": bool(a.allow_any_gpu),
           "note": "D1-30 timing design on Qwen2.5-VL-7B; per-item = batch time / B; dynamic rows bucketed by chosen fraction; vision share reported"}
    (out / f"timing_{a.bench}.json").write_text(V.jdump(rep, indent=1))
    return 0


BUNDLE_MARK = "timing_bundle.done.json"


def frozen_40a_timing(host, report: Path, allow_any_gpu=False):
    """Qwen's B* by the frozen R7-40a procedure: r7_d1_40_teacher.cmd_timing with exactly the arguments of the frozen wrapper's 'timing' mode
    (job 1035608): R7-20 GQA fixture, B in {1, 4, 16, 32}, pools 192 / 64, reference GPU only; writes <report dir>/timing_report.json."""
    from argparse import Namespace
    import r7_d1_40_teacher as TE40
    return TE40.cmd_timing(Namespace(out=str(Path(report).parent), fixture=str(TE40.FIXTURE), batches=[1, 4, 16, 32], n=192, n1=64,
                                     allow_any_gpu=allow_any_gpu), host)


def bundle(argv=None, host=None, bank=None, opener=None, bstar_fn=None) -> int:
    """ONE job on the reference node for every timing unit of R7-40c (owner rule 2026-10-13: units that must run on one specific node are one
    job that runs them back-to-back, so other users' jobs cannot slip in between).  One host load, then:
      part 0  Qwen's B*: the R7-40a reference report (outputs/R7_D1/r7_40/timing/timing_report.json) is reused when it exists; otherwise, with
              --b-star-mode in_bundle, the frozen R7-40a procedure runs here first and writes that report (never a second determination: an
              existing report is never re-measured); with --b-star-mode report a missing report stops the bundle
      parts 1-3  the D1-30 timing rows of gqa_dev, textvqa_val, chartqa_val (this module's main, unchanged) -> <out>/<bench>/timing_<bench>.json
    then the single bundle marker <out>/timing_bundle.done.json (parts, status, sha256, seconds, GPU, node, job).  The driver retries the
    whole bundle; a part whose output an earlier attempt already wrote is kept (listed as such in the marker)."""
    ap = argparse.ArgumentParser(); ap.add_argument("--sets-dir", required=True); ap.add_argument("--configs-dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--timing-report", default=str(PL.TIMING_REPORT)); ap.add_argument("--b-star-mode", choices=["in_bundle", "report"], default="report")
    ap.add_argument("--benches", nargs="+", default=list(PL.BENCHES)); ap.add_argument("--selectors-dir", default=str(RN.SELECTORS)); ap.add_argument("--pca", default=str(RN.PCA))
    ap.add_argument("--allow-any-gpu", action="store_true"); ap.add_argument("--b-star", default=None, help="smoke only (with --allow-any-gpu)")
    ap.add_argument("--n", type=int, default=512); ap.add_argument("--n-big", type=int, default=256); ap.add_argument("--n1", type=int, default=64)
    ap.add_argument("--batches", type=int, nargs="+", default=list(BATCHES)); a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t_all = time.time(); parts = []; report = Path(a.timing_report)
    if host is None:
        import r7_d1_40_host as QH
        host = QH.QwenPrefill(out)
    torch = host.torch; gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available() and gpu != REFERENCE_GPU and not a.allow_any_gpu:
        raise SystemExit(f"latency runs only on the reference GPU ({REFERENCE_GPU}), got {gpu}")
    t0 = time.time()
    if a.b_star is not None:
        if not a.allow_any_gpu:
            raise SystemExit("--b-star overrides are for smokes only (with --allow-any-gpu)")
        parts.append({"part": "b_star", "status": f"smoke override {a.b_star}"})
    elif report.exists():
        parts.append({"part": "b_star", "status": "reused existing R7-40a report", "file": str(report), "sha256": RN.sha_file(report)})
    elif a.b_star_mode == "in_bundle":
        (bstar_fn or frozen_40a_timing)(host, report, a.allow_any_gpu)
        if not report.exists():
            raise SystemExit(f"the frozen R7-40a timing procedure did not write {report}")
        parts.append({"part": "b_star", "status": "measured in this bundle (frozen R7-40a procedure)", "file": str(report), "sha256": RN.sha_file(report),
                      "seconds": time.time() - t0})
    else:
        raise SystemExit(f"no R7-40a timing report at {report} (--b-star-mode report): the bundle stops")
    common = ["--sets-dir", a.sets_dir, "--timing-report", str(report), "--selectors-dir", a.selectors_dir, "--pca", a.pca] + (["--allow-any-gpu"] if a.allow_any_gpu else [])
    common += (["--b-star", a.b_star] if a.b_star is not None else []) + ["--n", str(a.n), "--n-big", str(a.n_big), "--n1", str(a.n1), "--batches", *[str(x) for x in a.batches]]
    if bank is None:
        units = sorted({PL.parse(c)[1] for b in a.benches for c in PL.read_configs(Path(a.configs_dir) / f"timing_{b}.txt") if c.startswith(("sel:", "dyn:"))})
        bank = RN.QwenSelectorBank(Path(a.selectors_dir), Path(a.pca), units) if units else None
    for b in a.benches:
        f = out / b / f"timing_{b}.json"; t0 = time.time()
        if f.exists():
            parts.append({"part": b, "status": "kept from an earlier attempt of this bundle", "file": str(f), "sha256": RN.sha_file(f)}); continue
        main(common + ["--bench", b, "--configs-file", str(Path(a.configs_dir) / f"timing_{b}.txt"), "--out", str(out / b)], host=host, bank=bank, opener=opener)
        parts.append({"part": b, "status": "measured", "file": str(f), "sha256": RN.sha_file(f), "seconds": time.time() - t0})
    import os
    (out / BUNDLE_MARK).write_text(V.jdump({"parts": parts, "gpu": gpu, "host": __import__("socket").gethostname(), "job": os.environ.get("SLURM_JOB_ID"),
                                            "seconds": time.time() - t_all, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                            "rule": "owner 2026-10-13: every nodesumo01 unit of R7-40c in one job, back-to-back; one marker"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(bundle(sys.argv[2:]) if len(sys.argv) > 1 and sys.argv[1] == "bundle" else main())
