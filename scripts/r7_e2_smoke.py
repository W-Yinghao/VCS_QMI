"""r7 E2 GPU smoke (the coordinator's step before the freeze; code checks only — no FIT_FULL teacher, cache, selector or evaluation unit).
  gpu --out S        vl_baselines env, L40S.  ONE host load: (1) the frozen R7-40a launch probe (r7_d1_40_teacher.cmd_probe on the R7-20 GQA fixture;
                     S4 rate:0.9 and S1 same_gpu as frozen at D40a-1 / D40a-2) -> S/probe/probe_report.json, the E2 bundles' launch gate; (2) the GPU
                     replay of the reuse rule (E2-2, the analogue of D1-FULL's DF-2): the first chunk (16 images) of R7-40a's FIT-000 and FIT-001
                     recomputed with the frozen teacher (run_items with the bundle settings: prefill batches <= 16,384 tokens / 64 sequences) and
                     compared with the stored records -> S/reuse_replay.json; 'pass' iff every record is bit-identical AND the GPU is an L40S (the
                     class of every R7-40a record); on another GPU the differences are reported only
  memsmoke --out S   main env, RTX6000PRO / H100.  The frozen R7-40b trainer (r7_d1_40_selectors.train40: phase A with full CAL evaluations, phase B,
                     soft / hard gap; then dynamic_tau40) on FIT_FULL-size synthetic tensors -> S/<gpu>[_noexp]/memsmoke.json: static footprint, peak
                     allocated / reserved per objective, step costs and the unit-time estimate.  Shapes: the R7-40b cache's per-image token counts and
                     merged grids (the first 10,000 FIT_FULL images) cycled to 70,256 images; 280,787 questions (3-4 per image); every question at 40
                     tokens (worst case; the 10k maximum is 27); rank tables padded to the 1,280-token maximum; CAL = R7-40b's CAL shapes (2,000 images,
                     7,989 questions); one embedding-table copy per role on the GPU, as the trainer.  PYTORCH_CUDA_ALLOC_CONF is set by the wrapper.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "r7" / "package_r7_20261011" / "reference"))
import r7_vl_common as V  # noqa: E402

OUT = Path("/home/infres/yinwang/CS_QMI/outputs")
E2 = OUT / "R7_E2"
R40 = OUT / "R7_D1" / "r7_40"
DF_DATA = OUT / "R7_D1_FULL" / "data"
TEACHER_GPU = "NVIDIA L40S"
W_FIT_FULL = (1.0099743178401088, 0.9902207645648187)          # class weights from FIT_FULL (D1-FULL manifest meta), the trainer's E2 weights
VOCAB = 152064                                                  # Qwen2.5-VL input-embedding rows (embed_table.pt [152064, 3584] bf16)


# ------------------------------------------------------------------------------------------------------------------------------ replay
def T(pair):
    return 2.0 * pair[0] / max(pair[0] + pair[1], 1e-12) - 1.0


def compare_records(new, old):
    """new: run_items output (tuples), old: the stored JSONL records (lists).  Bit identity after the JSON round trip the bundle applies."""
    import r7_d1_40_masks as M4
    new = [json.loads(V.jdump(r)) for r in new]; ident = sum(1 for a, b in zip(new, old) if b is not None and a == b); dP, dT, qs = [], [], 0
    for a, b in zip(new, old):
        if b is None:
            continue
        pairs = [(a["full"], b["full"]), (a["text"], b["text"])] + [(a["masked"][m][k], b["masked"][m][k]) for m in M4.MECHANISMS for k in M4.FKEYS]
        dP += [abs(x[0] - y[0]) for x, y in pairs]; dT += [abs(T(x) - T(y)) for x, y in pairs]; qs += a["qsim_order"] == b["qsim_order"]
    missing = sum(b is None for b in old)
    return {"records": len(new), "missing_stored": missing, "bit_identical_records": ident, "qsim_orders_identical": qs,
            "max_abs_dP_yes": max(dP, default=0.0), "max_abs_dT": max(dT, default=0.0), "bit_identical": missing == 0 and ident == len(new)}


def replay(host, manifest_dir: Path, r40_teacher: Path, shards=("FIT-000", "FIT-001"), n_images=16, source=None, gpu_name=None):
    """The first chunk of each R7-40a shard recomputed with the frozen teacher (the bundle's first chunk: its first n_images images, all their
    questions in manifest order, run_items with the bundle defaults) and compared with the stored records."""
    import r7_d1_40_teacher as TE40
    plan = json.loads((Path(manifest_dir) / "d1_shards.json").read_text()); man = {}
    for x in (Path(manifest_dir) / "d1_manifest.jsonl").read_text().splitlines():
        if x.strip():
            r = json.loads(x)
            if r["role"] in ("FIT", "CAL"):
                man.setdefault(r["image_id"], []).append(r)
    src = source or TE40.ExtractedImageSource(); res = {"shards": {}, "gpu": gpu_name, "n_images": n_images}
    for s in shards:
        if not s.startswith(("FIT-", "CAL-")):
            raise SystemExit("the replay reads FIT / CAL records only")
        imgs = plan["shards"][s][:n_images]; items = [q for i in imgs for q in man[i]]
        stored = {r["qid"]: r for r in V.read_jsonl(Path(r40_teacher) / f"{s}.jsonl")}; done = json.loads((Path(r40_teacher) / f"{s}.done.json").read_text())
        new = TE40.run_items(host, items, {i: src.open(i) for i in imgs})
        res["shards"][s] = dict(compare_records(new, [stored.get(q["qid"]) for q in items]), stored_gpu=done.get("gpu"))
    S = res["shards"].values()
    res.update({"shards_checked": len(res["shards"]), "records": sum(v["records"] for v in S), "bit_identical": all(v["bit_identical"] for v in S),
                "max_abs_dP_yes": max((v["max_abs_dP_yes"] for v in S), default=0.0), "qsim_orders_identical": sum(v["qsim_orders_identical"] for v in S),
                "same_gpu_class": gpu_name == TEACHER_GPU and all(v["stored_gpu"] == TEACHER_GPU for v in S)})
    res["pass"] = bool(res["bit_identical"] and res["same_gpu_class"])
    res["rule"] = "pass iff every replayed record is bit-identical to R7-40a's and the replay ran on the R7-40a GPU class (L40S); otherwise reported only"
    return res


def cmd_gpu(a):
    import r7_d1_40_host as QH
    import r7_d1_40_teacher as TE40
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); host = QH.QwenPrefill(out); gpu = host.torch.cuda.get_device_name(0)
    rc = TE40.cmd_probe(Namespace(out=str(out / "probe"), fixture=str(TE40.FIXTURE), r720_preds=str(TE40.R720_QWEN), gen_batch=16,
                                  s4_rule="rate:0.9", s1_rule="same_gpu"), host)
    t0 = time.time(); rep = replay(host, Path(a.manifest_dir), Path(a.r40_teacher), tuple(a.shards), a.images, gpu_name=gpu)
    rep.update({"seconds": time.time() - t0, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "env": V.env_info(), "host_load_seconds": host.load_seconds})
    (out / "reuse_replay.json").write_text(V.jdump(rep, indent=1)); print(V.jdump({k: v for k, v in rep.items() if k != "env"}, indent=1))
    print(f"[r7_e2_smoke] probe rc {rc} (0 = pass), replay pass {rep['pass']}")
    return 0 if (rc == 0 and rep["pass"]) else (4 if rc else 5)


# ------------------------------------------------------------------------------------------------------------------------------ memory smoke
def synth(ntok_list, cols_list, n_img, n_q, q_tokens, nmax, emb, cfg, device, seed=0):
    """FIT_FULL-shaped TupleData40 (r7_d1_40_selectors, frozen): features generated on the device (the trainer copies the cache there), rank
    tables [n_q, 3, nmax] int16 (a rotation per mechanism: a valid ordering of each image's tokens, PAD_RANK on padding), k per fraction from
    each image's N, random targets.  Shapes, not values, set the memory."""
    import torch
    import r7_d1_40_masks as M4
    import r7_d1_40_selectors as S4
    g = np.random.default_rng(seed); ntok = np.resize(np.asarray(ntok_list, np.int64), n_img); cols = np.resize(np.asarray(cols_list, np.int64), n_img)
    if not (3 * n_img <= n_q <= 4 * n_img):
        raise SystemExit(f"{n_q} questions for {n_img} images: expected 3-4 per image")
    offsets = np.concatenate([[0], np.cumsum(ntok)[:-1]]); T_ = int(ntok.sum()); feats = torch.empty((T_, cfg.vis_dim), dtype=torch.float16, device=device)
    for s in range(0, T_, 1 << 22):
        feats[s:s + (1 << 22)].normal_()
    per = np.full(n_img, 4); per[: 4 * n_img - n_q] = 3; img_of = np.repeat(np.arange(n_img), per); Nq = torch.as_tensor(ntok[img_of], device=device)
    rank = torch.empty((n_q, 3, nmax), dtype=torch.int16, device=device); ar = torch.arange(nmax, device=device)
    for s in range(0, n_q, 20000):
        nq = Nq[s:s + 20000, None]
        for m in range(3):
            rank[s:s + 20000, m] = torch.where(ar[None, :] < nq, (ar[None, :] + 7 * m) % nq, torch.full_like(ar[None, :].expand(len(nq), -1), S4.PAD_RANK)).to(torch.int16)
    kk = np.stack([np.maximum(1, np.floor(f * ntok[img_of] + 0.5)).astype(np.int32) for f in M4.FRACTIONS], 1)
    C = np.where(g.uniform(size=n_q) < 0.5, 1, -1); h = g.uniform(-1, 1, size=(n_q, 12)).astype(np.float32); d = g.uniform(0, 1, size=(n_q, 12)).astype(np.float32)
    q_ids = g.integers(1, emb.shape[0], size=(n_q, q_tokens)); q_len = np.full(n_q, q_tokens)
    return S4.TupleData40(feats, offsets, ntok, cols, img_of, q_ids, q_len, emb, rank, kk, C, h, d, device, cfg.max_rows, cfg.max_cols)


def unit_hours(tm, steps_a=20000, steps_b=5000, eval_every=1000, cal_pairs=7989 * 12):
    a = tm["phase_a_train_s"] / max(tm["phase_a_steps"], 1); e = tm["eval_s"] / max(tm["eval_pairs"], 1); b = tm["phase_b_s"] / max(tm["phase_b_steps"], 1)
    return {"s_per_phase_a_step": a, "s_per_eval_pair": e, "s_per_phase_b_step": b, "unit_hours_estimate": (steps_a * a + steps_a // eval_every * cal_pairs * e + steps_b * b) / 3600}


def cmd_memsmoke(a):
    import torch
    import r7_d1_40_selectors as S4
    dev = "cuda"; gname = torch.cuda.get_device_name(0); conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    out = Path(a.out) / (gname.replace(" ", "_") + ("" if "expandable_segments:True" in conf else "_noexp")); out.mkdir(parents=True, exist_ok=True)
    fx, cx = (json.loads((Path(a.r40_cache) / f"images_{r}.json").read_text()) for r in ("FIT", "CAL"))
    cfg = S4.Cfg40(lr=3e-4, seed=0, steps_a=a.steps_a, steps_b=a.steps_b, eval_every=a.eval_every, warmup=min(10, a.steps_a))
    t0 = time.time(); emb = torch.randn(a.vocab, cfg.emb_dim, dtype=torch.bfloat16)
    cal = synth(cx["n"], [g[1] for g in cx["grid"]], len(cx["n"]), a.cal_questions, a.q_tokens, max(cx["n"]), emb, cfg, dev, seed=1)
    fit = synth(fx["n"], [g[1] for g in fx["grid"]], a.fit_images, a.fit_questions, a.q_tokens, a.nmax, emb, cfg, dev, seed=2)
    torch.cuda.synchronize(); static = torch.cuda.memory_allocated() / 2 ** 30
    res = {"gpu": gname, "gpu_total_gib": torch.cuda.get_device_properties(0).total_memory / 2 ** 30, "alloc_conf": conf, "setup_seconds": time.time() - t0,
           "static_gib": static, "fit": {"images": a.fit_images, "questions": fit.n, "tokens": int(fit.ntok.sum()), "q_tokens": a.q_tokens, "nmax": a.nmax},
           "cal": {"images": len(cx["n"]), "questions": cal.n, "tokens": int(cal.ntok.sum())}, "cfg": {"steps_a": a.steps_a, "eval_every": a.eval_every, "steps_b": a.steps_b},
           "objectives": {}, "env": V.env_info()}
    for obj in a.objectives:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); t = time.time()
        model, log = S4.train40(cfg, obj, fit, cal, W_FIT_FULL); tau = S4.dynamic_tau40(model, cal, cfg, 0.5); torch.cuda.synchronize()
        res["objectives"][obj] = {"peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30, "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2 ** 30,
                                  "seconds": time.time() - t, "evals": len(log["evals"]), "soft_hard_gap": log.get("soft_hard_gap"), "dynamic_tau_0.5": tau,
                                  **unit_hours(log["timing"])}
        print(json.dumps({obj: res["objectives"][obj]}, default=str), flush=True); del model
    P = res["objectives"].values()
    res["peak_allocated_gib_max"] = max(v["peak_allocated_gib"] for v in P); res["peak_reserved_gib_max"] = max(v["peak_reserved_gib"] for v in P)
    res["headroom_gib_on_this_gpu"] = res["gpu_total_gib"] - res["peak_reserved_gib_max"]
    (out / "memsmoke.json").write_text(V.jdump(res, indent=1)); print(V.jdump({k: v for k, v in res.items() if k not in ("env", "objectives")}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gpu"); g.add_argument("--out", default=str(E2 / "smoke")); g.add_argument("--manifest-dir", default=str(DF_DATA))
    g.add_argument("--r40-teacher", default=str(R40 / "teacher")); g.add_argument("--shards", nargs="+", default=["FIT-000", "FIT-001"]); g.add_argument("--images", type=int, default=16)
    m = sub.add_parser("memsmoke"); m.add_argument("--out", default=str(E2 / "smoke" / "memsmoke")); m.add_argument("--r40-cache", default=str(R40 / "cache"))
    m.add_argument("--fit-images", type=int, default=70256); m.add_argument("--fit-questions", type=int, default=280787); m.add_argument("--cal-questions", type=int, default=7989)
    m.add_argument("--q-tokens", type=int, default=40); m.add_argument("--nmax", type=int, default=1280); m.add_argument("--vocab", type=int, default=VOCAB)
    m.add_argument("--steps-a", type=int, default=30); m.add_argument("--eval-every", type=int, default=10); m.add_argument("--steps-b", type=int, default=20)
    m.add_argument("--objectives", nargs="+", default=["direct", "vcs", "logistic"])
    a = ap.parse_args(argv)
    return {"gpu": cmd_gpu, "memsmoke": cmd_memsmoke}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
