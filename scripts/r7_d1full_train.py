"""r7 R7-30 D1-FULL selector units = the FROZEN D1-20 trainer, unchanged (scripts/r7_d1_train.py train: 20,000 phase-A steps of 64 questions × 4
of their 12 D1-10 masks, AdamW with 1,000-step warm-up then cosine, CAL Spearman(r, h) over all CAL (question, mask) pairs every 1,000 steps with
the best state kept, 5,000 phase-B gate steps, CAL dynamic taus for mean kept fractions 0.25 / 0.5 / 0.75), run on the D1-FULL teacher records,
manifest (class weights from FIT_FULL) and feature cache.  The frozen trainer keeps every FIT_FULL feature on the GPU (~38.6 GiB fp16), so units
run on RTX6000PRO / H100 only (measured by `memsmoke`).  This wrapper adds bookkeeping only:
  train <obj> <lr> <seed> [--no-question] [--target d] [--root DF] [frozen trainer options] : the frozen trainer as a subprocess (same interpreter;
        it resumes bit-exactly from <tag>.resume.pt).  A unit whose <tag>.pt exists but whose log lacks the CAL taus (a stop between the frozen
        trainer's checkpoint write and its final log write, which a frozen rerun would report as complete) is finished with the frozen
        dynamic_tau on CAL.  The completion marker <tag>.complete.json is written only when <tag>.pt exists and the log carries the three taus.
  memsmoke [--n-img 70256 --n-q 281000 --q-len 40] : FIT_FULL-size synthetic tensors on the GPU and the frozen SEL.train (a few phase-A steps
        with full CAL evaluations, phase B, dynamic tau) at the worst-case question length: peak GPU memory and step times; plus the NFS read
        rate of the frozen D1-20 FIT cache and the frozen assemble() cost per FIT record (setup-time extrapolation).  Synthetic data only, except
        the frozen FIT records / cache read for the timing (never CAL / DEV).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r7_d1_plan20 as P20  # noqa: E402

OUT = Path("/home/infres/yinwang/CS_QMI/outputs")
D1, DF = OUT / "R7_D1", OUT / "R7_D1_FULL"
FRACTIONS = (0.25, 0.5, 0.75)


def has_taus(log):
    return isinstance(log, dict) and all(str(f) in (log.get("dynamic") or {}) for f in FRACTIONS)


def finish(root: Path, tag: str, dev=None):
    """The frozen r7_d1_train.main tail for a unit whose checkpoint exists: CAL dynamic taus (frozen SEL.dynamic_tau), coverage, env."""
    import torch
    import r7_d1_selectors as SEL
    import r7_d1_train as TR
    import r7_vl_common as V
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu"); sel = root / "selectors"
    ck = torch.load(sel / f"{tag}.pt", map_location=dev, weights_only=False); cfg = SEL.Cfg(**{k: (tuple(v) if k == "budgets" else v) for k, v in ck["cfg"].items()})
    model = SEL.Selector(cfg, ck["objective"]).to(dev); model.load_state_dict(ck["state_dict"])
    _, fit_cov = TR.role_records(root / "teacher", root / "data", "FIT"); cal_recs, cal_cov = TR.role_records(root / "teacher", root / "data", "CAL")
    cal = TR.tuple_data(cal_recs, root / "cache", "CAL", dev); log = json.loads((sel / f"{tag}.json").read_text())
    log["dynamic"] = {str(f): SEL.dynamic_tau(model, cal, cfg, f) for f in FRACTIONS}; log["coverage"] = {"FIT": fit_cov, "CAL": cal_cov}
    log["env"] = V.env_info(); log["finished_by"] = "r7_d1full_train finish: frozen SEL.dynamic_tau on CAL (the frozen trainer stopped after its checkpoint)"
    tmp = sel / f"{tag}.json.tmp"; tmp.write_text(V.jdump(log, indent=1)); os.replace(tmp, sel / f"{tag}.json")
    return log


def cmd_train(a, extra):
    root = Path(a.root); sel = root / "selectors"; sel.mkdir(parents=True, exist_ok=True)
    tag = P20.tag(a.objective, a.lr, a.seed, a.no_question, a.target == "d"); mk = sel / f"{tag}.complete.json"
    if mk.exists():
        print(f"[r7_d1full_train] {tag}: complete"); return 0
    pt, js = sel / f"{tag}.pt", sel / f"{tag}.json"
    if not pt.exists():
        cmd = [sys.executable, str(HERE / "r7_d1_train.py"), "train", "--objective", a.objective, "--lr", f"{a.lr:g}", "--seed", str(a.seed)]
        cmd += (["--no-question"] if a.no_question else []) + (["--target", "d"] if a.target == "d" else [])
        cmd += ["--teacher-dir", str(root / "teacher"), "--manifest-dir", str(root / "data"), "--cache", str(root / "cache"), "--out", str(sel)] + list(extra)
        print("[r7_d1full_train] " + " ".join(cmd), flush=True); rc = subprocess.call(cmd)
        if rc:
            return rc
    log = json.loads(js.read_text()) if js.exists() else None
    if log is None:
        raise SystemExit(f"{pt} exists without its log {js}")
    if not has_taus(log):
        print(f"[r7_d1full_train] {tag}: checkpoint without CAL taus -> finish", flush=True); log = finish(root, tag)
    import hashlib
    rec = {"tag": tag, "pt_sha256": hashlib.sha256(pt.read_bytes()).hexdigest(), "selected_step": log.get("selected_step"),
           "taus": {f: log["dynamic"][f]["tau"] for f in log["dynamic"]}, "finished_by": log.get("finished_by"), "gpu": (log.get("env") or {}).get("gpu"),
           "seconds": log.get("seconds"), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    tmp = sel / f"{tag}.complete.json.tmp"; tmp.write_text(json.dumps(rec, indent=1)); os.replace(tmp, mk)
    print(f"[r7_d1full_train] {tag}: marker {mk}"); return 0


def cmd_smokefinish(a):
    """Smoke of the finish path on a completed unit: drop its CAL taus and marker, rerun the wrapper (-> finish), compare the recomputed taus."""
    root = Path(a.root); sel = root / "selectors"; tag = P20.tag(a.objective, a.lr, a.seed, a.no_question, a.target == "d"); js = sel / f"{tag}.json"
    orig = json.loads(js.read_text()); taus0 = {f: orig["dynamic"][f]["tau"] for f in orig["dynamic"]}
    stripped = {k: v for k, v in orig.items() if k != "dynamic"}; js.write_text(json.dumps(stripped, indent=1)); (sel / f"{tag}.complete.json").unlink()
    rc = cmd_train(a, []); new = json.loads(js.read_text()); taus1 = {f: new["dynamic"][f]["tau"] for f in new["dynamic"]}
    rep = {"tag": tag, "rc": rc, "taus_before": taus0, "taus_after": taus1, "max_abs_tau_diff": max(abs(taus0[f] - taus1[f]) for f in taus0),
           "marker": (sel / f"{tag}.complete.json").exists(), "finished_by": new.get("finished_by"), "coverage": new.get("coverage"),
           "class_weights_used": "from " + str(root / "data" / "d1_manifest_meta.json")}
    rep["pass"] = rc == 0 and rep["marker"] and rep["max_abs_tau_diff"] <= 1e-6 and bool(rep["finished_by"])
    (root / "smoke_finish.json").write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))
    return 0 if rep["pass"] else 8


def cmd_memsmoke(a):
    if a.expandable_segments:                                                           # allocator layout only (no numerical effect); set before torch loads
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import resource

    import numpy as np
    import torch
    import r7_d1_selectors as SEL
    import r7_vl_common as V
    dev = "cuda"; g = torch.Generator(device=dev).manual_seed(0); rep = {"args": vars(a), "gpu": torch.cuda.get_device_name(0),
                                                                          "gpu_total_gib": torch.cuda.get_device_properties(0).total_memory / 2 ** 30}

    def data(n_img, n_q, L):
        f = torch.empty((n_img, 576, 512), dtype=torch.float16, device=dev)
        for c in range(0, n_img, 1024):
            f[c:c + 1024].normal_(generator=g)
        keep = torch.empty((n_q, 12, 576), dtype=torch.bool, device=dev); fr = torch.tensor([0.125, 0.25, 0.5, 0.75] * 3, device=dev)
        for c in range(0, n_q, 8192):
            keep[c:c + 8192] = torch.rand((min(8192, n_q - c), 12, 576), device=dev, generator=g) < fr[None, :, None]
        return SEL.TupleData(f, torch.randint(0, n_img, (n_q,), device=dev, generator=g), torch.randint(1, 32000, (n_q, L), device=dev, generator=g),
                             torch.full((n_q,), L, device=dev), emb, keep, torch.randint(0, 2, (n_q,), device=dev, generator=g) * 2 - 1,
                             torch.rand((n_q, 12), device=dev, generator=g) - 0.3, torch.rand((n_q, 12), device=dev, generator=g), dev)
    t0 = time.time(); emb = (torch.randn((32000, 4096), device=dev, generator=g) * 0.01).half()
    fit = data(a.n_img, a.n_q, a.q_len); cal = data(a.n_img_cal, a.n_q_cal, a.q_len); torch.cuda.synchronize()
    rep["static_allocated_gib"] = torch.cuda.memory_allocated() / 2 ** 30; rep["alloc_seconds"] = time.time() - t0; rep["runs"] = {}
    for obj in a.objectives:
        torch.cuda.reset_peak_memory_stats(); cfg = SEL.Cfg(steps_a=a.steps, steps_b=a.steps_b, eval_every=a.eval_every, seed=0, warmup=min(10, a.steps))
        t1 = time.time(); model, log = SEL.train(cfg, obj, fit, cal, (1.0, 1.0)); dyn = SEL.dynamic_tau(model, cal, cfg, 0.5); torch.cuda.synchronize()
        tm = log["timing"]; per = {"s_per_phase_a_step": tm["phase_a_train_s"] / max(tm["phase_a_steps"], 1), "s_per_eval_pair": tm["eval_s"] / max(tm["eval_pairs"], 1),
                                   "s_per_phase_b_step": tm["phase_b_s"] / max(tm["phase_b_steps"], 1)}
        full = SEL.Cfg(); per["unit_hours_estimate"] = (full.steps_a * per["s_per_phase_a_step"] + full.steps_a // full.eval_every * a.n_q_cal * 12 * per["s_per_eval_pair"]
                                                        + full.steps_b * per["s_per_phase_b_step"]) / 3600
        rep["runs"][obj] = {"peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30, "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2 ** 30,
                            "seconds": time.time() - t1, "per_unit": per, "dynamic_tau_0.5": dyn}
        del model; torch.cuda.empty_cache()
    rep["headroom_gib_vs_h100_nvl_93.1"] = 93.1 - max(r["peak_reserved_gib"] for r in rep["runs"].values())
    rep["fits_l40s_44.3"] = max(r["peak_reserved_gib"] for r in rep["runs"].values()) < 44.3
    del fit, cal; torch.cuda.empty_cache()
    if not a.skip_io:                                                                   # setup-time measurements on the frozen D1-20 FIT inputs
        f = D1 / "cache" / "feats_FIT.npy"; fd = os.open(f, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        t2 = time.time(); x = np.load(f); dt = time.time() - t2; rep["nfs_read"] = {"file": str(f), "gb": x.nbytes / 1e9, "seconds": dt, "mb_per_s": x.nbytes / 1e6 / dt}; del x
        import r7_d1_train as TR
        t3 = time.time(); recs, cov = TR.role_records(D1 / "teacher", D1 / "data", "FIT"); t_load = time.time() - t3
        idx = json.loads((D1 / "cache" / "images_FIT.json").read_text()); qid_ids = {r["qid"]: r["ids"] for r in V.read_jsonl(D1 / "cache" / "question_ids.jsonl")}
        t4 = time.time(); TR.assemble(recs, idx, qid_ids); t_asm = time.time() - t4
        rep["setup_frozen_fit"] = {"records": len(recs), "load_role_s": t_load, "assemble_s": t_asm, "s_per_record": (t_load + t_asm) / len(recs),
                                   "extrapolated_s_for_n_q": (t_load + t_asm) / len(recs) * a.n_q, "maxrss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20}
    rep["env"] = V.env_info(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rep["pytorch_cuda_alloc_conf"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    name = f"memsmoke_{rep['gpu'].replace(' ', '_')}{'_expandable' if a.expandable_segments else ''}.json"; (out / name).write_text(V.jdump(rep, indent=1)); print(V.jdump(rep, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("objective", choices=["vcs", "logistic", "direct"]); t.add_argument("lr", type=float); t.add_argument("seed", type=int)
    t.add_argument("--no-question", action="store_true"); t.add_argument("--target", choices=["h", "d"], default="h"); t.add_argument("--root", default=str(DF))
    sf = sub.add_parser("smokefinish"); sf.add_argument("objective", choices=["vcs", "logistic", "direct"]); sf.add_argument("lr", type=float); sf.add_argument("seed", type=int)
    sf.add_argument("--no-question", action="store_true"); sf.add_argument("--target", choices=["h", "d"], default="h"); sf.add_argument("--root", required=True)
    m = sub.add_parser("memsmoke"); m.add_argument("--out", default=str(DF / "smoke" / "memsmoke")); m.add_argument("--n-img", type=int, default=70256)
    m.add_argument("--n-q", type=int, default=281000); m.add_argument("--n-img-cal", type=int, default=2000); m.add_argument("--n-q-cal", type=int, default=7989)
    m.add_argument("--q-len", type=int, default=40); m.add_argument("--steps", type=int, default=30); m.add_argument("--steps-b", type=int, default=20)
    m.add_argument("--eval-every", type=int, default=15); m.add_argument("--objectives", nargs="+", default=["vcs", "direct"]); m.add_argument("--skip-io", action="store_true")
    m.add_argument("--expandable-segments", action="store_true")
    a, extra = ap.parse_known_args(argv)
    if a.cmd == "train":
        return cmd_train(a, extra)
    if extra:
        raise SystemExit(f"unknown options {extra}")
    return cmd_smokefinish(a) if a.cmd == "smokefinish" else cmd_memsmoke(a)


if __name__ == "__main__":
    sys.exit(main())
