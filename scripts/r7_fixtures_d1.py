"""r7 R7-20 D1 — host + one official token-pruning baseline on fixed small fixtures (r7 package directions/D1_token_evidence, D1-00).
Host LLaVA-1.5-7B (official LLaVA code); baseline SparseVLM (official code, Gumpest/SparseVLMs a9e7142, default USE_VERSION=1_0) at
RETAIN_TOKN 192 and 64 of 576 visual tokens.  Fixtures (fixed ids, grouped by image): 200 GQA val-balanced, 200 TextVQA val, 300 POPE-adversarial
(50 images; binary items for the finite-risk check).  Paper numbers are on GQA testdev / full TextVQA val / 3-category POPE, so they are references,
not targets.
    build : python scripts/r7_fixtures_d1.py build --out OUT                       (fixtures + weights_only check of the checkpoint shards)
    run   : python scripts/r7_fixtures_d1.py run --out OUT --config full|sparsevlm [--retain 192] [--fixtures gqa textvqa pope]
    score : python scripts/r7_fixtures_d1.py score --out OUT                       (official metrics, timing split, finite h vs d residual)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7_vl_common as V  # noqa: E402

SPARSEVLM = Path("/projects/EEG-foundation-model/yinghao/third_party/SparseVLMs")                 # Gumpest/SparseVLMs a9e7142 (Apache-2.0)
EVAL_FILES = ("scripts/convert_gqa_for_eval.py", "llava/eval/m4c_evaluator.py", "llava/eval/eval_pope.py",
              "llava/model/language_model/score.py", "llava/model/language_model/modelling_sparse_llama.py")


def cmd_build(a):
    out = Path(a.out); man = V.write_fixtures(out / "fixture")
    import torch
    chk = {}
    for f in sorted(V.LLAVA_DIR.glob("pytorch_model-*.bin")):
        t0 = time.time()
        try:                                                                           # zip-format checkpoints map lazily; legacy ones load fully
            sd = torch.load(str(f), map_location="cpu", weights_only=True, mmap=True); how = "weights_only+mmap"
        except (RuntimeError, ValueError):
            sd = torch.load(str(f), map_location="cpu", weights_only=True); how = "weights_only"
        chk[f.name] = {"load": how, "tensors": len(sd), "seconds": round(time.time() - t0, 1)}; del sd
    man["checkpoint_weights_only_check"] = chk
    (out / "fixture" / "fixture_manifest.json").write_text(V.jdump(man, indent=1))
    print(V.jdump(man, indent=1))
    return 0


def make_host(a, out):
    if a.config == "qwen25vl":                                                         # D1 plan's named host; lmms-eval style prompts (no OCR tokens)
        return V.QwenHost(out), (lambda it: it["question"] + V.SHORT)
    return V.Host(out, sparse=a.config == "sparsevlm", retain=a.retain, version=a.version), (lambda it: it["prompt"])


def cmd_run(a, host_factory=make_host):
    """One host configuration over the fixtures.  Every answer is appended and flushed as it is produced; run_<tag>.json is rewritten after
    each fixture; a rerun skips the (fixture, qid) pairs already on disk, so a late failure never discards finished generations."""
    out = Path(a.out); d = out / "d1"; d.mkdir(parents=True, exist_ok=True)
    tag = getattr(a, "tag", None) or {"full": "full", "sparsevlm": f"sparsevlm{a.retain}", "qwen25vl": "qwen25vl7b"}[a.config]
    pred_path, run_path = d / f"pred_{tag}.jsonl", d / f"run_{tag}.json"
    done = {(r["fixture"], r["qid"]): r for r in V.read_jsonl(pred_path)}
    if pred_path.exists():                                                             # drop a torn tail before appending
        pred_path.write_text("".join(V.jdump(r) + "\n" for r in done.values()))
    info = json.loads(run_path.read_text()) if run_path.exists() else {"fixtures": {}}
    fixtures = {name: V.load_fixture(out / "fixture", name) for name in a.fixtures}
    todo = {name: [it for it in items if (name, it["qid"]) not in done] for name, items in fixtures.items()}
    if not any(todo.values()) and all(name in info["fixtures"] for name in fixtures):
        print(f"[r7_d1] {tag}: all {sum(map(len, fixtures.values()))} items already on disk", flush=True)
        return 0
    host, prompt_of = host_factory(a, out); torch = host.torch
    info.update({"config": tag, "retain": a.retain if a.config == "sparsevlm" else None, "use_version": a.version if a.config == "sparsevlm" else None,
                 "load_seconds": host.load_seconds, "stage_notes": getattr(host, "stage_notes", []), "env": V.env_info(),
                 "yes_ids": host.yes_ids, "no_ids": host.no_ids})
    with open(pred_path, "a") as fh:
        for name, items in fixtures.items():
            rest = todo[name]
            if not rest and name in info["fixtures"]:
                continue
            for it in items[: a.warmup]:                                               # warm-up, not recorded
                host.answer(prompt_of(it), it["image_file"])
            host.reset_sparse_counters(); V.cuda_peak_reset(torch); t0 = time.time()
            for it in rest:
                r = host.answer(prompt_of(it), it["image_file"])
                fh.write(V.jdump({"fixture": name, "qid": it["qid"], "image_id": it["image_id"], **r}) + "\n"); fh.flush()
            info["fixtures"][name] = {"items": len(items), "items_timed": len(rest), "resumed_items": len(items) - len(rest),
                                      "wall_seconds": time.time() - t0, "peak_mem_gb": V.cuda_peak_gb(torch), **host.sparse_counters()}
            run_path.write_text(V.jdump(info, indent=1))
            print(f"[r7_d1] {tag} {name}: {len(rest)} new items ({len(items) - len(rest)} resumed) in {time.time() - t0:.0f}s", flush=True)
    return 0


def cmd_score(a):
    import numpy as np
    sys.path.insert(0, str(V.REF_DIR)); import vcs_vl_reference as REF  # noqa: E402
    out = Path(a.out); d = out / "d1"; fx = {n: {x["qid"]: x for x in V.load_fixture(out / "fixture", n)} for n in ("gqa", "textvqa", "pope")}
    tv = V.textvqa_scorer(); preds = {}
    for f in sorted(d.glob("pred_*.jsonl")):
        if (d / f"run_{f.stem[5:]}.json").exists():                                    # completed runs only (another job may be writing)
            preds[f.stem[5:]] = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    runs = {t: json.loads((d / f"run_{t}.json").read_text()) for t in preds}
    S = {"configs": list(preds), "runs": runs, "metrics": {}, "timing": {}, "finite_risk": {}, "paired_vs_full": {},
         "fixture_manifest": json.loads((out / "fixture" / "fixture_manifest.json").read_text()),
         "evaluator_sha256": {f: hashlib.sha256((SPARSEVLM / f).read_bytes()).hexdigest() for f in EVAL_FILES},
         "prediction_sha256": {f"pred_{t}.jsonl": hashlib.sha256((d / f"pred_{t}.jsonl").read_bytes()).hexdigest() for t in preds}}

    def score(name, r):
        it = fx[name][r["qid"]]
        if name == "gqa":
            return V.gqa_correct(r["text"], it["answer"])
        if name == "textvqa":
            return tv(r["text"], it["answers"])
        return float(V.pope_yes(r["text"]) == it["label"])
    per = {}
    for t, rows in preds.items():
        for name in fx:
            rr = [r for r in rows if r["fixture"] == name]
            if not rr:
                continue
            s = np.array([score(name, r) for r in rr]); g = [r["image_id"] for r in rr]; per[(t, name)] = {r["qid"]: (v, r) for v, r in zip(s, rr)}
            m = {"n": len(rr), "metric": "accuracy" if name != "textvqa" else "soft_accuracy", "value": 100 * float(s.mean()),
                 "ci95_image_bootstrap": [100 * x for x in V.cluster_bootstrap(s, g)]}
            if name == "pope":
                m.update({k: (100 * v if isinstance(v, float) else v) for k, v in V.pope_metrics([V.pope_yes(r["text"]) for r in rr], [fx[name][r["qid"]]["label"] for r in rr]).items()})
            if name == "gqa":
                b = [i for i, r in enumerate(rr) if fx[name][r["qid"]]["binary"]]; m["binary_items"] = len(b); m["binary_accuracy"] = 100 * float(s[b].mean()) if b else None
            S["metrics"].setdefault(t, {})[name] = m
            tim = {k: [r[k] for r in rr] for k in ("vision_ms", "selector_ms", "prefill_ms", "decode_ms", "wall_ms", "decode_steps", "prompt_tokens",
                                                   "new_tokens", "visual_tokens") if k in rr[0]}
            S["timing"].setdefault(t, {})[name] = {k: {"mean": float(np.mean(v)), "median": float(np.median(v))} for k, v in tim.items()}
    for t in preds:
        if not t.startswith("sparsevlm"):                                              # pruning vs the same host only
            continue
        for name in fx:
            if (t, name) not in per or ("full", name) not in per:
                continue
            A, B = per[(t, name)], per[("full", name)]; q = [k for k in B if k in A]
            diff = np.array([A[k][0] - B[k][0] for k in q]); g = [B[k][1]["image_id"] for k in q]
            S["paired_vs_full"].setdefault(t, {})[name] = {"n": len(q), "delta_points": 100 * float(diff.mean()),
                                                         "ci95_image_bootstrap": [100 * x for x in V.cluster_bootstrap(diff, g)],
                                                         "answer_changed_fraction": float(np.mean([A[k][1]["text"] != B[k][1]["text"] for k in q]))}
            # finite-model check on binary items: C = label, b = P_full(yes), a = P_pruned(yes) (yes/no mass renormalised)
            qb = [k for k in q if fx[name][k]["binary"]]
            if not qb:
                continue
            c = np.array([fx[name][k]["label"] for k in qb], float)
            pa = np.array([A[k][1]["p_yes"] / max(A[k][1]["p_yes"] + A[k][1]["p_no"], 1e-12) for k in qb])
            pb = np.array([B[k][1]["p_yes"] / max(B[k][1]["p_yes"] + B[k][1]["p_no"], 1e-12) for k in qb])
            h, dd, res = REF.finite_risk_delta(c, pa, pb); gb = [B[k][1]["image_id"] for k in qb]
            S["finite_risk"].setdefault(t, {})[name] = {
                "n": len(qb), "identity_max_abs_error": float(np.max(np.abs(h - dd - res))),
                "mean_h": float(h.mean()), "mean_h_ci95": V.cluster_bootstrap(h, gb), "mean_d": float(dd.mean()), "mean_d_ci95": V.cluster_bootstrap(dd, gb),
                "mean_residual": float(res.mean()), "mean_residual_ci95": V.cluster_bootstrap(res, gb),
                "frac_h_negative": float(np.mean(h < 0)), "frac_h_exceeds_d": float(np.mean(h > dd)),
                "corr_h_d": float(np.corrcoef(h, dd)[0, 1]) if h.std() > 0 and dd.std() > 0 else None,
                "brier_full": float(np.mean((c - pb) ** 2)), "brier_pruned": float(np.mean((c - pa) ** 2)),
                "yes_no_mass_full_mean": float(np.mean([B[k][1]["p_yes"] + B[k][1]["p_no"] for k in qb])),
                "yes_no_mass_pruned_mean": float(np.mean([A[k][1]["p_yes"] + A[k][1]["p_no"] for k in qb]))}
    (d / "d1_summary.json").write_text(V.jdump(S, indent=1))
    print(V.jdump({"metrics": S["metrics"], "paired_vs_full": S["paired_vs_full"], "finite_risk": S["finite_risk"]}, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--out", required=True)
    r = sub.add_parser("run"); r.add_argument("--out", required=True); r.add_argument("--config", choices=["full", "sparsevlm", "qwen25vl"], required=True)
    r.add_argument("--retain", type=int, default=192, choices=[192, 128, 96, 64]); r.add_argument("--version", default="1_0", choices=["1_0", "2_0"])
    r.add_argument("--fixtures", nargs="+", default=["gqa", "textvqa", "pope"]); r.add_argument("--warmup", type=int, default=3)
    r.add_argument("--tag", default=None, help="output tag override, e.g. a same-GPU timing rerun of the full host")
    s = sub.add_parser("score"); s.add_argument("--out", required=True)
    a = ap.parse_args()
    return {"build": cmd_build, "run": cmd_run, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
