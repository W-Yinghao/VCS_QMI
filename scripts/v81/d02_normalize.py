"""V8.1 D00 / D02 normalisation: every existing per-image score -> the package row schema (FULL_PLAN_CN "标准逐图行格式").

Inputs (all pre-existing; no label beyond the development roles is read):
  * V8 formal units (24 = 8 arms x seeds 0-2) and tuning units (24 = 8 arms x 3 lr at seed 17):
    pred/images_{DEV,CAL_SELECT,CAL_PROB}.jsonl -> reference/adapt_v8.py (unchanged) on a derived manifest whose
    `component_id` = V7 `prompt_group` (the near-duplicate component used for the role split; the package prefers it as cluster).
    The exported q are float32 softmax outputs whose sum differs from 1 by up to ~2e-7, which the package simplex check
    (atol 1e-8) rejects; a pre-pass renormalises q (q / sum q), recomputes u = tails(q) and keeps the exported mu as the
    score.  Every deviation is measured and stored in the unit's prenorm meta (max |sum q - 1|, |u - tails(q)|, |mu - q.L|).
  * External systems on DEV (V7X-30 / V7X-40 files; score_kind native_score unless the system is a 1-5 regressor whose
    outputs all lie in [1, 5]); B0 is the released FGA model trained on all EvalMuse TRAIN (contains DEV) -> name says exposed.
  * V7 formal units (N0 native L1, M0 / M1 / M2 semantic; readout O) seeds 0-2 -> pair with the same-seed V8 arms.
  * Kernel bridges (pca0 / 64 / 256 x KDE / Nystrom / RuLSIF; raw mu = 1 + sum u_raw as score, no q / u carried).
  * Label-free exports: GenAI-Rank (pool = prompt x generator, official candidate order) and TIFA160 (pool = prompt) for the
    24 formal units; label fields are absent by construction (GenAI-Bench pools are NOT built: package D00).
Outputs: /projects/EEG-foundation-model/yinghao/outputs_v81/normalized/** (rows + .meta.json) and combined files under
normalized/combined/.  Existing outputs are never overwritten (skip + log).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v81_common as C  # noqa: E402

core = C.load_core()
TOL = 2e-5


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------------------------------------------- manifest
def derived_manifest():
    out = C.NORM / "manifest_component.jsonl"
    if out.exists():
        return out
    rows = C.manifest()
    for r in rows:
        r["component_id"] = r["prompt_group"]
    C.write_jsonl(out, rows)
    C.jdump({"source": str(C.MANIFEST), "source_sha256": C.sha256(C.MANIFEST), "n_rows": len(rows),
             "component_id": "copied from prompt_group (V7 near-duplicate component used for the role split; manifest has no component_id)"},
            out.with_suffix(".meta.json"))
    return out


def dev_candidates(role="DEV"):
    """image -> (prompt_id, candidate_index) with adapt_v8.py's order: lexical image id inside the prompt."""
    recs = {r["image_id"]: r for r in C.manifest(role) if r.get("overall_votes")}
    per = defaultdict(list)
    for i, r in recs.items():
        per[str(r["prompt_id"])].append(i)
    order = {i: k for ids in per.values() for k, i in enumerate(sorted(ids, key=str))}
    return recs, order


# ------------------------------------------------------------------------------------------------------------------- prenorm
def prenorm(pred_path: Path, out_path: Path, score_key="mu"):
    """Renormalise q, recompute u, keep the exported score; measure the deviations.  Returns the meta dict."""
    meta_path = out_path.with_suffix(".meta.json")
    if out_path.exists():
        return C.jload(meta_path)
    rows = C.read_jsonl(pred_path)
    q = np.array([r["q"] for r in rows], float)
    u = np.array([r["u"] for r in rows], float)
    mu = np.array([r[score_key] for r in rows], float)
    qn = q / q.sum(1, keepdims=True)
    un = C.tails(qn)
    dev = {"max_abs_sum_q_minus_1": float(np.abs(q.sum(1) - 1).max()), "max_abs_u_minus_tails_q": float(np.abs(u - C.tails(q)).max()),
           "max_abs_u_minus_tails_qn": float(np.abs(u - un).max()), "max_abs_score_minus_qn_dot_levels": float(np.abs(mu - qn @ C.LEVELS).max()),
           "max_abs_score_minus_1_plus_sum_u": float(np.abs(mu - (1 + u.sum(1))).max()), "u_monotone": bool((np.diff(u, axis=1) <= 1e-10).all())}
    if dev["max_abs_u_minus_tails_qn"] > TOL or dev["max_abs_score_minus_qn_dot_levels"] > TOL:
        raise SystemExit(f"{pred_path}: export inconsistency above {TOL}: {dev}")
    out = []
    for r, qq, uu in zip(rows, qn, un):
        rr = dict(r)
        rr["q"] = [float(x) for x in qq]
        rr["u"] = [float(x) for x in uu]
        out.append(rr)
    C.write_jsonl(out_path, out)
    meta = {"source": str(pred_path), "source_sha256": C.sha256(pred_path), "n_rows": len(rows), "deviations": dev,
            "rule": "q renormalised by its sum (float32 export), u recomputed as tails(q), exported score kept"}
    C.jdump(meta, meta_path)
    return meta


# ------------------------------------------------------------------------------------------------------------------ V8 units
def adapt_unit(unit: Path, method: str, seed: int, role: str, out_dir: Path, mani: Path):
    out = out_dir / f"rows_{role}.jsonl"
    if out.exists():
        return out
    log_ = C.unit_log(unit)
    pre = out_dir / f"prenorm_{role}.jsonl"
    prenorm(unit / "pred" / f"images_{role}.jsonl", pre)
    cmd = [C.PYTHON, str(C.REF / "adapt_v8.py"), "--pred", str(pre), "--public-train-manifest", str(mani), "--role", role,
           "--method", method, "--seed", str(seed), "--checkpoint-id", C.ckpt_id(log_), "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"adapt_v8 failed for {unit} {role}: {r.stderr[-2000:]}")
    return out


def v8_units(kind: str, mani: Path):
    if kind == "fit":
        specs = [(C.fit_unit(a, s), a, s, C.ROLES_LABELLED) for a in C.ARMS for s in C.SEEDS]
    else:
        specs = [(C.tune_unit(a, lr), f"{a}-lr{lr}", C.TUNE_SEED, ["DEV"]) for a in C.ARMS for lr in C.TUNE_LRS]
    for unit, method, seed, roles in specs:
        if not (unit / "done.json").exists():
            raise SystemExit(f"unit not complete: {unit}")
        out_dir = C.NORM / f"v8{kind}" / unit.name
        for role in roles:
            adapt_unit(unit, method, seed, role, out_dir, mani)
        log(f"[v8{kind}] {unit.name} ok")


# ----------------------------------------------------------------------------------------------------------------- externals
def jl_noheader(p):
    return [r for r in C.read_jsonl(p) if "header" not in r]


def external_systems():
    """name -> (scores {image_id: float}, kind_rule, seed, checkpoint_id, source path)."""
    X = C.V7_OUT / "x30_dev"
    systems = []

    def native(name, path, seed):
        d = C.jload(path)
        systems.append((name, {r["img_path"]: float(r["score_result"]) for r in d}, "regressor_1_5", seed, "released" if seed == -1 else "selected", str(path)))

    native("B0_FGA_released_exposed", C.V7_OUT / "native_eval/B0_released/native_eval_DEV.json", -1)
    for s in C.SEEDS:
        native("V7-N0_native_l1", C.V7_OUT / f"units/V7-FIT-N0_native_l1-s{s}/native_eval_DEV.json", s)
    for tag in ("M0_semantic_l1", "M1_semantic_logistic", "M2_semantic_vcs"):
        for s in C.SEEDS:
            p = C.V7_OUT / f"units/V7-FIT-{tag}-s{s}/pred/images_DEV.jsonl"
            systems.append((f"V7-{tag}", {r["image_id"]: float(r["O"]) for r in C.read_jsonl(p)}, "regressor_1_5", s, "selected", str(p)))
    for tag, f in (("B1_VQAScore", "b1_vqascore_clipflant5xxl.jsonl"), ("B2_VQAScore_Qwen25VL7B", "b2_vqascore_qwen25vl7b.jsonl"),
                   ("B5_HPSv3", "b5_hpsv3.jsonl"), ("B4T_TIFA", "tifa_scores_0of1.jsonl"), ("B4D_DSG", "dsg_scores_0of1.jsonl"),
                   ("B3_PNVQA_own", "pnvqa_scores_own_0of1.jsonl"), ("B3_PNVQA_evalmuse", "pnvqa_scores_evalmuse_0of1.jsonl"),
                   ("SANITY_CLIPScore", "sanity_clipscore.jsonl")):
        p = X / f
        systems.append((tag, {r["image_id"]: float(r["score"]) for r in jl_noheader(p)}, "native", -1, "released", str(p)))
    return systems


def kernel_systems():
    systems = []
    for pc in ("pca0", "pca64", "pca256"):
        for route in ("K-KDE", "K-NYSTROM", "K-RULSIF"):
            p = C.KERNELS / pc / route / "pred_DEV.jsonl"
            systems.append((f"{route}-{pc}", {r["image_id"]: float(r["mu"]) for r in C.read_jsonl(p)}, "regressor_1_5", -1, "fit", str(p)))
    return systems


def rows_from_scores(name, scores, kind_rule, seed, ckpt, recs, order):
    missing = [i for i in recs if i not in scores]
    extra = [i for i in scores if i not in recs]
    vals = np.array([scores[i] for i in recs if i in scores], float)
    if kind_rule == "regressor_1_5" and len(vals) and vals.min() >= 1 - 1e-5 and vals.max() <= 5 + 1e-5:
        kind = "rating_mean"
    else:
        kind = "native_score"
    rows = []
    for i in sorted(recs, key=str):
        if i not in scores:
            continue
        r = recs[i]
        rows.append({"dataset": C.DATASET, "role": "DEV", "method": name, "seed": seed, "checkpoint_id": ckpt, "prompt_id": str(r["prompt_id"]),
                     "pool_id": str(r["prompt_id"]), "image_id": str(i), "candidate_index": order[i], "cluster_id": str(r["prompt_group"]),
                     "score": float(scores[i]), "score_kind": kind, "votes": r["overall_votes"]})
    meta = {"method": name, "seed": seed, "n_rows": len(rows), "n_missing_dev_images": len(missing), "n_extra_images": len(extra),
            "score_kind": kind, "score_min": float(vals.min()) if len(vals) else None, "score_max": float(vals.max()) if len(vals) else None,
            "kind_rule": kind_rule}
    return rows, meta


def externals(which="external"):
    recs, order = dev_candidates("DEV")
    systems = external_systems() if which == "external" else kernel_systems()
    skipped = []
    for name, scores, rule, seed, ckpt, src in systems:
        out_dir = C.NORM / which / f"{name}__s{seed}"
        out = out_dir / "rows_DEV.jsonl"
        if out.exists():
            continue
        rows, meta = rows_from_scores(name, scores, rule, seed, ckpt, recs, order)
        meta["source"] = src
        meta["source_sha256"] = C.sha256(src)
        if meta["n_missing_dev_images"]:
            meta["status"] = "skipped_incomplete_coverage"
            skipped.append(name)
            C.jdump(meta, out_dir / "SKIPPED.meta.json")
            log(f"[{which}] {name} s{seed}: {meta['n_missing_dev_images']} DEV images missing -> skipped (no silent intersection)")
            continue
        rows = core.validate_rows(rows, labelled=True)
        C.write_jsonl(out, rows)
        C.jdump(meta, out.with_suffix(".meta.json"))
        log(f"[{which}] {name} s{seed}: {len(rows)} rows, {meta['score_kind']} [{meta['score_min']:.3f}, {meta['score_max']:.3f}]")
    return skipped


# ---------------------------------------------------------------------------------------------------------------- label-free
LABEL_FREE = {"genai_rank": ("GenAI-Rank", "GENAI_RANK", "pred/genai_rank.jsonl", "prompt_x_generator"),
              "tifa160": ("TIFA160", "TIFA160", "pred/tifa160.jsonl", "prompt")}


def labelfree():
    for key, (dataset, role, rel, pooling) in LABEL_FREE.items():
        out = C.NORM / "labelfree" / f"{key}_v8fit.jsonl"
        if out.exists():
            continue
        allrows, sources = [], []
        for a in C.ARMS:
            for s in C.SEEDS:
                unit = C.fit_unit(a, s)
                pre = C.NORM / "v8fit" / unit.name / f"prenorm_{key}.jsonl"
                meta = prenorm(unit / rel, pre, score_key="score")
                sources.append({"unit": unit.name, **meta})
                ck = C.ckpt_id(C.unit_log(unit))
                for r in C.read_jsonl(pre):
                    pool = f"{r['prompt_id']}|{r['generator_id']}" if pooling == "prompt_x_generator" else str(r["prompt_id"])
                    allrows.append({"dataset": dataset, "role": role, "method": a, "seed": s, "checkpoint_id": ck, "prompt_id": str(r["prompt_id"]),
                                    "pool_id": pool, "image_id": str(r["image_id"]), "candidate_index": int(r["candidate_index"]),
                                    "cluster_id": str(r["prompt_id"]), "score": float(r["score"]), "score_kind": "rating_mean",
                                    "q": r["q"], "u": r["u"], "generator_id": r["generator_id"]})
        allrows = core.validate_rows(allrows, labelled=False)
        C.write_jsonl(out, allrows)
        pools = {(r["method"], r["seed"], r["pool_id"]) for r in allrows}
        C.jdump({"dataset": dataset, "role": role, "pooling": pooling, "n_rows": len(allrows), "n_models": 24,
                 "n_pools_per_model": len(pools) // 24, "label_fields": "none (validated label-free)", "sources": sources},
                out.with_suffix(".meta.json"))
        log(f"[labelfree] {key}: {len(allrows)} rows, {len(pools)//24} pools per model")


# ------------------------------------------------------------------------------------------------------------------- combine
def combine():
    cmb = C.NORM / "combined"
    cmb.mkdir(parents=True, exist_ok=True)

    def build(name, parts, labelled):
        out = cmb / f"{name}.jsonl"
        if out.exists():
            return
        rows, src = [], []
        for p in parts:
            rr = C.read_jsonl(p)
            rows.extend(rr)
            src.append({"path": str(p), "sha256": C.sha256(p), "n_rows": len(rr)})
        rows = core.validate_rows(rows, labelled=labelled)
        C.write_jsonl(out, rows)
        models = sorted({C.model_name(r) for r in rows})
        C.jdump({"n_rows": len(rows), "n_models": len(models), "models": models, "labelled": labelled, "sources": src}, out.with_suffix(".meta.json"))
        log(f"[combine] {name}: {len(rows)} rows, {len(models)} models")

    fit = [C.NORM / "v8fit" / C.fit_unit(a, s).name for a in C.ARMS for s in C.SEEDS]
    for role in C.ROLES_LABELLED:
        build(f"{role.lower()}_v8fit", [d / f"rows_{role}.jsonl" for d in fit], True)
    build("dev_v8tune", [C.NORM / "v8tune" / C.tune_unit(a, lr).name / "rows_DEV.jsonl" for a in C.ARMS for lr in C.TUNE_LRS], True)
    ext = sorted(p for p in (C.NORM / "external").glob("*/rows_DEV.jsonl"))
    ker = sorted(p for p in (C.NORM / "kernels").glob("*/rows_DEV.jsonl"))
    build("dev_all", [d / "rows_DEV.jsonl" for d in fit] + ext + ker, True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--what", default="all", choices=["all", "v8fit", "v8tune", "external", "kernels", "labelfree", "combine"])
    a = ap.parse_args()
    C.NORM.mkdir(parents=True, exist_ok=True)
    mani = derived_manifest()
    if a.what in ("all", "v8fit"):
        v8_units("fit", mani)
    if a.what in ("all", "v8tune"):
        v8_units("tune", mani)
    if a.what in ("all", "external"):
        externals("external")
    if a.what in ("all", "kernels"):
        externals("kernels")
    if a.what in ("all", "labelfree"):
        labelfree()
    if a.what in ("all", "combine"):
        combine()
    log("normalisation done")


if __name__ == "__main__":
    main()
