"""V7X-50 — GenAI-Rank prediction freeze (label-free).  Converts every GenAI-Rank prediction source into the frozen scorer's format
({prompt_id, generator_id, image_id, candidate_index, score}; scripts/v7/v7_score.py rank --methods), validates each file against the label-free
universe (outputs/V7/data/genai_rank_inputs.jsonl: every (prompt, generator, image) exactly once, the inputs' candidate index, finite scores),
and writes a SHA-256 manifest of sources and converted files plus the scorer's methods.json.  It never opens human_rating.json.

    python -I scripts/v7x/x50_rank_freeze.py --out /projects/EEG-foundation-model/yinghao/outputs_v7/x50_rank_freeze_v1 \
        --report reports/V7/V7X_50_RANK_PREDICTION_FREEZE
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

B = Path("/projects/EEG-foundation-model/yinghao/outputs_v7"); X30 = B / "x30"; U = B / "units"; NE = B / "native_eval"
INPUTS = Path("/home/infres/yinwang/CS_QMI/outputs/V7/data/genai_rank_inputs.jsonl")
SEEDS = (0, 1, 2)


def M(arm, field):
    f = "E|O" if field == "E" else field                     # frozen deployment rule: E undefined (decomposition_failed) -> O, flagged
    return [(U / f"V7-FIT-{arm}-s{s}" / "pred" / "genai_rank.jsonl", "jsonl", f) for s in SEEDS]


# name -> (family, [one source list per seed]); a source list may hold several shards (concatenated)
METHODS = {
    "M2_E": ("v2_external_primary (ours); v2_attribution; v1_registered", [[x] for x in M("M2_semantic_vcs", "E")]),
    "B0_FGA_released_O": ("v2_external_primary; v1_registered", [[(NE / "B0_released" / "native_eval_rank.json", "native", "score_result")]]),
    "B1_VQAScore": ("v2_external_primary; v1_registered", [[(B / "baselines" / "B1_vqascore_clipflant5xxl_rank.jsonl", "jsonl", "score")]]),
    "B4T_TIFA": ("v2_external_primary", [[(X30 / "tifa_scores_0of1.jsonl", "jsonl", "score")]]),
    "B4D_DSG": ("v2_external_primary", [[(X30 / f"dsg_scores_{i}of3.jsonl", "jsonl", "score") for i in range(3)]]),
    "M1_E": ("v2_attribution; v1_registered", [[x] for x in M("M1_semantic_logistic", "E")]),
    "M2_O": ("secondary", [[x] for x in M("M2_semantic_vcs", "O")]),
    "M1_O": ("secondary", [[x] for x in M("M1_semantic_logistic", "O")]),
    "M0_E": ("secondary", [[x] for x in M("M0_semantic_l1", "E")]),
    "M0_O": ("secondary", [[x] for x in M("M0_semantic_l1", "O")]),
    "N0_O": ("secondary (native FGA trainer reproduction)", [[(NE / f"N0_s{s}" / "native_eval_rank.json", "native", "score_result")] for s in SEEDS]),
    "B2_VQAScore_Qwen25VL7B": ("secondary", [[(X30 / "b2_vqascore_qwen25vl7b_0of1.jsonl", "jsonl", "score")]]),
    "B3_PNVQA_own": ("secondary", [[(X30 / f"pnvqa_scores_own_{i}of2.jsonl", "jsonl", "score") for i in range(2)]]),
    "B3P_PNVQA_parser": ("secondary", [[(X30 / "pnvqa_scores_parser_0of1.jsonl", "jsonl", "score")]]),
    "B5_HPSv3": ("secondary", [[(X30 / "b5_hpsv3_0of1.jsonl", "jsonl", "score")]]),
    "CLIPScore_sanity": ("secondary (sanity row)", [[(X30 / "sanity_clipscore.jsonl", "jsonl", "score")]]),
}
for meth in ("N-VCS-F", "N-BCE-F", "N-DV", "N-NWJ"):
    src = [[(B / "x21" / "formal" / f"{meth}_s{s}" / "genai_rank.jsonl", "jsonl", "E")] for s in SEEDS]
    METHODS[f"X21_{meth}_E"] = ("secondary (V7X-20 common features; parsed pools only: X21 'no score' rule)", [[(p, k, "E!skip") for p, k, _ in x] for x in src])
    METHODS[f"X21_{meth}_E_neutralfill"] = ("secondary sensitivity (full denominator, neutral tie fill, first-index)", src)
for k in ("K-KDE-S", "K-NYSTROM", "K-RULSIF"):
    src = [[(B / "x21" / "kernels" / k / "genai_rank.jsonl", "jsonl", "E")]]
    METHODS[f"X21_{k}_E"] = ("secondary (V7X-20 common features; parsed pools only: X21 'no score' rule)", [[(p, kk, "E!skip") for p, kk, _ in x] for x in src])
    METHODS[f"X21_{k}_E_neutralfill"] = ("secondary sensitivity (full denominator, neutral tie fill, first-index)", src)


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


TIE_FILL = 0.0      # a method's undefined score: every candidate of the prompt gets the same value -> the official first-index rule picks candidate 0


def read_source(path: Path, kind: str, field: str, fills: list | None = None, flagged: list | None = None, fallbacks: list | None = None) -> list[dict]:
    """field 'E|O': E, with the frozen O fallback where E is undefined; 'E!skip': E, rows with undefined E dropped (X21 'no score');
    any other field: undefined values of a flagged failure are tie-filled (TIE_FILL)."""
    if kind == "native":
        rows = json.loads(path.read_text())
        return [{"prompt_id": str(r["prompt_id"]), "generator_id": r["generator_id"], "image_id": r["img_path"], "candidate_index": int(r["candidate_index"]),
                 "score": float(r[field])} for r in rows]
    out = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if "header" in r:
            continue
        base = field.split("|")[0].split("!")[0]; v = r[base]
        if flagged is not None and (r.get("decomposition_failed") or r.get("no_questions")):
            flagged.append((r["generator_id"], str(r["prompt_id"])))
        if v is None and field == "E|O":
            if not r.get("decomposition_failed"):
                raise SystemExit(f"{path}: E undefined without decomposition_failed at {r['image_id']}")
            v = r["O"]
            if fallbacks is not None:
                fallbacks.append((r["generator_id"], str(r["prompt_id"])))
        if v is None and field == "E!skip":
            continue
        if v is None:                      # only a flagged method failure may be tie-filled; anything else stops the freeze
            if not (r.get("decomposition_failed") or r.get("no_questions")):
                raise SystemExit(f"{path}: undefined {field} without a failure flag at {r['prompt_id']} / {r['image_id']}")
            v = TIE_FILL
            if fills is not None:
                fills.append((r["generator_id"], str(r["prompt_id"])))
        out.append({"prompt_id": str(r["prompt_id"]), "generator_id": r["generator_id"], "image_id": r["image_id"], "candidate_index": int(r["candidate_index"]),
                    "score": float(v)})
    return out


def validate(rows: list[dict], universe: dict) -> dict:
    keys = [(r["prompt_id"], r["generator_id"], r["image_id"]) for r in rows]
    dup = len(keys) - len(set(keys)); miss = len(set(universe) - set(keys)); extra = len(set(keys) - set(universe))
    nonfinite = sum(1 for r in rows if not math.isfinite(r["score"]))
    badidx = sum(1 for r, k in zip(rows, keys) if k in universe and universe[k] != r["candidate_index"])
    ok = dup == 0 and miss == 0 and extra == 0 and nonfinite == 0 and badidx == 0
    return {"rows": len(rows), "duplicates": dup, "missing": miss, "extra": extra, "nonfinite": nonfinite, "candidate_index_mismatch": badidx, "ok": ok}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--report", required=True); a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}: a freeze is written once (use a new versioned directory)")
    out.mkdir(parents=True)
    universe = {}
    for line in INPUTS.read_text().splitlines():
        r = json.loads(line); universe[(str(r["prompt_id"]), r["generator_id"], r["image_id"])] = int(r["candidate_index"])
    man = {"what": "GenAI-Rank prediction freeze (label-free; human_rating.json not opened)", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "code_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "universe": {"file": str(INPUTS), "sha256": sha(INPUTS), "n": len(universe)}, "methods": {}}
    methods_json = {}; all_ok = True
    for name, (family, seeds) in METHODS.items():
        files, recs = [], []
        for si, srcs in enumerate(seeds):
            rows, fills, flagged, fallbacks = [], [], [], []
            skip = any(fd == "E!skip" for _, _, fd in srcs)
            for path, kind, field in srcs:
                rows += read_source(path, kind, field, fills, flagged, fallbacks)
            uni = universe
            if skip:                                   # parsed-pools universe: the failed pools are absent from this method by its own rule
                failed = set(flagged); uni = {k: c for k, c in universe.items() if (k[1], k[0]) not in failed}
            v = validate(rows, uni); all_ok &= v["ok"]; v["universe_rows"] = len(uni)
            v["O_fallback_pools"] = sorted(set(fallbacks)); v["n_O_fallback_pools"] = len(set(fallbacks))
            pools = {}
            for r in rows:
                pools.setdefault((r["generator_id"], r["prompt_id"]), set()).add(r["score"])
            tied = sorted(k for k, vals in pools.items() if len(vals) == 1)
            v["tie_filled_pools"] = sorted(set(fills)); v["n_tie_filled_pools"] = len(set(fills))
            v["flagged_failure_pools"] = sorted(set(flagged)); v["n_flagged_failure_pools"] = len(set(flagged))
            v["all_tied_pools"] = [list(k) for k in tied]; v["n_all_tied_pools"] = len(tied)
            rows.sort(key=lambda r: (r["generator_id"], r["prompt_id"], r["candidate_index"]))
            f = out / f"{name}{'_s%d' % si if len(seeds) > 1 else ''}.jsonl"
            f.write_text("".join(json.dumps(r) + "\n" for r in rows)); files.append(str(f))
            recs.append({"file": str(f), "sha256": sha(f), "validation": v,
                         "sources": [{"file": str(p), "sha256": sha(p), "kind": k, "field": fd} for p, k, fd in srcs]})
        methods_json[name] = files; man["methods"][name] = {"family": family, "n_seed_files": len(files), "files": recs}
    (out / "methods.json").write_text(json.dumps(methods_json, indent=1))
    man["methods_json"] = {"file": str(out / "methods.json"), "sha256": sha(out / "methods.json")}; man["all_valid"] = all_ok
    tied_any = sorted({tuple(k) for m in man["methods"].values() for r in m["files"] for k in r["validation"]["flagged_failure_pools"]
                       if m["family"].startswith(("v2_external_primary", "v2_attribution"))})
    man["undefined_score_rule"] = ("v2 (2026-10-14, before any GenAI-Rank label is read; corrects v1): M-arm E readouts follow the FROZEN deployment rule "
                                   "of scripts/v7/v7_parser.py / v7_train.py - decomposition_failed -> E undefined -> the score falls back to O (flagged); "
                                   "external systems keep their own outputs (TIFA / DSG / PN-VQA no-question prompts tie under their own scores; official "
                                   "first-index rule); X21 fixed-feature fits follow their protocol's 'no score' (main X21 rows on the parsed pools only, "
                                   "scored against labels filtered to the same pools) and a separately named neutral-fill full-denominator sensitivity "
                                   "(_neutralfill: one constant for all 9 candidates -> first index).  v1 had tie-filled the M arms, contradicting the frozen rule.")
    man["natural_ties_note"] = ("all-tied pools (a method's 9 scores equal) are counted per file; coarse scores (TIFA / DSG answer fractions) tie naturally; "
                                "the official first-index rule applies, and the frozen scorer's 'official/mean' tie-expectation sensitivity covers them")
    man["sensitivity_common_defined_subset"] = {"rule": "rescore the primary and attribution families without the pools where any of their methods has a "
                                                        "flagged failure (E undefined after decomposition failure; no questions generated)",
                                                "excluded_pools": [list(k) for k in tied_any], "n_excluded": len(tied_any)}
    man["scoring_commands_after_owner_approval"] = {
        "v2_external_primary": "python -I scripts/v7/v7_score.py rank --methods <methods.json> --labels <sealed human_rating> --out <new path> --ours M2_E "
                               "--comparators B0_FGA_released_O B1_VQAScore B4T_TIFA B4D_DSG --allow-scoring",
        "v2_attribution": "... --ours M2_E --comparators M1_E --allow-scoring", "v1_registered": "... --ours M2_E --comparators M1_E B0_FGA_released_O B1_VQAScore --allow-scoring"}
    rp = Path(a.report); rp.parent.mkdir(parents=True, exist_ok=True); rp.with_suffix(".json").write_text(json.dumps(man, indent=1))
    L = [f"# V7X-50 — GenAI-Rank prediction freeze ({man['utc']}, label-free)", "",
         f"Universe: `{INPUTS.name}` (n = {len(universe)}, sha256 `{man['universe']['sha256'][:16]}…`).  `human_rating.json` was not opened.  "
         f"All files valid: **{all_ok}**.  Converted files and `methods.json` (sha256 `{man['methods_json']['sha256'][:16]}…`) in `{out}`.", "",
         "| method | family | seed files | rows | valid | converted sha256 (first 16) |", "|---|---|---|---|---|---|"]
    for name, m in man["methods"].items():
        L.append(f"| {name} | {m['family']} | {m['n_seed_files']} | {', '.join(str(r['validation']['rows']) for r in m['files'])} | "
                 f"{all(r['validation']['ok'] for r in m['files'])} | {' '.join(r['sha256'][:16] for r in m['files'])} |")
    L[-1:-1] = []
    L += ["", f"Undefined-score rule: {man['undefined_score_rule']}.", "",
          f"Pre-declared sensitivity: the primary and attribution families rescored without the {man['sensitivity_common_defined_subset']['n_excluded']} "
          "pools where any of their methods has a flagged failure (list in the JSON).  " + man["natural_ties_note"] + ".", "",
          "| method | tie-filled / O-fallback pools | flagged-failure pools | all-tied pools (natural + filled; per seed file) |", "|---|---|---|---|"]
    for name, m in man["methods"].items():
        L.append(f"| {name} | {', '.join(str(r['validation']['n_tie_filled_pools']) + '/' + str(r['validation']['n_O_fallback_pools']) for r in m['files'])} | "
                 f"{', '.join(str(r['validation']['n_flagged_failure_pools']) for r in m['files'])} | {', '.join(str(r['validation']['n_all_tied_pools']) for r in m['files'])} |")
    L += ["", "Scoring (only after the owner's separate approval; frozen scorer, new output path, `--allow-scoring`):", ""]
    L += [f"- {k}: `{v}`" for k, v in man["scoring_commands_after_owner_approval"].items()]
    rp.with_suffix(".md").write_text("\n".join(L) + "\n"); print("\n".join(L)); return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
