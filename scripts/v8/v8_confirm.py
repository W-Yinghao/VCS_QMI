"""V8-05 confirmation evaluation on the registered untouched source (V8-03 `exposure`): the 850 GenAI-Bench prompts without a GenAI-Rank
text match, x 6 generators = 5,100 images, human 1-5 Likert ratings (several raters per image; the image score is the rater mean).

    python -I scripts/v8/v8_confirm.py prompt-list [--out F]        # the fixed list (exact text equality with the 800 GenAI-Rank prompts; no label)
    python -I scripts/v8/v8_confirm.py dry-run --out DIR [--hash]    # inputs, prediction completeness, file hashes; NEVER opens a rating column
    python -I scripts/v8/v8_confirm.py run --allow-scoring --out DIR  # the one label read (HumanRatings column); refuses without the confirmed
                                                                      # request file and a V8 ledger row; refuses to overwrite
    python -I scripts/v8/v8_confirm.py tifa160-inputs --images DIR --out F   # TIFA160 secondary: builds the V8 export inputs once the images exist

Metrics (GenAI-Bench is not a best-of-9 pool; one human score per image):
  spearman   pooled image-level Spearman rank correlation (average ranks on ties) between a method's score and the mean human rating over all
             confirmation images; per generator as a descriptive column
  pairwise   within-prompt pairwise preference accuracy over the 15 generator pairs of a prompt: pairs with equal human means are skipped,
             a method tie counts 0.5; prompt mean, then mean over prompts
Contrasts are U-CS (seed mean of mu over seeds 0-2) minus the comparator; prompt-cluster bootstrap (5,000 resamples, seed 2718) over the
confirmation prompts gives the 95 % percentile interval and the two-sided bootstrap p; Holm within each family.  Any image missing a
prediction in ANY method is dropped for every method and listed.  Every number of the report comes from confirm.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = Path("/home/infres/yinwang/CS_QMI/outputs/V7/data")
BENCH_INPUTS = DATA / "genai_bench_inputs.jsonl"          # prompt_id gb<Index>, generator_id, image_id, prompt (no rating field)
BENCH_PROMPTS = DATA / "genai_bench_prompts.jsonl"        # 1,600 prompts
RANK_PROMPTS = DATA / "genai_rank_prompts.jsonl"          # 800 prompts
PARQUET_DIR = Path("/projects/EEG-foundation-model/yinghao/data/genai_bench/data")   # 12 shards; HumanRatings column = the sealed labels
UNITS = Path("/projects/EEG-foundation-model/yinghao/outputs_v8/units")
X30 = Path("/projects/EEG-foundation-model/yinghao/outputs_v7/x30_bench")
EXTERNAL = {"B0_FGA_released_O": X30 / "native_B0_released" / "native_eval_bench.json",
            "B1_VQAScore": X30 / "b1_vqascore_clipflant5xxl.jsonl",
            "B4T_TIFA": X30 / "tifa_scores_0of1.jsonl",
            "B4D_DSG": X30 / "dsg_scores_0of1.jsonl"}
ARMS = ("U-CS", "U-BCE", "U-RPS", "U-CE", "U-MSE", "U-L1", "U-RANK", "U-LIST"); SEEDS = (0, 1, 2)
GENS = ("DALLE_3", "DeepFloyd_I_XL_v1", "Midjourney_6", "SDXL_2_1", "SDXL_Base", "SDXL_Turbo")
FAMILIES = {"primary_spearman": ("U-CS", list(EXTERNAL), "spearman"), "primary_pairwise": ("U-CS", list(EXTERNAL), "pairwise"),
            "attribution": ("U-CS", ["U-BCE"], "both"), "same_data_descriptive": ("U-CS", ["U-RPS", "U-CE", "U-MSE", "U-L1", "U-RANK", "U-LIST"], "both")}
BOOT, BOOT_SEED = 5000, 2718
REQUEST = ROOT / "reports" / "V8_UTILITY" / "V8_05_CONFIRMATION_REQUEST.md"
LEDGER = ROOT / "reports" / "V8_UTILITY" / "LABEL_ACCESS_LEDGER.csv"
TIFA_TEXT = Path("/projects/EEG-foundation-model/yinghao/third_party/tifa@8e37050/tifa_v1.0/tifa_v1.0_text_inputs.json")
DSG_DEV = Path("/projects/EEG-foundation-model/yinghao/third_party/DSG@7330119/dsg/data/tifa160-dev-anns.csv")


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


# ------------------------------------------------------------------------------------------------------------------ prompt list
def nonrank_prompts(bench_rows, rank_rows) -> dict:
    """Exact text equality (no normalisation) between a GenAI-Bench prompt and any GenAI-Rank prompt -> overlap; the rest = the confirmation set."""
    rank = {r["prompt"] for r in rank_rows}
    keep = sorted(r["prompt_id"] for r in bench_rows if r["prompt"] not in rank)
    drop = sorted(r["prompt_id"] for r in bench_rows if r["prompt"] in rank)
    return {"rule": "exact prompt text equality with the 800 GenAI-Rank prompts", "n_bench": len(bench_rows), "n_rank": len(rank_rows),
            "n_overlap": len(drop), "n_confirmation": len(keep), "prompt_ids": keep,
            "sha256_ids": hashlib.sha256("\n".join(keep).encode()).hexdigest()}


# ------------------------------------------------------------------------------------------------------------------ predictions
def unit_dir(arm: str, seed: int, units: Path = UNITS) -> Path:
    return units / f"V8-FIT-{arm}-s{seed}"


def load_v8(arm: str, seed: int, units: Path = UNITS) -> dict:
    p = unit_dir(arm, seed, units) / "pred" / "genai_bench.jsonl"
    return {r["image_id"]: float(r["score"]) for r in read_jsonl(p)}


def load_external(name: str, path: Path) -> dict:
    if name == "B0_FGA_released_O":
        return {r["img_path"]: float(r["score_result"]) for r in json.load(open(path))}
    rows = read_jsonl(path)
    return {r["image_id"]: float(r["score"]) for r in rows if "image_id" in r and "score" in r}   # VQAScore has a header line


def seed_mean(dicts: list[dict]) -> dict:
    keys = set.intersection(*(set(d) for d in dicts))
    return {k: float(np.mean([d[k] for d in dicts])) for k in keys}


# ------------------------------------------------------------------------------------------------------------------ metrics
def ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks on ties (1-based)."""
    x = np.asarray(x, float); order = np.argsort(x, kind="mergesort"); s = x[order]; r = np.empty(len(x)); i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and s[j + 1] == s[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return r


def spearman(a, b) -> float:
    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def pairwise_prompt(scores: np.ndarray, human: np.ndarray) -> float | None:
    """One prompt: 6 candidates; pairs with equal human means skipped; method tie 0.5.  None if no informative pair."""
    n = len(scores); num = 0.0; den = 0
    for i in range(n):
        for j in range(i + 1, n):
            if human[i] == human[j]:
                continue
            den += 1
            d = scores[i] - scores[j]
            num += 0.5 if d == 0 else float((d > 0) == (human[i] > human[j]))
    return None if den == 0 else num / den


class Panel:
    """Aligned arrays: images grouped by prompt (prompt order fixed), every method and the human mean on the same image order."""

    def __init__(self, prompt_of: dict, methods: dict, human: dict):
        imgs = sorted(set.intersection(*(set(m) for m in methods.values())) & set(human))
        self.by_prompt = defaultdict(list)
        for im in imgs:
            self.by_prompt[prompt_of[im]].append(im)
        self.prompts = sorted(self.by_prompt)
        self.images = [im for p in self.prompts for im in self.by_prompt[p]]
        self.idx = {p: np.array([self.images.index(im) for im in self.by_prompt[p]]) for p in self.prompts}
        self.h = np.array([human[im] for im in self.images])
        self.m = {k: np.array([v[im] for im in self.images]) for k, v in methods.items()}
        self.gen = np.array([im.split("/")[0] for im in self.images])

    def spearman(self, k: str, sel: np.ndarray | None = None) -> float:
        a, b = self.m[k], self.h
        return spearman(a if sel is None else a[sel], b if sel is None else b[sel])

    def pairwise_per_prompt(self, k: str) -> np.ndarray:
        out = []
        for p in self.prompts:
            ix = self.idx[p]; v = pairwise_prompt(self.m[k][ix], self.h[ix]); out.append(np.nan if v is None else v)
        return np.array(out)


def holm(ps: list[float]) -> list[float]:
    m = len(ps); order = np.argsort(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj.tolist()


def boot_p(bs: np.ndarray) -> float:
    return float(max(2 * min(np.mean(bs <= 0), np.mean(bs >= 0)), 1.0 / len(bs)))


def contrasts(panel: Panel, ours: str, comps: list[str], which: str, reps: int = BOOT, seed: int = BOOT_SEED) -> dict:
    """Prompt-cluster bootstrap of Δ = metric(ours) − metric(comp); both metrics when which == 'both'."""
    rng = np.random.default_rng(seed); P = len(panel.prompts); prompt_ix = [panel.idx[p] for p in panel.prompts]
    pw = {k: panel.pairwise_per_prompt(k) for k in [ours] + comps} if which in ("pairwise", "both") else {}
    draws = [rng.integers(0, P, P) for _ in range(reps)]
    out = {}
    for comp in comps:
        res = {}
        if which in ("spearman", "both"):
            pt = panel.spearman(ours) - panel.spearman(comp); bs = np.empty(reps)
            for r, dr in enumerate(draws):
                sel = np.concatenate([prompt_ix[i] for i in dr]); bs[r] = panel.spearman(ours, sel) - panel.spearman(comp, sel)
            res["spearman"] = {"delta": float(pt), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "p": boot_p(bs),
                               "value_ours": panel.spearman(ours), "value_comp": panel.spearman(comp)}
        if which in ("pairwise", "both"):
            d = pw[ours] - pw[comp]; ok = ~np.isnan(d); d = d[ok]; pt = float(np.mean(d)); bs = np.empty(reps)
            keep = np.where(ok)[0]; pos = {int(g): i for i, g in enumerate(keep)}
            for r, dr in enumerate(draws):
                sel = [pos[int(i)] for i in dr if int(i) in pos]; bs[r] = float(np.mean(d[sel])) if sel else np.nan
            bs = bs[~np.isnan(bs)]
            res["pairwise"] = {"delta": pt, "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "p": boot_p(bs),
                               "value_ours": float(np.nanmean(pw[ours])), "value_comp": float(np.nanmean(pw[comp])), "n_prompts_informative": int(ok.sum())}
        out[comp] = res
    return out


def add_holm(fam: dict) -> dict:
    for metric in ("spearman", "pairwise"):
        keys = [c for c in fam if metric in fam[c]]
        if keys:
            adj = holm([fam[c][metric]["p"] for c in keys])
            for c, a in zip(keys, adj):
                fam[c][metric]["p_holm"] = float(a)
    return fam


# ------------------------------------------------------------------------------------------------------------------ labels (run only)
def read_human(parquet_dir: Path, image_of: dict) -> dict:
    """The ONLY function that opens the HumanRatings column.  image_of[(prompt_id, generator)] -> image_id."""
    import pyarrow.parquet as pq
    human = {}
    for f in sorted(Path(parquet_dir).glob("*.parquet")):
        t = pq.read_table(f, columns=["Index", "HumanRatings"]).to_pylist()
        for row in t:
            pid = f"gb{int(row['Index']):04d}"; hr = row["HumanRatings"] or {}
            for g in GENS:
                v = hr.get(g)
                if v is None or len(v) == 0 or (pid, g) not in image_of:
                    continue
                human[image_of[(pid, g)]] = float(np.mean(v))
    return human


# ------------------------------------------------------------------------------------------------------------------ inventory
def inventory(units: Path = UNITS, external: dict = EXTERNAL, do_hash: bool = False) -> dict:
    bench = read_jsonl(BENCH_INPUTS); pl = nonrank_prompts(read_jsonl(BENCH_PROMPTS), read_jsonl(RANK_PROMPTS))
    keep = set(pl["prompt_ids"]); images = [r["image_id"] for r in bench if r["prompt_id"] in keep]
    inv = {"prompt_list": {k: v for k, v in pl.items() if k != "prompt_ids"}, "n_images_expected": len(images), "methods": {}, "missing_units": [],
           "parquet": [], "external_files": {}}
    for arm in ARMS:
        for s in SEEDS:
            p = unit_dir(arm, s, units) / "pred" / "genai_bench.jsonl"
            if not p.exists():
                inv["missing_units"].append(f"{arm}-s{s}"); continue
            pred = load_v8(arm, s, units); miss = [im for im in images if im not in pred]
            inv["methods"][f"{arm}-s{s}"] = {"file": str(p), "rows": len(pred), "missing_images": len(miss), "sha256": sha(p) if do_hash else None}
    for name, path in external.items():
        if not path.exists():
            inv["external_files"][name] = {"file": str(path), "present": False}; continue
        pred = load_external(name, path); miss = [im for im in images if im not in pred]
        inv["external_files"][name] = {"file": str(path), "present": True, "rows": len(pred), "missing_images": len(miss), "sha256": sha(path) if do_hash else None}
    for f in sorted(PARQUET_DIR.glob("*.parquet")):
        inv["parquet"].append({"file": f.name, "bytes": f.stat().st_size, "sha256": sha(f) if do_hash else None})
    inv["ready_for_run"] = (not inv["missing_units"] and all(m["missing_images"] == 0 for m in inv["methods"].values())
                            and all(e.get("present") and e["missing_images"] == 0 for e in inv["external_files"].values()) and len(inv["parquet"]) == 12)
    return inv


def cmd_prompt_list(a) -> int:
    pl = nonrank_prompts(read_jsonl(BENCH_PROMPTS), read_jsonl(RANK_PROMPTS))
    out = Path(a.out) if a.out else None
    if out:
        out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(pl, indent=1))
    print(json.dumps({k: v for k, v in pl.items() if k != "prompt_ids"}))
    return 0


def cmd_dry_run(a) -> int:
    inv = inventory(do_hash=a.hash); inv["utc"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); (out / "dry_run.json").write_text(json.dumps(inv, indent=1))
    print(f"[dry-run] confirmation prompts {inv['prompt_list']['n_confirmation']} (overlap {inv['prompt_list']['n_overlap']}), images {inv['n_images_expected']}; "
          f"units present {len(inv['methods'])} / 24, missing {inv['missing_units']}; external present {sum(1 for e in inv['external_files'].values() if e.get('present'))} / 4; "
          f"parquet shards {len(inv['parquet'])}; ready_for_run={inv['ready_for_run']}")
    return 0 if inv["ready_for_run"] else 1


def cmd_run(a) -> int:
    if not a.allow_scoring:
        raise SystemExit("run needs --allow-scoring")
    if not REQUEST.exists() or "CONFIRMED" not in REQUEST.read_text()[:600]:
        raise SystemExit(f"refusing: {REQUEST} missing or not CONFIRMED by the main session")
    out = Path(a.out)
    if (out / "confirm.json").exists():
        raise SystemExit(f"refusing: {out / 'confirm.json'} exists")
    inv = inventory(do_hash=True)
    if not inv["ready_for_run"]:
        raise SystemExit("refusing: inventory not ready (see dry-run)")
    bench = read_jsonl(BENCH_INPUTS); keep = set(nonrank_prompts(read_jsonl(BENCH_PROMPTS), read_jsonl(RANK_PROMPTS))["prompt_ids"])
    rows = [r for r in bench if r["prompt_id"] in keep]; prompt_of = {r["image_id"]: r["prompt_id"] for r in rows}
    image_of = {(r["prompt_id"], r["generator_id"]): r["image_id"] for r in rows}
    methods = {}
    for arm in ARMS:
        per = {s: {im: v for im, v in load_v8(arm, s).items() if im in prompt_of} for s in SEEDS}
        methods[arm] = seed_mean(list(per.values()))
        for s in SEEDS:
            methods[f"{arm}-s{s}"] = per[s]
    for name, path in EXTERNAL.items():
        methods[name] = {im: v for im, v in load_external(name, path).items() if im in prompt_of}
    human = read_human(PARQUET_DIR, image_of)                       # the one label read
    missing_h = [im for im in prompt_of if im not in human]
    panel = Panel(prompt_of, methods, human)
    dropped = sorted(set(prompt_of) - set(panel.images))
    res = {"utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "inventory": inv, "n_images_scored": len(panel.images), "n_prompts_scored": len(panel.prompts),
           "dropped_images": dropped, "images_without_rating": missing_h, "boot": {"reps": BOOT, "seed": BOOT_SEED, "unit": "prompt"},
           "levels": {k: {"spearman_pooled": panel.spearman(k), "pairwise": float(np.nanmean(panel.pairwise_per_prompt(k))),
                          "spearman_per_generator": {g: panel.spearman(k, panel.gen == g) for g in GENS}} for k in methods},
           "per_seed": {f"{arm}-s{s}": {"spearman_pooled": panel.spearman(f"{arm}-s{s}"), "pairwise": float(np.nanmean(panel.pairwise_per_prompt(f"{arm}-s{s}")))} for arm in ARMS for s in SEEDS},
           "families": {}}
    for fam, (ours, comps, which) in FAMILIES.items():
        res["families"][fam] = add_holm(contrasts(panel, ours, comps, which)) if fam != "same_data_descriptive" else contrasts(panel, ours, comps, which)
    out.mkdir(parents=True, exist_ok=True); (out / "confirm.json").write_text(json.dumps(res, indent=1)); (out / "confirm.md").write_text(markdown(res))
    print(f"-> {out / 'confirm.md'}")
    return 0


def _fmt(c):
    return f"{c['delta']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]"


def markdown(res: dict) -> str:
    L = [f"# V8-05 confirmation — GenAI-Bench non-Rank prompts (generated from confirm.json, {res['utc']})", "",
         f"Images scored {res['n_images_scored']} over {res['n_prompts_scored']} prompts; dropped (missing in some method) {len(res['dropped_images'])}; "
         f"without rating {len(res['images_without_rating'])}.  Bootstrap: {res['boot']['reps']} prompt-cluster resamples, seed {res['boot']['seed']}.", ""]
    for fam, F in res["families"].items():
        L += [f"## {fam}", "", "| contrast | metric | Δ [95 %] | p | p Holm | ours / comp |", "|---|---|---|---|---|---|"]
        for comp, R in F.items():
            for metric, c in R.items():
                L.append(f"| U-CS − {comp} | {metric} | {_fmt(c)} | {c['p']:.4f} | {c.get('p_holm', float('nan')):.4f} | {c['value_ours']:.3f} / {c['value_comp']:.3f} |")
        L.append("")
    L += ["## Levels (pooled Spearman / pairwise accuracy / Spearman per generator)", "", "| method | Spearman | pairwise | " + " | ".join(GENS) + " |", "|---|---|---|" + "---|" * len(GENS)]
    for k, v in res["levels"].items():
        if "-s" in k and k.split("-s")[-1].isdigit():
            continue
        L.append(f"| {k} | {v['spearman_pooled']:.3f} | {v['pairwise']:.3f} | " + " | ".join(f"{v['spearman_per_generator'][g]:.3f}" for g in GENS) + " |")
    L += ["", "## Per seed (V8 arms)", "", "| unit | Spearman | pairwise |", "|---|---|---|"]
    for k, v in res["per_seed"].items():
        L.append(f"| {k} | {v['spearman_pooled']:.3f} | {v['pairwise']:.3f} |")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------------------------ TIFA160 secondary
def cmd_tifa160_inputs(a) -> int:
    """Builds the V8 export inputs for the 160 TIFA160 prompts x 5 models from an image directory; prompts from the public tifa_v1.0 text inputs,
    the 160 ids from DSG's public dev annotations (item_id column only).  Image file names must contain the text id and a model tag (TO_VERIFY
    after the download; see the protocol).  No rating file is opened."""
    import csv
    ids = sorted({r["item_id"] for r in csv.DictReader(open(DSG_DEV))})
    cap = {x["id"]: x["caption"] for x in json.load(open(TIFA_TEXT))}
    imgs = sorted(p for p in Path(a.images).rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    rows = []
    for tid in ids:
        for p in imgs:
            if tid in p.stem:
                rows.append({"prompt_id": tid, "generator_id": p.stem.replace(tid, "").strip("_-") or p.parent.name, "image_id": str(p.relative_to(a.images)), "candidate_index": 0, "prompt": cap[tid]})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"TIFA160 inputs: {len(ids)} prompts, {len(rows)} images matched (expected 800) -> {a.out}")
    return 0 if len(rows) == 800 else 1


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompt-list"); p.add_argument("--out", default=None)
    d = sub.add_parser("dry-run"); d.add_argument("--out", required=True); d.add_argument("--hash", action="store_true")
    r = sub.add_parser("run"); r.add_argument("--out", required=True); r.add_argument("--allow-scoring", action="store_true")
    t = sub.add_parser("tifa160-inputs"); t.add_argument("--images", required=True); t.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"prompt-list": cmd_prompt_list, "dry-run": cmd_dry_run, "run": cmd_run, "tifa160-inputs": cmd_tifa160_inputs}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
