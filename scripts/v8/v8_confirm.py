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
TIFA160_MODELS = ("mini_dalle", "vq_diffusion", "stable_diffusion_v1_1", "stable_diffusion_v1_5", "stable_diffusion_v2_1")   # file-name tags of the official release (verified 2026-10-15)
DSG_T2I_TO_TAG = {"mini-dalle": "mini_dalle", "vq-diffusion": "vq_diffusion", "sd1dot1": "stable_diffusion_v1_1", "sd1dot5": "stable_diffusion_v1_5", "sd2dot1": "stable_diffusion_v2_1"}   # DSG's t2i_model names (public dsg/data_utils.py)
TIFA160_DIR = Path("/projects/EEG-foundation-model/yinghao/outputs_v8/confirm_v1/tifa160")
TIFA160_IMAGES = Path("/projects/EEG-foundation-model/yinghao/data/tifa160/images")
TIFA160_INPUTS = TIFA160_DIR / "tifa160_inputs.jsonl"
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
    imgs = sorted(p for p in Path(a.images).rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg") and "__MACOSX" not in p.parts)
    # VERIFIED 2026-10-15 on the official release tifa_annotated_images.zip (sha256 6a1e283a…): annotated_images/<text id>_<model>.jpg, 160 ids x the
    # five models below = 800 files; text ids contain underscores (coco_100240, partiprompt_153, ...), so the match is exact: stem == id + "_" + model.
    rows = []
    for p in imgs:
        hit = [t for t in ids if p.stem.startswith(t + "_") and p.stem[len(t) + 1:] in TIFA160_MODELS]
        if len(hit) != 1:
            raise SystemExit(f"{p.name}: file name does not map to exactly one (text id, model): {hit}")
        tid = hit[0]; model = p.stem[len(tid) + 1:]
        rows.append({"prompt_id": tid, "generator_id": model, "image_id": str(p.relative_to(a.images)), "candidate_index": TIFA160_MODELS.index(model), "prompt": cap[tid]})
    rows.sort(key=lambda r: (r["prompt_id"], r["candidate_index"]))
    per = {t: sorted(r["generator_id"] for r in rows if r["prompt_id"] == t) for t in ids}
    bad = [t for t in ids if per[t] != sorted(TIFA160_MODELS)]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"TIFA160 inputs: {len(ids)} prompts, {len(rows)} images (expected 800), ids without all 5 models: {len(bad)} -> {a.out}")
    return 0 if len(rows) == 800 and not bad else 1


# TIFA160 predictions (job v8_tifa160.sbatch) and sealed labels.  Paths only; nothing under CLOSED is opened by any command except tifa160-run.
TIFA160_EXTERNAL = {"B0_FGA_released_O": TIFA160_DIR / "native_B0_released" / "native_eval_tifa160.json",
                    "B1_VQAScore": TIFA160_DIR / "b1_vqascore_clipflant5xxl.jsonl",
                    "B4T_TIFA": TIFA160_DIR / "tifa_scores_0of1.jsonl",
                    "B4D_DSG": TIFA160_DIR / "dsg_scores_0of1.jsonl"}
TIFA160_EXPOSED = {"B4D_DSG": "in-context: the local DSG route's question examples came from DSG's tifa160-dev-anns.csv (human-written tuples for these 160 prompts)"}
CLOSED = Path("/projects/EEG-foundation-model/yinghao/third_party/closed")
TIFA160_LABELS = {"tifa_2rater": CLOSED / "tifa@8e37050" / "human_annotations" / "human_annotations.json",
                  "dsg_5rater": CLOSED / "DSG@7330119" / "tifa160-likert-anns.csv"}
TIFA160_SUMS = CLOSED / "SHA256SUMS_v7x_20261014.txt"
TIFA160_FAMILIES = {"tifa_primary": ("U-CS", list(TIFA160_EXTERNAL)), "tifa_attribution": ("U-CS", ["U-BCE"]),
                    "tifa_same_data_descriptive": ("U-CS", ["U-RPS", "U-CE", "U-MSE", "U-L1", "U-RANK", "U-LIST"])}


def load_v8_tifa160(arm: str, seed: int, units: Path = UNITS) -> dict:
    p = unit_dir(arm, seed, units) / "pred" / "tifa160.jsonl"
    return {r["image_id"]: float(r["score"]) for r in read_jsonl(p)}


def pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def kendall_tau_b(a, b) -> float:
    """Kendall tau-b (ties handled) — O(n^2), fine for 800 images."""
    a, b = np.asarray(a, float), np.asarray(b, float); n = len(a)
    da = np.sign(a[:, None] - a[None, :]); db = np.sign(b[:, None] - b[None, :]); iu = np.triu_indices(n, 1)
    da, db = da[iu], db[iu]; conc = float(np.sum((da * db) > 0)); disc = float(np.sum((da * db) < 0))
    ta = float(np.sum(da == 0)); tb = float(np.sum(db == 0)); n0 = len(da)
    den = math.sqrt((n0 - ta) * (n0 - tb))
    return float("nan") if den == 0 else (conc - disc) / den


try:                                                                       # identical estimators, faster (tests compare them with the reference code)
    from scipy.stats import kendalltau as _sp_kendalltau, rankdata as _sp_rankdata
except Exception:                                                          # pragma: no cover
    _sp_kendalltau = _sp_rankdata = None


def spearman_tifa(a, b) -> float:
    if _sp_rankdata is None:
        return spearman(a, b)
    ra, rb = _sp_rankdata(np.asarray(a, float)), _sp_rankdata(np.asarray(b, float))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def kendall_tifa(a, b) -> float:
    if _sp_kendalltau is None:
        return kendall_tau_b(a, b)
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(_sp_kendalltau(a, b).correlation)                         # scipy default = tau-b


CORR = {"spearman": spearman_tifa, "pearson": pearson, "kendall_tau_b": kendall_tifa}

# model-name aliases (normalised: lower case, alphanumerics only) -> official file tag
_MODEL_ALIAS = {}
for _tag, _al in {"mini_dalle": ("minidalle", "mindalle", "dallemini"),
                  "vq_diffusion": ("vqdiffusion",),
                  "stable_diffusion_v1_1": ("stablediffusionv11", "sd1dot1", "sdv11", "sd11", "stablediffusion11", "sdv1dot1", "stablediffusionv1dot1"),
                  "stable_diffusion_v1_5": ("stablediffusionv15", "sd1dot5", "sdv15", "sd15", "stablediffusion15", "sdv1dot5", "stablediffusionv1dot5"),
                  "stable_diffusion_v2_1": ("stablediffusionv21", "sd2dot1", "sdv21", "sd21", "stablediffusion21", "sdv2dot1", "stablediffusionv2dot1")}.items():
    for _a in _al:
        _MODEL_ALIAS[_a] = _tag
del _tag, _al, _a


def _norm(s) -> str:
    import re
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def model_alias(s):
    return _MODEL_ALIAS.get(_norm(s)) if s is not None else None


def _tifa160_resolver(image_of: dict):
    """String -> image_id for file names / stems / '<id>_<model>' with any model alias; ids longest-first (no prefix ambiguity)."""
    ids = sorted({t for t, _ in image_of}, key=len, reverse=True); by = {}
    for (tid, tag), im in image_of.items():
        for k in (f"{tid}_{tag}", im, im.split("/")[-1]):
            by[k] = im
    exts = (".jpg", ".jpeg", ".png", ".webp")

    def image_of_str(s):
        if not isinstance(s, str):
            return None
        base = s.replace("\\", "/").split("/")[-1]
        if base in by:
            return by[base]
        stem = base.rsplit(".", 1)[0] if base.lower().endswith(exts) else base
        if stem in by:
            return by[stem]
        for tid in ids:
            if len(stem) > len(tid) + 1 and stem.startswith(tid) and stem[len(tid)] in "_- /":
                tag = model_alias(stem[len(tid) + 1:])
                if tag is not None and (tid, tag) in image_of:
                    return image_of[(tid, tag)]
        return None
    return image_of_str, set(ids)


def read_tifa160_dsg(path: Path, image_of: dict) -> dict:
    """DSG's tifa160-likert-anns.csv.  Schema from the PUBLIC dsg/data_utils.py: item_id (DSG id, e.g. tifa160_63), source_id (TIFA text id),
    t2i_model in {mini-dalle, vq-diffusion, sd1dot1, sd1dot5, sd2dot1}, worker_id, answer (1-5 Likert).  Mean over all rows of an image (as DSG's
    load_human_likert_ann + np.mean).  The text-id column is whichever of source_id / item_id holds one of the 160 ids (counted); non-numeric answers
    are skipped and counted.  Opened by tifa160-run only."""
    import csv
    from collections import Counter
    ids = {t for t, _ in image_of}; acc = defaultdict(list); raters = defaultdict(set); st = Counter()
    with open(path, newline="") as f:
        rd = csv.DictReader(f); cols = list(rd.fieldnames or [])
        for r in rd:
            st["rows"] += 1
            tid = r.get("source_id") if r.get("source_id") in ids else (r.get("item_id") if r.get("item_id") in ids else None)
            tag = DSG_T2I_TO_TAG.get(r.get("t2i_model")) or model_alias(r.get("t2i_model"))
            if tid is None or tag is None or (tid, tag) not in image_of:
                st["rows_unmatched"] += 1; continue
            try:
                v = float(r.get("answer"))
            except (TypeError, ValueError):
                st["answer_not_numeric"] += 1; continue
            if not math.isfinite(v):
                st["answer_not_numeric"] += 1; continue
            im = image_of[(tid, tag)]; acc[im].append(v); raters[im].add(r.get("worker_id")); st["rows_used"] += 1
            st["text_id_from_" + ("source_id" if r.get("source_id") == tid else "item_id")] += 1
    vals = [x for v in acc.values() for x in v]
    return {"human": {im: float(np.mean(v)) for im, v in acc.items()}, "n_raters": {im: len(raters[im]) for im in acc},
            "meta": {"columns": cols, "counts": dict(st), "ratings_per_image": dict(sorted(Counter(len(v) for v in acc.values()).items())),
                     "raters_per_image": dict(sorted(Counter(len(raters[im]) for im in acc).items())),
                     "value_range": [min(vals), max(vals)] if vals else None}}


_EXCL_TOK = {"tifa", "clip", "blip", "blip2", "bleu", "rouge", "meteor", "spice", "cider", "vqa", "caption", "captions", "text", "prompt", "question",
             "questions", "id", "ids", "idx", "index", "num", "n", "count", "model", "models", "time", "seed", "width", "height", "len", "length",
             "rank", "dsg", "auto", "automatic", "metric", "metrics", "pred", "prediction", "std", "var", "variance", "stdev", "sd", "agreement",
             "kappa", "alpha", "confidence", "conf", "min", "max", "median", "bert", "bertscore", "clipscore", "tifascore", "vqascore", "pickscore",
             "hps", "imagereward", "aesthetic", "fid", "is", "version", "year", "elapsed", "duration", "seconds", "timestamp"}
_T1_TOK = {"human", "humans", "likert", "annotator", "annotators", "worker", "workers", "rater", "raters", "annotation", "annotations",
           "judgment", "judgments", "judgement", "judgements"}
_T2_TOK = {"score", "scores", "rating", "ratings", "avg", "average", "mean", "answer", "answers", "overall", "faithfulness"}
_AVG_TOK = {"avg", "average", "mean"}


def _key_tokens(k) -> set:
    import re
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(k))
    return {t for t in re.split(r"[^a-z0-9]+", s.lower()) if t}


def _path_tier(path: tuple):
    toks = set()
    for k in path:
        if k != "[*]":
            toks |= _key_tokens(k)
    if toks & _EXCL_TOK:
        return None
    if toks & _T1_TOK:
        return 1
    if toks & _T2_TOK:
        return 2
    return 3 if not toks else 4


def _numeric_leaves(node, path=()):
    import re
    if isinstance(node, bool) or node is None:
        return
    if isinstance(node, (int, float)):
        if math.isfinite(float(node)):
            yield path, float(node)
    elif isinstance(node, str):
        if re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", node):
            yield path, float(node)
    elif isinstance(node, list):
        for v in node:
            yield from _numeric_leaves(v, path + ("[*]",))
    elif isinstance(node, dict):
        for k, v in node.items():
            yield from _numeric_leaves(v, path + (str(k),))


def _rating_of(entry):
    """Human rating values of one entry: the best tier present (1 human/likert/annotator… > 2 score/rating/avg… > 3 bare number(s) > 4 other
    non-excluded numeric fields, used only when all lie in [0, 10]); within the tier one list path (raw ratings) wins, else one avg/mean path, else
    all scalar paths of the tier.  Returns (values, rule) or (None, reason)."""
    by = defaultdict(list)
    for p, v in _numeric_leaves(entry):
        by[p].append(v)
    tiers = defaultdict(list)
    for p in by:
        t = _path_tier(p)
        if t is not None:
            tiers[t].append(p)
    for t in (1, 2, 3, 4):
        ps = sorted(tiers.get(t, []))
        if not ps:
            continue
        if t == 4 and not all(0 <= v <= 10 for p in ps for v in by[p]):
            return None, "tier4 values outside [0, 10]"
        lists = [p for p in ps if p and p[-1] == "[*]"]
        avgs = [p for p in ps if any(_key_tokens(k) & _AVG_TOK for k in p if k != "[*]")]
        if len(lists) == 1:
            use, how = lists, "list"
        elif len(avgs) == 1:
            use, how = avgs, "avg"
        else:
            use, how = ps, "all"
        return [v for p in use for v in by[p]], f"tier{t}:{how}:" + " + ".join(".".join(p) or "<value>" for p in use)
    return None, "no numeric rating field"


def _json_shape(node, image_of_str, ids, depth=0):
    """Structure only (types, sizes, masked key names) — never a value."""
    import re
    if depth > 4:
        return "..."
    if isinstance(node, dict):
        def mask(k):
            k = str(k)
            if image_of_str(k):
                return "<image>"
            if k in ids:
                return "<text_id>"
            if model_alias(k):
                return "<model>"
            return k if re.fullmatch(r"[A-Za-z_][A-Za-z_ \-]{0,30}\d{0,2}", k) else "<key>"
        ks = list(node)[:4]
        return {"dict_len": len(node), "first_keys": {mask(k): _json_shape(node[k], image_of_str, ids, depth + 1) for k in ks}}
    if isinstance(node, list):
        return {"list_len": len(node), "item": _json_shape(node[0], image_of_str, ids, depth + 1) if node else None}
    return type(node).__name__


def read_tifa160_tifa(path: Path, image_of: dict) -> dict:
    """TIFA's human_annotations.json — layout NOT known before the read (the folder README is sealed; no public code parses it).  Layout-agnostic:
    entries are located by (a) a key that names an image (file name / stem / '<id>_<model>' with any model alias), (b) a text-id key then a model
    key, or (c) a record whose string fields name an image or a text id + a model (several records per image = per-rater rows, pooled); the rating
    is chosen per entry by _rating_of (field-name tiers, metric fields excluded) and the chosen paths are counted.  The result records the layout,
    the identification modes, the rating paths, ratings per image, the value range and a value-free structure summary.  Opened by tifa160-run only."""
    from collections import Counter
    raw = json.load(open(path)); image_of_str, ids = _tifa160_resolver(image_of)
    entries = defaultdict(list); how = Counter(); unknown = [0]

    def collect(node, tid=None, model=None, depth=0):
        if depth > 8:
            unknown[0] += 1; return
        if isinstance(node, dict):
            im = None; t, m = tid, model
            for v in node.values():
                if isinstance(v, str):
                    im = im or image_of_str(v)
                    if v in ids:
                        t = v
                    elif model_alias(v):
                        m = model_alias(v)
            if im is None and t is not None and m is not None and (t, m) in image_of and (t != tid or m != model):
                im = image_of[(t, m)]; how["record:id+model"] += 1; entries[im].append(node); return
            if im is not None:
                how["record:image"] += 1; entries[im].append(node); return
            for k, v in node.items():
                ks = str(k); imk = image_of_str(ks)
                if imk is not None:
                    how["key:image"] += 1; entries[imk].append(v); continue
                if ks in ids:
                    collect(v, ks, m, depth + 1); continue
                mk = model_alias(ks)
                if mk is not None:
                    if t is not None and (t, mk) in image_of:
                        how["key:id>model"] += 1; entries[image_of[(t, mk)]].append(v)
                    else:
                        collect(v, t, mk, depth + 1)
                    continue
                if isinstance(v, (dict, list)):
                    collect(v, t, m, depth + 1)
                elif not isinstance(v, str):
                    unknown[0] += 1
        elif isinstance(node, list):
            for v in node:
                collect(v, tid, model, depth + 1)
        elif not isinstance(node, str):
            unknown[0] += 1

    collect(raw)
    human, paths, per_image, no_rating = {}, Counter(), Counter(), 0
    for im, subs in entries.items():
        vals = []
        for s in subs:
            v, rule = _rating_of(s); paths[rule] += 1
            if v:
                vals += v
        if vals:
            human[im] = float(np.mean(vals)); per_image[len(vals)] += 1
        else:
            no_rating += 1
    layout = ("list" if isinstance(raw, list) else "dict[id][model]" if how.get("key:id>model") else "dict" if isinstance(raw, dict) else type(raw).__name__)
    allv = list(human.values())
    return {"human": human, "layout": layout, "unparsed_entries": unknown[0] + no_rating,
            "meta": {"identified_by": dict(how), "rating_paths": dict(paths.most_common(8)), "n_rating_paths": len(paths),
                     "images_located": len(entries), "images_without_rating": no_rating, "ratings_per_image": dict(sorted(per_image.items())),
                     "image_mean_range": [min(allv), max(allv)] if allv else None, "structure": _json_shape(raw, image_of_str, ids)}}


def tifa160_inventory(do_hash: bool = False, units: Path = UNITS) -> dict:
    cands = read_jsonl(TIFA160_INPUTS) if TIFA160_INPUTS.exists() else []
    images = [c["image_id"] for c in cands]
    inv = {"inputs": {"file": str(TIFA160_INPUTS), "rows": len(cands), "prompts": len({c["prompt_id"] for c in cands}), "sha256": sha(TIFA160_INPUTS) if do_hash and cands else None},
           "methods": {}, "missing_units": [], "external_files": {}, "exposed": TIFA160_EXPOSED, "sealed": {}}
    for arm in ARMS:
        for s in SEEDS:
            p = unit_dir(arm, s, units) / "pred" / "tifa160.jsonl"
            if not p.exists():
                inv["missing_units"].append(f"{arm}-s{s}"); continue
            pred = load_v8_tifa160(arm, s, units); miss = [im for im in images if im not in pred]
            inv["methods"][f"{arm}-s{s}"] = {"file": str(p), "rows": len(pred), "missing_images": len(miss), "sha256": sha(p) if do_hash else None}
    for name, path in TIFA160_EXTERNAL.items():
        if not path.exists():
            inv["external_files"][name] = {"file": str(path), "present": False, "exposed": name in TIFA160_EXPOSED}; continue
        pred = load_external(name, path); miss = [im for im in images if im not in pred]
        inv["external_files"][name] = {"file": str(path), "present": True, "rows": len(pred), "missing_images": len(miss), "sha256": sha(path) if do_hash else None, "exposed": name in TIFA160_EXPOSED}
    sums = {}
    if TIFA160_SUMS.exists():
        for line in TIFA160_SUMS.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                sums[parts[-1]] = parts[0]
    for name, path in TIFA160_LABELS.items():                             # bytes hashed only (the seal); never parsed here
        rec = {"file": str(path), "present": path.exists(), "bytes": path.stat().st_size if path.exists() else None, "sha256": sha(path) if (do_hash and path.exists()) else None}
        reg = [h for f, h in sums.items() if f.endswith(path.name) and path.parent.name in f]
        rec["seal_registered"] = reg[0] if reg else None; rec["seal_matches"] = (rec["sha256"] == reg[0]) if (reg and rec["sha256"]) else None
        inv["sealed"][name] = rec
    present_ext = [n for n, e in inv["external_files"].items() if e.get("present")]
    inv["ready_for_run"] = (len(cands) == 800 and not inv["missing_units"] and all(m["missing_images"] == 0 for m in inv["methods"].values())
                            and all(inv["external_files"][n]["missing_images"] == 0 for n in present_ext) and inv["sealed"]["dsg_5rater"]["present"])
    inv["external_present"] = present_ext
    return inv


def cmd_tifa160_dry_run(a) -> int:
    inv = tifa160_inventory(do_hash=a.hash); inv["utc"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); (out / "tifa160_dry_run.json").write_text(json.dumps(inv, indent=1))
    print(f"[tifa160-dry-run] inputs {inv['inputs']['rows']} rows / {inv['inputs']['prompts']} prompts; units present {len(inv['methods'])} / 24, missing {inv['missing_units']}; "
          f"external present {inv['external_present']} (absent: {[n for n in TIFA160_EXTERNAL if n not in inv['external_present']]}); sealed present "
          f"{[n for n, s in inv['sealed'].items() if s['present']]}; ready_for_run={inv['ready_for_run']}")
    return 0 if inv["ready_for_run"] else 1


def tifa160_contrasts(images: list, prompt_of: dict, methods: dict, human: dict, ours: str, comps: list, reps: int = BOOT, seed: int = BOOT_SEED) -> dict:
    """Δ of each correlation (ours − comp) with a bootstrap over the 160 text inputs (all five images of a prompt move together).  The resamples are
    drawn once per call (same RNG sequence as before) and ours' bootstrap values are computed once and reused for every comparator."""
    prompts = sorted({prompt_of[im] for im in images}); pos = defaultdict(list)
    for i, im in enumerate(images):
        pos[prompt_of[im]].append(i)
    idx = {p: np.array(pos[p]) for p in prompts}
    h = np.array([human[im] for im in images]); m = {k: np.array([methods[k][im] for im in images]) for k in [ours] + comps}
    rng = np.random.default_rng(seed); draws = [rng.integers(0, len(prompts), len(prompts)) for _ in range(reps)]
    sels = [np.concatenate([idx[prompts[i]] for i in dr]) for dr in draws]
    b_ours = {cname: np.array([fn(m[ours][s], h[s]) for s in sels]) for cname, fn in CORR.items()}
    out = {}
    for comp in comps:
        out[comp] = {}
        for cname, fn in CORR.items():
            pt = fn(m[ours], h) - fn(m[comp], h)
            bs = b_ours[cname] - np.array([fn(m[comp][s], h[s]) for s in sels]); bs = bs[~np.isnan(bs)]
            out[comp][cname] = {"delta": float(pt), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "p": boot_p(bs),
                                "value_ours": fn(m[ours], h), "value_comp": fn(m[comp], h), "exposed": comp in TIFA160_EXPOSED}
    return out


def add_holm_corr(fam: dict) -> dict:
    for cname in CORR:
        keys = [c for c in fam if cname in fam[c]]
        if keys:
            for c, adj in zip(keys, holm([fam[c][cname]["p"] for c in keys])):
                fam[c][cname]["p_holm"] = float(adj)
    return fam


def cmd_tifa160_run(a) -> int:
    if not a.allow_scoring:
        raise SystemExit("tifa160-run needs --allow-scoring")
    if not REQUEST.exists() or "TIFA160 CONFIRMED" not in REQUEST.read_text():
        raise SystemExit(f"refusing: {REQUEST} missing or without the line 'TIFA160 CONFIRMED' (main session, after the owner's TIFA160 approval)")
    out = Path(a.out)
    if (out / "tifa160_confirm.json").exists():
        raise SystemExit(f"refusing: {out / 'tifa160_confirm.json'} exists")
    inv = tifa160_inventory(do_hash=True)
    if not inv["ready_for_run"]:
        raise SystemExit("refusing: TIFA160 inventory not ready (see tifa160-dry-run)")
    cands = read_jsonl(TIFA160_INPUTS); prompt_of = {c["image_id"]: c["prompt_id"] for c in cands}; gen_of = {c["image_id"]: c["generator_id"] for c in cands}
    image_of = {(c["prompt_id"], c["generator_id"]): c["image_id"] for c in cands}
    methods = {}
    for arm in ARMS:
        per = {s: load_v8_tifa160(arm, s) for s in SEEDS}; methods[arm] = seed_mean(list(per.values()))
        for s in SEEDS:
            methods[f"{arm}-s{s}"] = per[s]
    comps_present = []
    for name, path in TIFA160_EXTERNAL.items():
        if path.exists():
            methods[name] = load_external(name, path); comps_present.append(name)

    def safe(fn, path):                                                                   # the one read must always leave a result file
        try:
            return fn(path, image_of)
        except Exception as e:
            return {"human": {}, "n_raters": {}, "layout": None, "unparsed_entries": None, "meta": {}, "error": f"{type(e).__name__}: {e}"}
    tifa = (safe(read_tifa160_tifa, TIFA160_LABELS["tifa_2rater"]) if TIFA160_LABELS["tifa_2rater"].exists()     # the TIFA160 label read (both files)
            else {"human": {}, "layout": None, "unparsed_entries": None, "meta": {}, "error": "file absent"})
    dsg = safe(read_tifa160_dsg, TIFA160_LABELS["dsg_5rater"])
    all_images = sorted(set.intersection(*(set(m) for m in methods.values())) & set(prompt_of))
    res = {"utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "inventory": inv, "boot": {"reps": BOOT, "seed": BOOT_SEED, "unit": "text input (160)"},
           "images_scored_by_every_method": len(all_images),
           "labels": {"tifa_2rater": {"images_with_rating": len(tifa["human"]), "layout_detected": tifa.get("layout"), "unparsed_entries": tifa.get("unparsed_entries"),
                                      "meta": tifa.get("meta", {}), "error": tifa.get("error")},
                      "dsg_5rater": {"images_with_rating": len(dsg["human"]), "raters_per_image": sorted(set(dsg.get("n_raters", {}).values())),
                                     "meta": dsg.get("meta", {}), "error": dsg.get("error")}},
           "comparators_present": comps_present, "comparators_absent": [n for n in TIFA160_EXTERNAL if n not in comps_present], "exposed": TIFA160_EXPOSED, "readouts": {}}
    for lab, H in (("tifa_2rater", tifa["human"]), ("dsg_5rater", dsg["human"])):
        if not H:
            res["readouts"][lab] = {"note": "no usable ratings parsed" + (f" ({res['labels'][lab]['error']})" if res["labels"][lab].get("error") else "")}; continue
        try:
            res["readouts"][lab] = tifa160_readout(H, methods, prompt_of, gen_of, all_images)
        except Exception as e:                                                              # keep the other label source and the parse record
            res["readouts"][lab] = {"note": f"readout failed: {type(e).__name__}: {e}"}
    out.mkdir(parents=True, exist_ok=True); (out / "tifa160_confirm.json").write_text(json.dumps(res, indent=1)); (out / "tifa160_confirm.md").write_text(tifa160_markdown(res))
    print(f"-> {out / 'tifa160_confirm.md'}")
    return 0


def tifa160_readout(H: dict, methods: dict, prompt_of: dict, gen_of: dict, all_images: list) -> dict:
    images = [im for im in all_images if im in H]; gens = sorted(set(gen_of[im] for im in images))
    h = np.array([H[im] for im in images])
    levels = {k: {cn: fn(np.array([methods[k][im] for im in images]), h) for cn, fn in CORR.items()} for k in methods}
    per_gen = {k: {g: spearman_tifa(np.array([methods[k][im] for im in images if gen_of[im] == g]), np.array([H[im] for im in images if gen_of[im] == g])) for g in gens} for k in methods if "-s" not in k}
    fams = {}
    for fam, (ours, comps) in TIFA160_FAMILIES.items():
        comps_here = [c for c in comps if c in methods]
        if not comps_here:
            continue
        fam_res = tifa160_contrasts(images, prompt_of, methods, H, ours, comps_here)
        fams[fam] = add_holm_corr(fam_res) if fam != "tifa_same_data_descriptive" else fam_res
    return {"n_images": len(images), "n_prompts": len({prompt_of[im] for im in images}), "n_dropped_no_rating": len(all_images) - len(images),
            "levels": levels, "spearman_per_generator": per_gen, "families": fams}


def tifa160_markdown(res: dict) -> str:
    L = [f"# V8-05 §B — TIFA160 secondary external evaluation (generated from tifa160_confirm.json, {res['utc']})", "",
         f"Secondary evaluation added 2026-10-15 (not the registered confirmation source).  Images scored by every method: {res['images_scored_by_every_method']}.  "
         f"Bootstrap: {res['boot']['reps']} resamples over the 160 text inputs, seed {res['boot']['seed']}.  Comparators present: {res['comparators_present']}; absent: "
         f"{res['comparators_absent']}; exposed: {list(res['exposed'])}.  Intervals only — no claim language.", ""]
    t, d = res["labels"]["tifa_2rater"], res["labels"]["dsg_5rater"]; tm, dm = t.get("meta", {}), d.get("meta", {})
    L += ["| label source | images with a rating | ratings per image (count of images) | parse record |", "|---|---|---|---|",
          f"| TIFA 2-rater mean (human_annotations.json) | {t['images_with_rating']} | {tm.get('ratings_per_image')} | layout {t['layout_detected']}; located by "
          f"{tm.get('identified_by')}; rating fields {tm.get('rating_paths')}; unparsed {t['unparsed_entries']}" + (f"; error {t['error']}" if t.get("error") else "") + " |",
          f"| DSG 5-rater mean (tifa160-likert-anns.csv) | {d['images_with_rating']} | {dm.get('ratings_per_image')} | raters per image {dm.get('raters_per_image')}; "
          f"row counts {dm.get('counts')}" + (f"; error {d['error']}" if d.get("error") else "") + " |", ""]
    for lab, R in res["readouts"].items():
        L += [f"## labels = {lab}", ""]
        if "note" in R:
            L += [R["note"], ""]; continue
        L += [f"Images {R['n_images']} over {R['n_prompts']} text inputs (dropped without a rating: {R.get('n_dropped_no_rating')}).", ""]
        for fam, F in R["families"].items():
            L += [f"### {fam}", "", "| contrast | correlation | Δ [95 %] | p | p Holm | ours / comp | exposed |", "|---|---|---|---|---|---|---|"]
            for comp, C_ in F.items():
                for cn, c in C_.items():
                    L.append(f"| U-CS − {comp} | {cn} | {_fmt(c)} | {c['p']:.4f} | {c.get('p_holm', float('nan')):.4f} | {c['value_ours']:.3f} / {c['value_comp']:.3f} | {'yes' if c['exposed'] else ''} |")
            L.append("")
        gens = sorted(next(iter(R["spearman_per_generator"].values())).keys()) if R["spearman_per_generator"] else []
        L += ["### Levels (Spearman / Pearson / Kendall tau-b; Spearman per generator)", "", "| method | Spearman | Pearson | Kendall | " + " | ".join(gens) + " |", "|---|---|---|---|" + "---|" * len(gens)]
        for k, v in R["levels"].items():
            if "-s" in k and k.split("-s")[-1].isdigit():
                continue
            pg = R["spearman_per_generator"].get(k, {})
            L.append(f"| {k} | {v['spearman']:.3f} | {v['pearson']:.3f} | {v['kendall_tau_b']:.3f} | " + " | ".join(f"{pg.get(g, float('nan')):.3f}" for g in gens) + " |")
        L.append("")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompt-list"); p.add_argument("--out", default=None)
    d = sub.add_parser("dry-run"); d.add_argument("--out", required=True); d.add_argument("--hash", action="store_true")
    r = sub.add_parser("run"); r.add_argument("--out", required=True); r.add_argument("--allow-scoring", action="store_true")
    t = sub.add_parser("tifa160-inputs"); t.add_argument("--images", required=True); t.add_argument("--out", required=True)
    td = sub.add_parser("tifa160-dry-run"); td.add_argument("--out", required=True); td.add_argument("--hash", action="store_true")
    tr = sub.add_parser("tifa160-run"); tr.add_argument("--out", required=True); tr.add_argument("--allow-scoring", action="store_true")
    a = ap.parse_args(argv)
    return {"prompt-list": cmd_prompt_list, "dry-run": cmd_dry_run, "run": cmd_run, "tifa160-inputs": cmd_tifa160_inputs,
            "tifa160-dry-run": cmd_tifa160_dry_run, "tifa160-run": cmd_tifa160_run}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
