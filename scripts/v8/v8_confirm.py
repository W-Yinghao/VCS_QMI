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


CORR = {"spearman": spearman, "pearson": pearson, "kendall_tau_b": kendall_tau_b}


def read_tifa160_dsg(path: Path, image_of: dict) -> dict:
    """DSG's tifa160-likert-anns.csv (schema from the PUBLIC dsg/data_utils.py: item_id, source_id, t2i_model, worker_id, answer): mean Likert per
    (source_id = TIFA text id, t2i_model -> file tag) over its raters.  image_of[(text id, tag)] -> image_id.  Opened by tifa160-run only."""
    import csv
    acc = defaultdict(list); raters = defaultdict(set)
    for r in csv.DictReader(open(path)):
        tag = DSG_T2I_TO_TAG.get(r["t2i_model"])
        key = (r["source_id"], tag)
        if tag is None or key not in image_of:
            continue
        acc[image_of[key]].append(float(r["answer"])); raters[image_of[key]].add(r["worker_id"])
    return {"human": {im: float(np.mean(v)) for im, v in acc.items()}, "n_raters": {im: len(raters[im]) for im in acc}}


def read_tifa160_tifa(path: Path, image_of: dict) -> dict:
    """TIFA's human_annotations.json — structure NOT known before the read (the folder README is sealed).  Tolerant parse: a dict keyed by image file
    name / stem / '<id>_<model>' with a number, a list of numbers or a dict holding scores; or a list of records.  Everything unrecognised is
    counted and reported; the detected layout is written to the result.  Opened by tifa160-run only."""
    raw = json.load(open(path)); stem_of = {}
    for (tid, tag), im in image_of.items():
        stem_of[f"{tid}_{tag}"] = im; stem_of[f"{tid}_{tag}.jpg"] = im; stem_of[im] = im; stem_of[im.split("/")[-1]] = im
    human, layout, unknown = {}, None, 0

    def score_of(v):
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
            return float(np.mean(v))
        if isinstance(v, dict):
            for k in ("human_score", "score", "likert", "likert_scores", "scores", "human_scores", "mean"):
                if k in v:
                    return score_of(v[k])
        return None

    if isinstance(raw, dict):
        layout = "dict"
        for k, v in raw.items():
            im = stem_of.get(str(k)) or stem_of.get(str(k).split("/")[-1])
            if im is None and isinstance(v, dict):                      # id -> model -> scores
                for k2, v2 in v.items():
                    im2 = stem_of.get(f"{k}_{k2}") or stem_of.get(str(k2)); s = score_of(v2)
                    if im2 is not None and s is not None:
                        human[im2] = s; layout = "dict[id][model]"
                    else:
                        unknown += 1
                continue
            s = score_of(v)
            if im is not None and s is not None:
                human[im] = s
            else:
                unknown += 1
    elif isinstance(raw, list):
        layout = "list"
        for r in raw:
            if not isinstance(r, dict):
                unknown += 1; continue
            key = r.get("image") or r.get("img_path") or r.get("image_id") or r.get("file") or r.get("filename")
            im = stem_of.get(str(key)) if key is not None else None
            if im is None and r.get("text_id") is not None and r.get("model") is not None:
                im = stem_of.get(f"{r['text_id']}_{r['model']}")
            s = score_of(r)
            if im is not None and s is not None:
                human[im] = s
            else:
                unknown += 1
    return {"human": human, "layout": layout, "unparsed_entries": unknown}


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
    """Δ of each correlation (ours − comp) with a bootstrap over the 160 text inputs (all five images of a prompt move together)."""
    prompts = sorted({prompt_of[im] for im in images}); idx = {p: np.array([i for i, im in enumerate(images) if prompt_of[im] == p]) for p in prompts}
    h = np.array([human[im] for im in images]); m = {k: np.array([methods[k][im] for im in images]) for k in [ours] + comps}
    rng = np.random.default_rng(seed); draws = [rng.integers(0, len(prompts), len(prompts)) for _ in range(reps)]
    out = {}
    for comp in comps:
        out[comp] = {}
        for cname, fn in CORR.items():
            pt = fn(m[ours], h) - fn(m[comp], h); bs = np.empty(reps)
            for r, dr in enumerate(draws):
                sel = np.concatenate([idx[prompts[i]] for i in dr]); bs[r] = fn(m[ours][sel], h[sel]) - fn(m[comp][sel], h[sel])
            bs = bs[~np.isnan(bs)]
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
    dsg = read_tifa160_dsg(TIFA160_LABELS["dsg_5rater"], image_of)                                   # the TIFA160 label read (DSG 5-rater)
    tifa = read_tifa160_tifa(TIFA160_LABELS["tifa_2rater"], image_of) if TIFA160_LABELS["tifa_2rater"].exists() else {"human": {}, "layout": None, "unparsed_entries": None}
    res = {"utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "inventory": inv, "boot": {"reps": BOOT, "seed": BOOT_SEED, "unit": "text input (160)"},
           "labels": {"dsg_5rater": {"images_with_rating": len(dsg["human"]), "raters_per_image": sorted(set(dsg["n_raters"].values()))},
                      "tifa_2rater": {"images_with_rating": len(tifa["human"]), "layout_detected": tifa["layout"], "unparsed_entries": tifa["unparsed_entries"]}},
           "comparators_present": comps_present, "comparators_absent": [n for n in TIFA160_EXTERNAL if n not in comps_present], "exposed": TIFA160_EXPOSED, "readouts": {}}
    for lab, H in (("dsg_5rater", dsg["human"]), ("tifa_2rater", tifa["human"])):
        if not H:
            res["readouts"][lab] = {"note": "no usable ratings parsed"}; continue
        images = sorted(set.intersection(*(set(m) for m in methods.values())) & set(H)); gens = sorted(set(gen_of[im] for im in images))
        h = np.array([H[im] for im in images])
        levels = {k: {cn: fn(np.array([methods[k][im] for im in images]), h) for cn, fn in CORR.items()} for k in methods}
        per_gen = {k: {g: spearman(np.array([methods[k][im] for im in images if gen_of[im] == g]), np.array([H[im] for im in images if gen_of[im] == g])) for g in gens} for k in methods if "-s" not in k}
        fams = {}
        for fam, (ours, comps) in TIFA160_FAMILIES.items():
            comps_here = [c for c in comps if c in methods]
            if not comps_here:
                continue
            fam_res = tifa160_contrasts(images, prompt_of, methods, H, ours, comps_here)
            fams[fam] = add_holm_corr(fam_res) if fam != "tifa_same_data_descriptive" else fam_res
        res["readouts"][lab] = {"n_images": len(images), "n_prompts": len({prompt_of[im] for im in images}), "levels": levels, "spearman_per_generator": per_gen, "families": fams}
    out.mkdir(parents=True, exist_ok=True); (out / "tifa160_confirm.json").write_text(json.dumps(res, indent=1)); (out / "tifa160_confirm.md").write_text(tifa160_markdown(res))
    print(f"-> {out / 'tifa160_confirm.md'}")
    return 0


def tifa160_markdown(res: dict) -> str:
    L = [f"# V8-05 §B — TIFA160 secondary external evaluation (generated from tifa160_confirm.json, {res['utc']})", "",
         f"Secondary evaluation added 2026-10-15 (not the registered confirmation source).  Labels: DSG 5-rater mean ({res['labels']['dsg_5rater']['images_with_rating']} images rated; "
         f"raters per image {res['labels']['dsg_5rater']['raters_per_image']}) and TIFA 2-rater mean ({res['labels']['tifa_2rater']['images_with_rating']} images parsed; layout "
         f"{res['labels']['tifa_2rater']['layout_detected']}; unparsed {res['labels']['tifa_2rater']['unparsed_entries']}).  Bootstrap: {res['boot']['reps']} resamples over the 160 text inputs, "
         f"seed {res['boot']['seed']}.  Comparators present: {res['comparators_present']}; absent: {res['comparators_absent']}; exposed: {list(res['exposed'])}.  Intervals only — no claim language.", ""]
    for lab, R in res["readouts"].items():
        L += [f"## labels = {lab}", ""]
        if "note" in R:
            L += [R["note"], ""]; continue
        L += [f"Images {R['n_images']} over {R['n_prompts']} text inputs.", ""]
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
