"""V7-01 data: EvalMuse public TRAIN -> audited image / element manifests, prompt-and-image components, the role split, FIT weights and prior.
CPU only.  Reads train_list.json and the 6-part image archive through a seekable multi-part reader (TRAIN members only; test.json and every
other member are never opened).  External terminal sets contribute prompt TEXT only (no label column is ever read).
    python -I scripts/v7/v7_data.py genai-prompts --parquet-dir D --out F          GenAI-Bench prompt text (id + prompt columns only)
    python -I scripts/v7/v7_data.py audit --out O [--external F ...] [--bert B] [--limit N]
Label rules, span rules, de-duplication rules and the split rule are fixed here before any model output exists (V7 package docs/04 §2–3).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import time
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

EVALMUSE = Path("/projects/EEG-foundation-model/yinghao/data/evalmuse")
ZIP_PREFIX = "dataset/images/"
SPLIT_SEED = 17071
ROLE_FRACTIONS = (("FIT", 0.80), ("CAL", 0.10), ("DEV", 0.10))
VALID_VOTES = (0, 1)
FLAG_MISSING = -1
NEAR_DUP_PROMPT_JACCARD = 0.90
NEAR_DUP_MIN_TOKENS = 4
DHASH_MAX_HAMMING = 4
NEGATION_CUES = {"no", "not", "without", "never", "none", "nobody", "nothing", "neither", "nor", "dont", "doesnt", "isnt", "arent", "wasnt",
                 "werent", "cannot", "cant", "lacking", "absent", "except", "minus"}
PARSER_VERSION_GT = "evalmuse_gt_elements_v1"


# ------------------------------------------------------------------------------------------------------------------------------------ archive
class MultiPartFile(io.RawIOBase):
    """Read-only, seekable concatenation of the archive parts (no merged copy on disk)."""

    def __init__(self, parts):
        self.parts = [str(p) for p in parts]; self.sizes = [os.path.getsize(p) for p in self.parts]; self.total = sum(self.sizes)
        self.starts = np.cumsum([0] + self.sizes[:-1]).tolist(); self.pos = 0; self.fh = {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.total + off)
        return self.pos

    def readinto(self, b):
        n, out = len(b), 0
        for path, start, size in zip(self.parts, self.starts, self.sizes):
            if out >= n:
                break
            if self.pos < start + size and self.pos >= start:
                f = self.fh.get(path) or self.fh.setdefault(path, open(path, "rb")); f.seek(self.pos - start)
                chunk = f.read(min(n - out, start + size - self.pos)); b[out:out + len(chunk)] = chunk; out += len(chunk); self.pos += len(chunk)
        return out

    def close(self):
        for f in self.fh.values():
            f.close()
        super().close()


def open_archive(root: Path = EVALMUSE) -> zipfile.ZipFile:
    parts = sorted(root.glob("images.zip.part-a?"))
    if not parts:
        raise SystemExit(f"no archive parts under {root}")
    return zipfile.ZipFile(io.BufferedReader(MultiPartFile(parts), buffer_size=1 << 20))


# ------------------------------------------------------------------------------------------------------------------------------------- labels
def parse_element_key(key: str):
    """'puffin (animal)' -> ('puffin', 'animal'); the native code keeps key.rpartition('(')[0] as the element text."""
    head, sep, tail = key.rpartition("(")
    if not sep:
        return key.strip(), None
    return head.strip(), tail.rstrip(")").strip() or None


def flag_fraction(codes):
    """Image-level annotator flags coded 0 / 1 with -1 = no answer -> (fraction of 1 among answered, n answered, n missing, n other)."""
    codes = list(codes or []); ans = [c for c in codes if c in (0, 1)]
    return (float(np.mean(ans)) if ans else None), len(ans), sum(c == FLAG_MISSING for c in codes), sum(c not in (0, 1, FLAG_MISSING) for c in codes)


def votes_summary(votes):
    votes = list(votes or []); valid = [v for v in votes if v in VALID_VOTES]
    return {"votes": valid, "n_valid": len(valid), "n_pos": int(sum(valid)), "n_unknown_codes": len(votes) - len(valid),
            "y": (float(np.mean(valid)) if valid else None)}


def element_status(vs, image_bad_decomposition: bool):
    if image_bad_decomposition:
        return "bad_decomposition"
    if vs["n_valid"] == 0:
        return "unknown" if vs["n_unknown_codes"] else "missing"
    return "valid"


# -------------------------------------------------------------------------------------------------------------------------------------- spans
def _word_pattern(name: str) -> str:
    left = r"(?<![0-9A-Za-z])" if name[:1].isalnum() else ""; right = r"(?![0-9A-Za-z])" if name[-1:].isalnum() else ""
    return left + re.escape(name) + right


def find_spans(prompt: str, name: str):
    """Deterministic character spans [start, end) of an element in its prompt, every occurrence (no 'first occurrence' guess):
    exact_word (case-insensitive, word boundaries) -> space_hyphen (whitespace / hyphen insensitive, word boundaries) -> inflection (+s / +es /
    -s, word boundaries) -> unmatched.  No bare-substring rule ('cat' must not match inside 'concatenate').  Returns (spans, rule)."""
    n = name.strip()
    if not n:
        return [], "empty"
    spans = [(m.start(), m.end()) for m in re.finditer(_word_pattern(n), prompt, flags=re.IGNORECASE)]
    if spans:
        return spans, "exact_word"
    toks = [t for t in re.split(r"[\s\-]+", n) if t]
    if len(toks) > 1:
        pat = r"[\s\-]*".join(re.escape(t) for t in toks); pat = (r"(?<![0-9A-Za-z])" if toks[0][:1].isalnum() else "") + pat + (r"(?![0-9A-Za-z])" if toks[-1][-1:].isalnum() else "")
        spans = [(m.start(), m.end()) for m in re.finditer(pat, prompt, flags=re.IGNORECASE)]
        if spans:
            return spans, "space_hyphen"
    variants = [n + "s", n + "es"] + ([n[:-1]] if n.lower().endswith("s") and len(n) > 3 else [])
    for v in variants:
        spans = [(m.start(), m.end()) for m in re.finditer(_word_pattern(v), prompt, flags=re.IGNORECASE)]
        if spans:
            return spans, "inflection"
    return [], "unmatched"


def negation_cue(prompt: str, spans, window: int = 3) -> bool:
    """True when every span is preceded (within `window` words) by a negation cue: a heuristic flag, not a label."""
    if not spans:
        return False
    def cued(a):
        words = re.findall(r"[a-z']+", prompt[:a].lower())[-window:]
        return any(w.replace("'", "") in NEGATION_CUES or w.endswith("n't") for w in words)
    return all(cued(a) for a, _ in spans)


def normalize_prompt(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).lower(); t = re.sub(r"\s+", " ", t).strip()
    return t.strip(" .,;:!?\"'")


def prompt_tokens(norm: str):
    return set(re.findall(r"[a-z0-9]+", norm))


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------------------------------------------------------ duplicates
class UnionFind:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if rb < ra:
                ra, rb = rb, ra
            self.p[rb] = ra

    def groups(self):
        g = defaultdict(list)
        for i in self.p:
            g[self.find(i)].append(i)
        return {k: sorted(v) for k, v in g.items()}


def dhash64(img) -> int:
    from PIL import Image
    g = img.convert("L").resize((9, 8), Image.Resampling.LANCZOS); a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten(); return int("".join("1" if b else "0" for b in bits), 2)


def near_duplicate_hashes(hashes: dict, max_hamming: int = DHASH_MAX_HAMMING):
    """Pairs of keys whose 64-bit dHashes differ in <= max_hamming bits.  8 bands of 8 bits: two hashes within distance <= 7 agree on at
    least one band (pigeonhole), so every qualifying pair is a candidate; candidates are then checked exactly."""
    if max_hamming > 7:
        raise ValueError("the 8-band index is exact only up to Hamming distance 7")
    buckets = defaultdict(list); keys = sorted(hashes)
    for k in keys:
        h = hashes[k]
        for band in range(8):
            buckets[(band, (h >> (8 * band)) & 0xFF)].append(k)
    pairs = set()
    for members in buckets.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if bin(hashes[a] ^ hashes[b]).count("1") <= max_hamming:
                    pairs.add((a, b) if a < b else (b, a))
    return sorted(pairs)


def near_duplicate_prompts(norms: dict, threshold: float = NEAR_DUP_PROMPT_JACCARD, min_tokens: int = NEAR_DUP_MIN_TOKENS):
    """Pairs of keys whose normalized prompts have token-set Jaccard >= threshold (both with >= min_tokens tokens); exact duplicates excluded."""
    toks = {k: prompt_tokens(v) for k, v in norms.items()}; keys = sorted(k for k in toks if len(toks[k]) >= min_tokens)
    index = defaultdict(set)
    for k in keys:
        for t in toks[k]:
            index[t].add(k)
    pairs = set()
    for k in keys:
        cand = Counter()
        for t in toks[k]:
            for o in index[t]:
                if o > k:
                    cand[o] += 1
        for o, inter in cand.items():
            if norms[o] == norms[k]:
                continue
            union = len(toks[k]) + len(toks[o]) - inter
            if union and inter / union >= threshold:
                pairs.add((k, o))
    return sorted(pairs)


# ------------------------------------------------------------------------------------------------------------------------------------- split
def assign_roles(components: dict, image_counts: dict, seed: int = SPLIT_SEED, excluded=frozenset()):
    """components {cid: [image ids]} -> {cid: role}: components in a seeded permutation (of sorted ids) fill FIT to >= 80 % of the images,
    then CAL to >= 90 %, the rest DEV; CAL components then fill CAL_SELECT to >= half of CAL's images, the rest CAL_PROB.  Excluded
    components (external overlap) get role EXCLUDED_OVERLAP."""
    cids = sorted(c for c in components if c not in excluded); order = [cids[i] for i in np.random.default_rng(seed).permutation(len(cids))]
    total = sum(image_counts[c] for c in cids); roles = {c: "EXCLUDED_OVERLAP" for c in components if c in excluded}; acc = 0
    bounds = np.cumsum([f for _, f in ROLE_FRACTIONS]); names = [r for r, _ in ROLE_FRACTIONS]
    for c in order:
        frac = acc / total if total else 0.0; k = int(np.searchsorted(bounds, frac, side="right")); roles[c] = names[min(k, len(names) - 1)]
        acc += image_counts[c]
    cal = [c for c in order if roles[c] == "CAL"]; cal_total = sum(image_counts[c] for c in cal); acc = 0
    for c in cal:
        roles[c] = "CAL_SELECT" if acc < cal_total / 2 else "CAL_PROB"; acc += image_counts[c]
    return roles


def fit_weights(images):
    """images: [{'image_id','prompt_id'}] of FIT -> per-image weight N_images / (N_prompts * n_images_in_prompt) (prompt -> image uniform)."""
    per = Counter(r["prompt_id"] for r in images); n_img, n_p = len(images), len(per)
    return {r["image_id"]: n_img / (n_p * per[r["prompt_id"]]) for r in images}


def fit_prior(elements, image_w):
    """pi = E_D[y] with D = prompt -> image -> element uniform over FIT valid elements; returns (pi, weights {element_id: w})."""
    by_img = defaultdict(list)
    for e in elements:
        by_img[e["image_id"]].append(e)
    w = {}; num = den = 0.0
    for img, es in by_img.items():
        for e in es:
            w[e["element_id"]] = image_w[img] / len(es); num += w[e["element_id"]] * e["y"]; den += w[e["element_id"]]
    return (num / den if den else None), w


# ------------------------------------------------------------------------------------------------------------------------------- commands
def cmd_genai_prompts(a):
    import pyarrow.parquet as pq
    files = sorted(Path(a.parquet_dir).glob("*.parquet")); schema = pq.read_schema(files[0]).names
    label_like = [c for c in schema if re.search(r"score|rating|human|vote|label|rank|annot", c, re.I)]
    pcol = [c for c in schema if re.fullmatch(r"(?i)prompt|text|caption", c)]; icol = [c for c in schema if re.fullmatch(r"(?i)id|index|prompt_id|idx", c)]
    if not pcol or not icol:
        raise SystemExit(f"cannot identify id / prompt columns in {schema}")
    cols = [icol[0], pcol[0]]
    assert not set(cols) & set(label_like), "refusing to read a label-like column"
    rows = {}
    for f in files:
        t = pq.read_table(f, columns=cols).to_pydict()
        for i, p in zip(t[cols[0]], t[cols[1]]):
            rows.setdefault(str(i), p)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fh:
        for i in sorted(rows, key=lambda x: (len(x), x)):
            fh.write(json.dumps({"source": "genai_bench", "id": i, "prompt": rows[i]}, ensure_ascii=False) + "\n")
    print(json.dumps({"schema": schema, "label_like_columns_not_read": label_like, "read": cols, "n_prompts": len(rows)}, indent=1))
    return 0


def cmd_audit(a):
    t0 = time.time(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    raw = json.loads((EVALMUSE / "train_list.json").read_text()); raw = raw[: a.limit] if a.limit else raw
    audit = {"source": {"train_list_sha256": hashlib.sha256((EVALMUSE / "train_list.json").read_bytes()).hexdigest(), "n_records": len(raw)}}
    # ---- field dictionary
    fields = defaultdict(lambda: {"present": 0, "types": Counter(), "codes": Counter()})
    for r in raw:
        for k, v in r.items():
            fd = fields[k]; fd["present"] += 1; fd["types"][type(v).__name__] += 1
            if k in ("total_score", "promt_meaningless", "split_confidence", "attribute_confidence"):
                fd["codes"].update(v)
            elif k == "element_score":
                for vs in v.values():
                    fd["codes"].update(vs)
            elif k in ("type",):
                fd["codes"][v] += 1
    audit["fields"] = {k: {"present": v["present"], "types": dict(v["types"]), "codes": {str(c): n for c, n in v["codes"].most_common(20)}} for k, v in fields.items()}
    # ---- images and elements
    images, elements = [], []; problems = Counter(); match_rules = Counter(); cats = Counter()
    seen_img = set()
    for r in raw:
        img = r["img_path"]; gen = img.split("/")[0]
        if img in seen_img:
            problems["duplicate_img_path"] += 1; continue
        seen_img.add(img)
        tv = [v for v in r.get("total_score") or [] if isinstance(v, int) and 1 <= v <= 5]
        mf = flag_fraction(r.get("promt_meaningless")); sf = flag_fraction(r.get("split_confidence")); af = flag_fraction(r.get("attribute_confidence"))
        meaningless = mf[0] is not None and mf[0] > 0.5; bad_dec = sf[0] is not None and sf[0] > 0.5
        n_votes = len(r.get("total_score") or [])
        elist = []
        for key, votes in (r.get("element_score") or {}).items():
            name, cat = parse_element_key(key); vs = votes_summary(votes); spans, rule = find_spans(r["prompt"], name)
            if len(votes) != n_votes:
                problems["element_votes_len_ne_total_votes"] += 1
            st = element_status(vs, bad_dec); match_rules[rule] += 1; cats[cat] += 1
            scope = "main" if (st == "valid" and not meaningless) else ("excluded_meaningless" if meaningless else f"excluded_{st}")
            elist.append({"element_id": f"{img}::{key}", "image_id": img, "raw_key": key, "raw_name": name, "type": cat, "spans": [list(s) for s in spans],
                          "match_rule": rule, "negation_cue": negation_cue(r["prompt"], spans), **vs, "label_status": st, "scope": scope})
        n_main = sum(e["scope"] == "main" for e in elist)
        images.append({"image_id": img, "prompt_id": str(r["prompt_id"]), "prompt": r["prompt"], "generator_id": gen, "type": r.get("type"),
                       "source_dataset": "evalmuse_train", "raw_label_path": "train_list.json", "overall_votes": tv, "overall_mean": (float(np.mean(tv)) if tv else None),
                       "n_annotators": n_votes, "meaningless_frac": mf[0], "split_wrong_frac": sf[0], "attribute_unsure_frac": (1 - af[0]) if af[0] is not None else None,
                       "flag_missing_codes": mf[2] + sf[2] + af[2], "flag_other_codes": mf[3] + sf[3] + af[3],
                       "label_status": "meaningless" if meaningless else ("bad_decomposition" if bad_dec else "ok"),
                       "element_count": len(elist), "element_count_main": n_main, "parser_version": PARSER_VERSION_GT,
                       "prompt_sha256": sha256_text(r["prompt"]), "prompt_norm": normalize_prompt(r["prompt"])})
        elements += elist
    audit["counts"] = {"images": len(images), "elements": len(elements), "problems": dict(problems), "element_match_rules": dict(match_rules),
                       "element_categories": dict(cats.most_common()), "generators": dict(Counter(i["generator_id"] for i in images).most_common()),
                       "image_label_status": dict(Counter(i["label_status"] for i in images)), "element_label_status": dict(Counter(e["label_status"] for e in elements)),
                       "element_scope": dict(Counter(e["scope"] for e in elements)), "annotators_per_image": dict(Counter(i["n_annotators"] for i in images)),
                       "negation_cue_elements": sum(e["negation_cue"] for e in elements),
                       "multi_span_elements": sum(len(e["spans"]) > 1 for e in elements)}
    # ---- prompt identity
    pid_prompts = defaultdict(set); norm_pids = defaultdict(set)
    for i in images:
        pid_prompts[i["prompt_id"]].add(i["prompt"]); norm_pids[i["prompt_norm"]].add(i["prompt_id"])
    audit["prompts"] = {"prompt_ids": len(pid_prompts), "prompt_ids_with_several_texts": sum(len(v) > 1 for v in pid_prompts.values()),
                        "normalized_texts": len(norm_pids), "normalized_texts_with_several_ids": sum(len(v) > 1 for v in norm_pids.values())}
    pid_norm = {}
    for i in images:
        pid_norm.setdefault(i["prompt_id"], i["prompt_norm"])
    near_p = near_duplicate_prompts(pid_norm); audit["prompts"]["near_duplicate_prompt_id_pairs"] = len(near_p)
    # ---- external prompt text (exposure)
    ext_hits = defaultdict(set)
    for f in a.external or []:
        ext = [json.loads(l) for l in open(f) if l.strip()]
        ext_norm = {f"{e['source']}:{e['id']}": normalize_prompt(e["prompt"]) for e in ext}; by_norm = defaultdict(set)
        for k, v in ext_norm.items():
            by_norm[v].add(k)
        for pid, nrm in pid_norm.items():
            if nrm in by_norm:
                ext_hits[pid] |= by_norm[nrm]
        merged = {**{f"E::{k}": v for k, v in ext_norm.items()}, **{f"P::{k}": v for k, v in pid_norm.items()}}
        for x, y in near_duplicate_prompts(merged):
            if x.startswith("E::") != y.startswith("E::"):
                e_, p_ = (x, y) if x.startswith("E::") else (y, x); ext_hits[p_[3:]].add(e_[3:])
        audit.setdefault("external", {})[Path(f).name] = {"n_prompts": len(ext)}
    audit["prompts"]["prompt_ids_overlapping_external"] = len(ext_hits)
    # ---- image hashes (TRAIN members only)
    sha, dh = {}, {}; z = open_archive(); names = set(z.namelist()); want = [ZIP_PREFIX + i["image_id"] for i in images]
    missing = [w for w in want if w not in names]; audit["archive"] = {"members": len(names), "train_members_missing": len(missing),
                                                                     "non_train_file_members_never_read": sum(1 for n in names if not n.endswith("/")) - (len(want) - len(missing))}
    from PIL import Image
    bad_images = []
    for k, i in enumerate(images):
        member = ZIP_PREFIX + i["image_id"]
        if member not in names:
            i["image_sha256"] = None; continue
        data = z.read(member); i["image_sha256"] = hashlib.sha256(data).hexdigest(); sha[i["image_id"]] = i["image_sha256"]
        try:
            im = Image.open(io.BytesIO(data)); im.load(); i["width"], i["height"] = im.size; dh[i["image_id"]] = dhash64(im)
        except Exception as e:  # noqa: BLE001 — recorded, the image stays in the manifest with its status
            bad_images.append([i["image_id"], f"{type(e).__name__}"]); i["image_decode_error"] = True
        if k % 2000 == 0:
            print(f"[v7_data] hashed {k}/{len(images)} images ({time.time() - t0:.0f}s)", flush=True)
    sha_groups = defaultdict(list)
    for k, v in sha.items():
        sha_groups[v].append(k)
    near_i = near_duplicate_hashes(dh)
    audit["images"] = {"decode_errors": len(bad_images), "exact_duplicate_groups": sum(len(v) > 1 for v in sha_groups.values()),
                       "near_duplicate_pairs": len(near_i)}
    # ---- components: prompt_id, normalized text, near-duplicate prompts, identical / near-identical images
    uf = UnionFind([i["image_id"] for i in images]); by_pid = defaultdict(list); by_norm = defaultdict(list)
    for i in images:
        by_pid[i["prompt_id"]].append(i["image_id"]); by_norm[i["prompt_norm"]].append(i["image_id"])
    for grp in list(by_pid.values()) + list(by_norm.values()) + [v for v in sha_groups.values() if len(v) > 1]:
        for x in grp[1:]:
            uf.union(grp[0], x)
    for p1, p2 in near_p:
        uf.union(by_pid[p1][0], by_pid[p2][0])
    for x, y in near_i:
        uf.union(x, y)
    comps = uf.groups(); img_comp = {m: c for c, ms in comps.items() for m in ms}
    excluded = {img_comp[by_pid[pid][0]] for pid in ext_hits}
    roles = assign_roles(comps, {c: len(m) for c, m in comps.items()}, excluded=excluded)
    for i in images:
        i["prompt_group"] = img_comp[i["image_id"]]; i["role"] = roles[i["prompt_group"]]; i["external_overlap"] = sorted(ext_hits.get(i["prompt_id"], []))
    role_of = {i["image_id"]: i["role"] for i in images}
    for e in elements:
        e["role"] = role_of[e["image_id"]]
    sizes = Counter(len(m) for m in comps.values())
    audit["components"] = {"n": len(comps), "largest": max(sizes), "size_histogram": {str(k): v for k, v in sorted(sizes.items())},
                           "excluded_external_overlap": len(excluded)}
    audit["roles"] = {r: {"images": sum(i["role"] == r for i in images), "prompt_ids": len({i["prompt_id"] for i in images if i["role"] == r}),
                          "components": sum(1 for c in comps if roles[c] == r), "elements_main": sum(e["role"] == r and e["scope"] == "main" for e in elements)}
                      for r in ("FIT", "CAL_SELECT", "CAL_PROB", "DEV", "EXCLUDED_OVERLAP")}
    # ---- FIT weights and prior (prompt -> image -> element uniform over FIT main elements with a span or not: the prior is over labels)
    fit_imgs = [i for i in images if i["role"] == "FIT"]; iw = fit_weights(fit_imgs)
    fit_el = [e for e in elements if e["role"] == "FIT" and e["scope"] == "main"]; pi, ew = fit_prior(fit_el, iw)
    pi_spanned, _ = fit_prior([e for e in fit_el if e["spans"]], iw)
    for i in images:
        i["fit_weight"] = iw.get(i["image_id"])
    audit["fit"] = {"pi": pi, "pi_spanned": pi_spanned, "n_images": len(fit_imgs), "n_elements_main": len(fit_el), "weight_min": min(iw.values()), "weight_max": max(iw.values()),
                    "pi_unweighted": float(np.mean([e["y"] for e in fit_el])) if fit_el else None,
                    "elements_main_with_span": sum(bool(e["spans"]) for e in fit_el)}
    # ---- tokenizer coverage (FIT prompts; the M interface keeps raw prompts, uncased BERT, offsets)
    if a.bert:
        from transformers import BertTokenizer, BertTokenizerFast
        fast = BertTokenizerFast.from_pretrained(a.bert); slow = BertTokenizer.from_pretrained(a.bert, truncation_side="right"); slow.add_special_tokens({"bos_token": "[DEC]"})
        lens, mism = [], 0
        for i in fit_imgs:
            ids_f = fast(i["prompt"]).input_ids; lens.append(len(ids_f)); mism += ids_f != slow(i["prompt"]).input_ids
        lens = np.array(lens)
        audit["tokens"] = {"fit_prompts": len(lens), "fast_vs_slow_id_mismatch": int(mism), "max": int(lens.max()), "p99": float(np.percentile(lens, 99)),
                           "over_32": int((lens > 32).sum()), "over_128": int((lens > 128).sum()), "over_512": int((lens > 512).sum())}
    # ---- outputs
    with open(out / "manifest_images.jsonl", "w") as fh:
        for i in images:
            fh.write(json.dumps({k: v for k, v in i.items() if k != "prompt_norm"}, ensure_ascii=False) + "\n")
    with open(out / "manifest_elements.jsonl", "w") as fh:
        for e in elements:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    rng = np.random.default_rng(SPLIT_SEED + 99); pick = sorted(rng.choice(len(images), size=min(20, len(images)), replace=False).tolist())
    (out / "manual_check_records.json").write_text(json.dumps([{"image": images[k], "elements": [e for e in elements if e["image_id"] == images[k]["image_id"]]} for k in pick],
                                                              ensure_ascii=False, indent=1))
    audit["rules"] = {"split_seed": SPLIT_SEED, "role_fractions": dict(ROLE_FRACTIONS), "near_dup_prompt_jaccard": NEAR_DUP_PROMPT_JACCARD,
                      "near_dup_min_tokens": NEAR_DUP_MIN_TOKENS, "dhash_max_hamming": DHASH_MAX_HAMMING, "majority_rule": "flag fraction > 0.5 among answered (−1 = no answer)",
                      "parser_version": PARSER_VERSION_GT}
    audit["files_sha256"] = {f: hashlib.sha256((out / f).read_bytes()).hexdigest() for f in ("manifest_images.jsonl", "manifest_elements.jsonl")}
    audit["seconds"] = time.time() - t0
    (out / "audit.json").write_text(json.dumps(audit, indent=1, ensure_ascii=False)); print(json.dumps({k: audit[k] for k in ("counts", "roles", "fit")}, indent=1))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("genai-prompts"); g.add_argument("--parquet-dir", required=True); g.add_argument("--out", required=True)
    u = sub.add_parser("audit"); u.add_argument("--out", required=True); u.add_argument("--external", nargs="*"); u.add_argument("--bert"); u.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    return cmd_genai_prompts(a) if a.cmd == "genai-prompts" else cmd_audit(a)


if __name__ == "__main__":
    sys.exit(main())
