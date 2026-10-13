"""r7 R7-40c — Qwen2.5-VL-7B-Instruct quality runner on the frozen D1-30 evaluation sets (vl_baselines env: torch 2.9 cu128, transformers
4.57.6; any of RTX6000PRO / H100 / L40S).  One host process per job (weights staged to node-local disk once).
Host interface = R7-40a (scripts/r7_d1_40_host.py, frozen): selection after the token merger, before the LLM; kept merged tokens spliced in
raster order between the chat-template text before / after the image placeholder; every kept token keeps its official 3-D rotary id from
get_rope_index on the FULL sequence, every text token its full-sequence position; decoding continues at max position + 1.  Generation =
QwenCore.generate (frozen): left-padded batched greedy with the shipped repetition penalty 1.05 (verified against HF generate on a tiny
Qwen2.5-VL and by sentinels S5 / S5b on the real model in R7-40a).
Prompts (OPEN-R740-7, recommended): the D1-30 pinned prompt of each item ('prompt': GQA / TextVQA question + LLaVA short-answer suffix, no OCR
tokens; ChartQA query + the pinned lmms-eval suffix) as the user text after the image, Qwen chat template (shipped system prompt).
max_new_tokens 128 (GQA / TextVQA) / 16 (ChartQA) as D30-7.  Images: the set's image_file only (GQA: the extracted folder; TextVQA / ChartQA:
the bytes written by the D1-30 build); a missing file stops the unit (no archive fallback, no skip).
Configurations: scripts/r7_d1_40c_plan.py (full, static:<mech>:<f>, sel:<unit>:<f>, dyn:<unit>:<op>; f = fraction of the image's own N).
Selectors: the frozen R7-40b units (Selector40) on the host's merged tokens with the 40b PCA applied exactly as the feature cache
(fp32 centre-and-project of the bf16 merged tokens, rounded to fp16) and the Qwen input embeddings of the raw question; one item at a time
(no padding), so a selection does not depend on chunking, batch composition or a resume.  The visual tokens of an image are computed per
image (the official path's tower call), so they do not depend on chunking either; generation batches do (bf16), as in D1-30.
Order: a shard's items (D1-30 sharding by image group, r7_d1_30_run.load_items) are sorted by the image's merged-token count (from the image
header with the processor's own smart_resize; checked against the encoder output, a mismatch stops the unit), then group, then qid, and
processed in chunks of 32 items: less padding per generation batch; deterministic.
Output: <out>/<bench>/<cfg with ':' -> '__'>@s<i>of<n>.jsonl rows {qid, image_id, pred, kept, n_vis, f, n_new, hit_max} — one file per
(configuration, shard): single writer (D1-30 execution note E1); appended per chunk; resumed per (configuration, qid) over every file of the
configuration; done_shard<i>of<n>.json written only when every (configuration, item) of the shard is answered.
    python -I scripts/r7_d1_40c_run.py run --sets-dir S --bench gqa_dev --configs-file C --shard i/n --out O [--selectors-dir D --pca P]
    python -I scripts/r7_d1_40c_run.py check --sets-dir S --configs-dir C --lr-choice L --selectors-dir D --pca P --out O   (CPU, stage 0)
    python -I scripts/r7_d1_40c_run.py official --sets-dir S --bench gqa_dev --shard i/n --out O       (the batch-1 official full-row anchor, decision 15)
    python -I scripts/r7_d1_40c_run.py complete --configs-dir C --pred-dir P --out O                                       (CPU, stage 2)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7_d1_30_run as RUN  # noqa: E402  (frozen D1-30: prediction-file layout, resume, D1-30 sharding)
import r7_d1_40_masks as M4  # noqa: E402  (frozen R7-40a: fraction budgets, qid-seeded orderings on the merged grid)
import r7_d1_40c_plan as PL  # noqa: E402
import r7_vl_common as V  # noqa: E402

ANSWER_KEYS = ("answer", "answers", "label")                 # never held by the runner (the scorer alone reads them)
PATCH, MERGE = 14, 2
FACTOR = PATCH * MERGE                                        # 28 px per merged token side
SELECTORS = PL.R40 / "selectors"
PCA = PL.R40 / "cache" / "pca.pt"


class MissingImage(FileNotFoundError):
    """An evaluation image is not where the set says: the unit stops (never skipped, never read from an archive)."""


def open_image(path):
    from PIL import Image
    p = Path(path)
    if not p.is_file():
        raise MissingImage(f"image {p} is missing; R7-40c never reads an archive (zips deleted 2026-10-12) — stop and report")
    return Image.open(p).convert("RGB")


def smart_resize(height: int, width: int, factor: int = FACTOR, min_pixels: int = 256 * 28 * 28, max_pixels: int = 1280 * 28 * 28):
    """Verbatim logic of transformers 4.57.6 models/qwen2_vl/image_processing_qwen2_vl.py::smart_resize (used by the fast processor);
    a CPU test checks it against the installed function."""
    import math
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def merged_grid(path):
    """(rows, cols) of the image's merged-token grid from its header (PIL size, nothing decoded) and the R7-20 pixel bounds."""
    from PIL import Image
    p = Path(path)
    if not p.is_file():
        raise MissingImage(f"image {p} is missing")
    with Image.open(p) as im:
        w, h = im.size
    hb, wb = smart_resize(h, w)
    return hb // FACTOR, wb // FACTOR


def load_items(sets_dir, bench, shard="0/1"):
    """The shard's items (D1-30 sharding by image group) without their answers."""
    return [{k: v for k, v in it.items() if k not in ANSWER_KEYS} for it in RUN.load_items(sets_dir, bench, shard)]


def sort_by_tokens(items, grids):
    """Items ordered by the image's merged-token count, then D1-30 group order, then qid (deterministic; less padding per batch)."""
    rank = {}
    for it in items:
        rank.setdefault(it["image_id"], len(rank))
    return sorted(items, key=lambda it: (grids[it["image_file"]][0] * grids[it["image_file"]][1], rank[it["image_id"]], it["qid"]))


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------------------------------ selectors
class QwenSelectorBank:
    """Frozen R7-40b units (r7_d1_40_selectors.Selector40) + the 40b PCA.  Per item (no padding): gate logits, hard top-k at k = round(f·N)
    over the image's tokens, predicted risk r of the top-k mask at every fraction of the dynamic grid {0.125, 0.25, 0.5, 0.75, 1}
    (r7_d1_40_selectors.risk_grid's rule) and the dynamic choice 'smallest f with r <= tau_op' (tau from the unit's R7-40b log 'dynamic')."""

    def __init__(self, sel_dir, pca_path, units, device="cuda", tau=None):
        import torch
        import r7_d1_40_selectors as S4
        import r7_d1_selectors as SEL
        self.torch, self.S4, self.SEL, self.dev = torch, S4, SEL, device
        pca = torch.load(str(pca_path), map_location=device, weights_only=False)
        self.mean, self.basis = pca["mean"].float().to(device), pca["basis"].float().to(device)
        self.models, self.tau, self.files = {}, {}, {"pca": {"path": str(pca_path), "sha256": sha_file(pca_path)}}
        for u in units:
            pt = Path(sel_dir) / f"{u}.pt"; ck = torch.load(str(pt), map_location=device, weights_only=False)
            cfg = S4.Cfg40(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in ck["cfg"].items()})
            m = S4.Selector40(cfg, ck["objective"]).to(device); m.load_state_dict(ck["state_dict"]); m.eval(); self.models[u] = m
            lp = Path(sel_dir) / f"{u}.json"; self.files[u] = {"pt_sha256": sha_file(pt), "json_sha256": sha_file(lp) if lp.exists() else None}
            if lp.exists():
                self.tau[u] = {PL.fkey(f): float(v["tau"]) for f, v in json.loads(lp.read_text()).get("dynamic", {}).items()}
        for u, t in (tau or {}).items():
            self.tau[u] = {PL.fkey(f): float(v) for f, v in t.items()}

    # one item ---------------------------------------------------------------------------------------------------------------------
    def prepare(self, feat, grid, q_emb):
        """feat [N, 3584] merged tokens (bf16), grid (rows, cols), q_emb [Lq, 3584] Qwen input embeddings of the raw question."""
        torch = self.torch; rows, cols = int(grid[0]), int(grid[1]); n = int(feat.shape[0])
        if rows * cols != n:
            raise RuntimeError(f"merged grid {rows} x {cols} != {n} tokens")
        cfg = next(iter(self.models.values())).cfg if self.models else None
        if cfg is not None and (rows > cfg.max_rows or cols > cfg.max_cols):
            raise SystemExit(f"merged grid {rows} x {cols} exceeds the selectors' {cfg.max_rows} x {cfg.max_cols} position tables (D40b-4): stop")
        v = ((feat.float().to(self.dev) - self.mean) @ self.basis).half().float()[None]
        ar = torch.arange(n, device=self.dev)
        return {"q": q_emb.float().to(self.dev)[None], "pad": torch.zeros(1, q_emb.shape[0], dtype=torch.bool, device=self.dev), "v": v,
                "rows": (ar // cols)[None], "cols": (ar % cols)[None], "valid": torch.ones(1, n, dtype=torch.bool, device=self.dev), "n": n, "_c": {}}

    def _gates(self, unit, st):
        key = ("g", unit)
        if key not in st["_c"]:
            torch, m = self.torch, self.models[unit]
            with torch.no_grad():
                tok, pb = m.encode(st["q"], st["pad"], st["v"], self.S4.full_bias(st["valid"]), st["rows"], st["cols"])
                st["_c"][key] = (m.gates40(tok, st["n"]), pb)
        return st["_c"][key]

    def _keep(self, g, valid, k):
        return self.S4.hard_topk40(g, self.torch.as_tensor([int(k)], device=self.dev), valid)

    def kept(self, unit, st, f):
        """Raster-ordered kept indices of the hard top-k at fraction f (f = 1: every token)."""
        g, _ = self._gates(unit, st); n = st["n"]; k = n if f >= 1.0 else M4.budget(f, n)
        return self.torch.nonzero(self._keep(g, st["valid"], k)[0]).flatten().cpu().numpy()

    def risks(self, unit, st):
        """Predicted risk of the top-k mask at every fraction of the dynamic grid -> [5]."""
        key = ("r", unit)
        if key not in st["_c"]:
            torch, m = self.torch, self.models[unit]; g, pb = self._gates(unit, st); n = st["n"]; out = []
            with torch.no_grad():
                for f in PL.DYN_GRID:
                    keep = self._keep(g, st["valid"], n if f >= 1.0 else M4.budget(f, n))
                    pa = m.encode(st["q"], st["pad"], st["v"], self.SEL.hard_bias(keep), st["rows"], st["cols"])[1]
                    out.append(float(m.readout(pb, pa)[0]))
            st["_c"][key] = np.asarray(out)
        return st["_c"][key]

    def dynamic_fraction(self, unit, st, op):
        if unit not in self.tau or PL.fkey(op) not in self.tau[unit]:
            raise SystemExit(f"no CAL tau for {unit} at operating point {op}")
        R = self.risks(unit, st); ok = R <= self.tau[unit][PL.fkey(op)]; ok[-1] = True
        return float(PL.DYN_GRID[int(np.argmax(ok))])

    # a batch (timing: the efficient deployment form) ------------------------------------------------------------------------------
    def batch(self, states):
        """Pad prepared single-item states into one batch (visual keys padded with the frozen −1e9 key bias; question keys likewise)."""
        torch = self.torch; B = len(states); Nb = max(s["n"] for s in states); Lq = max(s["q"].shape[1] for s in states); E = states[0]["q"].shape[2]
        q = torch.zeros(B, Lq, E, device=self.dev); pad = torch.ones(B, Lq, dtype=torch.bool, device=self.dev)
        v = torch.zeros(B, Nb, states[0]["v"].shape[2], device=self.dev); valid = torch.zeros(B, Nb, dtype=torch.bool, device=self.dev)
        rows = torch.zeros(B, Nb, dtype=torch.long, device=self.dev); cols = torch.zeros_like(rows)
        for i, s in enumerate(states):
            lq, n = s["q"].shape[1], s["n"]; q[i, :lq] = s["q"][0]; pad[i, :lq] = False; v[i, :n] = s["v"][0]; valid[i, :n] = True
            rows[i, :n] = s["rows"][0]; cols[i, :n] = s["cols"][0]
        return {"q": q, "pad": pad, "v": v, "valid": valid, "rows": rows, "cols": cols, "ns": [s["n"] for s in states], "Nb": Nb}

    def select_batch(self, unit, states, f=None, op=None):
        """Batched selection for a list of prepared states: fixed fraction f, or dynamic operating point op -> (kept index arrays, fractions)."""
        torch, m, S4, SEL = self.torch, self.models[unit], self.S4, self.SEL; b = self.batch(states); ns = b["ns"]
        with torch.no_grad():
            tok, pb = m.encode(b["q"], b["pad"], b["v"], S4.full_bias(b["valid"]), b["rows"], b["cols"]); g = m.gates40(tok, b["Nb"])
            ks = lambda fr: torch.as_tensor([n if fr >= 1.0 else M4.budget(fr, n) for n in ns], device=self.dev)  # noqa: E731
            if op is None:
                chosen = [float(f)] * len(ns)
            else:
                R = []
                for fr in PL.DYN_GRID:
                    keep = S4.hard_topk40(g, ks(fr), b["valid"])
                    R.append(m.readout(pb, m.encode(b["q"], b["pad"], b["v"], SEL.hard_bias(keep), b["rows"], b["cols"])[1]).float().cpu().numpy())
                R = np.stack(R, 1); ok = R <= self.tau[unit][PL.fkey(op)]; ok[:, -1] = True; chosen = [float(PL.DYN_GRID[j]) for j in np.argmax(ok, 1)]
            k = torch.as_tensor([n if fr >= 1.0 else M4.budget(fr, n) for n, fr in zip(ns, chosen)], device=self.dev)
            keep = S4.hard_topk40(g, k, b["valid"]).cpu().numpy()
        return [np.flatnonzero(keep[i, :n]) for i, n in enumerate(ns)], chosen


# ------------------------------------------------------------------------------------------------------------------------------ items
class ItemState:
    """One item's host-side pieces: merged tokens, grid, prompt segments and their embeddings, full-sequence rotary ids, question ids."""

    def __init__(self, host, it, enc):
        feat, grid, thw = enc; self.it, self.feat, self.grid, self.thw = it, feat, (int(grid[0]), int(grid[1])), thw
        self.n = int(feat.shape[0])
        if self.grid[0] * self.grid[1] != self.n:
            raise RuntimeError(f"{it['qid']}: merged grid {self.grid} != {self.n} tokens")
        self.pre, self.post = host.segments(it["prompt"], True)
        self.pre_e, self.post_e = host.core.embed_ids(self.pre), host.core.embed_ids(self.post)
        self.pos_full = host.core.full_positions(self.pre, self.n, self.post, thw)
        self.q_ids = list(host.tok(it["question"], add_special_tokens=False).input_ids)
        self.sel, self.qv = None, None

    def selector_state(self, host, bank):
        if self.sel is None:
            self.sel = bank.prepare(self.feat, self.grid, host.core.embed_ids(self.q_ids))
        return self.sel

    def question_vector(self, host):
        if self.qv is None:
            self.qv = host.question_vector(self.it["question"])
        return self.qv


def static_order(mech, st: ItemState, host=None):
    rows, cols = st.grid; qid = st.it["qid"]
    if mech == "grid":
        return M4.order_grid(qid, rows, cols, with_logp=False)[0]
    if mech == "random":
        return M4.order_random(qid, rows, cols)[0]
    f = st.feat.float(); qv = st.question_vector(host).to(f.device)
    sc = ((f @ qv) / (f.norm(dim=1) * qv.norm() + 1e-12)).cpu().numpy()
    return M4.order_qsim(qid, sc, st.n)[0]


def kept_for(cfg, st: ItemState, host, bank):
    """(raster-ordered kept indices or None for every token, fraction): the configuration's selection for one item."""
    kind, a, f = PL.parse(cfg)
    if kind == "full":
        return None, 1.0
    if kind == "static":
        return M4.kept(static_order(a, st, host), M4.budget(f, st.n)), f
    sel = st.selector_state(host, bank)
    if kind == "sel":
        return bank.kept(a, sel, f), f
    fr = bank.dynamic_fraction(a, sel, f)
    return (None, 1.0) if fr >= 1.0 else (bank.kept(a, sel, fr), fr)


def assemble(host, st: ItemState, kk):
    """Spliced LLM input of one item: embeddings, (3, L) rotary ids (kept tokens keep their full-sequence ids), prompt ids (penalty history)."""
    torch = host.torch
    if kk is None:
        return (torch.cat([st.pre_e, st.feat, st.post_e]), st.pos_full,
                np.concatenate([st.pre, np.full(st.n, host.image_token_id), st.post]))
    kk = np.asarray(kk, dtype=np.int64)
    import r7_d1_40_host as QH
    return (torch.cat([st.pre_e, st.feat[torch.as_tensor(kk, device=st.feat.device)], st.post_e]), QH.prune_positions(st.pos_full, len(st.pre), st.n, kk),
            np.concatenate([st.pre, np.full(len(kk), host.image_token_id), st.post]))


def new_token_stats(tokens, eos_ids, max_new):
    """(number of generated tokens up to and including the first eos, whether the row hit max_new_tokens without an eos)."""
    for j, t in enumerate(tokens):
        if t in eos_ids:
            return j + 1, False
    return len(tokens), len(tokens) >= max_new


def check_tau(configs, bank):
    missing = sorted({c for c in configs if c.startswith("dyn:") and PL.fkey(PL.parse(c)[2]) not in bank.tau.get(c.split(":")[1], {})})
    if missing:
        raise SystemExit(f"no CAL tau for dynamic rows {missing[:5]}")


# ------------------------------------------------------------------------------------------------------------------------------ run
def run(a, host=None, bank=None, opener=None, grid_of=None):
    """Quality answers of one (benchmark, shard) for every configuration of the configs file."""
    items = load_items(a.sets_dir, a.bench, a.shard)
    if getattr(a, "limit", 0):                                                         # smoke only
        items = items[: a.limit]
    configs = PL.read_configs(a.configs_file); [PL.parse(c) for c in configs]
    if a.bench not in PL.MAX_NEW:
        raise SystemExit(f"no generation length for benchmark {a.bench}")
    max_new = PL.MAX_NEW[a.bench]; out = Path(a.out) / a.bench; out.mkdir(parents=True, exist_ok=True)
    done = {c: RUN.answered(out, c) for c in configs}
    todo = [it for it in items if any(it["qid"] not in done[c] for c in configs)]
    if not todo:
        mark_done(out, a, items, configs); print(f"[r7_d1_40c] {a.bench} shard {a.shard}: complete ({len(items)} items × {len(configs)} configs)"); return 0
    grid_of = grid_of or merged_grid; grids = {}
    for it in todo:                                                                    # headers only; a missing image stops here
        if it["image_file"] not in grids:
            grids[it["image_file"]] = tuple(grid_of(it["image_file"]))
    todo = sort_by_tokens(todo, grids)
    if host is None:
        host = load_host(a.out)
    units = sorted({PL.parse(c)[1] for c in configs if c.startswith(("sel:", "dyn:"))})
    if bank is None and units:
        bank = QwenSelectorBank(Path(a.selectors_dir), Path(a.pca), units)
    if units:
        check_tau(configs, bank)
    opener = opener or open_image; eos = set(host.core.eos_ids); t0 = time.time(); n_gen = 0; tag = RUN.shard_tag(a.shard)
    for c0 in range(0, len(todo), a.chunk):
        chunk = todo[c0:c0 + a.chunk]; enc = {}
        for it in chunk:                                                               # one tower call per image (the official path's call)
            if it["image_file"] not in enc:
                enc[it["image_file"]] = host.encode([opener(it["image_file"])])[0]
                if tuple(enc[it["image_file"]][1]) != grids[it["image_file"]]:
                    raise SystemExit(f"{it['image_file']}: encoder grid {tuple(enc[it['image_file']][1])} != header estimate {grids[it['image_file']]}")
        states = [ItemState(host, it, enc[it["image_file"]]) for it in chunk]
        for cfg in configs:
            need = [j for j, it in enumerate(chunk) if it["qid"] not in done[cfg]]
            if not need:
                continue
            sel = {j: kept_for(cfg, states[j], host, bank) for j in need}; rows_out = []
            for b0 in range(0, len(need), a.batch):
                part = need[b0:b0 + a.batch]; seqs, poss, pids = zip(*[assemble(host, states[j], sel[j][0]) for j in part])
                new, _ = host.core.generate(list(seqs), list(poss), list(pids), max_new_tokens=max_new)
                for j, toks, tx in zip(part, new, host.decode(new)):
                    kk, fr = sel[j]; nn, hit = new_token_stats(toks, eos, max_new)
                    rows_out.append({"qid": chunk[j]["qid"], "image_id": chunk[j]["image_id"], "pred": tx, "kept": states[j].n if kk is None else int(len(kk)),
                                     "n_vis": states[j].n, "f": fr, "n_new": nn, "hit_max": hit})
            with open(out / f"{RUN.cfg_file(cfg)}@{tag}.jsonl", "a") as fh:                # this shard's own file (single writer, E1)
                for r in rows_out:
                    fh.write(V.jdump(r) + "\n"); done[cfg].add(r["qid"])
            n_gen += len(rows_out)
        print(f"[r7_d1_40c] {a.bench} shard {a.shard}: {min(c0 + a.chunk, len(todo))}/{len(todo)} items, {n_gen} answers, {time.time() - t0:.0f}s", flush=True)
    mark_done(out, a, items, configs, seconds=time.time() - t0, answers=n_gen, bank=bank)
    return 0


def mark_done(out: Path, a, items, configs, seconds=None, answers=None, bank=None):
    """Completion marker (the driver's file gate): written only when every (configuration, item) of the shard has an answer."""
    for c in configs:
        have = RUN.answered(out, c)
        if any(it["qid"] not in have for it in items):
            raise SystemExit(f"incomplete after run: {c}")
    sets_file = Path(a.sets_dir) / f"{a.bench}.jsonl"
    (out / f"done_shard{a.shard.replace('/', 'of')}.json").write_text(V.jdump({
        "items": len(items), "configs": len(configs), "seconds": seconds, "answers_this_session": answers,
        "seconds_per_answer": (seconds / answers) if seconds and answers else None, "env": V.env_info(),
        "sets_file_sha256": sha_file(sets_file) if sets_file.exists() else None, "configs_sha256": sha_file(a.configs_file),
        "selector_files": getattr(bank, "files", None), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))


# ------------------------------------------------------------------------------------------------------------------------------ official anchor
OFFICIAL = "full_official"          # coordinator decision 15 (2026-10-13): the full row through the official HF path at batch 1, a secondary diagnostic


def official_generate(host, prompt, image, max_new_tokens):
    """The official HF path at batch 1 — exactly the call of the frozen r7_d1_40_host.QwenPrefill.official_answer (the R7-20 QwenHost path and
    the R7-40a S5b reference): the processor builds the chat-template prompt with the image (Qwen's own preprocessing), model.generate with
    do_sample False, one beam, the shipped generation config (repetition penalty 1.05, eos / pad ids), KV cache; the new tokens decoded with
    skip_special_tokens and stripped.  Returns (text, visual-token count, new token ids)."""
    torch = host.torch; text = host.chat_text(prompt, image is not None)
    inputs = host.proc(text=[text], images=[image] if image is not None else None, return_tensors="pt").to(getattr(host.core, "dev", "cuda"))
    with torch.inference_mode():
        out = host.model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=max_new_tokens, use_cache=True,
                                  output_logits=True, return_dict_in_generate=True)
    n_in = inputs["input_ids"].shape[1]; new = out.sequences[:, n_in:]
    return (host.proc.batch_decode(new, skip_special_tokens=True)[0].strip(), int((inputs["input_ids"] == host.image_token_id).sum()),
            [int(x) for x in new[0].tolist()])


def load_host(out):
    """The Qwen host with the frozen R7-20 / R7-40 pixel bounds checked (one load per job)."""
    import r7_d1_40_host as QH
    host = QH.QwenPrefill(Path(out)); ip = host.proc.image_processor
    if (ip.size["shortest_edge"], ip.size["longest_edge"], ip.patch_size, ip.merge_size) != (QH.MIN_PIXELS, QH.MAX_PIXELS, PATCH, MERGE):
        raise SystemExit(f"processor settings differ from the R7-20 / R7-40 bounds: {ip.size} patch {ip.patch_size} merge {ip.merge_size}")
    return host


def run_official(a, host=None, opener=None):
    """The batch-1 official full-row anchor of one (benchmark, shard): one official HF generate call per item, in the D1-30 shard order, appended
    to <out>/<bench>/full_official@s<i>of<n>.jsonl (single writer) with the same fields as every other row (kept = n_vis = the processor's
    visual-token count, f = 1, n_new, hit_max) + 'path'; resumed per qid; marker done_official_shard<i>of<n>.json only when every item is
    answered.  Reported next to the batched full row (scorer: quality, identical-answer fraction, full − anchor); never in the primary family."""
    items = load_items(a.sets_dir, a.bench, a.shard)
    if getattr(a, "limit", 0):                                                         # smoke only
        items = items[: a.limit]
    if a.bench not in PL.MAX_NEW:
        raise SystemExit(f"no generation length for benchmark {a.bench}")
    max_new = PL.MAX_NEW[a.bench]; out = Path(a.out) / a.bench; out.mkdir(parents=True, exist_ok=True)
    done = RUN.answered(out, OFFICIAL); todo = [it for it in items if it["qid"] not in done]
    if not todo:
        mark_official(out, a, items); print(f"[r7_d1_40c] {a.bench} official shard {a.shard}: complete ({len(items)} items)"); return 0
    host = host or load_host(a.out); opener = opener or open_image; eos = set(host.core.eos_ids); t0 = time.time(); n = 0
    with open(out / f"{RUN.cfg_file(OFFICIAL)}@{RUN.shard_tag(a.shard)}.jsonl", "a") as fh:      # this shard's own file (single writer, E1)
        for it in todo:
            txt, n_vis, new = official_generate(host, it["prompt"], opener(it["image_file"]), max_new); nn, hit = new_token_stats(new, eos, max_new)
            fh.write(V.jdump({"qid": it["qid"], "image_id": it["image_id"], "pred": txt, "kept": n_vis, "n_vis": n_vis, "f": 1.0, "n_new": nn, "hit_max": hit,
                              "path": "official HF generate, batch 1"}) + "\n"); n += 1
            if n % 64 == 0:
                fh.flush(); print(f"[r7_d1_40c] {a.bench} official shard {a.shard}: {n}/{len(todo)} items, {time.time() - t0:.0f}s", flush=True)
    mark_official(out, a, items, seconds=time.time() - t0, answers=n)
    return 0


def mark_official(out: Path, a, items, seconds=None, answers=None):
    have = RUN.answered(out, OFFICIAL)
    if any(it["qid"] not in have for it in items):
        raise SystemExit(f"incomplete after run: {OFFICIAL}")
    sets_file = Path(a.sets_dir) / f"{a.bench}.jsonl"
    (out / f"done_official_shard{a.shard.replace('/', 'of')}.json").write_text(V.jdump({
        "items": len(items), "row": OFFICIAL, "seconds": seconds, "answers_this_session": answers,
        "seconds_per_item": (seconds / answers) if seconds and answers else None, "env": V.env_info(),
        "sets_file_sha256": sha_file(sets_file) if sets_file.exists() else None, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))


# ------------------------------------------------------------------------------------------------------------------------------ check (stage 0, CPU)
def grid_summary(sets_dir, benches=PL.BENCHES):
    """Per benchmark: images, missing image files, merged-grid maxima, images beyond the selectors' 64 × 64 tables, token-count summary.
    Reads each item's image_file and the image header only (no answer, no pixel decoded)."""
    out = {}
    for b in benches:
        items = load_items(sets_dir, b); grids, missing = {}, []
        for it in items:
            if it["image_file"] in grids or it["image_file"] in missing:
                continue
            try:
                grids[it["image_file"]] = merged_grid(it["image_file"])
            except MissingImage:
                missing.append(it["image_file"])
        n = np.array([grids[it["image_file"]][0] * grids[it["image_file"]][1] for it in items if it["image_file"] in grids])
        out[b] = {"items": len(items), "groups": len({it["image_id"] for it in items}), "images": len(grids), "missing_images": len(missing),
                  "missing_examples": missing[:5], "grid_max_rows": max((g[0] for g in grids.values()), default=0),
                  "grid_max_cols": max((g[1] for g in grids.values()), default=0), "images_over_64x64": sum(1 for g in grids.values() if g[0] > 64 or g[1] > 64),
                  "tokens_per_item": {"mean": float(n.mean()) if len(n) else None, "median": float(np.median(n)) if len(n) else None,
                                      "p90": float(np.percentile(n, 90)) if len(n) else None, "min": int(n.min()) if len(n) else None,
                                      "max": int(n.max()) if len(n) else None, "sum": int(n.sum()) if len(n) else 0}}
    return out


def check(a):
    """Stage 0 (CPU, after the freeze and R7-40b's completion; reads no answer, no prediction, no result): the frozen D1-30 sets match their
    build summary (sha256, counts); every item's image file exists and its merged grid (header + smart_resize) fits the selectors' 64 × 64
    tables (D40b-4: 'DEV checked at 40c'); token-count summary per benchmark; the 9 units used carry .pt and CAL taus; configuration files
    written; sha256 of every selector file, the PCA and lr_choice.json -> <out>/check.json ('pass') and <configs>/rows.json."""
    sets = Path(a.sets_dir); summ = json.loads((sets / "d1_30_sets.json").read_text()); ch = json.loads(Path(a.lr_choice).read_text())
    main = PL.unit_names(ch); res = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "benches": {}, "problems": []}
    for b in PL.BENCHES:
        f = sets / f"{b}.jsonl"; s = sha_file(f)
        if s != summ[b]["sha256"]:
            res["problems"].append(f"{b}: sha256 {s[:12]} != build summary {summ[b]['sha256'][:12]}")
        gs = grid_summary(sets, [b])[b]; res["benches"][b] = gs; items_n, missing, over = gs["items"], gs["missing_images"], gs["images_over_64x64"]
        if items_n != summ[b]["items"]:
            res["problems"].append(f"{b}: {items_n} items != build summary {summ[b]['items']}")
        if missing:
            res["problems"].append(f"{b}: {missing} missing images")
        if over:
            res["problems"].append(f"{b}: {over} images with a merged grid beyond the 64 × 64 position tables (decision needed, D40b-4)")
    sel = Path(a.selectors_dir); rows = {"lr_choice": {"file": str(a.lr_choice), "sha256": sha_file(a.lr_choice), "chosen_lr": {o: ch[o]["chosen_lr"] for o in PL.OBJECTIVES}},
                                         "pca": {"file": str(a.pca), "sha256": sha_file(a.pca) if Path(a.pca).exists() else None}, "units": {}}
    if rows["pca"]["sha256"] is None:
        res["problems"].append("PCA missing")
    for o in PL.OBJECTIVES:
        for u in main[o]:
            pt, js = sel / f"{u}.pt", sel / f"{u}.json"
            if not (pt.exists() and js.exists()):
                res["problems"].append(f"unit {u}: .pt or .json missing"); continue
            dyn = json.loads(js.read_text()).get("dynamic", {})                              # keys only (the operating points), no value printed
            if sorted(PL.fkey(k) for k in dyn) != sorted(PL.OPS):
                res["problems"].append(f"unit {u}: CAL taus missing ({sorted(dyn)})")
            rows["units"][u] = {"pt_sha256": sha_file(pt), "json_sha256": sha_file(js)}
    cfg = PL.write_configs(ch, Path(a.configs_dir), a.rows); rows["configs"] = cfg
    for b in PL.BENCHES:
        rows[f"configs_{b}_sha256"] = sha_file(Path(a.configs_dir) / f"configs_{b}.txt")
    Path(a.configs_dir).mkdir(parents=True, exist_ok=True); (Path(a.configs_dir) / "rows.json").write_text(json.dumps(rows, indent=1))
    res["configs"] = cfg; res["pass"] = not res["problems"]
    Path(a.out).mkdir(parents=True, exist_ok=True); (Path(a.out) / "check.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
    return 0 if res["pass"] else 5


def complete(a):
    """Stage 2 (CPU): every configured row covers exactly each benchmark's items (prediction files read for question ids only; no score)
    and every shard marker exists -> <out>/completeness.json."""
    res = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "benches": {}, "problems": []}
    for b in a.benches:
        qids = {it["qid"] for it in load_items(a.sets_dir, b)}; configs = PL.read_configs(Path(a.configs_dir) / f"configs_{b}.txt")
        out = Path(a.pred_dir) / b; short = {}
        for c in configs:
            have = RUN.answered(out, c)
            if have != qids:
                short[c] = {"missing": len(qids - have), "unexpected": len(have - qids)}
        if getattr(a, "official", True):                                               # the batch-1 official anchor (decision 15)
            have = RUN.answered(out, OFFICIAL)
            if have != qids:
                short[OFFICIAL] = {"missing": len(qids - have), "unexpected": len(have - qids)}
        shards = PL.QUALITY_SHARDS.get(b, 1); osh = PL.OFFICIAL_SHARDS.get(b, 1)
        markers = [f"done_shard{i}of{shards}.json" for i in range(shards)] + ([f"done_official_shard{i}of{osh}.json" for i in range(osh)] if getattr(a, "official", True) else [])
        miss = [m for m in markers if not (out / m).exists()]
        res["benches"][b] = {"items": len(qids), "configs": len(configs), "configs_incomplete": len(short), "examples": dict(list(short.items())[:5]),
                             "markers_missing": miss}
        if short or miss:
            res["problems"].append(f"{b}: {len(short)} incomplete configurations, {len(miss)} missing markers")
    res["complete"] = not res["problems"]
    Path(a.out).mkdir(parents=True, exist_ok=True); (Path(a.out) / "completeness.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
    return 0 if res["complete"] else 7


def main() -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--sets-dir", required=True); r.add_argument("--bench", required=True); r.add_argument("--configs-file", required=True)
    r.add_argument("--shard", default="0/1"); r.add_argument("--out", required=True); r.add_argument("--chunk", type=int, default=32); r.add_argument("--batch", type=int, default=32)
    r.add_argument("--selectors-dir", default=str(SELECTORS)); r.add_argument("--pca", default=str(PCA)); r.add_argument("--limit", type=int, default=0)
    c = sub.add_parser("check"); c.add_argument("--sets-dir", default=str(PL.D30_SETS)); c.add_argument("--configs-dir", required=True)
    c.add_argument("--lr-choice", required=True); c.add_argument("--selectors-dir", default=str(SELECTORS)); c.add_argument("--pca", default=str(PCA))
    c.add_argument("--out", required=True); c.add_argument("--rows", choices=PL.ROW_SETS, default="core")
    o = sub.add_parser("official"); o.add_argument("--sets-dir", default=str(PL.D30_SETS)); o.add_argument("--bench", required=True)
    o.add_argument("--shard", default="0/1"); o.add_argument("--out", required=True); o.add_argument("--limit", type=int, default=0)
    gsub = sub.add_parser("gridcheck"); gsub.add_argument("--sets-dir", default=str(PL.D30_SETS)); gsub.add_argument("--out", required=True)
    k = sub.add_parser("complete"); k.add_argument("--sets-dir", default=str(PL.D30_SETS)); k.add_argument("--configs-dir", required=True)
    k.add_argument("--pred-dir", required=True); k.add_argument("--out", required=True); k.add_argument("--benches", nargs="+", default=list(PL.BENCHES))
    k.add_argument("--no-official", dest="official", action="store_false")
    a = ap.parse_args()
    if a.cmd == "gridcheck":                                                           # image headers only (no answer): token counts and the D40b-4 check
        res = grid_summary(a.sets_dir); Path(a.out).mkdir(parents=True, exist_ok=True); (Path(a.out) / "gridcheck.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1)); return 0
    return {"run": run, "official": run_official, "check": check, "complete": complete}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
