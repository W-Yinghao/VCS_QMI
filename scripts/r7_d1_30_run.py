"""r7 R7-30 D1-30 quality runner (LLaVA-1.5-7B host in the SparseVLM env; H100 / L40S).  One host process per job.  The images of a chunk
are encoded once and every configuration's answer is generated from the same projected visual tokens; selection happens at the LLM input
(kept tokens spliced in raster order, positions re-indexed contiguously), as frozen for D1-10 / D1-20.
Configurations on the fixed budget grid k ∈ {72, 144, 288, 432} (12.5 stress / 25 / 50 / 75 %) and 576 (full):
  full                 all 576 tokens;  full_ocr (TextVQA only): the LLaVA OCR-token prompt, published-number anchor (not on any curve)
  static:<mech>:<k>    untrained D1-10 mechanisms: grid (uniform spatial), random, qsim (question similarity; reported as an extra baseline)
  sel:<unit>:<k>       hard top-k of a frozen D1-20 unit's gate logits (S1 / S2 / S3 chosen-rate units × seeds 0-2; A1 / A2 × seeds 0-2)
  dyn:<unit>:<f>       dynamic budget (S1 / S2 / S3 units): per item the smallest k in {72, 144, 288, 432, 576} whose top-k mask has predicted
                       risk r <= tau_f (tau_f fixed on CAL in D1-20 for mean kept fraction f = 0.25 / 0.5 / 0.75); same rule as
                       r7_d1_selectors.dynamic_tau
The user turn is '<image>\\n' + item['prompt'] (vicuna_v1), where the item's prompt carries its benchmark's official suffix.  Greedy;
max_new_tokens 128 for GQA / TextVQA (LLaVA model_vqa_loader default), 16 for ChartQA (pinned lmms-eval chartqa.yaml).
SparseVLM rows: scripts/r7_d1_30_sparsevlm.py.  Output <out>/<bench>/<config with ':' -> '__'>@s<i>of<n>.jsonl rows {qid, image_id, pred,
kept}: one file per (configuration, shard), so no file has two writers (execution note D1-30-E1; NFSv3 appends from different clients are
not atomic); appended per chunk and resumed per (config, qid) over every file of the configuration (cfg_paths).
    python scripts/r7_d1_30_run.py --sets-dir S --bench gqa_dev --configs-file C --shard i/n --out O [--selectors-dir D --pca P]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7_d1_masks as MK  # noqa: E402
import r7_vl_common as V  # noqa: E402

D1 = Path("/home/infres/yinwang/CS_QMI/outputs/R7_D1")
N_VIS = 576
GRID = tuple(MK.BUDGETS) + (N_VIS,)
FRACTIONS = (0.25, 0.5, 0.75)
OBJECTIVES = ("vcs", "logistic", "direct")
SEEDS = (0, 1, 2)
MAX_NEW = {"gqa_dev": 128, "textvqa_val": 128, "chartqa_val": 16, "gqa_fixture": 128, "textvqa_fixture": 128, "chartqa_proxy": 16}


def unit_names(lr_choice: dict):
    """Selector unit tags from D1-20's lr_choice.json: main {objective: [3 tags]}, ablations {A1: [...], A2: [...]}."""
    lr = {o: lr_choice[o]["chosen_lr"] for o in OBJECTIVES}
    main = {o: [f"{o}_lr{lr[o]:g}_s{s}" for s in SEEDS] for o in OBJECTIVES}
    abl = {"A1": [f"vcs_lr{lr['vcs']:g}_s{s}_noq" for s in SEEDS], "A2": [f"direct_lr{lr['direct']:g}_s{s}_dtarget" for s in SEEDS]}
    return main, abl


def config_list(main_units, abl_units, ocr_anchor=False):
    cfg = ["full"] + (["full_ocr"] if ocr_anchor else []) + [f"static:{m}:{k}" for m in ("grid", "random", "qsim") for k in MK.BUDGETS]
    cfg += [f"sel:{u}:{k}" for u in list(main_units) + list(abl_units) for k in MK.BUDGETS]
    cfg += [f"dyn:{u}:{f}" for u in main_units for f in FRACTIONS]
    return cfg


def cfg_file(cfg: str) -> str:
    return cfg.replace(":", "__")


def shard_tag(shard: str) -> str:
    i, n = (int(x) for x in shard.split("/"))
    return f"s{i}of{n}"


def cfg_paths(out: Path, cfg: str):
    """Every file holding answers of one configuration: <cfg>.jsonl (one writer: runs before execution note D1-30-E1) and the per-shard files
    <cfg>@s<i>of<n>.jsonl (one writer each).  No two processes append to the same file: outputs/ is NFSv3, where O_APPEND from different
    clients is not atomic (lines can be lost or interleaved)."""
    base = cfg_file(cfg)
    return [p for p in [Path(out) / f"{base}.jsonl"] + sorted(Path(out).glob(f"{base}@*.jsonl")) if p.exists()]


def answered(out: Path, cfg: str):
    """qids with a parseable answer for one configuration, over all its files."""
    return {r["qid"] for p in cfg_paths(out, cfg) for r in V.read_jsonl(p) if "qid" in r and "pred" in r}


def user_segments(host, text: str):
    """Token ids before / after the image placeholder of the user turn '<image>\\n' + text (vicuna_v1, as LLaVA's model_vqa_loader).  Unlike
    LlavaPrefill.segments, nothing is appended: the item's prompt already carries its benchmark's official suffix."""
    if not hasattr(host, "_tit"):                                                      # CPU test fakes
        return host.segments(text, True)
    conv = host._conv["vicuna_v1"].copy(); conv.append_message(conv.roles[0], host._img_tok + "\n" + text); conv.append_message(conv.roles[1], None)
    ids = host._tit(conv.get_prompt(), host.tok, host._idx, return_tensors="pt"); i = int((ids == host._idx).nonzero()[0, 0])
    return ids[:i], ids[i + 1:]


def question_embeddings(host, questions):
    """Frozen LLaVA input embeddings of the raw question text (no special tokens) — exactly the D1-20 cache's question input."""
    return [host.embed_ids(host.tok(q, add_special_tokens=False, return_tensors="pt").input_ids[0]) for q in questions]


class SelectorBank:
    """Frozen D1-20 units + the D1-20 PCA on the host's visual tokens.  prepare() builds one chunk's inputs; gates / risks are cached per
    (unit, chunk).  tau: {unit: {fraction: tau}} from each unit's D1-20 log ('dynamic'), or an explicit override (smoke only)."""

    def __init__(self, sel_dir: Path, pca_path: Path, units, device="cuda", tau=None):
        import torch
        import r7_d1_selectors as SEL
        self.torch, self.SEL, self.dev = torch, SEL, device
        pca = torch.load(str(pca_path), map_location=device, weights_only=False)
        self.mean, self.basis = pca["mean"].float().to(device), pca["basis"].float().to(device)
        self.models, self.tau = {}, {}
        for u in units:
            ck = torch.load(str(Path(sel_dir) / f"{u}.pt"), map_location=device, weights_only=False)
            cfg = SEL.Cfg(**{k: (tuple(v) if k == "budgets" else v) for k, v in ck["cfg"].items()})
            m = SEL.Selector(cfg, ck["objective"]).to(device); m.load_state_dict(ck["state_dict"]); m.eval(); self.models[u] = m
            lp = Path(sel_dir) / f"{u}.json"
            if lp.exists():
                self.tau[u] = {float(f): v["tau"] for f, v in json.loads(lp.read_text()).get("dynamic", {}).items()}
        for u, t in (tau or {}).items():
            self.tau[u] = {float(f): float(v) for f, v in t.items()}
        self._cache = {}

    def prepare(self, feats, questions_emb):
        """feats [n, 576, 4096] host visual tokens -> PCA 512 rounded to fp16 exactly like the D1-20 cache; padded question embeddings."""
        torch = self.torch; self._cache = {}
        v = ((feats.float() - self.mean) @ self.basis).half().float()
        L = max(e.shape[0] for e in questions_emb); q = torch.zeros(len(questions_emb), L, questions_emb[0].shape[1], device=self.dev)
        pad = torch.ones(len(questions_emb), L, dtype=torch.bool, device=self.dev)
        for i, e in enumerate(questions_emb):
            q[i, :e.shape[0]] = e.float(); pad[i, :e.shape[0]] = False
        return {"q": q, "pad": pad, "v": v, "n": v.shape[0]}

    def _gates(self, unit, st):
        key = ("g", unit)
        if key not in self._cache:
            torch, m = self.torch, self.models[unit]
            with torch.no_grad():
                tok, pb = m.encode(st["q"], st["pad"], st["v"], torch.zeros(st["n"], N_VIS, device=self.dev))
                self._cache[key] = (m.gates(tok), pb)
        return self._cache[key]

    def _topk(self, g, ks):
        return self.SEL.hard_topk(g, self.torch.as_tensor(np.asarray(ks), device=self.dev))

    def risks(self, unit, st):
        """Predicted risk r of the top-k mask at every grid budget -> [n, 5] (k = 72, 144, 288, 432, 576)."""
        key = ("r", unit)
        if key not in self._cache:
            torch, m, SEL = self.torch, self.models[unit], self.SEL; g, pb = self._gates(unit, st); cols = []
            with torch.no_grad():
                for k in GRID:
                    keep = self._topk(g, [k] * st["n"])
                    cols.append(m.readout(pb, m.encode(st["q"], st["pad"], st["v"], SEL.hard_bias(keep))[1]).float().cpu().numpy())
            self._cache[key] = np.stack(cols, 1)
        return self._cache[key]

    def dynamic_budgets(self, unit, st, frac):
        """Smallest grid budget with r <= tau (576 when none), as r7_d1_selectors.dynamic_tau."""
        R = self.risks(unit, st); ok = R <= self.tau[unit][frac]; ok[:, -1] = True
        return np.asarray(GRID)[np.argmax(ok, 1)]

    def kept(self, unit, st, budget=None, frac=None):
        g, _ = self._gates(unit, st)
        ks = [budget] * st["n"] if budget is not None else self.dynamic_budgets(unit, st, frac).tolist()
        keep = self._topk(g, ks)
        return [self.torch.nonzero(keep[i]).flatten().cpu().numpy() for i in range(st["n"])]


def calibrate_tau(R, fractions=FRACTIONS):
    """Smoke only: tau per fraction on the given risks R [n, 5] (the bisection of r7_d1_selectors.dynamic_tau, numpy)."""
    fr = np.asarray(GRID) / N_VIS; out = {}

    def mean_frac(t):
        ok = R <= t; ok[:, -1] = True
        return float(fr[np.argmax(ok, 1)].mean())
    for f in fractions:
        lo, hi = float(R.min()) - 1.0, float(R.max()) + 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if mean_frac(mid) > f else (lo, mid)
        out[f] = hi
    return out


def kept_for(cfg, items, feats, host, bank, st=None, qvecs=None):
    """Kept indices per item for one configuration (None = all 576)."""
    if cfg in ("full", "full_ocr"):
        return [None] * len(items)
    kind, a, b = cfg.split(":")
    if kind == "static":
        out = []
        for i, it in enumerate(items):
            if a == "qsim":
                f = feats[i].float(); qv = qvecs[i] if qvecs is not None else host.question_vector(it["question"])
                sc = ((f @ qv) / (f.norm(dim=1) * qv.norm() + 1e-12)).cpu().numpy(); order = MK.order_qsim(it["qid"], sc)[0]
            else:
                order = (MK.order_random if a == "random" else MK.order_grid)(it["qid"])[0]
            out.append(MK.kept(order, int(b)))
        return out
    if st is None:
        st = bank.prepare(feats, question_embeddings(host, [it["question"] for it in items]))
    return bank.kept(a, st, budget=int(b)) if kind == "sel" else bank.kept(a, st, frac=float(b))


def sequence(host, it, feat, kk, cfg):
    torch = host.torch; text = it["prompt_ocr_anchor"] if cfg == "full_ocr" else it["prompt"]
    pre, post = user_segments(host, text); vis = feat if kk is None else feat[torch.as_tensor(np.asarray(kk), device=feat.device)]
    return torch.cat([host.embed_ids(pre), vis, host.embed_ids(post)])


def load_items(sets_dir, bench, shard="0/1"):
    items = [json.loads(x) for x in (Path(sets_dir) / f"{bench}.jsonl").read_text().splitlines() if x.strip()]
    i, n = (int(x) for x in shard.split("/")); groups = sorted({it["image_id"] for it in items}, key=lambda g: V._h("d1-30-shard", g))
    mine = set(groups[i::n]); rank = {g: j for j, g in enumerate(groups)}
    return sorted((it for it in items if it["image_id"] in mine), key=lambda it: (rank[it["image_id"]], it["qid"]))


def encode_unique(host, chunk, opener):
    """Encode each distinct image of the chunk once; returns per-item visual tokens [n, 576, 4096]."""
    files = sorted({it["image_file"] for it in chunk}); f = host.encode([opener(fp) for fp in files]); pos = {fp: j for j, fp in enumerate(files)}
    return f[host.torch.as_tensor([pos[it["image_file"]] for it in chunk], device=f.device)]


def check_tau(configs, bank):
    missing = sorted({c.split(":")[1] for c in configs if c.startswith("dyn:") and float(c.split(":")[2]) not in bank.tau.get(c.split(":")[1], {})})
    if missing:
        raise SystemExit(f"no CAL tau for dynamic units {missing}")


def run(a, host=None, bank=None, opener=None):
    from PIL import Image
    items = load_items(a.sets_dir, a.bench, a.shard)
    if getattr(a, "limit", 0):                                                         # smoke only
        items = items[: a.limit]
    configs = [c.strip() for c in Path(a.configs_file).read_text().splitlines() if c.strip() and not c.startswith("#")]
    out = Path(a.out) / a.bench; out.mkdir(parents=True, exist_ok=True)
    done = {c: answered(out, c) for c in configs}
    todo = [it for it in items if any(it["qid"] not in done[c] for c in configs)]
    if not todo:
        mark_done(out, a, items, configs); print(f"[r7_d1_30] {a.bench} shard {a.shard}: complete ({len(items)} items × {len(configs)} configs)"); return 0
    if host is None:
        import r7_d1_teacher as TE
        host = TE.LlavaPrefill(Path(a.out))
    units = sorted({c.split(":")[1] for c in configs if c.startswith(("sel:", "dyn:"))})
    if bank is None and units:
        bank = SelectorBank(Path(a.selectors_dir), Path(a.pca), units)
    check_tau(configs, bank)
    opener = opener or (lambda fp: Image.open(fp).convert("RGB")); t0 = time.time(); n_gen = 0
    for c0 in range(0, len(todo), a.chunk):
        chunk = todo[c0:c0 + a.chunk]; feats = encode_unique(host, chunk, opener)
        qvecs = [host.question_vector(it["question"]) for it in chunk]
        st = bank.prepare(feats, question_embeddings(host, [it["question"] for it in chunk])) if units else None
        for cfg in configs:
            need = [j for j, it in enumerate(chunk) if it["qid"] not in done[cfg]]
            if not need:
                continue
            kept = kept_for(cfg, chunk, feats, host, bank, st, qvecs)                    # whole chunk (cached per unit), then the needed rows
            seqs = [sequence(host, chunk[j], feats[j], kept[j], cfg) for j in need]; texts = []
            for b0 in range(0, len(seqs), a.batch):
                texts += host.generate(seqs[b0:b0 + a.batch], max_new_tokens=MAX_NEW[a.bench])
            with open(out / f"{cfg_file(cfg)}@{shard_tag(a.shard)}.jsonl", "a") as fh:      # this shard's own file (single writer)
                for j, tx in zip(need, texts):
                    fh.write(V.jdump({"qid": chunk[j]["qid"], "image_id": chunk[j]["image_id"], "pred": tx,
                                      "kept": N_VIS if kept[j] is None else int(len(kept[j]))}) + "\n")
                    done[cfg].add(chunk[j]["qid"])
            n_gen += len(need)
        print(f"[r7_d1_30] {a.bench} shard {a.shard}: {min(c0 + a.chunk, len(todo))}/{len(todo)} items, {n_gen} answers, {time.time() - t0:.0f}s", flush=True)
    mark_done(out, a, items, configs, seconds=time.time() - t0)
    return 0


def mark_done(out: Path, a, items, configs, seconds=None):
    """Completion marker for the driver's file gate: written only when every (configuration, item) of the shard has an answer."""
    for c in configs:
        have = answered(out, c)
        if any(it["qid"] not in have for it in items):
            raise SystemExit(f"incomplete after run: {c}")
    (out / f"done_shard{a.shard.replace('/', 'of')}.json").write_text(V.jdump({"items": len(items), "configs": len(configs), "seconds": seconds,
                                                                              "env": V.env_info()}, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--sets-dir", required=True); ap.add_argument("--bench", required=True)
    ap.add_argument("--configs-file", required=True); ap.add_argument("--shard", default="0/1"); ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=32); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--selectors-dir", default=str(D1 / "selectors")); ap.add_argument("--pca", default=str(D1 / "cache" / "pca.pt"))
    ap.add_argument("--limit", type=int, default=0)
    return run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
