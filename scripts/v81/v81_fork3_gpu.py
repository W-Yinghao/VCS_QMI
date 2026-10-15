#!/usr/bin/env python
"""V8.1 fork 3 — GPU diagnostics on the 24 frozen V8 formal units (FULL_PLAN_CN §04 D05 checkpoint level, D06; §03 §3–5; §07 §3).

Read-only on every V8 artefact: best.pt / resume.pt are loaded into one isolated model object, nothing is written back, no training step is
taken, no GenAI / TIFA label or file is touched.  All outputs go under OUT (outputs_v81/fork3).  The V8 code is the read-only copy in
scripts/v81/v8_src (PROVENANCE.md); the forward path reproduces v8_train.forward_logits / predict exactly, except that the frozen EVA ViT-g
output is computed once per image and cached (the ViT has no trainable parameter and is in eval mode in every unit, so its output is the
same for all 24 units, both checkpoints and the reconstructed initialisation); the agreement with the frozen CAL_SELECT export of each
unit is measured and reported (max |Δμ|).

Stages (resumable through done markers; `all` runs them in order):
  timing         standard-batch timing on the frozen eval load: batch 32, 364-res full-frame, FIT-AUDIT prompts; full model, ViT only,
                 Q-former+head only (cached ViT output); peak memory; per-image CPU decode + preprocess cost.
  infer          best.pt + resume.pt of the 24 units and RECONSTRUCTED_INIT per seed, eval mode, on FIT-AUDIT and CAL_SELECT -> rows in the
                 package schema + metrics; checkpoint sha256 / epoch; best == resume (same epoch?) and state equality when so.
  interventions  best.pt (24) + RECONSTRUCTED_INIT (3) on FIT-AUDIT: normal pairs; images permuted across prompts by a fixed seeded
                 derangement; within-prompt candidate rotation (derived from the normal predictions); an all-grey image per prompt; the
                 eval-mode gradient of μ w.r.t. the input pixels and w.r.t. the ViT output on the first 64 FIT-AUDIT rows (normal / permuted).
  gradcheck      U-RANK / U-LIST / U-CS x seeds 0-2 x {RECONSTRUCTED_INIT, best}: point-term vs ranking-term gradients on the first 20 batches
                 of the frozen epoch-0 batch plan (hash checked against the unit log), at the head and at the trainable Q-former parameters.

    python -I scripts/v81/v81_fork3_gpu.py <stage|all> [--out OUT] [--ids reports/V8_1_DIAGNOSTICS/results/fit_audit_ids.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "v8_src" / "v8")); sys.path.insert(0, str(HERE / "v8_src" / "v7"))
import v8_core as C  # noqa: E402
import v8_labels as LB  # noqa: E402
import v8_train as T  # noqa: E402
import v7_common as VC  # noqa: E402

OUT_DEFAULT = Path("/projects/EEG-foundation-model/yinghao/outputs_v81/fork3")
UNITS = Path("/projects/EEG-foundation-model/yinghao/outputs_v8/units")
IDS_DEFAULT = HERE.parents[1] / "reports" / "V8_1_DIAGNOSTICS" / "results" / "fit_audit_ids.json"
ARMS = ["U-CS", "U-BCE", "U-RPS", "U-CE", "U-MSE", "U-L1", "U-RANK", "U-LIST"]
SEEDS = [0, 1, 2]
DATASET = "EvalMuse-public-train"
BATCH, PERM_SEED, N_GRAD_IMAGES, N_GRAD_BATCHES, GRAD_ARMS = 32, 2718, 64, 20, ["U-RANK", "U-LIST", "U-CS"]


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(msg):
    print(f"[fork3 {now()}] {msg}", flush=True)


def sha256_file(p: Path, chunk=1 << 24) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_suffix(p.suffix + ".tmp"); tmp.write_text(json.dumps(obj, indent=1)); os.replace(tmp, p)


def write_jsonl(p: Path, rows) -> str:
    p.parent.mkdir(parents=True, exist_ok=True); data = "".join(json.dumps(r, allow_nan=False) + "\n" for r in rows).encode()
    tmp = p.with_suffix(p.suffix + ".tmp"); tmp.write_bytes(data); os.replace(tmp, p); return sha256_bytes(data)


def spearman(a, b):
    v = VC.avg_rank_spearman(a, b); return None if (isinstance(v, float) and math.isnan(v)) else v


# ------------------------------------------------------------------------------------------------------------------------------- pure helpers
def derangement_across_prompts(prompt_of: list, seed: int = PERM_SEED) -> list:
    """A fixed permutation pi of row indices with prompt_of[pi[i]] != prompt_of[i] for every i (images moved to a different prompt)."""
    rng = np.random.default_rng(seed); n = len(prompt_of); perm = rng.permutation(n)
    for _ in range(100 * n + 10):
        bad = [i for i in range(n) if prompt_of[perm[i]] == prompt_of[i]]
        if not bad:
            return [int(x) for x in perm]
        for i in bad:
            j = int(rng.integers(n))
            if prompt_of[perm[j]] != prompt_of[i] and prompt_of[perm[i]] != prompt_of[j]:
                perm[i], perm[j] = perm[j], perm[i]
    raise RuntimeError("no derangement found (a prompt holds too many of the rows)")


def rotation_within_prompt(prompt_of: list, order_key: list) -> list:
    """For each row the index of the next candidate of the same prompt (lexical candidate order, cyclic); rows of singleton prompts map to themselves."""
    groups = defaultdict(list)
    for i, p in enumerate(prompt_of):
        groups[p].append(i)
    nxt = [None] * len(prompt_of)
    for p, idx in groups.items():
        idx = sorted(idx, key=lambda i: str(order_key[i]))
        for k, i in enumerate(idx):
            nxt[i] = idx[(k + 1) % len(idx)]
    return nxt


def candidate_indices(prompt_of: list, image_of: list) -> list:
    groups = defaultdict(list)
    for i, p in enumerate(prompt_of):
        groups[p].append(i)
    ci = [None] * len(prompt_of)
    for p, idx in groups.items():
        for k, i in enumerate(sorted(idx, key=lambda i: str(image_of[i]))):
            ci[i] = k
    return ci


def tails_from_q(q):
    q = np.asarray(q, float); return np.flip(np.cumsum(np.flip(q)))[1:]


def schema_row(rec: dict, pred: dict, role: str, method: str, seed: int, checkpoint_id: str, cand: int) -> dict:
    q = np.asarray(pred["q"], float); q = (q / q.sum()).tolist(); u = tails_from_q(q)
    return {"dataset": DATASET, "role": role, "method": method, "seed": int(seed), "checkpoint_id": checkpoint_id, "prompt_id": str(rec["prompt_id"]),
            "pool_id": str(rec["prompt_id"]), "image_id": str(rec["image_id"]), "candidate_index": int(cand), "cluster_id": str(rec["prompt_id"]),
            "score": float(1 + u.sum()), "score_kind": "rating_mean", "q": q, "u": [float(x) for x in u], "votes": [int(v) for v in rec["votes"]],
            "human_mean": float(rec["mean"]), "mu_model_float32": float(pred["mu"])}


def label_metrics(pred: list, data, prior) -> dict:
    """V8's CAL metrics (prompt-macro mean-rating MSE, RPS, CE, within-prompt pairwise accuracy) + pooled / prompt-macro Spearman with the human mean."""
    m = T.cal_metrics(pred, data, prior); by = {r["image_id"]: r for r in data.rows}
    mu = np.array([p["mu"] for p in pred]); hm = np.array([by[p["image_id"]]["mean"] for p in pred]); per = defaultdict(list)
    for p in pred:
        per[by[p["image_id"]]["prompt_id"]].append((p["mu"], by[p["image_id"]]["mean"]))
    macro = [spearman([a for a, _ in v], [b for _, b in v]) for v in per.values() if len(v) >= 3]
    macro = [x for x in macro if x is not None]
    m.update({"spearman_pooled": spearman(mu, hm), "spearman_prompt_macro": float(np.mean(macro)) if macro else None, "n_prompts_spearman": len(macro),
              "mu_mean": float(mu.mean()), "mu_std": float(mu.std()), "human_mean_mean": float(hm.mean()),
              "within_prompt_mu_std_mean": float(np.mean([np.std([a for a, _ in v]) for v in per.values() if len(v) >= 2])),
              "within_prompt_human_std_mean": float(np.mean([np.std([b for _, b in v]) for v in per.values() if len(v) >= 2])),
              "between_prompt_mu_std": float(np.std([np.mean([a for a, _ in v]) for v in per.values()]))})
    return m


# ------------------------------------------------------------------------------------------------------------------------------------ context
class Ctx:
    """Device, tokenizer, processor, image source, the role data (FIT full / CAL_SELECT / FIT-AUDIT), one model object, cached ViT outputs."""

    def __init__(self, out: Path, ids_path: Path, workers: int = 6, smoke: bool = False):
        import torch
        from transformers import BertTokenizerFast
        self.torch = torch; self.out = out; out.mkdir(parents=True, exist_ok=True); self.workers = workers; self.smoke = smoke
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        VC.stage_workdir(VC.BIG_OUT / "work"); self.tok = BertTokenizerFast.from_pretrained(str(VC.BERT_DIR))
        rows = LB.load(T.DATA); self.prompts = {r["image_id"]: r["prompt"] for r in rows}; self.R = LB.records(rows)
        self.fit = T.RatingData(self.R["FIT"], self.tok, self.prompts); self.cal = T.RatingData(self.R["CAL_SELECT"][:(16 if smoke else None)], self.tok, self.prompts)
        ids = json.loads(Path(ids_path).read_text()); self.ids_meta = {k: ids[k] for k in ("n_prompts", "n_images", "image_ids_sha256", "manifest_sha256")}
        if smoke:                                                                          # code-path check only: two audit prompts, tiny roles, two units
            ids["image_ids"] = [i for p in ids["prompts"][:2] for i in p["image_ids"]]; self.ids_meta["smoke_subset"] = len(ids["image_ids"])
        keep = set(ids["image_ids"])
        order = {i: k for k, i in enumerate(ids["image_ids"])}
        recs = sorted([r for r in self.R["FIT"] if r["image_id"] in keep], key=lambda r: order[r["image_id"]])
        if len(recs) != len(keep):
            raise SystemExit(f"FIT-AUDIT ids not all found among valid FIT records: {len(recs)} / {len(keep)}")
        self.audit = T.RatingData(recs, self.tok, self.prompts)
        self.prior = LB.fit_priors(self.R["FIT"]).tolist()
        self.fit_hist = ((np.array([r["w"] for r in self.R["FIT"]])[:, None] * np.array([r["hist"] for r in self.R["FIT"]])).sum(0) / sum(r["w"] for r in self.R["FIT"])).tolist()
        VC.import_lavis()
        from lavis.processors.blip_processors import BlipImageEvalProcessor
        self.proc = BlipImageEvalProcessor(image_size=364); self.images = VC.ArchiveImages(); self.model = None; self.model_info = None
        self.init_states = {}; self.vis = {}; self.loaded = None
        self.arms = ["U-CS", "U-RANK"] if smoke else ARMS; self.seeds = [0] if smoke else SEEDS
        self.env = {"smoke": smoke, "host": platform.node(), "gpu": torch.cuda.get_device_name(0) if self.device == "cuda" else "cpu", "torch": torch.__version__,
                    "numpy": np.__version__, "python": platform.python_version(), "cuda": torch.version.cuda, "utc_start": now()}
        log(f"ctx ready: FIT {len(self.fit)} CAL_SELECT {len(self.cal)} FIT-AUDIT {len(self.audit)} ({self.audit.prompt_images.__len__()} prompts) on {self.env['gpu']}")

    def mem(self):
        return self.torch.cuda.max_memory_allocated() / 2 ** 30 if self.device == "cuda" else None

    def reset_mem(self):
        if self.device == "cuda":
            self.torch.cuda.reset_peak_memory_stats()

    # ---- model and checkpoints
    def build(self):
        if self.model is None:
            self.model, self.model_info = T.build(0, self.device, self.fit_hist); self.model.eval()
        return self.model

    def init_state(self, seed: int) -> dict:
        """RECONSTRUCTED_INIT for a seed: V8's build() (seeded head weight, FIT log-histogram bias, V7's BLIP-2 COCO construction) -> trainable state on CPU."""
        if seed not in self.init_states:
            m, info = T.build(seed, "cpu", self.fit_hist); self.init_states[seed] = {"state": T.trainable_state(m), "info": info}; del m
        return self.init_states[seed]["state"]

    def load_state(self, state: dict, tag: str):
        model = self.build(); msg = model.load_state_dict(state, strict=False)
        missing = [k for k in msg.missing_keys if not k.startswith("visual_encoder.")]
        if missing or msg.unexpected_keys:
            raise SystemExit(f"{tag}: state does not match the model: missing {missing[:5]} unexpected {list(msg.unexpected_keys)[:5]}")
        model.eval(); self.loaded = tag

    def load_ckpt(self, path: Path, tag: str) -> dict:
        ck = self.torch.load(path, map_location="cpu", weights_only=False); self.load_state(ck["model"], tag)
        return {"epoch": ck.get("epoch"), "best": ck.get("best"), "step": ck.get("step"), "keys": len(ck["model"])}

    def load_init(self, seed: int):
        self.load_state(self.init_state(seed), f"RECONSTRUCTED_INIT-s{seed}")

    # ---- batches
    def text_batch(self, rows):
        torch = self.torch; L = max(len(r["ids"]) for r in rows); B = len(rows)
        ids = torch.zeros((B, L), dtype=torch.long); att = torch.zeros((B, L), dtype=torch.long)
        for b, r in enumerate(rows):
            ids[b, :len(r["ids"])] = torch.tensor(r["ids"]); att[b, :len(r["ids"])] = 1
        return ids.to(self.device), att.to(self.device)

    def image_loader(self, data, idx_batches):
        from torch.utils.data import DataLoader
        col = T.Collate(data, self.images, self.proc)
        return DataLoader(idx_batches, batch_size=None, shuffle=False, collate_fn=col, num_workers=self.workers, persistent_workers=False,
                          prefetch_factor=4 if self.workers else None)

    def vit(self, image):
        torch = self.torch; model = self.build()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
            return model.visual_encoder(image.to(self.device, non_blocking=True))

    def vis_cache(self, name: str, data):
        """ViT output for every row of `data` (in row order), fp16 on the device, computed once per process (batch 32, the eval processor)."""
        if name in self.vis:
            return self.vis[name]
        torch = self.torch; n = len(data); batches = [list(range(s, min(s + BATCH, n))) for s in range(0, n, BATCH)]; buf = None; t0 = time.time()
        for b in self.image_loader(data, batches):
            v = self.vit(b["image"]); v = v.half() if self.device == "cuda" else v.float()
            if buf is None:
                buf = torch.empty((n, *v.shape[1:]), dtype=v.dtype, device=self.device)
            buf[self._batch_slice(b, data)] = v
        self.vis[name] = buf; log(f"vis cache {name}: {tuple(buf.shape)} fp16 in {time.time() - t0:.0f}s")
        return buf

    def _batch_slice(self, b, data):
        pos = {r["image_id"]: k for k, r in enumerate(data.rows)} if not hasattr(data, "_pos") else data._pos
        data._pos = pos; return [pos[i] for i in b["image_id"]]

    # ---- forward from cached ViT output
    def logits_from_vis(self, vis, ids, att):
        torch = self.torch; model = self.build()
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
            image_embeds = model.ln_vision(vis); image_atts = torch.ones(image_embeds.shape[:-1], dtype=torch.long, device=self.device)
            q = model.query_tokens.expand(image_embeds.shape[0], -1, -1); q_att = torch.ones(q.shape[:-1], dtype=torch.long, device=self.device)
            out = model.Qformer.bert(ids, query_embeds=q, attention_mask=torch.cat([q_att, att], 1), encoder_hidden_states=image_embeds,
                                     encoder_attention_mask=image_atts, return_dict=True)
            pooled = out.last_hidden_state[:, :q.shape[1]].float().mean(1)
            with torch.autocast("cuda", enabled=False):
                return model.v8_head(pooled)

    def predict_vis(self, data, vis, vis_index=None, rows_index=None, grey_vis=None) -> list:
        """q / u / μ per row (eval mode, no grad), text from the row, ViT output from vis[vis_index[i]] (identity by default) or the grey image."""
        torch = self.torch; self.build().eval(); rows_index = list(range(len(data))) if rows_index is None else rows_index; out = []
        pt = torch.tensor(self.prior, device=self.device)
        with torch.no_grad():
            for s in range(0, len(rows_index), BATCH):
                ri = rows_index[s:s + BATCH]; rows = [data.rows[i] for i in ri]; ids, att = self.text_batch(rows)
                v = grey_vis.expand(len(ri), -1, -1) if grey_vis is not None else vis[[(vis_index[i] if vis_index is not None else i) for i in ri]]
                logits = self.logits_from_vis(v, ids, att).float(); o = C.outputs(logits, pt)
                for j, r in enumerate(rows):
                    out.append({"image_id": r["image_id"], "prompt_id": r["prompt_id"], "logits": logits[j].cpu().tolist(), "q": o["q"][j].cpu().tolist(),
                                "u": o["u"][j].cpu().tolist(), "mu": float(o["mu"][j])})
        return out


# ------------------------------------------------------------------------------------------------------------------------------------- stages
def unit_dir(arm, seed) -> Path:
    return UNITS / f"V8-FIT-{arm}-s{seed}"


def stage_timing(ctx: Ctx):
    torch = ctx.torch; done = ctx.out / "timing" / "timing.json"
    if done.exists():
        log("timing: done marker present, skip"); return json.loads(done.read_text())
    ctx.load_ckpt(unit_dir("U-CS", 0) / "best.pt", "timing:U-CS-s0 best"); model = ctx.build(); model.eval()
    n_b = 23; batches = [list(range(s, min(s + BATCH, len(ctx.audit)))) for s in range(0, len(ctx.audit), BATCH)][:n_b]
    t0 = time.time(); cpu = []; imgs = []
    col = T.Collate(ctx.audit, ctx.images, ctx.proc)
    for bi in batches:                                                                  # single-process decode + preprocess (CPU cost per image)
        t1 = time.time(); b = col(bi); cpu.append((time.time() - t1) / len(bi)); imgs.append(b)
    lens = [len(r["ids"]) for r in ctx.audit.rows]
    def timed(fn, label):
        for k, b in enumerate(imgs[:3]):
            fn(k, b)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); ts = []
        for k, b in enumerate(imgs):
            if k < 3:
                continue
            torch.cuda.synchronize(); t = time.time(); fn(k, b); torch.cuda.synchronize(); ts.append(time.time() - t)
        per_b = float(np.mean(ts)); n_img = float(np.mean([len(b["image_id"]) for b in imgs[3:]]))
        return {"label": label, "n_timed_batches": len(ts), "batch": BATCH, "s_per_batch_mean": per_b, "s_per_batch_std": float(np.std(ts)), "s_per_image": per_b / n_img,
                "images_per_s": n_img / per_b, "peak_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
    pt = torch.tensor(ctx.prior, device=ctx.device)
    def full(k, b):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            logits = T.forward_logits(model, b, ctx.device)
        C.outputs(logits.float(), pt)
    def vit_only(k, b):
        ctx.vit(b["image"])
    res = {"env": ctx.env, "checkpoint": "V8-FIT-U-CS-s0/best.pt (any unit has the same architecture)", "resolution": 364, "batch": BATCH,
           "prompt_tokens": {"mean": float(np.mean(lens)), "max": int(max(lens)), "min": int(min(lens))}, "n_images_loaded": sum(len(b["image_id"]) for b in imgs),
           "cpu_decode_preprocess_s_per_image_single_process": float(np.mean(cpu)), "full_model": timed(full, "ViT + Q-former + head (v8_train.forward_logits)"),
           "vit_only": timed(vit_only, "EVA ViT-g only")}
    cache = {k: ctx.vit(b["image"]).half() for k, b in enumerate(imgs)}
    def qf_only(k, b):
        ids, att = b["ids"].to(ctx.device), b["att"].to(ctx.device)
        with torch.no_grad():
            C.outputs(ctx.logits_from_vis(cache[k], ids, att).float(), pt)
    res["qformer_head_only"] = timed(qf_only, "ln_vision + Q-former + head on the cached ViT output")
    f = res["full_model"]["s_per_image"]
    res["estimates_s"] = {"fit_audit": f * len(ctx.audit), "cal_select": f * len(ctx.cal), "candidate_pool_15": f * 15, "candidate_pool_7": f * 7,
                          "one_checkpoint_both_roles_full": f * (len(ctx.audit) + len(ctx.cal)),
                          "one_checkpoint_both_roles_cached": res["qformer_head_only"]["s_per_image"] * (len(ctx.audit) + len(ctx.cal))}
    res["seconds"] = time.time() - t0; write_json(done, res); del cache
    log(f"timing: full {f * 1000:.1f} ms/img, ViT {res['vit_only']['s_per_image'] * 1000:.1f} ms/img, Q-former+head {res['qformer_head_only']['s_per_image'] * 1000:.1f} ms/img, peak {res['full_model']['peak_mem_gb']:.1f} GB")
    return res


def export_mu(unit: Path, role: str) -> dict:
    p = unit / "pred" / f"images_{role}.jsonl"
    return {json.loads(l)["image_id"]: json.loads(l)["mu"] for l in open(p)} if p.exists() else {}


def run_roles(ctx: Ctx, method: str, seed: int, ckpt_id: str, tag: str) -> dict:
    """Both roles for the loaded state: rows (schema) + metrics; returns the meta part."""
    res = {}
    for role, data, vis_name in (("FIT_AUDIT", ctx.audit, "audit"), ("CAL_SELECT", ctx.cal, "cal")):
        t0 = time.time(); vis = ctx.vis_cache(vis_name, data); pred = ctx.predict_vis(data, vis)
        cand = candidate_indices([r["prompt_id"] for r in data.rows], [r["image_id"] for r in data.rows])
        rows = [schema_row(r, p, role, method, seed, ckpt_id, cand[i]) for i, (r, p) in enumerate(zip(data.rows, pred))]
        path = ctx.out / "infer" / "rows" / f"{tag}__{role}.jsonl"; h = write_jsonl(path, rows)
        res[role] = {"n": len(rows), "rows": str(path), "rows_sha256": h, "metrics": label_metrics(pred, data, ctx.prior), "seconds": time.time() - t0}
    return res


def stage_infer(ctx: Ctx):
    torch = ctx.torch; done = ctx.out / "infer" / "done.json"
    if done.exists():
        log("infer: done marker present, skip"); return
    summary = ctx.out / "infer" / "summary.json"; S = json.loads(summary.read_text()) if summary.exists() else {"env": ctx.env, "fit_audit": ctx.ids_meta, "entries": {}}
    for seed in ctx.seeds:                                                               # RECONSTRUCTED_INIT once per seed
        tag = f"RECONSTRUCTED_INIT-s{seed}"
        if tag in S["entries"]:
            continue
        t0 = time.time(); ctx.load_init(seed); info = ctx.init_states[seed]["info"]
        S["entries"][tag] = {"method": "ALL_ARMS", "seed": seed, "checkpoint_id": "RECONSTRUCTED_INIT", "kind": "init",
                             "note": "V8 build(seed): V7 BLIP-2 COCO construction + seeded 5-logit head (weight N(0, 0.02) generator seed*7919+5, bias = log FIT histogram); same for the eight arms at a seed; not claimed bit-identical to the formal units' start",
                             "init_missing_keys": info.get("missing_keys"), "init_unexpected_keys": info.get("unexpected_keys"), "n_trainable_v8": info.get("n_trainable_v8"),
                             "roles": run_roles(ctx, "ALL_ARMS", seed, "RECONSTRUCTED_INIT", tag), "seconds": time.time() - t0}
        write_json(summary, S); log(f"infer {tag}: FIT-AUDIT Spearman {S['entries'][tag]['roles']['FIT_AUDIT']['metrics']['spearman_pooled']}")
    for arm in ctx.arms:
        for seed in ctx.seeds:
            u = unit_dir(arm, seed); name = u.name
            if all(f"{name}__{k}" in S["entries"] for k in ("best", "resume")):
                continue
            t0 = time.time(); L = json.loads((u / "log.json").read_text()); hashes = {k: sha256_file(u / f"{k}.pt") for k in ("best", "resume")}
            sizes = {k: (u / f"{k}.pt").stat().st_size for k in ("best", "resume")}; info = {}; states = {}
            for k in ("best", "resume"):
                tag = f"{name}__{k}"; ck = ctx.load_ckpt(u / f"{k}.pt", tag); info[k] = ck
                states[k] = {n: v.detach().cpu().clone() for n, v in ctx.build().state_dict().items() if not n.startswith("visual_encoder.")} if k == "best" else None
                cid = f"{k}_e{ck['epoch']}"; roles = run_roles(ctx, arm, seed, cid, tag)
                exp = export_mu(u, "CAL_SELECT"); mine = {}
                for l in open(roles["CAL_SELECT"]["rows"]):
                    r = json.loads(l); mine[r["image_id"]] = r["mu_model_float32"]
                d = [abs(mine[i] - exp[i]) for i in exp if i in mine]
                roles["CAL_SELECT"]["vs_frozen_export"] = ({"n": len(d), "max_abs_dmu": max(d), "mean_abs_dmu": float(np.mean(d)), "export_gpu": L.get("gpu"),
                                                            "applies": k == "best" and ck["epoch"] == L.get("selected_epoch")} if d else None)
                S["entries"][tag] = {"method": arm, "seed": seed, "unit": str(u), "checkpoint": f"{k}.pt", "checkpoint_id": cid, "kind": k, "epoch": ck["epoch"], "ckpt_best_field": ck.get("best"),
                                     "step": ck.get("step"), "sha256": hashes[k], "bytes": sizes[k], "log_selected_epoch": L.get("selected_epoch"), "log_epochs": L.get("epochs"),
                                     "roles": roles, "seconds": time.time() - t0}
                log(f"infer {tag}: epoch {ck['epoch']} CAL_SELECT MSE {roles['CAL_SELECT']['metrics']['mean_mse_prompt_macro']:.4f} vs export max|dmu| {roles['CAL_SELECT']['vs_frozen_export'] and roles['CAL_SELECT']['vs_frozen_export']['max_abs_dmu']}")
            same = info["best"]["epoch"] == info["resume"]["epoch"]; eq = None
            if same:                                                                       # resume is loaded now; compare with the saved best state
                cur = ctx.build().state_dict(); eq = max(float((cur[n].detach().cpu().float() - v.float()).abs().max()) for n, v in states["best"].items())
            for k in ("best", "resume"):
                S["entries"][f"{name}__{k}"].update({"best_epoch": info["best"]["epoch"], "resume_epoch": info["resume"]["epoch"], "best_equals_resume_epoch": same,
                                                     "best_resume_max_abs_param_diff": eq, "resume_best_field": info["resume"].get("best")})
            write_json(summary, S)
    S["utc_end"] = now(); write_json(summary, S); write_json(done, {"entries": len(S["entries"]), "utc": now()}); log(f"infer done: {len(S['entries'])} entries")


def grad_norms_input(ctx: Ctx, rows_index: list, vis_index: list, label: str) -> dict:
    """Eval-mode ∂μ/∂x (pixels, full ViT backward) and ∂μ/∂(ViT output) norms per image on the given rows (text from the row, image from vis_index)."""
    torch = ctx.torch; model = ctx.build(); model.eval(); pt = torch.tensor(ctx.prior, device=ctx.device); data = ctx.audit; vis = ctx.vis["audit"]
    col = T.Collate(data, ctx.images, ctx.proc); gx, gv, cls_share, max_share, mus = [], [], [], [], []
    for s in range(0, len(rows_index), 4):
        ri = rows_index[s:s + 4]; rows = [data.rows[i] for i in ri]; ids, att = ctx.text_batch(rows)
        img_items = [vis_index[i] for i in ri]; b = col(img_items); image = b["image"].to(ctx.device).requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=ctx.device == "cuda"):
            v = model.visual_encoder(image)                                                    # no no_grad: the frozen ViT passes gradients to its input
            logits = ctx.logits_from_vis(v, ids, att)
        mu = C.outputs(logits.float(), pt)["mu"]; mu.sum().backward()
        gx += image.grad.flatten(1).float().norm(dim=1).cpu().tolist(); mus += mu.detach().cpu().tolist(); image.grad = None; model.zero_grad(set_to_none=True)
        vv = vis[img_items].clone().float().requires_grad_(True)
        logits2 = ctx.logits_from_vis(vv, ids, att); mu2 = C.outputs(logits2.float(), pt)["mu"]; (g,) = torch.autograd.grad(mu2.sum(), vv)
        tok = g.float().norm(dim=2); tot = tok.norm(dim=1)
        gv += tot.cpu().tolist(); cls_share += (tok[:, 0] / tot.clamp_min(1e-12)).cpu().tolist(); max_share += (tok[:, 1:].max(1).values / tot.clamp_min(1e-12)).cpu().tolist()
    arr = lambda x: {"mean": float(np.mean(x)), "median": float(np.median(x)), "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90))}
    return {"label": label, "n": len(gx), "grad_pixels_l2": arr(gx), "grad_vit_output_l2": arr(gv), "vit_output_cls_token_share": arr(cls_share),
            "vit_output_max_patch_share": arr(max_share), "mu": arr(mus), "per_image": {"grad_pixels_l2": gx, "grad_vit_output_l2": gv}}


def stage_interventions(ctx: Ctx):
    torch = ctx.torch; done = ctx.out / "interventions" / "done.json"
    if done.exists():
        log("interventions: done marker present, skip"); return
    data = ctx.audit; vis = ctx.vis_cache("audit", data); prompt_of = [r["prompt_id"] for r in data.rows]; image_of = [r["image_id"] for r in data.rows]
    perm = derangement_across_prompts(prompt_of); nxt = rotation_within_prompt(prompt_of, image_of); hm = np.array([r["mean"] for r in data.rows])
    from PIL import Image
    grey_img = ctx.proc(Image.new("RGB", (364, 364), (128, 128, 128))).unsqueeze(0); grey_vis = ctx.vit(grey_img).to(vis.dtype)[0]
    prompts_first = {}                                                                     # one row per prompt for the grey condition
    for i, p in enumerate(prompt_of):
        prompts_first.setdefault(p, i)
    prompt_rows = list(prompts_first.values()); grad_rows = list(range(min(4 if ctx.smoke else N_GRAD_IMAGES, len(data)))); perm_grad = [perm[i] for i in grad_rows]
    summary = ctx.out / "interventions" / "summary.json"; S = json.loads(summary.read_text()) if summary.exists() else {
        "env": ctx.env, "fit_audit": ctx.ids_meta, "permutation_seed": PERM_SEED, "permutation": perm, "rotation_next": nxt, "grey_rgb": [128, 128, 128],
        "grad_rows": grad_rows, "grad_rows_permuted_partner": perm_grad, "entries": {}}
    write_json(ctx.out / "interventions" / "permutation.json", {"seed": PERM_SEED, "image_ids": image_of, "prompt_ids": prompt_of, "perm": perm, "rotation_next": nxt})
    models = [("RECONSTRUCTED_INIT", seed, None) for seed in ctx.seeds] + [(arm, seed, unit_dir(arm, seed) / "best.pt") for arm in ctx.arms for seed in ctx.seeds]
    for method, seed, path in models:
        tag = f"{method}-s{seed}" if path is None else f"{path.parent.name}__best"
        if tag in S["entries"]:
            continue
        t0 = time.time(); ctx.reset_mem()
        if path is None:
            ctx.load_init(seed); ck = {"epoch": None}
        else:
            ck = ctx.load_ckpt(path, tag)
        normal = ctx.predict_vis(data, vis); permuted = ctx.predict_vis(data, vis, vis_index=perm); grey = ctx.predict_vis(data, vis, rows_index=prompt_rows, grey_vis=grey_vis)
        mu_n = np.array([p["mu"] for p in normal]); mu_p = np.array([p["mu"] for p in permuted]); q_n = np.array([p["q"] for p in normal]); q_p = np.array([p["q"] for p in permuted])
        u_n = np.array([p["u"] for p in normal]); u_p = np.array([p["u"] for p in permuted]); d = np.abs(mu_p - mu_n)
        mu_g = np.array([p["mu"] for p in grey]); pm = defaultdict(list)
        for i, p in enumerate(prompt_of):
            pm[p].append(mu_n[i])
        gstat = []
        for g, i0 in zip(grey, prompt_rows):
            v = np.array(pm[prompt_of[i0]]); gstat.append((g["mu"] - v.mean(), g["mu"] >= v.max(), g["mu"] <= v.min(), hm[[k for k, p in enumerate(prompt_of) if p == prompt_of[i0]]].mean()))
        dmu_swap = np.array([mu_n[nxt[i]] - mu_n[i] for i in range(len(data))]); dh_swap = np.array([hm[nxt[i]] - hm[i] for i in range(len(data))]); ok = dh_swap != 0
        nm = label_metrics(normal, data, ctx.prior)
        ent = {"method": method, "seed": seed, "checkpoint": None if path is None else str(path), "epoch": ck.get("epoch"), "n_rows": len(data),
               "normal": nm,
               "permuted_across_prompts": {"abs_dmu_mean": float(d.mean()), "abs_dmu_median": float(np.median(d)), "abs_dmu_p90": float(np.percentile(d, 90)),
                                           "frac_abs_dmu_lt_0.05": float((d < 0.05).mean()), "frac_abs_dmu_lt_0.2": float((d < 0.2).mean()),
                                           "q_l1_mean": float(np.abs(q_p - q_n).sum(1).mean()), "u_abs_mean": float(np.abs(u_p - u_n).mean()),
                                           "spearman_permuted_vs_normal_same_row(text-following)": spearman(mu_p, mu_n),
                                           "spearman_permuted_vs_normal_of_placed_image(image-following)": spearman(mu_p, mu_n[perm]),
                                           "mu_mean": float(mu_p.mean()), "mu_std": float(mu_p.std()), "within_prompt_mu_std_mean": float(np.mean([np.std(mu_p[[k for k, p in enumerate(prompt_of) if p == pp]]) for pp in pm]))},
               "within_prompt_rotation": {"n_pairs": int(len(dmu_swap)), "n_pairs_human_differs": int(ok.sum()), "abs_dmu_mean": float(np.abs(dmu_swap).mean()),
                                          "spearman_dmu_vs_dhuman": spearman(dmu_swap[ok], dh_swap[ok]) if ok.sum() >= 3 else None,
                                          "sign_agreement_rate": float((np.sign(dmu_swap[ok]) == np.sign(dh_swap[ok])).mean()) if ok.any() else None,
                                          "frac_model_tie_when_human_differs": float((dmu_swap[ok] == 0).mean()) if ok.any() else None},
               "grey_image": {"n_prompts": len(grey), "mu_mean": float(mu_g.mean()), "mu_std": float(mu_g.std()), "mu_min": float(mu_g.min()), "mu_max": float(mu_g.max()),
                              "dmu_vs_prompt_mean_normal_mean": float(np.mean([g[0] for g in gstat])), "frac_grey_ge_prompt_max": float(np.mean([g[1] for g in gstat])),
                              "frac_grey_le_prompt_min": float(np.mean([g[2] for g in gstat])), "spearman_grey_mu_vs_prompt_human_mean": spearman(mu_g, [g[3] for g in gstat])}}
        ent["gradients"] = {"normal": grad_norms_input(ctx, grad_rows, grad_rows, "normal pairs"), "permuted": grad_norms_input(ctx, grad_rows, perm_grad, "images from another prompt")}
        rows_path = ctx.out / "interventions" / "rows" / f"{tag}.jsonl"
        ent["rows"] = str(rows_path); ent["rows_sha256"] = write_jsonl(rows_path, [
            {"row": i, "image_id": image_of[i], "prompt_id": prompt_of[i], "mu_normal": float(mu_n[i]), "mu_permuted": float(mu_p[i]), "permuted_image_id": image_of[perm[i]],
             "q_normal": normal[i]["q"], "q_permuted": permuted[i]["q"], "human_mean": float(hm[i])} for i in range(len(data))] + [
            {"grey_prompt_id": prompt_of[i0], "mu_grey": g["mu"], "q_grey": g["q"]} for g, i0 in zip(grey, prompt_rows)])
        ent["peak_mem_gb"] = ctx.mem(); ent["seconds"] = time.time() - t0; S["entries"][tag] = ent; write_json(summary, S)
        log(f"interventions {tag}: |dmu| perm mean {d.mean():.3f}, text-follow rho {ent['permuted_across_prompts']['spearman_permuted_vs_normal_same_row(text-following)']}, "
            f"image-follow rho {ent['permuted_across_prompts']['spearman_permuted_vs_normal_of_placed_image(image-following)']}, grad px {ent['gradients']['normal']['grad_pixels_l2']['mean']:.3g} ({time.time() - t0:.0f}s)")
    S["utc_end"] = now(); write_json(summary, S); write_json(done, {"entries": len(S["entries"]), "utc": now()}); log("interventions done")


def param_groups(model):
    g = {"head": [], "qformer": [], "ln_vision": [], "query_tokens": []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        key = "head" if n.startswith("v8_head.") else "qformer" if n.startswith("Qformer.") else "ln_vision" if n.startswith("ln_vision.") else "query_tokens" if n == "query_tokens" else None
        if key is None:
            raise SystemExit(f"unexpected trainable parameter {n}")
        g[key].append((n, p))
    return g


def grad_stats(groups: dict, g_point: list, g_rank: list | None) -> dict:
    """Per group: L2 norms of the point and ranking gradients, their cosine, non-finite flags; plus the totals over all trainable parameters.
    g_point / g_rank are positional (the parameters of the groups in iteration order, as returned by torch.autograd.grad)."""
    torch = __import__("torch"); out = {}; pos = {}; k = 0
    for gname, grp in groups.items():
        for n, p in grp:
            pos[(gname, n)] = k; k += 1
    def vec(gs, gname, grp):
        parts = [gs[pos[(gname, n)]].flatten().float() if gs[pos[(gname, n)]] is not None else torch.zeros(p.numel(), device=p.device) for n, p in grp]
        return torch.cat(parts) if parts else torch.zeros(0)
    vecs_p = {g: vec(g_point, g, grp) for g, grp in groups.items()}; vecs_r = {g: vec(g_rank, g, grp) for g, grp in groups.items()} if g_rank is not None else None
    vecs_p["all_trainable"] = torch.cat(list(vecs_p.values())) if vecs_p else torch.zeros(0)
    if vecs_r is not None:
        vecs_r["all_trainable"] = torch.cat([vecs_r[g] for g in groups])
    n_params = {g: int(sum(p.numel() for _, p in grp)) for g, grp in groups.items()}; n_params["all_trainable"] = sum(n_params.values())
    for name, vp in vecs_p.items():
        rec = {"n_params": n_params[name], "point_norm": float(vp.norm()), "point_nonfinite": bool((~torch.isfinite(vp)).any()) if vp.numel() else False}
        if vecs_r is not None:
            vr = vecs_r[name]; rec.update({"rank_norm": float(vr.norm()), "rank_nonfinite": bool((~torch.isfinite(vr)).any()) if vr.numel() else False,
                                          "cos_point_rank": float((vp @ vr) / (vp.norm() * vr.norm()).clamp_min(1e-30)) if vp.numel() else None,
                                          "rank_over_point": float(vr.norm() / vp.norm().clamp_min(1e-30)) if vp.numel() else None})
        out[name] = rec
    return out


def stage_gradcheck(ctx: Ctx):
    torch = ctx.torch; done = ctx.out / "gradcheck" / "done.json"
    if done.exists():
        log("gradcheck: done marker present, skip"); return
    summary = ctx.out / "gradcheck" / "summary.json"; S = json.loads(summary.read_text()) if summary.exists() else {"env": ctx.env, "n_batches": N_GRAD_BATCHES, "mode": "eval (dropout off), autocast fp16 forward as in training, fp32 losses from v8_core.batch_loss, unscaled autograd (no GradScaler), no optimiser step", "entries": {}}
    pt = torch.tensor(ctx.prior, device=ctx.device); col = T.Collate(ctx.fit, ctx.images, ctx.proc); plans = {}
    for seed in ctx.seeds:                                                                   # the frozen epoch-0 batch plan, hash-checked against a unit log
        plan = C.batch_manifest(ctx.fit.prompt_images, seed, 0); h = sha256_bytes(json.dumps(plan).encode())
        L = json.loads((unit_dir("U-CS", seed) / "log.json").read_text()); ok = L["batch_plan_sha256"][0] == h
        batches = []
        for items in plan[:(2 if ctx.smoke else N_GRAD_BATCHES)]:
            b = col(items); batches.append({"vis": ctx.vit(b["image"]).half(), "ids": b["ids"], "att": b["att"], "hist": b["hist"], "w": b["w"], "subgroup": b["subgroup"], "n_sub": b["n_sub"], "n": len(items)})
        plans[seed] = (batches, {"plan_sha256": h, "matches_unit_log": ok, "n_batches_plan": len(plan)})
        log(f"gradcheck seed {seed}: epoch-0 plan hash matches log {ok}; {len(batches)} batches cached")
    for arm in [a for a in GRAD_ARMS if a in ctx.arms]:
        kind = C.ARMS[arm]; ranking = kind in ("rank", "list")
        for seed in ctx.seeds:
            for ck_kind in ("RECONSTRUCTED_INIT", "best"):
                tag = f"{arm}-s{seed}__{ck_kind}"
                if tag in S["entries"]:
                    continue
                t0 = time.time()
                if ck_kind == "best":
                    ck = ctx.load_ckpt(unit_dir(arm, seed) / "best.pt", tag)
                else:
                    ctx.load_init(seed); ck = {"epoch": None}
                model = ctx.build(); model.eval(); groups = param_groups(model); params = [p for grp in groups.values() for _, p in grp]; per_batch = []
                for bi, b in enumerate(plans[seed][0]):
                    logits = ctx.logits_from_vis(b["vis"], b["ids"].to(ctx.device), b["att"].to(ctx.device)).float()
                    hist, w = b["hist"].to(ctx.device), b["w"].to(ctx.device)
                    lp, parts_p, o = C.batch_loss(logits, hist, w, pt, "mse" if ranking else kind, b["subgroup"], b["n_sub"])
                    g_point = list(torch.autograd.grad(lp, params, retain_graph=True, allow_unused=True)); g_rank = None; parts_t = {}
                    if ranking:
                        lt, parts_t, _ = C.batch_loss(logits, hist, w, pt, kind, b["subgroup"], b["n_sub"]); rk = lt - lp
                        g_rank = list(torch.autograd.grad(rk, params, retain_graph=False, allow_unused=True))
                    mu = o["mu"].detach(); tt = o["T"].detach()
                    rec = {"batch": bi, "n_images": b["n"], "point_loss": float(lp), "ranking_loss": float(parts_t.get("ranking", float("nan"))) if ranking else None,
                           "ranking_subgroups": parts_t.get("ranking_subgroups"), "ranking_pairs": parts_t.get("ranking_pairs"), "subgroups_in_batch": len(set(b["subgroup"])),
                           "mu_mean": float(mu.mean()), "mu_std": float(mu.std()), "mu_min": float(mu.min()), "mu_max": float(mu.max()),
                           "frac_mu_saturated": float(((mu < 1.05) | (mu > 4.95)).float().mean()), "abs_T_mean": float(tt.abs().mean()),
                           "groups": grad_stats(groups, g_point, g_rank)}
                    per_batch.append(rec); del g_point, g_rank, logits, o
                def agg(key, sub):
                    v = [r["groups"][key].get(sub) for r in per_batch if r["groups"][key].get(sub) is not None]
                    return {"mean": float(np.mean(v)), "median": float(np.median(v)), "min": float(np.min(v)), "max": float(np.max(v))} if v else None
                ent = {"arm": arm, "loss": kind, "seed": seed, "checkpoint": ck_kind, "epoch": ck.get("epoch"), "plan": plans[seed][1], "n_batches": len(per_batch),
                       "summary": {g: {s: agg(g, s) for s in ("point_norm", "rank_norm", "cos_point_rank", "rank_over_point")} for g in ("head", "qformer", "ln_vision", "query_tokens", "all_trainable")},
                       "nonfinite_batches": int(sum(any(r["groups"][g].get("point_nonfinite") or r["groups"][g].get("rank_nonfinite") for g in r["groups"]) for r in per_batch)),
                       "mu_saturated_frac_mean": float(np.mean([r["frac_mu_saturated"] for r in per_batch])), "mu_mean": float(np.mean([r["mu_mean"] for r in per_batch])),
                       "mu_within_batch_std_mean": float(np.mean([r["mu_std"] for r in per_batch])), "abs_T_mean": float(np.mean([r["abs_T_mean"] for r in per_batch])),
                       "point_loss_mean": float(np.mean([r["point_loss"] for r in per_batch])),
                       "ranking_loss_mean": float(np.mean([r["ranking_loss"] for r in per_batch])) if ranking else None,
                       "ranking_pairs_mean": float(np.mean([r["ranking_pairs"] for r in per_batch])) if ranking else None,
                       "ranking_subgroups_mean": float(np.mean([r["ranking_subgroups"] for r in per_batch])) if ranking else None,
                       "per_batch": per_batch, "seconds": time.time() - t0}
                S["entries"][tag] = ent; write_json(summary, S)
                hs = ent["summary"]["head"]; qs = ent["summary"]["qformer"]
                log(f"gradcheck {tag}: head point {hs['point_norm']['mean']:.3g} rank {hs['rank_norm'] and hs['rank_norm']['mean']} cos {hs['cos_point_rank'] and hs['cos_point_rank']['mean']}; "
                    f"qformer point {qs['point_norm']['mean']:.3g} rank {qs['rank_norm'] and qs['rank_norm']['mean']} cos {qs['cos_point_rank'] and qs['cos_point_rank']['mean']}; sat {ent['mu_saturated_frac_mean']:.3f} ({time.time() - t0:.0f}s)")
    S["utc_end"] = now(); write_json(summary, S); write_json(done, {"entries": len(S["entries"]), "utc": now()}); log("gradcheck done")


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["timing", "infer", "interventions", "gradcheck", "all"])
    ap.add_argument("--out", default=None); ap.add_argument("--ids", default=str(IDS_DEFAULT)); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smoke", action="store_true", help="code-path check on a tiny subset (2 audit prompts, 16 CAL images, U-CS / U-RANK seed 0); separate output dir")
    a = ap.parse_args(argv); out = Path(a.out) if a.out else (OUT_DEFAULT.with_name("fork3_smoke") if a.smoke else OUT_DEFAULT); ctx = Ctx(out, Path(a.ids), a.workers, a.smoke)
    stages = ["timing", "infer", "interventions", "gradcheck"] if a.stage == "all" else [a.stage]
    for s in stages:
        t0 = time.time(); {"timing": stage_timing, "infer": stage_infer, "interventions": stage_interventions, "gradcheck": stage_gradcheck}[s](ctx)
        log(f"stage {s} finished in {time.time() - t0:.0f}s; peak mem {ctx.mem()} GB")
    write_json(out / "env.json", {**ctx.env, "utc_end": now(), "stages": stages}); return 0


if __name__ == "__main__":
    raise SystemExit(main())
