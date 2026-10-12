"""r7 R7-30 D1-10 — the finite teacher of the binary risk object (package directions/D1_token_evidence §B, D1-10).
Host: LLaVA-1.5-7B (liuhaotian/llava-v1.5-7b, official LLaVA code as shipped in Gumpest/SparseVLMs a9e7142, dynamic_sparse=False), fp16,
prompt = question + LLaVA short-answer suffix (vicuna_v1).  Selection acts on the LLM input: kept visual tokens (raster order) are spliced
between the prompt prefix and suffix exactly as llava_arch.prepare_inputs_labels_for_multimodal splices all 576; positions are re-indexed
contiguously (as SparseVLM USE_VERSION 1_0 / FastV).  Per question: the full pass (b), 3 mechanisms × 4 nested budgets (a), a text-only
pass (0 %, diagnostic); prefill only; first-token yes / no probabilities; the posterior read-out on the balanced ±1 scale is
p̃ = P(yes) / (P(yes) + P(no)), T = 2 p̃ − 1.  Images come from the extracted GQA folder or straight from images.zip.
Modes (env /projects/EEG-foundation-model/yinghao/envs/sparsevlm, H100 / L40S only):
  bundle <id> : the shards of one bundle of DIR/d1_shards.json; per-image-chunk appends; a rerun skips questions already on disk
  probe       : QC on the R7-20 GQA fixture (binary items; never FIT / CAL / DEV): equality with the official R7-20 path, determinism,
                manipulation, yes / no mass, batched left-padded generation == official answers
  timing      : latency-regime calibration on fixture images (no answer is scored): vision, prefill-only and prefill + 8 decode steps per
                item for batch sizes × budgets
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import sys
import time
import zipfile
from argparse import Namespace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7_d1_masks as MK  # noqa: E402
import r7_vl_common as V  # noqa: E402

GQA_DIR = Path("/projects/EEG-foundation-model/yinghao/data/gqa")
GQA_ZIP = GQA_DIR / "zips" / "images.zip"
SHORT = V.SHORT


# ------------------------------------------------------------------------------------------------------------------------------ images
class ImageSource:
    """GQA image by id from the extracted folder if present, else from images.zip (member images/<id>.jpg, read without extraction)."""

    def __init__(self, folder=GQA_DIR / "images", zpath=GQA_ZIP):
        self.folder, self.zpath, self._z = Path(folder), Path(zpath), None

    def open(self, image_id: str):
        from PIL import Image
        f = self.folder / f"{image_id}.jpg"
        if f.exists():
            return Image.open(f).convert("RGB")
        if self._z is None:
            self._z = zipfile.ZipFile(self.zpath)
        return Image.open(io.BytesIO(self._z.read(f"images/{image_id}.jpg"))).convert("RGB")


# ------------------------------------------------------------------------------------------------------------------------------ host
class LlavaPrefill:
    """Official LLaVA-1.5-7B loaded once per process (node-local staging), with the pieces the masked prefill needs."""

    def __init__(self, workdir: Path):
        self.h = V.Host(workdir, sparse=False)                                           # official builder, staging, CLIP symlink farm
        import torch
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token
        self.torch, self.tok, self.model, self.proc = torch, self.h.tok, self.h.model, self.h.proc
        self._img_tok, self._idx, self._conv, self._pi, self._tit = DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, conv_templates, process_images, tokenizer_image_token
        self.yes_ids, self.no_ids, self.load_seconds = self.h.yes_ids, self.h.no_ids, self.h.load_seconds
        self.embed = self.model.get_model().embed_tokens

    def encode(self, images):
        t = self._pi(images, self.proc, self.model.config).to(dtype=self.torch.float16, device="cuda")
        with self.torch.inference_mode():
            return self.model.encode_images(t)                                             # [n, 576, 4096]

    def segments(self, question: str, with_image=True):
        qs = (self._img_tok + "\n" + question + SHORT) if with_image else (question + SHORT)
        conv = self._conv["vicuna_v1"].copy(); conv.append_message(conv.roles[0], qs); conv.append_message(conv.roles[1], None)
        ids = self._tit(conv.get_prompt(), self.tok, self._idx, return_tensors="pt")
        if not with_image:
            return ids, ids[:0]
        i = int((ids == self._idx).nonzero()[0, 0])
        return ids[:i], ids[i + 1:]

    def embed_ids(self, ids):
        with self.torch.inference_mode():
            return self.embed(ids.to("cuda"))

    def question_vector(self, question: str):
        ids = self.tok(question, add_special_tokens=False, return_tensors="pt").input_ids[0]
        return self.embed_ids(ids).float().mean(0)

    def last_logits(self, seqs):
        """Right-padded batched prefill; logits of each sequence's last real position (fp32)."""
        torch = self.torch; n, L = len(seqs), max(s.shape[0] for s in seqs); d = seqs[0].shape[1]
        x = torch.zeros(n, L, d, dtype=seqs[0].dtype, device="cuda"); am = torch.zeros(n, L, dtype=torch.long, device="cuda")
        for i, s in enumerate(seqs):
            x[i, :s.shape[0]] = s; am[i, :s.shape[0]] = 1
        with torch.inference_mode():
            hs = self.model.get_model()(inputs_embeds=x, attention_mask=am, use_cache=False, return_dict=True).last_hidden_state
            last = hs[torch.arange(n, device="cuda"), am.sum(1) - 1]
            return self.model.lm_head(last).float()

    def generate(self, seqs, max_new_tokens=128, min_new_tokens=0):
        """Batched greedy generation from spliced embeddings (left padding), bypassing LLaVA's image-only generate override."""
        torch = self.torch; from transformers import LlamaForCausalLM
        n, L = len(seqs), max(s.shape[0] for s in seqs); d = seqs[0].shape[1]
        x = torch.zeros(n, L, d, dtype=seqs[0].dtype, device="cuda"); am = torch.zeros(n, L, dtype=torch.long, device="cuda")
        for i, s in enumerate(seqs):
            x[i, L - s.shape[0]:] = s; am[i, L - s.shape[0]:] = 1
        with torch.inference_mode():
            out = LlamaForCausalLM.generate(self.model, inputs_embeds=x, attention_mask=am, do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
                                            min_new_tokens=min_new_tokens, use_cache=True, pad_token_id=self.tok.pad_token_id or 0)
        return [t.strip() for t in self.tok.batch_decode(out, skip_special_tokens=True)]


def yes_no(logits, yes_ids, no_ids):
    p = logits.softmax(-1)
    return p[:, yes_ids].sum(1).cpu().numpy(), p[:, no_ids].sum(1).cpu().numpy()


def b64(order):
    return base64.b64encode(np.asarray(order, dtype="<u2").tobytes()).decode()


def unb64(s):
    return np.frombuffer(base64.b64decode(s), dtype="<u2").astype(np.int64)


def digest(order):
    return hashlib.sha256(np.asarray(order, dtype="<u2").tobytes()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------------------------------------------ core
def variants_for(host, item, feats):
    """All sequences of one question: (key, embeddings).  key = ('full',), ('text',), (mechanism, budget)."""
    pre, post = host.segments(item["question"], True); pre_e, post_e = host.embed_ids(pre), host.embed_ids(post)
    t_ids, _ = host.segments(item["question"], False)
    qv = host.question_vector(item["question"]); f = feats.float()
    scores = ((f @ qv) / (f.norm(dim=1) * qv.norm() + 1e-12)).cpu().numpy()
    orders = MK.orderings(item["qid"], scores)
    seqs = [(("full",), host.torch.cat([pre_e, feats, post_e])), (("text",), host.embed_ids(t_ids))]
    for mech, (order, _) in orders.items():
        for k in MK.BUDGETS:
            idx = host.torch.as_tensor(MK.kept(order, k), device=feats.device)
            seqs.append(((mech, k), host.torch.cat([pre_e, feats[idx], post_e])))
    return seqs, orders, int(pre.shape[0] + post.shape[0])


def run_items(host, items, images, max_tokens=16384, max_seqs=64):
    """Teacher records for a chunk of questions whose PIL images are in `images` (image_id -> PIL)."""
    ids = sorted({it["image_id"] for it in items}); feats = host.encode([images[i] for i in ids]); fi = {i: feats[j] for j, i in enumerate(ids)}
    allseq, meta = [], {}
    for it in items:
        seqs, orders, n_text = variants_for(host, it, fi[it["image_id"]]); meta[it["qid"]] = (orders, n_text)
        allseq += [(it["qid"], key, s) for key, s in seqs]
    order = sorted(range(len(allseq)), key=lambda j: allseq[j][2].shape[0]); res = {}
    j = 0
    while j < len(order):
        batch = [order[j]]; L = allseq[order[j]][2].shape[0]; j += 1
        while j < len(order) and len(batch) < max_seqs and (len(batch) + 1) * max(L, allseq[order[j]][2].shape[0]) <= max_tokens:
            L = max(L, allseq[order[j]][2].shape[0]); batch.append(order[j]); j += 1
        py, pn = yes_no(host.last_logits([allseq[b][2] for b in batch]), host.yes_ids, host.no_ids)
        for b, y, n in zip(batch, py, pn):
            res[(allseq[b][0], allseq[b][1])] = (float(y), float(n))
    out = []
    for it in items:
        orders, n_text = meta[it["qid"]]
        out.append({"qid": it["qid"], "image_id": it["image_id"], "role": it.get("role"), "C": it["C"], "structural": it.get("structural"),
                    "semantic": it.get("semantic"), "text_tokens": n_text, "full": res[(it["qid"], ("full",))], "text": res[(it["qid"], ("text",))],
                    "masked": {m: {str(k): res[(it["qid"], (m, k))] for k in MK.BUDGETS} for m in MK.MECHANISMS},
                    "logp": {m: {str(k): orders[m][1][k] for k in MK.BUDGETS} for m in MK.MECHANISMS},
                    "order_digest": {m: digest(orders[m][0]) for m in MK.MECHANISMS}, "qsim_order": b64(orders["qsim"][0])})
    return out


# ------------------------------------------------------------------------------------------------------------------------------ modes
def launch_gates(root: Path, probe_report):
    """No teacher unit runs on an unfrozen manifest or without a passing probe (pre-registration discipline)."""
    meta = json.loads((root / "d1_manifest_meta.json").read_text())
    if meta.get("frozen") is not True:
        raise SystemExit(f"gate: manifest in {root} is not frozen (scripts/r7_d1_data.py freeze after the coordinator's go)")
    if hashlib.sha256((root / "d1_manifest.jsonl").read_bytes()).hexdigest() != meta.get("manifest_sha256"):
        raise SystemExit("gate: manifest sha256 differs from the frozen record")
    if probe_report is not None:
        rep = json.loads(Path(probe_report).read_text()) if Path(probe_report).exists() else {}
        if rep.get("pass") is not True:
            raise SystemExit(f"gate: probe report {probe_report} missing or not passing")


def cmd_bundle(a, host_factory=None, image_source=None):
    root = Path(a.manifest_dir); launch_gates(root, getattr(a, "probe_report", None))
    plan = json.loads((root / "d1_shards.json").read_text()); man = {}
    for x in (root / "d1_manifest.jsonl").read_text().splitlines():
        if x.strip():
            r = json.loads(x); man.setdefault(r["image_id"], []).append(r)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); host = None; src = image_source or ImageSource(); determinism_checked = False
    for shard in plan["bundles"][a.bundle]:
        imgs = plan["shards"][shard]; expected = [q for i in imgs for q in man[i]]
        path, done_path = out / f"{shard}.jsonl", out / f"{shard}.done.json"
        have = {r["qid"] for r in V.read_jsonl(path)}
        if done_path.exists() and len(have) == len(expected):
            print(f"[r7_d1_teacher] {shard}: complete ({len(have)})", flush=True); continue
        if path.exists():                                                              # drop a torn tail before appending
            path.write_text("".join(V.jdump(r) + "\n" for r in V.read_jsonl(path)))
        if host is None:
            host = (host_factory or (lambda: LlavaPrefill(out)))()
        t0, n_new, skipped = time.time(), 0, []
        todo_imgs = [i for i in imgs if any(q["qid"] not in have for q in man[i])]
        with open(path, "a") as fh:
            for c in range(0, len(todo_imgs), a.images_per_chunk):
                chunk = todo_imgs[c:c + a.images_per_chunk]; pil = {}
                for i in chunk:
                    try:
                        pil[i] = src.open(i)
                    except Exception as e:                                             # reason-coded, counted against coverage
                        skipped.append({"image_id": i, "reason": f"{type(e).__name__}: {e}"})
                items = [q for i in chunk if i in pil for q in man[i] if q["qid"] not in have]
                if not items:
                    continue
                recs = run_items(host, items, pil, a.max_tokens, a.max_seqs)
                if not determinism_checked:                                            # bit-identical replay of the first chunk of the bundle
                    again = run_items(host, items, pil, a.max_tokens, a.max_seqs)
                    if [r["masked"] for r in again] != [r["masked"] for r in recs] or [r["full"] for r in again] != [r["full"] for r in recs]:
                        raise RuntimeError("determinism sentinel failed: replay of the first chunk differs")
                    determinism_checked = True
                for r in recs:
                    fh.write(V.jdump(r) + "\n")
                fh.flush(); n_new += len(recs)
        rows = V.read_jsonl(path); keys = {r["qid"] for r in rows}
        rec = {"shard": shard, "expected": len(expected), "present": len(keys), "duplicates": len(rows) - len(keys), "new_this_session": n_new,
               "skipped_images": skipped, "seconds": time.time() - t0, "gpu": V.env_info().get("gpu"),
               "jsonl_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "min_full_yes_no_mass": min(sum(r["full"]) for r in rows) if rows else None}
        done_path.write_text(json.dumps(rec, indent=1))
        print(f"[r7_d1_teacher] {shard}: {len(keys)}/{len(expected)} ({n_new} new, {len(skipped)} images skipped) in {rec['seconds']:.0f}s", flush=True)
        if len(keys) + sum(len(man[s["image_id"]]) for s in skipped) < len(expected) or rec["duplicates"]:
            raise RuntimeError(f"coverage guard failed for {shard}")
    return 0


def fixture_items(fixture_path: Path):
    items = []
    for x in Path(fixture_path).read_text().splitlines():
        r = json.loads(x)
        if r.get("binary"):
            items.append({"qid": r["qid"], "image_id": r["image_id"], "image_file": r["image_file"], "question": r["question"],
                          "C": 1 if r["answer"] == "yes" else -1, "structural": r.get("structural"), "semantic": r.get("semantic"), "role": "PROBE"})
    return items


def cmd_probe(a, host=None):
    """Sentinels (all must hold before any FIT / CAL / DEV unit is fed):
    S1 the spliced full path reproduces the official R7-20 path (|ΔP(yes)| ≤ 0.02 on every binary fixture item, same yes / no argmax)
    S2 determinism: bit-identical replay;  S3 the manipulation acts (mean |T_72 − T_576| ≥ 0.02 for every mechanism)
    S4 yes / no mass ≥ 0.9 for the full pass on every item;  S5 batched left-padded generation reproduces the official answers on ≥ 98 %
    of the 200 GQA fixture items;  S6 nested kept sets and exact token counts (asserted from the recorded orderings)."""
    from PIL import Image
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); items = fixture_items(Path(a.fixture))
    host = host or LlavaPrefill(out); pil = {it["image_id"]: Image.open(it["image_file"]).convert("RGB") for it in items}
    t0 = time.time(); recs = run_items(host, items, pil); sec = time.time() - t0; again = run_items(host, items, pil)
    off = {r["qid"]: r for r in V.read_jsonl(Path(a.r720_preds)) if r["fixture"] == "gqa"}
    T = lambda pair: 2 * pair[0] / max(pair[0] + pair[1], 1e-12) - 1  # noqa: E731
    d_off = [abs(r["full"][0] - off[r["qid"]]["p_yes"]) for r in recs]
    flips = [(r["full"][0] > r["full"][1]) != (off[r["qid"]]["p_yes"] > off[r["qid"]]["p_no"]) for r in recs]
    manip = {m: float(np.mean([abs(T(r["masked"][m]["72"]) - T(r["full"])) for r in recs])) for m in MK.MECHANISMS}
    nested = True
    for r in recs:
        q = unb64(r["qsim_order"]); nested &= len(set(q.tolist())) == MK.N and all(set(MK.kept(q, k)) <= set(MK.kept(q, k2)) for k, k2 in zip(MK.BUDGETS, MK.BUDGETS[1:]))
    allf = [json.loads(x) for x in Path(a.fixture).read_text().splitlines()]; gen_same = None
    if a.check_generation:
        offa = {r["qid"]: r["text"] for r in V.read_jsonl(Path(a.r720_preds)) if r["fixture"] == "gqa"}; texts = []
        for c in range(0, len(allf), 16):
            chunk = allf[c:c + 16]; feats = host.encode([Image.open(x["image_file"]).convert("RGB") for x in chunk]); seqs = []
            for x, f in zip(chunk, feats):
                pre, post = host.segments(x["question"], True); seqs.append(host.torch.cat([host.embed_ids(pre), f, host.embed_ids(post)]))
            texts += host.generate(seqs)
        gen_same = float(np.mean([t == offa[x["qid"]] for t, x in zip(texts, allf)]))
    S = {"S1_max_abs_dP_yes_vs_official": max(d_off), "S1_argmax_flips": int(sum(flips)), "S2_bit_identical_replay": again == recs,
         "S3_mean_abs_dT_at_72": manip, "S4_min_full_mass": min(sum(r["full"]) for r in recs), "S5_generation_identical_fraction": gen_same,
         "S6_nested_and_counts": bool(nested)}
    ok = (S["S1_max_abs_dP_yes_vs_official"] <= 0.02 and not S["S1_argmax_flips"] and S["S2_bit_identical_replay"] and min(manip.values()) >= 0.02
          and S["S4_min_full_mass"] >= 0.9 and (gen_same is None or gen_same >= 0.98) and S["S6_nested_and_counts"])
    rep = {"items": len(recs), "images": len(pil), "seconds_teacher": sec, "sequences": len(recs) * 14, "sentinels": S, "pass": bool(ok),
           "load_seconds": host.load_seconds, "env": V.env_info()}
    (out / "probe_records.jsonl").write_text("".join(V.jdump(r) + "\n" for r in recs)); (out / "probe_report.json").write_text(V.jdump(rep, indent=1))
    print(V.jdump(rep, indent=1))
    return 0 if ok else 4


def cmd_timing(a, host=None):
    """Latency per item for batch sizes × budgets (grid masks), on fixture images; CUDA-synchronised; median of `reps` after one warm-up."""
    from PIL import Image
    torch = None; out = Path(a.out); out.mkdir(parents=True, exist_ok=True); host = host or LlavaPrefill(out); torch = host.torch
    allf = [json.loads(x) for x in Path(a.fixture).read_text().splitlines()]
    def sync_time(fn):
        torch.cuda.synchronize(); t = time.perf_counter(); r = fn(); torch.cuda.synchronize(); return time.perf_counter() - t, r
    rows = []
    for B in a.batches:
        chunk = (allf * ((B // len(allf)) + 1))[:B]; imgs = [Image.open(x["image_file"]).convert("RGB") for x in chunk]
        _, feats = sync_time(lambda: host.encode(imgs))
        for k in (576,) + tuple(reversed(MK.BUDGETS)):
            seqs = []
            for x, f in zip(chunk, feats):
                pre, post = host.segments(x["question"], True); idx = torch.as_tensor(MK.kept(MK.order_grid(x["qid"])[0], k) if k < 576 else np.arange(576), device="cuda")
                seqs.append(torch.cat([host.embed_ids(pre), f[idx], host.embed_ids(post)]))
            res = {"batch": B, "visual_tokens": k}
            for name, fn in (("vision", lambda: host.encode(imgs)), ("prefill", lambda: host.last_logits(seqs)),
                             ("prefill_decode8", lambda: host.generate(seqs, max_new_tokens=8, min_new_tokens=8))):
                sync_time(fn); ts = [sync_time(fn)[0] for _ in range(a.reps)]
                res[f"{name}_ms_per_item"] = 1e3 * float(np.median(ts)) / B; res[f"{name}_ms_batch"] = 1e3 * float(np.median(ts))
            res["total_ms_per_item"] = res["vision_ms_per_item"] + res["prefill_decode8_ms_per_item"]
            rows.append(res); print(V.jdump(res), flush=True)
    rep = {"rows": rows, "env": V.env_info(), "reps": a.reps, "note": "grid masks; batch of fixture items; vision is the official fp32 CLIP tower path"}
    (out / "timing_report.json").write_text(V.jdump(rep, indent=1))
    return 0


def cmd_smoke(a):
    """probe then timing in ONE host process (one weight load)."""
    host = LlavaPrefill(Path(a.out)); rc = cmd_probe(Namespace(out=str(Path(a.out) / "probe"), fixture=a.fixture, r720_preds=a.r720_preds,
                                                              check_generation=True), host)
    cmd_timing(Namespace(out=str(Path(a.out) / "timing"), fixture=a.fixture, batches=a.batches, reps=a.reps), host)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sm = sub.add_parser("smoke"); sm.add_argument("--out", required=True); sm.add_argument("--fixture", required=True)
    sm.add_argument("--r720-preds", required=True); sm.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 32]); sm.add_argument("--reps", type=int, default=3)
    b = sub.add_parser("bundle"); b.add_argument("bundle"); b.add_argument("--manifest-dir", required=True); b.add_argument("--out", required=True)
    b.add_argument("--images-per-chunk", type=int, default=16); b.add_argument("--max-tokens", type=int, default=16384); b.add_argument("--max-seqs", type=int, default=64)
    b.add_argument("--probe-report", default="/home/infres/yinwang/CS_QMI/outputs/R7_D1/smoke/probe/probe_report.json")
    p = sub.add_parser("probe"); p.add_argument("--out", required=True); p.add_argument("--fixture", required=True); p.add_argument("--r720-preds", required=True)
    p.add_argument("--check-generation", action="store_true")
    t = sub.add_parser("timing"); t.add_argument("--out", required=True); t.add_argument("--fixture", required=True)
    t.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 32]); t.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    return {"bundle": cmd_bundle, "probe": cmd_probe, "timing": cmd_timing, "smoke": cmd_smoke}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
