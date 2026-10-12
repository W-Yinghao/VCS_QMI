"""r7 R7-40a — the D1-10 finite teacher of the binary risk object on the second host, Qwen2.5-VL-7B-Instruct (plan R7_40_PLAN_DRAFT.yaml;
host interface in scripts/r7_d1_40_host.py; masks in scripts/r7_d1_40_masks.py).
Per question (frozen D1-10 rule, 14 prefill passes): the full pass (b), 3 mechanisms × 4 nested budgets (a) — budgets are fractions
0.125 / 0.25 / 0.5 / 0.75 of the image's own N merged tokens — and a text-only pass (diagnostic); first-token yes / no probabilities;
p̃ = P(yes) / (P(yes) + P(no)), T = 2 p̃ − 1.  Prompt: Qwen chat template, user content [image, question + LLaVA short-answer suffix] (the D1-10
teacher prompt and the R7-20 Qwen fixture prompt).  Selection acts after the token merger; kept tokens keep their original 3-D rotary ids.
Modes (env /projects/EEG-foundation-model/yinghao/envs/vl_baselines; any of RTX6000PRO / H100 / L40S):
  bundle <id> : the shards of one bundle of DIR/d1_shards.json (the frozen D1 plan); per-image-chunk appends to the shard's own file (one
                writer); a rerun skips questions already on disk; DEV shards refused unless --allow-dev (R7-40c only); images from the extracted
                GQA folder only (no zip fallback: the archives were deleted on 2026-10-12; a missing file stops the unit)
  imagecheck  : every manifest image present in the extracted folder (names only)
  probe       : sentinels S0–S7 on the R7-20 GQA fixture (never FIT / CAL / DEV)
  timing      : latency-regime timing on fixture images (no answer scored): frozen rule re-applied on the reference GPU (nodesumo01)
  smoke       : probe + throughput + memory probe in one host process (+ timing when --timing is given)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7_d1_40_host as QH  # noqa: E402
import r7_d1_40_masks as M4  # noqa: E402
import r7_d1_teacher as TE  # noqa: E402  (frozen D1-10 module: image source, launch gates, order encodings, fixture items)
import r7_vl_common as V  # noqa: E402

R40 = Path("/home/infres/yinwang/CS_QMI/outputs/R7_D1/r7_40")
GQA_IMAGES = Path("/projects/EEG-foundation-model/yinghao/data/gqa/images")
FIXTURE = Path("/projects/EEG-foundation-model/yinghao/data/r7_fixtures/gqa.jsonl")
R720_QWEN = Path("/home/infres/yinwang/CS_QMI/outputs/R7_fixtures/d1/pred_qwen25vl7b.jsonl")
REFERENCE_GPU = "NVIDIA H100 NVL"
SHORT = QH.SHORT


# ------------------------------------------------------------------------------------------------------------------------------ images
class MissingImage(FileNotFoundError):
    """A manifest image is not in the extracted folder: R7-40 stops (it never falls back to an archive)."""


class ExtractedImageSource:
    """GQA image by id from the extracted folder ONLY.  The image archives (gqa/zips/images.zip and others) were deleted by the owner on
    2026-10-12 09:26 UTC after a file-by-file check of the extracted folders (coordinator message; data/DATA_MANIFEST.json deletion_candidates);
    unlike the frozen LLaVA r7_d1_teacher.ImageSource there is no zip fallback: a missing file raises MissingImage, which stops the unit."""

    def __init__(self, folder=GQA_IMAGES):
        self.folder = Path(folder)

    def path(self, image_id: str) -> Path:
        return self.folder / f"{image_id}.jpg"

    def open(self, image_id: str):
        from PIL import Image
        f = self.path(image_id)
        if not f.is_file():
            raise MissingImage(f"GQA image {image_id} is missing from the extracted folder {self.folder}; R7-40 never reads a zip archive "
                               "(the archives were deleted on 2026-10-12) — stop and report")
        return Image.open(f).convert("RGB")


# ------------------------------------------------------------------------------------------------------------------------------ core
def variants_for(host, item, feat, grid, thw):
    """All sequences of one question: (key, embeddings, (3, L) position ids); key = ('full',), ('text',), (mechanism, fraction key)."""
    torch = host.torch; prompt = item["question"] + SHORT
    pre, post = host.segments(prompt, True); pre_e, post_e = host.core.embed_ids(pre), host.core.embed_ids(post)
    t_ids, _ = host.segments(prompt, False)
    rows, cols = grid; n = rows * cols
    if feat.shape[0] != n:
        raise RuntimeError(f"visual token count {feat.shape[0]} != grid {rows} x {cols}")
    pos_full = host.core.full_positions(pre, n, post, thw)
    qv = host.question_vector(item["question"]); f = feat.float()
    scores = ((f @ qv) / (f.norm(dim=1) * qv.norm() + 1e-12)).cpu().numpy()
    orders = M4.orderings(item["qid"], rows, cols, scores)
    seqs = [(("full",), torch.cat([pre_e, feat, post_e]), pos_full), (("text",), host.core.embed_ids(t_ids), host.core.text_positions(len(t_ids)))]
    for mech in M4.MECHANISMS:
        order = orders[mech][0]
        for key, k in M4.budgets(n).items():
            idx = M4.kept(order, k)
            seqs.append(((mech, key), torch.cat([pre_e, feat[torch.as_tensor(idx, device=feat.device)], post_e]),
                         QH.prune_positions(pos_full, len(pre), n, idx)))
    return seqs, orders, int(len(pre) + len(post)), n, (rows, cols)


def run_items(host, items, images, max_tokens=16384, max_seqs=64):
    """Teacher records for a chunk of questions whose PIL images are in `images` (image_id -> PIL)."""
    ids = sorted({it["image_id"] for it in items}); enc = host.encode([images[i] for i in ids]); fi = {i: enc[j] for j, i in enumerate(ids)}
    allseq, meta = [], {}
    for it in items:
        seqs, orders, n_text, n, grid = variants_for(host, it, *fi[it["image_id"]]); meta[it["qid"]] = (orders, n_text, n, grid)
        allseq += [(it["qid"], key, s, p) for key, s, p in seqs]
    order = sorted(range(len(allseq)), key=lambda j: allseq[j][2].shape[0]); res = {}; j = 0
    while j < len(order):
        batch = [order[j]]; L = allseq[order[j]][2].shape[0]; j += 1
        while j < len(order) and len(batch) < max_seqs and (len(batch) + 1) * max(L, allseq[order[j]][2].shape[0]) <= max_tokens:
            L = max(L, allseq[order[j]][2].shape[0]); batch.append(order[j]); j += 1
        py, pn = QH.yes_no(host.core.last_logits([allseq[b][2] for b in batch], [allseq[b][3] for b in batch]), host.yes_ids, host.no_ids)
        for b, y, nn in zip(batch, py, pn):
            res[(allseq[b][0], allseq[b][1])] = (float(y), float(nn))
    out = []
    for it in items:
        orders, n_text, n, grid = meta[it["qid"]]
        out.append({"qid": it["qid"], "image_id": it["image_id"], "role": it.get("role"), "C": it["C"], "structural": it.get("structural"),
                    "semantic": it.get("semantic"), "n_vis": n, "grid": list(grid), "budgets": M4.budgets(n), "text_tokens": n_text,
                    "full": res[(it["qid"], ("full",))], "text": res[(it["qid"], ("text",))],
                    "masked": {m: {key: res[(it["qid"], (m, key))] for key in M4.FKEYS} for m in M4.MECHANISMS},
                    "logp": {m: {key: orders[m][1][key] for key in M4.FKEYS} for m in M4.MECHANISMS},
                    "order_digest": {m: TE.digest(orders[m][0]) for m in M4.MECHANISMS}, "qsim_order": TE.b64(orders["qsim"][0])})
    return out


# ------------------------------------------------------------------------------------------------------------------------------ bundle
def cmd_bundle(a, host_factory=None, image_source=None):
    root = Path(a.manifest_dir); TE.launch_gates(root, getattr(a, "probe_report", None))
    plan = json.loads((root / "d1_shards.json").read_text()); man = {}
    shards = plan["bundles"][a.bundle]
    if any(s.startswith("DEV-") for s in shards) and not getattr(a, "allow_dev", False):
        raise SystemExit(f"bundle {a.bundle} holds DEV shards: DEV teacher passes are sealed until R7-40c (--allow-dev)")
    for x in (root / "d1_manifest.jsonl").read_text().splitlines():
        if x.strip():
            r = json.loads(x); man.setdefault(r["image_id"], []).append(r)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); host = None; src = image_source or ExtractedImageSource(); determinism_checked = False
    for shard in shards:
        imgs = plan["shards"][shard]; expected = [q for i in imgs for q in man[i]]
        path, done_path = out / f"{shard}.jsonl", out / f"{shard}.done.json"
        have = {r["qid"] for r in V.read_jsonl(path)}
        if done_path.exists() and len(have) == len(expected):
            print(f"[r7_d1_40_teacher] {shard}: complete ({len(have)})", flush=True); continue
        if path.exists():                                                              # drop a torn tail before appending (single writer)
            path.write_text("".join(V.jdump(r) + "\n" for r in V.read_jsonl(path)))
        if host is None:
            host = (host_factory or (lambda: QH.QwenPrefill(out)))()
        t0, n_new, skipped = time.time(), 0, []
        todo_imgs = [i for i in imgs if any(q["qid"] not in have for q in man[i])]
        with open(path, "a") as fh:
            for c in range(0, len(todo_imgs), a.images_per_chunk):
                chunk = todo_imgs[c:c + a.images_per_chunk]; pil = {}
                for i in chunk:
                    try:
                        pil[i] = src.open(i)
                    except MissingImage:                                               # never skipped silently, never read from a zip
                        raise
                    except Exception as e:                                             # unreadable file: reason-coded, counted against coverage
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
               "jsonl_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "min_full_yes_no_mass": min(sum(r["full"]) for r in rows) if rows else None,
               "mean_visual_tokens": float(np.mean([r["n_vis"] for r in rows])) if rows else None}
        done_path.write_text(json.dumps(rec, indent=1))
        print(f"[r7_d1_40_teacher] {shard}: {len(keys)}/{len(expected)} ({n_new} new, {len(skipped)} images skipped) in {rec['seconds']:.0f}s", flush=True)
        if len(keys) + sum(len(man[s["image_id"]]) for s in skipped) < len(expected) or rec["duplicates"]:
            raise RuntimeError(f"coverage guard failed for {shard}")
    (out / f"bundle_{a.bundle}.done.json").write_text(json.dumps({"bundle": a.bundle, "shards": shards, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                                  "gpu": V.env_info().get("gpu") if host is not None else None}, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------------------------------------ probe
def s4_pass(masses, rule: str) -> bool:
    """S4 yes / no mass rule.  'min' (frozen D1-10 sentinel): every binary probe item's full-pass mass >= 0.9.  'rate:<r>' (proposed for
    Qwen, OPEN decision): a fraction >= r of the binary probe items has mass >= 0.9."""
    m = np.asarray(masses, float)
    if rule == "min":
        return bool(m.min() >= 0.9)
    if rule.startswith("rate:"):
        return bool(np.mean(m >= 0.9) >= float(rule.split(":", 1)[1]))
    raise ValueError(f"unknown S4 rule {rule}")


def cmd_probe(a, host=None):
    """Sentinels on the R7-20 GQA fixture (binary items for S1–S4 / S6–S7; all 200 items for S0 / S5).  All must hold before any FIT / CAL
    unit is fed (the bundle refuses without a passing report):
    S0 host interface: spliced ids (pre + N image tokens + post) == the processor's input_ids; N == image tokens == t·h·w / 4 == the R7-20
       count; pixel values of the image processor == those of the processor call (bitwise)
    S1 the spliced full path (batched teacher prefill) reproduces the official path: |ΔP(yes)| <= 0.02, no yes / no argmax flip; rule
       --s1-rule 'r720' (frozen: against the R7-20 values, computed on an H100 NVL) or 'same_gpu' (S1c: against the official path on this
       GPU; OPEN decision — bf16 GPU-type differences reach 0.054 on L40S vs H100 NVL, S1d); S1b the spliced full prefill at batch 1 against
       the official path on the same GPU: |ΔP(yes)| <= 1e-3
    S2 bit-identical replay;  S3 masking acts: mean |T_0.125 − T_full| >= 0.02 for every mechanism
    S4 yes / no mass of the full pass (rule --s4-rule: 'min' = frozen, every item >= 0.9; 'rate:r' = OPEN proposal)
    S5 batched left-padded generation (batch 16) reproduces the official R7-20 answers on >= 98 % of the 200 items (frozen); S5b batch-1
       generation == the official generate on the same GPU on >= 98 % (expected 100 %)
    S6 nested kept sets with exact counts k = round(f·N); kept tokens keep the full sequence's (t, h, w) ids; image ids are (s, s + r, s + c)
    S7 all tokens kept through the masking path reproduce the full sequence (embeddings and position ids) bit for bit"""
    from PIL import Image
    torch = None; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    host = host or QH.QwenPrefill(out); torch = host.torch
    allf = [json.loads(x) for x in Path(a.fixture).read_text().splitlines() if x.strip()]; items = TE.fixture_items(Path(a.fixture))
    pil = {x["image_id"]: Image.open(x["image_file"]).convert("RGB") for x in allf}
    r720 = {r["qid"]: r for r in V.read_jsonl(Path(a.r720_preds)) if r["fixture"] == "gqa"}
    # S0 / S1b / S5b: the official path on this GPU (batch 1, as R7-20)
    s0 = {"ids_equal": 0, "n_vis_equal": 0, "pixels_equal": 0, "n_vis_equal_r720": 0}; official = {}; t0 = time.time()
    for x in allf:
        o = host.official_answer(x["question"] + SHORT, pil[x["image_id"]]); official[x["qid"]] = o
        pre, post = host.segments(x["question"] + SHORT, True); thw = o["image_grid_thw"][0].cpu(); n = int(thw.prod()) // host.core.merge ** 2
        mine = np.concatenate([pre, np.full(n, host.image_token_id), post])
        s0["ids_equal"] += int(np.array_equal(mine, o["input_ids"].numpy()))
        s0["n_vis_equal"] += int(n == int((o["input_ids"] == host.image_token_id).sum()))
        s0["n_vis_equal_r720"] += int(n == r720[x["qid"]]["visual_tokens"])
        pv = host.preprocess([pil[x["image_id"]]])["pixel_values"]
        s0["pixels_equal"] += int(torch.equal(pv, o.pop("pixel_values").cpu()))
    t_official = time.time() - t0
    # teacher records (batched, as in the bundles) + replay
    t0 = time.time(); recs = run_items(host, items, pil); sec = time.time() - t0; again = run_items(host, items, pil)
    T = lambda pair: 2 * pair[0] / max(pair[0] + pair[1], 1e-12) - 1  # noqa: E731
    d_r720 = [abs(r["full"][0] - r720[r["qid"]]["p_yes"]) for r in recs]
    d_batched_same = np.asarray([abs(r["full"][0] - official[r["qid"]]["p_yes"]) for r in recs])          # S1c: batching alone (same GPU)
    d_official_r720 = np.asarray([abs(official[r["qid"]]["p_yes"] - r720[r["qid"]]["p_yes"]) for r in recs])  # S1d: GPU type alone (batch 1)
    flips = [(r["full"][0] > r["full"][1]) != (r720[r["qid"]]["p_yes"] > r720[r["qid"]]["p_no"]) for r in recs]
    manip = {m: float(np.mean([abs(T(r["masked"][m]["0.125"]) - T(r["full"])) for r in recs])) for m in M4.MECHANISMS}
    masses = [sum(r["full"]) for r in recs]
    # S1b: spliced full prefill at batch 1 vs the official path on this GPU
    d_same, s6_ok, s7_ok = [], True, True
    for it in items:
        f, grid, thw = host.encode([pil[it["image_id"]]])[0]
        seqs, orders, _, n, _ = variants_for(host, it, f, grid, thw)
        full = next(s for s in seqs if s[0] == ("full",))
        py, _ = QH.yes_no(host.core.last_logits([full[1]], [full[2]]), host.yes_ids, host.no_ids)
        d_same.append(abs(float(py[0]) - official[it["qid"]]["p_yes"]))
        # S6: counts, nesting, ids of kept tokens
        rec = next(r for r in recs if r["qid"] == it["qid"]); rows, cols = rec["grid"]; ks = M4.budgets(rows * cols)
        q = TE.unb64(rec["qsim_order"]); pos_full = full[2]; pre, post = host.segments(it["question"] + SHORT, True)
        s = int(pos_full[0, len(pre)]); r_, c_ = np.divmod(np.arange(n), cols)
        img_ids = pos_full[:, len(pre):len(pre) + n].numpy()
        s6_ok &= bool(np.array_equal(img_ids, np.stack([np.full(n, s), s + r_, s + c_])))
        s6_ok &= bool(np.array_equal(pos_full[:, len(pre) + n:].numpy()[0], s + max(rows, cols) + np.arange(len(post))))
        for mech, order in (("random", M4.order_random(it["qid"], rows, cols)[0]), ("grid", M4.order_grid(it["qid"], rows, cols, False)[0]), ("qsim", q)):
            s6_ok &= len(set(order.tolist())) == n and TE.digest(order) == rec["order_digest"][mech]
            sets = [set(M4.kept(order, ks[key]).tolist()) for key in M4.FKEYS]
            s6_ok &= [len(x) for x in sets] == [ks[key] for key in M4.FKEYS] and all(x <= y for x, y in zip(sets, sets[1:]))
            for key in M4.FKEYS:
                kk = M4.kept(order, ks[key]); pp = QH.prune_positions(pos_full, len(pre), n, kk)
                s6_ok &= bool(torch.equal(pp[:, len(pre):len(pre) + len(kk)], pos_full[:, len(pre) + torch.as_tensor(kk)]))
        allk = np.arange(n); pe = torch.cat([host.core.embed_ids(pre), f[torch.as_tensor(allk, device=f.device)], host.core.embed_ids(post)])
        s7_ok &= bool(torch.equal(pe, full[1])) and bool(torch.equal(QH.prune_positions(pos_full, len(pre), n, allk), pos_full))
    # S5 / S5b: generation through the spliced path
    texts_b, texts_1 = {}, {}
    for bsz, store in ((a.gen_batch, texts_b), (1, texts_1)):
        for c in range(0, len(allf), bsz):
            chunk = allf[c:c + bsz]; enc = host.encode([pil[x["image_id"]] for x in chunk]); seqs, poss, pids = [], [], []
            for x, (f, grid, thw) in zip(chunk, enc):
                pre, post = host.segments(x["question"] + SHORT, True); n = f.shape[0]
                seqs.append(torch.cat([host.core.embed_ids(pre), f, host.core.embed_ids(post)]))
                poss.append(host.core.full_positions(pre, n, post, thw)); pids.append(np.concatenate([pre, np.full(n, host.image_token_id), post]))
            new, _ = host.core.generate(seqs, poss, pids, max_new_tokens=128)
            for x, t in zip(chunk, host.decode(new)):
                store[x["qid"]] = t
    gen_r720 = float(np.mean([texts_b[x["qid"]] == r720[x["qid"]]["text"] for x in allf]))
    gen_same = float(np.mean([texts_1[x["qid"]] == official[x["qid"]]["text"] for x in allf]))
    S = {"S0_host_interface": {**s0, "items": len(allf)}, "S1_max_abs_dP_yes_vs_r720": max(d_r720), "S1_argmax_flips": int(sum(flips)),
         "S1b_max_abs_dP_yes_same_gpu_batch1": max(d_same), "S1b_frac_within_1e-3": float(np.mean(np.asarray(d_same) <= 1e-3)),
         "S1c_batched_vs_official_same_gpu": {"max": float(d_batched_same.max()), "median": float(np.median(d_batched_same)), "n_gt_0.02": int((d_batched_same > 0.02).sum()),
                                              "flips": int(sum((r["full"][0] > r["full"][1]) != (official[r["qid"]]["p_yes"] > official[r["qid"]]["p_no"]) for r in recs))},
         "S1d_official_this_gpu_vs_r720": {"max": float(d_official_r720.max()), "median": float(np.median(d_official_r720)), "n_gt_0.02": int((d_official_r720 > 0.02).sum())},
         "S2_bit_identical_replay": again == recs, "S3_mean_abs_dT_at_0.125": manip,
         "S4_min_full_mass": float(min(masses)), "S4_frac_full_mass_ge_0.9": float(np.mean(np.asarray(masses) >= 0.9)),
         "S4_low_mass_items": [{"qid": r["qid"], "mass": sum(r["full"]), "structural": r["structural"], "semantic": r["semantic"]}
                               for r in recs if sum(r["full"]) < 0.9],
         "S5_generation_identical_fraction_vs_r720": gen_r720, "S5b_generation_identical_fraction_same_gpu_batch1": gen_same,
         "S6_nested_counts_and_ids": bool(s6_ok), "S7_all_kept_equals_full": bool(s7_ok)}
    s1 = {"r720": S["S1_max_abs_dP_yes_vs_r720"] <= 0.02 and not S["S1_argmax_flips"],                    # frozen: vs the R7-20 path (H100 NVL)
          "same_gpu": S["S1c_batched_vs_official_same_gpu"]["max"] <= 0.02 and not S["S1c_batched_vs_official_same_gpu"]["flips"]}  # OPEN D40-2
    ok_other = (all(v == len(allf) for k, v in s0.items()) and S["S1b_max_abs_dP_yes_same_gpu_batch1"] <= 1e-3 and S["S2_bit_identical_replay"]
                and min(manip.values()) >= 0.02 and gen_r720 >= 0.98 and gen_same >= 0.98 and S["S6_nested_counts_and_ids"] and S["S7_all_kept_equals_full"])
    ok_common = bool(ok_other and s1[a.s1_rule])
    by_rule = {f"S1={r1},S4={r4}": bool(ok_other and s1[r1] and s4_pass(masses, r4)) for r1 in ("r720", "same_gpu") for r4 in ("min", "rate:0.9")}
    rep = {"items": len(recs), "images": len({r["image_id"] for r in recs}), "fixture_items": len(allf), "seconds_teacher": sec,
           "seconds_official_200": t_official, "sequences": len(recs) * 14, "sentinels": S, "s4_rule": a.s4_rule, "s1_rule": a.s1_rule,
           "pass": bool(ok_common and s4_pass(masses, a.s4_rule)), "pass_without_s4": bool(ok_common), "pass_by_s4_rule": by_rule,
           "host": host.info, "load_seconds": host.load_seconds, "stage_notes": host.stage_notes, "env": V.env_info(),
           "mean_visual_tokens": float(np.mean([r["n_vis"] for r in recs]))}
    (out / "probe_records.jsonl").write_text("".join(V.jdump(r) + "\n" for r in recs)); (out / "probe_report.json").write_text(V.jdump(rep, indent=1))
    teach = {r["qid"]: r["full"] for r in recs}
    (out / "probe_generations.jsonl").write_text("".join(V.jdump({"qid": x["qid"], "batched": texts_b[x["qid"]], "batch1": texts_1[x["qid"]],
                                                                   "official_same_gpu": official[x["qid"]]["text"], "r720": r720[x["qid"]]["text"],
                                                                   "p_official_same_gpu": [official[x["qid"]]["p_yes"], official[x["qid"]]["p_no"]],
                                                                   "p_r720": [r720[x["qid"]]["p_yes"], r720[x["qid"]]["p_no"]], "p_teacher_batched": teach.get(x["qid"])}) + "\n" for x in allf))
    print(V.jdump(rep, indent=1))
    return 0 if rep["pass"] else 4


# ------------------------------------------------------------------------------------------------------------------------------ timing
def grid_batch(host, chunk, feats, f):
    """Spliced sequences of a batch at fraction f of each image's tokens (uniform grid mechanism), with position ids and prompt ids."""
    torch = host.torch; seqs, poss, pids = [], [], []
    for x, (feat, grid, thw) in zip(chunk, feats):
        pre, post = host.segments(x["question"] + SHORT, True); rows, cols = grid; n = rows * cols
        pos_full = host.core.full_positions(pre, n, post, thw)
        idx = np.arange(n) if f >= 1.0 else M4.kept(M4.order_grid(x["qid"], rows, cols, False)[0], M4.budget(f, n))
        seqs.append(torch.cat([host.core.embed_ids(pre), feat[torch.as_tensor(idx, device=feat.device)], host.core.embed_ids(post)]))
        poss.append(QH.prune_positions(pos_full, len(pre), n, idx)); pids.append(np.concatenate([pre, np.full(len(idx), host.image_token_id), post]))
    return seqs, poss, pids


def cmd_timing(a, host=None):
    """Latency-regime calibration (pre-declared in reports/R7/R7_40_PROTOCOL_DRAFT_20261012.yaml before this ran): GQA fixture items in
    file order; for B in {1, 4, 16, 32} consecutive batches of B over the first 64 items (B = 1) or the first 192 items (B >= 4); for each
    fraction f in {1, 0.75, 0.5, 0.25, 0.125} of uniform-grid kept tokens, one warm-up batch (not recorded), then per batch, CUDA-synchronised:
    vision = image processor + vision tower (images opened before the clock), select = grid ordering + gather + embedding assembly + position
    ids, prefill = a separate prefill-only pass (TTFT), llm = prefill + 7 decode forwards (8 forced new tokens).  Per-item total = (vision +
    select + llm) / B per batch; median over batches.  Rule (frozen D1 latency_regime, re-applied): B* = the smallest B whose 25 % row saves
    >= 25 % of the 100 % row's median per-item total; none -> no LLM-side latency saving on Qwen (primary at fixed fraction)."""
    from PIL import Image
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); host = host or QH.QwenPrefill(out); torch = host.torch
    gpu = torch.cuda.get_device_name(0)
    if gpu != REFERENCE_GPU and not a.allow_any_gpu:
        raise SystemExit(f"latency runs only on the reference GPU ({REFERENCE_GPU}), got {gpu}")
    allf = [json.loads(x) for x in Path(a.fixture).read_text().splitlines() if x.strip()]

    def clock():
        torch.cuda.synchronize(); return time.perf_counter()
    rows = []
    for B in a.batches:
        pool = allf[: (a.n1 if B == 1 else a.n)]; batches = [pool[i:i + B] for i in range(0, len(pool) - B + 1, B)]
        for f in (1.0, 0.75, 0.5, 0.25, 0.125):
            comp = {k: [] for k in ("vision", "select", "prefill", "llm")}; kept_tok = []
            for bi, chunk in enumerate([batches[0]] + batches):                        # first batch repeated as warm-up (not recorded)
                imgs = [Image.open(x["image_file"]).convert("RGB") for x in chunk]
                t0 = clock(); feats = host.encode(imgs); t1 = clock()
                seqs, poss, pids = grid_batch(host, chunk, feats, f); t2 = clock()
                host.core.last_logits(seqs, poss); t3 = clock()
                host.core.generate(seqs, poss, pids, max_new_tokens=8, min_new_tokens=8); t4 = clock()
                if bi:
                    for k, v in (("vision", t1 - t0), ("select", t2 - t1), ("prefill", t3 - t2), ("llm", t4 - t3)):
                        comp[k].append(1e3 * v)
                    kept_tok += [int((np.asarray(p) == host.image_token_id).sum()) for p in pids]
            c = {k: np.asarray(v) for k, v in comp.items()}; per = (c["vision"] + c["select"] + c["llm"]) / B
            per_d110 = (c["vision"] + c["llm"]) / B
            rows.append({"batch": B, "fraction": f, "n_batches": len(per), "per_item_ms_median": float(np.median(per)),
                         "per_item_ms_p95": float(np.percentile(per, 95)), "per_item_ms_median_no_select": float(np.median(per_d110)),
                         **{f"{k}_ms_per_item_median": float(np.median(v / B)) for k, v in c.items()}, "ttft_ms_batch_median": float(np.median(c["vision"] + c["select"] + c["prefill"])),
                         "mean_visual_tokens_kept": float(np.mean(kept_tok))})
            print(V.jdump(rows[-1]), flush=True)
    savings, savings_ns = {}, {}
    for B in a.batches:
        full = next(r for r in rows if r["batch"] == B and r["fraction"] == 1.0)
        for f in (0.75, 0.5, 0.25, 0.125):
            r = next(r for r in rows if r["batch"] == B and r["fraction"] == f)
            savings[f"B{B}_f{f:g}"] = 1 - r["per_item_ms_median"] / full["per_item_ms_median"]
            savings_ns[f"B{B}_f{f:g}"] = 1 - r["per_item_ms_median_no_select"] / full["per_item_ms_median_no_select"]
    qual = [B for B in a.batches if savings[f"B{B}_f0.25"] >= 0.25]
    rep = {"rows": rows, "saving_vs_full": savings, "saving_vs_full_without_select": savings_ns, "B_star": min(qual) if qual else None,
           "B_star_without_select": min([B for B in a.batches if savings_ns[f"B{B}_f0.25"] >= 0.25], default=None),
           "rule": "smallest B in {1, 4, 16, 32} whose uniform-grid 25 % row saves >= 25 % of the 100 % row's median per-item total (vision + select + prefill + 7 decode forwards)",
           "gpu": gpu, "host": __import__("socket").gethostname(), "env": V.env_info(), "torch_threads": torch.get_num_threads(),
           "pools": {"B1": a.n1, "B>=4": a.n}, "host_info": host.info}
    (out / "timing_report.json").write_text(V.jdump(rep, indent=1)); print(V.jdump({k: rep[k] for k in ("saving_vs_full", "B_star", "B_star_without_select")}, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------------------------------------ smoke extras
def cmd_throughput(a, host):
    """Teacher cost on the 200 fixture items (all treated as teacher questions; C is a placeholder) with the bundle chunking (16 images):
    the batched prefill of the bundles (<= 16,384 tokens / 64 sequences) and, for comparison, every pass at batch 1 (the official path's
    computation); per pass the batched-vs-batch-1 differences of P(yes) and T (bf16 batch-composition noise of the teacher)."""
    from PIL import Image
    allf = [json.loads(x) for x in Path(a.fixture).read_text().splitlines() if x.strip()]
    items = [{"qid": x["qid"], "image_id": x["image_id"], "question": x["question"], "C": 0, "role": "PROBE"} for x in allf]
    imgs = list(dict.fromkeys(x["image_id"] for x in allf)); pil = {x["image_id"]: Image.open(x["image_file"]).convert("RGB") for x in allf}
    torch = host.torch; res = {}; recs = {}
    for mode, kw in (("batched", {}), ("batch1", {"max_seqs": 1})):
        V.cuda_peak_reset(torch); torch.cuda.synchronize(); t0 = time.time(); n = 0; recs[mode] = []
        for c in range(0, len(imgs), 16):
            chunk = set(imgs[c:c + 16]); its = [it for it in items if it["image_id"] in chunk]; recs[mode] += run_items(host, its, {i: pil[i] for i in chunk}, **kw); n += len(its)
        torch.cuda.synchronize(); sec = time.time() - t0
        res[mode] = {"questions": n, "images": len(imgs), "seconds": sec, "s_per_question": sec / n, "peak_mem_gb": V.cuda_peak_gb(torch)}
    T = lambda pair: 2 * pair[0] / max(pair[0] + pair[1], 1e-12) - 1  # noqa: E731
    one = {r["qid"]: r for r in recs["batch1"]}; dP, dT, flips = [], [], 0
    for r in recs["batched"]:
        o = one[r["qid"]]; pairs = [(r["full"], o["full"]), (r["text"], o["text"])] + [(r["masked"][m][k], o["masked"][m][k]) for m in M4.MECHANISMS for k in M4.FKEYS]
        for x, y in pairs:
            dP.append(abs(x[0] - y[0])); dT.append(abs(T(x) - T(y))); flips += int((x[0] > x[1]) != (y[0] > y[1]))
    dP, dT = np.asarray(dP), np.asarray(dT)
    res["batched_vs_batch1"] = {"passes": int(len(dP)), "max_dP_yes": float(dP.max()), "median_dP_yes": float(np.median(dP)), "n_dP_gt_0.02": int((dP > 0.02).sum()),
                                "max_dT": float(dT.max()), "p99_dT": float(np.percentile(dT, 99)), "mean_dT": float(dT.mean()), "argmax_flips": flips}
    res["gpu"] = V.env_info().get("gpu")
    return res


def cmd_memprobe(a, host):
    """Worst-case memory of a teacher unit (E2 / E5 lesson): a chunk of 16 synthetic 1280-token images through the tower, then a
    max_tokens = 16384 prefill batch of the longest sequences (13 × 1,260 tokens)."""
    from PIL import Image
    torch = host.torch; g = np.random.default_rng(0)
    big = [Image.fromarray(g.integers(0, 255, (1008, 1008, 3), dtype=np.uint8)) for _ in range(16)]
    V.cuda_peak_reset(torch); enc = host.encode(big); peak_vis = V.cuda_peak_gb(torch); n = enc[0][0].shape[0]
    pre, post = host.segments("Is there either a dog or a giraffe in the picture?" + SHORT, True)
    seq = torch.cat([host.core.embed_ids(pre), enc[0][0], host.core.embed_ids(post)]); pos = host.core.full_positions(pre, n, post, enc[0][2])
    m = max(1, 16384 // seq.shape[0]); V.cuda_peak_reset(torch); host.core.last_logits([seq] * m, [pos] * m); peak_llm = V.cuda_peak_gb(torch)
    return {"tower_chunk_16_images_tokens_each": n, "peak_gb_tower": peak_vis, "prefill_batch": m, "prefill_seq_tokens": int(seq.shape[0]),
            "peak_gb_prefill": peak_llm, "gpu_total_gb": V.env_info().get("gpu_mem_gb")}


def cmd_smoke(a):
    """probe, throughput, memory probe (and timing with --timing) in ONE host process (one weight load)."""
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); host = QH.QwenPrefill(out)
    rc = cmd_probe(Namespace(out=str(out / "probe"), fixture=a.fixture, r720_preds=a.r720_preds, gen_batch=a.gen_batch, s4_rule=a.s4_rule,
                             s1_rule=a.s1_rule), host)
    extra = {"throughput": cmd_throughput(a, host), "memory": cmd_memprobe(a, host), "load_seconds": host.load_seconds, "stage_notes": host.stage_notes}
    (out / "smoke_extra.json").write_text(V.jdump(extra, indent=1)); print(V.jdump(extra, indent=1))
    if a.timing:
        cmd_timing(Namespace(out=str(out / "timing"), fixture=a.fixture, batches=a.batches, n=a.n, n1=a.n1, allow_any_gpu=a.allow_any_gpu), host)
    return rc


def cmd_imagecheck(a):
    """Every manifest image of the requested roles is a file in the extracted GQA folder (names only; nothing is decoded or scored)."""
    src = ExtractedImageSource(a.images); rows = [json.loads(x) for x in (Path(a.manifest_dir) / "d1_manifest.jsonl").read_text().splitlines() if x.strip()]
    res = {"folder": str(src.folder), "roles": {}}
    for role in a.roles:
        ids = sorted({r["image_id"] for r in rows if r["role"] == role}); missing = [i for i in ids if not src.path(i).is_file()]
        res["roles"][role] = {"images": len(ids), "missing": len(missing), "missing_examples": missing[:10]}
    res["complete"] = all(v["missing"] == 0 for v in res["roles"].values())
    Path(a.out).mkdir(parents=True, exist_ok=True); (Path(a.out) / "imagecheck.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
    return 0 if res["complete"] else 3


def main() -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    ic = sub.add_parser("imagecheck"); ic.add_argument("--manifest-dir", required=True); ic.add_argument("--out", required=True)
    ic.add_argument("--images", default=str(GQA_IMAGES)); ic.add_argument("--roles", nargs="+", default=["FIT", "CAL", "DEV"])
    b = sub.add_parser("bundle"); b.add_argument("bundle"); b.add_argument("--manifest-dir", required=True); b.add_argument("--out", required=True)
    b.add_argument("--images-per-chunk", type=int, default=16); b.add_argument("--max-tokens", type=int, default=16384); b.add_argument("--max-seqs", type=int, default=64)
    b.add_argument("--probe-report", default=str(R40 / "probe" / "probe_report.json")); b.add_argument("--allow-dev", action="store_true")
    for name in ("probe", "smoke"):
        p = sub.add_parser(name); p.add_argument("--out", required=True); p.add_argument("--fixture", default=str(FIXTURE))
        p.add_argument("--r720-preds", default=str(R720_QWEN)); p.add_argument("--gen-batch", type=int, default=16)
        p.add_argument("--s4-rule", default="min"); p.add_argument("--s1-rule", choices=["r720", "same_gpu"], default="r720")
        if name == "smoke":
            p.add_argument("--timing", action="store_true"); p.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 32])
            p.add_argument("--n", type=int, default=192); p.add_argument("--n1", type=int, default=64); p.add_argument("--allow-any-gpu", action="store_true")
    t = sub.add_parser("timing"); t.add_argument("--out", required=True); t.add_argument("--fixture", default=str(FIXTURE))
    t.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 32]); t.add_argument("--n", type=int, default=192); t.add_argument("--n1", type=int, default=64)
    t.add_argument("--allow-any-gpu", action="store_true")
    a = ap.parse_args()
    return {"bundle": cmd_bundle, "probe": cmd_probe, "timing": cmd_timing, "smoke": cmd_smoke, "imagecheck": cmd_imagecheck}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
