"""V7X-20: common element features from the M arms' initialisation (no EvalMuse update) — reports/V7/V7X_20_COMMON_FEATURE_PROTOCOL.yaml.
The image-grounded Q-Former pass of scripts/v7/v7_train.forward (same init, processor, tokenizer, max_txt_len, span tokens); the last hidden
state at the text positions is averaged over each element's span tokens (first 64, as pool_logits) -> 768-d float32.  Invariant per element:
the itm head applied to the stored feature reproduces the pooled token logit of the forward (the head is linear; fp16 forward vs fp32 recompute).
    extract --out DIR [--roles FIT CAL_SELECT CAL_PROB DEV] [--limit N]       EvalMuse local roles
    extract-rank --out DIR [--limit N]                                         GenAI-Rank candidates x the parser's v3 elements (no human score)
The v1 modules are imported read-only (scripts/v7 is not modified)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

V7 = Path(__file__).resolve().parents[1] / "v7"
sys.path.insert(0, str(V7))
import v7_common as VC  # noqa: E402
import v7_train as VT  # noqa: E402

TOL_ABS, TOL_REL = 2e-2, 2e-3            # |z0_forward - z0_feature| <= TOL_ABS + TOL_REL |z0|: fp16 head output vs fp32 recompute
LABELLED = {"FIT", "CAL_SELECT", "CAL_PROB"}  # y is written only for these roles; DEV / external y are never stored here


def forward_feats(model, b, device):
    """-> pooled features [E, 768] fp32, z0 from the forward's own pooled token logits [E] fp32, z0 recomputed from the features [E] fp32, O0 [B]."""
    import torch
    image = b["image"].to(device, non_blocking=True); ids, att = b["ids"].to(device), b["att"].to(device)
    image_embeds = model.ln_vision(model.visual_encoder(image)); image_atts = torch.ones(image_embeds.shape[:-1], dtype=torch.long, device=device)
    q = model.query_tokens.expand(image_embeds.shape[0], -1, -1); q_att = torch.ones(q.shape[:-1], dtype=torch.long, device=device)
    out = model.Qformer.bert(ids, query_embeds=q, attention_mask=torch.cat([q_att, att], 1), encoder_hidden_states=image_embeds,
                             encoder_attention_mask=image_atts, return_dict=True)
    hs = out.last_hidden_state; logit = model.itm_head(hs).float(); nq = q.shape[1]
    O0 = torch.softmax(logit[:, :nq], -1)[..., 1].mean(1) * 4 + 1; z_tok = logit[:, nq:, 1] - logit[:, nq:, 0]
    owner, tok_idx, tok_mask = b["owner"].to(device), b["tok_idx"].to(device), b["tok_mask"].to(device)
    if len(owner) == 0:
        e = torch.zeros(0, device=device)
        return torch.zeros((0, hs.shape[-1]), device=device), e, e, O0
    z0 = VC.pool_logits(z_tok, owner, tok_idx, tok_mask)
    h_text = hs[:, nq:, :].float(); g = h_text[owner[:, None], tok_idx]; m = tok_mask.to(g.dtype)[..., None]
    feat = (g * m).sum(1) / m.sum(1).clamp(min=1)
    W = model.itm_head.weight.float(); bias = model.itm_head.bias.float()
    z0_feat = feat @ (W[1] - W[0]) + (bias[1] - bias[0])
    return feat, z0, z0_feat, O0


def run(model, data, images, proc, device, out_dir: Path, tag: str, extra_cols, workers: int, keep_y: bool):
    import torch
    from torch.utils.data import DataLoader
    bt = VT.Batcher(data, images, proc); loader = DataLoader(range(len(data)), batch_size=32, shuffle=False, collate_fn=bt, num_workers=workers,
                                                          prefetch_factor=4 if workers else None)
    feats, idx_rows, img_rows = [], [], []; worst = 0.0; bad = 0; t0 = time.time(); start = 0
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device == "cuda"):
        for b in loader:
            f, z0, z0f, O0 = forward_feats(model, b, device); f, z0, z0f = f.cpu().numpy(), z0.float().cpu().numpy(), z0f.cpu().numpy()
            diff = np.abs(z0 - z0f); lim = TOL_ABS + TOL_REL * np.abs(z0); worst = max(worst, float(diff.max()) if len(diff) else 0.0); bad += int((diff > lim).sum())
            feats.append(f.astype(np.float32)); n_per = {}
            for k in range(len(b["element_id"])):
                r = int(b["owner"][k]); n_per[r] = n_per.get(r, 0) + 1
            for k, eid in enumerate(b["element_id"]):
                r = int(b["owner"][k]); row = data.rows[start + r]
                rec = {"element_id": eid, "image_id": row["image_id"], "prompt_id": row["prompt_id"], "w_img": row["w"], "n_el": n_per[r],
                       "n_tok": int(b["tok_mask"][k].sum()), "z0": float(z0[k])}
                if keep_y:
                    rec["y"] = float(b["y"][k])
                for c in extra_cols:
                    rec[c] = row[c]
                idx_rows.append(rec)
            for r in range(len(b["image_id"])):
                row = data.rows[start + r]; rec = {"image_id": row["image_id"], "prompt_id": row["prompt_id"], "O0": float(O0[r]), "n_el": n_per.get(r, 0)}
                for c in extra_cols:
                    rec[c] = row[c]
                img_rows.append(rec)
            start += len(b["image_id"])
    F = np.concatenate(feats) if feats else np.zeros((0, 768), np.float32)
    if len(F) != len(idx_rows):
        raise SystemExit(f"{tag}: {len(F)} features for {len(idx_rows)} index rows")
    np.save(out_dir / f"feats_{tag}.npy", F)
    with open(out_dir / f"index_{tag}.jsonl", "w") as fh:
        for r in idx_rows:
            fh.write(json.dumps(r) + "\n")
    with open(out_dir / f"images_{tag}.jsonl", "w") as fh:
        for r in img_rows:
            fh.write(json.dumps(r) + "\n")
    h = hashlib.sha256(); h.update(tag.encode()); [h.update(r["element_id"].encode()) for r in idx_rows]; h.update(F.tobytes())
    rec = {"tag": tag, "n_images": len(img_rows), "n_elements": len(idx_rows), "dim": int(F.shape[1]), "sha256": h.hexdigest(),
           "z0_invariant": {"max_abs_diff": worst, "violations": bad, "tol": f"{TOL_ABS} + {TOL_REL}|z0|"}, "seconds": time.time() - t0, "y_stored": keep_y}
    print(f"[x20] {tag}: {rec['n_images']} images, {rec['n_elements']} elements, z0 max diff {worst:.2e} (violations {bad}), {rec['seconds']:.0f}s", flush=True)
    return rec


def setup(a):
    import torch
    from transformers import BertTokenizerFast
    device = "cuda" if torch.cuda.is_available() else "cpu"; VC.stage_workdir(VC.BIG_OUT / "work")
    tok = BertTokenizerFast.from_pretrained(str(VC.BERT_DIR)); model, info = VT.build_model(128, 0, device)
    VC.import_lavis()
    from lavis.processors.blip_processors import BlipImageEvalProcessor
    return device, tok, model, info, BlipImageEvalProcessor(image_size=364)


def finish(out, info, recs, a, device):
    import torch
    code = hashlib.sha256(Path(__file__).read_bytes() + (V7 / "v7_train.py").read_bytes() + (V7 / "v7_common.py").read_bytes()).hexdigest()[:16]
    man = {"protocol": "reports/V7/V7X_20_COMMON_FEATURE_PROTOCOL.yaml", "model": info, "max_txt_len": 128, "image_size": 364, "pooling": "mean of span tokens (<= 64)",
           "pass": "image-grounded Q-Former (v7_train.forward), last hidden state at text positions", "code_sha256_16": code, "limit": a.limit,
           "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu", "parts": recs}
    p = out / ("manifest_rank.json" if a.cmd == "extract-rank" else "manifest.json"); p.write_text(json.dumps(man, indent=1))
    viol = sum(r["z0_invariant"]["violations"] for r in recs)
    print(f"[x20] wrote {p}; invariant violations {viol}", flush=True)
    return 0 if viol == 0 else 5


def cmd_extract(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); device, tok, model, info, proc = setup(a)
    imgs, by_img = VT.load_roles(Path(a.data_dir)); images = VC.ArchiveImages(); recs = []
    for role in a.roles:
        data = VT.ElementData(imgs, by_img, role, tok, 128, limit=a.limit)
        recs.append(run(model, data, images, proc, device, out, role, [], a.workers, keep_y=role in LABELLED))
    return finish(out, info, recs, a, device)


def cmd_extract_rank(a):
    """Rows exactly as v7_train.cmd_export_rank builds them (prompt tokens, parser v3 elements, spans -> tokens); no human score is read."""
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); device, tok, model, info, proc = setup(a)
    cands = VC.read_jsonl(a.inputs); cache = {r["prompt"]: r for r in VC.read_jsonl(a.parser_cache) if "prompt" in r}
    if a.limit:
        cands = cands[:a.limit]
    rows = []
    for c in cands:
        pr = cache.get(c["prompt"]); enc = tok(c["prompt"], return_offsets_mapping=True); ids = enc["input_ids"]
        if len(ids) > 128:
            raise SystemExit(f"external prompt longer than max_txt_len: {c['prompt_id']}")
        els = [] if pr is None or pr["status"] != "ok" else [{"element_id": f"{c['prompt_id']}::{e['id']}", "y": 0.0,
                                                             "tok": VC.token_spans(enc["offset_mapping"], e["spans"])[:VT.MAX_SPAN_TOKENS]} for e in pr["elements"]]
        els = [e for e in els if e["tok"]]
        rows.append({"image_id": c["image_id"], "prompt_id": c["prompt_id"], "prompt": c["prompt"], "ids": ids, "overall": 0.0, "w": 1.0, "elements": els,
                     "generator_id": c["generator_id"], "candidate_index": c["candidate_index"], "parser_status": "missing" if pr is None else pr["status"]})
    data = VT.ElementData.__new__(VT.ElementData); data.rows = rows
    rec = run(model, data, VT.RankImages(), proc, device, out, "GENAI_RANK", ["generator_id", "candidate_index", "parser_status"], a.workers, keep_y=False)
    return finish(out, info, [rec], a, device)


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    x = sub.add_parser("extract"); x.add_argument("--out", required=True); x.add_argument("--roles", nargs="+", default=["FIT", "CAL_SELECT", "CAL_PROB", "DEV"])
    x.add_argument("--data-dir", default=str(VC.DATA_OUT)); x.add_argument("--limit", type=int, default=None); x.add_argument("--workers", type=int, default=8)
    r = sub.add_parser("extract-rank"); r.add_argument("--out", required=True); r.add_argument("--inputs", default=str(VC.DATA_OUT / "genai_rank_inputs.jsonl"))
    r.add_argument("--parser-cache", default=str(VC.BIG_OUT / "parser" / "genai_prompts_cache_v3.jsonl")); r.add_argument("--limit", type=int, default=None)
    r.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    return {"extract": cmd_extract, "extract-rank": cmd_extract_rank}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
