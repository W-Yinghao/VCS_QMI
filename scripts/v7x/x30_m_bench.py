"""M arms (formal units) on an external candidate set read from image FILES (GenAI-Bench), with the same readouts as v7_train.export-rank (O, the
parser elements' logits, E = mean prior-corrected probability; decomposition_failed flagged).  The frozen export-rank writes a fixed genai_rank.jsonl
and reads the Rank zips, so this exporter reuses its functions (build_model, forward, Batcher, pool_logits) and writes pred/<tag>.jsonl.
    export --units U [U ...] --inputs I --images DIR --tag T [--parser-cache C]"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v7"))
import v7_common as VC  # noqa: E402
import v7_train as VT  # noqa: E402


class FileImages:
    def __init__(self, root):
        self.root = Path(root)

    def __call__(self, image_id):
        from PIL import Image
        return Image.open(self.root / image_id).convert("RGB")


def export(unit: Path, cands, cache, images, tag, device, tok, proc):
    import torch
    log = json.loads((unit / "log.json").read_text()); prior = log["prior_fit_spanned"]; L = log["model"]["max_txt_len"]
    model, _ = VT.build_model(L, log["seed"], device); ck = torch.load(unit / "best.pt", map_location="cpu", weights_only=False); model.load_state_dict(ck["model"], strict=False)
    rows = []
    for c in cands:
        pr = cache.get(c["prompt"]); enc = tok(c["prompt"], return_offsets_mapping=True); ids = enc["input_ids"]
        if len(ids) > L:
            raise SystemExit(f"prompt longer than max_txt_len: {c['prompt_id']}")
        els = [] if pr is None or pr["status"] != "ok" else [{"element_id": f"{c['prompt_id']}::{e['id']}", "y": 0.0, "tok": VC.token_spans(enc["offset_mapping"], e["spans"])[:VT.MAX_SPAN_TOKENS]} for e in pr["elements"]]
        rows.append({"image_id": c["image_id"], "prompt_id": c["prompt_id"], "prompt": c["prompt"], "ids": ids, "overall": 0.0, "w": 1.0, "elements": [e for e in els if e["tok"]],
                     "generator_id": c["generator_id"], "candidate_index": c["candidate_index"], "parser_status": "missing" if pr is None else pr["status"]})
    data = VT.ElementData.__new__(VT.ElementData); data.rows = rows; bt = VT.Batcher(data, images, proc); out = []; model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device == "cuda"):
        for s in range(0, len(rows), 32):
            idx = list(range(s, min(s + 32, len(rows)))); b = bt(idx); O, z_tok, _ = VT.forward(model, b, device)
            z = VC.pool_logits(z_tok, b["owner"].to(device), b["tok_idx"].to(device), b["tok_mask"].to(device)).float().cpu().numpy() if len(b["owner"]) else np.zeros(0)
            per = defaultdict(list)
            for k in range(len(z)):
                per[int(b["owner"][k])].append(float(z[k]))
            for j, ridx in enumerate(idx):
                r = rows[ridx]; zs = per.get(j, []); ps = VC.natural_probability(zs, prior).tolist() if zs else []
                out.append({"prompt_id": r["prompt_id"], "generator_id": r["generator_id"], "image_id": r["image_id"], "candidate_index": r["candidate_index"], "O": float(O[j]),
                            "E": (float(np.mean(ps)) if ps else None), "n_elements": len(zs), "decomposition_failed": not ps, "parser_status": r["parser_status"]})
    dst = unit / "pred" / f"{tag}.jsonl"
    if dst.exists():
        raise SystemExit(f"{dst} exists")
    with open(dst, "w") as fh:
        [fh.write(json.dumps(o) + "\n") for o in out]
    print(f"[m_bench] {unit.name}: {len(out)} candidates -> {dst.name}", flush=True)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    x = sub.add_parser("export"); x.add_argument("--units", nargs="+", required=True); x.add_argument("--inputs", required=True); x.add_argument("--images", required=True)
    x.add_argument("--tag", required=True); x.add_argument("--parser-cache", default=str(VC.BIG_OUT / "parser" / "genai_prompts_cache_v3.jsonl")); a = ap.parse_args()
    import torch
    from transformers import BertTokenizerFast
    device = "cuda" if torch.cuda.is_available() else "cpu"; VC.stage_workdir(VC.BIG_OUT / "work"); tok = BertTokenizerFast.from_pretrained(str(VC.BERT_DIR))
    VC.import_lavis()
    from lavis.processors.blip_processors import BlipImageEvalProcessor
    proc = BlipImageEvalProcessor(image_size=364); cands = VC.read_jsonl(a.inputs); cache = {r["prompt"]: r for r in VC.read_jsonl(a.parser_cache) if "prompt" in r}
    for u in a.units:
        export(Path(u), cands, cache, FileImages(a.images), a.tag, device, tok, proc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
