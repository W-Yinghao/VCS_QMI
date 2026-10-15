"""V8-05 §B — export the frozen V8 formal units on the 800 TIFA160 images (secondary external evaluation, owner-approved 2026-10-15).

Same model build, checkpoint, processor and `predict` as `v8_train.py export` (imported, not copied), but the external tag is `tifa160` so the
frozen `pred/genai_bench.jsonl` files are never touched.  Writes <unit>/pred/tifa160.jsonl (prompt_id, generator_id, image_id, candidate_index,
score = mu, q, u) and <unit>/pred/tifa160_export.json; refuses to overwrite.  No label is read (the inputs carry prompts and image paths only).
    python -I scripts/v8/v8_export_tifa160.py --units U [U ...] --inputs F --images DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "v7")); sys.path.insert(0, str(HERE.parent / "v7x"))
import v8_train as T  # noqa: E402
import v7_common as VC  # noqa: E402


def export_unit(unit: Path, cands: list[dict], images, tok, proc, device) -> dict:
    import torch
    log = json.loads((unit / "log.json").read_text()); prior = log["prior_fit"]
    model, _ = T.build(log["seed"], device, log["fit_hist"]); ck = torch.load(unit / "best.pt", map_location="cpu", weights_only=False)
    msg = model.load_state_dict(ck["model"], strict=False)
    if [k for k in msg.missing_keys if not k.startswith("visual_encoder.")] or msg.unexpected_keys:
        raise SystemExit(f"{unit.name}: selected checkpoint does not match the model: {msg}")
    rows = T.CandRows(cands, tok); pred = T.predict(model, rows, images, proc, device, prior); by = {c["image_id"]: c for c in cands}
    out = unit / "pred"; out.mkdir(exist_ok=True); f = out / "tifa160.jsonl"
    if f.exists():
        raise SystemExit(f"{f} exists (refusing to overwrite)")
    f.write_text("".join(json.dumps({"prompt_id": by[p["image_id"]]["prompt_id"], "generator_id": by[p["image_id"]]["generator_id"], "image_id": p["image_id"],
                                     "candidate_index": by[p["image_id"]]["candidate_index"], "score": p["mu"], "q": p["q"], "u": p["u"]}) + "\n" for p in pred))
    meta = {"unit": unit.name, "selected_epoch": ck["epoch"], "prior": prior, "n": len(pred), "inputs": str(cands_path), "tag": "tifa160"}
    (out / "tifa160_export.json").write_text(json.dumps(meta, indent=1)); del model; torch.cuda.empty_cache(); return meta


def main(argv=None):
    global cands_path
    ap = argparse.ArgumentParser(); ap.add_argument("--units", nargs="+", required=True); ap.add_argument("--inputs", required=True); ap.add_argument("--images", required=True)
    a = ap.parse_args(argv); cands_path = Path(a.inputs)
    import torch
    from transformers import BertTokenizerFast
    from x30_m_bench import FileImages
    device = "cuda" if torch.cuda.is_available() else "cpu"; VC.stage_workdir(VC.BIG_OUT / "work"); tok = BertTokenizerFast.from_pretrained(str(VC.BERT_DIR))
    VC.import_lavis()
    from lavis.processors.blip_processors import BlipImageEvalProcessor
    proc = BlipImageEvalProcessor(image_size=364); cands = VC.read_jsonl(a.inputs); images = FileImages(a.images)
    if len(cands) != 800 or len({c["image_id"] for c in cands}) != 800:
        raise SystemExit(f"expected 800 unique TIFA160 candidates, got {len(cands)}")
    rc = 0
    for u in a.units:
        unit = Path(u)
        if (unit / "pred" / "tifa160.jsonl").exists():
            print(f"[skip] {unit.name}: tifa160.jsonl exists", flush=True); continue
        if not (unit / "done.json").exists():
            print(f"[refuse] {unit.name}: no done.json", flush=True); rc = 1; continue
        meta = export_unit(unit, cands, images, tok, proc, device); print(f"[export] {unit.name}: {meta['n']} rows (epoch {meta['selected_epoch']})", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
