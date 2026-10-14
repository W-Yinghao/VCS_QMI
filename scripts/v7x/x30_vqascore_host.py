"""V7X-30 B2: VQAScore with the same modern host as the V7 parser (Qwen2.5-VL-7B-Instruct @ cc594898, local copy) through t2v-metrics 3.0's own
Qwen2VLModel.forward (first-token softmax probability of 'Yes' after the chat generation prompt), with B1's templates ('Does this figure show "{}"?
Please answer yes or no.' / 'Yes') so that only the host changes.  Scoped import as B1 (package __init__ files skipped); registry entry
'qwen2.5-vl-7b' attention 'sdpa' instead of 'flash_attention_2' (flash-attn is not installed; same attention, another kernel); decord (video only)
stubbed if its wheel does not load.  Host-upgrade effect only; not a VQAScore reproduction.
    rank --inputs I --out F [--shard k/n] [--limit N]"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
import time
import types
import zipfile
from pathlib import Path

HOST = Path("/projects/EEG-foundation-model/yinghao/models/Qwen2.5-VL-7B-Instruct"); RANK = Path("/projects/EEG-foundation-model/yinghao/data/genai_rank")
Q_TMPL, A_TMPL = 'Does this figure show "{}"? Please answer yes or no.', "Yes"


def import_qwen():
    try:
        import decord  # noqa: F401
    except Exception:
        d = types.ModuleType("decord"); d.VideoReader = d.cpu = None; sys.modules["decord"] = d
    root = Path(importlib.util.find_spec("t2v_metrics").submodule_search_locations[0])
    for name, sub in (("t2v_metrics", ""), ("t2v_metrics.models", "models"), ("t2v_metrics.models.vqascore_models", "models/vqascore_models")):
        if name not in sys.modules:
            m = types.ModuleType(name); m.__path__ = [str(root / sub) if sub else str(root)]; m.__package__ = name; sys.modules[name] = m
    Q = importlib.import_module("t2v_metrics.models.vqascore_models.qwen2vl_model")
    Q.QWEN2_VL_MODELS["qwen2.5-vl-7b"]["model"]["attn_implementation"] = "sdpa"
    return Q, root


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rank"); r.add_argument("--inputs", required=True); r.add_argument("--out", required=True); r.add_argument("--shard", default="0/1"); r.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    import torch
    Q, root = import_qwen(); k, n = (int(x) for x in a.shard.split("/"))
    rows = [json.loads(l) for i, l in enumerate(open(a.inputs)) if l.strip() and i % n == k][: a.limit or None]
    out = Path(a.out); done = {json.loads(l)["image_id"] for l in open(out) if l.strip() and "image_id" in l} if out.exists() else set(); todo = [r for r in rows if r["image_id"] not in done]
    local = Path(os.environ.get("V7X_LOCAL", "/tmp")) / "v7x_b2_images"; zs = {}; t0 = time.time()
    model = Q.Qwen2VLModel(model_name="qwen2.5-vl-7b", device="cuda", checkpoint=str(HOST))
    with open(out, "a") as fh:
        if not done:
            fh.write(json.dumps({"header": {"method": "B2 VQAScore, Qwen2.5-VL-7B host", "t2v_metrics": "3.0 (" + str(root) + ")", "torch": torch.__version__,
                                            "gpu": torch.cuda.get_device_name(0), "question_template": Q_TMPL, "answer_template": A_TMPL, "attention": "sdpa", "load_s": time.time() - t0}}) + "\n")
        for i, c in enumerate(todo):
            g = c["image_id"].split("/")[0]; z = zs.get(g) or zs.setdefault(g, zipfile.ZipFile(RANK / f"{g}.zip")); p = local / c["image_id"]
            if not p.exists():
                p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(z.read(c["image_id"]))
            v = float(model.forward([str(p)], [c["prompt"]], question_template=Q_TMPL, answer_template=A_TMPL)[0])
            fh.write(json.dumps({"prompt_id": c["prompt_id"], "generator_id": c["generator_id"], "image_id": c["image_id"], "candidate_index": c["candidate_index"], "score": v}) + "\n")
            if i % 500 == 0:
                fh.flush(); print(f"[b2] {i + 1}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
