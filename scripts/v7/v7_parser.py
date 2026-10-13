"""V7 prompt-only element parser (V7 package docs/03 §7; templates/PROMPT_ONLY_ELEMENTS.txt, sha256 recorded): Qwen2.5-VL-7B-Instruct
@cc594898 in TEXT-ONLY mode, the package's system template verbatim, greedy decoding (no sampling), bf16.  Every prompt of an external set is
parsed once and cached BEFORE any image score exists; every method shares the cache.  Fixed post-processing (decided before any scoring):
  span kept      -> in bounds (0 <= s < e <= len(prompt)), word-aligned (no cut inside a word) and sharing a word with the element text
  span re-derived-> otherwise, the element 'text' located with the deterministic EvalMuse span rules (v7_data.find_spans)
  element dropped-> neither works (counted); polarity outside {positive, negative} -> 'positive' with a flag
  decomposition_failed -> no parseable JSON object or no valid element: the deployment score falls back to O for every method (flag kept)
    run --prompts F [F ...] --out CACHE [--batch 16] [--limit N]     F: JSONL rows {source, id, prompt}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v7_data as VD  # noqa: E402

QWEN_DIR = "/projects/EEG-foundation-model/yinghao/models/Qwen2.5-VL-7B-Instruct"
TEMPLATE_FILE_SHA256 = "see V7 package templates/PROMPT_ONLY_ELEMENTS.txt (recorded at run time when the package copy is present)"
SYSTEM = ("Extract verifiable visual requirements from the supplied image-generation prompt. You see TEXT ONLY, not images, model scores, or human ratings.\n"
          "Return JSON: {\"elements\":[{\"id\":...,\"text\":...,\"spans\":[[start,end]],\"type\":...,\"polarity\":\"positive|negative\",\"depends_on\":[...]}],\"unresolved\":[...]}.\n"
          "Character offsets are zero-based, end-exclusive, and must refer to exact substrings of the original text. Do not invent entities, counts, colors, relations, "
          "or requirements. Keep object–attribute binding and relational participants explicit using multiple spans when required. Negative requirements stay negative. "
          "Avoid duplicated evidence. If a requirement is ambiguous, put it under unresolved rather than guess. Do not use likelihood of typical images to decide whether "
          "a requirement is fulfilled. Output no fulfillment scores.")
MAX_NEW_TOKENS = 1024
PARSER_VERSION = "v7_prompt_only_qwen2.5vl7b_cc594898_greedy_v1"


def first_json_object(text: str):
    """The first balanced {...} block that json-parses (string-aware brace matching); None if there is none."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for k in range(start, len(text)):
            ch = text[k]
            if in_str:
                esc = (ch == "\\") and not esc
                if ch == '"' and not esc:
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:k + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def word_aligned(prompt: str, s: int, e: int) -> bool:
    left_ok = s == 0 or not (prompt[s - 1].isalnum() and prompt[s].isalnum())
    right_ok = e == len(prompt) or not (prompt[e - 1].isalnum() and prompt[e].isalnum())
    return left_ok and right_ok and prompt[s:e].strip() != ""


def shares_word(span_text: str, element_text: str) -> bool:
    """A model span must share at least one word with the element's own text (an empty text accepts any span)."""
    w = set(re.findall(r"[a-z0-9]+", element_text.lower()))
    return not w or bool(set(re.findall(r"[a-z0-9]+", span_text.lower())) & w)


def validate(prompt: str, obj):
    """Parsed model output -> (elements, unresolved, stats).  Elements keep id, text, spans, type, polarity, depends_on, span_source."""
    stats = {"dropped": 0, "rederived": 0, "polarity_fixed": 0}; out = []
    if not isinstance(obj, dict) or not isinstance(obj.get("elements"), list):
        return [], [], {**stats, "no_elements_list": True}
    for k, el in enumerate(obj["elements"]):
        if not isinstance(el, dict):
            stats["dropped"] += 1; continue
        text = str(el.get("text") or "").strip(); spans = el.get("spans") or []; good = []
        if isinstance(spans, list):
            for sp in spans:
                if (isinstance(sp, (list, tuple)) and len(sp) == 2 and all(isinstance(x, int) for x in sp) and 0 <= sp[0] < sp[1] <= len(prompt)
                        and word_aligned(prompt, *sp) and shares_word(prompt[sp[0]:sp[1]], text)):
                    good.append([sp[0], sp[1]])
        source = "model"
        if not good and text:
            found, _rule = VD.find_spans(prompt, text); good = [list(s) for s in found]; source = "rederived" if good else "none"
            stats["rederived"] += bool(good)
        if not good:
            stats["dropped"] += 1; continue
        pol = str(el.get("polarity") or "positive").lower()
        if pol not in ("positive", "negative"):
            pol = "positive"; stats["polarity_fixed"] += 1
        out.append({"id": str(el.get("id", k)), "text": text, "spans": good, "type": el.get("type"), "polarity": pol,
                    "depends_on": el.get("depends_on") if isinstance(el.get("depends_on"), list) else [], "span_source": source})
    unresolved = obj.get("unresolved") if isinstance(obj.get("unresolved"), list) else []
    return out, unresolved, stats


def cmd_run(a):
    import torch
    import transformers
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    rows = []
    for f in a.prompts:
        rows += [json.loads(l) for l in open(f) if l.strip()]
    rows = rows[: a.limit] if a.limit else rows; out = Path(a.out); done = {}
    if out.exists():
        for l in open(out):
            r = json.loads(l)
            if "prompt" in r:
                done[(r["source"], r["id"])] = r
    todo = [r for r in rows if (r["source"], str(r["id"])) not in done]
    proc = AutoProcessor.from_pretrained(QWEN_DIR); proc.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(QWEN_DIR, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    pkg_tpl = Path("/home/infres/yinwang/CS_QMI/outputs/V7/package_ref/templates/PROMPT_ONLY_ELEMENTS.txt")
    head = {"parser_version": PARSER_VERSION, "model_dir": QWEN_DIR, "transformers": transformers.__version__, "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0), "decoding": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS}, "system_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
            "package_template_sha256": hashlib.sha256(pkg_tpl.read_bytes()).hexdigest() if pkg_tpl.exists() else None, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    t0 = time.time()
    with open(out, "a") as fh:
        if not done:
            fh.write(json.dumps({"header": head}) + "\n")
        for s in range(0, len(todo), a.batch):
            chunk = todo[s:s + a.batch]
            texts = [proc.apply_chat_template([{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                                               {"role": "user", "content": [{"type": "text", "text": r["prompt"]}]}], tokenize=False, add_generation_prompt=True) for r in chunk]
            enc = proc(text=texts, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, do_sample=False, max_new_tokens=MAX_NEW_TOKENS)
            outs = proc.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for r, raw in zip(chunk, outs):
                obj = first_json_object(raw); els, unres, st = validate(r["prompt"], obj)
                fh.write(json.dumps({"source": r["source"], "id": str(r["id"]), "prompt": r["prompt"], "prompt_sha256": hashlib.sha256(r["prompt"].encode()).hexdigest(),
                                     "raw_output": raw, "json_ok": obj is not None, "elements": els, "unresolved": unres, "stats": st,
                                     "status": "ok" if els else "decomposition_failed", "parser_version": PARSER_VERSION}, ensure_ascii=False) + "\n")
            fh.flush(); print(f"[v7_parser] {s + len(chunk)}/{len(todo)} prompts ({time.time() - t0:.0f}s)", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--prompts", nargs="+", required=True); r.add_argument("--out", required=True)
    r.add_argument("--batch", type=int, default=16); r.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    return cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
