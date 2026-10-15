#!/usr/bin/env python
"""V8.1 fork 3 — FIT-AUDIT: the frozen in-training audit sample (FULL_PLAN_CN §04 D05, V81_PROTOCOL.yaml fit_audit).

Rule: FIT prompts sorted by sha256('v81-fit-audit-2718|' + prompt_id); prompts are taken in that order with ALL their overall-valid FIT
images until 128 prompts are in, or until the next prompt would push the image count past 2,048 (whole prompts only; a prompt is never
truncated).  The sample is fixed by input ids, never by loss or error.  Reads the V7 public TRAIN manifest only (FIT role); the id file
carries no label value (votes stay in the manifest).

    python -I scripts/v81/v81_fork3_fit_audit.py --out reports/V8_1_DIAGNOSTICS/results/fit_audit_ids.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

MANIFEST = Path("/home/infres/yinwang/CS_QMI/outputs/V7/data/manifest_images.jsonl")
SALT, MAX_PROMPTS, MAX_IMAGES = "v81-fit-audit-2718|", 128, 2048


def prompt_key(prompt_id) -> str:
    return hashlib.sha256((SALT + str(prompt_id)).encode()).hexdigest()


def select(prompt_images: dict, max_prompts: int = MAX_PROMPTS, max_images: int = MAX_IMAGES) -> list:
    """prompt_id -> [image ids]  ->  [(prompt_id, key, [image ids sorted lexically])] in hash order, whole prompts, both caps."""
    out, n = [], 0
    for p in sorted(prompt_images, key=lambda p: (prompt_key(p), str(p))):
        imgs = sorted(prompt_images[p], key=str)
        if len(out) >= max_prompts or n + len(imgs) > max_images:
            break
        out.append((p, prompt_key(p), imgs)); n += len(imgs)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--manifest", default=str(MANIFEST)); a = ap.parse_args(argv)
    man = Path(a.manifest); raw = man.read_bytes(); per = defaultdict(list); n_role = 0; n_invalid = 0
    for line in raw.decode().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("role") != "FIT":
            continue
        n_role += 1
        if not r.get("overall_votes"):
            n_invalid += 1; continue
        per[str(r["prompt_id"])].append(str(r["image_id"]))
    chosen = select(per); image_ids = [i for _, _, imgs in chosen for i in imgs]
    res = {"rule": f"FIT prompts sorted by sha256('{SALT}' + prompt_id); whole prompts in that order until {MAX_PROMPTS} prompts or the next prompt would exceed {MAX_IMAGES} images",
           "manifest": str(man), "manifest_sha256": hashlib.sha256(raw).hexdigest(), "fit_images_in_manifest": n_role, "fit_images_without_valid_vote": n_invalid,
           "fit_prompts_valid": len(per), "n_prompts": len(chosen), "n_images": len(image_ids),
           "stopped_by": "prompt_cap" if len(chosen) >= MAX_PROMPTS else "image_cap",
           "prompts": [{"prompt_id": p, "hash": k, "n_images": len(imgs), "image_ids": imgs} for p, k, imgs in chosen],
           "image_ids": image_ids, "image_ids_sha256": hashlib.sha256(json.dumps(image_ids).encode()).hexdigest(),
           "candidate_order": "lexical image id within the prompt (development only)"}
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        old = json.loads(out.read_text())
        if old.get("image_ids_sha256") != res["image_ids_sha256"]:
            raise SystemExit(f"{out} exists with a different id set ({old.get('image_ids_sha256')}); refusing to overwrite a frozen sample")
        print(f"[fit-audit] unchanged: {res['n_prompts']} prompts, {res['n_images']} images, ids sha256 {res['image_ids_sha256'][:16]}"); return 0
    out.write_text(json.dumps(res, indent=1))
    print(f"[fit-audit] {res['n_prompts']} prompts, {res['n_images']} images (stopped by {res['stopped_by']}), ids sha256 {res['image_ids_sha256'][:16]} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
