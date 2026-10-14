"""V7X-30 secondary table inputs: EvalMuse DEV (the V7 prompt-disjoint holdout of public TRAIN) in the GenAI-Rank input format, label-free.
Writes evalmuse_dev_inputs.jsonl (prompt_id, generator_id, image_id, candidate_index = order within the prompt, prompt), evalmuse_dev_prompts.jsonl
and evalmuse_dev_elements.jsonl (element_id, image_id, prompt_id, text = the annotated element name 'name (type)'; main scope; NO votes / y), and
stages the DEV images as files.  The external scorers read only these files; DEV labels are read only by the V7X-40 scorer.
    python -I scripts/v7x/x30_dev_inputs.py --out-dir D --images DIR"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v7"))
import v7_common as VC  # noqa: E402
import v7_data as VD  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-dir", default=str(VC.DATA_OUT)); ap.add_argument("--images", required=True); a = ap.parse_args()
    out, img = Path(a.out_dir), Path(a.images); img.mkdir(parents=True, exist_ok=True)
    dev = [i for i in VC.read_jsonl(VC.DATA_OUT / "manifest_images.jsonl") if i["role"] == "DEV"]; per = defaultdict(list)
    for i in sorted(dev, key=lambda r: (r["prompt_id"], r["image_id"])):
        per[i["prompt_id"]].append(i)
    with open(out / "evalmuse_dev_inputs.jsonl", "w") as f, open(out / "evalmuse_dev_prompts.jsonl", "w") as g:
        for pid, rows in per.items():
            g.write(json.dumps({"prompt_id": pid, "prompt": rows[0]["prompt"]}) + "\n")
            for k, r in enumerate(rows):
                f.write(json.dumps({"prompt_id": pid, "generator_id": r["generator_id"], "image_id": r["image_id"], "candidate_index": k, "prompt": r["prompt"]}) + "\n")
    ids = {i["image_id"]: i["prompt_id"] for i in dev}; n_el = 0
    with open(out / "evalmuse_dev_elements.jsonl", "w") as f:
        for e in VC.read_jsonl(VC.DATA_OUT / "manifest_elements.jsonl"):
            if e["image_id"] in ids and e["scope"] == "main":
                f.write(json.dumps({"element_id": e["element_id"], "image_id": e["image_id"], "prompt_id": ids[e["image_id"]],
                                    "text": e["raw_key"], "type": e["type"]}) + "\n"); n_el += 1
    z = VD.open_archive(); n = 0
    for i in dev:
        p = img / i["image_id"]
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(z.read("dataset/images/" + i["image_id"])); n += 1
    print(json.dumps({"images": len(dev), "prompts": len(per), "elements_main": n_el, "staged_now": n}))


if __name__ == "__main__":
    main()
