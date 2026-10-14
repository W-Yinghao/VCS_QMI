"""V7X-30 SANITY: CLIPScore (Hessel et al. 2021) with OpenAI CLIP ViT-L/14-336 (local copy): CLIP-S = 2.5 * max(cos(image, text), 0); text
truncated at CLIP's 77 tokens (counted).  A floor row only: it cannot support an 'externally superior' claim.
    rank --inputs I --out F [--image-root DIR]   (GenAI-Rank candidates read from the zips unless --image-root is given)"""
import argparse
import io
import json
import sys
import time
import zipfile
from pathlib import Path

CLIP = Path("/projects/EEG-foundation-model/yinghao/models/clip-vit-large-patch14-336"); RANK = Path("/projects/EEG-foundation-model/yinghao/data/genai_rank")


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rank"); r.add_argument("--inputs", required=True); r.add_argument("--out", required=True); r.add_argument("--image-root", default=None)
    a = ap.parse_args(argv)
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    rows = [json.loads(l) for l in open(a.inputs) if l.strip()]; out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    model = CLIPModel.from_pretrained(str(CLIP), torch_dtype=torch.float16).to("cuda").eval(); proc = CLIPProcessor.from_pretrained(str(CLIP)); zs = {}; t0 = time.time(); trunc = 0
    with open(out, "w") as fh:
        fh.write(json.dumps({"header": {"method": "SANITY CLIPScore ViT-L/14-336", "formula": "2.5 * max(cos, 0)", "gpu": torch.cuda.get_device_name(0)}}) + "\n")
        for s in range(0, len(rows), 64):
            chunk = rows[s:s + 64]; ims = []
            for c in chunk:
                if a.image_root:
                    ims.append(Image.open(Path(a.image_root) / c["image_id"]).convert("RGB"))
                else:
                    g = c["image_id"].split("/")[0]; z = zs.get(g) or zs.setdefault(g, zipfile.ZipFile(RANK / f"{g}.zip")); ims.append(Image.open(io.BytesIO(z.read(c["image_id"]))).convert("RGB"))
            texts = [c["prompt"] for c in chunk]; trunc += sum(len(proc.tokenizer(t).input_ids) > 77 for t in texts)
            inp = proc(text=texts, images=ims, return_tensors="pt", padding=True, truncation=True, max_length=77).to("cuda")
            with torch.no_grad():
                o = model(**inp); ie = o.image_embeds / o.image_embeds.norm(dim=-1, keepdim=True); te = o.text_embeds / o.text_embeds.norm(dim=-1, keepdim=True)
                cs = (ie * te).sum(-1).float().clamp(min=0) * 2.5
            for c, v in zip(chunk, cs.cpu().tolist()):
                fh.write(json.dumps({"prompt_id": c["prompt_id"], "generator_id": c["generator_id"], "image_id": c["image_id"], "candidate_index": c["candidate_index"], "score": v}) + "\n")
        print(f"[clip] {len(rows)} rows, {trunc} prompts truncated at 77 tokens, {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
