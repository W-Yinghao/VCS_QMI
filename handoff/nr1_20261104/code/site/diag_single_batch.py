#!/usr/bin/env python3
"""Diagnose the package extractor's single-vs-batch check: per-tensor max diffs under precision settings."""
import json, sys, torch
from pathlib import Path
from torch.nn import functional as F
from PIL import Image, ImageOps
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from nr1.io import jsonl
from transformers import AutoModel, AutoProcessor
rev = sys.argv[1]; data = Path(sys.argv[2]); out = sys.argv[3]
ims = jsonl(data/'fit_images.jsonl')[:4]; txt = jsonl(data/'fit_texts.jsonl')[:4]
model = AutoModel.from_pretrained('google/siglip2-base-patch16-224', revision=rev, local_files_only=True, attn_implementation='eager', dtype=torch.float32).cuda().eval()
proc = AutoProcessor.from_pretrained('google/siglip2-base-patch16-224', revision=rev, local_files_only=True)
pil = []
for r in ims:
    with Image.open(r['path']) as im: pil.append(ImageOps.exif_transpose(im).convert('RGB').copy())
inp = proc(text=[r['text'].lower() for r in txt], images=pil, padding='max_length', max_length=64, truncation=True, return_tensors='pt')
inp = {k: v.cuda() for k, v in inp.items()}
res = {}
for name, conv_tf32, mm_tf32 in [('default', torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32), ('no_tf32', False, False), ('all_tf32', True, True)]:
    torch.backends.cudnn.allow_tf32 = conv_tf32; torch.backends.cuda.matmul.allow_tf32 = mm_tf32
    with torch.inference_mode():
        vi = model.vision_model(pixel_values=inp['pixel_values'], return_dict=True); v1 = model.vision_model(pixel_values=inp['pixel_values'][:1], return_dict=True)
        te = model.text_model(input_ids=inp['input_ids'], return_dict=True); t1 = model.text_model(input_ids=inp['input_ids'][:1], return_dict=True)
    d = {}
    for k, (a, b) in dict(img_pool=(v1.pooler_output, vi.pooler_output[:1]), txt_pool=(t1.pooler_output, te.pooler_output[:1]),
                          img_hidden=(v1.last_hidden_state, vi.last_hidden_state[:1]), txt_hidden=(t1.last_hidden_state, te.last_hidden_state[:1])).items():
        diff = (a-b).abs(); d[k] = dict(max_abs=float(diff.max()), max_ref=float(b.abs().max()),
                                       max_rel_excess=float((diff - 3e-5*b.abs()).max()), allclose_3e5=bool(torch.allclose(a, b, atol=3e-5, rtol=3e-5)))
    res[name] = dict(cudnn_tf32=conv_tf32, matmul_tf32=mm_tf32, diffs=d)
    print(name, json.dumps(res[name]), flush=True)
res['gpu'] = torch.cuda.get_device_name(); res['matmul_precision'] = torch.get_float32_matmul_precision()
json.dump(res, open(out, 'w'), indent=1)
