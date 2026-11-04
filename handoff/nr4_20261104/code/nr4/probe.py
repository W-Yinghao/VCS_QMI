#!/usr/bin/env python3
"""Feasibility/equivalence probe (Flickr FIT, 64 images + 64 texts; then a timing pass with a step-sized item set)."""
import json, time, sys
import numpy as np, torch
from PIL import Image, ImageOps
from torch.nn import functional as F
from nr1.io import FeatureBank, jsonl
from nr1.extract_siglip2 import vision, text
from nr4.tail_encoder import load_siglip, TailEncoder, MODEL_ID, REV, set_precision
from transformers import AutoProcessor
O1 = '/projects/EEG-foundation-model/yinghao/home_moved_20261104/outputs_nr1/NR1_20261104'
res = {}
proc = AutoProcessor.from_pretrained(MODEL_ID, revision=REV, local_files_only=True)
ims = jsonl(f'{O1}/data/fit_images.jsonl')[:64]; txs = jsonl(f'{O1}/data/fit_texts.jsonl')[:64]
pil = []
for r in ims:
    with Image.open(r['path'].replace('/home/infres/yinwang/CS_QMI/outputs_nr1', '/projects/EEG-foundation-model/yinghao/home_moved_20261104/outputs_nr1')) as im: pil.append(ImageOps.exif_transpose(im).convert('RGB').copy())
full = proc(images=pil, return_tensors='np')['pixel_values']
raw = proc.image_processor(images=pil, return_tensors='np', do_rescale=False, do_normalize=False)['pixel_values']
res['raw_dtype'] = str(raw.dtype); res['raw_is_integral'] = bool(np.all(raw == np.round(raw))); res['raw_minmax'] = [float(raw.min()), float(raw.max())]
u8 = raw.astype(np.uint8)
tok = proc(text=[r['text'].lower() for r in txs], padding='max_length', max_length=64, truncation=True, return_tensors='np')['input_ids'].astype(np.int32)
bank = FeatureBank(f'{O1}/cache/fit', 'fit', verify_hashes=False)
cx, cy = np.asarray(bank.x[:64]), np.asarray(bank.y[:64])
m0, _ = load_siglip(lora=False)
enc0 = TailEncoder(m0, u8, tok)
pv = enc0.pixel_values(np.arange(64))
res['pixel_values_vs_processor_max_abs'] = float((pv.cpu().numpy() - full).__abs__().max())
with torch.inference_mode():
    a = torch.cat([F.normalize(vision(m0, torch.from_numpy(full[s:s+32]).cuda(), 32)[0].float(), dim=-1) for s in (0, 32)]).cpu().numpy()
    b = torch.cat([F.normalize(text(m0, {'input_ids': torch.from_numpy(tok[s:s+32]).long().cuda()}, 32)[0].float(), dim=-1) for s in (0, 32)]).cpu().numpy()
res['extractor_path_vs_cache'] = dict(img=float(np.abs(a - cx).max()), txt=float(np.abs(b - cy).max()))
m1, info = load_siglip(lora=True)
enc = TailEncoder(m1, u8, tok)
fi, _ = enc.encode('image', np.arange(64)); ft, _ = enc.encode('text', np.arange(64))
res['split_nograd_lora0_vs_cache'] = dict(img=float((fi.cpu().numpy() - cx).__abs__().max()), txt=float((ft.cpu().numpy() - cy).__abs__().max()))
gi, _ = enc.encode('image', np.arange(64), grad=True); gt, _ = enc.encode('text', np.arange(64), grad=True)
res['split_grad_vs_nograd'] = dict(img=float((gi - fi).abs().max()), txt=float((gt - ft).abs().max()))
rows = np.r_[np.arange(40, 64), np.arange(0, 40)]; fr, _ = enc.encode('image', rows)
res['chunk_reorder_invariance_img'] = float((fr.cpu().numpy() - fi.cpu().numpy()[rows]).__abs__().max())
res['lora_info'] = info
# timing: 2080 images + 2080 texts (repeat the 64 items), pass1 no-grad + replay with grad + backward
n = 2080; ri = np.arange(n) % 64; rt = np.arange(n) % 64
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
with torch.no_grad():
    U, preI = enc.encode('image', ri); V, preT = enc.encode('text', rt)
torch.cuda.synchronize(); t1 = time.perf_counter()
gU = torch.randn_like(U); gV = torch.randn_like(V); o = 0
for (h, k) in preI:
    z = enc.tail('image', h, k); torch.autograd.backward(z, gU[o:o+k]); o += k
o = 0
for (h, k) in preT:
    z = enc.tail('text', h, k); torch.autograd.backward(z, gV[o:o+k]); o += k
torch.cuda.synchronize(); t2 = time.perf_counter()
res['timing_step_items'] = dict(n_images=n, n_texts=n, pass1_s=t1 - t0, replay_s=t2 - t1, peak_mem_GiB=torch.cuda.max_memory_allocated() / 2**30, gpu=torch.cuda.get_device_name())
print(json.dumps(res, indent=1, default=str))
