#!/usr/bin/env python3
import json, time, os
import numpy as np, torch
from nr4.tail_encoder import load_siglip, vision_prefix, vision_tail, text_prefix, text_tail, set_precision
res = {}; dev = 'cuda'
m, _ = load_siglip(lora=True)
def T(f, reps=3):
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(reps): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / reps
pv = torch.randn(256, 3, 224, 224, device=dev); ids = torch.randint(0, 250000, (256, 64), device=dev)
for Bs in (32, 128, 256):
    with torch.no_grad():
        res[f'img_full_B{Bs}_per_item_ms'] = 1000 * T(lambda: vision_tail(m, vision_prefix(m, pv[:Bs]))) / Bs
        res[f'txt_full_B{Bs}_per_item_ms'] = 1000 * T(lambda: text_tail(m, text_prefix(m, ids[:Bs]))) / Bs
        h = vision_prefix(m, pv[:Bs]); ht = text_prefix(m, ids[:Bs])
        res[f'img_tail_B{Bs}_per_item_ms'] = 1000 * T(lambda: vision_tail(m, h)) / Bs
        res[f'txt_tail_B{Bs}_per_item_ms'] = 1000 * T(lambda: text_tail(m, ht)) / Bs
    def fb():
        z = vision_tail(m, h); z.sum().backward()
    def fbt():
        z = text_tail(m, ht); z.sum().backward()
    res[f'img_tail_fwdbwd_B{Bs}_per_item_ms'] = 1000 * T(fb) / Bs
    res[f'txt_tail_fwdbwd_B{Bs}_per_item_ms'] = 1000 * T(fbt) / Bs
with torch.no_grad():
    a = vision_tail(m, vision_prefix(m, pv[:256])); b = torch.cat([vision_tail(m, vision_prefix(m, pv[s:s+32])) for s in range(0, 256, 32)])
    res['img_B256_vs_B32_max_abs'] = float((a - b).abs().max())
# NFS random-read speed of a 4 GB memmap of 196x768 fp32 rows
path = '/projects/EEG-foundation-model/yinghao/outputs_nr4/NR4_20261104/_probe_read.npy'
if not os.path.exists(path):
    mm = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(6800, 196, 768)); mm[:] = 1; mm.flush(); del mm
mm = np.load(path, mmap_mode='r'); idx = np.sort(np.random.default_rng(0).integers(0, 6800, 2080))
os.system(f'dd if={path} of=/dev/null bs=1M count=1 iflag=direct 2>/dev/null')
t = time.perf_counter(); x = np.ascontiguousarray(mm[idx]); res['memmap_read_2080_rows_s_first'] = time.perf_counter() - t
t = time.perf_counter(); x = np.ascontiguousarray(mm[idx]); res['memmap_read_2080_rows_s_second'] = time.perf_counter() - t
os.remove(path)
res['gpu'] = torch.cuda.get_device_name(); print(json.dumps(res, indent=1))
