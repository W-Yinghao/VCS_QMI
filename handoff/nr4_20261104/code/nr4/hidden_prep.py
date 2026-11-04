#!/usr/bin/env python3
"""NR4 frozen first-10-block hidden-state cache (GPU, FIT split only). For every FIT image/text (cache order) compute the output
of SigLIP2 embeddings + blocks 0-9 (frozen; LoRA lives only in blocks 10-11) at the NR1 fixed forward shape B=32, fp32,
TF32 off; store fp32 [N,196,768] (images) and [N,64,768] (texts). Independent of other items (sample-wise ops, fixed shape).
Verification before trusting the cache: for the first 256 items, cached-prefix + current tail (LoRA B=0) == NR1 global cache
(bitwise expected on the extraction GPU type), and == the raw full path."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np, torch
from nr1.io import FeatureBank
from nr4.tail_encoder import load_siglip, TailEncoder, vision_prefix, text_prefix, _pad
from nr4.raw_prep import SRC
RAW = Path('/projects/EEG-foundation-model/yinghao/outputs_nr4/NR4_20261104/raw')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dataset', required=True); a = ap.parse_args()
    od = RAW/a.dataset/'fit'; man = json.loads((od/'RAW_MANIFEST.json').read_text())
    if (od/'HIDDEN_MANIFEST.json').exists(): print('exists'); return
    pix = np.load(od/'pixels_u8.npy', mmap_mode='r'); tok = np.load(od/'tokens_i32.npy')
    m, _ = load_siglip(lora=True); enc = TailEncoder(m, pix, tok)
    t0 = time.perf_counter(); res = dict(dataset=a.dataset, raw_manifest_sha256=sha(od/'RAW_MANIFEST.json'))
    for kind, n, L, name in (('image', len(pix), 196, 'img_h10_f32.npy'), ('text', len(tok), 64, 'txt_h10_f32.npy')):
        out = np.lib.format.open_memmap(od/name, mode='w+', dtype=np.float32, shape=(n, L, 768)); t = time.perf_counter()
        with torch.no_grad():
            for s in range(0, n, 32):
                r = np.arange(s, min(s+32, n))
                if kind == 'image': x, k = _pad(enc.pixel_values(r)); h = vision_prefix(m, x)
                else: x, k = _pad(torch.from_numpy(np.ascontiguousarray(tok[r])).long().cuda()); h = text_prefix(m, x)
                out[s:s+k] = h[:k].cpu().numpy()
        out.flush(); del out; res[f'{kind}_seconds'] = time.perf_counter() - t; res[f'{kind}_shape'] = [n, L, 768]
    # verification vs NR1 global cache and vs raw full path (first 256 items)
    bank = FeatureBank(SRC[a.dataset]['cache']/'fit', 'fit', verify_hashes=False)
    hi, ht = np.load(od/'img_h10_f32.npy', mmap_mode='r'), np.load(od/'txt_h10_f32.npy', mmap_mode='r')
    ver = {}
    with torch.no_grad():
        for kind, H, C in (('image', hi, bank.x), ('text', ht, bank.y)):
            r = np.arange(256); fz = []
            for s in range(0, 256, 32):
                h, k = _pad(torch.from_numpy(np.ascontiguousarray(H[r[s:s+32]])).cuda()); fz.append(enc.tail(kind, h, k))
            fz = torch.cat(fz).cpu().numpy(); full = enc.encode(kind, r)[0].cpu().numpy()
            ver[kind] = dict(cached_vs_nr1_global_max_abs=float(np.abs(fz - np.asarray(C[:256])).max()), cached_vs_raw_full_max_abs=float(np.abs(fz - full).max()))
    res['verification_first256'] = ver; res['seconds'] = time.perf_counter() - t0; res['gpu'] = torch.cuda.get_device_name()
    res['img_h10_sha256'] = sha(od/'img_h10_f32.npy'); res['txt_h10_sha256'] = sha(od/'txt_h10_f32.npy')
    res['bytes'] = (od/'img_h10_f32.npy').stat().st_size + (od/'txt_h10_f32.npy').stat().st_size
    res['definition'] = 'output of embeddings + encoder.layers[0:10] (frozen, no LoRA), fixed forward B=32 (repeat-last padding, padded rows discarded), fp32, TF32 off, eager attention'
    ok = all(v['cached_vs_nr1_global_max_abs'] == 0.0 and v['cached_vs_raw_full_max_abs'] == 0.0 for v in ver.values())
    res['status'] = 'PASS' if ok else 'NOT_BITWISE'
    (od/'HIDDEN_MANIFEST.json').write_text(json.dumps(res, indent=1)); print(json.dumps(res))


if __name__ == '__main__':
    main()
