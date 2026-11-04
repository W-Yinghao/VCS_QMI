#!/usr/bin/env python3
"""NR4 raw-asset preparation (CPU). For one dataset/split, in the exact item order of the frozen NR1/NR2C/CC feature cache:
  images: sha256 of every raw file checked against the registered content hash (Flickr NR1 audit ID map, COCO NR2C audit ID map,
          CC RAW_CONTENT_HASHES byte_sha256); decoded exactly like nr1.extract_siglip2 (PIL open, exif_transpose, RGB) and resized
          by the locked SiglipImageProcessor with do_rescale/do_normalize off -> uint8 [N,3,224,224] (rescale/normalize are applied
          on GPU per step, verified bitwise vs the full processor);
  texts: locked tokenizer on the lowercased caption, padding max_length 64, truncation -> int32 input_ids [N,64].
Missing/hash-mismatched raw file -> BLOCKED_RAW_ASSET (written, exit 3); never silently replaced."""
import argparse, csv, hashlib, json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

H1 = Path('/projects/EEG-foundation-model/yinghao/home_moved_20261104/outputs_nr1/NR1_20261104')
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_nr4/NR4_20261104/raw')
SRC = dict(flickr=dict(data=H1/'data', cache=H1/'cache', idmap=lambda s: H1/f'audit/ID_MAP_{s}_images.tsv', key='filename'),
           coco=dict(data=Path('/projects/EEG-foundation-model/yinghao/outputs_nr2/NR2C_20261104/data'), cache=Path('/projects/EEG-foundation-model/yinghao/outputs_nr2/NR2C_20261104/cache'),
                     idmap=lambda s: Path(f'/projects/EEG-foundation-model/yinghao/outputs_nr2/NR2C_20261104/audit/ID_MAP_{s}_images.tsv'), key='relpath'),
           cc=dict(data=Path('/projects/EEG-foundation-model/yinghao/data/cc152k_raw/nr1/CC152K_RECOVERED_20261104/data'), cache=Path('/projects/EEG-foundation-model/yinghao/data/cc152k_raw/nr1/CC152K_RECOVERED_20261104/cache'),
                   idmap=None, key='path'))
PROC = None


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()


def init():
    global PROC
    from transformers import AutoProcessor
    PROC = AutoProcessor.from_pretrained('google/siglip2-base-patch16-224', revision='75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2', local_files_only=True)


def work(args):
    i, path, want = args
    from PIL import Image, ImageOps
    if not Path(path).is_file(): return i, 'MISSING', None
    got = sha(path)
    if want is not None and got != want: return i, 'HASH_MISMATCH', None
    with Image.open(path) as im: im = ImageOps.exif_transpose(im).convert('RGB').copy()
    a = PROC.image_processor(images=[im], return_tensors='np', do_rescale=False, do_normalize=False)['pixel_values'][0]
    if a.dtype != np.uint8: a = a.astype(np.uint8)
    return i, 'OK', a


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dataset', required=True); ap.add_argument('--split', required=True); ap.add_argument('--procs', type=int, default=16)
    a = ap.parse_args(); S = SRC[a.dataset]; od = OUT/a.dataset/a.split; od.mkdir(parents=True, exist_ok=True)
    if (od/'RAW_MANIFEST.json').exists(): print('exists'); return
    t0 = time.perf_counter()
    ims = [json.loads(l) for l in open(S['data']/f'{a.split}_images.jsonl')]; txs = [json.loads(l) for l in open(S['data']/f'{a.split}_texts.jsonl')]
    man = json.loads((S['cache']/a.split/'manifest.json').read_text())
    if sha(S['data']/f'{a.split}_images.jsonl') != man['images_manifest_sha256'] or sha(S['data']/f'{a.split}_texts.jsonl') != man['texts_manifest_sha256']:
        sys.exit('jsonl does not match the cache manifest')
    if [r['id'] for r in ims] != json.loads((S['cache']/a.split/'image_ids.json').read_text()) or [r['id'] for r in txs] != json.loads((S['cache']/a.split/'text_ids.json').read_text()):
        sys.exit('item order differs from cache')
    if a.dataset == 'cc':
        want = {r['path']: r['byte_sha256'] for r in csv.DictReader(open('/home/infres/yinwang/CS_QMI/handoff/nr1_20261104/cc152k/RAW_CONTENT_HASHES.csv'))}
        wl = [want.get(r['path']) for r in ims]
    else:
        idm = {r[S['key']]: r['image_sha256'] for r in csv.DictReader(open(S['idmap'](a.split)), delimiter='\t')}
        if a.dataset == 'flickr': wl = [idm.get(Path(r['path']).name) for r in ims]
        else: wl = [idm.get('/'.join(Path(r['path']).parts[-2:])) for r in ims]
    if any(w is None for w in wl): sys.exit('no registered hash for some images')
    pix = np.lib.format.open_memmap(od/'pixels_u8.npy', mode='w+', dtype=np.uint8, shape=(len(ims), 3, 224, 224))
    bad = []
    with Pool(a.procs, initializer=init) as pool:
        for i, st, arr in pool.imap_unordered(work, [(i, r['path'], w) for i, (r, w) in enumerate(zip(ims, wl))], chunksize=64):
            if st != 'OK': bad.append(dict(index=i, id=ims[i]['id'], path=ims[i]['path'], status=st))
            else: pix[i] = arr
    pix.flush(); del pix
    init()
    tok = PROC(text=[r['text'].lower() for r in txs], padding='max_length', max_length=64, truncation=True, return_tensors='np')['input_ids']
    assert tok.shape == (len(txs), 64); np.save(od/'tokens_i32.npy', tok.astype(np.int32))
    rec = dict(dataset=a.dataset, split=a.split, n_images=len(ims), n_texts=len(txs), cache_manifest_sha256=sha(S['cache']/a.split/'manifest.json'),
               images_jsonl_sha256=man['images_manifest_sha256'], texts_jsonl_sha256=man['texts_manifest_sha256'], raw_hash_checked=len(ims) - len(bad),
               failures=bad, pixels_sha256=sha(od/'pixels_u8.npy'), tokens_sha256=sha(od/'tokens_i32.npy'), seconds=time.perf_counter() - t0,
               pixel_definition='SiglipImageProcessor(do_rescale=False, do_normalize=False) after PIL exif_transpose+RGB; uint8 [N,3,224,224]',
               token_definition='tokenizer(lowercased caption, padding=max_length, max_length=64, truncation=True) input_ids int32')
    (od/('RAW_MANIFEST.json' if not bad else 'BLOCKED_RAW_ASSET.json')).write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != 'failures'}), 'failures', len(bad))
    sys.exit(3 if bad else 0)


if __name__ == '__main__':
    main()
