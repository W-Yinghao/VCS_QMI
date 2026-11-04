"""CR1 site adapter: locked NR1 SigLIP2 FIT caches + observed tables -> package FIT exchange banks.

Reads ONLY: FIT cache (globals + ids), FIT image/text jsonl (paths/strings), the observed table
of each condition, and the pinned SigLIP2 processor (offline). It never opens audit ID maps,
DEV caches, evaluator_only truth, noise files or clean mappings.

Content keys (package DATA_AND_CROSSFIT B/C):
  image: sha256 over dtype|shape|bytes of the processor pixel_values (float32, [3,224,224]),
         computed with exactly the NR1 extractor input path (exif_transpose -> RGB -> processor,
         same 32-item batches in the same storage order).
  text : sha256 over dtype|shape|bytes of the processor input_ids (lowercased text,
         padding=max_length, max_length=64, truncation) == NR1 extractor token ids.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

REV = '75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2'
MODEL = 'google/siglip2-base-patch16-224'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()


def tensor_key(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + b'|' + str(a.shape).encode() + b'|' + a.tobytes()).hexdigest()


def write_json(path, obj):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp'); tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + '\n'); tmp.replace(p)


def jsonl(path):
    with open(path, encoding='utf-8') as f: return [json.loads(x) for x in f if x.strip()]


_P = None
def _proc():
    global _P
    if _P is None:
        import torch; torch.set_num_threads(1)
        from transformers import AutoProcessor
        _P = AutoProcessor.from_pretrained(MODEL, revision=REV, local_files_only=True, trust_remote_code=False)
    return _P


def _image_batch(paths):
    from PIL import Image, ImageOps
    imgs = []
    for p in paths:
        with Image.open(p) as im: imgs.append(ImageOps.exif_transpose(im).convert('RGB').copy())
    pv = _proc()(images=imgs, return_tensors='pt')
    if set(pv) != {'pixel_values'}: raise ValueError('Unexpected image processor contract')
    a = pv['pixel_values'].numpy()
    if a.dtype != np.float32 or a.shape[1:] != (3, 224, 224): raise ValueError(f'Unexpected pixel tensor {a.dtype} {a.shape}')
    return [tensor_key(x) for x in a]


def _text_batch(texts):
    inp = _proc()(text=[t.lower() for t in texts], padding='max_length', max_length=64, truncation=True, return_tensors='pt')
    ids = inp['input_ids'].numpy().astype(np.int64)
    return [tensor_key(x) for x in ids]


def content_keys(images, texts, workers, B=32):
    t0 = time.time()
    ib = [[r['path'] for r in images[s:s+B]] for s in range(0, len(images), B)]
    tb = [[r['text'] for r in texts[s:s+B]] for s in range(0, len(texts), B)]
    with Pool(workers) as pool:
        ik = [k for b in pool.imap(_image_batch, ib, chunksize=4) for k in b]
        tk = [k for b in pool.imap(_text_batch, tb, chunksize=16) for k in b]
    return np.array(ik), np.array(tk), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)             # flickr | cc152k_recovered
    ap.add_argument('--cache', required=True)               # NR1 FIT cache dir
    ap.add_argument('--images', required=True); ap.add_argument('--texts', required=True)
    ap.add_argument('--condition', action='append', required=True, help='NAME=observed.tsv')
    ap.add_argument('--out', required=True); ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cache = Path(a.cache); man = json.load(open(cache/'manifest.json'))
    if man['model_id'] != MODEL or man['model_revision'] != REV or man['normalization'] != 'sample_l2_no_center' or man['split'] != 'fit':
        raise ValueError('Cache is not the locked NR1 FIT SigLIP2 sample-L2 cache')
    used = ['image_global.npy', 'text_global.npy', 'image_ids.json', 'text_ids.json']
    files = man['files'] if isinstance(man['files'], dict) else None
    for name in used:
        h = sha256(cache/name)
        if h != files[name]: raise ValueError(f'Cache hash mismatch {name}')
    img_ids = json.load(open(cache/'image_ids.json')); txt_ids = json.load(open(cache/'text_ids.json'))
    ims, txs = jsonl(a.images), jsonl(a.texts)
    if [r['id'] for r in ims] != img_ids or [r['id'] for r in txs] != txt_ids:
        raise ValueError('jsonl order differs from cache order')
    if sha256(a.images) != man['images_manifest_sha256'] or sha256(a.texts) != man['texts_manifest_sha256']:
        raise ValueError('jsonl manifests differ from the files bound in the cache manifest')
    X = np.load(cache/'image_global.npy').astype(np.float32); Y = np.load(cache/'text_global.npy').astype(np.float32)
    ck_path = out/f'content_keys_{a.dataset}.npz'
    if ck_path.exists():
        z = np.load(ck_path, allow_pickle=False); ik, tk = z['image_content_key'], z['text_content_key']; secs = None
    else:
        ik, tk, secs = content_keys(ims, txs, a.workers)
        np.savez(ck_path, image_content_key=ik, text_content_key=tk)
    if len(ik) != len(img_ids) or len(tk) != len(txt_ids): raise ValueError('content key count')
    ck_receipt = dict(dataset=a.dataset, file=str(ck_path), sha256=sha256(ck_path), seconds=secs,
                      n_image=len(ik), n_text=len(tk), unique_image_keys=int(len(set(ik.tolist()))),
                      unique_text_keys=int(len(set(tk.tolist()))),
                      image_key='sha256(dtype|shape|bytes) of SigLIP2 processor pixel_values (NR1 extractor path, B=32 order)',
                      text_key='sha256(dtype|shape|bytes) of processor input_ids (lowercased, max_length=64 padding, truncation)')
    write_json(out/f'content_keys_{a.dataset}.json', ck_receipt)
    imap = {x: i for i, x in enumerate(img_ids)}; tmap = {x: i for i, x in enumerate(txt_ids)}
    receipts = {}
    for spec in a.condition:
        name, obs = spec.split('=', 1)
        with open(obs, newline='', encoding='utf-8') as f:
            r = csv.DictReader(f, delimiter='\t')
            if set(r.fieldnames or []) != {'record_id', 'image_id', 'text_id'}: raise ValueError('Observed schema')
            rows = list(r)
        rec = [x['record_id'] for x in rows]
        if len(set(rec)) != len(rec): raise ValueError('duplicate records')
        trow = np.array([tmap[x['text_id']] for x in rows], np.int64)
        irow = np.array([imap[x['image_id']] for x in rows], np.int64)
        if len(np.unique(trow)) != len(trow): raise ValueError('text used by two records: bank needs record-level text rows')
        counts = np.bincount(irow, minlength=len(img_ids))
        keep = np.flatnonzero(counts > 0)       # explicit trim of zero observed-marginal images (if any)
        new_of_old = -np.ones(len(img_ids), np.int64); new_of_old[keep] = np.arange(len(keep))
        cdir = out/name; cdir.mkdir(parents=True, exist_ok=True)
        bank = cdir/'fit_bank.npz'
        if bank.exists(): raise FileExistsError(f'{bank} exists; refuse to overwrite')
        np.savez(bank, image_global=X[keep], text_global=Y[trow],
                 observed_image_for_text=new_of_old[irow],
                 image_id=np.array([img_ids[i] for i in keep]), text_id=np.array([txt_ids[j] for j in trow]),
                 image_content_key=ik[keep], text_content_key=tk[trow])
        b = dict(condition=name, model_id=MODEL, model_revision=REV, normalization='sample_l2_no_center', split='FIT',
                 bank_sha256=sha256(bank), source_cache=str(cache), source_cache_manifest_sha256=sha256(cache/'manifest.json'),
                 source_cache_files_used={k: files[k] for k in used},
                 observed_table=str(obs), observed_manifest_sha256=sha256(obs),
                 content_keys=ck_receipt,
                 membership=dict(n_image_cache=len(img_ids), n_image_bank=int(len(keep)),
                                 trimmed_zero_marginal_images=[img_ids[i] for i in np.flatnonzero(counts == 0)],
                                 image_rows='cache image order restricted to positive observed marginal (old->new map = rank among kept)',
                                 text_rows='observed table record order; text_id = cache text id (unique per record)',
                                 record_ids_sha256=hashlib.sha256('\n'.join(rec).encode()).hexdigest(),
                                 image_ids_sha256=hashlib.sha256('\n'.join(img_ids[i] for i in keep).encode()).hexdigest(),
                                 text_ids_sha256=hashlib.sha256('\n'.join(txt_ids[j] for j in trow).encode()).hexdigest()),
                 observed_image_max_count=int(counts.max()),
                 read_scope='FIT cache globals/ids, FIT jsonl, observed table, pinned processor; NO truth/DEV/noise/audit files',
                 utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        write_json(cdir/'fit_bank.binding.json', b)
        receipts[name] = dict(bank=str(bank), bank_sha256=b['bank_sha256'], n_image=int(len(keep)), n_text=int(len(trow)),
                              trimmed=len(b['membership']['trimmed_zero_marginal_images']))
        print(name, receipts[name], flush=True)
    write_json(out/f'BUILD_{a.dataset}.json', dict(content_keys=ck_receipt, conditions=receipts))


if __name__ == '__main__': main()
