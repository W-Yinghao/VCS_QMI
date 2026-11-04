#!/usr/bin/env python3
"""Supplement DOWNLOAD_STATUS for status=ok records: final URL + Content-Type (HEAD, redirects followed; GET range 0-0 if
HEAD is refused) and pixel SHA256 of the decoded RGB image (local). Same per-host politeness as the downloader."""
import hashlib, json, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse
import numpy as np, requests
from PIL import Image, ImageOps
Image.MAX_IMAGE_PIXELS = 200_000_000
R = Path('/projects/EEG-foundation-model/yinghao/data/cc152k_raw'); UA = 'Mozilla/5.0 (X11; Linux x86_64) research-image-dataset-fetcher/1.0'
sems, lk = {}, threading.Lock()
def sem(h):
    with lk:
        return sems.setdefault(h, threading.BoundedSemaphore(4))
loc = threading.local()
def one(r):
    if not hasattr(loc, 's'): loc.s = requests.Session(); loc.s.headers['User-Agent'] = UA
    out = dict(ncr_id=r['ncr_id'])
    try:
        with Image.open(r['path']) as im:
            a = np.asarray(ImageOps.exif_transpose(im).convert('RGB')); out['pixel_sha256'] = hashlib.sha256(a.tobytes()).hexdigest(); out['pixel_shape'] = list(a.shape)
    except Exception as e:
        out['pixel_error'] = type(e).__name__
    try:
        with sem(urlparse(r['url']).netloc):
            h = loc.s.head(r['url'], timeout=(10, 20), allow_redirects=True)
            if h.status_code >= 400:
                h = loc.s.get(r['url'], timeout=(10, 20), allow_redirects=True, stream=True, headers={'Range': 'bytes=0-0'}); h.close()
        out.update(final_url=h.url, content_type=h.headers.get('Content-Type'), head_status=h.status_code, head_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    except Exception as e:
        out.update(head_error=type(e).__name__)
    return out
for split in ('dev', 'train'):
    m = {json.loads(l)['ncr_id']: json.loads(l) for l in open(R/'manifest'/f'{split}_status.jsonl')}
    rp = R/'manifest'/f'{split}_status_retry1.jsonl'
    if rp.exists(): m.update({json.loads(l)['ncr_id']: json.loads(l) for l in open(rp)})  # retry overrides
    rs = [r for r in m.values() if r['status'] == 'ok']
    p = R/'manifest'/f'{split}_headpass.jsonl'; done = {json.loads(l)['ncr_id'] for l in open(p)} if p.exists() else set()
    todo = [r for r in rs if r['ncr_id'] not in done]; print(split, len(rs), 'todo', len(todo), flush=True)
    with open(p, 'a') as f, ThreadPoolExecutor(48) as ex:
        for i, o in enumerate(ex.map(one, todo)):
            f.write(json.dumps(o) + '\n')
            if i % 5000 == 0: f.flush(); print(split, i, flush=True)
    print(split, 'HEADPASS_DONE', flush=True)
