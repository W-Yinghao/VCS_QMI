#!/usr/bin/env python3
"""Polite parallel download of CC152K FIT(train)+DEV images for id-resolved records (TEST never touched).
Per-record status: ok / http_<code> / timeout / conn_error / too_large / not_image / corrupt. Resumable."""
import argparse, hashlib, io, json, os, sys, threading, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse
import requests
from PIL import Image
Image.MAX_IMAGE_PIXELS = 200_000_000
R = Path('/projects/EEG-foundation-model/yinghao/data/cc152k_raw')
UA = 'Mozilla/5.0 (X11; Linux x86_64) research-image-dataset-fetcher/1.0'
MAXB = 50 << 20
host_sem, host_lock = {}, threading.Lock()


def sem(host):
    with host_lock:
        if host not in host_sem: host_sem[host] = threading.BoundedSemaphore(4)  # per-host politeness
        return host_sem[host]


def fetch(rec, sess):
    url = rec['url']; host = urlparse(url).netloc
    for attempt in range(3):
        try:
            with sem(host):
                r = sess.get(url, timeout=(10, 30), stream=True, allow_redirects=True)
                if r.status_code != 200:
                    st = f'http_{r.status_code}'
                    if r.status_code in (429, 500, 502, 503, 504) and attempt < 2: time.sleep(2 + 4*attempt); continue
                    return dict(status=st)
                b = r.raw.read(MAXB + 1, decode_content=True)
            if len(b) > MAXB: return dict(status='too_large')
            break
        except requests.exceptions.Timeout:
            if attempt < 2: time.sleep(2); continue
            return dict(status='timeout')
        except Exception as e:
            if attempt < 2: time.sleep(2); continue
            return dict(status='conn_error', error=type(e).__name__)
    try:
        im = Image.open(io.BytesIO(b)); fmt = im.format; im.verify()
    except Exception:
        return dict(status='not_image', bytes=len(b))
    try:
        im = Image.open(io.BytesIO(b)); im.load(); w, h = im.size
    except Exception:
        return dict(status='corrupt', bytes=len(b))
    if w < 2 or h < 2: return dict(status='corrupt', bytes=len(b))
    ext = {'JPEG': 'jpg', 'PNG': 'png', 'GIF': 'gif', 'WEBP': 'webp', 'BMP': 'bmp'}.get(fmt, (fmt or 'bin').lower())
    d = R/'images'/rec['ncr_split']; p = d/f"{rec['ncr_id']}.{ext}"
    tmp = p.with_suffix('.part'); tmp.write_bytes(b); tmp.replace(p)
    return dict(status='ok', path=str(p), sha256=hashlib.sha256(b).hexdigest(), bytes=len(b), format=fmt, width=w, height=h)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--splits', nargs='+', default=['dev', 'train']); ap.add_argument('--workers', type=int, default=48)
    ap.add_argument('--retry-transient', action='store_true', help='one retry pass for conn_error/timeout/429/5xx -> {split}_status_retry1.jsonl')
    a = ap.parse_args()
    try:
        requests.get('https://huggingface.co', timeout=10, headers={'User-Agent': UA}); print('internet OK', flush=True)
    except Exception as e:
        sys.exit(f'NO_INTERNET: {e}')
    for split in a.splits:
        assert split in ('train', 'dev')
        (R/'images'/split).mkdir(parents=True, exist_ok=True)
        recs = [json.loads(l) for l in open(R/'mapping'/f'{split}_map.jsonl')]
        recs = [r for r in recs if r['resolved']]
        stat_p = R/'manifest'/f'{split}_status.jsonl'; stat_p.parent.mkdir(exist_ok=True)
        if a.retry_transient:
            base = {json.loads(l)['ncr_id']: json.loads(l)['status'] for l in open(stat_p)}
            tr = lambda x: x in ('conn_error', 'timeout', 'http_429') or (x.startswith('http_5') and x != 'http_501')
            stat_p = R/'manifest'/f'{split}_status_retry1.jsonl'
            done = {json.loads(l)['ncr_id'] for l in open(stat_p)} if stat_p.exists() else set()
            todo = [r for r in recs if tr(base[r['ncr_id']]) and r['ncr_id'] not in done]
        else:
            done = {json.loads(l)['ncr_id'] for l in open(stat_p)} if stat_p.exists() else set()
            todo = [r for r in recs if r['ncr_id'] not in done]
        print(split, 'resolved', len(recs), 'todo', len(todo), flush=True)
        t0 = time.time(); cnt = Counter(); lock = threading.Lock()
        with open(stat_p, 'a') as out, ThreadPoolExecutor(a.workers) as ex:
            sessions = threading.local()
            def job(r):
                if not hasattr(sessions, 's'):
                    sessions.s = requests.Session(); sessions.s.headers['User-Agent'] = UA
                return r, fetch(r, sessions.s)
            futs = [ex.submit(job, r) for r in todo]
            for i, f in enumerate(as_completed(futs)):
                r, res = f.result()
                with lock:
                    out.write(json.dumps(dict(ncr_id=r['ncr_id'], ncr_line=r['ncr_line'], url=r['url'], utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **res)) + '\n')
                    cnt[res['status']] += 1
                    if i % 2000 == 0: out.flush(); print(split, i, dict(cnt), f'{time.time()-t0:.0f}s', flush=True)
        print(split, 'DONE', dict(cnt), flush=True)


if __name__ == '__main__':
    main()
