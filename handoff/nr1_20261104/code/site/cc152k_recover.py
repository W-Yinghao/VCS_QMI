#!/usr/bin/env python3
"""Freeze the CC152K recovered snapshot (scenario C of CC152K_ACTIONS_CN.md §3) BEFORE any model score exists.
Binding: TRAIN by official NCR ID (train_caps_img_urls.csv@60256ba1); DEV by unique exact (NFC/whitespace) caption match to
CC3M validation consistent with the ID; everything else UNRESOLVED.  Exclusion rules fixed here, identical for all methods."""
import csv, hashlib, json, collections, sys, time
from pathlib import Path
R = Path('/projects/EEG-foundation-model/yinghao/data/cc152k_raw'); NCR = Path('/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data/cc152k_precomp')
H = Path('/home/infres/yinwang/CS_QMI/handoff/nr1_20261104'); C = H/'cc152k'
SNAP = '20261104'; COND = f'CC152K_RECOVERED_{SNAP}'; PROTO = f'NR1_CC152K_RECOVERED_SNAPSHOT_{SNAP}_V1'
SALT = 'NR1_20261104_opaque_v1'; op = lambda k, key: k + '_' + hashlib.sha256(f'{SALT}|{k}|{key}'.encode()).hexdigest()[:20]
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
out = R/'nr1'/COND
if (out/'FROZEN.json').exists(): sys.exit('already frozen')
audit = {(r['split'], r['ncr_id']): r for r in csv.DictReader(open(C/'CC_ID_JOIN_AUDIT.csv'))}
def mojibake_ok(o, l):
    for enc in ('cp1252', 'latin-1'):
        try:
            if o.encode(enc).decode('utf-8') == l: return True
        except Exception: pass
    return False
off = {l.split('\t')[0]: l.rstrip('\n').split('\t') for l in open(R/'official_ncr_metadata/train_caps_img_urls.csv', encoding='utf-8')}
caps = {s: [l.rstrip('\r\n').split('\t') for l in open(NCR/f'{s}_caps.tsv', encoding='utf-8')] for s in ('train', 'dev')}
stat = {s: {json.loads(l)['ncr_id']: json.loads(l) for l in open(R/'manifest'/f'{s}_status.jsonl')} for s in ('train', 'dev')}
for s in stat:  # one retry pass for transient failures overrides the first status (both files kept)
    rp = R/'manifest'/f'{s}_status_retry1.jsonl'
    if rp.exists():
        for l in open(rp):
            d = json.loads(l); d['retried'] = True; d['first_status'] = stat[s][d['ncr_id']]['status']; stat[s][d['ncr_id']] = d
head = {s: {json.loads(l)['ncr_id']: json.loads(l) for l in open(R/'manifest'/f'{s}_headpass.jsonl')} for s in ('train', 'dev')}
def binding(s, rid, cap):
    a = audit[(s, rid)]
    if s == 'train':
        if a['status'] == 'BOUND_OFFICIAL_ID': b = 'BOUND_OFFICIAL_ID'
        elif a['status'] == 'UNRESOLVED_CAPTION_CONFLICT' and mojibake_ok(off[rid][1], cap): b = 'BOUND_OFFICIAL_ID_CAPTION_MOJIBAKE'
        else: return 'UNRESOLVED_' + a['status']
        if a['url_equals_earlier_cc3m_mapping'] != 'True': return 'UNRESOLVED_OFFICIAL_URL_DIFFERS_FROM_CC3M_ROW_SAME_CRC32'
        return b
    return a['status'] if a['status'] == 'BOUND_UNIQUE_EXACT_CAPTION' else 'UNRESOLVED_' + a['status']
# placeholder rule (fixed): byte-identical content shared by >=3 records with >=3 distinct registered URLs, or imgur 161x81 'removed'
ok_all = [r for s in stat for r in stat[s].values() if r['status'] == 'ok']
grp = collections.defaultdict(list)
for r in ok_all: grp[r['sha256']].append(r)
placeholder = {h for h, v in grp.items() if len(v) >= 3 and len({x['url'] for x in v}) >= 3} | \
              {r['sha256'] for r in ok_all if 'imgur.com' in r['url'] and (r['width'], r['height']) == (161, 81)}
rec = {}; missing = []; dl = []
for s in ('train', 'dev'):
    for k, (rid, cap) in enumerate(caps[s]):
        b = binding(s, rid, cap); st = stat[s].get(rid, {}); hp = head[s].get(rid, {})
        ct = (hp.get('content_type') or '').lower()
        dl.append(dict(split=s, ncr_id=rid, binding=b, **{k2: v for k2, v in st.items() if k2 != 'ncr_id'}, **{k2: v for k2, v in hp.items() if k2 != 'ncr_id'}))
        reason = None
        if not b.startswith('BOUND'): reason = b
        elif st.get('status') != 'ok': reason = 'DOWNLOAD_' + st.get('status', 'missing')
        elif st['sha256'] in placeholder: reason = 'PLACEHOLDER_IMAGE'
        elif 'pixel_sha256' not in hp: reason = 'DECODE_FAILED_PIXEL_HASH'
        rec[(s, rid)] = dict(k=k, cap=cap, reason=reason, st=st, hp=hp)
        if reason: missing.append(dict(split=s, ncr_line=k, ncr_id=rid, reason=reason))
# cross-split rule (fixed): FIT records whose image bytes equal a kept DEV image are dropped from FIT
dev_sha = {v['st']['sha256'] for (s, _), v in rec.items() if s == 'dev' and not v['reason']}
for (s, rid), v in rec.items():
    if s == 'train' and not v['reason'] and v['st']['sha256'] in dev_sha:
        v['reason'] = 'FIT_IMAGE_BYTES_EQUAL_DEV_IMAGE'; missing.append(dict(split=s, ncr_line=v['k'], ncr_id=rid, reason=v['reason']))
(out/'data').mkdir(parents=True, exist_ok=True); (out/'evaluator_only').mkdir(exist_ok=True)
files = {}; counts = {}
for s, role in (('train', 'fit'), ('dev', 'dev')):
    keep = [(rid, v) for (ss, rid), v in sorted(rec.items(), key=lambda x: x[1]['k']) if ss == s and not v['reason']]
    ims = [dict(id=op('ccimg', rid), path=v['st']['path']) for rid, v in keep]; txs = [dict(id=op('cctxt', rid), text=v['cap']) for rid, v in keep]
    for nm, rs in (('images', ims), ('texts', txs)):
        p = out/'data'/f'{role}_{nm}.jsonl'
        with open(p, 'w', encoding='utf-8') as f:
            for x in rs: f.write(json.dumps(x, ensure_ascii=False) + '\n')
        files[f'{role}_{nm}'] = dict(path=str(p), sha256=sha(p), n=len(rs))
    p = out/'data'/f'{role}_membership_ncr_ids.txt'; p.write_text(''.join(rid + '\n' for rid, _ in keep)); files[f'{role}_membership'] = dict(path=str(p), sha256=sha(p), n=len(keep))
    if role == 'fit':
        p = out/'data'/f'observed_{COND}.tsv'
        with open(p, 'w') as f:
            f.write('record_id\timage_id\ttext_id\n')
            for rid, _ in keep: f.write(f"{op('ccrec', rid)}\t{op('ccimg', rid)}\t{op('cctxt', rid)}\n")
        files['observed'] = dict(path=str(p), sha256=sha(p), n=len(keep), relation='NCR observed caption<->image record relation (real web noise; no clean labels)')
    else:
        p = out/'evaluator_only'/'dev_positives.json'
        p.write_text(json.dumps(dict(protocol_id='NR1_EVAL_TRUTH_V1', positives=[[op('ccimg', rid), op('cctxt', rid)] for rid, _ in keep])))
        files['dev_truth'] = dict(path=str(p), sha256=sha(p), n=len(keep), note='observed NCR dev relation, evaluator-only; paired filtering of queries and gallery')
    c = collections.Counter(v['reason'] or 'KEPT' for (ss, _), v in rec.items() if ss == s)
    counts[role] = dict(original=len(caps[s]), kept=len(keep), coverage=len(keep)/len(caps[s]), by_reason=dict(c))
with open(C/'DOWNLOAD_STATUS.jsonl', 'w') as f:
    for d in dl: f.write(json.dumps(d) + '\n')
with open(C/'MISSING_IDS.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['split', 'ncr_line', 'ncr_id', 'reason']); w.writeheader(); w.writerows(sorted(missing, key=lambda x: (x['split'], x['ncr_line'])))
with open(C/'RAW_CONTENT_HASHES.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['split', 'ncr_id', 'kept', 'byte_sha256', 'pixel_sha256', 'bytes', 'format', 'width', 'height', 'path'])
    for (s, rid), v in sorted(rec.items(), key=lambda x: (x[0][0], x[1]['k'])):
        if v['st'].get('status') == 'ok':
            w.writerow([s, rid, not v['reason'], v['st']['sha256'], v['hp'].get('pixel_sha256', ''), v['st']['bytes'], v['st']['format'], v['st']['width'], v['st']['height'], v['st']['path']])
dl_dates = sorted({d.get('utc', '')[:10] for d in dl if d.get('utc')})
splits = dict(scenario='C (FIT partial and DEV partial)', condition=COND, protocol_id=PROTO, snapshot=SNAP, fetch_dates_utc=dl_dates,
              naming='recovered/rehydrated snapshot of web URLs; byte identity with the images used by the NCR authors is NOT certified (no original content hashes exist)',
              counts=counts, exclusion_rules=dict(binding='train: official NCR ID (+ caption equal after NFC/whitespace, or exact after documented cp1252/latin-1->utf-8 mojibake repair of the official CSV); 41 records whose official URL differs from the CC3M row URL with the same crc32 -> UNRESOLVED. dev: unique exact caption match in CC3M validation consistent with ID prefix/suffix; repeated captions -> UNRESOLVED',
              download='status must be ok (HTTP 200, decodable image <=50MB)', placeholder='byte sha256 shared by >=3 records with >=3 distinct URLs, or imgur 161x81 removed image', html='HTML/non-image 200 responses rejected at download time by image decoding (status not_image); the later HEAD pass is metadata only (18k shutterstock/picdn HEADs returned 403 text/html after the crawl, not used for exclusion)',
              cross_split='FIT records with image bytes identical to a kept DEV image are excluded from FIT', within_fit_duplicates='kept (recorded in RAW_CONTENT_HASHES.csv)'),
              placeholder_groups=len(placeholder), files=files, frozen_before_any_model_score=True, test_touched=False, frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
(out/'FROZEN.json').write_text(json.dumps(splits, indent=1)); (C/'RECOVERED_SPLITS.json').write_text(json.dumps(splits, indent=1))
for p in [C/'DOWNLOAD_STATUS.jsonl', C/'MISSING_IDS.csv', C/'RAW_CONTENT_HASHES.csv', out/'FROZEN.json']: p.chmod(0o444)
print(json.dumps({k: v for k, v in splits.items() if k != 'files'}, indent=1))
