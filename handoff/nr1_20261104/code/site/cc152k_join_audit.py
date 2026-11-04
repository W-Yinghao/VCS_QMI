#!/usr/bin/env python3
"""Join official NCR train_caps_img_urls.csv (60256ba1) to local NCR train_caps.tsv by official ID; audit schema;
verify the suffix rule crc32(url) on the OFFICIAL id/url pairs; compare with the earlier CC3M-row mapping.
DEV: CC3M validation (HF mirror of the official release) exact-caption candidates + id-rule evidence. TEST not read."""
import csv, hashlib, html, json, unicodedata, zlib
from collections import Counter, defaultdict
from pathlib import Path
import pyarrow.parquet as pq
R = Path('/projects/EEG-foundation-model/yinghao/data/cc152k_raw'); NCR = Path('/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data/cc152k_precomp')
H = Path('/home/infres/yinwang/CS_QMI/handoff/nr1_20261104/cc152k')
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
nfc = lambda s: ' '.join(unicodedata.normalize('NFC', s).replace('\r', '').split())  # documented: NFC + line-end + whitespace only
crc = lambda u: zlib.crc32(u.encode('utf-8')) & 0xffffffff
off_p = R/'official_ncr_metadata/train_caps_img_urls.csv'
off = [l.rstrip('\n').split('\t') for l in open(off_p, encoding='utf-8')]
schema = Counter(len(r) for r in off)
loc = [l.rstrip('\r\n').split('\t') for l in open(NCR/'train_caps.tsv', encoding='utf-8')]
oid = {r[0]: i for i, r in enumerate(off)}
audit_rows, st = [], Counter()
old = {json.loads(l)['ncr_id']: json.loads(l) for l in open(R/'mapping/train_map.jsonl')}
url_count = Counter(r[2] for r in off); cap_count = Counter(nfc(r[1]) for r in off)
for k, (rid, cap) in enumerate(loc):
    j = oid.get(rid)
    if j is None:
        st['id_missing_in_official'] += 1; audit_rows.append(dict(split='train', ncr_line=k, ncr_id=rid, status='UNRESOLVED_ID_MISSING')); continue
    o = off[j]; same_row = j == k; cap_eq = nfc(o[1]) == nfc(cap); crc_ok = crc(o[2]) == int(rid.split('_')[1])
    old_url_eq = old.get(rid, {}).get('url') == o[2]
    status = 'BOUND_OFFICIAL_ID' if cap_eq else 'UNRESOLVED_CAPTION_CONFLICT'
    st[status] += 1; st['same_row_order'] += same_row; st['caption_equal_nfc'] += cap_eq; st['suffix_equals_crc32_official_url'] += crc_ok
    st['url_equals_cc3m_row_mapping'] += old_url_eq; st['url_shared_by_multiple_records'] += url_count[o[2]] > 1
    audit_rows.append(dict(split='train', ncr_line=k, ncr_id=rid, official_row=j, same_row_order=same_row, caption_equal_nfc=cap_eq,
                           suffix_crc32_ok=crc_ok, url_equals_earlier_cc3m_mapping=old_url_eq, url_dup_count=url_count[o[2]], status=status, url=o[2]))
train = dict(official_file=str(off_p), bytes=off_p.stat().st_size, sha256=sha(off_p), git_blob_sha1='55c8664b727f068ba2abaa8f53add7872e28b70a',
             schema='TSV (tab-separated, no header): ncr_id, caption, image_url', field_count_hist=dict(schema), rows=len(off),
             unique_ids=len(oid), unique_urls=len(url_count), duplicate_caption_rows=sum(v for v in cap_count.values() if v > 1),
             local_records=len(loc), ids_equal_sets=set(oid) == {r[0] for r in loc}, **st)
# DEV
val = []
for f in sorted((R/'cc3m_hf/unlabeled').glob('validation-*.parquet')): val += pq.read_table(f).to_pylist()
vcap = defaultdict(list)
for i, r in enumerate(val): vcap[nfc(r['caption'])].append(i)
dloc = [l.rstrip('\r\n').split('\t') for l in open(NCR/'dev_caps.tsv', encoding='utf-8')]
dst = Counter()
for k, (rid, cap) in enumerate(dloc):
    pre, suf = (int(x) for x in rid.split('_')); cands = vcap.get(nfc(cap), [])
    crc_cands = [i for i in cands if crc(val[i]['image_url']) == suf]
    prefix_consistent = pre < len(val) and pre in cands
    if len(cands) == 1 and crc_cands == cands and prefix_consistent: s = 'BOUND_UNIQUE_EXACT_CAPTION'
    elif len(cands) == 0: s = 'UNRESOLVED_NO_CAPTION_MATCH'
    elif len(cands) == 1: s = 'UNRESOLVED_CONTRADICTS_ID'
    elif len(crc_cands) == 1 and crc_cands[0] == pre: s = 'AMBIGUOUS_CAPTION_ID_RULE_SINGLE_CANDIDATE'
    else: s = 'UNRESOLVED_AMBIGUOUS'
    dst[s] += 1
    audit_rows.append(dict(split='dev', ncr_line=k, ncr_id=rid, caption_candidates=len(cands), crc32_candidates=len(crc_cands),
                           prefix_row_is_candidate=prefix_consistent, status=s, url=val[crc_cands[0]]['image_url'] if len(crc_cands) == 1 else ''))
dev = dict(official_dev_index='absent: github.com/XLearning-SCU/2021-NeurIPS-NCR@60256ba1 cc152k/dev_caps_img_urls.csv -> 404 (review package; re-checked)',
           cc3m_validation_source='HF google-research-datasets/conceptual_captions@0bb028f2 unlabeled/validation (mirror of the official Validation_GCC-1.1.0 TSV; GCS gcc-data returned AccessDenied 2026-11-04)',
           cc3m_validation_rows=len(val), records=len(dloc), **dst)
keys = list(dict.fromkeys(k for r in audit_rows for k in r))
with open(H/'CC_ID_JOIN_AUDIT.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys, restval=''); w.writeheader(); w.writerows(audit_rows)
json.dump(dict(train=train, dev=dev), open(H/'CC_JOIN_SUMMARY.json', 'w'), indent=1)
print(json.dumps(dict(train=train, dev=dev), indent=1))
