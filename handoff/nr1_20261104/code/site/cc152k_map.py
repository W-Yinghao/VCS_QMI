#!/usr/bin/env python3
"""Map NCR CC152K records (train/dev only; test caption file is never opened) to Conceptual Captions rows.
NCR id = '<idx>_<number>'. Hypothesis checked: idx = row index in the CC3M split (train->train, dev->validation)
and number = zlib.crc32(image_url) & 0xffffffff. Independent check: caption-text matching after normalisation."""
import hashlib, html, json, re, sys, zlib
from collections import defaultdict
from pathlib import Path
import pyarrow.parquet as pq
NCR = Path('/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data/cc152k_precomp')
HF = Path('/projects/EEG-foundation-model/yinghao/data/cc152k_raw/cc3m_hf/unlabeled')
OUT = Path('/projects/EEG-foundation-model/yinghao/data/cc152k_raw/mapping'); OUT.mkdir(parents=True, exist_ok=True)
H = Path('/home/infres/yinwang/CS_QMI/handoff/nr1_20261104/cc152k')
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
n1 = lambda s: ' '.join(html.unescape(s).lower().split())
n2 = lambda s: re.sub(r'[^a-z0-9]', '', html.unescape(s).lower())
cc = {}
for name, files in [('train', sorted(HF.glob('train-*.parquet'))), ('validation', sorted(HF.glob('validation-*.parquet')))]:
    rows = []
    for f in files: rows += pq.read_table(f).to_pylist()
    cc[name] = rows
    print(name, len(rows), flush=True)
idx1 = defaultdict(list); idx2 = defaultdict(list)
for name, rows in cc.items():
    for i, r in enumerate(rows):
        idx1[n1(r['caption'])].append((name, i)); idx2[n2(r['caption'])].append((name, i))
report = dict(cc3m_source=dict(repo='google-research-datasets/conceptual_captions', revision='0bb028f274446e0b102c1253d087a98eeb4519a3', config='unlabeled',
              files={str(f.relative_to(HF.parent)): dict(bytes=f.stat().st_size, sha256=sha(f)) for f in sorted(HF.glob('*.parquet'))},
              rows={k: len(v) for k, v in cc.items()}, note='Google GCS TSVs (storage.googleapis.com/gcc-data) returned AccessDenied 2026-11-04; HF mirror of the same release used'),
              test_split='NOT read (NR1 never reads TEST)', splits={})
for split, ccsplit in [('train', 'train'), ('dev', 'validation')]:
    lines = [l.rstrip('\r\n').split('\t') for l in open(NCR/f'{split}_caps.tsv', encoding='utf-8')]
    recs, st = [], defaultdict(int)
    for k, (rid, cap) in enumerate(lines):
        idx, num = rid.split('_'); idx, num = int(idx), int(num)
        row = cc[ccsplit][idx] if idx < len(cc[ccsplit]) else None
        crc_ok = row is not None and (zlib.crc32(row['image_url'].encode('utf-8')) & 0xffffffff) == num
        cap_ok = row is not None and n1(row['caption']) == n1(cap)
        cap_ok2 = row is not None and n2(row['caption']) == n2(cap)
        t1 = idx1.get(n1(cap), []); t2 = idx2.get(n2(cap), [])
        text_status = 'unique' if len(t1) == 1 else ('ambiguous' if len(t1) > 1 else ('unique_loose' if len(t2) == 1 else ('ambiguous_loose' if len(t2) > 1 else 'unmatched')))
        st[f'text_{text_status}'] += 1
        st['id_row_and_crc32_match'] += crc_ok; st['id_row_caption_match_exact'] += cap_ok; st['id_row_caption_match_loose'] += cap_ok2
        if text_status in ('unique', 'unique_loose'):
            st['text_unique_agrees_with_id_row'] += ((t1 or t2)[0] == (ccsplit, idx))
        resolved = bool(crc_ok and cap_ok2)
        st['resolved'] += resolved
        recs.append(dict(ncr_split=split, ncr_line=k, ncr_id=rid, cc3m_split=ccsplit, cc3m_row=idx, url=row['image_url'] if row else None,
                         crc32_ok=bool(crc_ok), caption_ok=bool(cap_ok2), resolved=resolved, text_candidates=len(t1) or len(t2)))
    with open(OUT/f'{split}_map.jsonl', 'w') as f:
        for r in recs: f.write(json.dumps(r) + '\n')
    report['splits'][split] = dict(records=len(recs), unique_urls=len({r['url'] for r in recs if r['url']}), **st, map_file=str(OUT/f'{split}_map.jsonl'), map_sha256=sha(OUT/f'{split}_map.jsonl'))
    print(split, report['splits'][split], flush=True)
(H/'CC152K_MAPPING.json').write_text(json.dumps(report, indent=1)); print('wrote', H/'CC152K_MAPPING.json')
