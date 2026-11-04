#!/usr/bin/env python3
"""Build the NR1 CC152K_PARTIAL condition from downloaded images (train=FIT, dev=DEV; TEST never touched).
Exclusions (fixed before any feature/score exists): non-ok download status; byte-identical placeholder groups
(same sha256 for >=3 records with >=3 distinct URLs, or the imgur 'removed' 161x81 image).
Observed relation = NCR record (caption k <-> image of record k), real noise, no clean labels."""
import collections, hashlib, json, sys
from pathlib import Path
R = Path('/projects/EEG-foundation-model/yinghao/data/cc152k_raw'); NCR = Path('/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data/cc152k_precomp')
H = Path('/home/infres/yinwang/CS_QMI/handoff/nr1_20261104')
SALT = 'NR1_20261104_opaque_v1'
op = lambda k, key: k + '_' + hashlib.sha256(f'{SALT}|{k}|{key}'.encode()).hexdigest()[:20]
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
st = {s: {json.loads(l)['ncr_id']: json.loads(l) for l in open(R/'manifest'/f'{s}_status.jsonl')} for s in ('train', 'dev')}
caps = {s: [l.rstrip('\r\n').split('\t') for l in open(NCR/f'{s}_caps.tsv', encoding='utf-8')] for s in ('train', 'dev')}
ok = [r for s in st for r in st[s].values() if r['status'] == 'ok']
by_sha = collections.defaultdict(list)
for r in ok: by_sha[r['sha256']].append(r)
placeholder = {h for h, rs in by_sha.items() if len(rs) >= 3 and len({r['url'] for r in rs}) >= 3}
placeholder |= {r['sha256'] for r in ok if 'imgur.com' in r['url'] and (r['width'], r['height']) == (161, 81)}
top = sorted(((len(v), h[:16], v[0]['width'], v[0]['height'], v[0]['url'][:80]) for h, v in by_sha.items() if len(v) >= 2), reverse=True)[:25]
cov = {}
name = None
kept = {}
for s in ('train', 'dev'):
    c = collections.Counter(r['status'] for r in st[s].values())
    keep = [k for k, (rid, cap) in enumerate(caps[s]) if st[s].get(rid, {}).get('status') == 'ok' and st[s][rid]['sha256'] not in placeholder]
    kept[s] = keep
    cov[s] = dict(records=len(caps[s]), status_counts=dict(c), ok=c['ok'], placeholder_excluded=sum(1 for rid, _ in caps[s] if st[s].get(rid, {}).get('status') == 'ok' and st[s][rid]['sha256'] in placeholder),
                  kept=len(keep), coverage=len(keep)/len(caps[s]))
fp, dp = round(100*cov['train']['coverage']), round(100*cov['dev']['coverage'])
name = f'CC152K_PARTIAL_F{fp}D{dp}'
out = R/'nr1'/name; (out/'data').mkdir(parents=True, exist_ok=True); (out/'evaluator_only').mkdir(exist_ok=True)
if (out/'PREP.json').exists(): sys.exit('PREP exists; refuse to overwrite')
files = {}
for s, role in (('train', 'fit'), ('dev', 'dev')):
    ims, txs, mem = [], [], []
    for k in kept[s]:
        rid, cap = caps[s][k]; r = st[s][rid]
        ims.append(dict(id=op('ccimg', rid), path=r['path'])); txs.append(dict(id=op('cctxt', rid), text=cap)); mem.append(rid)
    for nm, recs in (('images', ims), ('texts', txs)):
        p = out/'data'/f'{role}_{nm}.jsonl'
        with open(p, 'w', encoding='utf-8') as f:
            for x in recs: f.write(json.dumps(x, ensure_ascii=False) + '\n')
        files[f'{role}_{nm}'] = dict(path=str(p), sha256=sha(p), n=len(recs))
    p = out/'data'/f'{role}_membership_ncr_ids.txt'; p.write_text('\n'.join(mem) + '\n'); files[f'{role}_membership'] = dict(path=str(p), sha256=sha(p), n=len(mem))
    if role == 'fit':
        p = out/'data'/f'observed_{name}.tsv'
        with open(p, 'w') as f:
            f.write('record_id\timage_id\ttext_id\n')
            for rid in mem: f.write(f"{op('ccrec', rid)}\t{op('ccimg', rid)}\t{op('cctxt', rid)}\n")
        files['observed'] = dict(path=str(p), sha256=sha(p), n=len(mem))
    else:
        p = out/'evaluator_only'/'dev_positives.json'
        p.write_text(json.dumps(dict(protocol_id='NR1_EVAL_TRUTH_V1', positives=[[op('ccimg', rid), op('cctxt', rid)] for rid in mem])))
        files['dev_truth'] = dict(path=str(p), sha256=sha(p), n=len(mem), note='DEV truth = NCR observed caption<->image relation (real data; no clean labels exist)')
ok_sha = {s: {st[s][caps[s][k][0]]['sha256'] for k in kept[s]} for s in st}
prep = dict(condition=name, coverage=cov, placeholder_rule='sha256 shared by >=3 records with >=3 distinct URLs, or imgur 161x81 removed image',
            placeholder_groups=len(placeholder), top_duplicate_groups=top, fit_dev_image_sha_intersection=len(ok_sha['train'] & ok_sha['dev']),
            duplicate_bytes_within_fit=cov['train']['kept'] - len(ok_sha['train']), files=files, test_touched=False)
(out/'PREP.json').write_text(json.dumps(prep, indent=1)); (H/'cc152k'/'CC152K_PARTIAL_PREP.json').write_text(json.dumps(prep, indent=1))
print(json.dumps({k: v for k, v in prep.items() if k != 'files'}, indent=1)[:4000])
