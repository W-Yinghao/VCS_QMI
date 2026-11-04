#!/usr/bin/env python3
"""Copy the home-tree inputs NR3 needs (Flickr caches/data/calibration/truth/proposal tables from outputs_nr1/outputs_nr2,
and the 9 Flickr NR2 HREF seed-1001 run artifacts) to the NR3 /projects tree; sha256 of every file before and after;
cross-check against PROTOCOL_FROZEN_NR2.json hashes and the cache manifests' internal file hashes. Never modifies sources."""
import hashlib, json, shutil, sys, time
from pathlib import Path
O1 = Path('/home/infres/yinwang/CS_QMI/outputs_nr1/NR1_20261104'); O2 = Path('/home/infres/yinwang/CS_QMI/outputs_nr2/NR2_20261104')
DST = Path('/projects/EEG-foundation-model/yinghao/outputs_nr3/NR3_20261104/inputs_flickr')
FZ = json.loads(Path('/home/infres/yinwang/CS_QMI/handoff/nr2_20261104/PROTOCOL_FROZEN_NR2.json').read_text())['hashes']
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()
items = []
for sub in ('cache/fit', 'cache/dev'):
    for f in sorted((O1/sub).iterdir()): items.append((f, DST/sub/f.name))
for c in ('Flickr00', 'Flickr60', 'Flickr80'):
    items += [(O1/f'data/observed_{c}.tsv', DST/f'data/observed_{c}.tsv'), (O1/f'calibration/{c}.json', DST/f'calibration/{c}.json')]
items += [(O1/'evaluator_only/dev_positives.json', DST/'evaluator_only/dev_positives.json'), (O1/'baseline/float64/METRICS.json', DST/'baseline/float64/METRICS.json'),
          (O2/'proposal/base_top100_fit.npz', DST/'proposal/base_top100_fit.npz'), (O2/'proposal/TABLES_MANIFEST.json', DST/'proposal/TABLES_MANIFEST.json')]
for c in ('Flickr00', 'Flickr60', 'Flickr80'):
    for o in ('J', 'BCE', 'NCE'):
        u = f'NR2_HREF_{c}_G_{o}_s1001'; r = O2/'runs'/u
        for f in ('BINDING.json', 'DONE.json', 'TRAIN.jsonl', 'FIT_READOUT.json', 'step_000000.pt', 'step_008000.pt', 'eval_008000/METRICS.json', 'eval_008000/PREDICTIONS.npz', 'eval_000000/METRICS.json'):
            items.append((r/f, DST/'parents'/u/f))
rec = dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), files=[])
for src, dst in items:
    s0 = sha(src); dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and sha(dst) == s0: pass
    else:
        tmp = dst.with_name(dst.name + '.tmp'); shutil.copyfile(src, tmp); tmp.replace(dst)
    s1 = sha(dst)
    if s1 != s0: sys.exit(f'COPY_MISMATCH {src}')
    e = dict(src=str(src), dst=str(dst), sha256=s1, bytes=dst.stat().st_size)
    if str(src) in FZ: e['nr2_frozen_match'] = FZ[str(src)] == s1
    rec['files'].append(e)
for sub in ('cache/fit', 'cache/dev'):
    man = json.loads((DST/sub/'manifest.json').read_text())
    bad = [n for n, h in man['files'].items() if sha(DST/sub/n) != h]
    rec[f'{sub}_manifest_internal_check'] = 'PASS' if not bad else bad
assert all(e.get('nr2_frozen_match', True) for e in rec['files'])
rec['n_files'] = len(rec['files']); rec['nr2_frozen_checked'] = sum('nr2_frozen_match' in e for e in rec['files']); rec['status'] = 'PASS'
(DST/'COPY_RECEIPT.json').write_text(json.dumps(rec, indent=1)); print(json.dumps({k: v for k, v in rec.items() if k != 'files'}))
