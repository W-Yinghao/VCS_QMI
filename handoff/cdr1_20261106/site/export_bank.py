#!/usr/bin/env python3
"""CDR1 site bridge: existing layout-free NR1/NR2C SigLIP2 global FIT caches -> curator bank.

Read-only on the source caches. Verifies file hashes against the recorded manifests,
joins captions to images by EXPLICIT ids from the audit ID maps (never row//5),
merges byte-identical images (same raw sha256) into one group before export.
Output is CURATOR-ONLY (bank_clean.npz + meta json).
"""
import argparse, csv, hashlib, json, sys, time
from pathlib import Path
import numpy as np

SRC = {
    'Flickr30K': dict(
        cache='/home/infres/yinwang/CS_QMI/outputs_nr1/NR1_20261104/cache/fit',
        audit='/home/infres/yinwang/CS_QMI/outputs_nr1/NR1_20261104/audit',
        join='karpathy_img_id',
        hashes_json='/home/infres/yinwang/CS_QMI/handoff/nr1_20261104/HASHES.json'),
    'COCO': dict(
        cache='/projects/EEG-foundation-model/yinghao/outputs_nr2/NR2C_20261104/cache/fit',
        audit='/projects/EEG-foundation-model/yinghao/outputs_nr2/NR2C_20261104/audit',
        join='cocoid', hashes_json=None),
}
REVISION = '75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2'
MODEL_LOCK_SHA = 'd4ff4992a1662ae237d1c86291ba8a7dd124892da8942b15a5329217b269c7dd'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(SRC))
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    t0 = time.time()
    s = SRC[a.dataset]
    cache, audit = Path(s['cache']), Path(s['audit'])
    man = json.load(open(cache / 'manifest.json'))
    assert man['model_id'] == 'google/siglip2-base-patch16-224' and man['model_revision'] == REVISION
    assert man['normalization'] == 'sample_l2_no_center' and man['split'] == 'fit'
    checks = {}
    for f in ('image_global.npy', 'text_global.npy', 'image_ids.json', 'text_ids.json'):
        h = sha(cache / f)
        assert h == man['files'][f], f'hash mismatch {f}'
        checks[f] = h
    if s['hashes_json']:
        hj = json.load(open(s['hashes_json']))['fit_cache_files']
        for f, h in checks.items():
            assert hj[f] == h, f'HASHES.json mismatch {f}'
    X = np.load(cache / 'image_global.npy')
    Y = np.load(cache / 'text_global.npy')
    iid = json.load(open(cache / 'image_ids.json'))
    tid = json.load(open(cache / 'text_ids.json'))
    assert X.shape[0] == len(iid) and Y.shape[0] == len(tid)
    # explicit joins
    imap = {}
    with open(audit / 'ID_MAP_fit_images.tsv') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            assert r['split'] == 'fit'
            imap[r['opaque_id']] = (r[s['join']], r['image_sha256'])
    tmap = {}
    with open(audit / 'ID_MAP_fit_texts.tsv') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            assert r['split'] == 'fit'
            tmap[r['opaque_id']] = r[s['join']]
    assert set(imap) == set(iid) and set(tmap) == set(tid)
    key2row = {}
    for k, o in enumerate(iid):
        key = imap[o][0]
        assert key not in key2row, 'duplicate join key'
        key2row[key] = k
    # group byte-identical raw images (same sha256): keep lowest row as representative
    sha2rep = {}
    rep = np.empty(len(iid), np.int64)
    for k, o in enumerate(iid):
        h = imap[o][1]
        if h not in sha2rep:
            sha2rep[h] = k
        rep[k] = sha2rep[h]
    merged = [(iid[k], iid[int(rep[k])]) for k in range(len(iid)) if rep[k] != k]
    keep = np.flatnonzero(rep == np.arange(len(iid)))
    newidx = -np.ones(len(iid), np.int64)
    newidx[keep] = np.arange(len(keep))
    feat_diff = [float(np.abs(X[k] - X[int(rep[k])]).max()) for k in range(len(iid)) if rep[k] != k]
    truth = np.array([newidx[rep[key2row[tmap[o]]]] for o in tid], np.int64)
    assert (truth >= 0).all()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bank = out / 'bank_clean.npz'
    np.savez(bank, image_features=X[keep], text_features=Y,
             true_image_index=truth,
             image_ids=np.array([iid[k] for k in keep]), text_ids=np.array(tid))
    content = {'source_cache_manifest_sha256': sha(cache / 'manifest.json'),
               'source_files_sha256': checks,
               'id_map_images_sha256': sha(audit / 'ID_MAP_fit_images.tsv'),
               'id_map_texts_sha256': sha(audit / 'ID_MAP_fit_texts.tsv')}
    content_sha = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    meta = {'annotation_role': 'benchmark_clean', 'has_task_adapter': False, 'source_split': 'FIT',
            'encoder': 'google/siglip2-base-patch16-224', 'revision': REVISION,
            'model_lock_full_sha256': MODEL_LOCK_SHA,
            'duplicate_groups_verified': True,
            'duplicate_rule': 'raw image sha256 identical -> one group (lowest cache row kept); captions remapped',
            'n_merged_images': len(merged), 'merged_pairs': merged,
            'merged_feature_max_abs_diff': max(feat_diff) if feat_diff else 0.0,
            'producer': 'handoff/cdr1_20261106/site/export_bank.py sha256=' + sha(__file__),
            'content_manifest_sha256': content_sha, 'content_manifest': content,
            'join': 'text ' + s['join'] + ' -> image ' + s['join'] + ' via audit ID maps (explicit, not row//5)',
            'normalization': 'per-item L2 at extraction (sample_l2_no_center); no batch/global centering',
            'n_images_groups': int(len(keep)), 'n_texts': int(len(tid)),
            'bank_sha256': None, 'export_seconds': None}
    meta['bank_sha256'] = sha(bank)
    meta['export_seconds'] = time.time() - t0
    json.dump(meta, open(out / 'bank_meta.json', 'w'), indent=1)
    print(json.dumps({k: meta[k] for k in ('n_images_groups', 'n_texts', 'n_merged_images',
                                           'merged_feature_max_abs_diff', 'bank_sha256', 'export_seconds')}))


if __name__ == '__main__':
    main()
