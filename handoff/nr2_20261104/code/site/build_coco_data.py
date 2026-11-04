#!/usr/bin/env python3
"""NR2C site adapter (copy of the NR1 Flickr builder logic for COCO): explicit map NCR coco_precomp order ->
Karpathy cocoid -> raw COCO file / caption text. Fails fast on any mismatch.

Reads ONLY NCR train_* / dev_* (train = Karpathy train+restval, dev = the native 1K-image dev = 1000 Karpathy 'val'
images); NCR test/testall files are never opened. dataset_coco.json is parsed and its 'test' entries are dropped on
read (never stored, mapped, hashed or extracted). Writes the NR1 package schema:
  data/{fit,dev}_images.jsonl, data/{fit,dev}_texts.jsonl, data/observed_{COCO00,COCO60,COCO80}.tsv,
  evaluator_only/dev_positives.json, audit/ID_MAP_{fit,dev}_{images,texts}.tsv, audit/DATA_AUDIT.json
"""
import argparse, hashlib, html, json, re, sys
from pathlib import Path
import numpy as np

NCR = Path('/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data/coco_precomp')
RAW = Path('/projects/EEG-foundation-model/yinghao/data/coco')
NOISE = {  # existing locked maps (never regenerated)
    'COCO60': ('/projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/noise/coco_precomp_0.6.npy',
               'b141047d84c422f52f13fa0d995472a18c72efa9373e93ea4a157110f07d3ecf',
               '/home/infres/yinwang/CS_QMI/handoff/r25_in2r_20261027/NOISE_LOCK.json', 'coco_precomp_0.6'),
    'COCO80': ('/projects/EEG-foundation-model/yinghao/outputs_v9/r33_baselines_20261102/noise/coco_precomp_0.8.npy',
               '98356e1016192bf3da3b36aea6e5dd812237ca656cf4815563a58a54f4fa1c9a',
               '/home/infres/yinwang/CS_QMI/handoff/r33_baselines_20261102/NOISE_LOCK.json', 'coco_precomp_0.8'),
}
SALT = 'NR2C_20261104_opaque_v1'


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()


def opaque(kind, key): return kind + '_' + hashlib.sha256(f'{SALT}|{kind}|{key}'.encode()).hexdigest()[:20]
def norm(s): return re.sub(r'[^a-z0-9]', '', html.unescape(s).lower().encode('ascii', 'ignore').decode())
def strict_norm(s): return re.sub(r'[^a-z0-9]', '', html.unescape(s).lower())
def die(msg): sys.exit('FAIL_FAST: ' + msg)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); a = ap.parse_args()
    out = Path(a.out)
    for d in ('data', 'evaluator_only', 'audit'): (out/d).mkdir(parents=True, exist_ok=True)
    if (out/'audit'/'DATA_AUDIT.json').exists(): die('DATA_AUDIT exists; refuse to rebuild over it')
    js = json.load(open(RAW/'dataset_coco.json'))
    by_id, nsplit = {}, {}
    for r in js['images']:
        nsplit[r['split']] = nsplit.get(r['split'], 0) + 1
        if r['split'] == 'test': continue          # dropped on read: never mapped / hashed / extracted
        if r['cocoid'] in by_id: die(f'duplicate cocoid {r["cocoid"]}')
        by_id[r['cocoid']] = r
    audit = dict(karpathy_json_sha256=sha256(RAW/'dataset_coco.json'), karpathy_split_counts=nsplit,
                 karpathy_test_entries='dropped on read; not mapped/hashed/extracted', ncr_files={},
                 ncr_test_files_opened=False)
    for f in ['train_ids.txt', 'train_caps.txt', 'dev_ids.txt', 'dev_caps.txt']:
        audit['ncr_files'][f] = sha256(NCR/f)
    caption_mismatch, nonascii_only, extra_sent_images = [], [], {'fit': 0, 'dev': 0}
    html_unescaped = {'fit': 0, 'dev': 0}; tables = {}
    for split, ncr_split, kar in [('fit', 'train', ('train', 'restval')), ('dev', 'dev', ('val',))]:
        caps = [l.rstrip('\n').rstrip('\r') for l in open(NCR/f'{ncr_split}_caps.txt', encoding='utf-8')]
        ids = [l.strip() for l in open(NCR/f'{ncr_split}_ids.txt')]
        if ncr_split == 'train':
            if len(ids) * 5 != len(caps): die('train ids*5 != caps')
            img_kid = [int(x) for x in ids]
        else:
            if len(ids) != len(caps) or len(caps) % 5: die('dev ids/caps length')
            img_kid = []
            for g in range(len(caps) // 5):
                grp = set(ids[5*g:5*g+5])
                if len(grp) != 1: die(f'dev group {g} not one image: {grp}')
                img_kid.append(int(ids[5*g]))
        if len(set(img_kid)) != len(img_kid): die(f'{split}: duplicate cocoids')
        images, texts, imap, tmap, cap_img = [], [], [], [], []
        for g, kid in enumerate(img_kid):
            r = by_id.get(kid)
            if r is None: die(f'{split}: cocoid {kid} absent from Karpathy non-test entries')
            if r['split'] not in kar: die(f'{split}: cocoid {kid} split {r["split"]} not in {kar}')
            if len(r['sentences']) < 5: die(f'{split}: cocoid {kid} has {len(r["sentences"])} captions')
            extra_sent_images[split] += len(r['sentences']) > 5
            path = RAW/r['filepath']/r['filename']
            if not path.is_file(): die(f'raw image missing: {path}')
            iid = opaque('img', r['filename'])
            images.append(dict(id=iid, path=str(path))); imap.append((iid, split, g, kid, f"{r['filepath']}/{r['filename']}"))
            for j in range(5):                       # NCR order = first five Karpathy sentences (checked below)
                k, s = 5*g + j, r['sentences'][j]
                if norm(caps[k]) != norm(s['raw']):
                    caption_mismatch.append(dict(split=split, ncr_index=k, ncr=caps[k], raw=s['raw']))
                    if len(caption_mismatch) > 20: die(f'caption mismatches, first: {caption_mismatch[:3]}')
                if strict_norm(caps[k]) != strict_norm(s['raw']):
                    nonascii_only.append(dict(split=split, ncr_index=k, ncr=caps[k], raw=s['raw']))
                text = html.unescape(s['raw']).strip(); html_unescaped[split] += text != s['raw']
                tid = opaque('txt', f'coco|{r["split"]}|{s["sentid"]}')
                texts.append(dict(id=tid, text=text)); tmap.append((tid, split, k, s['sentid'], kid)); cap_img.append(g)
        if caption_mismatch: die(f'{len(caption_mismatch)} caption mismatches, first: {caption_mismatch[:3]}')
        tables[split] = (images, texts, imap, tmap, np.array(cap_img))
        audit[split] = dict(n_images=len(images), n_texts=len(texts), karpathy_splits=list(kar), ncr_split=ncr_split,
                            caption_text_check='normalized [a-z0-9]+ sequence equal for all captions (NCR tokenized vs Karpathy raw, first 5 sentences)')
    img_sha = {}
    for split in ('fit', 'dev'):
        for e in tables[split][0]: img_sha[e['path']] = sha256(e['path'])
    fit_sha = {img_sha[e['path']] for e in tables['fit'][0]}; dev_sha = {img_sha[e['path']] for e in tables['dev'][0]}
    fit_txt = {norm(t['text']) for t in tables['fit'][1]}; dev_txt = [norm(t['text']) for t in tables['dev'][1]]
    audit['overlap'] = dict(fit_dev_image_sha256_intersection=len(fit_sha & dev_sha), fit_unique_image_sha=len(fit_sha),
                            dev_unique_image_sha=len(dev_sha), dev_captions_with_normalized_text_in_fit=int(sum(t in fit_txt for t in dev_txt)),
                            dev_captions=len(dev_txt))
    for split in ('fit', 'dev'):
        images, texts, imap, tmap, _ = tables[split]
        for name, recs in [('images', images), ('texts', texts)]:
            with open(out/'data'/f'{split}_{name}.jsonl', 'w', encoding='utf-8') as f:
                for r in recs: f.write(json.dumps(r, ensure_ascii=False) + '\n')
        with open(out/'audit'/f'ID_MAP_{split}_images.tsv', 'w') as f:
            f.write('opaque_id\tsplit\tncr_image_index\tcocoid\trelpath\timage_sha256\n')
            for r in imap: f.write('\t'.join(map(str, r)) + f"\t{img_sha[str(RAW/r[4])]}\n")
        with open(out/'audit'/f'ID_MAP_{split}_texts.tsv', 'w') as f:
            f.write('opaque_id\tsplit\tncr_caption_index\tkarpathy_sentid\tcocoid\n')
            for r in tmap: f.write('\t'.join(map(str, r)) + '\n')
    images, texts, _, _, cap_img = tables['fit']
    obs = {'COCO00': dict(source='identity (clean NCR order, caption k -> image k//5)', map=cap_img.copy(), sha=None)}
    for cond, (path, digest, lockfile, key) in NOISE.items():
        lock = json.load(open(lockfile))['files'][key]
        if lock['path'] != path or lock['sha256'] != digest: die(f'{cond}: lock entry differs')
        if sha256(path) != digest: die(f'{cond}: noise file hash mismatch')
        m = np.load(path)
        if m.shape != (len(texts),) or m.min() < 0 or m.max() >= len(images): die(f'{cond}: bad noise map shape/range')
        mism = float(np.mean(m != cap_img))
        if abs(mism - lock['realized_mismatch']) > 1e-12: die(f'{cond}: realized mismatch {mism} != lock {lock["realized_mismatch"]}')
        obs[cond] = dict(source=f'{lockfile}:{key}', map=m.astype(np.int64), sha=digest)
    audit['observed'] = {}
    for cond, o in obs.items():
        p = out/'data'/f'observed_{cond}.tsv'
        with open(p, 'w') as f:
            f.write('record_id\timage_id\ttext_id\n')
            for k in range(len(texts)):
                f.write(f'{opaque("rec", f"coco_fit|{k}")}\t{images[int(o["map"][k])]["id"]}\t{texts[k]["id"]}\n')
        cnt = np.bincount(o['map'], minlength=len(images))
        audit['observed'][cond] = dict(path=str(p), sha256=sha256(p), n_records=len(texts), source_noise_file_sha256=o['sha'],
                                       source=o['source'], realized_mismatch=float(np.mean(o['map'] != cap_img)),
                                       images_with_zero_observed_records=int((cnt == 0).sum()), max_records_per_image=int(cnt.max()))
    images, texts, _, _, cap_img = tables['dev']
    pos = [[images[int(cap_img[k])]['id'], texts[k]['id']] for k in range(len(texts))]
    tp = out/'evaluator_only'/'dev_positives.json'
    tp.write_text(json.dumps(dict(protocol_id='NR1_EVAL_TRUTH_V1', positives=pos)))
    audit['dev_truth'] = dict(path=str(tp), sha256=sha256(tp), n_pairs=len(pos))
    for split in ('fit', 'dev'):
        for name in ('images', 'texts'):
            p = out/'data'/f'{split}_{name}.jsonl'; audit.setdefault('manifests', {})[f'{split}_{name}'] = dict(path=str(p), sha256=sha256(p))
    audit['caption_text'] = dict(source='dataset_coco.json raw (first 5 sentences), html.unescape + strip', html_unescaped_or_stripped=html_unescaped,
                                 images_with_more_than_5_sentences=extra_sent_images, captions_equal_only_after_dropping_nonascii=nonascii_only)
    audit['status'] = 'PASS'
    (out/'audit'/'DATA_AUDIT.json').write_text(json.dumps(audit, indent=1))
    print(json.dumps({k: v for k, v in audit.items() if k != 'ncr_files'}, indent=1)[:6000])


if __name__ == '__main__':
    main()
