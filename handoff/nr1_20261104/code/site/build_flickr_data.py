#!/usr/bin/env python3
"""NR1 site adapter: explicit Flickr30K ID map from NCR precomp order -> raw files/captions.

Reads ONLY train+dev (Karpathy 'train'/'val'); test caption/id files are never opened,
test images are never extracted.  Fails fast on any mismatch.  Writes:
  data/{fit,dev}_images.jsonl, data/{fit,dev}_texts.jsonl          (package schema)
  data/observed_{Flickr00,Flickr60,Flickr80}.tsv                    (record_id,image_id,text_id)
  evaluator_only/dev_positives.json                                 (NR1_EVAL_TRUTH_V1)
  audit/ID_MAP_{fit,dev}_{images,texts}.tsv, audit/DATA_AUDIT.json
"""
import argparse, csv, hashlib, html, json, re, sys, zipfile
from pathlib import Path
import numpy as np

NCR = Path('/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data/f30k_precomp')
RAW = Path('/projects/EEG-foundation-model/yinghao/data/flickr30k')
NOISE = {  # existing locked maps (never regenerated)
    'Flickr60': ('/projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/noise/f30k_precomp_0.6.npy',
                 'd81939538fb43272bcb7aae6e408ded86c224ba9175896c3d786cbe895d42cd1',
                 '/home/infres/yinwang/CS_QMI/handoff/r25_in2r_20261027/NOISE_LOCK.json', 'f30k_precomp_0.6'),
    'Flickr80': ('/projects/EEG-foundation-model/yinghao/outputs_v9/r33_baselines_20261102/noise/f30k_precomp_0.8.npy',
                 'd4de36b7e05cd254f910c110adeb2a72e2a4aa606924d1c049128ba8ae7a68dc',
                 '/home/infres/yinwang/CS_QMI/handoff/r33_baselines_20261102/NOISE_LOCK.json', 'f30k_precomp_0.8'),
}
SALT = 'NR1_20261104_opaque_v1'


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()


def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def opaque(kind, key): return kind + '_' + hashlib.sha256(f'{SALT}|{kind}|{key}'.encode()).hexdigest()[:20]
def norm(s):
    # mapping check only: html-unescape, lowercase, drop non-ASCII and every non [a-z0-9] char
    # (NCR captions are tokenized with spaces around punctuation, "do n't", and keep non-ASCII
    # letters that the csv lost in 1 caption).
    return re.sub(r'[^a-z0-9]', '', html.unescape(s).lower().encode('ascii', 'ignore').decode())


def strict_norm(s):
    return re.sub(r'[^a-z0-9]', '', html.unescape(s).lower())
def die(msg): sys.exit('FAIL_FAST: ' + msg)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); a = ap.parse_args()
    out = Path(a.out); (out/'data').mkdir(parents=True, exist_ok=True)
    (out/'evaluator_only').mkdir(exist_ok=True); (out/'audit').mkdir(exist_ok=True)
    img_dir = out/'raw_images'/'flickr30k-images'; img_dir.mkdir(parents=True, exist_ok=True)
    if (out/'audit'/'DATA_AUDIT.json').exists(): die('DATA_AUDIT exists; refuse to rebuild over it')
    rows = list(csv.DictReader(open(RAW/'flickr_annotations_30k.csv', newline='', encoding='utf-8')))
    by_id = {}
    for r in rows:
        k = int(r['img_id'])
        if k in by_id: die(f'duplicate img_id {k}')
        by_id[k] = dict(raw=json.loads(r['raw']), sentids=json.loads(r['sentids']), split=r['split'], filename=r['filename'])
    splits = {s: sum(1 for r in by_id.values() if r['split'] == s) for s in ('train', 'val', 'test')}
    audit = dict(csv_sha256=sha256(RAW/'flickr_annotations_30k.csv'), csv_rows=len(rows), csv_split_counts=splits,
                 zip_sha256=None, ncr_files={}, test_files_opened=False)
    for f in ['train_ids.txt', 'train_caps.txt', 'dev_ids.txt', 'dev_caps.txt']:
        audit['ncr_files'][f] = sha256(NCR/f)
    z = zipfile.ZipFile(RAW/'flickr30k-images.zip'); members = set(z.namelist())
    caption_mismatch, nonascii_only = [], []
    html_unescaped = {'fit': 0, 'dev': 0}
    split_tables = {}
    for split, ncr_split, kar in [('fit', 'train', 'train'), ('dev', 'dev', 'val')]:
        caps = [l.rstrip('\n').rstrip('\r') for l in open(NCR/f'{ncr_split}_caps.txt', encoding='utf-8')]
        ids = [l.strip() for l in open(NCR/f'{ncr_split}_ids.txt')]
        if ncr_split == 'train':
            if len(ids) * 5 != len(caps): die('train ids*5 != caps')
            img_kid = [int(x) for x in ids]  # one line per image
        else:
            if len(ids) != len(caps) or len(caps) % 5: die('dev ids/caps length')
            img_kid = []
            for g in range(len(caps) // 5):
                grp = set(ids[5*g:5*g+5])
                if len(grp) != 1: die(f'dev group {g} not one image: {grp}')
                img_kid.append(int(ids[5*g]))
        if len(set(img_kid)) != len(img_kid): die(f'{split}: duplicate karpathy ids')
        images, texts, imap_rows, tmap_rows, cap_img = [], [], [], [], []
        for g, kid in enumerate(img_kid):
            r = by_id.get(kid)
            if r is None: die(f'{split}: karpathy id {kid} absent from csv')
            if r['split'] != kar: die(f'{split}: id {kid} csv split {r["split"]} != {kar}')
            if len(r['raw']) != 5: die(f'{split}: id {kid} has {len(r["raw"])} captions')
            member = 'flickr30k-images/' + r['filename']
            if member not in members: die(f'raw image missing in zip: {member}')
            iid = opaque('img', r['filename'])
            images.append(dict(id=iid, path=str(img_dir/r['filename'])))
            imap_rows.append((iid, split, g, kid, r['filename']))
            for j in range(5):
                k = 5*g + j
                if norm(caps[k]) != norm(r['raw'][j]):
                    caption_mismatch.append(dict(split=split, ncr_index=k, ncr=caps[k], raw=r['raw'][j]))
                if strict_norm(caps[k]) != strict_norm(r['raw'][j]):
                    nonascii_only.append(dict(split=split, ncr_index=k, ncr=caps[k], raw=r['raw'][j]))
                text = html.unescape(r['raw'][j])  # caption text = Karpathy raw, HTML entities decoded
                html_unescaped[split] += text != r['raw'][j]
                tid = opaque('txt', f'f30k|{kar}|{r["sentids"][j]}')
                texts.append(dict(id=tid, text=text))
                tmap_rows.append((tid, split, k, r['sentids'][j], kid))
                cap_img.append(g)
        if caption_mismatch: die(f'{len(caption_mismatch)} caption mismatches, first: {caption_mismatch[:3]}')
        split_tables[split] = (images, texts, imap_rows, tmap_rows, np.array(cap_img))
        audit[split] = dict(n_images=len(images), n_texts=len(texts), karpathy_split=kar, ncr_split=ncr_split,
                            caption_text_check='normalized [a-z0-9]+ token sequence equal for all captions (NCR tokenized vs raw csv)')
    # extract only fit+dev images and hash them
    img_sha = {}
    for split in ('fit', 'dev'):
        for e in split_tables[split][0]:
            fn = Path(e['path']).name; b = z.read('flickr30k-images/' + fn)
            p = Path(e['path'])
            if p.exists():
                if p.read_bytes() != b: die(f'existing extracted file differs: {p}')
            else:
                tmp = p.with_suffix('.tmp'); tmp.write_bytes(b); tmp.replace(p)
            img_sha[fn] = sha_bytes(b)
    audit['zip_sha256'] = sha256(RAW/'flickr30k-images.zip')
    fit_sha = {img_sha[Path(e['path']).name] for e in split_tables['fit'][0]}
    dev_sha = {img_sha[Path(e['path']).name] for e in split_tables['dev'][0]}
    fit_txt = {' '.join(norm(t['text'])) for t in split_tables['fit'][1]}
    dev_txt = [' '.join(norm(t['text'])) for t in split_tables['dev'][1]]
    audit['overlap'] = dict(fit_dev_image_sha256_intersection=len(fit_sha & dev_sha),
                            fit_unique_image_sha=len(fit_sha), dev_unique_image_sha=len(dev_sha),
                            dev_captions_with_normalized_text_in_fit=int(sum(t in fit_txt for t in dev_txt)),
                            dev_captions=len(dev_txt), raw_path_missing=0)
    for split in ('fit', 'dev'):
        images, texts, imap_rows, tmap_rows, _ = split_tables[split]
        for name, recs in [('images', images), ('texts', texts)]:
            with open(out/'data'/f'{split}_{name}.jsonl', 'w', encoding='utf-8') as f:
                for r in recs: f.write(json.dumps(r, ensure_ascii=False) + '\n')
        with open(out/'audit'/f'ID_MAP_{split}_images.tsv', 'w') as f:
            f.write('opaque_id\tsplit\tncr_image_index\tkarpathy_img_id\tfilename\timage_sha256\n')
            for r in imap_rows: f.write('\t'.join(map(str, r)) + f'\t{img_sha[r[4]]}\n')
        with open(out/'audit'/f'ID_MAP_{split}_texts.tsv', 'w') as f:
            f.write('opaque_id\tsplit\tncr_caption_index\tkarpathy_sentid\tkarpathy_img_id\n')
            for r in tmap_rows: f.write('\t'.join(map(str, r)) + '\n')
    # observed tables (FIT only)
    images, texts, _, _, cap_img = split_tables['fit']
    obs = {'Flickr00': dict(source='identity (clean NCR order, caption k -> image k//5)', map=cap_img.copy(), sha=None)}
    for cond, (path, digest, lockfile, key) in NOISE.items():
        lock = json.load(open(lockfile))['files'][key]
        if lock['path'] != path or lock['sha256'] != digest: die(f'{cond}: lock entry differs')
        if sha256(path) != digest: die(f'{cond}: noise file hash mismatch')
        m = np.load(path)
        if m.shape != (len(texts),) or m.min() < 0 or m.max() >= len(images): die(f'{cond}: bad noise map shape/range')
        mism = float(np.mean(m != cap_img))
        if abs(mism - lock['realized_mismatch']) > 1e-12: die(f'{cond}: realized mismatch {mism} != lock')
        obs[cond] = dict(source=f'{lockfile}:{key}', map=m.astype(np.int64), sha=digest, realized_mismatch=mism)
    audit['observed'] = {}
    for cond, o in obs.items():
        p = out/'data'/f'observed_{cond}.tsv'
        with open(p, 'w') as f:
            f.write('record_id\timage_id\ttext_id\n')
            for k in range(len(texts)):
                f.write(f'{opaque("rec", f"f30k_fit|{k}")}\t{images[int(o["map"][k])]["id"]}\t{texts[k]["id"]}\n')
        audit['observed'][cond] = dict(path=str(p), sha256=sha256(p), n_records=len(texts), source_noise_file_sha256=o['sha'],
                                       source=o['source'], realized_mismatch=float(np.mean(o['map'] != cap_img)))
    # evaluator-only DEV truth
    images, texts, _, _, cap_img = split_tables['dev']
    pos = [[images[int(cap_img[k])]['id'], texts[k]['id']] for k in range(len(texts))]
    tp = out/'evaluator_only'/'dev_positives.json'
    tp.write_text(json.dumps(dict(protocol_id='NR1_EVAL_TRUTH_V1', positives=pos)))
    audit['dev_truth'] = dict(path=str(tp), sha256=sha256(tp), n_pairs=len(pos))
    for split in ('fit', 'dev'):
        for name in ('images', 'texts'):
            p = out/'data'/f'{split}_{name}.jsonl'; audit.setdefault('manifests', {})[f'{split}_{name}'] = dict(path=str(p), sha256=sha256(p))
    audit['caption_text'] = dict(source='flickr_annotations_30k.csv raw (Karpathy), html.unescape applied',
                                 html_unescaped_captions=html_unescaped,
                                 captions_equal_only_after_dropping_nonascii=nonascii_only)
    audit['image_files'] = dict(n_extracted=len(img_sha), dir=str(img_dir))
    audit['status'] = 'PASS'
    (out/'audit'/'DATA_AUDIT.json').write_text(json.dumps(audit, indent=1))
    print(json.dumps({k: v for k, v in audit.items() if k not in ('ncr_files',)}, indent=1))


if __name__ == '__main__':
    main()
