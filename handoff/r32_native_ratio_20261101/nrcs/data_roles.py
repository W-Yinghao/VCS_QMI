"""NRCS-v1 data constructor: Flickr30K FIT/SELECT/AUDIT split by canonical image ID, then per-role observed noise.

This is the ONLY module that sees the clean caption->image relation of the training pool. It writes
  data/roles/{fit,select}_n{00,60}.npz : image_rows (into train_ims.npy), image_ids, caption_ids, tokens/offsets,
                                          observed_image_index  (NO clean map)
  data/roles/{audit,dev}.npz           : same fields, caption_image_index (clean evaluation map)
  data/evaluator_only/{fit,select}_clean_index.npy : quarantined clean maps (audit of realized noise only)
  data/DATA_MANIFEST.json              : sources + sha256, split/noise rules, counts, realized mismatch rates
TEST files are never opened. Python -B. Usage: data_roles.py --out <outputs>/data
"""
import argparse, hashlib, json, os, sys, time
import numpy as np

DATA = '/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data/f30k_precomp'
VOCAB = '/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/vocab/f30k_precomp_vocab.json'
GLOVE = '/projects/EEG-foundation-model/yinghao/models/glove840b/glove840b_r25_subset.npz'
SPLIT_SEED, NOISE_SEED = 20261101, 20261027
COUNTS = {'fit': 26100, 'select': 1450, 'audit': 1450}
CAPS_PER_IMAGE = 5


def sha(path, bs=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(bs), b''):
            h.update(b)
    return h.hexdigest()


def arr_sha(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.dtype.str.encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def tokenize_all(lines, vocab):
    import nltk
    w2i = vocab['word2idx']
    unk = w2i['<unk>']
    toks, offs = [], [0]
    for c in lines:
        ids = [w2i['<start>']] + [w2i.get(t, unk) for t in nltk.tokenize.word_tokenize(c.lower())] + [w2i['<end>']]
        toks.extend(ids); offs.append(len(toks))
    return np.asarray(toks, np.int32), np.asarray(offs, np.int64)


def gather_tokens(toks, offs, rows):
    lens = offs[rows + 1] - offs[rows]
    new_off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    out = np.concatenate([toks[offs[r]:offs[r + 1]] for r in rows]).astype(np.int32)
    return out, new_off


def in2r_subset_permutation(n_caps, clean_local, ratio, rs):
    """IN2R/NCR data.py rule (R25 make_noise.official) applied to a role-local caption list."""
    t2i = clean_local.copy()
    idx = np.arange(n_caps)
    rs.shuffle(idx)
    n = int(ratio * n_caps)
    sh = t2i[idx[:n]]
    rs.shuffle(sh)
    t2i[idx[:n]] = sh
    return t2i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    t0 = time.time()
    out = a.out
    os.makedirs(f'{out}/roles', exist_ok=True); os.makedirs(f'{out}/evaluator_only', exist_ok=True)
    src = {n: f'{DATA}/{n}' for n in ('train_ids.txt', 'train_caps.txt', 'train_ims.npy', 'dev_ids.txt', 'dev_caps.txt', 'dev_ims.npy')}
    src.update(vocab=VOCAB, glove=GLOVE)
    src_sha = {k: sha(p) for k, p in src.items()}
    train_ids = np.loadtxt(src['train_ids.txt'], dtype=np.int64)
    dev_ids_rows = np.loadtxt(src['dev_ids.txt'], dtype=np.int64)
    train_caps = open(src['train_caps.txt'], encoding='utf-8').read().splitlines()
    dev_caps = open(src['dev_caps.txt'], encoding='utf-8').read().splitlines()
    tr_ims = np.load(src['train_ims.npy'], mmap_mode='r'); dv_ims = np.load(src['dev_ims.npy'], mmap_mode='r')
    n_img = len(train_ids)
    assert n_img == 29000 and len(set(train_ids.tolist())) == n_img and len(train_caps) == CAPS_PER_IMAGE * n_img
    assert tr_ims.shape == (n_img, 36, 2048) and dv_ims.shape == (5070, 36, 2048) and len(dev_caps) == 5070
    # DEV: image rows repeated per caption; verify bitwise and dedupe.
    g = dev_ids_rows.reshape(-1, 5)
    assert (g == g[:, :1]).all()
    dev_rep_identical = all(np.array_equal(dv_ims[5 * i], dv_ims[5 * i + k]) for i in range(1014) for k in range(1, 5))
    assert dev_rep_identical
    dev_image_ids = g[:, 0]
    assert len(set(dev_image_ids.tolist())) == 1014 and not (set(dev_image_ids.tolist()) & set(train_ids.tolist()))
    vocab = json.load(open(VOCAB))
    toks_tr, offs_tr = tokenize_all(train_caps, vocab)
    toks_dv, offs_dv = tokenize_all(dev_caps, vocab)
    # Split by canonical image ID (row order of train_ids.txt), before any corruption.
    perm = np.random.RandomState(SPLIT_SEED).permutation(n_img)
    bounds = np.cumsum([0, COUNTS['fit'], COUNTS['select'], COUNTS['audit']])
    role_rows = {r: np.sort(perm[bounds[i]:bounds[i + 1]]) for i, r in enumerate(('fit', 'select', 'audit'))}
    man = {'version': 'NRCS-v1-20261101', 'route': 'VCS_NATIVE_RATIO_v1', 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'sources': {k: {'path': p, 'sha256': src_sha[k]} for k, p in src.items()},
           'test_files_opened': False,
           'split': {'seed': SPLIT_SEED, 'rule': 'np.random.RandomState(20261101).permutation(29000) over train_ids.txt row order; '
                     'first 26100 FIT, next 1450 SELECT, last 1450 AUDIT; all 5 captions of an image stay in its role; '
                     'rows sorted ascending within role (stable gallery order)', 'unit': 'canonical_image_id_before_corruption'},
           'noise': {'seed': NOISE_SEED, 'rule': 'IN2R data.py subset permutation (shuffle caption positions, take int(ratio*n), '
                     'permute their image indices among themselves; fixed points possible), applied separately inside FIT and inside '
                     'SELECT with a fresh np.random.RandomState(20261027) each; AUDIT and DEV never corrupted'},
           'caption_id_rule': 'f30k:<split>:cap:<row in caps file>; image_id = canonical Flickr ID from *_ids.txt',
           'tokenization': 'nltk word_tokenize(lower), ids = [<start>] + vocab(tokens; <unk> if missing) + [<end>] '
                           '(IN2R process_caption drop=False); training-time drop augmentation applied by the trainer',
           'roles': {}, 'files': {}}
    allsets = {}
    for role in ('fit', 'select', 'audit'):
        rows = role_rows[role]
        cap_rows = (rows[:, None] * CAPS_PER_IMAGE + np.arange(CAPS_PER_IMAGE)[None]).reshape(-1)
        clean_local = np.repeat(np.arange(len(rows)), CAPS_PER_IMAGE)
        tk, of = gather_tokens(toks_tr, offs_tr, cap_rows)
        img_ids = np.array([f'f30k:{v}' for v in train_ids[rows]])
        cap_ids = np.array([f'f30k:train:cap:{c}' for c in cap_rows])
        allsets[role] = (set(img_ids.tolist()), set(cap_ids.tolist()))
        base = dict(image_rows=rows.astype(np.int64), image_ids=img_ids, caption_ids=cap_ids, tokens=tk, token_offsets=of)
        info = {'images': int(len(rows)), 'captions': int(len(cap_rows)), 'image_rows_sha256': arr_sha(rows)}
        if role == 'audit':
            p = f'{out}/roles/audit.npz'
            np.savez(p, caption_image_index=clean_local.astype(np.int64), mapping_semantics='clean_evaluation_only', **base)
            man['files']['audit'] = {'path': p, 'sha256': sha(p)}
        else:
            np.save(f'{out}/evaluator_only/{role}_clean_index.npy', clean_local.astype(np.int64))
            man['files'][f'{role}_clean_index_EVALUATOR_ONLY'] = {'path': f'{out}/evaluator_only/{role}_clean_index.npy',
                                                                   'sha256': sha(f'{out}/evaluator_only/{role}_clean_index.npy')}
            info['noise'] = {}
            for noise in (0, 60):
                obs = clean_local.copy() if noise == 0 else in2r_subset_permutation(len(cap_rows), clean_local, noise / 100, np.random.RandomState(NOISE_SEED))
                p = f'{out}/roles/{role}_n{noise:02d}.npz'
                np.savez(p, observed_image_index=obs.astype(np.int64), mapping_semantics='observed_only', noise_percent=noise, **base)
                info['noise'][f'n{noise:02d}'] = {'observed_map_sha256': arr_sha(obs.astype(np.int64)),
                                                   'design_rate': noise / 100, 'realized_mismatch_rate': float((obs != clean_local).mean()),
                                                   'image_marginal_preserved': bool(np.array_equal(np.bincount(obs, minlength=len(rows)), np.bincount(clean_local, minlength=len(rows))))}
                man['files'][f'{role}_n{noise:02d}'] = {'path': p, 'sha256': sha(p)}
        man['roles'][role] = info
    # DEV (official)
    tk, of = gather_tokens(toks_dv, offs_dv, np.arange(5070))
    p = f'{out}/roles/dev.npz'
    np.savez(p, image_rows=np.arange(0, 5070, 5, dtype=np.int64), image_ids=np.array([f'f30k:{v}' for v in dev_image_ids]),
             caption_ids=np.array([f'f30k:dev:cap:{c}' for c in range(5070)]), tokens=tk, token_offsets=of,
             caption_image_index=(np.arange(5070) // 5).astype(np.int64), mapping_semantics='clean_evaluation_only')
    man['files']['dev'] = {'path': p, 'sha256': sha(p)}
    man['roles']['dev'] = {'images': 1014, 'captions': 5070, 'image_rows_rule': 'dev_ims.npy rows 0,5,...,5065 (5 identical repeats verified bitwise)'}
    allsets['dev'] = (set(f'f30k:{v}' for v in dev_image_ids), set(f'f30k:dev:cap:{c}' for c in range(5070)))
    ov = {}
    names = list(allsets)
    for i, r1 in enumerate(names):
        for r2 in names[i + 1:]:
            ov[f'{r1}|{r2}'] = [len(allsets[r1][0] & allsets[r2][0]), len(allsets[r1][1] & allsets[r2][1])]
    assert all(v == [0, 0] for v in ov.values()), ov
    man['identity_overlap_image_caption'] = ov
    man['split_manifest_sha256'] = arr_sha(np.concatenate([role_rows['fit'], role_rows['select'], role_rows['audit']]))
    man['elapsed_seconds'] = time.time() - t0
    json.dump(man, open(f'{out}/DATA_MANIFEST.json', 'w'), indent=1)
    print(json.dumps({k: man[k] for k in ('roles', 'identity_overlap_image_caption', 'split_manifest_sha256')}, indent=1))


if __name__ == '__main__':
    main()
