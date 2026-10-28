"""Build a vocabulary-restricted GloVe 840B cache that reproduces torchtext.vocab.GloVe lookups.

torchtext parsing (vocab/vectors.py): line.rstrip().split(b" "); word = entries[0]; skip 1-dim lines;
non-UTF8 tokens skipped; stoi = {w: i for i, w in enumerate(itos)} so the LAST duplicate wins.
Lines whose dimension differs from 300 would make torchtext raise; they are skipped here and counted.
Only tokens that IN2R's init_weights can look up are kept: each vocab word and its cleaned variant.
"""
import json, sys, zipfile, hashlib
import numpy as np

ZIP = '/projects/EEG-foundation-model/yinghao/models/glove840b/glove.840B.300d.zip'
OUT = '/projects/EEG-foundation-model/yinghao/models/glove840b/glove840b_r25_subset.npz'
VOC = '/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/vocab/%s_precomp_vocab.json'

needed = set()
for d in ('f30k', 'coco', 'cc152k'):
    w2i = json.load(open(VOC % d))['word2idx']
    for w in list(w2i) + ['<mask>']:
        needed.add(w)
        c = w.replace('-', '').replace('.', '').replace("'", '')
        if '/' in c:
            c = c.split('/')[0]
        needed.add(c)

vec, n_lines, n_bad_dim, n_nonutf8, n_dup = {}, 0, 0, 0, 0
with zipfile.ZipFile(ZIP) as z:
    with z.open('glove.840B.300d.txt') as f:
        for line in f:
            n_lines += 1
            e = line.rstrip().split(b' ')
            word, e = e[0], e[1:]
            if len(e) == 1:
                continue
            if len(e) != 300:
                n_bad_dim += 1
                continue
            try:
                word = word.decode('utf-8')
            except UnicodeDecodeError:
                n_nonutf8 += 1
                continue
            if word in needed:
                if word in vec:
                    n_dup += 1
                vec[word] = np.array([float(x) for x in e], dtype=np.float32)
words = sorted(vec)
np.savez(OUT, words=np.array(words, dtype=object), vectors=np.stack([vec[w] for w in words]))
summary = dict(lines=n_lines, bad_dim_lines_skipped=n_bad_dim, nonutf8_skipped=n_nonutf8,
               duplicates_last_wins=n_dup, needed=len(needed), kept=len(words), out=OUT,
               out_sha256=hashlib.sha256(open(OUT, 'rb').read()).hexdigest())
json.dump(summary, open(sys.argv[1], 'w'), indent=1)
print(summary)
