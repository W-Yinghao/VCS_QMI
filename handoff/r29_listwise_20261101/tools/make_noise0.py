#!/usr/bin/env python3
"""Write the 0% (identity) noise mapping for Flickr30K train in the format of the R25 noise files
(np.int64 array, length = #train captions, entry k = observed image index of caption k). Identity = data.py's own
t2i_index = arange(N) // 5, i.e. the original pairing without artificial shuffling. Reads only train_caps.txt (line count)
and the R25 0.6 file (dtype/shape reference). With --noise_ratio 0.0 data.py does not load the file (it already uses
arange(N)//5); the unit runner checks that this file equals that mapping, so the 0% cell is documented in the same format."""
import json, hashlib, numpy as np, os, time
SRC = '/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data/f30k_precomp/train_caps.txt'
REF = '/projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/noise/f30k_precomp_0.6.npy'
OUT = '/projects/EEG-foundation-model/yinghao/outputs_v9/r29_listwise_20261101/noise/f30k_precomp_0.0.npy'
n = sum(1 for _ in open(SRC))
ref = np.load(REF, mmap_mode='r')
assert ref.shape == (n,) and ref.dtype == np.int64
a = (np.arange(n) // 5).astype(np.int64)
assert not os.path.exists(OUT) or np.array_equal(np.load(OUT), a)
np.save(OUT, a)
b = np.load(OUT)
assert b.dtype == ref.dtype and b.shape == ref.shape and np.array_equal(b, a) and (b != a).sum() == 0
info = dict(file=OUT, sha256=hashlib.sha256(open(OUT, 'rb').read()).hexdigest(), n_captions=n, dtype=str(b.dtype),
            rule='identity: t2i_index[k] = k // 5 (original Flickr30K pairing, no artificial shuffling)',
            reference_format=REF, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), test_files_read=False)
json.dump(info, open(OUT.replace('.npy', '.json'), 'w'), indent=1)
print(json.dumps(info))
