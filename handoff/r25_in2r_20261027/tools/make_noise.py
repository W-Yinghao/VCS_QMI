"""Fixed full-pool noise mappings with IN2R/NCR's own algorithm (data.py PrecompDataset), seeded RandomState."""
import json, hashlib, sys
import numpy as np

DATA = '/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data'
OUT = '/projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/noise'
NOISE_SEED = 20261027


def official(length, im_div, ratio, rs):
    t2i = np.arange(0, length) // im_div
    idx = np.arange(length)
    rs.shuffle(idx)
    n = int(ratio * length)
    sh = t2i[idx[:n]]
    rs.shuffle(sh)
    t2i[idx[:n]] = sh
    return t2i


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    rec = {'noise_seed': NOISE_SEED, 'algorithm': 'IN2R data.py: shuffle caption ids, take int(ratio*N), permute their image ids among themselves (fixed points possible)', 'files': {}}
    for name, n_caps, im_div in (('f30k_precomp', 145000, 5), ('coco_precomp', 566435, 5)):
        clean = np.arange(n_caps) // im_div
        for ratio in (0.2, 0.4, 0.6):
            t2i = official(n_caps, im_div, ratio, np.random.RandomState(NOISE_SEED))
            p = f'{OUT}/{name}_{ratio}.npy'
            np.save(p, t2i)
            rec['files'][f'{name}_{ratio}'] = dict(path=p, sha256=hashlib.sha256(open(p, 'rb').read()).hexdigest(),
                                                  design_ratio=ratio, realized_mismatch=float((t2i != clean).mean()))
        for ratio in (0.2, 0.5):
            p = f'{DATA}/noise_index/{name}_{ratio}.npy'
            t2i = np.load(p)
            rec['files'][f'official_{name}_{ratio}'] = dict(path=p, sha256=hashlib.sha256(open(p, 'rb').read()).hexdigest(),
                                                           design_ratio=ratio, realized_mismatch=float((t2i != clean).mean()))
    json.dump(rec, open(sys.argv[1], 'w'), indent=1)
    print(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main()
