"""R33 noise maps. New ratio 0.8 (Flickr30K, COCO) with the R25 procedure (handoff/r25_in2r_20261027/tools/make_noise.py:
IN2R data.py algorithm, a fresh np.random.RandomState(20261027) per file); reused maps are only re-hashed.
Written once, before any R33 training. No TEST file is read (train caption count is fixed by the dataset)."""
import json, hashlib, os, sys
import numpy as np

NCR = '/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data'
R25N = '/projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/noise'
OUT = '/projects/EEG-foundation-model/yinghao/outputs_v9/r33_baselines_20261102/noise'
NOISE_SEED = 20261027
N_CAPS = {'f30k_precomp': 145000, 'coco_precomp': 566435}


def official(length, im_div, ratio, rs):  # verbatim from R25 tools/make_noise.py
    t2i = np.arange(0, length) // im_div
    idx = np.arange(length)
    rs.shuffle(idx)
    n = int(ratio * length)
    sh = t2i[idx[:n]]
    rs.shuffle(sh)
    t2i[idx[:n]] = sh
    return t2i


def rec(p, name, ratio, origin):
    t2i = np.load(p)
    clean = np.arange(N_CAPS[name]) // 5
    assert t2i.shape == clean.shape, (p, t2i.shape)
    return dict(path=p, sha256=hashlib.sha256(open(p, 'rb').read()).hexdigest(), design_ratio=ratio,
                realized_mismatch=float((t2i != clean).mean()), n_captions=int(t2i.size), origin=origin)


def main():
    lock = sys.argv[1]
    assert not os.path.exists(lock), 'lock already written'
    os.makedirs(OUT, exist_ok=True)
    files = {}
    for name in ('f30k_precomp', 'coco_precomp'):
        p = f'{OUT}/{name}_0.8.npy'
        assert not os.path.exists(p)
        np.save(p, official(N_CAPS[name], 5, 0.8, np.random.RandomState(NOISE_SEED)))
        files[f'{name}_0.8'] = rec(p, name, 0.8, 'new (R33), R25 procedure, RandomState(20261027)')
    files['coco_precomp_0.6'] = rec(f'{R25N}/coco_precomp_0.6.npy', 'coco_precomp', 0.6, 'reused: R25 NOISE_LOCK.json (same procedure); used by no earlier training')
    for name in ('f30k_precomp', 'coco_precomp'):
        files[f'official_{name}_0.2'] = rec(f'{NCR}/noise_index/{name}_0.2.npy', name, 0.2, 'official NCR-data index (read only)')
    r25 = json.load(open('/home/infres/yinwang/CS_QMI/handoff/r25_in2r_20261027/NOISE_LOCK.json'))['files']
    for k, src in (('coco_precomp_0.6', 'coco_precomp_0.6'), ('official_f30k_precomp_0.2', 'official_f30k_precomp_0.2'),
                   ('official_coco_precomp_0.2', 'official_coco_precomp_0.2')):
        assert files[k]['sha256'] == r25[src]['sha256'], k
        files[k]['matches_r25_lock'] = True
    out = dict(noise_seed=NOISE_SEED, algorithm='IN2R data.py: shuffle caption ids, take int(ratio*N), permute their image ids among '
               'themselves (fixed points possible); fresh RandomState(20261027) per file (= R25 tools/make_noise.py)', files=files,
               cells={'c20': 'official_coco_precomp_0.2', 'c60': 'coco_precomp_0.6', 'c80': 'coco_precomp_0.8',
                      'f20': 'official_f30k_precomp_0.2', 'f80': 'f30k_precomp_0.8'})
    for k in ('f30k_precomp_0.8', 'coco_precomp_0.8'):
        os.chmod(files[k]['path'], 0o444)
    json.dump(out, open(lock, 'w'), indent=1)
    os.chmod(lock, 0o444)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
