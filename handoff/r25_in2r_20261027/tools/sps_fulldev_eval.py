"""Offline full-DEV (1,014 images / 5,070 captions) evaluation of every saved SPS co-training checkpoint.

Uses SPS's own encode_data / shard_attn_scores / i2t / t2i and the native peer-averaged similarity
(train.validate). Native SPS validates on the first 1,000 caption rows (200 images); that internal
check is left untouched and is NOT used for R25 selection. Best = max full-DEV rSum (earliest tie).
"""
import os, sys, glob, json, re
import numpy as np
import torch

sys.path.insert(0, '/home/infres/yinwang/CS_QMI/handoff/r25_in2r_20261027/sps_code')
from vocab import deserialize_vocab
from model import SGRAF
from data import get_dataset, get_loader
from evaluation import encode_data, shard_attn_scores, i2t, t2i


def main(run_dir):
    ckpts = sorted(glob.glob(f'{run_dir}/checkpoint_*.pth.tar'), key=lambda p: int(re.findall(r'_(\d+)\.pth', p)[0]))
    opt = torch.load(ckpts[0], weights_only=False)['opt']
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, "%s_vocab.json" % opt.data_name))
    opt.vocab_size = len(vocab)
    caps, ims = get_dataset(opt.data_path, opt.data_name, "dev", vocab)
    loader = get_loader(caps, ims, "dev", opt.batch_size, 0)
    ds = loader.dataset
    ds.length = len(ds.captions)  # full DEV (native truncates to 1000 rows)
    ds.t2i_index = np.arange(0, ds.length) // ds.im_div
    per = 5
    rows = []
    for p in ckpts:
        ck = torch.load(p, weights_only=False)
        sims = 0
        for key in ('model_A', 'model_B'):
            m = SGRAF(opt); m.load_state_dict(ck[key]); m.val_start()
            with torch.no_grad():
                ie, ce, cl = encode_data(m, loader)
            ie = np.array([ie[i] for i in range(0, len(ie), per)])
            sims = sims + shard_attn_scores(m, ie, ce, cl, opt, shard_size=100)
        sims = sims / 2
        r = i2t(ie.shape[0], sims, per); ri = t2i(ie.shape[0], sims, per)
        row = dict(epoch=int(ck['epoch']), i2t_r1=r[0], i2t_r5=r[1], i2t_r10=r[2], t2i_r1=ri[0], t2i_r5=ri[1], t2i_r10=ri[2],
                   rsum=float(sum(r[:3]) + sum(ri[:3])), n_images=int(ie.shape[0]), n_captions=int(ce.shape[0]), native_best_rsum=float(ck.get('best_rsum', -1)))
        rows.append(row); print(row, flush=True)
    best = max(rows, key=lambda x: (x['rsum'], -x['epoch']))
    json.dump({'rows': rows, 'best': best, 'rule': 'max full-DEV rSum, earliest tie'}, open(f'{run_dir}/FULLDEV.json', 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1])
