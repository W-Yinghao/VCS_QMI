"""R27-A1X CPU gate: dataset ID relations for COCO 40% / CC152K (no GPU, no TEST). Writes OUT.json.
Usage: r27x_cpu_check.py OUT.json CELLS_A1X.json DATA_PATH   (DATA_PATH holds only train/dev files per dataset)

Per dataset (coco_precomp, cc152k_precomp; f30k_precomp as the host-rule reference):
  host_relation   host rule im_div = 5 if n_images != n_captions else 1 (data.PrecompDataset) gives n_caps == im_div * n_imgs;
                  COCO im_div 5, CC152K im_div 1, Flickr 5; get_dataset captions == independent parse of the caption file
  observed_map    warmup dataset (the one eval_train / the trainer use): observed t2i_index == noise file (COCO) or
                  arange (CC152K, noise forced 0); true _t2i_index == arange // im_div; all ids < n_imgs
  split_ids       R27 get_loader hook on a fixed random split: labeled/unlabeled r27_gidx disjoint and covering;
                  ds.captions[j] == captions[gidx[j]] and ds.t2i_index[j] == observed[gidx[j]] for EVERY j
  queue_ids       CPU MemoryQueue (author class) fed by enqueue_with_ids with real labeled-split IDs (l_gidx, l_obs) over
                  several wrap-arounds: every slot's (cap, img) equals the pair encoded in its feature row;
                  img == observed[cap]; cap < n_caps, img < n_imgs; search atoms: i2t = caption IDs, t2i = observed image IDs
  tokens          Ctx.tokens(cid) == process_caption(vocab, captions[cid], drop=False) for random caption IDs
  rows_np         Ctx.rows_np(idx) == per-row memmap reads (bitwise), incl. the last image row
Per cell: prefix_queues (R26 prefix: epoch 0, refiners present, no ID buffer -> all slots -1 -> search None), prefix opt
matches the cell (data_name, seed, noise file), noise file == the one R26 H0 used for the cell (RUN json).
"""
import sys, os, json, hashlib, glob, csv
import numpy as np
import torch

import r25_run
import r25_co_train as R
import r27_co_train as R27
import r27_interface as I
from r27_interface import Ctx
from data import get_dataset, process_caption
from vocab import deserialize_vocab
import co_train

res = {}
R26O = '/projects/EEG-foundation-model/yinghao/outputs_v9/r26_dist_20261027/runs'
VOCAB = '/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/vocab'


def ok(name, cond, **info):
    res[name] = dict(pass_=bool(cond), **info)
    print(name, 'PASS' if cond else 'FAIL', json.dumps(info, default=float)[:2500], flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def raw_captions(dpath, name):
    if name == 'cc152k_precomp':
        with open(os.path.join(dpath, name, 'train_caps.tsv')) as f:
            return [row[1].strip() for row in csv.reader(f, delimiter='\t')]
    return [l.strip() for l in open(os.path.join(dpath, name, 'train_caps.txt'))]


def stub_queue(Qs, d):
    MQ = co_train.MemoryQueue
    q = MQ.__new__(MQ)
    q.queue_size, q.dim, q.ptr, q.is_full = Qs, d, 0, False
    q.img_queue = torch.zeros(Qs, d); q.txt_queue = torch.zeros(Qs, d)
    return q


def check_dataset(dpath, name, noise_file, noise_ratio, expect_div):
    tag = name.split('_')[0]
    caps, ims = get_dataset(dpath, name, 'train')
    n_caps, n_imgs = len(caps), ims.shape[0]
    div = 5 if n_imgs != n_caps else 1
    raw = raw_captions(dpath, name)
    ok(f'{tag}_host_relation', div == expect_div and n_caps == div * n_imgs and raw == caps,
       n_caps=n_caps, n_imgs=n_imgs, im_div=div, expected_div=expect_div, captions_equal_independent_parse=raw == caps)

    vocab = deserialize_vocab(os.path.join(VOCAB, f'{name}_vocab.json')); vocab.add_word('<mask>')
    Ctx.captions = Ctx.images = Ctx.vocab = None; Ctx.tok = {}; Ctx.gather_mode = None
    opt = r25_run.parse(['--data_path', dpath, '--data_name', name, '--noise_file', noise_file, '--noise_ratio', str(noise_ratio),
                         '--output_dir', os.environ.get('TMPDIR', '/tmp')])
    R27.install()
    noisy_loader, data_size, clean_labels = R.get_loader(caps, ims, vocab, 'warmup', 128, 0, opt.noise_ratio, opt.noise_file)
    ds = noisy_loader.dataset
    obs = np.asarray(ds.t2i_index, dtype=np.int64); true = np.asarray(ds._t2i_index, dtype=np.int64)
    if opt.noise_ratio:
        ref = np.load(noise_file).astype(np.int64)
        obs_ok = np.array_equal(obs, ref)
    else:
        obs_ok = np.array_equal(obs, np.arange(n_caps) // div)
    noisy = float((obs != true).mean())
    ok(f'{tag}_observed_map', obs_ok and np.array_equal(true, np.arange(n_caps) // div) and obs.min() >= 0 and obs.max() < n_imgs
       and data_size == n_caps and Ctx.captions is caps and Ctx.images is ims,
       noise_ratio_opt=opt.noise_ratio, observed_matches_source=obs_ok, injected_mismatch_frac=noisy,
       obs_max=int(obs.max()), n_imgs=n_imgs, ctx_is_train=Ctx.captions is caps)

    g = np.random.default_rng(7)
    prob = g.random(n_caps)
    pred = prob > 0.45
    lt, ut = R.get_loader(caps, ims, vocab, 'train', 128, 0, opt.noise_ratio, opt.noise_file, pred=pred, prob=prob)
    lg, ug = lt.dataset.r27_gidx, ut.dataset.r27_gidx
    cover = np.array_equal(np.sort(np.concatenate([lg, ug])), np.arange(n_caps)) and len(np.intersect1d(lg, ug)) == 0
    good = cover
    for d, gi in ((lt.dataset, lg), (ut.dataset, ug)):
        good &= all(d.captions[j] == caps[gi[j]] for j in range(len(d)))
        good &= bool(np.array_equal(np.asarray(d.t2i_index, dtype=np.int64), obs[gi]))
        good &= bool(np.array_equal(np.asarray(d._t2i_index, dtype=np.int64), true[gi]))
    ok(f'{tag}_split_ids', good, n_labeled=len(lg), n_unlabeled=len(ug), cover_disjoint=cover)

    # queue ID semantics with real labeled-split IDs, several wraps of a small author MemoryQueue
    l_gidx, l_obs = lg, np.asarray(lt.dataset.t2i_index, dtype=np.int64)
    q = stub_queue(4096, 3)
    bad, written = 0, 0
    for step in range(200):
        bi = g.choice(len(lg), 128, replace=False)
        hc = bi[g.random(128) > 0.3]
        cap, img = l_gidx[hc], l_obs[hc]
        f = torch.from_numpy(np.stack([cap, img, np.full(len(hc), step)], 1).astype(np.float64)).float()
        I.enqueue_with_ids(q, f, f, cap, img, 3, step, 1)
        written += len(hc)
    ids = q.r27_ids
    capq, imgq = ids['cap'].numpy(), ids['img'].numpy()
    bad += int((capq.astype(np.float32) != q.img_queue[:, 0].numpy()).sum()) if n_caps < 2 ** 24 else 0
    bad += int((imgq.astype(np.float32) != q.img_queue[:, 1].numpy()).sum())
    rel = bool((imgq == obs[capq]).all()) and capq.max() < n_caps and imgq.max() < n_imgs and capq.min() >= 0
    q.img_queue.copy_(torch.nn.functional.normalize(torch.randn(4096, 3), dim=1)); q.txt_queue.copy_(q.img_queue)
    Si, _ = I.search(q, torch.nn.functional.normalize(torch.randn(16, 3), dim=1), 'i2t', 5)
    St, _ = I.search(q, torch.nn.functional.normalize(torch.randn(16, 3), dim=1), 't2i', 5)
    atoms = (bool(np.isin(Si['cols'].numpy(), capq).all()) and bool((Si['cols'] < n_caps).all())
             and bool(np.isin(St['cols'].numpy(), imgq).all()) and bool((St['cols'] < n_imgs).all())
             and bool((St['img'] == torch.as_tensor(obs)[St['cap']]).all()))
    ok(f'{tag}_queue_ids', bad == 0 and rel and atoms and q.is_full, wraps=written // 4096, misaligned=bad, img_is_observed_of_cap=rel,
       search_atoms_valid=atoms, injected_mismatch_slots=int((obs[capq] != true[capq]).sum()))

    cids = g.choice(n_caps, 300, replace=False)
    tk = all(torch.equal(Ctx.tokens(int(c)), process_caption(vocab, caps[int(c)], False).long()) and len(Ctx.tokens(int(c))) >= 2 for c in cids)
    ok(f'{tag}_tokens', tk, n=len(cids))

    idx = np.unique(np.concatenate([g.choice(n_imgs, 200, replace=False), [0, n_imgs - 1]]))
    a = Ctx.rows_np(idx)
    b = np.stack([np.array(ims[i]) for i in idx])
    ok(f'{tag}_rows_np', a.dtype == np.float32 and a.flags['C_CONTIGUOUS'] and a.tobytes() == b.tobytes(), n=len(idx), shape=list(a.shape))
    Ctx.captions = Ctx.images = Ctx.vocab = None; Ctx.tok = {}


def check_cells(cells):
    out, good = {}, True
    for cell, c in cells.items():
        st = torch.load(c['prefix'], map_location='cpu', weights_only=False)
        o = st['opt']
        info = dict(epoch=st['epoch'], data_name=o['data_name'], seed=o['seed'], noise_file=o['noise_file'], noise_ratio=o['noise_ratio'])
        g = st['epoch'] == 0 and o['data_name'] == c['data_name'] and o['seed'] == c['seed'] and o['noise_file'] == c['noise_file']
        for k in ('model_A', 'model_B'):
            s = st[k]
            q = stub_queue(s['queue']['img'].shape[0], s['queue']['img'].shape[1])
            q.img_queue.copy_(s['queue']['img']); q.txt_queue.copy_(s['queue']['txt']); q.ptr, q.is_full = s['queue']['ptr'], s['queue']['is_full']
            had = I.load_ids(q, s.get('r27_queue_ids'))
            n, v = I.usable(q)
            S, qs = I.search(q, torch.nn.functional.normalize(torch.randn(4, q.dim), dim=1), 'i2t', 5)
            info[k] = dict(ptr=int(q.ptr), is_full=bool(q.is_full), size=q.queue_size, has_ids=had, usable=int(v.sum()), filled=n,
                           refiners='i2t_refiner' in s and 't2i_refiner' in s, search_none=S is None)
            g &= (not had) and int(v.sum()) == 0 and S is None and info[k]['refiners'] and q.queue_size == 65536
        runs = sorted(glob.glob(f"{R26O}/{cell}/H0/RUN_*.json"))
        h0 = [json.load(open(r))['opt'] for r in runs]
        info['r26_H0_runs'] = len(h0)
        info['r26_H0_same_noise_seed'] = (bool(h0) and all(x['noise_file'] == c['noise_file'] and x['seed'] == c['seed'] and x['data_name'] == c['data_name']
                                                           for x in h0) and any(x['resume'] == c['prefix'] for x in h0))
        g &= info['r26_H0_same_noise_seed']
        if c['noise_file'] != 'none':
            info['noise_sha256'] = sha(c['noise_file'])
        info['prefix_sha256'] = sha(c['prefix'])
        out[cell] = info
        good &= g
        del st
    ok('prefix_queues', good, note='R26 prefixes are pre-ID: every slot cap=-1, excluded until overwritten (peer A has no candidates in co-train epoch 1)', **out)


if __name__ == '__main__':
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 2)))
    out_path, cells_path, dpath = sys.argv[1:4]
    cells = json.load(open(cells_path))
    nf = {c['data_name']: (c['noise_file'], c['noise_ratio']) for c in cells.values()}
    check_dataset(dpath, 'coco_precomp', *nf['coco_precomp'], expect_div=5)
    check_dataset(dpath, 'cc152k_precomp', *nf['cc152k_precomp'], expect_div=1)
    check_dataset(dpath, 'f30k_precomp', '/projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/noise/f30k_precomp_0.4.npy', 0.4, expect_div=5)
    check_cells(cells)
    here = os.path.dirname(os.path.abspath(__file__))
    res['code_sha256'] = {os.path.basename(f): sha(f) for f in sorted(glob.glob(os.path.join(here, '*.py')))}
    res['all_pass'] = all(v['pass_'] for v in res.values() if isinstance(v, dict) and 'pass_' in v)
    json.dump(res, open(out_path, 'w'), indent=1, default=float)
    print('ALL_PASS' if res['all_pass'] else 'SOME_FAIL')
    sys.exit(0 if res['all_pass'] else 1)
