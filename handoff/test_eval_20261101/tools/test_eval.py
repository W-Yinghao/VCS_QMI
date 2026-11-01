"""One-shot TEST evaluation of DEV-selected checkpoints listed in frozen manifests (GPU job).

Usage: test_eval.py --host in2r|sps [--dataset f30k|coco|cc152k|all] [--shard i/n] [--manifest PATH ...]
  Default manifests = every 'manifest_frozen' entry of TEST_ACCESS_LOG.jsonl (each re-verified by sha256).
  Units whose results/<checkpoint sha256[:16]>.json already holds status COMPLETE for the same sha256 are skipped, so the
  same command can be re-run after preemption or after a new increment.
Order per job: verify manifests + eval code -> append a 'test_read' line to TEST_ACCESS_LOG.jsonl -> stage DEV+TEST files of
the needed datasets to node-local tmp -> per unit: re-hash the checkpoint bytes (must equal the manifest), DEV recompute
(r25_co_train.validate path, shard 100) and gate (max |diff| <= 0.1 pp over the six recalls), then TEST (evalrank path, shard 1000).
A unit failing the gate still gets its TEST numbers written, flagged gate=FAIL (aggregate_test.py keeps it out of the tables).
"""
import os, sys, json, io, hashlib, time, argparse, shutil, datetime, fcntl

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(HERE, 'TEST_ACCESS_LOG.jsonl')
RES = os.path.join(HERE, 'results')
TOL_PP = 0.1
KEYS = ('i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10')
SPLITS = {'f30k_precomp': ('dev', 'test'), 'coco_precomp': ('dev', 'testall'), 'cc152k_precomp': ('dev', 'test')}
DN = {'f30k': 'f30k_precomp', 'coco': 'coco_precomp', 'cc152k': 'cc152k_precomp'}


def sha_file(p, bs=1 << 22):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(bs), b''):
            h.update(b)
    return h.hexdigest()


def append_log(rec):
    with open(LOG, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(rec) + '\n'); f.flush(); os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)


def frozen_manifests():
    out = []
    for line in open(LOG):
        ev = json.loads(line)
        if ev.get('event') == 'manifest_frozen':
            out.append((ev['manifest'], ev['sha256']))
    return out


def result_path(sha):
    return os.path.join(RES, sha[:16] + '.json')


def done(u):
    p = result_path(u['checkpoint_sha256'])
    if not os.path.exists(p):
        return False
    r = json.load(open(p))
    return r.get('status') == 'COMPLETE' and r.get('checkpoint_sha256') == u['checkpoint_sha256']


def stage(data_root, dn, dest, dev_only=False):
    os.makedirs(os.path.join(dest, dn), exist_ok=True)
    ext = 'tsv' if dn == 'cc152k_precomp' else 'txt'
    info = {}
    for s in (SPLITS[dn][:1] if dev_only else SPLITS[dn]):
        for f in (f'{s}_caps.{ext}', f'{s}_ims.npy'):
            src, dst = os.path.join(data_root, dn, f), os.path.join(dest, dn, f)
            shutil.copyfile(src, dst)
            assert os.stat(src).st_size == os.stat(dst).st_size
            info[f'{dn}/{f}'] = dict(bytes=os.stat(dst).st_size, sha256=sha_file(dst))
    return info


def recall_dict(r, ri):
    d = dict(zip(KEYS, map(float, (r[0], r[1], r[2], ri[0], ri[1], ri[2]))))
    d['rsum'] = float(sum(d[k] for k in KEYS))
    d['i2t_medr'], d['t2i_medr'] = float(r[3]), float(ri[3])
    return d


def gate(logged, recomputed):
    diffs = {k: recomputed[k] - logged[k] for k in KEYS + ('rsum',)}
    mx = max(abs(diffs[k]) for k in KEYS)
    # 1e-9: float rounding (one CC152K query = exactly 0.1 pp)
    ok = mx <= TOL_PP + 1e-9 and int(recomputed['n_images']) == int(logged['n_images']) and int(recomputed['n_captions']) == int(logged['n_captions'])
    return diffs, mx, ok


# ----------------------------------------------------------------------------------------------- IN2R (VSE) units
class In2r:
    def __init__(self, dp):
        sys.path.insert(0, os.path.join(HERE, 'eval_code'))
        import torch
        from data import get_dataset, get_loader
        from vse import VSE
        from vocab import deserialize_vocab
        from evaluation import i2t, t2i, encode_data, shard_attn_scores
        self.torch, self.get_dataset, self.get_loader, self.VSE = torch, get_dataset, get_loader, VSE
        self.deserialize_vocab, self.i2t, self.t2i, self.encode_data, self.shard = deserialize_vocab, i2t, t2i, encode_data, shard_attn_scores
        self.dp, self.vocabs, self.loaders = dp, {}, {}
        self.test_split, self.nfolds = None, 5   # smoke: test_split='dev', nfolds=1 (no TEST file is opened)

    def vocab(self, opt):
        k = (opt.vocab_path, opt.data_name)
        if k not in self.vocabs:
            v = self.deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name))
            v.add_word('<mask>')  # as r25_co_train.main / evaluation.evalrank
            self.vocabs[k] = v
        return self.vocabs[k]

    def loader(self, opt, split):
        k = (opt.data_name, split, opt.batch_size, opt.vocab_path)
        if k not in self.loaders:
            caps, ims = self.get_dataset(self.dp, opt.data_name, split)
            self.loaders[k] = self.get_loader(caps, ims, self.vocab(opt), split, opt.batch_size, 0)
        return self.loaders[k]

    def enc(self, m, loader):
        import numpy as np
        ie, ce, cl = self.encode_data(m, loader, 10 ** 9, logging=lambda *a, **k: None)
        return np.asarray(ie), np.asarray(ce), list(cl)

    def evaluate(self, u, ckpt):
        import numpy as np
        torch = self.torch
        opt = ckpt['opt']
        opt.data_path, opt.workers = self.dp, 0
        per = 1 if opt.data_name == 'cc152k_precomp' else 5
        vocab = self.vocab(opt)
        opt.vocab_size = len(vocab)
        models = []
        for key in ('model_A', 'model_B'):
            m = self.VSE(opt, word2idx=vocab.word2idx)
            m.load_state_dict(ckpt[key]); m.val_start(); models.append(m)
        out = {}
        with torch.no_grad():
            # DEV recompute (r25_co_train.validate)
            dl = self.loader(opt, 'dev')
            sims, peers = 0, {}
            for ind, m in enumerate(models):
                ie, ce, cl = self.enc(m, dl)
                ie = np.array([ie[i] for i in range(0, len(ie), per)])
                s = self.shard(m, ie, ce, cl, opt, shard_size=100)
                peers[f'peer{ind}_rsum'] = float(sum(self.i2t(ie.shape[0], s, per)[:3]) + sum(self.t2i(ie.shape[0], s, per)[:3]))
                sims = sims + s
            sims = sims / 2
            d = recall_dict(self.i2t(ie.shape[0], sims, per), self.t2i(ie.shape[0], sims, per))
            d.update(peers, n_images=int(ie.shape[0]), n_captions=int(ce.shape[0]))
            out['dev_recomputed'] = d
            del sims
            # TEST (evaluation.evalrank)
            split = self.test_split or ('testall' if opt.data_name == 'coco_precomp' else 'test')
            tl = self.loader(opt, split)
            embs = [self.enc(m, tl) for m in models]
            if opt.data_name != 'coco_precomp':
                out['test'] = self.full(models, embs, opt, per)
            else:
                out['test_5k'] = self.full(models, embs, opt, 5)
                folds = []
                for i in range(self.nfolds):
                    sl = slice(i * 5000, (i + 1) * 5000)
                    sims, peers = 0, {}
                    for ind, (m, (ie, ce, cl)) in enumerate(zip(models, embs)):
                        iei = ie[sl][::5]
                        s = self.shard(m, iei, ce[sl], cl[sl], opt, shard_size=1000)
                        peers[f'peer{ind}_rsum'] = float(sum(self.i2t(iei.shape[0], s, 5)[:3]) + sum(self.t2i(iei.shape[0], s, 5)[:3]))
                        sims = sims + s
                    sims = sims / 2
                    f = recall_dict(self.i2t(iei.shape[0], sims, 5), self.t2i(iei.shape[0], sims, 5))
                    f.update(peers, n_images=int(iei.shape[0]), n_captions=int(ce[sl].shape[0]))
                    folds.append(f)
                mean = {k: float(np.mean([f[k] for f in folds])) for k in KEYS + ('rsum', 'peer0_rsum', 'peer1_rsum')}
                mean.update(n_folds=self.nfolds, n_images_per_fold=1000, n_captions_per_fold=5000)
                out['test_5fold1k'] = dict(mean=mean, folds=folds)
        del models
        torch.cuda.empty_cache()
        return out

    def full(self, models, embs, opt, per):
        import numpy as np
        sims, peers = 0, {}
        for ind, (m, (ie, ce, cl)) in enumerate(zip(models, embs)):
            iei = np.array([ie[i] for i in range(0, len(ie), per)])
            s = self.shard(m, iei, ce, cl, opt, shard_size=1000)
            peers[f'peer{ind}_rsum'] = float(sum(self.i2t(iei.shape[0], s, per)[:3]) + sum(self.t2i(iei.shape[0], s, per)[:3]))
            sims = sims + s
        sims = sims / 2
        d = recall_dict(self.i2t(iei.shape[0], sims, per), self.t2i(iei.shape[0], sims, per))
        d.update(peers, n_images=int(iei.shape[0]), n_captions=int(ce.shape[0]))
        return d


# ----------------------------------------------------------------------------------------------- SPS (SGRAF) unit
class Sps:
    def __init__(self, dp):
        sys.path.insert(0, os.path.join(HERE, 'eval_code_sps'))
        import torch
        from vocab import deserialize_vocab
        from model import SGRAF
        from data import get_dataset, get_loader
        from evaluation import encode_data, shard_attn_scores, i2t, t2i
        self.torch, self.deserialize_vocab, self.SGRAF, self.get_dataset, self.get_loader = torch, deserialize_vocab, SGRAF, get_dataset, get_loader
        self.encode_data, self.shard, self.i2t, self.t2i, self.dp = encode_data, shard_attn_scores, i2t, t2i, dp
        self.test_split, self.nfolds = None, 5

    def evaluate(self, u, ckpt):
        import numpy as np
        torch = self.torch
        opt = ckpt['opt']
        opt.data_path, opt.workers = self.dp, 0
        vocab = self.deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name))
        opt.vocab_size = len(vocab)
        out = {}
        for split, shard, key in (('dev', 100, 'dev_recomputed'), (self.test_split or 'test', 1000, 'test')):
            caps, ims = self.get_dataset(self.dp, opt.data_name, split, vocab)
            loader = self.get_loader(caps, ims, split, opt.batch_size, 0)
            if split == 'dev':  # full DEV as tools/sps_fulldev_eval.py (native truncates DEV to 1000 rows)
                ds = loader.dataset
                ds.length = len(ds.captions)
                ds.t2i_index = np.arange(0, ds.length) // ds.im_div
            sims, peers = 0, {}
            for ind, k in enumerate(('model_A', 'model_B')):
                m = self.SGRAF(opt); m.load_state_dict(ckpt[k]); m.val_start()
                with torch.no_grad():
                    ie, ce, cl = self.encode_data(m, loader, 10 ** 9, logging=lambda *a, **kw: None)
                ie = np.array([ie[i] for i in range(0, len(ie), 5)])
                s = self.shard(m, ie, ce, cl, opt, shard_size=shard)
                peers[f'peer{ind}_rsum'] = float(sum(self.i2t(ie.shape[0], s, 5)[:3]) + sum(self.t2i(ie.shape[0], s, 5)[:3]))
                sims = sims + s
                del m
            sims = sims / 2
            d = recall_dict(self.i2t(ie.shape[0], sims, 5), self.t2i(ie.shape[0], sims, 5))
            d.update(peers, n_images=int(ie.shape[0]), n_captions=int(ce.shape[0]))
            out[key] = d
        torch.cuda.empty_cache()
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', required=True, choices=('in2r', 'sps'))
    ap.add_argument('--dataset', default='all', choices=('all', 'f30k', 'coco', 'cc152k'))
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--manifest', nargs='*')
    ap.add_argument('--smoke-dev', action='store_true', help='code check: first unit per dataset, DEV used in place of TEST, '
                    'no TEST file staged or read, no access-log entry, results to results_smoke/')
    a = ap.parse_args()
    global RES
    if a.smoke_dev:
        RES = os.path.join(HERE, 'results_smoke')
    fm = frozen_manifests()
    sel = fm if not a.manifest else [x for x in fm if x[0] in a.manifest]
    assert sel, 'no frozen manifest selected'
    units, mans = [], {}
    for mp, sha in sel:
        assert sha_file(mp) == sha, f'manifest changed after freezing: {mp}'
        man = json.load(open(mp))
        mans[mp] = man
        ec = man['eval_code'] if a.host == 'in2r' else man['eval_code_sps']
        for f, s in ec['sha256'].items():
            assert sha_file(os.path.join(ec['dir'], f)) == s, f'eval code changed: {f}'
        for u in man['units']:
            if u['host'] == a.host and (a.dataset == 'all' or u['dataset'] == a.dataset):
                units.append(dict(u, manifest=mp, manifest_sha256=sha))
    i, n = map(int, a.shard.split('/'))
    units = [u for k, u in enumerate(sorted(units, key=lambda x: x['unit_id'])) if k % n == i]
    todo = [u for u in units if not done(u)]
    if a.smoke_dev:
        seen, t2 = set(), []
        for u in todo:
            if u['dataset'] not in seen:
                seen.add(u['dataset']); t2.append(u)
        todo = t2
    print(f'{len(units)} units in selection, {len(todo)} to evaluate', flush=True)
    if not todo:
        return
    data_root = mans[todo[0]['manifest']]['data']['root']
    dns = sorted({DN[u['dataset']] for u in todo})
    files = [os.path.join(data_root, dn, f'{s}_{x}') for dn in dns for s in SPLITS[dn][1:]
             for x in (('caps.tsv' if dn == 'cc152k_precomp' else 'caps.txt'), 'ims.npy')]
    import torch
    gpu = torch.cuda.get_device_name(0)
    jid = os.environ.get('SLURM_JOB_ID')
    if a.smoke_dev:
        files = []
    else:
      append_log(dict(event='test_read', utc=datetime.datetime.utcnow().isoformat() + 'Z', job=jid,
                    restart=os.environ.get('SLURM_RESTART_COUNT'), node=os.uname().nodename, gpu=gpu, host_code=a.host,
                    manifests=sorted({u['manifest'] for u in todo}), units=[u['unit_id'] for u in todo], files_read=files))
    tmp = os.path.join(os.environ.get('TMPDIR') or '/tmp', f'te_{jid}_{a.host}_{os.getpid()}')
    os.makedirs(RES, exist_ok=True)
    try:
        staged = {}
        for dn in dns:
            staged.update(stage(data_root, dn, tmp, dev_only=a.smoke_dev))
        ev = In2r(tmp) if a.host == 'in2r' else Sps(tmp)
        if a.smoke_dev:
            ev.test_split, ev.nfolds = 'dev', 1
        env = dict(gpu=gpu, node=os.uname().nodename, job=jid, torch=torch.__version__, cuda=torch.version.cuda,
                   cudnn=torch.backends.cudnn.version(), matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                   cudnn_tf32=torch.backends.cudnn.allow_tf32)
        for u in todo:
            t0 = time.time()
            raw = open(u['checkpoint'], 'rb').read()
            sha = hashlib.sha256(raw).hexdigest()
            rec = dict(unit_id=u['unit_id'], checkpoint=u['checkpoint'], checkpoint_sha256=u['checkpoint_sha256'], manifest=u['manifest'],
                       manifest_sha256=u['manifest_sha256'], checkpoint_epoch=u['checkpoint_epoch'], logged_dev=u['logged_dev'])
            if sha != u['checkpoint_sha256']:
                rec.update(status='CHECKPOINT_CHANGED', sha_now=sha)
                json.dump(rec, open(result_path(u['checkpoint_sha256']), 'w'), indent=1)
                print('CHECKPOINT CHANGED, not evaluated:', u['unit_id'], flush=True)
                continue
            ckpt = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
            del raw
            assert int(ckpt['epoch']) == u['checkpoint_epoch']
            out = ev.evaluate(u, ckpt)
            diffs, mx, ok = gate(u['logged_dev'], out['dev_recomputed'])
            rec.update(out, dev_diff_recomputed_minus_logged=diffs, dev_max_abs_recall_diff_pp=mx, dev_gate='PASS' if ok else 'FAIL',
                       dev_gate_tolerance_pp=TOL_PP, env=env, staged_files={k: v for k, v in staged.items() if k.startswith(DN[u['dataset']])},
                       seconds=round(time.time() - t0, 1), status='COMPLETE', utc=datetime.datetime.utcnow().isoformat() + 'Z')
            p = result_path(u['checkpoint_sha256'])
            json.dump(rec, open(p + '.tmp', 'w'), indent=1)
            os.replace(p + '.tmp', p)
            t = rec.get('test') or rec['test_5fold1k']['mean']
            print(json.dumps(dict(unit=u['unit_id'], dev_logged=round(u['logged_dev']['rsum'], 3), dev_now=round(out['dev_recomputed']['rsum'], 3),
                                  maxdiff=round(mx, 4), gate=rec['dev_gate'], test_rsum=round(t['rsum'], 2), s=rec['seconds'])), flush=True)
            del ckpt
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
