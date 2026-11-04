"""Layout-free TEST re-read of the units already evaluated by tools/test_eval.py (GPU job; user authorisation 2026-11-04
~00:15UTC "TEST也再读一次"; reason: R44_AUDIT.md, native IN2R eval centres embeddings per sequential batch of 128 rows,
row r = image r//5 and caption r, so true pairs share the batch mean).

Same frozen manifests, same DEV-selected checkpoints (sha256 re-verified), no new selection. Per unit:
  1. native DEV recompute through the unmodified test_eval.In2r.evaluate path, gated with test_eval.gate (0.1 pp) vs logged DEV;
  2. native TEST (same call) compared with the recorded results/<sha16>.json TEST numbers (mismatch reported, old file untouched);
  3. raw encoder outputs (VSE.img_enc / txt_enc, i.e. forward_emb without the batch-mean subtraction; bound per instance,
     eval_code not modified) -> centring variants, then l2norm (vse.l2norm) and the host similarity (forward_sim = dot):
       native_rebuilt : per sequential loader batch (batch_size rows) mean, fp32 -- check that the raw path reproduces native;
       global         : one mean per modality over the split (unique images; all captions), fp64 accumulate;
       decoupled      : rows permuted independently for images (default_rng(44)) and captions (default_rng(4444)), batches of
                        batch_size rows of the permuted order, per-batch mean (image rows carry their image, duplicates included),
                        image j centred by the batch of its row per*j (as tools/r44_audit.py), fp64 accumulate.
     COCO: 5K over the whole testall split; 5-fold 1K with each fold's own rows (5000) for the means / permutations.
Hosts: only 'in2r' is layout dependent. 'sps' (SGRAF forward_emb, no centring) and 'nrcs' (gpo_adapter.encode_pool, no
centring) are not re-read; aggregate_layoutfree.py records them as not affected (layout-free = native).
Results: results_layoutfree/<sha16>.json. TEST reads logged in TEST_ACCESS_LOG.jsonl (event test_read, purpose layoutfree).
Usage: test_eval_layoutfree.py --gpu-name "<torch device name recorded in results/>" [--dataset ...] [--units ID ...] [--limit N]
"""
import os, sys, json, io, hashlib, time, argparse, shutil, datetime, types

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
import test_eval as TE  # noqa: E402

HERE = TE.HERE
RESLF = os.path.join(HERE, 'results_layoutfree')
SEED_IMG, SEED_CAP = 44, 4444
AUTH = 'user 2026-11-04 ~00:15UTC: "好，准备R49吧，smoke过了就提交，TEST也再读一次" (AGENTS.md); layout-free readouts of existing TEST units'


def lf_path(sha):
    return os.path.join(RESLF, sha[:16] + '.json')


def lf_done(u):
    p = lf_path(u['checkpoint_sha256'])
    if not os.path.exists(p):
        return False
    r = json.load(open(p))
    return r.get('status') == 'COMPLETE' and r.get('checkpoint_sha256') == u['checkpoint_sha256']


def raw_forward_emb(self, images, captions, lengths, imgae_lengths=None):
    """VSE.forward_emb without the per-batch mean subtraction + second l2norm (encoders unchanged)."""
    images, captions, imgae_lengths = images.cuda(), captions.cuda(), imgae_lengths.cuda()
    return self.img_enc(images, imgae_lengths), self.txt_enc(captions, lengths), lengths


class In2rLF(TE.In2r):
    def __init__(self, dp):
        super().__init__(dp)
        from vse import l2norm
        self.l2norm = l2norm
        self.captured = []
        base = self.VSE

        def factory(*a, **k):
            m = base(*a, **k); self.captured.append(m); return m
        self.VSE = factory

    # ---------------------------------------------------------------- centring
    def centre(self, X, kind, B, seed=None):
        """X: raw rows (n_rows, d) numpy (exact fp32 values). Returns centred+l2normed rows (torch fp32, cuda)."""
        torch = self.torch
        Xt = torch.from_numpy(X).float().cuda()
        n = Xt.shape[0]
        if kind == 'native_rebuilt':
            out = torch.empty_like(Xt)
            for lo in range(0, n, B):
                xb = Xt[lo:lo + B]
                out[lo:lo + B] = self.l2norm(xb - xb.mean(dim=0, keepdim=True))
            return out
        if kind == 'global':
            raise RuntimeError('global centring is computed per modality in variant()')
        # decoupled: row -> position in permuted order, batch = position // B
        import numpy as np
        perm = np.random.default_rng(seed).permutation(n)
        pos = np.empty(n, np.int64); pos[perm] = np.arange(n)
        b = torch.as_tensor(pos // B, device='cuda'); nb = int(b.max()) + 1
        S = torch.zeros(nb, Xt.shape[1], dtype=torch.float64, device='cuda').index_add_(0, b, Xt.double())
        cnt = torch.bincount(b, minlength=nb).double()[:, None]
        mub = (S / cnt).float()
        return self.l2norm(Xt - mub[b])

    def variant(self, kind, ie_rows, ce_rows, per, B):
        """ie_rows: raw image of every row (row r holds image r//per); returns (unique images centred, captions centred) numpy."""
        torch = self.torch
        if kind == 'global':
            iu = torch.from_numpy(ie_rows[::per]).float().cuda()
            ct = torch.from_numpy(ce_rows).float().cuda()
            ic = self.l2norm(iu - iu.double().mean(0, keepdim=True).float())
            cc = self.l2norm(ct - ct.double().mean(0, keepdim=True).float())
        else:
            ic = self.centre(ie_rows, kind, B, SEED_IMG)[::per]
            cc = self.centre(ce_rows, kind, B, SEED_CAP)
        return ic.cpu().numpy().astype('float64'), cc.cpu().numpy().astype('float64')

    def score(self, models, raws, kind, per, B, sl=None, shard=1000):
        import numpy as np
        sims, peers = 0, {}
        for ind, (m, (ie, ce, cl)) in enumerate(zip(models, raws)):
            if sl is not None:
                ie, ce, cl = ie[sl], ce[sl], cl[sl]
            ic, cc = self.variant(kind, ie, ce, per, B)
            s = self.shard(m, ic, cc, cl, None, shard_size=shard)
            peers[f'peer{ind}_rsum'] = float(sum(self.i2t(ic.shape[0], s, per)[:3]) + sum(self.t2i(ic.shape[0], s, per)[:3]))
            sims = sims + s
        sims = sims / 2
        d = TE.recall_dict(self.i2t(ic.shape[0], sims, per), self.t2i(ic.shape[0], sims, per))
        d.update(peers, n_images=int(ic.shape[0]), n_captions=int(cc.shape[0]))
        return d

    def raw_enc(self, m, loader):
        import numpy as np
        orig = m.forward_emb
        m.forward_emb = types.MethodType(raw_forward_emb, m)
        try:
            ie, ce, cl = self.encode_data(m, loader, 10 ** 9, logging=lambda *a, **k: None)
        finally:
            m.forward_emb = orig
        return np.asarray(ie), np.asarray(ce), np.asarray(list(cl))

    def evaluate_lf(self, u, ckpt):
        import numpy as np
        torch = self.torch
        self.captured = []
        native = self.evaluate(u, ckpt)            # unmodified native path (DEV recompute + TEST), builds models via factory
        models = self.captured; assert len(models) == 2
        opt = ckpt['opt']
        per = 1 if opt.data_name == 'cc152k_precomp' else 5
        out = dict(native=native)
        coco = opt.data_name == 'coco_precomp'
        split = self.test_split or ('testall' if coco else 'test')
        with torch.no_grad():
            dl = self.loader(opt, 'dev'); B = dl.batch_size
            draw = [self.raw_enc(m, dl) for m in models]
            tl = self.loader(opt, split); assert tl.batch_size == B
            traw = [self.raw_enc(m, tl) for m in models]
            for kind in ('native_rebuilt', 'global', 'decoupled'):
                v = dict(dev=self.score(models, draw, kind, per, B, shard=100))
                if not coco:
                    v['test'] = self.score(models, traw, kind, per, B)
                else:
                    v['test_5k'] = self.score(models, traw, kind, 5, B)
                    folds = [self.score(models, traw, kind, 5, B, sl=slice(i * 5000, (i + 1) * 5000)) for i in range(self.nfolds)]
                    mean = {k: float(np.mean([f[k] for f in folds])) for k in TE.KEYS + ('rsum', 'peer0_rsum', 'peer1_rsum')}
                    mean.update(n_folds=self.nfolds, n_images_per_fold=1000, n_captions_per_fold=5000)
                    v['test_5fold1k'] = dict(mean=mean, folds=folds)
                out[kind] = v
        out['batch_size'] = int(B)
        out['raw_l2_norm_check'] = float(max(abs(np.linalg.norm(draw[0][0][:50], axis=1) - 1).max(), abs(np.linalg.norm(draw[0][1][:50], axis=1) - 1).max()))
        self.captured = []
        del models
        torch.cuda.empty_cache()
        return out


def cmp_recalls(a, b):
    ks = TE.KEYS + ('rsum',)
    d = {k: a[k] - b[k] for k in ks}
    return d, max(abs(v) for v in d.values())


def compare_test(new, old):
    """native TEST now vs recorded TEST (results/<sha16>.json)."""
    rep = {}
    if 'test' in old:
        rep['test'] = cmp_recalls(new['test'], old['test'])
    else:
        rep['test_5k'] = cmp_recalls(new['test_5k'], old['test_5k'])
        rep['test_5fold1k_mean'] = cmp_recalls(new['test_5fold1k']['mean'], old['test_5fold1k']['mean'])
        for i, (f, g) in enumerate(zip(new['test_5fold1k']['folds'], old['test_5fold1k']['folds'])):
            rep[f'fold{i}'] = cmp_recalls(f, g)
    mx = max(v[1] for v in rep.values())
    return {k: dict(diff=v[0], max_abs=v[1]) for k, v in rep.items()}, mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu-name', required=True, help='recorded evaluation GPU (results/<sha16>.json env.gpu); job must run on it')
    ap.add_argument('--dataset', default='all', choices=('all', 'f30k', 'coco', 'cc152k'))
    ap.add_argument('--units', nargs='*')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--list-only', action='store_true')
    ap.add_argument('--smoke-dev', action='store_true', help='code check: DEV used in place of TEST (1 fold for COCO), no TEST file '
                    'staged or read, no access-log entry, results to results_layoutfree_smoke/')
    a = ap.parse_args()
    global RESLF
    if a.smoke_dev:
        RESLF = os.path.join(HERE, 'results_layoutfree_smoke')
    units, mans = [], {}
    for mp, sha in TE.frozen_manifests():
        assert TE.sha_file(mp) == sha, f'manifest changed after freezing: {mp}'
        man = json.load(open(mp)); mans[mp] = man
        if not any(u['host'] == 'in2r' for u in man['units']):
            continue
        for f, s in man['eval_code']['sha256'].items():
            assert TE.sha_file(os.path.join(man['eval_code']['dir'], f)) == s, f'eval code changed: {f}'
        for u in man['units']:
            if u['host'] != 'in2r' or (a.dataset != 'all' and u['dataset'] != a.dataset):
                continue
            rp = TE.result_path(u['checkpoint_sha256'])
            if not os.path.exists(rp):
                continue
            old = json.load(open(rp))
            if old.get('status') != 'COMPLETE' or old['env']['gpu'] != a.gpu_name:
                continue
            if a.units and u['unit_id'] not in a.units:
                continue
            units.append(dict(u, manifest=mp, manifest_sha256=sha, recorded_result=rp))
    i, n = map(int, a.shard.split('/'))
    units = [u for k, u in enumerate(sorted(units, key=lambda x: (x['dataset'], x['unit_id']))) if k % n == i]
    todo = [u for u in units if not lf_done(u)]
    if a.limit:
        todo = todo[:a.limit]
    print(f'{len(units)} in2r units recorded on {a.gpu_name!r} in selection, {len(todo)} to evaluate', flush=True)
    if a.list_only:
        for u in todo:
            print(' ', u['unit_id'], u['dataset'])
        return
    if not todo:
        return
    import torch
    gpu = torch.cuda.get_device_name(0)
    if gpu != a.gpu_name:
        print(f'GPU MISMATCH: got {gpu!r}, units recorded on {a.gpu_name!r}; nothing read', flush=True)
        sys.exit(4)
    data_root = mans[todo[0]['manifest']]['data']['root']
    dns = sorted({TE.DN[u['dataset']] for u in todo})
    files = [os.path.join(data_root, dn, f'{s}_{x}') for dn in dns for s in TE.SPLITS[dn][1:]
             for x in (('caps.tsv' if dn == 'cc152k_precomp' else 'caps.txt'), 'ims.npy')]
    jid = os.environ.get('SLURM_JOB_ID')
    if not a.smoke_dev:
      TE.append_log(dict(event='test_read', utc=datetime.datetime.utcnow().isoformat() + 'Z', job=jid,
                       restart=os.environ.get('SLURM_RESTART_COUNT'), node=os.uname().nodename, gpu=gpu, host_code='in2r',
                       manifests=sorted({u['manifest'] for u in todo}), units=[u['unit_id'] for u in todo], files_read=files,
                       purpose='layoutfree_reread (tools/test_eval_layoutfree.py)', authorization=AUTH))
    tmp = os.path.join(os.environ.get('TMPDIR') or '/tmp', f'telf_{jid}_{os.getpid()}')
    os.makedirs(RESLF, exist_ok=True)
    try:
        staged = {}
        for dn in dns:
            staged.update(TE.stage(data_root, dn, tmp, dev_only=a.smoke_dev))
        ev = In2rLF(tmp)
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
                       manifest_sha256=u['manifest_sha256'], checkpoint_epoch=u['checkpoint_epoch'], logged_dev=u['logged_dev'],
                       recorded_result=u['recorded_result'])
            if sha != u['checkpoint_sha256']:
                rec.update(status='CHECKPOINT_CHANGED', sha_now=sha)
                json.dump(rec, open(lf_path(u['checkpoint_sha256']), 'w'), indent=1)
                print('CHECKPOINT CHANGED, not evaluated:', u['unit_id'], flush=True)
                continue
            ckpt = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
            del raw
            assert int(ckpt['epoch']) == u['checkpoint_epoch']
            old = json.load(open(u['recorded_result']))
            out = ev.evaluate_lf(u, ckpt)
            nat = out['native']
            diffs, mx, ok = TE.gate(u['logged_dev'], nat['dev_recomputed'])
            tcmp, tmx = compare_test(nat, old) if not a.smoke_dev else ({}, float('nan'))
            _, dev_vs_rec = cmp_recalls(nat['dev_recomputed'], old['dev_recomputed'])
            _, reb_dev = cmp_recalls(out['native_rebuilt']['dev'], nat['dev_recomputed'])
            reb_test = cmp_recalls(out['native_rebuilt']['test'] if 'test' in nat else out['native_rebuilt']['test_5k'],
                                   nat['test'] if 'test' in nat else nat['test_5k'])[1]
            rec.update(out, dev_diff_recomputed_minus_logged=diffs, dev_max_abs_recall_diff_pp=mx, dev_gate='PASS' if ok else 'FAIL',
                       dev_gate_tolerance_pp=TE.TOL_PP, dev_gate_recorded=old.get('dev_gate'),
                       dev_native_now_minus_recorded_recompute_max_abs=dev_vs_rec,
                       test_native_vs_recorded=tcmp, test_native_vs_recorded_max_abs=tmx, test_native_match=bool(tmx < 1e-9),
                       native_rebuilt_minus_native_max_abs=dict(dev=reb_dev, test=reb_test),
                       decoupled_seeds=dict(images=SEED_IMG, captions=SEED_CAP), env=env, recorded_env=old.get('env'),
                       staged_files={k: v for k, v in staged.items() if k.startswith(TE.DN[u['dataset']])},
                       seconds=round(time.time() - t0, 1), status='COMPLETE', utc=datetime.datetime.utcnow().isoformat() + 'Z')
            p = lf_path(u['checkpoint_sha256'])
            json.dump(rec, open(p + '.tmp', 'w'), indent=1)
            os.replace(p + '.tmp', p)
            m = lambda v: (v.get('test') or v['test_5fold1k']['mean'])['rsum']
            print(json.dumps(dict(unit=u['unit_id'], gate=rec['dev_gate'], test_match=rec['test_native_match'], tmx=round(tmx, 6),
                                  dev=[round(x, 2) for x in (nat['dev_recomputed']['rsum'], out['global']['dev']['rsum'], out['decoupled']['dev']['rsum'])],
                                  test=[round(x, 2) for x in (m(nat), m(out['global']), m(out['decoupled']))],
                                  rebuilt=[round(reb_dev, 4), round(reb_test, 4)], s=rec['seconds'])), flush=True)
            del ckpt
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
