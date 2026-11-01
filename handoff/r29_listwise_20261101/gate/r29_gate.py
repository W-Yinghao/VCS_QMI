"""R29 real-data GPU gate. Usage: r29_gate.py SMOKE_DIR NOISE_FILE NOISE_RATIO DATA_PATH CODE_DIR -> SMOKE_DIR/GPU_GATE.json

Runs after the smoke (5 arms + an R27 A1 reference, each from the common UNTRAINED init: 2 limited warmup epochs + 2 co-train
epochs x 30 iterations, separate smoke dir). Technical checks only (no task metric is used):
  arms_ran            every smoke arm finished (DONE.json), finite clean losses / DEV rSum, warmup diagnostics written
  h_warmup_bitwise    from the untrained init, 3 warmup steps: native VSE.train(mode='warmup') vs r29 warmup_step(H) (with the
                      report-only autograd diagnostic on): parameters, AdamW state, RNG (python/numpy/torch/cuda), losses bitwise
  warmup_isolation    same init, same 3 batches for H/N/P/L/B: batch ids and RNG after the steps identical; at step 0 the
                      pre-clip encoder gradient minus the alignment-term gradient is the same for every arm (rel <= 1e-4);
                      identical initial encoder parameters; P's b receives a gradient
  h_cotrain_bitwise   from the H smoke state (queues filled): one/two real co-train steps of r27_co_train.train (A1) vs r29
                      train (H): encoder + refiner grads at optimizer.step, params after, AdamW state, RNG bitwise
  cotrain_isolation   same restored state, one step per arm: same labeled ids (observed-id hash), same noisy candidate
                      columns, noisy-branch native SCE values bitwise equal to H (= A1), same RNG after the step, other peer
                      gets no gradient, pre-clip grad minus alignment grad equal across arms (rel <= 1e-4); injected-noise
                      truth fields (_t2i_index) of all loaders poisoned (any read raises)
  resume_roundtrip    P and L (own smoke states): 3 steps -> save_full_state -> 3 more live steps vs fresh load + 3 steps:
                      params (incl. refiners, P's b), AdamW state, RNG, both queues + IDs bitwise
  gmm_unchanged       R.eval_train is co_train.eval_train; co_train.py / vse.py / data.py / r25_co_train.py / r27_*.py byte-identical
                      to the R27 code_v12 snapshot
  timing_memory       ms/step (20 co-train steps, same state) per arm + smoke peak memory (report only)
  h_vs_a1_crossprocess  H smoke state_last vs the A1 reference smoke (separate processes): max abs diff (report only;
                      cudnn.benchmark autotuning may differ across processes)
"""
import os, sys, json, copy, hashlib, time, glob
from collections import defaultdict
import numpy as np
import torch

CODE = sys.argv[5]
sys.path.insert(0, CODE)
import r25_run, r29_run
import r25_co_train as R
import r27_co_train as R27
import r27_interface as I
import r29_co_train as R29
import r29_align as AL
import co_train
from data import get_dataset
from vocab import deserialize_vocab

res = {}
ARMS = AL.ARMS
V12 = '/home/infres/yinwang/CS_QMI/handoff/r27_qtarget_20261029/code_v12'


def ok(name, cond, **info):
    res[name] = dict(pass_=bool(cond), **info)
    print(name, 'PASS' if cond else 'FAIL', json.dumps(info, default=float)[:3000], flush=True)


def h(t):
    return hashlib.sha256(np.ascontiguousarray(t.detach().cpu().numpy()).tobytes()).hexdigest()


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


class Poison:
    def _boom(self, *a, **k):
        raise RuntimeError('R29 gate: injected-noise truth (_t2i_index) was read')
    __getitem__ = __array__ = __len__ = __iter__ = _boom


def rng_eq(a, b):
    return (a['python'] == b['python'] and all(np.array_equal(x, y) for x, y in zip(a['numpy'][1:3], b['numpy'][1:3]))
            and torch.equal(a['torch'], b['torch']) and all(torch.equal(x, y) for x, y in zip(a['cuda'], b['cuda'])))


def own(m):
    ps = list(m.params)
    for name in ('i2t_refiner_node', 't2i_refiner_node'):
        if hasattr(m, name):
            ps += list(getattr(m, name).parameters())
    if hasattr(m, 'r29_b'):
        ps.append(m.r29_b)
    return ps


def flat(ps):
    return torch.cat([p.detach().flatten().double() for p in ps])


def adam_flat(m):
    return torch.cat([v.detach().flatten().double() for p in own(m) for k, v in m.optimizer.state.get(p, {}).items()
                      if torch.is_tensor(v) and v.dim() > 0] or [torch.zeros(0, dtype=torch.float64, device='cuda')])


def gdiff(a, b):
    m = 0.0
    for x, y in zip(a, b):
        if (x is None) != (y is None):
            return float('inf')
        if x is not None:
            m = max(m, float((x - y).abs().max()))
    return m


def main():
    smoke, noise, ratio, dpath = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    base = ['--noise_file', noise, '--noise_ratio', ratio, '--output_dir', f'{smoke}/gate', '--data_path', dpath, '--prefix_epoch', '-1']
    opt = r29_run.parse(['--r29_arm', 'h_native'] + base)
    os.makedirs(opt.output_dir, exist_ok=True)
    R29.install()
    r25_run.seed_all(opt.seed)
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, "%s_vocab.json" % opt.data_name)); vocab.add_word('<mask>')
    opt.vocab_size = len(vocab)
    w2i = vocab.word2idx
    caps, ims = get_dataset(opt.data_path, opt.data_name, "train")
    noisy_loader, data_size, clean_labels = R.get_loader(caps, ims, vocab, "warmup", opt.batch_size, 0, opt.noise_ratio, opt.noise_file)
    wds = noisy_loader.dataset
    wds._t2i_index = Poison()

    # ------------------------------------------------------------------ arms_ran
    info, good = {}, True
    for arm in list(ARMS) + ['f00_l_listwise_vcs']:   # last: the 0% path (identity noise file, noise_ratio 0)
        d = f'{smoke}/{arm}'
        e = {'done': os.path.exists(f'{d}/DONE.json'), 'warm_diag': os.path.exists(f'{d}/R29_WARMUP_DIAG.json')}
        if e['done']:
            hist = json.load(open(f'{d}/EPOCHS.json'))
            rs = [c['dev']['rsum'] for c in hist['cotrain']] + [w['dev']['rsum'] for w in hist['warmup']]
            cl = [c['train'][p].get('clean_loss_mean', float('nan')) for c in hist['cotrain'] for p in ('A', 'B')]
            e.update(n_cotrain=len(hist['cotrain']), rsum=rs, finite=bool(np.isfinite(rs).all() and np.isfinite(cl).all()),
                     gnorm_align=[c['train']['A'].get('r29_gnorm_align_mean') for c in hist['cotrain']])
            pk = glob.glob(f'{d}/PEAK_*.json')
            e['peak'] = json.load(open(pk[0])) if pk else None
        e['ok'] = e['done'] and e['warm_diag'] and e.get('finite', False) and e.get('n_cotrain') == 2
        if arm.startswith('f00') and e['done']:
            run = json.load(open(sorted(glob.glob(f'{d}/RUN_*.json'))[0]))['opt']
            e['noise_ratio'] = run['noise_ratio']
            e['ok'] &= run['noise_ratio'] == 0.0
        good &= e['ok']; info[arm] = e
    ok('arms_ran', good, **info)

    # ------------------------------------------------------------------ h_warmup_bitwise / warmup_isolation (untrained init)
    r25_run.seed_all(opt.seed)
    A = R.VSE(opt, w2i)
    init_enc = flat(list(A.params))
    init = (copy.deepcopy(A.img_enc.state_dict()), copy.deepcopy(A.txt_enc.state_dict()), copy.deepcopy(A.optimizer.state_dict()), R.rng_state())

    def load_init(m, with_opt=True):
        m.img_enc.load_state_dict(copy.deepcopy(init[0])); m.txt_enc.load_state_dict(copy.deepcopy(init[1]))
        if with_opt:
            m.optimizer.load_state_dict(copy.deepcopy(init[2]))
        m.Eiters = 0
        for p in own(m):
            p.grad = None
        R.set_rng_state(init[3])

    def run_warm(m, fn, n=3):
        load_init(m, with_opt=not hasattr(m, 'r29_b'))
        it = iter(noisy_loader); losses, idh = [], []
        for i in range(n):
            images, il, cp, lengths, ids = next(it)
            m.train_start()
            losses.append(fn(m, images, cp, lengths, il, ids, i))
            idh.append(hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest())
        return dict(params=flat(own(m)), enc=flat(list(m.params)), adam=adam_flat(m), rng=R.rng_state(), losses=losses, ids=idh)

    nat = run_warm(A, lambda m, im, cp, ln, il, ids, i: m.train(im, cp, ln, image_lengths=il, mode='warmup'))
    r29h = run_warm(A, lambda m, im, cp, ln, il, ids, i: R29.warmup_step(opt, m, im, cp, ln, il, R29._obs(wds, ids), defaultdict(float), i == 0))
    d = dict(param_maxdiff=float((nat['params'] - r29h['params']).abs().max()), adam_maxdiff=float((nat['adam'] - r29h['adam']).abs().max()),
             rng_equal=rng_eq(nat['rng'], r29h['rng']), losses_native=nat['losses'], losses_r29=r29h['losses'], ids_equal=nat['ids'] == r29h['ids'])
    ok('h_warmup_bitwise', d['param_maxdiff'] == 0 and d['adam_maxdiff'] == 0 and d['rng_equal'] and nat['losses'] == r29h['losses'] and d['ids_equal'], **d)

    # warmup isolation: per arm, same init + batches; capture pre-clip grads and the alignment-term gradient at step 0
    cap = {}
    orig_align, orig_clip = AL.align, R29.clip_grad_norm_

    def align_wrap(o, net, phase, sims, obs):
        l = orig_align(o, net, phase, sims, obs)
        if cap.get('want'):
            g = torch.autograd.grad(l, list(net.params), retain_graph=True, allow_unused=True)
            cap['g_align'] = [None if x is None else x.detach().clone() for x in g]
            cap['obs'] = hashlib.sha256(obs.cpu().numpy().tobytes()).hexdigest()
            cap['align'] = float(l.detach())
        return l

    def clip_wrap(params, mx):
        if cap.get('want'):
            cap['g_pre'] = [None if p.grad is None else p.grad.detach().clone() for p in params]
            cap['want'] = False
        return orig_clip(params, mx)

    AL.align, R29.clip_grad_norm_ = align_wrap, clip_wrap
    wiso = {}
    for arm in ARMS:
        opt.r29_arm = arm
        r25_run.seed_all(opt.seed)
        m = R.VSE(opt, w2i)
        init_eq = bool(torch.equal(flat(list(m.params)), init_enc))   # same seed -> same untrained construction for every arm

        def fn(mm, im, cp, ln, il, ids, i):
            cap['want'] = (i == 0)
            return R29.warmup_step(opt, mm, im, cp, ln, il, R29._obs(wds, ids), defaultdict(float), i == 0)
        out = run_warm(m, fn)
        g_rest = [None if a is None else (a - (b if b is not None else 0)) for a, b in zip(cap['g_pre'], cap['g_align'])]
        wiso[arm] = dict(out=out, g_rest=g_rest, init_eq=init_eq, b=float(m.r29_b.detach()) if hasattr(m, 'r29_b') else None,
                         b_grad=None if not hasattr(m, 'r29_b') or m.r29_b.grad is None else float(m.r29_b.grad), align0=cap['align'])
        del m
    AL.align, R29.clip_grad_norm_ = orig_align, orig_clip
    opt.r29_arm = 'h_native'
    href = wiso['h_native']
    gscale = max(float(x.abs().max()) for x in href['g_rest'] if x is not None)
    d = {}
    for arm, w in wiso.items():
        d[arm] = dict(ids_equal=w['out']['ids'] == href['out']['ids'], rng_equal=rng_eq(w['out']['rng'], href['out']['rng']),
                      g_rest_rel=gdiff(w['g_rest'], href['g_rest']) / gscale, align_step0=w['align0'], losses=w['out']['losses'], init_equal=w['init_eq'],
                      b_after=w['b'], b_grad_last=w['b_grad'])
    good = all(v['ids_equal'] and v['rng_equal'] and v['g_rest_rel'] <= 1e-4 and v['init_equal'] for v in d.values()) and d['p_pointwise_vcs']['b_after'] != AL.P_B0
    ok('warmup_isolation', good, g_scale=gscale, **d)
    del wiso

    # ------------------------------------------------------------------ co-train checks from the H smoke state
    S0 = f'{smoke}/h_native/state_last.pt'
    A = R.VSE(opt, w2i); B = R.VSE(opt, w2i)
    st = R.load_full_state(S0, opt, A, B)
    ep = st['epoch'] + 1
    prob_A, prob_B, _ = co_train.eval_train(opt, A, B, noisy_loader, data_size, copy.deepcopy(st['all_loss']), clean_labels, ep)
    pred_A = co_train.split_prob(prob_A, opt.p_threshold); pred_B = co_train.split_prob(prob_B, opt.p_threshold)
    for m, prob, pred in ((A, prob_A, pred_A), (B, prob_B, pred_B)):
        m.avg_clean_prob = prob[pred].mean()

    def loaders():
        lt, ut = R.get_loader(caps, ims, vocab, "train", opt.batch_size, 0, opt.noise_ratio, opt.noise_file, pred=pred_B, prob=prob_B)
        lt.dataset._t2i_index = Poison(); ut.dataset._t2i_index = Poison()
        return lt, ut
    lt, ut = loaders()

    def snapshot():
        return copy.deepcopy(R.model_full_state(A)), copy.deepcopy(R.model_full_state(B)), R.rng_state()

    def restore(snap):
        for m, s in ((A, snap[0]), (B, snap[1])):
            m.img_enc.load_state_dict(s['img_enc']); m.txt_enc.load_state_dict(s['txt_enc'])
            m.optimizer.load_state_dict(copy.deepcopy(s['optimizer']))  # deepcopy: load_state_dict aliases same-device tensors
            m.i2t_refiner_node.load_state_dict(s['i2t_refiner']); m.t2i_refiner_node.load_state_dict(s['t2i_refiner'])
            q = m.memory_queue; q.img_queue.copy_(s['queue']['img']); q.txt_queue.copy_(s['queue']['txt']); q.ptr, q.is_full = s['queue']['ptr'], s['queue']['is_full']
            I.load_ids(q, s['r27_queue_ids'])
            if hasattr(m, 'r29_b'):
                with torch.no_grad():
                    m.r29_b.copy_(s['r29_b'])
            m.Eiters = s['Eiters']
            for p in own(m):
                p.grad = None
        R.set_rng_state(snap[2])
        R27._PEER.clear(); R27._PEER[id(A)] = 0; R27._PEER[id(B)] = 1

    def one_step(fn, arm, n=1, probe=None, want=False):
        restore(snap)
        opt.r29_arm = arm
        c = {}
        orig = A.optimizer.step

        def step_probe(*a, **k):
            if 'enc' not in c:
                c['enc'] = [None if p.grad is None else p.grad.detach().clone() for p in A.params]
                c['ref'] = [None if p.grad is None else p.grad.detach().clone() for p in own(A)[len(A.params):]]
                c['peer_with_grad'] = sum(1 for p in own(B) if p.grad is not None)
            return orig(*a, **k)
        A.optimizer.step = step_probe
        cap.clear(); cap['want'] = want
        stats = R.new_stats()
        fn(opt, A, B, lt, ut, ep, stats, max_iters=n, probe=probe)
        A.optimizer.step = orig
        c.update(params=flat(own(A)), adam=adam_flat(A), rng=R.rng_state(), stats=dict(stats),
                 qB=(h(B.memory_queue.img_queue), h(B.memory_queue.txt_queue)), cap=dict(cap))
        return c

    snap = snapshot()
    a1 = one_step(R27.train, 'h_native', n=2)
    hh = one_step(R29.train, 'h_native', n=2)
    d = dict(enc_grad_maxdiff=gdiff(a1['enc'], hh['enc']), refiner_grad_maxdiff=gdiff(a1['ref'], hh['ref']),
             param_maxdiff=float((a1['params'] - hh['params']).abs().max()), adam_maxdiff=float((a1['adam'] - hh['adam']).abs().max()),
             rng_equal=rng_eq(a1['rng'], hh['rng']), noisy_steps=[a1['stats'].get('i2t_n'), a1['stats'].get('t2i_n')],
             raw_loss_equal=[a1['stats'].get(k) == hh['stats'].get(k) for k in ('i2t_raw_loss', 't2i_raw_loss', 'clean_loss')])
    ok('h_cotrain_bitwise', d['enc_grad_maxdiff'] == 0 and d['refiner_grad_maxdiff'] == 0 and d['param_maxdiff'] == 0 and d['adam_maxdiff'] == 0
       and d['rng_equal'] and all(d['raw_loss_equal']) and min(x or 0 for x in d['noisy_steps']) >= 1, **d)

    # cotrain isolation: attach P's b group to both peers (gate-only layout), new snapshot, one step per arm
    for m in (A, B):
        m.r29_b = torch.nn.Parameter(torch.full((), AL.P_B0, dtype=torch.float32, device='cuda'))
        m.optimizer.add_param_group({'params': [m.r29_b], 'lr': m.optimizer.param_groups[0]['lr'], 'weight_decay': 0.0})
    snap = snapshot()
    AL.align, R29.clip_grad_norm_ = align_wrap, clip_wrap
    ciso = {}
    for arm in ARMS:
        seen = {}

        def probe(branch, dd):
            seen[branch] = dict(cols=h(dd['S']['cols']), A=h(dd['S']['A']), l_nat=float(dd['l_nat'].detach()),
                                X_req=bool(dd['X'].requires_grad))
        c = one_step(R29.train, arm, n=1, probe=probe, want=True)
        g_rest = [None if a is None else (a - (b if b is not None else 0)) for a, b in zip(c['cap']['g_pre'], c['cap']['g_align'])]
        ciso[arm] = dict(c=c, seen=seen, g_rest=g_rest)
    AL.align, R29.clip_grad_norm_ = orig_align, orig_clip
    href = ciso['h_native']
    gscale = max(float(x.abs().max()) for x in href['g_rest'] if x is not None)
    d = {}
    for arm, w in ciso.items():
        c = w['c']
        d[arm] = dict(obs_equal=c['cap']['obs'] == href['c']['cap']['obs'], noisy_equal=w['seen'] == href['seen'] and len(w['seen']) == 2,
                      rng_equal=rng_eq(c['rng'], href['c']['rng']), peer_with_grad=c['peer_with_grad'], qB_equal=c['qB'] == href['c']['qB'],
                      g_rest_rel=gdiff(w['g_rest'], href['g_rest']) / gscale, align=c['cap']['align'],
                      gnorm_align=c['stats'].get('r29_gnorm_align'), gnorm_clean=c['stats'].get('gnorm_enc_clean'))
    good = all(v['obs_equal'] and v['noisy_equal'] and v['rng_equal'] and v['peer_with_grad'] == 0 and v['qB_equal'] and v['g_rest_rel'] <= 1e-4
               for v in d.values())
    ok('cotrain_isolation', good, g_scale=gscale, noisy_branch=href['seen'], **d)

    # timing (report only): 20 steps per arm from the same state
    tm = {}
    for arm in ARMS:
        torch.cuda.synchronize(); t = time.time()
        one_step(R29.train, arm, n=20)
        torch.cuda.synchronize(); tm[arm] = (time.time() - t) / 20 * 1000
    del A, B, snap, ciso
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ resume roundtrip (P, L own smoke states)
    rr, good = {}, True
    for arm in ('p_pointwise_vcs', 'l_listwise_vcs'):
        opt.r29_arm = arm
        A1m, B1m = R.VSE(opt, w2i), R.VSE(opt, w2i)
        st = R.load_full_state(f'{smoke}/{arm}/state_last.pt', opt, A1m, B1m)
        e = st['epoch'] + 1

        def steps(Am, Bm, n=3):
            l1, u1 = loaders()
            R29.train(opt, Am, Bm, l1, u1, e, R.new_stats(), max_iters=n)
        steps(A1m, B1m)
        tmp = f'{smoke}/gate/rr_{arm}.pt'
        R.save_full_state(tmp, opt, A1m, B1m, e, 0.0, [[], []], {'warmup': [], 'cotrain': []})
        steps(A1m, B1m)
        X = dict(pA=flat(own(A1m)), pB=flat(own(B1m)), adA=adam_flat(A1m), rng=R.rng_state(),
                 q=[h(m.memory_queue.img_queue) + h(m.memory_queue.txt_queue) + ''.join(h(v) for v in m.memory_queue.r27_ids.values()) for m in (A1m, B1m)],
                 b=float(A1m.r29_b) if hasattr(A1m, 'r29_b') else None)
        del A1m, B1m
        A2m, B2m = R.VSE(opt, w2i), R.VSE(opt, w2i)
        R.load_full_state(tmp, opt, A2m, B2m)
        steps(A2m, B2m)
        Y = dict(pA=flat(own(A2m)), pB=flat(own(B2m)), adA=adam_flat(A2m), rng=R.rng_state(),
                 q=[h(m.memory_queue.img_queue) + h(m.memory_queue.txt_queue) + ''.join(h(v) for v in m.memory_queue.r27_ids.values()) for m in (A2m, B2m)],
                 b=float(A2m.r29_b) if hasattr(A2m, 'r29_b') else None)
        del A2m, B2m
        os.remove(tmp)
        r = dict(pA=float((X['pA'] - Y['pA']).abs().max()), pB=float((X['pB'] - Y['pB']).abs().max()), adam=float((X['adA'] - Y['adA']).abs().max()),
                 rng_equal=rng_eq(X['rng'], Y['rng']), queues_equal=X['q'] == Y['q'], b_live=X['b'], b_resumed=Y['b'])
        r['ok'] = r['pA'] == 0 and r['pB'] == 0 and r['adam'] == 0 and r['rng_equal'] and r['queues_equal'] and X['b'] == Y['b']
        good &= r['ok']; rr[arm] = r
        torch.cuda.empty_cache()
    opt.r29_arm = 'h_native'
    ok('resume_roundtrip', good, **rr)

    # ------------------------------------------------------------------ gmm / host unchanged
    same = {f: sha(os.path.join(CODE, f)) == sha(os.path.join(V12, f)) for f in sorted(os.listdir(V12)) if f.endswith('.py')}
    ok('gmm_unchanged', R.eval_train is co_train.eval_train and all(same.values()) and R.N is co_train, files_identical_to_code_v12=same)

    ok('timing_memory', True, ms_per_cotrain_step=tm, smoke_peaks={a: info[a].get('peak') for a in ARMS})

    # ------------------------------------------------------------------ cross-process H vs A1 smoke (report only)
    try:
        sa = torch.load(f'{smoke}/a1_ref/state_last.pt', map_location='cpu', weights_only=False)
        sh = torch.load(S0, map_location='cpu', weights_only=False)
        mx = 0.0
        for key in ('model_A', 'model_B'):
            for part in ('img_enc', 'txt_enc', 'i2t_refiner', 't2i_refiner'):
                for k, v in sa[key][part].items():
                    if torch.is_floating_point(v):
                        mx = max(mx, float((v.double() - sh[key][part][k].double()).abs().max()))
        ha = json.load(open(f'{smoke}/a1_ref/EPOCHS.json')); hb = json.load(open(f'{smoke}/h_native/EPOCHS.json'))
        res['h_vs_a1_crossprocess'] = dict(report_only=True, param_maxabsdiff=mx, bitwise=mx == 0.0,
                                           dev_rsum_a1=[c['dev']['rsum'] for c in ha['cotrain']], dev_rsum_h=[c['dev']['rsum'] for c in hb['cotrain']])
    except Exception as ex:
        res['h_vs_a1_crossprocess'] = dict(report_only=True, error=repr(ex))
    print('h_vs_a1_crossprocess', json.dumps(res['h_vs_a1_crossprocess'])[:800])

    # ------------------------------------------------------------------ manifest / protocol
    root = os.path.dirname(os.path.abspath(CODE))
    im = json.load(open(os.path.join(CODE, 'INSTALL_MANIFEST.json')))
    code_ok = all(sha(os.path.join(CODE, f)) == hh_ for f, hh_ in im['files_sha256'].items()) and \
        sorted(im['files_sha256']) == sorted(f for f in os.listdir(CODE) if f.endswith('.py'))
    pf = os.path.join(root, 'PROTOCOL_FROZEN.json')
    frozen = os.path.isfile(pf) and json.load(open(pf)).get('frozen') is True
    checks = [k for k, v in res.items() if isinstance(v, dict) and 'pass_' in v]
    out = dict(all_pass=all(res[k]['pass_'] for k in checks) and code_ok and frozen, config_frozen=frozen, code_matches_manifest=code_ok,
               protocol_frozen_sha256=sha(pf) if os.path.isfile(pf) else None, code_files_sha256=im['files_sha256'],
               install_manifest_sha256=sha(os.path.join(CODE, 'INSTALL_MANIFEST.json')), job=os.environ.get('SLURM_JOB_ID'),
               gpu=torch.cuda.get_device_name(0), host=os.uname().nodename, smoke_dir=smoke, checks=res,
               utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    json.dump(out, open(f'{smoke}/GPU_GATE.json', 'w'), indent=1, default=float)
    print('ALL_PASS', out['all_pass'], flush=True)
    sys.exit(0 if out['all_pass'] else 1)


if __name__ == '__main__':
    main()
