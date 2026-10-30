"""R27-D real-data GPU gate (package docs/IMPLEMENTATION_CN.md §2). Run from the installed snapshot code_frozen/ (cwd).
Usage: r27d_gate.py SMOKE_DIR NOISE_FILE DATA_PATH GATE_STATE CODE_DIR PKG_DIR -> SMOKE_DIR/GPU_GATE.json

GATE_STATE: a real R27-S smoke state (both queues partly ID-bearing). From one restored state, one real training step per
arm (A1, S4, S6, D1-D4) with a probe that takes the per-direction gradient of that direction's loss with respect to the
encoder and current-peer refiner parameters (torch.autograd.grad, retain_graph; the production backward is untouched).
  routing          D arm x direction -> kind (r27d_policy); generate_cap calls per step = number of corrected directions
  identity_grad    uncorrected direction of D1-D4 == A1 same direction (loss value, encoder + refiner grads)
  cap_matches_S    corrected direction == S4 (VCS) / S6 (soft-NCE) same direction (loss, final target, grads, graph A)
  both_branches    i2t and t2i both trained (opt.i2t/opt.t2i on, one noisy step each), own refiners of both directions
                   get gradients, the other peer gets none and its queue/IDs are unchanged, one optimizer.step,
                   refiner forward count equal to A1
  resume           snapshot restore twice -> bitwise identical step (deepcopy, no optimizer aliasing); reload of the
                   checkpoint from disk -> identical step and queue IDs; run_unit identity: same arm resumes, other D arm
                   refused; directional policy re-derived from the arm after reload
  truth_isolation  capped targets within 15 deg of sg P; candidate graph equal to the native one; the noisy-branch source
                   (s_branch, noisy_branches, search, r27s_targets, r27d_policy) never references truth labels; train split
  timing           ms/step (30 iterations, same state) and peak memory per arm
  arms_ran         the four D smoke arms finished their smoke epoch with finite losses
"""
import os, sys, json, copy, hashlib, time, inspect, subprocess, tempfile, shutil
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(sys.argv[5]))   # installed snapshot code_frozen (this script lives outside it)
import r25_run
import r25_co_train as R
import r27_co_train as R27
import r27_interface as I
import r27s_targets as ST
import r27d_policy as P
from data import get_dataset
from vse import VSE
from vocab import deserialize_vocab
from co_train import eval_train, split_prob

res = {}
D_RUN = ['d1_vcs_i2t', 'd2_vcs_t2i', 'd3_snce_i2t', 'd4_snce_t2i']
REF = {'cap_vcs': 's4_single_cap_vcs', 'cap_snce': 's6_single_cap_snce', 'identity': 'a1_iface'}
LAM = 1.1448122442850819
TOL = 1e-6


def ok(name, cond, **info):
    res[name] = dict(pass_=bool(cond), **info)
    print(name, 'PASS' if cond else 'FAIL', json.dumps(info, default=float)[:3000], flush=True)


def h(t):
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()


def fsha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def main():
    smoke, noise, dpath, S0, code_dir, pkg = sys.argv[1:7]
    manifest = json.load(open(os.path.join(code_dir, 'INSTALL_MANIFEST.json')))
    code_now = {f: fsha(os.path.join(code_dir, f)) for f in manifest['files_sha256']}
    opt = r25_run.parse(['--noise_file', noise, '--output_dir', f'{smoke}/gate', '--resume', S0, '--data_path', dpath, '--max_iters', '30'])
    opt.arm, opt.lam = 'a1_iface', 0.0
    os.makedirs(opt.output_dir, exist_ok=True)
    R27.install()
    r25_run.seed_all(opt.seed)
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, "%s_vocab.json" % opt.data_name)); vocab.add_word('<mask>')
    opt.vocab_size = len(vocab)
    caps, ims = get_dataset(opt.data_path, opt.data_name, "train")
    noisy_loader, data_size, clean_labels = R.get_loader(caps, ims, vocab, "warmup", opt.batch_size, 0, opt.noise_ratio, opt.noise_file)
    A = VSE(opt, vocab.word2idx); B = VSE(opt, vocab.word2idx)
    st = R.load_full_state(S0, opt, A, B)
    ep = st['epoch'] + 1
    prob_A, prob_B, _ = eval_train(opt, A, B, noisy_loader, data_size, copy.deepcopy(st['all_loss']), clean_labels, ep)
    pred_A = split_prob(prob_A, opt.p_threshold); pred_B = split_prob(prob_B, opt.p_threshold)
    for m, prob, pred in ((A, prob_A, pred_A), (B, prob_B, pred_B)):
        m.avg_clean_prob = prob[pred].mean()
    lt, ut = R.get_loader(caps, ims, vocab, "train", opt.batch_size, 0, opt.noise_ratio, opt.noise_file, pred=pred_B, prob=prob_B)
    snap = copy.deepcopy(R.model_full_state(A)), copy.deepcopy(R.model_full_state(B)), R.rng_state()

    def restore():
        for m, s in ((A, snap[0]), (B, snap[1])):
            m.img_enc.load_state_dict(s['img_enc']); m.txt_enc.load_state_dict(s['txt_enc'])
            m.optimizer.load_state_dict(copy.deepcopy(s['optimizer']))  # deepcopy: load_state_dict aliases same-device tensors
            m.i2t_refiner_node.load_state_dict(s['i2t_refiner']); m.t2i_refiner_node.load_state_dict(s['t2i_refiner'])
            q = m.memory_queue; q.img_queue.copy_(s['queue']['img']); q.txt_queue.copy_(s['queue']['txt']); q.ptr, q.is_full = s['queue']['ptr'], s['queue']['is_full']
            I.load_ids(q, s['r27_queue_ids'])
            for p in list(m.params) + list(m.i2t_refiner_node.parameters()) + list(m.t2i_refiner_node.parameters()):
                p.grad = None
        R.set_rng_state(snap[2])
        R27._PEER.clear(); R27._PEER[id(A)] = 0; R27._PEER[id(B)] = 1

    own_ref = lambda: list(A.i2t_refiner_node.parameters()) + list(A.t2i_refiner_node.parameters())
    peer_all = lambda: list(B.params) + list(B.i2t_refiner_node.parameters()) + list(B.t2i_refiner_node.parameters())
    ids_of = lambda q: {k: v.clone() for k, v in getattr(q, 'r27_ids', {}).items()}

    gen_calls = []
    orig_gen = R27.generate_cap

    def counted_gen(kind, *a, **k):
        gen_calls.append(kind)
        return orig_gen(kind, *a, **k)
    R27.generate_cap = counted_gen

    ref_calls = [0]
    hooks = [m.register_forward_hook(lambda *a: ref_calls.__setitem__(0, ref_calls[0] + 1))
             for m in (A.i2t_refiner_node, A.t2i_refiner_node)]

    def one_step(arm, probe_grads=True, do_restore=True):
        if do_restore:
            restore()
        opt.arm, opt.lam = arm, (0.0 if arm == 'a1_iface' else LAM)
        cap, br = {'nstep': 0}, {}
        gen_calls.clear(); ref_calls[0] = 0
        params = list(A.params) + own_ref()

        def probe(branch, dd):
            e = dict(loss=float(dd['loss']), l_nat=float(dd['l_nat']), target=None if dd['target'] is None else dd['target'].detach().clone(),
                     graphA=dd['S']['A'].detach().clone(), cols=dd['S']['cols'].detach().clone(),
                     Pd_req=bool(dd['Pd'].requires_grad) if dd.get('Pd') is not None else None,
                     target_req=bool(dd['target'].requires_grad) if dd['target'] is not None else None,
                     X_req=bool(dd['X'].requires_grad), finite=bool(torch.isfinite(dd['loss'])))
            if dd['target'] is not None and dd.get('Pd') is not None:
                pn = dd['Pd'] / dd['Pd'].norm(dim=1, keepdim=True)
                e['min_cos_to_p'] = float((dd['target'] * pn).sum(1).min())
            if probe_grads:
                g = torch.autograd.grad(dd['loss'], params, retain_graph=True, allow_unused=True)
                e['grads'] = [None if x is None else x.detach().clone() for x in g]
            br[branch] = e
        orig = A.optimizer.step

        def step_probe(*a, **k):
            cap['nstep'] += 1
            cap['enc'] = [None if p.grad is None else p.grad.detach().clone() for p in A.params]
            cap['ref_i2t'] = [None if p.grad is None else p.grad.detach().clone() for p in A.i2t_refiner_node.parameters()]
            cap['ref_t2i'] = [None if p.grad is None else p.grad.detach().clone() for p in A.t2i_refiner_node.parameters()]
            cap['peer_with_grad'] = sum(1 for p in peer_all() if p.grad is not None)
            out = orig(*a, **k)
            cap['params'] = torch.cat([p.detach().flatten().double() for p in params])
            cap['adam'] = torch.cat([v.detach().flatten().double() for p in params for kk, v in A.optimizer.state.get(p, {}).items() if torch.is_tensor(v) and v.dim() > 0])
            return out
        A.optimizer.step = step_probe
        qh = (h(B.memory_queue.img_queue), h(B.memory_queue.txt_queue), {k: h(v) for k, v in ids_of(B.memory_queue).items()})
        stats = R.new_stats()
        R27.train(opt, A, B, lt, ut, ep, stats, max_iters=1, probe=probe)
        A.optimizer.step = orig
        cap.update(rng=R.rng_state(), stats=stats, br=br, gen_calls=list(gen_calls), ref_calls=ref_calls[0],
                   peer_queue_same=(h(B.memory_queue.img_queue), h(B.memory_queue.txt_queue), {k: h(v) for k, v in ids_of(B.memory_queue).items()}) == qh,
                   own_ids={k: v.clone() for k, v in ids_of(A.memory_queue).items()})
        return cap

    def gdiff(a, b):
        m = 0.0
        for x, y in zip(a, b):
            if (x is None) != (y is None):
                return float('inf')
            if x is not None:
                m = max(m, float((x - y).abs().max()))
        return m

    steps = {arm: one_step(arm) for arm in ['a1_iface', 's4_single_cap_vcs', 's6_single_cap_snce'] + D_RUN}

    # ---- routing
    rt, good = {}, True
    for arm in D_RUN:
        kinds = {b: R27.direction_kind(arm, b, R27.S_ARMS) for b in ('i2t', 't2i')}
        ncap = sum(k != 'identity' for k in kinds.values())
        e = dict(kinds=kinds, expected=P.D_ARMS[arm], gen_calls=steps[arm]['gen_calls'])
        good &= kinds == P.D_ARMS[arm] and len(steps[arm]['gen_calls']) == ncap and all(k in ('cap_vcs', 'cap_snce') for k in steps[arm]['gen_calls'])
        rt[arm] = e
    for arm in ('s4_single_cap_vcs', 's6_single_cap_snce'):
        rt[arm] = dict(kinds={b: R27.direction_kind(arm, b, R27.S_ARMS) for b in ('i2t', 't2i')}, gen_calls=steps[arm]['gen_calls'])
        good &= len(steps[arm]['gen_calls']) == 2
    rt['a1_iface'] = dict(gen_calls=steps['a1_iface']['gen_calls']); good &= not steps['a1_iface']['gen_calls']
    ok('routing', good, **rt)

    # ---- identity_grad / cap_matches_S (per direction)
    idg, capm, g_id, g_cap = {}, {}, True, True
    for arm in D_RUN:
        for b in ('i2t', 't2i'):
            kind = P.D_ARMS[arm][b]
            ref = steps[REF[kind]]['br'][b]; cur = steps[arm]['br'][b]
            # value check on the native SCE value l_nat (A1 reports 0.5*SCE as its loss, S/D report 0.5*l_ref + 0.5*l_enc; gradients are what must match)
            e = dict(ref=REF[kind], l_nat_ref=ref['l_nat'], l_nat_cur=cur['l_nat'], loss_absdiff=abs(ref['l_nat'] - cur['l_nat']),
                     total_loss_ref=ref['loss'], total_loss_cur=cur['loss'],
                     grad_maxdiff=gdiff(ref['grads'], cur['grads']), grad_none_pattern_equal=[x is None for x in ref['grads']] == [x is None for x in cur['grads']],
                     graph_equal=bool(torch.equal(ref['graphA'], cur['graphA']) and torch.equal(ref['cols'], cur['cols'])))
            e['bitwise'] = e['grad_maxdiff'] == 0 and e['loss_absdiff'] == 0
            pas = e['loss_absdiff'] <= TOL and e['grad_maxdiff'] <= TOL and e['grad_none_pattern_equal'] and e['graph_equal']
            if kind == 'identity':
                idg[f'{arm}/{b}'] = e; g_id &= pas
            else:
                e['target_maxdiff'] = float((ref['target'] - cur['target']).abs().max())
                e['total_loss_absdiff'] = abs(ref['loss'] - cur['loss'])
                pas &= e['target_maxdiff'] <= TOL and e['total_loss_absdiff'] <= TOL
                capm[f'{arm}/{b}'] = e; g_cap &= pas
    ok('identity_grad', g_id, **idg)
    ok('cap_matches_S', g_cap, **capm)

    # ---- both branches, calls
    bb, good = {}, True
    for arm in ['a1_iface'] + D_RUN:
        c = steps[arm]; s_ = c['stats']
        nz = lambda L: sum(1 for g in L if g is not None and float(g.abs().sum()) > 0)
        e = dict(opt_i2t=bool(opt.i2t), opt_t2i=bool(opt.t2i), i2t_n=s_['i2t_n'], t2i_n=s_['t2i_n'], enc_grad_nonzero=nz(c['enc']),
                 ref_i2t_with_grad=nz(c['ref_i2t']), ref_t2i_with_grad=nz(c['ref_t2i']), peer_params_with_grad=c['peer_with_grad'],
                 peer_queue_ids_same=c['peer_queue_same'], optimizer_steps=c['nstep'], refiner_forwards=c['ref_calls'],
                 refiner_forwards_a1=steps['a1_iface']['ref_calls'],
                 stopgrad=all((v['Pd_req'] in (False, None)) and (v['target_req'] in (False, None)) and v['X_req'] and v['finite'] for v in c['br'].values()))
        good &= (e['opt_i2t'] and e['opt_t2i'] and e['i2t_n'] == 1 and e['t2i_n'] == 1 and e['enc_grad_nonzero'] > 0 and e['ref_i2t_with_grad'] > 0
                 and e['ref_t2i_with_grad'] > 0 and e['peer_params_with_grad'] == 0 and e['peer_queue_ids_same'] and e['optimizer_steps'] == 1
                 and e['refiner_forwards'] == e['refiner_forwards_a1'] and e['stopgrad'])
        bb[arm] = e
    ok('both_branches', good, **bb)

    # ---- resume
    rs = {}
    a = one_step('d1_vcs_i2t', probe_grads=False); b = one_step('d1_vcs_i2t', probe_grads=False)
    rs['snapshot_twice'] = dict(param_maxdiff=float((a['params'] - b['params']).abs().max()), adam_maxdiff=float((a['adam'] - b['adam']).abs().max()),
                                rng_equal=bool(torch.equal(a['rng']['torch'], b['rng']['torch'])) and all(torch.equal(x, y) for x, y in zip(a['rng']['cuda'], b['rng']['cuda'])))
    # reload the checkpoint from disk into the same modules, then the same post-eval_train RNG and loaders
    R.load_full_state(S0, opt, A, B)
    for m, prob, pred in ((A, prob_A, pred_A), (B, prob_B, pred_B)):
        m.avg_clean_prob = prob[pred].mean()   # as after eval_train (load_full_state restores the checkpoint value)
    for m, s in ((A, snap[0]), (B, snap[1])):
        for p in list(m.params) + list(m.i2t_refiner_node.parameters()) + list(m.t2i_refiner_node.parameters()):
            p.grad = None
    R.set_rng_state(snap[2]); R27._PEER.clear(); R27._PEER[id(A)] = 0; R27._PEER[id(B)] = 1
    c = one_step('d1_vcs_i2t', probe_grads=False, do_restore=False)
    rs['disk_reload'] = dict(param_maxdiff=float((a['params'] - c['params']).abs().max()), adam_maxdiff=float((a['adam'] - c['adam']).abs().max()),
                             rng_equal=bool(torch.equal(a['rng']['torch'], c['rng']['torch'])),
                             queue_ids_equal=all(torch.equal(a['own_ids'][k], c['own_ids'][k]) for k in a['own_ids']) and set(a['own_ids']) == set(c['own_ids']))
    # D smoke checkpoint: saved arm and re-derived policy
    dst = torch.load(f'{smoke}/d1_vcs_i2t/state_last.pt', map_location='cpu', weights_only=False)
    saved_opt = dst.get('opt') if isinstance(dst, dict) else None
    saved_arm = (saved_opt.get('arm') if isinstance(saved_opt, dict) else getattr(saved_opt, 'arm', None)) if saved_opt is not None else None
    rs['d1_smoke_state'] = dict(keys=sorted(dst.keys())[:40] if isinstance(dst, dict) else None, saved_arm=saved_arm,
                                policy_from_arm={b: R27.direction_kind('d1_vcs_i2t', b, R27.S_ARMS) for b in ('i2t', 't2i')})
    # run_unit identity (dry-run only, temporary output root)
    tmp = tempfile.mkdtemp(prefix='r27d_gate_id_')
    cells = '/home/infres/yinwang/CS_QMI/handoff/r27_qtarget_20261029/CELLS.json'
    lamf = '/home/infres/yinwang/CS_QMI/handoff/r27_qtarget_20261029/LAMBDA_flickr.json'
    base = [sys.executable, '-B', os.path.join(pkg, 'tools/run_unit.py'), '--cells', cells, '--lambda-file', lamf, '--code-dir', code_dir,
            '--data-path', dpath, '--output-root', tmp]
    def ru(unit):
        p = subprocess.run(base + ['--unit', unit], capture_output=True, text=True)
        return p.returncode, (p.stdout + p.stderr).strip()[-400:]
    # make a real identity for d1 by mimicking run_unit's identity (dry run prints, so write via the same function's fields)
    code0, out0 = ru('f40_s1001__d1_vcs_i2t')
    os.makedirs(f'{tmp}/f40_s1001/d1_vcs_i2t'); os.makedirs(f'{tmp}/f40_s1001/d2_vcs_t2i')
    sys.path.insert(0, os.path.join(pkg, 'tools'))
    um = json.load(open(os.path.join(pkg, 'EXPERIMENT_MATRIX.json')))
    units = {u['unit_id']: u for u in um['units']}
    cl = json.load(open(cells))['f40_s1001']
    ident = lambda u: {'unit': units[u], 'prefix_sha256': fsha(cl['prefix']), 'noise_sha256': fsha(cl['noise_file']),
                       'source_files_sha256': manifest['files_sha256'], 'lambda_passthrough': float(json.load(open(lamf))['lambda'])}
    for arm_dir in ('d1_vcs_i2t', 'd2_vcs_t2i'):
        json.dump(ident('f40_s1001__d1_vcs_i2t'), open(f'{tmp}/f40_s1001/{arm_dir}/R27D_IDENTITY.json', 'w'), indent=2)
        open(f'{tmp}/f40_s1001/{arm_dir}/state_last.pt', 'w').close()
    code1, out1 = ru('f40_s1001__d1_vcs_i2t')
    code2, out2 = ru('f40_s1001__d2_vcs_t2i')
    shutil.rmtree(tmp)
    rs['run_unit_identity'] = dict(dry_fresh_rc=code0, dry_fresh_tail=out0[-200:], same_arm_rc=code1, same_arm_resumes_state_last='state_last.pt' in out1,
                                   other_arm_rc=code2, other_arm_msg=out2[-200:])
    good = (rs['snapshot_twice']['param_maxdiff'] == 0 and rs['snapshot_twice']['adam_maxdiff'] == 0 and rs['snapshot_twice']['rng_equal']
            and rs['disk_reload']['param_maxdiff'] <= TOL and rs['disk_reload']['adam_maxdiff'] <= TOL and rs['disk_reload']['rng_equal']
            and rs['disk_reload']['queue_ids_equal'] and rs['d1_smoke_state']['policy_from_arm'] == P.D_ARMS['d1_vcs_i2t']
            and code0 == 0 and code1 == 0 and rs['run_unit_identity']['same_arm_resumes_state_last'] and code2 != 0 and 'identity mismatch' in out2)
    if saved_arm is not None:
        good &= saved_arm == 'd1_vcs_i2t'
    ok('resume', good, **rs)

    # ---- truth isolation
    srcs = {n: inspect.getsource(f) for n, f in (('s_branch', R27.s_branch), ('noisy_branches', R27.noisy_branches), ('search', R27.search))}
    srcs['r27s_targets'] = inspect.getsource(ST); srcs['r27d_policy'] = inspect.getsource(P)
    import re
    bad = {n: [w for w in ('clean_labels', 'clean_label', 'gt_index', 'true_index', 'noise_file') if w in s] + re.findall(r'(?i)\btest\b', s) for n, s in srcs.items()}
    cone = {f'{arm}/{b}': steps[arm]['br'][b].get('min_cos_to_p') for arm in D_RUN for b in ('i2t', 't2i') if P.D_ARMS[arm][b] != 'identity'}
    good = all(v is not None and v >= ST.GAMMA - 1e-6 for v in cone.values()) and not any(bad.values()) and opt.data_name == 'f30k_precomp'
    ok('truth_isolation', good, min_cos_to_p=cone, gamma=ST.GAMMA, theta_deg=float(np.degrees(np.arccos(ST.GAMMA))), source_term_hits=bad,
       graph_equal_to_native='see identity_grad/cap_matches_S graph_equal', split='train (FIT) only; smoke DEV for arms_ran only; TEST never loaded')

    for hk in hooks:
        hk.remove()
    R27.generate_cap = orig_gen

    # ---- timing and memory
    tim = {}
    for arm in ['a1_iface', 's4_single_cap_vcs', 's6_single_cap_snce'] + D_RUN:
        restore(); opt.arm, opt.lam = arm, (0.0 if arm == 'a1_iface' else LAM)
        s = R.new_stats()
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
        R27.train(opt, A, B, lt, ut, ep, s, max_iters=30)
        torch.cuda.synchronize()
        tim[arm] = dict(ms_per_step=1000 * (time.time() - t0) / 30, peak_mem_GiB=torch.cuda.max_memory_allocated() / 2**30)
    for k in tim:
        tim[k]['ratio_vs_a1'] = tim[k]['ms_per_step'] / tim['a1_iface']['ms_per_step']
    ok('timing', True, gpu=torch.cuda.get_device_name(0), note='30 iterations incl. data loading; model A step only', **tim)

    # ---- arms ran
    arms_ok = {}
    for arm in D_RUN:
        e = json.load(open(f'{smoke}/{arm}/EPOCHS.json'))['cotrain']
        last = e[-1]
        # The package smoke is 1 co-train epoch: peer A's queue still holds only ID-less prefix slots in epoch 1
        # (known R27 prefix property, all arms alike: no_proto = 30/30), so finiteness is required where noisy steps ran.
        tr = last['train']
        ran = {f'{p}/{b}': tr[p].get(f'{b}_n') or 0 for p in ('A', 'B') for b in ('i2t', 't2i')}
        fin = all(np.isfinite(tr[p].get(f'{b}_raw_loss_mean', np.nan)) and np.isfinite(tr[p].get(f'{b}_enc_loss_mean', np.nan))
                  for p in ('A', 'B') for b in ('i2t', 't2i') if ran[f'{p}/{b}'] > 0)
        no_proto = {f'{p}/{b}': tr[p].get(f'{b}_no_proto') for p in ('A', 'B') for b in ('i2t', 't2i')}
        both_dirs = all(any(ran[f'{p}/{b}'] > 0 for p in ('A', 'B')) for b in ('i2t', 't2i'))
        corr = {b: (tr['B'].get(f'{b}_enc_loss_mean'), tr['B'].get(f'{b}_raw_loss_mean')) for b in ('i2t', 't2i')}
        routed = all((abs(corr[b][0] - corr[b][1]) > 0) == (P.D_ARMS[arm][b] != 'identity') for b in ('i2t', 't2i'))
        arms_ok[arm] = dict(finite=fin and both_dirs and routed, noisy_steps=ran, no_proto=no_proto, enc_vs_raw_peerB=corr,
                            enc_differs_only_on_corrected=routed, epochs=[x['epoch'] for x in e], dev_rsum=last['dev']['rsum'])
    ok('arms_ran', all(v['finite'] for v in arms_ok.values()), **arms_ok)

    checks = {k: v for k, v in res.items() if isinstance(v, dict)}
    out = {'all_pass': all(v['pass_'] for v in checks.values()) and code_now == manifest['files_sha256'],
           'code_files_sha256': code_now, 'code_matches_install_manifest': code_now == manifest['files_sha256'],
           'checks': checks, 'gpu': torch.cuda.get_device_name(0), 'job': os.environ.get('SLURM_JOB_ID'),
           'source_state': S0, 'source_state_sha256': fsha(S0), 'gate_script_sha256': fsha(os.path.abspath(__file__)),
           'raw_report_paths': [f'{smoke}/GPU_GATE.json'] + [f'{smoke}/{a}' for a in D_RUN]}
    json.dump(out, open(f'{smoke}/GPU_GATE.json', 'w'), indent=1, default=float)
    print('ALL_PASS', out['all_pass'], flush=True)
    sys.exit(0 if out['all_pass'] else 1)


if __name__ == '__main__':
    main()
