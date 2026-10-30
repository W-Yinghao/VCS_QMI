"""R27-O (ORACLE diagnostic) real-data GPU gate, modeled on R27-D gate/r27d_gate.py. Run from code_oracle (cwd).
Usage: r27o_gate.py SMOKE_DIR NOISE_FILE DATA_PATH GATE_STATE CODE_DIR -> SMOKE_DIR/GPU_GATE.json

From one restored real R27-S smoke state (both queues partly ID-bearing): one real training step of model A per arm with a
probe taking the per-direction gradient of that direction's loss w.r.t. the encoder + current-peer refiners
(torch.autograd.grad, retain_graph; the production backward is untouched). Arms: a1_iface, d2_vcs_t2i, o1/o2 with the
rescue ON and o1/o2 with the rescue switched OFF (r27o_rescue.ENABLED = False, gate only).
  rescue_off_equiv  O1-off == A1 and O2-off == D2 bitwise: full-step encoder/refiner grads at optimizer.step, params after
                    the step, AdamW state, RNG; per-direction probe grads; no rescue call
  i2t_untouched     rescue ON: i2t direction (loss value, graph, encoder+refiner grads) == A1 (O1) / D2 (O2) bitwise
  t2i_rescue        rescue ON: truth (caption j -> image j//5) present in 100% of t2i rows after rescue, exactly K=5 slots,
                    rows that already held the truth unchanged (IDs, slots, ID->mass), rows lacking it changed only in slot
                    5 (= truth), cols[inv] == img, A row sums 1 and A = production 1/K scatter rule; the refiner/SCE consumed
                    the rescued graph; >0 rows rescued; truth cross-checks (dataset _t2i_index, noise lock == loader pairing)
  both_branches     both directions trained once, both own refiners get grads, other peer no grad and its queue/IDs
                    unchanged, one optimizer.step, refiner forwards equal to A1, stop-grad targets
  resume            snapshot restore twice bitwise; reload from disk; smoke checkpoint arm; runner identity (same arm
                    resumes, other arm refused)
  capped_15deg      O2 / D2 t2i capped targets within 15 deg of sg P (min cos >= GAMMA)
  truth_isolation   truth-related tokens appear only in r27o_rescue among the noisy-branch sources; hook gated by arm
  timing            ms/step (30 iterations, same state) and peak memory per arm
  arms_ran          the two O smoke arms finished their smoke epoch with finite losses; rescue counters logged
"""
import os, sys, json, copy, hashlib, time, inspect, subprocess, tempfile, shutil, re
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(sys.argv[5]))
import r25_run
import r25_co_train as R
import r27_co_train as R27
import r27_interface as I
import r27s_targets as ST
import r27d_policy as P
import r27o_rescue as RX
from data import get_dataset
from vse import VSE
from vocab import deserialize_vocab
from co_train import eval_train, split_prob

res = {}
O_RUN = ['o1_rescue_native_t2i', 'o2_rescue_vcs_t2i']
REF = {'o1_rescue_native_t2i': 'a1_iface', 'o2_rescue_vcs_t2i': 'd2_vcs_t2i'}
LAM = 1.1448122442850819
TOL = 1e-6
HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(os.path.dirname(HERE), 'tools', 'run_unit_o.py')
CELLS = '/home/infres/yinwang/CS_QMI/handoff/r27_qtarget_20261029/CELLS.json'
LAMF = '/home/infres/yinwang/CS_QMI/handoff/r27_qtarget_20261029/LAMBDA_flickr.json'


def ok(name, cond, **info):
    res[name] = dict(pass_=bool(cond), **info)
    print(name, 'PASS' if cond else 'FAIL', json.dumps(info, default=float)[:3000], flush=True)


def h(t):
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()


def fsha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def gdiff(a, b):
    m = 0.0
    for x, y in zip(a, b):
        if (x is None) != (y is None):
            return float('inf')
        if x is not None:
            m = max(m, float((x - y).abs().max()))
    return m


def mass_rows(S):
    """per row: sorted list of (id, mass) with mass > 0"""
    out = []
    A, cols = S['A'].cpu(), S['cols'].cpu()
    for i in range(A.shape[0]):
        nz = torch.nonzero(A[i] > 0).flatten()
        out.append(sorted((int(cols[j]), float(A[i, j])) for j in nz))
    return out


def main():
    smoke, noise, dpath, S0, code_dir = sys.argv[1:6]
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
    NOISE = np.load(noise)
    CLEAN_T2I = np.asarray(noisy_loader.dataset._t2i_index, dtype=np.int64)   # evaluator-side truth (gate only)

    def restore():
        for m, s in ((A, snap[0]), (B, snap[1])):
            m.img_enc.load_state_dict(s['img_enc']); m.txt_enc.load_state_dict(s['txt_enc'])
            m.optimizer.load_state_dict(copy.deepcopy(s['optimizer']))
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

    rec = []
    orig_rx = RX.rescue_t2i

    def rec_rx(S, cap_gidx, k, epoch, stats=None):
        S2, info = orig_rx(S, cap_gidx, k, epoch, stats)
        rec.append(dict(Sin={kk: v.clone() for kk, v in S.items()}, Sout={kk: v.clone() for kk, v in S2.items()},
                        cap=cap_gidx.clone(), same_obj=S2 is S, info={kk: (v.clone() if torch.is_tensor(v) else v) for kk, v in info.items()}))
        return S2, info
    RX.rescue_t2i = rec_rx

    uids_rec = []
    orig_nb = R27.noisy_branches

    def rec_nb(opt_, net, net2, *a, **k):
        uids_rec.append(tuple(u.clone() for u in a[-2]))      # positional: ..., mi, mt, stats, uids, timed
        return orig_nb(opt_, net, net2, *a, **k)
    R27.noisy_branches = rec_nb

    ref_calls = [0]
    hooks = [m.register_forward_hook(lambda *a: ref_calls.__setitem__(0, ref_calls[0] + 1))
             for m in (A.i2t_refiner_node, A.t2i_refiner_node)]

    def one_step(arm, rescue=True, probe_grads=True, do_restore=True):
        if do_restore:
            restore()
        opt.arm, opt.lam = arm, (0.0 if arm == 'a1_iface' else LAM)
        RX.ENABLED = rescue
        cap, br = {'nstep': 0}, {}
        ref_calls[0] = 0; rec.clear(); uids_rec.clear()
        params = list(A.params) + own_ref()

        def probe(branch, dd):
            e = dict(loss=float(dd['loss']), l_nat=float(dd['l_nat']), target=None if dd['target'] is None else dd['target'].detach().clone(),
                     graphA=dd['S']['A'].detach().clone(), cols=dd['S']['cols'].detach().clone(), img=dd['S']['img'].detach().clone(),
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
            cap['adam_step'] = [float(v) for p in params for kk, v in A.optimizer.state.get(p, {}).items() if kk == 'step']
            return out
        A.optimizer.step = step_probe
        qh = (h(B.memory_queue.img_queue), h(B.memory_queue.txt_queue), {k: h(v) for k, v in ids_of(B.memory_queue).items()})
        stats = R.new_stats()
        try:
            R27.train(opt, A, B, lt, ut, ep, stats, max_iters=1, probe=probe)
        finally:
            A.optimizer.step = orig
            RX.ENABLED = True
        cap.update(rng=R.rng_state(), stats=stats, br=br, ref_calls=ref_calls[0], rescue_calls=list(rec), uids=list(uids_rec),
                   peer_queue_same=(h(B.memory_queue.img_queue), h(B.memory_queue.txt_queue), {k: h(v) for k, v in ids_of(B.memory_queue).items()}) == qh,
                   own_ids={k: v.clone() for k, v in ids_of(A.memory_queue).items()})
        return cap

    def rng_eq(a, b):
        na, nb_ = a['rng']['numpy'], b['rng']['numpy']
        return (bool(torch.equal(a['rng']['torch'], b['rng']['torch'])) and all(torch.equal(x, y) for x, y in zip(a['rng']['cuda'], b['rng']['cuda']))
                and na[0] == nb_[0] and np.array_equal(na[1], nb_[1]) and tuple(na[2:]) == tuple(nb_[2:]) and a['rng']['python'] == b['rng']['python'])

    steps = {'a1_iface': one_step('a1_iface'), 'd2_vcs_t2i': one_step('d2_vcs_t2i')}
    for arm in O_RUN:
        steps[arm] = one_step(arm, rescue=True)
        steps[arm + '/off'] = one_step(arm, rescue=False)

    # ---- (1) rescue off == reference, bitwise, full step
    r1, good = {}, True
    for arm in O_RUN:
        a, b = steps[arm + '/off'], steps[REF[arm]]
        e = dict(ref=REF[arm], rescue_calls=len(a['rescue_calls']), enc_grad_maxdiff=gdiff(a['enc'], b['enc']),
                 ref_i2t_grad_maxdiff=gdiff(a['ref_i2t'], b['ref_i2t']), ref_t2i_grad_maxdiff=gdiff(a['ref_t2i'], b['ref_t2i']),
                 param_maxdiff=float((a['params'] - b['params']).abs().max()), adam_maxdiff=float((a['adam'] - b['adam']).abs().max()),
                 adam_len_equal=a['adam'].numel() == b['adam'].numel(), adam_step_equal=a['adam_step'] == b['adam_step'], rng_equal=rng_eq(a, b),
                 dir_grad_maxdiff={d: gdiff(a['br'][d]['grads'], b['br'][d]['grads']) for d in ('i2t', 't2i')},
                 dir_lnat_absdiff={d: abs(a['br'][d]['l_nat'] - b['br'][d]['l_nat']) for d in ('i2t', 't2i')},
                 graph_equal={d: bool(torch.equal(a['br'][d]['graphA'], b['br'][d]['graphA']) and torch.equal(a['br'][d]['cols'], b['br'][d]['cols'])) for d in ('i2t', 't2i')})
        if arm == 'o2_rescue_vcs_t2i':
            e['t2i_target_maxdiff'] = float((a['br']['t2i']['target'] - b['br']['t2i']['target']).abs().max())
        e['bitwise'] = (e['rescue_calls'] == 0 and e['enc_grad_maxdiff'] == 0 and e['ref_i2t_grad_maxdiff'] == 0 and e['ref_t2i_grad_maxdiff'] == 0
                        and e['param_maxdiff'] == 0 and e['adam_maxdiff'] == 0 and e['adam_len_equal'] and e['adam_step_equal'] and e['rng_equal']
                        and all(v == 0 for v in e['dir_grad_maxdiff'].values()) and all(v == 0 for v in e['dir_lnat_absdiff'].values())
                        and all(e['graph_equal'].values()) and e.get('t2i_target_maxdiff', 0) == 0)
        good &= e['bitwise']
        r1[arm] = e
    ok('rescue_off_equiv', good, **r1)

    # ---- (2a) rescue on: i2t untouched
    r2, good = {}, True
    for arm in O_RUN:
        a, b = steps[arm]['br']['i2t'], steps[REF[arm]]['br']['i2t']
        e = dict(ref=REF[arm], grad_maxdiff=gdiff(a['grads'], b['grads']), l_nat_absdiff=abs(a['l_nat'] - b['l_nat']),
                 graph_equal=bool(torch.equal(a['graphA'], b['graphA']) and torch.equal(a['cols'], b['cols']) and torch.equal(a['img'], b['img'])),
                 none_pattern_equal=[x is None for x in a['grads']] == [x is None for x in b['grads']],
                 rescue_calls=len(steps[arm]['rescue_calls']))
        e['bitwise'] = e['grad_maxdiff'] == 0 and e['l_nat_absdiff'] == 0 and e['graph_equal'] and e['none_pattern_equal']
        good &= e['bitwise'] and e['rescue_calls'] == 1      # one call = t2i only, never i2t
        r2[arm] = e
    ok('i2t_untouched', good, **r2)

    # ---- (2b) rescue on: t2i rows
    r3, good = {}, True
    for arm in O_RUN:
        c = steps[arm]
        rc = c['rescue_calls'][0]
        Sin, Sout, capg = rc['Sin'], rc['Sout'], rc['cap']
        Bn, K = Sout['img'].shape
        tru = capg // 5
        pres_in = (Sin['img'] == tru.unsqueeze(1)).any(1)
        pres_out = (Sout['img'] == tru.unsqueeze(1)).any(1)
        need = ~pres_in
        unchanged_ok = bool(torch.equal(Sout['img'][pres_in], Sin['img'][pres_in]) and torch.equal(Sout['slots'][pres_in], Sin['slots'][pres_in]))
        mi, mo = mass_rows(Sin), mass_rows(Sout)
        unchanged_mass = all(mi[i] == mo[i] for i in range(Bn) if bool(pres_in[i]))
        changed_only5 = bool(torch.equal(Sout['img'][need][:, :K - 1], Sin['img'][need][:, :K - 1]) and torch.equal(Sout['img'][need][:, K - 1], tru[need])
                             and torch.equal(Sout['slots'][need][:, :K - 1], Sin['slots'][need][:, :K - 1]) and bool((Sout['slots'][need][:, K - 1] == -1).all()))
        diff_rows = (Sout['img'] != Sin['img']).any(1)
        A_ref = torch.zeros_like(Sout['A']); A_ref.scatter_add_(1, Sout['inv'], torch.full_like(Sout['inv'], 1.0 / K, dtype=torch.float32))
        consumed = bool(torch.equal(c['br']['t2i']['graphA'], Sout['A']) and torch.equal(c['br']['t2i']['cols'], Sout['cols']) and torch.equal(c['br']['t2i']['img'], Sout['img']))
        u0, u1 = c['uids'][0]
        cpu0, cpu1 = u0.cpu().numpy(), u1.cpu().numpy()
        e = dict(rows=Bn, K=K, rescued=int(need.sum()), already_present=int(pres_in.sum()),
                 stats_counts=dict(rows=c['stats']['cnt_t2i_rescue_rows'], present=c['stats']['cnt_t2i_truth_present_rows'], rescued=c['stats']['cnt_t2i_rescued_rows']),
                 truth_present_after_frac=float(pres_out.float().mean()), changed_rows_equal_need=bool(torch.equal(diff_rows, need)),
                 unchanged_rows_ids_equal=unchanged_ok, unchanged_rows_mass_equal=unchanged_mass, changed_only_slot5_to_truth=changed_only5,
                 cols_inv_img=bool(torch.equal(Sout['cols'][Sout['inv']], Sout['img'])), A_rowsum_maxdev=float((Sout['A'].sum(1) - 1).abs().max()),
                 A_equals_rule=bool(torch.equal(A_ref, Sout['A'])), refiner_consumed_rescued_graph=consumed,
                 cap_ids_are_query_uids0=bool(torch.equal(capg, u0)),
                 truth_matches_dataset_clean_index=bool((CLEAN_T2I[cpu0] == cpu0 // 5).all()),
                 noise_lock_equals_loader_pairing=bool((NOISE[cpu0] == cpu1).all()),
                 noisy_query_frac=float((NOISE[cpu0] != cpu0 // 5).mean()),
                 t2i_lnat_vs_ref=(c['br']['t2i']['l_nat'], steps[REF[arm]]['br']['t2i']['l_nat']))
        e['pass'] = (K == 5 and e['rescued'] > 0 and e['truth_present_after_frac'] == 1.0 and e['changed_rows_equal_need'] and unchanged_ok and unchanged_mass
                     and changed_only5 and e['cols_inv_img'] and e['A_rowsum_maxdev'] <= 1e-6 and e['A_equals_rule'] and consumed
                     and e['cap_ids_are_query_uids0'] and e['truth_matches_dataset_clean_index'] and e['noise_lock_equals_loader_pairing']
                     and e['stats_counts'] == dict(rows=Bn, present=e['already_present'], rescued=e['rescued']))
        good &= e['pass']
        r3[arm] = e
    ok('t2i_rescue', good, **r3)

    # ---- (3) both branches
    bb, good = {}, True
    for arm in ['a1_iface', 'd2_vcs_t2i'] + O_RUN:
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

    # ---- (5) capped targets within 15 deg
    cone = {a: steps[a]['br']['t2i'].get('min_cos_to_p') for a in ('o2_rescue_vcs_t2i', 'd2_vcs_t2i')}
    ok('capped_15deg', all(v is not None and v >= ST.GAMMA - 1e-6 for v in cone.values()), min_cos_to_p=cone, gamma=ST.GAMMA,
       theta_deg=float(np.degrees(np.arccos(ST.GAMMA))))

    # ---- (4) resume
    rs = {}
    a = one_step('o2_rescue_vcs_t2i', probe_grads=False); b = one_step('o2_rescue_vcs_t2i', probe_grads=False)
    rs['snapshot_twice'] = dict(param_maxdiff=float((a['params'] - b['params']).abs().max()), adam_maxdiff=float((a['adam'] - b['adam']).abs().max()), rng_equal=rng_eq(a, b))
    R.load_full_state(S0, opt, A, B)
    for m, prob, pred in ((A, prob_A, pred_A), (B, prob_B, pred_B)):
        m.avg_clean_prob = prob[pred].mean()
    for m in (A, B):
        for p in list(m.params) + list(m.i2t_refiner_node.parameters()) + list(m.t2i_refiner_node.parameters()):
            p.grad = None
    R.set_rng_state(snap[2]); R27._PEER.clear(); R27._PEER[id(A)] = 0; R27._PEER[id(B)] = 1
    c = one_step('o2_rescue_vcs_t2i', probe_grads=False, do_restore=False)
    rs['disk_reload'] = dict(param_maxdiff=float((a['params'] - c['params']).abs().max()), adam_maxdiff=float((a['adam'] - c['adam']).abs().max()),
                             rng_equal=rng_eq(a, c),
                             queue_ids_equal=all(torch.equal(a['own_ids'][k], c['own_ids'][k]) for k in a['own_ids']) and set(a['own_ids']) == set(c['own_ids']))
    saved = {}
    for arm in O_RUN:
        dst = torch.load(f'{smoke}/{arm}/state_last.pt', map_location='cpu', weights_only=False)
        so = dst.get('opt')
        saved[arm] = dict(saved_arm=so.get('arm') if isinstance(so, dict) else getattr(so, 'arm', None),
                          saved_rescue_flag=so.get('r27o_rescue_t2i') if isinstance(so, dict) else None,
                          policy_from_arm={bb_: R27.direction_kind(arm, bb_, R27.S_ARMS) for bb_ in ('i2t', 't2i')})
    rs['smoke_states'] = saved
    tmp = tempfile.mkdtemp(prefix='r27o_gate_id_')
    sys.path.insert(0, os.path.dirname(RUNNER)); import run_unit_o as RU
    base = [sys.executable, '-B', RUNNER, '--cells', CELLS, '--lambda-file', LAMF, '--code-dir', code_dir, '--data-path', dpath, '--output-root', tmp]

    def ru(unit):
        p = subprocess.run(base + ['--unit', unit], capture_output=True, text=True)
        return p.returncode, (p.stdout + p.stderr).strip()[-400:]
    code0, out0 = ru('f40_s1001__o1_rescue_native_t2i')
    ident = RU.identity_for('f40_s1001__o1_rescue_native_t2i', CELLS, LAMF, code_dir)[-1]
    for arm_dir in O_RUN:
        os.makedirs(f'{tmp}/f40_s1001/{arm_dir}')
        json.dump(ident, open(f'{tmp}/f40_s1001/{arm_dir}/R27O_IDENTITY.json', 'w'), indent=2)
        open(f'{tmp}/f40_s1001/{arm_dir}/state_last.pt', 'w').close()
    os.makedirs(f'{tmp}/f60_s1001/o1_rescue_native_t2i'); open(f'{tmp}/f60_s1001/o1_rescue_native_t2i/state_last.pt', 'w').close()
    code1, out1 = ru('f40_s1001__o1_rescue_native_t2i')
    code2, out2 = ru('f40_s1001__o2_rescue_vcs_t2i')
    code3, out3 = ru('f60_s1001__o1_rescue_native_t2i')
    code4, out4 = ru('f40_s1001__d2_vcs_t2i')
    shutil.rmtree(tmp)
    rs['run_unit_identity'] = dict(dry_fresh_rc=code0, dry_fresh_tail=out0[-300:], same_arm_rc=code1, same_arm_resumes_state_last='state_last.pt' in out1,
                                   other_arm_rc=code2, other_arm_msg=out2[-200:], unidentified_state_rc=code3, unidentified_msg=out3[-200:],
                                   non_oracle_unit_rc=code4, non_oracle_msg=out4[-200:])
    good = (rs['snapshot_twice']['param_maxdiff'] == 0 and rs['snapshot_twice']['adam_maxdiff'] == 0 and rs['snapshot_twice']['rng_equal']
            and rs['disk_reload']['param_maxdiff'] <= TOL and rs['disk_reload']['adam_maxdiff'] <= TOL and rs['disk_reload']['rng_equal']
            and rs['disk_reload']['queue_ids_equal']
            and all(v['saved_arm'] == k and v['saved_rescue_flag'] is True and v['policy_from_arm'] == RX.O_ARMS[k] for k, v in saved.items())
            and code0 == 0 and code1 == 0 and rs['run_unit_identity']['same_arm_resumes_state_last'] and code2 != 0 and 'identity mismatch' in out2
            and code3 != 0 and 'Unidentified' in out3 and code4 != 0)
    ok('resume', good, **rs)

    # ---- truth isolation
    srcs = {n: inspect.getsource(f) for n, f in (('s_branch', R27.s_branch), ('noisy_branches', orig_nb),
                                                 ('search', I.search), ('reencode', I.reencode))}
    srcs['r27s_targets'] = inspect.getsource(ST); srcs['r27d_policy'] = inspect.getsource(P)
    words = ('clean_labels', 'clean_label', 'gt_index', 'true_index', 'noise_file', '_t2i_index', '_labels', 'IM_DIV', 'true_image', '// 5', '//5')
    bad = {n: [w for w in words if w in s] + re.findall(r'(?i)\btest\b', s) for n, s in srcs.items()}
    nb = srcs['noisy_branches']
    hook_lines = [l.strip() for l in nb.splitlines() if 'RX.' in l]
    rx_src = inspect.getsource(RX)
    good = (not any(bad.values()) and hook_lines == ["if branch == 't2i' and opt.arm in O_ARMS and RX.ENABLED:",
                                                     "S, _ = RX.rescue_t2i(S, uids[0], opt.k, opt.r27_epoch, stats)   # uids[0] = global caption IDs of the query batch"]
            and 'IM_DIV' in rx_src and opt.data_name == 'f30k_precomp' and set(RX.O_ARMS) == set(O_RUN))
    ok('truth_isolation', good, source_term_hits=bad, hook_lines=hook_lines, truth_reader='r27o_rescue.true_image (caption j -> image j // 5)',
       split='train (FIT) only; smoke DEV for arms_ran only; TEST never loaded')

    for hk in hooks:
        hk.remove()
    RX.rescue_t2i = orig_rx
    R27.noisy_branches = orig_nb

    # ---- timing and memory
    tim = {}
    for arm in ['a1_iface', 'd2_vcs_t2i'] + O_RUN:
        restore(); opt.arm, opt.lam = arm, (0.0 if arm == 'a1_iface' else LAM)
        s = R.new_stats()
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
        R27.train(opt, A, B, lt, ut, ep, s, max_iters=30)
        torch.cuda.synchronize()
        tim[arm] = dict(ms_per_step=1000 * (time.time() - t0) / 30, peak_mem_GiB=torch.cuda.max_memory_allocated() / 2**30,
                        rescued_rows=s.get('cnt_t2i_rescued_rows', 0), rescue_rows=s.get('cnt_t2i_rescue_rows', 0))
    for k in tim:
        tim[k]['ratio_vs_a1'] = tim[k]['ms_per_step'] / tim['a1_iface']['ms_per_step']
    ok('timing', True, gpu=torch.cuda.get_device_name(0), note='30 iterations incl. data loading; model A step only', **tim)

    # ---- arms ran (smoke)
    arms_ok = {}
    for arm in O_RUN:
        e = json.load(open(f'{smoke}/{arm}/EPOCHS.json'))['cotrain']
        last = e[-1]; tr = last['train']
        ran = {f'{p}/{b}': tr[p].get(f'{b}_n') or 0 for p in ('A', 'B') for b in ('i2t', 't2i')}
        fin = all(np.isfinite(tr[p].get(f'{b}_raw_loss_mean', np.nan)) and np.isfinite(tr[p].get(f'{b}_enc_loss_mean', np.nan))
                  for p in ('A', 'B') for b in ('i2t', 't2i') if ran[f'{p}/{b}'] > 0)
        both_dirs = all(any(ran[f'{p}/{b}'] > 0 for p in ('A', 'B')) for b in ('i2t', 't2i'))
        cnt = {p: {k: tr[p].get(k) for k in ('cnt_t2i_rescue_rows', 'cnt_t2i_truth_present_rows', 'cnt_t2i_rescued_rows')} for p in ('A', 'B')}
        logged = all(cnt[p]['cnt_t2i_rescue_rows'] for p in ('A', 'B') if ran[f'{p}/t2i'] > 0) and any(ran[f'{p}/t2i'] > 0 for p in ('A', 'B'))
        cons = all((cnt[p]['cnt_t2i_truth_present_rows'] or 0) + (cnt[p]['cnt_t2i_rescued_rows'] or 0) == (cnt[p]['cnt_t2i_rescue_rows'] or 0) for p in ('A', 'B'))
        no_proto = {f'{p}/{b}': tr[p].get(f'{b}_no_proto') for p in ('A', 'B') for b in ('i2t', 't2i')}
        arms_ok[arm] = dict(finite=fin and both_dirs, rescue_counters=cnt, counters_logged=logged, counters_consistent=cons, noisy_steps=ran,
                            no_proto=no_proto, epochs=[x['epoch'] for x in e], dev_rsum=last['dev']['rsum'])
    ok('arms_ran', all(v['finite'] and v['counters_logged'] and v['counters_consistent'] for v in arms_ok.values()), **arms_ok)

    checks = {k: v for k, v in res.items() if isinstance(v, dict)}
    out = {'all_pass': all(v['pass_'] for v in checks.values()) and code_now == manifest['files_sha256'],
           'oracle_diagnostic': 'R27-O uses injected truth in the t2i candidate rescue; never part of a SOTA table',
           'code_files_sha256': code_now, 'code_matches_install_manifest': code_now == manifest['files_sha256'],
           'checks': checks, 'gpu': torch.cuda.get_device_name(0), 'job': os.environ.get('SLURM_JOB_ID'),
           'source_state': S0, 'source_state_sha256': fsha(S0), 'gate_script_sha256': fsha(os.path.abspath(__file__)),
           'runner_sha256': fsha(RUNNER), 'raw_report_paths': [f'{smoke}/GPU_GATE.json'] + [f'{smoke}/{a}' for a in O_RUN]}
    json.dump(out, open(f'{smoke}/GPU_GATE.json', 'w'), indent=1, default=float)
    print('ALL_PASS', out['all_pass'], flush=True)
    sys.exit(0 if out['all_pass'] else 1)


if __name__ == '__main__':
    main()
