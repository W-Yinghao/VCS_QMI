#!/usr/bin/env python3
"""R29 CPU checks (float64, synthetic data; no training data, no GPU). Usage: cpu_check.py CODE_DIR REFERENCE_DIR OUT_JSON

 1 package reference self-check (listwise_vcs_reference.run_checks) reproduces MATH_CHECKS.json all_pass
 2 L: production row loss / critic / autograd gradient == reference loss_and_grad (one-hot and soft targets)
 3 J identity S - J = E_M[(T - T*)^2], M = (a* + Unif(K))/2, with the PRODUCTION J and T; J <= S; one-hot optimum (K-1)/(K+1)
 4 B: production == direct formula -E_P log D - E_Q log(1-D), D = (1+T)/2, T = reference critic; gradient vs finite differences
 5 P: production == direct tanh(s/tau + b) formula; gradient (s and b) vs finite differences; reads only its own score
 6 N: == standard InfoNCE (F.cross_entropy and the native exp/sum CE expression) for diagonal targets; multi-positive CE
 7 permutation invariance (candidate slots; joint batch permutation of the full two-way alignment value) for N/P/L/B
 8 logit-shift invariance (gradient rows sum to 0) for N/L/B
 9 phase reduction rule: warmup = 0.5 (mean i2t + mean t2i); co-train = sum i2t + sum t2i
10 H unchanged: align('h_native') returns infonce(sims) / net.criterion(sims, labels=None) (call identity);
   r29 train()/warmup_step() source differs from r27 train() / VSE.train(mode='warmup') only at the alignment term + report-only lines
11 targets from observed ids: identity for distinct ids, row-stochastic, symmetric support; no truth fields referenced by r29 code
"""
import sys, os, json, inspect, importlib.util, difflib, re, hashlib
import numpy as np
import torch
import torch.nn.functional as F

code, refdir, out = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, code)
import r29_align as AL
torch.set_default_dtype(torch.float64)
res = {}


def ok(name, cond, **info):
    res[name] = dict(pass_=bool(cond), **info)
    print(name, 'PASS' if cond else 'FAIL', json.dumps(info, default=float)[:1500], flush=True)


spec = importlib.util.spec_from_file_location('listwise_vcs_reference', os.path.join(refdir, 'listwise_vcs_reference.py'))
REF = importlib.util.module_from_spec(spec); spec.loader.exec_module(REF)
ref_sha = hashlib.sha256(open(os.path.join(refdir, 'listwise_vcs_reference.py'), 'rb').read()).hexdigest()
mc = json.load(open(os.path.join(refdir, 'MATH_CHECKS.json')))
rc = REF.run_checks()
ok('01_reference_selfcheck', rc['all_pass'] and mc['all_pass'], reference_sha256=ref_sha, rerun=rc)

rng = np.random.default_rng(29)
tau = AL.TAU


def rowloss_and_grad(kind, S, A, b=None):
    St = torch.tensor(S, requires_grad=True)
    bt = None if b is None else torch.tensor(float(b), requires_grad=True)
    r = AL.row_losses(kind, St, torch.tensor(A), bt)
    g = torch.autograd.grad(r.sum(), [St] + ([bt] if bt is not None else []))
    return r.detach().numpy(), g[0].numpy(), (None if bt is None else float(g[1]))


# ---- 2 L vs reference
errs = {}
for name, k, n in (('k7', 7, 9), ('k128', 128, 16)):
    X = rng.normal(size=(n, k)) * 3.0
    for tname, A in (('onehot', np.eye(k)[rng.integers(k, size=n)]), ('soft', rng.dirichlet(np.ones(k), size=n))):
        lref, gref = REF.loss_and_grad(X, A)
        r, g, _ = rowloss_and_grad('l', X * tau, A)
        Tp = AL.critic_T('l', torch.tensor(X * tau)).numpy()
        errs[f'{name}_{tname}'] = dict(loss=float(np.abs(r - lref).max()), grad=float(np.abs(g * tau - gref).max()),
                                       critic=float(np.abs(Tp - REF.critic(REF.probabilities(X))).max()))
ok('02_L_vs_reference', all(v < 1e-10 for e in errs.values() for v in e.values()), max_abs=errs)

# ---- 3 J identity with production T/J
k, n = 9, 11
truth = rng.dirichlet(np.ones(k), size=n)
X = rng.normal(size=(n, k)) * 2
T = AL.critic_T('l', torch.tensor(X * tau)).numpy()
eta = REF.critic(truth)
Jp = AL.vcs_J_rows(torch.tensor(T), torch.tensor(truth)).numpy()
Sp = AL.vcs_J_rows(torch.tensor(eta), torch.tensor(truth)).numpy()
gap = (0.5 * (truth + 1.0 / k) * (T - eta) ** 2).sum(1)
oh = np.eye(k)[0]
J1 = float(AL.vcs_J_rows(torch.tensor(REF.critic(oh))[None], torch.tensor(oh)[None]))
ok('03_J_identity', np.abs(Sp - Jp - gap).max() < 1e-12 and (Jp <= Sp + 1e-12).all() and abs(J1 - (k - 1) / (k + 1)) < 1e-12,
   gap_err=float(np.abs(Sp - Jp - gap).max()), onehot_opt_err=abs(J1 - (k - 1) / (k + 1)))


def fd_grad(kind, S, A, b=None, eps=1e-6):
    g = np.zeros_like(S)
    f = lambda Z, bb=b: AL.row_losses(kind, torch.tensor(Z), torch.tensor(A), None if bb is None else torch.tensor(bb)).sum().item()
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            Zp, Zm = S.copy(), S.copy(); Zp[i, j] += eps; Zm[i, j] -= eps
            g[i, j] = (f(Zp) - f(Zm)) / (2 * eps)
    gb = None if b is None else (f(S, b + eps) - f(S, b - eps)) / (2 * eps)
    return g, gb


# ---- 4 B
k, n = 7, 6
S = rng.normal(size=(n, k)) * 0.1
A = rng.dirichlet(np.ones(k), size=n)
Tref = REF.critic(REF.probabilities(S / tau))
D = (1 + Tref) / 2
direct = -(A * np.log(D)).sum(1) - np.log(1 - D).mean(1)
r, g, _ = rowloss_and_grad('b', S, A)
gf, _ = fd_grad('b', S, A)
ok('04_B_direct_formula', np.abs(r - direct).max() < 1e-10 and np.abs(g - gf).max() < 1e-6,
   loss_err=float(np.abs(r - direct).max()), grad_fd_err=float(np.abs(g - gf).max()))

# ---- 5 P
bval = -0.7
Tp = np.tanh(S / tau + bval)
directP = (-(A * (Tp - 0.5 * Tp ** 2)).sum(1) + (Tp + 0.5 * Tp ** 2).mean(1))
r, g, gb = rowloss_and_grad('p', S, A, bval)
gf, gbf = fd_grad('p', S, A, bval)
S2 = S.copy(); S2[:, 1:] += rng.normal(size=(n, k - 1))
Tc = AL.critic_T('p', torch.tensor(S2), torch.tensor(bval)).numpy()  # col 0 unchanged -> T col 0 unchanged (own score only)
Tp = AL.critic_T('p', torch.tensor(S), torch.tensor(bval)).numpy()
ok('05_P_direct_formula', np.abs(r - directP).max() < 1e-10 and np.abs(g - gf).max() < 1e-6 and abs(gb - gbf) < 1e-6
   and np.abs(Tc[:, 0] - Tp[:, 0]).max() == 0.0,
   loss_err=float(np.abs(r - directP).max()), grad_fd_err=float(np.abs(g - gf).max()), b_grad_fd_err=abs(gb - gbf))

# ---- 6 N == standard InfoNCE
B = 12
St = torch.tensor(rng.normal(size=(B, B)) * 0.3)
I_ = torch.eye(B)
r1, r2 = AL.two_way('n', St, I_)
ce1, ce2 = F.cross_entropy(St / tau, torch.arange(B)), F.cross_entropy(St.t() / tau, torch.arange(B))
e = (St / tau).exp(); nat1 = -(e / e.sum(1, keepdim=True)).diag().log().mean(); nat2 = -(e.t() / e.t().sum(1, keepdim=True)).diag().log().mean()
Am = torch.tensor(rng.dirichlet(np.ones(B), size=B))
mp = AL.row_losses('n', St, Am)
mp_ref = -(Am * torch.log_softmax(St / tau, 1)).sum(1)
ok('06_N_infonce', max(abs(float(r1.mean() - ce1)), abs(float(r2.mean() - ce2)), abs(float(r1.mean() - nat1)), abs(float(r2.mean() - nat2))) < 1e-10
   and float((mp - mp_ref).abs().max()) < 1e-12, ce_err=max(abs(float(r1.mean() - ce1)), abs(float(r2.mean() - ce2))),
   native_expr_err=max(abs(float(r1.mean() - nat1)), abs(float(r2.mean() - nat2))))

# ---- 7 permutation invariance
perm_err = {}
S = rng.normal(size=(8, 10)) * 0.2
A = rng.dirichlet(np.ones(10), size=8)
pm = rng.permutation(10)
for kind in 'nplb':
    bb = 0.3 if kind == 'p' else None
    a0, _, _ = rowloss_and_grad(kind, S, A, bb); a1, _, _ = rowloss_and_grad(kind, S[:, pm], A[:, pm], bb)
    perm_err[kind + '_slots'] = float(np.abs(a0 - a1).max())


class _Opt:  # minimal opt / net stand-ins
    pass


obs = torch.tensor([3, 5, 3, 9, 1, 5, 7, 8, 2, 3, 4, 6])
Sb = torch.tensor(rng.normal(size=(12, 12)) * 0.3)
jp = torch.tensor(rng.permutation(12))
for arm in ('n_infonce', 'p_pointwise_vcs', 'l_listwise_vcs', 'b_listwise_bce'):
    o = _Opt(); o.r29_arm = arm
    net = _Opt(); net.r29_b = torch.tensor(-0.4)
    for ph in ('warmup', 'cotrain'):
        v0 = float(AL.align(o, net, ph, Sb, obs)); v1 = float(AL.align(o, net, ph, Sb[jp][:, jp], obs[jp]))
        perm_err[f'{arm}_{ph}_batch'] = abs(v0 - v1)
ok('07_permutation_invariance', all(v < 1e-10 for v in perm_err.values()), max_abs=perm_err)

# ---- 8 shift invariance
sh = {}
for kind in 'nlb':
    _, g, _ = rowloss_and_grad(kind, S, A)
    sh[kind] = float(np.abs(g.sum(1)).max())
ok('08_logit_shift_gradient_sum', all(v < 1e-12 for v in sh.values()), max_abs=sh)

# ---- 9 reduction rule
red = {}
A12 = AL.target_from_obs(obs, Sb.dtype)
for arm, kind in AL.KIND.items():
    o = _Opt(); o.r29_arm = arm
    net = _Opt(); net.r29_b = torch.tensor(-0.4)
    r1, r2 = AL.two_way(kind, Sb, A12, net.r29_b)
    red[arm] = max(abs(float(AL.align(o, net, 'warmup', Sb, obs) - 0.5 * (r1.mean() + r2.mean()))),
                   abs(float(AL.align(o, net, 'cotrain', Sb, obs) - (r1.sum() + r2.sum()))))
ok('09_phase_reduction', all(v < 1e-12 for v in red.values()), max_abs=red, rule={'warmup': '0.5*(mean_i2t+mean_t2i)', 'cotrain': 'sum_i2t+sum_t2i'})

# ---- 10 H unchanged
calls = []
orig_infonce = AL.infonce
AL.infonce = lambda s: calls.append(('infonce', s)) or 'NATIVE_W'
o = _Opt(); o.r29_arm = 'h_native'
net = _Opt(); net.criterion = lambda s, labels='x': calls.append(('criterion', s, labels)) or 'NATIVE_C'
w = AL.align(o, net, 'warmup', Sb, obs); c = AL.align(o, net, 'cotrain', Sb, obs)
AL.infonce = orig_infonce
h_call = (w == 'NATIVE_W' and c == 'NATIVE_C' and calls[0][0] == 'infonce' and calls[0][1] is Sb and calls[1][0] == 'criterion'
          and calls[1][1] is Sb and calls[1][2] is None and AL.infonce is __import__('vse').infonce)


def src_lines(f):
    return [l.rstrip() for l in inspect.getsource(f).splitlines()]


R27 = importlib.import_module('r27_co_train')
R29 = importlib.import_module('r29_co_train')
V = importlib.import_module('vse')
norm = lambda ls: [re.sub(r'\bR27\.', '', l.split('#')[0]).strip() for l in ls]
a, b = norm(src_lines(R27.train)), norm(src_lines(R29.train))
removed = [l[2:].strip() for l in difflib.ndiff(a, b) if l.startswith('- ')]
added = [l[2:].strip() for l in difflib.ndiff(a, b) if l.startswith('+ ')]
exp_removed = ['"""r25_co_train.train with the R27 noisy branch (identical clean part, line for line, plus ID writes)."""',
               'loss_1 = criterion(sims_l, labels=None)']
train_ok = (sorted(removed) == sorted(exp_removed) and all(any(t in l for t in ('AL.', "r29_", 'R29', '"""', 'obs_l = torch.as_tensor(l_obs[')) or l in ('if timed:', 'if hasattr(net, \'r29_b\'):') for l in added))
va = norm(src_lines(V.VSE.train)); wb = norm(src_lines(R29.warmup_step))
seq_native = ['self.Eiters += 1', 'img_embs, cap_embs, cap_lens = self.forward_emb(images, captions, lengths, imgae_lengths=image_lengths)',
              'sims = self.forward_sim(img_embs, cap_embs, cap_lens)', 'self.optimizer.zero_grad()',
              'img_embs2, cap_embs2, cap_lens2 = self.forward_emb(images, captions, lengths, imgae_lengths=image_lengths)',
              'sims1 = self.forward_sim(img_embs2, cap_embs2, cap_lens2)', 'img_sims = img_embs.mm(img_embs2.t())', 'cap_sims = cap_embs.mm(cap_embs2.t())',
              'loss = infonce(sims)', 'loss2 = infonce(img_sims)', 'loss3 = infonce(cap_sims)', 'loss = loss + loss2 + loss3', 'loss.backward()',
              'if self.grad_clip > 0:', 'clip_grad_norm_(self.params, self.grad_clip)', 'self.optimizer.step()']
vs = [l.strip() for l in va]
native_in_vse = all(s in vs for s in seq_native)
seq_r29 = [s.replace('self.', 'model.').replace('loss = infonce(sims)', "loss = AL.align(opt, model, 'warmup', sims, obs)") for s in seq_native]
ws = [l.strip() for l in wb if l.strip()]
pos = [ws.index(s) if s in ws else -1 for s in seq_r29]
warm_ok = native_in_vse and min(pos) >= 0 and pos == sorted(pos)
extra = [l for l in ws[pos[0]:pos[-1] + 1] if l not in seq_r29]
warm_ok = warm_ok and all(any(t in l for t in ('diag', 'st[', 'AL.readout', 'for k, v in')) for l in extra)
ok('10_H_unchanged', h_call and train_ok and warm_ok, h_call_identity=h_call, train_removed=removed, train_added=added,
   warmup_order_positions=pos, warmup_extra_lines=extra)

# ---- 11 targets + truth isolation
At = AL.target_from_obs(torch.arange(10))
Ad = AL.target_from_obs(obs)
supp = (Ad > 0)
scan = {}
for f in ('r29_align.py', 'r29_co_train.py', 'r29_run.py'):
    s = open(os.path.join(code, f)).read()
    scan[f] = dict(t2i_truth='_t2i_index' in s, labels_attr=bool(re.search(r'\._labels\b', s)),
                   clean_label_reads=[l.strip() for l in s.splitlines() if 'clean_labels' in l and 'next(' not in l])
truth_ok = all(not v['t2i_truth'] and not v['labels_attr'] and not v['clean_label_reads'] for v in scan.values())
ok('11_targets_and_truth_isolation', torch.equal(At, torch.eye(10)) and float((Ad.sum(1) - 1).abs().max()) < 1e-12 and torch.equal(supp, supp.t())
   and bool(supp.diagonal().all()) and truth_ok, token_scan=scan)

res['all_pass'] = all(v['pass_'] for v in res.values() if isinstance(v, dict))
res['code_sha256'] = {f: hashlib.sha256(open(os.path.join(code, f), 'rb').read()).hexdigest() for f in sorted(os.listdir(code)) if f.endswith('.py')}
json.dump(res, open(out, 'w'), indent=1, default=float)
print('CPU_ALL_PASS' if res['all_pass'] else 'CPU_FAIL')
sys.exit(0 if res['all_pass'] else 1)
