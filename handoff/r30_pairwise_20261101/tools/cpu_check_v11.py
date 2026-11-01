#!/usr/bin/env python3
"""R30 v1.1 CPU checks (float64 / float32, synthetic data; no training data, no GPU).
Usage: cpu_check_v11.py CODE_DIR REFERENCE_DIR PACKAGE_DIR OUT_JSON

 01 package reference objective_checks.run_checks(): 8 groups PASS; file sha256 == package MANIFEST
 02 production pair_loss == reference pair_loss (values and autograd gradients; random and extreme margins)
 03 production hard_pair_rows / two-way main term == reference hard_pair_rows (values, valid flags, gradients wrt S;
    multi-positive ids; both directions = reference on S and on S.t())
 04 PV single-comparison margin gradient never reversed: float64 grid (<= 0, == analytic -8p(1-p)^2[...]), float32 grid
    (any positive derivative only where |analytic| < 1e-6, i.e. saturation); score level: positives <= 0, the selected hard
    negatives >= 0, all other entries exactly 0 (PN/PV/PB)
 05 rules: multi-positive uniform average (manual), duplicate positive candidates, duplicate/tied negatives (first maximal
    index, value == reference, gradient only on that index), transposed direction, candidate permutation equivariance,
    no-negative rows (valid False, zero loss/gradient, finite, counted by the readout)
 06 L / N / B raw terms are r29_align.align(l_listwise_vcs / n_infonce / b_listwise_bce, 'cotrain') (call identity) ==
    reference list_loss(S/tau, a_obs, vcs / nce / bce) summed over rows of both directions (values, gradients)
 07 H term is net.criterion(sims, labels=None) returned unchanged (call identity, same object)
 08 scale: fit_static loss_1 = c x raw (values, gradients); native / 1 return raw unmultiplied (same object); check_term rules
 09 source structure: r30 train() differs from r27 train() only by the expected removed lines and R30-marked added lines;
    r30 warmup_main() == r25 main() up to the end of the warmup loop (resume branch removed) + R30-marked lines
 10 truth isolation: no _t2i_index / ._labels in r30 files, clean_labels only in pass-through unpack lines; co_train.eval_train
    does not use its clean_labels argument (AST)
 11 calibration formula / anomaly rule (r30_calibrate.coefficients)
 12 readout counts (invalid rows, comparisons, ties, top1, margins) == manual; readout selection index == loss selection index
 13 training-state fingerprint: deterministic; every component sensitive to a one-element change; torch.save/load and an AdamW
    state_dict round trip into a fresh optimizer leave every component unchanged
 14 per-epoch records: history annotation raw_epoch / completed / cumulative updates; METRICS rows accepted by the v1.1 package
    tools/report_stage.read_metrics (completed 0 boundary row, completed 10 = raw 9)
 15 fork / resume identity rule (r30_co_train.check_loadable) on synthetic saved states: accepts the registered parent (fork,
    calibration) and the unit's own state (resume); refuses wrong sha, id, source, a branch state as parent, a parent as
    resume source, other unit / objective / scale / coefficient / budget, completed budget, configuration differences
"""
import sys, os, json, inspect, importlib.util, difflib, re, hashlib, ast, math
import numpy as np
import torch
import torch.nn.functional as F

code, refdir, pkgdir, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
sys.path.insert(0, code)
res = {}


def ok(name, cond, **info):
    res[name] = dict(pass_=bool(cond), **info)
    print(name, 'PASS' if cond else 'FAIL', json.dumps(info, default=float)[:1500], flush=True)


# ---- 01 package reference (the module sets the default dtype to float64 at import)
ref_path = os.path.join(refdir, 'objective_checks.py')
spec = importlib.util.spec_from_file_location('objective_checks', ref_path)
REF = importlib.util.module_from_spec(spec); spec.loader.exec_module(REF)
torch.set_default_dtype(torch.float64)
ref_sha = hashlib.sha256(open(ref_path, 'rb').read()).hexdigest()
man = json.load(open(os.path.join(refdir, 'PACKAGE_MANIFEST_SHA256.json')))
rc = REF.run_checks()
ok('01_reference_selfcheck', rc['status'] == 'PASS' and rc['test_groups'] == 8 and ref_sha == man['reference/objective_checks.py'],
   reference_sha256=ref_sha, package_manifest_sha256=man['reference/objective_checks.py'], rerun_checks=list(rc['checks']))

import r29_align as AL
import r30_align as A30
TAU = A30.TAU
rng = np.random.default_rng(30)
KINDS = ('nce', 'vcs', 'bce')


def t64(x, grad=False):
    return torch.tensor(np.asarray(x, dtype=np.float64), requires_grad=grad)


# ---- 02 pair_loss
e2 = {}
m0 = np.concatenate([rng.normal(size=200) * 3, np.linspace(-60, 60, 241)])
for kind in KINDS:
    a, b = t64(m0, True), t64(m0, True)
    va, vb = A30.pair_loss(a, kind), REF.pair_loss(b, kind)
    ga = torch.autograd.grad(va.sum(), a)[0]; gb = torch.autograd.grad(vb.sum(), b)[0]
    e2[kind] = dict(value=float((va - vb).abs().max()), grad=float((ga - gb).abs().max()))
ok('02_pair_loss_vs_reference', all(v < 1e-13 for d in e2.values() for v in d.values()), max_abs=e2)


# ---- 03 hard_pair_rows + two-way
def ref_two_way(S, ids, kind):
    r1, v1 = REF.hard_pair_rows(S, ids, kind, TAU)
    r2, v2 = REF.hard_pair_rows(S.t(), ids, kind, TAU)
    return r1.sum() + r2.sum(), (r1, v1, r2, v2)


e3 = {}
for trial, (B, nid) in enumerate(((16, 16), (16, 6), (128, 128), (128, 40), (9, 3))):
    ids = torch.tensor(rng.integers(nid, size=B)) if nid < B else torch.tensor(rng.permutation(B))
    Snp = rng.normal(size=(B, B)) * 0.2
    for kind in KINDS:
        Sa, Sb = t64(Snp, True), t64(Snp, True)
        r1, v1, i1, p1 = A30.hard_pair_rows(Sa, ids, kind)
        q1, w1 = REF.hard_pair_rows(Sb, ids, kind, TAU)
        r2, v2, _, _ = A30.hard_pair_rows(Sa.t(), ids, kind)
        q2, w2 = REF.hard_pair_rows(Sb.t(), ids, kind, TAU)
        prod = A30.pair_two_way(kind, Sa, ids)
        refv = q1.sum() + q2.sum()
        gp = torch.autograd.grad(prod, Sa)[0]; gr = torch.autograd.grad(refv, Sb)[0]
        e3[f'{trial}_{kind}'] = dict(rows=float(max((r1 - q1).abs().max(), (r2 - q2).abs().max())), total=abs(float(prod - refv)),
                                    grad=float((gp - gr).abs().max()), valid_equal=bool(torch.equal(v1, w1) and torch.equal(v2, w2)),
                                    multipos_rows=int((p1.sum(1) > 1).sum()))
ok('03_hard_pair_vs_reference', all(d['rows'] < 1e-12 and d['total'] < 1e-10 and d['grad'] < 1e-12 and d['valid_equal'] for d in e3.values())
   and any(d['multipos_rows'] > 0 for d in e3.values()), max_abs=e3)

# ---- 04 PV margin monotonicity
mg = torch.linspace(-60, 60, 24001, dtype=torch.float64, requires_grad=True)
g64 = torch.autograd.grad(A30.pair_loss(mg, 'vcs').sum(), mg)[0]
p = mg.detach().sigmoid()
formula = -8 * p * (1 - p) ** 2 * ((1 + 2 * p) ** -3 + (3 - 2 * p) ** -3)
mg32 = mg.detach().float().requires_grad_()
g32 = torch.autograd.grad(A30.pair_loss(mg32, 'vcs').sum(), mg32)[0].double()
pos32 = g32 > 0
sat_ok = bool((formula[pos32].abs() < 1e-6).all()) if bool(pos32.any()) else True
score = {}
for kind in KINDS:
    B = 32
    ids = torch.tensor(rng.integers(12, size=B))
    S = t64(rng.normal(size=(B, B)) * 0.3, True)
    for dname, M in (('i2t', S), ('t2i', S.t())):
        rows, valid, idx, pos = A30.hard_pair_rows(M, ids, kind)
        g = torch.autograd.grad(rows.sum(), M if dname == 'i2t' else S)[0]
        g = g if dname == 'i2t' else g.t()                      # gradient in query-row layout
        sel = torch.zeros_like(pos); sel[torch.arange(B), idx] = True
        sel &= valid[:, None]
        other = ~pos & ~sel
        score[f'{kind}_{dname}'] = dict(max_pos=float(g[pos & valid[:, None]].max()), min_neg=float(g[sel].min()),
                                        max_other_abs=float(g[other].abs().max()) if bool(other.any()) else 0.0)
ok('04_pv_margin_monotone', float(g64.max()) <= 1e-15 and float((formula - g64).abs().max()) < 1e-12 and sat_ok
   and all(v['max_pos'] <= 0 and v['min_neg'] >= 0 and v['max_other_abs'] == 0 for v in score.values()),
   float64_max_derivative=float(g64.max()), formula_err=float((formula - g64).abs().max()),
   float32_positive_count=int(pos32.sum()), float32_positive_max=float(g32.max()), float32_positives_only_where_saturated=sat_ok,
   score_level=score)

# ---- 05 rules
r5 = {}
# multi-positive uniform average (manual)
ids = torch.tensor([0, 0, 1, 2, 2, 2, 3, 4])
Snp = rng.normal(size=(8, 8)) * 0.3
S = t64(Snp)
for kind in KINDS:
    rows, valid, idx, pos = A30.hard_pair_rows(S, ids, kind)
    man_rows = []
    for i in range(8):
        P = [j for j in range(8) if ids[j] == ids[i]]
        neg = max((Snp[i, k], k) for k in range(8) if ids[k] != ids[i])
        man_rows.append(float(np.mean([float(REF.pair_loss(torch.tensor((Snp[i, j] - neg[0]) / TAU), kind)) for j in P])))
    r5[f'multipos_{kind}'] = float(np.abs(rows.numpy() - np.array(man_rows)).max())
# duplicate positive candidates (identical columns sharing the query id) -> both are positives, each compared
Sd = Snp.copy(); Sd[:, 1] = Sd[:, 0]
rows, valid, idx, pos = A30.hard_pair_rows(t64(Sd), ids, 'vcs')
q, w = REF.hard_pair_rows(t64(Sd), ids, 'vcs', TAU)
r5['dup_positive_both_in_P'] = bool(pos[0, 0] and pos[0, 1]) and float((rows - q).abs().max()) < 1e-12
# tied / duplicate negatives: two identical non-positive columns hold the row max
St = rng.normal(size=(8, 8)) * 0.1
St[0, 3] = St[0, 6] = 0.5                        # ids 1 / 3: both non-positive for row 0 (id 0), tie
for kind in KINDS:
    Sa = t64(St, True)
    rows, valid, idx, pos = A30.hard_pair_rows(Sa, ids, kind)
    q, w = REF.hard_pair_rows(t64(St), ids, kind, TAU)
    g = torch.autograd.grad(rows.sum(), Sa)[0]
    r5[f'tie_{kind}'] = dict(first_index=int(idx[0]) == 3, value=float((rows - q).abs().max()), grad_first=float(g[0, 3]), grad_second=float(g[0, 6]))
# transposed direction + permutation equivariance (reference check 7 applied to production)
perm = torch.tensor(rng.permutation(8))
for kind in KINDS:
    rows, valid, _, _ = A30.hard_pair_rows(S.t(), ids, kind)
    q, w = REF.hard_pair_rows(S.t(), ids, kind, TAU)
    rp, vp, _, _ = A30.hard_pair_rows(S[perm][:, perm], ids[perm], kind)
    r5[f'transpose_{kind}'] = float((rows - q).abs().max())
    r5[f'perm_{kind}'] = float((rp - A30.hard_pair_rows(S, ids, kind)[0][perm]).abs().max())
# no-negative rows
Sa = t64(rng.normal(size=(4, 4)), True)
same = torch.zeros(4, dtype=torch.long)
nn_ok = {}
for kind in KINDS:
    rows, valid, _, _ = A30.hard_pair_rows(Sa, same, kind)
    tot = A30.pair_two_way(kind, Sa, same)
    g = torch.autograd.grad(tot, Sa)[0]
    ro = A30.readout_dir(Sa, same)
    nn_ok[kind] = bool((~valid).all() and float(tot) == 0.0 and float(g.abs().max()) == 0.0 and bool(torch.isfinite(g).all())
                       and float(ro[A30.RO_NAMES.index('invalid_rows')]) == 4.0 and float(ro[A30.RO_NAMES.index('comparisons')]) == 0.0)
r5['no_negative'] = nn_ok
ok('05_rules', all(r5[f'multipos_{k}'] < 1e-12 for k in KINDS) and r5['dup_positive_both_in_P']
   and all(r5[f'tie_{k}']['first_index'] and r5[f'tie_{k}']['value'] < 1e-12 and r5[f'tie_{k}']['grad_first'] > 0 and r5[f'tie_{k}']['grad_second'] == 0
           for k in KINDS)
   and all(r5[f'transpose_{k}'] < 1e-12 and r5[f'perm_{k}'] < 1e-12 for k in KINDS) and all(nn_ok.values()), **r5)

# ---- 06 L / N / B raw terms
calls = []
orig_align = AL.align
AL.align = lambda o, net, ph, s, obs: calls.append((o.r29_arm, ph, s, obs)) or ('R29', o.r29_arm)
vals = {obj: A30.raw_main(obj, None, 'S', 'OBS') for obj in ('L', 'N', 'B')}
AL.align = orig_align
call_ok = (vals == {'L': ('R29', 'l_listwise_vcs'), 'N': ('R29', 'n_infonce'), 'B': ('R29', 'b_listwise_bce')}
           and all(c[1:] == ('cotrain', 'S', 'OBS') for c in calls) and A30.AL is AL)
e6 = {}
for obj, kind in (('L', 'vcs'), ('N', 'nce'), ('B', 'bce')):
    for B, nid in ((16, 16), (128, 50)):
        ids = torch.tensor(rng.integers(nid, size=B)) if nid < B else torch.arange(B)
        Snp = rng.normal(size=(B, B)) * 0.2
        Sa, Sb = t64(Snp, True), t64(Snp, True)
        prod = A30.raw_main(obj, None, Sa, ids)
        A = AL.target_from_obs(ids, torch.float64)
        refv = REF.list_loss(Sb / TAU, A, kind).sum() + REF.list_loss(Sb.t() / TAU, A.t(), kind).sum()
        gp = torch.autograd.grad(prod, Sa)[0]; gr = torch.autograd.grad(refv, Sb)[0]
        e6[f'{obj}_{B}'] = dict(value=abs(float(prod - refv)) / max(abs(float(refv)), 1.0), grad=float((gp - gr).abs().max()))
ok('06_L_N_B_terms', call_ok and all(d['value'] < 1e-12 and d['grad'] < 1e-12 for d in e6.values()), call_identity=call_ok, max_abs=e6)


class _Net:
    pass


# ---- 07 H
hc = []
net = _Net(); net.criterion = lambda s, labels='x': hc.append((s, labels)) or hc
o = _Net(); o.r30_objective = 'H'; o.r30_scale = 'native'; o.r30_coef = None
l1, raw = A30.main_term(o, net, 'S', 'OBS')
ok('07_H_native', l1 is hc and raw is hc and hc[0] == ('S', None), calls=len(hc))

# ---- 08 scale rule
e8 = {}
ids = torch.tensor(rng.integers(30, size=64))
Snp = rng.normal(size=(64, 64)) * 0.2
for obj in ('L', 'PN', 'PV', 'PB'):
    o = _Net(); o.r30_objective = obj; o.r30_scale = 'fit_static'; o.r30_coef = 0.37
    Sa, Sb = t64(Snp, True), t64(Snp, True)
    l1, raw = A30.main_term(o, None, Sa, ids)
    rv = A30.raw_main(obj, None, Sb, ids)
    ga = torch.autograd.grad(l1, Sa)[0]; gb = torch.autograd.grad(rv, Sb)[0]
    e8[obj] = dict(value=abs(float(l1 - 0.37 * rv)), grad=float((ga - 0.37 * gb).abs().max()), raw=abs(float(raw - rv)))
same = {}
for obj in ('L', 'N', 'B'):
    o = _Net(); o.r30_objective = obj; o.r30_scale = '1'; o.r30_coef = 1.0
    l1, raw = A30.main_term(o, None, t64(Snp, True), ids)
    same[obj] = l1 is raw
rej = []
for obj, sc, c in (('H', 'native', 1.0), ('L', '1', 2.0), ('N', 'fit_static', 0.5), ('B', 'fit_static', 0.5), ('PV', '1', 1.0),
                   ('PV', 'fit_static', None), ('PV', 'fit_static', 0.0), ('PV', 'fit_static', float('nan')), ('L', 'fit_static', -1.0), ('H', '1', 1.0)):
    try:
        A30.check_term(obj, sc, c); rej.append(False)
    except AssertionError:
        rej.append(True)
acc = []
for obj, sc, c in (('H', 'native', None), ('L', '1', 1.0), ('N', '1', 1.0), ('B', '1', 1.0), ('L', 'fit_static', 0.2), ('PN', 'fit_static', 0.2),
                   ('PV', 'fit_static', 0.2), ('PB', 'fit_static', 0.2)):
    try:
        A30.check_term(obj, sc, c); acc.append(True)
    except AssertionError:
        acc.append(False)
ok('08_scale_rule', all(d['value'] < 1e-12 and d['grad'] < 1e-12 and d['raw'] == 0 for d in e8.values()) and all(same.values()) and all(rej) and all(acc),
   max_abs=e8, unmultiplied=same, rejections=rej, accepted=acc)

# ---- 09 source structure
import r25_co_train as R
import r27_co_train as R27
import r30_co_train as R30


def src(f):
    return inspect.getsource(f).splitlines()


def norm_line(l, strip_mod):
    return re.sub(strip_mod, '', l.split('#')[0]).strip()


a = [norm_line(l, r'\bR27\.') for l in src(R27.train)]
rb = src(R30.train)
b = [norm_line(l, r'\bR27\.') for l in rb]
marked = {norm_line(l, r'\bR27\.') for l in rb if '# R30' in l}
removed = [l[2:] for l in difflib.ndiff(a, b) if l.startswith('- ')]
added = [l[2:] for l in difflib.ndiff(a, b) if l.startswith('+ ')]
exp_removed = ['"""r25_co_train.train with the R27 noisy branch (identical clean part, line for line, plus ID writes)."""',
               'loss_1 = criterion(sims_l, labels=None)', 'clip_grad_norm_(net.params, net.grad_clip)']
doc = '"""r27_co_train.train; the only numerical change is loss_1 (R30 marks)."""'
train_ok = sorted(removed) == sorted(exp_removed) and all(l in marked or l == doc for l in added)
# warmup_main vs R.main up to the end of the warmup loop
ma = src(R.main)
ma_n = [norm_line(l, r'\bR\.') for l in ma]
ma_n = [l for l in ma_n if l]
end = ma_n.index("json.dump(history, open(rec_path, 'w'), indent=1)")
seg = ma_n[1:end + 1]
i0, i1 = seg.index('if opt.resume:'), seg.index('else:')
seg = seg[:i0] + seg[i1 + 1:]
wb = src(R30.warmup_main)
wn = [norm_line(l, r'\bR\.') for l in wb if '# R30' not in l]
wn = [l for l in wn if l][2:]                     # drop def + docstring
warm_ok = wn == seg
ok('09_source_structure', train_ok and warm_ok, train_removed=removed, train_added=added, warmup_equal=warm_ok,
   warmup_diff=[l for l in difflib.ndiff(seg, wn) if l[:1] in '+-'][:20])

# ---- 10 truth isolation
scan = {}
allowed_clean = ('next(', 'R.get_loader(captions_train, images_train, vocab, "warmup"')
for f in ('r30_align.py', 'r30_co_train.py', 'r30_run.py', 'r30_calibrate.py', 'r30_state.py'):
    s = open(os.path.join(code, f)).read()
    scan[f] = dict(t2i_truth='_t2i_index' in s, labels_attr=bool(re.search(r'\._labels\b', s)),
                   clean_label_reads=[l.strip() for l in s.splitlines() if 'clean_labels' in l and not any(t in l for t in allowed_clean)
                                      and not l.strip().startswith(('#', '"', "'")) and 'ignore' not in l],
                   truth_uses=[l.strip() for l in s.splitlines() if '_truth' in l and 'del _truth' not in l and 'R.get_loader' not in l
                               and not l.strip().startswith('#')])
import co_train
tree = ast.parse(inspect.getsource(co_train.eval_train))
fn = tree.body[0]
names_used = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
eval_ok = 'clean_labels' in [x.arg for x in fn.args.args] and 'clean_labels' not in names_used
ok('10_truth_isolation', all(not v['t2i_truth'] and not v['labels_attr'] and not v['clean_label_reads'] and not v['truth_uses'] for v in scan.values())
   and eval_ok, token_scan=scan, eval_train_ignores_clean_labels=eval_ok)

# ---- 11 calibration formula
import r30_calibrate as CAL
s1 = dict(H=4.0, L=1.0, PN=16.0, PV=0.25, PB=9.0)
c1 = CAL.coefficients(s1)
s2 = dict(H=4.0, L=0.0, PN=float('nan'), PV=float('inf'), PB=1.0)
c2 = CAL.coefficients(s2)
s3 = dict(H=0.0, L=1.0, PN=1.0, PV=1.0, PB=1.0)
c3 = CAL.coefficients(s3)
form_ok = (abs(c1['Ls']['c'] - 2.0) < 1e-15 and abs(c1['PN']['c'] - 0.5) < 1e-15 and abs(c1['PV']['c'] - 4.0) < 1e-15
           and abs(c1['PB']['c'] - 2 / 3) < 1e-15 and c1['L0']['c'] == 1.0 and all(c1[k]['status'] == 'OK' for k in A30.CALIBRATED)
           and c2['Ls']['status'] == c2['PN']['status'] == c2['PV']['status'] == 'TECHNICAL_ANOMALY' and c2['PB']['status'] == 'OK'
           and all(c3[k]['status'] == 'TECHNICAL_ANOMALY' and c3[k]['c'] is None for k in A30.CALIBRATED))
ok('11_calibration_formula', form_ok, ok_case={k: c1[k]['c'] for k in c1}, anomaly_case={k: c2[k]['status'] for k in A30.CALIBRATED})

# ---- 12 readout
B = 12
ids = torch.tensor([0, 0, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9])
Snp = rng.normal(size=(B, B)) * 0.2
Snp[2, 7] = Snp[2, 9] = 3.0                    # tie for row 2
S = t64(Snp)
ro = A30.readout_dir(S, ids)
rd = dict(zip(A30.RO_NAMES, ro.tolist()))
pos = (ids[:, None] == ids[None, :]).numpy()
negmax = np.array([max(Snp[i, k] for k in range(B) if not pos[i, k]) for i in range(B)])
m = (Snp - negmax[:, None]) / TAU
_, _, idx_loss, _ = A30.hard_pair_rows(S, ids, 'vcs')
masked = S.masked_fill(torch.tensor(pos), float('-inf'))
man = dict(rows=B, invalid_rows=0, multipos_rows=int((pos.sum(1) > 1).sum()), comparisons=int(pos.sum()),
           margin_sum=float(((Snp - negmax[:, None]) * pos).sum()), pos_beats_hn=int(((m > 0) & pos).sum()),
           tie_rows=1, top1pos_rows=int(pos[np.arange(B), Snp.argmax(1)].sum()))
rd_ok = all(abs(rd[k] - v) < 1e-10 for k, v in man.items()) and torch.equal(masked.argmax(1), idx_loss)
ok('12_readout', rd_ok, readout=rd, manual=man)

# ---- 13 fingerprint
import r30_state as ST
import tempfile, copy as _copy
torch.manual_seed(5)
lin = torch.nn.Linear(6, 4); optim = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=1e-4)
for _ in range(3):
    optim.zero_grad(); lin(torch.randn(8, 6)).pow(2).sum().backward(); optim.step()
def mfs(m, o):
    return {'img_enc': m.state_dict(), 'txt_enc': {'w': torch.arange(5.)}, 'optimizer': o.state_dict(), 'Eiters': 7, 'avg_clean_prob': None}
import random as _random
rngst = {'python': _random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(), 'cuda': []}
st0 = dict(model_A=_copy.deepcopy(mfs(lin, optim)), model_B=_copy.deepcopy(mfs(lin, optim)), rng=rngst, epoch=-1, best_rsum=0, all_loss=[[], []], opt={'x': 1}, history={})
f0 = ST.fingerprint(st0); f0b = ST.fingerprint(_copy.deepcopy(st0))
st1 = _copy.deepcopy(st0); st1['model_A']['img_enc']['weight'][0, 0] += 1e-6
st2 = _copy.deepcopy(st0); st2['model_B']['optimizer']['state'][0]['exp_avg_sq'][0, 0] *= 1.0000001
st3 = _copy.deepcopy(st0); st3['opt'] = {'x': 2}; st3['history'] = {'y': 1}
st4 = _copy.deepcopy(st0); st4['model_A']['Eiters'] = 8
tmp = tempfile.mkdtemp()
torch.save(st0, os.path.join(tmp, 's.pt')); ld = torch.load(os.path.join(tmp, 's.pt'), weights_only=False)
lin2 = torch.nn.Linear(6, 4); opt2 = torch.optim.AdamW(lin2.parameters(), lr=1e-3, weight_decay=1e-4)
lin2.load_state_dict(ld['model_A']['img_enc']); opt2.load_state_dict(ld['model_A']['optimizer'])
st5 = _copy.deepcopy(st0); st5['model_A'] = _copy.deepcopy(mfs(lin2, opt2))
fp_ok = dict(deterministic=f0 == f0b, param=ST.compare(f0, ST.fingerprint(st1))['differing'] == ['model_A.img_enc'],
             adam=ST.compare(f0, ST.fingerprint(st2))['differing'] == ['model_B.optimizer'],
             metadata_excluded=ST.compare(f0, ST.fingerprint(st3))['equal'], counter=ST.compare(f0, ST.fingerprint(st4))['differing'] == ['model_A.Eiters'],
             save_load=ST.compare(f0, ST.fingerprint(ld))['equal'], optimizer_roundtrip=ST.compare(f0, ST.fingerprint(st5))['equal'])
ok('13_fingerprint', all(fp_ok.values()), **fp_ok)

# ---- 14 history annotation + METRICS rows vs the package reader
import r30_co_train as R30m
sys.path.insert(0, os.path.join(pkgdir, 'tools'))
import planlib, report_stage
dev = lambda r: dict(i2t_r1=r, i2t_r5=r + 10, i2t_r10=r + 20, t2i_r1=r, t2i_r5=r + 5, t2i_r10=r + 9, rsum=6 * r + 44)
hist = {'warmup': [{'epoch': e, 'dev': dev(10.0 + e)} for e in range(5)], 'cotrain': []}
for e in range(10):
    hist['cotrain'].append({'epoch': e, 'dev': dev(20.0 + e), 'train': {'A': {'steps': 100 + e}, 'B': {'steps': 90}}})
    R30m.annotate(hist, e)
o = _Net(); o.r30_unit_id = 'r30v11_f60_s1002_HN'; o.noise_ratio = 0.6; o.seed = 1002; o.r30_source_warmup = 'H'; o.r30_objective = 'N'
mrows = R30m.metrics_rows(o, hist)
mp = os.path.join(tmp, 'M.jsonl'); open(mp, 'w').write(''.join(json.dumps(r) + '\n' for r in mrows))
matrix = planlib.load_matrix(__import__('pathlib').Path(pkgdir))
try:
    data = report_stage.read_metrics(__import__('pathlib').Path(mp), matrix); reader_ok = True
except Exception as ex:
    data, reader_ok = {}, repr(ex)
last = hist['cotrain'][-1]
ok('14_epoch_records', reader_ok is True and last['completed_cotrain_epochs'] == 10 and last['raw_epoch'] == 9
   and last['updates_cumulative'] == sum(190 + e for e in range(10)) and mrows[0]['completed_cotrain_epochs'] == 0 and mrows[0]['raw_epoch'] is None
   and len(data) == 11, reader=reader_ok, n_rows=len(mrows), e10=mrows[-1])

# ---- 15 fork / resume identity rule
def opt_like(**kw):
    o = _Net()
    base = {k: 0 for k in R30m.STRICT_KEYS}
    base.update(noise_ratio=0.6, seed=1001, max_iters=0, warmup_epoch=5)
    for k, v in {**base, **kw}.items():
        setattr(o, k, v)
    return o
par_opt = vars(opt_like(r30_stage='boundary', r30_parent_id='wp_f60_s1001_H_e5', r30_source_warmup='H', r29_arm='h_native'))
pst = dict(st0, opt=par_opt, epoch=-1)
pp = os.path.join(tmp, 'parent_state.pt'); torch.save(pst, pp)
psha = R30m.sha256(pp)
fork = dict(r30_mode='fork', r30_parent=pp, r30_parent_sha=psha, r30_parent_id='wp_f60_s1001_H_e5', r30_source_warmup='H')
u_kw = dict(r30_unit_id='r30_f60_s1001_PV', r30_parent_fingerprint='F', r30_objective='PV', r30_scale='fit_static', r30_coef=0.2,
            num_epochs=40, r30_milestone=10)
ust = dict(st0, opt=vars(opt_like(r30_stage='unit', r30_parent_id='wp_f60_s1001_H_e5', r30_parent_sha=psha, r30_source_warmup='H', **u_kw)), epoch=12)
up = os.path.join(tmp, 'state_last.pt'); torch.save(ust, up)
def verdict(path, **kw):
    try:
        return R30m.check_loadable(path, opt_like(**kw))
    except SystemExit:
        return 'REFUSED'
res15 = dict(
    fork_ok=verdict(pp, **fork) == 'fork', calib_ok=verdict(pp, **dict(fork, r30_mode='calibration')) == 'calibration',
    wrong_sha=verdict(pp, **dict(fork, r30_parent_sha='0' * 64)) == 'REFUSED', wrong_id=verdict(pp, **dict(fork, r30_parent_id='wp_f60_s1001_L_e5')) == 'REFUSED',
    wrong_source=verdict(pp, **dict(fork, r30_source_warmup='L')) == 'REFUSED', other_cell=verdict(pp, **dict(fork, seed=1002)) == 'REFUSED',
    smoke_vs_formal=verdict(pp, **dict(fork, max_iters=30)) == 'REFUSED', branch_as_parent=verdict(up, **dict(fork, r30_parent=up, r30_parent_sha=R30m.sha256(up))) == 'REFUSED',
    resume_ok=verdict(up, **dict(fork, r30_mode='resume', **u_kw)) == 'resume',
    parent_as_resume=verdict(pp, **dict(fork, r30_mode='resume', **u_kw)) == 'REFUSED',
    other_unit=verdict(up, **dict(fork, r30_mode='resume', **dict(u_kw, r30_unit_id='r30_f60_s1001_PB'))) == 'REFUSED',
    other_objective=verdict(up, **dict(fork, r30_mode='resume', **dict(u_kw, r30_objective='PB'))) == 'REFUSED',
    other_coef=verdict(up, **dict(fork, r30_mode='resume', **dict(u_kw, r30_coef=0.21))) == 'REFUSED',
    other_budget=verdict(up, **dict(fork, r30_mode='resume', **dict(u_kw, num_epochs=10))) == 'REFUSED',
    other_parent_fp=verdict(up, **dict(fork, r30_mode='resume', **dict(u_kw, r30_parent_fingerprint='G'))) == 'REFUSED',
    config_diff=verdict(up, **dict(fork, r30_mode='resume', margin=0.3, **u_kw)) == 'REFUSED')
ust2 = dict(ust, epoch=39); torch.save(ust2, up)
res15['completed_budget'] = verdict(up, **dict(fork, r30_mode='resume', **u_kw)) == 'REFUSED'
ok('15_fork_resume_identity', all(res15.values()), **res15)

res['all_pass'] = all(v['pass_'] for v in res.values() if isinstance(v, dict))
res['code_sha256'] = {f: hashlib.sha256(open(os.path.join(code, f), 'rb').read()).hexdigest() for f in sorted(os.listdir(code)) if f.endswith('.py')}
res['torch'] = torch.__version__
json.dump(res, open(out, 'w'), indent=1, default=float)
print('CPU_ALL_PASS' if res['all_pass'] else 'CPU_FAIL')
sys.exit(0 if res['all_pass'] else 1)
