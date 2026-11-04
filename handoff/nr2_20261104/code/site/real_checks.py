#!/usr/bin/env python3
"""NR2 real small-batch checks on the actual FIT cache / observed tables / base proposal tables (GPU job).
Every check records raw values; any failure -> exit 1. Tolerances are fixed here before the first run."""
import json, shutil, subprocess, sys, hashlib
from pathlib import Path
import numpy as np, torch
CODE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(CODE/'src'))
from nr1.io import FeatureBank, load_observed, read_json, sha256, write_json
from nr1.models import DirectScorer
from nr2.proposal_reference import proposal, log_reference_ratio, sampled_rows
from nr2.reference import Marginals, ReferenceSampler, log_w, step_loss, pair_scores, build_base_topk
from nr2 import train as nr2_train

O1 = Path('/home/infres/yinwang/CS_QMI/outputs_nr1/NR1_20261104'); O2 = Path('/home/infres/yinwang/CS_QMI/outputs_nr2/NR2_20261104')
H1 = Path('/home/infres/yinwang/CS_QMI/handoff/nr1_20261104'); H2 = Path('/home/infres/yinwang/CS_QMI/handoff/nr2_20261104')
TAB = O2/'proposal/base_top100_fit.npz'; SM = O2/'checks_smoke'
results, fails = {}, []


def rec(name, ok, **kw):
    kw = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in kw.items()}
    results[name] = dict(ok=bool(ok), **kw); print(('PASS ' if ok else 'FAIL ') + name, json.dumps(kw)[:500], flush=True)
    if not ok: fails.append(name)


def nonzero_head(model, seed=5):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.startswith('global_residual.4'): p.copy_(0.3*torch.randn(p.shape, generator=g).to(p))
    return model


def main():
    bank = FeatureBank(O1/'cache/fit', 'fit'); tab = np.load(TAB); top = dict(i2t=tab['i2t'], t2i=tab['t2i'])
    im, tx = load_observed(O1/'data/observed_Flickr60.tsv', bank); cal = read_json(O1/'calibration/Flickr60.json')
    marg = Marginals(im, tx, len(bank.image_ids), len(bank.text_ids))
    X = torch.from_numpy(np.array(bank.x, copy=True)).cuda(); Y = torch.from_numpy(np.array(bank.y, copy=True)).cuda()
    pc = dict(i2t=marg.p_txt, t2i=marg.p_img); g = np.random.default_rng(42)
    rec('marginal_support_full', (marg.p_img > 0).all() and (marg.p_txt > 0).all(),
        p_img_unique=np.unique(marg.p_img).tolist(), p_txt_unique=np.unique(marg.p_txt).tolist())
    # 1 normalisation, positivity, IW bound, production log_w == package log_reference_ratio
    worst = dict(sum_err=0., min_q=1., max_inv_w=0., lw_err=0.)
    for d in top:
        for x in g.integers(0, top[d].shape[0], 64):
            q = proposal(pc[d], top[d][x], .5)
            worst['sum_err'] = max(worst['sum_err'], abs(q.sum()-1)); worst['min_q'] = min(worst['min_q'], q.min())
            worst['max_inv_w'] = max(worst['max_inv_w'], (pc[d]/q).max())
            y = g.integers(0, len(pc[d]), 500); y[:50] = top[d][x][:50]
            worst['lw_err'] = max(worst['lw_err'], np.abs(log_w(np.full(500, x), y, top[d], pc[d], .5) - log_reference_ratio(pc[d], top[d][x], y, .5)).max())
    rec('proposal_normalised_positive_iw_le_2_logw_exact', worst['sum_err'] <= 1e-12 and worst['min_q'] > 0 and worst['max_inv_w'] <= 2+1e-12 and worst['lw_err'] <= 1e-13, **worst)
    # 2 sampling frequency vs exact mixture probability (3 fixed queries per direction, 504k draws each)
    freq = {}
    for d in top:
        for x in (0, 1, top[d].shape[0]-1):
            s = ReferenceSampler(marg, top['i2t'], top['t2i'], .5, 900+x, n_query=8000, n_ref=63)
            ys = s._draw_ref(d, np.full(8000, x)).reshape(-1); n = len(ys)
            q = proposal(pc[d], top[d][x], .5); hard = top[d][x]
            cnt = np.bincount(ys, minlength=len(q))
            mass = q[hard].sum(); f_hard = cnt[hard].sum()/n
            z_mass = (f_hard-mass)/np.sqrt(mass*(1-mass)/n)
            z_item = (cnt[hard]-n*q[hard])/np.sqrt(n*q[hard]*(1-q[hard]))
            nonhard = np.setdiff1d(np.arange(len(q)), hard); m_out = q[nonhard].sum(); f_out = cnt[nonhard].sum()/n
            freq[f'{d}_{x}'] = dict(n=n, exact_hard_mass=mass, emp_hard_mass=f_hard, z_mass=z_mass, max_abs_z_item=np.abs(z_item).max(),
                                    exact_outside_mass=m_out, emp_outside_mass=f_out)
    ok = all(abs(v['z_mass']) < 5 and v['max_abs_z_item'] < 5 for v in freq.values())
    rec('sampling_frequency_vs_exact_mixture', ok, **{k: {a: float(b) for a, b in v.items()} for k, v in freq.items()})
    # 3 P/Q collisions retained; HIW weight bound over a long run; coupling; positives identical across modes
    sH = ReferenceSampler(marg, top['i2t'], top['t2i'], .5, 1001); sU = ReferenceSampler(marg, top['i2t'], top['t2i'], 1.0, 1001)
    col = {d: [0, 0., 0, 0] for d in top}; max_inv_w = 0.; pos_same = True; same_pos_slot = {d: 0 for d in top}
    partners = {}; img_of_txt = np.empty(len(bank.text_ids), np.int64); img_of_txt[tx] = im
    for i in range(len(im)): partners.setdefault(int(im[i]), []).append(int(tx[i]))
    for _ in range(2000):
        bh, bu = sH.draw(), sU.draw()
        for d in top:
            pos_same &= np.array_equal(bh[d]['cand'][:, 0], bu[d]['cand'][:, 0]) and np.array_equal(bh[d]['query'], bu[d]['query'])
            col[d][0] += int(bh[d]['observed'][:, 1:].sum()); col[d][2] += bh[d]['observed'][:, 1:].size
            same_pos_slot[d] += int((bh[d]['cand'][:, 1:] == bh[d]['cand'][:, :1]).sum())
            max_inv_w = max(max_inv_w, float(np.exp(-bh[d]['log_w'][:, 1:]).max()))
            for r, x in enumerate(bh[d]['query']):   # exact expected collision probability of one reference draw
                if d == 'i2t':
                    ys = np.array(partners[int(x)]); col[d][1] += 63*(.5*pc[d][ys].sum() + .5*np.isin(ys, top[d][x]).sum()/100)
                else:
                    ys = np.array([img_of_txt[x]]); col[d][1] += 63*(.5*pc[d][ys].sum() + .5*np.isin(ys, top[d][x]).sum()/100)
    cz = {d: (c[0]-c[1])/np.sqrt(c[1]) for d, c in col.items()}
    rec('collisions_retained_rate_matches_exact', all(abs(z) < 5 for z in cz.values()) and all(c[0] > 0 for c in col.values()),
        observed=dict((d, c[0]) for d, c in col.items()), expected=dict((d, c[1]) for d, c in col.items()), z=cz,
        reference_slot_equals_own_positive=same_pos_slot)
    rec('hiw_inverse_weight_le_2', max_inv_w <= 2+1e-12, max_inverse_weight=max_inv_w)
    rec('positive_draws_identical_U_vs_HIW_HREF_same_seed', pos_same)
    # 4 HREF corrects positive AND reference slots; HIW weights only Q; on a real batch with a nonzero head (float64)
    m = nonzero_head(DirectScorer(**bank.dims, architecture='G', a0=cal['a0'], b0=cal['b0'], seed=1001)).cuda().double()
    Xd, Yd = X.double(), Y.double()
    b = ReferenceSampler(marg, top['i2t'], top['t2i'], .5, 77).draw()
    with torch.no_grad(): sc = pair_scores(m, Xd, Yd, b)
    errs = {}
    for obj in ('J', 'BCE', 'NCE'):
        man = 0.
        for d in top:
            lw = torch.as_tensor(b[d]['log_w'], device='cuda'); s = sc[d]
            msk = torch.as_tensor(b[d]['observed'], device='cuda') if obj == 'NCE' else None
            man += .5*float(sampled_rows(s-lw, torch.zeros_like(lw), obj, 'U', msk).mean())
        errs[f'HREF_{obj}'] = abs(float(step_loss(sc, b, obj, 'HREF'))-man)
    lw0 = np.concatenate([b[d]['log_w'][:, 0] for d in top]); pos_hard = np.concatenate([b[d]['hard'][:, 0] for d in top])
    rec('href_corrects_positive_and_reference', max(errs.values()) <= 1e-12 and np.allclose(lw0[~pos_hard], np.log(.5)) and (lw0 != 0).all(),
        loss_err=errs, positive_slot_log_w_values=sorted(set(np.round(lw0, 6).tolist())), positives_in_top100=int(pos_hard.sum()), n_pos=len(lw0))
    b2 = {d: dict(v, log_w=v['log_w'].copy()) for d, v in b.items()}
    for d in b2: b2[d]['log_w'][:, 0] = -9.
    rec('hiw_positive_term_unweighted', all(float(step_loss(sc, b, o, 'HIW')) == float(step_loss(sc, b2, o, 'HIW')) for o in ('J', 'BCE')))
    # 5 HIW unbiasedness on real pairs (nonzero head): IW proposal estimate of the product-Q reference term vs direct
    nQ = 400000; gg = np.random.default_rng(9)
    with torch.no_grad():
        f = lambda ii, tt: torch.cat([m.pairs(Xd[torch.as_tensor(ii[s:s+50000], device='cuda')], Yd[torch.as_tensor(tt[s:s+50000], device='cuda')]) for s in range(0, len(ii), 50000)]).cpu().numpy()
        sq = f(im[gg.integers(len(im), size=nQ)], tx[gg.integers(len(im), size=nQ)])
        ref = {}
        for d in top:
            r = gg.integers(len(im), size=nQ); x = im[r] if d == 'i2t' else tx[r]
            sp = ReferenceSampler(marg, top['i2t'], top['t2i'], .5, 31 if d == 'i2t' else 32, n_query=nQ, n_ref=1)
            y = sp._draw_ref(d, x)[:, 0]; lw = log_w(x, y, top[d], pc[d], .5)
            sv = f(x, y) if d == 'i2t' else f(y, x); t = np.tanh(sv/2); v = np.exp(-lw)*(t+t*t/2)
            ref[d] = (v.mean(), v.std()/np.sqrt(nQ))
    t0 = np.tanh(sq/2); v0 = t0+t0*t0/2; direct = (v0.mean(), v0.std()/np.sqrt(nQ))
    zz = {d: (ref[d][0]-direct[0])/np.hypot(ref[d][1], direct[1]) for d in ref}
    rec('hiw_reference_term_unbiased_real', all(abs(z) < 5 for z in zz.values()), direct=direct, iw=ref, z=zz)
    # 6 invariance with nonzero head (float64): reference-slot permutation, query-row permutation, chunked scoring
    base = {(o, md): float(step_loss(sc, b, o, md)) for o, md in (('J', 'HREF'), ('BCE', 'HIW'), ('NCE', 'HREF'), ('J', 'U'))}
    pr = np.random.default_rng(3); bp = {}
    for d in b:
        rp = pr.permutation(32); sp_ = np.concatenate([[0], 1+pr.permutation(63)])
        bp[d] = {k: (v[rp][:, sp_] if v.ndim == 2 else v[rp]) for k, v in b[d].items()}
    with torch.no_grad(): scp = pair_scores(m, Xd, Yd, bp)
    perm_err = max(abs(float(step_loss(scp, bp, o, md))-v) for (o, md), v in base.items())
    with torch.no_grad():
        chunked = {}
        for d in b:
            q = torch.as_tensor(b[d]['query'], device='cuda'); c = torch.as_tensor(b[d]['cand'], device='cuda')
            qq, cc = q[:, None].expand_as(c).reshape(-1), c.reshape(-1); parts = []
            for s in range(0, len(qq), 97):
                parts.append(m.pairs(Xd[qq[s:s+97]], Yd[cc[s:s+97]]) if d == 'i2t' else m.pairs(Xd[cc[s:s+97]], Yd[qq[s:s+97]]))
            chunked[d] = torch.cat(parts).reshape(c.shape)
    chunk_err = max(float((chunked[d]-sc[d]).abs().max()) for d in b)
    rec('loss_invariant_slot_row_permutation_and_chunking', perm_err <= 1e-12 and chunk_err <= 1e-12, perm_err=perm_err, chunk_err=chunk_err)
    # 7 ID rename + storage reorder of a FIT subset: tables, marginals and sampled loss identical (nonzero head, float64)
    sub_img = np.arange(2000); keep = np.isin(im, sub_img); si, st = im[keep], tx[keep]
    txt_rows = np.unique(st); tmap = {t: i for i, t in enumerate(txt_rows)}; st2 = np.array([tmap[t] for t in st])
    xf, yf = np.array(bank.x[sub_img]), np.array(bank.y[txt_rows]); iids = [bank.image_ids[i] for i in sub_img]; tids = [bank.text_ids[t] for t in txt_rows]
    def build(xf, yf, iids, tids, si, st2):
        ti, _ = build_base_topk(xf, yf, tids, 100); tt, _ = build_base_topk(yf, xf, iids, 100)
        return Marginals(si, st2, len(iids), len(tids)), ti, tt
    mA, tiA, ttA = build(xf, yf, iids, tids, si, st2)
    pi_, pt_ = np.random.default_rng(4).permutation(len(iids)), np.random.default_rng(5).permutation(len(tids))
    inv_i, inv_t = np.argsort(pi_), np.argsort(pt_)
    mB, tiB, ttB = build(xf[pi_], yf[pt_], [f'IMG{k}' for k in pi_], [f'TXT{k}' for k in pt_], inv_i[si], inv_t[st2])
    tab_same = np.array_equal(pt_[tiB[inv_i]], tiA) and np.array_equal(pi_[ttB[inv_t]], ttA)
    XA, YA = torch.as_tensor(xf, device='cuda').double(), torch.as_tensor(yf, device='cuda').double()
    XB, YB = XA[torch.as_tensor(pi_, device='cuda')], YA[torch.as_tensor(pt_, device='cuda')]
    lerr = 0.
    for alpha, mode, obj in ((1.0, 'U', 'NCE'), (.5, 'HIW', 'J'), (.5, 'HREF', 'BCE'), (.5, 'HREF', 'NCE')):
        sa, sb = ReferenceSampler(mA, tiA, ttA, alpha, 8), ReferenceSampler(mB, tiB, ttB, alpha, 8)
        for _ in range(3):
            ba, bb = sa.draw(), sb.draw()
            with torch.no_grad():
                la = float(step_loss(pair_scores(m, XA, YA, ba), ba, obj, mode)); lb = float(step_loss(pair_scores(m, XB, YB, bb), bb, obj, mode))
            lerr = max(lerr, abs(la-lb))
    rec('id_rename_storage_reorder_invariance_subset', tab_same and lerr <= 1e-12, tables_equal_by_identity=tab_same, max_loss_diff=lerr,
        note='record order fixed; storage rows permuted and IDs renamed (ID-hash tiebreak keys change)')
    # 8 proposal tables truth-free + chunk invariance on real cache; boundary ties are identical-content items
    qi = np.random.default_rng(6).integers(0, len(bank.image_ids), 512)
    re_i, _ = build_base_topk(np.array(bank.x[qi]), bank.y, bank.text_ids, 100, chunk=37)
    Xn = torch.nn.functional.normalize(X.double(), dim=1); Yn = torch.nn.functional.normalize(Y.double(), dim=1)
    src = (CODE/'site/build_tables.py').read_text()
    rec('tables_truth_free_and_chunk_invariant', np.array_equal(re_i, top['i2t'][qi]) and 'load_observed' not in src and 'evaluator_only' not in src,
        rebuilt_rows=512, tables_sha256=sha256(TAB))
    ties = {}
    for d, Qf, Gf in (('i2t', Xn, Yn), ('t2i', Yn, Xn)):
        rows = []
        for s in range(0, Qf.shape[0], 4096):
            sc_ = Qf[s:s+4096] @ Gf.T; v, ix = torch.topk(sc_, 101, dim=1)
            hit = (v[:, 99] == v[:, 100]).nonzero().flatten()
            for h in hit.tolist():
                r = s+h; boundary = (sc_[h] == v[h, 99]).nonzero().flatten()
                feats = Gf[boundary]; rows.append(float((feats - feats[0]).abs().max()))
        ties[d] = dict(rows=len(rows), max_feature_diff_among_tied=max(rows) if rows else 0.)
    rec('boundary_ties_are_identical_content', all(v['max_feature_diff_among_tied'] == 0. for v in ties.values()), **ties)
    # 9 trainer smoke on real data: all 8 registered (mode, objective) arms, step-0 contract, resume bit-equality
    if SM.exists(): shutil.rmtree(SM)
    cfg0 = read_json(H2/'locked_configs/NR2_HREF_Flickr60_G_J_s1001.json')
    res_ok, smoke = True, {}
    for mode, objs in (('U', ('J', 'BCE', 'NCE')), ('HIW', ('J', 'BCE')), ('HREF', ('J', 'BCE', 'NCE'))):
        for obj in objs:
            cfg = dict(cfg0, reference=mode, objective=obj, alpha=1.0 if mode == 'U' else .5, steps=20, checkpoints=[0, 10, 20],
                       unit_id=f'SMOKE_{mode}_{obj}')
            a, bdir = SM/f'{mode}_{obj}_straight', SM/f'{mode}_{obj}_resumed'
            args = (O1/'cache/fit', O1/'data/observed_Flickr60.tsv', O1/'calibration/Flickr60.json', TAB)
            nr2_train.run(cfg, *args, a); nr2_train.run(cfg, *args, bdir, max_steps=10); nr2_train.run(cfg, *args, bdir)
            ka = torch.load(a/'step_000020.pt', weights_only=False); kb = torch.load(bdir/'step_000020.pt', weights_only=False)
            eq = all(torch.equal(ka['model'][k], kb['model'][k]) for k in ka['model'])
            moved = any(not torch.equal(ka['model'][k], torch.load(a/'step_000000.pt', weights_only=False)['model'][k]) for k in ka['model'])
            log = [json.loads(l) for l in open(a/'TRAIN.jsonl')]
            fin = all(np.isfinite(r['loss']) for r in log); s0 = read_json(a/'STEP0_GRAD_CHECK.json')
            smoke[f'{mode}_{obj}'] = dict(resume_bitwise=eq, params_moved=moved, finite=fin, step0_err=s0['step0_score_vs_affine_cosine_max_abs'],
                                          hard_frac_ref_i2t=log[0]['i2t_hard_frac_ref'], max_inv_w=log[0]['i2t_max_inv_w_ref'])
            res_ok &= eq and moved and fin
    rec('trainer_smoke_8_arms_resume_bitwise', res_ok, **smoke)
    # 10 evaluator = NR1 evaluator (s only), step-0 order == base on a NR2 checkpoint, separate process
    ev_same = sha256(CODE/'src/nr1/evaluate.py') == sha256(H1/'code/src/nr1/evaluate.py') and 'nr2' not in (CODE/'src/nr1/evaluate.py').read_text()
    evd = SM/'eval_step0_HREF_J'
    p = subprocess.run([sys.executable, '-B', '-m', 'nr1.evaluate', '--cache', str(O1/'cache/dev'), '--truth', str(O1/'evaluator_only/dev_positives.json'),
                        '--checkpoint', str(SM/'HREF_J_straight/step_000000.pt'), '--out', str(evd), '--device', 'cuda', '--score-dtype', 'float64'],
                       capture_output=True, text=True)
    mt = read_json(evd/'METRICS.json') if (evd/'METRICS.json').exists() else None
    basem = read_json(O1/'baseline/float64/METRICS.json')
    rec('evaluator_identical_to_NR1_s_only_step0_equals_base', ev_same and p.returncode == 0 and mt is not None and
        mt['order_vs_base'] == dict(i2t_queries_order_differs=0, t2i_queries_order_differs=0) and abs(mt['metrics']['rSum']-basem['metrics']['rSum']) < 1e-9,
        rc=p.returncode, rSum=mt['metrics']['rSum'] if mt else None, base_rSum=basem['metrics']['rSum'], stderr=p.stderr[-300:])
    # 11 CONFIRM == NR1 seed-1001 recipe except seed/unit_id; trainer sources byte-identical to NR1
    cdiff = {}
    for f in sorted((H2/'locked_configs').glob('NR2_CONFIRM_*.json')):
        c = read_json(f); n = read_json(H1/f"locked_configs/NR1_{c['condition']}_G_{c['objective']}_s1001.json")
        cdiff[f.stem] = sorted(k for k in set(c) | set(n) if c.get(k) != n.get(k))
    src_same = all(sha256(p_) == sha256(H1/'code/src/nr1'/p_.name) for p_ in (CODE/'src/nr1').glob('*.py')) and \
        sha256(CODE/'site/train_unit_nr1.py') == sha256(H1/'code/site/train_unit.py')
    rec('confirm_configs_and_trainer_equal_NR1', len(cdiff) == 12 and all(v == ['seed', 'unit_id'] for v in cdiff.values()) and src_same,
        n=len(cdiff), diffs=sorted(set(map(tuple, cdiff.values()))), nr1_sources_byte_identical=src_same)
    write_json(H2/'checks/REAL_CHECKS.json', dict(all_pass=not fails, fails=fails, results=results, gpu=torch.cuda.get_device_name()))
    print('ALL_PASS' if not fails else f'FAIL {fails}')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
