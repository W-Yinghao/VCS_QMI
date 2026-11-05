#!/usr/bin/env python3
"""NR5 real-model acceptance (RTX PRO 6000 = hidden-cache GPU type). FIT only (+ DEV features/scores without truth).
EQ  NR5 trainer with the unchanged NR4 configs reproduces NR4: FROZEN Flickr60/J full 4000 steps vs NR3-B; TAIL Flickr60/J 250 steps
    vs the stored NR4 step_008250 (model, adapter, Adam moments, sampler state).
A   NR5-A FROZEN and TAIL (Flickr60/J, seeds 2002/102002/2002) draw identical batches (logged batch digests, 50 steps); TAIL step-0
    scores == folded prefix scores.
B0  Flickr00: fresh COS and G_ZERO at NR1 a0,b0 + zero LoRA: identical step-0 pair scores (bitwise) == a0 cos_base + b0; checkpoint
    source = locked pretrained only (no prefix / adapter state); U sampling: log w == 0, every P slot observed, candidate frequency vs
    observed marginal; NCE mask changes only NCE.
B1  250-step production training (COS-J straight + resume split, G_ZERO-NCE straight): resume bitwise, non-zero LoRA/head update,
    feature change, base digest unchanged.
D   invariance with the trained non-zero LoRA (reorder / single / chunk offset; DEV rename+reorder up to identical-feature groups).
E   two-pass replay vs direct backward for COS and G_ZERO heads (head and LoRA gradients) at the trained state.
Outputs: OUT/acceptance/ACCEPTANCE_NR5.json, REPLAY_EQUIVALENCE_NR5.json, MODULE_AND_GRAD_MANIFEST_ACCEPT_NR5.json."""
import json, shutil, time
from types import SimpleNamespace
import numpy as np, torch
from torch.nn import functional as F
from nr1.io import FeatureBank, load_observed, read_json, write_json
from nr2.reference import Marginals, ReferenceSampler, step_loss
from nr4.train import tail_items, tail_pair_scores, HiddenTail
from nr4.tail_encoder import load_siglip, TailEncoder, base_weight_digest
from nr5 import train as T
from nr5.heads import make_head
from nr5.evaluate import score_and_order
from nr5.paths import paths, OUT
H = '/home/infres/yinwang/CS_QMI/handoff/nr5_20261104'; N4R = '/projects/EEG-foundation-model/yinghao/outputs_nr4/NR4_20261104/runs'; N3R = '/projects/EEG-foundation-model/yinghao/outputs_nr3/NR3_20261104/runs'
mx = lambda a, b: float((a.double() - b.double()).abs().max())
cfgf = lambda u: read_json(f'{H}/configs_production/{u}.json')


def run(cfg, P, d, hidden, steps, prefix=True):
    return T.run(cfg, P['fit'], P['observed'], P['calibration'], cfg['prefix'] if (prefix and cfg.get('init') != 'NATIVE') else None, P['tables'], d, hidden, 'cuda', steps)


def ckcmp(a, b):
    m = max(mx(a['model'][k], b['model'][k]) for k in b['model'])
    ad = max([mx(a['adapter'][k], b['adapter'][k]) for k in b.get('adapter', {})] or [0.0])
    oa, ob = a['optimizer']['state'], b['optimizer']['state']
    op = max([float((oa[k][n].double() - ob[k][n].double()).abs().max()) for k in ob for n in ('exp_avg', 'exp_avg_sq')] or [0.0]) if set(oa) == set(ob) else float('inf')
    return dict(model=m, adapter=ad, adam=op, sampler_equal=json.dumps(a['rng_sampler'], default=str) == json.dumps(b['rng_sampler'], default=str))


def scorr(c, p):
    return None if np.std(p) == 0 or np.std(c) == 0 else float(np.corrcoef(c/c.sum(), p)[0, 1])


def mtv(c, p):
    r = np.random.default_rng(11).multinomial(int(c.sum()), p/p.sum()); return float(.5*np.abs(r/r.sum() - p/p.sum()).sum())


def main():
    acc = OUT/'acceptance'; acc.mkdir(parents=True, exist_ok=True); R = dict(gpu=torch.cuda.get_device_name()); dev = 'cuda'
    prev = read_json(acc/'ACCEPTANCE_NR5.json') if (acc/'ACCEPTANCE_NR5.json').exists() else {}   # resume: completed EQ/A sections are kept
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    save = lambda: write_json(acc/'ACCEPTANCE_NR5.json', R)
    P60, P00 = paths('Flickr60'), paths('Flickr00')
    # ---------- EQ
    if 'EQ_NR4' in prev and 'A' in prev and prev['gpu'] == R['gpu']:
        R['EQ_NR4'], R['A'] = prev['EQ_NR4'], prev['A']; R['EQ_A_from_previous_attempt'] = True
    else:
        eq = {}
        d = acc/'EQ_FROZEN'; shutil.rmtree(d, ignore_errors=True); c = cfgf('EQUIV_NR4_FROZENCHK_Flickr60_G_J_s1001')
        t = time.perf_counter(); T.run(c, P60['fit'], P60['observed'], P60['calibration'], P60['prefix'], P60['tables'], d, None, dev); eq['frozen_seconds'] = time.perf_counter() - t
        eq['FROZEN_vs_NR3B'] = {s: ckcmp(torch.load(d/f'step_{8000+s:06d}.pt', map_location='cpu', weights_only=False), torch.load(f'{N3R}/NR3_B_NCE2J_Flickr60_G_s1001/step_{8000+s:06d}.pt', map_location='cpu', weights_only=False)) for s in (0, 250, 500, 1000, 2000, 4000)}
        la = [json.loads(l)['loss'] for l in open(d/'TRAIN.jsonl')]; lb = [json.loads(l)['loss'] for l in open(f'{N3R}/NR3_B_NCE2J_Flickr60_G_s1001/TRAIN.jsonl')]
        eq['FROZEN_loss_max_abs'] = max(abs(x-y) for x, y in zip(la, lb)); eq['FROZEN_logged'] = [len(la), len(lb)]
        d = acc/'EQ_TAIL'; shutil.rmtree(d, ignore_errors=True); c = cfgf('EQUIV_NR4_TAIL_LORA8_Flickr60_G_J_s1001')
        T.run(c, P60['fit'], P60['observed'], P60['calibration'], P60['prefix'], P60['tables'], d, P60['hidden'], dev, 250)
        eq['TAIL_vs_NR4_step250'] = ckcmp(torch.load(d/'step_008250.pt', map_location='cpu', weights_only=False), torch.load(f'{N4R}/NR4_TAIL_LORA8_Flickr60_G_J_s1001/step_008250.pt', map_location='cpu', weights_only=False))
        la = [json.loads(l)['loss'] for l in open(d/'TRAIN.jsonl')]; lb = [json.loads(l)['loss'] for l in open(f'{N4R}/NR4_TAIL_LORA8_Flickr60_G_J_s1001/TRAIN.jsonl')][:len(la)]
        eq['TAIL_loss_max_abs'] = max(abs(x-y) for x, y in zip(la, lb)); R['EQ_NR4'] = eq; save()
        # ---------- A
        A = {}; hs = {}
        for sc in ('FROZEN', 'TAIL_LORA8'):
            u = f'NR5_A_Flickr60_{sc}_J_t2002'; d = acc/f'A_{sc}'; shutil.rmtree(d, ignore_errors=True); c = cfgf(u)
            run(c, P60, d, P60['hidden'] if sc == 'TAIL_LORA8' else None, 50)
            hs[sc] = [json.loads(l)['batch_sha256'] for l in open(d/'TRAIN.jsonl')]; A[f'{sc}_step0'] = read_json(d/'STEP0_GRAD_CHECK.json')['step0_scores_vs_prefix_max_abs']
        A['batch_digests_equal_FROZEN_TAIL'] = hs['FROZEN'] == hs['TAIL_LORA8']; A['n_digests'] = len(hs['FROZEN'])
        n4h = [json.loads(l).get('batch_sha256') for l in open(acc/'EQ_TAIL'/'TRAIN.jsonl')][:2]
        A['seed2002_batches_differ_from_seed101001'] = hs['FROZEN'][0] != n4h[0]; R['A'] = A; save()
    # ---------- B0
    B0 = {}
    fb = FeatureBank(P00['fit'], 'fit'); im, tx = load_observed(P00['observed'], fb); marg = Marginals(im, tx, len(fb.image_ids), len(fb.text_ids)); tab = np.load(P00['tables'])
    cal = read_json(P00['calibration']); cJ = cfgf('NR5_B_Flickr00_COS_J_s1001'); cG = cfgf('NR5_B_Flickr00_G_ZERO_NCE_s1001')
    smp = ReferenceSampler(marg, tab['i2t'], tab['t2i'], 1.0, int(cJ['train_sampler_seed']), 32, 63); batch = smp.draw()
    X = torch.from_numpy(np.array(fb.x, copy=True)).to(dev); Y = torch.from_numpy(np.array(fb.y, copy=True)).to(dev)
    m, linfo = load_siglip(dev, lora=True, seed=1001); enc = TailEncoder(m, None, None, dev)
    htl = HiddenTail(enc, np.load(P00['hidden']/'img_h10_f32.npy', mmap_mode='r'), np.load(P00['hidden']/'txt_h10_f32.npy', mmap_mode='r'))
    hC = make_head(dict(readout='COS', a0=cal['a0'], b0=cal['b0'])).to(dev); hG = make_head(dict(**fb.dims, architecture='G', a0=cal['a0'], b0=cal['b0'], seed=1001)).to(dev)
    with torch.no_grad():
        ui, inv_i, ut, inv_t = tail_items(batch); U = htl.encode('image', ui, False); V = htl.encode('text', ut, False)
        sC, sG = tail_pair_scores(hC, U, V, batch, inv_i, inv_t), tail_pair_scores(hG, U, V, batch, inv_i, inv_t)
    x0, y0 = F.normalize(X.double(), dim=-1), F.normalize(Y.double(), dim=-1); err = 0.0
    for dd, b in batch.items():
        q, cc = b['query'][:, None].repeat(b['cand'].shape[1], 1), b['cand']
        cb = (x0[q]*y0[cc]).sum(-1) if dd == 'i2t' else (x0[cc]*y0[q]).sum(-1); err = max(err, mx(sC[dd], cal['a0']*cb + cal['b0']))
    lw_zero = all(float(np.abs(b['log_w']).max()) == 0.0 for b in batch.values()); pobs = all(bool(b['observed'][:, 0].all()) for b in batch.values())
    cnt_t = np.zeros(len(fb.text_ids)); cnt_i = np.zeros(len(fb.image_ids)); s2 = ReferenceSampler(marg, tab['i2t'], tab['t2i'], 1.0, 7, 32, 63)
    for _ in range(400):
        bb = s2.draw(); np.add.at(cnt_t, bb['i2t']['cand'][:, 1:].reshape(-1), 1); np.add.at(cnt_i, bb['t2i']['cand'][:, 1:].reshape(-1), 1)
    B0.update(COS_vs_GZERO_step0_max_abs=max(mx(sC[k], sG[k]) for k in sC), step0_vs_a0cos_b0_max_abs=err, log_w_all_zero=lw_zero, P_slot_observed=pobs,
              tail_vs_cache_batch=dict(image=mx(U, X[torch.as_tensor(ui, device=dev)]), text=mx(V, Y[torch.as_tensor(ut, device=dev)])),
              candidate_freq_vs_marginal=dict(draws=int(cnt_t.sum()), text_corr=scorr(cnt_t, marg.p_txt), image_corr=scorr(cnt_i, marg.p_img), text_tv_of_a_multinomial_draw_from_p=mtv(cnt_t, marg.p_txt), image_tv_of_a_multinomial_draw_from_p=mtv(cnt_i, marg.p_img),
                                              text_tv=float(.5*np.abs(cnt_t/cnt_t.sum()-marg.p_txt).sum()), image_tv=float(.5*np.abs(cnt_i/cnt_i.sum()-marg.p_img).sum()),
                                              note='TV of an empirical draw of this size from p itself is O(sqrt(support/draws)); reported, not thresholded'))
    sc0 = {k: v.clone() for k, v in sC.items()}; b2 = {k: dict(v) for k, v in batch.items()}
    for k in b2: b2[k]['observed'] = np.ones_like(b2[k]['observed'])
    B0['mask_effect'] = {o: float(step_loss(sc0, b2, o, 'U') - step_loss(sc0, batch, o, 'U')) for o in ('J', 'BCE', 'NCE')}
    R['B0'] = B0; save()
    # ---------- B1
    B1 = {}
    for tag, c, split in (('COS_J', cJ, True), ('GZERO_NCE', cG, False)):
        d1 = acc/f'B_{tag}'; shutil.rmtree(d1, ignore_errors=True); t = time.perf_counter(); run(c, P00, d1, P00['hidden'], 250); ts = time.perf_counter() - t
        k1 = torch.load(d1/'step_000250.pt', map_location='cpu', weights_only=False); k0 = torch.load(d1/'step_000000.pt', map_location='cpu', weights_only=False)
        r = dict(seconds_250=ts, step0=read_json(d1/'STEP0_GRAD_CHECK.json'), binding_init=k0['binding']['init'], binding_prefix=k0['binding']['prefix_sha256'],
                 step0_head_equals_fresh=all(mx(k0['model'][n], v.cpu()) == 0 for n, v in make_head(k0['binding']['model_config']).state_dict().items()),
                 step0_lora_B_all_zero=all(float(v.abs().max()) == 0 for n, v in k0['adapter'].items() if n.endswith('.B')),
                 head_change=max(mx(k1['model'][n], k0['model'][n]) for n in k1['model']), lora_B_absmax=max(float(v.abs().max()) for n, v in k1['adapter'].items() if n.endswith('.B')),
                 lora_A_change=max(mx(k1['adapter'][n], k0['adapter'][n]) for n in k1['adapter'] if n.endswith('.A')), log_last={k: v for k, v in [json.loads(l) for l in open(d1/'TRAIN.jsonl')][-1].items() if not k.endswith('grad_bins')})
        if split:
            d2 = acc/f'B_{tag}_resume'; shutil.rmtree(d2, ignore_errors=True); run(c, P00, d2, P00['hidden'], 120); run(c, P00, d2, P00['hidden'], 250)
            r['resume_vs_straight'] = ckcmp(torch.load(d2/'step_000250.pt', map_location='cpu', weights_only=False), k1)
        B1[tag] = r
    R['B1'] = B1; save()
    # trained state from COS-J for D/E
    k1 = torch.load(acc/'B_COS_J'/'step_000250.pt', map_location='cpu', weights_only=False); dg0 = base_weight_digest(m)
    for n, p in m.named_parameters():
        if p.requires_grad: p.data.copy_(k1['adapter'][n].to(dev))
    B1['COS_J']['base_digest_unchanged_with_trained_lora'] = dg0 == base_weight_digest(m)
    with torch.no_grad():
        B1['COS_J']['fit_feature_change_first64'] = dict(image=mx(htl.encode('image', np.arange(64), False), X[:64]), text=mx(htl.encode('text', np.arange(64), False), Y[:64]))
    hC.load_state_dict(k1['model'])
    # ---------- D invariance (DEV raw path, trained non-zero LoRA)
    db = FeatureBank(P00['dev'], 'dev'); de = TailEncoder(m, np.load(P00['raw_dev']/'pixels_u8.npy', mmap_mode='r'), np.load(P00['raw_dev']/'tokens_i32.npy'), dev)
    rows, trow = np.arange(320), np.arange(640); rng = np.random.default_rng(7)
    with torch.no_grad():
        bi_, bt_ = de.encode('image', rows)[0], de.encode('text', trow)[0]; perm = rng.permutation(320); permt = rng.permutation(640)
        pi_, pt_ = de.encode('image', rows[perm])[0], de.encode('text', trow[permt])[0]
        sgl = torch.cat([de.encode('image', rows[k:k+1])[0] for k in range(40)]); sgt = torch.cat([de.encode('text', trow[k:k+1])[0] for k in range(40)])
        off = torch.cat([de.encode('image', rows[:7])[0], de.encode('image', rows[7:])[0]]); offt = torch.cat([de.encode('text', trow[:13])[0], de.encode('text', trow[13:])[0]])
        du = de.encode('image', np.arange(len(db.image_ids)))[0]; dv = de.encode('text', np.arange(len(db.text_ids)))[0]
        hE = make_head(k1['binding']['model_config']).to(dev); hE.load_state_dict(k1['model'])
        _, _, _, ni, nt, *_ = score_and_order(SimpleNamespace(x=db.x, y=db.y, image_ids=db.image_ids, text_ids=db.text_ids), du, dv, hE)
        pim, ptx = rng.permutation(len(db.image_ids)), rng.permutation(len(db.text_ids))
        rb = SimpleNamespace(x=np.asarray(db.x)[pim], y=np.asarray(db.y)[ptx], image_ids=[f'r{k}' for k in range(len(pim))], text_ids=[f's{k}' for k in range(len(ptx))])
        _, _, _, ni2, nt2, *_ = score_and_order(rb, du[torch.as_tensor(pim, device=dev)], dv[torch.as_tensor(ptx, device=dev)], hE)
    backi, backt = ptx[ni2][np.argsort(pim)], pim[nt2][np.argsort(ptx)]
    _, gt = np.unique(np.asarray(db.y), axis=0, return_inverse=True); _, gi = np.unique(np.asarray(db.x), axis=0, return_inverse=True); gt, gi = gt.ravel(), gi.ravel()
    R['D_invariance'] = dict(image_reorder=mx(pi_[np.argsort(perm)], bi_), text_reorder=mx(pt_[np.argsort(permt)], bt_), image_single=mx(sgl, bi_[:40]), text_single=mx(sgt, bt_[:40]),
                             image_chunk_offset=mx(off, bi_), text_chunk_offset=mx(offt, bt_), rename_reorder_i2t_equal_up_to_identical_features=bool(np.array_equal(gt[backi], gt[ni])),
                             rename_reorder_t2i_equal_up_to_identical_features=bool(np.array_equal(gi[backt], gi[nt])), i2t_queries_raw_order_differs=int((backi != ni).any(1).sum()),
                             t2i_queries_raw_order_differs=int((backt != nt).any(1).sum()), dev_feature_change_vs_cache=dict(image=mx(du, torch.from_numpy(np.asarray(db.x)).to(dev)), text=mx(dv, torch.from_numpy(np.asarray(db.y)).to(dev))))
    save()
    # ---------- E replay vs direct (COS trained head; G_ZERO trained head with the same LoRA state)
    kG = torch.load(acc/'B_GZERO_NCE'/'step_000250.pt', map_location='cpu', weights_only=False); hG.load_state_dict(kG['model'])
    batch2 = smp.draw(); E = []
    for tag, head, obj in (('COS_J', hC, 'J'), ('GZERO_NCE', hG, 'NCE'), ('GZERO_J', hG, 'J')):
        def grads(mode):
            head.train(); head.zero_grad(set_to_none=True); m.zero_grad(set_to_none=True); ui, inv_i, ut, inv_t = tail_items(batch2); e = 0.0
            if mode == 'direct':
                U = htl.encode('image', ui, True); V = htl.encode('text', ut, True)
            else:
                with torch.no_grad(): U0 = htl.encode('image', ui, False); V0 = htl.encode('text', ut, False)
                U = U0.clone().requires_grad_(True); V = V0.clone().requires_grad_(True)
            loss = step_loss(tail_pair_scores(head, U, V, batch2, inv_i, inv_t), batch2, obj, 'U'); loss.backward()
            if mode == 'replay':
                for kind, rws, gg, r0 in (('image', ui, U.grad, U0), ('text', ut, V.grad, V0)):
                    o = 0
                    for h, n in htl.chunks(kind, rws):
                        z = enc.tail(kind, h, n); e = max(e, mx(z.detach(), r0[o:o+n])); torch.autograd.backward(z, gg[o:o+n]); o += n
            return float(loss.detach()), {n: p.grad.detach().clone() for n, p in head.named_parameters()}, {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.requires_grad}, e
        l1, h1, g1, _ = grads('direct'); l2, h2, g2, e = grads('replay')
        rel = lambda a_, b_: max(float((a_[n]-b_[n]).double().norm() / max(a_[n].double().norm(), 1e-300)) for n in a_ if float(a_[n].abs().max()) > 0)
        E.append(dict(tag=tag, loss_direct=l1, loss_replay=l2, replay_output_max_abs=e, head_max_rel=rel(h1, h2), lora_max_rel=rel(g1, g2),
                      lora_A_grad_absmax=max(float(v.abs().max()) for n, v in g1.items() if n.endswith('.A')), head_grad_names=sorted(h1)))
    write_json(acc/'REPLAY_EQUIVALENCE_NR5.json', dict(gpu=R['gpu'], comparisons=E)); R['E_replay'] = E
    R['manifest'] = dict(lora_targets=linfo['target_paths'], lora_trainable=sum(p.numel() for p in m.parameters() if p.requires_grad),
                         COS_head_params=sum(p.numel() for p in hC.parameters()), GZERO_head_params=sum(p.numel() for p in hG.parameters()),
                         B_COS_trainable=98304 + sum(p.numel() for p in hC.parameters()), B_GZERO_trainable=98304 + sum(p.numel() for p in hG.parameters()))
    write_json(acc/'MODULE_AND_GRAD_MANIFEST_ACCEPT_NR5.json', R['manifest']); save(); print(json.dumps(R, indent=1, default=str)[:8000])


if __name__ == '__main__':
    main()
