#!/usr/bin/env python3
"""NR4 real two-tower acceptance (GPU, RTX PRO 6000 = extraction GPU type). Condition/objective default Flickr60 / J.
A module/parameter manifest; B t0 equivalence; C real training (250 steps) + resume equality, non-zero update and feature change;
D ID-rename / reorder / chunk invariance with a non-zero adapter; E two-pass replay vs direct backward gradients (head and LoRA);
F timing. FIT only (+ DEV features/scores without truth). Writes ACCEPTANCE_NR4.json, MODULE_AND_GRAD_MANIFEST_ACCEPT.json,
REPLAY_EQUIVALENCE.json under OUT/acceptance."""
import argparse, json, shutil, time
from types import SimpleNamespace
import numpy as np, torch
from torch.nn import functional as F
from nr1.io import FeatureBank, load_observed, read_json, write_json, sha256
from nr1.models import DirectScorer
from nr2.reference import Marginals, ReferenceSampler, pair_scores, step_loss
from nr4 import train as T
from nr4.paths import paths, OUT
from nr4.tail_encoder import load_siglip, TailEncoder, base_weight_digest, _pad
from nr4.evaluate import score_and_order
H = '/home/infres/yinwang/CS_QMI/handoff/nr4_20261104'
mx = lambda a, b: float((a.double() - b.double()).abs().max())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cond', default='Flickr60'); ap.add_argument('--obj', default='J'); a = ap.parse_args()
    acc = OUT/'acceptance'/f'{a.cond}_{a.obj}'; acc.mkdir(parents=True, exist_ok=True); R = dict(cond=a.cond, obj=a.obj, gpu=torch.cuda.get_device_name())
    cfgT = read_json(f'{H}/configs_production/NR4_TAIL_LORA8_{a.cond}_G_{a.obj}_s1001.json'); cfgP = read_json(f'{H}/configs_production/NR4_POST32_{a.cond}_G_{a.obj}_s1001.json')
    P = paths(a.cond, cfgT['reference_unit']); dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    # ---------- A manifest
    m, linfo = load_siglip(dev, lora=True, seed=1001)
    cfgm = m.config
    req = [(n, list(p.shape), p.numel()) for n, p in m.named_parameters() if p.requires_grad]
    mods = {n: type(md).__name__ for n, md in m.named_modules() if n in linfo['target_paths']}
    post = T.Post(768, 32, 1001)
    pre = torch.load(P['prefix'], map_location='cpu', weights_only=False); G = DirectScorer(**pre['binding']['model_config'])
    R['A_manifest'] = dict(vision_hidden=cfgm.vision_config.hidden_size, vision_layers=cfgm.vision_config.num_hidden_layers, text_hidden=cfgm.text_config.hidden_size,
                           text_layers=cfgm.text_config.num_hidden_layers, lora_targets=linfo['target_paths'], lora_module_types=mods,
                           lora_trainable=req, lora_trainable_count=sum(r[2] for r in req), siglip_total_params=sum(p.numel() for p in m.parameters()),
                           siglip_frozen_params=sum(p.numel() for p in m.parameters() if not p.requires_grad),
                           post_trainable={n: list(p.shape) for n, p in post.named_parameters()}, post_trainable_count=sum(p.numel() for p in post.parameters()),
                           head_count=sum(p.numel() for p in G.parameters()), head_plus_adapter=sum(p.numel() for p in G.parameters()) + 98304)
    assert R['A_manifest']['lora_trainable_count'] == 98304 and R['A_manifest']['post_trainable_count'] == 98304
    # ---------- B t0 equivalence
    bank = FeatureBank(P['fit'], 'fit'); dbank = FeatureBank(P['dev'], 'dev')
    rd = P['raw_dev']; enc = TailEncoder(m, np.load(rd/'pixels_u8.npy', mmap_mode='r'), np.load(rd/'tokens_i32.npy'), dev)
    du = enc.encode('image', np.arange(len(dbank.image_ids)))[0]; dv = enc.encode('text', np.arange(len(dbank.text_ids)))[0]
    X = torch.from_numpy(np.array(bank.x, copy=True)).to(dev); Y = torch.from_numpy(np.array(bank.y, copy=True)).to(dev)
    DX = torch.from_numpy(np.array(dbank.x, copy=True)).to(dev); DY = torch.from_numpy(np.array(dbank.y, copy=True)).to(dev)
    hm = read_json(P['hidden']/'HIDDEN_MANIFEST.json')
    fe = TailEncoder(m, np.load(P['hidden']/'pixels_u8.npy', mmap_mode='r'), np.load(P['hidden']/'tokens_i32.npy'), dev)
    fu = fe.encode('image', np.arange(64))[0]; fv = fe.encode('text', np.arange(64))[0]
    hi, ht = np.load(P['hidden']/'img_h10_f32.npy', mmap_mode='r'), np.load(P['hidden']/'txt_h10_f32.npy', mmap_mode='r'); ht_ = T.HiddenTail(enc, hi, ht)
    im, tx = load_observed(P['observed'], bank); marg = Marginals(im, tx, len(bank.image_ids), len(bank.text_ids)); tab = np.load(P['tables'])
    sampler = ReferenceSampler(marg, tab['i2t'], tab['t2i'], .5, 101001, 32, 63); batch = sampler.draw()
    G = G.to(dev); G.load_state_dict(pre['model']); G.eval()
    with torch.no_grad():
        ref = pair_scores(G, X, Y, batch); ui, inv_i, ut, inv_t = T.tail_items(batch)
        U = ht_.encode('image', ui, False); V = ht_.encode('text', ut, False); st = T.tail_pair_scores(G, U, V, batch, inv_i, inv_t)
        sp = T.post_pair_scores(G, post.to(dev), X, Y, batch)
    R['B_t0'] = dict(dev_raw_vs_dev_cache=dict(image=mx(du, DX), text=mx(dv, DY)), fit_raw_vs_fit_cache_first64=dict(image=mx(fu, X[:64]), text=mx(fv, Y[:64])),
                     fit_hidden_check=hm['verification_first256'], fit_hidden_status=hm['status'],
                     batch_items_hidden_tail_vs_cache=dict(image=mx(U, X[torch.as_tensor(ui, device=dev)]), text=mx(V, Y[torch.as_tensor(ut, device=dev)]), n_unique_images=len(ui), n_unique_texts=len(ut)),
                     batch_scores_tail_vs_prefix=max(mx(st[d], ref[d]) for d in ref), batch_scores_post_vs_prefix=max(mx(sp[d], ref[d]) for d in ref))
    write_json(acc/'ACCEPTANCE_NR4.json', R); del du, dv, fu, fv
    # ---------- E (t0 part) replay vs direct at B = 0 (A grads expected 0), then repeated at the trained state
    def grads(mode, batch):
        G.train(); G.zero_grad(set_to_none=True); m.zero_grad(set_to_none=True)
        ui, inv_i, ut, inv_t = T.tail_items(batch)
        if mode == 'direct':
            U = ht_.encode('image', ui, True); V = ht_.encode('text', ut, True)
        else:
            with torch.no_grad(): U0 = ht_.encode('image', ui, False); V0 = ht_.encode('text', ut, False)
            U = U0.clone().requires_grad_(True); V = V0.clone().requires_grad_(True)
        sc = T.tail_pair_scores(G, U, V, batch, inv_i, inv_t); loss = step_loss(sc, batch, a.obj, 'HREF'); loss.backward(); err = 0.0
        if mode == 'replay':
            for kind, rows, g, r0 in (('image', ui, U.grad, U0), ('text', ut, V.grad, V0)):
                o = 0
                for h, n in ht_.chunks(kind, rows):
                    z = enc.tail(kind, h, n); err = max(err, mx(z.detach(), r0[o:o+n])); torch.autograd.backward(z, g[o:o+n]); o += n
        gh = {n: p.grad.detach().clone() for n, p in G.named_parameters()}; gl = {n: p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for n, p in m.named_parameters() if p.requires_grad}
        return float(loss), gh, gl, err

    def compare(tag, batch):
        torch.cuda.reset_peak_memory_stats(); t = time.perf_counter(); l1, h1, g1, _ = grads('direct', batch); torch.cuda.synchronize(); td = time.perf_counter() - t; pd = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats(); t = time.perf_counter(); l2, h2, g2, err = grads('replay', batch); torch.cuda.synchronize(); tr = time.perf_counter() - t; pr = torch.cuda.max_memory_allocated()
        def d(a_, b_):
            return {n: dict(max_abs=mx(a_[n], b_[n]), ref_max_abs=float(a_[n].abs().max()), rel_to_norm=float((a_[n] - b_[n]).double().norm() / max(a_[n].double().norm(), 1e-300))) for n in a_}
        hh, ll = d(h1, h2), d(g1, g2)
        return dict(tag=tag, loss_direct=l1, loss_replay=l2, replay_output_max_abs=err, head=hh, lora=ll,
                    head_max_rel=max(v['rel_to_norm'] for v in hh.values()), lora_max_rel=max(v['rel_to_norm'] for k, v in ll.items() if v['ref_max_abs'] > 0) if any(v['ref_max_abs'] > 0 for v in ll.values()) else None,
                    lora_A_grad_absmax=max(v['ref_max_abs'] for k, v in ll.items() if k.endswith('.A')), lora_B_grad_absmax=max(v['ref_max_abs'] for k, v in ll.items() if k.endswith('.B')),
                    direct_seconds=td, replay_seconds=tr, direct_peak_bytes=pd, replay_peak_bytes=pr)
    E = [compare('t0_B_zero', batch)]
    # ---------- C real training through the production trainer (250 steps; resume split)
    C = {}
    for tag, cfg, hid in (('POST32', cfgP, None), ('TAIL_LORA8', cfgT, P['hidden'])):
        d1, d2 = acc/f'{tag}_straight', acc/f'{tag}_resume'
        for d_ in (d1, d2):
            if d_.exists(): shutil.rmtree(d_)
        t = time.perf_counter(); T.run(cfg, P['fit'], P['observed'], P['calibration'], P['prefix'], P['tables'], d1, hid, dev, 250); ts = time.perf_counter() - t
        T.run(cfg, P['fit'], P['observed'], P['calibration'], P['prefix'], P['tables'], d2, hid, dev, 120)
        T.run(cfg, P['fit'], P['observed'], P['calibration'], P['prefix'], P['tables'], d2, hid, dev, 250)
        k1 = torch.load(d1/'step_008250.pt', map_location='cpu', weights_only=False); k2 = torch.load(d2/'step_008250.pt', map_location='cpu', weights_only=False)
        k0 = torch.load(d1/'step_008000.pt', map_location='cpu', weights_only=False)
        logs = [json.loads(l) for l in open(d1/'TRAIN.jsonl')]
        C[tag] = dict(seconds_250_steps=ts, seconds_per_step_logged=logs[-1]['seconds'] / 250, step0=read_json(d1/'STEP0_GRAD_CHECK.json'),
                      resume_model_max_abs=max(mx(k1['model'][n], k2['model'][n]) for n in k1['model']), resume_adapter_max_abs=max(mx(k1['adapter'][n], k2['adapter'][n]) for n in k1['adapter']),
                      head_change_max_abs=max(mx(k1['model'][n], k0['model'][n]) for n in k1['model']),
                      adapter_B_absmax=max(float(v.abs().max()) for n, v in k1['adapter'].items() if n.endswith('B') or n.endswith('.B.weight')),
                      adapter_A_change_max_abs=max(mx(k1['adapter'][n], k0['adapter'][n]) for n in k1['adapter'] if n.endswith('A') or n.endswith('.A.weight')),
                      adapter_B_change_max_abs=max(mx(k1['adapter'][n], k0['adapter'][n]) for n in k1['adapter'] if n.endswith('B') or n.endswith('.B.weight')),
                      log_last=logs[-1])
        if tag == 'TAIL_LORA8':
            dg0 = base_weight_digest(m)
            for n, p in m.named_parameters():
                if p.requires_grad: p.data.copy_(k1['adapter'][n].to(dev))
            C[tag]['base_weight_digest_unchanged_after_loading_trained_lora'] = dg0 == base_weight_digest(m)
            with torch.no_grad():
                f1 = fe.encode('image', np.arange(64))[0]; f2 = fe.encode('text', np.arange(64))[0]
            C[tag]['raw_feature_change_first64_vs_cache'] = dict(image=mx(f1, X[:64]), text=mx(f2, Y[:64]))
            C[tag]['raw_vs_hidden_path_trained_first64'] = dict(image=mx(f1, ht_.encode('image', np.arange(64), False)), text=mx(f2, ht_.encode('text', np.arange(64), False)))
            Gt = DirectScorer(**pre['binding']['model_config']).to(dev); Gt.load_state_dict(k1['model'])
        else:
            pt = T.Post(768, 32, 1001).to(dev); pt.load_state_dict(k1['adapter'])
            with torch.no_grad(): C[tag]['feature_change_first64'] = dict(image=mx(pt.img(X[:64]), X[:64]), text=mx(pt.txt(Y[:64]), Y[:64]))
    R['C_training'] = C; write_json(acc/'ACCEPTANCE_NR4.json', R)
    # m now holds the trained (non-zero) TAIL adapter; G <- trained head for E/D
    G.load_state_dict(Gt.state_dict())
    batch2 = sampler.draw(); E.append(compare('trained_step250', batch2))
    write_json(acc/'REPLAY_EQUIVALENCE.json', dict(cond=a.cond, objective=a.obj, gpu=R['gpu'], comparisons=E,
               definition='direct = one backward through all retained fixed-B=32 chunk graphs; replay = no-grad encode, loss.backward to leaf outputs, re-encode each chunk with the same weights and backpropagate its output-gradient slice; both before any clip/step'))
    # ---------- D invariance with the trained non-zero adapter
    rows = np.arange(min(320, len(dbank.image_ids))); trow = np.arange(min(640, len(dbank.text_ids))); rng = np.random.default_rng(7)
    with torch.no_grad():
        base_i = enc.encode('image', rows)[0]; base_t = enc.encode('text', trow)[0]
        perm = rng.permutation(len(rows)); pi = enc.encode('image', rows[perm])[0]; inv = np.argsort(perm)
        permt = rng.permutation(len(trow)); ptt = enc.encode('text', trow[permt])[0]; invt = np.argsort(permt)
        single = torch.cat([enc.encode('image', rows[k:k+1])[0] for k in range(0, 40)]); single_t = torch.cat([enc.encode('text', trow[k:k+1])[0] for k in range(0, 40)])
        off = torch.cat([enc.encode('image', rows[:7])[0], enc.encode('image', rows[7:])[0]]); off_t = torch.cat([enc.encode('text', trow[:13])[0], enc.encode('text', trow[13:])[0]])
        du = enc.encode('image', np.arange(len(dbank.image_ids)))[0]; dv = enc.encode('text', np.arange(len(dbank.text_ids)))[0]
        gb = SimpleNamespace(x=dbank.x, y=dbank.y, image_ids=dbank.image_ids, text_ids=dbank.text_ids)
        _, _, _, ni, nt, pairs, z, _, _ = score_and_order(gb, du, dv, G)
        pim, ptx = rng.permutation(len(dbank.image_ids)), rng.permutation(len(dbank.text_ids))
        ren_i = [f'renamed_{k}_{x}' for k, x in enumerate(np.array(dbank.image_ids)[pim])]; ren_t = [f'renamed_{k}_{x}' for k, x in enumerate(np.array(dbank.text_ids)[ptx])]
        rb = SimpleNamespace(x=np.asarray(dbank.x)[pim], y=np.asarray(dbank.y)[ptx], image_ids=ren_i, text_ids=ren_t)
        _, _, _, ni2, nt2, pairs2, z2, _, _ = score_and_order(rb, du[torch.as_tensor(pim, device=dev)], dv[torch.as_tensor(ptx, device=dev)], G)
        back_i = ptx[ni2][np.argsort(pim)]; back_t = pim[nt2][np.argsort(ptx)]
        ties = int(len(z) - len(np.unique(z)))
    R['D_invariance'] = dict(image_reorder=mx(pi[inv], base_i), text_reorder=mx(ptt[invt], base_t), image_single_vs_chunk=mx(single, base_i[:40]), text_single_vs_chunk=mx(single_t, base_t[:40]),
                             image_chunk_offset=mx(off, base_i), text_chunk_offset=mx(off_t, base_t),
                             eval_rename_reorder_i2t_order_equal=bool(np.array_equal(back_i, ni)), eval_rename_reorder_t2i_order_equal=bool(np.array_equal(back_t, nt)),
                             i2t_queries_differing=int((back_i != ni).any(1).sum()), t2i_queries_differing=int((back_t != nt).any(1).sum()), exact_duplicate_pair_scores=ties,
                             dev_feature_change_vs_cache=dict(image=mx(du, DX), text=mx(dv, DY)))
    R['F_timing'] = dict(tail_s_per_step=C['TAIL_LORA8']['seconds_per_step_logged'], post_s_per_step=C['POST32']['seconds_per_step_logged'])
    write_json(acc/'ACCEPTANCE_NR4.json', R)
    man = dict(A=R['A_manifest'], C={k: {kk: vv for kk, vv in v.items() if kk not in ('log_last',)} for k, v in C.items()})
    write_json(acc/'MODULE_AND_GRAD_MANIFEST_ACCEPT.json', man); print(json.dumps(R, indent=1, default=str)[:6000])


if __name__ == '__main__':
    main()
