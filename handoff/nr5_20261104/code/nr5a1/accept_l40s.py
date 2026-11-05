#!/usr/bin/env python3
"""NR5_A1 short acceptance on a non-RTX GPU type (run on L40S before any NR5_A1 unit there). Uses the RTX-built NR4 caches as-is.
T0  DEV raw path (zero LoRA) vs frozen DEV cache, FIT hidden-cache tail vs frozen FIT cache (first 4096 items): max abs and max
    ratio |d| / (3e-5 + 3e-5|ref|).
TR  production NR5_A1 trainer, 50 steps: A TAIL Flickr60/J, A TAIL CC/J, B COS Flickr00/J, B G_ZERO CC/NCE -> step-0 contract ratios
    (must be <= 1), seconds per step.
S0  NR5_A1 evaluator on those stage-0 checkpoints (A vs NR3 eval_008000; B vs a0 cos_base + b0): score ratio, order differences,
    rule violations.
EQ  COS vs G_ZERO step-0 scores on one Flickr00 batch (bitwise expected on any GPU: same ops).
RP  two-pass replay vs direct backward at the RTX acceptance's trained non-zero LoRA / heads (B COS-J; G_ZERO-NCE).
Writes OUT/acceptance_a1_<gpu>/ACCEPTANCE_NR5_A1.json."""
import json, shutil, time
import numpy as np, torch
from nr1.io import FeatureBank, load_observed, read_json, write_json
from nr2.reference import Marginals, ReferenceSampler, step_loss
from nr4.train import tail_items, tail_pair_scores, HiddenTail
from nr4.tail_encoder import load_siglip, TailEncoder
from nr5a1 import train as T
from nr5a1 import evaluate as EV
from nr5.heads import make_head
from nr5.paths import paths, OUT
H = '/home/infres/yinwang/CS_QMI/handoff/nr5_20261104'
tol = lambda d, r: float((d.double() - r.double()).abs().div(3e-5 + 3e-5*r.double().abs()).max())
mx = lambda a, b: float((a.double() - b.double()).abs().max())


def main():
    gpu = torch.cuda.get_device_name(); tag = gpu.split()[1] if gpu.startswith('NVIDIA') else gpu.replace(' ', '_')
    acc = OUT/f'acceptance_a1_{tag}'; acc.mkdir(parents=True, exist_ok=True); R = dict(gpu=gpu, torch=torch.__version__, cuda=torch.version.cuda); dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    save = lambda: write_json(acc/'ACCEPTANCE_NR5_A1.json', R)
    m, _ = load_siglip(dev, lora=True, seed=1001); T0 = {}
    for c in ('Flickr60', 'CC152K_RECOVERED'):
        P = paths(c); db = FeatureBank(P['dev'], 'dev'); fb = FeatureBank(P['fit'], 'fit')
        de = TailEncoder(m, np.load(P['raw_dev']/'pixels_u8.npy', mmap_mode='r'), np.load(P['raw_dev']/'tokens_i32.npy'), dev)
        with torch.no_grad():
            du = de.encode('image', np.arange(len(db.image_ids)))[0]; dv = de.encode('text', np.arange(len(db.text_ids)))[0]
            ht = HiddenTail(TailEncoder(m, None, None, dev), np.load(P['hidden']/'img_h10_f32.npy', mmap_mode='r'), np.load(P['hidden']/'txt_h10_f32.npy', mmap_mode='r'))
            fu = ht.encode('image', np.arange(4096), False); fv = ht.encode('text', np.arange(4096), False)
        DX, DY = torch.from_numpy(np.asarray(db.x)).to(dev), torch.from_numpy(np.asarray(db.y)).to(dev); FX, FY = torch.from_numpy(np.asarray(fb.x[:4096])).to(dev), torch.from_numpy(np.asarray(fb.y[:4096])).to(dev)
        T0[c] = dict(dev_image=dict(max_abs=mx(du, DX), ratio=tol(du, DX)), dev_text=dict(max_abs=mx(dv, DY), ratio=tol(dv, DY)),
                     fit_image_4096=dict(max_abs=mx(fu, FX), ratio=tol(fu, FX)), fit_text_4096=dict(max_abs=mx(fv, FY), ratio=tol(fv, FY)))
    R['T0_features_vs_RTX_cache'] = T0; save()
    TR, S0 = {}, {}
    for u, steps in (('NR5_A_Flickr60_TAIL_LORA8_J_t2002', 50), ('NR5_A_CC152K_RECOVERED_TAIL_LORA8_J_t2002', 50), ('NR5_B_Flickr00_COS_J_s1001', 50), ('NR5_B_CC152K_RECOVERED_G_ZERO_NCE_s1001', 50)):
        cfg = read_json(f'{H}/configs_production/{u}.json'); c = cfg['condition']; P = paths(c); d = acc/u; shutil.rmtree(d, ignore_errors=True)
        nat = cfg['init'] == 'NATIVE'; t = time.perf_counter()
        T.run(cfg, P['fit'], P['observed'], P['calibration'], None if nat else cfg['prefix'], P['tables'], d, P['hidden'], dev, steps)
        lg = [json.loads(l) for l in open(d/'TRAIN.jsonl')]; s0 = read_json(d/'STEP0_GRAD_CHECK.json')
        TR[u] = dict(tail_backward=read_json(d/'BINDING.json')['config'].get('tail_backward'), peak_note='see TRAIN.jsonl', step0_vs_reference_ratio=s0['step0_vs_reference_max_tol_ratio'], step0_vs_reference_max_abs=s0['step0_scores_vs_prefix_max_abs'],
                     step0_native_ratio=s0['step0_native_vs_a0cos_b0_max_tol_ratio'], seconds_per_step=(lg[-1]['seconds'] - lg[0]['seconds'])/(lg[-1]['step'] - lg[0]['step']), wall=time.perf_counter() - t)
        p0 = 0 if nat else 8000; ck = d/f'step_{p0:06d}.pt'; od = d/'eval_stage0'; shutil.rmtree(od, ignore_errors=True)
        ref = None if nat else f"/projects/EEG-foundation-model/yinghao/outputs_nr3/NR3_20261104/runs/NR3_B_NCE2{cfg['objective']}_{c}_G_s1001/eval_008000/PREDICTIONS.npz"
        try:
            r = EV.evaluate(c, str(ck), P['dev'], P['truth'], od, ref, stage0_base=nat); S0[u] = dict(r['stage0_vs_prefix'], rSum=r['metrics']['rSum'])
        except RuntimeError as e:
            S0[u] = dict(error=str(e), stage0=read_json(od/'PREDICTIONS_SHA256.json')['stage0_vs_prefix'])
        R['TR'], R['S0'] = TR, S0; save()
    # EQ: COS vs G_ZERO step 0
    P = paths('Flickr00'); fb = FeatureBank(P['fit'], 'fit'); im, tx = load_observed(P['observed'], fb); tab = np.load(P['tables']); cal = read_json(P['calibration'])
    smp = ReferenceSampler(Marginals(im, tx, len(fb.image_ids), len(fb.text_ids)), tab['i2t'], tab['t2i'], 1.0, 20261104501, 32, 63); batch = smp.draw()
    for n_, p_ in m.named_parameters():
        if p_.requires_grad and n_.endswith('.B'): p_.data.zero_()
    ht = HiddenTail(TailEncoder(m, None, None, dev), np.load(P['hidden']/'img_h10_f32.npy', mmap_mode='r'), np.load(P['hidden']/'txt_h10_f32.npy', mmap_mode='r'))
    hC = make_head(dict(readout='COS', a0=cal['a0'], b0=cal['b0'])).to(dev); hG = make_head(dict(**fb.dims, architecture='G', a0=cal['a0'], b0=cal['b0'], seed=1001)).to(dev)
    with torch.no_grad():
        ui, ii, ut, it = tail_items(batch); U = ht.encode('image', ui, False); V = ht.encode('text', ut, False)
        sC, sG = tail_pair_scores(hC, U, V, batch, ii, it), tail_pair_scores(hG, U, V, batch, ii, it)
    R['EQ_COS_vs_GZERO_step0_max_abs'] = max(mx(sC[k], sG[k]) for k in sC); save()
    # RP: replay vs direct at the RTX acceptance trained states
    k1 = torch.load(OUT/'acceptance'/'B_COS_J'/'step_000250.pt', map_location='cpu', weights_only=False); kG = torch.load(OUT/'acceptance'/'B_GZERO_NCE'/'step_000250.pt', map_location='cpu', weights_only=False)
    for n_, p_ in m.named_parameters():
        if p_.requires_grad: p_.data.copy_(k1['adapter'][n_].to(dev))
    hC.load_state_dict(k1['model']); hG.load_state_dict(kG['model']); b2 = smp.draw(); RP = []
    for tg, head, obj in (('COS_J', hC, 'J'), ('GZERO_NCE', hG, 'NCE')):
        def grads(mode):
            head.zero_grad(set_to_none=True); m.zero_grad(set_to_none=True); ui, ii, ut, it = tail_items(b2); e = 0.
            if mode == 'direct': U = ht.encode('image', ui, True); V = ht.encode('text', ut, True)
            else:
                with torch.no_grad(): U0 = ht.encode('image', ui, False); V0 = ht.encode('text', ut, False)
                U = U0.clone().requires_grad_(True); V = V0.clone().requires_grad_(True)
            loss = step_loss(tail_pair_scores(head, U, V, b2, ii, it), b2, obj, 'U'); loss.backward()
            if mode == 'replay':
                for kind, rws, gg, r0 in (('image', ui, U.grad, U0), ('text', ut, V.grad, V0)):
                    o = 0
                    for h, n in ht.chunks(kind, rws):
                        z = ht.enc.tail(kind, h, n); e = max(e, mx(z.detach(), r0[o:o+n])); torch.autograd.backward(z, gg[o:o+n]); o += n
            return float(loss.detach()), {n: p.grad.detach().clone() for n, p in head.named_parameters()}, {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.requires_grad}, e
        l1, h1, g1, _ = grads('direct'); l2, h2, g2, e = grads('replay')
        rel = lambda a_, b_: max(float((a_[n]-b_[n]).double().norm() / a_[n].double().norm()) for n in a_ if float(a_[n].abs().max()) > 0)
        RP.append(dict(tag=tg, loss_direct=l1, loss_replay=l2, replay_output_max_abs=e, head_max_rel=rel(h1, h2), lora_max_rel=rel(g1, g2)))
    R['RP'] = RP
    R['all_pass'] = bool(len(TR) == 4 and all(v['step0_vs_reference_ratio'] <= 1 and (v['step0_native_ratio'] is None or v['step0_native_ratio'] <= 1) for v in TR.values())
                         and all(s.get('passed') for s in S0.values()) and R['EQ_COS_vs_GZERO_step0_max_abs'] == 0 and all(x['lora_max_rel'] < 1e-5 and x['head_max_rel'] < 1e-5 for x in RP))
    save(); print(json.dumps(R, indent=1, default=str))


if __name__ == '__main__':
    main()
