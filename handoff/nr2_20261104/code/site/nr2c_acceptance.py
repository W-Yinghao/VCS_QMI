#!/usr/bin/env python3
"""NR2C acceptance on COCO real data (GPU). Fail-fast record of raw values; tolerances fixed before the first run.
 1 global-only extractor == NR1 Flickr cache on the first 64 DEV images / captions (same B=32 batch composition)
 2 COCO marginals support, proposal normalisation, exact log_w vs package reference, IW bound, boundary ties
 3 sampler collisions retained (P slot observed; reference collisions present)
 4 trainer smoke on COCO (U-NCE, HIW-J, HREF-BCE; 20 steps, step-0 contract, resume bitwise)
 5 NR1 evaluator on a COCO smoke step-0 checkpoint: order == base, rSum == base readout
"""
import json, shutil, subprocess, sys
from pathlib import Path
import numpy as np, torch
CODE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(CODE/'src'))
from nr1.io import FeatureBank, load_observed, read_json, sha256, write_json, jsonl
from nr2.proposal_reference import proposal, log_reference_ratio
from nr2.reference import Marginals, ReferenceSampler, log_w
from nr2 import train as nr2_train

O1 = Path('/home/infres/yinwang/CS_QMI/outputs_nr1/NR1_20261104'); OC = Path('/projects/EEG-foundation-model/yinghao/outputs_nr2/NR2C_20261104')
H2 = Path('/home/infres/yinwang/CS_QMI/handoff/nr2_20261104'); SM = OC/'acceptance'; TAB = OC/'proposal/base_top100_fit.npz'
REV = '75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2'
results, fails = {}, []


def rec(name, ok, **kw):
    results[name] = dict(ok=bool(ok), **kw); print(('PASS ' if ok else 'FAIL ') + name, json.dumps(kw, default=float)[:600], flush=True)
    if not ok: fails.append(name)


def main():
    if SM.exists(): shutil.rmtree(SM)
    SM.mkdir(parents=True)
    # 1 extractor cross-check vs NR1 Flickr DEV cache rows 0..63
    for kind in ('images', 'texts'):
        rows = jsonl(O1/f'data/dev_{kind}.jsonl')[:64]
        with open(SM/f'flk_{kind}.jsonl', 'w') as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
    p = subprocess.run([sys.executable, '-B', '-m', 'nr2.extract_global', '--images', str(SM/'flk_images.jsonl'), '--texts', str(SM/'flk_texts.jsonl'),
                        '--split', 'dev', '--out', str(SM/'flk_cache'), '--revision', REV, '--device', 'cuda', '--batch-size', '32', '--local-only'],
                       capture_output=True, text=True)
    if p.returncode: rec('extract_global_vs_nr1_cache', False, stderr=p.stderr[-800:])
    else:
        ref = FeatureBank(O1/'cache/dev', 'dev', verify_hashes=False)
        new = FeatureBank(SM/'flk_cache', 'dev')
        di = float(np.abs(np.asarray(new.x) - np.asarray(ref.x[:64])).max()); dt = float(np.abs(np.asarray(new.y) - np.asarray(ref.y[:64])).max())
        rec('extract_global_vs_nr1_cache', max(di, dt) <= 3e-5, image_global_max_abs=di, text_global_max_abs=dt,
            bitwise=bool(di == 0 and dt == 0), nr1_cache_gpu=ref.manifest['gpu'], this_gpu=torch.cuda.get_device_name(), tol='3e-5 (package default)')
    # 2 proposal contract on COCO tables
    bank = FeatureBank(OC/'cache/fit', 'fit'); tab = np.load(TAB); top = dict(i2t=tab['i2t'], t2i=tab['t2i'])
    sup = {}
    for c in ('COCO00', 'COCO60', 'COCO80'):
        im, tx = load_observed(OC/f'data/observed_{c}.tsv', bank); m = Marginals(im, tx, len(bank.image_ids), len(bank.text_ids))
        sup[c] = dict(zero_p_img=int((m.p_img == 0).sum()), zero_p_txt=int((m.p_txt == 0).sum()), p_img_unique=np.unique(m.p_img).tolist()[:6])
    rec('marginal_support_full_all_conditions', all(v['zero_p_img'] == 0 and v['zero_p_txt'] == 0 for v in sup.values()), **sup)
    im, tx = load_observed(OC/'data/observed_COCO60.tsv', bank); marg = Marginals(im, tx, len(bank.image_ids), len(bank.text_ids))
    pc = dict(i2t=marg.p_txt, t2i=marg.p_img); g = np.random.default_rng(42); w = dict(sum_err=0., min_q=1., max_inv_w=0., lw_err=0.)
    for d in top:
        for x in g.integers(0, top[d].shape[0], 32):
            q = proposal(pc[d], top[d][x], .5); w['sum_err'] = max(w['sum_err'], abs(q.sum()-1)); w['min_q'] = min(w['min_q'], q.min())
            w['max_inv_w'] = max(w['max_inv_w'], (pc[d]/q).max())
            y = g.integers(0, len(pc[d]), 500); y[:50] = top[d][x][:50]
            w['lw_err'] = max(w['lw_err'], np.abs(log_w(np.full(500, x), y, top[d], pc[d], .5) - log_reference_ratio(pc[d], top[d][x], y, .5)).max())
    rec('proposal_normalised_iw_le_2_logw_exact', w['sum_err'] <= 1e-12 and w['min_q'] > 0 and w['max_inv_w'] <= 2+1e-12 and w['lw_err'] <= 1e-13, **w)
    tm = read_json(OC/'proposal/TABLES_MANIFEST.json')
    X = torch.from_numpy(np.array(bank.x, copy=True)).cuda().double(); Y = torch.from_numpy(np.array(bank.y, copy=True)).cuda().double()
    Xn, Yn = torch.nn.functional.normalize(X, dim=1), torch.nn.functional.normalize(Y, dim=1); ties = {}
    for d, Qf, Gf in (('i2t', Xn, Yn), ('t2i', Yn, Xn)):
        diffs = []
        for s in range(0, Qf.shape[0], 2048):
            sc = Qf[s:s+2048] @ Gf.T; v, _ = torch.topk(sc, 101, dim=1)
            for h in (v[:, 99] == v[:, 100]).nonzero().flatten().tolist():
                b = (sc[h] == v[h, 99]).nonzero().flatten(); diffs.append(float((Gf[b] - Gf[b[0]]).abs().max()))
        ties[d] = dict(rows=len(diffs), max_feature_diff_among_tied=max(diffs) if diffs else 0.)
    rec('boundary_ties_identical_content', all(v['max_feature_diff_among_tied'] == 0. for v in ties.values()), **ties, tables=tm['boundary'])
    # 3 collisions retained
    s = ReferenceSampler(marg, top['i2t'], top['t2i'], .5, 1001); col = {'i2t': 0, 't2i': 0}; pos_ok = True
    for _ in range(500):
        b = s.draw()
        for d in b: col[d] += int(b[d]['observed'][:, 1:].sum()); pos_ok &= bool(b[d]['observed'][:, 0].all())
    rec('collisions_retained_and_P_slot_observed', pos_ok and min(col.values()) > 0, reference_collisions_500_steps=col)
    del X, Y, Xn, Yn; torch.cuda.empty_cache()
    # 4 trainer smoke (COCO60)
    cfg0 = read_json(H2/'locked_configs_nr2c/NR2C_HREF_COCO60_G_J_s1001.json'); smoke = {}; ok = True
    args = (OC/'cache/fit', OC/'data/observed_COCO60.tsv', OC/'calibration/COCO60.json', TAB)
    for mode, obj in (('U', 'NCE'), ('HIW', 'J'), ('HREF', 'BCE')):
        cfg = dict(cfg0, reference=mode, objective=obj, alpha=1.0 if mode == 'U' else .5, steps=20, checkpoints=[0, 10, 20], unit_id=f'SMOKE_{mode}_{obj}')
        a, bd = SM/f'{mode}_{obj}_straight', SM/f'{mode}_{obj}_resumed'
        nr2_train.run(cfg, *args, a); nr2_train.run(cfg, *args, bd, max_steps=10); nr2_train.run(cfg, *args, bd)
        ka, kb = torch.load(a/'step_000020.pt', weights_only=False), torch.load(bd/'step_000020.pt', weights_only=False)
        eq = all(torch.equal(ka['model'][k], kb['model'][k]) for k in ka['model'])
        s0 = read_json(a/'STEP0_GRAD_CHECK.json'); log = [json.loads(l) for l in open(a/'TRAIN.jsonl')]
        smoke[f'{mode}_{obj}'] = dict(resume_bitwise=eq, step0_err=s0['step0_score_vs_affine_cosine_max_abs'], finite=all(np.isfinite(r['loss']) for r in log),
                                      hard_frac_ref_i2t=log[0]['i2t_hard_frac_ref'])
        ok &= eq and smoke[f'{mode}_{obj}']['finite']
    rec('trainer_smoke_resume_bitwise_coco', ok, **smoke)
    # 5 evaluator step 0 == base
    evd = SM/'eval_step0'
    p = subprocess.run([sys.executable, '-B', '-m', 'nr1.evaluate', '--cache', str(OC/'cache/dev'), '--truth', str(OC/'evaluator_only/dev_positives.json'),
                        '--checkpoint', str(SM/'HREF_BCE_straight/step_000000.pt'), '--out', str(evd), '--device', 'cuda', '--score-dtype', 'float64'],
                       capture_output=True, text=True)
    mt = read_json(evd/'METRICS.json') if (evd/'METRICS.json').exists() else None; bm = read_json(OC/'baseline/float64/METRICS.json')
    rec('evaluator_step0_order_equals_base', p.returncode == 0 and mt is not None and mt['order_vs_base'] == dict(i2t_queries_order_differs=0, t2i_queries_order_differs=0)
        and abs(mt['metrics']['rSum'] - bm['metrics']['rSum']) < 1e-9, rc=p.returncode, rSum=mt['metrics']['rSum'] if mt else None,
        base_rSum=bm['metrics']['rSum'], stderr=p.stderr[-300:])
    write_json(H2/'checks/NR2C_ACCEPTANCE.json', dict(all_pass=not fails, fails=fails, results=results, gpu=torch.cuda.get_device_name()))
    print('ALL_PASS' if not fails else f'FAIL {fails}'); sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
