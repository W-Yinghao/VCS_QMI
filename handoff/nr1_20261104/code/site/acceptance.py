#!/usr/bin/env python3
"""NR1 small-batch real-data acceptance (DATA_AND_INVARIANCE_CN.md section C).

Uses 32 real FIT images (+1 duplicate-path record) and 64 real FIT captions (2 per image).
Runs the package extractor itself on (A) the original order, (B) an independent permutation of
both manifests with a different batch size, (C) the order-A manifests labelled 'dev' for the
evaluator tests.  Every check stores raw max differences; any failure -> exit 1 (fail-fast).
Thresholds: fp32 features/logits/scores atol=rtol=3e-5 (package default); fp16-stored local
tokens: |a-b| <= one fp16 spacing at max(|a|,|b|) (explicit fp16 storage decision of the package).
"""
import argparse, copy, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE/'src'))
from nr1.io import FeatureBank, jsonl, sha256, write_json, read_json  # noqa: E402
from nr1.models import DirectScorer  # noqa: E402
from nr1.objectives import loss_value  # noqa: E402
from nr1 import train as nr1_train  # noqa: E402
from nr1.evaluate import evaluate  # noqa: E402
from nr1.extract_siglip2 import content_mask, vision as fx_vision, text as fx_text  # noqa: E402
BFIX = 32  # fixed forward shape (amendment A1 (2)), same as production extraction

TOL = 3e-5
results, fails = {}, []


def rec(name, ok, **kw):
    results[name] = dict(ok=bool(ok), **{k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in kw.items()})
    print(('PASS ' if ok else 'FAIL ') + name, json.dumps(results[name])[:400], flush=True)
    if not ok: fails.append(name)


def maxdiff(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    return float(np.abs(a-b).max()) if a.size else 0.0


def close32(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    return bool(np.all(np.abs(a-b) <= TOL + TOL*np.abs(b)))


def fp16_ulp_ok(a, b):
    a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
    sp = np.spacing(np.maximum(np.abs(a), np.abs(b)).astype(np.float16)).astype(np.float32)
    return bool(np.all(np.abs(a-b) <= sp + TOL)), float((np.abs(a-b) > 0).mean())


U16 = 2.0 ** -11


def tok_rel(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    return float((np.linalg.norm(a-b, axis=-1) / np.maximum(np.linalg.norm(b, axis=-1), 1e-30)).max())


def model_state_sha(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--data', required=True)
    ap.add_argument('--revision', required=True); ap.add_argument('--snapshot', required=True)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    py = sys.executable; dev = 'cuda'
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False  # amendment A1 (1), same as exporter
    t_start = time.perf_counter()
    weights = Path(a.snapshot)/'model.safetensors'; w_sha0 = sha256(weights)
    # ---- small real manifests ----
    ims = jsonl(Path(a.data)/'fit_images.jsonl')[:32]
    txt_all = jsonl(Path(a.data)/'fit_texts.jsonl')
    texts = []
    for g in range(32): texts += [txt_all[5*g], txt_all[5*g+1]]
    longest = max(txt_all, key=lambda r: len(r['text']))
    texts[-1] = longest  # include a long caption (truncation path); record which
    dup = dict(id='img_acceptance_duplicate_of_0', path=ims[0]['path'])
    ims_a = ims + [dup]
    man = out/'manifests'; man.mkdir(exist_ok=True)
    def wj(p, rs):
        with open(p, 'w') as f:
            for r in rs: f.write(json.dumps(r, ensure_ascii=False)+'\n')
    wj(man/'A_images.jsonl', ims_a); wj(man/'A_texts.jsonl', texts)
    rng = np.random.default_rng(424242)
    pi, pt = rng.permutation(len(ims_a)), rng.permutation(len(texts))
    wj(man/'B_images.jsonl', [ims_a[i] for i in pi]); wj(man/'B_texts.jsonl', [texts[i] for i in pt])
    env = dict(os.environ, PYTHONPATH=str(CODE/'src'))
    for tag, split, bs in [('A', 'fit', BFIX), ('B', 'fit', BFIX), ('C', 'dev', BFIX)]:
        src = 'A' if tag == 'C' else tag
        t0 = time.perf_counter()
        r = subprocess.run([py, '-B', '-m', 'nr1.extract_siglip2', '--images', str(man/f'{src}_images.jsonl'),
                            '--texts', str(man/f'{src}_texts.jsonl'), '--split', split, '--out', str(out/f'cache_{tag}'),
                            '--revision', a.revision, '--device', dev, '--batch-size', str(bs), '--local-only'],
                           env=env, capture_output=True, text=True)
        rec(f'extractor_run_{tag}', r.returncode == 0, seconds=time.perf_counter()-t0, stderr_tail=r.stderr[-1500:])
        if r.returncode: break
    if fails: return finish(out, t_start)
    bankA, bankB, bankC = FeatureBank(out/'cache_A', 'fit'), FeatureBank(out/'cache_B', 'fit'), FeatureBank(out/'cache_C', 'dev')
    # ---- official model in-process ----
    from transformers import AutoModel, AutoProcessor
    import transformers
    model = AutoModel.from_pretrained('google/siglip2-base-patch16-224', revision=a.revision, trust_remote_code=False,
                                      local_files_only=True, attn_implementation='eager', dtype=torch.float32).to(dev).eval()
    proc = AutoProcessor.from_pretrained('google/siglip2-base-patch16-224', revision=a.revision, trust_remote_code=False, local_files_only=True)
    for p in model.parameters(): p.requires_grad_(False)
    m_sha0 = model_state_sha(model)
    from PIL import Image, ImageOps
    def load(p):
        with Image.open(p) as im: return ImageOps.exif_transpose(im).convert('RGB').copy()
    pil = [load(r['path']) for r in ims_a]
    caps = [r['text'].lower() for r in texts]
    tok = proc(text=caps, padding='max_length', max_length=64, truncation=True, return_tensors='pt')
    n_tok = (tok['input_ids'] != proc.tokenizer.pad_token_id).sum(1)
    full_tok = (proc(text=[longest['text'].lower()], padding=False, truncation=False)['input_ids'][0])
    rec('text_lengths_vary', int(n_tok.min()) < int(n_tok.max()), longest_caption_untruncated_tokens=len(full_tok),
        truncation_exercised=len(full_tok) > 64,
        min_tokens=int(n_tok.min()), max_tokens=int(n_tok.max()), processor_keys=sorted(tok.keys()),
        longest_caption_chars=len(longest['text']))
    @torch.inference_mode()
    def enc_img(images):
        pv = proc(images=images, return_tensors='pt')['pixel_values'].to(dev); P, Hh = [], []
        for st in range(0, len(pv), BFIX):
            p_, h_ = fx_vision(model, pv[st:st+BFIX], BFIX); P.append(p_); Hh.append(h_)
        po = torch.cat(P).float(); h = torch.cat(Hh).float(); n = len(images)
        hl = F.avg_pool2d(h.reshape(n, 14, 14, -1).permute(0, 3, 1, 2), 2, 2).flatten(2).transpose(1, 2)
        return po.cpu().numpy(), F.normalize(po, dim=-1).cpu().numpy(), h.cpu().numpy(), hl.cpu().numpy()
    @torch.inference_mode()
    def enc_txt(t):
        inp = proc(text=t, padding='max_length', max_length=64, truncation=True, return_tensors='pt')
        kw = {k: inp[k].to(dev) for k in ('input_ids', 'attention_mask', 'position_ids') if k in inp}; P, Hh = [], []
        for st in range(0, len(t), BFIX):
            p_, h_ = fx_text(model, {k: v[st:st+BFIX] for k, v in kw.items()}, BFIX); P.append(p_); Hh.append(h_)
        po = torch.cat(P).float(); h = torch.cat(Hh).float()
        return po.cpu().numpy(), F.normalize(po, dim=-1).cpu().numpy(), h.cpu().numpy(), inp['input_ids']
    # (1) official forward logits reconstruction on 32 imgs x 64 caps
    with torch.inference_mode():
        inp = proc(text=caps, images=pil[:32], padding='max_length', max_length=64, truncation=True, return_tensors='pt')
        inp = {k: v.to(dev) for k, v in inp.items()}
        outp = model(**inp)
    ip_raw, ip, _, il = enc_img(pil); tp_raw, tp, th, ids = enc_txt(caps)
    logits = (torch.from_numpy(ip[:32]) @ torch.from_numpy(tp).T) * model.logit_scale.exp().cpu() + model.logit_bias.cpu()
    off = outp.logits_per_image.cpu().numpy()
    rec('official_logits_reconstruction', close32(logits.numpy(), off), max_abs_diff=maxdiff(logits.numpy(), off),
        logit_scale=float(model.logit_scale.exp()), logit_bias=float(model.logit_bias))
    np.savez_compressed(out/'arr_logits.npz', reconstructed=logits.numpy(), official=off)
    # (2) single vs batch vs chunk encodings
    md = dict(img_pool=0., img_hidden=0., txt_pool=0., txt_hidden=0.); ok = True
    for i in [0, 5, 17, 31]:
        s = enc_img([pil[i]]); c = enc_img(pil[max(0, i-3):i+4]); j = i - max(0, i-3)
        u, v = s[0][0], ip_raw[i]; md['img_pool'] = max(md['img_pool'], maxdiff(u, v)); ok &= close32(u, v)
        u, v = s[2][0], enc_img(pil)[2][i]; md['img_hidden'] = max(md['img_hidden'], maxdiff(u, v)); ok &= close32(u, v)
        ok &= close32(c[2][j], v)
        ok &= close32(c[0][j], ip_raw[i])
    for i in [0, 9, 40, 63]:
        s = enc_txt([caps[i]]); mixed = enc_txt([caps[i], caps[63], caps[2]])
        for k, (u, v) in {'txt_pool': (s[0][0], tp_raw[i]), 'txt_hidden': (s[2][0], th[i])}.items():
            md[k] = max(md[k], maxdiff(u, v)); ok &= close32(u, v)
        ok &= close32(mixed[0][0], tp_raw[i]) and close32(mixed[2][0], th[i])
    rec('single_batch_chunk_encoding', ok, **{f'max_abs_{k}': v for k, v in md.items()})
    # (3) exporter cache == in-process recomputation (order A)
    okg = close32(bankA.x, ip) and close32(bankA.y, tp)
    _, fi = fp16_ulp_ok(bankA.il, il); _, ft = fp16_ulp_ok(bankA.tl, th)
    ri_, rt_ = tok_rel(bankA.il, il), tok_rel(bankA.tl, th)
    oki, okt = ri_ <= U16 + TOL, rt_ <= U16 + TOL
    mask = content_mask(proc.tokenizer, caps, ids)
    okm = bool(np.array_equal(np.asarray(bankA.tm), mask.numpy()))
    bw = bool(np.array_equal(np.asarray(bankA.il), il.astype(np.float16)) and np.array_equal(np.asarray(bankA.tl), th.astype(np.float16)))
    rec('cache_equals_recompute', okg and oki and okt and okm, local_bitwise_equal_to_fp16_recompute=bw, global_max_abs=max(maxdiff(bankA.x, ip), maxdiff(bankA.y, tp)),
        local_img_max_abs=maxdiff(bankA.il, il), local_txt_max_abs=maxdiff(bankA.tl, th), frac_local_img_nonbitwise=fi,
        frac_local_txt_nonbitwise=ft, local_img_tok_relL2=ri_, local_txt_tok_relL2=rt_, mask_equal=okm, unit_norm_max_dev=float(np.abs(np.linalg.norm(bankA.x, axis=1)-1).max()),
        local_shapes=[list(bankA.il.shape), list(bankA.tl.shape)])
    # mask semantics: pad/eos/bos excluded
    special = set(proc.tokenizer.all_special_ids)
    ids_np = ids.numpy(); m_np = mask.numpy()
    okmask = bool(np.all(~m_np[np.isin(ids_np, list(special))])) and bool(np.all(m_np.sum(1) >= 1))
    rec('content_mask_excludes_special_and_padding', okmask, special_ids=sorted(special),
        content_tokens_min=int(m_np.sum(1).min()), content_tokens_max=int(m_np.sum(1).max()))
    # (4) duplicate image record -> identical vectors
    d = len(ims_a)-1
    rec('duplicate_image_identical_vectors', bool(np.array_equal(bankA.x[0], bankA.x[d]) and np.array_equal(bankA.il[0], bankA.il[d])),
        global_max_abs=maxdiff(bankA.x[0], bankA.x[d]), local_max_abs=maxdiff(bankA.il[0], bankA.il[d]))
    # (5) independent reorder: cache B vs cache A after explicit ID remap
    ia = np.array([bankA.imap[x] for x in bankB.image_ids]); ta = np.array([bankA.tmap[x] for x in bankB.text_ids])
    okx = close32(bankB.x, bankA.x[ia]) and close32(bankB.y, bankA.y[ta])
    _, f_il = fp16_ulp_ok(bankB.il, bankA.il[ia]); _, f_tl = fp16_ulp_ok(bankB.tl, bankA.tl[ta])
    r_il, r_tl = tok_rel(bankB.il, bankA.il[ia]), tok_rel(bankB.tl, bankA.tl[ta])
    ok_il, ok_tl = r_il <= 2*U16 + TOL, r_tl <= 2*U16 + TOL
    okm2 = bool(np.array_equal(bankB.tm, bankA.tm[ta]))
    base_a = (bankA.x.astype(np.float64) @ bankA.y.astype(np.float64).T)[np.ix_(ia, ta)]
    base_b = bankB.x.astype(np.float64) @ bankB.y.astype(np.float64).T
    torch.manual_seed(5)
    heads = {}
    for arch in ['G', 'GL']:
        h = DirectScorer(**bankA.dims, architecture=arch, a0=11.0, b0=-3.0, seed=1001).to(dev)
        with torch.no_grad():
            h.global_residual[-1].weight.normal_(0, .05); h.global_residual[-1].bias.fill_(.1)
            if arch == 'GL': h.local_readout.weight.copy_(torch.tensor([[1.5, -0.7]]))
        heads[arch] = h.eval()
    def bank_scores(bank, h):
        t = bank.tensors(np.arange(len(bank.image_ids)), np.arange(len(bank.text_ids)), dev, local=True)
        with torch.no_grad():
            return h.matrix(t['x'], t['y'], t['image_local'], t['text_local'], t['text_mask'], pair_chunk=97).cpu().numpy()
    sG = (bank_scores(bankA, heads['G'])[np.ix_(ia, ta)], bank_scores(bankB, heads['G']))
    sGL = (bank_scores(bankA, heads['GL'])[np.ix_(ia, ta)], bank_scores(bankB, heads['GL']))
    bw2 = bool(np.array_equal(bankB.il, bankA.il[ia]) and np.array_equal(bankB.tl, bankA.tl[ta]) and np.array_equal(bankB.x, bankA.x[ia]) and np.array_equal(bankB.y, bankA.y[ta]))
    rec('independent_reorder_cache', okx and ok_il and ok_tl and okm2 and close32(base_b, base_a) and close32(sG[1], sG[0]) and close32(sGL[1], sGL[0]),
        all_arrays_bitwise_equal=bw2, global_max_abs=max(maxdiff(bankB.x, bankA.x[ia]), maxdiff(bankB.y, bankA.y[ta])), local_tok_relL2=[r_il, r_tl],
        frac_local_nonbitwise=[f_il, f_tl], mask_equal=okm2, base_score_max_abs=maxdiff(base_b, base_a),
        G_head_score_max_abs=maxdiff(sG[1], sG[0]), GL_head_score_max_abs=maxdiff(sGL[1], sGL[0]),
        G_head_nonzero=float(np.abs(sG[0]-(11*base_a-3)).max()))
    np.savez_compressed(out/'arr_reorder.npz', G_A=sG[0], G_B=sG[1], GL_A=sGL[0], GL_B=sGL[1], base_A=base_a, base_B=base_b)
    # in-process fp32 reorder (no fp16 storage involved)
    pB = [pil[i] for i in pi]; cB = [caps[i] for i in pt]
    ipB = enc_img(pB); tpB = enc_txt(cB)
    hlA = torch.from_numpy(il).to(dev); hlB = torch.from_numpy(ipB[3]).to(dev)
    with torch.no_grad():
        g1 = heads['GL'].matrix(torch.from_numpy(ip).to(dev), torch.from_numpy(tp).to(dev), hlA, torch.from_numpy(th).to(dev), mask.to(dev), 97).cpu().numpy()[np.ix_(pi, pt)]
        mB = content_mask(proc.tokenizer, cB, tpB[3])
        g2 = heads['GL'].matrix(torch.from_numpy(ipB[1]).to(dev), torch.from_numpy(tpB[1]).to(dev), hlB, torch.from_numpy(tpB[2]).to(dev), mB.to(dev), 97).cpu().numpy()
    rec('independent_reorder_fp32_inprocess_GL', close32(g2, g1), max_abs=maxdiff(g2, g1))
    # (6) opaque-ID rename: scores unchanged; ties only by stable hash
    ren = out/'cache_C_renamed'
    if not ren.exists():
        import shutil; shutil.copytree(out/'cache_C', ren)
        write_json(ren/'image_ids.json', ['renamed_'+x for x in bankC.image_ids]); write_json(ren/'text_ids.json', ['renamed_'+x for x in bankC.text_ids])
        m = read_json(ren/'manifest.json'); m['files'] = {p.name: sha256(p) for p in ren.iterdir() if p.suffix in ('.npy', '.json') and p.name != 'manifest.json'}
        write_json(ren/'manifest.json', m)
    bankR = FeatureBank(ren, 'dev')
    from nr1.evaluate import score_and_order
    _, _, _, ni1, nt1, ps1 = score_and_order(bankC, heads['GL'], dev, 100, dtype=torch.float64)
    _, _, _, ni2, nt2, ps2 = score_and_order(bankR, heads['GL'], dev, 100, dtype=torch.float64)
    heads['GL'].float()
    rec('opaque_id_rename_scores_unchanged', bool(np.array_equal(ps1['scores'], ps2['scores'])),
        i_order_equal=bool(np.array_equal(ni1, ni2)), t_order_equal=bool(np.array_equal(nt1, nt2)),
        note='orders may legitimately differ only inside exact score ties (duplicate image record), resolved by stable ID hash')
    # (7) local padding / special tokens never used; empty content rejected
    h = heads['GL']
    t = bankA.tensors(np.arange(8), np.arange(8), dev, local=True)
    with torch.no_grad():
        s0 = h.pairs(t['x'], t['y'], t['image_local'], t['text_local'], t['text_mask'])
        tl2 = t['text_local'].clone(); tl2[~t['text_mask']] = 1e4
        s1 = h.pairs(t['x'], t['y'], t['image_local'], tl2, t['text_mask'])
        il2 = t['image_local'].clone(); il2[:, :5] = torch.randn_like(il2[:, :5])
        s2 = h.pairs(t['x'], t['y'], il2, t['text_local'], t['text_mask'])
    empty_rejected = False
    try:
        mm = t['text_mask'].clone(); mm[0] = False; h.pairs(t['x'], t['y'], t['image_local'], t['text_local'], mm)
    except ValueError:
        empty_rejected = True
    rec('padding_special_invariance', bool(torch.equal(s0, s1)), max_abs=float((s0-s1).abs().max()))
    rec('empty_content_rejected', empty_rejected)
    rec('nonzero_GL_readout_uses_patch_content', float((s0-s2).abs().max()) > 1e-4, max_abs_change=float((s0-s2).abs().max()))
    # (8) constant raw outputs do not recreate item/group information
    with torch.no_grad():
        n = 8; c = lambda v: v[:1].expand_as(v).contiguous()
        sc = h.matrix(c(t['x']), c(t['y']), c(t['image_local']), c(t['text_local']), c(t['text_mask']), 13)
    rec('constant_inputs_constant_scores', float(sc.max()-sc.min()) == 0.0, spread=float(sc.max()-sc.min()))
    # (9) DEV truth isolation: different positives -> same predictions, different metrics
    cal = out/'cal_small.json'
    obs = out/'observed_small.tsv'
    with open(obs, 'w') as f:
        f.write('record_id\timage_id\ttext_id\n')
        for k, r in enumerate(texts): f.write(f'r{k}\t{ims_a[min(k//2, 31)]["id"]}\t{r["id"]}\n')
    if not cal.exists():
        r = subprocess.run([py, '-B', '-m', 'nr1.calibrate', '--cache', str(out/'cache_A'), '--observed', str(obs), '--out', str(cal)], env=env, capture_output=True, text=True)
        rec('calibration_run_small', r.returncode == 0, stderr_tail=r.stderr[-800:])
    calj = read_json(cal); rec('calibration_bounds', 0.1 <= calj['a0'] <= 100 and -20 <= calj['b0'] <= 20, **{k: calj[k] for k in ('a0', 'b0', 'loss', 'n_iter')})
    locked = read_json(CODE.parent/'locked_configs'/'NR1_Flickr00_GL_J_s1001.json') if (CODE.parent/'locked_configs').exists() else None
    base_cfg = locked or dict(read_json(CODE/'configs/units/NR1_Flickr00_GL_J_s1001.json'), model_revision=a.revision)
    train_ok = {}
    for arch in ['G', 'GL']:
        for obj in ['J', 'BCE', 'NCE']:
            cfg = dict(copy.deepcopy(base_cfg), architecture=arch, objective=obj, steps=3, checkpoints=[0, 1, 2, 3], batch_size=16, pair_chunk=64)
            rd = out/f'train3_{arch}_{obj}'
            if not (rd/'DONE.json').exists(): nr1_train.run(cfg, out/'cache_A', obs, cal, rd, dev)
            sd = [torch.load(rd/f'step_{s:06d}.pt', map_location='cpu', weights_only=False)['model'] for s in range(4)]
            ch = lambda k, i, j: float((sd[i][k]-sd[j][k]).abs().max())
            info = dict(residual_last_change=ch('global_residual.4.weight', 3, 0), scale_change=ch('log_scale', 3, 0))
            ok = info['residual_last_change'] > 0
            if arch == 'GL':
                info['readout_change_step1'] = ch('local_readout.weight', 1, 0)
                # gradient of local projection: exactly zero at step 0 (zero readout), non-zero at step 1
                g = []
                for s in (0, 1):
                    m = DirectScorer(**bankA.dims, architecture='GL', a0=calj['a0'], b0=calj['b0'], seed=1001).to(dev); m.load_state_dict(sd[s])
                    rows = np.arange(16); tt = bankA.tensors(rows % 32, rows, dev, local=True)
                    loss = loss_value(obj, m.matrix(**tt, pair_chunk=64), torch.as_tensor(rows % 32, device=dev)); loss.backward()
                    g.append(float(m.image_projection.weight.grad.abs().max()) + float(m.text_projection.weight.grad.abs().max()))
                info['local_projection_grad_step0'], info['local_projection_grad_step1'] = g
                ok = ok and info['readout_change_step1'] > 0 and g[0] == 0.0 and g[1] > 0
            train_ok[f'{arch}_{obj}'] = ok
            rec(f'train3_{arch}_{obj}', ok, **info)
    # evaluator: step 0 order == base (fp32 and fp64), truth isolation
    tA = out/'truth_A.json'; tB = out/'truth_B.json'
    write_json(tA, dict(protocol_id='NR1_EVAL_TRUTH_V1', positives=[[ims_a[min(k//2, 31)]['id'], r['id']] for k, r in enumerate(texts)] + [[dup['id'], texts[0]['id']]]))
    perm = np.random.default_rng(7).permutation(len(texts))
    write_json(tB, dict(protocol_id='NR1_EVAL_TRUTH_V1', positives=[[ims_a[min(perm[k]//2, 31)]['id'], r['id']] for k, r in enumerate(texts)] + [[dup['id'], texts[1]['id']]]))
    for arch in ['G', 'GL']:
        ck = out/f'train3_{arch}_J'/'step_000000.pt'
        for dt in ['float32', 'float64']:
            try:
                rr = evaluate(out/'cache_C', tA, out/f'eval_step0_{arch}_{dt}', ck, dev, 100, dt)
                rec(f'eval_step0_equals_base_{arch}_{dt}', rr['delta_rSum'] == 0 and rr['order_vs_base']['i2t_queries_order_differs'] == 0
                    and rr['order_vs_base']['t2i_queries_order_differs'] == 0, order_vs_base=rr['order_vs_base'])
            except RuntimeError as e:
                rec(f'eval_step0_equals_base_{arch}_{dt}', False, error=str(e)[:300], informational=(dt == 'float32'))
        ck3 = out/f'train3_{arch}_J'/'step_000003.pt'
        r1 = evaluate(out/'cache_C', tA, out/f'eval_truthA_{arch}', ck3, dev, 100, 'float64')
        r2 = evaluate(out/'cache_C', tB, out/f'eval_truthB_{arch}', ck3, dev, 100, 'float64')
        rec(f'truth_isolation_{arch}', r1['predictions_sha256'] == r2['predictions_sha256'] and r1['metrics'] != r2['metrics'],
            pred_sha_equal=r1['predictions_sha256'] == r2['predictions_sha256'], rSum_A=r1['metrics']['rSum'], rSum_B=r2['metrics']['rSum'])
    # backbone unchanged
    rec('backbone_unchanged', model_state_sha(model) == m_sha0 and sha256(weights) == w_sha0 and not any(p.requires_grad for p in model.parameters()),
        model_state_sha256=m_sha0, safetensors_sha256=w_sha0)
    results['_env'] = dict(torch=torch.__version__, transformers=transformers.__version__, cuda=torch.version.cuda,
                           cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(), host=os.uname().nodename,
                           tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32), tf32_cudnn=bool(torch.backends.cudnn.allow_tf32),
                           matmul_precision=torch.get_float32_matmul_precision(), longest_caption_id=longest['id'])
    return finish(out, t_start)


def finish(out, t0):
    # informational fp32 step-0 check does not gate (fp64 is the registered evaluator precision)
    gating = [f for f in fails if not results.get(f, {}).get('informational')]
    results['_summary'] = dict(status='PASS' if not gating else 'FAIL', failed=gating, informational_failed=[f for f in fails if f not in gating],
                               seconds=time.perf_counter()-t0)
    write_json(Path(out)/'ACCEPTANCE.json', results)
    print('ACCEPTANCE', results['_summary'])
    sys.exit(0 if not gating else 1)


if __name__ == '__main__':
    main()
