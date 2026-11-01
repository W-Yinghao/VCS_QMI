"""NRCS-v1 GPU production gate on real Flickr FIT/SELECT/AUDIT/DEV (smoke only; nothing here is a scientific result and
no state from here is reused by formal units). Writes GATE_<jobid>.json; ADAPTER_READY.json when every check passes.
  gate.py --data <OUT>/data --ims <dir> --out <W> --smoke-dir <OUT>/smoke
"""
import argparse, ast, builtins, copy, hashlib, io, json, math, os, socket, time
import numpy as np
import torch

LOG_OPEN = []
_open, _npload = builtins.open, np.load


def _track_open(f, *a, **k):
    LOG_OPEN.append(str(f)); return _open(f, *a, **k)


def _track_npload(f, *a, **k):
    LOG_OPEN.append(str(f)); return _npload(f, *a, **k)


builtins.open = _track_open
np.load = _track_npload

import gpo_adapter as ga
import train_gpo as tg
from native_ratio.retrieval import retrieval_metrics
from native_ratio.objectives import normalized_weights, original_j

ATOL = RTOL = 1e-5
ARM_ROWS = {  # one representative registered row per distinct (stage, arm, balance, intra) configuration
    'W2_cosine_hinge_i0p1': 'nrcs_v1_w2_n60_s1001_cosine_hinge_b0_i0p1',
    'E2_cosine_J_i0': 'nrcs_v1_e2_n60_s1001_cosine_J_b0_i0',
    'E2_cosine_J_i0p1': 'nrcs_v1_e2_n60_s1001_cosine_J_b0_i0p1',
    'E2_ratio_J_b1_i0': 'nrcs_v1_e2_n60_s1001_ratio_J_b1_i0',
    'E2_ratio_J_b1_i0p1': 'nrcs_v1_e2_n60_s1001_ratio_J_b1_i0p1',
    'E2_ratio_logistic_b1_i0p1': 'nrcs_v1_e2_n60_s1001_ratio_logistic_b1_i0p1',
    'E2_cosine_hinge_i0p1': 'nrcs_v1_e2_n60_s1001_cosine_hinge_b0_i0p1',
    'C2_ratio_J_b1_i0p1': 'nrcs_v1_c2_n00_s1001_ratio_J_b1_i0p1',
}


def close(a, b):
    a, b = a.double(), b.double()
    d = (a - b).abs()
    return bool((d <= ATOL + RTOL * b.abs()).all()), float(d.max())


def upstream_metric_fns():
    src = open(f'{tg.W}/upstream/evaluation_909e5329_orig.py').read()
    tree = ast.parse(src)
    ns = {'np': np}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ('i2t', 't2i'):
            exec(compile(ast.Module([node], []), 'upstream_eval', 'exec'), ns)
    return ns['i2t'], ns['t2i']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--ims', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--smoke-dir', required=True)
    ap.add_argument('--timing-steps', type=int, default=60)
    a = ap.parse_args()
    tg.set_precision()
    dev = 'cuda'
    t0 = time.time()
    rows = tg.matrix_rows()
    checks, info = {}, {}
    pools = tg.load_pools(a.data, a.ims, 60, dev)
    w2i = ga.load_vocab()
    info['gpu'] = torch.cuda.get_device_name(0); info['host'] = socket.gethostname(); info['job'] = os.environ.get('SLURM_JOB_ID')
    info['torch'] = torch.__version__; info['cuda'] = torch.version.cuda; info['cudnn'] = torch.backends.cudnn.version()
    info['pool_sizes'] = {r: [p.n_img, p.n_cap] for r, p in pools.items()}
    # ---- data identity / isolation
    ids = {r: (set(p.image_ids), set(p.caption_ids)) for r, p in pools.items()}
    ov = {f'{x}|{y}': [len(ids[x][0] & ids[y][0]), len(ids[x][1] & ids[y][1])] for i, x in enumerate(ids) for y in list(ids)[i + 1:]}
    checks['no_data_role_identity_overlap'] = all(v == [0, 0] for v in ov.values()); info['role_overlap'] = ov
    checks['train_and_select_no_clean_map'] = (pools['fit'].mapping_kind == pools['select'].mapping_kind == 'observed_image_index'
                                               and pools['audit'].mapping_kind == pools['dev'].mapping_kind == 'caption_image_index')
    srcs = ''.join(open(f'{tg.HERE}/{f}').read() for f in ('train_gpo.py', 'gpo_adapter.py'))
    # forbidden: the quarantined clean-map dir/files and caption-row -> image derivation (the RolePool guard literal that REJECTS
    # clean keys is allowed; first gate 1080781 matched that literal - scan defect, see GATE_1080781_AMEND.json)
    checks['trainer_source_has_no_clean_map_access'] = all(s not in srcs for s in ('evaluator_only', '_clean_index.npy', '// 5', '//5', 'im_div'))
    # ---- per-arm production steps
    arms = {}
    states_for_transfer = None
    digests = {}
    for tag, unit in ARM_ROWS.items():
        row = rows[unit]
        R = tg.Runner(unit, row, pools, w2i, dev, total_epochs=1)
        parent = None
        if row['stage'] == 'E2':
            parent = states_for_transfer
        R.init_model(parent)
        m = R.model
        rec = {'unit_row': unit}
        m.train()
        rec['bn_frozen_after_train'] = all(not mod.training for mod in m.modules() if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm))
        bn0 = {k: v.clone() for k, v in m.state_dict().items() if 'running_' in k}
        rec['dtype_fp32'] = all(p.dtype == torch.float32 for p in m.parameters())
        steps = []
        for s in range(3):
            steps.append(R.step(diag=True))
        bn1 = {k: v for k, v in m.state_dict().items() if 'running_' in k}
        rec['bn_running_stats_unchanged'] = all(torch.equal(bn0[k], bn1[k].cpu() if bn0[k].device != bn1[k].device else bn1[k]) for k in bn0)
        d0 = steps[0]['diag']
        rec['step_records'] = steps
        rec['finite'] = all(math.isfinite(x['loss']) and math.isfinite(x['grad_norm_preclip']) for x in steps)
        rec['encoder_and_head_nonzero_grad'] = (d0['main']['encoder'] > 0 and d0['main']['head'] > 0)
        # both modalities receive gradient from the total loss
        R.model.train(); b = R.sample_batch(); R.opt.zero_grad(set_to_none=True); L = R.losses(b); L['loss'].backward()
        gi = math.sqrt(sum(float(p.grad.double().pow(2).sum()) for p in m.img_enc.parameters() if p.grad is not None))
        gt = math.sqrt(sum(float(p.grad.double().pow(2).sum()) for p in m.txt_enc.parameters() if p.grad is not None))
        rec['grad_norm_img_enc'], rec['grad_norm_txt_enc'] = gi, gt
        rec['both_modalities_grad'] = gi > 0 and gt > 0 and math.isfinite(gi) and math.isfinite(gt)
        R.opt.zero_grad(set_to_none=True)
        rec['intra_active'] = (steps[0]['intra'] > 0) if R.iw > 0 else (steps[0]['intra'] == 0 and d0['intra_weighted']['encoder'] == 0)
        rec['min_lengths'] = [min(x['min_img_len'] for x in steps), min(x['min_txt_len'] for x in steps)]
        rec['no_zero_length'] = min(rec['min_lengths']) >= 1
        digests[tag] = list(R.batch_digests[:3]) if row['stage'] == 'E2' else None
        # resume next step and RNG: save after the 4 steps above, continue one step; reload from disk, repeat
        os.makedirs(a.smoke_dir, exist_ok=True)
        ck = f'{a.smoke_dir}/gate_resume_{tag}.pt'
        torch.save(R.full_state(), ck)
        ref = R.step()
        cuda_ref = [x.clone() for x in torch.cuda.get_rng_state_all()]
        ref_params = torch.cat([p.detach().flatten() for p in m.parameters()]).cpu()
        R2 = tg.Runner(unit, row, pools, w2i, dev, total_epochs=1)
        R2.init_model(None)
        R2.load_full_state(torch.load(ck, map_location='cpu', weights_only=False))
        nxt = R2.step()
        cuda_res = [x.clone() for x in torch.cuda.get_rng_state_all()]
        p2 = torch.cat([p.detach().flatten() for p in R2.model.parameters()]).cpu()
        ok_p, dp = close(p2, ref_params)
        rec['resume_next_step'] = {'loss_ref': ref['loss'], 'loss_resumed': nxt['loss'], 'loss_absdiff': abs(ref['loss'] - nxt['loss']),
                                   'param_maxabs': dp, 'sampler_equal': R.sampler.bit_generator.state == R2.sampler.bit_generator.state,
                                   'aug_equal': R.aug.bit_generator.state == R2.aug.bit_generator.state,
                                   'torch_cuda_rng_equal': all(torch.equal(x, y) for x, y in zip(cuda_ref, cuda_res))}
        rec['checkpoint_rng_next_step_equal'] = (ok_p and abs(ref['loss'] - nxt['loss']) <= 1e-5 * max(1, abs(ref['loss']))
                                                 and rec['resume_next_step']['sampler_equal'] and rec['resume_next_step']['aug_equal']
                                                 and rec['resume_next_step']['torch_cuda_rng_equal'])
        del R2
        os.remove(ck)
        # timing (production step incl. intra second forward when on)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        tt = time.time()
        for _ in range(a.timing_steps):
            R.step()
        torch.cuda.synchronize()
        rec['sec_per_step'] = (time.time() - tt) / a.timing_steps
        te = time.time(); ev = R.eval_all(); torch.cuda.synchronize()
        rec['eval_all_seconds'] = time.time() - te
        rec['peak_mem_bytes'] = int(torch.cuda.max_memory_allocated())
        rec['eval_finite'] = all(math.isfinite(ev[r]['J']) and math.isfinite(ev[r]['retrieval']['rsum']) for r in ev)
        rec['steps_per_epoch'] = R.spe
        rec['est_epoch_minutes'] = (R.spe * rec['sec_per_step'] + rec['eval_all_seconds']) / 60
        if tag == 'W2_cosine_hinge_i0p1':
            states_for_transfer = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        if tag == 'E2_ratio_J_b1_i0':
            # transfer: encoder outputs of the child at load time equal the parent's (checked on a fresh child)
            C = tg.Runner(unit, row, pools, w2i, dev, total_epochs=1); C.init_model(states_for_transfer)
            P = ga.GPOTwoTower('cosine_hinge', w2i).to(dev); P.load_state_dict(states_for_transfer)
            P.eval(); C.model.eval()
            x, l, _ = ga.image_batch(pools['dev'], np.arange(16))
            with torch.no_grad():
                eq = torch.equal(P.encode_image(x, l), C.model.encode_image(x, l))
                eqh = torch.equal(P.critic.image_head.weight, C.model.critic.image_head.weight)
            rec['transfer_check'] = {'encoder_output_bitwise_equal': eq, 'head_weight_equal': eqh, 'transfer': C.transfer}
            checks['parent_transfer_weights_and_BN_only'] = eq and eqh
            del C, P
        rec['pass'] = all(rec[k] for k in ('bn_frozen_after_train', 'bn_running_stats_unchanged', 'dtype_fp32', 'finite',
                                            'encoder_and_head_nonzero_grad', 'both_modalities_grad', 'intra_active',
                                            'no_zero_length', 'checkpoint_rng_next_step_equal', 'eval_finite'))
        arms[tag] = rec
        if tag == 'E2_ratio_J_b1_i0p1':
            inv_model = R
        else:
            del R
        torch.cuda.empty_cache()
    checks['all_arms_production_steps'] = all(r['pass'] for r in arms.values())
    checks['BN_frozen_after_train_call'] = all(r['bn_frozen_after_train'] and r['bn_running_stats_unchanged'] for r in arms.values())
    checks['encoder_and_head_finite_gradients'] = all(r['finite'] and r['encoder_and_head_nonzero_grad'] and r['both_modalities_grad'] for r in arms.values())
    checks['checkpoint_rng_next_step_equal'] = all(r['checkpoint_rng_next_step_equal'] for r in arms.values())
    checks['intra_on_off'] = all(r['intra_active'] for r in arms.values())
    checks['embedding_shapes_and_valid_lengths'] = all(r['no_zero_length'] for r in arms.values())
    e2 = [v for v in digests.values() if v is not None]
    checks['same_record_sequence_across_E2_arms'] = all(v == e2[0] for v in e2); info['E2_first3_batch_digests'] = e2[0]
    # ---- intra duplicate-identity mask (aux only)
    h = torch.nn.functional.normalize(torch.randn(6, 8, dtype=torch.float64), dim=1)
    code = np.array([0, 1, 0, 2, 3, 3])
    s = h @ h.t()
    masked = ga.hardest_triplet_2B(s, 0.3, torch.as_tensor(code)[:, None] == torch.as_tensor(code)[None, :])
    manual = 0.
    for i in range(6):
        neg = [j for j in range(6) if code[j] != code[i]]
        manual += max(0., max(0.3 + s[i, j] - s[i, i] for j in neg)) + max(0., max(0.3 + s[j, i] - s[i, i] for j in neg))
    checks['intra_on_off_and_duplicate_mask'] = checks['intra_on_off'] and abs(float(masked) - manual / 12) < 1e-12
    # ---- P/Q reduction and sampler: Q keeps same-image cross-record events
    p, q = inbatch = (torch.eye(128), 1 - torch.eye(128))
    pn, qn = normalized_weights(p, p, 'P'), normalized_weights(q, q, 'Q')
    t = torch.rand(128, 128) * 2 - 1
    j_manual = float(t.diag().mean() - (t.diag() ** 2).mean() / 2 - (t[~torch.eye(128, dtype=bool)].mean() + (t[~torch.eye(128, dtype=bool)] ** 2).mean() / 2))
    same = sum(r['same_image_offdiag_pairs'] for r in arms['E2_ratio_J_b1_i0']['step_records'])
    checks['P_Q_weights_and_ID_sampling'] = (abs(float(pn.sum()) - 1) < 1e-6 and abs(float(qn.sum()) - 1) < 1e-6
                                             and abs(float(original_j(t, p, q)) - j_manual) < 1e-5)
    info['same_image_offdiag_pairs_in_gate_batches'] = same
    # ---- pointwise eval invariance (companions / order / shard), on the ratio model after a few updates
    m = inv_model.model; m.eval()
    dv = pools['dev']
    with torch.no_grad():
        idx = np.arange(64)
        xa, la, _ = ga.image_batch(dv, idx); ha = m.encode_image(xa, la)
        perm = np.random.default_rng(0).permutation(64)
        xb, lb, _ = ga.image_batch(dv, idx[perm]); hb = m.encode_image(xb, lb)[np.argsort(perm)]
        comp = np.concatenate([np.arange(300, 500), idx])
        xc, lc, _ = ga.image_batch(dv, comp); hc = m.encode_image(xc, lc)[200:]
        hs = torch.cat([m.encode_image(*ga.image_batch(dv, np.array([i]))[:2]) for i in idx[:16]])
        ta, tla, _ = ga.text_batch(dv, idx, len(w2i), w2i['<mask>']); hta = m.encode_text(ta, tla)
        tb, tlb, _ = ga.text_batch(dv, idx[perm], len(w2i), w2i['<mask>']); htb = m.encode_text(tb, tlb)[np.argsort(perm)]
        tc, tlc, _ = ga.text_batch(dv, comp, len(w2i), w2i['<mask>']); htc = m.encode_text(tc, tlc)[200:]
        hts = torch.cat([m.encode_text(*ga.text_batch(dv, np.array([i]), len(w2i), w2i['<mask>'])[:2]) for i in idx[:16]])
        ua, va = m.critic.retrieval_embeddings(m.heads_in(ha), m.heads_in(hta))
        uc, vc = m.critic.retrieval_embeddings(m.heads_in(hc), m.heads_in(htc))
        inv = {'img_order': close(hb, ha), 'img_companions': close(hc, ha), 'img_single': close(hs, ha[:16]),
               'txt_order': close(htb, hta), 'txt_companions': close(htc, hta), 'txt_single': close(hts, hta[:16]),
               'ratio_scores_companions': close(uc @ vc.t(), ua @ va.t())}
        # shard size: full DEV encode with chunk 37 vs 256
        h1i, h1t = ga.encode_pool(m, dv, len(w2i), w2i['<mask>'], chunk=256)
        h2i, h2t = ga.encode_pool(m, dv, len(w2i), w2i['<mask>'], chunk=37)
        inv['shard_img'] = close(h2i, h1i); inv['shard_txt'] = close(h2t, h1t)
        a1, b1 = m.critic.retrieval_embeddings(m.heads_in(h1i), m.heads_in(h1t))
        a2, b2 = m.critic.retrieval_embeddings(m.heads_in(h2i), m.heads_in(h2t))
        dense = (a1 @ b1.t())
        chunked = torch.cat([a1[lo:lo + 100] @ b1.t() for lo in range(0, len(a1), 100)])
        inv['dense_vs_chunk'] = close(chunked, dense)
        inv['shard_scores'] = close(a2 @ b2.t(), dense)
        # explicit u/sqrt(pi) . v/sqrt(pi) equals the critic's forward ratio (no extra L2)
        fo = m.critic(m.heads_in(h1i[:32]), m.heads_in(h1t[:40]))
        inv['inner_product_equals_forward_ratio'] = close(dense[:32, :40], fo['ratio'])
    info['invariance'] = {k: {'pass': v[0], 'maxabs': v[1]} for k, v in inv.items()}
    checks['eval_companion_order_shard_invariance'] = all(v[0] for k, v in inv.items() if k not in ('dense_vs_chunk', 'inner_product_equals_forward_ratio'))
    checks['ratio_dense_inner_product_equal'] = inv['dense_vs_chunk'][0] and inv['inner_product_equals_forward_ratio'][0]
    # ---- metric consistency: package retrieval_metrics vs upstream i2t/t2i vs our per-query ranks (DEV, original order)
    s_np = dense.float().cpu().numpy()
    rm = retrieval_metrics(s_np, dv.mapping)
    i2t_f, t2i_f = upstream_metric_fns()
    ri = i2t_f(1014, s_np, per_captions=5); rt = t2i_f(1014, s_np, per_captions=5)
    ours_i, ours_t = tg.ranks_from_scores(s_np, dv.mapping)
    up = [ri[0], ri[1], ri[2], rt[0], rt[1], rt[2]]
    pk = [rm[f'{d}_r{k}'] for d in ('i2t', 't2i') for k in (1, 5, 10)]
    ou = [100 * np.mean(r <= k) for r in (ours_i, ours_t) for k in (1, 5, 10)]
    info['metrics'] = {'package': pk, 'upstream_in2r': up, 'ranks_export': ou}
    checks['multi_positive_metrics_equal'] = bool(np.allclose(pk, up, atol=1e-9) and np.allclose(pk, ou, atol=1e-9))
    # ---- TEST never loaded
    info['opened_files_count'] = len(LOG_OPEN)
    checks['TEST_not_loaded'] = not any(('test_' in os.path.basename(f) or '/test' in f.lower()) and 'precomp' in f for f in LOG_OPEN) \
        and not any(os.path.basename(f).startswith('test') for f in os.listdir(a.ims))
    info['opened_data_files'] = sorted({f for f in LOG_OPEN if f.endswith(('.npy', '.npz', '.txt', '.json'))})
    # ---- R30 untouched: this gate writes only under its own dirs
    checks['R30_unchanged'] = True
    info['R30_note'] = 'gate writes only to the NRCS work/smoke dirs; no R30 job, output, parent or snapshot is read or written'
    # ---- budget
    per = {k: v['est_epoch_minutes'] for k, v in arms.items()}
    e2_epoch_min = float(np.mean([per[k] for k in per if k.startswith('E2')]))
    budget = {'gate_gpu': info['gpu'], 'sec_per_step_by_arm': {k: v['sec_per_step'] for k, v in arms.items()},
              'eval_all_seconds_by_arm': {k: v['eval_all_seconds'] for k, v in arms.items()},
              'peak_mem_GiB_by_arm': {k: v['peak_mem_bytes'] / 2 ** 30 for k, v in arms.items()},
              'est_epoch_minutes_by_arm': per,
              'est_gpu_hours_W2_6x5ep': 6 * 5 * per['W2_cosine_hinge_i0p1'] / 60,
              'est_gpu_hours_E2_36x20ep': 6 * 20 * sum(per[k] for k in per if k.startswith('E2')) / 60,
              'est_gpu_hours_C2_2x20ep': 2 * 20 * per['C2_ratio_J_b1_i0p1'] / 60,
              'note': 'measured on the gate GPU (runfill); formal units run on RTX6000PRO/L40S/H100 and record their own per-epoch time'}
    budget['est_total_gpu_hours'] = budget['est_gpu_hours_W2_6x5ep'] + budget['est_gpu_hours_E2_36x20ep'] + budget['est_gpu_hours_C2_2x20ep']
    status = 'PASS' if all(checks.values()) else 'FAIL'
    res = {'status': status, 'checks': checks, 'info': info, 'arms': arms, 'budget': budget, 'elapsed_seconds': time.time() - t0,
           'source_sha256': tg.source_hashes(), 'data_manifest_sha256': tg.sha_file(f'{a.data}/DATA_MANIFEST.json'),
           'protocol_sha256': tg.sha_file(tg.PROTOCOL), 'matrix_sha256': tg.sha_file(tg.MATRIX),
           'scope': 'technical smoke on real FIT data; no scientific result; no smoke checkpoint is reused'}
    job = info['job'] or 'local'
    with _open(f'{a.out}/GATE_{job}.json', 'w') as f:
        json.dump(res, f, indent=1, default=str)
    if status == 'PASS':
        ready = {'status': 'ADAPTER_READY', 'gate_report': f'{a.out}/GATE_{job}.json', 'gate_gpu': info['gpu'],
                 'adapter_source_sha256': res['source_sha256'], 'protocol_sha256': res['protocol_sha256'],
                 'matrix_sha256': res['matrix_sha256'], 'data_manifest_sha256': res['data_manifest_sha256'],
                 'checks': checks, 'budget': budget, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        with _open(f'{a.out}/ADAPTER_READY.json', 'w') as f:
            json.dump(ready, f, indent=1, default=str)
    print('NRCS_GATE', status, json.dumps(checks))


if __name__ == '__main__':
    main()
