"""CPU code-path check of the adapter/runner (2 steps per arm config, SELECT eval, resume roundtrip). Not the GPU gate."""
import sys, time, json, numpy as np, torch
import gpo_adapter as ga, train_gpo as tg
O = '/projects/EEG-foundation-model/yinghao/outputs_v9/r32_native_ratio_20261101'
IMS = '/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data/f30k_precomp'
tg.set_precision(); torch.set_num_threads(4)
rows = tg.matrix_rows()
pools = tg.load_pools(f'{O}/data', IMS, 60, 'cpu', roles=('fit', 'select', 'audit', 'dev'))
w2i = ga.load_vocab()
res = {}
parent = None
for u in ('nrcs_v1_w2_n60_s1001_cosine_hinge_b0_i0p1', 'nrcs_v1_e2_n60_s1001_ratio_J_b1_i0p1', 'nrcs_v1_e2_n60_s1001_ratio_logistic_b1_i0p1',
          'nrcs_v1_e2_n60_s1001_cosine_J_b0_i0', 'nrcs_v1_e2_n60_s1001_cosine_hinge_b0_i0p1'):
    R = tg.Runner(u, rows[u], pools, w2i, 'cpu', total_epochs=1)
    R.init_model(parent if rows[u]['stage'] == 'E2' else None)
    t = time.time(); recs = [R.step(diag=True) for _ in range(2)]
    res[u] = {'loss': [r['loss'] for r in recs], 'J': [r['J'] for r in recs], 'bal': [r['balance_raw'] for r in recs],
              'intra': [r['intra'] for r in recs], 'grad': [r['grad_norm_preclip'] for r in recs], 'sec': time.time() - t,
              'diag_main': recs[0]['diag']['main'], 'digest': R.batch_digests[:2], 'transfer': R.transfer}
    if rows[u]['stage'] == 'W2':
        parent = {k: v.clone() for k, v in R.model.state_dict().items()}
        t = time.time(); ev = R.eval_pool('select'); res[u]['select_eval'] = {'J': ev['J'], 'rsum': ev['retrieval']['rsum'], 'sec': time.time() - t}
    print(u, json.dumps(res[u], default=str)[:600], flush=True)
s = R.full_state(); R2 = tg.Runner(u, rows[u], pools, w2i, 'cpu', total_epochs=1); R2.init_model(None); R2.load_full_state(s)
a, b = R.step(), R2.step(); print('resume loss', a['loss'], b['loss'], abs(a['loss'] - b['loss']))
print('CPU_CHECK_DONE')
