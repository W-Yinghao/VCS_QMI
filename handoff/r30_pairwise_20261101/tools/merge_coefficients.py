"""Merge calib/<parent_id>/COEFFICIENTS_<parent_id>.json (written once per matching H parent by r30_calibrate.py) into
COEFFICIENTS.json (= STATIC_SCALES). Per parent, per arm (Ls/PN/PV/PB; L0 registered 1): c, numerator, denominator, status;
per-batch squared gradient norms and batch ids; code / parent / protocol sha256; parent fingerprint before/after; the source
record path and sha256 (the unit runner re-hashes it). Usage: merge_coefficients.py OUT_BASE ROOT"""
import hashlib, json, os, sys, time
out_base, root = sys.argv[1], sys.argv[2]
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
res = dict(version='R30-v1.1-merged-20261101',
           rule='SERVER_PLAN §7 / PROTOCOL_FROZEN calibration: c_a = sqrt(sum_b ||g_H,b||^2) / sqrt(sum_b ||g_a,b||^2), encoder gradient of the '
                'main term only, first 64 labeled FIT minibatches of each peer at the first GMM split of the matching H parent (s1001; A1: also s1002/s1003, '
                'own calibration per cell), both peers pooled; L (L0 and all stage paths) / N / B = 1; H native. Frozen for the whole run.',
           merged_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), parents={})
for pid in json.load(open(os.path.join(root, 'PROTOCOL_AMENDMENT_A1.json')))['changes']['calibration_parents']:
    p = os.path.join(out_base, 'calib', pid, f'COEFFICIENTS_{pid}.json')
    if not os.path.isfile(p):
        continue
    r = json.load(open(p))
    res['parents'][pid] = dict(coefficients=r['coefficients'], sum_sqnorm=r['sum_sqnorm'], n_batches_per_peer=r['n_batches_per_peer'],
                               n_gradient_pairs=r['n_gradient_pairs'], skipped_lt2=r['skipped_lt2'],
                               per_batch=[dict(peer=b['peer'], b=b['b'], n=b['n'], ids_sha256=b['ids_sha256'], ids=b['ids'],
                                               sqnorm=b['sqnorm'], raw=b['raw']) for b in r['batches']],
                               parent=r['parent'], parent_sha256=r['parent_sha256'], parent_sha256_after=r['parent_sha256_after'],
                               parent_fingerprint_before=r['parent_fingerprint_before'], parent_fingerprint_after=r['parent_fingerprint_after'],
                               parent_unchanged=r['parent_unchanged'], protocol_frozen_sha256=r['protocol_frozen_sha256'],
                               code_sha256=r['code_sha256'], all_ok=r['all_ok'], params_unchanged=r['params_unchanged'],
                               grads_none=r['grads_none_before'] and r['grads_none_after'], split_training_visible=r['split_training_visible'],
                               gpu=r['gpu'], host=r['host'], job=r['job'], seconds=r['seconds'], utc=r['utc'],
                               source_file=p, source_sha256=sha(p))
json.dump(res, open(os.path.join(root, 'COEFFICIENTS.json'), 'w'), indent=1)
print(json.dumps({c: {a: v['coefficients'][a].get('c') for a in v['coefficients']} for c, v in res['parents'].items()}, indent=1))
