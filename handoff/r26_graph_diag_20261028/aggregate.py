"""Run the package's graph_snapshot_audit.audit on every exported NPZ (frozen SCALE_<ds>.json c_J) and aggregate.
Usage: aggregate.py DIAG_DIR SCALE_JSON. Writes AUDIT/<npz>.json and AGGREGATE.json. NumPy only, no DEV/TEST."""
import sys, os, json, glob, hashlib
import numpy as np
sys.path.insert(0, '/home/infres/yinwang/CS_QMI/adjustments_20261028/VCS_Server_Adjustments_20261028/retrieval')
from graph_snapshot_audit import audit

d, scale_path = sys.argv[1], sys.argv[2]
sb = open(scale_path, 'rb').read()
scales = {k: float(json.loads(sb)['c_J'][k]) for k in ('d_vcs', 'd_bce', 'm_vcs')}
os.makedirs(os.path.join(d, 'AUDIT'), exist_ok=True)
groups = {}
for fn in sorted(glob.glob(os.path.join(d, '*.npz'))):
    with np.load(fn, allow_pickle=False) as z:
        r = audit(z['X'], z['A'], z['Y'], scales, z.files)
    r['snapshot_path'] = fn; r['scales_source'] = {'path': scale_path, 'sha256': hashlib.sha256(sb).hexdigest()}
    json.dump(r, open(os.path.join(d, 'AUDIT', os.path.basename(fn)[:-4] + '.json'), 'w'), indent=1, allow_nan=False)
    cell, label, peer, branch, _ = os.path.basename(fn)[:-4].split('__')
    g = r['gradient_comparisons']
    row = {'J_D': r['objective']['J_D'], 'J_M_minus_J_D': r['objective']['J_M_minus_J_D'],
           'gap_identity_abs_error': r['objective']['gap_identity_abs_error'],
           'M': r['shape']['M'], 'shared_atom_fraction': r['graph']['shared_atom_fraction'], 'max_column_mass': r['graph']['max_column_mass'],
           'P_F_median': r['scores']['P_weighted_quantiles']['0.5'], 'Q_F_median': r['scores']['Q_weighted_quantiles']['0.5'],
           'BCE_clipped_P_mass': r['scores']['BCE_clipped_P_mass'], 'BCE_clipped_Q_mass': r['scores']['BCE_clipped_Q_mass']}
    for lab in ('query', 'query_tangent'):
        for k in ('D_VCS_vs_BCE_cosine', 'D_VCS_vs_M_VCS_cosine', 'scaled_BCE_over_VCS_norm', 'scaled_BCE_minus_VCS_relative_norm',
                  'scaled_M_minus_D_relative_norm', 'raw_covariance_penalty_over_M_gradient_norm'):
            row[f'{lab}.{k}'] = g[lab][k]
    groups.setdefault(f'{cell}|{label}|{branch}', []).append(row)
agg = {'scales': scales, 'scale_sha256': hashlib.sha256(sb).hexdigest(), 'note': 'per (cell, checkpoint, direction): median [p10, p90] over both peers x fixed FIT batches; query-embedding gradients, not encoder gradients; frozen-checkpoint reconstruction', 'groups': {}}
for key, rows in groups.items():
    agg['groups'][key] = {'n': len(rows)}
    for m in rows[0]:
        v = np.array([r[m] for r in rows if r[m] is not None], dtype=float)
        agg['groups'][key][m] = [float(np.median(v)), float(np.quantile(v, .1)), float(np.quantile(v, .9))] if len(v) else None
json.dump(agg, open(os.path.join(d, 'AGGREGATE.json'), 'w'), indent=1)
print(json.dumps({k: {m: agg['groups'][k][m] for m in ('n', 'J_M_minus_J_D', 'query_tangent.D_VCS_vs_BCE_cosine', 'query_tangent.scaled_BCE_minus_VCS_relative_norm', 'query_tangent.scaled_M_minus_D_relative_norm')} for k in agg['groups']}, indent=1))
