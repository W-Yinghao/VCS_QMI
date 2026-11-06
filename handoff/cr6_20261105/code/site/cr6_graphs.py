"""CR6-A1 graph construction for the NEW source set (CPU, no gradients, no truth).
J_CAP/BCE_CAP: CR1 cr1_prod.py match --mode CAP (unchanged CR1 code: same top32 candidates/prior/slack/bound/tolerance).
J_KEEP_TOP1_src2002 / BCE_KEEP_TOP1_src2002: CR2 site/build_relation_pkg.py (unchanged), lambda .5.
J_MASS_OBS_src2002: CR3 package reference build_controls.mass_observation(J_CAP_new, observed) (same lines as CR3 build_graphs_cr3).
Score exports (GPU) are produced in the A1 job by CR1 cr1_prod.py score with --models-root = the CR6 expert root.
"""
from __future__ import annotations
import json, subprocess, sys, time
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix, save_npz
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import cr6_common as K                                                       # noqa: E402

A1 = K.OUT/'A1'; C = 'Flickr60'


def run(cmd):
    t = time.perf_counter(); p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode: raise RuntimeError(f'{cmd}\n{p.stderr[-3000:]}')
    return time.perf_counter() - t


def main():
    K.frozen_check(); bank = K.bank_path(C); secs = {}
    for S in ('J', 'BCE'):
        sc = A1/'scores'/f'{S}.npz'; rel = A1/'relations'/f'{S}_CAP.npz'
        if not Path(str(sc)+'.json').exists(): raise RuntimeError(f'missing score export {sc}')
        if not Path(str(rel)+'.json').exists():
            secs[f'{S}_CAP'] = run([K.PY, '-I', '-B', str(K.CR1_TREE/'code/site/cr1_prod.py'), 'match', '--bank', str(bank), '--scores', str(sc), '--mode', 'CAP', '--out', str(rel), '--device', 'cpu'])
        g = A1/'graphs'/f'{S}_KEEP_TOP1_src2002.npz'
        if not Path(str(g)+'.json').exists():
            secs[f'{S}_KEEP_TOP1'] = run([K.PY, '-I', '-B', str(K.CR2_TREE/'code/site/build_relation_pkg.py'), '--config', str(K.TREE/'configs/graphs'/f'CR6_A1_Flickr60_{S}_KEEP_TOP1_src2002.json'),
                                          '--bank', str(bank), '--candidates', str(K.prep_dir(C)/'graph.npz'), '--parent-graph', str(rel), '--out', str(g)])
    p = A1/'graphs'/'J_MASS_OBS_src2002.npz'
    if not Path(str(p)+'.json').exists():
        t = time.perf_counter(); sys.path.insert(0, str(K.CR3_TREE/'code'/'pkg'))
        from reference.build_controls import mass_observation          # CR3 package reference (byte copy)
        with np.load(bank, allow_pickle=False) as z: obs = z['observed_image_for_text'].copy(); ni = len(z['image_global'])
        par = A1/'relations'/'J_CAP.npz'; meta = K.read_json(str(par)+'.json')
        assert meta['coupling_sha256'] == K.sha256(par) and meta['source'] == 'J' and meta['mode'] == 'CAP'
        z = np.load(par); cc, w = z['candidates'], z['weights']
        Jc = csr_matrix((w.ravel(), (np.repeat(np.arange(len(cc)), cc.shape[1]), cc.ravel())), shape=(len(cc), ni))
        A = mass_observation(Jc, obs); save_npz(p, A)
        m = np.asarray(Jc.sum(1)).ravel(); rm = np.asarray(A.sum(1)).ravel()
        assert (A.indices == obs[np.repeat(np.arange(len(obs)), np.diff(A.indptr))]).all()
        K.write_json(str(p)+'.json', dict(relation='J_MASS_OBS_src2002', condition=C, built_by='CR3 package reference.build_controls.mass_observation(J_CAP_src2002, observed)',
                     bank_sha256=K.sha256(bank), parent_J_CAP=str(par), parent_J_CAP_sha256=K.sha256(par), row_mass_max_abs_diff_vs_J_CAP=float(np.abs(rm-m).max()),
                     total_mass=float(A.data.sum()), J_CAP_total_mass=float(w.sum()), nnz=int(A.nnz), edges_are_observed=True, relation_sha256=K.sha256(p), truth_read=False,
                     note='m_j is inherited row mass of the NEW source J_CAP, not a correctness probability'))
        secs['J_MASS_OBS'] = time.perf_counter() - t
    reg = {}
    for r in ('J_KEEP_TOP1_src2002', 'BCE_KEEP_TOP1_src2002', 'J_MASS_OBS_src2002'):
        q = A1/'graphs'/f'{r}.npz'; reg[r] = dict(path=str(q), relation_sha256=K.sha256(q), receipt_sha256=K.sha256(str(q)+'.json'))
    K.write_json(A1/'GRAPH_REGISTRY_SRC2002.json', dict(graphs=reg, cpu_seconds_this_run=secs, utc=K.utc(), truth_read=False))
    print('GRAPHS_DONE', json.dumps(reg)[:400])


if __name__ == '__main__': main()
