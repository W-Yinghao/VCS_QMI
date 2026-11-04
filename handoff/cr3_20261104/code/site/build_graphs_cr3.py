"""CR3 deployable graph registry (no truth). Builds OBS and J_MASS_OBS CSRs with the package
reference builders; registers the frozen CR2 KEEP_TOP1 / KEEP_ORACLE CSRs by path+sha (no copy, no change);
dumps observed_image_for_text (.npy) for the read-only transfer audit."""
import hashlib, json, sys
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'pkg'))
from reference.build_controls import observed_relation, mass_observation   # package reference (byte copy)
from reference.support_complete import canonical_relation
CR1 = Path('/projects/EEG-foundation-model/yinghao/outputs_cr1/CR1_20261104'); CR2 = Path('/projects/EEG-foundation-model/yinghao/outputs_cr2/CR2_20261104')
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_cr3/CR3_20261104')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
REL = {'Flickr00': ['OBS', 'J_MASS_OBS'], 'Flickr60': ['OBS', 'J_MASS_OBS', 'BASE_KEEP_TOP1', 'J_KEEP_TOP1', 'BCE_KEEP_TOP1', 'NCE_KEEP_TOP1', 'KEEP_ORACLE'],
       'CC152K_RECOVERED_20261104': ['OBS', 'J_MASS_OBS', 'BASE_KEEP_TOP1', 'J_KEEP_TOP1', 'BCE_KEEP_TOP1', 'NCE_KEEP_TOP1']}
registry = {}
for c, rels in REL.items():
    bank = CR1/'banks'/c/'fit_bank.npz'
    with np.load(bank, allow_pickle=False) as z: obs = z['observed_image_for_text'].copy(); ni = len(z['image_global'])
    d = OUT/'graphs'/c; d.mkdir(parents=True, exist_ok=True)
    np.save(d/'observed_image_for_text.npy', obs)
    for r in rels:
        if r == 'OBS':
            p = d/'OBS.npz'; A = canonical_relation(observed_relation(obs, ni)); save_npz(p, A)
            rec = dict(relation=r, condition=c, built_by='package reference.build_controls.observed_relation', bank_sha256=sha(bank), row_mass='1 per caption')
        elif r == 'J_MASS_OBS':
            par = CR1/c/'relations'/'J_CAP.npz'; meta = json.loads(Path(str(par)+'.json').read_text())
            assert meta['coupling_sha256'] == sha(par) and meta['source'] == 'J' and meta['mode'] == 'CAP'
            z = np.load(par); cc, w = z['candidates'], z['weights']
            Jc = csr_matrix((w.ravel(), (np.repeat(np.arange(len(cc)), cc.shape[1]), cc.ravel())), shape=(len(cc), ni))
            A = mass_observation(Jc, obs); p = d/'J_MASS_OBS.npz'; save_npz(p, A)
            m = np.asarray(Jc.sum(1)).ravel(); rm = np.asarray(A.sum(1)).ravel()
            assert (A.indices == obs[np.repeat(np.arange(len(obs)), np.diff(A.indptr))]).all()
            rec = dict(relation=r, condition=c, built_by='package reference.build_controls.mass_observation(J_CAP, observed)', bank_sha256=sha(bank),
                       parent_J_CAP=str(par), parent_J_CAP_sha256=sha(par), row_mass_max_abs_diff_vs_J_CAP=float(np.abs(rm-m).max()),
                       total_mass=float(A.data.sum()), J_CAP_total_mass=float(w.sum()), nnz=int(A.nnz), edges_are_observed=True,
                       note='m_j is inherited row mass, not a correctness probability')
        elif r == 'KEEP_ORACLE':
            p = CR2/'oracle'/c/f'CR2_{c}_KEEP_ORACLE_s1001.npz'; m = json.loads(Path(str(p)+'.json').read_text()); assert m['relation_sha256'] == sha(p)
            rec = dict(relation=r, condition=c, reused_frozen_CR2=str(p), label='ORACLE', cr2_receipt_sha256=sha(str(p)+'.json'))
        else:
            p = CR2/'relations'/c/f"CR2_{c}_{r.replace('_KEEP_TOP1', '')}_KEEP_TOP1_s1001.npz"; m = json.loads(Path(str(p)+'.json').read_text()); assert m['relation_sha256'] == sha(p)
            rec = dict(relation=r, condition=c, reused_frozen_CR2=str(p), cr2_receipt_sha256=sha(str(p)+'.json'))
        rec.update(path=str(p), relation_sha256=sha(p), shape=list(load_npz(p).shape), truth_read=False)
        if r in ('OBS', 'J_MASS_OBS'): Path(str(p)+'.json').write_text(json.dumps(rec, indent=1))
        registry[f'{c}:{r}'] = rec; print(c, r, rec['relation_sha256'][:12], rec.get('row_mass_max_abs_diff_vs_J_CAP', ''))
(OUT/'graphs'/'GRAPH_REGISTRY_DEPLOYABLE.json').write_text(json.dumps(registry, indent=1))
