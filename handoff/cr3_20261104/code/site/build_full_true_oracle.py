"""ORACLE ISOLATED PROCESS (Flickr60 only): FULL_TRUE_ORACLE = m_j(J_CAP) on the annotated image (explicit NR1 audit join,
never record//5). Output under outputs_cr3/.../oracle/, labelled ORACLE. No deployable process imports this file."""
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix, save_npz
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'pkg'))
from oracle.full_truth_control import full_truth_relation   # package reference (byte copy)
CR1 = Path('/projects/EEG-foundation-model/yinghao/outputs_cr1/CR1_20261104'); OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_cr3/CR3_20261104')
NR1 = '/home/infres/yinwang/CS_QMI/outputs_nr1/NR1_20261104'; c = 'Flickr60'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
bank = CR1/'banks'/c/'fit_bank.npz'; par = CR1/c/'relations'/'J_CAP.npz'
with np.load(bank, allow_pickle=False) as z: img = z['image_id'].astype(str); txt = z['text_id'].astype(str); obs = z['observed_image_for_text'].copy()
z = np.load(par); m = z['weights'].sum(1)
with open(f'{NR1}/audit/ID_MAP_fit_images.tsv') as f: im = {r['karpathy_img_id']: r['opaque_id'] for r in csv.DictReader(f, delimiter='\t')}
with open(f'{NR1}/audit/ID_MAP_fit_texts.tsv') as f: tx = {r['opaque_id']: r['karpathy_img_id'] for r in csv.DictReader(f, delimiter='\t')}
row = {x: i for i, x in enumerate(img)}; truth = np.array([row[im[tx[t]]] for t in txt], np.int64)
A = full_truth_relation(m, truth, len(img), condition=c, oracle=True).tocsr(); A.sum_duplicates(); A.eliminate_zeros(); A.sort_indices()
p = OUT/'oracle'/c/'FULL_TRUE_ORACLE.npz'; p.parent.mkdir(parents=True, exist_ok=True)
if p.exists(): raise FileExistsError(p)
save_npz(p, A)
rec = dict(relation='FULL_TRUE_ORACLE', condition=c, label='ORACLE', deployable=False, upper_bound=False, truth_read=True,
           truth_source='NR1 audit ID_MAP explicit karpathy join', path=str(p), relation_sha256=sha(p), shape=list(A.shape), nnz=int(A.nnz),
           row_mass_max_abs_diff_vs_J_CAP=float(np.abs(np.asarray(A.sum(1)).ravel()-m).max()), total_mass=float(A.data.sum()),
           rows_true_equals_observed=float((truth == obs).mean()), bank_sha256=sha(bank), parent_J_CAP_sha256=sha(par))
Path(str(p)+'.json').write_text(json.dumps(rec, indent=1)); print(json.dumps(rec, indent=1))
