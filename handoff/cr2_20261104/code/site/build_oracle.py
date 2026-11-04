"""ORACLE ISOLATED PROCESS (Flickr only). Builds the KEEP_ORACLE CSR from CR1 J_CAP + annotation truth.

Truth = explicit NR1 audit join (caption karpathy_img_id -> image opaque id), never record//5.
Output directory is separate (outputs_cr2/.../oracle/) and every receipt is marked ORACLE. The deployable
builder, trainers and model selection never import this file or read its truth inputs.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, sys
from pathlib import Path
import numpy as np
from scipy.sparse import save_npz

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'reference'))
from oracle_relations import oracle_missing   # noqa: E402  (package reference, byte copy)

NR1 = '/home/infres/yinwang/CS_QMI/outputs_nr1/NR1_20261104'
CR1 = Path('/projects/EEG-foundation-model/yinghao/outputs_cr1/CR1_20261104')


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True, type=Path); ap.add_argument('--out', required=True, type=Path)
    a = ap.parse_args(); cfg = json.loads(a.config.read_text())
    if not cfg['oracle'] or cfg['policy'] != 'KEEP_ORACLE' or not cfg['condition'].startswith('Flickr'): raise ValueError('Oracle config required (Flickr only)')
    if a.out.exists(): raise FileExistsError('Do not overwrite')
    c_ = cfg['condition']; bank = CR1/'banks'/c_/'fit_bank.npz'; prep = CR1/c_/'prepared'/'graph.npz'; par = CR1/c_/'relations'/'J_CAP.npz'
    meta = json.loads(Path(str(par)+'.json').read_text())
    if meta['source'] != 'J' or meta['mode'] != 'CAP' or meta['coupling_sha256'] != sha256(par) or meta['bank_sha256'] != sha256(bank):
        raise ValueError('Parent J_CAP provenance mismatch')
    with np.load(bank, allow_pickle=False) as z:
        obs = z['observed_image_for_text'].copy(); img_id = z['image_id'].astype(str); txt_id = z['text_id'].astype(str); n_image = len(img_id)
    with np.load(prep) as z: c = z['candidates'].copy()
    with np.load(par) as z:
        if not np.array_equal(z['candidates'], c): raise ValueError('Candidates changed')
        w = z['weights'].copy()
    with open(f'{NR1}/audit/ID_MAP_fit_images.tsv') as f: im = {r['karpathy_img_id']: r['opaque_id'] for r in csv.DictReader(f, delimiter='\t')}
    with open(f'{NR1}/audit/ID_MAP_fit_texts.tsv') as f: tx = {r['opaque_id']: r['karpathy_img_id'] for r in csv.DictReader(f, delimiter='\t')}
    row = {x: i for i, x in enumerate(img_id)}
    truth = np.array([row[im[tx[t]]] for t in txt_id], np.int64)
    r = oracle_missing(c, w, obs, truth, n_image)
    a.out.parent.mkdir(parents=True, exist_ok=True); save_npz(a.out, r.matrix)
    rec = dict(r.receipt, unit_id=cfg['unit_id'], protocol_id=cfg['protocol_id'], condition=c_, source='ORACLE', label='ORACLE',
               deployable=False, upper_bound=False, relation_sha256=sha256(a.out), truth_read=True,
               truth_source='NR1 audit ID_MAP explicit karpathy join',
               inputs_sha256={'config': sha256(a.config), 'bank': sha256(bank), 'candidates': sha256(prep), 'parent_graph': sha256(par),
                              'id_map_images': sha256(f'{NR1}/audit/ID_MAP_fit_images.tsv'), 'id_map_texts': sha256(f'{NR1}/audit/ID_MAP_fit_texts.tsv')},
               format='scipy_csr_text_image')
    Path(str(a.out)+'.json').write_text(json.dumps(rec, indent=2)+'\n'); print(json.dumps(rec, indent=1))


if __name__ == '__main__': main()
