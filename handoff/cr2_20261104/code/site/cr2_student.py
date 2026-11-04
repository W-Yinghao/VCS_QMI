"""CR2 student trainer: CR1 student algorithm with a CSR relation loader (site adapter).

Everything except the relation loader is imported unchanged from the frozen CR1 tree
(cr1_prod.optim_loop / ExposedRelation / setup_determinism / features / get_prepared, package
GlobalScorer / soft_nce). The step function below is a line-for-line copy of CR1 train_student's
step_fn. Init = the condition's CR1 PREP student_common_initialization; sampler seed 20261104201;
AdamW 1e-4 -> 1e-5 cosine, wd 1e-4, clip 2, 8000 steps; checkpoints 0/250/.../8000.
No truth input. The relation file must be a frozen CR2 CSR artifact registered in PROTOCOL_FROZEN_CR2.json.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import load_npz

HERE = Path(__file__).resolve().parent
CR1_SITE = Path('/home/infres/yinwang/CS_QMI/handoff/cr1_20261104/code/site')
sys.path.insert(0, str(CR1_SITE)); sys.path.insert(0, str(CR1_SITE.parent/'src'))
import cr1_prod as C1                                         # noqa: E402  frozen CR1 (read-only import)
from cr1.io import read_json, write_json, sha256              # noqa: E402
from cr1.models import GlobalScorer                           # noqa: E402
from cr1.objectives import soft_nce                           # noqa: E402

TREE = HERE.parents[1]; FROZEN = TREE/'PROTOCOL_FROZEN_CR2.json'; UNITS = TREE/'package'/'configs'/'units'


def frozen_check_cr2(unit_id=None, files=()):
    if not FROZEN.exists(): raise RuntimeError('PROTOCOL_FROZEN_CR2.json missing: formal units cannot run')
    fz = read_json(FROZEN)
    for rel, h in fz['code_sha256'].items():
        p = Path(rel) if rel.startswith('/') else TREE/rel
        if sha256(p) != h: raise RuntimeError(f'Frozen code changed: {rel}')
    if unit_id is not None and sha256(UNITS/f'{unit_id}.json') != fz['unit_config_sha256'][unit_id]:
        raise RuntimeError('Unit config changed')
    for p in files:
        if str(p) not in fz['data_sha256']: raise RuntimeError(f'Data file not frozen: {p}')
        if sha256(p) != fz['data_sha256'][str(p)]: raise RuntimeError(f'Frozen data changed: {p}')
    return sha256(FROZEN)


def train(bank_path, prepared, relation_path, out, unit, device, steps=8000, check_frozen=True, saves=None):
    det = C1.setup_determinism(device)
    files = [bank_path, Path(prepared)/'PREP.json', Path(prepared)/'graph.npz', relation_path]
    fsha = frozen_check_cr2(unit, files) if check_frozen else None
    bank, prep, graph = C1.get_prepared(bank_path, prepared)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if (out/'DONE.json').exists(): raise FileExistsError('Completed student exists; do not overwrite')
    rec = read_json(str(relation_path)+'.json')
    if rec['relation_sha256'] != sha256(relation_path): raise ValueError('Relation hash mismatch')
    if check_frozen and rec['unit_id'] != unit: raise ValueError('Relation/unit mismatch')
    mat = load_npz(relation_path).tocsr()
    if mat.shape != (len(bank.text_global), len(bank.image_global)): raise ValueError('Relation shape mismatch')
    relation = C1.ExposedRelation(mat)
    X, Y = C1.features(bank, device); cal = prep['student_common_initialization']
    model = GlobalScorer(bank.dim, cal['a'], cal['b'], 1001).to(device)
    rng = np.random.default_rng(20261104201)
    edge_hits = np.zeros(len(relation.edge_mass), np.int64); st = {'u_img': 0, 'u_txt': 0, 'pairs': 0, 'n': 0}
    img_q = np.zeros(len(bank.image_global), np.int64); txt_q = np.zeros(len(bank.text_global), np.int64)

    def step_fn(m):  # == CR1 train_student.step_fn (+ per-direction query exposure counters)
        b = relation.batch_edges(rng, 64); ix = torch.as_tensor(b['images'], device=X.device); tx = torch.as_tensor(b['texts'], device=X.device)
        s = m.matrix(X[ix], Y[tx]); mass = b['mass']/b['mass'].max()
        target = torch.as_tensor(mass, device=X.device, dtype=s.dtype)
        wi = torch.as_tensor(b['image_weight'], device=X.device, dtype=s.dtype)
        wt = torch.as_tensor(b['text_weight'], device=X.device, dtype=s.dtype)
        np.add.at(edge_hits, b['edges'], 1)
        img_q[b['images']] += np.round(b['image_weight']*64).astype(np.int64); txt_q[b['texts']] += np.round(b['text_weight']*64).astype(np.int64)
        info = {'unique_images': int(len(b['images'])), 'unique_texts': int(len(b['texts'])), 'target_nnz': int((mass > 0).sum())}
        st['u_img'] += info['unique_images']; st['u_txt'] += info['unique_texts']; st['pairs'] += b['pair_count']; st['n'] += 1
        return soft_nce(s, target, wi, wt), b['pair_count'], info
    binding = {'unit': unit, 'config_sha256': sha256(UNITS/f'{unit}.json') if (UNITS/f'{unit}.json').exists() else None,
               'bank_sha256': bank.digest, 'prep_sha256': sha256(Path(prepared)/'PREP.json'),
               'relation_sha256': rec['relation_sha256'], 'init': cal, 'seed': 1001, 'sampler_seed': 20261104201, 'steps': steps}
    old = C1.SAVES
    if saves is not None: C1.SAVES = saves
    try:
        receipt = C1.optim_loop(model, step_fn, out, steps, device, rng, binding)
    finally:
        C1.SAVES = old
    n = max(st['n'], 1)
    p = relation.edge_mass/relation.total
    exposure = {'scope': 'this attempt' if receipt['resumed_from_step'] else f'all {steps} updates',
                'sampled_pair_events': int(64*st['n']), 'distinct_edges_sampled': int((edge_hits > 0).sum()),
                'graph_edges': int(len(relation.edge_mass)), 'distinct_edge_fraction': float((edge_hits > 0).mean()),
                'mean_unique_images_per_batch': st['u_img']/n, 'mean_unique_texts_per_batch': st['u_txt']/n,
                'mean_rectangle_pairs': st['pairs']/n, 'pair_scores_total': int(st['pairs']),
                'max_edge_repeats': int(edge_hits.max()), 'edge_ESS': float(1/np.square(p).sum()),
                'distinct_images_queried': int((img_q > 0).sum()), 'distinct_texts_queried': int((txt_q > 0).sum()),
                'image_query_count_q50_q99_max': [float(np.quantile(img_q[img_q > 0], .5)), float(np.quantile(img_q[img_q > 0], .99)), int(img_q.max())],
                'text_query_count_q50_q99_max': [float(np.quantile(txt_q[txt_q > 0], .5)), float(np.quantile(txt_q[txt_q > 0], .99)), int(txt_q.max())]}
    np.save(out/'EDGE_EXPOSURE.npy', edge_hits.astype(np.int32))
    receipt.update({'stage': 'student', 'unit': unit, 'objective': 'SOFT_NCE', 'bank_sha256': bank.digest,
                    'relation_receipt': {k: rec.get(k) for k in ('policy', 'unit_id', 'relation_sha256', 'total_mass', 'nnz', 'row_mass_max_change', 'oracle')},
                    'relation_summary': relation.summary(), 'exposure': exposure,
                    'checkpoint': f'step_{steps:06d}.pt', 'checkpoint_sha256': sha256(out/f'step_{steps:06d}.pt'),
                    'determinism': det, 'frozen_protocol': fsha,
                    'initialization': 'new G (seed 1001) + CR1 condition common FIT calibration; NOT estimator/NCE prefix',
                    'algorithm': 'CR1 train_student (soft-NCE, CR1 Relation sampler) with CSR relation loader'})
    write_json(out/'DONE.json', receipt)
    return receipt


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--bank', required=True); ap.add_argument('--prepared', required=True)
    ap.add_argument('--relation', required=True); ap.add_argument('--out', required=True); ap.add_argument('--unit', required=True)
    ap.add_argument('--device', default='cuda'); a = ap.parse_args()
    train(a.bank, a.prepared, a.relation, a.out, a.unit, a.device)


if __name__ == '__main__': main()
