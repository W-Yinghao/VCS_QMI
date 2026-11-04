"""CR3 student: same fixed graph, same 64 P_R edge draws, two consumers.

RECT     = byte-for-byte the CR2 student step (CR1 ExposedRelation.batch_edges -> G.matrix -> CR1 soft_nce).
COMPLETE = same draw (edge indices from the identical CR1 sampler), then the package reference
           FixedRelation.build(.,'COMPLETE') adds each original query's full positive graph neighbourhood
           (old rectangle negatives kept, added items are not queries), one pair-score forward over the
           union, package conditional_loss; one backward / clip / AdamW step per update.
Init, optimizer, schedule, checkpoints, resume: CR1 cr1_prod.optim_loop (read-only import). No truth input.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import load_npz

HERE = Path(__file__).resolve().parent
CR1_SITE = Path('/home/infres/yinwang/CS_QMI/handoff/cr1_20261104/code/site')
sys.path.insert(0, str(CR1_SITE)); sys.path.insert(0, str(CR1_SITE.parent/'src')); sys.path.insert(0, str(HERE.parent/'pkg'))
import cr1_prod as C1                                                  # noqa: E402 frozen CR1
from cr1.io import read_json, write_json, sha256                       # noqa: E402
from cr1.models import GlobalScorer                                    # noqa: E402
from cr1.objectives import soft_nce                                    # noqa: E402
from reference.support_complete import FixedRelation, conditional_loss # noqa: E402 package reference (byte copy)

TREE = HERE.parents[1]; FROZEN = TREE/'PROTOCOL_FROZEN_CR3.json'; UNITS = TREE/'package'/'configs'/'units'


def frozen_check_cr3(unit_id=None, files=()):
    if not FROZEN.exists(): raise RuntimeError('PROTOCOL_FROZEN_CR3.json missing: formal units cannot run')
    fz = read_json(FROZEN)
    for rel, h in fz['code_sha256'].items():
        if sha256(Path(rel) if rel.startswith('/') else TREE/rel) != h: raise RuntimeError(f'Frozen code changed: {rel}')
    if unit_id is not None and sha256(UNITS/f'{unit_id}.json') != fz['unit_config_sha256'][unit_id]: raise RuntimeError('Unit config changed')
    for p in files:
        if sha256(p) != fz['data_sha256'][str(p)]: raise RuntimeError(f'Frozen data changed: {p}')
    gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else None
    if fz.get('gpu_env_lock') and gpu != fz['gpu_env_lock']['gpu_name']: raise RuntimeError(f'GPU lock violated: {gpu}')
    if fz.get('gpu_env_lock') and torch.__version__ != fz['gpu_env_lock']['torch']: raise RuntimeError('torch version differs from lock')
    return sha256(FROZEN)


def score_pairs(model, X, Y, pairs, chunk=4096):
    i = torch.as_tensor(pairs[:, 0], device=X.device); t = torch.as_tensor(pairs[:, 1], device=X.device)
    return torch.cat([model.pairs(X[i[k:k+chunk]], Y[t[k:k+chunk]]) for k in range(0, len(i), chunk)])


def train(bank_path, prepared, relation_path, out, unit, consumer, device, steps=8000, check_frozen=True, saves=None, draw_log=None):
    if consumer not in ('RECT', 'COMPLETE'): raise ValueError('consumer')
    det = C1.setup_determinism(device)
    files = [bank_path, Path(prepared)/'PREP.json', Path(prepared)/'graph.npz', relation_path]
    fsha = frozen_check_cr3(unit, files) if check_frozen else None
    bank, prep, graph = C1.get_prepared(bank_path, prepared)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if (out/'DONE.json').exists(): raise FileExistsError('Completed student exists; do not overwrite')
    mat = load_npz(relation_path).tocsr()
    if mat.shape != (len(bank.text_global), len(bank.image_global)): raise ValueError('Relation shape mismatch')
    relation = C1.ExposedRelation(mat)
    fr = None
    if consumer == 'COMPLETE':
        fr = FixedRelation(mat)
        if not (np.array_equal(fr.text_edges, relation.text_edges) and np.array_equal(fr.image_edges, relation.image_edges)
                and np.array_equal(fr.edge_mass, relation.edge_mass)):
            raise AssertionError('Edge enumeration differs between CR1 sampler and package FixedRelation')
    X, Y = C1.features(bank, device); cal = prep['student_common_initialization']
    model = GlobalScorer(bank.dim, cal['a'], cal['b'], 1001).to(device)
    rng = np.random.default_rng(20261104201)
    edge_hits = np.zeros(len(relation.edge_mass), np.int64)
    st = dict(n=0, pairs=0, old_pairs=0, cache_img=0, cache_txt=0, q_img=0, q_txt=0, build_s=0.0, sup_i2t=0, sup_t2i=0, max_pairs=0)
    img_q = np.zeros(len(bank.image_global), np.int64); txt_q = np.zeros(len(bank.text_global), np.int64)
    dl = open(draw_log, 'w') if draw_log else None

    def step_fn(m):
        b = relation.batch_edges(rng, 64)
        if dl is not None: dl.write(json.dumps({'e': b['edges'].tolist(), 'wi': b['image_weight'].tolist(), 'wt': b['text_weight'].tolist()})+'\n')
        np.add.at(edge_hits, b['edges'], 1)
        img_q[b['images']] += np.round(b['image_weight']*64).astype(np.int64); txt_q[b['texts']] += np.round(b['text_weight']*64).astype(np.int64)
        st['n'] += 1; st['q_img'] += len(b['images']); st['q_txt'] += len(b['texts']); st['old_pairs'] += b['pair_count']
        if consumer == 'RECT':   # == CR2 cr2_student.step_fn
            ix = torch.as_tensor(b['images'], device=X.device); tx = torch.as_tensor(b['texts'], device=X.device)
            s = m.matrix(X[ix], Y[tx]); mass = b['mass']/b['mass'].max()
            target = torch.as_tensor(mass, device=X.device, dtype=s.dtype)
            wi = torch.as_tensor(b['image_weight'], device=X.device, dtype=s.dtype)
            wt = torch.as_tensor(b['text_weight'], device=X.device, dtype=s.dtype)
            st['pairs'] += b['pair_count']; st['cache_img'] += len(b['images']); st['cache_txt'] += len(b['texts']); st['max_pairs'] = max(st['max_pairs'], b['pair_count'])
            return soft_nce(s, target, wi, wt), b['pair_count'], {'unique_images': int(len(b['images'])), 'unique_texts': int(len(b['texts'])), 'pairs': int(b['pair_count'])}
        t0 = time.perf_counter()
        d = fr.from_edge_indices(b['edges'])
        if not (np.array_equal(d.images, b['images']) and np.array_equal(d.texts, b['texts'])
                and np.allclose(d.image_weights, b['image_weight'], rtol=0, atol=0) and np.allclose(d.text_weights, b['text_weight'], rtol=0, atol=0)):
            raise AssertionError('COMPLETE query set differs from the shared draw')
        pb = fr.build(d, 'COMPLETE'); st['build_s'] += time.perf_counter()-t0
        ui, ut = np.unique(pb.pairs[:, 0]), np.unique(pb.pairs[:, 1])
        st['pairs'] += len(pb.pairs); st['cache_img'] += len(ui); st['cache_txt'] += len(ut); st['max_pairs'] = max(st['max_pairs'], len(pb.pairs))
        st['sup_i2t'] += sum(v.full_positive_count for v in pb.views if v.direction == 'i2t'); st['sup_t2i'] += sum(v.full_positive_count for v in pb.views if v.direction == 't2i')
        loss = conditional_loss(score_pairs(m, X, Y, pb.pairs), pb)
        return loss, int(len(pb.pairs)), {'unique_images': int(len(b['images'])), 'unique_texts': int(len(b['texts'])), 'pairs': int(len(pb.pairs)),
                                          'cache_images': int(len(ui)), 'cache_texts': int(len(ut))}
    binding = {'unit': unit, 'consumer': consumer, 'config_sha256': sha256(UNITS/f'{unit}.json') if (UNITS/f'{unit}.json').exists() else None,
               'bank_sha256': bank.digest, 'prep_sha256': sha256(Path(prepared)/'PREP.json'), 'relation_sha256': sha256(relation_path),
               'init': cal, 'seed': 1001, 'sampler_seed': 20261104201, 'steps': steps}
    old = C1.SAVES
    if saves is not None: C1.SAVES = saves
    if str(device).startswith('cuda'): torch.cuda.reset_peak_memory_stats()
    try:
        receipt = C1.optim_loop(model, step_fn, out, steps, device, rng, binding)
    finally:
        C1.SAVES = old
        if dl is not None: dl.close()
    n = max(st['n'], 1); p = relation.edge_mass/relation.total
    cost = {'consumer': consumer, 'scope': 'this attempt' if receipt['resumed_from_step'] else f'all {steps} updates',
            'updates': st['n'], 'query_events_per_update': 64, 'mean_queries_image': st['q_img']/n, 'mean_queries_text': st['q_txt']/n,
            'mean_pair_scores_per_update': st['pairs']/n, 'mean_old_rectangle_pairs_per_update': st['old_pairs']/n,
            'pair_scores_per_query_event': st['pairs']/n/64, 'max_pair_scores_in_an_update': st['max_pairs'],
            'mean_cache_reads_per_update': (st['cache_img']+st['cache_txt'])/n, 'mean_cache_image_reads': st['cache_img']/n, 'mean_cache_text_reads': st['cache_txt']/n,
            'mean_full_support_i2t_per_update': st['sup_i2t']/n if consumer == 'COMPLETE' else None,
            'mean_full_support_t2i_per_update': st['sup_t2i']/n if consumer == 'COMPLETE' else None,
            'complete_build_cpu_seconds': st['build_s'], 'synchronized_train_seconds': receipt['synchronized_train_seconds_this_attempt'],
            'peak_gpu_memory_bytes': int(torch.cuda.max_memory_allocated()) if str(device).startswith('cuda') else None,
            'optimizer_steps_per_update': 1, 'backward_reductions_per_update': 1}
    exposure = {'distinct_edges_sampled': int((edge_hits > 0).sum()), 'graph_edges': int(len(relation.edge_mass)), 'edge_ESS': float(1/np.square(p).sum()),
                'max_edge_repeats': int(edge_hits.max()), 'distinct_images_queried': int((img_q > 0).sum()), 'distinct_texts_queried': int((txt_q > 0).sum()),
                'image_query_count_q50_q99_max': [float(np.quantile(img_q[img_q > 0], .5)), float(np.quantile(img_q[img_q > 0], .99)), int(img_q.max())],
                'text_query_count_q50_q99_max': [float(np.quantile(txt_q[txt_q > 0], .5)), float(np.quantile(txt_q[txt_q > 0], .99)), int(txt_q.max())],
                'edge_hits_sha256': __import__('hashlib').sha256(edge_hits.astype(np.int64).tobytes()).hexdigest()}
    np.save(out/'EDGE_EXPOSURE.npy', edge_hits.astype(np.int32))
    receipt.update({'stage': 'student', 'unit': unit, 'consumer': consumer, 'bank_sha256': bank.digest, 'relation_sha256': sha256(relation_path),
                    'relation_summary': relation.summary(), 'exposure': exposure, 'cost': cost,
                    'checkpoint': f'step_{steps:06d}.pt', 'checkpoint_sha256': sha256(out/f'step_{steps:06d}.pt'),
                    'determinism': det, 'frozen_protocol': fsha, 'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                    'torch': torch.__version__, 'cuda': torch.version.cuda})
    write_json(out/'DONE.json', receipt)
    return receipt


def main():
    ap = argparse.ArgumentParser()
    for k in ('bank', 'prepared', 'relation', 'out', 'unit', 'consumer'): ap.add_argument('--'+k, required=True)
    ap.add_argument('--device', default='cuda'); a = ap.parse_args()
    train(a.bank, a.prepared, a.relation, a.out, a.unit, a.consumer, a.device)


if __name__ == '__main__': main()
