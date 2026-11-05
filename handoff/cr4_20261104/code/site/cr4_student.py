"""CR4 student: CR3 COMPLETE consumer with the final conditional loss replaced by
reference.positive_exclusive.packed_loss(scores, batch, kind='PC').

Everything else is the CR3 COMPLETE path: CR1 ExposedRelation.batch_edges draw (sampler seed 20261104201),
package FixedRelation.from_edge_indices/build(.,'COMPLETE') (byte-identical support_complete.py), one union pair-score
forward (chunk 4096, same score_pairs as CR3), CR1 optim_loop (AdamW, cosine, clip 2, checkpoints, resume, RNG).
kind='CE' replays CR3 COMPLETE (packed_loss delegates to the unchanged conditional_loss). No truth input.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import load_npz

HERE = Path(__file__).resolve().parent
CR1_SITE = Path('/home/infres/yinwang/CS_QMI/handoff/cr1_20261104/code/site')
sys.path.insert(0, str(CR1_SITE)); sys.path.insert(0, str(CR1_SITE.parent/'src')); sys.path.insert(0, str(HERE.parent/'pkg'))
import cr1_prod as C1                                                    # noqa: E402 frozen CR1
from cr1.io import read_json, write_json, sha256                         # noqa: E402
from cr1.models import GlobalScorer                                      # noqa: E402
from reference.support_complete import FixedRelation                     # noqa: E402 package (== CR3 bytes)
from reference.positive_exclusive import packed_loss                     # noqa: E402 package reference


def score_pairs(model, X, Y, pairs, chunk=4096):   # identical to CR3 cr3_student.score_pairs
    i = torch.as_tensor(pairs[:, 0], device=X.device); t = torch.as_tensor(pairs[:, 1], device=X.device)
    return torch.cat([model.pairs(X[i[k:k+chunk]], Y[t[k:k+chunk]]) for k in range(0, len(i), chunk)])


def train(bank_path, prepared, relation_path, out, unit, kind, device, steps=8000, saves=None, draw_log=None):
    if kind not in ('PC', 'CE'): raise ValueError('kind')
    det = C1.setup_determinism(device)
    bank, prep, graph = C1.get_prepared(bank_path, prepared)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if (out/'DONE.json').exists(): raise FileExistsError('Completed student exists; do not overwrite')
    mat = load_npz(relation_path).tocsr()
    if mat.shape != (len(bank.text_global), len(bank.image_global)): raise ValueError('Relation shape mismatch')
    relation = C1.ExposedRelation(mat); fr = FixedRelation(mat)
    if not (np.array_equal(fr.text_edges, relation.text_edges) and np.array_equal(fr.image_edges, relation.image_edges) and np.array_equal(fr.edge_mass, relation.edge_mass)):
        raise AssertionError('Edge enumeration differs between CR1 sampler and package FixedRelation')
    X, Y = C1.features(bank, device); cal = prep['student_common_initialization']
    model = GlobalScorer(bank.dim, cal['a'], cal['b'], 1001).to(device)
    rng = np.random.default_rng(20261104201)
    edge_hits = np.zeros(len(relation.edge_mass), np.int64)
    st = dict(n=0, pairs=0, cache=0, multi_pos_views=0, views=0, no_neg_views=0, build_s=0.0, max_pairs=0)
    dl = open(draw_log, 'w') if draw_log else None
    th = hashlib.sha256()

    def step_fn(m):
        b = relation.batch_edges(rng, 64)
        if dl is not None: dl.write(json.dumps({'e': b['edges'].tolist(), 'wi': b['image_weight'].tolist(), 'wt': b['text_weight'].tolist()})+'\n')
        np.add.at(edge_hits, b['edges'], 1)
        t0 = time.perf_counter(); d = fr.from_edge_indices(b['edges'])
        if not (np.array_equal(d.images, b['images']) and np.array_equal(d.texts, b['texts']) and np.array_equal(d.image_weights, b['image_weight']) and np.array_equal(d.text_weights, b['text_weight'])):
            raise AssertionError('query set differs from the shared draw')
        pb = fr.build(d, 'COMPLETE'); st['build_s'] += time.perf_counter()-t0
        th.update(pb.pairs.tobytes()); th.update(pb.targets.tobytes()); th.update(pb.query_weights.tobytes()); th.update(pb.mask.tobytes())
        pos = (pb.targets > 0).sum(1); neg = (pb.mask & (pb.targets == 0)).sum(1)
        st['n'] += 1; st['pairs'] += len(pb.pairs); st['max_pairs'] = max(st['max_pairs'], len(pb.pairs))
        st['cache'] += len(np.unique(pb.pairs[:, 0])) + len(np.unique(pb.pairs[:, 1]))
        st['views'] += len(pos); st['multi_pos_views'] += int((pos > 1).sum()); st['no_neg_views'] += int((neg == 0).sum())
        loss = packed_loss(score_pairs(m, X, Y, pb.pairs), pb, kind=kind)
        return loss, int(len(pb.pairs)), {'pairs': int(len(pb.pairs)), 'multi_pos_views': int((pos > 1).sum())}
    binding = {'unit': unit, 'consumer': f'COMPLETE_{kind}', 'bank_sha256': bank.digest, 'prep_sha256': sha256(Path(prepared)/'PREP.json'),
               'relation_sha256': sha256(relation_path), 'init': cal, 'seed': 1001, 'sampler_seed': 20261104201, 'steps': steps}
    old = C1.SAVES
    if saves is not None: C1.SAVES = saves
    if str(device).startswith('cuda'): torch.cuda.reset_peak_memory_stats()
    try:
        receipt = C1.optim_loop(model, step_fn, out, steps, device, rng, binding)
    finally:
        C1.SAVES = old
        if dl is not None: dl.close()
    n = max(st['n'], 1)
    cost = {'consumer': f'COMPLETE_{kind}', 'scope': 'this attempt' if receipt['resumed_from_step'] else f'all {steps} updates', 'updates': st['n'],
            'mean_pair_scores_per_update': st['pairs']/n, 'pair_scores_per_query_event': st['pairs']/n/64, 'max_pair_scores_in_an_update': st['max_pairs'],
            'mean_cache_reads_per_update': st['cache']/n, 'multi_positive_view_fraction': st['multi_pos_views']/max(st['views'], 1),
            'no_negative_view_fraction': st['no_neg_views']/max(st['views'], 1), 'complete_build_cpu_seconds': st['build_s'],
            'synchronized_train_seconds': receipt['synchronized_train_seconds_this_attempt'],
            'peak_gpu_memory_bytes': int(torch.cuda.max_memory_allocated()) if str(device).startswith('cuda') else None,
            'optimizer_steps_per_update': 1, 'backward_reductions_per_update': 1}
    receipt.update({'stage': 'student', 'unit': unit, 'consumer': f'COMPLETE_{kind}', 'bank_sha256': bank.digest, 'relation_sha256': sha256(relation_path),
                    'relation_summary': relation.summary(), 'cost': cost,
                    'same_draw_receipt': {'edge_hits_sha256': hashlib.sha256(edge_hits.astype(np.int64).tobytes()).hexdigest(),
                                          'packed_pairs_targets_weights_mask_sha256': th.hexdigest() if not receipt['resumed_from_step'] else 'resumed (partial)'},
                    'checkpoint': f'step_{steps:06d}.pt', 'checkpoint_sha256': sha256(out/f'step_{steps:06d}.pt'), 'determinism': det,
                    'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None, 'torch': torch.__version__, 'cuda': torch.version.cuda})
    np.save(out/'EDGE_EXPOSURE.npy', edge_hits.astype(np.int32))
    write_json(out/'DONE.json', receipt)
    return receipt


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    for k in ('bank', 'prepared', 'relation', 'out', 'unit', 'kind'): ap.add_argument('--'+k, required=True)
    a = ap.parse_args(); train(a.bank, a.prepared, a.relation, a.out, a.unit, a.kind, 'cuda')
