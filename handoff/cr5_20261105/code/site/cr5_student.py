"""CR5 seeded student: RECT (CR3 step, unchanged) or COMPLETE_PC (CR4 step, unchanged) with registered
head seed and sampler seed (CR3/CR4 hard-code 1001 / 20261104201; CR5 seeds 1002/1003 -> sampler 20261104202/203).
With seed 1001 / sampler 20261104201 this file reproduces cr3_student RECT and cr4_student PC (acceptance checks bitwise).
Imports frozen CR1/CR3/CR4 code read-only; no truth input.
"""
import hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import load_npz

HERE = Path(__file__).resolve().parent
sys.path.insert(0, '/home/infres/yinwang/CS_QMI/handoff/cr4_20261104/code/site')
import cr4_student as S4                                                 # noqa: E402 (CR4 score_pairs; pkg path setup)
C1 = S4.C1
from cr1.io import write_json, sha256                                   # noqa: E402
from cr1.models import GlobalScorer                                     # noqa: E402
from cr1.objectives import soft_nce                                     # noqa: E402
from reference.support_complete import FixedRelation                    # noqa: E402
from reference.positive_exclusive import packed_loss                    # noqa: E402


def train(bank_path, prepared, relation_path, out, unit, consumer, head_seed, sampler_seed, device, steps=8000, saves=None, draw_log=None):
    if consumer not in ('RECT', 'COMPLETE_PC'): raise ValueError(consumer)
    det = C1.setup_determinism(device)
    bank, prep, graph = C1.get_prepared(bank_path, prepared)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if (out/'DONE.json').exists(): raise FileExistsError('Completed student exists; do not overwrite')
    mat = load_npz(relation_path).tocsr()
    if mat.shape != (len(bank.text_global), len(bank.image_global)): raise ValueError('shape')
    relation = C1.ExposedRelation(mat); fr = FixedRelation(mat) if consumer == 'COMPLETE_PC' else None
    if fr is not None and not (np.array_equal(fr.text_edges, relation.text_edges) and np.array_equal(fr.image_edges, relation.image_edges) and np.array_equal(fr.edge_mass, relation.edge_mass)):
        raise AssertionError('edge enumeration differs')
    X, Y = C1.features(bank, device); cal = prep['student_common_initialization']
    model = GlobalScorer(bank.dim, cal['a'], cal['b'], int(head_seed)).to(device)
    rng = np.random.default_rng(int(sampler_seed))
    edge_hits = np.zeros(len(relation.edge_mass), np.int64); st = dict(n=0, pairs=0, build_s=0.0)
    dl = open(draw_log, 'w') if draw_log else None

    def step_fn(m):
        b = relation.batch_edges(rng, 64)
        if dl is not None: dl.write(json.dumps({'e': b['edges'].tolist()})+'\n')
        np.add.at(edge_hits, b['edges'], 1); st['n'] += 1
        if consumer == 'RECT':   # == CR3 cr3_student RECT step
            ix = torch.as_tensor(b['images'], device=X.device); tx = torch.as_tensor(b['texts'], device=X.device)
            s = m.matrix(X[ix], Y[tx]); mass = b['mass']/b['mass'].max()
            target = torch.as_tensor(mass, device=X.device, dtype=s.dtype)
            wi = torch.as_tensor(b['image_weight'], device=X.device, dtype=s.dtype); wt = torch.as_tensor(b['text_weight'], device=X.device, dtype=s.dtype)
            st['pairs'] += b['pair_count']
            return soft_nce(s, target, wi, wt), b['pair_count'], {'pairs': int(b['pair_count'])}
        t0 = time.perf_counter(); d = fr.from_edge_indices(b['edges'])   # == CR4 cr4_student PC step
        if not (np.array_equal(d.images, b['images']) and np.array_equal(d.texts, b['texts'])): raise AssertionError('query set differs')
        pb = fr.build(d, 'COMPLETE'); st['build_s'] += time.perf_counter()-t0; st['pairs'] += len(pb.pairs)
        return packed_loss(S4.score_pairs(m, X, Y, pb.pairs), pb, kind='PC'), int(len(pb.pairs)), {'pairs': int(len(pb.pairs))}
    binding = {'unit': unit, 'consumer': consumer, 'bank_sha256': bank.digest, 'prep_sha256': sha256(Path(prepared)/'PREP.json'), 'relation_sha256': sha256(relation_path),
               'init': cal, 'head_seed': int(head_seed), 'sampler_seed': int(sampler_seed), 'steps': steps}
    old = C1.SAVES
    if saves is not None: C1.SAVES = saves
    if str(device).startswith('cuda'): torch.cuda.reset_peak_memory_stats()
    try: receipt = C1.optim_loop(model, step_fn, out, steps, device, rng, binding)
    finally:
        C1.SAVES = old
        if dl is not None: dl.close()
    n = max(st['n'], 1)
    receipt.update({'stage': 'student', 'unit': unit, 'consumer': consumer, 'head_seed': int(head_seed), 'sampler_seed': int(sampler_seed), 'relation_sha256': sha256(relation_path),
                    'cost': {'mean_pair_scores_per_update': st['pairs']/n, 'complete_build_cpu_seconds': st['build_s'], 'synchronized_train_seconds': receipt['synchronized_train_seconds_this_attempt'],
                             'peak_gpu_memory_bytes': int(torch.cuda.max_memory_allocated()) if str(device).startswith('cuda') else None},
                    'same_draw_receipt': {'edge_hits_sha256': hashlib.sha256(edge_hits.astype(np.int64).tobytes()).hexdigest()},
                    'checkpoint_sha256': sha256(out/f'step_{steps:06d}.pt'), 'determinism': det, 'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None})
    write_json(out/'DONE.json', receipt)
    return receipt
