"""CR1 production driver (site). Mathematics is the package reference (src/cr1, byte copy).

Differences vs package src/cr1/pipeline.py (recorded in SITE_DIFF.md):
  * deterministic torch algorithms, TF32 off, CUBLAS workspace pinned;
  * every registered step checkpoint carries model+optimizer+scheduler+numpy/torch RNG state and a
    binding hash; an interrupted unit resumes exactly from its latest checkpoint (DONE never overwritten);
  * every formal command checks PROTOCOL_FROZEN_CR1.json (code/config/bank/prep hashes) first;
  * extra FIT-only receipts: raw/calibrated J+BCE on the registered CAL pools, per-step batch J,
    endpoint-exclusion audits, top-32 tie counts, per-expert score quantiles, student exposure counts.
No command accepts truth, noise flags, clean IDs or DEV data.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))  # python -I: explicit, absolute
from cr1.io import Bank, write_json, read_json, array_digest, sha256
from cr1.folds import EXCLUSIONS, content_folds, edge_experts, training_mask
from cr1.models import GlobalScorer
from cr1.objectives import estimator_loss, soft_nce, j_value, pq_weights
from cr1.calibration import fit_affine, apply_affine
from cr1.relations import Relation
from cr1.transport import solve
from cr1.pipeline import pools, pair_values, features, sync  # unchanged package helpers

SAVES = (0, 250, 500, 1000, 2000, 4000, 8000)
TREE = Path(__file__).resolve().parents[2]
FROZEN = TREE/'PROTOCOL_FROZEN_CR1.json'
UNITS = TREE/'package'/'configs'/'units'


def setup_determinism(device):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return dict(deterministic=True, matmul_tf32=False, cudnn_tf32=False, torch=torch.__version__,
                gpu=torch.cuda.get_device_name() if str(device).startswith('cuda') else None)


def frozen_check(unit_id=None, extra_files=()):
    """Formal units refuse to run unless every frozen hash still matches."""
    if not FROZEN.exists(): raise RuntimeError('PROTOCOL_FROZEN_CR1.json missing: formal units cannot run')
    fz = read_json(FROZEN)
    for rel, h in fz['code_sha256'].items():
        if sha256(TREE/rel) != h: raise RuntimeError(f'Frozen code changed: {rel}')
    if unit_id is not None:
        if unit_id not in fz['unit_config_sha256']: raise RuntimeError(f'Unit not registered: {unit_id}')
        if sha256(UNITS/f'{unit_id}.json') != fz['unit_config_sha256'][unit_id]: raise RuntimeError('Unit config changed')
    for p in extra_files:
        key = str(p)
        if key not in fz['data_sha256']: raise RuntimeError(f'Data file not frozen: {key}')
        if sha256(p) != fz['data_sha256'][key]: raise RuntimeError(f'Frozen data changed: {key}')
    return fz['frozen_sha256'] if 'frozen_sha256' in fz else sha256(FROZEN)


def unit_config(unit_id):
    return read_json(UNITS/f'{unit_id}.json')


# ----------------------------------------------------------------------------- prepare
@torch.no_grad()
def top_candidates_with_ties(X, Y, k=32, chunk=128):
    """Identical selection to package pipeline.top_candidates; additionally counts boundary ties."""
    if k > len(X): raise ValueError('Formal graph needs >= 32 supported images')
    out = np.empty((len(Y), k), np.int64); ties = 0; tied_rows = 0
    xd = F.normalize(X.double(), dim=1)
    for start in range(0, len(Y), chunk):
        s = (F.normalize(Y[start:start+chunk].double(), dim=1) @ xd.T).cpu().numpy()
        for j, row in enumerate(s):
            idx = np.argpartition(-row, k-1)[:k]
            threshold = row[idx].min()
            idx = np.flatnonzero(row >= threshold)
            if len(idx) > k: tied_rows += 1; ties += len(idx) - k
            idx = idx[np.lexsort((idx, -row[idx]))][:k]
            out[start+j] = idx
    return out, dict(rows_with_boundary_ties=tied_rows, extra_tied_items=ties,
                     tie_rule='score desc, then gallery index asc (stable, label-blind)')


def prepare(args):
    det = setup_determinism(args.device)
    bank = Bank(args.bank, formal=True); out = Path(args.out)
    if (out/'PREP.json').exists(): raise FileExistsError('PREP exists; do not overwrite')
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    X, Y = features(bank, args.device)
    fi = content_folds(list(bank.image_content_key), 'image'); ft = content_folds(list(bank.text_content_key), 'text')
    sync(args.device); tc = time.perf_counter()
    cand, ties = top_candidates_with_ties(X, Y, k=32)
    sync(args.device); tc = time.perf_counter() - tc
    np.savez_compressed(out/'graph.npz', candidates=cand, image_folds=fi, text_folds=ft)
    reports = {}
    for e, exc in enumerate(EXCLUSIONS):
        allowed = training_mask(bank.observed_image_for_text, np.arange(len(Y)), fi, ft, exc)
        rows = np.flatnonzero(allowed)
        np.save(out/f'allowed_E{exc[0]}{exc[1]}.npy', rows)
        pp = pools(bank, allowed, seed=20261104051+e, n=32768)
        a = fit_affine(pair_values(None, X, Y, pp[0], pp[1]), pair_values(None, X, Y, pp[2], pp[3]), start=(50., -5.))
        reports[f'E{exc[0]}{exc[1]}'] = {'excluded': list(exc), 'records': int(allowed.sum()),
            'record_fraction': float(allowed.mean()),
            'distinct_images': int(len(np.unique(bank.observed_image_for_text[rows]))),
            'allowed_record_hash': array_digest(rows), 'allowed_file_sha256': sha256(out/f'allowed_E{exc[0]}{exc[1]}.npy'),
            'base_calibration': a,
            'calibration_note': 'CAL pairs use allowed FIT endpoints only; not independent validation data'}
    pp = pools(bank, np.ones(len(Y), bool), seed=20261104151, n=32768)
    common = fit_affine(pair_values(None, X, Y, pp[0], pp[1]), pair_values(None, X, Y, pp[2], pp[3]), start=(50., -5.))
    obs_in = (cand == bank.observed_image_for_text[:, None]).any(1)
    write_json(out/'PREP.json', {'bank_sha256': bank.digest, 'graph_sha256': sha256(out/'graph.npz'),
        'condition': bank.binding['condition'], 'experts': reports, 'student_common_initialization': common,
        'fold_image_counts': np.bincount(fi, minlength=4).tolist(), 'fold_text_counts': np.bincount(ft, minlength=4).tolist(),
        'candidate_K': 32, 'candidate_ties': ties, 'n_text': int(len(Y)), 'n_image': int(len(X)),
        'observed_partner_in_top32_fraction_label_stat': float(obs_in.mean()),
        'scope': 'FIT only; candidate generation read no observed relation labels (observed-in-top32 is a post-hoc FIT statistic)',
        'determinism': det, 'seconds': {'total': time.perf_counter()-t0, 'top32': tc}})


def get_prepared(bank_path, prepared, device=None):
    bank = Bank(bank_path, formal=True); root = Path(prepared); prep = read_json(root/'PREP.json')
    if prep['bank_sha256'] != bank.digest or prep['graph_sha256'] != sha256(root/'graph.npz'):
        raise ValueError('Input manifest changed')
    with np.load(root/'graph.npz', allow_pickle=False) as z: graph = {k: z[k] for k in z.files}
    return bank, prep, graph


# ----------------------------------------------------------------------------- training loop
def _binding_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def optim_loop(model, step_fn, out, steps, device, rng, binding, log_extra=None):
    """Package optim_loop + exact resume. Same AdamW/CosineAnnealingLR/clip as the package."""
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)
    saves = sorted({s for s in SAVES if s <= steps} | {0, steps}); bsha = _binding_sha(binding)

    def save(step):
        p = out/f'step_{step:06d}.pt'; tmp = p.with_suffix('.tmp')
        torch.save({'model': model.state_dict(), 'step': step, 'dimension': model.global_residual[0].in_features//4,
                    'architecture': 'G', 'optimizer': opt.state_dict(), 'scheduler': sched.state_dict(),
                    'rng_numpy': rng.bit_generator.state, 'rng_torch': torch.get_rng_state(),
                    'rng_cuda': torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else None,
                    'binding': binding, 'binding_sha256': bsha}, tmp)
        tmp.replace(p)

    existing = sorted(out.glob('step_*.pt')); start = 0; resumed_from = None
    if existing:
        ck = torch.load(existing[-1], map_location=device, weights_only=False)
        if ck['binding_sha256'] != bsha: raise ValueError('Resume binding mismatch')
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); sched.load_state_dict(ck['scheduler'])
        rng.bit_generator.state = ck['rng_numpy']; torch.set_rng_state(ck['rng_torch'].cpu())
        if ck['rng_cuda'] is not None: torch.cuda.set_rng_state_all([s.cpu() for s in ck['rng_cuda']])
        start = ck['step']; resumed_from = start
    else:
        save(0)
    log = out/'TRAIN.jsonl'
    if log.exists():  # keep only records up to the resumed step
        keep = [l for l in log.read_text().splitlines() if l.strip() and json.loads(l)['step'] <= start]
        log.write_text(''.join(k+'\n' for k in keep))
    scores = 0; extra = {}
    sync(device); t0 = time.perf_counter()
    with open(log, 'a') as f:
        for step in range(start+1, steps+1):
            opt.zero_grad(set_to_none=True)
            loss, npairs, info = step_fn(model)
            if not torch.isfinite(loss): raise FloatingPointError(f'Non-finite loss at step {step}')
            loss.backward(); grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            opt.step(); sched.step(); scores += npairs
            if log_extra: log_extra(info)
            if step == 1 or step % 100 == 0 or step == steps:
                rec = {'step': step, 'loss': float(loss.detach()), 'grad_norm': float(grad), 'lr': sched.get_last_lr()[0]}
                rec.update({k: v for k, v in info.items() if not isinstance(v, np.ndarray)})
                f.write(json.dumps(rec)+'\n'); f.flush()
            if step in saves: save(step)
    sync(device)
    return {'updates': steps, 'updates_this_attempt': steps-start, 'resumed_from_step': resumed_from,
            'pair_scores_this_attempt': scores, 'synchronized_train_seconds_this_attempt': time.perf_counter()-t0,
            'trainable_parameters': sum(p.numel() for p in model.parameters()), 'backbone_updated': False,
            'binding_sha256': bsha, 'saved_steps': saves}


def jbce_on_pools(gp, gq):
    tp, tq = np.tanh(gp/2), np.tanh(gq/2)
    return {'J': float(np.mean(tp-tp*tp/2) - np.mean(tq+tq*tq/2)),
            'BCE': float(np.logaddexp(0, -gp).mean() + np.logaddexp(0, gq).mean()),
            'mean_P': float(gp.mean()), 'mean_Q': float(gq.mean()),
            'saturated_T_fraction': float(np.mean(np.abs(np.concatenate([tp, tq])) > .99))}


def fit_estimator(args):
    cfg = unit_config(args.unit); det = setup_determinism(args.device)
    fsha = frozen_check(args.unit, [args.bank, Path(args.prepared)/'PREP.json', Path(args.prepared)/'graph.npz'])
    bank, prep, graph = get_prepared(args.bank, args.prepared)
    if bank.binding['condition'] != cfg['condition']: raise ValueError('Condition mismatch')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if (out/'DONE.json').exists(): raise FileExistsError('Completed estimator exists; do not overwrite')
    exc = tuple(cfg['excluded_folds']); key = f'E{exc[0]}{exc[1]}'; e = EXCLUSIONS.index(exc); objective = cfg['objective']
    allowed = training_mask(bank.observed_image_for_text, np.arange(len(bank.text_global)), graph['image_folds'], graph['text_folds'], exc)
    rows = np.flatnonzero(allowed)
    if array_digest(rows) != prep['experts'][key]['allowed_record_hash']: raise ValueError('Fold binding mismatch')
    # endpoint audit: no training/CAL record touches an excluded fold on either side
    assert not np.isin(graph['image_folds'][bank.observed_image_for_text[rows]], exc).any()
    assert not np.isin(graph['text_folds'][rows], exc).any()
    cal = prep['experts'][key]['base_calibration']; X, Y = features(bank, args.device)
    m = GlobalScorer(bank.dim, cal['a'], cal['b'], 1001).to(args.device); rng = np.random.default_rng([1001, e, 51])
    pp = pools(bank, allowed, seed=20261104051+e, n=32768)
    for ir, tr in [(pp[0], pp[1]), (pp[2], pp[3])]:
        assert not np.isin(graph['image_folds'][ir], exc).any() and not np.isin(graph['text_folds'][tr], exc).any()
    m.eval(); j0 = jbce_on_pools(pair_values(m, X, Y, pp[0], pp[1]), pair_values(m, X, Y, pp[2], pp[3])); m.train()

    def step_fn(model):
        tx = rows[rng.integers(0, len(rows), 64)]; im = bank.observed_image_for_text[tx]
        it = torch.as_tensor(im, device=X.device); tt = torch.as_tensor(tx, device=X.device)
        g = model.matrix(X[it], Y[tt])
        loss = estimator_loss(objective, g, it)
        with torch.no_grad():
            p, q = pq_weights(64, device=g.device, dtype=g.dtype); t = torch.tanh(g/2)
            info = {'J_batch': float(j_value(g, p, q)), 'sat_T': float((t.abs() > .99).float().mean()),
                    'unique_images': int(len(np.unique(im))), 'unique_texts': int(len(np.unique(tx)))}
        return loss, 4096, info
    binding = {'unit': args.unit, 'config_sha256': sha256(UNITS/f'{args.unit}.json'), 'bank_sha256': bank.digest,
               'prep_sha256': sha256(Path(args.prepared)/'PREP.json'), 'objective': objective, 'excluded': list(exc),
               'init': cal, 'seed': 1001, 'stream': [1001, e, 51], 'steps': 8000}
    receipt = optim_loop(m, step_fn, out, 8000, args.device, rng, binding)
    m.eval(); tcal = time.perf_counter()
    gp, gq = pair_values(m, X, Y, pp[0], pp[1]), pair_values(m, X, Y, pp[2], pp[3])
    final_cal = fit_affine(gp, gq)
    receipt.update({'stage': 'estimator', 'unit': args.unit, 'objective': objective, 'excluded': list(exc), 'seed': 1001,
                    'bank_sha256': bank.digest, 'allowed_record_hash': array_digest(rows), 'allowed_records': int(len(rows)),
                    'calibration': final_cal, 'checkpoint': 'step_008000.pt', 'checkpoint_sha256': sha256(out/'step_008000.pt'),
                    'cal_pools': {'seed': 20261104051+e, 'n': 32768, 'endpoint_folds_excluded_checked': True},
                    'J_raw_step0': j0, 'J_raw_step8000': jbce_on_pools(gp, gq),
                    'J_calibrated_step8000': jbce_on_pools(apply_affine(gp, final_cal), apply_affine(gq, final_cal)),
                    'calibration_seconds': time.perf_counter()-tcal, 'determinism': det, 'frozen_protocol': fsha,
                    'score_label': 'calibrated mechanism log-odds, NOT probability of correct correspondence'})
    write_json(out/'DONE.json', receipt)


def score(args):
    det = setup_determinism(args.device)
    fsha = frozen_check(None, [args.bank, Path(args.prepared)/'PREP.json', Path(args.prepared)/'graph.npz'])
    bank, prep, graph = get_prepared(args.bank, args.prepared); X, Y = features(bank, args.device)
    p = Path(args.out)
    if p.exists(): raise FileExistsError('Score export exists; do not overwrite')
    c = graph['candidates']; tr = np.repeat(np.arange(len(c)), c.shape[1]); ir = c.ravel()
    owners = edge_experts(graph['image_folds'], graph['text_folds'], tr, ir)
    scores = np.empty(len(ir), np.float64); lineage = {}; t0 = time.perf_counter(); raw_all = np.empty(len(ir), np.float64)
    for e, exc in enumerate(EXCLUSIONS):
        key = f'E{exc[0]}{exc[1]}'; mask = owners == e
        # endpoint honesty: routed edges have BOTH endpoints inside the expert's excluded folds
        assert np.isin(graph['image_folds'][ir[mask]], exc).all() and np.isin(graph['text_folds'][tr[mask]], exc).all()
        if args.source == 'BASE':
            model = None; cal = prep['experts'][key]['base_calibration']; lineage[key] = {'base': True, 'calibration': cal}
        else:
            folder = Path(args.models_root)/args.source/key; receipt = read_json(folder/'DONE.json')
            if receipt['bank_sha256'] != bank.digest or receipt['excluded'] != list(exc) or receipt['objective'] != args.source:
                raise ValueError('Expert provenance mismatch')
            if receipt['allowed_record_hash'] != prep['experts'][key]['allowed_record_hash']:
                raise ValueError('Expert training membership mismatch')
            if sha256(folder/receipt['checkpoint']) != receipt['checkpoint_sha256']: raise ValueError('Checkpoint hash')
            model = GlobalScorer(bank.dim).to(args.device)
            model.load_state_dict(torch.load(folder/receipt['checkpoint'], map_location=args.device, weights_only=False)['model']); model.eval()
            cal = receipt['calibration']
            lineage[key] = {'unit': receipt['unit'], 'checkpoint_sha256': receipt['checkpoint_sha256'],
                            'calibration': cal, 'calibration_sha256': _binding_sha(cal)}
        raw = pair_values(model, X, Y, ir[mask], tr[mask]); raw_all[mask] = raw
        scores[mask] = apply_affine(raw, cal)
        q = np.quantile(scores[mask], [0, .01, .1, .25, .5, .75, .9, .99, 1]).tolist()
        lineage[key].update({'edges': int(mask.sum()), 'calibrated_quantiles_0_1_10_25_50_75_90_99_100': q,
                             'raw_quantiles_0_50_100': np.quantile(raw, [0, .5, 1]).tolist()})
    sync(args.device)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, scores=scores.reshape(c.shape), candidates=c)
    write_json(str(p)+'.json', {'bank_sha256': bank.digest, 'source': args.source, 'lineage': lineage,
        'score_sha256': sha256(p), 'endpoint_exclusion': 'both endpoints excluded from fit/calibration (asserted per edge)',
        'edges_per_expert': np.bincount(owners, minlength=6).tolist(), 'seconds': time.perf_counter()-t0,
        'determinism': det, 'frozen_protocol': fsha,
        'note': 'six OOF functions stitched; NOT a single J certificate'})


def match(args):
    fsha = frozen_check(None, [args.bank])
    bank = Bank(args.bank, formal=True); p = Path(args.out)
    if p.exists(): raise FileExistsError('Coupling exists; do not overwrite')
    z = np.load(args.scores, allow_pickle=False); meta = read_json(str(args.scores)+'.json')
    if meta['bank_sha256'] != bank.digest or meta['score_sha256'] != sha256(args.scores): raise ValueError('Score binding mismatch')
    t0 = time.process_time(); w0 = time.perf_counter()
    result = solve(z['candidates'], z['scores'], bank.p_image, capped=args.mode == 'CAP')
    secs = dict(cpu=time.process_time()-t0, wall=time.perf_counter()-w0)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, candidates=result.candidates, weights=result.weights, slack=result.slack,
                        prices=result.prices, loads=result.loads)
    cap = 2.0*len(bank.text_global)*bank.p_image/bank.p_image.sum()
    obs = bank.observed_image_for_text; on_obs = (result.candidates == obs[:, None])
    stats = {'active_capacity_fraction': float((result.prices > 1e-8).mean()),
             'saturated_capacity_fraction': float((result.loads >= cap*(1-1e-4)).mean()),
             'max_load': float(result.loads.max()), 'max_load_over_capacity': float((result.loads/cap).max()),
             'mean_slack': float(result.slack.mean()), 'slack_quantiles_10_50_90': np.quantile(result.slack, [.1, .5, .9]).tolist(),
             'mass_on_observed_edge': float((result.weights*on_obs).sum()/len(obs)),
             'mass_off_observed_edge_new': float((result.weights*~on_obs).sum()/len(obs)),
             'observed_mass_removed': float(1-(result.weights*on_obs).sum()/len(obs)),
             'rows_observed_in_candidates': float(on_obs.any(1).mean()),
             'top1_weight_is_observed': float((result.candidates[np.arange(len(obs)), result.weights.argmax(1)] == obs).mean())}
    write_json(str(p)+'.json', {'bank_sha256': bank.digest, 'source': meta['source'], 'mode': args.mode,
        'scores_sha256': meta['score_sha256'], 'coupling_sha256': sha256(p), 'solver': result.receipt, 'seconds': secs,
        'relation': Relation.graph(result.candidates, result.weights, len(bank.image_global)).summary(),
        'label_free_stats': stats, 'frozen_protocol': fsha})


class ExposedRelation(Relation):
    """Package Relation; batch() is byte-identical logic, and additionally returns sampled edges."""
    def batch_edges(self, rng, n_pairs=64):
        if n_pairs < 2: raise ValueError('n_pairs >= 2')
        e = np.searchsorted(self.cdf, rng.random(n_pairs), side='right')
        im, iw = np.unique(self.image_edges[e], return_counts=True)
        tx, tw = np.unique(self.text_edges[e], return_counts=True)
        m = self.matrix[tx][:, im].toarray().T
        if (m.sum(0) <= 0).any() or (m.sum(1) <= 0).any(): raise AssertionError('Sample lost a positive edge')
        return {'images': im, 'texts': tx, 'mass': m, 'image_weight': iw/n_pairs, 'text_weight': tw/n_pairs,
                'pair_count': int(len(im)*len(tx)), 'edges': e}


def train_student(args):
    cfg = unit_config(args.unit); det = setup_determinism(args.device)
    files = [args.bank, Path(args.prepared)/'PREP.json', Path(args.prepared)/'graph.npz']
    fsha = frozen_check(args.unit, files)
    bank, prep, graph = get_prepared(args.bank, args.prepared)
    if bank.binding['condition'] != cfg['condition']: raise ValueError('Condition mismatch')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if (out/'DONE.json').exists(): raise FileExistsError('Completed student exists; do not overwrite')
    arm = cfg['relation_arm']
    if arm == 'OBS':
        relation = ExposedRelation.observed(bank.observed_image_for_text, len(bank.image_global)); binding_rel = {'source': 'OBS'}
    else:
        rel = Path(args.relation)
        if rel.stem != arm: raise ValueError('Relation file does not match unit arm')
        z = np.load(rel, allow_pickle=False); binding_rel = read_json(str(rel)+'.json')
        if binding_rel['bank_sha256'] != bank.digest or binding_rel['coupling_sha256'] != sha256(rel): raise ValueError('Coupling binding mismatch')
        if f"{binding_rel['source']}_{binding_rel['mode']}" != arm: raise ValueError('Coupling arm mismatch')
        relation = ExposedRelation.graph(z['candidates'], z['weights'], len(bank.image_global))
    X, Y = features(bank, args.device); cal = prep['student_common_initialization']
    model = GlobalScorer(bank.dim, cal['a'], cal['b'], 1001).to(args.device)
    rng = np.random.default_rng(20261104201)
    edge_hits = np.zeros(len(relation.edge_mass), np.int64); stat = {'u_img': 0, 'u_txt': 0, 'pairs': 0, 'n': 0}

    def step_fn(m):
        b = relation.batch_edges(rng, 64); ix = torch.as_tensor(b['images'], device=X.device); tx = torch.as_tensor(b['texts'], device=X.device)
        s = m.matrix(X[ix], Y[tx]); mass = b['mass']/b['mass'].max()
        target = torch.as_tensor(mass, device=X.device, dtype=s.dtype)
        wi = torch.as_tensor(b['image_weight'], device=X.device, dtype=s.dtype)
        wt = torch.as_tensor(b['text_weight'], device=X.device, dtype=s.dtype)
        np.add.at(edge_hits, b['edges'], 1)
        info = {'unique_images': int(len(b['images'])), 'unique_texts': int(len(b['texts'])),
                'target_nnz': int((mass > 0).sum())}
        stat['u_img'] += info['unique_images']; stat['u_txt'] += info['unique_texts']; stat['pairs'] += b['pair_count']; stat['n'] += 1
        return soft_nce(s, target, wi, wt), b['pair_count'], info
    binding = {'unit': args.unit, 'config_sha256': sha256(UNITS/f'{args.unit}.json'), 'bank_sha256': bank.digest,
               'prep_sha256': sha256(Path(args.prepared)/'PREP.json'), 'arm': arm,
               'relation_sha256': binding_rel.get('coupling_sha256', 'OBS'), 'init': cal, 'seed': 1001,
               'sampler_seed': 20261104201, 'steps': 8000}
    receipt = optim_loop(model, step_fn, out, 8000, args.device, rng, binding)
    n = max(stat['n'], 1)
    exposure = {'scope': 'this attempt' if receipt['resumed_from_step'] else 'all 8000 updates',
                'sampled_pair_events': int(64*stat['n']), 'distinct_edges_sampled': int((edge_hits > 0).sum()),
                'graph_edges': int(len(relation.edge_mass)), 'distinct_edge_fraction': float((edge_hits > 0).mean()),
                'mean_unique_images_per_batch': stat['u_img']/n, 'mean_unique_texts_per_batch': stat['u_txt']/n,
                'mean_rectangle_pairs': stat['pairs']/n, 'repeat_rate_events': float(1-(edge_hits > 0).sum()/max(64*stat['n'], 1)),
                'max_edge_repeats': int(edge_hits.max())}
    np.save(out/'EDGE_EXPOSURE.npy', edge_hits.astype(np.int32))
    receipt.update({'stage': 'student', 'unit': args.unit, 'arm': arm, 'objective': 'SOFT_NCE', 'bank_sha256': bank.digest,
                    'relation_binding': {k: binding_rel[k] for k in binding_rel if k in ('source', 'mode', 'coupling_sha256', 'solver')},
                    'relation_summary': relation.summary(), 'exposure': exposure,
                    'checkpoint': 'step_008000.pt', 'checkpoint_sha256': sha256(out/'step_008000.pt'),
                    'determinism': det, 'frozen_protocol': fsha, 'source_estimator_not_used_at_inference': True,
                    'initialization': 'new G (seed 1001) + common FIT calibration; NOT estimator/NCE prefix'})
    write_json(out/'DONE.json', receipt)


def main():
    ap = argparse.ArgumentParser(description=__doc__); sp = ap.add_subparsers(dest='cmd', required=True)
    for name in ('prepare', 'fit-estimator', 'score', 'match', 'train-student'):
        p = sp.add_parser(name); p.add_argument('--bank', required=True); p.add_argument('--out', required=True)
        p.add_argument('--device', default='cuda')
        if name in ('fit-estimator', 'score', 'train-student'): p.add_argument('--prepared', required=True)
        if name in ('fit-estimator', 'train-student'): p.add_argument('--unit', required=True)
        if name == 'score':
            p.add_argument('--source', choices=['BASE', 'J', 'BCE', 'NCE'], required=True); p.add_argument('--models-root')
        if name == 'match': p.add_argument('--scores', required=True); p.add_argument('--mode', choices=['ROW', 'CAP'], required=True)
        if name == 'train-student': p.add_argument('--relation', required=True)
    a = ap.parse_args()
    {'prepare': prepare, 'fit-estimator': fit_estimator, 'score': score, 'match': match, 'train-student': train_student}[a.cmd](a)


if __name__ == '__main__': main()
