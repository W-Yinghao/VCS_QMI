"""Predict/evaluate helpers that mirror package tools/run_readouts.py exactly.

Used for registered readouts the package driver does not cover (profile/anchor models,
CC target galleries, wrong-pi sensitivity, oracle comparators). predict_* never opens
label files; evaluate_* never fits a model.
"""
from pathlib import Path
import numpy as np
from common import dump, load, sha
from cdr1.io import load_gallery_scores
from cdr1.calibrate import Calibrator
from cdr1.math_core import post_correct_log_ratio, mechanism_probability
from cdr1.retrieval import product_to_nonmatch_log_ratio, gallery_posteriors
from cdr1.metrics import prior_weights, selective_metrics, binary_metrics


def predict_one(model_path, scores_path, beta, pi_used, post_rho, pred_path, extra=None):
    pred_path = Path(pred_path)
    prov = {'model_sha256': sha(model_path), 'scores_sha256': sha(scores_path),
            'pi_used': pi_used, 'beta': beta, 'post_rho': post_rho}
    side = Path(str(pred_path) + '.json')
    if pred_path.exists():
        if load(side).get('provenance') != prov:
            raise RuntimeError(f'prediction binding mismatch {pred_path}')
        return 'REUSED'
    m = Calibrator.from_dict(load(model_path))
    s, qid, gid = load_gallery_scores(scores_path)
    ell = m.clean_log_ratio(s)
    bad = np.zeros_like(s, bool)
    if post_rho not in ('', None):
        if m.rho != 0:
            raise ValueError('post correction only on naive observed ratio')
        ell, bad = post_correct_log_ratio(ell, float(post_rho))
    bg, badbg = product_to_nonmatch_log_ratio(ell, beta)
    p = gallery_posteriors(bg, pi_used, base_scores=s)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pred_path, query_ids=qid, gallery_ids=gid, top_index=p['top_index'],
                        p_none=p['p_none'], p_present=p['p_present'], p_best_correct=p['p_best_correct'],
                        map_return=p['map_return'],
                        naive_top1_mechanism=mechanism_probability(ell[np.arange(len(s)), p['top_index']], .5))
    sc = {'provenance': prov, 'post_projection_fraction': float(bad.mean()),
          'background_projection_fraction': float(badbg.mean()), 'label_files_read': False}
    if extra:
        sc.update(extra)
    dump(side, sc)
    return 'PREDICTED'


def load_pred_labels(pred_path, labels_path):
    with np.load(pred_path, allow_pickle=False) as z:
        p = {k: z[k] for k in z.files}
    with np.load(labels_path, allow_pickle=False) as z:
        lab = {k: z[k] for k in z.files}
    if not np.array_equal(p['query_ids'], lab['query_ids']):
        raise RuntimeError('ID join failed')
    present = lab['true_index'] >= 0
    correct = present & (p['top_index'] == lab['true_index'])
    return p, lab, present, correct


def evaluate_one(pred_path, labels_path, pi_target, pi_used, rid):
    p, lab, present, correct = load_pred_labels(pred_path, labels_path)
    w = prior_weights(present, pi_target)
    return {'id': rid, 'prediction_sha256': sha(pred_path), 'n': int(len(present)),
            'n_present': int(present.sum()), 'n_absent': int((~present).sum()),
            'pi_used': pi_used, 'pi_target': pi_target,
            'selected': selective_metrics(p['p_best_correct'], correct, present, weights=w),
            'naive_top1_mechanism': selective_metrics(p['naive_top1_mechanism'], correct, present, weights=w),
            'presence': binary_metrics(p['p_present'], present, w),
            'map_action_accuracy': float(np.sum(w * np.where(p['map_return'], correct, ~present)))}


def decision_metrics(accept, correct, present, w, abst_cost):
    """Operating point for a hard accept rule (same formulas as metrics.selective_metrics)."""
    accept = np.asarray(accept, bool)
    w = np.asarray(w, float); w = w / w.sum()
    cov = float(w[accept].sum()); nullmass = float(w[~present].sum())
    return {'coverage': cov,
            'accepted_error_rate': float(np.sum(w[accept] * (~correct[accept])) / cov) if cov else None,
            'accepted_correct_mass': float(np.sum(w[accept] * correct[accept])),
            'absent_false_accept_rate': float(w[accept & ~present].sum() / nullmass) if nullmass else None,
            'constant_abstention_cost': float(np.sum(w[accept] * (~correct[accept])) + abst_cost * w[~accept].sum()),
            'abstention_cost_used': abst_cost}
