"""CR6-A1 source replicate: CR1 estimator fit cloned with ONLY the two training RNG roles changed.

RNG roles in the CR1 production estimator path (cr1_prod.fit_estimator + package src/cr1), audited by reading the code:
  R1 estimator weight init   : GlobalScorer(..., seed) -> torch.manual_seed(seed) inside fork_rng(devices=[]) (CPU)   CHANGED
  R2 training batch sampler  : numpy Generator, rng.integers(0, len(rows), 64) per step                              CHANGED
  R3 FIT CAL pools           : cr1.pipeline.pools(seed=20261104051+e)   (calibration pair pool, endpoint-excluded)   FIXED
  R4 folds / top32 candidates: content-hash folds, deterministic top32 (no RNG)                                       FIXED
  R5 student common init CAL : pools(seed=20261104151) in PREP (not used by estimators)                               FIXED
  R6 torch global CPU/CUDA RNG: not consumed by estimator training (no dropout, deterministic P/Q weights); asserted
     unchanged across training in every CR6 fit receipt.
  R7 score export / CAP solve / KEEP_TOP1 / J_MASS_OBS: deterministic, no RNG.
CR1 used R1 = 1001 and R2 = default_rng([1001, e, 51]) for BOTH objectives. CR6 uses, per expert index e (E01..E23 -> 0..5),
R1 = 2002000+e and R2 = default_rng(20261105200200+e), identical for J and BCE of the same expert (package MATRIX_NEW.csv).
With R1=1001, R2=[1001,e,51] this function replays CR1 (acceptance compares the step-250 checkpoint bitwise).
Everything else (allowed records, objective, model, a/b init, optimizer, 8000 steps, CAL pools, calibration fit) is the CR1 code.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import cr6_common as K                                                    # noqa: E402
import cr1_prod as C1                                                     # noqa: E402  (CR1 site driver, read-only)
from cr1.io import array_digest                                            # noqa: E402
from cr1.folds import EXCLUSIONS, training_mask                            # noqa: E402
from cr1.models import GlobalScorer                                        # noqa: E402
from cr1.objectives import estimator_loss, j_value, pq_weights             # noqa: E402
from cr1.calibration import fit_affine, apply_affine                       # noqa: E402
from cr1.pipeline import pools, pair_values, features                      # noqa: E402

NEW_SEEDS = {k: dict(head_seed=2002000+i, sampler_seed=20261105200200+i) for i, k in enumerate(K.EXPERTS)}


def fit(bank_path, prepared, objective, expert, out, device, head_seed, sampler_spec, steps=8000, saves=None, unit=None):
    det = C1.setup_determinism(device)
    bank, prep, graph = C1.get_prepared(bank_path, prepared)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if (out/'DONE.json').exists(): raise FileExistsError('Completed estimator exists; do not overwrite')
    e = K.EXPERTS.index(expert); exc = EXCLUSIONS[e]; key = f'E{exc[0]}{exc[1]}'
    assert key == expert
    allowed = training_mask(bank.observed_image_for_text, np.arange(len(bank.text_global)), graph['image_folds'], graph['text_folds'], exc)
    rows = np.flatnonzero(allowed)
    if array_digest(rows) != prep['experts'][key]['allowed_record_hash']: raise ValueError('Fold binding mismatch')
    assert not np.isin(graph['image_folds'][bank.observed_image_for_text[rows]], exc).any()
    assert not np.isin(graph['text_folds'][rows], exc).any()
    cal = prep['experts'][key]['base_calibration']; X, Y = features(bank, device)
    m = GlobalScorer(bank.dim, cal['a'], cal['b'], int(head_seed)).to(device)       # R1
    rng = np.random.default_rng(sampler_spec)                                        # R2
    pp = pools(bank, allowed, seed=20261104051+e, n=32768)                           # R3 (fixed)
    for ir, tr in [(pp[0], pp[1]), (pp[2], pp[3])]:
        assert not np.isin(graph['image_folds'][ir], exc).any() and not np.isin(graph['text_folds'][tr], exc).any()
    m.eval(); j0 = C1.jbce_on_pools(pair_values(m, X, Y, pp[0], pp[1]), pair_values(m, X, Y, pp[2], pp[3])); m.train()
    cpu0 = torch.get_rng_state().clone(); cuda0 = [s.clone() for s in torch.cuda.get_rng_state_all()] if str(device).startswith('cuda') else None

    def step_fn(model):   # == cr1_prod.fit_estimator.step_fn
        tx = rows[rng.integers(0, len(rows), 64)]; im = bank.observed_image_for_text[tx]
        it = torch.as_tensor(im, device=X.device); tt = torch.as_tensor(tx, device=X.device)
        g = model.matrix(X[it], Y[tt])
        loss = estimator_loss(objective, g, it)
        with torch.no_grad():
            p, q = pq_weights(64, device=g.device, dtype=g.dtype); t = torch.tanh(g/2)
            info = {'J_batch': float(j_value(g, p, q)), 'sat_T': float((t.abs() > .99).float().mean()),
                    'unique_images': int(len(np.unique(im))), 'unique_texts': int(len(np.unique(tx)))}
        return loss, 4096, info
    seed_repr = sampler_spec if isinstance(sampler_spec, int) else list(sampler_spec)
    binding = {'unit': unit, 'protocol': K.PROTOCOL_ID, 'bank_sha256': bank.digest, 'prep_sha256': K.sha256(Path(prepared)/'PREP.json'),
               'objective': objective, 'excluded': list(exc), 'init': cal, 'head_seed': int(head_seed), 'sampler_seed': seed_repr, 'steps': steps}
    old = C1.SAVES
    if saves is not None: C1.SAVES = saves
    try: receipt = C1.optim_loop(m, step_fn, out, steps, device, rng, binding)
    finally: C1.SAVES = old
    rng_untouched = bool(torch.equal(cpu0, torch.get_rng_state())) and (cuda0 is None or all(torch.equal(a, b) for a, b in zip(cuda0, torch.cuda.get_rng_state_all())))
    if steps != 8000:   # acceptance replay only
        receipt.update(torch_global_rng_untouched=rng_untouched); K.write_json(out/'REPLAY.json', receipt); return receipt
    m.eval(); tcal = time.perf_counter()
    gp, gq = pair_values(m, X, Y, pp[0], pp[1]), pair_values(m, X, Y, pp[2], pp[3])
    final_cal = fit_affine(gp, gq)          # same algorithm, same CAL pool as CR1
    receipt.update({'stage': 'estimator', 'unit': unit, 'objective': objective, 'excluded': list(exc), 'seed': int(head_seed),
                    'head_seed': int(head_seed), 'sampler_seed': seed_repr, 'source_set': 'NEW_SRC2002', 'source_seed': 2002,
                    'bank_sha256': bank.digest, 'allowed_record_hash': array_digest(rows), 'allowed_records': int(len(rows)),
                    'calibration': final_cal, 'checkpoint': 'step_008000.pt', 'checkpoint_sha256': K.sha256(out/'step_008000.pt'),
                    'cal_pools': {'seed': 20261104051+e, 'n': 32768, 'endpoint_folds_excluded_checked': True},
                    'J_raw_step0': j0, 'J_raw_step8000': C1.jbce_on_pools(gp, gq),
                    'J_calibrated_step8000': C1.jbce_on_pools(apply_affine(gp, final_cal), apply_affine(gq, final_cal)),
                    'calibration_seconds': time.perf_counter()-tcal, 'determinism': det, 'torch_global_rng_untouched': rng_untouched,
                    'gpu': torch.cuda.get_device_name() if str(device).startswith('cuda') else None,
                    'score_label': 'calibrated mechanism log-odds, NOT probability of correct correspondence'})
    K.write_json(out/'DONE.json', receipt)
    return receipt


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--objective', choices=['J', 'BCE'], required=True); ap.add_argument('--expert', choices=K.EXPERTS, required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--device', default='cuda'); a = ap.parse_args()
    K.frozen_check(); s = NEW_SEEDS[a.expert]
    fit(K.bank_path('Flickr60'), K.prep_dir('Flickr60'), a.objective, a.expert, a.out, a.device, s['head_seed'], s['sampler_seed'],
        unit=f'CR6_A1_Flickr60_{a.objective}_{a.expert}_src2002')


if __name__ == '__main__': main()
