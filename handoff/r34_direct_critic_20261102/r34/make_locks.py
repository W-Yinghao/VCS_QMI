"""CPU: build the locked evaluation-only DEV-observed maps and the fixed FIT-observed-J record subset,
then write the 12 unit configs + LOCK.json. Reads no data files (only lengths)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CONDS, OUT, ROOT, N_TRAIN_CAP, N_DEV_CAP, sha256, write_json, utc

DEV_NOISE_SEED = 20261102
FIT_EVAL_SEED = 20261103
FIT_EVAL_N = 14500


def official(length, im_div, ratio, rs):
    """IN2R data.py / R25 tools/make_noise.py mechanism (verbatim logic): shuffle caption ids, take int(ratio*N),
    permute their image ids among themselves (fixed points possible)."""
    t2i = np.arange(0, length) // im_div
    idx = np.arange(length)
    rs.shuffle(idx)
    n = int(ratio * length)
    sh = t2i[idx[:n]]
    rs.shuffle(sh)
    t2i[idx[:n]] = sh
    return t2i


UNIT_NAMES = {'D': 'D_dual_cosine', 'X': 'X_joint_mlp'}
OBJ_NAMES = {'J': 'J', 'BCE': 'BCE', 'NCE': 'InfoNCE'}


def main():
    ld = OUT / 'locks'; ld.mkdir(parents=True, exist_ok=True)
    locks = {'created_utc': utc(), 'dev_observed_maps': {}, 'fit_eval_subset': {}}
    clean = np.arange(N_DEV_CAP) // 5
    for cond, c in CONDS.items():
        m = official(N_DEV_CAP, 5, c['noise_ratio'], np.random.RandomState(DEV_NOISE_SEED)) if c['noise_ratio'] else clean.copy()
        p = ld / f'dev_observed_map_{cond}.npy'
        if p.exists():
            assert np.array_equal(np.load(p), m), 'existing lock differs'
        else:
            np.save(p, m)
        locks['dev_observed_maps'][cond] = dict(path=str(p), sha256=sha256(p), ratio=c['noise_ratio'], seed=DEV_NOISE_SEED,
                                                 mechanism='IN2R data.py subset caption->image permutation within DEV (1014 images x5)',
                                                 realized_mismatch=float((m != clean).mean()),
                                                 image_counts_preserved=bool(np.array_equal(np.bincount(m, minlength=1014), np.bincount(clean, minlength=1014))),
                                                 use='evaluation-only DEV-observed J; never batches/optimizer/selection')
    sub = np.sort(np.random.default_rng(FIT_EVAL_SEED).choice(N_TRAIN_CAP, FIT_EVAL_N, replace=False)).astype(np.int64)
    p = ld / 'fit_eval_subset.npy'
    if p.exists():
        assert np.array_equal(np.load(p), sub)
    else:
        np.save(p, sub)
    locks['fit_eval_subset'] = dict(path=str(p), sha256=sha256(p), n=FIT_EVAL_N, seed=FIT_EVAL_SEED,
                                    rule='uniform without replacement over the 145000 FIT caption records; observed map of the condition',
                                    use='FIT-observed J readout (same record U-statistic as training P/Q)')
    write_json(ld / 'LOCKS.json', locks)

    common_cfg = dict(seed=1001, steps=2000, batch_size=128,
                      optimizer=dict(name='AdamW', lr=1e-3, weight_decay=0.0, betas=[0.9, 0.999], eps=1e-8, amsgrad=False),
                      grad_clip=0.0, eval_every=250, primary_checkpoint='step_2000',
                      sampling='IID caption records with replacement: numpy default_rng(seed).integers(0,145000,128) per step; image = observed_map[record]',
                      P='record diagonal, weight 1/B', Q='off-diagonal record draws, weight 1/(B(B-1)); same-image retained (R32 F1)',
                      NCE='bidirectional CE on g, diagonal positives, all other in-batch draws as candidates (no masking)',
                      init='torch.manual_seed(seed) then build(structure); identical across objectives within a structure',
                      peers='shared critic params; loss = 0.5*(loss_peerA + loss_peerB) on the same records; eval g per peer then mean; no cross-peer products',
                      standardization='per peer, FIT-only (cache standardization_peer*.npz)',
                      precision='float32, matmul tf32 off (torch default), no AMP',
                      rank_score='peer-mean g (float64 of per-peer float32 g); upstream IN2R i2t/t2i argsort; tie-bounded values also reported',
                      test_read=False)
    cfgs = []
    for cond in CONDS:
        for s in ('D', 'X'):
            for o in ('J', 'BCE', 'NCE'):
                uid = f'direct_{cond}_s1001_{UNIT_NAMES[s]}_{OBJ_NAMES[o]}'
                cfg = dict(unit_id=uid, cond=cond, noise_percent=CONDS[cond]['noise_percent'], structure=s, objective=o,
                           encoder_source=CONDS[cond]['h_unit'], encoder_frozen=True,
                           structure_spec=('a*cos(Wx hx, Wy hy)+b, proj 256, a=2s,b=-2so, s init 2, o init 0.5' if s == 'D'
                                           else 'MLP([hx,hy]) 2048-256-GELU-256-GELU-1, blockwise first layer'),
                           **common_cfg)
                write_json(ROOT / 'configs' / f'{uid}.json', cfg); cfgs.append(cfg)
    write_json(ROOT / 'configs' / 'UNITS.json', [c['unit_id'] for c in cfgs])
    print(json.dumps(locks, indent=1)); print(len(cfgs), 'configs')


if __name__ == '__main__':
    main()
