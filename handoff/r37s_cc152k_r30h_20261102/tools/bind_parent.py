"""R37S parent binding (CPU, read-only on the parent run dir, cache and prep) -> PARENT_BINDING.json: the R37S CC152K R30-style H parent
r37s_cc_s1001_H0 (warmup 5 + H0 co-train 40, A1, K5, i2t/t2i on, seed 1001). Verifies DONE.json, recomputes the best co-train epoch by
the frozen DEV-best rule (strictly highest peer-averaged full-DEV rSum over co-train epochs 0..39, first maximum on ties, warmup never
eligible), checks model_best.pth.tar (epoch / best_rsum) against it, records the GPU of the attempt that ran the best epoch, and (once
built) the cache/prep manifests. Usage: python -I -B tools/bind_parent.py"""
import json, sys
from pathlib import Path
W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'r37s_critic'))
from common import CONDS, SERIES, PARENT_RECIPE, sha256, read_json, write_json, utc

OUTL = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r37s_cc152k_r30h_20261102')


def main():
    import torch
    c = CONDS['cc_r37s']; P = Path(c['parent_dir'])
    done = read_json(P / 'DONE.json'); ep = read_json(P / 'EPOCHS.json')
    co = ep['cotrain']; assert [e['epoch'] for e in co] == list(range(40)), [e['epoch'] for e in co]
    best_rsum, best = 0, None
    for e in co:   # native r25_co_train rule: is_best = rsum > best_rsum (best_rsum starts at 0 at the boundary)
        if e['dev']['rsum'] > best_rsum:
            best_rsum, best = e['dev']['rsum'], e
    logged_best = [e['epoch'] for e in co if e['is_best']]
    ck = torch.load(P / 'model_best.pth.tar', map_location='cpu', weights_only=False)
    attempts = [json.loads(l) for l in open(P / 'R37S_ATTEMPTS.jsonl')]
    runs = {a['job']: read_json(P / f"RUN_{a['job']}.json") for a in attempts if (P / f"RUN_{a['job']}.json").exists()}
    # attempt that ran the best epoch: the last attempt whose resume epoch is < best epoch
    owner = None
    for a in attempts:
        if a['resume_epoch'] is None or a['resume_epoch'] < best['epoch']:
            owner = a
    gpu_block = read_json(OUTL / 'GPU_BLOCK.json')
    run = runs[owner['job']]
    peak = {'parents/' + f.parent.name + '/' + f.name: read_json(f) for f in sorted(OUTL.glob('parents/*/PEAK_*.json'))}
    peak.update({'runs/' + P.name + '/' + f.name: read_json(f) for f in sorted(P.glob('PEAK_*.json'))})
    b = dict(created_utc=utc(), line='R37S critic line; development experiment, no TEST read',
             parent_unit=c['h_unit'], series=SERIES, never_mixed_with='R31 H0_default parents (R37C / R38 CC tables): different training recipe/series; separate tables only',
             dataset='CC152K', noise='real (uninjected), observed relation = native identity', parent_seed=c['parent_seed'], parent_dir=str(P),
             recipe={k: run['opt'][k] for k in ('warmup_epoch', 'k', 'i2t', 't2i', 'arm', 'lam', 'num_epochs', 'lr_update', 'learning_rate', 'batch_size', 'noise_ratio', 'seed', 'r30_objective')},
             recipe_matches_frozen=all(run['opt'][k] == v for k, v in PARENT_RECIPE.items()),
             warmup_parent=dict(state=str(OUTL / 'parents' / 'wp_cc_s1001_H_e5' / 'parent_state.pt'), sha256_logged=attempts[0]['command'][attempts[0]['command'].index('--r30_parent_sha') + 1],
                                warmup_dev_rsum=[e['dev']['rsum'] for e in ep['warmup']]),
             done=done, epochs_done=len(co), dev_rsum_by_epoch=[e['dev']['rsum'] for e in co],
             best_cotrain_epoch=best['epoch'], best_dev_rsum=best['dev']['rsum'], best_epoch_dev=best['dev'], logged_is_best_epochs=logged_best,
             rule='strictly highest peer-averaged full-DEV rSum over co-train epochs 0..39 (first maximum on ties; best_rsum starts at 0 at the boundary; warmup never eligible) -- native r25_co_train rule (R30 H0)',
             model_best=str(P / 'model_best.pth.tar'), model_best_sha256=sha256(P / 'model_best.pth.tar'), model_best_epoch=int(ck['epoch']), model_best_best_rsum=float(ck['best_rsum']),
             model_best_consistent=int(ck['epoch']) == best['epoch'] and float(ck['best_rsum']) == best['dev']['rsum'] == float(done['best_rsum']) and logged_best[-1] == best['epoch'],
             attempts=[dict(job=a['job'], mode=a['mode'], resume_epoch=a['resume_epoch'], gpu=runs.get(a['job'], {}).get('env', {}).get('gpu'), host=runs.get(a['job'], {}).get('env', {}).get('host')) for a in attempts],
             best_epoch_attempt=owner['job'], best_epoch_gpu=run['env']['gpu'], gpu_block=gpu_block,
             cost=dict(per_epoch_seconds_cotrain=[e['seconds'] for e in co], cotrain_seconds_sum=sum(e['seconds'] for e in co),
                       warmup_seconds_sum=sum(e.get('seconds', 0) for e in ep['warmup']), peak_files=peak))
    cdir = Path(c['cache_dir']); pdir = Path(c['prep_dir'])
    if (cdir / 'CACHE_MANIFEST.json').exists():
        man = read_json(cdir / 'CACHE_MANIFEST.json')
        b['cache'] = dict(dir=str(cdir), manifest_sha256=sha256(cdir / 'CACHE_MANIFEST.json'), native_dev=man['native_dev'], gpu=man['env']['gpu'], host=man['env']['host'], job=man['env']['job'])
    if (pdir / 'PREP_MANIFEST.json').exists():
        pm = read_json(pdir / 'PREP_MANIFEST.json')
        b['prep'] = dict(dir=str(pdir), manifest_sha256=sha256(pdir / 'PREP_MANIFEST.json'), files={n: r['sha256'] for n, r in pm['files'].items()},
                         a0b0={x: pm['calibration_init'][c['prep_key']][x] for x in ('a0', 'b0')}, candidates=pm['candidates'][c['prep_key']],
                         trust={x: pm['trust'][c['prep_key']][x] for x in ('n_subset', 'retention', 'prob_A_mean', 'prob_B_mean', 'agreement_A_B_ge_half')}, seconds=pm['seconds'])
    assert b['model_best_consistent'] and b['recipe_matches_frozen'] and done['status'] == 'COMPLETE', (b['model_best_consistent'], b['recipe_matches_frozen'], done)
    write_json(W / 'PARENT_BINDING.json', b)
    print('PARENT_BINDING', b['best_cotrain_epoch'], b['best_dev_rsum'], b['best_epoch_gpu'], 'cache' in b, 'prep' in b)


if __name__ == '__main__':
    main()
