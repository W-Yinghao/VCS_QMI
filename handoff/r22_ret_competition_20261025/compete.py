"""Competition-feature reliability estimators for V9 N2 (see PROTOCOL.json).

extract --seed S : GPU, frozen SGR teacher, in-batch competitor features for every C0 cache part (two independent passes)
fit --seed S     : CPU, rich_pilot fitting protocol for scalar vs scalar+competition, VCS and BCE; U export (h, T, w, rho)
analyze          : CPU, the only step that reads the synthetic-mismatch labels (eval_only), after all exports exist
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
REPO = BASE / 'ssl_pilot_v9_n2p_c'
RICH = REPO / 'v9_n2p/VCS_V9_N2P_V10_Review_Server_Package_v1_20261020/v9/rich_input'
C0 = BASE / 'outputs_v9_n2p/c_20261020/c0'
OLD = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/n2/f30k/rho0.4')
DATA = Path('/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data/f30k_precomp')
VOCAB = Path('/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/vocab/f30k_precomp_vocab.json')
sys.path.insert(0, str(RICH))
import rich_pilot as RP  # noqa: E402

PARTS = RP.PARTS
FEATURES = ('m_i2t_max', 'm_i2t_mean', 'r_i2t', 'm_t2i_max', 'm_t2i_mean', 'r_t2i', 'agree_top', 'rank_gap')
BATCH, MIN_LAST = 128, 32
SEEDS = (901, 902, 903)


def batches(n, rng):
    order = rng.permutation(n)
    cuts = list(range(0, n, BATCH))
    groups = [order[c:c + BATCH] for c in cuts]
    if len(groups) > 1 and len(groups[-1]) < MIN_LAST:
        groups[-2] = np.concatenate([groups[-2], groups[-1]]); groups.pop()
    return groups


def competition(sims, s, img):
    """sims[image_row, caption_col] for one batch; s = its diagonal; competitors exclude rows sharing the observed image."""
    n = len(s); same = img[:, None] == img[None, :]
    valid = ~same  # diagonal is same-image by construction
    out = np.zeros((n, len(FEATURES))); k = np.zeros((n, 2), dtype=int)
    for a in range(n):
        ci = sims[a, valid[a]]; ct = sims[valid[:, a], a]
        k[a] = len(ci), len(ct)
        if not len(ci) or not len(ct):
            raise ValueError('row without competitors')
        ri, rt = float(np.mean(ci >= s[a])), float(np.mean(ct >= s[a]))
        out[a] = (s[a] - ci.max(), s[a] - ci.mean(), ri, s[a] - ct.max(), s[a] - ct.mean(), rt, float(ri == 0 and rt == 0), abs(ri - rt))
    return out, k


def extract(seed):
    import torch
    from functools import partial
    sys.path.insert(0, str(REPO / 'src'))
    from v9.n2.data import load_precomp, load_vocab
    from v9.n2.fixtures import collate_pairs
    from v9.n2.teacher import load_teacher
    out = HERE / f's{seed}'; out.mkdir(exist_ok=True)
    assert not (out / 'FEATURES_MANIFEST.json').exists(), 'features already extracted'
    t0 = time.time()
    ds = load_precomp(DATA, 'train', load_vocab(VOCAB), cache_dir=C0 / f's{seed}/cache/tokens')
    collate = partial(collate_pairs, ds['images'], ds['captions'])
    host, cfg = load_teacher(OLD / f's{seed}/teacher/teacher.pt', 'cuda')
    manifest = {'seed': seed, 'batch': BATCH, 'min_last': MIN_LAST, 'features': FEATURES, 'parts': {},
                'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
                'teacher_sha256': RP.sha256(OLD / f's{seed}/teacher/teacher.pt'),
                'cache_manifest_sha256': RP.sha256(C0 / f's{seed}/cache/CACHE_MANIFEST.json')}
    for pi, part in enumerate(PARTS):
        with np.load(C0 / f's{seed}/cache/{part}.npz', allow_pickle=False) as z:
            rows = [dict(caption_id=str(c), observed_image_id=str(i)) for c, i in zip(z['caption_id'], z['observed_image_id'])]
            cached_s = np.asarray(z['s'], dtype=np.float64); pid = np.asarray(z['pair_id']).astype(str)
        img = np.array([r['observed_image_id'] for r in rows])
        arrays = {'pair_id': pid}; maxdiff = 0.
        for p in (0, 1):
            feats = np.full((len(rows), len(FEATURES)), np.nan); kk = np.zeros((len(rows), 2), dtype=int)
            with torch.no_grad():
                for g in batches(len(rows), np.random.default_rng([seed, 9040 + p, pi])):
                    ims, caps, lens = collate([rows[i] for i in g])
                    sims = host(ims.cuda(), caps.cuda(), lens).double().cpu().numpy()
                    d = np.diag(sims).copy(); maxdiff = max(maxdiff, float(np.abs(d - cached_s[g]).max()))
                    f, k = competition(sims, d, img[g]); feats[g] = f; kk[g] = k
            assert np.isfinite(feats).all()
            arrays[f'pass{p}'] = feats; arrays[f'k{p}'] = kk
        if maxdiff > 1e-4:
            raise ValueError(f'{part}: teacher diagonal differs from cached s by {maxdiff}')
        np.savez_compressed(out / f'comp_{part}.npz', **arrays)
        manifest['parts'][part] = dict(n=len(rows), max_abs_diag_minus_cached_s=maxdiff, sha256=RP.sha256(out / f'comp_{part}.npz'),
                                       k_min=int(min(arrays['k0'].min(), arrays['k1'].min())), k_mean=float(arrays['k0'].mean()))
        print(seed, part, manifest['parts'][part], flush=True)
    manifest['seconds'] = time.time() - t0
    RP.write_json(out / 'FEATURES_MANIFEST.json', manifest)


def load_part(seed, part):
    with np.load(C0 / f's{seed}/cache/{part}.npz', allow_pickle=False) as z:
        rec = {k: np.asarray(z[k]).astype(str) for k in ('pair_id', 'caption_id', 'observed_image_id')}
        rec['s'] = np.asarray(z['s'], dtype=np.float64)
        if part == 'U': rec['est_mask'] = np.asarray(z['est_mask'])
    with np.load(HERE / f's{seed}/comp_{part}.npz', allow_pickle=False) as c:
        assert np.array_equal(np.asarray(c['pair_id']).astype(str), rec['pair_id'])
        rec['comp'] = np.asarray(c['pass0'], dtype=np.float64)
    return rec


def fit(seed):
    cfg = RP.load_config(RICH / 'config.json')
    _, m, cache_hash = RP.load_cache(C0 / f's{seed}/cache', ())  # full hash/identity validation of the C0 cache
    man = json.loads((HERE / f's{seed}/FEATURES_MANIFEST.json').read_text())
    for part, spec in man['parts'].items(): assert RP.sha256(HERE / f's{seed}/comp_{part}.npz') == spec['sha256']
    d = {p: load_part(seed, p) for p in PARTS}
    norm = RP.standardizer(d['fit_P']['comp'], d['fit_Q']['comp'], cfg['standardization_std_floor'])
    design = lambda part, cls: (np.empty((len(d[part]['s']), 0)) if cls == 'scalar'
                                else (d[part]['comp'] - np.asarray(norm['center'])) / np.asarray(norm['scale']))
    out = {}; t0 = time.time()
    for cls in ('scalar', 'comp'):
        x = {p: design(p, cls) for p in PARTS}
        for obj in ('vcs', 'bce'):
            cands = []
            for lr in cfg['learning_rates']:
                for init in cfg['initializations']:
                    theta, diag = RP.fit_candidate(d['fit_P']['s'], x['fit_P'], d['fit_Q']['s'], x['fit_Q'], obj, lr, init, cfg)
                    hp = RP.linear_h(theta, d['select_P']['s'], x['select_P']); hq = RP.linear_h(theta, d['select_Q']['s'], x['select_Q'])
                    cands.append(dict(lr=lr, init=init, theta=theta, common_J=RP.common_j(hp, hq), select_auc=RP.select_auc(hp, hq), **diag))
            best = cands[0]
            for c in cands[1:]:
                if c['common_J'] > best['common_J'] + cfg['selection_tie_tolerance']: best = c
            th = best['theta']; h = {p: RP.linear_h(th, d[p]['s'], x[p]) for p in ('ref_P', 'ref_Q', 'U')}
            rho = RP.rho_moments(np.tanh(h['ref_P']), np.tanh(h['ref_Q']), np.tanh(h['U'])[d['U']['est_mask']], cfg)
            w = RP.posterior_from_h(h['U'], rho['projected']) if rho['status'] == 'point_estimate' else None
            key = f'{cls}_{obj}'
            np.savez_compressed(HERE / f's{seed}/{key}_U.npz', pair_id=d['U']['pair_id'], h=h['U'], T=np.tanh(h['U']),
                                **({'w': w} if w is not None else {}), est_mask=d['U']['est_mask'])
            out[key] = dict(selected=dict(lr=best['lr'], init=best['init'], common_J=best['common_J'], select_auc=best['select_auc'],
                                          model=RP.theta_model(th)), rho=rho, deployable=w is not None,
                            grid=[dict(lr=c['lr'], init=c['init'], common_J=c['common_J']) for c in cands])
    # consistency: the refit scalar critic must reproduce the C0 scalar export on U
    for obj in ('vcs', 'bce'):
        with np.load(C0 / f's{seed}/export/scalar_{obj}_scores.npz', allow_pickle=False) as z:
            with np.load(HERE / f's{seed}/scalar_{obj}_U.npz', allow_pickle=False) as y:
                assert np.array_equal(np.asarray(z['pair_id']).astype(str), np.asarray(y['pair_id']).astype(str))
                out[f'scalar_{obj}']['max_abs_h_diff_vs_C0_export'] = float(np.abs(np.asarray(z['h']) - np.asarray(y['h'])).max())
    RP.write_json(HERE / f's{seed}/FIT.json', dict(seed=seed, cache_manifest_sha256=cache_hash, normalization=norm, features=FEATURES,
                                                  models=out, seconds=time.time() - t0, labels_read=False, dev_read=False, test_read=False))
    print(json.dumps({k: (v['selected']['common_J'], v['rho']['status'], v.get('max_abs_h_diff_vs_C0_export')) for k, v in out.items()}), flush=True)


def auroc(pos, neg):
    s = np.r_[pos, neg]; y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    from scipy.stats import rankdata
    r = rankdata(s); return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def analyze():
    from scipy.stats import kendalltau
    res = {}
    for seed in SEEDS:
        labels = json.loads((OLD / f's{seed}/eval_only/pairs.eval_only.json').read_text())
        u = load_part(seed, 'U')
        mis = np.array([labels[p]['mismatched'] for p in u['pair_id']], dtype=bool)
        with np.load(HERE / f's{seed}/comp_U.npz', allow_pickle=False) as c: b = np.asarray(c['pass1'])
        correct = ~mis
        hard = correct & ((b[:, 2] > 0) | (b[:, 5] > 0)); easy = correct & ~hard
        mm = .5 * (b[:, 1] + b[:, 4]); q = np.quantile(mm[correct], .25); hard_m = correct & (mm <= q)
        held = ~u['est_mask']
        H = {}
        for key in ('scalar_vcs', 'scalar_bce', 'comp_vcs', 'comp_bce'):
            with np.load(HERE / f's{seed}/{key}_U.npz', allow_pickle=False) as z: H[key] = np.asarray(z['h'])
        for obj in ('vcs', 'bce'):
            with np.load(C0 / f's{seed}/export/rich_{obj}_scores.npz', allow_pickle=False) as z: H[f'rich_{obj}'] = np.asarray(z['h'])
        r = dict(n_U=int(len(mis)), n_mismatched=int(mis.sum()), frac_hard_correct=float(hard.sum() / correct.sum()), est={})
        for key, h in H.items():
            e = {}
            for sup in (.9, .8):
                thr = np.quantile(h[mis], sup); keep = h > thr
                e[f'sup{int(sup*100)}'] = dict(all_correct=float(keep[correct].mean()), hard=float(keep[hard].mean()), easy=float(keep[easy].mean()),
                                               hard_margin_q1=float(keep[hard_m].mean()),
                                               held_hard=float(keep[hard & held].mean()), held_easy=float(keep[easy & held].mean()))
            e['auroc'] = auroc(h[correct], h[mis]); e['auroc_hard'] = auroc(h[hard], h[mis]); e['auroc_easy'] = auroc(h[easy], h[mis])
            r['est'][key] = e
        for obj in ('vcs', 'bce'):
            r[f'kendall_scalar_comp_{obj}'] = float(kendalltau(H[f'scalar_{obj}'], H[f'comp_{obj}'])[0])
        fitj = json.loads((HERE / f's{seed}/FIT.json').read_text())['models']
        r['rho'] = {k: (v['rho']['status'], v['rho']['projected']) for k, v in fitj.items()}
        res[seed] = r
    def diffs(a, b, field, sup='sup90'):
        v = [100 * (res[s]['est'][a][sup][field] - res[s]['est'][b][sup][field]) for s in SEEDS]
        return dict(per_seed=v, mean=float(np.mean(v)), positive=int(sum(x > 0 for x in v)))
    contrasts = {f'{c}': {f: diffs(*c.split(' - '), f) for f in ('hard', 'easy', 'all_correct', 'hard_margin_q1')}
                 for c in ('comp_vcs - scalar_vcs', 'comp_bce - scalar_bce', 'comp_vcs - comp_bce', 'scalar_vcs - scalar_bce', 'rich_vcs - scalar_vcs', 'rich_bce - scalar_bce')}
    ok = lambda c: contrasts[c]['hard']['mean'] >= 5 and contrasts[c]['hard']['positive'] >= 2 and contrasts[c]['easy']['mean'] >= -1
    deploy = all(res[s]['rho'][k][0] == 'point_estimate' for s in SEEDS for k in ('comp_vcs', 'comp_bce'))
    interface = ok('comp_vcs - scalar_vcs') and ok('comp_bce - scalar_bce') and deploy
    v = contrasts['comp_vcs - comp_bce']
    vcs_specific = interface and v['hard']['mean'] >= 2 and v['hard']['positive'] >= 2 and v['easy']['mean'] >= -1
    RP.write_json(HERE / 'RESULT.json', dict(per_seed={str(k): v for k, v in res.items()}, contrasts_sup90_pp=contrasts,
                                             interface_GO=bool(interface), vcs_specific_GO=bool(vcs_specific), deployability_ok=bool(deploy),
                                             note='Labels (eval_only synthetic mismatch) read here only, after all fits/exports. Train roles only; no DEV/TEST.'))
    print(json.dumps(dict(interface_GO=interface, vcs_specific_GO=vcs_specific, deploy=deploy,
                          hard={c: round(x['hard']['mean'], 3) for c, x in contrasts.items()},
                          easy={c: round(x['easy']['mean'], 3) for c, x in contrasts.items()}), indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=('extract', 'fit', 'analyze')); p.add_argument('--seed', type=int, choices=SEEDS)
    a = p.parse_args()
    {'extract': lambda: extract(a.seed), 'fit': lambda: fit(a.seed), 'analyze': analyze}[a.mode]()
