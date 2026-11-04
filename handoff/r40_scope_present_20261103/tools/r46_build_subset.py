"""R46-A prep (r46o tree): fixed FIT pools of the COCO 80% parent R33 H0 c80 s1001 (R43 cache, read-only).
  image subsets: one permutation of the 113287 train images (np.random.default_rng(20261146)); S29 = sorted(first 29000), S58 = sorted(first 58000)
  (nested). Record pool R(S) = captions whose OBSERVED image (NOISE_LOCK map, unchanged) is in S (5 per image, asserted).
  Within the pool, with the frozen parent H (R43-A rules): record locks = R35 record_locks over len(R) records (same seeds) mapped to global ids;
  calibration a0/b0 = balanced logistic on cal_p / cal_q; natural top-32 tables: i2t for images in S over captions in R, t2i for captions in R over
  images in S (peer-mean native fp32 score, torch.topk(32)); rows outside the subset keep the R43 full-pool rows (never sampled: the sampler draws
  records from R only). JOINT prep: byte copy of the R43 c80in_s1001 JOINT prep (same cache; manifest cond renamed, source recorded).
No test file is opened. Usage: python -B tools/r46_build_subset.py"""
from __future__ import annotations
import json, os, shutil, sys, time
from pathlib import Path
import numpy as np

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'r46o'))
import common as C
from common import PEERS, N_TRAIN_IMG, N_TRAIN_CAP, install_guards, guard_record, sha256, write_json, read_json, utc
N_CAL_P, N_CAL_Q, N_LIST = 5000, 50000, 1024


def record_locks(n_rec):
    """R35 prep.record_locks verbatim (R37C / tools/r43_build_coco.py copy), parametrised by the number of records (seeds 20261110..20261113)."""
    rng = np.random.default_rng(20261110)
    perm = rng.permutation(n_rec)
    cal_p, ev_p = np.sort(perm[:N_CAL_P]), np.sort(perm[N_CAL_P:2 * N_CAL_P])
    def qpairs(r):
        a = r.integers(0, n_rec, N_CAL_Q); b = r.integers(0, n_rec, N_CAL_Q)
        keep = a != b
        while not keep.all():
            b[~keep] = r.integers(0, n_rec, int((~keep).sum())); keep = a != b
        return np.stack([a, b], 1)
    cal_q = qpairs(np.random.default_rng(20261111)); ev_q = qpairs(np.random.default_rng(20261112))
    r = np.random.default_rng(20261113)
    lp = r.permutation(n_rec)[:4 * N_LIST]
    L = dict(cal_p=cal_p, cal_q=cal_q, ev_p=ev_p, ev_q=ev_q,
             lcal_i=lp[:N_LIST], lcal_t=lp[N_LIST:2 * N_LIST], lev_i=lp[2 * N_LIST:3 * N_LIST], lev_t=lp[3 * N_LIST:])
    for k in ('lcal_i', 'lcal_t', 'lev_i', 'lev_t'):
        L[k + '_perm'] = np.stack([r.permutation(32) for _ in range(N_LIST)])
    return L


def main():
    import torch
    install_guards(role='gate')
    import train_unit as TU
    TU.set_determinism()
    import evals as EV
    from data35 import load_cond
    base = 'c80in_s1001'
    d = load_cond(base, torch.device('cuda'), need_prep=True); obs = d['obs_np']
    perm = np.random.default_rng(20261146).permutation(N_TRAIN_IMG)
    xs = [d[('fit', 'img', p)] for p in PEERS]; ys = [d[('fit', 'cap', p)] for p in PEERS]
    for n in (29000, 58000):
        cond = f'{base}_p{n // 1000}k'; c = C.CONDS[cond]; pdir = Path(c['prep_dir'])
        if (pdir / 'PREP_MANIFEST.json').exists():
            print('subset prep already built', cond); continue
        pdir.mkdir(parents=True, exist_ok=True); files = {}

        def save_npz(name, **arrs):
            np.savez(pdir / name, **arrs); files[name] = dict(sha256=sha256(pdir / name), keys=sorted(arrs))
        S = np.sort(perm[:n]); R = np.nonzero(np.isin(obs, S))[0]
        assert len(R) == 5 * n and (np.bincount(obs[R], minlength=N_TRAIN_IMG)[S] == 5).all()
        if n == 58000:
            assert np.isin(np.sort(perm[:29000]), S).all()
        save_npz(c['pool_file'], images=S, records=R)
        L0 = record_locks(len(R)); L = {}
        for k, v in L0.items():
            L[k] = v if k.endswith('_perm') else R[v]
        save_npz('locks_records.npz', **L)
        t0 = time.time()

        def nps(img_idx, cap_idx):
            out = 0
            with torch.no_grad():
                for p in PEERS:
                    s = (d[('fit', 'img', p)][torch.as_tensor(img_idx, device='cuda')] * d[('fit', 'cap', p)][torch.as_tensor(cap_idx, device='cuda')]).sum(-1)
                    out = out + s.double().cpu().numpy()
            return out / 2
        sp = nps(obs[L['cal_p']], L['cal_p']); sq = nps(obs[L['cal_q'][:, 0]], L['cal_q'][:, 1])
        g = np.concatenate([sp, sq]); wp = np.r_[np.full(len(sp), 1 / len(sp)), np.zeros(len(sq))]; wq = np.r_[np.zeros(len(sp)), np.full(len(sq), 1 / len(sq))]
        fit = EV.fit_affine(g, wp, wq, 'balanced_logistic'); assert fit['a'] > 0 and fit['finite']
        i2t = d['cand_i2t'].copy(); t2i = d['cand_t2i'].copy()
        St = torch.as_tensor(S, device='cuda'); Rt = torch.as_tensor(R, device='cuda')
        with torch.no_grad():
            yR = [y[Rt] for y in ys]; xS = [x[St] for x in xs]
            for i in range(0, n, 500):
                ii = St[i:i + 500]
                top = (sum(x[ii] @ y.T for x, y in zip(xs, yR)) / 2).topk(32, dim=1).indices
                i2t[S[i:i + 500]] = R[top.cpu().numpy()]
            for j in range(0, len(R), 4000):
                jj = Rt[j:j + 4000]
                top = (sum(y[jj] @ x.T for x, y in zip(xS, ys)) / 2).topk(32, dim=1).indices
                t2i[R[j:j + 4000]] = S[top.cpu().numpy()]
        name = f'candidates_{cond}.npz'; save_npz(name, i2t_top32=i2t.astype(np.int32), t2i_top32=t2i.astype(np.int32))
        assert np.isin(i2t[S], R).all() and np.isin(t2i[R], S).all()
        cov = dict(i2t_frac_subset_images_with_observed_caption_in_top32=float((obs[i2t[S].astype(np.int64)] == S[:, None]).any(1).mean()),
                   t2i_frac_pool_captions_observed_image_in_top32=float((t2i[R].astype(np.int64) == obs[R][:, None]).any(1).mean()),
                   i2t_frac_pool_records_query_present=float((obs[i2t[obs[R]].astype(np.int64)] == obs[R][:, None]).any(1).mean()))
        man = dict(line=f'R46-A subset prep ({n} images; R43-A rules within the pool)', created_utc=utc(), base_cond=base, subset=dict(n_images=n, n_records=int(len(R)), seed=20261146,
                   rule='S = sorted(first n of np.random.default_rng(20261146).permutation(113287)); R = records whose observed image (NOISE_LOCK map) is in S; nested 29k in 58k',
                   images_sha256=sha256(pdir / c['pool_file']), realized_mismatch_in_pool=float((obs[R] != R // 5).mean())),
                   base_prep_manifest_sha256=sha256(Path(C.CONDS[base]['prep_dir']) / 'PREP_MANIFEST.json'),
                   locks=dict(rule=f'R35 record_locks with n_records={len(R)} (seeds 20261110..20261113), indices mapped through R'),
                   calibration_init={cond: dict(a0=fit['a'], b0=fit['b'], balanced_logistic_value=fit['value'], fit_set='pool locks cal_p / cal_q, observed map',
                                                J_raw_fit=EV.j_weighted(g, wp, wq), J_affine_fit=EV.j_weighted(fit['a'] * g + fit['b'], wp, wq), dev_rank_invariance=EV.rank_invariance(d['native_dev'], fit['a'], fit['b']))},
                   candidates={cond: dict(cov, seconds=time.time() - t0, rule='peer-mean native fp32 score within the pool (S images x R captions), torch.topk(32); rows outside the pool = R43 full-pool rows (never sampled)')},
                   files=files, test_guard=guard_record(), env=dict(gpu=torch.cuda.get_device_name(0), job=os.environ.get('SLURM_JOB_ID')))
        write_json(pdir / 'PREP_MANIFEST.json', man)
        # JOINT prep: byte copy of the base JOINT prep, cond renamed (same cache manifest)
        src = Path(C.JOINT_PARENTS[base]['prep_dir']); dst = Path(C.JOINT_PARENTS[cond]['prep_dir']); dst.mkdir(parents=True, exist_ok=True)
        jm = read_json(src / 'JOINT_PREP_MANIFEST.json')
        for fn in jm['files']:
            shutil.copyfile(src / fn, dst / fn); assert sha256(dst / fn) == jm['files'][fn]['sha256']
        jm2 = dict(jm, cond=cond, target_cond=cond, prep_key=cond, copied_from=dict(dir=str(src), manifest_sha256=sha256(src / 'JOINT_PREP_MANIFEST.json')), line='R46-A JOINT prep (byte copy of the R43 c80in_s1001 JOINT prep; cond renamed)')
        write_json(dst / 'JOINT_PREP_MANIFEST.json', jm2)
        print('R46_SUBSET_DONE', cond, man['subset'], cov, fit['a'], flush=True)
    print('R46_BUILD_SUBSET_DONE')


if __name__ == '__main__':
    main()
