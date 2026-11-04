"""R46-B round-2 REBUILT candidate tables: for one condition and each objective, encode the full FIT pool with that objective's round-1 student
(step-8000 checkpoint of the round-1 N-JOINT unit; parent per-native-batch centring as in training), peer-mean student cosine, natural top-32
per query (torch.topk(32), the R35/R37 rule; i2t chunk 500 images, t2i chunk 4000 captions). Observed relations only (no truth). Writes
R46_PARENTS/r2cand/<cond>_<obj>.npz + R2CAND_MANIFEST_<cond>.json with coverage vs the parent (H) tables: share of FIT records whose
observed partner is in the top-32 (i2t query = observed image, t2i query = caption), positive-mass share.
No test file is opened. Usage: python -B tools/r46_build_r2cand.py --tree r46f|r46o --cond n60|f80in|c80in_s1001"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

W = Path(__file__).resolve().parents[1]
R39R = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r39_complete_joint_20261102/runs')
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r40_scope_present_20261103')
ROUND1 = {'n60': lambda o: R39R / f'R39_Flickr60_N_joint_global_{o}', 'f80in': lambda o: OUT / 'runs_r41' / f'R41_C_F80_IN2R_H1001_N_joint_{o}',
          'c80in_s1001': lambda o: OUT / 'runs_r43' / f'R43_A_C80_IN2R_H1001_N_joint_{o}'}


def coverage(obs, i2t, t2i, n_img):
    pres_i = (obs[i2t.astype(np.int64)] == np.arange(n_img)[:, None])      # per image: which top-32 captions are observed partners
    pres_t = (t2i.astype(np.int64) == obs[:, None])
    return dict(i2t_records_partner_in_top32=float(pres_i.any(1)[obs].mean()), t2i_records_partner_in_top32=float(pres_t.any(1).mean()),
                i2t_images_with_partner_in_top32=float(pres_i.any(1).mean()), i2t_mean_observed_partners_in_top32=float(pres_i.sum(1).mean()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tree', required=True, choices=('r46f', 'r46o')); ap.add_argument('--cond', required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(W / a.tree))
    import torch
    import common as C
    from common import PEERS, N_TRAIN_IMG, N_TRAIN_CAP, install_guards, guard_record, sha256, write_json, utc
    install_guards(role='gate')
    import train_unit as TU
    TU.set_determinism()
    import r39ext as R9
    from data35 import load_cond
    dev = torch.device('cuda')
    d = load_cond(a.cond, dev); obs = d['obs_np']
    J = R9.JointInputs(a.cond, d, dev)
    odir = C.R46_PARENTS / 'r2cand'; odir.mkdir(parents=True, exist_ok=True)
    mf = odir / f'R2CAND_MANIFEST_{a.cond}.json'
    man = json.loads(mf.read_text()) if mf.exists() else dict(line='R46-B round-2 REBUILT candidate tables', cond=a.cond, created_utc=utc(), parent_tables=dict(sha256=d['cand_sha256'], coverage=coverage(obs, d['cand_i2t'], d['cand_t2i'], N_TRAIN_IMG)), tables={})
    for o in ('J', 'BCE', 'NCE'):
        if o in man['tables']:
            print('already built', a.cond, o); continue
        u = ROUND1[a.cond](o); ck = u / 'ckpt_step8000.pt'; t0 = time.time()
        par = torch.load(ck, map_location=dev, weights_only=False); assert par['step'] == 8000 and par['config']['cond'] == a.cond and par['config']['objective'] == o
        stud = R9.build_students(J.ck, dev, ('img', 'txt')); stud.load_state_dict(par['student']); stud.train()
        J.clear_eval()
        xs = [J.table(stud[p], p, 'fit', 'img') for p in PEERS]; ys = [J.table(stud[p], p, 'fit', 'cap') for p in PEERS]
        i2t = np.zeros((N_TRAIN_IMG, 32), np.int32); t2i = np.zeros((N_TRAIN_CAP, 32), np.int32)
        with torch.no_grad():
            for i in range(0, N_TRAIN_IMG, 500):
                i2t[i:i + 500] = (sum(x[i:i + 500] @ y.T for x, y in zip(xs, ys)) / 2).topk(32, dim=1).indices.int().cpu().numpy()
            for j in range(0, N_TRAIN_CAP, 4000):
                t2i[j:j + 4000] = (sum(y[j:j + 4000] @ x.T for x, y in zip(xs, ys)) / 2).topk(32, dim=1).indices.int().cpu().numpy()
        fn = odir / f'{a.cond}_{o}.npz'; np.savez(fn, i2t_top32=i2t, t2i_top32=t2i)
        man['tables'][o] = dict(path=str(fn), sha256=sha256(fn), round1_unit=u.name, round1_ckpt=str(ck), round1_ckpt_sha256=sha256(ck), seconds=time.time() - t0,
                                coverage=coverage(obs, i2t, t2i, N_TRAIN_IMG), overlap_with_parent_tables=dict(i2t_mean_jaccard_frac=float(np.mean([len(np.intersect1d(a_, b_)) / 32 for a_, b_ in zip(i2t[:2000], d['cand_i2t'][:2000])])),
                                                                                                    t2i_mean_frac=float(np.mean([len(np.intersect1d(a_, b_)) / 32 for a_, b_ in zip(t2i[:10000], d['cand_t2i'][:10000])]))),
                                rule='round-1 student (step 8000), parent per-native-batch centring, peer-mean cosine, torch.topk(32)')
        man['env'] = dict(gpu=torch.cuda.get_device_name(0), job=os.environ.get('SLURM_JOB_ID')); man['test_guard'] = guard_record()
        write_json(mf, man); J.clear_eval(); del stud, par, xs, ys; torch.cuda.empty_cache()
        print('R46_R2CAND', a.cond, o, json.dumps(man['tables'][o]['coverage']), flush=True)
    print('R46_R2CAND_DONE', a.cond, json.dumps(man['parent_tables']['coverage']))


if __name__ == '__main__':
    main()
