"""R50 build (protocol FIT_FIXED_CENTER_V1; package VCS_Layout_Audit_R50_20261104 docs/SERVER_PLAN_CN.md section 1) of one base parent. No training, no TEST.
  FIT     the parent encoder outputs e (parent checkpoint through the native student modules: BN eval, dropout off, one-layer GRUs train mode; the R39/R47
          student at init) of every FIT item are recomputed (natural order, chunks 512 / 2048) and their mean mu_FIT (fp64 accumulation, one [1, d] row per
          peer / modality; images: the N_TRAIN_IMG unique FIT images, each once; captions: the N_TRAIN_CAP FIT caption records) is compared with the R47 G
          JOINT centring file center_fit_* and normalize(e - mu_FIT) with the R47 G FIT cache. Check A (full scale): the FIT pool re-encoded in a random order
          with chunks 384 / 1536 -> mean and centred vectors compared. The R50 FIT cache files are HARD LINKS of the R47 G FIT cache files (byte-identical by
          construction; the recomputation is the identity evidence).
  DEV     e of every DEV item centred by the R47 G FIT mean (the file values, never a DEV statistic): normalize(e_dev - mu_FIT); native_dev = peer-mean
          100x100-shard fp32 score in fp64 (native evaluator arithmetic) -> PRIMARY H. Also recorded: the R47 SECONDARY_TRANSDUCTIVE DEV (DEV split mean)
          reproduced against the R47 G DEV cache, ||mu_DEV - mu_FIT||, and the DEV H under the permuted-order FIT mean (check A at score level).
  prep    calibration a0/b0 (balanced logistic on the R35 / base locks cal_p / cal_q) and the natural top-32 tables (peer-mean FIT score, torch.topk(32))
          recomputed on the R50 FIT cache with the R47 rules and compared with the R47 G prep; when identical the R47 prep files are byte-copied (else the
          recomputed tables are written and the prep is flagged not identical).
  joint   <base>_F: centring_mode 'global'; center_fit_* AND center_dev_* = byte copies of the R47 G center_fit_* (the DEV student uses the FIT mean);
          student-init vs cache equivalence through the production JointInputs on the new cond.
  --h-only  (extra PRIMARY H of the other R48 parents): DEV cache + native_dev only (no FIT files, no prep / joint).
Writes OUT/r50_parents/{cache,prep,joint}/<base>[_F] and cache/<base>/NORMALIZATION_<base>.json.
Usage: python -B tools/r50_build.py <tree r50f|r50c|r47o> <base cond> [--h-only]"""
import json, os, shutil, sys, time
from pathlib import Path
import numpy as np
W = Path(__file__).resolve().parents[1]
TREE, BASE = sys.argv[1], sys.argv[2]; HONLY = '--h-only' in sys.argv
sys.path.insert(0, str(W / TREE))
import torch
import common as C
from common import PEERS, N_TRAIN_IMG, N_TRAIN_CAP, N_DEV_IMG, N_DEV_CAP, install_guards, guard_record, sha256, write_json, read_json, utc
install_guards(role='gate')
import train_unit as TU
TU.set_determinism()
import evals as EV, r39ext as R9
from data35 import load_cond, shard_mm
PID = 'FIT_FIXED_CENTER_V1'
dev = torch.device('cuda'); t0 = time.time()
R47P = C.OUT / 'r47_parents'; R50P = C.OUT / 'r50_parents'
gC, gP, gJ = R47P / 'cache' / BASE, R47P / 'prep' / BASE, R47P / 'joint' / f'{BASE}_G'
cdir, pdir, jdir = R50P / 'cache' / BASE, R50P / 'prep' / BASE, R50P / 'joint' / f'{BASE}_F'
assert not (cdir / 'CACHE_MANIFEST.json').exists(), f'{cdir} already built'
gman, gjman = read_json(gC / 'CACHE_MANIFEST.json'), read_json(gJ / 'JOINT_PREP_MANIFEST.json')
assert gjman['centring_mode'] == 'global'
NEW = f'{BASE}_lfF'
if not HONLY:
    assert NEW in C.CONDS and Path(C.CONDS[NEW]['cache_dir']) == cdir and Path(C.CONDS[NEW]['prep_dir']) == pdir, 'R50 cond missing in the tree common.py'
d = load_cond(BASE, dev); J = R9.JointInputs(BASE, d, dev)
stud = R9.build_students(J.ck, dev, ('img', 'txt')); stud.train()
n = dict(fit=dict(img=N_TRAIN_IMG, cap=N_TRAIN_CAP), dev=dict(img=N_DEV_IMG, cap=N_DEV_CAP))
TOL_VEC, TOL_MEAN = 1e-5, 1e-6
rng = np.random.default_rng(5050)


def enc(p, split, kind, order=None, ch=None):
    ids = torch.arange(n[split][kind]) if order is None else torch.as_tensor(order)
    ch = ch or (512 if kind == 'img' else 2048)
    with torch.no_grad():
        return torch.cat([J.raw_encode(stud[p], split, kind, ids[i:i + ch]) for i in range(0, len(ids), ch)])


def cmpd(a, b):
    dl = (a.double() - b.double()).abs()
    return dict(max_abs=float(dl.max()), bitwise_rows=float((dl == 0).all(-1).double().mean()))


E, MU, NORM, CHECK_A = {}, {}, {}, {}
for p in PEERS:
    for kind in ('img', 'cap'):
        key = f'{kind}_{p}'
        e = enc(p, 'fit', kind); mu = e.double().mean(0, keepdim=True).float()
        f47 = gJ / f'center_fit_{kind}_peer{p}.npy'; mu47 = torch.from_numpy(np.load(f47)).to(dev)
        assert sha256(f47) == gjman['files'][f47.name]['sha256']
        rec = dict(fit_mean_source=str(f47), fit_mean_sha256=sha256(f47), fit_mean_recomputed_equal_bitwise=bool(torch.equal(mu, mu47)),
                   fit_mean_recomputed_max_abs=float((mu - mu47).abs().max()), n_fit_rows=int(e.shape[0]),
                   weighting=('each FIT image once (N_TRAIN_IMG unique image rows; not per caption record)' if kind == 'img' and N_TRAIN_IMG != N_TRAIN_CAP else
                              'each FIT image row once (one image row per caption record in this dataset)' if kind == 'img' else 'each FIT caption record once'),
                   accumulation='fp64 sum / n over all FIT rows, cast to fp32 (r47_build_lf.py rule)')
        x = J.l2norm(e - mu47)
        if not HONLY:
            g = torch.from_numpy(np.load(gC / f'fit_{kind}_peer{p}.npy')).to(dev)
            rec['fit_cache_vs_r47G'] = cmpd(x, g); del g
        perm = rng.permutation(len(e)); inv = np.argsort(perm)
        ep = enc(p, 'fit', kind, order=perm, ch=384 if kind == 'img' else 1536)
        mup = ep.double().mean(0, keepdim=True).float()
        CHECK_A[key] = dict(mean_perm_vs_natural=cmpd(mup, mu), raw_perm_vs_natural=cmpd(ep[torch.as_tensor(inv, device=dev)], e),
                            centred_perm_vs_natural=cmpd(J.l2norm(ep[torch.as_tensor(inv, device=dev)] - mup), x))
        MU[(kind, p)] = (mu47, mup); del ep, e, x
        ed = enc(p, 'dev', kind); E[(kind, p)] = ed
        mud = ed.double().mean(0, keepdim=True).float()
        sec = J.l2norm(ed - mud); g = torch.from_numpy(np.load(gC / f'dev_{kind}_peer{p}.npy')).to(dev)
        rec['secondary_transductive_dev_reproduces_r47G_dev_cache'] = cmpd(sec, g)
        rec['dev_mean_minus_fit_mean_l2'] = float((mud - mu47).double().norm())
        uc = J.l2norm(ed - mu47)
        rec['variance'] = dict(dev_raw_total_var=float(ed.double().var(0).sum()), dev_fitcentred_minus_mean_sq_norm=float(((ed - mu47) ** 2).sum(1).double().mean()),
                               dev_primary_mean_cos_to_dev_mean=float(J.l2norm(uc.double().mean(0, keepdim=True)).float().mul(uc).sum(1).double().mean()))
        NORM[key] = rec
dvec = {(kind, p): J.l2norm(E[(kind, p)] - MU[(kind, p)][0]).float() for kind in ('img', 'cap') for p in PEERS}
dvec_perm = {(kind, p): J.l2norm(E[(kind, p)] - MU[(kind, p)][1]).float() for kind in ('img', 'cap') for p in PEERS}
sims = (sum(shard_mm(dvec[('img', p)], dvec[('cap', p)]) for p in PEERS) / 2).cpu().numpy()
sims_perm = (sum(shard_mm(dvec_perm[('img', p)], dvec_perm[('cap', p)]) for p in PEERS) / 2).cpu().numpy()
m, _, rk = EV.full_metrics(sims); mp, _, rkp = EV.full_metrics(sims_perm)
TOL_TIE = 1e-6
di = np.nonzero(rk['ranks_i'] != rkp['ranks_i'])[0]; dt = np.nonzero(rk['ranks_t'] != rkp['ranks_t'])[0]
CHECK_A['dev_scores_perm_fit_mean'] = dict(max_abs=float(np.abs(sims - sims_perm).max()), rsum=mp['rsum'], rsum_natural=m['rsum'], n_i2t_rank_diff=int(len(di)), n_t2i_rank_diff=int(len(dt)),
                                           near_tie_rule='rank differences allowed only where |delta s| <= 1e-6 (step-0 native rule)')
mx = lambda v: max(x['max_abs'] for x in v.values())
okA = all(v['mean_perm_vs_natural']['max_abs'] <= TOL_MEAN and v['centred_perm_vs_natural']['max_abs'] <= TOL_VEC for k, v in CHECK_A.items() if k != 'dev_scores_perm_fit_mean')
okA = okA and CHECK_A['dev_scores_perm_fit_mean']['max_abs'] <= TOL_VEC
CHECK_A['passed'] = bool(okA); CHECK_A['tolerances'] = dict(mean_max_abs=TOL_MEAN, vector_max_abs=TOL_VEC)
fit_identical = all(v['fit_mean_recomputed_equal_bitwise'] for v in NORM.values()) and (HONLY or all(v['fit_cache_vs_r47G']['max_abs'] == 0.0 for v in NORM.values()))
H = dict(rsum=m['rsum'], **{k: m[k] for k in ('i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10') if k in m})
print('R50_H', BASE, 'PRIMARY', H['rsum'], 'SECONDARY(R47G)', gman['native_dev']['rsum'], 'native', gman['native_dev'].get('native_layout_rsum'), 'fit_identical', fit_identical, 'A', okA, flush=True)
# ---------------------------------------------------------------- cache
cdir.mkdir(parents=True, exist_ok=True); files = {}
def save(name, arr):
    np.save(cdir / name, arr); files[name] = dict(sha256=sha256(cdir / name), shape=list(arr.shape), dtype=str(arr.dtype))
def link(name):
    src = gC / name
    try:
        os.link(src, cdir / name); how = 'hardlink'
    except OSError:
        shutil.copyfile(src, cdir / name); how = 'copy'
    files[name] = dict(sha256=sha256(cdir / name), shape=gman['files'][name]['shape'], dtype=gman['files'][name]['dtype'], source=str(src), source_sha256=gman['files'][name]['sha256'], how=how)
    assert files[name]['sha256'] == gman['files'][name]['sha256']
for (kind, p), x in dvec.items():
    save(f'dev_{kind}_peer{p}.npy', x.cpu().numpy())
save('native_dev_sims.npy', sims)
if not HONLY:
    for p in PEERS:
        for kind in ('img', 'cap'):
            link(f'fit_{kind}_peer{p}.npy')
    if 'observed_map.npy' in gman['files']:
        link('observed_map.npy')
bm = d['manifest']
nd = dict(H, protocol_id=PID, layout_free=True, reproduces_checkpoint_best_rsum_exactly=True, cache_rescore_identical=True,
          note='R50 PRIMARY: native_dev computed from these FIT-centred DEV vectors by construction (the two flags mean self-consistency as in R47, not reproduction of the native layout-centred DEV rSum)',
          secondary_transductive_rsum_r47G=gman['native_dev']['rsum'], native_layout_rsum=bm['native_dev']['rsum'])
NORMAN = dict(line=f'R50 normalisation manifest {BASE}', protocol_id=PID, base_cond=BASE, parent=bm['parent'], per_peer_modality=NORM, check_A_full_scale=CHECK_A,
              fit_side_identical_to_r47G=bool(fit_identical), h_only=HONLY, env=dict(gpu=torch.cuda.get_device_name(0), job=os.environ.get('SLURM_JOB_ID'), torch=torch.__version__,
              r47_build_gpu=gman['env']['gpu']), utc=utc())
write_json(cdir / f'NORMALIZATION_{BASE}.json', NORMAN)
man = dict(line=f'R50 FIT_FIXED_CENTER_V1 parent cache of {BASE}' + (' (H only: DEV files)' if HONLY else ''), created_utc=utc(), protocol_id=PID, base_cond=BASE,
           base_cache_manifest_sha256=d['manifest_sha256'], r47G_cache_manifest_sha256=sha256(gC / 'CACHE_MANIFEST.json'), parent=bm['parent'], observed_map=gman['observed_map'],
           native_dev=nd, files=files, h_only=HONLY,
           fit_center=dict(source='R47 G JOINT prep center_fit_* (ONE mean per peer / modality over all FIT items; recomputed and compared, NORMALIZATION_<base>.json)',
                           files={f'{k}_{p}': dict(path=NORM[f'{k}_{p}']['fit_mean_source'], sha256=NORM[f'{k}_{p}']['fit_mean_sha256']) for k in ('img', 'cap') for p in PEERS}),
           encoding=dict(mode='parent encoders through the native student modules (BN eval, dropout off, one-layer GRUs train mode)',
                         centering='FIT and DEV: normalize(e - mu_FIT) with ONE FIT mean per (modality, peer); no DEV statistic, no batch / layout grouping'),
           normalization_manifest_sha256=sha256(cdir / f'NORMALIZATION_{BASE}.json'),
           env=NORMAN['env'], test_guard=guard_record())
write_json(cdir / 'CACHE_MANIFEST.json', man)
print('R50_CACHE', BASE, json.dumps(H), flush=True)
if HONLY:
    print('R50_BUILD_DONE', BASE, 'h-only seconds', time.time() - t0); sys.exit(0)
# ---------------------------------------------------------------- prep
d2 = load_cond(NEW, dev, need_prep=False)
pdir.mkdir(parents=True, exist_ok=True); pfiles = {}
gpm = read_json(gP / 'PREP_MANIFEST.json'); cb = C.CONDS[BASE]
L = np.load(gP / 'locks_records.npz') if (gP / 'locks_records.npz').exists() else np.load(C.R35_PREP / 'locks_records.npz')
L = {k: L[k] for k in L.files}; obs = d2['obs_np']
def nps(img_idx, cap_idx):
    out = 0
    with torch.no_grad():
        for p in PEERS:
            out = out + (d2[('fit', 'img', p)][torch.as_tensor(img_idx, device=dev)] * d2[('fit', 'cap', p)][torch.as_tensor(cap_idx, device=dev)]).sum(-1).double().cpu().numpy()
    return out / 2
sp = nps(obs[L['cal_p']], L['cal_p']); sq = nps(obs[L['cal_q'][:, 0]], L['cal_q'][:, 1])
g = np.concatenate([sp, sq]); wp = np.r_[np.full(len(sp), 1 / len(sp)), np.zeros(len(sq))]; wq = np.r_[np.zeros(len(sp)), np.full(len(sq), 1 / len(sq))]
fit = EV.fit_affine(g, wp, wq, 'balanced_logistic'); assert fit['a'] > 0 and fit['finite']
xs = [d2[('fit', 'img', p)] for p in PEERS]; ys = [d2[('fit', 'cap', p)] for p in PEERS]
i2t = np.zeros((N_TRAIN_IMG, 32), np.int32); t2i = np.zeros((N_TRAIN_CAP, 32), np.int32)
with torch.no_grad():
    for i in range(0, N_TRAIN_IMG, 500):
        i2t[i:i + 500] = (sum(x[i:i + 500] @ y.T for x, y in zip(xs, ys)) / 2).topk(32, dim=1).indices.int().cpu().numpy()
    for j in range(0, N_TRAIN_CAP, 4000):
        t2i[j:j + 4000] = (sum(y[j:j + 4000] @ x.T for x, y in zip(xs, ys)) / 2).topk(32, dim=1).indices.int().cpu().numpy()
z47 = np.load(gP / f'candidates_{BASE}.npz'); c47 = gpm['calibration_init'][BASE]
same_cand = bool(np.array_equal(i2t, z47['i2t_top32']) and np.array_equal(t2i, z47['t2i_top32']))
same_cal = bool(fit['a'] == c47['a0'] and fit['b'] == c47['b0'])
recomp = dict(candidates_equal_r47G=same_cand, candidate_rows_differing=dict(i2t=int((i2t != z47['i2t_top32']).any(1).sum()), t2i=int((t2i != z47['t2i_top32']).any(1).sum())),
              calibration_equal_r47G_bitwise=same_cal, a0=fit['a'], b0=fit['b'], a0_r47G=c47['a0'], b0_r47G=c47['b0'])
fn = f'candidates_{BASE}.npz'
if same_cand:
    shutil.copyfile(gP / fn, pdir / fn)
else:
    np.savez(pdir / fn, i2t_top32=i2t, t2i_top32=t2i)
pfiles[fn] = dict(sha256=sha256(pdir / fn), copied_from=str(gP / fn) if same_cand else None)
for extra in [f for f in gpm['files'] if f != fn]:   # COCO / CC: locks_records.npz (+ CC B-TRUST file), byte copies
    shutil.copyfile(gP / extra, pdir / extra); pfiles[extra] = dict(sha256=sha256(pdir / extra), copied_from=str(gP / extra)); assert pfiles[extra]['sha256'] == gpm['files'][extra]['sha256']
cal = dict(a0=c47['a0'], b0=c47['b0']) if same_cal else dict(a0=fit['a'], b0=fit['b'])
cov = dict(i2t_partner_in_top32=float((obs[i2t.astype(np.int64)] == np.arange(N_TRAIN_IMG)[:, None]).any(1)[obs].mean()), t2i_partner_in_top32=float((t2i.astype(np.int64) == obs[:, None]).any(1).mean()))
pman = dict(line=f'R50 prep of {BASE} (FIT side recomputed on the R50 FIT cache = R47 G FIT cache)', created_utc=utc(), protocol_id=PID,
            base_prep_manifest_sha256=gpm.get('base_prep_manifest_sha256'), r47G_prep_manifest_sha256=sha256(gP / 'PREP_MANIFEST.json'),
            calibration_init={BASE: dict(cal, balanced_logistic_value=fit['value'], fit_set='locks cal_p / cal_q, R50 FIT cache', recomputed=recomp)},
            candidates={BASE: dict(cov, rule='peer-mean FIT score (R50 FIT cache), torch.topk(32)')}, recomputation=recomp,
            identical_to_r47G=bool(same_cand and same_cal), files=pfiles, test_guard=guard_record())
for k in ('r35_prep_manifest_sha256', 'shared_from_r35'):
    if k in gpm:
        pman[k] = gpm[k]
write_json(pdir / 'PREP_MANIFEST.json', pman)
print('R50_PREP', BASE, json.dumps(recomp), flush=True)
# ---------------------------------------------------------------- joint prep
jdir.mkdir(parents=True, exist_ok=True); jfiles = {}
for kind in ('img', 'cap'):
    for p in PEERS:
        src = gJ / f'center_fit_{kind}_peer{p}.npy'
        for split in ('fit', 'dev'):
            dst = jdir / f'center_{split}_{kind}_peer{p}.npy'; shutil.copyfile(src, dst)
            jfiles[dst.name] = dict(sha256=sha256(dst), shape=gjman['files'][src.name]['shape'], copied_from=str(src)); assert jfiles[dst.name]['sha256'] == gjman['files'][src.name]['sha256']
jman = dict(line=f'R50 JOINT prep {BASE} (FIT_FIXED_CENTER_V1)', created_utc=utc(), cond=NEW, target_cond=NEW, centring_mode='global', protocol_id=PID,
            parent=gjman['parent'], target_cache_manifest_sha256=sha256(cdir / 'CACHE_MANIFEST.json'), native_code=gjman['native_code'], tokens=gjman['tokens'], files=jfiles,
            centring_rule='ONE fixed parent FIT mean per modality / peer (R47 G center_fit_*) for FIT AND DEV; center_dev_* are byte copies of center_fit_*',
            base_joint_prep_manifest=gjman.get('line'), r47G_joint_prep_manifest_sha256=sha256(gJ / 'JOINT_PREP_MANIFEST.json'), test_guard=guard_record())
write_json(jdir / 'JOINT_PREP_MANIFEST.json', jman)
d3 = load_cond(NEW, dev)
J3 = R9.JointInputs(NEW, d3, dev); assert J3.mode == 'global'
eq = dict(dev=J3.cache_equivalence(stud, np.arange(N_DEV_IMG), np.arange(N_DEV_CAP), 'dev', grad=False))
ii = np.sort(rng.choice(N_TRAIN_IMG, 2000, replace=False)); cc = np.sort(rng.choice(N_TRAIN_CAP, 8000, replace=False))
eq['fit_sample'] = J3.cache_equivalence(stud, ii, cc, 'fit', grad=False)
jman['equivalence'] = eq; jman['equivalence_summary'] = dict(max_abs_over_all=max(v['max_abs'] for part in eq.values() for v in part.values()))
write_json(jdir / 'JOINT_PREP_MANIFEST.json', jman)
man_c = read_json(cdir / 'CACHE_MANIFEST.json'); assert jman['target_cache_manifest_sha256'] == sha256(cdir / 'CACHE_MANIFEST.json') == d3['manifest_sha256']
print('R50_JOINT', BASE, jman['equivalence_summary'], 'prep identical', pman['identical_to_r47G'], 'seconds', time.time() - t0, flush=True)
print('R50_BUILD_DONE', BASE)
