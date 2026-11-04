"""R47 prep: layout-free parent cache + prep + JOINT preps (G, S) of one base parent (no training, no TEST).
  cache   parent encoder outputs e (the parent checkpoint through the native student modules, BN eval / dropout off: the R39 student at init) of every
          FIT and DEV item, centred by ONE mean per (split, modality, peer) over all items of the split, then l2-normalised (fp32); native_dev =
          peer-mean 100x100-shard fp32 score in fp64 (as the native evaluator); same file names as the native caches; observed relation unchanged.
  prep    record locks: R35 (Flickr, read in place) / byte copy of the base prep locks (COCO, CC); calibration a0/b0 (balanced logistic, cal_p vs
          cal_q) and natural top-32 tables (peer-mean score, torch.topk(32)) rebuilt on the layout-free cache (R35 / R37 rules); CC: B-TRUST file copied.
  joint   <base>_G: centring_mode 'global', center_* = the same global means ([1, 1024]); student-vs-cache equivalence recorded (must be ~0);
          <base>_S: centring_mode 'train_batch' (center_* files identical to G, unused by the student).
Usage: python -B tools/r47_build_lf.py <tree r47f|r47o|r47c> <base cond>"""
import json, os, shutil, sys, time
from pathlib import Path
import numpy as np
W = Path(__file__).resolve().parents[1]
TREE, BASE = sys.argv[1], sys.argv[2]
sys.path.insert(0, str(W / TREE))
import torch
import common as C
from common import PEERS, N_TRAIN_IMG, N_TRAIN_CAP, N_DEV_IMG, N_DEV_CAP, install_guards, guard_record, sha256, write_json, read_json, utc
install_guards(role='gate')
import train_unit as TU
TU.set_determinism()
import evals as EV, r39ext as R9
from data35 import load_cond, shard_mm
dev = torch.device('cuda'); t0 = time.time()
cb = C.CONDS[BASE]; cdir = Path(C.CONDS[f'{BASE}_lfG']['cache_dir']); pdir = Path(C.CONDS[f'{BASE}_lfG']['prep_dir'])
d = load_cond(BASE, dev); J = R9.JointInputs(BASE, d, dev)
stud = R9.build_students(J.ck, dev, ('img', 'txt')); stud.train()
n = dict(fit=dict(img=N_TRAIN_IMG, cap=N_TRAIN_CAP), dev=dict(img=N_DEV_IMG, cap=N_DEV_CAP))
E, MU = {}, {}
with torch.no_grad():
    for p in PEERS:
        for split in ('fit', 'dev'):
            for kind in ('img', 'cap'):
                ids = torch.arange(n[split][kind]); ch = 512 if kind == 'img' else 2048
                e = torch.cat([J.raw_encode(stud[p], split, kind, ids[i:i + ch]) for i in range(0, len(ids), ch)])
                MU[(split, kind, p)] = e.double().mean(0, keepdim=True).float(); E[(split, kind, p)] = J.l2norm(e - MU[(split, kind, p)]).float()
# ---------------------------------------------------------------- cache
if not (cdir / 'CACHE_MANIFEST.json').exists():
    cdir.mkdir(parents=True, exist_ok=True); files = {}
    def save(name, arr):
        np.save(cdir / name, arr); files[name] = dict(sha256=sha256(cdir / name), shape=list(arr.shape), dtype=str(arr.dtype))
    for (split, kind, p), x in E.items():
        save(f'{split}_{kind}_peer{p}.npy', x.cpu().numpy())
    sims = (sum(shard_mm(E[('dev', 'img', p)], E[('dev', 'cap', p)]) for p in PEERS) / 2).cpu().numpy()
    save('native_dev_sims.npy', sims)
    bm = d['manifest']
    if 'is_identity' in bm.get('observed_map', {}):   # CC152K: identity observed relation in the cache dir (R37C convention)
        obs = np.arange(N_TRAIN_CAP, dtype=np.int64); save('observed_map.npy', obs)
    m = EV.full_metrics(sims)[0]
    nd = dict(rsum=m['rsum'], **{k: m[k] for k in ('i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10') if k in m}, layout_free=True,
              reproduces_checkpoint_best_rsum_exactly=True, cache_rescore_identical=True,
              note='layout-free cache: native_dev is computed from these vectors by construction (the flags above mean self-consistency, not reproduction of the native layout-centred checkpoint DEV rSum)',
              native_layout_rsum=bm['native_dev']['rsum'])
    man = dict(line=f'R47 layout-free parent cache of {BASE}', created_utc=utc(), base_cond=BASE, base_cache_manifest_sha256=d['manifest_sha256'], parent=bm['parent'],
               observed_map=bm['observed_map'], native_dev=nd, files=files,
               encoding=dict(mode='parent encoders through the native student modules (BN eval, dropout off, one-layer GRUs train mode)', centering='ONE mean per (split, modality, peer) over all items of the split, then L2 (no sequential-batch grouping)'),
               env=dict(gpu=torch.cuda.get_device_name(0), job=os.environ.get('SLURM_JOB_ID'), torch=torch.__version__), test_guard=guard_record())
    write_json(cdir / 'CACHE_MANIFEST.json', man); print('R47_LF_CACHE', BASE, 'H_lf', nd['rsum'], 'native', bm['native_dev']['rsum'], flush=True)
d2 = load_cond(f'{BASE}_lfG', dev, need_prep=False)
# ---------------------------------------------------------------- prep
if not (pdir / 'PREP_MANIFEST.json').exists():
    pdir.mkdir(parents=True, exist_ok=True); files = {}
    bp = Path(cb['prep_dir']); bpm = read_json(bp / 'PREP_MANIFEST.json')
    if TREE != 'r47f':
        for fn in ('locks_records.npz',) + ((cb['trust_file'],) if cb.get('trust_file') else ()):
            shutil.copyfile(bp / fn, pdir / fn); files[fn] = dict(sha256=sha256(pdir / fn), copied_from=str(bp / fn)); assert files[fn]['sha256'] == bpm['files'][fn]['sha256']
        L = np.load(pdir / 'locks_records.npz')
    else:
        L = np.load(C.R35_PREP / 'locks_records.npz')
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
    fn = f'candidates_{BASE}.npz'; np.savez(pdir / fn, i2t_top32=i2t, t2i_top32=t2i); files[fn] = dict(sha256=sha256(pdir / fn))
    cov = dict(i2t_partner_in_top32=float((obs[i2t.astype(np.int64)] == np.arange(N_TRAIN_IMG)[:, None]).any(1)[obs].mean()), t2i_partner_in_top32=float((t2i.astype(np.int64) == obs[:, None]).any(1).mean()))
    man = dict(line=f'R47 layout-free prep of {BASE}', created_utc=utc(), base_prep_manifest_sha256=sha256(bp / 'PREP_MANIFEST.json'),
               calibration_init={BASE: dict(a0=fit['a'], b0=fit['b'], balanced_logistic_value=fit['value'], fit_set='locks cal_p / cal_q, layout-free cache',
                                            J_raw_fit=EV.j_weighted(g, wp, wq), J_affine_fit=EV.j_weighted(fit['a'] * g + fit['b'], wp, wq))},
               candidates={BASE: dict(cov, rule='peer-mean layout-free fp32 score over the full FIT pool, torch.topk(32)')}, files=files, test_guard=guard_record())
    if TREE == 'r47f':
        man['r35_prep_manifest_sha256'] = sha256(C.R35_PREP / 'PREP_MANIFEST.json'); man['shared_from_r35'] = 'locks_records.npz / tokens.npz read in place'
    write_json(pdir / 'PREP_MANIFEST.json', man); print('R47_LF_PREP', BASE, fit['a'], json.dumps(cov), flush=True)
# ---------------------------------------------------------------- JOINT preps
bj = J.manifest
for v, mode in (('G', 'global'), ('S', 'train_batch')):
    jd = Path(C.JOINT_PARENTS[f'{BASE}_lf{v}']['prep_dir'])
    if (jd / 'JOINT_PREP_MANIFEST.json').exists():
        continue
    jd.mkdir(parents=True, exist_ok=True); files = {}
    for (split, kind, p), mu in MU.items():
        f = jd / f'center_{split}_{kind}_peer{p}.npy'; np.save(f, mu.cpu().numpy()); files[f.name] = dict(sha256=sha256(f), shape=list(mu.shape))
    eq = {}
    for (split, kind, p), x in E.items():
        dl = (x - d2[(split, kind, p)]).abs(); eq[f'{split}_{kind}_{p}'] = dict(max_abs=float(dl.max()), mean_abs=float(dl.mean()))
    man = dict(line=f'R47 JOINT prep {BASE} variant {v}', created_utc=utc(), cond=f'{BASE}_lf{v}', target_cond=f'{BASE}_lf{v}', centring_mode=mode,
               parent=bj['parent'], target_cache_manifest_sha256=d2['manifest_sha256'], native_code=bj['native_code'], tokens=bj['tokens'], files=files,
               centring_rule=('ONE parent mean per split / modality / peer (the means of the layout-free cache)' if v == 'G' else
                              'student own-output means over the unique items of each training batch; evaluation: fixed random 512-item batches (seed [4701, split, modality, peer]); center_* files unused'),
               equivalence=eq, equivalence_summary=dict(max_abs_over_all=max(x['max_abs'] for x in eq.values())), base_joint_prep_manifest=bj.get('line'), test_guard=guard_record())
    write_json(jd / 'JOINT_PREP_MANIFEST.json', man); print('R47_LF_JOINT', BASE, v, man['equivalence_summary'], flush=True)
print('R47_BUILD_LF_DONE', BASE, 'seconds', time.time() - t0)
