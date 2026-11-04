"""R50 CPU real-data checks A-G (one tree / cond per process; CPU, deterministic; the R50 cache / prep / JOINT prep of the cond must exist).
  A  statistic order / chunk invariance on a real FIT subset (raw parent outputs encoded on CPU in natural order with chunks 512/2048 and in a random order
     with chunks 97/301; fp64 mean -> fp32): means and centred vectors compared; cross-device content: CPU normalize(e - mu_FIT) of the subset vs the R50
     GPU-built FIT cache rows; DEV: CPU normalize(e_dev - mu_FIT) of ALL DEV items vs the R50 DEV cache (and, as a discriminating control, the same items
     centred by their own DEV mean, which must NOT match the R50 cache).
  B/C/D/F  r50ext.joint_checks (production module) on CPU with the parent student.
  E  r50ext.head_checks with a nonzero random residual (frozen path on the cache, JOINT path on the student).
  F+ positive control: the R39 native per-batch parent centring vectors (row r = image r//PER_CAP / caption r, 128-row batches) with a CONSTANT student:
     distinct DEV vectors and the group / pair shortcut (top-1 caption in the image's native batch; exact pair R@1) vs the FIT-fixed rule (one vector, score
     range 0).
  D+ sampler renaming: FIT image IDs renamed by an explicit permutation (observed map, candidate tables, cache rows remapped synchronously): the N /
     N_PRESENT_ONLY lists, labels, present masks and p_d are identical up to the map; critic scores of the sampled pairs identical.
  G  r50ext.truth_check (DEV truth relabel -> identical top-100 tables) and, in a separate trainer process, importing the oracle module is refused.
Prints R50_CPU_REAL <json>. Usage: python -I -B tests/r50_cpu_real.py <tree> <cond>"""
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
W = Path(__file__).resolve().parents[1]
TREE, COND = sys.argv[1], sys.argv[2]
sys.path.insert(0, str(W / TREE))
import torch
import common as C
C.install_guards('trainer')
import train_unit as TU
TU.set_determinism()
import r39ext as R9, r50ext as R50, data36 as D36
from heads import build_heads
from data35 import shard_mm
from common import PEERS, N_TRAIN_IMG, N_TRAIN_CAP, N_DEV_IMG, N_DEV_CAP
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
dev = torch.device('cpu'); t0 = time.time(); res = {}
gen = np.random.default_rng(5051)
ctx = TU.Ctx(COND, dev, need_local=False); ctx.ensure_joint(); J = ctx.joint; d = ctx.d
stud = R9.build_students(J.ck, dev, ('img', 'txt'))
TOLV, TOLX = 1e-4, 5e-4   # amendment A1: same-device vector tolerance / cross-device (CPU vs GPU-built cache) tolerance
cmp = R50._cmp


def enc(p, split, kind, ids, ch):
    ids = torch.as_tensor(np.asarray(ids))
    with torch.no_grad():
        return torch.cat([J.raw_encode(stud[p], split, kind, ids[i:i + ch]) for i in range(0, len(ids), ch)])


# ------------------------------------------------------------------ A
A = {}
for p in PEERS:
    for kind, nf, nd, sub in (('img', N_TRAIN_IMG, N_DEV_IMG, 1200), ('cap', N_TRAIN_CAP, N_DEV_CAP, 3000)):
        ids = np.sort(gen.choice(nf, sub, replace=False)); perm = gen.permutation(sub); inv = np.argsort(perm)
        e1 = enc(p, 'fit', kind, ids, 512 if kind == 'img' else 2048); e2 = enc(p, 'fit', kind, ids[perm], 97 if kind == 'img' else 301)[torch.as_tensor(inv)]
        m1 = e1.double().mean(0, keepdim=True).float(); m2 = e2.double().mean(0, keepdim=True).float()
        mu = J.mu[('fit', kind, p)]
        rec = dict(subset_mean_order_chunk=cmp(m1, m2), subset_centred_order_chunk=cmp(J.l2norm(e1 - m1), J.l2norm(e2 - m2)),
                   fit_cache_rows_vs_cpu_fitcentred=cmp(J.l2norm(e1 - mu), d[('fit', kind, p)][torch.as_tensor(ids)]))
        ed = enc(p, 'dev', kind, np.arange(nd), 512 if kind == 'img' else 2048)
        rec['dev_cache_vs_cpu_fitcentred'] = cmp(J.l2norm(ed - mu), d[('dev', kind, p)])
        rec['dev_cache_vs_cpu_devmeancentred_control'] = cmp(J.l2norm(ed - ed.double().mean(0, keepdim=True).float()), d[('dev', kind, p)])
        rec['passed'] = bool(rec['subset_mean_order_chunk']['max_abs'] <= 1e-6 and rec['subset_centred_order_chunk']['max_abs'] <= TOLV and rec['fit_cache_rows_vs_cpu_fitcentred']['max_abs'] <= TOLX
                             and rec['dev_cache_vs_cpu_fitcentred']['max_abs'] <= TOLX and rec['dev_cache_vs_cpu_devmeancentred_control']['max_abs'] > 2 * TOLX)
        A[f'{kind}_{p}'] = rec
A['passed'] = all(v['passed'] for v in A.values()); res['A'] = A
# ------------------------------------------------------------------ B C D F (production r50ext on CPU) + P
res['P'] = R50.protocol_check(ctx)
jc = R50.joint_checks(ctx, stud, gen, final=True); jc['passed'] = bool(jc['passed_B_D'] and jc['passed_C'] and jc['passed_F']); res['BCDF'] = jc
# ------------------------------------------------------------------ E
heads = build_heads('G', d['a0b0']['a0'], d['a0b0']['b0'], 1001)
e_f = R50.head_checks(ctx, heads, None, gen, TU.g_pairs_lists); e_j = R50.head_checks(ctx, heads, stud, gen, TU.g_pairs_lists)
res['E'] = dict(frozen=e_f, joint=e_j, passed=bool(e_f['passed'] and e_j['passed']))
# ------------------------------------------------------------------ F+ positive control (R39 native per-batch centring)
nat_dir = Path(C.JOINT_PARENTS[COND.replace('_lfF', '')]['prep_dir'])
nm = json.loads((nat_dir / 'JOINT_PREP_MANIFEST.json').read_text())
F = dict(native_prep=str(nat_dir), native_mode=nm.get('centring_mode', 'native_batch'))
with torch.no_grad():
    for p in PEERS[:1]:
        mi = torch.from_numpy(np.load(nat_dir / f'center_dev_img_peer{p}.npy')); mc = torch.from_numpy(np.load(nat_dir / f'center_dev_cap_peer{p}.npy'))
        c = J.l2norm(torch.randn(1, mi.shape[1], generator=torch.Generator().manual_seed(7)))
        bi = R9.batch_index('img', torch.arange(N_DEV_IMG)); bc = R9.batch_index('cap', torch.arange(N_DEV_CAP))
        ui = J.l2norm(c - mi[bi]); uc = J.l2norm(c - mc[bc]); s = ui @ uc.T
        top = s.argmax(1); gt_b = R9.batch_index('cap', torch.arange(N_DEV_IMG) * C.PER_CAP)
        F['native_batch'] = dict(n_batches_img=int(mi.shape[0]), distinct_img_vectors=int(torch.unique(ui, dim=0).shape[0]), distinct_cap_vectors=int(torch.unique(uc, dim=0).shape[0]),
                                 top1_caption_in_true_caption_batch=float((bc[top] == gt_b).double().mean()), chance_batch_rate=float(128 / N_DEV_CAP),
                                 exact_pair_top1=float(((top // C.PER_CAP) == torch.arange(N_DEV_IMG)).double().mean()), score_range=float(s.max() - s.min()))
        fi = J.l2norm(c - J.mu[('dev', 'img', p)]).expand(N_DEV_IMG, -1); fc = J.l2norm(c - J.mu[('dev', 'cap', p)]).expand(N_DEV_CAP, -1); sf = fi @ fc.T
        F['fit_fixed'] = dict(distinct_img_vectors=int(torch.unique(fi, dim=0).shape[0]), distinct_cap_vectors=int(torch.unique(fc, dim=0).shape[0]), score_range=float(sf.max() - sf.min()))
F['passed'] = bool(F['fit_fixed']['distinct_img_vectors'] == 1 and F['fit_fixed']['distinct_cap_vectors'] == 1 and F['fit_fixed']['score_range'] == 0.0
                   and F['native_batch']['distinct_img_vectors'] > 1)
res['F_positive_control'] = F
# ------------------------------------------------------------------ D+ sampler renaming
obs = d['obs_np']; ci, ct = d['cand_i2t'].astype(np.int64), d['cand_t2i'].astype(np.int64)
pi = gen.permutation(N_TRAIN_IMG)
d2 = dict(d, obs_np=pi[obs], cand_i2t=np.empty_like(ci), cand_t2i=pi[ct])
d2['cand_i2t'][pi] = ci
mp = lambda a: np.where(a == D36.NULL_ID, D36.NULL_ID, pi[np.where(a == D36.NULL_ID, 0, a)])
Dp = {}
for sup in ('OBS_MISSING_NULL', 'N_PRESENT_ONLY', 'ORACLE_MISSING'):
    l1, i1 = D36.sample(sup, np.random.default_rng([1001, 3]), None, d, 16, None); l2, i2 = D36.sample(sup, np.random.default_rng([1001, 3]), None, d2, 16, None)
    (q1, c1, a1), (r1, m1, b1) = l1; (q2, c2, a2), (r2, m2, b2) = l2
    Dp[sup] = dict(indices_equal=all(np.array_equal(x, y) for x, y in zip(i1, i2)), i2t_query_mapped=bool(np.array_equal(pi[q1], q2)), i2t_caps_equal=bool(np.array_equal(c1, c2)),
                   t2i_imgs_mapped=bool(np.array_equal(mp(m1), m2)), labels_equal=bool(np.array_equal(a1, a2) and np.array_equal(b1, b2)),
                   present_masks_equal=all(np.array_equal(x, y) for x, y in zip(D36.valid_rows(l1), D36.valid_rows(l2))))
pp1, pp2 = D36.presence_probs(d), D36.presence_probs(d2)
Dp['presence_probs_equal'] = bool(pp1['p_i2t'] == pp2['p_i2t'] and pp1['p_t2i'] == pp2['p_t2i'])
(q1, c1, a1), (r1, m1, b1) = D36.sample('N_PRESENT_ONLY', np.random.default_rng([1001, 3]), None, d, 16, None)[0]
img1 = np.concatenate([np.repeat(q1, c1.shape[1]).reshape(c1.shape)[c1 != -1], m1[m1 != -1]]); cap1 = np.concatenate([c1[c1 != -1], np.repeat(r1, m1.shape[1]).reshape(m1.shape)[m1 != -1]])
d3 = dict(d)
for p in PEERS:
    X = d[('fit', 'img', p)]; Xr = torch.empty_like(X); Xr[torch.as_tensor(pi)] = X; d3[('fit', 'img', p)] = Xr
from types import SimpleNamespace
hr = R50.copy.deepcopy(heads['A'])
with torch.no_grad():
    gtor = torch.Generator().manual_seed(11); hr.out.weight.copy_(torch.randn(hr.out.weight.shape, generator=gtor) * .1)
    s1 = TU.g_pairs_lists(hr, None, ctx, 'A', 'fit', img1, cap1); s2 = TU.g_pairs_lists(hr, None, SimpleNamespace(d=d3, dev=dev, raw=None), 'A', 'fit', pi[img1], cap1)
Dp['critic_scores_renamed_max_abs'] = float((s1 - s2).abs().max())
Dp['passed'] = bool(all(all(v.values()) for k, v in Dp.items() if isinstance(v, dict)) and Dp['presence_probs_equal'] and Dp['critic_scores_renamed_max_abs'] <= TOLV)
res['D_sampler_renaming'] = Dp
# ------------------------------------------------------------------ G
g = R50.truth_check(ctx, gen)
probe = "import sys; sys.path.insert(0, sys.argv[1]); import common as C; C.install_guards('trainer')\ntry:\n    import oracle_truth; print('IMPORTED')\nexcept ImportError as e:\n    print('REFUSED', e)"
pr = subprocess.run([sys.executable, '-I', '-B', '-c', probe, str(W / 'r50f')], capture_output=True, text=True, cwd='/tmp')
g['oracle_import_in_trainer_process'] = pr.stdout.strip()[:300]
g['passed'] = bool(g['passed'] and pr.stdout.startswith('REFUSED'))
res['G'] = g
res['ALL_PASS'] = bool(all(v['passed'] for v in res.values() if isinstance(v, dict)))
res['seconds'] = time.time() - t0
print('R50_CPU_REAL ' + json.dumps(res, default=str), flush=True)
