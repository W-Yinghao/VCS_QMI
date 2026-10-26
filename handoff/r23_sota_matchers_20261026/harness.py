"""R23 task A harness: V10 BF-SHG problems through the host data guard, per-method CAL configs, LOCK, DEV, analysis.

  harness.py check                 CPU unit tests (convention, RANSAC, host error, problem identity, monomodal recovery)
  harness.py smoke  --method M     GPU: first config on 2 FIT monomodal + 2 CAL multimodal problems
  harness.py cal    --method M     all pre-registered configs on CAL8 x 10 (resumable)
  harness.py lock                  pre-registered selection on CAL -> LOCK.json (no DEV read)
  harness.py dev    --method M     locked config on DEV32 x 10 (resumable)
  harness.py analyze               RESULTS.json / RESULTS_CN.txt (paired with stored V10 DEV rows)
Roles: FIT / CAL / DEV only (host DataGuard refuses the held TEST cores).  No training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
sys.path.insert(0, str(BASE / 'ssl_pilot_v10_r22c/src')); sys.path.insert(0, str(HERE))
from v10 import adapter as A  # noqa: E402
import methods as MM  # noqa: E402

Q, P = A.Q, A.Q.P
DATA_LOCK = BASE / 'handoff/v10_bf1_20261017/DATA_LOCK.json'
DATA_LOCK_SHA = 'e65c60a33f095933e673613fc58f2754485933524259e548c6a0f5dc380f72c5'
FROZEN = ('methods.py', 'harness.py', 'PROTOCOL.json')
H_ = BASE / 'handoff'
V10 = {
    'MF_VCS_M1': [H_ / f'v10_bf1_20261017/results/FINAL_DEV/MF_VCS_M1/seed_{s}/DEV.jsonl' for s in (1001, 1002, 1003)],
    'MLP_VCS_perm8': [H_ / f'v10_method_pipeline_20261019/families/0/results/FINAL_DEV/MLP_VCS_perm8/seed_{s}/DEV.jsonl' for s in (1001, 1002, 1003)],
    'InfoNCE_host': [H_ / f'v10_method_pipeline_20261019/families/1/results/FINAL_DEV/InfoNCE_host[J_perm1]/seed_{s}/DEV.jsonl' for s in (1001, 1002, 1003)],
    'CoMIR_native': [H_ / f'v10_native_geometry_recovery_20261019/families/4/results/FINAL_DEV/CoMIR_native/seed_{s}/DEV.jsonl' for s in (1001, 1002, 1003)],
    'MI_native': [H_ / 'v10_native_geometry_recovery_20261019/families/2/results/FINAL_DEV/MI_native/seed_1001/DEV.jsonl'],
    'NMI_native': [H_ / 'v10_method_pipeline_20261019/families/3/results/FINAL_DEV/NMI_native/seed_1001/DEV.jsonl'],
    'MIND_2D_adapted': [H_ / 'v10_method_pipeline_20261019/families/6/results/FINAL_DEV/MIND_2D_adapted/seed_1001/DEV.jsonl'],
}
PAIRED = ('MF_VCS_M1', 'MLP_VCS_perm8', 'InfoNCE_host', 'CoMIR_native')
SPARSE = ('sp_lg', 'disk_lg', 'aliked_lg', 'minima_sp_lg', 'xfeat', 'xfeat_lg', 'minima_loftr')


def configs(method):
    if method in SPARSE:
        return {f'{p}_r{r}': dict(preproc=p, rot_step=r) for p in MM.PREPROC for r in (10, 5)}
    if method in ('roma', 'minima_roma'):
        return {f'{p}_r{r}': dict(preproc=p, rot_step=r) for p in MM.PREPROC for r in (15, 10)}
    if method == 'globalign':
        return {f'{p}_q{q}': dict(preproc=p, Q=q) for p in ('gray', 'log', 'iwm') for q in (8, 16)}
    if method == 'dinov2_host':
        return {f'{a}_s{s}': dict(arch=a, scale=float(s), ref='rgb', mov='int') for a in ('vits14', 'vitb14') for s in (1, 2)}
    raise ValueError(method)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()


def write(p, d):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_name('.' + p.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(d, indent=1, allow_nan=False) + '\n'); tmp.replace(p)


def rows(p):
    p = Path(p); return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def verify_freeze():
    fz = json.loads((HERE / 'FREEZE.json').read_text())
    for f, h in fz['files'].items():
        assert sha(HERE / f) == h, 'Frozen file changed: ' + f
    return sha(HERE / 'FREEZE.json')


def make_guard():
    assert sha(DATA_LOCK) == DATA_LOCK_SHA, 'BF data lock changed'
    g = A.DataGuard(); lock = json.loads(DATA_LOCK.read_text()); files = {r['core']: r for r in lock['files']}
    for role in ('FIT', 'CAL', 'DEV'):
        cores = g.included(role)
        assert sorted(c for c, r in files.items() if r['role'] == role) == cores, 'Data lock role set differs: ' + role
        for c in cores:
            assert sha(files[c]['path']) == files[c]['sha256'], 'BF input changed since the data lock: ' + c
    return g


class Problems:
    """Host V10 problems for one role: GuardedInputs moving (perturbed SHG intensity / log) and reference (BF luminance),
    raw BF RGB from the same host cache file; truth = Q.perturbations_p2(core)[i]."""

    def __init__(self, g, role, cores=None, identity=False):
        self.g, self.role = g, role; self.cores = list(cores) if cores is not None else g.included(role); self.pert = {}
        for c in self.cores:
            g.log_rule(c, {role}, f'R23 SOTA {role} frozen perturbation rule')
            self.pert[c] = [[0.0, 0.0, 0.0]] * P.N_PERT if identity else Q.perturbations_p2(c)
        self.inp = A.GuardedInputs(g, {role}, f'R23 SOTA {role} registration', self.pert, 'cpu'); self.rgb = {}

    def ref(self, c):
        lum = self.inp.reference(c)[0][0].numpy()
        if c not in self.rgb:
            self.rgb = {c: np.load(self.g.check(c, {self.role}, f'R23 SOTA {self.role}:raw BF RGB'))['bf_rgb']}
        return {'lum': lum, 'rgb': self.rgb[c]}

    def mov(self, c, i):
        a, b = self.inp.moving(c, i); self.inp.mov.pop((c, i), None)
        return {'int': a[0].numpy(), 'log': b[0].numpy()}

    def clear(self):
        self.inp.ref.clear(); self.inp.mov.clear()

    def all(self, n=P.N_PERT):
        return [(c, i) for c in self.cores for i in range(n)]


def mono(fit, ident, c, i):
    """Monomodal sanity problem: unperturbed SHG as the reference, the host-perturbed SHG as moving (convention check)."""
    m0 = ident.mov(c, 0); u = MM.u8(m0['int']).astype(np.float32) / 255
    return {'lum': m0['int'], 'rgb': np.repeat(u[..., None], 3, -1)}, fit.mov(c, i)


def ctx_of(c, i):
    return dict(seed=Q.seed_of(f'R23-SOTA|{c}|{i}') % (2 ** 31), theta_max=float(Q.THMAX), t_max=float(P.TMAX), Q=Q,
                rng=np.random.default_rng(Q.seed_of(f'R6-REG-P2|pos|{c}|{i}')))


def score(out, truth):
    th, ty, tx = truth
    return Q.corner_error_full(out['theta_deg'], np.array([out['ty'], out['tx']]), th, (ty, tx))


def run_one(m, method, cid, ref, mov, truth, c, i, role):
    ctx = ctx_of(c, i); torch.manual_seed(ctx['seed']); np.random.seed(ctx['seed'] % (2 ** 32))
    out = m.register(ref, mov, ctx); err = score(out, truth)
    assert all(math.isfinite(out[k]) for k in ('theta_deg', 'ty', 'tx')) and math.isfinite(err)
    return dict(method=method, config=cid, role=role, core=c, perturbation=i, truth=[float(x) for x in truth],
                theta_hat=out['theta_deg'], t_hat=[out['ty'], out['tx']], err_full=float(err), rot_err_deg=abs(out['theta_deg'] - truth[0]),
                ok=bool(out['ok']), n_matches=int(out['n_matches']), n_inliers=int(out['n_inliers']), alpha=out['alpha'],
                seconds=float(out['seconds']), info=out.get('info'))


def metrics(rs):
    e = np.array([r['err_full'] for r in rs])
    return dict(n=len(e), success42=float(np.mean(e < 42)), success9=float(np.mean(e < 9)), median=float(np.median(e)),
                p95=float(np.percentile(e, 95)), seconds=float(np.mean([r['seconds'] for r in rs])),
                fail=int(sum(not r['ok'] for r in rs)))


def run_set(method, cfgs, probs, out_dir, device, role):
    freeze = verify_freeze(); env = A.environment('cuda' if torch.cuda.is_available() else None)
    for cid, cfg in cfgs.items():
        path = Path(out_dir) / f'{cid}.jsonl'; done = {(r['core'], r['perturbation']) for r in rows(path)}
        todo = [p for p in probs.all() if p not in done]
        if not todo:
            continue
        m = MM.build(method, cfg, device); t0 = time.time(); last = None
        with path.open('a') as f:
            for c, i in todo:
                if c != last and last is not None:
                    probs.clear()
                last = c
                r = run_one(m, method, cid, probs.ref(c), probs.mov(c, i), probs.pert[c][i], c, i, role)
                r.update(job=os.environ.get('SLURM_JOB_ID'), freeze_sha256=freeze); f.write(json.dumps(r) + '\n'); f.flush()
        print(method, cid, role, metrics(rows(path)), f'{time.time() - t0:.0f}s', flush=True)
        del m; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    assert {x['role'] for x in probs.g.reads} <= {'FIT', 'CAL', 'DEV'}
    write(Path(out_dir) / 'ENV.json', dict(env=env, gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                                           job=os.environ.get('SLURM_JOB_ID'), roles_read=sorted({x['role'] for x in probs.g.reads})))


# ------------------------------------------------------------------------------------------------------------ modes
def cmd_check():
    import cv2
    freeze = verify_freeze(); res = {}
    rng = np.random.default_rng(0); c = np.array([255.5, 255.5])
    p = rng.uniform(0, 511, (300, 2)); th, t = 17.3, np.array([12.5, -30.25])
    a = math.radians(th); R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]); q = (p - c) @ R.T + c + t
    q[:120] = rng.uniform(0, 511, (120, 2))                                      # 40 % outliers
    r = MM.ransac_rigid(p, q, c, seed=1, tau=1.0); res['ransac'] = dict(theta=r[0], t=list(r[1]), inliers=r[2])
    assert abs(r[0] - th) < 1e-6 and np.allclose(r[1], t, atol=1e-6) and r[2] == 180
    g = make_guard(); fit_core = g.included('FIT')[0]
    ident = Problems(g, 'FIT', [fit_core], identity=True); img = MM.u8(ident.mov(fit_core, 0)['int'])
    imt = torch.from_numpy(img.astype(np.float32))[None]
    for al in (-23.0, 11.0):
        mine = MM.rot_img(img, al).astype(np.float32)
        host = Q.sample_pts(torch, torch.nn.functional, imt, Q.apply_T(torch, torch.tensor([math.radians(al)]), torch.zeros(1, 2),
                            Q.grid_coords(torch, 0, 511, 'cpu')))[0, :, 0].reshape(512, 512).numpy()
        inner = np.s_[64:448, 64:448]; d = np.abs(mine[inner] - host[inner]); res[f'rot_img_vs_host_{al}'] = dict(max=float(d.max()), mean=float(d.mean()))
        assert d.mean() < 0.6 and d.max() < 3.0, d.max()
    stored = rows(V10['MF_VCS_M1'][0]); bad_err = bad_truth = 0
    for s in stored:
        e = Q.corner_error_full(s['theta_hat'], np.array(s['t_hat']), s['theta'], tuple(s['t']))
        bad_err += e != s['err_full']
        tr = Q.perturbations_p2(s['core'])[s['perturbation']]; bad_truth += [tr[0], tr[1], tr[2]] != [s['theta'], *s['t']]
    res['host_error_recomputed'] = dict(n=len(stored), mismatches=int(bad_err)); res['truth_identity_MF_VCS_M1_1001'] = dict(n=len(stored), mismatches=int(bad_truth))
    assert bad_err == 0 and bad_truth == 0 and len(stored) == 320 and sorted({s['core'] for s in stored}) == g.included('DEV')
    for name, paths in V10.items():
        for pth in paths:
            rs = rows(pth); assert len(rs) == 320
            assert all(Q.perturbations_p2(s['core'])[s['perturbation']] == [s['theta'], *s['t']] for s in rs), pth
    res['truth_identity_all_refs'] = sum(len(v) for v in V10.values())

    class ORB(MM.Method):
        def __init__(self, cfg, device):
            super().__init__(cfg, device); self.o = cv2.ORB_create(4000); self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        def match(self, a, b):
            ka, da = self.o.detectAndCompute(a, None); kb, db = self.o.detectAndCompute(b, None)
            if da is None or db is None:
                return np.zeros((0, 2)), np.zeros((0, 2))
            ms = self.bf.match(da, db)
            return (np.array([ka[m.queryIdx].pt[::-1] for m in ms]), np.array([kb[m.trainIdx].pt[::-1] for m in ms]))
    fit = Problems(g, 'FIT', [fit_core]); orb = ORB(dict(preproc='gray', rot_step=10), 'cpu'); errs = []
    for i in range(3):
        ref, mov = mono(fit, ident, fit_core, i); out = orb.register(ref, mov, ctx_of(fit_core, i)); errs.append(score(out, fit.pert[fit_core][i]))
    res['mono_orb_err_full'] = errs; assert max(errs) < 3.0, errs
    gl = MM.build('globalign', dict(preproc='gray', Q=16, angles_n=31, refine_n=8), 'cpu')
    ref, mov = mono(fit, ident, fit_core, 1); out = gl.register(ref, mov, ctx_of(fit_core, 1)); res['mono_globalign_err_full'] = score(out, fit.pert[fit_core][1])
    assert res['mono_globalign_err_full'] < 12.0, res['mono_globalign_err_full']
    assert {x['role'] for x in g.reads} <= {'FIT', 'DEV'}
    write(HERE / 'CPU_CHECK.json', dict(status='PASS', freeze_sha256=freeze, job=os.environ.get('SLURM_JOB_ID'), results=res))
    print('CPU_CHECK PASS', json.dumps(res)[:600], flush=True)


def cmd_smoke(method):
    freeze = verify_freeze(); g = make_guard(); dev = 'cuda'; cid, cfg = next(iter(configs(method).items()))
    fit_core = g.included('FIT')[1]; ident = Problems(g, 'FIT', [fit_core], identity=True); fit = Problems(g, 'FIT', [fit_core])
    cal = Problems(g, 'CAL', g.included('CAL')[:2]); m = MM.build(method, cfg, dev); out = []
    for i in (0, 4):
        ref, mov = mono(fit, ident, fit_core, i); out.append(dict(kind='mono', **run_one(m, method, cid, ref, mov, fit.pert[fit_core][i], fit_core, i, 'FIT')))
    for c in cal.cores:
        out.append(dict(kind='cal', **run_one(m, method, cid, cal.ref(c), cal.mov(c, 0), cal.pert[c][0], c, 0, 'CAL')))
    mono_err = [r['err_full'] for r in out if r['kind'] == 'mono']
    status = 'PASS' if max(mono_err) < 9 else 'FAIL_MONO'
    write(HERE / 'smoke' / f'{method}.json', dict(status=status, method=method, config=cid, freeze_sha256=freeze, job=os.environ.get('SLURM_JOB_ID'),
                                                  gpu=torch.cuda.get_device_name(), rows=out, note=getattr(m, 'load_note', None)))
    print('SMOKE', method, status, [(r['kind'], round(r['err_full'], 2), round(r['seconds'], 2), r['n_inliers']) for r in out], flush=True)


def cmd_cal(method):
    g = make_guard(); run_set(method, configs(method), Problems(g, 'CAL'), HERE / 'cal' / method, 'cuda', 'CAL')


def cmd_lock():
    freeze = verify_freeze(); out = {}
    for method in MM.FAMILIES:
        cf = configs(method); res = {}
        for cid in cf:
            rs = rows(HERE / 'cal' / method / f'{cid}.jsonl')
            if len(rs) == 80:
                res[cid] = metrics(rs)
        if len(res) != len(cf):
            out[method] = dict(status='INCOMPLETE', have=sorted(res)); continue
        best = max(res, key=lambda k: (res[k]['success42'], res[k]['success9'], -res[k]['median'], -res[k]['seconds']))
        out[method] = dict(status='LOCKED', config_id=best, config=cf[best], cal=res)
    write(HERE / 'LOCK.json', dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), freeze_sha256=freeze,
                                   rule='max CAL success42, then success9, then -median, then -seconds', methods=out, dev_read=False))
    print(json.dumps({k: (v['status'], v.get('config_id')) for k, v in out.items()}, indent=1))


def cmd_dev(method):
    lk = json.loads((HERE / 'LOCK.json').read_text())['methods'][method]; assert lk['status'] == 'LOCKED'
    g = make_guard(); run_set(method, {lk['config_id']: lk['config']}, Problems(g, 'DEV'), HERE / 'dev' / method, 'cuda', 'DEV')


def boot(d, B=10000, seed=20261026):
    r = np.random.default_rng(seed); n = len(d); m = d[r.integers(0, n, (B, n))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def cmd_analyze():
    lk = json.loads((HERE / 'LOCK.json').read_text())['methods']; out = dict(methods={}, references={}, paired={}); cores = None
    def core_rates(rs, thr):
        by = {}
        for r in rs: by.setdefault(r['core'], []).append(r['err_full'] < thr)
        return {c: float(np.mean(v)) for c, v in by.items()}
    ref_rates = {}
    for name, paths in V10.items():
        allr = [r for p in paths for r in rows(p)]
        out['references'][name] = dict(seeds=len(paths), files_sha256=[sha(p) for p in paths], **metrics([dict(err_full=r['err_full'], seconds=r.get('seconds', 0.0), ok=True) for r in allr]))
        per = [core_rates(rows(p), 9) for p in paths]; per42 = [core_rates(rows(p), 42) for p in paths]
        ref_rates[name] = ({c: float(np.mean([x[c] for x in per])) for c in per[0]}, {c: float(np.mean([x[c] for x in per42])) for c in per42[0]})
        cores = sorted(per[0])
    for method, v in lk.items():
        if v['status'] != 'LOCKED':
            continue
        rs = rows(HERE / 'dev' / method / f"{v['config_id']}.jsonl")
        if len(rs) != 320:
            out['methods'][method] = dict(status='INCOMPLETE', n=len(rs)); continue
        assert sorted({r['core'] for r in rs}) == cores
        out['methods'][method] = dict(status='COMPLETE', config_id=v['config_id'], **metrics(rs))
        s9, s42 = core_rates(rs, 9), core_rates(rs, 42); out['paired'][method] = {}
        for ref in PAIRED:
            d9 = np.array([s9[c] - ref_rates[ref][0][c] for c in cores]); d42 = np.array([s42[c] - ref_rates[ref][1][c] for c in cores])
            out['paired'][method][ref] = dict(strict=dict(mean=float(d9.mean()), ci95=boot(d9)), loose=dict(mean=float(d42.mean()), ci95=boot(d42)))
    gpus = {}
    for d in (HERE / 'cal').glob('*/ENV.json'):
        gpus[d.parent.name + ':cal'] = json.loads(d.read_text()).get('gpu')
    for d in (HERE / 'dev').glob('*/ENV.json'):
        gpus[d.parent.name + ':dev'] = json.loads(d.read_text()).get('gpu')
    out['gpus'] = gpus; out['utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()); out['freeze_sha256'] = verify_freeze()
    write(HERE / 'RESULTS.json', out)
    L = ['R23 task A: modern matcher baselines on the V10 BF-SHG DEV32 protocol (DEV, 320 problems; TEST not read)', '',
         'method              config        strict<9  loose<42  median   p95      s/problem  fail']
    for k, v in sorted(out['references'].items(), key=lambda kv: -kv[1]['success9']):
        L.append(f'V10 {k:<16}{"(" + str(v["seeds"]) + " seeds)":<14}{v["success9"]:.3f}     {v["success42"]:.3f}     {v["median"]:<8.2f} {v["p95"]:<8.1f}')
    for k, v in sorted(out['methods'].items(), key=lambda kv: -kv[1].get('success9', -1)):
        if v['status'] == 'COMPLETE':
            L.append(f'{k:<20}{v["config_id"]:<14}{v["success9"]:.3f}     {v["success42"]:.3f}     {v["median"]:<8.2f} {v["p95"]:<8.1f} {v["seconds"]:<10.2f} {v["fail"]}')
    L += ['', 'paired core-level differences (method - reference), mean [bootstrap 95% over 32 cores]']
    for k, v in out['paired'].items():
        L.append(k + ': ' + '; '.join(f"{r} strict {x['strict']['mean']:+.3f} [{x['strict']['ci95'][0]:+.3f},{x['strict']['ci95'][1]:+.3f}] loose {x['loose']['mean']:+.3f}" for r, x in v.items()))
    (HERE / 'RESULTS_CN.txt').write_text('\n'.join(L) + '\n'); print('\n'.join(L))


def cmd_freeze():
    write(HERE / 'FREEZE.json', dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), files={f: sha(HERE / f) for f in FROZEN}))
    print('FREEZE', sha(HERE / 'FREEZE.json'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['check', 'smoke', 'cal', 'lock', 'dev', 'analyze', 'freeze'])
    ap.add_argument('--method', choices=list(MM.FAMILIES)); a = ap.parse_args()
    {'check': cmd_check, 'lock': cmd_lock, 'analyze': cmd_analyze, 'freeze': cmd_freeze}.get(a.mode, lambda: None)()
    if a.mode == 'smoke': cmd_smoke(a.method)
    if a.mode == 'cal': cmd_cal(a.method)
    if a.mode == 'dev': cmd_dev(a.method)
