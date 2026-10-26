"""R23 task B runner: CoMIR / MultiRegEval patch protocols on DEV (CAL for baseline tuning).

  run.py check                                  CPU gates (host equivalence, geometry, metric, SIFT/SimpleITK conventions)
  run.py eval --family F --seed S [--norm patch|core] [--protocols comir,mre] [--role DEV] [--limit N]
  run.py mi_sitk [--role DEV] [--workers 32]    SimpleITK Mattes MI on full-res patches (CPU)
DEV/CAL only: the BF DataGuard refuses the held TEST cores; nothing reads data/bf_shg_4550300/closed."""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np
from common import (HERE, BASE, R, A, torch, F, CP, Q, P, TR, SREG, MIND, CEN, WIN, MOV_MARGIN, THETAS, TS, T_CLAMP, TH_CLAMP,
                    PROTOCOLS, THR, K_DRAWS, guard, problems, draw, corner_error, corner_error_direct, render_quarter, render_full,
                    positions, write, append, sha)
from energy import PatchEnergy, mf_scorer, mlp_scorer, mse_scorer, mi_scorer, optimise

UNITS = BASE / 'ssl_pilot_v10_r22c/v10/r22_pmi/units'
MANIFEST = BASE / 'ssl_pilot_v10_test_repeat/v10/test_repeat134/TEST_REPEAT134_MANIFEST.json'   # selections only (no TEST data)
LEARNED = {'MF_VCS_M1': 'M1', 'MLP_VCS_perm8': 'MLPp8', 'InfoNCE_host[J_perm1]': 'InfoNCE'}
SUPP = ('CoMIR_native', 'MI_native', 'MIND_2D_adapted')
OUT = HERE / 'runs'


def selection(family, seed):
    for u in json.loads(MANIFEST.read_text())['units']:
        if u['family'] == family and u['seed'] == seed:
            return u
    raise KeyError((family, seed))


def build(family, seed, g, device):
    if family in LEARNED:
        u = R.load_unit(UNITS / f'r22_pmi_bf_{LEARNED[family]}_s{seed}.json')
        t, meta = R.build_trainer(u, g, None, device)
        return dict(family=family, seed=seed, trainer=t, meta=dict(meta, checkpoint_sha256=u['checkpoint_sha256'], step=u['selected_step']))
    s = selection(family, seed); d = Path(s['unit_dir']); done = json.loads((d / 'COMPLETE.json').read_text())
    ident = done['identity']; assert sha(d / 'COMPLETE.json') == s['complete_sha256']
    if family == 'CoMIR_native':
        t = TR.new_trainer(family, ident['candidate'], seed, g, device)
        meta = CP.load(t, d / s['checkpoint']['path'], ident); assert meta['sha256'] == s['checkpoint']['sha256']
        return dict(family=family, seed=seed, trainer=t, meta=dict(checkpoint_sha256=meta['sha256'], step=meta['step']))
    return dict(family=family, seed=seed, trainer=None, candidate=ident['candidate'], meta=dict(step=0))


def pct_range(a):
    lo, hi = np.percentile(np.asarray(a), (0.5, 99.5)); assert hi > lo; return float(lo), float(hi)


def make_energy(model, inp, x, perms, host=False, host_ranges=None):
    fam, t = model['family'], model['trainer']; xt = torch.as_tensor(x, dtype=torch.float32, device=inp['mov_log'].device)
    with torch.no_grad():
        if fam in LEARNED:
            u = P.sample_map(torch, F, t.eU(inp['ref_rgb'][None])[0], xt)
            sc = mf_scorer(t.head) if fam == 'MF_VCS_M1' else mlp_scorer(t.critic, 8 if fam == 'MLP_VCS_perm8' else 1)
            return PatchEnergy('learned', x, perms, ref_feat=u, mov_map=inp['mov_log'], scorer=sc, eV=t.eV,
                               mov_margin=MOV_MARGIN['learned'], host=host)
        if fam == 'CoMIR_native':
            rl, ml = t.reference_encoder(inp['ref_rgb'][None])[0], t.moving_encoder(inp['mov_log'][None])[0]
            return PatchEnergy('sample', x, perms, ref_feat=Q.sample_pts(torch, F, rl, xt[None])[0], mov_map=ml, scorer=mse_scorer(True),
                               mov_margin=MOV_MARGIN['comir'], host=host), (rl, ml)
        if fam == 'MIND_2D_adapted':
            cfg = MIND.MINDConfig(**model['candidate']['descriptor'])
            dr, dm = MIND.mind2d(inp['ref_int'], cfg), MIND.mind2d(inp['mov_int'], cfg)
            return PatchEnergy('sample', x, perms, ref_feat=Q.sample_pts(torch, F, dr, xt[None])[0], mov_map=dm, scorer=mse_scorer(False),
                               mov_margin=MOV_MARGIN['mind'], host=host)
        if fam == 'MI_native':
            ranges = host_ranges or (pct_range(inp['patch']['ref_int']), pct_range(inp['patch']['mov_int']))
            return PatchEnergy('sample', x, perms, ref_feat=Q.sample_pts(torch, F, inp['ref_int'], xt[None])[0], mov_map=inp['mov_int'],
                               scorer=mi_scorer(ranges), mov_margin=MOV_MARGIN['mi'], host=host)
    raise ValueError(fam)


# ------------------------------------------------------------------------------------------------------------ SIFT on CoMIR
SIFT_GRID = [dict(up=u, thr=r) for u in (1, 2, 4) for r in (3.0, 6.0)]


def sift_register(ref_lat, mov_lat, up, thr, ratio=0.8):
    """OpenCV SIFT + RANSAC similarity on the CoMIR latent patches (stand-in for the paper's Fiji SIFT) -> (theta, ty, tx) | None."""
    import cv2
    def to8(a):
        lo, hi = np.percentile(a, (0.5, 99.5)); return np.clip((a - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    im = [cv2.resize(to8(a), None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC) if up != 1 else to8(a) for a in (ref_lat, mov_lat)]
    sift = cv2.SIFT_create(); (k1, d1), (k2, d2) = sift.detectAndCompute(im[0], None), sift.detectAndCompute(im[1], None)
    if d1 is None or d2 is None or len(k1) < 3 or len(k2) < 3:
        return None, 0
    good = [m for m, n in (p for p in cv2.BFMatcher().knnMatch(d1, d2, k=2) if len(p) == 2) if m.distance < ratio * n.distance]
    if len(good) < 3:
        return None, len(good)
    back = lambda pts: (np.float32(pts) + 0.5) / up - 0.5                     # resize pixel-centre convention -> patch px
    src, dst = back([k1[m.queryIdx].pt for m in good]), back([k2[m.trainIdx].pt for m in good])
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=thr / up, maxIters=5000, confidence=0.999)
    if M is None:
        return None, len(good)
    th = math.degrees(math.atan2(M[0, 1], M[0, 0]))                         # host (y,x) R(theta) == [[c, s], [-s, c]] in (x,y)
    c = CEN - WIN[0]; mx, my = M @ np.array([c, c, 1.0])
    return (th, my - c, mx - c), int(inl.sum()) if inl is not None else 0


def latent_patch(lat):
    return lat[0, WIN[0]:WIN[1] + 1, WIN[0]:WIN[1] + 1].detach().cpu().numpy().astype(np.float64)


# ------------------------------------------------------------------------------------------------------------ SimpleITK MI
MI_VARIANTS = {
    'reg': dict(prep='raw', bins=32, shrink=[8, 4, 2], sigmas=[4.0, 2.0, 1.0], starts=(-30.0, -15.0, 0.0, 15.0, 30.0)),
    'log421': dict(prep='log', bins=32, shrink=[4, 2, 1], sigmas=[2.0, 1.0, 0.0], starts=(-30.0, -15.0, 0.0, 15.0, 30.0)),
    'log421_b64': dict(prep='log', bins=64, shrink=[4, 2, 1], sigmas=[2.0, 1.0, 0.0], starts=(-30.0, -15.0, 0.0, 15.0, 30.0)),
    'rank421': dict(prep='rank', bins=32, shrink=[4, 2, 1], sigmas=[2.0, 1.0, 0.0], starts=(-30.0, -15.0, 0.0, 15.0, 30.0)),
    'log421_s9': dict(prep='log', bins=32, shrink=[4, 2, 1], sigmas=[2.0, 1.0, 0.0], starts=tuple(np.arange(-30.0, 30.1, 7.5))),
}


def mi_prep(ref, mov, prep):
    if prep == 'raw':
        return ref, mov
    if prep == 'log':
        return ref, np.log1p(mov * 1000.0).astype(np.float32)
    if prep == 'rank':
        rk = lambda a: (np.argsort(np.argsort(a.ravel())).reshape(a.shape) / (a.size - 1)).astype(np.float32)
        return rk(ref), rk(mov)
    raise ValueError(prep)


def sitk_mi(ref, mov, starts=(-30.0, -15.0, 0.0, 15.0, 30.0), seed=1234, variant='reg'):
    import SimpleITK as sitk
    V = MI_VARIANTS[variant]; starts = V['starts']; ref, mov = mi_prep(ref, mov, V['prep'])
    fixed, moving = sitk.GetImageFromArray(ref), sitk.GetImageFromArray(mov); c = ((ref.shape[1] - 1) / 2.0, (ref.shape[0] - 1) / 2.0)
    best = None
    for a0 in starts:
        T = sitk.Euler2DTransform(); T.SetCenter(c); T.SetAngle(math.radians(-a0))
        Rg = sitk.ImageRegistrationMethod()
        Rg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=V['bins'])
        Rg.SetMetricSamplingStrategy(Rg.RANDOM); Rg.SetMetricSamplingPercentage(0.25, seed)
        Rg.SetInterpolator(sitk.sitkLinear)
        Rg.SetOptimizerAsRegularStepGradientDescent(learningRate=4.0, minStep=1e-3, numberOfIterations=200, relaxationFactor=0.5)
        Rg.SetOptimizerScalesFromPhysicalShift()
        Rg.SetShrinkFactorsPerLevel(V['shrink']); Rg.SetSmoothingSigmasPerLevel(V['sigmas']); Rg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
        Rg.SetInitialTransform(T, inPlace=True)
        try:
            Rg.Execute(fixed, moving); val = Rg.GetMetricValue()
        except RuntimeError:
            continue
        if best is None or val < best[0]:
            tx, ty = T.GetTranslation(); best = (val, -math.degrees(T.GetAngle()), ty, tx, a0)
    return best


# ------------------------------------------------------------------------------------------------------------ check
def check():
    torch.set_num_threads(8); g = guard(); dev = torch.device('cpu'); out = {}
    cal = g.included('CAL'); dv = g.included('DEV'); assert len(cal) == 8 and len(dv) == 32 and not set(cal) & set(dv)
    # (1) sampling ranges / reproducibility
    for c in cal:
        for k in range(K_DRAWS):
            a = draw('comir', c, k); assert abs(a['theta']) <= 30 and max(abs(a['ty_full']), abs(a['tx_full'])) <= 100 and a == draw('comir', c, k)
            b = draw('mre', c, k); L = b['level']; assert 5 * (L - 1) <= abs(b['theta']) <= 5 * L and all(20 * (L - 1) <= abs(b[q]) <= 20 * L for q in ('ty_full', 'tx_full'))
    # (2) corner error: host == direct corner coordinates; zero at truth
    rng = np.random.default_rng(0); mx = 0.0
    for _ in range(200):
        e = [rng.uniform(-30, 30), rng.uniform(-30, 30), rng.uniform(-30, 30)]; t = [rng.uniform(-30, 30), rng.uniform(-25, 25), rng.uniform(-25, 25)]
        mx = max(mx, abs(corner_error(e, t) - corner_error_direct(e, t)))
    assert mx < 1e-9 and corner_error(t, t) == 0.0; out['corner_error_max_abs_diff'] = mx
    # (3) rendering: quarter-frame patch vs full-res warp-then-crop (area 4x4) agree; truth maps reference to moving
    prob = draw('comir', cal[0], 0); qn = render_quarter(g, 'CAL', prob, dev, norm='core'); ref_f, mov_f = render_full(g, 'CAL', prob)
    shl = P.load_img(g.r6_root, cal[0], ('shg_log',))[0]
    th, ty, tx = prob['truth']; full_q = Q.perturb_map(torch, F, torch.tensor(np.expm1(shl.astype(np.float64)), dtype=torch.float32)[None], th, (ty, tx))[0].numpy()
    a = P.area_down(mov_f[1:833, 1:833].astype(np.float64)); b = full_q[WIN[0]:WIN[0] + 208, WIN[0]:WIN[0] + 208]
    corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1]); assert corr > 0.95, corr; out['quarter_vs_full_corr'] = corr
    # (4) host equivalence of every energy kind with the mask disabled (host positions, boxes, canvases), CAL problem 0
    core = cal[0]; pert = {core: Q.perturbations_p2(core)}; inputs = A.GuardedInputs(g, {'CAL'}, 'R23-B host equivalence check', pert, dev)
    ref_i, ref_r = inputs.reference(core); mov_i, mov_l = inputs.moving(core, 0)
    host_inp = dict(ref_int=ref_i, ref_rgb=ref_r, mov_int=mov_i, mov_log=mov_l, patch=None)
    th0, t0 = pert[core][0][0], pert[core][0][1:]
    cand_th = torch.tensor([math.radians(th0), math.radians(th0 + 2.0), math.radians(th0 - 5.0)]); cand_t = torch.tensor([t0, [t0[0] + 3, t0[1] - 2], [t0[0] - 6, t0[1] + 1]], dtype=torch.float32)
    ts = torch.tensor(TS[::11], dtype=torch.float32); eq = {}
    for fam, seed in (('MF_VCS_M1', 1001), ('MLP_VCS_perm8', 1001), ('InfoNCE_host[J_perm1]', 1001), ('CoMIR_native', 1001), ('MIND_2D_adapted', 1001), ('MI_native', 1001)):
        model = build(fam, seed, g, dev); rng_h = np.random.default_rng(Q.seed_of(f'R6-REG-P2|pos|{core}|0'))
        ctx = CP.evaluation_state(model['trainer']) if model['trainer'] is not None else _null()
        with ctx, torch.no_grad():
            t = model['trainer']
            if fam == 'MF_VCS_M1':
                href = A.reference_descriptors(t.eU, inputs, core); hE = A.MFRigidEnergy(href, mov_l, rng_h, eV=t.eV, head=t.head, estimator='moments')
                perms = hE.perm[None]
            elif fam in LEARNED:
                from v10.supplement.learned import MLPRigidEnergy
                hE = MLPRigidEnergy(t.eU(ref_r[None])[0], mov_l, rng_h, encoder=t.eV, critic=t.critic, draws=8 if fam == 'MLP_VCS_perm8' else 1,
                                    extra_seed=Q.seed_of(f'V10|MLP-extra-registration-perms|{core}|0'))
                perms = hE.permutations
            elif fam == 'CoMIR_native':
                hE = Q.RigidEnergy(torch, F, 'comir', t.reference_encoder(ref_r[None])[0], t.moving_encoder(mov_l[None])[0], rng_h); perms = hE.perm[None]
            elif fam == 'MIND_2D_adapted':
                cfg = MIND.MINDConfig(**model['candidate']['descriptor'])
                hE = MIND.DescriptorRigidEnergy(MIND.mind2d(ref_i, cfg), MIND.mind2d(mov_i, cfg), rng_h); perms = hE.perm[None]
            else:
                rg = (SREG.intensity_range(ref_i), SREG.intensity_range(mov_i))
                hE = Q.RigidEnergy(torch, F, 'mi', ref_i, mov_i, rng_h, ranges=rg); perms = hE.perm[None]
            pE = make_energy(model, host_inp, hE.x.numpy(), perms.numpy(), host=True, host_ranges=rg if fam == 'MI_native' else None)
            pE = pE[0] if isinstance(pE, tuple) else pE
            hb, pb = hE.batch(cand_th, cand_t), pE.batch(cand_th, cand_t)
            hg, pg = hE.grid_values(float(THETAS[7]), ts), pE.grid_values(float(THETAS[7]), ts)
            d = max(float((hb - pb).abs().max()), float((hg - pg).abs().max())); scale = float(torch.cat([hb, hg]).abs().max())
            assert torch.allclose(hb, pb, atol=1e-5, rtol=1e-4) and torch.allclose(hg, pg, atol=1e-5, rtol=1e-4), (fam, d)
            eq[fam] = dict(max_abs_diff=d, scale=scale)
            # masked patch variant on the patch protocol: finite, all positions valid at the truth, truth beats a 6-px shift
            pp = draw('comir', core, 0); inp = render_quarter(g, 'CAL', pp, dev); x, pm = positions(pp)
            pe = make_energy(model, inp, x, pm); pe = pe[0] if isinstance(pe, tuple) else pe
            tr = pp['truth']; tt = torch.tensor([[tr[1], tr[2]], [tr[1] + 6, tr[2]]], dtype=torch.float32); thh = torch.tensor([math.radians(tr[0])] * 2)
            v = pe.batch(thh, tt); mvalid = float(pe.mask(thh[:1], tt[:1]).mean())
            assert torch.isfinite(v).all(); eq[fam].update(patch_truth=float(v[0]), patch_shift6=float(v[1]), truth_valid_fraction=mvalid)
            del model
    out['host_equivalence'] = eq
    # (5) SIFT and SimpleITK conventions on a monomodal synthetic (BF luminance warped with a known transform)
    lum = P.load_img(g.r6_root, cal[1], ('bf',))[0]; g.check(cal[1], {'CAL'}, 'R23-B convention check')
    syn = (12.0, 9.0, -14.0); w = Q.perturb_map(torch, F, torch.tensor(lum, dtype=torch.float32)[None], syn[0], syn[1:])[0].numpy()
    ref_p, mov_p = lum[WIN[0]:WIN[1] + 1, WIN[0]:WIN[1] + 1], w[WIN[0]:WIN[1] + 1, WIN[0]:WIN[1] + 1]
    est, n = sift_register(ref_p, mov_p, 2, 3.0); assert est is not None and corner_error(est, syn) < 9, (est, n); out['sift_convention_err'] = corner_error(est, syn)
    big = P.read_tiff(P.core_files()[cal[1]][1]).astype(np.float64) / 255.0; L = 0.299 * big[..., 0] + 0.587 * big[..., 1] + 0.114 * big[..., 2]
    sp = dict(theta=12.0, ty_full=36.0, tx_full=-56.0)
    from scipy.ndimage import map_coordinates
    ys, xs = np.mgrid[607:1441, 607:1441].astype(np.float64); r = math.radians(sp['theta']); c_, s_ = math.cos(-r), math.sin(-r)
    zy, zx = ys - 1023.5 - sp['ty_full'], xs - 1023.5 - sp['tx_full']
    movf = map_coordinates(L, [c_ * zy - s_ * zx + 1023.5, s_ * zy + c_ * zx + 1023.5], order=1)
    b = sitk_mi(L[607:1441, 607:1441].astype(np.float32), movf.astype(np.float32), variant='reg')
    est = (b[1], b[2] / 4, b[3] / 4); assert corner_error(est, (12.0, 9.0, -14.0)) < 9, b; out['sitk_convention_err'] = corner_error(est, (12.0, 9.0, -14.0))
    roles = sorted({r['role'] for r in g.reads}) if hasattr(g, 'reads') else None
    assert roles is None or set(roles) <= {'FIT', 'CAL', 'DEV'}, roles
    out.update(status='PASS', roles_read=roles, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    write(HERE / 'CPU_CHECK.json', out); print('R23B_CHECK PASS', json.dumps(out)[:2000], flush=True)


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


# ------------------------------------------------------------------------------------------------------------ eval
def eval_unit(family, seed, norm, protocols, role, limit=None, tag=''):
    assert role in ('DEV', 'CAL')
    assert json.loads((HERE / 'CPU_CHECK.json').read_text())['status'] == 'PASS' and (HERE / 'PROTOCOL.json').exists()
    dev = torch.device('cuda'); torch.set_num_threads(4); g = guard(); model = build(family, seed, g, dev); started = time.time()
    name = f"{family.replace('[', '_').replace(']', '')}_s{seed}_{norm}_{role}{tag}"; out = OUT / name; out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'COMPLETE.json').exists(), 'no silent rerun'
    job = os.environ.get('SLURM_JOB_ID', 'local'); write(out / f'ATTEMPT_{job}.json', dict(status='RUNNING', start=started))
    rows_path = out / 'ROWS.jsonl'; done = set()
    if rows_path.exists():
        done = {(r['protocol'], r['core'], r['k'], r['method']) for r in map(json.loads, open(rows_path))}
    ctx = CP.evaluation_state(model['trainer']) if model['trainer'] is not None else _null()
    sift_cfg = None
    with ctx:
        if family == 'CoMIR_native':                                         # tune SIFT on CAL (CoMIR protocol) before any DEV row
            cal_rows = {i: [] for i in range(len(SIFT_GRID))}
            for prob in problems(g, 'CAL', 'comir'):
                inp = render_quarter(g, 'CAL', prob, dev, norm); x, pm = positions(prob)
                _, (rl, ml) = make_energy(model, inp, x, pm); a, b = latent_patch(rl), latent_patch(ml)
                for i, cfg in enumerate(SIFT_GRID):
                    est, _ = sift_register(a, b, cfg['up'], cfg['thr']); e = corner_error(est, prob['truth']) if est else corner_error((0, 0, 0), prob['truth'])
                    cal_rows[i].append(e)
            score = lambda e: (np.mean(np.array(e) < 9), np.mean(np.array(e) < 42), -np.median(e))
            best = max(cal_rows, key=lambda i: score(cal_rows[i])); sift_cfg = dict(SIFT_GRID[best], cal=[dict(SIFT_GRID[i], s9=float(np.mean(np.array(v) < 9)), s42=float(np.mean(np.array(v) < 42))) for i, v in cal_rows.items()])
            write(out / 'SIFT_SELECTION.json', sift_cfg)
        for protocol in protocols:
            probs = problems(g, role, protocol)[:limit] if limit else problems(g, role, protocol)
            for prob in probs:
                if (protocol, prob['core'], prob['k'], family) in done:
                    continue
                inp = render_quarter(g, role, prob, dev, norm); x, pm = positions(prob)
                en = make_energy(model, inp, x, pm); lat = None
                if isinstance(en, tuple): en, lat = en
                torch.cuda.synchronize(); t0 = time.perf_counter()
                th, tv, ng, fhat = optimise(en, dev, THETAS, TS, T_CLAMP, TH_CLAMP)
                torch.cuda.synchronize(); sec = time.perf_counter() - t0
                tr = prob['truth']
                with torch.no_grad():
                    ftrue = float(en.batch(torch.tensor([math.radians(tr[0])], device=dev), torch.tensor([[tr[1], tr[2]]], dtype=torch.float32, device=dev))[0])
                err = corner_error((th, *tv), tr)
                row = dict(protocol=protocol, core=prob['core'], k=prob['k'], level=prob['level'], method=family, seed=seed, norm=norm,
                           truth=tr, est=[th, float(tv[0]), float(tv[1])], err_full=err, F_hat=fhat, F_true=ftrue, n_grid=ng, seconds=sec)
                assert math.isfinite(err) and math.isfinite(fhat)
                append(rows_path, row)
                if lat is not None:
                    est, nm = sift_register(latent_patch(lat[0]), latent_patch(lat[1]), sift_cfg['up'], sift_cfg['thr'])
                    e2 = corner_error(est, tr) if est else corner_error((0, 0, 0), tr)
                    append(rows_path, dict(row, method='CoMIR_SIFT', est=list(est) if est else None, err_full=e2, F_hat=None, F_true=None,
                                           n_grid=None, seconds=None, sift_matches=nm, sift_failed=est is None))
    rows = [json.loads(l) for l in open(rows_path)]
    roles = sorted({r['role'] for r in g.reads}) if hasattr(g, 'reads') else None
    assert roles is None or set(roles) <= {'FIT', 'CAL', 'DEV'}, roles
    res = dict(status='COMPLETE', family=family, seed=seed, norm=norm, role=role, protocols=protocols, n_rows=len(rows), meta=model['meta'],
               sift=sift_cfg, gpu=torch.cuda.get_device_name(), seconds=time.time() - started, roles_read=roles, job=job,
               rows_sha256=sha(rows_path), freeze_sha256=sha(HERE / 'FREEZE.json') if (HERE / 'FREEZE.json').exists() else None)
    write(out / 'COMPLETE.json', res); write(out / f'ATTEMPT_{job}.json', dict(status='COMPLETE', seconds=res['seconds']))
    print('R23B_EVAL_COMPLETE', name, len(rows), flush=True)


def _mi_worker(args):
    core, probs, role, variants = args
    g = guard(); out = []
    for prob in probs:
        ref, mov = render_full(g, role, prob)
        for variant in variants:
            out += _mi_one(core, prob, ref, mov, variant)
    return out, sorted({r['role'] for r in g.reads}) if hasattr(g, 'reads') else None


def _mi_one(core, prob, ref, mov, variant):
    out = []
    if True:
        t0 = time.perf_counter(); b = sitk_mi(ref, mov, variant=variant); sec = time.perf_counter() - t0
        est = (b[1], b[2] / 4.0, b[3] / 4.0) if b else (0.0, 0.0, 0.0)
        out.append(dict(protocol=prob['protocol'], core=core, k=prob['k'], level=prob['level'],
                        method='MI_SimpleITK' if variant == 'reg' else 'MI_SimpleITK_tuned', variant=variant, seed=0, norm='raw',
                        truth=prob['truth'], est=list(est), err_full=corner_error(est, prob['truth']), metric=b[0] if b else None,
                        start=b[4] if b else None, seconds=sec, failed=b is None))
    return out


def mi_sitk(role, workers, variants=('reg',), tag=''):
    from multiprocessing import Pool
    assert json.loads((HERE / 'CPU_CHECK.json').read_text())['status'] == 'PASS' and (HERE / 'PROTOCOL.json').exists()
    g = guard(); out = OUT / f'MI_SimpleITK{tag}_{role}'; out.mkdir(parents=True, exist_ok=True); assert not (out / 'COMPLETE.json').exists()
    tasks = []
    for core in g.included(role):
        tasks.append((core, [draw(p, core, k) for p in PROTOCOLS for k in range(K_DRAWS)], role, tuple(variants)))
    started = time.time(); rows = []; roles = set()
    with Pool(workers) as pool:
        for r, rr in pool.imap_unordered(_mi_worker, tasks):
            rows += r; roles |= set(rr or [])
            for x in r: append(out / 'ROWS.jsonl', x)
    assert roles <= {'FIT', 'CAL', 'DEV'}, roles
    write(out / 'COMPLETE.json', dict(status='COMPLETE', role=role, n_rows=len(rows), seconds=time.time() - started, roles_read=sorted(roles),
                                      rows_sha256=sha(out / 'ROWS.jsonl')))
    print('R23B_MI_COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['check', 'eval', 'mi_sitk'])
    ap.add_argument('--family'); ap.add_argument('--seed', type=int, default=1001); ap.add_argument('--norm', default='patch')
    ap.add_argument('--protocols', default='comir,mre'); ap.add_argument('--role', default='DEV'); ap.add_argument('--limit', type=int)
    ap.add_argument('--tag', default=''); ap.add_argument('--workers', type=int, default=32); ap.add_argument('--variants', default='reg'); a = ap.parse_args()
    if a.mode == 'check': check()
    elif a.mode == 'eval': eval_unit(a.family, a.seed, a.norm, a.protocols.split(','), a.role, a.limit, a.tag)
    else: mi_sitk(a.role, a.workers, a.variants.split(','), a.tag)
