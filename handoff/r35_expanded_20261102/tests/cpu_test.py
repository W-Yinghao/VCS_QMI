"""R35 CPU test: (1) the package reference checks (re-run in-process on the package's reference modules, results
written here, the package dir is not written), (2) production-vs-reference numeric alignment in float64,
(3) list construction, two-stage composite ranking, affine calibration, oracle isolation, static scans.
Usage: python -I -B cpu_test.py <package_dir>   -> ROOT/CPU_TEST.json"""
import importlib.util, json, os, re, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'r35'))
import numpy as np
import torch
from torch.nn import functional as F
from common import install_guards, guard_record, write_json, utc, sha256
import objectives as O
import heads as H
import evals as EV
import data35 as D35


def load_ref(pkg, name, rel):
    spec = importlib.util.spec_from_file_location(name, str(pkg / rel)); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m


def main():
    pkg = Path(sys.argv[1]).resolve()
    install_guards()
    torch.set_num_threads(4)
    checks = {}; t0 = time.time()

    def chk(name, fn):
        try:
            info = fn(); ok = info.pop('pass_', True) if isinstance(info, dict) else bool(info)
            checks[name] = dict(pass_=bool(ok), **(info if isinstance(info, dict) else {}))
        except Exception:
            checks[name] = dict(pass_=False, error=traceback.format_exc())
        print(('PASS ' if checks[name]['pass_'] else 'FAIL ') + name, flush=True)

    RO = load_ref(pkg, 'ref_objectives', 'reference/objectives.py')
    RM = load_ref(pkg, 'ref_models', 'reference/models.py')
    VM = load_ref(pkg, 'ref_validate_matrix', 'tools/validate_matrix.py')
    torch.set_default_dtype(torch.float64)

    # ---------------- (1) package reference checks (same assertions as reference/checks.py)
    def ref_checks():
        torch.manual_seed(123); out = {}
        p = torch.rand(17); p /= p.sum(); q = torch.rand(17); q /= q.sum(); m = (p + q) / 2; eta = (p - q) / (p + q)
        t = torch.tanh(torch.randn(17)); j = (p * (t - .5 * t * t) - q * (t + .5 * t * t)).sum()
        e = float(abs((m * eta ** 2).sum() - j - (m * (t - eta) ** 2).sum())); assert e < 1e-12; out['square_gap_identity'] = e
        g = torch.randn(9, 9); pw, qw = RO.pq_weights(9)
        assert abs(float(RO.bidirectional_nce(g) - RO.bidirectional_nce(g + 25))) < 1e-12 and abs(float(RO.original_j(g, pw, qw) - RO.original_j(g + 25, pw, qw))) > .1
        z = 2.7 * g - 3.9; assert torch.equal(g.argsort(0), z.argsort(0)) and torch.equal(g.argsort(1), z.argsort(1))
        x = F.normalize(torch.randn(7, 12), dim=-1); y = F.normalize(torch.randn(8, 12), dim=-1); base = 4 * (x @ y.T) - 2
        for nm, mod in [('identity_metric', RM.IdentityMetric(12, 4)), ('global_residual', RM.GlobalResidual(12, 32))]:
            zz = mod(x, y); assert float((zz - base).abs().max()) < 1e-12
        cr = RM.LocalResidual(12, 8); v = torch.randn(3, 5, 12); w = torch.randn(3, 6, 12)
        vm = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 1, 0, 0, 0]], dtype=torch.bool); wm = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
        assert bool((cr(v, w, vm, wm) == 0).all())
        g = torch.randn(5, 11); a = torch.zeros_like(g); a[:, 0] = .5; a[:, 3] = .5; perm = torch.randperm(11)
        for obj in ('J', 'BCE', 'NCE'):
            assert abs(float(RO.candidate_loss(g, a, obj) - RO.candidate_loss(g[:, perm], a[:, perm], obj))) < 1e-12
        out['matrix'] = VM.validate(); return out
    chk('package_reference_checks', ref_checks)

    # ---------------- (2) production vs reference
    def records_obj():
        torch.manual_seed(0); g = torch.randn(128, 128) * 3; p, q = O.pq_weights(128, 'cpu', torch.float64); rp, rq = RO.pq_weights(128)
        assert torch.equal(p, rp) and torch.equal(q, rq)
        e = dict(J=float(abs(-O.loss_records('J', g, p, q) - RO.original_j(g, rp, rq))), BCE=float(abs(O.loss_records('BCE', g, p, q) - RO.balanced_bce(g, rp, rq))),
                 NCE=float(abs(O.loss_records('NCE', g, p, q) - RO.bidirectional_nce(g))))
        return dict(pass_=max(e.values()) < 1e-12, abs_err=e)
    chk('records_objectives_equal_reference', records_obj)

    def list_obj():
        torch.manual_seed(1); g = torch.randn(16, 32) * 4; a = torch.zeros(16, 32)
        for r in range(16):
            k = 1 + r % 3; a[r, torch.randperm(32)[:k]] = 1 / k
        e = {o: float(abs(O.loss_list(o, g, a) - RO.candidate_loss(g, a, o))) for o in O.OBJECTIVES}
        perm = torch.randperm(32)
        pe = {o: float(abs(O.loss_list(o, g, a) - O.loss_list(o, g[:, perm], a[:, perm]))) for o in O.OBJECTIVES}
        gg = g.clone().requires_grad_()
        fin = all(bool(torch.isfinite(torch.autograd.grad(O.loss_list(o, gg, a), gg)[0]).all()) for o in O.OBJECTIVES)
        return dict(pass_=max(e.values()) < 1e-12 and max(pe.values()) < 1e-12 and fin, abs_err=e, perm_err=pe)
    chk('list_objectives_equal_reference_perm_invariant', list_obj)

    def heads_step0():
        torch.manual_seed(2); x = F.normalize(torch.randn(40, 64), dim=-1); y = F.normalize(torch.randn(50, 64), dim=-1)
        a0, b0 = 17.3, -6.1; out = {}
        for st in ('ID', 'G', 'L'):
            hs = H.build_heads(st, a0, b0, 1001)
            for p, h in hs.items():
                if st == 'ID':
                    h2 = H.IDHead(a0, b0, d=64, rank=8); g = h2.matrix(x, y)
                else:
                    h2 = H.GHead(a0, b0, d=64, local=(st == 'L'))
                    g = h2.matrix(x, y) if st == 'G' else None
                a, b = h2.affine.ab()
                if g is not None:
                    out[f'{st}_step0_max_abs_vs_affine'] = float((g - (a * (x @ y.T) + b)).abs().max())
                out[f'{st}_a_minus_a0'] = float(a - a0)
        # bitwise in float32 for the ID transform and G matrix
        x32, y32 = x.float(), y.float()
        idh = H.IDHead(a0, b0, d=64, rank=8).float(); gh = H.GHead(a0, b0, d=64).float()
        out['ID_fp32_bitwise_tx_identity'] = bool(torch.equal(idh.tx(x32), x32))
        a, b = gh.affine.ab(); out['G_fp32_bitwise_step0'] = bool(torch.equal(gh.matrix(x32, y32), a * (x32 @ y32.T) + b))
        ok = all(v < 1e-12 for k, v in out.items() if k.endswith('vs_affine')) and all(abs(v) < 1e-12 for k, v in out.items() if k.endswith('a_minus_a0')) \
            and out['ID_fp32_bitwise_tx_identity'] and out['G_fp32_bitwise_step0']
        return dict(pass_=ok, **out)
    chk('heads_step0_reproduce_common_affine', heads_step0)

    def heads_vs_reference():
        torch.manual_seed(3); d = 24; x = F.normalize(torch.randn(6, d), dim=-1); y = F.normalize(torch.randn(7, d), dim=-1)
        # ID vs reference IdentityMetric (reference normalises; ours rescales to |x| = 1 -> same up to fp64)
        ref = RM.IdentityMetric(d, 4, a=3.0, b=-1.0); mine = H.IDHead(3.0, -1.0, d=d, rank=4)
        with torch.no_grad():
            for nm in ('ax', 'ay'):
                getattr(mine, nm).weight.copy_(getattr(ref, nm).weight)
            for nm in ('bx', 'by'):
                w = torch.randn_like(getattr(ref, nm).weight) * .1; getattr(ref, nm).weight.copy_(w); getattr(mine, nm).weight.copy_(w)
            mine.affine.raw_a.copy_(ref.affine.raw_a); mine.affine.b.copy_(ref.affine.b)
        e_id = float((mine.matrix(x, y) - ref(x, y)).abs().max())
        # G vs reference GlobalResidual (nonzero final layer)
        gr = RM.GlobalResidual(d, 32, a=3.0, b=-1.0); gm = H.GHead(3.0, -1.0, d=d)
        gm.glob.l1 = torch.nn.Linear(4 * d + 1, 32); gm.glob.l2 = torch.nn.Linear(32, 16); gm.out = torch.nn.Linear(16, 1)
        with torch.no_grad():
            gm.glob.l1.load_state_dict(gr.net[0].state_dict()); gm.glob.l2.load_state_dict(gr.net[2].state_dict())
            gr.net[4].weight.normal_(); gr.net[4].bias.normal_(); gm.out.load_state_dict(gr.net[4].state_dict())
            gm.affine.raw_a.copy_(gr.affine.raw_a); gm.affine.b.copy_(gr.affine.b)
        e_g = float((gm.matrix(x, y) - gr(x, y)).abs().max())
        # local branch vs reference LocalResidual
        lr = RM.LocalResidual(d, 8); lb = H.LocalBranch(d, 8)
        with torch.no_grad():
            lb.vi.load_state_dict(lr.vi.state_dict()); lb.wt.load_state_dict(lr.wt.state_dict()); lb.mix.load_state_dict(lr.mix.state_dict())
            lr.out.weight.normal_(); lr.out.bias.normal_()
        v = torch.randn(3, 5, d); w = torch.randn(3, 6, d)
        vm = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 1, 0, 0, 0]], dtype=torch.bool); wm = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
        feats = lb.pair_features(lb.proj_regions(v), lb.proj_words(w), vm, wm)
        e_l = float(((feats @ lr.out.weight.T).squeeze(-1) + lr.out.bias - lr(v, w, vm, wm)).abs().max())
        vp = v.clone(); vp[~vm] = 1e4; e_mask = float((lb.pair_features(lb.proj_regions(vp), lb.proj_words(w), vm, wm) - feats).abs().max())
        return dict(pass_=max(e_id, e_g, e_l, e_mask) < 1e-12, ID=e_id, G=e_g, local=e_l, padding_mask=e_mask)
    chk('heads_equal_reference_modules', heads_vs_reference)

    def cl_zero_and_grad():
        torch.manual_seed(4); d = 32; h = H.GHead(5.0, -2.0, d=d, local=True); h.glob.l1 = torch.nn.Linear(4 * d + 1, 256)
        x = F.normalize(torch.randn(10, d), dim=-1); y = F.normalize(torch.randn(10, d), dim=-1); s = (x * y).sum(-1)
        lf = torch.randn(10, 256)
        g = h.paired(x, y, s, lf); a, b = h.affine.ab()
        z = float((g - (a * s + b)).abs().max())
        (g.square().sum()).backward()
        return dict(pass_=z == 0.0 and h.out.weight.grad.abs().sum() > 0, step0_residual_abs=z)
    chk('C_L_zero_init_residual_and_out_gradient', cl_zero_and_grad)

    torch.set_default_dtype(torch.float32)

    def lists():
        rng = np.random.default_rng(0); n_img, n_cap = 50, 250
        obs = np.arange(n_cap) // 5; obs[:40] = rng.permutation(obs[:40])
        cand_i2t = np.stack([rng.permutation(n_cap)[:32] for _ in range(n_img)]); cand_t2i = np.stack([rng.permutation(n_img)[:32] for _ in range(n_cap)])
        rec_i = rng.integers(0, n_cap, 16); rec_t = rng.integers(0, n_cap, 16)
        perm_i = np.stack([rng.permutation(32) for _ in range(16)]); perm_t = np.stack([rng.permutation(32) for _ in range(16)])
        (qi, caps, ai), (qc, imgs, at) = D35.build_lists(rec_i, rec_t, obs, cand_i2t, cand_t2i, perm_i, perm_t)
        ok = all(rec_i[r] in caps[r] and len(set(caps[r])) == 32 and len(set(imgs[r])) == 32 and obs[rec_t[r]] in imgs[r] for r in range(16))
        ok &= np.allclose(ai.sum(1), 1) and np.allclose(at.sum(1), 1)
        ok &= all(np.array_equal(ai[r] > 0, obs[caps[r]] == qi[r]) for r in range(16))  # every observed positive counted
        ok &= all(np.array_equal(at[r] > 0, imgs[r] == obs[rec_t[r]]) for r in range(16))
        ok &= all(caps[r][np.argsort(perm_i[r])][0] == rec_i[r] for r in range(16))  # slot 0 before permutation = positive
        return dict(pass_=bool(ok), mean_pos_slots_i2t=float((ai > 0).sum(1).mean()))
    chk('candidate_list_construction', lists)

    def composite():
        rng = np.random.default_rng(1); nat = rng.standard_normal((EV.N_DEV_IMG, EV.N_DEV_CAP)).astype(np.float32).astype(np.float64)
        for i in range(EV.N_DEV_IMG):
            nat[i, 5 * i:5 * i + 5] += 2.0
        tab = EV.dev_tables(nat)
        gi = 7.3 * np.take_along_axis(nat, tab['i_top'], 1) - 1.1; gt = 7.3 * np.take_along_axis(nat.T, tab['t_top'], 1) - 1.1
        m, tb, cov, rk = EV.composite_metrics(nat, tab, gi, gt)
        up, ri, rt = EV.upstream_metrics(nat)
        same = all(abs(m[k] - up[k]) < 1e-12 for k in ('i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10'))
        # a reversed critic inside top-100 changes ranks only inside the top-100
        m2, _, _, rk2 = EV.composite_metrics(nat, tab, -gi, -gt)
        inside = bool(((rk2['ranks_t'] < 100) == (rk['ranks_t'] < 100)).all())
        pos_i, pos_t, ai, at = EV.dev_list_positions(tab)
        return dict(pass_=same and np.array_equal(rk['ranks_i'], ri) and np.array_equal(rk['ranks_t'], rt) and inside and np.allclose(ai.sum(1), 1),
                    rsum=m['rsum'], rsum_upstream=up['rsum'], rsum_reversed=m2['rsum'], coverage=cov)
    chk('two_stage_composite_equals_upstream_at_affine', composite)

    def calib():
        rng = np.random.default_rng(2); gp = rng.normal(1.5, 1, 4000); gq = rng.normal(-1, 1, 40000)
        g = np.r_[gp, gq]; wp = np.r_[np.full(4000, 1 / 4000), np.zeros(40000)]; wq = np.r_[np.zeros(4000), np.full(40000, 1 / 40000)]
        f = EV.fit_affine(g, wp, wq); f2 = EV.fit_affine(g, wp, wq, 'max_J')
        jr, jc, jm = EV.j_weighted(g, wp, wq), EV.j_weighted(f['a'] * g + f['b'], wp, wq), EV.j_weighted(f2['a'] * g + f2['b'], wp, wq)
        s = rng.standard_normal((30, 40))
        return dict(pass_=f['a'] > 0 and f2['a'] > 0 and jm >= jc - 1e-9 and EV.rank_invariance(s, f['a'], f['b']),
                    a=f['a'], b=f['b'], J_raw=jr, J_logistic_affine=jc, J_maxJ_affine=jm)
    chk('affine_calibration_fit_and_rank_invariance', calib)

    def jmatrix():
        torch.manual_seed(5); g = torch.randn(20, 100, dtype=torch.float64); m = np.arange(100) // 5
        r = EV.j_from_matrix(g, m)
        P = np.zeros((20, 100)); P[m, np.arange(100)] = 1 / 100
        cnt = np.bincount(m, minlength=20).astype(float); Q = cnt[:, None] * np.ones((20, 100)); Q[m, np.arange(100)] -= 1; Q /= 100 * 99
        return dict(pass_=abs(r['J'] - EV.j_weighted(g.numpy(), P, Q)) < 1e-12, J=r['J'])
    chk('record_J_matrix_equals_weighted', jmatrix)

    def oracle_iso():
        res = {}
        try:
            import oracle_truth  # noqa
            res['import_in_non_oracle_process'] = 'NOT_REFUSED'
        except ImportError as e:
            res['import_in_non_oracle_process'] = 'refused: ' + str(e)
        return dict(pass_=res['import_in_non_oracle_process'].startswith('refused') and 'oracle_truth' not in sys.modules, **res)
    chk('oracle_import_refused_in_non_oracle_process', oracle_iso)

    def static():
        hits, truth_hits = [], []
        for f in sorted((ROOT / 'r35').glob('*.py')):
            for i, line in enumerate(f.read_text().splitlines()):
                if re.search(r"test_(ims|caps|ids|tags)|testall|test5k|'test'|\"test\"", line) and '_TEST_RE' not in line:
                    hits.append(f'{f.name}:{i+1}')
                if f.name not in ('oracle_truth.py',) and re.search(r'(N_TRAIN_CAP|145000)[^\n]*//\s*5', line):
                    truth_hits.append(f'{f.name}:{i+1}')
        imp = [f.name for f in (ROOT / 'r35').glob('*.py') if f.name not in ('train_oracle.py', 'oracle_truth.py') and re.search(r'^\s*(import|from)\s+oracle_truth', f.read_text(), re.M)]
        return dict(pass_=not hits and not truth_hits and not imp, test_path_hits=hits, fit_truth_expression_hits=truth_hits, oracle_importers=imp)
    chk('static_scan_no_test_paths_no_fit_truth_outside_oracle', static)

    allp = all(v['pass_'] for v in checks.values())
    write_json(ROOT / 'CPU_TEST.json', dict(ALL_PASS=allp, n_checks=len(checks), checks=checks, created_utc=utc(), seconds=time.time() - t0,
                                            host=os.uname().nodename, job=os.environ.get('SLURM_JOB_ID'), test_guard=guard_record(),
                                            code_sha256={f.name: sha256(f) for f in sorted((ROOT / 'r35').glob('*.py'))},
                                            package_reference_sha256={r: sha256(pkg / r) for r in ('reference/objectives.py', 'reference/models.py', 'reference/checks.py', 'tools/validate_matrix.py')}))
    print('R35_CPU', 'ALL_PASS' if allp else 'FAIL', [k for k, v in checks.items() if not v['pass_']])
    sys.exit(0 if allp else 1)


if __name__ == '__main__':
    main()
