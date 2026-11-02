"""R36 CPU test -> ROOT/CPU_TEST.json.
(1) package tests/test_reference.py run BY PATH in a subprocess (python -I -B), 9 checks;
(2) production alignment (float64): R36 list objectives vs package reference loss_rows; N null target vs package
    missing_observation_target; null-slot permutation invariance; oracle shield/relabel vs package (in the oracle
    runner process: train_oracle.py --selftest);
(3) sampler properties on toy tables: same queries/permutations as R35 sample_lists for the same seed (FORCED,
    RANDOM); RANDOM excludes the forced partner and draws 31 distinct; NULL lists = unmodified top-32 + null, rows
    sum to 1, null position covers all 33 slots, candidates independent of the observed map, i2t multi-positive;
(4) heads: NullSlots consumes no RNG (real heads bitwise the same-seed C heads);
(5) isolation: oracle_truth import refused in this process; trainer role refuses opening OUT/oracle_runs and
    OUT/audit and importing the evaluator-only module; static scan (only train_oracle imports oracle_truth, no
    trainer imports audit code, no TEST paths).
Usage: python -I -B cpu_test.py <package_dir>"""
import importlib.util, json, os, re, subprocess, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R36 = ROOT / 'r36'
sys.path.insert(0, str(R36))
import numpy as np
import torch
from common import install_guards, guard_record, write_json, utc, sha256, ORACLE_RUNS, OUT
import objectives as O
import heads as H
import data35 as D35
import data36 as D36


def main():
    pkg = Path(sys.argv[1]).resolve()
    install_guards(role='trainer'); torch.set_num_threads(4)
    checks = {}; t0 = time.time()

    def chk(name, fn):
        try:
            info = fn(); ok = info.pop('pass_', True) if isinstance(info, dict) else bool(info)
            checks[name] = dict(pass_=bool(ok), **(info if isinstance(info, dict) else {}))
        except Exception:
            checks[name] = dict(pass_=False, error=traceback.format_exc())
        print(('PASS ' if checks[name]['pass_'] else 'FAIL ') + name, flush=True)

    def pkg_tests():
        r = subprocess.run([sys.executable, '-I', '-B', str(pkg / 'tests' / 'test_reference.py')], capture_output=True, text=True)
        tail = (r.stdout + r.stderr)[-1500:]
        n_ok = len(re.findall(r' \.\.\. ok', r.stdout + r.stderr))
        return dict(pass_=r.returncode == 0 and n_ok == 9, rc=r.returncode, n_ok=n_ok, tail=tail)
    chk('package_tests_test_reference_9', pkg_tests)

    spec = importlib.util.spec_from_file_location('ref_objectives_r36', str(pkg / 'reference' / 'objectives.py'))
    RO = importlib.util.module_from_spec(spec); spec.loader.exec_module(RO)
    torch.set_default_dtype(torch.float64)

    def list_obj():
        torch.manual_seed(1); g = torch.randn(16, 32) * 4; a = torch.zeros(16, 32)
        for r in range(16):
            k = 1 + r % 3; a[r, torch.randperm(32)[:k]] = 1 / k
        q = torch.full_like(a, 1 / 32); v = torch.ones_like(a, dtype=torch.bool)
        e = {}
        for o in O.OBJECTIVES:
            ref = RO.loss_rows(g, a, q, o).mean()
            e[o] = float(abs(O.loss_list(o, g, a) - ref)); e[o + '_pq'] = float(abs(O.loss_list_pq(o, g, a, q, v) - ref))
        return dict(pass_=max(e.values()) < 1e-12, abs_err=e)
    chk('list_objectives_equal_package_loss_rows', list_obj)

    def null_target():
        rng = np.random.default_rng(3); m = rng.random((50, 32)) < .05; m[::7] = False
        n = m.sum(1, keepdims=True); a = np.concatenate([m / np.maximum(n, 1), (n == 0)], 1)
        rp, rq = RO.missing_observation_target(torch.tensor(m))
        g = torch.tensor(rng.normal(size=(50, 33)) * 2); perm = torch.tensor(rng.permutation(33))
        inv = {o: float((O.list_rows(g, torch.tensor(a), o) - O.list_rows(g[:, perm], torch.tensor(a)[:, perm], o)).abs().max()) for o in O.OBJECTIVES}
        ref = {o: float(abs(O.loss_list(o, g, torch.tensor(a)) - RO.loss_rows(g, rp, rq, o).mean())) for o in O.OBJECTIVES}
        return dict(pass_=np.allclose(a, rp.numpy(), atol=0) and max(inv.values()) < 1e-12 and max(ref.values()) < 1e-12, perm_err=inv, ref_err=ref)
    chk('null_target_equals_package_and_permutation_invariant', null_target)

    def j_parts():
        g = torch.randn(10, 33); a = torch.zeros(10, 33); a[:, 5] = 1; m = torch.zeros(10, 33, dtype=torch.bool); m[:, 32] = True
        tot = O.list_rows(g, a, 'J'); s = O.j_parts(g, a, ~m) + O.j_parts(g, a, m)
        return dict(pass_=float((tot - s).abs().max()) < 1e-12)
    chk('null_J_parts_sum_to_total', j_parts)

    def oracle_selftest():
        r = subprocess.run([sys.executable, '-B', str(R36 / 'train_oracle.py'), '--selftest', str(pkg)], capture_output=True, text=True)
        line = [l for l in r.stdout.splitlines() if l.startswith('R36_ORACLE_SELFTEST ')]
        res = json.loads(line[0].split(' ', 1)[1]) if line else dict(error=r.stdout[-1500:] + r.stderr[-1500:])
        return dict(pass_=r.returncode == 0 and bool(res.get('ALL_PASS')), **res)
    chk('oracle_shield_relabel_vs_package_in_oracle_process', oracle_selftest)

    # ---- sampler on toy tables
    rng0 = np.random.default_rng(0)
    obs = (np.arange(145000) // 5).copy(); flip = rng0.random(145000) < .6; obs[flip] = rng0.integers(0, 29000, flip.sum())
    ci2t = rng0.integers(0, 145000, (29000, 32)).astype(np.int32)
    for i in range(29000):   # unique per row (frozen-H table property)
        u = np.unique(ci2t[i])
        while len(u) < 32:
            u = np.unique(np.r_[u, rng0.integers(0, 145000, 32 - len(u))])
        ci2t[i] = rng0.permutation(u)
    ct2i = np.stack([rng0.choice(29000, 32, replace=False) for _ in range(2000)]).astype(np.int32)
    ct2i = ct2i[rng0.integers(0, 2000, 145000)]
    # fixture realism: frozen-H tables usually contain observed partners -> plant them (i2t: the observed captions of
    # each image in the first slots for 90% of images; t2i: the observed image in slot 0 for 40% of captions)
    order = np.argsort(obs, kind='stable'); starts = np.searchsorted(obs[order], np.arange(29001))
    for q in range(29000):
        if rng0.random() < .9:
            mine = order[starts[q]:starts[q + 1]][:8]
            rest = [c for c in ci2t[q] if c not in set(mine)][:32 - len(mine)]
            ci2t[q] = np.r_[mine, rest][:32]
    put = rng0.random(145000) < .4
    for r in np.nonzero(put)[0]:
        if obs[r] not in ct2i[r]:
            ct2i[r, 0] = obs[r]
    d = dict(obs_np=obs, cand_i2t=ci2t, cand_t2i=ct2i)

    def same_queries():
        out = {}
        for sup in ('OBS_HARD', 'RANDOM', 'TRUST_HARD', 'SHIELD_ORACLE'):
            r35 = D35.sample_lists(np.random.default_rng([1001, 3]), obs, ci2t, ct2i)
            r36 = D36.sample(sup, np.random.default_rng([1001, 3]), np.random.default_rng([1001, 3, 7]), d, pool=None)
            same_idx = all(np.array_equal(x, y) for x, y in zip(r35[1], r36[1]))
            if sup in ('OBS_HARD', 'TRUST_HARD', 'SHIELD_ORACLE'):
                same_lists = all(np.array_equal(x, y) for a, b in zip(r35[0], r36[0]) for x, y in zip(a, b))
            else:
                same_lists = None
            out[sup] = dict(same_draws=same_idx, same_lists=same_lists)
        ok = all(v['same_draws'] for v in out.values()) and all(v['same_lists'] in (True, None) for v in out.values())
        return dict(pass_=ok, **out)
    chk('sampler_same_queries_perms_as_R35_C', same_queries)

    def random_lists():
        lists, (ri, rt, pi, pt) = D36.sample('RANDOM', np.random.default_rng([1001, 3]), np.random.default_rng([1001, 3, 7]), d)
        (qi, caps, ai), (qc, imgs, at) = lists
        ok = all(rec in row for rec, row in zip(ri, caps)) and all(obs[r] in row for r, row in zip(rt, imgs))
        distinct = all(len(set(r)) == 32 for r in caps) and all(len(set(r)) == 32 for r in imgs)
        once = all((row == rec).sum() == 1 for rec, row in zip(ri, caps)) and all((row == obs[r]).sum() == 1 for r, row in zip(rt, imgs))
        rows = np.allclose(ai.sum(1), 1) and np.allclose(at.sum(1), 1)
        return dict(pass_=ok and distinct and once and rows)
    chk('random_lists_forced_partner_distinct_31', random_lists)

    def null_lists():
        r = np.random.default_rng([1001, 3]); pos = np.zeros(33, int); res = {}
        for _ in range(300):
            lists, (ri, rt, pi, pt) = D36.sample('OBS_MISSING_NULL', r, None, d)
            (qi, caps, ai), (qc, imgs, at) = lists
            for ids, a in ((caps, ai), (imgs, at)):
                assert ids.shape == (16, 33) and (ids == D36.NULL_ID).sum(1).tolist() == [1] * 16
                assert np.allclose(a.sum(1), 1)
                pos += np.bincount(np.nonzero(ids == D36.NULL_ID)[1], minlength=33)
            real_i = np.sort(np.where(caps == D36.NULL_ID, -2, caps), 1)[:, 1:]
            assert np.array_equal(real_i, np.sort(ci2t[obs[ri]].astype(np.int64), 1))
            real_t = np.sort(np.where(imgs == D36.NULL_ID, -2, imgs), 1)[:, 1:]
            assert np.array_equal(real_t, np.sort(ct2i[rt].astype(np.int64), 1))
        # candidates independent of the observed map; labels follow it
        rr = np.arange(16); perm = np.tile(np.arange(33), (16, 1))
        l1 = D36.null_lists(rr, rr, obs, ci2t, ct2i, perm, perm); obs2 = obs.copy(); obs2[rr] = (obs2[rr] + 1) % 29000
        l2 = D36.null_lists(rr, rr, obs2, ci2t, ct2i, perm, perm)
        t2i_same_cands = np.array_equal(l1[1][1], l2[1][1])
        # i2t multi-positive and t2i single positive
        lists, _ = D36.sample('OBS_MISSING_NULL', np.random.default_rng(9), None, d, n=2000)
        (qi, caps, ai), (qc, imgs, at) = lists
        npos_i = ((ai > 0) & (caps != D36.NULL_ID)).sum(1); npos_t = ((at > 0) & (imgs != D36.NULL_ID)).sum(1)
        res = dict(null_position_min_count=int(pos.min()), null_positions_all_33=bool((pos > 0).all()), t2i_candidates_unchanged_when_obs_changes=t2i_same_cands,
                   i2t_rows_multi_positive=int((npos_i > 1).sum()), t2i_max_positive=int(npos_t.max()),
                   null_rows_A_null_1=bool(np.allclose(ai[caps == D36.NULL_ID][npos_i == 0], 1)))
        return dict(pass_=res['null_positions_all_33'] and t2i_same_cands and res['i2t_rows_multi_positive'] > 0 and res['t2i_max_positive'] == 1 and res['null_rows_A_null_1'], **res)
    chk('null_lists_properties', null_lists)

    def trust_pool():
        pool = np.sort(np.random.default_rng(4).choice(145000, 61588, replace=False))
        lists, (ri, rt, pi, pt) = D36.sample('TRUST_HARD', np.random.default_rng([1001, 3]), None, d, pool=pool)
        return dict(pass_=bool(np.isin(ri, pool).all() and np.isin(rt, pool).all()))
    chk('trust_anchors_only_from_subset', trust_pool)

    def null_heads_no_rng():
        torch.set_default_dtype(torch.float32)
        h1 = H.build_heads('G', 6.0, -.5, 1001); torch.manual_seed(0); a = torch.rand(3)
        h2 = H.build_heads('G', 6.0, -.5, 1001); n = H.NullSlots(); torch.manual_seed(0); b = torch.rand(3)
        same = all(torch.equal(x, y) for x, y in zip(h1.state_dict().values(), h2.state_dict().values()))
        torch.set_default_dtype(torch.float64)
        return dict(pass_=same and torch.equal(a, b) and all(float(v.abs().sum()) == 0 for v in n.v.values()))
    chk('null_slots_zero_init_no_rng_and_heads_unchanged', null_heads_no_rng)

    def isolation():
        out = {}
        try:
            import oracle_truth  # noqa
            out['oracle_import'] = 'NOT_REFUSED'
        except ImportError:
            out['oracle_import'] = 'refused'
        for nm, path in (('oracle_runs', ORACLE_RUNS / 'x.json'), ('audit', OUT / 'audit' / 'AUDIT_EXPORT.json')):
            try:
                open(path).close(); out[nm] = 'OPENED'
            except PermissionError:
                out[nm] = 'refused'
            except FileNotFoundError:
                out[nm] = 'NOT_GUARDED(missing file)'
        sys.path.insert(0, str(ROOT / 'audit'))
        try:
            import audit_export  # noqa
            out['audit_import'] = 'NOT_REFUSED'
        except ImportError:
            out['audit_import'] = 'refused'
        sys.path.remove(str(ROOT / 'audit'))
        hits, imp, aud = [], [], []
        for f in sorted(R36.glob('*.py')):
            txt = f.read_text()
            for i, line in enumerate(txt.splitlines()):
                if re.search(r"test_(ims|caps|ids|tags)|testall|test5k|'test'|\"test\"", line) and '_TEST_RE' not in line:
                    hits.append(f'{f.name}:{i+1}')
            if f.name not in ('train_oracle.py', 'oracle_truth.py') and re.search(r'^\s*(import|from)\s+oracle_truth', txt, re.M):
                imp.append(f.name)
            if re.search(r'^\s*(import|from)\s+(audit_export|audit_candidate_lists)', txt, re.M):
                aud.append(f.name)
        out.update(static_test_path_hits=hits, static_oracle_importers=imp, static_audit_importers=aud)
        ok = out['oracle_import'] == 'refused' and out['oracle_runs'] == 'refused' and out['audit'] == 'refused' and out['audit_import'] == 'refused' and not hits and not imp and not aud
        return dict(pass_=ok, **out)
    chk('isolation_trainer_role', isolation)

    allp = all(v['pass_'] for v in checks.values())
    write_json(ROOT / 'CPU_TEST.json', dict(ALL_PASS=allp, checks=checks, created_utc=utc(), job=os.environ.get('SLURM_JOB_ID', 'local'),
                                            host=os.uname().nodename, code_sha256={f.name: sha256(f) for f in sorted(R36.glob('*.py'))},
                                            package_tests_sha256=sha256(pkg / 'tests' / 'test_reference.py'), guard=guard_record(), elapsed_seconds=time.time() - t0))
    print('R36_CPU_TEST', 'ALL_PASS' if allp else 'FAIL', [k for k, v in checks.items() if not v['pass_']])
    sys.exit(0 if allp else 1)


if __name__ == '__main__':
    main()
