"""R37 CPU test -> ROOT/CPU_TEST.json. (1) R37 objectives/heads/data36 == R36 modules on the shared functions (R36 files
imported by path, read-only); (2) N_PRESENT_ONLY masked loss: formula, == loss_list for mask 1 / p_d 1, gradient 0 on
masked rows; (3) toy tables: valid_rows == observed-present definition, presence_probs == brute force, TRUST_NATIVE_NULL
anchors only from the pool and lists == N null_lists, N_PRESENT_ONLY draws == N draws for the same seed; (4) NullSlots no
RNG; (5) isolation: trainer role refuses OUT/oracle_runs + OUT/audit; static scan (no TEST paths, no oracle/audit imports).
Usage: python -I -B cpu_test.py"""
import importlib.util, os, re, sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; R37 = ROOT / 'r37'
sys.path.insert(0, str(R37))
import numpy as np
import torch
from common import install_guards, write_json, utc, ORACLE_RUNS, OUT, R36_ROOT
import objectives as O
import heads as H
import data36 as D36


def load_r36(name):
    spec = importlib.util.spec_from_file_location(f'r36_{name}', str(R36_ROOT / 'r36' / f'{name}.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def main():
    install_guards(role='trainer'); torch.set_num_threads(4); torch.set_default_dtype(torch.float64)
    checks = {}; t0 = time.time()

    def chk(name, fn):
        try:
            info = fn(); ok = info.pop('pass_', True) if isinstance(info, dict) else bool(info)
            checks[name] = dict(pass_=bool(ok), **(info if isinstance(info, dict) else {}))
        except Exception:
            checks[name] = dict(pass_=False, error=traceback.format_exc())
        print(('PASS ' if checks[name]['pass_'] else 'FAIL ') + name, flush=True)

    def same_as_r36():
        RO = load_r36('objectives')
        torch.manual_seed(1); g = torch.randn(16, 33) * 4; a = torch.zeros(16, 33)
        for r in range(16):
            k = 1 + r % 3; a[r, torch.randperm(33)[:k]] = 1 / k
        ok = all(torch.equal(O.list_rows(g, a, o), RO.list_rows(g, a, o)) and torch.equal(O.loss_list(o, g, a), RO.loss_list(o, g, a)) for o in O.OBJECTIVES)
        shared = {}
        for f in ('heads.py', 'evals.py', 'native_eval.py'):
            shared[f] = (R37 / f).read_bytes() == (R36_ROOT / 'r36' / f).read_bytes()
        return dict(pass_=ok and all(shared.values()), byte_identical_files=shared)
    chk('objectives_equal_R36_and_heads_evals_byte_identical', same_as_r36)

    def masked():
        torch.manual_seed(2); g = (torch.randn(16, 33) * 3).requires_grad_(); a = torch.zeros(16, 33)
        for r in range(16):
            a[r, r % 33] = 1.
        m = (torch.arange(16) % 3 != 0).double(); p = 0.4
        ok = True; e = {}
        for o in O.OBJECTIVES:
            v = O.loss_list_masked(o, g, a, m, p)
            man = (O.list_rows(g, a, o) * m).sum() / (16 * p) * (-1 if o == 'J' else 1)
            ok &= abs(float(v - man)) < 1e-12
            ok &= abs(float(O.loss_list_masked(o, g, a, torch.ones(16), 1.0) - O.loss_list(o, g, a))) < 1e-12
            gr, = torch.autograd.grad(v, g)
            ok &= bool((gr[m == 0] == 0).all()) and bool((gr[m == 1].abs().sum(1) > 0).all())
            e[o] = float(v)
        return dict(pass_=ok, values=e)
    chk('present_only_masked_loss', masked)

    rng0 = np.random.default_rng(5)
    NI, NC = 29000, 145000
    obs = rng0.integers(0, NI, NC)
    ci = np.stack([rng0.choice(NC, 32, replace=False) for _ in range(NI)]).astype(np.int32)
    ct = np.stack([rng0.choice(NI, 32, replace=False) for _ in range(NC // 50)] * 50).astype(np.int32)
    # plant observed partners so that presence is non-trivial
    for i in range(0, NI, 3):
        ci[i, 0] = np.nonzero(obs == i)[0][0] if (obs == i).any() else ci[i, 0]
    for c in range(0, NC, 4):
        ct[c, 0] = obs[c]
    d = dict(obs_np=obs, cand_i2t=ci, cand_t2i=ct)

    def toy():
        ok = True
        pp = D36.presence_probs(d)
        bi = np.array([(obs[ci[obs[r]].astype(np.int64)] == obs[r]).any() for r in range(0, NC, 97)]).mean()
        bt = np.array([(ct[r].astype(np.int64) == obs[r]).any() for r in range(NC)]).mean()
        ok &= abs(pp['p_t2i'] - bt) < 1e-12
        ok &= abs(pp['p_i2t'] - bi) < .02
        rng = np.random.default_rng([1001, 3]); rngb = np.random.default_rng([1001, 3]); rngt = np.random.default_rng([1001, 5])
        pool = np.sort(rng0.choice(NC, 5000, replace=False)); pset = set(pool.tolist())
        for _ in range(20):
            lists, (ri, rt, pi, pt) = D36.sample('N_PRESENT_ONLY', rng, None, d)
            lb, idx_b = D36.sample('OBS_MISSING_NULL', rngb, None, d)
            ok &= all(np.array_equal(x, y) for k in range(2) for x, y in zip(lists[k], lb[k]))
            mi, mt = D36.valid_rows(lists)
            (qi, caps, ai), (qc, imgs, at) = lists
            ok &= bool(np.array_equal(mi, np.array([(obs[ci[q].astype(np.int64)] == q).any() for q in qi])))
            ok &= bool(np.array_equal(mt, np.array([(ct[r].astype(np.int64) == obs[r]).any() for r in qc])))
            l2, (ri2, rt2, pi2, pt2) = D36.sample('TRUST_NATIVE_NULL', rngt, None, d, pool=pool)
            ok &= all(x in pset for x in ri2.tolist() + rt2.tolist())
            ref = D36.null_lists(ri2, rt2, obs, ci, ct, pi2, pt2)
            ok &= all(np.array_equal(x, y) for k in range(2) for x, y in zip(l2[k], ref[k]))
            ok &= pi2.shape[1] == 33
        return dict(pass_=ok, presence=pp)
    chk('toy_tables_present_mask_presence_trust_null', toy)

    def nulls_no_rng():
        torch.manual_seed(7); s0 = torch.get_rng_state().clone(); H.NullSlots(); return bool(torch.equal(s0, torch.get_rng_state()))
    chk('null_slots_no_rng', nulls_no_rng)

    def isolation():
        res = []
        for p in (ORACLE_RUNS / 'x', OUT / 'audit' / 'x.json'):
            try:
                open(p); res.append('OPENED')
            except PermissionError:
                res.append('REFUSED')
            except FileNotFoundError:
                res.append('MISSING')
        hits = []
        for f in sorted(R37.glob('*.py')):
            if f.name == 'gate.py':
                continue
            t = f.read_text()
            for i, line in enumerate(t.splitlines()):
                if re.search(r"test_(ims|caps|ids|tags)|testall|test5k|'test'|\"test\"", line) and '_TEST_RE' not in line:
                    hits.append(f'{f.name}:{i+1}')
            if re.search(r'^\s*(import|from)\s+(oracle_truth|audit_export|audit_candidate_lists)', t, re.M):
                hits.append(f'{f.name}:oracle/audit import')
        return dict(pass_=res == ['REFUSED', 'REFUSED'] and not hits, opens=res, hits=hits)
    chk('isolation_and_static_scan', isolation)

    allp = all(v['pass_'] for v in checks.values())
    write_json(ROOT / 'CPU_TEST.json', dict(ALL_PASS=allp, checks=checks, created_utc=utc(), job=os.environ.get('SLURM_JOB_ID', 'local'), seconds=time.time() - t0))
    print('R37_CPU', 'ALL_PASS' if allp else 'FAIL'); sys.exit(0 if allp else 1)


if __name__ == '__main__':
    main()
