#!/usr/bin/env python3
"""CPU self-check (float64) of margin_core_v3: 13 fixed-list objects (S-J = independent gap, S=(K-m)/(K+m), pair J equals
-package pair_loss('vcs')); 14 identity audit rule on a hand-built batch. Usage: selfcheck_v3.py CODE_R29 REF_PY OUT.json"""
import importlib.util, json, os, sys, traceback
import torch
code_r29, ref_py, out_path = sys.argv[1:4]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(1, code_r29)
spec = importlib.util.spec_from_file_location('r30_reference', ref_py); REF = importlib.util.module_from_spec(spec); spec.loader.exec_module(REF)
torch.set_default_dtype(torch.float64)
import margin_core_v3 as M3
res = {}
def check(name):
    def deco(fn):
        try:
            res[name] = {'pass': True, **(fn() or {})}
        except Exception as e:
            res[name] = {'pass': False, 'error': repr(e), 'trace': traceback.format_exc()}
    return deco

@check('13_fixed_list_objects')
def _():
    g = torch.Generator().manual_seed(9)
    K = 128; S = (torch.rand(40, K, generator=g) * 2 - 1) * 0.6
    pos = torch.zeros(40, K, dtype=torch.bool); pos[torch.arange(40), torch.arange(40)] = True; pos[:5, 100] = True
    J, Sor, gap, m = M3.list_object(S, pos)
    r1 = float((Sor - J - gap).abs().max())
    assert torch.allclose(Sor, (K - m) / (K + m)) and r1 < 1e-12, r1
    sp, sc = S[:, 0], S[:, 1]
    Jp, Sp, gp = M3.pair_object(sp, sc)
    r2 = float((Sp - Jp - gp).abs().max())
    ref_loss = REF.pair_loss((sp - sc) / 0.05, 'vcs')
    r3 = float((Jp + ref_loss).abs().max())
    assert r2 < 1e-12 and r3 < 1e-12 and abs(float(Sp[0]) - 1 / 3) < 1e-15, (r2, r3)
    return {'max_resid_list': r1, 'max_resid_pair': r2, 'max_diff_vs_package_pair_loss': r3}

@check('14_identity_audit_rule')
def _():
    obs = torch.tensor([0, 1, 1, 2]); true = torch.tensor([0, 1, 0, 2])   # sample 2: mismatch, true image 0
    S = torch.tensor([[.9, .1, .2, .3], [.1, .8, .3, .2], [.6, .1, .4, .2], [.1, .2, .3, .7]])
    a = M3.audit_rows(S, obs, true)
    # t2i query 2 (caption 2, true image 0): observed non-positives are image rows with obs != 1 -> rows 0, 3; hardest = row 0 (.2 vs .2? col 2: .2, .3, .4, .3)
    col = S[:, 2]
    assert int(a['t2i']['k'][2]) == 3 and not bool(a['t2i']['P'][2, 3]) and bool(a['t2i']['avail'][2])
    # i2t query 0 (image 0): caption 2 is a true partner (true 0) but observed positive of image 1 -> non-positive; hardest non-pos of row 0 = col 3 (.3)
    assert int(a['i2t']['k'][0]) == 3
    S2 = S.clone(); S2[0, 2] = .95
    b = M3.audit_rows(S2, obs, true)
    assert int(b['i2t']['k'][0]) == 2 and bool(b['i2t']['P'][0, 2])          # competitor is a true match
    ref = {'all': torch.ones(4, dtype=torch.bool).numpy(), 'correct_ref_first': torch.tensor([1, 1, 0, 1]).bool().numpy(),
           'correct_ref_not_first': torch.zeros(4, dtype=torch.bool).numpy(), 'mismatch': torch.tensor([0, 0, 1, 0]).bool().numpy()}
    gr, sel = M3.audit_groups(ref, b['i2t'])
    assert bool(sel[0]) and gr['mismatch&true_available'].tolist() == [False, False, True, False]
    return {}

ok = all(v['pass'] for v in res.values())
json.dump({'status': 'ALL_PASS' if ok else 'FAIL', 'checks': res}, open(out_path, 'w'), indent=1)
print({k: v['pass'] for k, v in res.items()}, 'ALL_PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
