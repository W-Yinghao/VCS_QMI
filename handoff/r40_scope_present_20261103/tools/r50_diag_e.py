"""R50 diagnosis of the in-unit check E failure of R50_A_F60_R30H1001_PRESENT_ONLY_both_joint_NCE (job 1086393, step 8000: E.A max_abs 1.19e-4 > TOL_SCORE 1e-4).
Read-only (no training, writes only under OUT_R50/diag_e/<job>): loads the unit's saved checkpoints (steps given) and re-runs check E exactly as r50ext
(same generator stream as the in-unit call at that step, same redrawn head) in fp32, plus the same comparison with the redrawn head, the DEV cache and the
(fp32-encoded) student vectors in fp64 (the native GPO cannot run in fp64); records max|score|, residual std, fp32 ULP at max|score|, and the split of the fp32 difference into (a) student re-encoding of the
renamed / re-chunked pool and (b) pair order / chunking only. Usage: python -B tools/r50_diag_e.py <tree> <unit> <step,step,...>"""
import copy, json, os, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
W = Path(__file__).resolve().parents[1]
TREE, U, STEPS = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3].split(',')]
sys.path.insert(0, str(W / TREE))
import torch
import common as C
C.install_guards(role='trainer')
import train_unit as TU
TU.set_determinism()
import r39ext as R9, r50ext as R50, data36 as D36
from heads import build_heads, NullSlots
from common import PEERS, N_DEV_IMG, N_DEV_CAP
dev = torch.device('cuda')
cfg = TU.cfg_of(U); src = C.RUNS / U
ctx = TU.Ctx(cfg['cond'], dev, need_local=False); ctx.ensure_joint(); J = ctx.joint; d = ctx.d
heads = build_heads(cfg['structure'], d['a0b0']['a0'], d['a0b0']['b0'], cfg['init_seed']).to(dev)
stud = R9.build_students(J.ck, dev, C.ADAPTATIONS[cfg['adaptation']])
OD = C.R50_OUT / 'diag_e' / os.environ.get('SLURM_JOB_ID', 'local'); OD.mkdir(parents=True, exist_ok=True)
ulp = lambda x: float(np.spacing(np.float32(abs(x))))
out = dict(unit=U, tree=TREE, steps={}, gpu=torch.cuda.get_device_name(0), host=os.uname().nodename)


def E_run(step, dtype):
    """r50ext.head_checks re-implemented step by step with the in-unit generator stream, optionally in fp64 end to end."""
    gen = np.random.default_rng([R50.SEED, int(step)])
    R50.truth_check(ctx, gen)   # consume the generator exactly as layout_checks does before head_checks (truth relabel draw)
    n = 1500; img = gen.integers(0, N_DEV_IMG, n); cap = gen.integers(0, N_DEV_CAP, n)
    pi_i, pi_c = gen.permutation(N_DEV_IMG), gen.permutation(N_DEV_CAP); inv_i, inv_c = np.argsort(pi_i), np.argsort(pi_c)
    q = gen.permutation(n); qb = np.argsort(q)
    res = {}
    for p in PEERS:
        h = copy.deepcopy(heads[p]); h.eval()
        gt = torch.Generator().manual_seed(R50.SEED + PEERS.index(p))
        with torch.no_grad():
            h.out.weight.copy_((torch.randn(h.out.weight.shape, generator=gt) * .1).to(h.out.weight)); h.out.bias.copy_((torch.randn(h.out.bias.shape, generator=gt) * .1).to(h.out.bias))
        sp = stud[p]; J2 = J   # student encoder always fp32 (native GPO builds fp32 tensors internally); dtype applies to the head + cache + vectors
        if dtype == torch.float64:
            h = h.double()
        d2 = dict(d); d2[('dev', 'img', p)] = d[('dev', 'img', p)].to(dtype); d2[('dev', 'cap', p)] = d[('dev', 'cap', p)].to(dtype)
        cx = SimpleNamespace(d=d2, dev=dev, raw=None)
        it, ct = torch.as_tensor(img, device=dev), torch.as_tensor(cap, device=dev)
        with torch.no_grad():
            U_ = J2.encode(sp, p, 'dev', 'img', np.arange(N_DEV_IMG)).to(dtype); V_ = J2.encode(sp, p, 'dev', 'cap', np.arange(N_DEV_CAP)).to(dtype)
            s1 = TU.g_pairs_lists(h, None, cx, p, 'dev', img, cap, stu=(U_, it, V_, ct))
            X, Y = d2[('dev', 'img', p)], d2[('dev', 'cap', p)]
            Xr = torch.empty_like(X); Xr[torch.as_tensor(pi_i, device=dev)] = X; Yr = torch.empty_like(Y); Yr[torch.as_tensor(pi_c, device=dev)] = Y
            d3 = dict(d2); d3[('dev', 'img', p)] = Xr; d3[('dev', 'cap', p)] = Yr; cx3 = SimpleNamespace(d=d3, dev=dev, raw=None)
            ir, cr = pi_i[img][q], pi_c[cap][q]
            Ur = J2.encode(sp, p, 'dev', 'img', inv_i, chunk=97).to(dtype); Vr = J2.encode(sp, p, 'dev', 'cap', inv_c, chunk=301).to(dtype)
            s2 = TU.g_pairs_lists(h, None, cx3, p, 'dev', ir, cr, chunk=333, stu=(Ur, torch.as_tensor(ir, device=dev), Vr, torch.as_tensor(cr, device=dev)))[torch.as_tensor(qb, device=dev)]
            # (b) pair order / chunking only: renamed pool but the natural-order student vectors moved by the rename (no re-encoding)
            Um = torch.empty_like(U_); Um[torch.as_tensor(pi_i, device=dev)] = U_; Vm = torch.empty_like(V_); Vm[torch.as_tensor(pi_c, device=dev)] = V_
            s3 = TU.g_pairs_lists(h, None, cx3, p, 'dev', ir, cr, chunk=333, stu=(Um, torch.as_tensor(ir, device=dev), Vm, torch.as_tensor(cr, device=dev)))[torch.as_tensor(qb, device=dev)]
            a, b = (float(t) for t in h.affine.ab()); resid = s1.double() - (a * (X[it] * Y[ct]).sum(-1).double() + b)
            enc = dict(img=float((Ur[torch.as_tensor(pi_i, device=dev)] - U_).abs().max()), cap=float((Vr[torch.as_tensor(pi_c, device=dev)] - V_).abs().max()))
        dl = (s1.double() - s2.double()).abs(); m = float(s1.abs().max())
        mr = float(resid.abs().max())
        res[p] = dict(max_abs=float(dl.max()), max_abs_residual=mr, fp32_ulp_at_max_abs_residual=ulp(mr), max_rel=float((dl / s1.double().abs().clamp_min(1e-12)).max()), max_abs_score=m, fp32_ulp_at_max_abs_score=ulp(m),
                      max_abs_in_ulps_of_max_score=float(dl.max()) / ulp(m), residual_std=float(resid.std()), score_std=float(s1.double().std()),
                      pair_order_chunk_only_max_abs=float((s1.double() - s3.double()).abs().max()), student_reencoding_vector_max_abs=enc,
                      argmax_pair=int(dl.argmax()), score_at_argmax=float(s1[int(dl.argmax())]))
    return res


for st in STEPS:
    ck = torch.load(src / f'ckpt_step{st}.pt', map_location=dev, weights_only=False)
    heads.load_state_dict(ck['heads']); stud.load_state_dict(ck['student'])
    out['steps'][st] = dict(fp32=E_run(st, torch.float32), fp64=E_run(st, torch.float64))
    print('R50_DIAG_E step', st, json.dumps(out['steps'][st]), flush=True)
(OD / f'DIAG_E_{U}.json').write_text(json.dumps(out, indent=1))
print('R50_DIAG_E_DONE', OD)
