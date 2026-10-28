"""R26 CPU gate (DESIGN §4.1): PyTorch R26 objectives vs the package NumPy reference. Writes CPU_CHECK.json."""
import sys, json
import numpy as np
import torch

sys.path.insert(0, '/home/infres/yinwang/CS_QMI/r26_retrieval_arch/VCS_Retrieval_R26_Architecture_20261027/reference')
import quadratic_graph_reference as REF
from r26_graph import build_graph, coupling, j_graph, bce_graph, snce_graph, j_point, moment_j, arm_loss

res = {}


def ok(name, cond, **info):
    res[name] = dict(pass_=bool(cond), **info)
    print(name, 'PASS' if cond else 'FAIL', info)


def unit(x):
    return x / x.norm(dim=-1, keepdim=True)


torch.manual_seed(0)
g = np.random.default_rng(0)
d, B, K, NQ = 16, 12, 5, 40
# synthetic same-peer banks; duplicates forced by a small bank
ib = unit(torch.randn(NQ, d, dtype=torch.float64)); tb = unit(torch.randn(NQ, d, dtype=torch.float64))
qpeer = unit(torch.randn(B, d, dtype=torch.float64))
G = build_graph(qpeer, ib, tb, K, 'i2t')
A, Y = G['A'], G['Y']

# graph bookkeeping: merged duplicates, mass accumulation, W/r/c/Q, Q includes W-support (diagonal analogue)
idx = G['idx'].numpy()
A_ref = np.zeros((B, NQ))
for i in range(B):
    for s in idx[i]:
        A_ref[i, s] += 1.0 / K
cols = G['cols'].numpy()
W, c, Q = coupling(A)
r_np, c_np, Q_np = REF.coupling_marginals(A_ref[:, cols] / B)
ok('graph_merge', np.allclose(A.numpy(), A_ref[:, cols], atol=0) and np.array_equal(np.unique(idx), cols)
   and abs(float(W.sum()) - 1) < 1e-12 and np.allclose(c.numpy(), c_np, atol=1e-15) and np.allclose(Q.numpy(), Q_np, atol=1e-15)
   and bool((Q[W > 0] > 0).all()) and torch.allclose(Y, tb[G['cols']]),
   n_atoms=int(cols.size), shared_cols=int((A_ref[:, cols] > 0).sum(0).__gt__(1).sum()), max_col_mass=float(c.max()))

# J dense vs NumPy reference vs moment form (value and query gradient)
X = unit(torch.randn(B, d, dtype=torch.float64)).requires_grad_(True)
Fm = X.mm(Y.t())
J = j_graph(W, Q, Fm)
J_ref = REF.coupling_j(W.numpy(), Fm.detach().numpy())
Jm, gm_ref = REF.cosine_moment_value_and_query_gradient(W.numpy(), X.detach().numpy(), Y.numpy())
gJ = torch.autograd.grad(J, X)[0].numpy()
J_mom = moment_j(X, A, Y)
gmom = torch.autograd.grad(J_mom, X)[0].numpy()
ok('j_dense_moment_reference', abs(float(J) - J_ref) < 1e-14 and abs(float(J) - Jm) < 1e-14 and abs(float(J_mom) - Jm) < 1e-14
   and np.abs(gJ - gm_ref).max() < 1e-13 and np.abs(gmom - gm_ref).max() < 1e-13,
   J=float(J), J_ref=J_ref, J_moment_ref=Jm, grad_max_abs=float(np.abs(gJ - gm_ref).max()))

# finite-difference query gradient of the torch loss
eps = 1e-6
Xn = X.detach().numpy(); fd = np.zeros_like(Xn)
for i in range(B):
    for k in range(d):
        Xp = Xn.copy(); Xp[i, k] += eps; Xm = Xn.copy(); Xm[i, k] -= eps
        fd[i, k] = (float(j_graph(W, Q, torch.from_numpy(Xp).mm(Y.t()))) - float(j_graph(W, Q, torch.from_numpy(Xm).mm(Y.t())))) / (2 * eps)
ok('query_grad_fd', np.abs(fd - gJ).max() < 1e-8, max_abs=float(np.abs(fd - gJ).max()))

# square gap: J* - J(F) = 0.5 sum (W+Q)(F-T)^2
T, sup = REF.unrestricted_optimum(W.numpy(), Q.numpy())
Fd = Fm.detach().numpy()
gap = 0.5 * np.sum((W.numpy() + Q.numpy()) * (Fd - T) ** 2)
ok('square_gap', abs((sup - float(J)) - gap) < 1e-14, sup=sup, J=float(J), gap=gap)

# balanced BCE: unrestricted optimum D = W/(W+Q) i.e. F = T; gradient zero there (interior entries)
Tt = torch.tensor(T, dtype=torch.float64, requires_grad=True)
gb = torch.autograd.grad(bce_graph(W, Q, Tt), Tt)[0].numpy()
interior = (np.abs(T) < 1 - 1e-9) & ((W.numpy() + Q.numpy()) > 0)
gv = torch.autograd.grad(j_graph(W, Q, Tt), Tt)[0].numpy()
ok('bce_common_optimum', np.abs(gb[interior]).max() < 1e-10 and np.abs(gv[(W.numpy() + Q.numpy()) > 0]).max() < 1e-12,
   bce_grad_at_T=float(np.abs(gb[interior]).max()), vcs_grad_at_T=float(np.abs(gv[(W.numpy() + Q.numpy()) > 0]).max()),
   n_interior=int(interior.sum()))

# mean-point identity: E_A[S - S^2/2] = mean - mean^2/2 - Var/2 per query (unnormalized mean point loses 0.5 Var(x.Y))
S_all = Fm.detach()
Ai = A / A.sum(1, keepdim=True)
mean = (Ai * S_all).sum(1); var = (Ai * S_all ** 2).sum(1) - mean ** 2
lhs = (Ai * (S_all - 0.5 * S_all ** 2)).sum(1)
ybar = A.mm(Y); sp = (X.detach() * ybar).sum(1)
ok('mean_point_identity', torch.allclose(lhs, mean - 0.5 * mean ** 2 - 0.5 * var, atol=1e-14) and torch.allclose(sp, mean, atol=1e-14),
   max_var=float(var.max()))

# soft NCE: unrestricted stationary point logits F/tau = log(W_ij/(r_i c_j)) + const_i on support
tau = 0.05
Wn, cn = W.numpy(), c.numpy()
Fs = np.where(Wn > 0, tau * np.log(np.maximum(Wn, 1e-300) / (Wn.sum(1, keepdims=True) * cn[None, :])), -50.0)
Ft = torch.tensor(Fs, requires_grad=True)
gs = torch.autograd.grad(snce_graph(W, c, Ft, tau), Ft)[0].numpy()
ok('snce_stationary', np.abs(gs).max() < 1e-10, max_abs=float(np.abs(gs).max()))

# point arms equal the R25 dense diagonal form on the corresponding point targets
from noisy_loss import vcs_J_dense
l_m, _ = arm_loss('m_vcs', X.detach(), G); l_n, _ = arm_loss('n_vcs', X.detach(), G)
yn = ybar / ybar.norm(dim=1, keepdim=True)
ok('point_arms', abs(float(l_m) + float(vcs_J_dense(X.detach().mm(ybar.t())))) < 1e-14
   and abs(float(l_n) + float(vcs_J_dense(X.detach().mm(yn.t())))) < 1e-14 and float(ybar.norm(dim=1).max()) <= 1 + 1e-12)

# all arms: finite, gradient only reaches X
for arm in ('d_vcs', 'd_bce', 'd_snce', 'm_vcs', 'n_vcs'):
    Xa = X.detach().clone().requires_grad_(True)
    l, _ = arm_loss(arm, Xa, G)
    gx = torch.autograd.grad(l, Xa)[0]
    ok(f'arm_{arm}', bool(torch.isfinite(l)) and bool(torch.isfinite(gx).all()) and float(gx.abs().sum()) > 0 and not Y.requires_grad and not A.requires_grad,
       loss=float(l))

res['all_pass'] = all(v['pass_'] for v in res.values() if isinstance(v, dict))
json.dump(res, open(sys.argv[1], 'w'), indent=1, default=float)
sys.exit(0 if res['all_pass'] else 1)
