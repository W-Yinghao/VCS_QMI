import sys, torch
sys.path.insert(0, '/home/infres/yinwang/CS_QMI/handoff/r27_audit_20261029'); sys.path.insert(0, '/home/infres/yinwang/CS_QMI/handoff/r27_qtarget_20261029/code_v11')
import audit_constrained as K
DT = torch.float64
g = torch.Generator().manual_seed(7); d = 7; worst = 0.0; worst_comp = 0.0; neg = 0; n = 0
for t in range(2000):
    p = torch.randn(1, d, generator=g, dtype=DT); p = p / p.norm()
    c = float(torch.rand(1, generator=g)) * 0.95
    x = torch.randn(1, d, generator=g, dtype=DT) * float(torch.rand(1, generator=g) * 3) + [0, 2.0, -1.5, 0][t % 4] * p
    y = K.proj_ball_half(x, p, c)
    # KKT: x - y = a*y - b*p, a>=0 (ball active), b>=0 (halfspace active), complementary slackness
    r = (x - y)[0]; A = torch.stack([y[0], -p[0]], 1)
    sol = torch.linalg.lstsq(A, r.unsqueeze(1)).solution.flatten()
    a, b = float(sol[0]), float(sol[1])
    resid = float((A @ sol - r).norm()); nrm = float(y.norm()); pz = float((y * p).sum())
    comp = max(abs(a) * abs(nrm - 1) if a > 1e-12 else 0, abs(b) * abs(pz - c) if b > 1e-12 else 0)
    worst = max(worst, resid); worst_comp = max(worst_comp, comp); neg += (a < -1e-10) or (b < -1e-10); n += 1
print(dict(n=n, max_stationarity_resid=worst, max_compl=worst_comp, negative_multipliers=neg))
