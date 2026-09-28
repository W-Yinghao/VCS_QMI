"""S2 (P75) CPU check: the solo-learn VCS port computes the same loss and gradients as the ssl_pilot reference path
(vcs_pair_loss_negdetach + CosineCritic, same shifts) on random L2-normalised features.  float64, exact to round-off."""
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/infres/yinwang/CS_QMI/ssl_pilot")
sys.path.insert(0, "/home/infres/yinwang/CS_QMI/ssl_pilot/src")
sys.path.insert(0, "/home/infres/yinwang/CS_QMI/solo_learn")
from vcs_ssl.models.critic import CosineCritic as RefCritic  # noqa: E402
from vcs_ssl.objectives import vcs_pair_loss_negdetach  # noqa: E402
from solo.methods.vcs import CosineCritic, cyclic_negative_indices, vcs_from_scores  # noqa: E402

torch.set_default_dtype(torch.float64)
B, D, K = 256, 256, 8
g = torch.Generator().manual_seed(0)
z1 = F.normalize(torch.randn(B, D, generator=g), dim=1).requires_grad_()
z2 = F.normalize(torch.randn(B, D, generator=g) + 0.5 * z1.detach(), dim=1).requires_grad_()
ref_c, new_c = RefCritic(D, scale_init=5.0), CosineCritic(5.0)
with torch.no_grad():
    new_c.scale.fill_(5.3); new_c.bias.fill_(-0.2); ref_c.scale.fill_(5.3); ref_c.bias.fill_(-0.2)

s_ref, _ = vcs_pair_loss_negdetach(z1, z2, ref_c, k=K, generator=torch.Generator().manual_seed(7))
g_ref = torch.autograd.grad(s_ref["loss"], [z1, z2, ref_c.scale, ref_c.bias])

idx = cyclic_negative_indices(B, K, torch.Generator().manual_seed(7), z1.device)
t_pos = new_c(z1, z2)
t_neg = new_c(z1.unsqueeze(0).expand(K, -1, -1).reshape(-1, D), z2.detach()[idx].reshape(-1, D))
s_new = vcs_from_scores(t_pos, t_neg)
g_new = torch.autograd.grad(s_new["loss"], [z1, z2, new_c.scale, new_c.bias])

dl = abs(float(s_ref["loss"] - s_new["loss"]))
dg = max(float((a.reshape(-1) - b.reshape(-1)).abs().max()) for a, b in zip(g_ref, g_new))
print(f"loss ref {float(s_ref['loss']):.12f} port {float(s_new['loss']):.12f} |diff| {dl:.2e}; max |grad diff| {dg:.2e}")
print(f"J {float(s_new['J_raw']):.6f} t_pos {float(s_new['t_pos_mean']):.4f} t_neg {float(s_new['t_neg_mean']):.4f}")
ok = dl < 1e-12 and dg < 1e-12
print("EQUIVALENCE", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
