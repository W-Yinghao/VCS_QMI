"""V10 T03 critics (decision 9.2) on integer observations encoded one-hot.

MF:  the unchanged package SimplexBilinearCritic(m, m, K, K, temperature=1): Linear + softmax on each side, box head W = 0 at start,
     no tanh, no ridge.  T(x, y) = a(x)^T W b(y).
MLP: concat(onehot x, onehot y) -> Linear(2m, 64) -> GELU -> Linear(64, 64) -> GELU -> Linear(64, 1) = f;  T = tanh f.
Initial state: current PyTorch default initialisation (nn.Linear.reset_parameters, constructor order: MF left, right, then W zeros;
MLP layers in order) drawn from the CPU default generator seeded from the named 'init' stream (algo seed, condition, critic kind, K)
inside torch.random.fork_rng, so the global RNG is untouched and the state does not depend on the method: MF_VCS_perm1 / M0 / M1
share one initial state, MLP_VCS_perm1 / perm8 share one initial state; nothing is searched per method.
"""
from __future__ import annotations

import hashlib

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from v10.mf_vcs import SimplexBilinearCritic

import t03_common as C


class MLPCritic(nn.Module):
    def __init__(self, m: int, hidden: int = C.MLP_HIDDEN):
        super().__init__()
        self.m = int(m)
        self.net = nn.Sequential(nn.Linear(2 * self.m, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, xo: Tensor, yo: Tensor) -> Tensor:          # raw f; T = tanh f
        return self.net(torch.cat([xo, yo], -1)).squeeze(-1)


def critic_kind(method: str) -> str:
    if method in C.MF_METHODS:
        return "MF"
    if method in C.MLP_METHODS:
        return "MLP"
    raise ValueError(f"unknown T03 method {method!r}")


def init_stream(method: str, family: str, m: int, rho: float, K: int | None, algo_seed: int):
    kind = critic_kind(method)
    return C.stream(algo_seed, "init", family, m, rho, C.CRITIC_CODE[kind], int(K) if kind == "MF" else 0)


def build_critic(method: str, family: str, m: int, rho: float, K: int | None, algo_seed: int, *, dtype=torch.float32,
                 device="cpu") -> tuple[nn.Module, dict]:
    kind = critic_kind(method)
    if kind == "MF" and (K is None or int(K) < 1):
        raise ValueError("MF critics need a simplex width K")
    ss = init_stream(method, family, m, rho, K, algo_seed)
    seed = C.stream_seed64(ss)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)                  # CPU generator only (no CUDA side effect)
        model = SimplexBilinearCritic(m, m, int(K), int(K), temperature=C.TEMPERATURE) if kind == "MF" else MLPCritic(m)
    model = model.to(device=device, dtype=dtype)
    if kind == "MF" and bool(model.W.detach().any()):
        raise RuntimeError("MF head must start at W = 0")
    return model, {"kind": kind, "init_stream": C.stream_record(ss), "init_seed64": seed, "init_state_sha256": state_hash(model),
                   "n_parameters": sum(p.numel() for p in model.parameters()),
                   "init_rule": "torch default nn.Linear init, CPU default generator seeded from the init stream inside fork_rng"}


def onehot(idx, m: int, *, dtype=torch.float32, device="cpu") -> Tensor:
    t = torch.as_tensor(idx, dtype=torch.long, device=device)
    return F.one_hot(t, int(m)).to(dtype)


@torch.no_grad()
def critic_table(model: nn.Module, m: int) -> Tensor:
    """T on the full m x m state table (model dtype / device): MF a(x)^T W b(y); MLP tanh f(x, y)."""
    p = next(model.parameters())
    eye = torch.eye(int(m), dtype=p.dtype, device=p.device)
    if isinstance(model, SimplexBilinearCritic):
        a, b = model.features(eye, eye)
        return a @ model.W @ b.T
    xo = eye.repeat_interleave(int(m), 0)                         # row x * m + y
    yo = eye.repeat(int(m), 1)
    return torch.tanh(model(xo, yo)).reshape(int(m), int(m))


def state_hash(model: nn.Module) -> str:
    h = hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        t = t.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def clone_state(model: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
