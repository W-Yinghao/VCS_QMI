"""Encoder / projector / critic construction with fair, hashed initialization (spec §5, §13.1)."""
from __future__ import annotations

from typing import Any

import copy

import torch
from torch import nn

from ..utils import state_dict_sha256
from .backbone import build_resnet18_cifar
from .critic import build_critic, critic_impl_name
from .projector import build_predictor, build_projector

__all__ = ["build_models", "build_resnet18_cifar", "build_projector", "build_critic", "count_params"]


def count_params(m: nn.Module | None) -> int:
    return 0 if m is None else sum(p.numel() for p in m.parameters())


def build_models(cfg: dict[str, Any], *, seed: int, device: torch.device | str = "cpu") -> dict[str, Any]:
    """Build the three modules.

    Encoder and projector are initialized from ``torch.manual_seed(seed)`` so every method with the same seed starts
    from bit-identical encoder/projector weights.  The critic (VCS only) is initialized afterwards from a *separate*
    seed stream; data shuffling and augmentation use their own dedicated generators, so critic construction cannot
    perturb them.  Weight hashes are returned for cross-run verification.
    """
    mcfg = cfg["model"]
    if mcfg["weights"] is not None:
        raise ValueError("pretrained weights are forbidden")
    torch.manual_seed(seed)
    encoder = build_resnet18_cifar(mcfg["stem"])
    projector = build_projector(mcfg["projector"], in_dim=mcfg["h_dim"])
    critic = None
    if mcfg["critic"]["enabled"]:
        torch.manual_seed(seed * 1000003 + 7919)  # separate stream, disclosed engineering choice
        fdim = int(mcfg["h_dim"]) if mcfg["critic"].get("feature_source", "z") == "h_l2" else int(mcfg["projector"]["output_dim"])
        critic = build_critic(mcfg["critic"], feature_dim=fdim)
    predictor = None
    if mcfg["projector"].get("predictor", False):
        torch.manual_seed(seed * 1000003 + 104729)  # separate stream (disclosed)
        predictor = build_predictor(mcfg["projector"])
    hashes = {"encoder_init_sha256": state_dict_sha256(encoder), "projector_init_sha256": state_dict_sha256(projector),
              "critic_init_sha256": state_dict_sha256(critic) if critic is not None else None}
    encoder.to(device)
    projector.to(device)
    if critic is not None:
        critic.to(device)
    if predictor is not None:
        predictor.to(device)
    tb = cfg["train"].get("target_branch", "shared")
    teacher = None
    if tb.startswith("ema_"):
        teacher = {"encoder": copy.deepcopy(encoder), "projector": copy.deepcopy(projector), "tau": float(tb.split("_", 1)[1])}
        for m in (teacher["encoder"], teacher["projector"]):
            for q in m.parameters():
                q.requires_grad_(False)
    return {"encoder": encoder, "projector": projector, "critic": critic, "predictor": predictor, "teacher": teacher, "init_hashes": hashes,
            "params": {"encoder": count_params(encoder), "projector": count_params(projector), "critic": count_params(critic),
                       "predictor": count_params(predictor)},
            "model_strings": {"encoder": str(encoder), "projector": str(projector), "critic": str(critic)},
            "critic_impl": critic_impl_name(mcfg["critic"]) if critic is not None else None}


@torch.no_grad()
def ema_update(teacher: dict[str, Any], encoder: nn.Module, projector: nn.Module) -> None:
    """teacher <- tau * teacher + (1 - tau) * student for parameters; BN buffers copied from the student."""
    tau = teacher["tau"]
    for t_mod, s_mod in ((teacher["encoder"], encoder), (teacher["projector"], projector)):
        for tp, sp in zip(t_mod.parameters(), s_mod.parameters()):
            tp.mul_(tau).add_(sp.detach(), alpha=1.0 - tau)
        for tb_, sb_ in zip(t_mod.buffers(), s_mod.buffers()):
            tb_.copy_(sb_)
