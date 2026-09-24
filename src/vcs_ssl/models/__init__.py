"""Encoder / projector / critic construction with fair, hashed initialization (spec §5, §13.1)."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..utils import state_dict_sha256
from .backbone import build_resnet18_cifar
from .critic import build_critic, critic_impl_name
from .projector import build_projector

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
        critic = build_critic(mcfg["critic"], feature_dim=mcfg["projector"]["output_dim"])
    hashes = {"encoder_init_sha256": state_dict_sha256(encoder), "projector_init_sha256": state_dict_sha256(projector),
              "critic_init_sha256": state_dict_sha256(critic) if critic is not None else None}
    encoder.to(device)
    projector.to(device)
    if critic is not None:
        critic.to(device)
    return {"encoder": encoder, "projector": projector, "critic": critic, "init_hashes": hashes,
            "params": {"encoder": count_params(encoder), "projector": count_params(projector), "critic": count_params(critic)},
            "model_strings": {"encoder": str(encoder), "projector": str(projector), "critic": str(critic)},
            "critic_impl": critic_impl_name(mcfg["critic"]) if critic is not None else None}
