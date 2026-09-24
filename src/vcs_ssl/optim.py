"""AdamW with three parameter groups (spec §8): enc/proj matrices (wd 1e-4), enc/proj bias+BN (wd 0), critic (wd 0)."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn


def build_optimizer(encoder: nn.Module, projector: nn.Module, critic: nn.Module | None, ocfg: dict[str, Any]) -> torch.optim.AdamW:
    if ocfg["name"] != "adamw":
        raise ValueError("first round uses AdamW")
    decay, no_decay = [], []
    for module in (encoder, projector):
        for name, p in module.named_parameters():
            if not p.requires_grad:
                raise ValueError(f"frozen parameter in trainable module: {name}")
            (decay if p.ndim >= 2 else no_decay).append(p)
    groups = [
        {"name": "enc_proj_matrix", "params": decay, "lr": ocfg["lr"], "base_lr": ocfg["lr"],
         "weight_decay": ocfg["matrix_weight_decay_encoder_projector"]},
        {"name": "enc_proj_bias_norm", "params": no_decay, "lr": ocfg["lr"], "base_lr": ocfg["lr"],
         "weight_decay": ocfg["weight_decay_bias_norm"]},
    ]
    if critic is not None:
        cparams = [p for p in critic.parameters() if p.requires_grad]
        groups.append({"name": "critic", "params": cparams, "lr": ocfg["lr"] * ocfg["critic_lr_multiplier"],
                       "base_lr": ocfg["lr"] * ocfg["critic_lr_multiplier"], "weight_decay": ocfg["critic_weight_decay"]})
    opt = torch.optim.AdamW(groups, betas=tuple(ocfg["betas"]), eps=ocfg["eps"])
    verify_optimizer_coverage(opt, encoder, projector, critic)
    return opt


def verify_optimizer_coverage(opt: torch.optim.Optimizer, encoder: nn.Module, projector: nn.Module, critic: nn.Module | None) -> dict[str, int]:
    """Every trainable parameter appears exactly once; nothing foreign is present."""
    in_opt: list[int] = [id(p) for g in opt.param_groups for p in g["params"]]
    if len(in_opt) != len(set(in_opt)):
        raise ValueError("duplicate parameter in optimizer")
    expected: dict[int, str] = {}
    for mname, module in (("encoder", encoder), ("projector", projector), ("critic", critic)):
        if module is None:
            continue
        for pname, p in module.named_parameters():
            if p.requires_grad:
                expected[id(p)] = f"{mname}.{pname}"
    missing = [n for i, n in expected.items() if i not in set(in_opt)]
    foreign = [i for i in in_opt if i not in expected]
    if missing:
        raise ValueError(f"trainable parameters missing from optimizer: {missing[:5]} (+{max(0, len(missing) - 5)})")
    if foreign:
        raise ValueError("optimizer contains parameters not belonging to encoder/projector/critic")
    return {"n_params_in_optimizer": len(in_opt), "n_trainable_expected": len(expected)}


def set_lrs(opt: torch.optim.Optimizer, factor: float) -> dict[str, float]:
    out = {}
    for g in opt.param_groups:
        g["lr"] = g["base_lr"] * factor
        out[g["name"]] = g["lr"]
    return out


@torch.no_grad()
def grad_norms(encoder: nn.Module, projector: nn.Module, critic: nn.Module | None) -> dict[str, float | None]:
    """L2 norm of all gradients per module, collected after backward and before optimizer.step (no clipping)."""
    out: dict[str, float | None] = {}
    for name, m in (("encoder", encoder), ("projector", projector), ("critic", critic)):
        if m is None:
            out[f"grad_norm_{name}"] = None
            continue
        sq = 0.0
        any_grad = False
        for p in m.parameters():
            if p.grad is not None:
                any_grad = True
                sq += float(p.grad.detach().float().pow(2).sum())
        out[f"grad_norm_{name}"] = sq ** 0.5 if any_grad else None
    return out


@torch.no_grad()
def all_grads_finite(opt: torch.optim.Optimizer) -> bool:
    for g in opt.param_groups:
        for p in g["params"]:
            if p.grad is not None and not torch.isfinite(p.grad).all():
                return False
    return True
