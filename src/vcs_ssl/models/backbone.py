"""torchvision ResNet-18 with the CIFAR stem (spec §5.1); P130 adds a CIFAR-stem ResNet-50 (v6 §8.3 architecture check)."""
from __future__ import annotations

from typing import Any

from torch import nn
from torchvision.models import resnet18, resnet50


def build_resnet18_cifar(stem: dict[str, Any]) -> nn.Module:
    if stem["maxpool"]:
        raise ValueError("CIFAR stem removes maxpool")
    net = resnet18(weights=None)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=stem["kernel_size"], stride=stem["stride"], padding=stem["padding"], bias=stem["bias"])
    net.maxpool = nn.Identity()
    net.fc = nn.Identity()  # output h: [N, 512] after avgpool + flatten
    return net


# ---- P130 (v6 §8.3): CIFAR-stem ResNet-50, h = 2048-d.  Same stem surgery as ResNet-18; torchvision defaults otherwise
# (Bottleneck, zero_init_residual=False).  Existing resnet18_cifar configs never reach this code.
def build_resnet50_cifar(stem: dict[str, Any]) -> nn.Module:
    if stem["maxpool"]:
        raise ValueError("CIFAR stem removes maxpool")
    net = resnet50(weights=None)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=stem["kernel_size"], stride=stem["stride"], padding=stem["padding"], bias=stem["bias"])
    net.maxpool = nn.Identity()
    net.fc = nn.Identity()  # output h: [N, 2048] after avgpool + flatten
    return net


BACKBONE_H_DIM = {"resnet18_cifar": 512, "resnet50_cifar": 2048}


def build_backbone(name: str, stem: dict[str, Any]) -> nn.Module:
    if name == "resnet18_cifar":
        return build_resnet18_cifar(stem)
    if name == "resnet50_cifar":
        return build_resnet50_cifar(stem)
    raise ValueError(f"unknown backbone {name!r}")
