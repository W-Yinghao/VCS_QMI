"""torchvision ResNet-18 with the CIFAR stem (spec §5.1)."""
from __future__ import annotations

from typing import Any

from torch import nn
from torchvision.models import resnet18


def build_resnet18_cifar(stem: dict[str, Any]) -> nn.Module:
    if stem["maxpool"]:
        raise ValueError("CIFAR stem removes maxpool")
    net = resnet18(weights=None)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=stem["kernel_size"], stride=stem["stride"], padding=stem["padding"], bias=stem["bias"])
    net.maxpool = nn.Identity()
    net.fc = nn.Identity()  # output h: [N, 512] after avgpool + flatten
    return net
