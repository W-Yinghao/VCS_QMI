"""Two-view augmentation (train) and clean transform (evaluation).  Parameters are explicit (spec §4.3, W7)."""
from __future__ import annotations

from typing import Any

from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from ..utils import sha256_json

_INTERP = {"bilinear": InterpolationMode.BILINEAR, "bicubic": InterpolationMode.BICUBIC, "nearest": InterpolationMode.NEAREST}


def build_two_view_transform(views: dict[str, Any]) -> T.Compose:
    if views["gaussian_blur_p"] != 0.0 or views["solarize_p"] != 0.0:
        raise ValueError("blur/solarize are not implemented in the first-round recipe on purpose")
    rrc = views["random_resized_crop"]
    cj = views["color_jitter"]
    return T.Compose([
        T.RandomResizedCrop(rrc["size"], scale=tuple(rrc["scale"]), ratio=tuple(rrc["ratio"]),
                            interpolation=_INTERP[rrc["interpolation"]], antialias=rrc["antialias"]),
        T.RandomHorizontalFlip(p=views["horizontal_flip_p"]),
        T.RandomApply([T.ColorJitter(cj["brightness"], cj["contrast"], cj["saturation"], cj["hue"])], p=cj["p"]),
        T.RandomGrayscale(p=views["grayscale_p"]),
        T.ToTensor(),
        T.Normalize(mean=tuple(views["normalize_mean"]), std=tuple(views["normalize_std"])),
    ])


def build_clean_transform(views: dict[str, Any]) -> T.Compose:
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=tuple(views["normalize_mean"]), std=tuple(views["normalize_std"])),
    ])


def clean_transform_signature(views: dict[str, Any]) -> str:
    """Stable identifier for the evaluation transform (feature-cache key component)."""
    return sha256_json({"clean": "to_tensor_then_fixed_normalization_only", "mean": views["normalize_mean"],
                        "std": views["normalize_std"], "size": 32})


def two_view_transform_signature(views: dict[str, Any]) -> str:
    return sha256_json({"two_view": views})
