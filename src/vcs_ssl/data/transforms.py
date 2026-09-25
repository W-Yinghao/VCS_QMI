"""Two-view augmentation (train) and clean transform (evaluation).  Parameters are explicit (spec §4.3, W7)."""
from __future__ import annotations

from typing import Any

from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from ..utils import sha256_json

_INTERP = {"bilinear": InterpolationMode.BILINEAR, "bicubic": InterpolationMode.BICUBIC, "nearest": InterpolationMode.NEAREST}


def build_two_view_transform(views: dict[str, Any]) -> T.Compose:
    if views["solarize_p"] != 0.0:
        raise ValueError("solarize is not implemented on purpose")
    if not 0.0 <= views["gaussian_blur_p"] <= 1.0:
        raise ValueError("gaussian_blur_p must be in [0, 1]")
    rrc = views["random_resized_crop"]
    cj = views["color_jitter"]
    ops: list = [
        T.RandomResizedCrop(rrc["size"], scale=tuple(rrc["scale"]), ratio=tuple(rrc["ratio"]),
                            interpolation=_INTERP[rrc["interpolation"]], antialias=rrc["antialias"]),
        T.RandomHorizontalFlip(p=views["horizontal_flip_p"]),
        T.RandomApply([T.ColorJitter(cj["brightness"], cj["contrast"], cj["saturation"], cj["hue"])], p=cj["p"]),
        T.RandomGrayscale(p=views["grayscale_p"]),
    ]
    if views["gaussian_blur_p"] > 0.0:
        # named variant (not in the first-round recipe): 3x3 kernel for 32x32 images, sigma U(0.1, 2.0), applied after grayscale
        ops.append(T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=views["gaussian_blur_p"]))
    ops += [T.ToTensor(), T.Normalize(mean=tuple(views["normalize_mean"]), std=tuple(views["normalize_std"]))]
    return T.Compose(ops)


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
