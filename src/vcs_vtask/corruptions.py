"""V2 development corruptions: fixed, reproducible transforms of uint8 CIFAR images (no CIFAR-10-C, no official test images).

Six families × five severities; every random draw comes from a generator seeded by (family, severity, base-image uid), so a corrupted image is a
pure function of its uid.  These are development transforms in the spirit of Hendrycks & Dietterich (2019), not the standard benchmark."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter

FAMILIES = {
    "gaussian_noise": [0.04, 0.08, 0.12, 0.18, 0.26],     # sd on [0, 1] pixels
    "gaussian_blur": [0.5, 0.75, 1.0, 1.5, 2.0],          # PIL radius (px)
    "jpeg": [80, 60, 40, 25, 10],                         # quality
    "contrast": [0.75, 0.5, 0.4, 0.3, 0.2],               # factor towards the per-image mean
    "brightness": [0.1, 0.2, 0.3, 0.4, 0.5],              # additive on [0, 1]
    "pixelate": [28, 24, 20, 16, 12],                     # down-sample side (box), back up (nearest)
}
FAMILY_ID = {k: i for i, k in enumerate(FAMILIES)}


def corrupt(images: np.ndarray, uids: np.ndarray, family: str, severity: int) -> np.ndarray:
    """images uint8 [n, 32, 32, 3] (positions aligned with uids); severity 1..5.  Returns uint8 of the same shape."""
    p = FAMILIES[family][severity - 1]
    out = np.empty_like(images)
    for i, (im, u) in enumerate(zip(images, uids)):
        x = im.astype(np.float32) / 255.0
        if family == "gaussian_noise":
            rng = np.random.default_rng([FAMILY_ID[family], severity, int(u)])
            x = np.clip(x + rng.normal(0.0, p, size=x.shape), 0, 1)
        elif family == "contrast":
            m = x.mean(axis=(0, 1), keepdims=True); x = np.clip((x - m) * p + m, 0, 1)
        elif family == "brightness":
            x = np.clip(x + p, 0, 1)
        elif family == "gaussian_blur":
            out[i] = np.asarray(Image.fromarray(im).filter(ImageFilter.GaussianBlur(radius=p))); continue
        elif family == "jpeg":
            buf = io.BytesIO(); Image.fromarray(im).save(buf, format="JPEG", quality=int(p)); buf.seek(0)
            out[i] = np.asarray(Image.open(buf).convert("RGB")); continue
        elif family == "pixelate":
            out[i] = np.asarray(Image.fromarray(im).resize((p, p), Image.BOX).resize((32, 32), Image.NEAREST)); continue
        else:
            raise ValueError(family)
        out[i] = np.round(x * 255.0).astype(np.uint8)
    return out


def cells() -> list[tuple[str, int]]:
    return [(f, s) for f in FAMILIES for s in range(1, 6)]
