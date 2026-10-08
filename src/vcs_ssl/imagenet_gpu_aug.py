"""P139 (v7 V7-IMAGENET-PIPELINE): ImageNet multi-view input pipeline with the colour / grey / blur stage on the GPU.

The P132 pipeline (``vcs_ssl.imagenet.multiview_transform``) runs every op on PIL images in DataLoader workers; with the Gaussian blur
(kernel 23, sigma ~ U(0.1, 2.0), p 0.5) 32 workers deliver ~520 images/s (P132 report), below one L40S.  Here the work is split:

  CPU (DataLoader workers, unchanged PIL code):  JPEG decode once per image -> per view RandomResizedCrop(224, scale (0.08, 1), bilinear,
                                                 antialias) -> uint8 tensor [3, 224, 224]  (no float conversion on the CPU)
  GPU (this module, vectorised over N = V * b):  horizontal flip (p 0.5) -> ColorJitter(0.4, 0.4, 0.4, 0.1) with p 0.8, factors and the
                                                 random op order drawn per image -> RandomGrayscale(p 0.2) -> GaussianBlur(23, sigma
                                                 U(0.1, 2.0)) with p 0.5 -> ImageNet normalisation

Same probabilities, ranges, order, pixel range [0, 1], reflect padding and normalisation as P132.  Every parameter is drawn independently per
image and per view (``sample_params``) -- torchvision's v2 GaussianBlur draws ONE sigma per call, so it is never batch-called here; the blur
is a per-sample separable grouped convolution.  The GPU stage computes in float32 throughout; the PIL path rounds to uint8 after every op and
uses PIL's integer grey / contrast / HSV arithmetic, so the outputs are close but not bit-identical to P132 (quantified by
``scripts/p139_imagenet_pipeline.py check``).  The new stage is therefore a *new shared implementation* used for all three methods.
Equality of the vectorised per-sample ops with torchvision's own tensor functional ops (same parameters, one image at a time) is tested in
tests/test_p139_imagenet_pipeline.py.
"""
from __future__ import annotations

import os
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from .imagenet import IMAGENET_MEAN, IMAGENET_ROOT, class_index, read_manifest

KERNEL = 23                     # P132: GaussianBlur(kernel_size=23)
SIGMA = (0.1, 2.0)
JITTER = (0.4, 0.4, 0.4, 0.1)   # brightness, contrast, saturation, hue
P_JITTER, P_GRAY, P_BLUR, P_FLIP = 0.8, 0.2, 0.5, 0.5
GRAY_W = (0.2989, 0.587, 0.114)  # torchvision tensor rgb_to_grayscale weights


# ----------------------------------------------------------------------------------------------------------- CPU stage
class ImageNetCropViews(torch.utils.data.Dataset):
    """Decode once, RandomResizedCrop each view (PIL, exactly P132's crop), return (uint8 [V, 3, size, size], label, index)."""

    def __init__(self, split: str = "train", views: int = 2, size: int = 224, min_scale: float = 0.08, root: str = IMAGENET_ROOT,
                 rows: list[tuple[str, str]] | None = None, classes: dict[str, int] | None = None) -> None:
        from torchvision import transforms as T
        self.rows = rows if rows is not None else read_manifest(split)
        self.cls = classes if classes is not None else class_index(read_manifest("train"))
        self.root, self.views = root, int(views)
        self.crop = T.RandomResizedCrop(size, scale=(min_scale, 1.0), interpolation=T.InterpolationMode.BILINEAR, antialias=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        from PIL import Image
        from torchvision.transforms.functional import pil_to_tensor
        rel, wnid = self.rows[i]
        with Image.open(os.path.join(self.root, rel)) as im:
            im = im.convert("RGB")
            v = torch.stack([pil_to_tensor(self.crop(im)) for _ in range(self.views)])
        return v, self.cls[wnid], i


# ----------------------------------------------------------------------------------------------------------- params
def sample_params(n: int, generator: torch.Generator | None = None) -> dict[str, Tensor]:
    """Independent per-image draws for n images (CPU tensors; move with ``params_to``)."""
    g = generator
    u = lambda *s: torch.rand(*s, generator=g)  # noqa: E731
    b, c, s, h = JITTER
    return {
        "flip": u(n) < P_FLIP,
        "jitter": u(n) < P_JITTER,
        "order": torch.stack([torch.randperm(4, generator=g) for _ in range(n)]),        # ColorJitter.get_params: fn_idx = randperm(4)
        "brightness": (1 - b) + 2 * b * u(n),                                             # U[max(0, 1-b), 1+b]  (b < 1)
        "contrast": (1 - c) + 2 * c * u(n),
        "saturation": (1 - s) + 2 * s * u(n),
        "hue": -h + 2 * h * u(n),                                                         # U[-h, h]
        "gray": u(n) < P_GRAY,
        "blur": u(n) < P_BLUR,
        "sigma": SIGMA[0] + (SIGMA[1] - SIGMA[0]) * u(n),                                 # GaussianBlur.get_params: U(sigma_min, sigma_max)
    }


def params_to(p: dict[str, Tensor], device: torch.device | str) -> dict[str, Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in p.items()}


# ----------------------------------------------------------------------------------------------------------- ops (float, [N, 3, H, W])
def _gray(x: Tensor) -> Tensor:
    return (GRAY_W[0] * x[:, 0] + GRAY_W[1] * x[:, 1] + GRAY_W[2] * x[:, 2]).unsqueeze(1)


def _blend(x: Tensor, y: Tensor, f: Tensor) -> Tensor:
    f = f.view(-1, 1, 1, 1)
    return (f * x + (1.0 - f) * y).clamp(0.0, 1.0)


def adjust_brightness(x: Tensor, f: Tensor) -> Tensor:
    return _blend(x, torch.zeros_like(x), f)


def adjust_contrast(x: Tensor, f: Tensor) -> Tensor:
    mean = _gray(x).mean(dim=(-3, -2, -1), keepdim=True)
    return _blend(x, mean, f)


def adjust_saturation(x: Tensor, f: Tensor) -> Tensor:
    return _blend(x, _gray(x), f)


def adjust_hue(x: Tensor, hf: Tensor) -> Tensor:
    from torchvision.transforms._functional_tensor import _hsv2rgb, _rgb2hsv
    hsv = _rgb2hsv(x)
    h, s, v = hsv.unbind(dim=-3)
    h = torch.remainder(h + hf.view(-1, 1, 1), 1.0)
    return _hsv2rgb(torch.stack((h, s, v), dim=-3))


_OPS = (adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue)
_KEYS = ("brightness", "contrast", "saturation", "hue")


def gaussian_kernels(sigma: Tensor, k: int = KERNEL) -> Tensor:
    """[n, k] normalised 1-D kernels, torchvision's _get_gaussian_kernel1d per sample."""
    half = (k - 1) * 0.5
    x = torch.linspace(-half, half, steps=k, device=sigma.device, dtype=sigma.dtype)
    pdf = torch.exp(-0.5 * (x[None, :] / sigma[:, None]).pow(2))
    return pdf / pdf.sum(dim=1, keepdim=True)


def gaussian_blur_per_sample(x: Tensor, sigma: Tensor, k: int = KERNEL) -> Tensor:
    """Per-sample separable Gaussian blur (reflect padding k//2, as torchvision); one grouped conv per axis for the whole batch."""
    n, c, hgt, wid = x.shape
    ker = gaussian_kernels(sigma.to(x.dtype), k)                         # [n, k]
    w = ker.repeat_interleave(c, dim=0)                                  # [n*c, k]
    y = F.pad(x.reshape(1, n * c, hgt, wid), (k // 2, k // 2, k // 2, k // 2), mode="reflect")
    y = F.conv2d(y, w.view(n * c, 1, 1, k), groups=n * c)                # horizontal
    y = F.conv2d(y, w.view(n * c, 1, k, 1), groups=n * c)                # vertical
    return y.view(n, c, hgt, wid)


def apply_views(x_u8: Tensor, p: dict[str, Tensor], normalize: bool = True) -> Tensor:
    """uint8 [N, 3, H, W] (already cropped) -> float [N, 3, H, W]; flip -> jitter (p, per-image op order) -> grey -> blur -> normalise."""
    x = x_u8.float().div_(255.0)
    fl = p["flip"]
    if fl.any():
        x[fl] = x[fl].flip(-1)
    jit = p["jitter"]
    if jit.any():
        idx = jit.nonzero(as_tuple=True)[0]
        xj = x[idx]
        order = p["order"][idx]
        for step in range(4):
            for op in range(4):
                sel = (order[:, step] == op).nonzero(as_tuple=True)[0]
                if sel.numel():
                    xj[sel] = _OPS[op](xj[sel], p[_KEYS[op]][idx][sel])
        x[idx] = xj
    gr = p["gray"]
    if gr.any():
        x[gr] = _gray(x[gr]).expand(-1, 3, -1, -1)
    bl = p["blur"]
    if bl.any():
        x[bl] = gaussian_blur_per_sample(x[bl], p["sigma"][bl])
    if normalize:
        mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x - mean) / std
    return x


class GpuViewAug:
    """Draw per-image parameters with a dedicated CPU generator (seed per rank and step), apply on the device of the batch."""

    def __init__(self, seed: int = 0) -> None:
        self.g = torch.Generator().manual_seed(int(seed))

    def __call__(self, x_u8: Tensor, return_params: bool = False) -> Any:
        p = sample_params(x_u8.shape[0], self.g)
        out = apply_views(x_u8, params_to(p, x_u8.device))
        return (out, p) if return_params else out


# ----------------------------------------------------------------------------------------------------------- PIL reference
def pil_reference_view(img_u8: Tensor, p: dict[str, Tensor], i: int, normalize: bool = False) -> Tensor:
    """P132's torchvision ops on a PIL image with image i's parameters (same order), returned as float [3, H, W] in [0, 1]."""
    from torchvision.transforms import functional as TF
    im = TF.to_pil_image(img_u8)
    if bool(p["flip"][i]):
        im = TF.hflip(im)
    if bool(p["jitter"][i]):
        for op in p["order"][i].tolist():
            f = float(p[_KEYS[op]][i])
            im = (TF.adjust_brightness, TF.adjust_contrast, TF.adjust_saturation, TF.adjust_hue)[op](im, f)
    if bool(p["gray"][i]):
        im = TF.rgb_to_grayscale(im, num_output_channels=3)
    if bool(p["blur"][i]):
        s = float(p["sigma"][i]); im = TF.gaussian_blur(im, [KERNEL, KERNEL], [s, s])
    t = TF.to_tensor(im)
    if normalize:
        t = TF.normalize(t, IMAGENET_MEAN, (0.229, 0.224, 0.225))
    return t


def tensor_reference_view(img_u8: Tensor, p: dict[str, Tensor], i: int) -> Tensor:
    """torchvision's tensor functional ops, one image at a time, float [0, 1], same order (exactness reference for apply_views)."""
    from torchvision.transforms import functional as TF
    x = img_u8.float() / 255.0
    if bool(p["flip"][i]):
        x = TF.hflip(x)
    if bool(p["jitter"][i]):
        for op in p["order"][i].tolist():
            f = float(p[_KEYS[op]][i])
            x = (TF.adjust_brightness, TF.adjust_contrast, TF.adjust_saturation, TF.adjust_hue)[op](x, f)
    if bool(p["gray"][i]):
        x = TF.rgb_to_grayscale(x, num_output_channels=3)
    if bool(p["blur"][i]):
        s = float(p["sigma"][i]); x = TF.gaussian_blur(x, [KERNEL, KERNEL], [s, s])
    return x


# ----------------------------------------------------------------------------------------------------------- optional GPU decode + crop
# Variant for multi-GPU nodes where CPU decoding cannot keep up (P139 loader: ~60 images/s per worker at 2 views): workers only read the
# JPEG bytes; decode = nvjpeg (torchvision.io.decode_jpeg on CUDA, batched), RandomResizedCrop = torchvision's get_params algorithm per
# image and view (dedicated generator) + bilinear antialias resize per crop (torch), rounded to uint8 like the PIL path.  Files that the
# GPU decoder rejects (e.g. CMYK JPEGs) fall back to PIL decode on the CPU.  nvjpeg vs libjpeg and torch vs PIL resampling differ slightly,
# so this is again a NEW SHARED IMPLEMENTATION (difference quantified by the check job); it is a measured option, not the default.
class ImageNetJpegBytes(torch.utils.data.Dataset):
    def __init__(self, split: str = "train", root: str = IMAGENET_ROOT, rows: list[tuple[str, str]] | None = None,
                 classes: dict[str, int] | None = None) -> None:
        self.rows = rows if rows is not None else read_manifest(split)
        self.cls = classes if classes is not None else class_index(read_manifest("train"))
        self.root = root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        rel, wnid = self.rows[i]
        with open(os.path.join(self.root, rel), "rb") as fh:          # plain file I/O in workers (no torchvision op after fork)
            buf = bytearray(fh.read())
        return torch.frombuffer(buf, dtype=torch.uint8), self.cls[wnid], i


def collate_bytes(batch):
    data, lab, idx = zip(*batch)
    return list(data), torch.tensor(lab), torch.tensor(idx)


def rrc_params(h: int, w: int, g: torch.Generator, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3)) -> tuple[int, int, int, int]:
    """torchvision RandomResizedCrop.get_params, with an explicit generator."""
    import math
    area = h * w
    log_r = torch.log(torch.tensor(ratio))
    for _ in range(10):
        target = area * float(torch.empty(1).uniform_(scale[0], scale[1], generator=g))
        ar = math.exp(float(torch.empty(1).uniform_(float(log_r[0]), float(log_r[1]), generator=g)))
        cw, ch = int(round(math.sqrt(target * ar))), int(round(math.sqrt(target / ar)))
        if 0 < cw <= w and 0 < ch <= h:
            i = int(torch.randint(0, h - ch + 1, (1,), generator=g)); j = int(torch.randint(0, w - cw + 1, (1,), generator=g))
            return i, j, ch, cw
    in_r = w / h
    if in_r < min(ratio):
        cw = w; ch = int(round(cw / min(ratio)))
    elif in_r > max(ratio):
        ch = h; cw = int(round(ch * max(ratio)))
    else:
        cw, ch = w, h
    return (h - ch) // 2, (w - cw) // 2, ch, cw


def gpu_decode_crop(data: list[Tensor], views: int, device: torch.device, g: torch.Generator, size: int = 224) -> Tensor:
    """List of JPEG byte tensors -> uint8 [b, views, 3, size, size] on `device`."""
    from torchvision.io import ImageReadMode, decode_jpeg
    try:
        imgs = decode_jpeg(data, mode=ImageReadMode.RGB, device=device)
    except RuntimeError:
        imgs = []
        for d in data:
            try:
                imgs.append(decode_jpeg(d, mode=ImageReadMode.RGB, device=device))
            except RuntimeError:
                import io
                from PIL import Image
                from torchvision.transforms.functional import pil_to_tensor
                with Image.open(io.BytesIO(d.numpy().tobytes())) as im:
                    imgs.append(pil_to_tensor(im.convert("RGB")).to(device))
    out = torch.empty(len(imgs), views, 3, size, size, dtype=torch.uint8, device=device)
    for k, im in enumerate(imgs):
        h, w = im.shape[-2:]
        for v in range(views):
            i, j, ch, cw = rrc_params(h, w, g)
            crop = im[:, i:i + ch, j:j + cw].unsqueeze(0).float()
            out[k, v] = F.interpolate(crop, size=(size, size), mode="bilinear", antialias=True, align_corners=False)[0].round_().clamp_(0, 255).to(torch.uint8)
    return out
