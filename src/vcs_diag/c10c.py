"""P131 — CIFAR-10-C-style corruptions, regenerated locally (NOT the released CIFAR-10-C files).

Re-implementation of the corruption functions and CIFAR severity parameters of Hendrycks & Dietterich (ICLR 2019), repository
``hendrycks/robustness``, ``ImageNet-C/create_c/make_cifar_c.py``, with numpy / scipy / PIL only (no cv2, skimage, Wand on this server):

* exact re-implementations of the library calls that are missing here:
    - ``skimage.color.rgb2hsv / hsv2rgb``                   -> :func:`rgb2hsv`, :func:`hsv2rgb` (same branch order and tie rules)
    - ``skimage.filters.gaussian(..., multichannel=True)``  -> ``scipy.ndimage.gaussian_filter`` per channel, mode 'nearest', truncate 4
    - ``skimage.util.random_noise(mode='s&p')``             -> :func:`_salt_pepper` (per-element flip with prob = amount, salt:pepper 1:1)
    - ``cv2.GaussianBlur(ksize=(3,3)) / cv2.filter2D``       -> explicit 3x3 Gaussian (cv2.getGaussianKernel formula) and correlation,
                                                              border BORDER_REFLECT_101 == scipy mode 'mirror'
    - ``cv2.getAffineTransform / cv2.warpAffine``            -> 3-point affine solve + ``scipy.ndimage.affine_transform`` (bilinear,
                                                              'mirror'); cv2 quantises coordinates to 1/32 px, scipy does not
* omitted (record in the manifest): ``motion_blur`` and ``snow`` need ImageMagick through Wand (``MotionImage.motion_blur``), ``frost``
  needs the frost texture images shipped with the original repository, ``spatter`` (extra set) needs cv2 morphology / distance transforms.

Every function takes one uint8 HxWx3 image (32x32) and a ``numpy.random.RandomState`` (the original used the legacy global state, so
only the distribution, not the exact pixels, of the released set is reproducible) and returns float in [0, 255]; the generator casts
with ``np.uint8(...)`` exactly as the original script (truncation, not rounding).
"""
from __future__ import annotations

from io import BytesIO
from typing import Callable

import numpy as np
from PIL import Image
from scipy import ndimage

IMSIZE = 32

# ----------------------------------------------------------------------------------------------------------------------
# helpers (exact re-implementations of the missing library calls)
# ----------------------------------------------------------------------------------------------------------------------


def rgb2hsv(rgb: np.ndarray) -> np.ndarray:
    """skimage.color.rgb2hsv for float RGB in [0, 1] (branch order r, g, b; later branches overwrite ties; h = 0 where delta = 0)."""
    arr = np.asarray(rgb, dtype=np.float64)
    out_v = arr.max(-1)
    delta = np.ptp(arr, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out_s = delta / out_v
    out_s[delta == 0.0] = 0.0
    out_h = np.zeros_like(out_v)
    with np.errstate(invalid="ignore", divide="ignore"):
        idx = arr[..., 0] == out_v
        out_h[idx] = (arr[idx, 1] - arr[idx, 2]) / delta[idx]
        idx = arr[..., 1] == out_v
        out_h[idx] = 2.0 + (arr[idx, 2] - arr[idx, 0]) / delta[idx]
        idx = arr[..., 2] == out_v
        out_h[idx] = 4.0 + (arr[idx, 0] - arr[idx, 1]) / delta[idx]
    out_h = (out_h / 6.0) % 1.0
    out_h[delta == 0.0] = 0.0
    out = np.stack([out_h, out_s, out_v], axis=-1)
    out[np.isnan(out)] = 0
    return out


def hsv2rgb(hsv: np.ndarray) -> np.ndarray:
    """skimage.color.hsv2rgb."""
    arr = np.asarray(hsv, dtype=np.float64)
    hi = np.floor(arr[..., 0] * 6)
    f = arr[..., 0] * 6 - hi
    p = arr[..., 2] * (1 - arr[..., 1])
    q = arr[..., 2] * (1 - f * arr[..., 1])
    t = arr[..., 2] * (1 - (1 - f) * arr[..., 1])
    v = arr[..., 2]
    hi = np.stack([hi, hi, hi], axis=-1).astype(np.uint8) % 6
    return np.choose(hi, np.stack([np.stack((v, t, p), axis=-1), np.stack((q, v, p), axis=-1), np.stack((p, v, t), axis=-1),
                                   np.stack((p, q, v), axis=-1), np.stack((t, p, v), axis=-1), np.stack((v, p, q), axis=-1)]))


def sk_gaussian(x: np.ndarray, sigma: float) -> np.ndarray:
    """skimage.filters.gaussian(x, sigma, multichannel=True) for float images: per-channel scipy gaussian, mode 'nearest', truncate 4."""
    x = np.asarray(x, dtype=np.float64)
    return ndimage.gaussian_filter(x, sigma=(sigma, sigma, 0), mode="nearest", truncate=4.0)


def _salt_pepper(x: np.ndarray, amount: float, rs: np.random.RandomState) -> np.ndarray:
    """skimage.util.random_noise(x, mode='s&p', amount, salt_vs_pepper=0.5) for x in [0, 1] (low clip 0)."""
    out = x.copy()
    flipped = rs.choice([True, False], size=x.shape, p=[amount, 1 - amount])
    salted = rs.choice([True, False], size=x.shape, p=[0.5, 0.5])
    out[flipped & salted] = 1.0
    out[flipped & ~salted] = 0.0
    return out


def _cv2_gaussian_kernel_1d(ksize: int, sigma: float) -> np.ndarray:
    """cv2.getGaussianKernel(ksize, sigma) for sigma > 0."""
    i = np.arange(ksize, dtype=np.float64) - (ksize - 1) / 2.0
    g = np.exp(-(i ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


def disk(radius: float, alias_blur: float = 0.1, dtype=np.float32) -> np.ndarray:
    """make_cifar_c.disk: aliased disk on [-8, 8]^2 (radius <= 8), then cv2.GaussianBlur(ksize=(3, 3), sigmaX=alias_blur) (REFLECT_101)."""
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = 3
    else:
        L = np.arange(-radius, radius + 1)
        ksize = 5
    X, Y = np.meshgrid(L, L)
    aliased = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased /= np.sum(aliased)
    g = _cv2_gaussian_kernel_1d(ksize, alias_blur)
    return ndimage.correlate(aliased.astype(np.float64), np.outer(g, g), mode="mirror").astype(dtype)


def plasma_fractal(rs: np.random.RandomState, mapsize: int = 32, wibbledecay: float = 3) -> np.ndarray:
    """make_cifar_c.plasma_fractal (diamond-square, values in [0, 1]); keeps the original's ``wibble * uniform(-wibble, wibble)``."""
    assert mapsize & (mapsize - 1) == 0
    maparray = np.empty((mapsize, mapsize), dtype=np.float64)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100.0

    def wibbledmean(array):
        return array / 4 + wibble * rs.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        ms = maparray.shape[0]
        drgrid = maparray[stepsize // 2:ms:stepsize, stepsize // 2:ms:stepsize]
        ulgrid = maparray[0:ms:stepsize, 0:ms:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        maparray[0:ms:stepsize, stepsize // 2:ms:stepsize] = wibbledmean(ldrsum + lulsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        maparray[stepsize // 2:ms:stepsize, 0:ms:stepsize] = wibbledmean(tdrsum + tulsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay
    maparray -= maparray.min()
    return maparray / maparray.max()


def clipped_zoom(img: np.ndarray, zoom_factor: float) -> np.ndarray:
    h = img.shape[0]
    ch = int(np.ceil(h / zoom_factor))
    top = (h - ch) // 2
    img = ndimage.zoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    trim_top = (img.shape[0] - h) // 2
    return img[trim_top:trim_top + h, trim_top:trim_top + h]


def _affine_from_3pts(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """cv2.getAffineTransform: 2x3 M with M @ [x, y, 1]^T = [x', y']^T for the three (x, y) point pairs."""
    A = np.concatenate([src.astype(np.float64), np.ones((3, 1))], axis=1)  # 3x3
    return np.linalg.solve(A, dst.astype(np.float64)).T                      # 2x3


def _warp_affine_reflect101(image: np.ndarray, M: np.ndarray) -> np.ndarray:
    """cv2.warpAffine(image, M, (W, H), INTER_LINEAR, BORDER_REFLECT_101): dst(x', y') = src(M^-1 (x', y')); scipy works in (row, col)."""
    A, b = M[:, :2], M[:, 2]
    Ai = np.linalg.inv(A)
    bi = -Ai @ b                                       # src_xy = Ai @ dst_xy + bi
    mat_rc = np.array([[Ai[1, 1], Ai[1, 0]], [Ai[0, 1], Ai[0, 0]]])
    off_rc = np.array([bi[1], bi[0]])
    out = np.empty_like(image)
    for c in range(image.shape[2]):
        out[..., c] = ndimage.affine_transform(image[..., c], mat_rc, offset=off_rc, order=1, mode="mirror")
    return out


# ----------------------------------------------------------------------------------------------------------------------
# corruptions (CIFAR parameters of make_cifar_c.py); x: uint8 HxWx3, returns float in [0, 255]
# ----------------------------------------------------------------------------------------------------------------------


def gaussian_noise(x, severity, rs):
    c = [0.04, 0.06, .08, .09, .10][severity - 1]
    x = np.array(x) / 255.
    return np.clip(x + rs.normal(size=x.shape, scale=c), 0, 1) * 255


def shot_noise(x, severity, rs):
    c = [500, 250, 100, 75, 50][severity - 1]
    x = np.array(x) / 255.
    return np.clip(rs.poisson(x * c) / c, 0, 1) * 255


def impulse_noise(x, severity, rs):
    c = [.01, .02, .03, .05, .07][severity - 1]
    x = _salt_pepper(np.array(x) / 255., c, rs)
    return np.clip(x, 0, 1) * 255


def speckle_noise(x, severity, rs):
    c = [.06, .1, .12, .16, .2][severity - 1]
    x = np.array(x) / 255.
    return np.clip(x + x * rs.normal(size=x.shape, scale=c), 0, 1) * 255


def gaussian_blur(x, severity, rs):
    c = [.4, .6, 0.7, .8, 1][severity - 1]
    x = sk_gaussian(np.array(x) / 255., sigma=c)
    return np.clip(x, 0, 1) * 255


def glass_blur(x, severity, rs):
    # sigma, max_delta, iterations.  The pixel "swap" keeps the original's numpy semantics (a tuple assignment of views: the first
    # pixel receives the second's value, the second keeps its own) — replicated on purpose.
    c = [(0.05, 1, 1), (0.25, 1, 1), (0.4, 1, 1), (0.25, 1, 2), (0.4, 1, 2)][severity - 1]
    x = np.uint8(sk_gaussian(np.array(x) / 255., sigma=c[0]) * 255)
    for _ in range(c[2]):
        for h in range(IMSIZE - c[1], c[1], -1):
            for w in range(IMSIZE - c[1], c[1], -1):
                dx, dy = rs.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]
    return np.clip(sk_gaussian(x / 255., sigma=c[0]), 0, 1) * 255


def defocus_blur(x, severity, rs):
    c = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (1, 0.2), (1.5, 0.1)][severity - 1]
    x = np.array(x) / 255.
    kernel = disk(radius=c[0], alias_blur=c[1]).astype(np.float64)
    channels = np.stack([ndimage.correlate(x[:, :, d], kernel, mode="mirror") for d in range(3)], axis=-1)
    return np.clip(channels, 0, 1) * 255


def zoom_blur(x, severity, rs):
    c = [np.arange(1, 1.06, 0.01), np.arange(1, 1.11, 0.01), np.arange(1, 1.16, 0.01),
         np.arange(1, 1.21, 0.01), np.arange(1, 1.26, 0.01)][severity - 1]
    x = (np.array(x) / 255.).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)
    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255


def fog(x, severity, rs):
    c = [(.2, 3), (.5, 3), (.75, 2.5), (1, 2), (1.5, 1.75)][severity - 1]
    x = np.array(x) / 255.
    max_val = x.max()
    x = x + c[0] * plasma_fractal(rs, mapsize=32, wibbledecay=c[1])[:IMSIZE, :IMSIZE][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


def brightness(x, severity, rs):
    c = [.05, .1, .15, .2, .3][severity - 1]
    x = rgb2hsv(np.array(x) / 255.)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    return np.clip(hsv2rgb(x), 0, 1) * 255


def saturate(x, severity, rs):
    c = [(0.3, 0), (0.1, 0), (1.5, 0), (2, 0.1), (2.5, 0.2)][severity - 1]
    x = rgb2hsv(np.array(x) / 255.)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    return np.clip(hsv2rgb(x), 0, 1) * 255


def contrast(x, severity, rs):
    c = [.75, .5, .4, .3, 0.15][severity - 1]
    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def elastic_transform(image, severity, rs):
    c = [(IMSIZE * 0, IMSIZE * 0, IMSIZE * 0.08), (IMSIZE * 0.05, IMSIZE * 0.2, IMSIZE * 0.07), (IMSIZE * 0.08, IMSIZE * 0.06, IMSIZE * 0.06),
         (IMSIZE * 0.1, IMSIZE * 0.04, IMSIZE * 0.05), (IMSIZE * 0.1, IMSIZE * 0.03, IMSIZE * 0.03)][severity - 1]
    image = np.array(image, dtype=np.float32) / 255.
    shape = image.shape
    shape_size = shape[:2]
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + rs.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = _affine_from_3pts(pts1, pts2)
    image = _warp_affine_reflect101(image.astype(np.float64), M).astype(np.float32)
    # skimage.filters.gaussian(u, c[1], mode='reflect', truncate=3) == scipy gaussian_filter(mode='reflect'); sigma 0 (severity 1) is identity
    dx = (ndimage.gaussian_filter(rs.uniform(-1, 1, size=shape[:2]), c[1], mode="reflect", truncate=3) * c[0]).astype(np.float32)
    dy = (ndimage.gaussian_filter(rs.uniform(-1, 1, size=shape[:2]), c[1], mode="reflect", truncate=3) * c[0]).astype(np.float32)
    dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]
    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    return np.clip(ndimage.map_coordinates(image, indices, order=1, mode="reflect").reshape(shape), 0, 1) * 255


def pixelate(x, severity, rs):
    c = [0.95, 0.9, 0.85, 0.75, 0.65][severity - 1]
    x = Image.fromarray(np.asarray(x, dtype=np.uint8))
    x = x.resize((int(IMSIZE * c), int(IMSIZE * c)), Image.Resampling.BOX)
    x = x.resize((IMSIZE, IMSIZE), Image.Resampling.BOX)
    return np.asarray(x, dtype=np.float64)


def jpeg_compression(x, severity, rs):
    c = [80, 65, 58, 50, 40][severity - 1]
    buf = BytesIO()
    Image.fromarray(np.asarray(x, dtype=np.uint8)).save(buf, "JPEG", quality=c)
    return np.asarray(Image.open(buf), dtype=np.float64)


# standard 15 of CIFAR-10-C, grouped as in the paper; implemented = 12 (motion_blur, snow, frost omitted)
FAMILIES: dict[str, list[str]] = {
    "noise": ["gaussian_noise", "shot_noise", "impulse_noise"],
    "blur": ["defocus_blur", "glass_blur", "zoom_blur"],            # motion_blur omitted
    "weather": ["fog", "brightness"],                                 # snow, frost omitted
    "digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
}
PRIMARY: list[str] = [n for fam in FAMILIES.values() for n in fam]
EXTRA: list[str] = ["speckle_noise", "gaussian_blur", "saturate"]   # spatter omitted
OMITTED: dict[str, str] = {
    "motion_blur": "needs ImageMagick through Wand (MotionImage.motion_blur); Wand / ImageMagick bindings not installed, no pip installs allowed",
    "snow": "the snow layer is motion-blurred with Wand (ImageMagick); same reason as motion_blur",
    "frost": "needs the frost texture images shipped with the original repository (frost1.png ... frost6.jpg); not available, no downloads",
    "spatter (extra set)": "needs cv2 Canny / distanceTransform / morphology; cv2 not installed",
}
CORRUPTIONS: dict[str, Callable] = {n: globals()[n] for n in PRIMARY + EXTRA}


def corrupt_block(images: np.ndarray, name: str, severity: int, seed: int) -> np.ndarray:
    """Corrupt every image of ``images`` (N x 32 x 32 x 3 uint8) at one severity with a RandomState(seed); uint8 cast as the original."""
    fn = CORRUPTIONS[name]
    rs = np.random.RandomState(seed)
    out = np.empty_like(images)
    for i in range(len(images)):
        out[i] = np.uint8(fn(images[i], severity, rs))
    return out


def block_seed(base_seed: int, name: str, severity: int) -> int:
    """Order-independent seed per (corruption, severity) block."""
    names = PRIMARY + EXTRA
    return int(base_seed) * 1000 + names.index(name) * 10 + int(severity)
