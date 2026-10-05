"""P131: CIFAR-10-C-style corruption functions (regenerated locally) and the official-test gate of the generator."""
import numpy as np
import pytest

from vcs_diag import c10c


def _images(n=16):
    # smooth random colour fields (natural-image-like spectrum) so blur / pixelate / JPEG have structure to remove
    from scipy import ndimage
    rs = np.random.RandomState(0)
    f = ndimage.gaussian_filter(rs.rand(n, 32, 32, 3), sigma=(0, 2, 2, 0), mode="wrap")
    f = (f - f.min(axis=(1, 2, 3), keepdims=True)) / np.ptp(f, axis=(1, 2, 3), keepdims=True)
    return np.uint8(f * 200 + 20 + rs.randint(0, 16, size=f.shape))


ALL = c10c.PRIMARY + c10c.EXTRA


@pytest.mark.parametrize("name", ALL)
def test_shape_dtype_range(name):
    x = _images(4)
    for sev in range(1, 6):
        out = c10c.corrupt_block(x, name, sev, seed=123)
        assert out.shape == x.shape and out.dtype == np.uint8


@pytest.mark.parametrize("name", ALL)
def test_deterministic(name):
    x = _images(4)
    a = c10c.corrupt_block(x, name, 3, seed=7)
    b = c10c.corrupt_block(x, name, 3, seed=7)
    assert np.array_equal(a, b)


def _mse(name, sev, x):
    return float(((c10c.corrupt_block(x, name, sev, seed=c10c.block_seed(1, name, sev)).astype(float) - x.astype(float)) ** 2).mean())


# parameters that increase monotonically with severity in make_cifar_c.py -> MSE to the clean image increases (mean over images)
MONOTONE = ["gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise", "gaussian_blur", "defocus_blur", "zoom_blur", "fog",
            "brightness", "contrast", "pixelate", "jpeg_compression"]


@pytest.mark.parametrize("name", MONOTONE)
def test_severity_monotone(name):
    x = _images(16)
    m = [_mse(name, s, x) for s in range(1, 6)]
    assert all(m[i] <= m[i + 1] + 1e-9 for i in range(4)), m


@pytest.mark.parametrize("name", ["glass_blur", "elastic_transform", "saturate"])
def test_non_monotone_still_changes(name):
    # CIFAR parameters are not monotone for these (glass blur sigma/iterations, elastic affine vs elastic, saturate sign); only check they act
    x = _images(8)
    assert all(_mse(name, s, x) > 0 for s in range(1, 6))


def test_helpers():
    x = np.random.RandomState(0).rand(32, 32, 3)
    assert np.abs(c10c.hsv2rgb(c10c.rgb2hsv(x)) - x).max() < 1e-12
    assert np.isclose(c10c.disk(1.5, 0.1).sum(), 1.0)
    p = c10c.plasma_fractal(np.random.RandomState(0))
    assert p.shape == (32, 32) and p.min() == 0.0 and p.max() == 1.0
    # gray pixels: hue 0, saturation 0 (skimage convention)
    g = np.full((2, 2, 3), 0.5)
    hsv = c10c.rgb2hsv(g)
    assert np.all(hsv[..., 0] == 0) and np.all(hsv[..., 1] == 0) and np.allclose(hsv[..., 2], 0.5)
    # 3-point affine solve reproduces the points
    src = np.float32([[26, 26], [26, 6], [6, 6]]); dst = src + np.float32([[1, -2], [0.5, 0.5], [-1, 1]])
    M = c10c._affine_from_3pts(src, dst)
    assert np.allclose(M @ np.c_[src, np.ones(3)].T, dst.T, atol=1e-5)
    # identity affine warp is the identity
    img = np.random.RandomState(1).rand(32, 32, 3)
    assert np.allclose(c10c._warp_affine_reflect101(img, np.array([[1.0, 0, 0], [0, 1.0, 0]])), img)


def test_glass_blur_copy_semantics_replicated():
    # the original's tuple "swap" of numpy views copies the second pixel into the first and leaves the second unchanged
    a = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    a[0, 0], a[1, 1] = a[1, 1], a[0, 0]
    assert a[0, 0].tolist() == [9, 10, 11] and a[1, 1].tolist() == [9, 10, 11]


def test_block_seeds_unique():
    seeds = {c10c.block_seed(131, n, s) for n in ALL for s in range(1, 6)}
    assert len(seeds) == len(ALL) * 5


def test_generate_refuses_without_authorisation(monkeypatch):
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("p131", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "p131_c10c.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    monkeypatch.delenv("VCS_P131_C10C", raising=False)
    with pytest.raises(PermissionError):
        mod._load_test_images_for_corruption(True)
    monkeypatch.setenv("VCS_P131_C10C", "1")
    with pytest.raises(PermissionError):
        mod._load_test_images_for_corruption(False)
